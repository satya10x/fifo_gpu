# fifo_gpu — How it works and why it's built this way

A GPU/CPU engine that computes **realized PnL with FIFO lot-matching**, bucketed
by holding period, over an equities tradebook — any client, any symbol, any date
range — backed by a versioned, transparent columnar store.

This document answers five questions:

1. [Why we left the O(n²) Python approach](#1-from-on²-python-to-a-parallel-scan) — with a worked FIFO example
2. [Why Lance over Parquet](#2-why-lance-over-parquet)
3. [Why a CPU↔GPU router](#3-why-a-cpu↔gpu-router)
4. [What numbers we get](#4-what-numbers-we-get)
5. [What bucketing we do, and why](#5-bucketing--what-and-why)

For the decision *timeline* see `JOURNEY.md`; for raw benchmark dumps see
`BENCHMARKS.md`; for the system layout see `ARCHITECTURE.md`.

---

## 1. From O(n²) Python to a parallel scan

### What "realized PnL" actually requires
For each `(client, symbol)` you hold a running queue of **open buy lots**. Every
sell must be matched against those lots in **first-in-first-out** order: the
shares you bought earliest are the ones you're deemed to sell first. Each matched
slice produces a realized gain `qty × (sell_price − buy_price)` **and** a holding
period `sell_day − buy_day` that decides its tax bucket.

### Why the old way was O(n²)
The natural Python implementation is a nested loop: for each sell, walk the list
of prior buys, peeling shares off the front until the sell is filled.

```python
def realized_pnl(trades):           # trades sorted by time
    lots, pnl = [], 0.0             # lots = list of [qty, price] open buys
    for t in trades:
        if t.qty > 0:               # a buy
            lots.append([t.qty, t.price])
        else:                       # a sell — match against the front of the queue
            need = -t.qty
            while need > 0:
                lot = lots[0]
                take = min(need, lot[0])
                pnl += take * (t.price - lot[1])
                lot[0] -= take
                need  -= take
                if lot[0] == 0:
                    lots.pop(0)     # <-- list.pop(0) is O(len(lots))
    return pnl
```

Two things make this **O(n²)** in practice:
- `lots.pop(0)` (or rebuilding the list / `pandas` `groupby().apply()` with a
  Python-level inner loop) is **linear per sell** → `n` sells × `n` lots.
- It's inherently **sequential**: lot *k* can't be matched until lot *k−1* is
  resolved, so it never vectorizes and never touches a GPU.

On our skew-realistic data this is fatal: the largest single `(client,symbol)`
partition is **2.23 M trades**, and the top whale has **21.6 M**. An O(n²) inner
loop on a 21.6 M-trade book is ~4.6 × 10¹⁴ operations — effectively never
finishes. The whole reason whales exist (30 accounts = 99.8 % of volume) is the
worst case for a quadratic algorithm.

### The reframe: FIFO matching is interval-matching on cumulative quantity
The key insight: **FIFO doesn't need a queue.** If you label every *share* (not
trade) in order, FIFO simply says *"the k-th share sold is the k-th share
bought."* So:

1. Split the partition into buys and sells (positional order is preserved at
   ingest, so this is free).
2. Take the **prefix sum of bought quantity** and the **prefix sum of sold
   quantity** — two cumulative arrays.
3. The shares a given sell covers form a contiguous **interval** on the shared
   share-axis. **Binary-search (`searchsorted`)** that interval into the
   cumulative-buy array to find exactly which buy lots it overlaps and by how
   much.
4. **Segment-reduce** each matched slice's `qty × (sell−buy)` and holding bucket.

That's `O(n log n)`, **branch-free, and fully data-parallel** — every sell is
independent given the prefix sums, which is exactly what a GPU wants. This is the
within-partition kernel (`src/gpu.rs`): scan → searchsorted → segmented-reduce,
one GPU block per large partition.

### Worked example
Trades for one `(client, symbol)`:

| # | day | side | qty | price |
|---|-----|------|-----|-------|
| 1 | 1   | buy  | 100 | ₹200  |
| 2 | 2   | buy  | 100 | ₹210  |
| 3 | 5   | sell | 150 | ₹220  |

**Naive queue walk:** sell of 150 pops lot #1 (100 @ ₹200) fully, then 50 from
lot #2 (@ ₹210).

**Our way:**
- cumulative bought = `[100, 200]` (lot1 covers shares 0–100, lot2 covers 100–200)
- cumulative sold = `[150]` (this sell covers shares 0–150)
- `searchsorted(cumBuy, [0,150))` → overlaps lot1 for shares `[0,100)` and lot2
  for shares `[100,150)`.

Both give the same answer:
```
realized = 100 × (220 − 200)  +  50 × (220 − 210)
         =      2000          +       500         = ₹2 500
```
Holding periods: 100 shares held 4 days, 50 shares held 3 days → both **> same
day**, so the whole ₹2 500 lands in the **short-term** bucket. (Note a single sell
can split across *multiple* buckets if its matched lots have different holding
ages — see §5.)

The two methods are **bit-identical** (validated: matched-qty exact, realized to
1e-6 vs the CPU oracle), but the second is a parallel scan instead of a serial
queue — and that's the whole engine.

> The CPU path uses the straightforward lot queue (it's fast enough for small
> queries and supports any matching policy). The GPU path uses the
> scan/searchsorted form. They agree exactly; the router (§3) picks between them.

---

## 2. Why Lance over Parquet

We use **two tiers of storage**:
- the **tradebook** = the system of record (wide, all columns, audit trail);
- the **compute table** = a lean cache holding just what the PnL scan needs
  (`signed_qty`, `price_ticks`, `day` — a **12-byte** `PackedTrade` per row).

For the compute table we use a **transparent packed format** (the bytes on disk
*are* the `[PackedTrade]` array). The **versioned store** around it is **Lance**,
not Parquet. Why:

| Dimension | Parquet (status quo) | Lance |
|---|---|---|
| **Read for compute** | full re-scan; opaque row-group decode (LZ4/zstd) | records sit in a transparent `FixedSizeBinary(12)` column — the bytes *are* the records, so they **DMA straight to the GPU, no decompress** |
| **Random access** | scan whole matched row-groups | byte-addressable; a page-range index prunes to the date range |
| **Versioning / time-travel** | none (rewrite files) | built-in ACID snapshots → reproducible training sets, honest backtests |
| **Corrections / lineage** | rewrite the file | regenerate one partition, bump the version |
| **Self-describing** | yes (columns) | yes (per-row `client/symbol`) + a compact partition sidecar for fast loads |

**The deciding argument:** raw speed alone barely justifies a storage rebuild —
but **reproducibility, correction-lineage, and time-travel do**, and those are
Lance's strengths. The transparent column is what makes the GPU-direct read
possible at all.

**Honest caveat on compression.** Parquet compresses by *default* and is the
better pure compressor. Lance's default format stores **uncompressed** (we
measured the redundant `client/symbol` columns at full width — 3.8 GB/version).
We don't actually *want* compression on the hot `rec` column — it has to stay raw
for the GPU — so we only compress the redundant side columns, which required
opting into Lance's **V2.1 structural format** + a `zstd` hint (the default format
silently ignores it). That recovered **3.8 GB → 2.5 GB**. So: we use Lance for
*versioning*, not for *compression*; Parquet would win a bytes-on-disk contest but
loses the GPU-direct read and the version history.

---

## 3. Why a CPU↔GPU router

**The GPU is not always faster.** It has a fixed cost — host→device (H2D)
transfer over PCIe plus kernel-launch overhead — on the order of tens of
milliseconds before it does any useful work. For a small query (one client, one
day, a retail account with 5 trades) the CPU answers in **microseconds to low
milliseconds** and the GPU's fixed cost alone would lose. For a large recompute
(a whale's full history, or cross-client all-history at 168 M rows) the GPU's
parallelism dwarfs that fixed cost and it wins 2–2.5×.

So we route per query with a **small additive cost model** (`src/router.rs`),
fit empirically from the benchmark — not one magic threshold:

```text
cpu_ns = rows·cpu_per_row + checkpoints·ckpt_load
gpu_ns = gpu_fixed
       + rows·(h2d_per_row + gpu_per_row)      # transfer + compute, both linear
       + fanout·launch_overhead               # skew term ①
       + max_part·gpu_serial_per_row           # skew term ②
       + checkpoints·ckpt_load
route  = argmin(cpu_ns, gpu_ns)
```

The interesting part is the **two skew terms**, because skew bites the GPU two
opposite ways:
- **`fanout`** — many *tiny* partitions ⇒ launch/coordination overhead → favours
  CPU.
- **`max_part`** — the *largest* partition's residual within-block
  serialization (the whale tail). With the within-partition kernel this
  coefficient is tiny (≈ `gpu_per_row / block_size`), but it's the term that
  stops the router from shipping a single 2.23 M-row partition to one serial GPU
  thread.

`checkpoints` can dominate cross-client *narrow* ranges — another reason those go
to the pre-computed rollup, not to live compute.

**It's self-calibrating.** The benchmark measures each query's CPU time and —
when the GPU arm runs it — the **H2D and kernel times separately**, then fits the
coefficients by least-squares through the origin. So `h2d_per_row` and
`gpu_per_row` come from the *measured wire and kernel*, not a guess. Predictions
are logged predicted-vs-actual so the model can be recalibrated as the skew
distribution drifts.

---

## 4. What numbers we get

**Hardware:** Tesla T4 (16 GB, sm_75, **PCIe 3.0**), Ubuntu 24.04, CUDA 13.
**Dataset:** 168 M trades, 40 k clients, 1 k symbols, **30 whales = 99.8 % of
volume**, top whale 21.6 M trades, largest partition 2.23 M rows — power-law skew
(the realistic hard case). **Compute record: 12 bytes.** Every query's PnL is
asserted equal to the Parquet baseline; the GPU is validated bit-exact vs the CPU
oracle.

### Per-query: where each engine wins (end-to-end, disk→answer)

| query | rows touched | CPU | GPU | Parquet baseline | route / win |
|---|---|---|---|---|---|
| whale 1-day | 303 | 12 ms | — | 11 616 ms | **CPU** · 940× vs baseline |
| whale 1-month | 363 | 11 ms | — | 11 689 ms | **CPU** · 1 094× |
| whale random-range | 1 033 | 0.07 ms | — | 11 613 ms | **CPU** · 168 952× |
| retail all-history | 5 | ~0 ms | 40 ms | 11 227 ms | **CPU** (GPU fixed cost) |
| whale all-history | 21.6 M | 373 ms | **216 ms** | 11 729 ms | **GPU 1.7×** |
| cross-client all-history | 168 M | 2 885 ms | **1 153 ms** | 14 162 ms | **GPU 2.5×** |

**Throughput (cross-client 168 M fold):** GPU ≈ **145 M trades/s**, CPU ≈ 68 M/s,
Parquet baseline ≈ 13 M/s — **~12× over the status quo**, GPU ~2–2.5× over the
calibrated CPU on large recomputes.

### The 12-byte record flipped the bottleneck
Dropping the timestamp (32 B → 12 B record) halved the transfer and **inverted**
the binding constraint from transfer-bound to compute-bound:

| metric | 32-byte record | 12-byte record |
|---|---|---|
| H2D transfer | 1 313 ms | **640 ms** |
| kernel | 667 ms | 688 ms |
| binding constraint | **transfer** | **compute** (kernel > H2D) |
| GPU vs CPU full fold | 1.16× | **1.8×** |

So the next lever is kernel throughput, not fewer bytes.

### Storage footprint (168 M, `du -sh`)

| store | on disk | what |
|---|---|---|
| Parquet tradebook | 1.6 GB | wide source, compressed |
| Packed compute table (`.fifopack`) | 1.9 GB | transparent, **uncompressed** (GPU-direct, by design) |
| Lance, one version (V2.1 + zstd cols) | **2.5 GB** | `rec` ~2 GB + ~0.5 GB compressed side columns |
| Partition sidecar (`.parts.json`) | 1.3 MB | 74 k `(client,symbol,offset)` triples |

### Scale-up (extrapolated from the measured rates)
| rows | GPU (streamed) | CPU | Parquet baseline |
|---|---|---|---|
| 168 M (**measured**) | 1.3 s | 2.8 s | 14 s |
| 1 billion | ~5–7 s | ~17 s | ~80 s |
| 7.5 B (a full year) | ~40–55 s | ~2 min | ~10 min |

A billion-row from-scratch recompute is **~5–7 s on one mid-range GPU**; the
streaming fold bounds VRAM (~600 MB working set) so it fits the 16 GB card.
*Selective* queries are sub-millisecond regardless — you almost never fold a
billion.

---

## 5. Bucketing — what and why

### Why bucket at all
Realized PnL isn't one number — for **tax** it must be split by **holding
period**, because different holding ages are taxed differently. For Indian
equities the natural split is:

- **Intraday** — bought and sold the *same day* (speculative / business income);
- **Short-term** — held **≤ 365 days** (short-term capital gains);
- **Long-term** — held **> 365 days** (long-term capital gains).

Each matched lot-slice is classified by **its own** holding span, so a single
sell can contribute to *several* buckets at once (e.g. it consumes one lot bought
yesterday → short-term, and another bought two years ago → long-term).

### How it's expressed — generic K-way rules
Bucketing is a small rules object (`BucketRules` in `src/fifo.rs`), not hard-coded
tax law, so the same engine serves different regimes/instruments without code
changes:

- `intraday_same_day: bool` — put same-day round-trips in their own bucket 0;
- `boundaries: [i32; 8]` — ascending **inclusive** holding-day cut points;
- up to **`MAX_BUCKETS = 8`** buckets total.

The classifier is just a span lookup:
```rust
fn classify(rules, buy_day, sell_day) -> bucket_index {
    let span = sell_day - buy_day;
    if rules.intraday_same_day && span == 0 { return 0; }
    for i in 0..rules.n_bounds {
        if span <= rules.boundaries[i] { return base + i; }
    }
    base + rules.n_bounds   // beyond the last boundary → the final bucket
}
```

Presets and overrides:
- `BucketRules::equity(365)` → the default 3 buckets (intraday / ≤365 d / >365 d);
- `fifo query --bands 30,365` → arbitrary K-way bands (here: intraday / 1–30 d /
  31–365 d / >365 d);
- `--ltcg-days N` to move the long-term boundary; `--intraday-same-day` to toggle
  the same-day bucket.

### Why classify inside the kernel
The GPU does the bucketing **in the same pass** (the A.3 "K-way bucketing"
kernel): as each matched slice is reduced, its holding span picks one of K
accumulators, and the segmented-reduce writes into `n_partitions × K` outputs. No
second pass over the data — the buckets fall out of the same scan that computes
the PnL, and the result is **bit-identical to the CPU classifier** (validated with
`--bands 30,365`, GPU vs CPU, bucket-for-bucket).

### Beyond FIFO
The matching order is a second axis (`MatchPolicy`): **FIFO** (default, and the
GPU fast path), **LIFO**, and **HIFO** (highest-cost-first, for tax-loss
harvesting). LIFO/HIFO run on the CPU — the within-partition GPU kernel is
FIFO-specific (its parallelism comes from the FIFO share-ordering), so the clean
split is *GPU = FIFO fast path, CPU = any policy*.

---

*See `JOURNEY.md` for the chronological decision log, `BENCHMARKS.md` for raw
results, `DESIGN.md` for the generic-engine direction, and `ARCHITECTURE.md` for
the module layout.*
