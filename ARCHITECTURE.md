# fifo_gpu — Architecture & infrastructure

A GPU/CPU engine for **ordered, stateful, per-partition** computations over an
equities tradebook — today FIFO/LIFO/HIFO realized PnL bucketed into
jurisdiction-configurable holding-period bands, on the fly, for any date range,
per-client or cross-client. Compute runs on **CPU or GPU**, chosen per query by a
**cost-based router**. Storage is a transparent, fixed-width, struct-packed
columnar buffer laid out so trade data DMAs into GPU memory with **no decode step**.

> Companion docs: `BENCHMARKS.md` (measured results), `DESIGN.md` (the generic
> ordered-scan engine vision), `README.md` (quickstart).

---

## 1. The layered picture

```
┌───────────────────────────────────────────────────────────────────────────┐
│ CLI (`fifo`)   gen · stats · pack · checkpoint · query · rollup · correct · bench │
├───────────────────────────────────────────────────────────────────────────┤
│ ROUTER (router.rs)  cost model, calibrated from measured per-query H2D/kernel │
│        route = argmin(cpu_cost, gpu_cost);  logs predicted-vs-actual          │
├──────────────────────────────┬────────────────────────────────────────────┤
│ CPU engine (fifo.rs)         │ GPU engine (gpu.rs, --features gpu)          │
│  sequential per-partition    │  within-partition scan+searchsorted+         │
│  fold; ANY MatchPolicy        │  segmented-reduce (FIFO); streamed + pinned  │
│  (FIFO/LIFO/HIFO); ANY        │  + bounded VRAM; one-thread-per-partition    │
│  BucketRules (K bands)        │  for small partitions                        │
├───────────────────────────────────────────────────────────────────────────┤
│ STORAGE  packed compute table (packed.rs, 12 B transparent record)          │
│          + page-range index (index.rs) + checkpoints (checkpoint.rs)        │
│          + cross-client rollup (rollup.rs)   ── all over mmap, GPU-direct    │
├───────────────────────────────────────────────────────────────────────────┤
│ SOURCE   skew-realistic synthetic tradebook (generate.rs, Parquet)          │
│          [a real Lance-backed tradebook is the future system-of-record]     │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Storage layer

**Packed compute table** (`packed.rs`) — the GPU-fed substrate, realizing the
"Lance-shaped" design principles directly (we did **not** use the Lance library;
see `BENCHMARKS.md` §1 and DESIGN notes):
- **12-byte `#[repr(C)]` record** `{ i32 signed_qty, i32 price_ticks, i32 day }`.
  `ts` is *not* stored — ordering is **positional** (records written
  `(client,symbol,ts)`-sorted at build time). Dropping `ts` was the change that
  flipped the GPU regime transfer-bound → compute-bound.
- **Transparent / fixed-width** — every value is extractable from
  location+length, so the buffer **mmaps and `cudaMemcpy`s to the GPU verbatim**,
  no decompress. No opaque block compression on the hot path.
- **Clustered** by `(client,symbol,ts)`; partitions are contiguous → a
  `(client,symbol)` lookup is a binary search, not a scan.
- File layout: `[Header][part_client][part_symbol][part_offset][records]`,
  records 4 KiB-aligned (mmap page ↔ GPU transfer unit).

**Page-range index** (`index.rs`) — `(client,symbol,time) → page range`, so a
"one day" query on multi-year data prunes to a few pages instead of a full scan.

**Checkpoints** (`checkpoint.rs`) — versioned open-lot snapshots:
`(client,symbol,cutoff) → residual FIFO lot queue`. A `[lo,hi]` range query loads
the nearest-prior checkpoint and replays only `(cutoff,hi]` → work bounded by the
range, not history depth. `build_periodic` self-clears stale checkpoints (a
data-dependent correctness bug we hit + fixed).

**Cross-client rollup** (`rollup.rs`, "Option C") — incrementally-maintained
per-period (month) × per-bucket PnL sums, populated by the *same* fold that
produces per-client results (never compute PnL twice). A cross-client aggregate
becomes a KB lookup, not a billion-row fold.

---

## 3. Compute layer

**CPU fold** (`fifo.rs`) — the correctness oracle and a benchmark arm. Per
`(client,symbol)` partition, drain sells against open lots; emit matched
fragments; accumulate per-bucket PnL in `i128` (overflow-proof). It is the
**generic core** and parameterized on both engine axes:
- **`MatchPolicy`** — `Fifo` (queue front) / `Lifo` (back) / `Hifo` (max-cost).
- **`BucketRules`** — intraday flag + arbitrary ascending holding-day bands →
  up to `MAX_BUCKETS` (8) buckets; `classify` returns a bucket index.

**GPU engine** (`gpu.rs`, feature-gated `gpu`, validated on a Tesla T4) — FIFO
fast path:
- **Within-partition kernel** (`fifo_kernel_big`) for large partitions:
  cooperative **scan** (split buys/sells, cumulative-qty axes) →
  **searchsorted** interval matching (FIFO ⇒ n-th-sold ↔ n-th-bought) →
  **segmented-reduce** into 3 buckets. One block per whale partition — fixes the
  whale-tail that a naïve one-thread-per-partition kernel suffered.
- **One-thread-per-partition kernel** (`fifo_kernel`) for the small-partition tail.
- **Streamed fold** (`fold_total_streamed`) — chunked across 2 CUDA streams with
  the host mmap **page-locked** (`cuMemHostRegister`, READ_ONLY), overlapping H2D
  with kernel and **bounding VRAM to ~2 chunks** (removes the OOM ceiling).
- Validated **bit-exact** vs the CPU oracle (matched_qty exact, realized 1e-6).
- GPU is **FIFO + default 3 buckets**; LIFO/HIFO and K-bucket rulesets run on CPU.

**Router** (`router.rs`) — additive cost model
`route = argmin(cpu_cost, gpu_cost)` with coefficients **fit from measured
per-query H2D/kernel times** (not priors): `cpu_per_row`, `gpu_per_row` (kernel),
`h2d_per_row` (transfer), `launch_per_partition`, plus a `max_partition` skew term
(the whale-tail signal). Logs predicted-vs-actual for recalibration.

---

## 4. The generic engine (DESIGN.md) — three axes

| Axis | What varies | Status | CLI |
|---|---|---|---|
| **1 BucketRules** | classification into K holding-period bands | **A.1, A.2 done** (CPU; GPU=default 3) | `--ltcg-days N`, `--bands 30,365` |
| **2 MatchPolicy** | which open lot a sell consumes | **B.1 done** (CPU; GPU=FIFO) | `--policy fifo\|lifo\|hifo` |
| **3 Accumulator** | what the scan computes (PnL → VWAP, position, drawdown…) | future (C) | — |

The storage + execution layers are shared across all three; only the
per-partition logic varies. FIFO-PnL is the first instance.

---

## 5. Support components
- **Generator** (`generate.rs`, `skew.rs`, `symbols.rs`) — skew-realistic
  synthetic tradebook: power-law trade counts with explicit whale accounts (the
  hard case), per-client deterministic (regenerable bit-identically — the
  primitive the correction path needs).
- **Baseline** (`baseline.rs`) — status-quo Parquet full-rescan; the arm both
  engines must beat.
- **Correction path** (`correction.rs`) — back-dated/busted trade → regenerate
  just the affected partition(s) deterministically + the checkpoints to rebuild.
- **Benchmark harness** (`bench.rs`) — three arms (GPU / CPU / Parquet baseline)
  over the same store, end-to-end timed, per-query PnL asserted == baseline,
  three-way diagnostic for range mismatches, feeds the router fit.
- **Manifest / config / calendar / stats / writer / util** — plumbing.

---

## 6. What the benchmarks established (T4, 168 M trades, whale-heavy)
- **Packed store vs Parquet**: ~**5×** on equal work (no decode), up to **10⁷×**
  on selective queries (index + checkpoints avoid the rescan).
- **GPU vs CPU crossover** is empirical: CPU wins small/selective queries (no
  transfer tax); **GPU wins large recomputes** (~1.8–2.1×).
- After the 12 B record, the GPU full fold is **compute-bound** (kernel > H2D) —
  transfer is no longer the ceiling; **kernel throughput** is the next lever.
- **Hybrid CPU/GPU router** is the validated architecture.

---

## 7. Infrastructure / how to run
- **Repo**: `git@github.com:satya10x/fifo_gpu.git` (crate `fifo_gpu`, binary `fifo`).
- **Build**: `cargo build --release` (CPU, runs anywhere) · `cargo build
  --release --features gpu` (on an NVIDIA box w/ CUDA toolkit). `cargo test` (27).
- **GPU box**: rented Tesla T4 (16 GB, sm_75, PCIe 3.0), Ubuntu 24.04, CUDA 13.
  `scripts/bootstrap-gpu.sh` chains gen→pack→checkpoint→bench.
- **Pipeline**: `gen` → `pack` (+page index) → `checkpoint` → `query`/`bench`/`rollup`.

## 8. Not built / deferred (honest list)
- **Lance backend — Stage 1 DONE** (`lance_store.rs`, `--features lance`,
  `fifo lance`): the compute table writes to a **versioned Lance dataset** and
  reads back into an identical packed buffer (round-trip test passes). Gains
  versioning / time-travel / correction-lineage now. **Stage 2 (zero-copy via a
  custom transparent Lance encoding — decoder hands the GPU buffer back with no
  repack)** is the remaining piece.
- **A.3** — GPU K-way bucketing (custom bands currently CPU-only).
- **B.2 / B.1-heap** — GPU LIFO/HIFO and a HIFO max-heap; deferred (the
  within-partition kernel is FIFO-specific, so non-FIFO has no GPU parallelism →
  GPU=FIFO / CPU=any-policy is the clean split).
- **Axis C** — the accumulator library (VWAP, running position, drawdown…).
- **GPU residency / faster transport** — keep hot data on the GPU across queries;
  PCIe-4/5, GPUDirect Storage, multi-GPU for the 7.5 B trades/year scale.
- **Streaming packer** — `pack` holds records in RAM; the on-disk format already
  supports a two-file streaming assemble for billions of rows.
