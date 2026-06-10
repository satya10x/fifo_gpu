# fifo_gpu — The build journey, Lance vs Parquet, and the compute benchmarks

A single narrative of *what we built, in what order, why each decision was made,
and what it bought us* — followed by the Lance-vs-Parquet case and the measured
compute advantage. Companion to `ARCHITECTURE.md` (the static map), `DESIGN.md`
(the generic-engine vision), and `BENCHMARKS.md` (raw results).

---

# Part I — Decisions, in the order we made them

The guiding rule throughout (from the handoff): **benchmark, don't assume** —
the GPU-vs-CPU crossover is an empirical number, and storage layout is the main
lever. Several decisions below were *reversed by data*, which is the point.

### 1. Get the GPU arm to compile (cudarc)
**Decision:** add cudarc's `std` feature (so `DriverError`/`CompileError`
implement `Error`), target `compute_75` in NVRTC (so `atomicAdd(double)` exists),
and put `nvcc` on `PATH`.
**Why:** the GPU code had never been compiled against real CUDA.
**Result:** the kernel built on the Tesla T4 — the precondition for everything.

### 2. First benchmark — the naïve GPU *lost* to the CPU
**Decision (forced by data):** treat the whale skew, not raw volume, as the
problem.
**Why:** the one-thread-per-partition kernel folded each ~300 K–2 M-row whale
`(client,symbol)` partition on a *single* GPU thread → **kernel 903 ms**, slower
than the CPU's 679 ms full fold. Volume wasn't the issue; concentration was.
**Result:** gated the design — a GPU that loses on the realistic case isn't worth
shipping. This set the next decision.

### 3. The within-partition kernel — the pivotal GPU fix
**Decision:** implement the handoff's scan + searchsorted + segmented-reduce
formulation: **one block per big partition**, cooperatively scanning buys/sells
into cumulative-quantity axes, matching by interval overlap (FIFO ⇒ n-th-sold ↔
n-th-bought), reducing into buckets. Small partitions keep the one-thread kernel.
**Why:** FIFO is a *recurrence* that fights GPUs if done per-partition-serially;
the scan reformulation parallelizes *within* a whale.
**Result:** kernel **903 → 208 ms**; the GPU now *beat* the CPU on all-history,
**bit-exact** vs the oracle (matched_qty exact, realized 1e-6). The single most
important decision.

### 4. Router skew term
**Decision:** add `max_partition_rows` (concentration) to the cost model, not
just total rows touched.
**Why:** "10 M rows in one whale" ≠ "10 M across 2 M tiny clients" for the GPU.
**Result:** the router stopped mis-routing an all-in-one-partition whale to a
serial path.

### 5. Per-query GPU arm + a *calibrated* router
**Decision:** add a per-query GPU path (`fold_query` over a gathered subset) and
fit the router coefficients from **measured** per-query H2D/kernel times, not
priors.
**Why:** the router had only ever seen CPU observations; its `h2d_per_row=1.6`
implied ~86 ms H2D when the truth was ~346 ms.
**Result:** calibrated coefficients; cross-client all-history correctly routes to
GPU (it had been mis-routed to CPU).

### 6. OOM at scale → halve the GPU scratch
**Decision:** at 168 M rows the single-shot upload OOM'd the 16 GB T4 (96 B/row).
Pack the big-kernel's buy/sell scratch into shared arrays (**40 → 20 B/row**).
**Why:** buys + sells = partition size, so separate full-width arrays were 2×.
**Result:** 168 M fits; set up the streaming work.

### 7. CUDA streams + pinned host + chunking
**Decision:** process the table in ~4 M-row chunks across **two CUDA streams**
with the host buffer **page-locked** (`cuMemHostRegister`, `READ_ONLY` flag —
the read-only mmap rejects flag 0), overlapping H2D with kernel.
**Why:** Decision 3 said transfer ≥ kernel, so overlap + bounded VRAM matter.
**Result:** **VRAM bounded to ~2 chunks** (the OOM ceiling is gone — scales past
168 M) and ~1.18× overlap at the time. (Overlap later went moot — see #9.)

### 8. The stale-checkpoint bug
**Decision:** `build_periodic` now clears old `ckpt-*.json` before rebuilding.
**Why:** a regenerate left checkpoints at *different* cutoff days; a range query
loaded carry-in from the *previous* dataset → wrong PnL (caught by a three-way
diagnostic: `cpu_ckpt ≠ cpu_nockpt`, `nockpt == baseline`).
**Result:** range-query correctness restored and made un-repeatable.

### 9. The 12-byte record — transfer-bound → compute-bound
**Decision:** drop `ts` (ordering is **positional** after the build-time sort) and
the pad, and narrow fields to `i32` → **`{i32 signed_qty, i32 price_ticks, i32
day}` = 12 B** (was 32 B).
**Why:** the kernel/fold never read `ts`; it was 8 of every 32 transferred bytes
— pure waste on the hot path.
**Result:** `h2d_per_row` **6.97 → 2.78 ns**, H2D **1313 → 640 ms**, and the
regime **flipped to compute-bound** (kernel 687 > H2D 640) — the first time
moving bytes wasn't the bottleneck. GPU-vs-CPU **1.16× → 1.8×**, and the GPU now
also wins the 21.6 M-row whale. *Consequence:* bit-packing was **dropped** from
the roadmap (it only helps when transfer-bound) — data reversed a plan.

### 10. From FIFO-PnL to a generic engine (Axes 1 & 2)
**Decision:** recognize FIFO-PnL as one instance of an ordered per-partition
scan, and parameterize it:
- **A.1** configurable holding threshold (`--ltcg-days`),
- **A.2** arbitrary K-bucket rules (`[BucketPnl; 8]`, `classify`→index, `--bands 30,365`),
- **B.1** matching policy FIFO/LIFO/HIFO (`--policy`).
**Why:** jurisdictions differ (intraday/STCG/LTCG bands; tax-loss harvesting wants
HIFO); the storage + execution layers are shared, only per-partition logic varies.
**Result:** the same engine now does jurisdiction-configurable tax bucketing and
multiple cost-basis methods, all bit-reproducing the defaults.

### 11. Deferring GPU non-FIFO (B.2) — a decision *not* to build
**Decision:** keep LIFO/HIFO on the **CPU**; GPU stays the FIFO fast path.
**Why:** the within-partition scan identity is FIFO-specific; LIFO/HIFO could only
use the sequential kernel → the whale-tail returns → slow on the realistic data.
**Result:** a clean split (GPU = FIFO recomputes, CPU = any policy) instead of a
slow GPU path nobody would use. Knowing what *not* to build is a result too.

### 12. The Lance backend (Stages 1 → 2 → 2.1 → wired in → A.3)
**Decision:** make the versioned store real, in stages:
- **Stage 1** — write/read the compute table as a versioned Lance dataset (read→repack).
- **Stage 2** — store records in a transparent **`FixedSizeBinary(12)`** column
  (bytes = `[PackedTrade]` verbatim) and read into an **owned-buffer
  `PackedTable`** — no per-row rebuild, no temp file, GPU-DMA-able.
- **Stage 2.1** — *keep* per-row `client_id`/`symbol_id` (self-describing,
  queryable, audit-friendly; **zstd-compressed** via the `lance-encoding:compression`
  field metadata, since they're constant within a partition) **and** add a compact
  partition **sidecar**, so the fast read **projects only `rec`** + the sidecar (no
  redundant scan). *(Measured: without compression those columns stored ~uncompressed
  and doubled the dataset — see Part III; compression was added after seeing that.)*
- **Wired in** — `query`/`bench` take `--uri` to run straight off Lance; read cut
  to a single bulk copy.
- **A.3** — both GPU kernels generalized to **K-way bucketing**; `fifo query
  --gpu` honors custom `--bands`. *(Implemented; pending final T4 validation.)*
**Why:** speed barely justifies a rebuild — **reproducibility, correction-lineage,
and time-travel do**, and those are Lance's strengths; the compute table stays our
transparent format for the GPU hot path.
**Result:** versioned, self-describing storage with a fast (~3.8 s on 168 M)
project-only read, feeding the unchanged engine. Stage 2.1 specifically answered
"won't compacting lose context?" — no: keep the columns, add a sidecar, project.

### 13. Rename `fifo_lance` → `fifo_gpu`
**Decision:** rename the crate/binary (and later repo/dirs).
**Why:** the name should reflect what it is — a GPU/CPU FIFO engine over a
*Lance-shaped* store — not a Lance integration.
**Result:** honest naming; Lance is now a backend, not the identity.

**Cross-cutting decisions that shaped all of the above:** two-tier storage
(tradebook = system of record; compute table = lean cache); transparent encodings
only on the hot path (no decompress → GPU-direct); positional ordering (sort once
at ingest, never per query); and *benchmark-gated* everything.

---

# Part II — Why Lance over Parquet

> Scope note: for the **compute table** we use our own transparent packed format
> (the GPU hot path wants raw bytes, not a format library); the Lance backend is
> the **versioned store**. "Lance over Parquet" below is the case for the storage
> layer.

| Dimension | Parquet (status quo) | Lance / Lance-shaped store |
|---|---|---|
| **Read for compute** | full re-scan; row-group + opaque (LZ4/zstd) decode | transparent fixed-width column → bytes are the records; **DMA to GPU, no decompress** |
| **Random access** | scan whole matched partitions | byte-addressable; page-range index prunes to the range |
| **Versioning / time-travel** | none (rewrite files) | built-in versioned snapshots, ACID — reproducible training sets, honest backtests |
| **Corrections / lineage** | rewrite | regenerate one partition, bump version |
| **Self-describing** | yes (columns) | yes (per-row `client/symbol`, **zstd-compressed** since constant-within-partition) + compact sidecar for fast loads |
| **GPU-direct** | no (decode step) | yes — the whole reason for the transparent layout |

**Two measured wins over the Parquet baseline (168 M rows):**
1. **Same work, better bytes** (both fold all rows — isolates layout/decode):
   Parquet re-scan **12 972 ms** → packed CPU **2 461 ms** (**5.3×**) → packed GPU
   **1 162 ms** (**11.2×**). The 5.3× is layout alone (no decode); the GPU adds more.
2. **Less work** (index + checkpoints avoid the rescan): selective queries run
   **800×–10⁷×** faster than re-scanning whole histories.

**The deciding argument:** raw speed barely justifies a storage rebuild;
**reproducibility, correction-lineage, and time-travel do** — and that's exactly
what Lance adds over Parquet, while the transparent column is what makes the GPU
path possible at all.

---

# Part III — The compute advantage (measured)

**Hardware:** Tesla T4 (16 GB, sm_75, **PCIe 3.0**), Ubuntu 24.04, CUDA 13.
**Dataset:** **168 M trades**, 40 k clients, 1 k symbols, **30 whales = 99.8 % of
volume**, top whale 21.6 M trades, largest `(client,symbol)` partition 2.23 M rows
— power-law skew (the realistic hard case). **Compute record: 12 bytes.**
**Validation:** every query's PnL asserted equal to the Parquet baseline; GPU
validated bit-exact vs the CPU oracle (matched_qty exact, realized 1e-6).

### Per-query: where each engine wins (end-to-end, disk→answer)

| query | rows touched | CPU | GPU | Parquet baseline | winner / route |
|---|---|---|---|---|---|
| whale 1-day | 303 | 12 ms | — | 11 616 ms | **CPU** · 940× vs baseline |
| whale 1-month | 363 | 11 ms | — | 11 689 ms | **CPU** · 1 094× |
| whale random-range | 1 033 | 0.07 ms | — | 11 613 ms | **CPU** · 168 952× |
| retail all-history | 5 | ~0 ms | 40 ms | 11 227 ms | **CPU** (GPU fixed cost) |
| whale all-history | 21.6 M | 373 ms | **216 ms** | 11 729 ms | **GPU 1.7×** |
| cross-client all-history | 168 M | 2 885 ms | **1 153 ms** | 14 162 ms | **GPU 2.5×** |

**Throughput (cross-client 168 M fold):** GPU ≈ **145 M trades/s**, CPU ≈ 68 M/s,
Parquet baseline ≈ 13 M/s — so **~12× over the status quo**, and GPU ~2–2.5× over
the calibrated CPU on large recomputes.

### The full-table GPU fold, and the regime flip from the 12-byte record

| metric | 32-byte record | **12-byte record** |
|---|---|---|
| H2D transfer | 1 313 ms | **640 ms** |
| kernel | 667 ms | 688 ms |
| binding constraint | **transfer** (H2D > kernel) | **compute** (kernel > H2D) |
| `h2d_per_row` (fitted) | 6.97 ns | **2.78 ns** |
| GPU total vs CPU full fold | 1.16× | **1.8×** |

Dropping `ts` cut the transfer ~2× and **inverted the bottleneck** — the workload
is now compute-bound, so the next lever is kernel throughput, not fewer bytes.

### Lance backend timing (168 M)

| op | time | note |
|---|---|---|
| write (versioned dataset + sidecar) | ~13 s | keeps self-describing columns |
| read Stage 2 (all columns) | 6.66 s | derive partitions by scan |
| **read Stage 2.1 (project `rec` + sidecar)** | **3.77 s** | ~1.8× faster load |

### On-disk storage footprint (168 M, measured with `du -sh`)

| store | on disk | what |
|---|---|---|
| Parquet tradebook (`data/big`) | **1.6 GB** | wide source, compressed |
| Packed compute table (`.fifopack`) | **1.9 GB** | transparent, **uncompressed** (GPU-direct, by design — Decision 5) |
| Lance dataset, **one version, uncompressed cols** | **3.8 GB** | `rec` ~2 GB + per-row client/symbol ~1.8 GB *(uncompressed)* |
| — same, **two retained versions** | 7.6 GB | versioning copies on overwrite until compaction |
| — **one version, client/symbol zstd** | **~2 GB (expected)** | redundant columns crushed; ≈ fifopack, self-description kept |
| Partition sidecar (`.parts.json`) | 1.3 MB | compact 74 k `(client,symbol,offset)` triples |

**Findings:** (1) the packed format is slightly *larger* than Parquet on disk
because it's deliberately uncompressed for GPU-direct reads — a tiny price for the
~10× faster read. (2) Lance defaulted to **uncompressed** per-row columns (~24 B/row),
making a version ~2× the packed store; **zstd on those columns** (constant within a
partition) recovers ~1.8 GB → Lance ≈ fifopack while keeping self-description.
(3) overwrites **retain old versions** (time-travel) — compact / `rm`+rewrite to
reclaim.

### Crossover, in one sentence
**CPU wins selective/point/range queries (no transfer tax — 100×–10⁷× over the
Parquet status quo); GPU wins large recomputes (tens of millions of rows+, ~2×
over CPU); a calibrated cost-router picks per query.** Skew — not volume — was the
hard part, and the within-partition kernel plus the 12-byte record are what turned
the GPU from a loss into a 2× win on the realistic, whale-heavy data.
