# fifo_gpu — Benchmark summary & findings

GPU/CPU FIFO realized-PnL over a transparent, Lance-shaped packed store, with a
cost-based router. This document summarizes what was built and the three
benchmark comparisons that gate the design.

> **Scope note (read first).** We did **not** run the Lance library. We built our
> own transparent, fixed-width, struct-packed binary format (`src/packed.rs`)
> that *realizes Lance's design principles* — transparent encodings (no
> decompress), no row groups, struct packing, byte-addressable random access —
> so the buffer mmaps and DMAs to the GPU verbatim. "Lance advantage" below means
> "the advantage of a Lance-shaped store over Parquet," measured with that packed
> format. A real Lance backend slots behind the same interface later (for
> versioning / correction lineage, where Lance's value is highest).

---

## 0. What we built

| Component | Status |
|---|---|
| Skew-realistic synthetic tradebook generator (power-law + explicit whales) | ✅ validated |
| Transparent packed compute table (12 B struct, mmap, GPU-direct) + page-range index | ✅ validated |
| CPU FIFO fold (oracle + bench arm) | ✅ validated |
| Checkpoint table (versioned open-lot snapshots) for bounded range queries | ✅ validated (+ stale-file fix) |
| GPU kernel — batched one-thread-per-partition **+** within-partition scan/searchsorted/segmented-reduce for whales | ✅ validated EXACT on T4 |
| Cost-based router (calibrated from measured H2D/kernel) | ✅ |
| Streamed fold — 2 CUDA streams + page-locked H2D (overlap) + bounded VRAM | ✅ validated EXACT on T4 |
| Cross-client rollup (Option C) + correction path | ✅ built |

**Benchmark setup (headline run).**
- Dataset: **168 M trades**, 40 k clients, 30 whales; top whale 21.6 M trades;
  whales = **99.8 %** of volume; largest `(client,symbol)` partition = 2.23 M rows.
  Power-law skew with explicit whales (the realistic, hard case — §5).
- Compute record: **12 bytes** `{i32 signed_qty, i32 price_ticks, i32 day}` —
  `ts` is not stored (ordering is positional). This narrowing from an earlier
  32 B record is what flips the GPU regime (see §3).
- Hardware: **Tesla T4** (16 GB, sm_75, **PCIe 3.0**), Ubuntu 24.04, CUDA 13.
- Three arms over the *same* packed store, timed end-to-end (disk→answer):
  GPU, vectorized CPU, and status-quo Parquet full-rescan.
- Every query's PnL is asserted equal to the baseline (correctness gate); GPU is
  validated against the CPU oracle (matched_qty **exact**, realized within 1e-6).

---

## 1. Lance-shaped packed store vs Parquet

The status-quo baseline is a **Parquet full re-scan** (today's FIFO PnL path). The
packed store wins two different ways:

### (a) Same work, better bytes — the pure layout win
Cross-client all-history makes *both* engines fold all 168 M rows, so this
isolates storage layout + decode cost:

| | time | speedup |
|---|---|---|
| Parquet full re-scan (baseline) | 12 972 ms | 1× |
| Packed store, CPU fold | 2 461 ms | **5.3×** |
| Packed store, GPU fold | 1 162 ms | **11.2×** |

The 5× is the packed format alone: no Parquet decode, no row-group plumbing,
contiguous struct-packed records read straight into the fold. The GPU adds more
on top (§2).

### (b) Less work — random access + checkpoints avoid the rescan
The page-range `(client,symbol,time)` index and versioned checkpoints turn a
"scan everything" baseline into a bounded lookup:

| query | rows touched | packed CPU | Parquet baseline | speedup |
|---|---|---|---|---|
| whale 1-day | 303 | 12.0 ms | 10 852 ms | **906×** |
| whale 1-month | 363 | 10.5 ms | 10 913 ms | **1 042×** |
| whale random-range | 1 033 | 0.08 ms | 10 848 ms | **143 785×** |
| retail all-history | 5 | ~0 ms | 10 522 ms | **13 900 000×** |

These large factors are the **random-access advantage** (the Parquet baseline
re-scans the matched partitions' whole history; the packed store reads only the
range, with checkpoint carry-in for FIFO state). This is exactly where a
Lance-style format earns its keep.

**Takeaway.** Layout alone is ~5× on equal work; index + checkpoints are 100×–10⁷×
on selective queries. The transparent encoding is also what makes the GPU path
*possible at all* (DMA without a CPU decompress step).

---

## 2. GPU vs CPU — and on what data

Both arms run over the same packed store. The crossover is **empirical**, not
assumed, and it lands exactly where low arithmetic intensity predicts.

| query | rows touched | partitions | CPU | GPU | winner | router |
|---|---|---|---|---|---|---|
| whale 1-day | 303 | 50 | 12.0 ms | — | **CPU** (no transfer tax) | CPU |
| whale random-range | 1 033 | 50 | 0.08 ms | — | **CPU** | CPU |
| retail all-history | 5 | 2 | ~0 ms | 40 ms | **CPU** (GPU fixed cost) | CPU |
| whale all-history | 21.6 M | 50 | 317 ms | **227 ms** | **GPU 1.4×** | GPU |
| cross-client all-history | 168 M | 74 575 | 2 461 ms | **1 162 ms** | **GPU 2.1×** | GPU |

**Where GPU wins:** large, full-history recomputes that touch tens of millions to
billions of rows — where compute amortizes the host→device transfer. On the
168 M cross-client fold the GPU is **2.1× faster than CPU**, and after the 12 B
record narrowing (§3) it now also wins the 21.6 M-row whale all-history (227 vs
317 ms) — a query that used to favour CPU at 32 B/row.

**Where CPU wins:** everything small. Point/range queries (a day, a month, a
random range, retail accounts) — CPU finishes in sub-ms to low-ms with **no
transfer tax**, beating the GPU's fixed launch+H2D overhead by 100×–10⁷×.

**Crossover:** roughly **low tens of millions of rows touched** (it dropped after
the 12 B narrowing cut the transfer tax). Below it → CPU; above → GPU. The router
encodes this with measured coefficients (`cpu_per_row ≈ 14.6 ns`,
`gpu_per_row ≈ 4.2 ns` kernel, `h2d_per_row ≈ 2.8 ns` — down from 7.0 at 32 B)
plus a skew term for the largest partition, and routes each query accordingly.

**Skew was the real problem, not volume.** A naïve one-thread-per-partition GPU
kernel *lost* to CPU even on all-history (903 ms kernel, dominated by one thread
folding a multi-hundred-K-row whale). The within-partition scan+searchsorted+
segmented-reduce kernel cut that to ~210 ms and is what makes the GPU win at all.

---

## 3. Ceilings hit, and the roadmap

### The regime flip — transfer-bound → compute-bound
The single most informative result. At the original **32 B** record the fold was
**transfer-bound**: H2D **1313 ms** ≫ kernel **667 ms** (~4 GB/s, low arithmetic
intensity). Narrowing the record to **12 B** (drop the never-read `ts` + pad,
i32 fields) cut the transfer ~2.5× and **flipped the regime**:

| | 32 B record | 12 B record |
|---|---|---|
| `h2d_per_row` | 6.97 ns | **2.78 ns** |
| GPU disk+H2D | 1313 ms | **640 ms** |
| GPU kernel | 667 ms | 688 ms (unchanged) |
| binding term | **transfer** (H2D > kernel) | **compute** (kernel > H2D) |
| GPU vs CPU (full fold) | 1.16× | **1.80×** |

So the GPU's edge ~doubled, and **compute is now the bottleneck for the first
time** — `ts` was 8 of every 32 transferred bytes, pure waste on the hot path.

### Ceilings
1. **Now compute-bound** (kernel 688 ms > H2D 640 ms). Further transfer tricks
   (bit-packing, more overlap) no longer help; **kernel throughput is the lever**.
2. **CUDA-stream overlap is now ~neutral (1.00×).** At 32 B it bought 1.18× by
   hiding the kernel behind the transfer; at 12 B transfer ≈ kernel, so there's
   nothing left to hide and the stream/pin overhead cancels it. Streaming's
   remaining value is **bounded VRAM**, not speed.
3. **Bounded VRAM (kept).** Chunked streaming caps the working set at ~2 chunks
   (~600 MB) regardless of table size — removed the single-shot OOM wall (>~150 M
   rows on 16 GB) and is what lets us scale past 168 M.
4. **Pinning gave overlap, never bandwidth** — bytes come from the file-backed
   mmap/page cache, so H2D never beat ~PCIe-3 effective. Moot now that we're
   compute-bound.

### Roadmap
**Now that it's compute-bound:**
- **Kernel optimization** (newly the live lever) — the within-partition kernel is
  one block × 256 threads per whale; the 2.23 M-row max partition is the tail.
  More threads/occupancy could cut the 688 ms.
- **Keep hot data GPU-resident across queries** — still the biggest service-level
  lever: amortize even the now-halved transfer over many folds.
- **Cross-client aggregates → the rollup (Option C), not live compute** — KB
  lookups, not 168 M-row folds. Built; route to it.
- **Generic engine** (see `DESIGN.md`) — configurable bucket rules (A.1 done:
  `--ltcg-days`), then matching policies (LIFO/HIFO), then a window-function
  library — broadening *what* the engine computes, not just how fast.

**Bigger bets:** faster transport (PCIe-4/5, GPUDirect Storage, A100/H100+NVLink);
a real **Lance backend** for the versioned system-of-record (corrections /
time-travel); multi-GPU / sharding for the full 7.5 B trades/year scale.

> **Note — bit-packing dropped.** A sub-byte packed record (the old "Tier 3")
> only helps when transfer-bound; we are now compute-bound, so it would add
> in-kernel unpack cost for no gain. The 12 B narrowing was the right stopping
> point on the storage axis.

### Bottom line
The design's central bet holds: a transparent, Lance-shaped packed store beats
Parquet by 5×–10⁷×, and a **hybrid CPU/GPU router** is the right architecture —
CPU for selective queries, GPU for large recomputes. After the 12 B narrowing the
GPU's lead is a solid **~1.8–2.1×** on PCIe-3 and the workload is **compute-bound**
— the remaining levers are kernel throughput, data residency, and faster
transport, not fewer bytes.
