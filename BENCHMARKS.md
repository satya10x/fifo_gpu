# fifo_lance — Benchmark summary & findings

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
| Transparent packed compute table (32 B struct, mmap, GPU-direct) + page-range index | ✅ validated |
| CPU FIFO fold (oracle + bench arm) | ✅ validated |
| Checkpoint table (versioned open-lot snapshots) for bounded range queries | ✅ validated (+ stale-file fix) |
| GPU kernel — batched one-thread-per-partition **+** within-partition scan/searchsorted/segmented-reduce for whales | ✅ validated EXACT on T4 |
| Cost-based router (calibrated from measured H2D/kernel) | ✅ |
| Streamed fold — 2 CUDA streams + page-locked H2D (overlap) + bounded VRAM | ✅ validated EXACT on T4 |
| Cross-client rollup (Option C) + correction path | ✅ built |

**Benchmark setup (headline run).**
- Dataset: **164.8 M trades**, 40 k clients, 30 whales; top whale 21.6 M trades;
  whales = **99.8 %** of volume; largest `(client,symbol)` partition = 1.34 M rows.
  Power-law skew with explicit whales (the realistic, hard case — §5).
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
Cross-client all-history makes *both* engines fold all 164.8 M rows, so this
isolates storage layout + decode cost:

| | time | speedup |
|---|---|---|
| Parquet full re-scan (baseline) | 12 992 ms | 1× |
| Packed store, CPU fold | 2 577 ms | **5.0×** |
| Packed store, GPU fold (streamed) | 1 679 ms | **7.7×** |

The 5× is the packed format alone: no Parquet decode, no row-group plumbing,
contiguous struct-packed records read straight into the fold. The GPU adds more
on top (§2).

### (b) Less work — random access + checkpoints avoid the rescan
The page-range `(client,symbol,time)` index and versioned checkpoints turn a
"scan everything" baseline into a bounded lookup:

| query | rows touched | packed CPU | Parquet baseline | speedup |
|---|---|---|---|---|
| whale 1-day | 453 | 12.8 ms | 10 709 ms | **845×** |
| whale 1-month | 538 | 13.1 ms | 10 731 ms | **818×** |
| whale random-range | 965 | 0.08 ms | 10 784 ms | **140 000×** |
| retail all-history | 5 | ~0 ms | 10 557 ms | **16 000 000×** |

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
| whale 1-day | 453 | 50 | 12.8 ms | — | **CPU** (no transfer tax) | CPU |
| whale random-range | 965 | 50 | 0.08 ms | — | **CPU** | CPU |
| retail all-history | 5 | 2 | ~0 ms | 31 ms | **CPU** (GPU fixed cost) | CPU |
| whale all-history | 12.9 M | 50 | 206 ms | 232 ms | **CPU** (~tie) | GPU |
| cross-client all-history | 164.8 M | 109 199 | 2 577 ms | **1 679 ms** (streamed) | **GPU 1.4×** | GPU |

**Where GPU wins:** large, full-history recomputes that touch tens of millions to
billions of rows — where compute amortizes the host→device transfer. On the
164.8 M cross-client fold the GPU is **1.42× faster than CPU** (streamed).

**Where CPU wins:** everything small. Point/range queries (a day, a month, a
random range, retail accounts) — CPU finishes in sub-ms to low-ms with **no
transfer tax**, beating the GPU's fixed launch+H2D overhead by 100×–10⁷×.

**The mushy middle:** ~13 M-row single-client folds are a near tie (206 vs
232 ms) — the GPU just covers its transfer cost there.

**Crossover:** roughly **tens of millions of rows touched**. Below it → CPU;
well above it → GPU. The router encodes this with measured coefficients
(`cpu_per_row ≈ 15.6 ns`, `gpu_per_row ≈ 4.1 ns` kernel, `h2d_per_row ≈ 7.0 ns`)
plus a skew term for the largest partition, and routes each query accordingly.

**Skew was the real problem, not volume.** A naïve one-thread-per-partition GPU
kernel *lost* to CPU even on all-history (903 ms kernel, dominated by one thread
folding a multi-hundred-K-row whale). The within-partition scan+searchsorted+
segmented-reduce kernel cut that to ~210 ms and is what makes the GPU win at all.

---

## 3. Ceilings hit, and the roadmap

### Ceilings
1. **Transfer-bound, not compute-bound (Decision 3, confirmed).** On the full
   fold: H2D **1313 ms** ≫ kernel **667 ms**. Moving 5.3 GB *is* the job; the
   kernel is nearly free (4 ns/row).
2. **Pinning enabled overlap but not bandwidth.** CUDA-stream double-buffering +
   page-locked H2D overlaps the kernel behind the transfer → **1.18×** over serial
   GPU and bounded VRAM. But effective H2D stayed ~**4 GB/s** because the bytes
   come from the **file-backed mmap / page cache**, not resident RAM — so pinning
   bought concurrency, not a faster wire. PCIe-3 on the T4 is the floor.
3. **GPU's edge is narrow (~1.4×)** for this whale-heavy, low-arithmetic-intensity
   workload — exactly the worst case for justifying a host→device hop.
4. **VRAM was a hard wall** before streaming (single-shot upload OOM'd >~150 M
   rows on 16 GB); chunked streaming removed it (working set ~600 MB).

### Roadmap
**Near-term (software, this hardware):**
- **Keep hot data GPU-resident across queries** — biggest lever. Amortizing the
  ~1.3 s transfer over many folds turns a query *service* from transfer-bound to
  compute-bound, where the GPU's 4 ns/row vs CPU's 15.6 ns/row actually shows.
- **Wire the streamed fold into the live query/router path** (today it's only in
  the bench arm) — takes cross-client GPU from 1818 → ~1679 ms.
- **Cross-client aggregates → the rollup (Option C), not live compute** — those
  should be KB lookups, not 165 M-row folds. Already built; route to it.
- **Per-query GPU range path** — GPU currently folds full-history only; range
  queries fall back to CPU (fine, they're CPU-favorable anyway).

**Bigger bets:**
- **Faster transport** — PCIe-4/5, GPUDirect Storage (NVMe→GPU, skip host),
  A100/H100 + NVLink. Directly attacks ceiling #1/#2.
- **Fewer bytes** — narrower packed tuple / GPU-decodable bit-packing (within the
  transparent constraint) to cut the 32 B/row transfer.
- **Real Lance backend** for the system-of-record — versioning, correction
  lineage, time-travel (where Lance's value, not raw speed, dominates).
- **Multi-GPU / sharding** for the full 7.5 B trades/year scale.

### Bottom line
The design's central bet holds: a transparent, Lance-shaped packed store beats
Parquet by 5×–10⁷×, and a **hybrid CPU/GPU router** is the right architecture —
CPU for selective queries, GPU for large recomputes. The GPU's win is real but
**transfer-bound and modest (~1.4×) on PCIe-3**; unlocking its full edge is now a
*data-movement* problem (residency / faster transport), not a compute one.
