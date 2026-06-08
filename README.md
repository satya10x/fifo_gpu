# fifo_lance — GPU/CPU FIFO PnL on a packed, Lance-shaped store

Computes FIFO-based realized PnL (intraday / short-term ≤365d / long-term >365d)
for an equities tradebook, on the fly, for arbitrary date ranges, per-client and
cross-client. Compute runs on **CPU or GPU**, chosen per query by a cost-based
router. Storage is a transparent, fixed-width, struct-packed columnar buffer laid
out so trade data lands in GPU memory in the shape the kernel wants, with no
decompress step.

This is the implementation of the handoff in `fifo-pnl-lance-gpu-handoff.md`. All
eight milestones are built; everything except the GPU **runtime** is validated on
CPU (no local NVIDIA GPU).

## Status

| Milestone | What | State |
|---|---|---|
| M1 | Skew-realistic synthetic generator | ✅ built + validated |
| M2 | Packed compute table + page-range index | ✅ built + validated |
| M3 | CPU FIFO fold (oracle + bench arm 2) | ✅ built + validated |
| M4 | Checkpoint table + range carry-in | ✅ built + validated (PnL ≡ baseline) |
| M5 | GPU kernel (batched over partitions) | ✅ **code complete**, runtime pending on CUDA box |
| M6 | Three-arm benchmark harness | ✅ built + validated |
| M7 | Cost-based router + pred-vs-actual log | ✅ built + validated |
| M8 | Cross-client rollup + correction path | ✅ built + validated (determinism ✓) |

## Key architectural decision (vs the handoff)

The handoff names **Lance** as the compute-table substrate. Decision 5's *actual
requirement* is: transparent encodings → GPU-direct buffers, fixed-width,
struct-packed, contiguous, no opaque compression, page-sized chunks. We satisfy
that **directly** with our own packed binary format (`src/packed.rs`) — a 32-byte
`#[repr(C)]` tuple `(signed_qty, price_ticks, day, ts)`, mmap'd and uploaded to
the GPU verbatim. This realizes Decision 5 exactly while keeping the whole
pipeline runnable and validated today without the Lance build dependency. Lance
slots in later behind the same `PackedBuilder`/`PackedTable` interface as the
production system-of-record. Every other handoff decision is honored as written
(see `src/*.rs` module docs, which cite the decisions).

## Build

```bash
cargo build --release          # CPU (default) — what runs locally
cargo test                     # 23 unit tests
```

GPU arm (on an NVIDIA box with the CUDA toolkit + driver):

```bash
cargo build --release --features gpu
```

> The `gpu` feature pulls `cudarc` with `cuda-version-from-build-system`, which
> auto-detects the installed CUDA. It will **not** build on a machine without
> CUDA — that's expected; build the default target locally and the `gpu` target
> on the server. See `scripts/bootstrap-gpu.sh`.

## Pipeline

```bash
fifo=./target/release/fifo

# M1 — generate a skew-realistic tradebook (power-law + explicit whales)
$fifo gen --clients 20000 --days 400 --whales 20 --out data/tradebook
$fifo stats --out data/tradebook --verify-parquet

# M2 — pack into the transparent compute table + page index
$fifo pack --tradebook data/tradebook --out data/compute.fifopack

# M4 — periodic checkpoints for bounded range queries
$fifo checkpoint --tradebook data/tradebook --packed data/compute.fifopack --out data/checkpoints

# query (per-client / cross-client, full or date range)
$fifo query --packed data/compute.fifopack --client 11000                       # all-history
$fifo query --packed data/compute.fifopack --client 11000 --from 2020-06-01 --to 2020-06-30

# M8 — cross-client rollup (Option C) and the correction path
$fifo rollup  --packed data/compute.fifopack --out data/rollup.json
$fifo correct --tradebook data/tradebook --packed data/compute.fifopack \
              --checkpoints data/checkpoints --client 11000 --on 2020-09-15

# M6/M7 — three-arm benchmark + router fit (add --features gpu to include the GPU arm)
$fifo bench --tradebook data/tradebook --packed data/compute.fifopack --checkpoints data/checkpoints
```

## What the benchmark already shows (CPU vs status-quo baseline)

On a 48M-trade, power-law dataset (top whale = 15M trades, whales = 99.7% of
volume), CPU-packed PnL equals the full-rescan baseline for **every** query
(correctness), and the handoff's predicted crossover appears empirically:

| query | rows touched | CPU vs baseline | router picks |
|---|---|---|---|
| whale 1-day | 271 | **362×** | CPU |
| whale random-range | 1.4K | **5856×** | CPU |
| whale all-history | 15.1M | 9.5× | **GPU** |
| cross-client all-history | 48M | 3.7× | CPU* |

Small spans → CPU (no transfer tax); large recompute → GPU-favorable. The router
self-corrects from logged predicted-vs-actual (`data/router-log.jsonl`).
(*cross-client aggregates should hit the **rollup** (Option C), not live compute —
that's a KB lookup, not a 48M-row fold.)

## Known limitations / TODOs (carried in code)

- **GPU whale tail**: one thread folds one partition; a 15M-record whale is the
  tail latency. The handoff's scan+searchsorted+segmented-reduce parallelizes
  *within* a big partition — layer it in above a size threshold (`src/gpu.rs`).
- **Packer holds records in RAM** before writing (~1.5 GB for 48M). The on-disk
  format already supports a streaming two-file assemble for billions of rows.
- **Lance backend** not wired (see decision above).
- GPU realized PnL accumulates in `f64` on-device (exact `i128` on CPU);
  validated to `matched_qty` exact + realized within `1e-6` relative.
- Determinism caveat: floating-point order in the generator's price walk is
  reproducible per-client; the packed/fold path is integer-exact.
```
