//! GPU FIFO arm (M5) — benchmark arm 1. Feature-gated behind `--features gpu`;
//! compiles + runs on an NVIDIA box (CUDA driver + NVRTC). Built with cudarc
//! 0.12.
//!
//! **Parallelism model (Decision 2):** one thread per `(client,symbol)`
//! partition, thousands of partitions across the grid — *not* parallelism
//! within a partition. Each thread runs the same sequential FIFO drain as the
//! CPU oracle, so the GPU result validates bit-for-bit on `matched_qty` and to
//! f64 tolerance on realized PnL.
//!
//! The packed `records` buffer is uploaded **as-is** (transparent, fixed-width)
//! — no decode, the whole point of the M2 layout.
//!
//! TODO (whale optimization): a single thread folding a 15M-record whale is the
//! tail latency. The handoff's scan + searchsorted + segmented-reduce
//! formulation parallelizes *within* such a partition (prefix-sum the signed
//! qty, align cumulative buy/sell axes, searchsorted match, segmented reduce
//! into 3 buckets). Layer that in for partitions above a size threshold.

use crate::fifo::{BucketPnl, PartitionPnl};
use crate::packed::PackedTable;
use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

const KERNEL: &str = r#"
extern "C" {

struct Rec  { long long signed_qty; long long price_ticks; int day; int pad; long long ts; }; // 32B
struct Lot  { long long qty; long long price; int day; int pad; };                              // 24B

__global__ void fifo_kernel(
    const Rec*  recs,
    const unsigned long long* offsets,
    int          n_parts,
    Lot*         lots,           // scratch, one slot per record (worst case all-open)
    double*      out_realized,   // [n_parts*3]  intraday/short/long
    long long*   out_qty)        // [n_parts*3]
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_parts) return;

    unsigned long long start = offsets[p];
    unsigned long long end   = offsets[p + 1];
    Lot* q = lots + start;       // capacity (end - start)
    long long head = 0, tail = 0;

    double r0 = 0, r1 = 0, r2 = 0;
    long long m0 = 0, m1 = 0, m2 = 0;

    for (unsigned long long i = start; i < end; i++) {
        long long sq = recs[i].signed_qty;
        if (sq > 0) {
            q[tail].qty   = sq;
            q[tail].price = recs[i].price_ticks;
            q[tail].day   = recs[i].day;
            tail++;
        } else if (sq < 0) {
            long long rem = -sq;
            long long sd  = recs[i].day;
            long long sp  = recs[i].price_ticks;
            while (rem > 0 && head < tail) {
                Lot* L = &q[head];
                long long m   = rem < L->qty ? rem : L->qty;
                long long pnl = m * (sp - L->price);
                long long span = sd - (long long)L->day;
                if (span == 0)        { r0 += (double)pnl; m0 += m; }
                else if (span <= 365) { r1 += (double)pnl; m1 += m; }
                else                  { r2 += (double)pnl; m2 += m; }
                L->qty -= m;
                rem    -= m;
                if (L->qty == 0) head++;
            }
        }
    }
    out_realized[p*3+0] = r0; out_realized[p*3+1] = r1; out_realized[p*3+2] = r2;
    out_qty[p*3+0] = m0; out_qty[p*3+1] = m1; out_qty[p*3+2] = m2;
}

} // extern "C"
"#;

#[derive(Clone, Copy, Debug, Default)]
pub struct GpuTiming {
    pub h2d_ms: f64,
    pub kernel_ms: f64,
    pub d2h_ms: f64,
    pub total_ms: f64,
}

pub struct GpuEngine {
    dev: Arc<CudaDevice>,
}

impl GpuEngine {
    pub fn new(ordinal: usize) -> Result<Self> {
        let dev = CudaDevice::new(ordinal).context("CudaDevice::new")?;
        let ptx = compile_ptx(KERNEL).context("NVRTC compile of fifo_kernel")?;
        dev.load_ptx(ptx, "fifo", &["fifo_kernel"])?;
        Ok(GpuEngine { dev })
    }

    /// Fold every partition on the GPU; returns per-partition (realized[3], qty[3])
    /// plus a transfer/kernel timing breakdown (Decision 3: transfer ≥ kernel).
    pub fn fold_all(
        &self,
        table: &PackedTable,
    ) -> Result<(Vec<[f64; 3]>, Vec<[i64; 3]>, GpuTiming)> {
        let recs_bytes: &[u8] = bytemuck::cast_slice(table.records());
        let offsets: &[u64] = table.part_offset();
        let n_parts = table.n_parts() as i32;
        let n_rows = table.n_rows() as usize;

        let t_all = std::time::Instant::now();

        // ---- H2D: upload the transparent record buffer verbatim ----
        let t0 = std::time::Instant::now();
        let d_recs = self.dev.htod_sync_copy(recs_bytes)?;
        let d_offsets = self.dev.htod_sync_copy(offsets)?;
        let d_lots = self.dev.alloc_zeros::<u8>(n_rows * 24)?; // sizeof(Lot)
        let mut d_realized = self.dev.alloc_zeros::<f64>(n_parts as usize * 3)?;
        let mut d_qty = self.dev.alloc_zeros::<i64>(n_parts as usize * 3)?;
        self.dev.synchronize()?;
        let h2d_ms = t0.elapsed().as_secs_f64() * 1e3;

        // ---- kernel ----
        let t1 = std::time::Instant::now();
        let cfg = LaunchConfig::for_num_elems(n_parts as u32);
        let func = self.dev.get_func("fifo", "fifo_kernel").unwrap();
        unsafe {
            func.launch(
                cfg,
                (
                    &d_recs,
                    &d_offsets,
                    n_parts,
                    &d_lots,
                    &mut d_realized,
                    &mut d_qty,
                ),
            )?;
        }
        self.dev.synchronize()?;
        let kernel_ms = t1.elapsed().as_secs_f64() * 1e3;

        // ---- D2H ----
        let t2 = std::time::Instant::now();
        let realized = self.dev.dtoh_sync_copy(&d_realized)?;
        let qty = self.dev.dtoh_sync_copy(&d_qty)?;
        let d2h_ms = t2.elapsed().as_secs_f64() * 1e3;

        let timing = GpuTiming {
            h2d_ms,
            kernel_ms,
            d2h_ms,
            total_ms: t_all.elapsed().as_secs_f64() * 1e3,
        };

        let per_realized: Vec<[f64; 3]> = realized.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let per_qty: Vec<[i64; 3]> = qty.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        Ok((per_realized, per_qty, timing))
    }

    /// Whole-table totals as a [`PartitionPnl`] (realized PnL via f64 → ticks).
    pub fn fold_total(&self, table: &PackedTable) -> Result<(PartitionPnl, GpuTiming)> {
        let (realized, qty, timing) = self.fold_all(table)?;
        let mut pnl = PartitionPnl::default();
        for (r, q) in realized.iter().zip(qty.iter()) {
            pnl.intraday.merge(&BucketPnl {
                realized_ticks: r[0].round() as i128,
                matched_qty: q[0] as i128,
                fragments: 0,
            });
            pnl.short.merge(&BucketPnl {
                realized_ticks: r[1].round() as i128,
                matched_qty: q[1] as i128,
                fragments: 0,
            });
            pnl.long.merge(&BucketPnl {
                realized_ticks: r[2].round() as i128,
                matched_qty: q[2] as i128,
                fragments: 0,
            });
        }
        Ok((pnl, timing))
    }
}
