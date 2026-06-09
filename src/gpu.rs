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
//! **Whale parallelism (Decision 2).** A single thread folding a 300K–15M-record
//! partition was the tail latency. `fifo_kernel_big` parallelizes *within* such a
//! partition with one **block** per partition:
//!   1. cooperative chunked **scan** — split the interleaved record stream into
//!      time-ordered buy/sell arrays and build cumulative-quantity prefix axes
//!      (`buyCum`, `sellCum`);
//!   2. **searchsorted** matching — under strict FIFO with no shorting the n-th
//!      share sold matches the n-th share bought, so FIFO matching is interval
//!      overlap of the two monotonic axes (binary search, no recurrence);
//!   3. **segmented reduce** — each overlap fragment is tagged
//!      intraday/short/long and reduced into 3 buckets.
//! Fragment boundaries are exactly the CPU oracle's (a fragment ends when a buy
//! lot or a sell is exhausted), so matched_qty stays integer-exact and realized
//! PnL matches to f64 tolerance — the same validation the simple kernel passes.
//!
//! Partitions below [`BIG_PARTITION_THRESHOLD`] still use the one-thread-per-
//! partition `fifo_kernel` (launch/serial overhead is negligible there); it now
//! early-returns on big partitions so the two kernels partition the work cleanly.

use crate::fifo::{BucketPnl, PartitionPnl};
use crate::packed::PackedTable;
use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

/// Partitions with at least this many records go to the within-partition kernel.
/// The ~50 whale `(client,symbol)` partitions (~300K+ records) cross this; the
/// long tail of tiny retail partitions stays on the one-thread kernel.
pub const BIG_PARTITION_THRESHOLD: u64 = 1 << 15; // 32768

/// Threads per block for `fifo_kernel_big`. Static shared scratch is sized for
/// up to 1024 so this can be tuned up without touching the kernel.
const BIG_BLOCK: u32 = 256;

const KERNEL: &str = r#"
extern "C" {

struct Rec  { long long signed_qty; long long price_ticks; int day; int pad; long long ts; }; // 32B
struct Lot  { long long qty; long long price; int day; int pad; };                              // 24B

// ---- small-partition arm: one thread per partition, sequential FIFO drain ----
__global__ void fifo_kernel(
    const Rec*  recs,
    const unsigned long long* offsets,
    int          n_parts,
    unsigned long long big_threshold,
    Lot*         lots,           // scratch, one slot per record (worst case all-open)
    double*      out_realized,   // [n_parts*3]  intraday/short/long
    long long*   out_qty)        // [n_parts*3]
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_parts) return;

    unsigned long long start = offsets[p];
    unsigned long long end   = offsets[p + 1];
    if (end - start >= big_threshold) return;   // handled by fifo_kernel_big

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

// ---- big-partition arm: one block per partition (scan + searchsorted + reduce) ----
// Per-thread scan partials (sized for the max supported block of 1024).
__global__ void fifo_kernel_big(
    const Rec*  recs,
    const unsigned long long* offsets,
    const unsigned int* big_parts,   // [n_big] partition indices for this launch
    int          n_big,
    // per-record scratch, indexed from each partition's `start`. buys and sells
    // share these arrays: buys occupy [start, start+nb), sells [start+nb, end).
    // Since nb+ns == partition size, one set of full-width arrays suffices (half
    // the memory of separate buy/sell arrays).
    long long*   cum,  long long* price, int* day,
    double*      out_realized,
    long long*   out_qty)
{
    int blk = blockIdx.x;
    if (blk >= n_big) return;
    int p = (int)big_parts[blk];

    unsigned long long start = offsets[p];
    unsigned long long end   = offsets[p + 1];
    unsigned long long N     = end - start;

    int tid = threadIdx.x;
    int nt  = blockDim.x;

    __shared__ int       s_nb[1024];   // buys in thread t's chunk -> then buy_start[t]
    __shared__ int       s_ns[1024];   // sells   "       "        -> then sell_start[t]
    __shared__ long long s_bq[1024];   // buy qty in chunk -> then cum buy qty before chunk
    __shared__ long long s_sq[1024];
    __shared__ int       s_total_nb;
    __shared__ int       s_total_ns;

    // chunk [c0, c1) of the partition, contiguous to preserve time order
    unsigned long long C  = (N + (unsigned long long)nt - 1) / (unsigned long long)nt;
    unsigned long long c0 = start + (unsigned long long)tid * C;
    unsigned long long c1 = c0 + C; if (c1 > end) c1 = end; if (c0 > end) c0 = end;

    // pass A: per-thread counts and qty sums
    int nb = 0, ns = 0; long long bq = 0, sq = 0;
    for (unsigned long long i = c0; i < c1; i++) {
        long long s = recs[i].signed_qty;
        if (s > 0) { nb++; bq += s; }
        else if (s < 0) { ns++; sq += -s; }
    }
    s_nb[tid] = nb; s_ns[tid] = ns; s_bq[tid] = bq; s_sq[tid] = sq;
    __syncthreads();

    // pass B: exclusive scan over the (<=1024) per-thread partials in thread 0
    if (tid == 0) {
        int accb = 0, accs = 0; long long accbq = 0, accsq = 0;
        for (int t = 0; t < nt; t++) {
            int  vb = s_nb[t], vs = s_ns[t];
            long long vbq = s_bq[t], vsq = s_sq[t];
            s_nb[t] = accb; s_ns[t] = accs; s_bq[t] = accbq; s_sq[t] = accsq;
            accb += vb; accs += vs; accbq += vbq; accsq += vsq;
        }
        s_total_nb = accb; s_total_ns = accs;
    }
    __syncthreads();

    int tnb = s_total_nb, tns = s_total_ns;
    unsigned long long sb = start + (unsigned long long)tnb; // sell region base

    // pass C: re-walk chunk, scatter buys to [start..), sells to [sb..) + cum axes
    {
        int bi = s_nb[tid]; int si = s_ns[tid];
        long long bcum = s_bq[tid]; long long scum = s_sq[tid];
        for (unsigned long long i = c0; i < c1; i++) {
            long long s = recs[i].signed_qty;
            if (s > 0) {
                bcum += s;
                cum[start + bi]   = bcum;
                price[start + bi] = recs[i].price_ticks;
                day[start + bi]   = recs[i].day;
                bi++;
            } else if (s < 0) {
                scum += -s;
                cum[sb + si]   = scum;
                price[sb + si] = recs[i].price_ticks;
                day[sb + si]   = recs[i].day;
                si++;
            }
        }
    }
    __syncthreads();

    if (tnb == 0 || tns == 0) {
        if (tid == 0) {
            out_realized[p*3+0] = 0; out_realized[p*3+1] = 0; out_realized[p*3+2] = 0;
            out_qty[p*3+0] = 0; out_qty[p*3+1] = 0; out_qty[p*3+2] = 0;
        }
        return;
    }

    long long Bt = cum[start + tnb - 1];
    long long St = cum[sb + tns - 1];
    long long T  = Bt < St ? Bt : St;     // total matched shares (no shorting => St)

    // searchsorted match: each thread takes a stride of sells, finds the buy
    // interval containing its start via binary search, walks overlaps forward.
    double lr0 = 0, lr1 = 0, lr2 = 0;
    unsigned long long lm0 = 0, lm1 = 0, lm2 = 0;
    for (int j = tid; j < tns; j += nt) {
        long long sLo = (j == 0) ? 0 : cum[sb + j - 1];
        long long sHi = cum[sb + j];
        if (sLo >= T) continue;
        if (sHi > T) sHi = T;
        long long sp = price[sb + j];
        int sd = day[sb + j];

        // first buy i with cum[i] > sLo  (the interval covering position sLo)
        int lo = 0, hi = tnb;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (cum[start + mid] > sLo) hi = mid; else lo = mid + 1;
        }
        int i = lo;
        while (i < tnb) {
            long long bLo = (i == 0) ? 0 : cum[start + i - 1];
            if (bLo >= sHi) break;
            long long bHi = cum[start + i];
            long long segLo = sLo > bLo ? sLo : bLo;
            long long segHi = sHi < bHi ? sHi : bHi;
            long long q = segHi - segLo;
            if (q > 0) {
                double pnl = (double)q * (double)(sp - price[start + i]);
                int span = sd - day[start + i];
                if (span == 0)        { lr0 += pnl; lm0 += (unsigned long long)q; }
                else if (span <= 365) { lr1 += pnl; lm1 += (unsigned long long)q; }
                else                  { lr2 += pnl; lm2 += (unsigned long long)q; }
            }
            i++;
        }
    }

    // segmented reduce into the partition's 3 buckets (one block per partition,
    // so atomics here contend only within this block on 6 addresses).
    atomicAdd(&out_realized[p*3+0], lr0);
    atomicAdd(&out_realized[p*3+1], lr1);
    atomicAdd(&out_realized[p*3+2], lr2);
    atomicAdd((unsigned long long*)&out_qty[p*3+0], lm0);
    atomicAdd((unsigned long long*)&out_qty[p*3+1], lm1);
    atomicAdd((unsigned long long*)&out_qty[p*3+2], lm2);
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
        // Target sm_75 (Tesla T4) so atomicAdd(double) is available — the
        // default NVRTC arch is too low for double atomics in fifo_kernel_big.
        let opts = CompileOptions {
            arch: Some("compute_75"),
            ..Default::default()
        };
        let ptx =
            compile_ptx_with_opts(KERNEL, opts).context("NVRTC compile of fifo kernels")?;
        dev.load_ptx(ptx, "fifo", &["fifo_kernel", "fifo_kernel_big"])?;
        Ok(GpuEngine { dev })
    }

    /// Fold every partition on the GPU; returns per-partition (realized[3], qty[3])
    /// plus a transfer/kernel timing breakdown (Decision 3: transfer ≥ kernel).
    pub fn fold_all(
        &self,
        table: &PackedTable,
    ) -> Result<(Vec<[f64; 3]>, Vec<[i64; 3]>, GpuTiming)> {
        self.fold_buffers(table.records(), table.part_offset())
    }

    /// Fold an arbitrary set of partitions (full-history, no range carry-in) by
    /// gathering just those partitions' records into a contiguous buffer and
    /// uploading only that — so the H2D cost reflects the *query*, not the whole
    /// table. This is the per-query GPU arm the router calibrates against. Range
    /// queries (which need checkpoint carry-in) stay on CPU and aren't folded here.
    pub fn fold_query(
        &self,
        table: &PackedTable,
        parts: &[usize],
    ) -> Result<(PartitionPnl, GpuTiming)> {
        let total: usize = parts.iter().map(|&p| table.partition(p).len()).sum();
        let mut recs: Vec<crate::packed::PackedTrade> = Vec::with_capacity(total);
        let mut offsets: Vec<u64> = Vec::with_capacity(parts.len() + 1);
        offsets.push(0);
        for &p in parts {
            recs.extend_from_slice(table.partition(p));
            offsets.push(recs.len() as u64);
        }
        let (realized, qty, timing) = self.fold_buffers(&recs, &offsets)?;
        Ok((sum_pnl(&realized, &qty), timing))
    }

    /// Core GPU fold over a contiguous record buffer with prefix `offsets`
    /// (length `n_parts + 1`, starting at 0). Shared by [`fold_all`] (whole
    /// table) and [`fold_query`] (a gathered subset).
    fn fold_buffers(
        &self,
        records: &[crate::packed::PackedTrade],
        offsets: &[u64],
    ) -> Result<(Vec<[f64; 3]>, Vec<[i64; 3]>, GpuTiming)> {
        let recs_bytes: &[u8] = bytemuck::cast_slice(records);
        let n_parts = (offsets.len() - 1) as i32;
        let n_rows = records.len();

        // Split partitions: big ones (>= threshold) go to the within-partition
        // kernel, the rest to the one-thread-per-partition kernel.
        let big_threshold = BIG_PARTITION_THRESHOLD;
        let big_parts: Vec<u32> = (0..n_parts as usize)
            .filter(|&p| offsets[p + 1] - offsets[p] >= big_threshold)
            .map(|p| p as u32)
            .collect();
        let n_big = big_parts.len();

        let t_all = std::time::Instant::now();

        // ---- H2D: upload the transparent record buffer verbatim ----
        let t0 = std::time::Instant::now();
        let d_recs = self.dev.htod_sync_copy(recs_bytes)?;
        let d_offsets = self.dev.htod_sync_copy(offsets)?;
        let d_lots = self.dev.alloc_zeros::<u8>(n_rows * 24)?; // sizeof(Lot)
        let mut d_realized = self.dev.alloc_zeros::<f64>(n_parts as usize * 3)?;
        let mut d_qty = self.dev.alloc_zeros::<i64>(n_parts as usize * 3)?;
        // Per-record scratch for the big-partition kernel. buys and sells share
        // these arrays (buys [start..start+nb), sells [start+nb..end)), so one
        // set of full-width arrays suffices — 20 B/row, not 40.
        let d_big_parts = self.dev.htod_sync_copy(&big_parts)?;
        let mut d_cum = self.dev.alloc_zeros::<i64>(n_rows.max(1))?;
        let mut d_price = self.dev.alloc_zeros::<i64>(n_rows.max(1))?;
        let mut d_day = self.dev.alloc_zeros::<i32>(n_rows.max(1))?;
        self.dev.synchronize()?;
        let h2d_ms = t0.elapsed().as_secs_f64() * 1e3;

        // ---- kernels ----
        let t1 = std::time::Instant::now();
        // small-partition arm over all partitions (skips big ones internally)
        let cfg = LaunchConfig::for_num_elems(n_parts as u32);
        let func = self.dev.get_func("fifo", "fifo_kernel").unwrap();
        unsafe {
            func.launch(
                cfg,
                (
                    &d_recs,
                    &d_offsets,
                    n_parts,
                    big_threshold,
                    &d_lots,
                    &mut d_realized,
                    &mut d_qty,
                ),
            )?;
        }
        // big-partition arm: one block per big partition
        if n_big > 0 {
            let cfg_big = LaunchConfig {
                grid_dim: (n_big as u32, 1, 1),
                block_dim: (BIG_BLOCK, 1, 1),
                shared_mem_bytes: 0, // static shared scratch in the kernel
            };
            let func_big = self.dev.get_func("fifo", "fifo_kernel_big").unwrap();
            unsafe {
                func_big.launch(
                    cfg_big,
                    (
                        &d_recs,
                        &d_offsets,
                        &d_big_parts,
                        n_big as i32,
                        &mut d_cum,
                        &mut d_price,
                        &mut d_day,
                        &mut d_realized,
                        &mut d_qty,
                    ),
                )?;
            }
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
        Ok((sum_pnl(&realized, &qty), timing))
    }
}

/// Sum per-partition GPU outputs into a single [`PartitionPnl`] (realized PnL
/// rounded from the on-device f64 accumulator back to integer ticks).
fn sum_pnl(realized: &[[f64; 3]], qty: &[[i64; 3]]) -> PartitionPnl {
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
    pnl
}
