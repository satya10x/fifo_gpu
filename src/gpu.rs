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

use crate::fifo::{BucketPnl, BucketRules, PartitionPnl};
use crate::packed::PackedTable;
use anyhow::{Context, Result};
use cudarc::driver::{
    result, sys, CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

/// Partitions with at least this many records go to the within-partition kernel.
/// The ~50 whale `(client,symbol)` partitions (~300K+ records) cross this; the
/// long tail of tiny retail partitions stays on the one-thread kernel.
pub const BIG_PARTITION_THRESHOLD: u64 = 1 << 15; // 32768

/// Threads per block for `fifo_kernel_big`. Static shared scratch is sized for
/// up to 1024 so this can be tuned up without touching the kernel.
const BIG_BLOCK: u32 = 256;

/// Target rows per chunk for the streamed fold (~4M ≈ 128 MiB of records). Sets
/// the H2D/kernel overlap granularity and bounds VRAM to ~2 chunks regardless of
/// total table size.
const STREAM_CHUNK_ROWS: u64 = 4_000_000;

const KERNEL: &str = r#"
extern "C" {

#define MAXB 8   // == MAX_BUCKETS in fifo.rs

struct Rec  { int signed_qty; int price_ticks; int day; };          // 12B, matches PackedTrade
struct Lot  { long long qty; long long price; int day; int pad; };  // 24B (open-lot scratch)

// Bucket index for a holding span (days) — mirrors the CPU classify(): intraday
// (span==0) is bucket 0 when `intraday`; then ascending inclusive `bounds` pick
// the band; past the last bound -> the final bucket.
__device__ __forceinline__ int bucket_of(int span, int intraday, const int* bounds, int n_bounds) {
    int base = 0;
    if (intraday) { if (span == 0) return 0; base = 1; }
    for (int i = 0; i < n_bounds; i++) { if (span <= bounds[i]) return base + i; }
    return base + n_bounds;
}

// ---- small-partition arm: one thread per partition, sequential FIFO drain ----
__global__ void fifo_kernel(
    const Rec*  recs,
    const unsigned long long* offsets,
    int          n_parts,
    unsigned long long big_threshold,
    int          intraday,
    int          n_bounds,
    const int*   bounds,
    Lot*         lots,           // scratch, one slot per record (worst case all-open)
    double*      out_realized,   // [n_parts*k]
    long long*   out_qty)        // [n_parts*k]
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_parts) return;
    int k = intraday + n_bounds + 1;   // number of buckets (<= MAXB)

    unsigned long long start = offsets[p];
    unsigned long long end   = offsets[p + 1];
    if (end - start >= big_threshold) return;   // handled by fifo_kernel_big

    Lot* q = lots + start;       // capacity (end - start)
    long long head = 0, tail = 0;

    double r[MAXB]; long long m[MAXB];
    for (int b = 0; b < k; b++) { r[b] = 0; m[b] = 0; }

    for (unsigned long long i = start; i < end; i++) {
        long long sq = recs[i].signed_qty;
        if (sq > 0) {
            q[tail].qty   = sq;
            q[tail].price = recs[i].price_ticks;
            q[tail].day   = recs[i].day;
            tail++;
        } else if (sq < 0) {
            long long rem = -sq;
            int sd  = recs[i].day;
            long long sp  = recs[i].price_ticks;
            while (rem > 0 && head < tail) {
                Lot* L = &q[head];
                long long mm  = rem < L->qty ? rem : L->qty;
                long long pnl = mm * (sp - L->price);
                int b = bucket_of(sd - L->day, intraday, bounds, n_bounds);
                r[b] += (double)pnl; m[b] += mm;
                L->qty -= mm;
                rem    -= mm;
                if (L->qty == 0) head++;
            }
        }
    }
    for (int b = 0; b < k; b++) { out_realized[p*k+b] = r[b]; out_qty[p*k+b] = m[b]; }
}

// ---- big-partition arm: one block per partition (scan + searchsorted + reduce) ----
// Per-thread scan partials (sized for the max supported block of 1024).
__global__ void fifo_kernel_big(
    const Rec*  recs,
    const unsigned long long* offsets,
    const unsigned int* big_parts,   // [n_big] partition indices for this launch
    int          n_big,
    int          intraday,
    int          n_bounds,
    const int*   bounds,
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
    int k = intraday + n_bounds + 1;   // number of buckets (<= MAXB)

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
            for (int b = 0; b < k; b++) { out_realized[p*k+b] = 0; out_qty[p*k+b] = 0; }
        }
        return;
    }

    long long Bt = cum[start + tnb - 1];
    long long St = cum[sb + tns - 1];
    long long T  = Bt < St ? Bt : St;     // total matched shares (no shorting => St)

    // searchsorted match: each thread takes a stride of sells, finds the buy
    // interval containing its start via binary search, walks overlaps forward.
    double lr[MAXB]; unsigned long long lm[MAXB];
    for (int b = 0; b < k; b++) { lr[b] = 0; lm[b] = 0; }
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
                int b = bucket_of(sd - day[start + i], intraday, bounds, n_bounds);
                lr[b] += pnl; lm[b] += (unsigned long long)q;
            }
            i++;
        }
    }

    // segmented reduce into the partition's k buckets (one block per partition).
    for (int b = 0; b < k; b++) {
        atomicAdd(&out_realized[p*k+b], lr[b]);
        atomicAdd((unsigned long long*)&out_qty[p*k+b], lm[b]);
    }
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

    /// Fold every partition on the GPU; returns flat per-partition `realized`/
    /// `qty` arrays (`n_parts * k`), the bucket count `k`, and a transfer/kernel
    /// timing breakdown.
    pub fn fold_all(
        &self,
        table: &PackedTable,
        rules: &BucketRules,
    ) -> Result<(Vec<f64>, Vec<i64>, usize, GpuTiming)> {
        self.fold_buffers(table.records(), table.part_offset(), rules)
    }

    /// Fold an arbitrary set of partitions (full-history, no range carry-in) by
    /// gathering just those partitions' records into a contiguous buffer and
    /// uploading only that — so the H2D cost reflects the *query*, not the whole
    /// table. Honors `rules` (K-way bucketing). Range queries (checkpoint
    /// carry-in) stay on CPU and aren't folded here.
    pub fn fold_query(
        &self,
        table: &PackedTable,
        parts: &[usize],
        rules: &BucketRules,
    ) -> Result<(PartitionPnl, GpuTiming)> {
        let total: usize = parts.iter().map(|&p| table.partition(p).len()).sum();
        let mut recs: Vec<crate::packed::PackedTrade> = Vec::with_capacity(total);
        let mut offsets: Vec<u64> = Vec::with_capacity(parts.len() + 1);
        offsets.push(0);
        for &p in parts {
            recs.extend_from_slice(table.partition(p));
            offsets.push(recs.len() as u64);
        }
        let (realized, qty, k, timing) = self.fold_buffers(&recs, &offsets, rules)?;
        Ok((sum_pnl(&realized, &qty, k), timing))
    }

    /// Core GPU fold over a contiguous record buffer with prefix `offsets`
    /// (length `n_parts + 1`, starting at 0), bucketed per `rules` (K buckets).
    /// Returns flat `realized`/`qty` (`n_parts * k`) + `k`. Shared by [`fold_all`]
    /// and [`fold_query`].
    fn fold_buffers(
        &self,
        records: &[crate::packed::PackedTrade],
        offsets: &[u64],
        rules: &BucketRules,
    ) -> Result<(Vec<f64>, Vec<i64>, usize, GpuTiming)> {
        let recs_bytes: &[u8] = bytemuck::cast_slice(records);
        let n_parts = (offsets.len() - 1) as i32;
        let n_rows = records.len();

        // bucket-rule params for the kernels
        let k = rules.num_buckets();
        let intraday: i32 = if rules.intraday_same_day { 1 } else { 0 };
        let n_bounds = rules.n_bounds as i32;
        let bounds_host: Vec<i32> = if rules.n_bounds == 0 {
            vec![0]
        } else {
            rules.boundaries[..rules.n_bounds].to_vec()
        };

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
        let mut d_realized = self.dev.alloc_zeros::<f64>(n_parts as usize * k)?;
        let mut d_qty = self.dev.alloc_zeros::<i64>(n_parts as usize * k)?;
        let d_bounds = self.dev.htod_sync_copy(&bounds_host)?;
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
                    intraday,
                    n_bounds,
                    &d_bounds,
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
                        intraday,
                        n_bounds,
                        &d_bounds,
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

        Ok((realized, qty, k, timing))
    }

    /// Whole-table totals as a [`PartitionPnl`] (realized PnL via f64 → ticks).
    pub fn fold_total(&self, table: &PackedTable, rules: &BucketRules) -> Result<(PartitionPnl, GpuTiming)> {
        let (realized, qty, k, timing) = self.fold_all(table, rules)?;
        Ok((sum_pnl(&realized, &qty, k), timing))
    }

    /// Streamed whole-table fold (Decision 5): process the table in
    /// partition-chunks across **two CUDA streams** with the host records buffer
    /// **page-locked** (`cuMemHostRegister`), so each chunk's H2D overlaps the
    /// previous chunk's kernel. VRAM is bounded to ~2 chunks regardless of table
    /// size, which removes the single-shot OOM ceiling.
    ///
    /// Returns the totals, end-to-end wall time (ms), and whether page-locking
    /// succeeded (if not, copies still run correctly but won't overlap).
    pub fn fold_total_streamed(&self, table: &PackedTable) -> Result<(PartitionPnl, f64, bool)> {
        use crate::packed::PackedTrade;

        let records = table.records();
        let offsets = table.part_offset();
        let n_parts = offsets.len() - 1;
        if n_parts == 0 {
            return Ok((PartitionPnl::default(), 0.0, false));
        }
        let big_threshold = BIG_PARTITION_THRESHOLD;

        // 1. plan contiguous partition-chunks bounded by STREAM_CHUNK_ROWS
        let mut chunks: Vec<(usize, usize)> = Vec::new();
        let mut p0 = 0usize;
        while p0 < n_parts {
            let mut p1 = p0 + 1;
            while p1 < n_parts && (offsets[p1 + 1] - offsets[p0]) <= STREAM_CHUNK_ROWS {
                p1 += 1;
            }
            chunks.push((p0, p1));
            p0 = p1;
        }
        let max_rows = chunks
            .iter()
            .map(|&(a, b)| (offsets[b] - offsets[a]) as usize)
            .max()
            .unwrap()
            .max(1);
        let max_parts = chunks.iter().map(|&(a, b)| b - a).max().unwrap().max(1);

        // 2. page-lock the whole records mmap so async H2D reads directly from it
        // (no staging copy, and pinned bandwidth). It's a read-only file-backed
        // mapping, so flag 0 fails — use CU_MEMHOSTREGISTER_READ_ONLY.
        let rec_sz = std::mem::size_of::<PackedTrade>();
        const CU_MEMHOSTREGISTER_READ_ONLY: std::ffi::c_uint = 0x08;
        let host_ptr = records.as_ptr() as *mut std::ffi::c_void;
        let host_bytes = records.len() * rec_sz;
        let pinned = unsafe {
            sys::lib().cuMemHostRegister_v2(host_ptr, host_bytes, CU_MEMHOSTREGISTER_READ_ONLY)
        } == sys::CUresult::CUDA_SUCCESS;

        // 3. two streams + two device buffer slots (double buffering)
        let streams = [self.dev.fork_default_stream()?, self.dev.fork_default_stream()?];
        // records as raw bytes (the kernel reinterprets as Rec*) — avoids needing
        // DeviceRepr for PackedTrade, matching the serial fold_buffers path.
        let mut d_recs: Vec<CudaSlice<u8>> = Vec::new();
        let mut d_off: Vec<CudaSlice<u64>> = Vec::new();
        let mut d_bigp: Vec<CudaSlice<u32>> = Vec::new();
        let mut d_cum: Vec<CudaSlice<i64>> = Vec::new();
        let mut d_price: Vec<CudaSlice<i64>> = Vec::new();
        let mut d_day: Vec<CudaSlice<i32>> = Vec::new();
        let mut d_lots: Vec<CudaSlice<u8>> = Vec::new();
        for _ in 0..2 {
            unsafe {
                d_recs.push(self.dev.alloc::<u8>(max_rows * rec_sz)?);
                d_off.push(self.dev.alloc::<u64>(max_parts + 1)?);
                d_bigp.push(self.dev.alloc::<u32>(max_parts)?);
                d_cum.push(self.dev.alloc::<i64>(max_rows)?);
                d_price.push(self.dev.alloc::<i64>(max_rows)?);
                d_day.push(self.dev.alloc::<i32>(max_rows)?);
                d_lots.push(self.dev.alloc::<u8>(max_rows * 24)?);
            }
        }
        // ONE global output buffer (zeroed once). Each chunk's kernels write their
        // partition slice via a slice_mut view, and we do a SINGLE dtoh at the end
        // — no per-chunk device→host sync, which is what lets the streams overlap.
        let mut d_real = self.dev.alloc_zeros::<f64>(n_parts * 3)?;
        let mut d_qty = self.dev.alloc_zeros::<i64>(n_parts * 3)?;
        // The streamed arm uses the default 3-bucket ruleset (intraday/≤365d/long);
        // custom K-bucket rulesets use the non-streamed fold_query path.
        let d_bounds = self.dev.htod_sync_copy(&[365i32])?;

        let mut used = [false; 2];
        let t0 = std::time::Instant::now();
        for (i, &(cp0, cp1)) in chunks.iter().enumerate() {
            let slot = i % 2;
            // free this slot's input/staging buffers from the chunk 2 iters ago
            if used[slot] {
                unsafe { result::stream::synchronize(streams[slot].stream)? };
            }

            let r0 = offsets[cp0] as usize;
            let r1 = offsets[cp1] as usize;
            let parts = cp1 - cp0;
            let off_local: Vec<u64> = (cp0..=cp1).map(|p| offsets[p] - offsets[cp0]).collect();
            let big_local: Vec<u32> = (0..parts as u32)
                .filter(|&k| off_local[k as usize + 1] - off_local[k as usize] >= big_threshold)
                .collect();
            let nbig = big_local.len();

            let st = streams[slot].stream;
            unsafe {
                // async H2D straight from the page-locked mmap (overlaps the prior
                // chunk's kernel); tiny offset/index arrays are pageable.
                let rec_bytes: &[u8] = bytemuck::cast_slice(&records[r0..r1]);
                result::memcpy_htod_async(*d_recs[slot].device_ptr(), rec_bytes, st)?;
                result::memcpy_htod_async(*d_off[slot].device_ptr(), &off_local, st)?;
                if nbig > 0 {
                    result::memcpy_htod_async(*d_bigp[slot].device_ptr(), &big_local, st)?;
                }
            }

            // kernels write into this chunk's slice of the global output (the
            // small kernel overwrites small parts; the big kernel atomic-adds onto
            // the once-zeroed slice). Each slice_mut borrow ends when launch
            // returns, so chunks on different streams don't alias in Rust's view.
            let cfg_small = LaunchConfig::for_num_elems(parts as u32);
            let small = self.dev.get_func("fifo", "fifo_kernel").unwrap();
            unsafe {
                small.launch_on_stream(
                    &streams[slot],
                    cfg_small,
                    (
                        &d_recs[slot],
                        &d_off[slot],
                        parts as i32,
                        big_threshold,
                        1i32,        // intraday_same_day
                        1i32,        // n_bounds
                        &d_bounds,
                        &d_lots[slot],
                        &mut d_real.slice_mut(cp0 * 3..cp1 * 3),
                        &mut d_qty.slice_mut(cp0 * 3..cp1 * 3),
                    ),
                )?;
            }
            if nbig > 0 {
                let cfg_big = LaunchConfig {
                    grid_dim: (nbig as u32, 1, 1),
                    block_dim: (BIG_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                };
                let big = self.dev.get_func("fifo", "fifo_kernel_big").unwrap();
                unsafe {
                    big.launch_on_stream(
                        &streams[slot],
                        cfg_big,
                        (
                            &d_recs[slot],
                            &d_off[slot],
                            &d_bigp[slot],
                            nbig as i32,
                            1i32,        // intraday_same_day
                            1i32,        // n_bounds
                            &d_bounds,
                            &mut d_cum[slot],
                            &mut d_price[slot],
                            &mut d_day[slot],
                            &mut d_real.slice_mut(cp0 * 3..cp1 * 3),
                            &mut d_qty.slice_mut(cp0 * 3..cp1 * 3),
                        ),
                    )?;
                }
            }
            used[slot] = true;
        }

        // one sync + one dtoh of the whole output
        for s in &streams {
            unsafe { result::stream::synchronize(s.stream)? };
        }
        let rv = self.dev.dtoh_sync_copy(&d_real)?;
        let qv = self.dev.dtoh_sync_copy(&d_qty)?;
        let total_ms = t0.elapsed().as_secs_f64() * 1e3;

        if pinned {
            unsafe {
                let _ = sys::lib().cuMemHostUnregister(host_ptr);
            }
        }

        // streamed arm uses the default 3-bucket ruleset
        Ok((sum_pnl(&rv, &qv, 3), total_ms, pinned))
    }
}

/// Sum flat per-partition GPU outputs (`n_parts * k`) into a single
/// [`PartitionPnl`] (realized PnL rounded from the on-device f64 accumulator
/// back to integer ticks). The kernel emits the K buckets the active
/// [`BucketRules`] defines.
fn sum_pnl(realized: &[f64], qty: &[i64], k: usize) -> PartitionPnl {
    let mut pnl = PartitionPnl::default();
    let n_parts = if k > 0 { realized.len() / k } else { 0 };
    for p in 0..n_parts {
        for b in 0..k {
            pnl.bucket_mut(b).merge(&BucketPnl {
                realized_ticks: realized[p * k + b].round() as i128,
                matched_qty: qty[p * k + b] as i128,
                fragments: 0,
            });
        }
    }
    pnl
}
