//! Cost-based router (M7, Decision 9) — picks GPU vs CPU per query.
//!
//! Not one magic constant: a small additive cost model whose coefficients are
//! fit *empirically* from the benchmark and recalibrated from logged
//! predicted-vs-actual error as the skew distribution drifts.
//!
//! ```text
//! cpu_ns = rows·cpu_per_row + checkpoints·ckpt_load
//! gpu_ns = gpu_fixed + rows·(h2d_per_row + gpu_per_row)
//!          + fanout·launch_overhead + max_part·gpu_serial_per_row
//!          + checkpoints·ckpt_load
//! route  = argmin(cpu_ns, gpu_ns)
//! ```
//! Two skew terms, because skew bites the GPU two opposite ways:
//! - `fanout` — many tiny partitions ⇒ launch/coordination overhead (favours CPU);
//! - `max_part` — the *largest* partition's residual within-block serialization
//!   (the whale tail). With the within-partition kernel this coefficient is small
//!   (≈ `gpu_per_row / BIG_BLOCK`), but it's the term that stopped the router from
//!   sending an all-in-one-partition whale to a single serial GPU thread.
//! `checkpoints` can dominate cross-client narrow ranges — another reason those
//! go to the rollup, not live compute.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Engine {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RouterCoeffs {
    pub cpu_per_row_ns: f64,
    pub gpu_per_row_ns: f64,
    pub h2d_per_row_ns: f64,
    pub launch_per_partition_ns: f64,
    /// Residual serial cost per row of the *largest* partition (whale tail).
    pub gpu_serial_per_row_ns: f64,
    pub ckpt_load_ns: f64,
    pub gpu_fixed_ns: f64,
}

impl Default for RouterCoeffs {
    /// Order-of-magnitude priors; the bench replaces these via [`fit`].
    fn default() -> Self {
        RouterCoeffs {
            cpu_per_row_ns: 3.0,
            gpu_per_row_ns: 0.2,
            h2d_per_row_ns: 1.6, // ~20 GB/s effective / 32 B record ≈ 1.6 ns/row
            launch_per_partition_ns: 2_000.0,
            // within-partition kernel splits the biggest partition over BIG_BLOCK
            // threads, so the residual serial term is ~gpu_per_row / 256.
            gpu_serial_per_row_ns: 0.2 / 256.0,
            ckpt_load_ns: 500.0,
            gpu_fixed_ns: 50_000.0, // kernel launch + setup floor
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Prediction {
    pub engine: Engine,
    pub cpu_ns: f64,
    pub gpu_ns: f64,
}

impl RouterCoeffs {
    pub fn predict_cpu(&self, rows: u64, checkpoints: u64) -> f64 {
        rows as f64 * self.cpu_per_row_ns + checkpoints as f64 * self.ckpt_load_ns
    }

    pub fn predict_gpu(&self, rows: u64, fanout: u64, max_part: u64, checkpoints: u64) -> f64 {
        self.gpu_fixed_ns
            + rows as f64 * (self.h2d_per_row_ns + self.gpu_per_row_ns)
            + fanout as f64 * self.launch_per_partition_ns
            + max_part as f64 * self.gpu_serial_per_row_ns
            + checkpoints as f64 * self.ckpt_load_ns
    }

    pub fn route(&self, rows: u64, fanout: u64, max_part: u64, checkpoints: u64) -> Prediction {
        let cpu_ns = self.predict_cpu(rows, checkpoints);
        let gpu_ns = self.predict_gpu(rows, fanout, max_part, checkpoints);
        Prediction {
            engine: if gpu_ns < cpu_ns { Engine::Gpu } else { Engine::Cpu },
            cpu_ns,
            gpu_ns,
        }
    }
}

/// One observed query: signals + measured engine times (GPU optional locally).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Observation {
    pub rows: u64,
    pub fanout: u64,
    pub max_part: u64,
    pub checkpoints: u64,
    pub cpu_ns: f64,
    pub gpu_ns: Option<f64>,
    /// Measured H2D and kernel times for this query's GPU run, when the GPU arm
    /// ran it. These let [`fit`] calibrate the transfer and compute per-row
    /// rates directly instead of trusting the priors.
    pub gpu_h2d_ns: Option<f64>,
    pub gpu_kernel_ns: Option<f64>,
}

/// Least-squares slope through the origin: Σ(x·y)/Σx². Returns `None` if there
/// is no signal (Σx² == 0).
fn slope_through_origin<'a>(pts: impl Iterator<Item = (f64, f64)>) -> Option<f64> {
    let (mut sxy, mut sxx) = (0.0f64, 0.0f64);
    for (x, y) in pts {
        sxy += x * y;
        sxx += x * x;
    }
    if sxx > 0.0 {
        Some(sxy / sxx)
    } else {
        None
    }
}

/// Fit coefficients from observations. CPU per-row from regressing cpu_ns on
/// rows. When the GPU arm measured a query, we get its H2D and kernel times
/// *separately*, so `h2d_per_row` and `gpu_per_row` are fit directly from the
/// measured breakdown — no more guessing the transfer/compute split from an
/// end-to-end number. Terms with no signal keep their prior.
pub fn fit(samples: &[Observation]) -> RouterCoeffs {
    let mut c = RouterCoeffs::default();

    if let Some(m) = slope_through_origin(samples.iter().map(|s| (s.rows as f64, s.cpu_ns))) {
        c.cpu_per_row_ns = m.max(1e-3);
    }

    // H2D transfer rate from measured per-query H2D times (Decision 3: this is
    // the binding term, so calibrate it from the wire, not a prior).
    if let Some(m) = slope_through_origin(
        samples
            .iter()
            .filter_map(|s| s.gpu_h2d_ns.map(|y| (s.rows as f64, y))),
    ) {
        c.h2d_per_row_ns = m.max(1e-3);
    }
    // Kernel compute rate from measured per-query kernel times.
    if let Some(m) = slope_through_origin(
        samples
            .iter()
            .filter_map(|s| s.gpu_kernel_ns.map(|y| (s.rows as f64, y))),
    ) {
        c.gpu_per_row_ns = m.max(1e-4);
    }
    c
}

/// Append a predicted-vs-actual record for self-correction (Decision 9).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PredVsActual {
    pub rows: u64,
    pub fanout: u64,
    pub max_part: u64,
    pub checkpoints: u64,
    pub chosen: Engine,
    pub predicted_ns: f64,
    pub actual_ns: f64,
}

pub fn log_pred_vs_actual(path: &Path, rec: &PredVsActual) -> Result<()> {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{}", serde_json::to_string(rec)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_query_routes_cpu_huge_routes_gpu() {
        let c = RouterCoeffs::default();
        // tiny per-client lookup: 50 rows, 1 partition, 1 checkpoint
        assert_eq!(c.route(50, 1, 50, 1).engine, Engine::Cpu);
        // all-history whale recompute: 50M rows over a few partitions, biggest
        // ~15M rows, no checkpoints — GPU wins now that big partitions parallelize.
        assert_eq!(c.route(50_000_000, 4, 15_000_000, 0).engine, Engine::Gpu);
    }

    #[test]
    fn fit_recovers_cpu_slope() {
        let samples: Vec<Observation> = (1..=10)
            .map(|i| Observation {
                rows: i * 1000,
                fanout: 1,
                max_part: i * 1000,
                checkpoints: 0,
                cpu_ns: (i * 1000) as f64 * 4.0, // true slope 4 ns/row
                gpu_ns: None,
                gpu_h2d_ns: None,
                gpu_kernel_ns: None,
            })
            .collect();
        let c = fit(&samples);
        assert!((c.cpu_per_row_ns - 4.0).abs() < 1e-6, "got {}", c.cpu_per_row_ns);
    }
}
