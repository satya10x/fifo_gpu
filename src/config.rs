//! Generator configuration. Every field is captured into the manifest so a run
//! is fully reproducible from `(seed, config)`.

use crate::calendar;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenConfig {
    pub seed: u64,
    /// Number of distinct client accounts.
    pub clients: u64,
    /// First calendar day (days since Unix epoch). Weekends are skipped.
    pub start_day: i64,
    /// Number of *trading* days to span.
    pub days: u32,
    /// Symbol-universe size.
    pub symbols: u32,
    /// Target overall mean trades / client / day (sets the total trade count).
    pub mean_per_day: f64,
    /// Number of whale accounts (evenly scattered across the id space).
    pub n_whales: u64,

    // ---- body (long-tail retail) trade-count distribution: lognormal over the
    // whole period, chosen so the median client is single-digit. ----
    pub body_mu: f64,
    pub body_sigma: f64,
    /// Pareto shape for relative whale sizes (smaller α = a few mega-whales).
    pub whale_pareto_alpha: f64,

    // ---- per-trade field knobs ----
    pub qty_mu: f64,
    pub qty_sigma: f64,
    /// Probability the next trade lands on the *same* day (creates intraday round trips).
    pub p_intraday: f64,

    // ---- output ----
    pub out_dir: PathBuf,
    pub rows_per_file: usize,
    pub batch_rows: usize,
}

impl GenConfig {
    /// Total trades to emit across all clients (mean × clients × days).
    pub fn target_total(&self) -> u64 {
        (self.clients as f64 * self.mean_per_day * self.days as f64).round() as u64
    }

    /// Whale ids are scattered every `stride` positions so heavy partitions are
    /// spread across the dataset (good for testing page-range pruning), not
    /// bunched at the front.
    #[inline]
    pub fn whale_stride(&self) -> u64 {
        (self.clients / self.n_whales.max(1)).max(1)
    }

    #[inline]
    pub fn is_whale(&self, id: u64) -> bool {
        let s = self.whale_stride();
        id % s == 0 && id / s < self.n_whales
    }

    /// Index of a whale among whales (used to seed its relative weight).
    #[inline]
    pub fn whale_index(&self, id: u64) -> u64 {
        id / self.whale_stride()
    }

    pub fn trading_days(&self) -> Vec<i64> {
        calendar::trading_days(self.start_day, self.days)
    }

    pub fn defaults() -> Self {
        GenConfig {
            seed: 42,
            clients: 5_000,
            start_day: calendar::days_from_civil(2020, 1, 1),
            days: 250,
            symbols: 1_000,
            mean_per_day: 6.0,
            n_whales: 8,
            // median body client ≈ exp(1.1) ≈ 3 trades over the whole period.
            body_mu: 1.1,
            body_sigma: 1.3,
            whale_pareto_alpha: 1.5,
            // median order ≈ exp(4.6) ≈ 100 shares.
            qty_mu: 4.6,
            qty_sigma: 1.0,
            p_intraday: 0.35,
            out_dir: PathBuf::from("data/tradebook"),
            rows_per_file: 8_000_000,
            batch_rows: 1_000_000,
        }
    }
}
