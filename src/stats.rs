//! Distribution verifier — the gate on benchmark validity (§5: uniform data
//! gives a wrong crossover number). Recomputes the trade-count-per-client
//! distribution deterministically (no need to read the data) and reports the
//! skew shape; optionally cross-checks the realized Parquet row count.

use crate::config::GenConfig;
use crate::skew;
use anyhow::Result;
use std::path::Path;

pub struct SkewSummary {
    pub clients: u64,
    pub total: u128,
    pub days: u32,
    pub trades_per_day: f64,
    pub mean: f64,
    pub percentiles: Vec<(f64, u64)>, // (q, value)
    pub max: u64,
    pub n_whales: u64,
    pub whale_volume_share: f64,
    pub top_share: Vec<(f64, f64)>, // (top fraction, volume share)
    pub hist: Vec<(u64, u64)>,      // (bucket lower bound, client count)
}

/// Compute the full per-client count distribution and summarize it.
pub fn summarize(cfg: &GenConfig) -> SkewSummary {
    let plan = skew::build_plan(cfg);
    let mut counts: Vec<u64> = Vec::with_capacity(cfg.clients as usize);
    let mut whale_vol: u128 = 0;
    for id in 0..cfg.clients {
        let c = skew::client_count(cfg, &plan, id);
        if cfg.is_whale(id) {
            whale_vol += c as u128;
        }
        counts.push(c);
    }
    counts.sort_unstable();
    let total: u128 = counts.iter().map(|&c| c as u128).sum();
    let n = counts.len();

    let pct = |q: f64| counts[((q * (n as f64 - 1.0)).round() as usize).min(n - 1)];
    let percentiles = vec![
        (0.50, pct(0.50)),
        (0.90, pct(0.90)),
        (0.99, pct(0.99)),
        (0.999, pct(0.999)),
        (0.9999, pct(0.9999)),
    ];

    // Top-k volume shares (counts is ascending, so take from the tail).
    let top_share = [0.0001f64, 0.001, 0.01, 0.1]
        .iter()
        .map(|&f| {
            let k = ((f * n as f64).ceil() as usize).max(1).min(n);
            let s: u128 = counts[n - k..].iter().map(|&c| c as u128).sum();
            (f, s as f64 / total as f64)
        })
        .collect();

    // Log2 histogram of per-client counts.
    let mut hist: Vec<(u64, u64)> = Vec::new();
    let mut bound = 1u64;
    loop {
        let lo = bound;
        let hi = bound.saturating_mul(2);
        let c = counts.iter().filter(|&&v| v >= lo && v < hi).count() as u64;
        hist.push((lo, c));
        if hi > *counts.last().unwrap() {
            break;
        }
        bound = hi;
    }

    SkewSummary {
        clients: cfg.clients,
        total,
        days: cfg.days,
        trades_per_day: total as f64 / cfg.days as f64,
        mean: total as f64 / n as f64,
        percentiles,
        max: *counts.last().unwrap(),
        n_whales: plan.n_whales,
        whale_volume_share: whale_vol as f64 / total as f64,
        top_share,
        hist,
    }
}

impl SkewSummary {
    pub fn print(&self) {
        println!("── trade-count-per-client skew ──────────────────────────────");
        println!("clients            : {}", self.clients);
        println!("total trades       : {}", self.total);
        println!("trading days       : {}", self.days);
        println!("trades / day       : {:.0}", self.trades_per_day);
        println!("mean / client      : {:.2}", self.mean);
        for (q, v) in &self.percentiles {
            println!("p{:<16}: {}", format!("{:.2}%", q * 100.0), v);
        }
        println!("max (top whale)    : {}", self.max);
        println!("whales             : {}", self.n_whales);
        println!(
            "whale vol share    : {:.1}%",
            self.whale_volume_share * 100.0
        );
        println!("── top-k volume concentration ───────────────────────────────");
        for (f, s) in &self.top_share {
            println!("top {:<8}: {:.1}% of all trades", format!("{:.2}%", f * 100.0), s * 100.0);
        }
        println!("── log2 histogram (count range → #clients) ──────────────────");
        for (lo, c) in &self.hist {
            if *c == 0 {
                continue;
            }
            let bar = "█".repeat(((*c as f64).log10().max(0.0) * 6.0) as usize);
            println!("[{:>10}, {:>10}) {:>10}  {}", lo, lo * 2, c, bar);
        }
    }
}

/// Sum `num_rows` across all `part-*.parquet` files (metadata only).
pub fn count_parquet_rows(dir: &Path) -> Result<u64> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    let mut total = 0u64;
    let mut entries: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.starts_with("part-") && s.ends_with(".parquet"))
                .unwrap_or(false)
        })
        .collect();
    entries.sort();
    for p in entries {
        let f = std::fs::File::open(&p)?;
        let reader = SerializedFileReader::new(f)?;
        total += reader.metadata().file_metadata().num_rows() as u64;
    }
    Ok(total)
}
