//! Run manifest: the config + realized totals, persisted as JSON so downstream
//! stages (and `stats`) can reconstruct the exact generation parameters.

use crate::calendar::civil_from_days;
use crate::config::GenConfig;
use crate::skew::SkewPlan;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub config: GenConfig,
    pub target_total: u64,
    pub realized_total: u64,
    pub n_whales: u64,
    pub start_date: String,
    pub end_date_inclusive: String,
}

impl Manifest {
    pub fn new(cfg: &GenConfig, plan: &SkewPlan, realized_total: u64) -> Self {
        let cal = cfg.trading_days();
        let fmt = |z: i64| {
            let (y, m, d) = civil_from_days(z);
            format!("{y:04}-{m:02}-{d:02}")
        };
        Manifest {
            config: cfg.clone(),
            target_total: plan.target_total,
            realized_total,
            n_whales: plan.n_whales,
            start_date: fmt(*cal.first().unwrap_or(&cfg.start_day)),
            end_date_inclusive: fmt(*cal.last().unwrap_or(&cfg.start_day)),
        }
    }

    pub fn write(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let s = serde_json::to_string_pretty(self)?;
        std::fs::write(dir.join("manifest.json"), s)?;
        Ok(())
    }

    pub fn read(dir: &Path) -> Result<Self> {
        let p = dir.join("manifest.json");
        let s = std::fs::read_to_string(&p)
            .with_context(|| format!("reading manifest {}", p.display()))?;
        Ok(serde_json::from_str(&s)?)
    }
}
