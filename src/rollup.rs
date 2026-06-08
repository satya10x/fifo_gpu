//! Cross-client rollup (Option C, M8) — incrementally-maintained per-period,
//! per-bucket realized-PnL sums.
//!
//! It's a [`FragmentSink`], so it is populated by the *same* fold that produces
//! per-client results (Decision 8: never compute PnL twice). Once built, a
//! cross-client aggregate ("intraday PnL across all clients, March 2024") is a
//! KB-sized lookup over periods, not a billion-row compute job — which is why
//! cross-client range queries route here instead of the live engine.

use crate::calendar::civil_from_days;
use crate::fifo::{BucketPnl, Bucket, Fragment, FragmentSink};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Period key = `year*12 + (month-1)` (a dense, ordered month index).
#[inline]
pub fn period_of(day: i32) -> i32 {
    let (y, m, _) = civil_from_days(day as i64);
    (y as i32) * 12 + (m as i32 - 1)
}

pub fn period_label(period: i32) -> String {
    let y = period.div_euclid(12);
    let m = period.rem_euclid(12) + 1;
    format!("{y:04}-{m:02}")
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct PeriodBuckets {
    pub intraday: BucketPnl,
    pub short: BucketPnl,
    pub long: BucketPnl,
}

impl PeriodBuckets {
    fn bucket_mut(&mut self, b: Bucket) -> &mut BucketPnl {
        match b {
            Bucket::Intraday => &mut self.intraday,
            Bucket::Short => &mut self.short,
            Bucket::Long => &mut self.long,
        }
    }
    pub fn merge(&mut self, o: &PeriodBuckets) {
        self.intraday.merge(&o.intraday);
        self.short.merge(&o.short);
        self.long.merge(&o.long);
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Rollup {
    pub by_period: BTreeMap<i32, PeriodBuckets>,
}

impl Rollup {
    pub fn new() -> Self {
        Rollup::default()
    }

    /// Sum across `period ∈ [lo, hi]` (inclusive) — the cross-client lookup.
    pub fn aggregate(&self, period_lo: i32, period_hi: i32) -> PeriodBuckets {
        let mut acc = PeriodBuckets::default();
        for (_, pb) in self.by_period.range(period_lo..=period_hi) {
            acc.merge(pb);
        }
        acc
    }

    pub fn write(&self, path: &std::path::Path) -> anyhow::Result<()> {
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn read(path: &std::path::Path) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }
}

impl FragmentSink for Rollup {
    #[inline]
    fn emit(&mut self, f: &Fragment) {
        // PnL realizes at the sell — attribute to the sell's period.
        let pb = self.by_period.entry(period_of(f.sell_day)).or_default();
        pb.bucket_mut(f.bucket).add_frag(f.realized_ticks(), f.matched_qty);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendar::days_from_civil;
    use crate::fifo::{fold_partition};
    use crate::packed::PackedTrade;

    fn rec(sq: i64, px: i64, day: i32) -> PackedTrade {
        PackedTrade { signed_qty: sq, price_ticks: px, day, _pad: 0, ts: day as i64 }
    }

    #[test]
    fn rollup_buckets_by_sell_period() {
        let d_jan = days_from_civil(2024, 1, 10) as i32;
        let d_feb = days_from_civil(2024, 2, 10) as i32;
        let recs = [
            rec(100, 1000, d_jan),
            rec(-100, 1200, d_jan), // intraday, jan
            rec(50, 1000, d_jan),
            rec(-50, 1500, d_feb), // short, sold in feb
        ];
        let mut roll = Rollup::new();
        fold_partition(1, 1, &recs, &mut roll);
        let jan = period_of(d_jan);
        let feb = period_of(d_feb);
        assert_eq!(roll.by_period[&jan].intraday.realized_ticks, 100 * 200);
        assert_eq!(roll.by_period[&feb].short.realized_ticks, 50 * 500);
        // aggregate over both months
        let agg = roll.aggregate(jan, feb);
        assert_eq!(agg.intraday.realized_ticks, 100 * 200);
        assert_eq!(agg.short.realized_ticks, 50 * 500);
    }
}
