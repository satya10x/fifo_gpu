//! Checkpoint table (M4) — versioned open-lot snapshots for bounded range
//! queries (Decision 7).
//!
//! A range query needs the FIFO carry-in *as of the start of the range*, which
//! depends on all prior trades. A checkpoint at cutoff day `C` stores, per
//! `(client, symbol)`, the open-lot queue after folding every trade with
//! `day ≤ C`. A `[lo, hi]` query then loads the nearest checkpoint with
//! `C < lo`, replays only `(C, hi]`, and counts fragments selling in `[lo, hi]`
//! — work bounded by the range, independent of history depth.
//!
//! Checkpoints are also the seam for whale segmenting (split a huge partition at
//! cutoffs with carried state) and are versioned by cutoff day on disk.

use crate::fifo::{MatchPolicy, fold_core, BucketRules, Lot, NoopSink};
use crate::packed::PackedTable;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    pub cutoff_day: i32,
    /// `(client, symbol)` → residual open lots after folding `day ≤ cutoff_day`.
    /// Only non-empty queues are stored.
    pub lots: HashMap<String, Vec<Lot>>,
}

#[inline]
fn part_key(client: u64, symbol: u32) -> String {
    format!("{client}:{symbol}")
}

impl Checkpoint {
    /// Build the checkpoint at `cutoff_day` by folding each partition over its
    /// `day ≤ cutoff_day` records and snapshotting the residual open lots.
    pub fn build(table: &PackedTable, cutoff_day: i32) -> Self {
        let pc = table.part_client();
        let ps = table.part_symbol();
        let mut lots = HashMap::new();
        for p in 0..pc.len() {
            let recs = table.partition(p);
            let in_scope = table.day_range(recs, i32::MIN, cutoff_day);
            if in_scope.is_empty() {
                continue;
            }
            let mut carry: VecDeque<Lot> = VecDeque::new();
            fold_core(pc[p], ps[p], &mut carry, in_scope, &mut NoopSink, None, &BucketRules::default(), MatchPolicy::Fifo);
            if !carry.is_empty() {
                lots.insert(part_key(pc[p], ps[p]), carry.into_iter().collect());
            }
        }
        Checkpoint { cutoff_day, lots }
    }

    /// Carry-in queue for a partition (empty if none stored).
    pub fn carry_for(&self, client: u64, symbol: u32) -> VecDeque<Lot> {
        self.lots
            .get(&part_key(client, symbol))
            .map(|v| v.iter().copied().collect())
            .unwrap_or_default()
    }

    pub fn write(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("ckpt-{:08}.json", self.cutoff_day));
        std::fs::write(path, serde_json::to_string(self)?)?;
        Ok(())
    }
}

/// Manages a directory of periodic checkpoints, versioned by cutoff day.
pub struct CheckpointStore {
    pub dir: PathBuf,
}

impl CheckpointStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        CheckpointStore { dir: dir.into() }
    }

    /// Available cutoff days, ascending.
    pub fn cutoffs(&self) -> Result<Vec<i32>> {
        if !self.dir.exists() {
            return Ok(vec![]);
        }
        let mut v: Vec<i32> = std::fs::read_dir(&self.dir)?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let n = e.file_name().into_string().ok()?;
                let s = n.strip_prefix("ckpt-")?.strip_suffix(".json")?;
                s.trim_start_matches('0').parse().ok().or(Some(0))
            })
            .collect();
        v.sort_unstable();
        Ok(v)
    }

    /// Build checkpoints at each cutoff day and persist them.
    ///
    /// First clears any existing `ckpt-*.json` in the directory: checkpoints are
    /// derived from a specific packed dataset, so leftovers from a *prior* dataset
    /// (e.g. after a regenerate, which shifts the cutoff days) would otherwise
    /// survive and let [`load_nearest_before`] return carry-in from stale data.
    pub fn build_periodic(&self, table: &PackedTable, cutoffs: &[i32]) -> Result<()> {
        self.clear()?;
        for &c in cutoffs {
            Checkpoint::build(table, c).write(&self.dir)?;
        }
        Ok(())
    }

    /// Remove all `ckpt-*.json` checkpoint files in the directory (leaves any
    /// other files untouched).
    pub fn clear(&self) -> Result<()> {
        if !self.dir.exists() {
            return Ok(());
        }
        for entry in std::fs::read_dir(&self.dir)? {
            let path = entry?.path();
            let is_ckpt = path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.starts_with("ckpt-") && s.ends_with(".json"))
                .unwrap_or(false);
            if is_ckpt {
                std::fs::remove_file(&path)?;
            }
        }
        Ok(())
    }

    /// Load the nearest checkpoint with `cutoff_day < before_day` (i.e. valid
    /// carry-in for a range starting at `before_day`). None if no such checkpoint.
    pub fn load_nearest_before(&self, before_day: i32) -> Result<Option<Checkpoint>> {
        let cutoffs = self.cutoffs()?;
        let Some(&c) = cutoffs.iter().rev().find(|&&c| c < before_day) else {
            return Ok(None);
        };
        let path = self.dir.join(format!("ckpt-{:08}.json", c));
        let s = std::fs::read_to_string(&path)
            .with_context(|| format!("reading checkpoint {}", path.display()))?;
        Ok(Some(serde_json::from_str(&s)?))
    }

    /// Cutoffs invalidated by a correction at `correction_day` — every
    /// checkpoint with `cutoff_day ≥ correction_day` must be rebuilt (M8).
    pub fn invalidated_by(&self, correction_day: i32) -> Result<Vec<i32>> {
        Ok(self
            .cutoffs()?
            .into_iter()
            .filter(|&c| c >= correction_day)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fifo::{MatchPolicy, fold_partition, NoopSink};
    use crate::packed::{PackedBuilder, PackedTrade};

    fn rec(sq: i32, px: i32, day: i32) -> PackedTrade {
        PackedTrade { signed_qty: sq, price_ticks: px, day }
    }

    #[test]
    fn range_query_via_checkpoint_matches_truth() {
        // Partition: buy d10, buy d20, sell (crossing) d200, buy d300, sell d400.
        let recs = [
            rec(100, 1000, 10),
            rec(100, 1500, 20),
            rec(-150, 3000, 200),
            rec(50, 2000, 300),
            rec(-50, 2500, 400),
        ];
        let mut b = PackedBuilder::new();
        b.push_partition(1, 1, &recs);
        let path = std::env::temp_dir().join("fifo_ckpt_test.fifopack");
        b.write(&path).unwrap();
        let t = PackedTable::open(&path).unwrap();

        // Truth for the range [150, 250]: only the d200 sell should count.
        let truth = {
            // fold whole thing but only count sells in [150,250]
            let mut carry = VecDeque::new();
            fold_core(1, 1, &mut carry, &recs, &mut NoopSink, Some((150, 250)), &BucketRules::default(), MatchPolicy::Fifo)
        };

        // Via checkpoint at cutoff 100 (state after d10,d20 buys), replay (100, 250].
        let ckpt = Checkpoint::build(&t, 100);
        let mut carry = ckpt.carry_for(1, 1);
        let part = t.partition(0);
        let replay = t.day_range(part, 101, 250);
        let got = fold_core(1, 1, &mut carry, replay, &mut NoopSink, Some((150, 250)), &BucketRules::default(), MatchPolicy::Fifo);

        assert_eq!(got, truth);
        assert_eq!(got.short.matched_qty, 150); // the d200 sell crossed both lots
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn build_periodic_clears_stale_checkpoints() {
        let recs = [rec(100, 1000, 10), rec(-40, 1500, 20)];
        let mut b = PackedBuilder::new();
        b.push_partition(1, 1, &recs);
        let path = std::env::temp_dir().join("fifo_ckpt_stale.fifopack");
        b.write(&path).unwrap();
        let t = PackedTable::open(&path).unwrap();

        let dir = std::env::temp_dir().join("fifo_ckpt_stale_dir");
        let _ = std::fs::remove_dir_all(&dir);
        let store = CheckpointStore::new(&dir);

        // First dataset/cutoffs.
        store.build_periodic(&t, &[15, 25]).unwrap();
        assert_eq!(store.cutoffs().unwrap(), vec![15, 25]);

        // A regenerate shifts the cutoffs — the old [15,25] must NOT linger.
        store.build_periodic(&t, &[30, 40]).unwrap();
        assert_eq!(store.cutoffs().unwrap(), vec![30, 40]);

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn checkpoint_carry_is_correct() {
        let recs = [rec(100, 1000, 10), rec(40, 1500, 20), rec(-60, 3000, 30)];
        let mut b = PackedBuilder::new();
        b.push_partition(5, 9, &recs);
        let path = std::env::temp_dir().join("fifo_ckpt_test2.fifopack");
        b.write(&path).unwrap();
        let t = PackedTable::open(&path).unwrap();
        // checkpoint at day 25: buys present, sell (d30) not yet → 100@1000 + 40@1500
        let ckpt = Checkpoint::build(&t, 25);
        let carry = ckpt.carry_for(5, 9);
        assert_eq!(carry.len(), 2);
        assert_eq!(carry[0].qty, 100);
        assert_eq!(carry[1].qty, 40);
        // checkpoint at day 35: after the 60-share sell, 40@1000 + 40@1500 remain
        let ckpt2 = Checkpoint::build(&t, 35);
        let carry2 = ckpt2.carry_for(5, 9);
        assert_eq!(carry2[0].qty, 40);
        assert_eq!(carry2[1].qty, 40);
        let _ = fold_partition(5, 9, &recs, &mut NoopSink);
        let _ = std::fs::remove_file(&path);
    }
}
