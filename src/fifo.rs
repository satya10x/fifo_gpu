//! CPU FIFO fold — the correctness oracle (M3) *and* benchmark arm 2.
//!
//! Per `(client, symbol)` partition, drain each sell against the oldest open
//! buy lots (strict FIFO). Each matched fragment yields realized PnL and a
//! bucket tag:
//! - **intraday**  — buy and sell share a calendar day,
//! - **short**     — span ≤ 365 days,
//! - **long**      — span  > 365 days.
//!
//! PnL is accumulated in integer **tick·share** units (`i128`, overflow-proof
//! for whale volumes); multiply by [`TICK`] for a currency value.
//!
//! One fold feeds two consumers (Decision 8): it always accumulates the
//! per-partition [`PartitionPnl`] *and* streams every [`Fragment`] to a
//! [`FragmentSink`] (e.g. the cross-client rollup) — PnL is never computed twice.

use crate::generate::TICK;
use crate::packed::{PackedTrade, PackedTable};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

pub const LONG_TERM_DAYS: i32 = 365;

/// Max buckets a ruleset can define. 8 covers any realistic holding-period tax
/// regime (intraday + several bands).
pub const MAX_BUCKETS: usize = 8;

/// Data-driven bucket rules (Axis 1 of the generic engine — see DESIGN.md).
/// Classifies a matched fragment into one of up to [`MAX_BUCKETS`] buckets by
/// holding-period band: `intraday_same_day` puts same-day round-trips in bucket
/// 0; `boundaries` (ascending; first `n_bounds` used) are inclusive holding-day
/// upper bounds for each following band; anything past the last lands in the
/// final (e.g. long-term) bucket. The default reproduces intraday/short≤365/long.
#[derive(Clone, Copy, Debug)]
pub struct BucketRules {
    pub intraday_same_day: bool,
    pub boundaries: [i32; MAX_BUCKETS],
    pub n_bounds: usize,
}

impl Default for BucketRules {
    fn default() -> Self {
        BucketRules::equity(LONG_TERM_DAYS)
    }
}

impl BucketRules {
    /// intraday + one short/long threshold (the common equity case).
    pub fn equity(short_max_days: i32) -> Self {
        let mut boundaries = [0; MAX_BUCKETS];
        boundaries[0] = short_max_days;
        BucketRules { intraday_same_day: true, boundaries, n_bounds: 1 }
    }
    /// intraday + arbitrary ascending holding-day bands (e.g. multi-tier regimes).
    pub fn bands(intraday_same_day: bool, bounds: &[i32]) -> Self {
        let mut boundaries = [0; MAX_BUCKETS];
        let n = bounds.len().min(MAX_BUCKETS);
        boundaries[..n].copy_from_slice(&bounds[..n]);
        BucketRules { intraday_same_day, boundaries, n_bounds: n }
    }
    /// Number of buckets this ruleset produces.
    pub fn num_buckets(&self) -> usize {
        usize::from(self.intraday_same_day) + self.n_bounds + 1
    }
    #[inline]
    pub fn is_default_equity(&self) -> bool {
        self.intraday_same_day && self.n_bounds == 1 && self.boundaries[0] == LONG_TERM_DAYS
    }
    /// Human label for bucket `i` (e.g. "intraday", "0-365d", ">365d").
    pub fn label(&self, i: usize) -> String {
        let mut idx = i;
        if self.intraday_same_day {
            if idx == 0 {
                return "intraday".into();
            }
            idx -= 1;
        }
        if idx < self.n_bounds {
            let lo = if idx == 0 { if self.intraday_same_day { 1 } else { 0 } } else { self.boundaries[idx - 1] + 1 };
            format!("{}-{}d", lo, self.boundaries[idx])
        } else {
            format!(">{}d", self.boundaries[self.n_bounds - 1])
        }
    }
}

/// Lot-matching policy (Axis 2 of the generic engine — see DESIGN.md): which
/// open buy lot a sell consumes. The carry is kept in time order; the policy
/// only changes *selection*, so the same fold serves all of these.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MatchPolicy {
    /// Oldest open lot first (queue front). The classic, and the only policy the
    /// GPU scan kernel implements.
    #[default]
    Fifo,
    /// Newest open lot first (queue back).
    Lifo,
    /// Highest-cost lot first (tax-loss harvesting). O(n) lot selection here —
    /// fine as a correctness reference; a price-max heap is the optimization.
    Hifo,
}

/// Index of the open lot a sell drains next under `policy`. `carry` is non-empty.
#[inline]
fn pick_lot(carry: &VecDeque<Lot>, policy: MatchPolicy) -> usize {
    match policy {
        MatchPolicy::Fifo => 0,
        MatchPolicy::Lifo => carry.len() - 1,
        MatchPolicy::Hifo => {
            let mut best = 0;
            for i in 1..carry.len() {
                if carry[i].price_ticks > carry[best].price_ticks {
                    best = i;
                }
            }
            best
        }
    }
}

/// Bucket index (0..`num_buckets`) for a matched fragment under `rules`.
#[inline]
pub fn classify(rules: &BucketRules, buy_day: i32, sell_day: i32) -> usize {
    let span = sell_day - buy_day;
    let base = if rules.intraday_same_day {
        if span == 0 {
            return 0;
        }
        1
    } else {
        0
    };
    for i in 0..rules.n_bounds {
        if span <= rules.boundaries[i] {
            return base + i;
        }
    }
    base + rules.n_bounds
}

/// An open buy lot awaiting a matching sell.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Lot {
    pub qty: i64,
    pub price_ticks: i64,
    pub day: i32,
}

/// A matched buy⇄sell fragment.
#[derive(Clone, Copy, Debug)]
pub struct Fragment {
    pub client: u64,
    pub symbol: u32,
    pub buy_day: i32,
    pub sell_day: i32,
    pub matched_qty: i64,
    pub buy_ticks: i64,
    pub sell_ticks: i64,
    /// Bucket index (0..`num_buckets`) under the active [`BucketRules`].
    pub bucket: usize,
}

impl Fragment {
    /// Realized PnL in tick·share units.
    #[inline]
    pub fn realized_ticks(&self) -> i128 {
        self.matched_qty as i128 * (self.sell_ticks - self.buy_ticks) as i128
    }
}

/// Receives every matched fragment as the fold runs.
pub trait FragmentSink {
    fn emit(&mut self, f: &Fragment);
}

/// A sink that discards fragments (when only the partition total is wanted).
pub struct NoopSink;
impl FragmentSink for NoopSink {
    #[inline]
    fn emit(&mut self, _f: &Fragment) {}
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BucketPnl {
    pub realized_ticks: i128,
    pub matched_qty: i128,
    pub fragments: u64,
}

impl BucketPnl {
    #[inline]
    pub fn add_frag(&mut self, realized_ticks: i128, qty: i64) {
        self.realized_ticks += realized_ticks;
        self.matched_qty += qty as i128;
        self.fragments += 1;
    }
    /// Currency value of realized PnL.
    pub fn value(&self) -> f64 {
        self.realized_ticks as f64 * TICK
    }
    pub fn merge(&mut self, o: &BucketPnl) {
        self.realized_ticks += o.realized_ticks;
        self.matched_qty += o.matched_qty;
        self.fragments += o.fragments;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartitionPnl {
    /// One [`BucketPnl`] per bucket (0..`num_buckets`); unused tail stays zero.
    pub buckets: [BucketPnl; MAX_BUCKETS],
}

impl Default for PartitionPnl {
    fn default() -> Self {
        PartitionPnl { buckets: [BucketPnl::default(); MAX_BUCKETS] }
    }
}

impl PartitionPnl {
    #[inline]
    pub fn bucket_mut(&mut self, i: usize) -> &mut BucketPnl {
        &mut self.buckets[i]
    }
    #[inline]
    pub fn merge_default3(&mut self, o: &PartitionPnl) {
        self.buckets[0].merge(&o.buckets[0]);
        self.buckets[1].merge(&o.buckets[1]);
        self.buckets[2].merge(&o.buckets[2]);
    }
    pub fn merge(&mut self, o: &PartitionPnl) {
        for i in 0..MAX_BUCKETS {
            self.buckets[i].merge(&o.buckets[i]);
        }
    }
    pub fn total_ticks(&self) -> i128 {
        self.buckets.iter().map(|b| b.realized_ticks).sum()
    }
    // Convenience accessors for the default ruleset (intraday/short/long = 0/1/2).
    pub fn intraday(&self) -> &BucketPnl {
        &self.buckets[0]
    }
    pub fn short(&self) -> &BucketPnl {
        &self.buckets[1]
    }
    pub fn long(&self) -> &BucketPnl {
        &self.buckets[2]
    }
}

/// Core fold. The open-lot queue (`carry`) is mutated in place — on return it
/// holds the residual open lots, exactly what a checkpoint snapshots (M4).
///
/// `count` optionally restricts which fragments are *counted/emitted* by sell
/// day (`[lo, hi]` inclusive). Lots are always consumed regardless, so replaying
/// pre-range sells correctly advances state without polluting the range result —
/// the mechanism behind nearest-checkpoint range queries.
pub fn fold_core<S: FragmentSink>(
    client: u64,
    symbol: u32,
    carry: &mut VecDeque<Lot>,
    recs: &[PackedTrade],
    sink: &mut S,
    count: Option<(i32, i32)>,
    rules: &BucketRules,
    policy: MatchPolicy,
) -> PartitionPnl {
    if policy == MatchPolicy::Fifo {
        return fold_core_fifo(client, symbol, carry, recs, sink, count, rules);
    }

    let mut pnl = PartitionPnl::default();
    for r in recs {
        if r.signed_qty > 0 {
            carry.push_back(Lot {
                qty: r.signed_qty as i64,
                price_ticks: r.price_ticks as i64,
                day: r.day,
            });
        } else if r.signed_qty < 0 {
            let mut remaining = -(r.signed_qty as i64);
            let sell_ticks = r.price_ticks as i64;
            let sell_day = r.day;
            let counted = match count {
                Some((lo, hi)) => sell_day >= lo && sell_day <= hi,
                None => true,
            };
            while remaining > 0 {
                if carry.is_empty() {
                    // No open lot to match (generator never shorts; defensive).
                    break;
                }
                let idx = pick_lot(carry, policy);
                // mutate the chosen lot in a scoped borrow, copying what the
                // fragment needs, so we can `carry.remove(idx)` after.
                let (matched, buy_day, buy_ticks, drained) = {
                    let lot = &mut carry[idx];
                    let matched = remaining.min(lot.qty);
                    lot.qty -= matched;
                    (matched, lot.day, lot.price_ticks, lot.qty == 0)
                };
                if counted {
                    let bucket = classify(rules, buy_day, sell_day);
                    let frag = Fragment {
                        client,
                        symbol,
                        buy_day,
                        sell_day,
                        matched_qty: matched,
                        buy_ticks,
                        sell_ticks,
                        bucket,
                    };
                    pnl.bucket_mut(bucket).add_frag(frag.realized_ticks(), matched);
                    sink.emit(&frag);
                }
                remaining -= matched;
                if drained {
                    carry.remove(idx);
                }
            }
        }
    }
    pnl
}

fn fold_core_fifo<S: FragmentSink>(
    client: u64,
    symbol: u32,
    carry: &mut VecDeque<Lot>,
    recs: &[PackedTrade],
    sink: &mut S,
    count: Option<(i32, i32)>,
    rules: &BucketRules,
) -> PartitionPnl {
    let mut pnl = PartitionPnl::default();
    for r in recs {
        if r.signed_qty > 0 {
            carry.push_back(Lot {
                qty: r.signed_qty as i64,
                price_ticks: r.price_ticks as i64,
                day: r.day,
            });
            continue;
        }
        if r.signed_qty >= 0 {
            continue;
        }

        let mut remaining = -(r.signed_qty as i64);
        let sell_ticks = r.price_ticks as i64;
        let sell_day = r.day;
        let counted = match count {
            Some((lo, hi)) => sell_day >= lo && sell_day <= hi,
            None => true,
        };
        while remaining > 0 {
            let Some(lot) = carry.front_mut() else {
                break;
            };
            let matched = remaining.min(lot.qty);
            let buy_day = lot.day;
            let buy_ticks = lot.price_ticks;
            lot.qty -= matched;
            if counted {
                let bucket = classify(rules, buy_day, sell_day);
                let frag = Fragment {
                    client,
                    symbol,
                    buy_day,
                    sell_day,
                    matched_qty: matched,
                    buy_ticks,
                    sell_ticks,
                    bucket,
                };
                pnl.bucket_mut(bucket).add_frag(frag.realized_ticks(), matched);
                sink.emit(&frag);
            }
            remaining -= matched;
            if lot.qty == 0 {
                carry.pop_front();
            }
        }
    }
    pnl
}

pub fn fold_core_nosink(
    carry: &mut VecDeque<Lot>,
    recs: &[PackedTrade],
    count: Option<(i32, i32)>,
    rules: &BucketRules,
    policy: MatchPolicy,
) -> PartitionPnl {
    if policy == MatchPolicy::Fifo {
        if is_default_equity_rules(rules) {
            return match count {
                None => fold_core_fifo_default_full_nosink(carry, recs),
                Some(_) => fold_core_fifo_default_nosink(carry, recs, count),
            };
        }
        return fold_core_fifo_nosink(carry, recs, count, rules);
    }

    let mut pnl = PartitionPnl::default();
    for r in recs {
        if r.signed_qty > 0 {
            carry.push_back(Lot {
                qty: r.signed_qty as i64,
                price_ticks: r.price_ticks as i64,
                day: r.day,
            });
        } else if r.signed_qty < 0 {
            let mut remaining = -(r.signed_qty as i64);
            let sell_ticks = r.price_ticks as i64;
            let sell_day = r.day;
            let counted = match count {
                Some((lo, hi)) => sell_day >= lo && sell_day <= hi,
                None => true,
            };
            while remaining > 0 {
                if carry.is_empty() {
                    break;
                }
                let idx = pick_lot(carry, policy);
                let (matched, buy_day, buy_ticks, drained) = {
                    let lot = &mut carry[idx];
                    let matched = remaining.min(lot.qty);
                    lot.qty -= matched;
                    (matched, lot.day, lot.price_ticks, lot.qty == 0)
                };
                if counted {
                    let bucket = classify(rules, buy_day, sell_day);
                    let realized = matched as i128 * (sell_ticks - buy_ticks) as i128;
                    pnl.bucket_mut(bucket).add_frag(realized, matched);
                }
                remaining -= matched;
                if drained {
                    carry.remove(idx);
                }
            }
        }
    }
    pnl
}

#[inline]
fn is_default_equity_rules(rules: &BucketRules) -> bool {
    rules.is_default_equity()
}

fn fold_core_fifo_default_full_nosink(
    carry: &mut VecDeque<Lot>,
    recs: &[PackedTrade],
) -> PartitionPnl {
    let mut pnl = PartitionPnl::default();
    let mut lots = Vec::with_capacity(carry.len() + recs.len());
    lots.extend(carry.drain(..));
    let mut head = 0usize;

    let mut realized0 = 0i128;
    let mut realized1 = 0i128;
    let mut realized2 = 0i128;
    let mut qty0 = 0i128;
    let mut qty1 = 0i128;
    let mut qty2 = 0i128;
    let mut frags0 = 0u64;
    let mut frags1 = 0u64;
    let mut frags2 = 0u64;

    for r in recs {
        if r.signed_qty > 0 {
            lots.push(Lot {
                qty: r.signed_qty as i64,
                price_ticks: r.price_ticks as i64,
                day: r.day,
            });
            continue;
        }
        if r.signed_qty >= 0 {
            continue;
        }

        let mut remaining = -(r.signed_qty as i64);
        let sell_ticks = r.price_ticks as i64;
        let sell_day = r.day;
        while remaining > 0 && head < lots.len() {
            let lot = &mut lots[head];
            let matched = remaining.min(lot.qty);
            let buy_day = lot.day;
            let buy_ticks = lot.price_ticks;
            lot.qty -= matched;
            let realized = matched as i128 * (sell_ticks - buy_ticks) as i128;
            let span = sell_day - buy_day;
            if span == 0 {
                realized0 += realized;
                qty0 += matched as i128;
                frags0 += 1;
            } else if span <= LONG_TERM_DAYS {
                realized1 += realized;
                qty1 += matched as i128;
                frags1 += 1;
            } else {
                realized2 += realized;
                qty2 += matched as i128;
                frags2 += 1;
            }
            remaining -= matched;
            if lot.qty == 0 {
                head += 1;
            }
        }
    }

    if head < lots.len() {
        carry.extend(lots.into_iter().skip(head));
    }
    pnl.buckets[0] = BucketPnl { realized_ticks: realized0, matched_qty: qty0, fragments: frags0 };
    pnl.buckets[1] = BucketPnl { realized_ticks: realized1, matched_qty: qty1, fragments: frags1 };
    pnl.buckets[2] = BucketPnl { realized_ticks: realized2, matched_qty: qty2, fragments: frags2 };
    pnl
}

fn fold_core_fifo_default_nosink(
    carry: &mut VecDeque<Lot>,
    recs: &[PackedTrade],
    count: Option<(i32, i32)>,
) -> PartitionPnl {
    let mut pnl = PartitionPnl::default();
    let mut lots = Vec::with_capacity(carry.len() + recs.len());
    lots.extend(carry.drain(..));
    let mut head = 0usize;

    let mut realized0 = 0i128;
    let mut realized1 = 0i128;
    let mut realized2 = 0i128;
    let mut qty0 = 0i128;
    let mut qty1 = 0i128;
    let mut qty2 = 0i128;
    let mut frags0 = 0u64;
    let mut frags1 = 0u64;
    let mut frags2 = 0u64;

    for r in recs {
        if r.signed_qty > 0 {
            lots.push(Lot {
                qty: r.signed_qty as i64,
                price_ticks: r.price_ticks as i64,
                day: r.day,
            });
            continue;
        }
        if r.signed_qty >= 0 {
            continue;
        }

        let mut remaining = -(r.signed_qty as i64);
        let sell_ticks = r.price_ticks as i64;
        let sell_day = r.day;
        let counted = match count {
            Some((lo, hi)) => sell_day >= lo && sell_day <= hi,
            None => true,
        };
        while remaining > 0 && head < lots.len() {
            let lot = &mut lots[head];
            let matched = remaining.min(lot.qty);
            let buy_day = lot.day;
            let buy_ticks = lot.price_ticks;
            lot.qty -= matched;
            if counted {
                let realized = matched as i128 * (sell_ticks - buy_ticks) as i128;
                let span = sell_day - buy_day;
                if span == 0 {
                    realized0 += realized;
                    qty0 += matched as i128;
                    frags0 += 1;
                } else if span <= LONG_TERM_DAYS {
                    realized1 += realized;
                    qty1 += matched as i128;
                    frags1 += 1;
                } else {
                    realized2 += realized;
                    qty2 += matched as i128;
                    frags2 += 1;
                }
            }
            remaining -= matched;
            if lot.qty == 0 {
                head += 1;
            }
        }
    }

    if head < lots.len() {
        carry.extend(lots.into_iter().skip(head));
    }
    pnl.buckets[0] = BucketPnl { realized_ticks: realized0, matched_qty: qty0, fragments: frags0 };
    pnl.buckets[1] = BucketPnl { realized_ticks: realized1, matched_qty: qty1, fragments: frags1 };
    pnl.buckets[2] = BucketPnl { realized_ticks: realized2, matched_qty: qty2, fragments: frags2 };
    pnl
}

fn fold_core_fifo_nosink(
    carry: &mut VecDeque<Lot>,
    recs: &[PackedTrade],
    count: Option<(i32, i32)>,
    rules: &BucketRules,
) -> PartitionPnl {
    let mut pnl = PartitionPnl::default();
    let mut lots = Vec::with_capacity(carry.len() + recs.len());
    lots.extend(carry.drain(..));
    let mut head = 0usize;

    for r in recs {
        if r.signed_qty > 0 {
            lots.push(Lot {
                qty: r.signed_qty as i64,
                price_ticks: r.price_ticks as i64,
                day: r.day,
            });
            continue;
        }
        if r.signed_qty >= 0 {
            continue;
        }

        let mut remaining = -(r.signed_qty as i64);
        let sell_ticks = r.price_ticks as i64;
        let sell_day = r.day;
        let counted = match count {
            Some((lo, hi)) => sell_day >= lo && sell_day <= hi,
            None => true,
        };
        while remaining > 0 && head < lots.len() {
            let lot = &mut lots[head];
            let matched = remaining.min(lot.qty);
            let buy_day = lot.day;
            let buy_ticks = lot.price_ticks;
            lot.qty -= matched;
            if counted {
                let bucket = classify(rules, buy_day, sell_day);
                let realized = matched as i128 * (sell_ticks - buy_ticks) as i128;
                pnl.bucket_mut(bucket).add_frag(realized, matched);
            }
            remaining -= matched;
            if lot.qty == 0 {
                head += 1;
            }
        }
    }

    if head < lots.len() {
        carry.extend(lots.into_iter().skip(head));
    }
    pnl
}

/// Fold from a carry queue, counting every fragment, with the **default**
/// bucket rules. Custom rules go through [`fold_core`] directly (the live query
/// path); this convenience wrapper and its callers (tests, rollup, correction,
/// full-table bench) use the default ruleset.
pub fn fold_with_carry<S: FragmentSink>(
    client: u64,
    symbol: u32,
    carry: &mut VecDeque<Lot>,
    recs: &[PackedTrade],
    sink: &mut S,
) -> PartitionPnl {
    fold_core(client, symbol, carry, recs, sink, None, &BucketRules::default(), MatchPolicy::Fifo)
}

/// Fold a partition from a flat (no carry-in) state.
pub fn fold_partition<S: FragmentSink>(
    client: u64,
    symbol: u32,
    recs: &[PackedTrade],
    sink: &mut S,
) -> PartitionPnl {
    let mut carry = VecDeque::new();
    fold_with_carry(client, symbol, &mut carry, recs, sink)
}

/// Fold every partition of a packed table, summing into a single result and
/// streaming fragments to `sink`. This is benchmark arm 2 (full recompute).
pub fn fold_table<S: FragmentSink>(table: &PackedTable, sink: &mut S) -> PartitionPnl {
    let pc = table.part_client();
    let ps = table.part_symbol();
    let mut total = PartitionPnl::default();
    for p in 0..pc.len() {
        let part = fold_partition(pc[p], ps[p], table.partition(p), sink);
        total.merge(&part);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(sq: i32, px: i32, day: i32) -> PackedTrade {
        PackedTrade { signed_qty: sq, price_ticks: px, day }
    }

    #[test]
    fn bucket_rules_reclassify_by_threshold() {
        // a single 200-day-held round trip
        let recs = [rec(100, 1000, 0), rec(-100, 1200, 200)];
        // default (≤365 → short): 200 days is short-term
        let mut c1 = VecDeque::new();
        let d = fold_core(1, 1, &mut c1, &recs, &mut NoopSink, None, &BucketRules::default(), MatchPolicy::Fifo);
        assert_eq!(d.short().matched_qty, 100);
        assert_eq!(d.long().matched_qty, 0);
        // stricter jurisdiction (≤100 → short): the same trade is now long-term
        let mut c2 = VecDeque::new();
        let s = fold_core(
            1, 1, &mut c2, &recs, &mut NoopSink, None,
            &BucketRules::equity(100),
            MatchPolicy::Fifo,
        );
        assert_eq!(s.short().matched_qty, 0);
        assert_eq!(s.long().matched_qty, 100);
    }

    #[test]
    fn k_bucket_rules_classify_and_fold() {
        // intraday + two bands → 4 buckets: 0=intraday, 1=≤30d, 2=≤365d, 3=>365d
        let rules = BucketRules::bands(true, &[30, 365]);
        assert_eq!(rules.num_buckets(), 4);
        assert_eq!(classify(&rules, 5, 5), 0);
        assert_eq!(classify(&rules, 0, 10), 1);
        assert_eq!(classify(&rules, 0, 200), 2);
        assert_eq!(classify(&rules, 0, 500), 3);
        // a 200-day round trip lands in bucket 2, nothing elsewhere
        let recs = [rec(100, 1000, 0), rec(-100, 1200, 200)];
        let mut c = VecDeque::new();
        let pnl = fold_core(1, 1, &mut c, &recs, &mut NoopSink, None, &rules, MatchPolicy::Fifo);
        assert_eq!(pnl.buckets[2].matched_qty, 100);
        assert_eq!(pnl.buckets[2].realized_ticks, 100 * 200);
        assert_eq!(pnl.buckets[0].matched_qty, 0);
        assert_eq!(pnl.buckets[1].matched_qty, 0);
        assert_eq!(pnl.buckets[3].matched_qty, 0);
    }

    #[test]
    fn match_policy_picks_different_lots() {
        // three open buys at distinct prices, then one sell of 100 @5000
        let recs = [
            rec(100, 2000, 1), // oldest
            rec(100, 3000, 2), // highest cost
            rec(100, 1000, 3), // newest
            rec(-100, 5000, 4),
        ];
        let run = |p| {
            let mut c = VecDeque::new();
            fold_core(1, 1, &mut c, &recs, &mut NoopSink, None, &BucketRules::default(), p)
        };
        // FIFO drains oldest @2000; LIFO newest @1000; HIFO highest-cost @3000.
        assert_eq!(run(MatchPolicy::Fifo).short().realized_ticks, 100 * (5000 - 2000));
        assert_eq!(run(MatchPolicy::Lifo).short().realized_ticks, 100 * (5000 - 1000));
        assert_eq!(run(MatchPolicy::Hifo).short().realized_ticks, 100 * (5000 - 3000));
        // same matched quantity regardless of policy
        assert_eq!(run(MatchPolicy::Lifo).short().matched_qty, 100);
    }

    #[test]
    fn three_buckets() {
        // buy 100 @2000 d1; sell 60 @2200 d1 (intraday); sell 40 @2400 d100 (short)
        // buy 10 @2000 d200; sell 10 @2600 d700 (long, span 500)
        let recs = [
            rec(100, 2000, 1),
            rec(-60, 2200, 1),
            rec(-40, 2400, 100),
            rec(10, 2000, 200),
            rec(-10, 2600, 700),
        ];
        let pnl = fold_partition(7, 3, &recs, &mut NoopSink);
        assert_eq!(pnl.intraday().realized_ticks, 60 * 200); // 12000
        assert_eq!(pnl.intraday().matched_qty, 60);
        assert_eq!(pnl.short().realized_ticks, 40 * 400); // 16000
        assert_eq!(pnl.short().matched_qty, 40);
        assert_eq!(pnl.long().realized_ticks, 10 * 600); // 6000
        assert_eq!(pnl.long().matched_qty, 10);
    }

    #[test]
    fn fifo_drains_oldest_first() {
        // two buys at different prices, one big sell crossing both lots
        let recs = [
            rec(100, 1000, 1),
            rec(100, 2000, 2),
            rec(-150, 3000, 3), // matches 100@1000 then 50@2000
        ];
        let pnl = fold_partition(1, 1, &recs, &mut NoopSink);
        // span 1-2 days → short bucket
        let expect = 100 * (3000 - 1000) + 50 * (3000 - 2000);
        assert_eq!(pnl.short().realized_ticks, expect as i128);
        assert_eq!(pnl.short().matched_qty, 150);
    }

    #[test]
    fn carry_in_equivalent_to_full() {
        let part_a = [rec(100, 1000, 1), rec(50, 1100, 2)]; // two open buys
        let part_b = [rec(-120, 2000, 400)]; // sell crossing both, long span
        // full fold
        let full: Vec<PackedTrade> = part_a.iter().chain(part_b.iter()).copied().collect();
        let want = fold_partition(1, 1, &full, &mut NoopSink);
        // split fold: build carry from part_a, then fold part_b with carry
        let mut carry = VecDeque::new();
        fold_with_carry(1, 1, &mut carry, &part_a, &mut NoopSink);
        let got = fold_with_carry(1, 1, &mut carry, &part_b, &mut NoopSink);
        assert_eq!(want, got);
    }

    #[test]
    fn sink_sees_every_fragment() {
        struct Counter(u64, i128);
        impl FragmentSink for Counter {
            fn emit(&mut self, f: &Fragment) {
                self.0 += 1;
                self.1 += f.realized_ticks();
            }
        }
        let recs = [rec(100, 1000, 1), rec(-100, 1500, 1)];
        let mut c = Counter(0, 0);
        let pnl = fold_partition(1, 1, &recs, &mut c);
        assert_eq!(c.0, 1);
        assert_eq!(c.1, pnl.total_ticks());
    }
}
