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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bucket {
    Intraday,
    Short,
    Long,
}

/// Data-driven bucket boundaries (Axis 1 of the generic engine — see DESIGN.md).
/// A.1: configurable holding-period threshold + intraday-same-day. The default
/// reproduces intraday / short (≤365d) / long (>365d); a different jurisdiction
/// just sets `short_max_days`. K-bucket generalization is A.2.
#[derive(Clone, Copy, Debug)]
pub struct BucketRules {
    /// `span == 0` → Intraday (otherwise it falls into the short/long bands).
    pub intraday_same_day: bool,
    /// Holding span ≤ this many days → Short; otherwise Long.
    pub short_max_days: i32,
}

impl Default for BucketRules {
    fn default() -> Self {
        BucketRules { intraday_same_day: true, short_max_days: LONG_TERM_DAYS }
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

#[inline]
pub fn classify(rules: &BucketRules, buy_day: i32, sell_day: i32) -> Bucket {
    if rules.intraday_same_day && buy_day == sell_day {
        Bucket::Intraday
    } else if sell_day - buy_day <= rules.short_max_days {
        Bucket::Short
    } else {
        Bucket::Long
    }
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
    pub bucket: Bucket,
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

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PartitionPnl {
    pub intraday: BucketPnl,
    pub short: BucketPnl,
    pub long: BucketPnl,
}

impl PartitionPnl {
    pub fn bucket_mut(&mut self, b: Bucket) -> &mut BucketPnl {
        match b {
            Bucket::Intraday => &mut self.intraday,
            Bucket::Short => &mut self.short,
            Bucket::Long => &mut self.long,
        }
    }
    pub fn merge(&mut self, o: &PartitionPnl) {
        self.intraday.merge(&o.intraday);
        self.short.merge(&o.short);
        self.long.merge(&o.long);
    }
    pub fn total_ticks(&self) -> i128 {
        self.intraday.realized_ticks + self.short.realized_ticks + self.long.realized_ticks
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
        assert_eq!(d.short.matched_qty, 100);
        assert_eq!(d.long.matched_qty, 0);
        // stricter jurisdiction (≤100 → short): the same trade is now long-term
        let mut c2 = VecDeque::new();
        let s = fold_core(
            1, 1, &mut c2, &recs, &mut NoopSink, None,
            &BucketRules { intraday_same_day: true, short_max_days: 100 },
            MatchPolicy::Fifo,
        );
        assert_eq!(s.short.matched_qty, 0);
        assert_eq!(s.long.matched_qty, 100);
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
        assert_eq!(run(MatchPolicy::Fifo).short.realized_ticks, 100 * (5000 - 2000));
        assert_eq!(run(MatchPolicy::Lifo).short.realized_ticks, 100 * (5000 - 1000));
        assert_eq!(run(MatchPolicy::Hifo).short.realized_ticks, 100 * (5000 - 3000));
        // same matched quantity regardless of policy
        assert_eq!(run(MatchPolicy::Lifo).short.matched_qty, 100);
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
        assert_eq!(pnl.intraday.realized_ticks, 60 * 200); // 12000
        assert_eq!(pnl.intraday.matched_qty, 60);
        assert_eq!(pnl.short.realized_ticks, 40 * 400); // 16000
        assert_eq!(pnl.short.matched_qty, 40);
        assert_eq!(pnl.long.realized_ticks, 10 * 600); // 6000
        assert_eq!(pnl.long.matched_qty, 10);
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
        assert_eq!(pnl.short.realized_ticks, expect as i128);
        assert_eq!(pnl.short.matched_qty, 150);
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
