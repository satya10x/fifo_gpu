//! Query execution over the packed compute table.
//!
//! Resolves a [`Query`] to a set of partitions (exact `(client,symbol)` lookup,
//! per-client range, or full cross-client), folds each with the CPU oracle, and
//! — for date ranges — uses checkpoint carry-in (M4) so work is bounded by the
//! range, not history depth. Returns the PnL plus the cost-model signals the
//! router (M7) needs: `rows_touched`, `partition_fanout`, `checkpoints_loaded`.

use crate::checkpoint::CheckpointStore;
use crate::fifo::{fold_core, FragmentSink, PartitionPnl};
use crate::packed::PackedTable;
use anyhow::Result;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug)]
pub enum ClientSel {
    One(u64),
    All,
}

#[derive(Clone, Copy, Debug)]
pub enum Span {
    /// Whole history — no carry-in needed; fold partitions from flat.
    Full,
    /// `[lo, hi]` day-index range (inclusive) — needs checkpoint carry-in.
    Range(i32, i32),
}

#[derive(Clone, Copy, Debug)]
pub struct Query {
    pub clients: ClientSel,
    pub symbol: Option<u32>,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct QueryResult {
    pub pnl: PartitionPnl,
    pub rows_touched: u64,
    pub partition_fanout: u64,
    pub checkpoints_loaded: u64,
}

/// Partition indices selected by a query.
fn select_partitions(table: &PackedTable, q: &Query) -> Vec<usize> {
    match (q.clients, q.symbol) {
        (ClientSel::One(c), Some(s)) => table.find_partition(c, s).into_iter().collect(),
        (ClientSel::One(c), None) => table.partitions_for_client(c).collect(),
        (ClientSel::All, sym) => {
            let ps = table.part_symbol();
            (0..table.n_parts() as usize)
                .filter(|&p| sym.map_or(true, |s| ps[p] == s))
                .collect()
        }
    }
}

pub fn run_cpu<S: FragmentSink>(
    table: &PackedTable,
    ckpt: Option<&CheckpointStore>,
    q: &Query,
    sink: &mut S,
) -> Result<QueryResult> {
    let pc = table.part_client();
    let ps = table.part_symbol();
    let parts = select_partitions(table, q);

    // For a range, load the single nearest-prior checkpoint once.
    let (cutoff, ckpt_snap) = match (q.span, ckpt) {
        (Span::Range(lo, _), Some(store)) => {
            let snap = store.load_nearest_before(lo)?;
            let cutoff = snap.as_ref().map(|c| c.cutoff_day).unwrap_or(i32::MIN);
            (cutoff, snap)
        }
        _ => (i32::MIN, None),
    };

    let mut out = QueryResult::default();
    for &p in &parts {
        let (client, symbol) = (pc[p], ps[p]);
        let recs = table.partition(p);
        let part_pnl = match q.span {
            Span::Full => {
                out.rows_touched += recs.len() as u64;
                let mut carry = VecDeque::new();
                fold_core(client, symbol, &mut carry, recs, sink, None)
            }
            Span::Range(lo, hi) => {
                let mut carry = ckpt_snap
                    .as_ref()
                    .map(|c| c.carry_for(client, symbol))
                    .unwrap_or_default();
                if !carry.is_empty() {
                    out.checkpoints_loaded += 1;
                }
                // Replay only (cutoff, hi]; count only sells in [lo, hi].
                let replay = table.day_range(recs, cutoff.saturating_add(1), hi);
                out.rows_touched += replay.len() as u64;
                fold_core(client, symbol, &mut carry, replay, sink, Some((lo, hi)))
            }
        };
        out.pnl.merge(&part_pnl);
        out.partition_fanout += 1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fifo::{fold_partition, NoopSink};
    use crate::packed::{PackedBuilder, PackedTrade};

    fn rec(sq: i64, px: i64, day: i32) -> PackedTrade {
        PackedTrade { signed_qty: sq, price_ticks: px, day, _pad: 0, ts: day as i64 }
    }

    #[test]
    fn full_history_per_client_matches_direct_fold() {
        let mut b = PackedBuilder::new();
        let a = [rec(100, 1000, 1), rec(-100, 1200, 1)];
        let c = [rec(50, 500, 2), rec(-50, 700, 50)];
        b.push_partition(7, 1, &a);
        b.push_partition(7, 2, &c);
        let path = std::env::temp_dir().join("fifo_query_test.fifopack");
        b.write(&path).unwrap();
        let t = PackedTable::open(&path).unwrap();

        let q = Query { clients: ClientSel::One(7), symbol: None, span: Span::Full };
        let r = run_cpu(&t, None, &q, &mut NoopSink).unwrap();

        let mut want = fold_partition(7, 1, &a, &mut NoopSink);
        want.merge(&fold_partition(7, 2, &c, &mut NoopSink));
        assert_eq!(r.pnl, want);
        assert_eq!(r.partition_fanout, 2);
        assert_eq!(r.rows_touched, 4);
        let _ = std::fs::remove_file(&path);
    }
}
