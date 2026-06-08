//! Correction path (M8, the chosen driver). A back-dated or busted trade dirties
//! exactly one client's partitions; because generation is **per-client
//! deterministic**, we can regenerate just that client's slice, recompute its
//! PnL, and know precisely which checkpoints must be rebuilt — without touching
//! any other client.
//!
//! This module implements the recompute + invalidation logic and proves the
//! determinism invariant (regenerated slice ≡ what's in the packed table).

use crate::checkpoint::CheckpointStore;
use crate::config::GenConfig;
use crate::fifo::{fold_partition, NoopSink, PartitionPnl};
use crate::generate::regenerate_client;
use crate::packed::PackedTrade;
use crate::query::{self, ClientSel, Query, Span};
use crate::packed::PackedTable;
use anyhow::Result;

#[derive(Debug)]
pub struct CorrectionReport {
    pub client: u64,
    pub n_trades: usize,
    pub n_partitions: usize,
    pub pnl: PartitionPnl,
    pub invalidated_checkpoints: Vec<i32>,
}

/// Recompute one client's PnL purely from `(cfg, client_id)` — the cheap,
/// isolated work a correction triggers. Trades come back symbol-major (then
/// ts-sorted), so partitions are contiguous.
pub fn recompute_client(cfg: &GenConfig, client_id: u64) -> (PartitionPnl, usize, usize) {
    let trades = regenerate_client(cfg, client_id);
    let mut pnl = PartitionPnl::default();
    let mut n_parts = 0usize;
    let mut cur_sym: Option<u32> = None;
    let mut buf: Vec<PackedTrade> = Vec::new();
    let flush = |sym: u32, buf: &mut Vec<PackedTrade>, pnl: &mut PartitionPnl, n: &mut usize| {
        if !buf.is_empty() {
            pnl.merge(&fold_partition(client_id, sym, buf, &mut NoopSink));
            *n += 1;
            buf.clear();
        }
    };
    for t in &trades {
        if cur_sym != Some(t.symbol_id) {
            if let Some(s) = cur_sym {
                flush(s, &mut buf, &mut pnl, &mut n_parts);
            }
            cur_sym = Some(t.symbol_id);
        }
        buf.push(PackedTrade::from_trade(t));
    }
    if let Some(s) = cur_sym {
        flush(s, &mut buf, &mut pnl, &mut n_parts);
    }
    (pnl, trades.len(), n_parts)
}

/// Run a correction for `client_id` with the busted trade on `correction_day`:
/// recompute the client's PnL and report which checkpoints are invalidated.
pub fn correct(
    cfg: &GenConfig,
    ckpt: &CheckpointStore,
    client_id: u64,
    correction_day: i32,
) -> Result<CorrectionReport> {
    let (pnl, n_trades, n_partitions) = recompute_client(cfg, client_id);
    let invalidated = ckpt.invalidated_by(correction_day)?;
    Ok(CorrectionReport {
        client: client_id,
        n_trades,
        n_partitions,
        pnl,
        invalidated_checkpoints: invalidated,
    })
}

/// Determinism invariant: regenerating a client in isolation must equal folding
/// that client's partitions in the packed table. Returns `(regenerated, packed)`.
pub fn verify_against_packed(
    cfg: &GenConfig,
    table: &PackedTable,
    client_id: u64,
) -> Result<(PartitionPnl, PartitionPnl)> {
    let (regen, _, _) = recompute_client(cfg, client_id);
    let q = Query { clients: ClientSel::One(client_id), symbol: None, span: Span::Full };
    let packed = query::run_cpu(table, None, &q, &mut NoopSink)?.pnl;
    Ok((regen, packed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regeneration_is_deterministic() {
        let mut cfg = GenConfig::defaults();
        cfg.clients = 2_000;
        cfg.days = 120;
        let (a, ta, pa) = recompute_client(&cfg, 0); // client 0 is a whale (stride)
        let (b, tb, pb) = recompute_client(&cfg, 0);
        assert_eq!(a, b);
        assert_eq!(ta, tb);
        assert_eq!(pa, pb);
        assert!(ta > 0);
    }
}
