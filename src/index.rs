//! `(client, symbol, time)` page-range index (Decision 6).
//!
//! Two complementary structures:
//! - The **partition table** inside the packed file already answers exact
//!   `(client, symbol)` lookups in `O(log parts)` — that's the per-client index.
//! - This **page index** summarizes each ~8 MiB page's `(client, day)` extent so
//!   a *time-range* or *cross-client* scan can skip pages that can't overlap —
//!   the "March 2024 skips 2023/2025" pruning. It's a sidecar JSON next to the
//!   packed file, rebuilt deterministically from it.

use crate::packed::PackedTable;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Records per page. 262144 × 32 B = 8 MiB ≈ one GPU transfer / CUDA stream unit.
pub const PAGE_RECORDS: usize = 262_144;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PageSummary {
    pub row_start: u64,
    pub row_end: u64,
    pub min_client: u64,
    pub max_client: u64,
    pub min_day: i32,
    pub max_day: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PageIndex {
    pub page_records: usize,
    pub pages: Vec<PageSummary>,
}

impl PageIndex {
    pub fn build(table: &PackedTable) -> Self {
        Self::build_with(table, PAGE_RECORDS)
    }

    pub fn build_with(table: &PackedTable, page_records: usize) -> Self {
        let recs = table.records();
        let part_client = table.part_client();
        let part_offset = table.part_offset();
        let n = recs.len();
        let mut pages = Vec::with_capacity(n / page_records + 1);

        // Walk partitions so we can attribute each row's client cheaply.
        let mut p = 0usize; // current partition
        let mut row = 0usize;
        while row < n {
            let end = (row + page_records).min(n);
            let mut min_day = i32::MAX;
            let mut max_day = i32::MIN;
            for r in &recs[row..end] {
                min_day = min_day.min(r.day);
                max_day = max_day.max(r.day);
            }
            // advance partition cursor to the partition containing `row`
            while p + 1 < part_offset.len() && part_offset[p + 1] as usize <= row {
                p += 1;
            }
            let min_client = part_client[p];
            // last partition that starts before `end`
            let mut pe = p;
            while pe + 1 < part_offset.len() && (part_offset[pe + 1] as usize) < end {
                pe += 1;
            }
            let max_client = part_client[pe.min(part_client.len() - 1)];

            pages.push(PageSummary {
                row_start: row as u64,
                row_end: end as u64,
                min_client,
                max_client,
                min_day,
                max_day,
            });
            row = end;
        }
        PageIndex { page_records, pages }
    }

    /// Pages whose extent overlaps `client ∈ [c_lo, c_hi]` AND `day ∈ [d_lo, d_hi]`.
    /// Returns `(row_start, row_end)` ranges to scan; the rest are pruned.
    pub fn prune(&self, c_lo: u64, c_hi: u64, d_lo: i32, d_hi: i32) -> Vec<(u64, u64)> {
        self.pages
            .iter()
            .filter(|pg| {
                pg.max_client >= c_lo
                    && pg.min_client <= c_hi
                    && pg.max_day >= d_lo
                    && pg.min_day <= d_hi
            })
            .map(|pg| (pg.row_start, pg.row_end))
            .collect()
    }

    pub fn sidecar_path(packed_path: &Path) -> PathBuf {
        let mut s = packed_path.as_os_str().to_owned();
        s.push(".idx.json");
        PathBuf::from(s)
    }

    pub fn write(&self, packed_path: &Path) -> Result<()> {
        let p = Self::sidecar_path(packed_path);
        std::fs::write(p, serde_json::to_string(self)?)?;
        Ok(())
    }

    pub fn open(packed_path: &Path) -> Result<Self> {
        let p = Self::sidecar_path(packed_path);
        Ok(serde_json::from_str(&std::fs::read_to_string(p)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::{PackedBuilder, PackedTable, PackedTrade};

    fn rec(sq: i32, day: i32) -> PackedTrade {
        PackedTrade { signed_qty: sq, price_ticks: 100, day }
    }

    #[test]
    fn pages_prune_by_client_and_day() {
        let mut b = PackedBuilder::new();
        // client 1 trades early days, client 9 trades late days
        b.push_partition(1, 10, &[rec(1, 1), rec(-1, 2)]);
        b.push_partition(9, 10, &[rec(1, 900), rec(-1, 901)]);
        let path = std::env::temp_dir().join("fifo_idx_test.fifopack");
        b.write(&path).unwrap();
        let t = PackedTable::open(&path).unwrap();
        // tiny page size so each partition is its own page
        let idx = PageIndex::build_with(&t, 2);
        assert_eq!(idx.pages.len(), 2);
        // a query for early days, client 1, prunes the late page
        let sel = idx.prune(1, 1, 0, 10);
        assert_eq!(sel, vec![(0, 2)]);
        // a query for late days prunes the early page
        let sel = idx.prune(0, 100, 800, 1000);
        assert_eq!(sel, vec![(2, 4)]);
        let _ = std::fs::remove_file(&path);
    }
}
