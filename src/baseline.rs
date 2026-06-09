//! Benchmark arm 3 — the status-quo baseline both engines must beat.
//!
//! Simulates "current FIFO PnL over ClickHouse/Parquet on S3": a full Parquet
//! re-scan with no packed layout, no page index, and no checkpoints. For a
//! range query it folds the matched partitions' *entire* history (the
//! checkpoint-free cost) and counts only the in-range sells.

use crate::fifo::{fold_core, BucketRules, NoopSink, PartitionPnl};
use crate::packed::PackedTrade;
use crate::query::{ClientSel, Query, Span};
use anyhow::{Context, Result};
use arrow::array::{Float64Array, Int8Array, TimestampMicrosecondArray, UInt32Array, UInt64Array};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::VecDeque;
use std::fs::File;
use std::path::{Path, PathBuf};

const MICROS_PER_DAY: i64 = 86_400_000_000;

fn list_parts(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.starts_with("part-") && s.ends_with(".parquet"))
                .unwrap_or(false)
        })
        .collect();
    v.sort();
    Ok(v)
}

fn col<'a, T: 'static>(b: &'a RecordBatch, name: &str) -> Result<&'a T> {
    b.column_by_name(name)
        .with_context(|| format!("missing column {name}"))?
        .as_any()
        .downcast_ref::<T>()
        .with_context(|| format!("column {name} wrong type"))
}

fn matches(q: &Query, client: u64, symbol: u32) -> bool {
    let c_ok = match q.clients {
        ClientSel::One(c) => client == c,
        ClientSel::All => true,
    };
    let s_ok = q.symbol.map_or(true, |s| symbol == s);
    c_ok && s_ok
}

/// Fold a query directly off the tradebook Parquet (no acceleration).
pub fn baseline_query(tradebook_dir: &Path, q: &Query) -> Result<(PartitionPnl, u64)> {
    let count = match q.span {
        Span::Full => None,
        Span::Range(lo, hi) => Some((lo, hi)),
    };
    let mut pnl = PartitionPnl::default();
    let mut rows_touched = 0u64;

    let mut cur: Option<(u64, u32)> = None;
    let mut buf: Vec<PackedTrade> = Vec::new();
    let flush = |key: (u64, u32), buf: &mut Vec<PackedTrade>, pnl: &mut PartitionPnl, rows: &mut u64| {
        if !buf.is_empty() {
            if matches(q, key.0, key.1) {
                *rows += buf.len() as u64;
                let mut carry: VecDeque<_> = VecDeque::new();
                pnl.merge(&fold_core(key.0, key.1, &mut carry, buf, &mut NoopSink, count, &BucketRules::default()));
            }
            buf.clear();
        }
    };

    for path in list_parts(tradebook_dir)? {
        let file = File::open(&path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
            .with_batch_size(65_536)
            .build()?;
        for batch in reader {
            let batch = batch?;
            let client = col::<UInt64Array>(&batch, "client_id")?;
            let symbol = col::<UInt32Array>(&batch, "symbol_id")?;
            let side = col::<Int8Array>(&batch, "side")?;
            let qty = col::<UInt32Array>(&batch, "quantity")?;
            let price = col::<Float64Array>(&batch, "price")?;
            let ts = col::<TimestampMicrosecondArray>(&batch, "ts_micros")?;
            for i in 0..batch.num_rows() {
                let key = (client.value(i), symbol.value(i));
                if cur != Some(key) {
                    if let Some(k) = cur {
                        flush(k, &mut buf, &mut pnl, &mut rows_touched);
                    }
                    cur = Some(key);
                }
                let t = ts.value(i);
                buf.push(PackedTrade {
                    signed_qty: side.value(i) as i32 * qty.value(i) as i32,
                    price_ticks: crate::packed::price_to_ticks(price.value(i)),
                    day: (t / MICROS_PER_DAY) as i32,
                });
            }
        }
    }
    if let Some(k) = cur {
        flush(k, &mut buf, &mut pnl, &mut rows_touched);
    }
    Ok((pnl, rows_touched))
}
