//! The packer (M2): tradebook Parquet → packed compute table + page index.
//!
//! Streams the `(client,symbol,ts)`-sorted tradebook, cuts a new partition on
//! every `(client,symbol)` change, converts each row to the 32-byte
//! [`PackedTrade`] tuple, and writes the transparent packed buffer + sidecar
//! index.
//!
//! NOTE (scale): this builds the record buffer in RAM before writing. Fine for
//! dev/benchmark scales (tens of millions of rows ≈ 1.5 GB). Production
//! (billions) wants a streaming two-file assemble — left as a follow-up; the
//! on-disk format already supports it (offsets are computed up front).

use crate::index::PageIndex;
use crate::packed::{price_to_ticks, PackedBuilder, PackedTable, PackedTrade};
use anyhow::{Context, Result};
use arrow::array::{
    Float64Array, Int8Array, TimestampMicrosecondArray, UInt32Array, UInt64Array,
};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::{Path, PathBuf};

const MICROS_PER_DAY: i64 = 86_400_000_000;

#[derive(Debug)]
pub struct PackStats {
    pub n_rows: u64,
    pub n_parts: u64,
    pub n_pages: usize,
    pub bytes: u64,
}

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

fn col<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a T> {
    batch
        .column_by_name(name)
        .with_context(|| format!("missing column {name}"))?
        .as_any()
        .downcast_ref::<T>()
        .with_context(|| format!("column {name} has unexpected type"))
}

pub fn pack(tradebook_dir: &Path, out_path: &Path, page_records: usize) -> Result<PackStats> {
    let parts = list_parts(tradebook_dir)?;
    anyhow::ensure!(!parts.is_empty(), "no part-*.parquet in {}", tradebook_dir.display());

    let mut b = PackedBuilder::new();
    let mut cur: Option<(u64, u32)> = None;
    let mut cur_recs: Vec<PackedTrade> = Vec::new();

    for path in &parts {
        let file = File::open(path)?;
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
                    if let Some((cc, ss)) = cur {
                        b.push_partition(cc, ss, &cur_recs);
                        cur_recs.clear();
                    }
                    cur = Some(key);
                }
                let signed_qty = side.value(i) as i64 * qty.value(i) as i64;
                let t = ts.value(i);
                cur_recs.push(PackedTrade {
                    signed_qty,
                    price_ticks: price_to_ticks(price.value(i)),
                    day: (t / MICROS_PER_DAY) as i32,
                    _pad: 0,
                    ts: t,
                });
            }
        }
    }
    if let Some((cc, ss)) = cur {
        b.push_partition(cc, ss, &cur_recs);
    }

    b.write(out_path)?;
    let table = PackedTable::open(out_path)?;
    let idx = PageIndex::build_with(&table, page_records);
    idx.write(out_path)?;

    let bytes = std::fs::metadata(out_path)?.len();
    Ok(PackStats {
        n_rows: table.n_rows(),
        n_parts: table.n_parts(),
        n_pages: idx.pages.len(),
        bytes,
    })
}
