//! Lance backend (Stage 2) — versioned, **transparent** compute-table store.
//!
//! The records are stored as a single `FixedSizeBinary(12)` column whose bytes
//! are exactly our `[PackedTrade]` layout — a transparent, fixed-width encoding.
//! On read, that column's value buffer **is** the packed records: we slice it in
//! bulk (no per-row reconstruction) and assemble an owned-buffer [`PackedTable`]
//! in memory (no temp file). The CPU/GPU engine then runs on it unchanged, and
//! the buffer is GPU-DMA-able. `client_id`/`symbol_id` columns let us recover
//! the partition boundaries on read.
//!
//! This realizes "Lance as the transparent store whose decoder hands the GPU
//! buffer back" with one bulk copy on read (the unavoidable Lance→host read),
//! versus Stage 1's per-row rebuild + temp-file write.

use crate::packed::{serialize_packed, PackedTable, PackedTrade};
use anyhow::{Context, Result};
use arrow::array::{Array, FixedSizeBinaryArray, UInt32Array, UInt64Array};
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use futures::TryStreamExt;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use std::sync::Arc;

const REC_BYTES: i32 = 12; // size_of::<PackedTrade>()
const BATCH_ROWS: usize = 1 << 20; // ~1M rows per RecordBatch

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("client_id", DataType::UInt64, false),
        Field::new("symbol_id", DataType::UInt32, false),
        Field::new("rec", DataType::FixedSizeBinary(REC_BYTES), false),
    ]))
}

fn runtime() -> Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread().enable_all().build()?)
}

fn flush_batch(
    sch: &Arc<Schema>,
    cid: &mut Vec<u64>,
    sid: &mut Vec<u32>,
    rec_bytes: &[u8],
    out: &mut Vec<RecordBatch>,
) -> Result<()> {
    if cid.is_empty() {
        return Ok(());
    }
    let rec = FixedSizeBinaryArray::new(REC_BYTES, Buffer::from(rec_bytes), None);
    let b = RecordBatch::try_new(
        sch.clone(),
        vec![
            Arc::new(UInt64Array::from(std::mem::take(cid))),
            Arc::new(UInt32Array::from(std::mem::take(sid))),
            Arc::new(rec),
        ],
    )?;
    out.push(b);
    Ok(())
}

/// Write the compute table to a versioned Lance dataset (overwriting). Records
/// go in a transparent `FixedSizeBinary(12)` column. Returns the new version.
pub fn write(table: &PackedTable, uri: &str) -> Result<u64> {
    let sch = schema();
    let pc = table.part_client();
    let ps = table.part_symbol();
    let off = table.part_offset();
    let recs = table.records();

    let mut batches: Vec<RecordBatch> = Vec::new();
    let (mut cid, mut sid) = (Vec::<u64>::new(), Vec::<u32>::new());
    let mut batch_start = 0usize;
    let mut row = 0usize;
    for p in 0..pc.len() {
        let (c, s) = (pc[p], ps[p]);
        for _ in off[p]..off[p + 1] {
            cid.push(c);
            sid.push(s);
            row += 1;
            if cid.len() >= BATCH_ROWS {
                let rb: &[u8] = bytemuck::cast_slice(&recs[batch_start..row]);
                flush_batch(&sch, &mut cid, &mut sid, rb, &mut batches)?;
                batch_start = row;
            }
        }
    }
    let rb: &[u8] = bytemuck::cast_slice(&recs[batch_start..row]);
    flush_batch(&sch, &mut cid, &mut sid, rb, &mut batches)?;

    let reader = RecordBatchIterator::new(
        batches.into_iter().map(Ok::<RecordBatch, ArrowError>),
        sch.clone(),
    );
    let params = WriteParams { mode: WriteMode::Overwrite, ..Default::default() };
    let rt = runtime()?;
    let ds = rt
        .block_on(async { Dataset::write(reader, uri, Some(params)).await })
        .context("Dataset::write")?;
    Ok(ds.version().version)
}

/// Read a Lance compute dataset back into an owned-buffer [`PackedTable`]
/// (records sliced in bulk from the transparent column; partitions recovered
/// from the client/symbol columns). No per-row rebuild, no temp file.
pub fn open(uri: &str) -> Result<PackedTable> {
    let rt = runtime()?;
    let (records, pc, ps, poff) = rt.block_on(async {
        let ds = Dataset::open(uri).await.context("Dataset::open")?;
        let mut stream = ds.scan().try_into_stream().await.context("scan stream")?;

        let mut records: Vec<PackedTrade> = Vec::new();
        let mut pc: Vec<u64> = Vec::new();
        let mut ps: Vec<u32> = Vec::new();
        let mut poff: Vec<u64> = vec![0];
        let mut cur: Option<(u64, u32)> = None;
        let mut gr: u64 = 0;

        while let Some(batch) = stream.try_next().await? {
            let cid = batch.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
            let sid = batch.column(1).as_any().downcast_ref::<UInt32Array>().unwrap();
            let rec = batch.column(2).as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap();

            // bulk-append this batch's records straight from the transparent column
            let bytes = rec.value_data();
            debug_assert_eq!(bytes.len(), rec.len() * REC_BYTES as usize);
            records.extend_from_slice(bytemuck::cast_slice::<u8, PackedTrade>(bytes));

            // recover partition boundaries from (client,symbol) runs
            for i in 0..batch.num_rows() {
                let key = (cid.value(i), sid.value(i));
                match cur {
                    None => cur = Some(key),
                    Some(k) if k != key => {
                        pc.push(k.0);
                        ps.push(k.1);
                        poff.push(gr);
                        cur = Some(key);
                    }
                    _ => {}
                }
                gr += 1;
            }
        }
        if let Some(k) = cur {
            pc.push(k.0);
            ps.push(k.1);
            poff.push(gr);
        }
        Ok::<_, anyhow::Error>((records, pc, ps, poff))
    })?;

    // assemble the .fifopack byte layout in memory and open as an owned table
    let mut bytes: Vec<u8> = Vec::new();
    serialize_packed(&pc, &ps, &poff, &records, &mut bytes)?;
    PackedTable::from_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::PackedBuilder;

    fn rec(sq: i32, px: i32, day: i32) -> PackedTrade {
        PackedTrade { signed_qty: sq, price_ticks: px, day }
    }

    #[test]
    fn lance_roundtrip_is_identical() {
        let mut b = PackedBuilder::new();
        b.push_partition(1, 10, &[rec(100, 2000, 1), rec(-100, 2100, 2)]);
        b.push_partition(1, 20, &[rec(50, 500, 1)]);
        b.push_partition(2, 10, &[rec(10, 100, 5), rec(10, 110, 400), rec(-20, 120, 800)]);

        let tmp = std::env::temp_dir();
        let fp = tmp.join("fifo_gpu_lance_rt.fifopack");
        let uri = tmp.join("fifo_gpu_lance_rt.lance");
        let _ = std::fs::remove_dir_all(&uri);
        b.write(&fp).unwrap();
        let t = PackedTable::open(&fp).unwrap();

        let _version = write(&t, uri.to_str().unwrap()).unwrap();
        let t2 = open(uri.to_str().unwrap()).unwrap();

        // round-trip must reproduce the packed table exactly (and fold identically)
        assert_eq!(t2.records(), t.records());
        assert_eq!(t2.part_client(), t.part_client());
        assert_eq!(t2.part_symbol(), t.part_symbol());
        assert_eq!(t2.part_offset(), t.part_offset());
        let fold = |tab: &PackedTable| {
            crate::fifo::fold_table(tab, &mut crate::fifo::NoopSink).total_ticks()
        };
        assert_eq!(fold(&t2), fold(&t));

        let _ = std::fs::remove_dir_all(&uri);
        let _ = std::fs::remove_file(&fp);
    }
}
