//! Lance backend (Stage 1) — a versioned store for the compute table.
//!
//! Writes the packed compute table to a Lance dataset (columns `client_id,
//! symbol_id, signed_qty, price_ticks, day`; one row per trade, in
//! `(client,symbol,ts)`-clustered order) and reads it back, regrouping rows into
//! partitions to rebuild a [`PackedBuilder`]. This gives Lance's versioning /
//! time-travel / correction-lineage on the compute table; the CPU/GPU engine
//! then runs unchanged on the round-tripped buffer.
//!
//! Stage 1 is **read→repack** (Lance → Arrow → our 12 B buffer). Zero-copy — a
//! custom transparent Lance encoding whose decoder hands the GPU buffer back
//! verbatim — is a later stage (see DESIGN.md).

use crate::packed::{PackedBuilder, PackedTable, PackedTrade};
use anyhow::{Context, Result};
use arrow::array::{Array, Int32Array, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use futures::TryStreamExt;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use std::sync::Arc;

const BATCH_ROWS: usize = 1 << 20; // ~1M rows per RecordBatch

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("client_id", DataType::UInt64, false),
        Field::new("symbol_id", DataType::UInt32, false),
        Field::new("signed_qty", DataType::Int32, false),
        Field::new("price_ticks", DataType::Int32, false),
        Field::new("day", DataType::Int32, false),
    ]))
}

fn runtime() -> Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?)
}

#[allow(clippy::too_many_arguments)]
fn push_batch(
    sch: &Arc<Schema>,
    cid: &mut Vec<u64>,
    sid: &mut Vec<u32>,
    sq: &mut Vec<i32>,
    pt: &mut Vec<i32>,
    dy: &mut Vec<i32>,
    out: &mut Vec<RecordBatch>,
) -> Result<()> {
    if cid.is_empty() {
        return Ok(());
    }
    let b = RecordBatch::try_new(
        sch.clone(),
        vec![
            Arc::new(UInt64Array::from(std::mem::take(cid))),
            Arc::new(UInt32Array::from(std::mem::take(sid))),
            Arc::new(Int32Array::from(std::mem::take(sq))),
            Arc::new(Int32Array::from(std::mem::take(pt))),
            Arc::new(Int32Array::from(std::mem::take(dy))),
        ],
    )?;
    out.push(b);
    Ok(())
}

/// Write the compute table to a Lance dataset (overwriting any existing one).
/// Returns the new dataset version.
pub fn write(table: &PackedTable, uri: &str) -> Result<u64> {
    let sch = schema();
    let pc = table.part_client();
    let ps = table.part_symbol();
    let off = table.part_offset();
    let recs = table.records();

    let mut batches: Vec<RecordBatch> = Vec::new();
    let (mut cid, mut sid) = (Vec::new(), Vec::new());
    let (mut sq, mut pt, mut dy) = (Vec::new(), Vec::new(), Vec::new());
    for p in 0..pc.len() {
        let (c, s) = (pc[p], ps[p]);
        for r in &recs[off[p] as usize..off[p + 1] as usize] {
            cid.push(c);
            sid.push(s);
            sq.push(r.signed_qty);
            pt.push(r.price_ticks);
            dy.push(r.day);
            if cid.len() >= BATCH_ROWS {
                push_batch(&sch, &mut cid, &mut sid, &mut sq, &mut pt, &mut dy, &mut batches)?;
            }
        }
    }
    push_batch(&sch, &mut cid, &mut sid, &mut sq, &mut pt, &mut dy, &mut batches)?;

    let reader = RecordBatchIterator::new(
        batches.into_iter().map(Ok::<RecordBatch, ArrowError>),
        sch.clone(),
    );
    let params = WriteParams {
        mode: WriteMode::Overwrite,
        ..Default::default()
    };
    let rt = runtime()?;
    let ds = rt
        .block_on(async { Dataset::write(reader, uri, Some(params)).await })
        .context("Dataset::write")?;
    Ok(ds.version().version)
}

/// Read a Lance compute dataset back into a [`PackedBuilder`], regrouping
/// consecutive `(client,symbol)` rows into partitions.
pub fn open(uri: &str) -> Result<PackedBuilder> {
    let rt = runtime()?;
    let mut b = PackedBuilder::new();
    rt.block_on(async {
        let ds = Dataset::open(uri).await.context("Dataset::open")?;
        let mut stream = ds.scan().try_into_stream().await.context("scan stream")?;
        let mut cur: Option<(u64, u32)> = None;
        let mut buf: Vec<PackedTrade> = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            let cid = batch.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
            let sid = batch.column(1).as_any().downcast_ref::<UInt32Array>().unwrap();
            let sq = batch.column(2).as_any().downcast_ref::<Int32Array>().unwrap();
            let pt = batch.column(3).as_any().downcast_ref::<Int32Array>().unwrap();
            let dy = batch.column(4).as_any().downcast_ref::<Int32Array>().unwrap();
            for i in 0..batch.num_rows() {
                let key = (cid.value(i), sid.value(i));
                if cur != Some(key) {
                    if let Some((cc, ss)) = cur {
                        b.push_partition(cc, ss, &buf);
                        buf.clear();
                    }
                    cur = Some(key);
                }
                buf.push(PackedTrade {
                    signed_qty: sq.value(i),
                    price_ticks: pt.value(i),
                    day: dy.value(i),
                });
            }
        }
        if let Some((cc, ss)) = cur {
            b.push_partition(cc, ss, &buf);
        }
        Ok::<(), anyhow::Error>(())
    })?;
    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let b2 = open(uri.to_str().unwrap()).unwrap();

        // round-trip must reproduce the packed table exactly
        assert_eq!(&b2.records[..], t.records());
        assert_eq!(&b2.part_client[..], t.part_client());
        assert_eq!(&b2.part_symbol[..], t.part_symbol());
        assert_eq!(&b2.part_offset[..], t.part_offset());

        let _ = std::fs::remove_dir_all(&uri);
        let _ = std::fs::remove_file(&fp);
    }
}
