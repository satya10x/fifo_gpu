//! Lance backend (Stage 2.1) — versioned, transparent, self-describing store.
//!
//! The dataset keeps **per-row `client_id`/`symbol_id`** columns (so it stays
//! self-describing, queryable-by-client in Lance, and audit-friendly — Lance
//! RLE-compresses these hugely-repetitive columns, so they're nearly free on
//! disk) plus the records in a transparent **`FixedSizeBinary(12)`** column whose
//! bytes are exactly our `[PackedTrade]` layout.
//!
//! For the **fast compute-load path** we don't pay to materialize the redundant
//! per-row columns: a compact partition **sidecar** (`<uri>.parts.json`, ~the
//! 74 k `(client,symbol,offset)` boundaries) records the partition structure, and
//! the read **projects only the `rec` column**, slicing it in bulk into an
//! owned-buffer [`PackedTable`] (no temp file, GPU-DMA-able). If the sidecar is
//! absent (e.g. an externally-written dataset) we fall back to reading the
//! columns and deriving the boundaries.

use crate::packed::{serialize_packed, serialize_prefix, PackedTable, PackedTrade};
use anyhow::{Context, Result};
use arrow::array::{Array, FixedSizeBinaryArray, UInt32Array, UInt64Array};
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use futures::TryStreamExt;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

const REC_BYTES: i32 = 12; // size_of::<PackedTrade>()
const BATCH_ROWS: usize = 1 << 20; // ~1M rows per RecordBatch

/// Compact partition index stored alongside the dataset so the fast read can
/// skip the per-row client/symbol columns.
#[derive(Serialize, Deserialize)]
struct PartsSidecar {
    part_client: Vec<u64>,
    part_symbol: Vec<u32>,
    part_offset: Vec<u64>,
}

fn sidecar_path(uri: &str) -> String {
    format!("{uri}.parts.json")
}

fn schema() -> Arc<Schema> {
    // zstd-compress the highly-repetitive (constant-within-partition) client/symbol
    // columns — they're for self-description / Lance-native queries, not the fast
    // read, so compression doesn't touch the hot path. `rec` stays UNCOMPRESSED
    // (transparent) so the projected read is raw bytes, GPU-DMA-able.
    let zstd = HashMap::from([("lance-encoding:compression".to_string(), "zstd".to_string())]);
    Arc::new(Schema::new(vec![
        Field::new("client_id", DataType::UInt64, false).with_metadata(zstd.clone()),
        Field::new("symbol_id", DataType::UInt32, false).with_metadata(zstd),
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

/// Write the compute table to a versioned Lance dataset (self-describing
/// per-row columns + transparent `rec` column) plus a compact partition sidecar.
/// Returns the new dataset version.
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

    // compact partition sidecar for the fast read path
    let sidecar = PartsSidecar {
        part_client: pc.to_vec(),
        part_symbol: ps.to_vec(),
        part_offset: off.to_vec(),
    };
    std::fs::write(sidecar_path(uri), serde_json::to_vec(&sidecar)?)
        .context("write parts sidecar")?;

    Ok(ds.version().version)
}

/// Fallback: read all columns and derive partition boundaries (for datasets
/// without a sidecar).
async fn read_all_and_derive(uri: &str) -> Result<(Vec<PackedTrade>, Vec<u64>, Vec<u32>, Vec<u64>)> {
    let ds = Dataset::open(uri).await.context("Dataset::open")?;
    let mut stream = ds.scan().try_into_stream().await.context("scan stream")?;
    let mut records: Vec<PackedTrade> = Vec::new();
    let (mut pc, mut ps, mut poff) = (Vec::<u64>::new(), Vec::<u32>::new(), vec![0u64]);
    let mut cur: Option<(u64, u32)> = None;
    let mut gr: u64 = 0;
    while let Some(batch) = stream.try_next().await? {
        let cid = batch.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
        let sid = batch.column(1).as_any().downcast_ref::<UInt32Array>().unwrap();
        let rec = batch.column(2).as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap();
        records.extend_from_slice(bytemuck::cast_slice::<u8, PackedTrade>(rec.value_data()));
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
    Ok((records, pc, ps, poff))
}

/// Read a Lance compute dataset back into an owned-buffer [`PackedTable`]. Fast
/// path (sidecar present): project only `rec` + use the compact boundaries.
/// Fallback: read columns and derive boundaries.
pub fn open(uri: &str) -> Result<PackedTable> {
    let rt = runtime()?;
    match read_sidecar(uri)? {
        Some(s) => {
            // fast path, single copy: write header+partitions from the sidecar,
            // then stream the projected `rec` column straight into the buffer.
            let mut bytes: Vec<u8> = Vec::new();
            serialize_prefix(&s.part_client, &s.part_symbol, &s.part_offset, &mut bytes)?;
            rt.block_on(async {
                let ds = Dataset::open(uri).await.context("Dataset::open")?;
                let mut scan = ds.scan();
                scan.project(&["rec"]).context("project rec")?;
                let mut stream = scan.try_into_stream().await.context("scan stream")?;
                while let Some(batch) = stream.try_next().await? {
                    let rec = batch.column(0).as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap();
                    let b = rec.value_data();
                    debug_assert_eq!(b.len(), rec.len() * REC_BYTES as usize);
                    bytes.extend_from_slice(b);
                }
                Ok::<(), anyhow::Error>(())
            })?;
            PackedTable::from_bytes(bytes)
        }
        None => {
            // fallback: read all columns, derive boundaries, then serialize
            let (records, pc, ps, poff) = rt.block_on(read_all_and_derive(uri))?;
            let mut bytes: Vec<u8> = Vec::new();
            serialize_packed(&pc, &ps, &poff, &records, &mut bytes)?;
            PackedTable::from_bytes(bytes)
        }
    }
}

fn read_sidecar(uri: &str) -> Result<Option<PartsSidecar>> {
    let path = sidecar_path(uri);
    match std::fs::read(&path) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes).context("parse parts sidecar")?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::PackedBuilder;

    fn rec(sq: i32, px: i32, day: i32) -> PackedTrade {
        PackedTrade { signed_qty: sq, price_ticks: px, day }
    }

    fn sample_table(fp: &std::path::Path) -> PackedTable {
        let mut b = PackedBuilder::new();
        b.push_partition(1, 10, &[rec(100, 2000, 1), rec(-100, 2100, 2)]);
        b.push_partition(1, 20, &[rec(50, 500, 1)]);
        b.push_partition(2, 10, &[rec(10, 100, 5), rec(10, 110, 400), rec(-20, 120, 800)]);
        b.write(fp).unwrap();
        PackedTable::open(fp).unwrap()
    }

    fn assert_same(a: &PackedTable, b: &PackedTable) {
        assert_eq!(a.records(), b.records());
        assert_eq!(a.part_client(), b.part_client());
        assert_eq!(a.part_symbol(), b.part_symbol());
        assert_eq!(a.part_offset(), b.part_offset());
        let fold = |t: &PackedTable| crate::fifo::fold_table(t, &mut crate::fifo::NoopSink).total_ticks();
        assert_eq!(fold(a), fold(b));
    }

    #[test]
    fn lance_roundtrip_fast_path() {
        let tmp = std::env::temp_dir();
        let fp = tmp.join("fifo_gpu_lance21.fifopack");
        let uri = tmp.join("fifo_gpu_lance21.lance");
        let _ = std::fs::remove_dir_all(&uri);
        let _ = std::fs::remove_file(sidecar_path(uri.to_str().unwrap()));
        let t = sample_table(&fp);
        write(&t, uri.to_str().unwrap()).unwrap();
        // sidecar present → fast (project-rec) path
        let t2 = open(uri.to_str().unwrap()).unwrap();
        assert_same(&t, &t2);

        // remove sidecar → fallback path must produce the same table
        std::fs::remove_file(sidecar_path(uri.to_str().unwrap())).unwrap();
        let t3 = open(uri.to_str().unwrap()).unwrap();
        assert_same(&t, &t3);

        let _ = std::fs::remove_dir_all(&uri);
        let _ = std::fs::remove_file(&fp);
    }
}
