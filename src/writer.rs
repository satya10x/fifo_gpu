//! Chunked Parquet writer for the tradebook. Rows arrive already in
//! `(client_id, symbol_id, ts)` order (the generator emits them that way), so
//! files are clustered for free — a head start for the M2 packer.

use crate::schema::{symbols_schema, tradebook_schema, Trade};
use crate::symbols::SymbolUniverse;
use anyhow::Result;
use arrow::array::{
    Float32Array, Float64Array, Int8Array, TimestampMicrosecondArray, UInt32Array, UInt64Array,
    UInt8Array,
};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Default)]
struct ColBuf {
    trade_id: Vec<u64>,
    client_id: Vec<u64>,
    symbol_id: Vec<u32>,
    side: Vec<i8>,
    quantity: Vec<u32>,
    price: Vec<f64>,
    ts_micros: Vec<i64>,
    venue: Vec<u8>,
    fees: Vec<f32>,
}

impl ColBuf {
    fn push(&mut self, t: &Trade) {
        self.trade_id.push(t.trade_id);
        self.client_id.push(t.client_id);
        self.symbol_id.push(t.symbol_id);
        self.side.push(t.side);
        self.quantity.push(t.quantity);
        self.price.push(t.price);
        self.ts_micros.push(t.ts_micros);
        self.venue.push(t.venue);
        self.fees.push(t.fees);
    }

    fn len(&self) -> usize {
        self.trade_id.len()
    }

    fn take_batch(&mut self, schema: &Arc<arrow::datatypes::Schema>) -> Result<RecordBatch> {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(std::mem::take(&mut self.trade_id))),
                Arc::new(UInt64Array::from(std::mem::take(&mut self.client_id))),
                Arc::new(UInt32Array::from(std::mem::take(&mut self.symbol_id))),
                Arc::new(Int8Array::from(std::mem::take(&mut self.side))),
                Arc::new(UInt32Array::from(std::mem::take(&mut self.quantity))),
                Arc::new(Float64Array::from(std::mem::take(&mut self.price))),
                Arc::new(TimestampMicrosecondArray::from(std::mem::take(
                    &mut self.ts_micros,
                ))),
                Arc::new(UInt8Array::from(std::mem::take(&mut self.venue))),
                Arc::new(Float32Array::from(std::mem::take(&mut self.fees))),
            ],
        )?;
        Ok(batch)
    }
}

pub struct TradebookWriter {
    dir: PathBuf,
    schema: Arc<arrow::datatypes::Schema>,
    props: WriterProperties,
    rows_per_file: usize,
    batch_rows: usize,
    buf: ColBuf,
    cur: Option<ArrowWriter<File>>,
    cur_rows: usize,
    file_idx: usize,
    pub total_rows: u64,
}

impl TradebookWriter {
    pub fn new(dir: &Path, rows_per_file: usize, batch_rows: usize) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
            .build();
        Ok(TradebookWriter {
            dir: dir.to_path_buf(),
            schema: tradebook_schema(),
            props,
            rows_per_file,
            batch_rows,
            buf: ColBuf::default(),
            cur: None,
            cur_rows: 0,
            file_idx: 0,
            total_rows: 0,
        })
    }

    pub fn push(&mut self, t: &Trade) -> Result<()> {
        self.buf.push(t);
        if self.buf.len() >= self.batch_rows {
            self.flush_batch()?;
        }
        Ok(())
    }

    fn ensure_writer(&mut self) -> Result<()> {
        if self.cur.is_none() || self.cur_rows >= self.rows_per_file {
            if let Some(w) = self.cur.take() {
                w.close()?;
            }
            let path = self.dir.join(format!("part-{:05}.parquet", self.file_idx));
            self.file_idx += 1;
            let file = File::create(path)?;
            self.cur = Some(ArrowWriter::try_new(
                file,
                self.schema.clone(),
                Some(self.props.clone()),
            )?);
            self.cur_rows = 0;
        }
        Ok(())
    }

    fn flush_batch(&mut self) -> Result<()> {
        if self.buf.len() == 0 {
            return Ok(());
        }
        self.ensure_writer()?;
        let n = self.buf.len();
        let batch = self.buf.take_batch(&self.schema)?;
        self.cur.as_mut().unwrap().write(&batch)?;
        self.cur_rows += n;
        self.total_rows += n as u64;
        Ok(())
    }

    pub fn finish(mut self) -> Result<u64> {
        self.flush_batch()?;
        if let Some(w) = self.cur.take() {
            w.close()?;
        }
        Ok(self.total_rows)
    }
}

/// Write the symbol reference table (symbol_id → base price, vol).
pub fn write_symbols(dir: &Path, symbols: &SymbolUniverse) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let schema = symbols_schema();
    let ids: Vec<u32> = (0..symbols.n).collect();
    let base: Vec<f64> = ids.iter().map(|&id| symbols.base_price(id)).collect();
    let vol: Vec<f64> = ids.iter().map(|&id| symbols.daily_vol(id)).collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(ids)),
            Arc::new(Float64Array::from(base)),
            Arc::new(Float64Array::from(vol)),
        ],
    )?;
    let file = File::create(dir.join("symbols.parquet"))?;
    let mut w = ArrowWriter::try_new(file, schema, None)?;
    w.write(&batch)?;
    w.close()?;
    Ok(())
}
