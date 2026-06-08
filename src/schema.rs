//! Tradebook record shape + Arrow schema.
//!
//! The tradebook is the wide system-of-record (Decision 4). It keeps a little
//! context (venue, fees) beyond the bare FIFO inputs. Opaque compression is
//! fine *here* (storage table) — the "transparent only" rule applies to the
//! packed compute table (M2), not this one.

use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct Trade {
    pub trade_id: u64,
    pub client_id: u64,
    pub symbol_id: u32,
    pub side: i8, // +1 buy, -1 sell
    pub quantity: u32,
    pub price: f64,
    pub ts_micros: i64,
    pub venue: u8,
    pub fees: f32,
}

pub fn tradebook_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("trade_id", DataType::UInt64, false),
        Field::new("client_id", DataType::UInt64, false),
        Field::new("symbol_id", DataType::UInt32, false),
        Field::new("side", DataType::Int8, false),
        Field::new("quantity", DataType::UInt32, false),
        Field::new("price", DataType::Float64, false),
        Field::new(
            "ts_micros",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("venue", DataType::UInt8, false),
        Field::new("fees", DataType::Float32, false),
    ]))
}

pub fn symbols_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("symbol_id", DataType::UInt32, false),
        Field::new("base_price", DataType::Float64, false),
        Field::new("daily_vol", DataType::Float64, false),
    ]))
}
