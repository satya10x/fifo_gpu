//! fifo_gpu — Milestone 1: skew-realistic synthetic tradebook generator.
//!
//! Produces a `(client_id, symbol_id, ts)`-clustered tradebook (Parquet) whose
//! trade-count-per-client distribution is an explicit power-law with whale
//! accounts — the realistic-skew substrate the benchmark (§5 of the handoff)
//! depends on. Generation is **per-client deterministic**: every client's slice
//! is a pure function of `(seed, client_id)`, so any single client can later be
//! regenerated bit-identically — the primitive the correction path (M8) needs.

pub mod baseline;
pub mod bench;
pub mod calendar;
pub mod checkpoint;
pub mod config;
pub mod correction;
pub mod fifo;
pub mod generate;
pub mod index;
pub mod manifest;
pub mod pack;
pub mod packed;
pub mod query;
pub mod rollup;
pub mod router;
pub mod schema;
pub mod skew;
pub mod stats;
pub mod symbols;
pub mod util;
pub mod writer;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "lance")]
pub mod lance_store;

pub use config::GenConfig;
pub use generate::generate;
pub use manifest::Manifest;
pub use stats::summarize;
