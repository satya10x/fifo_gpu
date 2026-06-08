//! The trade generator: per-client inventory random walk producing realistic
//! round trips that exercise all three FIFO buckets (intraday / short / long).
//!
//! Ordering guarantee: clients are emitted in id order; within a client,
//! symbols ascending; within a symbol, timestamps strictly increasing. So the
//! output stream is globally `(client, symbol, ts)`-sorted — the exact
//! clustering the compute table wants.

use crate::calendar::{self, SESSION_US};
use crate::config::GenConfig;
use crate::manifest::Manifest;
use crate::schema::Trade;
use crate::skew;
use crate::symbols::SymbolUniverse;
use crate::util::{mix_seed, stream, SALT_TRADE};
use crate::writer::{write_symbols, TradebookWriter};
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use rand_distr::{Distribution, LogNormal, Normal};

pub const TICK: f64 = 0.05;

#[inline]
fn tick_round(p: f64) -> f64 {
    (p.max(TICK) / TICK).round() * TICK
}

/// Where generated trades go — the Parquet writer, or an in-memory Vec (used by
/// the correction path to regenerate a single client's slice).
pub trait TradeSink {
    fn push_trade(&mut self, t: &Trade) -> Result<()>;
}

impl TradeSink for TradebookWriter {
    fn push_trade(&mut self, t: &Trade) -> Result<()> {
        self.push(t)
    }
}

impl TradeSink for Vec<Trade> {
    fn push_trade(&mut self, t: &Trade) -> Result<()> {
        self.push(*t);
        Ok(())
    }
}

/// Deterministically regenerate every trade for one client (correction path,
/// M8). Pure function of `(cfg, client_id)` given the skew plan — the basis for
/// regenerating just the dirtied partition without touching anyone else.
pub fn regenerate_client(cfg: &GenConfig, client_id: u64) -> Vec<Trade> {
    let cal = cfg.trading_days();
    let symbols = SymbolUniverse::new(cfg.symbols);
    let plan = skew::build_plan(cfg);
    let count = skew::client_count(cfg, &plan, client_id);
    let mut out: Vec<Trade> = Vec::new();
    let mut trade_id = 0u64; // local ids; correction reassigns globally on repack
    gen_client(cfg, &symbols, &cal, client_id, count, &mut out, &mut trade_id)
        .expect("Vec sink never errors");
    out
}

/// Run the full generation, writing the tradebook + symbols + manifest.
pub fn generate(cfg: &GenConfig) -> Result<Manifest> {
    let cal = cfg.trading_days();
    let symbols = SymbolUniverse::new(cfg.symbols);
    let plan = skew::build_plan(cfg);

    let mut writer = TradebookWriter::new(&cfg.out_dir, cfg.rows_per_file, cfg.batch_rows)?;
    let mut trade_id: u64 = 0;

    let pb = ProgressBar::new(cfg.clients);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner} {pos}/{len} clients [{elapsed_precise}] {msg}",
        )
        .unwrap(),
    );
    pb.set_draw_target(indicatif::ProgressDrawTarget::stderr_with_hz(4));

    for id in 0..cfg.clients {
        let count = skew::client_count(cfg, &plan, id);
        gen_client(cfg, &symbols, &cal, id, count, &mut writer, &mut trade_id)?;
        if id % 4096 == 0 {
            pb.set_position(id);
        }
    }
    pb.finish_and_clear();

    let total = writer.finish()?;
    write_symbols(&cfg.out_dir, &symbols)?;

    debug_assert_eq!(total, trade_id);
    let manifest = Manifest::new(cfg, &plan, total);
    manifest.write(&cfg.out_dir)?;
    Ok(manifest)
}

/// Heuristic: more trades → more distinct symbols, capped.
fn symbol_count_for(count: u64) -> u32 {
    let n = (count as f64).powf(0.4).round() as u32;
    n.clamp(1, 50)
}

fn gen_client<W: TradeSink>(
    cfg: &GenConfig,
    symbols: &SymbolUniverse,
    cal: &[i64],
    client_id: u64,
    count: u64,
    writer: &mut W,
    trade_id: &mut u64,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    let mut r = stream(mix_seed(cfg.seed, client_id, SALT_TRADE));

    let n_sym = symbol_count_for(count).min(count as u32);
    let syms = symbols.sample_distinct(&mut r, n_sym);
    let alloc = split_count(&mut r, count, syms.len() as u32);

    for (sym, m) in syms.iter().zip(alloc.iter()) {
        gen_symbol(cfg, symbols, cal, &mut r, client_id, *sym, *m, writer, trade_id)?;
    }
    Ok(())
}

/// Split `count` across `n` symbols by random (Exp) weights, each ≥ 1, summing
/// exactly to `count`.
fn split_count<R: Rng>(rng: &mut R, count: u64, n: u32) -> Vec<u64> {
    let n = n.max(1) as usize;
    if n == 1 {
        return vec![count];
    }
    let weights: Vec<f64> = (0..n).map(|_| -rng.gen::<f64>().max(1e-12).ln()).collect();
    let wsum: f64 = weights.iter().sum();
    let mut alloc: Vec<u64> = weights
        .iter()
        .map(|w| ((count as f64) * w / wsum).floor().max(1.0) as u64)
        .collect();
    // Fix rounding so the sum is exactly `count`.
    let mut assigned: u64 = alloc.iter().sum();
    let mut i = 0usize;
    while assigned < count {
        alloc[i % n] += 1;
        assigned += 1;
        i += 1;
    }
    while assigned > count {
        // shave from the largest bucket that is > 1
        let (idx, _) = alloc
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 1)
            .max_by_key(|(_, &v)| v)
            .unwrap_or((0, &alloc[0]));
        alloc[idx] -= 1;
        assigned -= 1;
    }
    alloc
}

/// Heavy-tailed gap (in trading days) to the next trade — mostly a few days,
/// occasionally a long hold so the long-term bucket gets exercised.
fn sample_gap_days<R: Rng>(rng: &mut R, ndays: i64) -> i64 {
    let u = rng.gen::<f64>();
    if u < 0.03 && ndays > 3 {
        // long hold spanning a big fraction of the range
        rng.gen_range((ndays / 3).max(1)..ndays)
    } else {
        // 1 + geometric(p=0.25), capped
        let g = (rng.gen::<f64>().max(1e-12).ln() / (0.75f64).ln()).floor() as i64;
        (1 + g).clamp(1, 30)
    }
}

#[inline]
fn p_sell(inv: u64, avg_qty: f64) -> f64 {
    let push = (inv as f64 / (avg_qty * 5.0)).min(1.0) * 0.4;
    (0.5 + push).min(0.92)
}

#[allow(clippy::too_many_arguments)]
fn gen_symbol<R: Rng, W: TradeSink>(
    _cfg: &GenConfig,
    symbols: &SymbolUniverse,
    cal: &[i64],
    r: &mut R,
    client_id: u64,
    symbol_id: u32,
    m: u64,
    writer: &mut W,
    trade_id: &mut u64,
) -> Result<()> {
    if m == 0 {
        return Ok(());
    }
    let base = symbols.base_price(symbol_id);
    let dvol = symbols.daily_vol(symbol_id);
    let lo_log = base.ln() - 2.0; // clamp price to ~[0.13x, 7.4x] of base
    let hi_log = base.ln() + 2.0;
    let mut log_px = base.ln();

    let qty_dist = LogNormal::new(_cfg.qty_mu, _cfg.qty_sigma).expect("valid lognormal");
    let avg_qty = (_cfg.qty_mu + _cfg.qty_sigma * _cfg.qty_sigma / 2.0).exp();
    let norm = Normal::new(0.0, 1.0).expect("valid normal");

    let ndays = cal.len() as i64;
    let mut day_pos: i64 = r.gen_range(0..ndays);
    let mut prev_day_pos = day_pos;
    let mut last_z = i64::MIN;
    let mut last_off: i64 = 0;
    let mut inv: u64 = 0;

    for k in 0..m {
        if k > 0 {
            if r.gen::<f64>() < _cfg.p_intraday {
                // same day — don't advance day_pos
            } else {
                let gap = sample_gap_days(r, ndays);
                day_pos = (day_pos + gap).min(ndays - 1);
            }
        }

        // Price random walk; variance grows with elapsed days, then clamp.
        let dt = (day_pos - prev_day_pos).max(0) as f64;
        log_px += norm.sample(r) * dvol * (dt + 1.0).sqrt();
        log_px = log_px.clamp(lo_log, hi_log);
        prev_day_pos = day_pos;
        let price = tick_round(log_px.exp());

        // Strictly increasing intraday offset within a calendar day.
        let z = cal[day_pos as usize];
        let off = if z != last_z {
            r.gen_range(0..(SESSION_US / 2))
        } else {
            (last_off + r.gen_range(1..=(SESSION_US / 8))).min(SESSION_US - 1)
        };
        last_z = z;
        last_off = off;
        let ts_micros = calendar::ts_micros(z, off);

        // Side driven by inventory: buy when flat, else probabilistic sell that
        // grows with inventory so positions unwind (→ matched FIFO fragments).
        let side: i8 = if inv == 0 {
            1
        } else if r.gen::<f64>() < p_sell(inv, avg_qty) {
            -1
        } else {
            1
        };
        let qraw = (qty_dist.sample(r).round() as i64).max(1) as u64;
        let qty = if side < 0 { qraw.min(inv) } else { qraw };
        if side > 0 {
            inv += qty;
        } else {
            inv -= qty;
        }

        let notional = qty as f64 * price;
        let fees = (notional * 0.0003 + 0.35) as f32;
        let venue = (r.gen::<u32>() % 5) as u8;

        writer.push_trade(&Trade {
            trade_id: *trade_id,
            client_id,
            symbol_id,
            side,
            quantity: qty as u32,
            price,
            ts_micros,
            venue,
            fees,
        })?;
        *trade_id += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_count_sums_exactly() {
        let mut r = stream(7);
        for &(c, n) in &[(100u64, 5u32), (3, 4), (1, 1), (1_000_000, 50), (7, 3)] {
            let a = split_count(&mut r, c, n);
            assert_eq!(a.iter().sum::<u64>(), c, "count {c} n {n}");
            assert!(a.iter().all(|&v| v >= 1) || c < n as u64);
        }
    }

    #[test]
    fn tick_round_is_on_grid() {
        for p in [10.0, 10.02, 10.04, 99.99, 0.001] {
            let r = tick_round(p);
            let ticks = (r / TICK).round();
            assert!((r - ticks * TICK).abs() < 1e-9);
            assert!(r >= TICK);
        }
    }
}
