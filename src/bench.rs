//! Three-arm benchmark harness (M6) — the gate on the whole design (§5).
//!
//! Arms over the *same* packed storage, timed end-to-end:
//!   1. GPU pipeline (feature `gpu`) — with disk/H2D/kernel/D2H breakout.
//!   2. Vectorized CPU — packed table + checkpoints (this is what runs locally).
//!   3. Status-quo baseline — full Parquet re-scan ([`crate::baseline`]).
//!
//! Query matrix: {1 day, 1 month, random range, all-history} × {per-client
//! (whale + retail), cross-client}. CPU-packed PnL is asserted equal to the
//! baseline for every query (correctness), and observations feed the router fit.

use crate::baseline::baseline_query;
use crate::checkpoint::CheckpointStore;
use crate::config::GenConfig;
use crate::fifo::{NoopSink, PartitionPnl};
use crate::manifest::Manifest;
use crate::packed::PackedTable;
use crate::query::{run_cpu, ClientSel, Query, Span};
use crate::router::{self, Observation, RouterCoeffs};
use anyhow::Result;
use serde::Serialize;
use std::path::Path;
use std::time::Instant;

fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t0 = Instant::now();
    let r = f();
    (r, t0.elapsed().as_secs_f64() * 1e3)
}

fn span_label(s: &Span) -> String {
    match s {
        Span::Full => "all-history".into(),
        Span::Range(lo, hi) => format!("range[{lo},{hi}]"),
    }
}

fn fmt_pnl(p: &PartitionPnl) -> String {
    format!(
        "intra={:.0} short={:.0} long={:.0} (qty {}/{}/{})",
        p.intraday.value(),
        p.short.value(),
        p.long.value(),
        p.intraday.matched_qty,
        p.short.matched_qty,
        p.long.matched_qty
    )
}

#[derive(Serialize)]
struct BenchRow {
    name: String,
    span: String,
    rows_touched: u64,
    partition_fanout: u64,
    max_partition_rows: u64,
    checkpoints: u64,
    cpu_ms: f64,
    baseline_ms: f64,
    speedup_vs_baseline: f64,
    routed_engine: String,
    pnl_matches_baseline: bool,
}

/// Find the whale id with the most trades and a representative retail id.
fn pick_clients(table: &PackedTable, cfg: &GenConfig) -> (u64, u64) {
    let pc = table.part_client();
    let off = table.part_offset();
    // rows per client (partitions are contiguous per client)
    let mut best_whale = (0u64, 0u64);
    let mut retail = None;
    let mut p = 0usize;
    while p < pc.len() {
        let c = pc[p];
        let mut rows = 0u64;
        let q = p;
        let mut pp = p;
        while pp < pc.len() && pc[pp] == c {
            rows += off[pp + 1] - off[pp];
            pp += 1;
        }
        if cfg.is_whale(c) {
            if rows > best_whale.1 {
                best_whale = (c, rows);
            }
        } else if retail.is_none() && rows >= 4 {
            retail = Some(c);
        }
        p = pp;
        let _ = q;
    }
    (best_whale.0, retail.unwrap_or(pc[0]))
}

pub fn run_bench(
    tradebook_dir: &Path,
    packed_path: &Path,
    ckpt_dir: &Path,
    router_log: &Path,
) -> Result<()> {
    let manifest = Manifest::read(tradebook_dir)?;
    let cfg = &manifest.config;
    let table = PackedTable::open(packed_path)?;
    let td = cfg.trading_days();
    let n = td.len();

    // --- checkpoints: quarterly cutoffs so range queries get carry-in ---
    let store = CheckpointStore::new(ckpt_dir);
    let cutoffs: Vec<i32> = (1..4).map(|k| td[n * k / 4] as i32).collect();
    println!("Building {} checkpoints at cutoffs {:?} …", cutoffs.len(), cutoffs);
    let (_, ckpt_ms) = timed(|| store.build_periodic(&table, &cutoffs));
    println!("  checkpoints built in {:.0} ms", ckpt_ms);

    let (whale, retail) = pick_clients(&table, cfg);
    println!("Using whale client {whale}, retail client {retail}\n");

    let d = |i: usize| td[i.min(n - 1)] as i32;
    let mid = n / 2;
    let queries: Vec<(String, Query)> = vec![
        ("whale 1-day".into(), Query { clients: ClientSel::One(whale), symbol: None, span: Span::Range(d(mid), d(mid)) }),
        ("whale 1-month".into(), Query { clients: ClientSel::One(whale), symbol: None, span: Span::Range(d(mid), d(mid + 22)) }),
        ("whale random-range".into(), Query { clients: ClientSel::One(whale), symbol: None, span: Span::Range(d(n / 10), d(n * 7 / 10)) }),
        ("whale all-history".into(), Query { clients: ClientSel::One(whale), symbol: None, span: Span::Full }),
        ("retail 1-month".into(), Query { clients: ClientSel::One(retail), symbol: None, span: Span::Range(d(mid), d(mid + 22)) }),
        ("retail all-history".into(), Query { clients: ClientSel::One(retail), symbol: None, span: Span::Full }),
        ("cross-client all-history".into(), Query { clients: ClientSel::All, symbol: None, span: Span::Full }),
    ];

    let coeffs = RouterCoeffs::default();
    let mut rows_out: Vec<BenchRow> = Vec::new();
    let mut observations: Vec<Observation> = Vec::new();

    println!(
        "{:<26} {:>10} {:>8} {:>10} {:>10} {:>11} {:>9} {:>7}",
        "query", "rows", "parts", "maxpart", "cpu(ms)", "base(ms)", "speedup", "route"
    );
    println!("{}", "─".repeat(103));

    for (name, q) in &queries {
        let (cpu_res, cpu_ms) = timed(|| run_cpu(&table, Some(&store), q, &mut NoopSink).unwrap());
        let ((base_pnl, _base_rows), base_ms) = timed(|| baseline_query(tradebook_dir, q).unwrap());
        let matches = cpu_res.pnl == base_pnl;

        let pred = coeffs.route(
            cpu_res.rows_touched,
            cpu_res.partition_fanout,
            cpu_res.max_partition_rows,
            cpu_res.checkpoints_loaded,
        );
        let speedup = if cpu_ms > 0.0 { base_ms / cpu_ms } else { 0.0 };

        println!(
            "{:<26} {:>10} {:>8} {:>10} {:>10.2} {:>11.2} {:>8.1}x {:>7}",
            name,
            cpu_res.rows_touched,
            cpu_res.partition_fanout,
            cpu_res.max_partition_rows,
            cpu_ms,
            base_ms,
            speedup,
            format!("{:?}", pred.engine)
        );
        if !matches {
            println!("    ⚠ PnL MISMATCH vs baseline!");
            println!("      cpu : {}", fmt_pnl(&cpu_res.pnl));
            println!("      base: {}", fmt_pnl(&base_pnl));
        }

        observations.push(Observation {
            rows: cpu_res.rows_touched,
            fanout: cpu_res.partition_fanout,
            max_part: cpu_res.max_partition_rows,
            checkpoints: cpu_res.checkpoints_loaded,
            cpu_ns: cpu_ms * 1e6,
            gpu_ns: None,
        });
        rows_out.push(BenchRow {
            name: name.clone(),
            span: span_label(&q.span),
            rows_touched: cpu_res.rows_touched,
            partition_fanout: cpu_res.partition_fanout,
            max_partition_rows: cpu_res.max_partition_rows,
            checkpoints: cpu_res.checkpoints_loaded,
            cpu_ms,
            baseline_ms: base_ms,
            speedup_vs_baseline: speedup,
            routed_engine: format!("{:?}", pred.engine),
            pnl_matches_baseline: matches,
        });
    }

    // --- GPU arm (only with --features gpu on an NVIDIA box) ---
    run_gpu_arm(&table)?;

    // --- router fit ---
    let fitted = router::fit(&observations);
    println!("\nRouter coefficients (fitted from CPU observations):");
    println!("  cpu_per_row_ns      = {:.3}", fitted.cpu_per_row_ns);
    println!("  gpu_per_row_ns      = {:.3}", fitted.gpu_per_row_ns);
    println!("  h2d_per_row_ns      = {:.3}", fitted.h2d_per_row_ns);
    println!("  launch_per_part_ns  = {:.1}", fitted.launch_per_partition_ns);
    println!("  gpu_serial_per_row  = {:.5}", fitted.gpu_serial_per_row_ns);
    for (obs, row) in observations.iter().zip(rows_out.iter()) {
        let pred = fitted.route(obs.rows, obs.fanout, obs.max_part, obs.checkpoints);
        let actual_ns = row.cpu_ms * 1e6;
        let _ = router::log_pred_vs_actual(
            router_log,
            &router::PredVsActual {
                rows: obs.rows,
                fanout: obs.fanout,
                max_part: obs.max_part,
                checkpoints: obs.checkpoints,
                chosen: pred.engine,
                predicted_ns: pred.cpu_ns.min(pred.gpu_ns),
                actual_ns,
            },
        );
    }
    println!("  predicted-vs-actual logged to {}", router_log.display());

    let out_json = packed_path.with_extension("bench.json");
    std::fs::write(&out_json, serde_json::to_string_pretty(&rows_out)?)?;
    println!("Bench results written to {}", out_json.display());
    Ok(())
}

#[cfg(feature = "gpu")]
fn run_gpu_arm(table: &PackedTable) -> Result<()> {
    use crate::fifo::fold_table;
    use crate::gpu::GpuEngine;
    println!("\n── GPU arm (all-history full-table fold) ────────────────────");
    let eng = GpuEngine::new(0)?;
    let (gpu_pnl, t) = eng.fold_total(table)?;
    let (cpu_pnl, cpu_ms) = timed(|| fold_table(table, &mut NoopSink));
    println!(
        "  disk+H2D: {:.1} ms | kernel: {:.1} ms | D2H: {:.1} ms | total: {:.1} ms",
        t.h2d_ms, t.kernel_ms, t.d2h_ms, t.total_ms
    );
    println!("  CPU full-table fold: {:.1} ms", cpu_ms);
    // validate: matched_qty exact, realized PnL within f64 tolerance
    let q_match = gpu_pnl.intraday.matched_qty == cpu_pnl.intraday.matched_qty
        && gpu_pnl.short.matched_qty == cpu_pnl.short.matched_qty
        && gpu_pnl.long.matched_qty == cpu_pnl.long.matched_qty;
    let rel = |a: i128, b: i128| {
        let (a, b) = (a as f64, b as f64);
        if b.abs() < 1.0 { (a - b).abs() } else { ((a - b) / b).abs() }
    };
    let pnl_ok = rel(gpu_pnl.intraday.realized_ticks, cpu_pnl.intraday.realized_ticks) < 1e-6
        && rel(gpu_pnl.short.realized_ticks, cpu_pnl.short.realized_ticks) < 1e-6
        && rel(gpu_pnl.long.realized_ticks, cpu_pnl.long.realized_ticks) < 1e-6;
    println!(
        "  validation vs CPU oracle: matched_qty {}, realized {}",
        if q_match { "EXACT ✓" } else { "MISMATCH ✗" },
        if pnl_ok { "within 1e-6 ✓" } else { "DIVERGES ✗" }
    );
    println!("  {}", fmt_pnl(&gpu_pnl));
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn run_gpu_arm(_table: &PackedTable) -> Result<()> {
    println!("\n── GPU arm skipped (build with --features gpu on an NVIDIA box) ──");
    Ok(())
}
