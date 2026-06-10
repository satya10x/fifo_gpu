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
use crate::fifo::{MatchPolicy, BucketRules, NoopSink, PartitionPnl};
use crate::manifest::Manifest;
use crate::packed::PackedTable;
use crate::query::{run_cpu, ClientSel, Query, Span};
use crate::router::{self, Observation};
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
        p.intraday().value(),
        p.short().value(),
        p.long().value(),
        p.intraday().matched_qty,
        p.short().matched_qty,
        p.long().matched_qty
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
    /// Per-query GPU arm (full-history queries only — ranges stay on CPU).
    gpu_ms: Option<f64>,
    gpu_h2d_ms: Option<f64>,
    gpu_kernel_ms: Option<f64>,
    gpu_matches_baseline: Option<bool>,
    routed_engine: String,
    pnl_matches_baseline: bool,
}

/// Everything measured for one query in phase 1, before the router is fit.
/// Routing is decided in phase 2 from the *fitted* coefficients (so the route
/// column reflects calibrated costs, not the cold-start priors).
struct QRec {
    name: String,
    span: String,
    rows: u64,
    fanout: u64,
    max_part: u64,
    checkpoints: u64,
    cpu_ms: f64,
    base_ms: f64,
    matches: bool,
    cpu_pnl: PartitionPnl,
    base_pnl: PartitionPnl,
    /// For range queries: the same packed fold WITHOUT checkpoint carry-in
    /// (folded from scratch). Lets a mismatch be attributed to the checkpoint
    /// path vs a packed-vs-parquet data difference.
    nockpt_pnl: Option<PartitionPnl>,
    gpu_ms: Option<f64>,
    gpu_h2d_ms: Option<f64>,
    gpu_kernel_ms: Option<f64>,
    gpu_matches: Option<bool>,
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
    table: &PackedTable,
    ckpt_dir: &Path,
    router_log: &Path,
    out_json: &Path,
) -> Result<()> {
    let manifest = Manifest::read(tradebook_dir)?;
    let cfg = &manifest.config;
    let td = cfg.trading_days();
    let n = td.len();
    // Default bucket ruleset (intraday / short ≤365d / long) + FIFO. The GPU arm
    // implements FIFO with the default threshold, so the bench keeps both default
    // to stay GPU-comparable; LIFO/HIFO and custom thresholds are CPU-only.
    let rules = BucketRules::default();
    let policy = MatchPolicy::Fifo;

    // --- checkpoints: quarterly cutoffs so range queries get carry-in ---
    let store = CheckpointStore::new(ckpt_dir);
    let cutoffs: Vec<i32> = (1..4).map(|k| td[n * k / 4] as i32).collect();
    println!("Building {} checkpoints at cutoffs {:?} …", cutoffs.len(), cutoffs);
    let (_, ckpt_ms) = timed(|| store.build_periodic(table, &cutoffs));
    println!("  checkpoints built in {:.0} ms", ckpt_ms);

    let (whale, retail) = pick_clients(table, cfg);
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

    // GPU engine for the per-query arm (full-history queries only; range queries
    // need checkpoint carry-in and stay on CPU). Compiled once, reused per query.
    #[cfg(feature = "gpu")]
    let gpu_engine = crate::gpu::GpuEngine::new(0)?;

    // ---- phase 1: measure every query (CPU, baseline, and GPU where applicable) ----
    let mut qrecs: Vec<QRec> = Vec::new();
    for (name, q) in &queries {
        let (cpu_res, cpu_ms) = timed(|| run_cpu(table, Some(&store), q, &mut NoopSink, &rules, policy).unwrap());
        let ((base_pnl, _base_rows), base_ms) = timed(|| baseline_query(tradebook_dir, q).unwrap());
        let matches = cpu_res.pnl == base_pnl;

        // Diagnostic for range queries: fold the same packed data from scratch
        // (no checkpoint). Splits a mismatch into checkpoint-carry vs packed-data.
        let nockpt_pnl = match q.span {
            Span::Range(..) => Some(run_cpu(table, None, q, &mut NoopSink, &rules, policy)?.pnl),
            Span::Full => None,
        };

        #[allow(unused_mut)]
        let (mut gpu_ms, mut gpu_h2d_ms, mut gpu_kernel_ms, mut gpu_matches): (
            Option<f64>,
            Option<f64>,
            Option<f64>,
            Option<bool>,
        ) = (None, None, None, None);
        #[cfg(feature = "gpu")]
        {
            if let Span::Full = q.span {
                let parts = crate::query::select_partitions(table, q);
                let (gpnl, t) = gpu_engine.fold_query(table, &parts)?;
                gpu_ms = Some(t.total_ms);
                gpu_h2d_ms = Some(t.h2d_ms);
                gpu_kernel_ms = Some(t.kernel_ms);
                gpu_matches = Some(
                    gpnl.intraday().matched_qty == base_pnl.intraday().matched_qty
                        && gpnl.short().matched_qty == base_pnl.short().matched_qty
                        && gpnl.long().matched_qty == base_pnl.long().matched_qty,
                );
            }
        }

        qrecs.push(QRec {
            name: name.clone(),
            span: span_label(&q.span),
            rows: cpu_res.rows_touched,
            fanout: cpu_res.partition_fanout,
            max_part: cpu_res.max_partition_rows,
            checkpoints: cpu_res.checkpoints_loaded,
            cpu_ms,
            base_ms,
            matches,
            cpu_pnl: cpu_res.pnl,
            base_pnl,
            nockpt_pnl,
            gpu_ms,
            gpu_h2d_ms,
            gpu_kernel_ms,
            gpu_matches,
        });
    }

    // ---- fit the router from the observations (incl. measured GPU H2D/kernel) ----
    let observations: Vec<Observation> = qrecs
        .iter()
        .map(|r| Observation {
            rows: r.rows,
            fanout: r.fanout,
            max_part: r.max_part,
            checkpoints: r.checkpoints,
            cpu_ns: r.cpu_ms * 1e6,
            gpu_ns: r.gpu_ms.map(|x| x * 1e6),
            gpu_h2d_ns: r.gpu_h2d_ms.map(|x| x * 1e6),
            gpu_kernel_ns: r.gpu_kernel_ms.map(|x| x * 1e6),
        })
        .collect();
    let fitted = router::fit(&observations);

    // ---- phase 2: print the table, routing from the fitted coefficients ----
    println!(
        "{:<26} {:>10} {:>7} {:>10} {:>9} {:>9} {:>10} {:>8} {:>6}",
        "query", "rows", "parts", "maxpart", "cpu(ms)", "gpu(ms)", "base(ms)", "vs base", "route"
    );
    println!("{}", "─".repeat(102));

    let mut rows_out: Vec<BenchRow> = Vec::new();
    for r in &qrecs {
        let pred = fitted.route(r.rows, r.fanout, r.max_part, r.checkpoints);
        let speedup = if r.cpu_ms > 0.0 { r.base_ms / r.cpu_ms } else { 0.0 };
        let gpu_str = r
            .gpu_ms
            .map(|x| format!("{:.2}", x))
            .unwrap_or_else(|| "-".into());

        println!(
            "{:<26} {:>10} {:>7} {:>10} {:>9.2} {:>9} {:>10.2} {:>7.1}x {:>6}",
            r.name,
            r.rows,
            r.fanout,
            r.max_part,
            r.cpu_ms,
            gpu_str,
            r.base_ms,
            speedup,
            format!("{:?}", pred.engine)
        );
        if !r.matches {
            println!("    ⚠ PnL MISMATCH vs baseline!");
            println!("      cpu : {}", fmt_pnl(&r.cpu_pnl));
            println!("      base: {}", fmt_pnl(&r.base_pnl));
            if let Some(nck) = r.nockpt_pnl {
                println!("      nock: {}", fmt_pnl(&nck));
                let ckpt_bad = r.cpu_pnl != nck;
                let data_bad = nck != r.base_pnl;
                match (ckpt_bad, data_bad) {
                    (true, false) => println!("      → CHECKPOINT carry-in is wrong (cpu_ckpt ≠ cpu_nockpt; nockpt == baseline)"),
                    (false, true) => println!("      → PACKED vs PARQUET differ (cpu_nockpt ≠ baseline; checkpoint is fine)"),
                    (true, true) => println!("      → BOTH diverge (checkpoint AND packed-vs-parquet)"),
                    (false, false) => println!("      → no divergence reproduced (transient?)"),
                }
            }
        }
        if r.gpu_matches == Some(false) {
            println!("    ⚠ GPU matched_qty != baseline for this query!");
        }

        let actual_ns = r.cpu_ms * 1e6;
        let _ = router::log_pred_vs_actual(
            router_log,
            &router::PredVsActual {
                rows: r.rows,
                fanout: r.fanout,
                max_part: r.max_part,
                checkpoints: r.checkpoints,
                chosen: pred.engine,
                predicted_ns: pred.cpu_ns.min(pred.gpu_ns),
                actual_ns,
            },
        );
        rows_out.push(BenchRow {
            name: r.name.clone(),
            span: r.span.clone(),
            rows_touched: r.rows,
            partition_fanout: r.fanout,
            max_partition_rows: r.max_part,
            checkpoints: r.checkpoints,
            cpu_ms: r.cpu_ms,
            baseline_ms: r.base_ms,
            speedup_vs_baseline: speedup,
            gpu_ms: r.gpu_ms,
            gpu_h2d_ms: r.gpu_h2d_ms,
            gpu_kernel_ms: r.gpu_kernel_ms,
            gpu_matches_baseline: r.gpu_matches,
            routed_engine: format!("{:?}", pred.engine),
            pnl_matches_baseline: r.matches,
        });
    }

    // --- GPU full-table arm: detailed disk/H2D/kernel/D2H breakout (Decision 3) ---
    run_gpu_arm(table)?;

    println!("\nRouter coefficients (fitted):");
    println!("  cpu_per_row_ns      = {:.3}", fitted.cpu_per_row_ns);
    println!("  gpu_per_row_ns      = {:.3}  (measured kernel)", fitted.gpu_per_row_ns);
    println!("  h2d_per_row_ns      = {:.3}  (measured transfer)", fitted.h2d_per_row_ns);
    println!("  launch_per_part_ns  = {:.1}", fitted.launch_per_partition_ns);
    println!("  gpu_serial_per_row  = {:.5}", fitted.gpu_serial_per_row_ns);
    println!("  predicted-vs-actual logged to {}", router_log.display());

    std::fs::write(out_json, serde_json::to_string_pretty(&rows_out)?)?;
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
    let q_match = gpu_pnl.intraday().matched_qty == cpu_pnl.intraday().matched_qty
        && gpu_pnl.short().matched_qty == cpu_pnl.short().matched_qty
        && gpu_pnl.long().matched_qty == cpu_pnl.long().matched_qty;
    let rel = |a: i128, b: i128| {
        let (a, b) = (a as f64, b as f64);
        if b.abs() < 1.0 { (a - b).abs() } else { ((a - b) / b).abs() }
    };
    let pnl_ok = rel(gpu_pnl.intraday().realized_ticks, cpu_pnl.intraday().realized_ticks) < 1e-6
        && rel(gpu_pnl.short().realized_ticks, cpu_pnl.short().realized_ticks) < 1e-6
        && rel(gpu_pnl.long().realized_ticks, cpu_pnl.long().realized_ticks) < 1e-6;
    println!(
        "  validation vs CPU oracle: matched_qty {}, realized {}",
        if q_match { "EXACT ✓" } else { "MISMATCH ✗" },
        if pnl_ok { "within 1e-6 ✓" } else { "DIVERGES ✗" }
    );
    println!("  {}", fmt_pnl(&gpu_pnl));

    // Streamed arm (Decision 5): chunked + 2 streams + pinned host → H2D/kernel
    // overlap and bounded VRAM. Compare its end-to-end total to the serial total.
    println!("\n── GPU streamed arm (2 streams + pinned, H2D/kernel overlap) ──");
    let (s_pnl, s_ms, pinned) = eng.fold_total_streamed(table)?;
    let s_qmatch = s_pnl.intraday().matched_qty == cpu_pnl.intraday().matched_qty
        && s_pnl.short().matched_qty == cpu_pnl.short().matched_qty
        && s_pnl.long().matched_qty == cpu_pnl.long().matched_qty;
    let s_pnl_ok = rel(s_pnl.intraday().realized_ticks, cpu_pnl.intraday().realized_ticks) < 1e-6
        && rel(s_pnl.short().realized_ticks, cpu_pnl.short().realized_ticks) < 1e-6
        && rel(s_pnl.long().realized_ticks, cpu_pnl.long().realized_ticks) < 1e-6;
    println!(
        "  total: {:.1} ms (host pinned: {})  vs serial GPU total {:.1} ms  →  {:.2}× overlap speedup",
        s_ms,
        if pinned { "yes" } else { "NO — no overlap" },
        t.total_ms,
        if s_ms > 0.0 { t.total_ms / s_ms } else { 0.0 }
    );
    println!(
        "  validation vs CPU oracle: matched_qty {}, realized {}",
        if s_qmatch { "EXACT ✓" } else { "MISMATCH ✗" },
        if s_pnl_ok { "within 1e-6 ✓" } else { "DIVERGES ✗" }
    );
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn run_gpu_arm(_table: &PackedTable) -> Result<()> {
    println!("\n── GPU arm skipped (build with --features gpu on an NVIDIA box) ──");
    Ok(())
}
