//! `fifo` — CLI for the GPU/CPU FIFO-PnL system. M1 subcommands: `gen`, `stats`.
//! Later milestones add `pack`, `fold`, `query`, `checkpoint`, `rollup`, `bench`.

use anyhow::Result;
use clap::{Parser, Subcommand};
use fifo_gpu::checkpoint::CheckpointStore;
use fifo_gpu::config::GenConfig;
use fifo_gpu::fifo::{fold_table, PartitionPnl};
use fifo_gpu::manifest::Manifest;
use fifo_gpu::packed::PackedTable;
use fifo_gpu::query::{run_cpu, ClientSel, Query, Span};
use fifo_gpu::rollup::Rollup;
use fifo_gpu::{calendar, correction};
use std::path::PathBuf;

fn print_pnl(p: &PartitionPnl) {
    println!(
        "  intraday : {:>16.2}   (qty {})",
        p.intraday.value(),
        p.intraday.matched_qty
    );
    println!(
        "  short    : {:>16.2}   (qty {})",
        p.short.value(),
        p.short.matched_qty
    );
    println!(
        "  long     : {:>16.2}   (qty {})",
        p.long.value(),
        p.long.matched_qty
    );
    println!(
        "  TOTAL    : {:>16.2}",
        (p.total_ticks() as f64) * fifo_gpu::generate::TICK
    );
}

#[derive(Parser)]
#[command(name = "fifo", about = "GPU/CPU FIFO PnL on a Lance-backed store")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate a skew-realistic synthetic tradebook (M1).
    Gen(GenArgs),
    /// Summarize the trade-count-per-client skew of a generated dataset (M1).
    Stats(StatsArgs),
    /// Pack the tradebook into the transparent compute table + page index (M2).
    Pack(PackArgs),
    /// Build periodic checkpoints for bounded range queries (M4).
    Checkpoint(CheckpointArgs),
    /// Run a FIFO PnL query (per-client or cross-client, full or range).
    Query(QueryArgs),
    /// Build the cross-client per-period rollup (M8 / Option C).
    Rollup(RollupArgs),
    /// Run a back-dated correction for one client (M8 correction path).
    Correct(CorrectArgs),
    /// Run the three-arm benchmark (M6): CPU packed vs status-quo (vs GPU).
    Bench(BenchArgs),
}

#[derive(Parser)]
struct CheckpointArgs {
    #[arg(long, default_value = "data/compute.fifopack")]
    packed: PathBuf,
    #[arg(long, default_value = "data/tradebook")]
    tradebook: PathBuf,
    #[arg(long, default_value = "data/checkpoints")]
    out: PathBuf,
    /// Number of evenly-spaced cutoffs across the date range.
    #[arg(long, default_value_t = 4)]
    cutoffs: usize,
}

#[derive(Parser)]
struct QueryArgs {
    #[arg(long, default_value = "data/compute.fifopack")]
    packed: PathBuf,
    /// Client id; omit for a cross-client query.
    #[arg(long)]
    client: Option<u64>,
    /// Symbol id filter (optional).
    #[arg(long)]
    symbol: Option<u32>,
    /// Range start YYYY-MM-DD (omit both --from/--to for all-history).
    #[arg(long)]
    from: Option<String>,
    /// Range end YYYY-MM-DD inclusive.
    #[arg(long)]
    to: Option<String>,
    /// Checkpoint dir (used for range carry-in).
    #[arg(long, default_value = "data/checkpoints")]
    checkpoints: PathBuf,
    /// Holding-period (days) at/below which a round-trip is short-term; above is
    /// long-term. Jurisdiction-configurable (e.g. 365). Intraday (same-day) is
    /// always its own bucket. See DESIGN.md (Axis 1).
    #[arg(long, default_value_t = fifo_gpu::fifo::LONG_TERM_DAYS)]
    ltcg_days: i32,
    /// Lot-matching policy: `fifo` | `lifo` | `hifo`. See DESIGN.md (Axis 2).
    /// Non-FIFO is CPU-only and (for range queries) needs policy-matched
    /// checkpoints; full-history is fine on any policy.
    #[arg(long, default_value = "fifo")]
    policy: String,
}

#[derive(Parser)]
struct RollupArgs {
    #[arg(long, default_value = "data/compute.fifopack")]
    packed: PathBuf,
    #[arg(long, default_value = "data/rollup.json")]
    out: PathBuf,
}

#[derive(Parser)]
struct CorrectArgs {
    #[arg(long, default_value = "data/tradebook")]
    tradebook: PathBuf,
    #[arg(long, default_value = "data/compute.fifopack")]
    packed: PathBuf,
    #[arg(long, default_value = "data/checkpoints")]
    checkpoints: PathBuf,
    /// Client whose trade was corrected.
    #[arg(long)]
    client: u64,
    /// Date of the corrected/back-dated trade (YYYY-MM-DD).
    #[arg(long)]
    on: String,
}

#[derive(Parser)]
struct BenchArgs {
    #[arg(long, default_value = "data/tradebook")]
    tradebook: PathBuf,
    #[arg(long, default_value = "data/compute.fifopack")]
    packed: PathBuf,
    #[arg(long, default_value = "data/checkpoints")]
    checkpoints: PathBuf,
    #[arg(long, default_value = "data/router-log.jsonl")]
    router_log: PathBuf,
}

#[derive(Parser)]
struct GenArgs {
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 5_000)]
    clients: u64,
    #[arg(long, default_value = "2020-01-01")]
    start: String,
    #[arg(long, default_value_t = 250)]
    days: u32,
    #[arg(long, default_value_t = 1_000)]
    symbols: u32,
    #[arg(long, default_value_t = 6.0)]
    mean_per_day: f64,
    #[arg(long, default_value_t = 8)]
    whales: u64,
    #[arg(long, default_value_t = 0.35)]
    p_intraday: f64,
    #[arg(long, default_value = "data/tradebook")]
    out: PathBuf,
    #[arg(long, default_value_t = 8_000_000)]
    rows_per_file: usize,
    #[arg(long, default_value_t = 1_000_000)]
    batch_rows: usize,
}

impl GenArgs {
    fn to_config(&self) -> Result<GenConfig> {
        let mut c = GenConfig::defaults();
        c.seed = self.seed;
        c.clients = self.clients;
        c.start_day = calendar::parse_date(&self.start)?;
        c.days = self.days;
        c.symbols = self.symbols;
        c.mean_per_day = self.mean_per_day;
        c.n_whales = self.whales;
        c.p_intraday = self.p_intraday;
        c.out_dir = self.out.clone();
        c.rows_per_file = self.rows_per_file;
        c.batch_rows = self.batch_rows;
        Ok(c)
    }
}

#[derive(Parser)]
struct PackArgs {
    /// Tradebook directory (the M1 output).
    #[arg(long, default_value = "data/tradebook")]
    tradebook: PathBuf,
    /// Output packed compute table path.
    #[arg(long, default_value = "data/compute.fifopack")]
    out: PathBuf,
    /// Records per page (8 MiB ≈ 262144 × 32 B).
    #[arg(long, default_value_t = fifo_gpu::index::PAGE_RECORDS)]
    page_records: usize,
}

#[derive(Parser)]
struct StatsArgs {
    /// Dataset directory containing manifest.json.
    #[arg(long, default_value = "data/tradebook")]
    out: PathBuf,
    /// Also scan the Parquet files and confirm the realized row count.
    #[arg(long, default_value_t = false)]
    verify_parquet: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Gen(args) => {
            let cfg = args.to_config()?;
            println!(
                "Generating {} clients × ~{:.0} trades/day over {} trading days …",
                cfg.clients, cfg.mean_per_day, cfg.days
            );
            let t0 = std::time::Instant::now();
            let manifest = fifo_gpu::generate::generate(&cfg)?;
            let dt = t0.elapsed();
            println!(
                "Wrote {} trades to {} in {:.2}s ({:.1}M trades/s)",
                manifest.realized_total,
                cfg.out_dir.display(),
                dt.as_secs_f64(),
                manifest.realized_total as f64 / dt.as_secs_f64() / 1e6
            );
            println!("Range: {} … {}", manifest.start_date, manifest.end_date_inclusive);
            fifo_gpu::stats::summarize(&cfg).print();
        }
        Cmd::Pack(args) => {
            println!("Packing {} → {} …", args.tradebook.display(), args.out.display());
            let t0 = std::time::Instant::now();
            let st = fifo_gpu::pack::pack(&args.tradebook, &args.out, args.page_records)?;
            let dt = t0.elapsed();
            println!(
                "Packed {} rows into {} partitions, {} pages, {:.1} MiB in {:.2}s",
                st.n_rows,
                st.n_parts,
                st.n_pages,
                st.bytes as f64 / (1 << 20) as f64,
                dt.as_secs_f64()
            );
            println!(
                "  transparent fixed-width buffer — mmap'd records DMA to GPU with no decompress"
            );
        }
        Cmd::Checkpoint(args) => {
            let manifest = Manifest::read(&args.tradebook)?;
            let td = manifest.config.trading_days();
            let n = td.len();
            let cutoffs: Vec<i32> = (1..=args.cutoffs)
                .map(|k| td[(n * k / (args.cutoffs + 1)).min(n - 1)] as i32)
                .collect();
            let table = PackedTable::open(&args.packed)?;
            let store = CheckpointStore::new(&args.out);
            println!("Building {} checkpoints at cutoffs {:?} …", cutoffs.len(), cutoffs);
            let t0 = std::time::Instant::now();
            store.build_periodic(&table, &cutoffs)?;
            println!("Done in {:.2}s → {}", t0.elapsed().as_secs_f64(), args.out.display());
        }
        Cmd::Query(args) => {
            let table = PackedTable::open(&args.packed)?;
            let clients = match args.client {
                Some(c) => ClientSel::One(c),
                None => ClientSel::All,
            };
            let span = match (&args.from, &args.to) {
                (Some(f), Some(t)) => {
                    Span::Range(calendar::parse_date(f)? as i32, calendar::parse_date(t)? as i32)
                }
                (None, None) => Span::Full,
                _ => anyhow::bail!("provide both --from and --to, or neither"),
            };
            let q = Query { clients, symbol: args.symbol, span };
            let store = CheckpointStore::new(&args.checkpoints);
            let ckpt = matches!(span, Span::Range(..)).then_some(&store);
            let rules = fifo_gpu::fifo::BucketRules {
                intraday_same_day: true,
                short_max_days: args.ltcg_days,
            };
            use fifo_gpu::fifo::MatchPolicy;
            let policy = match args.policy.as_str() {
                "fifo" => MatchPolicy::Fifo,
                "lifo" => MatchPolicy::Lifo,
                "hifo" => MatchPolicy::Hifo,
                other => anyhow::bail!("unknown --policy {other:?} (expected fifo|lifo|hifo)"),
            };
            let t0 = std::time::Instant::now();
            let r = run_cpu(&table, ckpt, &q, &mut fifo_gpu::fifo::NoopSink, &rules, policy)?;
            let dt = t0.elapsed();
            println!(
                "Query: {:?} symbol={:?} span={:?}",
                args.client, args.symbol, span
            );
            println!(
                "  {} partitions, {} rows touched, {} checkpoints, {:.2} ms",
                r.partition_fanout,
                r.rows_touched,
                r.checkpoints_loaded,
                dt.as_secs_f64() * 1e3
            );
            print_pnl(&r.pnl);
        }
        Cmd::Rollup(args) => {
            let table = PackedTable::open(&args.packed)?;
            let mut roll = Rollup::new();
            println!("Folding all partitions, accumulating cross-client rollup …");
            let t0 = std::time::Instant::now();
            let total = fold_table(&table, &mut roll);
            roll.write(&args.out)?;
            println!(
                "Rolled up {} periods in {:.2}s → {}",
                roll.by_period.len(),
                t0.elapsed().as_secs_f64(),
                args.out.display()
            );
            println!("Cross-client total (all periods):");
            print_pnl(&total);
        }
        Cmd::Correct(args) => {
            let manifest = Manifest::read(&args.tradebook)?;
            let table = PackedTable::open(&args.packed)?;
            let store = CheckpointStore::new(&args.checkpoints);
            let day = calendar::parse_date(&args.on)? as i32;
            let report = correction::correct(&manifest.config, &store, args.client, day)?;
            println!(
                "Correction for client {} on {} (day {}):",
                report.client, args.on, day
            );
            println!(
                "  regenerated {} trades across {} partitions (deterministic, isolated)",
                report.n_trades, report.n_partitions
            );
            print_pnl(&report.pnl);
            println!(
                "  checkpoints to rebuild (cutoff ≥ {}): {:?}",
                day, report.invalidated_checkpoints
            );
            // Verify the determinism invariant against the packed table.
            let (regen, packed) = correction::verify_against_packed(&manifest.config, &table, args.client)?;
            println!(
                "  determinism check (regenerated ≡ packed): {}",
                if regen == packed { "✓ MATCH" } else { "✗ MISMATCH" }
            );
        }
        Cmd::Bench(args) => {
            fifo_gpu::bench::run_bench(
                &args.tradebook,
                &args.packed,
                &args.checkpoints,
                &args.router_log,
            )?;
        }
        Cmd::Stats(args) => {
            let manifest = Manifest::read(&args.out)?;
            fifo_gpu::stats::summarize(&manifest.config).print();
            if args.verify_parquet {
                let rows = fifo_gpu::stats::count_parquet_rows(&args.out)?;
                println!("\nparquet rows on disk: {}", rows);
                println!("manifest realized   : {}", manifest.realized_total);
                if rows == manifest.realized_total {
                    println!("✓ row count matches manifest");
                } else {
                    println!("✗ MISMATCH");
                }
            }
        }
    }
    Ok(())
}
