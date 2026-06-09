# fifo_gpu — Design: toward a generic ordered-scan compute engine

## Thesis
FIFO realized-PnL is **one instance** of a broader primitive: an **ordered,
stateful, per-partition scan** over a transparent packed store, dispatched
CPU-or-GPU by a cost router. The engine generalizes along three independent axes:

1. **Bucketing** — how matched results are classified/aggregated (intraday /
   short / long today; jurisdiction-configurable tax bands tomorrow).
2. **Matching policy** — how sells consume open lots (FIFO today; LIFO, HIFO,
   average-cost, specific-lot).
3. **Accumulator** — what the per-partition scan computes (realized PnL today;
   VWAP, running position, drawdown, unrealized, … as a library).

The storage layer (transparent packed records, `(client,symbol,ts)`-clustered,
GPU-direct) and the execution layer (batched-over-partitions, scan +
searchsorted + segmented-reduce, streamed/bounded-VRAM, cost-routed) are
**shared** across all three axes. Only the per-partition logic varies.

```
            ┌─────────────────────────── shared ───────────────────────────┐
 storage    │ transparent packed store · (client,symbol,ts) clustered ·     │
            │ page-range index · checkpoints · cross-client rollup          │
 execution  │ batched partitions · scan/searchsorted/segmented-reduce ·     │
            │ CUDA streams (overlap) · bounded VRAM · cost router           │
            └───────────────────────────────────────────────────────────────┘
 per-query    [ Accumulator ]  ×  [ MatchPolicy ]  →  emits fragments
 per-fragment                       [ BucketRules ]  →  K labelled buckets
```

---

## Axis 1 — BucketRules (data-driven classification)

**Today:** hardcoded `classify(buy, sell)` → `{Intraday, Short, Long}` with
`==` / `≤365` / `>365`.

**Target:** a config the caller supplies per query (different clients /
jurisdictions / instruments can differ):

```rust
struct BucketRules {
    intraday_same_day: bool,    // span == 0 → its own bucket
    boundaries_days: Vec<i32>,  // ascending holding-day upper bounds
    labels: Vec<String>,        // K = (intraday?1:0) + boundaries.len() + 1
}
// classify(&rules, holding_days) -> bucket index  (0..K)
```
Holding-period bands cover most real tax regimes (India equity STCG ≤12mo /
LTCG; US >1yr long-term; India debt/property 3-band). Richer predicates
(instrument class, account type) are a later extension of the same struct.

**GPU:** thresholds become kernel params; segmented-reduce into *K* buckets
instead of a fixed 3.

**Phasing:**
- **A.1 (now):** `BucketRules{ intraday_same_day, short_max_days }` driving
  `classify`, threaded through the fold; default reproduces today's results.
  3 named buckets retained; **threshold configurable** (the jurisdiction lever).
  GPU unchanged (honors the default threshold only — see A.3).
- **A.2:** generalize `PartitionPnl` to `[BucketPnl; MAX_BUCKETS]` (fixed array,
  stays `Copy`); `classify` returns an index; arbitrary K-band rules on CPU.
- **A.3:** thread thresholds + K into the GPU kernels (param array; K-way
  segmented-reduce) so the GPU honors custom rulesets too.

---

## Axis 2 — MatchPolicy (FIFO / LIFO / HIFO / …)

**Today:** drain the oldest open lot (FIFO).

**Target:** a policy that selects which open lot a sell consumes:

| policy | lot selection | structure |
|---|---|---|
| FIFO | oldest | queue (front) |
| LIFO | newest | stack (back) |
| HIFO | highest cost | cost-max heap (tax-loss harvesting) |
| average-cost | blended basis | single running lot |
| specific-lot | caller-chosen | external selection |

**CPU:** `fold_core<P: MatchPolicy>` — emission logic identical; only lot
selection differs. Trivial.

**GPU:**
- **FIFO** keeps the fast **scan + searchsorted** kernel (the n-th-share-sold ↔
  n-th-share-bought identity is FIFO-specific).
- **LIFO / HIFO / specific** use the **sequential one-block-per-partition
  kernel** (already built, general, less parallel within a partition).
- The router picks the fast path when the policy allows it.

**Phasing:** B.1 — CPU `MatchPolicy` trait + FIFO/LIFO/HIFO. B.2 — GPU sequential
kernel generalized to policy; FIFO stays on the scan kernel.

---

## Axis 3 — Accumulator (generic window functions)

**Today:** the lot-matching fold *is* the accumulator (realized PnL + fragments).

**Target:** a library of per-partition accumulators sharing the
batched-partition + segmented-reduce machinery:
- realized PnL (per policy), unrealized/mark-to-market, VWAP / average cost,
  running position & exposure, max drawdown, holding-period histograms.

The shape is an ordered `step(state, event) -> (state, emit?)` over a sorted
partition, batched across partitions. FIFO-PnL becomes one registered
accumulator. *Arbitrary user code on the GPU* is a DSL/codegen project and is
explicitly **out of scope** — the pragmatic target is a curated built-in library
+ the config-driven bucketing of Axis 1.

**Phasing:** C is the largest and last; do A and B first (they cover the
near-term tax/PnL use-cases).

---

## Storage implications
The packed record (`{i32 signed_qty, i32 price_ticks, i32 day}`, 12 B) already
carries what FIFO/LIFO/HIFO and holding-period bucketing need. Richer rules
(instrument class, account type) would add a narrow field or a side column —
deferred until a rule actually needs it (keep the hot record lean; the transfer
is the bottleneck).

## Non-goals
- Arbitrary JIT/DSL kernels (Axis 3 stays a curated library).
- Putting any of this in a Lance *decoder* (Decision 1) — compute stays above
  storage; the decoder only serves the packed input bytes.

## Roadmap order
**A.1 → A.2 → B.1 → A.3/B.2 → C.** A.1 first: smallest real step, fully
CPU-testable, and it forces the right generalization (rules as data, not
constants) that everything else builds on.
