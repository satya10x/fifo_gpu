//! Power-law trade-count-per-client model with an explicit whale tier.
//!
//! Two tiers, separately scaled so the shape is faithful (§2: "skew is the
//! hardest problem"):
//! - **Body** (long-tail retail): per-client lognormal total, single-digit
//!   median, drawn as-is.
//! - **Whales**: a small scattered set that *fills the remaining trade budget*
//!   (`target_total − Σ body`) split by Pareto weights — so whales carry the
//!   bulk of the volume while most clients stay tiny.
//!
//! Every count is a pure function of `(seed, client_id)` given a [`SkewPlan`],
//! and the plan itself is a deterministic O(clients) pre-pass — so a single
//! client's count can be recomputed in isolation (correction-friendly).

use crate::config::GenConfig;
use crate::util::{mix_seed, stream, SALT_COUNT, SALT_WHALE_W};
use rand_distr::{Distribution, LogNormal, Pareto};

/// Aggregates from the deterministic pre-pass needed to resolve any single
/// client's count.
#[derive(Clone, Copy, Debug)]
pub struct SkewPlan {
    pub target_total: u64,
    pub sum_body: u128,
    pub sum_whale_weight: f64,
    pub whale_budget: u64,
    pub n_whales: u64,
}

/// Raw body count for a non-whale client (lognormal, floored at 1).
fn body_count(cfg: &GenConfig, id: u64) -> u64 {
    let mut r = stream(mix_seed(cfg.seed, id, SALT_COUNT));
    let d = LogNormal::new(cfg.body_mu, cfg.body_sigma).expect("valid lognormal");
    d.sample(&mut r).round().max(1.0) as u64
}

/// Relative weight of a whale (Pareto; scale = 1).
fn whale_weight(cfg: &GenConfig, id: u64) -> f64 {
    let mut r = stream(mix_seed(cfg.seed, id, SALT_WHALE_W));
    let d = Pareto::new(1.0, cfg.whale_pareto_alpha).expect("valid pareto");
    d.sample(&mut r)
}

/// O(clients) deterministic pre-pass: total body volume and whale weight mass.
pub fn build_plan(cfg: &GenConfig) -> SkewPlan {
    let mut sum_body: u128 = 0;
    let mut sum_whale_weight = 0.0f64;
    let mut n_whales = 0u64;
    for id in 0..cfg.clients {
        if cfg.is_whale(id) {
            sum_whale_weight += whale_weight(cfg, id);
            n_whales += 1;
        } else {
            sum_body += body_count(cfg, id) as u128;
        }
    }
    let target_total = cfg.target_total();
    let body_capped = sum_body.min(target_total as u128) as u64;
    SkewPlan {
        target_total,
        sum_body,
        sum_whale_weight,
        whale_budget: target_total.saturating_sub(body_capped),
        n_whales,
    }
}

/// Resolve one client's trade count. Pure function of `(cfg, plan, id)`.
pub fn client_count(cfg: &GenConfig, plan: &SkewPlan, id: u64) -> u64 {
    if cfg.is_whale(id) {
        if plan.sum_whale_weight <= 0.0 {
            return 1;
        }
        let w = whale_weight(cfg, id);
        let share = (plan.whale_budget as f64) * w / plan.sum_whale_weight;
        share.round().max(1.0) as u64
    } else {
        body_count(cfg, id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> GenConfig {
        let mut c = GenConfig::defaults();
        c.clients = 10_000;
        c.n_whales = 10;
        c
    }

    #[test]
    fn deterministic_counts() {
        let c = cfg();
        let p = build_plan(&c);
        for id in [0u64, 1, 7, 999, 5000] {
            assert_eq!(client_count(&c, &p, id), client_count(&c, &p, id));
        }
    }

    #[test]
    fn whales_dominate_volume() {
        let c = cfg();
        let p = build_plan(&c);
        let mut whale_sum = 0u128;
        let mut total = 0u128;
        let mut n_whales = 0;
        for id in 0..c.clients {
            let n = client_count(&c, &p, id) as u128;
            total += n;
            if c.is_whale(id) {
                whale_sum += n;
                n_whales += 1;
            }
        }
        assert_eq!(n_whales, 10);
        // A handful of whales carry the majority of all trades.
        assert!(
            whale_sum * 2 > total,
            "whales should dominate: {whale_sum} of {total}"
        );
        // Total lands close to the requested mean × clients × days.
        let target = c.target_total() as f64;
        let got = total as f64;
        assert!((got - target).abs() / target < 0.05, "got {got}, target {target}");
    }

    #[test]
    fn body_median_is_small() {
        let c = cfg();
        let p = build_plan(&c);
        let mut body: Vec<u64> = (0..c.clients)
            .filter(|&id| !c.is_whale(id))
            .map(|id| client_count(&c, &p, id))
            .collect();
        body.sort_unstable();
        let median = body[body.len() / 2];
        assert!(median <= 12, "body median should be single/low double digit, got {median}");
    }
}
