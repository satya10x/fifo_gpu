//! Symbol universe: deterministic per-symbol base price + volatility, and a
//! Zipf popularity law so clients concentrate on a few liquid names.

use crate::util::{splitmix64, SALT_SYM_PRICE};
use rand::Rng;
use std::collections::HashSet;

pub struct SymbolUniverse {
    pub n: u32,
    /// Prefix sums of Zipf weights for popularity sampling.
    cdf: Vec<f64>,
    total: f64,
}

impl SymbolUniverse {
    pub fn new(n: u32) -> Self {
        // Zipf: weight(rank) = 1 / (rank+1)^s. s≈1.07 → realistic heavy head.
        let s = 1.07f64;
        let mut cdf = Vec::with_capacity(n as usize);
        let mut acc = 0.0f64;
        for r in 0..n {
            acc += 1.0 / ((r as f64 + 1.0).powf(s));
            cdf.push(acc);
        }
        let total = acc;
        SymbolUniverse { n, cdf, total }
    }

    /// Deterministic base price for a symbol, in [~20, ~8000].
    pub fn base_price(&self, id: u32) -> f64 {
        let h = splitmix64(id as u64 ^ SALT_SYM_PRICE);
        let u = (h >> 11) as f64 / (1u64 << 53) as f64; // (0,1)
        (3.0 + 6.0 * u).exp() // exp(3..9) ≈ 20..8100
    }

    /// Deterministic daily volatility for a symbol, in [0.005, 0.05].
    pub fn daily_vol(&self, id: u32) -> f64 {
        let h = splitmix64((id as u64).wrapping_mul(0x9E37_79B9) ^ SALT_SYM_PRICE);
        let u = (h >> 11) as f64 / (1u64 << 53) as f64;
        0.005 + 0.045 * u
    }

    /// Sample `k` distinct symbol ids, weighted by Zipf popularity.
    pub fn sample_distinct<R: Rng>(&self, rng: &mut R, k: u32) -> Vec<u32> {
        let k = k.min(self.n);
        let mut set = HashSet::with_capacity(k as usize);
        // Bounded attempts so a pathological draw can't loop forever; fall back
        // to filling sequentially from the head (most-popular) names.
        let mut attempts = 0u32;
        while (set.len() as u32) < k && attempts < k * 8 + 16 {
            let u = rng.gen::<f64>() * self.total;
            let idx = self.cdf.partition_point(|&c| c < u) as u32;
            set.insert(idx.min(self.n - 1));
            attempts += 1;
        }
        let mut fill = 0u32;
        while (set.len() as u32) < k {
            set.insert(fill);
            fill += 1;
        }
        let mut v: Vec<u32> = set.into_iter().collect();
        v.sort_unstable(); // clustering wants symbols ascending within a client
        v
    }
}
