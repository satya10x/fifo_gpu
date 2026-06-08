//! Deterministic RNG plumbing.
//!
//! Every per-client stream is seeded from `mix_seed(global_seed, client_id, salt)`
//! so streams are independent *and* reproducible from the id alone — no
//! cross-client coupling. Distinct salts give a client independent count- and
//! trade-generation streams.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// SplitMix64 — fast, well-distributed 64-bit mixer (public domain).
#[inline]
pub fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Derive a stream seed from `(global_seed, id, salt)`. Order-independent of
/// other ids: the value depends only on these three inputs.
#[inline]
pub fn mix_seed(seed: u64, id: u64, salt: u64) -> u64 {
    let a = splitmix64(id.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt);
    splitmix64(seed ^ a)
}

/// A fresh, seeded ChaCha8 stream. ChaCha8 is overkill statistically but cheap
/// to seed (~5M inits in the count pre-pass is sub-second).
#[inline]
pub fn stream(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

// Salts: keep the count stream and trade stream independent for a given client.
pub const SALT_COUNT: u64 = 0x0000_0000_0000_00C0;
pub const SALT_TRADE: u64 = 0x0000_0000_0000_007E;
pub const SALT_SYM_PRICE: u64 = 0x0000_0000_0050_5800;
pub const SALT_WHALE_W: u64 = 0x0000_0000_0057_4157;
