//! Trading calendar + intraday timestamps, no external date dependency.
//!
//! Uses Howard Hinnant's `days_from_civil` / `civil_from_days` algorithms
//! (public domain). A "calendar day" for FIFO bucketing is the day-index
//! (days since the Unix epoch); two trades are intraday-eligible iff they share
//! a day-index. Weekends are skipped so ~250 trading days ≈ one year, matching
//! the §2 data spec.

/// Market session in UTC. NSE opens 09:15 IST = 03:45 UTC; runs 6h15m.
pub const OPEN_OFFSET_S: i64 = 3 * 3600 + 45 * 60; // 03:45:00 UTC
pub const SESSION_S: i64 = 6 * 3600 + 15 * 60; // 6h15m
pub const SESSION_US: i64 = SESSION_S * 1_000_000;
const DAY_S: i64 = 86_400;

/// Days since 1970-01-01 for a proleptic-Gregorian civil date.
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = (y - era * 400) as i64; // [0, 399]
    let mp = if m > 2 { m - 3 } else { m + 9 }; // [0, 11]
    let doy = (153 * mp as i64 + 2) / 5 + d as i64 - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

/// Inverse of [`days_from_civil`]: day-index → (year, month, day).
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = z - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// 0 = Sunday .. 6 = Saturday. (1970-01-01 was a Thursday → day 0 maps to 4.)
#[inline]
pub fn weekday(z: i64) -> u32 {
    (((z % 7) + 4 + 7) % 7) as u32
}

#[inline]
pub fn is_weekend(z: i64) -> bool {
    matches!(weekday(z), 0 | 6)
}

/// Parse `YYYY-MM-DD` into a day-index.
pub fn parse_date(s: &str) -> anyhow::Result<i64> {
    let parts: Vec<&str> = s.split('-').collect();
    anyhow::ensure!(parts.len() == 3, "date must be YYYY-MM-DD, got {s:?}");
    let y: i64 = parts[0].parse()?;
    let m: u32 = parts[1].parse()?;
    let d: u32 = parts[2].parse()?;
    anyhow::ensure!((1..=12).contains(&m) && (1..=31).contains(&d), "bad date {s:?}");
    Ok(days_from_civil(y, m, d))
}

/// Collect `days` trading day-indices starting at `start_day`, skipping weekends.
pub fn trading_days(start_day: i64, days: u32) -> Vec<i64> {
    let mut out = Vec::with_capacity(days as usize);
    let mut z = start_day;
    while out.len() < days as usize {
        if !is_weekend(z) {
            out.push(z);
        }
        z += 1;
    }
    out
}

/// Microseconds-since-epoch for `intraday_offset_us` into the session on day `z`.
#[inline]
pub fn ts_micros(z: i64, intraday_offset_us: i64) -> i64 {
    (z * DAY_S + OPEN_OFFSET_S) * 1_000_000 + intraday_offset_us
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_civil_days() {
        for &(y, m, d) in &[(1970, 1, 1), (2020, 2, 29), (2024, 12, 31), (1999, 7, 15)] {
            let z = days_from_civil(y, m, d);
            assert_eq!(civil_from_days(z), (y, m, d));
        }
    }

    #[test]
    fn epoch_is_thursday() {
        assert_eq!(weekday(0), 4); // 1970-01-01 = Thursday
        assert_eq!(days_from_civil(1970, 1, 1), 0);
    }

    #[test]
    fn weekends_skipped() {
        // 2020-01-01 is a Wednesday; 5 trading days lands on Tue 2020-01-07.
        let start = days_from_civil(2020, 1, 1);
        let td = trading_days(start, 5);
        assert_eq!(td.len(), 5);
        for &z in &td {
            assert!(!is_weekend(z));
        }
        assert_eq!(civil_from_days(td[4]), (2020, 1, 7));
    }

    #[test]
    fn intraday_stays_within_day() {
        let z = days_from_civil(2020, 6, 1);
        let t0 = ts_micros(z, 0);
        let t1 = ts_micros(z, SESSION_US - 1);
        // Both timestamps fall on the same UTC calendar day-index.
        assert_eq!(t0 / (DAY_S * 1_000_000), z);
        assert_eq!(t1 / (DAY_S * 1_000_000), z);
    }
}
