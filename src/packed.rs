//! The packed compute table — Decision 5 realized directly as a transparent,
//! fixed-width, struct-packed binary buffer (no Lance encoder in the loop).
//!
//! Why this shape:
//! - **Struct-packed** 32-byte FIFO tuple `(side⊕qty, price_ticks, day, ts)` —
//!   all fields the kernel needs in one contiguous, 8-byte-aligned record.
//! - **Transparent**: every value is extractable from location+length alone, so
//!   the `records` buffer mmaps and DMAs to the GPU with **no CPU decompress**.
//!   No LZ4/Snappy/zstd on this hot path.
//! - **Clustered** by `(client, symbol, ts)` (the generator already emits that
//!   order), so partitions are contiguous and a `(client,symbol)` lookup is a
//!   binary search, not a scan.
//!
//! Layout (one file): `[Header][part_client][part_symbol][part_offset][records]`
//! with `records` aligned to 4 KiB so an mmap'd page maps cleanly to a GPU
//! transfer unit (the ~8 MiB page ≈ one CUDA stream unit, Decision 5).

use crate::generate::TICK;
use anyhow::{ensure, Result};
use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::mem::size_of;
use std::path::Path;

/// One trade, packed for the FIFO kernel. `#[repr(C)]`, **12 bytes**, no padding.
///
/// `ts` is intentionally NOT stored: ordering is **positional** — records are
/// written `(client, symbol, ts)`-sorted at build time, so the array index *is*
/// time order and the fold never needs a per-row timestamp. Raw `ts` and audit
/// context live in the wide tradebook. Dropping it (8 B) plus the old pad (4 B)
/// took the record 32 → 12 B — pure H2D savings on the GPU hot path. Fields are
/// `i32`: a single trade's quantity/price-ticks/day fit comfortably (accumulated
/// PnL and qty stay wide — i128/i64 — in the fold, not in the record).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct PackedTrade {
    /// `+qty` for a buy, `-qty` for a sell (side ⊕ qty).
    pub signed_qty: i32,
    /// Price in integer ticks (price / TICK), exact and fixed-width.
    pub price_ticks: i32,
    /// Calendar day-index (days since epoch) — for intraday/short/long bucketing.
    pub day: i32,
}

impl PackedTrade {
    #[inline]
    pub fn is_buy(&self) -> bool {
        self.signed_qty > 0
    }
    #[inline]
    pub fn qty(&self) -> i64 {
        self.signed_qty.abs() as i64
    }
    #[inline]
    pub fn price(&self) -> f64 {
        self.price_ticks as f64 * TICK
    }
}

#[inline]
pub fn price_to_ticks(price: f64) -> i32 {
    (price / TICK).round() as i32
}

const MICROS_PER_DAY: i64 = 86_400_000_000;

impl PackedTrade {
    /// Convert a wide tradebook row into the packed FIFO tuple.
    pub fn from_trade(t: &crate::schema::Trade) -> Self {
        PackedTrade {
            signed_qty: t.side as i32 * t.quantity as i32,
            price_ticks: price_to_ticks(t.price),
            day: (t.ts_micros / MICROS_PER_DAY) as i32,
        }
    }
}

const MAGIC: [u8; 8] = *b"FIFOPK01";
const RECORD_ALIGN: u64 = 4096;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Header {
    magic: [u8; 8],
    version: u32,
    _pad: u32,
    n_rows: u64,
    n_parts: u64,
    tick: f64,
    off_part_client: u64,
    off_part_symbol: u64,
    off_part_offset: u64,
    off_records: u64,
    file_len: u64,
}

#[inline]
fn align_up(x: u64, a: u64) -> u64 {
    (x + a - 1) / a * a
}

/// In-memory builder; the packer streams partitions into it (or writes directly).
#[derive(Default)]
pub struct PackedBuilder {
    pub records: Vec<PackedTrade>,
    pub part_client: Vec<u64>,
    pub part_symbol: Vec<u32>,
    /// Prefix offsets, length `n_parts + 1`; last entry == n_rows.
    pub part_offset: Vec<u64>,
}

impl PackedBuilder {
    pub fn new() -> Self {
        let mut b = PackedBuilder::default();
        b.part_offset.push(0);
        b
    }

    /// Append a whole partition's already-ts-sorted records.
    pub fn push_partition(&mut self, client: u64, symbol: u32, recs: &[PackedTrade]) {
        if recs.is_empty() {
            return;
        }
        self.part_client.push(client);
        self.part_symbol.push(symbol);
        self.records.extend_from_slice(recs);
        self.part_offset.push(self.records.len() as u64);
    }

    pub fn n_parts(&self) -> usize {
        self.part_client.len()
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        serialize_packed(
            &self.part_client,
            &self.part_symbol,
            &self.part_offset,
            &self.records,
            &mut w,
        )?;
        w.flush()?;
        Ok(())
    }
}

fn pad_to<W: Write>(w: &mut W, pos: &mut u64, target: u64) -> Result<()> {
    while *pos < target {
        w.write_all(&[0u8])?;
        *pos += 1;
    }
    Ok(())
}

/// Serialize the packed-table layout (`[Header][part_client][part_symbol]
/// [part_offset][records]`) to any writer. Shared by [`PackedBuilder::write`]
/// (to a file) and the Lance backend (to an in-memory `Vec<u8>` → [`PackedTable::from_bytes`]).
pub fn serialize_packed<W: Write>(
    part_client: &[u64],
    part_symbol: &[u32],
    part_offset: &[u64],
    records: &[PackedTrade],
    w: &mut W,
) -> Result<()> {
    serialize_prefix(part_client, part_symbol, part_offset, w)?;
    w.write_all(bytemuck::cast_slice(records))?;
    Ok(())
}

/// Serialize everything *except* the records — the header and partition arrays,
/// padded up to the records offset. The caller then appends exactly
/// `part_offset.last()` records' worth of bytes to complete the buffer. Lets the
/// Lance fast path stream the transparent `rec` column straight into the packed
/// buffer (one copy, no intermediate `Vec<PackedTrade>`).
pub fn serialize_prefix<W: Write>(
    part_client: &[u64],
    part_symbol: &[u32],
    part_offset: &[u64],
    w: &mut W,
) -> Result<()> {
    let n_rows = *part_offset.last().unwrap_or(&0);
    let n_parts = part_client.len() as u64;
    debug_assert_eq!(part_offset.len() as u64, n_parts + 1);

    let h_size = size_of::<Header>() as u64;
    let off_part_client = align_up(h_size, 8);
    let off_part_symbol = align_up(off_part_client + n_parts * 8, 8);
    let off_part_offset = align_up(off_part_symbol + n_parts * 4, 8);
    let off_records = align_up(off_part_offset + (n_parts + 1) * 8, RECORD_ALIGN);
    let file_len = off_records + n_rows * size_of::<PackedTrade>() as u64;

    let header = Header {
        magic: MAGIC,
        version: 1,
        _pad: 0,
        n_rows,
        n_parts,
        tick: TICK,
        off_part_client,
        off_part_symbol,
        off_part_offset,
        off_records,
        file_len,
    };

    let mut pos = 0u64;
    w.write_all(bytemuck::bytes_of(&header))?;
    pos += h_size;
    pad_to(w, &mut pos, off_part_client)?;
    w.write_all(bytemuck::cast_slice(part_client))?;
    pos += n_parts * 8;
    pad_to(w, &mut pos, off_part_symbol)?;
    w.write_all(bytemuck::cast_slice(part_symbol))?;
    pos += n_parts * 4;
    pad_to(w, &mut pos, off_part_offset)?;
    w.write_all(bytemuck::cast_slice(part_offset))?;
    pos += (n_parts + 1) * 8;
    pad_to(w, &mut pos, off_records)?;
    Ok(())
}

/// Backing bytes for a [`PackedTable`]: either an `mmap` of a `.fifopack` file
/// (the default, true zero-copy on disk) or an owned buffer assembled in memory
/// (e.g. by the Lance backend reading the transparent records column). Either
/// way the `records` slice is contiguous and DMAs straight to the GPU.
enum Backing {
    Mmap(Mmap),
    Owned(Vec<u8>),
}

impl Backing {
    #[inline]
    fn bytes(&self) -> &[u8] {
        match self {
            Backing::Mmap(m) => &m[..],
            Backing::Owned(v) => &v[..],
        }
    }
}

/// A zero-copy view of a packed compute table. The `records` slice IS the
/// backing buffer — no decode, no copy. This is the buffer the GPU uploader
/// hands straight to `cudaMemcpy`.
pub struct PackedTable {
    store: Backing,
    header: Header,
}

fn parse_header(bytes: &[u8]) -> Result<Header> {
    ensure!(
        bytes.len() >= size_of::<Header>(),
        "too small to be a packed table"
    );
    let header: Header = *bytemuck::from_bytes(&bytes[..size_of::<Header>()]);
    ensure!(header.magic == MAGIC, "bad magic — not a FIFOPK buffer");
    ensure!(bytes.len() as u64 >= header.file_len, "truncated packed table");
    Ok(header)
}

impl PackedTable {
    pub fn open(path: &Path) -> Result<Self> {
        let f = File::open(path)?;
        // SAFETY: file is read-only for the lifetime of the mmap.
        let mmap = unsafe { Mmap::map(&f)? };
        let header = parse_header(&mmap)?;
        Ok(PackedTable { store: Backing::Mmap(mmap), header })
    }

    /// Open a packed table from an in-memory buffer already in the `.fifopack`
    /// layout (see [`serialize_packed`]). Used by the Lance backend so the
    /// engine/GPU run on Lance-sourced data with no temp file.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let header = parse_header(&bytes)?;
        Ok(PackedTable { store: Backing::Owned(bytes), header })
    }

    #[inline]
    fn section<T: Pod>(&self, off: u64, count: u64) -> &[T] {
        let start = off as usize;
        let bytes = &self.store.bytes()[start..start + count as usize * size_of::<T>()];
        bytemuck::cast_slice(bytes)
    }

    pub fn n_rows(&self) -> u64 {
        self.header.n_rows
    }
    pub fn n_parts(&self) -> u64 {
        self.header.n_parts
    }
    pub fn records(&self) -> &[PackedTrade] {
        self.section(self.header.off_records, self.header.n_rows)
    }
    pub fn part_client(&self) -> &[u64] {
        self.section(self.header.off_part_client, self.header.n_parts)
    }
    pub fn part_symbol(&self) -> &[u32] {
        self.section(self.header.off_part_symbol, self.header.n_parts)
    }
    pub fn part_offset(&self) -> &[u64] {
        self.section(self.header.off_part_offset, self.header.n_parts + 1)
    }

    /// Records of partition `p`.
    pub fn partition(&self, p: usize) -> &[PackedTrade] {
        let off = self.part_offset();
        &self.records()[off[p] as usize..off[p + 1] as usize]
    }

    /// Partition index for an exact `(client, symbol)`, if present.
    pub fn find_partition(&self, client: u64, symbol: u32) -> Option<usize> {
        let pc = self.part_client();
        let ps = self.part_symbol();
        // partitions are sorted by (client, symbol)
        let mut lo = 0usize;
        let mut hi = pc.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            let key = (pc[mid], ps[mid]);
            match key.cmp(&(client, symbol)) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    /// Half-open range of partition indices belonging to `client`.
    pub fn partitions_for_client(&self, client: u64) -> std::ops::Range<usize> {
        let pc = self.part_client();
        let lo = pc.partition_point(|&c| c < client);
        let hi = pc.partition_point(|&c| c <= client);
        lo..hi
    }

    /// Sub-slice of a partition restricted to `day ∈ [lo, hi]`. Records are
    /// ts-sorted, hence day-sorted, so this is two binary searches.
    pub fn day_range<'a>(&self, recs: &'a [PackedTrade], lo: i32, hi: i32) -> &'a [PackedTrade] {
        let s = recs.partition_point(|r| r.day < lo);
        let e = recs.partition_point(|r| r.day <= hi);
        &recs[s..e]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(sq: i32, px: i32, day: i32) -> PackedTrade {
        PackedTrade { signed_qty: sq, price_ticks: px, day }
    }

    #[test]
    fn record_is_12_bytes() {
        assert_eq!(size_of::<PackedTrade>(), 12);
    }

    #[test]
    fn write_read_roundtrip_and_lookup() {
        let mut b = PackedBuilder::new();
        b.push_partition(1, 10, &[rec(100, 2000, 1), rec(-100, 2100, 2)]);
        b.push_partition(1, 20, &[rec(50, 500, 1)]);
        b.push_partition(2, 10, &[rec(10, 100, 5), rec(10, 110, 400), rec(-20, 120, 800)]);

        let path = std::env::temp_dir().join("fifo_pack_test.fifopack");
        b.write(&path).unwrap();
        let t = PackedTable::open(&path).unwrap();

        assert_eq!(t.n_parts(), 3);
        assert_eq!(t.n_rows(), 6);
        // zero-copy bytes equal what we wrote
        assert_eq!(t.partition(0), &[rec(100, 2000, 1), rec(-100, 2100, 2)]);

        let p = t.find_partition(2, 10).unwrap();
        assert_eq!(t.partition(p).len(), 3);
        assert!(t.find_partition(9, 9).is_none());

        assert_eq!(t.partitions_for_client(1), 0..2);
        assert_eq!(t.partitions_for_client(2), 2..3);

        // day-range prune within the (2,10) partition
        let recs = t.partition(p);
        assert_eq!(t.day_range(recs, 0, 10).len(), 1); // only day 5
        assert_eq!(t.day_range(recs, 400, 800).len(), 2);
        let _ = std::fs::remove_file(&path);
    }
}
