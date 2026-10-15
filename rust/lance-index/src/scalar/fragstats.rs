// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fragment Column Statistics Index
//!
//! One record per fragment for a single column: typed widened bounds under
//! Arrow Rust ordering, plus physical row/null/NaN/negative-NaN counts. The
//! format contract lives in `docs/src/format/index/scalar/fragment_column_stats.md`.
//!
//! The index serves planning-time pruning: given a sargable predicate it
//! answers "which fragments provably contain no matching row". It is an
//! inexact filter: false positives are possible, false negatives are not.
//! A fragment without a record always remains a scan candidate, and the
//! coverage bitmap must stay a subset of the fragments with records.
//!
//! The existing global ZoneMap index is unrelated. This module reuses the
//! generic zone aggregation primitives (`ZoneTrainer`, `ZoneMapProcessor`)
//! and the shared summary evaluation rules, not the ZoneMap index itself.

use crate::pb;
use crate::scalar::expression::ScalarQueryParser;
use crate::scalar::registry::{
    BasicTrainer, ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
};
use crate::scalar::seed::FragmentSeed;
use crate::scalar::zonemap::{
    ZoneMapProcessor, ZoneMapSeedWriter, ZoneMapStatistics, evaluate_summary_against_query,
    summary_scalar_is_nan,
};
use crate::scalar::{CreatedIndex, IndexFile, SargableQuery, ScalarIndexParams, UpdateCriteria};
use lance_core::cache::LanceCache;
use lance_core::utils::row_addr_remap::RowAddrRemap;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use datafusion_common::ScalarValue;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;

use super::zoned::ZoneTrainer;
use super::{AnyQuery, IndexStore, MetricsCollector, RowIdRemapper, ScalarIndex, SearchResult};
use crate::{Index, IndexType};

pub const FRAGMENT_STATS_FILENAME: &str = "fragment_stats.lance";
pub const FRAGMENT_COLUMN_STATS_INDEX_VERSION: u32 = 0;

/// Default zone size for the fragment-local statistics written with data
/// files. An experiment parameter, not a correctness dependency: folding is
/// defined over whatever zones a fragment carries.
pub const FRAGMENT_STATS_DEFAULT_ROWS_PER_ZONE: u64 = 8192;

/// Zone capacity used when computing records from a data stream: large enough
/// that a fragment never splits into multiple zones (row offsets are 32-bit),
/// so the trainer's fragment-boundary flush yields one summary per fragment.
const ONE_ZONE_PER_FRAGMENT_CAPACITY: u64 = u32::MAX as u64;

pub fn make_fragment_column_stats_details() -> prost_types::Any {
    prost_types::Any::from_msg(&pb::FragmentColumnStatsIndexDetails {}).unwrap()
}

/// One column summary for one fragment, matching the on-file schema.
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentStatsRecord {
    pub fragment_id: u64,
    /// Widened lower bound over non-null, non-NaN values; null = unavailable.
    pub min: ScalarValue,
    /// Widened upper bound; null = unavailable.
    pub max: ScalarValue,
    /// Physical rows summarized, including rows later hidden by deletions.
    pub row_count: u64,
    pub null_count: u64,
    /// All NaNs in the physical rows (zero for non-floating types).
    pub nan_count: u64,
    /// NaNs with the sign bit set (positive count = nan_count - this).
    /// `None` = the sign split is unknown to the source that produced this
    /// record (version 0 seeds, in-memory zone scans); version 1 seeds
    /// (#9564) carry it exactly.
    pub negative_nan_count: Option<u64>,
}

impl DeepSizeOf for FragmentStatsRecord {
    fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
        (self.min.size() - std::mem::size_of::<ScalarValue>())
            + (self.max.size() - std::mem::size_of::<ScalarValue>())
    }
}

/// Fold a fragment's zone statistics into one record.
///
/// Soundness rules match the format document: the zones must cover exactly
/// `expected_row_count` physical rows; a zone with missing extrema that still
/// holds comparable values poisons the bounds (they become null); all-null or
/// all-NaN zones contribute counts but no bounds. `ScalarValue`'s total order
/// ranks NaN above every finite value, so a NaN zone max propagates into the
/// folded max and the shared evaluation stays conservative.
///
pub(crate) fn fold_zones_to_record(
    fragment_id: u64,
    zones: &[ZoneMapStatistics],
    expected_row_count: u64,
    data_type: &DataType,
) -> Option<FragmentStatsRecord> {
    if zones.is_empty() {
        return None;
    }
    let covered: u64 = zones.iter().map(|z| z.bound.length as u64).sum();
    if covered != expected_row_count {
        return None;
    }

    let mut null_count: u64 = 0;
    let mut nan_count: u64 = 0;
    let mut min: Option<ScalarValue> = None;
    let mut max: Option<ScalarValue> = None;
    let mut poisoned = false;

    for zone in zones {
        null_count += u64::from(zone.null_count);
        nan_count += u64::from(zone.nan_count);

        if data_type.is_nested() {
            continue;
        }
        let missing_extrema = zone.min.is_null() || zone.max.is_null();
        let comparable =
            zone.bound.length as u128 > u128::from(zone.null_count) + u128::from(zone.nan_count);
        if missing_extrema {
            if comparable {
                // Bounds unknown but live comparable values exist: folding the
                // other zones would exclude those values. Bounds become null.
                poisoned = true;
            }
            continue;
        }
        if min
            .as_ref()
            .is_none_or(|cur| zone.min.partial_cmp(cur).is_some_and(|o| o.is_lt()))
        {
            min = Some(zone.min.clone());
        }
        if max
            .as_ref()
            .is_none_or(|cur| zone.max.partial_cmp(cur).is_some_and(|o| o.is_gt()))
        {
            max = Some(zone.max.clone());
        }
    }

    let null_scalar = ScalarValue::try_new_null(data_type).ok()?;
    let (mut min, mut max) = if poisoned {
        (null_scalar.clone(), null_scalar.clone())
    } else {
        (
            min.unwrap_or_else(|| null_scalar.clone()),
            max.unwrap_or_else(|| null_scalar.clone()),
        )
    };
    // The format keeps NaN out of bounds: a NaN extremum (NaN zones propagate
    // through ScalarValue's total order) becomes a null bound, and the exact
    // nan_count keeps the evaluation conservative.
    if summary_scalar_is_nan(&min) {
        min = null_scalar.clone();
    }
    if summary_scalar_is_nan(&max) {
        max = null_scalar;
    }
    Some(FragmentStatsRecord {
        fragment_id,
        min,
        max,
        row_count: expected_row_count,
        null_count,
        nan_count,
        // Zone statistics and version 0 seeds carry no sign split.
        negative_nan_count: if nan_count == 0 { Some(0) } else { None },
    })
}

/// Fold one harvested seed payload into a record.
///
/// `expected_row_count` must be the fragment's physical row count from its
/// metadata; a payload that does not cover it completely yields `None`.
/// Unreadable or incompatible payloads also yield `None` (the fragment simply
/// stays a candidate), never an error.
pub(crate) fn fold_seed_to_record(
    seed: &FragmentSeed,
    expected_row_count: u64,
    data_type: &DataType,
) -> Option<FragmentStatsRecord> {
    // metadata_value convention: "<buf_index>:<rows_per_zone>"
    let rows_per_zone = seed
        .metadata_value
        .split(':')
        .nth(1)
        .and_then(|s| s.parse::<u64>().ok())?;
    let (zones, _null_bitmap) =
        ZoneMapSeedWriter::deserialize_seed(seed.fragment_id, &seed.bytes, rows_per_zone).ok()?;
    fold_zones_to_record(seed.fragment_id, &zones, expected_row_count, data_type)
}

fn records_as_batch(records: &[FragmentStatsRecord], data_type: &DataType) -> Result<RecordBatch> {
    let mins = if records.is_empty() {
        arrow_array::new_empty_array(data_type)
    } else {
        ScalarValue::iter_to_array(records.iter().map(|r| r.min.clone()))?
    };
    let maxs = if records.is_empty() {
        arrow_array::new_empty_array(data_type)
    } else {
        ScalarValue::iter_to_array(records.iter().map(|r| r.max.clone()))?
    };
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        Field::new("fragment_id", DataType::UInt64, false),
        Field::new("min", data_type.clone(), true),
        Field::new("max", data_type.clone(), true),
        Field::new("row_count", DataType::UInt64, false),
        Field::new("null_count", DataType::UInt64, false),
        Field::new("nan_count", DataType::UInt64, false),
        Field::new("negative_nan_count", DataType::UInt64, true),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from_iter_values(
                records.iter().map(|r| r.fragment_id),
            )),
            mins,
            maxs,
            Arc::new(UInt64Array::from_iter_values(
                records.iter().map(|r| r.row_count),
            )),
            Arc::new(UInt64Array::from_iter_values(
                records.iter().map(|r| r.null_count),
            )),
            Arc::new(UInt64Array::from_iter_values(
                records.iter().map(|r| r.nan_count),
            )),
            Arc::new(UInt64Array::from_iter(
                records.iter().map(|r| r.negative_nan_count),
            )),
        ],
    )?)
}

/// Truncate a string upper bound to at most `k` bytes, widened upward: the
/// last kept byte is incremented, dropping trailing 0xFF bytes; total
/// overflow returns `None` (that side becomes a null bound = unbounded).
fn truncate_upper(value: &str, k: usize) -> Option<String> {
    if value.len() <= k {
        return Some(value.to_string());
    }
    // Per the format: character-boundary truncation with a valid successor.
    // UTF-8 byte order equals code-point order, so replacing the final kept
    // character by its next code point (skipping the surrogate gap) bounds
    // every string sharing the original prefix.
    let mut end = k;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    let mut chars: Vec<char> = value[..end].chars().collect();
    while let Some(&last) = chars.last() {
        match (last as u32 + 1..=char::MAX as u32).find_map(char::from_u32) {
            Some(successor) => {
                *chars.last_mut().unwrap() = successor;
                return Some(chars.into_iter().collect());
            }
            // A trailing char::MAX has no successor: drop it and bump the
            // previous character instead.
            None => {
                chars.pop();
            }
        }
    }
    None
}

fn truncate_lower(value: &str, k: usize) -> String {
    let bytes = value.as_bytes();
    if bytes.len() <= k {
        return value.to_string();
    }
    let mut end = k;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_string()
}

/// Apply widening-only truncation to string bounds when it costs no pruning
/// precision on this record set: the smallest k whose widened ranges add no
/// new overlaps between records is used; otherwise bounds stay lossless.
fn truncate_string_bounds(records: &mut [FragmentStatsRecord]) {
    let mut by_min: Vec<usize> = (0..records.len())
        .filter(|&i| !records[i].min.is_null() && !records[i].max.is_null())
        .collect();
    if by_min.len() < 2 {
        return;
    }
    let get = |value: &ScalarValue| -> Option<String> {
        match value {
            ScalarValue::Utf8(Some(s)) | ScalarValue::LargeUtf8(Some(s)) => Some(s.clone()),
            _ => None,
        }
    };
    by_min.sort_by_key(|&i| get(&records[i].min));
    let overlaps = |bounds: &[(String, Option<String>)]| -> usize {
        let mut count = 0;
        for window in bounds.windows(2) {
            match &window[0].1 {
                None => count += 1,
                Some(upper) if *upper >= window[1].0 => count += 1,
                _ => {}
            }
        }
        count
    };
    let original: Vec<(String, Option<String>)> = by_min
        .iter()
        .filter_map(|&i| Some((get(&records[i].min)?, get(&records[i].max))))
        .collect();
    if original.len() != by_min.len() {
        return;
    }
    let baseline = overlaps(&original);
    for k in [8usize, 16] {
        let truncated: Vec<(String, Option<String>)> = original
            .iter()
            .map(|(min, max)| {
                (
                    truncate_lower(min, k),
                    // An upper bound whose truncation overflows keeps its
                    // exact value: a null bound would be wider than needed
                    // and disqualify the record from the packed layout.
                    max.as_ref()
                        .map(|m| truncate_upper(m, k).unwrap_or_else(|| m.clone())),
                )
            })
            .collect();
        if overlaps(&truncated) == baseline {
            for (slot, &i) in by_min.iter().enumerate() {
                let (min, max) = &truncated[slot];
                let large = matches!(records[i].min, ScalarValue::LargeUtf8(_));
                let make = |v: Option<String>| {
                    if large {
                        ScalarValue::LargeUtf8(v)
                    } else {
                        ScalarValue::Utf8(v)
                    }
                };
                records[i].min = make(Some(min.clone()));
                records[i].max = make(max.clone());
            }
            return;
        }
    }
}

pub(crate) async fn write_records(
    records: Vec<FragmentStatsRecord>,
    data_type: &DataType,
    index_store: &dyn IndexStore,
) -> Result<Vec<IndexFile>> {
    // Deterministic order per the format; last writer wins on duplicates
    // (update paths append replacement records after the ones they replace).
    let mut by_id: std::collections::BTreeMap<u64, FragmentStatsRecord> =
        std::collections::BTreeMap::new();
    for record in records {
        by_id.insert(record.fragment_id, record);
    }
    let mut records: Vec<FragmentStatsRecord> = by_id.into_values().collect();
    if matches!(data_type, DataType::Utf8 | DataType::LargeUtf8) {
        truncate_string_bounds(&mut records);
    }
    let batch = records_as_batch(&records, data_type)?;
    let mut index_file = index_store
        .new_index_file(FRAGMENT_STATS_FILENAME, batch.schema())
        .await?;
    index_file.write_record_batch(batch).await?;
    let file = index_file.finish_with_metadata(HashMap::new()).await?;
    Ok(vec![file])
}

/// Write one statistics segment file from fully-formed records.
///
/// Records are sorted by fragment id with last-writer-wins dedup. Returns the
/// created index description; the caller owns registering it (with a coverage
/// bitmap that is a subset of the written records).
pub async fn write_fragment_stats_segment(
    records: Vec<FragmentStatsRecord>,
    data_type: &DataType,
    index_store: &dyn IndexStore,
) -> Result<CreatedIndex> {
    let files = write_records(records, data_type, index_store).await?;
    Ok(CreatedIndex {
        index_details: make_fragment_column_stats_details(),
        index_version: FRAGMENT_COLUMN_STATS_INDEX_VERSION,
        files,
    })
}

/// Fold one seed payload into a record (see the module rules); `None` means
/// the fragment stays uncovered.
pub fn fold_seed_payload(
    seed: &FragmentSeed,
    expected_row_count: u64,
    data_type: &DataType,
) -> Option<FragmentStatsRecord> {
    fold_seed_to_record(seed, expected_row_count, data_type)
}

/// Build index files from pre-harvested fragment seeds.
///
/// Returns the created index and the fragment ids that actually received
/// records; the committed coverage bitmap must claim only those.
pub async fn build_fragment_stats_from_seeds(
    seeds: Vec<(FragmentSeed, u64)>,
    data_type: &DataType,
    index_store: &dyn IndexStore,
) -> Result<(CreatedIndex, Vec<u32>)> {
    let mut records = Vec::with_capacity(seeds.len());
    for (seed, expected_row_count) in &seeds {
        if let Some(record) = fold_seed_to_record(seed, *expected_row_count, data_type) {
            records.push(record);
        }
    }
    let covered = records
        .iter()
        .map(|r| r.fragment_id as u32)
        .collect::<Vec<_>>();
    let files = write_records(records, data_type, index_store).await?;
    Ok((
        CreatedIndex {
            index_details: make_fragment_column_stats_details(),
            index_version: FRAGMENT_COLUMN_STATS_INDEX_VERSION,
            files,
        },
        covered,
    ))
}

/// Zero-decode packed arrays for the hot evaluation path. Built at load for
/// supported types; evaluation semantics are identical to the record path
/// (equivalence is asserted in tests), only the memory layout differs:
/// typed bound arrays, counts elided when all zero, validity as a bitset,
/// string bounds in contiguous arenas. Read access stays direct slicing.
enum PackedBounds {
    I64 {
        mins: Vec<i64>,
        maxs: Vec<i64>,
    },
    Utf8 {
        min_arena: Vec<u8>,
        min_offsets: Vec<u32>,
        max_arena: Vec<u8>,
        max_offsets: Vec<u32>,
    },
}

struct PackedStats {
    fragment_ids: Vec<u32>,
    /// None = all zero.
    null_counts: Option<Vec<u64>>,
    nan_counts: Option<Vec<u64>>,
    row_counts: Vec<u64>,
    /// Bitset; None = every record has usable bounds.
    bounds_valid: Option<Vec<u64>>,
    bounds: PackedBounds,
}

fn pack_bitset(bools: &[bool]) -> Option<Vec<u64>> {
    if bools.iter().all(|b| *b) {
        return None;
    }
    let mut words = vec![0u64; bools.len().div_ceil(64)];
    for (i, b) in bools.iter().enumerate() {
        if *b {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }
    Some(words)
}

#[inline(always)]
fn bit_at(words: &Option<Vec<u64>>, i: usize) -> bool {
    match words {
        None => true,
        Some(words) => (words[i / 64] >> (i % 64)) & 1 == 1,
    }
}

impl PackedStats {
    /// Build the packed form for supported types. Returns `None` (record-path
    /// fallback) for other types, fragment ids above u32, or records with
    /// exactly one null bound (truncation overflow; rare, record path is
    /// correct for them).
    fn try_build(records: &[FragmentStatsRecord], data_type: &DataType) -> Option<Self> {
        let scalar_i64 = |v: &ScalarValue| -> Option<i64> {
            match v {
                ScalarValue::Int64(Some(v))
                | ScalarValue::TimestampSecond(Some(v), _)
                | ScalarValue::TimestampMillisecond(Some(v), _)
                | ScalarValue::TimestampMicrosecond(Some(v), _)
                | ScalarValue::TimestampNanosecond(Some(v), _) => Some(*v),
                _ => None,
            }
        };
        fn scalar_str(v: &ScalarValue) -> Option<&str> {
            match v {
                ScalarValue::Utf8(Some(s)) | ScalarValue::LargeUtf8(Some(s)) => Some(s.as_str()),
                _ => None,
            }
        }
        let is_i64 = matches!(data_type, DataType::Int64 | DataType::Timestamp(_, _));
        let is_str = matches!(data_type, DataType::Utf8 | DataType::LargeUtf8);
        if !is_i64 && !is_str {
            return None;
        }
        let n = records.len();
        let mut fragment_ids = Vec::with_capacity(n);
        let mut row_counts = Vec::with_capacity(n);
        let mut null_counts = Vec::with_capacity(n);
        let mut nan_counts = Vec::with_capacity(n);
        let mut valid = Vec::with_capacity(n);
        for record in records {
            if record.min.is_null() != record.max.is_null() {
                return None;
            }
            fragment_ids.push(u32::try_from(record.fragment_id).ok()?);
            row_counts.push(record.row_count);
            null_counts.push(record.null_count);
            nan_counts.push(record.nan_count);
            valid.push(!record.min.is_null());
        }
        let bounds = if is_i64 {
            let mut mins = Vec::with_capacity(n);
            let mut maxs = Vec::with_capacity(n);
            for (i, record) in records.iter().enumerate() {
                if valid[i] {
                    mins.push(scalar_i64(&record.min)?);
                    maxs.push(scalar_i64(&record.max)?);
                } else {
                    mins.push(i64::MIN);
                    maxs.push(i64::MAX);
                }
            }
            PackedBounds::I64 { mins, maxs }
        } else {
            let mut min_arena = Vec::new();
            let mut min_offsets = Vec::with_capacity(n + 1);
            let mut max_arena = Vec::new();
            let mut max_offsets = Vec::with_capacity(n + 1);
            min_offsets.push(0u32);
            max_offsets.push(0u32);
            for (i, record) in records.iter().enumerate() {
                if valid[i] {
                    min_arena.extend_from_slice(scalar_str(&record.min)?.as_bytes());
                    max_arena.extend_from_slice(scalar_str(&record.max)?.as_bytes());
                }
                min_offsets.push(u32::try_from(min_arena.len()).ok()?);
                max_offsets.push(u32::try_from(max_arena.len()).ok()?);
            }
            min_arena.shrink_to_fit();
            max_arena.shrink_to_fit();
            PackedBounds::Utf8 {
                min_arena,
                min_offsets,
                max_arena,
                max_offsets,
            }
        };
        Some(Self {
            fragment_ids,
            null_counts: if null_counts.iter().all(|c| *c == 0) {
                None
            } else {
                Some(null_counts)
            },
            nan_counts: if nan_counts.iter().all(|c| *c == 0) {
                None
            } else {
                Some(nan_counts)
            },
            row_counts,
            bounds_valid: pack_bitset(&valid),
            bounds,
        })
    }

    #[inline(always)]
    fn null_count(&self, i: usize) -> u64 {
        self.null_counts.as_ref().map_or(0, |c| c[i])
    }

    #[inline(always)]
    fn nan_count(&self, i: usize) -> u64 {
        self.nan_counts.as_ref().map_or(0, |c| c[i])
    }

    fn heap_size(&self) -> usize {
        let bounds = match &self.bounds {
            PackedBounds::I64 { mins, maxs } => (mins.capacity() + maxs.capacity()) * 8,
            PackedBounds::Utf8 {
                min_arena,
                min_offsets,
                max_arena,
                max_offsets,
            } => {
                min_arena.capacity()
                    + max_arena.capacity()
                    + (min_offsets.capacity() + max_offsets.capacity()) * 4
            }
        };
        self.fragment_ids.capacity() * 4
            + self.null_counts.as_ref().map_or(0, |v| v.capacity() * 8)
            + self.nan_counts.as_ref().map_or(0, |v| v.capacity() * 8)
            + self.row_counts.capacity() * 8
            + self.bounds_valid.as_ref().map_or(0, |v| v.capacity() * 8)
            + bounds
    }

    #[inline(always)]
    fn comparable(&self, i: usize) -> bool {
        (self.row_counts[i] as u128)
            > u128::from(self.null_count(i)) + u128::from(self.nan_count(i))
    }

    /// Exclusion over the packed layout for the supported query kinds, with
    /// consecutive excluded ids inserted as ranges (clustered layouts produce
    /// long runs). `None` = unsupported query shape; use the record path.
    ///
    /// Packed value types carry no NaN values (integers, timestamps,
    /// strings), so the zone evaluation's NaN branches are vacuous here; the
    /// null/comparable conservatism is identical.
    fn excluded(&self, query: &SargableQuery) -> Option<RoaringBitmap> {
        use std::ops::Bound;
        enum Pred<'a> {
            NoNulls,
            I64Range(Option<(i64, bool)>, Option<(i64, bool)>),
            I64Eq(i64),
            I64In(&'a [ScalarValue]),
            StrRange(Option<(&'a [u8], bool)>, Option<(&'a [u8], bool)>),
            StrEq(&'a [u8]),
            StrIn(&'a [ScalarValue]),
        }
        fn as_i64(v: &ScalarValue) -> Option<i64> {
            match v {
                ScalarValue::Int64(Some(v))
                | ScalarValue::TimestampSecond(Some(v), _)
                | ScalarValue::TimestampMillisecond(Some(v), _)
                | ScalarValue::TimestampMicrosecond(Some(v), _)
                | ScalarValue::TimestampNanosecond(Some(v), _) => Some(*v),
                _ => None,
            }
        }
        fn as_bytes(v: &ScalarValue) -> Option<&[u8]> {
            match v {
                ScalarValue::Utf8(Some(s)) | ScalarValue::LargeUtf8(Some(s)) => Some(s.as_bytes()),
                _ => None,
            }
        }
        fn conv_i64(b: &Bound<ScalarValue>) -> Option<Option<(i64, bool)>> {
            match b {
                Bound::Unbounded => Some(None),
                Bound::Included(v) => Some(Some((as_i64(v)?, true))),
                Bound::Excluded(v) => Some(Some((as_i64(v)?, false))),
            }
        }
        fn conv_bytes(b: &Bound<ScalarValue>) -> Option<Option<(&[u8], bool)>> {
            match b {
                Bound::Unbounded => Some(None),
                Bound::Included(v) => Some(Some((as_bytes(v)?, true))),
                Bound::Excluded(v) => Some(Some((as_bytes(v)?, false))),
            }
        }
        let is_i64 = matches!(self.bounds, PackedBounds::I64 { .. });
        let pred = match query {
            SargableQuery::IsNull() => Pred::NoNulls,
            SargableQuery::Equals(v) if v.is_null() => Pred::NoNulls,
            SargableQuery::Equals(v) => {
                if is_i64 {
                    Pred::I64Eq(as_i64(v)?)
                } else {
                    Pred::StrEq(as_bytes(v)?)
                }
            }
            SargableQuery::Range(start, end) => {
                if is_i64 {
                    Pred::I64Range(conv_i64(start)?, conv_i64(end)?)
                } else {
                    Pred::StrRange(conv_bytes(start)?, conv_bytes(end)?)
                }
            }
            SargableQuery::IsIn(values) => {
                if is_i64 {
                    if !values.iter().all(|v| v.is_null() || as_i64(v).is_some()) {
                        return None;
                    }
                    Pred::I64In(values)
                } else {
                    if !values.iter().all(|v| v.is_null() || as_bytes(v).is_some()) {
                        return None;
                    }
                    Pred::StrIn(values)
                }
            }
            _ => return None,
        };

        let n = self.fragment_ids.len();
        let mut excluded = RoaringBitmap::new();
        let mut run_start: Option<u32> = None;
        let mut run_prev: u32 = 0;
        let flush = |run_start: &mut Option<u32>, run_prev: u32, out: &mut RoaringBitmap| {
            if let Some(first) = run_start.take() {
                out.insert_range(first..=run_prev);
            }
        };
        for i in 0..n {
            let may = match &pred {
                Pred::NoNulls => self.null_count(i) > 0,
                // Set membership decides per value: a null element tests the
                // null count even when bounds are unavailable.
                Pred::I64In(values) => match &self.bounds {
                    PackedBounds::I64 { mins, maxs } => {
                        let valid = bit_at(&self.bounds_valid, i);
                        values.iter().any(|v| {
                            if v.is_null() {
                                self.null_count(i) > 0
                            } else if !valid {
                                self.comparable(i)
                            } else {
                                as_i64(v).is_some_and(|t| t >= mins[i] && t <= maxs[i])
                            }
                        })
                    }
                    _ => unreachable!(),
                },
                Pred::StrIn(values) => match &self.bounds {
                    PackedBounds::Utf8 {
                        min_arena,
                        min_offsets,
                        max_arena,
                        max_offsets,
                    } => {
                        let valid = bit_at(&self.bounds_valid, i);
                        let min = &min_arena[min_offsets[i] as usize..min_offsets[i + 1] as usize];
                        let max = &max_arena[max_offsets[i] as usize..max_offsets[i + 1] as usize];
                        values.iter().any(|v| {
                            if v.is_null() {
                                self.null_count(i) > 0
                            } else if !valid {
                                self.comparable(i)
                            } else {
                                as_bytes(v).is_some_and(|t| t >= min && t <= max)
                            }
                        })
                    }
                    _ => unreachable!(),
                },
                _ if !bit_at(&self.bounds_valid, i) => self.comparable(i),
                Pred::I64Eq(t) => match &self.bounds {
                    PackedBounds::I64 { mins, maxs } => *t >= mins[i] && *t <= maxs[i],
                    _ => unreachable!(),
                },
                Pred::I64Range(start, end) => match &self.bounds {
                    PackedBounds::I64 { mins, maxs } => {
                        let start_ok = match start {
                            None => true,
                            Some((s, true)) => maxs[i] >= *s,
                            Some((s, false)) => maxs[i] > *s,
                        };
                        let end_ok = match end {
                            None => true,
                            Some((e, true)) => mins[i] <= *e,
                            Some((e, false)) => mins[i] < *e,
                        };
                        start_ok && end_ok
                    }
                    _ => unreachable!(),
                },
                Pred::StrEq(t) => match &self.bounds {
                    PackedBounds::Utf8 {
                        min_arena,
                        min_offsets,
                        max_arena,
                        max_offsets,
                    } => {
                        let min = &min_arena[min_offsets[i] as usize..min_offsets[i + 1] as usize];
                        let max = &max_arena[max_offsets[i] as usize..max_offsets[i + 1] as usize];
                        *t >= min && *t <= max
                    }
                    _ => unreachable!(),
                },
                Pred::StrRange(start, end) => match &self.bounds {
                    PackedBounds::Utf8 {
                        min_arena,
                        min_offsets,
                        max_arena,
                        max_offsets,
                    } => {
                        let min = &min_arena[min_offsets[i] as usize..min_offsets[i + 1] as usize];
                        let max = &max_arena[max_offsets[i] as usize..max_offsets[i + 1] as usize];
                        let start_ok = match start {
                            None => true,
                            Some((s, true)) => max >= *s,
                            Some((s, false)) => max > *s,
                        };
                        let end_ok = match end {
                            None => true,
                            Some((e, true)) => min <= *e,
                            Some((e, false)) => min < *e,
                        };
                        start_ok && end_ok
                    }
                    _ => unreachable!(),
                },
            };
            if !may {
                let id = self.fragment_ids[i];
                match run_start {
                    Some(_) if id == run_prev.wrapping_add(1) => {}
                    Some(_) => {
                        flush(&mut run_start, run_prev, &mut excluded);
                        run_start = Some(id);
                    }
                    None => run_start = Some(id),
                }
                run_prev = id;
            } else {
                flush(&mut run_start, run_prev, &mut excluded);
            }
        }
        flush(&mut run_start, run_prev, &mut excluded);
        Some(excluded)
    }
}

/// Fragment column statistics index: one summary per fragment for one column.
pub struct FragmentColumnStatsIndex {
    records: Vec<FragmentStatsRecord>,
    packed: Option<PackedStats>,
    data_type: DataType,
}

impl std::fmt::Debug for FragmentColumnStatsIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FragmentColumnStatsIndex")
            .field("num_fragments", &self.records.len())
            .field("data_type", &self.data_type)
            .finish()
    }
}

impl DeepSizeOf for FragmentColumnStatsIndex {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        let records: usize = self
            .records
            .iter()
            .map(|r| r.deep_size_of_children(context) + std::mem::size_of::<FragmentStatsRecord>())
            .sum();
        records + self.packed.as_ref().map_or(0, |p| p.heap_size())
    }
}

impl FragmentColumnStatsIndex {
    pub async fn load(store: Arc<dyn IndexStore>) -> Result<Arc<Self>> {
        let index_file = store.open_index_file(FRAGMENT_STATS_FILENAME).await?;
        let data = index_file
            .read_range(0..index_file.num_rows(), None)
            .await?;
        Self::try_from_serialized(data)
    }

    fn try_from_serialized(data: RecordBatch) -> Result<Arc<Self>> {
        let get = |name: &str| {
            data.column_by_name(name).ok_or_else(|| {
                Error::invalid_input(format!("FragmentColumnStatsIndex: missing '{name}' column"))
            })
        };
        let u64_col = |name: &str| -> Result<&UInt64Array> {
            get(name)?
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "FragmentColumnStatsIndex: '{name}' is not UInt64"
                    ))
                })
        };
        let fragment_ids = u64_col("fragment_id")?;
        let min_col = get("min")?;
        let max_col = get("max")?;
        let row_counts = u64_col("row_count")?;
        let null_counts = u64_col("null_count")?;
        let nan_counts = u64_col("nan_count")?;
        let negative_nan_counts = u64_col("negative_nan_count")?;

        let data_type = min_col.data_type().clone();
        let mut records = Vec::with_capacity(data.num_rows());
        for i in 0..data.num_rows() {
            let (min, max) = if data_type.is_nested() {
                (
                    ScalarValue::try_new_null(&data_type)?,
                    ScalarValue::try_new_null(&data_type)?,
                )
            } else {
                (
                    ScalarValue::try_from_array(min_col, i)?,
                    ScalarValue::try_from_array(max_col, i)?,
                )
            };
            records.push(FragmentStatsRecord {
                fragment_id: fragment_ids.value(i),
                min,
                max,
                row_count: row_counts.value(i),
                null_count: null_counts.value(i),
                nan_count: nan_counts.value(i),
                negative_nan_count: negative_nan_counts
                    .is_valid(i)
                    .then(|| negative_nan_counts.value(i)),
            });
        }
        let packed = PackedStats::try_build(&records, &data_type);
        Ok(Arc::new(Self {
            records,
            packed,
            data_type,
        }))
    }

    fn record_may_match(
        &self,
        record: &FragmentStatsRecord,
        query: &SargableQuery,
    ) -> Result<bool> {
        evaluate_summary_against_query(
            &self.data_type,
            &record.min,
            &record.max,
            record.null_count,
            record.nan_count,
            record.row_count,
            query,
        )
    }

    /// Fragments that have a record in this segment.
    pub fn covered_fragments(&self) -> RoaringBitmap {
        self.records.iter().map(|r| r.fragment_id as u32).collect()
    }

    /// Fragments whose record proves the query cannot match any of their rows.
    ///
    /// The complement discipline belongs to the caller: fragments WITHOUT
    /// records are never excludable, so candidates = live minus excluded.
    pub fn excluded_fragments(&self, query: &SargableQuery) -> Result<RoaringBitmap> {
        if let Some(packed) = &self.packed
            && let Some(excluded) = packed.excluded(query)
        {
            return Ok(excluded);
        }
        let mut excluded = RoaringBitmap::new();
        for record in &self.records {
            if !self.record_may_match(record, query)? {
                excluded.insert(record.fragment_id as u32);
            }
        }
        Ok(excluded)
    }

    pub fn num_fragments(&self) -> usize {
        self.records.len()
    }

    /// The records of this segment, ordered by fragment id.
    pub fn records(&self) -> &[FragmentStatsRecord] {
        &self.records
    }

    pub fn value_data_type(&self) -> &DataType {
        &self.data_type
    }
}

#[async_trait]
impl Index for FragmentColumnStatsIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    async fn prewarm(&self) -> Result<()> {
        Ok(())
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "num_fragments": self.records.len(),
        }))
    }

    fn index_type(&self) -> IndexType {
        IndexType::FragmentColumnStats
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        Ok(self.covered_fragments())
    }
}

#[async_trait]
impl ScalarIndex for FragmentColumnStatsIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        // Spans over recorded fragments would be a FALSE-NEGATIVE answer for
        // any fragment without a record, and SearchResult cannot express
        // "plus everything uncovered". No query parser routes here; refuse so
        // a generic caller cannot silently drop rows. Pruning goes through
        // `excluded_fragments`, whose contract is subtractive.
        let _ = (query, metrics);
        Err(Error::not_supported(
            "FragmentColumnStatsIndex does not answer row queries; \
             use excluded_fragments for fragment pruning",
        ))
    }

    fn results_are_row_addresses(&self) -> bool {
        true
    }

    fn can_remap(&self) -> bool {
        false
    }

    async fn remap(
        &self,
        _mapping: &RowAddrRemap,
        _dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        Err(Error::invalid_input_source(
            "FragmentColumnStatsIndex does not support remap; rewritten fragments \
             get fresh statistics from their own write-time collection"
                .into(),
        ))
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
        _old_data_filter: Option<super::OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        // Generic fallback used by index maintenance: scan the new fragments'
        // column data and append one record per fragment.
        let value_type = new_data.schema().field(0).data_type().clone();
        let processor = ZoneMapProcessor::new(value_type.clone())?;
        let trainer = ZoneTrainer::new(processor, ONE_ZONE_PER_FRAGMENT_CAPACITY)?;
        let (zones, _null_rows) = trainer.train(new_data).await?;

        let mut combined = self.records.clone();
        for zone in &zones {
            let expected = zone.bound.length as u64;
            if let Some(record) = fold_zones_to_record(
                zone.bound.fragment_id,
                std::slice::from_ref(zone),
                expected,
                &self.data_type,
            ) {
                combined.push(record);
            }
        }
        let files = write_records(combined, &self.data_type, dest_store).await?;
        Ok(CreatedIndex {
            index_details: make_fragment_column_stats_details(),
            index_version: FRAGMENT_COLUMN_STATS_INDEX_VERSION,
            files,
        })
    }

    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria::only_new_data(
            TrainingCriteria::new(TrainingOrdering::Addresses).with_row_addr(),
        )
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        Ok(ScalarIndexParams::new("FragmentColumnStats".to_string()))
    }
}

pub struct FragmentColumnStatsTrainingRequest {
    criteria: TrainingCriteria,
}

impl Default for FragmentColumnStatsTrainingRequest {
    fn default() -> Self {
        Self {
            criteria: TrainingCriteria::new(TrainingOrdering::Addresses).with_row_addr(),
        }
    }
}

impl TrainingRequest for FragmentColumnStatsTrainingRequest {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

#[derive(Debug, Default)]
pub struct FragmentColumnStatsIndexPlugin;

#[async_trait]
impl BasicTrainer for FragmentColumnStatsIndexPlugin {
    fn new_training_request(
        &self,
        _params: &str,
        _field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        Ok(Box::new(FragmentColumnStatsTrainingRequest::default()))
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        _request: Box<dyn TrainingRequest>,
        _fragment_ids: Option<Vec<u32>>,
        _progress: Arc<dyn crate::progress::IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        let value_type = data.schema().field(0).data_type().clone();
        let processor = ZoneMapProcessor::new(value_type.clone())?;
        let trainer = ZoneTrainer::new(processor, ONE_ZONE_PER_FRAGMENT_CAPACITY)?;
        let (zones, _null_rows) = trainer.train(data).await?;
        let mut records = Vec::with_capacity(zones.len());
        for zone in &zones {
            let expected = zone.bound.length as u64;
            if let Some(record) = fold_zones_to_record(
                zone.bound.fragment_id,
                std::slice::from_ref(zone),
                expected,
                &value_type,
            ) {
                records.push(record);
            }
        }
        let files = write_records(records, &value_type, index_store).await?;
        Ok(CreatedIndex {
            index_details: make_fragment_column_stats_details(),
            index_version: FRAGMENT_COLUMN_STATS_INDEX_VERSION,
            files,
        })
    }
}

#[async_trait]
impl ScalarIndexPlugin for FragmentColumnStatsIndexPlugin {
    fn basic_trainer(&self) -> Option<&dyn BasicTrainer> {
        Some(self)
    }

    fn name(&self) -> &str {
        "FragmentColumnStats"
    }

    fn provides_exact_answer(&self) -> bool {
        false
    }

    fn version(&self) -> u32 {
        FRAGMENT_COLUMN_STATS_INDEX_VERSION
    }

    /// No query parser: this index does not participate in generic predicate
    /// planning. The planner's fragment-scope resolver opens it directly.
    fn new_query_parser(
        &self,
        _index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        None
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
        index_version: u32,
        _frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        _cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        if index_version > FRAGMENT_COLUMN_STATS_INDEX_VERSION {
            return Err(Error::invalid_input(format!(
                "fragment column stats index version {index_version} is newer than supported"
            )));
        }
        Ok(FragmentColumnStatsIndex::load(index_store).await? as Arc<dyn ScalarIndex>)
    }

    fn might_use_seeds(&self, _index_details: &prost_types::Any) -> bool {
        true
    }

    async fn create_seed_writer(
        &self,
        field_path: &str,
        data_type: &DataType,
        _index_details: &prost_types::Any,
    ) -> Result<Option<Box<dyn crate::scalar::seed::IndexSeedWriter>>> {
        Ok(Some(Box::new(ZoneMapSeedWriter::new(
            field_path,
            FRAGMENT_STATS_DEFAULT_ROWS_PER_ZONE,
            data_type.clone(),
        )?)))
    }

    async fn update_from_seeds(
        &self,
        seeds: Vec<FragmentSeed>,
        reference_index: Arc<dyn ScalarIndex>,
        _index_details: &prost_types::Any,
        dest_store: &dyn IndexStore,
    ) -> Result<Option<CreatedIndex>> {
        let Some(reference) = reference_index
            .as_any()
            .downcast_ref::<FragmentColumnStatsIndex>()
        else {
            return Ok(None);
        };
        let mut combined = reference.records.clone();
        for seed in &seeds {
            // Seed payloads observed every row written to the fragment, so the
            // span they cover is self-consistent; the fold rejects internally
            // unsound payloads, leaving those fragments unknown (safe).
            let Some(rows_per_zone) = seed
                .metadata_value
                .split(':')
                .nth(1)
                .and_then(|s| s.parse::<u64>().ok())
            else {
                continue;
            };
            let Ok((zones, _)) =
                ZoneMapSeedWriter::deserialize_seed(seed.fragment_id, &seed.bytes, rows_per_zone)
            else {
                continue;
            };
            let covered: u64 = zones.iter().map(|z| z.bound.length as u64).sum();
            if let Some(record) =
                fold_zones_to_record(seed.fragment_id, &zones, covered, &reference.data_type)
            {
                combined.push(record);
            }
        }
        let files = write_records(combined, &reference.data_type, dest_store).await?;
        Ok(Some(CreatedIndex {
            index_details: make_fragment_column_stats_details(),
            index_version: FRAGMENT_COLUMN_STATS_INDEX_VERSION,
            files,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::NoOpMetricsCollector;
    use crate::scalar::zoned::ZoneBound;
    use std::ops::Bound;

    #[allow(clippy::too_many_arguments)]
    fn zone_stats(
        fragment_id: u64,
        start: u64,
        length: usize,
        min: ScalarValue,
        max: ScalarValue,
        null_count: u32,
        nan_count: u32,
    ) -> ZoneMapStatistics {
        ZoneMapStatistics {
            min,
            max,
            null_count,
            nan_count,
            bound: ZoneBound {
                fragment_id,
                start,
                length,
            },
        }
    }

    fn int_scalar(v: i64) -> ScalarValue {
        ScalarValue::Int64(Some(v))
    }

    fn record(
        fragment_id: u64,
        min: ScalarValue,
        max: ScalarValue,
        null_count: u64,
        rows: u64,
    ) -> FragmentStatsRecord {
        FragmentStatsRecord {
            fragment_id,
            min,
            max,
            row_count: rows,
            null_count,
            nan_count: 0,
            negative_nan_count: Some(0),
        }
    }

    fn int_index(records: Vec<FragmentStatsRecord>) -> FragmentColumnStatsIndex {
        let packed = PackedStats::try_build(&records, &DataType::Int64);
        FragmentColumnStatsIndex {
            records,
            packed,
            data_type: DataType::Int64,
        }
    }

    #[test]
    fn excluded_fragments_range() {
        let index = int_index(vec![
            record(0, int_scalar(0), int_scalar(9), 0, 10),
            record(1, int_scalar(10), int_scalar(19), 0, 10),
            record(3, int_scalar(90), int_scalar(100), 0, 10),
        ]);
        let query = SargableQuery::Range(Bound::Included(int_scalar(15)), Bound::Unbounded);
        let excluded = index.excluded_fragments(&query).unwrap();
        assert!(excluded.contains(0));
        assert!(!excluded.contains(1));
        assert!(!excluded.contains(3));
        // Fragment 2 has no record: it must never appear in the excluded set.
        assert!(!excluded.contains(2));
    }

    #[test]
    fn fold_rejects_partial_coverage() {
        let zones = vec![zone_stats(7, 0, 5, int_scalar(0), int_scalar(4), 0, 0)];
        assert!(fold_zones_to_record(7, &zones, 10, &DataType::Int64).is_none());
        let folded = fold_zones_to_record(7, &zones, 5, &DataType::Int64).unwrap();
        assert_eq!(folded.fragment_id, 7);
        assert_eq!(folded.min, int_scalar(0));
        assert_eq!(folded.max, int_scalar(4));
        assert_eq!(folded.row_count, 5);
    }

    #[test]
    fn fold_multiple_zones() {
        let zones = vec![
            zone_stats(2, 5, 5, int_scalar(5), int_scalar(9), 1, 0),
            zone_stats(2, 0, 5, int_scalar(0), int_scalar(3), 2, 0),
        ];
        let folded = fold_zones_to_record(2, &zones, 10, &DataType::Int64).unwrap();
        assert_eq!(folded.min, int_scalar(0));
        assert_eq!(folded.max, int_scalar(9));
        assert_eq!(folded.null_count, 3);
        assert_eq!(folded.row_count, 10);
    }

    #[test]
    fn fold_all_null_fragment() {
        let null = ScalarValue::Int64(None);
        let zones = vec![zone_stats(4, 0, 5, null.clone(), null.clone(), 5, 0)];
        let folded = fold_zones_to_record(4, &zones, 5, &DataType::Int64).unwrap();
        assert!(folded.min.is_null());
        assert_eq!(folded.null_count, 5);
        let index = int_index(vec![folded]);
        let excluded = index
            .excluded_fragments(&SargableQuery::Equals(int_scalar(1)))
            .unwrap();
        assert!(excluded.contains(4));
        let excluded = index.excluded_fragments(&SargableQuery::IsNull()).unwrap();
        assert!(!excluded.contains(4));
    }

    #[test]
    fn fold_poisoned_bounds_keep_counts() {
        let null = ScalarValue::Int64(None);
        // 5 rows, only 1 null: comparable values exist but bounds are unknown.
        // The record keeps exact counts with null bounds, per the format.
        let zones = vec![zone_stats(9, 0, 5, null.clone(), null, 1, 0)];
        let folded = fold_zones_to_record(9, &zones, 5, &DataType::Int64).unwrap();
        assert!(folded.min.is_null() && folded.max.is_null());
        assert_eq!(folded.null_count, 1);
        // Null bounds with comparable values must not exclude value queries.
        let index = int_index(vec![folded]);
        let excluded = index
            .excluded_fragments(&SargableQuery::Equals(int_scalar(42)))
            .unwrap();
        assert!(!excluded.contains(9));
    }

    #[test]
    fn fold_nan_fragment_keeps_counts_with_null_bounds() {
        // A NaN extremum propagates into the zone max under total order; the
        // folded record must carry exact counts but null that bound.
        let zones = vec![zone_stats(
            5,
            0,
            4,
            ScalarValue::Float64(Some(1.0)),
            ScalarValue::Float64(Some(f64::NAN)),
            0,
            2,
        )];
        let folded = fold_zones_to_record(5, &zones, 4, &DataType::Float64).unwrap();
        assert_eq!(folded.nan_count, 2);
        assert_eq!(folded.row_count, 4);
        assert!(!folded.min.is_null());
        assert!(folded.max.is_null());
        // Zone statistics cannot split the sign; the field stays unknown.
        assert_eq!(folded.negative_nan_count, None);

        // Finite bounds with NaNs counted stay as-is.
        let zones = vec![zone_stats(
            5,
            0,
            4,
            ScalarValue::Float64(Some(1.0)),
            ScalarValue::Float64(Some(2.0)),
            0,
            2,
        )];
        let folded = fold_zones_to_record(5, &zones, 4, &DataType::Float64).unwrap();
        assert_eq!(folded.nan_count, 2);
        assert!(!folded.max.is_null());
    }

    #[tokio::test]
    async fn search_is_not_supported() {
        let index = int_index(vec![record(0, int_scalar(0), int_scalar(9), 0, 10)]);
        let query = SargableQuery::Equals(int_scalar(5));
        let result = index.search(&query, &NoOpMetricsCollector).await;
        assert!(matches!(result, Err(Error::NotSupported { .. })));
    }

    fn record_only(index: &FragmentColumnStatsIndex) -> FragmentColumnStatsIndex {
        FragmentColumnStatsIndex {
            records: index.records.clone(),
            packed: None,
            data_type: index.data_type.clone(),
        }
    }

    fn str_scalar(v: &str) -> ScalarValue {
        ScalarValue::Utf8(Some(v.to_string()))
    }

    fn str_index(records: Vec<FragmentStatsRecord>) -> FragmentColumnStatsIndex {
        let packed = PackedStats::try_build(&records, &DataType::Utf8);
        FragmentColumnStatsIndex {
            records,
            packed,
            data_type: DataType::Utf8,
        }
    }

    #[test]
    fn packed_matches_record_path_int() {
        let mut records = vec![
            record(0, int_scalar(0), int_scalar(9), 0, 10),
            record(1, int_scalar(10), int_scalar(19), 2, 10),
            // All-null fragment: null bounds, nothing comparable.
            record(
                3,
                ScalarValue::Int64(None),
                ScalarValue::Int64(None),
                10,
                10,
            ),
            // Poisoned bounds but comparable rows: must always be retained.
            record(4, ScalarValue::Int64(None), ScalarValue::Int64(None), 2, 10),
            record(7, int_scalar(100), int_scalar(100), 0, 10),
        ];
        // A long consecutive run to exercise insert_range.
        for id in 100..300u64 {
            records.push(record(id, int_scalar(1000), int_scalar(2000), 0, 10));
        }
        let index = int_index(records);
        assert!(index.packed.is_some());
        let fallback = record_only(&index);
        let queries = vec![
            SargableQuery::IsNull(),
            SargableQuery::Equals(ScalarValue::Int64(None)),
            SargableQuery::Equals(int_scalar(12)),
            SargableQuery::Equals(int_scalar(-5)),
            SargableQuery::Range(Bound::Included(int_scalar(15)), Bound::Unbounded),
            SargableQuery::Range(Bound::Excluded(int_scalar(19)), Bound::Unbounded),
            SargableQuery::Range(Bound::Unbounded, Bound::Excluded(int_scalar(0))),
            SargableQuery::Range(Bound::Unbounded, Bound::Included(int_scalar(0))),
            SargableQuery::Range(
                Bound::Included(int_scalar(100)),
                Bound::Included(int_scalar(999)),
            ),
            SargableQuery::IsIn(vec![int_scalar(5), int_scalar(2000)]),
            SargableQuery::IsIn(vec![ScalarValue::Int64(None)]),
            SargableQuery::IsIn(vec![]),
        ];
        for query in queries {
            let packed = index.excluded_fragments(&query).unwrap();
            let plain = fallback.excluded_fragments(&query).unwrap();
            assert_eq!(packed, plain, "query {:?}", query);
            // The poisoned-bounds comparable fragment is never excluded by a
            // value predicate, except the vacuous empty IsIn (no value can
            // match anywhere) and the count-based IsNull.
            let vacuous = matches!(&query, SargableQuery::IsIn(values) if values.is_empty());
            if !matches!(query, SargableQuery::IsNull()) && !vacuous {
                assert!(!packed.contains(4), "query {:?}", query);
            }
        }
        // Run output sanity: the 100..300 block excludes as one range.
        let excluded = index
            .excluded_fragments(&SargableQuery::Equals(int_scalar(12)))
            .unwrap();
        assert!((100..300u32).all(|id| excluded.contains(id)));
    }

    #[test]
    fn packed_matches_record_path_utf8() {
        let records = vec![
            record(0, str_scalar("apple"), str_scalar("banana"), 0, 10),
            record(1, str_scalar("cherry"), str_scalar("durian"), 1, 10),
            record(2, ScalarValue::Utf8(None), ScalarValue::Utf8(None), 10, 10),
            record(5, str_scalar("melon"), str_scalar("melon"), 0, 10),
        ];
        let index = str_index(records);
        assert!(index.packed.is_some());
        let fallback = record_only(&index);
        let queries = vec![
            SargableQuery::IsNull(),
            SargableQuery::Equals(str_scalar("coconut")),
            SargableQuery::Equals(str_scalar("zzz")),
            SargableQuery::Range(Bound::Included(str_scalar("melon")), Bound::Unbounded),
            SargableQuery::Range(Bound::Excluded(str_scalar("melon")), Bound::Unbounded),
            SargableQuery::Range(Bound::Unbounded, Bound::Excluded(str_scalar("apple"))),
            SargableQuery::IsIn(vec![str_scalar("banana"), ScalarValue::Utf8(None)]),
        ];
        for query in queries {
            let packed = index.excluded_fragments(&query).unwrap();
            let plain = fallback.excluded_fragments(&query).unwrap();
            assert_eq!(packed, plain, "query {:?}", query);
        }
    }

    #[test]
    fn packed_skips_single_null_bound_records() {
        // A record with exactly one null bound cannot be packed; the index
        // must fall back to the record path rather than misrepresent it.
        let records = vec![
            record(0, int_scalar(0), int_scalar(9), 0, 10),
            record(1, int_scalar(10), ScalarValue::Int64(None), 0, 10),
        ];
        let index = int_index(records);
        assert!(index.packed.is_none());
        let excluded = index
            .excluded_fragments(&SargableQuery::Equals(int_scalar(50)))
            .unwrap();
        assert!(excluded.contains(0));
        assert!(!excluded.contains(1));
    }

    #[test]
    fn truncate_upper_stays_valid_utf8_and_widens() {
        // The gatekeeper reproducer: a prefix whose last byte increment is
        // invalid UTF-8. The character-successor recipe must still produce a
        // valid, strictly-wider bound.
        let v = "\u{ff}".repeat(10); // 20 bytes of C3 BF
        let upper = truncate_upper(&v, 8).unwrap();
        assert!(upper.len() < v.len());
        assert!(upper.as_str() > v.as_str());

        // A trailing char::MAX cannot be incremented: it is dropped and the
        // previous character bumped.
        let v = format!("a{}", char::MAX);
        let upper = truncate_upper(&v, 1).unwrap();
        assert_eq!(upper, "b");
        assert!(upper.as_str() > v.as_str());

        // All char::MAX: no valid shortened upper bound exists.
        let v = char::MAX.to_string().repeat(4);
        assert!(truncate_upper(&v, 4).is_none());
    }

    #[test]
    fn truncation_widens_only_and_rejects_new_overlaps() {
        let long = |prefix: &str, tail: &str| format!("{}{}", prefix.repeat(8), tail);
        let mut records = vec![
            record(
                0,
                str_scalar(&long("aa", "001")),
                str_scalar(&long("aa", "999")),
                0,
                10,
            ),
            record(
                1,
                str_scalar(&long("bb", "001")),
                str_scalar(&long("bb", "999")),
                0,
                10,
            ),
            record(
                2,
                str_scalar(&long("cc", "001")),
                str_scalar(&long("cc", "999")),
                0,
                10,
            ),
        ];
        let originals = records.clone();
        truncate_string_bounds(&mut records);
        let text = |v: &ScalarValue| match v {
            ScalarValue::Utf8(Some(s)) => s.clone(),
            _ => panic!("expected utf8"),
        };
        let mut shortened = false;
        for (orig, trunc) in originals.iter().zip(records.iter()) {
            assert!(text(&trunc.min) <= text(&orig.min));
            assert!(text(&trunc.max) >= text(&orig.max));
            shortened |= text(&trunc.min).len() < text(&orig.min).len();
        }
        assert!(shortened, "distinct prefixes should allow truncation");
        // No new overlaps: pairwise disjoint inputs stay pairwise disjoint.
        for i in 0..records.len() {
            for j in 0..records.len() {
                if i != j {
                    assert!(
                        text(&records[i].max) < text(&records[j].min)
                            || text(&records[j].max) < text(&records[i].min)
                    );
                }
            }
        }

        // Shared long prefix: truncation at any k would merge the ranges, so
        // bounds must stay untouched.
        let mut same = vec![
            record(
                0,
                str_scalar(&long("xy", "aaa")),
                str_scalar(&long("xy", "bbb")),
                0,
                10,
            ),
            record(
                1,
                str_scalar(&long("xy", "ccc")),
                str_scalar(&long("xy", "ddd")),
                0,
                10,
            ),
        ];
        let before = same.clone();
        truncate_string_bounds(&mut same);
        for (orig, kept) in before.iter().zip(same.iter()) {
            assert_eq!(text(&orig.min), text(&kept.min));
            assert_eq!(text(&orig.max), text(&kept.max));
        }
    }
}
