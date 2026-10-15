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
    let batch = records_as_batch(&records, data_type)?;
    let mut index_file = index_store
        .new_index_file(FRAGMENT_STATS_FILENAME, batch.schema())
        .await?;
    index_file.write_record_batch(batch).await?;
    let file = index_file.finish_with_metadata(HashMap::new()).await?;
    Ok(vec![file])
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

/// Fragment column statistics index: one summary per fragment for one column.
pub struct FragmentColumnStatsIndex {
    records: Vec<FragmentStatsRecord>,
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
        self.records
            .iter()
            .map(|r| r.deep_size_of_children(context) + std::mem::size_of::<FragmentStatsRecord>())
            .sum()
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
        Ok(Arc::new(Self { records, data_type }))
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
        FragmentColumnStatsIndex {
            records,
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
}
