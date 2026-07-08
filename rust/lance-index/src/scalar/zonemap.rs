// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Zone Map Index
//!
//! Zone maps are a columnar database technique for predicate pushdown and scan pruning.
//! They break data into fixed-size chunks called "zones" and maintain summary statistics
//! (min, max, null count) for each zone. This enables efficient filtering by eliminating
//! zones that cannot contain matching values.
//!
//! Zone maps are "inexact" filters - they can definitively exclude zones but may include
//! false positives that require rechecking.
//!
//!
use crate::Any;
use crate::pbold;
use crate::scalar::expression::{SargableQueryParser, ScalarQueryParser};
use crate::scalar::registry::{
    BasicTrainer, ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
};
use crate::scalar::{
    BuiltinIndexType, CreatedIndex, IndexFile, SargableQuery, ScalarIndexParams, UpdateCriteria,
    compute_next_prefix,
};
use lance_arrow_stats::StatisticsAccumulator;
use lance_core::cache::{LanceCache, WeakLanceCache};
use lance_core::utils::row_addr_remap::RowAddrRemap;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

use arrow_array::{
    ArrayRef, RecordBatch, UInt32Array, UInt64Array, new_empty_array, new_null_array,
};
use arrow_schema::{DataType, Field};
use datafusion::execution::SendableRecordBatchStream;
use datafusion_common::ScalarValue;
use lance_select::RowAddrTreeMap;
use std::{collections::HashMap, sync::Arc};

use super::{AnyQuery, IndexStore, MetricsCollector, ScalarIndex, SearchResult};
use crate::scalar::RowIdRemapper;
use crate::{Index, IndexType};
use async_trait::async_trait;
use lance_core::Error;
use lance_core::Result;
use lance_core::deepsize::DeepSizeOf;
use roaring::RoaringBitmap;

use super::zoned::{ZoneBound, ZoneProcessor, ZoneTrainer, rebuild_zones, search_zones};
const ROWS_PER_ZONE_DEFAULT: u64 = 8192; // 1 zone every two batches

const ZONEMAP_FILENAME: &str = "zonemap.lance";
const ZONEMAP_SIZE_META_KEY: &str = "rows_per_zone";
const NULL_BITMAP_META_KEY: &str = "null_bitmap";
const ZONEMAP_INDEX_VERSION: u32 = 0;

/// Basic stats about zonemap index
#[derive(Debug, PartialEq, Clone)]
struct ZoneMapStatistics {
    min: ScalarValue,
    max: ScalarValue,
    null_count: u32,
    // only apply to float type
    nan_count: u32,
    // Bound of this zone within the fragment. Persisted as three separate columns
    // (fragment_id, zone_start, zone_length) in the index file.
    bound: ZoneBound,
}

impl DeepSizeOf for ZoneMapStatistics {
    fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
        // Estimate sizes for ScalarValue
        let min_size = self.min.size() - std::mem::size_of::<ScalarValue>();
        let max_size = self.max.size() - std::mem::size_of::<ScalarValue>();

        min_size + max_size
    }
}

impl AsRef<ZoneBound> for ZoneMapStatistics {
    fn as_ref(&self) -> &ZoneBound {
        &self.bound
    }
}

/// ZoneMap index
/// At high level it's a columnar database technique for predicate push down and scan pruning.
/// It breaks data into fixed-size chunks called `zones` and store summary statistics(min, max, null_count,
/// nan_count, fragment_id, local_row_offset) for each zone. It enables efficient filtering by skipping zones that do not contain matching values
///
/// This is an inexact filter, similar to a bloom filter. It can return false positives that require rechecking.
///
/// Note that it cannot return false negatives.
/// Input:
/// * Fragment 1: - 10 rows   -> 0  -> 9
/// * Fragment 2: - 7 rows    -> 10 -> 16
/// * Fragment 3: - 4 rows    -> 20 -> 23
/// * Zone size AKA “rows_per_zone” (from user) - 5
///
/// Output:
/// fragment id | min | max | zone_length
/// 1           | 0   |  4  | 5
/// 1           | 5   |  9  | 5
/// 2           | 10  | 14  | 5
/// 2           | 15  | 16  | 2
/// 3           | 20  | 23  | 4
pub struct ZoneMapIndex {
    zones: Vec<ZoneMapStatistics>,
    data_type: DataType,
    // The maximum rows per zone provided by user
    rows_per_zone: u64,
    store: Arc<dyn IndexStore>,
    fri: Option<Arc<dyn RowIdRemapper>>,
    index_cache: WeakLanceCache,
    // Exact set of null row addresses across all zones; None when loaded from an
    // older index that did not persist this bitmap.
    null_rows: Option<RowAddrTreeMap>,
}

impl std::fmt::Debug for ZoneMapIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZoneMapIndex")
            .field("zones", &self.zones)
            .field("data_type", &self.data_type)
            .field("rows_per_zone", &self.rows_per_zone)
            .field("store", &self.store)
            .field("fri", &self.fri)
            .field("index_cache", &self.index_cache)
            .finish()
    }
}

impl DeepSizeOf for ZoneMapIndex {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.zones.deep_size_of_children(context) + self.null_rows.deep_size_of_children(context)
    }
}

impl ZoneMapIndex {
    fn scalar_is_nan(value: &ScalarValue) -> bool {
        match value {
            ScalarValue::Float16(Some(value)) => value.is_nan(),
            ScalarValue::Float32(Some(value)) => value.is_nan(),
            ScalarValue::Float64(Some(value)) => value.is_nan(),
            _ => false,
        }
    }

    /// Returns true if the zone has a non-null, non-NaN min value.
    fn zone_has_finite_min(zone: &ZoneMapStatistics) -> bool {
        !(zone.min.is_null() || Self::scalar_is_nan(&zone.min))
    }

    /// Returns true if both min and max are non-null / non-NaN.
    fn zone_has_finite_extrema(zone: &ZoneMapStatistics) -> bool {
        Self::zone_has_finite_min(zone) && !(zone.max.is_null() || Self::scalar_is_nan(&zone.max))
    }

    /// Global `[min, max]` folded across one or more ZoneMap segments (the
    /// disjoint per-column segments of a multi-segment index), without a scan.
    ///
    /// `None` when no zone has a finite bound, or when any zone's `max` is NaN:
    /// `ScalarValue`'s total order ranks NaN above every finite value, so a
    /// NaN-bearing zone hides its true finite max and no sound upper bound exists
    /// without a scan — folding only the finite maxes would yield a *subset* that
    /// prunes live rows. Folding raw zones (not each segment's `value_range`)
    /// keeps this exact across segments.
    ///
    /// Otherwise the range is a superset of the segments' live values,
    /// conservative under deletion vectors: safe to prune with, not guaranteed
    /// tight. The caller must ensure the segments jointly cover every live
    /// fragment.
    pub fn value_range_over<'a>(
        segments: impl IntoIterator<Item = &'a Self>,
    ) -> Option<(ScalarValue, ScalarValue)> {
        let mut min: Option<&ScalarValue> = None;
        let mut max: Option<&ScalarValue> = None;
        for zone in segments.into_iter().flat_map(|seg| seg.zones.iter()) {
            if Self::scalar_is_nan(&zone.max) {
                return None;
            }
            if Self::scalar_is_finite_bound(&zone.min)
                && min.is_none_or(|cur| zone.min.partial_cmp(cur).is_some_and(|o| o.is_lt()))
            {
                min = Some(&zone.min);
            }
            if Self::scalar_is_finite_bound(&zone.max)
                && max.is_none_or(|cur| zone.max.partial_cmp(cur).is_some_and(|o| o.is_gt()))
            {
                max = Some(&zone.max);
            }
        }
        Some((min?.clone(), max?.clone()))
    }

    /// A scalar usable as a global-range bound: non-null and, for floats, non-NaN.
    fn scalar_is_finite_bound(v: &ScalarValue) -> bool {
        !v.is_null() && !Self::scalar_is_nan(v)
    }

    /// Evaluates whether a zone could potentially contain values matching the query.
    ///
    /// NaN query values use the explicit `nan_count`. For finite query values,
    /// `ScalarValue` total ordering keeps finite values below a stored NaN max,
    /// so zones with finite values plus NaNs remain conservative false positives.
    fn evaluate_zone_against_query(
        &self,
        zone: &ZoneMapStatistics,
        query: &SargableQuery,
    ) -> Result<bool> {
        use std::ops::Bound;

        match query {
            SargableQuery::IsNull() => {
                // Zone contains matching values if it has any null values
                Ok(zone.null_count > 0)
            }
            SargableQuery::Equals(target) => {
                // Zone contains matching values if target falls within [min, max] range
                // Handle null values - if target is null, check null_count
                if target.is_null() {
                    return Ok(zone.null_count > 0);
                }

                // Handle NaN values - if target is NaN, check nan_count
                let is_nan = match target {
                    ScalarValue::Float16(Some(f)) => f.is_nan(),
                    ScalarValue::Float32(Some(f)) => f.is_nan(),
                    ScalarValue::Float64(Some(f)) => f.is_nan(),
                    _ => false,
                };

                if is_nan {
                    return Ok(zone.nan_count > 0);
                }

                if !Self::zone_has_finite_min(zone) {
                    return Ok(false);
                }

                Ok(target >= &zone.min && target <= &zone.max)
            }
            SargableQuery::Range(start, end) => {
                // Zone overlaps with query range if there's any intersection between
                // the zone's [min, max] and the query's range
                if !Self::zone_has_finite_min(zone) {
                    return Ok(false);
                }

                let zone_min = &zone.min;
                let zone_max = &zone.max;

                let start_check = match start {
                    Bound::Unbounded => true,
                    Bound::Included(s) => {
                        // Handle NaN in range bounds - NaN is greater than all finite values
                        match s {
                            ScalarValue::Float16(Some(f)) => {
                                if f.is_nan() {
                                    return Ok(zone.nan_count > 0);
                                }
                            }
                            ScalarValue::Float32(Some(f)) => {
                                if f.is_nan() {
                                    return Ok(zone.nan_count > 0);
                                }
                            }
                            ScalarValue::Float64(Some(f)) => {
                                if f.is_nan() {
                                    return Ok(zone.nan_count > 0);
                                }
                            }
                            _ => {}
                        }
                        // Handle the case where zone_max is NaN
                        // If zone_max is NaN, the zone contains both finite values and NaN
                        // Since we don't know the actual max, we'll be conservative and include the zone
                        match zone_max {
                            ScalarValue::Float16(Some(f)) if f.is_nan() => true,
                            ScalarValue::Float32(Some(f)) if f.is_nan() => true,
                            ScalarValue::Float64(Some(f)) if f.is_nan() => true,
                            _ => zone_max >= s,
                        }
                    }
                    Bound::Excluded(s) => {
                        // Handle NaN in range bounds
                        match s {
                            ScalarValue::Float16(Some(f)) => {
                                if f.is_nan() {
                                    return Ok(false); // Nothing is greater than NaN
                                }
                            }
                            ScalarValue::Float32(Some(f)) => {
                                if f.is_nan() {
                                    return Ok(false); // Nothing is greater than NaN
                                }
                            }
                            ScalarValue::Float64(Some(f)) => {
                                if f.is_nan() {
                                    return Ok(false); // Nothing is greater than NaN
                                }
                            }
                            _ => {}
                        }
                        zone_max > s
                    }
                };

                let end_check = match end {
                    Bound::Unbounded => true,
                    Bound::Included(e) => {
                        // Handle NaN in range bounds
                        match e {
                            ScalarValue::Float16(Some(f)) => {
                                if f.is_nan() {
                                    // NaN is included, so check if zone has NaN values or finite values
                                    return Ok(zone.nan_count > 0 || zone_min <= e);
                                }
                            }
                            ScalarValue::Float32(Some(f)) => {
                                if f.is_nan() {
                                    return Ok(zone.nan_count > 0 || zone_min <= e);
                                }
                            }
                            ScalarValue::Float64(Some(f)) => {
                                if f.is_nan() {
                                    return Ok(zone.nan_count > 0 || zone_min <= e);
                                }
                            }
                            _ => {}
                        }
                        zone_min <= e
                    }
                    Bound::Excluded(e) => {
                        // Handle NaN in range bounds
                        match e {
                            ScalarValue::Float16(Some(f)) => {
                                if f.is_nan() {
                                    // Everything is less than NaN, so include all finite values
                                    return Ok(true);
                                }
                            }
                            ScalarValue::Float32(Some(f)) => {
                                if f.is_nan() {
                                    return Ok(true);
                                }
                            }
                            ScalarValue::Float64(Some(f)) => {
                                if f.is_nan() {
                                    return Ok(true);
                                }
                            }
                            _ => {}
                        }
                        zone_min < e
                    }
                };

                Ok(start_check && end_check)
            }
            SargableQuery::IsIn(values) => {
                // Zone contains matching values if any value in the set falls within [min, max]
                Ok(values.iter().any(|value| {
                    if value.is_null() {
                        zone.null_count > 0
                    } else {
                        match value {
                            ScalarValue::Float16(Some(f)) => {
                                if f.is_nan() {
                                    zone.nan_count > 0
                                } else if !Self::zone_has_finite_min(zone) {
                                    false
                                } else {
                                    value >= &zone.min && value <= &zone.max
                                }
                            }
                            ScalarValue::Float32(Some(f)) => {
                                if f.is_nan() {
                                    zone.nan_count > 0
                                } else if !Self::zone_has_finite_min(zone) {
                                    false
                                } else {
                                    value >= &zone.min && value <= &zone.max
                                }
                            }
                            ScalarValue::Float64(Some(f)) => {
                                if f.is_nan() {
                                    zone.nan_count > 0
                                } else if !Self::zone_has_finite_min(zone) {
                                    false
                                } else {
                                    value >= &zone.min && value <= &zone.max
                                }
                            }
                            _ => {
                                Self::zone_has_finite_extrema(zone)
                                    && value >= &zone.min
                                    && value <= &zone.max
                            }
                        }
                    }
                }))
            }
            SargableQuery::FullTextSearch(_) => Err(Error::not_supported_source(
                "full text search is not supported for zonemap indexes".into(),
            )),
            SargableQuery::LikePrefix(prefix) => {
                // For prefix matching, a zone can match if:
                // - zone.max >= prefix (there could be values >= prefix)
                // - zone.min < next_prefix (there could be values < next_prefix)
                //
                // For example, prefix "foo":
                // - Zone [aaa, azz]: max="azz" < "foo", so no match
                // - Zone [fa, foz]: min="fa" < "fop", max="foz" >= "foo", so potential match
                // - Zone [fop, fzz]: min="fop" >= "fop", so no match

                let prefix_str = match prefix {
                    ScalarValue::Utf8(Some(s)) => s.as_str(),
                    ScalarValue::LargeUtf8(Some(s)) => s.as_str(),
                    _ => return Ok(true), // Conservative: include zone if not a string prefix
                };

                // Empty prefix matches everything
                if prefix_str.is_empty() {
                    return Ok(true);
                }

                // Check zone.max >= prefix
                let max_check = &zone.max >= prefix;
                if !max_check {
                    return Ok(false);
                }

                // Compute next_prefix by incrementing the last byte
                // If the prefix ends with 0xFF bytes, we need to handle overflow
                let next_prefix = compute_next_prefix(prefix_str);

                match next_prefix {
                    Some(next) => {
                        // Check zone.min < next_prefix
                        let next_scalar = match prefix {
                            ScalarValue::Utf8(_) => ScalarValue::Utf8(Some(next)),
                            ScalarValue::LargeUtf8(_) => ScalarValue::LargeUtf8(Some(next)),
                            _ => return Ok(true),
                        };
                        Ok(zone.min < next_scalar)
                    }
                    None => {
                        // No upper bound (prefix is all 0xFF), so any zone with max >= prefix matches
                        Ok(true)
                    }
                }
            }
        }
    }

    /// Load the scalar index from storage
    async fn load(
        store: Arc<dyn IndexStore>,
        fri: Option<Arc<dyn RowIdRemapper>>,
        index_cache: &LanceCache,
    ) -> Result<Arc<Self>>
    where
        Self: Sized,
    {
        let index_file = store.open_index_file(ZONEMAP_FILENAME).await?;
        let zone_maps = index_file
            .read_range(0..index_file.num_rows(), None)
            .await?;
        let file_schema = index_file.schema();

        let rows_per_zone: u64 = file_schema
            .metadata
            .get(ZONEMAP_SIZE_META_KEY)
            .and_then(|bs| bs.parse().ok())
            .unwrap_or(ROWS_PER_ZONE_DEFAULT);

        let null_rows = if let Some(idx_str) = file_schema.metadata.get(NULL_BITMAP_META_KEY) {
            let idx = idx_str.parse::<u32>().map_err(|e| {
                Error::invalid_input(format!("invalid null bitmap buffer index: {e}"))
            })?;
            let bytes = index_file.read_global_buffer(idx).await?;
            Some(RowAddrTreeMap::deserialize_from(bytes.as_ref())?)
        } else {
            None
        };

        Ok(Arc::new(Self::try_from_serialized(
            zone_maps,
            store,
            fri,
            index_cache,
            rows_per_zone,
            null_rows,
        )?))
    }

    fn try_from_serialized(
        data: RecordBatch,
        store: Arc<dyn IndexStore>,
        fri: Option<Arc<dyn RowIdRemapper>>,
        index_cache: &LanceCache,
        rows_per_zone: u64,
        null_rows: Option<RowAddrTreeMap>,
    ) -> Result<Self> {
        // The RecordBatch should have columns: min, max, null_count
        let min_col = data
            .column_by_name("min")
            .ok_or_else(|| Error::invalid_input("ZoneMapIndex: missing 'min' column"))?;
        let max_col = data
            .column_by_name("max")
            .ok_or_else(|| Error::invalid_input("ZoneMapIndex: missing 'max' column"))?;
        let null_count_col = data
            .column_by_name("null_count")
            .ok_or_else(|| Error::invalid_input("ZoneMapIndex: missing 'null_count' column"))?
            .as_any()
            .downcast_ref::<arrow_array::UInt32Array>()
            .ok_or_else(|| {
                Error::invalid_input("ZoneMapIndex: 'null_count' column is not UInt32")
            })?;
        let nan_count_col = data
            .column_by_name("nan_count")
            .ok_or_else(|| Error::invalid_input("ZoneMapIndex: missing 'nan_count' column"))?
            .as_any()
            .downcast_ref::<arrow_array::UInt32Array>()
            .ok_or_else(|| {
                Error::invalid_input("ZoneMapIndex: 'nan_count' column is not UInt32")
            })?;
        let zone_length = data
            .column_by_name("zone_length")
            .ok_or_else(|| Error::invalid_input("ZoneMapIndex: missing 'zone_length' column"))?
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| {
                Error::invalid_input("ZoneMapIndex: 'zone_length' column is not UInt64")
            })?;

        let fragment_id_col = data
            .column_by_name("fragment_id")
            .ok_or_else(|| Error::invalid_input("ZoneMapIndex: missing 'fragment_id' column"))?
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| {
                Error::invalid_input("ZoneMapIndex: 'fragment_id' column is not UInt64")
            })?;

        let zone_start_col = data
            .column_by_name("zone_start")
            .ok_or_else(|| Error::invalid_input("ZoneMapIndex: missing 'zone_start' column"))?
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| {
                Error::invalid_input("ZoneMapIndex: 'zone_start' column is not UInt64")
            })?;

        let data_type = min_col.data_type().clone();

        if data.num_rows() == 0 {
            return Ok(Self {
                zones: Vec::new(),
                data_type,
                rows_per_zone,
                store,
                fri,
                index_cache: WeakLanceCache::from(index_cache),
                null_rows,
            });
        }

        let num_zones = data.num_rows();
        let mut zones = Vec::with_capacity(num_zones);

        for i in 0..num_zones {
            let min = ScalarValue::try_from_array(min_col, i)?;
            let max = ScalarValue::try_from_array(max_col, i)?;
            let null_count = null_count_col.value(i);
            let nan_count = nan_count_col.value(i);
            zones.push(ZoneMapStatistics {
                min,
                max,
                null_count,
                nan_count,
                bound: ZoneBound {
                    fragment_id: fragment_id_col.value(i),
                    start: zone_start_col.value(i),
                    length: zone_length.value(i) as usize,
                },
            });
        }

        Ok(Self {
            zones,
            data_type,
            rows_per_zone,
            store,
            fri,
            index_cache: WeakLanceCache::from(index_cache),
            null_rows,
        })
    }
}

#[async_trait]
impl Index for ZoneMapIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    async fn prewarm(&self) -> Result<()> {
        // Not much to prewarm
        Ok(())
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "num_zones": self.zones.len(),
            "rows_per_zone": self.rows_per_zone,
        }))
    }

    fn index_type(&self) -> IndexType {
        IndexType::ZoneMap
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let mut frag_ids = RoaringBitmap::new();

        // Loop through zones and add unique fragment IDs to the bitmap
        for zone in &self.zones {
            frag_ids.insert(zone.bound.fragment_id as u32);
        }

        Ok(frag_ids)
    }
}

#[async_trait]
impl ScalarIndex for ZoneMapIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let query = query.as_any().downcast_ref::<SargableQuery>().unwrap();
        if let SargableQuery::IsNull() = query
            && let Some(null_rows) = &self.null_rows
        {
            return Ok(SearchResult::exact(null_rows.clone()));
        }

        search_zones(&self.zones, metrics, |zone| {
            self.evaluate_zone_against_query(zone, query)
        })
    }

    fn can_remap(&self) -> bool {
        false
    }

    /// Remap the row ids, creating a new remapped version of this index in `dest_store`
    async fn remap(
        &self,
        _mapping: &RowAddrRemap,
        _dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        Err(Error::invalid_input_source(
            "ZoneMapIndex does not support remap".into(),
        ))
    }

    /// Add the new data , creating an updated version of the index in `dest_store`
    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
        _old_data_filter: Option<super::OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        // Train new zones for the incoming data stream
        let schema = new_data.schema();
        let value_type = schema.field(0).data_type().clone();

        let options = ZoneMapIndexBuilderParams::new(self.rows_per_zone);
        let processor = ZoneMapProcessor::new(value_type.clone())?;
        let trainer = ZoneTrainer::new(processor, self.rows_per_zone)?;
        let (updated_zones, new_null_rows) = rebuild_zones(&self.zones, trainer, new_data).await?;

        // Merge existing and new null rows.  If the existing index had no null bitmap
        // (legacy format — null positions unknown), preserve that None: updating cannot
        // recover the missing information, and claiming the result has zero nulls would
        // be a false negative.  Only a full retrain produces a fresh, complete bitmap.
        let merged_null_rows = self.null_rows.as_ref().map(|existing| {
            let mut merged = existing.clone();
            merged |= &new_null_rows;
            merged
        });

        // Serialize the combined zones back into the index file
        let mut builder = ZoneMapIndexBuilder::try_new(options, self.data_type.clone())?;
        builder.options.rows_per_zone = self.rows_per_zone;
        builder.maps = updated_zones;
        builder.null_rows = merged_null_rows;
        let files = builder.write_index(dest_store).await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::ZoneMapIndexDetails::default())
                .unwrap(),
            index_version: ZONEMAP_INDEX_VERSION,
            files,
        })
    }

    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria::only_new_data(
            TrainingCriteria::new(TrainingOrdering::Addresses).with_row_addr(),
        )
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        let params = serde_json::to_value(ZoneMapIndexBuilderParams::new(self.rows_per_zone))?;
        Ok(ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap).with_params(&params))
    }

    /// Single-segment `[min, max]` folded from this index's zones; see
    /// [`value_range_over`](Self::value_range_over) for the full contract.
    fn value_range(&self) -> Option<(ScalarValue, ScalarValue)> {
        Self::value_range_over([self])
    }
}

/// Merge caller-selected ZoneMap segments into one self-contained segment.
pub async fn merge_zonemap_indices(
    source_indices: &[&ZoneMapIndex],
    dest_store: &dyn IndexStore,
    fragment_filter: &RoaringBitmap,
) -> Result<CreatedIndex> {
    let first = source_indices.first().ok_or_else(|| {
        Error::invalid_input("merge_zonemap_indices requires at least one source index")
    })?;
    let rows_per_zone = first.rows_per_zone;
    let data_type = first.data_type.clone();

    let mut zones = Vec::new();
    let mut merged_null_rows = RowAddrTreeMap::new();
    let mut any_missing_bitmap = false;
    for source in source_indices {
        if source.rows_per_zone != rows_per_zone {
            return Err(Error::invalid_input(format!(
                "cannot merge ZoneMap segments with different rows_per_zone values: {} and {}",
                rows_per_zone, source.rows_per_zone
            )));
        }
        if source.data_type != data_type {
            return Err(Error::invalid_input(format!(
                "cannot merge ZoneMap segments with different value types: {:?} and {:?}",
                data_type, source.data_type
            )));
        }
        zones.extend(
            source
                .zones
                .iter()
                .filter(|zone| {
                    u32::try_from(zone.bound.fragment_id)
                        .is_ok_and(|fragment_id| fragment_filter.contains(fragment_id))
                })
                .cloned(),
        );
        match &source.null_rows {
            Some(null_rows) => {
                let mut filtered = null_rows.clone();
                filtered.retain_fragments(fragment_filter.iter());
                merged_null_rows |= &filtered;
            }
            None => any_missing_bitmap = true,
        }
    }
    zones.sort_by_key(|zone| (zone.bound.fragment_id, zone.bound.start));

    let mut builder =
        ZoneMapIndexBuilder::try_new(ZoneMapIndexBuilderParams::new(rows_per_zone), data_type)?;
    builder.maps = zones;
    if !any_missing_bitmap {
        builder.null_rows = Some(merged_null_rows);
    }
    let files = builder.write_index(dest_store).await?;

    Ok(CreatedIndex {
        index_details: prost_types::Any::from_msg(&pbold::ZoneMapIndexDetails::default()).unwrap(),
        index_version: ZONEMAP_INDEX_VERSION,
        files,
    })
}

fn default_rows_per_zone() -> u64 {
    *DEFAULT_ROWS_PER_ZONE
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneMapIndexBuilderParams {
    #[serde(default = "default_rows_per_zone")]
    rows_per_zone: u64,
}

static DEFAULT_ROWS_PER_ZONE: LazyLock<u64> = LazyLock::new(|| {
    std::env::var("LANCE_ZONEMAP_DEFAULT_ROWS_PER_ZONE")
        .unwrap_or_else(|_| (ROWS_PER_ZONE_DEFAULT).to_string())
        .parse()
        .expect("failed to parse LANCE_ZONEMAP_DEFAULT_ROWS_PER_ZONE")
});

impl Default for ZoneMapIndexBuilderParams {
    fn default() -> Self {
        Self {
            rows_per_zone: *DEFAULT_ROWS_PER_ZONE,
        }
    }
}

impl ZoneMapIndexBuilderParams {
    pub fn new(rows_per_zone: u64) -> Self {
        Self { rows_per_zone }
    }

    pub fn rows_per_zone(&self) -> u64 {
        self.rows_per_zone
    }
}

// A builder for zonemap index
pub struct ZoneMapIndexBuilder {
    options: ZoneMapIndexBuilderParams,

    items_type: DataType,
    maps: Vec<ZoneMapStatistics>,
    // None means "legacy index — null positions unknown"; Some means a complete bitmap.
    // write_index omits the null-bitmap global buffer when this is None, preserving the
    // legacy format so that downstream searches remain conservative.
    null_rows: Option<RowAddrTreeMap>,
}

impl ZoneMapIndexBuilder {
    pub fn try_new(options: ZoneMapIndexBuilderParams, items_type: DataType) -> Result<Self> {
        Ok(Self {
            options,
            items_type,
            maps: Vec::new(),
            null_rows: None,
        })
    }

    /// Train the builder using the shared zone trainer.  The input stream must contain
    /// the value column followed by `_rowaddr`, matching the dataset scan order enforced
    /// by the scalar index registry.
    pub async fn train(&mut self, batches_source: SendableRecordBatchStream) -> Result<()> {
        let processor = ZoneMapProcessor::new(self.items_type.clone())?;
        let trainer = ZoneTrainer::new(processor, self.options.rows_per_zone)?;
        let (maps, null_rows) = trainer.train(batches_source).await?;
        self.maps = maps;
        self.null_rows = Some(null_rows);
        Ok(())
    }

    fn zonemap_stats_as_batch(&self) -> Result<RecordBatch> {
        // Flush self.maps as a RecordBatch
        let mins = if self.maps.is_empty() {
            new_empty_array(&self.items_type)
        } else {
            ScalarValue::iter_to_array(self.maps.iter().map(|stat| stat.min.clone()))?
        };
        let maxs = if self.maps.is_empty() {
            new_empty_array(&self.items_type)
        } else {
            ScalarValue::iter_to_array(self.maps.iter().map(|stat| stat.max.clone()))?
        };
        let null_counts =
            UInt32Array::from_iter_values(self.maps.iter().map(|stat| stat.null_count));

        let nan_counts = UInt32Array::from_iter_values(self.maps.iter().map(|stat| stat.nan_count));

        let fragment_ids =
            UInt64Array::from_iter_values(self.maps.iter().map(|stat| stat.bound.fragment_id));

        let zone_lengths =
            UInt64Array::from_iter_values(self.maps.iter().map(|stat| stat.bound.length as u64));

        let zone_starts =
            UInt64Array::from_iter_values(self.maps.iter().map(|stat| stat.bound.start));

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            // min and max can be null if the entire batch is null values
            Field::new("min", self.items_type.clone(), true),
            Field::new("max", self.items_type.clone(), true),
            Field::new("null_count", DataType::UInt32, false),
            Field::new("nan_count", DataType::UInt32, false),
            Field::new("fragment_id", DataType::UInt64, false),
            Field::new("zone_start", DataType::UInt64, false),
            Field::new("zone_length", DataType::UInt64, false),
        ]));

        let columns: Vec<ArrayRef> = vec![
            mins,
            maxs,
            Arc::new(null_counts) as ArrayRef,
            Arc::new(nan_counts) as ArrayRef,
            Arc::new(fragment_ids) as ArrayRef,
            Arc::new(zone_starts) as ArrayRef,
            Arc::new(zone_lengths) as ArrayRef,
        ];
        Ok(RecordBatch::try_new(schema, columns)?)
    }

    pub async fn write_index(self, index_store: &dyn IndexStore) -> Result<Vec<IndexFile>> {
        let record_batch = self.zonemap_stats_as_batch()?;

        let mut file_schema = record_batch.schema().as_ref().clone();
        file_schema.metadata.insert(
            ZONEMAP_SIZE_META_KEY.to_string(),
            self.options.rows_per_zone.to_string(),
        );

        let mut index_file = index_store
            .new_index_file(ZONEMAP_FILENAME, Arc::new(file_schema))
            .await?;
        index_file.write_record_batch(record_batch).await?;

        let zonemap_file = if let Some(null_rows) = self.null_rows {
            let mut null_bitmap_bytes = Vec::with_capacity(null_rows.serialized_size());
            null_rows.serialize_into(&mut null_bitmap_bytes)?;
            let null_bitmap_idx = index_file
                .add_global_buffer(bytes::Bytes::from(null_bitmap_bytes))
                .await?;
            index_file
                .finish_with_metadata(HashMap::from([(
                    NULL_BITMAP_META_KEY.to_string(),
                    null_bitmap_idx.to_string(),
                )]))
                .await?
        } else {
            index_file.finish_with_metadata(HashMap::new()).await?
        };

        Ok(vec![zonemap_file])
    }
}

/// Index-specific processor that computes min/max statistics for each zone while the
/// trainer takes care of chunking and fragment boundaries.
struct ZoneMapProcessor {
    data_type: DataType,
    statistics: StatisticsAccumulator,
}

impl ZoneMapProcessor {
    fn new(data_type: DataType) -> Result<Self> {
        Ok(Self {
            statistics: StatisticsAccumulator::new(&data_type),
            data_type,
        })
    }

    fn scalar_value_from_stat(
        value: Option<&ArrayRef>,
        data_type: &DataType,
    ) -> Result<ScalarValue> {
        let array = value
            .cloned()
            .unwrap_or_else(|| new_null_array(data_type, 1));
        Ok(ScalarValue::try_from_array(&array, 0)?)
    }

    fn stat_count_to_u32(name: &str, value: u64) -> Result<u32> {
        u32::try_from(value).map_err(|_| {
            Error::invalid_input(format!(
                "{} value {} exceeds the supported UInt32 range",
                name, value
            ))
        })
    }

    fn nan_scalar(data_type: &DataType) -> Option<ScalarValue> {
        match data_type {
            DataType::Float16 => Some(ScalarValue::Float16(Some(half::f16::NAN))),
            DataType::Float32 => Some(ScalarValue::Float32(Some(f32::NAN))),
            DataType::Float64 => Some(ScalarValue::Float64(Some(f64::NAN))),
            _ => None,
        }
    }

    fn max_value_from_stats(
        value: Option<&ArrayRef>,
        data_type: &DataType,
        nan_count: u32,
    ) -> Result<ScalarValue> {
        if nan_count > 0
            && let Some(nan) = Self::nan_scalar(data_type)
        {
            // DataFusion's max accumulator surfaced NaN as the zone max.  Keep
            // that stored zonemap shape while using arrow_stats so existing
            // range/equality pruning remains conservative around NaN.
            return Ok(nan);
        }
        Self::scalar_value_from_stat(value, data_type)
    }
}

impl ZoneProcessor for ZoneMapProcessor {
    type ZoneStatistics = ZoneMapStatistics;

    fn process_chunk(&mut self, array: &ArrayRef) -> Result<()> {
        self.statistics.update(array)?;
        Ok(())
    }

    fn finish_zone(&mut self, bound: ZoneBound) -> Result<Self::ZoneStatistics> {
        let statistics = self.statistics.statistics();
        let nan_count = Self::stat_count_to_u32("nan_count", statistics.nan_count.unwrap_or(0))?;
        Ok(ZoneMapStatistics {
            min: Self::scalar_value_from_stat(
                statistics.min.as_ref().map(|scalar| scalar.as_array()),
                &self.data_type,
            )?,
            max: Self::max_value_from_stats(
                statistics.max.as_ref().map(|scalar| scalar.as_array()),
                &self.data_type,
                nan_count,
            )?,
            null_count: Self::stat_count_to_u32("null_count", statistics.null_count)?,
            nan_count,
            bound,
        })
    }

    fn reset(&mut self) -> Result<()> {
        self.statistics.reset();
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct ZoneMapIndexPlugin;

impl ZoneMapIndexPlugin {
    async fn train_zonemap_index(
        batches_source: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        options: Option<ZoneMapIndexBuilderParams>,
    ) -> Result<Vec<IndexFile>> {
        let value_type = batches_source.schema().field(0).data_type().clone();

        let mut builder = ZoneMapIndexBuilder::try_new(options.unwrap_or_default(), value_type)?;

        builder.train(batches_source).await?;

        builder.write_index(index_store).await
    }
}

pub struct ZoneMapIndexTrainingRequest {
    pub params: ZoneMapIndexBuilderParams,
    pub criteria: TrainingCriteria,
}

impl ZoneMapIndexTrainingRequest {
    pub fn new(params: ZoneMapIndexBuilderParams) -> Self {
        Self {
            params,
            criteria: TrainingCriteria::new(TrainingOrdering::Addresses).with_row_addr(),
        }
    }
}

impl TrainingRequest for ZoneMapIndexTrainingRequest {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

#[async_trait]
impl BasicTrainer for ZoneMapIndexPlugin {
    fn new_training_request(
        &self,
        params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        if field.data_type().is_nested() {
            return Err(Error::invalid_input_source(
                "A zone map index can only be created on a non-nested field.".into(),
            ));
        }

        let params = serde_json::from_str::<ZoneMapIndexBuilderParams>(params)?;

        Ok(Box::new(ZoneMapIndexTrainingRequest::new(params)))
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        _fragment_ids: Option<Vec<u32>>,
        _progress: Arc<dyn crate::progress::IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        let request = (request as Box<dyn std::any::Any>)
            .downcast::<ZoneMapIndexTrainingRequest>()
            .map_err(|_| {
                Error::invalid_input_source(
                    "must provide training request created by new_training_request".into(),
                )
            })?;
        let files = Self::train_zonemap_index(data, index_store, Some(request.params)).await?;
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::ZoneMapIndexDetails::default())
                .unwrap(),
            index_version: ZONEMAP_INDEX_VERSION,
            files,
        })
    }
}

#[async_trait]
impl ScalarIndexPlugin for ZoneMapIndexPlugin {
    fn basic_trainer(&self) -> Option<&dyn BasicTrainer> {
        Some(self)
    }

    fn name(&self) -> &str {
        "ZoneMap"
    }

    fn provides_exact_answer(&self) -> bool {
        false
    }

    fn version(&self) -> u32 {
        ZONEMAP_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        Some(Box::new(SargableQueryParser::new(
            index_name,
            self.name().to_string(),
            true,
        )))
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        Ok(ZoneMapIndex::load(index_store, frag_reuse_index, cache).await? as Arc<dyn ScalarIndex>)
    }
}

#[cfg(test)]
mod tests {
    use crate::scalar::registry::VALUE_COLUMN_NAME;
    use crate::scalar::{IndexStore, zonemap::ROWS_PER_ZONE_DEFAULT};
    use std::sync::Arc;

    use crate::scalar::zoned::ZoneBound;
    use crate::scalar::zonemap::{ZoneMapIndexPlugin, ZoneMapStatistics};
    use arrow::datatypes::{ArrowPrimitiveType, Float32Type, Int64Type};
    use arrow_array::{Array, PrimitiveArray, RecordBatch, UInt64Array, record_batch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::execution::SendableRecordBatchStream;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use datafusion_common::ScalarValue;
    use futures::{StreamExt, TryStreamExt, stream};
    use lance_core::utils::tempfile::TempObjDir;
    use lance_core::{
        ROW_ADDR,
        cache::{LanceCache, WeakLanceCache},
    };
    use lance_datafusion::datagen::DatafusionDatagenExt;
    use lance_datagen::ArrayGeneratorExt;
    use lance_datagen::{BatchCount, RowCount, array};
    use lance_io::object_store::ObjectStore;
    use lance_select::RowAddrTreeMap;

    use crate::scalar::{
        SargableQuery, ScalarIndex, SearchResult,
        lance_format::LanceIndexStore,
        zonemap::{
            ZONEMAP_FILENAME, ZONEMAP_SIZE_META_KEY, ZoneMapIndex, ZoneMapIndexBuilderParams,
            merge_zonemap_indices,
        },
    };

    // Add missing imports for the tests
    use crate::Index; // Import Index trait to access calculate_included_frags
    use crate::metrics::NoOpMetricsCollector;
    use roaring::RoaringBitmap; // Import RoaringBitmap for the test
    use std::collections::Bound;

    // Adds a _rowaddr column emulating each batch as a new fragment
    fn add_row_addr(stream: SendableRecordBatchStream) -> SendableRecordBatchStream {
        let schema = stream.schema();
        let schema_with_row_addr = Arc::new(Schema::new(vec![
            schema.field(0).clone(),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let schema = schema_with_row_addr.clone();
        let stream = stream.enumerate().map(move |(frag_id, batch)| {
            let batch = batch.unwrap();
            let row_addr = Arc::new(UInt64Array::from_iter_values(
                (0..batch.num_rows() as u64).map(|off| off + ((frag_id as u64) << 32)),
            ));
            Ok(RecordBatch::try_new(
                schema_with_row_addr.clone(),
                vec![batch.column(0).clone(), row_addr],
            )?)
        });
        Box::pin(RecordBatchStreamAdapter::new(schema, stream))
    }

    /// Build a single-column ZoneMap of primitive type `T` from `fragments`
    /// (one batch -> one fragment), with small zones, then load it back.
    async fn train_and_load<T: ArrowPrimitiveType>(
        fragments: Vec<Vec<Option<T::Native>>>,
    ) -> Arc<ZoneMapIndex>
    where
        PrimitiveArray<T>: From<Vec<Option<T::Native>>>,
    {
        let tmpdir = TempObjDir::default();
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        let schema = Arc::new(Schema::new(vec![Field::new(
            VALUE_COLUMN_NAME,
            T::DATA_TYPE,
            true,
        )]));
        let batches: Vec<RecordBatch> = fragments
            .into_iter()
            .map(|vals| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(PrimitiveArray::<T>::from(vals))],
                )
                .unwrap()
            })
            .collect();
        let stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::iter(batches.into_iter().map(Ok)),
        ));
        let stream = add_row_addr(stream);

        ZoneMapIndexPlugin::train_zonemap_index(
            stream,
            test_store.as_ref(),
            Some(ZoneMapIndexBuilderParams::new(2)),
        )
        .await
        .unwrap();

        ZoneMapIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load ZoneMapIndex")
    }

    #[tokio::test]
    async fn test_value_range_spans_fragments() {
        // Two fragments, multiple zones each; global min/max straddle both.
        let index = train_and_load::<Int64Type>(vec![
            vec![Some(10), Some(50), Some(30)],
            vec![Some(5), Some(99), Some(42)],
        ])
        .await;
        assert_eq!(
            index.value_range(),
            Some((ScalarValue::Int64(Some(5)), ScalarValue::Int64(Some(99))))
        );
    }

    #[tokio::test]
    async fn test_value_range_all_null_is_none() {
        let index = train_and_load::<Int64Type>(vec![vec![None, None, None]]).await;
        assert_eq!(index.value_range(), None);
    }

    #[tokio::test]
    async fn test_value_range_nan_max_is_none() {
        // Zones of size 2: [1.0, 2.0] then [100.0, NaN]. The NaN-bearing zone hides
        // its finite max (100.0), so the only sound answer is None.
        let index = train_and_load::<Float32Type>(vec![vec![
            Some(1.0),
            Some(2.0),
            Some(100.0),
            Some(f32::NAN),
        ]])
        .await;
        assert_eq!(index.value_range(), None);
    }

    #[tokio::test]
    async fn test_value_range_over_folds_segments() {
        // Two disjoint segments of one logical index; the global range straddles
        // both (min and max from `b`), proving the fold spans segments.
        let a = train_and_load::<Int64Type>(vec![vec![Some(5), Some(9)]]).await;
        let b = train_and_load::<Int64Type>(vec![vec![Some(1), Some(20)]]).await;
        assert_eq!(
            ZoneMapIndex::value_range_over([a.as_ref(), b.as_ref()]),
            Some((ScalarValue::Int64(Some(1)), ScalarValue::Int64(Some(20))))
        );
    }

    #[tokio::test]
    async fn test_value_range_over_nan_in_any_segment_is_none() {
        // NaN in one segment hides that segment's finite max; the cross-segment
        // fold must bail to None just as the single-segment path does.
        let a = train_and_load::<Float32Type>(vec![vec![Some(1.0), Some(2.0)]]).await;
        let b = train_and_load::<Float32Type>(vec![vec![Some(100.0), Some(f32::NAN)]]).await;
        assert_eq!(
            ZoneMapIndex::value_range_over([a.as_ref(), b.as_ref()]),
            None
        );
    }

    #[tokio::test]
    async fn test_value_range_over_skips_all_null_segment() {
        // An all-null segment yields no finite zone; folding it with a finite
        // segment returns the finite segment's range (null contributes nothing).
        let a = train_and_load::<Int64Type>(vec![vec![None, None]]).await;
        let b = train_and_load::<Int64Type>(vec![vec![Some(3), Some(7)]]).await;
        assert_eq!(
            ZoneMapIndex::value_range_over([a.as_ref(), b.as_ref()]),
            Some((ScalarValue::Int64(Some(3)), ScalarValue::Int64(Some(7))))
        );
    }

    #[tokio::test]
    async fn test_empty_zonemap_index() {
        let tmpdir = TempObjDir::default();
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let data = arrow_array::Int32Array::from(Vec::<i32>::new());
        let row_ids = arrow_array::UInt64Array::from(Vec::<u64>::new());
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Int32, false),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let data =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();

        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        ZoneMapIndexPlugin::train_zonemap_index(data_stream, test_store.as_ref(), None)
            .await
            .unwrap();

        log::debug!("Successfully wrote the index file");

        // Read the index file back and check its contents
        let index = ZoneMapIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load ZoneMapIndex");
        assert_eq!(index.zones.len(), 0);
        assert_eq!(index.data_type, DataType::Int32);
        assert_eq!(index.rows_per_zone, ROWS_PER_ZONE_DEFAULT);

        // Equals query: null (should match nothing, as there are no nulls)
        let query = SargableQuery::Equals(ScalarValue::Int32(None));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));
    }

    #[tokio::test]
    // Test that a zonemap index can be created with null values from few fragments
    async fn test_null_zonemap_index() {
        let tmpdir = TempObjDir::default();
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let stream = lance_datagen::gen_batch()
            .col(
                VALUE_COLUMN_NAME,
                array::rand::<Float32Type>().with_nulls(&[true, false, false, false, false]),
            )
            .into_df_stream(RowCount::from(5000), BatchCount::from(10));

        // Add _rowaddr column
        let stream = add_row_addr(stream);

        ZoneMapIndexPlugin::train_zonemap_index(
            stream,
            test_store.as_ref(),
            Some(ZoneMapIndexBuilderParams::new(5000)),
        )
        .await
        .unwrap();

        log::debug!("Successfully wrote the index file");

        // Read the index file back and check its contents
        let index = ZoneMapIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load ZoneMapIndex");
        assert_eq!(index.zones.len(), 10);
        for (i, zone) in index.zones.iter().enumerate() {
            assert_eq!(zone.null_count, 1000);
            assert_eq!(zone.nan_count, 0, "Zone {} should have nan_count = 0", i);
            assert_eq!(zone.bound.length, 5000);
            assert_eq!(zone.bound.fragment_id, i as u64);
        }

        // Equals query: null (should match all zones since they contain null values)
        let query = SargableQuery::Equals(ScalarValue::Int32(None));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Create expected RowAddrTreeMap with all zones since they contain null values
        let mut expected = RowAddrTreeMap::new();
        for fragment_id in 0..10 {
            let start = (fragment_id as u64) << 32;
            let end = start + 5000;
            expected.insert_range(start..end);
        }
        assert_eq!(result, SearchResult::at_most(expected));

        // Test update - add new data with Float32 values (matching the original data type)
        let new_data =
            arrow_array::Float32Array::from_iter_values((0..5000).map(|i| i as f32 / 1000.0));
        // Create row addresses for fragment 10 (next fragment after 0-9)
        let new_row_addr =
            UInt64Array::from_iter_values((0..5000).map(|i| (10u64 << 32) | (i as u64)));
        let new_schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Float32, false), // Match original schema
            Field::new(ROW_ADDR, DataType::UInt64, false), // Use _rowaddr as expected by the builder
        ]));
        let new_data_batch = RecordBatch::try_new(
            new_schema.clone(),
            vec![Arc::new(new_data), Arc::new(new_row_addr)],
        )
        .unwrap();
        let new_data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            new_schema,
            stream::once(std::future::ready(Ok(new_data_batch))),
        ));

        // Directly pass the stream with proper row addresses instead of using MockTrainingSource
        // which would regenerate row addresses starting from 0
        index
            .update(new_data_stream, test_store.as_ref(), None)
            .await
            .unwrap();

        // Verify the updated index has more zones
        let updated_index = ZoneMapIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load updated ZoneMapIndex");

        // Should have original 10 zones + 1 new zone (5000 rows with zone size 5000)
        assert_eq!(updated_index.zones.len(), 11);

        // Verify the new zone was added
        let new_zone = &updated_index.zones[10]; // Last zone should be the new one
        assert_eq!(new_zone.bound.fragment_id, 10u64); // New fragment ID
        assert_eq!(new_zone.bound.length, 5000);
        assert_eq!(new_zone.null_count, 0); // New data has no nulls
        assert_eq!(new_zone.nan_count, 0); // New data has no NaN values

        // Test search on updated index - search for null values should still work
        let query = SargableQuery::Equals(ScalarValue::Float32(None));
        let result = updated_index
            .search(&query, &NoOpMetricsCollector)
            .await
            .unwrap();

        // Should match original 10 zones (with nulls) but not the new zone (no nulls)
        let mut expected = RowAddrTreeMap::new();
        for fragment_id in 0..10 {
            let start = (fragment_id as u64) << 32;
            let end = start + 5000;
            expected.insert_range(start..end);
        }
        assert_eq!(result, SearchResult::at_most(expected));

        // Test search for a value that should be in the new zone
        let query = SargableQuery::Equals(ScalarValue::Float32(Some(2.5))); // Value 2500/1000 = 2.5
        let result = updated_index
            .search(&query, &NoOpMetricsCollector)
            .await
            .unwrap();

        // Should match the new zone (fragment 10)
        let mut expected = RowAddrTreeMap::new();
        let start = 10u64 << 32;
        let end = start + 5000;
        expected.insert_range(start..end);
        assert_eq!(result, SearchResult::at_most(expected));
    }

    #[tokio::test]
    async fn test_zonemap_null_handling_in_queries() {
        // Test that zonemap index correctly returns null_list for queries
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Create test data: [0, 5, null]
        let batch = record_batch!(
            (VALUE_COLUMN_NAME, Int64, [Some(0), Some(5), None]),
            (ROW_ADDR, UInt64, [0, 1, 2])
        )
        .unwrap();
        let schema = batch.schema();
        let stream = stream::once(async move { Ok(batch) });
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));

        // Train and write the zonemap index
        ZoneMapIndexPlugin::train_zonemap_index(stream, store.as_ref(), None)
            .await
            .unwrap();

        let cache = LanceCache::with_capacity(1024 * 1024);
        let index = ZoneMapIndex::load(store.clone(), None, &cache)
            .await
            .unwrap();

        // Test 1: Search for value 5 - zonemap should return at_most with all rows
        // Since ZoneMap returns AtMost (superset), it's correct to include nulls in the result
        let query = SargableQuery::Equals(ScalarValue::Int64(Some(5)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        match result {
            SearchResult::AtMost(row_ids) => {
                // Zonemap can't determine exact matches, so it returns all rows in the zone
                // This includes nulls because ZoneMap can't prove they don't match
                let all_rows: Vec<u64> = row_ids
                    .true_rows()
                    .row_addrs()
                    .unwrap()
                    .map(u64::from)
                    .collect();
                assert_eq!(
                    all_rows,
                    vec![0, 1, 2],
                    "Should return all rows (including nulls) since ZoneMap is inexact"
                );

                // For AtMost results, nulls are included in the superset
                // Downstream processing will handle null filtering
            }
            _ => panic!("Expected AtMost search result from zonemap"),
        }

        // Test 2: Range query - should also return all rows as AtMost
        let query = SargableQuery::Range(
            std::ops::Bound::Included(ScalarValue::Int64(Some(0))),
            std::ops::Bound::Included(ScalarValue::Int64(Some(3))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        match result {
            SearchResult::AtMost(row_ids) => {
                // Again, ZoneMap returns superset including nulls
                let all_rows: Vec<u64> = row_ids
                    .true_rows()
                    .row_addrs()
                    .unwrap()
                    .map(u64::from)
                    .collect();
                assert_eq!(
                    all_rows,
                    vec![0, 1, 2],
                    "Should return all rows in zone as possible matches"
                );
            }
            _ => panic!("Expected AtMost search result from zonemap"),
        }
    }

    #[tokio::test]
    async fn test_nan_zonemap_index() {
        let tmpdir = TempObjDir::default();
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Create deterministic data with NaN values
        // Pattern: [1.0, 2.0, NaN, 3.0, 4.0, 5.0, NaN, 6.0, 7.0, 8.0, ...]
        let mut values = Vec::new();
        for i in 0..500 {
            if i % 5 == 2 {
                values.push(f32::NAN);
            } else {
                // Other values are sequential numbers
                values.push(i as f32);
            }
        }

        let float_data = arrow_array::Float32Array::from(values);
        let row_ids = UInt64Array::from_iter_values((0..float_data.len()).map(|i| i as u64));
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Float32, true),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(float_data.clone()), Arc::new(row_ids)],
        )
        .unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        ZoneMapIndexPlugin::train_zonemap_index(
            data_stream,
            test_store.as_ref(),
            Some(ZoneMapIndexBuilderParams::new(100)),
        )
        .await
        .unwrap();

        // Load the index
        let index = ZoneMapIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load ZoneMapIndex");

        // Should have 5 zones since we have 500 rows and zone size is 100
        assert_eq!(index.zones.len(), 5);

        // Check that each zone has the expected NaN count
        // Each zone has 100 values, and every 5th value (indices 2, 7, 12, ...) is NaN
        // So each zone should have 20 NaN values (100/5 = 20)
        for (i, zone) in index.zones.iter().enumerate() {
            assert_eq!(zone.nan_count, 20, "Zone {} should have 20 NaN values", i);
            assert_eq!(
                zone.bound.length, 100,
                "Zone {} should have zone_length 100",
                i
            );
            assert_eq!(
                zone.bound.fragment_id, 0u64,
                "Zone {} should have fragment_id 0",
                i
            );
        }

        let zone = &index.zones[0];
        assert!(matches!(
            zone.max,
            ScalarValue::Float32(Some(value)) if value.is_nan()
        ));
        let finite_target = ScalarValue::Float32(Some(1000.0));
        assert!(
            finite_target >= zone.min && finite_target <= zone.max,
            "ScalarValue total ordering keeps finite values below NaN max"
        );

        // Test search for NaN values using Equals with NaN
        let query = SargableQuery::Equals(ScalarValue::Float32(Some(f32::NAN)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match all zones since they all contain NaN values
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(0..500); // All rows since NaN is in every zone
        assert_eq!(result, SearchResult::at_most(expected));

        // Test search for a specific finite value that exists in the data
        let query = SargableQuery::Equals(ScalarValue::Float32(Some(5.0)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match only the first zone since 5.0 only exists in rows 0-99
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(0..100);
        assert_eq!(result, SearchResult::at_most(expected));

        // Test search for a value that doesn't exist
        let query = SargableQuery::Equals(ScalarValue::Float32(Some(1000.0)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Since zones contain NaN values, their max will be NaN, so they will be included
        // as potential matches for any finite target (false positive, but acceptable for zone maps)
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(0..500);
        assert_eq!(result, SearchResult::at_most(expected));

        // Test range query that should include finite values
        let query = SargableQuery::Range(
            Bound::Included(ScalarValue::Float32(Some(0.0))),
            Bound::Included(ScalarValue::Float32(Some(250.0))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match the first three zones since they contain values in the range [0, 250]
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(0..300);
        assert_eq!(result, SearchResult::at_most(expected));

        // Test IsIn query with NaN and finite values
        let query = SargableQuery::IsIn(vec![
            ScalarValue::Float32(Some(f32::NAN)),
            ScalarValue::Float32(Some(5.0)),
            ScalarValue::Float32(Some(150.0)), // This value exists in the second zone
        ]);
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match all zones since they all contain NaN values
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(0..500);
        assert_eq!(result, SearchResult::at_most(expected));

        // Test range query that excludes all values
        let query = SargableQuery::Range(
            Bound::Included(ScalarValue::Float32(Some(1000.0))),
            Bound::Included(ScalarValue::Float32(Some(2000.0))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Since zones contain NaN values, their max will be NaN, so they will be included
        // as potential matches for any range query (false positive, but acceptable for zone maps)
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(0..500);
        assert_eq!(result, SearchResult::at_most(expected));

        // Test IsNull query (should match nothing since there are no null values)
        let query = SargableQuery::IsNull();
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::exact(RowAddrTreeMap::new()));

        // Test range queries with NaN bounds
        // Range with NaN as start bound (included)
        let query = SargableQuery::Range(
            Bound::Included(ScalarValue::Float32(Some(f32::NAN))),
            Bound::Unbounded,
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        // Should match all zones since they all contain NaN values
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(0..500);
        assert_eq!(result, SearchResult::at_most(expected));

        // Range with NaN as end bound (included)
        let query = SargableQuery::Range(
            Bound::Unbounded,
            Bound::Included(ScalarValue::Float32(Some(f32::NAN))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        // Should match all zones since they all contain NaN values
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(0..500);
        assert_eq!(result, SearchResult::at_most(expected));

        // Range with NaN as end bound (excluded)
        let query = SargableQuery::Range(
            Bound::Unbounded,
            Bound::Excluded(ScalarValue::Float32(Some(f32::NAN))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        // Should match all zones since everything is less than NaN
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(0..500);
        assert_eq!(result, SearchResult::at_most(expected));

        // Range with NaN as start bound (excluded)
        let query = SargableQuery::Range(
            Bound::Excluded(ScalarValue::Float32(Some(f32::NAN))),
            Bound::Unbounded,
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        // Should match nothing since nothing is greater than NaN
        assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));

        // Test IsIn query with mixed float types (Float16, Float32, Float64)
        let query = SargableQuery::IsIn(vec![
            ScalarValue::Float16(Some(half::f16::NAN)),
            ScalarValue::Float32(Some(f32::NAN)),
            ScalarValue::Float64(Some(f64::NAN)),
            ScalarValue::Float32(Some(5.0)),
        ]);
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        // Should match all zones since they all contain NaN values
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(0..500);
        assert_eq!(result, SearchResult::at_most(expected));
    }

    #[tokio::test]
    // Test data that belongs to the same fragment but coming from different batches
    async fn test_basic_zonemap_index() {
        let tmpdir = TempObjDir::default();
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let data = arrow_array::Int32Array::from_iter_values(0..=100);
        let row_ids = UInt64Array::from_iter_values((0..data.len()).map(|i| i as u64));
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Int32, false),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let data =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        ZoneMapIndexPlugin::train_zonemap_index(
            data_stream,
            test_store.as_ref(),
            Some(ZoneMapIndexBuilderParams::new(100)),
        )
        .await
        .unwrap();

        log::debug!("Successfully wrote the index file");

        // Read the raw index file back and check its contents
        let index_file = test_store.open_index_file(ZONEMAP_FILENAME).await.unwrap();
        // Print the metadata from the index_file
        let metadata = index_file.schema().metadata.clone();
        let record_batch = index_file
            .read_record_batch(0, index_file.num_rows() as u64)
            .await
            .unwrap();
        assert_eq!(record_batch.num_rows(), 2);
        assert_eq!(
            record_batch
                .column(0)
                .as_any()
                .downcast_ref::<arrow_array::Int32Array>()
                .unwrap()
                .values(),
            &[0, 100]
        );
        assert_eq!(
            record_batch
                .column(1)
                .as_any()
                .downcast_ref::<arrow_array::Int32Array>()
                .unwrap()
                .values(),
            &[99, 100]
        );
        assert_eq!(
            record_batch
                .column(2)
                .as_any()
                .downcast_ref::<arrow_array::UInt32Array>()
                .unwrap()
                .values(),
            &[0, 0]
        );
        assert_eq!(metadata.get(ZONEMAP_SIZE_META_KEY).unwrap(), "100");

        // Read the index file back and check its contents
        let index = ZoneMapIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load ZoneMapIndex");
        assert_eq!(index.zones.len(), 2);
        assert_eq!(
            index.zones,
            vec![
                ZoneMapStatistics {
                    min: ScalarValue::Int32(Some(0)),
                    max: ScalarValue::Int32(Some(99)),
                    null_count: 0,
                    nan_count: 0,
                    bound: ZoneBound {
                        fragment_id: 0,
                        start: 0,
                        length: 100,
                    },
                },
                ZoneMapStatistics {
                    min: ScalarValue::Int32(Some(100)),
                    max: ScalarValue::Int32(Some(100)),
                    null_count: 0,
                    nan_count: 0,
                    bound: ZoneBound {
                        fragment_id: 0,
                        start: 100,
                        length: 1,
                    },
                }
            ]
        );
        // Verify nan_count is 0 for all zones (no NaN values in integer data)
        for (i, zone) in index.zones.iter().enumerate() {
            assert_eq!(zone.nan_count, 0, "Zone {} should have nan_count = 0", i);
        }

        assert_eq!(index.data_type, DataType::Int32);
        assert_eq!(index.rows_per_zone, 100);
        assert_eq!(
            index.calculate_included_frags().await.unwrap(),
            RoaringBitmap::from_iter(0..1)
        );

        // Test search functionality

        // 1. Range query: (50, +inf)
        let query = SargableQuery::Range(
            Bound::Excluded(ScalarValue::Int32(Some(50))),
            Bound::Unbounded,
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::at_most(0..=100));

        // 2. Range query: [0, 50]
        let query = SargableQuery::Range(
            Bound::Included(ScalarValue::Int32(Some(0))),
            Bound::Included(ScalarValue::Int32(Some(50))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::at_most(0..=99));

        // 3. Range query: [101, 200] (should only match the second zone, which is row 100)
        let query = SargableQuery::Range(
            Bound::Included(ScalarValue::Int32(Some(101))),
            Bound::Included(ScalarValue::Int32(Some(200))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        // Only row 100 is in the second zone, but its value is 100, so this should be empty
        assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));

        // 4. Range query: [100, 100] (should match only the last row)
        let query = SargableQuery::Range(
            Bound::Included(ScalarValue::Int32(Some(100))),
            Bound::Included(ScalarValue::Int32(Some(100))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::at_most(100..=100));

        // 5. Equals query: 0 (should match first row)
        let query = SargableQuery::Equals(ScalarValue::Int32(Some(0)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::at_most(0..=99));

        // 6. Equals query: 100 (should match only last row)
        let query = SargableQuery::Equals(ScalarValue::Int32(Some(100)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::at_most(100..=100));

        // 7. Equals query: 101 (should match nothing)
        let query = SargableQuery::Equals(ScalarValue::Int32(Some(101)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));

        // 8. IsNull query (no nulls in data, should match nothing)
        let query = SargableQuery::IsNull();
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::exact(RowAddrTreeMap::new()));
        // 9. IsIn query: [0, 100, 101, 50]
        let query = SargableQuery::IsIn(vec![
            ScalarValue::Int32(Some(0)),
            ScalarValue::Int32(Some(100)),
            ScalarValue::Int32(Some(101)),
            ScalarValue::Int32(Some(50)),
        ]);
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        // 0 and 50 are in the first zone, 100 in the second, 101 is not present
        assert_eq!(result, SearchResult::at_most(0..=100));

        // 10. IsIn query: [101, 102] (should match nothing)
        let query = SargableQuery::IsIn(vec![
            ScalarValue::Int32(Some(101)),
            ScalarValue::Int32(Some(102)),
        ]);
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));

        // 11. IsIn query: [null] (should match nothing, as there are no nulls)
        let query = SargableQuery::IsIn(vec![ScalarValue::Int32(None)]);
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));

        // 12. Equals query: null (should match nothing, as there are no nulls)
        let query = SargableQuery::Equals(ScalarValue::Int32(None));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));
    }

    #[tokio::test]
    // Test zonemap with same fragment from multiple batches
    async fn test_complex_zonemap_index() {
        let tmpdir = TempObjDir::default();
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Create data that will produce the expected zonemap zones
        let data =
            arrow_array::Int64Array::from_iter_values(0..(ROWS_PER_ZONE_DEFAULT * 2 + 42) as i64);
        let row_ids = UInt64Array::from_iter_values((0..data.len()).map(|i| i as u64));
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Int64, false),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let data =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        ZoneMapIndexPlugin::train_zonemap_index(
            data_stream,
            test_store.as_ref(),
            Some(ZoneMapIndexBuilderParams::default()),
        )
        .await
        .unwrap();

        log::debug!("Successfully wrote the index file");

        // Read the index file back and check its contents
        let index = ZoneMapIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load ZoneMapIndex");
        assert_eq!(index.zones.len(), 3);
        assert_eq!(
            index.zones,
            vec![
                ZoneMapStatistics {
                    min: ScalarValue::Int64(Some(0)),
                    max: ScalarValue::Int64(Some(8191)),
                    null_count: 0,
                    nan_count: 0,
                    bound: ZoneBound {
                        fragment_id: 0,
                        start: 0,
                        length: 8192,
                    },
                },
                ZoneMapStatistics {
                    min: ScalarValue::Int64(Some(8192)),
                    max: ScalarValue::Int64(Some(16383)),
                    null_count: 0,
                    nan_count: 0,
                    bound: ZoneBound {
                        fragment_id: 0,
                        start: 8192,
                        length: 8192,
                    },
                },
                ZoneMapStatistics {
                    min: ScalarValue::Int64(Some(16384)),
                    max: ScalarValue::Int64(Some(16425)),
                    null_count: 0,
                    nan_count: 0,
                    bound: ZoneBound {
                        fragment_id: 0,
                        start: 16384,
                        length: 42,
                    },
                }
            ]
        );
        // Verify nan_count is 0 for all zones (no NaN values in integer data)
        for (i, zone) in index.zones.iter().enumerate() {
            assert_eq!(zone.nan_count, 0, "Zone {} should have nan_count = 0", i);
        }

        assert_eq!(index.data_type, DataType::Int64);
        assert_eq!(index.rows_per_zone, ROWS_PER_ZONE_DEFAULT);
        assert_eq!(
            index.calculate_included_frags().await.unwrap(),
            RoaringBitmap::from_iter(0..1)
        );

        // TODO: Test search functionality
        // Test search functionality

        // Search for a value in the first zone
        let query = SargableQuery::Equals(ScalarValue::Int64(Some(1000)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        // Should match row 1000 in fragment 0: row address = (0 << 32) + 1000 = 1000
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(0..=8191);
        assert_eq!(result, SearchResult::at_most(expected));

        // Search for a value in the second zone
        let query = SargableQuery::Equals(ScalarValue::Int64(Some(9000)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        // Should match row 9000 in fragment 0: row address = (0 << 32) + 9000 = 9000
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(8192..=16383);
        assert_eq!(result, SearchResult::at_most(expected));

        // Search for a value not present in any zone
        let query = SargableQuery::Equals(ScalarValue::Int64(Some(20000)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));

        // Search for a range that spans multiple zones
        let query = SargableQuery::Range(
            Bound::Included(ScalarValue::Int64(Some(9000))),
            Bound::Included(ScalarValue::Int64(Some(16400))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        // Should match all rows from 8000 to 16400 (inclusive)
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(8192..=16425);
        assert_eq!(result, SearchResult::at_most(expected));
    }

    #[tokio::test]
    // Test zonemap with multiple fragments from different batches
    async fn test_multiple_fragments_zonemap() {
        let tmpdir = TempObjDir::default();
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Int64, false),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));

        // Create multiple fragments with data that will produce expected zones
        // Fragment 0: values 0-8191 (first zone)
        let fragment0_data =
            arrow_array::Int64Array::from_iter_values(0..ROWS_PER_ZONE_DEFAULT as i64);
        let fragment0_row_ids = UInt64Array::from_iter_values(0..ROWS_PER_ZONE_DEFAULT);
        let fragment0_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(fragment0_data), Arc::new(fragment0_row_ids)],
        )
        .unwrap();

        // Fragment 1: values 8192-16383 (second zone)
        let fragment1_data = arrow_array::Int64Array::from_iter_values(
            (ROWS_PER_ZONE_DEFAULT as i64)..((ROWS_PER_ZONE_DEFAULT * 2) as i64),
        );
        let fragment1_row_ids =
            UInt64Array::from_iter_values((0..ROWS_PER_ZONE_DEFAULT).map(|i| i + (1 << 32)));
        let fragment1_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(fragment1_data), Arc::new(fragment1_row_ids)],
        )
        .unwrap();

        // Fragment 2: values 16384-16426 (third zone)
        let fragment2_data = arrow_array::Int64Array::from_iter_values(
            ((ROWS_PER_ZONE_DEFAULT * 2) as i64)..((ROWS_PER_ZONE_DEFAULT * 2 + 42) as i64),
        );
        let fragment2_row_ids =
            UInt64Array::from_iter_values((0..42).map(|i| (i as u64) + (2 << 32)));
        let fragment2_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(fragment2_data), Arc::new(fragment2_row_ids)],
        )
        .unwrap();

        // Each fragment is broken into few batches
        {
            // Create a stream with multiple batches (fragments)
            let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
                schema.clone(),
                stream::iter(vec![
                    Ok(fragment0_batch.clone()),
                    Ok(fragment1_batch.clone()),
                    Ok(fragment2_batch.clone()),
                ]),
            ));
            ZoneMapIndexPlugin::train_zonemap_index(
                data_stream,
                test_store.as_ref(),
                Some(ZoneMapIndexBuilderParams::new(5000)),
            )
            .await
            .unwrap();

            // Read the index file back and check its contents
            let index = ZoneMapIndex::load(test_store.clone(), None, &LanceCache::no_cache())
                .await
                .expect("Failed to load ZoneMapIndex");
            assert_eq!(index.zones.len(), 5);
            assert_eq!(
                index.zones,
                vec![
                    ZoneMapStatistics {
                        min: ScalarValue::Int64(Some(0)),
                        max: ScalarValue::Int64(Some(4999)),
                        null_count: 0,
                        nan_count: 0,
                        bound: ZoneBound {
                            fragment_id: 0,
                            start: 0,
                            length: 5000,
                        },
                    },
                    ZoneMapStatistics {
                        min: ScalarValue::Int64(Some(5000)),
                        max: ScalarValue::Int64(Some(8191)),
                        null_count: 0,
                        nan_count: 0,
                        bound: ZoneBound {
                            fragment_id: 0,
                            start: 5000,
                            length: 3192,
                        },
                    },
                    ZoneMapStatistics {
                        min: ScalarValue::Int64(Some(8192)),
                        max: ScalarValue::Int64(Some(13191)),
                        null_count: 0,
                        nan_count: 0,
                        bound: ZoneBound {
                            fragment_id: 1,
                            start: 0,
                            length: 5000,
                        },
                    },
                    ZoneMapStatistics {
                        min: ScalarValue::Int64(Some(13192)),
                        max: ScalarValue::Int64(Some(16383)),
                        null_count: 0,
                        nan_count: 0,
                        bound: ZoneBound {
                            fragment_id: 1,
                            start: 5000,
                            length: 3192,
                        },
                    },
                    ZoneMapStatistics {
                        min: ScalarValue::Int64(Some(16384)),
                        max: ScalarValue::Int64(Some(16425)),
                        null_count: 0,
                        nan_count: 0,
                        bound: ZoneBound {
                            fragment_id: 2,
                            start: 0,
                            length: 42,
                        },
                    }
                ]
            );
            // Verify nan_count is 0 for all zones (no NaN values in integer data)
            for (i, zone) in index.zones.iter().enumerate() {
                assert_eq!(zone.nan_count, 0, "Zone {} should have nan_count = 0", i);
            }

            assert_eq!(index.data_type, DataType::Int64);
            assert_eq!(index.rows_per_zone, 5000);
            assert_eq!(
                index.calculate_included_frags().await.unwrap(),
                RoaringBitmap::from_iter(0..3)
            );

            // Verify _rowaddr column values are properly assigned
            let verify_data_stream: SendableRecordBatchStream =
                Box::pin(RecordBatchStreamAdapter::new(
                    schema.clone(),
                    stream::iter(vec![
                        Ok(fragment0_batch.clone()),
                        Ok(fragment1_batch.clone()),
                        Ok(fragment2_batch.clone()),
                    ]),
                ));
            let batches: Vec<RecordBatch> = verify_data_stream.try_collect().await.unwrap();

            assert_eq!(batches.len(), 3);

            // Check fragment 0 _rowaddr values (should start from 0)
            let fragment0_rowaddr_col = batches[0].column_by_name(ROW_ADDR).unwrap();
            let fragment0_rowaddrs = fragment0_rowaddr_col
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            assert_eq!(
                fragment0_rowaddrs.values().len(),
                ROWS_PER_ZONE_DEFAULT as usize
            );
            assert_eq!(fragment0_rowaddrs.values()[0], 0);
            assert_eq!(
                fragment0_rowaddrs.values()[fragment0_rowaddrs.values().len() - 1],
                8191
            );

            // Check fragment 1 _rowaddr values (should start from fragment_id=1)
            let fragment1_rowaddr_col = batches[1].column_by_name(ROW_ADDR).unwrap();
            let fragment1_rowaddrs = fragment1_rowaddr_col
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            assert_eq!(
                fragment1_rowaddrs.values().len(),
                ROWS_PER_ZONE_DEFAULT as usize
            );
            assert_eq!(fragment1_rowaddrs.values()[0], 1u64 << 32); // fragment_id=1, local_offset=0
            assert_eq!(
                fragment1_rowaddrs.values()[fragment1_rowaddrs.values().len() - 1],
                8191 | (1u64 << 32)
            );

            // Check fragment 2 _rowaddr values (should start from fragment_id=2)
            let fragment2_rowaddr_col = batches[2].column_by_name(ROW_ADDR).unwrap();
            let fragment2_rowaddrs = fragment2_rowaddr_col
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            assert_eq!(fragment2_rowaddrs.values().len(), 42);
            assert_eq!(fragment2_rowaddrs.values()[0], 2u64 << 32); // fragment_id=2, local_offset=0
            assert_eq!(
                fragment2_rowaddrs.values()[fragment2_rowaddrs.values().len() - 1],
                (2u64 << 32) | 41
            );

            // Add a few tests for search functionality

            // Test range query that spans multiple fragments
            let query = SargableQuery::Range(
                Bound::Included(ScalarValue::Int64(Some(5000))),
                Bound::Included(ScalarValue::Int64(Some(12000))),
            );
            let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
            // Should include zones from fragments 0 and 1 since they overlap with range 5000-12000
            let mut expected = RowAddrTreeMap::new();
            // zone 1
            expected.insert_range(5000..8192);
            // zone 2
            expected.insert_range((1u64 << 32)..((1u64 << 32) + 5000));
            assert_eq!(result, SearchResult::at_most(expected));

            // Test exact match query from zone 2
            let query = SargableQuery::Equals(ScalarValue::Int64(Some(8192)));
            let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
            // Should include zone 2 since it contains value 8192
            let mut expected = RowAddrTreeMap::new();
            expected.insert_range((1u64 << 32)..((1u64 << 32) + 5000));
            assert_eq!(result, SearchResult::at_most(expected));

            // Test exact match query from zone 4
            let query = SargableQuery::Equals(ScalarValue::Int64(Some(16385)));
            let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
            // Should include zone 4 since it contains value 16385
            let mut expected = RowAddrTreeMap::new();
            expected.insert_range(2u64 << 32..((2u64 << 32) + 42));
            assert_eq!(result, SearchResult::at_most(expected));

            // Test query that matches nothing
            let query = SargableQuery::Equals(ScalarValue::Int64(Some(99999)));
            let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
            assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));

            // Test is_in query
            let query = SargableQuery::IsIn(vec![ScalarValue::Int64(Some(16385))]);
            let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
            let mut expected = RowAddrTreeMap::new();
            expected.insert_range(2u64 << 32..((2u64 << 32) + 42));
            assert_eq!(result, SearchResult::at_most(expected));

            // Test equals query with null
            let query = SargableQuery::Equals(ScalarValue::Int64(None));
            let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
            let mut expected = RowAddrTreeMap::new();
            expected.insert_range(0..=16425);
            // expected = {:?}", expected
            assert_eq!(result, SearchResult::at_most(RowAddrTreeMap::new()));
        }

        //  Each fragment is its own batch
        {
            // Create a stream with multiple batches (fragments)
            let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
                schema.clone(),
                stream::iter(vec![
                    Ok(fragment0_batch.clone()),
                    Ok(fragment1_batch.clone()),
                    Ok(fragment2_batch.clone()),
                ]),
            ));
            ZoneMapIndexPlugin::train_zonemap_index(
                data_stream,
                test_store.as_ref(),
                Some(ZoneMapIndexBuilderParams::default()),
            )
            .await
            .unwrap();

            // Read the index file back and check its contents
            let index = ZoneMapIndex::load(test_store.clone(), None, &LanceCache::no_cache())
                .await
                .expect("Failed to load ZoneMapIndex");
            assert_eq!(index.zones.len(), 3);
            assert_eq!(
                index.zones,
                vec![
                    ZoneMapStatistics {
                        min: ScalarValue::Int64(Some(0)),
                        max: ScalarValue::Int64(Some(8191)),
                        null_count: 0,
                        nan_count: 0,
                        bound: ZoneBound {
                            fragment_id: 0,
                            start: 0,
                            length: 8192,
                        },
                    },
                    ZoneMapStatistics {
                        min: ScalarValue::Int64(Some(8192)),
                        max: ScalarValue::Int64(Some(16383)),
                        null_count: 0,
                        nan_count: 0,
                        bound: ZoneBound {
                            fragment_id: 1,
                            start: 0,
                            length: 8192,
                        },
                    },
                    ZoneMapStatistics {
                        min: ScalarValue::Int64(Some(16384)),
                        max: ScalarValue::Int64(Some(16425)),
                        null_count: 0,
                        nan_count: 0,
                        bound: ZoneBound {
                            fragment_id: 2,
                            start: 0,
                            length: 42,
                        },
                    }
                ]
            );
            // Verify nan_count is 0 for all zones (no NaN values in integer data)
            for (i, zone) in index.zones.iter().enumerate() {
                assert_eq!(zone.nan_count, 0, "Zone {} should have nan_count = 0", i);
            }

            assert_eq!(index.data_type, DataType::Int64);
            assert_eq!(index.rows_per_zone, ROWS_PER_ZONE_DEFAULT);
            assert_eq!(
                index.calculate_included_frags().await.unwrap(),
                RoaringBitmap::from_iter(0..3)
            );
        }

        //  All fragments are in the same batch
        {
            // Create a stream with multiple batches (fragments)
            let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
                schema.clone(),
                stream::iter(vec![
                    Ok(fragment0_batch.clone()),
                    Ok(fragment1_batch.clone()),
                    Ok(fragment2_batch.clone()),
                ]),
            ));
            ZoneMapIndexPlugin::train_zonemap_index(
                data_stream,
                test_store.as_ref(),
                Some(ZoneMapIndexBuilderParams::new(ROWS_PER_ZONE_DEFAULT * 3)),
            )
            .await
            .unwrap();

            // Read the index file back and check its contents
            let index = ZoneMapIndex::load(test_store.clone(), None, &LanceCache::no_cache())
                .await
                .expect("Failed to load ZoneMapIndex");
            assert_eq!(index.zones.len(), 3);
            assert_eq!(
                index.zones,
                vec![
                    ZoneMapStatistics {
                        min: ScalarValue::Int64(Some(0)),
                        max: ScalarValue::Int64(Some(8191)),
                        null_count: 0,
                        nan_count: 0,
                        bound: ZoneBound {
                            fragment_id: 0,
                            start: 0,
                            length: 8192,
                        },
                    },
                    ZoneMapStatistics {
                        min: ScalarValue::Int64(Some(8192)),
                        max: ScalarValue::Int64(Some(16383)),
                        null_count: 0,
                        nan_count: 0,
                        bound: ZoneBound {
                            fragment_id: 1,
                            start: 0,
                            length: 8192,
                        },
                    },
                    ZoneMapStatistics {
                        min: ScalarValue::Int64(Some(16384)),
                        max: ScalarValue::Int64(Some(16425)),
                        null_count: 0,
                        nan_count: 0,
                        bound: ZoneBound {
                            fragment_id: 2,
                            start: 0,
                            length: 42,
                        },
                    }
                ]
            );
            // Verify nan_count is 0 for all zones (no NaN values in integer data)
            for (i, zone) in index.zones.iter().enumerate() {
                assert_eq!(zone.nan_count, 0, "Zone {} should have nan_count = 0", i);
            }

            assert_eq!(index.data_type, DataType::Int64);
            assert_eq!(index.rows_per_zone, ROWS_PER_ZONE_DEFAULT * 3);
        }
    }

    #[tokio::test]
    async fn test_fragment_id_assignment() {
        // Test that fragment IDs are properly assigned in _rowaddr values
        let schema = Arc::new(Schema::new(vec![Field::new(
            VALUE_COLUMN_NAME,
            DataType::Int32,
            false,
        )]));

        // Create multiple fragments
        let fragment0_data = arrow_array::Int32Array::from_iter_values(0..5);
        let fragment0_batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(fragment0_data)]).unwrap();

        let fragment1_data = arrow_array::Int32Array::from_iter_values(5..10);
        let fragment1_batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(fragment1_data)]).unwrap();

        let aligned_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(vec![Ok(fragment0_batch), Ok(fragment1_batch)]),
        ));

        let aligned_stream = add_row_addr(aligned_stream);

        let batches: Vec<RecordBatch> = aligned_stream.try_collect().await.unwrap();

        assert_eq!(batches.len(), 2);

        // Check fragment 0 _rowaddr values
        let fragment0_rowaddr_col = batches[0].column_by_name(ROW_ADDR).unwrap();
        let fragment0_rowaddrs = fragment0_rowaddr_col
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        // Fragment 0 should have _rowaddr values: 0, 1, 2, 3, 4
        assert_eq!(fragment0_rowaddrs.values(), &[0, 1, 2, 3, 4]);

        // Check fragment 1 _rowaddr values
        let fragment1_rowaddr_col = batches[1].column_by_name(ROW_ADDR).unwrap();
        let fragment1_rowaddrs = fragment1_rowaddr_col
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        // Fragment 1 should have _rowaddr values: (1 << 32) | 0, (1 << 32) | 1, etc.
        // which is: 4294967296, 4294967297, 4294967298, 4294967299, 4294967300
        assert_eq!(
            fragment1_rowaddrs.values(),
            &[4294967296, 4294967297, 4294967298, 4294967299, 4294967300]
        );
    }

    #[tokio::test]
    async fn test_like_prefix_query() {
        let tmpdir = TempObjDir::default();
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Create zones with different string ranges
        // Zone 0: ["aaa", "azz"] - should NOT match "foo%"
        // Zone 1: ["bar", "baz"] - should NOT match "foo%"
        // Zone 2: ["fa", "foz"]  - should match "foo%" (contains potential matches)
        // Zone 3: ["fop", "fzz"] - should NOT match "foo%" (all values >= "fop")
        // Zone 4: ["foo", "foobar"] - should match "foo%"
        // Zone 5: ["gaa", "gzz"] - should NOT match "foo%"

        let zones = vec![
            ZoneMapStatistics {
                min: ScalarValue::Utf8(Some("aaa".to_string())),
                max: ScalarValue::Utf8(Some("azz".to_string())),
                null_count: 0,
                nan_count: 0,
                bound: ZoneBound {
                    fragment_id: 0,
                    start: 0,
                    length: 100,
                },
            },
            ZoneMapStatistics {
                min: ScalarValue::Utf8(Some("bar".to_string())),
                max: ScalarValue::Utf8(Some("baz".to_string())),
                null_count: 0,
                nan_count: 0,
                bound: ZoneBound {
                    fragment_id: 1,
                    start: 0,
                    length: 100,
                },
            },
            ZoneMapStatistics {
                min: ScalarValue::Utf8(Some("fa".to_string())),
                max: ScalarValue::Utf8(Some("foz".to_string())),
                null_count: 0,
                nan_count: 0,
                bound: ZoneBound {
                    fragment_id: 2,
                    start: 0,
                    length: 100,
                },
            },
            ZoneMapStatistics {
                min: ScalarValue::Utf8(Some("fop".to_string())),
                max: ScalarValue::Utf8(Some("fzz".to_string())),
                null_count: 0,
                nan_count: 0,
                bound: ZoneBound {
                    fragment_id: 3,
                    start: 0,
                    length: 100,
                },
            },
            ZoneMapStatistics {
                min: ScalarValue::Utf8(Some("foo".to_string())),
                max: ScalarValue::Utf8(Some("foobar".to_string())),
                null_count: 0,
                nan_count: 0,
                bound: ZoneBound {
                    fragment_id: 4,
                    start: 0,
                    length: 100,
                },
            },
            ZoneMapStatistics {
                min: ScalarValue::Utf8(Some("gaa".to_string())),
                max: ScalarValue::Utf8(Some("gzz".to_string())),
                null_count: 0,
                nan_count: 0,
                bound: ZoneBound {
                    fragment_id: 5,
                    start: 0,
                    length: 100,
                },
            },
        ];

        let index = ZoneMapIndex {
            zones,
            data_type: DataType::Utf8,
            rows_per_zone: ROWS_PER_ZONE_DEFAULT,
            store: test_store,
            fri: None,
            index_cache: WeakLanceCache::from(&LanceCache::no_cache()),
            null_rows: None,
        };

        // Test LikePrefix query for "foo"
        let query = SargableQuery::LikePrefix(ScalarValue::Utf8(Some("foo".to_string())));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match zones 2 and 4 only
        let mut expected = RowAddrTreeMap::new();
        // Zone 2: fragment 2
        expected.insert_range((2u64 << 32)..((2u64 << 32) + 100));
        // Zone 4: fragment 4
        expected.insert_range((4u64 << 32)..((4u64 << 32) + 100));

        assert_eq!(result, SearchResult::at_most(expected));
    }

    #[tokio::test]
    async fn test_like_prefix_edge_cases() {
        let tmpdir = TempObjDir::default();
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Test edge cases for LIKE prefix
        let zones = vec![
            // Zone with values that contain the prefix exactly
            ZoneMapStatistics {
                min: ScalarValue::Utf8(Some("test".to_string())),
                max: ScalarValue::Utf8(Some("test".to_string())),
                null_count: 0,
                nan_count: 0,
                bound: ZoneBound {
                    fragment_id: 0,
                    start: 0,
                    length: 100,
                },
            },
            // Zone with values that span across the prefix boundary
            ZoneMapStatistics {
                min: ScalarValue::Utf8(Some("te".to_string())),
                max: ScalarValue::Utf8(Some("tf".to_string())),
                null_count: 0,
                nan_count: 0,
                bound: ZoneBound {
                    fragment_id: 1,
                    start: 0,
                    length: 100,
                },
            },
            // Zone completely before prefix
            ZoneMapStatistics {
                min: ScalarValue::Utf8(Some("abc".to_string())),
                max: ScalarValue::Utf8(Some("def".to_string())),
                null_count: 0,
                nan_count: 0,
                bound: ZoneBound {
                    fragment_id: 2,
                    start: 0,
                    length: 100,
                },
            },
        ];

        let index = ZoneMapIndex {
            zones,
            data_type: DataType::Utf8,
            rows_per_zone: ROWS_PER_ZONE_DEFAULT,
            store: test_store,
            fri: None,
            index_cache: WeakLanceCache::from(&LanceCache::no_cache()),
            null_rows: None,
        };

        // Test LikePrefix "test"
        let query = SargableQuery::LikePrefix(ScalarValue::Utf8(Some("test".to_string())));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match zones 0 and 1
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(0..100); // Zone 0: fragment 0
        expected.insert_range((1u64 << 32)..((1u64 << 32) + 100));

        assert_eq!(result, SearchResult::at_most(expected));

        // Test empty prefix - should match all zones
        let query = SargableQuery::LikePrefix(ScalarValue::Utf8(Some("".to_string())));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(0..100); // Zone 0: fragment 0
        expected.insert_range((1u64 << 32)..((1u64 << 32) + 100));
        expected.insert_range((2u64 << 32)..((2u64 << 32) + 100));

        assert_eq!(result, SearchResult::at_most(expected));
    }

    #[tokio::test]
    async fn test_like_prefix_large_utf8() {
        let tmpdir = TempObjDir::default();
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Test with LargeUtf8 type
        let zones = vec![
            ZoneMapStatistics {
                min: ScalarValue::LargeUtf8(Some("aaa".to_string())),
                max: ScalarValue::LargeUtf8(Some("azz".to_string())),
                null_count: 0,
                nan_count: 0,
                bound: ZoneBound {
                    fragment_id: 0,
                    start: 0,
                    length: 100,
                },
            },
            ZoneMapStatistics {
                min: ScalarValue::LargeUtf8(Some("foo".to_string())),
                max: ScalarValue::LargeUtf8(Some("foobar".to_string())),
                null_count: 0,
                nan_count: 0,
                bound: ZoneBound {
                    fragment_id: 1,
                    start: 0,
                    length: 100,
                },
            },
        ];

        let index = ZoneMapIndex {
            zones,
            data_type: DataType::LargeUtf8,
            rows_per_zone: ROWS_PER_ZONE_DEFAULT,
            store: test_store,
            fri: None,
            index_cache: WeakLanceCache::from(&LanceCache::no_cache()),
            null_rows: None,
        };

        // Test LikePrefix with LargeUtf8
        let query = SargableQuery::LikePrefix(ScalarValue::LargeUtf8(Some("foo".to_string())));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match only zone 1
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range((1u64 << 32)..((1u64 << 32) + 100));

        assert_eq!(result, SearchResult::at_most(expected));
    }

    #[test]
    fn test_compute_next_prefix() {
        use super::compute_next_prefix;

        // Basic cases
        assert_eq!(compute_next_prefix("foo"), Some("fop".to_string()));
        assert_eq!(compute_next_prefix("abc"), Some("abd".to_string()));
        assert_eq!(compute_next_prefix("a"), Some("b".to_string()));
        assert_eq!(compute_next_prefix("z"), Some("{".to_string())); // 'z' + 1 = '{'

        // Edge case: prefix with 'z' at the end
        assert_eq!(compute_next_prefix("abz"), Some("ab{".to_string()));

        // Edge case with tilde (~) which is 0x7E
        assert_eq!(compute_next_prefix("ab~"), Some("ab\x7f".to_string()));

        // Empty prefix
        assert_eq!(compute_next_prefix(""), None);

        // Non-ASCII: works correctly by incrementing Unicode code points
        // é (U+00E9) -> ê (U+00EA)
        assert_eq!(compute_next_prefix("café"), Some("cafê".to_string()));
        // 中 (U+4E2D) -> 丮 (U+4E2E)
        assert_eq!(compute_next_prefix("abc中"), Some("abc丮".to_string()));
        // ÿ (U+00FF) -> Ā (U+0100) - crosses byte boundary but works
        assert_eq!(compute_next_prefix("cafÿ"), Some("cafĀ".to_string()));

        // Edge case: character just before surrogate range
        // U+D7FF -> U+E000 (skips surrogate range U+D800-U+DFFF)
        assert_eq!(
            compute_next_prefix("a\u{D7FF}"),
            Some("a\u{E000}".to_string())
        );

        // Edge case: max Unicode character U+10FFFF, falls back to previous char
        assert_eq!(compute_next_prefix("ab\u{10FFFF}"), Some("ac".to_string()));
        // All max characters
        assert_eq!(compute_next_prefix("\u{10FFFF}\u{10FFFF}"), None);
    }

    // When merging zone map segments, if ANY source segment has null_rows = None
    // (legacy — null positions unknown), the merged result must also be None.
    // The bug: any_null_bitmap is set to true as soon as one source has Some(...),
    // and the None sources are silently skipped.  The merged index then has
    // null_rows = Some(partial_bitmap), so an IsNull search returns exact results
    // that only cover the modern segment's nulls — a false negative for the legacy
    // segment whose null positions were never tracked.
    #[tokio::test]
    async fn test_merge_with_legacy_none_segment_not_treated_as_no_nulls() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        use arrow_array::{Int32Array, UInt32Array};

        // Index A: fragment 0, modern — has a complete null bitmap with 2 known null rows.
        let schema_a = Arc::new(Schema::new(vec![
            Field::new("min", DataType::Int32, true),
            Field::new("max", DataType::Int32, true),
            Field::new("null_count", DataType::UInt32, false),
            Field::new("nan_count", DataType::UInt32, false),
            Field::new("fragment_id", DataType::UInt64, false),
            Field::new("zone_start", DataType::UInt64, false),
            Field::new("zone_length", DataType::UInt64, false),
        ]));
        let batch_a = RecordBatch::try_new(
            schema_a,
            vec![
                Arc::new(Int32Array::from(vec![Some(1i32)])) as _,
                Arc::new(Int32Array::from(vec![Some(5i32)])) as _,
                Arc::new(UInt32Array::from(vec![2u32])) as _,
                Arc::new(UInt32Array::from(vec![0u32])) as _,
                Arc::new(UInt64Array::from(vec![0u64])) as _,
                Arc::new(UInt64Array::from(vec![0u64])) as _,
                Arc::new(UInt64Array::from(vec![10u64])) as _,
            ],
        )
        .unwrap();
        let mut modern_null_rows = RowAddrTreeMap::new();
        modern_null_rows.insert(3); // frag 0 row 3
        modern_null_rows.insert(7); // frag 0 row 7
        let cache = LanceCache::no_cache();
        let index_a = Arc::new(
            ZoneMapIndex::try_from_serialized(
                batch_a,
                store.clone(),
                None,
                &cache,
                10,
                Some(modern_null_rows), // modern: complete bitmap
            )
            .unwrap(),
        );

        // Index B: fragment 1, legacy — null_rows = None despite null_count = 3.
        let schema_b = Arc::new(Schema::new(vec![
            Field::new("min", DataType::Int32, true),
            Field::new("max", DataType::Int32, true),
            Field::new("null_count", DataType::UInt32, false),
            Field::new("nan_count", DataType::UInt32, false),
            Field::new("fragment_id", DataType::UInt64, false),
            Field::new("zone_start", DataType::UInt64, false),
            Field::new("zone_length", DataType::UInt64, false),
        ]));
        let batch_b = RecordBatch::try_new(
            schema_b,
            vec![
                Arc::new(Int32Array::from(vec![Some(10i32)])) as _,
                Arc::new(Int32Array::from(vec![Some(20i32)])) as _,
                Arc::new(UInt32Array::from(vec![3u32])) as _,
                Arc::new(UInt32Array::from(vec![0u32])) as _,
                Arc::new(UInt64Array::from(vec![1u64])) as _,
                Arc::new(UInt64Array::from(vec![0u64])) as _,
                Arc::new(UInt64Array::from(vec![10u64])) as _,
            ],
        )
        .unwrap();
        let index_b = Arc::new(
            ZoneMapIndex::try_from_serialized(
                batch_b,
                store.clone(),
                None,
                &cache,
                10,
                None, // legacy: null positions unknown
            )
            .unwrap(),
        );

        let dest_tmpdir = TempObjDir::default();
        let dest_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            dest_tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let all_frags = RoaringBitmap::from_iter([0u32, 1]);
        merge_zonemap_indices(
            &[index_a.as_ref(), index_b.as_ref()],
            dest_store.as_ref(),
            &all_frags,
        )
        .await
        .unwrap();

        let merged = ZoneMapIndex::load(dest_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

        // Index B had null_rows = None, so the merged index cannot know all null positions.
        // IsNull must NOT return exact — that would be a false negative for fragment 1's nulls.
        let result = merged
            .search(&SargableQuery::IsNull(), &NoOpMetricsCollector)
            .await
            .unwrap();

        // With the bug: any_null_bitmap=true (from A) → null_rows=Some(A's bitmap only)
        //              → IsNull returns exact, missing B's unknown nulls ← FALSE NEGATIVE
        // With the fix: any_null_bitmap=false (because B is None) → null_rows=None
        //              → IsNull falls through to zone scan → AtMost
        assert!(
            !result.is_exact(),
            "IsNull on a merged index where one source had null_rows=None must not return \
             exact; the legacy segment had null_count=3 so its nulls exist at unknown positions"
        );
    }

    // Writes a zonemap file in the legacy format (no null bitmap global buffer),
    // simulating an index created before the null bitmap feature was added.
    async fn write_legacy_zonemap(store: &dyn IndexStore, null_count: u32) {
        use arrow_array::{Int32Array, UInt32Array};
        let schema = Arc::new(Schema::new(vec![
            Field::new("min", DataType::Int32, true),
            Field::new("max", DataType::Int32, true),
            Field::new("null_count", DataType::UInt32, false),
            Field::new("nan_count", DataType::UInt32, false),
            Field::new("fragment_id", DataType::UInt64, false),
            Field::new("zone_start", DataType::UInt64, false),
            Field::new("zone_length", DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![Some(0)])) as _,
                Arc::new(Int32Array::from(vec![Some(99)])) as _,
                Arc::new(UInt32Array::from(vec![null_count])) as _,
                Arc::new(UInt32Array::from(vec![0u32])) as _,
                Arc::new(UInt64Array::from(vec![0u64])) as _,
                Arc::new(UInt64Array::from(vec![0u64])) as _,
                Arc::new(UInt64Array::from(vec![100u64])) as _,
            ],
        )
        .unwrap();
        let mut file_schema = schema.as_ref().clone();
        file_schema
            .metadata
            .insert(ZONEMAP_SIZE_META_KEY.to_string(), "8192".to_string());
        let mut writer = store
            .new_index_file(ZONEMAP_FILENAME, Arc::new(file_schema))
            .await
            .unwrap();
        writer.write_record_batch(batch).await.unwrap();
        writer.finish().await.unwrap();
    }

    #[tokio::test]
    async fn test_legacy_zonemap_no_null_bitmap() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Write a legacy index with one zone that has nulls but no null bitmap.
        write_legacy_zonemap(store.as_ref(), 10).await;

        let index = ZoneMapIndex::load(store, None, &LanceCache::no_cache())
            .await
            .expect("failed to load legacy zonemap");

        assert!(
            index.null_rows.is_none(),
            "legacy index should have no null bitmap"
        );

        // IS NULL should fall back to the zone-scan path and return AtMost, not Exact.
        let result = index
            .search(&SargableQuery::IsNull(), &NoOpMetricsCollector)
            .await
            .unwrap();
        assert!(
            !result.is_exact(),
            "IS NULL on a legacy index should not be exact"
        );
    }
}
