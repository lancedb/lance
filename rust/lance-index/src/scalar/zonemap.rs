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
use crate::pbold;
use crate::scalar::expression::{SargableQueryParser, ScalarQueryParser};
use crate::scalar::registry::{
    ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
};
use crate::scalar::{
    BuiltinIndexType, CreatedIndex, SargableQuery, ScalarIndexParams, UpdateCriteria,
};
use crate::Any;
use datafusion::functions_aggregate::min_max::{MaxAccumulator, MinAccumulator};
use datafusion_expr::Accumulator;
use lance_core::cache::{LanceCache, WeakLanceCache};
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

use arrow_array::{new_empty_array, ArrayRef, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field};
use datafusion::execution::SendableRecordBatchStream;
use datafusion_common::ScalarValue;
use std::{collections::HashMap, sync::Arc};

use super::{AnyQuery, IndexStore, MetricsCollector, ScalarIndex, SearchResult};
use crate::scalar::FragReuseIndex;
use crate::vector::VectorIndex;
use crate::{Index, IndexType};
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use lance_core::Error;
use lance_core::Result;
use roaring::RoaringBitmap;
use snafu::location;

use super::zoned::{rebuild_zones, search_zones, ZoneBound, ZoneProcessor, ZoneTrainer};
const ROWS_PER_ZONE_DEFAULT: u64 = 8192; // 1 zone every two batches

const ZONEMAP_FILENAME: &str = "zonemap.lance";
const ZONEMAP_SIZE_META_KEY: &str = "rows_per_zone";
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
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
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
    fri: Option<Arc<FragReuseIndex>>,
    index_cache: WeakLanceCache,
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
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.zones.deep_size_of_children(context)
    }
}

impl ZoneMapIndex {
    /// Evaluates whether a zone could potentially contain values matching the query
    /// For NaN, total order is used here
    /// reference: https://doc.rust-lang.org/std/primitive.f64.html#method.total_cmp
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

                // Check if target is within the zone's range
                // Handle the case where zone.max is NaN (zone contains both finite values and NaN)
                let min_check = target >= &zone.min;
                let max_check = match &zone.max {
                    ScalarValue::Float16(Some(f)) if f.is_nan() => true,
                    ScalarValue::Float32(Some(f)) if f.is_nan() => true,
                    ScalarValue::Float64(Some(f)) if f.is_nan() => true,
                    _ => target <= &zone.max,
                };
                Ok(min_check && max_check)
            }
            SargableQuery::Range(start, end) => {
                // Zone overlaps with query range if there's any intersection between
                // the zone's [min, max] and the query's range
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
                                } else {
                                    value >= &zone.min && value <= &zone.max
                                }
                            }
                            ScalarValue::Float32(Some(f)) => {
                                if f.is_nan() {
                                    zone.nan_count > 0
                                } else {
                                    value >= &zone.min && value <= &zone.max
                                }
                            }
                            ScalarValue::Float64(Some(f)) => {
                                if f.is_nan() {
                                    zone.nan_count > 0
                                } else {
                                    value >= &zone.min && value <= &zone.max
                                }
                            }
                            _ => value >= &zone.min && value <= &zone.max,
                        }
                    }
                }))
            }
            SargableQuery::FullTextSearch(_) => Err(Error::NotSupported {
                source: "full text search is not supported for zonemap indexes".into(),
                location: location!(),
            }),
        }
    }

    /// Load the scalar index from storage
    async fn load(
        store: Arc<dyn IndexStore>,
        fri: Option<Arc<FragReuseIndex>>,
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
        Ok(Arc::new(Self::try_from_serialized(
            zone_maps,
            store,
            fri,
            index_cache,
            rows_per_zone,
        )?))
    }

    fn try_from_serialized(
        data: RecordBatch,
        store: Arc<dyn IndexStore>,
        fri: Option<Arc<FragReuseIndex>>,
        index_cache: &LanceCache,
        rows_per_zone: u64,
    ) -> Result<Self> {
        // The RecordBatch should have columns: min, max, null_count
        let min_col = data.column_by_name("min").ok_or_else(|| {
            Error::invalid_input("ZoneMapIndex: missing 'min' column", location!())
        })?;
        let max_col = data.column_by_name("max").ok_or_else(|| {
            Error::invalid_input("ZoneMapIndex: missing 'max' column", location!())
        })?;
        let null_count_col = data
            .column_by_name("null_count")
            .ok_or_else(|| {
                Error::invalid_input("ZoneMapIndex: missing 'null_count' column", location!())
            })?
            .as_any()
            .downcast_ref::<arrow_array::UInt32Array>()
            .ok_or_else(|| {
                Error::invalid_input(
                    "ZoneMapIndex: 'null_count' column is not UInt32",
                    location!(),
                )
            })?;
        let nan_count_col = data
            .column_by_name("nan_count")
            .ok_or_else(|| {
                Error::invalid_input("ZoneMapIndex: missing 'nan_count' column", location!())
            })?
            .as_any()
            .downcast_ref::<arrow_array::UInt32Array>()
            .ok_or_else(|| {
                Error::invalid_input(
                    "ZoneMapIndex: 'nan_count' column is not UInt32",
                    location!(),
                )
            })?;
        let zone_length = data
            .column_by_name("zone_length")
            .ok_or_else(|| {
                Error::invalid_input("ZoneMapIndex: missing 'zone_length' column", location!())
            })?
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| {
                Error::invalid_input(
                    "ZoneMapIndex: 'zone_length' column is not UInt64",
                    location!(),
                )
            })?;

        let fragment_id_col = data
            .column_by_name("fragment_id")
            .ok_or_else(|| {
                Error::invalid_input("ZoneMapIndex: missing 'fragment_id' column", location!())
            })?
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| {
                Error::invalid_input(
                    "ZoneMapIndex: 'fragment_id' column is not UInt64",
                    location!(),
                )
            })?;

        let zone_start_col = data
            .column_by_name("zone_start")
            .ok_or_else(|| {
                Error::invalid_input("ZoneMapIndex: missing 'zone_start' column", location!())
            })?
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| {
                Error::invalid_input(
                    "ZoneMapIndex: 'zone_start' column is not UInt64",
                    location!(),
                )
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

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn VectorIndex>> {
        Err(Error::InvalidInput {
            source: "ZoneMapIndex is not a vector index".into(),
            location: location!(),
        })
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
        _mapping: &HashMap<u64, Option<u64>>,
        _dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        Err(Error::InvalidInput {
            source: "ZoneMapIndex does not support remap".into(),
            location: location!(),
        })
    }

    /// Add the new data , creating an updated version of the index in `dest_store`
    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        // Train new zones for the incoming data stream
        let schema = new_data.schema();
        let value_type = schema.field(0).data_type().clone();

        let options = ZoneMapIndexBuilderParams::new(self.rows_per_zone);
        let processor = ZoneMapProcessor::new(value_type.clone())?;
        let trainer = ZoneTrainer::new(processor, self.rows_per_zone)?;
        let updated_zones = rebuild_zones(&self.zones, trainer, new_data).await?;

        // Serialize the combined zones back into the index file
        let mut builder = ZoneMapIndexBuilder::try_new(options, self.data_type.clone())?;
        builder.options.rows_per_zone = self.rows_per_zone;
        builder.maps = updated_zones;
        builder.write_index(dest_store).await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::ZoneMapIndexDetails::default())
                .unwrap(),
            index_version: ZONEMAP_INDEX_VERSION,
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
}

impl ZoneMapIndexBuilder {
    pub fn try_new(options: ZoneMapIndexBuilderParams, items_type: DataType) -> Result<Self> {
        Ok(Self {
            options,
            items_type,
            maps: Vec::new(),
        })
    }

    /// Train the builder using the shared zone trainer.  The input stream must contain
    /// the value column followed by `_rowaddr`, matching the dataset scan order enforced
    /// by the scalar index registry.
    pub async fn train(&mut self, batches_source: SendableRecordBatchStream) -> Result<()> {
        let processor = ZoneMapProcessor::new(self.items_type.clone())?;
        let trainer = ZoneTrainer::new(processor, self.options.rows_per_zone)?;
        self.maps = trainer.train(batches_source).await?;
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

    pub async fn write_index(self, index_store: &dyn IndexStore) -> Result<()> {
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
        index_file.finish().await?;
        Ok(())
    }
}

/// Index-specific processor that computes min/max statistics for each zone while the
/// trainer takes care of chunking and fragment boundaries.
struct ZoneMapProcessor {
    data_type: DataType,
    min: MinAccumulator,
    max: MaxAccumulator,
    null_count: u32,
    nan_count: u32,
}

impl ZoneMapProcessor {
    fn new(data_type: DataType) -> Result<Self> {
        let min = MinAccumulator::try_new(&data_type)?;
        let max = MaxAccumulator::try_new(&data_type)?;
        Ok(Self {
            data_type,
            min,
            max,
            null_count: 0,
            nan_count: 0,
        })
    }

    fn count_nans(array: &ArrayRef) -> u32 {
        match array.data_type() {
            DataType::Float16 => {
                let array = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float16Array>()
                    .unwrap();
                array.values().iter().filter(|&&x| x.is_nan()).count() as u32
            }
            DataType::Float32 => {
                let array = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .unwrap();
                array.values().iter().filter(|&&x| x.is_nan()).count() as u32
            }
            DataType::Float64 => {
                let array = array
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .unwrap();
                array.values().iter().filter(|&&x| x.is_nan()).count() as u32
            }
            _ => 0,
        }
    }
}

impl ZoneProcessor for ZoneMapProcessor {
    type ZoneStatistics = ZoneMapStatistics;

    fn process_chunk(&mut self, array: &ArrayRef) -> Result<()> {
        self.null_count += array.null_count() as u32;
        self.nan_count += Self::count_nans(array);
        self.min.update_batch(std::slice::from_ref(array))?;
        self.max.update_batch(std::slice::from_ref(array))?;
        Ok(())
    }

    fn finish_zone(&mut self, bound: ZoneBound) -> Result<Self::ZoneStatistics> {
        Ok(ZoneMapStatistics {
            min: self.min.evaluate()?,
            max: self.max.evaluate()?,
            null_count: self.null_count,
            nan_count: self.nan_count,
            bound,
        })
    }

    fn reset(&mut self) -> Result<()> {
        self.min = MinAccumulator::try_new(&self.data_type)?;
        self.max = MaxAccumulator::try_new(&self.data_type)?;
        self.null_count = 0;
        self.nan_count = 0;
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
    ) -> Result<()> {
        // train_zonemap_index: calling scan_aligned_chunks
        let value_type = batches_source.schema().field(0).data_type().clone();

        let mut builder = ZoneMapIndexBuilder::try_new(options.unwrap_or_default(), value_type)?;

        builder.train(batches_source).await?;

        builder.write_index(index_store).await?;
        Ok(())
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
impl ScalarIndexPlugin for ZoneMapIndexPlugin {
    fn name(&self) -> &str {
        "ZoneMap"
    }

    fn new_training_request(
        &self,
        params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        if field.data_type().is_nested() {
            return Err(Error::InvalidInput {
                source: "A zone map index can only be created on a non-nested field.".into(),
                location: location!(),
            });
        }

        let params = serde_json::from_str::<ZoneMapIndexBuilderParams>(params)?;

        Ok(Box::new(ZoneMapIndexTrainingRequest::new(params)))
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
        Some(Box::new(SargableQueryParser::new(index_name, true)))
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        fragment_ids: Option<Vec<u32>>,
    ) -> Result<CreatedIndex> {
        if fragment_ids.is_some() {
            return Err(Error::InvalidInput {
                source: "ZoneMap index does not support fragment training".into(),
                location: location!(),
            });
        }

        let request = (request as Box<dyn std::any::Any>)
            .downcast::<ZoneMapIndexTrainingRequest>()
            .map_err(|_| Error::InvalidInput {
                source: "must provide training request created by new_training_request".into(),
                location: location!(),
            })?;
        Self::train_zonemap_index(data, index_store, Some(request.params)).await?;
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::ZoneMapIndexDetails::default())
                .unwrap(),
            index_version: ZONEMAP_INDEX_VERSION,
        })
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        Ok(ZoneMapIndex::load(index_store, frag_reuse_index, cache).await? as Arc<dyn ScalarIndex>)
    }
}

#[cfg(test)]
mod tests {
    use crate::scalar::registry::VALUE_COLUMN_NAME;
    use crate::scalar::{zonemap::ROWS_PER_ZONE_DEFAULT, IndexStore};
    use std::sync::Arc;

    use crate::scalar::zoned::ZoneBound;
    use crate::scalar::zonemap::{ZoneMapIndexPlugin, ZoneMapStatistics};
    use arrow::datatypes::Float32Type;
    use arrow_array::{Array, RecordBatch, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::execution::SendableRecordBatchStream;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use datafusion_common::ScalarValue;
    use futures::{stream, StreamExt, TryStreamExt};
    use lance_core::utils::tempfile::TempObjDir;
    use lance_core::{cache::LanceCache, utils::mask::RowAddrTreeMap, ROW_ADDR};
    use lance_datafusion::datagen::DatafusionDatagenExt;
    use lance_datagen::ArrayGeneratorExt;
    use lance_datagen::{array, BatchCount, RowCount};
    use lance_io::object_store::ObjectStore;

    use crate::scalar::{
        lance_format::LanceIndexStore,
        zonemap::{
            ZoneMapIndex, ZoneMapIndexBuilderParams, ZONEMAP_FILENAME, ZONEMAP_SIZE_META_KEY,
        },
        SargableQuery, ScalarIndex, SearchResult,
    };

    // Add missing imports for the tests
    use crate::metrics::NoOpMetricsCollector;
    use crate::Index; // Import Index trait to access calculate_included_frags
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
        assert_eq!(result, SearchResult::AtMost(RowAddrTreeMap::new()));
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
        assert_eq!(result, SearchResult::AtMost(expected));

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
            .update(new_data_stream, test_store.as_ref())
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
        assert_eq!(result, SearchResult::AtMost(expected));

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
        assert_eq!(result, SearchResult::AtMost(expected));
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

        // Test search for NaN values using Equals with NaN
        let query = SargableQuery::Equals(ScalarValue::Float32(Some(f32::NAN)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match all zones since they all contain NaN values
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(0..500); // All rows since NaN is in every zone
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a specific finite value that exists in the data
        let query = SargableQuery::Equals(ScalarValue::Float32(Some(5.0)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match only the first zone since 5.0 only exists in rows 0-99
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(0..100);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a value that doesn't exist
        let query = SargableQuery::Equals(ScalarValue::Float32(Some(1000.0)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Since zones contain NaN values, their max will be NaN, so they will be included
        // as potential matches for any finite target (false positive, but acceptable for zone maps)
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(0..500);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test range query that should include finite values
        let query = SargableQuery::Range(
            Bound::Included(ScalarValue::Float32(Some(0.0))),
            Bound::Included(ScalarValue::Float32(Some(250.0))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match the first three zones since they contain values in the range [0, 250]
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(0..300);
        assert_eq!(result, SearchResult::AtMost(expected));

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
        assert_eq!(result, SearchResult::AtMost(expected));

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
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test IsNull query (should match nothing since there are no null values)
        let query = SargableQuery::IsNull();
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::AtMost(RowAddrTreeMap::new()));

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
        assert_eq!(result, SearchResult::AtMost(expected));

        // Range with NaN as end bound (included)
        let query = SargableQuery::Range(
            Bound::Unbounded,
            Bound::Included(ScalarValue::Float32(Some(f32::NAN))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        // Should match all zones since they all contain NaN values
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(0..500);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Range with NaN as end bound (excluded)
        let query = SargableQuery::Range(
            Bound::Unbounded,
            Bound::Excluded(ScalarValue::Float32(Some(f32::NAN))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        // Should match all zones since everything is less than NaN
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(0..500);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Range with NaN as start bound (excluded)
        let query = SargableQuery::Range(
            Bound::Excluded(ScalarValue::Float32(Some(f32::NAN))),
            Bound::Unbounded,
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        // Should match nothing since nothing is greater than NaN
        assert_eq!(result, SearchResult::AtMost(RowAddrTreeMap::new()));

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
        assert_eq!(result, SearchResult::AtMost(expected));
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
        assert_eq!(
            result,
            SearchResult::AtMost(RowAddrTreeMap::from_iter(0..=100))
        );

        // 2. Range query: [0, 50]
        let query = SargableQuery::Range(
            Bound::Included(ScalarValue::Int32(Some(0))),
            Bound::Included(ScalarValue::Int32(Some(50))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(
            result,
            SearchResult::AtMost(RowAddrTreeMap::from_iter(0..=99))
        );

        // 3. Range query: [101, 200] (should only match the second zone, which is row 100)
        let query = SargableQuery::Range(
            Bound::Included(ScalarValue::Int32(Some(101))),
            Bound::Included(ScalarValue::Int32(Some(200))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        // Only row 100 is in the second zone, but its value is 100, so this should be empty
        assert_eq!(result, SearchResult::AtMost(RowAddrTreeMap::new()));

        // 4. Range query: [100, 100] (should match only the last row)
        let query = SargableQuery::Range(
            Bound::Included(ScalarValue::Int32(Some(100))),
            Bound::Included(ScalarValue::Int32(Some(100))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(
            result,
            SearchResult::AtMost(RowAddrTreeMap::from_iter(100..=100))
        );

        // 5. Equals query: 0 (should match first row)
        let query = SargableQuery::Equals(ScalarValue::Int32(Some(0)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(
            result,
            SearchResult::AtMost(RowAddrTreeMap::from_iter(0..100))
        );

        // 6. Equals query: 100 (should match only last row)
        let query = SargableQuery::Equals(ScalarValue::Int32(Some(100)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(
            result,
            SearchResult::AtMost(RowAddrTreeMap::from_iter(100..=100))
        );

        // 7. Equals query: 101 (should match nothing)
        let query = SargableQuery::Equals(ScalarValue::Int32(Some(101)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::AtMost(RowAddrTreeMap::new()));

        // 8. IsNull query (no nulls in data, should match nothing)
        let query = SargableQuery::IsNull();
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::AtMost(RowAddrTreeMap::new()));

        // 9. IsIn query: [0, 100, 101, 50]
        let query = SargableQuery::IsIn(vec![
            ScalarValue::Int32(Some(0)),
            ScalarValue::Int32(Some(100)),
            ScalarValue::Int32(Some(101)),
            ScalarValue::Int32(Some(50)),
        ]);
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        // 0 and 50 are in the first zone, 100 in the second, 101 is not present
        assert_eq!(
            result,
            SearchResult::AtMost(RowAddrTreeMap::from_iter(0..=100))
        );

        // 10. IsIn query: [101, 102] (should match nothing)
        let query = SargableQuery::IsIn(vec![
            ScalarValue::Int32(Some(101)),
            ScalarValue::Int32(Some(102)),
        ]);
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::AtMost(RowAddrTreeMap::new()));

        // 11. IsIn query: [null] (should match nothing, as there are no nulls)
        let query = SargableQuery::IsIn(vec![ScalarValue::Int32(None)]);
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::AtMost(RowAddrTreeMap::new()));

        // 12. Equals query: null (should match nothing, as there are no nulls)
        let query = SargableQuery::Equals(ScalarValue::Int32(None));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::AtMost(RowAddrTreeMap::new()));
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
        assert_eq!(result, SearchResult::AtMost(expected));

        // Search for a value in the second zone
        let query = SargableQuery::Equals(ScalarValue::Int64(Some(9000)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        // Should match row 9000 in fragment 0: row address = (0 << 32) + 9000 = 9000
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(8192..=16383);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Search for a value not present in any zone
        let query = SargableQuery::Equals(ScalarValue::Int64(Some(20000)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::AtMost(RowAddrTreeMap::new()));

        // Search for a range that spans multiple zones
        let query = SargableQuery::Range(
            Bound::Included(ScalarValue::Int64(Some(9000))),
            Bound::Included(ScalarValue::Int64(Some(16400))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        // Should match all rows from 8000 to 16400 (inclusive)
        let mut expected = RowAddrTreeMap::new();
        expected.insert_range(8192..=16425);
        assert_eq!(result, SearchResult::AtMost(expected));
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
            assert_eq!(result, SearchResult::AtMost(expected));

            // Test exact match query from zone 2
            let query = SargableQuery::Equals(ScalarValue::Int64(Some(8192)));
            let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
            // Should include zone 2 since it contains value 8192
            let mut expected = RowAddrTreeMap::new();
            expected.insert_range((1u64 << 32)..((1u64 << 32) + 5000));
            assert_eq!(result, SearchResult::AtMost(expected));

            // Test exact match query from zone 4
            let query = SargableQuery::Equals(ScalarValue::Int64(Some(16385)));
            let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
            // Should include zone 4 since it contains value 16385
            let mut expected = RowAddrTreeMap::new();
            expected.insert_range(2u64 << 32..((2u64 << 32) + 42));
            assert_eq!(result, SearchResult::AtMost(expected));

            // Test query that matches nothing
            let query = SargableQuery::Equals(ScalarValue::Int64(Some(99999)));
            let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
            assert_eq!(result, SearchResult::AtMost(RowAddrTreeMap::new()));

            // Test is_in query
            let query = SargableQuery::IsIn(vec![ScalarValue::Int64(Some(16385))]);
            let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
            let mut expected = RowAddrTreeMap::new();
            expected.insert_range(2u64 << 32..((2u64 << 32) + 42));
            assert_eq!(result, SearchResult::AtMost(expected));

            // Test equals query with null
            let query = SargableQuery::Equals(ScalarValue::Int64(None));
            let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
            let mut expected = RowAddrTreeMap::new();
            expected.insert_range(0..=16425);
            // expected = {:?}", expected
            assert_eq!(result, SearchResult::AtMost(RowAddrTreeMap::new()));
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
}
