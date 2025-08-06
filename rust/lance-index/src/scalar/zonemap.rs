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
use super::btree::TrainingSource;
use crate::scalar::SargableQuery;
use crate::Any;
use datafusion::functions_aggregate::min_max::{MaxAccumulator, MinAccumulator};
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_expr::Accumulator;
use futures::TryStreamExt;
use lance_core::cache::LanceCache;
use lance_core::ROW_ADDR;
use std::env::temp_dir;
use std::sync::LazyLock;
use tempfile::{tempdir, TempDir};

use arrow_array::{new_empty_array, ArrayRef, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::execution::SendableRecordBatchStream;
use datafusion_common::ScalarValue;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use super::{AnyQuery, IndexReader, IndexStore, MetricsCollector, ScalarIndex, SearchResult};
use crate::scalar::FragReuseIndex;
use crate::vector::VectorIndex;
use crate::{Index, IndexType};
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use lance_core::Result;
use lance_core::{utils::mask::RowIdTreeMap, Error};
use log::debug;
use roaring::RoaringBitmap;
use snafu::location;
const ZONEMAP_DEFAULT_SIZE: usize = 8192; // 1 zone every two batches

const ZONEMAP_FILENAME: &str = "zonemap.lance";
const ZONEMAP_SIZE_META_KEY: &str = "zonemap_size";

/// Basic stats about zonemap index
#[derive(Debug, PartialEq)]
struct ZoneMapStatistics {
    min: ScalarValue,
    max: ScalarValue,
    null_count: u32,
    fragment_id: u64,
    zone_size: usize,
}

impl DeepSizeOf for ZoneMapStatistics {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        // Estimate sizes for ScalarValue
        let min_size = match &self.min {
            ScalarValue::Int32(_) => 4,
            ScalarValue::Int64(_) => 8,
            ScalarValue::Float32(_) => 4,
            ScalarValue::Float64(_) => 8,
            ScalarValue::Utf8(Some(s)) => s.len(),
            ScalarValue::Utf8(None) => 0,
            ScalarValue::LargeUtf8(None) => 0,
            _ => 16, // Default estimate
        };

        let max_size = match &self.max {
            ScalarValue::Int32(_) => 4,
            ScalarValue::Int64(_) => 8,
            ScalarValue::Float32(_) => 4,
            ScalarValue::Float64(_) => 8,
            ScalarValue::Utf8(Some(s)) => s.len(),
            ScalarValue::Utf8(None) => 0,
            ScalarValue::LargeUtf8(None) => 0,
            _ => 16, // Default estimate
        };

        min_size + max_size + 4 + 8 + 8
    }
}

/// ZoneMap index
/// At high level it's a columnar database technique for predicate push down and scan pruning.
/// It breaks data into fixed-size chunks called `zones` and store summary statistics(min, max, null_count) for each zone. It enables efficient filtering by skipping zones that do not contain matching values
///
/// This is an inexact filter, similar to a bloom filter. It can return false positives that require rechecking.
///
/// Note that it cannot return false negatives.
pub struct ZoneMapIndex {
    zones: Vec<ZoneMapStatistics>,
    data_type: DataType,
    // The maximum size of a zone provided by user
    max_zonemap_size: usize,
    store: Arc<dyn IndexStore>,
    fri: Option<Arc<FragReuseIndex>>,
    index_cache: LanceCache,
}

impl std::fmt::Debug for ZoneMapIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZoneMapIndex")
            .field("data_type", &self.data_type)
            .field("max_zonemap_size", &self.max_zonemap_size)
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

                // Check if target is within the zone's range
                Ok(target >= &zone.min && target <= &zone.max)
            }
            SargableQuery::Range(start, end) => {
                // Zone overlaps with query range if there's any intersection between
                // the zone's [min, max] and the query's range
                let zone_min = &zone.min;
                let zone_max = &zone.max;

                let start_check = match start {
                    Bound::Unbounded => true,
                    Bound::Included(s) => zone_max >= s,
                    Bound::Excluded(s) => zone_max > s,
                };

                let end_check = match end {
                    Bound::Unbounded => true,
                    Bound::Included(e) => zone_min <= e,
                    Bound::Excluded(e) => zone_min < e,
                };

                Ok(start_check && end_check)
            }
            SargableQuery::IsIn(values) => {
                // Zone contains matching values if any value in the set falls within [min, max]
                Ok(values.iter().any(|value| {
                    if value.is_null() {
                        zone.null_count > 0
                    } else {
                        value >= &zone.min && value <= &zone.max
                    }
                }))
            }
            SargableQuery::FullTextSearch(_) => {
                return Err(Error::NotSupported {
                    source: "full text search is not supported for zonemap indexes".into(),
                    location: location!(),
                });
            }
        }
    }

    fn try_from_serialized(
        data: RecordBatch,
        store: Arc<dyn IndexStore>,
        fri: Option<Arc<FragReuseIndex>>,
        index_cache: LanceCache,
        max_zonemap_size: usize,
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
        let zone_size = data
            .column_by_name("zone_size")
            .ok_or_else(|| {
                Error::invalid_input("ZoneMapIndex: missing 'zone_size' column", location!())
            })?
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| {
                Error::invalid_input(
                    "ZoneMapIndex: 'zone_size' column is not Uint64",
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

        let data_type = min_col.data_type().clone();

        if data.num_rows() == 0 {
            return Ok(Self {
                zones: Vec::new(),
                data_type: data_type,
                max_zonemap_size: max_zonemap_size,
                store,
                fri,
                index_cache,
            });
        }

        let num_zones = data.num_rows();
        let mut zones = Vec::with_capacity(num_zones);

        for i in 0..num_zones {
            let min = ScalarValue::try_from_array(min_col, i)?;
            let max = ScalarValue::try_from_array(max_col, i)?;
            let null_count = null_count_col.value(i);
            zones.push(ZoneMapStatistics {
                min,
                max,
                null_count,
                fragment_id: fragment_id_col.value(i),
                zone_size: zone_size.value(i) as usize,
            });
        }

        Ok(Self {
            zones,
            data_type,
            max_zonemap_size,
            store,
            fri,
            index_cache,
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
            "type": "ZoneMap",
            "num_zones": self.zones.len(),
            "max_zonemap_size": self.max_zonemap_size,
        }))
    }

    fn index_type(&self) -> IndexType {
        IndexType::ZoneMap
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let frag_ids = RoaringBitmap::new();
        Ok(frag_ids)
    }
}

#[async_trait]
impl ScalarIndex for ZoneMapIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        _metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let query = query.as_any().downcast_ref::<SargableQuery>().unwrap();

        let mut matching_zones = Vec::new();

        for (zone_idx, zone) in self.zones.iter().enumerate() {
            if self.evaluate_zone_against_query(zone, query)? {
                matching_zones.push(zone_idx);
            }
        }

        // TODO: Convert matching zone indices to actual row ID ranges
        // Each zone covers zonemap_size rows, except possibly the last zone which may have zonemap_offset rows
        // For now, return empty result - this is the missing piece for a complete implementation
        Ok(SearchResult::AtMost(RowIdTreeMap::new()))
    }

    fn can_answer_exact(&self, _: &dyn AnyQuery) -> bool {
        false
    }

    /// Load the scalar index from storage
    async fn load(
        store: Arc<dyn IndexStore>,
        fri: Option<Arc<FragReuseIndex>>,
        index_cache: LanceCache,
    ) -> Result<Arc<Self>>
    where
        Self: Sized,
    {
        let index_file = store.open_index_file(ZONEMAP_FILENAME).await?;
        let zone_maps = index_file
            .read_range(0..index_file.num_rows(), None)
            .await?;
        let file_schema = index_file.schema();
        // zonemap_size is expected to be a usize (or possibly u64), parsed from a string in the metadata
        let max_zonemap_size: usize = file_schema
            .metadata
            .get(ZONEMAP_SIZE_META_KEY)
            .and_then(|bs| bs.parse().ok())
            .unwrap_or(ZONEMAP_DEFAULT_SIZE);
        Ok(Arc::new(Self::try_from_serialized(
            zone_maps,
            store,
            fri,
            index_cache,
            max_zonemap_size,
        )?))
    }

    /// Remap the row ids, creating a new remapped version of this index in `dest_store`
    async fn remap(
        &self,
        _mapping: &HashMap<u64, Option<u64>>,
        _dest_store: &dyn IndexStore,
    ) -> Result<()> {
        // TODO: Implement remap logic
        Ok(())
    }

    /// Add the new data into the index, creating an updated version of the index in `dest_store`
    async fn update(
        &self,
        _new_data: SendableRecordBatchStream,
        _dest_store: &dyn IndexStore,
    ) -> Result<()> {
        // TODO: Implement update logic
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ZoneMapIndexBuilderOptions {
    rows_per_zone: usize,
}

static DEFAULT_ROWS_PER_ZONE: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("LANCE_ZONEMAP_DEFAULT_ROWS_PER_ZONE")
        .unwrap_or_else(|_| (ZONEMAP_DEFAULT_SIZE).to_string())
        .parse()
        .expect("failed to parse Lance_ZONEMAP_DEFAULT_ROWS_PER_ZONE")
});

impl Default for ZoneMapIndexBuilderOptions {
    fn default() -> Self {
        Self {
            rows_per_zone: *DEFAULT_ROWS_PER_ZONE,
        }
    }
}

impl ZoneMapIndexBuilderOptions {
    fn new(rows_per_zone: usize) -> Self {
        Self { rows_per_zone }
    }
}

// A builder for zonemap index
pub struct ZoneMapIndexBuilder {
    options: ZoneMapIndexBuilderOptions,

    items_type: DataType,
    maps: Vec<ZoneMapStatistics>,
    // TODO: check with weston on usize or u32
    cur_zone_offset: usize,
    cur_fragment_id: u64,

    min: MinAccumulator,
    max: MaxAccumulator,
    null_count: u32,
}

impl ZoneMapIndexBuilder {
    fn try_new(options: ZoneMapIndexBuilderOptions, items_type: DataType) -> Result<Self> {
        let min = MinAccumulator::try_new(&items_type)?;
        let max = MaxAccumulator::try_new(&items_type)?;
        Ok(Self {
            options: options.clone(),
            items_type,
            maps: Vec::new(),
            cur_zone_offset: 0,
            cur_fragment_id: 0,
            min,
            max,
            null_count: 0,
        })
    }

    fn update_stats(&mut self, array: &ArrayRef) -> Result<()> {
        self.null_count += array.null_count() as u32;
        self.min.update_batch(&[array.clone()])?;
        self.max.update_batch(&[array.clone()])?;
        Ok(())
    }

    fn new_map(&mut self, fragment_id: u64) -> Result<()> {
        let new_map = ZoneMapStatistics {
            min: self.min.evaluate()?,
            max: self.max.evaluate()?,
            null_count: self.null_count,
            fragment_id: fragment_id,
            zone_size: self.cur_zone_offset,
        };
        println!("new_map = {:?}", new_map);

        self.maps.push(new_map);

        self.cur_zone_offset = 0;
        self.min = MinAccumulator::try_new(&self.items_type)?;
        self.max = MaxAccumulator::try_new(&self.items_type)?;
        self.null_count = 0;
        Ok(())
    }

    pub async fn train(&mut self, mut batches_source: SendableRecordBatchStream) -> Result<()> {
        assert!(batches_source.schema().field_with_name(ROW_ADDR).is_ok());

        while let Some(batch) = batches_source.try_next().await? {
            println!("get a new batch");
            if batch.num_rows() == 0 {
                continue;
            }

            let data_array: &arrow_array::ArrayRef = batch.column(0);
            let row_addrs_array = batch
                .column_by_name(ROW_ADDR)
                .unwrap()
                .as_any()
                .downcast_ref::<arrow_array::UInt64Array>()
                .unwrap();

            let mut remaining = batch.num_rows() as usize;
            let mut array_offset: usize = 0;

            while remaining > 0 {
                // Find the next fragment boundary in this batch
                let next_fragment_index = (array_offset..row_addrs_array.len()).find(|&i| {
                    let row_addr = row_addrs_array.value(i);
                    let fragment_id = (row_addr >> 32) as u64;
                    fragment_id == self.cur_fragment_id + 1
                });
                let empty_rows_left_in_cur_zone: usize =
                    self.options.rows_per_zone - self.cur_zone_offset;

                // Check if there is enough data from the current fragment to fill the current zone
                let desired = if let Some(idx) = next_fragment_index {
                    // We found a fragment boundary, update cur_fragment_id
                    let next_fragment_id = (row_addrs_array.value(idx) >> 32) as u64;
                    self.cur_fragment_id = next_fragment_id;
                    // Don't create an empty zone, just process the data up to the boundary
                    idx - array_offset
                } else {
                    empty_rows_left_in_cur_zone
                };
                println!(
                    "desired = {}, next_fragment_index = {:?}, remaining = {}",
                    desired, next_fragment_index, remaining
                );

                if desired > remaining {
                    // Not enough data to fill a map, just increment counts
                    self.update_stats(&data_array.slice(array_offset, remaining as usize))?;
                    self.cur_zone_offset += remaining;
                    break;
                } else if desired > 0 {
                    // There is enough data, create a new zone map
                    self.update_stats(&data_array.slice(array_offset, desired as usize))?;
                    self.cur_zone_offset += desired;
                    self.new_map(row_addrs_array.value(array_offset) >> 32)?;
                } else if desired == 0 {
                    // The new batch starts with a new fragment. Flush the current zone if it's not empty
                    if self.cur_zone_offset > 0 {
                        self.new_map(self.cur_fragment_id - 1)?;
                    }
                    // Let the loop run again
                    // to find the next fragment boundary
                    continue;
                }
                array_offset += desired as usize;
                remaining = remaining.saturating_sub(desired);
                println!("array_offset = {}, remaining = {}", array_offset, remaining);
            }
        }
        // Create the final map
        if self.cur_zone_offset > 0 {
            self.new_map(self.cur_fragment_id)?;
        }

        // TODO return anything here???
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

        let fragment_ids =
            UInt64Array::from_iter_values(self.maps.iter().map(|stat| stat.fragment_id));

        let zone_sizes =
            UInt64Array::from_iter_values(self.maps.iter().map(|stat| stat.zone_size as u64));

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            // min and max can be null if the entire batch is null values
            Field::new("min", self.items_type.clone(), true),
            Field::new("max", self.items_type.clone(), true),
            Field::new("null_count", DataType::UInt32, false),
            Field::new("fragment_id", DataType::UInt64, false),
            Field::new("zone_size", DataType::UInt64, false),
        ]));

        let columns: Vec<ArrayRef> = vec![
            mins,
            maxs,
            Arc::new(null_counts) as ArrayRef,
            Arc::new(fragment_ids) as ArrayRef,
            Arc::new(zone_sizes) as ArrayRef,
        ];
        Ok(RecordBatch::try_new(schema, columns)?)
    }

    pub async fn write_index(mut self, index_store: &dyn IndexStore) -> Result<()> {
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

pub async fn train_zonemap_index(
    data_source: Box<dyn TrainingSource + Send>,
    index_store: &dyn IndexStore,
    options: Option<ZoneMapIndexBuilderOptions>,
) -> Result<()> {
    println!("train_zonemap_index: calling scan_aligned_chunks");
    let batches_source = data_source.scan_aligned_chunks(4096).await?;
    let value_type = batches_source.schema().field(0).data_type().clone();

    let mut builder = ZoneMapIndexBuilder::try_new(
        options.unwrap_or(ZoneMapIndexBuilderOptions::default()),
        value_type,
    )?;

    builder.train(batches_source).await?;

    builder.write_index(index_store).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::scalar::{zonemap::ZONEMAP_DEFAULT_SIZE, IndexStore};
    use std::{
        collections::{HashMap, HashSet},
        sync::Arc,
    };

    use crate::scalar::zonemap::{train_zonemap_index, ZoneMapStatistics};
    use arrow::datatypes::{Float32Type, UInt64Type};
    use arrow_array::{Array, RecordBatch, StringArray, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        execution::SendableRecordBatchStream, physical_plan::stream::RecordBatchStreamAdapter,
    };
    use datafusion_common::DataFusionError;
    use datafusion_common::ScalarValue;
    use datafusion_sql::sqlparser::keywords::ZONE;
    use futures::{stream, StreamExt, TryStreamExt};
    use itertools::Itertools;
    use lance_core::{cache::LanceCache, utils::mask::RowIdTreeMap, ROW_ADDR};
    use lance_datafusion::datagen::DatafusionDatagenExt;
    use lance_datagen::ArrayGeneratorExt;
    use lance_datagen::{array, gen, BatchCount, ByteCount, RowCount};
    use lance_io::object_store::ObjectStore;
    use object_store::path::Path;
    use tantivy::tokenizer::TextAnalyzer;
    use tempfile::{tempdir, TempDir};

    use crate::scalar::lance_format::tests::MockTrainingSource;

    use crate::scalar::{
        lance_format::LanceIndexStore,
        zonemap::{
            ZoneMapIndex, ZoneMapIndexBuilder, ZoneMapIndexBuilderOptions, ZONEMAP_FILENAME,
            ZONEMAP_SIZE_META_KEY,
        },
        ScalarIndex, SearchResult, TextQuery,
    };

    use crate::scalar::btree::TrainingSource;

    #[tokio::test]
    async fn test_empty_zonemap_index() {
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        let data = arrow_array::Int32Array::from(Vec::<i32>::new());
        let row_ids = arrow_array::UInt64Array::from(Vec::<u64>::new());
        let schema = Arc::new(Schema::new(vec![
            Field::new("values", DataType::Int32, false),
            Field::new("row_ids", DataType::UInt64, false),
        ]));
        let data =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();

        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        let data_source = Box::new(MockTrainingSource::from(data_stream));
        train_zonemap_index(
            data_source,
            test_store.as_ref(),
            Some(ZoneMapIndexBuilderOptions::default()),
        )
        .await
        .unwrap();

        log::debug!("Successfully wrote the index file");

        // Read the index file back and check its contents
        let index = ZoneMapIndex::load(test_store.clone(), None, LanceCache::no_cache())
            .await
            .expect("Failed to load ZoneMapIndex");
        assert_eq!(index.zones.len(), 0);
        assert_eq!(index.data_type, DataType::Int32);
        assert_eq!(index.max_zonemap_size, ZONEMAP_DEFAULT_SIZE);
    }

    #[tokio::test]
    // Test that a zonemap index can be created with null values
    async fn test_null_zonemap_index() {
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        let stream = gen()
            .col(
                "value",
                array::rand::<Float32Type>().with_nulls(&[true, false, false, false, false]),
            )
            .col("_rowid", array::step::<UInt64Type>())
            .into_df_stream(RowCount::from(5000), BatchCount::from(10));
        let data_source = Box::new(MockTrainingSource::from(stream));
        train_zonemap_index(
            data_source,
            test_store.as_ref(),
            Some(ZoneMapIndexBuilderOptions::new(5000)),
        )
        .await
        .unwrap();

        log::debug!("Successfully wrote the index file");

        // Read the index file back and check its contents
        let index = ZoneMapIndex::load(test_store.clone(), None, LanceCache::no_cache())
            .await
            .expect("Failed to load ZoneMapIndex");
        assert_eq!(index.zones.len(), 10);
        for (i, zone) in index.zones.iter().enumerate() {
            println!("zone = {:?} i= {}", zone, i);
            assert_eq!(zone.null_count, 1000);
            assert_eq!(zone.zone_size, 5000);
            assert_eq!(zone.fragment_id, i as u64);
        }
    }

    #[tokio::test]
    // Test data that belongs to the same fragment but coming from different batches
    async fn test_basic_zonemap_index() {
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        let data = arrow_array::Int32Array::from_iter_values(0..=100);
        let row_ids = UInt64Array::from_iter_values((0..data.len()).map(|i| i as u64));
        let schema = Arc::new(Schema::new(vec![
            Field::new("values", DataType::Int32, false),
            Field::new("row_ids", DataType::UInt64, false),
        ]));
        let data =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        let data_source = Box::new(MockTrainingSource::from(data_stream));
        train_zonemap_index(
            data_source,
            test_store.as_ref(),
            Some(ZoneMapIndexBuilderOptions::new(100)),
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
        let index = ZoneMapIndex::load(test_store.clone(), None, LanceCache::no_cache())
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
                    fragment_id: 0,
                    zone_size: 100,
                },
                ZoneMapStatistics {
                    min: ScalarValue::Int32(Some(100)),
                    max: ScalarValue::Int32(Some(100)),
                    null_count: 0,
                    fragment_id: 0,
                    zone_size: 1,
                }
            ]
        );
        assert_eq!(index.data_type, DataType::Int32);
        assert_eq!(index.max_zonemap_size, 100);
    }

    #[tokio::test]
    async fn test_complex_zonemap_index() {
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Create data that will produce the expected zonemap zones
        let data =
            arrow_array::Int64Array::from_iter_values(0..(ZONEMAP_DEFAULT_SIZE * 2 + 42) as i64);
        let row_ids = UInt64Array::from_iter_values((0..data.len()).map(|i| i as u64));
        let schema = Arc::new(Schema::new(vec![
            Field::new("values", DataType::Int64, false),
            Field::new("row_ids", DataType::UInt64, false),
        ]));
        let data =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        let data_source = Box::new(MockTrainingSource::from(data_stream));
        train_zonemap_index(
            data_source,
            test_store.as_ref(),
            Some(ZoneMapIndexBuilderOptions::default()),
        )
        .await
        .unwrap();

        log::debug!("Successfully wrote the index file");

        // Read the index file back and check its contents
        let index = ZoneMapIndex::load(test_store.clone(), None, LanceCache::no_cache())
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
                    fragment_id: 0,
                    zone_size: 8192,
                },
                ZoneMapStatistics {
                    min: ScalarValue::Int64(Some(8192)),
                    max: ScalarValue::Int64(Some(16383)),
                    null_count: 0,
                    fragment_id: 0,
                    zone_size: 8192,
                },
                ZoneMapStatistics {
                    min: ScalarValue::Int64(Some(16384)),
                    max: ScalarValue::Int64(Some(16425)),
                    null_count: 0,
                    fragment_id: 0,
                    zone_size: 42,
                }
            ]
        );
        assert_eq!(index.data_type, DataType::Int64);
        assert_eq!(index.max_zonemap_size, ZONEMAP_DEFAULT_SIZE);

        // TODO: Test search functionality
    }

    #[tokio::test]
    // Test zonemap with multiple fragments from different batches
    async fn test_multiple_fragments_zonemap() {
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        let schema = Arc::new(Schema::new(vec![
            Field::new("values", DataType::Int64, false),
            Field::new("row_ids", DataType::UInt64, false),
        ]));

        // Create multiple fragments with data that will produce expected zones
        // Fragment 0: values 0-8191 (first zone)
        let fragment0_data =
            arrow_array::Int64Array::from_iter_values(0..ZONEMAP_DEFAULT_SIZE as i64);
        let fragment0_row_ids =
            UInt64Array::from_iter_values((0..ZONEMAP_DEFAULT_SIZE as u64).map(|i| i as u64));
        let fragment0_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(fragment0_data), Arc::new(fragment0_row_ids)],
        )
        .unwrap();

        // Fragment 1: values 8192-16383 (second zone)
        let fragment1_data = arrow_array::Int64Array::from_iter_values(
            (ZONEMAP_DEFAULT_SIZE as i64)..((ZONEMAP_DEFAULT_SIZE * 2) as i64),
        );
        let fragment1_row_ids =
            UInt64Array::from_iter_values((0..ZONEMAP_DEFAULT_SIZE as u64).map(|i| i as u64));
        let fragment1_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(fragment1_data), Arc::new(fragment1_row_ids)],
        )
        .unwrap();

        // Fragment 2: values 16384-16426 (third zone)
        let fragment2_data = arrow_array::Int64Array::from_iter_values(
            ((ZONEMAP_DEFAULT_SIZE * 2) as i64)..((ZONEMAP_DEFAULT_SIZE * 2 + 42) as i64),
        );
        let fragment2_row_ids = UInt64Array::from_iter_values((0..42).map(|i| i as u64));
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
            let data_source = Box::new(MockTrainingSource::from(data_stream));
            train_zonemap_index(
                data_source,
                test_store.as_ref(),
                Some(ZoneMapIndexBuilderOptions::new(5000)),
            )
            .await
            .unwrap();

            // Read the index file back and check its contents
            let index = ZoneMapIndex::load(test_store.clone(), None, LanceCache::no_cache())
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
                        fragment_id: 0,
                        zone_size: 5000,
                    },
                    ZoneMapStatistics {
                        min: ScalarValue::Int64(Some(5000)),
                        max: ScalarValue::Int64(Some(8191)),
                        null_count: 0,
                        fragment_id: 0,
                        zone_size: 3192,
                    },
                    ZoneMapStatistics {
                        min: ScalarValue::Int64(Some(8192)),
                        max: ScalarValue::Int64(Some(13191)),
                        null_count: 0,
                        fragment_id: 1,
                        zone_size: 5000,
                    },
                    ZoneMapStatistics {
                        min: ScalarValue::Int64(Some(13192)),
                        max: ScalarValue::Int64(Some(16383)),
                        null_count: 0,
                        fragment_id: 1,
                        zone_size: 3192,
                    },
                    ZoneMapStatistics {
                        min: ScalarValue::Int64(Some(16384)),
                        max: ScalarValue::Int64(Some(16425)),
                        null_count: 0,
                        fragment_id: 2,
                        zone_size: 42,
                    }
                ]
            );
            assert_eq!(index.data_type, DataType::Int64);
            assert_eq!(index.max_zonemap_size, 5000);

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
            let verify_data_source = Box::new(MockTrainingSource::from(verify_data_stream));
            let aligned_stream = verify_data_source.scan_aligned_chunks(4096).await.unwrap();
            let batches: Vec<RecordBatch> = aligned_stream.try_collect().await.unwrap();

            assert_eq!(batches.len(), 3);

            // Check fragment 0 _rowaddr values (should start from 0)
            let fragment0_rowaddr_col = batches[0].column_by_name("_rowaddr").unwrap();
            let fragment0_rowaddrs = fragment0_rowaddr_col
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            assert_eq!(fragment0_rowaddrs.values().len(), ZONEMAP_DEFAULT_SIZE);
            assert_eq!(fragment0_rowaddrs.values()[0], 0);
            assert_eq!(
                fragment0_rowaddrs.values()[fragment0_rowaddrs.values().len() - 1],
                8191
            );

            // Check fragment 1 _rowaddr values (should start from fragment_id=1)
            let fragment1_rowaddr_col = batches[1].column_by_name("_rowaddr").unwrap();
            let fragment1_rowaddrs = fragment1_rowaddr_col
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            assert_eq!(fragment1_rowaddrs.values().len(), ZONEMAP_DEFAULT_SIZE);
            assert_eq!(fragment1_rowaddrs.values()[0], 1u64 << 32); // fragment_id=1, local_offset=0
            assert_eq!(
                fragment1_rowaddrs.values()[fragment1_rowaddrs.values().len() - 1],
                8191 | 1u64 << 32
            );

            // Check fragment 2 _rowaddr values (should start from fragment_id=2)
            let fragment2_rowaddr_col = batches[2].column_by_name("_rowaddr").unwrap();
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
            let data_source = Box::new(MockTrainingSource::from(data_stream));
            train_zonemap_index(
                data_source,
                test_store.as_ref(),
                Some(ZoneMapIndexBuilderOptions::default()),
            )
            .await
            .unwrap();

            // Read the index file back and check its contents
            let index = ZoneMapIndex::load(test_store.clone(), None, LanceCache::no_cache())
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
                        fragment_id: 0,
                        zone_size: 8192,
                    },
                    ZoneMapStatistics {
                        min: ScalarValue::Int64(Some(8192)),
                        max: ScalarValue::Int64(Some(16383)),
                        null_count: 0,
                        fragment_id: 1,
                        zone_size: 8192,
                    },
                    ZoneMapStatistics {
                        min: ScalarValue::Int64(Some(16384)),
                        max: ScalarValue::Int64(Some(16425)),
                        null_count: 0,
                        fragment_id: 2,
                        zone_size: 42,
                    }
                ]
            );
            assert_eq!(index.data_type, DataType::Int64);
            assert_eq!(index.max_zonemap_size, ZONEMAP_DEFAULT_SIZE);
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
            let data_source = Box::new(MockTrainingSource::from(data_stream));
            train_zonemap_index(
                data_source,
                test_store.as_ref(),
                Some(ZoneMapIndexBuilderOptions::new(ZONEMAP_DEFAULT_SIZE * 3)),
            )
            .await
            .unwrap();

            // Read the index file back and check its contents
            let index = ZoneMapIndex::load(test_store.clone(), None, LanceCache::no_cache())
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
                        fragment_id: 0,
                        zone_size: 8192,
                    },
                    ZoneMapStatistics {
                        min: ScalarValue::Int64(Some(8192)),
                        max: ScalarValue::Int64(Some(16383)),
                        null_count: 0,
                        fragment_id: 1,
                        zone_size: 8192,
                    },
                    ZoneMapStatistics {
                        min: ScalarValue::Int64(Some(16384)),
                        max: ScalarValue::Int64(Some(16425)),
                        null_count: 0,
                        fragment_id: 2,
                        zone_size: 42,
                    }
                ]
            );
            assert_eq!(index.data_type, DataType::Int64);
            assert_eq!(index.max_zonemap_size, ZONEMAP_DEFAULT_SIZE * 3);
        }
    }

    #[tokio::test]
    async fn test_fragment_id_assignment() {
        // Test that fragment IDs are properly assigned in _rowaddr values
        let schema = Arc::new(Schema::new(vec![
            Field::new("values", DataType::Int32, false),
            Field::new("row_ids", DataType::UInt64, false),
        ]));

        // Create multiple fragments
        let fragment0_data = arrow_array::Int32Array::from_iter_values(0..5);
        let fragment0_row_ids = UInt64Array::from_iter_values((0..5).map(|i| i as u64));
        let fragment0_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(fragment0_data), Arc::new(fragment0_row_ids)],
        )
        .unwrap();

        let fragment1_data = arrow_array::Int32Array::from_iter_values(5..10);
        let fragment1_row_ids = UInt64Array::from_iter_values((0..5).map(|i| i as u64));
        let fragment1_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(fragment1_data), Arc::new(fragment1_row_ids)],
        )
        .unwrap();

        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(vec![Ok(fragment0_batch), Ok(fragment1_batch)]),
        ));

        let data_source = Box::new(MockTrainingSource::from(data_stream));

        // Use scan_aligned_chunks to get the data with _rowaddr
        let aligned_stream = data_source.scan_aligned_chunks(100).await.unwrap();
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
    // Create another test following test_page_cache in  btree.rs
}
