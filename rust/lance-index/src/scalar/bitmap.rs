// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    any::Any,
    collections::{BTreeMap, HashMap},
    fmt::Debug,
    ops::Bound,
    sync::Arc,
};

use crate::pbold;
use arrow::array::BinaryBuilder;
use arrow_array::{new_null_array, Array, BinaryArray, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion_common::ScalarValue;
use deepsize::DeepSizeOf;
use futures::{stream, StreamExt, TryStreamExt};
use lance_core::{
    cache::{CacheKey, LanceCache, WeakLanceCache},
    error::LanceOptionExt,
    utils::{mask::RowAddrTreeMap, tokio::get_num_compute_intensive_cpus},
    Error, Result, ROW_ID,
};
use roaring::RoaringBitmap;
use serde::Serialize;
use snafu::location;
use tracing::instrument;

use super::{
    btree::OrderableScalarValue, BuiltinIndexType, SargableQuery, ScalarIndexParams, SearchResult,
};
use super::{AnyQuery, IndexStore, ScalarIndex};
use crate::{
    frag_reuse::FragReuseIndex,
    scalar::{
        expression::SargableQueryParser,
        registry::{
            DefaultTrainingRequest, ScalarIndexPlugin, TrainingCriteria, TrainingOrdering,
            TrainingRequest, VALUE_COLUMN_NAME,
        },
        CreatedIndex, UpdateCriteria,
    },
};
use crate::{metrics::MetricsCollector, Index, IndexType};
use crate::{scalar::expression::ScalarQueryParser, scalar::IndexReader};

pub const BITMAP_LOOKUP_NAME: &str = "bitmap_page_lookup.lance";
pub const INDEX_STATS_METADATA_KEY: &str = "lance:index_stats";

const MAX_BITMAP_ARRAY_LENGTH: usize = i32::MAX as usize - 1024 * 1024; // leave headroom

const MAX_ROWS_PER_CHUNK: usize = 2 * 1024;

const BITMAP_INDEX_VERSION: u32 = 0;

// We only need to open a file reader if we need to load a bitmap. If all
// bitmaps are cached we don't open it. If we do open it we should only open it once.
#[derive(Clone)]
struct LazyIndexReader {
    index_reader: Arc<tokio::sync::Mutex<Option<Arc<dyn IndexReader>>>>,
    store: Arc<dyn IndexStore>,
}

impl std::fmt::Debug for LazyIndexReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyIndexReader")
            .field("store", &self.store)
            .finish()
    }
}

impl LazyIndexReader {
    fn new(store: Arc<dyn IndexStore>) -> Self {
        Self {
            index_reader: Arc::new(tokio::sync::Mutex::new(None)),
            store,
        }
    }

    async fn get(&self) -> Result<Arc<dyn IndexReader>> {
        let mut reader = self.index_reader.lock().await;
        if reader.is_none() {
            let index_reader = self.store.open_index_file(BITMAP_LOOKUP_NAME).await?;
            *reader = Some(index_reader);
        }
        Ok(reader.as_ref().unwrap().clone())
    }
}

/// A scalar index that stores a bitmap for each possible value
///
/// This index works best for low-cardinality columns, where the number of unique values is small.
/// The bitmap stores a list of row ids where the value is present.
#[derive(Clone, Debug)]
pub struct BitmapIndex {
    /// Maps each unique value to its bitmap location in the index file
    /// The usize value is the row offset in the bitmap_page_lookup.lance file
    /// for quickly locating the row and reading it out
    index_map: BTreeMap<OrderableScalarValue, usize>,

    null_map: Arc<RowAddrTreeMap>,

    value_type: DataType,

    store: Arc<dyn IndexStore>,

    index_cache: WeakLanceCache,

    frag_reuse_index: Option<Arc<FragReuseIndex>>,

    lazy_reader: LazyIndexReader,
}

#[derive(Debug, Clone)]
pub struct BitmapKey {
    value: OrderableScalarValue,
}

impl CacheKey for BitmapKey {
    type ValueType = RowAddrTreeMap;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("{}", self.value.0).into()
    }
}

impl BitmapIndex {
    fn new(
        index_map: BTreeMap<OrderableScalarValue, usize>,
        null_map: Arc<RowAddrTreeMap>,
        value_type: DataType,
        store: Arc<dyn IndexStore>,
        index_cache: WeakLanceCache,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Self {
        let lazy_reader = LazyIndexReader::new(store.clone());
        Self {
            index_map,
            null_map,
            value_type,
            store,
            index_cache,
            frag_reuse_index,
            lazy_reader,
        }
    }

    pub(crate) async fn load(
        store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        index_cache: &LanceCache,
    ) -> Result<Arc<Self>> {
        let page_lookup_file = store.open_index_file(BITMAP_LOOKUP_NAME).await?;
        let total_rows = page_lookup_file.num_rows();

        if total_rows == 0 {
            let schema = page_lookup_file.schema();
            let data_type = schema.fields[0].data_type();
            return Ok(Arc::new(Self::new(
                BTreeMap::new(),
                Arc::new(RowAddrTreeMap::default()),
                data_type,
                store,
                WeakLanceCache::from(index_cache),
                frag_reuse_index,
            )));
        }

        let mut index_map: BTreeMap<OrderableScalarValue, usize> = BTreeMap::new();
        let mut null_map = Arc::new(RowAddrTreeMap::default());
        let mut value_type: Option<DataType> = None;
        let mut null_location: Option<usize> = None;
        let mut row_offset = 0;

        for start_row in (0..total_rows).step_by(MAX_ROWS_PER_CHUNK) {
            let end_row = (start_row + MAX_ROWS_PER_CHUNK).min(total_rows);
            let chunk = page_lookup_file
                .read_range(start_row..end_row, Some(&["keys"]))
                .await?;

            if chunk.num_rows() == 0 {
                continue;
            }

            if value_type.is_none() {
                value_type = Some(chunk.schema().field(0).data_type().clone());
            }

            let dict_keys = chunk.column(0);

            for idx in 0..chunk.num_rows() {
                let key = OrderableScalarValue(ScalarValue::try_from_array(dict_keys, idx)?);

                if key.0.is_null() {
                    null_location = Some(row_offset);
                } else {
                    index_map.insert(key, row_offset);
                }

                row_offset += 1;
            }
        }

        if let Some(null_loc) = null_location {
            let batch = page_lookup_file
                .read_range(null_loc..null_loc + 1, Some(&["bitmaps"]))
                .await?;

            let binary_bitmaps = batch
                .column(0)
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| Error::Internal {
                    message: "Invalid bitmap column type".to_string(),
                    location: location!(),
                })?;
            let bitmap_bytes = binary_bitmaps.value(0);
            let mut bitmap = RowAddrTreeMap::deserialize_from(bitmap_bytes).unwrap();

            // Apply fragment remapping if needed
            if let Some(fri) = &frag_reuse_index {
                bitmap = fri.remap_row_addrs_tree_map(&bitmap);
            }

            null_map = Arc::new(bitmap);
        }

        let final_value_type = value_type.expect_ok()?;

        Ok(Arc::new(Self::new(
            index_map,
            null_map,
            final_value_type,
            store,
            WeakLanceCache::from(index_cache),
            frag_reuse_index,
        )))
    }

    async fn load_bitmap(
        &self,
        key: &OrderableScalarValue,
        metrics: Option<&dyn MetricsCollector>,
    ) -> Result<Arc<RowAddrTreeMap>> {
        if key.0.is_null() {
            return Ok(self.null_map.clone());
        }

        let cache_key = BitmapKey { value: key.clone() };

        if let Some(cached) = self.index_cache.get_with_key(&cache_key).await {
            return Ok(cached);
        }

        // Record that we're loading a partition from disk
        if let Some(metrics) = metrics {
            metrics.record_part_load();
        }

        let row_offset = match self.index_map.get(key) {
            Some(loc) => *loc,
            None => return Ok(Arc::new(RowAddrTreeMap::default())),
        };

        let page_lookup_file = self.lazy_reader.get().await?;
        let batch = page_lookup_file
            .read_range(row_offset..row_offset + 1, Some(&["bitmaps"]))
            .await?;

        let binary_bitmaps = batch
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| Error::Internal {
                message: "Invalid bitmap column type".to_string(),
                location: location!(),
            })?;
        let bitmap_bytes = binary_bitmaps.value(0); // First (and only) row
        let mut bitmap = RowAddrTreeMap::deserialize_from(bitmap_bytes).unwrap();

        if let Some(fri) = &self.frag_reuse_index {
            bitmap = fri.remap_row_addrs_tree_map(&bitmap);
        }

        self.index_cache
            .insert_with_key(&cache_key, Arc::new(bitmap.clone()))
            .await;

        Ok(Arc::new(bitmap))
    }
}

impl DeepSizeOf for BitmapIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        let mut total_size = 0;

        total_size += self.index_map.deep_size_of_children(context);
        total_size += self.store.deep_size_of_children(context);

        total_size
    }
}

#[derive(Serialize)]
struct BitmapStatistics {
    num_bitmaps: usize,
}

#[async_trait]
impl Index for BitmapIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(Error::NotSupported {
            source: "BitmapIndex is not a vector index".into(),
            location: location!(),
        })
    }

    async fn prewarm(&self) -> Result<()> {
        let page_lookup_file = self.lazy_reader.get().await?;
        let total_rows = page_lookup_file.num_rows();

        if total_rows == 0 {
            return Ok(());
        }

        for start_row in (0..total_rows).step_by(MAX_ROWS_PER_CHUNK) {
            let end_row = (start_row + MAX_ROWS_PER_CHUNK).min(total_rows);
            let chunk = page_lookup_file
                .read_range(start_row..end_row, None)
                .await?;

            if chunk.num_rows() == 0 {
                continue;
            }

            let dict_keys = chunk.column(0);
            let binary_bitmaps = chunk.column(1);
            let bitmap_binary_array = binary_bitmaps
                .as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap();

            for idx in 0..chunk.num_rows() {
                let key = OrderableScalarValue(ScalarValue::try_from_array(dict_keys, idx)?);

                if key.0.is_null() {
                    continue;
                }

                let bitmap_bytes = bitmap_binary_array.value(idx);
                let mut bitmap = RowAddrTreeMap::deserialize_from(bitmap_bytes).unwrap();

                if let Some(frag_reuse_index_ref) = self.frag_reuse_index.as_ref() {
                    bitmap = frag_reuse_index_ref.remap_row_addrs_tree_map(&bitmap);
                }

                let cache_key = BitmapKey { value: key };
                self.index_cache
                    .insert_with_key(&cache_key, Arc::new(bitmap))
                    .await;
            }
        }

        Ok(())
    }

    fn index_type(&self) -> IndexType {
        IndexType::Bitmap
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let stats = BitmapStatistics {
            num_bitmaps: self.index_map.len() + if !self.null_map.is_empty() { 1 } else { 0 },
        };
        serde_json::to_value(stats).map_err(|e| Error::Internal {
            message: format!("failed to serialize bitmap index statistics: {}", e),
            location: location!(),
        })
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!()
    }
}

#[async_trait]
impl ScalarIndex for BitmapIndex {
    #[instrument(name = "bitmap_search", level = "debug", skip_all)]
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let query = query.as_any().downcast_ref::<SargableQuery>().unwrap();

        let row_ids = match query {
            SargableQuery::Equals(val) => {
                metrics.record_comparisons(1);
                if val.is_null() {
                    (*self.null_map).clone()
                } else {
                    let key = OrderableScalarValue(val.clone());
                    let bitmap = self.load_bitmap(&key, Some(metrics)).await?;
                    (*bitmap).clone()
                }
            }
            SargableQuery::Range(start, end) => {
                let range_start = match start {
                    Bound::Included(val) => Bound::Included(OrderableScalarValue(val.clone())),
                    Bound::Excluded(val) => Bound::Excluded(OrderableScalarValue(val.clone())),
                    Bound::Unbounded => Bound::Unbounded,
                };

                let range_end = match end {
                    Bound::Included(val) => Bound::Included(OrderableScalarValue(val.clone())),
                    Bound::Excluded(val) => Bound::Excluded(OrderableScalarValue(val.clone())),
                    Bound::Unbounded => Bound::Unbounded,
                };

                let keys: Vec<_> = self
                    .index_map
                    .range((range_start, range_end))
                    .map(|(k, _v)| k.clone())
                    .collect();

                metrics.record_comparisons(keys.len());

                if keys.is_empty() {
                    RowAddrTreeMap::default()
                } else {
                    let bitmaps: Vec<_> = stream::iter(
                        keys.into_iter()
                            .map(|key| async move { self.load_bitmap(&key, None).await }),
                    )
                    .buffer_unordered(get_num_compute_intensive_cpus())
                    .try_collect()
                    .await?;

                    let bitmap_refs: Vec<_> = bitmaps.iter().map(|b| b.as_ref()).collect();
                    RowAddrTreeMap::union_all(&bitmap_refs)
                }
            }
            SargableQuery::IsIn(values) => {
                metrics.record_comparisons(values.len());

                // Collect keys that exist in the index, tracking if we need nulls
                let mut has_null = false;
                let keys: Vec<_> = values
                    .iter()
                    .filter_map(|val| {
                        if val.is_null() {
                            has_null = true;
                            None
                        } else {
                            let key = OrderableScalarValue(val.clone());
                            if self.index_map.contains_key(&key) {
                                Some(key)
                            } else {
                                None
                            }
                        }
                    })
                    .collect();

                if keys.is_empty() && (!has_null || self.null_map.is_empty()) {
                    RowAddrTreeMap::default()
                } else {
                    // Load bitmaps in parallel
                    let mut bitmaps: Vec<_> = stream::iter(
                        keys.into_iter()
                            .map(|key| async move { self.load_bitmap(&key, None).await }),
                    )
                    .buffer_unordered(get_num_compute_intensive_cpus())
                    .try_collect()
                    .await?;

                    // Add null bitmap if needed
                    if has_null && !self.null_map.is_empty() {
                        bitmaps.push(self.null_map.clone());
                    }

                    if bitmaps.is_empty() {
                        RowAddrTreeMap::default()
                    } else {
                        // Convert Arc<RowAddrTreeMap> to &RowAddrTreeMap for union_all
                        let bitmap_refs: Vec<_> = bitmaps.iter().map(|b| b.as_ref()).collect();
                        RowAddrTreeMap::union_all(&bitmap_refs)
                    }
                }
            }
            SargableQuery::IsNull() => {
                metrics.record_comparisons(1);
                (*self.null_map).clone()
            }
            SargableQuery::FullTextSearch(_) => {
                return Err(Error::NotSupported {
                    source: "full text search is not supported for bitmap indexes".into(),
                    location: location!(),
                });
            }
        };

        Ok(SearchResult::Exact(row_ids))
    }

    fn can_remap(&self) -> bool {
        true
    }

    /// Remap the row ids, creating a new remapped version of this index in `dest_store`
    async fn remap(
        &self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        let mut state = HashMap::new();

        for key in self.index_map.keys() {
            let bitmap = self.load_bitmap(key, None).await?;
            let remapped_bitmap =
                RowAddrTreeMap::from_iter(bitmap.row_addrs().unwrap().filter_map(|addr| {
                    let addr_as_u64 = u64::from(addr);
                    mapping
                        .get(&addr_as_u64)
                        .copied()
                        .unwrap_or(Some(addr_as_u64))
                }));
            state.insert(key.0.clone(), remapped_bitmap);
        }

        if !self.null_map.is_empty() {
            let remapped_null =
                RowAddrTreeMap::from_iter(self.null_map.row_addrs().unwrap().filter_map(|addr| {
                    let addr_as_u64 = u64::from(addr);
                    mapping
                        .get(&addr_as_u64)
                        .copied()
                        .unwrap_or(Some(addr_as_u64))
                }));
            state.insert(ScalarValue::try_from(&self.value_type)?, remapped_null);
        }

        BitmapIndexPlugin::write_bitmap_index(state, dest_store, &self.value_type).await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::BitmapIndexDetails::default())
                .unwrap(),
            index_version: BITMAP_INDEX_VERSION,
        })
    }

    /// Add the new data into the index, creating an updated version of the index in `dest_store`
    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        let mut state = HashMap::new();

        // Initialize builder with existing data
        for key in self.index_map.keys() {
            let bitmap = self.load_bitmap(key, None).await?;
            state.insert(key.0.clone(), (*bitmap).clone());
        }

        if !self.null_map.is_empty() {
            let ex_null = new_null_array(&self.value_type, 1);
            let ex_null = ScalarValue::try_from_array(ex_null.as_ref(), 0)?;
            state.insert(ex_null, (*self.null_map).clone());
        }

        BitmapIndexPlugin::do_train_bitmap_index(new_data, state, dest_store).await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::BitmapIndexDetails::default())
                .unwrap(),
            index_version: BITMAP_INDEX_VERSION,
        })
    }

    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria::only_new_data(TrainingCriteria::new(TrainingOrdering::None).with_row_id())
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        Ok(ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap))
    }
}

#[derive(Debug, Default)]
pub struct BitmapIndexPlugin;

impl BitmapIndexPlugin {
    fn get_batch_from_arrays(
        keys: Arc<dyn Array>,
        binary_bitmaps: Arc<dyn Array>,
    ) -> Result<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("keys", keys.data_type().clone(), true),
            Field::new("bitmaps", binary_bitmaps.data_type().clone(), true),
        ]));

        let columns = vec![keys, binary_bitmaps];

        Ok(RecordBatch::try_new(schema, columns)?)
    }

    async fn write_bitmap_index(
        state: HashMap<ScalarValue, RowAddrTreeMap>,
        index_store: &dyn IndexStore,
        value_type: &DataType,
    ) -> Result<()> {
        let num_bitmaps = state.len();
        let schema = Arc::new(Schema::new(vec![
            Field::new("keys", value_type.clone(), true),
            Field::new("bitmaps", DataType::Binary, true),
        ]));

        let mut bitmap_index_file = index_store
            .new_index_file(BITMAP_LOOKUP_NAME, schema)
            .await?;

        let mut cur_keys = Vec::new();
        let mut cur_bitmaps = Vec::new();
        let mut cur_bytes = 0;

        for (key, bitmap) in state.into_iter() {
            let mut bytes = Vec::new();
            bitmap.serialize_into(&mut bytes).unwrap();
            let bitmap_size = bytes.len();

            if cur_bytes + bitmap_size > MAX_BITMAP_ARRAY_LENGTH {
                let keys_array = ScalarValue::iter_to_array(cur_keys.clone().into_iter()).unwrap();
                let mut binary_builder = BinaryBuilder::new();
                for b in &cur_bitmaps {
                    binary_builder.append_value(b);
                }
                let bitmaps_array = Arc::new(binary_builder.finish()) as Arc<dyn Array>;

                let record_batch = Self::get_batch_from_arrays(keys_array, bitmaps_array)?;
                bitmap_index_file.write_record_batch(record_batch).await?;

                cur_keys.clear();
                cur_bitmaps.clear();
                cur_bytes = 0;
            }

            cur_keys.push(key);
            cur_bitmaps.push(bytes);
            cur_bytes += bitmap_size;
        }

        // Flush any remaining
        if !cur_keys.is_empty() {
            let keys_array = ScalarValue::iter_to_array(cur_keys).unwrap();
            let mut binary_builder = BinaryBuilder::new();
            for b in &cur_bitmaps {
                binary_builder.append_value(b);
            }
            let bitmaps_array = Arc::new(binary_builder.finish()) as Arc<dyn Array>;

            let record_batch = Self::get_batch_from_arrays(keys_array, bitmaps_array)?;
            bitmap_index_file.write_record_batch(record_batch).await?;
        }

        // Finish file with metadata that allows lightweight statistics reads
        let stats_json = serde_json::to_string(&BitmapStatistics { num_bitmaps }).map_err(|e| {
            Error::Internal {
                message: format!("failed to serialize bitmap statistics: {e}"),
                location: location!(),
            }
        })?;
        let mut metadata = HashMap::new();
        metadata.insert(INDEX_STATS_METADATA_KEY.to_string(), stats_json);

        bitmap_index_file.finish_with_metadata(metadata).await?;

        Ok(())
    }

    async fn do_train_bitmap_index(
        mut data_source: SendableRecordBatchStream,
        mut state: HashMap<ScalarValue, RowAddrTreeMap>,
        index_store: &dyn IndexStore,
    ) -> Result<()> {
        let value_type = data_source.schema().field(0).data_type().clone();
        while let Some(batch) = data_source.try_next().await? {
            let values = batch.column_by_name(VALUE_COLUMN_NAME).expect_ok()?;
            let row_ids = batch.column_by_name(ROW_ID).expect_ok()?;
            debug_assert_eq!(row_ids.data_type(), &DataType::UInt64);

            let row_id_column = row_ids.as_any().downcast_ref::<UInt64Array>().unwrap();

            for i in 0..values.len() {
                let row_id = row_id_column.value(i);
                let key = ScalarValue::try_from_array(values.as_ref(), i)?;
                state.entry(key.clone()).or_default().insert(row_id);
            }
        }

        Self::write_bitmap_index(state, index_store, &value_type).await
    }

    pub async fn train_bitmap_index(
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
    ) -> Result<()> {
        // mapping from item to list of the row ids where it is present
        let dictionary: HashMap<ScalarValue, RowAddrTreeMap> = HashMap::new();

        Self::do_train_bitmap_index(data, dictionary, index_store).await
    }
}

#[async_trait]
impl ScalarIndexPlugin for BitmapIndexPlugin {
    fn name(&self) -> &str {
        "Bitmap"
    }

    fn new_training_request(
        &self,
        _params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        if field.data_type().is_nested() {
            return Err(Error::InvalidInput {
                source: "A bitmap index can only be created on a non-nested field.".into(),
                location: location!(),
            });
        }
        Ok(Box::new(DefaultTrainingRequest::new(
            TrainingCriteria::new(TrainingOrdering::None).with_row_id(),
        )))
    }

    fn provides_exact_answer(&self) -> bool {
        true
    }

    fn version(&self) -> u32 {
        BITMAP_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        Some(Box::new(SargableQueryParser::new(index_name, false)))
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        _request: Box<dyn TrainingRequest>,
        fragment_ids: Option<Vec<u32>>,
    ) -> Result<CreatedIndex> {
        if fragment_ids.is_some() {
            return Err(Error::InvalidInput {
                source: "Bitmap index does not support fragment training".into(),
                location: location!(),
            });
        }

        Self::train_bitmap_index(data, index_store).await?;
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::BitmapIndexDetails::default())
                .unwrap(),
            index_version: BITMAP_INDEX_VERSION,
        })
    }

    /// Load an index from storage
    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        Ok(BitmapIndex::load(index_store, frag_reuse_index, cache).await? as Arc<dyn ScalarIndex>)
    }

    async fn load_statistics(
        &self,
        index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
    ) -> Result<Option<serde_json::Value>> {
        let reader = index_store.open_index_file(BITMAP_LOOKUP_NAME).await?;
        if let Some(value) = reader.schema().metadata.get(INDEX_STATS_METADATA_KEY) {
            let stats = serde_json::from_str(value).map_err(|e| Error::Internal {
                message: format!("failed to parse bitmap statistics metadata: {e}"),
                location: location!(),
            })?;
            Ok(Some(stats))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::metrics::NoOpMetricsCollector;
    use crate::scalar::lance_format::LanceIndexStore;
    use arrow_array::{RecordBatch, StringArray, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream;
    use lance_core::utils::{address::RowAddress, tempfile::TempObjDir};
    use lance_io::object_store::ObjectStore;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_bitmap_lazy_loading_and_cache() {
        // Create a temporary directory for the index
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Create test data with low cardinality column
        let colors = vec![
            "red", "blue", "green", "red", "yellow", "blue", "red", "green", "blue", "yellow",
            "red", "red", "blue", "green", "yellow",
        ];

        let row_ids = (0u64..15u64).collect::<Vec<_>>();

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Utf8, false),
            Field::new("_rowid", DataType::UInt64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(colors.clone())),
                Arc::new(UInt64Array::from(row_ids.clone())),
            ],
        )
        .unwrap();

        let stream = stream::once(async move { Ok(batch) });
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));

        // Train and write the bitmap index
        BitmapIndexPlugin::train_bitmap_index(stream, store.as_ref())
            .await
            .unwrap();

        // Create a cache with limited capacity
        let cache = LanceCache::with_capacity(1024 * 1024); // 1MB cache

        // Load the index (should only load metadata, not bitmaps)
        let index = BitmapIndex::load(store.clone(), None, &cache)
            .await
            .unwrap();

        assert_eq!(index.index_map.len(), 4); // 4 non-null unique values (red, blue, green, yellow)
        assert!(index.null_map.is_empty()); // No nulls in test data

        // Test 1: Search for "red"
        let query = SargableQuery::Equals(ScalarValue::Utf8(Some("red".to_string())));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Verify results
        let expected_red_rows = vec![0u64, 3, 6, 10, 11];
        if let SearchResult::Exact(row_ids) = result {
            let mut actual: Vec<u64> = row_ids.row_addrs().unwrap().map(|id| id.into()).collect();
            actual.sort();
            assert_eq!(actual, expected_red_rows);
        } else {
            panic!("Expected exact search result");
        }

        // Test 2: Search for "red" again - should hit cache
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        if let SearchResult::Exact(row_ids) = result {
            let mut actual: Vec<u64> = row_ids.row_addrs().unwrap().map(|id| id.into()).collect();
            actual.sort();
            assert_eq!(actual, expected_red_rows);
        }

        // Test 3: Range query
        let query = SargableQuery::Range(
            std::ops::Bound::Included(ScalarValue::Utf8(Some("blue".to_string()))),
            std::ops::Bound::Included(ScalarValue::Utf8(Some("green".to_string()))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        let expected_range_rows = vec![1u64, 2, 5, 7, 8, 12, 13];
        if let SearchResult::Exact(row_ids) = result {
            let mut actual: Vec<u64> = row_ids.row_addrs().unwrap().map(|id| id.into()).collect();
            actual.sort();
            assert_eq!(actual, expected_range_rows);
        }

        // Test 4: IsIn query
        let query = SargableQuery::IsIn(vec![
            ScalarValue::Utf8(Some("red".to_string())),
            ScalarValue::Utf8(Some("yellow".to_string())),
        ]);
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        let expected_in_rows = vec![0u64, 3, 4, 6, 9, 10, 11, 14];
        if let SearchResult::Exact(row_ids) = result {
            let mut actual: Vec<u64> = row_ids.row_addrs().unwrap().map(|id| id.into()).collect();
            actual.sort();
            assert_eq!(actual, expected_in_rows);
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_big_bitmap_index() {
        // WARNING: This test allocates a huge state to force overflow over int32 on BinaryArray
        // You must run it only on a machine with enough resources (or skip it normally).
        use super::{BitmapIndex, BITMAP_LOOKUP_NAME};
        use crate::scalar::lance_format::LanceIndexStore;
        use crate::scalar::IndexStore;
        use arrow_schema::DataType;
        use datafusion_common::ScalarValue;
        use lance_core::cache::LanceCache;
        use lance_core::utils::mask::RowAddrTreeMap;
        use lance_io::object_store::ObjectStore;
        use std::collections::HashMap;
        use std::sync::Arc;

        // Adjust these numbers so that:
        //     m * (serialized size per bitmap) > 2^31 bytes.
        //
        // For example, if we assume each bitmap serializes to ~1000 bytes,
        // you need m > 2.1e6.
        let m: u32 = 2_500_000;
        let per_bitmap_size = 1000; // assumed bytes per bitmap

        let mut state = HashMap::new();
        for i in 0..m {
            // Create a bitmap that contains, say, 1000 row IDs.
            let bitmap = RowAddrTreeMap::from_iter(0..per_bitmap_size);

            let key = ScalarValue::UInt32(Some(i));
            state.insert(key, bitmap);
        }

        // Create a temporary store.
        let tmpdir = TempObjDir::default();
        let test_store = LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        );

        // This call should never trigger a "byte array offset overflow" error since now the code supports
        // read by chunks
        let result =
            BitmapIndexPlugin::write_bitmap_index(state, &test_store, &DataType::UInt32).await;

        assert!(
            result.is_ok(),
            "Failed to write bitmap index: {:?}",
            result.err()
        );

        // Verify the index file exists
        let index_file = test_store.open_index_file(BITMAP_LOOKUP_NAME).await;
        assert!(
            index_file.is_ok(),
            "Failed to open index file: {:?}",
            index_file.err()
        );
        let index_file = index_file.unwrap();

        // Print stats about the index file
        tracing::info!(
            "Index file contains {} rows in total",
            index_file.num_rows()
        );

        // Load the index using BitmapIndex::load
        tracing::info!("Loading index from disk...");
        let loaded_index = BitmapIndex::load(Arc::new(test_store), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load bitmap index");

        // Verify the loaded index has the correct number of entries
        assert_eq!(
            loaded_index.index_map.len(),
            m as usize,
            "Loaded index has incorrect number of keys (expected {}, got {})",
            m,
            loaded_index.index_map.len()
        );

        // Manually verify specific keys without using search()
        let test_keys = [0, m / 2, m - 1]; // Beginning, middle, and end
        for &key_val in &test_keys {
            let key = OrderableScalarValue(ScalarValue::UInt32(Some(key_val)));
            // Load the bitmap for this key
            let bitmap = loaded_index
                .load_bitmap(&key, None)
                .await
                .unwrap_or_else(|_| panic!("Key {} should exist", key_val));

            // Convert RowAddrTreeMap to a vector for easier assertion
            let row_addrs: Vec<u64> = bitmap.row_addrs().unwrap().map(u64::from).collect();

            // Verify length
            assert_eq!(
                row_addrs.len(),
                per_bitmap_size as usize,
                "Bitmap for key {} has wrong size",
                key_val
            );

            // Verify first few and last few elements
            for i in 0..5.min(per_bitmap_size) {
                assert!(
                    row_addrs.contains(&i),
                    "Bitmap for key {} should contain row_id {}",
                    key_val,
                    i
                );
            }

            for i in (per_bitmap_size - 5)..per_bitmap_size {
                assert!(
                    row_addrs.contains(&i),
                    "Bitmap for key {} should contain row_id {}",
                    key_val,
                    i
                );
            }

            // Verify exact range
            let expected_range: Vec<u64> = (0..per_bitmap_size).collect();
            assert_eq!(
                row_addrs, expected_range,
                "Bitmap for key {} doesn't contain expected values",
                key_val
            );

            tracing::info!(
                "✓ Verified bitmap for key {}: {} rows as expected",
                key_val,
                row_addrs.len()
            );
        }

        tracing::info!("Test successful! Index properly contains {} keys", m);
    }

    #[tokio::test]
    async fn test_bitmap_prewarm() {
        // Create a temporary directory for the index
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Create test data with low cardinality
        let colors = vec![
            "red", "blue", "green", "red", "yellow", "blue", "red", "green", "blue", "yellow",
            "red", "red", "blue", "green", "yellow",
        ];

        let row_ids = (0u64..15u64).collect::<Vec<_>>();

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Utf8, false),
            Field::new("_rowid", DataType::UInt64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(colors.clone())),
                Arc::new(UInt64Array::from(row_ids.clone())),
            ],
        )
        .unwrap();

        let stream = stream::once(async move { Ok(batch) });
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));

        // Train and write the bitmap index
        BitmapIndexPlugin::train_bitmap_index(stream, store.as_ref())
            .await
            .unwrap();

        // Create a cache with metrics tracking
        let cache = LanceCache::with_capacity(1024 * 1024); // 1MB cache

        // Load the index (should only load metadata, not bitmaps)
        let index = BitmapIndex::load(store.clone(), None, &cache)
            .await
            .unwrap();

        // Verify no bitmaps are cached yet
        let cache_key_red = BitmapKey {
            value: OrderableScalarValue(ScalarValue::Utf8(Some("red".to_string()))),
        };
        let cache_key_blue = BitmapKey {
            value: OrderableScalarValue(ScalarValue::Utf8(Some("blue".to_string()))),
        };

        assert!(cache
            .get_with_key::<BitmapKey>(&cache_key_red)
            .await
            .is_none());
        assert!(cache
            .get_with_key::<BitmapKey>(&cache_key_blue)
            .await
            .is_none());

        // Call prewarm
        index.prewarm().await.unwrap();

        // Verify all bitmaps are now cached
        assert!(cache
            .get_with_key::<BitmapKey>(&cache_key_red)
            .await
            .is_some());
        assert!(cache
            .get_with_key::<BitmapKey>(&cache_key_blue)
            .await
            .is_some());

        // Verify cached bitmaps have correct content
        let cached_red = cache
            .get_with_key::<BitmapKey>(&cache_key_red)
            .await
            .unwrap();
        let red_rows: Vec<u64> = cached_red.row_addrs().unwrap().map(u64::from).collect();
        assert_eq!(red_rows, vec![0, 3, 6, 10, 11]);

        // Call prewarm again - should be idempotent
        index.prewarm().await.unwrap();

        // Verify cache still contains the same items
        let cached_red_2 = cache
            .get_with_key::<BitmapKey>(&cache_key_red)
            .await
            .unwrap();
        let red_rows_2: Vec<u64> = cached_red_2.row_addrs().unwrap().map(u64::from).collect();
        assert_eq!(red_rows_2, vec![0, 3, 6, 10, 11]);
    }

    #[tokio::test]
    async fn test_remap_bitmap_with_null() {
        use arrow_array::UInt32Array;

        // Create a temporary store.
        let tmpdir = TempObjDir::default();
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Create test data that simulates:
        // frag 1 - { 0: null, 1: null, 2: 1 }
        // frag 2 - { 0: 1, 1: 2, 2: 2 }
        // We'll create this data with specific row addresses
        let values = vec![
            None,       // row 0: null (will be at address (1,0))
            None,       // row 1: null (will be at address (1,1))
            Some(1u32), // row 2: 1    (will be at address (1,2))
            Some(1u32), // row 3: 1    (will be at address (2,0))
            Some(2u32), // row 4: 2    (will be at address (2,1))
            Some(2u32), // row 5: 2    (will be at address (2,2))
        ];

        // Create row IDs with specific fragment addresses
        let row_ids: Vec<u64> = vec![
            RowAddress::new_from_parts(1, 0).into(),
            RowAddress::new_from_parts(1, 1).into(),
            RowAddress::new_from_parts(1, 2).into(),
            RowAddress::new_from_parts(2, 0).into(),
            RowAddress::new_from_parts(2, 1).into(),
            RowAddress::new_from_parts(2, 2).into(),
        ];

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::UInt32, true),
            Field::new("_rowid", DataType::UInt64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(values)),
                Arc::new(UInt64Array::from(row_ids)),
            ],
        )
        .unwrap();

        let stream = stream::once(async move { Ok(batch) });
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));

        // Create the bitmap index
        BitmapIndexPlugin::train_bitmap_index(stream, test_store.as_ref())
            .await
            .unwrap();

        // Load the index
        let index = BitmapIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load bitmap index");

        // Verify initial state
        assert_eq!(index.index_map.len(), 2); // 2 non-null values (1 and 2)
        assert!(!index.null_map.is_empty()); // Should have null values

        // Create a remap that simulates compaction of frags 1 and 2 into frag 3
        let mut row_addr_map = HashMap::<u64, Option<u64>>::new();
        row_addr_map.insert(
            RowAddress::new_from_parts(1, 0).into(),
            Some(RowAddress::new_from_parts(3, 0).into()),
        );
        row_addr_map.insert(
            RowAddress::new_from_parts(1, 1).into(),
            Some(RowAddress::new_from_parts(3, 1).into()),
        );
        row_addr_map.insert(
            RowAddress::new_from_parts(1, 2).into(),
            Some(RowAddress::new_from_parts(3, 2).into()),
        );
        row_addr_map.insert(
            RowAddress::new_from_parts(2, 0).into(),
            Some(RowAddress::new_from_parts(3, 3).into()),
        );
        row_addr_map.insert(
            RowAddress::new_from_parts(2, 1).into(),
            Some(RowAddress::new_from_parts(3, 4).into()),
        );
        row_addr_map.insert(
            RowAddress::new_from_parts(2, 2).into(),
            Some(RowAddress::new_from_parts(3, 5).into()),
        );

        // Perform remap
        index
            .remap(&row_addr_map, test_store.as_ref())
            .await
            .unwrap();

        // Reload and check
        let reloaded_idx = BitmapIndex::load(test_store, None, &LanceCache::no_cache())
            .await
            .expect("Failed to load remapped bitmap index");

        // Verify the null bitmap was remapped correctly
        let expected_null_addrs: Vec<u64> = vec![
            RowAddress::new_from_parts(3, 0).into(),
            RowAddress::new_from_parts(3, 1).into(),
        ];
        let actual_null_addrs: Vec<u64> = reloaded_idx
            .null_map
            .row_addrs()
            .unwrap()
            .map(u64::from)
            .collect();
        assert_eq!(
            actual_null_addrs, expected_null_addrs,
            "Null bitmap not remapped correctly"
        );

        // Search for value 1 and verify remapped addresses
        let query = SargableQuery::Equals(ScalarValue::UInt32(Some(1)));
        let result = reloaded_idx
            .search(&query, &NoOpMetricsCollector)
            .await
            .unwrap();
        if let crate::scalar::SearchResult::Exact(row_ids) = result {
            let mut actual: Vec<u64> = row_ids.row_addrs().unwrap().map(u64::from).collect();
            actual.sort();
            let expected: Vec<u64> = vec![
                RowAddress::new_from_parts(3, 2).into(),
                RowAddress::new_from_parts(3, 3).into(),
            ];
            assert_eq!(actual, expected, "Value 1 bitmap not remapped correctly");
        }

        // Search for value 2 and verify remapped addresses
        let query = SargableQuery::Equals(ScalarValue::UInt32(Some(2)));
        let result = reloaded_idx
            .search(&query, &NoOpMetricsCollector)
            .await
            .unwrap();
        if let crate::scalar::SearchResult::Exact(row_ids) = result {
            let mut actual: Vec<u64> = row_ids.row_addrs().unwrap().map(u64::from).collect();
            actual.sort();
            let expected: Vec<u64> = vec![
                RowAddress::new_from_parts(3, 4).into(),
                RowAddress::new_from_parts(3, 5).into(),
            ];
            assert_eq!(actual, expected, "Value 2 bitmap not remapped correctly");
        }

        // Search for null values
        let query = SargableQuery::IsNull();
        let result = reloaded_idx
            .search(&query, &NoOpMetricsCollector)
            .await
            .unwrap();
        if let crate::scalar::SearchResult::Exact(row_ids) = result {
            let mut actual: Vec<u64> = row_ids.row_addrs().unwrap().map(u64::from).collect();
            actual.sort();
            assert_eq!(
                actual, expected_null_addrs,
                "Null search results not correct"
            );
        }
    }
}
