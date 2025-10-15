// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Geo Index
//!
//! Geo indices are spatial database structures for efficient spatial queries.
//! They enable efficient filtering by location-based predicates.
//!
//! ## Requirements
//!
//! Geo indices can only be created on fields with GeoArrow metadata. The field must:
//! - Be a Struct data type
//! - Have `ARROW:extension:name` metadata starting with `geoarrow.` (e.g., `geoarrow.point`, `geoarrow.polygon`)
//!
//! ## Query Support
//!
//! Geo indices are "inexact" filters - they can definitively exclude regions but may include
//! false positives that require rechecking.
//!

use crate::pbold;
use crate::scalar::bkd::{BKDTreeBuilder, BKDTreeLookup, point_in_bbox};
use crate::scalar::expression::{GeoQueryParser, ScalarQueryParser};
use crate::scalar::registry::{
    ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
};
use crate::scalar::{
    BuiltinIndexType, CreatedIndex, GeoQuery, ScalarIndexParams, UpdateCriteria,
};
use crate::Any;
use futures::TryStreamExt;
use lance_core::cache::{CacheKey, LanceCache, WeakLanceCache};
use lance_core::{ROW_ADDR, ROW_ID};
use serde::{Deserialize, Serialize};

use arrow_array::{Array, ArrayRef, Float64Array, RecordBatch, UInt32Array, UInt8Array};
use arrow_array::cast::AsArray;
use arrow_schema::{DataType, Field, Schema};
use datafusion::execution::SendableRecordBatchStream;
use std::{collections::HashMap, sync::Arc};

use super::{AnyQuery, IndexReader, IndexStore, MetricsCollector, ScalarIndex, SearchResult};
use crate::scalar::FragReuseIndex;
use crate::vector::VectorIndex;
use crate::{Index, IndexType};
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use lance_core::Result;
use lance_core::{utils::mask::RowIdTreeMap, Error};
use roaring::RoaringBitmap;
use snafu::location;

const BKD_TREE_FILENAME: &str = "bkd_tree.lance";
const LEAF_FILENAME_PREFIX: &str = "leaf_";
const GEO_INDEX_VERSION: u32 = 0;
const LEAF_SIZE_META_KEY: &str = "leaf_size";
const DEFAULT_LEAF_SIZE: u32 = 100;

fn leaf_filename(leaf_id: u32) -> String {
    format!("{}{}.lance", LEAF_FILENAME_PREFIX, leaf_id)
}

/// Lazy reader for BKD leaf file
#[derive(Clone)]
struct LazyIndexReader {
    index_reader: Arc<tokio::sync::Mutex<Option<Arc<dyn IndexReader>>>>,
    store: Arc<dyn IndexStore>,
    filename: String,
}

impl LazyIndexReader {
    fn new(store: Arc<dyn IndexStore>, filename: &str) -> Self {
        Self {
            index_reader: Arc::new(tokio::sync::Mutex::new(None)),
            store,
            filename: filename.to_string(),
        }
    }

    async fn get(&self) -> Result<Arc<dyn IndexReader>> {
        let mut reader = self.index_reader.lock().await;
        if reader.is_none() {
            let r = self.store.open_index_file(&self.filename).await?;
            *reader = Some(r);
        }
        Ok(reader.as_ref().unwrap().clone())
    }
}

/// Cache key for BKD leaf nodes
#[derive(Debug, Clone)]
struct BKDLeafKey {
    leaf_id: u32,
}

impl CacheKey for BKDLeafKey {
    type ValueType = CachedLeafData;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("bkd-leaf-{}", self.leaf_id).into()
    }
}

/// Cached leaf data
#[derive(Debug, Clone)]
struct CachedLeafData(RecordBatch);

impl DeepSizeOf for CachedLeafData {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        // Approximate size of RecordBatch
        self.0.get_array_memory_size()
    }
}

impl CachedLeafData {
    fn new(batch: RecordBatch) -> Self {
        Self(batch)
    }

    fn into_inner(self) -> RecordBatch {
        self.0
    }
}

/// Geo index
pub struct GeoIndex {
    data_type: DataType,
    store: Arc<dyn IndexStore>,
    fri: Option<Arc<FragReuseIndex>>,
    index_cache: WeakLanceCache,
    bkd_tree: Arc<BKDTreeLookup>,
    leaf_size: u32,
}

impl std::fmt::Debug for GeoIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeoIndex")
            .field("data_type", &self.data_type)
            .field("store", &self.store)
            .field("fri", &self.fri)
            .field("index_cache", &self.index_cache)
            .field("bkd_tree", &self.bkd_tree)
            .field("leaf_size", &self.leaf_size)
            .finish()
    }
}

impl DeepSizeOf for GeoIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.bkd_tree.deep_size_of_children(context) + self.store.deep_size_of_children(context)
    }
}

impl GeoIndex {
    /// Load the geo index from storage
    async fn load(
        store: Arc<dyn IndexStore>,
        fri: Option<Arc<FragReuseIndex>>,
        index_cache: &LanceCache,
    ) -> Result<Arc<Self>>
    where
        Self: Sized,
    {
        // Load BKD tree structure (inner nodes)
        let tree_file = store.open_index_file(BKD_TREE_FILENAME).await?;
        let tree_data = tree_file
            .read_range(0..tree_file.num_rows(), None)
            .await?;

        // Deserialize tree structure
        let bkd_tree = BKDTreeLookup::from_record_batch(tree_data)?;

        // Extract metadata
        let schema = tree_file.schema();
        let leaf_size = schema
            .metadata
            .get(LEAF_SIZE_META_KEY)
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_LEAF_SIZE);

        // Get data type from schema
        let data_type = schema.fields[0].data_type().clone();

        Ok(Arc::new(Self {
            data_type,
            store,
            fri,
            index_cache: WeakLanceCache::from(index_cache),
            bkd_tree: Arc::new(bkd_tree),
            leaf_size,
        }))
    }

    /// Load a specific leaf from storage (from leaf_{id}.lance file)
    async fn load_leaf(
        &self,
        leaf_id: u32,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        // Check cache first
        let cache_key = BKDLeafKey { leaf_id };
        let store = self.store.clone();

        let cached = self
            .index_cache
            .get_or_insert_with_key(cache_key, move || async move {
                metrics.record_part_load();
                let filename = leaf_filename(leaf_id);
                let reader = store.open_index_file(&filename).await?;
                // Read the entire leaf file
                let num_rows = reader.num_rows();
                let batch = reader.read_range(0..num_rows, None).await?;
                Ok(CachedLeafData::new(batch))
            })
            .await?;

        Ok(cached.as_ref().clone().into_inner())
    }

    /// Search a specific leaf for points within the query bbox
    async fn search_leaf(
        &self,
        leaf_id: u32,
        query_bbox: [f64; 4],
        metrics: &dyn MetricsCollector,
    ) -> Result<RowIdTreeMap> {
        let leaf_data = self.load_leaf(leaf_id, metrics).await?;

        println!(
            "🔍 Searching leaf {} with {} points, query_bbox: {:?}",
            leaf_id,
            leaf_data.num_rows(),
            query_bbox
        );

        // Filter points within this leaf
        let mut row_ids = RowIdTreeMap::new();
        let x_array = leaf_data
            .column(0)
            .as_primitive::<arrow_array::types::Float64Type>();
        let y_array = leaf_data
            .column(1)
            .as_primitive::<arrow_array::types::Float64Type>();
        let row_id_array = leaf_data
            .column(2)
            .as_primitive::<arrow_array::types::UInt64Type>();

        let mut matched_count = 0;
        for i in 0..leaf_data.num_rows() {
            let x = x_array.value(i);
            let y = y_array.value(i);
            let row_id = row_id_array.value(i);

            // Debug: dump all points in leaf 16 to find San Francisco
            if leaf_id == 16 && i < 10 {
                println!("  🔎 Leaf 16 point {}: ({}, {}) row_id={}", i, x, y, row_id);
            }

            if point_in_bbox(x, y, &query_bbox) {
                row_ids.insert(row_id);
                matched_count += 1;
                
                // Log first few matches for debugging
                if matched_count <= 3 {
                    println!(
                        "  ✅ Match {}: point({}, {}) -> row_id {}",
                        matched_count, x, y, row_id
                    );
                }
            }
        }

        println!(
            "📊 Leaf {} matched {} out of {} points",
            leaf_id,
            matched_count,
            leaf_data.num_rows()
        );

        Ok(row_ids)
    }
}

#[async_trait]
impl Index for GeoIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn VectorIndex>> {
        Err(Error::InvalidInput {
            source: "GeoIndex is not a vector index".into(),
            location: location!(),
        })
    }

    async fn prewarm(&self) -> Result<()> {
        Ok(())
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "type": "geo",
        }))
    }

    fn index_type(&self) -> IndexType {
        IndexType::Geo
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let frag_ids = RoaringBitmap::new();
        Ok(frag_ids)
    }
}

#[async_trait]
impl ScalarIndex for GeoIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let geo_query = query.as_any().downcast_ref::<GeoQuery>()
            .ok_or_else(|| Error::InvalidInput {
                source: "Geo index only supports GeoQuery".into(),
                location: location!(),
            })?;

        match geo_query {
            GeoQuery::Intersects(min_x, min_y, max_x, max_y) => {
                let query_bbox = [*min_x, *min_y, *max_x, *max_y];

                println!(
                    "\n🔍 Geo index search: st_intersects with bbox({}, {}, {}, {})",
                    min_x, min_y, max_x, max_y
                );

                // Step 1: Find intersecting leaves using in-memory tree traversal
                let leaf_ids = self.bkd_tree.find_intersecting_leaves(query_bbox);

                println!(
                    "📊 BKD tree traversal found {} intersecting leaves out of {} total leaves",
                    leaf_ids.len(),
                    self.bkd_tree.num_leaves
                );

                // Step 2: Lazy-load and filter each leaf
                let mut all_row_ids = RowIdTreeMap::new();

                for leaf_id in &leaf_ids {
                    let leaf_row_ids = self
                        .search_leaf(*leaf_id, query_bbox, metrics)
                        .await?;
                    // Collect row IDs from the leaf and add them to the result set
                    let row_ids: Option<Vec<u64>> = leaf_row_ids.row_ids()
                        .map(|iter| iter.map(|row_addr| u64::from(row_addr)).collect());
                    if let Some(row_ids) = row_ids {
                        all_row_ids.extend(row_ids);
                    }
                }

                println!(
                    "✅ Geo index searched {} leaves and returning {} row IDs\n",
                    leaf_ids.len(),
                    all_row_ids.len().unwrap_or(0)
                );

                // We return Exact because we already filtered points in search_leaf
                Ok(SearchResult::Exact(all_row_ids))
            }
        }
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
            source: "GeoIndex does not support remap".into(),
            location: location!(),
        })
    }

    /// Add the new data , creating an updated version of the index in `dest_store`
    async fn update(
        &self,
        _new_data: SendableRecordBatchStream,
        _dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        Err(Error::InvalidInput {
            source: "GeoIndex does not support update".into(),
            location: location!(),
        })
    }

    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria::only_new_data(
            TrainingCriteria::new(TrainingOrdering::Addresses).with_row_addr(),
        )
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        let params = serde_json::to_value(GeoIndexBuilderParams::default())?;
        Ok(ScalarIndexParams::for_builtin(BuiltinIndexType::Geo).with_params(&params))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoIndexBuilderParams {
    #[serde(default = "default_leaf_size")]
    pub leaf_size: u32,
}

fn default_leaf_size() -> u32 {
    DEFAULT_LEAF_SIZE
}

impl Default for GeoIndexBuilderParams {
    fn default() -> Self {
        Self {
            leaf_size: default_leaf_size(),
        }
    }
}

impl GeoIndexBuilderParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_leaf_size(mut self, leaf_size: u32) -> Self {
        self.leaf_size = leaf_size;
        self
    }
}

// A builder for geo index
pub struct GeoIndexBuilder {
    options: GeoIndexBuilderParams,
    items_type: DataType,
    // Accumulated points: (x, y, row_id)
    points: Vec<(f64, f64, u64)>,
}

impl GeoIndexBuilder {
    pub fn try_new(options: GeoIndexBuilderParams, items_type: DataType) -> Result<Self> {
        Ok(Self {
            options,
            items_type,
            points: Vec::new(),
        })
    }

    pub async fn train(&mut self, batches_source: SendableRecordBatchStream) -> Result<()> {
        assert!(batches_source.schema().field_with_name(ROW_ADDR).is_ok());

        let mut batches_source = batches_source;

        while let Some(batch) = batches_source.try_next().await? {
            // Extract GeoArrow point coordinates
            let geom_array = batch.column(0).as_any().downcast_ref::<arrow_array::StructArray>()
                .ok_or_else(|| Error::InvalidInput {
                    source: "Expected Struct array for GeoArrow data".into(),
                    location: location!(),
                })?;

            let x_array = geom_array
                .column(0)
                .as_primitive::<arrow_array::types::Float64Type>();
            let y_array = geom_array
                .column(1)
                .as_primitive::<arrow_array::types::Float64Type>();
            let row_ids = batch
                .column_by_name(ROW_ADDR)
                .unwrap()
                .as_primitive::<arrow_array::types::UInt64Type>();

            for i in 0..batch.num_rows() {
                self.points.push((
                    x_array.value(i),
                    y_array.value(i),
                    row_ids.value(i),
                ));
            }
        }

        log::debug!("Accumulated {} points for BKD tree", self.points.len());

        Ok(())
    }

    pub async fn write_index(mut self, index_store: &dyn IndexStore) -> Result<()> {
        if self.points.is_empty() {
            // Write empty index files
            self.write_empty_index(index_store).await?;
            return Ok(());
        }

        // Build BKD tree
        let (tree_nodes, leaf_batches) = self.build_bkd_tree()?;

        // Write tree structure
        let tree_batch = self.serialize_tree_nodes(&tree_nodes)?;
        let mut tree_file = index_store
            .new_index_file(BKD_TREE_FILENAME, tree_batch.schema())
            .await?;
        tree_file.write_record_batch(tree_batch).await?;
        tree_file
            .finish_with_metadata(HashMap::from([(
                LEAF_SIZE_META_KEY.to_string(),
                self.options.leaf_size.to_string(),
            )]))
            .await?;

        // Write each leaf to a separate file
        let leaf_schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        
        println!("📝 Writing {} leaf files", leaf_batches.len());
        for (leaf_id, leaf_batch) in leaf_batches.iter().enumerate() {
            let filename = leaf_filename(leaf_id as u32);
            println!("  Writing {}: {} rows", filename, leaf_batch.num_rows());
            let mut leaf_file = index_store
                .new_index_file(&filename, leaf_schema.clone())
                .await?;
            leaf_file.write_record_batch(leaf_batch.clone()).await?;
            leaf_file.finish().await?;
        }
        println!("✅ Finished writing {} leaf files\n", leaf_batches.len());

        log::debug!(
            "Wrote BKD tree with {} nodes",
            tree_nodes.len()
        );

        Ok(())
    }

    async fn write_empty_index(&self, index_store: &dyn IndexStore) -> Result<()> {
        // Write empty tree file
        let tree_schema = Arc::new(Schema::new(vec![
            Field::new("min_x", DataType::Float64, false),
            Field::new("min_y", DataType::Float64, false),
            Field::new("max_x", DataType::Float64, false),
            Field::new("max_y", DataType::Float64, false),
            Field::new("split_dim", DataType::UInt8, false),
            Field::new("split_value", DataType::Float64, false),
            Field::new("left_child", DataType::UInt32, true),
            Field::new("right_child", DataType::UInt32, true),
            Field::new("leaf_id", DataType::UInt32, true),
        ]));

        let empty_batch = RecordBatch::new_empty(tree_schema);
        let mut tree_file = index_store
            .new_index_file(BKD_TREE_FILENAME, empty_batch.schema())
            .await?;
        tree_file.write_record_batch(empty_batch).await?;
        tree_file.finish().await?;

        // No leaf files needed for empty index

        Ok(())
    }

    fn serialize_tree_nodes(&self, nodes: &[crate::scalar::bkd::BKDNode]) -> Result<RecordBatch> {
        let mut min_x_vals = Vec::with_capacity(nodes.len());
        let mut min_y_vals = Vec::with_capacity(nodes.len());
        let mut max_x_vals = Vec::with_capacity(nodes.len());
        let mut max_y_vals = Vec::with_capacity(nodes.len());
        let mut split_dim_vals = Vec::with_capacity(nodes.len());
        let mut split_value_vals = Vec::with_capacity(nodes.len());
        let mut left_child_vals = Vec::with_capacity(nodes.len());
        let mut right_child_vals = Vec::with_capacity(nodes.len());
        let mut leaf_id_vals = Vec::with_capacity(nodes.len());

        for node in nodes {
            min_x_vals.push(node.bounds[0]);
            min_y_vals.push(node.bounds[1]);
            max_x_vals.push(node.bounds[2]);
            max_y_vals.push(node.bounds[3]);
            split_dim_vals.push(node.split_dim);
            split_value_vals.push(node.split_value);
            left_child_vals.push(node.left_child);
            right_child_vals.push(node.right_child);
            leaf_id_vals.push(node.leaf_id);
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("min_x", DataType::Float64, false),
            Field::new("min_y", DataType::Float64, false),
            Field::new("max_x", DataType::Float64, false),
            Field::new("max_y", DataType::Float64, false),
            Field::new("split_dim", DataType::UInt8, false),
            Field::new("split_value", DataType::Float64, false),
            Field::new("left_child", DataType::UInt32, true),
            Field::new("right_child", DataType::UInt32, true),
            Field::new("leaf_id", DataType::UInt32, true),
        ]));

        let columns: Vec<ArrayRef> = vec![
            Arc::new(Float64Array::from(min_x_vals)),
            Arc::new(Float64Array::from(min_y_vals)),
            Arc::new(Float64Array::from(max_x_vals)),
            Arc::new(Float64Array::from(max_y_vals)),
            Arc::new(UInt8Array::from(split_dim_vals)),
            Arc::new(Float64Array::from(split_value_vals)),
            Arc::new(UInt32Array::from(left_child_vals)),
            Arc::new(UInt32Array::from(right_child_vals)),
            Arc::new(UInt32Array::from(leaf_id_vals)),
        ];

        Ok(RecordBatch::try_new(schema, columns)?)
    }

    // Build BKD tree using the BKDTreeBuilder
    fn build_bkd_tree(&mut self) -> Result<(Vec<crate::scalar::bkd::BKDNode>, Vec<RecordBatch>)> {
        let builder = BKDTreeBuilder::new(self.options.leaf_size as usize);
        builder.build(&mut self.points)
    }
}

#[derive(Debug, Default)]
pub struct GeoIndexPlugin;

impl GeoIndexPlugin {
    async fn train_geo_index(
        batches_source: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        options: Option<GeoIndexBuilderParams>,
    ) -> Result<()> {
        let value_type = batches_source.schema().field(0).data_type().clone();

        let mut builder = GeoIndexBuilder::try_new(options.unwrap_or_default(), value_type)?;

        builder.train(batches_source).await?;

        builder.write_index(index_store).await?;
        Ok(())
    }
}

pub struct GeoIndexTrainingRequest {
    pub params: GeoIndexBuilderParams,
    pub criteria: TrainingCriteria,
}

impl GeoIndexTrainingRequest {
    pub fn new(params: GeoIndexBuilderParams) -> Self {
        Self {
            params,
            criteria: TrainingCriteria::new(TrainingOrdering::Addresses).with_row_addr(),
        }
    }
}

impl TrainingRequest for GeoIndexTrainingRequest {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

#[async_trait]
impl ScalarIndexPlugin for GeoIndexPlugin {
    fn new_training_request(
        &self,
        params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        // Check that the field is a Struct type
        if !matches!(field.data_type(), DataType::Struct(_)) {
            return Err(Error::InvalidInput {
                source: "A geo index can only be created on a Struct field.".into(),
                location: location!(),
            });
        }

        // Check for GeoArrow metadata
        let is_geoarrow = field
            .metadata()
            .get("ARROW:extension:name")
            .map(|name| name.starts_with("geoarrow."))
            .unwrap_or(false);

        if !is_geoarrow {
            return Err(Error::InvalidInput {
                source: format!(
                    "Geo index requires GeoArrow metadata on field '{}'. \
                     The field must have 'ARROW:extension:name' metadata starting with 'geoarrow.'",
                    field.name()
                )
                .into(),
                location: location!(),
            });
        }

        let params = serde_json::from_str::<GeoIndexBuilderParams>(params)?;

        Ok(Box::new(GeoIndexTrainingRequest::new(params)))
    }

    fn provides_exact_answer(&self) -> bool {
        true // We do exact point-in-bbox filtering in search_leaf
    }

    fn version(&self) -> u32 {
        GEO_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        Some(Box::new(GeoQueryParser::new(index_name)))
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
                source: "Geo index does not support fragment training".into(),
                location: location!(),
            });
        }

        let request = (request as Box<dyn std::any::Any>)
            .downcast::<GeoIndexTrainingRequest>()
            .map_err(|_| Error::InvalidInput {
                source: "must provide training request created by new_training_request".into(),
                location: location!(),
            })?;
        Self::train_geo_index(data, index_store, Some(request.params)).await?;
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::GeoIndexDetails::default())
                .unwrap(),
            index_version: GEO_INDEX_VERSION,
        })
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        Ok(GeoIndex::load(index_store, frag_reuse_index, cache).await? as Arc<dyn ScalarIndex>)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::RecordBatch;
    use arrow_schema::{DataType, Field, Fields, Schema};
    use datafusion::execution::SendableRecordBatchStream;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream;
    use lance_core::cache::LanceCache;
    use lance_core::utils::tempfile::TempObjDir;
    use lance_core::ROW_ADDR;
    use lance_io::object_store::ObjectStore;

    use crate::scalar::lance_format::LanceIndexStore;

    #[tokio::test]
    async fn test_empty_geo_index() {
        let tmpdir = TempObjDir::default();
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let data = arrow_array::StructArray::from(vec![]);
        let row_ids = arrow_array::UInt64Array::from(Vec::<u64>::new());
        let fields: Fields = Vec::<Field>::new().into();
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Struct(fields), false),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let data =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();

        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        GeoIndexPlugin::train_geo_index(data_stream, test_store.as_ref(), None)
            .await
            .unwrap();

        // Read the index file back and check its contents
        let _index = GeoIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load GeoIndex");
    }
}

