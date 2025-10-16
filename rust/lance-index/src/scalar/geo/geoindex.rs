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
use super::bkd::{self, BKDTreeBuilder, BKDTreeLookup, point_in_bbox, BKDNode, BKDLeafNode};
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

use arrow_array::{ArrayRef, Float64Array, RecordBatch, UInt32Array, UInt8Array};
use arrow_array::cast::AsArray;
use arrow_schema::{DataType, Field, Schema};
use datafusion::execution::SendableRecordBatchStream;
use std::{collections::HashMap, sync::Arc};

use crate::scalar::{AnyQuery, IndexReader, IndexStore, MetricsCollector, ScalarIndex, SearchResult};
use crate::scalar::FragReuseIndex;
use crate::vector::VectorIndex;
use crate::{Index, IndexType};
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use lance_core::Result;
use lance_core::{utils::mask::RowIdTreeMap, Error};
use roaring::RoaringBitmap;
use snafu::location;

const BKD_TREE_INNER_FILENAME: &str = "bkd_tree_inner.lance";
const BKD_TREE_LEAF_FILENAME: &str = "bkd_tree_leaf.lance";
const LEAF_GROUP_PREFIX: &str = "leaf_group_";
const DEFAULT_BATCHES_PER_LEAF_FILE: u32 = 5; // Default number of leaf batches per file
const GEO_INDEX_VERSION: u32 = 0;
const MAX_POINTS_PER_LEAF_META_KEY: &str = "max_points_per_leaf";
const BATCHES_PER_FILE_META_KEY: &str = "batches_per_file";
const DEFAULT_MAX_POINTS_PER_LEAF: u32 = 100; // for test
// const DEFAULT_MAX_POINTS_PER_LEAF: u32 = 1024; // for production

/// Get the file name for a leaf group
fn leaf_group_filename(group_id: u32) -> String {
    format!("{}{}.lance", LEAF_GROUP_PREFIX, group_id)
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
    max_points_per_leaf: u32,
}

impl std::fmt::Debug for GeoIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeoIndex")
            .field("data_type", &self.data_type)
            .field("store", &self.store)
            .field("fri", &self.fri)
            .field("index_cache", &self.index_cache)
            .field("bkd_tree", &self.bkd_tree)
            .field("max_points_per_leaf", &self.max_points_per_leaf)
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
    pub async fn load(
        store: Arc<dyn IndexStore>,
        fri: Option<Arc<FragReuseIndex>>,
        index_cache: &LanceCache,
    ) -> Result<Arc<Self>>
    where
        Self: Sized,
    {
        // Load inner nodes
        let inner_file = store.open_index_file(BKD_TREE_INNER_FILENAME).await?;
        let inner_data = inner_file
            .read_range(0..inner_file.num_rows(), None)
            .await?;

        // Load leaf metadata
        let leaf_file = store.open_index_file(BKD_TREE_LEAF_FILENAME).await?;
        let leaf_data = leaf_file
            .read_range(0..leaf_file.num_rows(), None)
            .await?;

        // Deserialize tree structure from both files
        let bkd_tree = BKDTreeLookup::from_record_batches(inner_data, leaf_data)?;

        // Extract metadata from inner file
        let schema = inner_file.schema();
        let max_points_per_leaf = schema
            .metadata
            .get(MAX_POINTS_PER_LEAF_META_KEY)
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MAX_POINTS_PER_LEAF);

        // Get data type from schema
        let data_type = schema.fields[0].data_type().clone();

        Ok(Arc::new(Self {
            data_type,
            store,
            fri,
            index_cache: WeakLanceCache::from(index_cache),
            bkd_tree: Arc::new(bkd_tree),
            max_points_per_leaf,
        }))
    }

    /// Load a specific leaf from storage
    async fn load_leaf(
        &self,
        leaf: &BKDLeafNode,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        let file_id = leaf.file_id;
        let row_offset = leaf.row_offset;
        let num_rows = leaf.num_rows;
        
        // Use (file_id, row_offset) as cache key
        // Combine file_id and row_offset into a single u32 (file_id should be small)
        let cache_key = BKDLeafKey { leaf_id: file_id * 100_000 + (row_offset as u32) };
        let store = self.store.clone();

        let cached = self
            .index_cache
            .get_or_insert_with_key(cache_key, move || async move {
                metrics.record_part_load();
                
                let filename = leaf_group_filename(file_id);
                
                // Open the leaf group file and read the specific row range
                let reader = store.open_index_file(&filename).await?;
                let batch = reader.read_range(
                    row_offset as usize..(row_offset + num_rows) as usize,
                    None
                ).await?;
                
                Ok(CachedLeafData::new(batch))
            })
            .await?;

        Ok(cached.as_ref().clone().into_inner())
    }

    /// Get the number of leaves in the BKD tree (useful for benchmarking)
    pub fn num_leaves(&self) -> usize {
        self.bkd_tree.num_leaves as usize
    }

    /// Search all leaves without using BKD tree pruning (useful for benchmarking)
    pub async fn search_all_leaves(
        &self,
        query_bbox: [f64; 4],
        metrics: &dyn MetricsCollector,
    ) -> Result<RowIdTreeMap> {
        let mut all_row_ids = RowIdTreeMap::new();
        
        // Iterate through all nodes and search every leaf
        for node in self.bkd_tree.nodes.iter() {
            if let BKDNode::Leaf(leaf) = node {
                let leaf_row_ids = self.search_leaf(leaf, query_bbox, metrics).await?;
                let row_ids: Option<Vec<u64>> = leaf_row_ids.row_ids()
                    .map(|iter| iter.map(|row_addr| u64::from(row_addr)).collect());
                if let Some(row_ids) = row_ids {
                    all_row_ids.extend(row_ids);
                }
            }
        }
        
        Ok(all_row_ids)
    }

    /// Search a specific leaf for points within the query bbox
    async fn search_leaf(
        &self,
        leaf: &BKDLeafNode,
        query_bbox: [f64; 4],
        metrics: &dyn MetricsCollector,
    ) -> Result<RowIdTreeMap> {
        let leaf_data = self.load_leaf(leaf, metrics).await?;

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

        for i in 0..leaf_data.num_rows() {
            let x = x_array.value(i);
            let y = y_array.value(i);
            let row_id = row_id_array.value(i);

            if point_in_bbox(x, y, &query_bbox) {
                row_ids.insert(row_id);
            }
        }

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
        // No-op: geo index uses lazy loading
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

                // Step 1: Find intersecting leaves using in-memory tree traversal
                let leaves = self.bkd_tree.find_intersecting_leaves(query_bbox)?;

                // Step 2: Lazy-load and filter each leaf
                let mut all_row_ids = RowIdTreeMap::new();

                for leaf_node in &leaves {
                    let leaf_row_ids = self
                        .search_leaf(leaf_node, query_bbox, metrics)
                        .await?;
                    // Collect row IDs from the leaf and add them to the result set
                    let row_ids: Option<Vec<u64>> = leaf_row_ids.row_ids()
                        .map(|iter| iter.map(|row_addr| u64::from(row_addr)).collect());
                    if let Some(row_ids) = row_ids {
                        all_row_ids.extend(row_ids);
                    }
                }

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
    #[serde(default = "default_max_points_per_leaf")]
    pub max_points_per_leaf: u32,
    #[serde(default = "default_batches_per_file")]
    pub batches_per_file: u32,
}

fn default_max_points_per_leaf() -> u32 {
    DEFAULT_MAX_POINTS_PER_LEAF
}

fn default_batches_per_file() -> u32 {
    DEFAULT_BATCHES_PER_LEAF_FILE
}

impl Default for GeoIndexBuilderParams {
    fn default() -> Self {
        Self {
            max_points_per_leaf: default_max_points_per_leaf(),
            batches_per_file: default_batches_per_file(),
        }
    }
}

impl GeoIndexBuilderParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_points_per_leaf(mut self, max_points_per_leaf: u32) -> Self {
        self.max_points_per_leaf = max_points_per_leaf;
        self
    }
}

// A builder for geo index
pub struct GeoIndexBuilder {
    options: GeoIndexBuilderParams,
    items_type: DataType,
    // Accumulated points: (x, y, row_id)
    pub points: Vec<(f64, f64, u64)>,
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
            return Ok(());
        }

        // Build BKD tree
        let (tree_nodes, leaf_batches) = self.build_bkd_tree()?;

        // Write tree structure to separate inner and leaf files
        let (inner_batch, leaf_metadata_batch) = self.serialize_tree_nodes(&tree_nodes)?;
        
        // Write inner nodes
        let mut inner_file = index_store
            .new_index_file(BKD_TREE_INNER_FILENAME, inner_batch.schema())
            .await?;
        inner_file.write_record_batch(inner_batch).await?;
        inner_file
            .finish_with_metadata(HashMap::from([
                (MAX_POINTS_PER_LEAF_META_KEY.to_string(), self.options.max_points_per_leaf.to_string()),
                (BATCHES_PER_FILE_META_KEY.to_string(), self.options.batches_per_file.to_string()),
            ]))
            .await?;

        // Write leaf metadata
        let mut leaf_meta_file = index_store
            .new_index_file(BKD_TREE_LEAF_FILENAME, leaf_metadata_batch.schema())
            .await?;
        leaf_meta_file.write_record_batch(leaf_metadata_batch).await?;
        leaf_meta_file.finish().await?;

        // Write actual leaf data grouped into files (multiple batches per file)
        let leaf_schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        
        let batches_per_file = self.options.batches_per_file;
        let num_groups = (leaf_batches.len() as u32 + batches_per_file - 1) / batches_per_file;
        
        for group_id in 0..num_groups {
            let start_idx = (group_id * batches_per_file) as usize;
            let end_idx = ((group_id + 1) * batches_per_file).min(leaf_batches.len() as u32) as usize;
            let group_batches = &leaf_batches[start_idx..end_idx];
            
            let filename = leaf_group_filename(group_id);
            
            let mut leaf_file = index_store
                .new_index_file(&filename, leaf_schema.clone())
                .await?;
                
            for leaf_batch in group_batches.iter() {
                leaf_file.write_record_batch(leaf_batch.clone()).await?;
            }
            
            leaf_file.finish().await?;
        }

        log::debug!(
            "Wrote BKD tree with {} nodes",
            tree_nodes.len()
        );

        Ok(())
    }

    // Serialize tree nodes to separate inner and leaf RecordBatches
    fn serialize_tree_nodes(&self, nodes: &[BKDNode]) -> Result<(RecordBatch, RecordBatch)> {
        
        // Separate inner and leaf nodes with their indices
        let mut inner_nodes = Vec::new();
        let mut leaf_nodes = Vec::new();
        
        for (idx, node) in nodes.iter().enumerate() {
            match node {
                BKDNode::Inner(_) => inner_nodes.push((idx as u32, node)),
                BKDNode::Leaf(_) => leaf_nodes.push((idx as u32, node)),
            }
        }
        
        // Serialize inner nodes
        let inner_batch = Self::serialize_inner_nodes(&inner_nodes)?;
        
        // Serialize leaf nodes
        let leaf_batch = Self::serialize_leaf_nodes(&leaf_nodes)?;
        
        Ok((inner_batch, leaf_batch))
    }
    
    fn serialize_inner_nodes(nodes: &[(u32, &BKDNode)]) -> Result<RecordBatch> {
        
        let mut node_id_vals = Vec::with_capacity(nodes.len());
        let mut min_x_vals = Vec::with_capacity(nodes.len());
        let mut min_y_vals = Vec::with_capacity(nodes.len());
        let mut max_x_vals = Vec::with_capacity(nodes.len());
        let mut max_y_vals = Vec::with_capacity(nodes.len());
        let mut split_dim_vals = Vec::with_capacity(nodes.len());
        let mut split_value_vals = Vec::with_capacity(nodes.len());
        let mut left_child_vals = Vec::with_capacity(nodes.len());
        let mut right_child_vals = Vec::with_capacity(nodes.len());

        for (idx, node) in nodes {
            if let BKDNode::Inner(inner) = node {
                node_id_vals.push(*idx);
                min_x_vals.push(inner.bounds[0]);
                min_y_vals.push(inner.bounds[1]);
                max_x_vals.push(inner.bounds[2]);
                max_y_vals.push(inner.bounds[3]);
                split_dim_vals.push(inner.split_dim);
                split_value_vals.push(inner.split_value);
                left_child_vals.push(inner.left_child);
                right_child_vals.push(inner.right_child);
            }
        }

        let schema = bkd::inner_node_schema();

        let columns: Vec<ArrayRef> = vec![
            Arc::new(UInt32Array::from(node_id_vals)),
            Arc::new(Float64Array::from(min_x_vals)),
            Arc::new(Float64Array::from(min_y_vals)),
            Arc::new(Float64Array::from(max_x_vals)),
            Arc::new(Float64Array::from(max_y_vals)),
            Arc::new(UInt8Array::from(split_dim_vals)),
            Arc::new(Float64Array::from(split_value_vals)),
            Arc::new(UInt32Array::from(left_child_vals)),
            Arc::new(UInt32Array::from(right_child_vals)),
        ];

        Ok(RecordBatch::try_new(schema, columns)?)
    }
    
    fn serialize_leaf_nodes(nodes: &[(u32, &BKDNode)]) -> Result<RecordBatch> {
        use arrow_array::UInt64Array;
        
        let mut node_id_vals = Vec::with_capacity(nodes.len());
        let mut min_x_vals = Vec::with_capacity(nodes.len());
        let mut min_y_vals = Vec::with_capacity(nodes.len());
        let mut max_x_vals = Vec::with_capacity(nodes.len());
        let mut max_y_vals = Vec::with_capacity(nodes.len());
        let mut file_id_vals = Vec::with_capacity(nodes.len());
        let mut row_offset_vals = Vec::with_capacity(nodes.len());
        let mut num_rows_vals = Vec::with_capacity(nodes.len());

        for (idx, node) in nodes {
            if let BKDNode::Leaf(leaf) = node {
                node_id_vals.push(*idx);
                min_x_vals.push(leaf.bounds[0]);
                min_y_vals.push(leaf.bounds[1]);
                max_x_vals.push(leaf.bounds[2]);
                max_y_vals.push(leaf.bounds[3]);
                file_id_vals.push(leaf.file_id);
                row_offset_vals.push(leaf.row_offset);
                num_rows_vals.push(leaf.num_rows);
            }
        }

        let schema = bkd::leaf_node_schema();

        let columns: Vec<ArrayRef> = vec![
            Arc::new(UInt32Array::from(node_id_vals)),
            Arc::new(Float64Array::from(min_x_vals)),
            Arc::new(Float64Array::from(min_y_vals)),
            Arc::new(Float64Array::from(max_x_vals)),
            Arc::new(Float64Array::from(max_y_vals)),
            Arc::new(UInt32Array::from(file_id_vals)),
            Arc::new(UInt64Array::from(row_offset_vals)),
            Arc::new(UInt64Array::from(num_rows_vals)),
        ];

        Ok(RecordBatch::try_new(schema, columns)?)
    }

    // Build BKD tree using the BKDTreeBuilder
    fn build_bkd_tree(&mut self) -> Result<(Vec<BKDNode>, Vec<RecordBatch>)> {
        let builder = BKDTreeBuilder::new(self.options.max_points_per_leaf as usize);
        builder.build(&mut self.points, self.options.batches_per_file)
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

    use arrow_schema::{DataType, Field};
    use lance_core::cache::LanceCache;
    use lance_core::utils::tempfile::TempObjDir;
    use lance_io::object_store::ObjectStore;

    use crate::scalar::lance_format::LanceIndexStore;

    fn create_test_store() -> Arc<LanceIndexStore> {
        let tmpdir = TempObjDir::default();
    
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        return test_store;
    }

    /// Validates the BKD tree structure recursively
    async fn validate_bkd_tree(index: &GeoIndex) -> Result<()> {
        use crate::metrics::NoOpMetricsCollector;
        
        let tree = &index.bkd_tree;
        
        // Verify root exists
        if tree.nodes.is_empty() {
            return Err(Error::InvalidInput {
                source: "BKD tree has no nodes".into(),
                location: location!(),
            });
        }

        let root_id = tree.root_id as usize;
        if root_id >= tree.nodes.len() {
            return Err(Error::InvalidInput {
                source: format!("Root node id {} out of bounds (tree has {} nodes)", root_id, tree.nodes.len()).into(),
                location: location!(),
            });
        }

        // Count leaves and validate structure
        let mut visited = vec![false; tree.nodes.len()];
        let mut leaf_count = 0u32;
        
        // Start with None for expected_split_dim - root can use any dimension
        // (typically starts with dimension 0, but we don't enforce it)
        let metrics = NoOpMetricsCollector {};
        validate_node_recursive(index, tree, root_id as u32, &mut visited, &mut leaf_count, None, None, &metrics).await?;

        // Verify all leaves were counted
        if leaf_count != tree.num_leaves {
            return Err(Error::InvalidInput {
                source: format!("Leaf count mismatch: found {} leaves but tree claims {}", leaf_count, tree.num_leaves).into(),
                location: location!(),
            });
        }

        // Check for orphaned nodes (nodes not reachable from root)
        for (idx, was_visited) in visited.iter().enumerate() {
            if !was_visited {
                return Err(Error::InvalidInput {
                    source: format!("Node {} is orphaned (not reachable from root)", idx).into(),
                    location: location!(),
                });
            }
        }

        Ok(())
    }

    /// Helper function to recursively validate a node and its descendants
    async fn validate_node_recursive(
        index: &GeoIndex,
        tree: &BKDTreeLookup,
        node_id: u32,
        visited: &mut Vec<bool>,
        leaf_count: &mut u32,
        parent_bounds: Option<[f64; 4]>,
        _parent_split: Option<(u8, f64, bool)>, // (split_dim, split_value, is_left_child) from parent
        metrics: &dyn MetricsCollector,
    ) -> Result<()> {
        let node_idx = node_id as usize;
        
        // Check node exists
        if node_idx >= tree.nodes.len() {
            return Err(Error::InvalidInput {
                source: format!("Node id {} out of bounds (tree has {} nodes)", node_id, tree.nodes.len()).into(),
                location: location!(),
            });
        }

        // Mark as visited
        if visited[node_idx] {
            return Err(Error::InvalidInput {
                source: format!("Node {} visited multiple times (cycle detected)", node_id).into(),
                location: location!(),
            });
        }
        visited[node_idx] = true;

        let node = &tree.nodes[node_idx];
        let bounds = node.bounds();

        // Validate bounds are well-formed
        if bounds[0] > bounds[2] || bounds[1] > bounds[3] {
            return Err(Error::InvalidInput {
                source: format!("Node {} has invalid bounds: [{}, {}, {}, {}]", 
                    node_id, bounds[0], bounds[1], bounds[2], bounds[3]).into(),
                location: location!(),
            });
        }

        // Verify child bounds are within parent bounds
        if let Some(parent_bounds) = parent_bounds {
            if bounds[0] < parent_bounds[0] || bounds[1] < parent_bounds[1] ||
               bounds[2] > parent_bounds[2] || bounds[3] > parent_bounds[3] {
                return Err(Error::InvalidInput {
                    source: format!(
                        "Node {} bounds [{}, {}, {}, {}] exceed parent bounds [{}, {}, {}, {}]",
                        node_id, bounds[0], bounds[1], bounds[2], bounds[3],
                        parent_bounds[0], parent_bounds[1], parent_bounds[2], parent_bounds[3]
                    ).into(),
                    location: location!(),
                });
            }
        }

        match node {
            BKDNode::Inner(inner) => {
                // Validate split dimension
                if inner.split_dim > 1 {
                    return Err(Error::InvalidInput {
                        source: format!("Node {} has invalid split_dim: {} (must be 0 or 1)", node_id, inner.split_dim).into(),
                        location: location!(),
                    });
                }

                // Validate split value is within bounds
                let min_val = if inner.split_dim == 0 { bounds[0] } else { bounds[1] };
                let max_val = if inner.split_dim == 0 { bounds[2] } else { bounds[3] };
                if inner.split_value < min_val || inner.split_value > max_val {
                    return Err(Error::InvalidInput {
                        source: format!(
                            "Node {} split_value {} is outside dimension bounds [{}, {}]",
                            node_id, inner.split_value, min_val, max_val
                        ).into(),
                        location: location!(),
                    });
                }

                // Validate that children are properly sorted by split dimension
                // If points are sorted, left_max <= split_value <= right_min
                // Which means left_max <= right_min (no overlap)
                let left_node = &tree.nodes[inner.left_child as usize];
                let right_node = &tree.nodes[inner.right_child as usize];
                
                let left_bounds = left_node.bounds();
                let right_bounds = right_node.bounds();
                
                let (left_max, right_min) = if inner.split_dim == 0 {
                    (left_bounds[2], right_bounds[0])  // max_x of left, min_x of right
                } else {
                    (left_bounds[3], right_bounds[1])  // max_y of left, min_y of right
                };
                
                // Simple check: left_max should not be greater than right_min
                // If it is, points were not properly sorted before splitting!
                if left_max > right_min {
                    return Err(Error::InvalidInput {
                        source: format!(
                            "Node {} (split_dim={}, split_value={}): left child max {}={} > right child min {}={}. \
                            Points were not properly sorted before splitting!",
                            node_id, inner.split_dim, inner.split_value,
                            if inner.split_dim == 0 { "x" } else { "y" },
                            left_max,
                            if inner.split_dim == 0 { "x" } else { "y" },
                            right_min
                        ).into(),
                        location: location!(),
                    });
                }

                // Recursively validate children
                Box::pin(validate_node_recursive(
                    index,
                    tree,
                    inner.left_child,
                    visited,
                    leaf_count,
                    Some(bounds),
                    Some((inner.split_dim, inner.split_value, true)),
                    metrics,
                )).await?;
                Box::pin(validate_node_recursive(
                    index,
                    tree,
                    inner.right_child,
                    visited,
                    leaf_count,
                    Some(bounds),
                    Some((inner.split_dim, inner.split_value, false)),
                    metrics,
                )).await?;
            }
            bkd::BKDNode::Leaf(leaf) => {
                // Validate leaf data
                if leaf.num_rows == 0 {
                    return Err(Error::InvalidInput {
                        source: format!("Leaf node {} has zero rows", node_id).into(),
                        location: location!(),
                    });
                }

                // No extra validation needed for leaves - will be checked from parent

                *leaf_count += 1;
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_geo_intersects_with_custom_max_points_per_leaf() {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        
        // Test with different max points per leaf
        for max_points_per_leaf in [10, 50, 100, 200] {
            let test_store = create_test_store();
            
            let params = GeoIndexBuilderParams {
                max_points_per_leaf,
                batches_per_file: 5,
            };

            let mut builder = GeoIndexBuilder::try_new(params, DataType::Struct(Vec::<Field>::new().into()))
                .unwrap();

            // Create 500 RANDOM points (not sequential!) to catch sorting bugs
            let mut rng = StdRng::seed_from_u64(42);
            for i in 0..500 {
                let x = rng.random_range(0.0..100.0);
                let y = rng.random_range(0.0..100.0);
                builder.points.push((x, y, i as u64));
            }

            // Write index
            builder.write_index(test_store.as_ref()).await.unwrap();

            // Load index and verify
            let index = GeoIndex::load(test_store.clone(), None, &LanceCache::no_cache())
                .await
                .expect("Failed to load GeoIndex");

            assert_eq!(index.max_points_per_leaf, max_points_per_leaf);
            
          
            
            // Validate tree structure
            validate_bkd_tree(&index).await
                .expect(&format!("BKD tree validation failed for max_points_per_leaf={}", max_points_per_leaf));
        }
    }

    #[tokio::test]
    async fn test_geo_index_with_custom_batches_per_file() {
        // Test with different batches_per_file configurations
        for batches_per_file in [1, 3, 5, 10, 20] {
            let test_store = create_test_store();
            
            let params = GeoIndexBuilderParams {
                max_points_per_leaf: 50,
                batches_per_file,
            };

            let mut builder = GeoIndexBuilder::try_new(params, DataType::Struct(Vec::<Field>::new().into()))
                .unwrap();

            // Create 500 points (BKD spatial partitioning determines actual leaf count)
            for i in 0..500 {
                let x = (i % 100) as f64;
                let y = (i / 100) as f64;
                builder.points.push((x, y, i as u64));
            }

            // Write index
            builder.write_index(test_store.as_ref()).await.unwrap();

            // Load index and verify
            let index = GeoIndex::load(test_store.clone(), None, &LanceCache::no_cache())
                .await
                .expect("Failed to load GeoIndex");

            // Validate tree structure
            validate_bkd_tree(&index).await
                .expect(&format!("BKD tree validation failed for batches_per_file={}", batches_per_file));
            
        }
    }

    #[tokio::test]
    async fn test_geo_intersects_query_correctness_various_configs() {
        use crate::metrics::NoOpMetricsCollector;
        
        // Test query correctness with different configurations
        let configs = vec![
            (10, 1),   // Small leaves, one per file
            (50, 5),   // Default-ish
            (100, 10), // Larger leaves, many per file
        ];

        for (max_points_per_leaf, batches_per_file) in configs {
            let test_store = create_test_store();
            
            let params = GeoIndexBuilderParams {
                max_points_per_leaf,
                batches_per_file,
            };

            let mut builder = GeoIndexBuilder::try_new(params, DataType::Struct(Vec::<Field>::new().into()))
                .unwrap();

            // Create a grid of points
            for x in 0..20 {
                for y in 0..20 {
                    let row_id = (x * 20 + y) as u64;
                    builder.points.push((x as f64, y as f64, row_id));
                }
            }

            // Write and load index
            builder.write_index(test_store.as_ref()).await.unwrap();
            let index = GeoIndex::load(test_store.clone(), None, &LanceCache::no_cache())
                .await
                .unwrap();
            
            // Validate tree structure
            validate_bkd_tree(&index).await
                .expect(&format!("BKD tree validation failed for max_points_per_leaf={}, batches_per_file={}", max_points_per_leaf, batches_per_file));

            // Query: bbox [5, 5, 10, 10] should return points in that region
            let query = GeoQuery::Intersects(5.0, 5.0, 10.0, 10.0);
            
            let metrics = NoOpMetricsCollector {};
            let result = index.search(&query, &metrics).await.unwrap();
            
            // Should find points (5,5) to (10,10) = 6x6 = 36 points
            match result {
                crate::scalar::SearchResult::Exact(row_ids) => {
                    assert_eq!(row_ids.len().unwrap_or(0), 36,
                              "Expected 36 points for config max_points_per_leaf={}, batches_per_file={}, got {}",
                              max_points_per_leaf, batches_per_file, row_ids.len().unwrap_or(0));
                    
                    // Verify correct row IDs
                    for x in 5..=10 {
                        for y in 5..=10 {
                            let expected_row_id = (x * 20 + y) as u64;
                            assert!(row_ids.contains(expected_row_id),
                                   "Missing row_id {} for point ({}, {})",
                                   expected_row_id, x, y);
                        }
                    }
                }
                _ => panic!("Expected Exact search result"),
            }
        }
    }

    #[tokio::test]
    async fn test_geo_intersects_single_leaf() {
        // Edge case: all points fit in single leaf
        let test_store = create_test_store();
        
        let params = GeoIndexBuilderParams {
            max_points_per_leaf: 1000,
            batches_per_file: 5,
        };

        let mut builder = GeoIndexBuilder::try_new(params, DataType::Struct(Vec::<Field>::new().into()))
            .unwrap();

        // Create only 50 points
        for i in 0..50 {
            builder.points.push((i as f64, i as f64, i as u64));
        }

        builder.write_index(test_store.as_ref()).await.unwrap();
        let index = GeoIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

        // Should have only 1 leaf
        assert_eq!(index.bkd_tree.num_leaves, 1);
        assert_eq!(index.bkd_tree.nodes.len(), 1);
        
        // Validate tree structure
        validate_bkd_tree(&index).await
            .expect("BKD tree validation failed for single leaf test");
        
        // Single leaf should be in file 0 at offset 0
        match &index.bkd_tree.nodes[0] {
            BKDNode::Leaf(leaf) => {
                assert_eq!(leaf.file_id, 0);
                assert_eq!(leaf.row_offset, 0);
                assert_eq!(leaf.num_rows, 50);
            }
            _ => panic!("Expected leaf node"),
        }
    }

    #[tokio::test]
    async fn test_geo_intersects_many_small_leaves() {
        // Stress test: many small leaves, test file grouping
        let test_store = create_test_store();
        
        let params = GeoIndexBuilderParams {
            max_points_per_leaf: 5,      // Very small leaves
            batches_per_file: 3, // Few batches per file
        };

        let mut builder = GeoIndexBuilder::try_new(params, DataType::Struct(Vec::<Field>::new().into()))
            .unwrap();

        // Create 100 points (BKD spatial partitioning determines actual leaf count)
        for i in 0..100 {
            builder.points.push((i as f64, i as f64, i as u64));
        }

        builder.write_index(test_store.as_ref()).await.unwrap();
        let index = GeoIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

        // BKD tree creates more leaves due to spatial partitioning
        assert_eq!(index.bkd_tree.num_leaves, 32);
        
        // Validate tree structure
        validate_bkd_tree(&index).await
            .expect("BKD tree validation failed for many small leaves test");

        // 32 leaves / 3 batches_per_file = 11 files (ceil)
        let file_ids: std::collections::HashSet<u32> = index
            .bkd_tree
            .nodes
            .iter()
            .filter_map(|n| n.as_leaf())
            .map(|l| l.file_id)
            .collect();
        
        assert_eq!(file_ids.len(), 11);

        // Verify row offsets are cumulative within each file
        let leaves: Vec<_> = index
            .bkd_tree
            .nodes
            .iter()
            .filter_map(|n| n.as_leaf())
            .collect();

        for file_id in 0..11 {
            let leaves_in_file: Vec<_> = leaves
                .iter()
                .filter(|l| l.file_id == file_id)
                .collect();
            
            let mut expected_offset = 0u64;
            for leaf in leaves_in_file {
                assert_eq!(leaf.row_offset, expected_offset,
                          "Incorrect offset in file {}", file_id);
                expected_offset += leaf.num_rows;
            }
        }
    }

    #[tokio::test]
    async fn test_geo_index_data_integrity_after_serialization() {
        // Verify every point written can be read back exactly
        use crate::metrics::NoOpMetricsCollector;
        
        let configs = vec![
            (10, 1),   // Small leaves, one per file
            (50, 3),   // Medium leaves, few per file
            (100, 10), // Large leaves, many per file
        ];

        for (max_points_per_leaf, batches_per_file) in configs {
            let test_store = create_test_store();
            
            let params = GeoIndexBuilderParams {
                max_points_per_leaf,
                batches_per_file,
            };

            let mut builder = GeoIndexBuilder::try_new(params, DataType::Struct(Vec::<Field>::new().into()))
                .unwrap();

            // Create test points with known values
            let mut original_points = Vec::new();
            for i in 0..200 {
                let x = (i % 50) as f64 + 0.123; // Non-integer to test precision
                let y = (i / 50) as f64 + 0.456;
                let row_id = i as u64;
                original_points.push((x, y, row_id));
                builder.points.push((x, y, row_id));
            }

            // Write index
            builder.write_index(test_store.as_ref()).await.unwrap();

            // Load index
            let index = GeoIndex::load(test_store.clone(), None, &LanceCache::no_cache())
                .await
                .unwrap();

            // Read back all points by querying each leaf
            let metrics = NoOpMetricsCollector {};
            let mut recovered_points = std::collections::HashMap::new();

            for leaf in index.bkd_tree.nodes.iter().filter_map(|n| n.as_leaf()) {
                // Load the leaf data
                let leaf_data = index.load_leaf(leaf, &metrics).await.unwrap();
                
                // Extract points from this leaf
                let x_array = leaf_data
                    .column(0)
                    .as_primitive::<arrow_array::types::Float64Type>();
                let y_array = leaf_data
                    .column(1)
                    .as_primitive::<arrow_array::types::Float64Type>();
                let row_id_array = leaf_data
                    .column(2)
                    .as_primitive::<arrow_array::types::UInt64Type>();

                for i in 0..leaf_data.num_rows() {
                    let x = x_array.value(i);
                    let y = y_array.value(i);
                    let row_id = row_id_array.value(i);
                    recovered_points.insert(row_id, (x, y));
                }
            }

            // Verify all points were recovered
            assert_eq!(recovered_points.len(), original_points.len(),
                      "Lost points with config max_points_per_leaf={}, batches_per_file={}",
                      max_points_per_leaf, batches_per_file);

            // Verify each point matches exactly
            for (original_x, original_y, row_id) in &original_points {
                let (recovered_x, recovered_y) = recovered_points.get(row_id)
                    .expect(&format!("Missing row_id {} in recovered data", row_id));
                
                assert_eq!(*recovered_x, *original_x,
                          "X coordinate mismatch for row_id {}: expected {}, got {}",
                          row_id, original_x, recovered_x);
                assert_eq!(*recovered_y, *original_y,
                          "Y coordinate mismatch for row_id {}: expected {}, got {}",
                          row_id, original_y, recovered_y);
            }
        }
    }

    #[tokio::test]
    async fn test_geo_index_no_duplicate_points() {
        // Ensure no points are duplicated during write/read
        let test_store = create_test_store();
        
        let params = GeoIndexBuilderParams {
            max_points_per_leaf: 7,  // Odd number to test edge cases
            batches_per_file: 3,
        };

        let mut builder = GeoIndexBuilder::try_new(params, DataType::Struct(Vec::<Field>::new().into()))
            .unwrap();

        // Create 100 unique points
        for i in 0..100 {
            builder.points.push((i as f64, i as f64, i as u64));
        }

        builder.write_index(test_store.as_ref()).await.unwrap();
        let index = GeoIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

        // Read all row_ids from all leaves
        use crate::metrics::NoOpMetricsCollector;
        let metrics = NoOpMetricsCollector {};
        let mut all_row_ids = Vec::new();

        for leaf in index.bkd_tree.nodes.iter().filter_map(|n| n.as_leaf()) {
            let leaf_data = index.load_leaf(leaf, &metrics).await.unwrap();
            let row_id_array = leaf_data
                .column(2)
                .as_primitive::<arrow_array::types::UInt64Type>();

            for i in 0..leaf_data.num_rows() {
                all_row_ids.push(row_id_array.value(i));
            }
        }

        // Should have exactly 100 row_ids
        assert_eq!(all_row_ids.len(), 100, "Wrong number of points recovered");

        // All should be unique
        let unique_row_ids: std::collections::HashSet<u64> = all_row_ids.iter().copied().collect();
        assert_eq!(unique_row_ids.len(), 100, "Found duplicate points!");

        // Should be exactly 0..99
        for i in 0..100 {
            assert!(unique_row_ids.contains(&(i as u64)),
                   "Missing row_id {}", i);
        }
    }

    #[tokio::test]
    async fn test_geo_intersects_lazy_loading() {
        use crate::metrics::NoOpMetricsCollector;
        
        // Test that leaves are loaded lazily (not all at once)
        let test_store = create_test_store();
        
        let params = GeoIndexBuilderParams {
            max_points_per_leaf: 10,
            batches_per_file: 5,
        };

        let mut builder = GeoIndexBuilder::try_new(params, DataType::Struct(Vec::<Field>::new().into()))
            .unwrap();

        // Create 100 points in a grid
        for x in 0..10 {
            for y in 0..10 {
                builder.points.push((x as f64 * 10.0, y as f64 * 10.0, (x * 10 + y) as u64));
            }
        }

        builder.write_index(test_store.as_ref()).await.unwrap();
        let index = GeoIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

        // Query a small region - should only load relevant leaves, not all of them
        let query = GeoQuery::Intersects(0.0, 0.0, 20.0, 20.0);
        let metrics = NoOpMetricsCollector {};
        
        // Find which leaves would be touched
        let query_bbox = [0.0, 0.0, 20.0, 20.0];
        let intersecting_leaves = index.bkd_tree.find_intersecting_leaves(query_bbox).unwrap();
        
        // Verify we're not loading ALL leaves for a small query
        assert!(
            intersecting_leaves.len() < index.bkd_tree.num_leaves as usize,
            "Lazy loading test: small query should not touch all leaves! \
             Touched {}/{} leaves",
            intersecting_leaves.len(),
            index.bkd_tree.num_leaves
        );
        
        // Execute the query
        let result = index.search(&query, &metrics).await.unwrap();
        
        // Manually verify correctness: which points SHOULD be in bbox [0, 0, 20, 20]?
        let mut expected_row_ids = std::collections::HashSet::new();
        for x in 0..10 {
            for y in 0..10 {
                let point_x = x as f64 * 10.0;
                let point_y = y as f64 * 10.0;
                // st_intersects: point is inside bbox if min_x <= x <= max_x && min_y <= y <= max_y
                if point_x >= 0.0 && point_x <= 20.0 && point_y >= 0.0 && point_y <= 20.0 {
                    expected_row_ids.insert((x * 10 + y) as u64);
                }
            }
        }
        
        // Verify results match our manual calculation
        match result {
            crate::scalar::SearchResult::Exact(row_ids) => {
                let actual_count = row_ids.len().unwrap_or(0) as usize;
                
                assert_eq!(actual_count, expected_row_ids.len(),
                          "Expected {} points, got {}",
                          expected_row_ids.len(), actual_count);
                
                // Verify each returned row_id is in our expected set
                if let Some(iter) = row_ids.row_ids() {
                    for row_addr in iter {
                        let row_id = u64::from(row_addr);
                        assert!(expected_row_ids.contains(&row_id),
                               "Unexpected row_id {} in results", row_id);
                    }
                }
                
                // Verify we didn't miss any expected points
                if let Some(iter) = row_ids.row_ids() {
                    let found_ids: std::collections::HashSet<u64> = 
                        iter.map(|addr| u64::from(addr)).collect();
                    for expected_id in &expected_row_ids {
                        assert!(found_ids.contains(expected_id),
                               "Missing expected row_id {} in results", expected_id);
                    }
                }
            }
            _ => panic!("Expected Exact search result"),
        }
    }

    #[tokio::test]
    #[ignore] // Expensive test - run with: cargo test -- --ignored
    async fn test_geo_intersects_large_scale_lazy_loading() {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        use std::sync::atomic::{AtomicUsize, Ordering};
        
        // Custom metrics collector to track I/O operations
        struct LoadTracker {
            part_loads: AtomicUsize,
        }
        impl crate::metrics::MetricsCollector for LoadTracker {
            fn record_part_load(&self) {
                self.part_loads.fetch_add(1, Ordering::Relaxed);
            }
            fn record_parts_loaded(&self, _num_parts: usize) {}
            fn record_index_loads(&self, _num_indices: usize) {}
            fn record_comparisons(&self, _num_comparisons: usize) {}
        }
        
        let test_store = create_test_store();
        
        let params = GeoIndexBuilderParams {
            max_points_per_leaf: 1000,  // Realistic leaf size
            batches_per_file: 10,
        };

        let mut builder = GeoIndexBuilder::try_new(params, DataType::Struct(Vec::<Field>::new().into()))
            .unwrap();

        // Create 1 million random points across a 1000x1000 grid
        let mut rng = StdRng::seed_from_u64(42);
        for i in 0..1_000_000 {
            let x = rng.random_range(0.0..1000.0);
            let y = rng.random_range(0.0..1000.0);
            builder.points.push((x, y, i as u64));
        }

        builder.write_index(test_store.as_ref()).await.unwrap();
        
        let index = GeoIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

        // Run 100 random queries with load tracking
        let metrics = LoadTracker { part_loads: AtomicUsize::new(0) };
        let mut total_leaves_touched = 0;
        
        for _i in 0..100 {
            // Random query bbox with varying sizes (1x1 to 50x50 regions in 1000x1000 space)
            let width = rng.random_range(1.0..50.0);
            let height = rng.random_range(1.0..50.0);
            let min_x = rng.random_range(0.0..(1000.0 - width));
            let min_y = rng.random_range(0.0..(1000.0 - height));
            let max_x = min_x + width;
            let max_y = min_y + height;
            
            let query = GeoQuery::Intersects(min_x, min_y, max_x, max_y);
            
            // Count leaves touched
            let query_bbox = [min_x, min_y, max_x, max_y];
            let intersecting_leaves = index.bkd_tree.find_intersecting_leaves(query_bbox).unwrap();
            total_leaves_touched += intersecting_leaves.len();
            
            // Execute query
            let _result = index.search(&query, &metrics).await.unwrap();
        }
        
        let total_io_ops = metrics.part_loads.load(Ordering::Relaxed);
        
        // CRITICAL: Verify lazy loading is working!
        // We should NOT load all leaves - only the ones intersecting query bboxes
        assert!(
            total_io_ops < index.bkd_tree.num_leaves as usize,
            "❌ LAZY LOADING FAILED: Loaded {} leaves but index only has {} leaves! \
             Should load much fewer than total.",
            total_io_ops, index.bkd_tree.num_leaves
        );
        
        // Verify lazy loading is effective (< 10% of leaves touched on average per query)
        let avg_leaves_touched = total_leaves_touched as f64 / 100.0;
        let total_leaves = index.bkd_tree.num_leaves as f64;
        assert!(
            avg_leaves_touched < total_leaves * 0.1,
            "❌ Lazy loading ineffective: touching {:.1}% of leaves on average",
            (avg_leaves_touched / total_leaves) * 100.0
        );
    }

    #[tokio::test]
    async fn test_geo_index_points_in_correct_leaves() {
        // Verify points are in leaves with correct bounding boxes
        let test_store = create_test_store();
        
        let params = GeoIndexBuilderParams {
            max_points_per_leaf: 10,
            batches_per_file: 5,
        };

        let mut builder = GeoIndexBuilder::try_new(params, DataType::Struct(Vec::<Field>::new().into()))
            .unwrap();

        // Create a grid of points
        for x in 0..20 {
            for y in 0..20 {
                builder.points.push((x as f64, y as f64, (x * 20 + y) as u64));
            }
        }

        builder.write_index(test_store.as_ref()).await.unwrap();
        let index = GeoIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

        // Verify each point is within its leaf's bounding box
        use crate::metrics::NoOpMetricsCollector;
        let metrics = NoOpMetricsCollector {};

        for leaf in index.bkd_tree.nodes.iter().filter_map(|n| n.as_leaf()) {
            let leaf_data = index.load_leaf(leaf, &metrics).await.unwrap();
            
            let x_array = leaf_data
                .column(0)
                .as_primitive::<arrow_array::types::Float64Type>();
            let y_array = leaf_data
                .column(1)
                .as_primitive::<arrow_array::types::Float64Type>();

            for i in 0..leaf_data.num_rows() {
                let x = x_array.value(i);
                let y = y_array.value(i);
                
                // Point must be within leaf's bounding box
                assert!(x >= leaf.bounds[0] && x <= leaf.bounds[2],
                       "Point x={} outside leaf bounds [{}, {}]",
                       x, leaf.bounds[0], leaf.bounds[2]);
                assert!(y >= leaf.bounds[1] && y <= leaf.bounds[3],
                       "Point y={} outside leaf bounds [{}, {}]",
                       y, leaf.bounds[1], leaf.bounds[3]);
            }
        }
    }

    #[tokio::test]
    async fn test_geo_index_duplicate_coordinates_different_row_ids() {
        // Test that duplicate coordinates with different row_ids are all stored correctly
        use crate::metrics::NoOpMetricsCollector;
        
        let test_store = create_test_store();
        
        let params = GeoIndexBuilderParams {
            max_points_per_leaf: 10,
            batches_per_file: 3,
        };

        let mut builder = GeoIndexBuilder::try_new(params, DataType::Struct(Vec::<Field>::new().into()))
            .unwrap();

        // Create multiple points at the same coordinates with different row_ids
        // This simulates real-world scenarios where multiple records have the same location
        for i in 0..20 {
            // 5 points at location (10.0, 10.0) with different row_ids
            builder.points.push((10.0, 10.0, i as u64));
        }
        
        // Add some other points at different locations
        for i in 20..50 {
            builder.points.push((i as f64, i as f64, i as u64));
        }

        builder.write_index(test_store.as_ref()).await.unwrap();
        let index = GeoIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

        // Query the region containing the duplicates
        let query = GeoQuery::Intersects(9.0, 9.0, 11.0, 11.0);
        let metrics = NoOpMetricsCollector {};
        let result = index.search(&query, &metrics).await.unwrap();
        
        match result {
            crate::scalar::SearchResult::Exact(row_ids) => {
                // Should find all 20 points at (10.0, 10.0)
                assert_eq!(row_ids.len().unwrap_or(0), 20,
                          "Expected 20 duplicate coordinate entries");
                
                // Verify all row_ids 0..19 are present
                for i in 0..20 {
                    assert!(row_ids.contains(i as u64),
                           "Missing row_id {} for duplicate coordinate", i);
                }
            }
            _ => panic!("Expected Exact search result"),
        }

        // Verify data integrity: all 50 points should be stored
        let mut all_row_ids = Vec::new();
        for leaf in index.bkd_tree.nodes.iter().filter_map(|n| n.as_leaf()) {
            let leaf_data = index.load_leaf(leaf, &metrics).await.unwrap();
            let row_id_array = leaf_data
                .column(2)
                .as_primitive::<arrow_array::types::UInt64Type>();

            for i in 0..leaf_data.num_rows() {
                all_row_ids.push(row_id_array.value(i));
            }
        }

        assert_eq!(all_row_ids.len(), 50, "Should store all 50 points including duplicates");
        
        let unique_row_ids: std::collections::HashSet<u64> = all_row_ids.iter().copied().collect();
        assert_eq!(unique_row_ids.len(), 50, "All 50 row_ids should be unique");
    }

    #[tokio::test]
    async fn test_geo_index_many_duplicates_at_same_location() {
        // Test with many duplicate coordinates to ensure they don't cause issues
        use crate::metrics::NoOpMetricsCollector;
        
        let test_store = create_test_store();
        
        let params = GeoIndexBuilderParams {
            max_points_per_leaf: 20,
            batches_per_file: 5,
        };

        let mut builder = GeoIndexBuilder::try_new(params, DataType::Struct(Vec::<Field>::new().into()))
            .unwrap();

        // Create 100 points all at the exact same location
        for i in 0..100 {
            builder.points.push((50.0, 50.0, i as u64));
        }

        builder.write_index(test_store.as_ref()).await.unwrap();
        let index = GeoIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

        // Validate tree structure
        validate_bkd_tree(&index).await
            .expect("BKD tree validation failed with many duplicates");

        // Query should return all 100 points
        let query = GeoQuery::Intersects(49.0, 49.0, 51.0, 51.0);
        let metrics = NoOpMetricsCollector {};
        let result = index.search(&query, &metrics).await.unwrap();
        
        match result {
            crate::scalar::SearchResult::Exact(row_ids) => {
                assert_eq!(row_ids.len().unwrap_or(0), 100,
                          "Expected all 100 duplicate coordinate entries");
                
                // Verify all row_ids are present
                for i in 0..100 {
                    assert!(row_ids.contains(i as u64),
                           "Missing row_id {} in duplicate set", i);
                }
            }
            _ => panic!("Expected Exact search result"),
        }
    }

    #[tokio::test]
    async fn test_geo_index_duplicates_across_multiple_locations() {
        // Test duplicates at multiple different locations
        use crate::metrics::NoOpMetricsCollector;
        
        let test_store = create_test_store();
        
        let params = GeoIndexBuilderParams {
            max_points_per_leaf: 15,
            batches_per_file: 3,
        };

        let mut builder = GeoIndexBuilder::try_new(params, DataType::Struct(Vec::<Field>::new().into()))
            .unwrap();

        // Create clusters of duplicates at different locations
        let locations = vec![
            (10.0, 10.0),
            (20.0, 20.0),
            (30.0, 30.0),
            (40.0, 40.0),
        ];

        let mut row_id = 0u64;
        for (x, y) in &locations {
            // 10 duplicates at each location
            for _ in 0..10 {
                builder.points.push((*x, *y, row_id));
                row_id += 1;
            }
        }

        // Add some unique points
        for i in 0..20 {
            builder.points.push((i as f64, i as f64 + 50.0, row_id));
            row_id += 1;
        }

        let total_points = row_id;

        builder.write_index(test_store.as_ref()).await.unwrap();
        let index = GeoIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

        // Validate tree structure
        validate_bkd_tree(&index).await
            .expect("BKD tree validation failed with duplicates at multiple locations");

        let metrics = NoOpMetricsCollector {};

        // Query each location cluster
        for (i, (x, y)) in locations.iter().enumerate() {
            let query = GeoQuery::Intersects(x - 1.0, y - 1.0, x + 1.0, y + 1.0);
            let result = index.search(&query, &metrics).await.unwrap();
            
            match result {
                crate::scalar::SearchResult::Exact(row_ids) => {
                    assert_eq!(row_ids.len().unwrap_or(0), 10,
                              "Expected 10 duplicates at location {}", i);
                    
                    // Verify the correct row_ids for this cluster
                    let expected_start = (i * 10) as u64;
                    let expected_end = expected_start + 10;
                    for expected_id in expected_start..expected_end {
                        assert!(row_ids.contains(expected_id),
                               "Missing row_id {} in cluster at ({}, {})", expected_id, x, y);
                    }
                }
                _ => panic!("Expected Exact search result"),
            }
        }

        // Verify total count
        let mut all_row_ids = Vec::new();
        for leaf in index.bkd_tree.nodes.iter().filter_map(|n| n.as_leaf()) {
            let leaf_data = index.load_leaf(leaf, &metrics).await.unwrap();
            let row_id_array = leaf_data
                .column(2)
                .as_primitive::<arrow_array::types::UInt64Type>();

            for i in 0..leaf_data.num_rows() {
                all_row_ids.push(row_id_array.value(i));
            }
        }

        assert_eq!(all_row_ids.len(), total_points as usize,
                  "Should store all {} points including duplicates", total_points);
    }

    #[tokio::test]
    async fn test_geo_index_duplicate_handling_with_leaf_splits() {
        // Test that duplicates are handled correctly when they cause leaf splits
        use crate::metrics::NoOpMetricsCollector;
        
        let test_store = create_test_store();
        
        let params = GeoIndexBuilderParams {
            max_points_per_leaf: 10,  // Small leaf size to force splits
            batches_per_file: 3,
        };

        let mut builder = GeoIndexBuilder::try_new(params, DataType::Struct(Vec::<Field>::new().into()))
            .unwrap();

        // Create 50 points at the same location - should span multiple leaves
        for i in 0..50 {
            builder.points.push((25.0, 25.0, i as u64));
        }

        builder.write_index(test_store.as_ref()).await.unwrap();
        let index = GeoIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

        // Validate tree structure
        validate_bkd_tree(&index).await
            .expect("BKD tree validation failed with duplicate-induced splits");

        // Query should return all 50 points
        let query = GeoQuery::Intersects(24.0, 24.0, 26.0, 26.0);
        let metrics = NoOpMetricsCollector {};
        let result = index.search(&query, &metrics).await.unwrap();
        
        match result {
            crate::scalar::SearchResult::Exact(row_ids) => {
                assert_eq!(row_ids.len().unwrap_or(0), 50,
                          "Expected all 50 duplicate entries after leaf splits");
                
                // Verify all row_ids are present
                for i in 0..50 {
                    assert!(row_ids.contains(i as u64),
                           "Missing row_id {} after leaf split", i);
                }
            }
            _ => panic!("Expected Exact search result"),
        }
    }

    #[tokio::test]
    async fn test_geo_index_mixed_duplicates_and_unique_points() {
        // Realistic test: mix of unique points and duplicates
        use crate::metrics::NoOpMetricsCollector;
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        
        let test_store = create_test_store();
        
        let params = GeoIndexBuilderParams {
            max_points_per_leaf: 20,
            batches_per_file: 5,
        };

        let mut builder = GeoIndexBuilder::try_new(params, DataType::Struct(Vec::<Field>::new().into()))
            .unwrap();

        let mut rng = StdRng::seed_from_u64(123);
        let mut row_id = 0u64;

        // Add 100 random unique points
        for _ in 0..100 {
            let x = rng.random_range(0.0..100.0);
            let y = rng.random_range(0.0..100.0);
            builder.points.push((x, y, row_id));
            row_id += 1;
        }

        // Add 20 duplicates at a specific location
        for _ in 0..20 {
            builder.points.push((50.0, 50.0, row_id));
            row_id += 1;
        }

        // Add 50 more random unique points
        for _ in 0..50 {
            let x = rng.random_range(0.0..100.0);
            let y = rng.random_range(0.0..100.0);
            builder.points.push((x, y, row_id));
            row_id += 1;
        }

        let total_points = 170;

        builder.write_index(test_store.as_ref()).await.unwrap();
        let index = GeoIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

        // Validate tree structure
        validate_bkd_tree(&index).await
            .expect("BKD tree validation failed with mixed duplicates and unique points");

        let metrics = NoOpMetricsCollector {};

        // Query the duplicate cluster
        let query = GeoQuery::Intersects(49.0, 49.0, 51.0, 51.0);
        let result = index.search(&query, &metrics).await.unwrap();
        
        match result {
            crate::scalar::SearchResult::Exact(row_ids) => {
                assert_eq!(row_ids.len().unwrap_or(0), 20,
                          "Expected 20 duplicate entries at (50, 50)");
                
                // Verify the duplicate row_ids (100..119)
                for expected_id in 100..120 {
                    assert!(row_ids.contains(expected_id as u64),
                           "Missing row_id {} in duplicate cluster", expected_id);
                }
            }
            _ => panic!("Expected Exact search result"),
        }

        // Verify total count
        let mut all_row_ids = Vec::new();
        for leaf in index.bkd_tree.nodes.iter().filter_map(|n| n.as_leaf()) {
            let leaf_data = index.load_leaf(leaf, &metrics).await.unwrap();
            let row_id_array = leaf_data
                .column(2)
                .as_primitive::<arrow_array::types::UInt64Type>();

            for i in 0..leaf_data.num_rows() {
                all_row_ids.push(row_id_array.value(i));
            }
        }

        assert_eq!(all_row_ids.len(), total_points,
                  "Should store all {} points", total_points);
        
        // Verify all row_ids are unique (no double-counting)
        let unique_row_ids: std::collections::HashSet<u64> = all_row_ids.iter().copied().collect();
        assert_eq!(unique_row_ids.len(), total_points,
                  "All row_ids should be unique");
    }
}

