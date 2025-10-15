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

use arrow_array::{Array, ArrayRef, Float64Array, RecordBatch, UInt32Array, UInt64Array, UInt8Array};
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
const BKD_LEAVES_FILENAME: &str = "bkd_leaves.lance";
const GEO_INDEX_VERSION: u32 = 0;
const LEAF_SIZE_META_KEY: &str = "leaf_size";
const DEFAULT_LEAF_SIZE: u32 = 4096;

/// BKD Tree node representing either an inner node or a leaf
#[derive(Debug, Clone, DeepSizeOf)]
struct BKDNode {
    /// Bounding box: [min_x, min_y, max_x, max_y]
    bounds: [f64; 4],
    /// Split dimension: 0=X, 1=Y
    split_dim: u8,
    /// Split value along split_dim
    split_value: f64,
    /// Left child node index (for inner nodes)
    left_child: Option<u32>,
    /// Right child node index (for inner nodes)
    right_child: Option<u32>,
    /// Leaf ID in bkd_leaves.lance (for leaf nodes)
    leaf_id: Option<u32>,
}

/// In-memory BKD tree structure for efficient spatial queries
#[derive(Debug, DeepSizeOf)]
struct BKDTreeLookup {
    nodes: Vec<BKDNode>,
    root_id: u32,
    num_leaves: u32,
}

impl BKDTreeLookup {
    fn new(nodes: Vec<BKDNode>, root_id: u32, num_leaves: u32) -> Self {
        Self {
            nodes,
            root_id,
            num_leaves,
        }
    }

    /// Find all leaf IDs that intersect with the query bounding box
    fn find_intersecting_leaves(&self, query_bbox: [f64; 4]) -> Vec<u32> {
        let mut leaf_ids = Vec::new();
        let mut stack = vec![self.root_id];

        while let Some(node_id) = stack.pop() {
            if node_id as usize >= self.nodes.len() {
                continue;
            }

            let node = &self.nodes[node_id as usize];

            // Check if node's bounding box intersects with query bbox
            if !bboxes_intersect(&node.bounds, &query_bbox) {
                continue;
            }

            // If this is a leaf node, add its leaf_id
            if let Some(leaf_id) = node.leaf_id {
                leaf_ids.push(leaf_id);
            } else {
                // Inner node - traverse children
                if let Some(left) = node.left_child {
                    stack.push(left);
                }
                if let Some(right) = node.right_child {
                    stack.push(right);
                }
            }
        }

        leaf_ids
    }

    /// Deserialize from RecordBatch
    fn from_record_batch(batch: RecordBatch) -> Result<Self> {
        if batch.num_rows() == 0 {
            return Ok(Self::new(vec![], 0, 0));
        }

        let min_x = batch.column(0).as_primitive::<arrow_array::types::Float64Type>();
        let min_y = batch.column(1).as_primitive::<arrow_array::types::Float64Type>();
        let max_x = batch.column(2).as_primitive::<arrow_array::types::Float64Type>();
        let max_y = batch.column(3).as_primitive::<arrow_array::types::Float64Type>();
        let split_dim = batch.column(4).as_primitive::<arrow_array::types::UInt8Type>();
        let split_value = batch.column(5).as_primitive::<arrow_array::types::Float64Type>();
        let left_child = batch.column(6).as_primitive::<arrow_array::types::UInt32Type>();
        let right_child = batch.column(7).as_primitive::<arrow_array::types::UInt32Type>();
        let leaf_id = batch.column(8).as_primitive::<arrow_array::types::UInt32Type>();

        let mut nodes = Vec::with_capacity(batch.num_rows());
        let mut num_leaves = 0;

        for i in 0..batch.num_rows() {
            let leaf_id_val = if leaf_id.is_null(i) {
                None
            } else {
                num_leaves += 1;
                Some(leaf_id.value(i))
            };

            nodes.push(BKDNode {
                bounds: [
                    min_x.value(i),
                    min_y.value(i),
                    max_x.value(i),
                    max_y.value(i),
                ],
                split_dim: split_dim.value(i),
                split_value: split_value.value(i),
                left_child: if left_child.is_null(i) {
                    None
                } else {
                    Some(left_child.value(i))
                },
                right_child: if right_child.is_null(i) {
                    None
                } else {
                    Some(right_child.value(i))
                },
                leaf_id: leaf_id_val,
            });
        }

        Ok(Self::new(nodes, 0, num_leaves))
    }
}

/// Check if two bounding boxes intersect
fn bboxes_intersect(bbox1: &[f64; 4], bbox2: &[f64; 4]) -> bool {
    // bbox format: [min_x, min_y, max_x, max_y]
    !(bbox1[2] < bbox2[0] || bbox1[0] > bbox2[2] || bbox1[3] < bbox2[1] || bbox1[1] > bbox2[3])
}

/// Check if a point is within a bounding box
fn point_in_bbox(x: f64, y: f64, bbox: &[f64; 4]) -> bool {
    x >= bbox[0] && x <= bbox[2] && y >= bbox[1] && y <= bbox[3]
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

    /// Load a specific leaf from storage
    async fn load_leaf(
        &self,
        leaf_id: u32,
        index_reader: LazyIndexReader,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        // Check cache first
        let cache_key = BKDLeafKey { leaf_id };

        let cached = self
            .index_cache
            .get_or_insert_with_key(cache_key, move || async move {
                metrics.record_part_load();
                let reader = index_reader.get().await?;
                let batch = reader
                    .read_record_batch(leaf_id as u64, self.leaf_size as u64)
                    .await?;
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
        index_reader: LazyIndexReader,
        metrics: &dyn MetricsCollector,
    ) -> Result<RowIdTreeMap> {
        let leaf_data = self.load_leaf(leaf_id, index_reader, metrics).await?;

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

            if point_in_bbox(x, y, &query_bbox) {
                let row_id = row_id_array.value(i);
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

                log::debug!(
                    "Geo index search: st_intersects with bbox({}, {}, {}, {})",
                    min_x, min_y, max_x, max_y
                );

                // Step 1: Find intersecting leaves using in-memory tree traversal
                let leaf_ids = self.bkd_tree.find_intersecting_leaves(query_bbox);

                log::debug!(
                    "BKD tree found {} intersecting leaves",
                    leaf_ids.len()
                );

                // Step 2: Lazy-load and filter each leaf
                let mut all_row_ids = RowIdTreeMap::new();
                let lazy_reader = LazyIndexReader::new(self.store.clone(), BKD_LEAVES_FILENAME);

                for leaf_id in leaf_ids {
                    let leaf_row_ids = self
                        .search_leaf(leaf_id, query_bbox, lazy_reader.clone(), metrics)
                        .await?;
                    // Collect row IDs from the leaf and add them to the result set
                    let row_ids: Option<Vec<u64>> = leaf_row_ids.row_ids()
                        .map(|iter| iter.map(|row_addr| u64::from(row_addr)).collect());
                    if let Some(row_ids) = row_ids {
                        all_row_ids.extend(row_ids);
                    }
                }

                log::debug!(
                    "Geo index returning {:?} row IDs",
                    all_row_ids.len()
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

        // Write leaf data
        let leaf_schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        let mut leaf_file = index_store
            .new_index_file(BKD_LEAVES_FILENAME, leaf_schema)
            .await?;
        for leaf_batch in leaf_batches {
            leaf_file.write_record_batch(leaf_batch).await?;
        }
        leaf_file.finish().await?;

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

        // Write empty leaves file
        let leaf_schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        let empty_leaves = RecordBatch::new_empty(leaf_schema);
        let mut leaf_file = index_store
            .new_index_file(BKD_LEAVES_FILENAME, empty_leaves.schema())
            .await?;
        leaf_file.write_record_batch(empty_leaves).await?;
        leaf_file.finish().await?;

        Ok(())
    }

    fn serialize_tree_nodes(&self, nodes: &[BKDNode]) -> Result<RecordBatch> {
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

    // Build BKD tree using Lucene's algorithm (deferred for next step)
    fn build_bkd_tree(&mut self) -> Result<(Vec<BKDNode>, Vec<RecordBatch>)> {
        // For now, implement a simple single-leaf approach as placeholder
        // This will be replaced with the full BKD tree building algorithm
        log::warn!("Using simplified BKD tree builder (full implementation pending)");

        let num_points = self.points.len();
        let leaf_size = self.options.leaf_size as usize;

        // Calculate bounding box for all points
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;

        for (x, y, _) in &self.points {
            min_x = min_x.min(*x);
            min_y = min_y.min(*y);
            max_x = max_x.max(*x);
            max_y = max_y.max(*y);
        }

        // Sort points by X coordinate for better spatial locality
        self.points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Split points into leaf batches
        let mut leaf_batches = Vec::new();
        let mut nodes = Vec::new();

        for (leaf_id, chunk) in self.points.chunks(leaf_size).enumerate() {
            // Create leaf batch
            let x_vals: Vec<f64> = chunk.iter().map(|(x, _, _)| *x).collect();
            let y_vals: Vec<f64> = chunk.iter().map(|(_, y, _)| *y).collect();
            let row_ids: Vec<u64> = chunk.iter().map(|(_, _, r)| *r).collect();

            let schema = Arc::new(Schema::new(vec![
                Field::new("x", DataType::Float64, false),
                Field::new("y", DataType::Float64, false),
                Field::new(ROW_ID, DataType::UInt64, false),
            ]));

            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Float64Array::from(x_vals)) as ArrayRef,
                    Arc::new(Float64Array::from(y_vals)) as ArrayRef,
                    Arc::new(UInt64Array::from(row_ids)) as ArrayRef,
                ],
            )?;

            leaf_batches.push(batch);

            // Create leaf node (simplified - one node per leaf)
            nodes.push(BKDNode {
                bounds: [min_x, min_y, max_x, max_y],
                split_dim: 0,
                split_value: 0.0,
                left_child: None,
                right_child: None,
                leaf_id: Some(leaf_id as u32),
            });
        }

        // If we have multiple leaves, create a simple root node
        // (This is a placeholder - full tree construction will be more sophisticated)
        if nodes.len() > 1 {
            // For now, just use the first leaf node as root
            // Full implementation will build proper hierarchical tree
        }

        log::debug!(
            "Built simplified BKD tree: {} points -> {} leaves",
            num_points,
            leaf_batches.len()
        );

        Ok((nodes, leaf_batches))
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

