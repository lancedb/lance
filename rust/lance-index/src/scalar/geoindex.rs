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

use arrow_array::{ArrayRef, Float64Array, RecordBatch, UInt32Array, UInt64Array, UInt8Array};
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

const BKD_TREE_INNER_FILENAME: &str = "bkd_tree_inner.lance";
const BKD_TREE_LEAF_FILENAME: &str = "bkd_tree_leaf.lance";
const LEAF_GROUP_PREFIX: &str = "leaf_group_";
const DEFAULT_BATCHES_PER_LEAF_FILE: u32 = 5; // Default number of leaf batches per file
const GEO_INDEX_VERSION: u32 = 0;
const LEAF_SIZE_META_KEY: &str = "leaf_size";
const BATCHES_PER_FILE_META_KEY: &str = "batches_per_file";
const DEFAULT_LEAF_SIZE: u32 = 100; // for test
// const DEFAULT_LEAF_SIZE: u32 = 1024; // for production

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
        leaf: &crate::scalar::bkd::BKDLeafNode,
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

    /// Search a specific leaf for points within the query bbox
    async fn search_leaf(
        &self,
        leaf: &crate::scalar::bkd::BKDLeafNode,
        query_bbox: [f64; 4],
        metrics: &dyn MetricsCollector,
    ) -> Result<RowIdTreeMap> {
        let leaf_data = self.load_leaf(leaf, metrics).await?;

        let file_id = leaf.file_id;
        let row_offset = leaf.row_offset;
        let num_rows = leaf.num_rows;
        
        println!(
            "🔍 Searching leaf (file={}, offset={}, rows={}) with {} points, query_bbox: {:?}",
            file_id,
            row_offset,
            num_rows,
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
        
        // Debug: Check if SF (row_id=0) is in this leaf's actual data
        let contains_sf = (0..leaf_data.num_rows()).any(|i| row_id_array.value(i) == 0);
        if contains_sf {
            println!("  🔎 Leaf (file={}, offset={}) contains SF (row_id=0)!", file_id, row_offset);
            for i in 0..leaf_data.num_rows() {
                if row_id_array.value(i) == 0 {
                    let x = x_array.value(i);
                    let y = y_array.value(i);
                    println!("  🎯 Found SF at index {}: ({}, {}) row_id=0", i, x, y);
                    println!("  🎯 point_in_bbox result: {}", point_in_bbox(x, y, &query_bbox));
                    break;
                }
            }
        }
        
        for i in 0..leaf_data.num_rows() {
            let x = x_array.value(i);
            let y = y_array.value(i);
            let row_id = row_id_array.value(i);

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
            "📊 Leaf (file={}, offset={}) matched {} out of {} points",
            file_id,
            row_offset,
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
                let leaves = self.bkd_tree.find_intersecting_leaves(query_bbox)?;

                println!(
                    "📊 BKD tree traversal found {} intersecting leaves out of {} total leaves",
                    leaves.len(),
                    self.bkd_tree.num_leaves
                );

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

                println!(
                    "✅ Geo index searched {} leaves and returning {} row IDs\n",
                    leaves.len(),
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

        // Write tree structure to separate inner and leaf files
        let (inner_batch, leaf_metadata_batch) = self.serialize_tree_nodes(&tree_nodes)?;
        
        // Write inner nodes
        let mut inner_file = index_store
            .new_index_file(BKD_TREE_INNER_FILENAME, inner_batch.schema())
            .await?;
        inner_file.write_record_batch(inner_batch).await?;
        inner_file
            .finish_with_metadata(HashMap::from([(
                LEAF_SIZE_META_KEY.to_string(),
                self.options.leaf_size.to_string(),
            )]))
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
        
        let num_groups = (leaf_batches.len() as u32 + DEFAULT_BATCHES_PER_LEAF_FILE - 1) / DEFAULT_BATCHES_PER_LEAF_FILE;
        println!("📝 Writing {} leaf batches into {} group files ({} batches per file)", 
                 leaf_batches.len(), num_groups, DEFAULT_BATCHES_PER_LEAF_FILE);
        
        for group_id in 0..num_groups {
            let start_idx = (group_id * DEFAULT_BATCHES_PER_LEAF_FILE) as usize;
            let end_idx = ((group_id + 1) * DEFAULT_BATCHES_PER_LEAF_FILE).min(leaf_batches.len() as u32) as usize;
            let group_batches = &leaf_batches[start_idx..end_idx];
            
            let filename = leaf_group_filename(group_id);
            println!("  Writing {}: {} batches ({} rows total)", 
                     filename, 
                     group_batches.len(),
                     group_batches.iter().map(|b| b.num_rows()).sum::<usize>());
            
            let mut leaf_file = index_store
                .new_index_file(&filename, leaf_schema.clone())
                .await?;
                
            for (batch_idx, leaf_batch) in group_batches.iter().enumerate() {
                let batch_id = leaf_file.write_record_batch(leaf_batch.clone()).await?;
                println!("    Batch {}: {} rows (batch_id={})", 
                         start_idx + batch_idx, leaf_batch.num_rows(), batch_id);
            }
            
            leaf_file.finish().await?;
        }
        println!("✅ Finished writing {} group files\n", num_groups);

        log::debug!(
            "Wrote BKD tree with {} nodes",
            tree_nodes.len()
        );

        Ok(())
    }

    async fn write_empty_index(&self, index_store: &dyn IndexStore) -> Result<()> {
        // Write empty inner node file
        let inner_schema = crate::scalar::bkd::inner_node_schema();
        let empty_inner = RecordBatch::new_empty(inner_schema);
        let mut inner_file = index_store
            .new_index_file(BKD_TREE_INNER_FILENAME, empty_inner.schema())
            .await?;
        inner_file.write_record_batch(empty_inner).await?;
        inner_file.finish().await?;

        // Write empty leaf metadata file
        let leaf_schema = crate::scalar::bkd::leaf_node_schema();
        let empty_leaf = RecordBatch::new_empty(leaf_schema);
        let mut leaf_file = index_store
            .new_index_file(BKD_TREE_LEAF_FILENAME, empty_leaf.schema())
            .await?;
        leaf_file.write_record_batch(empty_leaf).await?;
        leaf_file.finish().await?;

        // No actual leaf data files needed for empty index

        Ok(())
    }

    // Serialize tree nodes to separate inner and leaf RecordBatches
    fn serialize_tree_nodes(&self, nodes: &[crate::scalar::bkd::BKDNode]) -> Result<(RecordBatch, RecordBatch)> {
        use crate::scalar::bkd::BKDNode;
        
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
    
    fn serialize_inner_nodes(nodes: &[(u32, &crate::scalar::bkd::BKDNode)]) -> Result<RecordBatch> {
        use crate::scalar::bkd::BKDNode;
        
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

        let schema = crate::scalar::bkd::inner_node_schema();

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
    
    fn serialize_leaf_nodes(nodes: &[(u32, &crate::scalar::bkd::BKDNode)]) -> Result<RecordBatch> {
        use crate::scalar::bkd::BKDNode;
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

        let schema = crate::scalar::bkd::leaf_node_schema();

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
    fn build_bkd_tree(&mut self) -> Result<(Vec<crate::scalar::bkd::BKDNode>, Vec<RecordBatch>)> {
        let builder = BKDTreeBuilder::new(self.options.leaf_size as usize);
        builder.build(&mut self.points, DEFAULT_BATCHES_PER_LEAF_FILE)
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

