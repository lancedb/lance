//! R-tree spatial index implementation using rstar

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use rstar::{RTree, RTreeObject, AABB};
use serde::{Deserialize, Serialize};

use crate::rtree::{BoundingBox, GeoIndex, SpatialQuery};
use crate::metrics::MetricsCollector;
use crate::frag_reuse::FragReuseIndex;
use crate::scalar::{AnyQuery, IndexStore, ScalarIndex, SearchResult};
use crate::{Index, IndexType};
use lance_core::cache::LanceCache;
use lance_core::utils::mask::RowIdTreeMap;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;
use datafusion::execution::SendableRecordBatchStream;


/// An entry in the R-tree that associates a geometry's bounding box with its row ID
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RTreeEntry {
    /// The bounding box of the geometry
    pub bbox: BoundingBox,
    /// The row ID in the Lance dataset
    pub row_id: u64,
}

impl RTreeObject for RTreeEntry {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        self.bbox.to_aabb()
    }
}

impl DeepSizeOf for RTreeEntry {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.bbox.deep_size_of_children(context)
    }
}

/// R-tree based spatial index
#[derive(Debug)]
pub struct RTreeIndex {
    /// The R-tree data structure
    rtree: RTree<RTreeEntry>,
    /// Store for reading/writing index data
    store: Arc<dyn IndexStore>,
    /// Total bounds of all geometries in the index
    total_bounds: Option<BoundingBox>,
}

impl DeepSizeOf for RTreeIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        // Approximate the rtree size - rstar doesn't implement DeepSizeOf
        std::mem::size_of::<RTree<RTreeEntry>>() * self.rtree.size() +
        self.store.deep_size_of_children(context) +
        self.total_bounds.deep_size_of_children(context)
    }
}

impl RTreeIndex {
    /// Create a new R-tree index
    pub fn new(store: Arc<dyn IndexStore>) -> Self {
        Self {
            rtree: RTree::new(),
            store,
            total_bounds: None,
        }
    }

    /// Create an R-tree index from existing entries
    pub fn from_entries(entries: Vec<RTreeEntry>, store: Arc<dyn IndexStore>) -> Self {
        let total_bounds = if entries.is_empty() {
            None
        } else {
            let mut min_x = f64::INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut max_y = f64::NEG_INFINITY;

            for entry in &entries {
                min_x = min_x.min(entry.bbox.min_x);
                min_y = min_y.min(entry.bbox.min_y);
                max_x = max_x.max(entry.bbox.max_x);
                max_y = max_y.max(entry.bbox.max_y);
            }

            Some(BoundingBox::new(min_x, min_y, max_x, max_y))
        };

        Self {
            rtree: RTree::bulk_load(entries),
            store,
            total_bounds,
        }
    }

    /// Add a geometry bounding box with its row ID to the index
    pub fn insert(&mut self, bbox: BoundingBox, row_id: u64) {
        let entry = RTreeEntry { bbox: bbox.clone(), row_id };
        
        // Update total bounds
        match &mut self.total_bounds {
            Some(bounds) => {
                bounds.min_x = bounds.min_x.min(bbox.min_x);
                bounds.min_y = bounds.min_y.min(bbox.min_y);
                bounds.max_x = bounds.max_x.max(bbox.max_x);
                bounds.max_y = bounds.max_y.max(bbox.max_y);
            }
            None => {
                self.total_bounds = Some(bbox);
            }
        }

        self.rtree.insert(entry);
    }

    /// Perform a spatial query and return matching row IDs
    pub fn query(&self, query: &SpatialQuery) -> Result<Vec<u64>> {
        println!("🌍 GEO INDEX QUERY: {:?}", query);
        println!("🌍 R-tree size: {} entries", self.rtree.size());
        
        match query {
            SpatialQuery::Intersects(bbox) => {
                println!("🌍 INTERSECTS query: bbox={:?}", bbox);
                let query_aabb = bbox.to_aabb();
                println!("🌍 Query AABB: {:?}", query_aabb);
                
                let matching_entries: Vec<_> = self.rtree
                    .locate_in_envelope_intersecting(&query_aabb)
                    .collect();
                
                println!("🌍 Found {} intersecting entries:", matching_entries.len());
                // for (i, entry) in matching_entries.iter().enumerate() {
                //     println!("🌍   Entry {}: row_id={}, bbox={:?}", i, entry.row_id, entry.bbox);
                // }
                
                let results: Vec<u64> = matching_entries.into_iter()
                    .map(|entry| entry.row_id)
                    .collect();
                
                println!("🌍 INTERSECTS result: {} matching row IDs: {:?}", results.len(), results);
                Ok(results)
            }
            SpatialQuery::Within(bbox) => {
                log::info!("🌍 WITHIN query: bbox={:?}", bbox);
                let query_aabb = bbox.to_aabb();
                let results: Vec<u64> = self.rtree
                    .locate_in_envelope(&query_aabb)
                    .map(|entry| entry.row_id)
                    .collect();
                log::info!("🌍 WITHIN result: {} matching row IDs", results.len());
                Ok(results)
            }
            SpatialQuery::Contains(point) => {
                log::info!("🌍 CONTAINS query: point={:?}", point);
                let point_aabb = AABB::from_point(point.to_array());
                let results: Vec<u64> = self.rtree
                    .locate_in_envelope_intersecting(&point_aabb)
                    .filter(|entry| {
                        // Check if the bounding box actually contains the point
                        point.x >= entry.bbox.min_x &&
                        point.x <= entry.bbox.max_x &&
                        point.y >= entry.bbox.min_y &&
                        point.y <= entry.bbox.max_y
                    })
                    .map(|entry| entry.row_id)
                    .collect();
                log::info!("🌍 CONTAINS result: {} matching row IDs", results.len());
                Ok(results)
            }
            SpatialQuery::DWithin(point, distance) => {
                log::info!("🌍 DWITHIN query: point={:?}, distance={}", point, distance);
                let expanded_bbox = BoundingBox::new(
                    point.x - distance,
                    point.y - distance,
                    point.x + distance,
                    point.y + distance,
                );
                let query_aabb = expanded_bbox.to_aabb();
                let results: Vec<u64> = self.rtree
                    .locate_in_envelope_intersecting(&query_aabb)
                    .map(|entry| entry.row_id)
                    .collect();
                log::info!("🌍 DWITHIN result: {} matching row IDs", results.len());
                Ok(results)
            }
            SpatialQuery::Touches(bbox) => {
                log::info!("🌍 TOUCHES query: bbox={:?}", bbox);
                // For now, approximate touches as intersects
                // A proper implementation would check for boundary intersection only
                let query_aabb = bbox.to_aabb();
                let results: Vec<u64> = self.rtree
                    .locate_in_envelope_intersecting(&query_aabb)
                    .map(|entry| entry.row_id)
                    .collect();
                log::info!("🌍 TOUCHES result: {} matching row IDs", results.len());
                Ok(results)
            }
            SpatialQuery::Disjoint(bbox) => {
                log::info!("🌍 DISJOINT query: bbox={:?}", bbox);
                // Return all entries that don't intersect with the bounding box
                let query_aabb = bbox.to_aabb();
                let intersecting: std::collections::HashSet<u64> = self.rtree
                    .locate_in_envelope_intersecting(&query_aabb)
                    .map(|entry| entry.row_id)
                    .collect();
                
                let all_results: Vec<u64> = self.rtree
                    .iter()
                    .map(|entry| entry.row_id)
                    .filter(|row_id| !intersecting.contains(row_id))
                    .collect();
                log::info!("🌍 DISJOINT result: {} matching row IDs", all_results.len());
                Ok(all_results)
            }
        }
    }

    /// Save the index to storage using binary serialization
    pub async fn save(&self) -> Result<()> {
        println!("🌍 Saving RTreeIndex to IndexStore");
        // Collect all R-tree entries
        let entries: Vec<RTreeEntry> = self.rtree.iter().cloned().collect();
        
        // Serialize entries using serde_json (could also use bincode for better performance)
        let serialized = serde_json::to_vec(&entries).map_err(|e| Error::Arrow {
            message: format!("Failed to serialize R-tree entries: {}", e),
            location: snafu::location!(),
        })?;

        // Create a binary schema for the serialized data
        let schema = Arc::new(Schema::new(vec![
            Field::new("rtree_data", DataType::Binary, false),
        ]));

        // Create record batch with the serialized data
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(arrow_array::BinaryArray::from_iter_values([serialized]))],
        ).map_err(|e| Error::Arrow {
            message: format!("Failed to create record batch: {}", e),
            location: snafu::location!(),
        })?;

        let mut writer = self.store.new_index_file("page_data.lance", schema).await?;
        writer.write_record_batch(batch).await?;
        
        // Save metadata about total bounds and entry count
        let metadata = if let Some(bounds) = &self.total_bounds {
            HashMap::from([
                ("total_min_x".to_string(), bounds.min_x.to_string()),
                ("total_min_y".to_string(), bounds.min_y.to_string()),
                ("total_max_x".to_string(), bounds.max_x.to_string()),
                ("total_max_y".to_string(), bounds.max_y.to_string()),
                ("num_entries".to_string(), entries.len().to_string()),
            ])
        } else {
            HashMap::from([
                ("num_entries".to_string(), entries.len().to_string()),
            ])
        };
        println!("🌍 Saving metadata: {:?}", metadata);
        writer.finish_with_metadata(metadata).await?;

        // Create a lookup file (required by Lance index system)
        let lookup_schema = Arc::new(Schema::new(vec![
            Field::new("num_entries", DataType::UInt64, false),
        ]));
        
        let lookup_batch = RecordBatch::try_new(
            lookup_schema.clone(),
            vec![Arc::new(arrow_array::UInt64Array::from(vec![entries.len() as u64]))],
        ).map_err(|e| Error::Arrow { 
            message: format!("Failed to create lookup batch: {}", e),
            location: snafu::location!(),
        })?;
        
        let mut lookup_writer = self.store.new_index_file("page_lookup.lance", lookup_schema).await?;
        lookup_writer.write_record_batch(lookup_batch).await?;
        lookup_writer.finish().await?;

        Ok(())
    }

    /// Load the index from storage using binary deserialization
    pub async fn load(store: Arc<dyn IndexStore>) -> Result<Self> {
        println!("🌍 Loading RTreeIndex from IndexStore");
        let reader = store.open_index_file("page_data.lance").await?;
        let num_rows = reader.num_rows();
        
        if num_rows == 0 {
            return Ok(Self::new(store));
        }

        let batch = reader.read_range(0..num_rows, None).await?;
        
        // Expect single binary column with serialized data
        let binary_data = batch.column(0)
            .as_any()
            .downcast_ref::<arrow_array::BinaryArray>()
            .ok_or_else(|| Error::Index {
                message: "Expected BinaryArray for rtree_data column".to_string(),
                location: snafu::location!(),
            })?;

        // Get the serialized data from the first (and only) row
        let serialized_bytes = binary_data.value(0);
        
        // Deserialize the R-tree entries
        let entries: Vec<RTreeEntry> = serde_json::from_slice(serialized_bytes).map_err(|e| Error::Index {
            message: format!("Failed to deserialize R-tree entries: {}", e),
            location: snafu::location!(),
        })?;

        println!("🌍 Loaded {} R-tree entries using binary deserialization", entries.len());
        Ok(Self::from_entries(entries, store))
    }
}

#[async_trait]
impl GeoIndex for RTreeIndex {
    async fn search(
        &self,
        query: &SpatialQuery,
        _metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let row_ids = self.query(query)?;
        let row_id_set: Vec<u64> = row_ids.into_iter().collect();
        Ok(SearchResult::Exact(RowIdTreeMap::from_iter(row_id_set)))
    }

    fn can_answer_exact(&self, query: &SpatialQuery) -> bool {
        match query {
            SpatialQuery::Intersects(_) => true,
            SpatialQuery::Within(_) => true,
            SpatialQuery::Contains(_) => false, // Needs recheck for exact containment
            SpatialQuery::DWithin(_, _) => false, // Needs recheck for exact distance
            SpatialQuery::Touches(_) => false, // Needs recheck for exact boundary touching
            SpatialQuery::Disjoint(_) => true,
        }
    }

    async fn load(store: Arc<dyn IndexStore>) -> Result<Arc<Self>> {
        println!("🌍 Loading RTreeIndex from IndexStore (GeoIndex trait)");
        
        // Use the simpler RTreeIndex::load method with binary deserialization
        let index = Self::load(store).await?;
        Ok(Arc::new(index))
    }

    fn total_bounds(&self) -> Option<BoundingBox> {
        self.total_bounds.clone()
    }

    fn size(&self) -> usize {
        self.rtree.size()
    }
}

#[async_trait]
impl Index for RTreeIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(Error::Index {
            message: "GeoIndex cannot be cast to VectorIndex".to_string(),
            location: snafu::location!(),
        })
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let stats = serde_json::json!({
            "index_type": "RTree",
            "size": self.size(),
            "total_bounds": self.total_bounds
        });
        Ok(stats)
    }

    async fn prewarm(&self) -> Result<()> {
        // R-tree is already in memory, nothing to prewarm
        Ok(())
    }

    fn index_type(&self) -> IndexType {
        IndexType::Scalar
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        // For now, assume all fragments are included
        // In a real implementation, this would examine the row IDs to determine fragments
        let mut bitmap = RoaringBitmap::new();
        for entry in self.rtree.iter() {
            // Extract fragment ID from row ID (implementation depends on Lance's row ID scheme)
            let fragment_id = (entry.row_id >> 32) as u32;
            bitmap.insert(fragment_id);
        }
        Ok(bitmap)
    }
}

#[async_trait]
impl ScalarIndex for RTreeIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        // Try to downcast to SpatialQuery
        if let Some(spatial_query) = query.as_any().downcast_ref::<SpatialQuery>() {
            GeoIndex::search(self, spatial_query, metrics).await
        } else {
            Err(Error::Index {
                message: "RTreeIndex can only handle SpatialQuery".to_string(),
                location: snafu::location!(),
            })
        }
    }

    fn can_answer_exact(&self, query: &dyn AnyQuery) -> bool {
        if let Some(spatial_query) = query.as_any().downcast_ref::<SpatialQuery>() {
            GeoIndex::can_answer_exact(self, spatial_query)
        } else {
            false
        }
    }

    async fn load(
        store: Arc<dyn IndexStore>,
        _frag_reuse_index: Option<Arc<FragReuseIndex>>,
        _index_cache: LanceCache,
    ) -> Result<Arc<Self>> {
        let index = Self::load(store).await?;
        Ok(Arc::new(index))
    }

    async fn remap(
        &self,
        _mapping: &HashMap<u64, Option<u64>>,
        _dest_store: &dyn IndexStore,
    ) -> Result<()> {
        // TODO: Implement remapping for geo index
        Err(Error::Index {
            message: "Remapping not yet implemented for RTreeIndex".to_string(),
            location: snafu::location!(),
        })
    }

    async fn update(
        &self,
        _new_data: SendableRecordBatchStream,
        _dest_store: &dyn IndexStore,
    ) -> Result<()> {
        // TODO: Implement updating for geo index
        Err(Error::Index {
            message: "Update not yet implemented for RTreeIndex".to_string(),
            location: snafu::location!(),
        })
    }
}