//! Simplified tiled spatial index (points + ST_Within only) using Morton/Z-order sorting
//!
//! Build time (offline):
//! 1. Compute Morton codes → sort → cut into tiles
//! 2. For each tile, write a columnar file (tiles/tile_000123.entries.lance) with:
//!    - row_id: u64
//!    - morton_code: u64
//!    - minx, miny, maxx, maxy: f64 (bbox of point; min==max)
//!    - (optional) lon, lat: f32 (kept for compatibility; not required)
//! 3. Write a manifest (morton.lance) with one row per tile:
//!    - tile_id, zmin, zmax, minx, miny, maxx, maxy, count, path, crs, version
//! 4. No R-trees are stored on disk
//!
//! Query time (online):
//! 1. Use the manifest to route to relevant tiles by ST_Within on tile bbox
//! 2. For each relevant tile:
//!    - Read *.entries.lance
//!    - Build an in-memory RTree<TiledSpatialEntry> via RTree::bulk_load(entries)
//!    - Query the tile's R-tree using locate_in_envelope (ST_Within)

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use deepsize::DeepSizeOf;
use rstar::{RTree, RTreeObject, AABB};
use serde::{Deserialize, Serialize};

use crate::rtree::{BoundingBox, GeoIndex, SpatialQuery, SpatialGeometry};
use crate::metrics::MetricsCollector;
use crate::scalar::{AnyQuery, IndexStore, ScalarIndex, SearchResult};
use crate::frag_reuse::FragReuseIndex;
use crate::{Index, IndexType};
use lance_core::utils::mask::RowIdTreeMap;
use lance_core::Result;
use snafu::location;
use datafusion::execution::SendableRecordBatchStream;

/// Configuration for the tiled R-tree
#[derive(Debug, Clone, Serialize, Deserialize, DeepSizeOf)]
pub struct TiledRTreeConfig {
    /// Target number of entries per tile
    pub entries_per_tile: usize,
    /// Maximum depth for Morton encoding (affects precision)
    pub morton_depth: u32,
}

impl Default for TiledRTreeConfig {
    fn default() -> Self {
        Self {
            entries_per_tile: 1000,
            morton_depth: 20, // 20 bits per dimension = 40 bits total
        }
    }
}

/// A spatial entry stored in columnar format for tiles
#[derive(Debug, Clone, Serialize, Deserialize, DeepSizeOf)]
pub struct TiledSpatialEntry {
    /// The row ID in the Lance dataset
    pub row_id: u64,
    /// Morton code for this entry (optional but handy)
    pub morton_code: u64,
    /// Bounding box coordinates (for points min==max)
    pub minx: f64,
    pub miny: f64,
    pub maxx: f64,
    pub maxy: f64,
    /// Optional lon/lat for cheap refine for radius queries
    pub lon: Option<f32>,
    pub lat: Option<f32>,
}

impl TiledSpatialEntry {
    pub fn new(geometry: SpatialGeometry, row_id: u64, bounds: &BoundingBox, morton_depth: u32) -> Self {
        let bbox = geometry.envelope();
        let morton_code = Self::compute_morton_code(&geometry, bounds, morton_depth);
        
        // Extract lon/lat for points (for cheap radius query refinement)
        let (lon, lat) = match &geometry {
            SpatialGeometry::Point(point) => (Some(point.x as f32), Some(point.y as f32)),
        };
        
        Self {
            row_id,
            morton_code,
            minx: bbox.min_x,
            miny: bbox.min_y,
            maxx: bbox.max_x,
            maxy: bbox.max_y,
            lon,
            lat,
        }
    }

    /// Compute Morton code for this geometry
    fn compute_morton_code(geometry: &SpatialGeometry, bounds: &BoundingBox, morton_depth: u32) -> u64 {
        match geometry {
            SpatialGeometry::Point(point) => {
                Self::morton_encode(point.x, point.y, bounds, morton_depth)
            }
        }
    }

    /// Calculate Morton/Z-order code for a point with configurable precision
    fn morton_encode(x: f64, y: f64, bounds: &BoundingBox, morton_depth: u32) -> u64 {
        // Normalize coordinates to [0, 1] range, guarding against zero spans
        let dx = bounds.max_x - bounds.min_x;
        let dy = bounds.max_y - bounds.min_y;
        let norm_x = if dx == 0.0 {
            0.5
        } else {
            ((x - bounds.min_x) / dx).clamp(0.0, 1.0)
        };
        let norm_y = if dy == 0.0 {
            0.5
        } else {
            ((y - bounds.min_y) / dy).clamp(0.0, 1.0)
        };
        
        // Convert to integer coordinates with specified precision
        let max_val = (1u64 << morton_depth) - 1;
        let int_x = (norm_x * max_val as f64) as u32;
        let int_y = (norm_y * max_val as f64) as u32;
        
        // Use morton-encoding crate for efficient Morton encoding
        morton_encoding::morton_encode([int_x, int_y])
    }

    /// Get the bounding box of this entry
    pub fn bbox(&self) -> BoundingBox {
        BoundingBox::new(self.minx, self.miny, self.maxx, self.maxy)
    }
    
    /// Check if this entry matches a spatial query (for refinement)
    pub fn matches_query(&self, query: &SpatialQuery) -> bool {
        match query {
            SpatialQuery::Within(query_bbox) => self.bbox().within(query_bbox),
            _ => false,
        }
    }
}

impl RTreeObject for TiledSpatialEntry {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        self.bbox().to_aabb()
    }
}

/// Manifest entry for each tile (stored in morton.lance)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileManifestEntry {
    /// Unique tile identifier
    pub tile_id: u64,
    /// Bounding box encompassing all entries in this tile
    pub minx: f64,
    pub miny: f64,
    pub maxx: f64,
    pub maxy: f64,
}

impl RTreeObject for TileManifestEntry {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners([self.minx, self.miny], [self.maxx, self.maxy])
    }
}

// No deep children for the simplified manifest entry
impl DeepSizeOf for TileManifestEntry {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize { 0 }
}
    
    /// A tile containing a subset of spatially-coherent entries with its own R-tree
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SpatialTile {
        /// Unique tile identifier
        pub tile_id: u64,
        /// Morton code range for this tile (min, max)
        pub morton_range: (u64, u64),
        /// Bounding box encompassing all entries in this tile
        pub bbox: BoundingBox,
        /// Number of entries in this tile
        pub entry_count: usize,
    }
    
    impl RTreeObject for SpatialTile {
        type Envelope = AABB<[f64; 2]>;
    
        fn envelope(&self) -> Self::Envelope {
            self.bbox.to_aabb()
        }
    }
    
    impl DeepSizeOf for SpatialTile {
        fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
            self.bbox.deep_size_of_children(context)
        }
    }
    
/// Manager for tile-specific R-trees and manifest
    #[derive(Debug)]
    pub struct TileManager {
        store: Arc<dyn IndexStore>,
        config: TiledRTreeConfig,
    }
    
    impl TileManager {
        pub fn new(store: Arc<dyn IndexStore>, config: TiledRTreeConfig) -> Self {
            Self { store, config }
        }
    
    /// Create tiles from Morton-sorted entries (offline build process)
    pub async fn create_tiles(&self, mut entries: Vec<TiledSpatialEntry>) -> Result<Vec<TileManifestEntry>> {
            if entries.is_empty() {
                return Ok(Vec::new());
            }
    
            println!("🔲 Creating tiles from {} Morton-sorted entries", entries.len());
            
            // Sort by Morton code for optimal spatial locality
            entries.sort_by_key(|entry| entry.morton_code);
            println!("🔲 Sorted {} entries by Morton code", entries.len());
    
        let mut manifest_entries = Vec::new();
            let mut tile_id = 0u64;
    
            // Create tiles by chunking Morton-sorted entries
            for chunk in entries.chunks(self.config.entries_per_tile) {
                if chunk.is_empty() {
                    continue;
                }
    
                let tile_entries = chunk.to_vec();
                let entry_count = tile_entries.len();
                
                // Calculate Morton range and bounding box for this tile
                let morton_min = tile_entries.first().unwrap().morton_code;
                let morton_max = tile_entries.last().unwrap().morton_code;
                let tile_bbox = Self::compute_tile_bbox(&tile_entries);
                
            // Write columnar tile file (tiles/tile_000123.entries.lance)
            self.store_tile_entries(tile_id, &tile_entries).await?;
            
                println!("🔲 Created tile {} with {} entries, Morton range: {}-{}, bbox: {:?}", 
                         tile_id, entry_count, morton_min, morton_max, tile_bbox);
    
            // Create manifest entry (tile id + bbox only)
            let manifest_entry = TileManifestEntry {
                tile_id,
                minx: tile_bbox.min_x,
                miny: tile_bbox.min_y,
                maxx: tile_bbox.max_x,
                maxy: tile_bbox.max_y,
            };

            manifest_entries.push(manifest_entry);
                tile_id += 1;
            }
    
        // Store the manifest (morton.lance)
        self.store_manifest(&manifest_entries).await?;

        println!("🔲 Created {} tiles with optimal Morton/Z-order spatial locality", manifest_entries.len());
        Ok(manifest_entries)
    }

    /// Store tile entries in columnar format
    async fn store_tile_entries(&self, tile_id: u64, entries: &[TiledSpatialEntry]) -> Result<()> {
        use arrow_array::{Float32Array, Float64Array, UInt64Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};

        // Create schema for columnar tile storage
        let schema = Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::UInt64, false),
            Field::new("morton_code", DataType::UInt64, false),
            Field::new("minx", DataType::Float64, false),
            Field::new("miny", DataType::Float64, false),
            Field::new("maxx", DataType::Float64, false),
            Field::new("maxy", DataType::Float64, false),
            Field::new("lon", DataType::Float32, true), // Optional
            Field::new("lat", DataType::Float32, true), // Optional
        ]));

        // Extract data into columnar arrays
        let row_ids: Vec<u64> = entries.iter().map(|e| e.row_id).collect();
        let morton_codes: Vec<u64> = entries.iter().map(|e| e.morton_code).collect();
        let minx: Vec<f64> = entries.iter().map(|e| e.minx).collect();
        let miny: Vec<f64> = entries.iter().map(|e| e.miny).collect();
        let maxx: Vec<f64> = entries.iter().map(|e| e.maxx).collect();
        let maxy: Vec<f64> = entries.iter().map(|e| e.maxy).collect();
        let lon: Vec<Option<f32>> = entries.iter().map(|e| e.lon).collect();
        let lat: Vec<Option<f32>> = entries.iter().map(|e| e.lat).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(row_ids)),
                Arc::new(UInt64Array::from(morton_codes)),
                Arc::new(Float64Array::from(minx)),
                Arc::new(Float64Array::from(miny)),
                Arc::new(Float64Array::from(maxx)),
                Arc::new(Float64Array::from(maxy)),
                Arc::new(Float32Array::from(lon)),
                Arc::new(Float32Array::from(lat)),
            ],
        ).map_err(|e| lance_core::Error::Arrow {
            message: format!("Failed to create tile entries batch: {}", e),
            location: location!(),
        })?;

        let tile_file_name = format!("tile_{:06}.entries.lance", tile_id);
        let mut writer = self.store.new_index_file(&tile_file_name, schema).await?;
            writer.write_record_batch(batch).await?;
            writer.finish().await?;
    
            Ok(())
        }
    
    /// Store manifest entries to morton.lance
    async fn store_manifest(&self, manifest_entries: &[TileManifestEntry]) -> Result<()> {
        use arrow_array::{Float64Array, UInt64Array, RecordBatch};
            use arrow_schema::{DataType, Field, Schema};
    
            let schema = Arc::new(Schema::new(vec![
            Field::new("tile_id", DataType::UInt64, false),
            Field::new("minx", DataType::Float64, false),
            Field::new("miny", DataType::Float64, false),
            Field::new("maxx", DataType::Float64, false),
            Field::new("maxy", DataType::Float64, false),
        ]));

        let tile_ids: Vec<u64> = manifest_entries.iter().map(|e| e.tile_id).collect();
        let minxs: Vec<f64> = manifest_entries.iter().map(|e| e.minx).collect();
        let minys: Vec<f64> = manifest_entries.iter().map(|e| e.miny).collect();
        let maxxs: Vec<f64> = manifest_entries.iter().map(|e| e.maxx).collect();
        let maxys: Vec<f64> = manifest_entries.iter().map(|e| e.maxy).collect();
    
            let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(tile_ids)),
                Arc::new(Float64Array::from(minxs)),
                Arc::new(Float64Array::from(minys)),
                Arc::new(Float64Array::from(maxxs)),
                Arc::new(Float64Array::from(maxys)),
            ],
            ).map_err(|e| lance_core::Error::Arrow {
            message: format!("Failed to create manifest batch: {}", e),
                location: location!(),
            })?;
    
        let mut writer = self.store.new_index_file("morton.lance", schema).await?;
        writer.write_record_batch(batch).await?;
        writer.finish().await?;

        Ok(())
    }

    /// Load manifest from morton.lance
    pub async fn load_manifest(&self) -> Result<Vec<TileManifestEntry>> {
        // Load from storage
        let reader = self.store.open_index_file("morton.lance").await?;
        let batch = reader.read_range(0..reader.num_rows(), None).await?;
        
        let manifest_entries = self.deserialize_manifest(batch)?;
        Ok(manifest_entries)
    }

    /// Get relevant tiles for routing (by bbox overlap and/or z-range)
    pub async fn get_relevant_tiles(&self, query: &SpatialQuery) -> Result<Vec<TileManifestEntry>> {
        // Support both ST_Within and ST_Intersects (both use bounding box intersection for routing)
        let bbox = match query {
            SpatialQuery::Within(b) => b,
            SpatialQuery::Intersects(b) => b,
            _ => {
                return Err(lance_core::Error::NotSupported {
                    source: "Only SpatialQuery::Within and SpatialQuery::Intersects are supported".into(),
                    location: location!(),
                });
            }
        };

        let manifest_entries = self.load_manifest().await?;

        // Build a temporary R-tree over manifest entries for routing (no caching)
        let manifest_rtree = RTree::bulk_load(manifest_entries.clone());
        let query_aabb = bbox.to_aabb();
        let relevant_tiles: Vec<TileManifestEntry> = manifest_rtree
            .locate_in_envelope_intersecting(&query_aabb)
            .cloned()
            .collect();
        println!("🔍 [TILED] Found {} relevant tiles to query", relevant_tiles.len());

        Ok(relevant_tiles)
    }

    /// Load a tile's R-tree from storage (online query process)
    pub async fn load_tile_rtree(&self, tile_id: u64) -> Result<Arc<RTree<TiledSpatialEntry>>> {
        // Load tile entries from columnar file
        let tile_file_name = format!("tile_{:06}.entries.lance", tile_id);
        let reader = self.store.open_index_file(&tile_file_name).await?;
        let batch = reader.read_range(0..reader.num_rows(), None).await?;
        
        // Deserialize tile entries from columnar format
        let entries = self.deserialize_tile_entries_columnar(batch)?;
        
        // Build in-memory R-tree from entries via RTree::bulk_load
        let rtree = Arc::new(RTree::bulk_load(entries));
        Ok(rtree)
    }

    /// Deserialize tile entries from columnar format
    fn deserialize_tile_entries_columnar(&self, batch: arrow_array::RecordBatch) -> Result<Vec<TiledSpatialEntry>> {
        use arrow_array::{Float32Array, Float64Array, UInt64Array, Array};

        let row_ids = batch.column(0).as_any().downcast_ref::<UInt64Array>()
            .ok_or_else(|| lance_core::Error::Arrow {
                message: "Failed to downcast row_id column".to_string(),
                location: location!(),
            })?;
        
        let morton_codes = batch.column(1).as_any().downcast_ref::<UInt64Array>()
            .ok_or_else(|| lance_core::Error::Arrow {
                message: "Failed to downcast morton_code column".to_string(),
                location: location!(),
            })?;
    
        let minx_array = batch.column(2).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| lance_core::Error::Arrow {
                message: "Failed to downcast minx column".to_string(),
                location: location!(),
            })?;
        
        let miny_array = batch.column(3).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| lance_core::Error::Arrow {
                message: "Failed to downcast miny column".to_string(),
                location: location!(),
            })?;
    
        let maxx_array = batch.column(4).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| lance_core::Error::Arrow {
                message: "Failed to downcast maxx column".to_string(),
                location: location!(),
            })?;
        
        let maxy_array = batch.column(5).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| lance_core::Error::Arrow {
                message: "Failed to downcast maxy column".to_string(),
                location: location!(),
            })?;
        
        let lon_array = batch.column(6).as_any().downcast_ref::<Float32Array>()
                .ok_or_else(|| lance_core::Error::Arrow {
                message: "Failed to downcast lon column".to_string(),
                    location: location!(),
                })?;
    
        let lat_array = batch.column(7).as_any().downcast_ref::<Float32Array>()
            .ok_or_else(|| lance_core::Error::Arrow {
                message: "Failed to downcast lat column".to_string(),
                location: location!(),
            })?;
    
        let mut entries = Vec::new();
        for i in 0..batch.num_rows() {
            let entry = TiledSpatialEntry {
                row_id: row_ids.value(i),
                morton_code: morton_codes.value(i),
                minx: minx_array.value(i),
                miny: miny_array.value(i),
                maxx: maxx_array.value(i),
                maxy: maxy_array.value(i),
                lon: if lon_array.is_null(i) { None } else { Some(lon_array.value(i)) },
                lat: if lat_array.is_null(i) { None } else { Some(lat_array.value(i)) },
            };
            entries.push(entry);
        }

        Ok(entries)
    }

    /// Deserialize manifest from columnar format
    fn deserialize_manifest(&self, batch: arrow_array::RecordBatch) -> Result<Vec<TileManifestEntry>> {
        use arrow_array::{Float64Array, UInt64Array, Array};

        let tile_ids = batch.column(0).as_any().downcast_ref::<UInt64Array>()
            .ok_or_else(|| lance_core::Error::Arrow {
                message: "Failed to downcast tile_id column".to_string(),
                location: location!(),
            })?;

        let minxs = batch.column(1).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| lance_core::Error::Arrow {
                message: "Failed to downcast minx column".to_string(),
                location: location!(),
            })?;

        let minys = batch.column(2).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| lance_core::Error::Arrow {
                message: "Failed to downcast miny column".to_string(),
                    location: location!(),
            })?;

        let maxxs = batch.column(3).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| lance_core::Error::Arrow {
                message: "Failed to downcast maxx column".to_string(),
                location: location!(),
            })?;

        let maxys = batch.column(4).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| lance_core::Error::Arrow {
                message: "Failed to downcast maxy column".to_string(),
                location: location!(),
            })?;

        let mut entries = Vec::new();
        for i in 0..batch.num_rows() {
            let entry = TileManifestEntry {
                tile_id: tile_ids.value(i),
                minx: minxs.value(i),
                miny: minys.value(i),
                maxx: maxxs.value(i),
                maxy: maxys.value(i),
            };
            entries.push(entry);
        }

        Ok(entries)
    }

    /// Compute bounding box for a set of tile entries
    fn compute_tile_bbox(entries: &[TiledSpatialEntry]) -> BoundingBox {
        if entries.is_empty() {
            return BoundingBox::new(0.0, 0.0, 0.0, 0.0);
        }

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for entry in entries {
            min_x = min_x.min(entry.minx);
            min_y = min_y.min(entry.miny);
            max_x = max_x.max(entry.maxx);
            max_y = max_y.max(entry.maxy);
        }

        BoundingBox::new(min_x, min_y, max_x, max_y)
    }
}

/// Metadata for the tiled R-tree index
#[derive(Debug, Clone, Serialize, Deserialize, DeepSizeOf)]
pub struct TiledRTreeMetadata {
    /// Total number of tiles
    pub num_tiles: u64,
    /// Total number of data entries across all tiles
    pub num_entries: u64,
    /// Configuration used to build this index
    pub config: TiledRTreeConfig,
    /// Total bounds of all geometries
    pub total_bounds: BoundingBox,
}

/// Main tiled R-tree index using Morton/Z-order sorting with columnar storage
#[derive(Debug)]
pub struct TiledRTreeIndex {
    /// Manager for tiles and manifest
    tile_manager: Arc<TileManager>,
    /// Index metadata
    metadata: TiledRTreeMetadata,
}

impl DeepSizeOf for TiledRTreeIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.metadata.deep_size_of_children(context)
    }
}

impl TiledRTreeIndex {
    /// Build the tiled R-tree index from spatial data entries (offline build process)
    pub async fn build_from_entries(
        entries: Vec<crate::rtree::paged_leaf_rtree::SpatialDataEntry>,
        store: Arc<dyn IndexStore>,
        config: TiledRTreeConfig,
    ) -> Result<Self> {
        println!("🔲 Building tiled R-tree with {} entries using Morton/Z-order", entries.len());

        if entries.is_empty() {
            return Err(lance_core::Error::Index {
                message: "Cannot build tiled R-tree from empty entries".to_string(),
                location: location!(),
            });
        }

        // Calculate total bounds for Morton encoding
        let total_bounds = Self::compute_total_bounds(&entries);
        println!("🔲 Total bounds: {:?}", total_bounds);

        // Convert to tiled entries with Morton codes
        let tiled_entries: Vec<TiledSpatialEntry> = entries
            .into_iter()
            .map(|entry| TiledSpatialEntry::new(
                entry.geometry,
                entry.row_id,
                &total_bounds,
                config.morton_depth,
            ))
            .collect();

        println!("🔲 Computed Morton codes for {} entries", tiled_entries.len());

        // Create tile manager and tiles
        let tile_manager = Arc::new(TileManager::new(store.clone(), config.clone()));
        let num_entries_total = tiled_entries.len() as u64;
        let manifest_entries = tile_manager.create_tiles(tiled_entries).await?;

        let metadata = TiledRTreeMetadata {
            num_tiles: manifest_entries.len() as u64,
            num_entries: num_entries_total,
            config,
            total_bounds,
        };

        let index = Self {
            tile_manager,
            metadata: metadata.clone(),
        };

        // Store metadata to enable loading
        index.store_metadata().await?;

        println!("🔲 Tiled R-tree index built successfully: {} tiles, {} entries", 
                 metadata.num_tiles, metadata.num_entries);

        Ok(index)
    }

    /// Perform a spatial query using the manifest for tile routing (online query process)
    pub async fn query(&self, query: &SpatialQuery) -> Result<Vec<u64>> {
        // Support both ST_Within and ST_Intersects (both use bounding box intersection)
        match query {
            SpatialQuery::Within(b) => {
                println!("🔍 [TILED] Starting ST_Within query: {:?}", b);
            },
            SpatialQuery::Intersects(b) => {
                println!("🔍 [TILED] Starting ST_Intersects query: {:?}", b);
            },
            _ => {
                return Err(lance_core::Error::NotSupported {
                    source: "Only SpatialQuery::Within and SpatialQuery::Intersects are supported".into(),
                    location: location!(),
                });
            }
        };
        println!("🔍 [TILED] Index has {} tiles, {} total entries", 
                 self.metadata.num_tiles, self.metadata.num_entries);

        // Use manifest to route to relevant tiles (by bbox overlap and/or z-range)
        let relevant_tiles = self.tile_manager.get_relevant_tiles(query).await?;
        println!("🔍 [TILED] Found {} relevant tiles to query", relevant_tiles.len());

        let mut results = Vec::new();

        // For each tile you touch: read *.entries.lance → build in-memory R-tree → query
        for tile in relevant_tiles {
            println!("🔍 [TILED] Querying tile {}", tile.tile_id);
            
            // Load tile's R-tree (cached Arc<RTree<…>> by tile_id)
            let tile_rtree = self.tile_manager.load_tile_rtree(tile.tile_id).await?;
            
            // Query the tile's R-tree (locate_in_envelope_intersecting) and optionally refine
            let tile_results = self.query_tile_rtree(&tile_rtree, query).await?;
            println!("🔍 [TILED] Tile {} returned {} results", tile.tile_id, tile_results.len());
            
            results.extend(tile_results);
        }

        println!("🔍 [TILED] Query complete: {} total results", results.len());
        Ok(results)
    }

    /// Query a specific tile's R-tree
    async fn query_tile_rtree(&self, tile_rtree: &RTree<TiledSpatialEntry>, query: &SpatialQuery) -> Result<Vec<u64>> {
        let bbox = match query {
            SpatialQuery::Within(b) | SpatialQuery::Intersects(b) => b,
            _ => unreachable!("query validated to Within/Intersects before calling"),
        };

        let query_aabb = bbox.to_aabb();
        let matching_entries: Vec<&TiledSpatialEntry> = tile_rtree.locate_in_envelope(&query_aabb).collect();

        let mut results = Vec::new();
        for entry in matching_entries {
            // For points, locate_in_envelope with bbox ensures intersection/within
            results.push(entry.row_id);
        }

        Ok(results)
    }

    /// Compute total bounding box for all entries
    fn compute_total_bounds(entries: &[crate::rtree::paged_leaf_rtree::SpatialDataEntry]) -> BoundingBox {
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for entry in entries {
            let bbox = entry.bbox();
            min_x = min_x.min(bbox.min_x);
            min_y = min_y.min(bbox.min_y);
            max_x = max_x.max(bbox.max_x);
            max_y = max_y.max(bbox.max_y);
        }

        BoundingBox::new(min_x, min_y, max_x, max_y)
    }

    pub fn metadata(&self) -> &TiledRTreeMetadata {
        &self.metadata
    }

    pub fn size(&self) -> usize {
        self.metadata.num_entries as usize
    }

    pub fn total_bounds(&self) -> BoundingBox {
        self.metadata.total_bounds.clone()
    }

    /// Store metadata to storage (similar to PagedLeafRTreeIndex)
    async fn store_metadata(&self) -> Result<()> {
        use arrow_array::{BinaryArray, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};

        // Serialize metadata using serde_json
        let metadata_bytes = serde_json::to_vec(&self.metadata).map_err(|e| lance_core::Error::Arrow {
            message: format!("Failed to serialize tiled metadata: {}", e),
            location: location!(),
        })?;

        // Create schema and batch for metadata
        let schema = Arc::new(Schema::new(vec![
            Field::new("metadata", DataType::Binary, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(BinaryArray::from_iter_values([metadata_bytes]))],
        ).map_err(|e| lance_core::Error::Arrow {
            message: format!("Failed to create tiled metadata batch: {}", e),
            location: location!(),
        })?;

        // Store metadata
        let mut writer = self.tile_manager.store.new_index_file("metadata.lance", schema).await?;
        writer.write_record_batch(batch).await?;
        writer.finish().await?;

        println!("🔲 Stored tiled R-tree metadata");
        Ok(())
    }

    /// Load metadata from storage
    async fn load_metadata(store: &dyn IndexStore) -> Result<TiledRTreeMetadata> {
        use arrow_array::BinaryArray;

        let reader = store.open_index_file("metadata.lance").await?;
        let batch = reader.read_range(0..reader.num_rows(), None).await?;

        // Extract the serialized metadata
        let metadata_data = batch.column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| lance_core::Error::Arrow {
                message: "Failed to downcast metadata column".to_string(),
                location: location!(),
            })?;

        let metadata_bytes = metadata_data.value(0);
        let metadata: TiledRTreeMetadata = serde_json::from_slice(metadata_bytes).map_err(|e| lance_core::Error::Arrow {
            message: format!("Failed to deserialize tiled metadata: {}", e),
            location: location!(),
        })?;

        Ok(metadata)
    }
}

/// Implement GeoIndex trait for TiledRTreeIndex
#[async_trait]
impl GeoIndex for TiledRTreeIndex {
    async fn search(
        &self,
        query: &SpatialQuery,
        _metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let row_ids = self.query(query).await?;
        let row_id_map = RowIdTreeMap::from_iter(row_ids);
        Ok(SearchResult::Exact(row_id_map))
    }

    fn can_answer_exact(&self, query: &SpatialQuery) -> bool {
        matches!(query, SpatialQuery::Within(_) | SpatialQuery::Intersects(_))
    }

    async fn load(store: Arc<dyn IndexStore>) -> Result<Arc<Self>>
    where
        Self: Sized,
    {
        println!("🌍 Loading TiledRTreeIndex from IndexStore");
        
        // Load metadata
        let metadata = Self::load_metadata(store.as_ref()).await?;
        println!("🌍 Loaded tiled metadata: {} tiles, {} entries", 
                 metadata.num_tiles, metadata.num_entries);
        
        // Create tile manager with the loaded config
        let tile_manager = Arc::new(TileManager::new(store.clone(), metadata.config.clone()));
        
        let index = Self {
            tile_manager,
            metadata,
        };
        
        println!("🌍 ✅ Successfully loaded tiled R-tree index");
        Ok(Arc::new(index))
    }

    fn total_bounds(&self) -> Option<BoundingBox> {
        Some(self.total_bounds())
    }

    fn size(&self) -> usize {
        self.size()
    }
}

/// Implement Index trait for TiledRTreeIndex
#[async_trait]
impl Index for TiledRTreeIndex {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(lance_core::Error::NotSupported {
            source: "TiledRTreeIndex cannot be cast to VectorIndex".into(),
            location: location!(),
        })
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let stats = serde_json::json!({
            "index_type": "TiledSpatial",
            "num_tiles": self.metadata.num_tiles,
            "num_entries": self.metadata.num_entries,
            "total_bounds": self.metadata.total_bounds,
            "config": {
                "entries_per_tile": self.metadata.config.entries_per_tile,
                "morton_depth": self.metadata.config.morton_depth
            }
        });
        Ok(stats)
    }

    async fn prewarm(&self) -> Result<()> {
        Ok(())
    }

    fn index_type(&self) -> IndexType {
        IndexType::RTree
    }

    async fn calculate_included_frags(&self) -> Result<roaring::RoaringBitmap> {
        let mut bitmap = roaring::RoaringBitmap::new();
        bitmap.insert(0);
        Ok(bitmap)
    }
}

/// Implement ScalarIndex trait for TiledRTreeIndex
#[async_trait]
impl ScalarIndex for TiledRTreeIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        if let Some(spatial_query) = query.as_any().downcast_ref::<SpatialQuery>() {
            GeoIndex::search(self, spatial_query, metrics).await
        } else {
            Err(lance_core::Error::NotSupported {
                source: "TiledRTreeIndex only supports SpatialQuery types".into(),
                location: location!(),
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
        _index_cache: lance_core::cache::LanceCache,
    ) -> Result<Arc<Self>>
    where
        Self: Sized,
    {
        GeoIndex::load(store).await
    }

    async fn remap(
        &self,
        _mapping: &HashMap<u64, Option<u64>>,
        _dest_store: &dyn IndexStore,
    ) -> Result<()> {
        Err(lance_core::Error::NotSupported {
            source: "TiledRTreeIndex remap not yet implemented".into(),
            location: location!(),
        })
    }

    async fn update(
        &self,
        _new_data: SendableRecordBatchStream,
        _dest_store: &dyn IndexStore,
    ) -> Result<()> {
        Err(lance_core::Error::NotSupported {
            source: "TiledRTreeIndex update not yet implemented".into(),
            location: location!(),
        })
    }
}
