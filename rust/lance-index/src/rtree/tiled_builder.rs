//! Tiled R-tree index training/building functionality
//!
//! This builder creates a tiled R-tree index using Morton/Z-order sorting for optimal
//! spatial locality. The build process is:
//! 1. Extract geometries and compute Morton codes
//! 2. Sort by Morton codes for spatial clustering
//! 3. Cut into tiles with configurable size
//! 4. Write each tile as columnar Lance files
//! 5. Create and store manifest for tile routing

use std::sync::Arc;

use arrow_array::Array;
use futures::TryStreamExt;

use crate::rtree::{GeoIndexParams, SpatialGeometry, Point};
use crate::rtree::tiled_rtree::{TiledRTreeIndex, TiledRTreeConfig};
use crate::scalar::{IndexStore, btree::TrainingSource};
use lance_core::{Error, Result};

/// Train a tiled R-tree index from a TrainingSource
pub async fn train_tiled_rtree_index(
    data_source: Box<dyn TrainingSource + Send>,
    store: Arc<dyn IndexStore>,
    params: &GeoIndexParams,
) -> Result<()> {
    println!("🔲 Starting TILED R-TREE INDEX TRAINING (Morton/Z-order)");
    
    // Create configuration from parameters
    let config = TiledRTreeConfig {
        entries_per_tile: 100,// default for test
        morton_depth: 20, // Default 20 bits per dimension
    };
    
    let mut spatial_entries: Vec<crate::rtree::paged_leaf_rtree::SpatialDataEntry> = Vec::new();
    let mut batches_source = data_source.scan_unordered_chunks(4096).await?;
    
    while let Some(batch) = batches_source.try_next().await? {
        println!("🔲 Processing batch with {} rows", batch.num_rows());
        
        if batch.num_columns() != 2 {
            return Err(Error::Index {
                message: format!("Expected batch with 2 columns (geometry, row_ids), got {}", batch.num_columns()),
                location: snafu::location!(),
            });
        }
        if *batch.column(1).data_type() != arrow_schema::DataType::UInt64 {
            return Err(Error::Index {
                message: format!("Expected row ID column to be UInt64, got {:?}", batch.column(1).data_type()),
                location: snafu::location!(),
            });
        }
        let geometry_column = batch.column(0);
        let row_id_column = batch.column(1)
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| Error::Index {
                message: "Failed to downcast row ID column to UInt64Array".to_string(),
                location: snafu::location!(),
            })?;
        let schema = batch.schema();
        let geometry_field = schema.field(0);
        
        // Extract geometries from the geometry column
        let geometries = extract_geometries(geometry_field, geometry_column.as_ref())?;
        println!("🔲 Extracted {} geometries from batch", geometries.len());
        
        // Create spatial data entries
        for (i, geometry) in geometries.into_iter().enumerate() {
            let row_id = row_id_column.value(i);
            spatial_entries.push(crate::rtree::paged_leaf_rtree::SpatialDataEntry {
                geometry,
                row_id,
            });
        }
    }
    
    println!("🔲 Building tiled R-tree with {} entries using Morton/Z-order", spatial_entries.len());
    
    // Build the tiled R-tree index
    let _tiled_index = TiledRTreeIndex::build_from_entries(
        spatial_entries,
        store,
        config,
    ).await?;
    
    println!("🔲 TILED R-TREE INDEX TRAINING COMPLETE - Index saved successfully");
    Ok(())
}

/// Extract geometries from a geometry column
/// This is a simplified version - in practice you'd handle more geometry types
fn extract_geometries(
    geometry_field: &arrow_schema::Field,
    geometry_column: &dyn Array,
) -> Result<Vec<SpatialGeometry>> {
    let mut geometries = Vec::new();
    
    match geometry_field.data_type() {
        arrow_schema::DataType::Struct(_) => {
            // Handle struct-based geometries (e.g., GeoArrow)
            let struct_array = geometry_column
                .as_any()
                .downcast_ref::<arrow_array::StructArray>()
                .ok_or_else(|| Error::Index {
                    message: "Failed to downcast geometry column to StructArray".to_string(),
                    location: snafu::location!(),
                })?;
            
            // For now, assume point geometries with x, y fields
            if struct_array.num_columns() >= 2 {
                let x_column = struct_array.column(0)
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .ok_or_else(|| Error::Index {
                        message: "Failed to downcast x column to Float64Array".to_string(),
                        location: snafu::location!(),
                    })?;
                
                let y_column = struct_array.column(1)
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .ok_or_else(|| Error::Index {
                        message: "Failed to downcast y column to Float64Array".to_string(),
                        location: snafu::location!(),
                    })?;
                
                for i in 0..struct_array.len() {
                    if !x_column.is_null(i) && !y_column.is_null(i) {
                        let point = Point::new(x_column.value(i), y_column.value(i));
                        geometries.push(SpatialGeometry::Point(point));
                    }
                }
            }
        }
        arrow_schema::DataType::FixedSizeList(_, size) if *size == 2 => {
            // Handle fixed-size list [x, y] format
            let list_array = geometry_column
                .as_any()
                .downcast_ref::<arrow_array::FixedSizeListArray>()
                .ok_or_else(|| Error::Index {
                    message: "Failed to downcast geometry column to FixedSizeListArray".to_string(),
                    location: snafu::location!(),
                })?;
            
            let values = list_array.values()
                .as_any()
                .downcast_ref::<arrow_array::Float64Array>()
                .ok_or_else(|| Error::Index {
                    message: "Failed to downcast list values to Float64Array".to_string(),
                    location: snafu::location!(),
                })?;
            
            for i in 0..list_array.len() {
                if !list_array.is_null(i) {
                    let base_idx = i * 2;
                    let x = values.value(base_idx);
                    let y = values.value(base_idx + 1);
                    let point = Point::new(x, y);
                    geometries.push(SpatialGeometry::Point(point));
                }
            }
        }
        arrow_schema::DataType::List(_) => {
            // Handle variable-length list format
            let list_array = geometry_column
                .as_any()
                .downcast_ref::<arrow_array::ListArray>()
                .ok_or_else(|| Error::Index {
                    message: "Failed to downcast geometry column to ListArray".to_string(),
                    location: snafu::location!(),
                })?;
            
            let values = list_array.values()
                .as_any()
                .downcast_ref::<arrow_array::Float64Array>()
                .ok_or_else(|| Error::Index {
                    message: "Failed to downcast list values to Float64Array".to_string(),
                    location: snafu::location!(),
                })?;
            
            for i in 0..list_array.len() {
                if !list_array.is_null(i) {
                    let start = list_array.value_offsets()[i] as usize;
                    let end = list_array.value_offsets()[i + 1] as usize;
                    
                    if end - start >= 2 {
                        let x = values.value(start);
                        let y = values.value(start + 1);
                        let point = Point::new(x, y);
                        geometries.push(SpatialGeometry::Point(point));
                    }
                }
            }
        }
        _ => {
            return Err(Error::Index {
                message: format!("Unsupported geometry column type: {:?}", geometry_field.data_type()),
                location: snafu::location!(),
            });
        }
    }
    
    Ok(geometries)
}

/// Configuration parameters for tiled R-tree building
#[derive(Debug, Clone)]
pub struct TiledRTreeParams {
    /// Number of entries per tile (affects tile size and query performance)
    pub entries_per_tile: usize,
    /// Morton encoding depth (bits per dimension, affects precision)
    pub morton_depth: u32,
}

impl Default for TiledRTreeParams {
    fn default() -> Self {
        Self {
            entries_per_tile: 1000,
            morton_depth: 20,
        }
    }
}

impl TiledRTreeParams {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_entries_per_tile(mut self, entries_per_tile: usize) -> Self {
        self.entries_per_tile = entries_per_tile;
        self
    }
    
    pub fn with_morton_depth(mut self, morton_depth: u32) -> Self {
        self.morton_depth = morton_depth;
        self
    }
}
