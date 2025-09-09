//! Geo index training/building functionality

use std::sync::Arc;

use arrow_array::Array;
use futures::TryStreamExt;

use crate::rtree::{GeoIndexParams, SpatialGeometry, Point};
use crate::rtree::paged_leaf_rtree::{PagedLeafRTreeIndex, PagedLeafConfig, SpatialDataEntry};
use crate::scalar::{IndexStore, btree::TrainingSource};
use lance_core::{Error, Result};

/// Train an rtree index from a TrainingSource (consistent with other index types)
pub async fn train_rtree_index(
    data_source: Box<dyn TrainingSource + Send>,
    store: Arc<dyn IndexStore>,
    _params: &GeoIndexParams,
) -> Result<()> {
    println!("🌲 Starting GEO INDEX TRAINING (Paged Leaf R-tree)");
    let mut spatial_entries: Vec<SpatialDataEntry> = Vec::new();
    let mut batches_source = data_source.scan_unordered_chunks(4096).await?;
    
    while let Some(batch) = batches_source.try_next().await? {
        println!("🌲 Processing batch with {} rows", batch.num_rows());
        
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
        
        // Extract geometries from the geometry column (world-class approach!)
        let geometries = extract_geometries(geometry_field, geometry_column.as_ref())?;
        println!("🌲 Extracted {} geometries", geometries.len());
        
        // Create spatial data entries with ACTUAL row IDs from the batch
        for (i, geometry) in geometries.iter().enumerate() {
            let actual_row_id = row_id_column.value(i);
            spatial_entries.push(SpatialDataEntry::new(geometry.clone(), actual_row_id));
        }
    }
    
    println!("🌲 Building paged leaf R-tree with {} entries", spatial_entries.len());
    
    // Build using the paged leaf R-tree
    let config = PagedLeafConfig::default();
    let _index = PagedLeafRTreeIndex::build_from_entries(spatial_entries, store, config).await?;
    
    println!("🌲 GEO INDEX TRAINING COMPLETE - Paged leaf R-tree built and stored successfully");
    Ok(())
}


/// Extract geometries directly from geometry columns (world-class approach!)
/// 
/// This function handles various GeoArrow formats and creates proper geometry objects
/// instead of just extracting bounding boxes.
pub fn extract_geometries(geometry_field: &arrow_schema::Field, geometry_array: &dyn Array) -> Result<Vec<SpatialGeometry>> {
    use arrow_schema::DataType;
    
    // Check for GeoArrow extension metadata first
    if let Some(extension_name) = geometry_field.metadata().get("ARROW:extension:name") {
        println!("🌍 Found GeoArrow extension: {}", extension_name);
        
        match extension_name.as_str() {
            "geoarrow.point" => extract_geoarrow_points(geometry_array),
            _ => Err(Error::Index {
                message: format!("Unsupported GeoArrow extension: {}. Only geoarrow.point is currently supported.", extension_name),
                location: snafu::location!(),
            }),
        }
    } else {
        // Fallback: check for struct<x: double, y: double> format (legacy point data)
        match geometry_field.data_type() {
            DataType::Struct(fields) => {
                let field_names: Vec<&str> = fields.iter().map(|f| f.name().as_str()).collect();
                
                if field_names.len() == 2 && field_names.contains(&"x") && field_names.contains(&"y") {
                    println!("🌍 Found struct<x,y> point format (legacy)");
                    extract_points_from_struct(geometry_array)
                } else {
                    Err(Error::Index {
                        message: format!("Unsupported struct format. Expected struct with x,y fields, got fields: {:?}", field_names),
                        location: snafu::location!(),
                    })
                }
            }
            _ => Err(Error::Index {
                message: format!("Unsupported geometry data type: {:?}. Expected GeoArrow extension or struct<x: double, y: double>", geometry_field.data_type()),
                location: snafu::location!(),
            }),
        }
    }
}

/// Extract GeoArrow points (proper extension format)
fn extract_geoarrow_points(array: &dyn Array) -> Result<Vec<SpatialGeometry>> {
    use arrow_array::StructArray;
    
    let struct_array = array.as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| Error::Index {
            message: "Expected StructArray for GeoArrow point data".to_string(),
            location: snafu::location!(),
        })?;
    
    // Get x and y columns from the struct
    let x_column = struct_array.column_by_name("x")
        .ok_or_else(|| Error::Index {
            message: "GeoArrow Point struct must have 'x' field".to_string(),
            location: snafu::location!(),
        })?;
        
    let y_column = struct_array.column_by_name("y")
        .ok_or_else(|| Error::Index {
            message: "GeoArrow Point struct must have 'y' field".to_string(),
            location: snafu::location!(),
        })?;

    extract_points_from_coordinates(x_column.as_ref(), y_column.as_ref())
}

/// Extract points from struct<x,y> format (legacy support)
fn extract_points_from_struct(array: &dyn Array) -> Result<Vec<SpatialGeometry>> {
    use arrow_array::StructArray;
    
    let struct_array = array.as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| Error::Index {
            message: "Expected StructArray for point data".to_string(),
            location: snafu::location!(),
        })?;
    
    // Get x and y columns from the struct
    let x_column = struct_array.column_by_name("x")
        .ok_or_else(|| Error::Index {
            message: "Point struct must have 'x' field".to_string(),
            location: snafu::location!(),
        })?;
        
    let y_column = struct_array.column_by_name("y")
        .ok_or_else(|| Error::Index {
            message: "Point struct must have 'y' field".to_string(),
            location: snafu::location!(),
        })?;

    extract_points_from_coordinates(x_column.as_ref(), y_column.as_ref())
}


/// Extract points from coordinate arrays (the world-class way!)
fn extract_points_from_coordinates(x_array: &dyn Array, y_array: &dyn Array) -> Result<Vec<SpatialGeometry>> {
    use arrow_array::{Float64Array, Float32Array};
    
    let len = x_array.len().min(y_array.len());
    let mut geometries = Vec::with_capacity(len);
    
    // Handle Float64 coordinates
    if let (Some(x_f64), Some(y_f64)) = (
        x_array.as_any().downcast_ref::<Float64Array>(),
        y_array.as_any().downcast_ref::<Float64Array>()
    ) {
        for i in 0..len {
            if x_f64.is_null(i) || y_f64.is_null(i) {
                continue;
            }
            
            let x = x_f64.value(i);
            let y = y_f64.value(i);
            
            // Create actual Point geometry (not a degenerate bounding box!)
            let point = Point::new(x, y);
            geometries.push(SpatialGeometry::Point(point));
        }
    }
    // Handle Float32 coordinates
    else if let (Some(x_f32), Some(y_f32)) = (
        x_array.as_any().downcast_ref::<Float32Array>(),
        y_array.as_any().downcast_ref::<Float32Array>()
    ) {
        for i in 0..len {
            if x_f32.is_null(i) || y_f32.is_null(i) {
                continue;
            }
            
            let x = x_f32.value(i) as f64;
            let y = y_f32.value(i) as f64;
            
            // Create actual Point geometry
            let point = Point::new(x, y);
            geometries.push(SpatialGeometry::Point(point));
        }
    }
    else {
        return Err(Error::Index {
            message: "Unsupported coordinate data type (expected Float32 or Float64)".to_string(),
            location: snafu::location!(),
        });
    }
    
    Ok(geometries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rtree::BoundingBox;
    use arrow_array::{Float64Array, StructArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;
    
    #[test]
    fn test_extract_points_from_coordinates() {
        let x_array = Float64Array::from(vec![1.0, 2.0, 3.0]);
        let y_array = Float64Array::from(vec![4.0, 5.0, 6.0]);
        
        let geometries = extract_points_from_coordinates(&x_array, &y_array).unwrap();
        
        assert_eq!(geometries.len(), 3);
        assert_eq!(geometries[0], SpatialGeometry::Point(Point::new(1.0, 4.0)));
        assert_eq!(geometries[1], SpatialGeometry::Point(Point::new(2.0, 5.0)));
        assert_eq!(geometries[2], SpatialGeometry::Point(Point::new(3.0, 6.0)));
        
        // Test that bounding boxes are computed correctly
        let bbox0 = geometries[0].envelope();
        assert_eq!(bbox0, BoundingBox::new(1.0, 4.0, 1.0, 4.0));
    }
    
    #[test]
    fn test_extract_geometries_from_geoarrow_point() {
        use std::collections::HashMap;
        
        let x_array = Arc::new(Float64Array::from(vec![1.0, 2.0]));
        let y_array = Arc::new(Float64Array::from(vec![3.0, 4.0]));
        
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]));
        
        let struct_array = StructArray::new(
            schema.fields().clone(),
            vec![x_array, y_array],
            None,
        );
        
        // Create a GeoArrow Point field with proper extension metadata
        let mut metadata = HashMap::new();
        metadata.insert("ARROW:extension:name".to_string(), "geoarrow.point".to_string());
        let geoarrow_field = Field::new("geometry", struct_array.data_type().clone(), false)
            .with_metadata(metadata);
        
        let geometries = extract_geometries(&geoarrow_field, &struct_array).unwrap();
        
        assert_eq!(geometries.len(), 2);
        assert_eq!(geometries[0], SpatialGeometry::Point(Point::new(1.0, 3.0)));
        assert_eq!(geometries[1], SpatialGeometry::Point(Point::new(2.0, 4.0)));
        
        // Test that bounding boxes are computed correctly on demand
        let bbox0 = geometries[0].envelope();
        let bbox1 = geometries[1].envelope();
        assert_eq!(bbox0, BoundingBox::new(1.0, 3.0, 1.0, 3.0));
        assert_eq!(bbox1, BoundingBox::new(2.0, 4.0, 2.0, 4.0));
    }
}