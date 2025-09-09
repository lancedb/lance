//! Geo index training/building functionality

use std::sync::Arc;

use arrow_array::Array;
use datafusion::execution::SendableRecordBatchStream;
use futures::TryStreamExt;

use crate::rtree::{BoundingBox, GeoIndexParams};
use crate::rtree::simple_rtree::RTreeEntry;
use crate::scalar::IndexStore;
use lance_core::{Error, Result};

/// Train a geo index from a stream of record batches
pub async fn train_geo_index(
    mut data_source: SendableRecordBatchStream,
    store: Arc<dyn IndexStore>,
    _params: &GeoIndexParams,
) -> Result<()> {
    println!("🌍 Starting GEO INDEX TRAINING");
    
    let mut entries: Vec<RTreeEntry> = Vec::new();
    let mut row_id = 0u64;
    
    while let Some(batch) = data_source.try_next().await? {
        println!("🌍 Processing batch with {} rows", batch.num_rows());
        
        // Get the geometry column (should be the first column)
        if batch.num_columns() == 0 {
            continue;
        }
        
        let geometry_column = batch.column(0);
        let schema = batch.schema();
        let geometry_field = schema.field(0);
        
        // Extract bounding boxes from the geometry column
        let bboxes = extract_bounding_boxes(geometry_field, geometry_column.as_ref())?;
        println!("🌍 Extracted {} bounding boxes", bboxes.len());
        
        // Create R-tree entries
        for bbox in bboxes {
            entries.push(RTreeEntry { bbox, row_id });
            row_id += 1;
        }
    }
    
    println!("🌍 Building R-tree with {} entries", entries.len());
    
    // Save R-tree data using Lance's IndexStore infrastructure
    save_rtree_to_index_store(&entries, store.as_ref()).await?;
    
    println!("🌍 GEO INDEX TRAINING COMPLETE - Index saved successfully");
    Ok(())
}

/// Save R-tree entries using binary serialization
async fn save_rtree_to_index_store(entries: &[RTreeEntry], index_store: &dyn IndexStore) -> Result<()> {
    use arrow_array::{BinaryArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;
    
    println!("🌍 Saving R-tree entries using binary serialization");
    
    // Serialize entries using serde_json (could also use bincode for better performance)
    let serialized = serde_json::to_vec(entries).map_err(|e| Error::Arrow {
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
        vec![Arc::new(BinaryArray::from_iter_values([serialized]))],
    ).map_err(|e| Error::Arrow {
        message: format!("Failed to create record batch: {}", e),
        location: snafu::location!(),
    })?;

    // Save using IndexStore with standard Lance file names
    let mut geo_data_file = index_store
        .new_index_file("page_data.lance", schema.clone())
        .await?;
    geo_data_file.write_record_batch(batch).await?;
    geo_data_file.finish().await?;

    println!("🌍 R-tree data saved to IndexStore successfully using binary serialization");
    
    
    // Create a lookup file (required by Lance index system)
    let lookup_schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("num_entries", arrow_schema::DataType::UInt64, false),
    ]));
    
    let lookup_batch = arrow_array::RecordBatch::try_new(
        lookup_schema.clone(),
        vec![Arc::new(arrow_array::UInt64Array::from(vec![entries.len() as u64]))],
    ).map_err(|e| Error::Arrow { 
        message: format!("Failed to create lookup batch: {}", e),
        location: snafu::location!(),
    })?;
    
    let mut geo_lookup_file = index_store
        .new_index_file("page_lookup.lance", lookup_schema)
        .await?;
    geo_lookup_file.write_record_batch(lookup_batch).await?;
    geo_lookup_file.finish().await?;
    
    println!("🌍 R-tree data saved to IndexStore successfully using binary serialization");
    Ok(())
}

/// Extract bounding boxes from geometry columns
/// 
/// This function handles struct<x: double, y: double> format for point data
/// and extracts bounding boxes for spatial indexing.
pub fn extract_bounding_boxes(geometry_field: &arrow_schema::Field, geometry_array: &dyn Array) -> Result<Vec<BoundingBox>> {
    use arrow_schema::DataType;
    
    match geometry_field.data_type() {
        DataType::Struct(fields) => {
            // Check if it's a struct with x, y fields (point data)
            let field_names: Vec<&str> = fields.iter().map(|f| f.name().as_str()).collect();
            
            if field_names.len() == 2 && field_names.contains(&"x") && field_names.contains(&"y") {
                extract_bboxes_from_point_struct(geometry_array)
            } else {
                Err(Error::Index {
                    message: format!("Unsupported struct format. Expected struct with x,y fields, got fields: {:?}", field_names),
                    location: snafu::location!(),
                })
            }
        }
        _ => Err(Error::Index {
            message: format!("Unsupported geometry data type: {:?}. Expected struct<x: double, y: double>", geometry_field.data_type()),
            location: snafu::location!(),
        }),
    }
}

fn extract_bboxes_from_point_struct(array: &dyn Array) -> Result<Vec<BoundingBox>> {
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

    extract_point_bboxes(x_column.as_ref(), y_column.as_ref())
}

fn extract_point_bboxes(x_array: &dyn Array, y_array: &dyn Array) -> Result<Vec<BoundingBox>> {
    use arrow_array::{Float64Array, Float32Array};
    
    let len = x_array.len().min(y_array.len());
    let mut bboxes = Vec::with_capacity(len);
    
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
            
            // For points, bounding box is just the point itself
            bboxes.push(BoundingBox::new(x, y, x, y));
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
            
            bboxes.push(BoundingBox::new(x, y, x, y));
        }
    }
    else {
        return Err(Error::Index {
            message: "Unsupported coordinate data type (expected Float32 or Float64)".to_string(),
            location: snafu::location!(),
        });
    }
    
    Ok(bboxes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float64Array, StructArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;
    
    #[test]
    fn test_extract_point_bboxes() {
        let x_array = Float64Array::from(vec![1.0, 2.0, 3.0]);
        let y_array = Float64Array::from(vec![4.0, 5.0, 6.0]);
        
        let bboxes = extract_point_bboxes(&x_array, &y_array).unwrap();
        
        assert_eq!(bboxes.len(), 3);
        assert_eq!(bboxes[0], BoundingBox::new(1.0, 4.0, 1.0, 4.0));
        assert_eq!(bboxes[1], BoundingBox::new(2.0, 5.0, 2.0, 5.0));
        assert_eq!(bboxes[2], BoundingBox::new(3.0, 6.0, 3.0, 6.0));
    }
    
    #[test]
    fn test_extract_bboxes_from_geoarrow_point() {
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
        
        let bboxes = extract_bounding_boxes(&geoarrow_field, &struct_array).unwrap();
        
        assert_eq!(bboxes.len(), 2);
        assert_eq!(bboxes[0], BoundingBox::new(1.0, 3.0, 1.0, 3.0));
        assert_eq!(bboxes[1], BoundingBox::new(2.0, 4.0, 2.0, 4.0));
    }
}