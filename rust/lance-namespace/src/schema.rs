//! Schema conversion utilities for Lance Namespace.
//!
//! This module provides functions to convert between JsonArrow schema representations
//! and Arrow schema types.

use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use lance_namespace_reqwest_client::models::{JsonArrowDataType, JsonArrowField, JsonArrowSchema};

use crate::namespace::{NamespaceError, Result};

/// Convert JsonArrowSchema to Arrow Schema
pub fn convert_json_arrow_schema(json_schema: &JsonArrowSchema) -> Result<ArrowSchema> {
    let fields: Result<Vec<Field>> = json_schema
        .fields
        .iter()
        .map(|f| convert_json_arrow_field(f))
        .collect();

    let metadata = json_schema
        .metadata
        .as_ref()
        .map(|m| m.clone())
        .unwrap_or_default();

    Ok(ArrowSchema::new_with_metadata(fields?, metadata))
}

/// Convert JsonArrowField to Arrow Field
pub fn convert_json_arrow_field(json_field: &JsonArrowField) -> Result<Field> {
    let data_type = convert_json_arrow_type(&json_field.r#type)?;
    let nullable = json_field.nullable;

    Ok(Field::new(&json_field.name, data_type, nullable))
}

/// Convert JsonArrowDataType to Arrow DataType
pub fn convert_json_arrow_type(json_type: &JsonArrowDataType) -> Result<DataType> {
    let type_name = json_type.r#type.to_lowercase();

    match type_name.as_str() {
        "null" => Ok(DataType::Null),
        "bool" | "boolean" => Ok(DataType::Boolean),
        "int8" => Ok(DataType::Int8),
        "uint8" => Ok(DataType::UInt8),
        "int16" => Ok(DataType::Int16),
        "uint16" => Ok(DataType::UInt16),
        "int32" => Ok(DataType::Int32),
        "uint32" => Ok(DataType::UInt32),
        "int64" => Ok(DataType::Int64),
        "uint64" => Ok(DataType::UInt64),
        "float32" => Ok(DataType::Float32),
        "float64" => Ok(DataType::Float64),
        "utf8" => Ok(DataType::Utf8),
        "binary" => Ok(DataType::Binary),
        _ => Err(NamespaceError::Other(format!(
            "Unsupported Arrow type: {}",
            type_name
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_convert_basic_types() {
        // Test int32
        let int_type = JsonArrowDataType::new("int32".to_string());
        let result = convert_json_arrow_type(&int_type).unwrap();
        assert_eq!(result, DataType::Int32);

        // Test utf8
        let string_type = JsonArrowDataType::new("utf8".to_string());
        let result = convert_json_arrow_type(&string_type).unwrap();
        assert_eq!(result, DataType::Utf8);

        // Test float64
        let float_type = JsonArrowDataType::new("float64".to_string());
        let result = convert_json_arrow_type(&float_type).unwrap();
        assert_eq!(result, DataType::Float64);

        // Test binary
        let binary_type = JsonArrowDataType::new("binary".to_string());
        let result = convert_json_arrow_type(&binary_type).unwrap();
        assert_eq!(result, DataType::Binary);
    }

    #[test]
    fn test_convert_field() {
        let int_type = JsonArrowDataType::new("int32".to_string());
        let field = JsonArrowField {
            name: "test_field".to_string(),
            r#type: Box::new(int_type),
            nullable: false,
            metadata: None,
        };

        let result = convert_json_arrow_field(&field).unwrap();
        assert_eq!(result.name(), "test_field");
        assert_eq!(result.data_type(), &DataType::Int32);
        assert!(!result.is_nullable());
    }

    #[test]
    fn test_convert_schema() {
        let int_type = JsonArrowDataType::new("int32".to_string());
        let string_type = JsonArrowDataType::new("utf8".to_string());

        let id_field = JsonArrowField {
            name: "id".to_string(),
            r#type: Box::new(int_type),
            nullable: false,
            metadata: None,
        };

        let name_field = JsonArrowField {
            name: "name".to_string(),
            r#type: Box::new(string_type),
            nullable: true,
            metadata: None,
        };

        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), "value".to_string());

        let schema = JsonArrowSchema {
            fields: vec![id_field, name_field],
            metadata: Some(metadata.clone()),
        };

        let result = convert_json_arrow_schema(&schema).unwrap();
        assert_eq!(result.fields().len(), 2);
        assert_eq!(result.field(0).name(), "id");
        assert_eq!(result.field(1).name(), "name");
        assert_eq!(result.metadata(), &metadata);
    }

    #[test]
    fn test_unsupported_type() {
        let unsupported_type = JsonArrowDataType::new("unsupported".to_string());
        let result = convert_json_arrow_type(&unsupported_type);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unsupported Arrow type"));
    }
}
