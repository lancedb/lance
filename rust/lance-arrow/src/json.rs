// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! JSON support for Apache Arrow.

use std::convert::TryFrom;
use std::sync::Arc;

use arrow_array::builder::LargeBinaryBuilder;
use arrow_array::{Array, ArrayRef, LargeBinaryArray, LargeStringArray, StringArray};
use arrow_data::ArrayData;
use arrow_schema::{ArrowError, DataType, Field as ArrowField};

use crate::ARROW_EXT_NAME_KEY;

/// Arrow extension type name for JSON data
pub const JSON_EXT_NAME: &str = "lance.json";

/// Check if a field is a JSON extension field
pub fn is_json_field(field: &ArrowField) -> bool {
    field.data_type() == &DataType::LargeBinary
        && field
            .metadata()
            .get(ARROW_EXT_NAME_KEY)
            .map(|name| name == JSON_EXT_NAME)
            .unwrap_or_default()
}

/// Check if a field or any of its descendants is a JSON field
pub fn has_json_fields(field: &ArrowField) -> bool {
    if is_json_field(field) {
        return true;
    }

    match field.data_type() {
        DataType::Struct(fields) => fields.iter().any(|f| has_json_fields(f)),
        DataType::List(f) | DataType::LargeList(f) | DataType::FixedSizeList(f, _) => {
            has_json_fields(f)
        }
        DataType::Map(f, _) => has_json_fields(f),
        _ => false,
    }
}

/// Create a JSON field with the appropriate extension metadata
pub fn json_field(name: &str, nullable: bool) -> ArrowField {
    let mut field = ArrowField::new(name, DataType::LargeBinary, nullable);
    let mut metadata = std::collections::HashMap::new();
    metadata.insert(ARROW_EXT_NAME_KEY.to_string(), JSON_EXT_NAME.to_string());
    field.set_metadata(metadata);
    field
}

/// A specialized array for JSON data stored as JSONB binary format
#[derive(Debug, Clone)]
pub struct JsonArray {
    inner: LargeBinaryArray,
}

impl JsonArray {
    /// Create a new JsonArray from an iterator of JSON strings
    pub fn try_from_iter<I, S>(iter: I) -> Result<Self, ArrowError>
    where
        I: IntoIterator<Item = Option<S>>,
        S: AsRef<str>,
    {
        let mut builder = LargeBinaryBuilder::new();

        for json_str in iter {
            match json_str {
                Some(s) => {
                    let encoded = encode_json(s.as_ref()).map_err(|e| {
                        ArrowError::InvalidArgumentError(format!("Failed to encode JSON: {}", e))
                    })?;
                    builder.append_value(&encoded);
                }
                None => builder.append_null(),
            }
        }

        Ok(Self {
            inner: builder.finish(),
        })
    }

    /// Get the underlying LargeBinaryArray
    pub fn into_inner(self) -> LargeBinaryArray {
        self.inner
    }

    /// Get a reference to the underlying LargeBinaryArray
    pub fn inner(&self) -> &LargeBinaryArray {
        &self.inner
    }

    /// Get the value at index i as decoded JSON string
    pub fn value(&self, i: usize) -> Result<String, ArrowError> {
        if self.inner.is_null(i) {
            return Err(ArrowError::InvalidArgumentError(
                "Value is null".to_string(),
            ));
        }

        let jsonb_bytes = self.inner.value(i);
        decode_json(jsonb_bytes)
            .map_err(|e| ArrowError::InvalidArgumentError(format!("Failed to decode JSON: {}", e)))
    }

    /// Get the value at index i as raw JSONB bytes
    pub fn value_bytes(&self, i: usize) -> &[u8] {
        self.inner.value(i)
    }

    /// Get JSONPath value from the JSON at index i
    pub fn json_path(&self, i: usize, path: &str) -> Result<Option<String>, ArrowError> {
        if self.inner.is_null(i) {
            return Ok(None);
        }

        let jsonb_bytes = self.inner.value(i);
        get_json_path(jsonb_bytes, path).map_err(|e| {
            ArrowError::InvalidArgumentError(format!("Failed to extract JSONPath: {}", e))
        })
    }
}

impl Array for JsonArray {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn to_data(&self) -> ArrayData {
        self.inner.to_data()
    }

    fn into_data(self) -> ArrayData {
        self.inner.into_data()
    }

    fn data_type(&self) -> &DataType {
        &DataType::LargeBinary
    }

    fn slice(&self, offset: usize, length: usize) -> ArrayRef {
        Arc::new(Self {
            inner: self.inner.slice(offset, length),
        })
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn offset(&self) -> usize {
        self.inner.offset()
    }

    fn nulls(&self) -> Option<&arrow_buffer::NullBuffer> {
        self.inner.nulls()
    }

    fn get_buffer_memory_size(&self) -> usize {
        self.inner.get_buffer_memory_size()
    }

    fn get_array_memory_size(&self) -> usize {
        self.inner.get_array_memory_size()
    }
}

// TryFrom implementations for string arrays
impl TryFrom<StringArray> for JsonArray {
    type Error = ArrowError;

    fn try_from(array: StringArray) -> Result<Self, Self::Error> {
        Self::try_from(&array)
    }
}

impl TryFrom<&StringArray> for JsonArray {
    type Error = ArrowError;

    fn try_from(array: &StringArray) -> Result<Self, Self::Error> {
        let mut builder = LargeBinaryBuilder::with_capacity(array.len(), array.value_data().len());

        for i in 0..array.len() {
            if array.is_null(i) {
                builder.append_null();
            } else {
                let json_str = array.value(i);
                let encoded = encode_json(json_str).map_err(|e| {
                    ArrowError::InvalidArgumentError(format!("Failed to encode JSON: {}", e))
                })?;
                builder.append_value(&encoded);
            }
        }

        Ok(Self {
            inner: builder.finish(),
        })
    }
}

impl TryFrom<LargeStringArray> for JsonArray {
    type Error = ArrowError;

    fn try_from(array: LargeStringArray) -> Result<Self, Self::Error> {
        Self::try_from(&array)
    }
}

impl TryFrom<&LargeStringArray> for JsonArray {
    type Error = ArrowError;

    fn try_from(array: &LargeStringArray) -> Result<Self, Self::Error> {
        let mut builder = LargeBinaryBuilder::with_capacity(array.len(), array.value_data().len());

        for i in 0..array.len() {
            if array.is_null(i) {
                builder.append_null();
            } else {
                let json_str = array.value(i);
                let encoded = encode_json(json_str).map_err(|e| {
                    ArrowError::InvalidArgumentError(format!("Failed to encode JSON: {}", e))
                })?;
                builder.append_value(&encoded);
            }
        }

        Ok(Self {
            inner: builder.finish(),
        })
    }
}

/// Encode JSON string to JSONB format
pub fn encode_json(json_str: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let value = jsonb::parse_value(json_str.as_bytes())?;
    Ok(value.to_vec())
}

/// Decode JSONB bytes to JSON string
pub fn decode_json(jsonb_bytes: &[u8]) -> Result<String, Box<dyn std::error::Error>> {
    let raw_jsonb = jsonb::RawJsonb::new(jsonb_bytes);
    Ok(raw_jsonb.to_string())
}

/// Extract JSONPath value from JSONB
fn get_json_path(
    jsonb_bytes: &[u8],
    path: &str,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let json_path = jsonb::jsonpath::parse_json_path(path.as_bytes())?;
    let raw_jsonb = jsonb::RawJsonb::new(jsonb_bytes);

    match raw_jsonb.select_by_path(&json_path) {
        Ok(values) => {
            if values.is_empty() {
                Ok(None)
            } else {
                Ok(Some(values[0].to_string()))
            }
        }
        Err(e) => Err(Box::new(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_field_creation() {
        let field = json_field("data", true);
        assert_eq!(field.name(), "data");
        assert_eq!(field.data_type(), &DataType::LargeBinary);
        assert!(field.is_nullable());
        assert!(is_json_field(&field));
    }

    #[test]
    fn test_json_array_from_strings() {
        let json_strings = vec![
            Some(r#"{"name": "Alice", "age": 30}"#),
            None,
            Some(r#"{"name": "Bob", "age": 25}"#),
        ];

        let array = JsonArray::try_from_iter(json_strings).unwrap();
        assert_eq!(array.len(), 3);
        assert!(!array.is_null(0));
        assert!(array.is_null(1));
        assert!(!array.is_null(2));

        let decoded = array.value(0).unwrap();
        assert!(decoded.contains("Alice"));
    }

    #[test]
    fn test_json_array_from_string_array() {
        let string_array = StringArray::from(vec![
            Some(r#"{"name": "Alice"}"#),
            Some(r#"{"name": "Bob"}"#),
            None,
        ]);

        let json_array = JsonArray::try_from(string_array).unwrap();
        assert_eq!(json_array.len(), 3);
        assert!(!json_array.is_null(0));
        assert!(!json_array.is_null(1));
        assert!(json_array.is_null(2));
    }

    #[test]
    fn test_json_path_extraction() {
        let json_array = JsonArray::try_from_iter(vec![
            Some(r#"{"user": {"name": "Alice", "age": 30}}"#),
            Some(r#"{"user": {"name": "Bob"}}"#),
        ])
        .unwrap();

        let name = json_array.json_path(0, "$.user.name").unwrap();
        assert_eq!(name, Some("\"Alice\"".to_string()));

        let age = json_array.json_path(1, "$.user.age").unwrap();
        assert_eq!(age, None);
    }
}
