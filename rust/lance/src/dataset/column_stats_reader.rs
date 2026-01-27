// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! High-level reader for consolidated column statistics with automatic type dispatching.
//!
//! This module provides a convenient API for reading column statistics from consolidated
//! stats files (created by [`column_stats_consolidator`](crate::dataset::column_stats_consolidator)) with automatic
//! type conversion based on the dataset schema.
//!
//! # Overview
//!
//! Consolidated stats files store min/max values as strings. This module:
//! 1. Reads the consolidated stats RecordBatch (list-based layout)
//! 2. Converts string-encoded min/max values to strongly-typed [`ScalarValue`] based on
//!    the dataset schema
//! 3. Provides a convenient query API via [`ColumnStatsReader`]
//!

use std::sync::Arc;

use arrow_array::{Array, ListArray, RecordBatch, StringArray, UInt32Array, UInt64Array};
use datafusion::scalar::ScalarValue;
use lance_core::Result;
use lance_core::datatypes::Schema;
use snafu::location;

use crate::Error;

/// High-level reader for column statistics with automatic type dispatching.
///
/// This reader provides convenient access to column statistics stored in
/// consolidated stats files. It automatically converts min/max values to
/// strongly-typed ScalarValue based on the dataset schema.
pub struct ColumnStatsReader {
    dataset_schema: Arc<Schema>,
    stats_batch: RecordBatch,
}

/// Statistics for a single column, with strongly-typed min/max values.
#[derive(Debug, Clone)]
pub struct ColumnStats {
    pub fragment_ids: Vec<u64>,
    pub zone_starts: Vec<u64>,
    pub zone_lengths: Vec<u64>,
    pub null_counts: Vec<u32>,
    pub nan_counts: Vec<u32>,
    pub min_values: Vec<ScalarValue>,
    pub max_values: Vec<ScalarValue>,
}

impl ColumnStatsReader {
    /// Create a new reader from a consolidated stats RecordBatch.
    ///
    /// # Arguments
    ///
    /// * `dataset_schema` - The schema of the dataset (for type information)
    /// * `stats_batch` - The consolidated stats RecordBatch
    pub fn new(dataset_schema: Arc<Schema>, stats_batch: RecordBatch) -> Self {
        Self {
            dataset_schema,
            stats_batch,
        }
    }

    /// Get the list of column names that have statistics available.
    pub fn column_names(&self) -> Result<Vec<String>> {
        use lance_file::writer::COLUMN_STATS_COLUMN_NAME_FIELD;
        let column_names = self
            .stats_batch
            .column_by_name(COLUMN_STATS_COLUMN_NAME_FIELD)
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "Expected column '{}' in stats batch",
                    COLUMN_STATS_COLUMN_NAME_FIELD
                ),
                location: location!(),
            })?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Internal {
                message: "Expected StringArray for column_names".to_string(),
                location: location!(),
            })?;

        Ok((0..column_names.len())
            .map(|i| column_names.value(i).to_string())
            .collect())
    }

    /// Read statistics for a specific column.
    ///
    /// Returns `None` if the column has no statistics available.
    pub fn read_column_stats(&self, column_name: &str) -> Result<Option<ColumnStats>> {
        use lance_file::writer::COLUMN_STATS_COLUMN_NAME_FIELD;
        // Find the row index for this column
        let column_names = self
            .stats_batch
            .column_by_name(COLUMN_STATS_COLUMN_NAME_FIELD)
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "Expected column '{}' in stats batch",
                    COLUMN_STATS_COLUMN_NAME_FIELD
                ),
                location: location!(),
            })?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Internal {
                message: "Expected StringArray for column_names".to_string(),
                location: location!(),
            })?;

        // Check if column exists in stats batch
        let row_idx = (0..column_names.len()).find(|&i| column_names.value(i) == column_name);

        if row_idx.is_none() {
            // Column not in stats - return None (no stats available)
            return Ok(None);
        }
        let row_idx = row_idx.unwrap();

        // Get the field from the dataset schema
        let field = self.dataset_schema.field(column_name);

        if field.is_none() {
            // Column not in schema - return None (no stats available)
            return Ok(None);
        }
        let field = field.unwrap();

        // Extract arrays for this column using column names for better readability
        use lance_file::writer::{
            COLUMN_STATS_MAX_VALUE_FIELD, COLUMN_STATS_MIN_VALUE_FIELD,
            COLUMN_STATS_NAN_COUNT_FIELD, COLUMN_STATS_NULL_COUNT_FIELD,
            COLUMN_STATS_ZONE_LENGTH_FIELD, COLUMN_STATS_ZONE_START_FIELD,
        };

        let fragment_ids_ref = self
            .stats_batch
            .column_by_name("fragment_ids")
            .ok_or_else(|| Error::Internal {
                message: "Expected 'fragment_ids' column in stats batch".to_string(),
                location: location!(),
            })?
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| Error::Internal {
                message: "Expected ListArray for fragment_ids".to_string(),
                location: location!(),
            })?
            .value(row_idx);
        let fragment_ids = fragment_ids_ref
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal {
                message: "Expected UInt64Array in fragment_ids list".to_string(),
                location: location!(),
            })?;

        let zone_starts_ref = self
            .stats_batch
            .column_by_name("zone_starts")
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "Expected 'zone_starts' column ({}) in stats batch",
                    COLUMN_STATS_ZONE_START_FIELD
                ),
                location: location!(),
            })?
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| Error::Internal {
                message: "Expected ListArray for zone_starts".to_string(),
                location: location!(),
            })?
            .value(row_idx);
        let zone_starts = zone_starts_ref
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal {
                message: "Expected UInt64Array in zone_starts list".to_string(),
                location: location!(),
            })?;

        let zone_lengths_ref = self
            .stats_batch
            .column_by_name("zone_lengths")
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "Expected 'zone_lengths' column ({}) in stats batch",
                    COLUMN_STATS_ZONE_LENGTH_FIELD
                ),
                location: location!(),
            })?
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| Error::Internal {
                message: "Expected ListArray for zone_lengths".to_string(),
                location: location!(),
            })?
            .value(row_idx);
        let zone_lengths = zone_lengths_ref
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal {
                message: "Expected UInt64Array in zone_lengths list".to_string(),
                location: location!(),
            })?;

        let null_counts_ref = self
            .stats_batch
            .column_by_name("null_counts")
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "Expected 'null_counts' column ({}) in stats batch",
                    COLUMN_STATS_NULL_COUNT_FIELD
                ),
                location: location!(),
            })?
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| Error::Internal {
                message: "Expected ListArray for null_counts".to_string(),
                location: location!(),
            })?
            .value(row_idx);
        let null_counts = null_counts_ref
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| Error::Internal {
                message: "Expected UInt32Array in null_counts list".to_string(),
                location: location!(),
            })?;

        let nan_counts_ref = self
            .stats_batch
            .column_by_name("nan_counts")
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "Expected 'nan_counts' column ({}) in stats batch",
                    COLUMN_STATS_NAN_COUNT_FIELD
                ),
                location: location!(),
            })?
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| Error::Internal {
                message: "Expected ListArray for nan_counts".to_string(),
                location: location!(),
            })?
            .value(row_idx);
        let nan_counts = nan_counts_ref
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| Error::Internal {
                message: "Expected UInt32Array in nan_counts list".to_string(),
                location: location!(),
            })?;

        let min_values_ref = self
            .stats_batch
            .column_by_name("min_values")
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "Expected 'min_values' column ({}) in stats batch",
                    COLUMN_STATS_MIN_VALUE_FIELD
                ),
                location: location!(),
            })?
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| Error::Internal {
                message: "Expected ListArray for min_values".to_string(),
                location: location!(),
            })?
            .value(row_idx);
        let min_values_str = min_values_ref
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Internal {
                message: "Expected StringArray in min_values list".to_string(),
                location: location!(),
            })?;

        let max_values_ref = self
            .stats_batch
            .column_by_name("max_values")
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "Expected 'max_values' column ({}) in stats batch",
                    COLUMN_STATS_MAX_VALUE_FIELD
                ),
                location: location!(),
            })?
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| Error::Internal {
                message: "Expected ListArray for max_values".to_string(),
                location: location!(),
            })?
            .value(row_idx);
        let max_values_str = max_values_ref
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Internal {
                message: "Expected StringArray in max_values list".to_string(),
                location: location!(),
            })?;

        // Parse min/max values with automatic type dispatching
        let mut min_values = Vec::with_capacity(min_values_str.len());
        let mut max_values = Vec::with_capacity(max_values_str.len());

        for i in 0..min_values_str.len() {
            let min_str = min_values_str.value(i);
            let max_str = max_values_str.value(i);

            let min_val = parse_scalar_value(min_str, &field.data_type())?;
            let max_val = parse_scalar_value(max_str, &field.data_type())?;

            min_values.push(min_val);
            max_values.push(max_val);
        }

        Ok(Some(ColumnStats {
            fragment_ids: fragment_ids.values().to_vec(),
            zone_starts: zone_starts.values().to_vec(),
            zone_lengths: zone_lengths.values().to_vec(),
            null_counts: null_counts.values().to_vec(),
            nan_counts: nan_counts.values().to_vec(),
            min_values,
            max_values,
        }))
    }
}

/// Parse a ScalarValue from a debug-format string based on the expected type.
fn parse_scalar_value(s: &str, data_type: &arrow_schema::DataType) -> Result<ScalarValue> {
    use arrow_schema::DataType;

    // The string now contains just the value without type prefix
    // E.g., "42", "3.14", "hello" (no "Int32(...)" wrapper)

    match data_type {
        DataType::Int8 => Ok(ScalarValue::Int8(Some(s.parse().map_err(|e| {
            Error::Internal {
                message: format!("Failed to parse Int8 from '{}': {}", s, e),
                location: location!(),
            }
        })?))),
        DataType::Int16 => Ok(ScalarValue::Int16(Some(s.parse().map_err(|e| {
            Error::Internal {
                message: format!("Failed to parse Int16 from '{}': {}", s, e),
                location: location!(),
            }
        })?))),
        DataType::Int32 => Ok(ScalarValue::Int32(Some(s.parse().map_err(|e| {
            Error::Internal {
                message: format!("Failed to parse Int32 from '{}': {}", s, e),
                location: location!(),
            }
        })?))),
        DataType::Int64 => Ok(ScalarValue::Int64(Some(s.parse().map_err(|e| {
            Error::Internal {
                message: format!("Failed to parse Int64 from '{}': {}", s, e),
                location: location!(),
            }
        })?))),
        DataType::UInt8 => Ok(ScalarValue::UInt8(Some(s.parse().map_err(|e| {
            Error::Internal {
                message: format!("Failed to parse UInt8 from '{}': {}", s, e),
                location: location!(),
            }
        })?))),
        DataType::UInt16 => Ok(ScalarValue::UInt16(Some(s.parse().map_err(|e| {
            Error::Internal {
                message: format!("Failed to parse UInt16 from '{}': {}", s, e),
                location: location!(),
            }
        })?))),
        DataType::UInt32 => Ok(ScalarValue::UInt32(Some(s.parse().map_err(|e| {
            Error::Internal {
                message: format!("Failed to parse UInt32 from '{}': {}", s, e),
                location: location!(),
            }
        })?))),
        DataType::UInt64 => Ok(ScalarValue::UInt64(Some(s.parse().map_err(|e| {
            Error::Internal {
                message: format!("Failed to parse UInt64 from '{}': {}", s, e),
                location: location!(),
            }
        })?))),
        DataType::Float32 => Ok(ScalarValue::Float32(Some(s.parse().map_err(|e| {
            Error::Internal {
                message: format!("Failed to parse Float32 from '{}': {}", s, e),
                location: location!(),
            }
        })?))),
        DataType::Float64 => Ok(ScalarValue::Float64(Some(s.parse().map_err(|e| {
            Error::Internal {
                message: format!("Failed to parse Float64 from '{}': {}", s, e),
                location: location!(),
            }
        })?))),
        DataType::Utf8 => Ok(ScalarValue::Utf8(Some(s.to_string()))),
        DataType::LargeUtf8 => Ok(ScalarValue::LargeUtf8(Some(s.to_string()))),
        _ => Err(Error::Internal {
            message: format!("Unsupported data type for stats parsing: {:?}", data_type),
            location: location!(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Re-import types that are used by the parent module but not re-exported
    use crate::dataset::column_stats_consolidator::create_consolidated_stats_schema;
    use arrow_array::builder::{ListBuilder, StringBuilder, UInt32Builder, UInt64Builder};
    use arrow_array::{RecordBatch, StringArray as ArrowStringArray};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::datatypes::Schema;
    use lance_file::writer::{
        COLUMN_STATS_MAX_VALUE_FIELD, COLUMN_STATS_MIN_VALUE_FIELD, COLUMN_STATS_NAN_COUNT_FIELD,
        COLUMN_STATS_NULL_COUNT_FIELD, COLUMN_STATS_ZONE_LENGTH_FIELD,
        COLUMN_STATS_ZONE_START_FIELD,
    };

    fn create_test_schema() -> Arc<Schema> {
        Arc::new(
            Schema::try_from(&ArrowSchema::new(vec![
                ArrowField::new("id", DataType::Int32, false),
                ArrowField::new("name", DataType::Utf8, false),
                ArrowField::new("score", DataType::Float64, false),
            ]))
            .unwrap(),
        )
    }

    fn create_test_stats_batch() -> RecordBatch {
        // Create a consolidated stats batch with 2 columns: "id" and "name"
        // Use the shared schema creation function from column_stats_consolidator.rs
        let schema = create_consolidated_stats_schema();

        // Build lists for "id" column (Int32) - use constants to match the schema
        // Note: "fragment_id" is used in consolidated layout (not in flat layout constants)
        let mut fragment_ids_builder = ListBuilder::new(UInt64Builder::new())
            .with_field(ArrowField::new("fragment_id", DataType::UInt64, false));
        fragment_ids_builder.values().append_value(0);
        fragment_ids_builder.values().append_value(1);
        fragment_ids_builder.append(true);

        let mut zone_starts_builder = ListBuilder::new(UInt64Builder::new()).with_field(
            ArrowField::new(COLUMN_STATS_ZONE_START_FIELD, DataType::UInt64, false),
        );
        zone_starts_builder.values().append_value(0);
        zone_starts_builder.values().append_value(100);
        zone_starts_builder.append(true);

        let mut zone_lengths_builder = ListBuilder::new(UInt64Builder::new()).with_field(
            ArrowField::new(COLUMN_STATS_ZONE_LENGTH_FIELD, DataType::UInt64, false),
        );
        zone_lengths_builder.values().append_value(100);
        zone_lengths_builder.values().append_value(100);
        zone_lengths_builder.append(true);

        let mut null_counts_builder = ListBuilder::new(UInt32Builder::new()).with_field(
            ArrowField::new(COLUMN_STATS_NULL_COUNT_FIELD, DataType::UInt32, false),
        );
        null_counts_builder.values().append_value(0);
        null_counts_builder.values().append_value(0);
        null_counts_builder.append(true);

        let mut nan_counts_builder = ListBuilder::new(UInt32Builder::new()).with_field(
            ArrowField::new(COLUMN_STATS_NAN_COUNT_FIELD, DataType::UInt32, false),
        );
        nan_counts_builder.values().append_value(0);
        nan_counts_builder.values().append_value(0);
        nan_counts_builder.append(true);

        let mut mins_builder = ListBuilder::new(StringBuilder::new()).with_field(ArrowField::new(
            COLUMN_STATS_MIN_VALUE_FIELD,
            DataType::Utf8,
            false,
        ));
        mins_builder.values().append_value("0");
        mins_builder.values().append_value("100");
        mins_builder.append(true);

        let mut maxs_builder = ListBuilder::new(StringBuilder::new()).with_field(ArrowField::new(
            COLUMN_STATS_MAX_VALUE_FIELD,
            DataType::Utf8,
            false,
        ));
        maxs_builder.values().append_value("99");
        maxs_builder.values().append_value("199");
        maxs_builder.append(true);

        // Build lists for "name" column (Utf8)
        fragment_ids_builder.values().append_value(0);
        fragment_ids_builder.values().append_value(1);
        fragment_ids_builder.append(true);

        zone_starts_builder.values().append_value(0);
        zone_starts_builder.values().append_value(100);
        zone_starts_builder.append(true);

        zone_lengths_builder.values().append_value(100);
        zone_lengths_builder.values().append_value(100);
        zone_lengths_builder.append(true);

        null_counts_builder.values().append_value(0);
        null_counts_builder.values().append_value(0);
        null_counts_builder.append(true);

        nan_counts_builder.values().append_value(0);
        nan_counts_builder.values().append_value(0);
        nan_counts_builder.append(true);

        mins_builder.values().append_value("alice");
        mins_builder.values().append_value("mike");
        mins_builder.append(true);

        maxs_builder.values().append_value("jenny");
        maxs_builder.values().append_value("zoe");
        maxs_builder.append(true);

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(ArrowStringArray::from(vec!["id", "name"])),
                Arc::new(fragment_ids_builder.finish()),
                Arc::new(zone_starts_builder.finish()),
                Arc::new(zone_lengths_builder.finish()),
                Arc::new(null_counts_builder.finish()),
                Arc::new(nan_counts_builder.finish()),
                Arc::new(mins_builder.finish()),
                Arc::new(maxs_builder.finish()),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_read_column_stats_int32() {
        let schema = create_test_schema();
        let stats_batch = create_test_stats_batch();
        let reader = ColumnStatsReader::new(schema, stats_batch);

        let stats = reader.read_column_stats("id").unwrap().unwrap();

        // Verify fragment_ids
        assert_eq!(stats.fragment_ids, vec![0, 1]);

        // Verify zone_starts
        assert_eq!(stats.zone_starts, vec![0, 100]);

        // Verify zone_lengths
        assert_eq!(stats.zone_lengths, vec![100, 100]);

        // Verify null_counts
        assert_eq!(stats.null_counts, vec![0, 0]);

        // Verify nan_counts
        assert_eq!(stats.nan_counts, vec![0, 0]);

        // Verify min_values
        assert_eq!(stats.min_values.len(), 2);
        assert_eq!(stats.min_values[0], ScalarValue::Int32(Some(0)));
        assert_eq!(stats.min_values[1], ScalarValue::Int32(Some(100)));

        // Verify max_values
        assert_eq!(stats.max_values.len(), 2);
        assert_eq!(stats.max_values[0], ScalarValue::Int32(Some(99)));
        assert_eq!(stats.max_values[1], ScalarValue::Int32(Some(199)));
    }

    #[test]
    fn test_read_column_stats_utf8() {
        let schema = create_test_schema();
        let stats_batch = create_test_stats_batch();
        let reader = ColumnStatsReader::new(schema, stats_batch);

        let stats = reader.read_column_stats("name").unwrap().unwrap();

        // Verify fragment_ids
        assert_eq!(stats.fragment_ids, vec![0, 1]);

        // Verify min_values (strings)
        assert_eq!(stats.min_values.len(), 2);
        assert_eq!(
            stats.min_values[0],
            ScalarValue::Utf8(Some("alice".to_string()))
        );
        assert_eq!(
            stats.min_values[1],
            ScalarValue::Utf8(Some("mike".to_string()))
        );

        // Verify max_values (strings)
        assert_eq!(stats.max_values.len(), 2);
        assert_eq!(
            stats.max_values[0],
            ScalarValue::Utf8(Some("jenny".to_string()))
        );
        assert_eq!(
            stats.max_values[1],
            ScalarValue::Utf8(Some("zoe".to_string()))
        );
    }

    #[test]
    fn test_read_column_stats_nonexistent_column() {
        let schema = create_test_schema();
        let stats_batch = create_test_stats_batch();
        let reader = ColumnStatsReader::new(schema, stats_batch);

        let result = reader.read_column_stats("nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_read_column_stats_column_not_in_schema() {
        let schema = create_test_schema();
        let stats_batch = create_test_stats_batch();
        let reader = ColumnStatsReader::new(schema, stats_batch);

        // "score" is in schema but not in stats_batch
        let result = reader.read_column_stats("score").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_scalar_value_int_types() {
        let cases = vec![
            (DataType::Int8, "42", ScalarValue::Int8(Some(42))),
            (DataType::Int16, "1000", ScalarValue::Int16(Some(1000))),
            (DataType::Int32, "100000", ScalarValue::Int32(Some(100000))),
            (
                DataType::Int64,
                "9999999999",
                ScalarValue::Int64(Some(9999999999)),
            ),
            (DataType::UInt8, "255", ScalarValue::UInt8(Some(255))),
            (DataType::UInt16, "65535", ScalarValue::UInt16(Some(65535))),
            (
                DataType::UInt32,
                "4294967295",
                ScalarValue::UInt32(Some(4294967295)),
            ),
            (
                DataType::UInt64,
                "18446744073709551615",
                ScalarValue::UInt64(Some(18446744073709551615)),
            ),
        ];

        for (data_type, input, expected) in cases {
            let result = parse_scalar_value(input, &data_type).unwrap();
            assert_eq!(result, expected, "Failed for type {:?}", data_type);
        }
    }

    #[test]
    fn test_parse_scalar_value_float_types() {
        let result = parse_scalar_value("2.5", &DataType::Float32).unwrap();
        assert_eq!(result, ScalarValue::Float32(Some(2.5)));

        let result = parse_scalar_value("1.234567890123456", &DataType::Float64).unwrap();
        assert_eq!(result, ScalarValue::Float64(Some(1.234567890123456)));
    }

    #[test]
    fn test_parse_scalar_value_string_types() {
        let result = parse_scalar_value("hello", &DataType::Utf8).unwrap();
        assert_eq!(result, ScalarValue::Utf8(Some("hello".to_string())));

        let result = parse_scalar_value("world", &DataType::LargeUtf8).unwrap();
        assert_eq!(result, ScalarValue::LargeUtf8(Some("world".to_string())));
    }

    #[test]
    fn test_parse_scalar_value_invalid_format() {
        let result = parse_scalar_value("not_a_number", &DataType::Int32);
        assert!(result.is_err());

        let result = parse_scalar_value("not_a_float", &DataType::Float64);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_scalar_value_unsupported_type() {
        let result = parse_scalar_value("true", &DataType::Boolean);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unsupported data type")
        );
    }

    #[test]
    fn test_empty_stats_batch() {
        let schema = create_test_schema();

        // Create empty stats batch using the shared schema function
        let stats_schema = create_consolidated_stats_schema();

        let empty_batch = RecordBatch::new_empty(stats_schema);
        let reader = ColumnStatsReader::new(schema, empty_batch);

        // Reading from empty batch should return None (no stats available)
        let result = reader.read_column_stats("id").unwrap();
        assert!(result.is_none());
    }
}
