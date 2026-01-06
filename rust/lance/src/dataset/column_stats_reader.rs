// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! High-level reader for column statistics with automatic type dispatching.
//!
//! This module provides a convenient API for reading column statistics
//! from consolidated stats files with automatic type conversion based on
//! the dataset schema.

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
        let column_names = self
            .stats_batch
            .column(0)
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
        // Find the row index for this column
        let column_names = self
            .stats_batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Internal {
                message: "Expected StringArray for column_names".to_string(),
                location: location!(),
            })?;

        let row_idx = (0..column_names.len())
            .find(|&i| column_names.value(i) == column_name)
            .ok_or_else(|| Error::Internal {
                message: format!("Column '{}' not found in statistics", column_name),
                location: location!(),
            })?;

        // Get the field from the dataset schema
        let field = self
            .dataset_schema
            .field(column_name)
            .ok_or_else(|| Error::Internal {
                message: format!("Column '{}' not found in dataset schema", column_name),
                location: location!(),
            })?;

        // Extract arrays for this column
        let fragment_ids_ref = self
            .stats_batch
            .column(1)
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
            .column(2)
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
            .column(3)
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
            .column(4)
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
            .column(5)
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
            .column(6)
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
            .column(7)
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

    // The format is typically like: Int32(123), Float64(45.6), Utf8("hello")
    // We need to extract the value and parse it according to the expected type

    match data_type {
        DataType::Int8 => {
            let val = extract_numeric_value(s)?;
            Ok(ScalarValue::Int8(Some(val.parse().map_err(|e| {
                Error::Internal {
                    message: format!("Failed to parse Int8: {}", e),
                    location: location!(),
                }
            })?)))
        }
        DataType::Int16 => {
            let val = extract_numeric_value(s)?;
            Ok(ScalarValue::Int16(Some(val.parse().map_err(|e| {
                Error::Internal {
                    message: format!("Failed to parse Int16: {}", e),
                    location: location!(),
                }
            })?)))
        }
        DataType::Int32 => {
            let val = extract_numeric_value(s)?;
            Ok(ScalarValue::Int32(Some(val.parse().map_err(|e| {
                Error::Internal {
                    message: format!("Failed to parse Int32: {}", e),
                    location: location!(),
                }
            })?)))
        }
        DataType::Int64 => {
            let val = extract_numeric_value(s)?;
            Ok(ScalarValue::Int64(Some(val.parse().map_err(|e| {
                Error::Internal {
                    message: format!("Failed to parse Int64: {}", e),
                    location: location!(),
                }
            })?)))
        }
        DataType::UInt8 => {
            let val = extract_numeric_value(s)?;
            Ok(ScalarValue::UInt8(Some(val.parse().map_err(|e| {
                Error::Internal {
                    message: format!("Failed to parse UInt8: {}", e),
                    location: location!(),
                }
            })?)))
        }
        DataType::UInt16 => {
            let val = extract_numeric_value(s)?;
            Ok(ScalarValue::UInt16(Some(val.parse().map_err(|e| {
                Error::Internal {
                    message: format!("Failed to parse UInt16: {}", e),
                    location: location!(),
                }
            })?)))
        }
        DataType::UInt32 => {
            let val = extract_numeric_value(s)?;
            Ok(ScalarValue::UInt32(Some(val.parse().map_err(|e| {
                Error::Internal {
                    message: format!("Failed to parse UInt32: {}", e),
                    location: location!(),
                }
            })?)))
        }
        DataType::UInt64 => {
            let val = extract_numeric_value(s)?;
            Ok(ScalarValue::UInt64(Some(val.parse().map_err(|e| {
                Error::Internal {
                    message: format!("Failed to parse UInt64: {}", e),
                    location: location!(),
                }
            })?)))
        }
        DataType::Float32 => {
            let val = extract_numeric_value(s)?;
            Ok(ScalarValue::Float32(Some(val.parse().map_err(|e| {
                Error::Internal {
                    message: format!("Failed to parse Float32: {}", e),
                    location: location!(),
                }
            })?)))
        }
        DataType::Float64 => {
            let val = extract_numeric_value(s)?;
            Ok(ScalarValue::Float64(Some(val.parse().map_err(|e| {
                Error::Internal {
                    message: format!("Failed to parse Float64: {}", e),
                    location: location!(),
                }
            })?)))
        }
        DataType::Utf8 => {
            let val = extract_string_value(s)?;
            Ok(ScalarValue::Utf8(Some(val.to_string())))
        }
        DataType::LargeUtf8 => {
            let val = extract_string_value(s)?;
            Ok(ScalarValue::LargeUtf8(Some(val.to_string())))
        }
        _ => Err(Error::Internal {
            message: format!("Unsupported data type for stats parsing: {:?}", data_type),
            location: location!(),
        }),
    }
}

/// Extract numeric value from debug format like "Int32(123)" -> "123"
fn extract_numeric_value(s: &str) -> Result<&str> {
    if let Some(start) = s.find('(') {
        if let Some(end) = s.rfind(')') {
            return Ok(&s[start + 1..end]);
        }
    }
    Err(Error::Internal {
        message: format!("Invalid numeric value format: {}", s),
        location: location!(),
    })
}

/// Extract string value from debug format like 'Utf8("hello")' -> "hello"
fn extract_string_value(s: &str) -> Result<&str> {
    if let Some(start) = s.find('"') {
        if let Some(end) = s.rfind('"') {
            if end > start {
                return Ok(&s[start + 1..end]);
            }
        }
    }
    Err(Error::Internal {
        message: format!("Invalid string value format: {}", s),
        location: location!(),
    })
}
