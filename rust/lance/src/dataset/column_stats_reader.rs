// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! High-level reader for consolidated column statistics with automatic type dispatching.
//!
//! This module provides a convenient API for reading column statistics from consolidated
//! stats files (created by [`column_stats_consolidator`](crate::dataset::column_stats_consolidator)) with automatic
//! type conversion based on the dataset schema.
//!

use std::sync::Arc;

use arrow_array::{Array, ListArray, RecordBatch, StructArray, UInt32Array, UInt64Array};
use lance_arrow_stats::ArrowScalar;
use lance_core::Result;
use lance_core::datatypes::Schema;

use crate::Error;

/// High-level reader for column statistics with automatic type dispatching.
///
/// This reader provides convenient access to column statistics stored in
/// consolidated stats files. It automatically converts min/max values to
/// Arrow-native scalar values based on the dataset schema.
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
    pub min_values: Vec<ArrowScalar>,
    pub max_values: Vec<ArrowScalar>,
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
    ///
    /// In the new columnar format, column names are the schema field names
    /// (one column per dataset column in the stats batch).
    pub fn column_names(&self) -> Result<Vec<String>> {
        // In the new format, each column in the stats batch corresponds to a dataset column
        Ok(self
            .stats_batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect())
    }

    /// Read statistics for a specific column.
    ///
    /// Returns `None` if the column has no statistics available.
    ///
    /// In the new columnar format, the stats batch has one column per dataset column,
    /// each containing a `List<struct>` with zone statistics.
    pub fn read_column_stats(&self, column_name: &str) -> Result<Option<ColumnStats>> {
        // Check if column exists in stats batch (one column per dataset column)
        let column_array = self.stats_batch.column_by_name(column_name);

        if column_array.is_none() {
            // Column not in stats - return None (no stats available)
            return Ok(None);
        }

        let column_array = column_array.unwrap();

        // Get the field from the dataset schema
        let field = self.dataset_schema.field(column_name);

        if field.is_none() {
            // Column not in schema - return None (no stats available)
            return Ok(None);
        }
        let _ = field.unwrap();

        // Extract the ListArray for this column (one row total, so use row 0)
        let list_array = column_array
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| {
                Error::internal(format!("Expected ListArray for column '{}'", column_name))
            })?;

        // Check if batch is empty (0 rows)
        if list_array.len() == 0 {
            return Ok(None);
        }

        // Extract the StructArray from the list (row 0, since there's only one row)
        if list_array.is_null(0) || list_array.value_length(0) == 0 {
            return Ok(None);
        }

        let struct_array_ref = list_array.value(0);
        let struct_array = struct_array_ref
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                Error::internal(format!(
                    "Expected StructArray in list for column '{}'",
                    column_name
                ))
            })?;

        // Extract fields from the struct
        let fragment_id_array = struct_array
            .column_by_name("fragment_id")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Missing 'fragment_id' field in struct for column '{}'",
                    column_name
                ))
            })?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                Error::internal(format!(
                    "Expected UInt64Array for 'fragment_id' in column '{}'",
                    column_name
                ))
            })?;

        let zone_start_array = struct_array
            .column_by_name("zone_start")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Missing 'zone_start' field in struct for column '{}'",
                    column_name
                ))
            })?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                Error::internal(format!(
                    "Expected UInt64Array for 'zone_start' in column '{}'",
                    column_name
                ))
            })?;

        let zone_length_array = struct_array
            .column_by_name("zone_length")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Missing 'zone_length' field in struct for column '{}'",
                    column_name
                ))
            })?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                Error::internal(format!(
                    "Expected UInt64Array for 'zone_length' in column '{}'",
                    column_name
                ))
            })?;

        let null_count_array = struct_array
            .column_by_name("null_count")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Missing 'null_count' field in struct for column '{}'",
                    column_name
                ))
            })?
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| {
                Error::internal(format!(
                    "Expected UInt32Array for 'null_count' in column '{}'",
                    column_name
                ))
            })?;

        let nan_count_array = struct_array
            .column_by_name("nan_count")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Missing 'nan_count' field in struct for column '{}'",
                    column_name
                ))
            })?
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| {
                Error::internal(format!(
                    "Expected UInt32Array for 'nan_count' in column '{}'",
                    column_name
                ))
            })?;

        let min_value_array = struct_array.column_by_name("min_value").ok_or_else(|| {
            Error::internal(format!(
                "Missing 'min_value' field in struct for column '{}'",
                column_name
            ))
        })?;

        let max_value_array = struct_array.column_by_name("max_value").ok_or_else(|| {
            Error::internal(format!(
                "Missing 'max_value' field in struct for column '{}'",
                column_name
            ))
        })?;

        // Min/max are stored in the column's Arrow type; convert to ArrowScalar per zone
        let num_zones = fragment_id_array.len();
        let mut min_values = Vec::with_capacity(num_zones);
        let mut max_values = Vec::with_capacity(num_zones);

        for i in 0..num_zones {
            let min_val = ArrowScalar::try_new(min_value_array, i).map_err(|e| {
                Error::internal(format!(
                    "Failed to get min ArrowScalar for column '{}' zone {}: {}",
                    column_name, i, e
                ))
            })?;
            let max_val = ArrowScalar::try_new(max_value_array, i).map_err(|e| {
                Error::internal(format!(
                    "Failed to get max ArrowScalar for column '{}' zone {}: {}",
                    column_name, i, e
                ))
            })?;
            min_values.push(min_val);
            max_values.push(max_val);
        }

        Ok(Some(ColumnStats {
            fragment_ids: fragment_id_array.values().to_vec(),
            zone_starts: zone_start_array.values().to_vec(),
            zone_lengths: zone_length_array.values().to_vec(),
            null_counts: null_count_array.values().to_vec(),
            nan_counts: nan_count_array.values().to_vec(),
            min_values,
            max_values,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Re-import types that are used by the parent module but not re-exported
    use crate::dataset::column_stats_consolidator::create_consolidated_stats_schema;
    use arrow_array::{ArrayRef, ListArray, RecordBatch};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::datatypes::Schema;

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
        // New format: one row total, one column per dataset column, each containing List<struct>
        // min_value/max_value use the column's Arrow type (Int32 for id, Utf8 for name)
        use arrow_array::{Int32Array, StringArray as ArrowStringArray, StructArray};
        use arrow_buffer::OffsetBuffer;
        use lance_file::writer::create_consolidated_zone_struct_type;

        let _dataset_schema = create_test_schema();
        let id_zone_type = create_consolidated_zone_struct_type(&DataType::Int32);
        let name_zone_type = create_consolidated_zone_struct_type(&DataType::Utf8);

        // Build struct array for "id" column: 2 zones (min/max as Int32)
        let id_struct_array = StructArray::from(vec![
            (
                Arc::new(ArrowField::new("fragment_id", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![0, 1])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("zone_start", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![0, 100])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("zone_length", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![100, 100])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("null_count", DataType::UInt32, false)),
                Arc::new(UInt32Array::from(vec![0, 0])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("nan_count", DataType::UInt32, false)),
                Arc::new(UInt32Array::from(vec![0, 0])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("min_value", DataType::Int32, true)),
                Arc::new(Int32Array::from(vec![0, 100])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("max_value", DataType::Int32, true)),
                Arc::new(Int32Array::from(vec![99, 199])) as ArrayRef,
            ),
        ]);

        // Build struct array for "name" column: 2 zones (min/max as Utf8)
        let name_struct_array = StructArray::from(vec![
            (
                Arc::new(ArrowField::new("fragment_id", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![0, 1])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("zone_start", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![0, 100])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("zone_length", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![100, 100])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("null_count", DataType::UInt32, false)),
                Arc::new(UInt32Array::from(vec![0, 0])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("nan_count", DataType::UInt32, false)),
                Arc::new(UInt32Array::from(vec![0, 0])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("min_value", DataType::Utf8, true)),
                Arc::new(ArrowStringArray::from(vec!["alice", "mike"])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("max_value", DataType::Utf8, true)),
                Arc::new(ArrowStringArray::from(vec!["jenny", "zoe"])) as ArrayRef,
            ),
        ]);

        // Wrap each struct array in a ListArray (one list per column, one row total)
        let id_list_field = Arc::new(ArrowField::new("zone", id_zone_type, false));
        let name_list_field = Arc::new(ArrowField::new("zone", name_zone_type, false));
        let id_list = ListArray::try_new(
            id_list_field.clone(),
            OffsetBuffer::from_lengths([2]),
            Arc::new(id_struct_array) as ArrayRef,
            None,
        )
        .unwrap();

        let name_list = ListArray::try_new(
            name_list_field.clone(),
            OffsetBuffer::from_lengths([2]),
            Arc::new(name_struct_array) as ArrayRef,
            None,
        )
        .unwrap();

        // Schema has 3 fields (id, name, score), but we only create stats for id and name
        let stats_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::List(id_list_field), false),
            ArrowField::new("name", DataType::List(name_list_field), false),
        ]));

        RecordBatch::try_new(
            stats_schema,
            vec![
                Arc::new(id_list) as ArrayRef,
                Arc::new(name_list) as ArrayRef,
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
        assert_eq!(stats.min_values[0], ArrowScalar::from(0i32));
        assert_eq!(stats.min_values[1], ArrowScalar::from(100i32));

        // Verify max_values
        assert_eq!(stats.max_values.len(), 2);
        assert_eq!(stats.max_values[0], ArrowScalar::from(99i32));
        assert_eq!(stats.max_values[1], ArrowScalar::from(199i32));
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
        assert_eq!(stats.min_values[0], ArrowScalar::from("alice"));
        assert_eq!(stats.min_values[1], ArrowScalar::from("mike"));

        // Verify max_values (strings)
        assert_eq!(stats.max_values.len(), 2);
        assert_eq!(stats.max_values[0], ArrowScalar::from("jenny"));
        assert_eq!(stats.max_values[1], ArrowScalar::from("zoe"));
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
    fn test_read_column_stats_null_min_max() {
        use arrow_array::{Int32Array, StructArray};
        use arrow_buffer::OffsetBuffer;
        use lance_file::writer::create_consolidated_zone_struct_type;

        let schema = create_test_schema();
        let id_zone_type = create_consolidated_zone_struct_type(&DataType::Int32);

        let id_struct_array = StructArray::from(vec![
            (
                Arc::new(ArrowField::new("fragment_id", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![0])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("zone_start", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![0])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("zone_length", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![10])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("null_count", DataType::UInt32, false)),
                Arc::new(UInt32Array::from(vec![10])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("nan_count", DataType::UInt32, false)),
                Arc::new(UInt32Array::from(vec![0])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("min_value", DataType::Int32, true)),
                Arc::new(Int32Array::from(vec![None])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("max_value", DataType::Int32, true)),
                Arc::new(Int32Array::from(vec![None])) as ArrayRef,
            ),
        ]);

        let id_list_field = Arc::new(ArrowField::new("zone", id_zone_type, false));
        let id_list = ListArray::try_new(
            id_list_field.clone(),
            OffsetBuffer::from_lengths([1]),
            Arc::new(id_struct_array) as ArrayRef,
            None,
        )
        .unwrap();

        let stats_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::List(id_list_field),
            false,
        )]));
        let stats_batch =
            RecordBatch::try_new(stats_schema, vec![Arc::new(id_list) as ArrayRef]).unwrap();

        let reader = ColumnStatsReader::new(schema, stats_batch);
        let stats = reader.read_column_stats("id").unwrap().unwrap();

        assert_eq!(
            stats.min_values[0],
            ArrowScalar::new_null(&DataType::Int32).unwrap()
        );
        assert_eq!(
            stats.max_values[0],
            ArrowScalar::new_null(&DataType::Int32).unwrap()
        );
    }

    #[test]
    fn test_empty_stats_batch() {
        let schema = create_test_schema();

        // Create empty stats batch using the shared schema function
        let stats_schema = create_consolidated_stats_schema(&schema);

        let empty_batch = RecordBatch::new_empty(stats_schema);
        let reader = ColumnStatsReader::new(schema, empty_batch);

        // Reading from empty batch should return None (no stats available)
        let result = reader.read_column_stats("id").unwrap();
        assert!(result.is_none());
    }
}
