// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! In-memory BTree index for scalar fields.
//!
//! Provides O(log n) lookups and range queries using crossbeam-skiplist.
//! Used for primary key lookups and scalar column filtering.

use arrow_array::types::*;
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType;
use crossbeam_skiplist::SkipMap;
use datafusion::common::ScalarValue;
use lance_core::{Error, Result};
use lance_index::scalar::btree::OrderableScalarValue;
use snafu::location;

use super::RowPosition;

/// Composite key for BTree index.
///
/// By combining (scalar_value, row_position), each entry is unique.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexKey {
    /// The indexed scalar value.
    pub value: OrderableScalarValue,
    /// Row position (makes the key unique for non-unique indexes).
    pub row_position: RowPosition,
}

impl PartialOrd for IndexKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IndexKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // First compare by value, then by row_position
        match self.value.cmp(&other.value) {
            std::cmp::Ordering::Equal => self.row_position.cmp(&other.row_position),
            ord => ord,
        }
    }
}

/// In-memory BTree index for scalar fields.
///
/// Represents the in-memory portion of Lance's on-disk BTree index.
/// Implemented using crossbeam-skiplist for concurrent access with O(log n) operations.
#[derive(Debug)]
pub struct BTreeMemIndex {
    /// Ordered map: (scalar_value, row_position) -> ()
    lookup: SkipMap<IndexKey, ()>,
    /// Field ID this index is built on.
    field_id: i32,
    /// Column name (for Arrow batch lookups).
    column_name: String,
}

impl BTreeMemIndex {
    /// Create a new BTree index for the given field.
    pub fn new(field_id: i32, column_name: String) -> Self {
        Self {
            lookup: SkipMap::new(),
            field_id,
            column_name,
        }
    }

    /// Get the field ID this index is built on.
    pub fn field_id(&self) -> i32 {
        self.field_id
    }

    /// Insert rows from a batch into the index.
    pub fn insert(&self, batch: &RecordBatch, row_offset: u64) -> Result<()> {
        let col_idx = batch
            .schema()
            .column_with_name(&self.column_name)
            .map(|(idx, _)| idx)
            .ok_or_else(|| {
                Error::invalid_input(
                    format!("Column '{}' not found in batch", self.column_name),
                    location!(),
                )
            })?;

        let column = batch.column(col_idx);
        self.insert_array(column.as_ref(), row_offset)
    }

    /// Insert values from an Arrow array into the index.
    fn insert_array(&self, array: &dyn Array, row_offset: u64) -> Result<()> {
        macro_rules! insert_primitive {
            ($array_type:ty, $scalar_variant:ident) => {{
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<$array_type>>()
                    .unwrap();
                for (row_idx, value) in typed_array.iter().enumerate() {
                    let row_position = row_offset + row_idx as u64;
                    let key = IndexKey {
                        value: OrderableScalarValue(ScalarValue::$scalar_variant(value)),
                        row_position,
                    };
                    self.lookup.insert(key, ());
                }
            }};
        }

        match array.data_type() {
            DataType::Int8 => insert_primitive!(Int8Type, Int8),
            DataType::Int16 => insert_primitive!(Int16Type, Int16),
            DataType::Int32 => insert_primitive!(Int32Type, Int32),
            DataType::Int64 => insert_primitive!(Int64Type, Int64),
            DataType::UInt8 => insert_primitive!(UInt8Type, UInt8),
            DataType::UInt16 => insert_primitive!(UInt16Type, UInt16),
            DataType::UInt32 => insert_primitive!(UInt32Type, UInt32),
            DataType::UInt64 => insert_primitive!(UInt64Type, UInt64),
            DataType::Float32 => insert_primitive!(Float32Type, Float32),
            DataType::Float64 => insert_primitive!(Float64Type, Float64),
            DataType::Date32 => insert_primitive!(Date32Type, Date32),
            DataType::Date64 => insert_primitive!(Date64Type, Date64),
            DataType::Utf8 => {
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .unwrap();
                for (row_idx, value) in typed_array.iter().enumerate() {
                    let row_position = row_offset + row_idx as u64;
                    let key = IndexKey {
                        value: OrderableScalarValue(ScalarValue::Utf8(
                            value.map(|s| s.to_string()),
                        )),
                        row_position,
                    };
                    self.lookup.insert(key, ());
                }
            }
            DataType::LargeUtf8 => {
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow_array::LargeStringArray>()
                    .unwrap();
                for (row_idx, value) in typed_array.iter().enumerate() {
                    let row_position = row_offset + row_idx as u64;
                    let key = IndexKey {
                        value: OrderableScalarValue(ScalarValue::LargeUtf8(
                            value.map(|s| s.to_string()),
                        )),
                        row_position,
                    };
                    self.lookup.insert(key, ());
                }
            }
            DataType::Boolean => {
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow_array::BooleanArray>()
                    .unwrap();
                for (row_idx, value) in typed_array.iter().enumerate() {
                    let row_position = row_offset + row_idx as u64;
                    let key = IndexKey {
                        value: OrderableScalarValue(ScalarValue::Boolean(value)),
                        row_position,
                    };
                    self.lookup.insert(key, ());
                }
            }
            // Fallback for other types - use per-row extraction
            _ => {
                for row_idx in 0..array.len() {
                    let value = ScalarValue::try_from_array(array, row_idx)?;
                    let row_position = row_offset + row_idx as u64;
                    let key = IndexKey {
                        value: OrderableScalarValue(value),
                        row_position,
                    };
                    self.lookup.insert(key, ());
                }
            }
        }
        Ok(())
    }

    /// Look up row positions for an exact value.
    pub fn get(&self, value: &ScalarValue) -> Vec<RowPosition> {
        let orderable = OrderableScalarValue(value.clone());
        let start = IndexKey {
            value: orderable.clone(),
            row_position: 0,
        };
        let end = IndexKey {
            value: orderable,
            row_position: u64::MAX,
        };

        // Range scan: all entries with the same value
        self.lookup
            .range(start..=end)
            .map(|entry| entry.key().row_position)
            .collect()
    }

    /// Get the number of entries (not unique values).
    pub fn len(&self) -> usize {
        self.lookup.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.lookup.is_empty()
    }

    /// Get the column name.
    pub fn column_name(&self) -> &str {
        &self.column_name
    }

    /// Get a snapshot of all entries grouped by value in sorted order.
    pub fn snapshot(&self) -> Vec<(OrderableScalarValue, Vec<RowPosition>)> {
        let mut result: Vec<(OrderableScalarValue, Vec<RowPosition>)> = Vec::new();

        for entry in self.lookup.iter() {
            let key = entry.key();
            if let Some(last) = result.last_mut() {
                if last.0 == key.value {
                    last.1.push(key.row_position);
                    continue;
                }
            }
            result.push((key.value.clone(), vec![key.row_position]));
        }

        result
    }

    /// Get the data type of the indexed column.
    ///
    /// Returns None if the index is empty.
    pub fn data_type(&self) -> Option<arrow_schema::DataType> {
        self.lookup
            .front()
            .map(|entry| entry.key().value.0.data_type())
    }

    /// Export the index data as sorted RecordBatches for BTree index training.
    pub fn to_training_batches(&self, batch_size: usize) -> Result<Vec<RecordBatch>> {
        use arrow_schema::{DataType, Field, Schema};
        use lance_core::ROW_ID;
        use lance_index::scalar::registry::VALUE_COLUMN_NAME;
        use std::sync::Arc;

        if self.lookup.is_empty() {
            return Ok(vec![]);
        }

        // Get the data type from the first key
        let first_entry = self.lookup.front().unwrap();
        let data_type = first_entry.key().value.0.data_type();

        // Create schema for training data
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, data_type, true),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));

        let mut batches = Vec::new();
        let mut values: Vec<ScalarValue> = Vec::with_capacity(batch_size);
        let mut row_ids: Vec<u64> = Vec::with_capacity(batch_size);

        for entry in self.lookup.iter() {
            let key = entry.key();
            values.push(key.value.0.clone());
            row_ids.push(key.row_position);

            if values.len() >= batch_size {
                // Build and emit a batch
                let batch = self.build_training_batch(&schema, &values, &row_ids)?;
                batches.push(batch);
                values.clear();
                row_ids.clear();
            }
        }

        // Emit any remaining data
        if !values.is_empty() {
            let batch = self.build_training_batch(&schema, &values, &row_ids)?;
            batches.push(batch);
        }

        Ok(batches)
    }

    /// Build a single training batch from values and row IDs.
    fn build_training_batch(
        &self,
        schema: &std::sync::Arc<arrow_schema::Schema>,
        values: &[ScalarValue],
        row_ids: &[u64],
    ) -> Result<RecordBatch> {
        use arrow_array::UInt64Array;
        use std::sync::Arc;

        // Convert ScalarValues to Arrow array
        let value_array = ScalarValue::iter_to_array(values.iter().cloned())?;

        // Create row_id array
        let row_id_array = Arc::new(UInt64Array::from(row_ids.to_vec()));

        RecordBatch::try_new(schema.clone(), vec![value_array, row_id_array]).map_err(|e| {
            Error::io(
                format!("Failed to create training batch: {}", e),
                location!(),
            )
        })
    }
}

/// Configuration for a BTree scalar index.
#[derive(Debug, Clone)]
pub struct BTreeIndexConfig {
    /// Index name.
    pub name: String,
    /// Field ID the index is built on.
    pub field_id: i32,
    /// Column name (for Arrow batch lookups).
    pub column: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use std::sync::Arc;

    fn create_test_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn create_test_batch(schema: &ArrowSchema, start_id: i32) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(vec![start_id, start_id + 1, start_id + 2])),
                Arc::new(StringArray::from(vec!["alice", "bob", "charlie"])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_btree_index_insert_and_lookup() {
        let schema = create_test_schema();
        let index = BTreeMemIndex::new(0, "id".to_string());

        let batch = create_test_batch(&schema, 0);
        // row_offset = 0 for first batch
        index.insert(&batch, 0).unwrap();

        assert_eq!(index.len(), 3);

        // Row positions are 0, 1, 2 for the first batch
        let result = index.get(&ScalarValue::Int32(Some(0)));
        assert!(!result.is_empty());
        assert_eq!(result, vec![0]);

        let result = index.get(&ScalarValue::Int32(Some(1)));
        assert!(!result.is_empty());
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn test_btree_index_multiple_batches() {
        let schema = create_test_schema();
        let index = BTreeMemIndex::new(0, "id".to_string());

        let batch1 = create_test_batch(&schema, 0);
        let batch2 = create_test_batch(&schema, 10);

        // First batch: rows 0-2
        index.insert(&batch1, 0).unwrap();
        // Second batch: rows 3-5 (row_offset = 3 since batch1 had 3 rows)
        index.insert(&batch2, 3).unwrap();

        assert_eq!(index.len(), 6);

        // Value 10 is at row position 3 (first row of second batch)
        let result = index.get(&ScalarValue::Int32(Some(10)));
        assert!(!result.is_empty());
        assert_eq!(result, vec![3]);
    }

    #[test]
    fn test_btree_index_to_training_batches() {
        use lance_core::ROW_ID;
        use lance_index::scalar::registry::VALUE_COLUMN_NAME;

        let schema = create_test_schema();
        let index = BTreeMemIndex::new(0, "id".to_string());

        let batch1 = create_test_batch(&schema, 0); // ids: 0, 1, 2
        let batch2 = create_test_batch(&schema, 10); // ids: 10, 11, 12

        index.insert(&batch1, 0).unwrap(); // row positions 0, 1, 2
        index.insert(&batch2, 3).unwrap(); // row positions 3, 4, 5

        // Export as training batches (batch_size = 100 to get all in one batch)
        let batches = index.to_training_batches(100).unwrap();
        assert_eq!(batches.len(), 1);

        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 6);

        // Check schema
        assert_eq!(batch.schema().field(0).name(), VALUE_COLUMN_NAME);
        assert_eq!(batch.schema().field(1).name(), ROW_ID);

        // Data should be sorted by value (0, 1, 2, 10, 11, 12)
        let values = batch
            .column_by_name(VALUE_COLUMN_NAME)
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(values.value(0), 0);
        assert_eq!(values.value(1), 1);
        assert_eq!(values.value(2), 2);
        assert_eq!(values.value(3), 10);
        assert_eq!(values.value(4), 11);
        assert_eq!(values.value(5), 12);

        // Check row IDs match positions
        let row_ids = batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .unwrap();
        assert_eq!(row_ids.value(0), 0); // id=0 -> row 0
        assert_eq!(row_ids.value(1), 1); // id=1 -> row 1
        assert_eq!(row_ids.value(2), 2); // id=2 -> row 2
        assert_eq!(row_ids.value(3), 3); // id=10 -> row 3
        assert_eq!(row_ids.value(4), 4); // id=11 -> row 4
        assert_eq!(row_ids.value(5), 5); // id=12 -> row 5
    }

    #[test]
    fn test_btree_index_snapshot() {
        let schema = create_test_schema();
        let index = BTreeMemIndex::new(0, "id".to_string());

        let batch = create_test_batch(&schema, 0);
        index.insert(&batch, 0).unwrap();

        let snapshot = index.snapshot();
        assert_eq!(snapshot.len(), 3);

        // Snapshot should be in sorted order
        assert_eq!(snapshot[0].0 .0, ScalarValue::Int32(Some(0)));
        assert_eq!(snapshot[1].0 .0, ScalarValue::Int32(Some(1)));
        assert_eq!(snapshot[2].0 .0, ScalarValue::Int32(Some(2)));
    }
}
