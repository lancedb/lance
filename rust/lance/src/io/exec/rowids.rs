// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::hash::Hash;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::UInt64Type;
use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Schema};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::ColumnarValue;
use datafusion_physical_expr::PhysicalExpr;
use lance_table::rowids::RowIdIndex;

/// A physical expression that returns the row address for a given row id.
#[derive(Debug, PartialEq, Hash)]
pub struct RowAddrExpr {
    row_id_index: Option<Arc<RowIdIndex>>,
    rowid_pos: usize,
}

impl RowAddrExpr {
    pub fn new(row_id_index: Option<Arc<RowIdIndex>>, rowid_pos: usize) -> Self {
        Self {
            row_id_index,
            rowid_pos,
        }
    }
}

impl PartialEq<dyn std::any::Any + 'static> for RowAddrExpr {
    fn eq(&self, other: &(dyn std::any::Any + 'static)) -> bool {
        other.downcast_ref::<Self>().map_or(false, |o| self == o)
    }
}

impl std::fmt::Display for RowAddrExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "RowAddrExpr")
    }
}

impl PhysicalExpr for RowAddrExpr {
    fn as_any(&self) -> &(dyn std::any::Any + 'static) {
        self
    }

    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::UInt64)
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(true)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let rowids = batch
            .column(self.rowid_pos)
            .as_primitive_opt::<UInt64Type>()
            .ok_or_else(|| {
                DataFusionError::Internal("RowAddrExpr: rowid column is not a UInt64Type".into())
            })?;

        if let Some(row_id_index) = &self.row_id_index {
            if rowids.null_count() > 0 {
                let mut builder = arrow::array::UInt64Builder::with_capacity(rowids.len());
                for rowid in rowids.iter() {
                    if let Some(rowid) = rowid {
                        if let Some(row_addr) = row_id_index.get(rowid) {
                            builder.append_value(row_addr.into());
                        } else {
                            return Err(DataFusionError::Internal(
                                "RowAddrExpr: rowid not found in index".into(),
                            ));
                        }
                    } else {
                        builder.append_null();
                    }
                }
                Ok(ColumnarValue::Array(Arc::new(builder.finish())))
            } else {
                // Fast path - no branching for null values
                let mut rowaddrs: Vec<u64> = Vec::with_capacity(rowids.len());
                for rowid in rowids.values() {
                    if let Some(row_addr) = row_id_index.get(*rowid) {
                        rowaddrs.push(row_addr.into());
                    } else {
                        return Err(DataFusionError::Internal(
                            "RowAddrExpr: rowid not found in index".into(),
                        ));
                    }
                }
                Ok(ColumnarValue::Array(Arc::new(UInt64Array::from(rowaddrs))))
            }
        } else {
            // No index, then we should just copy the rowids
            Ok(ColumnarValue::Array(batch.column(self.rowid_pos).clone()))
        }
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(self)
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        let mut s = state;
        self.hash(&mut s);
    }
}

#[cfg(test)]
mod test {
    use arrow_array::Int32Array;
    use arrow_schema::Field;
    use lance_table::rowids::RowIdSequence;

    use super::*;

    #[tokio::test]
    async fn test_address_style_ids() {
        // Passing no index means rowids are same as rowaddrs
        let row_id_index: Option<Arc<RowIdIndex>> = None;
        let rowid_pos: usize = 0;

        // Create a record batch with rowids
        let rowids = Arc::new(UInt64Array::from(vec![1, 2, 3]));
        let schema = Schema::new(vec![Field::new("rowid", DataType::UInt64, true)]);
        let record_batch = RecordBatch::try_new(Arc::new(schema), vec![rowids.clone()]).unwrap();

        // Create the RowAddrExpr instance
        let row_addr_expr = RowAddrExpr::new(row_id_index, rowid_pos);

        // Evaluate the RowAddrExpr
        let result = row_addr_expr.evaluate(&record_batch).unwrap();
        let result = result.into_array(rowids.len()).unwrap();

        // Verify the result
        assert_eq!(result.as_ref(), rowids.as_ref() as &dyn Array);
        assert_eq!(Arc::as_ptr(&result), Arc::as_ptr(&rowids));
    }

    fn sample_row_id_index() -> Arc<RowIdIndex> {
        // Create a row id index
        // 0 -> 10
        // 1 -> 20
        // 2 -> 30
        let mut row_id_seq = RowIdSequence::from(100..110); // 10 rows
        row_id_seq.extend((0..1).into()); // rowid=0
        row_id_seq.extend((111..120).into()); // 9 rows
        row_id_seq.extend((1..2).into()); // rowid=1
        row_id_seq.extend((131..140).into()); // 9 rows
        row_id_seq.extend((2..3).into()); // rowid=2
        Arc::new(RowIdIndex::new(&[(0, Arc::new(row_id_seq))]).unwrap())
    }

    #[tokio::test]
    async fn test_row_ids_no_nulls() {
        let row_id_index = sample_row_id_index();

        // Set the rowid_pos
        let rowid_pos: usize = 0;

        // Create a record batch with rowids
        let rowids = Arc::new(UInt64Array::from(vec![0, 1, 2]));
        let schema = Schema::new(vec![Field::new("rowid", DataType::UInt64, true)]);
        let record_batch = RecordBatch::try_new(Arc::new(schema), vec![rowids.clone()]).unwrap();

        // Create the RowAddrExpr instance
        let row_addr_expr = RowAddrExpr::new(Some(row_id_index.clone()), rowid_pos);

        // Evaluate the RowAddrExpr
        let result = row_addr_expr.evaluate(&record_batch).unwrap();
        let result = result.into_array(rowids.len()).unwrap();

        // Verify the result
        assert_eq!(
            result.as_ref(),
            Arc::new(UInt64Array::from(vec![10, 20, 30])).as_ref() as &dyn Array
        );
    }

    #[tokio::test]
    async fn test_row_ids_with_nulls() {
        let row_id_index = sample_row_id_index();

        // Set the rowid_pos
        let rowid_pos: usize = 0;

        // Create a record batch with rowids and null values
        let rowids = Arc::new(UInt64Array::from(vec![Some(0), None, Some(2)]));
        let schema = Schema::new(vec![Field::new("rowid", DataType::UInt64, true)]);
        let record_batch = RecordBatch::try_new(Arc::new(schema), vec![rowids.clone()]).unwrap();

        // Create the RowAddrExpr instance
        let row_addr_expr = RowAddrExpr::new(Some(row_id_index.clone()), rowid_pos);

        // Evaluate the RowAddrExpr
        let result = row_addr_expr.evaluate(&record_batch).unwrap();
        let result = result.into_array(rowids.len()).unwrap();

        // Verify the result
        assert_eq!(
            result.as_ref(),
            Arc::new(UInt64Array::from(vec![Some(10), None, Some(30)])).as_ref() as &dyn Array
        );
    }

    #[tokio::test]
    async fn test_invalid_schema() {
        let row_id_index = sample_row_id_index();

        // Set the rowid_pos
        let rowid_pos: usize = 0;

        // Create a record batch with rowids
        let rowids = Arc::new(Int32Array::from(vec![0, 1, 2]));
        let schema = Schema::new(vec![Field::new("invalid", DataType::Int32, true)]);
        let record_batch = RecordBatch::try_new(Arc::new(schema), vec![rowids.clone()]).unwrap();

        // Create the RowAddrExpr instance
        let row_addr_expr = RowAddrExpr::new(Some(row_id_index.clone()), rowid_pos);

        // Evaluate the RowAddrExpr
        let result = row_addr_expr.evaluate(&record_batch);

        // Verify the result
        assert!(result.is_err());
    }
}
