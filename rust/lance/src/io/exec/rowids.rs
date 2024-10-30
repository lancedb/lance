// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{Arc, OnceLock};

use arrow_array::{cast::AsArray, types::UInt64Type, Array, ArrayRef, RecordBatch, UInt64Array};
use arrow_schema::{Schema, SchemaRef};
use datafusion::common::stats::Precision;
use datafusion::common::ColumnStatistics;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::EquivalenceProperties;
use futures::StreamExt;
use lance_core::{ROW_ADDR_FIELD, ROW_ID};
use lance_table::rowids::RowIdIndex;

use crate::dataset::rowids::get_row_id_index;
use crate::utils::future::SharedPrerequisite;
use crate::Dataset;

/// Add a `_rowaddr` column to a stream of record batches that have a `_rowid`.
///
/// It's generally more efficient to scan the `_rowaddr` column, but this can be
/// useful when reading secondary indices, which only have the `_rowid` column.
pub struct AddRowAddrExec {
    input: Arc<dyn ExecutionPlan>,
    dataset: Arc<Dataset>,
    /// Task to get the rowid index. Is not initialized until the first call to
    /// `execute`.
    row_id_index: OnceLock<Arc<SharedPrerequisite<Option<Arc<RowIdIndex>>>>>,
    /// Position in the input schema where the rowids are located
    rowid_pos: usize,
    /// Position in the output schema where to insert the row address
    rowaddr_pos: usize,
    output_schema: SchemaRef,
    properties: PlanProperties,
}

impl std::fmt::Debug for AddRowAddrExec {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("AddRowAddrExec")
            .field("input", &self.input)
            .field("dataset", &self.dataset)
            .field("rowid_pos", &self.rowid_pos)
            .field("rowaddr_pos", &self.rowaddr_pos)
            .field("output_schema", &self.output_schema)
            .field("properties", &self.properties)
            .finish()
    }
}

impl AddRowAddrExec {
    /// Create a new AddRowAddrExec node.
    ///
    /// This adds a `_rowaddr` column to streams where there is a `_rowid`
    /// column.
    ///
    /// # Errors
    ///
    /// If the `_rowid` field is not found in the input schema.
    ///
    /// # Arguments
    /// * `input` - The input plan to add row addresses to.
    /// * `dataset` - The dataset to get the row id index from.
    /// * `rowaddr_pos` - The position in the output schema where to insert the row address.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        rowaddr_pos: usize,
    ) -> Result<Self> {
        // Need to know the physical position of the row id field, so we don't
        // have to do a schema lookup for every batch.
        let input_schema = input.schema();
        let rowid_pos = input_schema
            .fields()
            .iter()
            .position(|f| f.name() == ROW_ID)
            .ok_or_else(|| {
                DataFusionError::Internal("rowid field not found in input schema".into())
            })?;

        let mut fields = input_schema.fields().iter().cloned().collect::<Vec<_>>();
        fields.insert(rowaddr_pos, Arc::new(ROW_ADDR_FIELD.clone()));
        let output_schema = Arc::new(Schema::new_with_metadata(
            fields,
            input_schema.metadata().clone(),
        ));

        let row_id_index = OnceLock::new();

        // Is just a simple projections, so it inherits the partitioning and
        // execution mode from parent.
        let properties = input
            .properties()
            .clone()
            .with_eq_properties(EquivalenceProperties::new(output_schema.clone()));

        Ok(Self {
            input,
            dataset,
            row_id_index,
            rowid_pos,
            rowaddr_pos,
            output_schema,
            properties,
        })
    }

    fn compute_row_addrs(
        row_ids: &ArrayRef,
        row_id_index: Option<&RowIdIndex>,
    ) -> Result<ArrayRef> {
        let row_id_values = row_ids.as_primitive_opt::<UInt64Type>().ok_or_else(|| {
            DataFusionError::Internal("AddRowAddrExec: rowid column is not a UInt64Array".into())
        })?;
        if let Some(row_id_index) = row_id_index {
            if row_id_values.null_count() > 0 {
                let mut builder = arrow::array::UInt64Builder::with_capacity(row_id_values.len());
                for rowid in row_id_values.iter() {
                    if let Some(rowid) = rowid {
                        if let Some(row_addr) = row_id_index.get(rowid) {
                            builder.append_value(row_addr.into());
                        } else {
                            return Err(DataFusionError::Internal(format!(
                                "AddRowAddrExec: rowid not found in index: {}",
                                rowid
                            )));
                        }
                    } else {
                        builder.append_null();
                    }
                }
                Ok(Arc::new(builder.finish()))
            } else {
                // Fast path - no branching for null values
                let mut rowaddrs: Vec<u64> = Vec::with_capacity(row_id_values.len());
                for rowid in row_id_values.values() {
                    if let Some(row_addr) = row_id_index.get(*rowid) {
                        rowaddrs.push(row_addr.into());
                    } else {
                        return Err(DataFusionError::Internal(format!(
                            "AddRowAddrExec: rowid not found in index: {}",
                            rowid
                        )));
                    }
                }
                Ok(Arc::new(UInt64Array::from(rowaddrs)))
            }
        } else {
            // No index, then we should just copy the rowids
            Ok(row_ids.clone())
        }
    }
}

impl DisplayAs for AddRowAddrExec {
    fn fmt_as(
        &self,
        _format_type: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "AddRowAddrExec")
    }
}

impl ExecutionPlan for AddRowAddrExec {
    fn name(&self) -> &str {
        "AddRowAddrExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> Arc<Schema> {
        self.output_schema.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        todo!()
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let index_prereq = self
            .row_id_index
            .get_or_init(|| {
                let dataset = self.dataset.clone();
                let fut = async move { get_row_id_index(dataset.as_ref()).await };
                SharedPrerequisite::spawn(fut)
            })
            .clone();

        let input_stream = self.input.execute(partition, context)?;

        let rowid_pos = self.rowid_pos;
        let rowaddr_pos = self.rowaddr_pos;
        let output_schema = self.output_schema.clone();
        let stream = input_stream.then(move |batch| {
            let output_schema = output_schema.clone();
            let index_prereq = index_prereq.clone();
            async move {
                let batch = batch?;
                index_prereq.wait_ready().await?;
                let row_id_index = index_prereq.get_ready();
                let index_ref = row_id_index.as_deref();

                let row_addr = Self::compute_row_addrs(batch.column(rowid_pos), index_ref)?;

                let mut columns = Vec::with_capacity(batch.num_columns() + 1);
                let existing_columns = batch.columns();
                columns.extend_from_slice(&existing_columns[..rowaddr_pos]);
                columns.push(row_addr);
                columns.extend_from_slice(&existing_columns[rowaddr_pos..]);

                Ok(RecordBatch::try_new(output_schema.clone(), columns)?)
            }
        });

        let stream = RecordBatchStreamAdapter::new(self.output_schema.clone(), stream.boxed());
        Ok(Box::pin(stream))
    }

    fn statistics(&self) -> Result<datafusion::physical_plan::Statistics> {
        let mut stats = self.input.statistics()?;

        let row_id_col_stats = stats.column_statistics.get(self.rowid_pos).ok_or_else(|| {
            DataFusionError::Internal("RowAddrExec: rowid column stats not found".into())
        })?;
        let row_addr_col_stats = ColumnStatistics {
            null_count: row_id_col_stats.null_count.clone(),
            distinct_count: row_id_col_stats.distinct_count.clone(),
            max_value: Precision::Absent,
            min_value: Precision::Absent,
        };

        let base_size = std::mem::size_of::<UInt64Array>();
        // Buffer size is the number of rows times 8 bytes per row, but there
        // is a minimum size of 64 bytes.
        let mut added_byte_size = stats
            .num_rows
            .clone()
            .map(|n| (n * 8).max(64))
            .add(&Precision::Exact(base_size));
        if row_id_col_stats
            .null_count
            .get_value()
            .map(|v| *v > 0)
            .unwrap_or_default()
        {
            // Account for null buffer.
            added_byte_size =
                added_byte_size.add(&stats.num_rows.clone().map(|n| n.div_ceil(8).max(64)));
        }
        stats.total_byte_size = stats.total_byte_size.add(&added_byte_size);
        stats
            .column_statistics
            .insert(self.rowaddr_pos, row_addr_col_stats);

        Ok(stats)
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }
}

#[cfg(test)]
mod test {
    use arrow_array::{Int32Array, RecordBatchIterator};
    use arrow_schema::{DataType, Field};
    use datafusion::{physical_plan::memory::MemoryExec, prelude::SessionContext};
    use futures::TryStreamExt;
    use lance_core::{ROW_ADDR, ROW_ID_FIELD};

    use crate::dataset::WriteParams;

    use super::*;

    async fn apply_to_batch(batch: RecordBatch, dataset: Arc<Dataset>) -> Result<RecordBatch> {
        let schema = batch.schema();
        let memory_exec = MemoryExec::try_new(&[vec![batch]], schema, None).unwrap();
        let exec = AddRowAddrExec::try_new(Arc::new(memory_exec), dataset, 0)?;
        let session = SessionContext::new();
        let task_ctx = session.task_ctx();
        let stream = exec.execute(0, task_ctx)?;
        let batches = stream.try_collect::<Vec<_>>().await?;
        assert_eq!(batches.len(), 1);
        Ok(batches.into_iter().next().unwrap())
    }

    #[tokio::test]
    async fn test_address_style_ids() {
        // Creating a dataset with no stable row ids means that the row address
        // will be the same as the row id.
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let reader = RecordBatchIterator::new(vec![], schema.clone());
        let dataset = Dataset::write(
            reader,
            "memory://",
            Some(WriteParams {
                enable_move_stable_row_ids: false,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let dataset = Arc::new(dataset);

        let rowids = Arc::new(UInt64Array::from(vec![1, 2, 3]));
        let schema = Schema::new(vec![ROW_ID_FIELD.clone()]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![rowids.clone()]).unwrap();

        let result = apply_to_batch(batch, dataset).await.unwrap();
        let result = result[ROW_ADDR].clone();

        assert_eq!(result.as_ref(), rowids.as_ref() as &dyn Array);
        // The array should be just a copy of the _rowid array pointer.
        assert_eq!(Arc::as_ptr(&result), Arc::as_ptr(&rowids));
    }

    async fn sample_dataset_with_rowid_index() -> Arc<Dataset> {
        // Create a row id index
        // 0 -> 0
        // 1 -> 1 << 32
        // 2 -> 2 << 32
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let dataset = Dataset::write(
            reader,
            "memory://",
            Some(WriteParams {
                enable_move_stable_row_ids: true,
                max_rows_per_file: 1,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 3);
        Arc::new(dataset)
    }

    #[tokio::test]
    async fn test_row_ids_no_nulls() {
        let dataset = sample_dataset_with_rowid_index().await;

        let rowids = Arc::new(UInt64Array::from(vec![0, 1, 2]));
        let schema = Schema::new(vec![ROW_ID_FIELD.clone()]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![rowids.clone()]).unwrap();

        let result = apply_to_batch(batch, dataset).await.unwrap();
        let result = result[ROW_ADDR].clone();

        assert_eq!(
            result.as_ref(),
            Arc::new(UInt64Array::from(vec![0, 1 << 32, 2 << 32])).as_ref() as &dyn Array
        );
    }

    #[tokio::test]
    async fn test_row_ids_with_nulls() {
        let dataset = sample_dataset_with_rowid_index().await;

        let rowids = Arc::new(UInt64Array::from(vec![Some(0), None, Some(2)]));
        let schema = Schema::new(vec![ROW_ID_FIELD.clone()]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![rowids.clone()]).unwrap();

        let result = apply_to_batch(batch, dataset).await.unwrap();
        let result = result[ROW_ADDR].clone();

        assert_eq!(
            result.as_ref(),
            Arc::new(UInt64Array::from(vec![Some(0), None, Some(2 << 32)])).as_ref() as &dyn Array
        );
    }

    #[tokio::test]
    async fn test_invalid_schema() {
        let dataset = sample_dataset_with_rowid_index().await;

        let rowids = Arc::new(Int32Array::from(vec![0, 1, 2]));
        let schema = Schema::new(vec![Field::new("invalid", DataType::Int32, true)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![rowids.clone()]).unwrap();

        let result = apply_to_batch(batch, dataset).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_stats() {
        let dataset = sample_dataset_with_rowid_index().await;

        let rowids = Arc::new(UInt64Array::from(vec![Some(0), None, Some(2)]));
        let schema = Arc::new(Schema::new(vec![ROW_ID_FIELD.clone()]));
        let batch = RecordBatch::try_new(schema.clone(), vec![rowids.clone()]).unwrap();

        let exec = AddRowAddrExec::try_new(
            Arc::new(MemoryExec::try_new(&[vec![batch.clone()]], schema.clone(), None).unwrap()),
            dataset.clone(),
            0,
        )
        .unwrap();
        let stats = exec.statistics().unwrap();
        let result = apply_to_batch(batch, dataset).await.unwrap();

        assert_eq!(stats.num_rows, Precision::Exact(3));
        assert_eq!(stats.column_statistics.len(), 2);
        assert_eq!(stats.column_statistics[0].null_count, Precision::Exact(1));
        assert_eq!(stats.column_statistics[1].null_count, Precision::Exact(1));

        let actual_byte_size = result
            .columns()
            .iter()
            .fold(0, |acc, col| acc + col.get_array_memory_size());
        assert_eq!(stats.total_byte_size, Precision::Exact(actual_byte_size));
    }
}
