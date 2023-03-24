// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::{cast::as_primitive_array, RecordBatch, UInt64Array};
use arrow_schema::{Schema as ArrowSchema, SchemaRef};
use datafusion::error::{DataFusionError, Result};
use datafusion::physical_plan::{ExecutionPlan, RecordBatchStream, SendableRecordBatchStream};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use tokio::sync::mpsc::{self, Receiver};
use tokio::task::JoinHandle;

use crate::dataset::{Dataset, ROW_ID};
use crate::datatypes::Schema;
use crate::{arrow::*, Error};

/// Dataset Take Node.
///
/// [Take] node takes the filtered batch from the child node.
///
/// It uses the `_rowid` to random access on [Dataset] to gather the final results.
pub struct Take {
    rx: Receiver<Result<RecordBatch>>,
    _bg_thread: JoinHandle<()>,

    output_schema: SchemaRef,
}

impl Take {
    /// Create a Take node with
    ///
    ///  - Dataset: the dataset to read from
    ///  - projection: extra columns to take from the dataset.
    ///  - output_schema: the output schema of the take node.
    ///  - child: the upstream stream to feed data in.
    fn new(
        dataset: Arc<Dataset>,
        projection: Arc<Schema>,
        output_schema: SchemaRef,
        child: SendableRecordBatchStream,
    ) -> Self {
        let (tx, rx) = mpsc::channel(4);

        let bg_thread = tokio::spawn(async move {
            if let Err(e) = child
                .zip(stream::repeat_with(|| {
                    (dataset.clone(), projection.clone())
                }))
                .then(|(batch, (dataset, extra))| async move {
                    let batch = batch?;
                    let row_id_arr = batch.column_by_name(ROW_ID).unwrap();
                    let row_ids: &UInt64Array = as_primitive_array(row_id_arr);
                    let rows = if extra.fields.is_empty() {
                        batch
                    } else {
                        dataset
                            .take_rows(row_ids.values(), &extra)
                            .await?
                            .merge(&batch)?
                    };
                    Ok::<RecordBatch, Error>(rows)
                })
                .map(|r| r.map_err(|e| DataFusionError::Execution(e.to_string())))
                .try_for_each(|b| async {
                    if tx.is_closed() {
                        eprintln!("ExecNode(Take): channel closed");
                        return Err(DataFusionError::Execution(
                            "ExecNode(Take): channel closed".to_string(),
                        ));
                    }
                    if let Err(e) = tx.send(Ok(b)).await {
                        eprintln!("ExecNode(Take): {}", e);
                        return Err(DataFusionError::Execution(
                            "ExecNode(Take): channel closed".to_string(),
                        ));
                    }
                    Ok(())
                })
                .await
            {
                if let Err(e) = tx.send(Err(e)).await {
                    eprintln!("ExecNode(Take): {}", e);
                }
            }
            drop(tx)
        });

        Self {
            rx,
            _bg_thread: bg_thread,
            output_schema,
        }
    }
}

impl Stream for Take {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::into_inner(self).rx.poll_recv(cx)
    }
}

impl RecordBatchStream for Take {
    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }
}

/// [`TakeExec`] is a [`ExecutionPlan`] that enriches the input [`RecordBatch`]
/// with extra columns from [`Dataset`].
///
/// The rows are identified by the inexplicit row IDs from `input` plan.
///
/// The output schema will be the input schema, merged with extra schemas from the dataset.
#[derive(Debug)]
pub(crate) struct TakeExec {
    /// Dataset to read from.
    dataset: Arc<Dataset>,

    pub(crate) extra_schema: Arc<Schema>,

    input: Arc<dyn ExecutionPlan>,

    /// Output schema is the merged schema between input schema and extra schema.
    output_schema: Schema,
}

impl TakeExec {
    /// Create a [`TakeExec`] node.
    ///
    /// - dataset: the dataset to read from
    /// - input: the upstream [`ExecutionPlan`] to feed data in.
    /// - extra_schema: the extra schema to take / read from the dataset.
    pub fn try_new(
        dataset: Arc<Dataset>,
        input: Arc<dyn ExecutionPlan>,
        extra_schema: Arc<Schema>,
    ) -> Result<Self> {
        if input.schema().column_with_name(ROW_ID).is_none() {
            return Err(DataFusionError::Plan(
                "TakeExec requires the input plan to have a column named '_rowid'".to_string(),
            ));
        }

        let input_schema = Schema::try_from(input.schema().as_ref())?;
        let output_schema = input_schema.merge(&extra_schema);

        let remaining_schema = extra_schema.exclude(&input_schema)?;

        Ok(Self {
            dataset,
            extra_schema: Arc::new(remaining_schema),
            input,
            output_schema,
        })
    }
}

impl ExecutionPlan for TakeExec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        ArrowSchema::try_from(&self.output_schema).unwrap().into()
    }

    fn output_partitioning(&self) -> datafusion::physical_plan::Partitioning {
        self.input.output_partitioning()
    }

    fn output_ordering(&self) -> Option<&[datafusion::physical_expr::PhysicalSortExpr]> {
        self.input.output_ordering()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            dataset: self.dataset.clone(),
            extra_schema: self.extra_schema.clone(),
            input: _children[0].clone(),
            output_schema: self.output_schema.clone(),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        Ok(Box::pin(Take::new(
            self.dataset.clone(),
            self.extra_schema.clone(),
            self.schema(),
            input_stream,
        )))
    }

    fn statistics(&self) -> datafusion::physical_plan::Statistics {
        self.input.statistics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{ArrayRef, Float32Array, Int32Array, RecordBatchReader, StringArray};
    use arrow_schema::{DataType, Field};
    use tempfile::tempdir;

    use crate::{dataset::WriteParams, io::exec::LanceScanExec};

    async fn create_dataset() -> Arc<Dataset> {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("f", DataType::Float32, false),
            Field::new("s", DataType::Utf8, false),
        ]));

        // Write 3 batches.
        let expected_batches: Vec<RecordBatch> = (0..3)
            .map(|batch_id| {
                let value_range = batch_id * 10..batch_id * 10 + 10;
                let columns: Vec<ArrayRef> = vec![
                    Arc::new(Int32Array::from_iter_values(value_range.clone())),
                    Arc::new(Float32Array::from_iter(
                        value_range.clone().map(|v| v as f32),
                    )),
                    Arc::new(StringArray::from_iter_values(
                        value_range.clone().map(|v| format!("str-{v}")),
                    )),
                ];
                RecordBatch::try_new(schema.clone(), columns).unwrap()
            })
            .collect();
        let batches = RecordBatchBuffer::new(expected_batches.clone());

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut params = WriteParams::default();
        params.max_rows_per_group = 10;
        let mut reader: Box<dyn RecordBatchReader> = Box::new(batches);
        Dataset::write(&mut reader, test_uri, Some(params))
            .await
            .unwrap();

        Arc::new(Dataset::open(test_uri).await.unwrap())
    }

    #[tokio::test]
    async fn test_take_schema() {
        let dataset = create_dataset().await;

        let scan_arrow_schema = ArrowSchema::new(vec![Field::new("i", DataType::Int32, false)]);
        let scan_schema = Arc::new(Schema::try_from(&scan_arrow_schema).unwrap());

        let extra_arrow_schema = ArrowSchema::new(vec![Field::new("s", DataType::Int32, false)]);
        let extra_schema = Arc::new(Schema::try_from(&extra_arrow_schema).unwrap());

        // With row id
        let input = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            scan_schema.clone(),
            10,
            10,
            true,
        ));
        let take_exec = TakeExec::try_new(dataset.clone(), input, extra_schema.clone()).unwrap();
        let schema = take_exec.schema();
        assert_eq!(
            schema.fields.iter().map(|f| f.name()).collect::<Vec<_>>(),
            vec!["i", "_rowid", "s"]
        );
    }

    #[tokio::test]
    async fn test_take_no_extra_columns() {
        let dataset = create_dataset().await;

        let scan_arrow_schema = ArrowSchema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("s", DataType::Int32, false),
        ]);
        let scan_schema = Arc::new(Schema::try_from(&scan_arrow_schema).unwrap());

        // Extra column is already read.
        let extra_arrow_schema = ArrowSchema::new(vec![Field::new("s", DataType::Int32, false)]);
        let extra_schema = Arc::new(Schema::try_from(&extra_arrow_schema).unwrap());

        let input = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            scan_schema.clone(),
            10,
            10,
            true,
        ));
        let take_exec = TakeExec::try_new(dataset.clone(), input, extra_schema.clone()).unwrap();
        let schema = take_exec.schema();
        assert_eq!(
            schema.fields.iter().map(|f| f.name()).collect::<Vec<_>>(),
            vec!["i", "s", "_rowid"]
        );
    }

    #[tokio::test]
    async fn test_take_no_row_id() {
        let dataset = create_dataset().await;

        let scan_arrow_schema = ArrowSchema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("s", DataType::Int32, false),
        ]);
        let scan_schema = Arc::new(Schema::try_from(&scan_arrow_schema).unwrap());

        let extra_arrow_schema = ArrowSchema::new(vec![Field::new("s", DataType::Int32, false)]);
        let extra_schema = Arc::new(Schema::try_from(&extra_arrow_schema).unwrap());

        // No row ID
        let input = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            scan_schema.clone(),
            10,
            10,
            false,
        ));
        assert!(TakeExec::try_new(dataset.clone(), input, extra_schema.clone()).is_err());
    }
}
