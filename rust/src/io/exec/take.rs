// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::cast::as_primitive_array;
use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema as ArrowSchema, SchemaRef};
use datafusion::error::{DataFusionError, Result};
use datafusion::physical_plan::{
    ExecutionPlan, RecordBatchStream, SendableRecordBatchStream, Statistics,
};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use futures::{FutureExt, TryFutureExt};
use tokio::sync::mpsc::{self, Receiver};
use tokio::task::JoinHandle;

use crate::arrow::*;
use crate::dataset::{Dataset, ROW_ID};
use crate::datatypes::Schema;

/// Dataset Take Node.
///
/// [Take] node takes the filtered batch from the child node.
/// It uses the `_rowid` to random access on [Dataset] to gather the final results.
pub struct Take {
    rx: Receiver<Result<RecordBatch>>,
    _bg_thread: JoinHandle<()>,

    schema: Arc<Schema>,
}

impl Take {
    /// Create a Take node with
    ///
    ///  - Dataset: the dataset to read from
    ///  - schema: projection schema for take node.
    ///  - child: the upstream ExedNode to feed data in.
    fn new(
        dataset: Arc<Dataset>,
        schema: Arc<Schema>,
        child: SendableRecordBatchStream,
        drop_row_id: bool,
    ) -> Self {
        let (tx, rx) = mpsc::channel(4);

        let projection = schema.clone();
        let bg_thread = tokio::spawn(async move {
            if let Err(e) = child
                .zip(stream::repeat_with(|| {
                    (dataset.clone(), projection.clone())
                }))
                .then(|(batch, (dataset, projection))| async move {
                    let batch = batch?;
                    // println!("GlobalTake Batch is {:?}", batch);
                    let row_id_arr = batch.column_by_name(ROW_ID).unwrap();
                    let row_ids: &UInt64Array = as_primitive_array(row_id_arr);
                    let rows = if projection.fields.is_empty() {
                        batch
                    } else {
                        dataset
                            .take_rows(row_ids.values(), &projection)
                            .await?
                            .merge(&batch)?
                    };
                    // println!(
                    //    "Global batch after merge is: drop_column={drop_row_id} {:?}",
                    //    rows
                    //);
                    if drop_row_id {
                        rows.drop_column(ROW_ID)
                    } else {
                        Ok(rows)
                    }
                })
                .map(|r| {
                    r.map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))
                })
                .try_for_each(|b| async {
                    if tx.is_closed() {
                        eprintln!("ExecNode(Take): channel closed");
                        return Err(datafusion::error::DataFusionError::Execution(
                            "ExecNode(Take): channel closed".to_string(),
                        ));
                    }
                    if let Err(e) = tx.send(Ok(b)).await {
                        eprintln!("ExecNode(Take): {}", e);
                        return Err(datafusion::error::DataFusionError::Execution(
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
            schema,
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
        Arc::new(self.schema.as_ref().into())
    }
}

/// GlobalTake is a physical [`ExecutionPlan`] that takes the rows globally cross the [`Dataset`].
///
/// The rows are identified by the inexplicit row IDs.
#[derive(Debug)]
pub struct GlobalTakeExec {
    dataset: Arc<Dataset>,
    schema: Arc<Schema>,
    input: Arc<dyn ExecutionPlan>,
    drop_row_id: bool,
}

impl GlobalTakeExec {
    pub fn new(
        dataset: Arc<Dataset>,
        schema: Arc<Schema>,
        input: Arc<dyn ExecutionPlan>,
        drop_row_id: bool,
    ) -> Self {
        assert!(input.schema().column_with_name(ROW_ID).is_some());
        Self {
            dataset,
            schema,
            input,
            drop_row_id,
        }
    }
}

fn projection_with_row_id(projection: &Schema, drop_row_id: bool) -> SchemaRef {
    let schema = ArrowSchema::from(projection);
    if drop_row_id {
        Arc::new(schema)
    } else {
        let mut fields = schema.fields;
        fields.push(Field::new(ROW_ID, DataType::UInt64, false));
        Arc::new(ArrowSchema::new(fields))
    }
}

impl ExecutionPlan for GlobalTakeExec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        projection_with_row_id(&self.schema, self.drop_row_id)
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
        todo!()
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        Ok(Box::pin(Take::new(
            self.dataset.clone(),
            self.schema.clone(),
            input_stream,
            self.drop_row_id,
        )))
    }

    fn statistics(&self) -> datafusion::physical_plan::Statistics {
        todo!()
    }
}

pub struct LocalTake {
    /// The output schema.
    schema: Arc<Schema>,

    rx: Receiver<Result<RecordBatch>>,
    _bg_thread: JoinHandle<()>,
}

impl LocalTake {
    pub fn try_new(
        input: SendableRecordBatchStream,
        dataset: Arc<Dataset>,
        schema: Arc<Schema>,
        ann_schema: Option<Arc<Schema>>, // TODO add input/output schema contract to exec nodes and remove this
        drop_row_id: bool,
    ) -> Result<Self> {
        let (tx, rx) = mpsc::channel(4);
        let inner_schema = Schema::try_from(input.schema().as_ref())?;
        let mut take_schema = schema.exclude(&inner_schema)?;
        if ann_schema.is_some() {
            take_schema = take_schema.exclude(&ann_schema.unwrap())?;
        }
        let projection = schema.clone();

        let _bg_thread = tokio::spawn(async move {
            if let Err(e) = input
                .zip(stream::repeat_with(|| {
                    (dataset.clone(), take_schema.clone(), projection.clone())
                }))
                .then(|(b, (dataset, take_schema, projection))| async move {
                    // TODO: need to cache the fragments.
                    let batch = b?;
                    let projection_schema = ArrowSchema::from(projection.as_ref());
                    if batch.num_rows() == 0 {
                        return Ok(RecordBatch::new_empty(Arc::new(projection_schema)));
                    }

                    let row_id_arr = batch.column_by_name(ROW_ID).unwrap();
                    let row_ids: &UInt64Array = as_primitive_array(row_id_arr);
                    let batch = if take_schema.fields.is_empty() {
                        batch.project_by_schema(&projection_schema)?
                    } else {
                        let remaining_columns =
                            dataset.take_rows(row_ids.values(), &take_schema).await?;
                        batch
                            .merge(&remaining_columns)?
                            .project_by_schema(&projection_schema)?
                    };

                    if !drop_row_id {
                        Ok(batch.try_with_column(
                            Field::new(ROW_ID, DataType::UInt64, false),
                            Arc::new(row_id_arr.clone()),
                        )?)
                    } else {
                        Ok(batch)
                    }
                })
                .try_for_each(|b| async {
                    if tx.is_closed() {
                        return Err(datafusion::error::DataFusionError::Execution(
                            "ExecNode(Take): channel closed".to_string(),
                        ));
                    }
                    if let Err(_) = tx.send(Ok(b)).await {
                        return Err(datafusion::error::DataFusionError::Execution(
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
        Ok(Self {
            schema,
            rx,
            _bg_thread,
        })
    }
}

impl Stream for LocalTake {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::into_inner(self).rx.poll_recv(cx)
    }
}

impl RecordBatchStream for LocalTake {
    fn schema(&self) -> SchemaRef {
        Arc::new(self.schema.as_ref().into())
    }
}

/// [LocalTakeExec] is a physical [`ExecutionPlan`] that takes the rows within the same fragment
/// as its children [super::LanceScanExec] node.
///
/// It is used to support filter/predicates push-down:
///
///  `LocalTakeExec` -> `FilterExec` -> `LanceScanExec`:
///
#[derive(Debug)]
pub struct LocalTakeExec {
    dataset: Arc<Dataset>,
    input: Arc<dyn ExecutionPlan>,
    schema: Arc<Schema>,
    ann_schema: Option<Arc<Schema>>,
    drop_row_id: bool,
}

impl LocalTakeExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        schema: Arc<Schema>,
        ann_schema: Option<Arc<Schema>>,
        drop_row_id: bool,
    ) -> Self {
        assert!(input.schema().column_with_name(ROW_ID).is_some());
        Self {
            dataset,
            input,
            schema,
            ann_schema,
            drop_row_id,
        }
    }
}

impl ExecutionPlan for LocalTakeExec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        projection_with_row_id(&self.schema, self.drop_row_id)
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
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Plan(
                "LocalTakeExec only takes 1 child".to_string(),
            ));
        }
        Ok(Arc::new(Self {
            input: children[0].clone(),
            dataset: self.dataset.clone(),
            schema: self.schema.clone(),
            ann_schema: self.ann_schema.clone(),
            drop_row_id: self.drop_row_id,
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        Ok(Box::pin(LocalTake::try_new(
            input_stream,
            self.dataset.clone(),
            self.schema.clone(),
            self.ann_schema.clone(),
            self.drop_row_id,
        )?))
    }

    fn statistics(&self) -> datafusion::physical_plan::Statistics {
        Statistics {
            num_rows: None,
            total_byte_size: None,
            column_statistics: None,
            is_exact: false,
        }
    }
}
