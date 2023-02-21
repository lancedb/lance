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
use arrow_schema::{SchemaRef, Schema as ArrowSchema};
use datafusion::error::{DataFusionError, Result};
use datafusion::physical_plan::{
    ExecutionPlan, RecordBatchStream, SendableRecordBatchStream, Statistics,
};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
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
    fn new(dataset: Arc<Dataset>, schema: Arc<Schema>, child: SendableRecordBatchStream) -> Self {
        let (tx, rx) = mpsc::channel(4);

        let projection = schema.clone();
        let bg_thread = tokio::spawn(async move {
            if let Err(e) = child
                .zip(stream::repeat_with(|| {
                    (dataset.clone(), projection.clone())
                }))
                .then(|(batch, (dataset, projection))| async move {
                    let batch = batch?;
                    let row_id_arr = batch.column_by_name("_rowid").unwrap();
                    let row_ids: &UInt64Array = as_primitive_array(row_id_arr);
                    let rows = dataset.take_rows(row_ids.values(), &projection).await?;
                    rows.merge(&batch)?.drop_column("_rowid")
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
}

impl GlobalTakeExec {
    pub fn new(dataset: Arc<Dataset>, schema: Arc<Schema>, input: Arc<dyn ExecutionPlan>) -> Self {
        Self {
            dataset,
            schema,
            input,
        }
    }
}

impl ExecutionPlan for GlobalTakeExec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.input.schema()
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
    ) -> Result<Self> {
        let (tx, rx) = mpsc::channel(4);

        let inner_schema = Schema::try_from(input.schema().as_ref())?;
        let take_schema = schema.exclude(&inner_schema)?;
        let projection = schema.clone();

        let _bg_thread = tokio::spawn(async move {
            if let Err(e) = input
                .zip(stream::repeat_with(|| {
                    (dataset.clone(), take_schema.clone(), projection.clone())
                }))
                .then(|(b, (dataset, take_schema, projection))| async move {
                    // TODO: need to cache the fragments.
                    let batch = b?;
                    let row_id_arr = batch.column_by_name(ROW_ID).unwrap();
                    let row_ids: &UInt64Array = as_primitive_array(row_id_arr);
                    let remaining_columns =
                        dataset.take_rows(row_ids.values(), &take_schema).await?;
                    let projection_schema = ArrowSchema::from(projection.as_ref());
                    Ok(batch
                        .merge(&remaining_columns)?
                        .drop_column(ROW_ID)?
                        .project_by_schema(&projection_schema)?)
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

impl<'a> RecordBatchStream for LocalTake {
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
}

impl LocalTakeExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, dataset: Arc<Dataset>, schema: Arc<Schema>) -> Self {
        assert!(input.schema().column_with_name(ROW_ID).is_some());
        Self {
            dataset,
            input,
            schema,
        }
    }
}

impl ExecutionPlan for LocalTakeExec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::new(self.schema.as_ref().into())
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
