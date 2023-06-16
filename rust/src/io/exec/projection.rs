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

//! Projection
//!

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_schema::{Schema as ArrowSchema, SchemaRef};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::physical_plan::{
    DisplayFormatType, ExecutionPlan, RecordBatchStream, SendableRecordBatchStream,
};
use futures::{stream, Stream, StreamExt, TryStreamExt};
use tokio::sync::mpsc::Receiver;
use tokio::task::JoinHandle;

use crate::arrow::*;
use crate::datatypes::Schema;
use crate::Result;

/// Executing Projection on a stream of record batches.
pub struct ProjectionStream {
    rx: Receiver<DataFusionResult<RecordBatch>>,

    _bg_thread: JoinHandle<()>,

    projection: Arc<ArrowSchema>,
}

impl ProjectionStream {
    fn new(input: SendableRecordBatchStream, projection: &Schema) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(2);

        let schema = Arc::new(ArrowSchema::try_from(projection).unwrap());
        let schema_clone = schema.clone();
        let bg_thread = tokio::spawn(async move {
            if let Err(e) = input
                .zip(stream::repeat_with(|| schema_clone.clone()))
                .then(|(batch, schema)| async move {
                    let batch = batch?;
                    batch.project_by_schema(schema.as_ref())
                })
                .map(|r| r.map_err(|e| DataFusionError::Execution(e.to_string())))
                .try_for_each(|b| async {
                    if tx.is_closed() {
                        eprintln!("ExecNode(Projection): channel closed");
                        return Err(DataFusionError::Execution(
                            "ExecNode(Projection): channel closed".to_string(),
                        ));
                    }
                    if let Err(e) = tx.send(Ok(b)).await {
                        eprintln!("ExecNode(Projection): {}", e);
                        return Err(DataFusionError::Execution(
                            "ExecNode(Projection): channel closed".to_string(),
                        ));
                    }
                    Ok(())
                })
                .await
            {
                if let Err(e) = tx.send(Err(e)).await {
                    eprintln!("ExecNode(Projection): {}", e);
                }
            }
        });

        Self {
            rx,
            _bg_thread: bg_thread,
            projection: schema,
        }
    }
}

impl Stream for ProjectionStream {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::into_inner(self).rx.poll_recv(cx)
    }
}

impl RecordBatchStream for ProjectionStream {
    fn schema(&self) -> arrow_schema::SchemaRef {
        self.projection.clone()
    }
}

pub(crate) struct ProjectionExec {
    input: Arc<dyn ExecutionPlan>,
    project: Arc<Schema>,
}

impl std::fmt::Debug for ProjectionExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let columns = self
            .project
            .fields
            .iter()
            .map(|f| f.name.clone())
            .collect::<Vec<_>>();
        write!(
            f,
            "Projection(schema={:?},\n\tchild={:?})",
            columns, self.input
        )
    }
}

impl ProjectionExec {
    pub fn try_new(input: Arc<dyn ExecutionPlan>, project: Arc<Schema>) -> Result<Self> {
        Ok(Self { input, project })
    }
}

impl ExecutionPlan for ProjectionExec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        let arrow_schema =
            ArrowSchema::try_from(self.project.as_ref()).expect("convert to arrow schema");
        arrow_schema.into()
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
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        todo!()
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> datafusion::error::Result<datafusion::physical_plan::SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        Ok(Box::pin(ProjectionStream::new(input_stream, &self.project)))
    }

    fn statistics(&self) -> datafusion::physical_plan::Statistics {
        self.input.statistics()
    }

    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let columns = self
            .project
            .fields
            .iter()
            .map(|f| f.name.clone())
            .collect::<Vec<_>>();
        write!(f, "LanceProjectionExec: projection={:?}", columns)
    }
}
