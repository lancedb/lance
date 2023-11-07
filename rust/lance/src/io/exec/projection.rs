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
    DisplayAs, DisplayFormatType, ExecutionPlan, RecordBatchStream, SendableRecordBatchStream,
};
use futures::{stream, FutureExt, Stream, StreamExt, TryStreamExt};
use tokio::sync::mpsc::Receiver;
use tokio::task::JoinHandle;

use crate::arrow::*;
use crate::datatypes::Schema;
use crate::Result;

/// Executing Projection on a stream of record batches.
pub struct ProjectionStream {
    rx: Receiver<DataFusionResult<RecordBatch>>,

    bg_thread: Option<JoinHandle<()>>,

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
                    if tx.send(Ok(b)).await.is_err() {
                        // If channel is closed, make sure we return an error to end the stream.
                        return Err(DataFusionError::Internal(
                            "ExecNode(Projection): channel closed".to_string(),
                        ));
                    }
                    Ok(())
                })
                .await
            {
                if let Err(e) = tx.send(Err(e)).await {
                    if let Err(e) = e.0 {
                        // if channel was closed, it was cancelled by the receiver.
                        // But if there was a different error we should send it
                        // or log it.
                        if !e.to_string().contains("channel closed") {
                            log::error!("channel was closed by receiver, but error occurred in background thread: {:?}", e);
                        }
                    }
                }
            }
        });

        Self {
            rx,
            bg_thread: Some(bg_thread),
            projection: schema,
        }
    }
}

impl Stream for ProjectionStream {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = Pin::into_inner(self);
        // We need to check the JoinHandle to make sure the thread hasn't panicked.
        let bg_thread_completed = if let Some(bg_thread) = &mut this.bg_thread {
            match bg_thread.poll_unpin(cx) {
                Poll::Ready(Ok(())) => true,
                Poll::Ready(Err(join_error)) => {
                    return Poll::Ready(Some(Err(DataFusionError::Execution(format!(
                        "ExecNode(Projection): thread panicked: {}",
                        join_error
                    )))));
                }
                Poll::Pending => false,
            }
        } else {
            false
        };
        if bg_thread_completed {
            // Need to take it, since we aren't allowed to poll if again after.
            this.bg_thread.take();
        }
        // this.rx.
        this.rx.poll_recv(cx)
    }
}

impl RecordBatchStream for ProjectionStream {
    fn schema(&self) -> arrow_schema::SchemaRef {
        self.projection.clone()
    }
}

#[derive(Debug)]
pub struct ProjectionExec {
    input: Arc<dyn ExecutionPlan>,
    project: Arc<Schema>,
}

impl DisplayAs for ProjectionExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let columns = self
                    .project
                    .fields
                    .iter()
                    .map(|f| f.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "Projection: fields=[{}]", columns)
            }
        }
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
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::try_new(
            children[0].clone(),
            self.project.clone(),
        )?))
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
}
