// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Projection
//!

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_schema::{Schema as ArrowSchema, SchemaRef};
use datafusion::common::Statistics;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, RecordBatchStream,
    SendableRecordBatchStream,
};
use datafusion_physical_expr::EquivalenceProperties;
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

        let schema = Arc::new(ArrowSchema::from(projection));
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
    properties: PlanProperties,
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
        let arrow_schema = ArrowSchema::from(project.as_ref());
        // TODO: we reset the EquivalenceProperties here but we could probably just project
        // them, that way ordering is maintained (or just use DF project?)
        let properties = input
            .properties()
            .clone()
            .with_eq_properties(EquivalenceProperties::new(Arc::new(arrow_schema)));
        Ok(Self {
            input,
            project,
            properties,
        })
    }
}

impl ExecutionPlan for ProjectionExec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        let arrow_schema = ArrowSchema::from(self.project.as_ref());
        arrow_schema.into()
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

    fn statistics(&self) -> datafusion::error::Result<datafusion::physical_plan::Statistics> {
        let num_rows = self.input.statistics()?.num_rows;
        Ok(Statistics {
            num_rows,
            ..datafusion::physical_plan::Statistics::new_unknown(self.schema().as_ref())
        })
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }
}
