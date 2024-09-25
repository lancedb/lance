// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::{cast::as_primitive_array, RecordBatch, UInt64Array};
use arrow_schema::{Schema as ArrowSchema, SchemaRef};
use datafusion::common::Statistics;
use datafusion::error::{DataFusionError, Result};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, RecordBatchStream,
    SendableRecordBatchStream,
};
use datafusion_physical_expr::EquivalenceProperties;
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use futures::{Future, FutureExt};
use tokio::sync::mpsc::{self, Receiver};
use tokio::task::JoinHandle;
use tracing::{instrument, Instrument};

use crate::dataset::{Dataset, ProjectionRequest, ROW_ID};
use crate::datatypes::Schema;
use crate::{arrow::*, Error};

/// Dataset Take Node.
///
/// [Take] node takes the filtered batch from the child node.
///
/// It uses the `_rowid` to random access on [Dataset] to gather the final results.
pub struct Take {
    rx: Receiver<Result<RecordBatch>>,
    bg_thread: Option<JoinHandle<()>>,

    output_schema: SchemaRef,
}

impl Take {
    /// Create a Take node with
    ///
    ///  - Dataset: the dataset to read from
    ///  - projection: extra columns to take from the dataset.
    ///  - output_schema: the output schema of the take node.
    ///  - child: the upstream stream to feed data in.
    ///  - batch_readahead: max number of batches to readahead, potentially concurrently
    #[instrument(level = "debug", skip_all, name = "Take::new")]
    fn new(
        dataset: Arc<Dataset>,
        projection: Arc<Schema>,
        output_schema: SchemaRef,
        child: SendableRecordBatchStream,
        batch_readahead: usize,
    ) -> Self {
        let (tx, rx) = mpsc::channel(4);

        let bg_thread = tokio::spawn(
            async move {
                if let Err(e) = child
                    .zip(stream::repeat_with(|| {
                        (dataset.clone(), projection.clone())
                    }))
                    .map(|(batch, (dataset, extra))| async move {
                        Self::take_batch(batch?, dataset, extra).await
                    })
                    .buffered(batch_readahead)
                    .map(|r| r.map_err(|e| DataFusionError::Execution(e.to_string())))
                    .try_for_each(|b| async {
                        if tx.send(Ok(b)).await.is_err() {
                        // If channel is closed, make sure we return an error to end the stream. 
                        return Err(DataFusionError::Internal(
                            "ExecNode(Take): channel closed".to_string(),
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
                drop(tx)
            }
            .in_current_span(),
        );

        Self {
            rx,
            bg_thread: Some(bg_thread),
            output_schema,
        }
    }

    /// Given a batch with a _rowid column, retrieve extra columns from dataset.
    // This method mostly exists to annotate the Send bound so the compiler
    // doesn't produce a higher-order lifetime error.
    // manually implemented async for Send bound
    #[allow(clippy::manual_async_fn)]
    #[instrument(level = "debug", skip_all)]
    fn take_batch(
        batch: RecordBatch,
        dataset: Arc<Dataset>,
        extra: Arc<Schema>,
    ) -> impl Future<Output = Result<RecordBatch, Error>> + Send {
        async move {
            let row_id_arr = batch.column_by_name(ROW_ID).unwrap();
            let row_ids: &UInt64Array = as_primitive_array(row_id_arr);
            let rows = if extra.fields.is_empty() {
                batch
            } else {
                let new_columns = dataset
                    .take_rows(row_ids.values(), ProjectionRequest::Schema(extra))
                    .await?;
                debug_assert_eq!(batch.num_rows(), new_columns.num_rows());
                batch.merge(&new_columns)?
            };
            Ok::<RecordBatch, Error>(rows)
        }
        .in_current_span()
    }
}

impl Stream for Take {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = Pin::into_inner(self);
        // We need to check the JoinHandle to make sure the thread hasn't panicked.
        let bg_thread_completed = if let Some(bg_thread) = &mut this.bg_thread {
            match bg_thread.poll_unpin(cx) {
                Poll::Ready(Ok(())) => true,
                Poll::Ready(Err(join_error)) => {
                    return Poll::Ready(Some(Err(DataFusionError::Execution(format!(
                        "ExecNode(Take): thread panicked: {}",
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
pub struct TakeExec {
    /// Dataset to read from.
    dataset: Arc<Dataset>,

    pub(crate) extra_schema: Arc<Schema>,

    input: Arc<dyn ExecutionPlan>,

    /// Output schema is the merged schema between input schema and extra schema.
    output_schema: Schema,

    batch_readahead: usize,

    properties: PlanProperties,
}

impl DisplayAs for TakeExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let extra_fields = self
            .extra_schema
            .fields
            .iter()
            .map(|f| f.name.clone())
            .collect::<HashSet<_>>();
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let columns = self
                    .output_schema
                    .fields
                    .iter()
                    .map(|f| {
                        if extra_fields.contains(&f.name) {
                            format!("({})", f.name.as_str())
                        } else {
                            f.name.clone()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "Take: columns={:?}", columns)
            }
        }
    }
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
        batch_readahead: usize,
    ) -> Result<Self> {
        if input.schema().column_with_name(ROW_ID).is_none() {
            return Err(DataFusionError::Plan(
                "TakeExec requires the input plan to have a column named '_rowid'".to_string(),
            ));
        }

        let input_schema = Schema::try_from(input.schema().as_ref())?;
        let output_schema = input_schema.merge(extra_schema.as_ref())?;

        let remaining_schema = extra_schema.exclude(&input_schema)?;

        let output_arrow = Arc::new(ArrowSchema::from(&output_schema));
        let properties = input
            .properties()
            .clone()
            .with_eq_properties(EquivalenceProperties::new(output_arrow));

        Ok(Self {
            dataset,
            extra_schema: Arc::new(remaining_schema),
            input,
            output_schema,
            batch_readahead,
            properties,
        })
    }

    /// Get the dataset.
    ///
    /// WARNING: Internal API with no stability guarantees.
    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }
}

impl ExecutionPlan for TakeExec {
    fn name(&self) -> &str {
        "TakeExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        ArrowSchema::from(&self.output_schema).into()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    /// This preserves the output schema.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "TakeExec wrong number of children".to_string(),
            ));
        }

        let child = &children[0];

        let extra_schema = self.output_schema.exclude(child.schema().as_ref())?;

        let plan = Self::try_new(
            self.dataset.clone(),
            children[0].clone(),
            Arc::new(extra_schema),
            self.batch_readahead,
        )?;

        Ok(Arc::new(plan))
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
            self.batch_readahead,
        )))
    }

    fn statistics(&self) -> Result<datafusion::physical_plan::Statistics> {
        Ok(Statistics {
            num_rows: self.input.statistics()?.num_rows,
            ..Statistics::new_unknown(self.schema().as_ref())
        })
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{ArrayRef, Float32Array, Int32Array, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field};
    use tempfile::tempdir;

    use crate::{
        dataset::{scanner::DEFAULT_IO_BUFFER_SIZE, WriteParams},
        io::exec::LanceScanExec,
    };

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
                        value_range.map(|v| format!("str-{v}")),
                    )),
                ];
                RecordBatch::try_new(schema.clone(), columns).unwrap()
            })
            .collect();

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let params = WriteParams {
            max_rows_per_group: 10,
            ..Default::default()
        };
        let reader =
            RecordBatchIterator::new(expected_batches.clone().into_iter().map(Ok), schema.clone());
        Dataset::write(reader, test_uri, Some(params))
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
            None,
            scan_schema,
            10,
            10,
            Some(4),
            *DEFAULT_IO_BUFFER_SIZE,
            true,
            false,
            false,
            true,
        ));
        let take_exec = TakeExec::try_new(dataset, input, extra_schema, 10).unwrap();
        let schema = take_exec.schema();
        assert_eq!(
            schema.fields.iter().map(|f| f.name()).collect::<Vec<_>>(),
            vec!["i", ROW_ID, "s"]
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
            None,
            scan_schema,
            10,
            10,
            Some(4),
            *DEFAULT_IO_BUFFER_SIZE,
            true,
            false,
            false,
            true,
        ));
        let take_exec = TakeExec::try_new(dataset, input, extra_schema, 10).unwrap();
        let schema = take_exec.schema();
        assert_eq!(
            schema.fields.iter().map(|f| f.name()).collect::<Vec<_>>(),
            vec!["i", "s", ROW_ID]
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
            None,
            scan_schema,
            10,
            10,
            Some(4),
            *DEFAULT_IO_BUFFER_SIZE,
            false,
            false,
            false,
            true,
        ));
        assert!(TakeExec::try_new(dataset, input, extra_schema, 10).is_err());
    }

    #[tokio::test]
    async fn test_with_new_children() -> Result<()> {
        let dataset = create_dataset().await;

        let input = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            Arc::new(dataset.schema().project(&["i"])?),
            10,
            10,
            Some(4),
            *DEFAULT_IO_BUFFER_SIZE,
            true,
            false,
            false,
            true,
        ));
        assert_eq!(input.schema().field_names(), vec!["i", ROW_ID],);
        let take_exec = TakeExec::try_new(
            dataset.clone(),
            input.clone(),
            Arc::new(dataset.schema().project(&["s"])?),
            10,
        )?;
        assert_eq!(take_exec.schema().field_names(), vec!["i", ROW_ID, "s"],);
        let outer_take = Arc::new(TakeExec::try_new(
            dataset.clone(),
            Arc::new(take_exec),
            Arc::new(dataset.schema().project(&["f"])?),
            10,
        )?);
        assert_eq!(
            outer_take.schema().field_names(),
            vec!["i", ROW_ID, "s", "f"],
        );

        // with_new_children should preserve the output schema.
        let edited = outer_take.with_new_children(vec![input])?;
        assert_eq!(edited.schema().field_names(), vec!["i", ROW_ID, "s", "f"],);
        Ok(())
    }
}
