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
use lance_core::datatypes::{Field, OnMissing, Projection};
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

        let output_schema_copy = output_schema.clone();
        let bg_thread = tokio::spawn(
            async move {
                if let Err(e) = child
                    .zip(stream::repeat_with(|| {
                        (dataset.clone(), projection.clone())
                    }))
                    .map(|(batch, (dataset, extra))| {
                        let output_schema_copy = output_schema_copy.clone();
                        async move {
                            Self::take_batch(batch?, dataset, extra, output_schema_copy).await
                        }})
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
        output_schema: SchemaRef,
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
                batch.merge_with_schema(&new_columns, &output_schema)?
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
    pub(crate) dataset: Arc<Dataset>,

    /// The original projection is kept to recalculate `with_new_children`.
    pub(crate) original_projection: Arc<Schema>,

    /// The schema to pass to dataset.take, this should be the original projection
    /// minus any fields in the input schema.
    schema_to_take: Arc<Schema>,

    input: Arc<dyn ExecutionPlan>,

    /// Output schema is the merged schema between input schema and extra schema and
    /// tells us how to merge the input and extra columns.
    output_schema: Arc<Schema>,

    batch_readahead: usize,

    properties: PlanProperties,
}

impl DisplayAs for TakeExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let extra_fields = self
            .schema_to_take
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
    /// - projection: the desired output projection, can overlap with the input schema if desired
    ///
    /// Returns None if no extra columns are required (everything in the projection exists in the input schema).
    pub fn try_new(
        dataset: Arc<Dataset>,
        input: Arc<dyn ExecutionPlan>,
        projection: Projection,
        batch_readahead: usize,
    ) -> Result<Option<Self>> {
        let original_projection = projection.clone().into_schema_ref();
        let projection =
            projection.subtract_arrow_schema(input.schema().as_ref(), OnMissing::Ignore)?;
        if projection.is_empty() {
            return Ok(None);
        }

        // We actually need a take so lets make sure we have a row id
        if input.schema().column_with_name(ROW_ID).is_none() {
            return Err(DataFusionError::Plan(
                "TakeExec requires the input plan to have a column named '_rowid'".to_string(),
            ));
        }

        // Can't use take if we don't want any fields and we can't use take to add row_id or row_addr
        assert!(
            !projection.with_row_id && !projection.with_row_addr,
            "Take cannot insert row_id / row_addr: {:#?}",
            projection
        );

        let output_schema = Arc::new(Self::calculate_output_schema(
            dataset.schema(),
            &input.schema(),
            &projection,
        ));
        let output_arrow = Arc::new(ArrowSchema::from(output_schema.as_ref()));
        let properties = input
            .properties()
            .clone()
            .with_eq_properties(EquivalenceProperties::new(output_arrow));

        Ok(Some(Self {
            dataset,
            original_projection,
            schema_to_take: projection.into_schema_ref(),
            input,
            output_schema,
            batch_readahead,
            properties,
        }))
    }

    /// The output of a take operation will be all columns from the input schema followed
    /// by any new columns from the dataset.
    ///
    /// The output fields will always be added in dataset schema order
    ///
    /// Nested columns in the input schema may have new fields inserted into them.
    ///
    /// If this happens the order of the new nested fields will match the order defined in
    /// the dataset schema.
    fn calculate_output_schema(
        dataset_schema: &Schema,
        input_schema: &ArrowSchema,
        projection: &Projection,
    ) -> Schema {
        // TakeExec doesn't reorder top-level fields and so the first thing we need to do is determine the
        // top-level field order.
        let mut top_level_fields_added = HashSet::with_capacity(input_schema.fields.len());
        let projected_schema = projection.to_schema();

        let mut output_fields =
            Vec::with_capacity(input_schema.fields.len() + projected_schema.fields.len());
        // TakeExec always moves the _rowid to the start of the output schema
        output_fields.extend(input_schema.fields.iter().map(|f| {
            let f = Field::try_from(f.as_ref()).unwrap();
            if let Some(ds_field) = dataset_schema.field(&f.name) {
                top_level_fields_added.insert(ds_field.id);
                // Field is in the dataset, it might have new fields added to it
                if let Some(projected_field) = ds_field.apply_projection(projection) {
                    f.merge_with_reference(&projected_field, ds_field)
                } else {
                    // No new fields added, keep as-is
                    f
                }
            } else {
                // Field not in dataset, not possible to add extra fields, use as-is
                f
            }
        }));

        // Now we add to the end any brand new top-level fields.  These will be added
        // dataset schema order.
        output_fields.extend(
            projected_schema
                .fields
                .into_iter()
                .filter(|f| !top_level_fields_added.contains(&f.id)),
        );
        Schema {
            fields: output_fields,
            metadata: dataset_schema.metadata.clone(),
        }
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
        Arc::new(self.output_schema.as_ref().into())
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

        let projection = self
            .dataset
            .empty_projection()
            .union_schema(&self.original_projection);

        let plan = Self::try_new(
            self.dataset.clone(),
            children[0].clone(),
            projection,
            self.batch_readahead,
        )?;

        if let Some(plan) = plan {
            Ok(Arc::new(plan))
        } else {
            // Is this legal or do we need to insert a no-op node?
            Ok(children[0].clone())
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        let output_schema_arrow = Arc::new(ArrowSchema::from(self.output_schema.as_ref()));
        Ok(Box::pin(Take::new(
            self.dataset.clone(),
            self.schema_to_take.clone(),
            output_schema_arrow,
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

    use arrow_array::{
        ArrayRef, Float32Array, Int32Array, RecordBatchIterator, StringArray, StructArray,
    };
    use arrow_schema::{DataType, Field, Fields};
    use datafusion::execution::TaskContext;
    use lance_core::datatypes::OnMissing;
    use tempfile::{tempdir, TempDir};

    use crate::{
        dataset::WriteParams,
        io::exec::{LanceScanConfig, LanceScanExec},
    };

    struct TestFixture {
        dataset: Arc<Dataset>,
        _tmp_dir_guard: TempDir,
    }

    async fn test_fixture() -> TestFixture {
        let struct_fields = Fields::from(vec![
            Arc::new(Field::new("x", DataType::Int32, false)),
            Arc::new(Field::new("y", DataType::Int32, false)),
        ]);

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("f", DataType::Float32, false),
            Field::new("s", DataType::Utf8, false),
            Field::new("struct", DataType::Struct(struct_fields.clone()), false),
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
                    Arc::new(StructArray::new(
                        struct_fields.clone(),
                        vec![
                            Arc::new(Int32Array::from_iter(value_range.clone())),
                            Arc::new(Int32Array::from_iter(value_range)),
                        ],
                        None,
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

        TestFixture {
            dataset: Arc::new(Dataset::open(test_uri).await.unwrap()),
            _tmp_dir_guard: test_dir,
        }
    }

    #[tokio::test]
    async fn test_take_schema() {
        let TestFixture { dataset, .. } = test_fixture().await;

        let scan_arrow_schema = ArrowSchema::new(vec![Field::new("i", DataType::Int32, false)]);
        let scan_schema = Arc::new(Schema::try_from(&scan_arrow_schema).unwrap());

        // With row id
        let config = LanceScanConfig {
            with_row_id: true,
            ..Default::default()
        };
        let input = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            scan_schema,
            config,
        ));

        let projection = dataset
            .empty_projection()
            .union_column("s", OnMissing::Error)
            .unwrap();
        let take_exec = TakeExec::try_new(dataset, input, projection, 10)
            .unwrap()
            .unwrap();
        let schema = take_exec.schema();
        assert_eq!(
            schema.fields.iter().map(|f| f.name()).collect::<Vec<_>>(),
            vec!["i", ROW_ID, "s"]
        );
    }

    #[tokio::test]
    async fn test_take_struct() {
        // When taking fields into an existing struct, the field order should be maintained
        // according the the schema of the struct.
        let TestFixture {
            dataset,
            _tmp_dir_guard,
        } = test_fixture().await;

        let scan_schema = Arc::new(dataset.schema().project(&["struct.y"]).unwrap());

        let config = LanceScanConfig {
            with_row_id: true,
            ..Default::default()
        };
        let input = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            scan_schema,
            config,
        ));

        let projection = dataset
            .empty_projection()
            .union_column("struct.x", OnMissing::Error)
            .unwrap();

        let take_exec = TakeExec::try_new(dataset, input, projection, 10)
            .unwrap()
            .unwrap();

        let expected_schema = ArrowSchema::new(vec![
            Field::new(
                "struct",
                DataType::Struct(Fields::from(vec![
                    Arc::new(Field::new("x", DataType::Int32, false)),
                    Arc::new(Field::new("y", DataType::Int32, false)),
                ])),
                false,
            ),
            Field::new(ROW_ID, DataType::UInt64, true),
        ]);
        let schema = take_exec.schema();
        assert_eq!(schema.as_ref(), &expected_schema);

        let mut stream = take_exec
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();

        while let Some(batch) = stream.try_next().await.unwrap() {
            assert_eq!(batch.schema().as_ref(), &expected_schema);
        }
    }

    #[tokio::test]
    async fn test_take_no_row_id() {
        let TestFixture { dataset, .. } = test_fixture().await;

        let scan_arrow_schema = ArrowSchema::new(vec![Field::new("i", DataType::Int32, false)]);
        let scan_schema = Arc::new(Schema::try_from(&scan_arrow_schema).unwrap());

        let projection = dataset
            .empty_projection()
            .union_column("s", OnMissing::Error)
            .unwrap();

        // No row ID
        let input = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            scan_schema,
            LanceScanConfig::default(),
        ));
        assert!(TakeExec::try_new(dataset, input, projection, 10).is_err());
    }

    #[tokio::test]
    async fn test_with_new_children() -> Result<()> {
        let TestFixture { dataset, .. } = test_fixture().await;

        let config = LanceScanConfig {
            with_row_id: true,
            ..Default::default()
        };

        let input_schema = Arc::new(dataset.schema().project(&["i"])?);
        let projection = dataset
            .empty_projection()
            .union_column("s", OnMissing::Error)
            .unwrap();

        let input = Arc::new(LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            input_schema,
            config,
        ));

        assert_eq!(input.schema().field_names(), vec!["i", ROW_ID],);
        let take_exec = TakeExec::try_new(dataset.clone(), input.clone(), projection, 10)?.unwrap();
        assert_eq!(take_exec.schema().field_names(), vec!["i", ROW_ID, "s"],);

        let projection = dataset
            .empty_projection()
            .union_columns(["s", "f"], OnMissing::Error)
            .unwrap();

        let outer_take =
            Arc::new(TakeExec::try_new(dataset, Arc::new(take_exec), projection, 10)?.unwrap());
        assert_eq!(
            outer_take.schema().field_names(),
            vec!["i", ROW_ID, "s", "f"],
        );

        // with_new_children should preserve the output schema.
        let edited = outer_take.with_new_children(vec![input])?;
        assert_eq!(edited.schema().field_names(), vec!["i", ROW_ID, "f", "s"],);
        Ok(())
    }
}
