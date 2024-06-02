// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::datatypes::UInt32Type;
use arrow_array::cast::AsArray;
use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::common::stats::Precision;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::physical_plan::{
    stream::RecordBatchStreamAdapter, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning,
    RecordBatchStream as DFRecordBatchStream, SendableRecordBatchStream, Statistics,
};
use datafusion::physical_plan::{ExecutionMode, PlanProperties};
use datafusion_physical_expr::EquivalenceProperties;
use futures::stream::repeat_with;
use futures::{stream, FutureExt, Stream, StreamExt, TryFutureExt, TryStreamExt};
use lance_core::utils::mask::{RowIdMask, RowIdTreeMap};
use lance_core::{ROW_ID, ROW_ID_FIELD};
use lance_index::vector::{flat::flat_search, Query, DIST_COL, PART_ID_COLUMN};
use lance_io::stream::RecordBatchStream;
use lance_table::format::Index;
use log::warn;
use snafu::{location, Location};
use tokio::sync::mpsc::Receiver;
use tokio::task::JoinHandle;
use tracing::{instrument, Instrument};

use crate::dataset::scanner::DatasetRecordBatchStream;
use crate::dataset::Dataset;
use crate::index::prefilter::{FilterLoader, PreFilter};
use crate::index::vector::ivf::IVFIndex;
use crate::index::DatasetIndexInternalExt;
use crate::{Error, Result};

/// KNN node for post-filtering.
pub struct KNNFlatStream {
    rx: Receiver<DataFusionResult<RecordBatch>>,
    bg_thread: Option<JoinHandle<()>>,
}

impl KNNFlatStream {
    /// Construct a [`KNNFlatStream`] node.
    #[instrument(level = "debug", skip_all, name = "KNNFlatStream::new")]
    pub(crate) fn new(child: SendableRecordBatchStream, query: &Query) -> Self {
        let stream = DatasetRecordBatchStream::new(child);
        Self::from_stream(stream, query)
    }

    fn from_stream(stream: impl RecordBatchStream + 'static, query: &Query) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(2);

        let q = query.clone();
        let bg_thread = tokio::spawn(
            async move {
                let batch = match flat_search(stream, &q).await {
                    Ok(b) => b,
                    Err(e) => {
                        tx.send(Err(DataFusionError::Execution(format!(
                            "Failed to compute distances: {e}"
                        ))))
                        .await
                        .expect("KNNFlat failed to send message");
                        return;
                    }
                };

                if !tx.is_closed() {
                    if let Err(e) = tx.send(Ok(batch)).await {
                        eprintln!("KNNFlat tx.send error: {e}")
                    };
                }
                drop(tx);
            }
            .in_current_span(),
        );

        Self {
            rx,
            bg_thread: Some(bg_thread),
        }
    }
}

impl Stream for KNNFlatStream {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = Pin::into_inner(self);
        // We need to check the JoinHandle to make sure the thread hasn't panicked.
        let bg_thread_completed = if let Some(bg_thread) = &mut this.bg_thread {
            match bg_thread.poll_unpin(cx) {
                Poll::Ready(Ok(())) => true,
                Poll::Ready(Err(join_error)) => {
                    return Poll::Ready(Some(Err(DataFusionError::Execution(format!(
                        "ExecNode(KNNFlatStream): thread panicked: {}",
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

impl DFRecordBatchStream for KNNFlatStream {
    fn schema(&self) -> arrow_schema::SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new(DIST_COL, DataType::Float32, true),
            ROW_ID_FIELD.clone(),
        ]))
    }
}

/// Check vector column exists and has the correct data type.
fn check_vector_column(schema: &Schema, column: &str) -> Result<()> {
    let field = schema.field_with_name(column).map_err(|_| {
        Error::io(
            format!("Query column '{}' not found in input schema", column),
            location!(),
        )
    })?;
    match field.data_type() {
        DataType::FixedSizeList(list_field, _)
            if matches!(
                list_field.data_type(),
                DataType::UInt8 | DataType::Float16 | DataType::Float32 | DataType::Float64
            ) => Ok(()),
        _ => {
           Err(Error::io(
                format!(
                    "KNNFlatExec node: query column {} is not a vector. Expect FixedSizeList<Float32>, got {}",
                    column, field.data_type()
                ),
                location!(),
            ))
        }
    }
}

/// [ExecutionPlan] for Flat KNN (bruteforce) search.
///
/// Preconditions:
/// - `input` schema must contains `query.column`,
/// - The column must be a vector.
/// - `input` schema does not have "_distance" column.
#[derive(Debug)]
pub struct KNNFlatExec {
    /// Inner input node.
    pub input: Arc<dyn ExecutionPlan>,

    /// The vector query to execute.
    pub query: Query,
    output_schema: SchemaRef,
    properties: PlanProperties,
}

impl DisplayAs for KNNFlatExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "KNNFlat: k={} metric={}",
                    self.query.k, self.query.metric_type
                )
            }
        }
    }
}

impl KNNFlatExec {
    /// Create a new [KNNFlatExec] node.
    ///
    /// Returns an error if the preconditions are not met.
    pub fn try_new(input: Arc<dyn ExecutionPlan>, query: Query) -> Result<Self> {
        let schema = input.schema();
        check_vector_column(&schema, &query.column)?;

        let mut fields = schema.fields().to_vec();
        if schema.field_with_name(DIST_COL).is_err() {
            fields.push(Arc::new(Field::new(DIST_COL, DataType::Float32, true)));
        }

        let output_schema = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));

        // This node has the same partitioning & boundedness as the input node
        // but it destroys any ordering.
        let properties = input
            .properties()
            .clone()
            .with_eq_properties(EquivalenceProperties::new(output_schema.clone()));

        Ok(Self {
            input,
            query,
            output_schema,
            properties,
        })
    }
}

impl ExecutionPlan for KNNFlatExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Flat KNN inherits the schema from input node, and add one distance column.
    fn schema(&self) -> arrow_schema::SchemaRef {
        self.output_schema.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        Ok(Box::pin(KNNFlatStream::new(
            self.input.execute(partition, context)?,
            &self.query,
        )))
    }

    fn statistics(&self) -> DataFusionResult<Statistics> {
        Ok(Statistics {
            num_rows: Precision::Exact(self.query.k),
            ..Statistics::new_unknown(self.schema().as_ref())
        })
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }
}

// Utility to convert an input (containing row ids) into a prefilter
struct FilteredRowIdsToPrefilter(SendableRecordBatchStream);

#[async_trait]
impl FilterLoader for FilteredRowIdsToPrefilter {
    async fn load(mut self: Box<Self>) -> Result<RowIdMask> {
        let mut allow_list = RowIdTreeMap::new();
        while let Some(batch) = self.0.next().await {
            let batch = batch?;
            let row_ids = batch.column_by_name(ROW_ID).expect(
                "input batch missing row id column even though it is in the schema for the stream",
            );
            let row_ids = row_ids
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("row id column in input batch had incorrect type");
            allow_list.extend(row_ids.iter().flatten())
        }
        Ok(RowIdMask::from_allowed(allow_list))
    }
}

// Utility to convert a serialized selection vector into a prefilter
struct SelectionVectorToPrefilter(SendableRecordBatchStream);

#[async_trait]
impl FilterLoader for SelectionVectorToPrefilter {
    async fn load(mut self: Box<Self>) -> Result<RowIdMask> {
        let batch = self
            .0
            .try_next()
            .await?
            .ok_or_else(|| Error::Internal {
                message: "Selection vector source for prefilter did not yield any batches".into(),
                location: location!(),
            })
            .unwrap();
        RowIdMask::from_arrow(batch["result"].as_binary_opt::<i32>().ok_or_else(|| {
            Error::Internal {
                message: format!(
                    "Expected selection vector input to yield binary arrays but got {}",
                    batch["result"].data_type()
                ),
                location: location!(),
            }
        })?)
    }
}

fn knn_index_stream(
    query: Query,
    dataset: Arc<Dataset>,
    index_meta: Vec<Index>,
    allow_list_input: Option<Box<dyn FilterLoader>>,
) -> impl DFRecordBatchStream {
    let pre_filter = Arc::new(PreFilter::new(
        dataset.clone(),
        &index_meta,
        allow_list_input,
    ));

    let schema = Arc::new(Schema::new(vec![
        Field::new(DIST_COL, DataType::Float32, true),
        ROW_ID_FIELD.clone(),
    ]));

    let s = stream::iter(index_meta)
        .zip(stream::repeat((
            dataset.clone(),
            pre_filter.clone(),
            query.clone(),
        )))
        .map(|(idx, (ds, pre_filter, query))| async move {
            let index = ds
                .open_vector_index(&query.column, &idx.uuid.to_string())
                .await?;
            index.search(&query, pre_filter.clone()).await
        })
        .buffer_unordered(num_cpus::get())
        .map(|r| {
            r.map_err(|e| DataFusionError::Execution(format!("Failed to calculate KNN: {}", e)))
        })
        .boxed();
    RecordBatchStreamAdapter::new(schema, s)
}

#[derive(Debug)]
pub enum PreFilterSource {
    /// The prefilter input is an array of row ids that match the filter condition
    FilteredRowIds(Arc<dyn ExecutionPlan>),
    /// The prefilter input is a selection vector from an index query
    ScalarIndexQuery(Arc<dyn ExecutionPlan>),
    /// There is no prefilter
    None,
}

lazy_static::lazy_static! {
    pub static ref KNN_INDEX_SCHEMA: SchemaRef = Arc::new(Schema::new(vec![
        Field::new(DIST_COL, DataType::Float32, true),
        ROW_ID_FIELD.clone(),
    ]));

    static ref KNN_PARTITION_SCHEMA: SchemaRef = Arc::new(Schema::new(vec![
        Field::new(PART_ID_COLUMN, DataType::UInt32, false),
    ]));
}

/// [ExecutionPlan] for KNNIndex node.
#[derive(Debug)]
pub struct KNNIndexExec {
    /// Dataset to read from.
    dataset: Arc<Dataset>,
    /// Prefiltering input
    prefilter_source: PreFilterSource,
    /// The index metadata
    indices: Vec<Index>,
    /// The vector query to execute.
    query: Query,
    /// The datafusion plan properties
    properties: PlanProperties,
}

impl DisplayAs for KNNIndexExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "KNNIndex: name={}, k={}, deltas={}",
                    self.indices[0].name,
                    self.query.k * self.query.refine_factor.unwrap_or(1) as usize,
                    self.indices.len()
                )
            }
        }
    }
}

impl KNNIndexExec {
    /// Create a new [KNNIndexExec].
    pub fn try_new(
        dataset: Arc<Dataset>,
        indices: &[Index],
        query: &Query,
        prefilter_source: PreFilterSource,
    ) -> Result<Self> {
        if indices.is_empty() {
            return Err(Error::io(
                "KNNIndexExec node: no index found for query".to_string(),
                location!(),
            ));
        }
        let schema = dataset.schema();
        check_vector_column(&schema.into(), &query.column)?;

        let properties = PlanProperties::new(
            EquivalenceProperties::new(KNN_INDEX_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            ExecutionMode::Bounded,
        );

        Ok(Self {
            dataset,
            indices: indices.to_vec(),
            query: query.clone(),
            prefilter_source,
            properties,
        })
    }
}

impl ExecutionPlan for KNNIndexExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        KNN_INDEX_SCHEMA.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        match &self.prefilter_source {
            PreFilterSource::None => vec![],
            PreFilterSource::FilteredRowIds(src) => vec![src.clone()],
            PreFilterSource::ScalarIndexQuery(src) => vec![src.clone()],
        }
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> DataFusionResult<datafusion::physical_plan::SendableRecordBatchStream> {
        let prefilter_loader = match &self.prefilter_source {
            PreFilterSource::FilteredRowIds(src_node) => {
                let stream = src_node.execute(partition, context)?;
                Some(Box::new(FilteredRowIdsToPrefilter(stream)) as Box<dyn FilterLoader>)
            }
            PreFilterSource::ScalarIndexQuery(src_node) => {
                let stream = src_node.execute(partition, context)?;
                Some(Box::new(SelectionVectorToPrefilter(stream)) as Box<dyn FilterLoader>)
            }
            PreFilterSource::None => None,
        };

        Ok(Box::pin(knn_index_stream(
            self.query.clone(),
            self.dataset.clone(),
            self.indices.clone(),
            prefilter_loader,
        )))
    }

    fn statistics(&self) -> DataFusionResult<datafusion::physical_plan::Statistics> {
        Ok(Statistics {
            num_rows: Precision::Exact(
                self.query.k * self.query.refine_factor.unwrap_or(1) as usize,
            ),
            ..Statistics::new_unknown(self.schema().as_ref())
        })
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }
}

/// Physical Plan to execute the IVF partition search.
///
/// It searches the partition IDs using the input query.
#[derive(Debug)]
pub struct ANNIvfPartitionExec {
    dataset: Arc<Dataset>,

    query: Query,

    properties: PlanProperties,
}

impl ANNIvfPartitionExec {
    pub fn try_new(dataset: Arc<Dataset>, query: Query) -> Result<Self> {
        let dataset_schema = dataset.schema();
        check_vector_column(&dataset_schema.into(), &query.column)?;

        let schema = KNN_PARTITION_SCHEMA.clone();
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::RoundRobinBatch(1),
            ExecutionMode::Bounded,
        );

        Ok(Self {
            dataset,
            query,
            properties,
        })
    }
}

impl DisplayAs for ANNIvfPartitionExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "ANNIVFPartitionExec: nprobes={}", self.query.nprobes)
            }
        }
    }
}

impl ExecutionPlan for ANNIvfPartitionExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        KNN_PARTITION_SCHEMA.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        warn!("ANNIVFPartitionExec: with_new_children called, but no children to replace");
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        todo!()
    }
}

/// Physical Plan to execute on a IVF sub-partition.
///
/// A IVF-{PQ/SQ/HNSW} query plan is:
///
/// AnnSubIndexExec -> AnnPartitionExec
#[derive(Debug)]
pub struct ANNSubIndexExec {
    /// Inner input source node.
    input: Arc<dyn ExecutionPlan>,

    dataset: Arc<Dataset>,

    /// Index metadata.
    index_meta: Index,

    /// Vector Query.
    query: Query,

    /// Prefiltering input
    prefilter_source: PreFilterSource,

    /// Datafusion Plan Properties
    properties: PlanProperties,
}

impl ANNSubIndexExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        index_meta: Index,
        query: Query,
        prefilter_source: PreFilterSource,
    ) -> Result<Self> {
        if input.schema().field_with_name(PART_ID_COLUMN).is_err() {
            return Err(Error::Index {
                message: format!(
                    "ANNSubIndexExec node: input schema does not have \"{}\" column",
                    PART_ID_COLUMN
                ),
                location: location!(),
            });
        }
        let properties = input.properties().clone();
        Ok(Self {
            input,
            dataset,
            index_meta,
            query,
            prefilter_source,
            properties,
        })
    }
}

impl DisplayAs for ANNSubIndexExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "ANNSubIndex: name={}, k={}",
                    self.index_meta.name,
                    self.query.k * self.query.refine_factor.unwrap_or(1) as usize,
                )
            }
        }
    }
}

impl ExecutionPlan for ANNSubIndexExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        KNN_INDEX_SCHEMA.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        match &self.prefilter_source {
            PreFilterSource::None => vec![],
            PreFilterSource::FilteredRowIds(src) => vec![src.clone()],
            PreFilterSource::ScalarIndexQuery(src) => vec![src.clone()],
        }
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> DataFusionResult<datafusion::physical_plan::SendableRecordBatchStream> {
        let prefilter_loader = match &self.prefilter_source {
            PreFilterSource::FilteredRowIds(src_node) => {
                let stream = src_node.execute(partition, context.clone())?;
                Some(Box::new(FilteredRowIdsToPrefilter(stream)) as Box<dyn FilterLoader>)
            }
            PreFilterSource::ScalarIndexQuery(src_node) => {
                let stream = src_node.execute(partition, context.clone())?;
                Some(Box::new(SelectionVectorToPrefilter(stream)) as Box<dyn FilterLoader>)
            }
            PreFilterSource::None => None,
        };

        let pre_filter = Arc::new(PreFilter::new(
            self.dataset.clone(),
            &[self.index_meta.clone()],
            prefilter_loader,
        ));

        let input_stream = self.input.execute(partition, context)?;
        let schema = self.schema();
        let uuid = self.index_meta.uuid.to_string();
        let query = self.query.clone();
        let ds = self.dataset.clone();
        let column = self.query.column.clone();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            input_stream
                .and_then(move |batch| {
                    let ds = ds.clone();
                    let uuid = uuid.clone();
                    let column = column.clone();

                    async move {
                        let part_id_arr = batch.column_by_name(PART_ID_COLUMN).ok_or(
                            DataFusionError::Execution(
                                "ANNSubIndexExec: input missing part_id column".to_string(),
                            ),
                        )?;
                        let part_id_arr = part_id_arr.as_primitive_opt::<UInt32Type>().ok_or(
                            DataFusionError::Execution(
                                "ANNSubIndexExec: part_id column has incorrect type".to_string(),
                            ),
                        )?;

                        let raw_index = ds.open_vector_index(&column, &uuid).await?;

                        Ok(stream::iter(part_id_arr.values().to_vec().into_iter())
                            .zip(repeat_with(move || raw_index.clone()))
                            .map(Ok::<_, DataFusionError>))
                    }
                })
                .try_flatten_unordered(None)
                .map(move |result| {
                    let pre_filter = pre_filter.clone();
                    let query = query.clone();

                    async move {
                        let (part_id, raw_index) = result?;

                        let index = raw_index.as_any().downcast_ref::<IVFIndex>().ok_or(
                            DataFusionError::Execution(
                                "ANNSubIndexExec: sub-index is not a IVF type".to_string(),
                            ),
                        )?;
                        index
                            .search_in_partition(part_id as usize, &query, pre_filter)
                            .map_err(|e| {
                                DataFusionError::Execution(format!(
                                    "Failed to calculate KNN: {}",
                                    e
                                ))
                            })
                            .await
                    }
                })
                .buffered(num_cpus::get())
                .boxed(),
        )))
    }

    fn statistics(&self) -> DataFusionResult<datafusion::physical_plan::Statistics> {
        Ok(Statistics {
            num_rows: Precision::Exact(
                self.query.k
                    * self.query.refine_factor.unwrap_or(1) as usize
                    * self.input.statistics()?.num_rows.get_value().unwrap_or(&1),
            ),
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

    use arrow_array::RecordBatchIterator;
    use arrow_array::{cast::as_primitive_array, FixedSizeListArray, Int32Array, StringArray};
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
    use lance_linalg::distance::MetricType;
    use lance_testing::datagen::generate_random_array;
    use tempfile::tempdir;

    use crate::arrow::*;
    use crate::dataset::WriteParams;
    use crate::io::exec::testing::TestingExec;

    #[tokio::test]
    async fn knn_flat_search() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("key", DataType::Int32, false),
            ArrowField::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    128,
                ),
                true,
            ),
            ArrowField::new("uri", DataType::Utf8, true),
        ]));

        let batches: Vec<RecordBatch> = (0..20)
            .map(|i| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)),
                        Arc::new(
                            FixedSizeListArray::try_new_from_values(
                                generate_random_array(128 * 20),
                                128,
                            )
                            .unwrap(),
                        ),
                        Arc::new(StringArray::from_iter_values(
                            (i * 20..(i + 1) * 20).map(|i| format!("s3://bucket/file-{}", i)),
                        )),
                    ],
                )
                .unwrap()
            })
            .collect();

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            ..Default::default()
        };
        let vector_arr = batches[0].column_by_name("vector").unwrap();
        let q = as_fixed_size_list_array(&vector_arr).value(5);

        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let stream = dataset
            .scan()
            .nearest("vector", as_primitive_array(&q), 10)
            .unwrap()
            .try_into_stream()
            .await
            .unwrap();
        let results = stream.try_collect::<Vec<_>>().await.unwrap();

        assert!(results[0].schema().column_with_name(DIST_COL).is_some());

        assert_eq!(results.len(), 1);

        let stream = dataset.scan().try_into_stream().await.unwrap();
        let expected = flat_search(
            stream,
            &Query {
                column: "vector".to_string(),
                key: q,
                k: 10,
                nprobes: 0,
                ef: None,
                refine_factor: None,
                metric_type: MetricType::L2,
                use_index: false,
            },
        )
        .await
        .unwrap();

        assert_eq!(expected, results[0]);
    }

    #[test]
    fn test_create_knn_flat() {
        let dim: usize = 128;
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("key", DataType::Int32, false),
            ArrowField::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                true,
            ),
            ArrowField::new("uri", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::new_empty(schema);

        let query = Query {
            column: "vector".to_string(),
            key: Arc::new(generate_random_array(dim)),
            k: 10,
            nprobes: 0,
            ef: None,
            refine_factor: None,
            metric_type: MetricType::L2,
            use_index: false,
        };

        let input: Arc<dyn ExecutionPlan> = Arc::new(TestingExec::new(vec![batch]));
        let idx = KNNFlatExec::try_new(input, query).unwrap();
        println!("{:?}", idx);
        assert_eq!(
            idx.schema().as_ref(),
            &ArrowSchema::new(vec![
                ArrowField::new("key", DataType::Int32, false),
                ArrowField::new(
                    "vector",
                    DataType::FixedSizeList(
                        Arc::new(ArrowField::new("item", DataType::Float32, true)),
                        dim as i32,
                    ),
                    true,
                ),
                ArrowField::new("uri", DataType::Utf8, true),
                ArrowField::new(DIST_COL, DataType::Float32, true),
            ])
        );
    }
}
