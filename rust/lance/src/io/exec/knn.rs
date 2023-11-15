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

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::cast::AsArray;
use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning,
    RecordBatchStream as DFRecordBatchStream, SendableRecordBatchStream, Statistics,
};
use futures::stream::Stream;
use futures::{FutureExt, StreamExt, TryStreamExt};
use lance_core::utils::mask::RowIdMask;
use lance_core::{ROW_ID, ROW_ID_FIELD};
use lance_index::vector::{flat::flat_search, Query, DIST_COL};
use roaring::RoaringTreemap;
use snafu::{location, Location};
use tokio::sync::mpsc::Receiver;
use tokio::task::JoinHandle;
use tracing::{instrument, Instrument};

use crate::dataset::scanner::DatasetRecordBatchStream;
use crate::dataset::Dataset;
use crate::format::Index;
use crate::index::prefilter::{FilterLoader, PreFilter};
use crate::index::DatasetIndexInternalExt;
use crate::io::RecordBatchStream;
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

/// [ExecutionPlan] for Flat KNN (bruteforce) search.
///
/// Preconditions:
/// - `input` schema must contains `query.column`,
/// - The column must be a vector.
/// - `input` schema does not have "_distance" column.
#[derive(Debug)]
pub struct KNNFlatExec {
    /// Input node.
    input: Arc<dyn ExecutionPlan>,

    /// The query to execute.
    query: Query,
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
        let field = schema
            .field_with_name(&query.column)
            .map_err(|_| Error::IO {
                message: format!(
                    "KNNFlatExec node: query column {} not found in input schema",
                    query.column
                ),
                location: location!(),
            })?;
        match field.data_type() {
            DataType::FixedSizeList(list_field, _)
                if matches!(list_field.data_type(), DataType::Float32) => {}
            _ => {
                return Err(Error::IO {
                    message: format!(
                        "KNNFlatExec node: query column {} is not a vector. Expect FixedSizeList<Float32>, got {}",
                        query.column, field.data_type()
                    ),
                    location: location!(),
                });
            }
        }

        Ok(Self { input, query })
    }
}

impl ExecutionPlan for KNNFlatExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Flat KNN inherits the schema from input node, and add one distance column.
    fn schema(&self) -> arrow_schema::SchemaRef {
        let input_schema = self.input.schema();
        let mut fields = input_schema.fields().to_vec();
        if input_schema.field_with_name(DIST_COL).is_err() {
            fields.push(Arc::new(Field::new(DIST_COL, DataType::Float32, true)));
        }

        Arc::new(Schema::new_with_metadata(
            fields,
            input_schema.metadata().clone(),
        ))
    }

    fn output_partitioning(&self) -> Partitioning {
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

    fn statistics(&self) -> Statistics {
        Statistics {
            num_rows: Some(self.query.k),
            ..Default::default()
        }
    }
}

// Utility to convert an input (containing row ids) into a prefilter
struct FilteredRowIdsToPrefilter(SendableRecordBatchStream);

#[async_trait]
impl FilterLoader for FilteredRowIdsToPrefilter {
    async fn load(mut self: Box<Self>) -> Result<RowIdMask> {
        let mut allow_list = RoaringTreemap::new();
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

/// KNN Node from reading a vector index.
pub struct KNNIndexStream {
    rx: Receiver<datafusion::error::Result<RecordBatch>>,
    bg_thread: Option<JoinHandle<()>>,
}

impl KNNIndexStream {
    async fn knn_stream(
        query: Query,
        dataset: Arc<Dataset>,
        index_meta: Index,
        allow_list_input: Option<Box<dyn FilterLoader>>,
    ) -> Result<RecordBatch> {
        let index = dataset
            .open_vector_index(&query.column, &index_meta.uuid.to_string())
            .await?;
        let pre_filter = Arc::new(PreFilter::new(dataset, index_meta, allow_list_input));
        index.search(&query, pre_filter).await
    }

    #[instrument(level = "debug", skip_all, name = "KNNIndexStream::new")]
    pub fn new(
        dataset: Arc<Dataset>,
        index: &Index,
        query: &Query,
        allow_list: Option<Box<dyn FilterLoader>>,
    ) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(2);
        let q = query.clone();
        let index = index.clone();
        let bg_thread = tokio::spawn(
            async move {
                let result = match Self::knn_stream(q, dataset, index, allow_list).await {
                    Ok(b) => b,
                    Err(e) => {
                        tx.send(Err(datafusion::error::DataFusionError::Execution(format!(
                            "Failed to calculate KNN: {e}"
                        ))))
                        .await
                        .expect("KNNIndex failed to send message");
                        return;
                    }
                };

                if !tx.is_closed() {
                    if let Err(e) = tx.send(Ok(result)).await {
                        eprintln!("KNNIndex tx.send error: {e}")
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

impl DFRecordBatchStream for KNNIndexStream {
    fn schema(&self) -> arrow_schema::SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new(DIST_COL, DataType::Float32, true),
            ROW_ID_FIELD.clone(),
        ]))
    }
}

impl Stream for KNNIndexStream {
    type Item = std::result::Result<RecordBatch, datafusion::error::DataFusionError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = Pin::into_inner(self);
        // We need to check the JoinHandle to make sure the thread hasn't panicked.
        let bg_thread_completed = if let Some(bg_thread) = &mut this.bg_thread {
            match bg_thread.poll_unpin(cx) {
                Poll::Ready(Ok(())) => true,
                Poll::Ready(Err(join_error)) => {
                    return Poll::Ready(Some(Err(DataFusionError::Execution(format!(
                        "ExecNode(KNNIndexStream): thread panicked: {}",
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

#[derive(Debug)]
pub enum PreFilterSource {
    /// The prefilter input is an array of row ids that match the filter condition
    FilteredRowIds(Arc<dyn ExecutionPlan>),
    /// The prefilter input is a selection vector from an index query
    ScalarIndexQuery(Arc<dyn ExecutionPlan>),
    /// There is no prefilter
    None,
}

/// [ExecutionPlan] for KNNIndex node.
#[derive(Debug)]
pub struct KNNIndexExec {
    /// Dataset to read from.
    dataset: Arc<Dataset>,
    /// Prefiltering input
    prefilter_source: PreFilterSource,
    /// The index metadata
    index: Index,
    /// The vector query to execute.
    query: Query,
}

impl DisplayAs for KNNIndexExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "KNNIndex: name={}, k={}", self.index.uuid, self.query.k)
            }
        }
    }
}

impl KNNIndexExec {
    /// Create a new [KNNIndexExec].
    pub fn try_new(
        dataset: Arc<Dataset>,
        index: Index,
        query: &Query,
        prefilter_source: PreFilterSource,
    ) -> Result<Self> {
        let schema = dataset.schema();
        if schema.field(query.column.as_str()).is_none() {
            return Err(Error::IO {
                message: format!(
                    "KNNIndexExec node: query column {} does not exist in dataset.",
                    query.column
                ),
                location: location!(),
            });
        };

        Ok(Self {
            dataset,
            index,
            query: query.clone(),
            prefilter_source,
        })
    }
}

impl ExecutionPlan for KNNIndexExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new(DIST_COL, DataType::Float32, true),
            ROW_ID_FIELD.clone(),
        ]))
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::RoundRobinBatch(1)
    }

    fn output_ordering(&self) -> Option<&[datafusion::physical_expr::PhysicalSortExpr]> {
        None
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

        Ok(Box::pin(KNNIndexStream::new(
            self.dataset.clone(),
            &self.index,
            &self.query,
            prefilter_loader,
        )))
    }

    fn statistics(&self) -> datafusion::physical_plan::Statistics {
        Statistics {
            num_rows: Some(self.query.k * self.query.refine_factor.unwrap_or(1) as usize),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::RecordBatchIterator;
    use arrow_array::{cast::as_primitive_array, FixedSizeListArray, Int32Array, StringArray};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use lance_linalg::distance::MetricType;
    use lance_testing::datagen::generate_random_array;
    use tempfile::tempdir;

    use crate::arrow::*;
    use crate::dataset::{Dataset, WriteParams};
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
