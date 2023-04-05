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

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::physical_plan::{
    ExecutionPlan, Partitioning, RecordBatchStream as DFRecordBatchStream,
    SendableRecordBatchStream, Statistics,
};
use futures::stream::Stream;
use tokio::sync::mpsc::Receiver;
use tokio::task::JoinHandle;

use crate::dataset::scanner::RecordBatchStream;
use crate::dataset::{Dataset, ROW_ID};
use crate::index::vector::flat::flat_search;
use crate::index::vector::{open_index, Query, SCORE_COL};
use crate::{Error, Result};

/// KNN node for post-filtering.
pub struct KNNFlatStream {
    rx: Receiver<DataFusionResult<RecordBatch>>,

    _bg_thread: JoinHandle<()>,
}

impl KNNFlatStream {
    /// Construct a [KNNFlat] node.
    pub(crate) fn new(child: SendableRecordBatchStream, query: &Query) -> Self {
        let stream = RecordBatchStream::new(child);
        KNNFlatStream::from_stream(stream, query)
    }

    pub(crate) fn from_vec(children: Vec<SendableRecordBatchStream>, query: &Query) -> Self {
        let stream = RecordBatchStream::from_vec(children);
        KNNFlatStream::from_stream(stream, query)
    }

    fn from_stream(stream: RecordBatchStream, query: &Query) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(2);

        let q = query.clone();
        let bg_thread = tokio::spawn(async move {
            let batch = match flat_search(stream, &q).await {
                Ok(b) => b,
                Err(e) => {
                    tx.send(Err(DataFusionError::Execution(format!(
                        "Failed to compute scores: {e}"
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
        });

        Self {
            rx,
            _bg_thread: bg_thread,
        }
    }
}

impl Stream for KNNFlatStream {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::into_inner(self).rx.poll_recv(cx)
    }
}

impl DFRecordBatchStream for KNNFlatStream {
    fn schema(&self) -> arrow_schema::SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("score", DataType::Float32, false),
            Field::new(ROW_ID, DataType::UInt16, false),
        ]))
    }
}

/// [ExecutionPlan] for Flat KNN (bruteforce) search.
///
/// Preconditions:
/// - `input` schema must contains `query.column`,
/// - The column must be a vector.
/// - `input` schema does not have "score" column.
pub struct KNNFlatExec {
    /// Input node.
    inputs: Vec<Arc<dyn ExecutionPlan>>,

    /// The query to execute.
    query: Query,
}

impl std::fmt::Debug for KNNFlatExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KNN(flat, k={}, metric={})",
            self.query.k, self.query.metric_type
        )
    }
}

impl KNNFlatExec {
    /// Create a new [KNNFlatExec] node.
    ///
    /// Returns an error if the preconditions are not met.
    pub fn try_new(inputs: Vec<Arc<dyn ExecutionPlan>>, query: Query) -> Result<Self> {
        let schema = inputs[0].schema(); // TODO validate schema
        let field = schema.field_with_name(&query.column).map_err(|_| {
            Error::IO(format!(
                "KNNFlatExec node: query column {} not found in input schema",
                query.column
            ))
        })?;
        let is_vector = match field.data_type() {
            DataType::FixedSizeList(item, _) => item.as_ref().data_type() == &DataType::Float32,
            _ => false,
        };
        if !is_vector {
            return Err(Error::IO(format!(
                "KNNFlatExec node: query column {} is not a vector",
                query.column
            )));
        };

        Ok(Self { inputs, query })
    }
}

impl ExecutionPlan for KNNFlatExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Flat KNN inherits the schema from input node, and add one score column.
    fn schema(&self) -> arrow_schema::SchemaRef {
        let input_schema = self.inputs[0].schema();
        let mut fields = input_schema.fields().to_vec();
        if !input_schema.field_with_name(SCORE_COL).is_ok() {
            fields.push(Field::new(SCORE_COL, DataType::Float32, false));
        }

        Arc::new(Schema::new_with_metadata(
            fields,
            input_schema.metadata().clone(),
        ))
    }

    fn output_partitioning(&self) -> Partitioning {
        self.inputs[0].output_partitioning()
    }

    fn output_ordering(&self) -> Option<&[datafusion::physical_expr::PhysicalSortExpr]> {
        self.inputs[0].output_ordering()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        self.inputs.clone()
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
        let streams: DataFusionResult<Vec<SendableRecordBatchStream>> = self
            .inputs
            .iter()
            .map(|node| node.execute(partition, context.clone()))
            .collect::<Vec<_>>()
            .into_iter()
            .collect();
        Ok(Box::pin(KNNFlatStream::from_vec(streams?, &self.query)))
    }

    fn statistics(&self) -> Statistics {
        Statistics {
            num_rows: Some(self.query.k as usize),
            ..Default::default()
        }
    }
}

/// KNN Node from reading a vector index.
pub struct KNNIndexStream {
    rx: Receiver<datafusion::error::Result<RecordBatch>>,

    _bg_thread: JoinHandle<()>,
}

impl KNNIndexStream {
    pub fn new(dataset: Arc<Dataset>, index_name: &str, query: &Query) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(2);

        let q = query.clone();
        let name = index_name.to_string();
        let bg_thread = tokio::spawn(async move {
            let index = match open_index(dataset.as_ref(), &name).await {
                Ok(idx) => idx,
                Err(e) => {
                    tx.send(Err(datafusion::error::DataFusionError::Execution(format!(
                        "Failed to open vector index: {name}: {e}"
                    ))))
                    .await
                    .expect("KNNFlat failed to send message");
                    return;
                }
            };
            let result = match index.search(&q).await {
                Ok(b) => b,
                Err(e) => {
                    tx.send(Err(datafusion::error::DataFusionError::Execution(format!(
                        "Failed to compute scores: {e}"
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
        });

        Self {
            rx,
            _bg_thread: bg_thread,
        }
    }
}

impl DFRecordBatchStream for KNNIndexStream {
    fn schema(&self) -> arrow_schema::SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new(SCORE_COL, DataType::Float32, false),
            Field::new(ROW_ID, DataType::UInt16, false),
        ]))
    }
}

impl Stream for KNNIndexStream {
    type Item = std::result::Result<RecordBatch, datafusion::error::DataFusionError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::into_inner(self).rx.poll_recv(cx)
    }
}

/// [ExecutionPlan] for KNNIndex node.
pub struct KNNIndexExec {
    /// Dataset to read from.
    dataset: Arc<Dataset>,
    /// The UUID of the index.
    index_name: String,
    /// The vector query to execute.
    query: Query,
}

impl std::fmt::Debug for KNNIndexExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KNN(index, name={}, k={})",
            self.index_name, self.query.k
        )
    }
}

impl KNNIndexExec {
    /// Create a new [KNNIndexExec].
    pub fn try_new(dataset: Arc<Dataset>, index_name: &str, query: &Query) -> Result<Self> {
        let schema = dataset.schema();
        if schema.field(query.column.as_str()).is_none() {
            return Err(Error::IO(format!(
                "KNNIndexExec node: query column {} does not exist in dataset.",
                query.column
            )));
        };

        Ok(Self {
            dataset,
            index_name: index_name.to_string(),
            query: query.clone(),
        })
    }
}

impl ExecutionPlan for KNNIndexExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new(SCORE_COL, DataType::Float32, false),
            Field::new(ROW_ID, DataType::UInt16, false),
        ]))
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::RoundRobinBatch(1)
    }

    fn output_ordering(&self) -> Option<&[datafusion::physical_expr::PhysicalSortExpr]> {
        None
    }

    /// KNNIndex is a leaf node, so returns zero children.
    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<datafusion::execution::context::TaskContext>,
    ) -> DataFusionResult<datafusion::physical_plan::SendableRecordBatchStream> {
        Ok(Box::pin(KNNIndexStream::new(
            self.dataset.clone(),
            &self.index_name,
            &self.query,
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

    use arrow_array::{
        cast::as_primitive_array, FixedSizeListArray, Int32Array, RecordBatchReader, StringArray,
    };
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use tempfile::tempdir;

    use crate::arrow::*;
    use crate::dataset::{Dataset, WriteParams};
    use crate::index::vector::MetricType;
    use crate::io::exec::testing::TestingExec;
    use crate::utils::testing::generate_random_array;

    #[tokio::test]
    async fn knn_flat_search() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("key", DataType::Int32, false),
            ArrowField::new(
                "vector",
                DataType::FixedSizeList(
                    Box::new(ArrowField::new("item", DataType::Float32, true)),
                    128,
                ),
                true,
            ),
            ArrowField::new("uri", DataType::Utf8, true),
        ]));

        let batches = RecordBatchBuffer::new(
            (0..20)
                .map(|i| {
                    RecordBatch::try_new(
                        schema.clone(),
                        vec![
                            Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)),
                            Arc::new(
                                FixedSizeListArray::try_new(generate_random_array(128 * 20), 128)
                                    .unwrap(),
                            ),
                            Arc::new(StringArray::from_iter_values(
                                (i * 20..(i + 1) * 20).map(|i| format!("s3://bucket/file-{}", i)),
                            )),
                        ],
                    )
                    .unwrap()
                })
                .collect(),
        );

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let mut write_params = WriteParams::default();
        write_params.max_rows_per_file = 40;
        write_params.max_rows_per_group = 10;
        let vector_arr = batches.batches[0].column_by_name("vector").unwrap();
        let q = as_fixed_size_list_array(&vector_arr).value(5);

        let mut reader: Box<dyn RecordBatchReader> = Box::new(batches);
        Dataset::write(&mut reader, test_uri, Some(write_params))
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

        assert!(results[0].schema().column_with_name("score").is_some());

        assert_eq!(results.len(), 1);

        let stream = dataset.scan().try_into_stream().await.unwrap();
        let expected = flat_search(
            stream,
            &Query {
                column: "vector".to_string(),
                key: Arc::new(as_primitive_array(&q).clone()),
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
                    Box::new(ArrowField::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                true,
            ),
            ArrowField::new("uri", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::new_empty(schema.clone());

        let query = Query {
            column: "vector".to_string(),
            key: Arc::new(generate_random_array(dim)),
            k: 10,
            nprobes: 0,
            refine_factor: None,
            metric_type: MetricType::L2,
            use_index: false,
        };

        let input: Arc<dyn ExecutionPlan> = Arc::new(TestingExec::new(vec![batch.into()]));
        let idx = KNNFlatExec::try_new(vec![input], query).unwrap();
        assert_eq!(
            idx.schema().as_ref(),
            &ArrowSchema::new(vec![
                ArrowField::new("key", DataType::Int32, false),
                ArrowField::new(
                    "vector",
                    DataType::FixedSizeList(
                        Box::new(ArrowField::new("item", DataType::Float32, true)),
                        dim as i32,
                    ),
                    true,
                ),
                ArrowField::new("uri", DataType::Utf8, true),
                ArrowField::new(SCORE_COL, DataType::Float32, false),
            ])
        );
    }
}
