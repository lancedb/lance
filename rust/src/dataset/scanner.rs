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

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::{Float32Array, RecordBatch};
use arrow_schema::DataType::Float32;
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema, SchemaRef};
use datafusion::execution::{
    context::SessionState,
    runtime_env::{RuntimeConfig, RuntimeEnv},
};
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::{
    limit::GlobalLimitExec, ExecutionPlan, PhysicalExpr, SendableRecordBatchStream,
};
use datafusion::prelude::*;
use futures::stream::{Stream, StreamExt};

use super::Dataset;
use crate::datafusion::physical_expr::column_names_in_expr;
use crate::datatypes::Schema;
use crate::format::Index;
use crate::index::vector::{MetricType, Query};
use crate::io::exec::{GlobalTakeExec, KNNFlatExec, KNNIndexExec, LanceScanExec, LocalTakeExec};
use crate::utils::sql::parse_sql_filter;
use crate::{Error, Result};

/// Column name for the meta row ID.
pub const ROW_ID: &str = "_rowid";
pub const DEFAULT_BATCH_SIZE: usize = 8192;

const PREFETCH_SIZE: usize = 8;

/// Dataset Scanner
///
/// ```rust,ignore
/// let dataset = Dataset::open(uri).await.unwrap();
/// let stream = dataset.scan()
///     .project(&["col", "col2.subfield"]).unwrap()
///     .limit(10)
///     .into_stream();
/// stream
///   .map(|batch| batch.num_rows())
///   .buffered(16)
///   .sum()
/// ```
pub struct Scanner {
    dataset: Arc<Dataset>,

    projections: Schema,

    /// Optional filters string.
    filter: Option<String>,

    /// The batch size controls the maximum size of rows to return for each read.
    batch_size: usize,

    limit: Option<i64>,
    offset: Option<i64>,

    nearest: Option<Query>,

    /// Scan the dataset with a meta column: "_rowid"
    with_row_id: bool,
}

impl Scanner {
    pub fn new(dataset: Arc<Dataset>) -> Self {
        let projection = dataset.schema().clone();
        Self {
            dataset,
            projections: projection,
            filter: None,
            batch_size: DEFAULT_BATCH_SIZE,
            limit: None,
            offset: None,
            nearest: None,
            with_row_id: false,
        }
    }

    /// Projection.
    ///
    /// Only seelect the specific columns. If not specifid, all columns will be scanned.
    pub fn project(&mut self, columns: &[&str]) -> Result<&mut Self> {
        self.projections = self.dataset.schema().project(columns)?;
        Ok(self)
    }

    /// Apply filters
    ///
    /// The filters can be presented as the string, as in WHERE clause in SQL.
    ///
    /// ```rust,ignore
    /// let dataset = Dataset::open(uri).await.unwrap();
    /// let stream = dataset.scan()
    ///     .project(&["col", "col2.subfield"]).unwrap()
    ///     .filter("a > 10 AND b < 200").unwrap()
    ///     .limit(10)
    ///     .into_stream();
    /// ```
    ///
    /// Once the filter is applied, Lance will create an optimized I/O plan for filtering.
    ///
    pub fn filter(&mut self, filter: &str) -> Result<&mut Self> {
        parse_sql_filter(filter)?;
        self.filter = Some(filter.to_string());
        Ok(self)
    }

    /// Set the batch size.
    pub fn batch_size(&mut self, batch_size: usize) -> &mut Self {
        self.batch_size = batch_size;
        self
    }

    /// Set limit and offset.
    pub fn limit(&mut self, limit: i64, offset: Option<i64>) -> Result<&mut Self> {
        if limit < 0 {
            return Err(Error::IO("Limit must be non-negative".to_string()));
        }
        if let Some(off) = offset {
            if off < 0 {
                return Err(Error::IO("Offset must be non-negative".to_string()));
            }
        }
        self.limit = Some(limit);
        self.offset = offset;
        Ok(self)
    }

    /// Find k-nearest neighbour within the vector column.
    pub fn nearest(&mut self, column: &str, q: &Float32Array, k: usize) -> Result<&mut Self> {
        if k == 0 {
            return Err(Error::IO("k must be positive".to_string()));
        }
        if q.is_empty() {
            return Err(Error::IO(
                "Query vector must have non-zero length".to_string(),
            ));
        }
        // make sure the field exists
        self.dataset.schema().project(&[column])?;
        self.nearest = Some(Query {
            column: column.to_string(),
            key: Arc::new(q.clone()),
            k,
            nprobs: 1,
            refine_factor: None,
            metric_type: MetricType::L2,
            use_index: true,
        });
        Ok(self)
    }

    pub fn nprobs(&mut self, n: usize) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.nprobs = n;
        }
        self
    }

    /// Apply a refine step to the vector search.
    ///
    /// A refine step uses the original vector values to re-rank the distances.
    pub fn refine(&mut self, factor: u32) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.refine_factor = Some(factor)
        };
        self
    }

    /// Change the distance [MetricType], i.e, L2 or Cosine distance.
    pub fn distance_metric(&mut self, metric_type: MetricType) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.metric_type = metric_type
        }
        self
    }

    /// Set whether to use the index if available
    pub fn use_index(&mut self, use_index: bool) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.use_index = use_index
        }
        self
    }

    /// Instruct the scanner to return the `_rowid` meta column from the dataset.
    pub fn with_row_id(&mut self) -> &mut Self {
        self.with_row_id = true;
        self
    }

    /// The Arrow schema of the output, including projections and vector / score
    pub fn schema(&self) -> Result<SchemaRef> {
        self.scanner_output_schema()
            .map(|s| SchemaRef::new(ArrowSchema::from(s.as_ref())))
    }

    fn scanner_output_schema(&self) -> Result<Arc<Schema>> {
        if self.nearest.as_ref().is_some() {
            let merged = self.projections.merge(&self.vector_search_schema()?);
            Ok(Arc::new(merged))
        } else {
            Ok(Arc::new(self.projections.clone()))
        }
    }

    fn vector_search_schema(&self) -> Result<Schema> {
        let q = self.nearest.as_ref().unwrap();
        let vector_schema = self.dataset.schema().project(&[&q.column])?;
        let score = ArrowField::new("score", Float32, false);
        let score_schema = Schema::try_from(&ArrowSchema::new(vec![score]))?;
        Ok(vector_schema.merge(&score_schema))
    }

    /// Create a stream of this Scanner.
    ///
    /// TODO: implement as IntoStream/IntoIterator.
    pub async fn try_into_stream(&self) -> Result<RecordBatchStream> {
        let with_row_id = self.with_row_id;
        let projection = &self.projections;

        let filter_expr = if let Some(filter) = self.filter.as_ref() {
            let planner = crate::io::exec::Planner::new(Arc::new(self.dataset.schema().into()));
            let logical_expr = planner.parse_filter(filter)?;
            Some(planner.create_physical_expr(&logical_expr)?)
        } else {
            None
        };

        let mut plan: Arc<dyn ExecutionPlan> = if let Some(q) = self.nearest.as_ref() {
            let column_id = self.dataset.schema().field_id(q.column.as_str())?;
            let use_index = self.nearest.as_ref().map(|q| q.use_index).unwrap_or(false);
            let indices = if use_index {
                self.dataset.load_indices().await?
            } else {
                vec![]
            };
            let qcol_index = indices.iter().find(|i| i.fields.contains(&column_id));
            if let Some(index) = qcol_index {
                // There is an index built for the column.
                // We will use the index.
                if let Some(rf) = q.refine_factor {
                    if rf == 0 {
                        return Err(Error::IO("Refine factor can not be zero".to_string()));
                    }
                }

                let knn_node = self.ann(q, &index); // score, _rowid
                let with_vector = self.dataset.schema().project(&[&q.column])?;
                let knn_node_with_vector = self.take(knn_node, &with_vector, false);
                let knn_node = if q.refine_factor.is_some() {
                    self.flat_knn(knn_node_with_vector, q)
                } else {
                    knn_node_with_vector
                }; // vector, score, _rowid

                let knn_node = filter_expr
                    .map(|f| self.filter_knn(knn_node.clone(), f))
                    .unwrap_or(Ok(knn_node))?; // vector, score, _rowid
                self.take(knn_node, projection, true)
            } else {
                let vector_scan_projection =
                    Arc::new(self.dataset.schema().project(&[&q.column]).unwrap());
                let scan_node = self.scan(true, vector_scan_projection);
                let knn_node = self.flat_knn(scan_node, q);

                let knn_node = filter_expr
                    .map(|f| self.filter_knn(knn_node.clone(), f))
                    .unwrap_or(Ok(knn_node))?; // vector, score, _rowid
                self.take(knn_node, projection, true)
            }
        } else if let Some(filter) = filter_expr {
            let columns_in_filter = column_names_in_expr(filter.as_ref());
            let filter_schema = Arc::new(
                self.dataset.schema().project(
                    &columns_in_filter
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>(),
                )?,
            );
            let scan = self.scan(true, filter_schema);
            self.filter_node(filter, scan, true, None)?
        } else {
            self.scan(with_row_id, Arc::new(self.projections.clone()))
        };

        if (self.limit.unwrap_or(0) > 0) || self.offset.is_some() {
            plan = self.limit_node(plan);
        }

        let session_config = SessionConfig::new();
        let runtime_config = RuntimeConfig::new();
        let runtime_env = Arc::new(RuntimeEnv::new(runtime_config)?);
        let session_state = SessionState::with_config_rt(session_config, runtime_env);
        Ok(RecordBatchStream::new(
            plan.execute(0, session_state.task_ctx())?,
        ))
    }

    fn filter_knn(
        &self,
        knn_node: Arc<dyn ExecutionPlan>,
        filter_expression: Arc<dyn PhysicalExpr>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let columns_in_filter = column_names_in_expr(filter_expression.as_ref());
        let columns_refs = columns_in_filter
            .iter()
            .map(|c| c.as_str())
            .collect::<Vec<_>>();
        let filter_projection = self.dataset.schema().project(&columns_refs)?;

        let take_node = Arc::new(GlobalTakeExec::new(
            self.dataset.clone(),
            Arc::new(filter_projection),
            knn_node,
            false,
        ));
        self.filter_node(
            filter_expression,
            take_node,
            false,
            Some(Arc::new(self.vector_search_schema()?)),
        )
    }

    /// Create an Execution plan with a scan node
    fn scan(&self, with_row_id: bool, projection: Arc<Schema>) -> Arc<dyn ExecutionPlan> {
        Arc::new(LanceScanExec::new(
            self.dataset.clone(),
            self.dataset.fragments().clone(),
            projection,
            self.batch_size,
            PREFETCH_SIZE,
            with_row_id,
        ))
    }

    /// Add a knn search node to the input plan
    fn flat_knn(&self, input: Arc<dyn ExecutionPlan>, q: &Query) -> Arc<dyn ExecutionPlan> {
        Arc::new(KNNFlatExec::new(input, q.clone()))
    }

    /// Create an Execution plan to do indexed ANN search
    fn ann(&self, q: &Query, index: &&Index) -> Arc<dyn ExecutionPlan> {
        let mut inner_query = q.clone();
        inner_query.k = q.k * q.refine_factor.unwrap_or(1) as usize;
        Arc::new(KNNIndexExec::new(
            self.dataset.clone(),
            &index.uuid.to_string(),
            &inner_query,
        ))
    }

    /// Take row indices produced by input plan from the dataset (with projection)
    fn take(
        &self,
        input: Arc<dyn ExecutionPlan>,
        projection: &Schema,
        drop_row_id: bool,
    ) -> Arc<dyn ExecutionPlan> {
        Arc::new(GlobalTakeExec::new(
            self.dataset.clone(),
            Arc::new(projection.clone()),
            input,
            drop_row_id,
        ))
    }

    /// Global offset-limit of the result of the input plan
    fn limit_node(&self, plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        Arc::new(GlobalLimitExec::new(
            plan,
            *self.offset.as_ref().unwrap_or(&0) as usize,
            self.limit.map(|l| l as usize),
        ))
    }

    fn filter_node(
        &self,
        filter: Arc<dyn PhysicalExpr>,
        input: Arc<dyn ExecutionPlan>,
        drop_row_id: bool,
        ann_schema: Option<Arc<Schema>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let filter_node = Arc::new(FilterExec::try_new(filter, input)?);
        let output_schema = self.scanner_output_schema()?;
        Ok(Arc::new(LocalTakeExec::new(
            filter_node,
            self.dataset.clone(),
            output_schema,
            ann_schema,
            drop_row_id,
        )))
    }
}

/// ScannerStream is a container to wrap different types of ExecNode.
#[pin_project::pin_project]
pub struct RecordBatchStream {
    #[pin]
    exec_node: SendableRecordBatchStream,
}

impl RecordBatchStream {
    pub fn new(exec_node: SendableRecordBatchStream) -> Self {
        Self { exec_node }
    }
}

impl Stream for RecordBatchStream {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        match this.exec_node.poll_next_unpin(cx) {
            Poll::Ready(result) => {
                Poll::Ready(result.map(|r| r.map_err(|e| Error::IO(e.to_string()))))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod test {

    use std::path::PathBuf;

    use super::*;

    use arrow::compute::concat_batches;
    use arrow_array::{ArrayRef, Int32Array, Int64Array, RecordBatchReader, StringArray};
    use arrow_schema::DataType;
    use futures::TryStreamExt;
    use tempfile::tempdir;

    use crate::{arrow::RecordBatchBuffer, dataset::WriteParams};

    #[tokio::test]
    async fn test_batch_size() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, true),
            ArrowField::new("s", DataType::Utf8, true),
        ]));

        let batches = RecordBatchBuffer::new(
            (0..5)
                .map(|i| {
                    RecordBatch::try_new(
                        schema.clone(),
                        vec![
                            Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)),
                            Arc::new(StringArray::from_iter_values(
                                (i * 20..(i + 1) * 20).map(|v| format!("s-{}", v)),
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
        let mut batches: Box<dyn RecordBatchReader> = Box::new(batches);
        Dataset::write(&mut batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let mut stream = dataset
            .scan()
            .batch_size(8)
            .try_into_stream()
            .await
            .unwrap();
        for expected_len in [8, 8, 4, 8, 8, 4] {
            assert_eq!(
                stream.next().await.unwrap().unwrap().num_rows(),
                expected_len as usize
            );
        }
    }

    #[tokio::test]
    async fn test_filter_parsing() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, true),
            ArrowField::new("s", DataType::Utf8, true),
        ]));

        let batches = RecordBatchBuffer::new(vec![RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..100)),
                Arc::new(StringArray::from_iter_values(
                    (0..100).map(|v| format!("s-{}", v)),
                )),
            ],
        )
        .unwrap()]);

        let mut batches: Box<dyn RecordBatchReader> = Box::new(batches);
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        Dataset::write(&mut batches, test_uri, None).await.unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let mut scan = dataset.scan();
        assert!(scan.filter.is_none());

        scan.filter("i > 50").unwrap();
        assert_eq!(scan.filter, Some("i > 50".to_string()));

        let batches = scan
            .project(&["s"])
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let batch = concat_batches(&batches[0].schema(), &batches).unwrap();

        let expected_batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "s",
                DataType::Utf8,
                true,
            )])),
            vec![Arc::new(StringArray::from_iter_values(
                (51..100).map(|v| format!("s-{}", v)),
            ))],
        )
        .unwrap();
        assert_eq!(batch, expected_batch);
    }

    #[tokio::test]
    async fn test_limit() {
        let temp = tempdir().unwrap();
        let mut file_path = PathBuf::from(temp.as_ref());
        file_path.push("limit_test.lance");
        let path = file_path.to_str().unwrap();
        let expected_batches = write_data(path).await;
        let expected_combined =
            concat_batches(&expected_batches[0].schema(), &expected_batches).unwrap();

        let dataset = Dataset::open(path).await.unwrap();
        let mut scanner = dataset.scan();
        scanner.limit(2, Some(19)).unwrap();
        let actual_batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .map(|b| b.unwrap())
            .collect::<Vec<RecordBatch>>()
            .await;
        let actual_combined = concat_batches(&actual_batches[0].schema(), &actual_batches).unwrap();

        assert_eq!(expected_combined.slice(19, 2), actual_combined);
        // skipped 1 batch
        assert_eq!(actual_batches.len(), 2);
    }

    async fn write_data(path: &str) -> Vec<RecordBatch> {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int64,
            true,
        )])) as SchemaRef;

        // Write 3 batches.
        let expected_batches: Vec<RecordBatch> = (0..3)
            .map(|batch_id| {
                let value_range = batch_id * 10..batch_id * 10 + 10;
                let columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from_iter(
                    value_range.clone().collect::<Vec<_>>(),
                ))];
                RecordBatch::try_new(schema.clone(), columns).unwrap()
            })
            .collect();
        let batches = RecordBatchBuffer::new(expected_batches.clone());
        let mut params = WriteParams::default();
        params.max_rows_per_group = 10;
        let mut reader: Box<dyn RecordBatchReader> = Box::new(batches);
        Dataset::write(&mut reader, path, Some(params))
            .await
            .unwrap();
        expected_batches
    }
}
