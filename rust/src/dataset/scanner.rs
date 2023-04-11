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
use arrow_schema::DataType;
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema, SchemaRef};
use datafusion::execution::{
    context::SessionState,
    runtime_env::{RuntimeConfig, RuntimeEnv},
};
use datafusion::physical_plan::{
    filter::FilterExec, limit::GlobalLimitExec, union::UnionExec, ExecutionPlan,
    SendableRecordBatchStream,
};
use datafusion::prelude::*;
use futures::stream::{Stream, StreamExt};

use super::Dataset;
use crate::datafusion::physical_expr::column_names_in_expr;
use crate::datatypes::Schema;
use crate::format::{Fragment, Index};
use crate::index::vector::{MetricType, Query};
use crate::io::exec::{
    KNNFlatExec, KNNIndexExec, LanceScanExec, Planner, ProjectionExec, TakeExec,
};
use crate::io::RecordBatchStream;
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
            nprobes: 1,
            refine_factor: None,
            metric_type: MetricType::L2,
            use_index: true,
        });
        Ok(self)
    }

    pub fn nprobs(&mut self, n: usize) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.nprobes = n;
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
        let schema = self
            .output_schema()
            .map(|s| SchemaRef::new(ArrowSchema::from(s.as_ref())))?;
        Ok(schema)
    }

    /// The output schema of the Scanner, in Lance Schema format.
    fn output_schema(&self) -> Result<Arc<Schema>> {
        let mut extra_columns = vec![];

        if let Some(q) = self.nearest.as_ref() {
            let vector_field = self
                .dataset
                .schema()
                .field(&q.column)
                .ok_or(Error::IO(format!("Column {} not found", q.column)))?;
            let vector_field = ArrowField::try_from(vector_field).map_err(|e| {
                Error::IO(format!("Failed to convert vector field: {}", e.to_string()))
            })?;
            extra_columns.push(vector_field);
            extra_columns.push(ArrowField::new("score", DataType::Float32, false));
        };
        if self.with_row_id {
            extra_columns.push(ArrowField::new(ROW_ID, DataType::UInt64, false));
        }

        let schema = if !extra_columns.is_empty() {
            let extra_schema = Schema::try_from(&ArrowSchema::new(extra_columns))?;
            self.projections.merge(&extra_schema)
        } else {
            self.projections.clone()
        };
        Ok(Arc::new(schema))
    }

    /// Create a stream of this Scanner.
    ///
    /// TODO: implement as IntoStream/IntoIterator.
    pub async fn try_into_stream(&self) -> Result<impl RecordBatchStream> {
        let plan = self.create_plan().await?;

        let session_config = SessionConfig::new();
        let runtime_config = RuntimeConfig::new();
        let runtime_env = Arc::new(RuntimeEnv::new(runtime_config)?);
        let session_state = SessionState::with_config_rt(session_config, runtime_env);
        Ok(ScannerRecordBatchStream::new(
            plan.execute(0, session_state.task_ctx())?,
        ))
    }

    /// Create [`ExecutionPlan`] for Scan.
    ///
    /// An ExecutionPlan is a graph of operators that can be executed.
    ///
    /// The following plans are supported:
    ///
    ///  - **Plain scan without filter or limits.**
    ///
    ///  ```ignore
    ///  Scan(projections)
    ///  ```
    ///
    ///  - **Scan with filter and/or limits.**
    ///
    ///  ```ignore
    ///  Scan(filtered_cols) -> Filter(expr)
    ///     -> (*LimitExec(limit, offset))
    ///     -> Take(remaining_cols) -> Projection()
    ///  ```
    ///
    ///  - **Use KNN Index (with filter and/or limits)**
    ///
    /// ```ignore
    /// KNNIndex() -> Take(vector) -> FlatRefine()
    ///     -> Take(filtered_cols) -> Filter(expr)
    ///     -> (*LimitExec(limit, offset))
    ///     -> Take(remaining_cols) -> Projection()
    /// ```
    ///
    /// - **Use KNN flat (brute force) with filter and/or limits**
    ///
    /// ```ignore
    /// Scan(vector) -> FlatKNN()
    ///     -> Take(filtered_cols) -> Filter(expr)
    ///     -> (*LimitExec(limit, offset))
    ///     -> Take(remaining_cols) -> Projection()
    /// ```
    ///
    /// In general, a plan has 4 stages:
    ///
    /// 1. Source (from dataset Scan or from index)
    /// 2. Filter
    /// 3. Limit / Offset
    /// 4. Take remaining columns / Projection
    async fn create_plan(&self) -> Result<Arc<dyn ExecutionPlan>> {
        let filter_expr = if let Some(filter) = self.filter.as_ref() {
            let planner = Planner::new(Arc::new(self.dataset.schema().into()));
            let logical_expr = planner.parse_filter(filter)?;
            Some(planner.create_physical_expr(&logical_expr)?)
        } else {
            None
        };

        // Stage 1: source
        let mut plan: Arc<dyn ExecutionPlan> = if self.nearest.is_some() {
            self.knn().await?
        } else if let Some(expr) = filter_expr.as_ref() {
            let columns_in_filter = column_names_in_expr(expr.as_ref());
            let filter_schema = Arc::new(self.dataset.schema().project(&columns_in_filter)?);
            self.scan(true, filter_schema)
        } else {
            // Scan without filter or limits
            self.scan(self.with_row_id, self.projections.clone().into())
        };

        // Stage 2: filter
        if let Some(predicates) = filter_expr.as_ref() {
            let columns_in_filter = column_names_in_expr(predicates.as_ref());
            let filter_schema = Arc::new(self.dataset.schema().project(&columns_in_filter)?);
            let remaining_schema = filter_schema.exclude(plan.schema().as_ref())?;
            if !remaining_schema.fields.is_empty() {
                // Not all columns for filter are ready, so we need to take them first
                plan = self.take(plan, &remaining_schema)?;
            }
            plan = Arc::new(FilterExec::try_new(predicates.clone(), plan)?);
        }

        // Stage 3: limit / offset
        if (self.limit.unwrap_or(0) > 0) || self.offset.is_some() {
            plan = self.limit_node(plan);
        }

        // Stage 4: take remaining columns / projection
        let output_schema = self.output_schema()?;
        let remaining_schema = output_schema.exclude(plan.schema().as_ref())?;
        if !remaining_schema.fields.is_empty() {
            plan = self.take(plan, &remaining_schema)?;
        }
        plan = Arc::new(ProjectionExec::try_new(plan, output_schema.clone())?);

        Ok(plan)
    }

    //
    async fn knn(&self) -> Result<Arc<dyn ExecutionPlan>> {
        let Some(q) = self.nearest.as_ref() else {
            return Err(Error::IO("No nearest query".to_string()));
        };

        let column_id = self.dataset.schema().field_id(q.column.as_str())?;
        let use_index = self.nearest.as_ref().map(|q| q.use_index).unwrap_or(false);
        let indices = if use_index {
            self.dataset.load_indices().await?
        } else {
            vec![]
        };
        let knn_idx = indices.iter().find(|i| i.fields.contains(&column_id));
        if let Some(index) = knn_idx {
            // There is an index built for the column.
            // We will use the index.
            if let Some(rf) = q.refine_factor {
                if rf == 0 {
                    return Err(Error::IO("Refine factor can not be zero".to_string()));
                }
            }

            let knn_node = self.ann(q, &index)?; // score, _rowid
            let with_vector = self.dataset.schema().project(&[&q.column])?;
            let knn_node_with_vector = self.take(knn_node, &with_vector)?;
            let mut knn_node = if q.refine_factor.is_some() {
                self.flat_knn(knn_node_with_vector, q)?
            } else {
                knn_node_with_vector
            }; // vector, score, _rowid

            knn_node = self.knn_combined(&q, index, knn_node).await?;

            Ok(knn_node)
        } else {
            // No index found. use flat search.
            let vector_scan_projection =
                Arc::new(self.dataset.schema().project(&[&q.column]).unwrap());
            let scan_node = self.scan(true, vector_scan_projection);
            Ok(self.flat_knn(scan_node, q)?)
        }
    }

    /// Combine ANN results with KNN results for data appended after index creation
    async fn knn_combined(
        &self,
        q: &&Query,
        index: &Index,
        knn_node: Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Check if we've created new versions since the index
        let version = index.dataset_version;
        if version != self.dataset.version().version {
            // If we've added more rows, then we'll have new fragments
            let ds = self.dataset.checkout_version(version).await?;
            let max_fragment_id_idx = ds
                .manifest
                .max_fragment_id()
                .ok_or_else(|| Error::IO("No fragments in index version".to_string()))?;
            let max_fragment_id_ds = self
                .dataset
                .manifest
                .max_fragment_id()
                .ok_or_else(|| Error::IO("No fragments in dataset version".to_string()))?;
            // If we have new fragments, then we need to do a combined search
            if max_fragment_id_idx < max_fragment_id_ds {
                let vector_scan_projection =
                    Arc::new(self.dataset.schema().project(&[&q.column]).unwrap());
                let scan_node = self.scan_fragments(
                    true,
                    vector_scan_projection,
                    Arc::new(self.dataset.manifest.fragments_since(&ds.manifest)?),
                );
                // first we do flat search on just the new data
                let topk_appended = self.flat_knn(scan_node, q)?;
                // union
                let unioned = UnionExec::new(vec![topk_appended, knn_node]);
                // then we do a flat search on KNN(new data) + ANN(indexed data)
                return self.flat_knn(Arc::new(unioned), q);
            }
        }
        Ok(knn_node)
    }

    /// Create an Execution plan with a scan node
    fn scan(&self, with_row_id: bool, projection: Arc<Schema>) -> Arc<dyn ExecutionPlan> {
        self.scan_fragments(with_row_id, projection, self.dataset.fragments().clone())
    }

    fn scan_fragments(
        &self,
        with_row_id: bool,
        projection: Arc<Schema>,
        fragments: Arc<Vec<Fragment>>,
    ) -> Arc<dyn ExecutionPlan> {
        Arc::new(LanceScanExec::new(
            self.dataset.clone(),
            fragments,
            projection,
            self.batch_size,
            PREFETCH_SIZE,
            with_row_id,
        ))
    }

    /// Add a knn search node to the input plan
    fn flat_knn(&self, input: Arc<dyn ExecutionPlan>, q: &Query) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(KNNFlatExec::try_new(input, q.clone())?))
    }

    /// Create an Execution plan to do indexed ANN search
    fn ann(&self, q: &Query, index: &Index) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(KNNIndexExec::try_new(
            self.dataset.clone(),
            &index.uuid.to_string(),
            q,
        )?))
    }

    /// Take row indices produced by input plan from the dataset (with projection)
    fn take(
        &self,
        input: Arc<dyn ExecutionPlan>,
        projection: &Schema,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(TakeExec::try_new(
            self.dataset.clone(),
            input,
            Arc::new(projection.clone()),
        )?))
    }

    /// Global offset-limit of the result of the input plan
    fn limit_node(&self, plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        Arc::new(GlobalLimitExec::new(
            plan,
            *self.offset.as_ref().unwrap_or(&0) as usize,
            self.limit.map(|l| l as usize),
        ))
    }
}

/// ScannerStream is a container to wrap different types of ExecNode.
#[pin_project::pin_project]
pub struct ScannerRecordBatchStream {
    #[pin]
    exec_node: SendableRecordBatchStream,
}

impl ScannerRecordBatchStream {
    pub fn new(exec_node: SendableRecordBatchStream) -> Self {
        Self { exec_node }
    }
}

impl RecordBatchStream for ScannerRecordBatchStream {
    fn schema(&self) -> SchemaRef {
        self.exec_node.schema()
    }
}

impl Stream for ScannerRecordBatchStream {
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

    use std::collections::BTreeSet;
    use std::path::PathBuf;
    use std::vec;

    use arrow::array::as_primitive_array;
    use arrow::compute::concat_batches;
    use arrow::datatypes::Int32Type;
    use arrow_array::{
        ArrayRef, FixedSizeListArray, Int32Array, Int64Array, LargeStringArray, RecordBatchReader,
        StringArray,
    };
    use arrow_schema::DataType;
    use futures::TryStreamExt;
    use tempfile::tempdir;

    use super::*;
    use crate::arrow::*;
    use crate::index::{vector::VectorIndexParams, IndexType};
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

    async fn create_vector_dataset(path: &str, build_index: bool) -> Arc<Dataset> {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, true),
            ArrowField::new("s", DataType::Utf8, true),
            ArrowField::new(
                "vec",
                DataType::FixedSizeList(
                    Box::new(ArrowField::new("item", DataType::Float32, true)),
                    32,
                ),
                true,
            ),
        ]));

        let batches = RecordBatchBuffer::new(
            (0..5)
                .map(|i| {
                    let vector_values: Float32Array = (0..32 * 80).map(|v| v as f32).collect();
                    let vectors = FixedSizeListArray::try_new(&vector_values, 32).unwrap();
                    RecordBatch::try_new(
                        schema.clone(),
                        vec![
                            Arc::new(Int32Array::from_iter_values(i * 80..(i + 1) * 80)),
                            Arc::new(StringArray::from_iter_values(
                                (i * 80..(i + 1) * 80).map(|v| format!("s-{}", v)),
                            )),
                            Arc::new(vectors),
                        ],
                    )
                    .unwrap()
                })
                .collect(),
        );

        let mut params = WriteParams::default();
        params.max_rows_per_group = 10;
        let mut reader: Box<dyn RecordBatchReader> = Box::new(batches);

        let dataset = Dataset::write(&mut reader, path, Some(params))
            .await
            .unwrap();

        if build_index {
            let params = VectorIndexParams::ivf_pq(2, 8, 2, false, MetricType::L2, 2);
            dataset
                .create_index(
                    &["vec"],
                    IndexType::Vector,
                    Some("idx".to_string()),
                    &params,
                    true,
                )
                .await
                .unwrap();
        }

        Arc::new(Dataset::open(path).await.unwrap())
    }

    #[tokio::test]
    async fn test_knn_nodes() {
        for build_index in &[true, false] {
            let test_dir = tempdir().unwrap();
            let test_uri = test_dir.path().to_str().unwrap();
            let dataset = create_vector_dataset(test_uri, *build_index).await;
            let mut scan = dataset.scan();
            let key: Float32Array = (32..64).map(|v| v as f32).collect();
            scan.nearest("vec", &key, 5).unwrap();
            scan.refine(5);

            let results = scan
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();

            assert_eq!(results.len(), 1);
            let batch = &results[0];

            assert_eq!(batch.num_rows(), 5);
            assert_eq!(
                batch.schema().as_ref(),
                &ArrowSchema::new(vec![
                    ArrowField::new("i", DataType::Int32, true),
                    ArrowField::new("s", DataType::Utf8, true),
                    ArrowField::new(
                        "vec",
                        DataType::FixedSizeList(
                            Box::new(ArrowField::new("item", DataType::Float32, true)),
                            32,
                        ),
                        true,
                    ),
                    ArrowField::new("score", DataType::Float32, false),
                ])
            );

            let expected_i = BTreeSet::from_iter(vec![1, 81, 161, 241, 321]);
            let column_i = batch.column_by_name("i").unwrap();
            let actual_i: BTreeSet<i32> = as_primitive_array::<Int32Type>(column_i.as_ref())
                .values()
                .iter()
                .copied()
                .collect();
            assert_eq!(expected_i, actual_i);
        }
    }

    #[tokio::test]
    async fn test_knn_with_filter() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_vector_dataset(test_uri, true).await;
        let mut scan = dataset.scan();
        let key: Float32Array = (32..64).map(|v| v as f32).collect();
        scan.nearest("vec", &key, 5).unwrap();
        scan.filter("i > 100").unwrap();
        scan.project(&["i"]).unwrap();
        scan.refine(5);

        let results = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        let batch = &results[0];

        assert_eq!(batch.num_rows(), 3);
        assert_eq!(
            batch.schema().as_ref(),
            &ArrowSchema::new(vec![
                ArrowField::new("i", DataType::Int32, true),
                ArrowField::new(
                    "vec",
                    DataType::FixedSizeList(
                        Box::new(ArrowField::new("item", DataType::Float32, true)),
                        32,
                    ),
                    true,
                ),
                ArrowField::new("score", DataType::Float32, false),
            ])
        );

        let expected_i = BTreeSet::from_iter(vec![161, 241, 321]);
        let column_i = batch.column_by_name("i").unwrap();
        let actual_i: BTreeSet<i32> = as_primitive_array::<Int32Type>(column_i.as_ref())
            .values()
            .iter()
            .copied()
            .collect();
        assert_eq!(expected_i, actual_i);
    }

    #[tokio::test]
    async fn test_refine_factor() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let dataset = create_vector_dataset(test_uri, true).await;
        let mut scan = dataset.scan();
        let key: Float32Array = (32..64).map(|v| v as f32).collect();
        scan.nearest("vec", &key, 5).unwrap();
        scan.refine(5);

        let results = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        let batch = &results[0];

        assert_eq!(batch.num_rows(), 5);
        assert_eq!(
            batch.schema().as_ref(),
            &ArrowSchema::new(vec![
                ArrowField::new("i", DataType::Int32, true),
                ArrowField::new("s", DataType::Utf8, true),
                ArrowField::new(
                    "vec",
                    DataType::FixedSizeList(
                        Box::new(ArrowField::new("item", DataType::Float32, true)),
                        32,
                    ),
                    true,
                ),
                ArrowField::new("score", DataType::Float32, false),
            ])
        );

        let expected_i = BTreeSet::from_iter(vec![1, 81, 161, 241, 321]);
        let column_i = batch.column_by_name("i").unwrap();
        let actual_i: BTreeSet<i32> = as_primitive_array::<Int32Type>(column_i.as_ref())
            .values()
            .iter()
            .copied()
            .collect();
        assert_eq!(expected_i, actual_i);
    }

    #[tokio::test]
    async fn test_simple_scan_plan() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_vector_dataset(test_uri, false).await;

        let scan = dataset.scan();
        let plan = scan.create_plan().await.unwrap();

        assert!(plan.as_any().is::<ProjectionExec>());
        assert_eq!(plan.schema().field_names(), ["i", "s", "vec"]);

        let scan = &plan.children()[0];
        assert!(scan.as_any().is::<LanceScanExec>());
        assert_eq!(plan.schema().field_names(), ["i", "s", "vec"]);

        let mut scan = dataset.scan();
        scan.project(&["s"]).unwrap();
        let plan = scan.create_plan().await.unwrap();
        assert!(plan.as_any().is::<ProjectionExec>());
        assert_eq!(plan.schema().field_names(), ["s"]);

        let scan = &plan.children()[0];
        assert!(scan.as_any().is::<LanceScanExec>());
        assert_eq!(scan.schema().field_names(), ["s"]);
    }

    #[tokio::test]
    async fn test_scan_with_row_id() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_vector_dataset(test_uri, false).await;

        let mut scan = dataset.scan();
        scan.project(&["i"]).unwrap();
        scan.with_row_id();
        let plan = scan.create_plan().await.unwrap();

        assert!(plan.as_any().is::<ProjectionExec>());
        assert_eq!(plan.schema().field_names(), &["i", "_rowid"]);
        let scan = &plan.children()[0];
        assert!(scan.as_any().is::<LanceScanExec>());
        assert_eq!(scan.schema().field_names(), &["i", "_rowid"]);
    }

    /// Test scan with filter.
    ///
    /// Query:
    ///
    /// ```
    /// SELECT s FROM dataset WHERE i > 10 and i < 20
    /// ```
    ///
    /// Expected plan:
    ///  scan(i) -> filter(i) -> take(s) -> projection(s)
    #[tokio::test]
    async fn test_scan_with_filter() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_vector_dataset(test_uri, false).await;

        let mut scan = dataset.scan();
        scan.project(&["s"]).unwrap();
        scan.filter("i > 10 and i < 20").unwrap();
        let plan = scan.create_plan().await.unwrap();

        assert!(plan.as_any().is::<ProjectionExec>());
        assert_eq!(plan.schema().field_names(), ["s"]);

        let take = &plan.children()[0];
        assert!(take.as_any().is::<TakeExec>());
        assert_eq!(take.schema().field_names(), ["i", "_rowid", "s"]);

        let filter = &take.children()[0];
        assert!(filter.as_any().is::<FilterExec>());
        assert_eq!(filter.schema().field_names(), ["i", "_rowid"]);

        let scan = &filter.children()[0];
        assert!(scan.as_any().is::<LanceScanExec>());
        assert_eq!(filter.schema().field_names(), ["i", "_rowid"]);
    }

    /// Test KNN with index
    ///
    /// Query: nearest(vec, [...], 10) + filter(i > 10 and i < 20)
    ///
    /// Expected plan:
    ///  KNNIndex(vec) -> Take(i) -> filter(i) -> take(s, vec) -> projection(s, vec, score)
    #[tokio::test]
    async fn test_ann_with_index() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_vector_dataset(test_uri, true).await;

        let mut scan = dataset.scan();
        let key: Float32Array = (32..64).map(|v| v as f32).collect();
        scan.nearest("vec", &key, 10).unwrap();
        scan.project(&["s"]).unwrap();
        scan.filter("i > 10 and i < 20").unwrap();

        let plan = scan.create_plan().await.unwrap();

        assert!(plan.as_any().is::<ProjectionExec>());
        assert_eq!(
            plan.schema()
                .fields()
                .iter()
                .map(|f| f.name())
                .collect::<Vec<_>>(),
            vec!["s", "vec", "score"]
        );

        let take = &plan.children()[0];
        let take = take.as_any().downcast_ref::<TakeExec>().unwrap();
        assert_eq!(
            take.schema().field_names(),
            ["score", "_rowid", "vec", "i", "s"]
        );
        assert_eq!(
            take.extra_schema
                .fields
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>(),
            vec!["s"]
        );

        let filter = &take.children()[0];
        assert!(filter.as_any().is::<FilterExec>());
        assert_eq!(
            filter.schema().field_names(),
            ["score", "_rowid", "vec", "i"]
        );

        let take = &filter.children()[0];
        let take = take.as_any().downcast_ref::<TakeExec>().unwrap();
        assert_eq!(take.schema().field_names(), ["score", "_rowid", "vec", "i"]);
        assert_eq!(
            take.extra_schema
                .fields
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>(),
            vec!["i"]
        );

        // TODO: Two continuous take execs, we can merge them into one.
        let take = &take.children()[0];
        let take = take.as_any().downcast_ref::<TakeExec>().unwrap();
        assert_eq!(take.schema().field_names(), ["score", "_rowid", "vec"]);
        assert_eq!(
            take.extra_schema
                .fields
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>(),
            vec!["vec"]
        );

        let knn = &take.children()[0];
        assert!(knn.as_any().is::<KNNIndexExec>());
        assert_eq!(knn.schema().field_names(), ["score", "_rowid"]);
    }

    /// Test KNN index with refine factor
    ///
    /// Query: nearest(vec, [...], 10, refine_factor=10) + filter(i > 10 and i < 20)
    ///
    /// Expected plan:
    ///  KNNIndex(vec) -> Take(vec) -> KNNFlat(vec, 10) -> Take(i) -> Filter(i)
    ///     -> take(s, vec) -> projection(s, vec, score)
    #[tokio::test]
    async fn test_knn_with_refine() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_vector_dataset(test_uri, true).await;

        let mut scan = dataset.scan();
        let key: Float32Array = (32..64).map(|v| v as f32).collect();
        scan.nearest("vec", &key, 10).unwrap();
        scan.refine(10);
        scan.project(&["s"]).unwrap();
        scan.filter("i > 10 and i < 20").unwrap();

        let plan = scan.create_plan().await.unwrap();

        assert!(plan.as_any().is::<ProjectionExec>());
        assert_eq!(
            plan.schema()
                .fields()
                .iter()
                .map(|f| f.name())
                .collect::<Vec<_>>(),
            vec!["s", "vec", "score"]
        );

        let take = &plan.children()[0];
        let take = take.as_any().downcast_ref::<TakeExec>().unwrap();
        assert_eq!(
            take.schema().field_names(),
            ["score", "_rowid", "vec", "i", "s"]
        );
        assert_eq!(
            take.extra_schema
                .fields
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>(),
            vec!["s"]
        );

        let filter = &take.children()[0];
        assert!(filter.as_any().is::<FilterExec>());
        assert_eq!(
            filter.schema().field_names(),
            ["score", "_rowid", "vec", "i"]
        );

        let take = &filter.children()[0];
        let take = take.as_any().downcast_ref::<TakeExec>().unwrap();
        assert_eq!(take.schema().field_names(), ["score", "_rowid", "vec", "i"]);
        assert_eq!(
            take.extra_schema
                .fields
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>(),
            vec!["i"]
        );

        // Flat refine step
        let flat = &take.children()[0];
        assert!(flat.as_any().is::<KNNFlatExec>());

        let take = &flat.children()[0];
        let take = take.as_any().downcast_ref::<TakeExec>().unwrap();
        assert_eq!(take.schema().field_names(), ["score", "_rowid", "vec"]);
        assert_eq!(
            take.extra_schema
                .fields
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>(),
            vec!["vec"]
        );

        let knn = &take.children()[0];
        assert!(knn.as_any().is::<KNNIndexExec>());
        assert_eq!(knn.schema().field_names(), ["score", "_rowid"]);
    }

    #[tokio::test]
    async fn test_filter_on_large_utf8() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "ls",
            DataType::LargeUtf8,
            true,
        )]));

        let batches = RecordBatchBuffer::new(vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(LargeStringArray::from_iter_values(
                (0..10).map(|v| format!("s-{}", v)),
            ))],
        )
        .unwrap()]);

        let write_params = WriteParams::default();
        let mut batches: Box<dyn RecordBatchReader> = Box::new(batches);
        Dataset::write(&mut batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let mut scan = dataset.scan();
        scan.filter("ls = 's-8'").unwrap();

        let batches = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let batch = &batches[0];

        let expected = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(LargeStringArray::from_iter_values(
                (8..9).map(|v| format!("s-{}", v)),
            ))],
        )
        .unwrap();

        assert_eq!(batch, &expected);
    }
}
