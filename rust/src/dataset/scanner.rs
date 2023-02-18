// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::{Float32Array, RecordBatch};
use arrow_schema::DataType::Float32;
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema, SchemaRef};
use datafusion::execution::context::SessionState;
use datafusion::execution::runtime_env::{RuntimeConfig, RuntimeEnv};
use datafusion::physical_plan::limit::GlobalLimitExec;
use datafusion::physical_plan::{ExecutionPlan, SendableRecordBatchStream};
use datafusion::prelude::*;
use futures::stream::{Stream, StreamExt};
use object_store::path::Path;
use sqlparser::ast::{Expr, SetExpr, Statement};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use super::Dataset;
use crate::datatypes::Schema;
use crate::format::{Fragment, Index, Manifest};
use crate::index::vector::Query;
use crate::io::exec::{GlobalTakeExec, KNNFlatExec, KNNIndexExec, LanceScanExec};
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

    /// Optional filters.
    filter: Option<Expr>,

    /// The batch size controls the maximum size of rows to return for each read.
    batch_size: usize,

    limit: Option<i64>,
    offset: Option<i64>,

    fragments: Arc<Vec<Fragment>>,

    nearest: Option<Query>,

    /// Scan the dataset with a meta column: "_rowid"
    with_row_id: bool,
}

impl Scanner {
    pub fn new(dataset: Arc<Dataset>) -> Self {
        let projection = dataset.schema().clone();
        let fragments = dataset.fragments().clone();
        Self {
            dataset,
            projections: projection,
            filter: None,
            batch_size: DEFAULT_BATCH_SIZE,
            limit: None,
            offset: None,
            fragments,
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
        let sql = format!("SELECT 1 FROM t WHERE {filter}");

        let dialect = GenericDialect {};
        let stmts = Parser::parse_sql(&dialect, sql.as_str())?;
        if stmts.len() != 1 {
            return Err(Error::IO(format!("Filter is not valid: {filter}")));
        }
        if let Statement::Query(query) = &stmts[0] {
            if let SetExpr::Select(s) = query.body.as_ref() {
                self.filter = s.selection.clone();
                return Ok(self);
            }
        }

        return Err(Error::IO(format!("Filter is not valid: {filter}")));
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

    /// Instruct the scanner to return the `_rowid` meta column from the dataset.
    pub fn with_row_id(&mut self) -> &mut Self {
        self.with_row_id = true;
        self
    }

    /// The schema of the output, a.k.a, projection schema.
    pub fn schema(&self) -> Result<SchemaRef> {
        if self.nearest.as_ref().is_some() {
            let q = self.nearest.as_ref().unwrap();
            let column: ArrowField = self
                .dataset
                .schema()
                .field(q.column.as_str())
                .ok_or_else(|| {
                    Error::Schema(format!("Vector column {} not found in schema", q.column))
                })?
                .into();
            let score = ArrowField::new("score", Float32, false);
            let score_schema = ArrowSchema::new(vec![column, score]);
            let to_merge = &Schema::try_from(&score_schema).unwrap();
            let merged = self.projections.merge(to_merge);
            Ok(SchemaRef::new(ArrowSchema::from(&merged)))
        } else {
            Ok(Arc::new(ArrowSchema::from(&self.projections)))
        }
    }

    fn should_use_index(&self) -> bool {
        self.nearest.is_some()
    }

    /// Create a stream of this Scanner.
    ///
    /// TODO: implement as IntoStream/IntoIterator.
    pub async fn try_into_stream(&self) -> Result<RecordBatchStream> {
        let data_dir = self.dataset.data_dir();
        let manifest = self.dataset.manifest.clone();
        let with_row_id = self.with_row_id;
        let projection = &self.projections;

        let indices = if self.should_use_index() {
            self.dataset.load_indices().await?
        } else {
            vec![]
        };

        // TODO: refactor to a DataFusion QueryPlanner
        let mut plan: Arc<dyn ExecutionPlan> = if let Some(q) = self.nearest.as_ref() {
            let column_id = self
                .dataset
                .schema()
                .field(&q.column)
                .expect("vector column does not exist")
                .id;

            if let Some(rf) = q.refine_factor {
                if rf == 0 {
                    return Err(Error::IO("Refine factor can not be zero".to_string()));
                }
            }
            let nn_node: Arc<dyn ExecutionPlan> =
                if let Some(index) = indices.iter().find(|i| i.fields.contains(&column_id)) {
                    // There is an index built for the column.
                    // We will use the index.
                    self.ann(q, &index)
                } else {
                    let vector_scan_projection =
                        Arc::new(self.dataset.schema().project(&[&q.column]).unwrap());
                    let scan_node =
                        self.scan(&data_dir, manifest.clone(), true, vector_scan_projection);
                    self.knn(scan_node, &q)
                };

            let take_node = self.take(nn_node, projection);

            if q.refine_factor.is_some() {
                self.knn(take_node, &q)
            } else {
                take_node
            }
        } else {
            self.scan(
                &data_dir,
                manifest,
                with_row_id,
                Arc::new(self.projections.clone()),
            )
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

    /// Create an Execution plan with a scan node
    fn scan(
        &self,
        data_dir: &Path,
        manifest: Arc<Manifest>,
        with_row_id: bool,
        projection: Arc<Schema>,
    ) -> Arc<dyn ExecutionPlan> {
        Arc::new(LanceScanExec::new(
            self.dataset.object_store.clone(),
            data_dir.clone(),
            self.fragments.clone(),
            projection,
            manifest.clone(),
            self.batch_size,
            PREFETCH_SIZE,
            with_row_id,
        ))
    }

    /// Add a knn search node to the input plan
    fn knn(&self, input: Arc<dyn ExecutionPlan>, q: &Query) -> Arc<dyn ExecutionPlan> {
        Arc::new(KNNFlatExec::new(input, q.clone()))
    }

    /// Create an Execution plan to do indexed ANN search
    fn ann(&self, q: &Query, index: &&Index) -> Arc<dyn ExecutionPlan> {
        let mut inner_query = q.clone();
        inner_query.k = q.k * (q.refine_factor.unwrap_or(1) as usize);
        Arc::new(KNNIndexExec::new(
            self.dataset.clone(),
            &index.uuid.to_string(),
            &inner_query,
        ))
    }

    /// Take row indices produced by input plan from the dataset (with projection)
    fn take(&self, indices: Arc<dyn ExecutionPlan>, projection: &Schema) -> Arc<dyn ExecutionPlan> {
        Arc::new(GlobalTakeExec::new(
            self.dataset.clone(),
            Arc::new(projection.clone()),
            indices,
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
    use sqlparser::ast::*;
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

        scan.filter("a > 50").unwrap();
        println!("Filter is: {:?}", scan.filter);
        assert_eq!(
            scan.filter,
            Some(Expr::BinaryOp {
                left: Box::new(Expr::Identifier(Ident::new("a"))),
                op: BinaryOperator::Gt,
                right: Box::new(Expr::Value(Value::Number(String::from("50"), false)))
            })
        );
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
