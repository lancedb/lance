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

use arrow_array::{Array, Float32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema, SchemaRef};
use datafusion::execution::{
    context::SessionState,
    runtime_env::{RuntimeConfig, RuntimeEnv},
};
use datafusion::logical_expr::AggregateFunction;
use datafusion::physical_plan::{
    aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
    display::DisplayableExecutionPlan,
    expressions::{create_aggregate_expr, Literal},
    filter::FilterExec,
    limit::GlobalLimitExec,
    repartition::RepartitionExec,
    union::UnionExec,
    ExecutionPlan, PhysicalExpr, SendableRecordBatchStream,
};
use datafusion::prelude::*;
use datafusion::scalar::ScalarValue;
use futures::stream::{Stream, StreamExt};
use lance_linalg::distance::MetricType;
use log::debug;
use tracing::{info_span, instrument, Span};

use super::Dataset;
use crate::datafusion::physical_expr::column_names_in_expr;
use crate::datatypes::Schema;
use crate::format::{Fragment, Index};
use crate::index::vector::Query;
use crate::io::{
    exec::{KNNFlatExec, KNNIndexExec, LanceScanExec, Planner, ProjectionExec, TakeExec},
    RecordBatchStream,
};
use crate::utils::sql::parse_sql_filter;
use crate::{Error, Result};
use snafu::{location, Location};
/// Column name for the meta row ID.
pub const ROW_ID: &str = "_rowid";
pub const DEFAULT_BATCH_SIZE: usize = 8192;

// Same as pyarrow Dataset::scanner()
const DEFAULT_BATCH_READAHEAD: usize = 16;

// Same as pyarrow Dataset::scanner()
const DEFAULT_FRAGMENT_READAHEAD: usize = 4;

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

    /// If true then the filter will be applied before an index scan
    prefilter: bool,

    /// Optional filters string.
    filter: Option<String>,

    /// The batch size controls the maximum size of rows to return for each read.
    batch_size: usize,

    /// Number of batches to prefetch
    batch_readahead: usize,

    /// Number of fragments to read concurrently
    fragment_readahead: usize,

    limit: Option<i64>,
    offset: Option<i64>,

    nearest: Option<Query>,

    /// Scan the dataset with a meta column: "_rowid"
    with_row_id: bool,

    /// Whether to scan in deterministic order (default: true)
    ordered: bool,

    /// If set, this scanner serves only these fragments.
    fragments: Option<Vec<Fragment>>,
}

impl Scanner {
    pub fn new(dataset: Arc<Dataset>) -> Self {
        let projection = dataset.schema().clone();

        // Default batch size to be large enough so that a i32 column can be
        // read in a single range request. For the object store default of
        // 64KB, this is 16K rows. For local file systems, the default block size
        // is just 4K, which would mean only 1K rows, which might be a little small.
        // So we use a default minimum of 8K rows.
        let batch_size = std::cmp::max(dataset.object_store().block_size() / 4, DEFAULT_BATCH_SIZE);
        Self {
            dataset,
            projections: projection,
            prefilter: false,
            filter: None,
            batch_size,
            batch_readahead: DEFAULT_BATCH_READAHEAD,
            fragment_readahead: DEFAULT_FRAGMENT_READAHEAD,
            limit: None,
            offset: None,
            nearest: None,
            with_row_id: false,
            ordered: true,
            fragments: None,
        }
    }

    pub fn from_fragment(dataset: Arc<Dataset>, fragment: Fragment) -> Self {
        let projection = dataset.schema().clone();
        let batch_size = std::cmp::max(dataset.object_store().block_size() / 4, DEFAULT_BATCH_SIZE);
        Self {
            dataset,
            projections: projection,
            prefilter: false,
            filter: None,
            batch_size,
            batch_readahead: DEFAULT_BATCH_READAHEAD,
            fragment_readahead: DEFAULT_FRAGMENT_READAHEAD,
            limit: None,
            offset: None,
            nearest: None,
            with_row_id: false,
            ordered: true,
            fragments: Some(vec![fragment]),
        }
    }

    /// Set which fragments should be scanned.
    ///
    /// If scan_in_order is set to true, the fragments will be scanned in the order of the vector.
    pub fn with_fragments(&mut self, fragments: Vec<Fragment>) -> &mut Self {
        self.fragments = Some(fragments);
        self
    }

    fn ensure_not_fragment_scan(&self) -> Result<()> {
        if self.is_fragment_scan() {
            Err(Error::IO {
                message: "This operation is not supported for fragment scan".to_string(),
                location: location!(),
            })
        } else {
            Ok(())
        }
    }

    fn is_fragment_scan(&self) -> bool {
        self.fragments.is_some()
    }

    /// Projection.
    ///
    /// Only select the specified columns. If not specified, all columns will be scanned.
    pub fn project<T: AsRef<str>>(&mut self, columns: &[T]) -> Result<&mut Self> {
        self.projections = self.dataset.schema().project(columns)?;
        Ok(self)
    }

    /// Should the filter run before the vector index is applied
    ///
    /// If true then the filter will be applied before the vector index.  This
    /// means the results will be accurate but the overall query may be more expensive.
    ///
    /// If false then the filter will be applied to the nearest results.  This means
    /// you may get back fewer results than you ask for (or none at all) if the closest
    /// results do not match the filter.
    ///
    /// EXPERIMENTAL: Currently, prefiltering is only supported when a vector index
    /// is not used and the query is performing a flat knn search.  If the filter matches
    /// many rows then the query could be very expensive.
    pub fn prefilter(&mut self, should_prefilter: bool) -> &mut Self {
        self.prefilter = should_prefilter;
        self
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

    /// Set the prefetch size.
    pub fn batch_readahead(&mut self, nbatches: usize) -> &mut Self {
        self.batch_readahead = nbatches;
        self
    }

    /// Set the fragment readahead.
    ///
    /// This is only used if ``scan_in_order`` is set to false.
    pub fn fragment_readahead(&mut self, nfragments: usize) -> &mut Self {
        self.fragment_readahead = nfragments;
        self
    }

    /// Set whether to read data in order (default: true)
    ///
    /// If true, will scan the fragments and batches within fragments in order.
    /// If false, scan will read fragments concurrently and may yield batches
    /// out of order, potentially improving throughput.
    pub fn scan_in_order(&mut self, ordered: bool) -> &mut Self {
        self.ordered = ordered;
        self
    }

    /// Set limit and offset.
    ///
    /// If offset is set, the first offset rows will be skipped. If limit is set,
    /// only the provided number of rows will be returned. These can be set
    /// independently. For example, setting offset to 10 and limit to None will
    /// skip the first 10 rows and return the rest of the rows in the dataset.
    pub fn limit(&mut self, limit: Option<i64>, offset: Option<i64>) -> Result<&mut Self> {
        if limit.unwrap_or_default() < 0 {
            return Err(Error::IO {
                message: "Limit must be non-negative".to_string(),
                location: location!(),
            });
        }
        if let Some(off) = offset {
            if off < 0 {
                return Err(Error::IO {
                    message: "Offset must be non-negative".to_string(),
                    location: location!(),
                });
            }
        }
        self.limit = limit;
        self.offset = offset;
        Ok(self)
    }

    /// Find k-nearest neighbor within the vector column.
    pub fn nearest(&mut self, column: &str, q: &Float32Array, k: usize) -> Result<&mut Self> {
        self.ensure_not_fragment_scan()?;

        if k == 0 {
            return Err(Error::IO {
                message: "k must be positive".to_string(),
                location: location!(),
            });
        }
        if q.is_empty() {
            return Err(Error::IO {
                message: "Query vector must have non-zero length".to_string(),
                location: location!(),
            });
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

    /// The Arrow schema of the output, including projections and vector / _distance
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
            let vector_field = self.dataset.schema().field(&q.column).ok_or(Error::IO {
                message: format!("Column {} not found", q.column),
                location: location!(),
            })?;
            let vector_field = ArrowField::try_from(vector_field).map_err(|e| Error::IO {
                message: format!("Failed to convert vector field: {}", e),
                location: location!(),
            })?;
            extra_columns.push(vector_field);
            extra_columns.push(ArrowField::new("_distance", DataType::Float32, false));
        };
        if self.with_row_id {
            extra_columns.push(ArrowField::new(ROW_ID, DataType::UInt64, false));
        }

        let schema = if !extra_columns.is_empty() {
            self.projections.merge(&ArrowSchema::new(extra_columns))?
        } else {
            self.projections.clone()
        };
        Ok(Arc::new(schema))
    }

    fn execute_plan(&self, plan: Arc<dyn ExecutionPlan>) -> Result<SendableRecordBatchStream> {
        let session_config = SessionConfig::new();
        let runtime_config = RuntimeConfig::new();
        let runtime_env = Arc::new(RuntimeEnv::new(runtime_config)?);
        let session_state = SessionState::with_config_rt(session_config, runtime_env);
        // NOTE: we are only executing the first partition here. Therefore, if
        // the plan has more than one partition, we will be missing data.
        assert_eq!(plan.output_partitioning().partition_count(), 1);
        Ok(plan.execute(0, session_state.task_ctx())?)
    }

    /// Create a stream from the Scanner.
    #[instrument(skip_all)]
    pub async fn try_into_stream(&self) -> Result<DatasetRecordBatchStream> {
        let plan = self.create_plan().await?;
        Ok(DatasetRecordBatchStream::new(self.execute_plan(plan)?))
    }

    /// Scan and return the number of matching rows
    #[instrument(skip_all)]
    pub async fn count_rows(&self) -> Result<u64> {
        let plan = self.create_plan().await?;
        // Datafusion interprets COUNT(*) as COUNT(1)
        let one = Arc::new(Literal::new(ScalarValue::UInt8(Some(1))));
        let count_expr = create_aggregate_expr(
            &AggregateFunction::Count,
            false,
            &[one],
            &[],
            &plan.schema(),
            "",
        )?;
        let plan_schema = plan.schema().clone();
        let count_plan = Arc::new(AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(Vec::new()),
            vec![count_expr],
            vec![None],
            vec![None],
            plan,
            plan_schema,
        )?);
        let mut stream = self.execute_plan(count_plan)?;

        // A count plan will always return a single batch with a single row.
        if let Some(first_batch) = stream.next().await {
            let batch = first_batch?;
            let array = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or(Error::IO {
                    message: "Count plan did not return a UInt64Array".to_string(),
                    location: location!(),
                })?;
            Ok(array.value(0) as u64)
        } else {
            Ok(0)
        }
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
    /// 1. Source (from dataset Scan or from index, may include prefilter)
    /// 2. Filter
    /// 3. Limit / Offset
    /// 4. Take remaining columns / Projection
    async fn create_plan(&self) -> Result<Arc<dyn ExecutionPlan>> {
        // NOTE: we only support node that have one partition. So any nodes that
        // produce multiple need to be repartitioned to 1.
        let mut filter_expr = if let Some(filter) = self.filter.as_ref() {
            let planner = Planner::new(Arc::new(self.dataset.schema().into()));
            let logical_expr = planner.parse_filter(filter)?;
            Some(planner.create_physical_expr(&logical_expr)?)
        } else {
            None
        };

        // Stage 1: source
        let mut plan: Arc<dyn ExecutionPlan> = if self.nearest.is_some() {
            if self.prefilter {
                let prefilter = filter_expr;
                filter_expr = None;
                self.knn(prefilter).await?
            } else {
                self.knn(None).await?
            }
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
                plan = self.take(plan, &remaining_schema, self.batch_readahead)?;
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
            plan = self.take(plan, &remaining_schema, self.batch_readahead)?;
        }
        plan = Arc::new(ProjectionExec::try_new(plan, output_schema)?);

        debug!("Execution plan:\n{:?}", plan);

        Ok(plan)
    }

    // KNN search execution node with optional prefilter
    async fn knn(
        &self,
        prefilter: Option<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let Some(q) = self.nearest.as_ref() else {
            return Err(Error::IO {
                message: "No nearest query".to_string(),
                location: location!(),
            });
        };

        // Santity check
        let schema = self.dataset.schema();
        if let Some(field) = schema.field(&q.column) {
            match field.data_type() {
                DataType::FixedSizeList(subfield, _)
                    if matches!(subfield.data_type(), DataType::Float32) => {}
                _ => {
                    return Err(Error::IO {
                        message: format!(
                            "Vector search error: column {} is not a vector type: expected FixedSizeList<Float32>, got {}",
                            q.column, field.data_type(),
                        ),
                        location: location!(),
                    });
                }
            }
        } else {
            return Err(Error::IO {
                message: format!("Vector search error: column {} not found", q.column),
                location: location!(),
            });
        }

        let column_id = self.dataset.schema().field_id(q.column.as_str())?;
        let use_index = self.nearest.as_ref().map(|q| q.use_index).unwrap_or(false);
        let indices = if use_index {
            self.dataset.load_indices().await?
        } else {
            vec![]
        };
        let knn_idx = indices.iter().find(|i| i.fields.contains(&column_id));
        if let Some(index) = knn_idx {
            if prefilter.is_some() {
                return Err(Error::NotSupported { source: "A prefilter cannot currently be used with a vector index.  Set use_index to false or remove the prefilter.".into() });
            }
            // There is an index built for the column.
            // We will use the index.
            if let Some(rf) = q.refine_factor {
                if rf == 0 {
                    return Err(Error::IO {
                        message: "Refine factor can not be zero".to_string(),
                        location: location!(),
                    });
                }
            }

            let knn_node = self.ann(q, index)?; // _distance, _rowid
            let with_vector = self.dataset.schema().project(&[&q.column])?;
            let knn_node_with_vector = self.take(knn_node, &with_vector, self.batch_readahead)?;
            let mut knn_node = if q.refine_factor.is_some() {
                self.flat_knn(knn_node_with_vector, q)?
            } else {
                knn_node_with_vector
            }; // vector, _distance, _rowid

            knn_node = self.knn_combined(&q, index, knn_node).await?;

            Ok(knn_node)
        } else {
            let mut columns = vec![q.column.clone()];
            if let Some(prefilter) = prefilter.as_ref() {
                columns.extend(column_names_in_expr(prefilter.as_ref()));
            }
            // No index found. use flat search.
            let vector_scan_projection = Arc::new(self.dataset.schema().project(&columns).unwrap());
            let mut plan = self.scan(true, vector_scan_projection);
            if let Some(prefilter) = prefilter {
                plan = Arc::new(FilterExec::try_new(prefilter.clone(), plan)?);
            }
            Ok(self.flat_knn(plan, q)?)
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
        let unindexed_fragments = index.unindexed_fragments(self.dataset.as_ref()).await?;
        if !unindexed_fragments.is_empty() {
            let vector_scan_projection =
                Arc::new(self.dataset.schema().project(&[&q.column]).unwrap());
            let scan_node = self.scan_fragments(
                true,
                vector_scan_projection,
                Arc::new(unindexed_fragments),
                // We are re-ordering anyways, so no need to get data in data
                // in a deterministic order.
                false,
            );
            // first we do flat search on just the new data
            let topk_appended = self.flat_knn(scan_node, q)?;

            // To do a union, we need to make the schemas match. Right now
            // knn_node: _distance, _rowid, vector
            // topk_appended: vector, _rowid, _distance
            let new_schema = Schema::try_from(
                &topk_appended
                    .schema()
                    .project(&[2, 1, 0])?
                    .with_metadata(knn_node.schema().metadata.clone()),
            )?;
            let topk_appended = ProjectionExec::try_new(topk_appended, Arc::new(new_schema))?;
            assert_eq!(topk_appended.schema(), knn_node.schema());
            // union
            let unioned = UnionExec::new(vec![Arc::new(topk_appended), knn_node]);
            // Enforce only 1 partition.
            let unioned = RepartitionExec::try_new(
                Arc::new(unioned),
                datafusion::physical_plan::Partitioning::RoundRobinBatch(1),
            )?;
            // then we do a flat search on KNN(new data) + ANN(indexed data)
            return self.flat_knn(Arc::new(unioned), q);
        }

        Ok(knn_node)
    }

    /// Create an Execution plan with a scan node
    fn scan(&self, with_row_id: bool, projection: Arc<Schema>) -> Arc<dyn ExecutionPlan> {
        let fragments = if let Some(fragment) = self.fragments.as_ref() {
            Arc::new(fragment.clone())
        } else {
            self.dataset.fragments().clone()
        };
        self.scan_fragments(with_row_id, projection, fragments, self.ordered)
    }

    fn scan_fragments(
        &self,
        with_row_id: bool,
        projection: Arc<Schema>,
        fragments: Arc<Vec<Fragment>>,
        ordered: bool,
    ) -> Arc<dyn ExecutionPlan> {
        Arc::new(LanceScanExec::new(
            self.dataset.clone(),
            fragments,
            projection,
            self.batch_size,
            self.batch_readahead,
            self.fragment_readahead,
            with_row_id,
            ordered,
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
        batch_readahead: usize,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(TakeExec::try_new(
            self.dataset.clone(),
            input,
            Arc::new(projection.clone()),
            batch_readahead,
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

    pub async fn explain_plan(&self, verbose: bool) -> Result<String> {
        let plan = self.create_plan().await?;
        let display = DisplayableExecutionPlan::new(plan.as_ref());

        Ok(format!("{}", display.indent(verbose)))
    }
}

/// [`DatasetRecordBatchStream`] wraps the dataset into a [`RecordBatchStream`] for
/// consumption by the user.
///
#[pin_project::pin_project]
pub struct DatasetRecordBatchStream {
    #[pin]
    exec_node: SendableRecordBatchStream,
    span: Span,
}

impl DatasetRecordBatchStream {
    pub fn new(exec_node: SendableRecordBatchStream) -> Self {
        let span = info_span!("DatasetRecordBatchStream");
        Self { exec_node, span }
    }
}

impl RecordBatchStream for DatasetRecordBatchStream {
    fn schema(&self) -> SchemaRef {
        self.exec_node.schema()
    }
}

impl Stream for DatasetRecordBatchStream {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        let _guard = this.span.enter();
        match this.exec_node.poll_next_unpin(cx) {
            Poll::Ready(result) => Poll::Ready(result.map(|r| {
                r.map_err(|e| Error::IO {
                    message: e.to_string(),
                    location: location!(),
                })
            })),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl From<DatasetRecordBatchStream> for SendableRecordBatchStream {
    fn from(stream: DatasetRecordBatchStream) -> Self {
        stream.exec_node
    }
}

#[cfg(test)]
mod test {

    use std::collections::{BTreeSet, HashMap};
    use std::path::PathBuf;
    use std::vec;

    use arrow::array::as_primitive_array;
    use arrow::compute::concat_batches;
    use arrow::datatypes::Int32Type;
    use arrow_array::{
        ArrayRef, FixedSizeListArray, Int32Array, Int64Array, LargeStringArray,
        RecordBatchIterator, StringArray, StructArray,
    };
    use arrow_ord::sort::sort_to_indices;
    use arrow_schema::DataType;
    use arrow_select::take;
    use futures::TryStreamExt;
    use lance_testing::datagen::{BatchGenerator, IncrementingInt32};
    use tempfile::tempdir;

    use super::*;
    use crate::arrow::*;
    use crate::dataset::WriteMode;
    use crate::dataset::WriteParams;
    use crate::index::{
        DatasetIndexExt,
        {vector::VectorIndexParams, IndexType},
    };

    #[tokio::test]
    async fn test_batch_size() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, true),
            ArrowField::new("s", DataType::Utf8, true),
        ]));

        let batches: Vec<RecordBatch> = (0..5)
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
            .collect();

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let mut stream = dataset
            .scan()
            .batch_size(8)
            .try_into_stream()
            .await
            .unwrap();
        for expected_len in [8, 2, 8, 2, 8, 2, 8, 2, 8, 2] {
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

        let batches: Vec<RecordBatch> = vec![RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..100)),
                Arc::new(StringArray::from_iter_values(
                    (0..100).map(|v| format!("s-{}", v)),
                )),
            ],
        )
        .unwrap()];

        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        Dataset::write(batches, test_uri, None).await.unwrap();

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
        scanner.limit(Some(2), Some(19)).unwrap();
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
                    value_range.collect::<Vec<_>>(),
                ))];
                RecordBatch::try_new(schema.clone(), columns).unwrap()
            })
            .collect();
        let params = WriteParams {
            max_rows_per_group: 10,
            ..Default::default()
        };
        let reader =
            RecordBatchIterator::new(expected_batches.clone().into_iter().map(Ok), schema.clone());
        Dataset::write(reader, path, Some(params)).await.unwrap();
        expected_batches
    }

    async fn create_vector_dataset(path: &str, build_index: bool) -> Arc<Dataset> {
        // Make sure the schema has metadata so it tests all paths that re-construct the schema along the way
        let metadata: HashMap<String, String> = vec![("dataset".to_string(), "vector".to_string())]
            .into_iter()
            .collect();

        let schema = Arc::new(ArrowSchema::new_with_metadata(
            vec![
                ArrowField::new("i", DataType::Int32, true),
                ArrowField::new("s", DataType::Utf8, true),
                ArrowField::new(
                    "vec",
                    DataType::FixedSizeList(
                        Arc::new(ArrowField::new("item", DataType::Float32, true)),
                        32,
                    ),
                    true,
                ),
            ],
            metadata,
        ));

        let batches: Vec<RecordBatch> = (0..5)
            .map(|i| {
                let vector_values: Float32Array = (0..32 * 80).map(|v| v as f32).collect();
                let vectors = FixedSizeListArray::try_new_from_values(vector_values, 32).unwrap();
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
            .collect();

        let params = WriteParams {
            max_rows_per_group: 10,
            ..Default::default()
        };
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(reader, path, Some(params)).await.unwrap();

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
                            Arc::new(ArrowField::new("item", DataType::Float32, true)),
                            32,
                        ),
                        true,
                    ),
                    ArrowField::new("_distance", DataType::Float32, false),
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
    async fn test_knn_with_new_data() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_vector_dataset(test_uri, true).await;

        // Insert more data
        // (0, 0, ...), (1, 1, ...), (2, 2, ...)
        let vector_values: Float32Array =
            (0..10).flat_map(|i| [i as f32; 32].into_iter()).collect();
        let new_vectors = FixedSizeListArray::try_new_from_values(vector_values, 32).unwrap();
        let new_data: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from_iter_values(400..410)), // 5 * 80
            Arc::new(StringArray::from_iter_values(
                (400..410).map(|v| format!("s-{}", v)),
            )),
            Arc::new(new_vectors),
        ];
        let schema: Arc<ArrowSchema> = Arc::new(dataset.schema().try_into().unwrap());
        let new_data_reader = RecordBatchIterator::new(
            vec![RecordBatch::try_new(schema.clone(), new_data).unwrap()]
                .into_iter()
                .map(Ok),
            schema.clone(),
        );
        let dataset = Dataset::write(
            new_data_reader,
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Create a bunch of queries
        let key: Float32Array = [0f32; 32].into_iter().collect();
        // Set as larger than the number of new rows that aren't in the index to
        // force result sets to be combined between index and flat scan.
        let k = 20;

        #[derive(Debug)]
        struct TestCase {
            filter: Option<&'static str>,
            limit: Option<i64>,
            use_index: bool,
        }

        let mut cases = vec![];
        for filter in [Some("i > 100"), None] {
            for limit in [None, Some(10)] {
                for use_index in [true, false] {
                    cases.push(TestCase {
                        filter,
                        limit,
                        use_index,
                    });
                }
            }
        }

        // Validate them all.
        for case in cases {
            let mut scanner = dataset.scan();
            scanner
                .nearest("vec", &key, k)
                .unwrap()
                .limit(case.limit, None)
                .unwrap()
                .refine(3)
                .use_index(case.use_index);
            if let Some(filter) = case.filter {
                scanner.filter(filter).unwrap();
            }

            let result = scanner
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            assert!(!result.is_empty());
            let result = concat_batches(&result[0].schema(), result.iter()).unwrap();

            if case.filter.is_some() {
                let result_rows = result.num_rows();
                let expected_rows = case.limit.unwrap_or(k as i64) as usize;
                assert!(
                    result_rows <= expected_rows,
                    "Expected less than {} rows, got {}",
                    expected_rows,
                    result_rows
                );
            } else {
                // Exactly equal count
                assert_eq!(result.num_rows(), case.limit.unwrap_or(k as i64) as usize);
            }

            // Top one should be the first value of new data
            assert_eq!(
                as_primitive_array::<Int32Type>(result.column(0).as_ref()).value(0),
                400
            );
        }
    }

    #[tokio::test]
    async fn test_knn_with_prefilter() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_vector_dataset(test_uri, true).await;
        let mut scan = dataset.scan();
        let key: Float32Array = (32..64).map(|v| v as f32).collect();
        scan.filter("i > 100").unwrap();
        scan.prefilter(true);
        scan.project(&["i"]).unwrap();
        scan.nearest("vec", &key, 5).unwrap();

        // Prefilter with index not supported yet
        assert!(scan.try_into_stream().await.is_err());

        scan.use_index(false);

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
                ArrowField::new(
                    "vec",
                    DataType::FixedSizeList(
                        Arc::new(ArrowField::new("item", DataType::Float32, true)),
                        32,
                    ),
                    true,
                ),
                ArrowField::new("_distance", DataType::Float32, false),
            ])
        );

        // These match the query exactly.  The 5 results must include these 3.
        let exact_i = BTreeSet::from_iter(vec![161, 241, 321]);
        // These also include those 1 off from the query.  The remaining 2 results must be in this set.
        let close_i = BTreeSet::from_iter(vec![161, 241, 321, 160, 162, 240, 242, 320, 322]);
        let column_i = batch.column_by_name("i").unwrap();
        let actual_i: BTreeSet<i32> = as_primitive_array::<Int32Type>(column_i.as_ref())
            .values()
            .iter()
            .copied()
            .collect();
        assert!(exact_i.is_subset(&actual_i));
        assert!(actual_i.is_subset(&close_i));
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
                        Arc::new(ArrowField::new("item", DataType::Float32, true)),
                        32,
                    ),
                    true,
                ),
                ArrowField::new("_distance", DataType::Float32, false),
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
                        Arc::new(ArrowField::new("item", DataType::Float32, true)),
                        32,
                    ),
                    true,
                ),
                ArrowField::new("_distance", DataType::Float32, false),
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

    #[tokio::test]
    async fn test_scan_unordered_with_row_id() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = create_vector_dataset(test_uri, false).await;

        let mut scan = dataset.scan();
        scan.with_row_id();

        let ordered_batches = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();
        assert!(ordered_batches.len() > 2);
        let ordered_batch =
            concat_batches(&ordered_batches[0].schema(), ordered_batches.iter()).unwrap();

        // Attempt to get out-of-order scan, but that might take multiple attempts.
        scan.scan_in_order(false);
        for _ in 0..10 {
            let unordered_batches = scan
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<RecordBatch>>()
                .await
                .unwrap();
            let unordered_batch =
                concat_batches(&unordered_batches[0].schema(), unordered_batches.iter()).unwrap();

            assert_eq!(ordered_batch.num_rows(), unordered_batch.num_rows());

            // If they aren't equal, they should be equal if we sort by row id
            if ordered_batch != unordered_batch {
                let sort_indices = sort_to_indices(&unordered_batch["_rowid"], None, None).unwrap();

                let ordered_i = ordered_batch["i"].clone();
                let sorted_i = take::take(&unordered_batch["i"], &sort_indices, None).unwrap();

                assert_eq!(&ordered_i, &sorted_i);

                break;
            }
        }
    }

    #[tokio::test]
    async fn test_scan_order() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            true,
        )]));

        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]))],
        )
        .unwrap();

        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![6, 7, 8]))],
        )
        .unwrap();

        let params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };

        let write_batch = |batch: RecordBatch| async {
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
            Dataset::write(reader, test_uri, Some(params)).await
        };

        write_batch.clone()(batch1.clone()).await.unwrap();
        write_batch(batch2.clone()).await.unwrap();

        let dataset = Arc::new(Dataset::open(test_uri).await.unwrap());
        let fragment1 = dataset.get_fragment(0).unwrap().metadata().clone();
        let fragment2 = dataset.get_fragment(1).unwrap().metadata().clone();

        // 1 then 2
        let mut scanner = dataset.scan();
        scanner.with_fragments(vec![fragment1.clone(), fragment2.clone()]);
        let output = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], batch1);
        assert_eq!(output[1], batch2);

        // 2 then 1
        let mut scanner = dataset.scan();
        scanner.with_fragments(vec![fragment2, fragment1]);
        let output = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], batch2);
        assert_eq!(output[1], batch1);
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
    ///  KNNIndex(vec) -> Take(i) -> filter(i) -> take(s, vec) -> projection(s, vec, _distance)
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
            vec!["s", "vec", "_distance"]
        );

        let take = &plan.children()[0];
        let take = take.as_any().downcast_ref::<TakeExec>().unwrap();
        assert_eq!(
            take.schema().field_names(),
            ["_distance", "_rowid", "vec", "i", "s"]
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
            ["_distance", "_rowid", "vec", "i"]
        );

        let take = &filter.children()[0];
        let take = take.as_any().downcast_ref::<TakeExec>().unwrap();
        assert_eq!(
            take.schema().field_names(),
            ["_distance", "_rowid", "vec", "i"]
        );
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
        assert_eq!(take.schema().field_names(), ["_distance", "_rowid", "vec"]);
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
        assert_eq!(knn.schema().field_names(), ["_distance", "_rowid"]);
    }

    /// Test KNN index with refine factor
    ///
    /// Query: nearest(vec, [...], 10, refine_factor=10) + filter(i > 10 and i < 20)
    ///
    /// Expected plan:
    ///  KNNIndex(vec) -> Take(vec) -> KNNFlat(vec, 10) -> Take(i) -> Filter(i)
    ///     -> take(s, vec) -> projection(s, vec, _distance)
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
            vec!["s", "vec", "_distance"]
        );

        let take = &plan.children()[0];
        let take = take.as_any().downcast_ref::<TakeExec>().unwrap();
        assert_eq!(
            take.schema().field_names(),
            ["_distance", "_rowid", "vec", "i", "s"]
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
            ["_distance", "_rowid", "vec", "i"]
        );

        let take = &filter.children()[0];
        let take = take.as_any().downcast_ref::<TakeExec>().unwrap();
        assert_eq!(
            take.schema().field_names(),
            ["_distance", "_rowid", "vec", "i"]
        );
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
        assert_eq!(take.schema().field_names(), ["_distance", "_rowid", "vec"]);
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
        assert_eq!(knn.schema().field_names(), ["_distance", "_rowid"]);
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

        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(LargeStringArray::from_iter_values(
                (0..10).map(|v| format!("s-{}", v)),
            ))],
        )
        .unwrap()];

        let write_params = WriteParams::default();
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params))
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

    #[tokio::test]
    async fn test_filter_with_regex() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "ls",
            DataType::Utf8,
            true,
        )]));

        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from_iter_values(
                (0..20).map(|v| format!("s-{}", v)),
            ))],
        )
        .unwrap()];

        let write_params = WriteParams::default();
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let mut scan = dataset.scan();
        scan.filter("regexp_match(ls, 's-1.')").unwrap();

        let stream = scan.try_into_stream().await.unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        let batch = &batches[0];

        let expected = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from_iter_values(
                (10..=19).map(|v| format!("s-{}", v)),
            ))],
        )
        .unwrap();

        assert_eq!(batch, &expected);
    }

    #[tokio::test]
    async fn test_filter_proj_bug() {
        let struct_i_field = ArrowField::new("i", DataType::Int32, true);
        let struct_o_field = ArrowField::new("o", DataType::Utf8, true);
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new(
                "struct",
                DataType::Struct(vec![struct_i_field.clone(), struct_o_field.clone()].into()),
                true,
            ),
            ArrowField::new("s", DataType::Utf8, true),
        ]));

        let input_batches: Vec<RecordBatch> = (0..5)
            .map(|i| {
                let struct_i_arr: Arc<Int32Array> =
                    Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20));
                let struct_o_arr: Arc<StringArray> = Arc::new(StringArray::from_iter_values(
                    (i * 20..(i + 1) * 20).map(|v| format!("o-{:02}", v)),
                ));
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(StructArray::from(vec![
                            (Arc::new(struct_i_field.clone()), struct_i_arr as ArrayRef),
                            (Arc::new(struct_o_field.clone()), struct_o_arr as ArrayRef),
                        ])),
                        Arc::new(StringArray::from_iter_values(
                            (i * 20..(i + 1) * 20).map(|v| format!("s-{}", v)),
                        )),
                    ],
                )
                .unwrap()
            })
            .collect();
        let batches =
            RecordBatchIterator::new(input_batches.clone().into_iter().map(Ok), schema.clone());
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            ..Default::default()
        };
        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let batches = dataset
            .scan()
            .filter("struct.i >= 20")
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let batch = concat_batches(&batches[0].schema(), &batches).unwrap();

        let expected_batch = concat_batches(&schema, &input_batches.as_slice()[1..]).unwrap();
        assert_eq!(batch, expected_batch);

        // different order
        let batches = dataset
            .scan()
            .filter("struct.o >= 'o-20'")
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let batch = concat_batches(&batches[0].schema(), &batches).unwrap();
        assert_eq!(batch, expected_batch);

        // other reported bug with nested top level column access
        let batches = dataset
            .scan()
            .project(vec!["struct"].as_slice())
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        concat_batches(&batches[0].schema(), &batches).unwrap();
    }

    #[tokio::test]
    async fn test_ann_with_deletion() {
        let vec_params = vec![
            // TODO: re-enable diskann test when we can tune to get reproducible results.
            // VectorIndexParams::with_diskann_params(MetricType::L2, DiskANNParams::new(10, 1.5, 10)),
            VectorIndexParams::ivf_pq(4, 8, 2, false, MetricType::L2, 2),
        ];
        for params in vec_params {
            let test_dir = tempdir().unwrap();
            let test_uri = test_dir.path().to_str().unwrap();

            // make dataset
            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("i", DataType::Int32, true),
                ArrowField::new(
                    "vec",
                    DataType::FixedSizeList(
                        Arc::new(ArrowField::new("item", DataType::Float32, true)),
                        32,
                    ),
                    true,
                ),
            ]));

            // vectors are [1, 1, 1, ...] [2, 2, 2, ...]
            let vector_values: Float32Array =
                (0..32 * 512).map(|v| (v / 32) as f32 + 1.0).collect();
            let vectors = FixedSizeListArray::try_new_from_values(vector_values, 32).unwrap();

            let batches = vec![RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..512)),
                    Arc::new(vectors.clone()),
                ],
            )
            .unwrap()];

            let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
            let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

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

            let mut scan = dataset.scan();
            // closest be i = 0..5
            let key: Float32Array = (0..32).map(|_v| 1.0_f32).collect();
            scan.nearest("vec", &key, 5).unwrap();
            scan.refine(100);
            scan.nprobs(100);

            let results = scan
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();

            assert_eq!(results.len(), 1);
            let batch = &results[0];

            let expected_i = BTreeSet::from_iter(vec![0, 1, 2, 3, 4]);
            let column_i = batch.column_by_name("i").unwrap();
            let actual_i: BTreeSet<i32> = as_primitive_array::<Int32Type>(column_i.as_ref())
                .values()
                .iter()
                .copied()
                .collect();
            assert_eq!(expected_i, actual_i);

            // DELETE top result and search again

            dataset.delete("i = 1").await.unwrap();
            let mut scan = dataset.scan();
            scan.nearest("vec", &key, 5).unwrap();
            scan.refine(100);
            scan.nprobs(100);

            let results = scan
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();

            assert_eq!(results.len(), 1);
            let batch = &results[0];

            // i=1 was deleted, and 5 is the next best, the reset shouldn't change
            let expected_i = BTreeSet::from_iter(vec![0, 2, 3, 4, 5]);
            let column_i = batch.column_by_name("i").unwrap();
            let actual_i: BTreeSet<i32> = as_primitive_array::<Int32Type>(column_i.as_ref())
                .values()
                .iter()
                .copied()
                .collect();
            assert_eq!(expected_i, actual_i);

            // Add a second fragment and test the case where there are no deletion
            // files but there are missing fragments.
            let batches = vec![RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(512..1024)),
                    Arc::new(vectors),
                ],
            )
            .unwrap()];

            let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
            let mut dataset = Dataset::write(
                reader,
                test_uri,
                Some(WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
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

            dataset.delete("i < 512").await.unwrap();

            let mut scan = dataset.scan();
            scan.nearest("vec", &key, 5).unwrap();
            scan.refine(100);
            scan.nprobs(100);

            let results = scan
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();

            assert_eq!(results.len(), 1);
            let batch = &results[0];

            // It should not pick up any results from the first fragment
            let expected_i = BTreeSet::from_iter(vec![512, 513, 514, 515, 516]);
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
    async fn test_count_rows_with_filter() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut data_gen = BatchGenerator::new().col(Box::new(
            IncrementingInt32::new().named("Filter_me".to_owned()),
        ));
        Dataset::write(data_gen.batch(32), test_uri, None)
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        assert_eq!(32, dataset.scan().count_rows().await.unwrap());
        assert_eq!(
            16,
            dataset
                .scan()
                .filter("`Filter_me` > 15")
                .unwrap()
                .count_rows()
                .await
                .unwrap()
        );
    }
}
