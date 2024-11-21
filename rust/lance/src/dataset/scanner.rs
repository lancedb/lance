// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::{Array, Float32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema, SchemaRef, SortOptions};
use arrow_select::concat::concat_batches;
use async_recursion::async_recursion;
use datafusion::functions_aggregate::count::count_udaf;
use datafusion::logical_expr::{lit, Expr};
use datafusion::physical_expr::PhysicalSortExpr;
use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::expressions;
use datafusion::physical_plan::projection::ProjectionExec as DFProjectionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::{
    aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
    display::DisplayableExecutionPlan,
    expressions::Literal,
    filter::FilterExec,
    limit::GlobalLimitExec,
    repartition::RepartitionExec,
    udaf::create_aggregate_expr,
    union::UnionExec,
    ExecutionPlan, SendableRecordBatchStream,
};
use datafusion::scalar::ScalarValue;
use datafusion_physical_expr::{Partitioning, PhysicalExpr};
use futures::stream::{Stream, StreamExt};
use futures::TryStreamExt;
use lance_arrow::floats::{coerce_float_vector, FloatType};
use lance_arrow::DataTypeExt;
use lance_core::datatypes::Field;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{ROW_ADDR, ROW_ADDR_FIELD, ROW_ID, ROW_ID_FIELD};
use lance_datafusion::exec::{execute_plan, LanceExecutionOptions};
use lance_datafusion::projection::ProjectionPlan;
use lance_index::scalar::expression::PlannerIndexExt;
use lance_index::scalar::inverted::{FTS_SCHEMA, SCORE_COL};
use lance_index::scalar::{FullTextSearchQuery, ScalarIndexType};
use lance_index::vector::{Query, DIST_COL};
use lance_index::{scalar::expression::ScalarIndexExpr, DatasetIndexExt};
use lance_io::stream::RecordBatchStream;
use lance_linalg::distance::MetricType;
use lance_table::format::{Fragment, Index};
use roaring::RoaringBitmap;
use tracing::{info_span, instrument, Span};

use super::Dataset;
use crate::datatypes::Schema;
use crate::index::scalar::detect_scalar_index_type;
use crate::index::DatasetIndexInternalExt;
use crate::io::exec::fts::{FlatFtsExec, FtsExec};
use crate::io::exec::scalar_index::{MaterializeIndexExec, ScalarIndexExec};
use crate::io::exec::{get_physical_optimizer, LanceScanConfig};
use crate::io::exec::{
    knn::new_knn_exec, project, AddRowAddrExec, FilterPlan, KNNVectorDistanceExec,
    LancePushdownScanExec, LanceScanExec, Planner, PreFilterSource, ScanConfig, TakeExec,
};
use crate::{Error, Result};
use snafu::{location, Location};

#[cfg(feature = "substrait")]
use lance_datafusion::substrait::parse_substrait;

pub(crate) const BATCH_SIZE_FALLBACK: usize = 8192;
// For backwards compatibility / historical reasons we re-calculate the default batch size
// on each call
pub fn get_default_batch_size() -> Option<usize> {
    std::env::var("LANCE_DEFAULT_BATCH_SIZE")
        .map(|val| Some(val.parse().unwrap()))
        .unwrap_or(None)
}

pub const LEGACY_DEFAULT_FRAGMENT_READAHEAD: usize = 4;
lazy_static::lazy_static! {
    pub static ref DEFAULT_FRAGMENT_READAHEAD: Option<usize> = std::env::var("LANCE_DEFAULT_FRAGMENT_READAHEAD")
        .map(|val| Some(val.parse().unwrap())).unwrap_or(None);
}

// We want to support ~256 concurrent reads to maximize throughput on cloud storage systems
// Our typical page size is 8MiB (though not all reads are this large yet due to offset buffers, validity buffers, etc.)
// So we want to support 256 * 8MiB ~= 2GiB of queued reads
lazy_static::lazy_static! {
    pub static ref DEFAULT_IO_BUFFER_SIZE: u64 = std::env::var("LANCE_DEFAULT_IO_BUFFER_SIZE")
        .map(|val| val.parse().unwrap()).unwrap_or(2 * 1024 * 1024 * 1024);
}

/// Defines an ordering for a single column
///
/// Floats are sorted using the IEEE 754 total ordering
/// Strings are sorted using UTF-8 lexicographic order (i.e. we sort the binary)
pub struct ColumnOrdering {
    pub ascending: bool,
    pub nulls_first: bool,
    pub column_name: String,
}

impl ColumnOrdering {
    pub fn asc_nulls_first(column_name: String) -> Self {
        Self {
            ascending: true,
            nulls_first: true,
            column_name,
        }
    }

    pub fn asc_nulls_last(column_name: String) -> Self {
        Self {
            ascending: true,
            nulls_first: false,
            column_name,
        }
    }

    pub fn desc_nulls_first(column_name: String) -> Self {
        Self {
            ascending: false,
            nulls_first: true,
            column_name,
        }
    }

    pub fn desc_nulls_last(column_name: String) -> Self {
        Self {
            ascending: false,
            nulls_first: false,
            column_name,
        }
    }
}

/// Materialization style for the scanner
///
/// This only affects columns that are not used in a filter
///
/// Early materialization will fetch the entire column and throw
/// away the rows that are not needed.  This fetches more data but
/// uses fewer I/O requests.
///
/// Late materialization will only fetch the rows that are needed.
/// This fetches less data but uses more I/O requests.
///
/// This parameter only affects scans.  Vector search and full text search
/// always use late materialization.
pub enum MaterializationStyle {
    /// Heuristic-based materialization style
    ///
    /// The default approach depends on the type of object storage.  For
    /// cloud storage (e.g. S3, GCS, etc.) we only use late materialization
    /// for columns that are more than 1000 bytes in size.
    ///
    /// For local storage we use late materialization for columns that are
    /// more than 10 bytes in size.
    ///
    /// These values are based on experimentation and the assumption that a
    /// filter will be selecting ~0.1% of the rows in a column.
    Heuristic,
    /// All columns will be fetched with late materialization where possible
    AllLate,
    /// All columns will be fetched with early materialization where possible
    AllEarly,
    /// All columns will be fetched with late materialization except for the specified columns
    AllEarlyExcept(Vec<u32>),
}

impl MaterializationStyle {
    pub fn all_early_except(columns: &[impl AsRef<str>], schema: &Schema) -> Result<Self> {
        let field_ids = schema
            .project(columns)?
            .field_ids()
            .into_iter()
            .map(|id| id as u32)
            .collect();
        Ok(Self::AllEarlyExcept(field_ids))
    }
}

/// Filter for filtering rows
pub enum LanceFilter {
    /// The filter is an SQL string
    Sql(String),
    /// The filter is a Substrait expression
    Substrait(Vec<u8>),
    /// The filter is a Datafusion expression
    Datafusion(Expr),
}

impl LanceFilter {
    /// Converts the filter to a Datafusion expression
    ///
    /// The schema for this conversion should be the full schema available to
    /// the filter (`full_schema`).  However, due to a limitation in the way
    /// we do Substrait conversion today we can only do Substrait conversion with
    /// the dataset schema (`dataset_schema`).  This means that Substrait will
    /// not be able to access columns that are not in the dataset schema (e.g.
    /// _rowid, _rowaddr, etc.)
    #[allow(unused)]
    #[instrument(level = "trace", name = "filter_to_df", skip_all)]
    pub fn to_datafusion(&self, dataset_schema: &Schema, full_schema: &Schema) -> Result<Expr> {
        match self {
            Self::Sql(sql) => {
                let schema = Arc::new(ArrowSchema::from(full_schema));
                let planner = Planner::new(schema);
                let filter = planner.parse_filter(sql)?;
                planner.optimize_expr(filter)
            }
            #[cfg(feature = "substrait")]
            Self::Substrait(expr) => {
                use futures::FutureExt;

                let schema = Arc::new(ArrowSchema::from(dataset_schema));
                let expr = parse_substrait(expr, schema.clone())
                    .now_or_never()
                    .expect("could not parse the Substrait filter in a synchronous fashion")?;
                let planner = Planner::new(schema);
                planner.optimize_expr(expr)
            }
            #[cfg(not(feature = "substrait"))]
            Self::Substrait(_) => {
                panic!("Substrait filter is not supported in this build");
            }
            Self::Datafusion(expr) => Ok(expr.clone()),
        }
    }
}

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

    projection_plan: ProjectionPlan,

    /// If true then the filter will be applied before an index scan
    prefilter: bool,

    /// Materialization style controls when columns are fetched
    materialization_style: MaterializationStyle,

    /// Optional filter expression.
    filter: Option<LanceFilter>,

    /// Optional full text search query
    full_text_query: Option<FullTextSearchQuery>,

    /// The batch size controls the maximum size of rows to return for each read.
    batch_size: Option<usize>,

    /// Number of batches to prefetch
    batch_readahead: usize,

    /// Number of fragments to read concurrently
    fragment_readahead: Option<usize>,

    /// Number of bytes to allow to queue up in the I/O buffer
    io_buffer_size: Option<u64>,

    limit: Option<i64>,
    offset: Option<i64>,

    /// If Some then results will be ordered by the provided ordering
    ///
    /// If there are multiple columns the the results will first be ordered
    /// by the first column.  Then, any values whose first column is equal
    /// will be sorted by the next column, and so on.
    ///
    /// If this is Some then the value of `ordered` is ignored.  The scan
    /// will always be unordered since we are just going to reorder it anyways.
    ordering: Option<Vec<ColumnOrdering>>,

    nearest: Option<Query>,

    /// If false, do not use any scalar indices for the scan
    ///
    /// This can be used to pick a more efficient plan for certain queries where
    /// scalar indices do not work well (though we should also improve our planning
    /// to handle this better in the future as well)
    use_scalar_index: bool,

    /// Scan the dataset with a meta column: "_rowid"
    with_row_id: bool,

    /// Scan the dataset with a meta column: "_rowaddr"
    with_row_address: bool,

    /// Whether to use statistics to optimize the scan (default: true)
    ///
    /// This is used for debugging or benchmarking purposes.
    use_stats: bool,

    /// Whether to scan in deterministic order (default: true)
    ///
    /// This field is ignored if `ordering` is defined
    ordered: bool,

    /// If set, this scanner serves only these fragments.
    fragments: Option<Vec<Fragment>>,

    /// Only search the data being indexed (weak consistency search).
    ///
    /// Default value is false.
    ///
    /// This is essentially a weak consistency search. Users can run index or optimize index
    /// to make the index catch up with the latest data.
    fast_search: bool,
}

fn escape_column_name(name: &str) -> String {
    name.split('.')
        .map(|s| format!("`{}`", s))
        .collect::<Vec<_>>()
        .join(".")
}

impl Scanner {
    pub fn new(dataset: Arc<Dataset>) -> Self {
        // By default, we only scan the local schema
        let projection_plan = ProjectionPlan::new_empty(
            Arc::new(dataset.local_schema().clone()),
            /*load_blobs= */ false,
        );
        Self {
            dataset,
            projection_plan,
            prefilter: false,
            materialization_style: MaterializationStyle::Heuristic,
            filter: None,
            full_text_query: None,
            batch_size: None,
            batch_readahead: get_num_compute_intensive_cpus(),
            fragment_readahead: None,
            io_buffer_size: None,
            limit: None,
            offset: None,
            ordering: None,
            nearest: None,
            use_stats: true,
            with_row_id: false,
            with_row_address: false,
            ordered: true,
            fragments: None,
            fast_search: false,
            use_scalar_index: true,
        }
    }

    pub fn from_fragment(dataset: Arc<Dataset>, fragment: Fragment) -> Self {
        Self {
            fragments: Some(vec![fragment]),
            ..Self::new(dataset)
        }
    }

    /// Set which fragments should be scanned.
    ///
    /// If scan_in_order is set to true, the fragments will be scanned in the order of the vector.
    pub fn with_fragments(&mut self, fragments: Vec<Fragment>) -> &mut Self {
        self.fragments = Some(fragments);
        self
    }

    fn get_batch_size(&self) -> usize {
        // Default batch size to be large enough so that a i32 column can be
        // read in a single range request. For the object store default of
        // 64KB, this is 16K rows. For local file systems, the default block size
        // is just 4K, which would mean only 1K rows, which might be a little small.
        // So we use a default minimum of 8K rows.
        get_default_batch_size().unwrap_or_else(|| {
            self.batch_size.unwrap_or_else(|| {
                std::cmp::max(
                    self.dataset.object_store().block_size() / 4,
                    BATCH_SIZE_FALLBACK,
                )
            })
        })
    }

    fn ensure_not_fragment_scan(&self) -> Result<()> {
        if self.is_fragment_scan() {
            Err(Error::io(
                "This operation is not supported for fragment scan".to_string(),
                location!(),
            ))
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
        self.project_with_transform(
            &columns
                .iter()
                .map(|c| (c.as_ref(), escape_column_name(c.as_ref())))
                .collect::<Vec<_>>(),
        )
    }

    /// Projection with transform
    ///
    /// Only select the specified columns with the given transform.
    pub fn project_with_transform(
        &mut self,
        columns: &[(impl AsRef<str>, impl AsRef<str>)],
    ) -> Result<&mut Self> {
        let base_schema = self.scan_output_schema(self.dataset.schema(), true)?;
        self.projection_plan =
            ProjectionPlan::try_new(&base_schema, columns, /*load_blobs=*/ false)?;
        if self.projection_plan.sibling_schema.is_some() {
            return Err(Error::NotSupported {
                source: "Scanning columns with non-default storage class is not yet supported"
                    .into(),
                location: location!(),
            });
        }
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
    pub fn prefilter(&mut self, should_prefilter: bool) -> &mut Self {
        self.prefilter = should_prefilter;
        self
    }

    /// Set the materialization style for the scan
    ///
    /// This controls when columns are fetched from storage.  The default should work
    /// well for most cases.
    ///
    /// If you know (in advance) a query will return relatively few results (less than
    /// 0.1% of the rows) then you may want to experiment with applying late materialization
    /// to more (or all) columns.
    ///
    /// If you know a query is going to return many rows then you may want to experiment
    /// with applying early materialization to more (or all) columns.
    pub fn materialization_style(&mut self, style: MaterializationStyle) -> &mut Self {
        self.materialization_style = style;
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
        self.filter = Some(LanceFilter::Sql(filter.to_string()));
        Ok(self)
    }

    /// Filter by full text search
    /// The column must be a string column.
    /// The query is a string to search for.
    /// The search is case-insensitive, BM25 scoring is used.
    ///
    /// ```rust,ignore
    /// let dataset = Dataset::open(uri).await.unwrap();
    /// let stream = dataset.scan()
    ///    .project(&["col", "col2.subfield"]).unwrap()
    ///    .full_text_search("col", "query").unwrap()
    ///    .limit(10)
    ///    .into_stream();
    /// ```
    pub fn full_text_search(&mut self, query: FullTextSearchQuery) -> Result<&mut Self> {
        if !query.columns.is_empty() {
            for column in &query.columns {
                if self.dataset.schema().field(column).is_none() {
                    return Err(Error::invalid_input(
                        format!("Column {} not found", column),
                        location!(),
                    ));
                }
            }
        }

        self.full_text_query = Some(query);
        Ok(self)
    }

    /// Set a filter using a Substrait ExtendedExpression message
    ///
    /// The message must contain exactly one expression and that expression
    /// must be a scalar expression whose return type is boolean.
    pub fn filter_substrait(&mut self, filter: &[u8]) -> Result<&mut Self> {
        self.filter = Some(LanceFilter::Substrait(filter.to_vec()));
        Ok(self)
    }

    pub(crate) fn filter_expr(&mut self, filter: Expr) -> &mut Self {
        self.filter = Some(LanceFilter::Datafusion(filter));
        self
    }

    /// Set the batch size.
    pub fn batch_size(&mut self, batch_size: usize) -> &mut Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Set the I/O buffer size
    ///
    /// This is the amount of RAM that will be reserved for holding I/O received from
    /// storage before it is processed.  This is used to control the amount of memory
    /// used by the scanner.  If the buffer is full then the scanner will block until
    /// the buffer is processed.
    ///
    /// Generally this should scale with the number of concurrent I/O threads.  The
    /// default is 2GiB which comfortably provides enough space for somewhere between
    /// 32 and 256 concurrent I/O threads.
    ///
    /// This value is not a hard cap on the amount of RAM the scanner will use.  Some
    /// space is used for the compute (which can be controlled by the batch size) and
    /// Lance does not keep track of memory after it is returned to the user.
    ///
    /// Currently, if there is a single batch of data which is larger than the io buffer
    /// size then the scanner will deadlock.  This is a known issue and will be fixed in
    /// a future release.
    pub fn io_buffer_size(&mut self, size: u64) -> &mut Self {
        self.io_buffer_size = Some(size);
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
        self.fragment_readahead = Some(nfragments);
        self
    }

    /// Set whether to read data in order (default: true)
    ///
    /// A scan will always read from the disk concurrently.  If this property
    /// is true then a ready batch (a batch that has been read from disk) will
    /// only be returned if it is the next batch in the sequence.  Otherwise,
    /// the batch will be held until the stream catches up.  This means the
    /// sequence is returned in order but there may be slightly less parallelism.
    ///
    /// If this is false, then batches will be returned as soon as they are
    /// available, potentially increasing throughput slightly
    ///
    /// If an ordering is defined (using [Self::order_by]) then the scan will
    /// always scan in parallel and any value set here will be ignored.
    pub fn scan_in_order(&mut self, ordered: bool) -> &mut Self {
        self.ordered = ordered;
        self
    }

    /// Set whether to use scalar index.
    ///
    /// By default, scalar indices will be used to optimize a query if available.
    /// However, in some corner cases, scalar indices may not be the best choice.
    /// This option allows users to disable scalar indices for a query.
    pub fn use_scalar_index(&mut self, use_scalar_index: bool) -> &mut Self {
        self.use_scalar_index = use_scalar_index;
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
            return Err(Error::invalid_input(
                "Limit must be non-negative".to_string(),
                location!(),
            ));
        }
        if let Some(off) = offset {
            if off < 0 {
                return Err(Error::invalid_input(
                    "Offset must be non-negative".to_string(),
                    location!(),
                ));
            }
        }
        self.limit = limit;
        self.offset = offset;
        Ok(self)
    }

    /// Find k-nearest neighbor within the vector column.
    pub fn nearest(&mut self, column: &str, q: &Float32Array, k: usize) -> Result<&mut Self> {
        if !self.prefilter {
            // We can allow fragment scan if the input to nearest is a prefilter.
            // The fragment scan will be performed by the prefilter.
            self.ensure_not_fragment_scan()?;
        }

        if k == 0 {
            return Err(Error::invalid_input(
                "k must be positive".to_string(),
                location!(),
            ));
        }
        if q.is_empty() {
            return Err(Error::invalid_input(
                "Query vector must have non-zero length".to_string(),
                location!(),
            ));
        }
        // make sure the field exists
        let field = self
            .dataset
            .schema()
            .field(column)
            .ok_or(Error::invalid_input(
                format!("Column {} not found", column),
                location!(),
            ))?;
        let key = match field.data_type() {
            DataType::FixedSizeList(dt, _) => {
                if dt.data_type().is_floating() {
                    coerce_float_vector(q, FloatType::try_from(dt.data_type())?)?
                } else {
                    return Err(Error::invalid_input(
                        format!(
                            "Column {} is not a vector column (type: {})",
                            column,
                            field.data_type()
                        ),
                        location!(),
                    ));
                }
            }
            _ => {
                return Err(Error::invalid_input(
                    format!(
                        "Column {} is not a vector column (type: {})",
                        column,
                        field.data_type()
                    ),
                    location!(),
                ));
            }
        };

        self.nearest = Some(Query {
            column: column.to_string(),
            key: key.into(),
            k,
            nprobes: 1,
            ef: None,
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

    pub fn ef(&mut self, ef: usize) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.ef = Some(ef);
        }
        self
    }

    /// Only search the data being indexed.
    ///
    /// Default value is false.
    ///
    /// This is essentially a weak consistency search, only on the indexed data.
    pub fn fast_search(&mut self) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.use_index = true;
        }
        self.fast_search = true;
        self
    }

    /// Apply a refine step to the vector search.
    ///
    /// A refine improves query accuracy but also makes search slower, by reading extra elements
    /// and using the original vector values to re-rank the distances.
    ///
    /// * `factor` - the factor of extra elements to read.  For example, if factor is 2, then
    ///              the search will read 2x more elements than the requested k before performing
    ///              the re-ranking. Note: even if the factor is 1, the  results will still be
    ///              re-ranked without fetching additional elements.
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

    /// Sort the results of the scan by one or more columns
    ///
    /// If Some, then the resulting stream will be sorted according to the given ordering.
    /// This may increase the latency of the first result since all data must be read before
    /// the first batch can be returned.
    pub fn order_by(&mut self, ordering: Option<Vec<ColumnOrdering>>) -> Result<&mut Self> {
        if let Some(ordering) = &ordering {
            if ordering.is_empty() {
                self.ordering = None;
                return Ok(self);
            }
            // Verify early that the fields exist
            for column in ordering {
                self.dataset
                    .schema()
                    .field(&column.column_name)
                    .ok_or(Error::invalid_input(
                        format!("Column {} not found", &column.column_name),
                        location!(),
                    ))?;
            }
        }
        self.ordering = ordering;
        Ok(self)
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

    /// Instruct the scanner to return the `_rowaddr` meta column from the dataset.
    pub fn with_row_address(&mut self) -> &mut Self {
        self.with_row_address = true;
        self
    }

    /// Set whether to use statistics to optimize the scan (default: true)
    ///
    /// This is used for debugging or benchmarking purposes.
    pub fn use_stats(&mut self, use_stats: bool) -> &mut Self {
        self.use_stats = use_stats;
        self
    }

    /// The Arrow schema of the output, including projections and vector / _distance
    pub async fn schema(&self) -> Result<SchemaRef> {
        let plan = self.create_plan().await?;
        Ok(plan.schema())
    }

    /// Fetches the currently set filter
    ///
    /// Note that this forces the filter to be evaluated and the result will depend on
    /// the current state of the scanner (e.g. if with_row_id has been called then _rowid
    /// will be available for filtering but not otherwise) and so you may want to call this
    /// after setting all other options.
    pub fn get_filter(&self) -> Result<Option<Expr>> {
        if let Some(filter) = &self.filter {
            let filter_schema = self.scan_input_schema()?;
            Ok(Some(filter.to_datafusion(
                self.dataset.schema(),
                filter_schema.as_ref(),
            )?))
        } else {
            Ok(None)
        }
    }

    fn get_extra_columns(&self, force_row_id: bool) -> Vec<ArrowField> {
        let mut extra_columns = vec![];

        if self.nearest.as_ref().is_some() {
            extra_columns.push(ArrowField::new(DIST_COL, DataType::Float32, true));
        };

        if self.full_text_query.is_some() {
            extra_columns.push(ArrowField::new(SCORE_COL, DataType::Float32, true));
        }

        if self.with_row_id || force_row_id {
            extra_columns.push(ROW_ID_FIELD.clone());
        }

        if self.with_row_address {
            extra_columns.push(ROW_ADDR_FIELD.clone());
        }

        extra_columns
    }

    pub(crate) fn scan_input_schema(&self) -> Result<Arc<Schema>> {
        let extra_columns = self.get_extra_columns(false);

        if !extra_columns.is_empty() {
            let physical_schema = self
                .dataset
                .schema()
                .merge(&ArrowSchema::new(extra_columns))?;
            Ok(Arc::new(physical_schema))
        } else {
            Ok(Arc::new(self.dataset.schema().clone()))
        }
    }

    /// The output schema from the initial scan stage of a plan
    ///
    /// This includes columns that are added by the scan but don't exist in the dataset
    /// schema (e.g. _distance, _rowid, _rowaddr)
    pub(crate) fn scan_output_schema(
        &self,
        base_schema: &Schema,
        force_row_id: bool,
    ) -> Result<Arc<Schema>> {
        let extra_columns = self.get_extra_columns(force_row_id);

        let schema = if !extra_columns.is_empty() {
            base_schema.merge(&ArrowSchema::new(extra_columns))?
        } else {
            base_schema.clone()
        };

        // drop metadata
        // NOTE: this is the current behavior as we don't return metadata in queries
        // but do return metadata for regular scans
        // We should make this behavior consistent -- probably by not returning metadata always
        if self.nearest.is_some() {
            Ok(Arc::new(Schema {
                fields: schema.fields,
                metadata: HashMap::new(),
            }))
        } else {
            Ok(Arc::new(schema))
        }
    }

    pub(crate) fn output_expr(&self) -> Result<Vec<(Arc<dyn PhysicalExpr>, String)>> {
        // Append the extra columns
        let mut output_expr = self.projection_plan.to_physical_exprs()?;

        let physical_schema = ArrowSchema::from(
            self.scan_output_schema(&self.projection_plan.physical_schema, false)?
                .as_ref(),
        );

        // distance goes before the row_id column
        if self.nearest.is_some() && output_expr.iter().all(|(_, name)| name != DIST_COL) {
            let vector_expr = expressions::col(DIST_COL, &physical_schema)?;
            output_expr.push((vector_expr, DIST_COL.to_string()));
        }

        if self.full_text_query.is_some() && output_expr.iter().all(|(_, name)| name != SCORE_COL) {
            let score_expr = expressions::col(SCORE_COL, &physical_schema)?;
            output_expr.push((score_expr, SCORE_COL.to_string()));
        }

        if self.with_row_id && output_expr.iter().all(|(_, name)| name != ROW_ID) {
            let row_id_expr = expressions::col(ROW_ID, &physical_schema)?;
            output_expr.push((row_id_expr, ROW_ID.to_string()));
        }

        if self.with_row_address && output_expr.iter().all(|(_, name)| name != ROW_ADDR) {
            let row_addr_expr = expressions::col(ROW_ADDR, &physical_schema)?;
            output_expr.push((row_addr_expr, ROW_ADDR.to_string()));
        }

        Ok(output_expr)
    }

    /// Create a stream from the Scanner.
    #[instrument(skip_all)]
    pub async fn try_into_stream(&self) -> Result<DatasetRecordBatchStream> {
        let plan = self.create_plan().await?;
        Ok(DatasetRecordBatchStream::new(execute_plan(
            plan,
            LanceExecutionOptions::default(),
        )?))
    }

    pub(crate) async fn try_into_dfstream(
        &self,
        options: LanceExecutionOptions,
    ) -> Result<SendableRecordBatchStream> {
        let plan = self.create_plan().await?;
        execute_plan(plan, options)
    }

    pub async fn try_into_batch(&self) -> Result<RecordBatch> {
        let stream = self.try_into_stream().await?;
        let schema = stream.schema();
        let batches = stream.try_collect::<Vec<_>>().await?;
        Ok(concat_batches(&schema, &batches)?)
    }

    /// Scan and return the number of matching rows
    #[instrument(skip_all)]
    pub async fn count_rows(&self) -> Result<u64> {
        let plan = self.create_plan().await?;
        // Datafusion interprets COUNT(*) as COUNT(1)
        let one = Arc::new(Literal::new(ScalarValue::UInt8(Some(1))));
        let count_expr = create_aggregate_expr(
            &count_udaf(),
            &[one],
            &[lit(1)],
            &[],
            &[],
            &plan.schema(),
            None,
            false,
            false,
        )?;
        let plan_schema = plan.schema();
        let count_plan = Arc::new(AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(Vec::new()),
            vec![count_expr],
            vec![None],
            plan,
            plan_schema,
        )?);
        let mut stream = execute_plan(count_plan, LanceExecutionOptions::default())?;

        // A count plan will always return a single batch with a single row.
        if let Some(first_batch) = stream.next().await {
            let batch = first_batch?;
            let array = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or(Error::io(
                    "Count plan did not return a UInt64Array".to_string(),
                    location!(),
                ))?;
            Ok(array.value(0) as u64)
        } else {
            Ok(0)
        }
    }

    /// Given a base schema and a list of desired fields figure out which fields, if any, still need loaded
    fn calc_new_fields<S: AsRef<str>>(
        &self,
        base_schema: &Schema,
        columns: &[S],
    ) -> Result<Option<Schema>> {
        let new_schema = self.dataset.schema().project(columns)?;
        let new_schema = new_schema.exclude(base_schema)?;
        if new_schema.fields.is_empty() {
            Ok(None)
        } else {
            Ok(Some(new_schema))
        }
    }

    // A "narrow" field is a field that is so small that we are better off reading the
    // entire column and filtering in memory rather than "take"ing the column.
    //
    // The exact threshold depends on a two factors:
    // 1. The number of rows returned by the filter
    // 2. The number of rows in the dataset
    // 3. The IOPS/bandwidth ratio of the storage system
    // 4. The size of each value in the column
    //
    // We don't (today) have a good way of knowing #1 or #4.  #2 is easy to know.  We can
    // combine 1 & 2 into "percentage of rows returned" but since we don't know #1 it
    // doesn't really help.  #3 is complex but as a rule of thumb we can use:
    //
    //   Local storage: 1 IOP for ever ten thousand bytes
    //   Cloud storage: 1 IOP for every million bytes
    //
    // Our current heuristic today is to assume a filter will return 0.1% of the rows in the dataset.
    //
    // This means, for cloud storage, a field is "narrow" if there are 1KB of data per row and
    // for local disk a field is "narrow" if there are 10 bytes of data per row.
    fn is_early_field(&self, field: &Field) -> bool {
        match self.materialization_style {
            MaterializationStyle::AllEarly => true,
            MaterializationStyle::AllLate => false,
            MaterializationStyle::AllEarlyExcept(ref cols) => !cols.contains(&(field.id as u32)),
            MaterializationStyle::Heuristic => {
                let byte_width = field.data_type().byte_width_opt();
                let is_cloud = self.dataset.object_store().is_cloud();
                if is_cloud {
                    byte_width.map_or(false, |bw| bw < 1000)
                } else {
                    byte_width.map_or(false, |bw| bw < 10)
                }
            }
        }
    }

    fn calc_eager_columns(&self, filter_plan: &FilterPlan) -> Result<Arc<Schema>> {
        let columns = filter_plan.refine_columns();
        // If the column didn't exist in the scan output schema then we wouldn't make
        // it to this point.  However, there may be columns (like _rowid, _distance, etc.)
        // which do not exist in the dataset schema but are added by the scan.  We can ignore
        // those as eager columns.
        let filter_schema = self.dataset.schema().project_or_drop(&columns)?;
        if filter_schema.fields.iter().any(|f| !f.is_default_storage()) {
            return Err(Error::NotSupported {
                source: "non-default storage columns cannot be used as filters".into(),
                location: location!(),
            });
        }
        let physical_schema = self.projection_plan.physical_schema.clone();
        let remaining_schema = physical_schema.exclude(&filter_schema)?;

        let narrow_fields = remaining_schema
            .fields
            .iter()
            .filter(|f| self.is_early_field(f))
            .cloned()
            .collect::<Vec<_>>();

        if narrow_fields.is_empty() {
            Ok(Arc::new(filter_schema))
        } else {
            let mut new_fields = filter_schema.fields;
            new_fields.extend(narrow_fields);
            Ok(Arc::new(Schema {
                fields: new_fields,
                metadata: HashMap::new(),
            }))
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
    /// In general, a plan has 5 stages:
    ///
    /// 1. Source (from dataset Scan or from index, may include prefilter)
    /// 2. Filter
    /// 3. Sort
    /// 4. Limit / Offset
    /// 5. Take remaining columns / Projection
    #[instrument(level = "debug", skip_all)]
    pub async fn create_plan(&self) -> Result<Arc<dyn ExecutionPlan>> {
        if self.projection_plan.physical_schema.fields.is_empty()
            && !self.with_row_id
            && !self.with_row_address
        {
            return Err(Error::InvalidInput {
                source:
                    "no columns were selected and with_row_id is false, there is nothing to scan"
                        .into(),
                location: location!(),
            });
        }
        if let Some(first_blob_col) = self
            .projection_plan
            .physical_schema
            .fields
            .iter()
            .find(|f| !f.is_default_storage())
        {
            return Err(Error::NotSupported {
                source: format!(
                    "Scanning blob columns such as \"{}\" is not yet supported",
                    first_blob_col.name
                )
                .into(),
                location: location!(),
            });
        }

        // Scalar indices are only used when prefiltering
        let use_scalar_index = self.use_scalar_index && (self.prefilter || self.nearest.is_none());

        let filter_schema = self.scan_input_schema()?;
        let planner = Planner::new(Arc::new(filter_schema.as_ref().into()));

        let mut filter_plan = if let Some(filter) = self.filter.as_ref() {
            let filter = filter.to_datafusion(self.dataset.schema(), filter_schema.as_ref())?;
            let index_info = self.dataset.scalar_index_info().await?;
            let filter_plan =
                planner.create_filter_plan(filter.clone(), &index_info, use_scalar_index)?;

            // This tests if any of the fragments are missing the physical_rows property (old style)
            // If they are then we cannot use scalar indices
            if filter_plan.index_query.is_some() {
                let fragments = if let Some(fragments) = self.fragments.as_ref() {
                    fragments
                } else {
                    self.dataset.fragments()
                };
                let mut has_missing_row_count = false;
                for frag in fragments {
                    if frag.physical_rows.is_none() {
                        has_missing_row_count = true;
                        break;
                    }
                }
                if has_missing_row_count {
                    // We need row counts to use scalar indices.  If we don't have them then
                    // fallback to a non-indexed filter
                    planner.create_filter_plan(filter.clone(), &index_info, false)?
                } else {
                    filter_plan
                }
            } else {
                filter_plan
            }
        } else {
            FilterPlan::default()
        };

        let scan_range = if filter_plan.has_any_filter() {
            // If there is a filter we can't pushdown limit / offset
            None
        } else {
            match (self.limit, self.offset) {
                (None, None) => None,
                (Some(limit), None) => Some(0..limit as u64),
                (None, Some(offset)) => {
                    let num_rows = self.dataset.count_all_rows().await?;
                    Some(offset as u64..num_rows as u64)
                }
                (Some(limit), Some(offset)) => Some(offset as u64..(offset + limit) as u64),
            }
        };
        let mut use_limit_node = true;

        // Stage 1: source (either an (K|A)NN search, full text search or or a (full|indexed) scan)
        let mut plan: Arc<dyn ExecutionPlan> = match (&self.nearest, &self.full_text_query) {
            (Some(_), None) => {
                // The source is an nearest neighbor search
                if self.prefilter {
                    // If we are prefiltering then the knn node will take care of the filter
                    let source = self.knn(&filter_plan).await?;
                    filter_plan = FilterPlan::default();
                    source
                } else {
                    // If we are postfiltering then we can't use scalar indices for the filter
                    // and will need to run the postfilter in memory
                    filter_plan.make_refine_only();
                    self.knn(&FilterPlan::default()).await?
                }
            }
            (None, Some(query)) => {
                // The source is an FTS search
                if self.prefilter {
                    // If we are prefiltering then the fts node will take care of the filter
                    let source = self.fts(&filter_plan, query).await?;
                    filter_plan = FilterPlan::default();
                    source
                } else {
                    // If we are postfiltering then we can't use scalar indices for the filter
                    // and will need to run the postfilter in memory
                    filter_plan.make_refine_only();
                    self.fts(&FilterPlan::default(), query).await?
                }
            }
            (None, None) => {
                let fragments = if let Some(fragments) = self.fragments.as_ref() {
                    fragments
                } else {
                    self.dataset.fragments()
                };
                // Avoid pushdown scan node if using v2 files
                let use_stats = if fragments.iter().any(|f| !f.has_legacy_files()) {
                    false
                } else {
                    self.use_stats
                };
                match (&filter_plan.index_query, &mut filter_plan.refine_expr) {
                    (Some(index_query), None) => {
                        self.scalar_indexed_scan(
                            self.projection_plan.physical_schema.as_ref(),
                            index_query,
                        )
                        .await?
                    }
                    // TODO: support combined pushdown and scalar index scan
                    (Some(index_query), Some(_)) => {
                        // If there is a filter then just load the eager columns and
                        // "take" the other columns later.
                        let eager_schema = self.calc_eager_columns(&filter_plan)?;
                        self.scalar_indexed_scan(&eager_schema, index_query).await?
                    }
                    (None, Some(_)) if use_stats && self.batch_size.is_none() => {
                        self.pushdown_scan(false, filter_plan.refine_expr.take().unwrap())?
                    }
                    (None, _) => {
                        // The source is a full scan of the table
                        let with_row_id = filter_plan.has_refine() || self.with_row_id;
                        let eager_schema = if filter_plan.has_refine() {
                            // If there is a filter then only load the filter columns in the
                            // initial scan.  We will `take` the remaining columns later
                            self.calc_eager_columns(&filter_plan)?
                        } else {
                            // If there is no filter we eagerly load everything
                            self.projection_plan.physical_schema.clone()
                        };
                        if scan_range.is_some() && !self.dataset.is_legacy_storage() {
                            // If this is a v2 dataset with no filter then we can pushdown
                            // limit/offset (via scan_range and we zero out limit/offset
                            // so we don't apply it twice)
                            use_limit_node = false;
                        }
                        self.scan(
                            with_row_id,
                            self.with_row_address,
                            false,
                            scan_range,
                            eager_schema,
                        )
                    }
                }
            }
            _ => {
                return Err(Error::InvalidInput {
                    source: "Cannot have both nearest and full text search".into(),
                    location: location!(),
                })
            }
        };

        // Stage 1.5 load columns needed for stages 2 & 3
        let mut additional_schema = None;
        // We may need to take filter columns if we are going to refine
        // an indexed scan.  Otherwise, the filter was applied during the scan
        // and this should be false
        if filter_plan.has_refine() {
            let eager_schema = self.calc_eager_columns(&filter_plan)?;
            let base_schema = Schema::try_from(plan.schema().as_ref())?;
            let still_to_load = eager_schema.exclude(base_schema)?;
            if still_to_load.fields.is_empty() {
                additional_schema = None;
            } else {
                additional_schema = Some(still_to_load);
            }
        }
        if let Some(ordering) = &self.ordering {
            additional_schema = self.calc_new_fields(
                &additional_schema
                    .map(Ok::<Schema, Error>)
                    .unwrap_or_else(|| Schema::try_from(plan.schema().as_ref()))?,
                &ordering
                    .iter()
                    .map(|col| &col.column_name)
                    .collect::<Vec<_>>(),
            )?;
        }
        if let Some(additional_schema) = additional_schema {
            plan = self.take(plan, &additional_schema, self.batch_readahead)?;
        }

        // Stage 2: filter
        if let Some(refine_expr) = filter_plan.refine_expr {
            // We create a new planner specific to the node's schema, since
            // physical expressions reference column by index rather than by name.
            let planner = Planner::new(plan.schema());
            let physical_refine_expr = planner.create_physical_expr(&refine_expr)?;

            plan = Arc::new(FilterExec::try_new(physical_refine_expr, plan)?);
        }

        // Stage 3: sort
        if let Some(ordering) = &self.ordering {
            let order_by_schema = Arc::new(
                self.dataset.schema().project(
                    &ordering
                        .iter()
                        .map(|col| &col.column_name)
                        .collect::<Vec<_>>(),
                )?,
            );
            let remaining_schema = order_by_schema.exclude(plan.schema().as_ref())?;
            if !remaining_schema.fields.is_empty() {
                // We haven't loaded the sort column yet so take it now
                plan = self.take(plan, &remaining_schema, self.batch_readahead)?;
            }
            let col_exprs = ordering
                .iter()
                .map(|col| {
                    Ok(PhysicalSortExpr {
                        expr: expressions::col(&col.column_name, plan.schema().as_ref())?,
                        options: SortOptions {
                            descending: !col.ascending,
                            nulls_first: col.nulls_first,
                        },
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            plan = Arc::new(SortExec::new(col_exprs, plan));
        }

        // Stage 4: limit / offset
        if use_limit_node && (self.limit.unwrap_or(0) > 0 || self.offset.is_some()) {
            plan = self.limit_node(plan);
        }

        // Stage 5: take remaining columns required for projection
        let physical_schema =
            self.scan_output_schema(&self.projection_plan.physical_schema, false)?;
        let remaining_schema = physical_schema.exclude(plan.schema().as_ref())?;
        if !remaining_schema.fields.is_empty() {
            plan = self.take(plan, &remaining_schema, self.batch_readahead)?;
        }
        // Stage 6: physical projection -- reorder physical columns needed before final projection
        let output_arrow_schema = physical_schema.as_ref().into();
        if plan.schema().as_ref() != &output_arrow_schema {
            plan = Arc::new(project(plan, &physical_schema.as_ref().into())?);
        }

        // Stage 7: final projection
        plan = Arc::new(DFProjectionExec::try_new(self.output_expr()?, plan)?);

        let optimizer = get_physical_optimizer();
        let options = Default::default();
        for rule in optimizer.rules {
            plan = rule.optimize(plan, &options)?;
        }

        Ok(plan)
    }

    // Create an execution plan to do full text search
    async fn fts(
        &self,
        filter_plan: &FilterPlan,
        query: &FullTextSearchQuery,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let columns = if query.columns.is_empty() {
            let string_columns = self.dataset.schema().fields.iter().filter_map(|f| {
                if f.data_type() == DataType::Utf8 || f.data_type() == DataType::LargeUtf8 {
                    Some(&f.name)
                } else {
                    None
                }
            });

            let mut indexed_columns = Vec::new();
            for column in string_columns {
                let index = self.dataset.load_scalar_index_for_column(column).await?;
                if let Some(index) = index {
                    let index_type = detect_scalar_index_type(
                        &self.dataset,
                        &index,
                        column,
                        &self.dataset.session,
                    )
                    .await?;
                    if matches!(index_type, ScalarIndexType::Inverted) {
                        indexed_columns.push(column.clone());
                    }
                }
            }

            indexed_columns
        } else {
            query.columns.clone()
        };

        if columns.is_empty() {
            return Err(Error::invalid_input(
                "Cannot perform full text search unless an INVERTED index has been created on at least one column".to_string(),
                location!(),
            ));
        }

        // rewrite the query to be with the columns and limit
        let query = query
            .clone()
            .columns(Some(columns.clone()))
            .limit(self.limit);

        // load indices
        let mut column_inputs = HashMap::with_capacity(columns.len());
        for column in columns {
            let index = self
                .dataset
                .load_scalar_index_for_column(&column)
                .await?
                .ok_or(Error::invalid_input(
                    format!("Column {} has no inverted index", column),
                    location!(),
                ))?;
            let index_uuids: Vec<_> = self
                .dataset
                .load_indices_by_name(&index.name)
                .await?
                .into_iter()
                .collect();

            let unindexed_fragments = self.dataset.unindexed_fragments(&index.name).await?;
            let unindexed_scan_node = if unindexed_fragments.is_empty() {
                Arc::new(EmptyExec::new(FTS_SCHEMA.clone()))
            } else {
                let mut columns = vec![column.clone()];
                if let Some(expr) = filter_plan.full_expr.as_ref() {
                    let filter_columns = Planner::column_names_in_expr(expr);
                    columns.extend(filter_columns);
                }
                let flat_fts_scan_schema =
                    Arc::new(self.dataset.schema().project(&columns).unwrap());
                let mut scan_node = self.scan_fragments(
                    true,
                    false,
                    true,
                    flat_fts_scan_schema,
                    Arc::new(unindexed_fragments),
                    None,
                    false,
                );

                if let Some(expr) = filter_plan.full_expr.as_ref() {
                    // If there is a prefilter we need to manually apply it to the new data
                    let planner = Planner::new(scan_node.schema());
                    let physical_refine_expr = planner.create_physical_expr(expr)?;
                    scan_node = Arc::new(FilterExec::try_new(physical_refine_expr, scan_node)?);
                }

                scan_node
            };

            column_inputs.insert(column.clone(), (index_uuids, unindexed_scan_node));
        }

        let indices = column_inputs
            .iter()
            .map(|(col, (idx, _))| (col.clone(), idx.clone()))
            .collect();
        let prefilter_source = self.prefilter_source(filter_plan).await?;
        let fts_plan = Arc::new(FtsExec::new(
            self.dataset.clone(),
            indices,
            query.clone(),
            prefilter_source,
        )) as Arc<dyn ExecutionPlan>;
        let flat_fts_plan = Arc::new(FlatFtsExec::new(self.dataset.clone(), column_inputs, query));
        let fts_node = Arc::new(UnionExec::new(vec![fts_plan, flat_fts_plan]));
        let fts_node = Arc::new(RepartitionExec::try_new(
            fts_node,
            Partitioning::RoundRobinBatch(1),
        )?);
        let sort_expr = PhysicalSortExpr {
            expr: expressions::col(SCORE_COL, fts_node.schema().as_ref())?,
            options: SortOptions {
                descending: true,
                nulls_first: false,
            },
        };

        Ok(Arc::new(
            SortExec::new(vec![sort_expr], fts_node).with_fetch(self.limit.map(|l| l as usize)),
        ))
    }

    // ANN/KNN search execution node with optional prefilter
    async fn knn(&self, filter_plan: &FilterPlan) -> Result<Arc<dyn ExecutionPlan>> {
        let Some(q) = self.nearest.as_ref() else {
            return Err(Error::invalid_input(
                "No nearest query".to_string(),
                location!(),
            ));
        };

        // Sanity check
        let schema = self.dataset.schema();
        if let Some(field) = schema.field(&q.column) {
            match field.data_type() {
                DataType::FixedSizeList(subfield, _) if subfield.data_type().is_floating() => {}
                _ => {
                    return Err(Error::invalid_input(
                        format!(
                            "Vector search error: column {} is not a vector type: expected FixedSizeList<Float32>, got {}",
                            q.column, field.data_type(),
                        ),
                        location!(),
                    ));
                }
            }
        } else {
            return Err(Error::invalid_input(
                format!("Vector search error: column {} not found", q.column),
                location!(),
            ));
        }

        let column_id = self.dataset.schema().field_id(q.column.as_str())?;
        let use_index = self.nearest.as_ref().map(|q| q.use_index).unwrap_or(false);
        let indices = if use_index {
            self.dataset.load_indices().await?
        } else {
            Arc::new(vec![])
        };
        if let Some(index) = indices.iter().find(|i| i.fields.contains(&column_id)) {
            // There is an index built for the column.
            // We will use the index.
            if matches!(q.refine_factor, Some(0)) {
                return Err(Error::invalid_input(
                    "Refine factor can not be zero".to_string(),
                    location!(),
                ));
            }

            // Find all deltas with the same index name.
            let deltas = self.dataset.load_indices_by_name(&index.name).await?;
            let ann_node = self.ann(q, &deltas, filter_plan).await?; // _distance, _rowid

            let mut knn_node = if q.refine_factor.is_some() {
                let with_vector = self.dataset.schema().project(&[&q.column])?;
                let knn_node_with_vector =
                    self.take(ann_node, &with_vector, self.batch_readahead)?;
                // TODO: now we just open an index to get its metric type.
                let idx = self
                    .dataset
                    .open_vector_index(q.column.as_str(), &index.uuid.to_string())
                    .await?;
                let mut q = q.clone();
                q.metric_type = idx.metric_type();
                self.flat_knn(knn_node_with_vector, &q)?
            } else {
                ann_node
            }; // vector, _distance, _rowid

            if !self.fast_search {
                knn_node = self.knn_combined(q, index, knn_node, filter_plan).await?;
            }

            Ok(knn_node)
        } else {
            // No index found. use flat search.
            let mut columns = vec![q.column.clone()];
            if let Some(refine_expr) = filter_plan.refine_expr.as_ref() {
                columns.extend(Planner::column_names_in_expr(refine_expr));
            }
            let vector_scan_projection = Arc::new(self.dataset.schema().project(&columns).unwrap());
            let mut plan = if let Some(index_query) = &filter_plan.index_query {
                self.scalar_indexed_scan(&vector_scan_projection, index_query)
                    .await?
            } else {
                self.scan(true, false, true, None, vector_scan_projection)
            };
            if let Some(refine_expr) = &filter_plan.refine_expr {
                let planner = Planner::new(plan.schema());
                let physical_refine_expr = planner.create_physical_expr(refine_expr)?;

                plan = Arc::new(FilterExec::try_new(physical_refine_expr, plan)?);
            }
            Ok(self.flat_knn(plan, q)?)
        }
    }

    /// Combine ANN results with KNN results for data appended after index creation
    async fn knn_combined(
        &self,
        q: &Query,
        index: &Index,
        mut knn_node: Arc<dyn ExecutionPlan>,
        filter_plan: &FilterPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Check if we've created new versions since the index was built.
        let unindexed_fragments = self.dataset.unindexed_fragments(&index.name).await?;
        if !unindexed_fragments.is_empty() {
            // If the vector column is not present, we need to take the vector column, so
            // that the distance value is comparable with the flat search ones.
            if knn_node.schema().column_with_name(&q.column).is_none() {
                let with_vector = self.dataset.schema().project(&[&q.column])?;
                knn_node = self.take(knn_node, &with_vector, self.batch_readahead)?;
            }

            let mut columns = vec![q.column.clone()];
            if let Some(expr) = filter_plan.full_expr.as_ref() {
                let filter_columns = Planner::column_names_in_expr(expr);
                columns.extend(filter_columns);
            }
            let vector_scan_projection = Arc::new(self.dataset.schema().project(&columns).unwrap());
            // Note: we could try and use the scalar indices here to reduce the scope of this scan but the
            // most common case is that fragments that are newer than the vector index are going to be newer
            // than the scalar indices anyways
            let mut scan_node = self.scan_fragments(
                true,
                false,
                true,
                vector_scan_projection,
                Arc::new(unindexed_fragments),
                // Can't pushdown limit/offset in an ANN search
                None,
                // We are re-ordering anyways, so no need to get data in data
                // in a deterministic order.
                false,
            );

            if let Some(expr) = filter_plan.full_expr.as_ref() {
                // If there is a prefilter we need to manually apply it to the new data
                let planner = Planner::new(scan_node.schema());
                let physical_refine_expr = planner.create_physical_expr(expr)?;
                scan_node = Arc::new(FilterExec::try_new(physical_refine_expr, scan_node)?);
            }
            // first we do flat search on just the new data
            let topk_appended = self.flat_knn(scan_node, q)?;

            // To do a union, we need to make the schemas match. Right now
            // knn_node: _distance, _rowid, vector
            // topk_appended: vector, <filter columns?>, _rowid, _distance
            let topk_appended = project(topk_appended, knn_node.schema().as_ref())?;
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

    #[async_recursion]
    async fn fragments_covered_by_index_query(
        &self,
        index_expr: &ScalarIndexExpr,
    ) -> Result<RoaringBitmap> {
        match index_expr {
            ScalarIndexExpr::And(lhs, rhs) => {
                Ok(self.fragments_covered_by_index_query(lhs).await?
                    & self.fragments_covered_by_index_query(rhs).await?)
            }
            ScalarIndexExpr::Or(lhs, rhs) => Ok(self.fragments_covered_by_index_query(lhs).await?
                & self.fragments_covered_by_index_query(rhs).await?),
            ScalarIndexExpr::Not(expr) => self.fragments_covered_by_index_query(expr).await,
            ScalarIndexExpr::Query(column, _) => {
                let idx = self
                    .dataset
                    .load_scalar_index_for_column(column)
                    .await?
                    .expect("Index not found even though it must have been found earlier");
                Ok(idx
                    .fragment_bitmap
                    .expect("scalar indices should always have a fragment bitmap"))
            }
        }
    }

    // First perform a lookup in a scalar index for ids and then perform a take on the
    // target fragments with those ids
    async fn scalar_indexed_scan(
        &self,
        projection: &Schema,
        index_expr: &ScalarIndexExpr,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // One or more scalar indices cover this data and there is a filter which is
        // compatible with the indices.  Use that filter to perform a take instead of
        // a full scan.
        let fragments = if let Some(fragment) = self.fragments.as_ref() {
            fragment.clone()
        } else {
            (**self.dataset.fragments()).clone()
        };

        // Figure out which fragments are covered by ALL of the indices we are using
        let covered_frags = self.fragments_covered_by_index_query(index_expr).await?;
        let mut relevant_frags = Vec::with_capacity(fragments.len());
        let mut missing_frags = Vec::with_capacity(fragments.len());
        for fragment in fragments {
            if covered_frags.contains(fragment.id as u32) {
                relevant_frags.push(fragment);
            } else {
                missing_frags.push(fragment);
            }
        }

        let mut plan: Arc<dyn ExecutionPlan> = Arc::new(MaterializeIndexExec::new(
            self.dataset.clone(),
            index_expr.clone(),
            Arc::new(relevant_frags),
        ));

        // If there is more than just _rowid in projection
        let needs_take = match projection.fields.len() {
            0 => false,
            1 => projection.fields[0].name != ROW_ID,
            _ => true,
        };
        if needs_take {
            plan = self.take(plan, projection, self.batch_readahead)?;
        }

        if self.with_row_address {
            plan = Arc::new(AddRowAddrExec::try_new(plan, self.dataset.clone(), 0)?);
        }

        let new_data_path: Option<Arc<dyn ExecutionPlan>> = if !missing_frags.is_empty() {
            // If there is new data then we need this:
            //
            // MaterializeIndexExec(old_frags) -> Take -> Union
            // Scan(new_frags) -> Filter -> Project    -|
            //
            // The project is to drop any columns we had to include
            // in the full scan merely for the sake of fulfilling the
            // filter.
            //
            // If there were no extra columns then we still need the project
            // because Materialize -> Take puts the row id at the left and
            // Scan puts the row id at the right
            let filter_expr = index_expr.to_expr();
            let filter_cols = Planner::column_names_in_expr(&filter_expr);
            let full_schema = self
                .calc_new_fields(projection, &filter_cols)?
                .map(|filter_only_schema| projection.merge(&filter_only_schema))
                .transpose()?;
            let schema = full_schema.as_ref().unwrap_or(projection);

            let planner = Planner::new(Arc::new(schema.into()));
            let optimized_filter = planner.optimize_expr(filter_expr)?;
            let physical_refine_expr = planner.create_physical_expr(&optimized_filter)?;

            let new_data_scan = self.scan_fragments(
                true,
                self.with_row_address,
                false,
                Arc::new(schema.clone()),
                missing_frags.into(),
                // No pushdown of limit/offset when doing scalar indexed scan
                None,
                false,
            );
            let filtered = Arc::new(FilterExec::try_new(physical_refine_expr, new_data_scan)?);
            Some(Arc::new(project(filtered, plan.schema().as_ref())?))
        } else {
            None
        };

        if let Some(new_data_path) = new_data_path {
            let unioned = UnionExec::new(vec![plan, new_data_path]);
            // Enforce only 1 partition.
            let unioned = RepartitionExec::try_new(
                Arc::new(unioned),
                datafusion::physical_plan::Partitioning::RoundRobinBatch(1),
            )?;
            Ok(Arc::new(unioned))
        } else {
            Ok(plan)
        }
    }

    fn get_io_buffer_size(&self) -> u64 {
        self.io_buffer_size.unwrap_or(*DEFAULT_IO_BUFFER_SIZE)
    }

    /// Create an Execution plan with a scan node
    ///
    /// Setting `with_make_deletions_null` will use the validity of the _rowid
    /// column as a selection vector. Read more in [crate::io::FileReader].
    pub(crate) fn scan(
        &self,
        with_row_id: bool,
        with_row_address: bool,
        with_make_deletions_null: bool,
        range: Option<Range<u64>>,
        projection: Arc<Schema>,
    ) -> Arc<dyn ExecutionPlan> {
        let fragments = if let Some(fragment) = self.fragments.as_ref() {
            Arc::new(fragment.clone())
        } else {
            self.dataset.fragments().clone()
        };
        let ordered = if self.ordering.is_some() || self.nearest.is_some() {
            // If we are sorting the results there is no need to scan in order
            false
        } else {
            self.ordered
        };
        self.scan_fragments(
            with_row_id,
            with_row_address,
            with_make_deletions_null,
            projection,
            fragments,
            range,
            ordered,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn scan_fragments(
        &self,
        with_row_id: bool,
        with_row_address: bool,
        with_make_deletions_null: bool,
        projection: Arc<Schema>,
        fragments: Arc<Vec<Fragment>>,
        range: Option<Range<u64>>,
        ordered: bool,
    ) -> Arc<dyn ExecutionPlan> {
        let config = LanceScanConfig {
            batch_size: self.get_batch_size(),
            batch_readahead: self.batch_readahead,
            fragment_readahead: self.fragment_readahead,
            io_buffer_size: self.get_io_buffer_size(),
            with_row_id,
            with_row_address,
            with_make_deletions_null,
            ordered_output: ordered,
        };
        Arc::new(LanceScanExec::new(
            self.dataset.clone(),
            fragments,
            range,
            projection,
            config,
        ))
    }

    fn pushdown_scan(
        &self,
        make_deletions_null: bool,
        predicate: Expr,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let config = ScanConfig {
            batch_readahead: self.batch_readahead,
            fragment_readahead: self
                .fragment_readahead
                .unwrap_or(LEGACY_DEFAULT_FRAGMENT_READAHEAD),
            with_row_id: self.with_row_id,
            with_row_address: self.with_row_address,
            make_deletions_null,
            ordered_output: self.ordered,
        };

        let fragments = if let Some(fragment) = self.fragments.as_ref() {
            Arc::new(fragment.clone())
        } else {
            self.dataset.fragments().clone()
        };

        Ok(Arc::new(LancePushdownScanExec::try_new(
            self.dataset.clone(),
            fragments,
            self.projection_plan.physical_schema.clone(),
            predicate,
            config,
        )?))
    }

    /// Add a knn search node to the input plan
    fn flat_knn(&self, input: Arc<dyn ExecutionPlan>, q: &Query) -> Result<Arc<dyn ExecutionPlan>> {
        let flat_dist = Arc::new(KNNVectorDistanceExec::try_new(
            input,
            &q.column,
            q.key.clone(),
            q.metric_type,
        )?);

        // Use DataFusion's [SortExec] for Top-K search
        let sort = SortExec::new(
            vec![PhysicalSortExpr {
                expr: expressions::col(DIST_COL, flat_dist.schema().as_ref())?,
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            }],
            flat_dist,
        )
        .with_fetch(Some(q.k));

        let not_nulls = FilterExec::try_new(
            expressions::is_not_null(expressions::col(DIST_COL, sort.schema().as_ref())?)?,
            Arc::new(sort),
        )?;

        Ok(Arc::new(not_nulls))
    }

    /// Create an Execution plan to do indexed ANN search
    async fn ann(
        &self,
        q: &Query,
        index: &[Index],
        filter_plan: &FilterPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let prefilter_source = self.prefilter_source(filter_plan).await?;

        let inner_fanout_search = new_knn_exec(self.dataset.clone(), index, q, prefilter_source)?;
        let sort_expr = PhysicalSortExpr {
            expr: expressions::col(DIST_COL, inner_fanout_search.schema().as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        };
        Ok(Arc::new(
            SortExec::new(vec![sort_expr], inner_fanout_search)
                .with_fetch(Some(q.k * q.refine_factor.unwrap_or(1) as usize)),
        ))
    }

    /// Create prefilter source from filter plan
    async fn prefilter_source(&self, filter_plan: &FilterPlan) -> Result<PreFilterSource> {
        let prefilter_source = match (
            &filter_plan.index_query,
            &filter_plan.refine_expr,
            self.prefilter,
        ) {
            (Some(index_query), Some(refine_expr), _) => {
                // The filter is only partially satisfied by the index.  We need
                // to do an indexed scan and then refine the results to determine
                // the row ids.
                let columns_in_filter = Planner::column_names_in_expr(refine_expr);
                let filter_schema = Arc::new(self.dataset.schema().project(&columns_in_filter)?);
                let filter_input = self
                    .scalar_indexed_scan(&filter_schema, index_query)
                    .await?;
                let planner = Planner::new(filter_input.schema());
                let physical_refine_expr = planner.create_physical_expr(refine_expr)?;
                let filtered_row_ids =
                    Arc::new(FilterExec::try_new(physical_refine_expr, filter_input)?);
                PreFilterSource::FilteredRowIds(filtered_row_ids)
            } // Should be index_scan -> filter
            (Some(index_query), None, true) => {
                // Index scan doesn't honor the fragment allowlist today.
                // TODO: we could filter the index scan results to only include the allowed fragments.
                self.ensure_not_fragment_scan()?;

                // The filter is completely satisfied by the index.  We
                // only need to search the index to determine the valid row
                // ids.
                let index_query = Arc::new(ScalarIndexExec::new(
                    self.dataset.clone(),
                    index_query.clone(),
                ));
                PreFilterSource::ScalarIndexQuery(index_query)
            }
            (None, Some(refine_expr), true) => {
                // No indices match the filter.  We need to do a full scan
                // of the filter columns to determine the valid row ids.
                let columns_in_filter = Planner::column_names_in_expr(refine_expr);
                let filter_schema = Arc::new(self.dataset.schema().project(&columns_in_filter)?);
                let filter_input = self.scan(true, false, true, None, filter_schema);
                let planner = Planner::new(filter_input.schema());
                let physical_refine_expr = planner.create_physical_expr(refine_expr)?;
                let filtered_row_ids =
                    Arc::new(FilterExec::try_new(physical_refine_expr, filter_input)?);
                PreFilterSource::FilteredRowIds(filtered_row_ids)
            }
            // No prefilter
            (None, None, true) => PreFilterSource::None,
            (_, _, false) => PreFilterSource::None,
        };

        Ok(prefilter_source)
    }

    /// Take row indices produced by input plan from the dataset (with projection)
    fn take(
        &self,
        input: Arc<dyn ExecutionPlan>,
        projection: &Schema,
        batch_readahead: usize,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let coalesced = Arc::new(CoalesceBatchesExec::new(input, self.get_batch_size()));
        Ok(Arc::new(TakeExec::try_new(
            self.dataset.clone(),
            coalesced,
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

    #[instrument(level = "info", skip(self))]
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
            Poll::Ready(result) => {
                Poll::Ready(result.map(|r| r.map_err(|e| Error::io(e.to_string(), location!()))))
            }
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
pub mod test_dataset {

    use super::*;

    use std::vec;

    use arrow_array::{ArrayRef, FixedSizeListArray, Int32Array, RecordBatchIterator, StringArray};
    use arrow_schema::ArrowError;
    use lance_file::version::LanceFileVersion;
    use lance_index::scalar::InvertedIndexParams;
    use lance_index::{scalar::ScalarIndexParams, IndexType};
    use tempfile::{tempdir, TempDir};

    use crate::arrow::*;
    use crate::dataset::WriteParams;
    use crate::index::vector::VectorIndexParams;

    // Creates a dataset with 5 batches where each batch has 80 rows
    //
    // The dataset has the following columns:
    //
    //  i   - i32      : [0, 1, ..., 399]
    //  s   - &str     : ["s-0", "s-1", ..., "s-399"]
    //  vec - [f32; 32]: [[0, 1, ... 31], [32, ..., 63], ... [..., (80 * 5 * 32) - 1]]
    //
    // An IVF-PQ index with 2 partitions is trained on this data
    pub struct TestVectorDataset {
        pub tmp_dir: TempDir,
        pub schema: Arc<ArrowSchema>,
        pub dataset: Dataset,
        dimension: u32,
    }

    impl TestVectorDataset {
        pub async fn new(
            data_storage_version: LanceFileVersion,
            stable_row_ids: bool,
        ) -> Result<Self> {
            Self::new_with_dimension(data_storage_version, stable_row_ids, 32).await
        }

        pub async fn new_with_dimension(
            data_storage_version: LanceFileVersion,
            stable_row_ids: bool,
            dimension: u32,
        ) -> Result<Self> {
            let tmp_dir = tempdir()?;
            let path = tmp_dir.path().to_str().unwrap();

            // Make sure the schema has metadata so it tests all paths that re-construct the schema along the way
            let metadata: HashMap<String, String> =
                vec![("dataset".to_string(), "vector".to_string())]
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
                            dimension as i32,
                        ),
                        true,
                    ),
                ],
                metadata,
            ));

            let batches: Vec<RecordBatch> = (0..5)
                .map(|i| {
                    let vector_values: Float32Array =
                        (0..dimension * 80).map(|v| v as f32).collect();
                    let vectors =
                        FixedSizeListArray::try_new_from_values(vector_values, dimension as i32)
                            .unwrap();
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
                })
                .collect::<std::result::Result<Vec<_>, ArrowError>>()?;

            let params = WriteParams {
                max_rows_per_group: 10,
                max_rows_per_file: 200,
                data_storage_version: Some(data_storage_version),
                enable_move_stable_row_ids: stable_row_ids,
                ..Default::default()
            };
            let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

            let dataset = Dataset::write(reader, path, Some(params)).await?;

            Ok(Self {
                tmp_dir,
                schema,
                dataset,
                dimension,
            })
        }

        pub async fn make_vector_index(&mut self) -> Result<()> {
            let params = VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 2);
            self.dataset
                .create_index(
                    &["vec"],
                    IndexType::Vector,
                    Some("idx".to_string()),
                    &params,
                    true,
                )
                .await
        }

        pub async fn make_scalar_index(&mut self) -> Result<()> {
            self.dataset
                .create_index(
                    &["i"],
                    IndexType::Scalar,
                    None,
                    &ScalarIndexParams::default(),
                    true,
                )
                .await
        }

        pub async fn make_fts_index(&mut self) -> Result<()> {
            self.dataset
                .create_index(
                    &["s"],
                    IndexType::Inverted,
                    None,
                    &InvertedIndexParams::default(),
                    true,
                )
                .await
        }

        pub async fn append_new_data(&mut self) -> Result<()> {
            let vector_values: Float32Array = (0..10)
                .flat_map(|i| vec![i as f32; self.dimension as usize].into_iter())
                .collect();
            let new_vectors =
                FixedSizeListArray::try_new_from_values(vector_values, self.dimension as i32)
                    .unwrap();
            let new_data: Vec<ArrayRef> = vec![
                Arc::new(Int32Array::from_iter_values(400..410)), // 5 * 80
                Arc::new(StringArray::from_iter_values(
                    (400..410).map(|v| format!("s-{}", v)),
                )),
                Arc::new(new_vectors),
            ];
            let reader = RecordBatchIterator::new(
                vec![RecordBatch::try_new(self.schema.clone(), new_data).unwrap()]
                    .into_iter()
                    .map(Ok),
                self.schema.clone(),
            );
            self.dataset.append(reader, None).await?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod test {

    use std::collections::BTreeSet;
    use std::sync::Mutex;
    use std::vec;

    use arrow::array::as_primitive_array;
    use arrow::datatypes::Int32Type;
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float32Type, UInt64Type};
    use arrow_array::{
        ArrayRef, FixedSizeListArray, Float16Array, Int32Array, LargeStringArray, PrimitiveArray,
        RecordBatchIterator, StringArray, StructArray,
    };
    use arrow_ord::sort::sort_to_indices;
    use arrow_select::take;
    use datafusion::logical_expr::{col, lit};
    use half::f16;
    use lance_datagen::{array, gen, BatchCount, ByteCount, Dimension, RowCount};
    use lance_file::version::LanceFileVersion;
    use lance_index::scalar::InvertedIndexParams;
    use lance_index::vector::hnsw::builder::HnswBuildParams;
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::pq::PQBuildParams;
    use lance_index::vector::sq::builder::SQBuildParams;
    use lance_index::{scalar::ScalarIndexParams, IndexType};
    use lance_io::object_store::ObjectStoreParams;
    use lance_linalg::distance::DistanceType;
    use lance_testing::datagen::{BatchGenerator, IncrementingInt32, RandomVector};
    use rstest::rstest;
    use tempfile::{tempdir, TempDir};

    use super::*;
    use crate::arrow::*;
    use crate::dataset::optimize::{compact_files, CompactionOptions};
    use crate::dataset::scanner::test_dataset::TestVectorDataset;
    use crate::dataset::WriteMode;
    use crate::dataset::WriteParams;
    use crate::index::vector::{StageParams, VectorIndexParams};
    use crate::utils::test::{IoStats, IoTrackingStore};

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

        for use_filter in [false, true] {
            let test_dir = tempdir().unwrap();
            let test_uri = test_dir.path().to_str().unwrap();
            let write_params = WriteParams {
                max_rows_per_file: 40,
                max_rows_per_group: 10,
                ..Default::default()
            };
            let batches =
                RecordBatchIterator::new(batches.clone().into_iter().map(Ok), schema.clone());
            Dataset::write(batches, test_uri, Some(write_params))
                .await
                .unwrap();

            let dataset = Dataset::open(test_uri).await.unwrap();
            let mut builder = dataset.scan();
            builder.batch_size(8);
            if use_filter {
                builder.filter("i IS NOT NULL").unwrap();
            }
            let mut stream = builder.try_into_stream().await.unwrap();
            let mut rows_read = 0;
            while let Some(next) = stream.next().await {
                let next = next.unwrap();
                let expected = 8.min(100 - rows_read);
                assert_eq!(next.num_rows(), expected);
                rows_read += next.num_rows();
            }
        }
    }

    #[cfg(not(windows))]
    #[tokio::test]
    async fn test_local_object_store() {
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
        let batches = RecordBatchIterator::new(batches.clone().into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(&format!("file-object-store://{}", test_uri))
            .await
            .unwrap();
        let mut builder = dataset.scan();
        builder.batch_size(8);
        let mut stream = builder.try_into_stream().await.unwrap();
        let mut rows_read = 0;
        while let Some(next) = stream.next().await {
            let next = next.unwrap();
            let expected = 8.min(100 - rows_read);
            assert_eq!(next.num_rows(), expected);
            rows_read += next.num_rows();
        }
    }

    #[tokio::test]
    async fn test_filter_parsing() -> Result<()> {
        let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false).await?;
        let dataset = &test_ds.dataset;

        let mut scan = dataset.scan();
        assert!(scan.filter.is_none());

        scan.filter("i > 50")?;
        assert_eq!(scan.get_filter().unwrap(), Some(col("i").gt(lit(50))));

        for use_stats in [false, true] {
            let batches = scan
                .project(&["s"])?
                .use_stats(use_stats)
                .try_into_stream()
                .await?
                .try_collect::<Vec<_>>()
                .await?;
            let batch = concat_batches(&batches[0].schema(), &batches)?;

            let expected_batch = RecordBatch::try_new(
                // Projected just "s"
                Arc::new(test_ds.schema.project(&[1])?),
                vec![Arc::new(StringArray::from_iter_values(
                    (51..400).map(|v| format!("s-{}", v)),
                ))],
            )?;
            assert_eq!(batch, expected_batch);
        }
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_limit(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) -> Result<()> {
        let test_ds = TestVectorDataset::new(data_storage_version, false).await?;
        let dataset = &test_ds.dataset;

        let full_data = dataset.scan().try_into_batch().await?.slice(19, 2);

        let actual = dataset
            .scan()
            .limit(Some(2), Some(19))?
            .try_into_batch()
            .await?;

        assert_eq!(actual, full_data);
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_knn_nodes(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] stable_row_ids: bool,
    ) {
        for build_index in &[true, false] {
            let mut test_ds = TestVectorDataset::new(data_storage_version, stable_row_ids)
                .await
                .unwrap();
            if *build_index {
                test_ds.make_vector_index().await.unwrap();
            }
            let dataset = &test_ds.dataset;

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
                    ArrowField::new(DIST_COL, DataType::Float32, true),
                ])
                .with_metadata([("dataset".into(), "vector".into())].into())
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

    #[rstest]
    #[tokio::test]
    async fn test_knn_with_new_data(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] stable_row_ids: bool,
    ) {
        let mut test_ds = TestVectorDataset::new(data_storage_version, stable_row_ids)
            .await
            .unwrap();
        test_ds.make_vector_index().await.unwrap();
        test_ds.append_new_data().await.unwrap();
        let dataset = &test_ds.dataset;

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

    #[rstest]
    #[tokio::test]
    async fn test_knn_with_prefilter(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] stable_row_ids: bool,
    ) {
        let mut test_ds = TestVectorDataset::new(data_storage_version, stable_row_ids)
            .await
            .unwrap();
        test_ds.make_vector_index().await.unwrap();
        let dataset = &test_ds.dataset;

        let mut scan = dataset.scan();
        let key: Float32Array = (32..64).map(|v| v as f32).collect();
        scan.filter("i > 100").unwrap();
        scan.prefilter(true);
        scan.project(&["i", "vec"]).unwrap();
        scan.nearest("vec", &key, 5).unwrap();
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
                ArrowField::new(DIST_COL, DataType::Float32, true),
            ])
            .with_metadata([("dataset".into(), "vector".into())].into())
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

    #[rstest]
    #[tokio::test]
    async fn test_knn_filter_new_data(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] stable_row_ids: bool,
    ) {
        // This test verifies that a filter (prefilter or postfilter) gets applied to the flat KNN results
        // in a combined KNN scan (a scan that combines results from an indexed ANN with an unindexed flat
        // search of new data)
        let mut test_ds = TestVectorDataset::new(data_storage_version, stable_row_ids)
            .await
            .unwrap();
        test_ds.make_vector_index().await.unwrap();
        test_ds.append_new_data().await.unwrap();
        let dataset = &test_ds.dataset;

        // This query will match exactly the new row with i = 400 which should be excluded by the prefilter
        let key: Float32Array = [0f32; 32].into_iter().collect();

        let mut query = dataset.scan();
        query.nearest("vec", &key, 20).unwrap();

        // Sanity check that 400 is in our results
        let results = query
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let results_i = results[0]["i"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();

        assert!(results_i.contains(&400));

        // Both prefilter and postfilter should remove 400 from our results
        for prefilter in [false, true] {
            let mut query = dataset.scan();
            query
                .filter("i != 400")
                .unwrap()
                .prefilter(prefilter)
                .nearest("vec", &key, 20)
                .unwrap();

            let results = query
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();

            let results_i = results[0]["i"]
                .as_primitive::<Int32Type>()
                .values()
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();

            assert!(!results_i.contains(&400));
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_knn_with_filter(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] stable_row_ids: bool,
    ) {
        let test_ds = TestVectorDataset::new(data_storage_version, stable_row_ids)
            .await
            .unwrap();
        let dataset = &test_ds.dataset;

        let mut scan = dataset.scan();
        let key: Float32Array = (32..64).map(|v| v as f32).collect();
        scan.nearest("vec", &key, 5).unwrap();
        scan.filter("i > 100").unwrap();
        scan.project(&["i", "vec"]).unwrap();
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
                ArrowField::new(DIST_COL, DataType::Float32, true),
            ])
            .with_metadata([("dataset".into(), "vector".into())].into())
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

    #[rstest]
    #[tokio::test]
    async fn test_refine_factor(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] stable_row_ids: bool,
    ) {
        let test_ds = TestVectorDataset::new(data_storage_version, stable_row_ids)
            .await
            .unwrap();
        let dataset = &test_ds.dataset;

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
                ArrowField::new(DIST_COL, DataType::Float32, true),
            ])
            .with_metadata([("dataset".into(), "vector".into())].into())
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

    #[rstest]
    #[tokio::test]
    async fn test_only_row_id(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_ds = TestVectorDataset::new(data_storage_version, false)
            .await
            .unwrap();
        let dataset = &test_ds.dataset;

        let mut scan = dataset.scan();
        scan.project::<&str>(&[]).unwrap().with_row_id();

        let batch = scan.try_into_batch().await.unwrap();

        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.num_rows(), 400);
        let expected_schema =
            ArrowSchema::new(vec![ArrowField::new(ROW_ID, DataType::UInt64, true)])
                .with_metadata(dataset.schema().metadata.clone());
        assert_eq!(batch.schema().as_ref(), &expected_schema,);

        let expected_row_ids: Vec<u64> = (0..200_u64).chain((1 << 32)..((1 << 32) + 200)).collect();
        let actual_row_ids: Vec<u64> = as_primitive_array::<UInt64Type>(batch.column(0).as_ref())
            .values()
            .iter()
            .copied()
            .collect();
        assert_eq!(expected_row_ids, actual_row_ids);
    }

    #[tokio::test]
    async fn test_scan_unordered_with_row_id() {
        // This test doesn't make sense for v2 files, there is no way to get an out-of-order scan
        let test_ds = TestVectorDataset::new(LanceFileVersion::Legacy, false)
            .await
            .unwrap();
        let dataset = &test_ds.dataset;

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
                let sort_indices = sort_to_indices(&unordered_batch[ROW_ID], None, None).unwrap();

                let ordered_i = ordered_batch["i"].clone();
                let sorted_i = take::take(&unordered_batch["i"], &sort_indices, None).unwrap();

                assert_eq!(&ordered_i, &sorted_i);

                break;
            }
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_scan_order(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
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
            data_storage_version: Some(data_storage_version),
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

    #[rstest]
    #[tokio::test]
    async fn test_scan_sort(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let data = gen()
            .col("int", array::cycle::<Int32Type>(vec![5, 4, 1, 2, 3]))
            .col(
                "str",
                array::cycle_utf8_literals(&["a", "b", "c", "e", "d"]),
            );

        let sorted_by_int = gen()
            .col("int", array::cycle::<Int32Type>(vec![1, 2, 3, 4, 5]))
            .col(
                "str",
                array::cycle_utf8_literals(&["c", "e", "d", "b", "a"]),
            )
            .into_batch_rows(RowCount::from(5))
            .unwrap();

        let sorted_by_str = gen()
            .col("int", array::cycle::<Int32Type>(vec![5, 4, 1, 3, 2]))
            .col(
                "str",
                array::cycle_utf8_literals(&["a", "b", "c", "d", "e"]),
            )
            .into_batch_rows(RowCount::from(5))
            .unwrap();

        Dataset::write(
            data.into_reader_rows(RowCount::from(5), BatchCount::from(1)),
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let dataset = Arc::new(Dataset::open(test_uri).await.unwrap());

        let batches_by_int = dataset
            .scan()
            .order_by(Some(vec![ColumnOrdering::asc_nulls_first(
                "int".to_string(),
            )]))
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(batches_by_int[0], sorted_by_int);

        let batches_by_str = dataset
            .scan()
            .order_by(Some(vec![ColumnOrdering::asc_nulls_first(
                "str".to_string(),
            )]))
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(batches_by_str[0], sorted_by_str);

        // Ensure an empty sort vec does not break anything (sorting is disabled)
        dataset
            .scan()
            .order_by(Some(vec![]))
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
    }

    #[rstest]
    #[tokio::test]
    async fn test_sort_multi_columns(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let data = gen()
            .col("int", array::cycle::<Int32Type>(vec![5, 5, 1, 1, 3]))
            .col(
                "float",
                array::cycle::<Float32Type>(vec![7.3, -f32::NAN, f32::NAN, 4.3, f32::INFINITY]),
            );

        let sorted_by_int_then_float = gen()
            .col("int", array::cycle::<Int32Type>(vec![1, 1, 3, 5, 5]))
            .col(
                "float",
                // floats should be sorted using total order so -NAN is before all and NAN is after all
                array::cycle::<Float32Type>(vec![4.3, f32::NAN, f32::INFINITY, -f32::NAN, 7.3]),
            )
            .into_batch_rows(RowCount::from(5))
            .unwrap();

        Dataset::write(
            data.into_reader_rows(RowCount::from(5), BatchCount::from(1)),
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let dataset = Arc::new(Dataset::open(test_uri).await.unwrap());

        let batches_by_int_then_float = dataset
            .scan()
            .order_by(Some(vec![
                ColumnOrdering::asc_nulls_first("int".to_string()),
                ColumnOrdering::asc_nulls_first("float".to_string()),
            ]))
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(batches_by_int_then_float[0], sorted_by_int_then_float);
    }

    #[rstest]
    #[tokio::test]
    async fn test_ann_prefilter(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] stable_row_ids: bool,
        #[values(
            VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 2),
            VectorIndexParams::with_ivf_hnsw_sq_params(
                MetricType::L2,
                IvfBuildParams::new(2),
                HnswBuildParams::default(),
                SQBuildParams::default()
            )
        )]
        index_params: VectorIndexParams,
    ) {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("filterable", DataType::Int32, true),
            ArrowField::new("vector", fixed_size_list_type(2, DataType::Float32), true),
        ]));

        let vector_values = Float32Array::from_iter_values((0..600).map(|x| x as f32));

        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..300)),
                Arc::new(FixedSizeListArray::try_new_from_values(vector_values, 2).unwrap()),
            ],
        )
        .unwrap()];

        let write_params = WriteParams {
            data_storage_version: Some(data_storage_version),
            max_rows_per_file: 300, // At least two files to make sure stable row ids make a difference
            enable_move_stable_row_ids: stable_row_ids,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        dataset
            .create_index(&["vector"], IndexType::Vector, None, &index_params, false)
            .await
            .unwrap();

        let query_key = Arc::new(Float32Array::from_iter_values((0..2).map(|x| x as f32)));
        let mut scan = dataset.scan();
        scan.filter("filterable > 5").unwrap();
        scan.nearest("vector", &query_key, 1).unwrap();
        scan.with_row_id();

        let batches = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(batches.len(), 0);

        scan.prefilter(true);
        let batches = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(batches.len(), 1);

        let first_match = batches[0][ROW_ID].as_primitive::<UInt64Type>().values()[0];

        assert_eq!(6, first_match);
    }

    #[rstest]
    #[tokio::test]
    async fn test_filter_on_large_utf8(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
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

        let write_params = WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
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

    #[rstest]
    #[tokio::test]
    async fn test_filter_with_regex(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
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

        let write_params = WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
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
            data_storage_version: Some(LanceFileVersion::Legacy),
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

    #[rstest]
    #[tokio::test]
    async fn test_ann_with_deletion(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] stable_row_ids: bool,
    ) {
        let vec_params = vec![
            // TODO: re-enable diskann test when we can tune to get reproducible results.
            // VectorIndexParams::with_diskann_params(MetricType::L2, DiskANNParams::new(10, 1.5, 10)),
            VectorIndexParams::ivf_pq(4, 8, 2, MetricType::L2, 2),
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
            let mut dataset = Dataset::write(
                reader,
                test_uri,
                Some(WriteParams {
                    data_storage_version: Some(data_storage_version),
                    enable_move_stable_row_ids: stable_row_ids,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

            assert_eq!(dataset.index_cache_entry_count(), 0);
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

            assert_eq!(
                dataset.index_cache_entry_count(),
                1, // 1 for index metadata
            );
            let results = scan
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();

            assert_eq!(
                dataset.index_cache_entry_count(),
                5 + dataset.versions().await.unwrap().len()
            );
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
                    data_storage_version: Some(data_storage_version),
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

    #[rstest]
    #[tokio::test]
    async fn test_count_rows_with_filter(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut data_gen = BatchGenerator::new().col(Box::new(
            IncrementingInt32::new().named("Filter_me".to_owned()),
        ));
        Dataset::write(
            data_gen.batch(32),
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
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

    #[rstest]
    #[tokio::test]
    async fn test_dynamic_projection(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("i".to_owned())));
        Dataset::write(
            data_gen.batch(32),
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        assert_eq!(32, dataset.scan().count_rows().await.unwrap());

        let mut scanner = dataset.scan();

        let scan_res = scanner
            .project_with_transform(&[("bool", "i > 15")])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(1, scan_res.num_columns());

        let bool_col = scan_res
            .column_by_name("bool")
            .expect("bool column should exist");
        let bool_arr = bool_col.as_boolean();
        for i in 0..32 {
            if i > 15 {
                assert!(bool_arr.value(i));
            } else {
                assert!(!bool_arr.value(i));
            }
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_column_casting_function(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut data_gen =
            BatchGenerator::new().col(Box::new(RandomVector::new().named("vec".to_owned())));
        Dataset::write(
            data_gen.batch(32),
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        assert_eq!(32, dataset.scan().count_rows().await.unwrap());

        let mut scanner = dataset.scan();

        let scan_res = scanner
            .project_with_transform(&[("f16", "_cast_list_f16(vec)")])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(1, scan_res.num_columns());
        assert_eq!(32, scan_res.num_rows());
        assert_eq!("f16", scan_res.schema().field(0).name());

        let mut scanner = dataset.scan();
        let scan_res_original = scanner
            .project(&["vec"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let f32_col: &Float32Array = scan_res_original
            .column_by_name("vec")
            .unwrap()
            .as_fixed_size_list()
            .values()
            .as_primitive();
        let f16_col: &Float16Array = scan_res
            .column_by_name("f16")
            .unwrap()
            .as_fixed_size_list()
            .values()
            .as_primitive();

        for (f32_val, f16_val) in f32_col.iter().zip(f16_col.iter()) {
            let f32_val = f32_val.unwrap();
            let f16_val = f16_val.unwrap();
            assert_eq!(f16::from_f32(f32_val), f16_val);
        }
    }

    struct ScalarIndexTestFixture {
        _test_dir: TempDir,
        dataset: Dataset,
        sample_query: Arc<dyn Array>,
        delete_query: Arc<dyn Array>,
        // The original version of the data, two fragments, rows 0-1000
        original_version: u64,
        // The original version of the data, 1 row deleted, compacted to a single fragment
        compact_version: u64,
        // The original version of the data + an extra 1000 unindexed
        append_version: u64,
        // The original version of the data + an extra 1000 rows, with indices updated so all rows indexed
        updated_version: u64,
        // The original version of the data with 1 deleted row
        delete_version: u64,
        // The original version of the data + an extra 1000 uindexed + 1 deleted row
        append_then_delete_version: u64,
    }

    #[derive(Debug, PartialEq)]
    struct ScalarTestParams {
        use_index: bool,
        use_projection: bool,
        use_deleted_data: bool,
        use_new_data: bool,
        with_row_id: bool,
        use_compaction: bool,
        use_updated: bool,
    }

    impl ScalarIndexTestFixture {
        async fn new(data_storage_version: LanceFileVersion, use_stable_row_ids: bool) -> Self {
            let test_dir = tempdir().unwrap();
            let test_uri = test_dir.path().to_str().unwrap();

            // Write 1000 rows.  Train indices.  Then write 1000 new rows with the same vector data.
            // Then delete a row from the trained data.
            //
            // The first row where indexed == 50 is our sample query.
            // The first row where indexed == 75 is our deleted row (and delete query)
            let data = gen()
                .col(
                    "vector",
                    array::rand_vec::<Float32Type>(Dimension::from(32)),
                )
                .col("indexed", array::step::<Int32Type>())
                .col("not_indexed", array::step::<Int32Type>())
                .into_batch_rows(RowCount::from(1000))
                .unwrap();

            // Write as two batches so we can later compact
            let mut dataset = Dataset::write(
                RecordBatchIterator::new(vec![Ok(data.clone())], data.schema().clone()),
                test_uri,
                Some(WriteParams {
                    max_rows_per_file: 500,
                    data_storage_version: Some(data_storage_version),
                    enable_move_stable_row_ids: use_stable_row_ids,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

            dataset
                .create_index(
                    &["vector"],
                    IndexType::Vector,
                    None,
                    &VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 2),
                    false,
                )
                .await
                .unwrap();

            dataset
                .create_index(
                    &["indexed"],
                    IndexType::Scalar,
                    None,
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();

            let original_version = dataset.version().version;
            let sample_query = data["vector"].as_fixed_size_list().value(50);
            let delete_query = data["vector"].as_fixed_size_list().value(75);

            // APPEND DATA

            // Re-use the vector column in the new batch but add 1000 to the indexed/not_indexed columns so
            // they are distinct.  This makes our checks easier.
            let new_indexed =
                arrow_arith::numeric::add(&data["indexed"], &Int32Array::new_scalar(1000)).unwrap();
            let new_not_indexed =
                arrow_arith::numeric::add(&data["indexed"], &Int32Array::new_scalar(1000)).unwrap();
            let append_data = RecordBatch::try_new(
                data.schema(),
                vec![data["vector"].clone(), new_indexed, new_not_indexed],
            )
            .unwrap();

            dataset
                .append(
                    RecordBatchIterator::new(vec![Ok(append_data)], data.schema()),
                    Some(WriteParams {
                        data_storage_version: Some(data_storage_version),
                        ..Default::default()
                    }),
                )
                .await
                .unwrap();

            let append_version = dataset.version().version;

            // UPDATE

            dataset.optimize_indices(&Default::default()).await.unwrap();
            let updated_version = dataset.version().version;

            // APPEND -> DELETE

            dataset.checkout_version(append_version).await.unwrap();
            dataset.restore().await.unwrap();

            dataset.delete("not_indexed = 75").await.unwrap();

            let append_then_delete_version = dataset.version().version;

            // DELETE

            let mut dataset = dataset.checkout_version(original_version).await.unwrap();
            dataset.restore().await.unwrap();

            dataset.delete("not_indexed = 75").await.unwrap();

            let delete_version = dataset.version().version;

            // COMPACT (this should materialize the deletion)
            compact_files(&mut dataset, CompactionOptions::default(), None)
                .await
                .unwrap();
            let compact_version = dataset.version().version;
            dataset.checkout_version(original_version).await.unwrap();
            dataset.restore().await.unwrap();

            Self {
                _test_dir: test_dir,
                dataset,
                sample_query,
                delete_query,
                original_version,
                compact_version,
                append_version,
                updated_version,
                delete_version,
                append_then_delete_version,
            }
        }

        fn sample_query(&self) -> &PrimitiveArray<Float32Type> {
            self.sample_query.as_primitive::<Float32Type>()
        }

        fn delete_query(&self) -> &PrimitiveArray<Float32Type> {
            self.delete_query.as_primitive::<Float32Type>()
        }

        async fn get_dataset(&self, params: &ScalarTestParams) -> Dataset {
            let version = if params.use_compaction {
                // These combinations should not be possible
                if params.use_deleted_data || params.use_new_data || params.use_updated {
                    panic!(
                        "There is no test data combining new/deleted/updated data with compaction"
                    );
                } else {
                    self.compact_version
                }
            } else if params.use_updated {
                // These combinations should not be possible
                if params.use_deleted_data || params.use_new_data || params.use_compaction {
                    panic!(
                        "There is no test data combining updated data with new/deleted/compaction"
                    );
                } else {
                    self.updated_version
                }
            } else {
                match (params.use_new_data, params.use_deleted_data) {
                    (false, false) => self.original_version,
                    (false, true) => self.delete_version,
                    (true, false) => self.append_version,
                    (true, true) => self.append_then_delete_version,
                }
            };
            self.dataset.checkout_version(version).await.unwrap()
        }

        async fn run_query(
            &self,
            query: &str,
            vector: Option<&PrimitiveArray<Float32Type>>,
            params: &ScalarTestParams,
        ) -> (String, RecordBatch) {
            let dataset = self.get_dataset(params).await;
            let mut scan = dataset.scan();
            if let Some(vector) = vector {
                scan.nearest("vector", vector, 10).unwrap();
            }
            if params.use_projection {
                scan.project(&["indexed"]).unwrap();
            }
            if params.with_row_id {
                scan.with_row_id();
            }
            scan.scan_in_order(true);
            scan.use_index(params.use_index);
            scan.filter(query).unwrap();
            scan.prefilter(true);

            let plan = scan.explain_plan(true).await.unwrap();
            let batch = scan.try_into_batch().await.unwrap();

            if params.use_projection {
                // 1 projected column
                let mut expected_columns = 1;
                if vector.is_some() {
                    // distance column if included always (TODO: it shouldn't)
                    expected_columns += 1;
                }
                if params.with_row_id {
                    expected_columns += 1;
                }
                assert_eq!(batch.num_columns(), expected_columns);
            } else {
                let mut expected_columns = 3;
                if vector.is_some() {
                    // distance column
                    expected_columns += 1;
                }
                if params.with_row_id {
                    expected_columns += 1;
                }
                // vector, indexed, not_indexed, _distance
                assert_eq!(batch.num_columns(), expected_columns);
            }

            (plan, batch)
        }

        fn assert_none<F: Fn(i32) -> bool>(
            &self,
            batch: &RecordBatch,
            predicate: F,
            message: &str,
        ) {
            let indexed = batch["indexed"].as_primitive::<Int32Type>();
            if indexed.iter().map(|val| val.unwrap()).any(predicate) {
                panic!("{}", message);
            }
        }

        fn assert_one<F: Fn(i32) -> bool>(&self, batch: &RecordBatch, predicate: F, message: &str) {
            let indexed = batch["indexed"].as_primitive::<Int32Type>();
            if !indexed.iter().map(|val| val.unwrap()).any(predicate) {
                panic!("{}", message);
            }
        }

        async fn check_vector_scalar_indexed_and_refine(&self, params: &ScalarTestParams) {
            let (query_plan, batch) = self
                .run_query(
                    "indexed != 50 AND ((not_indexed < 100) OR (not_indexed >= 1000 AND not_indexed < 1100))",
                    Some(self.sample_query()),
                    params,
                )
                .await;
            // Materialization is always required if there is a refine
            assert!(query_plan.contains("MaterializeIndex"));
            // The result should not include the sample query
            self.assert_none(
                &batch,
                |val| val == 50,
                "The query contained 50 even though it was filtered",
            );
            if !params.use_new_data {
                // Refine should have been applied
                self.assert_none(
                    &batch,
                    |val| (100..1000).contains(&val) || (val >= 1100),
                    "The non-indexed refine filter was not applied",
                );
            }

            // If there is new data then the dupe of row 50 should be in the results
            if params.use_new_data || params.use_updated {
                self.assert_one(
                    &batch,
                    |val| val == 1050,
                    "The query did not contain 1050 from the new data",
                );
            }
        }

        async fn check_vector_scalar_indexed_only(&self, params: &ScalarTestParams) {
            let (query_plan, batch) = self
                .run_query("indexed != 50", Some(self.sample_query()), params)
                .await;
            if params.use_index {
                // An ANN search whose prefilter is fully satisfied by the index should be
                // able to use a ScalarIndexQuery
                assert!(query_plan.contains("ScalarIndexQuery"));
            } else {
                // A KNN search requires materialization of the index
                assert!(query_plan.contains("MaterializeIndex"));
            }
            // The result should not include the sample query
            self.assert_none(
                &batch,
                |val| val == 50,
                "The query contained 50 even though it was filtered",
            );
            // If there is new data then the dupe of row 50 should be in the results
            if params.use_new_data {
                self.assert_one(
                    &batch,
                    |val| val == 1050,
                    "The query did not contain 1050 from the new data",
                );
                if !params.use_new_data {
                    // Let's also make sure our filter can target something in the new data only
                    let (_, batch) = self
                        .run_query("indexed == 1050", Some(self.sample_query()), params)
                        .await;
                    assert_eq!(batch.num_rows(), 1);
                }
            }
            if params.use_deleted_data {
                let (_, batch) = self
                    .run_query("indexed == 75", Some(self.delete_query()), params)
                    .await;
                if !params.use_new_data {
                    assert_eq!(batch.num_rows(), 0);
                }
            }
        }

        async fn check_vector_queries(&self, params: &ScalarTestParams) {
            self.check_vector_scalar_indexed_only(params).await;
            self.check_vector_scalar_indexed_and_refine(params).await;
        }

        async fn check_simple_indexed_only(&self, params: &ScalarTestParams) {
            let (query_plan, batch) = self.run_query("indexed != 50", None, params).await;
            // Materialization is always required for non-vector search
            assert!(query_plan.contains("MaterializeIndex"));
            // The result should not include the targeted row
            self.assert_none(
                &batch,
                |val| val == 50,
                "The query contained 50 even though it was filtered",
            );
            let mut expected_num_rows = if params.use_new_data || params.use_updated {
                1999
            } else {
                999
            };
            if params.use_deleted_data || params.use_compaction {
                expected_num_rows -= 1;
            }
            assert_eq!(batch.num_rows(), expected_num_rows);

            // Let's also make sure our filter can target something in the new data only
            if params.use_new_data || params.use_updated {
                let (_, batch) = self.run_query("indexed == 1050", None, params).await;
                assert_eq!(batch.num_rows(), 1);
            }

            // Also make sure we don't return deleted data
            if params.use_deleted_data || params.use_compaction {
                let (_, batch) = self.run_query("indexed == 75", None, params).await;
                assert_eq!(batch.num_rows(), 0);
            }
        }

        async fn check_simple_indexed_and_refine(&self, params: &ScalarTestParams) {
            let (query_plan, batch) = self.run_query(
                "indexed != 50 AND ((not_indexed < 100) OR (not_indexed >= 1000 AND not_indexed < 1100))",
                None,
                params
            ).await;
            // Materialization is always required for non-vector search
            assert!(query_plan.contains("MaterializeIndex"));
            // The result should not include the targeted row
            self.assert_none(
                &batch,
                |val| val == 50,
                "The query contained 50 even though it was filtered",
            );
            // The refine should be applied
            self.assert_none(
                &batch,
                |val| (100..1000).contains(&val) || (val >= 1100),
                "The non-indexed refine filter was not applied",
            );

            let mut expected_num_rows = if params.use_new_data || params.use_updated {
                199
            } else {
                99
            };
            if params.use_deleted_data || params.use_compaction {
                expected_num_rows -= 1;
            }
            assert_eq!(batch.num_rows(), expected_num_rows);
        }

        async fn check_simple_queries(&self, params: &ScalarTestParams) {
            self.check_simple_indexed_only(params).await;
            self.check_simple_indexed_and_refine(params).await;
        }
    }

    // There are many different ways that a query can be run and they all have slightly different
    // effects on the plan that gets built.  This test attempts to run the same queries in various
    // different configurations to ensure that we get consistent results
    #[rstest]
    #[tokio::test]
    async fn test_secondary_index_scans(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] use_stable_row_ids: bool,
    ) {
        let fixture = ScalarIndexTestFixture::new(data_storage_version, use_stable_row_ids).await;

        for use_index in [false, true] {
            for use_projection in [false, true] {
                for use_deleted_data in [false, true] {
                    for use_new_data in [false, true] {
                        // Don't test compaction in conjunction with deletion and new data, it's too
                        // many combinations with no clear benefit.  Feel free to update if there is
                        // a need
                        // TODO: enable compaction for stable row id once supported.
                        let compaction_choices =
                            if use_deleted_data || use_new_data || use_stable_row_ids {
                                vec![false]
                            } else {
                                vec![false, true]
                            };
                        for use_compaction in compaction_choices {
                            let updated_choices =
                                if use_deleted_data || use_new_data || use_compaction {
                                    vec![false]
                                } else {
                                    vec![false, true]
                                };
                            for use_updated in updated_choices {
                                for with_row_id in [false, true] {
                                    let params = ScalarTestParams {
                                        use_index,
                                        use_projection,
                                        use_deleted_data,
                                        use_new_data,
                                        with_row_id,
                                        use_compaction,
                                        use_updated,
                                    };
                                    fixture.check_vector_queries(&params).await;
                                    fixture.check_simple_queries(&params).await;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Assert that the plan when formatted matches the expected string.
    ///
    /// Within expected, you can use `...` to match any number of characters.
    async fn assert_plan_equals(
        dataset: &Dataset,
        plan: impl Fn(&mut Scanner) -> Result<&mut Scanner>,
        expected: &str,
    ) -> Result<()> {
        let mut scan = dataset.scan();
        plan(&mut scan)?;
        let exec_plan = scan.create_plan().await?;
        let plan_desc = format!(
            "{}",
            datafusion::physical_plan::displayable(exec_plan.as_ref()).indent(true)
        );

        let to_match = expected.split("...").collect::<Vec<_>>();
        let num_pieces = to_match.len();
        let mut remainder = plan_desc.as_str().trim_end_matches('\n');
        for (i, piece) in to_match.into_iter().enumerate() {
            let res = match i {
                0 => remainder.starts_with(piece),
                _ if i == num_pieces - 1 => remainder.ends_with(piece),
                _ => remainder.contains(piece),
            };
            if !res {
                break;
            }
            let idx = remainder.find(piece).unwrap();
            remainder = &remainder[idx + piece.len()..];
        }
        if !remainder.is_empty() {
            panic!(
                "Expected plan to match:\nExpected: {}\nActual: {}",
                expected, plan_desc
            )
        }
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_late_materialization(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // Create a large dataset with a scalar indexed column and a sorted but not scalar
        // indexed column
        use lance_table::io::commit::RenameCommitHandler;
        let data = gen()
            .col(
                "vector",
                array::rand_vec::<Float32Type>(Dimension::from(32)),
            )
            .col("indexed", array::step::<Int32Type>())
            .col("not_indexed", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(1000), BatchCount::from(20));

        let (io_stats_wrapper, io_stats) = IoTrackingStore::new_wrapper();
        let mut dataset = Dataset::write(
            data,
            "memory://test",
            Some(WriteParams {
                store_params: Some(ObjectStoreParams {
                    object_store_wrapper: Some(io_stats_wrapper),
                    ..Default::default()
                }),
                commit_handler: Some(Arc::new(RenameCommitHandler)),
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["indexed"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        let get_bytes = || io_stats.lock().unwrap().read_bytes;

        // First run a full scan to get a baseline
        let start_bytes = get_bytes();
        dataset.scan().try_into_batch().await.unwrap();
        let full_scan_bytes = get_bytes() - start_bytes;

        // Next do a scan without pushdown, we should still see a benefit from late materialization
        let start_bytes = get_bytes();
        dataset
            .scan()
            .use_stats(false)
            .filter("not_indexed = 50")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let filtered_scan_bytes = get_bytes() - start_bytes;

        assert!(filtered_scan_bytes < full_scan_bytes);

        // Now do a scan with pushdown, the benefit should be even greater
        let start_bytes = get_bytes();
        dataset
            .scan()
            .filter("not_indexed = 50")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let pushdown_scan_bytes = get_bytes() - start_bytes;

        assert!(pushdown_scan_bytes < filtered_scan_bytes);

        // Now do a scalar index scan, this should be better than a
        // full scan but since we have to load the index might be more
        // expensive than late / pushdown scan
        let start_bytes = get_bytes();
        dataset
            .scan()
            .filter("indexed = 50")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let index_scan_bytes = get_bytes() - start_bytes;
        assert!(index_scan_bytes < full_scan_bytes);

        // A second scalar index scan should be cheaper than the first
        // since we should have the index in cache
        let start_bytes = get_bytes();
        dataset
            .scan()
            .filter("indexed = 50")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let second_index_scan_bytes = get_bytes() - start_bytes;
        assert!(second_index_scan_bytes < index_scan_bytes);
    }

    #[rstest]
    #[tokio::test]
    async fn test_project_nested(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) -> Result<()> {
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
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();

        let batches = dataset
            .scan()
            .project(&["struct.i"])
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let batch = concat_batches(&batches[0].schema(), &batches).unwrap();
        assert!(batch.column_by_name("struct.i").is_some());
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_plans(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] stable_row_id: bool,
    ) -> Result<()> {
        // Create a vector dataset
        let mut dataset =
            TestVectorDataset::new_with_dimension(data_storage_version, stable_row_id, 256).await?;
        let lance_schema = dataset.dataset.schema();

        // Scans
        // ---------------------------------------------------------------------
        // V2 writer does not use LancePushdownScan
        if data_storage_version == LanceFileVersion::Legacy {
            assert_plan_equals(
                &dataset.dataset,
                |scan| scan.project(&["s"])?.filter("i > 10 and i < 20"),
                "LancePushdownScan: uri=..., projection=[s], predicate=i > Int32(10) AND i < Int32(20), row_id=false, row_addr=false, ordered=true"
            ).await?;
        }

        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.use_stats(false)
                    .project(&["s"])?
                    .filter("i > 10 and i < 20")
            },
            "ProjectionExec: expr=[s@2 as s]
  Take: columns=\"i, _rowid, (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: i@0 > 10 AND i@0 < 20
        LanceScan: uri..., projection=[i], row_id=true, row_addr=false, ordered=true",
        )
        .await?;

        // Integer fields will be eagerly materialized while string/vec fields
        // are not.
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.use_stats(false).filter("s IS NOT NULL"),
            "ProjectionExec: expr=[i@1 as i, s@0 as s, vec@3 as vec]
  Take: columns=\"s, i, _rowid, (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: s@0 IS NOT NULL
        LanceScan: uri..., projection=[s, i], row_id=true, row_addr=false, ordered=true",
        )
        .await?;

        // Custom materialization
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.use_stats(false)
                    .materialization_style(MaterializationStyle::AllEarly)
                    .filter("s IS NOT NULL")
            },
            "ProjectionExec: expr=[i@1 as i, s@0 as s, vec@2 as vec]
  FilterExec: s@0 IS NOT NULL
    LanceScan: uri..., projection=[s, i, vec], row_id=true, row_addr=false, ordered=true",
        )
        .await?;

        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.use_stats(false)
                    .materialization_style(MaterializationStyle::AllLate)
                    .filter("s IS NOT NULL")
            },
            "ProjectionExec: expr=[i@2 as i, s@0 as s, vec@3 as vec]
  Take: columns=\"s, _rowid, (i), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: s@0 IS NOT NULL
        LanceScan: uri..., projection=[s], row_id=true, row_addr=false, ordered=true",
        )
        .await?;

        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.use_stats(false)
                    .materialization_style(
                        MaterializationStyle::all_early_except(&["i"], lance_schema).unwrap(),
                    )
                    .filter("s IS NOT NULL")
            },
            "ProjectionExec: expr=[i@3 as i, s@0 as s, vec@1 as vec]
  Take: columns=\"s, vec, _rowid, (i)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: s@0 IS NOT NULL
        LanceScan: uri..., projection=[s, vec], row_id=true, row_addr=false, ordered=true",
        )
        .await?;

        assert_plan_equals(
            &dataset.dataset,
            |scan| Ok(scan.project(&["s"])?.with_row_id().scan_in_order(false)),
            "LanceScan: uri=..., projection=[s], row_id=true, row_addr=false, ordered=false",
        )
        .await?;

        // KNN
        // ---------------------------------------------------------------------
        let q: Float32Array = (32..64).map(|v| v as f32).collect();
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.nearest("vec", &q, 5),
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@0 as vec, _distance@2 as _distance]
  Take: columns=\"vec, _rowid, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@2 IS NOT NULL
        SortExec: TopK(fetch=5), expr=...
          KNNVectorDistance: metric=l2
            LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false",
        )
        .await?;

        // ANN
        // ---------------------------------------------------------------------
        dataset.make_vector_index().await?;
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.nearest("vec", &q, 42),
            "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  Take: columns=\"_distance, _rowid, (i), (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: TopK(fetch=42), expr=...
        ANNSubIndex: name=..., k=42, deltas=1
          ANNIvfPartition: uuid=..., nprobes=1, deltas=1",
        )
        .await?;

        assert_plan_equals(
            &dataset.dataset,
            |scan| Ok(scan.nearest("vec", &q, 10)?.refine(4)),
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  Take: columns=\"_rowid, vec, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=10), expr=...
          KNNVectorDistance: metric=l2
            Take: columns=\"_distance, _rowid, (vec)\"
              CoalesceBatchesExec: target_batch_size=8192
                SortExec: TopK(fetch=40), expr=...
                  ANNSubIndex: name=..., k=40, deltas=1
                    ANNIvfPartition: uuid=..., nprobes=1, deltas=1",
        )
        .await?;

        // use_index = False -> same plan as KNN
        assert_plan_equals(
            &dataset.dataset,
            |scan| Ok(scan.nearest("vec", &q, 13)?.use_index(false)),
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@0 as vec, _distance@2 as _distance]
  Take: columns=\"vec, _rowid, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=13), expr=...
          KNNVectorDistance: metric=l2
            LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false",
        )
        .await?;

        // with filter and projection
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                Ok(scan
                    .nearest("vec", &q, 17)?
                    .filter("i > 10")?
                    .project(&["s", "vec"])?
                    .with_row_id())
            },
            "ProjectionExec: expr=[s@3 as s, vec@4 as vec, _distance@0 as _distance, _rowid@1 as _rowid]
  Take: columns=\"_distance, _rowid, i, (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: i@2 > 10
        Take: columns=\"_distance, _rowid, (i)\"
          CoalesceBatchesExec: target_batch_size=8192
            SortExec: TopK(fetch=17), expr=...
              ANNSubIndex: name=..., k=17, deltas=1
                ANNIvfPartition: uuid=..., nprobes=1, deltas=1",
        )
        .await?;

        // with prefilter
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                Ok(scan
                    .nearest("vec", &q, 17)?
                    .filter("i > 10")?
                    .prefilter(true))
            },
            "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  Take: columns=\"_distance, _rowid, (i), (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: TopK(fetch=17), expr=...
        ANNSubIndex: name=..., k=17, deltas=1
          ANNIvfPartition: uuid=..., nprobes=1, deltas=1
          FilterExec: i@0 > 10
            LanceScan: uri=..., projection=[i], row_id=true, row_addr=false, ordered=false",
        )
        .await?;

        dataset.append_new_data().await?;
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.nearest("vec", &q, 6),
            // TODO: we could write an optimizer rule to eliminate the last Projection
            // by doing it as part of the last Take. This would likely have minimal impact though.
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  Take: columns=\"_rowid, vec, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=6), expr=...
          KNNVectorDistance: metric=l2
            RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
              UnionExec
                ProjectionExec: expr=[_distance@2 as _distance, _rowid@1 as _rowid, vec@0 as vec]
                  FilterExec: _distance@... IS NOT NULL
                    SortExec: TopK(fetch=6), expr=...
                      KNNVectorDistance: metric=l2
                        LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false
                Take: columns=\"_distance, _rowid, (vec)\"
                  CoalesceBatchesExec: target_batch_size=8192
                    SortExec: TopK(fetch=6), expr=...
                      ANNSubIndex: name=..., k=6, deltas=1
                        ANNIvfPartition: uuid=..., nprobes=1, deltas=1",
        )
        .await?;

        // new data and with filter
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.nearest("vec", &q, 15)?.filter("i > 10"),
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  Take: columns=\"_rowid, vec, _distance, i, (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: i@3 > 10
        Take: columns=\"_rowid, vec, _distance, (i)\"
          CoalesceBatchesExec: target_batch_size=8192
            FilterExec: _distance@... IS NOT NULL
              SortExec: TopK(fetch=15), expr=...
                KNNVectorDistance: metric=l2
                  RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
                    UnionExec
                      ProjectionExec: expr=[_distance@2 as _distance, _rowid@1 as _rowid, vec@0 as vec]
                        FilterExec: _distance@... IS NOT NULL
                          SortExec: TopK(fetch=15), expr=...
                            KNNVectorDistance: metric=l2
                              LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false
                      Take: columns=\"_distance, _rowid, (vec)\"
                        CoalesceBatchesExec: target_batch_size=8192
                          SortExec: TopK(fetch=15), expr=...
                            ANNSubIndex: name=..., k=15, deltas=1
                              ANNIvfPartition: uuid=..., nprobes=1, deltas=1",
        )
        .await?;

        // new data and with prefilter
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                Ok(scan
                    .nearest("vec", &q, 5)?
                    .filter("i > 10")?
                    .prefilter(true))
            },
            // TODO: i is scanned on both sides but is projected away mid-plan
            // only to be taken again later. We should fix this.
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  Take: columns=\"_rowid, vec, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=5), expr=...
          KNNVectorDistance: metric=l2
            RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
              UnionExec
                ProjectionExec: expr=[_distance@3 as _distance, _rowid@2 as _rowid, vec@0 as vec]
                  FilterExec: _distance@... IS NOT NULL
                    SortExec: TopK(fetch=5), expr=...
                      KNNVectorDistance: metric=l2
                        FilterExec: i@1 > 10
                          LanceScan: uri=..., projection=[vec, i], row_id=true, row_addr=false, ordered=false
                Take: columns=\"_distance, _rowid, (vec)\"
                  CoalesceBatchesExec: target_batch_size=8192
                    SortExec: TopK(fetch=5), expr=...
                      ANNSubIndex: name=..., k=5, deltas=1
                        ANNIvfPartition: uuid=..., nprobes=1, deltas=1
                        FilterExec: i@0 > 10
                          LanceScan: uri=..., projection=[i], row_id=true, row_addr=false, ordered=false",
        )
        .await?;

        // ANN with scalar index
        // ---------------------------------------------------------------------
        // Make sure both indices are up-to-date to start
        dataset.make_vector_index().await?;
        dataset.make_scalar_index().await?;

        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                Ok(scan
                    .nearest("vec", &q, 5)?
                    .filter("i > 10")?
                    .prefilter(true))
            },
            "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  Take: columns=\"_distance, _rowid, (i), (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: TopK(fetch=5), expr=...
        ANNSubIndex: name=..., k=5, deltas=1
          ANNIvfPartition: uuid=..., nprobes=1, deltas=1
          ScalarIndexQuery: query=i > 10",
        )
        .await?;

        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                Ok(scan
                    .nearest("vec", &q, 5)?
                    .use_scalar_index(false)
                    .filter("i > 10")?
                    .prefilter(true))
            },
            "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  Take: columns=\"_distance, _rowid, (i), (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: TopK(fetch=5), expr=...
        ANNSubIndex: name=..., k=5, deltas=1
          ANNIvfPartition: uuid=..., nprobes=1, deltas=1
          FilterExec: i@0 > 10
            LanceScan: uri=..., projection=[i], row_id=true, row_addr=false, ordered=false",
        )
        .await?;

        dataset.append_new_data().await?;

        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                Ok(scan
                    .nearest("vec", &q, 8)?
                    .filter("i > 10")?
                    .prefilter(true))
            },
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  Take: columns=\"_rowid, vec, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=8), expr=...
          KNNVectorDistance: metric=l2
            RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
              UnionExec
                ProjectionExec: expr=[_distance@3 as _distance, _rowid@2 as _rowid, vec@0 as vec]
                  FilterExec: _distance@... IS NOT NULL
                    SortExec: TopK(fetch=8), expr=...
                      KNNVectorDistance: metric=l2
                        FilterExec: i@1 > 10
                          LanceScan: uri=..., projection=[vec, i], row_id=true, row_addr=false, ordered=false
                Take: columns=\"_distance, _rowid, (vec)\"
                  CoalesceBatchesExec: target_batch_size=8192
                    SortExec: TopK(fetch=8), expr=...
                      ANNSubIndex: name=..., k=8, deltas=1
                        ANNIvfPartition: uuid=..., nprobes=1, deltas=1
                        ScalarIndexQuery: query=i > 10",
        )
        .await?;

        // Update scalar index but not vector index
        dataset.make_scalar_index().await?;
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                Ok(scan
                    .nearest("vec", &q, 11)?
                    .filter("i > 10")?
                    .prefilter(true))
            },
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  Take: columns=\"_rowid, vec, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=11), expr=...
          KNNVectorDistance: metric=l2
            RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
              UnionExec
                ProjectionExec: expr=[_distance@3 as _distance, _rowid@2 as _rowid, vec@0 as vec]
                  FilterExec: _distance@... IS NOT NULL
                    SortExec: TopK(fetch=11), expr=...
                      KNNVectorDistance: metric=l2
                        FilterExec: i@1 > 10
                          LanceScan: uri=..., projection=[vec, i], row_id=true, row_addr=false, ordered=false
                Take: columns=\"_distance, _rowid, (vec)\"
                  CoalesceBatchesExec: target_batch_size=8192
                    SortExec: TopK(fetch=11), expr=...
                      ANNSubIndex: name=..., k=11, deltas=1
                        ANNIvfPartition: uuid=..., nprobes=1, deltas=1
                        ScalarIndexQuery: query=i > 10",
        )
        .await?;

        // Scans with scalar index
        // ---------------------------------------------------------------------
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.project(&["s"])?.filter("i > 10"),
            "ProjectionExec: expr=[s@1 as s]
  Take: columns=\"_rowid, (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      MaterializeIndex: query=i > 10",
        )
        .await?;

        if data_storage_version != LanceFileVersion::Legacy {
            assert_plan_equals(
                &dataset.dataset,
                |scan| {
                    scan.project(&["s"])?
                        .use_scalar_index(false)
                        .filter("i > 10")
                },
                "ProjectionExec: expr=[s@2 as s]
  Take: columns=\"i, _rowid, (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: i@0 > 10
        LanceScan: uri=..., projection=[i], row_id=true, row_addr=false, ordered=true",
            )
            .await?;
        }

        // Empty projection
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.filter("i > 10")
                    .unwrap()
                    .with_row_address()
                    .project::<&str>(&[])
            },
            "ProjectionExec: expr=[_rowaddr@0 as _rowaddr]
  AddRowAddrExec
    MaterializeIndex: query=i > 10",
        )
        .await?;

        dataset.append_new_data().await?;
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.project(&["s"])?.filter("i > 10"),
            "ProjectionExec: expr=[s@1 as s]
  RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
    UnionExec
      Take: columns=\"_rowid, (s)\"
        CoalesceBatchesExec: target_batch_size=8192
          MaterializeIndex: query=i > 10
      ProjectionExec: expr=[_rowid@2 as _rowid, s@0 as s]
        FilterExec: i@1 > 10
          LanceScan: uri=..., projection=[s, i], row_id=true, row_addr=false, ordered=false",
        )
        .await?;

        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.filter("i > 10")
                    .unwrap()
                    .with_row_address()
                    .project::<&str>(&[])
            },
            "ProjectionExec: expr=[_rowaddr@0 as _rowaddr]
  RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
    UnionExec
      AddRowAddrExec
        MaterializeIndex: query=i > 10
      ProjectionExec: expr=[_rowaddr@2 as _rowaddr, _rowid@1 as _rowid]
        FilterExec: i@0 > 10
          LanceScan: uri=..., projection=[i], row_id=true, row_addr=true, ordered=false",
        )
        .await?;

        // Empty projection
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.filter("i > 10")
                    .unwrap()
                    .with_row_address()
                    .project::<&str>(&[])
            },
            "ProjectionExec: expr=[_rowaddr@0 as _rowaddr]
  RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
    UnionExec
      AddRowAddrExec
        MaterializeIndex: query=i > 10
      ProjectionExec: expr=[_rowaddr@2 as _rowaddr, _rowid@1 as _rowid]
        FilterExec: i@0 > 10
          LanceScan: uri=..., projection=[i], row_id=true, row_addr=true, ordered=false",
        )
        .await?;

        // Scans with dynamic projection
        // When an expression is specified in the projection, the plan should include a ProjectionExec
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.project_with_transform(&[("matches", "regexp_match(s, \".*\")")])?
                    .filter("i > 10")
            },
            "ProjectionExec: expr=[regexp_match(s@0, .*) as matches]
  ProjectionExec: expr=[s@1 as s]
    RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
      UnionExec
        Take: columns=\"_rowid, (s)\"
          CoalesceBatchesExec: target_batch_size=8192
            MaterializeIndex: query=i > 10
        ProjectionExec: expr=[_rowid@2 as _rowid, s@0 as s]
          FilterExec: i@1 > 10
            LanceScan: uri=..., projection=[s, i], row_id=true, row_addr=false, ordered=false",
        )
        .await?;

        // FTS
        // ---------------------------------------------------------------------
        // All rows are indexed
        dataset.make_fts_index().await?;
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.project(&["s"])?
                    .with_row_id()
                    .full_text_search(FullTextSearchQuery::new("hello".to_owned()))
            },
            r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: expr=[_score@1 DESC NULLS LAST], preserve_partitioning=[false]
        RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
          UnionExec
            Fts: query=hello
            FlatFts: query=hello
              EmptyExec"#,
        )
        .await?;

        // With prefilter
        dataset.make_fts_index().await?;
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.project(&["s"])?
                    .with_row_id()
                    .filter("i > 10")?
                    .prefilter(true)
                    .full_text_search(FullTextSearchQuery::new("hello".to_owned()))
            },
            r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: expr=[_score@1 DESC NULLS LAST], preserve_partitioning=[false]
        RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
          UnionExec
            Fts: query=hello
              ScalarIndexQuery: query=i > 10
            FlatFts: query=hello
              EmptyExec"#,
        )
        .await?;

        // With unindexed rows
        dataset.append_new_data().await?;
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.project(&["s"])?
                    .with_row_id()
                    .full_text_search(FullTextSearchQuery::new("hello".to_owned()))
            },
            r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: expr=[_score@1 DESC NULLS LAST], preserve_partitioning=[false]
        RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
          UnionExec
            Fts: query=hello
            FlatFts: query=hello
              LanceScan: uri=..., projection=[s], row_id=true, row_addr=false, ordered=false"#,
        )
        .await?;

        // With unindexed data & prefilter
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.project(&["s"])?
                    .with_row_id()
                    .filter("i > 10")?
                    .prefilter(true)
                    .full_text_search(FullTextSearchQuery::new("hello".to_owned()))
            },
            r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: expr=[_score@1 DESC NULLS LAST], preserve_partitioning=[false]
        RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
          UnionExec
            Fts: query=hello
              ScalarIndexQuery: query=i > 10
            FlatFts: query=hello
              FilterExec: i@1 > 10
                LanceScan: uri=..., projection=[s, i], row_id=true, row_addr=false, ordered=false"#,
        )
        .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_fast_search_plan() {
        // Create a vector dataset
        let mut dataset = TestVectorDataset::new(LanceFileVersion::Stable, true)
            .await
            .unwrap();
        dataset.make_vector_index().await.unwrap();
        dataset.append_new_data().await.unwrap();

        let q: Float32Array = (32..64).map(|v| v as f32).collect();

        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.nearest("vec", &q, 32)?
                    .fast_search()
                    .project(&["_rowid", "_distance"])
            },
            "ProjectionExec: expr=[_rowid@1 as _rowid, _distance@0 as _distance]
  SortExec: TopK(fetch=32), expr=[_distance@0 ASC NULLS LAST]...
    ANNSubIndex: name=idx, k=32, deltas=1
      ANNIvfPartition: uuid=..., nprobes=1, deltas=1",
        )
        .await
        .unwrap();

        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.nearest("vec", &q, 33)?
                    .fast_search()
                    .with_row_id()
                    .project(&["_rowid", "_distance"])
            },
            "ProjectionExec: expr=[_rowid@1 as _rowid, _distance@0 as _distance]
  SortExec: TopK(fetch=33), expr=[_distance@0 ASC NULLS LAST]...
    ANNSubIndex: name=idx, k=33, deltas=1
      ANNIvfPartition: uuid=..., nprobes=1, deltas=1",
        )
        .await
        .unwrap();

        // Not `fast_scan` case
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.nearest("vec", &q, 34)?
                    .with_row_id()
                    .project(&["_rowid", "_distance"])
            },
            "ProjectionExec: expr=[_rowid@0 as _rowid, _distance@2 as _distance]
  FilterExec: _distance@2 IS NOT NULL
    SortExec: TopK(fetch=34), expr=[_distance@2 ASC NULLS LAST]...
      KNNVectorDistance: metric=l2
        RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
          UnionExec
            ProjectionExec: expr=[_distance@2 as _distance, _rowid@1 as _rowid, vec@0 as vec]
              FilterExec: _distance@2 IS NOT NULL
                SortExec: TopK(fetch=34), expr=[_distance@2 ASC NULLS LAST]...
                  KNNVectorDistance: metric=l2
                    LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false
            Take: columns=\"_distance, _rowid, (vec)\"
              CoalesceBatchesExec: target_batch_size=8192
                SortExec: TopK(fetch=34), expr=[_distance@0 ASC NULLS LAST]...
                  ANNSubIndex: name=idx, k=34, deltas=1
                    ANNIvfPartition: uuid=..., nprobes=1, deltas=1",
        )
        .await
        .unwrap();
    }

    #[rstest]
    #[tokio::test]
    pub async fn test_scan_planning_io(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // Create a large dataset with a scalar indexed column and a sorted but not scalar
        // indexed column
        let data = gen()
            .col(
                "vector",
                array::rand_vec::<Float32Type>(Dimension::from(32)),
            )
            .col("text", array::rand_utf8(ByteCount::from(4), false))
            .col("indexed", array::step::<Int32Type>())
            .col("not_indexed", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(100), BatchCount::from(5));

        let (io_stats_wrapper, io_stats) = IoTrackingStore::new_wrapper();
        let mut dataset = Dataset::write(
            data,
            "memory://test",
            Some(WriteParams {
                store_params: Some(ObjectStoreParams {
                    object_store_wrapper: Some(io_stats_wrapper),
                    ..Default::default()
                }),
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["indexed"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                None,
                &VectorIndexParams {
                    metric_type: DistanceType::L2,
                    stages: vec![
                        StageParams::Ivf(IvfBuildParams {
                            max_iters: 2,
                            num_partitions: 2,
                            sample_rate: 2,
                            ..Default::default()
                        }),
                        StageParams::PQ(PQBuildParams {
                            max_iters: 2,
                            num_sub_vectors: 2,
                            ..Default::default()
                        }),
                    ],
                    version: crate::index::vector::IndexFileVersion::Legacy,
                },
                false,
            )
            .await
            .unwrap();

        struct IopsTracker {
            baseline: u64,
            new_iops: u64,
            io_stats: Arc<Mutex<IoStats>>,
        }

        impl IopsTracker {
            fn update(&mut self) {
                let iops = self.io_stats.lock().unwrap().read_iops;
                self.new_iops = iops - self.baseline;
                self.baseline = iops;
            }

            fn new_iops(&mut self) -> u64 {
                self.update();
                self.new_iops
            }
        }

        let mut tracker = IopsTracker {
            baseline: 0,
            new_iops: 0,
            io_stats,
        };

        // First planning cycle needs to do some I/O to determine what scalar indices are available
        dataset
            .scan()
            .prefilter(true)
            .filter("indexed > 10")
            .unwrap()
            .explain_plan(true)
            .await
            .unwrap();

        // First pass will need to perform some IOPs to determine what scalar indices are available
        assert!(tracker.new_iops() > 0);

        // Second planning cycle should not perform any I/O
        dataset
            .scan()
            .prefilter(true)
            .filter("indexed > 10")
            .unwrap()
            .explain_plan(true)
            .await
            .unwrap();

        assert_eq!(tracker.new_iops(), 0);

        dataset
            .scan()
            .prefilter(true)
            .filter("true")
            .unwrap()
            .explain_plan(true)
            .await
            .unwrap();

        assert_eq!(tracker.new_iops(), 0);

        dataset
            .scan()
            .prefilter(true)
            .materialization_style(MaterializationStyle::AllEarly)
            .filter("true")
            .unwrap()
            .explain_plan(true)
            .await
            .unwrap();

        assert_eq!(tracker.new_iops(), 0);

        dataset
            .scan()
            .prefilter(true)
            .materialization_style(MaterializationStyle::AllLate)
            .filter("true")
            .unwrap()
            .explain_plan(true)
            .await
            .unwrap();

        assert_eq!(tracker.new_iops(), 0);
    }
}
