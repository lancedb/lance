// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;
use std::pin::Pin;
use std::sync::{Arc, LazyLock};
use std::task::{Context, Poll};

use arrow::array::AsArray;
use arrow_array::{Array, Float32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema, SchemaRef, SortOptions};
use arrow_select::concat::concat_batches;
use async_recursion::async_recursion;
use datafusion::common::{DFSchema, SchemaExt};
use datafusion::functions_aggregate;
use datafusion::functions_aggregate::count::count_udaf;
use datafusion::logical_expr::{col, lit, Expr};
use datafusion::physical_expr::PhysicalSortExpr;
use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion::physical_plan::expressions;
use datafusion::physical_plan::projection::ProjectionExec as DFProjectionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::{
    aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
    display::DisplayableExecutionPlan,
    expressions::Literal,
    limit::GlobalLimitExec,
    repartition::RepartitionExec,
    union::UnionExec,
    ExecutionPlan, SendableRecordBatchStream,
};
use datafusion::scalar::ScalarValue;
use datafusion_expr::execution_props::ExecutionProps;
use datafusion_expr::ExprSchemable;
use datafusion_physical_expr::{aggregate::AggregateExprBuilder, expressions::Column};
use datafusion_physical_expr::{create_physical_expr, LexOrdering, Partitioning, PhysicalExpr};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{empty::EmptyExec, joins::HashJoinExec};
use futures::future::BoxFuture;
use futures::stream::{Stream, StreamExt};
use futures::{FutureExt, TryStreamExt};
use lance_arrow::floats::{coerce_float_vector, FloatType};
use lance_arrow::DataTypeExt;
use lance_core::datatypes::{Field, OnMissing, Projection};
use lance_core::error::LanceOptionExt;
use lance_core::utils::address::RowAddress;
use lance_core::utils::mask::{RowIdMask, RowIdTreeMap};
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{ROW_ADDR, ROW_ID, ROW_OFFSET};
use lance_datafusion::exec::{
    analyze_plan, execute_plan, LanceExecutionOptions, OneShotExec, StrictBatchSizeExec,
};
use lance_datafusion::expr::safe_coerce_scalar;
use lance_datafusion::projection::ProjectionPlan;
use lance_file::v2::reader::FileReaderOptions;
use lance_index::scalar::expression::{IndexExprResult, PlannerIndexExt, INDEX_EXPR_RESULT_SCHEMA};
use lance_index::scalar::inverted::query::{
    fill_fts_query_column, FtsQuery, FtsSearchParams, MatchQuery,
};
use lance_index::scalar::inverted::SCORE_COL;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::vector::{Query, DIST_COL};
use lance_index::ScalarIndexCriteria;
use lance_index::{metrics::NoOpMetricsCollector, scalar::inverted::FTS_SCHEMA};
use lance_index::{scalar::expression::ScalarIndexExpr, DatasetIndexExt};
use lance_io::stream::RecordBatchStream;
use lance_linalg::distance::MetricType;
use lance_table::format::{Fragment, Index};
use roaring::RoaringBitmap;
use tracing::{info_span, instrument, Span};

use super::Dataset;
use crate::dataset::row_offsets_to_row_addresses;
use crate::dataset::utils::wrap_json_stream_for_reading;
use crate::index::vector::utils::{get_vector_dim, get_vector_type};
use crate::index::DatasetIndexInternalExt;
use crate::io::exec::filtered_read::{FilteredReadExec, FilteredReadOptions};
use crate::io::exec::fts::{BoostQueryExec, FlatMatchQueryExec, MatchQueryExec, PhraseQueryExec};
use crate::io::exec::knn::MultivectorScoringExec;
use crate::io::exec::scalar_index::{MaterializeIndexExec, ScalarIndexExec};
use crate::io::exec::{get_physical_optimizer, AddRowOffsetExec, LanceFilterExec, LanceScanConfig};
use crate::io::exec::{
    knn::new_knn_exec, project, AddRowAddrExec, FilterPlan, KNNVectorDistanceExec,
    LancePushdownScanExec, LanceScanExec, Planner, PreFilterSource, ScanConfig, TakeExec,
};
use crate::{datatypes::Schema, io::exec::fts::BooleanQueryExec};
use crate::{Error, Result};

use snafu::location;

pub use lance_datafusion::exec::{ExecutionStatsCallback, ExecutionSummaryCounts};
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

pub static DEFAULT_FRAGMENT_READAHEAD: LazyLock<Option<usize>> = LazyLock::new(|| {
    std::env::var("LANCE_DEFAULT_FRAGMENT_READAHEAD")
        .map(|val| Some(val.parse().unwrap()))
        .unwrap_or(None)
});

pub static DEFAULT_XTR_OVERFETCH: LazyLock<u32> = LazyLock::new(|| {
    std::env::var("LANCE_XTR_OVERFETCH")
        .map(|val| val.parse().unwrap())
        .unwrap_or(10)
});

// We want to support ~256 concurrent reads to maximize throughput on cloud storage systems
// Our typical page size is 8MiB (though not all reads are this large yet due to offset buffers, validity buffers, etc.)
// So we want to support 256 * 8MiB ~= 2GiB of queued reads
pub static DEFAULT_IO_BUFFER_SIZE: LazyLock<u64> = LazyLock::new(|| {
    std::env::var("LANCE_DEFAULT_IO_BUFFER_SIZE")
        .map(|val| val.parse().unwrap())
        .unwrap_or(2 * 1024 * 1024 * 1024)
});

/// Defines an ordering for a single column
///
/// Floats are sorted using the IEEE 754 total ordering
/// Strings are sorted using UTF-8 lexicographic order (i.e. we sort the binary)
#[derive(Debug, Clone)]
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
#[derive(Clone)]
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

#[derive(Debug)]
struct PlannedFilteredScan {
    plan: Arc<dyn ExecutionPlan>,
    limit_pushed_down: bool,
    filter_pushed_down: bool,
}

/// Filter for filtering rows
#[derive(Debug, Clone)]
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
                let planner = Planner::new(schema.clone());
                let filter = planner.parse_filter(sql)?;

                let df_schema = DFSchema::try_from(schema)?;
                let (ret_type, _) = filter.data_type_and_nullable(&df_schema)?;
                if ret_type != DataType::Boolean {
                    return Err(Error::InvalidInput {
                        source: format!("The filter {} does not return a boolean", filter).into(),
                        location: location!(),
                    });
                }

                planner.optimize_expr(filter).map_err(|e| {
                    Error::invalid_input(
                        format!("Error optimizing sql filter: {sql} ({e})"),
                        location!(),
                    )
                })
            }
            #[cfg(feature = "substrait")]
            Self::Substrait(expr) => {
                use lance_datafusion::exec::{get_session_context, LanceExecutionOptions};

                let ctx = get_session_context(&LanceExecutionOptions::default());
                let state = ctx.state();
                let schema = Arc::new(ArrowSchema::from(dataset_schema));
                let expr = parse_substrait(expr, schema.clone(), &ctx.state())
                    .now_or_never()
                    .expect("could not parse the Substrait filter in a synchronous fashion")?;
                let planner = Planner::new(schema);
                planner.optimize_expr(expr.clone()).map_err(|e| {
                    Error::invalid_input(
                        format!("Error optimizing substrait filter: {expr:?} ({e})"),
                        location!(),
                    )
                })
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
#[derive(Clone)]
pub struct Scanner {
    dataset: Arc<Dataset>,

    /// The projection plan for the scanner
    ///
    /// This includes
    /// - The physical projection that must be read from the dataset
    /// - Dynamic expressions that are evaluated after the physical projection
    /// - The names of the output columns
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
    /// If there are multiple columns the results will first be ordered
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

    /// If true, the scanner will emit deleted rows
    include_deleted_rows: bool,

    /// If set, this callback will be called after the scan with summary statistics
    scan_stats_callback: Option<ExecutionStatsCallback>,

    /// Whether the result returned by the scanner must be of the size of the batch_size.
    /// By default, it is false.
    /// Mainly, if the result is returned strictly according to the batch_size,
    /// batching and waiting are required, and the performance will decrease.
    strict_batch_size: bool,

    /// File reader options to use when reading data files.
    file_reader_options: Option<FileReaderOptions>,

    // Legacy fields to help migrate some old projection behavior to new behavior
    //
    // There are two behaviors we are moving away from:
    //
    // First, the old behavior used methods like with_row_id and with_row_addr to add
    // "system" columns.  The new behavior is to specify them in the projection like any
    // other column.  The only difference between a system column and a regular column is
    // that system columns are not returned in the schema and are not returned by default
    // (i.e. "SELECT *")
    //
    // Second, the old behavior would _always_ add the _score or _distance columns to the
    // output and there was no way for the user to opt out.  The new behavior treats the
    // _score and _distance as regular output columns of the "search table function".  If
    // the user does not specify a projection (i.e. "SELECT *") then we will add the _score
    // and _distance columns to the end.  If the user does specify a projection then they
    // must request those columns for them to show up.
    //
    // --------------------------------------------------------------------------
    /// Whether the user wants the row id on top of the projection, will always come last
    /// except possibly before _rowaddr
    legacy_with_row_id: bool,
    /// Whether the user wants the row address on top of the projection, will always come last
    legacy_with_row_addr: bool,
    /// Whether the user explicitly requested a projection.  If they did then we will warn them
    /// if they do not specify _score / _distance unless legacy_projection_behavior is set to false
    explicit_projection: bool,
    /// Whether the user wants to use the legacy projection behavior.
    autoproject_scoring_columns: bool,
}

fn escape_column_name(name: &str) -> String {
    name.split('.')
        .map(|s| format!("`{}`", s))
        .collect::<Vec<_>>()
        .join(".")
}

/// Represents a user-requested take operation
#[derive(Debug, Clone)]
pub enum TakeOperation {
    /// Take rows by row id
    RowIds(Vec<u64>),
    /// Take rows by row address
    RowAddrs(Vec<u64>),
    /// Take rows by row offset
    ///
    /// The row offset is the offset of the row in the dataset.  This can
    /// be converted to row addresses using the fragment sizes.
    RowOffsets(Vec<u64>),
}

impl TakeOperation {
    fn extract_u64_list(list: &[Expr]) -> Option<Vec<u64>> {
        let mut u64s = Vec::with_capacity(list.len());
        for expr in list {
            if let Expr::Literal(lit, _) = expr {
                if let Some(ScalarValue::UInt64(Some(val))) =
                    safe_coerce_scalar(lit, &DataType::UInt64)
                {
                    u64s.push(val);
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }
        Some(u64s)
    }

    fn merge(self, other: Self) -> Option<Self> {
        match (self, other) {
            (Self::RowIds(mut left), Self::RowIds(right)) => {
                left.extend(right);
                Some(Self::RowIds(left))
            }
            (Self::RowAddrs(mut left), Self::RowAddrs(right)) => {
                left.extend(right);
                Some(Self::RowAddrs(left))
            }
            (Self::RowOffsets(mut left), Self::RowOffsets(right)) => {
                left.extend(right);
                Some(Self::RowOffsets(left))
            }
            _ => None,
        }
    }

    /// Attempts to create a take operation from an expression.  This will succeed if the expression
    /// has one of the following forms:
    ///  - `_rowid = 10`
    ///  - `_rowid = 10 OR _rowid = 20 OR _rowid = 30`
    ///  - `_rowid IN (10, 20, 30)`
    ///  - `_rowaddr = 10`
    ///  - `_rowaddr = 10 OR _rowaddr = 20 OR _rowaddr = 30`
    ///  - `_rowaddr IN (10, 20, 30)`
    ///  - `_rowoffset = 10`
    ///  - `_rowoffset = 10 OR _rowoffset = 20 OR _rowoffset = 30`
    ///  - `_rowoffset IN (10, 20, 30)`
    ///
    /// The _rowid / _rowaddr / _rowoffset determine if we are taking by row id, address, or offset.
    ///
    /// If a take expression is combined with some other filter via an AND then the remainder will be
    /// returned as well.  For example, `_rowid = 10` will return (take_op, None) and
    /// `_rowid = 10 AND x > 70` will return (take_op, Some(x > 70)).
    fn try_from_expr(expr: &Expr) -> Option<(Self, Option<Expr>)> {
        if let Expr::BinaryExpr(binary) = expr {
            match binary.op {
                datafusion_expr::Operator::And => {
                    let left_take = Self::try_from_expr(&binary.left);
                    let right_take = Self::try_from_expr(&binary.right);
                    match (left_take, right_take) {
                        (Some(_), Some(_)) => {
                            // This is something like...
                            //
                            // _rowid = 10 AND _rowid = 20
                            //
                            // ...which is kind of nonsensical.  Better to just return None.
                            return None;
                        }
                        (Some((left_op, left_rem)), None) => {
                            let remainder = match left_rem {
                                // If there is a remainder on the left side we combine it.  This _should_
                                // be something like converting (_rowid = 10 AND x > 70) AND y > 80
                                // to (_rowid = 10) AND (x > 70 AND y > 80) which should be valid
                                Some(expr) => Expr::and(expr, binary.right.as_ref().clone()),
                                None => binary.right.as_ref().clone(),
                            };
                            return Some((left_op, Some(remainder)));
                        }
                        (None, Some((right_op, right_rem))) => {
                            let remainder = match right_rem {
                                Some(expr) => Expr::and(expr, binary.left.as_ref().clone()),
                                None => binary.left.as_ref().clone(),
                            };
                            return Some((right_op, Some(remainder)));
                        }
                        (None, None) => {
                            return None;
                        }
                    }
                }
                datafusion_expr::Operator::Eq => {
                    // Check for _rowid = literal
                    if let (Expr::Column(col), Expr::Literal(lit, _)) =
                        (binary.left.as_ref(), binary.right.as_ref())
                    {
                        if let Some(ScalarValue::UInt64(Some(val))) =
                            safe_coerce_scalar(lit, &DataType::UInt64)
                        {
                            if col.name == ROW_ID {
                                return Some((Self::RowIds(vec![val]), None));
                            } else if col.name == ROW_ADDR {
                                return Some((Self::RowAddrs(vec![val]), None));
                            } else if col.name == ROW_OFFSET {
                                return Some((Self::RowOffsets(vec![val]), None));
                            }
                        }
                    }
                }
                datafusion_expr::Operator::Or => {
                    let left_take = Self::try_from_expr(&binary.left);
                    let right_take = Self::try_from_expr(&binary.right);
                    if let (Some(left), Some(right)) = (left_take, right_take) {
                        if left.1.is_some() || right.1.is_some() {
                            // This would be something like...
                            //
                            // (_rowid = 10 AND x > 70) OR _rowid = 20
                            //
                            // I don't think it's correct to convert this into a take operation
                            // which would give us (_rowid = 10 OR _rowid = 20) AND x > 70
                            return None;
                        }
                        return left.0.merge(right.0).map(|op| (op, None));
                    }
                }
                _ => {}
            }
        } else if let Expr::InList(in_expr) = expr {
            if let Expr::Column(col) = in_expr.expr.as_ref() {
                if let Some(u64s) = Self::extract_u64_list(&in_expr.list) {
                    if col.name == ROW_ID {
                        return Some((Self::RowIds(u64s), None));
                    } else if col.name == ROW_ADDR {
                        return Some((Self::RowAddrs(u64s), None));
                    } else if col.name == ROW_OFFSET {
                        return Some((Self::RowOffsets(u64s), None));
                    }
                }
            }
        }
        None
    }
}

impl Scanner {
    pub fn new(dataset: Arc<Dataset>) -> Self {
        let projection_plan = ProjectionPlan::full(dataset.clone()).unwrap();
        let file_reader_options = dataset.file_reader_options.clone();
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
            ordered: true,
            fragments: None,
            fast_search: false,
            use_scalar_index: true,
            include_deleted_rows: false,
            scan_stats_callback: None,
            strict_batch_size: false,
            file_reader_options,
            legacy_with_row_addr: false,
            legacy_with_row_id: false,
            explicit_projection: false,
            autoproject_scoring_columns: true,
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

    /// Empty Projection (useful for count queries)
    ///
    /// The row_address will be scanned (no I/O required) but not included in the output
    pub fn empty_project(&mut self) -> Result<&mut Self> {
        self.project(&[] as &[&str])
    }

    /// Projection.
    ///
    /// Only select the specified columns. If not specified, all columns will be scanned.
    pub fn project<T: AsRef<str>>(&mut self, columns: &[T]) -> Result<&mut Self> {
        let transformed_columns: Vec<(&str, String)> = columns
            .iter()
            .map(|c| (c.as_ref(), escape_column_name(c.as_ref())))
            .collect();

        self.project_with_transform(&transformed_columns)
    }

    /// Projection with transform
    ///
    /// Only select the specified columns with the given transform.
    pub fn project_with_transform(
        &mut self,
        columns: &[(impl AsRef<str>, impl AsRef<str>)],
    ) -> Result<&mut Self> {
        self.explicit_projection = true;
        self.projection_plan = ProjectionPlan::from_expressions(self.dataset.clone(), columns)?;
        if self.legacy_with_row_id {
            self.projection_plan.include_row_id();
        }
        if self.legacy_with_row_addr {
            self.projection_plan.include_row_addr();
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

    /// Set the callback to be called after the scan with summary statistics
    pub fn scan_stats_callback(&mut self, callback: ExecutionStatsCallback) -> &mut Self {
        self.scan_stats_callback = Some(callback);
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
        let fields = query.columns();
        if !fields.is_empty() {
            for field in fields.iter() {
                if self.dataset.schema().field(field).is_none() {
                    return Err(Error::invalid_input(
                        format!("Column {} not found", field),
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

    pub fn filter_expr(&mut self, filter: Expr) -> &mut Self {
        self.filter = Some(LanceFilter::Datafusion(filter));
        self
    }

    /// Set the batch size.
    pub fn batch_size(&mut self, batch_size: usize) -> &mut Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Include deleted rows
    ///
    /// These are rows that have been deleted from the dataset but are still present in the
    /// underlying storage.  These rows will have the `_rowid` column set to NULL.  The other columns
    /// (include _rowaddr) will be set to their deleted values.
    ///
    /// This can be useful for generating aligned fragments or debugging
    ///
    /// Note: when entire fragments are deleted, the scanner will not emit any rows for that fragment
    /// since the fragment is no longer present in the dataset.
    pub fn include_deleted_rows(&mut self) -> &mut Self {
        self.include_deleted_rows = true;
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
    /// Ignored in v2 and newer format
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

    /// Set whether to use strict batch size.
    ///
    /// If this is true then output batches (except the last batch) will have exactly `batch_size` rows.
    /// By default, this is False and output batches are allowed to have fewer than `batch_size` rows
    /// Setting this to True will require us to merge batches, incurring a data copy, for a minor performance
    /// penalty.
    pub fn strict_batch_size(&mut self, strict_batch_size: bool) -> &mut Self {
        self.strict_batch_size = strict_batch_size;
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
    /// the query can be a Float16Array, Float32Array, Float64Array, UInt8Array,
    /// or a ListArray/FixedSizeListArray of the above types.
    pub fn nearest(&mut self, column: &str, q: &dyn Array, k: usize) -> Result<&mut Self> {
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
        let (vector_type, element_type) = get_vector_type(self.dataset.schema(), column)?;
        let dim = get_vector_dim(self.dataset.schema(), column)?;

        let q = match q.data_type() {
            DataType::List(_) | DataType::FixedSizeList(_, _) => {
                if !matches!(vector_type, DataType::List(_)) {
                    return Err(Error::invalid_input(
                        format!(
                            "Query is multivector but column {}({})is not multivector",
                            column, vector_type,
                        ),
                        location!(),
                    ));
                }

                if let Some(list_array) = q.as_list_opt::<i32>() {
                    for i in 0..list_array.len() {
                        let vec = list_array.value(i);
                        if vec.len() != dim {
                            return Err(Error::invalid_input(
                                format!(
                                    "query dim({}) doesn't match the column {} vector dim({})",
                                    vec.len(),
                                    column,
                                    dim,
                                ),
                                location!(),
                            ));
                        }
                    }
                    list_array.values().clone()
                } else {
                    let fsl = q.as_fixed_size_list();
                    if fsl.value_length() as usize != dim {
                        return Err(Error::invalid_input(
                            format!(
                                "query dim({}) doesn't match the column {} vector dim({})",
                                fsl.value_length(),
                                column,
                                dim,
                            ),
                            location!(),
                        ));
                    }
                    fsl.values().clone()
                }
            }
            _ => {
                if q.len() != dim {
                    return Err(Error::invalid_input(
                        format!(
                            "query dim({}) doesn't match the column {} vector dim({})",
                            q.len(),
                            column,
                            dim,
                        ),
                        location!(),
                    ));
                }
                q.slice(0, q.len())
            }
        };

        let key = match element_type {
            dt if dt == *q.data_type() => q,
            dt if dt.is_floating() => coerce_float_vector(
                q.as_any().downcast_ref::<Float32Array>().unwrap(),
                FloatType::try_from(&dt)?,
            )?,
            _ => {
                return Err(Error::invalid_input(
                    format!(
                        "Column {} has element type {} and the query vector is {}",
                        column,
                        element_type,
                        q.data_type(),
                    ),
                    location!(),
                ));
            }
        };

        self.nearest = Some(Query {
            column: column.to_string(),
            key,
            k,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 20,
            maximum_nprobes: None,
            ef: None,
            refine_factor: None,
            metric_type: MetricType::L2,
            use_index: true,
        });
        Ok(self)
    }

    #[cfg(test)]
    fn nearest_mut(&mut self) -> Option<&mut Query> {
        self.nearest.as_mut()
    }

    /// Set the distance thresholds for the nearest neighbor search.
    pub fn distance_range(
        &mut self,
        lower_bound: Option<f32>,
        upper_bound: Option<f32>,
    ) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.lower_bound = lower_bound;
            q.upper_bound = upper_bound;
        }
        self
    }

    /// Configures how many partititions will be searched in the vector index.
    ///
    /// This method is a convenience method that sets both [Self::minimum_nprobes] and
    /// [Self::maximum_nprobes] to the same value.
    pub fn nprobs(&mut self, n: usize) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.minimum_nprobes = n;
            q.maximum_nprobes = Some(n);
        } else {
            log::warn!("nprobes is not set because nearest has not been called yet");
        }
        self
    }

    /// Configures the minimum number of partitions to search in the vector index.
    ///
    /// If we have found k matching results after searching this many partitions then
    /// the search will stop.  Increasing this number can increase recall but will increase
    /// latency on all queries.
    pub fn minimum_nprobes(&mut self, n: usize) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.minimum_nprobes = n;
        } else {
            log::warn!("minimum_nprobes is not set because nearest has not been called yet");
        }
        self
    }

    /// Configures the maximum number of partitions to search in the vector index.
    ///
    /// These partitions will only be searched if we have not found `k` results after
    /// searching the minimum number of partitions.  Setting this to None (the default)
    /// will search all partitions if needed.
    ///
    /// This setting only takes effect when a prefilter is in place.  In that case we
    /// can spend more effort to try and find results when the filter is highly selective.
    ///
    /// If there is no prefilter, or the results are not highly selective, this value will
    /// have no effect.
    pub fn maximum_nprobes(&mut self, n: usize) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.maximum_nprobes = Some(n);
        } else {
            log::warn!("maximum_nprobes is not set because nearest has not been called yet");
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
        self.projection_plan.include_row_id(); // fast search requires _rowid
        self
    }

    /// Apply a refine step to the vector search.
    ///
    /// A refine improves query accuracy but also makes search slower, by reading extra elements
    /// and using the original vector values to re-rank the distances.
    ///
    /// * `factor` - the factor of extra elements to read.  For example, if factor is 2, then
    ///   the search will read 2x more elements than the requested k before performing
    ///   the re-ranking. Note: even if the factor is 1, the  results will still be
    ///   re-ranked without fetching additional elements.
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
        self.legacy_with_row_id = true;
        self.projection_plan.include_row_id();
        self
    }

    /// Instruct the scanner to return the `_rowaddr` meta column from the dataset.
    pub fn with_row_address(&mut self) -> &mut Self {
        self.legacy_with_row_addr = true;
        self.projection_plan.include_row_addr();
        self
    }

    /// Instruct the scanner to disable automatic projection of scoring columns
    ///
    /// In the future, this will be the default behavior.  This method is useful for
    /// opting in to the new behavior early to avoid breaking changes (and a warning
    /// message)
    ///
    /// Once the default switches, the old autoprojection behavior will be removed.
    ///
    /// The autoprojection behavior (current default) includes the _score or _distance
    /// column even if a projection is manually specified with `[project]` or
    /// `[project_with_transform]`.
    ///
    /// The new behavior will only include the _score or _distance column if no projection
    /// is specified or if the user explicitly includes the _score or _distance column
    /// in the projection.
    pub fn disable_scoring_autoprojection(&mut self) -> &mut Self {
        self.autoproject_scoring_columns = false;
        self
    }

    /// Set the file reader options to use when reading data files.
    pub fn with_file_reader_options(&mut self, options: FileReaderOptions) -> &mut Self {
        self.file_reader_options = Some(options);
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
            let filter_schema = self.filterable_schema()?;
            Ok(Some(filter.to_datafusion(
                self.dataset.schema(),
                filter_schema.as_ref(),
            )?))
        } else {
            Ok(None)
        }
    }

    fn add_extra_columns(&self, schema: Schema) -> Result<Schema> {
        let mut extra_columns = vec![ArrowField::new(ROW_OFFSET, DataType::UInt64, true)];

        if self.nearest.as_ref().is_some() {
            extra_columns.push(ArrowField::new(DIST_COL, DataType::Float32, true));
        };

        if self.full_text_query.is_some() {
            extra_columns.push(ArrowField::new(SCORE_COL, DataType::Float32, true));
        }

        schema.merge(&ArrowSchema::new(extra_columns))
    }

    /// The full schema available to filters
    ///
    /// This is the schema of the dataset, any metadata columns like _rowid or _rowaddr
    /// and any extra columns like _distance or _score
    fn filterable_schema(&self) -> Result<Arc<Schema>> {
        let base_schema = Projection::full(self.dataset.clone())
            .with_row_id()
            .with_row_addr()
            .to_schema();

        Ok(Arc::new(self.add_extra_columns(base_schema)?))
    }

    /// This takes the current output, and the user's requested projection, and calculates the
    /// final projection expression.
    ///
    /// This final expression may reorder columns, drop columns, or calculate new columns
    pub(crate) fn calculate_final_projection(
        &self,
        current_schema: &ArrowSchema,
    ) -> Result<Vec<(Arc<dyn PhysicalExpr>, String)>> {
        // Select the columns from the output schema based on the user's projection (or the list
        // of all available columns if the user did not specify a projection)
        let mut output_expr = self.projection_plan.to_physical_exprs(current_schema)?;

        // Make sure _distance and _score are _always_ in the output unless user has opted out of the legacy
        // projection behavior
        if self.autoproject_scoring_columns {
            if self.nearest.is_some() && output_expr.iter().all(|(_, name)| name != DIST_COL) {
                if self.explicit_projection {
                    log::warn!("Deprecation warning, this behavior will change in the future. This search specified output columns but did not include `_distance`.  Currently the `_distance` column will be included.  In the future it will not.  Call `disable_scoring_autoprojection` to to adopt the future behavior and avoid this warning");
                }
                let vector_expr = expressions::col(DIST_COL, current_schema)?;
                output_expr.push((vector_expr, DIST_COL.to_string()));
            }
            if self.full_text_query.is_some()
                && output_expr.iter().all(|(_, name)| name != SCORE_COL)
            {
                if self.explicit_projection {
                    log::warn!("Deprecation warning, this behavior will change in the future. This search specified output columns but did not include `_score`.  Currently the `_score` column will be included.  In the future it will not.  Call `disable_scoring_autoprojection` to adopt the future behavior and avoid this warning");
                }
                let score_expr = expressions::col(SCORE_COL, current_schema)?;
                output_expr.push((score_expr, SCORE_COL.to_string()));
            }
        }

        if self.legacy_with_row_id {
            let row_id_pos = output_expr
                .iter()
                .position(|(_, name)| name == ROW_ID)
                .ok_or_else(|| Error::Internal {
                    message:
                        "user specified with_row_id but the _rowid column was not in the output"
                            .to_string(),
                    location: location!(),
                })?;
            if row_id_pos != output_expr.len() - 1 {
                // Row id is not last column.  Need to rotate it to the last spot.
                let row_id_expr = output_expr.remove(row_id_pos);
                output_expr.push(row_id_expr);
            }
        }

        if self.legacy_with_row_addr {
            let row_addr_pos = output_expr.iter().position(|(_, name)| name == ROW_ADDR).ok_or_else(|| {
                Error::Internal {
                    message: "user specified with_row_address but the _rowaddr column was not in the output".to_string(),
                    location: location!(),
                }
            })?;
            if row_addr_pos != output_expr.len() - 1 {
                // Row addr is not last column.  Need to rotate it to the last spot.
                let row_addr_expr = output_expr.remove(row_addr_pos);
                output_expr.push(row_addr_expr);
            }
        }

        Ok(output_expr)
    }

    /// Create a stream from the Scanner.
    #[instrument(skip_all)]
    pub fn try_into_stream(&self) -> BoxFuture<'_, Result<DatasetRecordBatchStream>> {
        // Future intentionally boxed here to avoid large futures on the stack
        async move {
            let plan = self.create_plan().await?;

            Ok(DatasetRecordBatchStream::new(execute_plan(
                plan,
                LanceExecutionOptions {
                    batch_size: self.batch_size,
                    execution_stats_callback: self.scan_stats_callback.clone(),
                    ..Default::default()
                },
            )?))
        }
        .boxed()
    }

    pub(crate) async fn try_into_dfstream(
        &self,
        mut options: LanceExecutionOptions,
    ) -> Result<SendableRecordBatchStream> {
        let plan = self.create_plan().await?;

        // Use the scan stats callback if the user didn't set an execution stats callback
        if options.execution_stats_callback.is_none() {
            options.execution_stats_callback = self.scan_stats_callback.clone();
        }

        execute_plan(plan, options)
    }

    pub async fn try_into_batch(&self) -> Result<RecordBatch> {
        let stream = self.try_into_stream().await?;
        let schema = stream.schema();
        let batches = stream.try_collect::<Vec<_>>().await?;
        Ok(concat_batches(&schema, &batches)?)
    }

    fn create_count_plan(&self) -> BoxFuture<'_, Result<Arc<dyn ExecutionPlan>>> {
        // Future intentionally boxed here to avoid large futures on the stack
        async move {
            if self.projection_plan.physical_projection.is_empty() {
                return Err(Error::invalid_input("count_rows called but with_row_id is false".to_string(), location!()));
            }
            if !self.projection_plan.physical_projection.is_metadata_only() {
                let physical_schema = self.projection_plan.physical_projection.to_schema();
                let columns: Vec<&str> = physical_schema.fields
                .iter()
                .map(|field| field.name.as_str())
                .collect();

                let msg = format!(
                    "count_rows should not be called on a plan selecting columns. selected columns: [{}]",
                    columns.join(", ")
                );

                return Err(Error::invalid_input(msg, location!()));
            }

            if self.limit.is_some() || self.offset.is_some() {
                log::warn!(
                    "count_rows called with limit or offset which could have surprising results"
                );
            }

            let plan = self.create_plan().await?;
            // Datafusion interprets COUNT(*) as COUNT(1)
            let one = Arc::new(Literal::new(ScalarValue::UInt8(Some(1))));

            let input_phy_exprs: &[Arc<dyn PhysicalExpr>] = &[one];
            let schema = plan.schema();

            let mut builder = AggregateExprBuilder::new(count_udaf(), input_phy_exprs.to_vec());
            builder = builder.schema(schema);
            builder = builder.alias("count_rows".to_string());

            let count_expr = builder.build()?;

            let plan_schema = plan.schema();
            Ok(Arc::new(AggregateExec::try_new(
                AggregateMode::Single,
                PhysicalGroupBy::new_single(Vec::new()),
                vec![Arc::new(count_expr)],
                vec![None],
                plan,
                plan_schema,
            )?) as Arc<dyn ExecutionPlan>)
        }
        .boxed()
    }

    /// Scan and return the number of matching rows
    ///
    /// Note: calling [`Dataset::count_rows`] can be more efficient than calling this method
    /// especially if there is no filter.
    #[instrument(skip_all)]
    pub fn count_rows(&self) -> BoxFuture<'_, Result<u64>> {
        // Future intentionally boxed here to avoid large futures on the stack
        async move {
            let count_plan = self.create_count_plan().await?;
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
        .boxed()
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
                if field.is_blob() {
                    // By default, blobs are loaded as descriptions, and so should be early
                    //
                    // TODO: Once we make blob handling configurable, we should use the blob
                    // handling setting here.
                    return true;
                }

                let byte_width = field.data_type().byte_width_opt();
                let is_cloud = self.dataset.object_store().is_cloud();
                if is_cloud {
                    byte_width.is_some_and(|bw| bw < 1000)
                } else {
                    byte_width.is_some_and(|bw| bw < 10)
                }
            }
        }
    }

    // If we are going to filter on `filter_plan`, then which columns are so small it is
    // cheaper to read the entire column and filter in memory.
    //
    // Note: only add columns that we actually need to read
    fn calc_eager_projection(
        &self,
        filter_plan: &FilterPlan,
        desired_projection: &Projection,
    ) -> Result<Projection> {
        // Note: We use all_columns and not refine_columns here.  If a column is covered by an index but
        // the user has requested it, then we do not use it for late materialization.
        //
        // Either that column is covered by an exact filter (e.g. string with bitmap/btree) and there is no
        // need for late materialization or that column is covered by an inexact filter (e.g. ngram) in which
        // case we are going to load the column anyways for the recheck.
        let filter_columns = filter_plan.all_columns();

        let filter_schema = self
            .dataset
            .empty_projection()
            .union_columns(filter_columns, OnMissing::Error)?
            .into_schema();
        if filter_schema.fields.iter().any(|f| !f.is_default_storage()) {
            return Err(Error::NotSupported {
                source: "non-default storage columns cannot be used as filters".into(),
                location: location!(),
            });
        }

        // Start with the desired fields
        Ok(desired_projection
            .clone()
            // Subtract columns that are expensive
            .subtract_predicate(|f| !self.is_early_field(f))
            // Add back columns that we need for filtering
            .union_schema(&filter_schema))
    }

    fn validate_options(&self) -> Result<()> {
        if self.include_deleted_rows && !self.projection_plan.physical_projection.with_row_id {
            return Err(Error::InvalidInput {
                source: "include_deleted_rows is set but with_row_id is false".into(),
                location: location!(),
            });
        }

        Ok(())
    }

    async fn create_filter_plan(&self, use_scalar_index: bool) -> Result<FilterPlan> {
        let filter_schema = self.filterable_schema()?;
        let planner = Planner::new(Arc::new(filter_schema.as_ref().into()));

        if let Some(filter) = self.filter.as_ref() {
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
                    Ok(planner.create_filter_plan(filter.clone(), &index_info, false)?)
                } else {
                    Ok(filter_plan)
                }
            } else {
                Ok(filter_plan)
            }
        } else {
            Ok(FilterPlan::default())
        }
    }

    async fn get_scan_range(&self, filter_plan: &FilterPlan) -> Result<Option<Range<u64>>> {
        if filter_plan.has_any_filter() {
            // If there is a filter we can't pushdown limit / offset
            Ok(None)
        } else {
            match (self.limit, self.offset) {
                (None, None) => Ok(None),
                (Some(limit), None) => {
                    let num_rows = self.dataset.count_all_rows().await? as i64;
                    Ok(Some(0..limit.min(num_rows) as u64))
                }
                (None, Some(offset)) => {
                    let num_rows = self.dataset.count_all_rows().await? as i64;
                    Ok(Some(offset.min(num_rows) as u64..num_rows as u64))
                }
                (Some(limit), Some(offset)) => {
                    let num_rows = self.dataset.count_all_rows().await? as i64;
                    Ok(Some(
                        offset.min(num_rows) as u64..(offset + limit).min(num_rows) as u64,
                    ))
                }
            }
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
        log::trace!("creating scanner plan");
        self.validate_options()?;

        // Scalar indices are only used when prefiltering
        let use_scalar_index = self.use_scalar_index && (self.prefilter || self.nearest.is_none());
        let mut filter_plan = self.create_filter_plan(use_scalar_index).await?;

        let mut use_limit_node = true;
        // Stage 1: source (either an (K|A)NN search, full text search or or a (full|indexed) scan)
        let mut plan: Arc<dyn ExecutionPlan> = match (&self.nearest, &self.full_text_query) {
            (Some(_), None) => self.vector_search_source(&mut filter_plan).await?,
            (None, Some(query)) => self.fts_search_source(&mut filter_plan, query).await?,
            (None, None) => {
                if self.projection_plan.has_output_cols()
                    && self.projection_plan.physical_projection.is_empty()
                {
                    // This means the user is doing something like `SELECT 1 AS foo`.  We don't support this and
                    // I'm not sure we should.  Users should use a full SQL API to do something like this.
                    //
                    // It's also possible we get here from `SELECT does_not_exist`

                    // Note: even though we are just going to return an error we still want to calculate the
                    // final projection here.  This lets us distinguish between a user doing something like:
                    //
                    // SELECT 1 FROM t (not supported error)
                    // SELECT non_existent_column FROM t (column not found error)
                    let output_expr = self.calculate_final_projection(&ArrowSchema::empty())?;
                    return Err(Error::NotSupported {
                        source: format!("Scans must request at least one column.  Received only dynamic expressions: {:?}", output_expr).into(),
                        location: location!(),
                    });
                }

                let take_op = filter_plan
                    .full_expr
                    .as_ref()
                    .and_then(TakeOperation::try_from_expr);
                if let Some((take_op, remainder)) = take_op {
                    // If there is any remainder use it as the filter (we don't even try and combine an indexed
                    // search on the filter with a take as that seems excessive)
                    filter_plan = remainder
                        .map(FilterPlan::new_refine_only)
                        .unwrap_or(FilterPlan::default());
                    self.take_source(take_op).await?
                } else {
                    let planned_read = self.filtered_read_source(&mut filter_plan).await?;
                    if planned_read.limit_pushed_down {
                        use_limit_node = false;
                    }
                    if planned_read.filter_pushed_down {
                        filter_plan = FilterPlan::default();
                    }
                    planned_read.plan
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
        // Calculate the schema needed for the filter and ordering.
        let mut pre_filter_projection = self.dataset.empty_projection();

        // We may need to take filter columns if we are going to refine
        // an indexed scan.
        if filter_plan.has_refine() {
            // It's ok for some filter columns to be missing (e.g. _rowid)
            pre_filter_projection = pre_filter_projection
                .union_columns(filter_plan.refine_columns(), OnMissing::Ignore)?;
        }

        // TODO: Does it always make sense to take the ordering columns here?  If there is a filter then
        // maybe we wait until after the filter to take the ordering columns?  Maybe it would be better to
        // grab the ordering column in the initial scan (if it is eager) and if it isn't then we should
        // take it after the filtering phase, if any (we already have a take there).
        if let Some(ordering) = &self.ordering {
            pre_filter_projection = pre_filter_projection.union_columns(
                ordering.iter().map(|col| &col.column_name),
                OnMissing::Error,
            )?;
        }

        plan = self.take(plan, pre_filter_projection)?;

        // Stage 2: filter
        if let Some(refine_expr) = filter_plan.refine_expr {
            // We create a new planner specific to the node's schema, since
            // physical expressions reference column by index rather than by name.
            plan = Arc::new(LanceFilterExec::try_new(refine_expr, plan)?);
        }

        // Stage 3: sort
        if let Some(ordering) = &self.ordering {
            let ordering_columns = ordering.iter().map(|col| &col.column_name);
            let projection_with_ordering = self
                .dataset
                .empty_projection()
                .union_columns(ordering_columns, OnMissing::Error)?;
            // We haven't loaded the sort column yet so take it now
            plan = self.take(plan, projection_with_ordering)?;
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
            plan = Arc::new(SortExec::new(LexOrdering::new(col_exprs), plan));
        }

        // Stage 4: limit / offset
        if use_limit_node && (self.limit.unwrap_or(0) > 0 || self.offset.is_some()) {
            plan = self.limit_node(plan);
        }

        // Stage 5: take remaining columns required for projection
        plan = self.take(plan, self.projection_plan.physical_projection.clone())?;

        // Stage 6: if requested, add the row offset column
        if self.projection_plan.must_add_row_offset {
            plan = Arc::new(AddRowOffsetExec::try_new(plan, self.dataset.clone()).await?);
        }

        // Stage 7: final projection
        let final_projection = self.calculate_final_projection(plan.schema().as_ref())?;

        plan = Arc::new(DFProjectionExec::try_new(final_projection, plan)?);

        // Stage 8: If requested, apply a strict batch size to the final output
        if self.strict_batch_size {
            plan = Arc::new(StrictBatchSizeExec::new(plan, self.get_batch_size()));
        }

        let optimizer = get_physical_optimizer();
        let options = Default::default();
        for rule in optimizer.rules {
            plan = rule.optimize(plan, &options)?;
        }

        Ok(plan)
    }

    // Helper function for filtered_read
    //
    // Do not call this directly, use filtered_read instead
    //
    // First return value is the plan, second is whether the limit was pushed down
    async fn legacy_filtered_read(
        &self,
        filter_plan: &FilterPlan,
        projection: Projection,
        make_deletions_null: bool,
        fragments: Option<Arc<Vec<Fragment>>>,
        scan_range: Option<Range<u64>>,
        is_prefilter: bool,
    ) -> Result<PlannedFilteredScan> {
        let fragments = fragments.unwrap_or(self.dataset.fragments().clone());
        let mut filter_pushed_down = false;

        let plan: Arc<dyn ExecutionPlan> = if filter_plan.has_index_query() {
            if self.include_deleted_rows {
                return Err(Error::InvalidInput {
                    source: "Cannot include deleted rows in a scalar indexed scan".into(),
                    location: location!(),
                });
            }
            self.scalar_indexed_scan(projection, filter_plan, fragments)
                .await
        } else if !is_prefilter
            && filter_plan.has_refine()
            && self.batch_size.is_none()
            && self.use_stats
        {
            filter_pushed_down = true;
            self.pushdown_scan(false, filter_plan)
        } else {
            let ordered = if self.ordering.is_some() || self.nearest.is_some() {
                // If we are sorting the results there is no need to scan in order
                false
            } else {
                self.ordered
            };

            let projection = if let Some(refine_expr) = filter_plan.refine_expr.as_ref() {
                if is_prefilter {
                    let refine_cols = Planner::column_names_in_expr(refine_expr);
                    projection.union_columns(refine_cols, OnMissing::Error)?
                } else {
                    projection
                }
            } else {
                projection
            };

            // Can't push down limit for legacy scan if there is a refine step
            let scan_range = if filter_plan.has_refine() {
                None
            } else {
                scan_range
            };

            let scan = self.scan_fragments(
                projection.with_row_id,
                self.projection_plan.physical_projection.with_row_addr,
                make_deletions_null,
                Arc::new(projection.to_bare_schema()),
                fragments,
                scan_range,
                ordered,
            );

            if filter_plan.has_refine() && is_prefilter {
                Ok(Arc::new(LanceFilterExec::try_new(
                    filter_plan.refine_expr.clone().unwrap(),
                    scan,
                )?) as Arc<dyn ExecutionPlan>)
            } else {
                Ok(scan)
            }
        }?;
        Ok(PlannedFilteredScan {
            plan,
            limit_pushed_down: false,
            filter_pushed_down,
        })
    }

    // Helper function for filtered_read
    //
    // Do not call this directly, use filtered_read instead
    async fn new_filtered_read(
        &self,
        filter_plan: &FilterPlan,
        projection: Projection,
        make_deletions_null: bool,
        fragments: Option<Arc<Vec<Fragment>>>,
        scan_range: Option<Range<u64>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut read_options = FilteredReadOptions::basic_full_read(&self.dataset)
            .with_filter_plan(filter_plan.clone())
            .with_projection(projection);

        if let Some(fragments) = fragments {
            read_options = read_options.with_fragments(fragments);
        }

        if let Some(scan_range) = scan_range {
            read_options = read_options.with_scan_range_before_filter(scan_range)?;
        }

        if let Some(batch_size) = self.batch_size {
            read_options = read_options.with_batch_size(batch_size as u32);
        }

        if let Some(fragment_readahead) = self.fragment_readahead {
            read_options = read_options.with_fragment_readahead(fragment_readahead);
        }

        if make_deletions_null {
            read_options = read_options.with_deleted_rows()?;
        }

        let index_input = filter_plan.index_query.clone().map(|index_query| {
            Arc::new(ScalarIndexExec::new(self.dataset.clone(), index_query))
                as Arc<dyn ExecutionPlan>
        });

        Ok(Arc::new(FilteredReadExec::try_new(
            self.dataset.clone(),
            read_options,
            index_input,
        )?))
    }

    // Helper function for filtered read
    //
    // Delegates to legacy or new filtered read based on dataset storage version
    async fn filtered_read(
        &self,
        filter_plan: &FilterPlan,
        projection: Projection,
        make_deletions_null: bool,
        fragments: Option<Arc<Vec<Fragment>>>,
        scan_range: Option<Range<u64>>,
        is_prefilter: bool,
    ) -> Result<PlannedFilteredScan> {
        if self.dataset.is_legacy_storage() {
            self.legacy_filtered_read(
                filter_plan,
                projection,
                make_deletions_null,
                fragments,
                scan_range,
                is_prefilter,
            )
            .await
        } else {
            let limit_pushed_down = scan_range.is_some();
            let plan = self
                .new_filtered_read(
                    filter_plan,
                    projection,
                    make_deletions_null,
                    fragments,
                    scan_range,
                )
                .await?;
            Ok(PlannedFilteredScan {
                filter_pushed_down: true,
                limit_pushed_down,
                plan,
            })
        }
    }

    fn u64s_as_take_input(&self, u64s: Vec<u64>) -> Result<Arc<dyn ExecutionPlan>> {
        let row_ids = RowIdTreeMap::from_iter(u64s);
        let row_id_mask = RowIdMask::from_allowed(row_ids);
        let index_result = IndexExprResult::Exact(row_id_mask);
        let fragments_covered =
            RoaringBitmap::from_iter(self.dataset.fragments().iter().map(|f| f.id as u32));
        let batch = index_result.serialize_to_arrow(&fragments_covered)?;
        let stream = futures::stream::once(async move { Ok(batch) });
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            INDEX_EXPR_RESULT_SCHEMA.clone(),
            stream,
        ));
        Ok(Arc::new(OneShotExec::new(stream)))
    }

    async fn take_source(&self, take_op: TakeOperation) -> Result<Arc<dyn ExecutionPlan>> {
        // We generally assume that late materialization does not make sense for take operations
        // so we can just use the physical projection
        let projection = self.projection_plan.physical_projection.clone();

        let input = match take_op {
            TakeOperation::RowIds(ids) => self.u64s_as_take_input(ids),
            TakeOperation::RowAddrs(addrs) => self.u64s_as_take_input(addrs),
            TakeOperation::RowOffsets(offsets) => {
                let mut addrs =
                    row_offsets_to_row_addresses(self.dataset.as_ref(), &offsets).await?;
                addrs.retain(|addr| *addr != RowAddress::TOMBSTONE_ROW);
                self.u64s_as_take_input(addrs)
            }
        }?;

        Ok(Arc::new(FilteredReadExec::try_new(
            self.dataset.clone(),
            FilteredReadOptions::new(projection),
            Some(input),
        )?))
    }

    async fn filtered_read_source(
        &self,
        filter_plan: &mut FilterPlan,
    ) -> Result<PlannedFilteredScan> {
        log::trace!("source is a filtered read");
        let mut projection = if filter_plan.has_refine() {
            // If the filter plan has two steps (a scalar indexed portion and a refine portion) then
            // it makes sense to grab cheap columns during the first step to avoid taking them for
            // the second step.
            self.calc_eager_projection(filter_plan, &self.projection_plan.physical_projection)?
                .with_row_id()
        } else {
            // If the filter plan only has one step then we just do a filtered read of all the
            // columns that the user asked for.
            self.projection_plan.physical_projection.clone()
        };

        if projection.is_empty() {
            // If the user is not requesting any columns then we will scan the row address which
            // is cheap
            projection.with_row_addr = true;
        }

        let scan_range = if filter_plan.is_empty() {
            log::trace!("pushing scan_range into filtered_read");
            self.get_scan_range(filter_plan).await?
        } else {
            None
        };

        self.filtered_read(
            filter_plan,
            projection,
            self.include_deleted_rows,
            self.fragments.clone().map(Arc::new),
            scan_range,
            /*is_prefilter= */ false,
        )
        .await
    }

    async fn fts_search_source(
        &self,
        filter_plan: &mut FilterPlan,
        query: &FullTextSearchQuery,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        log::trace!("source is an fts search");
        if self.include_deleted_rows {
            return Err(Error::InvalidInput {
                source: "Cannot include deleted rows in an FTS search".into(),
                location: location!(),
            });
        }

        // The source is an FTS search
        if self.prefilter {
            // If we are prefiltering then the fts node will take care of the filter
            let source = self.fts(filter_plan, query).await?;
            *filter_plan = FilterPlan::default();
            Ok(source)
        } else {
            // If we are postfiltering then we can't use scalar indices for the filter
            // and will need to run the postfilter in memory
            filter_plan.make_refine_only();
            self.fts(&FilterPlan::default(), query).await
        }
    }

    async fn vector_search_source(
        &self,
        filter_plan: &mut FilterPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if self.include_deleted_rows {
            return Err(Error::InvalidInput {
                source: "Cannot include deleted rows in a nearest neighbor search".into(),
                location: location!(),
            });
        }

        if self.prefilter {
            log::trace!("source is a vector search (prefilter)");
            // If we are prefiltering then the ann / knn node will take care of the filter
            let source = self.vector_search(filter_plan).await?;
            *filter_plan = FilterPlan::default();
            Ok(source)
        } else {
            log::trace!("source is a vector search (postfilter)");
            // If we are postfiltering then we can't use scalar indices for the filter
            // and will need to run the postfilter in memory
            filter_plan.make_refine_only();
            self.vector_search(&FilterPlan::default()).await
        }
    }

    async fn fragments_covered_by_fts_leaf(
        &self,
        column: &str,
        accum: &mut RoaringBitmap,
    ) -> Result<bool> {
        let index = self
            .dataset
            .load_scalar_index(
                ScalarIndexCriteria::default()
                    .for_column(column)
                    .supports_fts(),
            )
            .await?
            .ok_or(Error::invalid_input(
                format!("Column {} has no inverted index", column),
                location!(),
            ))?;
        if let Some(fragmap) = &index.fragment_bitmap {
            *accum |= fragmap;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    #[async_recursion]
    async fn fragments_covered_by_fts_query_helper(
        &self,
        query: &FtsQuery,
        accum: &mut RoaringBitmap,
    ) -> Result<bool> {
        match query {
            FtsQuery::Match(match_query) => {
                self.fragments_covered_by_fts_leaf(
                    match_query.column.as_ref().ok_or(Error::invalid_input(
                        "the column must be specified in the query".to_string(),
                        location!(),
                    ))?,
                    accum,
                )
                .await
            }
            FtsQuery::Boost(boost) => Ok(self
                .fragments_covered_by_fts_query_helper(&boost.negative, accum)
                .await?
                & self
                    .fragments_covered_by_fts_query_helper(&boost.positive, accum)
                    .await?),
            FtsQuery::MultiMatch(multi_match) => {
                for mq in &multi_match.match_queries {
                    if !self
                        .fragments_covered_by_fts_leaf(
                            mq.column.as_ref().ok_or(Error::invalid_input(
                                "the column must be specified in the query".to_string(),
                                location!(),
                            ))?,
                            accum,
                        )
                        .await?
                    {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            FtsQuery::Phrase(phrase_query) => {
                self.fragments_covered_by_fts_leaf(
                    phrase_query.column.as_ref().ok_or(Error::invalid_input(
                        "the column must be specified in the query".to_string(),
                        location!(),
                    ))?,
                    accum,
                )
                .await
            }
            FtsQuery::Boolean(bool_query) => {
                for query in bool_query.must.iter() {
                    if !self
                        .fragments_covered_by_fts_query_helper(query, accum)
                        .await?
                    {
                        return Ok(false);
                    }
                }
                for query in &bool_query.should {
                    if !self
                        .fragments_covered_by_fts_query_helper(query, accum)
                        .await?
                    {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }

    async fn fragments_covered_by_fts_query(&self, query: &FtsQuery) -> Result<RoaringBitmap> {
        let all_fragments = self.get_fragments_as_bitmap();

        let mut referenced_fragments = RoaringBitmap::new();
        if !self
            .fragments_covered_by_fts_query_helper(query, &mut referenced_fragments)
            .await?
        {
            // One or more indices is missing the fragment bitmap, require all fragments in prefilter
            Ok(all_fragments)
        } else {
            // Fragments required for prefilter is intersection of index fragments and query fragments
            Ok(all_fragments & referenced_fragments)
        }
    }

    // Create an execution plan to do full text search
    async fn fts(
        &self,
        filter_plan: &FilterPlan,
        query: &FullTextSearchQuery,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let columns = query.columns();
        let mut params = query.params();
        if params.limit.is_none() {
            let search_limit = match (self.limit, self.offset) {
                (Some(limit), Some(offset)) => Some((limit + offset) as usize),
                (Some(limit), None) => Some(limit as usize),
                (None, Some(_)) => None, // No limit but has offset - fetch all and let limit_node handle
                (None, None) => None,
            };
            params = params.with_limit(search_limit);
        }
        let query = if columns.is_empty() {
            // the field is not specified,
            // try to search over all indexed fields
            let string_columns =
                self.dataset
                    .schema()
                    .fields
                    .iter()
                    .filter_map(|f| match f.data_type() {
                        DataType::Utf8 | DataType::LargeUtf8 => Some(&f.name),
                        DataType::List(field) | DataType::LargeList(field) => {
                            if matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8) {
                                Some(&f.name)
                            } else {
                                None
                            }
                        }
                        _ => None,
                    });

            let mut indexed_columns = Vec::new();
            for column in string_columns {
                let has_fts_index = self
                    .dataset
                    .load_scalar_index(
                        ScalarIndexCriteria::default()
                            .for_column(column)
                            .supports_fts(),
                    )
                    .await?
                    .is_some();
                if has_fts_index {
                    indexed_columns.push(column.clone());
                }
            }

            fill_fts_query_column(&query.query, &indexed_columns, false)?
        } else {
            query.query.clone()
        };

        // TODO: Could maybe walk the query here to find all the indices that will be
        // involved in the query to calculate a more accuarate required_fragments than
        // get_fragments_as_bitmap but this is safe for now.
        let prefilter_source = self
            .prefilter_source(
                filter_plan,
                self.fragments_covered_by_fts_query(&query).await?,
            )
            .await?;
        let fts_exec = self
            .plan_fts(&query, &params, filter_plan, &prefilter_source)
            .await?;
        Ok(fts_exec)
    }

    async fn plan_fts(
        &self,
        query: &FtsQuery,
        params: &FtsSearchParams,
        filter_plan: &FilterPlan,
        prefilter_source: &PreFilterSource,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let plan: Arc<dyn ExecutionPlan> = match query {
            FtsQuery::Match(query) => {
                self.plan_match_query(query, params, filter_plan, prefilter_source)
                    .await?
            }
            FtsQuery::Phrase(query) => Arc::new(PhraseQueryExec::new(
                self.dataset.clone(),
                query.clone(),
                params.clone(),
                prefilter_source.clone(),
            )),

            FtsQuery::Boost(query) => {
                // for boost query, we need to erase the limit so that we can find
                // the documents that are not in the top-k results of the positive query,
                // but in the final top-k results.
                let unlimited_params = params.clone().with_limit(None);
                let positive_exec = Box::pin(self.plan_fts(
                    &query.positive,
                    &unlimited_params,
                    filter_plan,
                    prefilter_source,
                ));
                let negative_exec = Box::pin(self.plan_fts(
                    &query.negative,
                    &unlimited_params,
                    filter_plan,
                    prefilter_source,
                ));
                let (positive_exec, negative_exec) =
                    futures::future::try_join(positive_exec, negative_exec).await?;
                Arc::new(BoostQueryExec::new(
                    query.clone(),
                    params.clone(),
                    positive_exec,
                    negative_exec,
                ))
            }

            FtsQuery::MultiMatch(query) => {
                let mut children = Vec::with_capacity(query.match_queries.len());
                for match_query in &query.match_queries {
                    let child =
                        self.plan_match_query(match_query, params, filter_plan, prefilter_source);
                    children.push(child);
                }
                let children = futures::future::try_join_all(children).await?;

                let schema = children[0].schema();
                let group_expr = vec![(
                    expressions::col(ROW_ID, schema.as_ref())?,
                    ROW_ID.to_string(),
                )];

                let fts_node = Arc::new(UnionExec::new(children));
                let fts_node = Arc::new(RepartitionExec::try_new(
                    fts_node,
                    Partitioning::RoundRobinBatch(1),
                )?);
                // dedup by row_id and return the max score as final score
                let fts_node = Arc::new(AggregateExec::try_new(
                    AggregateMode::Single,
                    PhysicalGroupBy::new_single(group_expr),
                    vec![Arc::new(
                        AggregateExprBuilder::new(
                            functions_aggregate::min_max::max_udaf(),
                            vec![expressions::col(SCORE_COL, &schema)?],
                        )
                        .schema(schema.clone())
                        .alias(SCORE_COL)
                        .build()?,
                    )],
                    vec![None],
                    fts_node,
                    schema,
                )?);
                let sort_expr = PhysicalSortExpr {
                    expr: expressions::col(SCORE_COL, fts_node.schema().as_ref())?,
                    options: SortOptions {
                        descending: true,
                        nulls_first: false,
                    },
                };

                Arc::new(
                    SortExec::new(LexOrdering::new(vec![sort_expr]), fts_node)
                        .with_fetch(self.limit.map(|l| l as usize)),
                )
            }
            FtsQuery::Boolean(query) => {
                // TODO: rewrite the query for better performance

                // we need to remove the limit from the params,
                // so that we won't miss possible matches
                let unlimited_params = params.clone().with_limit(None);

                // For should queries, union the results of each subquery
                let mut should = Vec::with_capacity(query.should.len());
                for subquery in &query.should {
                    let plan = Box::pin(self.plan_fts(
                        subquery,
                        &unlimited_params,
                        filter_plan,
                        prefilter_source,
                    ))
                    .await?;
                    should.push(plan);
                }
                let should = if should.is_empty() {
                    Arc::new(EmptyExec::new(FTS_SCHEMA.clone()))
                } else if should.len() == 1 {
                    should.pop().unwrap()
                } else {
                    let unioned = Arc::new(UnionExec::new(should));
                    Arc::new(RepartitionExec::try_new(
                        unioned,
                        Partitioning::RoundRobinBatch(1),
                    )?)
                };

                // For must queries, inner join the results of each subquery on row_id
                let mut must = None;
                for query in &query.must {
                    let plan = Box::pin(self.plan_fts(
                        query,
                        &unlimited_params,
                        filter_plan,
                        prefilter_source,
                    ))
                    .await?;
                    if let Some(joined_plan) = must {
                        must = Some(Arc::new(HashJoinExec::try_new(
                            joined_plan,
                            plan,
                            vec![(
                                Arc::new(Column::new_with_schema(ROW_ID, &FTS_SCHEMA)?),
                                Arc::new(Column::new_with_schema(ROW_ID, &FTS_SCHEMA)?),
                            )],
                            None,
                            &datafusion_expr::JoinType::Inner,
                            None,
                            datafusion_physical_plan::joins::PartitionMode::CollectLeft,
                            false,
                        )?) as _);
                    } else {
                        must = Some(plan);
                    }
                }

                // For must_not queries, union the results of each subquery
                let mut must_not = Vec::with_capacity(query.must_not.len());
                for query in &query.must_not {
                    let plan = Box::pin(self.plan_fts(
                        query,
                        &unlimited_params,
                        filter_plan,
                        prefilter_source,
                    ))
                    .await?;
                    must_not.push(plan);
                }
                let must_not = if must_not.is_empty() {
                    Arc::new(EmptyExec::new(FTS_SCHEMA.clone()))
                } else if must_not.len() == 1 {
                    must_not.pop().unwrap()
                } else {
                    let unioned = Arc::new(UnionExec::new(must_not));
                    Arc::new(RepartitionExec::try_new(
                        unioned,
                        Partitioning::RoundRobinBatch(1),
                    )?)
                };

                if query.should.is_empty() && must.is_none() {
                    return Err(Error::invalid_input(
                        "boolean query must have at least one should/must query".to_string(),
                        location!(),
                    ));
                }

                Arc::new(BooleanQueryExec::new(
                    query.clone(),
                    params.clone(),
                    should,
                    must,
                    must_not,
                ))
            }
        };

        Ok(plan)
    }

    async fn plan_match_query(
        &self,
        query: &MatchQuery,
        params: &FtsSearchParams,
        filter_plan: &FilterPlan,
        prefilter_source: &PreFilterSource,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let column = query
            .column
            .as_ref()
            .ok_or(Error::invalid_input(
                "the column must be specified in the query".to_string(),
                location!(),
            ))?
            .clone();

        let index = self
            .dataset
            .load_scalar_index(
                ScalarIndexCriteria::default()
                    .for_column(&column)
                    .supports_fts(),
            )
            .await?
            .ok_or(Error::invalid_input(
                format!(
                    "Column {} has no inverted index",
                    query.column.as_ref().unwrap()
                ),
                location!(),
            ))?;

        let unindexed_fragments = self.dataset.unindexed_fragments(&index.name).await?;
        let mut match_plan: Arc<dyn ExecutionPlan> = Arc::new(MatchQueryExec::new(
            self.dataset.clone(),
            query.clone(),
            params.clone(),
            prefilter_source.clone(),
        ));
        if !unindexed_fragments.is_empty() {
            let mut columns = vec![column.clone()];
            if let Some(expr) = filter_plan.full_expr.as_ref() {
                let filter_columns = Planner::column_names_in_expr(expr);
                columns.extend(filter_columns);
            }
            let flat_fts_scan_schema = Arc::new(self.dataset.schema().project(&columns).unwrap());
            let mut scan_node = self.scan_fragments(
                true,
                false,
                false,
                flat_fts_scan_schema,
                Arc::new(unindexed_fragments),
                None,
                false,
            );

            if let Some(expr) = filter_plan.full_expr.as_ref() {
                // If there is a prefilter we need to manually apply it to the new data
                scan_node = Arc::new(LanceFilterExec::try_new(expr.clone(), scan_node)?);
            }

            let flat_match_plan = Arc::new(FlatMatchQueryExec::new(
                self.dataset.clone(),
                query.clone(),
                params.clone(),
                scan_node,
            ));

            match_plan = Arc::new(UnionExec::new(vec![match_plan, flat_match_plan]));
            match_plan = Arc::new(RepartitionExec::try_new(
                match_plan,
                Partitioning::RoundRobinBatch(1),
            )?);
            let sort_expr = PhysicalSortExpr {
                expr: expressions::col(SCORE_COL, match_plan.schema().as_ref())?,
                options: SortOptions {
                    descending: true,
                    nulls_first: false,
                },
            };
            match_plan = Arc::new(
                SortExec::new(LexOrdering::new(vec![sort_expr]), match_plan)
                    .with_fetch(params.limit),
            );
        }
        Ok(match_plan)
    }

    // ANN/KNN search execution node with optional prefilter
    async fn vector_search(&self, filter_plan: &FilterPlan) -> Result<Arc<dyn ExecutionPlan>> {
        let Some(q) = self.nearest.as_ref() else {
            return Err(Error::invalid_input(
                "No nearest query".to_string(),
                location!(),
            ));
        };

        // Sanity check
        let (vector_type, _) = get_vector_type(self.dataset.schema(), &q.column)?;

        let column_id = self.dataset.schema().field_id(q.column.as_str())?;
        let use_index = self.nearest.as_ref().map(|q| q.use_index).unwrap_or(false);
        let indices = if use_index {
            self.dataset.load_indices().await?
        } else {
            Arc::new(vec![])
        };
        if let Some(index) = indices.iter().find(|i| i.fields.contains(&column_id)) {
            log::trace!("index found for vector search");
            // There is an index built for the column.
            // We will use the index.
            if matches!(q.refine_factor, Some(0)) {
                return Err(Error::invalid_input(
                    "Refine factor cannot be zero".to_string(),
                    location!(),
                ));
            }

            // Find all deltas with the same index name.
            let deltas = self.dataset.load_indices_by_name(&index.name).await?;
            let ann_node = match vector_type {
                DataType::FixedSizeList(_, _) => self.ann(q, &deltas, filter_plan).await?,
                DataType::List(_) => self.multivec_ann(q, &deltas, filter_plan).await?,
                _ => unreachable!(),
            };

            let mut knn_node = if q.refine_factor.is_some() {
                let vector_projection = self
                    .dataset
                    .empty_projection()
                    .union_column(&q.column, OnMissing::Error)
                    .unwrap();
                let knn_node_with_vector = self.take(ann_node, vector_projection)?;
                // TODO: now we just open an index to get its metric type.
                let idx = self
                    .dataset
                    .open_vector_index(
                        q.column.as_str(),
                        &index.uuid.to_string(),
                        &NoOpMetricsCollector,
                    )
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
            let mut vector_scan_projection = self
                .dataset
                .empty_projection()
                .with_row_id()
                .union_columns(&columns, OnMissing::Error)?;

            vector_scan_projection.with_row_addr =
                self.projection_plan.physical_projection.with_row_addr;

            let PlannedFilteredScan { mut plan, .. } = self
                .filtered_read(
                    filter_plan,
                    vector_scan_projection,
                    /*include_deleted_rows=*/ true,
                    None,
                    None,
                    /*is_prefilter= */ true,
                )
                .await?;

            if let Some(refine_expr) = &filter_plan.refine_expr {
                plan = Arc::new(LanceFilterExec::try_new(refine_expr.clone(), plan)?);
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
            // need to set the metric type to be the same as the index
            // to make sure the distance is comparable.
            let idx = self
                .dataset
                .open_vector_index(
                    q.column.as_str(),
                    &index.uuid.to_string(),
                    &NoOpMetricsCollector,
                )
                .await?;
            let mut q = q.clone();
            q.metric_type = idx.metric_type();

            // If the vector column is not present, we need to take the vector column, so
            // that the distance value is comparable with the flat search ones.
            if knn_node.schema().column_with_name(&q.column).is_none() {
                let vector_projection = self
                    .dataset
                    .empty_projection()
                    .union_column(&q.column, OnMissing::Error)
                    .unwrap();
                knn_node = self.take(knn_node, vector_projection)?;
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
                false,
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
                scan_node = Arc::new(LanceFilterExec::try_new(expr.clone(), scan_node)?);
            }
            // first we do flat search on just the new data
            let topk_appended = self.flat_knn(scan_node, &q)?;

            // To do a union, we need to make the schemas match. Right now
            // knn_node: _distance, _rowid, vector
            // topk_appended: vector, <filter columns?>, _rowid, _distance
            let topk_appended = project(topk_appended, knn_node.schema().as_ref())?;
            assert!(topk_appended
                .schema()
                .equivalent_names_and_types(&knn_node.schema()));
            // union
            let unioned = UnionExec::new(vec![Arc::new(topk_appended), knn_node]);
            // Enforce only 1 partition.
            let unioned = RepartitionExec::try_new(
                Arc::new(unioned),
                datafusion::physical_plan::Partitioning::RoundRobinBatch(1),
            )?;
            // then we do a flat search on KNN(new data) + ANN(indexed data)
            return self.flat_knn(Arc::new(unioned), &q);
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
            ScalarIndexExpr::Query(search) => {
                let idx = self
                    .dataset
                    .load_scalar_index(ScalarIndexCriteria::default().with_name(&search.index_name))
                    .await?
                    .expect("Index not found even though it must have been found earlier");
                Ok(idx
                    .fragment_bitmap
                    .expect("scalar indices should always have a fragment bitmap"))
            }
        }
    }

    /// Given an index query, split the fragments into two sets
    ///
    /// The first set is the relevant fragments, which are covered by ALL indices in the query
    /// The second set is the missing fragments, which are missed by at least one index
    ///
    /// There is no point in handling the case where a fragment is covered by some (but not all)
    /// of the indices.  If we have to do a full scan of the fragment then we do it
    async fn partition_frags_by_coverage(
        &self,
        index_expr: &ScalarIndexExpr,
        fragments: Arc<Vec<Fragment>>,
    ) -> Result<(Vec<Fragment>, Vec<Fragment>)> {
        let covered_frags = self.fragments_covered_by_index_query(index_expr).await?;
        let mut relevant_frags = Vec::with_capacity(fragments.len());
        let mut missing_frags = Vec::with_capacity(fragments.len());
        for fragment in fragments.iter() {
            if covered_frags.contains(fragment.id as u32) {
                relevant_frags.push(fragment.clone());
            } else {
                missing_frags.push(fragment.clone());
            }
        }
        Ok((relevant_frags, missing_frags))
    }

    // First perform a lookup in a scalar index for ids and then perform a take on the
    // target fragments with those ids
    async fn scalar_indexed_scan(
        &self,
        projection: Projection,
        filter_plan: &FilterPlan,
        fragments: Arc<Vec<Fragment>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        log::trace!("scalar indexed scan");
        // One or more scalar indices cover this data and there is a filter which is
        // compatible with the indices.  Use that filter to perform a take instead of
        // a full scan.

        // If this unwrap fails we have a bug because we shouldn't be using this function unless we've already
        // checked that there is an index query
        let index_expr = filter_plan.index_query.as_ref().unwrap();

        let needs_recheck = index_expr.needs_recheck();

        // Figure out which fragments are covered by ALL indices
        let (relevant_frags, missing_frags) = self
            .partition_frags_by_coverage(index_expr, fragments)
            .await?;

        let mut plan: Arc<dyn ExecutionPlan> = Arc::new(MaterializeIndexExec::new(
            self.dataset.clone(),
            index_expr.clone(),
            Arc::new(relevant_frags),
        ));

        let refine_expr = filter_plan.refine_expr.as_ref();

        // If all we want is the row ids then we can skip the take.  However, if there is a refine
        // or a recheck then we still need to do a take because we need filter columns.
        let needs_take =
            needs_recheck || projection.has_data_fields() || filter_plan.refine_expr.is_some();
        if needs_take {
            let mut take_projection = projection.clone();
            if needs_recheck {
                // If we need to recheck then we need to also take the columns used for the filter
                let filter_expr = index_expr.to_expr();
                let filter_cols = Planner::column_names_in_expr(&filter_expr);
                take_projection = take_projection.union_columns(filter_cols, OnMissing::Error)?;
            }
            if let Some(refine_expr) = refine_expr {
                let refine_cols = Planner::column_names_in_expr(refine_expr);
                take_projection = take_projection.union_columns(refine_cols, OnMissing::Error)?;
            }
            log::trace!("need to take additional columns for scalar_indexed_scan");
            plan = self.take(plan, take_projection)?;
        }

        let post_take_filter = match (needs_recheck, refine_expr) {
            (false, None) => None,
            (true, None) => {
                // If we need to recheck then we need to apply the filter to the results
                Some(index_expr.to_expr())
            }
            (true, Some(_)) => Some(filter_plan.full_expr.as_ref().unwrap().clone()),
            (false, Some(refine_expr)) => Some(refine_expr.clone()),
        };

        if let Some(post_take_filter) = post_take_filter {
            let planner = Planner::new(plan.schema());
            let optimized_filter = planner.optimize_expr(post_take_filter)?;

            log::trace!("applying post-take filter to indexed scan");
            plan = Arc::new(LanceFilterExec::try_new(optimized_filter, plan)?);
        }

        if self.projection_plan.physical_projection.with_row_addr {
            plan = Arc::new(AddRowAddrExec::try_new(plan, self.dataset.clone(), 0)?);
        }

        let new_data_path: Option<Arc<dyn ExecutionPlan>> = if !missing_frags.is_empty() {
            log::trace!(
                "scalar_indexed_scan will need full scan of {} missing fragments",
                missing_frags.len()
            );

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
            let filter = filter_plan.full_expr.as_ref().unwrap();
            let filter_cols = Planner::column_names_in_expr(filter);
            let scan_projection = projection.union_columns(filter_cols, OnMissing::Error)?;

            let scan_schema = Arc::new(scan_projection.to_bare_schema());
            let scan_arrow_schema = Arc::new(scan_schema.as_ref().into());
            let planner = Planner::new(scan_arrow_schema);
            let optimized_filter = planner.optimize_expr(filter.clone())?;

            let new_data_scan = self.scan_fragments(
                true,
                self.projection_plan.physical_projection.with_row_addr,
                false,
                scan_schema,
                missing_frags.into(),
                // No pushdown of limit/offset when doing scalar indexed scan
                None,
                false,
            );
            let filtered = Arc::new(LanceFilterExec::try_new(optimized_filter, new_data_scan)?);
            Some(Arc::new(project(filtered, plan.schema().as_ref())?))
        } else {
            log::trace!("scalar_indexed_scan will not need full scan of any missing fragments");
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
        log::trace!("scan_fragments covered {} fragments", fragments.len());
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
        filter_plan: &FilterPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        log::trace!("pushdown_scan");

        let config = ScanConfig {
            batch_readahead: self.batch_readahead,
            fragment_readahead: self
                .fragment_readahead
                .unwrap_or(LEGACY_DEFAULT_FRAGMENT_READAHEAD),
            with_row_id: self.projection_plan.physical_projection.with_row_id,
            with_row_address: self.projection_plan.physical_projection.with_row_addr,
            make_deletions_null,
            ordered_output: self.ordered,
            file_reader_options: self
                .file_reader_options
                .clone()
                .or_else(|| self.dataset.file_reader_options.clone()),
        };

        let fragments = if let Some(fragment) = self.fragments.as_ref() {
            Arc::new(fragment.clone())
        } else {
            self.dataset.fragments().clone()
        };

        Ok(Arc::new(LancePushdownScanExec::try_new(
            self.dataset.clone(),
            fragments,
            Arc::new(self.projection_plan.physical_projection.to_bare_schema()),
            filter_plan.refine_expr.clone().unwrap(),
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

        let lower: Option<(Expr, Arc<dyn PhysicalExpr>)> = q
            .lower_bound
            .map(|v| -> Result<(Expr, Arc<dyn PhysicalExpr>)> {
                let logical = col(DIST_COL).gt_eq(lit(v));
                let schema = flat_dist.schema();
                let df_schema = DFSchema::try_from(schema)?;
                let physical = create_physical_expr(&logical, &df_schema, &ExecutionProps::new())?;
                Ok::<(Expr, Arc<dyn PhysicalExpr>), _>((logical, physical))
            })
            .transpose()?;

        let upper = q
            .upper_bound
            .map(|v| -> Result<(Expr, Arc<dyn PhysicalExpr>)> {
                let logical = col(DIST_COL).lt(lit(v));
                let schema = flat_dist.schema();
                let df_schema = DFSchema::try_from(schema)?;
                let physical = create_physical_expr(&logical, &df_schema, &ExecutionProps::new())?;
                Ok::<(Expr, Arc<dyn PhysicalExpr>), _>((logical, physical))
            })
            .transpose()?;

        let filter_expr = match (lower, upper) {
            (Some((llog, _)), Some((ulog, _))) => {
                let logical = llog.and(ulog);
                let schema = flat_dist.schema();
                let df_schema = DFSchema::try_from(schema)?;
                let physical = create_physical_expr(&logical, &df_schema, &ExecutionProps::new())?;
                Some((logical, physical))
            }
            (Some((llog, lphys)), None) => Some((llog, lphys)),
            (None, Some((ulog, uphys))) => Some((ulog, uphys)),
            (None, None) => None,
        };

        let knn_plan: Arc<dyn ExecutionPlan> = if let Some(filter_expr) = filter_expr {
            Arc::new(LanceFilterExec::try_new(filter_expr.0, flat_dist)?)
        } else {
            flat_dist
        };

        // Use DataFusion's [SortExec] for Top-K search
        let sort = SortExec::new(
            LexOrdering::new(vec![
                PhysicalSortExpr {
                    expr: expressions::col(DIST_COL, knn_plan.schema().as_ref())?,
                    options: SortOptions {
                        descending: false,
                        nulls_first: false,
                    },
                },
                PhysicalSortExpr {
                    expr: expressions::col(ROW_ID, knn_plan.schema().as_ref())?,
                    options: SortOptions {
                        descending: false,
                        nulls_first: false,
                    },
                },
            ]),
            knn_plan,
        )
        .with_fetch(Some(q.k));

        let logical_not_null = col(DIST_COL).is_not_null();
        let not_nulls = Arc::new(LanceFilterExec::try_new(logical_not_null, Arc::new(sort))?);

        Ok(not_nulls)
    }

    fn get_fragments_as_bitmap(&self) -> RoaringBitmap {
        if let Some(fragments) = &self.fragments {
            RoaringBitmap::from_iter(fragments.iter().map(|f| f.id as u32))
        } else {
            RoaringBitmap::from_iter(self.dataset.fragments().iter().map(|f| f.id as u32))
        }
    }

    fn get_indexed_frags(&self, index: &[Index]) -> RoaringBitmap {
        let all_fragments = self.get_fragments_as_bitmap();

        let mut all_indexed_frags = RoaringBitmap::new();
        for idx in index {
            if let Some(fragmap) = idx.fragment_bitmap.as_ref() {
                all_indexed_frags |= fragmap;
            } else {
                // If any index is missing the fragment bitmap it is safest to just assume we
                // need all fragments
                return all_fragments;
            }
        }

        all_indexed_frags & all_fragments
    }

    /// Create an Execution plan to do indexed ANN search
    async fn ann(
        &self,
        q: &Query,
        index: &[Index],
        filter_plan: &FilterPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let prefilter_source = self
            .prefilter_source(filter_plan, self.get_indexed_frags(index))
            .await?;
        let inner_fanout_search = new_knn_exec(self.dataset.clone(), index, q, prefilter_source)?;
        let sort_expr = PhysicalSortExpr {
            expr: expressions::col(DIST_COL, inner_fanout_search.schema().as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        };
        let sort_expr_row_id = PhysicalSortExpr {
            expr: expressions::col(ROW_ID, inner_fanout_search.schema().as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        };
        Ok(Arc::new(
            SortExec::new(
                LexOrdering::new(vec![sort_expr, sort_expr_row_id]),
                inner_fanout_search,
            )
            .with_fetch(Some(q.k * q.refine_factor.unwrap_or(1) as usize)),
        ))
    }

    // Create an Execution plan to do ANN over multivectors
    async fn multivec_ann(
        &self,
        q: &Query,
        index: &[Index],
        filter_plan: &FilterPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // we split the query procedure into two steps:
        // 1. collect the candidates by vector searching on each query vector
        // 2. scoring the candidates

        let over_fetch_factor = *DEFAULT_XTR_OVERFETCH;

        let prefilter_source = self
            .prefilter_source(filter_plan, self.get_indexed_frags(index))
            .await?;
        let dim = get_vector_dim(self.dataset.schema(), &q.column)?;

        let num_queries = q.key.len() / dim;
        let new_queries = (0..num_queries)
            .map(|i| q.key.slice(i * dim, dim))
            .map(|query_vec| {
                let mut new_query = q.clone();
                new_query.key = query_vec;
                // with XTR, we don't need to refine the result with original vectors,
                // but here we really need to over-fetch the candidates to reach good enough recall.
                // TODO: improve the recall with WARP, expose this parameter to the users.
                new_query.refine_factor = Some(over_fetch_factor);
                new_query
            });
        let mut ann_nodes = Vec::with_capacity(new_queries.len());
        for query in new_queries {
            // this produces `nprobes * k * over_fetch_factor * num_indices` candidates
            let ann_node = new_knn_exec(
                self.dataset.clone(),
                index,
                &query,
                prefilter_source.clone(),
            )?;
            let sort_expr = PhysicalSortExpr {
                expr: expressions::col(DIST_COL, ann_node.schema().as_ref())?,
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            };
            let sort_expr_row_id = PhysicalSortExpr {
                expr: expressions::col(ROW_ID, ann_node.schema().as_ref())?,
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            };
            let ann_node = Arc::new(
                SortExec::new(
                    LexOrdering::new(vec![sort_expr, sort_expr_row_id]),
                    ann_node,
                )
                .with_fetch(Some(q.k * over_fetch_factor as usize)),
            );
            ann_nodes.push(ann_node as Arc<dyn ExecutionPlan>);
        }

        let ann_node = Arc::new(MultivectorScoringExec::try_new(ann_nodes, q.clone())?);

        let sort_expr = PhysicalSortExpr {
            expr: expressions::col(DIST_COL, ann_node.schema().as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        };
        let sort_expr_row_id = PhysicalSortExpr {
            expr: expressions::col(ROW_ID, ann_node.schema().as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        };
        let ann_node = Arc::new(
            SortExec::new(
                LexOrdering::new(vec![sort_expr, sort_expr_row_id]),
                ann_node,
            )
            .with_fetch(Some(q.k * q.refine_factor.unwrap_or(1) as usize)),
        );

        Ok(ann_node)
    }

    /// Create prefilter source from filter plan
    ///
    /// A prefilter is an input to a vector or fts search.  It tells us which rows are eligible
    /// for the search.  A prefilter is calculated by doing a filtered read of the row id column.
    async fn prefilter_source(
        &self,
        filter_plan: &FilterPlan,
        required_frags: RoaringBitmap,
    ) -> Result<PreFilterSource> {
        if filter_plan.is_empty() {
            log::trace!("no filter plan, no prefilter");
            return Ok(PreFilterSource::None);
        }

        let fragments = Arc::new(
            self.dataset
                .manifest
                .fragments
                .iter()
                .filter(|f| required_frags.contains(f.id as u32))
                .cloned()
                .collect::<Vec<_>>(),
        );

        // Can only use ScalarIndexExec when the scalar index is exact and we are not scanning
        // a subset of the fragments.
        //
        // TODO: We could enhance ScalarIndexExec with a fragment bitmap to filter out rows that
        // are not in the fragments we are scanning.
        if filter_plan.is_exact_index_search() && self.fragments.is_none() {
            let index_query = filter_plan.index_query.as_ref().expect_ok()?;
            let (_, missing_frags) = self
                .partition_frags_by_coverage(index_query, fragments.clone())
                .await?;

            if missing_frags.is_empty() {
                log::trace!("prefilter entirely satisfied by exact index search");
                // We can only avoid materializing the index for a prefilter if:
                // 1. The search is indexed
                // 2. The index search is an exact search with no recheck or refine
                // 3. The indices cover at least the same fragments as the vector index
                return Ok(PreFilterSource::ScalarIndexQuery(Arc::new(
                    ScalarIndexExec::new(self.dataset.clone(), index_query.clone()),
                )));
            } else {
                log::trace!("exact index search did not cover all fragments");
            }
        }

        // If one of our criteria is not met, we need to do a filtered read of just the row id column
        log::trace!(
            "prefilter is a filtered read of {} fragments",
            fragments.len()
        );
        let PlannedFilteredScan { plan, .. } = self
            .filtered_read(
                filter_plan,
                self.dataset.empty_projection().with_row_id(),
                false,
                Some(fragments),
                None,
                /*is_prefilter= */ true,
            )
            .await?;
        Ok(PreFilterSource::FilteredRowIds(plan))
    }

    /// Take row indices produced by input plan from the dataset (with projection)
    fn take(
        &self,
        input: Arc<dyn ExecutionPlan>,
        output_projection: Projection,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let coalesced = Arc::new(CoalesceBatchesExec::new(
            input.clone(),
            self.get_batch_size(),
        ));
        if let Some(take_plan) =
            TakeExec::try_new(self.dataset.clone(), coalesced, output_projection)?
        {
            Ok(Arc::new(take_plan))
        } else {
            // No new columns needed
            Ok(input)
        }
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
    pub async fn analyze_plan(&self) -> Result<String> {
        let plan = self.create_plan().await?;

        analyze_plan(
            plan,
            LanceExecutionOptions {
                batch_size: self.batch_size,
                ..Default::default()
            },
        )
        .await
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
        // Convert lance.json (JSONB) back to arrow.json (strings) for reading
        //
        // This is so bad, we need to find a way to remove this.
        let exec_node = wrap_json_stream_for_reading(exec_node);

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

    use std::{collections::HashMap, vec};

    use arrow_array::{
        ArrayRef, FixedSizeListArray, Int32Array, RecordBatch, RecordBatchIterator, StringArray,
    };
    use arrow_schema::{ArrowError, DataType};
    use lance_file::version::LanceFileVersion;
    use lance_index::{
        scalar::{inverted::tokenizer::InvertedIndexParams, ScalarIndexParams},
        IndexType,
    };
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
                enable_stable_row_ids: stable_row_ids,
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
    use arrow::datatypes::{Int32Type, Int64Type};
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float32Type, UInt64Type};
    use arrow_array::{
        ArrayRef, FixedSizeListArray, Float16Array, Int32Array, LargeStringArray, PrimitiveArray,
        RecordBatchIterator, StringArray, StructArray,
    };
    use arrow_ord::sort::sort_to_indices;
    use arrow_schema::Fields;
    use arrow_select::take;
    use datafusion::logical_expr::{col, lit};
    use half::f16;
    use lance_arrow::SchemaExt;
    use lance_datagen::{array, gen_batch, BatchCount, ByteCount, Dimension, RowCount};
    use lance_file::version::LanceFileVersion;
    use lance_index::scalar::inverted::query::{MatchQuery, PhraseQuery};
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
    use crate::utils::test::{
        assert_plan_node_equals, DatagenExt, FragmentCount, FragmentRowCount, IoStats,
        IoTrackingStore,
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

    #[tokio::test]
    async fn test_strict_batch_size() {
        let dataset = lance_datagen::gen_batch()
            .col("x", array::step::<Int32Type>())
            .anon_col(array::step::<Int64Type>())
            .into_ram_dataset(FragmentCount::from(7), FragmentRowCount::from(6))
            .await
            .unwrap();

        let mut scan = dataset.scan();
        scan.batch_size(10)
            .strict_batch_size(true)
            .filter("x % 2 == 0")
            .unwrap();

        let batches = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let batch_sizes = batches.iter().map(|b| b.num_rows()).collect::<Vec<_>>();
        assert_eq!(batch_sizes, vec![10, 10, 1]);
    }

    #[tokio::test]
    async fn test_column_not_exist() {
        let dataset = lance_datagen::gen_batch()
            .col("x", array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(7), FragmentRowCount::from(6))
            .await
            .unwrap();

        let check_err_msg = |r: Result<DatasetRecordBatchStream>| {
            let Err(err) = r else {
                panic!(
                    "Expected an error to be raised saying column y is not found but got no error"
                )
            };

            assert!(
                err.to_string().contains("No field named y"),
                "Expected error to contain 'No field named y' but got {}",
                err
            );
        };

        let mut scan = dataset.scan();
        scan.project(&["x", "y"]).unwrap();
        check_err_msg(scan.try_into_stream().await);

        let mut scan = dataset.scan();
        scan.project(&["y"]).unwrap();
        check_err_msg(scan.try_into_stream().await);

        // This represents a query like `SELECT 1 AS foo` which we could _technically_ satisfy
        // but it is not supported today
        let mut scan = dataset.scan();
        scan.project_with_transform(&[("foo", "1")]).unwrap();
        match scan.try_into_stream().await {
            Ok(_) => panic!("Expected an error to be raised saying not supported"),
            Err(e) => {
                assert!(
                    e.to_string().contains("Received only dynamic expressions"),
                    "Expected error to contain 'Received only dynamic expressions' but got {}",
                    e
                );
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

    #[tokio::test]
    async fn test_nested_projection() {
        let point_fields: Fields = vec![
            ArrowField::new("x", DataType::Float32, true),
            ArrowField::new("y", DataType::Float32, true),
        ]
        .into();
        let metadata_fields: Fields = vec![
            ArrowField::new("location", DataType::Struct(point_fields), true),
            ArrowField::new("age", DataType::Int32, true),
        ]
        .into();
        let metadata_field = ArrowField::new("metadata", DataType::Struct(metadata_fields), true);
        let schema = Arc::new(ArrowSchema::new(vec![
            metadata_field,
            ArrowField::new("idx", DataType::Int32, true),
        ]));
        let data = lance_datagen::rand(&schema)
            .into_ram_dataset(FragmentCount::from(7), FragmentRowCount::from(6))
            .await
            .unwrap();

        let mut scan = data.scan();
        scan.project(&["metadata.location.x", "metadata.age"])
            .unwrap();
        let batch = scan.try_into_batch().await.unwrap();

        assert_eq!(
            batch.schema().as_ref(),
            &ArrowSchema::new(vec![
                ArrowField::new("metadata.location.x", DataType::Float32, true),
                ArrowField::new("metadata.age", DataType::Int32, true),
            ])
        );

        // 0 - metadata
        // 2 - x
        // 4 - age
        let take_schema = data.schema().project_by_ids(&[0, 2, 4], false);

        let taken = data.take_rows(&[0, 5], take_schema).await.unwrap();

        // The expected schema drops y from the location field
        let part_point_fields = Fields::from(vec![ArrowField::new("x", DataType::Float32, true)]);
        let part_metadata_fields = Fields::from(vec![
            ArrowField::new("location", DataType::Struct(part_point_fields), true),
            ArrowField::new("age", DataType::Int32, true),
        ]);
        let part_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "metadata",
            DataType::Struct(part_metadata_fields),
            true,
        )]));

        assert_eq!(taken.schema(), part_schema);
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

        assert_eq!(actual.num_rows(), 2);
        assert_eq!(actual, full_data);
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_knn_nodes(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] stable_row_ids: bool,
        #[values(false, true)] build_index: bool,
    ) {
        let mut test_ds = TestVectorDataset::new(data_storage_version, stable_row_ids)
            .await
            .unwrap();
        if build_index {
            test_ds.make_vector_index().await.unwrap();
        }
        let dataset = &test_ds.dataset;

        let mut scan = dataset.scan();
        let key: Float32Array = (32..64).map(|v| v as f32).collect();
        scan.nearest("vec", &key, 5).unwrap();
        scan.refine(5);

        let batch = scan.try_into_batch().await.unwrap();

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
    async fn test_can_project_distance() {
        let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
            .await
            .unwrap();
        let dataset = &test_ds.dataset;

        let mut scan = dataset.scan();
        let key: Float32Array = (32..64).map(|v| v as f32).collect();
        scan.nearest("vec", &key, 5).unwrap();
        scan.refine(5);
        scan.project(&["_distance"]).unwrap();

        let batch = scan.try_into_batch().await.unwrap();

        assert_eq!(batch.num_rows(), 5);
        assert_eq!(batch.num_columns(), 1);
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

        let data = gen_batch()
            .col("int", array::cycle::<Int32Type>(vec![5, 4, 1, 2, 3]))
            .col(
                "str",
                array::cycle_utf8_literals(&["a", "b", "c", "e", "d"]),
            );

        let sorted_by_int = gen_batch()
            .col("int", array::cycle::<Int32Type>(vec![1, 2, 3, 4, 5]))
            .col(
                "str",
                array::cycle_utf8_literals(&["c", "e", "d", "b", "a"]),
            )
            .into_batch_rows(RowCount::from(5))
            .unwrap();

        let sorted_by_str = gen_batch()
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

        let data = gen_batch()
            .col("int", array::cycle::<Int32Type>(vec![5, 5, 1, 1, 3]))
            .col(
                "float",
                array::cycle::<Float32Type>(vec![7.3, -f32::NAN, f32::NAN, 4.3, f32::INFINITY]),
            );

        let sorted_by_int_then_float = gen_batch()
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
            enable_stable_row_ids: stable_row_ids,
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
        scan.nearest("vector", query_key.as_ref(), 1).unwrap();
        scan.minimum_nprobes(100);
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
                    enable_stable_row_ids: stable_row_ids,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

            assert_eq!(dataset.index_cache_entry_count().await, 0);
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
            scan.minimum_nprobes(100);

            assert_eq!(
                dataset.index_cache_entry_count().await,
                2, // 2 for index metadata at version 1 and 2.
            );
            let results = scan
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();

            assert_eq!(
                dataset.index_cache_entry_count().await,
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
            scan.minimum_nprobes(100);

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
            scan.minimum_nprobes(100);

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
    async fn test_projection_order() {
        let vec_params = VectorIndexParams::ivf_pq(4, 8, 2, MetricType::L2, 2);
        let mut data = gen_batch()
            .col("vec", array::rand_vec::<Float32Type>(Dimension::from(4)))
            .col("text", array::rand_utf8(ByteCount::from(10), false))
            .into_ram_dataset(FragmentCount::from(3), FragmentRowCount::from(100))
            .await
            .unwrap();
        data.create_index(&["vec"], IndexType::Vector, None, &vec_params, true)
            .await
            .unwrap();

        let mut scan = data.scan();
        scan.nearest("vec", &Float32Array::from(vec![1.0, 1.0, 1.0, 1.0]), 5)
            .unwrap();
        scan.with_row_id().project(&["text"]).unwrap();

        let results = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(
            results[0].schema().field_names(),
            vec!["text", "_distance", "_rowid"]
        );
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
        assert_eq!(32, dataset.count_rows(None).await.unwrap());
        assert_eq!(
            16,
            dataset
                .count_rows(Some("`Filter_me` > 15".to_string()))
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
        assert_eq!(dataset.count_rows(None).await.unwrap(), 32);

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
        assert_eq!(dataset.count_rows(None).await.unwrap(), 32);

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
            let data = gen_batch()
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
                    enable_stable_row_ids: use_stable_row_ids,
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
            if self.dataset.is_legacy_storage() {
                assert!(query_plan.contains("MaterializeIndex"));
            }
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
            if self.dataset.is_legacy_storage() {
                if params.use_index {
                    // An ANN search whose prefilter is fully satisfied by the index should be
                    // able to use a ScalarIndexQuery
                    assert!(query_plan.contains("ScalarIndexQuery"));
                } else {
                    // A KNN search requires materialization of the index
                    assert!(query_plan.contains("MaterializeIndex"));
                }
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
            if self.dataset.is_legacy_storage() {
                assert!(query_plan.contains("MaterializeIndex"));
            } else {
                assert!(query_plan.contains("LanceRead"));
            }
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
            if self.dataset.is_legacy_storage() {
                assert!(query_plan.contains("MaterializeIndex"));
            } else {
                assert!(query_plan.contains("LanceRead"));
            }
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

    #[tokio::test]
    async fn can_filter_row_id() {
        let dataset = lance_datagen::gen_batch()
            .col("x", array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(1000))
            .await
            .unwrap();

        let mut scan = dataset.scan();
        scan.with_row_id();
        scan.project::<&str>(&[]).unwrap();
        scan.filter("_rowid == 50").unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.column(0).as_primitive::<UInt64Type>().values()[0], 50);
    }

    #[rstest]
    #[tokio::test]
    async fn test_index_take_batch_size() {
        let fixture = ScalarIndexTestFixture::new(LanceFileVersion::Stable, false).await;
        let stream = fixture
            .dataset
            .scan()
            .filter("indexed > 0")
            .unwrap()
            .batch_size(16)
            .try_into_stream()
            .await
            .unwrap();
        let batches = stream.collect::<Vec<_>>().await;
        assert_eq!(batches.len(), 1000_usize.div_ceil(16));
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
        assert_plan_node_equals(exec_plan, expected).await
    }

    #[tokio::test]
    async fn test_count_plan() {
        // A count rows operation should load the minimal amount of data
        let dim = 256;
        let fixture = TestVectorDataset::new_with_dimension(LanceFileVersion::Stable, true, dim)
            .await
            .unwrap();

        // By default, all columns are returned, this is bad for a count_rows op
        let err = fixture
            .dataset
            .scan()
            .create_count_plan()
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }));

        let mut scan = fixture.dataset.scan();
        scan.project(&Vec::<String>::default()).unwrap();

        // with_row_id needs to be specified
        let err = scan.create_count_plan().await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }));

        scan.with_row_id();

        let plan = scan.create_count_plan().await.unwrap();

        assert_plan_node_equals(
            plan,
            "AggregateExec: mode=Single, gby=[], aggr=[count_rows]
  LanceRead: uri=..., projection=[], num_fragments=2, range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=--, refine_filter=--",
        )
        .await
        .unwrap();

        scan.filter("s == ''").unwrap();

        let plan = scan.create_count_plan().await.unwrap();

        assert_plan_node_equals(
            plan,
            "AggregateExec: mode=Single, gby=[], aggr=[count_rows]
  ProjectionExec: expr=[_rowid@1 as _rowid]
    LanceRead: uri=..., projection=[s], num_fragments=2, range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=s = Utf8(\"\"), refine_filter=s = Utf8(\"\")",
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_inexact_scalar_index_plans() {
        let data = gen_batch()
            .col("ngram", array::rand_utf8(ByteCount::from(5), false))
            .col("exact", array::rand_type(&DataType::UInt32))
            .col("no_index", array::rand_type(&DataType::UInt32))
            .into_reader_rows(RowCount::from(1000), BatchCount::from(5));

        let mut dataset = Dataset::write(data, "memory://test", None).await.unwrap();
        dataset
            .create_index(
                &["ngram"],
                IndexType::NGram,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        dataset
            .create_index(
                &["exact"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // Simple in-exact filter
        assert_plan_equals(
            &dataset,
            |scanner| scanner.filter("contains(ngram, 'test string')"),
            "LanceRead: uri=..., projection=[ngram, exact, no_index], num_fragments=1, \
             range_before=None, range_after=None, row_id=false, row_addr=false, \
             full_filter=contains(ngram, Utf8(\"test string\")), refine_filter=--
               ScalarIndexQuery: query=[contains(ngram, Utf8(\"test string\"))]@ngram_idx",
        )
        .await
        .unwrap();

        // Combined with exact filter
        assert_plan_equals(
            &dataset,
            |scanner| scanner.filter("contains(ngram, 'test string') and exact < 50"),
            "LanceRead: uri=..., projection=[ngram, exact, no_index], num_fragments=1, \
            range_before=None, range_after=None, row_id=false, row_addr=false, \
            full_filter=contains(ngram, Utf8(\"test string\")) AND exact < UInt32(50), \
            refine_filter=--
              ScalarIndexQuery: query=AND([contains(ngram, Utf8(\"test string\"))]@ngram_idx,[exact < 50]@exact_idx)",
        )
        .await
        .unwrap();

        // All three filters
        assert_plan_equals(
            &dataset,
            |scanner| {
                scanner.filter("contains(ngram, 'test string') and exact < 50 AND no_index > 100")
            },
            "ProjectionExec: expr=[ngram@0 as ngram, exact@1 as exact, no_index@2 as no_index]
  LanceRead: uri=..., projection=[ngram, exact, no_index], num_fragments=1, range_before=None, \
  range_after=None, row_id=true, row_addr=false, full_filter=contains(ngram, Utf8(\"test string\")) AND exact < UInt32(50) AND no_index > UInt32(100), \
  refine_filter=no_index > UInt32(100)
    ScalarIndexQuery: query=AND([contains(ngram, Utf8(\"test string\"))]@ngram_idx,[exact < 50]@exact_idx)",
        )
        .await
        .unwrap();
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
        let data = gen_batch()
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
        // Pushdown only works with the legacy format for now.
        if data_storage_version == LanceFileVersion::Legacy {
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
        }

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

        use lance_index::scalar::inverted::query::BoostQuery;
        let dim = 256;
        let mut dataset =
            TestVectorDataset::new_with_dimension(data_storage_version, stable_row_id, dim).await?;
        let lance_schema = dataset.dataset.schema();

        // Scans
        // ---------------------------------------------------------------------
        // V2 writer does not use LancePushdownScan
        if data_storage_version == LanceFileVersion::Legacy {
            log::info!("Test case: Pushdown scan");
            assert_plan_equals(
                &dataset.dataset,
                |scan| scan.project(&["s"])?.filter("i > 10 and i < 20"),
                "LancePushdownScan: uri=..., projection=[s], predicate=i > Int32(10) AND i < Int32(20), row_id=false, row_addr=false, ordered=true"
            ).await?;
        }

        log::info!("Test case: Project and filter");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[s@2 as s]
  Take: columns=\"i, _rowid, (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: i@0 > 10 AND i@0 < 20
        LanceScan: uri..., projection=[i], row_id=true, row_addr=false, ordered=true, range=None"
        } else {
            "ProjectionExec: expr=[s@2 as s]
  Take: columns=\"i, _rowid, (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      LanceRead: ..., projection=[i], num_fragments=2, range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=i > Int32(10) AND i < Int32(20), refine_filter=i > Int32(10) AND i < Int32(20)"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.use_stats(false)
                    .project(&["s"])?
                    .filter("i > 10 and i < 20")
            },
            expected,
        )
        .await?;

        // Integer fields will be eagerly materialized while string/vec fields
        // are not.
        log::info!("Test case: Late materialization");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[i@0 as i, s@1 as s, vec@3 as vec]
            Take: columns=\"i, s, _rowid, (vec)\"
              CoalesceBatchesExec: target_batch_size=8192
                FilterExec: s@1 IS NOT NULL
                  LanceScan: uri..., projection=[i, s], row_id=true, row_addr=false, ordered=true, range=None"
        } else {
            "ProjectionExec: expr=[i@0 as i, s@1 as s, vec@3 as vec]
  Take: columns=\"i, s, _rowid, (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      LanceRead: uri=..., projection=[i, s], num_fragments=2, range_before=None, range_after=None, \
      row_id=true, row_addr=false, full_filter=s IS NOT NULL, refine_filter=s IS NOT NULL"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.use_stats(false).filter("s IS NOT NULL"),
            expected,
        )
        .await?;

        // Custom materialization
        log::info!("Test case: Custom materialization (all early)");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[i@0 as i, s@1 as s, vec@2 as vec]
  FilterExec: s@1 IS NOT NULL
    LanceScan: uri..., projection=[i, s, vec], row_id=true, row_addr=false, ordered=true, range=None"
        } else {
            "ProjectionExec: expr=[i@0 as i, s@1 as s, vec@2 as vec]
  LanceRead: uri=..., projection=[i, s, vec], num_fragments=2, range_before=None, \
  range_after=None, row_id=true, row_addr=false, full_filter=s IS NOT NULL, refine_filter=s IS NOT NULL"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.use_stats(false)
                    .materialization_style(MaterializationStyle::AllEarly)
                    .filter("s IS NOT NULL")
            },
            expected,
        )
        .await?;

        log::info!("Test case: Custom materialization 2 (all late)");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[i@2 as i, s@0 as s, vec@3 as vec]
  Take: columns=\"s, _rowid, (i), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: s@0 IS NOT NULL
        LanceScan: uri..., projection=[s], row_id=true, row_addr=false, ordered=true, range=None"
        } else {
            "ProjectionExec: expr=[i@2 as i, s@0 as s, vec@3 as vec]
  Take: columns=\"s, _rowid, (i), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      LanceRead: uri=..., projection=[s], num_fragments=2, range_before=None, \
      range_after=None, row_id=true, row_addr=false, full_filter=s IS NOT NULL, refine_filter=s IS NOT NULL"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.use_stats(false)
                    .materialization_style(MaterializationStyle::AllLate)
                    .filter("s IS NOT NULL")
            },
            expected,
        )
        .await?;

        log::info!("Test case: Custom materialization 3 (mixed)");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[i@3 as i, s@0 as s, vec@1 as vec]
  Take: columns=\"s, vec, _rowid, (i)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: s@0 IS NOT NULL
        LanceScan: uri..., projection=[s, vec], row_id=true, row_addr=false, ordered=true, range=None"
        } else {
            "ProjectionExec: expr=[i@3 as i, s@0 as s, vec@1 as vec]
  Take: columns=\"s, vec, _rowid, (i)\"
    CoalesceBatchesExec: target_batch_size=8192
      LanceRead: uri=..., projection=[s, vec], num_fragments=2, range_before=None, range_after=None, \
      row_id=true, row_addr=false, full_filter=s IS NOT NULL, refine_filter=s IS NOT NULL"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.use_stats(false)
                    .materialization_style(
                        MaterializationStyle::all_early_except(&["i"], lance_schema).unwrap(),
                    )
                    .filter("s IS NOT NULL")
            },
            expected,
        )
        .await?;

        log::info!("Test case: Scan out of order");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "LanceScan: uri=..., projection=[s], row_id=true, row_addr=false, ordered=false, range=None"
        } else {
            "LanceRead: uri=..., projection=[s], num_fragments=2, range_before=None, range_after=None, row_id=true, \
            row_addr=false, full_filter=--, refine_filter=--"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| Ok(scan.project(&["s"])?.with_row_id().scan_in_order(false)),
            expected,
        )
        .await?;

        // KNN
        // ---------------------------------------------------------------------
        let q: Float32Array = (32..32 + dim).map(|v| v as f32).collect();
        log::info!("Test case: Basic KNN");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@0 as vec, _distance@2 as _distance]
  Take: columns=\"vec, _rowid, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@2 IS NOT NULL
        SortExec: TopK(fetch=5), expr=...
          KNNVectorDistance: metric=l2
            LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false, range=None"
        } else {
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@0 as vec, _distance@2 as _distance]
  Take: columns=\"vec, _rowid, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@2 IS NOT NULL
        SortExec: TopK(fetch=5), expr=...
          KNNVectorDistance: metric=l2
            LanceRead: uri=..., projection=[vec], num_fragments=2, range_before=None, range_after=None, \
            row_id=true, row_addr=false, full_filter=--, refine_filter=--"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.nearest("vec", &q, 5),
            expected,
        )
        .await?;

        // KNN + Limit (arguably the user, or us, should fold the limit into the KNN but we don't today)
        // ---------------------------------------------------------------------
        let q: Float32Array = (32..32 + dim).map(|v| v as f32).collect();
        log::info!("Test case: KNN with extraneous limit");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@0 as vec, _distance@2 as _distance]
  Take: columns=\"vec, _rowid, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      GlobalLimitExec: skip=0, fetch=1
        FilterExec: _distance@2 IS NOT NULL
          SortExec: TopK(fetch=5), expr=...
            KNNVectorDistance: metric=l2
              LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false, range=None"
        } else {
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@0 as vec, _distance@2 as _distance]
  Take: columns=\"vec, _rowid, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      GlobalLimitExec: skip=0, fetch=1
        FilterExec: _distance@2 IS NOT NULL
          SortExec: TopK(fetch=5), expr=...
            KNNVectorDistance: metric=l2
              LanceRead: uri=..., projection=[vec], num_fragments=2, range_before=None, range_after=None, \
              row_id=true, row_addr=false, full_filter=--, refine_filter=--"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.nearest("vec", &q, 5)?.limit(Some(1), None),
            expected,
        )
        .await?;

        // ANN
        // ---------------------------------------------------------------------
        dataset.make_vector_index().await?;
        log::info!("Test case: Basic ANN");
        let expected =
            "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  Take: columns=\"_distance, _rowid, (i), (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: TopK(fetch=42), expr=...
        ANNSubIndex: name=..., k=42, deltas=1
          ANNIvfPartition: uuid=..., minimum_nprobes=20, maximum_nprobes=None, deltas=1";
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.nearest("vec", &q, 42),
            expected,
        )
        .await?;

        log::info!("Test case: ANN with refine");
        let expected =
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
                    ANNIvfPartition: uuid=..., minimum_nprobes=20, maximum_nprobes=None, deltas=1";
        assert_plan_equals(
            &dataset.dataset,
            |scan| Ok(scan.nearest("vec", &q, 10)?.refine(4)),
            expected,
        )
        .await?;

        // use_index = False -> same plan as KNN
        log::info!("Test case: ANN with index disabled");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@0 as vec, _distance@2 as _distance]
  Take: columns=\"vec, _rowid, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=13), expr=...
          KNNVectorDistance: metric=l2
            LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false, range=None"
        } else {
            "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@0 as vec, _distance@2 as _distance]
  Take: columns=\"vec, _rowid, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=13), expr=...
          KNNVectorDistance: metric=l2
            LanceRead: uri=..., projection=[vec], num_fragments=2, range_before=None, range_after=None, \
            row_id=true, row_addr=false, full_filter=--, refine_filter=--"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| Ok(scan.nearest("vec", &q, 13)?.use_index(false)),
            expected,
        )
        .await?;

        log::info!("Test case: ANN with postfilter");
        let expected = "ProjectionExec: expr=[s@3 as s, vec@4 as vec, _distance@0 as _distance, _rowid@1 as _rowid]
  Take: columns=\"_distance, _rowid, i, (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: i@2 > 10
        Take: columns=\"_distance, _rowid, (i)\"
          CoalesceBatchesExec: target_batch_size=8192
            SortExec: TopK(fetch=17), expr=...
              ANNSubIndex: name=..., k=17, deltas=1
                ANNIvfPartition: uuid=..., minimum_nprobes=20, maximum_nprobes=None, deltas=1";
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                Ok(scan
                    .nearest("vec", &q, 17)?
                    .filter("i > 10")?
                    .project(&["s", "vec"])?
                    .with_row_id())
            },
            expected,
        )
        .await?;

        log::info!("Test case: ANN with prefilter");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  Take: columns=\"_distance, _rowid, (i), (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: TopK(fetch=17), expr=...
        ANNSubIndex: name=..., k=17, deltas=1
          ANNIvfPartition: uuid=..., minimum_nprobes=20, maximum_nprobes=None, deltas=1
          FilterExec: i@0 > 10
            LanceScan: uri=..., projection=[i], row_id=true, row_addr=false, ordered=false, range=None"
        } else {
            "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  Take: columns=\"_distance, _rowid, (i), (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: TopK(fetch=17), expr=...
        ANNSubIndex: name=..., k=17, deltas=1
          ANNIvfPartition: uuid=..., minimum_nprobes=20, maximum_nprobes=None, deltas=1
          LanceRead: uri=..., projection=[], num_fragments=2, range_before=None, range_after=None, \
          row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=i > Int32(10)
"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                Ok(scan
                    .nearest("vec", &q, 17)?
                    .filter("i > 10")?
                    .prefilter(true))
            },
            expected,
        )
        .await?;

        dataset.append_new_data().await?;
        log::info!("Test case: Combined KNN/ANN");
        let expected = "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
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
                        LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false, range=None
                Take: columns=\"_distance, _rowid, (vec)\"
                  CoalesceBatchesExec: target_batch_size=8192
                    SortExec: TopK(fetch=6), expr=...
                      ANNSubIndex: name=..., k=6, deltas=1
                        ANNIvfPartition: uuid=..., minimum_nprobes=20, maximum_nprobes=None, deltas=1";
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.nearest("vec", &q, 6),
            // TODO: we could write an optimizer rule to eliminate the last Projection
            // by doing it as part of the last Take. This would likely have minimal impact though.
            expected,
        )
        .await?;

        // new data and with filter
        log::info!("Test case: Combined KNN/ANN with postfilter");
        let expected = "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
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
                              LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false, range=None
                      Take: columns=\"_distance, _rowid, (vec)\"
                        CoalesceBatchesExec: target_batch_size=8192
                          SortExec: TopK(fetch=15), expr=...
                            ANNSubIndex: name=..., k=15, deltas=1
                              ANNIvfPartition: uuid=..., minimum_nprobes=20, maximum_nprobes=None, deltas=1";
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.nearest("vec", &q, 15)?.filter("i > 10"),
            expected,
        )
        .await?;

        // new data and with prefilter
        log::info!("Test case: Combined KNN/ANN with prefilter");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
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
                          LanceScan: uri=..., projection=[vec, i], row_id=true, row_addr=false, ordered=false, range=None
                Take: columns=\"_distance, _rowid, (vec)\"
                  CoalesceBatchesExec: target_batch_size=8192
                    SortExec: TopK(fetch=5), expr=...
                      ANNSubIndex: name=..., k=5, deltas=1
                        ANNIvfPartition: uuid=..., minimum_nprobes=20, maximum_nprobes=None, deltas=1
                        FilterExec: i@0 > 10
                          LanceScan: uri=..., projection=[i], row_id=true, row_addr=false, ordered=false, range=None"
        } else {
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
                          LanceScan: uri=..., projection=[vec, i], row_id=true, row_addr=false, ordered=false, range=None
                Take: columns=\"_distance, _rowid, (vec)\"
                  CoalesceBatchesExec: target_batch_size=8192
                    SortExec: TopK(fetch=5), expr=...
                      ANNSubIndex: name=..., k=5, deltas=1
                        ANNIvfPartition: uuid=..., minimum_nprobes=20, maximum_nprobes=None, deltas=1
                        LanceRead: uri=..., projection=[], num_fragments=2, range_before=None, range_after=None, \
                          row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=i > Int32(10)"
        };
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
            expected,
        )
        .await?;

        // ANN with scalar index
        // ---------------------------------------------------------------------
        // Make sure both indices are up-to-date to start
        dataset.make_vector_index().await?;
        dataset.make_scalar_index().await?;

        log::info!("Test case: ANN with scalar index");
        let expected =
            "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  Take: columns=\"_distance, _rowid, (i), (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: TopK(fetch=5), expr=...
        ANNSubIndex: name=..., k=5, deltas=1
          ANNIvfPartition: uuid=..., minimum_nprobes=20, maximum_nprobes=None, deltas=1
          ScalarIndexQuery: query=[i > 10]@i_idx";
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                Ok(scan
                    .nearest("vec", &q, 5)?
                    .filter("i > 10")?
                    .prefilter(true))
            },
            expected,
        )
        .await?;

        log::info!("Test case: ANN with scalar index disabled");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  Take: columns=\"_distance, _rowid, (i), (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: TopK(fetch=5), expr=...
        ANNSubIndex: name=..., k=5, deltas=1
          ANNIvfPartition: uuid=..., minimum_nprobes=20, maximum_nprobes=None, deltas=1
          FilterExec: i@0 > 10
            LanceScan: uri=..., projection=[i], row_id=true, row_addr=false, ordered=false, range=None"
        } else {
            "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  Take: columns=\"_distance, _rowid, (i), (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: TopK(fetch=5), expr=...
        ANNSubIndex: name=..., k=5, deltas=1
          ANNIvfPartition: uuid=..., minimum_nprobes=20, maximum_nprobes=None, deltas=1
          LanceRead: uri=..., projection=[], num_fragments=3, range_before=None, \
          range_after=None, row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=i > Int32(10)"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                Ok(scan
                    .nearest("vec", &q, 5)?
                    .use_scalar_index(false)
                    .filter("i > 10")?
                    .prefilter(true))
            },
            expected,
        )
        .await?;

        dataset.append_new_data().await?;

        log::info!("Test case: Combined KNN/ANN with scalar index");
        let expected = "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
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
                          LanceScan: uri=..., projection=[vec, i], row_id=true, row_addr=false, ordered=false, range=None
                Take: columns=\"_distance, _rowid, (vec)\"
                  CoalesceBatchesExec: target_batch_size=8192
                    SortExec: TopK(fetch=8), expr=...
                      ANNSubIndex: name=..., k=8, deltas=1
                        ANNIvfPartition: uuid=..., minimum_nprobes=20, maximum_nprobes=None, deltas=1
                        ScalarIndexQuery: query=[i > 10]@i_idx";
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                Ok(scan
                    .nearest("vec", &q, 8)?
                    .filter("i > 10")?
                    .prefilter(true))
            },
            expected,
        )
        .await?;

        // Update scalar index but not vector index
        log::info!(
            "Test case: Combined KNN/ANN with updated scalar index and outdated vector index"
        );
        let expected = "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
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
                          LanceScan: uri=..., projection=[vec, i], row_id=true, row_addr=false, ordered=false, range=None
                Take: columns=\"_distance, _rowid, (vec)\"
                  CoalesceBatchesExec: target_batch_size=8192
                    SortExec: TopK(fetch=11), expr=...
                      ANNSubIndex: name=..., k=11, deltas=1
                        ANNIvfPartition: uuid=..., minimum_nprobes=20, maximum_nprobes=None, deltas=1
                        ScalarIndexQuery: query=[i > 10]@i_idx";
        dataset.make_scalar_index().await?;
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                Ok(scan
                    .nearest("vec", &q, 11)?
                    .filter("i > 10")?
                    .prefilter(true))
            },
            expected,
        )
        .await?;

        // Scans with scalar index
        // ---------------------------------------------------------------------
        log::info!("Test case: Filtered read with scalar index");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[s@1 as s]
  Take: columns=\"_rowid, (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      MaterializeIndex: query=[i > 10]@i_idx"
        } else {
            "LanceRead: uri=..., projection=[s], num_fragments=4, range_before=None, \
            range_after=None, row_id=false, row_addr=false, full_filter=i > Int32(10), refine_filter=--
              ScalarIndexQuery: query=[i > 10]@i_idx"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.project(&["s"])?.filter("i > 10"),
            expected,
        )
        .await?;

        if data_storage_version != LanceFileVersion::Legacy {
            log::info!(
                "Test case: Filtered read with scalar index disabled (late materialization)"
            );
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
      LanceRead: uri=..., projection=[i], num_fragments=4, range_before=None, \
      range_after=None, row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=i > Int32(10)",
            )
            .await?;
        }

        log::info!("Test case: Empty projection");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[_rowaddr@0 as _rowaddr]
  AddRowAddrExec
    MaterializeIndex: query=[i > 10]@i_idx"
        } else {
            "LanceRead: uri=..., projection=[], num_fragments=4, range_before=None, \
            range_after=None, row_id=false, row_addr=true, full_filter=i > Int32(10), refine_filter=--
              ScalarIndexQuery: query=[i > 10]@i_idx"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.filter("i > 10")
                    .unwrap()
                    .with_row_address()
                    .project::<&str>(&[])
            },
            expected,
        )
        .await?;

        dataset.append_new_data().await?;
        log::info!("Test case: Combined Scalar/non-scalar filtered read");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[s@1 as s]
  RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
    UnionExec
      Take: columns=\"_rowid, (s)\"
        CoalesceBatchesExec: target_batch_size=8192
          MaterializeIndex: query=[i > 10]@i_idx
      ProjectionExec: expr=[_rowid@2 as _rowid, s@1 as s]
        FilterExec: i@0 > 10
          LanceScan: uri=..., projection=[i, s], row_id=true, row_addr=false, ordered=false, range=None"
        } else {
            "LanceRead: uri=..., projection=[s], num_fragments=5, range_before=None, \
            range_after=None, row_id=false, row_addr=false, full_filter=i > Int32(10), refine_filter=--
              ScalarIndexQuery: query=[i > 10]@i_idx"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.project(&["s"])?.filter("i > 10"),
            expected,
        )
        .await?;

        log::info!("Test case: Combined Scalar/non-scalar filtered read with empty projection");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[_rowaddr@0 as _rowaddr]
  RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
    UnionExec
      AddRowAddrExec
        MaterializeIndex: query=[i > 10]@i_idx
      ProjectionExec: expr=[_rowaddr@2 as _rowaddr, _rowid@1 as _rowid]
        FilterExec: i@0 > 10
          LanceScan: uri=..., projection=[i], row_id=true, row_addr=true, ordered=false, range=None"
        } else {
            "LanceRead: uri=..., projection=[], num_fragments=5, range_before=None, \
            range_after=None, row_id=false, row_addr=true, full_filter=i > Int32(10), refine_filter=--
              ScalarIndexQuery: query=[i > 10]@i_idx"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.filter("i > 10")
                    .unwrap()
                    .with_row_address()
                    .project::<&str>(&[])
            },
            expected,
        )
        .await?;

        // Scans with dynamic projection
        // When an expression is specified in the projection, the plan should include a ProjectionExec
        log::info!("Test case: Dynamic projection");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            "ProjectionExec: expr=[regexp_match(s@1, .*) as matches]
  RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
    UnionExec
      Take: columns=\"_rowid, (s)\"
        CoalesceBatchesExec: target_batch_size=8192
          MaterializeIndex: query=[i > 10]@i_idx
      ProjectionExec: expr=[_rowid@2 as _rowid, s@1 as s]
        FilterExec: i@0 > 10
          LanceScan: uri=..., row_id=true, row_addr=false, ordered=false, range=None"
        } else {
            "ProjectionExec: expr=[regexp_match(s@0, .*) as matches]
  LanceRead: uri=..., projection=[s], num_fragments=5, range_before=None, \
  range_after=None, row_id=false, row_addr=false, full_filter=i > Int32(10), refine_filter=--
    ScalarIndexQuery: query=[i > 10]@i_idx"
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.project_with_transform(&[("matches", "regexp_match(s, \".*\")")])?
                    .filter("i > 10")
            },
            expected,
        )
        .await?;

        // FTS
        // ---------------------------------------------------------------------
        // All rows are indexed
        dataset.make_fts_index().await?;
        log::info!("Test case: Full text search (match query)");
        let expected = r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      MatchQuery: query=hello"#;
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.project(&["s"])?
                    .with_row_id()
                    .full_text_search(FullTextSearchQuery::new("hello".to_owned()))
            },
            expected,
        )
        .await?;

        log::info!("Test case: Full text search (phrase query)");
        let expected = r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      PhraseQuery: query=hello world"#;
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                let query = PhraseQuery::new("hello world".to_owned());
                scan.project(&["s"])?
                    .with_row_id()
                    .full_text_search(FullTextSearchQuery::new_query(query.into()))
            },
            expected,
        )
        .await?;

        log::info!("Test case: Full text search (boost query)");
        let expected = r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      BoostQuery: negative_boost=1
        MatchQuery: query=hello
        MatchQuery: query=world"#;
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                let positive =
                    MatchQuery::new("hello".to_owned()).with_column(Some("s".to_owned()));
                let negative =
                    MatchQuery::new("world".to_owned()).with_column(Some("s".to_owned()));
                let query = BoostQuery::new(positive.into(), negative.into(), Some(1.0));
                scan.project(&["s"])?
                    .with_row_id()
                    .full_text_search(FullTextSearchQuery::new_query(query.into()))
            },
            expected,
        )
        .await?;

        log::info!("Test case: Full text search with prefilter");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      MatchQuery: query=hello
        RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
          UnionExec
            MaterializeIndex: query=[i > 10]@i_idx
            ProjectionExec: expr=[_rowid@1 as _rowid]
              FilterExec: i@0 > 10
                LanceScan: uri=..., projection=[i], row_id=true, row_addr=false, ordered=false, range=None"#
        } else {
            r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      MatchQuery: query=hello
        LanceRead: uri=..., projection=[], num_fragments=5, range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=--
          ScalarIndexQuery: query=[i > 10]@i_idx"#
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.project(&["s"])?
                    .with_row_id()
                    .filter("i > 10")?
                    .prefilter(true)
                    .full_text_search(FullTextSearchQuery::new("hello".to_owned()))
            },
            expected,
        )
        .await?;

        log::info!("Test case: Full text search with unindexed rows");
        let expected = r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: expr=[_score@1 DESC NULLS LAST], preserve_partitioning=[false]
        RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
          UnionExec
            MatchQuery: query=hello
            FlatMatchQuery: query=hello
              LanceScan: uri=..., projection=[s], row_id=true, row_addr=false, ordered=false, range=None"#;
        dataset.append_new_data().await?;
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.project(&["s"])?
                    .with_row_id()
                    .full_text_search(FullTextSearchQuery::new("hello".to_owned()))
            },
            expected,
        )
        .await?;

        log::info!("Test case: Full text search with unindexed rows and prefilter");
        let expected = if data_storage_version == LanceFileVersion::Legacy {
            r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: expr=[_score@1 DESC NULLS LAST], preserve_partitioning=[false]
        RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
          UnionExec
            MatchQuery: query=hello
              RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
                UnionExec
                  MaterializeIndex: query=[i > 10]@i_idx
                  ProjectionExec: expr=[_rowid@1 as _rowid]
                    FilterExec: i@0 > 10
                      LanceScan: uri=..., projection=[i], row_id=true, row_addr=false, ordered=false, range=None
            FlatMatchQuery: query=hello
              FilterExec: i@1 > 10
                LanceScan: uri=..., projection=[s, i], row_id=true, row_addr=false, ordered=false, range=None"#
        } else {
            r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: expr=[_score@1 DESC NULLS LAST], preserve_partitioning=[false]
        RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
          UnionExec
            MatchQuery: query=hello
              LanceRead: uri=..., projection=[], num_fragments=5, range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=--
                ScalarIndexQuery: query=[i > 10]@i_idx
            FlatMatchQuery: query=hello
              FilterExec: i@1 > 10
                LanceScan: uri=..., projection=[s, i], row_id=true, row_addr=false, ordered=false, range=None"#
        };
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.project(&["s"])?
                    .with_row_id()
                    .filter("i > 10")?
                    .prefilter(true)
                    .full_text_search(FullTextSearchQuery::new("hello".to_owned()))
            },
            expected,
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
                    .project(&["_distance", "_rowid"])
            },
            "SortExec: TopK(fetch=32), expr=[_distance@0 ASC NULLS LAST, _rowid@1 ASC NULLS LAST]...
    ANNSubIndex: name=idx, k=32, deltas=1
      ANNIvfPartition: uuid=..., minimum_nprobes=20, maximum_nprobes=None, deltas=1",
        )
        .await
        .unwrap();

        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.nearest("vec", &q, 33)?
                    .fast_search()
                    .with_row_id()
                    .project(&["_distance", "_rowid"])
            },
            "SortExec: TopK(fetch=33), expr=[_distance@0 ASC NULLS LAST, _rowid@1 ASC NULLS LAST]...
    ANNSubIndex: name=idx, k=33, deltas=1
      ANNIvfPartition: uuid=..., minimum_nprobes=20, maximum_nprobes=None, deltas=1",
        )
        .await
        .unwrap();

        // Not `fast_scan` case
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.nearest("vec", &q, 34)?
                    .with_row_id()
                    .project(&["_distance", "_rowid"])
            },
            "ProjectionExec: expr=[_distance@2 as _distance, _rowid@0 as _rowid]
  FilterExec: _distance@2 IS NOT NULL
    SortExec: TopK(fetch=34), expr=[_distance@2 ASC NULLS LAST, _rowid@0 ASC NULLS LAST]...
      KNNVectorDistance: metric=l2
        RepartitionExec: partitioning=RoundRobinBatch(1), input_partitions=2
          UnionExec
            ProjectionExec: expr=[_distance@2 as _distance, _rowid@1 as _rowid, vec@0 as vec]
              FilterExec: _distance@2 IS NOT NULL
                SortExec: TopK(fetch=34), expr=[_distance@2 ASC NULLS LAST, _rowid@1 ASC NULLS LAST]...
                  KNNVectorDistance: metric=l2
                    LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false, range=None
            Take: columns=\"_distance, _rowid, (vec)\"
              CoalesceBatchesExec: target_batch_size=8192
                SortExec: TopK(fetch=34), expr=[_distance@0 ASC NULLS LAST, _rowid@1 ASC NULLS LAST]...
                  ANNSubIndex: name=idx, k=34, deltas=1
                    ANNIvfPartition: uuid=..., minimum_nprobes=20, maximum_nprobes=None, deltas=1",
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

        use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
        let data = gen_batch()
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
                            num_partitions: Some(2),
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

    #[rstest]
    #[tokio::test]
    pub async fn test_row_meta_columns(
        #[values(
            (true, false),  // Test row_id only
            (false, true),  // Test row_address only
            (true, true)    // Test both
        )]
        columns: (bool, bool),
    ) {
        let (with_row_id, with_row_address) = columns;
        let test_dir = tempdir().unwrap();
        let uri = test_dir.path().to_str().unwrap();

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("data_item_id", arrow_schema::DataType::Int32, false),
            arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false),
        ]));

        let data = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1001, 1002, 1003])),
                Arc::new(Int32Array::from(vec![1, 2, 3])),
            ],
        )
        .unwrap();

        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(data)], schema.clone()),
            uri,
            None,
        )
        .await
        .unwrap();

        // Test explicit projection
        let mut scanner = dataset.scan();

        let mut projection = vec!["data_item_id".to_string()];
        if with_row_id {
            scanner.with_row_id();
            projection.push(ROW_ID.to_string());
        }
        if with_row_address {
            scanner.with_row_address();
            projection.push(ROW_ADDR.to_string());
        }

        scanner.project(&projection).unwrap();
        let stream = scanner.try_into_stream().await.unwrap();
        let batch = stream.try_collect::<Vec<_>>().await.unwrap().pop().unwrap();

        // Verify column existence and data type
        if with_row_id {
            let column = batch.column_by_name(ROW_ID).unwrap();
            assert_eq!(column.data_type(), &DataType::UInt64);
        }
        if with_row_address {
            let column = batch.column_by_name(ROW_ADDR).unwrap();
            assert_eq!(column.data_type(), &DataType::UInt64);
        }

        // Test implicit inclusion
        let mut scanner = dataset.scan();
        if with_row_id {
            scanner.with_row_id();
        }
        if with_row_address {
            scanner.with_row_address();
        }
        scanner.project(&["data_item_id"]).unwrap();
        let stream = scanner.try_into_stream().await.unwrap();
        let batch = stream.try_collect::<Vec<_>>().await.unwrap().pop().unwrap();
        let meta_column = batch.column_by_name(if with_row_id { ROW_ID } else { ROW_ADDR });
        assert!(meta_column.is_some());

        // Test error case
        let mut scanner = dataset.scan();
        if with_row_id {
            scanner.project(&[ROW_ID]).unwrap();
        } else {
            scanner.project(&[ROW_ADDR]).unwrap();
        };
        let stream = scanner.try_into_stream().await.unwrap();
        assert_eq!(stream.schema().fields().len(), 1);
        if with_row_id {
            assert!(stream.schema().field_with_name(ROW_ID).is_ok());
        } else {
            assert!(stream.schema().field_with_name(ROW_ADDR).is_ok());
        }
    }

    async fn limit_offset_equivalency_test(scanner: &Scanner) {
        async fn test_one(
            scanner: &Scanner,
            full_result: &RecordBatch,
            limit: Option<i64>,
            offset: Option<i64>,
        ) {
            let mut new_scanner = scanner.clone();
            new_scanner.limit(limit, offset).unwrap();
            if let Some(nearest) = new_scanner.nearest_mut() {
                nearest.k = offset.unwrap_or(0).saturating_add(limit.unwrap_or(10_000)) as usize;
            }
            let result = new_scanner.try_into_batch().await.unwrap();

            let resolved_offset = offset.unwrap_or(0).min(full_result.num_rows() as i64);
            let resolved_length = limit
                .unwrap_or(i64::MAX)
                .min(full_result.num_rows() as i64 - resolved_offset);

            let expected = full_result.slice(resolved_offset as usize, resolved_length as usize);

            if expected != result {
                let plan = new_scanner.analyze_plan().await.unwrap();
                assert_eq!(
                    &expected, &result,
                    "Limit: {:?}, Offset: {:?}, Plan: \n{}",
                    limit, offset, plan
                );
            }
        }

        let mut scanner_full = scanner.clone();
        if let Some(nearest) = scanner_full.nearest_mut() {
            nearest.k = 500;
        }
        let full_results = scanner_full.try_into_batch().await.unwrap();

        test_one(scanner, &full_results, Some(1), None).await;
        test_one(scanner, &full_results, Some(1), Some(1)).await;
        test_one(scanner, &full_results, Some(1), Some(2)).await;
        test_one(scanner, &full_results, Some(1), Some(10)).await;

        test_one(scanner, &full_results, Some(3), None).await;
        test_one(scanner, &full_results, Some(3), Some(2)).await;
        test_one(scanner, &full_results, Some(3), Some(4)).await;

        test_one(scanner, &full_results, None, Some(3)).await;
        test_one(scanner, &full_results, None, Some(10)).await;
    }

    #[tokio::test]
    async fn test_scan_limit_offset() {
        let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        let scanner = test_ds.dataset.scan();
        limit_offset_equivalency_test(&scanner).await;
    }

    #[tokio::test]
    async fn test_knn_limit_offset() {
        let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        let query_vector = Float32Array::from(vec![0.0; 32]);
        let mut scanner = test_ds.dataset.scan();
        scanner
            .nearest("vec", &query_vector, 5)
            .unwrap()
            .project(&["i"])
            .unwrap();
        limit_offset_equivalency_test(&scanner).await;
    }

    #[tokio::test]
    async fn test_ivf_pq_limit_offset() {
        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        test_ds.make_vector_index().await.unwrap();
        test_ds.append_new_data().await.unwrap();
        let query_vector = Float32Array::from(vec![0.0; 32]);
        let mut scanner = test_ds.dataset.scan();
        scanner.nearest("vec", &query_vector, 500).unwrap();
        limit_offset_equivalency_test(&scanner).await;
    }

    #[tokio::test]
    async fn test_fts_limit_offset() {
        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        test_ds.make_fts_index().await.unwrap();
        test_ds.append_new_data().await.unwrap();
        let mut scanner = test_ds.dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new("4".into()))
            .unwrap();
        limit_offset_equivalency_test(&scanner).await;
    }

    async fn test_row_offset_read_helper(
        ds: &Dataset,
        scan_builder: impl FnOnce(&mut Scanner) -> &mut Scanner,
        expected_cols: &[&str],
        expected_row_offsets: &[u64],
    ) {
        let mut scanner = ds.scan();
        let scanner = scan_builder(&mut scanner);
        let stream = scanner.try_into_stream().await.unwrap();

        let schema = stream.schema();
        let actual_cols = schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>();
        assert_eq!(&actual_cols, expected_cols);

        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        let batch = arrow_select::concat::concat_batches(&schema, &batches).unwrap();

        let row_offsets = batch
            .column_by_name(ROW_OFFSET)
            .unwrap()
            .as_primitive::<UInt64Type>()
            .values();
        assert_eq!(row_offsets.as_ref(), expected_row_offsets);
    }

    #[tokio::test]
    async fn test_row_offset_read() {
        let mut ds = lance_datagen::gen_batch()
            .col("idx", array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(3), FragmentRowCount::from(3))
            .await
            .unwrap();
        // [0, 1, 2], [3, 4, 5], [6, 7, 8]

        // Delete [2, 3, 4, 5, 6]
        ds.delete("idx >= 2 AND idx <= 6").await.unwrap();

        // Normal read, all columns plus row offset
        test_row_offset_read_helper(
            &ds,
            |scanner| scanner.project(&["idx", ROW_OFFSET]).unwrap(),
            &["idx", ROW_OFFSET],
            &[0, 1, 2, 3],
        )
        .await;

        // Read with row offset only
        test_row_offset_read_helper(
            &ds,
            |scanner| scanner.project(&[ROW_OFFSET]).unwrap(),
            &[ROW_OFFSET],
            &[0, 1, 2, 3],
        )
        .await;

        // Filtered read of row offset
        test_row_offset_read_helper(
            &ds,
            |scanner| {
                scanner
                    .filter("idx > 1")
                    .unwrap()
                    .project(&[ROW_OFFSET])
                    .unwrap()
            },
            &[ROW_OFFSET],
            &[2, 3],
        )
        .await;
    }

    #[tokio::test]
    async fn test_filter_to_take() {
        let mut ds = lance_datagen::gen_batch()
            .col("idx", array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(3), FragmentRowCount::from(100))
            .await
            .unwrap();

        let row_ids = ds
            .scan()
            .project(&Vec::<&str>::default())
            .unwrap()
            .with_row_id()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let schema = row_ids[0].schema();
        let row_ids = concat_batches(&schema, row_ids.iter()).unwrap();
        let row_ids = row_ids.column(0).as_primitive::<UInt64Type>().clone();

        let row_addrs = ds
            .scan()
            .project(&Vec::<&str>::default())
            .unwrap()
            .with_row_address()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let schema = row_addrs[0].schema();
        let row_addrs = concat_batches(&schema, row_addrs.iter()).unwrap();
        let row_addrs = row_addrs.column(0).as_primitive::<UInt64Type>().clone();

        ds.delete("idx >= 190 AND idx < 210").await.unwrap();

        let ds_copy = ds.clone();
        let do_check = async move |filt: &str, expected_idx: &[i32], applies_optimization: bool| {
            let mut scanner = ds_copy.scan();
            scanner.filter(filt).unwrap();
            // Verify the optimization is applied
            let plan = scanner.explain_plan(true).await.unwrap();
            if applies_optimization {
                assert!(
                    plan.contains("OneShotStream"),
                    "expected take optimization to be applied. Filter: '{}'.  Plan:\n{}",
                    filt,
                    plan
                );
            } else {
                assert!(
                    !plan.contains("OneShotStream"),
                    "expected take optimization to not be applied. Filter: '{}'.  Plan:\n{}",
                    filt,
                    plan
                );
            }

            // Verify the results
            let stream = scanner.try_into_stream().await.unwrap();
            let batches = stream.try_collect::<Vec<_>>().await.unwrap();
            let idx = batches
                .iter()
                .map(|b| b.column_by_name("idx").unwrap().as_ref())
                .collect::<Vec<_>>();

            if idx.is_empty() {
                assert!(expected_idx.is_empty());
                return;
            }

            let idx = arrow::compute::concat(&idx).unwrap();
            assert_eq!(idx.as_primitive::<Int32Type>().values(), expected_idx);
        };
        let check =
            async |filt: &str, expected_idx: &[i32]| do_check(filt, expected_idx, true).await;
        let check_no_opt = async |filt: &str, expected_idx: &[i32]| {
            do_check(filt, expected_idx, false).await;
        };

        // Simple case, no deletions yet
        check("_rowid = 50", &[50]).await;
        check("_rowaddr = 50", &[50]).await;
        check("_rowoffset = 50", &[50]).await;

        check(
            "_rowid = 50 OR _rowid = 51 OR _rowid = 52 OR _rowid = 49",
            &[49, 50, 51, 52],
        )
        .await;
        check(
            "_rowaddr = 50 OR _rowaddr = 51 OR _rowaddr = 52 OR _rowaddr = 49",
            &[49, 50, 51, 52],
        )
        .await;
        check(
            "_rowoffset = 50 OR _rowoffset = 51 OR _rowoffset = 52 OR _rowoffset = 49",
            &[49, 50, 51, 52],
        )
        .await;

        check("_rowid IN (52, 51, 50, 17)", &[17, 50, 51, 52]).await;
        check("_rowaddr IN (52, 51, 50, 17)", &[17, 50, 51, 52]).await;
        check("_rowoffset IN (52, 51, 50, 17)", &[17, 50, 51, 52]).await;

        // Taking _rowid / _rowaddr of deleted row

        // When using rowid / rowaddr we get an empty
        check(&format!("_rowid = {}", row_ids.value(190)), &[]).await;
        check(&format!("_rowaddr = {}", row_addrs.value(190)), &[]).await;
        // When using rowoffset it just skips the deleted rows (impossible to create an offset
        // into a deleted row)
        check("_rowoffset = 190", &[210]).await;

        // Grabbing after the deleted rows
        check(&format!("_rowid = {}", row_ids.value(250)), &[250]).await;
        check(&format!("_rowaddr = {}", row_addrs.value(250)), &[250]).await;
        check("_rowoffset = 250", &[270]).await;

        // Grabbing past the end
        check("_rowoffset = 1000", &[]).await;

        // Combine take and filter
        check("_rowid IN (5, 10, 15) AND idx > 10", &[15]).await;
        check("_rowaddr IN (5, 10, 15) AND idx > 10", &[15]).await;
        check("_rowoffset IN (5, 10, 15) AND idx > 10", &[15]).await;
        check("idx > 10 AND _rowid IN (5, 10, 15)", &[15]).await;
        check("idx > 10 AND _rowaddr IN (5, 10, 15)", &[15]).await;
        check("idx > 10 AND _rowoffset IN (5, 10, 15)", &[15]).await;
        // Get's simplified into _rowid = 50 and so we catch it
        check("_rowid = 50 AND _rowid = 50", &[50]).await;

        // Filters that cannot be converted into a take
        check_no_opt("_rowid = 50 AND _rowid = 51", &[]).await;
        check_no_opt("(_rowid = 50 AND idx < 100) OR _rowid = 51", &[50, 51]).await;

        // Dynamic projection
        let mut scanner = ds.scan();
        scanner.filter("_rowoffset = 77").unwrap();
        scanner
            .project_with_transform(&[("foo", "idx * 2")])
            .unwrap();
        let stream = scanner.try_into_stream().await.unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(batches[0].schema().field(0).name(), "foo");
        let val = batches[0].column(0).as_primitive::<Int32Type>().values()[0];
        assert_eq!(val, 154);
    }
}
