// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::any::Any;
use std::{ops::Range, sync::Arc};

use arrow::array::AsArray;
use arrow::datatypes::UInt32Type;
use arrow_array::RecordBatch;
use datafusion::common::stats::Precision;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    execution_plan::{Boundedness, EmissionType},
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
};
use datafusion_expr::Expr;
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::metrics::MetricsSet;
use datafusion_physical_plan::Statistics;
use futures::{StreamExt, TryStreamExt};
use lance_core::datatypes::OnMissing;
use lance_core::utils::mask::RowIdMask;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{datatypes::Projection, Error, Result};
use lance_datafusion::planner::Planner;
use lance_index::scalar::expression::{FilterPlan, IndexExprResult};
use lance_table::format::Fragment;
use roaring::RoaringBitmap;
use snafu::location;
use tokio::sync::Mutex as AsyncMutex;

use crate::Dataset;

// Re-export types that have been moved to planned_filter_read
pub use super::planned_filter_read::{
    DvToValidRanges, FilteredReadGlobalMetrics, FilteredReadStream, FilteredReadThreadingMode,
};

#[derive(Debug)]
pub struct EvaluatedIndex {
    pub(crate) index_result: IndexExprResult,
    pub(crate) applicable_fragments: RoaringBitmap,
}

impl EvaluatedIndex {
    pub fn try_from_arrow(batch: &RecordBatch) -> Result<Self> {
        if batch.num_rows() != 2 {
            return Err(Error::InvalidInput {
                source: format!(
                    "Expected a batch with exactly one row but there are {} rows",
                    batch.num_rows()
                )
                .into(),
                location: location!(),
            });
        }
        if batch.num_columns() != 3 {
            return Err(Error::InvalidInput {
                source: format!(
                    "Expected a batch with exactly two columns but there are {} columns",
                    batch.num_columns()
                )
                .into(),
                location: location!(),
            });
        }
        let row_id_mask = RowIdMask::from_arrow(batch.column(0).as_binary())?;
        let match_type = batch.column(1).as_primitive::<UInt32Type>().values()[0];
        let index_result = IndexExprResult::from_parts(row_id_mask, match_type)?;

        let applicable_fragments = batch.column(2).as_binary::<i32>();
        let applicable_fragments = RoaringBitmap::deserialize_from(applicable_fragments.value(0))?;

        Ok(Self {
            index_result,
            applicable_fragments,
        })
    }
}

/// Options for a filtered read.
#[derive(Debug, Clone)]
pub struct FilteredReadOptions {
    /// The range of rows to read before applying the filter.
    pub scan_range_before_filter: Option<Range<u64>>,
    /// The range of rows to read after applying the filter.
    pub scan_range_after_filter: Option<Range<u64>>,
    /// Include deleted rows in the scan
    pub with_deleted_rows: bool,
    /// The maximum number of rows per batch
    pub batch_size: Option<u32>,
    /// Controls how many fragments to read ahead
    pub fragment_readahead: Option<usize>,
    /// The fragments to read
    pub fragments: Option<Arc<Vec<Fragment>>>,
    /// The projection to use for the scan
    pub projection: Projection,
    /// If there is a scalar index input, and the index result we get from that input is exact,
    /// then we will only apply the refine filter to batches covered by the result.
    pub refine_filter: Option<Expr>,
    /// The filter to apply during the read.  If possible we will try and use the scalar index
    /// result to avoid applying this (and instead only apply the refine filter) but in some cases
    /// the index result does not cover all fragments or is not exact.
    pub full_filter: Option<Expr>,
    /// The threading mode to use for the scan
    pub threading_mode: FilteredReadThreadingMode,
}

impl FilteredReadOptions {
    /// Create a basic full scan of the dataset
    ///
    /// This will read all data, without any filters, and will read all
    /// columns (but not the row id or row address).  Deleted rows will
    /// not be included and the default batch size will be used.
    ///
    /// This is the default behavior and you can use the various builder
    /// methods on this type to modify the behavior.
    pub fn basic_full_read(dataset: &Arc<Dataset>) -> Self {
        Self::new(dataset.full_projection())
    }

    pub fn new(projection: Projection) -> Self {
        Self {
            scan_range_before_filter: None,
            scan_range_after_filter: None,
            with_deleted_rows: false,
            batch_size: None,
            fragment_readahead: None,
            fragments: None,
            projection,
            refine_filter: None,
            full_filter: None,
            threading_mode: FilteredReadThreadingMode::OnePartitionMultipleThreads(
                get_num_compute_intensive_cpus(),
            ),
        }
    }

    /// Include deleted rows in the scan
    ///
    /// This is currently only supported if there is no scan_range specified
    ///
    /// The projection will be updated to always include the row id column.  The
    /// row id column will be null for all deleted rows.
    ///
    /// This function only materializes deleted rows that are masked by a deletion
    /// vector.  If the deleted row has been materialized via compaction, or if an
    /// entire fragment was deleted, it will not be read by this function.
    pub fn with_deleted_rows(mut self) -> Result<Self> {
        if self.scan_range_before_filter.is_some() || self.scan_range_after_filter.is_some() {
            return Err(Error::InvalidInput {
                source: "with_deleted_rows is not supported when there is a scan range".into(),
                location: location!(),
            });
        }
        self.with_deleted_rows = true;
        Ok(self)
    }

    /// Specify the range of rows to read before applying the filter.
    ///
    /// This can be used to pushdown a limit/offset when there is no filter.
    ///
    /// It's also possible to specify this when there is a filter, in order to only scan
    /// a subset of the data (and apply the filter on this subset).  For example, if the
    /// data as a column `count` that steps from 0 to 1000 and the filter is `count > 200`
    /// and the range is 100..300, then scan will read rows 100..300 and return rows 200..300
    pub fn with_scan_range_before_filter(mut self, scan_range: Range<u64>) -> Result<Self> {
        if self.with_deleted_rows {
            return Err(Error::InvalidInput {
                source: "with_deleted_rows is not supported when there is a scan range".into(),
                location: location!(),
            });
        }
        self.scan_range_before_filter = Some(scan_range);
        Ok(self)
    }

    /// The range of rows to read after applying the filter.
    ///
    /// In many cases we are not able to push this down and the range will be applied after-the-fact.
    ///
    /// However, if there is a scalar index on the column, and that scalar index returns an exact
    /// match, then we can use this to skip reading the data entirely.
    ///
    /// We currently do not support setting this when there is more than one partition.
    pub fn with_scan_range_after_filter(mut self, scan_range: Range<u64>) -> Result<Self> {
        if self.with_deleted_rows {
            return Err(Error::InvalidInput {
                source: "with_deleted_rows is not supported when there is a scan range".into(),
                location: location!(),
            });
        }
        self.scan_range_after_filter = Some(scan_range);
        Ok(self)
    }

    /// Specify the fragments to read.
    ///
    /// Scan results will be returned in the order of the fragments given here.
    pub fn with_fragments(mut self, fragments: Arc<Vec<Fragment>>) -> Self {
        self.fragments = Some(fragments);
        self
    }

    /// Specify the batch size to use for the read
    ///
    /// This will be a maximum number of rows per batch.  It is possible for batches to be smaller
    /// either due to filtering or because we have reached the end of a fragment (we do not combine
    /// batches across fragments).
    ///
    /// A CoalesceBatchesExec can (and often should) be used to merge together tiny batches
    pub fn with_batch_size(mut self, batch_size: u32) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Controls how many fragments to read ahead.
    ///
    /// If not set, the default will be 2 * the I/O parallelism.  Generally, reading ahead
    /// in fragments is very cheap.  We will accumulate more I/O requests but these are very tiny.
    /// This has no significant impact on the RAM cost of the scan.  Backpressure is handled by the
    /// scheduler.
    pub fn with_fragment_readahead(mut self, fragment_readahead: usize) -> Self {
        self.fragment_readahead = Some(fragment_readahead);
        self
    }

    /// Specify the filter plan to use for the scan.
    ///
    /// This consists of up to two filters.  The full filter is the filter that needs to be satisfied
    /// by the read.
    ///
    /// The refine filter is a smaller filter that is applied to batches that have exact matches from the
    /// index search.  Since these batches matched the index exactly we already know some predicates about
    /// the rows in the batch and may not have to apply the full filter.
    ///
    /// If the full_filter is None then the refine_filter must be None.
    ///
    /// If the full_filter is Some and the refine_filter is None then that means the filter is completely
    /// satisfied by the index search.  If we get an exact match from the index search we can skip filtering
    /// entirely.
    pub fn with_filter(
        mut self,
        refine_filter: Option<Expr>,
        full_filter: Option<Expr>,
    ) -> Result<Self> {
        if refine_filter.is_some() && full_filter.is_none() {
            return Err(Error::InvalidInput {
                source: "refine_filter is set but full_filter is not".into(),
                location: location!(),
            });
        }
        self.refine_filter = refine_filter;
        self.full_filter = full_filter;
        Ok(self)
    }

    /// An alternative to [`Self::with_filter`] to set the filters from a FilterPlan if you already have one
    pub fn with_filter_plan(mut self, filter_plan: FilterPlan) -> Self {
        self.refine_filter = filter_plan.refine_expr;
        self.full_filter = filter_plan.full_expr;
        self
    }

    /// Specify the projection to use for the scan
    ///
    /// If the row id or row address are requested then they will be placed at the end
    /// of the output schema.  If both are requested then the row id will come before
    /// the row address.
    pub fn with_projection(mut self, projection: Projection) -> Self {
        self.projection = projection;
        self
    }
}

/// A plan node that reads a dataset, applying an optional filter and projection.
///
/// This node may execute a scan or it may execute a take.  By default, it picks the best
/// approach based the expected query cost which is determined by:
///  - Size of data in desired columns
///  - Number of rows matching the index search
///  - Filesystem parameters (e.g. block size)
///
/// This decision is made during execution, after the index search is complete, and not during
/// planning.
///
/// In the future, we may introduce high-level cardinality statistics similar to those used by query
/// engines like Postgres.  This might allow us to know, without executing an index search, that a scan
/// would be better.  In that case we accept the force_scan hint to skip the index search.
#[derive(Debug)]
pub struct FilteredReadExec {
    dataset: Arc<Dataset>,
    options: FilteredReadOptions,
    properties: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
    index_input: Option<Arc<dyn ExecutionPlan>>,
    // When execute is first called we will initialize the PlannedFilterReadExec. In order to support
    // multiple partitions, each partition will share the exec.
    planned_exec: Arc<AsyncMutex<Option<Arc<super::planned_filter_read::PlannedFilterReadExec>>>>,
}

impl FilteredReadExec {
    pub fn try_new(
        dataset: Arc<Dataset>,
        mut options: FilteredReadOptions,
        index_input: Option<Arc<dyn ExecutionPlan>>,
    ) -> Result<Self> {
        if options.with_deleted_rows {
            // Ensure we have the row id column if with_deleted_rows is set
            options.projection = options.projection.with_row_id();
        }

        if options.projection.is_empty() {
            return Err(Error::InvalidInput {
                source:
                    "no columns were selected and with_row_id / with_row_address is false, there is nothing to scan"
                        .into(),
                location: location!(),
            });
        }

        if options.scan_range_after_filter.is_some() {
            // Validate that there's a filter when using scan_range_after_filter
            if options.full_filter.is_none()
                && options.refine_filter.is_none()
                && index_input.is_none()
            {
                return Err(Error::InvalidInput {
                    source: "scan_range_after_filter requires a filter to be applied. Use scan_range_before_filter for unfiltered scans."
                        .into(),
                    location: location!(),
                });
            }

            // TODO: support multi partition
            if matches!(
                options.threading_mode,
                FilteredReadThreadingMode::MultiplePartitions(_)
            ) {
                return Err(Error::NotSupported {
                    source: "scan_range_after_filter not yet supported with multiple partitions"
                        .to_string()
                        .into(),
                    location: location!(),
                });
            }
        }
        let output_schema = Arc::new(options.projection.to_arrow_schema());
        let num_partitions = match options.threading_mode {
            FilteredReadThreadingMode::OnePartitionMultipleThreads(_) => 1,
            FilteredReadThreadingMode::MultiplePartitions(n) => n,
        };

        let properties = PlanProperties::new(
            EquivalenceProperties::new(output_schema),
            Partitioning::RoundRobinBatch(num_partitions),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        let metrics = ExecutionPlanMetricsSet::new();

        Ok(Self {
            dataset,
            options,
            properties,
            planned_exec: Arc::new(AsyncMutex::new(None)),
            metrics,
            index_input,
        })
    }

    fn obtain_stream(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> SendableRecordBatchStream {
        // There are two subtleties here:
        //
        // First, we need to defer execution until first polled (hence the once/flatten)
        //
        // Second, multiple partitions all share the same underlying task stream (see get_stream)
        let planned_exec_lock = self.planned_exec.clone();
        let dataset = self.dataset.clone();
        let options = self.options.clone();
        let index_input = self.index_input.clone();

        let stream = futures::stream::once(async move {
            let mut planned_exec_guard = planned_exec_lock.lock().await;
            if let Some(planned_exec) = &*planned_exec_guard {
                DataFusionResult::<SendableRecordBatchStream>::Ok(
                    planned_exec.execute(partition, context)?,
                )
            } else {
                let mut evaluated_index = None;
                if let Some(index_input) = index_input {
                    let mut index_search = index_input.execute(partition, context.clone())?;
                    let index_search_result =
                        index_search.next().await.ok_or_else(|| Error::Internal {
                            message: "Index search did not yield any results".to_string(),
                            location: location!(),
                        })??;
                    evaluated_index = Some(Arc::new(EvaluatedIndex::try_from_arrow(
                        &index_search_result,
                    )?));
                }

                let fragments = options
                    .fragments
                    .clone()
                    .unwrap_or_else(|| dataset.fragments().clone());

                let (planned_fragments, planned_options) =
                    super::planned_filter_read::FilteredReadUtils::plan_scan(
                        &dataset,
                        fragments.as_ref().clone(),
                        evaluated_index,
                        &options,
                    )
                    .await
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;

                let planned_exec =
                    Arc::new(super::planned_filter_read::PlannedFilterReadExec::new(
                        dataset.clone(),
                        planned_fragments,
                        planned_options,
                    ));
                let first_stream = planned_exec.execute(partition, context)?;
                *planned_exec_guard = Some(planned_exec);
                DataFusionResult::Ok(first_stream)
            }
        })
        .try_flatten();

        Box::pin(RecordBatchStreamAdapter::new(self.schema(), stream))
    }

    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    pub fn options(&self) -> &FilteredReadOptions {
        &self.options
    }

    pub fn index_input(&self) -> Option<&Arc<dyn ExecutionPlan>> {
        self.index_input.as_ref()
    }
}

impl DisplayAs for FilteredReadExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let columns = self
            .options
            .projection
            .to_bare_schema()
            .fields
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "LanceRead: uri={}, projection=[{}], num_fragments={}, range_before={:?}, range_after={:?}, row_id={}, row_addr={}, full_filter={}, refine_filter={}",
                    self.dataset.data_dir(),
                    columns,
                    self.options.fragments.as_ref().map(|f| f.len()).unwrap_or(self.dataset.fragments().len()),
                    self.options.scan_range_before_filter,
                    self.options.scan_range_after_filter,
                    self.options.projection.with_row_id,
                    self.options.projection.with_row_addr,
                    self.options.full_filter.as_ref().map(|i| i.to_string()).unwrap_or("--".to_string()),
                    self.options.refine_filter.as_ref().map(|i| i.to_string()).unwrap_or("--".to_string()),
                )
            }
            DisplayFormatType::TreeRender => {
                write!(f, "LanceRead\nuri={}\nprojection=[{}]\nnum_fragments={}\nrange_before={:?}\nrange_after={:?}\nrow_id={}\nrow_addr={}\nfull_filter={}\nrefine_filter={}",
                self.dataset.data_dir(),
                columns,
                self.options.fragments.as_ref().map(|f| f.len()).unwrap_or(self.dataset.fragments().len()),
                self.options.scan_range_before_filter,
                self.options.scan_range_after_filter,
                self.options.projection.with_row_id,
                self.options.projection.with_row_addr,
                self.options.full_filter.as_ref().map(|i| i.to_string()).unwrap_or("true".to_string()),
                self.options.refine_filter.as_ref().map(|i| i.to_string()).unwrap_or("true".to_string()),
            )
            }
        }
    }
}

impl ExecutionPlan for FilteredReadExec {
    fn name(&self) -> &str {
        "FilteredReadExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        if let Some(index_input) = &self.index_input {
            vec![index_input]
        } else {
            vec![]
        }
    }

    fn metrics(&self) -> Option<MetricsSet> {
        // Try to get metrics from the planned exec if it's been initialized
        // Otherwise return empty metrics
        let planned_exec_lock = self.planned_exec.try_lock();
        if let Ok(guard) = planned_exec_lock {
            if let Some(planned_exec) = &*guard {
                return planned_exec.metrics();
            }
        }
        // If planned exec hasn't been initialized yet, return empty metrics
        Some(self.metrics.clone_inner())
    }

    fn partition_statistics(
        &self,
        partition: Option<usize>,
    ) -> datafusion::error::Result<Statistics> {
        let fragments = self
            .options
            .fragments
            .clone()
            .unwrap_or_else(|| self.dataset.fragments().clone());

        if fragments.iter().any(|f| f.num_rows().is_none()) {
            return Err(DataFusionError::Internal(
                "Fragments are missing row count stats".to_string(),
            ));
        }

        let total_rows: u64 = fragments.iter().map(|f| f.num_rows().unwrap() as u64).sum();

        if self.options.full_filter.is_none() {
            // If there is no filter, we just return the total number of rows (sans any before-filter range)
            // divided by the number of partitions.
            let total_rows =
                if let Some(scan_range_before_filter) = &self.options.scan_range_before_filter {
                    total_rows.min(scan_range_before_filter.end - scan_range_before_filter.start)
                } else {
                    total_rows
                };

            let total_rows = if partition.is_some() {
                match self.options.threading_mode {
                    FilteredReadThreadingMode::MultiplePartitions(num_partitions) => {
                        total_rows / num_partitions as u64
                    }
                    // Pretty sure this shouldn't be encountered in practice
                    FilteredReadThreadingMode::OnePartitionMultipleThreads(_) => total_rows,
                }
            } else {
                total_rows
            };

            Ok(Statistics {
                num_rows: Precision::Exact(total_rows as usize),
                ..datafusion::physical_plan::Statistics::new_unknown(self.schema().as_ref())
            })
        } else {
            // We could evaluate the indexed filter here but this is still during the planning
            // phase so we want to avoid that.
            //
            // Instead, we create a mock input which is the filtered read (without the filter)
            // and then use DF's FilterExec logic to calculate the statistics (which uses column
            // stats and basic filter shape)
            let filter = self.options.full_filter.as_ref().unwrap();

            // Need to add in filter columns even though they aren't part of the projection
            let filter_columns = Planner::column_names_in_expr(filter);
            let read_projection = self
                .options
                .projection
                .clone()
                .union_columns(filter_columns, OnMissing::Error)?;

            let read_schema = Arc::new(read_projection.to_arrow_schema());

            let planner = Arc::new(Planner::new(read_schema.clone()));
            let physical_filter = planner.create_physical_expr(filter)?;

            let mock_input = Arc::new(Self::try_new(
                self.dataset.clone(),
                FilteredReadOptions {
                    scan_range_after_filter: None,
                    refine_filter: None,
                    full_filter: None,
                    projection: read_projection,
                    ..self.options.clone()
                },
                None,
            )?);
            let df_filter_exec = FilterExec::try_new(physical_filter, mock_input)?;
            let mut df_stats = df_filter_exec.partition_statistics(partition)?;

            // If we have an after-filter range, we should apply it to the stats (the before-filter range
            // is applied in the mock input)
            let total_rows = if let Some(scan_range_after_filter) =
                &self.options.scan_range_after_filter
            {
                df_stats.num_rows.min(&Precision::Exact(
                    scan_range_after_filter.end as usize - scan_range_after_filter.start as usize,
                ))
            } else {
                df_stats.num_rows
            };
            df_stats.num_rows = total_rows;

            let schema = self.schema();

            // We might have added some columns to the schema so the filter compiles but we drop this
            // columns during the filtered read and they aren't part of the output.  So we need to make
            // sure and drop them from the column stats as well.
            assert_eq!(read_schema.fields.len(), df_stats.column_statistics.len());
            let mut proj_iter = schema.fields.iter().peekable();
            let mut stats_iter = read_schema.fields.iter();
            df_stats.column_statistics.retain(|_| {
                let stats_field = stats_iter.next().unwrap();
                if let Some(proj_field) = proj_iter.peek() {
                    if proj_field.name() == stats_field.name() {
                        proj_iter.next();
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            });

            Ok(df_stats)
        }
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() > 1 {
            Err(DataFusionError::External(
                Error::Internal {
                    message: "A FilteredReadExec cannot have two children".to_string(),
                    location: location!(),
                }
                .into(),
            ))
        } else {
            let index_input = children.into_iter().next();
            Ok(Arc::new(Self {
                dataset: self.dataset.clone(),
                options: self.options.clone(),
                properties: self.properties.clone(),
                metrics: self.metrics.clone(),
                // Seems unlikely this would already be initialized but clear it
                // out just in case
                planned_exec: Arc::new(AsyncMutex::new(None)),
                index_input,
            }))
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        Ok(self.obtain_stream(partition, context))
    }

    fn fetch(&self) -> Option<usize> {
        if self.options.full_filter.is_none() {
            self.options
                .scan_range_before_filter
                .as_ref()
                .map(|range| (range.end - range.start) as usize)
        } else {
            self.options
                .scan_range_after_filter
                .as_ref()
                .map(|range| (range.end - range.start) as usize)
        }
    }

    fn supports_limit_pushdown(&self) -> bool {
        // This is to push the limit through the node and into an upstream node.
        // The only upstream node is the index search and we can't push the limit
        // to that node.
        false
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        // TODO: Support multiple partitions in the future by coordinating limits across partitions
        if matches!(
            self.options.threading_mode,
            FilteredReadThreadingMode::MultiplePartitions(_)
        ) {
            return None;
        }
        let limit = limit?;

        let mut updated_options = self.options.clone();

        if self.options.full_filter.is_none() && self.options.refine_filter.is_none() {
            if self.options.scan_range_before_filter.is_some() {
                return None;
            }
            updated_options.scan_range_before_filter = Some(0..(limit as u64));
        } else {
            if self.options.scan_range_after_filter.is_some() {
                return None;
            }
            updated_options.scan_range_after_filter = Some(0..(limit as u64));
        }

        match Self::try_new(
            self.dataset.clone(),
            updated_options,
            self.index_input.clone(),
        ) {
            Ok(exec) => Some(Arc::new(exec)),
            Err(e) => {
                log::warn!(
                    "Failed to create FilteredReadExec for {} with fetch limit: {}",
                    self.dataset.uri(),
                    e
                );
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow::{
        compute::concat_batches,
        datatypes::{Float32Type, UInt32Type, UInt64Type},
    };
    use arrow_array::{cast::AsArray, Array, UInt32Array};
    use itertools::Itertools;
    use lance_core::datatypes::OnMissing;
    use lance_core::utils::deletion::DeletionVector;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::{array, gen_batch, BatchCount, Dimension, RowCount};
    use lance_index::{
        optimize::OptimizeOptions,
        scalar::{expression::PlannerIndexExt, ScalarIndexParams},
        DatasetIndexExt, IndexType,
    };

    use crate::{
        dataset::{InsertBuilder, WriteDestination, WriteMode, WriteParams},
        index::DatasetIndexInternalExt,
        io::exec::scalar_index::ScalarIndexExec,
        utils::test::{DatagenExt, FragmentCount, FragmentRowCount},
    };

    use super::*;

    struct TestFixture {
        _tmp_path: TempStrDir,
        dataset: Arc<Dataset>,
    }

    /// The test dataset first creates 200 rows and then 200 more, each
    /// with 100 rows per fragment, for a total of 4 fragments.  The column
    /// fully_indexed is indexed on all 4 fragments.  The column partly_indexed
    /// is only indexed on the first 2 fragments.
    ///
    /// The second fragment is then deleted, leaving a gap in the fragment sequence
    /// The third fragment has a deletion file with 50 rows deleted.
    ///
    /// The fragment ids are 0 (values 0..100), 2 (values 250..300), 3 (values 300..400)
    impl TestFixture {
        async fn new() -> Self {
            let tmp_path = TempStrDir::default();

            let mut dataset = gen_batch()
                .col("fully_indexed", array::step::<UInt32Type>())
                .col("partly_indexed", array::step::<UInt64Type>())
                .col("not_indexed", array::step::<UInt32Type>())
                .col(
                    "recheck_idx",
                    array::cycle_utf8_literals(&["cat", "caterpillar", "dog"]),
                )
                .col("vector", array::rand_vec::<Float32Type>(Dimension::from(4)))
                .into_dataset(
                    tmp_path.as_str(),
                    FragmentCount::from(2),
                    FragmentRowCount::from(100),
                )
                .await
                .unwrap();

            dataset
                .create_index(
                    &["fully_indexed"],
                    IndexType::BTree,
                    None,
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
            dataset
                .create_index(
                    &["partly_indexed"],
                    IndexType::BTree,
                    None,
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
            dataset
                .create_index(
                    &["recheck_idx"],
                    IndexType::NGram,
                    None,
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();

            let new_data = gen_batch()
                .col("fully_indexed", array::step_custom::<UInt32Type>(200, 1))
                .col("partly_indexed", array::step_custom::<UInt64Type>(200, 1))
                .col("not_indexed", array::step_custom::<UInt32Type>(200, 1))
                .into_reader_rows(RowCount::from(100), BatchCount::from(2))
                .try_collect()
                .unwrap();

            let mut dataset =
                InsertBuilder::new(WriteDestination::Dataset(Arc::new(dataset.clone())))
                    .with_params(&WriteParams {
                        mode: WriteMode::Append,
                        max_rows_per_file: 100,
                        ..Default::default()
                    })
                    .execute(new_data)
                    .await
                    .unwrap();

            dataset
                .optimize_indices(&OptimizeOptions::new().index_names(vec![
                    "fully_indexed_idx".to_string(),
                    "recheck_idx_idx".to_string(),
                ]))
                .await
                .unwrap();

            dataset
                .delete("fully_indexed >= 100 AND fully_indexed < 250")
                .await
                .unwrap();

            dataset.load_indices().await.unwrap();

            Self {
                _tmp_path: tmp_path,
                dataset: Arc::new(dataset),
            }
        }

        async fn index_input(
            &self,
            options: &FilteredReadOptions,
        ) -> Option<Arc<dyn ExecutionPlan>> {
            if let Some(filter) = &options.full_filter {
                let planner = Planner::new(Arc::new(self.dataset.schema().into()));
                let index_info = self.dataset.scalar_index_info().await.unwrap();
                let filter_plan = planner
                    .create_filter_plan(filter.clone(), &index_info, true)
                    .unwrap();
                if let Some(index_query) = filter_plan.index_query {
                    Some(Arc::new(ScalarIndexExec::new(
                        self.dataset.clone(),
                        index_query,
                    )))
                } else {
                    None
                }
            } else {
                None
            }
        }

        async fn make_plan(&self, options: FilteredReadOptions) -> FilteredReadExec {
            let index_input = self.index_input(&options).await;
            FilteredReadExec::try_new(self.dataset.clone(), options, index_input).unwrap()
        }

        async fn test_plan(&self, options: FilteredReadOptions, expected: &dyn Array) {
            let index_input = self.index_input(&options).await;
            let plan =
                FilteredReadExec::try_new(self.dataset.clone(), options, index_input).unwrap();

            let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
            let schema = stream.schema();
            let batches = stream.try_collect::<Vec<_>>().await.unwrap();

            let batch = concat_batches(&schema, &batches).unwrap();

            assert_eq!(batch.num_rows(), expected.len());

            let col = batch.column(0);
            assert_eq!(col.as_ref(), expected);
        }

        fn frags(&self, ids: &[u32]) -> Arc<Vec<Fragment>> {
            Arc::new(
                ids.iter()
                    .map(|id| {
                        self.dataset
                            .fragments()
                            .iter()
                            .find(|f| f.id == *id as u64)
                            .unwrap()
                            .clone()
                    })
                    .collect(),
            )
        }

        async fn filter_plan(&self, filter: &str, use_scalar_index: bool) -> FilterPlan {
            let arrow_schema = Arc::new(arrow_schema::Schema::from(self.dataset.schema()));
            let planner = Planner::new(arrow_schema);
            let expr = planner.parse_filter(filter).unwrap();
            let index_info = self.dataset.scalar_index_info().await.unwrap();
            planner
                .create_filter_plan(expr, &index_info, use_scalar_index)
                .unwrap()
        }
    }

    fn u32s(ranges: Vec<Range<u32>>) -> Arc<dyn Array> {
        Arc::new(UInt32Array::from_iter_values(
            ranges.into_iter().flat_map(|r| r.into_iter()),
        ))
    }

    #[test_log::test(tokio::test)]
    async fn test_range_no_scalar_index() {
        let fixture = TestFixture::new().await;

        let base_options = FilteredReadOptions::basic_full_read(&fixture.dataset);
        // Basic full scan
        fixture
            .test_plan(base_options.clone(), &u32s(vec![0..100, 250..400]))
            .await;

        // Basic range scan (whole dataset, no filter)
        let options = base_options
            .clone()
            .with_scan_range_before_filter(25..125)
            .unwrap();
        fixture
            .test_plan(options, &u32s(vec![25..100, 250..275]))
            .await;

        // Range scan against user-specified fragments
        let options = base_options
            .clone()
            .with_fragments(fixture.frags(&[3, 2]))
            .with_scan_range_before_filter(25..125)
            .unwrap();
        fixture
            .test_plan(options, &u32s(vec![325..400, 250..275]))
            .await;

        // Range scan that goes past the end of the dataset (100 rows
        // requested, only 50 can be returned)
        let options = base_options
            .clone()
            .with_scan_range_before_filter(200..300)
            .unwrap();
        fixture.test_plan(options, &u32s(vec![350..400])).await;

        // Range scan that completely misses the dataset
        let options = base_options
            .clone()
            .with_scan_range_before_filter(300..400)
            .unwrap();
        fixture.test_plan(options, &u32s(vec![])).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_batch_size() {
        let fixture = TestFixture::new().await;

        // First, test with the default batch size, which is bigger than any fragment in our
        // test dataset (we have tests for larger batch sizes in python, let's avoid duplicating
        // them here)
        let base_options = FilteredReadOptions::basic_full_read(&fixture.dataset);

        let plan = fixture.make_plan(base_options.clone()).await;

        let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        let batch_sizes = batches.iter().map(|b| b.num_rows()).collect::<Vec<_>>();
        assert_eq!(batch_sizes, vec![100, 50, 100]);

        // Now, test with a batch size that is smaller than any fragment in our
        // test dataset
        let options = base_options.with_batch_size(35);

        let plan = fixture.make_plan(options).await;

        let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        let batch_sizes = batches.iter().map(|b| b.num_rows()).collect::<Vec<_>>();

        // Some batches will be smaller because we don't coalesce batches across fragments
        assert_eq!(batch_sizes, vec![35, 35, 30, 35, 15, 35, 35, 30]);
    }

    #[test_log::test(tokio::test)]
    async fn test_recheck() {
        let fixture = TestFixture::new().await;

        // First, test with the default batch size, which is bigger than any fragment in our
        // test dataset (we have tests for larger batch sizes in python, let's avoid duplicating
        // them here)
        let base_options = FilteredReadOptions::basic_full_read(&fixture.dataset);

        let filter_plan = fixture
            .filter_plan("contains(recheck_idx, 'cat')", true)
            .await;

        let options = base_options.clone().with_filter_plan(filter_plan);
        let plan = fixture.make_plan(options).await;

        let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        let batch_sizes = batches.iter().map(|b| b.num_rows()).collect::<Vec<_>>();
        assert_eq!(batch_sizes, vec![67]);
    }

    #[test_log::test(tokio::test)]
    async fn test_projection() {
        let fixture = Arc::new(TestFixture::new().await);

        // By default we get all columns
        let base_options = FilteredReadOptions::basic_full_read(&fixture.dataset);

        let check_projection =
            |projection: Option<Projection>, expected_columns: Vec<&'static str>| {
                let fixture = fixture.clone();
                let base_options = base_options.clone();
                async move {
                    let mut options = base_options.clone();
                    if let Some(projection) = projection {
                        options = options.with_projection(projection);
                    }
                    let plan = fixture.make_plan(options).await;

                    let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
                    let batches = stream.try_collect::<Vec<_>>().await.unwrap();
                    for batch in batches {
                        assert_eq!(batch.num_columns(), expected_columns.len());
                        for (i, col) in batch.schema().fields().iter().enumerate() {
                            assert_eq!(col.name(), expected_columns[i]);
                        }
                    }
                }
            };

        check_projection(
            None,
            vec![
                "fully_indexed",
                "partly_indexed",
                "not_indexed",
                "recheck_idx",
                "vector",
            ],
        )
        .await;
        let projection = fixture
            .dataset
            .empty_projection()
            .union_column("fully_indexed", OnMissing::Error)
            .unwrap();
        check_projection(Some(projection), vec!["fully_indexed"]).await;
        let row_id_only = fixture.dataset.empty_projection().with_row_id();
        check_projection(Some(row_id_only), vec!["_rowid"]).await;
        let row_addr_only = fixture.dataset.empty_projection().with_row_addr();
        check_projection(Some(row_addr_only), vec!["_rowaddr"]).await;
        let everything = fixture
            .dataset
            .full_projection()
            .with_row_addr()
            .with_row_id();
        check_projection(
            Some(everything),
            vec![
                "fully_indexed",
                "partly_indexed",
                "not_indexed",
                "recheck_idx",
                "vector",
                "_rowid",
                "_rowaddr",
            ],
        )
        .await;

        // It is an error to scan an empty projection
        let options = base_options
            .clone()
            .with_projection(fixture.dataset.empty_projection());
        let index_input = fixture.index_input(&options).await;
        let Err(Error::InvalidInput { source, .. }) =
            FilteredReadExec::try_new(fixture.dataset.clone(), options, index_input)
        else {
            panic!("Expected an InvalidInput error when given an empty projection");
        };
        assert!(source.to_string().contains("no columns were selected"));
    }

    #[test_log::test(tokio::test)]
    async fn test_filter_no_scalar_index() {
        let fixture = Arc::new(TestFixture::new().await);

        let base_options = FilteredReadOptions::basic_full_read(&fixture.dataset);

        // Basic filter
        let filter_plan = fixture.filter_plan("not_indexed >= 75", false).await;
        let options = base_options.clone().with_filter_plan(filter_plan);
        fixture
            .test_plan(options, &u32s(vec![75..100, 250..400]))
            .await;

        // Filter matches no rows
        let filter_plan = fixture.filter_plan("not_indexed >= 1000", false).await;
        let options = base_options.clone().with_filter_plan(filter_plan);
        fixture.test_plan(options, &u32s(vec![])).await;

        // Filter with before_filter scan range
        let filter_plan = fixture.filter_plan("not_indexed >= 75", false).await;
        let options = base_options
            .clone()
            .with_scan_range_before_filter(25..125)
            .unwrap()
            .with_filter_plan(filter_plan);
        fixture
            .test_plan(options, &u32s(vec![75..100, 250..275]))
            .await;

        // Filter removes all rows specified by the scan range
        let filter_plan = fixture.filter_plan("not_indexed >= 75", false).await;
        let options = base_options
            .clone()
            .with_scan_range_before_filter(25..50)
            .unwrap()
            .with_filter_plan(filter_plan);
        fixture.test_plan(options, &u32s(vec![])).await;

        // Can filter on columns with scalar index info, if use_scalar_index is false
        let filter_plan = fixture.filter_plan("fully_indexed >= 200", false).await;
        let options = base_options.clone().with_filter_plan(filter_plan);
        fixture.test_plan(options, &u32s(vec![250..400])).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_filter_scalar_index() {
        let fixture = Arc::new(TestFixture::new().await);

        let base_options = FilteredReadOptions::basic_full_read(&fixture.dataset);

        for index in ["fully_indexed", "partly_indexed"] {
            // Basic filter
            let filter_plan = fixture.filter_plan(&format!("{index} >= 200"), true).await;
            let options = base_options.clone().with_filter_plan(filter_plan);
            fixture.test_plan(options, &u32s(vec![250..400])).await;

            let filter_plan = fixture
                .filter_plan(&format!("{index} >= 230 AND {index} < 270"), true)
                .await;
            let options = base_options.clone().with_filter_plan(filter_plan);
            fixture.test_plan(options, &u32s(vec![250..270])).await;

            // Filter with before filter scan range
            let filter_plan = fixture.filter_plan(&format!("{index} < 270"), true).await;
            let options = base_options
                .clone()
                .with_scan_range_before_filter(25..125)
                .unwrap()
                .with_filter_plan(filter_plan);
            fixture
                .test_plan(options, &u32s(vec![25..100, 250..270]))
                .await;

            // Query asks for a subset of columns that does not include the
            // filter columns.
            let filter_plan = fixture.filter_plan(&format!("{index} >= 200"), true).await;
            let options = base_options
                .clone()
                .with_projection(
                    fixture
                        .dataset
                        .empty_projection()
                        .union_column("not_indexed", OnMissing::Error)
                        .unwrap(),
                )
                .with_filter_plan(filter_plan);
            fixture.test_plan(options, &u32s(vec![250..400])).await;
        }
    }

    #[test_log::test(tokio::test)]
    async fn test_filter_empty_batches() {
        let fixture = Arc::new(TestFixture::new().await);

        let base_options = FilteredReadOptions::basic_full_read(&fixture.dataset);

        let filter_plan = fixture.filter_plan("not_indexed == 317", false).await;
        let options = base_options
            .clone()
            .with_filter_plan(filter_plan)
            .with_batch_size(10);

        let plan = fixture.make_plan(options).await;

        let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
    }

    #[test_log::test(tokio::test)]
    async fn test_with_deleted_rows() {
        let fixture = Arc::new(TestFixture::new().await);

        let base_options = FilteredReadOptions::basic_full_read(&fixture.dataset);

        // Basic full scan
        fixture
            .test_plan(
                base_options.clone().with_deleted_rows().unwrap(),
                &u32s(vec![0..100, 200..400]),
            )
            .await;

        // With only row id
        let options = base_options
            .clone()
            .with_deleted_rows()
            .unwrap()
            .with_projection(fixture.dataset.empty_projection().with_row_id());
        let plan = fixture.make_plan(options).await;
        let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        let num_rows = stream
            .map_ok(|batch| batch.num_rows())
            .try_fold(0, |acc, val| std::future::ready(Ok(acc + val)))
            .await
            .unwrap();
        assert_eq!(num_rows, 300);
    }

    #[test]
    fn test_dv_to_ranges() {
        let dv = Arc::new(DeletionVector::from_iter(vec![1]));
        let ranges = DvToValidRanges::new(dv.iter().map(|i| i as u64), 2).collect::<Vec<_>>();
        assert_eq!(ranges, vec![0..1]);
    }

    #[tokio::test]
    async fn test_statistics() {
        let fixture = Arc::new(TestFixture::new().await);

        let base_options = FilteredReadOptions::basic_full_read(&fixture.dataset);

        let plan = fixture.make_plan(base_options.clone()).await;

        let stats = plan.partition_statistics(None).unwrap();
        // With no filter and no range we have an exact count
        assert_eq!(stats.num_rows, Precision::Exact(250));

        // No filter with range (before or after) is still exact
        let options = base_options
            .clone()
            .with_scan_range_before_filter(25..125)
            .unwrap();
        let plan = fixture.make_plan(options).await;
        let stats = plan.partition_statistics(None).unwrap();
        assert_eq!(stats.num_rows, Precision::Exact(100));

        // With a filter, we don't know the exact count but DF can make some guesses

        // In this case DF recognizes the expression as simple and without column stats it errs on
        // the side of nothing getting filtered out.
        let options = base_options
            .clone()
            .with_filter_plan(fixture.filter_plan("not_indexed >= 200", false).await);
        let plan = fixture.make_plan(options).await;
        let stats = plan.partition_statistics(None).unwrap();
        assert_eq!(stats.num_rows, Precision::Inexact(250));

        // In this case DF doesn't recognize the expression as simple and so it assumes a default
        // selectivity of 0.2
        let options = base_options
            .clone()
            .with_filter_plan(fixture.filter_plan("random() < 0.5", false).await);
        let plan = fixture.make_plan(options).await;
        let stats = plan.partition_statistics(None).unwrap();
        assert_eq!(stats.num_rows, Precision::Inexact(50));

        // Filter columns not part of projection, make sure statistics using correct input schema
        let options = base_options
            .clone()
            .with_filter_plan(fixture.filter_plan("not_indexed >= 200", false).await)
            .with_projection(
                fixture
                    .dataset
                    .empty_projection()
                    // Loading a vector here regresses a bug found during development where the input schema
                    // to the filter exec in statistics was incorrect.
                    .union_column("vector", OnMissing::Error)
                    .unwrap(),
            );
        let plan = fixture.make_plan(options).await;
        let stats = plan.partition_statistics(None).unwrap();
        assert_eq!(stats.num_rows, Precision::Inexact(250));
        assert_eq!(stats.column_statistics.len(), 1);
    }

    #[test_log::test(tokio::test)]
    async fn test_limit_offset_with_deleted_rows() {
        // This test reproduces the issue from the Python test_limit_offset[stable] failure
        // Create a simple dataset with 10 rows (0-9)
        let tmp_path = TempStrDir::default();
        let mut dataset = gen_batch()
            .col("a", array::step::<UInt32Type>())
            .into_dataset(
                tmp_path.as_str(),
                FragmentCount::from(1),
                FragmentRowCount::from(10),
            )
            .await
            .unwrap();

        // Delete rows where a > 2 AND a < 7 (should delete a=3,4,5,6)
        // This leaves: a=0,1,2,7,8,9
        dataset.delete("a > 2 AND a < 7").await.unwrap();
        let dataset = Arc::new(dataset);

        // Test offset=3, limit=1 which should return a=7 (the 4th remaining row)
        let base_options = FilteredReadOptions::basic_full_read(&dataset);
        let options = base_options.with_scan_range_before_filter(3..4).unwrap();

        let plan = FilteredReadExec::try_new(dataset.clone(), options, None).unwrap();
        let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        let schema = stream.schema();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        let batch = concat_batches(&schema, &batches).unwrap();

        // This should return 1 row with a=7
        assert_eq!(
            batch.num_rows(),
            1,
            "Expected 1 row but got {}",
            batch.num_rows()
        );

        if batch.num_rows() > 0 {
            let col = batch.column(0).as_primitive::<UInt32Type>();
            assert_eq!(col.value(0), 7, "Expected a=7 but got a={}", col.value(0));
        }
    }

    #[tokio::test]
    async fn test_with_fetch_limit_pushdown() {
        // Test that with_fetch() properly updates scan ranges for limit pushdown
        let fixture = Arc::new(TestFixture::new().await);
        let base_options = FilteredReadOptions::basic_full_read(&fixture.dataset);

        // Case 1: No filter, no existing scan_range - should set scan_range_before_filter
        {
            let plan = fixture.make_plan(base_options.clone()).await;
            assert_eq!(plan.options().scan_range_before_filter, None);
            assert_eq!(plan.fetch(), None);
            let new_plan = plan.with_fetch(Some(100)).unwrap();
            let new_plan = new_plan
                .as_any()
                .downcast_ref::<FilteredReadExec>()
                .unwrap();
            assert_eq!(new_plan.options().scan_range_before_filter, Some(0..100));
            assert_eq!(new_plan.fetch(), Some(100));
        }

        // Case 2: No filter with existing scan_range_before_filter - should reject (return None)
        {
            let options = base_options
                .clone()
                .with_scan_range_before_filter(50..200)
                .unwrap();
            let plan = fixture.make_plan(options).await;
            assert_eq!(plan.options().scan_range_before_filter, Some(50..200));
            assert_eq!(plan.fetch(), Some(150));

            // Should return None because scan_range_before_filter already exists
            let result = plan.with_fetch(Some(80));
            assert!(result.is_none());
        }

        // Case 3: With filter, no existing scan_range_after_filter - should set scan_range_after_filter
        {
            let filter_plan = fixture.filter_plan("fully_indexed < 200", false).await;
            let options = base_options.clone().with_filter_plan(filter_plan);
            let plan = fixture.make_plan(options).await;
            assert_eq!(plan.options().scan_range_after_filter, None);
            assert_eq!(plan.fetch(), None);
            let new_plan = plan.with_fetch(Some(50)).unwrap();
            let new_plan = new_plan
                .as_any()
                .downcast_ref::<FilteredReadExec>()
                .unwrap();
            assert_eq!(new_plan.options().scan_range_after_filter, Some(0..50));
            assert_eq!(new_plan.fetch(), Some(50));
        }

        // Case 4: With filter and existing scan_range_after_filter - should reject (return None)
        {
            let filter_plan = fixture.filter_plan("fully_indexed < 200", false).await;
            let options = base_options
                .clone()
                .with_filter_plan(filter_plan)
                .with_scan_range_after_filter(100..300)
                .unwrap();
            let plan = fixture.make_plan(options).await;
            assert_eq!(plan.options().scan_range_after_filter, Some(100..300));

            // Should return None because scan_range_after_filter already exists
            let result = plan.with_fetch(Some(50));
            assert!(result.is_none());
        }

        // Case 5: Multiple partitions mode - with_fetch should reject pushdown
        {
            let mut options = base_options.clone();
            options.threading_mode = FilteredReadThreadingMode::MultiplePartitions(4);
            let filter_plan = fixture.filter_plan("fully_indexed < 200", false).await;
            options = options.with_filter_plan(filter_plan);
            let plan = fixture.make_plan(options).await;
            let result = plan.with_fetch(Some(100));
            assert!(result.is_none());
        }

        // Case 6: None limit value - should be rejected
        {
            let plan = fixture.make_plan(base_options.clone()).await;
            let result = plan.with_fetch(None);
            assert!(result.is_none());
        }
    }

    #[tokio::test]
    async fn test_limit_pushdown_comprehensive() {
        let fixture = Arc::new(TestFixture::new().await);
        let base_options = FilteredReadOptions::basic_full_read(&fixture.dataset);

        // Test 1: No index with limit - should pushdown to scan_range_before_filter
        let options = base_options
            .clone()
            .with_scan_range_before_filter(0..100)
            .unwrap();
        let plan = fixture.make_plan(options.clone()).await;
        assert_eq!(plan.options().scan_range_before_filter, Some(0..100));
        assert_eq!(plan.options().scan_range_after_filter, None);
        test_scan_range(&fixture, options, (0..100).collect(), "No index with limit").await;

        // Test 2: Exact match index with limit
        let filter_plan = fixture.filter_plan("fully_indexed < 50", false).await;
        let options = base_options
            .clone()
            .with_filter_plan(filter_plan)
            .with_scan_range_after_filter(0..25)
            .unwrap()
            .with_batch_size(10);
        let plan = fixture.make_plan(options.clone()).await;
        assert_eq!(plan.options().scan_range_after_filter, Some(0..25));
        assert_eq!(plan.options().scan_range_before_filter, None);
        test_scan_range(
            &fixture,
            options,
            (0..25).collect(),
            "Exact match index with limit",
        )
        .await;

        // Test 3: Regression test for batch boundary bug
        let filter_plan = fixture.filter_plan("not_indexed >= 0", false).await;
        let options = base_options
            .with_filter_plan(filter_plan)
            .with_scan_range_after_filter(0..250)
            .unwrap()
            .with_batch_size(50);
        let expected_values: Vec<u32> = (0..100).chain(250..400).take(250).collect();
        test_scan_range(
            &fixture,
            options,
            expected_values,
            "Batch boundary regression",
        )
        .await;
    }

    /// Helper to extract fully_indexed column values from batches
    async fn get_fully_indexed_values(batches: Vec<RecordBatch>) -> Vec<u32> {
        batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("fully_indexed")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<arrow_array::UInt32Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Helper to test scan range with expected values
    async fn test_scan_range(
        fixture: &TestFixture,
        options: FilteredReadOptions,
        expected_values: Vec<u32>,
        test_description: &str,
    ) {
        let plan = fixture.make_plan(options).await;
        let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        let actual_values = get_fully_indexed_values(batches).await;
        assert_eq!(
            actual_values, expected_values,
            "Failed test: {}",
            test_description
        );
    }

    /// Helper to compute expected values for scan range tests
    /// Dataset layout: [0..100] deleted:[100..250] [250..400]
    fn compute_range_values(range: Range<u64>) -> Vec<u32> {
        let mut result = Vec::new();
        for pos in range {
            if pos < 100 {
                result.push(pos as u32);
            } else if pos < 250 {
                // Positions 100-249 map to values 250-399
                result.push((250 + (pos - 100)) as u32);
            }
        }
        result
    }

    #[tokio::test]
    async fn test_no_filter_scan_range_before_filter() {
        let fixture = Arc::new(TestFixture::new().await);
        let base_options = FilteredReadOptions::basic_full_read(&fixture.dataset);

        // Test cases: (scan_range, description)
        let test_cases = vec![
            // Basic cases
            (0..50, "Limit from start"),
            (30..80, "Offset + limit"),
            (0..250, "Limit equals total rows"),
            (0..500, "Limit exceeds total rows"),
            // Edge cases
            (0..1, "Single row"),
            (99..100, "Last row of first fragment"),
            (100..101, "First row of second fragment (deleted area)"),
            (249..250, "Last available row"),
            // Fragment boundaries
            (0..100, "Entire first fragment"),
            (100..200, "Middle of dataset (deleted area)"),
            (50..150, "Across fragment boundary"),
            (90..110, "Around deletion boundary"),
            // Large offsets
            (200..250, "Large offset into second fragment"),
            (240..260, "Near end with overrun"),
            (300..400, "Beyond available data"),
            // Zero-width ranges
            (50..50, "Empty range in data"),
            (150..150, "Empty range in deleted area"),
            (400..400, "Empty range beyond data"),
        ];

        for (range, description) in test_cases {
            let options = base_options
                .clone()
                .with_scan_range_before_filter(range.clone())
                .unwrap();
            let expected = compute_range_values(range);
            test_scan_range(&fixture, options, expected, description).await;
        }
    }

    #[tokio::test]
    async fn test_exact_match_filter_scan_range_after_filter() {
        let fixture = Arc::new(TestFixture::new().await);
        let base_options = FilteredReadOptions::basic_full_read(&fixture.dataset);

        // Test cases: (filter, scan_range, expected_values, description)
        let test_cases = vec![
            // Basic limit tests with diverse ranges
            (
                "fully_indexed < 100",
                0..50,
                (0..50).collect(),
                "Limit < matches",
            ),
            (
                "fully_indexed < 100",
                20..50,
                (20..50).collect(),
                "Offset + limit within matches",
            ),
            (
                "fully_indexed < 100",
                0..100,
                (0..100).collect(),
                "Limit = matches",
            ),
            (
                "fully_indexed < 50",
                0..200,
                (0..50).collect(),
                "Limit > matches",
            ),
            ("fully_indexed < 100", 0..1, vec![0], "Single row"),
            (
                "fully_indexed < 100",
                99..100,
                vec![99],
                "Last matching row",
            ),
            (
                "fully_indexed < 100",
                5..15,
                (5..15).collect(),
                "Small window",
            ),
            (
                "fully_indexed < 100",
                90..110,
                (90..100).collect(),
                "Range beyond matches",
            ),
            (
                "fully_indexed < 100",
                45..55,
                (45..55).collect(),
                "Mid-range window",
            ),
            (
                "fully_indexed < 100",
                0..10000,
                (0..100).collect(),
                "Huge limit",
            ),
            // Range filter tests with more diverse ranges
            (
                "fully_indexed >= 50 AND fully_indexed < 80",
                0..20,
                (50..70).collect(),
                "Range filter with limit",
            ),
            (
                "fully_indexed >= 50 AND fully_indexed < 80",
                10..25,
                (60..75).collect(),
                "Range filter with offset+limit",
            ),
            (
                "fully_indexed >= 50 AND fully_indexed < 80",
                0..30,
                (50..80).collect(),
                "Range filter exact match",
            ),
            (
                "fully_indexed >= 50 AND fully_indexed < 80",
                0..100,
                (50..80).collect(),
                "Range filter limit exceeds",
            ),
            (
                "fully_indexed >= 50 AND fully_indexed < 80",
                0..5,
                (50..55).collect(),
                "First 5 rows",
            ),
            (
                "fully_indexed >= 50 AND fully_indexed < 80",
                25..30,
                (75..80).collect(),
                "Last 5 rows",
            ),
            (
                "fully_indexed >= 50 AND fully_indexed < 80",
                15..16,
                vec![65],
                "Single row middle",
            ),
            (
                "fully_indexed >= 50 AND fully_indexed < 80",
                2..8,
                (52..58).collect(),
                "Small offset window",
            ),
            (
                "fully_indexed >= 50 AND fully_indexed < 80",
                100..200,
                vec![],
                "Offset beyond data",
            ),
            // Boundary tests
            ("fully_indexed = 0", 0..10, vec![0], "Single value at start"),
            (
                "fully_indexed = 99",
                0..10,
                vec![99],
                "Single value at fragment end",
            ),
            (
                "fully_indexed = 250",
                0..10,
                vec![250],
                "Single value at second fragment start",
            ),
            (
                "fully_indexed = 399",
                0..10,
                vec![399],
                "Single value at dataset end",
            ),
            // Empty result tests
            (
                "fully_indexed = 150",
                0..10,
                vec![],
                "No match in deleted range",
            ),
            (
                "fully_indexed > 500",
                0..100,
                vec![],
                "No match beyond data",
            ),
            // Fragment boundary tests with diverse ranges
            (
                "fully_indexed > 200",
                0..100,
                (250..350).collect(),
                "Filter skips deleted fragment",
            ),
            (
                "fully_indexed >= 250",
                0..50,
                (250..300).collect(),
                "Start of second fragment",
            ),
            (
                "fully_indexed >= 350",
                0..100,
                (350..400).collect(),
                "End of second fragment",
            ),
            (
                "fully_indexed < 400",
                200..250,
                (350..400).collect(),
                "Large offset into second fragment",
            ),
            (
                "fully_indexed >= 250",
                0..1,
                vec![250],
                "First row second fragment",
            ),
            (
                "fully_indexed >= 250",
                149..150,
                vec![399],
                "Last row second fragment",
            ),
            (
                "fully_indexed >= 250",
                10..20,
                (260..270).collect(),
                "Small window in second",
            ),
            (
                "fully_indexed >= 250",
                75..100,
                (325..350).collect(),
                "Middle of second fragment",
            ),
            (
                "fully_indexed >= 250",
                100..200,
                (350..400).collect(),
                "End portion of second",
            ),
            (
                "fully_indexed >= 300",
                25..75,
                (325..375).collect(),
                "Mid to late second fragment",
            ),
            // Complex filters with various ranges
            (
                "fully_indexed IN (5, 15, 25, 35, 45)",
                0..10,
                vec![5, 15, 25, 35, 45],
                "IN clause all",
            ),
            (
                "fully_indexed IN (5, 15, 25, 35, 45)",
                0..3,
                vec![5, 15, 25],
                "IN clause first 3",
            ),
            (
                "fully_indexed IN (5, 15, 25, 35, 45)",
                2..4,
                vec![25, 35],
                "IN clause middle 2",
            ),
            (
                "fully_indexed IN (5, 15, 25, 35, 45)",
                1..5,
                vec![15, 25, 35, 45],
                "IN clause skip first",
            ),
            (
                "fully_indexed % 10 = 0",
                0..15,
                vec![
                    0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 250, 260, 270, 280, 290,
                ],
                "Modulo all",
            ),
            (
                "fully_indexed % 10 = 0",
                0..3,
                vec![0, 10, 20],
                "Modulo first 3",
            ),
            (
                "fully_indexed % 10 = 0",
                5..10,
                vec![50, 60, 70, 80, 90],
                "Modulo middle range",
            ),
            (
                "fully_indexed % 10 = 0",
                8..12,
                vec![80, 90, 250, 260],
                "Modulo cross fragment",
            ),
            (
                "fully_indexed % 10 = 0",
                10..15,
                vec![250, 260, 270, 280, 290],
                "Modulo second fragment",
            ),
            (
                "fully_indexed >= 80 AND fully_indexed <= 280",
                0..50,
                vec![
                    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                    250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265,
                    266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279,
                ],
                "Cross-fragment full",
            ),
            (
                "fully_indexed >= 80 AND fully_indexed <= 280",
                0..10,
                (80..90).collect(),
                "Cross-fragment first 10",
            ),
            (
                "fully_indexed >= 80 AND fully_indexed <= 280",
                15..25,
                vec![95, 96, 97, 98, 99, 250, 251, 252, 253, 254],
                "Cross-fragment boundary",
            ),
            (
                "fully_indexed >= 80 AND fully_indexed <= 280",
                20..40,
                (250..270).collect(),
                "Cross-fragment second only",
            ),
            (
                "fully_indexed >= 80 AND fully_indexed <= 280",
                18..22,
                vec![98, 99, 250, 251],
                "Cross-fragment exact boundary",
            ),
            // Edge cases
            ("fully_indexed < 400", 0..0, vec![], "Zero-width range"),
            (
                "fully_indexed >= 0",
                1000..2000,
                vec![],
                "Huge offset beyond data",
            ),
            (
                "fully_indexed BETWEEN 95 AND 255",
                3..8,
                vec![98, 99, 250, 251, 252],
                "BETWEEN crossing deletion",
            ),
        ];

        for (filter_expr, range, expected, description) in test_cases {
            let filter_plan = fixture.filter_plan(filter_expr, false).await;
            let options = base_options
                .clone()
                .with_filter_plan(filter_plan)
                .with_scan_range_after_filter(range)
                .unwrap();
            test_scan_range(&fixture, options, expected, description).await;
        }
    }

    #[tokio::test]
    async fn test_at_least_match_filter_scan_range_after_filter() {
        let fixture = Arc::new(TestFixture::new().await);
        let base_options = FilteredReadOptions::basic_full_read(&fixture.dataset);

        struct TestCase {
            filter: &'static str,
            scan_range: Range<u64>,
            validate: Box<dyn Fn(Vec<u32>)>,
        }

        let test_cases = vec![
            TestCase {
                filter: "recheck_idx = 'cat'",
                scan_range: 0..30,
                validate: Box::new(|values| {
                    assert!(values.len() <= 30, "Should have at most 30 rows");
                    for val in &values {
                        assert_eq!(*val % 3, 0, "Values should be multiples of 3");
                    }
                }),
            },
            TestCase {
                filter: "recheck_idx = 'cat'",
                scan_range: 10..40,
                validate: Box::new(|values| {
                    assert!(values.len() <= 30, "Should have at most 30 rows");
                    assert!(
                        values[0] > 0,
                        "Should have skipped initial matches due to offset"
                    );
                    for val in &values {
                        assert_eq!(*val % 3, 0, "Values should be multiples of 3");
                    }
                }),
            },
            TestCase {
                filter: "recheck_idx = 'cat' AND fully_indexed < 100",
                scan_range: 0..20,
                validate: Box::new(|values| {
                    assert!(values.len() <= 20, "Should have at most 20 rows");
                    for val in &values {
                        assert!(*val < 100, "Values should be < 100");
                        assert_eq!(*val % 3, 0, "Values should be multiples of 3");
                    }
                }),
            },
        ];

        for test_case in test_cases {
            let filter_plan = fixture.filter_plan(test_case.filter, false).await;
            let options = base_options
                .clone()
                .with_filter_plan(filter_plan)
                .with_scan_range_after_filter(test_case.scan_range)
                .unwrap();

            let plan = fixture.make_plan(options).await;
            let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
            let batches = stream.try_collect::<Vec<_>>().await.unwrap();
            let values = get_fully_indexed_values(batches).await;
            (test_case.validate)(values);
        }
    }

    #[tokio::test]
    async fn test_edge_cases_limit_pushdown() {
        let fixture = Arc::new(TestFixture::new().await);
        let base_options = FilteredReadOptions::basic_full_read(&fixture.dataset);

        // Test 5.1: Batch boundary test (regression for original bug)
        let filter_plan = fixture.filter_plan("not_indexed >= 0", false).await;
        let options = base_options
            .clone()
            .with_filter_plan(filter_plan)
            .with_scan_range_after_filter(0..250)
            .unwrap()
            .with_batch_size(24);
        let plan = fixture.make_plan(options).await;
        let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 250);

        // Test 5.2: Empty result set
        let filter_plan = fixture.filter_plan("fully_indexed < 0", false).await;
        let options = base_options
            .clone()
            .with_filter_plan(filter_plan)
            .with_scan_range_after_filter(0..100)
            .unwrap();
        let plan = fixture.make_plan(options).await;
        let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        let num_rows = stream
            .map_ok(|batch| batch.num_rows())
            .try_fold(0, |acc, val| std::future::ready(Ok(acc + val)))
            .await
            .unwrap();
        assert_eq!(num_rows, 0);

        // Test 5.3: Offset + Limit combination
        let options = base_options
            .clone()
            .with_scan_range_before_filter(100..150)
            .unwrap();
        let plan = fixture.make_plan(options).await;
        let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 50);

        // Due to fragment deletion, rows 100-199 don't exist
        // Row offset 100 starts at fragment 2 which has values 250+
        let all_values: Vec<u32> = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("fully_indexed")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<arrow_array::UInt32Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect();
        let expected: Vec<u32> = (250..300).collect();
        assert_eq!(all_values, expected);
    }

    #[tokio::test]
    async fn test_metrics_with_limit_partial_fragment() {
        let fixture = TestFixture::new().await;
        let options = FilteredReadOptions::basic_full_read(&fixture.dataset).with_batch_size(10);
        let filtered_read =
            Arc::new(FilteredReadExec::try_new(fixture.dataset.clone(), options, None).unwrap());

        let batches = filtered_read
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .take(3)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 30);

        // Check metrics reflect partial fragment read
        let metrics = filtered_read.metrics().unwrap();

        // Should show approximately 30 rows scanned (might be slightly more due to buffering)
        // But should be significantly less than full fragment (100 rows)
        let rows_scanned = metrics
            .sum_by_name("rows_scanned")
            .map(|v| v.as_usize())
            .unwrap_or(0);
        assert!(
            (30..100).contains(&rows_scanned),
            "rows_scanned ({}) should be close to limit (30), not full fragment (100)",
            rows_scanned
        );

        // Should show 1 fragment was accessed
        let fragments_scanned = metrics
            .sum_by_name("fragments_scanned")
            .map(|v| v.as_usize())
            .unwrap_or(0);
        assert_eq!(fragments_scanned, 1);

        let ranges_scanned = metrics
            .sum_by_name("ranges_scanned")
            .map(|v| v.as_usize())
            .unwrap_or(0);
        assert!(ranges_scanned > 0, "Should have scanned some ranges");

        // Should have some IO metrics
        let iops = metrics
            .sum_by_name("iops")
            .map(|v| v.as_usize())
            .unwrap_or(0);
        assert!(iops > 0, "Should have recorded IO operations");
    }
}
