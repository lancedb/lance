// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::{ops::Range, sync::Arc};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::common::stats::Precision;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    execution_plan::EmissionType, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
};
use datafusion_expr::Expr;
use datafusion_physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion_physical_plan::metrics::{BaselineMetrics, Count, MetricsSet, Time};
use datafusion_physical_plan::Statistics;
use futures::stream::BoxStream;
use futures::{future, FutureExt, Stream, StreamExt, TryFutureExt, TryStreamExt};
use lance_arrow::RecordBatchExt;
use lance_core::datatypes::OnMissing;
use lance_core::utils::futures::FinallyStreamExt;
use lance_core::{datatypes::Projection, Error, Result};
use lance_datafusion::planner::Planner;
use lance_datafusion::utils::{
    ExecutionPlanMetricsSetExt, FRAGMENTS_SCANNED_METRIC, RANGES_SCANNED_METRIC,
    ROWS_SCANNED_METRIC, TASK_WAIT_TIME_METRIC,
};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_table::utils::stream::ReadBatchFut;
use snafu::location;
use tokio::sync::Mutex as AsyncMutex;
use tracing::{instrument, Instrument};

use crate::dataset::fragment::{FileFragment, FragReadConfig};
use crate::dataset::rowids::load_row_id_sequence;
use crate::Dataset;
use lance_table::format::Fragment;

use super::filtered_read::{EvaluatedIndex, FilteredReadOptions};
use super::utils::IoMetrics;

// ============================================================================
// Planning types and logic
// ============================================================================

/// Options for PlannedFilterReadExec after planning is complete
///
/// This is a simplified version of FilteredReadOptions that only contains
/// the fields needed for execution, not planning.
#[derive(Debug, Clone)]
pub struct PlannedFilterReadOptions {
    /// The projection to use for the scan
    pub projection: Projection,
    /// The refine filter to apply (for exact index matches)
    pub refine_filter: Option<Expr>,
    /// The full filter to apply
    pub full_filter: Option<Expr>,
    /// The threading mode to use for the scan
    pub threading_mode: super::filtered_read::FilteredReadThreadingMode,
    /// Include deleted rows in the scan
    pub with_deleted_rows: bool,
    /// The maximum number of rows per batch (required, no Option)
    pub batch_size: u32,
    /// Controls how many fragments to read ahead (required, no Option)
    pub fragment_readahead: usize,
    /// Range to apply after filtering (if limit not pushed down during planning)
    pub scan_range_after_filter: Option<Range<u64>>,
}

use lance_core::utils::deletion::DeletionVector;
use lance_index::scalar::expression::IndexExprResult;
use lance_table::rowids::RowIdSequence;
use std::collections::HashMap;

/// A fragment with all of its metadata loaded
pub(crate) struct LoadedFragment {
    pub(crate) row_id_sequence: Arc<RowIdSequence>,
    pub(crate) deletion_vector: Option<Arc<DeletionVector>>,
    pub(crate) fragment: FileFragment,
    // The number of physical rows in the fragment
    //
    // This count includes deleted rows
    pub(crate) num_physical_rows: u64,
    // The number of logical rows in the fragment
    //
    // This count does not include deleted rows
    pub(crate) num_logical_rows: u64,
}

/// A planned fragment read with all information needed for execution
#[derive(Debug, Clone)]
pub struct PlannedFragmentRead {
    pub fragment: FileFragment,
    pub ranges: Vec<Range<u64>>,
    /// Whether to use refine filter for this fragment (true for exact index matches)
    pub use_refine: bool,
    pub priority: u32,
}

/// Public API for planning fragment scans
pub struct FilteredReadUtils;

impl FilteredReadUtils {
    /// Public API for planning fragment scans
    ///
    /// Returns: (planned fragment reads, updated planned options ready for execution)
    #[instrument(name = "plan_scan", skip_all)]
    pub async fn plan_scan(
        dataset: &Dataset,
        fragments: Vec<Fragment>,
        evaluated_index: Option<Arc<EvaluatedIndex>>,
        options: &FilteredReadOptions,
    ) -> lance_core::Result<(Vec<PlannedFragmentRead>, PlannedFilterReadOptions)> {
        let io_parallelism = dataset.object_store().io_parallelism();
        let dataset_arc = Arc::new(dataset.clone());
        let frag_futs = fragments
            .iter()
            .map(|frag| {
                Result::Ok(Self::load_fragment(
                    dataset_arc.clone(),
                    frag.clone(),
                    options.with_deleted_rows,
                ))
            })
            .collect::<Vec<_>>();
        let loaded_fragments = futures::stream::iter(frag_futs)
            // Cannot use unordered because we need to populate logical_offset based on user-provided order
            .try_buffered(io_parallelism)
            .try_collect::<Vec<_>>()
            .await?;

        // For pushing down scan_range_after_filter (or limit in v2)
        let mut scan_planned_with_limit_pushed_down = false;
        let mut to_skip = options
            .scan_range_after_filter
            .as_ref()
            .map(|r| r.start)
            .unwrap_or(0);
        let mut to_take = options
            .scan_range_after_filter
            .as_ref()
            .map(|r| r.end - r.start)
            .unwrap_or(u64::MAX);

        // Full fragment ranges to read before applying scan_range_after_filter
        let mut fragments_to_read: HashMap<u32, Vec<Range<u64>>> = HashMap::new();
        // Fragment ranges to read after applying scan_range_after_filter
        let mut scan_push_down_fragments_to_read: HashMap<u32, Vec<Range<u64>>> = HashMap::new();

        // The current offset, includes filtered rows, but not deleted rows
        let mut range_offset = 0;
        for LoadedFragment {
            row_id_sequence,
            fragment,
            num_logical_rows,
            num_physical_rows,
            deletion_vector,
        } in loaded_fragments.iter()
        {
            if let Some(range_before_filter) = &options.scan_range_before_filter {
                if range_offset >= range_before_filter.end {
                    break;
                }
            }

            let mut to_read: Vec<Range<u64>> =
                Self::full_frag_range(*num_physical_rows, deletion_vector);

            if let Some(range_before_filter) = &options.scan_range_before_filter {
                let range_start = range_offset;
                let range_end = if options.with_deleted_rows {
                    range_offset += num_physical_rows;
                    range_start + num_physical_rows
                } else {
                    range_offset += num_logical_rows;
                    range_start + num_logical_rows
                };
                to_read = Self::trim_ranges(to_read, range_start..range_end, range_before_filter);
                if to_read.is_empty() {
                    continue;
                }
            }

            // Apply index and apply scan range after filter if applicable
            Self::apply_index_to_fragment(
                &evaluated_index,
                fragment,
                row_id_sequence,
                to_read,
                &mut to_skip,
                &mut to_take,
                &mut fragments_to_read,
                &mut scan_push_down_fragments_to_read,
            );

            if to_take == 0 {
                scan_planned_with_limit_pushed_down = true;
                fragments_to_read = scan_push_down_fragments_to_read;
                break;
            }
        }

        let mut planned_fragments = Vec::with_capacity(loaded_fragments.len());
        for (priority, fragment) in loaded_fragments.into_iter().enumerate() {
            let fragment_id = fragment.fragment.id() as u32;
            if let Some(to_read) = fragments_to_read.get(&fragment_id) {
                if !to_read.is_empty() {
                    let use_refine = Self::determine_fragment_filter(
                        fragment_id,
                        &evaluated_index,
                        scan_planned_with_limit_pushed_down,
                    );

                    log::trace!(
                        "Planning {} ranges ({} rows) from fragment {} with use_refine: {}",
                        to_read.len(),
                        to_read.iter().map(|r| r.end - r.start).sum::<u64>(),
                        fragment.fragment.id(),
                        use_refine
                    );

                    planned_fragments.push(PlannedFragmentRead {
                        fragment: fragment.fragment.clone(),
                        ranges: to_read.clone(),
                        use_refine,
                        priority: priority as u32,
                    });
                }
            }
        }

        // Create PlannedFilterReadOptions with defaults applied
        let io_parallelism = dataset.object_store().io_parallelism();
        let planned_options = PlannedFilterReadOptions {
            projection: options.projection.clone(),
            refine_filter: options.refine_filter.clone(),
            full_filter: options.full_filter.clone(),
            threading_mode: options.threading_mode,
            with_deleted_rows: options.with_deleted_rows,
            batch_size: options.batch_size.unwrap_or(8192),
            fragment_readahead: options
                .fragment_readahead
                .unwrap_or(io_parallelism * 2)
                .max(1),
            scan_range_after_filter: if scan_planned_with_limit_pushed_down {
                None // Limit was pushed down into ranges
            } else {
                options.scan_range_after_filter.clone()
            },
        };

        Ok((planned_fragments, planned_options))
    }

    pub fn apply_hard_range<S>(
        stream: S,
        range: Range<u64>,
    ) -> impl Stream<Item = Result<RecordBatch>>
    where
        S: Stream<Item = Result<RecordBatch>>,
    {
        let start = range.start as usize;
        let end = range.end as usize;
        let rows_seen = Arc::new(AtomicUsize::new(0));

        stream.try_filter_map(move |batch| {
            if batch.num_rows() == 0 {
                return future::ready(Ok(None));
            }

            let batch_rows = batch.num_rows();
            let current_position = rows_seen.fetch_add(batch_rows, Ordering::Relaxed);
            let batch_end = current_position + batch_rows;

            if batch_end <= start || current_position >= end {
                return future::ready(Ok(None));
            }

            let skip = start.saturating_sub(current_position);
            let end_pos = (end - current_position).min(batch_rows);
            let take = end_pos.saturating_sub(skip);

            if take == 0 {
                return future::ready(Ok(None));
            }

            let result = if skip == 0 && take == batch_rows {
                batch
            } else {
                batch.slice(skip, take)
            };
            future::ready(Ok(Some(result)))
        })
    }

    async fn load_fragment(
        dataset: Arc<Dataset>,
        frag: Fragment,
        include_deleted_rows: bool,
    ) -> Result<LoadedFragment> {
        let file_fragment = FileFragment::new(dataset.clone(), frag.clone());
        let deletion_vector = if include_deleted_rows {
            None
        } else {
            file_fragment.get_deletion_vector().await?
        };

        let num_physical_rows = file_fragment.physical_rows().await? as u64;
        let (row_id_sequence, num_logical_rows) = if dataset.manifest.uses_stable_row_ids() {
            let row_id_sequence = load_row_id_sequence(dataset.as_ref(), &frag).await?;
            let num_logical_rows = row_id_sequence.len();
            (row_id_sequence, num_logical_rows)
        } else {
            let row_ids_start = frag.id << 32;
            let row_ids_end = row_ids_start + num_physical_rows;
            let num_logical_rows = file_fragment.count_rows(None).await? as u64;
            let addrs_as_ids = Arc::new(RowIdSequence::from(row_ids_start..row_ids_end));
            (addrs_as_ids, num_logical_rows)
        };
        Ok(LoadedFragment {
            row_id_sequence,
            fragment: file_fragment,
            num_physical_rows,
            num_logical_rows,
            deletion_vector,
        })
    }

    /// Determine whether to use refine filter for a fragment based on index results
    ///
    /// Returns: true if refine filter should be used, false for full filter
    fn determine_fragment_filter(
        fragment_id: u32,
        evaluated_index: &Option<Arc<EvaluatedIndex>>,
        limit_pushed: bool,
    ) -> bool {
        if let Some(index) = evaluated_index {
            if index.applicable_fragments.contains(fragment_id) {
                match &index.index_result {
                    IndexExprResult::Exact(_) => true,
                    IndexExprResult::AtLeast(_) if limit_pushed => true,
                    _ => false,
                }
            } else {
                false
            }
        } else {
            false
        }
    }

    #[instrument(level = "debug", skip_all)]
    fn full_frag_range(
        num_physical_rows: u64,
        deletion_vector: &Option<Arc<DeletionVector>>,
    ) -> Vec<Range<u64>> {
        if let Some(deletion_vector) = deletion_vector {
            DvToValidRanges::new(
                deletion_vector.to_sorted_iter().map(|pos| pos as u64),
                num_physical_rows,
            )
            .collect()
        } else {
            #[allow(clippy::single_range_in_vec_init)]
            {
                vec![0..num_physical_rows]
            }
        }
    }

    #[instrument(level = "debug", skip_all)]
    fn trim_ranges(
        physical_ranges: Vec<Range<u64>>,
        logical_position: Range<u64>,
        bounds: &Range<u64>,
    ) -> Vec<Range<u64>> {
        let num_logical_rows = logical_position.end - logical_position.start;
        let (mut to_skip, mut to_take) = Self::calculate_fetch(logical_position, bounds);

        if to_skip == 0 && to_take == num_logical_rows {
            return physical_ranges;
        }

        let mut trimmed = Vec::with_capacity(physical_ranges.len());
        for range in physical_ranges {
            let range_len = range.end - range.start;
            if to_skip >= range_len {
                to_skip -= range_len;
                continue;
            }
            let avail_here = range_len - to_skip;
            let to_take_here = avail_here.min(to_take);
            to_take -= to_take_here;
            if to_take_here > 0 {
                trimmed.push(range.start + to_skip..range.start + to_skip + to_take_here);
            }
            to_skip = 0;
            if to_take == 0 {
                break;
            }
        }

        trimmed
    }

    // Given a logical position and bounds, calculate the number of rows to skip and take
    fn calculate_fetch(
        position: Range<u64>, // position of the fragment in dataset/fragment coordinates
        bounds: &Range<u64>,  // bounds of the scan in dataset/fragment coordinates
    ) -> (u64, u64) {
        // Position:         | --- |
        // Bounds  : | --- |
        // Result  : to_skip = 0, to_take = 0
        //
        // Position: | --- |
        // Bounds  :         | --- |
        // Result  : to_skip = 0, to_take = 0
        //
        // Position: | --- |
        // Bounds  :   | --- |
        // Result  : to_skip > 0, to_take = (position.end - bounds.start)
        //
        // Position:   | --- |
        // Bounds  : | -------- |
        // Result  : to_skip = 0, to_take = (position.end - position.start)
        //
        // Position:   | --- |
        // Bounds  : | --- |
        // Result  : to_skip = 0, to_take = (bounds.end - position.start)
        let to_skip = bounds.start.saturating_sub(position.start);
        let to_take = bounds
            .end
            .min(position.end)
            .saturating_sub(position.start.max(bounds.start));

        // Note: to_skip may be > 0 even if to_take == 0
        (to_skip, to_take)
    }

    /// Apply index to a fragment and apply skip/take to matched ranges if possible
    #[allow(clippy::too_many_arguments)]
    fn apply_index_to_fragment(
        evaluated_index: &Option<Arc<EvaluatedIndex>>,
        fragment: &FileFragment,
        row_id_sequence: &Arc<RowIdSequence>,
        to_read: Vec<Range<u64>>,
        to_skip: &mut u64,
        to_take: &mut u64,
        fragments_to_read: &mut HashMap<u32, Vec<Range<u64>>>,
        scan_push_down_fragments_to_read: &mut HashMap<u32, Vec<Range<u64>>>,
    ) {
        let fragment_id = fragment.id() as u32;

        if let Some(evaluated_index) = evaluated_index {
            if evaluated_index.applicable_fragments.contains(fragment_id) {
                let _span = tracing::span!(tracing::Level::DEBUG, "apply_index_result").entered();

                match &evaluated_index.index_result {
                    IndexExprResult::Exact(row_id_mask) => {
                        let valid_ranges = row_id_sequence.mask_to_offset_ranges(row_id_mask);
                        let mut matched_ranges = Self::intersect_ranges(&to_read, &valid_ranges);
                        fragments_to_read.insert(fragment_id, matched_ranges.clone());

                        Self::apply_skip_take_to_ranges(&mut matched_ranges, to_skip, to_take);
                        scan_push_down_fragments_to_read.insert(fragment_id, matched_ranges);
                    }
                    IndexExprResult::AtMost(row_id_mask) => {
                        // Cannot push down skip/take for AtMost
                        let valid_ranges = row_id_sequence.mask_to_offset_ranges(row_id_mask);
                        let matched_ranges = Self::intersect_ranges(&to_read, &valid_ranges);
                        fragments_to_read.insert(fragment_id, matched_ranges);
                    }
                    IndexExprResult::AtLeast(row_id_mask) => {
                        let valid_ranges = row_id_sequence.mask_to_offset_ranges(row_id_mask);
                        let mut guaranteed_ranges = Self::intersect_ranges(&to_read, &valid_ranges);
                        fragments_to_read.insert(fragment_id, guaranteed_ranges.clone());

                        Self::apply_skip_take_to_ranges(&mut guaranteed_ranges, to_skip, to_take);
                        scan_push_down_fragments_to_read.insert(fragment_id, guaranteed_ranges);
                    }
                }
            } else {
                // Fragment not indexed - add full fragment to unindexed_ranges
                fragments_to_read.insert(fragment_id, to_read);
            }
        } else {
            // No index at all - add full fragment to unindexed_ranges
            fragments_to_read.insert(fragment_id, to_read);
        }
    }

    /// Apply skip and take to ranges and update the counters
    fn apply_skip_take_to_ranges(
        to_read: &mut Vec<Range<u64>>,
        to_skip: &mut u64,
        to_take: &mut u64,
    ) {
        if *to_take == 0 {
            to_read.clear();
            *to_skip = 0;
            return;
        }
        let original_rows: u64 = to_read.iter().map(|r| r.end - r.start).sum();
        if *to_skip >= original_rows {
            to_read.clear();
            *to_skip -= original_rows;
            return;
        }
        *to_read = Self::trim_ranges_by_offset(to_read, *to_skip, *to_take);
        let rows_taken: u64 = to_read.iter().map(|r| r.end - r.start).sum();
        *to_skip = 0;
        *to_take = to_take.saturating_sub(rows_taken);
    }

    /// Trim ranges by offset (skip) and limit (take)
    fn trim_ranges_by_offset(ranges: &[Range<u64>], skip: u64, take: u64) -> Vec<Range<u64>> {
        let mut result = Vec::new();
        let mut skipped = 0u64;
        let mut taken = 0u64;

        for range in ranges {
            let range_len = range.end - range.start;

            // Still skipping rows
            if skipped < skip {
                let to_skip_in_range = (skip - skipped).min(range_len);
                skipped += to_skip_in_range;

                // If we've skipped past this entire range, continue
                if to_skip_in_range >= range_len {
                    continue;
                }

                // Partial skip within this range
                let new_start = range.start + to_skip_in_range;
                let remaining = range.end - new_start;
                let to_take_here = remaining.min(take - taken);

                if to_take_here > 0 {
                    result.push(new_start..(new_start + to_take_here));
                    taken += to_take_here;
                }
            } else {
                // No more skipping, just take rows
                let to_take_here = range_len.min(take - taken);
                if to_take_here > 0 {
                    result.push(range.start..(range.start + to_take_here));
                    taken += to_take_here;
                }
            }

            if taken >= take {
                break;
            }
        }

        result
    }

    /// Intersect two sets of sorted ranges
    fn intersect_ranges(ranges1: &[Range<u64>], ranges2: &[Range<u64>]) -> Vec<Range<u64>> {
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < ranges1.len() && j < ranges2.len() {
            let r1 = &ranges1[i];
            let r2 = &ranges2[j];

            // Check for intersection
            let start = r1.start.max(r2.start);
            let end = r1.end.min(r2.end);

            if start < end {
                result.push(start..end);
            }

            // Advance the range that ends first
            if r1.end <= r2.end {
                i += 1;
            } else {
                j += 1;
            }
        }

        result
    }
}

// ============================================================================
// Execution types and logic
// ============================================================================

/// Core executor that takes pre-planned fragments and uses FilteredReadStream
pub struct PlannedFilterReadExec {
    dataset: Arc<Dataset>,
    planned_fragments: Vec<PlannedFragmentRead>,
    options: PlannedFilterReadOptions,
    metrics: ExecutionPlanMetricsSet,
    properties: PlanProperties,
    // When execute is first called we will initialize the FilteredReadStream.  In order to support
    // multiple partitions, each partition will share the stream.
    running_stream: Arc<AsyncMutex<Option<FilteredReadStream>>>,
    // Pre-calculated exact statistics based on planned_fragments
    planned_statistics: Statistics,
}

impl PlannedFilterReadExec {
    pub fn new(
        dataset: Arc<Dataset>,
        planned_fragments: Vec<PlannedFragmentRead>,
        options: PlannedFilterReadOptions,
    ) -> Self {
        let output_schema = Arc::new(options.projection.to_arrow_schema());

        // Calculate exact row count from planned ranges
        let exact_row_count: usize = planned_fragments
            .iter()
            .map(|pf| {
                pf.ranges
                    .iter()
                    .map(|r| (r.end - r.start) as usize)
                    .sum::<usize>()
            })
            .sum();

        // Apply scan_range_after_filter if not already pushed down during planning
        let exact_row_count = if let Some(range) = &options.scan_range_after_filter {
            exact_row_count.min((range.end - range.start) as usize)
        } else {
            exact_row_count
        };

        // Store pre-calculated exact statistics
        let planned_statistics = Statistics {
            num_rows: Precision::Exact(exact_row_count),
            ..Statistics::new_unknown(output_schema.as_ref())
        };

        let properties = PlanProperties::new(
            EquivalenceProperties::new(output_schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        );

        let metrics = ExecutionPlanMetricsSet::new();

        Self {
            dataset,
            planned_fragments,
            options,
            metrics,
            properties,
            running_stream: Arc::new(AsyncMutex::new(None)),
            planned_statistics,
        }
    }
}

impl std::fmt::Debug for PlannedFilterReadExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlannedFilterReadExec")
            .field("num_fragments", &self.planned_fragments.len())
            .field(
                "scan_range_after_filter",
                &self.options.scan_range_after_filter,
            )
            .finish()
    }
}

impl DisplayAs for PlannedFilterReadExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "PlannedFilterReadExec: {} fragments",
                    self.planned_fragments.len()
                )?;
                if let Some(ref range) = self.options.scan_range_after_filter {
                    write!(f, ", scan_range={:?}", range)?;
                }
                Ok(())
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "PlannedFilterReadExec\nfragments={}\nscan_range={:?}",
                    self.planned_fragments.len(),
                    self.options.scan_range_after_filter
                )
            }
        }
    }
}

impl ExecutionPlan for PlannedFilterReadExec {
    fn name(&self) -> &str {
        "PlannedFilterReadExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::new(self.options.projection.to_arrow_schema())
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn supports_limit_pushdown(&self) -> bool {
        // This is to push the limit through the node and into an upstream node.
        // The only upstream node is the index search and we can't push the limit
        // to that node.
        false
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
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Execution(format!(
                "PlannedFilterReadExec only supports a single partition, got {}",
                partition
            )));
        }
        let running_stream_lock = self.running_stream.clone();
        let dataset = self.dataset.clone();
        let planned_fragments = self.planned_fragments.clone();
        let options = self.options.clone();
        let metrics = self.metrics.clone();

        let stream = futures::stream::once(async move {
            let mut running_stream_lock_guard = running_stream_lock.lock().await;
            if let Some(running_stream) = &*running_stream_lock_guard {
                DataFusionResult::<SendableRecordBatchStream>::Ok(
                    running_stream.get_stream(&metrics, partition),
                )
            } else {
                let new_running_stream =
                    FilteredReadStream::from_planned(dataset, planned_fragments, options, &metrics);

                let first_stream = new_running_stream.get_stream(&metrics, partition);
                *running_stream_lock_guard = Some(new_running_stream);
                DataFusionResult::Ok(first_stream)
            }
        })
        .try_flatten();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }

    fn partition_statistics(
        &self,
        partition: Option<usize>,
    ) -> datafusion::error::Result<Statistics> {
        let mut stats = self.planned_statistics.clone();

        // Adjust for partition if needed
        if partition.is_some() {
            match self.options.threading_mode {
                FilteredReadThreadingMode::MultiplePartitions(num_partitions) => {
                    if let Precision::Exact(rows) = stats.num_rows {
                        stats.num_rows = Precision::Exact(rows / num_partitions);
                    }
                }
                FilteredReadThreadingMode::OnePartitionMultipleThreads(_) => {
                    // Single partition - no adjustment needed
                }
            }
        }

        Ok(stats)
    }
}

/// A fragment along with ranges of row offsets to read
struct ScopedFragmentRead {
    fragment: FileFragment,
    ranges: Vec<Range<u64>>,
    projection: Arc<Projection>,
    with_deleted_rows: bool,
    batch_size: u32,
    // An in-memory filter to apply after reading the fragment (whatever couldn't be
    // pushed down into the index query)
    filter: Option<Expr>,
    priority: u32,
    scan_scheduler: Arc<ScanScheduler>,
}

impl ScopedFragmentRead {
    fn frag_read_config(&self) -> FragReadConfig {
        FragReadConfig::default()
            .with_row_id(self.with_deleted_rows || self.projection.with_row_id)
            .with_row_address(self.projection.with_row_addr)
            .with_scan_scheduler(self.scan_scheduler.clone())
            .with_reader_priority(self.priority)
    }
}

/// Given a sorted iterator of deleted row offsets, return a sorted iterator of valid row ranges
///
/// For example, given a fragment with 100 rows, and a deletion vector of 10, 15, 16 this would
/// return 0..10, 11..15, 17..100
pub struct DvToValidRanges<I: Iterator<Item = u64> + Send> {
    deleted_rows: I,
    num_rows: u64,
    position: u64,
}

impl<I: Iterator<Item = u64> + Send> DvToValidRanges<I> {
    pub fn new(deleted_rows: I, num_rows: u64) -> Self {
        Self {
            deleted_rows,
            num_rows,
            position: 0,
        }
    }
}

impl<I: Iterator<Item = u64> + Send> Iterator for DvToValidRanges<I> {
    type Item = Range<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.num_rows {
            return None;
        }
        for next_deleted_row in self.deleted_rows.by_ref() {
            if next_deleted_row == self.position {
                self.position += 1;
            } else {
                let position = self.position;
                self.position = next_deleted_row + 1;
                return Some(position..next_deleted_row);
            }
        }
        let position = self.position;
        self.position = self.num_rows;
        if position == self.num_rows {
            // Last deleted row was end of the fragment, return None
            None
        } else {
            // Still some rows after the last deleted row, return them
            Some(position..self.num_rows)
        }
    }
}

/// Global metrics for the FilteredReadExec node
///
/// These represent work that is not divisible by partition and this work is always
/// reported on partition 0
pub struct FilteredReadGlobalMetrics {
    fragments_scanned: Count,
    ranges_scanned: Count,
    rows_scanned: Count,
    io_metrics: IoMetrics,
}

impl FilteredReadGlobalMetrics {
    pub fn new(metrics: &ExecutionPlanMetricsSet) -> Self {
        Self {
            fragments_scanned: metrics.new_count(FRAGMENTS_SCANNED_METRIC, 0),
            ranges_scanned: metrics.new_count(RANGES_SCANNED_METRIC, 0),
            rows_scanned: metrics.new_count(ROWS_SCANNED_METRIC, 0),
            io_metrics: IoMetrics::new(metrics, 0),
        }
    }
}

/// Partition metrics for the FilteredReadExec node
///
/// These represent work that is divisible by partition and this work is reported on the
/// partition that it belongs to
struct FilteredReadPartitionMetrics {
    // Records the amount of time spent waiting on the lock to grab the next task
    //
    // This should typically be fairly small relative to the overall execution time.  If this
    // value is large then it means we are bottlenecked on the read scheduler which is preventing
    // this partition from being utilized.
    task_wait_time: Time,
    baseline_metrics: BaselineMetrics,
}

impl FilteredReadPartitionMetrics {
    pub fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            task_wait_time: metrics.new_time(TASK_WAIT_TIME_METRIC, partition),
            baseline_metrics: BaselineMetrics::new(metrics, partition),
        }
    }
}

/// Tracks the number of ranges scanned based on the number of rows processed
struct RangeMetricsTracker {
    ranges: Vec<Range<u64>>,
    cumulative_rows: usize,
    current_range_index: usize,
    rows_processed_in_range: usize,
}

impl RangeMetricsTracker {
    fn new(ranges: Vec<Range<u64>>) -> Self {
        Self {
            ranges,
            cumulative_rows: 0,
            current_range_index: 0,
            rows_processed_in_range: 0,
        }
    }

    // Counts ranges started scanning (not necessarily finished).
    fn incremental_ranges_scanned(&mut self, num_rows: usize) -> usize {
        self.cumulative_rows += num_rows;
        let mut additional_ranges = 0;

        while self.current_range_index < self.ranges.len() {
            let current_range = &self.ranges[self.current_range_index];
            let range_size = (current_range.end - current_range.start) as usize;

            if self.cumulative_rows >= range_size {
                // We've completed this range
                if self.rows_processed_in_range == 0 {
                    // We are completing a range we never started
                    additional_ranges += 1;
                }
                self.cumulative_rows -= range_size;
                self.current_range_index += 1;
                self.rows_processed_in_range = 0;
            } else {
                // Still within the current range
                if self.rows_processed_in_range == 0 {
                    additional_ranges += 1;
                }
                self.rows_processed_in_range += num_rows;
                break;
            }
        }

        additional_ranges
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilteredReadThreadingMode {
    /// This mode allows for multi-threading to be used even if there is only a single
    /// partition.  In this mode, readahead will be added via the try_buffered method.
    ///
    /// This mode is slightly less efficient as it is unlikely the decode will happen
    /// on the same thread as any downstream logic.  However, it is simple, and the reads
    /// are sequential.
    ///
    /// The number of threads is specified by the parameter
    OnePartitionMultipleThreads(usize),

    /// This mode will use a single thread per partition.  This is more traditional for
    /// DataFusion and should give better performance for complex queries that have a
    /// lot of downstream processing.  However, you will want to make sure to create the
    /// node with enough partitions or else you will not get any parallelism.
    ///
    /// The number of partitions is specified by the parameter.
    MultiplePartitions(usize),
}

/// The stream of filtered rows that satisfies the FilteredReadExec node
///
/// This represents a scan of a Lance dataset.  Upon creation of the stream we will
/// load the fragments, execute any scalar index query, and then plan out which portions
/// of the fragments we need to read.
///
/// For each fragment, we may read the entire fragment or we may read a portion of it.  We
/// can use both the scan range and the index result to limit the amount of a fragment that
/// we read.
pub struct FilteredReadStream {
    /// The schema of the output of the scan
    output_schema: SchemaRef,
    /// The stream of filtered rows, expressed as a stream of tasks (batch futures)
    ///
    /// This stream can be shared by multiple partitions
    task_stream: Arc<AsyncMutex<BoxStream<'static, Result<ReadBatchFut>>>>,
    /// The scan scheduler for the scan
    scan_scheduler: Arc<ScanScheduler>,
    /// The global metrics for the scan
    metrics: Arc<FilteredReadGlobalMetrics>,
    /// The number of partitions currently running
    ///
    /// We need to know when the final partition completes so that we can
    /// gather the final I/O stats
    active_partitions_counter: Arc<AtomicUsize>,
    /// The threading mode for the scan
    threading_mode: FilteredReadThreadingMode,
    /// Range to apply to the result stream if not already pushed down in planning phase
    scan_range_after_filter: Option<Range<u64>>,
}

impl std::fmt::Debug for FilteredReadStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilteredReadStream").finish()
    }
}

impl FilteredReadStream {
    /// Create FilteredReadStream from pre-planned fragments
    pub(crate) fn from_planned(
        dataset: Arc<Dataset>,
        planned_fragments: Vec<PlannedFragmentRead>,
        options: PlannedFilterReadOptions,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Self {
        let global_metrics = Arc::new(FilteredReadGlobalMetrics::new(metrics));
        let output_schema = Arc::new(options.projection.to_arrow_schema());

        let obj_store = dataset.object_store().clone();
        let scheduler_config = SchedulerConfig::max_bandwidth(&obj_store);
        let scan_scheduler = ScanScheduler::new(Arc::new(obj_store), scheduler_config);

        let fragment_readahead = options.fragment_readahead;

        let scoped_fragments: Vec<ScopedFragmentRead> = planned_fragments
            .into_iter()
            .map(|pf| ScopedFragmentRead {
                fragment: pf.fragment,
                ranges: pf.ranges,
                projection: Arc::new(options.projection.clone()),
                with_deleted_rows: options.with_deleted_rows,
                batch_size: options.batch_size,
                filter: if pf.use_refine {
                    options.refine_filter.clone()
                } else {
                    options.full_filter.clone()
                },
                priority: pf.priority,
                scan_scheduler: scan_scheduler.clone(),
            })
            .collect();

        let global_metrics_clone = global_metrics.clone();
        let scan_range_after_filter = options.scan_range_after_filter.clone();

        let fragment_streams = futures::stream::iter(scoped_fragments)
            .map({
                let scan_range_after_filter_clone = scan_range_after_filter.clone();
                move |scoped_fragment| {
                    let metrics = global_metrics_clone.clone();
                    let limit = scan_range_after_filter_clone.as_ref().map(|r| r.end);
                    tokio::task::spawn(
                        Self::read_fragment(scoped_fragment, metrics, limit).in_current_span(),
                    )
                    .map(|thread_result| thread_result.unwrap())
                }
            })
            .buffered(fragment_readahead);
        let task_stream = fragment_streams.try_flatten().boxed();

        Self {
            output_schema,
            task_stream: Arc::new(AsyncMutex::new(task_stream)),
            scan_scheduler,
            metrics: global_metrics,
            active_partitions_counter: Arc::new(AtomicUsize::new(0)),
            threading_mode: options.threading_mode,
            scan_range_after_filter,
        }
    }

    // There is one underlying task stream, and it can be shared by as many partitions as we
    // want.
    //
    // The behavior of this method depends on the threading mode.  If the threading mode is
    // `OneThreadedPartition` then this method should only be called once.  We will create a
    // stream with readahead using buffered.
    //
    // If the threading mode is `MultiplePartitions` then this method should be called once per
    // partition.  Each stream will have a copy of the same underlying task stream.  Only one stream
    // can poll the underlying task stream at a time (there is a lock on the task stream).  This is
    // generally fine because grabbing a task is cheap (unless we are waiting on I/O).
    //
    // If the threading mode is `MultiplePartitions` then we may operate on the data out-of-order.
    fn get_stream(
        &self,
        metrics: &ExecutionPlanMetricsSet,
        partition: usize,
    ) -> SendableRecordBatchStream {
        self.active_partitions_counter
            .fetch_add(1, Ordering::Relaxed);

        // Each partition needs these to record incremental metrics.
        let global_metrics = self.metrics.clone();
        let scan_scheduler = self.scan_scheduler.clone();

        let partition_metrics = Arc::new(FilteredReadPartitionMetrics::new(metrics, partition));

        match self.threading_mode {
            FilteredReadThreadingMode::OnePartitionMultipleThreads(num_threads) => {
                assert_eq!(partition, 0);
                let output_schema = self.output_schema.clone();
                let task_stream = self.task_stream.clone();
                let partition_metrics_clone = partition_metrics.clone();
                let futures_stream = futures::stream::try_unfold(task_stream, {
                    move |task_stream| {
                        let partition_metrics = partition_metrics_clone.clone();
                        async move {
                            // There is no compute we can meaningfully measure here.  The actual work is
                            // done by spawned background threads.
                            let _timer =
                                partition_metrics.baseline_metrics.elapsed_compute().timer();
                            let _task_wait_timer = partition_metrics.task_wait_time.timer();
                            let maybe_task = task_stream.lock().await.next().await.transpose()?;
                            Result::Ok(maybe_task.map(|task| (task, task_stream)))
                        }
                    }
                });
                let partition_metrics_clone = partition_metrics.clone();
                let base_batch_stream =
                    futures_stream
                        .try_buffered(num_threads)
                        .try_filter_map(move |batch| {
                            std::future::ready(Ok(if batch.num_rows() == 0 {
                                None
                            } else {
                                Some(batch)
                            }))
                        });

                let batch_stream = if let Some(ref range) = self.scan_range_after_filter {
                    FilteredReadUtils::apply_hard_range(base_batch_stream, range.clone()).boxed()
                } else {
                    // Need to box here otherwise the if/else returns incompatible types
                    base_batch_stream.boxed()
                };

                let batch_stream = batch_stream
                    .inspect_ok(move |batch| {
                        partition_metrics_clone
                            .baseline_metrics
                            .record_output(batch.num_rows());
                        global_metrics.io_metrics.record(&scan_scheduler);
                    })
                    .finally(move || {
                        partition_metrics.baseline_metrics.done();
                    })
                    .map_err(|e: lance_core::Error| DataFusionError::External(e.into()))
                    .boxed();

                Box::pin(RecordBatchStreamAdapter::new(output_schema, batch_stream))
            }
            FilteredReadThreadingMode::MultiplePartitions(num_partitions) => {
                assert!(partition < num_partitions);
                let output_schema = self.output_schema.clone();
                let task_stream = self.task_stream.clone();
                let global_metrics_clone = global_metrics.clone();
                let scan_scheduler_clone = scan_scheduler.clone();
                let batch_stream = futures::stream::try_unfold(task_stream, {
                    move |task_stream| {
                        let partition_metrics = partition_metrics.clone();
                        let global_metrics = global_metrics_clone.clone();
                        let scan_scheduler = scan_scheduler_clone.clone();
                        async move {
                            // This isn't quite right.  It's counting I/O time in addition to
                            // compute time.
                            //
                            // TODO: Modify the "read task" concept to have a way of marking when
                            // the 'wait' portion of the task is complete.
                            let _timer =
                                partition_metrics.baseline_metrics.elapsed_compute().timer();
                            let maybe_task = {
                                let _task_wait_timer = partition_metrics.task_wait_time.timer();
                                task_stream.lock().await.next().await
                            };
                            if let Some(task) = maybe_task {
                                let task = task?;
                                let batch = task.await?;
                                partition_metrics
                                    .baseline_metrics
                                    .record_output(batch.num_rows());

                                global_metrics.io_metrics.record(&scan_scheduler);

                                Ok(Some((batch, task_stream)))
                            } else {
                                partition_metrics.baseline_metrics.done();
                                Ok(None)
                            }
                        }
                        .instrument(tracing::debug_span!("filtered_read_task"))
                    }
                })
                .try_filter_map(move |batch| {
                    std::future::ready(Ok(if batch.num_rows() == 0 {
                        None
                    } else {
                        Some(batch)
                    }))
                })
                .map_err(|e: lance_core::Error| DataFusionError::External(e.into()));
                Box::pin(RecordBatchStreamAdapter::new(output_schema, batch_stream))
            }
        }
    }

    // Reads a single fragment into a stream of batch tasks
    #[instrument(name = "read_fragment", skip_all)]
    async fn read_fragment(
        mut fragment_read_task: ScopedFragmentRead,
        global_metrics: Arc<FilteredReadGlobalMetrics>,
        fragment_soft_limit: Option<u64>,
    ) -> Result<impl Stream<Item = Result<ReadBatchFut>>> {
        let output_schema = Arc::new(fragment_read_task.projection.to_arrow_schema());

        if let Some(filter) = &fragment_read_task.filter {
            let filter_cols = Planner::column_names_in_expr(filter);
            if !filter_cols.is_empty() {
                fragment_read_task.projection = Arc::new(
                    fragment_read_task
                        .projection
                        .as_ref()
                        .clone()
                        .union_columns(filter_cols, OnMissing::Error)?,
                );
            }
        }

        let read_schema = fragment_read_task.projection.to_bare_schema();
        let mut fragment_reader = fragment_read_task
            .fragment
            .open(&read_schema, fragment_read_task.frag_read_config())
            .await?;

        if fragment_read_task.with_deleted_rows {
            fragment_reader.with_make_deletions_null();
        }

        // The reader expects sorted ranges and it may be possible to get non-sorted ranges if
        // the row ids are not contiguous
        fragment_read_task.ranges.sort_by_key(|r| r.start);

        let physical_filter = fragment_read_task
            .filter
            .map(|filter| {
                let planner =
                    Planner::new(Arc::new(fragment_read_task.projection.to_arrow_schema()));
                planner.create_physical_expr(&filter)
            })
            .transpose()?;

        // We are going to count the fragment as scanned on the first batch we
        // read. This might miss empty fragments, but we assume that wouldn't be
        // used in the scan anyways.
        let fragment_counted = Arc::new(AtomicBool::new(false));
        let range_tracker = Arc::new(Mutex::new(RangeMetricsTracker::new(
            fragment_read_task.ranges.clone(),
        )));

        let fragment_stream = fragment_reader
            .read_ranges(
                fragment_read_task.ranges.into(),
                fragment_read_task.batch_size,
            )?
            .map(move |batch_fut: ReadBatchFut| {
                let global_metrics = global_metrics.clone();
                let fragment_counted = fragment_counted.clone();
                let range_tracker = range_tracker.clone();
                batch_fut
                    .inspect_ok(move |batch| {
                        let num_rows = batch.num_rows();
                        global_metrics.rows_scanned.add(num_rows);
                        if !fragment_counted.swap(true, Ordering::Relaxed) {
                            global_metrics.fragments_scanned.add(1);
                        }
                        // Note: this is an approximation. Batches may come in out-of-order,
                        // in which case this might be inaccurate.
                        if let Ok(mut range_tracker) = range_tracker.lock() {
                            let additional_ranges =
                                range_tracker.incremental_ranges_scanned(num_rows);
                            global_metrics.ranges_scanned.add(additional_ranges);
                        }
                    })
                    .boxed()
            })
            .zip(futures::stream::repeat((
                physical_filter.clone(),
                output_schema.clone(),
            )))
            .map(|(batch_fut, args)| Self::wrap_with_filter(batch_fut, args.0, args.1));

        let result: Pin<Box<dyn Stream<Item = Result<ReadBatchFut>> + Send>> =
            if let Some(limit) = fragment_soft_limit {
                Box::pin(Self::apply_soft_limit(fragment_stream, limit))
            } else {
                Box::pin(fragment_stream)
            };
        Ok(result)
    }

    fn wrap_with_filter(
        batch_fut: ReadBatchFut,
        filter: Option<Arc<dyn PhysicalExpr>>,
        output_schema: SchemaRef,
    ) -> Result<ReadBatchFut> {
        if let Some(filter) = filter {
            Ok(batch_fut
                .map(move |batch| {
                    let batch = batch?;
                    let batch = datafusion_physical_plan::filter::batch_filter(&batch, &filter)
                        .map_err(|e| Error::Execution {
                            message: format!("Error applying filter expression to batch: {e}"),
                            location: location!(),
                        })?;
                    // Drop any fields loaded purely for the purpose of applying the filter
                    Ok(batch.project_by_schema(output_schema.as_ref())?)
                })
                .boxed())
        } else {
            Ok(batch_fut)
        }
    }

    fn apply_soft_limit<S>(stream: S, limit: u64) -> impl Stream<Item = Result<ReadBatchFut>>
    where
        S: Stream<Item = Result<ReadBatchFut>>,
    {
        let rows_read = Arc::new(AtomicUsize::new(0));

        stream
            .take_while({
                let rows_read = rows_read.clone();
                move |_| future::ready(rows_read.load(Ordering::Relaxed) < limit as usize)
            })
            .map(move |batch_fut_result| {
                let rows_read = rows_read.clone();
                batch_fut_result.map(move |batch_fut| {
                    batch_fut
                        .map(move |batch_result| {
                            batch_result.inspect(|batch| {
                                let batch_rows = batch.num_rows();
                                rows_read.fetch_add(batch_rows, Ordering::Relaxed);
                            })
                        })
                        .boxed()
                })
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::types::UInt32Type;
    use arrow_schema::DataType;
    use futures::TryStreamExt;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::{array, gen_batch};

    #[test]
    fn test_trim_ranges() {
        // Test case 1: Trim ranges that fully overlap
        let ranges = vec![0..10, 10..20, 20..30];
        let fragment_range = 0..30;
        let logical_range = 5..25;
        let result = FilteredReadUtils::trim_ranges(ranges, fragment_range, &logical_range);
        assert_eq!(result, vec![5..10, 10..20, 20..25]);

        // Test case 2: Trim with partial overlap at start
        let ranges = vec![0..10, 10..20];
        let fragment_range = 0..20;
        let logical_range = 5..30;
        let result = FilteredReadUtils::trim_ranges(ranges, fragment_range, &logical_range);
        assert_eq!(result, vec![5..10, 10..20]);

        // Test case 3: Trim with no overlap
        let ranges = vec![0..10, 10..20];
        let fragment_range = 0..20;
        let logical_range = 30..40;
        let result = FilteredReadUtils::trim_ranges(ranges, fragment_range, &logical_range);
        assert_eq!(result, Vec::<Range<u64>>::new());

        // Test case 4: Trim with single range
        let ranges = vec![10..50];
        let fragment_range = 10..50;
        let logical_range = 20..30;
        let result = FilteredReadUtils::trim_ranges(ranges, fragment_range, &logical_range);
        assert_eq!(result, vec![20..30]);

        // Test case 5: Trim ranges with gaps (deletion vector case)
        // The logical range 5..28 only includes 23 logical rows (5..28)
        // From ranges: 0..10 (10 rows), 15..20 (5 rows), 25..30 (5 rows) = 20 logical rows total
        // Skip first 5 rows from 0..10, take all 15 remaining: 5..10 (5 rows), 15..20 (5 rows), 25..30 (5 rows)
        let ranges = vec![0..10, 15..20, 25..30];
        let fragment_range = 0..30;
        let logical_range = 5..28;
        let result = FilteredReadUtils::trim_ranges(ranges, fragment_range, &logical_range);
        // We skip 5 from the first range (leaving 5..10), take all of 15..20, and all of 25..30
        // because we only have 20 logical rows total and need 23
        assert_eq!(result, vec![5..10, 15..20, 25..30]);
    }

    #[test]
    fn test_intersect_ranges() {
        // Test case 1: Perfect overlap
        let ranges1 = vec![0..10, 20..30];
        let ranges2 = vec![0..10, 20..30];
        let result = FilteredReadUtils::intersect_ranges(&ranges1, &ranges2);
        assert_eq!(result, vec![0..10, 20..30]);

        // Test case 2: Partial overlap
        let ranges1 = vec![0..15, 20..30];
        let ranges2 = vec![10..25];
        let result = FilteredReadUtils::intersect_ranges(&ranges1, &ranges2);
        assert_eq!(result, vec![10..15, 20..25]);

        // Test case 3: No overlap
        let ranges1 = vec![0..10];
        let ranges2 = vec![20..30];
        let result = FilteredReadUtils::intersect_ranges(&ranges1, &ranges2);
        assert_eq!(result, Vec::<Range<u64>>::new());

        // Test case 4: Multiple intersections
        let ranges1 = vec![0..10, 15..25, 30..40];
        let ranges2 = vec![5..20, 28..35];
        let result = FilteredReadUtils::intersect_ranges(&ranges1, &ranges2);
        assert_eq!(result, vec![5..10, 15..20, 30..35]);

        // Test case 5: One range fully contains another
        let ranges1 = vec![0..100];
        let ranges2 = vec![10..20, 30..40, 50..60];
        let result = FilteredReadUtils::intersect_ranges(&ranges1, &ranges2);
        assert_eq!(result, vec![10..20, 30..40, 50..60]);
    }

    #[test]
    fn test_apply_skip_take_to_ranges() {
        // Test case 1: Skip first range completely
        let mut ranges = vec![0..10, 10..20, 20..30];
        let mut to_skip = 10;
        let mut to_take = 15;
        FilteredReadUtils::apply_skip_take_to_ranges(&mut ranges, &mut to_skip, &mut to_take);
        assert_eq!(ranges, vec![10..20, 20..25]);
        assert_eq!(to_skip, 0);
        assert_eq!(to_take, 0);

        // Test case 2: Skip partial range
        let mut ranges = vec![0..10, 10..20, 20..30];
        let mut to_skip = 5;
        let mut to_take = 10;
        FilteredReadUtils::apply_skip_take_to_ranges(&mut ranges, &mut to_skip, &mut to_take);
        assert_eq!(ranges, vec![5..10, 10..15]);
        assert_eq!(to_skip, 0);
        assert_eq!(to_take, 0);

        // Test case 3: Take more than available
        let mut ranges = vec![0..10, 10..20];
        let mut to_skip = 0;
        let mut to_take = 100;
        FilteredReadUtils::apply_skip_take_to_ranges(&mut ranges, &mut to_skip, &mut to_take);
        assert_eq!(ranges, vec![0..10, 10..20]);
        assert_eq!(to_skip, 0);
        assert_eq!(to_take, 80);

        // Test case 4: Skip everything
        let mut ranges = vec![0..10, 10..20];
        let mut to_skip = 25;
        let mut to_take = 10;
        FilteredReadUtils::apply_skip_take_to_ranges(&mut ranges, &mut to_skip, &mut to_take);
        assert_eq!(ranges, Vec::<Range<u64>>::new());
        assert_eq!(to_skip, 5);
        assert_eq!(to_take, 10);

        // Test case 5: Take exactly one range
        let mut ranges = vec![0..10, 10..20, 20..30];
        let mut to_skip = 10;
        let mut to_take = 10;
        FilteredReadUtils::apply_skip_take_to_ranges(&mut ranges, &mut to_skip, &mut to_take);
        assert_eq!(ranges, vec![10..20]);
        assert_eq!(to_skip, 0);
        assert_eq!(to_take, 0);
    }

    #[test]
    fn test_dv_to_valid_ranges() {
        // Test case 1: No deletions
        let dv = vec![];
        let ranges: Vec<Range<u64>> = DvToValidRanges::new(dv.into_iter(), 10).collect();
        assert_eq!(ranges, vec![0..10]);

        // Test case 2: Single deletion at start
        let dv = vec![0];
        let ranges: Vec<Range<u64>> = DvToValidRanges::new(dv.into_iter(), 10).collect();
        assert_eq!(ranges, vec![1..10]);

        // Test case 3: Single deletion at end
        let dv = vec![9];
        let ranges: Vec<Range<u64>> = DvToValidRanges::new(dv.into_iter(), 10).collect();
        assert_eq!(ranges, vec![0..9]);

        // Test case 4: Multiple consecutive deletions
        let dv = vec![3, 4, 5];
        let ranges: Vec<Range<u64>> = DvToValidRanges::new(dv.into_iter(), 10).collect();
        assert_eq!(ranges, vec![0..3, 6..10]);

        // Test case 5: Multiple non-consecutive deletions
        let dv = vec![2, 5, 8];
        let ranges: Vec<Range<u64>> = DvToValidRanges::new(dv.into_iter(), 10).collect();
        assert_eq!(ranges, vec![0..2, 3..5, 6..8, 9..10]);

        // Test case 6: All rows deleted
        let dv: Vec<u64> = (0..10).collect();
        let ranges: Vec<Range<u64>> = DvToValidRanges::new(dv.into_iter(), 10).collect();
        assert_eq!(ranges, Vec::<Range<u64>>::new());

        // Test case 7: Example from docstring
        let dv = vec![10, 15, 16];
        let ranges: Vec<Range<u64>> = DvToValidRanges::new(dv.into_iter(), 100).collect();
        assert_eq!(ranges, vec![0..10, 11..15, 17..100]);
    }

    #[tokio::test]
    async fn test_metrics_with_limit_partial_fragment() {
        // Create a simple dataset with 100 rows in one fragment
        let tmp_path = TempStrDir::default();
        let dataset = gen_batch()
            .col("id", array::step::<UInt32Type>())
            .col("value", array::rand_type(&DataType::Float32))
            .into_dataset(
                tmp_path.as_str(),
                FragmentCount::from(1),
                FragmentRowCount::from(100),
            )
            .await
            .unwrap();
        let dataset = Arc::new(dataset);

        // Plan a scan that only reads the first 30 rows
        let fragments = dataset.fragments().as_ref().clone();
        let options = FilteredReadOptions::basic_full_read(&dataset).with_batch_size(10);
        let (planned_fragments, planned_options) =
            FilteredReadUtils::plan_scan(&dataset, fragments, None, &options)
                .await
                .unwrap();

        // Create PlannedFilterReadExec and execute
        let planned_read = Arc::new(PlannedFilterReadExec::new(
            dataset.clone(),
            planned_fragments,
            planned_options,
        ));

        // Execute and take only 30 rows (3 batches of 10)
        let batches = planned_read
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .take(3)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 30);

        // Check metrics reflect partial fragment read
        let metrics = planned_read.metrics().unwrap();

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
