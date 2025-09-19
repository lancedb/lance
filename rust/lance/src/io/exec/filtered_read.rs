// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::any::Any;
use std::iter::Peekable;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{ops::Range, sync::Arc};

use arrow::array::AsArray;
use arrow::datatypes::UInt32Type;
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
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
use datafusion_physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::metrics::{BaselineMetrics, Count, MetricsSet, Time};
use datafusion_physical_plan::Statistics;
use futures::stream::BoxStream;
use futures::{FutureExt, Stream, StreamExt, TryStreamExt};
use lance_arrow::RecordBatchExt;
use lance_core::datatypes::OnMissing;
use lance_core::utils::deletion::DeletionVector;
use lance_core::utils::futures::FinallyStreamExt;
use lance_core::utils::mask::RowIdMask;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{datatypes::Projection, Error, Result};
use lance_datafusion::planner::Planner;
use lance_datafusion::utils::{
    ExecutionPlanMetricsSetExt, FRAGMENTS_SCANNED_METRIC, RANGES_SCANNED_METRIC,
    ROWS_SCANNED_METRIC, TASK_WAIT_TIME_METRIC,
};
use lance_index::scalar::expression::{FilterPlan, IndexExprResult};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_table::format::Fragment;
use lance_table::rowids::RowIdSequence;
use lance_table::utils::stream::ReadBatchFut;
use roaring::RoaringBitmap;
use snafu::location;
use tokio::sync::Mutex as AsyncMutex;
use tracing::{instrument, Instrument};

use crate::dataset::fragment::{FileFragment, FragReadConfig};
use crate::dataset::rowids::load_row_id_sequence;
use crate::dataset::scanner::{
    get_default_batch_size, BATCH_SIZE_FALLBACK, DEFAULT_FRAGMENT_READAHEAD,
};
use crate::Dataset;

use super::utils::IoMetrics;

#[derive(Debug)]
pub struct EvaluatedIndex {
    index_result: IndexExprResult,
    applicable_fragments: RoaringBitmap,
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

/// A fragment with all of its metadata loaded
struct LoadedFragment {
    row_id_sequence: Arc<RowIdSequence>,
    deletion_vector: Option<Arc<DeletionVector>>,
    fragment: FileFragment,
    // The number of physical rows in the fragment
    //
    // This count includes deleted rows
    num_physical_rows: u64,
    // The number of logical rows in the fragment
    //
    // This count does not include deleted rows
    num_logical_rows: u64,
}

/// Given a sorted iterator of deleted row offsets, return a sorted iterator of valid row ranges
///
/// For example, given a fragment with 100 rows, and a deletion vector of 10, 15, 16 this would
/// return 0..10, 11..15, 17..100
struct DvToValidRanges<I: Iterator<Item = u64> + Send> {
    deleted_rows: I,
    num_rows: u64,
    position: u64,
}

impl<I: Iterator<Item = u64> + Send> DvToValidRanges<I> {
    fn new(deleted_rows: I, num_rows: u64) -> Self {
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

/// Given a sorted iterator of ranges and a deletion vector, return a sorted iterator of
/// ranges that have been filtered by the deletion vector
///
/// For example, given the ranges [0..10, 50..60] and the deletion vector [7, 52, 53, 54]
/// then return the ranges [0..7, 8..10, 50..52, 55..60]
struct DvRangeFilter<I: Iterator<Item = Range<u64>>, D: Iterator<Item = u64>> {
    ranges: Peekable<I>,
    deletion_vector: Peekable<D>,
}

impl<I: Iterator<Item = Range<u64>>, D: Iterator<Item = u64>> DvRangeFilter<I, D> {
    fn new(
        ranges: impl IntoIterator<IntoIter = I>,
        deletion_vector: impl IntoIterator<IntoIter = D>,
    ) -> Self {
        Self {
            ranges: ranges.into_iter().peekable(),
            deletion_vector: deletion_vector.into_iter().peekable(),
        }
    }
}

impl<I: Iterator<Item = Range<u64>>, D: Iterator<Item = u64>> Iterator for DvRangeFilter<I, D> {
    type Item = Range<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(next_range) = self.ranges.peek_mut() {
            while let Some(cur_dv_pos) = self.deletion_vector.peek() {
                if *cur_dv_pos < next_range.start {
                    // Delete is before the range, skip the delete
                    self.deletion_vector.next();
                } else if *cur_dv_pos >= next_range.end {
                    // Delete is after the range, return the range, keep the delete
                    return self.ranges.next();
                } else {
                    // Delete intersects the range
                    if *cur_dv_pos == next_range.start {
                        next_range.start += 1;
                        if next_range.start == next_range.end {
                            // The range is now empty, consume it and grab the next range
                            break;
                        }
                    } else {
                        // Delete is in the middle of the range, split the range
                        let new_range = next_range.start..*cur_dv_pos;
                        next_range.start = *cur_dv_pos + 1;
                        if next_range.start == next_range.end {
                            // The range is now empty, consume it
                            self.ranges.next();
                        }
                        return Some(new_range);
                    }
                }
            }
            if !next_range.is_empty() {
                // No more deletes, consume and return the range
                return self.ranges.next();
            } else {
                // We got here because we broke out on an empty range, move to next range
                self.ranges.next();
            }
        }
        // No more ranges, return None
        None
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
pub struct FilteredReadPartitionMetrics {
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
struct FilteredReadStream {
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
}

impl std::fmt::Debug for FilteredReadStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilteredReadStream").finish()
    }
}

impl FilteredReadStream {
    #[instrument(name = "init_filtered_read_stream", skip_all)]
    async fn try_new(
        dataset: Arc<Dataset>,
        options: FilteredReadOptions,
        metrics: &ExecutionPlanMetricsSet,
        evaluated_index: Option<Arc<EvaluatedIndex>>,
    ) -> DataFusionResult<Self> {
        let global_metrics = FilteredReadGlobalMetrics::new(metrics);

        let threading_mode = options.threading_mode;

        let io_parallelism = dataset.object_store.io_parallelism();
        let fragment_readahead = options
            .fragment_readahead
            .unwrap_or_else(|| ((*DEFAULT_FRAGMENT_READAHEAD).unwrap_or(io_parallelism * 2)))
            .max(1);

        let fragments = options
            .fragments
            .clone()
            .unwrap_or_else(|| dataset.fragments().clone());
        global_metrics.fragments_scanned.add(fragments.len());

        // Ideally we don't need to collect here but if we don't we get "implementation of FnOnce is
        // not general enough" false positives from rustc
        let frag_futs = fragments
            .iter()
            .map(|frag| {
                Result::Ok(Self::load_fragment(
                    dataset.clone(),
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

        let output_schema = Arc::new(options.projection.to_arrow_schema());

        let obj_store = dataset.object_store.clone();
        let scheduler_config = SchedulerConfig::max_bandwidth(obj_store.as_ref());
        let scan_scheduler = ScanScheduler::new(obj_store, scheduler_config);

        let scoped_fragments = Self::plan_scan(
            dataset.as_ref(),
            loaded_fragments,
            &evaluated_index,
            options,
            &global_metrics,
            scan_scheduler.clone(),
        )
        .await?;

        let fragment_streams = futures::stream::iter(scoped_fragments)
            .map(|scoped_fragment| {
                tokio::task::spawn(Self::read_fragment(scoped_fragment))
                    .map(|thread_result| thread_result.unwrap())
            })
            .buffered(fragment_readahead);
        let task_stream = fragment_streams.try_flatten().boxed();

        Ok(Self {
            output_schema,
            task_stream: Arc::new(AsyncMutex::new(task_stream)),
            scan_scheduler,
            metrics: Arc::new(global_metrics),
            active_partitions_counter: Arc::new(AtomicUsize::new(0)),
            threading_mode,
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

    // This method is a bit complicated
    //
    // We start with a list of fragments, potentially a scalar index result, and a scan range.
    //
    // We need to figure out which ranges to read from each fragment.
    //
    // If the scan range is ignoring the filters we can push it down here.
    // If the scan range is not ignoring the filters we can only push it down if the index
    // result is an exact match.
    #[instrument(name = "plan_scan", skip_all)]
    async fn plan_scan(
        dataset: &Dataset,
        fragments: Vec<LoadedFragment>,
        evaluated_index: &Option<Arc<EvaluatedIndex>>,
        options: FilteredReadOptions,
        global_metrics: &FilteredReadGlobalMetrics,
        scan_scheduler: Arc<ScanScheduler>,
    ) -> Result<Vec<ScopedFragmentRead>> {
        let mut scoped_fragments = Vec::with_capacity(fragments.len());
        let projection = Arc::new(options.projection);

        // The current offset, includes filtered rows, but not deleted rows
        let mut range_offset = 0;
        // The current offset, does not include filtered or deleted rows
        //
        // This will be set to None once we encounter a fragment where we don't
        // know the number of filtered rows up front
        let mut _filtered_range_offset = Some(0);

        let refine_filter = options.refine_filter;
        let full_filter = options.full_filter;

        for (
            priority,
            LoadedFragment {
                row_id_sequence,
                fragment,
                num_logical_rows,
                num_physical_rows,
                deletion_vector,
            },
        ) in fragments.into_iter().enumerate()
        {
            let range_start = range_offset;
            let range_end = if options.with_deleted_rows {
                range_offset += num_physical_rows;
                range_start + num_physical_rows
            } else {
                range_offset += num_logical_rows;
                range_start + num_logical_rows
            };

            // By default we assume we will need to apply the full filter
            // This will get refined if we have an exact match
            let mut filter = &full_filter;

            let mut to_read: Vec<Range<u64>> = if let Some(evaluated_index) = evaluated_index {
                if evaluated_index
                    .applicable_fragments
                    .contains(fragment.id() as u32)
                {
                    let _span =
                        tracing::span!(tracing::Level::DEBUG, "apply_index_result").entered();
                    // There is an index result, and it applies to this fragment, so we can maybe
                    // reduce the amount of data we read, and we can potentially push the scan range
                    // down
                    match &evaluated_index.index_result {
                        IndexExprResult::Exact(row_id_mask) => {
                            // The index result is an exact match, so we can both reduce the amount
                            // of data we read, and we can push a filtered scan range down

                            // Also, with an exact match, we only need to apply the refine filter
                            filter = &refine_filter;

                            let valid_ranges = row_id_sequence.mask_to_offset_ranges(row_id_mask);
                            if let Some(deletion_vector) = &deletion_vector {
                                Self::filter_deleted_rows(valid_ranges, deletion_vector)
                            } else {
                                valid_ranges
                            }
                        }
                        IndexExprResult::AtMost(row_id_mask) => {
                            // The index result is an at most, so we can reduce the amount of data
                            // we read, but we can't push a filtered scan range down
                            let valid_ranges = row_id_sequence.mask_to_offset_ranges(row_id_mask);
                            if let Some(deletion_vector) = &deletion_vector {
                                Self::filter_deleted_rows(valid_ranges, deletion_vector)
                            } else {
                                valid_ranges
                            }
                        }
                        // If index result is AtLeast then must read entire fragment
                        //
                        // TODO: In the future we should be able to reduce the compute cost by only
                        // applying the index filter to the rows that don't match the mask (we can
                        // assume rows that match the mask are true)
                        IndexExprResult::AtLeast(_) => {
                            Self::full_frag_range(num_physical_rows, &deletion_vector)
                        }
                    }
                } else {
                    // Index result does not apply to fragment, must read entire fragment
                    // minus deletion vector (if any)
                    Self::full_frag_range(num_physical_rows, &deletion_vector)
                }
            } else {
                // No scalar index result, must read entire fragment minus deletion vector (if any)
                Self::full_frag_range(num_physical_rows, &deletion_vector)
            };

            if let Some(range_before_filter) = &options.scan_range_before_filter {
                to_read = Self::trim_ranges(to_read, range_start..range_end, range_before_filter);
            }

            if !to_read.is_empty() {
                global_metrics
                    .rows_scanned
                    .add(to_read.iter().map(|r| r.end - r.start).sum::<u64>() as usize);
                global_metrics.ranges_scanned.add(to_read.len());
                log::trace!(
                    "Reading {} ranges ({} rows) from fragment {} with filter: {:?}",
                    to_read.len(),
                    to_read.iter().map(|r| r.end - r.start).sum::<u64>(),
                    fragment.id(),
                    filter
                );
                scoped_fragments.push(ScopedFragmentRead {
                    fragment: fragment.clone(),
                    ranges: to_read,
                    projection: projection.clone(),
                    with_deleted_rows: options.with_deleted_rows,
                    batch_size: options.batch_size.unwrap_or(
                        get_default_batch_size().unwrap_or_else(|| {
                            std::cmp::max(
                                dataset.object_store().block_size() / 4,
                                BATCH_SIZE_FALLBACK,
                            )
                        }) as u32,
                    ),
                    filter: filter.clone(),
                    priority: priority as u32,
                    scan_scheduler: scan_scheduler.clone(),
                });
            } else {
                log::trace!(
                    "Skipping fragment {} because it was outside the scan range",
                    fragment.id()
                );
            }
        }

        Ok(scoped_fragments)
    }

    #[instrument(level = "debug", skip_all)]
    fn filter_deleted_rows(
        ranges: Vec<Range<u64>>,
        deletion_vector: &Arc<DeletionVector>,
    ) -> Vec<Range<u64>> {
        DvRangeFilter::new(
            ranges,
            deletion_vector.to_sorted_iter().map(|pos| pos as u64),
        )
        .collect()
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
            vec![0..num_physical_rows]
        }
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

        // Each partition needs a copy of these things because the last partition
        // to finish has to record the I/O stats
        let active_partitions_counter = self.active_partitions_counter.clone();
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
                let batch_stream = futures_stream
                    .try_buffered(num_threads)
                    .try_filter_map(move |batch| {
                        std::future::ready(Ok(if batch.num_rows() == 0 {
                            None
                        } else {
                            Some(batch)
                        }))
                    })
                    .inspect_ok(move |batch| {
                        partition_metrics_clone
                            .baseline_metrics
                            .record_output(batch.num_rows());
                    })
                    .finally(move || {
                        if active_partitions_counter.fetch_sub(1, Ordering::Relaxed) == 1 {
                            global_metrics.io_metrics.record_final(&scan_scheduler);
                        }
                        partition_metrics.baseline_metrics.done();
                    })
                    .map_err(|e: lance_core::Error| DataFusionError::External(e.into()));
                Box::pin(RecordBatchStreamAdapter::new(output_schema, batch_stream))
            }
            FilteredReadThreadingMode::MultiplePartitions(num_partitions) => {
                assert!(partition < num_partitions);
                let output_schema = self.output_schema.clone();
                let task_stream = self.task_stream.clone();
                let batch_stream = futures::stream::try_unfold(task_stream, {
                    move |task_stream| {
                        let partition_metrics = partition_metrics.clone();
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
                .finally(move || {
                    if active_partitions_counter.fetch_sub(1, Ordering::Relaxed) == 1 {
                        global_metrics.io_metrics.record_final(&scan_scheduler);
                    }
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

        Ok(fragment_reader
            .read_ranges(
                fragment_read_task.ranges.into(),
                fragment_read_task.batch_size,
            )?
            .zip(futures::stream::repeat((
                physical_filter.clone(),
                output_schema.clone(),
            )))
            .map(|(batch_fut, args)| Self::wrap_with_filter(batch_fut, args.0, args.1)))
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
    // When execute is first called we will initialize the FilteredReadStream.  In order to support
    // multiple partitions, each partition will share the stream.
    running_stream: Arc<AsyncMutex<Option<FilteredReadStream>>>,
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
            // TODO: Add support for this.  It isn't too tricky to apply the limit when we have a scalar index.
            // However, we need to make sure and apply the limit with unindexed fragments too.  This is also not
            // too bad...unless there are multiple partitions.  In that case we have several readers reading in
            // parallel and applying the limit after the fact is tricky.
            return Err(Error::NotSupported {
                source: "scan_range_after_filter not yet implemented"
                    .to_string()
                    .into(),
                location: location!(),
            });
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
            running_stream: Arc::new(AsyncMutex::new(None)),
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
        let running_stream_lock = self.running_stream.clone();
        let dataset = self.dataset.clone();
        let options = self.options.clone();
        let metrics = self.metrics.clone();
        let index_input = self.index_input.clone();

        let stream = futures::stream::once(async move {
            let mut running_stream = running_stream_lock.lock().await;
            if let Some(running_stream) = &*running_stream {
                DataFusionResult::<SendableRecordBatchStream>::Ok(
                    running_stream.get_stream(&metrics, partition),
                )
            } else {
                let mut evaluated_index = None;
                if let Some(index_input) = index_input {
                    let mut index_search = index_input.execute(partition, context)?;
                    let index_search_result =
                        index_search.next().await.ok_or_else(|| Error::Internal {
                            message: "Index search did not yield any results".to_string(),
                            location: location!(),
                        })??;
                    evaluated_index = Some(Arc::new(EvaluatedIndex::try_from_arrow(
                        &index_search_result,
                    )?));
                }

                let new_running_stream =
                    FilteredReadStream::try_new(dataset, options, &metrics, evaluated_index)
                        .await?;
                let first_stream = new_running_stream.get_stream(&metrics, partition);
                *running_stream = Some(new_running_stream);
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
            // If there is no filter, we just return the total number of rows (sans any before/after-filter range)
            // divided by the number of partitions.
            let total_rows =
                if let Some(scan_range_after_filter) = &self.options.scan_range_after_filter {
                    total_rows.min(scan_range_after_filter.end - scan_range_after_filter.start)
                } else {
                    total_rows
                };
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
                running_stream: Arc::new(AsyncMutex::new(None)),
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
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use arrow::{
        compute::concat_batches,
        datatypes::{Float32Type, UInt32Type, UInt64Type},
    };
    use arrow_array::{cast::AsArray, Array, UInt32Array};
    use itertools::Itertools;
    use lance_core::datatypes::OnMissing;
    use lance_datagen::{array, gen_batch, BatchCount, Dimension, RowCount};
    use lance_index::{
        optimize::OptimizeOptions,
        scalar::{expression::PlannerIndexExt, ScalarIndexParams},
        DatasetIndexExt, IndexType,
    };
    use tempfile::TempDir;

    use crate::{
        dataset::{InsertBuilder, WriteDestination, WriteMode, WriteParams},
        index::DatasetIndexInternalExt,
        io::exec::scalar_index::ScalarIndexExec,
        utils::test::{DatagenExt, FragmentCount, FragmentRowCount},
    };

    use super::*;

    struct TestFixture {
        _tmp_path: TempDir,
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
            let tmp_path = tempfile::tempdir().unwrap();

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
                    tmp_path.path().to_str().unwrap(),
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

    #[test]
    fn test_dv_range_filter() {
        let check = |dv: DeletionVector, ranges: Vec<Range<u64>>, expected: Vec<Range<u64>>| {
            let ranges = DvRangeFilter::new(ranges, dv.into_sorted_iter().map(|val| val as u64))
                .collect::<Vec<_>>();
            assert_eq!(ranges, expected);
        };

        // Range already doesn't include 10, no change
        let dv = DeletionVector::from_iter(vec![10, 50]);
        check(dv.clone(), vec![0..10, 11..20], vec![0..10, 11..20]);

        // Range starts with 10
        check(dv.clone(), vec![10..20, 21..30], vec![11..20, 21..30]);

        // Range ends with 10
        check(dv, vec![0..3, 7..11, 15..16], vec![0..3, 7..10, 15..16]);

        // Sequence of deleted rows
        let dv = DeletionVector::from_iter(vec![15, 16, 17, 18, 19, 20]);
        check(dv.clone(), vec![0..3, 15..21], vec![0..3]);

        check(dv, vec![0..3, 15..30], vec![0..3, 21..30]);
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
        let tmp_path = tempfile::tempdir().unwrap();
        let mut dataset = gen_batch()
            .col("a", array::step::<UInt32Type>())
            .into_dataset(
                tmp_path.path().to_str().unwrap(),
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

    #[test]
    fn test_trim_ranges() {
        let ranges = vec![0..10, 15..25, 30..40];

        assert_eq!(
            FilteredReadStream::trim_ranges(ranges.clone(), 0..25, &(0..10)),
            vec![0..10]
        );

        assert_eq!(
            FilteredReadStream::trim_ranges(ranges.clone(), 0..25, &(10..15)),
            vec![15..20]
        );

        assert_eq!(
            FilteredReadStream::trim_ranges(ranges.clone(), 0..25, &(15..20)),
            vec![20..25]
        );

        assert_eq!(
            FilteredReadStream::trim_ranges(ranges, 0..25, &(15..25)),
            vec![20..25, 30..35]
        );
    }

    #[test]
    fn test_full_frag_range() {
        let dv = Arc::new(DeletionVector::Set(HashSet::from_iter([
            13, 52, 51, 51, 17,
        ])));
        let ranges = FilteredReadStream::full_frag_range(53, &Some(dv));
        let expected = vec![0..13, 14..17, 18..51];
        assert_eq!(ranges, expected);
    }
}
