// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::any::Any;
use std::iter::Peekable;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{ops::Range, sync::Arc};

use arrow_schema::SchemaRef;
use async_recursion::async_recursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    execution_plan::{Boundedness, EmissionType},
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion_physical_plan::metrics::{BaselineMetrics, Count, Time};
use futures::stream::BoxStream;
use futures::{FutureExt, Stream, StreamExt, TryStreamExt};
use lance_core::utils::deletion::DeletionVector;
use lance_core::utils::futures::FinallyStreamExt;
use lance_core::{datatypes::Projection, Error, Result};
use lance_datafusion::planner::Planner;
use lance_datafusion::utils::{
    ExecutionPlanMetricsSetExt, FRAGMENTS_SCANNED_METRIC, RANGES_SCANNED_METRIC,
    ROWS_SCANNED_METRIC, TASK_WAIT_TIME_METRIC,
};
use lance_index::scalar::expression::{FilterPlan, IndexExprResult, ScalarIndexExpr};
use lance_index::DatasetIndexExt;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_table::format::Fragment;
use lance_table::rowids::RowIdSequence;
use lance_table::utils::stream::ReadBatchFut;
use roaring::RoaringBitmap;
use snafu::location;
use tokio::sync::Mutex as AsyncMutex;

use crate::dataset::fragment::{FileFragment, FragReadConfig};
use crate::dataset::rowids::load_row_id_sequence;
use crate::dataset::scanner::{
    get_default_batch_size, BATCH_SIZE_FALLBACK, DEFAULT_FRAGMENT_READAHEAD,
};
use crate::Dataset;

use super::utils::{IndexMetrics, IoMetrics};

struct EvaluatedIndex {
    index_result: IndexExprResult,
    applicable_fragments: RoaringBitmap,
}

/// A fragment along with ranges of row offsets to read
struct ScopedFragmentRead {
    fragment: FileFragment,
    ranges: Vec<Range<u64>>,
    projection: Arc<Projection>,
    with_deleted_rows: bool,
    batch_size: u32,
    filter: Option<Arc<dyn PhysicalExpr>>,
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
                self.position = next_deleted_row;
                return Some(position..next_deleted_row);
            }
        }
        let position = self.position;
        self.position = self.num_rows;
        Some(position..self.num_rows)
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
}

impl std::fmt::Debug for FilteredReadStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilteredReadStream").finish()
    }
}

impl FilteredReadStream {
    async fn try_new(
        dataset: Arc<Dataset>,
        options: FilteredReadOptions,
        metrics: &ExecutionPlanMetricsSet,
    ) -> DataFusionResult<Self> {
        // All stats are reported on partition 0
        let index_metrics = IndexMetrics::new(metrics, /*partition=*/ 0);
        let evaluated_index =
            Self::evaluate_index_query(&options.filter_plan, dataset.as_ref(), &index_metrics)
                .await?;

        let global_metrics = FilteredReadGlobalMetrics::new(metrics);

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

        let output_schema = Arc::new(options.projection.to_arrow_schema()?);

        let obj_store = dataset.object_store.clone();
        let scheduler_config = SchedulerConfig::max_bandwidth(obj_store.as_ref());
        let scan_scheduler = ScanScheduler::new(obj_store, scheduler_config);

        let scoped_fragments = Self::plan_scan(
            dataset.as_ref(),
            output_schema.clone(),
            loaded_fragments,
            &evaluated_index,
            options,
            &global_metrics,
            scan_scheduler.clone(),
        )
        .await?;

        let fragment_streams = futures::stream::iter(scoped_fragments)
            .map(Self::read_fragment)
            .buffered(fragment_readahead);
        let task_stream = fragment_streams.try_flatten().boxed();

        Ok(Self {
            output_schema,
            task_stream: Arc::new(AsyncMutex::new(task_stream)),
            scan_scheduler,
            metrics: Arc::new(global_metrics),
            active_partitions_counter: Arc::new(AtomicUsize::new(0)),
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
        let (row_id_sequence, num_logical_rows) = if dataset.manifest.uses_move_stable_row_ids() {
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
    async fn plan_scan(
        dataset: &Dataset,
        output_schema: SchemaRef,
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

        let planner = Planner::new(output_schema);
        let refine_filter = options
            .filter_plan
            .refine_expr
            .map(|refine_expr| planner.create_physical_expr(&refine_expr))
            .transpose()?;
        let full_filter = options
            .filter_plan
            .full_expr
            .map(|full_expr| planner.create_physical_expr(&full_expr))
            .transpose()?;

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

                            // let filtered_start = filtered_row_offset;
                            // let num_filtered_rows =
                            //     frag_ranges.iter().map(|r| r.end - r.start).sum::<u64>();
                            // let filtered_end = filtered_start + num_filtered_rows;

                            // // TODO: Even if a row matches the index result, it still might be deleted
                            // // and so it wouldn't be accurate to adjust filtered_row_offset in this way.
                            // filtered_row_offset += num_filtered_rows;

                            // match &options.scan_range {
                            //     ScanRange::RangeWithinDataset(range) => {
                            //         if is_whole_dataset {
                            //             Self::trim_ranges(
                            //                 frag_ranges,
                            //                 // filtered_start..filtered_end is in terms of dataset rows
                            //                 // because we are scanning the whole dataset and the fragments
                            //                 // have been sorted by dataset offset
                            //                 filtered_start..filtered_end,
                            //                 range,
                            //             )
                            //         } else {
                            //             // If we're not scanning the whole dataset, and the requested
                            //             // range is in terms of filtered rows of the dataset, then we
                            //             // can't push it down because we don't know how many filtered
                            //             // rows were inside of fragments we skipped (in theory we
                            //             // could still look at fragments we aren't reading but lets
                            //             // save that for another day)
                            //             frag_ranges
                            //         }
                            //     }
                            //     ScanRange::RangeWithinFragments(range) => Self::trim_ranges(
                            //         frag_ranges,
                            //         filtered_start..filtered_end,
                            //         range,
                            //     ),
                            //     _ => frag_ranges,
                            // }
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
            if to_skip > range_len {
                to_skip -= range_len;
                continue;
            }
            let to_take_here = range_len.min(to_take);
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

    async fn evaluate_index_query(
        filter: &FilterPlan,
        dataset: &Dataset,
        index_metrics: &IndexMetrics,
    ) -> DataFusionResult<Option<Arc<EvaluatedIndex>>> {
        if let Some(index_query) = &filter.index_query {
            let index_result = index_query.evaluate(dataset, index_metrics).await?;
            let applicable_fragments =
                Self::fragments_covered_by_index_query(index_query, dataset).await?;
            Ok(Some(Arc::new(EvaluatedIndex {
                index_result,
                applicable_fragments,
            })))
        } else {
            Ok(None)
        }
    }

    // There is one underlying task stream, and it can be shared by as many partitions as we
    // want.
    //
    // The resulting stream is a stream of batches.  Each time it is polled it first acquires
    // a lock and grabs the next task (this is very often synchronous).  It then runs the task
    // to decode the batch.  This keeps any downstream work on the same core as the decode
    //
    // TODO: Currently this is implemented in a first-come first-serve fashion.  If we change
    // it to round-robin we would sacrifice some work-stealing but we would be able to re-sequence
    // the fragments downstream to maintain ordered execution in the face of parallelism.
    //
    // At the very least, this should be an option.
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
                    let _timer = partition_metrics.baseline_metrics.elapsed_compute().timer();
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
            }
        })
        .finally(move || {
            if active_partitions_counter.fetch_sub(1, Ordering::Relaxed) == 1 {
                global_metrics.io_metrics.record_final(&scan_scheduler);
            }
        })
        .map_err(|e: lance_core::Error| DataFusionError::External(e.into()));
        Box::pin(RecordBatchStreamAdapter::new(output_schema, batch_stream))
    }

    // Reads a single fragment into a stream of batch tasks
    async fn read_fragment(
        mut fragment_read_task: ScopedFragmentRead,
    ) -> Result<impl Stream<Item = Result<ReadBatchFut>>> {
        let projection = fragment_read_task.projection.to_schema();
        let mut fragment_reader = fragment_read_task
            .fragment
            .open(&projection, fragment_read_task.frag_read_config())
            .await?;

        if fragment_read_task.with_deleted_rows {
            fragment_reader.with_make_deletions_null();
        }

        // The reader expects sorted ranges and it may be possible to get non-sorted ranges if
        // the row ids are not contiguous
        fragment_read_task.ranges.sort_by_key(|r| r.start);

        Ok(fragment_reader
            .read_ranges(
                fragment_read_task.ranges.into(),
                fragment_read_task.batch_size,
            )?
            .zip(futures::stream::repeat(fragment_read_task.filter.clone()))
            .map(|(batch_fut, filter)| Self::wrap_with_filter(batch_fut, filter)))
    }

    fn wrap_with_filter(
        batch_fut: ReadBatchFut,
        filter: Option<Arc<dyn PhysicalExpr>>,
    ) -> Result<ReadBatchFut> {
        if let Some(filter) = filter {
            Ok(batch_fut
                .map(move |batch| {
                    batch.and_then(move |batch| {
                        datafusion_physical_plan::filter::batch_filter(&batch, &filter).map_err(
                            |e| Error::Execution {
                                message: format!("Error applying filter expression to batch: {e}"),
                                location: location!(),
                            },
                        )
                    })
                })
                .boxed())
        } else {
            Ok(batch_fut)
        }
    }

    #[async_recursion]
    async fn fragments_covered_by_index_query(
        index_expr: &ScalarIndexExpr,
        dataset: &Dataset,
    ) -> Result<RoaringBitmap> {
        match index_expr {
            ScalarIndexExpr::And(lhs, rhs) => {
                Ok(Self::fragments_covered_by_index_query(lhs, dataset).await?
                    & Self::fragments_covered_by_index_query(rhs, dataset).await?)
            }
            ScalarIndexExpr::Or(lhs, rhs) => {
                Ok(Self::fragments_covered_by_index_query(lhs, dataset).await?
                    & Self::fragments_covered_by_index_query(rhs, dataset).await?)
            }
            ScalarIndexExpr::Not(expr) => {
                Self::fragments_covered_by_index_query(expr, dataset).await
            }
            ScalarIndexExpr::Query(column, _) => {
                let idx = dataset
                    .load_scalar_index_for_column(column)
                    .await?
                    .expect("Index not found even though it must have been found earlier");
                Ok(idx
                    .fragment_bitmap
                    .expect("scalar indices should always have a fragment bitmap"))
            }
        }
    }
}

/// Options for a filtered read.
#[derive(Debug, Clone)]
pub struct FilteredReadOptions {
    scan_range_before_filter: Option<Range<u64>>,
    scan_range_after_filter: Option<Range<u64>>,
    with_deleted_rows: bool,
    batch_size: Option<u32>,
    fragment_readahead: Option<usize>,
    fragments: Option<Arc<Vec<Fragment>>>,
    projection: Projection,
    filter_plan: FilterPlan,
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
        Self {
            scan_range_before_filter: None,
            scan_range_after_filter: None,
            with_deleted_rows: false,
            batch_size: None,
            fragment_readahead: None,
            fragments: None,
            projection: dataset.full_projection(),
            filter_plan: FilterPlan::empty(),
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
    pub fn with_deleted_rows(mut self) -> Self {
        self.with_deleted_rows = true;
        self
    }

    /// Specify the range of rows to read before applying the filter.
    ///
    /// This can be used to pushdown a limit/offset when there is no filter.
    ///
    /// It's also possible to specify this when there is a filter, in order to only scan
    /// a subset of the data (and apply the filter on this subset).  For example, if the
    /// data as a column `count` that steps from 0 to 1000 and the filter is `count > 200`
    /// and the range is 100..300, then scan will read rows 100..300 and return rows 200..300
    pub fn with_scan_range_before_filter(mut self, scan_range: Range<u64>) -> Self {
        self.scan_range_before_filter = Some(scan_range);
        self
    }

    /// The range of rows to read after applying the filter.
    ///
    /// In many cases we are not able to push this down and the range will be applied after-the-fact.
    ///
    /// However, if there is a scalar index on the column, and that scalar index returns an exact
    /// match, then we can use this to skip reading the data entirely.
    ///
    /// We currently do not support setting this when there is more than one partition.
    pub fn with_scan_range_after_filter(mut self, scan_range: Range<u64>) -> Self {
        self.scan_range_after_filter = Some(scan_range);
        self
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
    /// This will be used to filter the rows after they are read.
    ///
    /// The filter plan is always exactly applied, via a recheck step, even when there is no
    /// scalar index or the scalar index returns an AtMost / AtLeast result.
    pub fn with_filter_plan(mut self, filter_plan: FilterPlan) -> Self {
        self.filter_plan = filter_plan;
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
    // When execute is first called we will initialize the FilteredReadStream.  In order to support
    // multiple partitions, each partition will share the stream.
    running_stream: Arc<AsyncMutex<Option<FilteredReadStream>>>,
}

impl FilteredReadExec {
    pub fn try_new(dataset: Arc<Dataset>, mut options: FilteredReadOptions) -> Result<Self> {
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
        let output_schema = Arc::new(options.projection.to_arrow_schema()?);

        let properties = PlanProperties::new(
            EquivalenceProperties::new(output_schema),
            Partitioning::RoundRobinBatch(1),
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
        })
    }

    fn obtain_stream(&self, partition: usize) -> SendableRecordBatchStream {
        // There are two subtleties here:
        //
        // First, we need to defer execution until first polled (hence the once/flatten)
        //
        // Second, multiple partitions all share the same underlying task stream (see get_stream)
        let running_stream_lock = self.running_stream.clone();
        let dataset = self.dataset.clone();
        let options = self.options.clone();
        let metrics = self.metrics.clone();

        let stream = futures::stream::once(async move {
            let mut running_stream = running_stream_lock.lock().await;
            if let Some(running_stream) = &*running_stream {
                DataFusionResult::<SendableRecordBatchStream>::Ok(
                    running_stream.get_stream(&metrics, partition),
                )
            } else {
                let new_running_stream =
                    FilteredReadStream::try_new(dataset, options, &metrics).await?;
                let first_stream = new_running_stream.get_stream(&metrics, partition);
                *running_stream = Some(new_running_stream);
                DataFusionResult::Ok(first_stream)
            }
        })
        .try_flatten();

        Box::pin(RecordBatchStreamAdapter::new(self.schema(), stream))
    }
}

impl DisplayAs for FilteredReadExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let columns = self
                    .options
                    .projection
                    .to_schema()
                    .fields
                    .iter()
                    .map(|f| f.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "LanceRead: uri={}, projection=[{}], num_fragments={}, range_before={:?}, range_after={:?}, row_id={}, row_addr={}, indexed_filter={}, refine_filter={}",
                    self.dataset.data_dir(),
                    columns,
                    self.options.fragments.as_ref().map(|f| f.len()).unwrap_or(self.dataset.fragments().len()),
                    self.options.scan_range_before_filter,
                    self.options.scan_range_after_filter,
                    self.options.projection.with_row_id,
                    self.options.projection.with_row_addr,
                    self.options.filter_plan.index_query.as_ref().map(|i| i.to_string()).unwrap_or("true".to_string()),
                    self.options.filter_plan.refine_expr.as_ref().map(|i| i.to_string()).unwrap_or("true".to_string()),
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
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            Err(DataFusionError::External(
                Error::Internal {
                    message: "A FilteredReadExec may never have children".to_string(),
                    location: location!(),
                }
                .into(),
            ))
        } else {
            // Clear out the running stream, just in case
            self.running_stream.blocking_lock().take();
            Ok(self)
        }
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        Ok(self.obtain_stream(partition))
    }
}

#[cfg(test)]
mod tests {
    use arrow::{
        compute::concat_batches,
        datatypes::{UInt16Type, UInt32Type, UInt64Type},
    };
    use arrow_array::{Array, UInt32Array};
    use itertools::Itertools;
    use lance_core::datatypes::OnMissing;
    use lance_datagen::{array, r#gen, BatchCount, RowCount};
    use lance_index::{
        optimize::OptimizeOptions,
        scalar::{expression::PlannerIndexExt, ScalarIndexParams},
        IndexType,
    };
    use tempfile::TempDir;

    use crate::{
        dataset::{InsertBuilder, WriteDestination, WriteMode, WriteParams},
        index::DatasetIndexInternalExt,
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

            let mut dataset = gen()
                .col("fully_indexed", array::step::<UInt32Type>())
                .col("partly_indexed", array::step::<UInt64Type>())
                .col("not_indexed", array::step::<UInt16Type>())
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

            let new_data = gen()
                .col("fully_indexed", array::step_custom::<UInt32Type>(200, 1))
                .col("partly_indexed", array::step_custom::<UInt64Type>(200, 1))
                .col("not_indexed", array::step_custom::<UInt16Type>(200, 1))
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
                .optimize_indices(&OptimizeOptions {
                    num_indices_to_merge: 1,
                    index_names: Some(vec!["fully_indexed_idx".to_string()]),
                    retrain: false,
                })
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

        async fn test_plan(&self, options: FilteredReadOptions, expected: &dyn Array) {
            let plan = FilteredReadExec::try_new(self.dataset.clone(), options).unwrap();

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
        let options = base_options.clone().with_scan_range_before_filter(25..125);
        fixture
            .test_plan(options, &u32s(vec![25..100, 250..275]))
            .await;

        // Range scan against user-specified fragments
        let options = base_options
            .clone()
            .with_fragments(fixture.frags(&[3, 2]))
            .with_scan_range_before_filter(25..125);
        fixture
            .test_plan(options, &u32s(vec![325..400, 250..275]))
            .await;

        // Range scan that goes past the end of the dataset (100 rows
        // requested, only 50 can be returned)
        let options = base_options.clone().with_scan_range_before_filter(200..300);
        fixture.test_plan(options, &u32s(vec![350..400])).await;

        // Range scan that completely misses the dataset
        let options = base_options.clone().with_scan_range_before_filter(300..400);
        fixture.test_plan(options, &u32s(vec![])).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_batch_size() {
        let fixture = TestFixture::new().await;

        // First, test with the default batch size, which is bigger than any fragment in our
        // test dataset (we have tests for larger batch sizes in python, let's avoid duplicating
        // them here)
        let base_options = FilteredReadOptions::basic_full_read(&fixture.dataset);

        let plan =
            FilteredReadExec::try_new(fixture.dataset.clone(), base_options.clone()).unwrap();

        let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        let batch_sizes = batches.iter().map(|b| b.num_rows()).collect::<Vec<_>>();
        assert_eq!(batch_sizes, vec![100, 50, 100]);

        // Now, test with a batch size that is smaller than any fragment in our
        // test dataset
        let options = base_options.with_batch_size(35);

        let plan = FilteredReadExec::try_new(fixture.dataset.clone(), options).unwrap();

        let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        let batch_sizes = batches.iter().map(|b| b.num_rows()).collect::<Vec<_>>();

        // Some batches will be smaller because we don't coalesce batches across fragments
        assert_eq!(batch_sizes, vec![35, 35, 30, 35, 15, 35, 35, 30]);
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
                    let plan = FilteredReadExec::try_new(fixture.dataset.clone(), options).unwrap();

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

        check_projection(None, vec!["fully_indexed", "partly_indexed", "not_indexed"]).await;
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
                "_rowid",
                "_rowaddr",
            ],
        )
        .await;

        // It is an error to scan an empty projection
        let options = base_options
            .clone()
            .with_projection(fixture.dataset.empty_projection());
        let Err(Error::InvalidInput { source, .. }) =
            FilteredReadExec::try_new(fixture.dataset.clone(), options)
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
            .with_filter_plan(filter_plan);
        fixture
            .test_plan(options, &u32s(vec![75..100, 250..275]))
            .await;

        // Filter removes all rows specified by the scan range
        let filter_plan = fixture.filter_plan("not_indexed >= 75", false).await;
        let options = base_options
            .clone()
            .with_scan_range_before_filter(25..50)
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
                .with_filter_plan(filter_plan);
            fixture
                .test_plan(options, &u32s(vec![25..100, 250..270]))
                .await;
        }
    }

    #[test_log::test(tokio::test)]
    async fn test_with_deleted_rows() {
        let fixture = Arc::new(TestFixture::new().await);

        let base_options = FilteredReadOptions::basic_full_read(&fixture.dataset);

        // Basic full scan
        fixture
            .test_plan(
                base_options.clone().with_deleted_rows(),
                &u32s(vec![0..100, 200..400]),
            )
            .await;

        // With before filter scan range
        fixture
            .test_plan(
                base_options
                    .clone()
                    .with_deleted_rows()
                    .with_scan_range_before_filter(25..125),
                &u32s(vec![25..100, 200..225]),
            )
            .await;

        // With only row id
        let options = base_options
            .clone()
            .with_deleted_rows()
            .with_projection(fixture.dataset.empty_projection().with_row_id());
        let plan = FilteredReadExec::try_new(fixture.dataset.clone(), options).unwrap();
        let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        let num_rows = stream
            .map_ok(|batch| batch.num_rows())
            .try_fold(0, |acc, val| std::future::ready(Ok(acc + val)))
            .await
            .unwrap();
        assert_eq!(num_rows, 300);
    }
}
