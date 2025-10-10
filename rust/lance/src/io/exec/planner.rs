// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use datafusion_expr::Expr;
use lance_core::datatypes::Projection;
use lance_core::utils::deletion::DeletionVector;
use lance_core::Result;
use lance_index::scalar::expression::IndexExprResult;
use lance_table::rowids::RowIdSequence;
use tracing::instrument;

use crate::dataset::fragment::FileFragment;
use crate::dataset::rowids::load_row_id_sequence;
use crate::dataset::scanner::{get_default_batch_size, BATCH_SIZE_FALLBACK};
use crate::Dataset;

use super::filtered_read::{EvaluatedIndex, FilteredReadOptions};

/// A fragment with all of its metadata loaded
pub struct LoadedFragment {
    pub row_id_sequence: Arc<RowIdSequence>,
    pub deletion_vector: Option<Arc<DeletionVector>>,
    pub fragment: FileFragment,
    /// The number of physical rows in the fragment (includes deleted rows)
    pub num_physical_rows: u64,
    /// The number of logical rows in the fragment (excludes deleted rows)
    pub num_logical_rows: u64,
}

/// A planned fragment read with all information needed for execution
#[derive(Debug, Clone)]
pub struct PlannedFragmentRead {
    pub fragment: FileFragment,
    pub ranges: Vec<Range<u64>>,
    pub projection: Arc<Projection>,
    pub with_deleted_rows: bool,
    pub batch_size: u32,
    /// An in-memory filter to apply after reading the fragment
    pub filter: Option<Expr>,
    pub priority: u32,
}

/// Public API for planning fragment scans
pub struct ScanPlanner;

impl ScanPlanner {
    /// Load fragment metadata that will be needed for planning
    #[instrument(name = "load_fragments", skip_all)]
    pub async fn load_fragments(
        dataset: &Dataset,
        fragments: Vec<lance_table::format::Fragment>,
    ) -> Result<Vec<LoadedFragment>> {
        let loaded_fragments = fragments.into_iter().map(|fragment| async move {
            let file_fragment = FileFragment::new(Arc::new(dataset.clone()), fragment.clone());

            let num_physical_rows = file_fragment.physical_rows().await? as u64;

            // Check if dataset uses stable row IDs
            let (row_id_sequence, num_logical_rows) = if dataset.manifest.uses_stable_row_ids() {
                let row_id_sequence = load_row_id_sequence(dataset, &fragment).await?;
                let num_logical_rows = row_id_sequence.len();
                (row_id_sequence, num_logical_rows)
            } else {
                // Create synthetic row ID sequence from row addresses
                let row_ids_start = fragment.id << 32;
                let row_ids_end = row_ids_start + num_physical_rows;
                let num_logical_rows = file_fragment.count_rows(None).await? as u64;
                let addrs_as_ids = Arc::new(RowIdSequence::from(row_ids_start..row_ids_end));
                (addrs_as_ids, num_logical_rows)
            };

            let deletion_vector = file_fragment.get_deletion_vector().await?;

            // Adjust num_logical_rows if there's a deletion vector and we're not using stable row IDs
            let num_logical_rows = match (&deletion_vector, dataset.manifest.uses_stable_row_ids())
            {
                (Some(dv), false) => num_physical_rows - dv.len() as u64,
                _ => num_logical_rows,
            };

            Result::Ok(LoadedFragment {
                row_id_sequence,
                deletion_vector,
                fragment: file_fragment,
                num_physical_rows,
                num_logical_rows,
            })
        });

        futures::future::try_join_all(loaded_fragments).await
    }

    /// Public API for planning fragment scans
    ///
    /// Returns: (planned fragment reads, whether limit was pushed down to fragment ranges)
    #[instrument(name = "plan_scan", skip_all)]
    pub async fn plan_scan(
        dataset: &Dataset,
        fragments: Vec<lance_table::format::Fragment>,
        evaluated_index: Option<Arc<EvaluatedIndex>>,
        options: &FilteredReadOptions,
    ) -> Result<(Vec<PlannedFragmentRead>, bool)> {
        let loaded_fragments = Self::load_fragments(dataset, fragments).await?;

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

        let default_batch_size = options.batch_size.unwrap_or_else(|| {
            get_default_batch_size().unwrap_or_else(|| {
                std::cmp::max(dataset.object_store().block_size() / 4, BATCH_SIZE_FALLBACK)
            }) as u32
        });

        let projection = Arc::new(options.projection.clone());

        let mut planned_fragments = Vec::with_capacity(loaded_fragments.len());
        for (priority, fragment) in loaded_fragments.into_iter().enumerate() {
            let fragment_id = fragment.fragment.id() as u32;
            if let Some(to_read) = fragments_to_read.get(&fragment_id) {
                if !to_read.is_empty() {
                    let filter = Self::determine_fragment_filter(
                        fragment_id,
                        &evaluated_index,
                        options,
                        scan_planned_with_limit_pushed_down,
                    );

                    log::trace!(
                        "Planning {} ranges ({} rows) from fragment {} with filter: {:?}",
                        to_read.len(),
                        to_read.iter().map(|r| r.end - r.start).sum::<u64>(),
                        fragment.fragment.id(),
                        filter
                    );

                    planned_fragments.push(PlannedFragmentRead {
                        fragment: fragment.fragment.clone(),
                        ranges: to_read.clone(),
                        projection: projection.clone(),
                        with_deleted_rows: options.with_deleted_rows,
                        batch_size: default_batch_size,
                        filter,
                        priority: priority as u32,
                    });
                }
            }
        }

        Ok((planned_fragments, scan_planned_with_limit_pushed_down))
    }

    /// Determine the appropriate filter for a fragment based on index results
    pub fn determine_fragment_filter(
        fragment_id: u32,
        evaluated_index: &Option<Arc<EvaluatedIndex>>,
        options: &FilteredReadOptions,
        limit_pushed: bool,
    ) -> Option<Expr> {
        if let Some(index) = evaluated_index {
            if index.applicable_fragments.contains(fragment_id) {
                match &index.index_result {
                    IndexExprResult::Exact(_) => options.refine_filter.clone(),
                    IndexExprResult::AtLeast(_) if limit_pushed => options.refine_filter.clone(),
                    _ => options.full_filter.clone(),
                }
            } else {
                options.full_filter.clone()
            }
        } else {
            options.full_filter.clone()
        }
    }

    /// Get the full fragment range, excluding deleted rows if deletion vector exists
    fn full_frag_range(
        num_physical_rows: u64,
        deletion_vector: &Option<Arc<DeletionVector>>,
    ) -> Vec<Range<u64>> {
        if let Some(deletion_vector) = deletion_vector {
            // Use DvToValidRanges to convert deletion vector to valid ranges
            use crate::io::exec::filtered_read::DvToValidRanges;
            DvToValidRanges::new(
                deletion_vector.to_sorted_iter().map(|pos| pos as u64),
                num_physical_rows,
            )
            .collect()
        } else {
            vec![0..num_physical_rows]
        }
    }

    /// Trim ranges to fit within a logical range
    fn trim_ranges(
        ranges: Vec<Range<u64>>,
        fragment_range: Range<u64>,
        logical_range: &Range<u64>,
    ) -> Vec<Range<u64>> {
        let mut trimmed_ranges = Vec::new();

        // Calculate the overlap between fragment_range and logical_range
        let overlap_start = fragment_range.start.max(logical_range.start);
        let overlap_end = fragment_range.end.min(logical_range.end);

        if overlap_start >= overlap_end {
            return trimmed_ranges;
        }

        // Calculate how much to skip from the beginning and take
        let skip = overlap_start.saturating_sub(fragment_range.start);
        let take = overlap_end - overlap_start;

        let mut skipped = 0;
        let mut taken = 0;

        for range in ranges {
            let range_len = range.end - range.start;

            // Skip ranges that are before our skip point
            if skipped + range_len <= skip {
                skipped += range_len;
                continue;
            }

            // We've taken enough
            if taken >= take {
                break;
            }

            // Calculate the portion of this range to take
            let start_offset = skip.saturating_sub(skipped);
            let range_start = range.start + start_offset;
            let range_take = (range_len - start_offset).min(take - taken);
            let range_end = range_start + range_take;

            trimmed_ranges.push(range_start..range_end);
            taken += range_take;
            skipped += range_len;
        }

        trimmed_ranges
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

    /// Apply skip and take to ranges in place
    fn apply_skip_take_to_ranges(
        ranges: &mut Vec<Range<u64>>,
        to_skip: &mut u64,
        to_take: &mut u64,
    ) {
        let mut new_ranges = Vec::new();

        for range in ranges.iter() {
            let range_len = range.end - range.start;

            if *to_skip >= range_len {
                // Skip entire range
                *to_skip -= range_len;
                continue;
            }

            let start = range.start + *to_skip;
            let available = range.end - start;
            let take_from_range = available.min(*to_take);
            let end = start + take_from_range;

            if start < end {
                new_ranges.push(start..end);
                *to_take -= take_from_range;
                *to_skip = 0;

                if *to_take == 0 {
                    break;
                }
            }
        }

        *ranges = new_ranges;
    }

    /// Intersect two sets of ranges
    fn intersect_ranges(ranges1: &[Range<u64>], ranges2: &[Range<u64>]) -> Vec<Range<u64>> {
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < ranges1.len() && j < ranges2.len() {
            let r1 = &ranges1[i];
            let r2 = &ranges2[j];

            let start = r1.start.max(r2.start);
            let end = r1.end.min(r2.end);

            if start < end {
                result.push(start..end);
            }

            if r1.end < r2.end {
                i += 1;
            } else {
                j += 1;
            }
        }

        result
    }
}
