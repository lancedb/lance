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

//! Table maintenance for optimizing table layout.
//!
//! As a table is updated, it's layout can become suboptimal. For example, if
//! a series of small streaming appends are performed, eventually there will be
//! a large number of small files. This imposes an overhead to track the large
//! number of files and for very small files can make it harder to read data
//! efficiently. In this case, files can be compacted into fewer larger files.
//!
//! To compact files in a table, use the [compact_files] method. This currently
//! can compact in two cases:
//!
//! 1. If a fragment has fewer rows than the target number of rows per fragment.
//!    The fragment must also have neighbors that are also candidates for
//!    compaction.
//! 2. If a fragment has a higher percentage of deleted rows than the provided
//!    threshold.
//!
//! In addition to the rules above there may be restrictions due to indexes.
//! When a fragment is compacted its row ids change and any index that contained
//! that fragment will be remapped.  However, we cannot combine indexed fragments
//! with unindexed fragments.
//!
//! ```rust
//! # use std::sync::Arc;
//! # use tokio::runtime::Runtime;
//! # use arrow_array::{RecordBatch, RecordBatchIterator, Int64Array};
//! # use arrow_schema::{Schema, Field, DataType};
//! use lance::{dataset::WriteParams, Dataset, dataset::optimize::compact_files};
//! // Remapping indices is ignored in this example.
//! use lance::dataset::optimize::IgnoreRemap;
//!
//! # let mut rt = Runtime::new().unwrap();
//! # rt.block_on(async {
//! #
//! # let test_dir = tempfile::tempdir().unwrap();
//! # let uri = test_dir.path().to_str().unwrap().to_string();
//! let schema = Arc::new(Schema::new(vec![Field::new("test", DataType::Int64, false)]));
//! let data = RecordBatch::try_new(
//!     schema.clone(),
//!     vec![Arc::new(Int64Array::from_iter_values(0..10_000))]
//! ).unwrap();
//! let reader = RecordBatchIterator::new(vec![Ok(data)], schema);
//!
//! // Write 100 small files
//! let write_params = WriteParams { max_rows_per_file: 100, ..Default::default()};
//! let mut dataset = Dataset::write(reader, &uri, Some(write_params)).await.unwrap();
//! assert_eq!(dataset.get_fragments().len(), 100);
//!
//! // Use compact_files() to consolidate the data to 1 fragment
//! let metrics = compact_files(&mut dataset, Default::default(), None).await.unwrap();
//! assert_eq!(metrics.fragments_removed, 100);
//! assert_eq!(metrics.fragments_added, 1);
//! assert_eq!(dataset.get_fragments().len(), 1);
//! # })
//! ```
//!
//! ## Distributed execution
//!
//! The [compact_files] method internally can use multiple threads, but
//! sometimes you might want to run it across multiple machines. To do this,
//! use the task API.
//!
//! ```text
//!                                      ┌──► CompactionTask.execute() ─► RewriteResult ─┐
//! plan_compaction() ─► CompactionPlan ─┼──► CompactionTask.execute() ─► RewriteResult ─┼─► commit_compaction()
//!                                      └──► CompactionTask.execute() ─► RewriteResult ─┘
//! ```
//!
//! [plan_compaction()] produces a [CompactionPlan]. This can be split into multiple
//! [CompactionTask], which can be serialized and sent to other machines. Calling
//! [CompactionTask::execute()] performs the compaction and returns a [RewriteResult].
//! The [RewriteResult] can be sent back to the coordinator, which can then call
//! [commit_compaction()] to commit the changes to the dataset.
//!
//! It's not required that all tasks are passed to [commit_compaction]. If some
//! didn't complete successfully or before a deadline, they can be omitted and
//! the successful tasks can be committed. You can also commit in batches if
//! you wish. As long as the tasks don't rewrite any of the same fragments,
//! they can be committed in any order.
use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::{AddAssign, Range};
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::{StreamExt, TryStreamExt};
use roaring::{RoaringBitmap, RoaringTreemap};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::format::RowAddress;
use crate::io::commit::{commit_transaction, migrate_fragments};
use crate::Result;
use crate::{format::Fragment, index::DatasetIndexExt, Dataset};

use super::fragment::FileFragment;
use super::index::DatasetIndexRemapperOptions;
use super::transaction::{Operation, RewriteGroup, RewrittenIndex, Transaction};
use super::utils::make_rowid_capture_stream;
use super::{write_fragments_internal, WriteMode, WriteParams};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemappedIndex {
    original: Uuid,
    new: Uuid,
}

impl RemappedIndex {
    pub fn new(original: Uuid, new: Uuid) -> Self {
        Self { original, new }
    }
}

/// When compaction runs the row ids will change.  This typically means that
/// indices will need to be remapped.  The details of how this happens are not
/// a part of the compaction process and so a trait is defined here to allow
/// for inversion of control.
#[async_trait]
pub trait IndexRemapper: Send + Sync {
    async fn remap_indices(
        &self,
        index_map: HashMap<u64, Option<u64>>,
        affected_fragment_ids: &[u64],
    ) -> Result<Vec<RemappedIndex>>;
}

/// Options for creating an [IndexRemapper]
///
/// Currently we don't have any options but we may need options in the future and so we
/// want to keep a placeholder
pub trait IndexRemapperOptions: Send + Sync {
    fn create_remapper(&self, dataset: &Dataset) -> Result<Box<dyn IndexRemapper>>;
}

/// Options to be passed to [compact_files].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionOptions {
    /// Target number of rows per file. Defaults to 1 million.
    ///
    /// This is used to determine which fragments need compaction, as any
    /// fragments that have fewer rows than this value will be candidates for
    /// compaction.
    pub target_rows_per_fragment: usize,
    /// Max number of rows per group
    ///
    /// This does not affect which fragments need compaction, but does affect
    /// how they are re-written if selected.
    pub max_rows_per_group: usize,
    /// Whether to compact fragments with deletions so there are no deletions.
    /// Defaults to true.
    pub materialize_deletions: bool,
    /// The fraction of rows that need to be deleted in a fragment before
    /// materializing the deletions. Defaults to 10% (0.1). Setting to zero (or
    /// lower) will materialize deletions for all fragments with deletions.
    /// Setting above 1.0 will never materialize deletions.
    pub materialize_deletions_threshold: f32,
    /// The number of threads to use. Defaults to the number of cores.
    pub num_threads: usize,
}

impl Default for CompactionOptions {
    fn default() -> Self {
        Self {
            // Matching defaults for WriteParams
            target_rows_per_fragment: 1024 * 1024,
            max_rows_per_group: 1024,
            materialize_deletions: true,
            materialize_deletions_threshold: 0.1,
            num_threads: num_cpus::get(),
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct IgnoreRemap {}

#[async_trait]
impl IndexRemapper for IgnoreRemap {
    async fn remap_indices(
        &self,
        _: HashMap<u64, Option<u64>>,
        _: &[u64],
    ) -> Result<Vec<RemappedIndex>> {
        Ok(Vec::new())
    }
}

impl IndexRemapperOptions for IgnoreRemap {
    fn create_remapper(&self, _: &Dataset) -> Result<Box<dyn IndexRemapper>> {
        Ok(Box::new(Self {}))
    }
}

impl CompactionOptions {
    pub fn validate(&mut self) {
        // If threshold is 100%, same as turning off deletion materialization.
        if self.materialize_deletions && self.materialize_deletions_threshold >= 1.0 {
            self.materialize_deletions = false;
        }
    }
}

/// Metrics returned by [compact_files].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionMetrics {
    /// The number of fragments that have been overwritten.
    pub fragments_removed: usize,
    /// The number of new fragments that have been added.
    pub fragments_added: usize,
    /// The number of files that have been removed, including deletion files.
    pub files_removed: usize,
    /// The number of files that have been added, which is always equal to the
    /// number of fragments.
    pub files_added: usize,
}

impl AddAssign for CompactionMetrics {
    fn add_assign(&mut self, rhs: Self) {
        self.fragments_removed += rhs.fragments_removed;
        self.fragments_added += rhs.fragments_added;
        self.files_removed += rhs.files_removed;
        self.files_added += rhs.files_added;
    }
}

/// Compacts the files in the dataset without reordering them.
///
/// This does a few things:
///  * Removes deleted rows from fragments.
///  * Removes dropped columns from fragments.
///  * Merges fragments that are too small.
///
/// This method tries to preserve the insertion order of rows in the dataset.
///
/// If no compaction is needed, this method will not make a new version of the table.
pub async fn compact_files(
    dataset: &mut Dataset,
    mut options: CompactionOptions,
    remap_options: Option<Arc<dyn IndexRemapperOptions>>,
) -> Result<CompactionMetrics> {
    options.validate();

    let compaction_plan: CompactionPlan = plan_compaction(dataset, &options).await?;

    // If nothing to compact, don't make a commit.
    if compaction_plan.tasks().is_empty() {
        return Ok(CompactionMetrics::default());
    }

    let dataset_ref = &dataset.clone();

    let result_stream = futures::stream::iter(compaction_plan.tasks.into_iter())
        .map(|task| rewrite_files(Cow::Borrowed(dataset_ref), task, &options))
        .buffer_unordered(options.num_threads);

    let completed_tasks: Vec<RewriteResult> = result_stream.try_collect().await?;
    let remap_options = remap_options.unwrap_or(Arc::new(DatasetIndexRemapperOptions::default()));
    let metrics = commit_compaction(dataset, completed_tasks, remap_options).await?;

    Ok(metrics)
}

/// Information about a fragment used to decide it's fate in compaction
#[derive(Debug)]
struct FragmentMetrics {
    /// The number of original rows in the fragment
    pub physical_rows: usize,
    /// The number of rows that have been deleted
    pub num_deletions: usize,
}

impl FragmentMetrics {
    /// The fraction of rows that have been deleted
    fn deletion_percentage(&self) -> f32 {
        if self.physical_rows > 0 {
            self.num_deletions as f32 / self.physical_rows as f32
        } else {
            0.0
        }
    }

    /// The number of rows that are still in the fragment
    fn num_rows(&self) -> usize {
        self.physical_rows - self.num_deletions
    }
}

async fn collect_metrics(fragment: &FileFragment) -> Result<FragmentMetrics> {
    let physical_rows = fragment.physical_rows();
    let num_deletions = fragment.count_deletions();
    let (physical_rows, num_deletions) =
        futures::future::try_join(physical_rows, num_deletions).await?;
    Ok(FragmentMetrics {
        physical_rows,
        num_deletions,
    })
}

/// A plan for what groups of fragments to compact.
///
/// See [plan_compaction()] for more details.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompactionPlan {
    tasks: Vec<TaskData>,
    read_version: u64,
    options: CompactionOptions,
}

impl CompactionPlan {
    /// Retrieve standalone tasks that be be executed in a distributed fashion.
    pub fn compaction_tasks(&self) -> impl Iterator<Item = CompactionTask> + '_ {
        let read_version = self.read_version;
        let options = self.options.clone();
        self.tasks.iter().map(move |task| CompactionTask {
            task: task.clone(),
            read_version,
            options: options.clone(),
        })
    }

    /// The number of tasks in the plan.
    pub fn num_tasks(&self) -> usize {
        self.tasks.len()
    }

    /// The version of the dataset that was read to produce this plan.
    pub fn read_version(&self) -> u64 {
        self.read_version
    }

    /// The options used to produce this plan.
    pub fn options(&self) -> &CompactionOptions {
        &self.options
    }
}

/// A single group of fragments to compact, which is a view into the compaction
/// plan. We keep the `replace_range` indices so we can map the result of the
/// compact back to the fragments it replaces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskData {
    /// The fragments to compact.
    pub fragments: Vec<Fragment>,
}

/// A standalone task that can be serialized and sent to another machine for
/// execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionTask {
    pub task: TaskData,
    pub read_version: u64,
    options: CompactionOptions,
}

impl CompactionTask {
    /// Run the compaction task and return the result.
    ///
    /// This result should be later passed to [commit_compaction()] to commit
    /// the changes to the dataset.
    ///
    /// Note: you should pass the version of the dataset that is the same as
    /// the read version for this task (the same version from which the
    /// compaction was planned).
    pub async fn execute(&self, dataset: &Dataset) -> Result<RewriteResult> {
        let dataset = if dataset.manifest.version == self.read_version {
            Cow::Borrowed(dataset)
        } else {
            Cow::Owned(dataset.checkout_version(self.read_version).await?)
        };
        rewrite_files(dataset, self.task.clone(), &self.options).await
    }
}

impl CompactionPlan {
    fn new(read_version: u64, options: CompactionOptions) -> Self {
        Self {
            tasks: Vec::new(),
            read_version,
            options,
        }
    }

    fn extend_tasks(&mut self, tasks: impl IntoIterator<Item = TaskData>) {
        self.tasks.extend(tasks);
    }

    fn tasks(&self) -> &[TaskData] {
        &self.tasks
    }
}

#[derive(Debug, Clone)]
enum CompactionCandidacy {
    /// Compact the fragment if it has neighbors that are also candidates
    CompactWithNeighbors,
    /// Compact the fragment regardless.
    CompactItself,
}

/// Internal struct used for planning compaction.
struct CandidateBin {
    pub fragments: Vec<Fragment>,
    pub pos_range: Range<usize>,
    pub candidacy: Vec<CompactionCandidacy>,
    pub row_counts: Vec<usize>,
    pub indices: Vec<usize>,
}

impl CandidateBin {
    /// Return true if compacting these fragments wouldn't do anything.
    fn is_noop(&self) -> bool {
        if self.fragments.is_empty() {
            return true;
        }
        // If there's only one fragment, it's a noop if it's not CompactItself
        if self.fragments.len() == 1 {
            matches!(self.candidacy[0], CompactionCandidacy::CompactWithNeighbors)
        } else {
            false
        }
    }

    /// Split into one or more bins with at least `min_num_rows` in them.
    fn split_for_size(mut self, min_num_rows: usize) -> Vec<Self> {
        let mut bins = Vec::new();

        loop {
            let mut bin_len = 0;
            let mut bin_row_count = 0;
            while bin_row_count < min_num_rows && bin_len < self.row_counts.len() {
                bin_row_count += self.row_counts[bin_len];
                bin_len += 1;
            }

            // If there's enough remaining to make another worthwhile bin, then
            // push what we have as a bin.
            if self.row_counts[bin_len..].iter().sum::<usize>() >= min_num_rows {
                bins.push(Self {
                    fragments: self.fragments.drain(0..bin_len).collect(),
                    pos_range: self.pos_range.start..(self.pos_range.start + bin_len),
                    candidacy: self.candidacy.drain(0..bin_len).collect(),
                    row_counts: self.row_counts.drain(0..bin_len).collect(),
                    // By the time we are splitting for size we are done considering indices
                    indices: Vec::new(),
                });
                self.pos_range.start += bin_len;
            } else {
                // Otherwise, just push the remaining fragments into the last bin
                bins.push(self);
                break;
            }
        }

        bins
    }
}

async fn load_index_fragmaps(dataset: &Dataset) -> Result<Vec<RoaringBitmap>> {
    let indices = dataset.load_indices().await?;
    let mut index_fragmaps = Vec::with_capacity(indices.len());
    for index in indices {
        if let Some(fragment_bitmap) = index.fragment_bitmap {
            index_fragmaps.push(fragment_bitmap.clone());
        } else {
            let dataset_at_index = dataset.checkout_version(index.dataset_version).await?;
            let frags = 0..dataset_at_index.manifest.max_fragment_id;
            index_fragmaps.push(RoaringBitmap::from_sorted_iter(frags).unwrap());
        }
    }
    Ok(index_fragmaps)
}

/// Formulate a plan to compact the files in a dataset
///
/// The compaction plan will contain a list of tasks to execute. Each task
/// will contain approximately `target_rows_per_fragment` rows and will be
/// rewriting fragments that are adjacent in the dataset's fragment list. Some
/// tasks may contain a single fragment when that fragment has deletions that
/// are being materialized and doesn't have any neighbors that need to be
/// compacted.
pub async fn plan_compaction(
    dataset: &Dataset,
    options: &CompactionOptions,
) -> Result<CompactionPlan> {
    // We assume here that get_fragments is returning the fragments in a
    // meaningful order that we want to preserve.
    let mut fragment_metrics = futures::stream::iter(dataset.get_fragments())
        .map(|fragment| async move {
            match collect_metrics(&fragment).await {
                Ok(metrics) => Ok((fragment.metadata, metrics)),
                Err(e) => Err(e),
            }
        })
        .buffered(num_cpus::get() * 2);

    let index_fragmaps = load_index_fragmaps(dataset).await?;
    let indices_containing_frag = |frag_id: u32| {
        index_fragmaps
            .iter()
            .enumerate()
            .filter(|(_, bitmap)| bitmap.contains(frag_id))
            .map(|(pos, _)| pos)
            .collect::<Vec<_>>()
    };

    let mut candidate_bins: Vec<CandidateBin> = Vec::new();
    let mut current_bin: Option<CandidateBin> = None;
    let mut i = 0;

    while let Some(res) = fragment_metrics.next().await {
        let (fragment, metrics) = res?;

        let candidacy = if options.materialize_deletions
            && metrics.deletion_percentage() > options.materialize_deletions_threshold
        {
            Some(CompactionCandidacy::CompactItself)
        } else if metrics.physical_rows < options.target_rows_per_fragment {
            // Only want to compact if their are neighbors to compact such that
            // we can get a larger fragment.
            Some(CompactionCandidacy::CompactWithNeighbors)
        } else {
            // Not a candidate
            None
        };

        let indices = indices_containing_frag(fragment.id as u32);

        match (candidacy, &mut current_bin) {
            (None, None) => {} // keep searching
            (Some(candidacy), None) => {
                // Start a new bin
                current_bin = Some(CandidateBin {
                    fragments: vec![fragment],
                    pos_range: i..(i + 1),
                    candidacy: vec![candidacy],
                    row_counts: vec![metrics.num_rows()],
                    indices,
                });
            }
            (Some(candidacy), Some(bin)) => {
                // We cannot mix "indexed" and "non-indexed" fragments and so we only consider
                // the existing bin if it contains the same indices
                if bin.indices == indices {
                    // Add to current bin
                    bin.fragments.push(fragment);
                    bin.pos_range.end += 1;
                    bin.candidacy.push(candidacy);
                    bin.row_counts.push(metrics.num_rows());
                } else {
                    // Index set is different.  Complete previous bin and start new one
                    candidate_bins.push(current_bin.take().unwrap());
                    current_bin = Some(CandidateBin {
                        fragments: vec![fragment],
                        pos_range: i..(i + 1),
                        candidacy: vec![candidacy],
                        row_counts: vec![metrics.num_rows()],
                        indices,
                    });
                }
            }
            (None, Some(_)) => {
                // Bin is complete
                candidate_bins.push(current_bin.take().unwrap());
            }
        }

        i += 1;
    }

    // Flush the last bin
    if let Some(bin) = current_bin {
        candidate_bins.push(bin);
    }

    let final_bins = candidate_bins
        .into_iter()
        .filter(|bin| !bin.is_noop())
        .flat_map(|bin| bin.split_for_size(options.target_rows_per_fragment))
        .map(|bin| TaskData {
            fragments: bin.fragments,
        });

    let mut compaction_plan = CompactionPlan::new(dataset.manifest.version, options.clone());
    compaction_plan.extend_tasks(final_bins);

    Ok(compaction_plan)
}

/// The result of a single compaction task.
///
/// This should be passed to [commit_compaction()] to commit the operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RewriteResult {
    pub metrics: CompactionMetrics,
    pub new_fragments: Vec<Fragment>,
    /// The version of the dataset that was read to perform this compaction.
    pub read_version: u64,
    /// The original fragments being replaced
    pub original_fragments: Vec<Fragment>,
    pub row_id_map: HashMap<u64, Option<u64>>,
}

/// Iterator that yields row_ids that are in the given fragments but not in
/// the given row_ids iterator.
struct MissingIds<'a, I: Iterator<Item = u64>> {
    row_ids: I,
    expected_row_id: u64,
    current_fragment_idx: usize,
    last: Option<u64>,
    fragments: &'a Vec<Fragment>,
}

impl<'a, I: Iterator<Item = u64>> MissingIds<'a, I> {
    /// row_ids must be sorted in the same order in which the rows would be
    /// found by scanning fragments in the order they are presented in.
    /// fragments is not guaranteed to be sorted by id.
    fn new(row_ids: I, fragments: &'a Vec<Fragment>) -> Self {
        assert!(!fragments.is_empty());
        let first_frag = &fragments[0];
        Self {
            row_ids,
            expected_row_id: first_frag.id * RowAddress::FRAGMENT_SIZE,
            current_fragment_idx: 0,
            last: None,
            fragments,
        }
    }
}

impl<'a, I: Iterator<Item = u64>> Iterator for MissingIds<'a, I> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_fragment_idx >= self.fragments.len() {
                return None;
            }
            let val = if let Some(last) = self.last {
                self.last = None;
                last
            } else {
                // If we've exhausted row_ids but we aren't done then use 0 which
                // is guaranteed to not match because that would mean that row_ids
                // was empty and we check for that earlier.
                self.row_ids.next().unwrap_or(0)
            };

            let current_fragment = &self.fragments[self.current_fragment_idx];
            let frag = val / RowAddress::FRAGMENT_SIZE;
            let expected_row_id = self.expected_row_id;
            self.expected_row_id += 1;
            // We validate before this we should have physical rows recorded
            let current_physical_rows = current_fragment
                .physical_rows
                .expect("Fragment doesn't have physical rows recorded");
            if (self.expected_row_id % RowAddress::FRAGMENT_SIZE) == current_physical_rows as u64 {
                self.current_fragment_idx += 1;
                if self.current_fragment_idx < self.fragments.len() {
                    self.expected_row_id =
                        self.fragments[self.current_fragment_idx].id * RowAddress::FRAGMENT_SIZE;
                }
            }
            if frag != current_fragment.id {
                self.last = Some(val);
                return Some(expected_row_id);
            }
            if val != expected_row_id {
                self.last = Some(val);
                return Some(expected_row_id);
            }
        }
    }
}

fn transpose_row_ids(
    row_ids: RoaringTreemap,
    old_fragments: &Vec<Fragment>,
    new_fragments: &[Fragment],
) -> HashMap<u64, Option<u64>> {
    let new_ids = new_fragments.iter().flat_map(|frag| {
        (0..frag.physical_rows.unwrap() as u32).map(|offset| {
            Some(u64::from(RowAddress::new_from_parts(
                frag.id as u32,
                offset,
            )))
        })
    });
    // The hashmap will have an entry for each row id to map plus all rows that
    // were deleted.
    let expected_size = row_ids.len() as usize
        + old_fragments
            .iter()
            .map(|frag| {
                frag.deletion_file
                    .as_ref()
                    .and_then(|d| d.num_deleted_rows)
                    .unwrap_or(0)
            })
            .sum::<usize>();
    // We expect row ids to be unique, so we should already not get many collisions.
    // The default hasher is designed to be resistance to DoS attacks, which is
    // more than we need for this use case.
    let mut mapping: HashMap<u64, Option<u64>> = HashMap::with_capacity(expected_size);
    mapping.extend(row_ids.iter().zip(new_ids));
    MissingIds::new(row_ids.into_iter(), old_fragments).for_each(|id| {
        mapping.insert(id, None);
    });
    mapping
}

async fn reserve_fragment_ids(dataset: &Dataset, fragments: &mut Vec<Fragment>) -> Result<()> {
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::ReserveFragments {
            num_fragments: fragments.len() as u32,
        },
        None,
    );

    let manifest = commit_transaction(
        dataset,
        dataset.object_store(),
        &transaction,
        &Default::default(),
        &Default::default(),
    )
    .await?;

    // Need +1 since max_fragment_id is inclusive in this case and ranges are exclusive
    let new_max_exclusive = manifest.max_fragment_id + 1;
    let reserved_ids = (new_max_exclusive - fragments.len() as u32)..(new_max_exclusive);

    for (fragment, new_id) in fragments.iter_mut().zip(reserved_ids) {
        fragment.id = new_id as u64;
    }

    Ok(())
}

/// Rewrite the files in a single task.
///
/// This assumes that the dataset is the correct read version to be compacted.
async fn rewrite_files(
    dataset: Cow<'_, Dataset>,
    task: TaskData,
    options: &CompactionOptions,
) -> Result<RewriteResult> {
    let mut metrics = CompactionMetrics::default();

    if task.fragments.is_empty() {
        return Ok(RewriteResult {
            metrics,
            new_fragments: Vec::new(),
            read_version: dataset.manifest.version,
            original_fragments: task.fragments,
            row_id_map: HashMap::new(),
        });
    }

    let previous_writer_version = &dataset.manifest.writer_version;
    // The versions of Lance prior to when we started writing the writer version
    // sometimes wrote incorrect `Fragment.phyiscal_rows` values, so we should
    // make sure to recompute them.
    // See: https://github.com/lancedb/lance/issues/1531
    let recompute_stats = previous_writer_version.is_none();

    // It's possible the fragments are old and don't have physical rows or
    // num deletions recorded. If that's the case, we need to grab and set that
    // information.
    let fragments = migrate_fragments(dataset.as_ref(), &task.fragments, recompute_stats).await?;
    let mut scanner = dataset.scan();
    scanner.with_fragments(fragments.clone()).with_row_id();

    let data = SendableRecordBatchStream::from(scanner.try_into_stream().await?);
    let row_ids = Arc::new(RwLock::new(RoaringTreemap::new()));
    let data_no_row_ids = make_rowid_capture_stream(row_ids.clone(), data)?;

    let params = WriteParams {
        max_rows_per_file: options.target_rows_per_fragment,
        max_rows_per_group: options.max_rows_per_group,
        mode: WriteMode::Append,
        ..Default::default()
    };
    let mut new_fragments = write_fragments_internal(
        dataset.object_store.clone(),
        &dataset.base,
        dataset.schema(),
        data_no_row_ids,
        params,
    )
    .await?;

    let row_ids = Arc::try_unwrap(row_ids)
        .expect("Row ids lock still owned")
        .into_inner()
        .expect("Row ids mutex still locked");

    reserve_fragment_ids(&dataset, &mut new_fragments).await?;

    let row_id_map: HashMap<u64, Option<u64>> =
        transpose_row_ids(row_ids, &fragments, &new_fragments);

    metrics.files_removed = task
        .fragments
        .iter()
        .map(|f| f.files.len() + f.deletion_file.is_some() as usize)
        .sum();
    metrics.fragments_removed = task.fragments.len();
    metrics.fragments_added = new_fragments.len();
    metrics.files_added = new_fragments
        .iter()
        .map(|f| f.files.len() + f.deletion_file.is_some() as usize)
        .sum();

    Ok(RewriteResult {
        metrics,
        new_fragments,
        read_version: dataset.manifest.version,
        original_fragments: task.fragments,
        row_id_map,
    })
}

/// Commit the results of file compaction.
///
/// It is not required that all tasks are passed to this method. If some failed,
/// they can be omitted and the successful tasks can be committed. However, once
/// some of the tasks have been committed, the remainder of the tasks will not
/// be able to be committed and should be considered cancelled.
pub async fn commit_compaction(
    dataset: &mut Dataset,
    completed_tasks: Vec<RewriteResult>,
    options: Arc<dyn IndexRemapperOptions>,
) -> Result<CompactionMetrics> {
    if completed_tasks.is_empty() {
        return Ok(CompactionMetrics::default());
    }

    let mut rewrite_groups = Vec::with_capacity(completed_tasks.len());
    let mut metrics = CompactionMetrics::default();

    let mut row_id_map: HashMap<u64, Option<u64>> = HashMap::default();

    for task in completed_tasks {
        metrics += task.metrics;
        let rewrite_group = RewriteGroup {
            old_fragments: task.original_fragments,
            new_fragments: task.new_fragments,
        };
        row_id_map.extend(task.row_id_map);
        rewrite_groups.push(rewrite_group);
    }

    let index_remapper = options.create_remapper(dataset)?;
    let affected_ids = rewrite_groups
        .iter()
        .flat_map(|group| group.old_fragments.iter().map(|frag| frag.id))
        .collect::<Vec<_>>();

    let remapped_indices = index_remapper
        .remap_indices(row_id_map, &affected_ids)
        .await?;
    let rewritten_indices = remapped_indices
        .iter()
        .map(|rewritten| RewrittenIndex {
            old_id: rewritten.original,
            new_id: rewritten.new,
        })
        .collect();

    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::Rewrite {
            groups: rewrite_groups,
            rewritten_indices,
        },
        None,
    );

    let manifest = commit_transaction(
        dataset,
        dataset.object_store(),
        &transaction,
        &Default::default(),
        &Default::default(),
    )
    .await?;

    dataset.manifest = Arc::new(manifest);

    Ok(metrics)
}

#[cfg(test)]
mod tests {

    use arrow_array::{Float32Array, Int64Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use arrow_select::concat::concat_batches;
    use futures::TryStreamExt;
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_missing_indices() {
        // Sanity test to make sure MissingIds works.  Does not test actual functionality so
        // feel free to remove if it becomes inconvenient
        let frags = vec![
            Fragment {
                id: 0,
                files: Vec::new(),
                deletion_file: None,
                physical_rows: Some(5),
            },
            Fragment {
                id: 3,
                files: Vec::new(),
                deletion_file: None,
                physical_rows: Some(3),
            },
        ];
        let rows = [(0, 1), (0, 3), (0, 4), (3, 0), (3, 2)]
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into());

        let missing = MissingIds::new(rows, &frags).collect::<Vec<_>>();
        let expected_missing = [(0, 0), (0, 2), (3, 1)]
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into())
            .collect::<Vec<u64>>();
        assert_eq!(missing, expected_missing);
    }

    #[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
    struct IgnoreRemap {}

    #[async_trait]
    impl IndexRemapper for IgnoreRemap {
        async fn remap_indices(
            &self,
            _: HashMap<u64, Option<u64>>,
            _: &[u64],
        ) -> Result<Vec<RemappedIndex>> {
            Ok(Vec::new())
        }
    }

    impl IndexRemapperOptions for IgnoreRemap {
        fn create_remapper(&self, _: &Dataset) -> Result<Box<dyn IndexRemapper>> {
            Ok(Box::new(Self {}))
        }
    }

    #[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
    struct MockIndexRemapperExpectation {
        expected: HashMap<u64, Option<u64>>,
        answer: Vec<RemappedIndex>,
    }

    #[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
    struct MockIndexRemapper {
        expectations: Vec<MockIndexRemapperExpectation>,
    }

    impl MockIndexRemapper {
        fn stringify_map(map: &HashMap<u64, Option<u64>>) -> String {
            let mut sorted_keys = map.keys().collect::<Vec<_>>();
            sorted_keys.sort();
            let mut first_keys = sorted_keys
                .into_iter()
                .take(10)
                .map(|key| {
                    format!(
                        "{}:{:?}",
                        RowAddress::new_from_id(*key),
                        map[key].map(RowAddress::new_from_id)
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            if map.len() > 10 {
                first_keys.push_str(", ...");
            }
            let mut result_str = format!("(len={})", map.len());
            result_str.push_str(&first_keys);
            result_str
        }

        fn in_any_order(expectations: &[Self]) -> Self {
            let expectations = expectations
                .iter()
                .flat_map(|item| item.expectations.clone())
                .collect::<Vec<_>>();
            Self { expectations }
        }
    }

    #[async_trait]
    impl IndexRemapper for MockIndexRemapper {
        async fn remap_indices(
            &self,
            index_map: HashMap<u64, Option<u64>>,
            _: &[u64],
        ) -> Result<Vec<RemappedIndex>> {
            for expectation in &self.expectations {
                if expectation.expected == index_map {
                    return Ok(expectation.answer.clone());
                }
            }
            panic!(
                "Unexpected index map (len={}): {}\n  Options: {}",
                index_map.len(),
                Self::stringify_map(&index_map),
                self.expectations
                    .iter()
                    .map(|expectation| Self::stringify_map(&expectation.expected))
                    .collect::<Vec<_>>()
                    .join("\n  ")
            );
        }
    }

    impl IndexRemapperOptions for MockIndexRemapper {
        fn create_remapper(&self, _: &Dataset) -> Result<Box<dyn IndexRemapper>> {
            Ok(Box::new(self.clone()))
        }
    }

    #[test]
    fn test_candidate_bin() {
        let empty_bin = CandidateBin {
            fragments: vec![],
            pos_range: 0..0,
            candidacy: vec![],
            row_counts: vec![],
            indices: vec![],
        };
        assert!(empty_bin.is_noop());

        let fragment = Fragment {
            id: 0,
            files: vec![],
            deletion_file: None,
            physical_rows: Some(0),
        };
        let single_bin = CandidateBin {
            fragments: vec![fragment.clone()],
            pos_range: 0..1,
            candidacy: vec![CompactionCandidacy::CompactWithNeighbors],
            row_counts: vec![100],
            indices: vec![],
        };
        assert!(single_bin.is_noop());

        let single_bin = CandidateBin {
            fragments: vec![fragment.clone()],
            pos_range: 0..1,
            candidacy: vec![CompactionCandidacy::CompactItself],
            row_counts: vec![100],
            indices: vec![],
        };
        // Not a no-op because it's CompactItself
        assert!(!single_bin.is_noop());

        let big_bin = CandidateBin {
            fragments: std::iter::repeat(fragment.clone()).take(8).collect(),
            pos_range: 0..8,
            candidacy: std::iter::repeat(CompactionCandidacy::CompactItself)
                .take(8)
                .collect(),
            row_counts: vec![100, 400, 200, 200, 400, 300, 300, 100],
            indices: vec![],
            // Will group into: [[100, 400], [200, 200, 400], [300, 300, 100]]
            // with size = 500
        };
        assert!(!big_bin.is_noop());
        let split = big_bin.split_for_size(500);
        assert_eq!(split.len(), 3);
        assert_eq!(split[0].pos_range, 0..2);
        assert_eq!(split[1].pos_range, 2..5);
        assert_eq!(split[2].pos_range, 5..8);
    }

    fn sample_data() -> RecordBatch {
        let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]);

        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Int64Array::from_iter_values(0..10_000))],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_compact_empty() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        // Compact an empty table
        let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]);

        let reader = RecordBatchIterator::new(vec![].into_iter().map(Ok), Arc::new(schema));
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        let plan = plan_compaction(&dataset, &CompactionOptions::default())
            .await
            .unwrap();
        assert_eq!(plan.tasks().len(), 0);

        let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();

        assert_eq!(metrics, CompactionMetrics::default());
        assert_eq!(dataset.manifest.version, 1);
    }

    #[tokio::test]
    async fn test_compact_all_good() {
        // Compact a table with nothing to do
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let data = sample_data();
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
        // Just one file
        let write_params = WriteParams {
            max_rows_per_file: 10_000,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // There's only one file, so we can't compact any more if we wanted to.
        let plan = plan_compaction(&dataset, &CompactionOptions::default())
            .await
            .unwrap();
        assert_eq!(plan.tasks().len(), 0);

        // Now split across multiple files
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 3_000,
            max_rows_per_group: 1_000,
            mode: WriteMode::Overwrite,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 3_000,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 0);
    }

    fn row_ids(frag_idx: u32, offsets: Range<u32>) -> Range<u64> {
        let start = RowAddress::new_from_parts(frag_idx, offsets.start);
        let end = RowAddress::new_from_parts(frag_idx, offsets.end);
        start.into()..end.into()
    }

    // The outer list has one item per new fragment
    // The inner list has ranges of old row ids that map to the new fragment, in order
    fn expect_remap(
        ranges: &[Vec<(Range<u64>, bool)>],
        starting_new_frag_idx: u32,
    ) -> MockIndexRemapper {
        let mut expected_remap: HashMap<u64, Option<u64>> = HashMap::default();
        expected_remap.reserve(ranges.iter().map(|r| r.len()).sum());
        for (new_frag_offset, new_frag_ranges) in ranges.iter().enumerate() {
            let new_frag_idx = starting_new_frag_idx + new_frag_offset as u32;
            let mut row_offset = 0;
            for (old_id_range, is_found) in new_frag_ranges.iter() {
                for old_id in old_id_range.clone() {
                    if *is_found {
                        let new_id = RowAddress::new_from_parts(new_frag_idx, row_offset);
                        expected_remap.insert(old_id, Some(new_id.into()));
                        row_offset += 1;
                    } else {
                        expected_remap.insert(old_id, None);
                    }
                }
            }
        }
        MockIndexRemapper {
            expectations: vec![MockIndexRemapperExpectation {
                expected: expected_remap,
                answer: vec![],
            }],
        }
    }

    #[tokio::test]
    async fn test_compact_many() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let data = sample_data();

        // Create a table with 3 small fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 1200))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 400,
            ..Default::default()
        };
        Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Append 2 large fragments (1k rows)
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(1200, 2000))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 1000,
            mode: WriteMode::Append,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Delete 1 row from first large fragment
        dataset.delete("a = 1300").await.unwrap();

        // Delete 20% of rows from second large fragment
        dataset.delete("a >= 2400 AND a < 2600").await.unwrap();

        // Append 2 small fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(3200, 600))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 300,
            mode: WriteMode::Append,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let first_new_frag_idx = 7;
        // Predicting the remap is difficult.  One task will remap to fragments 7/8 and the other
        // will remap to fragments 9/10 but we don't know which is which and so we just allow ourselves
        // to expect both possibilities.
        let remap_a = expect_remap(
            &[
                vec![
                    // 3 small fragments are rewritten to frags 7 & 8
                    (row_ids(0, 0..400), true),
                    (row_ids(1, 0..400), true),
                    (row_ids(2, 0..200), true),
                ],
                vec![(row_ids(2, 200..400), true)],
                // frag 3 is skipped since it does not have enough missing data
                // Frags 4, 5, and 6 are rewritten to frags 9 & 10
                vec![
                    // Only 800 of the 1000 rows taken from frag 4
                    (row_ids(4, 0..200), true),
                    (row_ids(4, 200..400), false),
                    (row_ids(4, 400..1000), true),
                    // frags 5 compacted with frag 4
                    (row_ids(5, 0..200), true),
                ],
                vec![(row_ids(5, 200..300), true), (row_ids(6, 0..300), true)],
            ],
            first_new_frag_idx,
        );
        let remap_b = expect_remap(
            &[
                // Frags 4, 5, and 6 are rewritten to frags 7 & 8
                vec![
                    (row_ids(4, 0..200), true),
                    (row_ids(4, 200..400), false),
                    (row_ids(4, 400..1000), true),
                    (row_ids(5, 0..200), true),
                ],
                vec![(row_ids(5, 200..300), true), (row_ids(6, 0..300), true)],
                // 3 small fragments rewritten to frags 9 & 10
                vec![
                    (row_ids(0, 0..400), true),
                    (row_ids(1, 0..400), true),
                    (row_ids(2, 0..200), true),
                ],
                vec![(row_ids(2, 200..400), true)],
            ],
            first_new_frag_idx,
        );

        // Create compaction plan
        let options = CompactionOptions {
            target_rows_per_fragment: 1000,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 2);
        assert_eq!(plan.tasks()[0].fragments.len(), 3);
        assert_eq!(plan.tasks()[1].fragments.len(), 3);

        assert_eq!(
            plan.tasks()[0]
                .fragments
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            plan.tasks()[1]
                .fragments
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            vec![4, 5, 6]
        );

        let mock_remapper = MockIndexRemapper::in_any_order(&[remap_a, remap_b]);

        // Run compaction
        let metrics = compact_files(&mut dataset, options, Some(Arc::new(mock_remapper)))
            .await
            .unwrap();

        // Assert on metrics
        assert_eq!(metrics.fragments_removed, 6);
        assert_eq!(metrics.fragments_added, 4);
        assert_eq!(metrics.files_removed, 7); // 6 data files + 1 deletion file
        assert_eq!(metrics.files_added, 4);

        let fragment_ids = dataset
            .get_fragments()
            .iter()
            .map(|f| f.id())
            .collect::<Vec<_>>();
        // Fragment ids are assigned on task completion, but that isn't deterministic.
        // But we can say the old fragment id=3 should be in the middle, and all
        // the other ids should be greater than 6.
        assert_eq!(fragment_ids[2], 3);
        assert!(fragment_ids.iter().all(|id| *id > 6 || *id == 3));
        dataset.validate().await.unwrap();
    }

    #[tokio::test]
    async fn test_compact_data_files() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let data = sample_data();

        // Create a table with 2 small fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 5_000,
            max_rows_per_group: 1_000,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Add a column
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("x", DataType::Float32, false),
        ]);

        let data = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..10_000)),
                Arc::new(Float32Array::from_iter_values(
                    (0..10_000).map(|x| x as f32 * std::f32::consts::PI),
                )),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());

        dataset.merge(reader, "a", "a").await.unwrap();

        let expected_remap = expect_remap(
            &[vec![
                // 3 small fragments are rewritten entirely
                (row_ids(0, 0..5000), true),
                (row_ids(1, 0..5000), true),
            ]],
            2,
        );

        let plan = plan_compaction(
            &dataset,
            &CompactionOptions {
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(plan.tasks().len(), 1);
        assert_eq!(plan.tasks()[0].fragments.len(), 2);

        let metrics = compact_files(&mut dataset, plan.options, Some(Arc::new(expected_remap)))
            .await
            .unwrap();

        assert_eq!(metrics.files_removed, 4); // 2 fragments with 2 data files
        assert_eq!(metrics.files_added, 1); // 1 fragment with 1 data file
        assert_eq!(metrics.fragments_removed, 2);
        assert_eq!(metrics.fragments_added, 1);

        // Assert order unchanged and data is all there.
        let scanner = dataset.scan();
        let batches = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let scanned_data = concat_batches(&batches[0].schema(), &batches).unwrap();

        assert_eq!(scanned_data, data);
    }

    #[tokio::test]
    async fn test_compact_deletions() {
        // For files that have few rows, we don't want to compact just 1 since
        // that won't do anything. But if there are deletions to materialize,
        // we want to do groups of 1. This test checks that.
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let data = sample_data();

        // Create a table with 1 fragment
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 1000))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 1000,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        dataset.delete("a <= 500").await.unwrap();

        // Threshold must be satisfied
        let mut options = CompactionOptions {
            materialize_deletions_threshold: 0.8,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 0);

        // Ignore deletions if materialize_deletions is false
        options.materialize_deletions_threshold = 0.1;
        options.materialize_deletions = false;
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 0);

        // Materialize deletions if threshold is met
        options.materialize_deletions = true;
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 1);

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert_eq!(metrics.fragments_removed, 1);
        assert_eq!(metrics.files_removed, 2);
        assert_eq!(metrics.fragments_added, 1);

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert!(fragments[0].metadata.deletion_file.is_none());
    }

    #[tokio::test]
    async fn test_compact_distributed() {
        // Can run the tasks independently
        // Can provide subset of tasks to commit_compaction
        // Once committed, can't commit remaining tasks
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let data = sample_data();

        // Write dataset as 9 1k row fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 9000))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 1000,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Plan compaction with 3 tasks
        let options = CompactionOptions {
            target_rows_per_fragment: 3_000,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 3);

        let dataset_ref = &dataset;
        let mut results = futures::stream::iter(plan.compaction_tasks())
            .then(|task| async move { task.execute(dataset_ref).await.unwrap() })
            .collect::<Vec<_>>()
            .await;

        assert_eq!(results.len(), 3);

        assert_eq!(
            results[0]
                .original_fragments
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(results[0].metrics.files_removed, 3);
        assert_eq!(results[0].metrics.files_added, 1);

        // Just commit the last task
        commit_compaction(
            &mut dataset,
            vec![results.pop().unwrap()],
            Arc::new(IgnoreRemap::default()),
        )
        .await
        .unwrap();
        // 1 commit for each task's reserve fragments plus 1 for
        // the call to commit_compaction
        assert_eq!(dataset.manifest.version, 5);

        // Can commit the remaining tasks
        commit_compaction(&mut dataset, results, Arc::new(IgnoreRemap::default()))
            .await
            .unwrap();
        // The reserve fragments call already happened for this task
        // and so we just see the bump from the commit_compaction
        assert_eq!(dataset.manifest.version, 6);
    }

    #[test]
    fn test_missing_ids() {
        // test with missing first row
        // test with missing last row
        // test fragment ids out of order

        let fragments = vec![
            Fragment {
                id: 0,
                files: Vec::new(),
                deletion_file: None,
                physical_rows: Some(5),
            },
            Fragment {
                id: 3,
                files: Vec::new(),
                deletion_file: None,
                physical_rows: Some(3),
            },
            Fragment {
                id: 1,
                files: Vec::new(),
                deletion_file: None,
                physical_rows: Some(3),
            },
        ];

        // Written as pairs of (fragment_id, offset)
        let row_ids = vec![
            (0, 1),
            (0, 3),
            (0, 4),
            (3, 0),
            (3, 2),
            (1, 0),
            (1, 1),
            (1, 2),
        ];
        let row_ids = row_ids
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into());
        let result = MissingIds::new(row_ids, &fragments).collect::<Vec<_>>();

        let expected = vec![(0, 0), (0, 2), (3, 1)];
        let expected = expected
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into())
            .collect::<Vec<u64>>();
        assert_eq!(result, expected);
    }
}
