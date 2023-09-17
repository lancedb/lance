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
//! ```rust
//! # use std::sync::Arc;
//! # use tokio::runtime::Runtime;
//! # use arrow_array::{RecordBatch, RecordBatchIterator, Int64Array};
//! # use arrow_schema::{Schema, Field, DataType};
//! use lance::{dataset::WriteParams, Dataset, dataset::optimize::compact_files};
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
//! let metrics = compact_files(&mut dataset, Default::default()).await.unwrap();
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
use std::ops::{AddAssign, Range};
use std::sync::Arc;

use datafusion::physical_plan::SendableRecordBatchStream;
use futures::{StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};

use crate::io::commit::commit_transaction;
use crate::Result;
use crate::{format::Fragment, Dataset};

use super::fragment::FileFragment;
use super::transaction::{Operation, RewriteGroup, Transaction};
use super::{write_fragments, WriteMode, WriteParams};

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
    let metrics = commit_compaction(dataset, completed_tasks).await?;

    Ok(metrics)
}

/// Information about a fragment used to decide it's fate in compaction
#[derive(Debug)]
struct FragmentMetrics {
    /// The number of original rows in the fragment
    pub fragment_length: usize,
    /// The number of rows that have been deleted
    pub num_deletions: usize,
}

impl FragmentMetrics {
    /// The fraction of rows that have been deleted
    fn deletion_percentage(&self) -> f32 {
        if self.fragment_length > 0 {
            self.num_deletions as f32 / self.fragment_length as f32
        } else {
            0.0
        }
    }

    /// The number of rows that are still in the fragment
    fn num_rows(&self) -> usize {
        self.fragment_length - self.num_deletions
    }
}

async fn collect_metrics(fragment: &FileFragment) -> Result<FragmentMetrics> {
    let fragment_length = fragment.fragment_length();
    let num_deletions = fragment.count_deletions();
    let (fragment_length, num_deletions) =
        futures::future::try_join(fragment_length, num_deletions).await?;
    Ok(FragmentMetrics {
        fragment_length,
        num_deletions,
    })
}

/// A plan for what groups of fragments to compact.
///
/// See [plan_compaction()] for more details.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    let mut candidate_bins: Vec<CandidateBin> = Vec::new();
    let mut current_bin: Option<CandidateBin> = None;
    let mut i = 0;

    while let Some(res) = fragment_metrics.next().await {
        let (fragment, metrics) = res?;

        let candidacy = if options.materialize_deletions
            && metrics.deletion_percentage() > options.materialize_deletions_threshold
        {
            Some(CompactionCandidacy::CompactItself)
        } else if metrics.fragment_length < options.target_rows_per_fragment {
            // Only want to compact if their are neighbors to compact such that
            // we can get a larger fragment.
            Some(CompactionCandidacy::CompactWithNeighbors)
        } else {
            // Not a candidate
            None
        };

        match (candidacy, &mut current_bin) {
            (None, None) => {} // keep searching
            (Some(candidacy), None) => {
                // Start a new bin
                current_bin = Some(CandidateBin {
                    fragments: vec![fragment],
                    pos_range: i..(i + 1),
                    candidacy: vec![candidacy],
                    row_counts: vec![metrics.num_rows()],
                });
            }
            (Some(candidacy), Some(bin)) => {
                // Add to current bin
                bin.fragments.push(fragment);
                bin.pos_range.end += 1;
                bin.candidacy.push(candidacy);
                bin.row_counts.push(metrics.num_rows());
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RewriteResult {
    pub metrics: CompactionMetrics,
    pub new_fragments: Vec<Fragment>,
    /// The version of the dataset that was read to perform this compaction.
    pub read_version: u64,
    /// The original fragments being replaced
    pub original_fragments: Vec<Fragment>,
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
        });
    }

    let fragments = task.fragments.to_vec();
    let mut scanner = dataset.scan();
    scanner.with_fragments(fragments);

    let data = SendableRecordBatchStream::from(scanner.try_into_stream().await?);

    let params = WriteParams {
        max_rows_per_file: options.target_rows_per_fragment,
        max_rows_per_group: options.max_rows_per_group,
        mode: WriteMode::Append,
        ..Default::default()
    };
    let new_fragments = write_fragments(
        dataset.object_store.clone(),
        &dataset.base,
        dataset.schema(),
        data,
        params,
    )
    .await?;

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
) -> Result<CompactionMetrics> {
    if completed_tasks.is_empty() {
        return Ok(CompactionMetrics::default());
    }

    let mut rewrite_groups = Vec::with_capacity(completed_tasks.len());
    let mut metrics = CompactionMetrics::default();

    for task in completed_tasks {
        metrics += task.metrics;
        let rewrite_group = RewriteGroup {
            old_fragments: task.original_fragments,
            new_fragments: task.new_fragments,
        };
        rewrite_groups.push(rewrite_group);
    }

    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::Rewrite {
            groups: rewrite_groups,
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
    fn test_candidate_bin() {
        let empty_bin = CandidateBin {
            fragments: vec![],
            pos_range: 0..0,
            candidacy: vec![],
            row_counts: vec![],
        };
        assert!(empty_bin.is_noop());

        let fragment = Fragment {
            id: 0,
            files: vec![],
            deletion_file: None,
            fragment_length: 0,
        };
        let single_bin = CandidateBin {
            fragments: vec![fragment.clone()],
            pos_range: 0..1,
            candidacy: vec![CompactionCandidacy::CompactWithNeighbors],
            row_counts: vec![100],
        };
        assert!(single_bin.is_noop());

        let single_bin = CandidateBin {
            fragments: vec![fragment.clone()],
            pos_range: 0..1,
            candidacy: vec![CompactionCandidacy::CompactItself],
            row_counts: vec![100],
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

        let metrics = compact_files(&mut dataset, CompactionOptions::default())
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

        // Run compaction
        let metrics = compact_files(&mut dataset, options).await.unwrap();

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

        let plan = plan_compaction(&dataset, &CompactionOptions::default())
            .await
            .unwrap();
        assert_eq!(plan.tasks().len(), 1);
        assert_eq!(plan.tasks()[0].fragments.len(), 2);

        let metrics = compact_files(&mut dataset, CompactionOptions::default())
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

        let metrics = compact_files(&mut dataset, options).await.unwrap();
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
        commit_compaction(&mut dataset, vec![results.pop().unwrap()])
            .await
            .unwrap();
        assert_eq!(dataset.manifest.version, 2);

        // Can commit the remaining tasks
        commit_compaction(&mut dataset, results).await.unwrap();
        assert_eq!(dataset.manifest.version, 3);
    }
}
