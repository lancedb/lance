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

use futures::{StreamExt, TryStreamExt};
use object_store::path::Path;

use crate::{format::Fragment, Dataset};
use crate::{Error, Result};

use super::fragment::FileFragment;
use super::write_fragments;

#[derive(Debug, Clone)]
pub struct CompactionOptions {
    /// The target size of files after compaction. Defaults to 1GB.
    ///
    /// When fragments with multiple data files are compacted, the resulting
    /// fragments will have a single data file that is close to this size.
    target_file_size: u64,
    /// Whether to compact fragments with deletions so there are no deletions.
    /// Defaults to true.
    materialize_deletions: bool,
    /// The fraction of rows that need to be deleted in a fragment before
    /// materializing the deletions. Defaults to 10% (0.1). Setting to zero (or
    /// lower) will materialize deletions for all fragments with deletions.
    /// Setting above 1.0 will never materialize deletions.
    materialize_deletion_threshold: f32,
}

impl Default for CompactionOptions {
    fn default() -> Self {
        Self {
            target_file_size: 1024 * 1024 * 1024, // 1GB
            materialize_deletions: true,
            materialize_deletion_threshold: 0.1,
        }
    }
}

impl CompactionOptions {
    pub fn validate(&mut self) {
        // Ensure the target file size is at least 1MB.
        if self.target_file_size <= 1024 * 1024 {
            self.target_file_size = 1024 * 1024;
        }

        // If threshold is 100%, same as turning off deletion materialization.
        if self.materialize_deletions && self.materialize_deletion_threshold > 1.0 {
            self.materialize_deletions = false;
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompactionMetrics {
    fragments_removed: u32,
    fragments_added: u32,
    files_removed: u32,
    files_added: u32,
}

impl CompactionMetrics {
    pub fn new() -> Self {
        Self {
            fragments_removed: 0,
            fragments_added: 0,
            files_removed: 0,
            files_added: 0,
        }
    }
}

/// Compacts the files in the dataset without reordering them.
///
/// This does a few things:
///  * Removes deleted rows from fragments.
///  * Removed dropped columns from fragments.
///  * Merges fragments that are too small.
///
/// This method tries to preserve the insertion order of rows in the dataset.
pub async fn compact_files(
    dataset: &mut Dataset,
    mut options: CompactionOptions,
) -> Result<CompactionMetrics> {
    // First, validate the arguments.
    options.validate();

    // Then, build a plan about which fragments to compact, and in what groups

    let compaction_groups: Vec<Vec<FileFragment>> = plan_compaction(&dataset, &options).await?;

    // Finally, run a compaction job to compact the fragments. This works by:
    // Scanning the fragments in each group, and writing the rows to a new file
    // until we have reached the appropriate size. Then, we move to writing a new
    // file.
    let mut metrics = CompactionMetrics::new();

    // Once all the files are written, we collect the metadata and commit.

    // Finally, we return the metrics.
    Ok(metrics)
}

// TODO: ideally these metrics should already be in the manifest, so we don't
// have to scan during compaction.

/// Information about a fragment used to decide it's fate in compaction
struct FragmentMetrics {
    pub disk_size: u64,
    // num_rows and num_deletions skipped if not materializing deletions.
    pub num_rows: Option<usize>,
    pub num_deletions: Option<usize>,
}

impl FragmentMetrics {
    fn deletion_percentage(&self) -> f32 {
        if let Some(num_rows) = self.num_rows {
            self.num_deletions.unwrap_or_default() as f32 / num_rows as f32
        } else {
            0.0
        }
    }
}

async fn collect_metrics(fragment: &FileFragment, get_counts: bool) -> Result<FragmentMetrics> {
    // Get disk size of all data files.
    // TODO: should we be doing something other than summing?
    let object_store = fragment.dataset().object_store();
    let disk_size = futures::stream::iter(&fragment.metadata.files)
        .map(|file| async {
            let path = Path::from(file.path.clone());
            object_store.size(&path).await
        })
        .buffer_unordered(num_cpus::get() * 2)
        .map(|size| size.map(|size| size as u64))
        .try_fold(0, |acc, size| async move { Ok(acc + size) })
        .await?;

    let (num_rows, num_deletions) = if get_counts {
        let num_rows = fragment.fragment_length();
        let num_deletions = fragment.count_deletions();
        let (num_rows, num_deletions) = futures::future::try_join(num_rows, num_deletions).await?;
        (Some(num_rows), Some(num_deletions))
    } else {
        (None, None)
    };

    Ok(FragmentMetrics {
        disk_size,
        num_rows,
        num_deletions,
    })
}

async fn plan_compaction(
    dataset: &Dataset,
    options: &CompactionOptions,
) -> Result<Vec<Vec<FileFragment>>> {
    // We assume here that get_fragments is returning the fragments in a
    // meaningful order that we want to preserve.
    let mut fragment_metrics = futures::stream::iter(dataset.get_fragments())
        .map(|fragment| async move {
            match collect_metrics(&fragment, options.materialize_deletions).await {
                Ok(metrics) => Ok((fragment, metrics)),
                Err(e) => Err(e),
            }
        })
        .buffered(num_cpus::get() * 2);

    let mut groups = Vec::new();
    let mut current_group = Vec::new();

    while let Some(res) = fragment_metrics.next().await {
        let (fragment, metrics) = res?;

        // If the fragment is too small, add it to the current group.
        if metrics.disk_size < options.target_file_size {
            current_group.push(fragment);
        } else if options.materialize_deletions
            && metrics.deletion_percentage() > options.materialize_deletion_threshold
        {
            // If the fragment has deletions, and we are materializing deletions,
            // add it to the current group.
            current_group.push(fragment);
        } else {
            // Otherwise, add the current group to the list of groups, and start
            // a new group with this fragment.
            groups.push(std::mem::take(&mut current_group));
        }
    }

    // Cleanup: remove any lone files we don't have reason to compact.
    let mut to_drop = Vec::new();
    for (i, group) in groups.iter().enumerate() {
        if group.len() == 1 && group[0].metadata.deletion_file.is_none() {
            to_drop.push(i);
        }
    }
    for i in to_drop {
        groups.remove(i);
    }

    Ok(groups)
}

async fn rewrite_files(group: Vec<FileFragment>) -> Result<(CompactionMetrics, Vec<Fragment>)> {
    let mut metrics = CompactionMetrics::new();

    let dataset = group[0].dataset();
    let fragments = group
        .iter()
        .map(|fragment| fragment.metadata.clone())
        .collect();
    let scanner = dataset.scan().with_fragments(fragments);

    let data = scanner.try_into_stream().await?;

    let new_fragments = write_fragments(object_store, base_dir, schema, data, params).await?;

    metrics.files_removed = group
        .iter()
        .map(|f| f.metadata.files.len() as u32 + f.metadata.deletion_file.is_some() as u32)
        .sum();
    metrics.fragments_removed = group.len() as u32;
    metrics.fragments_added = new_fragments.len() as u32;
    metrics.files_added = new_fragments
        .iter()
        .map(|f| f.files.len() as u32 + f.deletion_file.is_some() as u32)
        .sum();

    Ok((metrics, new_fragments))
}
