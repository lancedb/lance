// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cleanup split into a plan, sweep tasks that other processes can run, and a finish.
//!
//! The plan reads every manifest once, as a single-process cleanup does, and splits
//! the listing of the Lance-managed directories into key ranges. Each sweep task lists
//! one range and removes the files there that no retained version references, using
//! the same rules as [`CleanupOperation::execute`]. The finish removes the expired
//! manifests once every task has reported.
//!
//! Each task first reads the manifests committed after the plan and keeps their
//! files. The age threshold for files no manifest references is fixed when the plan
//! is created, so a task that runs later is only more conservative.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use lance_core::{Error, Result};
use object_store::path::Path;
use serde::{Deserialize, Serialize};
use tracing::info;
use uuid::Uuid;

use super::shard::{self, KeyRange, ReferenceSlice};
use super::{
    CleanupAction, CleanupCandidateFile, CleanupExplanation, CleanupInspection, CleanupOperation,
    CleanupPolicy, CleanupReferencedBranch, CleanupRunResult, CleanupSubtree, CleanupTask,
    DEFAULT_EXPLANATION_MAX_CANDIDATE_FILES, ExpiredManifest, ReferencedFiles, RemovalStats,
    SweepEvidence, tag_is_on_branch, validate_delete_concurrency, validate_delete_rate_limit,
};
use crate::Dataset;

/// Version of the serialized [`CleanupSweepTask`] and [`CleanupSweepResult`].
const SWEEP_FORMAT: u32 = 1;
/// Tasks only run on the Lance version that planned them, so the rules a task
/// applies are always the rules its plan was built for.
const LANCE_VERSION: &str = env!("CARGO_PKG_VERSION");

impl CleanupOperation<'_> {
    /// Plan a cleanup whose listing and deletion can be split across processes.
    ///
    /// Run every task from [`CleanupPlan::tasks`], in any order and on any machine
    /// with access to the dataset, then pass all of their results to
    /// [`CleanupPlan::finish`]. Tasks are idempotent and may be retried. Nothing is
    /// deleted by the plan itself.
    ///
    /// Returns [`Error::NotSupported`] when the policy cleans referenced branches.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::cleanup::{CleanupPolicy, SweepOptions};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let plan = dataset.cleanup(CleanupPolicy::default()).plan().await?;
    /// let mut results = Vec::new();
    /// for task in plan.tasks(16)? {
    ///     // A task can be serialized and run against the same dataset elsewhere.
    ///     results.push(task.execute(dataset, &SweepOptions::default()).await?);
    /// }
    /// let stats = plan.finish(dataset, &results).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn plan(&self) -> Result<CleanupPlan> {
        CleanupPlan::try_new(self, CleanupAction::Execute).await
    }

    /// Like [`Self::plan`], but its tasks delete nothing and the plan is finished
    /// with [`CleanupPlan::finish_explain`], which matches [`Self::explain`].
    pub async fn plan_explain(&self) -> Result<CleanupPlan> {
        let action = CleanupAction::Explain {
            max_candidate_files: self.max_candidate_files,
        };
        CleanupPlan::try_new(self, action).await
    }
}

/// Options for running one [`CleanupSweepTask`].
///
/// They are not taken from the plan: each task is throttled on its own, so a
/// budget for the whole cleanup has to be divided among the tasks running at once.
#[derive(Clone, Debug, Default)]
pub struct SweepOptions {
    delete_concurrency: Option<usize>,
    delete_rate_limit: Option<u64>,
}

impl SweepOptions {
    /// Maximum in-flight deletes for this task. Defaults to the object store's I/O
    /// parallelism.
    pub fn with_delete_concurrency(mut self, concurrency: usize) -> Result<Self> {
        validate_delete_concurrency(concurrency)?;
        self.delete_concurrency = Some(concurrency);
        Ok(self)
    }

    /// Maximum delete requests per second for this task. Unlimited by default.
    pub fn with_delete_rate_limit(mut self, rate: u64) -> Result<Self> {
        validate_delete_rate_limit(rate)?;
        self.delete_rate_limit = Some(rate);
        Ok(self)
    }
}

/// A cleanup planned by [`CleanupOperation::plan`].
///
/// The plan holds the references of every retained version and is meant to stay
/// on the process that finishes it; only its tasks are sent elsewhere.
#[derive(Debug)]
pub struct CleanupPlan {
    id: Uuid,
    store_prefix: String,
    base: Path,
    branch: Option<String>,
    read_version: u64,
    action: CleanupAction,
    delete_unverified: bool,
    verification_threshold: DateTime<Utc>,
    listings: Vec<(CleanupSubtree, Option<DateTime<Utc>>)>,
    referenced: ReferenceSlice,
    verified: ReferenceSlice,
    old_manifests: HashMap<Path, ExpiredManifest>,
    retired_records: HashMap<u64, String>,
    referenced_branches: Vec<(String, u64)>,
    /// False when a ranged listing would list the whole subtree again, so each
    /// subtree gets a single task.
    is_shardable: bool,
    delete_concurrency: Option<usize>,
    delete_rate_limit: Option<u64>,
}
impl CleanupPlan {
    async fn try_new(operation: &CleanupOperation<'_>, action: CleanupAction) -> Result<Self> {
        if operation.policy.clean_referenced_branches {
            return Err(Error::not_supported(
                "a distributed cleanup cannot clean referenced branches; \
                 run cleanup_old_versions instead",
            ));
        }
        let dataset = operation.dataset;
        let task = CleanupTask::new(operation, action)?;
        let referenced_branches = task.find_referenced_branches().await?;
        let inspection = task.inspect(&referenced_branches, &HashSet::new()).await?;
        let evidence = task.prepare_sweep(inspection).await;
        let listings = evidence.listings(operation.policy.delete_unverified);
        let SweepEvidence {
            inspection,
            verification_threshold,
            ..
        } = evidence;
        let store = &dataset.object_store;
        let plan = Self {
            id: Uuid::new_v4(),
            store_prefix: store.store_prefix.clone(),
            base: dataset.base.clone(),
            branch: dataset.manifest.branch.clone(),
            read_version: task.read_version,
            action,
            delete_unverified: operation.policy.delete_unverified,
            verification_threshold,
            listings,
            referenced: ReferenceSlice::from_files(&inspection.referenced_files),
            verified: ReferenceSlice::from_files(&inspection.verified_files),
            old_manifests: inspection.old_manifests,
            retired_records: inspection.retired_records,
            referenced_branches,
            // A local filesystem lists the whole directory for every range, but
            // that is cheap there.
            is_shardable: store.list_is_lexically_ordered || task.removes_empty_dirs(),
            delete_concurrency: operation.delete_concurrency,
            delete_rate_limit: operation
                .delete_rate_limit
                .or(operation.policy.delete_rate_limit),
        };
        info!(
            plan_id = %plan.id,
            read_version = plan.read_version,
            old_versions = plan.old_manifests.len(),
            "Planned distributed cleanup"
        );
        Ok(plan)
    }

    /// Dataset version the plan was created from.
    pub fn read_version(&self) -> u64 {
        self.read_version
    }

    /// Number of manifests [`Self::finish`] removes.
    pub fn num_old_versions(&self) -> usize {
        self.old_manifests.len()
    }

    /// Split the sweep into tasks.
    ///
    /// Every listed directory gets at least one task, so there can be more than
    /// `max_tasks` of them. The split is deterministic. Returns an invalid-input
    /// error if `max_tasks` is zero.
    pub fn tasks(&self, max_tasks: usize) -> Result<Vec<CleanupSweepTask>> {
        self.tasks_with_min_keys(max_tasks, shard::KEYS_PER_TASK)
    }

    fn tasks_with_min_keys(
        &self,
        max_tasks: usize,
        keys_per_task: usize,
    ) -> Result<Vec<CleanupSweepTask>> {
        if max_tasks == 0 {
            return Err(Error::invalid_input("max_tasks must be greater than 0"));
        }
        let candidates = self
            .listings
            .iter()
            .map(|(subtree, _)| {
                shard::boundary_candidates(*subtree, &self.referenced, &self.verified)
            })
            .collect::<Vec<_>>();
        let ranges = if self.is_shardable {
            let known_keys = candidates.iter().map(Vec::len).collect::<Vec<_>>();
            shard::allocate(&known_keys, max_tasks, keys_per_task)
        } else {
            vec![1; self.listings.len()]
        };
        let shards = self
            .listings
            .iter()
            .zip(candidates)
            .zip(ranges)
            .flat_map(|(((subtree, unmodified_since), candidates), num_ranges)| {
                shard::split(
                    *subtree,
                    &self.referenced,
                    &self.verified,
                    &candidates,
                    num_ranges,
                )
                .into_iter()
                .map(move |shard| (shard, *unmodified_since))
            })
            .collect::<Vec<_>>();
        let count = u32::try_from(shards.len()).map_err(|_| {
            Error::invalid_input(format!("too many cleanup tasks: {}", shards.len()))
        })?;
        let mut expired_versions = self
            .old_manifests
            .values()
            .map(|expired| expired.version)
            .collect::<Vec<_>>();
        expired_versions.sort_unstable();
        Ok(shards
            .into_iter()
            .zip(0..)
            .map(|((shard, unmodified_since), index)| CleanupSweepTask {
                format: SWEEP_FORMAT,
                lance_version: LANCE_VERSION.to_string(),
                plan_id: self.id,
                index,
                count,
                store_prefix: self.store_prefix.clone(),
                base: self.base.to_string(),
                branch: self.branch.clone(),
                read_version: self.read_version,
                expired_versions: expired_versions.clone(),
                referenced_branches: self.referenced_branches.clone(),
                explain_max_candidate_files: self.action.candidate_file_limit(),
                delete_unverified: self.delete_unverified,
                verification_threshold: self.verification_threshold,
                subtree: shard.subtree,
                range: shard.range,
                unmodified_since,
                referenced: shard.referenced,
                verified: shard.verified,
            })
            .collect())
    }
    /// Remove the expired manifests once every task has run, returning the
    /// statistics of the whole cleanup.
    ///
    /// `results` must hold exactly one result for each task of one
    /// [`Self::tasks`] call; otherwise nothing is removed and an invalid-input error
    /// is returned. A sweep that did not finish leaves files that the expired
    /// manifests prove removable, and removing the manifests would make those files
    /// wait out the unverified-retention period instead.
    pub async fn finish(
        &self,
        dataset: &Dataset,
        results: &[CleanupSweepResult],
    ) -> Result<RemovalStats> {
        if !self.action.deletes_files() {
            return Err(Error::invalid_input(
                "this cleanup plan only explains; call finish_explain",
            ));
        }
        let mut stats = self.finish_run(dataset, results).await?.stats;
        for result in results {
            stats.merge(&result.stats);
        }
        info!(
            plan_id = %self.id,
            old_versions = stats.old_versions,
            bytes_removed = stats.bytes_removed,
            failed_deletes = stats.failed_deletes,
            "Finished distributed cleanup"
        );
        Ok(stats)
    }

    /// Like [`Self::finish`] for a plan from [`CleanupOperation::plan_explain`]:
    /// removes nothing and explains what the cleanup would remove.
    pub async fn finish_explain(
        &self,
        dataset: &Dataset,
        results: &[CleanupSweepResult],
    ) -> Result<CleanupExplanation> {
        let Some(limit) = self.action.candidate_file_limit() else {
            return Err(Error::invalid_input(
                "this cleanup plan deletes files; call finish",
            ));
        };
        let manifests = self.finish_run(dataset, results).await?;
        let mut sorted_results = results.iter().collect::<Vec<_>>();
        sorted_results.sort_by_key(|result| result.index);
        let mut run = CleanupRunResult::default();
        let mut warnings = Vec::new();
        for result in sorted_results {
            run.merge(
                CleanupRunResult {
                    stats: result.stats.clone(),
                    candidate_files: result.candidate_files.clone(),
                    candidate_files_truncated: result.candidate_files_truncated,
                    ..Default::default()
                },
                Some(limit),
            );
            warnings.extend(result.warnings.iter().cloned());
        }
        run.merge(manifests, Some(limit));
        if run.candidate_files_truncated {
            warnings.push(format!("candidate_files truncated to {limit} entries"));
        }
        Ok(CleanupExplanation {
            read_version: self.read_version,
            stats: run.stats,
            candidate_files: run.candidate_files,
            candidate_files_truncated: run.candidate_files_truncated,
            candidate_file_limit: limit,
            referenced_branches: self
                .referenced_branches
                .iter()
                .map(|(name, referenced_version)| CleanupReferencedBranch {
                    name: name.clone(),
                    referenced_version: *referenced_version,
                    cleanup_candidate: false,
                })
                .collect(),
            warnings,
        })
    }

    /// Removes, or when explaining records, the expired manifests.
    async fn finish_run(
        &self,
        dataset: &Dataset,
        results: &[CleanupSweepResult],
    ) -> Result<CleanupRunResult> {
        check_dataset(
            dataset,
            &self.store_prefix,
            self.base.as_ref(),
            &self.branch,
        )?;
        self.check_results(results)?;
        let task = cleanup_task(
            dataset,
            self.action,
            self.delete_unverified,
            self.read_version,
            self.delete_concurrency,
            self.delete_rate_limit,
        )?;
        let expired_versions = self
            .old_manifests
            .values()
            .map(|expired| expired.version)
            .collect::<HashSet<_>>();
        check_protection(dataset, &expired_versions, &self.referenced_branches).await?;
        let manifests = task.expired_manifest_files(self.old_manifests.clone());
        let outcome = task.remove_files(manifests, &self.old_manifests).await?;
        if self.action.deletes_files() {
            task.retire_records(&self.retired_records, &outcome.undeleted_manifest_versions)
                .await?;
            let index_dirs = results
                .iter()
                .flat_map(|result| &result.index_dirs_to_remove)
                .map(|dir| absolute(&self.base, dir))
                .collect::<Result<HashSet<_>>>()?;
            let retained = self
                .referenced
                .index_uuids()
                .iter()
                .cloned()
                .collect::<HashSet<_>>();
            task.remove_empty_index_dirs(&retained, index_dirs, self.verification_threshold)
                .await;
        }
        Ok(outcome.result)
    }

    /// Checks that `results` are those of one [`Self::tasks`] call and that their
    /// ranges cover every listed directory exactly once.
    fn check_results(&self, results: &[CleanupSweepResult]) -> Result<()> {
        let mismatch = |message: String| {
            Err(Error::invalid_input(format!(
                "cannot finish cleanup plan {}: {message}",
                self.id
            )))
        };
        let Some(count) = results.first().map(|result| result.count) else {
            return mismatch("no sweep results".to_string());
        };
        let mut seen = vec![false; count as usize];
        let mut ranges = HashMap::<CleanupSubtree, Vec<&KeyRange>>::new();
        for result in results {
            if result.format != SWEEP_FORMAT || result.plan_id != self.id {
                return mismatch(format!(
                    "result {} belongs to plan {} (format {})",
                    result.index, result.plan_id, result.format
                ));
            }
            let Some(is_seen) = seen.get_mut(result.index as usize) else {
                return mismatch(format!("result {} of {count} tasks", result.index));
            };
            if result.count != count || *is_seen {
                return mismatch(format!(
                    "result {} of {} tasks repeats or mixes task sets of {count}",
                    result.index, result.count
                ));
            }
            *is_seen = true;
            ranges
                .entry(result.subtree)
                .or_default()
                .push(&result.range);
        }
        if let Some(missing) = seen.iter().position(|is_seen| !is_seen) {
            return mismatch(format!("no result for task {missing} of {count}"));
        }
        for (subtree, _) in &self.listings {
            let mut subtree_ranges = ranges.remove(subtree).unwrap_or_default();
            subtree_ranges.sort_by(|a, b| a.start_after.cmp(&b.start_after));
            // Each range must start where the previous one ended, from and to an
            // open bound.
            let mut next_start = Some(None);
            for range in subtree_ranges {
                if next_start.as_ref() != Some(&range.start_after.as_ref()) {
                    return mismatch(format!("the ranges of {subtree} overlap or leave a gap"));
                }
                next_start = range.end_inclusive.as_ref().map(Some);
            }
            if next_start.is_some() {
                return mismatch(format!("the ranges of {subtree} do not cover it"));
            }
        }
        if let Some(subtree) = ranges.keys().next() {
            return mismatch(format!("{subtree} is not listed by this plan"));
        }
        Ok(())
    }
}

/// One key range of one directory to sweep, created by [`CleanupPlan::tasks`].
///
/// A task serializes with serde, but only for running it on another process with
/// the same Lance version: the encoding may change between releases, and
/// [`Self::execute`] rejects a task from another version.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CleanupSweepTask {
    format: u32,
    lance_version: String,
    plan_id: Uuid,
    index: u32,
    count: u32,
    store_prefix: String,
    base: String,
    branch: Option<String>,
    read_version: u64,
    expired_versions: Vec<u64>,
    referenced_branches: Vec<(String, u64)>,
    /// Set when the plan only explains.
    explain_max_candidate_files: Option<usize>,
    delete_unverified: bool,
    verification_threshold: DateTime<Utc>,
    subtree: CleanupSubtree,
    range: KeyRange,
    unmodified_since: Option<DateTime<Utc>>,
    referenced: ReferenceSlice,
    verified: ReferenceSlice,
}
impl CleanupSweepTask {
    /// Position of this task among the tasks of its plan.
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Number of tasks in the [`CleanupPlan::tasks`] call that created this one.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// The directory this task lists.
    pub fn subtree(&self) -> CleanupSubtree {
        self.subtree
    }

    /// Dataset-relative path after which this task's range starts; `None` from the
    /// start of the directory.
    pub fn start_after(&self) -> Option<&str> {
        self.range.start_after.as_deref()
    }

    /// Dataset-relative path at which this task's range ends; `None` to the end of
    /// the directory.
    pub fn end_inclusive(&self) -> Option<&str> {
        self.range.end_inclusive.as_deref()
    }

    /// Remove the unreferenced files in this task's range, or only record them when
    /// the plan explains.
    ///
    /// `dataset` must be the planned dataset and branch; any version of it will do.
    /// The task fails without deleting anything if a tag or a new branch has
    /// protected a version the plan expires. A file that fails to delete is counted
    /// in [`RemovalStats::failed_deletes`] and does not fail the task.
    pub async fn execute(
        &self,
        dataset: &Dataset,
        options: &SweepOptions,
    ) -> Result<CleanupSweepResult> {
        if self.format != SWEEP_FORMAT || self.lance_version != LANCE_VERSION {
            return Err(Error::invalid_input(format!(
                "cleanup sweep task {} was created by Lance {} (format {}) and cannot run on \
                 Lance {LANCE_VERSION} (format {SWEEP_FORMAT})",
                self.index, self.lance_version, self.format
            )));
        }
        check_dataset(dataset, &self.store_prefix, &self.base, &self.branch)?;
        let action = match self.explain_max_candidate_files {
            Some(max_candidate_files) => CleanupAction::Explain {
                max_candidate_files,
            },
            None => CleanupAction::Execute,
        };
        let task = cleanup_task(
            dataset,
            action,
            self.delete_unverified,
            self.read_version,
            options.delete_concurrency,
            options.delete_rate_limit,
        )?;
        info!(
            plan_id = %self.plan_id,
            task = self.index,
            subtree = %self.subtree,
            start_after = self.start_after(),
            end_inclusive = self.end_inclusive(),
            "Starting cleanup sweep task"
        );
        let expired_versions = self.expired_versions.iter().copied().collect();
        check_protection(dataset, &expired_versions, &self.referenced_branches).await?;
        let (refresh, refreshed_versions) = self.refresh(&task).await?;
        let result = match refresh {
            Some(refresh) => {
                let refresh = ReferenceSlice::from_files(&refresh);
                self.sweep(
                    &task,
                    refresh.restrict(self.subtree, &self.range),
                    refreshed_versions,
                )
                .await?
            }
            None => {
                let mut result = self.result(refreshed_versions);
                result.warnings.push(format!(
                    "skipped {}: a fragment reuse index committed after the plan cannot be \
                     interpreted",
                    self.subtree
                ));
                result
            }
        };
        info!(
            plan_id = %self.plan_id,
            task = self.index,
            subtree = %self.subtree,
            refreshed_versions = result.refreshed_versions,
            files_removed = result.stats.data_files_removed
                + result.stats.transaction_files_removed
                + result.stats.index_files_removed
                + result.stats.deletion_files_removed,
            bytes_removed = result.stats.bytes_removed,
            failed_deletes = result.stats.failed_deletes,
            "Finished cleanup sweep task"
        );
        Ok(result)
    }

    fn result(&self, refreshed_versions: u64) -> CleanupSweepResult {
        CleanupSweepResult {
            format: SWEEP_FORMAT,
            plan_id: self.plan_id,
            index: self.index,
            count: self.count,
            subtree: self.subtree,
            range: self.range.clone(),
            stats: RemovalStats::default(),
            refreshed_versions,
            index_dirs_to_remove: Vec::new(),
            candidate_files: Vec::new(),
            candidate_files_truncated: false,
            warnings: Vec::new(),
        }
    }

    /// Lists this task's range and removes what the plan's references, plus
    /// `refreshed` from the manifests committed since, do not keep.
    async fn sweep(
        &self,
        task: &CleanupTask<'_>,
        refreshed: ReferenceSlice,
        refreshed_versions: u64,
    ) -> Result<CleanupSweepResult> {
        let base = &task.dataset.base;
        let mut referenced = self.referenced.clone();
        referenced.extend(refreshed);
        let evidence = SweepEvidence {
            inspection: CleanupInspection {
                referenced_files: referenced.to_files()?,
                verified_files: self.verified.to_files()?,
                ..Default::default()
            },
            collect_frag_reuse_maps: true,
            verification_threshold: self.verification_threshold,
        };
        let bound =
            |key: &Option<String>| key.as_deref().map(|key| absolute(base, key)).transpose();
        let start_after = bound(&self.range.start_after)?;
        let end_inclusive = bound(&self.range.end_inclusive)?;
        let files = task.list_subtree(
            self.subtree,
            start_after.as_ref(),
            end_inclusive.as_ref(),
            self.unmodified_since,
            &evidence,
        );
        let outcome = task.remove_files(files, &HashMap::new()).await?;
        let mut result = self.result(refreshed_versions);
        result.stats = outcome.result.stats;
        result.candidate_files = outcome.result.candidate_files;
        result.candidate_files_truncated = outcome.result.candidate_files_truncated;
        result.index_dirs_to_remove = outcome
            .index_dirs_to_remove
            .iter()
            .map(|dir| super::remove_prefix(dir, base).to_string())
            .collect();
        result.index_dirs_to_remove.sort_unstable();
        Ok(result)
    }

    /// Reads the manifests committed after the plan, returning the files they
    /// reference and how many there were. `None` when one of them carries a fragment
    /// reuse index this client cannot interpret and this task lists `_fri/`.
    async fn refresh(&self, task: &CleanupTask<'_>) -> Result<(Option<ReferencedFiles>, u64)> {
        let dataset = task.dataset;
        let inspection = Mutex::new(CleanupInspection::default());
        let scans_managed_blobs =
            matches!(self.subtree, CleanupSubtree::Data | CleanupSubtree::Blobs);
        let locations = dataset
            .commit_handler
            .list_manifest_locations_since(&dataset.base, &dataset.object_store, self.read_version)
            .try_collect::<Vec<_>>()
            .await?;
        for location in &locations {
            let Some((manifest, indexes)) = task
                .load_manifest_for_cleanup(location, &inspection)
                .await?
            else {
                continue;
            };
            if scans_managed_blobs && manifest.has_managed_blobs() {
                let snapshot = dataset
                    .checkout_version((manifest.branch.as_deref(), Some(manifest.version)))
                    .await?;
                task.inspect_managed_blobs(&snapshot, true, &inspection)
                    .await?;
            }
            let mut guard = inspection.lock().unwrap();
            task.process_manifest(&manifest, &indexes, true, &mut guard)?;
        }
        let mut inspection = inspection.into_inner().unwrap();
        let refreshed_versions = locations.len() as u64;
        if self.subtree == CleanupSubtree::FragReuse
            && !task.resolve_frag_reuse_map_ids(&mut inspection).await
        {
            return Ok((None, refreshed_versions));
        }
        Ok((Some(inspection.referenced_files), refreshed_versions))
    }
}

/// What one [`CleanupSweepTask`] removed, to pass to [`CleanupPlan::finish`].
///
/// Serializes like [`CleanupSweepTask`], with the same version caveat.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CleanupSweepResult {
    format: u32,
    plan_id: Uuid,
    index: u32,
    count: u32,
    subtree: CleanupSubtree,
    range: KeyRange,
    stats: RemovalStats,
    /// Manifests committed after the plan that the task read.
    refreshed_versions: u64,
    /// Dataset-relative parents of removed index files, on local stores.
    index_dirs_to_remove: Vec<String>,
    candidate_files: Vec<CleanupCandidateFile>,
    candidate_files_truncated: bool,
    warnings: Vec<String>,
}

impl CleanupSweepResult {
    /// [`CleanupSweepTask::index`] of the task that produced this result.
    pub fn index(&self) -> u32 {
        self.index
    }

    /// The files this task removed, or would remove when explaining.
    pub fn stats(&self) -> &RemovalStats {
        &self.stats
    }

    /// Number of manifests committed after the plan that the task read first.
    pub fn refreshed_versions(&self) -> u64 {
        self.refreshed_versions
    }

    /// Non-fatal warnings, such as a directory the task did not sweep.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }
}

fn check_dataset(
    dataset: &Dataset,
    store_prefix: &str,
    base: &str,
    branch: &Option<String>,
) -> Result<()> {
    if dataset.object_store.store_prefix != store_prefix
        || dataset.base.as_ref() != base
        || &dataset.manifest.branch != branch
    {
        return Err(Error::invalid_input(format!(
            "the cleanup was planned for {store_prefix}/{base} on branch {branch:?}, not \
             {}/{} on branch {:?}",
            dataset.object_store.store_prefix, dataset.base, dataset.manifest.branch
        )));
    }
    Ok(())
}

/// Fails when a version the plan expires is now protected: by a tag on this
/// branch, or by a branch the plan did not know about.
async fn check_protection(
    dataset: &Dataset,
    expired_versions: &HashSet<u64>,
    referenced_branches: &[(String, u64)],
) -> Result<()> {
    let changed = |what: String| {
        Err(Error::Cleanup {
            message: format!("{what} since the cleanup plan was created; create a new plan"),
        })
    };
    let branch = dataset.manifest.branch.as_ref();
    let tags = dataset.tags().list().await?;
    if let Some((name, tag)) = tags
        .iter()
        .find(|(_, tag)| tag_is_on_branch(tag, branch) && expired_versions.contains(&tag.version))
    {
        return changed(format!(
            "tag {name:?} was created on version {}",
            tag.version
        ));
    }
    let branches = dataset.branches().list().await?;
    let children = dataset
        .branch_identifier()
        .await?
        .collect_referenced_versions(&branches);
    if let Some((name, version)) = children
        .iter()
        .find(|child| expired_versions.contains(&child.1) && !referenced_branches.contains(child))
    {
        return changed(format!(
            "branch {name:?} was created from version {version}"
        ));
    }
    Ok(())
}

/// A task that runs the cleanup phases of a plan, not of `dataset`'s version.
fn cleanup_task<'a>(
    dataset: &'a Dataset,
    action: CleanupAction,
    delete_unverified: bool,
    read_version: u64,
    delete_concurrency: Option<usize>,
    delete_rate_limit: Option<u64>,
) -> Result<CleanupTask<'a>> {
    let operation = CleanupOperation {
        dataset,
        policy: CleanupPolicy {
            delete_unverified,
            ..Default::default()
        },
        max_candidate_files: DEFAULT_EXPLANATION_MAX_CANDIDATE_FILES,
        delete_concurrency,
        delete_rate_limit,
    };
    let mut task = CleanupTask::new(&operation, action)?;
    task.read_version = read_version;
    Ok(task)
}

fn absolute(base: &Path, relative: &str) -> Result<Path> {
    let path = if base.as_ref().is_empty() {
        relative.to_string()
    } else {
        format!("{base}/{relative}")
    };
    Path::parse(&path).map_err(|error| {
        Error::invalid_input(format!(
            "invalid path {relative:?} in a cleanup sweep: {error}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::fmt::Debug;
    use std::sync::Arc;

    use super::*;
    use crate::dataset::cleanup::CleanupPolicyBuilder;
    use crate::dataset::cleanup::tests::{
        MockDatasetFixture, blob_v2_batch, managed_blob_batch, write_dummy_index_artifact,
    };
    use crate::dataset::{WriteMode, WriteParams};
    use crate::index::DatasetIndexExt;
    use crate::index::vector::VectorIndexParams;
    use crate::utils::temporal::utc_now;
    use arrow_array::RecordBatchIterator;
    use chrono::TimeDelta;
    use lance_index::IndexType;
    use lance_linalg::distance::MetricType;
    use lance_table::io::commit::RenameCommitHandler;
    use mock_instant::thread_local::MockClock;
    use rstest::rstest;
    use serde::de::DeserializeOwned;

    fn days(days: i64) -> std::time::Duration {
        TimeDelta::try_days(days).unwrap().to_std().unwrap()
    }

    fn round_trip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) -> T {
        let decoded = serde_json::from_str(&serde_json::to_string(value).unwrap()).unwrap();
        assert_eq!(&decoded, value);
        decoded
    }

    /// Runs every task, last first, through a JSON round trip.
    async fn run_tasks(
        dataset: &Dataset,
        plan: &CleanupPlan,
        max_tasks: usize,
    ) -> Result<Vec<CleanupSweepResult>> {
        let mut results = Vec::new();
        for task in plan.tasks_with_min_keys(max_tasks, 1)?.iter().rev() {
            let result = round_trip(task)
                .execute(dataset, &SweepOptions::default())
                .await?;
            results.push(round_trip(&result));
        }
        Ok(results)
    }

    async fn list_files(dataset: &Dataset) -> BTreeSet<String> {
        dataset
            .object_store
            .read_dir_all(&dataset.base, None)
            .map_ok(|meta| meta.location.to_string())
            .try_collect()
            .await
            .unwrap()
    }

    fn sorted_candidates(explanation: &CleanupExplanation) -> Vec<CleanupCandidateFile> {
        let mut files = explanation.candidate_files.clone();
        files.sort_by(|a, b| a.path.cmp(&b.path));
        files
    }

    async fn put(dataset: &Dataset, path: Path) {
        dataset.object_store.put(&path, b"orphan").await.unwrap();
    }

    #[derive(Clone, Copy, Debug)]
    enum Scenario {
        /// Versions with an index, a replaced index, deletions, a tag on an old
        /// version and orphans of every kind.
        Versions,
        BlobSidecars,
        ManagedBlobs,
        /// A branch created from a version cleanup would otherwise remove.
        Lineage,
    }

    impl Scenario {
        async fn create_index(fixture: &MockDatasetFixture, replace: bool) {
            let mut dataset = fixture.open().await.unwrap();
            let params = VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 5);
            dataset
                .create_index(
                    &["indexable"],
                    IndexType::Vector,
                    Some("some_index".to_owned()),
                    &params,
                    replace,
                )
                .await
                .unwrap();
        }

        async fn write_blobs(fixture: &MockDatasetFixture, blob_len: usize, mode: WriteMode) {
            Dataset::write(
                blob_v2_batch(blob_len),
                &fixture.dataset_path,
                Some(WriteParams {
                    store_params: Some(fixture.os_params()),
                    commit_handler: Some(Arc::new(RenameCommitHandler)),
                    mode,
                    data_storage_version: Some(lance_file::version::LanceFileVersion::V2_2),
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        }

        /// Builds the dataset and returns the policy to clean it with.
        async fn build(
            self,
            fixture: &MockDatasetFixture,
            delete_unverified: bool,
        ) -> CleanupPolicy {
            let hour = TimeDelta::try_hours(1).unwrap().to_std().unwrap();
            let mut now = std::time::Duration::ZERO;
            let mut tick = || {
                now += hour;
                MockClock::set_system_time(now);
            };
            MockClock::set_system_time(std::time::Duration::ZERO);
            let builder = CleanupPolicyBuilder::default()
                .delete_unverified(delete_unverified)
                .error_if_tagged_old_versions(false);
            match self {
                Self::Versions => {
                    fixture.create_some_data().await.unwrap();
                    tick();
                    Self::create_index(fixture, false).await;
                    tick();
                    Self::create_index(fixture, true).await;
                    tick();
                    fixture.append_some_data().await.unwrap();
                    let dataset = fixture.open().await.unwrap();
                    dataset
                        .tags()
                        .create("keep", dataset.version_id())
                        .await
                        .unwrap();
                    tick();
                    fixture.delete_data("_rowid < 20").await.unwrap();
                    tick();
                    fixture.overwrite_some_data().await.unwrap();
                    put(&dataset, dataset.data_dir().join("orphan.lance")).await;
                    put(&dataset, dataset.data_dir().join(".tmpAb3Xz9")).await;
                    put(&dataset, dataset.transactions_dir().join("0-orphan.txn")).await;
                    put(&dataset, dataset.deletions_dir().join("0-0-1.arrow")).await;
                    put(&dataset, dataset.versions_dir().join(".tmp_orphan")).await;
                    write_dummy_index_artifact(&dataset, Uuid::new_v4())
                        .await
                        .unwrap();
                    MockClock::set_system_time(days(9));
                    // Too recent to remove unless `delete_unverified` is set.
                    put(&dataset, dataset.data_dir().join("recent.lance")).await;
                    write_dummy_index_artifact(&dataset, Uuid::new_v4())
                        .await
                        .unwrap();
                }
                Self::BlobSidecars => {
                    Self::write_blobs(fixture, 100 * 1024, WriteMode::Create).await;
                    tick();
                    Self::write_blobs(fixture, 1024, WriteMode::Overwrite).await;
                    let dataset = fixture.open().await.unwrap();
                    let orphan = dataset
                        .data_dir()
                        .join("0101")
                        .join(format!("{:032b}.blob", 1));
                    put(&dataset, orphan).await;
                }
                Self::ManagedBlobs => {
                    let batch = managed_blob_batch();
                    let mut dataset = Dataset::write(
                        RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                        &fixture.dataset_path,
                        Some(WriteParams {
                            store_params: Some(fixture.os_params()),
                            data_storage_version: Some(lance_file::version::LanceFileVersion::V2_2),
                            ..Default::default()
                        }),
                    )
                    .await
                    .unwrap();
                    tick();
                    dataset.delete("id = 0").await.unwrap();
                    tick();
                    dataset.update_config([("after", "delete")]).await.unwrap();
                }
                Self::Lineage => {
                    fixture.create_some_data().await.unwrap();
                    let mut dataset = fixture.open().await.unwrap();
                    fixture
                        .create_branch_and_load(&mut dataset, "child", 1)
                        .await
                        .unwrap();
                    tick();
                    fixture.append_some_data().await.unwrap();
                    tick();
                    fixture.overwrite_some_data().await.unwrap();
                }
            }
            MockClock::set_system_time(days(10));
            builder.before_timestamp(utc_now()).build()
        }
    }
    /// Every directory left under `_indices/` still holds a file.
    fn assert_no_empty_index_dirs(dataset: &Dataset) {
        let root = std::path::Path::new("/").join(dataset.indices_dir().as_ref());
        let Ok(entries) = std::fs::read_dir(&root) else {
            return;
        };
        for entry in entries {
            let dir = entry.unwrap().path();
            assert!(std::fs::read_dir(&dir).unwrap().next().is_some(), "{dir:?}");
        }
    }

    #[rstest]
    #[tokio::test]
    async fn distributed_cleanup_matches_single_process(
        #[values(
            Scenario::Versions,
            Scenario::BlobSidecars,
            Scenario::ManagedBlobs,
            Scenario::Lineage
        )]
        scenario: Scenario,
        #[values(1, 2, 7, 64)] max_tasks: usize,
        #[values(false, true)] delete_unverified: bool,
    ) {
        let fixture = MockDatasetFixture::try_new().unwrap();
        let policy = scenario.build(&fixture, delete_unverified).await;
        let dataset = fixture.open().await.unwrap();
        let before = list_files(&dataset).await;
        let operation = dataset.cleanup(policy).with_max_candidate_files(usize::MAX);
        let expected = operation.explain().await.unwrap();
        assert!(!expected.candidate_files.is_empty());

        let plan = operation.plan_explain().await.unwrap();
        let results = run_tasks(&dataset, &plan, max_tasks).await.unwrap();
        let explained = plan.finish_explain(&dataset, &results).await.unwrap();
        assert_eq!(sorted_candidates(&explained), sorted_candidates(&expected));
        assert_eq!(explained.stats, expected.stats);
        assert_eq!(explained.referenced_branches, expected.referenced_branches);
        assert_eq!(list_files(&dataset).await, before);

        let plan = operation.plan().await.unwrap();
        assert_eq!(plan.read_version(), expected.read_version);
        let results = run_tasks(&dataset, &plan, max_tasks).await.unwrap();
        if max_tasks > plan.listings.len() {
            assert!(results.len() > plan.listings.len(), "{}", results.len());
        }
        let stats = plan.finish(&dataset, &results).await.unwrap();
        assert_eq!(stats, expected.stats);
        let removed = expected
            .candidate_files
            .iter()
            .map(|file| file.path.clone())
            .collect::<BTreeSet<_>>();
        let remaining = before
            .difference(&removed)
            .cloned()
            .collect::<BTreeSet<_>>();
        assert_eq!(list_files(&dataset).await, remaining);
        assert_no_empty_index_dirs(&dataset);
        // Nothing is left for a single-process cleanup to remove.
        assert_eq!(
            operation.explain().await.unwrap().stats,
            RemovalStats::default()
        );
        // A second finish removes nothing more.
        plan.finish(&dataset, &results).await.unwrap();
        assert_eq!(list_files(&dataset).await, remaining);
        assert_eq!(
            fixture.open().await.unwrap().version_id(),
            plan.read_version()
        );
    }
    /// The data files a task of `plan` would remove without reading the manifests
    /// committed after the plan.
    async fn unrefreshed_data_removals(dataset: &Dataset, plan: &CleanupPlan) -> u64 {
        let mut removed = 0;
        for task in plan.tasks(1).unwrap() {
            let action = CleanupAction::Explain {
                max_candidate_files: 0,
            };
            let cleanup = cleanup_task(
                dataset,
                action,
                task.delete_unverified,
                task.read_version,
                None,
                None,
            )
            .unwrap();
            let result = task
                .sweep(&cleanup, ReferenceSlice::default(), 0)
                .await
                .unwrap();
            removed += result.stats.data_files_removed;
        }
        removed
    }

    #[rstest]
    #[case::restored_expired_version(false)]
    #[case::appended_with_delete_unverified(true)]
    #[tokio::test]
    async fn sweep_keeps_files_of_versions_committed_after_the_plan(
        #[case] delete_unverified: bool,
    ) {
        MockClock::set_system_time(std::time::Duration::ZERO);
        let fixture = MockDatasetFixture::try_new().unwrap();
        fixture.create_some_data().await.unwrap();
        if !delete_unverified {
            fixture.overwrite_some_data().await.unwrap();
        }
        MockClock::set_system_time(days(10));
        let dataset = fixture.open().await.unwrap();
        let policy = CleanupPolicyBuilder::default()
            .before_timestamp(utc_now())
            .delete_unverified(delete_unverified)
            .build();
        let plan = dataset.cleanup(policy).plan().await.unwrap();

        if delete_unverified {
            fixture.append_some_data().await.unwrap();
        } else {
            // Version 1 expires, then comes back as version 3.
            assert_eq!(plan.num_old_versions(), 1);
            let mut restored = dataset.checkout_version(1).await.unwrap();
            restored.restore().await.unwrap();
        }
        // The race is real: without the refresh a data file of the new version goes.
        assert_eq!(unrefreshed_data_removals(&dataset, &plan).await, 1);

        let results = run_tasks(&dataset, &plan, 1).await.unwrap();
        assert!(results.iter().all(|result| result.refreshed_versions == 1));
        let stats = plan.finish(&dataset, &results).await.unwrap();
        assert_eq!(stats.data_files_removed, 0);
        let latest = fixture.open().await.unwrap();
        assert_eq!(latest.version_id(), plan.read_version() + 1);
        let expected_rows = if delete_unverified { 1024 } else { 512 };
        assert_eq!(latest.count_rows(None).await.unwrap(), expected_rows);
    }

    #[tokio::test]
    async fn verification_threshold_is_fixed_by_the_plan() {
        MockClock::set_system_time(std::time::Duration::ZERO);
        let fixture = MockDatasetFixture::try_new().unwrap();
        fixture.create_some_data().await.unwrap();
        let dataset = fixture.open().await.unwrap();
        MockClock::set_system_time(days(5));
        let orphan = dataset.data_dir().join("orphan.lance");
        put(&dataset, orphan.clone()).await;
        let index_uuid = Uuid::new_v4();
        write_dummy_index_artifact(&dataset, index_uuid)
            .await
            .unwrap();
        let orphans = list_files(&dataset).await;

        // Five days old when planned, twelve when the tasks run.
        MockClock::set_system_time(days(10));
        let policy = CleanupPolicyBuilder::default()
            .before_timestamp(utc_now())
            .build();
        let plan = dataset.cleanup(policy.clone()).plan().await.unwrap();
        MockClock::set_system_time(days(17));
        let results = run_tasks(&dataset, &plan, 1).await.unwrap();
        let stats = plan.finish(&dataset, &results).await.unwrap();
        assert_eq!(stats, RemovalStats::default());
        assert_eq!(list_files(&dataset).await, orphans);

        // A plan created now removes both.
        let plan = dataset.cleanup(policy).plan().await.unwrap();
        let results = run_tasks(&dataset, &plan, 1).await.unwrap();
        let stats = plan.finish(&dataset, &results).await.unwrap();
        assert_eq!(stats.data_files_removed, 1);
        assert_eq!(stats.index_files_removed, 2);
        assert!(!dataset.object_store.exists(&orphan).await.unwrap());
    }
    /// Two versions at day 0, cleaned at day 10, so version 1 expires.
    async fn two_versions() -> (MockDatasetFixture, Box<Dataset>, CleanupPolicy) {
        MockClock::set_system_time(std::time::Duration::ZERO);
        let fixture = MockDatasetFixture::try_new().unwrap();
        fixture.create_some_data().await.unwrap();
        fixture.overwrite_some_data().await.unwrap();
        MockClock::set_system_time(days(10));
        let dataset = fixture.open().await.unwrap();
        let policy = CleanupPolicyBuilder::default()
            .before_timestamp(utc_now())
            .build();
        (fixture, dataset, policy)
    }

    fn assert_plan_is_stale(error: Error) {
        assert!(matches!(error, Error::Cleanup { .. }), "{error}");
        assert!(error.to_string().contains("create a new plan"), "{error}");
    }

    #[rstest]
    #[case::tag(false)]
    #[case::branch(true)]
    #[tokio::test]
    async fn protecting_an_expired_version_after_the_plan_fails(#[case] is_branch: bool) {
        let (fixture, mut dataset, policy) = two_versions().await;
        let plan = dataset.cleanup(policy).plan().await.unwrap();
        let results = run_tasks(&dataset, &plan, 1).await.unwrap();
        let files = list_files(&dataset).await;
        if is_branch {
            fixture
                .create_branch_and_load(&mut dataset, "late", 1)
                .await
                .unwrap();
        } else {
            dataset.tags().create("late", 1).await.unwrap();
        }
        assert_plan_is_stale(plan.finish(&dataset, &results).await.unwrap_err());
        assert_plan_is_stale(run_tasks(&dataset, &plan, 1).await.unwrap_err());
        // Nothing of the main branch was removed.
        let main_files = list_files(&dataset)
            .await
            .into_iter()
            .filter(|path| !path.contains("/tree/") && !path.contains("/_refs/"))
            .collect::<BTreeSet<_>>();
        assert_eq!(main_files, files);
    }

    #[tokio::test]
    async fn plan_rejects_cleaning_referenced_branches() {
        let (_fixture, dataset, mut policy) = two_versions().await;
        policy.clean_referenced_branches = true;
        let error = dataset.cleanup(policy).plan().await.unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(error.to_string().contains("referenced branches"), "{error}");
    }

    #[tokio::test]
    async fn finish_requires_one_result_per_task() {
        let (fixture, dataset, policy) = two_versions().await;
        let plan = dataset.cleanup(policy.clone()).plan().await.unwrap();
        let results = run_tasks(&dataset, &plan, 64).await.unwrap();
        assert!(results.len() > plan.listings.len());
        let other_plan = dataset.cleanup(policy).plan().await.unwrap();
        let other_results = run_tasks(&dataset, &other_plan, 64).await.unwrap();

        let mut duplicated = results.clone();
        duplicated[1] = duplicated[0].clone();
        let mut gap = results.clone();
        let with_end = gap
            .iter_mut()
            .find(|result| result.range.end_inclusive.is_some())
            .unwrap();
        with_end.range.end_inclusive = Some("data/a".to_string());
        let rejected = [
            (Vec::new(), "no sweep results"),
            (results[1..].to_vec(), "no result for task"),
            (duplicated, "repeats"),
            (other_results, "belongs to plan"),
            (gap, "gap"),
        ];
        for (results, message) in rejected {
            let error = plan.finish(&dataset, &results).await.unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            assert!(error.to_string().contains(message), "{message}: {error}");
        }
        assert_eq!(fixture.count_files().await.unwrap().num_manifest_files, 2);

        let error = plan.finish_explain(&dataset, &results).await.unwrap_err();
        assert!(error.to_string().contains("call finish"), "{error}");
        assert_eq!(
            plan.finish(&dataset, &results).await.unwrap().old_versions,
            1
        );
    }

    #[tokio::test]
    async fn task_rejects_another_dataset_or_lance_version() {
        let (fixture, mut dataset, policy) = two_versions().await;
        let plan = dataset.cleanup(policy).plan().await.unwrap();
        let task = plan.tasks(1).unwrap().remove(0);

        let mut other_version = task.clone();
        other_version.lance_version = "0.0.1".to_string();
        let mut other_format = task.clone();
        other_format.format = SWEEP_FORMAT + 1;
        for task in [other_version, other_format] {
            let error = round_trip(&task)
                .execute(&dataset, &SweepOptions::default())
                .await
                .unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            assert!(error.to_string().contains("cannot run on"), "{error}");
        }

        let other = MockDatasetFixture::try_new().unwrap();
        other.create_some_data().await.unwrap();
        let branch = fixture
            .create_branch_and_load(&mut dataset, "branch", 2)
            .await
            .unwrap();
        for dataset in [&*other.open().await.unwrap(), &branch] {
            let error = task
                .execute(dataset, &SweepOptions::default())
                .await
                .unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            assert!(error.to_string().contains("was planned for"), "{error}");
        }
    }
    #[tokio::test]
    async fn finish_keeps_the_record_of_a_manifest_it_cannot_delete() {
        use crate::dataset::cleanup::tests::{IdentifiedStore, MockObjectStore};
        use crate::dataset::{InsertBuilder, WriteDestination};
        use lance_io::object_store::ObjectStoreParams;
        use lance_table::io::commit::CommitHandler;
        use lance_table::io::commit::external_manifest::ExternalManifestCommitHandler;

        MockClock::set_system_time(std::time::Duration::ZERO);
        let store = Arc::new(IdentifiedStore::default());
        let handler: Arc<dyn CommitHandler> = Arc::new(ExternalManifestCommitHandler {
            external_manifest_store: store.clone(),
        });
        let mock_store = Arc::new(MockObjectStore::new());
        let uri = lance_core::utils::tempfile::TempStrDir::default();
        let params = |mode| WriteParams {
            mode,
            commit_handler: Some(handler.clone()),
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(mock_store.clone()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let batch = || arrow_array::record_batch!(("i", Int32, [1, 2, 3])).unwrap();
        let mut dataset = InsertBuilder::new(uri.as_str())
            .with_params(&params(WriteMode::Create))
            .execute(vec![batch()])
            .await
            .unwrap();
        for _ in 0..2 {
            dataset = InsertBuilder::new(WriteDestination::Dataset(Arc::new(dataset)))
                .with_params(&params(WriteMode::Append))
                .execute(vec![batch()])
                .await
                .unwrap();
        }
        MockClock::set_system_time(days(10));
        let policy = CleanupPolicyBuilder::default()
            .before_timestamp(utc_now())
            .build();
        let plan = dataset.cleanup(policy).plan().await.unwrap();
        assert_eq!(plan.num_old_versions(), 2);
        let results = run_tasks(&dataset, &plan, 1).await.unwrap();
        mock_store.policy.lock().unwrap().set_before_policy(
            "block_delete_manifest",
            Arc::new(|op: &str, path: &Path| -> Result<()> {
                if op.contains("delete") && path.extension() == Some("manifest") {
                    Err(Error::internal("Delete manifest blocked".to_string()))
                } else {
                    Ok(())
                }
            }),
        );

        let stats = plan.finish(&dataset, &results).await.unwrap();
        assert_eq!(stats.failed_deletes, 2);
        assert_eq!(stats.old_versions, 0);
        let mut remaining = store
            .rows
            .lock()
            .unwrap()
            .keys()
            .copied()
            .collect::<Vec<_>>();
        remaining.sort_unstable();
        assert_eq!(remaining, vec![1, 2, 3]);
    }
}
