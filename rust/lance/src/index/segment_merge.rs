// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Serializable contracts for distributed index segment merges.
//!
//! A distributed merge has three actors and three artifacts:
//!
//! ```text
//! coordinator                worker(s)                    coordinator
//! plan_index_segment_merge   execute_index_merge_task     commit_index_merge_results
//!        |                          |                             |
//!   IndexMergePlan  --------->  IndexMergeTask  ---------> IndexMergeResult
//! ```
//!
//! The plan is built from one manifest read and pins everything the workers and
//! the commit need to prove that what they are about to write still describes
//! the data it was planned against: the read version, the exact source segment
//! UUIDs, where those segments live (base-aware, so an imported segment of a
//! shallow clone resolves), the coverage each one claimed, and a fingerprint of
//! the index model. Workers reopen the dataset at the pinned version and
//! revalidate before writing a byte. The commit compare-and-swaps the exact
//! source UUID set instead of re-deriving what to remove from the
//! coordinator's current coverage, which is what makes a merge planned at
//! version V safe to publish at version V+n.
//!
//! Every type carries [`INDEX_MERGE_CONTRACT_VERSION`] so a driver and its
//! executors can detect a version skew instead of misreading each other's
//! fields.
//!
//! Fragment coverage travels as a sorted `Vec<u32>` rather than a serialized
//! [`RoaringBitmap`]. These artifacts cross a language boundary (a Spark or Ray
//! driver ships them to executors as JSON), and a legible list of fragment ids
//! is worth more there than the compression, which only matters for in-memory
//! working sets.

use std::collections::HashSet;
use std::sync::Arc;

use lance_table::format::{IndexFile, IndexMetadata};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{Error, Result};

/// Version of the plan/task/result contract.
///
/// Bump this whenever a field changes meaning or a new field becomes required.
/// Both the worker and the commit reject an artifact whose version they do not
/// recognize, so a driver running an older Lance than its executors fails loudly
/// rather than merging against fields it cannot interpret.
pub const INDEX_MERGE_CONTRACT_VERSION: u32 = 1;

/// Identity of the index model every segment in one merge must share.
///
/// Derived from [`IndexMetadata`] alone, so a coordinator builds it from the
/// single manifest read it already performs. Opening each segment to compare
/// IVF centroids and quantizer parameters would turn planning into N async
/// segment opens. That deeper check belongs at the worker, where the merge
/// already parses every shard's quantizer metadata and cross-checks codebooks
/// and rotation matrices at no extra I/O.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexMergeFingerprint {
    /// `index_details.type_url`, which names the index family.
    pub index_type_url: String,
    /// On-disk format version within that family.
    pub index_version: i32,
    /// Full field declaration, keyed columns first.
    pub fields: Vec<i32>,
    /// Suffix of [`Self::fields`] the index carries but is not keyed on.
    pub covering_fields: Vec<i32>,
}

impl IndexMergeFingerprint {
    /// Read the fingerprint off a segment.
    ///
    /// Fails on a segment without index details or without exactly one keyed
    /// field, because neither can be merged: the merge dispatches on the detail
    /// type and every downstream check compares a single keyed column.
    pub fn try_from_metadata(segment: &IndexMetadata) -> Result<Self> {
        let details = segment.index_details.as_ref().ok_or_else(|| {
            Error::invalid_input(format!(
                "index segment {} of '{}' has no index details, so its type cannot be \
                 proven mergeable; rebuild the index first",
                segment.uuid, segment.name
            ))
        })?;
        if segment.keyed_field().is_none() {
            return Err(Error::invalid_input(format!(
                "index segment {} of '{}' is not keyed on a single field (fields {:?}, \
                 carried {:?}); merging requires a single keyed field",
                segment.uuid, segment.name, segment.fields, segment.covering_fields
            )));
        }
        Ok(Self {
            index_type_url: details.type_url.clone(),
            index_version: segment.index_version,
            fields: segment.fields.clone(),
            covering_fields: segment.covering_fields.clone(),
        })
    }

    /// The single column the index is keyed on.
    pub fn keyed_field(&self) -> Option<i32> {
        let keyed = self.fields.len().saturating_sub(self.covering_fields.len());
        match self.fields.get(..keyed) {
            Some([only]) => Some(*only),
            _ => None,
        }
    }

    /// Reject a segment whose model differs from this fingerprint.
    ///
    /// `context` names where the segment came from so the message points at the
    /// actor that must act: the plan, a worker's snapshot, or the commit.
    pub fn check(&self, segment: &IndexMetadata, context: &str) -> Result<()> {
        let observed = Self::try_from_metadata(segment)?;
        if observed != *self {
            return Err(Error::invalid_input(format!(
                "{context}: segment {} no longer matches the planned index model \
                 (expected {self:?}, found {observed:?}); re-plan the merge",
                segment.uuid
            )));
        }
        Ok(())
    }
}

/// One file of a merged segment.
///
/// Mirrors [`IndexFile`], which is not serializable and lives in a crate that
/// should not gain a wire format for this contract's sake.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexMergeFile {
    /// Path relative to the segment directory.
    pub path: String,
    /// Size of the file in bytes.
    pub size_bytes: u64,
}

impl From<&IndexFile> for IndexMergeFile {
    fn from(file: &IndexFile) -> Self {
        Self {
            path: file.path.clone(),
            size_bytes: file.size_bytes,
        }
    }
}

impl From<&IndexMergeFile> for IndexFile {
    fn from(file: &IndexMergeFile) -> Self {
        Self {
            path: file.path.clone(),
            size_bytes: file.size_bytes,
        }
    }
}

/// One source segment of a merge, with everything needed to find it again.
///
/// [`Self::store_prefix`] and [`Self::path`] together are the base-aware
/// location. A shallow clone keeps `base_id` on its imported segments and
/// resolves them under the base dataset's index directory, not the clone's, so
/// a task that recorded only a UUID would be non-executable for exactly the
/// segments a clone did not rewrite.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexMergeSource {
    /// Physical segment id.
    pub uuid: Uuid,
    /// Dataset version this segment was built at.
    pub dataset_version: u64,
    /// Manifest base path id, or `None` when the segment lives in this dataset.
    pub base_id: Option<u32>,
    /// Identity of the object store holding the segment.
    pub store_prefix: String,
    /// Directory holding the segment's files, within `store_prefix`.
    pub path: String,
    /// Fragments this segment claimed at the plan's read version, sorted.
    pub fragment_ids: Vec<u32>,
}

impl IndexMergeSource {
    /// Reject a segment whose recorded provenance no longer holds.
    ///
    /// Coverage is the field that moves: an indexed-field `Update` or an
    /// in-place `Merge` prunes fragments out of the segments they invalidate,
    /// so a coverage change is precisely the signal that this merge's inputs
    /// describe data that no longer exists.
    pub fn check(&self, segment: &IndexMetadata, context: &str) -> Result<()> {
        if segment.dataset_version != self.dataset_version {
            return Err(Error::invalid_input(format!(
                "{context}: segment {} was planned at dataset version {} but is now \
                 recorded at version {}; re-plan the merge",
                self.uuid, self.dataset_version, segment.dataset_version
            )));
        }
        if segment.base_id != self.base_id {
            return Err(Error::invalid_input(format!(
                "{context}: segment {} was planned under base {:?} but is now under \
                 base {:?}; re-plan the merge",
                self.uuid, self.base_id, segment.base_id
            )));
        }
        let observed = fragment_ids(segment.fragment_bitmap.as_ref());
        if observed != self.fragment_ids {
            return Err(Error::invalid_input(format!(
                "{context}: segment {} covered fragments {:?} when the merge was planned \
                 but now covers {:?}; a concurrent update pruned its coverage, so the \
                 merged output would republish stale entries. Re-plan the merge.",
                self.uuid, self.fragment_ids, observed
            )));
        }
        Ok(())
    }
}

/// One independent unit of distributed merge work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexMergeTask {
    /// Contract version, see [`INDEX_MERGE_CONTRACT_VERSION`].
    pub contract_version: u32,
    /// Plan this task belongs to.
    pub plan_id: Uuid,
    /// Position of this task within its plan.
    pub task_id: u32,
    /// Segments to merge, in merge order.
    pub sources: Vec<IndexMergeSource>,
}

impl IndexMergeTask {
    /// Union of the coverage this task's sources claimed, sorted.
    pub fn coverage(&self) -> Vec<u32> {
        let mut coverage = self
            .sources
            .iter()
            .flat_map(|source| source.fragment_ids.iter().copied())
            .collect::<Vec<_>>();
        coverage.sort_unstable();
        coverage.dedup();
        coverage
    }
}

/// A coordinator's plan for merging one logical index's segments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexMergePlan {
    /// Contract version, see [`INDEX_MERGE_CONTRACT_VERSION`].
    pub contract_version: u32,
    /// Identity of this planning round, echoed by every task and result.
    pub plan_id: Uuid,
    /// Logical index being merged.
    pub index_name: String,
    /// Dataset version the plan was built against. Workers reopen here.
    pub read_version: u64,
    /// Model every source and every output must match.
    pub fingerprint: IndexMergeFingerprint,
    /// Every segment the logical index had at [`Self::read_version`], planned or
    /// not. Recorded so a caller can tell a stale plan from a fresh one without
    /// re-reading the manifest.
    pub source_frontier: Vec<Uuid>,
    /// Union of the coverage of all planned sources, sorted.
    pub expected_coverage: Vec<u32>,
    /// Independent units of work. Their coverage sets are disjoint.
    pub tasks: Vec<IndexMergeTask>,
}

impl IndexMergePlan {
    /// Reject an artifact written against a contract this build cannot read.
    pub fn check_contract_version(contract_version: u32, context: &str) -> Result<()> {
        if contract_version != INDEX_MERGE_CONTRACT_VERSION {
            return Err(Error::invalid_input(format!(
                "{context}: index merge contract version {contract_version} is not \
                 supported by this build, which speaks version {INDEX_MERGE_CONTRACT_VERSION}"
            )));
        }
        Ok(())
    }

    /// Look up one task by id.
    pub fn task(&self, task_id: u32) -> Result<&IndexMergeTask> {
        self.tasks
            .iter()
            .find(|task| task.task_id == task_id)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "index merge plan {} has no task {} (it has {} tasks)",
                    self.plan_id,
                    task_id,
                    self.tasks.len()
                ))
            })
    }
}

/// What one worker attempt produced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexMergeOutput {
    /// Physical id of the merged segment.
    pub uuid: Uuid,
    /// Dataset version the merged segment claims.
    pub dataset_version: u64,
    /// On-disk format version of the merged segment.
    pub index_version: i32,
    /// `index_details.type_url` of the merged segment.
    pub index_type_url: String,
    /// Encoded `index_details` payload of the merged segment.
    pub index_details: Vec<u8>,
    /// Fragments the merged segment covers, sorted.
    pub fragment_ids: Vec<u32>,
    /// Files the worker wrote, relative to the segment directory.
    pub files: Vec<IndexMergeFile>,
}

/// One worker attempt's report, sufficient on its own to commit the segment.
///
/// Carries the exact source UUIDs rather than leaving the commit to re-derive
/// removals from the coordinator's current coverage, which is what let a merge
/// planned at version V silently re-add a fragment that was invalidated at V+1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexMergeResult {
    /// Contract version, see [`INDEX_MERGE_CONTRACT_VERSION`].
    pub contract_version: u32,
    /// Plan this result belongs to.
    pub plan_id: Uuid,
    /// Task this result answers.
    pub task_id: u32,
    /// Identity of this attempt. Two attempts of the same task differ here.
    pub attempt_id: Uuid,
    /// Dataset version the worker merged at, always the plan's read version.
    pub read_version: u64,
    /// Model the worker validated its sources against.
    pub fingerprint: IndexMergeFingerprint,
    /// Sources the worker actually merged, as it observed them.
    pub sources: Vec<IndexMergeSource>,
    /// The merged segment.
    pub output: IndexMergeOutput,
}

impl IndexMergeResult {
    /// Rebuild the segment metadata to commit from this report.
    pub fn output_metadata(&self, index_name: &str) -> IndexMetadata {
        IndexMetadata {
            uuid: self.output.uuid,
            name: index_name.to_owned(),
            fields: self.fingerprint.fields.clone(),
            covering_fields: self.fingerprint.covering_fields.clone(),
            dataset_version: self.output.dataset_version,
            fragment_bitmap: Some(self.output.fragment_ids.iter().copied().collect()),
            index_details: Some(Arc::new(prost_types::Any {
                type_url: self.output.index_type_url.clone(),
                value: self.output.index_details.clone(),
            })),
            index_version: self.output.index_version,
            created_at: Some(chrono::Utc::now()),
            // The merged files were written into this dataset's index directory
            // even when some sources were imported from a base.
            base_id: None,
            files: Some(self.output.files.iter().map(IndexFile::from).collect()),
        }
    }

    /// Reject a report that does not answer the task it claims to.
    pub fn check_against(&self, plan: &IndexMergePlan) -> Result<()> {
        IndexMergePlan::check_contract_version(self.contract_version, "index merge result")?;
        if self.plan_id != plan.plan_id {
            return Err(Error::invalid_input(format!(
                "index merge result for task {} belongs to plan {} but was submitted \
                 against plan {}",
                self.task_id, self.plan_id, plan.plan_id
            )));
        }
        if self.read_version != plan.read_version {
            return Err(Error::invalid_input(format!(
                "index merge result for task {} was produced at dataset version {} but \
                 the plan pinned version {}",
                self.task_id, self.read_version, plan.read_version
            )));
        }
        if self.fingerprint != plan.fingerprint {
            return Err(Error::invalid_input(format!(
                "index merge result for task {} reports index model {:?} but the plan \
                 pinned {:?}",
                self.task_id, self.fingerprint, plan.fingerprint
            )));
        }
        let task = plan.task(self.task_id)?;
        if self.sources != task.sources {
            return Err(Error::invalid_input(format!(
                "index merge result for task {} merged sources {:?} but the plan assigned {:?}",
                self.task_id,
                self.sources.iter().map(|s| s.uuid).collect::<Vec<_>>(),
                task.sources.iter().map(|s| s.uuid).collect::<Vec<_>>()
            )));
        }
        // A merged segment must claim exactly what its inputs claimed. Claiming
        // less orphans fragments from the index, and claiming more would index rows
        // this merge never read.
        let expected = task.coverage();
        if self.output.fragment_ids != expected {
            return Err(Error::invalid_input(format!(
                "index merge result for task {} covers fragments {:?} but its sources \
                 covered {:?}",
                self.task_id, self.output.fragment_ids, expected
            )));
        }
        if self.output.index_type_url != plan.fingerprint.index_type_url {
            return Err(Error::invalid_input(format!(
                "index merge result for task {} produced index type '{}' but the plan \
                 pinned '{}'",
                self.task_id, self.output.index_type_url, plan.fingerprint.index_type_url
            )));
        }
        Ok(())
    }
}

/// Sorted fragment ids of a coverage bitmap, empty when coverage is unknown.
pub(crate) fn fragment_ids(bitmap: Option<&RoaringBitmap>) -> Vec<u32> {
    bitmap
        .map(|bitmap| bitmap.iter().collect())
        .unwrap_or_default()
}

/// Keep one result per task, deterministically.
///
/// Two attempts of the same task merged the same sources at the same pinned
/// version, so their outputs are interchangeable and there is nothing to
/// choose on merit. Ordering by attempt id makes the choice reproducible from
/// the results alone: every coordinator that sees the same reports commits the
/// same segment, which is what lets a retried round be replayed. Returns the
/// winners in task order alongside the outputs that lost, whose files are
/// orphaned and can be deleted.
pub(crate) fn reconcile_attempts(
    results: Vec<IndexMergeResult>,
) -> (Vec<IndexMergeResult>, Vec<IndexMergeResult>) {
    let mut winners: Vec<IndexMergeResult> = Vec::with_capacity(results.len());
    let mut orphaned = Vec::new();
    for result in results {
        match winners
            .iter_mut()
            .find(|winner| winner.task_id == result.task_id)
        {
            None => winners.push(result),
            Some(winner) => {
                // The same attempt reported twice is a duplicate delivery, not a
                // second attempt: it names files that are still in use.
                if winner.attempt_id == result.attempt_id {
                    continue;
                }
                if result.attempt_id < winner.attempt_id {
                    orphaned.push(std::mem::replace(winner, result));
                } else {
                    orphaned.push(result);
                }
            }
        }
    }
    winners.sort_by_key(|result| result.task_id);
    (winners, orphaned)
}

/// Reject a set of results that cannot be committed as one round.
///
/// A subset of the plan's tasks is allowed: tasks are disjoint by construction,
/// so committing the merges that succeeded is safe and lets a round survive a
/// worker failure. What is never allowed is two winners covering the same
/// fragment, which would leave the index double-counting rows.
pub(crate) fn check_disjoint_outputs(results: &[IndexMergeResult]) -> Result<()> {
    let mut seen = HashSet::new();
    for result in results {
        for fragment_id in &result.output.fragment_ids {
            if !seen.insert(*fragment_id) {
                return Err(Error::invalid_input(format!(
                    "index merge results overlap on fragment {fragment_id}; tasks of one \
                     plan must cover disjoint fragments"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn result(task_id: u32, attempt_id: Uuid, output: Uuid) -> IndexMergeResult {
        IndexMergeResult {
            contract_version: INDEX_MERGE_CONTRACT_VERSION,
            plan_id: Uuid::nil(),
            task_id,
            attempt_id,
            read_version: 7,
            fingerprint: IndexMergeFingerprint {
                index_type_url: "example.VectorIndexDetails".to_owned(),
                index_version: 3,
                fields: vec![1],
                covering_fields: vec![],
            },
            sources: vec![],
            output: IndexMergeOutput {
                uuid: output,
                dataset_version: 7,
                index_version: 3,
                index_type_url: "example.VectorIndexDetails".to_owned(),
                index_details: vec![],
                fragment_ids: vec![task_id],
                files: vec![],
            },
        }
    }

    fn uuid(byte: u8) -> Uuid {
        Uuid::from_bytes([byte; 16])
    }

    /// Reconciliation has to be a pure function of the reports, or two
    /// coordinators replaying the same round would publish different segments.
    #[rstest]
    #[case::single_attempt(vec![(0, 1, 10)], vec![10], vec![])]
    #[case::duplicate_delivery(vec![(0, 1, 10), (0, 1, 10)], vec![10], vec![])]
    #[case::lowest_attempt_wins(vec![(0, 2, 20), (0, 1, 10)], vec![10], vec![20])]
    #[case::lowest_attempt_wins_reordered(vec![(0, 1, 10), (0, 2, 20)], vec![10], vec![20])]
    #[case::three_attempts(vec![(0, 3, 30), (0, 1, 10), (0, 2, 20)], vec![10], vec![30, 20])]
    #[case::distinct_tasks(vec![(1, 9, 90), (0, 1, 10)], vec![10, 90], vec![])]
    fn test_reconcile_attempts(
        #[case] reports: Vec<(u32, u8, u8)>,
        #[case] expected_winners: Vec<u8>,
        #[case] expected_orphans: Vec<u8>,
    ) {
        let (winners, orphaned) = reconcile_attempts(
            reports
                .into_iter()
                .map(|(task, attempt, output)| result(task, uuid(attempt), uuid(output)))
                .collect(),
        );
        assert_eq!(
            winners
                .iter()
                .map(|result| result.output.uuid)
                .collect::<Vec<_>>(),
            expected_winners.into_iter().map(uuid).collect::<Vec<_>>()
        );
        assert_eq!(
            orphaned
                .iter()
                .map(|result| result.output.uuid)
                .collect::<Vec<_>>(),
            expected_orphans.into_iter().map(uuid).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_check_disjoint_outputs_rejects_overlap() {
        let mut first = result(0, uuid(1), uuid(10));
        let mut second = result(1, uuid(2), uuid(20));
        first.output.fragment_ids = vec![1, 2];
        second.output.fragment_ids = vec![2, 3];
        let err = check_disjoint_outputs(&[first, second]).unwrap_err();
        assert!(
            err.to_string().contains("overlap on fragment 2"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_contract_version_mismatch_is_rejected() {
        let err = IndexMergePlan::check_contract_version(
            INDEX_MERGE_CONTRACT_VERSION + 1,
            "index merge plan",
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("is not supported by this build"),
            "unexpected error: {err}"
        );
    }

    #[rstest]
    #[case::not_covered(vec![7], vec![], Some(7))]
    #[case::covered(vec![7, 11], vec![11], Some(7))]
    #[case::composite(vec![7, 11], vec![], None)]
    #[case::no_fields(vec![], vec![], None)]
    fn test_fingerprint_keyed_field(
        #[case] fields: Vec<i32>,
        #[case] covering_fields: Vec<i32>,
        #[case] expected: Option<i32>,
    ) {
        let fingerprint = IndexMergeFingerprint {
            index_type_url: "example.Details".to_owned(),
            index_version: 1,
            fields,
            covering_fields,
        };
        assert_eq!(fingerprint.keyed_field(), expected);
    }
}
