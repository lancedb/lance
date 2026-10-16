// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use crate::dataset::transaction::{Operation, Transaction};
use crate::io::commit::conflict_resolver::TransactionRebase;
use crate::{Dataset, Result};
use lance_core::Error;

use super::{
    CompactionMetrics, CompactionOptions, IndexRemapperOptions, RewriteResult,
    commit_compaction_impl,
};

/// The rewrite groups and live rows successfully published by a compaction commit.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CompactionCommitResult {
    /// File and fragment statistics for the committed groups only.
    pub metrics: CompactionMetrics,
    /// Number of rewrite groups committed, including groups containing no live rows.
    pub committed_groups: usize,
    /// Number of live rows represented by the committed groups.
    pub committed_rows: u64,
}

/// Commit deferred-remap compaction, retaining valid groups after concurrent writes.
///
/// Conflicting groups are dropped and the fragment reuse index is regenerated before
/// each retry. The original task read versions remain authoritative. Exhausting all
/// groups returns a conflict error. Uncommitted files are left for garbage collection.
/// Inline index remapping requires atomic completion and is rejected.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use lance::{Dataset, Result};
/// # use lance::dataset::optimize::*;
/// # use lance::dataset::index::DatasetIndexRemapperOptions;
/// # async fn example(dataset: &mut Dataset, results: Vec<RewriteResult>) -> Result<()> {
/// let options = CompactionOptions { defer_index_remap: true, ..Default::default() };
/// let result = commit_compaction_with_partial_progress(
///     dataset, results, Arc::new(DatasetIndexRemapperOptions::default()), &options,
/// ).await?;
/// # Ok(())
/// # }
/// ```
pub async fn commit_compaction_with_partial_progress(
    dataset: &mut Dataset,
    completed_tasks: Vec<RewriteResult>,
    remap_options: Arc<dyn IndexRemapperOptions>,
    options: &CompactionOptions,
) -> Result<CompactionCommitResult> {
    if !options.defer_index_remap {
        return Err(Error::invalid_input(
            "partial compaction requires defer_index_remap",
        ));
    }
    commit_compaction_impl(dataset, completed_tasks, remap_options, options, true)
        .await
        .inspect_err(|error| {
            log::warn!(
                "Partial compaction commit failed; leaving rewritten files in place for GC: {error}"
            )
        })
}

pub(super) async fn surviving_groups(
    dataset: &mut Dataset,
    transaction: &Transaction,
) -> Result<Vec<bool>> {
    let Operation::Rewrite { groups, .. } = &transaction.operation else {
        return Err(Error::invalid_input(
            "partial compaction requires a rewrite transaction",
        ));
    };
    let read = dataset.checkout_version(transaction.read_version).await?;
    let resolver = TransactionRebase::try_new(&read, transaction.clone(), None).await?;
    dataset.checkout_latest().await?;
    let mut keep = vec![true; groups.len()];
    for version in transaction.read_version + 1..=dataset.version().version {
        let other = dataset
            .read_transaction_by_version(version)
            .await?
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "missing transaction for compaction conflict checking at version {version}"
                ))
            })?;
        for (index, keep) in keep.iter_mut().enumerate() {
            if *keep {
                match resolver.rewrite_group_conflicts(&other, version, index) {
                    Ok(()) => {}
                    Err(
                        Error::RetryableCommitConflict { .. }
                        | Error::IncompatibleTransaction { .. },
                    ) => *keep = false,
                    Err(error) => return Err(error),
                }
            }
        }
    }
    Ok(keep)
}

pub(super) fn retain_groups<T>(groups: &mut Vec<T>, keep: &[bool]) {
    let mut index = 0;
    groups.retain(|_| {
        let retain = keep[index];
        index += 1;
        retain
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::index::DatasetIndexRemapperOptions;
    use crate::dataset::{WriteParams, optimize::plan_compaction, transaction::RewriteGroup};
    use crate::index::{DatasetIndexExt, load_all_indices};
    use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_index::IndexType;
    use lance_index::scalar::ScalarIndexParams;

    async fn fixture(tagged: bool) -> (TempStrDir, Dataset, Vec<RewriteResult>, CompactionOptions) {
        let dir = TempStrDir::default();
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(0..40))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            &dir,
            Some(WriteParams {
                max_rows_per_file: 10,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["a"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        if tagged {
            let indices = load_all_indices(&dataset).await.unwrap().as_ref().clone();
            let manifest = Arc::make_mut(&mut dataset.manifest);
            manifest.reader_feature_flags |= lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
            manifest.writer_feature_flags |= lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
            crate::index::frag_reuse_reader::tests::persist_fixture(&mut dataset, indices).await;
        }
        let options = CompactionOptions {
            target_rows_per_fragment: 20,
            defer_index_remap: true,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        let mut results = Vec::new();
        for task in plan.compaction_tasks() {
            results.push(task.execute(&dataset).await.unwrap());
        }
        assert_eq!(results.len(), 2);
        (dir, dataset, results, options)
    }

    #[rstest::rstest]
    #[case(false)]
    #[case(true)]
    #[tokio::test]
    async fn partial_commit_preserves_concurrent_deletions_and_index_queries(#[case] tagged: bool) {
        let (_dir, mut dataset, results, options) = fixture(tagged).await;
        dataset.delete("a = 0").await.unwrap();
        let result = commit_compaction_with_partial_progress(
            &mut dataset,
            results,
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();
        assert_eq!(result.committed_groups, 1);
        assert_eq!(result.committed_rows, 20);
        assert_eq!(result.metrics.fragments_removed, 2);
        assert_eq!(dataset.count_rows(None).await.unwrap(), 39);
        assert_eq!(
            dataset.count_rows(Some("a = 0".to_string())).await.unwrap(),
            0
        );
        assert_eq!(
            dataset
                .count_rows(Some("a >= 20".to_string()))
                .await
                .unwrap(),
            20
        );
        let transaction = dataset.read_transaction().await.unwrap().unwrap();
        let Operation::Rewrite {
            groups,
            frag_reuse_index,
            ..
        } = transaction.operation
        else {
            panic!("expected rewrite");
        };
        assert_eq!(groups.len(), 1);
        assert!(frag_reuse_index.is_some());
    }

    #[tokio::test]
    async fn exhausting_all_groups_fails_without_resurrecting_rows() {
        let (_dir, mut dataset, results, options) = fixture(false).await;
        dataset.delete("a = 0 OR a = 20").await.unwrap();
        let error = commit_compaction_with_partial_progress(
            &mut dataset,
            results,
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap_err();
        assert!(
            matches!(error, Error::RetryableCommitConflict { .. }),
            "{error}"
        );
        assert_eq!(dataset.count_rows(None).await.unwrap(), 38);
        assert!(error.to_string().contains("conflict"));
    }

    #[tokio::test]
    async fn successive_updates_remove_only_newly_conflicting_groups() {
        let (_dir, mut dataset, results, _options) = fixture(false).await;
        let transaction = Transaction::new(
            results[0].read_version,
            Operation::Rewrite {
                groups: results
                    .iter()
                    .map(|r| RewriteGroup {
                        old_fragments: r.original_fragments.clone(),
                        new_fragments: r.new_fragments.clone(),
                    })
                    .collect(),
                rewritten_indices: Vec::new(),
                frag_reuse_index: None,
            },
            None,
        );
        dataset.delete("a = 0").await.unwrap();
        assert_eq!(
            surviving_groups(&mut dataset, &transaction).await.unwrap(),
            vec![false, true]
        );
        dataset.delete("a = 20").await.unwrap();
        assert_eq!(
            surviving_groups(&mut dataset, &transaction).await.unwrap(),
            vec![false, false]
        );
    }

    #[tokio::test]
    async fn committed_rows_exclude_deletions_present_in_the_plan() {
        let (_dir, mut dataset, _, options) = fixture(false).await;
        dataset.delete("a = 0").await.unwrap();
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        let mut results = Vec::new();
        for task in plan.compaction_tasks() {
            results.push(task.execute(&dataset).await.unwrap());
        }
        dataset.delete("a = 20").await.unwrap();
        let result = commit_compaction_with_partial_progress(
            &mut dataset,
            results,
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();
        assert_eq!(result.committed_rows, 19);
        assert_eq!(dataset.count_rows(None).await.unwrap(), 38);
    }

    #[tokio::test]
    async fn concurrent_tagged_rewrite_retains_disjoint_groups() {
        let (_dir, mut dataset, results, options) = fixture(true).await;
        super::super::commit_compaction(
            &mut dataset,
            vec![results[0].clone()],
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();
        let result = commit_compaction_with_partial_progress(
            &mut dataset,
            results,
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();
        assert_eq!(result.committed_groups, 1);
        assert_eq!(result.committed_rows, 20);
        assert_eq!(dataset.count_rows(None).await.unwrap(), 40);
        assert_eq!(
            dataset
                .count_rows(Some("a >= 20".to_string()))
                .await
                .unwrap(),
            20
        );
    }

    #[test]
    fn retention_preserves_group_order() {
        let mut values = vec![10, 20, 30, 40];
        retain_groups(&mut values, &[false, true, false, true]);
        assert_eq!(values, vec![20, 40]);
    }
}
