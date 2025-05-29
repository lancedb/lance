// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    sync::Arc,
};

use futures::{StreamExt, TryStreamExt};
use lance_core::{
    utils::{deletion::DeletionVector, mask::RowIdTreeMap},
    Result,
};
use lance_table::{
    format::Fragment,
    io::deletion::{deletion_file_path, write_deletion_file},
};
use snafu::{location, Location};

use crate::io::deletion::read_dataset_deletion_file;
use crate::{
    dataset::transaction::{ConflictResult, Operation, Transaction},
    Dataset,
};

pub struct TransactionRebase<'a> {
    transaction: Transaction,
    /// Relevant fragments as they were at the read version of the transaction.
    /// Has original fragment, plus a bool indicating whether a rewrite is needed.
    initial_fragments: HashMap<u64, (Fragment, bool)>,
    affected_rows: Option<&'a RowIdTreeMap>,
}

impl<'a> TransactionRebase<'a> {
    pub async fn try_new(
        dataset: &Dataset,
        transaction: Transaction,
        affected_rows: Option<&'a RowIdTreeMap>,
    ) -> Result<Self> {
        // We might not need to check for row-level conflicts anyways.
        if !matches!(&transaction.operation,
            Operation::Update { updated_fragments, .. } |
            Operation::Delete { updated_fragments, .. }
            if !updated_fragments.is_empty() && affected_rows.is_some(),
        ) {
            return Ok(Self {
                transaction,
                initial_fragments: HashMap::new(),
                affected_rows: None,
            });
        }

        let dataset = if dataset.manifest.version != transaction.read_version {
            Cow::Owned(dataset.checkout_version(transaction.read_version).await?)
        } else {
            Cow::Borrowed(dataset)
        };

        let fragments = dataset.fragments().as_slice();

        let modified_fragment_ids = transaction
            .operation
            .modified_fragment_ids()
            .collect::<HashSet<_>>();

        let initial_fragments = fragments
            .iter()
            .filter(|fragment| {
                // Check if the fragment is modified by the transaction.
                modified_fragment_ids.contains(&fragment.id)
            })
            .map(|fragment| (fragment.id, (fragment.clone(), false)))
            .collect::<HashMap<_, _>>();

        Ok(Self {
            initial_fragments,
            transaction,
            affected_rows,
        })
    }

    fn retryable_conflict_err(
        &self,
        other_transaction: &Transaction,
        other_version: u64,
        location: Location,
    ) -> crate::Error {
        crate::Error::RetryableCommitConflict {
            version: other_version,
            source: format!(
                "This {} transaction was preempted by concurrent transaction {} at version {}. Please retry.",
                self.transaction.operation, other_transaction.operation, other_version).into(),
            location,
        }
    }

    /// Check whether the transaction conflicts with another transaction.
    ///
    /// Will return an error if the transaction is not valid. Otherwise, it will
    /// return Ok(()).
    pub fn check_txn(&mut self, other_transaction: &Transaction, other_version: u64) -> Result<()> {
        match self.transaction.conflicts_with(other_transaction) {
            ConflictResult::Compatible => Ok(()),
            ConflictResult::NotCompatible => {
                Err(crate::Error::CommitConflict {
                    version: other_version,
                    source: format!(
                        "This {} transaction is incompatible with concurrent transaction {} at version {}.",
                        self.transaction.operation, other_transaction.operation, other_version).into(),
                    location: location!(),
                })
            },
            ConflictResult::Retryable => {
                match &other_transaction.operation {
                    Operation::Update { updated_fragments, removed_fragment_ids, .. } |
                    Operation::Delete { updated_fragments, deleted_fragment_ids: removed_fragment_ids, .. } => {
                        if self.affected_rows.is_none() {
                            // We don't have any affected rows, so we can't
                            // do the rebase anyways.
                            return Err(self.retryable_conflict_err(
                                other_transaction,
                                other_version,
                                location!()
                            ));
                        }
                        for updated in updated_fragments {
                            if let Some((fragment, needs_rewrite)) = self.initial_fragments.get_mut(&updated.id) {
                                // If data files, not just deletion files, are modified,
                                // then we can't rebase.
                                if fragment.files != updated.files {
                                    return Err(self.retryable_conflict_err(
                                        other_transaction,
                                        other_version,
                                        location!()
                                    ));
                                }

                                // Mark any modified fragments as needing a rewrite.
                                *needs_rewrite |= updated.deletion_file != fragment.deletion_file;
                            }
                        }

                        for removed_fragment_id in removed_fragment_ids {
                            if self.initial_fragments.contains_key(removed_fragment_id) {
                                return Err(self.retryable_conflict_err(
                                        other_transaction,
                                        other_version,
                                        location!()
                                    ));
                            }
                        }
                        return Ok(());
                    },
                    _ => {}
                }

                Err(self.retryable_conflict_err(other_transaction, other_version, location!()))
            }
        }
    }

    /// Writes
    pub async fn finish(mut self, dataset: &Dataset) -> Result<Transaction> {
        if self
            .initial_fragments
            .iter()
            .any(|(_, (_, needs_rewrite))| *needs_rewrite)
        {
            if let Some(affected_rows) = self.affected_rows {
                // Then we do the rebase

                // 1. Load the deletion files that need a rewrite.
                // 2. Validate there is no overlap with the affected rows. (if there is, return retryable conflict error)
                // 3. Write out new deletion files with existing deletes | affected rows.
                // 4. Update the transaction with the new deletion files.

                let fragments_ids_to_rewrite = self
                    .initial_fragments
                    .iter()
                    .filter_map(|(_, (fragment, needs_rewrite))| {
                        if *needs_rewrite {
                            Some(fragment.id)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                // We are rewriting the deletion files on the *current* dataset.
                let files_to_rewrite = dataset
                    .fragments()
                    .as_slice()
                    .iter()
                    .filter_map(|fragment| {
                        if fragments_ids_to_rewrite.contains(&fragment.id) {
                            Some((fragment.id, fragment.deletion_file.clone()))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                let existing_deletion_vecs = futures::stream::iter(files_to_rewrite)
                    .map(|(fragment_id, deletion_file)| async move {
                        read_dataset_deletion_file(
                            dataset,
                            fragment_id,
                            &deletion_file.expect("there should be a deletion file"),
                        )
                        .await
                        .map(|dv| (fragment_id, dv))
                    })
                    .buffered(dataset.object_store().io_parallelism())
                    .try_collect::<Vec<_>>()
                    .await?;

                // Check for row-level conflicts
                let mut existing_deletions = RowIdTreeMap::new();
                for (fragment_id, deletion_vec) in existing_deletion_vecs {
                    existing_deletions
                        .insert_bitmap(fragment_id as u32, deletion_vec.as_ref().into());
                }
                let conflicting_rows = existing_deletions.clone() & affected_rows.clone();
                if conflicting_rows.len().map(|v| v > 0).unwrap_or(true) {
                    let sample_addressed = conflicting_rows
                        .row_ids()
                        .unwrap()
                        .take(5)
                        .collect::<Vec<_>>();
                    return Err(crate::Error::RetryableCommitConflict {
                        version: dataset.manifest.version,
                        source: format!(
                            "This {} transaction was preempted by concurrent transaction {} (both modified rows at addresses {:?}). Please retry",
                            self.transaction.uuid,
                            dataset.manifest.version,
                            sample_addressed.as_slice()
                        )
                        .into(),
                        location: location!(),
                    });
                }

                let merged = existing_deletions.clone() | affected_rows.clone();

                let mut new_deletion_files = HashMap::with_capacity(fragments_ids_to_rewrite.len());
                for fragment_id in fragments_ids_to_rewrite.iter() {
                    let dv = DeletionVector::from(
                        merged
                            .get_fragment_bitmap(*fragment_id as u32)
                            .unwrap()
                            .clone(),
                    );
                    let new_deletion_file = write_deletion_file(
                        &dataset.base,
                        *fragment_id,
                        dataset.manifest.version,
                        &dv,
                        dataset.object_store(),
                    )
                    .await?;

                    // Make sure this is available in the cache for future conflict resolution.
                    let path = deletion_file_path(
                        &dataset.base,
                        *fragment_id,
                        new_deletion_file.as_ref().unwrap(),
                    );
                    dataset
                        .session
                        .file_metadata_cache
                        .insert(path, Arc::new(dv));

                    // TODO: also cleanup the old deletion file.
                    new_deletion_files.insert(*fragment_id, new_deletion_file);
                }

                match &mut self.transaction.operation {
                    Operation::Update {
                        updated_fragments, ..
                    }
                    | Operation::Delete {
                        updated_fragments, ..
                    } => {
                        for updated in updated_fragments {
                            if let Some(new_deletion_file) = new_deletion_files.get(&updated.id) {
                                updated.deletion_file = new_deletion_file.clone();
                            }
                        }
                    }
                    _ => {}
                }

                Ok(Transaction {
                    read_version: dataset.manifest.version,
                    ..self.transaction
                })
            } else {
                // We shouldn't hit this.
                Err(crate::Error::Internal {
                        message: "We have a transaction that needs to be rebased, but we don't have any affected rows.".into(),
                        location: location!(),
                    })
            }
        } else {
            Ok(Transaction {
                read_version: dataset.manifest.version,
                ..self.transaction
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{num::NonZero, sync::Arc};

    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use lance_file::version::LanceFileVersion;
    use lance_io::object_store::ObjectStoreParams;
    use lance_table::io::deletion::{deletion_file_path, read_deletion_file};

    use crate::{
        dataset::{CommitBuilder, InsertBuilder, WriteParams},
        utils::test::StatsHolder,
    };

    use super::*;

    async fn test_dataset(num_rows: usize, num_fragments: usize) -> (Dataset, Arc<StatsHolder>) {
        let io_stats = Arc::new(StatsHolder::default());
        let write_params = WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(io_stats.clone()),
                ..Default::default()
            }),
            max_rows_per_file: num_rows / num_fragments,
            ..Default::default()
        };
        let data = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("a", DataType::Int32, false),
                Field::new("b", DataType::Int32, true),
            ])),
            vec![
                Arc::new(Int32Array::from_iter_values(0..num_rows as i32)),
                Arc::new(Int32Array::from_iter_values(std::iter::repeat_n(
                    0, num_rows,
                ))),
            ],
        )
        .unwrap();
        let dataset = InsertBuilder::new("memory://")
            .with_params(&write_params)
            .execute(vec![data])
            .await
            .unwrap();
        (dataset, io_stats)
    }

    #[tokio::test]
    async fn test_non_overlapping_rebase() {
        let (dataset, io_tracker) = test_dataset(5, 5).await;
        let operation = Operation::Update {
            updated_fragments: vec![Fragment::new(0)],
            removed_fragment_ids: vec![],
            new_fragments: vec![],
            fields_modified: vec![],
        };
        let transaction = Transaction::new_from_version(1, operation);
        let other_operations = [
            Operation::Update {
                updated_fragments: vec![Fragment::new(1)],
                removed_fragment_ids: vec![2],
                new_fragments: vec![],
                fields_modified: vec![],
            },
            Operation::Delete {
                deleted_fragment_ids: vec![3],
                updated_fragments: vec![],
                predicate: "a > 0".to_string(),
            },
            Operation::Update {
                removed_fragment_ids: vec![],
                updated_fragments: vec![Fragment::new(4)],
                new_fragments: vec![],
                fields_modified: vec![],
            },
        ];
        let other_transactions = other_operations.map(|op| Transaction::new_from_version(2, op));
        let mut rebase = TransactionRebase::try_new(&dataset, transaction.clone(), None)
            .await
            .unwrap();

        io_tracker.incremental_stats(); // reset
        for (other_version, other_transaction) in other_transactions.iter().enumerate() {
            rebase
                .check_txn(other_transaction, other_version as u64)
                .unwrap();
            let io_stats = io_tracker.incremental_stats();
            assert_eq!(io_stats.read_iops, 0);
            assert_eq!(io_stats.write_iops, 0);
        }

        let expected_transaction = Transaction {
            // This doesn't really exercise it, since the other transactions
            // haven't been applied yet, but just doing this for completeness.
            read_version: dataset.manifest.version,
            ..transaction
        };
        let rebased_transaction = rebase.finish(&dataset).await.unwrap();
        assert_eq!(rebased_transaction, expected_transaction);
        // We didn't need to do any IO, so the stats should be 0.
        let io_stats = io_tracker.incremental_stats();
        assert_eq!(io_stats.read_iops, 0);
        assert_eq!(io_stats.write_iops, 0);
    }

    async fn apply_deletion(
        delete_rows: &[u32],
        fragment: &mut Fragment,
        dataset: &Dataset,
    ) -> Fragment {
        let mut current_deletions = if let Some(deletion_file) = &fragment.deletion_file {
            read_deletion_file(
                fragment.id,
                deletion_file,
                &dataset.base,
                dataset.object_store(),
            )
            .await
            .unwrap()
        } else {
            DeletionVector::default()
        };

        current_deletions.extend(delete_rows.iter().copied());

        fragment.deletion_file = write_deletion_file(
            &dataset.base,
            fragment.id,
            dataset.manifest.version,
            &current_deletions,
            dataset.object_store(),
        )
        .await
        .unwrap();

        let path = deletion_file_path(
            &dataset.base,
            fragment.id,
            fragment.deletion_file.as_ref().unwrap(),
        );
        dataset
            .session
            .file_metadata_cache
            .insert(path, Arc::new(current_deletions));

        fragment.clone()
    }

    #[tokio::test]
    #[rstest::rstest]
    async fn test_non_conflicting_rebase() {
        // 5 rows, all in one fragment. Each transaction modifies a different row.
        let (mut dataset, io_tracker) = test_dataset(5, 1).await;
        let mut fragment = dataset.fragments().as_slice()[0].clone();

        // Other operations modify the 1st, 2nd, and 3rd rows sequentially.
        let sample_file = Fragment::new(0)
            .with_file(
                "path1",
                vec![0],
                vec![0],
                &LanceFileVersion::V2_0,
                NonZero::new(10),
            )
            .with_physical_rows(3);
        let operations = [
            Operation::Update {
                updated_fragments: vec![apply_deletion(&[0], &mut fragment, &dataset).await],
                removed_fragment_ids: vec![],
                new_fragments: vec![sample_file.clone()],
                fields_modified: vec![],
            },
            Operation::Delete {
                updated_fragments: vec![apply_deletion(&[1], &mut fragment, &dataset).await],
                deleted_fragment_ids: vec![],
                predicate: "a > 0".to_string(),
            },
            Operation::Update {
                updated_fragments: vec![apply_deletion(&[2], &mut fragment, &dataset).await],
                removed_fragment_ids: vec![],
                new_fragments: vec![sample_file],
                fields_modified: vec![],
            },
        ];
        let transactions =
            operations.map(|op| Transaction::new_from_version(dataset.manifest.version, op));

        for (i, transaction) in transactions.iter().enumerate() {
            let previous_transactions = transactions.iter().take(i).cloned().collect::<Vec<_>>();

            let affected_rows = RowIdTreeMap::from_iter([i as u64]);
            let mut rebase =
                TransactionRebase::try_new(&dataset, transaction.clone(), Some(&affected_rows))
                    .await
                    .unwrap();

            io_tracker.incremental_stats(); // reset
            for (other_version, other_transaction) in previous_transactions.iter().enumerate() {
                rebase
                    .check_txn(other_transaction, other_version as u64)
                    .unwrap();
                let io_stats = io_tracker.incremental_stats();
                assert_eq!(io_stats.read_iops, 0);
                assert_eq!(io_stats.write_iops, 0);
            }

            // First iteration, we don't need to rewrite the deletion file.
            let expected_rewrite = i > 0;

            let rebased_transaction = rebase.finish(&dataset).await.unwrap();
            assert_eq!(rebased_transaction.read_version, dataset.manifest.version);

            let io_stats = io_tracker.incremental_stats();
            if expected_rewrite {
                // Read the current deletion file, and write the new one.
                assert_eq!(io_stats.read_iops, 0); // Cached
                assert_eq!(io_stats.write_iops, 1);

                // TODO: The old deletion file should be gone.
                // This can be done later, as it will be cleaned up by the
                // background cleanup process for now.
                // let original_fragment = match &original_transaction.operation {
                //     Operation::Update {
                //         updated_fragments, ..
                //     }
                //     | Operation::Delete {
                //         updated_fragments, ..
                //     } => updated_fragments[0].clone(),
                //     _ => {
                //         panic!("Expected an update or delete operation");
                //     }
                // };
                // let old_path = deletion_file_path(
                //     &dataset.base,
                //     original_fragment.id,
                //     original_fragment.deletion_file.as_ref().unwrap(),
                // );
                // assert!(!dataset.object_store().exists(&old_path).await.unwrap());
                // The new deletion file should exist.
                let final_fragment = match &rebased_transaction.operation {
                    Operation::Update {
                        updated_fragments, ..
                    }
                    | Operation::Delete {
                        updated_fragments, ..
                    } => updated_fragments[0].clone(),
                    _ => {
                        panic!("Expected an update or delete operation");
                    }
                };
                let new_path = deletion_file_path(
                    &dataset.base,
                    final_fragment.id,
                    final_fragment.deletion_file.as_ref().unwrap(),
                );
                assert!(dataset.object_store().exists(&new_path).await.unwrap());

                assert_eq!(io_stats.num_hops, 1);
            } else {
                // No IO should have happened.
                assert_eq!(io_stats.read_iops, 0);
                assert_eq!(io_stats.write_iops, 0);
            }

            dataset = CommitBuilder::new(Arc::new(dataset))
                .execute(rebased_transaction)
                .await
                .unwrap();
        }
    }

    /// Validate we get a conflict error when rebasing `operation` on top of `other`.
    #[tokio::test]
    #[rstest::rstest]
    async fn test_conflicting_rebase(
        #[values("update_full", "update_partial", "delete_full", "delete_partial")] ours: &str,
        #[values("update_full", "update_partial", "delete_full", "delete_partial")] other: &str,
    ) {
        // 5 rows, all in one fragment. Each transaction modifies the same row.
        let (dataset, io_tracker) = test_dataset(5, 1).await;
        let mut fragment = dataset.fragments().as_slice()[0].clone();

        let sample_file = Fragment::new(0)
            .with_file(
                "path1",
                vec![0],
                vec![0],
                &LanceFileVersion::V2_0,
                NonZero::new(10),
            )
            .with_physical_rows(3);

        let operations = [
            (
                "update_full",
                Operation::Update {
                    updated_fragments: vec![],
                    removed_fragment_ids: vec![0],
                    new_fragments: vec![sample_file.clone()],
                    fields_modified: vec![],
                },
            ),
            (
                "update_partial",
                Operation::Update {
                    updated_fragments: vec![apply_deletion(&[0], &mut fragment, &dataset).await],
                    removed_fragment_ids: vec![],
                    new_fragments: vec![sample_file.clone()],
                    fields_modified: vec![],
                },
            ),
            (
                "delete_full",
                Operation::Delete {
                    updated_fragments: vec![],
                    deleted_fragment_ids: vec![0],
                    predicate: "a > 0".to_string(),
                },
            ),
            (
                "delete_partial",
                Operation::Delete {
                    updated_fragments: vec![apply_deletion(&[0], &mut fragment, &dataset).await],
                    deleted_fragment_ids: vec![],
                    predicate: "a > 0".to_string(),
                },
            ),
        ];

        let operation = operations
            .iter()
            .find(|(name, _)| *name == ours)
            .unwrap()
            .1
            .clone();
        let other_op = operations
            .iter()
            .find(|(name, _)| *name == other)
            .unwrap()
            .1
            .clone();

        let other_txn = Transaction::new_from_version(dataset.manifest.version, other_op);
        let txn = Transaction::new_from_version(dataset.manifest.version, operation);

        // Can apply first transaction to create the conflict
        let latest_dataset = CommitBuilder::new(Arc::new(dataset.clone()))
            .execute(other_txn.clone())
            .await
            .unwrap();

        let affected_rows = RowIdTreeMap::from_iter([0]);

        io_tracker.incremental_stats(); // reset
        let mut rebase = TransactionRebase::try_new(&dataset, txn.clone(), Some(&affected_rows))
            .await
            .unwrap();

        let io_stats = io_tracker.incremental_stats();
        assert_eq!(io_stats.read_iops, 0);
        assert_eq!(io_stats.write_iops, 0);

        let res = rebase.check_txn(&other_txn, 1);
        if other.ends_with("full") || ours.ends_with("full") {
            // If the other transaction fully deleted a fragment, we can error early.
            assert!(matches!(
                res,
                Err(crate::Error::RetryableCommitConflict { .. })
            ));
            return;
        } else {
            assert!(res.is_ok());
        }

        assert_eq!(
            rebase
                .initial_fragments
                .iter()
                .map(|(id, (_, needs_rewrite))| (*id, *needs_rewrite))
                .collect::<Vec<_>>(),
            vec![(0, true)],
        );

        let io_stats = io_tracker.incremental_stats();
        assert_eq!(io_stats.read_iops, 0);
        assert_eq!(io_stats.write_iops, 0);

        let res = rebase.finish(&latest_dataset).await;
        assert!(matches!(
            res,
            Err(crate::Error::RetryableCommitConflict { .. })
        ));

        let io_stats = io_tracker.incremental_stats();
        assert_eq!(io_stats.read_iops, 0); // Cached deletion file
        assert_eq!(io_stats.write_iops, 0); // Failed before writing
    }
}
