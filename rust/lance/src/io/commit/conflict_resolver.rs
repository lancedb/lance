// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

use futures::{StreamExt, TryStreamExt};
use lance_core::{
    utils::{deletion::DeletionVector, mask::RowIdTreeMap},
    Result,
};
use lance_table::{
    format::Fragment,
    io::deletion::{read_deletion_file_cached, write_deletion_file},
};
use snafu::{location, Location};

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
    pub fn check_txn(
        &mut self,
        other_transaction: Option<&Transaction>,
        other_version: u64,
    ) -> Result<()> {
        let Some(other_transaction) = other_transaction else {
            return Err(crate::Error::Internal {
                message: format!(
                    "There was a conflicting transaction at version {}, \
                    and it was missing transaction metadata.",
                    other_version
                ),
                location: location!(),
            });
        };

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
                    Operation::Update { updated_fragments, .. } |
                    Operation::Delete { updated_fragments, .. } => {
                        for updated in updated_fragments {
                            if let Some((fragment, needs_rewrite)) = self.initial_fragments.get_mut(&updated.id) {
                                if self.affected_rows.is_none() {
                                    // We don't have any affected rows, so we can't
                                    // do the rebase anyways.
                                    return Err(self.retryable_conflict_err(
                                        other_transaction,
                                        other_version,
                                        location!()
                                    ));
                                }

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
                        read_deletion_file_cached(
                            fragment_id,
                            &deletion_file.expect("there should be a deletion file"),
                            &dataset.base,
                            dataset.object_store(),
                            &dataset.session.file_metadata_cache,
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
                            "Found conflicts for row addresses {:?}",
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

                // TODO: what do we do with the old deletion files?

                Ok(self.transaction)
            } else {
                // We shouldn't hit this.
                Err(crate::Error::Internal {
                        message: "We have a transaction that needs to be rebased, but we don't have any affected rows.".into(),
                        location: location!(),
                    })
            }
        } else {
            Ok(self.transaction)
        }
    }
}
