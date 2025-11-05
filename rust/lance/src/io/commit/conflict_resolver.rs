// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::index::frag_reuse::{build_frag_reuse_index_metadata, load_frag_reuse_index_details};
use crate::io::deletion::read_dataset_deletion_file;
use crate::{
    dataset::transaction::{Operation, Transaction},
    Dataset,
};
use futures::{StreamExt, TryStreamExt};
use lance_core::{
    utils::{deletion::DeletionVector, mask::RowIdTreeMap},
    Error, Result,
};
use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use lance_index::mem_wal::MemWal;
use lance_table::format::IndexMetadata;
use lance_table::{format::Fragment, io::deletion::write_deletion_file};
use snafu::{location, Location};
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    sync::Arc,
};

#[derive(Debug)]
pub struct TransactionRebase<'a> {
    transaction: Transaction,
    /// Relevant fragments as they were at the read version of the transaction.
    /// Has original fragment, plus a bool indicating whether a rewrite is needed.
    initial_fragments: HashMap<u64, (Fragment, bool)>,
    /// Fragments that have been deleted or modified
    modified_fragment_ids: HashSet<u64>,
    affected_rows: Option<&'a RowIdTreeMap>,
    conflicting_frag_reuse_indices: Vec<IndexMetadata>,
}

impl<'a> TransactionRebase<'a> {
    pub async fn try_new(
        dataset: &Dataset,
        transaction: Transaction,
        affected_rows: Option<&'a RowIdTreeMap>,
    ) -> Result<Self> {
        match &transaction.operation {
            // These operations add new fragments or don't modify any.
            Operation::Append { .. }
            | Operation::Overwrite { .. }
            | Operation::CreateIndex { .. }
            | Operation::ReserveFragments { .. }
            | Operation::Project { .. }
            | Operation::UpdateConfig { .. }
            | Operation::UpdateMemWalState { .. }
            | Operation::Clone { .. }
            | Operation::Restore { .. }
            | Operation::UpdateBases { .. } => Ok(Self {
                transaction,
                affected_rows,
                initial_fragments: HashMap::new(),
                modified_fragment_ids: HashSet::new(),
                conflicting_frag_reuse_indices: Vec::new(),
            }),
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids,
                ..
            }
            | Operation::Update {
                updated_fragments,
                removed_fragment_ids: deleted_fragment_ids,
                ..
            } => {
                let modified_fragment_ids = updated_fragments
                    .iter()
                    .map(|f| f.id)
                    .chain(deleted_fragment_ids.iter().copied())
                    .collect::<HashSet<_>>();

                // short circuit for full fragment update or delete case
                // set affected_rows as None with non-empty modified_fragment_ids
                // to indicate this condition to be used in [check_delete_update_txn]
                if updated_fragments.is_empty() && affected_rows.is_some() {
                    return Ok(Self {
                        transaction,
                        initial_fragments: HashMap::new(),
                        modified_fragment_ids,
                        affected_rows: None,
                        conflicting_frag_reuse_indices: Vec::new(),
                    });
                }

                let initial_fragments =
                    initial_fragments_for_rebase(dataset, &transaction, &modified_fragment_ids)
                        .await;
                Ok(Self {
                    transaction,
                    affected_rows,
                    initial_fragments,
                    modified_fragment_ids,
                    conflicting_frag_reuse_indices: Vec::new(),
                })
            }
            Operation::Rewrite { groups, .. } => {
                let modified_fragment_ids = groups
                    .iter()
                    .flat_map(|f| f.old_fragments.iter().map(|f| f.id))
                    .collect::<HashSet<_>>();

                let initial_fragments =
                    initial_fragments_for_rebase(dataset, &transaction, &modified_fragment_ids)
                        .await;
                Ok(Self {
                    transaction,
                    affected_rows,
                    initial_fragments,
                    modified_fragment_ids,
                    conflicting_frag_reuse_indices: Vec::new(),
                })
            }
            Operation::DataReplacement { replacements } => {
                let modified_fragment_ids =
                    replacements.iter().map(|r| r.0).collect::<HashSet<_>>();
                let initial_fragments =
                    initial_fragments_for_rebase(dataset, &transaction, &modified_fragment_ids)
                        .await;
                Ok(Self {
                    transaction,
                    affected_rows,
                    initial_fragments,
                    modified_fragment_ids,
                    conflicting_frag_reuse_indices: Vec::new(),
                })
            }
            Operation::Merge { fragments, .. } => {
                let modified_fragment_ids = fragments.iter().map(|f| f.id).collect::<HashSet<_>>();
                let initial_fragments =
                    initial_fragments_for_rebase(dataset, &transaction, &modified_fragment_ids)
                        .await;
                Ok(Self {
                    transaction,
                    affected_rows,
                    initial_fragments,
                    modified_fragment_ids,
                    conflicting_frag_reuse_indices: Vec::new(),
                })
            }
        }
    }

    fn retryable_conflict_err(
        &self,
        other_transaction: &Transaction,
        other_version: u64,
        location: Location,
    ) -> Error {
        Error::RetryableCommitConflict {
            version: other_version,
            source: format!(
                "This {} transaction was preempted by concurrent transaction {} at version {}. Please retry.",
                self.transaction.operation, other_transaction.operation, other_version).into(),
            location,
        }
    }

    fn incompatible_conflict_err(
        &self,
        other_transaction: &Transaction,
        other_version: u64,
        location: Location,
    ) -> Error {
        Error::CommitConflict {
            version: other_version,
            source: format!(
                "This {} transaction is incompatible with concurrent transaction {} at version {}.",
                self.transaction.operation, other_transaction.operation, other_version
            )
            .into(),
            location,
        }
    }

    /// Check whether the transaction conflicts with another transaction.
    /// Mutate the current [TransactionRebase] based on [other_transaction] to be used for
    /// eventually [finish] the rebase process.
    ///
    /// Will return an error if the transaction is not valid. Otherwise, it will
    /// return Ok(()).
    pub fn check_txn(&mut self, other_transaction: &Transaction, other_version: u64) -> Result<()> {
        let op = &self.transaction.operation;
        match op {
            Operation::Delete { .. } => self.check_delete_txn(other_transaction, other_version),
            Operation::Update { .. } => self.check_update_txn(other_transaction, other_version),
            Operation::CreateIndex { .. } => {
                self.check_create_index_txn(other_transaction, other_version)
            }
            Operation::Rewrite { .. } => self.check_rewrite_txn(other_transaction, other_version),
            Operation::Overwrite { .. } => {
                self.check_overwrite_txn(other_transaction, other_version)
            }
            Operation::Append { .. } => self.check_append_txn(other_transaction, other_version),
            Operation::DataReplacement { .. } => {
                self.check_data_replacement_txn(other_transaction, other_version)
            }
            Operation::Merge { .. } => self.check_merge_txn(other_transaction, other_version),
            Operation::Restore { .. } => self.check_restore_txn(other_transaction, other_version),
            Operation::ReserveFragments { .. } => {
                self.check_reserve_fragments_txn(other_transaction, other_version)
            }
            Operation::Project { .. } => self.check_project_txn(other_transaction, other_version),
            Operation::UpdateConfig { .. } => {
                self.check_update_config_txn(other_transaction, other_version)
            }
            Operation::UpdateMemWalState { .. } => {
                self.check_update_mem_wal_state_txn(other_transaction, other_version)
            }
            Operation::Clone { .. } => Ok(()),
            Operation::UpdateBases { .. } => {
                self.check_add_bases_txn(other_transaction, other_version)
            }
        }
    }

    fn check_delete_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        if let Operation::Delete { .. } = &self.transaction.operation {
            match &other_transaction.operation {
                Operation::CreateIndex { .. }
                | Operation::ReserveFragments { .. }
                | Operation::Clone { .. }
                | Operation::Project { .. }
                | Operation::Append { .. }
                | Operation::UpdateConfig { .. }
                | Operation::UpdateBases { .. } => Ok(()),
                Operation::Rewrite { groups, .. } => {
                    if groups
                        .iter()
                        .flat_map(|f| f.old_fragments.iter().map(|f| f.id))
                        .any(|id| self.modified_fragment_ids.contains(&id))
                    {
                        Err(self.retryable_conflict_err(
                            other_transaction,
                            other_version,
                            location!(),
                        ))
                    } else {
                        Ok(())
                    }
                }
                Operation::DataReplacement { replacements, .. } => {
                    if replacements
                        .iter()
                        .map(|r| r.0)
                        .any(|id| self.modified_fragment_ids.contains(&id))
                    {
                        Err(self.retryable_conflict_err(
                            other_transaction,
                            other_version,
                            location!(),
                        ))
                    } else {
                        Ok(())
                    }
                }
                Operation::Update {
                    updated_fragments,
                    removed_fragment_ids,
                    ..
                }
                | Operation::Delete {
                    updated_fragments,
                    deleted_fragment_ids: removed_fragment_ids,
                    ..
                } => {
                    if !updated_fragments
                        .iter()
                        .map(|f| f.id)
                        .chain(removed_fragment_ids.iter().copied())
                        .any(|id| self.modified_fragment_ids.contains(&id))
                    {
                        return Ok(());
                    }

                    if self.affected_rows.is_none() {
                        // We don't have any affected rows, so we can't
                        // do the rebase anyways.
                        return Err(self.retryable_conflict_err(
                            other_transaction,
                            other_version,
                            location!(),
                        ));
                    }
                    for updated in updated_fragments {
                        if let Some((fragment, needs_rewrite)) =
                            self.initial_fragments.get_mut(&updated.id)
                        {
                            // If data files, not just deletion files, are modified,
                            // then we can't rebase.
                            if fragment.files != updated.files {
                                return Err(self.retryable_conflict_err(
                                    other_transaction,
                                    other_version,
                                    location!(),
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
                                location!(),
                            ));
                        }
                    }
                    Ok(())
                }
                Operation::Merge { .. } => {
                    Err(self.retryable_conflict_err(other_transaction, other_version, location!()))
                }
                Operation::Overwrite { .. }
                | Operation::Restore { .. }
                | Operation::UpdateMemWalState { .. } => Err(self.incompatible_conflict_err(
                    other_transaction,
                    other_version,
                    location!(),
                )),
            }
        } else {
            Err(wrong_operation_err(&self.transaction.operation))
        }
    }

    fn check_update_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        if let Operation::Update {
            mem_wal_to_merge, ..
        } = &self.transaction.operation
        {
            match &other_transaction.operation {
                Operation::CreateIndex { .. }
                | Operation::ReserveFragments { .. }
                | Operation::Project { .. }
                | Operation::Append { .. }
                | Operation::Clone { .. }
                | Operation::UpdateConfig { .. }
                | Operation::UpdateBases { .. } => Ok(()),
                Operation::Rewrite { groups, .. } => {
                    if groups
                        .iter()
                        .flat_map(|f| f.old_fragments.iter().map(|f| f.id))
                        .any(|id| self.modified_fragment_ids.contains(&id))
                    {
                        Err(self.retryable_conflict_err(
                            other_transaction,
                            other_version,
                            location!(),
                        ))
                    } else {
                        Ok(())
                    }
                }
                Operation::DataReplacement { replacements, .. } => {
                    if replacements
                        .iter()
                        .map(|r| r.0)
                        .any(|id| self.modified_fragment_ids.contains(&id))
                    {
                        Err(self.retryable_conflict_err(
                            other_transaction,
                            other_version,
                            location!(),
                        ))
                    } else {
                        Ok(())
                    }
                }
                Operation::Update {
                    updated_fragments,
                    removed_fragment_ids,
                    ..
                }
                | Operation::Delete {
                    updated_fragments,
                    deleted_fragment_ids: removed_fragment_ids,
                    ..
                } => {
                    if !updated_fragments
                        .iter()
                        .map(|f| f.id)
                        .chain(removed_fragment_ids.iter().copied())
                        .any(|id| self.modified_fragment_ids.contains(&id))
                    {
                        return Ok(());
                    }

                    if self.affected_rows.is_none() {
                        // We don't have any affected rows, so we can't
                        // do the rebase anyways.
                        return Err(self.retryable_conflict_err(
                            other_transaction,
                            other_version,
                            location!(),
                        ));
                    }
                    for updated in updated_fragments {
                        if let Some((fragment, needs_rewrite)) =
                            self.initial_fragments.get_mut(&updated.id)
                        {
                            // If data files, not just deletion files, are modified,
                            // then we can't rebase.
                            if fragment.files != updated.files {
                                return Err(self.retryable_conflict_err(
                                    other_transaction,
                                    other_version,
                                    location!(),
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
                                location!(),
                            ));
                        }
                    }
                    Ok(())
                }
                Operation::Merge { .. } => {
                    Err(self.retryable_conflict_err(other_transaction, other_version, location!()))
                }
                Operation::Overwrite { .. } | Operation::Restore { .. } => Err(
                    self.incompatible_conflict_err(other_transaction, other_version, location!())
                ),
                Operation::UpdateMemWalState { added, updated, .. } => {
                    self.check_update_mem_wal_state_not_modify_same_mem_wal(
                        added,
                        mem_wal_to_merge.as_slice(),
                        other_transaction,
                        other_version,
                    )?;
                    self.check_update_mem_wal_state_not_modify_same_mem_wal(
                        updated,
                        mem_wal_to_merge.as_slice(),
                        other_transaction,
                        other_version,
                    )?;
                    Ok(())
                }
            }
        } else {
            Err(wrong_operation_err(&self.transaction.operation))
        }
    }

    fn check_create_index_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        if let Operation::CreateIndex {
            new_indices,
            removed_indices,
            ..
        } = &self.transaction.operation
        {
            match &other_transaction.operation {
                Operation::Append { .. }
                | Operation::Clone { .. }
                | Operation::UpdateBases { .. } => Ok(()),
                // Indices are identified by UUIDs, so they shouldn't conflict.
                // unless it is the same frag reuse index
                Operation::CreateIndex {
                    new_indices: created_indices,
                    ..
                } => {
                    if new_indices
                        .iter()
                        .any(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
                        && created_indices
                            .iter()
                            .any(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
                    {
                        Err(self.retryable_conflict_err(
                            other_transaction,
                            other_version,
                            location!(),
                        ))
                    } else {
                        Ok(())
                    }
                }
                // Although some of the rows we indexed may have been deleted / moved,
                // row ids are still valid, so we allow this optimistically.
                Operation::Delete { .. } | Operation::Update { .. } => Ok(()),
                // Merge, reserve, and project don't change row ids, so this should be fine.
                Operation::Merge { .. } => Ok(()),
                Operation::ReserveFragments { .. } => Ok(()),
                Operation::Project { .. } => Ok(()),
                // Should be compatible with rewrite if it didn't move the rows
                // we indexed. If it did, we could retry.
                // TODO: this will change with stable row ids.
                Operation::Rewrite {
                    groups,
                    frag_reuse_index,
                    ..
                } => {
                    // if frag_reuse_index is available, index remapping is deferred and
                    // there is no conflict with concurrent CreateIndex of column indices.
                    // The only case that needs rebasing is when the frag_reuse_index cleanup
                    // triggers a CreateIndex, and it needs to add the new reuse
                    // version created by the rewrite
                    if let Some(committed_fri) = frag_reuse_index {
                        if new_indices
                            .iter()
                            .any(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
                        {
                            // this should not happen today since we don't support committing
                            // a mixture of frag_reuse_index and other indices.
                            if new_indices.len() != 1 || removed_indices.len() != 1 {
                                return Err(self.incompatible_conflict_err(
                                    other_transaction,
                                    other_version,
                                    location!(),
                                ));
                            }

                            self.conflicting_frag_reuse_indices
                                .push(committed_fri.clone());
                            Ok(())
                        } else {
                            Ok(())
                        }
                    } else {
                        let mut affected_ids = HashSet::new();
                        for index in new_indices.iter() {
                            if let Some(frag_bitmap) = &index.fragment_bitmap {
                                affected_ids.extend(frag_bitmap.iter());
                            } else {
                                return Err(self.retryable_conflict_err(
                                    other_transaction,
                                    other_version,
                                    location!(),
                                ));
                            }
                        }

                        if groups
                            .iter()
                            .flat_map(|f| f.old_fragments.iter().map(|f| f.id))
                            .any(|id| affected_ids.contains(&(id as u32)))
                        {
                            Err(self.retryable_conflict_err(
                                other_transaction,
                                other_version,
                                location!(),
                            ))
                        } else {
                            Ok(())
                        }
                    }
                }
                Operation::UpdateConfig { .. } => Ok(()),
                Operation::DataReplacement { replacements } => {
                    // A data replacement only conflicts if it is updating the field that
                    // is being indexed.
                    let newly_indexed_fields = new_indices
                        .iter()
                        .flat_map(|idx| idx.fields.iter())
                        .collect::<HashSet<_>>();
                    for replacement in replacements {
                        for field in &replacement.1.fields {
                            if newly_indexed_fields.contains(&field) {
                                return Err(self.retryable_conflict_err(
                                    other_transaction,
                                    other_version,
                                    location!(),
                                ));
                            }
                        }
                    }
                    Ok(())
                }
                Operation::Overwrite { .. }
                | Operation::Restore { .. }
                | Operation::UpdateMemWalState { .. } => Err(self.incompatible_conflict_err(
                    other_transaction,
                    other_version,
                    location!(),
                )),
            }
        } else {
            Err(wrong_operation_err(&self.transaction.operation))
        }
    }

    fn check_rewrite_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        if let Operation::Rewrite {
            groups,
            frag_reuse_index,
            ..
        } = &self.transaction.operation
        {
            match &other_transaction.operation {
                // Rewrite is only compatible with operations that don't touch
                // existing fragments or update fragments we don't touch.
                Operation::Append { .. }
                | Operation::ReserveFragments { .. }
                | Operation::Project { .. }
                | Operation::Clone { .. }
                | Operation::UpdateConfig { .. }
                | Operation::UpdateMemWalState { .. }
                | Operation::UpdateBases { .. } => Ok(()),
                Operation::Delete {
                    updated_fragments,
                    deleted_fragment_ids,
                    ..
                }
                | Operation::Update {
                    updated_fragments,
                    removed_fragment_ids: deleted_fragment_ids,
                    ..
                } => {
                    if updated_fragments
                        .iter()
                        .map(|f| f.id)
                        .chain(deleted_fragment_ids.iter().copied())
                        .any(|id| self.modified_fragment_ids.contains(&id))
                    {
                        Err(self.retryable_conflict_err(
                            other_transaction,
                            other_version,
                            location!(),
                        ))
                    } else {
                        Ok(())
                    }
                }
                Operation::Rewrite {
                    groups,
                    frag_reuse_index: committed_fri,
                    ..
                } => {
                    if groups
                        .iter()
                        .flat_map(|f| f.old_fragments.iter().map(|f| f.id))
                        .any(|id| self.modified_fragment_ids.contains(&id))
                    {
                        Err(self.retryable_conflict_err(
                            other_transaction,
                            other_version,
                            location!(),
                        ))
                    } else if committed_fri.is_some() && frag_reuse_index.is_some() {
                        // Do not commit concurrent rewrites that could produce conflicting frag_reuse_indexes.
                        // The other rewrite must retry.
                        // TODO: could potentially rebase to combine both frag_reuse_indexes,
                        //   but today it is already rare to run concurrent rewrites.
                        Err(self.retryable_conflict_err(
                            other_transaction,
                            other_version,
                            location!(),
                        ))
                    } else {
                        Ok(())
                    }
                }
                Operation::DataReplacement { .. } | Operation::Merge { .. } => {
                    // TODO(rmeng): check that the fragments being replaced are not part of the groups
                    Err(self.retryable_conflict_err(other_transaction, other_version, location!()))
                }
                Operation::CreateIndex {
                    new_indices,
                    removed_indices,
                    ..
                } => {
                    match (
                        new_indices
                            .iter()
                            .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME),
                        &frag_reuse_index,
                    ) {
                        // If the rewrite produces a frag_reuse_index, but frag_reuse_index was cleaned up
                        // in the other transaction, the frag_reuse_index produced by the rewrite should
                        // be cleaned up in the same way as a part of the rebase.
                        (Some(committed_fri), Some(_)) => {
                            // this should not happen today since we don't support committing
                            // a mixture of frag_reuse_index and other indices.
                            if new_indices.len() != 1 || removed_indices.len() != 1 {
                                return Err(self.incompatible_conflict_err(
                                    other_transaction,
                                    other_version,
                                    location!(),
                                ));
                            }

                            self.conflicting_frag_reuse_indices
                                .push(committed_fri.clone());
                            Ok(())
                        }
                        // If rewrite defers index remap,
                        // then it does not conflict with index creation
                        (None, Some(_)) => Ok(()),
                        // Rewrite with remapping and frag_reuse_index creation can commit without conflict
                        (Some(_), None) => {
                            // this should not happen today since we don't support committing
                            // a mixture of frag_reuse_index and other indices.
                            if new_indices.len() != 1 || removed_indices.len() != 1 {
                                Err(self.incompatible_conflict_err(
                                    other_transaction,
                                    other_version,
                                    location!(),
                                ))
                            } else {
                                Ok(())
                            }
                        }
                        // Rewrite with remapping will conflict with
                        // index creation that touches overlapping fragments.
                        (_, None) => {
                            let mut affected_ids = HashSet::new();
                            for index in new_indices {
                                if let Some(frag_bitmap) = &index.fragment_bitmap {
                                    affected_ids.extend(frag_bitmap.iter());
                                } else {
                                    return Err(self.retryable_conflict_err(
                                        other_transaction,
                                        other_version,
                                        location!(),
                                    ));
                                }
                            }
                            if groups
                                .iter()
                                .flat_map(|f| f.old_fragments.iter().map(|f| f.id))
                                .any(|id| affected_ids.contains(&(id as u32)))
                            {
                                Err(self.retryable_conflict_err(
                                    other_transaction,
                                    other_version,
                                    location!(),
                                ))
                            } else {
                                Ok(())
                            }
                        }
                    }
                }
                Operation::Overwrite { .. } | Operation::Restore { .. } => Err(
                    self.incompatible_conflict_err(other_transaction, other_version, location!())
                ),
            }
        } else {
            Err(wrong_operation_err(&self.transaction.operation))
        }
    }

    fn check_overwrite_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        match &other_transaction.operation {
            // Overwrite only conflicts with another operation modifying the same update config
            Operation::Overwrite { .. } | Operation::UpdateConfig { .. } => {
                if self
                    .transaction
                    .operation
                    .upsert_key_conflict(&other_transaction.operation)
                {
                    Err(self.incompatible_conflict_err(
                        other_transaction,
                        other_version,
                        location!(),
                    ))
                } else {
                    Ok(())
                }
            }
            Operation::UpdateMemWalState { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version, location!()))
            }
            Operation::Append { .. }
            | Operation::Clone { .. }
            | Operation::Delete { .. }
            | Operation::CreateIndex { .. }
            | Operation::Rewrite { .. }
            | Operation::DataReplacement { .. }
            | Operation::Merge { .. }
            | Operation::Restore { .. }
            | Operation::ReserveFragments { .. }
            | Operation::Update { .. }
            | Operation::Project { .. }
            | Operation::UpdateBases { .. } => Ok(()),
        }
    }

    fn check_append_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        match &other_transaction.operation {
            // Append is not compatible with any operation that completely
            // overwrites the schema.
            Operation::Overwrite { .. }
            | Operation::Restore { .. }
            | Operation::UpdateMemWalState { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version, location!()))
            }
            Operation::Append { .. }
            | Operation::Rewrite { .. }
            | Operation::CreateIndex { .. }
            | Operation::Delete { .. }
            | Operation::Update { .. }
            | Operation::ReserveFragments { .. }
            | Operation::Project { .. }
            | Operation::UpdateBases { .. }
            | Operation::Merge { .. }
            | Operation::UpdateConfig { .. }
            | Operation::Clone { .. }
            | Operation::DataReplacement { .. } => Ok(()),
        }
    }

    fn check_data_replacement_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        if let Operation::DataReplacement { replacements } = &self.transaction.operation {
            match &other_transaction.operation {
                Operation::Append { .. }
                | Operation::Clone { .. }
                | Operation::Delete { .. }
                | Operation::Update { .. }
                | Operation::Merge { .. }
                | Operation::UpdateConfig { .. }
                | Operation::ReserveFragments { .. }
                | Operation::Project { .. }
                | Operation::UpdateBases { .. } => Ok(()),
                Operation::CreateIndex { new_indices, .. } => {
                    // A data replacement only conflicts if it is updating the field that
                    // is being indexed.
                    //
                    // TODO: We could potentially just drop the fragments being replaced from
                    // the index's fragment bitmap, which would lead to fewer conflicts.  However
                    // this would introduce fragment bitmaps with holes which may not be well tested
                    // yet.  For now, we don't allow this case.
                    let newly_indexed_fields = new_indices
                        .iter()
                        .flat_map(|idx| idx.fields.iter())
                        .collect::<HashSet<_>>();
                    for replacement in replacements {
                        for field in &replacement.1.fields {
                            if newly_indexed_fields.contains(&field) {
                                return Err(self.retryable_conflict_err(
                                    other_transaction,
                                    other_version,
                                    location!(),
                                ));
                            }
                        }
                    }
                    Ok(())
                }
                Operation::Rewrite { .. } => {
                    // TODO(rmeng): check that the fragments being replaced are not part of the groups
                    Err(self.incompatible_conflict_err(
                        other_transaction,
                        other_version,
                        location!(),
                    ))
                }
                Operation::DataReplacement { .. } => {
                    // TODO(rmeng): check cell conflicts
                    Err(self.incompatible_conflict_err(
                        other_transaction,
                        other_version,
                        location!(),
                    ))
                }
                Operation::Overwrite { .. }
                | Operation::Restore { .. }
                | Operation::UpdateMemWalState { .. } => Err(self.incompatible_conflict_err(
                    other_transaction,
                    other_version,
                    location!(),
                )),
            }
        } else {
            Err(wrong_operation_err(&self.transaction.operation))
        }
    }

    fn check_merge_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        match &other_transaction.operation {
            Operation::CreateIndex { .. }
            | Operation::ReserveFragments { .. }
            | Operation::Clone { .. }
            | Operation::UpdateConfig { .. }
            | Operation::UpdateBases { .. } => Ok(()),

            Operation::Update { .. }
            | Operation::Append { .. }
            | Operation::Delete { .. }
            | Operation::Rewrite { .. }
            | Operation::Merge { .. }
            | Operation::DataReplacement { .. } => {
                Err(self.retryable_conflict_err(other_transaction, other_version, location!()))
            }
            Operation::Overwrite { .. }
            | Operation::Restore { .. }
            | Operation::Project { .. }
            | Operation::UpdateMemWalState { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version, location!()))
            }
        }
    }

    fn check_restore_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        match &other_transaction.operation {
            Operation::Append { .. }
            | Operation::Delete { .. }
            | Operation::Overwrite { .. }
            | Operation::CreateIndex { .. }
            | Operation::Rewrite { .. }
            | Operation::DataReplacement { .. }
            | Operation::Merge { .. }
            | Operation::Restore { .. }
            | Operation::ReserveFragments { .. }
            | Operation::UpdateBases { .. }
            | Operation::Update { .. }
            | Operation::Project { .. }
            | Operation::Clone { .. }
            | Operation::UpdateConfig { .. } => Ok(()),
            Operation::UpdateMemWalState { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version, location!()))
            }
        }
    }

    fn check_reserve_fragments_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        match &other_transaction.operation {
            Operation::Overwrite { .. } | Operation::Restore { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version, location!()))
            }
            Operation::Append { .. }
            | Operation::Delete { .. }
            | Operation::CreateIndex { .. }
            | Operation::Rewrite { .. }
            | Operation::DataReplacement { .. }
            | Operation::Merge { .. }
            | Operation::ReserveFragments { .. }
            | Operation::Update { .. }
            | Operation::Project { .. }
            | Operation::Clone { .. }
            | Operation::UpdateConfig { .. }
            | Operation::UpdateMemWalState { .. }
            | Operation::UpdateBases { .. } => Ok(()),
        }
    }

    fn check_project_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        match &other_transaction.operation {
            // Project is compatible with anything that doesn't change the schema
            Operation::Append { .. }
            | Operation::Update { .. }
            | Operation::Delete { .. }
            | Operation::UpdateConfig { .. }
            | Operation::CreateIndex { .. }
            | Operation::DataReplacement { .. }
            | Operation::Rewrite { .. }
            | Operation::Clone { .. }
            | Operation::ReserveFragments { .. }
            | Operation::UpdateBases { .. } => Ok(()),
            Operation::Merge { .. } | Operation::Project { .. } => {
                // Need to recompute the schema
                Err(self.retryable_conflict_err(other_transaction, other_version, location!()))
            }
            Operation::Overwrite { .. }
            | Operation::Restore { .. }
            | Operation::UpdateMemWalState { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version, location!()))
            }
        }
    }

    fn check_update_config_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        if let Operation::UpdateConfig {
            schema_metadata_updates,
            field_metadata_updates,
            ..
        } = &self.transaction.operation
        {
            match &other_transaction.operation {
                Operation::Overwrite { .. } => {
                    // Updates to schema metadata or field metadata conflict with any kind
                    // of overwrite.
                    if schema_metadata_updates.is_some()
                        || !field_metadata_updates.is_empty()
                        || self
                            .transaction
                            .operation
                            .upsert_key_conflict(&other_transaction.operation)
                    {
                        Err(self.incompatible_conflict_err(
                            other_transaction,
                            other_version,
                            location!(),
                        ))
                    } else {
                        Ok(())
                    }
                }
                Operation::UpdateConfig { .. } => {
                    if self
                        .transaction
                        .operation
                        .upsert_key_conflict(&other_transaction.operation)
                        | self
                            .transaction
                            .operation
                            .modifies_same_metadata(&other_transaction.operation)
                    {
                        Err(self.incompatible_conflict_err(
                            other_transaction,
                            other_version,
                            location!(),
                        ))
                    } else {
                        Ok(())
                    }
                }
                Operation::Append { .. }
                | Operation::Clone { .. }
                | Operation::Delete { .. }
                | Operation::CreateIndex { .. }
                | Operation::Rewrite { .. }
                | Operation::DataReplacement { .. }
                | Operation::Merge { .. }
                | Operation::Restore { .. }
                | Operation::ReserveFragments { .. }
                | Operation::Update { .. }
                | Operation::Project { .. }
                | Operation::UpdateMemWalState { .. }
                | Operation::UpdateBases { .. } => Ok(()),
            }
        } else {
            Err(wrong_operation_err(&self.transaction.operation))
        }
    }

    fn check_update_mem_wal_state_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        if let Operation::UpdateMemWalState {
            added,
            updated,
            removed: _,
            ..
        } = &self.transaction.operation
        {
            match &other_transaction.operation {
                Operation::UpdateMemWalState {
                    added: committed_added,
                    updated: committed_updated,
                    removed: _,
                } => {
                    // 1. if the current or last committed job is trimming flushed MemWALs,
                    // it is compatible with any other UpdateMemWalState commits
                    if (committed_added.is_empty() && committed_updated.is_empty())
                        || (added.is_empty() && updated.is_empty())
                    {
                        return Ok(());
                    }

                    // 2. MemWALs of different regions can be changed at the same time
                    self.check_update_mem_wal_state_not_modify_same_mem_wal(
                        committed_added,
                        added,
                        other_transaction,
                        other_version,
                    )?;
                    self.check_update_mem_wal_state_not_modify_same_mem_wal(
                        committed_added,
                        updated,
                        other_transaction,
                        other_version,
                    )?;
                    self.check_update_mem_wal_state_not_modify_same_mem_wal(
                        committed_updated,
                        added,
                        other_transaction,
                        other_version,
                    )?;
                    self.check_update_mem_wal_state_not_modify_same_mem_wal(
                        committed_updated,
                        updated,
                        other_transaction,
                        other_version,
                    )?;
                    Ok(())
                }
                Operation::Update {
                    mem_wal_to_merge, ..
                } => {
                    if mem_wal_to_merge.is_some() {
                        // TODO: This check could be more detailed, there is an assumption that
                        //  once a MemWAL is sealed, there is no other operation that could change
                        //  the state back to open, and at that point it can always be flushed.
                        Ok(())
                    } else {
                        Err(self.incompatible_conflict_err(
                            other_transaction,
                            other_version,
                            location!(),
                        ))
                    }
                }
                Operation::UpdateConfig { .. }
                | Operation::Rewrite { .. }
                | Operation::CreateIndex { .. }
                | Operation::ReserveFragments { .. }
                | Operation::UpdateBases { .. } => Ok(()),
                Operation::Append { .. }
                | Operation::Overwrite { .. }
                | Operation::Delete { .. }
                | Operation::DataReplacement { .. }
                | Operation::Merge { .. }
                | Operation::Restore { .. }
                | Operation::Clone { .. }
                | Operation::Project { .. } => Err(self.incompatible_conflict_err(
                    other_transaction,
                    other_version,
                    location!(),
                )),
            }
        } else {
            Err(wrong_operation_err(&self.transaction.operation))
        }
    }

    fn check_add_bases_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        if let Operation::UpdateBases { new_bases } = &self.transaction.operation {
            match &other_transaction.operation {
                Operation::UpdateBases {
                    new_bases: committed_bases,
                } => {
                    // Check if any of the bases being added conflict with committed bases
                    for new_base in new_bases {
                        for committed_base in committed_bases {
                            // Check for ID conflicts (if both have non-zero IDs)
                            if new_base.id != 0
                                && committed_base.id != 0
                                && new_base.id == committed_base.id
                            {
                                return Err(self.incompatible_conflict_err(
                                    other_transaction,
                                    other_version,
                                    location!(),
                                ));
                            }
                            // Check for name conflicts
                            if new_base.name == committed_base.name && new_base.name.is_some() {
                                return Err(self.incompatible_conflict_err(
                                    other_transaction,
                                    other_version,
                                    location!(),
                                ));
                            }
                            // Check for path conflicts
                            if new_base.path == committed_base.path {
                                return Err(self.incompatible_conflict_err(
                                    other_transaction,
                                    other_version,
                                    location!(),
                                ));
                            }
                        }
                    }
                    Ok(())
                }
                // UpdateBases doesn't conflict with data operations
                _ => Ok(()),
            }
        } else {
            Err(wrong_operation_err(&self.transaction.operation))
        }
    }

    fn check_update_mem_wal_state_not_modify_same_mem_wal(
        &self,
        committed: &[MemWal],
        to_commit: &[MemWal],
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        if !committed.is_empty() {
            if to_commit.is_empty() {
                return Ok(());
            }

            if committed.len() > 1 {
                return Err(Error::Internal {
                    message: format!(
                        "Committing multiple MemWALs is not supported, but found committed: {:?}",
                        committed
                    ),
                    location: location!(),
                });
            }

            if to_commit.len() > 1 {
                return Err(Error::NotSupported {
                    source: format!(
                        "Committing multiple MemWALs is not supported, but found attempt to commit: {:?}",
                        to_commit
                    )
                    .into(),
                    location: location!(),
                });
            }

            let committed_mem_wal = committed.first().unwrap();
            let to_commit_mem_wal = to_commit.first().unwrap();
            if committed_mem_wal.id == to_commit_mem_wal.id {
                return Err(self.incompatible_conflict_err(
                    other_transaction,
                    other_version,
                    location!(),
                ));
            }
        }

        Ok(())
    }

    /// Writes
    pub async fn finish(self, dataset: &Dataset) -> Result<Transaction> {
        match &self.transaction.operation {
            Operation::Delete { .. } | Operation::Update { .. } => {
                self.finish_delete_update(dataset).await
            }
            Operation::CreateIndex { .. } => self.finish_create_index(dataset).await,
            Operation::Rewrite { .. } => self.finish_rewrite(dataset).await,
            Operation::Append { .. }
            | Operation::Overwrite { .. }
            | Operation::DataReplacement { .. }
            | Operation::Merge { .. }
            | Operation::Restore { .. }
            | Operation::ReserveFragments { .. }
            | Operation::Project { .. }
            | Operation::Clone { .. }
            | Operation::UpdateConfig { .. }
            | Operation::UpdateMemWalState { .. }
            | Operation::UpdateBases { .. } => Ok(self.transaction),
        }
    }

    async fn finish_delete_update(mut self, dataset: &Dataset) -> Result<Transaction> {
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

                let mut new_deleted_frag_ids = Vec::new();
                let mut new_deletion_files = HashMap::with_capacity(fragments_ids_to_rewrite.len());
                for fragment_id in fragments_ids_to_rewrite.iter() {
                    let dv = DeletionVector::from(
                        merged
                            .get_fragment_bitmap(*fragment_id as u32)
                            .unwrap()
                            .clone(),
                    );
                    // If we've deleted all rows in the fragment, we can delete it.
                    // It's acceptable if we don't handle it here, as the commit step
                    // can handle it later. Though it should be rare that physical_rows
                    // is missing.
                    if let Some(physical_rows) = self
                        .initial_fragments
                        .get(fragment_id)
                        .and_then(|(fragment, _)| fragment.physical_rows)
                    {
                        if dv.len() == physical_rows {
                            new_deleted_frag_ids.push(*fragment_id);
                            continue;
                        }
                    }

                    let new_deletion_file = write_deletion_file(
                        &dataset.base,
                        *fragment_id,
                        dataset.manifest.version,
                        &dv,
                        dataset.object_store(),
                    )
                    .await?;

                    // Make sure this is available in the cache for future conflict resolution.
                    let deletion_file = new_deletion_file.as_ref().unwrap();
                    let key = crate::session::caches::DeletionFileKey {
                        fragment_id: *fragment_id,
                        deletion_file,
                    };
                    dataset
                        .metadata_cache
                        .insert_with_key(&key, Arc::new(dv))
                        .await;

                    // TODO: also cleanup the old deletion file.
                    new_deletion_files.insert(*fragment_id, new_deletion_file);
                }

                match &mut self.transaction.operation {
                    Operation::Update {
                        updated_fragments,
                        removed_fragment_ids,
                        ..
                    }
                    | Operation::Delete {
                        updated_fragments,
                        deleted_fragment_ids: removed_fragment_ids,
                        ..
                    } => {
                        for updated in updated_fragments {
                            if let Some(new_deletion_file) = new_deletion_files.get(&updated.id) {
                                updated.deletion_file = new_deletion_file.clone();
                            }
                        }
                        removed_fragment_ids.extend(new_deleted_frag_ids);
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

    async fn finish_create_index(mut self, dataset: &Dataset) -> Result<Transaction> {
        if let Operation::CreateIndex { new_indices, .. } = &mut self.transaction.operation {
            if !new_indices
                .iter()
                .any(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
            {
                return Ok(self.transaction);
            }

            if self.conflicting_frag_reuse_indices.is_empty() {
                return Ok(self.transaction);
            }

            // had at least 1 previous rewrite conflict
            // get the max reuse version from each run to be added to the cleaned up index
            let mut max_versions = Vec::with_capacity(self.conflicting_frag_reuse_indices.len());
            for committed_fri in &self.conflicting_frag_reuse_indices {
                let committed_fri_details = Arc::try_unwrap(
                    load_frag_reuse_index_details(dataset, committed_fri)
                        .await
                        .unwrap(),
                )
                .unwrap();
                let max_version = committed_fri_details
                    .versions
                    .into_iter()
                    .max_by_key(|v| v.dataset_version)
                    .unwrap();
                max_versions.push(max_version);
            }

            // there should be only 1 frag_reuse_index in new indices
            let new_fri = &new_indices[0];
            let mut new_fri_details = Arc::try_unwrap(
                load_frag_reuse_index_details(dataset, new_fri)
                    .await
                    .unwrap(),
            )
            .unwrap();
            new_fri_details.versions.extend(max_versions);

            let new_frag_bitmap = new_fri_details.new_frag_bitmap();

            let new_frag_reuse_index_meta = build_frag_reuse_index_metadata(
                dataset,
                Some(new_fri),
                new_fri_details,
                new_frag_bitmap,
            )
            .await?;

            new_indices.retain(|idx| idx.name != FRAG_REUSE_INDEX_NAME);
            new_indices.push(new_frag_reuse_index_meta);
            Ok(self.transaction)
        } else {
            Err(wrong_operation_err(&self.transaction.operation))
        }
    }

    async fn finish_rewrite(mut self, dataset: &Dataset) -> Result<Transaction> {
        if let Operation::Rewrite {
            frag_reuse_index, ..
        } = &mut self.transaction.operation
        {
            if let Some(new_fri) = frag_reuse_index {
                if self.conflicting_frag_reuse_indices.is_empty() {
                    return Ok(self.transaction);
                }

                let mut new_fri_details = Arc::try_unwrap(
                    load_frag_reuse_index_details(dataset, new_fri)
                        .await
                        .unwrap(),
                )
                .unwrap();
                let mut min_dataset_version = new_fri_details
                    .versions
                    .iter()
                    .map(|v| v.dataset_version)
                    .min()
                    .unwrap();
                for committed_fri in self.conflicting_frag_reuse_indices.into_iter() {
                    let committed_fri_details =
                        load_frag_reuse_index_details(dataset, &committed_fri)
                            .await
                            .unwrap();
                    let committed_min_dataset_version = committed_fri_details
                        .versions
                        .iter()
                        .map(|v| v.dataset_version)
                        .min();

                    // For example, if we have new_fri has reuse versions [1, 2, 3]
                    // If committed_fri has versions [2], that means 1 is cleaned up,
                    // then [2, 3] should be retained in the new_fri.
                    // If committed_fri is empty, that means everything is cleaned up.
                    // then only the last item in committed_fri should be retained, which is [3].
                    // Note that this is under the assumption that the sequence of
                    // conflicting_frag_reuse_indices all come from frag_reuse_index cleanup rebase.
                    match committed_min_dataset_version {
                        Some(committed_min_dataset_version) => {
                            if committed_min_dataset_version > min_dataset_version {
                                min_dataset_version = committed_min_dataset_version;
                            }
                        }
                        None => {
                            min_dataset_version = new_fri_details
                                .versions
                                .iter()
                                .map(|v| v.dataset_version)
                                .max()
                                .unwrap();
                        }
                    }
                }

                new_fri_details
                    .versions
                    .retain(|v| v.dataset_version >= min_dataset_version);
                let new_frag_bitmap = new_fri_details.new_frag_bitmap();

                let new_frag_reuse_index_meta = build_frag_reuse_index_metadata(
                    dataset,
                    Some(new_fri),
                    new_fri_details,
                    new_frag_bitmap,
                )
                .await?;

                *frag_reuse_index = Some(new_frag_reuse_index_meta);
                Ok(self.transaction)
            } else {
                Ok(self.transaction)
            }
        } else {
            Err(wrong_operation_err(&self.transaction.operation))
        }
    }
}

async fn initial_fragments_for_rebase(
    dataset: &Dataset,
    transaction: &Transaction,
    modified_fragment_ids: &HashSet<u64>,
) -> HashMap<u64, (Fragment, bool)> {
    if modified_fragment_ids.is_empty() {
        return HashMap::new();
    }

    let dataset = if dataset.manifest.version != transaction.read_version {
        Cow::Owned(
            dataset
                .checkout_version(transaction.read_version)
                .await
                .unwrap(),
        )
    } else {
        Cow::Borrowed(dataset)
    };

    dataset
        .fragments()
        .iter()
        .filter(|fragment| {
            // Check if the fragment is modified by the transaction.
            modified_fragment_ids.contains(&fragment.id)
        })
        .map(|fragment| (fragment.id, (fragment.clone(), false)))
        .collect::<HashMap<_, _>>()
}

fn wrong_operation_err(op: &Operation) -> Error {
    Error::Internal {
        message: format!("function called against a wrong operation: {}", op),
        location: location!(),
    }
}

#[cfg(test)]
mod tests {
    use std::{num::NonZero, sync::Arc};

    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use lance_core::Error;
    use lance_file::version::LanceFileVersion;
    use lance_io::assert_io_eq;
    use lance_io::object_store::ObjectStoreParams;
    use lance_io::utils::tracking_store::IOTracker;
    use lance_table::format::IndexMetadata;
    use lance_table::io::deletion::{deletion_file_path, read_deletion_file};

    use super::*;
    use crate::dataset::transaction::RewriteGroup;
    use crate::session::caches::DeletionFileKey;
    use crate::{
        dataset::{CommitBuilder, InsertBuilder, WriteParams},
        io,
    };

    async fn test_dataset(num_rows: usize, num_fragments: usize) -> (Dataset, Arc<IOTracker>) {
        let io_tracker = Arc::new(IOTracker::default());
        let write_params = WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(io_tracker.clone()),
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
        (dataset, io_tracker)
    }

    /// Helper function for tests to create UpdateConfig operations using old-style parameters
    #[cfg(test)]
    fn create_update_config_for_test(
        upsert_values: Option<HashMap<String, String>>,
        delete_keys: Option<Vec<String>>,
        schema_metadata: Option<HashMap<String, String>>,
        field_metadata: Option<HashMap<u32, HashMap<String, String>>>,
    ) -> Operation {
        use crate::dataset::transaction::{
            translate_config_updates, translate_schema_metadata_updates,
        };

        let config_updates = if let Some(upsert) = &upsert_values {
            if let Some(delete) = &delete_keys {
                Some(translate_config_updates(upsert, delete))
            } else {
                Some(translate_config_updates(upsert, &[]))
            }
        } else {
            delete_keys
                .as_ref()
                .map(|delete| translate_config_updates(&HashMap::new(), delete))
        };

        let schema_metadata_updates = schema_metadata
            .as_ref()
            .map(translate_schema_metadata_updates);

        let field_metadata_updates = field_metadata
            .unwrap_or_default()
            .into_iter()
            .map(|(field_id, metadata)| {
                (
                    field_id as i32,
                    translate_schema_metadata_updates(&metadata),
                )
            })
            .collect();

        Operation::UpdateConfig {
            config_updates,
            table_metadata_updates: None,
            schema_metadata_updates,
            field_metadata_updates,
        }
    }

    #[tokio::test]
    async fn test_non_overlapping_rebase_delete_update() {
        let (dataset, io_tracker) = test_dataset(5, 5).await;
        let operation = Operation::Update {
            updated_fragments: vec![Fragment::new(0)],
            removed_fragment_ids: vec![],
            new_fragments: vec![],
            fields_modified: vec![],
            mem_wal_to_merge: None,
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: None,
        };
        let transaction = Transaction::new_from_version(1, operation);
        let other_operations = [
            Operation::Update {
                updated_fragments: vec![Fragment::new(1)],
                removed_fragment_ids: vec![2],
                new_fragments: vec![],
                fields_modified: vec![],
                mem_wal_to_merge: None,
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: None,
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
                mem_wal_to_merge: None,
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: None,
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
            assert_io_eq!(io_stats, read_iops, 0);
            assert_io_eq!(io_stats, write_iops, 0);
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
        assert_io_eq!(io_stats, read_iops, 0);
        assert_io_eq!(io_stats, write_iops, 0);
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
                // Reference deletion file should never enter this apply_deletion. So base path is fine.
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

        let deletion_file = fragment.deletion_file.as_ref().unwrap();
        let key = DeletionFileKey {
            fragment_id: fragment.id,
            deletion_file,
        };
        dataset
            .metadata_cache
            .insert_with_key(&key, Arc::new(current_deletions))
            .await;

        fragment.clone()
    }

    #[tokio::test]
    #[rstest::rstest]
    async fn test_non_conflicting_rebase_delete_update() {
        // 5 rows, all in one fragment. Each transaction modifies a different row.
        let (mut dataset, io_tracker) = test_dataset(5, 1).await;
        let mut fragment = dataset.fragments().as_slice()[0].clone();

        // Other operations modify the 1st, 2nd, and 3rd rows sequentially.
        let sample_file = Fragment::new(0)
            .with_file(
                "path1",
                vec![0],
                vec![0],
                &LanceFileVersion::Stable,
                NonZero::new(10),
            )
            .with_physical_rows(3);
        let operations = [
            Operation::Update {
                updated_fragments: vec![apply_deletion(&[0], &mut fragment, &dataset).await],
                removed_fragment_ids: vec![],
                new_fragments: vec![sample_file.clone()],
                fields_modified: vec![],
                mem_wal_to_merge: None,
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: None,
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
                mem_wal_to_merge: None,
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: None,
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
                assert_io_eq!(io_stats, read_iops, 0);
                assert_io_eq!(io_stats, write_iops, 0);
            }

            // First iteration, we don't need to rewrite the deletion file.
            let expected_rewrite = i > 0;

            let rebased_transaction = rebase.finish(&dataset).await.unwrap();
            assert_eq!(rebased_transaction.read_version, dataset.manifest.version);

            let io_stats = io_tracker.incremental_stats();
            if expected_rewrite {
                // Read the current deletion file, and write the new one.
                assert_io_eq!(io_stats, read_iops, 0, "deletion file should be cached");
                assert_io_eq!(io_stats, write_iops, 1, "write one deletion file");

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

                assert_io_eq!(io_stats, num_hops, 1);
            } else {
                // No IO should have happened.
                assert_io_eq!(io_stats, read_iops, 0);
                assert_io_eq!(io_stats, write_iops, 0);
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
                &LanceFileVersion::Stable,
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
                    mem_wal_to_merge: None,
                    fields_for_preserving_frag_bitmap: vec![],
                    update_mode: None,
                },
            ),
            (
                "update_partial",
                Operation::Update {
                    updated_fragments: vec![apply_deletion(&[0], &mut fragment, &dataset).await],
                    removed_fragment_ids: vec![],
                    new_fragments: vec![sample_file.clone()],
                    fields_modified: vec![],
                    mem_wal_to_merge: None,
                    fields_for_preserving_frag_bitmap: vec![],
                    update_mode: None,
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
        assert_io_eq!(io_stats, read_iops, 0);
        assert_io_eq!(io_stats, write_iops, 0);

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
        assert_io_eq!(io_stats, read_iops, 0);
        assert_io_eq!(io_stats, write_iops, 0);

        let res = rebase.finish(&latest_dataset).await;
        assert!(matches!(
            res,
            Err(crate::Error::RetryableCommitConflict { .. })
        ));

        let io_stats = io_tracker.incremental_stats();
        assert_io_eq!(io_stats, read_iops, 0, "deletion file should be cached");
        assert_io_eq!(io_stats, write_iops, 0, "failed before writing");
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum ConflictResult {
        Compatible,
        NotCompatible,
        Retryable,
    }

    #[test]
    fn test_conflicts() {
        use io::commit::conflict_resolver::tests::{modified_fragment_ids, ConflictResult::*};

        let index0 = IndexMetadata {
            uuid: uuid::Uuid::new_v4(),
            name: "test".to_string(),
            fields: vec![0],
            dataset_version: 1,
            fragment_bitmap: None,
            index_details: None,
            index_version: 0,
            created_at: None, // Test index, not setting timestamp
            base_id: None,
        };
        let fragment0 = Fragment::new(0);
        let fragment1 = Fragment::new(1);
        let fragment2 = Fragment::new(2);
        // The transactions that will be checked against
        let other_operations = [
            Operation::Append {
                fragments: vec![fragment0.clone()],
            },
            Operation::CreateIndex {
                new_indices: vec![index0.clone()],
                removed_indices: vec![index0.clone()],
            },
            Operation::Delete {
                updated_fragments: vec![fragment0.clone()],
                deleted_fragment_ids: vec![2],
                predicate: "x > 2".to_string(),
            },
            Operation::Merge {
                fragments: vec![fragment0.clone(), fragment2.clone()],
                schema: lance_core::datatypes::Schema::default(),
            },
            Operation::Overwrite {
                fragments: vec![fragment0.clone(), fragment2.clone()],
                schema: lance_core::datatypes::Schema::default(),
                config_upsert_values: Some(HashMap::from_iter(vec![(
                    "overwrite-key".to_string(),
                    "value".to_string(),
                )])),
                initial_bases: None,
            },
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments: vec![fragment0.clone()],
                    new_fragments: vec![fragment1.clone()],
                }],
                rewritten_indices: vec![],
                frag_reuse_index: None,
            },
            Operation::ReserveFragments { num_fragments: 3 },
            Operation::Update {
                removed_fragment_ids: vec![1],
                updated_fragments: vec![fragment0.clone()],
                new_fragments: vec![fragment2.clone()],
                fields_modified: vec![0],
                mem_wal_to_merge: None,
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: None,
            },
            create_update_config_for_test(
                Some(HashMap::from_iter(vec![(
                    "lance.test".to_string(),
                    "value".to_string(),
                )])),
                Some(vec!["remove-key".to_string()]),
                Some(HashMap::from_iter(vec![(
                    "schema-key".to_string(),
                    "schema-value".to_string(),
                )])),
                Some(HashMap::from_iter(vec![(
                    0,
                    HashMap::from_iter(vec![("field-key".to_string(), "field-value".to_string())]),
                )])),
            ),
        ];
        let other_transactions = other_operations
            .iter()
            .map(|op| Transaction::new(0, op.clone(), None))
            .collect::<Vec<_>>();

        // Transactions and whether they are expected to conflict with each
        // of other_transactions
        let cases = [
            (
                Operation::Append {
                    fragments: vec![fragment0.clone()],
                },
                [
                    Compatible,    // append
                    Compatible,    // create index
                    Compatible,    // delete
                    Compatible,    // merge
                    NotCompatible, // overwrite
                    Compatible,    // rewrite
                    Compatible,    // reserve
                    Compatible,    // update
                    Compatible,    // update config
                ],
            ),
            (
                Operation::Delete {
                    // Delete that affects fragments different from other transactions
                    updated_fragments: vec![fragment1.clone()],
                    deleted_fragment_ids: vec![],
                    predicate: "x > 2".to_string(),
                },
                [
                    Compatible,    // append
                    Compatible,    // create index
                    Compatible,    // delete
                    Retryable,     // merge
                    NotCompatible, // overwrite
                    Compatible,    // rewrite
                    Compatible,    // reserve
                    Retryable,     // update
                    Compatible,    // update config
                ],
            ),
            (
                Operation::Delete {
                    // Delete that affects same fragments as other transactions
                    updated_fragments: vec![fragment0.clone(), fragment2.clone()],
                    deleted_fragment_ids: vec![],
                    predicate: "x > 2".to_string(),
                },
                [
                    Compatible,    // append
                    Compatible,    // create index
                    Retryable,     // delete
                    Retryable,     // merge
                    NotCompatible, // overwrite
                    Retryable,     // rewrite
                    Compatible,    // reserve
                    Retryable,     // update
                    Compatible,    // update config
                ],
            ),
            (
                Operation::Overwrite {
                    fragments: vec![fragment0.clone(), fragment2.clone()],
                    schema: lance_core::datatypes::Schema::default(),
                    config_upsert_values: None,
                    initial_bases: None,
                },
                // No conflicts: overwrite can always happen since it doesn't
                // depend on previous state of the table.
                [Compatible; 9],
            ),
            (
                Operation::CreateIndex {
                    new_indices: vec![index0.clone()],
                    removed_indices: vec![index0],
                },
                // Will only conflict with operations that modify row ids.
                [
                    Compatible,    // append
                    Compatible,    // create index
                    Compatible,    // delete
                    Compatible,    // merge
                    NotCompatible, // overwrite
                    Retryable,     // rewrite
                    Compatible,    // reserve
                    Compatible,    // update
                    Compatible,    // update config
                ],
            ),
            (
                // Rewrite that affects different fragments
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments: vec![fragment1],
                        new_fragments: vec![fragment0.clone()],
                    }],
                    rewritten_indices: Vec::new(),
                    frag_reuse_index: None,
                },
                [
                    Compatible,    // append
                    Retryable,     // create index
                    Compatible,    // delete
                    Retryable,     // merge
                    NotCompatible, // overwrite
                    Compatible,    // rewrite
                    Compatible,    // reserve
                    Retryable,     // update
                    Compatible,    // update config
                ],
            ),
            (
                // Rewrite that affects the same fragments
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments: vec![fragment0.clone(), fragment2.clone()],
                        new_fragments: vec![fragment0.clone()],
                    }],
                    rewritten_indices: Vec::new(),
                    frag_reuse_index: None,
                },
                [
                    Compatible,    // append
                    Retryable,     // create index
                    Retryable,     // delete
                    Retryable,     // merge
                    NotCompatible, // overwrite
                    Retryable,     // rewrite
                    Compatible,    // reserve
                    Retryable,     // update
                    Compatible,    // update config
                ],
            ),
            (
                Operation::Merge {
                    fragments: vec![fragment0.clone(), fragment2.clone()],
                    schema: lance_core::datatypes::Schema::default(),
                },
                // Merge conflicts with everything except CreateIndex and ReserveFragments.
                [
                    Retryable,     // append
                    Compatible,    // create index
                    Retryable,     // delete
                    Retryable,     // merge
                    NotCompatible, // overwrite
                    Retryable,     // rewrite
                    Compatible,    // reserve
                    Retryable,     // update
                    Compatible,    // update config
                ],
            ),
            (
                Operation::ReserveFragments { num_fragments: 2 },
                // ReserveFragments only conflicts with Overwrite and Restore.
                [
                    Compatible,    // append
                    Compatible,    // create index
                    Compatible,    // delete
                    Compatible,    // merge
                    NotCompatible, // overwrite
                    Compatible,    // rewrite
                    Compatible,    // reserve
                    Compatible,    // update
                    Compatible,    // update config
                ],
            ),
            (
                Operation::Update {
                    // Update that affects same fragments as other transactions
                    updated_fragments: vec![fragment0],
                    removed_fragment_ids: vec![],
                    new_fragments: vec![fragment2],
                    fields_modified: vec![0],
                    mem_wal_to_merge: None,
                    fields_for_preserving_frag_bitmap: vec![],
                    update_mode: None,
                },
                [
                    Compatible,    // append
                    Compatible,    // create index
                    Retryable,     // delete
                    Retryable,     // merge
                    NotCompatible, // overwrite
                    Retryable,     // rewrite
                    Compatible,    // reserve
                    Retryable,     // update
                    Compatible,    // update config
                ],
            ),
            (
                // Update config that should not conflict with anything
                create_update_config_for_test(
                    Some(HashMap::from_iter(vec![(
                        "other-key".to_string(),
                        "new-value".to_string(),
                    )])),
                    None,
                    None,
                    None,
                ),
                [Compatible; 9],
            ),
            (
                // Update config that conflicts with key being upserted by other UpdateConfig operation
                create_update_config_for_test(
                    Some(HashMap::from_iter(vec![(
                        "lance.test".to_string(),
                        "new-value".to_string(),
                    )])),
                    None,
                    None,
                    None,
                ),
                [
                    Compatible,    // append
                    Compatible,    // create index
                    Compatible,    // delete
                    Compatible,    // merge
                    Compatible,    // overwrite
                    Compatible,    // rewrite
                    Compatible,    // reserve
                    Compatible,    // update
                    NotCompatible, // update config
                ],
            ),
            (
                // Update config that conflicts with key being deleted by other UpdateConfig operation
                create_update_config_for_test(
                    Some(HashMap::from_iter(vec![(
                        "remove-key".to_string(),
                        "new-value".to_string(),
                    )])),
                    None,
                    None,
                    None,
                ),
                [
                    Compatible,    // append
                    Compatible,    // create index
                    Compatible,    // delete
                    Compatible,    // merge
                    Compatible,    // overwrite
                    Compatible,    // rewrite
                    Compatible,    // reserve
                    Compatible,    // update
                    NotCompatible, // update config
                ],
            ),
            (
                // Delete config keys currently being deleted by other UpdateConfig operation
                create_update_config_for_test(
                    None,
                    Some(vec!["remove-key".to_string()]),
                    None,
                    None,
                ),
                [Compatible; 9],
            ),
            (
                // Delete config keys currently being upserted by other UpdateConfig operation
                create_update_config_for_test(
                    None,
                    Some(vec!["lance.test".to_string()]),
                    None,
                    None,
                ),
                [
                    Compatible,    // append
                    Compatible,    // create index
                    Compatible,    // delete
                    Compatible,    // merge
                    Compatible,    // overwrite
                    Compatible,    // rewrite
                    Compatible,    // reserve
                    Compatible,    // update
                    NotCompatible, // update config
                ],
            ),
            (
                // Changing schema metadata conflicts with another update changing schema
                // metadata or with an overwrite
                create_update_config_for_test(
                    None,
                    None,
                    Some(HashMap::from_iter(vec![(
                        "schema-key".to_string(),
                        "new-value".to_string(),
                    )])),
                    None,
                ),
                [
                    Compatible,    // append
                    Compatible,    // create index
                    Compatible,    // delete
                    Compatible,    // merge
                    NotCompatible, // overwrite
                    Compatible,    // rewrite
                    Compatible,    // reserve
                    Compatible,    // update
                    NotCompatible, // update config
                ],
            ),
            (
                // Changing field metadata conflicts with another update changing same field
                // metadata or overwrite
                create_update_config_for_test(
                    None,
                    None,
                    None,
                    Some(HashMap::from_iter(vec![(
                        0,
                        HashMap::from_iter(vec![(
                            "field_key".to_string(),
                            "field_value".to_string(),
                        )]),
                    )])),
                ),
                [
                    Compatible,    // append
                    Compatible,    // create index
                    Compatible,    // delete
                    Compatible,    // merge
                    NotCompatible, // overwrite
                    Compatible,    // rewrite
                    Compatible,    // reserve
                    Compatible,    // update
                    NotCompatible, // update config
                ],
            ),
            (
                // Updates to different field metadata are allowed
                create_update_config_for_test(
                    None,
                    None,
                    None,
                    Some(HashMap::from_iter(vec![(
                        1,
                        HashMap::from_iter(vec![(
                            "field_key".to_string(),
                            "field_value".to_string(),
                        )]),
                    )])),
                ),
                [
                    Compatible,    // append
                    Compatible,    // create index
                    Compatible,    // delete
                    Compatible,    // merge
                    NotCompatible, // overwrite
                    Compatible,    // rewrite
                    Compatible,    // reserve
                    Compatible,    // update
                    Compatible,    // update config
                ],
            ),
        ];

        for (operation, expected_conflicts) in &cases {
            let transaction = Transaction::new(0, operation.clone(), None);
            let mut rebase = TransactionRebase {
                transaction,
                initial_fragments: HashMap::new(),
                modified_fragment_ids: modified_fragment_ids(operation).collect::<HashSet<_>>(),
                affected_rows: None,
                conflicting_frag_reuse_indices: Vec::new(),
            };

            for (other, expected_conflict) in other_transactions.iter().zip(expected_conflicts) {
                match expected_conflict {
                    Compatible => {
                        let result = rebase.check_txn(other, 1);
                        assert!(
                            result.is_ok(),
                            "Transaction {:?} should {:?} with {:?}, but was {:?}",
                            operation,
                            expected_conflict,
                            other,
                            result
                        )
                    }
                    NotCompatible => {
                        let result = rebase.check_txn(other, 1);
                        assert!(
                            matches!(result, Err(Error::CommitConflict { .. })),
                            "Transaction {:?} should be {:?} with {:?}, but was: {:?}",
                            operation,
                            expected_conflict,
                            other,
                            result
                        )
                    }
                    Retryable => {
                        let result = rebase.check_txn(other, 1);
                        assert!(
                            matches!(result, Err(Error::RetryableCommitConflict { .. })),
                            "Transaction {:?} should be {:?} with {:?}, but was {:?}",
                            operation,
                            expected_conflict,
                            other,
                            result
                        )
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_add_bases_non_conflicting() {
        let (dataset, _) = test_dataset(10, 2).await;

        // Create two transactions adding different bases
        let txn1 = Transaction::new_from_version(
            1,
            Operation::UpdateBases {
                new_bases: vec![lance_table::format::BasePath {
                    id: 1,
                    path: "s3://bucket1/path1".to_string(),
                    name: Some("base1".to_string()),
                    is_dataset_root: false,
                }],
            },
        );

        let txn2 = Transaction::new_from_version(
            1,
            Operation::UpdateBases {
                new_bases: vec![lance_table::format::BasePath {
                    id: 2,
                    path: "s3://bucket2/path2".to_string(),
                    name: Some("base2".to_string()),
                    is_dataset_root: false,
                }],
            },
        );

        // txn1 should not conflict with txn2
        let mut rebase = TransactionRebase::try_new(&dataset, txn1, None)
            .await
            .unwrap();
        assert!(rebase.check_txn(&txn2, 2).is_ok());
    }

    #[tokio::test]
    async fn test_add_bases_name_conflict() {
        let (dataset, _) = test_dataset(10, 2).await;

        // Create two transactions adding bases with the same name
        let txn1 = Transaction::new_from_version(
            1,
            Operation::UpdateBases {
                new_bases: vec![lance_table::format::BasePath {
                    id: 1,
                    path: "s3://bucket1/path1".to_string(),
                    name: Some("duplicate_name".to_string()),
                    is_dataset_root: false,
                }],
            },
        );

        let txn2 = Transaction::new_from_version(
            1,
            Operation::UpdateBases {
                new_bases: vec![lance_table::format::BasePath {
                    id: 2,
                    path: "s3://bucket2/path2".to_string(),
                    name: Some("duplicate_name".to_string()),
                    is_dataset_root: false,
                }],
            },
        );

        // txn1 should conflict with txn2 due to duplicate name
        let mut rebase = TransactionRebase::try_new(&dataset, txn1, None)
            .await
            .unwrap();
        let result = rebase.check_txn(&txn2, 2);
        assert!(
            matches!(result, Err(Error::CommitConflict { .. })),
            "Expected CommitConflict error for duplicate name, got {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_add_bases_path_conflict() {
        let (dataset, _) = test_dataset(10, 2).await;

        // Create two transactions adding bases with the same path
        let txn1 = Transaction::new_from_version(
            1,
            Operation::UpdateBases {
                new_bases: vec![lance_table::format::BasePath {
                    id: 1,
                    path: "s3://bucket/duplicate_path".to_string(),
                    name: Some("base1".to_string()),
                    is_dataset_root: false,
                }],
            },
        );

        let txn2 = Transaction::new_from_version(
            1,
            Operation::UpdateBases {
                new_bases: vec![lance_table::format::BasePath {
                    id: 2,
                    path: "s3://bucket/duplicate_path".to_string(),
                    name: Some("base2".to_string()),
                    is_dataset_root: false,
                }],
            },
        );

        // txn1 should conflict with txn2 due to duplicate path
        let mut rebase = TransactionRebase::try_new(&dataset, txn1, None)
            .await
            .unwrap();
        let result = rebase.check_txn(&txn2, 2);
        assert!(
            matches!(result, Err(Error::CommitConflict { .. })),
            "Expected CommitConflict error for duplicate path, got {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_add_bases_id_conflict() {
        let (dataset, _) = test_dataset(10, 2).await;

        // Create two transactions adding bases with the same non-zero ID
        let txn1 = Transaction::new_from_version(
            1,
            Operation::UpdateBases {
                new_bases: vec![lance_table::format::BasePath {
                    id: 42,
                    path: "s3://bucket1/path1".to_string(),
                    name: Some("base1".to_string()),
                    is_dataset_root: false,
                }],
            },
        );

        let txn2 = Transaction::new_from_version(
            1,
            Operation::UpdateBases {
                new_bases: vec![lance_table::format::BasePath {
                    id: 42,
                    path: "s3://bucket2/path2".to_string(),
                    name: Some("base2".to_string()),
                    is_dataset_root: false,
                }],
            },
        );

        // txn1 should conflict with txn2 due to duplicate non-zero ID
        let mut rebase = TransactionRebase::try_new(&dataset, txn1, None)
            .await
            .unwrap();
        let result = rebase.check_txn(&txn2, 2);
        assert!(
            matches!(result, Err(Error::CommitConflict { .. })),
            "Expected CommitConflict error for duplicate ID, got {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_add_bases_no_conflict_with_data_operations() {
        let (dataset, _) = test_dataset(10, 2).await;

        let add_bases_txn = Transaction::new_from_version(
            1,
            Operation::UpdateBases {
                new_bases: vec![lance_table::format::BasePath {
                    id: 1,
                    path: "s3://bucket/path".to_string(),
                    name: Some("base1".to_string()),
                    is_dataset_root: false,
                }],
            },
        );

        // Test against various data operations
        let data_operations = vec![
            Operation::Append { fragments: vec![] },
            Operation::Delete {
                deleted_fragment_ids: vec![0],
                updated_fragments: vec![],
                predicate: "a > 5".to_string(),
            },
            Operation::Update {
                updated_fragments: vec![Fragment::new(0)],
                removed_fragment_ids: vec![],
                new_fragments: vec![],
                fields_modified: vec![],
                mem_wal_to_merge: None,
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: None,
            },
        ];

        for operation in data_operations {
            let data_txn = Transaction::new_from_version(1, operation.clone());
            let mut rebase = TransactionRebase::try_new(&dataset, add_bases_txn.clone(), None)
                .await
                .unwrap();
            assert!(
                rebase.check_txn(&data_txn, 2).is_ok(),
                "UpdateBases should not conflict with {:?}",
                operation
            );
        }
    }

    #[tokio::test]
    async fn test_add_bases_multiple_bases() {
        let (dataset, _) = test_dataset(10, 2).await;

        // txn1 adds two bases
        let txn1 = Transaction::new_from_version(
            1,
            Operation::UpdateBases {
                new_bases: vec![
                    lance_table::format::BasePath {
                        id: 1,
                        path: "s3://bucket1/path1".to_string(),
                        name: Some("base1".to_string()),
                        is_dataset_root: false,
                    },
                    lance_table::format::BasePath {
                        id: 2,
                        path: "s3://bucket2/path2".to_string(),
                        name: Some("base2".to_string()),
                        is_dataset_root: false,
                    },
                ],
            },
        );

        // txn2 adds a base that conflicts with one of txn1's bases
        let txn2 = Transaction::new_from_version(
            1,
            Operation::UpdateBases {
                new_bases: vec![lance_table::format::BasePath {
                    id: 3,
                    path: "s3://bucket1/path1".to_string(), // Same path as txn1's first base
                    name: Some("base3".to_string()),
                    is_dataset_root: false,
                }],
            },
        );

        // Should conflict due to path conflict
        let mut rebase = TransactionRebase::try_new(&dataset, txn1, None)
            .await
            .unwrap();
        let result = rebase.check_txn(&txn2, 2);
        assert!(
            matches!(result, Err(Error::CommitConflict { .. })),
            "Expected CommitConflict error, got {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_add_bases_with_none_name() {
        let (dataset, _) = test_dataset(10, 2).await;

        // Bases with None names should not conflict on name
        let txn1 = Transaction::new_from_version(
            1,
            Operation::UpdateBases {
                new_bases: vec![lance_table::format::BasePath {
                    id: 1,
                    path: "s3://bucket1/path1".to_string(),
                    name: None,
                    is_dataset_root: false,
                }],
            },
        );

        let txn2 = Transaction::new_from_version(
            1,
            Operation::UpdateBases {
                new_bases: vec![lance_table::format::BasePath {
                    id: 2,
                    path: "s3://bucket2/path2".to_string(),
                    name: None,
                    is_dataset_root: false,
                }],
            },
        );

        // Should not conflict despite both having None names
        let mut rebase = TransactionRebase::try_new(&dataset, txn1, None)
            .await
            .unwrap();
        assert!(rebase.check_txn(&txn2, 2).is_ok());
    }

    #[tokio::test]
    async fn test_add_bases_with_zero_id() {
        let (dataset, _) = test_dataset(10, 2).await;

        // Bases with zero IDs should not conflict on ID
        let txn1 = Transaction::new_from_version(
            1,
            Operation::UpdateBases {
                new_bases: vec![lance_table::format::BasePath {
                    id: 0,
                    path: "s3://bucket1/path1".to_string(),
                    name: Some("base1".to_string()),
                    is_dataset_root: false,
                }],
            },
        );

        let txn2 = Transaction::new_from_version(
            1,
            Operation::UpdateBases {
                new_bases: vec![lance_table::format::BasePath {
                    id: 0,
                    path: "s3://bucket2/path2".to_string(),
                    name: Some("base2".to_string()),
                    is_dataset_root: false,
                }],
            },
        );

        // Should not conflict despite both having zero IDs
        let mut rebase = TransactionRebase::try_new(&dataset, txn1, None)
            .await
            .unwrap();
        assert!(rebase.check_txn(&txn2, 2).is_ok());
    }

    /// Returns the IDs of fragments that have been modified by this operation.
    ///
    /// This does not include new fragments.
    fn modified_fragment_ids(operation: &Operation) -> Box<dyn Iterator<Item = u64> + '_> {
        match operation {
            // These operations add new fragments or don't modify any.
            Operation::Append { .. }
            | Operation::Clone { .. }
            | Operation::Overwrite { .. }
            | Operation::CreateIndex { .. }
            | Operation::ReserveFragments { .. }
            | Operation::Project { .. }
            | Operation::UpdateConfig { .. }
            | Operation::UpdateBases { .. }
            | Operation::Restore { .. }
            | Operation::UpdateMemWalState { .. } => Box::new(std::iter::empty()),
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids,
                ..
            } => Box::new(
                updated_fragments
                    .iter()
                    .map(|f| f.id)
                    .chain(deleted_fragment_ids.iter().copied()),
            ),
            Operation::Rewrite { groups, .. } => Box::new(
                groups
                    .iter()
                    .flat_map(|f| f.old_fragments.iter().map(|f| f.id)),
            ),
            Operation::Merge { fragments, .. } => Box::new(fragments.iter().map(|f| f.id)),
            Operation::Update {
                updated_fragments,
                removed_fragment_ids,
                ..
            } => Box::new(
                updated_fragments
                    .iter()
                    .map(|f| f.id)
                    .chain(removed_fragment_ids.iter().copied()),
            ),
            Operation::DataReplacement { replacements } => {
                Box::new(replacements.iter().map(|r| r.0))
            }
        }
    }
}
