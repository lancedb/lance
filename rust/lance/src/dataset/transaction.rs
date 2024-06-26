// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Transaction definitions for updating datasets
//!
//! Prior to creating a new manifest, a transaction must be created representing
//! the changes being made to the dataset. By representing them as incremental
//! changes, we can detect whether concurrent operations are compatible with
//! one another. We can also rebuild manifests when retrying committing a
//! manifest.
//!
//! ## Conflict Resolution
//!
//! Transactions are compatible with one another if they don't conflict.
//! Currently, conflict resolution always assumes a Serializable isolation
//! level.
//!
//! Below are the compatibilities between conflicting transactions. The columns
//! represent the operation that has been applied, while the rows represent the
//! operation that is being checked for compatibility to see if it can retry.
//! ✅ indicates that the operation is compatible, while ❌ indicates that it is
//! a conflict. Some operations have additional conditions that must be met for
//! them to be compatible.
//!
//! |                  | Append | Delete | Overwrite/Create | Create Index | Rewrite | Merge | Project |
//! |------------------|--------|--------|------------------|--------------|---------|-------|---------|
//! | Append           | ✅     | ✅     | ❌               | ✅           | ✅      | ❌    | ❌      |
//! | Delete           | ❌     | (1)    | ❌               | ✅           | (1)     | ❌    | ❌      |
//! | Overwrite/Create | ✅     | ✅     | ✅               | ✅           | ✅      | ✅    | ✅      |
//! | Create index     | ✅     | ✅     | ❌               | ✅           | ✅      | ✅    | ✅      |
//! | Rewrite          | ✅     | (1)    | ❌               | ❌           | (1)     | ❌    | ❌      |
//! | Merge            | ❌     | ❌     | ❌               | ❌           | ✅      | ❌    | ❌      |
//! | Project          | ✅     | ✅     | ❌               | ❌           | ✅      | ❌    | ✅      |
//!
//! (1) Delete and rewrite are compatible with each other and themselves only if
//! they affect distinct fragments. Otherwise, they conflict.

use std::{collections::HashSet, sync::Arc};

use lance_core::{datatypes::Schema, Error, Result};
use lance_file::datatypes::Fields;
use lance_io::object_store::ObjectStore;
use lance_table::{
    format::{
        pb::{self, IndexMetadata},
        Fragment, Index, Manifest, RowIdMeta,
    },
    io::{
        commit::CommitHandler,
        manifest::{read_manifest, read_manifest_indexes},
    },
    rowids::{write_row_ids, RowIdSequence},
};
use object_store::path::Path;
use roaring::RoaringBitmap;
use snafu::{location, Location};
use uuid::Uuid;

use super::ManifestWriteConfig;
use crate::utils::temporal::timestamp_to_nanos;
use lance_table::feature_flags::{apply_feature_flags, FLAG_MOVE_STABLE_ROW_IDS};

/// A change to a dataset that can be retried
///
/// This contains enough information to be able to build the next manifest,
/// given the current manifest.
#[derive(Debug, Clone)]
pub struct Transaction {
    /// The version of the table this transaction is based off of. If this is
    /// the first transaction, this should be 0.
    pub read_version: u64,
    pub uuid: String,
    pub operation: Operation,
    pub tag: Option<String>,
}

/// An operation on a dataset.
#[derive(Debug, Clone)]
pub enum Operation {
    /// Adding new fragments to the dataset. The fragments contained within
    /// haven't yet been assigned a final ID.
    Append { fragments: Vec<Fragment> },
    /// Updated fragments contain those that have been modified with new deletion
    /// files. The deleted fragment IDs are those that should be removed from
    /// the manifest.
    Delete {
        updated_fragments: Vec<Fragment>,
        deleted_fragment_ids: Vec<u64>,
        predicate: String,
    },
    /// Overwrite the entire dataset with the given fragments. This is also
    /// used when initially creating a table.
    Overwrite {
        fragments: Vec<Fragment>,
        schema: Schema,
    },
    /// A new index has been created.
    CreateIndex {
        /// The new secondary indices that are being added
        new_indices: Vec<Index>,
        /// The indices that have been modified.
        removed_indices: Vec<Index>,
    },
    /// Data is rewritten but *not* modified. This is used for things like
    /// compaction or re-ordering. Contains the old fragments and the new
    /// ones that have been replaced.
    ///
    /// This operation will modify the row addresses of existing rows and
    /// so any existing index covering a rewritten fragment will need to be
    /// remapped.
    Rewrite {
        /// Groups of fragments that have been modified
        groups: Vec<RewriteGroup>,
        /// Indices that have been updated with the new row addresses
        rewritten_indices: Vec<RewrittenIndex>,
    },
    /// Merge a new column in
    Merge {
        fragments: Vec<Fragment>,
        schema: Schema,
    },
    /// Restore an old version of the database
    Restore { version: u64 },
    /// Reserves fragment ids for future use
    /// This can be used when row ids need to be known before a transaction
    /// has been committed.  It is used during a rewrite operation to allow
    /// indices to be remapped to the new row ids as part of the operation.
    ReserveFragments { num_fragments: u32 },

    /// Update values in the dataset.
    Update {
        /// Ids of fragments that have been moved
        removed_fragment_ids: Vec<u64>,
        /// Fragments that have been updated
        updated_fragments: Vec<Fragment>,
        /// Fragments that have been added
        new_fragments: Vec<Fragment>,
    },

    /// Project to a new schema. This only changes the schema, not the data.
    Project { schema: Schema },
}

#[derive(Debug, Clone)]
pub struct RewrittenIndex {
    pub old_id: Uuid,
    pub new_id: Uuid,
}

#[derive(Debug, Clone)]
pub struct RewriteGroup {
    pub old_fragments: Vec<Fragment>,
    pub new_fragments: Vec<Fragment>,
}

impl Operation {
    /// Returns the IDs of fragments that have been modified by this operation.
    ///
    /// This does not include new fragments.
    fn modified_fragment_ids(&self) -> Box<dyn Iterator<Item = u64> + '_> {
        match self {
            // These operations add new fragments or don't modify any.
            Self::Append { .. }
            | Self::Overwrite { .. }
            | Self::CreateIndex { .. }
            | Self::ReserveFragments { .. }
            | Self::Project { .. }
            | Self::Restore { .. } => Box::new(std::iter::empty()),
            Self::Delete {
                updated_fragments,
                deleted_fragment_ids,
                ..
            } => Box::new(
                updated_fragments
                    .iter()
                    .map(|f| f.id)
                    .chain(deleted_fragment_ids.iter().copied()),
            ),
            Self::Rewrite { groups, .. } => Box::new(
                groups
                    .iter()
                    .flat_map(|f| f.old_fragments.iter().map(|f| f.id)),
            ),
            Self::Merge { fragments, .. } => Box::new(fragments.iter().map(|f| f.id)),
            Self::Update {
                updated_fragments,
                removed_fragment_ids,
                ..
            } => Box::new(
                updated_fragments
                    .iter()
                    .map(|f| f.id)
                    .chain(removed_fragment_ids.iter().copied()),
            ),
        }
    }

    /// Check whether another operation modifies the same fragment IDs as this one.
    fn modifies_same_ids(&self, other: &Self) -> bool {
        let self_ids = self.modified_fragment_ids().collect::<HashSet<_>>();
        let mut other_ids = other.modified_fragment_ids();
        other_ids.any(|id| self_ids.contains(&id))
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Append { .. } => "Append",
            Self::Delete { .. } => "Delete",
            Self::Overwrite { .. } => "Overwrite",
            Self::CreateIndex { .. } => "CreateIndex",
            Self::Rewrite { .. } => "Rewrite",
            Self::Merge { .. } => "Merge",
            Self::ReserveFragments { .. } => "ReserveFragments",
            Self::Restore { .. } => "Restore",
            Self::Update { .. } => "Update",
            Self::Project { .. } => "Project",
        }
    }
}

impl Transaction {
    pub fn new(read_version: u64, operation: Operation, tag: Option<String>) -> Self {
        let uuid = uuid::Uuid::new_v4().hyphenated().to_string();
        Self {
            read_version,
            uuid,
            operation,
            tag,
        }
    }

    /// Returns true if the transaction cannot be committed if the other
    /// transaction is committed first.
    pub fn conflicts_with(&self, other: &Self) -> bool {
        // This assumes IsolationLevel is Snapshot Isolation, which is more
        // permissive than Serializable. In particular, it allows a Delete
        // transaction to succeed after a concurrent Append, even if the Append
        // added rows that would be deleted.
        match &self.operation {
            Operation::Append { .. } => match &other.operation {
                // Append is compatible with anything that doesn't change the schema
                Operation::Append { .. } => false,
                Operation::Rewrite { .. } => false,
                Operation::CreateIndex { .. } => false,
                Operation::Delete { .. } | Operation::Update { .. } => false,
                Operation::ReserveFragments { .. } => false,
                Operation::Project { .. } => false,
                _ => true,
            },
            Operation::Rewrite { .. } => match &other.operation {
                // Rewrite is only compatible with operations that don't touch
                // existing fragments.
                // TODO: it could also be compatible with operations that update
                // fragments we don't touch.
                Operation::Append { .. } => false,
                Operation::ReserveFragments { .. } => false,
                Operation::Delete { .. } | Operation::Rewrite { .. } | Operation::Update { .. } => {
                    // As long as they rewrite disjoint fragments they shouldn't conflict.
                    self.operation.modifies_same_ids(&other.operation)
                }
                Operation::Project { .. } => false,
                _ => true,
            },
            // Overwrite and Restore always succeed
            Operation::Overwrite { .. } => false,
            Operation::Restore { .. } => false,
            // ReserveFragments is compatible with anything that doesn't reset the
            // max fragment id.
            Operation::ReserveFragments { .. } => matches!(
                &other.operation,
                Operation::Overwrite { .. } | Operation::Restore { .. }
            ),
            Operation::CreateIndex { .. } => match &other.operation {
                Operation::Append { .. } => false,
                // Indices are identified by UUIDs, so they shouldn't conflict.
                Operation::CreateIndex { .. } => false,
                // Although some of the rows we indexed may have been deleted / moved,
                // row ids are still valid, so we allow this optimistically.
                Operation::Delete { .. } | Operation::Update { .. } => false,
                // Merge & reserve don't change row ids, so this should be fine.
                Operation::Merge { .. } => false,
                Operation::ReserveFragments { .. } => false,
                // Rewrite likely changed many of the row ids, so our index is
                // likely useless. It should be rebuilt.
                // TODO: we could be smarter here and only invalidate the index
                // if the rewrite changed more than X% of row ids.
                Operation::Rewrite { .. } => true,
                _ => true,
            },
            Operation::Delete { .. } | Operation::Update { .. } => match &other.operation {
                Operation::CreateIndex { .. } => false,
                Operation::ReserveFragments { .. } => false,
                Operation::Delete { .. } | Operation::Rewrite { .. } | Operation::Update { .. } => {
                    // If we update the same fragments, we conflict.
                    self.operation.modifies_same_ids(&other.operation)
                }
                Operation::Project { .. } => false,
                Operation::Append { .. } => false,
                _ => true,
            },
            // Merge changes the schema, but preserves row ids, so the only operations
            // it's compatible with is CreateIndex and ReserveFragments.
            Operation::Merge { .. } => !matches!(
                &other.operation,
                Operation::CreateIndex { .. } | Operation::ReserveFragments { .. }
            ),
            Operation::Project { .. } => match &other.operation {
                // Project is compatible with anything that doesn't change the schema
                Operation::CreateIndex { .. } => false,
                Operation::Overwrite { .. } => false,
                _ => true,
            },
        }
    }

    fn fragments_with_ids<'a, T>(
        new_fragments: T,
        fragment_id: &'a mut u64,
    ) -> impl Iterator<Item = Fragment> + 'a
    where
        T: IntoIterator<Item = Fragment> + 'a,
    {
        new_fragments.into_iter().map(move |mut f| {
            if f.id == 0 {
                f.id = *fragment_id;
                *fragment_id += 1;
            }
            f
        })
    }

    pub(crate) async fn restore_old_manifest(
        object_store: &ObjectStore,
        commit_handler: &dyn CommitHandler,
        base_path: &Path,
        version: u64,
        config: &ManifestWriteConfig,
        tx_path: &str,
    ) -> Result<(Manifest, Vec<Index>)> {
        let path = commit_handler
            .resolve_version(base_path, version, &object_store.inner)
            .await?;
        let mut manifest = read_manifest(object_store, &path).await?;
        manifest.set_timestamp(timestamp_to_nanos(config.timestamp));
        manifest.transaction_file = Some(tx_path.to_string());
        let indices = read_manifest_indexes(object_store, &path, &manifest).await?;
        Ok((manifest, indices))
    }

    /// Create a new manifest from the current manifest and the transaction.
    ///
    /// `current_manifest` should only be None if the dataset does not yet exist.
    pub(crate) fn build_manifest(
        &self,
        current_manifest: Option<&Manifest>,
        current_indices: Vec<Index>,
        transaction_file_path: &str,
        config: &ManifestWriteConfig,
    ) -> Result<(Manifest, Vec<Index>)> {
        if config.use_move_stable_row_ids
            && current_manifest
                .map(|m| !m.uses_move_stable_row_ids())
                .unwrap_or_default()
        {
            return Err(Error::NotSupported {
                source: "Cannot enable stable row ids on existing dataset".into(),
                location: location!(),
            });
        }

        // Get the schema and the final fragment list
        let schema = match self.operation {
            Operation::Overwrite { ref schema, .. } => schema.clone(),
            Operation::Merge { ref schema, .. } => schema.clone(),
            Operation::Project { ref schema, .. } => schema.clone(),
            _ => {
                if let Some(current_manifest) = current_manifest {
                    current_manifest.schema.clone()
                } else {
                    return Err(Error::Internal {
                        message: "Cannot create a new dataset without a schema".to_string(),
                        location: location!(),
                    });
                }
            }
        };

        let mut fragment_id = if matches!(self.operation, Operation::Overwrite { .. }) {
            0
        } else {
            current_manifest
                .and_then(|m| m.max_fragment_id())
                .map(|id| id + 1)
                .unwrap_or(0)
        };
        let mut final_fragments = Vec::new();
        let mut final_indices = current_indices;

        let mut next_row_id = {
            // Only use row ids if the feature flag is set already or
            match (current_manifest, config.use_move_stable_row_ids) {
                (Some(manifest), _)
                    if manifest.reader_feature_flags & FLAG_MOVE_STABLE_ROW_IDS != 0 =>
                {
                    Some(manifest.next_row_id)
                }
                (None, true) => Some(0),
                (_, false) => None,
                (Some(_), true) => {
                    return Err(Error::NotSupported {
                        source: "Cannot enable stable row ids on existing dataset".into(),
                        location: location!(),
                    });
                }
            }
        };

        let maybe_existing_fragments =
            current_manifest
                .map(|m| m.fragments.as_ref())
                .ok_or_else(|| Error::Internal {
                    message: format!(
                        "No current manifest was provided while building manifest for operation {}",
                        self.operation.name()
                    ),
                    location: location!(),
                });

        match &self.operation {
            Operation::Append { ref fragments } => {
                final_fragments.extend(maybe_existing_fragments?.clone());
                let mut new_fragments =
                    Self::fragments_with_ids(fragments.clone(), &mut fragment_id)
                        .collect::<Vec<_>>();
                if let Some(next_row_id) = &mut next_row_id {
                    Self::assign_row_ids(next_row_id, new_fragments.as_mut_slice())?;
                }
                final_fragments.extend(new_fragments);
            }
            Operation::Delete {
                ref updated_fragments,
                ref deleted_fragment_ids,
                ..
            } => {
                // Remove the deleted fragments
                final_fragments.extend(maybe_existing_fragments?.clone());
                final_fragments.retain(|f| !deleted_fragment_ids.contains(&f.id));
                final_fragments.iter_mut().for_each(|f| {
                    for updated in updated_fragments {
                        if updated.id == f.id {
                            *f = updated.clone();
                        }
                    }
                });
            }
            Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
            } => {
                final_fragments.extend(maybe_existing_fragments?.iter().filter_map(|f| {
                    if removed_fragment_ids.contains(&f.id) {
                        return None;
                    }
                    if let Some(updated) = updated_fragments.iter().find(|uf| uf.id == f.id) {
                        Some(updated.clone())
                    } else {
                        Some(f.clone())
                    }
                }));
                let mut new_fragments =
                    Self::fragments_with_ids(new_fragments.clone(), &mut fragment_id)
                        .collect::<Vec<_>>();
                if let Some(next_row_id) = &mut next_row_id {
                    Self::assign_row_ids(next_row_id, new_fragments.as_mut_slice())?;
                }
                final_fragments.extend(new_fragments);
            }
            Operation::Overwrite { ref fragments, .. } => {
                let mut new_fragments =
                    Self::fragments_with_ids(fragments.clone(), &mut fragment_id)
                        .collect::<Vec<_>>();
                if let Some(next_row_id) = &mut next_row_id {
                    Self::assign_row_ids(next_row_id, new_fragments.as_mut_slice())?;
                }
                final_fragments.extend(new_fragments);
                final_indices = Vec::new();
            }
            Operation::Rewrite {
                ref groups,
                ref rewritten_indices,
            } => {
                final_fragments.extend(maybe_existing_fragments?.clone());
                let current_version = current_manifest.map(|m| m.version).unwrap_or_default();
                Self::handle_rewrite_fragments(
                    &mut final_fragments,
                    groups,
                    &mut fragment_id,
                    current_version,
                )?;
                Self::handle_rewrite_indices(&mut final_indices, rewritten_indices, groups)?;
            }
            Operation::CreateIndex {
                new_indices,
                removed_indices,
            } => {
                final_fragments.extend(maybe_existing_fragments?.clone());
                final_indices.retain(|existing_index| {
                    !new_indices
                        .iter()
                        .any(|new_index| new_index.name == existing_index.name)
                        && !removed_indices
                            .iter()
                            .any(|old_index| old_index.uuid == existing_index.uuid)
                });
                final_indices.extend(new_indices.clone());
            }
            Operation::ReserveFragments { .. } => {
                final_fragments.extend(maybe_existing_fragments?.clone());
            }
            Operation::Merge { ref fragments, .. } => {
                final_fragments.extend(fragments.clone());

                // Some fields that have indices may have been removed, so we should
                // remove those indices as well.
                Self::retain_relevant_indices(&mut final_indices, &schema)
            }
            Operation::Project { .. } => {
                final_fragments.extend(maybe_existing_fragments?.clone());

                // We might have removed all fields for certain data files, so
                // we should remove the data files that are no longer relevant.
                let remaining_field_ids = schema
                    .fields_pre_order()
                    .map(|f| f.id)
                    .collect::<HashSet<_>>();
                for fragment in final_fragments.iter_mut() {
                    fragment.files.retain(|file| {
                        file.fields
                            .iter()
                            .any(|field_id| remaining_field_ids.contains(field_id))
                    });
                }

                // Some fields that have indices may have been removed, so we should
                // remove those indices as well.
                Self::retain_relevant_indices(&mut final_indices, &schema)
            }
            Operation::Restore { .. } => {
                unreachable!()
            }
        };

        // If a fragment was reserved then it may not belong at the end of the fragments list.
        final_fragments.sort_by_key(|frag| frag.id);

        let mut manifest = if let Some(current_manifest) = current_manifest {
            Manifest::new_from_previous(current_manifest, schema, Arc::new(final_fragments))
        } else {
            Manifest::new(schema, Arc::new(final_fragments))
        };

        manifest.tag.clone_from(&self.tag);

        if config.auto_set_feature_flags {
            apply_feature_flags(&mut manifest, config.use_move_stable_row_ids)?;
        }
        manifest.set_timestamp(timestamp_to_nanos(config.timestamp));

        manifest.update_max_fragment_id();

        if let Operation::ReserveFragments { num_fragments } = self.operation {
            manifest.max_fragment_id += num_fragments;
        }

        manifest.transaction_file = Some(transaction_file_path.to_string());

        if let Some(next_row_id) = next_row_id {
            manifest.next_row_id = next_row_id;
        }

        Ok((manifest, final_indices))
    }

    fn retain_relevant_indices(indices: &mut Vec<Index>, schema: &Schema) {
        let field_ids = schema
            .fields_pre_order()
            .map(|f| f.id)
            .collect::<HashSet<_>>();
        indices.retain(|existing_index| {
            existing_index
                .fields
                .iter()
                .all(|field_id| field_ids.contains(field_id))
        });
    }

    fn recalculate_fragment_bitmap(
        old: &RoaringBitmap,
        groups: &[RewriteGroup],
    ) -> Result<RoaringBitmap> {
        let mut new_bitmap = old.clone();
        for group in groups {
            let any_in_index = group
                .old_fragments
                .iter()
                .any(|frag| old.contains(frag.id as u32));
            let all_in_index = group
                .old_fragments
                .iter()
                .all(|frag| old.contains(frag.id as u32));
            // Any rewrite group may or may not be covered by the index.  However, if any fragment
            // in a rewrite group was previously covered by the index then all fragments in the rewrite
            // group must have been previously covered by the index.  plan_compaction takes care of
            // this for us so this should be safe to assume.
            if any_in_index {
                if all_in_index {
                    for frag_id in group.old_fragments.iter().map(|frag| frag.id as u32) {
                        new_bitmap.remove(frag_id);
                    }
                    new_bitmap.extend(group.new_fragments.iter().map(|frag| frag.id as u32));
                } else {
                    return Err(Error::invalid_input("The compaction plan included a rewrite group that was a split of indexed and non-indexed data", location!()));
                }
            }
        }
        Ok(new_bitmap)
    }

    fn handle_rewrite_indices(
        indices: &mut [Index],
        rewritten_indices: &[RewrittenIndex],
        groups: &[RewriteGroup],
    ) -> Result<()> {
        let mut modified_indices = HashSet::new();

        for rewritten_index in rewritten_indices {
            if !modified_indices.insert(rewritten_index.old_id) {
                return Err(Error::invalid_input(format!("An invalid compaction plan must have been generated because multiple tasks modified the same index: {}", rewritten_index.old_id), location!()));
            }

            let index = indices
                .iter_mut()
                .find(|idx| idx.uuid == rewritten_index.old_id)
                .ok_or_else(|| {
                    Error::invalid_input(
                        format!(
                            "Invalid compaction plan refers to index {} which does not exist",
                            rewritten_index.old_id
                        ),
                        location!(),
                    )
                })?;

            index.fragment_bitmap = Some(Self::recalculate_fragment_bitmap(
                index.fragment_bitmap.as_ref().ok_or_else(|| {
                    Error::invalid_input(
                        format!(
                            "Cannot rewrite index {} which did not store fragment bitmap",
                            index.uuid
                        ),
                        location!(),
                    )
                })?,
                groups,
            )?);
            index.uuid = rewritten_index.new_id;
        }
        Ok(())
    }

    fn handle_rewrite_fragments(
        final_fragments: &mut Vec<Fragment>,
        groups: &[RewriteGroup],
        fragment_id: &mut u64,
        version: u64,
    ) -> Result<()> {
        for group in groups {
            // If the old fragments are contiguous, find the range
            let replace_range = {
                let start = final_fragments.iter().enumerate().find(|(_, f)| f.id == group.old_fragments[0].id)
                    .ok_or_else(|| Error::CommitConflict { version, source:
                        format!("dataset does not contain a fragment a rewrite operation wants to replace: id={}", group.old_fragments[0].id).into() , location:location!()})?.0;

                // Verify old_fragments matches contiguous range
                let mut i = 1;
                loop {
                    if i == group.old_fragments.len() {
                        break Some(start..start + i);
                    }
                    if final_fragments[start + i].id != group.old_fragments[i].id {
                        break None;
                    }
                    i += 1;
                }
            };

            let new_fragments = Self::fragments_with_ids(group.new_fragments.clone(), fragment_id);
            if let Some(replace_range) = replace_range {
                // Efficiently path using slice
                final_fragments.splice(replace_range, new_fragments);
            } else {
                // Slower path for non-contiguous ranges
                for fragment in group.old_fragments.iter() {
                    final_fragments.retain(|f| f.id != fragment.id);
                }
                final_fragments.extend(new_fragments);
            }
        }
        Ok(())
    }

    fn assign_row_ids(next_row_id: &mut u64, fragments: &mut [Fragment]) -> Result<()> {
        for fragment in fragments {
            let physical_rows = fragment.physical_rows.ok_or_else(|| Error::Internal {
                message: "Fragment does not have physical rows".into(),
                location: location!(),
            })? as u64;
            let row_ids = *next_row_id..(*next_row_id + physical_rows);
            let sequence = RowIdSequence::from(row_ids);
            // TODO: write to a separate file if large. Possibly share a file with other fragments.
            let serialized = write_row_ids(&sequence);
            fragment.row_id_meta = Some(RowIdMeta::Inline(serialized));
            *next_row_id += physical_rows;
        }
        Ok(())
    }
}

impl TryFrom<pb::Transaction> for Transaction {
    type Error = Error;

    fn try_from(message: pb::Transaction) -> Result<Self> {
        let operation = match message.operation {
            Some(pb::transaction::Operation::Append(pb::transaction::Append { fragments })) => {
                Operation::Append {
                    fragments: fragments
                        .into_iter()
                        .map(Fragment::try_from)
                        .collect::<Result<Vec<_>>>()?,
                }
            }
            Some(pb::transaction::Operation::Delete(pb::transaction::Delete {
                updated_fragments,
                deleted_fragment_ids,
                predicate,
            })) => Operation::Delete {
                updated_fragments: updated_fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<Vec<_>>>()?,
                deleted_fragment_ids: deleted_fragment_ids.clone(),
                predicate: predicate.clone(),
            },
            Some(pb::transaction::Operation::Overwrite(pb::transaction::Overwrite {
                fragments,
                schema,
                schema_metadata: _schema_metadata, // TODO: handle metadata
            })) => Operation::Overwrite {
                fragments: fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<Vec<_>>>()?,
                schema: Schema::from(&Fields(schema.clone())),
            },
            Some(pb::transaction::Operation::ReserveFragments(
                pb::transaction::ReserveFragments { num_fragments },
            )) => Operation::ReserveFragments { num_fragments },
            Some(pb::transaction::Operation::Rewrite(pb::transaction::Rewrite {
                old_fragments,
                new_fragments,
                groups,
                rewritten_indices,
            })) => {
                let groups = if !groups.is_empty() {
                    groups
                        .into_iter()
                        .map(RewriteGroup::try_from)
                        .collect::<Result<_>>()?
                } else {
                    vec![RewriteGroup {
                        old_fragments: old_fragments
                            .into_iter()
                            .map(Fragment::try_from)
                            .collect::<Result<Vec<_>>>()?,
                        new_fragments: new_fragments
                            .into_iter()
                            .map(Fragment::try_from)
                            .collect::<Result<Vec<_>>>()?,
                    }]
                };
                let rewritten_indices = rewritten_indices
                    .iter()
                    .map(RewrittenIndex::try_from)
                    .collect::<Result<_>>()?;

                Operation::Rewrite {
                    groups,
                    rewritten_indices,
                }
            }
            Some(pb::transaction::Operation::CreateIndex(pb::transaction::CreateIndex {
                new_indices,
                removed_indices,
            })) => Operation::CreateIndex {
                new_indices: new_indices
                    .into_iter()
                    .map(Index::try_from)
                    .collect::<Result<_>>()?,
                removed_indices: removed_indices
                    .into_iter()
                    .map(Index::try_from)
                    .collect::<Result<_>>()?,
            },
            Some(pb::transaction::Operation::Merge(pb::transaction::Merge {
                fragments,
                schema,
                schema_metadata: _schema_metadata, // TODO: handle metadata
            })) => Operation::Merge {
                fragments: fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<Vec<_>>>()?,
                schema: Schema::from(&Fields(schema.clone())),
            },
            Some(pb::transaction::Operation::Restore(pb::transaction::Restore { version })) => {
                Operation::Restore { version }
            }
            Some(pb::transaction::Operation::Update(pb::transaction::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
            })) => Operation::Update {
                removed_fragment_ids: removed_fragment_ids.clone(),
                updated_fragments: updated_fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<Vec<_>>>()?,
                new_fragments: new_fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<Vec<_>>>()?,
            },
            Some(pb::transaction::Operation::Project(pb::transaction::Project { schema })) => {
                Operation::Project {
                    schema: Schema::from(&Fields(schema.clone())),
                }
            }
            None => {
                return Err(Error::Internal {
                    message: "Transaction message did not contain an operation".to_string(),
                    location: location!(),
                });
            }
        };
        Ok(Self {
            read_version: message.read_version,
            uuid: message.uuid.clone(),
            operation,
            tag: if message.tag.is_empty() {
                None
            } else {
                Some(message.tag.clone())
            },
        })
    }
}

impl TryFrom<&pb::transaction::rewrite::RewrittenIndex> for RewrittenIndex {
    type Error = Error;

    fn try_from(message: &pb::transaction::rewrite::RewrittenIndex) -> Result<Self> {
        Ok(Self {
            old_id: message
                .old_id
                .as_ref()
                .map(Uuid::try_from)
                .ok_or_else(|| {
                    Error::io(
                        "required field (old_id) missing from message".to_string(),
                        location!(),
                    )
                })??,
            new_id: message
                .new_id
                .as_ref()
                .map(Uuid::try_from)
                .ok_or_else(|| {
                    Error::io(
                        "required field (new_id) missing from message".to_string(),
                        location!(),
                    )
                })??,
        })
    }
}

impl TryFrom<pb::transaction::rewrite::RewriteGroup> for RewriteGroup {
    type Error = Error;

    fn try_from(message: pb::transaction::rewrite::RewriteGroup) -> Result<Self> {
        Ok(Self {
            old_fragments: message
                .old_fragments
                .into_iter()
                .map(Fragment::try_from)
                .collect::<Result<Vec<_>>>()?,
            new_fragments: message
                .new_fragments
                .into_iter()
                .map(Fragment::try_from)
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl From<&Transaction> for pb::Transaction {
    fn from(value: &Transaction) -> Self {
        let operation = match &value.operation {
            Operation::Append { fragments } => {
                pb::transaction::Operation::Append(pb::transaction::Append {
                    fragments: fragments.iter().map(pb::DataFragment::from).collect(),
                })
            }
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids,
                predicate,
            } => pb::transaction::Operation::Delete(pb::transaction::Delete {
                updated_fragments: updated_fragments
                    .iter()
                    .map(pb::DataFragment::from)
                    .collect(),
                deleted_fragment_ids: deleted_fragment_ids.clone(),
                predicate: predicate.clone(),
            }),
            Operation::Overwrite { fragments, schema } => {
                pb::transaction::Operation::Overwrite(pb::transaction::Overwrite {
                    fragments: fragments.iter().map(pb::DataFragment::from).collect(),
                    schema: Fields::from(schema).0,
                    schema_metadata: Default::default(), // TODO: handle metadata
                })
            }
            Operation::ReserveFragments { num_fragments } => {
                pb::transaction::Operation::ReserveFragments(pb::transaction::ReserveFragments {
                    num_fragments: *num_fragments,
                })
            }
            Operation::Rewrite {
                groups,
                rewritten_indices,
            } => pb::transaction::Operation::Rewrite(pb::transaction::Rewrite {
                groups: groups
                    .iter()
                    .map(pb::transaction::rewrite::RewriteGroup::from)
                    .collect(),
                rewritten_indices: rewritten_indices
                    .iter()
                    .map(|rewritten| rewritten.into())
                    .collect(),
                ..Default::default()
            }),
            Operation::CreateIndex {
                new_indices,
                removed_indices,
            } => pb::transaction::Operation::CreateIndex(pb::transaction::CreateIndex {
                new_indices: new_indices.iter().map(IndexMetadata::from).collect(),
                removed_indices: removed_indices.iter().map(IndexMetadata::from).collect(),
            }),
            Operation::Merge { fragments, schema } => {
                pb::transaction::Operation::Merge(pb::transaction::Merge {
                    fragments: fragments.iter().map(pb::DataFragment::from).collect(),
                    schema: Fields::from(schema).0,
                    schema_metadata: Default::default(), // TODO: handle metadata
                })
            }
            Operation::Restore { version } => {
                pb::transaction::Operation::Restore(pb::transaction::Restore { version: *version })
            }
            Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
            } => pb::transaction::Operation::Update(pb::transaction::Update {
                removed_fragment_ids: removed_fragment_ids.clone(),
                updated_fragments: updated_fragments
                    .iter()
                    .map(pb::DataFragment::from)
                    .collect(),
                new_fragments: new_fragments.iter().map(pb::DataFragment::from).collect(),
            }),
            Operation::Project { schema } => {
                pb::transaction::Operation::Project(pb::transaction::Project {
                    schema: Fields::from(schema).0,
                })
            }
        };

        Self {
            read_version: value.read_version,
            uuid: value.uuid.clone(),
            operation: Some(operation),
            tag: value.tag.clone().unwrap_or("".to_string()),
        }
    }
}

impl From<&RewrittenIndex> for pb::transaction::rewrite::RewrittenIndex {
    fn from(value: &RewrittenIndex) -> Self {
        Self {
            old_id: Some((&value.old_id).into()),
            new_id: Some((&value.new_id).into()),
        }
    }
}

impl From<&RewriteGroup> for pb::transaction::rewrite::RewriteGroup {
    fn from(value: &RewriteGroup) -> Self {
        Self {
            old_fragments: value
                .old_fragments
                .iter()
                .map(pb::DataFragment::from)
                .collect(),
            new_fragments: value
                .new_fragments
                .iter()
                .map(pb::DataFragment::from)
                .collect(),
        }
    }
}

/// Validate the operation is valid for the given manifest.
pub fn validate_operation(manifest: Option<&Manifest>, operation: &Operation) -> Result<()> {
    let manifest = match (manifest, operation) {
        (None, Operation::Overwrite { fragments, schema }) => {
            // Validate here because we are going to return early.
            schema_fragments_valid(schema, fragments)?;

            return Ok(());
        }
        (Some(manifest), _) => manifest,
        (None, _) => {
            return Err(Error::invalid_input(
                format!(
                    "Cannot apply operation {} to non-existent dataset",
                    operation.name()
                ),
                location!(),
            ));
        }
    };

    match operation {
        Operation::Append { fragments } => {
            // Fragments must contain all fields in the schema
            schema_fragments_valid(&manifest.schema, fragments)
        }
        Operation::Project { schema } => {
            schema_fragments_valid(schema, manifest.fragments.as_ref())
        }
        Operation::Merge { fragments, schema } | Operation::Overwrite { fragments, schema } => {
            schema_fragments_valid(schema, fragments)
        }
        Operation::Update {
            updated_fragments,
            new_fragments,
            ..
        } => {
            schema_fragments_valid(&manifest.schema, updated_fragments)?;
            schema_fragments_valid(&manifest.schema, new_fragments)
        }
        _ => Ok(()),
    }
}

/// Check that each fragment contains all fields in the schema.
/// It is not required that the schema contains all fields in the fragment.
/// There may be masked fields.
fn schema_fragments_valid(schema: &Schema, fragments: &[Fragment]) -> Result<()> {
    // TODO: add additional validation. Consider consolidating with various
    // validate() methods in the codebase.
    for fragment in fragments {
        for field in schema.fields_pre_order() {
            if !fragment
                .files
                .iter()
                .flat_map(|f| f.fields.iter())
                .any(|f_id| f_id == &field.id)
            {
                return Err(Error::invalid_input(
                    format!(
                        "Fragment {} does not contain field {:?}",
                        fragment.id, field
                    ),
                    location!(),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conflicts() {
        let index0 = Index {
            uuid: uuid::Uuid::new_v4(),
            name: "test".to_string(),
            fields: vec![0],
            dataset_version: 1,
            fragment_bitmap: None,
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
                schema: Schema::default(),
            },
            Operation::Overwrite {
                fragments: vec![fragment0.clone(), fragment2.clone()],
                schema: Schema::default(),
            },
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments: vec![fragment0.clone()],
                    new_fragments: vec![fragment1.clone()],
                }],
                rewritten_indices: vec![],
            },
            Operation::ReserveFragments { num_fragments: 3 },
            Operation::Update {
                removed_fragment_ids: vec![1],
                updated_fragments: vec![fragment0.clone()],
                new_fragments: vec![fragment2.clone()],
            },
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
                [false, false, false, true, true, false, false, false],
            ),
            (
                Operation::Delete {
                    // Delete that affects fragments different from other transactions
                    updated_fragments: vec![fragment1.clone()],
                    deleted_fragment_ids: vec![],
                    predicate: "x > 2".to_string(),
                },
                [false, false, false, true, true, false, false, true],
            ),
            (
                Operation::Delete {
                    // Delete that affects same fragments as other transactions
                    updated_fragments: vec![fragment0.clone(), fragment2.clone()],
                    deleted_fragment_ids: vec![],
                    predicate: "x > 2".to_string(),
                },
                [false, false, true, true, true, true, false, true],
            ),
            (
                Operation::Overwrite {
                    fragments: vec![fragment0.clone(), fragment2.clone()],
                    schema: Schema::default(),
                },
                // No conflicts: overwrite can always happen since it doesn't
                // depend on previous state of the table.
                [false, false, false, false, false, false, false, false],
            ),
            (
                Operation::CreateIndex {
                    new_indices: vec![index0.clone()],
                    removed_indices: vec![index0.clone()],
                },
                // Will only conflict with operations that modify row ids.
                [false, false, false, false, true, true, false, false],
            ),
            (
                // Rewrite that affects different fragments
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments: vec![fragment1.clone()],
                        new_fragments: vec![fragment0.clone()],
                    }],
                    rewritten_indices: Vec::new(),
                },
                [false, true, false, true, true, false, false, true],
            ),
            (
                // Rewrite that affects the same fragments
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments: vec![fragment0.clone(), fragment2.clone()],
                        new_fragments: vec![fragment0.clone()],
                    }],
                    rewritten_indices: Vec::new(),
                },
                [false, true, true, true, true, true, false, true],
            ),
            (
                Operation::Merge {
                    fragments: vec![fragment0.clone(), fragment2.clone()],
                    schema: Schema::default(),
                },
                // Merge conflicts with everything except CreateIndex and ReserveFragments.
                [true, false, true, true, true, true, false, true],
            ),
            (
                Operation::ReserveFragments { num_fragments: 2 },
                // ReserveFragments only conflicts with Overwrite and Restore.
                [false, false, false, false, true, false, false, false],
            ),
            (
                Operation::Update {
                    // Update that affects same fragments as other transactions
                    updated_fragments: vec![fragment0.clone()],
                    removed_fragment_ids: vec![],
                    new_fragments: vec![fragment2.clone()],
                },
                [false, false, true, true, true, true, false, true],
            ),
        ];

        for (operation, expected_conflicts) in &cases {
            let transaction = Transaction::new(0, operation.clone(), None);
            for (other, expected_conflict) in other_transactions.iter().zip(expected_conflicts) {
                assert_eq!(
                    transaction.conflicts_with(other),
                    *expected_conflict,
                    "Transaction {:?} should {} with {:?}",
                    transaction,
                    if *expected_conflict {
                        "conflict"
                    } else {
                        "not conflict"
                    },
                    other
                );
            }
        }
    }

    #[test]
    fn test_rewrite_fragments() {
        let existing_fragments: Vec<Fragment> = (0..10).map(Fragment::new).collect();

        let mut final_fragments = existing_fragments.clone();
        let rewrite_groups = vec![
            // Since these are contiguous, they will be put in the same location
            // as 1 and 2.
            RewriteGroup {
                old_fragments: vec![Fragment::new(1), Fragment::new(2)],
                // These two fragments were previously reserved
                new_fragments: vec![Fragment::new(15), Fragment::new(16)],
            },
            // These are not contiguous, so they will be inserted at the end.
            RewriteGroup {
                old_fragments: vec![Fragment::new(5), Fragment::new(8)],
                // We pretend this id was not reserved.  Does not happen in practice today
                // but we want to leave the door open.
                new_fragments: vec![Fragment::new(0)],
            },
        ];

        let mut fragment_id = 20;
        let version = 0;

        Transaction::handle_rewrite_fragments(
            &mut final_fragments,
            &rewrite_groups,
            &mut fragment_id,
            version,
        )
        .unwrap();

        assert_eq!(fragment_id, 21);

        let expected_fragments: Vec<Fragment> = vec![
            Fragment::new(0),
            Fragment::new(15),
            Fragment::new(16),
            Fragment::new(3),
            Fragment::new(4),
            Fragment::new(6),
            Fragment::new(7),
            Fragment::new(9),
            Fragment::new(20),
        ];

        assert_eq!(final_fragments, expected_fragments);
    }
}
