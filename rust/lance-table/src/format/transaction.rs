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
//! |                  | Append | Delete / Update | Overwrite/Create | Create Index | Rewrite | Merge | Project | UpdateConfig |
//! |------------------|--------|-----------------|------------------|--------------|---------|-------|---------|-------------|
//! | Append           | ✅     | ✅              | ❌               | ✅           | ✅      | ❌    | ❌      | ✅           |
//! | Delete / Update  | ✅     | (1)             | ❌               | ✅           | (1)     | ❌    | ❌      | ✅           |
//! | Overwrite/Create | ✅     | ✅              | ✅               | ✅           | ✅      | ✅    | ✅      | (2)          |
//! | Create index     | ✅     | ✅              | ❌               | ✅           | ✅      | ✅    | ✅      | ✅           |
//! | Rewrite          | ✅     | (1)             | ❌               | ❌           | (1)     | ❌    | ❌      | ✅           |
//! | Merge            | ❌     | ❌              | ❌               | ❌           | ✅      | ❌    | ❌      | ✅           |
//! | Project          | ✅     | ✅              | ❌               | ❌           | ✅      | ❌    | ✅      | ✅           |
//! | UpdateConfig     | ✅     | ✅              | (2)              | ✅           | ✅      | ✅    | ✅      | (2)          |
//!
//! (1) Delete, update, and rewrite are compatible with each other and themselves only if
//! they affect distinct fragments. Otherwise, they conflict.
//! (2) Operations that mutate the config conflict if one of the operations upserts a key
//! that if referenced by another concurrent operation.

use std::{collections::HashSet, sync::Arc, time::SystemTime};

use crate::{
    format::{pb, DataStorageFormat, Fragment, Index, Manifest, RowIdMeta},
    io::{
        commit::CommitHandler,
        manifest::{read_manifest, read_manifest_indexes},
    },
    rowids::{write_row_ids, RowIdSequence},
};
use deepsize::DeepSizeOf;
use lance_core::{datatypes::Schema, Error, Result};
use lance_file::version::LanceFileVersion;
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use roaring::RoaringBitmap;
use snafu::{location, Location};

use crate::feature_flags::{apply_feature_flags, FLAG_MOVE_STABLE_ROW_IDS};
use crate::utils::timestamp_to_nanos;

mod v1;

pub use v1::{validate_operation, Operation, RewriteGroup, RewrittenIndex};

/// A change to a dataset that can be retried
///
/// This contains enough information to be able to build the next manifest,
/// given the current manifest.
#[derive(Debug, Clone, DeepSizeOf)]
pub struct Transaction {
    /// The version of the table this transaction is based off of. If this is
    /// the first transaction, this should be 0.
    pub read_version: u64,
    pub uuid: String,
    pub operation: Operation,
    /// If the transaction modified the blobs dataset, this is the operation
    /// to apply to the blobs dataset.
    ///
    /// If this is `None`, then the blobs dataset was not modified
    pub blobs_op: Option<Operation>,
    pub tag: Option<String>,
}

#[derive(Debug)]
pub struct ManifestWriteConfig {
    pub auto_set_feature_flags: bool,              // default true
    pub timestamp: Option<SystemTime>,             // default None
    pub use_move_stable_row_ids: bool,             // default false
    pub use_legacy_format: Option<bool>,           // default None
    pub storage_format: Option<DataStorageFormat>, // default None
}

impl Default for ManifestWriteConfig {
    fn default() -> Self {
        Self {
            auto_set_feature_flags: true,
            timestamp: None,
            use_move_stable_row_ids: false,
            use_legacy_format: None,
            storage_format: None,
        }
    }
}

impl Transaction {
    pub fn new(
        read_version: u64,
        operation: Operation,
        blobs_op: Option<Operation>,
        tag: Option<String>,
    ) -> Self {
        let uuid = uuid::Uuid::new_v4().hyphenated().to_string();
        Self {
            read_version,
            uuid,
            operation,
            blobs_op,
            tag,
        }
    }

    /// Returns true if the transaction cannot be committed if the other
    /// transaction is committed first.
    pub fn conflicts_with(&self, other: &Self) -> bool {
        self.operation.conflicts_with(&other.operation)
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

    fn data_storage_format_from_files(
        fragments: &[Fragment],
        user_requested: Option<LanceFileVersion>,
    ) -> Result<DataStorageFormat> {
        if let Some(file_version) = Fragment::try_infer_version(fragments)? {
            // Ensure user-requested matches data files
            if let Some(user_requested) = user_requested {
                if user_requested != file_version {
                    return Err(Error::invalid_input(
                        format!("User requested data storage version ({}) does not match version in data files ({})", user_requested, file_version),
                        location!(),
                    ));
                }
            }
            Ok(DataStorageFormat::new(file_version))
        } else {
            // If no files use user-requested or default
            Ok(user_requested
                .map(DataStorageFormat::new)
                .unwrap_or_default())
        }
    }

    /// Restore an old manifest from the given version.
    pub async fn restore_old_manifest(
        object_store: &ObjectStore,
        commit_handler: &dyn CommitHandler,
        base_path: &Path,
        version: u64,
        config: &ManifestWriteConfig,
        tx_path: &str,
    ) -> Result<(Manifest, Vec<Index>)> {
        let location = commit_handler
            .resolve_version_location(base_path, version, &object_store.inner)
            .await?;
        let mut manifest = read_manifest(object_store, &location.path, location.size).await?;
        manifest.set_timestamp(timestamp_to_nanos(config.timestamp));
        manifest.transaction_file = Some(tx_path.to_string());
        let indices = read_manifest_indexes(object_store, &location.path, &manifest).await?;
        Ok((manifest, indices))
    }

    /// Create a new manifest from the current manifest and the transaction.
    ///
    /// `current_manifest` should only be None if the dataset does not yet exist.
    pub fn build_manifest(
        &self,
        current_manifest: Option<&Manifest>,
        current_indices: Vec<Index>,
        transaction_file_path: &str,
        config: &ManifestWriteConfig,
        new_blob_version: Option<u64>,
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
                Self::retain_relevant_indices(&mut final_indices, &schema, &final_fragments)
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
                Self::retain_relevant_indices(&mut final_indices, &schema, &final_fragments)
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

                if next_row_id.is_some() {
                    // We can re-use indices, but need to rewrite the fragment bitmaps
                    debug_assert!(rewritten_indices.is_empty());
                    for index in final_indices.iter_mut() {
                        if let Some(fragment_bitmap) = &mut index.fragment_bitmap {
                            *fragment_bitmap =
                                Self::recalculate_fragment_bitmap(fragment_bitmap, groups)?;
                        }
                    }
                } else {
                    Self::handle_rewrite_indices(&mut final_indices, rewritten_indices, groups)?;
                }
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
                Self::retain_relevant_indices(&mut final_indices, &schema, &final_fragments)
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
                Self::retain_relevant_indices(&mut final_indices, &schema, &final_fragments)
            }
            Operation::Restore { .. } => {
                unreachable!()
            }
            Operation::UpdateConfig { .. } => {}
        };

        // If a fragment was reserved then it may not belong at the end of the fragments list.
        final_fragments.sort_by_key(|frag| frag.id);

        let user_requested_version = match (&config.storage_format, config.use_legacy_format) {
            (Some(storage_format), _) => Some(storage_format.lance_file_version()?),
            (None, Some(true)) => Some(LanceFileVersion::Legacy),
            (None, Some(false)) => Some(LanceFileVersion::V2_0),
            (None, None) => None,
        };

        let mut manifest = if let Some(current_manifest) = current_manifest {
            let mut prev_manifest = Manifest::new_from_previous(
                current_manifest,
                schema,
                Arc::new(final_fragments),
                new_blob_version,
            );
            if user_requested_version.is_some()
                && matches!(self.operation, Operation::Overwrite { .. })
            {
                // If this is an overwrite operation and the user has requested a specific version
                // then overwrite with that version.  Otherwise, if the user didn't request a specific
                // version, then overwrite with whatever version we had before.
                prev_manifest.data_storage_format =
                    DataStorageFormat::new(user_requested_version.unwrap());
            }
            prev_manifest
        } else {
            let data_storage_format =
                Self::data_storage_format_from_files(&final_fragments, user_requested_version)?;
            Manifest::new(
                schema,
                Arc::new(final_fragments),
                data_storage_format,
                new_blob_version,
            )
        };

        manifest.tag.clone_from(&self.tag);

        if config.auto_set_feature_flags {
            apply_feature_flags(&mut manifest, config.use_move_stable_row_ids)?;
        }
        manifest.set_timestamp(timestamp_to_nanos(config.timestamp));

        manifest.update_max_fragment_id();

        match &self.operation {
            Operation::Overwrite {
                config_upsert_values: Some(tm),
                ..
            } => manifest.update_config(tm.clone()),
            Operation::UpdateConfig {
                upsert_values,
                delete_keys,
            } => {
                // Delete is handled first. If the same key is referenced by upsert and
                // delete, then upserted key-value pair will remain.
                if let Some(delete_keys) = delete_keys {
                    manifest.delete_config_keys(
                        delete_keys
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .as_slice(),
                    )
                }
                if let Some(upsert_values) = upsert_values {
                    manifest.update_config(upsert_values.clone());
                }
            }
            _ => {}
        }

        if let Operation::ReserveFragments { num_fragments } = self.operation {
            manifest.max_fragment_id += num_fragments;
        }

        manifest.transaction_file = Some(transaction_file_path.to_string());

        if let Some(next_row_id) = next_row_id {
            manifest.next_row_id = next_row_id;
        }

        Ok((manifest, final_indices))
    }

    fn retain_relevant_indices(indices: &mut Vec<Index>, schema: &Schema, fragments: &[Fragment]) {
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

        // We might have also removed all fragments that an index was covering, so
        // we should remove those indices as well.
        let fragment_ids = fragments.iter().map(|f| f.id).collect::<HashSet<_>>();
        indices.retain(|existing_index| {
            existing_index
                .fragment_bitmap
                .as_ref()
                .map(|bitmap| bitmap.iter().any(|id| fragment_ids.contains(&(id as u64))))
                .unwrap_or(true)
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
            Some(op) => Operation::try_from(op)?,
            None => {
                return Err(Error::Internal {
                    message: "Transaction message did not contain an operation".to_string(),
                    location: location!(),
                });
            }
        };
        let blobs_op = message
            .blob_operation
            .map(|blob_op| blob_op.try_into())
            .transpose()?;
        Ok(Self {
            read_version: message.read_version,
            uuid: message.uuid.clone(),
            operation,
            blobs_op,
            tag: if message.tag.is_empty() {
                None
            } else {
                Some(message.tag.clone())
            },
        })
    }
}

impl From<&Transaction> for pb::Transaction {
    fn from(value: &Transaction) -> Self {
        let operation = (&value.operation).into();

        let blob_operation = value.blobs_op.as_ref().map(|op| op.into());

        Self {
            read_version: value.read_version,
            uuid: value.uuid.clone(),
            operation: Some(operation),
            blob_operation,
            tag: value.tag.clone().unwrap_or("".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rewrite_fragments() {
        let existing_fragments: Vec<Fragment> = (0..10).map(Fragment::new).collect();

        let mut final_fragments = existing_fragments;
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
