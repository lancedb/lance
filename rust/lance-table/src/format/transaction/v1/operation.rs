// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};

use deepsize::DeepSizeOf;
use lance_core::datatypes::Schema;
use uuid::Uuid;

use lance_core::{Error, Result};
use snafu::{location, Location};

use crate::format::pb;
use crate::format::{transaction::v2::action::Action, Fragment, Index, Manifest};

/// An operation on a dataset.
#[derive(Debug, Clone, DeepSizeOf)]
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
        config_upsert_values: Option<HashMap<String, String>>,
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

    /// Update the dataset configuration.
    UpdateConfig {
        upsert_values: Option<HashMap<String, String>>,
        delete_keys: Option<Vec<String>>,
    },
}

#[derive(Debug, Clone)]
pub struct RewrittenIndex {
    pub old_id: Uuid,
    pub new_id: Uuid,
}

impl DeepSizeOf for RewrittenIndex {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        0
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
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
            | Self::UpdateConfig { .. }
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

    /// Returns the config keys that have been upserted by this operation.
    fn get_upsert_config_keys(&self) -> Vec<String> {
        match self {
            Self::Overwrite {
                config_upsert_values: Some(upsert_values),
                ..
            } => {
                let vec: Vec<String> = upsert_values.keys().cloned().collect();
                vec
            }
            Self::UpdateConfig {
                upsert_values: Some(uv),
                ..
            } => {
                let vec: Vec<String> = uv.keys().cloned().collect();
                vec
            }
            _ => Vec::<String>::new(),
        }
    }

    /// Returns the config keys that have been deleted by this operation.
    fn get_delete_config_keys(&self) -> &[String] {
        match self {
            Self::UpdateConfig {
                delete_keys: Some(dk),
                ..
            } => dk.as_slice(),
            _ => &[],
        }
    }

    /// Check whether another operation modifies the same fragment IDs as this one.
    fn modifies_same_ids(&self, other: &Self) -> bool {
        let self_ids = self.modified_fragment_ids().collect::<HashSet<_>>();
        let mut other_ids = other.modified_fragment_ids();
        other_ids.any(|id| self_ids.contains(&id))
    }

    /// Check whether another operation upserts a key that is referenced by another operation
    fn upsert_key_conflict(&self, other: &Self) -> bool {
        let self_upsert_keys = self.get_upsert_config_keys();
        let other_upsert_keys = other.get_upsert_config_keys();

        let self_delete_keys = self.get_delete_config_keys();
        let other_delete_keys = other.get_delete_config_keys();

        self_upsert_keys
            .iter()
            .any(|x| other_upsert_keys.contains(x) || other_delete_keys.contains(x))
            || other_upsert_keys
                .iter()
                .any(|x| self_upsert_keys.contains(x) || self_delete_keys.contains(x))
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
            Self::UpdateConfig { .. } => "UpdateConfig",
        }
    }

    fn actions(self) -> Vec<Action> {
        match self {
            Self::Append { fragments } => vec![Action::AddFragments { fragments }],
            Self::Delete {
                updated_fragments,
                deleted_fragment_ids,
                ..
            } => vec![
                Action::UpdateFragments {
                    fragments: updated_fragments,
                },
                Action::DeleteFragments {
                    deleted_fragment_ids,
                },
            ],
            Self::Overwrite {
                fragments,
                schema,
                config_upsert_values,
            } => vec![
                Action::ReplaceSchema { schema },
                Action::ReplaceFragments { fragments },
                Action::UpsertConfig {
                    upsert_values: config_upsert_values.unwrap_or_default(),
                    delete_keys: Vec::new(),
                },
            ],
            Self::CreateIndex {
                new_indices,
                removed_indices,
            } => vec![
                Action::AddIndices { index: new_indices },
                Action::RemoveIndices {
                    index: removed_indices,
                },
            ],
            Self::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
            } => vec![
                Action::UpdateFragments {
                    fragments: updated_fragments,
                },
                Action::AddFragments {
                    fragments: new_fragments,
                },
                Action::DeleteFragments {
                    deleted_fragment_ids: removed_fragment_ids,
                },
            ],
            Self::Rewrite { groups, .. } => {
                let mut actions = Vec::new();
                for group in groups {
                    actions.push(Action::ReplaceFragments {
                        fragments: group.new_fragments,
                    });
                    actions.push(Action::DeleteFragments {
                        deleted_fragment_ids: group.old_fragments.iter().map(|f| f.id).collect(),
                    });
                }
                actions
            }
            Self::Merge { fragments, schema } => vec![
                Action::ReplaceFragments { fragments },
                Action::ReplaceSchema { schema },
            ],
            Self::ReserveFragments { .. } => Vec::new(),
            Self::Restore { .. } => Vec::new(),
            Self::Project { schema } => vec![Action::ReplaceSchema { schema }],
            Self::UpdateConfig {
                upsert_values,
                delete_keys,
            } => vec![Action::UpsertConfig {
                upsert_values: upsert_values.unwrap_or_default(),
                delete_keys: delete_keys.unwrap_or_default(),
            }],
        }
    }

    /// Creates a description of the operation.
    ///
    /// These are meant to be user-facing and shown in a SHOW HISTORY command.
    /// They should reference rows and predicate. They should NOT reference low
    /// level details like fragments or data files.
    fn description(&self) -> String {
        match self {
            Self::Append { fragments } => {
                let num_rows: usize = fragments
                    .iter()
                    .map(|f| f.physical_rows.unwrap_or_default())
                    .sum();
                format!("Append {} rows.", num_rows)
            }
            Self::Delete { predicate, .. } => {
                // Keep up to 256 characters of the predicate
                let truncated_predicate = if predicate.len() > 256 {
                    format!("{}...", &predicate[..253])
                } else {
                    predicate.clone()
                };
                format!("Deleted with predicate: {}", truncated_predicate)
            }
            Self::Overwrite { fragments, .. } => {
                let num_rows: usize = fragments
                    .iter()
                    .map(|f| f.physical_rows.unwrap_or_default())
                    .sum();
                format!("Replaced dataset. Wrote {} rows.", num_rows)
            }
            Self::CreateIndex { new_indices, .. } => {
                let mut description = "Created indices: ".to_string();
                for index in new_indices {
                    description.push_str(&index.name);
                    description.push_str(" (");
                    description.push_str(&index.uuid.to_string());
                    description.push_str("), ");
                }
                description.pop();
                description.pop();
                description
            }
            Self::Update { .. } => {
                // This is used by merge-insert, so it's hard to know how many
                // rows were updated, deleted, or inserted.
                "Update".into()
            }
            Self::Rewrite { .. } => "Rewrite".to_string(),
            Self::Merge { .. } => "Merge".into(),
            Self::ReserveFragments { num_fragments } => {
                format!("Reserve {} fragments.", num_fragments)
            }
            Self::Restore { version } => format!("Restore version {}.", version),
            Self::Project { .. } => "Project".to_string(),
            Self::UpdateConfig {
                upsert_values,
                delete_keys,
            } => {
                let mut description = "Update config: ".to_string();
                if let Some(upsert_values) = upsert_values {
                    for (key, value) in upsert_values {
                        description.push_str(key);
                        description.push_str("=");
                        description.push_str(value);
                        description.push_str(", ");
                    }
                }
                if let Some(delete_keys) = delete_keys {
                    for key in delete_keys {
                        description.push_str("delete ");
                        description.push_str(key);
                        description.push_str(", ");
                    }
                }
                description.pop();
                description.pop();
                description
            }
        }
    }

    pub fn conflicts_with(self, other: &Self) -> bool {
        // This assumes IsolationLevel is Snapshot Isolation, which is more
        // permissive than Serializable. In particular, it allows a Delete
        // transaction to succeed after a concurrent Append, even if the Append
        // added rows that would be deleted.
        match &self {
            Operation::Append { .. } => match &other {
                // Append is compatible with anything that doesn't change the schema
                Operation::Append { .. } => false,
                Operation::Rewrite { .. } => false,
                Operation::CreateIndex { .. } => false,
                Operation::Delete { .. } | Operation::Update { .. } => false,
                Operation::ReserveFragments { .. } => false,
                Operation::Project { .. } => false,
                Operation::UpdateConfig { .. } => false,
                _ => true,
            },
            Operation::Rewrite { .. } => match &other {
                // Rewrite is only compatible with operations that don't touch
                // existing fragments.
                // TODO: it could also be compatible with operations that update
                // fragments we don't touch.
                Operation::Append { .. } => false,
                Operation::ReserveFragments { .. } => false,
                Operation::Delete { .. } | Operation::Rewrite { .. } | Operation::Update { .. } => {
                    // As long as they rewrite disjoint fragments they shouldn't conflict.
                    self.modifies_same_ids(&other)
                }
                Operation::Project { .. } => false,
                Operation::UpdateConfig { .. } => false,
                _ => true,
            },
            // Restore always succeeds
            Operation::Restore { .. } => false,
            // ReserveFragments is compatible with anything that doesn't reset the
            // max fragment id.
            Operation::ReserveFragments { .. } => matches!(
                &other,
                Operation::Overwrite { .. } | Operation::Restore { .. }
            ),
            Operation::CreateIndex { .. } => match &other {
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
                Operation::UpdateConfig { .. } => false,
                _ => true,
            },
            Operation::Delete { .. } | Operation::Update { .. } => match &other {
                Operation::CreateIndex { .. } => false,
                Operation::ReserveFragments { .. } => false,
                Operation::Delete { .. } | Operation::Rewrite { .. } | Operation::Update { .. } => {
                    // If we update the same fragments, we conflict.
                    self.modifies_same_ids(&other)
                }
                Operation::Project { .. } => false,
                Operation::Append { .. } => false,
                Operation::UpdateConfig { .. } => false,
                _ => true,
            },
            Operation::Overwrite { .. } | Operation::UpdateConfig { .. } => match &other {
                Operation::Overwrite { .. } | Operation::UpdateConfig { .. } => {
                    self.upsert_key_conflict(&other)
                }
                _ => false,
            },
            // Merge changes the schema, but preserves row ids, so the only operations
            // it's compatible with is CreateIndex, ReserveFragments, SetMetadata and DeleteMetadata.
            Operation::Merge { .. } => !matches!(
                &other,
                Operation::CreateIndex { .. }
                    | Operation::ReserveFragments { .. }
                    | Operation::UpdateConfig { .. }
            ),
            Operation::Project { .. } => match &other {
                // Project is compatible with anything that doesn't change the schema
                Operation::CreateIndex { .. } => false,
                Operation::Overwrite { .. } => false,
                Operation::UpdateConfig { .. } => false,
                _ => true,
            },
        }
    }
}

/// Validate the operation is valid for the given manifest.
pub fn validate_operation(manifest: Option<&Manifest>, operation: &Operation) -> Result<()> {
    let manifest = match (manifest, operation) {
        (
            None,
            Operation::Overwrite {
                fragments,
                schema,
                config_upsert_values: None,
            },
        ) => {
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
        Operation::Merge { fragments, schema }
        | Operation::Overwrite {
            fragments,
            schema,
            config_upsert_values: None,
        } => schema_fragments_valid(schema, fragments),
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
