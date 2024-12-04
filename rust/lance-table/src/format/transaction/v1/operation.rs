// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};

use deepsize::DeepSizeOf;
use lance_core::datatypes::Schema;
use lance_file::datatypes::Fields;

use crate::format::pb;
use crate::format::{pb::IndexMetadata, Fragment, Index, Manifest};
use lance_core::{Error, Result};
use snafu::{location, Location};
use uuid::Uuid;

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
    fn get_delete_config_keys(&self) -> Vec<String> {
        match self {
            Self::UpdateConfig {
                delete_keys: Some(dk),
                ..
            } => dk.clone(),
            _ => Vec::<String>::new(),
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

    /// Returns true if the transaction cannot be committed if the other
    /// transaction is committed first.
    pub fn conflicts_with(&self, other: &Self) -> bool {
        // This assumes IsolationLevel is Snapshot Isolation, which is more
        // permissive than Serializable. In particular, it allows a Delete
        // transaction to succeed after a concurrent Append, even if the Append
        // added rows that would be deleted.
        match &self {
            Self::Append { .. } => match &other {
                // Append is compatible with anything that doesn't change the schema
                Self::Append { .. } => false,
                Self::Rewrite { .. } => false,
                Self::CreateIndex { .. } => false,
                Self::Delete { .. } | Self::Update { .. } => false,
                Self::ReserveFragments { .. } => false,
                Self::Project { .. } => false,
                Self::UpdateConfig { .. } => false,
                _ => true,
            },
            Self::Rewrite { .. } => match &other {
                // Rewrite is only compatible with operations that don't touch
                // existing fragments.
                // TODO: it could also be compatible with operations that update
                // fragments we don't touch.
                Self::Append { .. } => false,
                Self::ReserveFragments { .. } => false,
                Self::Delete { .. } | Self::Rewrite { .. } | Self::Update { .. } => {
                    // As long as they rewrite disjoint fragments they shouldn't conflict.
                    self.modifies_same_ids(other)
                }
                Self::Project { .. } => false,
                Self::UpdateConfig { .. } => false,
                _ => true,
            },
            // Restore always succeeds
            Self::Restore { .. } => false,
            // ReserveFragments is compatible with anything that doesn't reset the
            // max fragment id.
            Self::ReserveFragments { .. } => {
                matches!(&other, Self::Overwrite { .. } | Self::Restore { .. })
            }
            Self::CreateIndex { .. } => match &other {
                Self::Append { .. } => false,
                // Indices are identified by UUIDs, so they shouldn't conflict.
                Self::CreateIndex { .. } => false,
                // Although some of the rows we indexed may have been deleted / moved,
                // row ids are still valid, so we allow this optimistically.
                Self::Delete { .. } | Self::Update { .. } => false,
                // Merge & reserve don't change row ids, so this should be fine.
                Self::Merge { .. } => false,
                Self::ReserveFragments { .. } => false,
                // Rewrite likely changed many of the row ids, so our index is
                // likely useless. It should be rebuilt.
                // TODO: we could be smarter here and only invalidate the index
                // if the rewrite changed more than X% of row ids.
                Self::Rewrite { .. } => true,
                Self::UpdateConfig { .. } => false,
                _ => true,
            },
            Self::Delete { .. } | Self::Update { .. } => match &other {
                Self::CreateIndex { .. } => false,
                Self::ReserveFragments { .. } => false,
                Self::Delete { .. } | Self::Rewrite { .. } | Self::Update { .. } => {
                    // If we update the same fragments, we conflict.
                    self.modifies_same_ids(other)
                }
                Self::Project { .. } => false,
                Self::Append { .. } => false,
                Self::UpdateConfig { .. } => false,
                _ => true,
            },
            Self::Overwrite { .. } | Self::UpdateConfig { .. } => match &other {
                Self::Overwrite { .. } | Self::UpdateConfig { .. } => {
                    self.upsert_key_conflict(other)
                }
                _ => false,
            },
            // Merge changes the schema, but preserves row ids, so the only operations
            // it's compatible with is CreateIndex, ReserveFragments, SetMetadata and DeleteMetadata.
            Self::Merge { .. } => !matches!(
                &other,
                Self::CreateIndex { .. }
                    | Self::ReserveFragments { .. }
                    | Self::UpdateConfig { .. }
            ),
            Self::Project { .. } => match &other {
                // Project is compatible with anything that doesn't change the schema
                Self::CreateIndex { .. } => false,
                Self::Overwrite { .. } => false,
                Self::UpdateConfig { .. } => false,
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

impl TryFrom<pb::transaction::Operation> for Operation {
    type Error = Error;

    fn try_from(operation: pb::transaction::Operation) -> std::result::Result<Self, Self::Error> {
        match operation {
            pb::transaction::Operation::Append(pb::transaction::Append { fragments }) => {
                Ok(Self::Append {
                    fragments: fragments
                        .into_iter()
                        .map(Fragment::try_from)
                        .collect::<Result<Vec<_>>>()?,
                })
            }
            pb::transaction::Operation::Delete(pb::transaction::Delete {
                updated_fragments,
                deleted_fragment_ids,
                predicate,
            }) => Ok(Self::Delete {
                updated_fragments: updated_fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<Vec<_>>>()?,
                deleted_fragment_ids,
                predicate,
            }),
            pb::transaction::Operation::Overwrite(pb::transaction::Overwrite {
                fragments,
                schema,
                schema_metadata,
                config_upsert_values,
            }) => {
                let config_upsert_option = if config_upsert_values.is_empty() {
                    Some(config_upsert_values)
                } else {
                    None
                };

                Ok(Self::Overwrite {
                    fragments: fragments
                        .into_iter()
                        .map(Fragment::try_from)
                        .collect::<Result<Vec<_>>>()?,
                    schema: convert_schema(schema, schema_metadata)?,
                    config_upsert_values: config_upsert_option,
                })
            }
            pb::transaction::Operation::ReserveFragments(pb::transaction::ReserveFragments {
                num_fragments,
            }) => Ok(Self::ReserveFragments { num_fragments }),
            pb::transaction::Operation::Rewrite(pb::transaction::Rewrite {
                old_fragments,
                new_fragments,
                groups,
                rewritten_indices,
            }) => {
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

                Ok(Self::Rewrite {
                    groups,
                    rewritten_indices,
                })
            }
            pb::transaction::Operation::CreateIndex(pb::transaction::CreateIndex {
                new_indices,
                removed_indices,
            }) => Ok(Self::CreateIndex {
                new_indices: new_indices
                    .into_iter()
                    .map(Index::try_from)
                    .collect::<Result<_>>()?,
                removed_indices: removed_indices
                    .into_iter()
                    .map(Index::try_from)
                    .collect::<Result<_>>()?,
            }),
            pb::transaction::Operation::Merge(pb::transaction::Merge {
                fragments,
                schema,
                schema_metadata,
            }) => Ok(Self::Merge {
                fragments: fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<Vec<_>>>()?,
                schema: convert_schema(schema, schema_metadata)?,
            }),
            pb::transaction::Operation::Restore(pb::transaction::Restore { version }) => {
                Ok(Self::Restore { version })
            }
            pb::transaction::Operation::Update(pb::transaction::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
            }) => Ok(Self::Update {
                removed_fragment_ids,
                updated_fragments: updated_fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<Vec<_>>>()?,
                new_fragments: new_fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<Vec<_>>>()?,
            }),
            pb::transaction::Operation::Project(pb::transaction::Project { schema }) => {
                Ok(Self::Project {
                    schema: Schema::from(&Fields(schema)),
                })
            }
            pb::transaction::Operation::UpdateConfig(pb::transaction::UpdateConfig {
                upsert_values,
                delete_keys,
            }) => {
                let upsert_values = match upsert_values.len() {
                    0 => None,
                    _ => Some(upsert_values),
                };
                let delete_keys = match delete_keys.len() {
                    0 => None,
                    _ => Some(delete_keys),
                };
                Ok(Self::UpdateConfig {
                    upsert_values,
                    delete_keys,
                })
            }
        }
    }
}

impl TryFrom<pb::transaction::BlobOperation> for Operation {
    type Error = Error;

    fn try_from(blob_op: pb::transaction::BlobOperation) -> std::result::Result<Self, Self::Error> {
        match blob_op {
            pb::transaction::BlobOperation::BlobAppend(pb::transaction::Append { fragments }) => {
                Result::Ok(Self::Append {
                    fragments: fragments
                        .into_iter()
                        .map(Fragment::try_from)
                        .collect::<Result<Vec<_>>>()?,
                })
            }
            pb::transaction::BlobOperation::BlobOverwrite(pb::transaction::Overwrite {
                fragments,
                schema,
                schema_metadata,
                config_upsert_values,
            }) => {
                let config_upsert_option = if config_upsert_values.is_empty() {
                    Some(config_upsert_values)
                } else {
                    None
                };

                Ok(Self::Overwrite {
                    fragments: fragments
                        .into_iter()
                        .map(Fragment::try_from)
                        .collect::<Result<Vec<_>>>()?,
                    schema: convert_schema(schema, schema_metadata)?,
                    config_upsert_values: config_upsert_option,
                })
            }
        }
    }
}

impl From<&Operation> for pb::transaction::Operation {
    fn from(operation: &Operation) -> Self {
        match operation {
            Operation::Append { fragments } => Self::Append(pb::transaction::Append {
                fragments: fragments.iter().map(pb::DataFragment::from).collect(),
            }),
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids,
                predicate,
            } => Self::Delete(pb::transaction::Delete {
                updated_fragments: updated_fragments
                    .iter()
                    .map(pb::DataFragment::from)
                    .collect(),
                deleted_fragment_ids: deleted_fragment_ids.clone(),
                predicate: predicate.clone(),
            }),
            Operation::Overwrite {
                fragments,
                schema,
                config_upsert_values,
            } => Self::Overwrite(pb::transaction::Overwrite {
                fragments: fragments.iter().map(pb::DataFragment::from).collect(),
                schema: Fields::from(schema).0,
                schema_metadata: extract_metadata(&schema.metadata),
                config_upsert_values: config_upsert_values.clone().unwrap_or(Default::default()),
            }),
            Operation::ReserveFragments { num_fragments } => {
                Self::ReserveFragments(pb::transaction::ReserveFragments {
                    num_fragments: *num_fragments,
                })
            }
            Operation::Rewrite {
                groups,
                rewritten_indices,
            } => Self::Rewrite(pb::transaction::Rewrite {
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
            } => Self::CreateIndex(pb::transaction::CreateIndex {
                new_indices: new_indices.iter().map(IndexMetadata::from).collect(),
                removed_indices: removed_indices.iter().map(IndexMetadata::from).collect(),
            }),
            Operation::Merge { fragments, schema } => Self::Merge(pb::transaction::Merge {
                fragments: fragments.iter().map(pb::DataFragment::from).collect(),
                schema: Fields::from(schema).0,
                schema_metadata: extract_metadata(&schema.metadata),
            }),
            Operation::Restore { version } => {
                Self::Restore(pb::transaction::Restore { version: *version })
            }
            Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
            } => Self::Update(pb::transaction::Update {
                removed_fragment_ids: removed_fragment_ids.clone(),
                updated_fragments: updated_fragments
                    .iter()
                    .map(pb::DataFragment::from)
                    .collect(),
                new_fragments: new_fragments.iter().map(pb::DataFragment::from).collect(),
            }),
            Operation::Project { schema } => Self::Project(pb::transaction::Project {
                schema: Fields::from(schema).0,
            }),
            Operation::UpdateConfig {
                upsert_values,
                delete_keys,
            } => Self::UpdateConfig(pb::transaction::UpdateConfig {
                upsert_values: upsert_values.clone().unwrap_or(Default::default()),
                delete_keys: delete_keys.clone().unwrap_or(Default::default()),
            }),
        }
    }
}

impl From<&Operation> for pb::transaction::BlobOperation {
    fn from(operation: &Operation) -> Self {
        match operation {
            Operation::Append { fragments } => Self::BlobAppend(pb::transaction::Append {
                fragments: fragments.iter().map(pb::DataFragment::from).collect(),
            }),
            Operation::Overwrite {
                fragments,
                schema,
                config_upsert_values,
            } => Self::BlobOverwrite(pb::transaction::Overwrite {
                fragments: fragments.iter().map(pb::DataFragment::from).collect(),
                schema: Fields::from(schema).0,
                schema_metadata: extract_metadata(&schema.metadata),
                config_upsert_values: config_upsert_values.clone().unwrap_or(Default::default()),
            }),
            _ => unimplemented!(),
        }
    }
}

fn convert_schema(
    fields: Vec<lance_file::format::pb::Field>,
    metadata: HashMap<String, Vec<u8>>,
) -> Result<Schema> {
    let mut schema = Schema::from(&Fields(fields));
    schema.metadata = metadata
        .into_iter()
        .map(|(k, v)| {
            let value = String::from_utf8(v).map_err(|err| {
                Error::invalid_input(
                    format!("Schema metadata value is not valid UTF-8: {}", err),
                    location!(),
                )
            })?;
            Ok((k, value))
        })
        .collect::<Result<_>>()?;
    Ok(schema)
}

fn extract_metadata(metadata: &HashMap<String, String>) -> HashMap<String, Vec<u8>> {
    metadata
        .iter()
        .map(|(k, v)| (k.clone(), v.clone().into_bytes()))
        .collect()
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
            index_details: None,
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
                config_upsert_values: Some(HashMap::from_iter(vec![(
                    "overwrite-key".to_string(),
                    "value".to_string(),
                )])),
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
            Operation::UpdateConfig {
                upsert_values: Some(HashMap::from_iter(vec![(
                    "lance.test".to_string(),
                    "value".to_string(),
                )])),
                delete_keys: Some(vec!["remove-key".to_string()]),
            },
        ];

        // Transactions and whether they are expected to conflict with each
        // of other_transactions
        let cases = [
            (
                Operation::Append {
                    fragments: vec![fragment0.clone()],
                },
                [false, false, false, true, true, false, false, false, false],
            ),
            (
                Operation::Delete {
                    // Delete that affects fragments different from other transactions
                    updated_fragments: vec![fragment1.clone()],
                    deleted_fragment_ids: vec![],
                    predicate: "x > 2".to_string(),
                },
                [false, false, false, true, true, false, false, true, false],
            ),
            (
                Operation::Delete {
                    // Delete that affects same fragments as other transactions
                    updated_fragments: vec![fragment0.clone(), fragment2.clone()],
                    deleted_fragment_ids: vec![],
                    predicate: "x > 2".to_string(),
                },
                [false, false, true, true, true, true, false, true, false],
            ),
            (
                Operation::Overwrite {
                    fragments: vec![fragment0.clone(), fragment2.clone()],
                    schema: Schema::default(),
                    config_upsert_values: None,
                },
                // No conflicts: overwrite can always happen since it doesn't
                // depend on previous state of the table.
                [
                    false, false, false, false, false, false, false, false, false,
                ],
            ),
            (
                Operation::CreateIndex {
                    new_indices: vec![index0.clone()],
                    removed_indices: vec![index0],
                },
                // Will only conflict with operations that modify row ids.
                [false, false, false, false, true, true, false, false, false],
            ),
            (
                // Rewrite that affects different fragments
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments: vec![fragment1],
                        new_fragments: vec![fragment0.clone()],
                    }],
                    rewritten_indices: Vec::new(),
                },
                [false, true, false, true, true, false, false, true, false],
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
                [false, true, true, true, true, true, false, true, false],
            ),
            (
                Operation::Merge {
                    fragments: vec![fragment0.clone(), fragment2.clone()],
                    schema: Schema::default(),
                },
                // Merge conflicts with everything except CreateIndex and ReserveFragments.
                [true, false, true, true, true, true, false, true, false],
            ),
            (
                Operation::ReserveFragments { num_fragments: 2 },
                // ReserveFragments only conflicts with Overwrite and Restore.
                [false, false, false, false, true, false, false, false, false],
            ),
            (
                Operation::Update {
                    // Update that affects same fragments as other transactions
                    updated_fragments: vec![fragment0],
                    removed_fragment_ids: vec![],
                    new_fragments: vec![fragment2],
                },
                [false, false, true, true, true, true, false, true, false],
            ),
            (
                // Update config that should not conflict with anything
                Operation::UpdateConfig {
                    upsert_values: Some(HashMap::from_iter(vec![(
                        "other-key".to_string(),
                        "new-value".to_string(),
                    )])),
                    delete_keys: None,
                },
                [
                    false, false, false, false, false, false, false, false, false,
                ],
            ),
            (
                // Update config that conflicts with key being upserted by other UpdateConfig operation
                Operation::UpdateConfig {
                    upsert_values: Some(HashMap::from_iter(vec![(
                        "lance.test".to_string(),
                        "new-value".to_string(),
                    )])),
                    delete_keys: None,
                },
                [false, false, false, false, false, false, false, false, true],
            ),
            (
                // Update config that conflicts with key being deleted by other UpdateConfig operation
                Operation::UpdateConfig {
                    upsert_values: Some(HashMap::from_iter(vec![(
                        "remove-key".to_string(),
                        "new-value".to_string(),
                    )])),
                    delete_keys: None,
                },
                [false, false, false, false, false, false, false, false, true],
            ),
            (
                // Delete config keys currently being deleted by other UpdateConfig operation
                Operation::UpdateConfig {
                    upsert_values: None,
                    delete_keys: Some(vec!["remove-key".to_string()]),
                },
                [
                    false, false, false, false, false, false, false, false, false,
                ],
            ),
            (
                // Delete config keys currently being upserted by other UpdateConfig operation
                Operation::UpdateConfig {
                    upsert_values: None,
                    delete_keys: Some(vec!["lance.test".to_string()]),
                },
                [false, false, false, false, false, false, false, false, true],
            ),
        ];

        for (operation, expected_conflicts) in &cases {
            for (other, expected_conflict) in other_operations.iter().zip(expected_conflicts) {
                assert_eq!(
                    operation.conflicts_with(other),
                    *expected_conflict,
                    "Operation {:?} should {} with {:?}",
                    operation,
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
}
