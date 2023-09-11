// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
//! |                  | Append | Delete | Overwrite/Create | Create Index | Rewrite | Merge |
//! |------------------|--------|--------|------------------|--------------|---------|-------|
//! | Append           | ✅     | ✅     | ❌               | ✅           | ✅      | ❌    |
//! | Delete           | ❌     | (1)    | ❌               | ✅           | (1)     | ❌    |
//! | Overwrite/Create | ❌     | ❌     | ❌               | ❌           | ❌      | ❌    |
//! | Create index     | ✅     | ✅     | ❌               | ✅           | ✅      | ✅    |
//! | Rewrite          | ✅     | (1)    | ❌               | ❌           | (1)     | ❌    |
//! | Merge            | ❌     | ❌     | ❌               | ❌           | ✅      | ❌    |
//!
//! (1) Delete and rewrite are compatible with each other and themselves only if
//! they affect distinct fragments. Otherwise, they conflict.

use std::{collections::HashSet, sync::Arc};

use object_store::path::Path;

use crate::{
    datatypes::Schema,
    format::Index,
    format::{
        pb::{self, IndexMetadata},
        Fragment, Manifest,
    },
    io::{read_manifest, reader::read_manifest_indexes, ObjectStore},
};

use super::{feature_flags::apply_feature_flags, ManifestWriteConfig};
use crate::{Error, Result};

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
    },
    /// Data is rewritten but *not* modified. This is used for things like
    /// compaction or re-ordering. Contains the old fragments and the new
    /// ones that have been replaced.
    Rewrite { groups: Vec<RewriteGroup> },
    /// Merge a new column in
    Merge {
        fragments: Vec<Fragment>,
        schema: Schema,
    },
    /// Restore an old version of the database
    Restore { version: u64 },
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
            Self::Rewrite { groups } => Box::new(
                groups
                    .iter()
                    .flat_map(|f| f.old_fragments.iter().map(|f| f.id)),
            ),
            Self::Merge { fragments, .. } => Box::new(fragments.iter().map(|f| f.id)),
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
            Self::Restore { .. } => "Restore",
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
        // TODO: this assume IsolationLevel is Serializable, but we could also
        // support Snapshot Isolation, which is more permissive. In particular,
        // it would allow a Delete transaction to succeed after a concurrent
        // Append, even if the Append added rows that would be deleted.
        match &self.operation {
            Operation::Append { .. } => match &other.operation {
                // Append is compatible with anything that doesn't change the schema
                Operation::Append { .. } => false,
                Operation::Rewrite { .. } => false,
                Operation::CreateIndex { .. } => false,
                Operation::Delete { .. } => false,
                _ => true,
            },
            Operation::Rewrite { .. } => match &other.operation {
                // Rewrite is only compatible with operations that don't touch
                // existing fragments.
                // TODO: it could also be compatible with operations that update
                // fragments we don't touch.
                Operation::Append { .. } => false,
                Operation::Delete { .. } => {
                    // If we rewrote any fragments that were modified by delete,
                    // we conflict.
                    self.operation.modifies_same_ids(&other.operation)
                }
                Operation::Rewrite { .. } => {
                    // As long as they rewrite disjoint fragments they shouldn't conflict.
                    self.operation.modifies_same_ids(&other.operation)
                }
                _ => true,
            },
            // Overwrite and Restore always succeed
            Operation::Overwrite { .. } => false,
            Operation::Restore { .. } => false,
            Operation::CreateIndex { .. } => match &other.operation {
                Operation::Append { .. } => false,
                // Indices are identified by UUIDs, so they shouldn't conflict.
                Operation::CreateIndex { .. } => false,
                // Although some of the rows we indexed may have been deleted,
                // row ids are still valid, so we allow this optimistically.
                Operation::Delete { .. } => false,
                // Merge doesn't change row ids, so this should be fine.
                Operation::Merge { .. } => false,
                // Rewrite likely changed many of the row ids, so our index is
                // likely useless. It should be rebuilt.
                // TODO: we could be smarter here and only invalidate the index
                // if the rewrite changed more than X% of row ids.
                Operation::Rewrite { .. } => true,
                _ => true,
            },
            Operation::Delete { .. } => match &other.operation {
                Operation::CreateIndex { .. } => false,
                Operation::Delete { .. } => {
                    // If we update the same fragments, we conflict.
                    self.operation.modifies_same_ids(&other.operation)
                }
                Operation::Rewrite { .. } => {
                    // If we update any fragments that were rewritten, we conflict.
                    self.operation.modifies_same_ids(&other.operation)
                }
                _ => true,
            },
            // Merge changes the schema, but preserves row ids, so the only operation
            // it's compatible with is CreateIndex.
            Operation::Merge { .. } => !matches!(&other.operation, Operation::CreateIndex { .. }),
        }
    }

    fn fragments_with_ids<'a, T>(
        new_fragments: T,
        fragment_id: &'a mut u64,
    ) -> impl Iterator<Item = Fragment> + 'a
    where
        T: IntoIterator<Item = Fragment> + 'a,
    {
        new_fragments.into_iter().map(|mut f| {
            f.id = *fragment_id;
            *fragment_id += 1;
            f
        })
    }

    pub(crate) async fn restore_old_manifest(
        object_store: &ObjectStore,
        base_path: &Path,
        version: u64,
        config: &ManifestWriteConfig,
        tx_path: &str,
    ) -> Result<(Manifest, Vec<Index>)> {
        let path = object_store
            .commit_handler
            .resolve_version(base_path, version, object_store)
            .await?;
        let mut manifest = read_manifest(object_store, &path).await?;
        manifest.set_timestamp(config.timestamp);
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
        // Get the schema and the final fragment list
        let schema = match self.operation {
            Operation::Overwrite { ref schema, .. } => schema.clone(),
            Operation::Merge { ref schema, .. } => schema.clone(),
            _ => {
                if let Some(current_manifest) = current_manifest {
                    current_manifest.schema.clone()
                } else {
                    return Err(Error::Internal {
                        message: "Cannot create a new dataset without a schema".to_string(),
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

        let maybe_existing_fragments =
            current_manifest
                .map(|m| m.fragments.as_ref())
                .ok_or_else(|| Error::Internal {
                    message: format!(
                        "No current manifest was provided while building manifest for operation {}",
                        self.operation.name()
                    ),
                });

        match &self.operation {
            Operation::Append { ref fragments } => {
                final_fragments.extend(maybe_existing_fragments?.clone());
                final_fragments.extend(Self::fragments_with_ids(
                    fragments.clone(),
                    &mut fragment_id,
                ));
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
            Operation::Overwrite { ref fragments, .. } => {
                final_fragments.extend(Self::fragments_with_ids(
                    fragments.clone(),
                    &mut fragment_id,
                ));
                final_indices = Vec::new();
            }
            Operation::Rewrite { ref groups, .. } => {
                final_fragments.extend(maybe_existing_fragments?.clone());
                let current_version = current_manifest.map(|m| m.version).unwrap_or_default();
                Self::handle_rewrite_fragments(
                    &mut final_fragments,
                    groups,
                    fragment_id,
                    current_version,
                )?;
            }
            Operation::CreateIndex { new_indices } => {
                final_fragments.extend(maybe_existing_fragments?.clone());
                final_indices.retain(|existing_index| {
                    !new_indices
                        .iter()
                        .any(|new_index| new_index.name == existing_index.name)
                });
                final_indices.extend(new_indices.clone());
            }
            Operation::Merge { ref fragments, .. } => {
                final_fragments.extend(fragments.clone());
            }
            Operation::Restore { .. } => {
                unreachable!()
            }
        };

        let mut manifest = if let Some(current_manifest) = current_manifest {
            Manifest::new_from_previous(current_manifest, &schema, Arc::new(final_fragments))
        } else {
            Manifest::new(&schema, Arc::new(final_fragments))
        };

        manifest.tag = self.tag.clone();

        if config.auto_set_feature_flags {
            apply_feature_flags(&mut manifest);
        }
        manifest.set_timestamp(config.timestamp);

        manifest.update_max_fragment_id();

        manifest.transaction_file = Some(transaction_file_path.to_string());

        Ok((manifest, final_indices))
    }

    fn handle_rewrite_fragments(
        final_fragments: &mut Vec<Fragment>,
        groups: &[RewriteGroup],
        mut fragment_id: u64,
        version: u64,
    ) -> Result<()> {
        for group in groups {
            // If the old fragments are contiguous, find the range
            let replace_range = {
                let start = final_fragments.iter().enumerate().find(|(_, f)| f.id == group.old_fragments[0].id)
                    .ok_or_else(|| Error::CommitConflict { version, source:
                        format!("dataset does not contain a fragment a rewrite operation wants to replace: id={}", group.old_fragments[0].id).into() })?.0;

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

            let new_fragments =
                Self::fragments_with_ids(group.new_fragments.clone(), &mut fragment_id);
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
}

impl TryFrom<&pb::Transaction> for Transaction {
    type Error = Error;

    fn try_from(message: &pb::Transaction) -> Result<Self> {
        let operation = match &message.operation {
            Some(pb::transaction::Operation::Append(pb::transaction::Append { fragments })) => {
                Operation::Append {
                    fragments: fragments.iter().map(Fragment::from).collect(),
                }
            }
            Some(pb::transaction::Operation::Delete(pb::transaction::Delete {
                updated_fragments,
                deleted_fragment_ids,
                predicate,
            })) => Operation::Delete {
                updated_fragments: updated_fragments.iter().map(Fragment::from).collect(),
                deleted_fragment_ids: deleted_fragment_ids.clone(),
                predicate: predicate.clone(),
            },
            Some(pb::transaction::Operation::Overwrite(pb::transaction::Overwrite {
                fragments,
                schema,
                schema_metadata: _schema_metadata, // TODO: handle metadata
            })) => Operation::Overwrite {
                fragments: fragments.iter().map(Fragment::from).collect(),
                schema: Schema::from(schema),
            },
            Some(pb::transaction::Operation::Rewrite(pb::transaction::Rewrite {
                old_fragments,
                new_fragments,
                groups,
            })) => {
                let groups = if !groups.is_empty() {
                    groups
                        .iter()
                        .map(RewriteGroup::try_from)
                        .collect::<Result<_>>()?
                } else {
                    vec![RewriteGroup {
                        old_fragments: old_fragments.iter().map(Fragment::from).collect(),
                        new_fragments: new_fragments.iter().map(Fragment::from).collect(),
                    }]
                };
                Operation::Rewrite { groups }
            }
            Some(pb::transaction::Operation::CreateIndex(pb::transaction::CreateIndex {
                new_indices,
            })) => Operation::CreateIndex {
                new_indices: new_indices
                    .iter()
                    .map(Index::try_from)
                    .collect::<Result<_>>()?,
            },
            Some(pb::transaction::Operation::Merge(pb::transaction::Merge {
                fragments,
                schema,
                schema_metadata: _schema_metadata, // TODO: handle metadata
            })) => Operation::Merge {
                fragments: fragments.iter().map(Fragment::from).collect(),
                schema: Schema::from(schema),
            },
            Some(pb::transaction::Operation::Restore(pb::transaction::Restore { version })) => {
                Operation::Restore { version: *version }
            }
            None => {
                return Err(Error::Internal {
                    message: "Transaction message did not contain an operation".to_string(),
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

impl TryFrom<&pb::transaction::rewrite::RewriteGroup> for RewriteGroup {
    type Error = Error;

    fn try_from(message: &pb::transaction::rewrite::RewriteGroup) -> Result<Self> {
        Ok(Self {
            old_fragments: message.old_fragments.iter().map(Fragment::from).collect(),
            new_fragments: message.new_fragments.iter().map(Fragment::from).collect(),
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
                    schema: schema.into(),
                    schema_metadata: Default::default(), // TODO: handle metadata
                })
            }
            Operation::Rewrite { groups } => {
                pb::transaction::Operation::Rewrite(pb::transaction::Rewrite {
                    groups: groups
                        .iter()
                        .map(pb::transaction::rewrite::RewriteGroup::from)
                        .collect(),
                    ..Default::default()
                })
            }
            Operation::CreateIndex { new_indices } => {
                pb::transaction::Operation::CreateIndex(pb::transaction::CreateIndex {
                    new_indices: new_indices.iter().map(IndexMetadata::from).collect(),
                })
            }
            Operation::Merge { fragments, schema } => {
                pb::transaction::Operation::Merge(pb::transaction::Merge {
                    fragments: fragments.iter().map(pb::DataFragment::from).collect(),
                    schema: schema.into(),
                    schema_metadata: Default::default(), // TODO: handle metadata
                })
            }
            Operation::Restore { version } => {
                pb::transaction::Operation::Restore(pb::transaction::Restore { version: *version })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conflicts() {
        let index0 = Index::new(uuid::Uuid::new_v4(), "test", &[0], 1);
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
                [false, false, false, true, true, false],
            ),
            (
                Operation::Delete {
                    // Delete that affects fragments different from other transactions
                    updated_fragments: vec![fragment1.clone()],
                    deleted_fragment_ids: vec![],
                    predicate: "x > 2".to_string(),
                },
                [true, false, false, true, true, false],
            ),
            (
                Operation::Delete {
                    // Delete that affects same fragments as other transactions
                    updated_fragments: vec![fragment0.clone(), fragment2.clone()],
                    deleted_fragment_ids: vec![],
                    predicate: "x > 2".to_string(),
                },
                [true, false, true, true, true, true],
            ),
            (
                Operation::Overwrite {
                    fragments: vec![fragment0.clone(), fragment2.clone()],
                    schema: Schema::default(),
                },
                // No conflicts: overwrite can always happen since it doesn't
                // depend on previous state of the table.
                [false, false, false, false, false, false],
            ),
            (
                Operation::CreateIndex {
                    new_indices: vec![index0],
                },
                // Will only conflict with operations that modify row ids.
                [false, false, false, false, true, true],
            ),
            (
                // Rewrite that affects different fragments
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments: vec![fragment1.clone()],
                        new_fragments: vec![fragment0.clone()],
                    }],
                },
                [false, true, false, true, true, false],
            ),
            (
                // Rewrite that affects the same fragments
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments: vec![fragment0.clone(), fragment2.clone()],
                        new_fragments: vec![fragment0.clone()],
                    }],
                },
                [false, true, true, true, true, true],
            ),
            (
                Operation::Merge {
                    fragments: vec![fragment0.clone(), fragment2.clone()],
                    schema: Schema::default(),
                },
                // Merge conflicts with everything except CreateIndex.
                [true, false, true, true, true, true],
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
}
