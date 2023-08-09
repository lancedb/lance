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

// TODO:
// [ ] Module docs
// [ ] Conflict resolution Tests
// [x] Serialization
// [x] Deserialization
// [x] Commit function
// [ ] Commit tests
// [ ] Python API
// [ ] Python API docs

use std::{collections::HashSet, sync::Arc};

use crate::{
    datatypes::Schema,
    format::{pb, Fragment, Manifest},
};

use super::{feature_flags::apply_feature_flags, ManifestWriteConfig};
use crate::{Error, Result};

/// A change to a dataset that can be retried
///
/// This contains enough information to be able to build the next manifest,
/// given the current manifest.
#[derive(Debug, Clone)]
pub struct Transaction {
    pub read_version: u64,
    pub operation: Operation,
    pub tag: Option<String>,
}
// phalanx/src/catalog.rs

// TODO: This should have a protobuf message

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
    /// Overwrite the entire dataset with the given fragments.
    Overwrite {
        fragments: Vec<Fragment>,
        schema: Schema,
    },
    /// A new index has been created.
    CreateIndex,
    /// Data is rewritten but *not* modified. This is used for things like
    /// compaction or re-ordering. Contains the old fragments and the new
    /// ones that have been replaced.
    Rewrite {
        old_fragments: Vec<Fragment>,
        new_fragments: Vec<Fragment>,
    },
    /// Merge a new column in
    Merge {
        fragments: Vec<Fragment>,
        schema: Schema,
    },
    // TODO: a Custom one to allow arbitrary changes that will have same conflict
    // resolution rules as unknown?
}

impl Transaction {
    pub fn new(read_version: u64, operation: Operation, tag: Option<String>) -> Self {
        Self {
            read_version,
            operation,
            tag,
        }
    }

    /// Returns true if the transaction cannot be committed if the other
    /// transaction is committed first.
    pub fn conflicts_with(&self, other: &Self) -> bool {
        // TODO: this assume IsolationLevel is Serializable, but we could also
        // support Snapshot Isolation, which is more permissive.
        match &self.operation {
            Operation::Append { .. } => match &other.operation {
                // Append is compatible with anything that doesn't change the schema
                Operation::Append { .. } => false,
                Operation::Rewrite { .. } => false,
                Operation::CreateIndex { .. } => false,
                Operation::Delete { .. } => false,
                _ => true,
            },
            Operation::Rewrite { old_fragments, .. } => match &other.operation {
                // Rewrite is only compatible with operations that don't touch
                // existing fragments.
                // TODO: it could also be compatible with operations that update
                // fragments we don't touch.
                Operation::Append { .. } => false,
                Operation::Delete {
                    updated_fragments,
                    deleted_fragment_ids,
                    ..
                } => {
                    // If we rewrote any fragments that were modified by delete,
                    // we conflict.
                    let left_ids: HashSet<u64> = old_fragments.iter().map(|f| f.id).collect();
                    deleted_fragment_ids.iter().any(|f| left_ids.contains(f))
                        || updated_fragments.iter().any(|f| left_ids.contains(&f.id))
                }
                _ => true,
            },
            // Overwrite always succeeds
            Operation::Overwrite { .. } => false,
            Operation::CreateIndex { .. } => match &other.operation {
                Operation::Append { .. } => false,
                // Indices are identified by UUIDs, so they shouldn't conflict.
                Operation::CreateIndex { .. } => false,
                // Rewrite likely changed many of the row ids, so our index is
                // likely useless. It should be rebuilt.
                // TODO: we could be smarter here and only invalidate the index
                // if the rewrite changed more than X% of row ids.
                Operation::Rewrite { .. } => true,
                _ => true,
            },
            Operation::Delete {
                updated_fragments: left_fragments,
                ..
            } => match &other.operation {
                Operation::CreateIndex { .. } => false,
                Operation::Delete {
                    updated_fragments: right_fragments,
                    ..
                } => {
                    // If we update the same fragments, we conflict.
                    let left_ids: HashSet<u64> = left_fragments.iter().map(|f| f.id).collect();
                    right_fragments.iter().any(|f| left_ids.contains(&f.id))
                }
                _ => true,
            },
            _ => true,
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

    /// Create a new manifest from the current manifest and the transaction.
    pub(crate) fn build_manifest(
        &self,
        current_manifest: &Manifest,
        config: &ManifestWriteConfig,
    ) -> Result<Manifest> {
        // Get the schema and the final fragment list
        let schema = match self.operation {
            Operation::Overwrite { ref schema, .. } => schema.clone(),
            Operation::Merge { ref schema, .. } => schema.clone(),
            _ => current_manifest.schema.clone(),
        };

        let mut fragment_id = current_manifest.max_fragment_id().unwrap_or(0) + 1;
        let mut final_fragments = Vec::new();
        match &self.operation {
            Operation::Append { ref fragments } => {
                final_fragments.extend(current_manifest.fragments.as_ref().clone());
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
                final_fragments.extend(current_manifest.fragments.as_ref().clone());
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
            }
            Operation::Rewrite {
                ref new_fragments, ..
            } => {
                final_fragments.extend(Self::fragments_with_ids(
                    new_fragments.clone(),
                    &mut fragment_id,
                ));
            }
            Operation::CreateIndex { .. } => {
                final_fragments.extend(current_manifest.fragments.as_ref().clone())
            }
            Operation::Merge { ref fragments, .. } => {
                final_fragments.extend(Self::fragments_with_ids(
                    fragments.clone(),
                    &mut fragment_id,
                ));
            }
        };

        let mut manifest =
            Manifest::new_from_previous(current_manifest, &schema, Arc::new(final_fragments));

        manifest.tag = self.tag.clone();

        if config.auto_set_feature_flags {
            apply_feature_flags(&mut manifest);
        }
        manifest.set_timestamp(config.timestamp);

        manifest.update_max_fragment_id();

        manifest.set_transaction(Some(self))?;

        Ok(manifest)
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
            })) => Operation::Overwrite {
                fragments: fragments.iter().map(Fragment::from).collect(),
                schema: Schema::from(schema),
            },
            Some(pb::transaction::Operation::Rewrite(pb::transaction::Rewrite {
                old_fragments,
                new_fragments,
            })) => Operation::Rewrite {
                old_fragments: old_fragments.iter().map(Fragment::from).collect(),
                new_fragments: new_fragments.iter().map(Fragment::from).collect(),
            },
            Some(pb::transaction::Operation::CreateIndex(_)) => Operation::CreateIndex,
            Some(pb::transaction::Operation::Merge(pb::transaction::Merge {
                fragments,
                schema,
            })) => Operation::Merge {
                fragments: fragments.iter().map(Fragment::from).collect(),
                schema: Schema::from(schema),
            },
            None => {
                return Err(Error::Internal {
                    message: "Transaction message did not contain an operation".to_string(),
                });
            }
        };
        Ok(Self {
            read_version: message.read_version,
            operation,
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
                })
            }
            Operation::Rewrite {
                old_fragments,
                new_fragments,
            } => pb::transaction::Operation::Rewrite(pb::transaction::Rewrite {
                old_fragments: old_fragments.iter().map(pb::DataFragment::from).collect(),
                new_fragments: new_fragments.iter().map(pb::DataFragment::from).collect(),
            }),
            Operation::CreateIndex => {
                pb::transaction::Operation::CreateIndex(pb::transaction::CreateIndex {})
            }
            Operation::Merge { fragments, schema } => {
                pb::transaction::Operation::Merge(pb::transaction::Merge {
                    fragments: fragments.iter().map(pb::DataFragment::from).collect(),
                    schema: schema.into(),
                })
            }
        };

        Self {
            read_version: value.read_version,
            operation: Some(operation),
            tag: value.tag.clone().unwrap_or("".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conflicts() {
        let fragment0 = Fragment::new(0);
        let fragment1 = Fragment::new(1);
        let fragment2 = Fragment::new(2);
        // The transactions that will be checked against
        let other_operations = [
            Operation::Append {
                fragments: vec![fragment0.clone()],
            },
            Operation::CreateIndex,
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
                old_fragments: vec![fragment0.clone()],
                new_fragments: vec![fragment2.clone()],
            },
        ];
        let other_transactions = other_operations
            .iter()
            .map(|op| Transaction::new(0, op.clone(), None))
            .collect::<Vec<_>>();

        // Transactions and whether they are expected to conflict with each
        // of other_transactions
        let cases = [(
            Operation::Append {
                fragments: vec![fragment0.clone()],
            },
            [false, false, false, true, true, false],
            // TODO: more cases
        )];

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
