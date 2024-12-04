// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::format::{pb, Fragment, Index, Manifest};
use deepsize::DeepSizeOf;
use futures::future::Either;
use lance_core::{datatypes::Schema, Error, Result};

/// A change to a [`Manifest`].
#[derive(Debug, Clone, DeepSizeOf)]
pub enum Action {
    // Fragment changes
    AddFragments {
        fragments: Vec<Fragment>,
    },
    DeleteFragments {
        deleted_fragment_ids: Vec<u64>,
    },
    UpdateFragments {
        fragments: Vec<Fragment>,
    },
    ReplaceFragments {
        fragments: Vec<Fragment>,
    },

    // Schema changes
    ReplaceSchema {
        schema: Schema,
    },

    // Config changes
    UpsertConfig {
        upsert_values: HashMap<String, String>,
        delete_keys: Vec<String>,
    },

    // Index changes
    AddIndices {
        index: Vec<Index>,
    },
    RemoveIndices {
        index: Vec<Index>,
    },

    ReserveFragments {
        num_fragments: u32,
    },
}

impl TryFrom<pb::transaction::action::Action> for Action {
    type Error = Error;

    fn try_from(value: pb::transaction::action::Action) -> std::result::Result<Self, Self::Error> {
        use pb::transaction::action::Action::*;
        match value {
            AddFragments(action) => Ok(Action::AddFragments {
                fragments: action
                    .fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<_>>()?,
            }),
            DeleteFragments(action) => Ok(Action::DeleteFragments {
                deleted_fragment_ids: action.deleted_fragment_ids,
            }),
            UpdateFragments(action) => Ok(Action::UpdateFragments {
                fragments: action
                    .updated_fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<_>>()?,
            }),
            ReplaceFragments(action) => Ok(Action::ReplaceFragments {
                fragments: action
                    .fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<_>>()?,
            }),
            ReplaceSchema(action) => {
                let mut schema = todo!();
                schema.metadata = action.schema_metadata;
                Ok(Action::ReplaceSchema { schema })
            }
            UpsertConfig(action) => Ok(Action::UpsertConfig {
                upsert_values: action.upsert_values,
                delete_keys: action.delete_keys,
            }),
            AddIndices(action) => Ok(Action::AddIndices {
                index: action
                    .indices
                    .into_iter()
                    .map(Index::try_from)
                    .collect::<Result<_>>()?,
            }),
            RemoveIndices(action) => Ok(Action::RemoveIndices {
                index: action
                    .indices
                    .into_iter()
                    .map(Index::try_from)
                    .collect::<Result<_>>()?,
            }),
            ReserveFragments(action) => Ok(Action::ReserveFragments {
                num_fragments: action.num_fragments,
            }),
        }
    }
}

impl Action {
    pub fn modified_fragment_ids(&self) -> impl Iterator<Item = u64> + '_ {
        match self {
            Action::DeleteFragments {
                deleted_fragment_ids,
            } => Either::Left(deleted_fragment_ids.iter().copied()),
            Action::UpdateFragments { fragments } => Either::Right(fragments.iter().map(|f| f.id)),
            // We don't care about add or replace, since they are new fragments
            _ => Either::Left([].iter().copied()),
        }
    }
}

pub fn apply_actions(
    manifest: &mut Manifest,
    indices: &mut Vec<Index>,
    mut actions: &mut Vec<Action>,
) -> Result<()> {
    let mut next_fragment_id = manifest.max_fragment_id().map(|id| id + 1).unwrap_or(0);

    let mut next_row_id = if manifest.uses_move_stable_row_ids() {
        Some(manifest.next_row_id)
    } else {
        None
    };

    let mut final_fragments = manifest
        .fragments
        .iter()
        .map(|f| (f.id, f))
        .collect::<HashMap<_, _>>();

    for action in actions {
        match action {
            Action::AddFragments { mut fragments } => {
                assign_fragment_ids(&mut fragments, &mut next_fragment_id);
                assign_row_ids(&mut fragments, &mut next_row_id);
                final_fragments.extend(fragments.iter().map(|f| (f.id, f)));
            }
            Action::ReplaceFragments { fragments } => {
                let mut fragments = fragments.clone();
                // Can reset the fragment ids, but only if we also delete all the indices
                next_fragment_id = 0;
                indices.clear();
                assign_fragment_ids(&mut fragments, &mut next_fragment_id);
                assign_row_ids(&mut fragments, &mut next_row_id);
                final_fragments.clear();
                final_fragments.extend(fragments.iter().map(|f| (f.id, f)));
            }
            Action::DeleteFragments {
                deleted_fragment_ids,
            } => {
                for id in deleted_fragment_ids {
                    final_fragments.remove(&id);
                }
            }
            _ => todo!(),
        }
    }

    let mut final_fragments = final_fragments
        .into_iter()
        .map(|(_, f)| f.clone())
        .collect::<Vec<_>>();
    final_fragments.sort_by_key(|f| f.id);
    manifest.fragments = Arc::new(final_fragments);

    Ok(())
}

fn assign_fragment_ids(fragments: &mut [Fragment], next_fragment_id: &mut u64) {
    for fragment in fragments {
        // If the id is already non-zero, let's assume that it is committing an
        // id that was already assigned.
        if fragment.id == 0 {
            fragment.id = *next_fragment_id;
            *next_fragment_id += 1;
        }
    }
}

fn assign_row_ids(fragments: &mut [Fragment], next_row_id: &mut Option<u64>) {
    todo!()
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

fn retain_relevant_data_files(fragments: &mut [Fragment], schema: &Schema) {
    // We might have removed all fields for certain data files, so
    // we should remove the data files that are no longer relevant.
    let remaining_field_ids = schema
        .fields_pre_order()
        .map(|f| f.id)
        .collect::<HashSet<_>>();
    for fragment in fragments.iter_mut() {
        fragment.files.retain(|file| {
            file.fields
                .iter()
                .any(|field_id| remaining_field_ids.contains(field_id))
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_actions() {
        // Test apply_actions
    }
}
