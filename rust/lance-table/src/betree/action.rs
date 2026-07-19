// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! PROTOTYPE (discussion #7499): ε-buffer actions.
//!
//! Actions are the messages buffered in every node. A commit appends actions
//! instead of rewriting fragment state; overflow flushes a batch of them down
//! one level. This module builds actions and replays them onto a fragment map.

use std::collections::BTreeMap;

use prost::Message;

use crate::format::pb::{self, fragment_action::Action};
use crate::format::{DataFile, DeletionFile, Fragment};
use lance_core::{Error, Result};

/// Build the backfill / add-column action: attach `file` to fragment `frag_id`.
pub fn add_data_file(frag_id: u64, file: &DataFile) -> pb::FragmentAction {
    pb::FragmentAction {
        action: Some(Action::AddDataFile(pb::AddDataFile {
            frag_id,
            file: Some(pb::DataFile::from(file)),
        })),
    }
}

/// Build a data-replacement half: drop the data file at `path` from `frag_id`.
pub fn remove_data_file(frag_id: u64, path: impl Into<String>) -> pb::FragmentAction {
    pb::FragmentAction {
        action: Some(Action::RemoveDataFile(pb::RemoveDataFile {
            frag_id,
            path: path.into(),
        })),
    }
}

/// Build an add-fragment action.
pub fn add_fragment(fragment: &Fragment) -> pb::FragmentAction {
    pb::FragmentAction {
        action: Some(Action::AddFragment(pb::DataFragment::from(fragment))),
    }
}

/// Build a remove-fragment action.
pub fn remove_fragment(frag_id: u64) -> pb::FragmentAction {
    pb::FragmentAction {
        action: Some(Action::RemoveFragment(frag_id)),
    }
}

/// The fragment id an action targets, used to route it to the owning child.
pub fn target_frag_id(action: &pb::FragmentAction) -> Option<u64> {
    match action.action.as_ref()? {
        Action::AddFragment(f) => Some(f.id),
        Action::RemoveFragment(id) => Some(*id),
        Action::AddDataFile(a) => Some(a.frag_id),
        Action::RemoveDataFile(a) => Some(a.frag_id),
        Action::AddDeletionFile(a) => Some(a.frag_id),
    }
}

/// Encoded byte size of an action, for ε-buffer cap accounting.
pub fn encoded_len(action: &pb::FragmentAction) -> usize {
    action.encoded_len()
}

/// Replay one action onto an id-keyed fragment map (newest-wins via call order).
///
/// Used both when flushing a batch into a child and when overlaying the root
/// buffer during materialization.
pub fn apply(frags: &mut BTreeMap<u64, Fragment>, action: pb::FragmentAction) -> Result<()> {
    let Some(action) = action.action else {
        return Ok(());
    };
    match action {
        Action::AddFragment(f) => {
            let fragment = Fragment::try_from(f)?;
            frags.insert(fragment.id, fragment);
        }
        Action::RemoveFragment(id) => {
            frags.remove(&id);
        }
        Action::AddDataFile(a) => {
            let file = DataFile::try_from(
                a.file
                    .ok_or_else(|| Error::invalid_input("AddDataFile action missing file"))?,
            )?;
            let fragment = frags.get_mut(&a.frag_id).ok_or_else(|| {
                Error::invalid_input(format!("AddDataFile for unknown fragment {}", a.frag_id))
            })?;
            fragment.files.push(file);
        }
        Action::RemoveDataFile(a) => {
            if let Some(fragment) = frags.get_mut(&a.frag_id) {
                fragment.files.retain(|f| f.path != a.path);
            }
        }
        Action::AddDeletionFile(a) => {
            // Only set when the action actually carries a deletion file — an empty
            // AddDeletionFile must not clear an existing one.
            if let (Some(df), Some(fragment)) = (a.deletion_file, frags.get_mut(&a.frag_id)) {
                fragment.deletion_file = Some(DeletionFile::try_from(df)?);
            }
        }
    }
    Ok(())
}
