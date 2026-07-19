// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! PROTOTYPE (discussion #7499): ε-buffer action builders + routing key.
//!
//! Actions are the messages buffered in every node. Application (newest-wins by
//! msn) lives in [`crate::betree::node::apply_actions`].

use crate::format::pb::{self, fragment_action::Action};
use crate::format::{DataFile, Fragment};

/// The backfill / add-column action: attach `file` to fragment `frag_id`.
pub fn add_data_file(frag_id: u64, file: &DataFile) -> pb::FragmentAction {
    pb::FragmentAction {
        action: Some(Action::AddDataFile(pb::AddDataFile {
            frag_id,
            file: Some(pb::DataFile::from(file)),
        })),
    }
}

/// A data-replacement half: drop the data file at `path` from `frag_id`.
pub fn remove_data_file(frag_id: u64, path: impl Into<String>) -> pb::FragmentAction {
    pb::FragmentAction {
        action: Some(Action::RemoveDataFile(pb::RemoveDataFile {
            frag_id,
            path: path.into(),
        })),
    }
}

/// Add a whole fragment.
pub fn add_fragment(fragment: &Fragment) -> pb::FragmentAction {
    pb::FragmentAction {
        action: Some(Action::AddFragment(pb::DataFragment::from(fragment))),
    }
}

/// Remove a whole fragment (tombstone).
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
