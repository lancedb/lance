// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Snapshot-level field assignment metadata.

use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use serde::{Deserialize, Serialize};

use super::pb;

/// Reference to an immutable field-assignment object under a dataset root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct FieldAssignmentFile {
    /// Path relative to the referenced dataset root.
    pub path: String,
    /// Exact encoded size in bytes.
    pub size_bytes: u64,
    /// Optional external dataset base ID.
    pub base_id: Option<u32>,
}

impl From<&FieldAssignmentFile> for pb::FieldAssignmentFile {
    fn from(value: &FieldAssignmentFile) -> Self {
        Self {
            path: value.path.clone(),
            size_bytes: value.size_bytes,
            base_id: value.base_id,
        }
    }
}

impl TryFrom<pb::FieldAssignmentFile> for FieldAssignmentFile {
    type Error = Error;

    fn try_from(value: pb::FieldAssignmentFile) -> Result<Self> {
        if value.path.is_empty() {
            return Err(Error::invalid_input(
                "Field assignment file path must not be empty",
            ));
        }
        Ok(Self {
            path: value.path,
            size_bytes: value.size_bytes,
            base_id: value.base_id,
        })
    }
}

/// Manifest descriptor for one tracked stable field ID.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct FieldAssignmentState {
    /// Stable Lance schema field ID.
    pub field_id: i32,
    /// Immutable root for this snapshot.
    pub root: FieldAssignmentFile,
}

impl From<&FieldAssignmentState> for pb::FieldAssignmentState {
    fn from(value: &FieldAssignmentState) -> Self {
        Self {
            field_id: value.field_id,
            root: Some((&value.root).into()),
        }
    }
}

impl TryFrom<pb::FieldAssignmentState> for FieldAssignmentState {
    type Error = Error;

    fn try_from(value: pb::FieldAssignmentState) -> Result<Self> {
        let root = value.root.ok_or_else(|| {
            Error::invalid_input(format!(
                "Field assignment state for field ID {} is missing its root",
                value.field_id
            ))
        })?;
        Ok(Self {
            field_id: value.field_id,
            root: root.try_into()?,
        })
    }
}

/// Materialized immutable root for one tracked field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldAssignmentRoot {
    /// Non-empty fragment states, sorted by fragment ID.
    pub fragments: Vec<FieldAssignmentFragment>,
}

/// Materialized assignment state for one physical fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldAssignmentFragment {
    /// Fragment ID in the snapshot.
    pub fragment_id: u64,
    /// Number of physical rows when this state was written.
    pub physical_rows: u64,
    /// Compact assignment state.
    pub state: FieldAssignmentFragmentState,
}

/// Compact assignment representation for a fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldAssignmentFragmentState {
    /// Every physical row is assigned.
    All,
    /// A non-empty, non-full Roaring bitmap of physical row offsets.
    Partial(FieldAssignmentFile),
}

impl From<&FieldAssignmentRoot> for pb::FieldAssignmentRoot {
    fn from(value: &FieldAssignmentRoot) -> Self {
        Self {
            fragments: value.fragments.iter().map(Into::into).collect(),
        }
    }
}

impl TryFrom<pb::FieldAssignmentRoot> for FieldAssignmentRoot {
    type Error = Error;

    fn try_from(value: pb::FieldAssignmentRoot) -> Result<Self> {
        let mut fragments = Vec::with_capacity(value.fragments.len());
        let mut previous = None;
        for fragment in value.fragments {
            let fragment: FieldAssignmentFragment = fragment.try_into()?;
            if previous.is_some_and(|id| id >= fragment.fragment_id) {
                return Err(Error::invalid_input(
                    "Field assignment root fragment IDs must be strictly increasing",
                ));
            }
            previous = Some(fragment.fragment_id);
            fragments.push(fragment);
        }
        Ok(Self { fragments })
    }
}

impl From<&FieldAssignmentFragment> for pb::FieldAssignmentFragment {
    fn from(value: &FieldAssignmentFragment) -> Self {
        let state = match &value.state {
            FieldAssignmentFragmentState::All => {
                pb::field_assignment_fragment::State::AllAssigned(true)
            }
            FieldAssignmentFragmentState::Partial(file) => {
                pb::field_assignment_fragment::State::Partial(file.into())
            }
        };
        Self {
            fragment_id: value.fragment_id,
            physical_rows: value.physical_rows,
            state: Some(state),
        }
    }
}

impl TryFrom<pb::FieldAssignmentFragment> for FieldAssignmentFragment {
    type Error = Error;

    fn try_from(value: pb::FieldAssignmentFragment) -> Result<Self> {
        if value.physical_rows == 0 {
            return Err(Error::invalid_input(format!(
                "Field assignment fragment {} must have at least one physical row",
                value.fragment_id
            )));
        }
        let state = match value.state.ok_or_else(|| {
            Error::invalid_input(format!(
                "Field assignment fragment {} is missing its state",
                value.fragment_id
            ))
        })? {
            pb::field_assignment_fragment::State::AllAssigned(true) => {
                FieldAssignmentFragmentState::All
            }
            pb::field_assignment_fragment::State::AllAssigned(false) => {
                return Err(Error::invalid_input(format!(
                    "Field assignment fragment {} encodes all_assigned=false",
                    value.fragment_id
                )));
            }
            pb::field_assignment_fragment::State::Partial(file) => {
                let file: FieldAssignmentFile = file.try_into()?;
                if file.size_bytes == 0 {
                    return Err(Error::invalid_input(format!(
                        "Partial field assignment file '{}' must have a non-zero size",
                        file.path
                    )));
                }
                FieldAssignmentFragmentState::Partial(file)
            }
        };
        Ok(Self {
            fragment_id: value.fragment_id,
            physical_rows: value.physical_rows,
            state,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_round_trip_and_validation() {
        let root = FieldAssignmentRoot {
            fragments: vec![
                FieldAssignmentFragment {
                    fragment_id: 1,
                    physical_rows: 5,
                    state: FieldAssignmentFragmentState::All,
                },
                FieldAssignmentFragment {
                    fragment_id: 3,
                    physical_rows: 8,
                    state: FieldAssignmentFragmentState::Partial(FieldAssignmentFile {
                        path: "_field_assignments/bitmaps/part.rbm".to_string(),
                        size_bytes: 12,
                        base_id: Some(4),
                    }),
                },
            ],
        };
        let proto = pb::FieldAssignmentRoot::from(&root);
        assert_eq!(FieldAssignmentRoot::try_from(proto).unwrap(), root);

        let duplicate = pb::FieldAssignmentRoot {
            fragments: vec![
                pb::FieldAssignmentFragment::from(&root.fragments[0]),
                pb::FieldAssignmentFragment::from(&root.fragments[0]),
            ],
        };
        assert!(FieldAssignmentRoot::try_from(duplicate).is_err());
    }
}
