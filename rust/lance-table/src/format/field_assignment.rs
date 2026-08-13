// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Snapshot-level field assignment metadata.

use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use object_store::path::Path;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::pb;

const MAX_INLINE_FIELD_ASSIGNMENT_ROOT_BYTES: usize = 64 * 1024;

/// Reference to an immutable field-assignment object under a dataset root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct FieldAssignmentFile {
    /// Path relative to the referenced dataset root.
    pub path: String,
    /// Exact encoded size in bytes.
    pub size_bytes: u64,
    /// Optional external dataset base ID.
    pub base_id: Option<u32>,
    /// Optional exact inline copy of a small immutable root object.
    #[serde(default)]
    pub inline_bytes: Option<Vec<u8>>,
}

impl From<&FieldAssignmentFile> for pb::FieldAssignmentFile {
    fn from(value: &FieldAssignmentFile) -> Self {
        Self {
            path: value.path.clone(),
            size_bytes: value.size_bytes,
            base_id: value.base_id,
            inline_bytes: value.inline_bytes.clone(),
        }
    }
}

impl TryFrom<pb::FieldAssignmentFile> for FieldAssignmentFile {
    type Error = Error;

    fn try_from(value: pb::FieldAssignmentFile) -> Result<Self> {
        let file = Self {
            path: value.path,
            size_bytes: value.size_bytes,
            base_id: value.base_id,
            inline_bytes: value.inline_bytes,
        };
        file.validate_inline_copy()?;
        file.validate_namespace()?;
        Ok(file)
    }
}

impl FieldAssignmentFile {
    fn validate_inline_copy(&self) -> Result<()> {
        if let Some(bytes) = self.inline_bytes.as_ref()
            && bytes.len() as u64 != self.size_bytes
        {
            return Err(Error::invalid_input(format!(
                "Inline field assignment file '{}' has size {}, expected {}",
                self.path,
                bytes.len(),
                self.size_bytes
            )));
        }
        if self
            .inline_bytes
            .as_ref()
            .is_some_and(|bytes| bytes.len() > MAX_INLINE_FIELD_ASSIGNMENT_ROOT_BYTES)
        {
            return Err(Error::invalid_input(format!(
                "Inline field assignment file '{}' has size {}, maximum is {}",
                self.path, self.size_bytes, MAX_INLINE_FIELD_ASSIGNMENT_ROOT_BYTES
            )));
        }
        Ok(())
    }

    fn validate_namespace(&self) -> Result<()> {
        let path = Path::parse(&self.path).map_err(|error| {
            Error::invalid_input(format!(
                "Invalid field assignment file path '{}': {}",
                self.path, error
            ))
        })?;
        let mut parts = path.parts();
        if !parts
            .next()
            .is_some_and(|part| part.as_ref() == "_field_assignments")
        {
            return Err(Error::invalid_input(format!(
                "Field assignment file '{}' must be under '_field_assignments'",
                self.path
            )));
        }
        Ok(())
    }

    fn validate_kind(&self, kind: &str, expected_parts: usize, suffix: &str) -> Result<()> {
        self.validate_namespace()?;
        let path = Path::parse(&self.path)?;
        let parts = path
            .parts()
            .map(|part| part.as_ref().to_string())
            .collect::<Vec<_>>();
        if parts.len() != expected_parts || parts.get(1).map(String::as_str) != Some(kind) {
            return Err(Error::invalid_input(format!(
                "Field assignment {} file '{}' has an invalid path layout",
                kind, self.path
            )));
        }
        parts[2].parse::<i32>().map_err(|_| {
            Error::invalid_input(format!(
                "Field assignment {} file '{}' has an invalid field ID segment",
                kind, self.path
            ))
        })?;
        if kind == "bitmaps" {
            parts[3].parse::<u64>().map_err(|_| {
                Error::invalid_input(format!(
                    "Field assignment bitmap file '{}' has an invalid fragment ID segment",
                    self.path
                ))
            })?;
        }
        let file_name = parts.last().expect("validated non-empty path");
        let uuid = file_name.strip_suffix(suffix).ok_or_else(|| {
            Error::invalid_input(format!(
                "Field assignment {} file '{}' must end in '{}'",
                kind, self.path, suffix
            ))
        })?;
        Uuid::parse_str(uuid).map_err(|_| {
            Error::invalid_input(format!(
                "Field assignment {} file '{}' has an invalid immutable object ID",
                kind, self.path
            ))
        })?;
        Ok(())
    }

    /// Validate that this file is an immutable assignment root.
    pub fn validate_root_path(&self) -> Result<()> {
        self.validate_inline_copy()?;
        self.validate_kind("roots", 4, ".root")
    }

    /// Validate this root's namespace against its manifest field ID.
    pub fn validate_root_path_for_field(&self, field_id: i32) -> Result<()> {
        self.validate_root_path()?;
        let path_field_id = Path::parse(&self.path)?
            .parts()
            .nth(2)
            .expect("validated root field ID segment")
            .as_ref()
            .parse::<i32>()
            .expect("validated root field ID");
        if path_field_id != field_id {
            return Err(Error::invalid_input(format!(
                "Field assignment root '{}' is under field ID {}, expected {}",
                self.path, path_field_id, field_id
            )));
        }
        Ok(())
    }

    /// Validate that this file is an immutable partial-assignment bitmap.
    pub fn validate_bitmap_path(&self) -> Result<()> {
        self.validate_inline_copy()?;
        self.validate_kind("bitmaps", 5, ".rbm")?;
        if self.inline_bytes.is_some() {
            return Err(Error::invalid_input(format!(
                "Field assignment bitmap file '{}' cannot contain inline root bytes",
                self.path
            )));
        }
        Ok(())
    }

    /// Validate this bitmap's namespace against its root entry.
    pub fn validate_bitmap_path_for_fragment(&self, field_id: i32, fragment_id: u64) -> Result<()> {
        self.validate_bitmap_path()?;
        let path = Path::parse(&self.path)?;
        let parts = path.parts().collect::<Vec<_>>();
        let path_field_id = parts[2]
            .as_ref()
            .parse::<i32>()
            .expect("validated bitmap field ID");
        let path_fragment_id = parts[3]
            .as_ref()
            .parse::<u64>()
            .expect("validated bitmap fragment ID");
        if path_field_id != field_id || path_fragment_id != fragment_id {
            return Err(Error::invalid_input(format!(
                "Field assignment bitmap '{}' is under field ID {} and fragment {}, expected field ID {} and fragment {}",
                self.path, path_field_id, path_fragment_id, field_id, fragment_id
            )));
        }
        Ok(())
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
        let root: FieldAssignmentFile = root.try_into()?;
        root.validate_root_path_for_field(value.field_id)?;
        Ok(Self {
            field_id: value.field_id,
            root,
        })
    }
}

/// Materialized immutable root for one tracked field.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct FieldAssignmentRoot {
    /// Non-empty fragment states, sorted by fragment ID.
    pub fragments: Vec<FieldAssignmentFragment>,
}

/// Materialized assignment state for one physical fragment.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct FieldAssignmentFragment {
    /// Fragment ID in the snapshot.
    pub fragment_id: u64,
    /// Number of physical rows when this state was written.
    pub physical_rows: u64,
    /// Compact assignment state.
    pub state: FieldAssignmentFragmentState,
}

/// Compact assignment representation for a fragment.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub enum FieldAssignmentFragmentState {
    /// Every physical row is assigned.
    All,
    /// A non-empty, non-full Roaring bitmap of physical row offsets.
    Partial(FieldAssignmentFile),
    /// A small non-empty, non-full portable Roaring bitmap embedded in the root.
    InlinePartial(Vec<u8>),
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
            FieldAssignmentFragmentState::InlinePartial(bytes) => {
                pb::field_assignment_fragment::State::InlinePartial(bytes.clone())
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
                file.validate_bitmap_path()?;
                if file.size_bytes == 0 {
                    return Err(Error::invalid_input(format!(
                        "Partial field assignment file '{}' must have a non-zero size",
                        file.path
                    )));
                }
                FieldAssignmentFragmentState::Partial(file)
            }
            pb::field_assignment_fragment::State::InlinePartial(bytes) => {
                if bytes.is_empty() {
                    return Err(Error::invalid_input(format!(
                        "Inline partial field assignment for fragment {} must be non-empty",
                        value.fragment_id
                    )));
                }
                FieldAssignmentFragmentState::InlinePartial(bytes)
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
                        path: "_field_assignments/bitmaps/7/3/00000000-0000-0000-0000-000000000001.rbm"
                            .to_string(),
                        size_bytes: 12,
                        base_id: Some(4),
                        inline_bytes: None,
                    }),
                },
                FieldAssignmentFragment {
                    fragment_id: 5,
                    physical_rows: 10,
                    state: FieldAssignmentFragmentState::InlinePartial(vec![1, 2, 3]),
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

        let empty_inline = pb::FieldAssignmentRoot {
            fragments: vec![pb::FieldAssignmentFragment {
                fragment_id: 1,
                physical_rows: 5,
                state: Some(pb::field_assignment_fragment::State::InlinePartial(
                    Vec::new(),
                )),
            }],
        };
        assert!(FieldAssignmentRoot::try_from(empty_inline).is_err());
    }

    #[test]
    fn field_assignment_files_reject_wrong_roles_and_layouts() {
        let root = FieldAssignmentFile {
            path: "_field_assignments/roots/7/00000000-0000-0000-0000-000000000001.root"
                .to_string(),
            size_bytes: 12,
            base_id: None,
            inline_bytes: Some(vec![0; 12]),
        };
        assert!(root.validate_root_path().is_ok());
        assert!(root.validate_bitmap_path().is_err());

        let bitmap = FieldAssignmentFile {
            path: "_field_assignments/bitmaps/7/3/00000000-0000-0000-0000-000000000001.rbm"
                .to_string(),
            size_bytes: 12,
            base_id: None,
            inline_bytes: None,
        };
        assert!(bitmap.validate_bitmap_path().is_ok());
        assert!(bitmap.validate_root_path().is_err());

        for invalid_path in [
            "outside/roots/7/00000000-0000-0000-0000-000000000001.root",
            "_field_assignments/roots/not-a-field/00000000-0000-0000-0000-000000000001.root",
            "_field_assignments/roots/7/not-a-uuid.root",
            "_field_assignments/bitmaps/7/not-a-fragment/00000000-0000-0000-0000-000000000001.rbm",
            "_field_assignments/bitmaps/7/3/00000000-0000-0000-0000-000000000001.root",
            "_field_assignments/bitmaps/7/3/00000000-0000-0000-0000-000000000001.rbm/extra",
        ] {
            let file = FieldAssignmentFile {
                path: invalid_path.to_string(),
                size_bytes: 12,
                base_id: None,
                inline_bytes: None,
            };
            assert!(
                file.validate_root_path().is_err() && file.validate_bitmap_path().is_err(),
                "invalid path was accepted: {invalid_path}"
            );
        }
    }
}
