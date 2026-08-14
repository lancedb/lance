// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Snapshot-level field-scoped Boolean flag metadata.

use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use object_store::path::Path;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::pb;

const MAX_INLINE_CELL_FLAG_ROOT_BYTES: usize = 64 * 1024;

/// Stable schema-level identity for a field-scoped Boolean flag.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct CellFlagDefinition {
    /// Dataset-unique ID that is never reused.
    pub flag_id: u32,
    /// Stable Lance schema field ID this flag is scoped to.
    pub field_id: i32,
    /// User-visible name, unique among flags registered for this field.
    pub name: String,
}

impl From<&CellFlagDefinition> for pb::CellFlagDefinition {
    fn from(value: &CellFlagDefinition) -> Self {
        Self {
            flag_id: value.flag_id,
            field_id: value.field_id,
            name: value.name.clone(),
        }
    }
}

impl TryFrom<pb::CellFlagDefinition> for CellFlagDefinition {
    type Error = Error;

    fn try_from(value: pb::CellFlagDefinition) -> Result<Self> {
        if value.name.is_empty() {
            return Err(Error::invalid_input(format!(
                "Cell flag {} for field ID {} has an empty name",
                value.flag_id, value.field_id
            )));
        }
        Ok(Self {
            flag_id: value.flag_id,
            field_id: value.field_id,
            name: value.name,
        })
    }
}

/// Reference to an immutable cell-flag object under a dataset root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct CellFlagFile {
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

impl From<&CellFlagFile> for pb::CellFlagFile {
    fn from(value: &CellFlagFile) -> Self {
        Self {
            path: value.path.clone(),
            size_bytes: value.size_bytes,
            base_id: value.base_id,
            inline_bytes: value.inline_bytes.clone(),
        }
    }
}

impl TryFrom<pb::CellFlagFile> for CellFlagFile {
    type Error = Error;

    fn try_from(value: pb::CellFlagFile) -> Result<Self> {
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

impl CellFlagFile {
    fn validate_inline_copy(&self) -> Result<()> {
        if let Some(bytes) = self.inline_bytes.as_ref()
            && bytes.len() as u64 != self.size_bytes
        {
            return Err(Error::invalid_input(format!(
                "Inline cell flag file '{}' has size {}, expected {}",
                self.path,
                bytes.len(),
                self.size_bytes
            )));
        }
        if self
            .inline_bytes
            .as_ref()
            .is_some_and(|bytes| bytes.len() > MAX_INLINE_CELL_FLAG_ROOT_BYTES)
        {
            return Err(Error::invalid_input(format!(
                "Inline cell flag file '{}' has size {}, maximum is {}",
                self.path, self.size_bytes, MAX_INLINE_CELL_FLAG_ROOT_BYTES
            )));
        }
        Ok(())
    }

    fn validate_namespace(&self) -> Result<()> {
        let path = Path::parse(&self.path).map_err(|error| {
            Error::invalid_input(format!(
                "Invalid cell flag file path '{}': {}",
                self.path, error
            ))
        })?;
        let mut parts = path.parts();
        if !parts
            .next()
            .is_some_and(|part| part.as_ref() == "_cell_flags")
        {
            return Err(Error::invalid_input(format!(
                "Cell flag file '{}' must be under '_cell_flags'",
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
                "Cell flag {} file '{}' has an invalid path layout",
                kind, self.path
            )));
        }
        parts[2].parse::<u32>().map_err(|_| {
            Error::invalid_input(format!(
                "Cell flag {} file '{}' has an invalid flag ID segment",
                kind, self.path
            ))
        })?;
        if kind == "bitmaps" {
            parts[3].parse::<u64>().map_err(|_| {
                Error::invalid_input(format!(
                    "Cell flag bitmap file '{}' has an invalid fragment ID segment",
                    self.path
                ))
            })?;
        }
        let file_name = parts.last().ok_or_else(|| {
            Error::invalid_input(format!(
                "Cell flag {} file '{}' has an empty path",
                kind, self.path
            ))
        })?;
        let uuid = file_name.strip_suffix(suffix).ok_or_else(|| {
            Error::invalid_input(format!(
                "Cell flag {} file '{}' must end in '{}'",
                kind, self.path, suffix
            ))
        })?;
        Uuid::parse_str(uuid).map_err(|_| {
            Error::invalid_input(format!(
                "Cell flag {} file '{}' has an invalid immutable object ID",
                kind, self.path
            ))
        })?;
        Ok(())
    }

    /// Validate that this file is an immutable flag root.
    pub fn validate_root_path(&self) -> Result<()> {
        self.validate_inline_copy()?;
        self.validate_kind("roots", 4, ".root")
    }

    /// Validate this root's namespace against its manifest flag ID.
    pub fn validate_root_path_for_flag(&self, flag_id: u32) -> Result<()> {
        self.validate_root_path()?;
        let path_flag_id = Path::parse(&self.path)?
            .parts()
            .nth(2)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Cell flag root '{}' is missing its flag ID segment",
                    self.path
                ))
            })?
            .as_ref()
            .parse::<u32>()
            .map_err(|_| {
                Error::invalid_input(format!(
                    "Cell flag root '{}' has an invalid flag ID segment",
                    self.path
                ))
            })?;
        if path_flag_id != flag_id {
            return Err(Error::invalid_input(format!(
                "Cell flag root '{}' is under flag ID {}, expected {}",
                self.path, path_flag_id, flag_id
            )));
        }
        Ok(())
    }

    /// Validate that this file is an immutable partial flag bitmap.
    pub fn validate_bitmap_path(&self) -> Result<()> {
        self.validate_inline_copy()?;
        self.validate_kind("bitmaps", 5, ".rbm")?;
        if self.inline_bytes.is_some() {
            return Err(Error::invalid_input(format!(
                "Cell flag bitmap file '{}' cannot contain inline root bytes",
                self.path
            )));
        }
        Ok(())
    }

    /// Validate this bitmap's namespace against its root entry.
    pub fn validate_bitmap_path_for_fragment(&self, flag_id: u32, fragment_id: u64) -> Result<()> {
        self.validate_bitmap_path()?;
        let path = Path::parse(&self.path)?;
        let parts = path.parts().collect::<Vec<_>>();
        let path_flag_id = parts
            .get(2)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Cell flag bitmap '{}' is missing its flag ID segment",
                    self.path
                ))
            })?
            .as_ref()
            .parse::<u32>()
            .map_err(|_| {
                Error::invalid_input(format!(
                    "Cell flag bitmap '{}' has an invalid flag ID segment",
                    self.path
                ))
            })?;
        let path_fragment_id = parts
            .get(3)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Cell flag bitmap '{}' is missing its fragment ID segment",
                    self.path
                ))
            })?
            .as_ref()
            .parse::<u64>()
            .map_err(|_| {
                Error::invalid_input(format!(
                    "Cell flag bitmap '{}' has an invalid fragment ID segment",
                    self.path
                ))
            })?;
        if path_flag_id != flag_id || path_fragment_id != fragment_id {
            return Err(Error::invalid_input(format!(
                "Cell flag bitmap '{}' is under flag ID {} and fragment {}, expected flag ID {} and fragment {}",
                self.path, path_flag_id, path_fragment_id, flag_id, fragment_id
            )));
        }
        Ok(())
    }
}

/// Manifest descriptor for one registered flag with at least one true row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct CellFlagState {
    /// Stable dataset flag ID.
    pub flag_id: u32,
    /// Immutable root for this snapshot.
    pub root: CellFlagFile,
}

impl From<&CellFlagState> for pb::CellFlagState {
    fn from(value: &CellFlagState) -> Self {
        Self {
            flag_id: value.flag_id,
            root: Some((&value.root).into()),
        }
    }
}

impl TryFrom<pb::CellFlagState> for CellFlagState {
    type Error = Error;

    fn try_from(value: pb::CellFlagState) -> Result<Self> {
        let root = value.root.ok_or_else(|| {
            Error::invalid_input(format!(
                "Cell flag state for flag ID {} is missing its root",
                value.flag_id
            ))
        })?;
        let root: CellFlagFile = root.try_into()?;
        root.validate_root_path_for_flag(value.flag_id)?;
        Ok(Self {
            flag_id: value.flag_id,
            root,
        })
    }
}

/// Materialized immutable root for one registered flag.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct CellFlagRoot {
    /// Non-empty fragment states, sorted by fragment ID.
    pub fragments: Vec<CellFlagFragment>,
}

/// Materialized flag state for one physical fragment.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct CellFlagFragment {
    /// Fragment ID in the snapshot.
    pub fragment_id: u64,
    /// Number of physical rows when this state was written.
    pub physical_rows: u64,
    /// Compact flag state.
    pub state: CellFlagFragmentState,
}

/// Compact flag representation for a fragment.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub enum CellFlagFragmentState {
    /// Every physical row is true.
    All,
    /// A non-empty, non-full Roaring bitmap of physical row offsets.
    Partial(CellFlagFile),
    /// A small non-empty, non-full portable Roaring bitmap embedded in the root.
    InlinePartial(Vec<u8>),
}

impl From<&CellFlagRoot> for pb::CellFlagRoot {
    fn from(value: &CellFlagRoot) -> Self {
        Self {
            fragments: value.fragments.iter().map(Into::into).collect(),
        }
    }
}

impl TryFrom<pb::CellFlagRoot> for CellFlagRoot {
    type Error = Error;

    fn try_from(value: pb::CellFlagRoot) -> Result<Self> {
        let mut fragments = Vec::with_capacity(value.fragments.len());
        let mut previous = None;
        for fragment in value.fragments {
            let fragment: CellFlagFragment = fragment.try_into()?;
            if previous.is_some_and(|id| id >= fragment.fragment_id) {
                return Err(Error::invalid_input(
                    "Cell flag root fragment IDs must be strictly increasing",
                ));
            }
            previous = Some(fragment.fragment_id);
            fragments.push(fragment);
        }
        Ok(Self { fragments })
    }
}

impl From<&CellFlagFragment> for pb::CellFlagFragment {
    fn from(value: &CellFlagFragment) -> Self {
        let state = match &value.state {
            CellFlagFragmentState::All => pb::cell_flag_fragment::State::AllSet(true),
            CellFlagFragmentState::Partial(file) => {
                pb::cell_flag_fragment::State::Partial(file.into())
            }
            CellFlagFragmentState::InlinePartial(bytes) => {
                pb::cell_flag_fragment::State::InlinePartial(bytes.clone())
            }
        };
        Self {
            fragment_id: value.fragment_id,
            physical_rows: value.physical_rows,
            state: Some(state),
        }
    }
}

impl TryFrom<pb::CellFlagFragment> for CellFlagFragment {
    type Error = Error;

    fn try_from(value: pb::CellFlagFragment) -> Result<Self> {
        if value.physical_rows == 0 {
            return Err(Error::invalid_input(format!(
                "Cell flag fragment {} must have at least one physical row",
                value.fragment_id
            )));
        }
        let state = match value.state.ok_or_else(|| {
            Error::invalid_input(format!(
                "Cell flag fragment {} is missing its state",
                value.fragment_id
            ))
        })? {
            pb::cell_flag_fragment::State::AllSet(true) => CellFlagFragmentState::All,
            pb::cell_flag_fragment::State::AllSet(false) => {
                return Err(Error::invalid_input(format!(
                    "Cell flag fragment {} encodes all_set=false",
                    value.fragment_id
                )));
            }
            pb::cell_flag_fragment::State::Partial(file) => {
                let file: CellFlagFile = file.try_into()?;
                file.validate_bitmap_path()?;
                if file.size_bytes == 0 {
                    return Err(Error::invalid_input(format!(
                        "Partial cell flag file '{}' must have a non-zero size",
                        file.path
                    )));
                }
                CellFlagFragmentState::Partial(file)
            }
            pb::cell_flag_fragment::State::InlinePartial(bytes) => {
                if bytes.is_empty() {
                    return Err(Error::invalid_input(format!(
                        "Inline partial cell flag for fragment {} must be non-empty",
                        value.fragment_id
                    )));
                }
                CellFlagFragmentState::InlinePartial(bytes)
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
        let root = CellFlagRoot {
            fragments: vec![
                CellFlagFragment {
                    fragment_id: 1,
                    physical_rows: 5,
                    state: CellFlagFragmentState::All,
                },
                CellFlagFragment {
                    fragment_id: 3,
                    physical_rows: 8,
                    state: CellFlagFragmentState::Partial(CellFlagFile {
                        path: "_cell_flags/bitmaps/7/3/00000000-0000-0000-0000-000000000001.rbm"
                            .to_string(),
                        size_bytes: 12,
                        base_id: Some(4),
                        inline_bytes: None,
                    }),
                },
                CellFlagFragment {
                    fragment_id: 5,
                    physical_rows: 10,
                    state: CellFlagFragmentState::InlinePartial(vec![1, 2, 3]),
                },
            ],
        };
        let proto = pb::CellFlagRoot::from(&root);
        assert_eq!(CellFlagRoot::try_from(proto).unwrap(), root);

        let duplicate = pb::CellFlagRoot {
            fragments: vec![
                pb::CellFlagFragment::from(&root.fragments[0]),
                pb::CellFlagFragment::from(&root.fragments[0]),
            ],
        };
        assert!(CellFlagRoot::try_from(duplicate).is_err());

        let empty_inline = pb::CellFlagRoot {
            fragments: vec![pb::CellFlagFragment {
                fragment_id: 1,
                physical_rows: 5,
                state: Some(pb::cell_flag_fragment::State::InlinePartial(Vec::new())),
            }],
        };
        assert!(CellFlagRoot::try_from(empty_inline).is_err());
    }

    #[test]
    fn cell_flag_files_reject_wrong_roles_and_layouts() {
        let root = CellFlagFile {
            path: "_cell_flags/roots/7/00000000-0000-0000-0000-000000000001.root".to_string(),
            size_bytes: 12,
            base_id: None,
            inline_bytes: Some(vec![0; 12]),
        };
        assert!(root.validate_root_path().is_ok());
        assert!(root.validate_bitmap_path().is_err());

        let bitmap = CellFlagFile {
            path: "_cell_flags/bitmaps/7/3/00000000-0000-0000-0000-000000000001.rbm".to_string(),
            size_bytes: 12,
            base_id: None,
            inline_bytes: None,
        };
        assert!(bitmap.validate_bitmap_path().is_ok());
        assert!(bitmap.validate_root_path().is_err());

        for invalid_path in [
            "outside/roots/7/00000000-0000-0000-0000-000000000001.root",
            "_cell_flags/roots/not-a-field/00000000-0000-0000-0000-000000000001.root",
            "_cell_flags/roots/7/not-a-uuid.root",
            "_cell_flags/bitmaps/7/not-a-fragment/00000000-0000-0000-0000-000000000001.rbm",
            "_cell_flags/bitmaps/7/3/00000000-0000-0000-0000-000000000001.root",
            "_cell_flags/bitmaps/7/3/00000000-0000-0000-0000-000000000001.rbm/extra",
        ] {
            let file = CellFlagFile {
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
