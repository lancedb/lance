// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! EXPERIMENTAL: action-based (`UserOperation`) transactions.
//!
//! This module is a skeleton for the "Action-based Transactions" milestone
//! (design discussion #5960). It models a transaction as an ordered list of
//! granular [`Action`]s that commit atomically, which is what enables compound
//! commits such as *append data + create index* in a single manifest change.
//!
//! # Stability
//!
//! Everything here is **unstable and gated** — it compiles only under the
//! non-default `unstable-action-transactions` Cargo feature, so it is absent
//! from released artifacts. It is the first consumer of the general
//! experimental-feature mechanism: a dataset that persists one of these
//! transactions declares the [`FEATURE_NAME`] experimental feature, which sets
//! the `FLAG_EXPERIMENTAL` bit so that libraries which do not understand the
//! format refuse to commit. The wire format
//! (`protos/transaction_experimental.proto`) carries no compatibility guarantee
//! and may change or be removed in any release until it is finalized and
//! ratified by PMC vote, at which point it graduates to its own feature-flag
//! bit. See `rust/lance-table/design/experimental_feature_flags.md`.
//!
//! # Scope
//!
//! [`AddFragments`] is a proof of shape. [`AddBases`], [`ReplaceFragmentColumns`],
//! [`AddFields`], [`ChangeSchema`], and [`RefreshRowVersionMetadata`] are the
//! first real vertical slice, covering the decomposition of the legacy
//! `UpdateBases`, `DataReplacement`, and `Merge` operations. The remaining
//! actions land in follow-up vertical slices.

use lance_core::datatypes::{Field, Schema};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use lance_file::datatypes::{Fields, FieldsWithMeta};

use crate::format::{BasePath, DataFile, Fragment, pb};

/// The experimental-feature name a dataset declares when its transaction log
/// uses action-based transactions. Registered in
/// [`crate::feature_flags::known_experimental_features`] under this crate's
/// `unstable-action-transactions` feature.
pub const FEATURE_NAME: &str = "action-transactions";

/// A user-facing, composable transaction: an ordered list of [`UserAction`]s
/// that commit atomically as a single manifest change.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct UserOperation {
    /// Human-readable description, e.g. `"INSERT INTO t VALUES (1)"`.
    pub description: String,
    /// Unique identifier for this operation.
    pub uuid: String,
    /// The dataset version this operation was planned against.
    pub read_version: u64,
    /// The ordered list of user actions applied by this operation.
    pub actions: Vec<UserAction>,
}

/// A single user-recognizable step within a [`UserOperation`] (e.g. "append
/// batch", "rebuild index"), carrying a description and the granular actions it
/// expands to. Action lists are flattened when applied to the manifest.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct UserAction {
    /// Human-readable description of this step.
    pub description: String,
    /// The granular manifest changes this step expands to.
    pub actions: Vec<Action>,
}

/// A granular change to the manifest. Traditional operations decompose into an
/// ordered list of these.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
#[non_exhaustive]
pub enum Action {
    /// Append new fragments to the table.
    AddFragments(AddFragments),
    /// Add new base paths to the manifest.
    AddBases(AddBases),
    /// Replace some of a fragment's data files with new ones.
    ReplaceFragmentColumns(ReplaceFragmentColumns),
    /// Add new fields (and any data files that back them) to the schema.
    AddFields(AddFields),
    /// Replace the manifest schema wholesale.
    ChangeSchema(ChangeSchema),
    /// Refresh row-version metadata for fragments whose columns were merged in place.
    RefreshRowVersionMetadata(RefreshRowVersionMetadata),
}

/// Append new fragments to the table.
///
/// Covers Append, the "add new rows" part of Overwrite, and new fragments
/// produced by compaction.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddFragments {
    /// The new fragments to append. Fragment IDs are assigned at apply time
    /// from the manifest's counters (late binding), not carried here.
    pub fragments: Vec<Fragment>,
}

impl From<&AddFragments> for pb::AddFragments {
    fn from(action: &AddFragments) -> Self {
        Self {
            fragments: action
                .fragments
                .iter()
                .map(pb::DataFragment::from)
                .collect(),
        }
    }
}

impl TryFrom<pb::AddFragments> for AddFragments {
    type Error = Error;

    fn try_from(proto: pb::AddFragments) -> Result<Self> {
        Ok(Self {
            fragments: proto
                .fragments
                .into_iter()
                .map(Fragment::try_from)
                .collect::<Result<_>>()?,
        })
    }
}

/// A symbolic reference to a base path that may not be committed yet: either an
/// already-committed base id, or a same-operation [`AddBases`] placeholder
/// resolved to its assigned id at apply time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeepSizeOf)]
pub enum BaseRef {
    /// An already-committed base path id.
    Committed(u32),
    /// The n-th unassigned (`id == 0`) base path minted by an [`AddBases`]
    /// action in the same operation (0-indexed, in the order such entries are
    /// encountered scanning the operation's actions top-to-bottom).
    Local(u32),
}

/// A data file paired with a symbolic reference to the base it lives under.
///
/// `file.base_id` is ignored and must be `None`: the real value is stamped in
/// at apply time once `base` is resolved. `base` of `None` means the
/// dataset's default base.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct NewDataFile {
    pub base: Option<BaseRef>,
    pub file: DataFile,
}

/// Add new base paths to the manifest.
///
/// Mirrors the legacy `UpdateBases` operation: a base with `id == 0` is
/// unassigned and is allocated from the manifest's base-id counter at apply
/// time.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddBases {
    pub bases: Vec<BasePath>,
}

/// How [`ReplaceFragmentColumns`] installs `new_data_files` onto the target
/// fragment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeepSizeOf)]
pub enum ColumnReplacement {
    /// Swap data files that match an existing file's fields and format
    /// version; a file whose fields are entirely uncovered by any existing
    /// file (e.g. an all-NULL column being backfilled) is appended instead.
    InPlace,
    /// `new_data_files` is the fragment's complete post-rewrite file list,
    /// installed verbatim.
    Tombstone,
}

/// Replace some of a fragment's data files with new ones.
///
/// Covers the per-fragment replacement in the legacy `DataReplacement`
/// operation.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct ReplaceFragmentColumns {
    pub fragment_id: u64,
    /// The field ids covered by `new_data_files`. Used for conflict detection
    /// and to invalidate covering index entries for `fragment_id`.
    pub field_ids: Vec<i32>,
    pub new_data_files: Vec<NewDataFile>,
    pub replacement: ColumnReplacement,
    /// Reserved for a future partial-row-level replacement; always `None` in
    /// the current translation from the legacy `DataReplacement` operation,
    /// which replaces whole files.
    pub updated_row_offsets: Option<Vec<u8>>,
}

/// New fields, and any new data files that back them, added to the schema.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddFields {
    pub new_fields: Vec<Field>,
    pub fragment_files: Vec<(u64, Vec<NewDataFile>)>,
}

/// Replace the manifest schema wholesale.
///
/// Used alongside [`AddFields`] for the "recast" (column type/encoding
/// change) shape of the legacy `Merge` operation: `AddFields` lands the new
/// data files on the affected fragments, and this action fixes up field
/// identity/types.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct ChangeSchema {
    pub schema: Schema,
}

/// Refresh row-version metadata (last-updated / created-at) for fragments
/// whose columns were merged in place, mirroring the implicit refresh the
/// legacy `Merge` operation performs on stable-row-id datasets.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct RefreshRowVersionMetadata {
    pub fragment_ids: Vec<u64>,
}

impl From<&BaseRef> for pb::BaseRef {
    fn from(base_ref: &BaseRef) -> Self {
        let reference = match base_ref {
            BaseRef::Committed(id) => pb::base_ref::Reference::Committed(*id),
            BaseRef::Local(id) => pb::base_ref::Reference::Local(*id),
        };
        Self {
            reference: Some(reference),
        }
    }
}

impl TryFrom<pb::BaseRef> for BaseRef {
    type Error = Error;

    fn try_from(proto: pb::BaseRef) -> Result<Self> {
        match proto.reference {
            Some(pb::base_ref::Reference::Committed(id)) => Ok(Self::Committed(id)),
            Some(pb::base_ref::Reference::Local(id)) => Ok(Self::Local(id)),
            None => Err(Error::invalid_input(
                "BaseRef protobuf has no variant set".to_string(),
            )),
        }
    }
}

impl From<&NewDataFile> for pb::NewDataFile {
    fn from(new_data_file: &NewDataFile) -> Self {
        Self {
            base: new_data_file.base.as_ref().map(pb::BaseRef::from),
            file: Some(pb::DataFile::from(&new_data_file.file)),
        }
    }
}

impl TryFrom<pb::NewDataFile> for NewDataFile {
    type Error = Error;

    fn try_from(proto: pb::NewDataFile) -> Result<Self> {
        let file = proto
            .file
            .ok_or_else(|| Error::invalid_input("NewDataFile is missing its file".to_string()))?;
        Ok(Self {
            base: proto.base.map(BaseRef::try_from).transpose()?,
            file: DataFile::try_from(file)?,
        })
    }
}

impl From<&AddBases> for pb::AddBases {
    fn from(action: &AddBases) -> Self {
        Self {
            bases: action
                .bases
                .iter()
                .cloned()
                .map(pb::BasePath::from)
                .collect(),
        }
    }
}

impl TryFrom<pb::AddBases> for AddBases {
    type Error = Error;

    fn try_from(proto: pb::AddBases) -> Result<Self> {
        Ok(Self {
            bases: proto.bases.into_iter().map(BasePath::from).collect(),
        })
    }
}

impl From<ColumnReplacement> for pb::ColumnReplacement {
    fn from(replacement: ColumnReplacement) -> Self {
        match replacement {
            ColumnReplacement::InPlace => Self::InPlace,
            ColumnReplacement::Tombstone => Self::Tombstone,
        }
    }
}

impl TryFrom<i32> for ColumnReplacement {
    type Error = Error;

    fn try_from(value: i32) -> Result<Self> {
        match pb::ColumnReplacement::try_from(value) {
            Ok(pb::ColumnReplacement::InPlace) => Ok(Self::InPlace),
            Ok(pb::ColumnReplacement::Tombstone) => Ok(Self::Tombstone),
            Err(_) => Err(Error::invalid_input(format!(
                "unrecognized ColumnReplacement value {}",
                value
            ))),
        }
    }
}

impl From<&ReplaceFragmentColumns> for pb::ReplaceFragmentColumns {
    fn from(action: &ReplaceFragmentColumns) -> Self {
        let replacement: pb::ColumnReplacement = action.replacement.into();
        Self {
            fragment_id: action.fragment_id,
            field_ids: action.field_ids.clone(),
            new_data_files: action
                .new_data_files
                .iter()
                .map(pb::NewDataFile::from)
                .collect(),
            replacement: replacement.into(),
            updated_row_offsets: action.updated_row_offsets.clone(),
        }
    }
}

impl TryFrom<pb::ReplaceFragmentColumns> for ReplaceFragmentColumns {
    type Error = Error;

    fn try_from(proto: pb::ReplaceFragmentColumns) -> Result<Self> {
        Ok(Self {
            fragment_id: proto.fragment_id,
            field_ids: proto.field_ids,
            new_data_files: proto
                .new_data_files
                .into_iter()
                .map(NewDataFile::try_from)
                .collect::<Result<_>>()?,
            replacement: ColumnReplacement::try_from(proto.replacement)?,
            updated_row_offsets: proto.updated_row_offsets,
        })
    }
}

impl From<&AddFields> for pb::AddFields {
    fn from(action: &AddFields) -> Self {
        let new_fields = action
            .new_fields
            .iter()
            .flat_map(|field| Fields::from(field).0)
            .collect();
        let fragment_files = action
            .fragment_files
            .iter()
            .map(|(fragment_id, files)| pb::add_fields::FragmentFiles {
                fragment_id: *fragment_id,
                files: files.iter().map(pb::NewDataFile::from).collect(),
            })
            .collect();
        Self {
            new_fields,
            fragment_files,
        }
    }
}

impl TryFrom<pb::AddFields> for AddFields {
    type Error = Error;

    fn try_from(proto: pb::AddFields) -> Result<Self> {
        let new_fields = Schema::from(&Fields(proto.new_fields)).fields;
        let fragment_files = proto
            .fragment_files
            .into_iter()
            .map(|ff| {
                let files = ff
                    .files
                    .into_iter()
                    .map(NewDataFile::try_from)
                    .collect::<Result<_>>()?;
                Ok((ff.fragment_id, files))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            new_fields,
            fragment_files,
        })
    }
}

impl From<&ChangeSchema> for pb::ChangeSchema {
    fn from(action: &ChangeSchema) -> Self {
        let fields_with_meta = FieldsWithMeta::from(&action.schema);
        Self {
            fields: fields_with_meta.fields.0,
            schema_metadata: fields_with_meta.metadata,
        }
    }
}

impl TryFrom<pb::ChangeSchema> for ChangeSchema {
    type Error = Error;

    fn try_from(proto: pb::ChangeSchema) -> Result<Self> {
        let schema = Schema::from(FieldsWithMeta {
            fields: Fields(proto.fields),
            metadata: proto.schema_metadata,
        });
        Ok(Self { schema })
    }
}

impl From<&RefreshRowVersionMetadata> for pb::RefreshRowVersionMetadata {
    fn from(action: &RefreshRowVersionMetadata) -> Self {
        Self {
            fragment_ids: action.fragment_ids.clone(),
        }
    }
}

impl TryFrom<pb::RefreshRowVersionMetadata> for RefreshRowVersionMetadata {
    type Error = Error;

    fn try_from(proto: pb::RefreshRowVersionMetadata) -> Result<Self> {
        Ok(Self {
            fragment_ids: proto.fragment_ids,
        })
    }
}

impl From<&Action> for pb::Action {
    fn from(action: &Action) -> Self {
        let action = match action {
            Action::AddFragments(add) => pb::action::Action::AddFragments(add.into()),
            Action::AddBases(add) => pb::action::Action::AddBases(add.into()),
            Action::ReplaceFragmentColumns(replace) => {
                pb::action::Action::ReplaceFragmentColumns(replace.into())
            }
            Action::AddFields(add) => pb::action::Action::AddFields(add.into()),
            Action::ChangeSchema(change) => pb::action::Action::ChangeSchema(change.into()),
            Action::RefreshRowVersionMetadata(refresh) => {
                pb::action::Action::RefreshRowVersionMetadata(refresh.into())
            }
        };
        Self {
            action: Some(action),
        }
    }
}

impl TryFrom<pb::Action> for Action {
    type Error = Error;

    fn try_from(proto: pb::Action) -> Result<Self> {
        match proto.action {
            Some(pb::action::Action::AddFragments(add)) => Ok(Self::AddFragments(add.try_into()?)),
            Some(pb::action::Action::AddBases(add)) => Ok(Self::AddBases(add.try_into()?)),
            Some(pb::action::Action::ReplaceFragmentColumns(replace)) => {
                Ok(Self::ReplaceFragmentColumns(replace.try_into()?))
            }
            Some(pb::action::Action::AddFields(add)) => Ok(Self::AddFields(add.try_into()?)),
            Some(pb::action::Action::ChangeSchema(change)) => {
                Ok(Self::ChangeSchema(change.try_into()?))
            }
            Some(pb::action::Action::RefreshRowVersionMetadata(refresh)) => {
                Ok(Self::RefreshRowVersionMetadata(refresh.try_into()?))
            }
            None => Err(Error::invalid_input(
                "Action protobuf has no variant set".to_string(),
            )),
        }
    }
}

impl From<&UserAction> for pb::UserAction {
    fn from(user_action: &UserAction) -> Self {
        Self {
            description: user_action.description.clone(),
            actions: user_action.actions.iter().map(pb::Action::from).collect(),
        }
    }
}

impl TryFrom<pb::UserAction> for UserAction {
    type Error = Error;

    fn try_from(proto: pb::UserAction) -> Result<Self> {
        Ok(Self {
            description: proto.description,
            actions: proto
                .actions
                .into_iter()
                .map(Action::try_from)
                .collect::<Result<_>>()?,
        })
    }
}

impl From<&UserOperation> for pb::UserOperation {
    fn from(op: &UserOperation) -> Self {
        Self {
            description: op.description.clone(),
            uuid: op.uuid.clone(),
            read_version: op.read_version,
            actions: op.actions.iter().map(pb::UserAction::from).collect(),
        }
    }
}

impl TryFrom<pb::UserOperation> for UserOperation {
    type Error = Error;

    fn try_from(proto: pb::UserOperation) -> Result<Self> {
        Ok(Self {
            description: proto.description,
            uuid: proto.uuid,
            read_version: proto.read_version,
            actions: proto
                .actions
                .into_iter()
                .map(UserAction::try_from)
                .collect::<Result<_>>()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use arrow_schema::DataType;

    #[test]
    fn user_operation_roundtrips_through_protobuf() {
        let op = UserOperation {
            description: "INSERT INTO t VALUES (1)".to_string(),
            uuid: "test-uuid".to_string(),
            read_version: 7,
            actions: vec![UserAction {
                description: "append batch".to_string(),
                actions: vec![Action::AddFragments(AddFragments {
                    fragments: vec![Fragment::new(0), Fragment::new(1)],
                })],
            }],
        };

        let proto = pb::UserOperation::from(&op);
        let roundtripped = UserOperation::try_from(proto).unwrap();

        assert_eq!(op, roundtripped);
    }

    fn assert_action_roundtrips(action: Action) {
        let proto = pb::Action::from(&action);
        let roundtripped = Action::try_from(proto).unwrap();
        assert_eq!(action, roundtripped);
    }

    #[test]
    fn add_bases_action_roundtrips() {
        assert_action_roundtrips(Action::AddBases(AddBases {
            bases: vec![
                BasePath::new(0, "s3://bucket/new-base".to_string(), None, true),
                BasePath::new(
                    5,
                    "s3://bucket/other-base".to_string(),
                    Some("other".to_string()),
                    false,
                ),
            ],
        }));
    }

    #[test]
    fn replace_fragment_columns_action_roundtrips() {
        assert_action_roundtrips(Action::ReplaceFragmentColumns(ReplaceFragmentColumns {
            fragment_id: 3,
            field_ids: vec![1, 2],
            new_data_files: vec![
                NewDataFile {
                    base: None,
                    file: DataFile::new("data/new.lance", vec![1, 2], vec![], 2, 1, None, None),
                },
                NewDataFile {
                    base: Some(BaseRef::Local(0)),
                    file: DataFile::new("data/external.lance", vec![3], vec![], 2, 1, None, None),
                },
            ],
            replacement: ColumnReplacement::InPlace,
            updated_row_offsets: None,
        }));

        assert_action_roundtrips(Action::ReplaceFragmentColumns(ReplaceFragmentColumns {
            fragment_id: 3,
            field_ids: vec![1],
            new_data_files: vec![NewDataFile {
                base: Some(BaseRef::Committed(9)),
                file: DataFile::new("data/tombstone.lance", vec![1], vec![], 2, 1, None, None),
            }],
            replacement: ColumnReplacement::Tombstone,
            updated_row_offsets: None,
        }));
    }

    #[test]
    fn add_fields_action_roundtrips() {
        let mut field = Field::new_arrow("new_col", DataType::Int32, true).unwrap();
        field.id = 10;

        assert_action_roundtrips(Action::AddFields(AddFields {
            new_fields: vec![field],
            fragment_files: vec![(
                3,
                vec![NewDataFile {
                    base: None,
                    file: DataFile::new("data/new_col.lance", vec![10], vec![], 2, 1, None, None),
                }],
            )],
        }));
    }

    #[test]
    fn change_schema_action_roundtrips() {
        let mut field = Field::new_arrow("recast_col", DataType::Int64, true).unwrap();
        field.id = 4;
        let schema = Schema {
            fields: vec![field],
            metadata: HashMap::from([("key".to_string(), "value".to_string())]),
        };

        assert_action_roundtrips(Action::ChangeSchema(ChangeSchema { schema }));
    }

    #[test]
    fn refresh_row_version_metadata_action_roundtrips() {
        assert_action_roundtrips(Action::RefreshRowVersionMetadata(
            RefreshRowVersionMetadata {
                fragment_ids: vec![1, 2, 3],
            },
        ));
    }
}
