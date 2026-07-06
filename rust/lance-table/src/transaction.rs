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
//! Only [`AddFragments`] is implemented, as a proof of shape. The remaining
//! actions and the translation/conflict-resolution machinery land in follow-up
//! vertical slices.

use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

use crate::format::{Fragment, pb};

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

impl From<&Action> for pb::Action {
    fn from(action: &Action) -> Self {
        let action = match action {
            Action::AddFragments(add) => pb::action::Action::AddFragments(add.into()),
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
    use super::*;

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
}
