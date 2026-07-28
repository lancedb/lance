// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Action-based transactions: composable manifest deltas.
//!
//! A [`UserOperation`] is an ordered list of [`UserAction`] steps, each expanding
//! to granular [`Action`] deltas. An action records the *change* to the manifest
//! rather than a post-image, which is what lets several of them commit atomically
//! and lets two branches be merged. Minted identifiers travel as [`Ref::Local`]
//! tokens and are allocated when the operation is applied, so two concurrent
//! operations can never collide on an id.
//!
//! The wire format is `protos/transaction/actions.proto`, which holds the design
//! rationale and is authoritative for the vocabulary.
//!
//! The split of responsibility here is deliberate:
//!
//! ```text
//! action        the list of actions, and a router with one line per variant
//! action/mask     the manifest regions an action writes, and how they collide
//! action/refs     Local/Committed references and the apply-time bindings
//! action/apply    assembling a manifest from an operation's actions
//! action/translate  the legacy Operation vocabulary expressed as actions
//! action/<name>   one module per action: its struct, wire form, logic, tests
//! ```
//!
//! Every method on [`Action`] is a one-line-per-variant match. If a router method
//! grows a body, the body belongs in the action's own module.

pub mod add_base;
pub mod apply;
pub mod mask;
pub mod refs;
pub mod translate;

pub use add_base::AddBase;
pub use refs::Ref;

use crate::format::{Manifest, pb};
use mask::ManifestMask;
use refs::TxnContext;

use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// A single granular change to the manifest.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub enum Action {
    AddBase(AddBase),
}

impl Action {
    pub(crate) fn apply(&self, manifest: &mut Manifest, ctx: &mut TxnContext) -> Result<()> {
        match self {
            Self::AddBase(action) => action.apply(manifest, ctx),
        }
    }

    /// The manifest regions this action writes. Conflict is intersection; see
    /// [`ManifestMask`].
    pub(crate) fn writes(&self) -> ManifestMask {
        match self {
            Self::AddBase(action) => action.writes(),
        }
    }
}

/// A single user-recognizable step within a [`UserOperation`], e.g. "append
/// batch". The description keeps the transaction history readable when a range of
/// transactions is squashed.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct UserAction {
    pub description: String,
    pub actions: Vec<Action>,
}

/// A user-facing, composable transaction: an ordered list of steps that commit
/// atomically as a single manifest change.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct UserOperation {
    /// Human-readable description, e.g. `"INSERT INTO t VALUES (1)"`.
    pub description: String,
    /// Unique identifier for this operation, matching `Transaction::uuid` semantics.
    pub uuid: String,
    /// The dataset version this operation was planned against.
    pub read_version: u64,
    pub actions: Vec<UserAction>,
}

impl UserOperation {
    /// Every action in the operation, in application order.
    pub(crate) fn actions(&self) -> impl Iterator<Item = &Action> {
        self.actions.iter().flat_map(|step| step.actions.iter())
    }

    /// The union of the operation's actions' footprints.
    pub(crate) fn writes(&self) -> ManifestMask {
        self.actions()
            .fold(ManifestMask::default(), |mask, action| {
                mask.union(action.writes())
            })
    }
}

impl TryFrom<pb::Action> for Action {
    type Error = Error;

    fn try_from(message: pb::Action) -> Result<Self> {
        match message.action {
            Some(pb::action::Action::AddBase(action)) => Ok(Self::AddBase(action.try_into()?)),
            // An unrecognized arm — including one proto3 decodes to `None` because
            // it was written by a newer version — must error, never be skipped.
            // `load_and_sort_new_transactions` collects concurrent transactions with
            // `try_collect`, so failing here aborts the commit; parsing leniently
            // would instead drop a concurrent action out of conflict detection and
            // let two colliding commits both succeed. Do NOT make this lenient.
            Some(other) => Err(Error::not_supported(format!(
                "this version of Lance does not support the transaction action {other:?}"
            ))),
            None => Err(Error::not_supported(
                "Action message did not contain a recognized action; \
                 it was written by a newer version of Lance",
            )),
        }
    }
}

impl From<&Action> for pb::Action {
    fn from(action: &Action) -> Self {
        let action = match action {
            Action::AddBase(action) => pb::action::Action::AddBase(action.into()),
        };
        Self {
            action: Some(action),
        }
    }
}

impl TryFrom<pb::UserAction> for UserAction {
    type Error = Error;

    fn try_from(message: pb::UserAction) -> Result<Self> {
        Ok(Self {
            description: message.description,
            actions: message
                .actions
                .into_iter()
                .map(Action::try_from)
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl From<&UserAction> for pb::UserAction {
    fn from(step: &UserAction) -> Self {
        Self {
            description: step.description.clone(),
            actions: step.actions.iter().map(pb::Action::from).collect(),
        }
    }
}

impl TryFrom<pb::UserOperation> for UserOperation {
    type Error = Error;

    fn try_from(message: pb::UserOperation) -> Result<Self> {
        Ok(Self {
            description: message.description,
            uuid: message.uuid,
            read_version: message.read_version,
            actions: message
                .actions
                .into_iter()
                .map(UserAction::try_from)
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl From<&UserOperation> for pb::UserOperation {
    fn from(operation: &UserOperation) -> Self {
        Self {
            description: operation.description.clone(),
            uuid: operation.uuid.clone(),
            read_version: operation.read_version,
            actions: operation.actions.iter().map(pb::UserAction::from).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add_base(local: u32, name: Option<&str>, path: &str) -> Action {
        Action::AddBase(AddBase {
            local,
            name: name.map(str::to_string),
            is_dataset_root: false,
            path: path.to_string(),
        })
    }

    fn two_step_operation() -> UserOperation {
        UserOperation {
            description: "ALTER TABLE t ADD BASE".to_string(),
            uuid: "9f0b8f4e-0000-4000-8000-000000000000".to_string(),
            read_version: 3,
            actions: vec![
                UserAction {
                    description: "register warm base".to_string(),
                    actions: vec![add_base(0, Some("warm"), "s3://bucket/warm")],
                },
                UserAction {
                    description: "register unnamed base".to_string(),
                    actions: vec![add_base(1, None, "/local/cold")],
                },
            ],
        }
    }

    #[test]
    fn test_user_operation_roundtrips() {
        let operation = two_step_operation();
        let message = pb::UserOperation::from(&operation);
        assert_eq!(UserOperation::try_from(message).unwrap(), operation);
    }

    #[test]
    fn test_action_without_arm_is_rejected() {
        let err = Action::try_from(pb::Action { action: None }).unwrap_err();
        assert!(
            matches!(err, Error::NotSupported { .. }),
            "expected NotSupported, got: {err:?}"
        );
        assert!(err.to_string().contains("newer version of Lance"));
    }

    #[test]
    fn test_user_operation_containing_undecodable_action_is_rejected() {
        // Fail-closed must propagate through the envelope: a `UserOperation` whose
        // action list holds one undecodable entry is itself undecodable.
        let message = pb::UserOperation {
            description: "op".to_string(),
            uuid: "u".to_string(),
            read_version: 1,
            actions: vec![pb::UserAction {
                description: "step".to_string(),
                actions: vec![pb::Action { action: None }],
            }],
        };
        let err = UserOperation::try_from(message).unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }));
    }

    #[test]
    fn test_unsupported_action_arm_is_rejected() {
        // An arm this version does not implement yet must be rejected too, not
        // silently ignored.
        let message = pb::Action {
            action: Some(pb::action::Action::ResetTable(pb::ResetTable {})),
        };
        let err = Action::try_from(message).unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }));
        assert!(err.to_string().contains("does not support"));
    }

    #[test]
    fn test_writes_unions_across_steps() {
        let mask = two_step_operation().writes();
        assert_eq!(
            mask.conflicts_with(&add_base(0, Some("warm"), "s3://elsewhere").writes()),
            Some(mask::Conflict::Incompatible)
        );
        assert_eq!(
            mask.conflicts_with(&add_base(0, None, "/local/cold").writes()),
            Some(mask::Conflict::Incompatible)
        );
        assert_eq!(
            mask.conflicts_with(&add_base(0, Some("other"), "/local/other").writes()),
            None
        );
    }

    #[test]
    fn test_writes_of_empty_operation_is_disjoint_from_everything() {
        let empty = UserOperation {
            description: String::new(),
            uuid: String::new(),
            read_version: 0,
            actions: vec![],
        };
        assert_eq!(
            empty.writes().conflicts_with(&ManifestMask::everything()),
            None
        );
    }
}
