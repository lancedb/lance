// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::format::{pb, Fragment};
use deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use snafu::{location, Location};

/// A change to a [`Manifest`].
#[derive(Debug, Clone, DeepSizeOf)]
pub enum Action {
    // Fragment changes
    AddFragments { fragments: Vec<Fragment> },
    DeleteFragments { deleted_fragment_ids: Vec<u64> },
    UpdateFragments { fragments: Vec<Fragment> },
}

impl TryFrom<pb::transaction::Action> for Action {
    type Error = Error;

    fn try_from(value: pb::transaction::Action) -> std::result::Result<Self, Self::Error> {
        if let Some(action) = value.action {
            Self::try_from(action)
        } else {
            Err(Error::NotSupported {
                source: "No known action was found".into(),
                location: location!(),
            })
        }
    }
}

impl TryFrom<pb::transaction::action::Action> for Action {
    type Error = Error;

    fn try_from(value: pb::transaction::action::Action) -> std::result::Result<Self, Self::Error> {
        use pb::transaction::action::Action::*;
        match value {
            AddFragments(action) => Ok(Self::AddFragments {
                fragments: action
                    .fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<_>>()?,
            }),
            DeleteFragments(action) => Ok(Self::DeleteFragments {
                deleted_fragment_ids: action.deleted_fragment_ids,
            }),
            UpdateFragments(action) => Ok(Self::UpdateFragments {
                fragments: action
                    .updated_fragments
                    .into_iter()
                    .map(Fragment::try_from)
                    .collect::<Result<_>>()?,
            }),
        }
    }
}

impl From<&Action> for pb::transaction::Action {
    fn from(value: &Action) -> Self {
        use pb::transaction::action::{self as pb_action};
        match value {
            Action::AddFragments { fragments } => Self {
                action: Some(pb_action::Action::AddFragments(pb_action::AddFragments {
                    fragments: fragments.iter().map(Into::into).collect(),
                })),
            },
            Action::DeleteFragments {
                deleted_fragment_ids,
            } => Self {
                action: Some(pb_action::Action::DeleteFragments(
                    pb_action::DeleteFragments {
                        deleted_fragment_ids: deleted_fragment_ids.clone(),
                    },
                )),
            },
            Action::UpdateFragments { fragments } => Self {
                action: Some(pb_action::Action::UpdateFragments(
                    pb_action::UpdateFragments {
                        updated_fragments: fragments.iter().map(Into::into).collect(),
                    },
                )),
            },
        }
    }
}
