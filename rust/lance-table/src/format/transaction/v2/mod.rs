// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod action;

pub use action::Action;

use crate::format::pb;
use deepsize::DeepSizeOf;
use lance_core::{Error, Result};

#[derive(Debug, Clone, DeepSizeOf)]
pub struct Transaction {
    pub read_version: u64,
    pub uuid: String,
    pub operations: Vec<UserOperation>,
    pub blob_ops: Vec<UserOperation>,
}

/// A group of actions that make up a user-facing operation.
///
/// For example, a user might call a `CREATE TABLE` statement, which includes
/// both a [Action::ReplaceSchema] and an [Action::AddFragments].
#[derive(Debug, Clone, DeepSizeOf, Default)]
pub struct UserOperation {
    pub read_version: u64,
    pub uuid: String,
    pub description: String,
    pub actions: Vec<Action>,
}

impl From<&UserOperation> for pb::transaction::UserOperation {
    fn from(value: &UserOperation) -> Self {
        Self {
            read_version: value.read_version,
            uuid: value.uuid.clone(),
            description: value.description.clone(),
            actions: value.actions.iter().map(Into::into).collect(),
        }
    }
}

impl TryFrom<pb::transaction::UserOperation> for UserOperation {
    type Error = Error;

    fn try_from(value: pb::transaction::UserOperation) -> std::result::Result<Self, Self::Error> {
        Ok(Self {
            read_version: value.read_version,
            uuid: value.uuid,
            description: value.description,
            actions: value
                .actions
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_>>()?,
        })
    }
}

impl From<&Transaction> for pb::Transaction {
    fn from(value: &Transaction) -> Self {
        let operation =
            pb::transaction::Operation::CompositeOperation(pb::transaction::CompositeOperation {
                user_operations: value.operations.iter().map(Into::into).collect(),
            });

        // TODO: Handle blob operations
        if !value.blob_ops.is_empty() {
            unimplemented!("Blob operations are not yet supported");
        }

        Self {
            read_version: value.read_version,
            uuid: value.uuid.clone(),
            operation: Some(operation),
            blob_operation: None,
            tag: Default::default(),
        }
    }
}
