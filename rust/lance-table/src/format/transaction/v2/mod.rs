// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use deepsize::DeepSizeOf;

pub mod action;

use super::UserOperation;

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
