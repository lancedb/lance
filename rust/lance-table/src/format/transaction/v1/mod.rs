// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;

use deepsize::DeepSizeOf;
use lance_core::{datatypes::Schema, Error};
use uuid::Uuid;

use crate::format::{action::Action, Fragment, Index};

pub mod operation;

/// A change to a dataset that can be retried
///
/// This contains enough information to be able to build the next manifest,
/// given the current manifest.
#[derive(Debug, Clone, DeepSizeOf)]
pub struct Transaction {
    /// The version of the table this transaction is based off of. If this is
    /// the first transaction, this should be 0.
    pub read_version: u64,
    pub uuid: String,
    pub operation: Operation,
    /// If the transaction modified the blobs dataset, this is the operation
    /// to apply to the blobs dataset.
    ///
    /// If this is `None`, then the blobs dataset was not modified
    pub blobs_op: Option<Operation>,
}

impl TryFrom<Transaction> for super::v2::Transaction {
    type Error = Error;

    fn try_from(value: Transaction) -> std::result::Result<Self, Self::Error> {
        let description = value.operation.description();
        let actions = value.operation.actions();
        let operation = super::v2::UserOperation {
            read_version: value.read_version,
            uuid: value.uuid,
            description,
            actions,
        };
        let blob_op = value.blobs_op.map(|op| {
            let description = op.description();
            let actions = op.actions();
            super::v2::UserOperation {
                read_version: value.read_version,
                uuid: value.uuid,
                description,
                actions,
            }
        });
        let blobs_op = if let Some(op) = blob_op {
            vec![op]
        } else {
            vec![]
        };
        Ok(Self {
            read_version: value.read_version,
            uuid: value.uuid,
            operations: vec![operation],
            blob_ops: blobs_op,
        })
    }
}
