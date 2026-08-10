// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Transaction struct for lance-table format layer.
//!
//! This struct is introduced to provide a Struct-first API for passing transaction
//! information within the lance-table crate. It mirrors the protobuf Transaction
//! message at a semantic level while remaining crate-local, so lance-table does
//! not depend on higher layers (e.g., lance crate).
//!
//! Conversion to protobuf occurs at the write boundary. See the `From<Transaction>`
//! implementation below.

use crate::format::pb;

#[derive(Clone, Debug, PartialEq)]
pub struct Transaction {
    /// Crate-local representation backing: protobuf Transaction.
    /// Keeping this simple avoids ring dependencies while still enabling
    /// Struct-first parameter passing in lance-table.
    pub inner: pb::Transaction,
}

impl Transaction {
    /// Accessor for testing or internal inspection if needed.
    pub fn as_pb(&self) -> &pb::Transaction {
        &self.inner
    }

    /// Whether this transaction can change the schema, and so can introduce or
    /// worsen an invalid primary key.
    ///
    /// The rest leave the key exactly as they found it, so a table that already
    /// carries an invalid one stays writable through them -- including the
    /// deletes needed to repair it. An unrecognized operation counts as
    /// schema-changing: an unknown write is not a safe one to exempt.
    pub fn may_change_schema(&self) -> bool {
        use pb::transaction::Operation;
        match self.inner.operation.as_ref() {
            Some(
                Operation::Append(_)
                | Operation::Delete(_)
                | Operation::CreateIndex(_)
                | Operation::Rewrite(_)
                | Operation::DataReplacement(_)
                | Operation::ReserveFragments(_)
                | Operation::Update(_)
                | Operation::UpdateConfig(_)
                | Operation::UpdateMemWalState(_)
                | Operation::UpdateBases(_),
            ) => false,
            _ => true,
        }
    }
}

/// Write-boundary conversion: serialize using protobuf at the last step.
impl From<Transaction> for pb::Transaction {
    fn from(tx: Transaction) -> Self {
        tx.inner
    }
}

impl From<pb::Transaction> for Transaction {
    fn from(pb_tx: pb::Transaction) -> Self {
        Self { inner: pb_tx }
    }
}
