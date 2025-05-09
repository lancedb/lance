// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::Result;
use lance_table::format::{Fragment, Manifest};
use snafu::location;

use crate::{
    dataset::transaction::{ConflictResult, Transaction},
    Dataset,
};

pub struct ConflictResolver {
    /// Relevant fragments as they were at the read version of the transaction.
    /// Has original fragment, plus a bool indicating whether a rewrite is needed.
    initial_fragments: Vec<(Fragment, bool)>,
}

impl ConflictResolver {
    pub async fn try_new(dataset: &Dataset, transaction: &Transaction) -> Result<Self> {
        todo!("assert that dataset is at the read version of the transaction");

        // Get the affected fragments, so we can watch them.

        todo!()
    }

    /// Check whether the transaction conflicts with another transaction.
    ///
    /// Will return an error if the transaction is not valid. Otherwise, it will
    /// return Ok(()).
    pub fn check_txn(
        &mut self,
        transaction: &Transaction,
        other_transaction: Option<&Transaction>,
        other_manifest: &Manifest,
    ) -> Result<()> {
        let other_version = other_manifest.version;
        let Some(other_transaction) = other_transaction else {
            return Err(crate::Error::Internal {
                message: format!(
                    "There was a conflicting transaction at version {}, \
                    and it was missing transaction metadata.",
                    other_version
                ),
                location: location!(),
            });
        };

        match transaction.conflicts_with(other_transaction) {
            ConflictResult::Compatible => Ok(()),
            ConflictResult::NotCompatible => {
                Err(crate::Error::CommitConflict {
                    version: other_version,
                    source: format!(
                        "This {} transaction is incompatible with concurrent transaction {} at version {}.",
                        transaction.operation, other_transaction.operation, other_version).into(),
                    location: location!(),
                })
            },
            ConflictResult::Retryable => {
                todo!("check if we can just rewrite deletion files");

                Err(crate::Error::RetryableCommitConflict {
                    version: other_version,
                    source: format!(
                        "This {} transaction was preempted by concurrent transaction {} at version {}. Please retry.",
                        transaction.operation, other_transaction.operation, other_version).into(),
                    location: location!() })
            }
        }
    }

    /// Writes
    pub async fn update_files(&self) -> Result<Transaction> {
        // TODO: What do we do when the other transaction was an upsert that rewrote
        // fragments rather than just touched the deletion file? This would generally
        // be easier to handle if the transaction was simply a diff.
        // Maybe what we can do, is:
        // (1) We grab the fragments as they are at read_version
        // (2) We keep around the data files that were (a) present at read_version
        //     and (b) are in fragments this transaction is touching.
        // (3) We can use those sets of data files to check if the other transaction
        //     modified the data files we are touching, and not just the deletion files.
        todo!()
    }
}
