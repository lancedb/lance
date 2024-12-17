// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::utils::mask::RowIdMask;
use lance_io::object_store::ObjectStore;
use lance_table::format::Fragment;

use crate::{io::commit::Transaction, session::Session, Dataset, Result};

async fn resolve_conflicts(transaction: Transaction, dataset: &Dataset) -> Result<Transaction> {
    // Maybe I should grab them in here?
    // TODO: return cleanup task too?
    // TODO: nice errors differentiate retry-able and non-retry-able conflicts
    // TODO: get diff on deletions
    todo!()
}

/// Identify which rows have been deleted or moved by the transaction.
async fn build_diff(
    transaction: &Transaction,
    old_fragments: &[Fragment],
    object_store: &ObjectStore,
    session: &Session,
) -> Result<RowIdMask> {
    todo!()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::dataset::{InsertBuilder, MergeInsertBuilder};

    use super::*;

    #[tokio::test]
    async fn test_resolve_conflicts_noop() {
        todo!("Test that append, and other non-conflicting ones just return the same thing")
    }

    #[tokio::test]
    async fn test_resolve_upsert() {
        // TODO: measure the IOPS too
        // create a test dataset
        let batch = todo!();
        let mut dataset = InsertBuilder::new("memory://")
            .execute(vec![batch])
            .await
            .unwrap();
        let dataset = Arc::new(dataset);

        // do two upsert transactions
        let res = MergeInsertBuilder::try_new(dataset, id);

        // check we get Ok() if we upsert a different row from original read version

        // Check we get Ok() if we upsert same row from current read version

        // Check we get Err(RetryableFailure) if we upsert same row from original version

        todo!("Test that upserts are resolved correctly")
        // assert clean up task is returned and does proper cleanup
    }
}
