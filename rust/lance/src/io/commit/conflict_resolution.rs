// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

/// Given a set of uncommitted transactions, detect conflicts and attempt to
/// resolve them.
///
/// Conflicts are detected both with respect to the current state of the dataset
/// (transactions since the `read_version` of a new transaction) and with respect
/// to others in the set of uncommitted transactions.
///
/// If `allow_partial` is true, the resolution may result in a partial transaction
/// that can be committed. Any transactions that can't be automatically resolved
/// will be returned in the `rejected` field of the result.
///
/// When transactions are merged, the files they reference may be rewritten.
/// If a file is rewritten, the original ones will be deleted. However, the
/// rejected transactions will not be cleaned up.
fn resolve_conflicts(
    dataset: &Dataset,
    transactions: Vec<Transaction>,
    allow_partial: bool,
) -> Result<ResolutionResult> {
    todo!()
}

struct ResolutionResult {
    pub merged: Option<Transaction>,
    pub rejected: Vec<Transaction>,
}

fn cleanup_transaction(dataset: &Dataset, transaction: &Transaction) -> Result<()> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test cleanup
    // test with stable row ids
}
