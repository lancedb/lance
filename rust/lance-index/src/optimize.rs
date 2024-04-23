// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

/// Options for optimizing all indices.
#[derive(Debug)]
pub struct OptimizeOptions {
    /// Number of delta indices to merge for one column. Default: 1.
    ///
    /// If `num_indices_to_merge` is 0, a new delta index will be created.
    /// If `num_indices_to_merge` is 1, the delta updates will be merged into the latest index.
    /// If `num_indices_to_merge` is more than 1, the delta updates and latest N indices
    /// will be merged into one single index.
    ///
    /// It is up to the caller to decide how many indices to merge / keep. Callers can
    /// find out how many indices are there by calling [`Dataset::index_statistics`].
    ///
    /// A common usage pattern will be that, the caller can keep a large snapshot of the index of the base version,
    /// and accumulate a few delta indices, then merge them into the snapshot.
    pub num_indices_to_merge: usize,
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        Self {
            num_indices_to_merge: 1,
        }
    }
}
