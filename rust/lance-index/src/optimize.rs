// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

/// Options for optimizing all indices.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct OptimizeOptions {
    /// Number of delta indices to merge for one column. Default: 1.
    ///
    /// If `num_indices_to_merge` is None, lance will create a new delta index if no partition is split, otherwise it will merge all delta indices.
    /// If `num_indices_to_merge` is Some(N), the delta updates and latest N indices
    /// will be merged into one single index.
    ///
    /// It is up to the caller to decide how many indices to merge / keep. Callers can
    /// find out how many indices are there by calling [`Dataset::index_statistics`].
    ///
    /// A common usage pattern will be that, the caller can keep a large snapshot of the index of the base version,
    /// and accumulate a few delta indices, then merge them into the snapshot.
    pub num_indices_to_merge: Option<usize>,

    /// the index names to optimize. If None, all indices will be optimized.
    pub index_names: Option<Vec<String>>,

    /// whether to retrain the whole index. Default: false.
    ///
    /// If true, the index will be retrained based on the current data,
    /// `num_indices_to_merge` will be ignored, and all indices will be merged into one.
    /// If false, the index will be optimized by merging `num_indices_to_merge` indices.
    ///
    /// This is useful when the data distribution has changed significantly,
    /// and we want to retrain the index to improve the search quality.
    /// This would be faster than re-create the index from scratch.
    ///
    /// NOTE: this option is only supported for v3 vector indices.
    pub retrain: bool,
}

impl OptimizeOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn merge(num: usize) -> Self {
        Self {
            num_indices_to_merge: Some(num),
            index_names: None,
            ..Default::default()
        }
    }

    pub fn append() -> Self {
        Self {
            num_indices_to_merge: Some(0),
            index_names: None,
            ..Default::default()
        }
    }

    pub fn retrain() -> Self {
        Self {
            num_indices_to_merge: None,
            index_names: None,
            retrain: true,
        }
    }

    pub fn num_indices_to_merge(mut self, num: Option<usize>) -> Self {
        self.num_indices_to_merge = num;
        self
    }

    pub fn index_names(mut self, names: Vec<String>) -> Self {
        self.index_names = Some(names);
        self
    }
}
