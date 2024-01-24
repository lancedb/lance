// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
