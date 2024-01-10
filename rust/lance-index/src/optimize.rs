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

pub struct OptimizeOptions {
    /// The number of existing indices to merge into a single index, plus the un-indexed fragments.
    ///
    /// - If this is zero, then no merging will be done, which means a new delta index will be created
    ///   just to cover un-indexed fragments.
    /// - If it is one, we will append the un-indexed fragments to the last index.
    /// - If it is greater than one, we will merge the last `num_indices_to_merge` indices into a single
    ///   one, thus reduce the number of indices for this column.
    /// - If this number exceeds the number of existing indices, we will merge all existing indices into
    ///   a single one. So it is a re-write of the entire index.
    ///
    /// Note that no re-train of the index happens during the operation.
    pub num_indices_to_merge: usize,
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        Self {
            num_indices_to_merge: 0,
        }
    }
}
