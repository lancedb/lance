// Copyright 2023 Lance Developers.
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

//! HashJoiner

use std::collections::HashMap;

use arrow_array::RecordBatch;

/// `HashJoiner` does hash join on two datasets.
pub(super) struct HashJoiner {
    /// Hash value to row index map.
    index_map: HashMap<u64, Vec<usize>>,
}

impl HashJoiner {
    /// Create a new `HashJoiner`.
    pub fn new() -> Self {
        Self {
            index_map: HashMap::new(),
        }
    }

    /// Append a batch to the hash joiner.
    pub fn append_batch(&mut self, batch: &RecordBatch, on: &str, start_idx: usize) {
        let hash_column = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap();
        for i in 0..hash_column.len() {
            let hash = hash_column.value(i);
            let index = batch.index();
            self.index_map
                .entry(hash)
                .and_modify(|v| v.push(index))
                .or_insert(vec![index]);
        }
    }
}