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

use arrow_array::{Array, RecordBatchReader};

use crate::arrow::{RecordBatchBuffer, hash};
use crate::{Error, Result};

/// `HashJoiner` does hash join on two datasets.
pub(super) struct HashJoiner {
    /// Hash value to row index map.
    index_map: HashMap<u64, Vec<usize>>,

    data: RecordBatchBuffer,

    on_column: String,
}


impl HashJoiner {
    /// Create a new `HashJoiner`.
    pub fn try_new(reader: &mut dyn RecordBatchReader, on: &str) -> Result<Self> {
        // Check column exist
        reader.schema().field_with_name(on)?;

        Ok(Self {
            index_map: HashMap::new(),

            // Hold all data in memory for simple implementation. Can do external sort later.
            data: reader.collect::<std::result::Result<RecordBatchBuffer, _>>()?,

            on_column: on.to_string(),
        })
    }

    pub fn build(&mut self) -> Result<()> {
        let mut start_idx = 0;

        for batch in &self.data.batches {
            let key_column = batch.column_by_name(&self.on_column).ok_or_else(|| {
                Error::IO(format!("HashJoiner: Column {} not found", self.on_column))
            })?;

            let hashes = hash(key_column.as_ref())?;
            for (i, hash_value) in hashes.iter().enumerate() {
                let idx = start_idx + i;
                if let Some(v) = hash_value {
                    self.index_map
                        .entry(v)
                        .or_insert_with(Vec::new)
                        .push(idx);
                }
            }
            start_idx += batch.num_rows();
        }
        Ok(())
    }
}
