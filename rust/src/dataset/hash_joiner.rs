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

use arrow_array::StructArray;
use arrow_array::{
    builder::UInt64Builder, cast::as_struct_array, Array, RecordBatch, RecordBatchReader,
};
use arrow_select::{concat::concat_batches, take::take};

use crate::arrow::hash;
use crate::{Error, Result};

/// `HashJoiner` does hash join on two datasets.
pub(super) struct HashJoiner {
    /// Hash value to row index map.
    index_map: HashMap<u64, usize>,

    batch: RecordBatch,

    on_column: String,
}

impl HashJoiner {
    /// Create a new `HashJoiner`.
    pub fn try_new(reader: &mut dyn RecordBatchReader, on: &str) -> Result<Self> {
        // Check column exist
        reader.schema().field_with_name(on)?;

        // Hold all data in memory for simple implementation. Can do external sort later.
        let batches = reader.collect::<std::result::Result<Vec<RecordBatch>, _>>()?;
        if batches.is_empty() {
            return Err(Error::IO("HashJoiner: No data".to_string()));
        };
        let batch = concat_batches(&batches[0].schema(), &batches)?;

        Ok(Self {
            index_map: HashMap::new(),
            batch,
            on_column: on.to_string(),
        })
    }

    /// Build the hash index.
    pub(super) fn build(&mut self) -> Result<()> {
        let key_column = self
            .batch
            .column_by_name(&self.on_column)
            .ok_or_else(|| Error::IO(format!("HashJoiner: Column {} not found", self.on_column)))?;

        let hashes = hash(key_column.as_ref())?;
        for (i, hash_value) in hashes.iter().enumerate() {
            let Some(key) = hash_value else {
                    continue;
                };

            if self.index_map.contains_key(&key) {
                return Err(Error::IO(format!("HashJoiner: Duplicate key {}", key)));
            }
            // TODO: use [`HashMap::try_insert`] when it's stable.
            self.index_map.insert(key, i);
        }

        Ok(())
    }

    /// Collecting the data using the index column from left table.
    pub(super) fn collect(&self, index_column: &dyn Array) -> Result<RecordBatch> {
        let hashes = hash(index_column)?;
        let mut builder = UInt64Builder::with_capacity(index_column.len());
        for hash_value in hashes.iter() {
            let Some(key) = hash_value else {
                builder.append_null();
                continue;
            };

            if let Some(idx) = self.index_map.get(&key) {
                builder.append_value(*idx as u64);
            } else {
                builder.append_null();
            }
        }
        let indices = builder.finish();

        let struct_arr = StructArray::from(self.batch.clone());
        let results = take(&struct_arr, &indices, None)?;
        Ok(as_struct_array(&results).into())
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use std::sync::Arc;

    use arrow_array::{Int32Array, StringArray, UInt32Array};
    use arrow_schema::{DataType, Field, Schema};

    use crate::arrow::RecordBatchBuffer;

    #[test]
    fn test_joiner_collect() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("s", DataType::Utf8, false),
        ]));

        let mut batch_buffer: RecordBatchBuffer = (0..5)
            .map(|v| {
                let values = (v * 10..v * 10 + 10).collect::<Vec<_>>();
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from_iter(values.iter().copied())),
                        Arc::new(StringArray::from_iter_values(
                            values.iter().map(|v| format!("str_{}", v)),
                        )),
                    ],
                )
                .unwrap()
            })
            .collect();
        let mut joiner = HashJoiner::try_new(&mut batch_buffer, "i").unwrap();
        joiner.build().unwrap();

        let indices = UInt32Array::from_iter(&[
            Some(15),
            None,
            Some(10),
            Some(0),
            None,
            None,
            Some(22),
            Some(11111), // not found
        ]);
        let results = joiner.collect(&indices).unwrap();

        assert_eq!(
            results.column_by_name("s").unwrap().as_ref(),
            &StringArray::from(vec![
                Some("str_15"),
                None,
                Some("str_10"),
                Some("str_0"),
                None,
                None,
                Some("str_22"),
                None // 11111 not found
            ])
        );
    }
}
