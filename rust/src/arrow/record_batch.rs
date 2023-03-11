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

//! Additional utility for [`RecordBatch`]
//!

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use arrow_select::concat::concat_batches;

use crate::Result;

#[derive(Debug)]
pub struct RecordBatchBuffer {
    pub batches: Vec<RecordBatch>,
    idx: usize,
}

impl RecordBatchBuffer {
    pub fn new(batches: Vec<RecordBatch>) -> Self {
        Self { batches, idx: 0 }
    }

    pub fn empty() -> Self {
        Self {
            batches: vec![],
            idx: 0,
        }
    }

    pub fn num_rows(&self) -> usize {
        self.batches.iter().map(|b| b.num_rows()).sum()
    }

    pub fn finish(&self) -> Result<RecordBatch> {
        // TODO arrow concat here concat_batches messes up the dictionaries
        // FIX write multiple arrays in the same group
            //  group -> lance layout grouping rows together, each encoder write 1 group
            //  group has multiple column, each page is one column
            //  in arrow
            //     RecordBatch == group
            //     Column == Page
            //  here just return the vec / list of batches
            //  this is for later / not needed
            //      but each batch should have at most "params.max_rows_per_group" rows
            //      if not move to another batch
            //      do a slice_window without copying the data
        Ok(concat_batches(&self.schema(), self.batches.iter())?)
    }
}

impl RecordBatchReader for RecordBatchBuffer {
    fn schema(&self) -> SchemaRef {
        self.batches[0].schema()
    }
}

impl Iterator for RecordBatchBuffer {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.batches.len() {
            let idx = self.idx;
            self.idx += 1;
            Some(Ok(self.batches[idx].clone()))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow::{array::StringDictionaryBuilder, datatypes::Int32Type};
    use arrow_array::{StructArray, ArrayRef};
    use arrow_schema::{Schema as ArrowSchema, Field as ArrowField, DataType};

    #[test]
    fn test_batch_dict_arrays() {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "s",
            DataType::Struct(vec![ArrowField::new(
                "d",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            )]),
            true,
        )]));

        let batches: Vec<RecordBatch> = (0..10).map(|v| {
            let mut dict_builder = StringDictionaryBuilder::<Int32Type>::new();
            dict_builder.append_null();
            dict_builder.append("a").unwrap();
            dict_builder.append("b").unwrap();
            dict_builder.append("c").unwrap();

            let struct_array = Arc::new(StructArray::from(vec![(
                ArrowField::new(
                    "d",
                    DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                    true,
                ),
                Arc::new(dict_builder.finish()) as ArrayRef,
            )]));

            RecordBatch::try_new(arrow_schema.clone(), vec![struct_array.clone()]).unwrap()
        }).collect();
        let buffer = RecordBatchBuffer::new(batches);
        let batch = buffer.finish().unwrap();
        println!("Batch is: {:?}", batch);
    }
}

