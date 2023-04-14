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

use arrow_array::{cast::as_struct_array, Array, RecordBatch, RecordBatchReader, StructArray};
use arrow_schema::{ArrowError, SchemaRef};
use arrow_select::interleave::interleave;

use crate::Result;

/// RecordBatchBuffer is a in-memory buffer for multiple [`RecordBatch`]s.
///
///
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

    pub fn finish(&self) -> Result<Vec<RecordBatch>> {
        Ok(self.batches.clone())
    }

    /// Make interleaving indices to be used with [`arrow_select::interleave::interleave`].
    ///
    fn make_interleaving_indices(&self, indices: &[usize]) -> Vec<(usize, usize)> {
        let mut lengths = vec![0_usize];
        for batch in self.batches.iter() {
            lengths.push(lengths.last().unwrap() + batch.num_rows());
        }

        let mut idx = vec![];
        for i in indices {
            let batch_id = match lengths.binary_search(&i) {
                Ok(i) => i,
                Err(i) => i - 1,
            };
            idx.push((batch_id, i - lengths[batch_id]));
        }
        idx
    }

    /// Take rows by indices.
    pub fn take_rows(&self, indices: &[usize]) -> Result<RecordBatch> {
        let arrays = self
            .batches
            .iter()
            .map(|batch| StructArray::from(batch.clone()))
            .collect::<Vec<_>>();
        let refs = arrays.iter().map(|a| a as &dyn Array).collect::<Vec<_>>();

        let interleaving_indices = self.make_interleaving_indices(indices);
        let array = interleave(&refs, &interleaving_indices)?;
        Ok(as_struct_array(&array).into())
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

impl FromIterator<RecordBatch> for RecordBatchBuffer {
    fn from_iter<T: IntoIterator<Item = RecordBatch>>(iter: T) -> Self {
        let batches = iter.into_iter().collect::<Vec<_>>();
        Self::new(batches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::{Float64Array, Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    #[test]
    fn test_take() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("f", DataType::Float64, false),
            Field::new("s", DataType::Utf8, false),
        ]));

        let batch_buffer: RecordBatchBuffer = (0..5)
            .map(|v| {
                let values = (v * 10..v * 10 + 10).collect::<Vec<_>>();
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from_iter(values.iter().copied())),
                        Arc::new(Float64Array::from_iter(values.iter().map(|v| *v as f64))),
                        Arc::new(StringArray::from_iter_values(
                            values.iter().map(|v| format!("str_{}", v)),
                        )),
                    ],
                )
                .unwrap()
            })
            .collect();
        let batch = batch_buffer.take_rows(&[10, 14, 30, 49, 0, 22]).unwrap();
        assert_eq!(batch.num_rows(), 6);
        assert_eq!(
            batch.column_by_name("i").unwrap().as_ref(),
            &Int32Array::from(vec![10, 14, 30, 49, 0, 22])
        );
    }
}
