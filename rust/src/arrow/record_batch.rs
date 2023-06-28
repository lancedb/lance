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

use crate::{Error, Result};

/// RecordBatchBuffer is a in-memory buffer for multiple [`RecordBatch`]s.
///
///
#[derive(Debug)]
pub struct RecordBatchBuffer {
    pub batches: Vec<RecordBatch>,
    pub schema: Option<SchemaRef>,
    idx: usize,
}

impl RecordBatchBuffer {
    pub fn new(batches: Vec<RecordBatch>, schema: Option<SchemaRef>) -> Self {
        Self {
            batches,
            idx: 0,
            schema,
        }
    }

    pub fn empty(schema: Option<SchemaRef>) -> Result<Self> {
        match schema {
            Some(schm) => Ok(Self {
                batches: vec![],
                idx: 0,
                schema: Some(schm),
            }),
            None => Err(Error::EmptyDatasetWithoutSchema {}),
        }
    }

    pub fn num_rows(&self) -> usize {
        self.batches.iter().map(|b| b.num_rows()).sum()
    }

    pub fn finish(&self) -> Result<Vec<RecordBatch>> {
        Ok(self.batches.clone())
    }
}

impl RecordBatchReader for RecordBatchBuffer {
    fn schema(&self) -> SchemaRef {
        if !self.batches.is_empty() {
            self.batches[0].schema()
        } else {
            self.schema.clone().unwrap()
        }
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
        Self::new(batches, None)
    }
}
