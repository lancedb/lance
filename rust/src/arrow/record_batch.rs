// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Additional utility for [`RecordBatch`]
//!

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use arrow_select::concat::concat_batches;

use crate::Result;

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
