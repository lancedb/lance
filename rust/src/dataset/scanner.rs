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

use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, Schema as ArrowSchema, SchemaRef};

use super::Dataset;
use crate::datatypes::Schema;
use crate::error::Result;

/// Dataset Scanner
#[derive(Debug)]
pub struct Scanner<'a> {
    dataset: &'a Dataset,

    projections: Schema,

    batch_size: usize,

    // filter: how to present filter
    limit: Option<i64>,
    offset: Option<i64>,
}

impl<'a> Scanner<'a> {
    pub fn new(dataset: &'a Dataset) -> Self {
        Self {
            dataset,
            projections: dataset.schema().clone(),
            batch_size: 1024,
            limit: None,
            offset: None,
        }
    }

    pub fn project(&mut self, columns: &[&str]) -> Result<&mut Self> {
        self.projections = self.dataset.schema().project(columns)?;
        Ok(self)
    }

    pub fn limit(&mut self, limit: i64, offset: Option<i64>) -> &mut Self {
        self.limit = Some(limit);
        self.offset = offset;
        self
    }
}

impl<'a> RecordBatchReader for Scanner<'a> {
    fn schema(&self) -> SchemaRef {
        Arc::new(ArrowSchema::from(&self.projections))
    }
}

impl<'a> Iterator for Scanner<'a> {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}
