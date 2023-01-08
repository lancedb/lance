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

use arrow_array::RecordBatch;
use arrow_schema::{Schema as ArrowSchema, SchemaRef};

use super::Dataset;
use crate::datatypes::Schema;
use crate::error::Result;
use crate::format::Fragment;
use crate::io::reader::FileReader;

/// Dataset Scanner
///
/// TODO: RecordBatchReader does not support async iterator yet
/// we need to use a tokio::rt::block_on for the iterator.
pub struct Scanner<'a> {
    dataset: &'a Dataset,

    projections: Schema,

    // filter: how to present filter
    limit: Option<i64>,
    offset: Option<i64>,

    fragments: Vec<Fragment>,

    // use for iterator
    fragment_idx: usize,
    batch_id: i32,
    reader: Option<FileReader<'a>>,
}

impl<'a> Scanner<'a> {
    pub fn new(dataset: &'a Dataset) -> Result<Self> {
        Ok(Self {
            dataset,
            projections: dataset.schema().clone(),
            limit: None,
            offset: None,
            fragments: dataset.fragments().to_vec(),

            fragment_idx: 0,
            batch_id: 0,
            reader: None,
        })
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

    pub async fn next_batch(&mut self) -> Option<Result<RecordBatch>> {
        if self.fragment_idx >= self.fragments.len() {
            return None;
        }

        let fragment = &self.fragments[self.fragment_idx];
        // Only support 1 data file (no schema evolution for now);
        assert!(fragment.files.len() == 1);
        let data_file = &fragment.files[0];
        self.fragment_idx += 1;
        let path = self.dataset.data_dir().child(data_file.path.clone());
        if self.reader.is_none() {
            self.reader = Some(
                FileReader::new(
                    &self.dataset.object_store,
                    &path,
                    Some(&self.dataset.manifest),
                )
                .await
                .unwrap(),
            );
        }

        if let Some(reader) = &self.reader {
            let batch = reader.read_batch(self.batch_id).await;
            self.batch_id += 1;
            if self.batch_id as usize >= reader.num_batches() {
                self.batch_id = 0;
                self.fragment_idx += 1;
                self.reader = None;
            }
            return Some(batch);
        } else {
            panic!("should not reach here");
        }
    }

    pub fn schema(&self) -> SchemaRef {
        Arc::new(ArrowSchema::from(&self.projections))
    }
}
