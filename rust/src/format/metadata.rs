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

use crate::format::{pb, ProtoStruct};

/// Data File Metadata
#[derive(Debug, Default, PartialEq)]
pub struct Metadata {
    /// Offset of each record batch.
    pub batch_offsets: Vec<i32>,

    /// The file position of the page table in the file.
    pub page_table_position: usize,

    /// The file position of the manifest block in the file.
    pub manifest_position: Option<usize>,
}

impl ProtoStruct for Metadata {
    type Proto = pb::Metadata;
}

impl From<&Metadata> for pb::Metadata {
    fn from(m: &Metadata) -> Self {
        Self {
            batch_offsets: m.batch_offsets.clone(),
            page_table_position: m.page_table_position as u64,
            manifest_position: m.manifest_position.unwrap_or(0) as u64,
        }
    }
}

impl From<pb::Metadata> for Metadata {
    fn from(m: pb::Metadata) -> Self {
        Self {
            batch_offsets: m.batch_offsets.clone(),
            page_table_position: m.page_table_position as usize,
            manifest_position: Some(m.manifest_position as usize),
        }
    }
}

impl Metadata {
    /// Get the number of batches in this file.
    pub fn num_batches(&self) -> usize {
        if self.batch_offsets.is_empty() {
            0
        } else {
            self.batch_offsets.len() - 1
        }
    }

    /// Get the number of records in this file
    pub fn len(&self) -> usize {
        *self.batch_offsets.last().unwrap_or(&0) as usize
    }

    /// Push the length of the batch.
    pub fn push_batch_length(&mut self, batch_len: i32) {
        if self.batch_offsets.is_empty() {
            self.batch_offsets.push(0)
        }
        self.batch_offsets
            .push(batch_len + self.batch_offsets.last().unwrap())
    }

    /// Get the starting offset of the batch.
    pub fn get_offset(&self, batch_id: i32) -> Option<i32> {
        self.batch_offsets.get(batch_id as usize).copied()
    }
}

#[cfg(test)]
mod tests {}
