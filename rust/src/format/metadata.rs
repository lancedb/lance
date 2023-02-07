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

use std::collections::BTreeMap;

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

#[derive(Debug, PartialEq)]
pub(crate) struct BatchOffsets {
    pub batch_id: i32,
    pub offsets: Vec<u32>,
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

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

    /// Get the length of the batch.
    pub fn get_batch_length(&self, batch_id: i32) -> Option<i32> {
        self.get_offset(batch_id + 1)
            .map(|o| o - self.get_offset(batch_id).unwrap_or_default())
    }

    /// Group row indices into each batch.
    ///
    /// The indices must be sorted.
    pub(crate) fn group_indices_to_batches(&self, indices: &[u32]) -> Vec<BatchOffsets> {
        let mut batch_id: i32 = 0;
        let num_batches = self.num_batches() as i32;
        let mut indices_per_batch: BTreeMap<i32, Vec<u32>> = BTreeMap::new();

        let mut indices = Vec::from(indices);
        indices.sort();

        for idx in indices.iter() {
            while batch_id < num_batches && *idx >= self.batch_offsets[batch_id as usize + 1] as u32
            {
                batch_id += 1;
            }
            indices_per_batch
                .entry(batch_id)
                .and_modify(|v| v.push(*idx))
                .or_insert(vec![*idx]);
        }

        indices_per_batch
            .iter()
            .map(|(batch_id, indices)| {
                let batch_offset = self.batch_offsets[*batch_id as usize];
                // Adjust indices to be the in-batch offsets.
                let in_batch_offsets = indices
                    .iter()
                    .map(|i| i - batch_offset as u32)
                    .collect::<Vec<_>>();
                BatchOffsets {
                    batch_id: *batch_id,
                    offsets: in_batch_offsets,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_indices_to_batch() {
        let mut metadata = Metadata::default();
        metadata.push_batch_length(20);
        metadata.push_batch_length(20);

        let batches = metadata.group_indices_to_batches(&[6, 24]);
        assert_eq!(batches.len(), 2);
        assert_eq!(
            batches,
            vec![
                BatchOffsets {
                    batch_id: 0,
                    offsets: vec![6]
                },
                BatchOffsets {
                    batch_id: 1,
                    offsets: vec![4]
                }
            ]
        );
    }
}
