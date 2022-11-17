//  Copyright 2022 Lance Authors
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

use crate::format::pb;
use arrow2::error::Error;

pub struct Metadata {
    pb: pb::Metadata,
}

impl Metadata {
    pub fn new(pb: pb::Metadata) -> Self {
        Self { pb }
    }

    pub fn num_batches(&self) -> usize {
        self.pb.batch_offsets.len() - 1
    }

    pub fn length(&self) -> usize {
        // Metadata::length
        // TODO need a link to the design doc
        match self.pb.batch_offsets.last() {
            None => 0,
            Some(v) => *v as usize,
        }
    }

    pub fn add_batch_length(&mut self, length: i32) {
        if self.pb.batch_offsets.len() == 0 {
            self.pb.batch_offsets.push(0)
        }
        self.pb.batch_offsets.push(self.length() as i32 + length)
    }

    pub fn locate_batch(&self, row_index: i32) -> Result<(i32, i32), Error> {
        // Metadata::LocateBatch
        let len = self.length();

        if row_index < 0 || row_index >= len as i32 {
            return Err(Error::InvalidArgumentError(format!(
                "Row index out of range: {} of {}",
                row_index, len
            )));
        }

        let bound_idx = self.pb.batch_offsets.partition_point(|x| x <= &row_index) as i32 - 1;

        if bound_idx < 0 {
            return Err(Error::InvalidArgumentError(format!(
                "metadata is not valid {:?}",
                self.pb.batch_offsets
            )));
        }

        let offset = row_index - self.pb.batch_offsets[bound_idx as usize];

        Ok((bound_idx, offset))
    }
}
