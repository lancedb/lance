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

use std::collections::HashMap;

use arrow_array::types::Int64;

use crate::encodings::plain::PlainDecoder;
use crate::io::object_store::ObjectReader;

#[derive(Debug)]
pub struct PageInfo {
    pub position: usize,
    pub length: usize,
}

#[derive(Debug, Default)]
pub struct PageTable {
    pages: HashMap<i32, HashMap<i32, PageInfo>>,
}

impl PageTable {
    /// Create page table from disk.
    pub async fn new(
        reader: &ObjectReader,
        position: usize,
        num_columns: i32,
        num_batches: i32,
    ) -> Result<Self> {
        let length = num_columns * num_batches * 2;
        let decoder = PlainDecoder::<Int64Type>::new(reader, reader.path, position, length);
        let arr = decode.decode().await?.as_any().downcast_array::<Int64Array>()?;

        for col in 0..num_columns {
            pages[col] = HashMap::default();
            for batch in 0..num_batches {
                let idx = col * num_batches + batch;
                let batch_position = arr.value(idx * 2);
                let batch_length = arr.value(idx * 2 + 1);
                pages[col][batch] = PageInfo{
                    position: batch_position,
                    length: batch_length,
                }
            }
        }
    }

    pub fn set_page_info(&mut self) {}

    pub fn get(&self, column: i32, batch: i32) -> Option<PageInfo> {
        pages.get(column).get(batch)
    }
}

#[cfg(test)]
mod tests {

}