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

use arrow_array::Int64Array;
use arrow_schema::DataType;
use std::collections::HashMap;

use crate::encodings::plain::PlainDecoder;
use crate::encodings::Decoder;
use crate::error::Result;
use crate::io::object_reader::ObjectReader;

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
    pub async fn new<'a>(
        reader: &'a ObjectReader<'_>,
        position: usize,
        num_columns: i32,
        num_batches: i32,
    ) -> Result<Self> {
        let length = num_columns * num_batches * 2;
        let decoder = PlainDecoder::new(reader, &DataType::Int64, position, length as usize)?;
        let raw_arr = decoder.decode().await?;
        let arr = raw_arr.as_any().downcast_ref::<Int64Array>().unwrap();

        let mut pages = HashMap::default();
        for col in 0..num_columns {
            pages.insert(col, HashMap::default());
            for batch in 0..num_batches {
                let idx = col * num_batches + batch;
                let batch_position = &arr.value((idx * 2) as usize);
                let batch_length = &arr.value((idx * 2 + 1) as usize);
                pages.get_mut(&col).unwrap().insert(
                    batch,
                    PageInfo {
                        position: *batch_position as usize,
                        length: *batch_length as usize,
                    },
                );
            }
        }

        Ok(Self { pages })
    }

    pub fn set_page_info(&mut self) {}

    pub fn get(&self, column: i32, batch: i32) -> Option<&PageInfo> {
        self.pages
            .get(&column)
            .map(|c_map| c_map.get(&batch))
            .flatten()
    }
}

#[cfg(test)]
mod tests {}
