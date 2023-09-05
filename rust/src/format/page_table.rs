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

use arrow_array::builder::Int64Builder;
use arrow_array::{Array, Int64Array};
use arrow_schema::DataType;
use std::collections::BTreeMap;
use tokio::io::AsyncWriteExt;

use crate::encodings::plain::PlainDecoder;
use crate::encodings::Decoder;
use crate::error::Result;
use crate::io::object_reader::ObjectReader;
use crate::io::object_writer::ObjectWriter;
use crate::Error;

#[derive(Clone, Debug, PartialEq)]
pub struct PageInfo {
    pub position: usize,
    pub length: usize,
}

impl PageInfo {
    pub fn new(position: usize, length: usize) -> Self {
        Self { position, length }
    }
}

/// Page lookup table.
///
#[derive(Debug, Default)]
pub struct PageTable {
    /// map[field-id,  map[batch-id, PageInfo]]
    pages: BTreeMap<i32, BTreeMap<i32, PageInfo>>,
}

impl PageTable {
    /// Load [PageTable] from disk.
    pub async fn load<'a>(
        reader: &dyn ObjectReader,
        position: usize,
        num_columns: i32,
        num_batches: i32,
    ) -> Result<Self> {
        let length = num_columns * num_batches * 2;
        let decoder = PlainDecoder::new(reader, &DataType::Int64, position, length as usize)?;
        let raw_arr = decoder.decode().await?;
        let arr = raw_arr.as_any().downcast_ref::<Int64Array>().unwrap();

        let mut pages = BTreeMap::default();
        for col in 0..num_columns {
            pages.insert(col, BTreeMap::default());
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

    pub async fn write(&self, writer: &mut ObjectWriter) -> Result<usize> {
        if self.pages.is_empty() {
            return Err(Error::InvalidInput {
                source: "empty page table".into(),
            });
        }

        let pos = writer.tell();
        let num_columns = self.pages.keys().max().unwrap() + 1;
        let num_batches = self
            .pages
            .values()
            .flat_map(|c_map| c_map.keys().max())
            .max()
            .unwrap()
            + 1;

        let mut builder = Int64Builder::with_capacity((num_columns * num_batches) as usize);
        for col in 0..num_columns {
            for batch in 0..num_batches {
                if let Some(page_info) = self.get(col, batch) {
                    builder.append_value(page_info.position as i64);
                    builder.append_value(page_info.length as i64);
                } else {
                    builder.append_slice(&[0, 0]);
                }
            }
        }
        let arr = builder.finish();
        writer
            .write_all(arr.into_data().buffers()[0].as_slice())
            .await?;

        Ok(pos)
    }

    /// Set page lookup info for a page identified by `(column, batch)` pair.
    pub fn set(&mut self, column: i32, batch: i32, page_info: PageInfo) {
        self.pages
            .entry(column)
            .or_insert_with(BTreeMap::default)
            .insert(batch, page_info);
    }

    pub fn get(&self, column: i32, batch: i32) -> Option<&PageInfo> {
        self.pages.get(&column).and_then(|c_map| c_map.get(&batch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_page_info() {
        let mut page_table = PageTable::default();
        let page_info = PageInfo::new(1, 2);
        page_table.set(10, 20, page_info.clone());

        let actual = page_table.get(10, 20).unwrap();
        assert_eq!(actual, &page_info);
    }
}
