// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::builder::Int64Builder;
use arrow_array::{Array, Int64Array};
use arrow_schema::DataType;
use lance_io::encodings::plain::PlainDecoder;
use lance_io::encodings::Decoder;
use snafu::{location, Location};
use std::collections::BTreeMap;
use tokio::io::AsyncWriteExt;

use lance_core::{Error, Result};
use lance_io::traits::{Reader, Writer};

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
#[derive(Debug, Default, Clone, PartialEq)]
pub struct PageTable {
    /// map[field-id,  map[batch-id, PageInfo]]
    pages: BTreeMap<i32, BTreeMap<i32, PageInfo>>,
}

impl PageTable {
    /// Load [PageTable] from disk.
    ///
    /// The field_ids that are loaded are `field_id_offset` to `field_id_offset + num_columns`.
    /// `field_id_offset` should be the smallest field_id in the schema. `num_columns` should
    /// be the total unique number of field ids, including struct fields despite the fact
    /// they have no data pages.
    pub async fn load<'a>(
        reader: &dyn Reader,
        position: usize,
        num_columns: i32,
        num_batches: i32,
        field_id_offset: i32,
    ) -> Result<Self> {
        let length = num_columns * num_batches * 2;
        let decoder = PlainDecoder::new(reader, &DataType::Int64, position, length as usize)?;
        let raw_arr = decoder.decode().await?;
        let arr = raw_arr.as_any().downcast_ref::<Int64Array>().unwrap();

        let mut pages = BTreeMap::default();
        for col in 0..num_columns {
            let field_id = col + field_id_offset;
            pages.insert(field_id, BTreeMap::default());
            for batch in 0..num_batches {
                let idx = col * num_batches + batch;
                let batch_position = &arr.value((idx * 2) as usize);
                let batch_length = &arr.value((idx * 2 + 1) as usize);
                pages.get_mut(&field_id).unwrap().insert(
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

    /// Write [PageTable] to disk.
    ///
    /// `field_id_offset` is the smallest field_id that is present in the schema.
    /// This might be a struct field, which has no data pages, but it still must
    /// be serialized to the page table per the format spec.
    ///
    /// Any (field_id, batch_id) combinations that are not present in the page table
    /// will be written as (0, 0) to indicate an empty page.
    pub async fn write(&self, writer: &mut dyn Writer, field_id_offset: i32) -> Result<usize> {
        if self.pages.is_empty() {
            return Err(Error::InvalidInput {
                source: "empty page table".into(),
                location: location!(),
            });
        }

        dbg!(self);

        let pos = writer.tell().await?;
        let num_field_ids = self.pages.keys().max().unwrap() + 1 - field_id_offset;
        dbg!(num_field_ids);
        let num_batches = self
            .pages
            .values()
            .flat_map(|c_map| c_map.keys().max())
            .max()
            .unwrap()
            + 1;

        let mut builder = Int64Builder::with_capacity((num_field_ids * num_batches) as usize);
        for col in 0..num_field_ids {
            for batch in 0..num_batches {
                let field_id = col + field_id_offset;
                if let Some(page_info) = self.get(field_id, batch) {
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
    pub fn set(&mut self, field_id: i32, batch: i32, page_info: PageInfo) {
        self.pages
            .entry(field_id)
            .or_default()
            .insert(batch, page_info);
    }

    pub fn get(&self, field_id: i32, batch: i32) -> Option<&PageInfo> {
        self.pages
            .get(&field_id)
            .and_then(|c_map| c_map.get(&batch))
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use pretty_assertions::assert_eq;

    use lance_io::local::LocalObjectReader;

    #[test]
    fn test_set_page_info() {
        let mut page_table = PageTable::default();
        let page_info = PageInfo::new(1, 2);
        page_table.set(10, 20, page_info.clone());

        let actual = page_table.get(10, 20).unwrap();
        assert_eq!(actual, &page_info);
    }

    #[tokio::test]
    async fn test_roundtrip_page_info() {
        let mut page_table = PageTable::default();
        let page_info = PageInfo::new(1, 2);

        // Add fields 10..14, 4 batches with some missing
        page_table.set(10, 2, page_info.clone());
        page_table.set(11, 1, page_info.clone());
        // A hole at 12
        page_table.set(13, 0, page_info.clone());
        page_table.set(13, 1, page_info.clone());
        page_table.set(13, 2, page_info.clone());
        page_table.set(13, 3, page_info.clone());

        let test_dir = tempfile::tempdir().unwrap();
        let path = test_dir.path().join("test");

        // The first field_id with entries is 10, but if it's inside of a struct
        // the struct itself needs to be included in the page table. We use 9
        // here to represent the struct.
        let starting_field_id = 9;

        let mut writer = tokio::fs::File::create(&path).await.unwrap();
        let pos = page_table
            .write(&mut writer, starting_field_id)
            .await
            .unwrap();
        writer.shutdown().await.unwrap();

        let reader = LocalObjectReader::open_local_path(&path, 1024)
            .await
            .unwrap();
        let actual = PageTable::load(
            reader.as_ref(),
            pos,
            3,                 // There are three columns
            4,                 // 4 batches
            starting_field_id, // First field id is 10, but we want to start at 9
        )
        .await
        .unwrap();

        // Output should have filled in the empty pages.
        let mut expected = actual.clone();
        let default_page_info = PageInfo::new(0, 0);
        let expected_default_pages = [
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (10, 0),
            (10, 1),
            (10, 3),
            (11, 0),
            (11, 2),
            (11, 3),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
        ];
        for (field_id, batch) in expected_default_pages.iter() {
            expected.set(*field_id, *batch, default_page_info.clone());
        }

        assert_eq!(expected, actual);
    }
}
