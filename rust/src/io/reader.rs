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

//! Lance Data File Reader

// Standard
use std::cmp::max;
use std::io::{Error, ErrorKind, Result};
use std::ops::Range;
use std::sync::Arc;

use arrow_array::types::{Int16Type, Int32Type, Int64Type, Int8Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type, Float16Type, Float32Type, Float64Type};
// Third party
use arrow_array::{ArrayRef, RecordBatch, Int16Array, Int32Array};
use arrow_schema::DataType;
use byteorder::{ByteOrder, LittleEndian};
use object_store::path::Path;
use prost::Message;

// Lance
use crate::datatypes::{Field, Schema};
use crate::encodings::plain::PlainDecoder;
use crate::encodings::Decoder;
use crate::format::Manifest;
use crate::format::{pb, Metadata, PageTable};
use crate::io::object_reader::ObjectReader;
use crate::io::{read_message, read_metadata_offset};

use super::object_store::ObjectStore;

/// Read Manifest on URI.
pub async fn read_manifest(object_store: &ObjectStore, path: &Path) -> Result<Manifest> {
    let file_size = object_store.inner.head(path).await?.size;
    const PREFETCH_SIZE: usize = 64 * 1024;
    let range = Range {
        start: std::cmp::max(file_size as i64 - PREFETCH_SIZE as i64, 0) as usize,
        end: file_size,
    };
    let buf = object_store.inner.get_range(path, range).await?;
    if buf.len() < 16 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Invalid format: file size is smaller than 8 bytes",
        ));
    }
    if !buf.ends_with(super::MAGIC) {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Invalid format: magic number does not match",
        ));
    }
    let manifest_pos = LittleEndian::read_i64(&buf[buf.len() - 16..buf.len() - 8]) as usize;
    assert!(file_size - manifest_pos < buf.len());
    let proto =
        pb::Manifest::decode(&buf[buf.len() - (file_size - manifest_pos) + 4..buf.len() - 16])?;
    Ok(Manifest::from(&proto))
}

/// Lance File Reader.
///
/// It reads arrow data from one data file.
pub struct FileReader<'a> {
    object_reader: ObjectReader<'a>,
    metadata: Metadata,
    manifest: Manifest,
    page_table: PageTable,
    projection: Option<Schema>,
}

impl<'a> FileReader<'a> {
    /// Open file reader
    pub async fn new(
        object_store: &'a ObjectStore,
        path: &Path,
        manifest: Option<Manifest>,
    ) -> Result<FileReader<'a>> {
        let mut object_reader =
            ObjectReader::new(object_store, path.clone(), object_store.prefetch_size())?;

        let file_size = object_reader.size().await?;
        let tail_bytes = object_reader
            .object_store
            .inner
            .get_range(
                &path,
                max(0 as usize, file_size - object_store.prefetch_size())..file_size,
            )
            .await?;
        let metadata_pos = read_metadata_offset(&tail_bytes)? as usize;
        let metadata_pb = if metadata_pos < file_size - tail_bytes.len() {
            // We have not read the metadata bytes yet.
            object_reader
                .read_message::<pb::Metadata>(metadata_pos)
                .await?
        } else {
            let offset = tail_bytes.len() - (file_size - metadata_pos);
            read_message::<pb::Metadata>(&tail_bytes.slice(offset..))?
        };
        let metadata = Metadata::from(&metadata_pb);

        let manifest_for_file = match manifest {
            Some(m) => m,
            None => {
                let manifest_pb = object_reader
                    .read_message::<pb::Manifest>(metadata.manifest_position.unwrap())
                    .await?;
                Manifest::from(&manifest_pb)
            }
        };
        let num_columns = manifest_for_file.schema.max_field_id().unwrap();
        let page_table = PageTable::new(
            &object_reader,
            metadata.page_table_position,
            num_columns,
            metadata.num_batches() as i32,
        )
        .await?;

        Ok(Self {
            object_reader,
            metadata,
            projection: Some(manifest_for_file.schema.clone()),
            page_table,
            manifest: manifest_for_file,
        })
    }

    pub fn set_projection(&mut self, schema: Schema) {
        self.projection = Some(schema)
    }

    pub fn schema(&self) -> &Schema {
        self.projection.as_ref().unwrap()
    }

    pub async fn read_batch(&self, batch_id: i32) -> Result<RecordBatch> {
        let schema = self.projection.as_ref().unwrap();
        // TODO spawn more threads
        let mut arrs = vec![];
        for field in schema.fields.iter() {
            let arr = self.read_array(&field, batch_id).await?;
            arrs.push(arr);
        }
        RecordBatch::try_new(Arc::new(schema.into()), arrs)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))
    }

    /// Read primitive array for batch `batch_idx`.
    ///
    async fn read_primitive_array(&self, field: &Field, batch_id: i32) -> Result<ArrayRef> {
        let column = field.id;
        let page_info = self.page_table.get(column, batch_id).ok_or(Error::new(
            ErrorKind::InvalidData,
            format!(
                "field: {} batch {} does not exist in page table",
                field, batch_id
            ),
        ))?;
        use DataType::*;

        macro_rules! create_plain_decoder {
            ($a:ty) => {
                Box::new(PlainDecoder::<$a>::new(
                    &self.object_reader,
                    page_info.position,
                    page_info.length,
                )?)
            };
        }

        let decoder: Box<dyn Decoder> = match field.data_type() {
            Int8 => create_plain_decoder!(Int8Type),
            Int16 => create_plain_decoder!(Int16Type),
            Int32 => create_plain_decoder!(Int32Type),
            Int64 => create_plain_decoder!(Int64Type),
            UInt8 => create_plain_decoder!(UInt8Type),
            UInt16 => create_plain_decoder!(UInt16Type),
            UInt32 => create_plain_decoder!(UInt32Type),
            UInt64 => create_plain_decoder!(UInt64Type),
            Float16 => create_plain_decoder!(Float16Type),
            Float32 => create_plain_decoder!(Float32Type),
            Float64 => create_plain_decoder!(Float64Type),
            _ => return Err(Error::new(ErrorKind::InvalidData, format!("Unsupport primitive type: {}", field.data_type()))),
        };
        decoder.decode().await
    }

    /// Read an array of the batch.
    async fn read_array(&self, field: &Field, batch_id: i32) -> Result<ArrayRef> {
        if field.data_type().is_numeric() {
            self.read_primitive_array(field, batch_id).await
        } else {
            println!("Reading: {}", field);
            let page_info = self.page_table.get(field.id, batch_id).unwrap();
            Ok(Arc::new(Int32Array::from((0..page_info.length as i32).collect::<Vec<_>>())))
        }
    }
}

#[cfg(test)]
mod tests {
    use object_store::path::Path;

    use super::FileReader;
    use crate::io::ObjectStore;

    #[tokio::test]
    async fn test_read_real_file() {
        let store = ObjectStore::new("/Users/lei/work/lance/rust").unwrap();
        let path = Path::from("/Users/lei/work/lance/rust/data/7d159bc9-36f2-40ae-8cfa-baadd7178981_0.lance");

        let reader = FileReader::new(&store, &path, None).await.unwrap();
        let batch = reader.read_batch(0).await.unwrap();
    }
}
