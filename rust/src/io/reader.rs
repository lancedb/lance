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
use std::ops::Range;
use std::sync::Arc;

use arrow_arith::arithmetic::subtract_scalar;
use arrow_array::cast::as_primitive_array;
use arrow_array::{ArrayRef, ListArray, RecordBatch, StructArray};
use arrow_schema::DataType;
use async_recursion::async_recursion;
use byteorder::{ByteOrder, LittleEndian};
use object_store::path::Path;
use prost::Message;

use crate::arrow::*;
use crate::datatypes::is_fixed_stride;
use crate::encodings::{dictionary::DictionaryDecoder, Decoder};
use crate::error::{Error, Result};
use crate::format::Manifest;
use crate::format::{pb, Metadata, PageTable};
use crate::io::object_reader::ObjectReader;
use crate::io::{read_metadata_offset, read_struct};
use crate::{
    datatypes::{Field, Schema},
    format::PageInfo,
};

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
        return Err(Error::IO(
            "Invalid format: file size is smaller than 16 bytes".to_string(),
        ));
    }
    if !buf.ends_with(super::MAGIC) {
        return Err(Error::IO(
            "Invalid format: magic number does not match".to_string(),
        ));
    }
    let manifest_pos = LittleEndian::read_i64(&buf[buf.len() - 16..buf.len() - 8]) as usize;
    assert!(file_size - manifest_pos < buf.len());
    let proto =
        pb::Manifest::decode(&buf[buf.len() - (file_size - manifest_pos) + 4..buf.len() - 16])?;
    Ok(Manifest::from(proto))
}

/// Lance File Reader.
///
/// It reads arrow data from one data file.
pub struct FileReader<'a> {
    object_reader: ObjectReader<'a>,
    metadata: Metadata,
    page_table: PageTable,
    projection: Option<Schema>,
}

impl<'a> FileReader<'a> {
    /// Open file reader
    pub async fn new(
        object_store: &'a ObjectStore,
        path: &Path,
        manifest: Option<&Manifest>,
    ) -> Result<FileReader<'a>> {
        let mut object_reader =
            ObjectReader::new(object_store, path.clone(), object_store.prefetch_size())?;

        let file_size = object_reader.size().await?;
        let tail_bytes = object_reader
            .object_store
            .inner
            .get_range(
                path,
                max(0, file_size - object_store.prefetch_size())..file_size,
            )
            .await?;
        let metadata_pos = read_metadata_offset(&tail_bytes)?;

        let metadata: Metadata = if metadata_pos < file_size - tail_bytes.len() {
            // We have not read the metadata bytes yet.
            object_reader.read_struct(metadata_pos).await?
        } else {
            let offset = tail_bytes.len() - (file_size - metadata_pos);
            read_struct(&tail_bytes.slice(offset..))?
        };

        let (projection, num_columns) = if let Some(m) = manifest {
            (m.schema.clone(), m.schema.max_field_id().unwrap())
        } else {
            let m: Manifest = object_reader
                .read_struct(metadata.manifest_position.unwrap())
                .await?;
            (m.schema.clone(), m.schema.max_field_id().unwrap())
        };
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
            projection: Some(projection),
            page_table,
        })
    }

    /// Set the projection [Schema].
    pub fn set_projection(&mut self, schema: Schema) {
        self.projection = Some(schema)
    }

    /// Schema of the returning RecordBatch.
    pub fn schema(&self) -> &Schema {
        self.projection.as_ref().unwrap()
    }

    pub fn num_batches(&self) -> usize {
        self.metadata.num_batches()
    }

    /// Read a batch of data from the file.
    ///
    /// The schema of the returned [RecordBatch] is set by [`FileReader::schema()`].
    pub async fn read_batch(&self, batch_id: i32) -> Result<RecordBatch> {
        let schema = self.projection.as_ref().unwrap();
        // TODO spawn more threads
        let mut arrs = vec![];
        for field in schema.fields.iter() {
            let arr = self.read_array(field, batch_id).await?;
            arrs.push(arr);
        }
        Ok(RecordBatch::try_new(Arc::new(schema.into()), arrs)?)
    }

    fn page_info(&self, field: &Field, batch_id: i32) -> Result<&PageInfo> {
        let column = field.id;
        self.page_table.get(column, batch_id).ok_or_else(|| {
            Error::IO(format!(
                "No page info found for field: {}, batch={}",
                field.name, batch_id
            ))
        })
    }

    /// Read primitive array for batch `batch_idx`.
    async fn read_fixed_stride_array(&self, field: &Field, batch_id: i32) -> Result<ArrayRef> {
        let page_info = self.page_info(field, batch_id)?;

        self.object_reader
            .read_fixed_stride_array(&field.data_type(), page_info.position, page_info.length)
            .await
    }

    async fn read_binary_array(&self, field: &Field, batch_id: i32) -> Result<ArrayRef> {
        let page_info = self.page_info(field, batch_id)?;

        self.object_reader
            .read_binary_array(&field.data_type(), page_info.position, page_info.length)
            .await
    }

    async fn read_dictionary_array(&self, field: &Field, batch_id: i32) -> Result<ArrayRef> {
        let page_info = self.page_info(field, batch_id)?;
        let data_type = field.data_type();
        let decoder = DictionaryDecoder::new(
            &self.object_reader,
            page_info.position,
            page_info.length,
            &data_type,
            field
                .dictionary
                .as_ref()
                .unwrap()
                .values
                .as_ref()
                .unwrap()
                .clone(),
        );
        decoder.decode().await
    }

    async fn read_struct_array(&self, field: &Field, batch_id: i32) -> Result<ArrayRef> {
        // TODO: use tokio to make the reads in parallel.
        let mut sub_arrays = vec![];
        for child in field.children.as_slice() {
            let arr = self.read_array(child, batch_id).await?;
            sub_arrays.push((child.into(), arr));
        }

        Ok(Arc::new(StructArray::from(sub_arrays)))
    }

    async fn read_list_array(&self, field: &Field, batch_id: i32) -> Result<ArrayRef> {
        let page_info = self.page_info(field, batch_id)?;
        let position_arr = self
            .object_reader
            .read_fixed_stride_array(&DataType::Int32, page_info.position, page_info.length)
            .await?;
        let positions = as_primitive_array(position_arr.as_ref());
        let start_position = positions.value(0);
        // Compute offsets
        let offset_arr = subtract_scalar(positions, start_position)?;
        let value_arrs = self.read_array(&field.children[0], batch_id).await?;

        Ok(Arc::new(ListArray::new(value_arrs, &offset_arr)?))
    }

    /// Read an array of the batch.
    #[async_recursion]
    async fn read_array(&self, field: &Field, batch_id: i32) -> Result<ArrayRef> {
        let data_type = field.data_type();

        use DataType::*;

        if is_fixed_stride(&data_type) {
            self.read_fixed_stride_array(field, batch_id).await
        } else {
            match data_type {
                Utf8 | LargeUtf8 | Binary | LargeBinary => {
                    self.read_binary_array(field, batch_id).await
                }
                Struct(_) => self.read_struct_array(field, batch_id).await,
                Dictionary(_, _) => self.read_dictionary_array(field, batch_id).await,
                List(_) => self.read_list_array(field, batch_id).await,
                _ => {
                    unimplemented!("{}", format!("No support for {data_type} yet"));
                }
            }
        }
    }
}
