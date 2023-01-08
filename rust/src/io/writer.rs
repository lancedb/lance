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

use arrow_array::cast::as_struct_array;
use arrow_array::{Array, ArrayRef, RecordBatch, StructArray};
use arrow_schema::DataType;
use async_recursion::async_recursion;
use tokio::io::AsyncWriteExt;

use crate::arrow::*;
use crate::datatypes::{is_fixed_stride, Field, Schema};
use crate::encodings::binary::BinaryEncoder;
use crate::encodings::Encoder;
use crate::encodings::{plain::PlainEncoder, Encoding};
use crate::format::{Manifest, Metadata, PageInfo, PageTable, MAGIC, MAJOR_VERSION, MINOR_VERSION};
use crate::io::object_writer::ObjectWriter;
use crate::{Error, Result};

/// FileWriter writes Arrow Table to a file.
pub struct FileWriter<'a> {
    object_writer: ObjectWriter,
    schema: &'a Schema,
    batch_id: i32,
    page_table: PageTable,
    metadata: Metadata,
}

impl<'a> FileWriter<'a> {
    pub fn new(object_writer: ObjectWriter, schema: &'a Schema) -> Self {
        Self {
            object_writer,
            schema,
            batch_id: 0,
            page_table: PageTable::default(),
            metadata: Metadata::default(),
        }
    }

    /// Write a [RecordBatch] to the open file.
    ///
    /// Returns [Err] if the schema does not match with the batch.
    pub async fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        for field in &self.schema.fields {
            let column_id = batch.schema().index_of(&field.name)?;
            let array = batch.column(column_id);
            self.write_array(field, array).await?;
        }
        self.metadata.push_batch_length(batch.num_rows() as i32);
        self.batch_id += 1;
        Ok(())
    }

    // pub async fn abort(&self) -> Result<()> {
    //     // self.object_writer.
    //     Ok(())
    // }

    pub async fn finish(&mut self) -> Result<()> {
        self.write_footer().await?;
        self.object_writer.shutdown().await
    }

    #[async_recursion]
    async fn write_array(&mut self, field: &Field, array: &ArrayRef) -> Result<()> {
        let data_type = array.data_type();
        if is_fixed_stride(array.data_type()) {
            self.write_fixed_stride_array(field, array).await?;
        } else if matches!(array.data_type(), DataType::Struct(_)) {
            let struct_arr = as_struct_array(array);
            self.write_struct_array(field, struct_arr).await?;
        } else if data_type.is_binary_like() {
            self.write_binary_array(field, array).await?;
        };

        Ok(())
    }

    /// Write fixed size array, including, primtiives, fixed size binary, and fixed size list.
    async fn write_fixed_stride_array(&mut self, field: &Field, array: &ArrayRef) -> Result<()> {
        assert_eq!(field.encoding, Some(Encoding::Plain));
        let mut encoder = PlainEncoder::new(&mut self.object_writer, array.data_type());
        let pos = encoder.encode(array).await?;
        let page_info = PageInfo::new(pos, array.len());
        self.page_table.set(field.id, self.batch_id, page_info);
        Ok(())
    }

    /// Write var-length binary arrays.
    async fn write_binary_array(&mut self, field: &Field, array: &ArrayRef) -> Result<()> {
        assert_eq!(field.encoding, Some(Encoding::VarBinary));
        let mut encoder = BinaryEncoder::new(&mut self.object_writer);
        let pos = encoder.encode(array).await?;
        let page_info = PageInfo::new(pos, array.len());
        self.page_table.set(field.id, self.batch_id, page_info);
        Ok(())
    }

    #[async_recursion]
    async fn write_struct_array(&mut self, field: &Field, array: &StructArray) -> Result<()> {
        assert_eq!(array.num_columns(), field.children.len());
        for child in &field.children {
            if let Some(arr) = array.column_by_name(&child.name) {
                self.write_array(child, arr).await?;
            } else {
                return Err(Error::Schema(format!(
                    "FileWriter: schema mismatch: column {} does not exist in array: {:?}",
                    child.name,
                    array.data_type()
                )));
            }
        }

        Ok(())
    }

    async fn write_footer(&mut self) -> Result<()> {
        // Step 1. write dictionary values.

        // Step 2. Write page table.

        // Step 3. Write manifest.
        let manifest = Manifest::new(self.schema);
        let pos = self.object_writer.write_struct(&manifest).await?;

        // Step 4. Write metadata.
        self.metadata.manifest_position = Some(pos);
        let pos = self.object_writer.write_struct(&self.metadata).await?;

        // Step 5. Write magics.
        self.write_magics(pos).await
    }

    async fn write_magics(&mut self, pos: usize) -> Result<()> {
        self.object_writer.write_i64_le(pos as i64).await?;
        self.object_writer.write_i16_le(MAJOR_VERSION).await?;
        self.object_writer.write_i16_le(MINOR_VERSION).await?;
        self.object_writer.write_all(MAGIC).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    #[tokio::test]
    async fn test_write_file() {}
}
