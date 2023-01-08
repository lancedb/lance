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

use arrow_array::{ArrayRef, RecordBatch};
use tokio::io::AsyncWriteExt;

use crate::datatypes::{Field, Schema, is_fixed_stride};
use crate::encodings::plain::PlainEncoder;
use crate::io::object_writer::ObjectWriter;
use crate::Result;

/// FileWriter writes Arrow Table to a file.
pub struct FileWriter<'a> {
    object_writer: ObjectWriter,
    schema: &'a Schema,
    batch_id: i32,
}

impl<'a> FileWriter<'a> {
    /// Write a [RecordBatch] to the open file.
    ///
    /// Returns [Err] if the schema does not match with the batch.
    pub async fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        for field in &self.schema.fields {
            let column_id = batch.schema().index_of(&field.name)?;
            let array = batch.column(column_id);
            self.write_array(field, array).await?;
        }
        self.batch_id += 1;
        Ok(())
    }

    // pub async fn abort(&self) -> Result<()> {
    //     // self.object_writer.
    //     Ok(())
    // }

    pub async fn shutdown(&mut self) -> Result<()> {
        Ok(self.object_writer.shutdown().await?)
    }

    async fn write_array(&mut self, field: &Field, array: &ArrayRef) -> Result<()> {
        if is_fixed_stride(array.data_type()) {
            self.write_fixed_stride_array(field, array).await?;
        }

        Ok(())
    }

    /// Write fixed size array, including, primtiives, fixed size binary, and fixed size list.
    async fn write_fixed_stride_array(&mut self, field: &Field, array: &ArrayRef) -> Result<()> {
        let mut encoder = PlainEncoder::new(&mut self.object_writer, array.data_type());
        let pos = encoder.encode(array).await?;
        Ok(())
    }

}
