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

use arrow_array::cast::{as_dictionary_array, as_struct_array};
use arrow_array::{
    types::{
        Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    },
    Array, ArrayRef, RecordBatch, StructArray,
};
use arrow_schema::DataType;
use async_recursion::async_recursion;
use tokio::io::AsyncWriteExt;

use crate::arrow::*;
use crate::datatypes::{Field, Schema};
use crate::encodings::dictionary::DictionaryEncoder;
use crate::encodings::{binary::BinaryEncoder, plain::PlainEncoder, Encoder, Encoding};
use crate::format::{Manifest, Metadata, PageInfo, PageTable, MAGIC, MAJOR_VERSION, MINOR_VERSION};
use crate::io::object_writer::ObjectWriter;
use crate::{Error, Result};

pub async fn write_manifest(writer: &mut ObjectWriter, manifest: &mut Manifest) -> Result<usize> {
    // Write dictionary values.
    let max_field_id = manifest.schema.max_field_id().unwrap_or(-1);
    for field_id in 1..max_field_id + 1 {
        if let Some(field) = manifest.schema.mut_field_by_id(field_id) {
            if field.data_type().is_dictionary() {
                let dict_info = field.dictionary.as_mut().ok_or_else(|| {
                    Error::IO(format!("Lance field {} misses dictionary info", field.name))
                })?;

                let value_arr = dict_info.values.as_ref().ok_or_else(|| {
                    Error::IO(format!(
                        "Lance field {} is dictionary type, but misses the dictionary value array",
                        field.name
                    ))
                })?;

                let data_type = value_arr.data_type();
                let pos = match data_type {
                    dt if dt.is_numeric() => {
                        let mut encoder = PlainEncoder::new(writer, dt);
                        encoder.encode(value_arr).await?
                    }
                    dt if dt.is_binary_like() => {
                        let mut encoder = BinaryEncoder::new(writer);
                        encoder.encode(value_arr).await?
                    }
                    _ => {
                        return Err(Error::IO(format!(
                            "Does not support {} as dictionary value type",
                            value_arr.data_type()
                        )));
                    }
                };
                dict_info.offset = pos;
                dict_info.length = value_arr.len();
            }
        }
    }
    writer.write_struct(manifest).await
}

/// [FileWriter] writes Arrow [RecordBatch] to a file.
pub struct FileWriter<'a> {
    object_writer: ObjectWriter,
    schema: &'a Schema,
    batch_id: i32,
    page_table: PageTable,
    metadata: Metadata,

    // Lazily loaded dictoinary value arrays, <field_id, value_arr>.
    // It is populared during the write() process, and later will be
    // serialized to the manifest.
    dictionary_value_arrs: HashMap<i32, ArrayRef>,
}

impl<'a> FileWriter<'a> {
    pub fn new(object_writer: ObjectWriter, schema: &'a Schema) -> Self {
        Self {
            object_writer,
            schema,
            batch_id: 0,
            page_table: PageTable::default(),
            metadata: Metadata::default(),
            dictionary_value_arrs: HashMap::default(),
        }
    }

    /// Write a [RecordBatch] to the open file.
    ///
    /// Returns [Err] if the schema does not match with the batch.
    pub async fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        for field in self.schema.fields.as_slice() {
            let column_id = batch.schema().index_of(&field.name)?;
            let array = batch.column(column_id);
            self.write_array(field, array).await?;
        }
        self.metadata.push_batch_length(batch.num_rows() as i32);
        self.batch_id += 1;
        Ok(())
    }

    pub async fn finish(&mut self) -> Result<()> {
        self.write_footer().await?;
        self.object_writer.shutdown().await
    }

    #[async_recursion]
    async fn write_array(&mut self, field: &Field, array: &ArrayRef) -> Result<()> {
        let data_type = array.data_type();
        match data_type {
            dt if dt.is_fixed_stride() => self.write_fixed_stride_array(field, array).await,
            dt if dt.is_binary_like() => self.write_binary_array(field, array).await,
            DataType::Dictionary(key_type, _) => {
                self.write_dictionary_arr(field, array, key_type).await
            }
            dt if dt.is_struct() => {
                let struct_arr = as_struct_array(array);
                self.write_struct_array(field, struct_arr).await
            }
            _ => {
                return Err(Error::Schema(format!(
                    "FileWriter::write: unsupported data type: {:?}",
                    data_type
                )))
            }
        }
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

    async fn write_dictionary_arr(
        &mut self,
        field: &Field,
        array: &ArrayRef,
        key_type: &DataType,
    ) -> Result<()> {
        assert_eq!(field.encoding, Some(Encoding::Dictionary));

        if self.batch_id == 0 {
            // Only load value arrays for the first batch.
            assert!(!self.dictionary_value_arrs.contains_key(&field.id));
            use DataType::*;
            let values = match key_type {
                UInt8 => as_dictionary_array::<UInt8Type>(array).values(),
                UInt16 => as_dictionary_array::<UInt16Type>(array).values(),
                UInt32 => as_dictionary_array::<UInt32Type>(array).values(),
                UInt64 => as_dictionary_array::<UInt64Type>(array).values(),
                Int8 => as_dictionary_array::<Int8Type>(array).values(),
                Int16 => as_dictionary_array::<Int16Type>(array).values(),
                Int32 => as_dictionary_array::<Int32Type>(array).values(),
                Int64 => as_dictionary_array::<Int64Type>(array).values(),
                _ => {
                    return Err(Error::Schema(format!(
                        "DictionaryEncoder: unsurpported key type: {:?}",
                        key_type,
                    )))
                }
            };
            self.dictionary_value_arrs.insert(field.id, values.clone());
        };

        // Write data.
        let mut encoder = DictionaryEncoder::new(&mut self.object_writer, key_type);
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
        // Step 1. Write page table.
        let pos = self.page_table.write(&mut self.object_writer).await?;
        self.metadata.page_table_position = pos;

        // Step 2. Write manifest and dictionary values.
        // Populate schema
        let mut schema = self.schema.clone();
        for (field_id, value_arrs) in &self.dictionary_value_arrs {
            let field = schema
                .mut_field_by_id(*field_id)
                .ok_or_else(|| Error::IO("Schema mismatch".to_string()))?;
            field.set_dictionary_values(value_arrs);
        }
        // Write dictionary values to disk.
        let mut manifest = Manifest::new(&schema);
        let pos = write_manifest(&mut self.object_writer, &mut manifest).await?;

        // Step 3. Write metadata.
        self.metadata.manifest_position = Some(pos);
        let pos = self.object_writer.write_struct(&self.metadata).await?;

        // Step 4. Write magics.
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
    use super::*;

    use std::sync::Arc;

    use arrow_array::{
        types::UInt32Type, BooleanArray, DictionaryArray, Float32Array, Int64Array, StringArray,
    };
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use object_store::path::Path;

    use crate::io::{FileReader, ObjectStore};

    #[tokio::test]
    async fn test_write_file() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("bool", DataType::Boolean, true),
            ArrowField::new("i", DataType::Int64, true),
            ArrowField::new("f", DataType::Float32, false),
            ArrowField::new("b", DataType::Utf8, true),
            ArrowField::new(
                "d",
                DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                true,
            ),
            ArrowField::new(
                "s",
                DataType::Struct(vec![
                    ArrowField::new("si", DataType::Int64, true),
                    ArrowField::new("sb", DataType::Utf8, true),
                ]),
                true,
            ),
        ]);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        let store = ObjectStore::memory();
        let path = Path::from("/foo");
        let writer = store.create(&path).await.unwrap();

        let dict_vec = (0..100)
            .into_iter()
            .map(|n| ["a", "b", "c"][n % 3])
            .collect::<Vec<_>>();
        let dict_arr: DictionaryArray<UInt32Type> = dict_vec.into_iter().collect();

        let columns: Vec<ArrayRef> = vec![
            Arc::new(BooleanArray::from_iter(
                (0..100).map(|f| Some(f % 3 == 0)).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from_iter((0..100).collect::<Vec<_>>())),
            Arc::new(Float32Array::from_iter(
                (0..100).map(|n| n as f32).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                (0..100).map(|n| n.to_string()).collect::<Vec<_>>(),
            )),
            Arc::new(dict_arr),
            Arc::new(StructArray::from(vec![
                (
                    ArrowField::new("si", DataType::Int64, true),
                    Arc::new(Int64Array::from_iter((100..200).collect::<Vec<_>>())) as ArrayRef,
                ),
                (
                    ArrowField::new("sb", DataType::Utf8, true),
                    Arc::new(StringArray::from(
                        (0..100).map(|n| n.to_string()).collect::<Vec<_>>(),
                    )) as ArrayRef,
                ),
            ])),
        ];
        let batch = RecordBatch::try_new(Arc::new(arrow_schema), columns).unwrap();
        let mut file_writer = FileWriter::new(writer, &schema);
        file_writer.write(&batch).await.unwrap();
        file_writer.finish().await.unwrap();

        let reader = FileReader::new(&store, &path, None).await.unwrap();
        let actual = reader.read_batch(0).await.unwrap();
        assert_eq!(actual, batch);
    }
}
