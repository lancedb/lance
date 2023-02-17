// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;

use arrow_arith::arithmetic::subtract_scalar;
use arrow_array::cast::{as_large_list_array, as_list_array, as_struct_array};
use arrow_array::{Array, ArrayRef, Int32Array, Int64Array, RecordBatch, StructArray};
use arrow_schema::DataType;
use async_recursion::async_recursion;
use object_store::path::Path;

use crate::arrow::*;
use crate::datatypes::{Field, Schema};
use crate::encodings::dictionary::DictionaryEncoder;
use crate::encodings::{binary::BinaryEncoder, plain::PlainEncoder, Encoder, Encoding};
use crate::format::{pb, Index, Manifest, Metadata, PageInfo, PageTable};
use crate::io::object_writer::ObjectWriter;
use crate::{Error, Result};

use super::ObjectStore;

/// Write manifest to an open file.
pub async fn write_manifest(
    writer: &mut ObjectWriter,
    manifest: &mut Manifest,
    indices: Option<Vec<Index>>,
) -> Result<usize> {
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

    // Write indices if presented.
    if let Some(indices) = indices.as_ref() {
        let section: pb::IndexSection = indices.into();
        let pos = writer.write_protobuf(&section).await?;
        manifest.index_section = Some(pos);
    }

    writer.write_struct(manifest).await
}

/// [FileWriter] writes Arrow [RecordBatch] to one Lance file.
///
/// ```ignored
/// use lance::io::FileWriter;
/// use futures::stream::Stream;
///
/// let mut file_writer = FileWriter::new(object_store, &path, &schema);
/// while let Ok(batch) = stream.next().await {
///     file_writer.write(&batch).unwrap();
/// }
/// // Need to close file writer to flush buffer and footer.
/// file_writer.shutdown();
/// ```
pub struct FileWriter<'a> {
    object_writer: ObjectWriter,
    schema: &'a Schema,
    batch_id: i32,
    page_table: PageTable,
    metadata: Metadata,
}

impl<'a> FileWriter<'a> {
    pub async fn try_new(
        object_store: &ObjectStore,
        path: &Path,
        schema: &'a Schema,
    ) -> Result<FileWriter<'a>> {
        let object_writer = object_store.create(path).await?;
        Ok(Self {
            object_writer,
            schema,
            batch_id: 0,
            page_table: PageTable::default(),
            metadata: Metadata::default(),
        })
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

    /// Total records written in this file.
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[async_recursion]
    async fn write_array(&mut self, field: &Field, array: &ArrayRef) -> Result<()> {
        let data_type = array.data_type();
        match data_type {
            DataType::Null => self.write_null_array(field, array).await,
            dt if dt.is_fixed_stride() => self.write_fixed_stride_array(field, array).await,
            dt if dt.is_binary_like() => self.write_binary_array(field, array).await,
            DataType::Dictionary(key_type, _) => {
                self.write_dictionary_arr(field, array, key_type).await
            }
            dt if dt.is_struct() => {
                let struct_arr = as_struct_array(array);
                self.write_struct_array(field, struct_arr).await
            }
            DataType::FixedSizeList(_, _) | DataType::FixedSizeBinary(_) => {
                self.write_fixed_stride_array(field, array).await
            }
            DataType::List(_) => self.write_list_array(field, array).await,
            DataType::LargeList(_) => self.write_large_list_array(field, array).await,
            _ => {
                return Err(Error::Schema(format!(
                    "FileWriter::write: unsupported data type: {data_type}"
                )))
            }
        }
    }

    async fn write_null_array(&mut self, field: &Field, array: &ArrayRef) -> Result<()> {
        let page_info = PageInfo::new(self.object_writer.tell(), array.len());
        self.page_table.set(field.id, self.batch_id, page_info);
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

    async fn write_dictionary_arr(
        &mut self,
        field: &Field,
        array: &ArrayRef,
        key_type: &DataType,
    ) -> Result<()> {
        assert_eq!(field.encoding, Some(Encoding::Dictionary));

        // Write the dictionary keys.
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

    async fn write_list_array(&mut self, field: &Field, array: &ArrayRef) -> Result<()> {
        let list_arr = as_list_array(array);
        let offsets: Int32Array = list_arr.value_offsets().iter().copied().collect();
        assert!(!offsets.is_empty());
        let offsets = Arc::new(subtract_scalar(&offsets, offsets.value(0))?) as ArrayRef;
        self.write_fixed_stride_array(field, &offsets).await?;
        self.write_array(&field.children[0], list_arr.values())
            .await
    }

    async fn write_large_list_array(&mut self, field: &Field, array: &ArrayRef) -> Result<()> {
        let list_arr = as_large_list_array(array);
        let offsets: Int64Array = list_arr.value_offsets().iter().copied().collect();
        assert!(!offsets.is_empty());
        let offsets = Arc::new(subtract_scalar(&offsets, offsets.value(0))?) as ArrayRef;
        self.write_fixed_stride_array(field, &offsets).await?;
        self.write_array(&field.children[0], list_arr.values())
            .await
    }

    async fn write_footer(&mut self) -> Result<()> {
        // Step 1. Write page table.
        let pos = self.page_table.write(&mut self.object_writer).await?;
        self.metadata.page_table_position = pos;

        // Step 2. Write manifest and dictionary values.
        let mut manifest = Manifest::new(self.schema, Arc::new(vec![]));
        let pos = write_manifest(&mut self.object_writer, &mut manifest, None).await?;

        // Step 3. Write metadata.
        self.metadata.manifest_position = Some(pos);
        let pos = self.object_writer.write_struct(&self.metadata).await?;

        // Step 4. Write magics.
        self.object_writer.write_magics(pos).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::{
        types::UInt32Type, BooleanArray, Decimal128Array, Decimal256Array, DictionaryArray,
        DurationMicrosecondArray, DurationMillisecondArray, DurationNanosecondArray,
        DurationSecondArray, FixedSizeBinaryArray, FixedSizeListArray, Float32Array, Int64Array,
        LargeListArray, ListArray, NullArray, StringArray, UInt8Array,
    };
    use arrow_buffer::i256;
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema, TimeUnit};
    use object_store::path::Path;

    use crate::io::{FileReader, ObjectStore};

    #[tokio::test]
    async fn test_write_file() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("null", DataType::Null, true),
            ArrowField::new("bool", DataType::Boolean, true),
            ArrowField::new("i", DataType::Int64, true),
            ArrowField::new("f", DataType::Float32, false),
            ArrowField::new("b", DataType::Utf8, true),
            ArrowField::new("decimal128", DataType::Decimal128(7, 3), false),
            ArrowField::new("decimal256", DataType::Decimal256(7, 3), false),
            ArrowField::new("duration_sec", DataType::Duration(TimeUnit::Second), false),
            ArrowField::new(
                "duration_msec",
                DataType::Duration(TimeUnit::Millisecond),
                false,
            ),
            ArrowField::new(
                "duration_usec",
                DataType::Duration(TimeUnit::Microsecond),
                false,
            ),
            ArrowField::new(
                "duration_nsec",
                DataType::Duration(TimeUnit::Nanosecond),
                false,
            ),
            ArrowField::new(
                "d",
                DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                true,
            ),
            ArrowField::new(
                "fixed_size_list",
                DataType::FixedSizeList(
                    Box::new(ArrowField::new("item", DataType::Float32, true)),
                    16,
                ),
                true,
            ),
            ArrowField::new("fixed_size_binary", DataType::FixedSizeBinary(8), true),
            ArrowField::new(
                "l",
                DataType::List(Box::new(ArrowField::new("item", DataType::Utf8, true))),
                true,
            ),
            ArrowField::new(
                "large_l",
                DataType::LargeList(Box::new(ArrowField::new("item", DataType::Utf8, true))),
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
        let mut schema = Schema::try_from(&arrow_schema).unwrap();

        let dict_vec = (0..100)
            .into_iter()
            .map(|n| ["a", "b", "c"][n % 3])
            .collect::<Vec<_>>();
        let dict_arr: DictionaryArray<UInt32Type> = dict_vec.into_iter().collect();

        let fixed_size_list_arr = FixedSizeListArray::try_new(
            Float32Array::from_iter((0..1600).map(|n| n as f32).collect::<Vec<_>>()),
            16,
        )
        .unwrap();

        let binary_data: [u8; 800] = [123; 800];
        let fixed_size_binary_arr =
            FixedSizeBinaryArray::try_new(&UInt8Array::from_iter(binary_data), 8).unwrap();

        let list_offsets = (0..202).step_by(2).collect();
        let list_values =
            StringArray::from((0..200).map(|n| format!("str-{}", n)).collect::<Vec<_>>());
        let list_arr = ListArray::try_new(list_values, &list_offsets).unwrap();

        let large_list_offsets: Int64Array = (0..202).step_by(2).collect();
        let large_list_values =
            StringArray::from((0..200).map(|n| format!("str-{}", n)).collect::<Vec<_>>());
        let large_list_arr =
            LargeListArray::try_new(large_list_values, &large_list_offsets).unwrap();

        let columns: Vec<ArrayRef> = vec![
            Arc::new(NullArray::new(100)),
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
            Arc::new(
                Decimal128Array::from_iter_values((0..100).into_iter())
                    .with_precision_and_scale(7, 3)
                    .unwrap(),
            ),
            Arc::new(
                Decimal256Array::from_iter_values(
                    (0..100).into_iter().map(|v| i256::from_i128(v as i128)),
                )
                .with_precision_and_scale(7, 3)
                .unwrap(),
            ),
            Arc::new(DurationSecondArray::from_iter_values((0..100).into_iter())),
            Arc::new(DurationMillisecondArray::from_iter_values(
                (0..100).into_iter(),
            )),
            Arc::new(DurationMicrosecondArray::from_iter_values(
                (0..100).into_iter(),
            )),
            Arc::new(DurationNanosecondArray::from_iter_values(
                (0..100).into_iter(),
            )),
            Arc::new(dict_arr),
            Arc::new(fixed_size_list_arr),
            Arc::new(fixed_size_binary_arr),
            Arc::new(list_arr),
            Arc::new(large_list_arr),
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
        schema.set_dictionary(&batch).unwrap();

        let store = ObjectStore::memory();
        let path = Path::from("/foo");
        let mut file_writer = FileWriter::try_new(&store, &path, &schema).await.unwrap();
        file_writer.write(&batch).await.unwrap();
        file_writer.finish().await.unwrap();

        let reader = FileReader::try_new(&store, &path).await.unwrap();
        let actual = reader.read_batch(0, ..).await.unwrap();
        assert_eq!(actual, batch);
    }
}
