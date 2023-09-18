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

use arrow_array::builder::{ArrayBuilder, PrimitiveBuilder};
use arrow_array::cast::{as_large_list_array, as_list_array, as_struct_array};
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, RecordBatch, StructArray};
use arrow_buffer::ArrowNativeType;
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
    for field_id in 0..max_field_id + 1 {
        if let Some(field) = manifest.schema.mut_field_by_id(field_id) {
            if field.data_type().is_dictionary() {
                let dict_info = field.dictionary.as_mut().ok_or_else(|| Error::IO {
                    message: format!("Lance field {} misses dictionary info", field.name),
                })?;

                let value_arr = dict_info.values.as_ref().ok_or_else(|| Error::IO {
                    message: format!(
                        "Lance field {} is dictionary type, but misses the dictionary value array",
                        field.name
                    ),
                })?;

                let data_type = value_arr.data_type();
                let pos = match data_type {
                    dt if dt.is_numeric() => {
                        let mut encoder = PlainEncoder::new(writer, dt);
                        encoder.encode(&[value_arr]).await?
                    }
                    dt if dt.is_binary_like() => {
                        let mut encoder = BinaryEncoder::new(writer);
                        encoder.encode(&[value_arr]).await?
                    }
                    _ => {
                        return Err(Error::IO {
                            message: format!(
                                "Does not support {} as dictionary value type",
                                value_arr.data_type()
                            ),
                        });
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
pub struct FileWriter {
    object_writer: ObjectWriter,
    schema: Schema,
    batch_id: i32,
    page_table: PageTable,
    metadata: Metadata,
}

impl FileWriter {
    pub async fn try_new(object_store: &ObjectStore, path: &Path, schema: Schema) -> Result<Self> {
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
    /// All RecordBatch will be treated as one RecordBatch on disk
    ///
    /// Returns [Err] if the schema does not match with the batch.
    pub async fn write(&mut self, batches: &[RecordBatch]) -> Result<()> {
        // Copy a list of fields to avoid borrow checker error.
        let fields = self.schema.fields.clone();
        for field in fields.iter() {
            let arrs = batches
                .iter()
                .map(|batch| {
                    batch.column_by_name(&field.name).ok_or_else(|| Error::IO {
                        message: format!("FileWriter::write: Field {} not found", field.name),
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            self.write_array(field, &arrs).await?;
        }
        let batch_length = batches.iter().map(|b| b.num_rows() as i32).sum();
        self.metadata.push_batch_length(batch_length);
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

    /// Returns the in-flight multipart ID.
    pub fn multipart_id(&self) -> &str {
        &self.object_writer.multipart_id
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[async_recursion]
    async fn write_array(&mut self, field: &Field, arrs: &[&ArrayRef]) -> Result<()> {
        assert!(!arrs.is_empty());
        let data_type = arrs[0].data_type();
        let arrs_ref = arrs.iter().map(|a| a.as_ref()).collect::<Vec<_>>();

        match data_type {
            DataType::Null => self.write_null_array(field, arrs_ref.as_slice()).await,
            dt if dt.is_fixed_stride() => {
                self.write_fixed_stride_array(field, arrs_ref.as_slice())
                    .await
            }
            dt if dt.is_binary_like() => self.write_binary_array(field, arrs_ref.as_slice()).await,
            DataType::Dictionary(key_type, _) => {
                self.write_dictionary_arr(field, arrs_ref.as_slice(), key_type)
                    .await
            }
            dt if dt.is_struct() => {
                let struct_arrays = arrs.iter().map(|a| as_struct_array(a)).collect::<Vec<_>>();
                self.write_struct_array(field, struct_arrays.as_slice())
                    .await
            }
            DataType::FixedSizeList(_, _) | DataType::FixedSizeBinary(_) => {
                self.write_fixed_stride_array(field, arrs_ref.as_slice())
                    .await
            }
            DataType::List(_) => self.write_list_array(field, arrs_ref.as_slice()).await,
            DataType::LargeList(_) => {
                self.write_large_list_array(field, arrs_ref.as_slice())
                    .await
            }
            _ => Err(Error::Schema {
                message: format!("FileWriter::write: unsupported data type: {data_type}"),
            }),
        }
    }

    async fn write_null_array(&mut self, field: &Field, arrs: &[&dyn Array]) -> Result<()> {
        let arrs_length: i32 = arrs.iter().map(|a| a.len() as i32).sum();
        let page_info = PageInfo::new(self.object_writer.tell(), arrs_length as usize);
        self.page_table.set(field.id, self.batch_id, page_info);
        Ok(())
    }

    /// Write fixed size array, including, primtiives, fixed size binary, and fixed size list.
    async fn write_fixed_stride_array(&mut self, field: &Field, arrs: &[&dyn Array]) -> Result<()> {
        assert_eq!(field.encoding, Some(Encoding::Plain));
        assert!(!arrs.is_empty());
        let data_type = arrs[0].data_type();

        let mut encoder = PlainEncoder::new(&mut self.object_writer, data_type);
        let pos = encoder.encode(arrs).await?;
        let arrs_length: i32 = arrs.iter().map(|a| a.len() as i32).sum();
        let page_info = PageInfo::new(pos, arrs_length as usize);
        self.page_table.set(field.id, self.batch_id, page_info);
        Ok(())
    }

    /// Write var-length binary arrays.
    async fn write_binary_array(&mut self, field: &Field, arrs: &[&dyn Array]) -> Result<()> {
        assert_eq!(field.encoding, Some(Encoding::VarBinary));
        let mut encoder = BinaryEncoder::new(&mut self.object_writer);
        let pos = encoder.encode(arrs).await?;
        let arrs_length: i32 = arrs.iter().map(|a| a.len() as i32).sum();
        let page_info = PageInfo::new(pos, arrs_length as usize);
        self.page_table.set(field.id, self.batch_id, page_info);
        Ok(())
    }

    async fn write_dictionary_arr(
        &mut self,
        field: &Field,
        arrs: &[&dyn Array],
        key_type: &DataType,
    ) -> Result<()> {
        assert_eq!(field.encoding, Some(Encoding::Dictionary));

        // Write the dictionary keys.
        let mut encoder = DictionaryEncoder::new(&mut self.object_writer, key_type);
        let pos = encoder.encode(arrs).await?;
        let arrs_length: i32 = arrs.iter().map(|a| a.len() as i32).sum();
        let page_info = PageInfo::new(pos, arrs_length as usize);
        self.page_table.set(field.id, self.batch_id, page_info);
        Ok(())
    }

    #[async_recursion]
    async fn write_struct_array(&mut self, field: &Field, arrays: &[&StructArray]) -> Result<()> {
        arrays
            .iter()
            .for_each(|a| assert_eq!(a.num_columns(), field.children.len()));

        for child in &field.children {
            let mut arrs: Vec<&ArrayRef> = Vec::new();
            for struct_array in arrays {
                let arr = struct_array
                    .column_by_name(&child.name)
                    .ok_or(Error::Schema {
                        message: format!(
                            "FileWriter: schema mismatch: column {} does not exist in array: {:?}",
                            child.name,
                            struct_array.data_type()
                        ),
                    })?;
                arrs.push(arr);
            }
            self.write_array(child, arrs.as_slice()).await?;
        }
        Ok(())
    }

    async fn write_list_array(&mut self, field: &Field, arrs: &[&dyn Array]) -> Result<()> {
        let capacity: usize = arrs.iter().map(|a| a.len()).sum();
        let mut list_arrs: Vec<ArrayRef> = Vec::new();
        let mut pos_builder: PrimitiveBuilder<Int32Type> =
            PrimitiveBuilder::with_capacity(capacity);

        let mut last_offset: usize = 0;
        pos_builder.append_value(last_offset as i32);
        for array in arrs.iter() {
            let list_arr = as_list_array(*array);
            let offsets = list_arr.value_offsets();

            assert!(!offsets.is_empty());
            let start_offset = offsets[0].as_usize();
            let end_offset = offsets[offsets.len() - 1].as_usize();

            let list_values = list_arr.values();
            let sliced_values = list_values.slice(start_offset, end_offset - start_offset);
            list_arrs.push(sliced_values);

            offsets
                .iter()
                .skip(1)
                .map(|b| b.as_usize() - start_offset + last_offset)
                .for_each(|o| pos_builder.append_value(o as i32));
            last_offset = pos_builder.values_slice()[pos_builder.len() - 1_usize] as usize;
        }

        let positions: &dyn Array = &pos_builder.finish();
        self.write_fixed_stride_array(field, &[positions]).await?;
        let arrs = list_arrs.iter().collect::<Vec<_>>();
        self.write_array(&field.children[0], arrs.as_slice()).await
    }

    async fn write_large_list_array(&mut self, field: &Field, arrs: &[&dyn Array]) -> Result<()> {
        let capacity: usize = arrs.iter().map(|a| a.len()).sum();
        let mut list_arrs: Vec<ArrayRef> = Vec::new();
        let mut pos_builder: PrimitiveBuilder<Int64Type> =
            PrimitiveBuilder::with_capacity(capacity);

        let mut last_offset: usize = 0;
        pos_builder.append_value(last_offset as i64);
        for array in arrs.iter() {
            let list_arr = as_large_list_array(*array);
            let offsets = list_arr.value_offsets();

            assert!(!offsets.is_empty());
            let start_offset = offsets[0].as_usize();
            let end_offset = offsets[offsets.len() - 1].as_usize();

            let sliced_values = list_arr
                .values()
                .slice(start_offset, end_offset - start_offset);
            list_arrs.push(sliced_values);

            offsets
                .iter()
                .skip(1)
                .map(|b| b.as_usize() - start_offset + last_offset)
                .for_each(|o| pos_builder.append_value(o as i64));
            last_offset = pos_builder.values_slice()[pos_builder.len() - 1_usize] as usize;
        }

        let positions: &dyn Array = &pos_builder.finish();
        self.write_fixed_stride_array(field, &[positions]).await?;
        let arrs = list_arrs.iter().collect::<Vec<_>>();
        self.write_array(&field.children[0], arrs.as_slice()).await
    }

    async fn write_footer(&mut self) -> Result<()> {
        // Step 1. Write page table.
        let pos = self.page_table.write(&mut self.object_writer).await?;
        self.metadata.page_table_position = pos;

        // Step 2. Write manifest and dictionary values.
        let mut manifest = Manifest::new(&self.schema, Arc::new(vec![]));
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
        NullArray, StringArray, TimestampMicrosecondArray, TimestampSecondArray, UInt8Array,
    };
    use arrow_buffer::i256;
    use arrow_schema::{
        DataType, Field as ArrowField, Fields as ArrowFields, Schema as ArrowSchema, TimeUnit,
    };
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
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    16,
                ),
                true,
            ),
            ArrowField::new("fixed_size_binary", DataType::FixedSizeBinary(8), true),
            ArrowField::new(
                "l",
                DataType::List(Arc::new(ArrowField::new("item", DataType::Utf8, true))),
                true,
            ),
            ArrowField::new(
                "large_l",
                DataType::LargeList(Arc::new(ArrowField::new("item", DataType::Utf8, true))),
                true,
            ),
            ArrowField::new(
                "l_dict",
                DataType::List(Arc::new(ArrowField::new(
                    "item",
                    DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                    true,
                ))),
                true,
            ),
            ArrowField::new(
                "large_l_dict",
                DataType::LargeList(Arc::new(ArrowField::new(
                    "item",
                    DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                    true,
                ))),
                true,
            ),
            ArrowField::new(
                "s",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("si", DataType::Int64, true),
                    ArrowField::new("sb", DataType::Utf8, true),
                ])),
                true,
            ),
        ]);
        let mut schema = Schema::try_from(&arrow_schema).unwrap();

        let dict_vec = (0..100).map(|n| ["a", "b", "c"][n % 3]).collect::<Vec<_>>();
        let dict_arr: DictionaryArray<UInt32Type> = dict_vec.into_iter().collect();

        let fixed_size_list_arr = FixedSizeListArray::try_new_from_values(
            Float32Array::from_iter((0..1600).map(|n| n as f32).collect::<Vec<_>>()),
            16,
        )
        .unwrap();

        let binary_data: [u8; 800] = [123; 800];
        let fixed_size_binary_arr =
            FixedSizeBinaryArray::try_new_from_values(&UInt8Array::from_iter(binary_data), 8)
                .unwrap();

        let list_offsets = (0..202).step_by(2).collect();
        let list_values =
            StringArray::from((0..200).map(|n| format!("str-{}", n)).collect::<Vec<_>>());
        let list_arr: arrow_array::GenericListArray<i32> =
            try_new_generic_list_array(list_values, &list_offsets).unwrap();

        let large_list_offsets: Int64Array = (0..202).step_by(2).collect();
        let large_list_values =
            StringArray::from((0..200).map(|n| format!("str-{}", n)).collect::<Vec<_>>());
        let large_list_arr: arrow_array::GenericListArray<i64> =
            try_new_generic_list_array(large_list_values, &large_list_offsets).unwrap();

        let list_dict_offsets = (0..202).step_by(2).collect();
        let list_dict_vec = (0..200).map(|n| ["a", "b", "c"][n % 3]).collect::<Vec<_>>();
        let list_dict_arr: DictionaryArray<UInt32Type> = list_dict_vec.into_iter().collect();
        let list_dict_arr: arrow_array::GenericListArray<i32> =
            try_new_generic_list_array(list_dict_arr, &list_dict_offsets).unwrap();

        let large_list_dict_offsets: Int64Array = (0..202).step_by(2).collect();
        let large_list_dict_vec = (0..200).map(|n| ["a", "b", "c"][n % 3]).collect::<Vec<_>>();
        let large_list_dict_arr: DictionaryArray<UInt32Type> =
            large_list_dict_vec.into_iter().collect();
        let large_list_dict_arr: arrow_array::GenericListArray<i64> =
            try_new_generic_list_array(large_list_dict_arr, &large_list_dict_offsets).unwrap();

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
                Decimal128Array::from_iter_values(0..100)
                    .with_precision_and_scale(7, 3)
                    .unwrap(),
            ),
            Arc::new(
                Decimal256Array::from_iter_values((0..100).map(|v| i256::from_i128(v as i128)))
                    .with_precision_and_scale(7, 3)
                    .unwrap(),
            ),
            Arc::new(DurationSecondArray::from_iter_values(0..100)),
            Arc::new(DurationMillisecondArray::from_iter_values(0..100)),
            Arc::new(DurationMicrosecondArray::from_iter_values(0..100)),
            Arc::new(DurationNanosecondArray::from_iter_values(0..100)),
            Arc::new(dict_arr),
            Arc::new(fixed_size_list_arr),
            Arc::new(fixed_size_binary_arr),
            Arc::new(list_arr),
            Arc::new(large_list_arr),
            Arc::new(list_dict_arr),
            Arc::new(large_list_dict_arr),
            Arc::new(StructArray::from(vec![
                (
                    Arc::new(ArrowField::new("si", DataType::Int64, true)),
                    Arc::new(Int64Array::from_iter((100..200).collect::<Vec<_>>())) as ArrayRef,
                ),
                (
                    Arc::new(ArrowField::new("sb", DataType::Utf8, true)),
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
        let mut file_writer = FileWriter::try_new(&store, &path, schema).await.unwrap();
        file_writer.write(&[batch.clone()]).await.unwrap();
        file_writer.finish().await.unwrap();

        let reader = FileReader::try_new(&store, &path).await.unwrap();
        let actual = reader.read_batch(0, .., reader.schema()).await.unwrap();
        assert_eq!(actual, batch);
    }

    #[tokio::test]
    async fn test_dictionary_first_element_file() {
        let arrow_schema = ArrowSchema::new(vec![ArrowField::new(
            "d",
            DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
            true,
        )]);
        let mut schema = Schema::try_from(&arrow_schema).unwrap();

        let dict_vec = (0..100).map(|n| ["a", "b", "c"][n % 3]).collect::<Vec<_>>();
        let dict_arr: DictionaryArray<UInt32Type> = dict_vec.into_iter().collect();

        let columns: Vec<ArrayRef> = vec![Arc::new(dict_arr)];
        let batch = RecordBatch::try_new(Arc::new(arrow_schema), columns).unwrap();
        schema.set_dictionary(&batch).unwrap();

        let store = ObjectStore::memory();
        let path = Path::from("/foo");
        let mut file_writer = FileWriter::try_new(&store, &path, schema).await.unwrap();
        file_writer.write(&[batch.clone()]).await.unwrap();
        file_writer.finish().await.unwrap();

        let reader = FileReader::try_new(&store, &path).await.unwrap();
        let actual = reader.read_batch(0, .., reader.schema()).await.unwrap();
        assert_eq!(actual, batch);
    }

    #[tokio::test]
    async fn test_write_temporal_types() {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new(
                "ts_notz",
                DataType::Timestamp(TimeUnit::Second, None),
                false,
            ),
            ArrowField::new(
                "ts_tz",
                DataType::Timestamp(TimeUnit::Microsecond, Some("America/Los_Angeles".into())),
                false,
            ),
        ]));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(TimestampSecondArray::from(vec![11111111, 22222222])),
            Arc::new(
                TimestampMicrosecondArray::from(vec![3333333, 4444444])
                    .with_timezone("America/Los_Angeles"),
            ),
        ];
        let batch = RecordBatch::try_new(arrow_schema.clone(), columns).unwrap();

        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();
        let store = ObjectStore::memory();
        let path = Path::from("/foo");
        let mut file_writer = FileWriter::try_new(&store, &path, schema).await.unwrap();
        file_writer.write(&[batch.clone()]).await.unwrap();
        file_writer.finish().await.unwrap();

        let reader = FileReader::try_new(&store, &path).await.unwrap();
        let actual = reader.read_batch(0, .., reader.schema()).await.unwrap();
        assert_eq!(actual, batch);
    }
}
