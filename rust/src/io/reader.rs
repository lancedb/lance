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

//! Lance Data File Reader

// Standard
use std::ops::Range;
use std::sync::Arc;

use arrow_arith::arithmetic::subtract_scalar;
use arrow_array::cast::as_primitive_array;
use arrow_array::{
    ArrayRef, Int64Array, LargeListArray, ListArray, NullArray, RecordBatch, StructArray,
    UInt64Array,
};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow_select::concat::concat_batches;
use async_recursion::async_recursion;
use byteorder::{ByteOrder, LittleEndian};
use futures::stream::{self, Stream, TryStreamExt};
use futures::StreamExt;
use object_store::path::Path;
use prost::Message;

use super::ReadBatchParams;
use crate::arrow::*;
use crate::encodings::{dictionary::DictionaryDecoder, AsyncIndex};
use crate::error::{Error, Result};
use crate::format::Manifest;
use crate::format::{pb, Metadata, PageTable};
use crate::io::object_reader::{read_fixed_stride_array, read_struct, ObjectReader};
use crate::io::{read_metadata_offset, read_struct_from_buf};
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
    assert!(file_size - manifest_pos <= buf.len());
    let proto =
        pb::Manifest::decode(&buf[buf.len() - (file_size - manifest_pos) + 4..buf.len() - 16])?;
    Ok(Manifest::from(proto))
}

/// Compute row id from `fragment_id` and the `offset` of the row in the fragment.
fn compute_row_id(fragment_id: u64, offset: i32) -> u64 {
    (fragment_id << 32) + offset as u64
}

/// Lance File Reader.
///
/// It reads arrow data from one data file.
pub struct FileReader<'a> {
    object_reader: Box<dyn ObjectReader + 'a>,
    metadata: Metadata,
    page_table: PageTable,
    projection: Option<Schema>,

    /// The id of the fragment which this file belong to.
    /// For simple file access, this can just be zero.
    fragment_id: u64,

    /// If set true, returns the row ID from the dataset alongside with the
    /// actual data.
    with_row_id: bool,
}

impl<'a> FileReader<'a> {
    /// Open file reader
    pub(crate) async fn try_new_with_fragment(
        object_store: &'a ObjectStore,
        path: &Path,
        fragment_id: u64,
        manifest: Option<&Manifest>,
    ) -> Result<FileReader<'a>> {
        let object_reader = object_store.open(path).await?;

        let file_size = object_reader.size().await?;
        let begin = if file_size < object_store.prefetch_size() {
            0
        } else {
            file_size - object_store.prefetch_size()
        };
        let tail_bytes = object_reader.get_range(begin..file_size).await?;
        let metadata_pos = read_metadata_offset(&tail_bytes)?;

        let metadata: Metadata = if metadata_pos < file_size - tail_bytes.len() {
            // We have not read the metadata bytes yet.
            read_struct(object_reader.as_ref(), metadata_pos).await?
        } else {
            let offset = tail_bytes.len() - (file_size - metadata_pos);
            read_struct_from_buf(&tail_bytes.slice(offset..))?
        };

        let (projection, num_columns) = if let Some(m) = manifest {
            (m.schema.clone(), m.schema.max_field_id().unwrap() + 1)
        } else {
            let mut m: Manifest =
                read_struct(object_reader.as_ref(), metadata.manifest_position.unwrap()).await?;
            m.schema.load_dictionary(object_reader.as_ref()).await?;
            (m.schema.clone(), m.schema.max_field_id().unwrap() + 1)
        };
        let page_table = PageTable::load(
            object_reader.as_ref(),
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
            fragment_id,
            with_row_id: false,
        })
    }

    /// Open one Lance data file for read.
    pub async fn try_new(object_store: &'a ObjectStore, path: &Path) -> Result<FileReader<'a>> {
        Self::try_new_with_fragment(object_store, path, 0, None).await
    }

    /// Set the projection [Schema].
    pub fn set_projection(&mut self, schema: Schema) {
        self.projection = Some(schema)
    }

    /// Instruct the FileReader to return meta row id column.
    pub(crate) fn with_row_id(&mut self, v: bool) -> &mut Self {
        self.with_row_id = v;
        self
    }

    /// Schema of the returning RecordBatch.
    pub fn schema(&self) -> &Schema {
        self.projection.as_ref().unwrap()
    }

    pub fn num_batches(&self) -> usize {
        self.metadata.num_batches()
    }

    /// Get the number of rows in this batch
    pub fn num_rows_in_batch(&self, batch_id: i32) -> usize {
        self.metadata.get_batch_length(batch_id).unwrap_or_default() as usize
    }

    /// Count the number of rows in this file.
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    /// Read a batch of data from the file.
    ///
    /// The schema of the returned [RecordBatch] is set by [`FileReader::schema()`].
    pub(crate) async fn read_batch(
        &self,
        batch_id: i32,
        params: impl Into<ReadBatchParams>,
    ) -> Result<RecordBatch> {
        let schema = self.projection.as_ref().unwrap();
        read_batch(self, &params.into(), schema, batch_id, self.with_row_id).await
    }

    /// Take by records by indices within the file.
    ///
    /// The indices must be sorted.
    pub async fn take(&self, indices: &[u32]) -> Result<RecordBatch> {
        let indices_in_batches = self.metadata.group_indices_to_batches(indices);
        let batches = stream::iter(indices_in_batches)
            .map(|batch| async move {
                self.read_batch(batch.batch_id, batch.offsets.as_slice())
                    .await
            })
            .buffered(8)
            .try_collect::<Vec<_>>()
            .await?;
        let schema = Arc::new(ArrowSchema::from(self.schema()));
        Ok(concat_batches(&schema, &batches)?)
    }

    /// Convert this [`FileReader`] into a [Stream] / [AsyncIterator](std::async_iter::AsyncIterator).
    ///
    /// Currently, it only does batch based scan.
    /// Will add support for scanning with batch size later.
    ///
    // TODO: use IntoStream trait?
    pub fn into_stream(&self) -> impl Stream<Item = Result<RecordBatch>> + '_ {
        let num_batches = self.num_batches() as i32;

        // Deref a bunch.
        let schema = self.schema();
        let with_row_id = self.with_row_id;

        stream::unfold(0_i32, move |batch_id| async move {
            let num_batches = num_batches;
            if batch_id < num_batches {
                let batch = read_batch(
                    self,
                    &ReadBatchParams::RangeFull,
                    schema,
                    batch_id,
                    with_row_id,
                )
                .await;
                Some((batch, batch_id + 1))
            } else {
                None
            }
        })
    }
}

/// Read a batch.
async fn read_batch(
    reader: &FileReader<'_>,
    params: &ReadBatchParams,
    schema: &Schema,
    batch_id: i32,
    with_row_id: bool,
) -> Result<RecordBatch> {
    let arrs = stream::iter(&schema.fields)
        .then(|f| async move { read_array(reader, f, batch_id, params).await })
        .try_collect::<Vec<_>>()
        .await?;
    let mut batch = RecordBatch::try_new(Arc::new(schema.into()), arrs)?;
    if with_row_id {
        let ids_in_batch: Vec<i32> = match params {
            ReadBatchParams::Indices(indices) => {
                indices.values().iter().map(|v| *v as i32).collect()
            }
            ReadBatchParams::Range(r) => r.clone().map(|v| v as i32).collect(),
            ReadBatchParams::RangeFull => (0..batch.num_rows() as i32).collect(),
            ReadBatchParams::RangeTo(r) => (0..r.end).map(|v| v as i32).collect(),
            ReadBatchParams::RangeFrom(r) => (r.start..r.start + batch.num_rows())
                .map(|v| v as i32)
                .collect(),
        };
        let batch_offset = reader
            .metadata
            .get_offset(batch_id)
            .ok_or_else(|| Error::IO(format!("batch {batch_id} does not exist")))?;
        let row_id_arr = Arc::new(UInt64Array::from_iter_values(
            ids_in_batch
                .iter()
                .map(|o| compute_row_id(reader.fragment_id, *o + batch_offset)),
        ));
        batch = batch.try_with_column(
            ArrowField::new("_rowid", DataType::UInt64, false),
            row_id_arr,
        )?;
    }
    Ok(batch)
}

#[async_recursion]
async fn read_array(
    reader: &FileReader<'_>,
    field: &Field,
    batch_id: i32,
    params: &ReadBatchParams,
) -> Result<ArrayRef> {
    let data_type = field.data_type();

    use DataType::*;

    if data_type.is_fixed_stride() {
        _read_fixed_stride_array(reader, field, batch_id, params).await
    } else {
        match data_type {
            Null => read_null_array(reader, field, batch_id, params),
            Utf8 | LargeUtf8 | Binary | LargeBinary => {
                read_binary_array(reader, field, batch_id, params).await
            }
            Struct(_) => read_struct_array(reader, field, batch_id, params).await,
            Dictionary(_, _) => read_dictionary_array(reader, field, batch_id, params).await,
            List(_) => read_list_array(reader, field, batch_id, params).await,
            LargeList(_) => read_large_list_array(reader, field, batch_id, params).await,
            _ => {
                unimplemented!("{}", format!("No support for {data_type} yet"));
            }
        }
    }
}

fn get_page_info<'a>(
    page_table: &'a PageTable,
    field: &'a Field,
    batch_id: i32,
) -> Result<&'a PageInfo> {
    page_table.get(field.id, batch_id).ok_or_else(|| {
        Error::IO(format!(
            "No page info found for field: {}, field_id={} batch={}",
            field.name, field.id, batch_id
        ))
    })
}

/// Read primitive array for batch `batch_idx`.
async fn _read_fixed_stride_array(
    reader: &FileReader<'_>,
    field: &Field,
    batch_id: i32,
    params: &ReadBatchParams,
) -> Result<ArrayRef> {
    let page_info = get_page_info(&reader.page_table, field, batch_id)?;

    read_fixed_stride_array(
        reader.object_reader.as_ref(),
        &field.data_type(),
        page_info.position,
        page_info.length,
        params.clone(),
    )
    .await
}

fn read_null_array(
    reader: &FileReader<'_>,
    field: &Field,
    batch_id: i32,
    params: &ReadBatchParams,
) -> Result<ArrayRef> {
    let page_info = get_page_info(&reader.page_table, field, batch_id)?;

    let length_output = match params {
        ReadBatchParams::Indices(indices) => {
            if indices.is_empty() {
                0
            } else {
                let idx_max = *indices.values().iter().max().unwrap() as u64;
                if idx_max >= page_info.length.try_into().unwrap() {
                    return Err(Error::IO(format!(
                        "NullArray Reader: request([{}]) out of range: [0..{}]",
                        idx_max, page_info.length
                    )));
                }
                indices.len()
            }
        }
        _ => {
            let (idx_start, idx_end) = match params {
                ReadBatchParams::Range(r) => (r.start, r.end),
                ReadBatchParams::RangeFull => (0, page_info.length),
                ReadBatchParams::RangeTo(r) => (0, r.end),
                ReadBatchParams::RangeFrom(r) => (r.start, page_info.length),
                _ => unreachable!(),
            };
            if idx_end > page_info.length {
                return Err(Error::IO(format!(
                    "NullArray Reader: request([{}..{}]) out of range: [0..{}]",
                    idx_start, idx_end, page_info.length
                )));
            }
            idx_end - idx_start
        }
    };

    Ok(Arc::new(NullArray::new(length_output)))
}

async fn read_binary_array(
    reader: &FileReader<'_>,
    field: &Field,
    batch_id: i32,
    params: &ReadBatchParams,
) -> Result<ArrayRef> {
    let page_info = get_page_info(&reader.page_table, field, batch_id)?;

    use crate::io::object_reader::read_binary_array;
    read_binary_array(
        reader.object_reader.as_ref(),
        &field.data_type(),
        field.nullable,
        page_info.position,
        page_info.length,
        params,
    )
    .await
}

async fn read_dictionary_array(
    reader: &FileReader<'_>,
    field: &Field,
    batch_id: i32,
    params: &ReadBatchParams,
) -> Result<ArrayRef> {
    let page_info = get_page_info(&reader.page_table, field, batch_id)?;
    let data_type = field.data_type();
    let decoder = DictionaryDecoder::new(
        reader.object_reader.as_ref(),
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
    decoder.get(params.clone()).await
}

async fn read_struct_array(
    reader: &FileReader<'_>,
    field: &Field,
    batch_id: i32,
    params: &ReadBatchParams,
) -> Result<ArrayRef> {
    // TODO: use tokio to make the reads in parallel.
    let mut sub_arrays = vec![];
    for child in field.children.as_slice() {
        let arr = read_array(reader, child, batch_id, params).await?;
        sub_arrays.push((child.into(), arr));
    }

    Ok(Arc::new(StructArray::from(sub_arrays)))
}

async fn read_list_array(
    reader: &FileReader<'_>,
    field: &Field,
    batch_id: i32,
    params: &ReadBatchParams,
) -> Result<ArrayRef> {
    let page_info = get_page_info(&reader.page_table, field, batch_id)?;

    let position_arr = read_fixed_stride_array(
        reader.object_reader.as_ref(),
        &DataType::Int32,
        page_info.position,
        page_info.length,
        params,
    )
    .await?;
    let positions = as_primitive_array(position_arr.as_ref());
    let start_position = positions.value(0);
    // Compute offsets
    let offset_arr = subtract_scalar(positions, start_position)?;
    let value_arrs = read_array(reader, &field.children[0], batch_id, params).await?;
    Ok(Arc::new(ListArray::try_new(value_arrs, &offset_arr)?))
}

// TODO: merge with [read_list_array]?
async fn read_large_list_array(
    reader: &FileReader<'_>,
    field: &Field,
    batch_id: i32,
    params: &ReadBatchParams,
) -> Result<ArrayRef> {
    let page_info = get_page_info(&reader.page_table, field, batch_id)?;
    let position_arr = read_fixed_stride_array(
        reader.object_reader.as_ref(),
        &DataType::Int64,
        page_info.position,
        page_info.length,
        params,
    )
    .await?;
    let positions: &Int64Array = as_primitive_array(position_arr.as_ref());
    let start_position = positions.value(0);
    // Compute offsets
    let offset_arr = subtract_scalar(positions, start_position)?;
    let value_arrs = read_array(reader, &field.children[0], batch_id, params).await?;

    Ok(Arc::new(LargeListArray::try_new(value_arrs, &offset_arr)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{
        builder::{Int32Builder, ListBuilder, StringBuilder},
        cast::{as_primitive_array, as_string_array, as_struct_array},
        types::UInt8Type,
        DictionaryArray, Float32Array, Int64Array, NullArray, StringArray, StructArray,
        UInt32Array, UInt8Array,
    };
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
    use futures::StreamExt;

    use crate::io::FileWriter;

    #[tokio::test]
    async fn file_reader_into_stream() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int64, true),
            ArrowField::new("f", DataType::Float32, false),
        ]);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        let store = ObjectStore::memory();
        let path = Path::from("/foo");
        // Write 5 batches.
        let mut file_writer = FileWriter::try_new(&store, &path, &schema).await.unwrap();
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from_iter((0..100).collect::<Vec<_>>())),
            Arc::new(Float32Array::from_iter(
                (0..100).map(|n| n as f32).collect::<Vec<_>>(),
            )),
        ];
        let batch = RecordBatch::try_new(Arc::new(arrow_schema.clone()), columns).unwrap();
        for _ in 0..5 {
            file_writer.write(&batch).await.unwrap();
        }
        file_writer.finish().await.unwrap();

        let reader = FileReader::try_new(&store, &path).await.unwrap();
        let stream = reader.into_stream();

        assert_eq!(stream.count().await, 5);

        let stream = reader.into_stream();
        assert!(
            stream
                .map(|f| f.unwrap() == batch)
                .all(|f| async move { f })
                .await
        );
    }

    #[tokio::test]
    async fn read_with_row_id() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int64, true),
            ArrowField::new("f", DataType::Float32, false),
        ]);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        let store = ObjectStore::memory();
        let path = Path::from("/foo");

        // Write 10 batches.
        let mut file_writer = FileWriter::try_new(&store, &path, &schema).await.unwrap();
        for batch_id in 0..10 {
            let value_range = batch_id * 10..batch_id * 10 + 10;
            let columns: Vec<ArrayRef> = vec![
                Arc::new(Int64Array::from_iter(
                    value_range.clone().collect::<Vec<_>>(),
                )),
                Arc::new(Float32Array::from_iter(
                    value_range.map(|n| n as f32).collect::<Vec<_>>(),
                )),
            ];
            let batch = RecordBatch::try_new(Arc::new(arrow_schema.clone()), columns).unwrap();
            file_writer.write(&batch).await.unwrap();
        }
        file_writer.finish().await.unwrap();

        let fragment = 123;
        let mut reader = FileReader::try_new_with_fragment(&store, &path, fragment, None)
            .await
            .unwrap();
        reader.with_row_id(true);

        for b in 0..10 {
            let batch = reader.read_batch(b, ..).await.unwrap();
            let row_ids_col = &batch["_rowid"];
            // Do the same computation as `compute_row_id`.
            let start_pos = (fragment << 32) as u64 + 10 * b as u64;

            assert_eq!(
                &UInt64Array::from_iter_values(start_pos..start_pos + 10),
                as_primitive_array(row_ids_col)
            );
        }
    }

    #[tokio::test]
    async fn test_take() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int64, true),
            ArrowField::new("f", DataType::Float32, false),
            ArrowField::new("s", DataType::Utf8, false),
            ArrowField::new(
                "d",
                DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8)),
                false,
            ),
        ]);
        let mut schema = Schema::try_from(&arrow_schema).unwrap();

        let store = ObjectStore::memory();
        let path = Path::from("/take_test");

        // Write 10 batches.
        let values = StringArray::from_iter_values(["a", "b", "c", "d", "e", "f", "g"]);
        let mut batches = vec![];
        for batch_id in 0..10 {
            let value_range: Range<i64> = batch_id * 10..batch_id * 10 + 10;
            let keys = UInt8Array::from_iter_values(value_range.clone().map(|v| (v % 7) as u8));
            let columns: Vec<ArrayRef> = vec![
                Arc::new(Int64Array::from_iter(
                    value_range.clone().collect::<Vec<_>>(),
                )),
                Arc::new(Float32Array::from_iter(
                    value_range.clone().map(|n| n as f32).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from_iter_values(
                    value_range
                        .clone()
                        .map(|n| format!("str-{}", n))
                        .collect::<Vec<_>>(),
                )),
                Arc::new(DictionaryArray::<UInt8Type>::try_new(&keys, &values).unwrap()),
            ];
            batches.push(RecordBatch::try_new(Arc::new(arrow_schema.clone()), columns).unwrap());
        }
        schema.set_dictionary(&batches[0]).unwrap();

        let mut file_writer = FileWriter::try_new(&store, &path, &schema).await.unwrap();
        for batch in batches.iter() {
            file_writer.write(&batch).await.unwrap();
        }
        file_writer.finish().await.unwrap();

        let reader = FileReader::try_new(&store, &path).await.unwrap();
        let batch = reader.take(&[1, 15, 20, 25, 30, 48, 90]).await.unwrap();
        let dict_keys = UInt8Array::from_iter_values([1, 1, 6, 4, 2, 6, 6]);
        assert_eq!(
            batch,
            RecordBatch::try_new(
                batch.schema(),
                vec![
                    Arc::new(Int64Array::from_iter_values([1, 15, 20, 25, 30, 48, 90])),
                    Arc::new(Float32Array::from_iter_values([
                        1.0, 15.0, 20.0, 25.0, 30.0, 48.0, 90.0
                    ])),
                    Arc::new(StringArray::from_iter_values([
                        "str-1", "str-15", "str-20", "str-25", "str-30", "str-48", "str-90"
                    ])),
                    Arc::new(DictionaryArray::try_new(&dict_keys, &values).unwrap()),
                ]
            )
            .unwrap()
        );
    }

    async fn test_write_null_string_in_struct(field_nullable: bool) {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "parent",
            DataType::Struct(vec![ArrowField::new("str", DataType::Utf8, field_nullable)]),
            true,
        )]));

        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();

        let store = ObjectStore::memory();
        let path = Path::from("/null_strings");

        let string_arr = Arc::new(StringArray::from_iter([Some("a"), Some(""), Some("b")]));
        let struct_arr = Arc::new(StructArray::from(vec![(
            ArrowField::new("str", DataType::Utf8, field_nullable),
            string_arr.clone() as ArrayRef,
        )]));
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![struct_arr]).unwrap();

        let mut file_writer = FileWriter::try_new(&store, &path, &schema).await.unwrap();
        file_writer.write(&batch).await.unwrap();
        file_writer.finish().await.unwrap();

        let reader = FileReader::try_new(&store, &path).await.unwrap();
        let actual_batch = reader.read_batch(0, ..).await.unwrap();

        if field_nullable {
            assert_eq!(
                &StringArray::from_iter(vec![Some("a"), None, Some("b")]),
                as_string_array(
                    as_struct_array(actual_batch.column_by_name("parent").unwrap().as_ref())
                        .column_by_name("str")
                        .unwrap()
                        .as_ref()
                )
            );
        } else {
            assert_eq!(actual_batch, batch);
        }
    }

    #[tokio::test]
    async fn read_nullable_string_in_struct() {
        test_write_null_string_in_struct(true).await;
        test_write_null_string_in_struct(false).await;
    }

    #[tokio::test]
    async fn test_read_struct_of_list_arrays() {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "s",
            DataType::Struct(vec![
                ArrowField::new(
                    "li",
                    DataType::List(Box::new(ArrowField::new("item", DataType::Int32, true))),
                    true,
                ),
                ArrowField::new(
                    "ls",
                    DataType::List(Box::new(ArrowField::new("item", DataType::Utf8, true))),
                    true,
                ),
            ]),
            true,
        )]));

        let store = ObjectStore::memory();
        let path = Path::from("/null_strings");
        let schema: Schema = Schema::try_from(arrow_schema.as_ref()).unwrap();

        let mut li_builder = ListBuilder::new(Int32Builder::new());
        let mut ls_builder = ListBuilder::new(StringBuilder::new());
        for i in 0..10 {
            for j in 0..10 {
                li_builder.values().append_value(i * 10 + j);
                ls_builder
                    .values()
                    .append_value(format!("str-{}", i * 10 + j));
            }
            li_builder.append(true);
            ls_builder.append(true);
        }
        let struct_array = Arc::new(StructArray::from(vec![
            (
                ArrowField::new(
                    "li",
                    DataType::List(Box::new(ArrowField::new("item", DataType::Int32, true))),
                    true,
                ),
                Arc::new(li_builder.finish()) as ArrayRef,
            ),
            (
                ArrowField::new(
                    "ls",
                    DataType::List(Box::new(ArrowField::new("item", DataType::Utf8, true))),
                    true,
                ),
                Arc::new(ls_builder.finish()) as ArrayRef,
            ),
        ]));
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![struct_array]).unwrap();

        let mut file_writer = FileWriter::try_new(&store, &path, &schema).await.unwrap();
        file_writer.write(&batch).await.unwrap();
        file_writer.finish().await.unwrap();

        let reader = FileReader::try_new(&store, &path).await.unwrap();
        let actual_batch = reader.read_batch(0, ..).await.unwrap();

        assert_eq!(batch, actual_batch);
        println!("actual batch: {:?}", actual_batch);
    }

    #[tokio::test]
    async fn test_read_nullable_arrays() {
        use arrow_array::Array;

        // create a record batch with a null array column
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int64, false),
            ArrowField::new("n", DataType::Null, true),
        ]);
        let schema = Schema::try_from(&arrow_schema).unwrap();
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from_iter_values(0..100)),
            Arc::new(NullArray::new(100)),
        ];
        let batch = RecordBatch::try_new(Arc::new(arrow_schema), columns).unwrap();

        // write to a lance file
        let store = ObjectStore::memory();
        let path = Path::from("/takes");
        let mut file_writer = FileWriter::try_new(&store, &path, &schema).await.unwrap();
        file_writer.write(&batch).await.unwrap();
        file_writer.finish().await.unwrap();

        // read the file back
        let reader = FileReader::try_new(&store, &path).await.unwrap();

        async fn read_array_w_params(
            reader: &FileReader<'_>,
            field: &Field,
            params: ReadBatchParams,
        ) -> ArrayRef {
            let arr = read_array(reader, field, 0, &params)
                .await
                .expect("Error reading back the null array from file");
            arr
        }

        let arr = read_array_w_params(&reader, &schema.fields[1], ReadBatchParams::RangeFull).await;
        assert_eq!(100, arr.len());
        assert_eq!(100, arr.null_count());

        let arr =
            read_array_w_params(&reader, &schema.fields[1], ReadBatchParams::Range(10..25)).await;
        assert_eq!(15, arr.len());
        assert_eq!(15, arr.null_count());

        let arr =
            read_array_w_params(&reader, &schema.fields[1], ReadBatchParams::RangeFrom(60..)).await;
        assert_eq!(40, arr.len());
        assert_eq!(40, arr.null_count());

        let arr =
            read_array_w_params(&reader, &schema.fields[1], ReadBatchParams::RangeTo(..25)).await;
        assert_eq!(25, arr.len());
        assert_eq!(25, arr.null_count());

        let arr = read_array_w_params(
            &reader,
            &schema.fields[1],
            ReadBatchParams::Indices(UInt32Array::from(vec![1, 9, 30, 72])),
        )
        .await;
        assert_eq!(4, arr.len());
        assert_eq!(4, arr.null_count());

        // raise error if take indices are out of bounds
        let params = ReadBatchParams::Indices(UInt32Array::from(vec![1, 9, 30, 72, 100]));
        let arr = read_array(&reader, &schema.fields[1], 0, &params);
        assert!(arr.await.is_err());

        // raise error if range indices are out of bounds
        let params = ReadBatchParams::RangeTo(..107);
        let arr = read_array(&reader, &schema.fields[1], 0, &params);
        assert!(arr.await.is_err());
    }
}
