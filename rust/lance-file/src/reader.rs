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
use std::ops::{Range, RangeTo};
use std::sync::Arc;

use arrow_arith::numeric::sub;
use arrow_array::{
    builder::PrimitiveBuilder,
    cast::AsArray,
    types::{Int32Type, Int64Type},
    ArrayRef, ArrowNativeTypeOp, ArrowNumericType, NullArray, OffsetSizeTrait, PrimitiveArray,
    RecordBatch, StructArray, UInt32Array, UInt64Array,
};
use arrow_array::{make_array, BooleanArray};
use arrow_buffer::{ArrowNativeType, NullBuffer};
use arrow_schema::{DataType, FieldRef, Schema as ArrowSchema};
use arrow_select::concat::{self, concat_batches};
use arrow_select::filter::filter_record_batch;
use async_recursion::async_recursion;
use futures::{stream, Future, FutureExt, StreamExt, TryStreamExt};
use lance_arrow::*;
use lance_core::cache::FileMetadataCache;
use lance_core::datatypes::{Field, Schema};
use lance_core::utils::deletion::DeletionVector;
use lance_core::{Error, Result, ROW_ID, ROW_ID_FIELD};
use lance_io::encodings::dictionary::DictionaryDecoder;
use lance_io::encodings::AsyncIndex;
use lance_io::stream::{RecordBatchStream, RecordBatchStreamAdapter};
use lance_io::traits::Reader;
use lance_io::utils::{
    read_fixed_stride_array, read_metadata_offset, read_struct, read_struct_from_buf,
};
use lance_io::{object_store::ObjectStore, ReadBatchParams};

use object_store::path::Path;
use snafu::{location, Location};
use tracing::instrument;

use crate::format::metadata::Metadata;
use crate::page_table::{PageInfo, PageTable};

/// Compute row id from `fragment_id` and the `offset` of the row in the fragment.
fn compute_row_id(fragment_id: u64, offset: i32) -> u64 {
    (fragment_id << 32) + offset as u64
}

/// Lance File Reader.
///
/// It reads arrow data from one data file.
#[derive(Clone)]
pub struct FileReader {
    pub object_reader: Arc<dyn Reader>,
    metadata: Arc<Metadata>,
    page_table: Arc<PageTable>,
    schema: Schema,

    /// The id of the fragment which this file belong to.
    /// For simple file access, this can just be zero.
    fragment_id: u64,

    /// If set true, returns the row ID from the dataset alongside with the
    /// actual data.
    with_row_id: bool,

    /// If true, instead of removing deleted rows, the _rowid column value may be
    /// marked as null. This is used as a performance optimization to
    /// avoid copying data.
    make_deletions_null: bool,

    /// Page table for statistics
    stats_page_table: Arc<Option<PageTable>>,
}

impl std::fmt::Debug for FileReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FileReader(fragment={}, path={})",
            self.fragment_id,
            self.object_reader.path()
        )
    }
}

impl FileReader {
    /// Open file reader
    ///
    /// Open the file at the given path using the provided object store.
    ///
    /// The passed fragment ID determines the first 32-bits of the row IDs.
    ///
    /// If a manifest is passed in, it will be used to load the schema and dictionary.
    /// This is typically done if the file is part of a dataset fragment. If no manifest
    /// is passed in, then it is read from the file itself.
    ///
    /// The session passed in is used to cache metadata about the file. If no session
    /// is passed in, there will be no caching.
    #[instrument(level = "debug", skip(object_store, schema, session))]
    pub async fn try_new_with_fragment_id(
        object_store: &ObjectStore,
        path: &Path,
        schema: Schema,
        fragment_id: u32,
        field_id_offset: u32,
        num_fields: u32,
        session: Option<&FileMetadataCache>,
    ) -> Result<Self> {
        let object_reader = object_store.open(path).await?;

        let metadata = Self::read_metadata(object_reader.as_ref(), session).await?;

        Self::try_new_from_reader(
            path,
            object_reader.into(),
            Some(metadata),
            schema,
            fragment_id,
            field_id_offset,
            num_fields,
            session,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn try_new_from_reader(
        path: &Path,
        object_reader: Arc<dyn Reader>,
        metadata: Option<Arc<Metadata>>,
        schema: Schema,
        fragment_id: u32,
        field_id_offset: u32,
        num_fields: u32,
        session: Option<&FileMetadataCache>,
    ) -> Result<Self> {
        let metadata = match metadata {
            Some(metadata) => metadata,
            None => Self::read_metadata(object_reader.as_ref(), session).await?,
        };

        let page_table = async {
            Self::load_from_cache(session, path, |_| async {
                PageTable::load(
                    object_reader.as_ref(),
                    metadata.page_table_position,
                    num_fields as i32,
                    metadata.num_batches() as i32,
                    field_id_offset as i32,
                )
                .await
            })
            .await
        };

        let stats_page_table = Self::read_stats_page_table(object_reader.as_ref(), session);

        // Can concurrently load page tables
        let (page_table, stats_page_table) = futures::try_join!(page_table, stats_page_table)?;

        Ok(Self {
            object_reader,
            metadata,
            schema,
            page_table,
            fragment_id: fragment_id as u64,
            with_row_id: false,
            make_deletions_null: false,
            stats_page_table,
        })
    }

    pub async fn read_metadata(
        object_reader: &dyn Reader,
        cache: Option<&FileMetadataCache>,
    ) -> Result<Arc<Metadata>> {
        Self::load_from_cache(cache, object_reader.path(), |_| async {
            let file_size = object_reader.size().await?;
            let begin = if file_size < object_reader.block_size() {
                0
            } else {
                file_size - object_reader.block_size()
            };
            let tail_bytes = object_reader.get_range(begin..file_size).await?;
            let metadata_pos = read_metadata_offset(&tail_bytes)?;

            let metadata: Metadata = if metadata_pos < file_size - tail_bytes.len() {
                // We have not read the metadata bytes yet.
                read_struct(object_reader, metadata_pos).await?
            } else {
                let offset = tail_bytes.len() - (file_size - metadata_pos);
                read_struct_from_buf(&tail_bytes.slice(offset..))?
            };
            Ok(metadata)
        })
        .await
    }

    /// Get the statistics page table. This will read the metadata if it is not cached.
    ///
    /// The page table is cached.
    async fn read_stats_page_table(
        reader: &dyn Reader,
        cache: Option<&FileMetadataCache>,
    ) -> Result<Arc<Option<PageTable>>> {
        // To prevent collisions, we cache this at a child path
        Self::load_from_cache(cache, &reader.path().child("stats"), |_| async {
            let metadata = Self::read_metadata(reader, cache).await?;

            if let Some(stats_meta) = metadata.stats_metadata.as_ref() {
                Ok(Some(
                    PageTable::load(
                        reader,
                        stats_meta.page_table_position,
                        stats_meta.leaf_field_ids.len() as i32,
                        /*num_batches=*/ 1,
                        /*field_id_offset=*/ 0,
                    )
                    .await?,
                ))
            } else {
                Ok(None)
            }
        })
        .await
    }

    /// Load some metadata about the fragment from the cache, if there is one.
    async fn load_from_cache<T: Send + Sync + 'static, F, Fut>(
        cache: Option<&FileMetadataCache>,
        path: &Path,
        loader: F,
    ) -> Result<Arc<T>>
    where
        F: Fn(&Path) -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        if let Some(cache) = cache {
            cache.get_or_insert(path, loader).await
        } else {
            Ok(Arc::new(loader(path).await?))
        }
    }

    /// Open one Lance data file for read.
    pub async fn try_new(object_store: &ObjectStore, path: &Path, schema: Schema) -> Result<Self> {
        // If just reading a lance data file we assume the schema is the schema of the data file
        let num_fields = schema.max_field_id().unwrap_or_default() + 1;
        Self::try_new_with_fragment_id(object_store, path, schema, 0, 0, num_fields as u32, None)
            .await
    }

    /// Instruct the FileReader to return meta row id column.
    ///
    /// The row id column will have the name `_rowid` and will be of type `UInt64`.
    ///
    /// The row id is a 64-bit integer that is unique within the dataset. The
    /// most significant 32 bits are the fragment id, and the least significant
    /// 32 bits are the row offset within the fragment.
    pub fn with_row_id(&mut self, v: bool) -> &mut Self {
        self.with_row_id = v;
        self
    }

    /// Instruct the FileReader that instead of removing deleted rows, it may
    /// simply mark the _rowid value as null. Some rows may still be removed,
    /// for example if the entire batch is deleted. This is a performance
    /// optimization where the null bitmap of the _rowid column serves as a
    /// selection vector.
    pub fn with_make_deletions_null(&mut self, val: bool) -> &mut Self {
        self.make_deletions_null = val;
        self
    }

    /// Requested projection of the data in this file, excluding the row id column.
    pub fn schema(&self) -> &Schema {
        &self.schema
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
    #[instrument(level = "debug", skip(self, params, projection))]
    pub async fn read_batch(
        &self,
        batch_id: i32,
        params: impl Into<ReadBatchParams>,
        projection: &Schema,
        deletion_vector: Option<&DeletionVector>,
    ) -> Result<RecordBatch> {
        read_batch(
            self,
            &params.into(),
            projection,
            batch_id,
            self.with_row_id,
            deletion_vector,
        )
        .await
    }

    /// Read a range of records into one batch.
    ///
    /// Note that it might call concat if the range is crossing multiple batches, which
    /// makes it less efficient than [`FileReader::read_batch()`].
    #[instrument(level = "debug", skip(self, projection))]
    pub async fn read_range(
        &self,
        range: Range<usize>,
        projection: &Schema,
        deletion_vector: Option<&DeletionVector>,
    ) -> Result<RecordBatch> {
        let range_in_batches = self.metadata.range_to_batches(range)?;
        let batches = stream::iter(range_in_batches)
            .map(|(batch_id, range)| async move {
                self.read_batch(batch_id, range, projection, deletion_vector)
                    .await
            })
            .buffered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await?;
        if batches.len() == 1 {
            return Ok(batches[0].clone());
        }
        let schema = batches[0].schema();
        Ok(tokio::task::spawn_blocking(move || concat_batches(&schema, &batches)).await??)
    }

    /// Take by records by indices within the file.
    ///
    /// The indices must be sorted.
    #[instrument(level = "debug", skip_all)]
    pub async fn take(
        &self,
        indices: &[u32],
        projection: &Schema,
        deletion_vector: Option<&DeletionVector>,
    ) -> Result<RecordBatch> {
        let indices_in_batches = self.metadata.group_indices_to_batches(indices);
        let batches = stream::iter(indices_in_batches)
            .map(|batch| async move {
                self.read_batch(
                    batch.batch_id,
                    batch.offsets.as_slice(),
                    projection,
                    deletion_vector,
                )
                .await
            })
            .buffered(num_cpus::get() * 4)
            .try_collect::<Vec<_>>()
            .await?;

        let mut schema = ArrowSchema::from(projection);
        if self.with_row_id {
            schema = schema.try_with_column(ROW_ID_FIELD.clone())?;
        }
        let schema = Arc::new(schema);

        Ok(tokio::task::spawn_blocking(move || concat_batches(&schema, &batches)).await??)
    }

    /// Get the schema of the statistics page table, for the given data field ids.
    pub fn page_stats_schema(&self, field_ids: &[i32]) -> Option<Schema> {
        self.metadata.stats_metadata.as_ref().map(|meta| {
            let mut stats_field_ids = vec![];
            for stats_field in &meta.schema.fields {
                if let Ok(stats_field_id) = stats_field.name.parse::<i32>() {
                    if field_ids.contains(&stats_field_id) {
                        stats_field_ids.push(stats_field.id);
                        for child in &stats_field.children {
                            stats_field_ids.push(child.id);
                        }
                    }
                }
            }
            meta.schema.project_by_ids(&stats_field_ids)
        })
    }

    /// Get the page statistics for the given data field ids.
    pub async fn read_page_stats(&self, field_ids: &[i32]) -> Result<Option<RecordBatch>> {
        if let Some(stats_page_table) = self.stats_page_table.as_ref() {
            let projection = self.page_stats_schema(field_ids).unwrap();
            // It's possible none of the requested fields have stats.
            if projection.fields.is_empty() {
                return Ok(None);
            }
            let arrays = futures::stream::iter(projection.fields.iter().cloned())
                .map(|field| async move {
                    read_array(
                        self,
                        &field,
                        0,
                        stats_page_table,
                        &ReadBatchParams::RangeFull,
                    )
                    .await
                })
                .buffered(num_cpus::get())
                .try_collect::<Vec<_>>()
                .await?;

            let schema = ArrowSchema::from(&projection);
            let batch = RecordBatch::try_new(Arc::new(schema), arrays)?;
            Ok(Some(batch))
        } else {
            Ok(None)
        }
    }
}

/// Stream desired full batches from the file.
///
/// Parameters:
/// - **reader**: An opened file reader.
/// - **projection**: The schema of the returning [RecordBatch].
/// - **predicate**: A function that takes a batch ID and returns true if the batch should be
///                  returned.
///
/// Returns:
/// - A stream of [RecordBatch]s, each one corresponding to one full batch in the file.
pub fn batches_stream(
    reader: FileReader,
    projection: Schema,
    predicate: impl FnMut(&i32) -> bool + Send + Sync + 'static,
) -> impl RecordBatchStream {
    // Make projection an Arc so we can clone it and pass between threads.
    let projection = Arc::new(projection);
    let arrow_schema = ArrowSchema::from(projection.as_ref());

    let total_batches = reader.num_batches() as i32;
    let batches = (0..total_batches).filter(predicate);
    // Make another copy of self so we can clone it and pass between threads.
    let this = Arc::new(reader.clone());
    let inner = stream::iter(batches)
        .zip(stream::repeat_with(move || {
            (this.clone(), projection.clone())
        }))
        .map(move |(batch_id, (reader, projection))| async move {
            reader
                .read_batch(batch_id, ReadBatchParams::RangeFull, &projection, None)
                .await
        })
        .buffered(2)
        .boxed();
    RecordBatchStreamAdapter::new(Arc::new(arrow_schema), inner)
}

/// Read a batch.
///
/// `schema` may only be empty if `with_row_id` is also true. This function
/// panics otherwise.
async fn read_batch(
    reader: &FileReader,
    params: &ReadBatchParams,
    schema: &Schema,
    batch_id: i32,
    with_row_id: bool,
    deletion_vector: Option<&DeletionVector>,
) -> Result<RecordBatch> {
    let batch = if !schema.fields.is_empty() {
        // We box this because otherwise we get a higher-order lifetime error.
        let arrs = stream::iter(&schema.fields)
            .map(|f| async { read_array(reader, f, batch_id, &reader.page_table, params).await })
            .buffered(num_cpus::get() * 4)
            .try_collect::<Vec<_>>()
            .boxed();
        let arrs = arrs.await?;
        Some(RecordBatch::try_new(Arc::new(schema.into()), arrs)?)
    } else {
        // If the schema is empty, we are just fetching row ids.
        assert!(with_row_id);
        None
    };

    let should_fetch_row_id =
        with_row_id || !matches!(deletion_vector, None | Some(DeletionVector::NoDeletions));

    let num_rows = if let Some(batch) = &batch {
        batch.num_rows()
    } else {
        let total_rows = reader.num_rows_in_batch(batch_id);
        match params {
            ReadBatchParams::Indices(indices) => indices.len(),
            ReadBatchParams::Range(r) => r.len(),
            ReadBatchParams::RangeFull => total_rows,
            ReadBatchParams::RangeTo(r) => r.end,
            ReadBatchParams::RangeFrom(r) => total_rows - r.start,
        }
    };

    let row_ids = if should_fetch_row_id {
        let ids_in_batch: Vec<i32> = match params {
            ReadBatchParams::Indices(indices) => {
                indices.values().iter().map(|v| *v as i32).collect()
            }
            ReadBatchParams::Range(r) => r.clone().map(|v| v as i32).collect(),
            ReadBatchParams::RangeFull => (0..num_rows as i32).collect(),
            ReadBatchParams::RangeTo(r) => (0..r.end).map(|v| v as i32).collect(),
            ReadBatchParams::RangeFrom(r) => {
                (r.start..r.start + num_rows).map(|v| v as i32).collect()
            }
        };
        let batch_offset = reader
            .metadata
            .get_offset(batch_id)
            .ok_or_else(|| Error::IO {
                message: format!("batch {batch_id} does not exist"),
                location: location!(),
            })?;
        let row_ids: Vec<u64> = ids_in_batch
            .iter()
            .map(|o| compute_row_id(reader.fragment_id, *o + batch_offset))
            .collect();

        Some(row_ids)
    } else {
        None
    };

    // TODO: This is a minor cop out. Pushing deletion vector in to the decoders is hard
    // so I'm going to just leave deletion filter at this layer for now.
    // We should push this down futurther when we get to statistics-based predicate pushdown

    // This function is meant to be IO bound, but we are doing CPU-bound work here
    // We should try to move this to later.
    let span = tracing::span!(tracing::Level::DEBUG, "apply_deletions");
    let _enter = span.enter();
    let deletion_mask =
        deletion_vector.and_then(|v| v.build_predicate(row_ids.as_ref().unwrap().iter()));

    let batch = match batch {
        Some(batch) => {
            if with_row_id {
                let row_id_arr = Arc::new(UInt64Array::from(row_ids.unwrap()));
                batch.try_with_column(ROW_ID_FIELD.clone(), row_id_arr)?
            } else {
                batch
            }
        }
        None if with_row_id => {
            let row_id_arr = Arc::new(UInt64Array::from(row_ids.unwrap()));
            RecordBatch::try_new(
                Arc::new(ArrowSchema::new(vec![ROW_ID_FIELD.clone()])),
                vec![row_id_arr as ArrayRef],
            )?
        }
        _ => unreachable!(),
    };

    match (deletion_mask, reader.make_deletions_null) {
        (None, _) => Ok(batch),
        (Some(mask), false) => Ok(filter_record_batch(&batch, &mask)?),
        (Some(mask), true) => Ok(apply_deletions_as_nulls(batch, &mask)?),
    }
}

/// Apply a mask to the batch, where rows are "deleted" by the _rowid column null.
/// This is used as a performance optimization to avoid copying data.
fn apply_deletions_as_nulls(batch: RecordBatch, mask: &BooleanArray) -> Result<RecordBatch> {
    // Transform mask into null buffer. Null means deleted, though note that
    // null buffers are actually validity buffers, so True means not null
    // and thus not deleted.
    let mask_buffer = NullBuffer::new(mask.values().clone());

    match mask_buffer.null_count() {
        // All rows are deleted
        n if n == mask_buffer.len() => return Ok(RecordBatch::new_empty(batch.schema())),
        // No rows are deleted
        0 => return Ok(batch),
        _ => {}
    }

    // For each column convert to data
    let new_columns = batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(field, col)| {
            if field.name() == ROW_ID {
                let col_data = col.to_data();
                // If it already has a validity bitmap, then AND it with the mask.
                // Otherwise, use the boolean buffer as the mask.
                let null_buffer = NullBuffer::union(col_data.nulls(), Some(&mask_buffer));

                Ok(col_data
                    .into_builder()
                    .null_bit_buffer(null_buffer.map(|b| b.buffer().clone()))
                    .build()
                    .map(make_array)?)
            } else {
                Ok(col.clone())
            }
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(RecordBatch::try_new(batch.schema(), new_columns)?)
}

#[async_recursion]
async fn read_array(
    reader: &FileReader,
    field: &Field,
    batch_id: i32,
    page_table: &PageTable,
    params: &ReadBatchParams,
) -> Result<ArrayRef> {
    let data_type = field.data_type();

    use DataType::*;

    if data_type.is_fixed_stride() {
        _read_fixed_stride_array(reader, field, batch_id, page_table, params).await
    } else {
        match data_type {
            Null => read_null_array(field, batch_id, page_table, params),
            Utf8 | LargeUtf8 | Binary | LargeBinary => {
                read_binary_array(reader, field, batch_id, page_table, params).await
            }
            Struct(_) => read_struct_array(reader, field, batch_id, page_table, params).await,
            Dictionary(_, _) => {
                read_dictionary_array(reader, field, batch_id, page_table, params).await
            }
            List(_) => {
                read_list_array::<Int32Type>(reader, field, batch_id, page_table, params).await
            }
            LargeList(_) => {
                read_list_array::<Int64Type>(reader, field, batch_id, page_table, params).await
            }
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
    page_table.get(field.id, batch_id).ok_or_else(|| Error::IO {
        message: format!(
            "No page info found for field: {}, field_id={} batch={}",
            field.name, field.id, batch_id
        ),
        location: location!(),
    })
}

/// Read primitive array for batch `batch_idx`.
async fn _read_fixed_stride_array(
    reader: &FileReader,
    field: &Field,
    batch_id: i32,
    page_table: &PageTable,
    params: &ReadBatchParams,
) -> Result<ArrayRef> {
    let page_info = get_page_info(page_table, field, batch_id)?;

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
    field: &Field,
    batch_id: i32,
    page_table: &PageTable,
    params: &ReadBatchParams,
) -> Result<ArrayRef> {
    let page_info = get_page_info(page_table, field, batch_id)?;

    let length_output = match params {
        ReadBatchParams::Indices(indices) => {
            if indices.is_empty() {
                0
            } else {
                let idx_max = *indices.values().iter().max().unwrap() as u64;
                if idx_max >= page_info.length.try_into().unwrap() {
                    return Err(Error::IO {
                        message: format!(
                            "NullArray Reader: request([{}]) out of range: [0..{}]",
                            idx_max, page_info.length
                        ),
                        location: location!(),
                    });
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
                return Err(Error::IO {
                    message: format!(
                        "NullArray Reader: request([{}..{}]) out of range: [0..{}]",
                        idx_start, idx_end, page_info.length
                    ),
                    location: location!(),
                });
            }
            idx_end - idx_start
        }
    };

    Ok(Arc::new(NullArray::new(length_output)))
}

async fn read_binary_array(
    reader: &FileReader,
    field: &Field,
    batch_id: i32,
    page_table: &PageTable,
    params: &ReadBatchParams,
) -> Result<ArrayRef> {
    let page_info = get_page_info(page_table, field, batch_id)?;

    lance_io::utils::read_binary_array(
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
    reader: &FileReader,
    field: &Field,
    batch_id: i32,
    page_table: &PageTable,
    params: &ReadBatchParams,
) -> Result<ArrayRef> {
    let page_info = get_page_info(page_table, field, batch_id)?;
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
    reader: &FileReader,
    field: &Field,
    batch_id: i32,
    page_table: &PageTable,
    params: &ReadBatchParams,
) -> Result<ArrayRef> {
    // TODO: use tokio to make the reads in parallel.
    let mut sub_arrays: Vec<(FieldRef, ArrayRef)> = vec![];

    for child in field.children.as_slice() {
        let arr = read_array(reader, child, batch_id, page_table, params).await?;
        sub_arrays.push((Arc::new(child.into()), arr));
    }

    Ok(Arc::new(StructArray::from(sub_arrays)))
}

async fn take_list_array<T: ArrowNumericType>(
    reader: &FileReader,
    field: &Field,
    batch_id: i32,
    page_table: &PageTable,
    positions: &PrimitiveArray<T>,
    indices: &UInt32Array,
) -> Result<ArrayRef>
where
    T::Native: ArrowNativeTypeOp + OffsetSizeTrait,
{
    let first_idx = indices.value(0);
    // Range of values for each index
    let ranges = indices
        .values()
        .iter()
        .map(|i| (*i - first_idx).as_usize())
        .map(|idx| positions.value(idx).as_usize()..positions.value(idx + 1).as_usize())
        .collect::<Vec<_>>();
    let field = field.clone();
    let mut list_values: Vec<ArrayRef> = vec![];
    // TODO: read them in parallel.
    for range in ranges.iter() {
        list_values.push(
            read_array(
                reader,
                &field.children[0],
                batch_id,
                page_table,
                &(range.clone()).into(),
            )
            .await?,
        );
    }

    let value_refs = list_values
        .iter()
        .map(|arr| arr.as_ref())
        .collect::<Vec<_>>();
    let mut offsets_builder = PrimitiveBuilder::<T>::new();
    offsets_builder.append_value(T::Native::usize_as(0));
    let mut off = 0_usize;
    for range in ranges {
        off += range.len();
        offsets_builder.append_value(T::Native::usize_as(off));
    }
    let all_values = concat::concat(value_refs.as_slice())?;
    let offset_arr = offsets_builder.finish();
    let arr = try_new_generic_list_array(all_values, &offset_arr)?;
    Ok(Arc::new(arr) as ArrayRef)
}

async fn read_list_array<T: ArrowNumericType>(
    reader: &FileReader,
    field: &Field,
    batch_id: i32,
    page_table: &PageTable,
    params: &ReadBatchParams,
) -> Result<ArrayRef>
where
    T::Native: ArrowNativeTypeOp + OffsetSizeTrait,
{
    // Offset the position array by 1 in order to include the upper bound of the last element
    let positions_params = match params {
        ReadBatchParams::Range(range) => ReadBatchParams::from(range.start..(range.end + 1)),
        ReadBatchParams::RangeTo(range) => ReadBatchParams::from(..range.end + 1),
        ReadBatchParams::Indices(indices) => {
            (indices.value(0).as_usize()..indices.value(indices.len() - 1).as_usize() + 2).into()
        }
        p => p.clone(),
    };

    let page_info = get_page_info(&reader.page_table, field, batch_id)?;
    let position_arr = read_fixed_stride_array(
        reader.object_reader.as_ref(),
        &T::DATA_TYPE,
        page_info.position,
        page_info.length,
        positions_params,
    )
    .await?;

    let positions: &PrimitiveArray<T> = position_arr.as_primitive();

    // Recompute params so they align with the offset array
    let value_params = match params {
        ReadBatchParams::Range(range) => ReadBatchParams::from(
            positions.value(0).as_usize()..positions.value(range.end - range.start).as_usize(),
        ),
        ReadBatchParams::RangeTo(RangeTo { end }) => {
            ReadBatchParams::from(..positions.value(*end).as_usize())
        }
        ReadBatchParams::RangeFrom(_) => ReadBatchParams::from(positions.value(0).as_usize()..),
        ReadBatchParams::RangeFull => ReadBatchParams::from(
            positions.value(0).as_usize()..positions.value(positions.len() - 1).as_usize(),
        ),
        ReadBatchParams::Indices(indices) => {
            return take_list_array(reader, field, batch_id, page_table, positions, indices).await;
        }
    };

    let start_position = PrimitiveArray::<T>::new_scalar(positions.value(0));
    let offset_arr = sub(positions, &start_position)?;
    let offset_arr_ref = offset_arr.as_primitive::<T>();
    let value_arrs = read_array(
        reader,
        &field.children[0],
        batch_id,
        page_table,
        &value_params,
    )
    .await?;
    let arr = try_new_generic_list_array(value_arrs, offset_arr_ref)?;
    Ok(Arc::new(arr) as ArrayRef)
}

#[cfg(test)]
mod tests {
    use crate::writer::{FileWriter, NotSelfDescribing};

    use super::*;

    use arrow_array::Int32Array;
    use arrow_array::{
        builder::{Int32Builder, LargeListBuilder, ListBuilder, StringBuilder},
        cast::{as_string_array, as_struct_array},
        types::{Int32Type, UInt8Type},
        Array, BooleanArray, DictionaryArray, Float32Array, Int64Array, LargeListArray, ListArray,
        NullArray, StringArray, StructArray, UInt32Array, UInt8Array,
    };
    use arrow_schema::{Field as ArrowField, Fields as ArrowFields, Schema as ArrowSchema};
    use roaring::RoaringBitmap;

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
        let mut file_writer = FileWriter::<NotSelfDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();
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
            file_writer.write(&[batch]).await.unwrap();
        }
        file_writer.finish().await.unwrap();

        let fragment = 123_u64;
        let mut reader = FileReader::try_new_with_fragment_id(
            &store,
            &path,
            schema,
            fragment as u32,
            0,
            2,
            None,
        )
        .await
        .unwrap();
        reader.with_row_id(true);

        for b in 0..10 {
            let batch = reader
                .read_batch(b, .., reader.schema(), None)
                .await
                .unwrap();
            let row_ids_col = &batch[ROW_ID];
            // Do the same computation as `compute_row_id`.
            let start_pos = (fragment << 32) + 10 * b as u64;

            assert_eq!(
                &UInt64Array::from_iter_values(start_pos..start_pos + 10),
                row_ids_col.as_primitive(),
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
        let values_ref = Arc::new(values);
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
                    value_range.clone().map(|n| format!("str-{}", n)),
                )),
                Arc::new(DictionaryArray::<UInt8Type>::try_new(keys, values_ref.clone()).unwrap()),
            ];
            batches.push(RecordBatch::try_new(Arc::new(arrow_schema.clone()), columns).unwrap());
        }
        schema.set_dictionary(&batches[0]).unwrap();

        let mut file_writer = FileWriter::<NotSelfDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();
        for batch in batches.iter() {
            file_writer.write(&[batch.clone()]).await.unwrap();
        }
        file_writer.finish().await.unwrap();

        let reader = FileReader::try_new(&store, &path, schema).await.unwrap();
        let batch = reader
            .take(&[1, 15, 20, 25, 30, 48, 90], reader.schema(), None)
            .await
            .unwrap();
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
                    Arc::new(DictionaryArray::try_new(dict_keys, values_ref.clone()).unwrap()),
                ]
            )
            .unwrap()
        );
    }

    #[tokio::test]
    async fn read_with_delete() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int64, true),
            ArrowField::new("f", DataType::Float32, false),
        ]);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        let (store, path) = ObjectStore::from_uri("memory:///foo").await.unwrap();

        // Write 10 batches.
        let mut file_writer = FileWriter::<NotSelfDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();
        for batch_id in 0..10 {
            let value_range = batch_id * 10..batch_id * 10 + 10;
            let columns: Vec<ArrayRef> = vec![
                Arc::new(Int64Array::from_iter(value_range.clone())),
                Arc::new(Float32Array::from_iter(value_range.map(|n| n as f32))),
            ];
            let batch = RecordBatch::try_new(Arc::new(arrow_schema.clone()), columns).unwrap();
            file_writer.write(&[batch]).await.unwrap();
        }
        file_writer.finish().await.unwrap();

        let fragment = 123;

        // delete even rows
        let dv = DeletionVector::Bitmap(RoaringBitmap::from_iter((0..100).filter(|x| x % 2 == 0)));

        let mut reader = FileReader::try_new_with_fragment_id(
            &store,
            &path,
            schema,
            fragment as u32,
            0,
            2,
            None,
        )
        .await
        .unwrap();
        reader.with_row_id(true);

        for b in 0..10 {
            let batch = reader
                .read_batch(b, .., reader.schema(), Some(&dv))
                .await
                .unwrap();
            let row_ids_col = &batch[ROW_ID];
            // Do the same computation as `compute_row_id`.
            let start_pos = (fragment << 32) + 10 * b as u64;

            assert_eq!(
                &UInt64Array::from_iter_values((start_pos..start_pos + 10).filter(|i| i % 2 == 1)),
                row_ids_col.as_primitive(),
            );
        }
    }

    #[tokio::test]
    async fn read_with_delete_without_row_id() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int64, true),
            ArrowField::new("f", DataType::Float32, false),
        ]);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        let (store, path) = ObjectStore::from_uri("memory:///foo").await.unwrap();

        // Write 10 batches.
        let mut file_writer = FileWriter::<NotSelfDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();
        for batch_id in 0..10 {
            let value_range = batch_id * 10..batch_id * 10 + 10;
            let columns: Vec<ArrayRef> = vec![
                Arc::new(Int64Array::from_iter(value_range.clone())),
                Arc::new(Float32Array::from_iter(value_range.map(|n| n as f32))),
            ];
            let batch = RecordBatch::try_new(Arc::new(arrow_schema.clone()), columns).unwrap();
            file_writer.write(&[batch]).await.unwrap();
        }
        file_writer.finish().await.unwrap();

        let fragment = 123;

        // delete even rows
        let dv = DeletionVector::Bitmap(RoaringBitmap::from_iter((0..100).filter(|x| x % 2 == 0)));

        let mut reader = FileReader::try_new_with_fragment_id(
            &store,
            &path,
            schema,
            fragment as u32,
            0,
            2,
            None,
        )
        .await
        .unwrap();
        reader.with_row_id(false);

        for b in 0..10 {
            let batch = reader
                .read_batch(b, .., reader.schema(), Some(&dv))
                .await
                .unwrap();
            // if we didn't request rowid we should not get it back
            assert!(!batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .any(|name| name == ROW_ID))
        }
    }

    async fn test_write_null_string_in_struct(field_nullable: bool) {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "parent",
            DataType::Struct(ArrowFields::from(vec![ArrowField::new(
                "str",
                DataType::Utf8,
                field_nullable,
            )])),
            true,
        )]));

        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();

        let store = ObjectStore::memory();
        let path = Path::from("/null_strings");

        let string_arr = Arc::new(StringArray::from_iter([Some("a"), Some(""), Some("b")]));
        let struct_arr = Arc::new(StructArray::from(vec![(
            Arc::new(ArrowField::new("str", DataType::Utf8, field_nullable)),
            string_arr.clone() as ArrayRef,
        )]));
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![struct_arr]).unwrap();

        let mut file_writer = FileWriter::<NotSelfDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();
        file_writer.write(&[batch.clone()]).await.unwrap();
        file_writer.finish().await.unwrap();

        let reader = FileReader::try_new(&store, &path, schema).await.unwrap();
        let actual_batch = reader
            .read_batch(0, .., reader.schema(), None)
            .await
            .unwrap();

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
        let store = ObjectStore::memory();
        let path = Path::from("/null_strings");

        let arrow_schema = make_schema_of_list_array();
        let schema: Schema = Schema::try_from(arrow_schema.as_ref()).unwrap();

        let batches = (0..3)
            .map(|_| {
                let struct_array = make_struct_of_list_array(10, 10);
                RecordBatch::try_new(arrow_schema.clone(), vec![struct_array]).unwrap()
            })
            .collect::<Vec<_>>();
        let batches_ref = batches.iter().collect::<Vec<_>>();

        let mut file_writer = FileWriter::<NotSelfDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();
        file_writer.write(&batches).await.unwrap();
        file_writer.finish().await.unwrap();

        let reader = FileReader::try_new(&store, &path, schema).await.unwrap();
        let actual_batch = reader
            .read_batch(0, .., reader.schema(), None)
            .await
            .unwrap();
        let expected = concat_batches(&arrow_schema, batches_ref).unwrap();
        assert_eq!(expected, actual_batch);
    }

    #[tokio::test]
    async fn test_scan_struct_of_list_arrays() {
        let store = ObjectStore::memory();
        let path = Path::from("/null_strings");

        let arrow_schema = make_schema_of_list_array();
        let struct_array = make_struct_of_list_array(3, 10);
        let schema: Schema = Schema::try_from(arrow_schema.as_ref()).unwrap();
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![struct_array.clone()]).unwrap();

        let mut file_writer = FileWriter::<NotSelfDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();
        file_writer.write(&[batch]).await.unwrap();
        file_writer.finish().await.unwrap();

        let mut expected_columns: Vec<ArrayRef> = Vec::new();
        for c in struct_array.columns().iter() {
            expected_columns.push(c.slice(1, 1));
        }

        let expected_struct = match arrow_schema.fields[0].data_type() {
            DataType::Struct(subfields) => subfields
                .iter()
                .zip(expected_columns)
                .map(|(f, d)| (f.clone(), d))
                .collect::<Vec<_>>(),
            _ => panic!("unexpected field"),
        };

        let expected_struct_array = StructArray::from(expected_struct);
        let expected_batch = RecordBatch::from(&StructArray::from(vec![(
            Arc::new(arrow_schema.fields[0].as_ref().clone()),
            Arc::new(expected_struct_array) as ArrayRef,
        )]));

        let reader = FileReader::try_new(&store, &path, schema).await.unwrap();
        let params = ReadBatchParams::Range(1..2);
        let slice_of_batch = reader
            .read_batch(0, params, reader.schema(), None)
            .await
            .unwrap();
        assert_eq!(expected_batch, slice_of_batch);
    }

    fn make_schema_of_list_array() -> Arc<arrow_schema::Schema> {
        Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "s",
            DataType::Struct(ArrowFields::from(vec![
                ArrowField::new(
                    "li",
                    DataType::List(Arc::new(ArrowField::new("item", DataType::Int32, true))),
                    true,
                ),
                ArrowField::new(
                    "ls",
                    DataType::List(Arc::new(ArrowField::new("item", DataType::Utf8, true))),
                    true,
                ),
                ArrowField::new(
                    "ll",
                    DataType::LargeList(Arc::new(ArrowField::new("item", DataType::Int32, true))),
                    false,
                ),
            ])),
            true,
        )]))
    }

    fn make_struct_of_list_array(rows: i32, num_items: i32) -> Arc<StructArray> {
        let mut li_builder = ListBuilder::new(Int32Builder::new());
        let mut ls_builder = ListBuilder::new(StringBuilder::new());
        let ll_value_builder = Int32Builder::new();
        let mut large_list_builder = LargeListBuilder::new(ll_value_builder);
        for i in 0..rows {
            for j in 0..num_items {
                li_builder.values().append_value(i * 10 + j);
                ls_builder
                    .values()
                    .append_value(format!("str-{}", i * 10 + j));
                large_list_builder.values().append_value(i * 10 + j);
            }
            li_builder.append(true);
            ls_builder.append(true);
            large_list_builder.append(true);
        }
        Arc::new(StructArray::from(vec![
            (
                Arc::new(ArrowField::new(
                    "li",
                    DataType::List(Arc::new(ArrowField::new("item", DataType::Int32, true))),
                    true,
                )),
                Arc::new(li_builder.finish()) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new(
                    "ls",
                    DataType::List(Arc::new(ArrowField::new("item", DataType::Utf8, true))),
                    true,
                )),
                Arc::new(ls_builder.finish()) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new(
                    "ll",
                    DataType::LargeList(Arc::new(ArrowField::new("item", DataType::Int32, true))),
                    false,
                )),
                Arc::new(large_list_builder.finish()) as ArrayRef,
            ),
        ]))
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
        let mut file_writer = FileWriter::<NotSelfDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();
        file_writer.write(&[batch]).await.unwrap();
        file_writer.finish().await.unwrap();

        // read the file back
        let reader = FileReader::try_new(&store, &path, schema.clone())
            .await
            .unwrap();

        async fn read_array_w_params(
            reader: &FileReader,
            field: &Field,
            params: ReadBatchParams,
        ) -> ArrayRef {
            read_array(reader, field, 0, reader.page_table.as_ref(), &params)
                .await
                .expect("Error reading back the null array from file") as _
        }

        let arr = read_array_w_params(&reader, &schema.fields[1], ReadBatchParams::RangeFull).await;
        assert_eq!(100, arr.len());
        assert_eq!(arr.data_type(), &DataType::Null);

        let arr =
            read_array_w_params(&reader, &schema.fields[1], ReadBatchParams::Range(10..25)).await;
        assert_eq!(15, arr.len());
        assert_eq!(arr.data_type(), &DataType::Null);

        let arr =
            read_array_w_params(&reader, &schema.fields[1], ReadBatchParams::RangeFrom(60..)).await;
        assert_eq!(40, arr.len());
        assert_eq!(arr.data_type(), &DataType::Null);

        let arr =
            read_array_w_params(&reader, &schema.fields[1], ReadBatchParams::RangeTo(..25)).await;
        assert_eq!(25, arr.len());
        assert_eq!(arr.data_type(), &DataType::Null);

        let arr = read_array_w_params(
            &reader,
            &schema.fields[1],
            ReadBatchParams::Indices(UInt32Array::from(vec![1, 9, 30, 72])),
        )
        .await;
        assert_eq!(4, arr.len());
        assert_eq!(arr.data_type(), &DataType::Null);

        // raise error if take indices are out of bounds
        let params = ReadBatchParams::Indices(UInt32Array::from(vec![1, 9, 30, 72, 100]));
        let arr = read_array(
            &reader,
            &schema.fields[1],
            0,
            reader.page_table.as_ref(),
            &params,
        );
        assert!(arr.await.is_err());

        // raise error if range indices are out of bounds
        let params = ReadBatchParams::RangeTo(..107);
        let arr = read_array(
            &reader,
            &schema.fields[1],
            0,
            reader.page_table.as_ref(),
            &params,
        );
        assert!(arr.await.is_err());
    }

    #[tokio::test]
    async fn test_take_lists() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new(
                "l",
                DataType::List(Arc::new(ArrowField::new("item", DataType::Int32, true))),
                false,
            ),
            ArrowField::new(
                "ll",
                DataType::LargeList(Arc::new(ArrowField::new("item", DataType::Int32, true))),
                false,
            ),
        ]);

        let value_builder = Int32Builder::new();
        let mut list_builder = ListBuilder::new(value_builder);
        let ll_value_builder = Int32Builder::new();
        let mut large_list_builder = LargeListBuilder::new(ll_value_builder);
        for i in 0..100 {
            list_builder.values().append_value(i);
            large_list_builder.values().append_value(i);
            if (i + 1) % 10 == 0 {
                list_builder.append(true);
                large_list_builder.append(true);
            }
        }
        let list_arr = Arc::new(list_builder.finish());
        let large_list_arr = Arc::new(large_list_builder.finish());

        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema.clone()),
            vec![list_arr as ArrayRef, large_list_arr as ArrayRef],
        )
        .unwrap();

        // write to a lance file
        let store = ObjectStore::memory();
        let path = Path::from("/take_list");
        let schema: Schema = (&arrow_schema).try_into().unwrap();
        let mut file_writer = FileWriter::<NotSelfDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();
        file_writer.write(&[batch]).await.unwrap();
        file_writer.finish().await.unwrap();

        // read the file back
        let reader = FileReader::try_new(&store, &path, schema.clone())
            .await
            .unwrap();
        let actual = reader.take(&[1, 3, 5, 9], &schema, None).await.unwrap();

        let value_builder = Int32Builder::new();
        let mut list_builder = ListBuilder::new(value_builder);
        let ll_value_builder = Int32Builder::new();
        let mut large_list_builder = LargeListBuilder::new(ll_value_builder);
        for i in [1, 3, 5, 9] {
            for j in 0..10 {
                list_builder.values().append_value(i * 10 + j);
                large_list_builder.values().append_value(i * 10 + j);
            }
            list_builder.append(true);
            large_list_builder.append(true);
        }
        let expected_list = list_builder.finish();
        let expected_large_list = large_list_builder.finish();

        assert_eq!(actual.column_by_name("l").unwrap().as_ref(), &expected_list);
        assert_eq!(
            actual.column_by_name("ll").unwrap().as_ref(),
            &expected_large_list
        );
    }

    #[tokio::test]
    async fn test_list_array_with_offsets() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new(
                "l",
                DataType::List(Arc::new(ArrowField::new("item", DataType::Int32, true))),
                false,
            ),
            ArrowField::new(
                "ll",
                DataType::LargeList(Arc::new(ArrowField::new("item", DataType::Int32, true))),
                false,
            ),
        ]);

        let store = ObjectStore::memory();
        let path = Path::from("/lists");

        let list_array = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(1), Some(2)]),
            Some(vec![Some(3), Some(4)]),
            Some((0..2_000).map(Some).collect::<Vec<_>>()),
        ])
        .slice(1, 1);
        let large_list_array = LargeListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(10), Some(11)]),
            Some(vec![Some(12), Some(13)]),
            Some((0..2_000).map(Some).collect::<Vec<_>>()),
        ])
        .slice(1, 1);

        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema.clone()),
            vec![Arc::new(list_array), Arc::new(large_list_array)],
        )
        .unwrap();

        let schema: Schema = (&arrow_schema).try_into().unwrap();
        let mut file_writer = FileWriter::<NotSelfDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();
        file_writer.write(&[batch.clone()]).await.unwrap();
        file_writer.finish().await.unwrap();

        // Make sure the big array was not written to the file
        let file_size_bytes = store.size(&path).await.unwrap();
        assert!(file_size_bytes < 1_000);

        let reader = FileReader::try_new(&store, &path, schema).await.unwrap();
        let actual_batch = reader
            .read_batch(0, .., reader.schema(), None)
            .await
            .unwrap();
        assert_eq!(batch, actual_batch);
    }

    #[tokio::test]
    async fn test_read_ranges() {
        // create a record batch with a null array column
        let arrow_schema = ArrowSchema::new(vec![ArrowField::new("i", DataType::Int64, false)]);
        let schema = Schema::try_from(&arrow_schema).unwrap();
        let columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from_iter_values(0..100))];
        let batch = RecordBatch::try_new(Arc::new(arrow_schema), columns).unwrap();

        // write to a lance file
        let store = ObjectStore::memory();
        let path = Path::from("/read_range");
        let mut file_writer = FileWriter::<NotSelfDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();
        file_writer.write(&[batch]).await.unwrap();
        file_writer.finish().await.unwrap();

        let reader = FileReader::try_new(&store, &path, schema).await.unwrap();
        let actual_batch = reader
            .read_range(7..25, reader.schema(), None)
            .await
            .unwrap();

        assert_eq!(
            actual_batch.column_by_name("i").unwrap().as_ref(),
            &Int64Array::from_iter_values(7..25)
        );
    }

    #[tokio::test]
    async fn test_batches_stream() {
        let store = ObjectStore::memory();
        let path = Path::from("/batch_stream");

        let arrow_schema = ArrowSchema::new(vec![ArrowField::new("i", DataType::Int32, true)]);
        let schema = Schema::try_from(&arrow_schema).unwrap();
        let mut writer = FileWriter::<NotSelfDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();
        for i in 0..10 {
            let batch = RecordBatch::try_new(
                Arc::new(arrow_schema.clone()),
                vec![Arc::new(Int32Array::from_iter_values(i * 10..(i + 1) * 10))],
            )
            .unwrap();
            writer.write(&[batch]).await.unwrap();
        }
        writer.finish().await.unwrap();

        let reader = FileReader::try_new(&store, &path, schema.clone())
            .await
            .unwrap();
        let stream = batches_stream(reader, schema, |id| id % 2 == 0);
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();

        assert_eq!(batches.len(), 5);
        for (i, batch) in batches.iter().enumerate() {
            assert_eq!(
                batch,
                &RecordBatch::try_new(
                    Arc::new(arrow_schema.clone()),
                    vec![Arc::new(Int32Array::from_iter_values(
                        i as i32 * 2 * 10..(i as i32 * 2 + 1) * 10
                    ))],
                )
                .unwrap()
            )
        }
    }

    #[tokio::test]
    async fn test_take_boolean_beyond_chunk() {
        let mut store = ObjectStore::memory();
        store.set_block_size(256);
        let path = Path::from("/take_bools");

        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "b",
            DataType::Boolean,
            false,
        )]));
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();
        let mut file_writer = FileWriter::<NotSelfDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();

        let array = BooleanArray::from((0..5000).map(|v| v % 5 == 0).collect::<Vec<_>>());
        let batch =
            RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(array.clone())]).unwrap();
        file_writer.write(&[batch]).await.unwrap();
        file_writer.finish().await.unwrap();

        let reader = FileReader::try_new(&store, &path, schema.clone())
            .await
            .unwrap();
        let actual = reader
            .take(&[2, 4, 5, 8, 4555], &schema, None)
            .await
            .unwrap();

        assert_eq!(
            actual.column_by_name("b").unwrap().as_ref(),
            &BooleanArray::from(vec![false, false, true, false, true])
        );
    }

    #[tokio::test]
    async fn test_read_projection() {
        // The dataset schema may be very large.  The file reader should support reading
        // a small projection of that schema (this just tests the field_offset / num_fields
        // parameters)
        let store = ObjectStore::memory();
        let path = Path::from("/partial_read");

        // Create a large schema
        let mut fields = vec![];
        for i in 0..100 {
            fields.push(ArrowField::new(format!("f{}", i), DataType::Int32, false));
        }
        let arrow_schema = ArrowSchema::new(fields);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        let partial_schema = schema.project(&["f50"]).unwrap();
        let partial_arrow: ArrowSchema = (&partial_schema).into();

        let mut file_writer = FileWriter::<NotSelfDescribing>::try_new(
            &store,
            &path,
            partial_schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();

        let array = Int32Array::from(vec![0; 15]);
        let batch =
            RecordBatch::try_new(Arc::new(partial_arrow), vec![Arc::new(array.clone())]).unwrap();
        file_writer.write(&[batch.clone()]).await.unwrap();
        file_writer.finish().await.unwrap();

        let field_id = partial_schema.fields.first().unwrap().id;
        let num_fields = 1;
        let reader = FileReader::try_new_with_fragment_id(
            &store,
            &path,
            schema.clone(),
            0,
            field_id as u32,
            num_fields,
            None,
        )
        .await
        .unwrap();
        let actual = reader
            .read_batch(0, ReadBatchParams::RangeFull, &partial_schema, None)
            .await
            .unwrap();

        assert_eq!(actual, batch);
    }
}
