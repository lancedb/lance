// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::BTreeSet, io::Cursor, ops::Range, pin::Pin, sync::Arc};

use arrow_schema::Schema as ArrowSchema;
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use bytes::{Bytes, BytesMut};
use futures::{stream::BoxStream, Stream, StreamExt};
use lance_arrow::DataTypeExt;
use lance_encoding::{
    decoder::{
        BatchDecodeStream, ColumnInfo, CoreFieldDecoderStrategy, DecodeBatchScheduler, PageInfo,
        ReadBatchTask,
    },
    EncodingsIo,
};
use log::debug;
use prost::{Message, Name};
use snafu::{location, Location};

use lance_core::{
    datatypes::{Field, Schema},
    Error, Result,
};
use lance_encoding::format::pb as pbenc;
use lance_io::{
    scheduler::FileScheduler,
    stream::{RecordBatchStream, RecordBatchStreamAdapter},
    ReadBatchParams,
};
use tokio::sync::mpsc;

use crate::{
    datatypes::{Fields, FieldsWithMeta},
    format::{pb, pbfile, MAGIC, MAJOR_VERSION, MINOR_VERSION_NEXT},
};

use super::io::LanceEncodingsIo;

// For now, we don't use global buffers for anything other than schema.  If we
// use these later we should make them lazily loaded and then cached once loaded.
//
// We store their position / length for debugging purposes
#[derive(Debug)]
pub struct BufferDescriptor {
    pub position: u64,
    pub size: u64,
}

// TODO: Caching
#[derive(Debug)]
pub struct CachedFileMetadata {
    /// The schema of the file
    pub file_schema: Arc<Schema>,
    /// The column metadatas
    pub column_metadatas: Vec<pbfile::ColumnMetadata>,
    pub column_infos: Vec<Arc<ColumnInfo>>,
    /// The number of rows in the file
    pub num_rows: u64,
    pub file_buffers: Vec<BufferDescriptor>,
    /// The number of bytes contained in the data page section of the file
    pub num_data_bytes: u64,
    /// The number of bytes contained in the column metadata (not including buffers
    /// referenced by the metadata)
    pub num_column_metadata_bytes: u64,
    /// The number of bytes contained in global buffers
    pub num_global_buffer_bytes: u64,
    /// The number of bytes contained in the CMO and GBO tables
    pub num_footer_bytes: u64,
    pub major_version: u16,
    pub minor_version: u16,
}

/// Selecting columns from a lance file requires specifying both the
/// index of the column and the data type of the column
///
/// Partly, this is because it is not strictly required that columns
/// be read into the same type.  For example, a string column may be
/// read as a string, large_string or string_view type.
///
/// A read will only succeed if the decoder for a column is capable
/// of decoding into the requested type.
///
/// Note that this should generally be limited to different in-memory
/// representations of the same semantic type.  An encoding could
/// theoretically support "casting" (e.g. int to string,  etc.) but
/// there is little advantage in doing so here.
#[derive(Debug, Clone)]
pub struct ReaderProjection {
    /// The data types (schema) of the selected columns.  The names
    /// of the schema are arbitrary and ignored.
    pub schema: Arc<ArrowSchema>,
    /// The indices of the columns to load.  Note, these are the
    /// indices of the top level fields only
    pub column_indices: Vec<u32>,
}

#[derive(Debug)]
pub struct FileReader {
    scheduler: Arc<LanceEncodingsIo>,
    // The default projection to be applied to all reads
    base_projection: ReaderProjection,
    num_rows: u64,
    metadata: Arc<CachedFileMetadata>,
}

struct Footer {
    #[allow(dead_code)]
    column_meta_start: u64,
    // We don't use this today because we always load metadata for every column
    // and don't yet support "metadata projection"
    #[allow(dead_code)]
    column_meta_offsets_start: u64,
    global_buff_offsets_start: u64,
    num_global_buffers: u32,
    num_columns: u32,
    major_version: u16,
    minor_version: u16,
}

const FOOTER_LEN: usize = 40;

impl FileReader {
    pub fn metadata(&self) -> &Arc<CachedFileMetadata> {
        &self.metadata
    }

    async fn read_tail(scheduler: &FileScheduler) -> Result<(Bytes, u64)> {
        let file_size = scheduler.reader().size().await? as u64;
        let begin = if file_size < scheduler.reader().block_size() as u64 {
            0
        } else {
            file_size - scheduler.reader().block_size() as u64
        };
        let tail_bytes = scheduler.submit_single(begin..file_size, 0).await?;
        Ok((tail_bytes, file_size))
    }

    // Checks to make sure the footer is written correctly and returns the
    // position of the file descriptor (which comes from the footer)
    fn decode_footer(footer_bytes: &Bytes) -> Result<Footer> {
        let len = footer_bytes.len();
        if len < FOOTER_LEN {
            return Err(Error::io(
                format!(
                    "does not have sufficient data, len: {}, bytes: {:?}",
                    len, footer_bytes
                ),
                location!(),
            ));
        }
        let mut cursor = Cursor::new(footer_bytes.slice(len - FOOTER_LEN..));

        let column_meta_start = cursor.read_u64::<LittleEndian>()?;
        let column_meta_offsets_start = cursor.read_u64::<LittleEndian>()?;
        let global_buff_offsets_start = cursor.read_u64::<LittleEndian>()?;
        let num_global_buffers = cursor.read_u32::<LittleEndian>()?;
        let num_columns = cursor.read_u32::<LittleEndian>()?;
        let major_version = cursor.read_u16::<LittleEndian>()?;
        let minor_version = cursor.read_u16::<LittleEndian>()?;

        if major_version != MAJOR_VERSION as u16 || minor_version != MINOR_VERSION_NEXT {
            return Err(Error::io(
                format!(
                    "Attempt to use the lance v0.2 reader to read a file with version {}.{}",
                    major_version, minor_version
                ),
                location!(),
            ));
        }

        let magic_bytes = footer_bytes.slice(len - 4..);
        if magic_bytes.as_ref() != MAGIC {
            return Err(Error::io(
                format!(
                    "file does not appear to be a Lance file (invalid magic: {:?})",
                    MAGIC
                ),
                location!(),
            ));
        }
        Ok(Footer {
            column_meta_start,
            column_meta_offsets_start,
            global_buff_offsets_start,
            num_global_buffers,
            num_columns,
            major_version,
            minor_version,
        })
    }

    // TODO: Once we have coalesced I/O we should only read the column metadatas that we need
    async fn read_all_column_metadata(
        column_metadata_bytes: Bytes,
        footer: &Footer,
    ) -> Result<Vec<pbfile::ColumnMetadata>> {
        let column_metadata_start = footer.column_meta_start;
        // cmo == column_metadata_offsets
        let cmo_table_size = 16 * footer.num_columns as usize;
        let cmo_table = column_metadata_bytes.slice(column_metadata_bytes.len() - cmo_table_size..);

        (0..footer.num_columns)
            .map(|col_idx| {
                let offset = (col_idx * 16) as usize;
                let position = LittleEndian::read_u64(&cmo_table[offset..offset + 8]);
                let length = LittleEndian::read_u64(&cmo_table[offset + 8..offset + 16]);
                let normalized_position = (position - column_metadata_start) as usize;
                let normalized_end = normalized_position + (length as usize);
                Ok(pbfile::ColumnMetadata::decode(
                    &column_metadata_bytes[normalized_position..normalized_end],
                )?)
            })
            .collect::<Result<Vec<_>>>()
    }

    async fn optimistic_tail_read(
        data: &Bytes,
        start_pos: u64,
        scheduler: &FileScheduler,
        file_len: u64,
    ) -> Result<Bytes> {
        let num_bytes_needed = (file_len - start_pos) as usize;
        if data.len() >= num_bytes_needed {
            Ok(data.slice((data.len() - num_bytes_needed)..))
        } else {
            let num_bytes_missing = (num_bytes_needed - data.len()) as u64;
            let start = file_len - num_bytes_needed as u64;
            let missing_bytes = scheduler
                .submit_single(start..start + num_bytes_missing, 0)
                .await?;
            let mut combined = BytesMut::with_capacity(data.len() + num_bytes_missing as usize);
            combined.extend(missing_bytes);
            combined.extend(data);
            Ok(combined.freeze())
        }
    }

    async fn decode_gbo_table(
        tail_bytes: &Bytes,
        file_len: u64,
        scheduler: &FileScheduler,
        footer: &Footer,
    ) -> Result<Vec<BufferDescriptor>> {
        // This could, in theory, trigger another IOP but the GBO table should never be large
        // enough for that to happen
        let gbo_bytes = Self::optimistic_tail_read(
            tail_bytes,
            footer.global_buff_offsets_start,
            scheduler,
            file_len,
        )
        .await?;
        let mut global_bufs_cursor = Cursor::new(&gbo_bytes);

        let mut global_buffers = Vec::with_capacity(footer.num_global_buffers as usize);
        for _ in 0..footer.num_global_buffers {
            let buf_pos = global_bufs_cursor.read_u64::<LittleEndian>()?;
            let buf_size = global_bufs_cursor.read_u64::<LittleEndian>()?;
            global_buffers.push(BufferDescriptor {
                position: buf_pos,
                size: buf_size,
            });
        }

        Ok(global_buffers)
    }

    fn decode_schema(schema_bytes: Bytes) -> Result<(u64, lance_core::datatypes::Schema)> {
        let file_descriptor = pb::FileDescriptor::decode(schema_bytes)?;
        let pb_schema = file_descriptor.schema.unwrap();
        let num_rows = file_descriptor.length;
        let fields_with_meta = FieldsWithMeta {
            fields: Fields(pb_schema.fields),
            metadata: pb_schema.metadata,
        };
        let schema = lance_core::datatypes::Schema::from(fields_with_meta);
        Ok((num_rows, schema))
    }

    // TODO: Support late projection.  Currently, if we want to perform a
    // projected read of a file, we load all of the column metadata, and then
    // only read the column data that is requested.  This is fine for most cases.
    //
    // However, if there are many columns then loading all of the column metadata
    // may be expensive.  We should support a mode where we only load the column
    // metadata for the columns that are requested (the file format supports this).
    //
    // The main challenge is that we either need to ignore the column metadata cache
    // or have a more sophisticated cache that can cache per-column metadata.
    //
    // Also, if the number of columns is fairly small, it's faster to read them as a
    // single IOP, but we can fix this through coalescing.
    async fn read_all_metadata(scheduler: &FileScheduler) -> Result<CachedFileMetadata> {
        // 1. read the footer
        let (tail_bytes, file_len) = Self::read_tail(scheduler).await?;
        let footer = Self::decode_footer(&tail_bytes)?;

        let gbo_table = Self::decode_gbo_table(&tail_bytes, file_len, scheduler, &footer).await?;
        if gbo_table.is_empty() {
            return Err(Error::Internal {
                message: "File did not contain any global buffers, schema expected".to_string(),
                location: location!(),
            });
        }
        let schema_start = gbo_table[0].position;
        let schema_size = gbo_table[0].size;

        let num_footer_bytes = file_len - schema_start;

        // By default we read all column metadatas.  We do NOT read the column metadata buffers
        // at this point.  We only want to read the column metadata for columns we are actually loading.
        let all_metadata_bytes =
            Self::optimistic_tail_read(&tail_bytes, schema_start, scheduler, file_len).await?;

        let schema_bytes = all_metadata_bytes.slice(0..schema_size as usize);
        let (num_rows, schema) = Self::decode_schema(schema_bytes)?;

        // Next, read the metadata for the columns
        // This is both the column metadata and the CMO table
        let column_metadata_start = (footer.column_meta_start - schema_start) as usize;
        let column_metadata_end = (footer.global_buff_offsets_start - schema_start) as usize;
        let column_metadata_bytes =
            all_metadata_bytes.slice(column_metadata_start..column_metadata_end);
        let column_metadatas =
            Self::read_all_column_metadata(column_metadata_bytes, &footer).await?;

        let footer_start = file_len - FOOTER_LEN as u64;
        let num_data_bytes = footer.column_meta_start;
        let num_global_buffer_bytes = gbo_table.iter().map(|buf| buf.size).sum::<u64>()
            + (footer_start - footer.global_buff_offsets_start);
        let num_column_metadata_bytes = footer.global_buff_offsets_start - footer.column_meta_start;

        let column_infos = Self::meta_to_col_infos(&column_metadatas);

        Ok(CachedFileMetadata {
            file_schema: Arc::new(schema),
            column_metadatas,
            column_infos,
            num_rows,
            num_data_bytes,
            num_column_metadata_bytes,
            num_global_buffer_bytes,
            num_footer_bytes,
            file_buffers: gbo_table,
            major_version: footer.major_version,
            minor_version: footer.minor_version,
        })
    }

    fn fetch_encoding<M: Default + Name + Sized>(encoding: &pbfile::Encoding) -> M {
        match &encoding.location {
            Some(pbfile::encoding::Location::Indirect(_)) => todo!(),
            Some(pbfile::encoding::Location::Direct(encoding)) => {
                let encoding_buf = Bytes::from(encoding.encoding.clone());
                let encoding_any = prost_types::Any::decode(encoding_buf).unwrap();
                encoding_any.to_msg::<M>().unwrap()
            }
            Some(pbfile::encoding::Location::None(_)) => panic!(),
            None => panic!(),
        }
    }

    fn meta_to_col_infos(column_metadatas: &[pbfile::ColumnMetadata]) -> Vec<Arc<ColumnInfo>> {
        column_metadatas
            .iter()
            .enumerate()
            .map(|(col_idx, col_meta)| {
                let page_infos = col_meta
                    .pages
                    .iter()
                    .map(|page| {
                        let num_rows = page.length;
                        let encoding = Self::fetch_encoding(page.encoding.as_ref().unwrap());
                        let buffer_offsets_and_sizes = Arc::new(
                            page.buffer_offsets
                                .iter()
                                .zip(page.buffer_sizes.iter())
                                .map(|(offset, size)| (*offset, *size))
                                .collect(),
                        );
                        PageInfo {
                            buffer_offsets_and_sizes,
                            encoding,
                            num_rows,
                        }
                    })
                    .collect::<Vec<_>>();
                Arc::new(ColumnInfo {
                    index: col_idx as u32,
                    page_infos: Arc::new(page_infos),
                    buffer_offsets_and_sizes: vec![],
                    encoding: Self::fetch_encoding(dbg!(col_meta).encoding.as_ref().unwrap()),
                })
            })
            .collect::<Vec<_>>()
    }

    fn validate_projection(
        projection: &ReaderProjection,
        metadata: &CachedFileMetadata,
    ) -> Result<()> {
        if projection.schema.fields.is_empty() {
            return Err(Error::invalid_input(
                "Attempt to read zero columns from the file, at least one column must be specified"
                    .to_string(),
                location!(),
            ));
        }
        if projection.schema.fields.len() != projection.column_indices.len() {
            return Err(Error::invalid_input(format!("The projection schema has {} top level fields but only {} column indices were provided", projection.schema.fields.len(), projection.column_indices.len()), location!()));
        }
        let mut column_indices_seen = BTreeSet::new();
        for column_index in &projection.column_indices {
            if !column_indices_seen.insert(*column_index) {
                return Err(Error::invalid_input(
                    format!(
                        "The projection specified the column index {} more than once",
                        column_index
                    ),
                    location!(),
                ));
            }
            if *column_index >= metadata.column_infos.len() as u32 {
                return Err(Error::invalid_input(format!("The projection specified the column index {} but there are only {} columns in the file", column_index, metadata.column_infos.len()), location!()));
            }
        }
        Ok(())
    }

    // Helper function for `default_projection` to determine how many columns are occupied
    // by a lance field.
    fn default_column_count(field: &Field) -> u32 {
        if field.data_type().is_binary_like() {
            2
        } else {
            1 + field
                .children
                .iter()
                .map(Self::default_column_count)
                .sum::<u32>()
        }
    }

    // This function is one of the few spots in the reader where we rely on Lance table
    // format and the fact that we wrote a Lance table schema into the global buffers.
    //
    // TODO: In the future it would probably be better for the "default type" of a column
    // to be something that can be provided dynamically via the encodings registry.  We
    // could pass the pages of the column to some logic that picks a data type based on the
    // page encodings.

    /// Loads a default projection for all columns in the file, using the data type that
    /// was provided when the file was written.
    fn default_projection(lance_schema: &Schema) -> ReaderProjection {
        let schema = Arc::new(ArrowSchema::from(lance_schema));
        let mut column_indices = Vec::with_capacity(lance_schema.fields.len());
        let mut column_index = 0;
        for field in &lance_schema.fields {
            column_indices.push(column_index);
            column_index += Self::default_column_count(field);
        }
        ReaderProjection {
            schema,
            column_indices,
        }
    }

    /// Opens a new file reader without any pre-existing knowledge
    ///
    /// This will read the file schema from the file itself and thus requires a bit more I/O
    ///
    /// A `base_projection` can also be provided.  If provided, then the projection will apply
    /// to all reads from the file that do not specify their own projection.
    pub async fn try_open(
        scheduler: FileScheduler,
        base_projection: Option<ReaderProjection>,
    ) -> Result<Self> {
        let file_metadata = Arc::new(Self::read_all_metadata(&scheduler).await?);
        if let Some(base_projection) = base_projection.as_ref() {
            Self::validate_projection(base_projection, &file_metadata)?;
        }
        let num_rows = file_metadata.num_rows;
        Ok(Self {
            scheduler: Arc::new(LanceEncodingsIo(scheduler)),
            base_projection: base_projection
                .unwrap_or(Self::default_projection(file_metadata.file_schema.as_ref())),
            num_rows,
            metadata: file_metadata,
        })
    }

    fn collect_columns(
        &self,
        field: &Field,
        column_idx: &mut usize,
        column_infos: &mut Vec<Arc<ColumnInfo>>,
    ) -> Result<()> {
        column_infos.push(self.metadata.column_infos[*column_idx].clone());
        *column_idx += 1;
        if field.data_type().is_binary_like() {
            // These types are 2 columns in a lance file but a single field id in a lance schema
            column_infos.push(self.metadata.column_infos[*column_idx].clone());
            *column_idx += 1;
        }
        for child in &field.children {
            self.collect_columns(child, column_idx, column_infos)?;
        }
        Ok(())
    }

    // The actual decoder needs all the column infos that make up a type.  In other words, if
    // the first type in the schema is Struct<i32, i32> then the decoder will need 3 column infos.
    //
    // This is a file reader concern because the file reader needs to support late projection of columns
    // and so it will need to figure this out anyways.
    //
    // It's a bit of a tricky process though because the number of column infos may depend on the
    // encoding.  Considering the above example, if we wrote it with a packed encoding, then there would
    // only be a single column in the file (and not 3).
    //
    // At the moment this method words because our rules are simple and we just repeat them here.  See
    // Self::default_projection for a similar problem.  In the future this is something the encodings
    // registry will need to figure out.
    fn collect_columns_from_projection(
        &self,
        projection: &ReaderProjection,
    ) -> Result<Vec<Arc<ColumnInfo>>> {
        let mut column_infos = Vec::with_capacity(projection.column_indices.len());
        for (field, starting_column) in projection
            .schema
            .fields
            .iter()
            .zip(projection.column_indices.iter())
        {
            let mut starting_column = *starting_column as usize;
            let lance_field = Field::try_from(field.as_ref())?;
            self.collect_columns(&lance_field, &mut starting_column, &mut column_infos)?;
        }
        Ok(column_infos)
    }

    fn read_range(
        &self,
        range: Range<u64>,
        batch_size: u32,
        projection: &ReaderProjection,
    ) -> Result<BoxStream<'static, ReadBatchTask>> {
        let column_infos = self.collect_columns_from_projection(projection)?;
        debug!(
            "Reading range {:?} with batch_size {} from columns {:?}",
            range,
            batch_size,
            column_infos.iter().map(|ci| ci.index).collect::<Vec<_>>()
        );
        let mut decode_scheduler = DecodeBatchScheduler::try_new(
            &projection.schema,
            column_infos.iter().map(|ci| ci.as_ref()),
            &vec![],
            &CoreFieldDecoderStrategy,
        )?;

        let root_decoder = decode_scheduler
            .root_scheduler
            .new_root_decoder_ranges(&[range.clone()]);

        let (tx, rx) = mpsc::unbounded_channel();

        let num_rows_to_read = range.end - range.start;

        let scheduler = self.scheduler.clone() as Arc<dyn EncodingsIo>;
        tokio::task::spawn(async move { decode_scheduler.schedule_range(range, tx, scheduler) });

        Ok(BatchDecodeStream::new(rx, batch_size, num_rows_to_read, root_decoder).into_stream())
    }

    fn take_rows(
        &self,
        indices: Vec<u64>,
        batch_size: u32,
        projection: &ReaderProjection,
    ) -> Result<BoxStream<'static, ReadBatchTask>> {
        let column_infos = self.collect_columns_from_projection(projection)?;
        debug!(
            "Taking {} rows spread across range {}..{} with batch_size {} from columns {:?}",
            indices.len(),
            indices[0],
            indices[indices.len() - 1],
            batch_size,
            column_infos.iter().map(|ci| ci.index).collect::<Vec<_>>()
        );
        let mut decode_scheduler = DecodeBatchScheduler::try_new(
            &projection.schema,
            column_infos.iter().map(|ci| ci.as_ref()),
            &vec![],
            &CoreFieldDecoderStrategy,
        )?;

        let root_decoder = decode_scheduler
            .root_scheduler
            .new_root_decoder_indices(&indices);

        let (tx, rx) = mpsc::unbounded_channel();

        let num_rows_to_read = indices.len() as u64;

        let scheduler = self.scheduler.clone() as Arc<dyn EncodingsIo>;
        tokio::task::spawn(async move { decode_scheduler.schedule_take(&indices, tx, scheduler) });

        Ok(BatchDecodeStream::new(rx, batch_size, num_rows_to_read, root_decoder).into_stream())
    }

    /// Creates a stream of "read tasks" to read the data from the file
    ///
    /// The arguments are similar to [`Self::read_stream_projected`] but instead of returning a stream
    /// of record batches it returns a stream of "read tasks".
    ///
    /// The tasks should be consumed with some kind of `buffered` argument if CPU parallelism is desired.
    ///
    /// Note that "read task" is probably a bit imprecise.  The tasks are actually "decode tasks".  The
    /// reading happens asynchronously in the background.  In other words, a single read task may map to
    /// multiple I/O operations or a single I/O operation may map to multiple read tasks.
    pub fn read_tasks(
        &self,
        params: ReadBatchParams,
        batch_size: u32,
        projection: &ReaderProjection,
    ) -> Result<Pin<Box<dyn Stream<Item = ReadBatchTask> + Send>>> {
        Self::validate_projection(projection, &self.metadata)?;
        let verify_bound = |params: &ReadBatchParams, bound: u64, inclusive: bool| {
            if bound > self.num_rows || bound == self.num_rows && inclusive {
                Err(Error::invalid_input(
                    format!(
                        "cannot read {:?} from file with {} rows",
                        params, self.num_rows
                    ),
                    location!(),
                ))
            } else {
                Ok(())
            }
        };
        match &params {
            ReadBatchParams::Indices(indices) => {
                for idx in indices {
                    match idx {
                        None => {
                            return Err(Error::invalid_input(
                                "Null value in indices array",
                                location!(),
                            ));
                        }
                        Some(idx) => {
                            verify_bound(&params, idx as u64, true)?;
                        }
                    }
                }
                let indices = indices.iter().map(|idx| idx.unwrap() as u64).collect();
                self.take_rows(indices, batch_size, projection)
            }
            ReadBatchParams::Range(range) => {
                verify_bound(&params, range.end as u64, false)?;
                self.read_range(range.start as u64..range.end as u64, batch_size, projection)
            }
            ReadBatchParams::RangeFrom(range) => {
                verify_bound(&params, range.start as u64, true)?;
                self.read_range(range.start as u64..self.num_rows, batch_size, projection)
            }
            ReadBatchParams::RangeTo(range) => {
                verify_bound(&params, range.end as u64, false)?;
                self.read_range(0..range.end as u64, batch_size, projection)
            }
            ReadBatchParams::RangeFull => self.read_range(0..self.num_rows, batch_size, projection),
        }
    }

    /// Reads data from the file as a stream of record batches
    ///
    /// * `params` - Specifies the range (or indices) of data to read
    /// * `batch_size` - The maximum size of a single batch.  A batch may be smaller
    ///   if it is the last batch or if it is not possible to create a batch of the
    ///   requested size.
    ///
    ///   For example, if the batch size is 1024 and one of the columns is a string
    ///   column then there may be some ranges of 1024 rows that contain more than
    ///   2^31 bytes of string data (which is the maximum size of a string column
    ///   in Arrow).  In this case smaller batches may be emitted.
    /// * `batch_readahead` - The number of batches to read ahead.  This controls the
    ///   amount of CPU parallelism of the read.  In other words it controlls how many
    ///   batches will be decoded in parallel.  It has no effect on the I/O parallelism
    ///   of the read (how many I/O requests are in flight at once).
    ///
    ///   This parameter also is also related to backpressure.  If the consumer of the
    ///   stream is slow then the reader will build up RAM.
    /// * `projection` - A projection to apply to the read.  This controls which columns
    ///   are read from the file.  The projection is NOT applied on top of the base
    ///   projection.  The projection is applied directly to the file schema.
    pub fn read_stream_projected(
        &self,
        params: ReadBatchParams,
        batch_size: u32,
        batch_readahead: u32,
        projection: &ReaderProjection,
    ) -> Result<Pin<Box<dyn RecordBatchStream>>> {
        let tasks_stream = self.read_tasks(params, batch_size, projection)?;
        let batch_stream = tasks_stream
            .map(|task| task.task)
            .buffered(batch_readahead as usize)
            .boxed();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            projection.schema.clone(),
            batch_stream,
        )))
    }

    /// Reads data from the file as a stream of record batches
    ///
    /// This is similar to [`Self::read_stream_projected`] but uses the base projection
    /// provided when the file was opened (or reads all columns if the file was
    /// opened without a base projection)
    pub fn read_stream(
        &self,
        params: ReadBatchParams,
        batch_size: u32,
        batch_readahead: u32,
    ) -> Result<Pin<Box<dyn RecordBatchStream>>> {
        self.read_stream_projected(params, batch_size, batch_readahead, &self.base_projection)
    }

    pub fn schema(&self) -> Arc<Schema> {
        self.metadata.file_schema.clone()
    }
}

/// Inspects a page and returns a String describing the page's encoding
pub fn describe_encoding(page: &pbfile::column_metadata::Page) -> String {
    if let Some(encoding) = &page.encoding {
        if let Some(style) = &encoding.location {
            match style {
                pbfile::encoding::Location::Indirect(indirect) => {
                    format!(
                        "IndirectEncoding(pos={},size={})",
                        indirect.buffer_location, indirect.buffer_length
                    )
                }
                pbfile::encoding::Location::Direct(direct) => {
                    let encoding_any =
                        prost_types::Any::decode(Bytes::from(direct.encoding.clone()))
                            .expect("failed to deserialize encoding as protobuf");
                    if encoding_any.type_url == "/lance.encodings.ArrayEncoding" {
                        let encoding = encoding_any.to_msg::<pbenc::ArrayEncoding>();
                        match encoding {
                            Ok(encoding) => {
                                format!("{:#?}", encoding)
                            }
                            Err(err) => {
                                format!("Unsupported(decode_err={})", err)
                            }
                        }
                    } else {
                        format!("Unrecognized(type_url={})", encoding_any.type_url)
                    }
                }
                pbfile::encoding::Location::None(_) => "NoEncodingDescription".to_string(),
            }
        } else {
            "MISSING STYLE".to_string()
        }
    } else {
        "MISSING".to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::{pin::Pin, sync::Arc};

    use arrow_array::{types::Float64Type, RecordBatch, RecordBatchReader};
    use arrow_schema::{ArrowError, DataType, Field, Fields, Schema as ArrowSchema};
    use futures::StreamExt;
    use lance_arrow::RecordBatchExt;
    use lance_core::datatypes::Schema;
    use lance_datagen::{array, gen, BatchCount, RowCount};
    use lance_io::{
        object_store::ObjectStore, scheduler::ScanScheduler, stream::RecordBatchStream,
    };
    use log::debug;
    use object_store::path::Path;
    use tempfile::TempDir;

    use crate::v2::{
        reader::{FileReader, ReaderProjection},
        writer::{FileWriter, FileWriterOptions},
    };

    struct FsFixture {
        _tmp_dir: TempDir,
        tmp_path: Path,
        object_store: Arc<ObjectStore>,
        scheduler: Arc<ScanScheduler>,
    }

    impl Default for FsFixture {
        fn default() -> Self {
            let tmp_dir = tempfile::tempdir().unwrap();
            let tmp_path: String = tmp_dir.path().to_str().unwrap().to_owned();
            let tmp_path = Path::parse(tmp_path).unwrap();
            let tmp_path = tmp_path.child("some_file.lance");
            let object_store = Arc::new(ObjectStore::local());
            let scheduler = ScanScheduler::new(object_store.clone(), 8);
            Self {
                _tmp_dir: tmp_dir,
                object_store,
                tmp_path,
                scheduler,
            }
        }
    }

    async fn create_some_file(
        object_store: &ObjectStore,
        path: &Path,
    ) -> (Arc<Schema>, Vec<RecordBatch>) {
        let location_type = DataType::Struct(Fields::from(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
        ]));
        let categories_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));

        let reader = gen()
            .col("score", array::rand::<Float64Type>())
            .col("location", array::rand_type(&location_type))
            .col("categories", array::rand_type(&categories_type))
            .into_reader_rows(RowCount::from(1000), BatchCount::from(100));

        let writer = object_store.create(path).await.unwrap();

        let lance_schema =
            lance_core::datatypes::Schema::try_from(reader.schema().as_ref()).unwrap();

        let mut file_writer = FileWriter::try_new(
            writer,
            path.to_string(),
            lance_schema.clone(),
            FileWriterOptions::default(),
        )
        .unwrap();

        let data = reader
            .collect::<std::result::Result<Vec<_>, ArrowError>>()
            .unwrap();

        for batch in &data {
            file_writer.write_batch(batch).await.unwrap();
        }
        file_writer.add_schema_metadata("foo", "bar");
        file_writer.finish().await.unwrap();
        (Arc::new(lance_schema), data)
    }

    type Transformer = Box<dyn Fn(&RecordBatch) -> RecordBatch>;

    async fn verify_expected(
        expected: &[RecordBatch],
        mut actual: Pin<Box<dyn RecordBatchStream>>,
        read_size: u32,
        transform: Option<Transformer>,
    ) {
        let mut remaining = expected.iter().map(|batch| batch.num_rows()).sum::<usize>() as u32;
        let mut expected_iter = expected.iter().map(|batch| {
            if let Some(transform) = &transform {
                transform(batch)
            } else {
                batch.clone()
            }
        });
        let mut next_expected = expected_iter.next().unwrap().clone();
        while let Some(actual) = actual.next().await {
            let mut actual = actual.unwrap();
            let mut rows_to_verify = actual.num_rows() as u32;
            let expected_length = remaining.min(read_size);
            assert_eq!(expected_length, rows_to_verify);

            while rows_to_verify > 0 {
                let next_slice_len = (next_expected.num_rows() as u32).min(rows_to_verify);
                assert_eq!(
                    next_expected.slice(0, next_slice_len as usize),
                    actual.slice(0, next_slice_len as usize)
                );
                remaining -= next_slice_len;
                rows_to_verify -= next_slice_len;
                if remaining > 0 {
                    if next_slice_len == next_expected.num_rows() as u32 {
                        next_expected = expected_iter.next().unwrap().clone();
                    } else {
                        next_expected = next_expected.slice(
                            next_slice_len as usize,
                            next_expected.num_rows() - next_slice_len as usize,
                        );
                    }
                }
                if rows_to_verify > 0 {
                    actual = actual.slice(
                        next_slice_len as usize,
                        actual.num_rows() - next_slice_len as usize,
                    );
                }
            }
        }
        assert_eq!(remaining, 0);
    }

    #[tokio::test]
    async fn test_round_trip() {
        let fs = FsFixture::default();

        let (_, data) = create_some_file(&fs.object_store, &fs.tmp_path).await;

        for read_size in [32, 1024, 1024 * 1024] {
            let file_scheduler = fs.scheduler.open_file(&fs.tmp_path).await.unwrap();
            let file_reader = FileReader::try_open(file_scheduler, None).await.unwrap();

            let schema = file_reader.schema();
            assert_eq!(schema.metadata.get("foo").unwrap(), "bar");

            let batch_stream = file_reader
                .read_stream(lance_io::ReadBatchParams::RangeFull, read_size, 16)
                .unwrap();

            verify_expected(&data, batch_stream, read_size, None).await;
        }
    }

    #[test_log::test(tokio::test)]
    async fn test_projection() {
        let fs = FsFixture::default();

        let (schema, data) = create_some_file(&fs.object_store, &fs.tmp_path).await;
        let file_scheduler = fs.scheduler.open_file(&fs.tmp_path).await.unwrap();

        for columns in [
            vec!["score"],
            vec!["location"],
            vec!["categories"],
            vec!["score.x"],
            vec!["score", "categories"],
            vec!["score", "location"],
            vec!["location", "categories"],
            vec!["score.y", "location", "categories"],
        ] {
            debug!("Testing round trip with projection {:?}", columns);
            // We can specify the projection as part of the read operation via read_stream_projected
            let file_reader = FileReader::try_open(file_scheduler.clone(), None)
                .await
                .unwrap();

            let projection = schema.project(&columns).unwrap();
            let projection_arrow = Arc::new(ArrowSchema::from(&projection));
            let projection = ReaderProjection {
                schema: projection_arrow,
                column_indices: projection.fields.iter().map(|f| f.id as u32).collect(),
            };

            let batch_stream = file_reader
                .read_stream_projected(lance_io::ReadBatchParams::RangeFull, 1024, 16, &projection)
                .unwrap();

            let projection_copy = projection.clone();
            verify_expected(
                &data,
                batch_stream,
                1024,
                Some(Box::new(move |batch: &RecordBatch| {
                    batch.project_by_schema(&projection_copy.schema).unwrap()
                })),
            )
            .await;

            // We can also specify the projection as a base projection when we open the file
            let file_reader =
                FileReader::try_open(file_scheduler.clone(), Some(projection.clone()))
                    .await
                    .unwrap();

            let batch_stream = file_reader
                .read_stream(lance_io::ReadBatchParams::RangeFull, 1024, 16)
                .unwrap();

            let projection_copy = projection.clone();
            verify_expected(
                &data,
                batch_stream,
                1024,
                Some(Box::new(move |batch: &RecordBatch| {
                    batch.project_by_schema(&projection_copy.schema).unwrap()
                })),
            )
            .await;
        }

        let empty_projection = ReaderProjection {
            column_indices: Vec::default(),
            schema: Arc::new(ArrowSchema::new(Vec::<Field>::default())),
        };

        assert!(
            FileReader::try_open(file_scheduler.clone(), Some(empty_projection))
                .await
                .is_err()
        );

        let projection_with_dupes = ReaderProjection {
            column_indices: vec![0, 0],
            schema: Arc::new(ArrowSchema::new(vec![
                Field::new("x", DataType::Int32, true),
                Field::new("y", DataType::Int32, true),
            ])),
        };

        assert!(
            FileReader::try_open(file_scheduler.clone(), Some(projection_with_dupes))
                .await
                .is_err()
        );
    }

    struct EnvVarGuard {
        key: String,
        original_value: Option<String>,
    }

    impl EnvVarGuard {
        fn new(key: &str, new_value: &str) -> Self {
            let original_value = std::env::var(key).ok();
            std::env::set_var(key, new_value);
            Self {
                key: key.to_string(),
                original_value,
            }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(ref value) = self.original_value {
                std::env::set_var(&self.key, value);
            } else {
                std::env::remove_var(&self.key);
            }
        }
    }

    #[test_log::test(tokio::test)]
    async fn test_compressing_buffer() {
        let fs = FsFixture::default();
        // set env var temporarily to test compressed page
        let _env_guard = EnvVarGuard::new("LANCE_PAGE_COMPRESSION", "zstd");

        let (schema, data) = create_some_file(&fs.object_store, &fs.tmp_path).await;
        let file_scheduler = fs.scheduler.open_file(&fs.tmp_path).await.unwrap();

        // We can specify the projection as part of the read operation via read_stream_projected
        let file_reader = FileReader::try_open(file_scheduler.clone(), None)
            .await
            .unwrap();

        let projection = schema.project(&["score"]).unwrap();
        let projection_arrow = Arc::new(ArrowSchema::from(&projection));
        let projection = ReaderProjection {
            schema: projection_arrow,
            column_indices: projection.fields.iter().map(|f| f.id as u32).collect(),
        };

        let batch_stream = file_reader
            .read_stream_projected(lance_io::ReadBatchParams::RangeFull, 1024, 16, &projection)
            .unwrap();

        let projection_copy = projection.clone();
        verify_expected(
            &data,
            batch_stream,
            1024,
            Some(Box::new(move |batch: &RecordBatch| {
                batch.project_by_schema(&projection_copy.schema).unwrap()
            })),
        )
        .await;
    }
}
