// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{io::Cursor, ops::Range, pin::Pin, sync::Arc};

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use bytes::{Bytes, BytesMut};
use futures::{stream::BoxStream, StreamExt};
use lance_encoding::{
    decoder::{BatchDecodeStream, ColumnInfo, DecodeBatchScheduler, PageInfo},
    EncodingsIo,
};
use prost::Message;
use snafu::{location, Location};

use lance_core::{Error, Result};
use lance_encoding::format::pb as pbenc;
use lance_io::{
    scheduler::FileScheduler,
    stream::{RecordBatchStream, RecordBatchStreamAdapter},
    ReadBatchParams,
};
use tokio::{sync::mpsc, task::JoinHandle};

use crate::{
    datatypes::{Fields, FieldsWithMeta},
    format::{pb, pbfile, MAGIC, MAJOR_VERSION, MINOR_VERSION_NEXT},
};

use super::io::LanceEncodingsIo;

// For now, we don't use global buffers for anything other than schema.  If we
// use these later we should make them lazily loaded and then cached once loaded.
//
// We store their position / length for debugging purposes
pub struct BufferDescriptor {
    pub position: u64,
    pub size: u64,
}

// TODO: Caching
pub struct CachedFileMetadata {
    /// The schema of the file
    pub file_schema: Schema,
    /// The column metadatas
    pub column_metadatas: Vec<pbfile::ColumnMetadata>,
    /// The number of rows in the file
    pub num_rows: u64,
    pub file_buffers: Vec<BufferDescriptor>,
    /// The number of bytes contained in the data page section of the file
    pub num_data_bytes: u64,
    /// The number of bytes contained in the column metadata section of the file
    pub num_column_metadata_bytes: u64,
    /// The number of bytes contained in the global buffer section of the file
    pub num_global_buffer_bytes: u64,
    pub major_version: u16,
    pub minor_version: u16,
}

pub struct FileReader {
    scheduler: Arc<LanceEncodingsIo>,
    file_schema: Schema,
    column_infos: Vec<ColumnInfo>,
    num_rows: u64,
    metadata: Arc<CachedFileMetadata>,
}

struct Footer {
    column_meta_start: u64,
    // We don't use this today because we always load metadata for every column
    // and don't yet support "metadata projection"
    #[allow(dead_code)]
    column_meta_offsets_start: u64,
    global_buff_start: u64,
    global_buff_offsets_start: u64,
    num_global_buffers: u32,
    num_columns: u32,
    major_version: u16,
    minor_version: u16,
}

const FOOTER_LEN: usize = 48;

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
        let tail_bytes = scheduler.submit_single(begin..file_size).await?;
        Ok((tail_bytes, file_size))
    }

    // Checks to make sure the footer is written correctly and returns the
    // position of the file descriptor (which comes from the footer)
    fn decode_footer(footer_bytes: &Bytes) -> Result<Footer> {
        let len = footer_bytes.len();
        if len < FOOTER_LEN {
            return Err(Error::IO {
                message: format!(
                    "does not have sufficient data, len: {}, bytes: {:?}",
                    len, footer_bytes
                ),
                location: location!(),
            });
        }
        let mut cursor = Cursor::new(footer_bytes.slice(len - FOOTER_LEN..));

        let column_meta_start = cursor.read_u64::<LittleEndian>()?;
        let column_meta_offsets_start = cursor.read_u64::<LittleEndian>()?;
        let global_buff_start = cursor.read_u64::<LittleEndian>()?;
        let global_buff_offsets_start = cursor.read_u64::<LittleEndian>()?;
        let num_global_buffers = cursor.read_u32::<LittleEndian>()?;
        let num_columns = cursor.read_u32::<LittleEndian>()?;
        let major_version = cursor.read_u16::<LittleEndian>()?;
        let minor_version = cursor.read_u16::<LittleEndian>()?;

        if major_version != MAJOR_VERSION as u16 || minor_version != MINOR_VERSION_NEXT {
            return Err(Error::IO {
                message: format!(
                    "Attempt to use the lance v0.2 reader to read a file with version {}.{}",
                    major_version, minor_version
                ),
                location: location!(),
            });
        }

        let magic_bytes = footer_bytes.slice(len - 4..);
        if magic_bytes.as_ref() != MAGIC {
            return Err(Error::IO {
                message: format!(
                    "file does not appear to be a Lance file (invalid magic: {:?})",
                    MAGIC
                ),
                location: location!(),
            });
        }
        Ok(Footer {
            column_meta_start,
            column_meta_offsets_start,
            global_buff_start,
            global_buff_offsets_start,
            num_global_buffers,
            num_columns,
            major_version,
            minor_version,
        })
    }

    // TODO: Once we have coalesced I/O we should only read the column metadatas that we need
    async fn read_all_column_metadata(
        scheduler: &FileScheduler,
        footer: &Footer,
    ) -> Result<Vec<pbfile::ColumnMetadata>> {
        let column_metadata_start = footer.column_meta_start;
        // This range includes both the offsets table and all of the column metadata
        // We can't just grab col_meta_start..cmo_table_start because there may be padding
        // between the last column and the start of the cmo table.
        let column_metadata_range = column_metadata_start..footer.global_buff_start;
        let column_metadata_bytes = scheduler.submit_single(column_metadata_range).await?;

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

    async fn get_all_meta_bytes(
        tail_bytes: Bytes,
        file_len: u64,
        scheduler: &FileScheduler,
        footer: &Footer,
    ) -> Result<Bytes> {
        let num_bytes_needed = (file_len - footer.column_meta_start) as usize;
        if tail_bytes.len() >= num_bytes_needed {
            Ok(tail_bytes.slice(tail_bytes.len() - num_bytes_needed..))
        } else {
            let num_bytes_missing = (num_bytes_needed - tail_bytes.len()) as u64;
            let missing_bytes = scheduler
                .submit_single(
                    footer.column_meta_start..footer.column_meta_start + num_bytes_missing,
                )
                .await;
            let mut combined =
                BytesMut::with_capacity(tail_bytes.len() + num_bytes_missing as usize);
            combined.extend(missing_bytes);
            combined.extend(tail_bytes);
            Ok(combined.freeze())
        }
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

    // TODO: Support projection
    async fn read_all_metadata(
        scheduler: &FileScheduler,
        _projection: &Schema,
    ) -> Result<CachedFileMetadata> {
        // 1. read the footer
        let (tail_bytes, file_len) = Self::read_tail(scheduler).await?;
        let footer = Self::decode_footer(&tail_bytes)?;

        let all_metadata_bytes =
            Self::get_all_meta_bytes(tail_bytes, file_len, scheduler, &footer).await?;
        let meta_offset = footer.column_meta_start;

        // 2. read any global buffers (just the schema right now)
        let global_bufs_table_nbytes = footer.num_global_buffers as usize * 16;
        let global_bufs_table_start = (footer.global_buff_offsets_start - meta_offset) as usize;
        let global_bufs_table_end = global_bufs_table_start + global_bufs_table_nbytes;
        let global_bufs_table =
            all_metadata_bytes.slice(global_bufs_table_start..global_bufs_table_end);
        let mut global_bufs_cursor = Cursor::new(&global_bufs_table);
        let schema_pos = global_bufs_cursor.read_u64::<LittleEndian>()? - meta_offset;
        let schema_size = global_bufs_cursor.read_u64::<LittleEndian>()?;
        let schema_end = schema_pos + schema_size;
        let schema_bytes = all_metadata_bytes.slice(schema_pos as usize..schema_end as usize);
        let (num_rows, schema) = Self::decode_schema(schema_bytes)?;

        // Next, read the metadata for the columns
        let column_metadatas = Self::read_all_column_metadata(scheduler, &footer).await?;

        let footer_start = file_len - FOOTER_LEN as u64;
        let num_data_bytes = footer.column_meta_start;
        let num_column_metadata_bytes = footer.global_buff_start - footer.column_meta_start;
        let num_global_buffer_bytes = footer_start - footer.global_buff_start;

        let global_bufs_table_nbytes = footer.num_global_buffers as usize * 16;
        let global_bufs_table_start = (footer.global_buff_offsets_start - meta_offset) as usize;
        let global_bufs_table_end = global_bufs_table_start + global_bufs_table_nbytes;
        let global_bufs_table =
            all_metadata_bytes.slice(global_bufs_table_start..global_bufs_table_end);
        let mut global_bufs_cursor = Cursor::new(&global_bufs_table);

        let mut global_buffers = Vec::with_capacity(footer.num_global_buffers as usize);
        for _ in 0..footer.num_global_buffers {
            let buf_pos = global_bufs_cursor.read_u64::<LittleEndian>()? - meta_offset;
            let buf_size = global_bufs_cursor.read_u64::<LittleEndian>()?;
            global_buffers.push(BufferDescriptor {
                position: buf_pos,
                size: buf_size,
            });
        }

        Ok(CachedFileMetadata {
            file_schema: Schema::from(&schema),
            column_metadatas,
            num_rows,
            num_data_bytes,
            num_column_metadata_bytes,
            num_global_buffer_bytes,
            file_buffers: global_buffers,
            major_version: footer.major_version,
            minor_version: footer.minor_version,
        })
    }

    pub async fn print_all_metadata(metadata: &CachedFileMetadata) -> Result<()> {
        // 1. read and print the footer
        println!("# Footer");
        println!();
        println!(
            "File version           : {}.{}",
            MAJOR_VERSION, MINOR_VERSION_NEXT
        );
        println!("Data bytes             : {}", metadata.num_data_bytes);
        println!("Col. meta bytes: {}", metadata.num_column_metadata_bytes);
        println!("Glo. data bytes: {}", metadata.num_global_buffer_bytes);

        // 2. print the global buffers
        println!("Global buffers:");
        for file_buffer in &metadata.file_buffers {
            println!(
                " * {}..{}",
                file_buffer.position,
                file_buffer.position + file_buffer.size
            );
        }

        println!("Columns:");
        for (idx, col) in metadata.column_metadatas.iter().enumerate() {
            println!(" * Column {}", idx);
            println!();
            println!("   Buffers:");
            for idx in 0..col.buffer_offsets.len() {
                println!(
                    "    * {}..{}",
                    col.buffer_offsets[idx],
                    col.buffer_offsets[idx] + col.buffer_sizes[idx]
                );
            }
            println!("   Pages:");
            println!();
            for (page_idx, page) in col.pages.iter().enumerate() {
                println!("   * Page {}", page_idx);
                println!();
                println!("     Buffers:");
                for buf_idx in 0..page.buffer_offsets.len() {
                    println!(
                        "      * {}..{}",
                        page.buffer_offsets[buf_idx],
                        page.buffer_offsets[buf_idx] + page.buffer_sizes[buf_idx]
                    );
                }
                let encoding = page.encoding.as_ref().unwrap();
                let encoding = Self::fetch_encoding(encoding);
                println!("     Encoding:");
                println!();
                let encoding_dbg = format!("{:#?}", encoding);
                for line in encoding_dbg.lines() {
                    println!("       {}", line);
                }
                println!();
            }
        }

        Ok(())
    }

    fn fetch_encoding(encoding: &pbfile::Encoding) -> pbenc::ArrayEncoding {
        match &encoding.style {
            Some(pbfile::encoding::Style::Deferred(_)) => todo!(),
            Some(pbfile::encoding::Style::Direct(encoding)) => encoding
                .encoding
                .as_ref()
                .unwrap()
                .to_msg::<pbenc::ArrayEncoding>()
                .unwrap(),
            None => panic!(),
        }
    }

    fn meta_to_col_infos(meta: &CachedFileMetadata) -> Vec<ColumnInfo> {
        meta.column_metadatas
            .iter()
            .map(|col_meta| {
                let page_infos = col_meta
                    .pages
                    .iter()
                    .map(|page| {
                        let num_rows = page.length;
                        let buffer_offsets = Arc::new(page.buffer_offsets.clone());
                        let encoding = Self::fetch_encoding(page.encoding.as_ref().unwrap());
                        Arc::new(PageInfo {
                            buffer_offsets,
                            encoding,
                            num_rows,
                        })
                    })
                    .collect::<Vec<_>>();
                ColumnInfo {
                    page_infos,
                    buffer_offsets: vec![],
                }
            })
            .collect::<Vec<_>>()
    }

    /// Opens a new file reader without any pre-existing knowledge
    ///
    /// This will read the file schema and desired column metadata from the
    pub async fn try_open(scheduler: FileScheduler, projection: Schema) -> Result<Self> {
        let file_metadata = Arc::new(Self::read_all_metadata(&scheduler, &projection).await?);
        let column_infos = Self::meta_to_col_infos(&file_metadata);
        let num_rows = file_metadata.num_rows;
        Ok(Self {
            scheduler: Arc::new(LanceEncodingsIo(scheduler)),
            column_infos,
            file_schema: file_metadata.file_schema.clone(),
            num_rows,
            metadata: file_metadata,
        })
    }

    async fn read_range(
        &self,
        range: Range<u64>,
        batch_size: u32,
    ) -> BoxStream<'static, JoinHandle<Result<RecordBatch>>> {
        let mut decode_scheduler =
            DecodeBatchScheduler::new(&self.file_schema, &self.column_infos, &vec![]);

        let (tx, rx) = mpsc::unbounded_channel();

        let scheduler = self.scheduler.clone() as Arc<dyn EncodingsIo>;
        // FIXME: spawn this, change this method to sync
        decode_scheduler
            .schedule_range(range, tx, &scheduler)
            .await
            .unwrap();

        BatchDecodeStream::new(rx, batch_size, self.num_rows).into_stream()
    }

    // TODO: change output to sendable record batch stream

    pub async fn read_stream(
        &self,
        params: ReadBatchParams,
        batch_size: u32,
    ) -> Pin<Box<dyn RecordBatchStream>> {
        let futures_stream = match params {
            ReadBatchParams::Indices(_) => todo!(),
            ReadBatchParams::Range(range) => {
                // TODO: Make err
                assert!((range.end as u64) < self.num_rows);
                self.read_range(range.start as u64..range.end as u64, batch_size)
                    .await
            }
            ReadBatchParams::RangeFrom(range) => {
                self.read_range(range.start as u64..self.num_rows, batch_size)
                    .await
            }
            ReadBatchParams::RangeTo(range) => {
                // TODO: Make err
                assert!((range.end as u64) < self.num_rows);
                self.read_range(0..range.end as u64, batch_size).await
            }
            ReadBatchParams::RangeFull => self.read_range(0..self.num_rows, batch_size).await,
        };
        let batch_stream = futures_stream
            .buffered(16)
            // JoinHandle returns Result<Result<...>> where the outer Result means the thread
            // task panic'd so we propagate that panic here.
            .map(|res_res| res_res.unwrap())
            .boxed();
        Box::pin(RecordBatchStreamAdapter::new(
            Arc::new(self.file_schema.clone()),
            batch_stream,
        ))
    }
}

/// Inspects a page and returns a String describing the page's encoding
pub fn describe_encoding(page: &pbfile::column_metadata::Page) -> String {
    if let Some(encoding) = &page.encoding {
        if let Some(style) = &encoding.style {
            match style {
                pbfile::encoding::Style::Deferred(deferred) => {
                    format!(
                        "DeferredEncoding(pos={},size={})",
                        deferred.buffer_location, deferred.buffer_length
                    )
                }
                pbfile::encoding::Style::Direct(direct) => {
                    if let Some(encoding) = &direct.encoding {
                        if encoding.type_url == "/lance.encodings.ArrayEncoding" {
                            let encoding = encoding.to_msg::<pbenc::ArrayEncoding>();
                            match encoding {
                                Ok(encoding) => {
                                    format!("{:#?}", encoding)
                                }
                                Err(err) => {
                                    format!("Unsupported(decode_err={})", err)
                                }
                            }
                        } else {
                            format!("Unrecognized(type_url={})", encoding.type_url)
                        }
                    } else {
                        "MISSING DIRECT VALUE".to_string()
                    }
                }
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
    use std::sync::Arc;

    use arrow_array::{types::Float64Type, RecordBatchReader};
    use arrow_schema::ArrowError;
    use futures::StreamExt;
    use lance_datagen::{array, gen, BatchCount, RowCount};
    use lance_io::{object_store::ObjectStore, scheduler::StoreScheduler};
    use object_store::path::Path;

    use crate::v2::{
        reader::FileReader,
        writer::{FileWriter, FileWriterOptions},
    };

    #[tokio::test]
    async fn test_round_trip() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path: String = tmp_dir.path().to_str().unwrap().to_owned();
        let tmp_path = Path::parse(tmp_path).unwrap();
        let tmp_path = tmp_path.child("some_file.lance");
        let obj_store = Arc::new(ObjectStore::local());
        let scheduler = StoreScheduler::new(obj_store.clone(), 8);

        let reader = gen()
            .col(Some("score".to_string()), array::rand::<Float64Type>())
            .into_reader_rows(RowCount::from(1000), BatchCount::from(100));

        let writer = obj_store.create(&tmp_path).await.unwrap();

        let mut file_writer = FileWriter::try_new(
            writer,
            (*reader.schema()).clone(),
            FileWriterOptions::default(),
        )
        .unwrap();

        let schema = reader.schema();
        let data = reader
            .collect::<std::result::Result<Vec<_>, ArrowError>>()
            .unwrap();

        for batch in &data {
            file_writer.write_batch(batch).await.unwrap();
        }
        file_writer.finish().await.unwrap();

        for read_size in [32, 1024, 1024 * 1024] {
            let file_scheduler = scheduler.open_file(&tmp_path).await.unwrap();
            let file_reader = FileReader::try_open(file_scheduler, (*schema).clone())
                .await
                .unwrap();

            let mut batch_stream = file_reader
                .read_stream(lance_io::ReadBatchParams::RangeFull, read_size)
                .await;

            let mut total_remaining = 1000 * 100;
            let mut expected_iter = data.iter();
            let mut next_expected = expected_iter.next().unwrap().clone();
            while let Some(actual) = batch_stream.next().await {
                let mut actual = actual.unwrap();
                let mut rows_to_verify = actual.num_rows() as u32;
                let expected_length = total_remaining.min(read_size);
                assert_eq!(expected_length, rows_to_verify);

                while rows_to_verify > 0 {
                    let next_slice_len = (next_expected.num_rows() as u32).min(rows_to_verify);
                    assert_eq!(
                        next_expected.slice(0, next_slice_len as usize),
                        actual.slice(0, next_slice_len as usize)
                    );
                    total_remaining -= next_slice_len;
                    rows_to_verify -= next_slice_len;
                    if total_remaining > 0 {
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
            debug_assert_eq!(total_remaining, 0);
        }
    }
}
