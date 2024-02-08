use std::{io::Cursor, ops::Range, sync::Arc};

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use bytes::{Bytes, BytesMut};
use futures::{stream::BoxStream, StreamExt};
use lance_encoding::{
    decoder::{
        BatchDecodeStream, ColumnInfo, DecodeBatchScheduler, PageInfo, PhysicalPageScheduler,
    },
    encodings::physical::{basic::BasicPageScheduler, value::ValuePageScheduler},
    EncodingsIo,
};
use prost::Message;
use snafu::{location, Location};

use lance_core::{Error, Result};
use lance_encoding::format::pb as pbenc;
use lance_io::{scheduler::FileScheduler, ReadBatchParams};
use tokio::{sync::mpsc, task::JoinHandle};

use crate::{
    datatypes::{Fields, FieldsWithMeta},
    format::{pb, pbfile, MAGIC, MAJOR_VERSION, MINOR_VERSION_TWO},
};

use super::io::LanceEncodingsIo;

// TODO: Caching
pub struct CachedFileMetadata {
    /// The schema of the file
    pub file_schema: Schema,
    /// The column metadatas
    pub column_metadatas: Vec<pbfile::ColumnMetadata>,
    /// The number of rows in the file
    pub num_rows: u64,
}

pub struct FileReader {
    scheduler: Arc<LanceEncodingsIo>,
    file_schema: Schema,
    column_infos: Vec<ColumnInfo>,
    num_rows: u64,
}

struct Footer {
    column_meta_start: u64,
    // We don't use this today because we get here by subtracting backward from global_buf_start
    #[allow(dead_code)]
    column_meta_offsets_start: u64,
    global_buff_start: u64,
    global_buff_offsets_start: u64,
    num_global_buffers: u32,
    num_columns: u32,
}

const FOOTER_LEN: usize = 48;

impl FileReader {
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

        if major_version != MAJOR_VERSION as u16 || minor_version != MINOR_VERSION_TWO {
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

        Ok(CachedFileMetadata {
            file_schema: Schema::from(&schema),
            column_metadatas,
            num_rows,
        })
    }

    pub async fn print_all_metadata(scheduler: &FileScheduler) -> Result<()> {
        // 1. read and print the footer
        let (tail_bytes, file_len) = Self::read_tail(scheduler).await?;
        let footer = Self::decode_footer(&tail_bytes)?;

        println!("# Footer");
        println!();
        println!(
            "File version           : {}.{}",
            MAJOR_VERSION, MINOR_VERSION_TWO
        );
        println!("Data bytes             : {}", footer.column_meta_start);
        println!(
            "Col. meta size (padded): {}",
            footer.column_meta_offsets_start - footer.column_meta_start
        );
        println!(
            "Glo. buff size (padded): {}",
            footer.global_buff_offsets_start - footer.global_buff_start
        );

        let all_metadata_bytes =
            Self::get_all_meta_bytes(tail_bytes, file_len, scheduler, &footer).await?;
        let meta_offset = footer.column_meta_start;

        // 2. print the global buffers
        let global_bufs_table_nbytes = footer.num_global_buffers as usize * 16;
        let global_bufs_table_start = (footer.global_buff_offsets_start - meta_offset) as usize;
        let global_bufs_table_end = global_bufs_table_start + global_bufs_table_nbytes;
        let global_bufs_table =
            all_metadata_bytes.slice(global_bufs_table_start..global_bufs_table_end);
        let mut global_bufs_cursor = Cursor::new(&global_bufs_table);
        println!("Global buffers:");
        for _ in 0..footer.num_global_buffers {
            let buf_pos = global_bufs_cursor.read_u64::<LittleEndian>()? - meta_offset;
            let buf_size = global_bufs_cursor.read_u64::<LittleEndian>()?;
            println!(" * {}..{}", buf_pos, buf_pos + buf_size);
        }

        let col_meta = Self::read_all_column_metadata(scheduler, &footer).await?;
        println!("Columns:");
        for (idx, col) in col_meta.iter().enumerate() {
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
                let decoder = Self::decoder_meta_to_decoder(encoding);
                println!("     Encoding:");
                println!();
                let encoding_dbg = format!("{:#?}", decoder);
                for line in encoding_dbg.lines() {
                    println!("       {}", line);
                }
                println!();
            }
        }

        Ok(())
    }

    fn decode_meta_array_to_decoder(
        array_encoding: &pbenc::array_encoding::ArrayEncoding,
    ) -> Arc<dyn PhysicalPageScheduler> {
        match array_encoding {
            pbenc::array_encoding::ArrayEncoding::Basic(basic) => {
                match basic.nullability.as_ref().unwrap() {
                    pbenc::basic::Nullability::AllNulls(_) => todo!(),
                    pbenc::basic::Nullability::SomeNulls(_) => todo!(),
                    pbenc::basic::Nullability::NoNulls(values) => {
                        let values_encoding = values.values.as_ref().unwrap();
                        let values_decoder = match values_encoding.buffer_encoding.as_ref().unwrap()
                        {
                            pbenc::buffer_encoding::BufferEncoding::Value(value) => {
                                let buffer_offset = value.buffer.as_ref().unwrap().file_offset;
                                Box::new(ValuePageScheduler::new(
                                    value.bytes_per_value,
                                    buffer_offset,
                                ))
                            }
                            pbenc::buffer_encoding::BufferEncoding::Bitmap(_) => {
                                todo!()
                            }
                        };
                        Arc::new(BasicPageScheduler::new_non_nullable(values_decoder))
                    }
                }
            }
            pbenc::array_encoding::ArrayEncoding::List(list) => {
                let indices_encoding = list
                    .offsets
                    .as_ref()
                    .unwrap()
                    .as_ref()
                    .array_encoding
                    .as_ref()
                    .unwrap();
                Self::decode_meta_array_to_decoder(indices_encoding)
            }
            pbenc::array_encoding::ArrayEncoding::Struct(_) => todo!(),
        }
    }

    fn decoder_meta_to_decoder(encoding: &pbfile::Encoding) -> Arc<dyn PhysicalPageScheduler> {
        match &encoding.style {
            Some(pbfile::encoding::Style::Deferred(_)) => todo!(),
            Some(pbfile::encoding::Style::Direct(encoding)) => {
                let encoding = encoding
                    .encoding
                    .as_ref()
                    .unwrap()
                    .to_msg::<pbenc::ArrayEncoding>()
                    .unwrap();
                Self::decode_meta_array_to_decoder(encoding.array_encoding.as_ref().unwrap())
            }
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
                        let decoder =
                            Self::decoder_meta_to_decoder(page.encoding.as_ref().unwrap());
                        Arc::new(PageInfo {
                            buffer_offsets,
                            decoder,
                            num_rows,
                        })
                    })
                    .collect::<Vec<_>>();
                ColumnInfo { page_infos }
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
        })
    }

    async fn read_range(
        &self,
        range: Range<u64>,
        batch_size: u32,
    ) -> BoxStream<'static, JoinHandle<Result<RecordBatch>>> {
        let mut decode_scheduler = DecodeBatchScheduler::new(&self.file_schema, &self.column_infos);

        let (tx, rx) = mpsc::channel(1024);

        let scheduler = self.scheduler.clone() as Arc<dyn EncodingsIo>;
        // FIXME: spawn this, change this method to sync
        decode_scheduler
            .schedule_range(range, tx, &scheduler)
            .await
            .unwrap();

        BatchDecodeStream::new(rx, self.file_schema.clone(), batch_size, self.num_rows)
            .into_stream()
    }

    // TODO: change output to sendable record batch stream

    pub async fn read_stream(
        &self,
        params: ReadBatchParams,
        batch_size: u32,
    ) -> BoxStream<'static, Result<RecordBatch>> {
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
        futures_stream
            .buffered(1)
            // JoinHandle returns Result<Result<...>> where the outer Result means the thread
            // task panic'd so we propagate that panic here.
            .map(|res_res| res_res.unwrap())
            .boxed()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        cast::AsArray,
        types::{Float64Type, Int32Type},
        ListArray,
    };
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::{DataType, Field};
    use futures::StreamExt;
    use lance_arrow::RecordBatchExt;
    use lance_datagen::{array, gen, RowCount};
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

        let batch = gen()
            .col(Some("score".to_string()), array::rand::<Float64Type>())
            .into_batch_rows(RowCount::from(1000))
            .unwrap();

        let items = gen()
            .col(None, array::rand::<Int32Type>())
            .into_batch_rows(RowCount::from(10000))
            .unwrap();
        let items = items.column(0).clone();
        let offsets = gen()
            .col(None, array::step_custom::<Int32Type>(0, 10))
            .into_batch_rows(RowCount::from(1001))
            .unwrap();
        let offsets = offsets
            .column(0)
            .as_primitive::<Int32Type>()
            .values()
            .clone();
        let offsets = OffsetBuffer::<i32>::new(offsets);
        let list_arr = ListArray::try_new(
            Arc::new(Field::new("item", DataType::Int32, false)),
            offsets,
            items,
            None,
        )
        .unwrap();

        let batch = batch
            .try_with_column(
                Field::new(
                    "my_list",
                    DataType::List(Arc::new(Field::new("item", DataType::Int32, false))),
                    false,
                ),
                Arc::new(list_arr),
            )
            .unwrap();

        let writer = obj_store.create(&tmp_path).await.unwrap();

        let mut file_writer = FileWriter::try_new(
            writer,
            (*batch.schema()).clone(),
            FileWriterOptions::default(),
        )
        .unwrap();

        file_writer.write_batch(&batch).await.unwrap();
        file_writer.finish().await.unwrap();

        let file_scheduler = scheduler.open_file(&tmp_path).await.unwrap();
        FileReader::print_all_metadata(&file_scheduler)
            .await
            .unwrap();
        let file_reader = FileReader::try_open(file_scheduler, (*batch.schema()).clone())
            .await
            .unwrap();

        let mut batch_stream = file_reader
            .read_stream(lance_io::ReadBatchParams::RangeFull, 128)
            .await;

        let mut data_offset = 0;
        while let Some(actual) = batch_stream.next().await {
            let expected_len = 128.min(batch.num_rows() - data_offset);
            let expected = batch.slice(data_offset, expected_len);
            assert_eq!(expected, actual.unwrap());
            data_offset += 128;
        }
    }
}
