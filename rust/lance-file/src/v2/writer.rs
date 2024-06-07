// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::RecordBatch;

use bytes::{BufMut, Bytes, BytesMut};
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::{Error, Result};
use lance_encoding::encoder::{
    BatchEncoder, CoreFieldEncodingStrategy, EncodeTask, EncodedBatch, EncodedPage, FieldEncoder,
    FieldEncodingStrategy,
};
use lance_io::object_writer::ObjectWriter;
use lance_io::traits::Writer;
use log::debug;
use prost::Message;
use prost_types::Any;
use snafu::{location, Location};
use tokio::io::AsyncWriteExt;

use crate::datatypes::FieldsWithMeta;
use crate::format::pb;
use crate::format::pbfile;
use crate::format::pbfile::DirectEncoding;
use crate::format::MAGIC;
use crate::format::MAJOR_VERSION;
use crate::format::MINOR_VERSION_NEXT;

#[derive(Debug, Clone, Default)]
pub struct FileWriterOptions {
    /// How many bytes to use for buffering column data
    ///
    /// When data comes in small batches the writer will buffer column data so that
    /// larger pages can be created.  This value will be divided evenly across all of the
    /// columns.  Generally you want this to be at least large enough to match your
    /// filesystem's ideal read size per column.
    ///
    /// In some cases you might want this value to be even larger if you have highly
    /// compressible data.  However, if this is too large, then the writer could require
    /// a lot of memory and write performance may suffer if the CPU-expensive encoding
    /// falls behind and can't be interleaved with the I/O expensive flushing.
    ///
    /// The default will use 8MiB per column which should be reasonable for most cases.
    // TODO: Do we need to be able to set this on a per-column basis?
    pub data_cache_bytes: Option<u64>,
    /// The file writer buffers columns until enough data has arrived to flush a page
    /// to disk.
    ///
    /// Some columns with small data types may not flush very often.  These arrays can
    /// stick around for a long time.  These arrays might also be keeping larger data
    /// structures alive.  By default, the writer will make a deep copy of this array
    /// to avoid any potential memory leaks.  However, this can be disabled for a
    /// (probably minor) performance boost if you are sure that arrays are not keeping
    /// any sibling structures alive (this typically means the array was allocated in
    /// the same language / runtime as the writer)
    ///
    /// Do not enable this if your data is arriving from the C data interface.
    /// Data typically arrives one "batch" at a time (encoded in the C data interface
    /// as a struct array).  Each array in that batch keeps the entire batch alive.
    /// This means a small boolean array (which we will buffer in memory for quite a
    /// while) might keep a much larger record batch around in memory (even though most
    /// of that batch's data has been written to disk)
    pub keep_original_array: Option<bool>,
    pub encoding_strategy: Option<Arc<dyn FieldEncodingStrategy>>,
}

pub struct FileWriter {
    writer: ObjectWriter,
    path: String,
    schema: LanceSchema,
    column_writers: Vec<Box<dyn FieldEncoder>>,
    column_metadata: Vec<pbfile::ColumnMetadata>,
    field_id_to_column_indices: Vec<(i32, i32)>,
    num_columns: u32,
    rows_written: u64,
}

fn initial_column_metadata() -> pbfile::ColumnMetadata {
    pbfile::ColumnMetadata {
        pages: Vec::new(),
        buffer_offsets: Vec::new(),
        buffer_sizes: Vec::new(),
        encoding: Some(pbfile::Encoding {
            location: Some(pbfile::encoding::Location::None(())),
        }),
    }
}

impl FileWriter {
    /// Create a new FileWriter
    pub fn try_new(
        object_writer: ObjectWriter,
        path: String,
        schema: LanceSchema,
        options: FileWriterOptions,
    ) -> Result<Self> {
        let cache_bytes_per_column = if let Some(data_cache_bytes) = options.data_cache_bytes {
            data_cache_bytes / schema.fields.len() as u64
        } else {
            8 * 1024 * 1024
        };

        schema.validate()?;

        let keep_original_array = options.keep_original_array.unwrap_or(false);
        let encoding_strategy = options
            .encoding_strategy
            .unwrap_or_else(|| Arc::new(CoreFieldEncodingStrategy::default()));

        let encoder = BatchEncoder::try_new(
            &schema,
            encoding_strategy.as_ref(),
            cache_bytes_per_column,
            keep_original_array,
        )?;
        let num_columns = encoder.num_columns();

        let column_writers = encoder.field_encoders;
        let column_metadata = vec![initial_column_metadata(); num_columns as usize];

        Ok(Self {
            writer: object_writer,
            path,
            schema,
            column_writers,
            column_metadata,
            num_columns,
            rows_written: 0,
            field_id_to_column_indices: encoder.field_id_to_column_index,
        })
    }

    async fn write_page(&mut self, encoded_page: EncodedPage) -> Result<()> {
        let mut buffers = encoded_page.array.buffers;
        buffers.sort_by_key(|b| b.index);
        let mut buffer_offsets = Vec::with_capacity(buffers.len());
        let mut buffer_sizes = Vec::with_capacity(buffers.len());
        for buffer in buffers {
            buffer_offsets.push(self.writer.tell().await? as u64);
            buffer_sizes.push(
                buffer
                    .parts
                    .iter()
                    .map(|part| part.len() as u64)
                    .sum::<u64>(),
            );
            // Note: could potentially use write_vectored here but there is no
            // write_vectored_all and object_store doesn't support it anyways and
            // buffers won't normally be in *too* many parts so its unlikely to
            // have much benefit in most cases.
            for part in &buffer.parts {
                self.writer.write_all(part).await?;
            }
        }
        let encoded_encoding = Any::from_msg(&encoded_page.array.encoding)?.encode_to_vec();
        let page = pbfile::column_metadata::Page {
            buffer_offsets,
            buffer_sizes,
            encoding: Some(pbfile::Encoding {
                location: Some(pbfile::encoding::Location::Direct(DirectEncoding {
                    encoding: encoded_encoding,
                })),
            }),
            length: encoded_page.num_rows,
        };
        self.column_metadata[encoded_page.column_idx as usize]
            .pages
            .push(page);
        Ok(())
    }

    async fn write_pages(
        &mut self,
        mut encoding_tasks: FuturesUnordered<EncodeTask>,
    ) -> Result<()> {
        // As soon as an encoding task is done we write it.  There is no parallelism
        // needed here because "writing" is really just submitting the buffer to the
        // underlying write scheduler (either the OS or object_store's scheduler for
        // cloud writes).  The only time we might truly await on write_page is if the
        // scheduler's write queue is full.
        //
        // Also, there is no point in trying to make write_page parallel anyways
        // because we wouldn't want buffers getting mixed up across pages.
        while let Some(encoding_task) = encoding_tasks.next().await {
            let encoded_page = encoding_task?;
            self.write_page(encoded_page).await?;
        }
        // It's important to flush here, we don't know when the next batch will arrive
        // and the underlying cloud store could have writes in progress that won't advance
        // until we interact with the writer again.  These in-progress writes will time out
        // if we don't flush.
        self.writer.flush().await?;
        Ok(())
    }

    /// Schedule batches of data to be written to the file
    pub async fn write_batches(
        &mut self,
        batches: impl Iterator<Item = &RecordBatch>,
    ) -> Result<()> {
        for batch in batches {
            self.write_batch(batch).await?;
        }
        Ok(())
    }

    /// Schedule a batch of data to be written to the file
    ///
    /// Note: the future returned by this method may complete before the data has been fully
    /// flushed to the file (some data may be in the data cache or the I/O cache)
    pub async fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        debug!(
            "write_batch called with {} bytes of data",
            batch.get_array_memory_size()
        );
        let num_rows = batch.num_rows() as u64;
        if num_rows == 0 {
            return Ok(());
        }
        if num_rows > u32::MAX as u64 {
            return Err(Error::InvalidInput {
                source: "cannot write Lance files with more than 2^32 rows".into(),
                location: location!(),
            });
        }
        self.rows_written = match self.rows_written.checked_add(batch.num_rows() as u64) {
            Some(rows_written) => rows_written,
            None => {
                return Err(Error::InvalidInput { source: format!("cannot write batch with {} rows because {} rows have already been written and Lance files cannot contain more than 2^32 rows", num_rows, self.rows_written).into(), location: location!() });
            }
        };
        // First we push each array into its column writer.  This may or may not generate enough
        // data to trigger an encoding task.  We collect any encoding tasks into a queue.
        let encoding_tasks = self
            .schema
            .fields
            .iter()
            .zip(self.column_writers.iter_mut())
            .map(|(field, column_writer)| {
                let array = batch
                    .column_by_name(&field.name)
                    .ok_or(Error::InvalidInput {
                        source: format!(
                            "Cannot write batch.  The batch was missing the column `{}`",
                            field.name
                        )
                        .into(),
                        location: location!(),
                    })?;
                column_writer.maybe_encode(array.clone())
            })
            .collect::<Result<Vec<_>>>()?;
        let encoding_tasks = encoding_tasks
            .into_iter()
            .flatten()
            .collect::<FuturesUnordered<_>>();

        self.write_pages(encoding_tasks).await?;

        Ok(())
    }

    async fn write_column_metadata(
        &mut self,
        metadata: pbfile::ColumnMetadata,
    ) -> Result<(u64, u64)> {
        let metadata_bytes = metadata.encode_to_vec();
        let position = self.writer.tell().await? as u64;
        let len = metadata_bytes.len() as u64;
        self.writer.write_all(&metadata_bytes).await?;
        Ok((position, len))
    }

    async fn write_column_metadatas(&mut self) -> Result<Vec<(u64, u64)>> {
        let mut metadatas = Vec::new();
        std::mem::swap(&mut self.column_metadata, &mut metadatas);
        let mut metadata_positions = Vec::with_capacity(metadatas.len());
        for metadata in metadatas {
            metadata_positions.push(self.write_column_metadata(metadata).await?);
        }
        Ok(metadata_positions)
    }

    fn make_file_descriptor(
        schema: &lance_core::datatypes::Schema,
        num_rows: u64,
    ) -> Result<pb::FileDescriptor> {
        let fields_with_meta = FieldsWithMeta::from(schema);
        Ok(pb::FileDescriptor {
            schema: Some(pb::Schema {
                fields: fields_with_meta.fields.0,
                metadata: fields_with_meta.metadata,
            }),
            length: num_rows,
        })
    }

    async fn write_global_buffers(&mut self) -> Result<Vec<(u64, u64)>> {
        let file_descriptor = Self::make_file_descriptor(&self.schema, self.rows_written)?;
        let file_descriptor_bytes = file_descriptor.encode_to_vec();
        let file_descriptor_len = file_descriptor_bytes.len() as u64;
        let file_descriptor_position = self.writer.tell().await? as u64;
        self.writer.write_all(&file_descriptor_bytes).await?;
        Ok(vec![(file_descriptor_position, file_descriptor_len)])
    }

    /// Add a metadata entry to the schema
    ///
    /// This method is useful because sometimes the metadata is not known until after the
    /// data has been written.  This method allows you to alter the schema metadata.  It
    /// must be called before `finish` is called.
    pub fn add_schema_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.schema.metadata.insert(key.into(), value.into());
    }

    /// Finishes writing the file
    ///
    /// This method will wait until all data has been flushed to the file.  Then it
    /// will write the file metadata and the footer.  It will not return until all
    /// data has been flushed and the file has been closed.
    ///
    /// Returns the total number of rows written
    pub async fn finish(&mut self) -> Result<u64> {
        // 1. flush any remaining data and write out those pages
        let encoding_tasks = self
            .column_writers
            .iter_mut()
            .map(|writer| writer.flush())
            .collect::<Result<Vec<_>>>()?;
        let encoding_tasks = encoding_tasks
            .into_iter()
            .flatten()
            .collect::<FuturesUnordered<_>>();
        self.write_pages(encoding_tasks).await?;

        // No data, so don't create a file
        if self.rows_written == 0 {
            self.writer.shutdown().await?;
            return Ok(0);
        }

        // 2. write the column metadata buffers (coming soon)
        // 3. write global buffers (we write the schema here)
        let global_buffer_offsets = self.write_global_buffers().await?;
        let num_global_buffers = global_buffer_offsets.len() as u32;

        // 4. write the column metadatas
        let column_metadata_start = self.writer.tell().await? as u64;
        let metadata_positions = self.write_column_metadatas().await?;

        // 5. write the column metadata offset table
        let cmo_table_start = self.writer.tell().await? as u64;
        for (meta_pos, meta_len) in metadata_positions {
            self.writer.write_u64_le(meta_pos).await?;
            self.writer.write_u64_le(meta_len).await?;
        }

        // 6. write global buffers offset table
        let gbo_table_start = self.writer.tell().await? as u64;
        for (gbo_pos, gbo_len) in global_buffer_offsets {
            self.writer.write_u64_le(gbo_pos).await?;
            self.writer.write_u64_le(gbo_len).await?;
        }

        // 7. write the footer
        self.writer.write_u64_le(column_metadata_start).await?;
        self.writer.write_u64_le(cmo_table_start).await?;
        self.writer.write_u64_le(gbo_table_start).await?;
        self.writer.write_u32_le(num_global_buffers).await?;
        self.writer.write_u32_le(self.num_columns).await?;
        self.writer.write_u16_le(MAJOR_VERSION as u16).await?;
        self.writer.write_u16_le(MINOR_VERSION_NEXT).await?;
        self.writer.write_all(MAGIC).await?;

        // 7. close the writer
        self.writer.shutdown().await?;
        Ok(self.rows_written)
    }

    pub fn multipart_id(&self) -> &str {
        &self.writer.multipart_id
    }

    pub async fn tell(&mut self) -> Result<u64> {
        Ok(self.writer.tell().await? as u64)
    }

    pub fn field_id_to_column_indices(&self) -> &[(i32, i32)] {
        &self.field_id_to_column_indices
    }

    pub fn path(&self) -> &str {
        &self.path
    }
}

/// Utility trait for converting EncodedBatch to Bytes using the
/// lance file format
pub trait EncodedBatchWriteExt {
    /// Serializes into a lance file, including the schema
    fn try_to_self_described_lance(&self) -> Result<Bytes>;
    /// Serializes into a lance file, without the schema.
    ///
    /// The schema must be provided to deserialize the buffer
    fn try_to_mini_lance(&self) -> Result<Bytes>;
}

// Creates a lance footer and appends it to the encoded data
//
// The logic here is very similar to logic in the FileWriter except we
// are using BufMut (put_xyz) instead of AsyncWrite (write_xyz).
fn concat_lance_footer(batch: &EncodedBatch, write_schema: bool) -> Result<Bytes> {
    // Estimating 1MiB for file footer
    let mut data = BytesMut::with_capacity(batch.data.len() + 1024 * 1024);
    data.put(batch.data.clone());
    // Write column metadata buffers
    let column_metadata_buffers_start = data.len() as u64;
    // write global buffers (we write the schema here)
    let global_buffers_start = data.len() as u64;
    let global_buffers = if write_schema {
        let lance_schema = lance_core::datatypes::Schema::try_from(batch.schema.as_ref())?;
        let descriptor = FileWriter::make_file_descriptor(&lance_schema, batch.num_rows)?;
        let descriptor_bytes = descriptor.encode_to_vec();
        let descriptor_len = descriptor_bytes.len() as u64;
        data.put(descriptor_bytes.as_slice());

        vec![(global_buffers_start, descriptor_len)]
    } else {
        vec![]
    };
    let col_metadata_start = data.len() as u64;

    let mut col_metadata_positions = Vec::new();
    // Write column metadata
    for col in &batch.page_table {
        let position = data.len() as u64;
        let pages = col
            .page_infos
            .iter()
            .map(|page_info| {
                let encoded_encoding = Any::from_msg(&page_info.encoding)?.encode_to_vec();
                let (buffer_offsets, buffer_sizes): (Vec<_>, Vec<_>) = page_info
                    .buffer_offsets_and_sizes
                    .as_ref()
                    .iter()
                    .cloned()
                    .unzip();
                Ok(pbfile::column_metadata::Page {
                    buffer_offsets,
                    buffer_sizes,
                    encoding: Some(pbfile::Encoding {
                        location: Some(pbfile::encoding::Location::Direct(DirectEncoding {
                            encoding: encoded_encoding,
                        })),
                    }),
                    length: page_info.num_rows,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let (buffer_offsets, buffer_sizes): (Vec<_>, Vec<_>) =
            col.buffer_offsets_and_sizes.iter().cloned().unzip();
        let column = pbfile::ColumnMetadata {
            pages,
            buffer_offsets,
            buffer_sizes,
            encoding: Some(pbfile::Encoding {
                location: Some(pbfile::encoding::Location::None(())),
            }),
        };
        let column_bytes = column.encode_to_vec();
        col_metadata_positions.push((position, data.len() as u64));
        data.put(column_bytes.as_slice());
    }
    // Write column metadata offsets table
    let cmo_table_start = data.len() as u64;
    for (meta_pos, meta_len) in col_metadata_positions {
        data.put_u64_le(meta_pos);
        data.put_u64_le(meta_len);
    }
    // Write global buffers offsets table
    let gbo_table_start = data.len() as u64;
    let num_global_buffers = global_buffers.len() as u32;
    for (gbo_pos, gbo_len) in global_buffers {
        data.put_u64_le(gbo_pos);
        data.put_u64_le(gbo_len);
    }

    // write the footer
    data.put_u64_le(column_metadata_buffers_start);
    data.put_u64_le(global_buffers_start);
    data.put_u64_le(col_metadata_start);
    data.put_u64_le(cmo_table_start);
    data.put_u64_le(gbo_table_start);
    data.put_u32_le(num_global_buffers);
    data.put_u32_le(batch.page_table.len() as u32);
    data.put_u16_le(MAJOR_VERSION as u16);
    data.put_u16_le(MINOR_VERSION_NEXT);
    data.put(MAGIC.as_slice());

    Ok(data.freeze())
}

impl EncodedBatchWriteExt for EncodedBatch {
    fn try_to_self_described_lance(&self) -> Result<Bytes> {
        concat_lance_footer(self, true)
    }

    fn try_to_mini_lance(&self) -> Result<Bytes> {
        concat_lance_footer(self, false)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        types::Float64Type, RecordBatch, RecordBatchReader, StringArray, UInt32Array,
    };
    use arrow_schema::{DataType, Field, Schema};
    use lance_datagen::{array, gen, BatchCount, RowCount};
    use lance_io::object_store::ObjectStore;
    use object_store::path::Path;

    use crate::v2::reader::FileReader;
    use crate::v2::writer::{FileWriter, FileWriterOptions};

    use lance_io::scheduler::ScanScheduler;

    use futures::StreamExt;

    #[tokio::test]
    async fn test_basic_write() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path: String = tmp_dir.path().to_str().unwrap().to_owned();
        let tmp_path = Path::parse(tmp_path).unwrap();
        let tmp_path = tmp_path.child("some_file.lance");
        let obj_store = Arc::new(ObjectStore::local());

        let reader = gen()
            .col("score", array::rand::<Float64Type>())
            .into_reader_rows(RowCount::from(1000), BatchCount::from(10));

        let writer = obj_store.create(&tmp_path).await.unwrap();

        let lance_schema =
            lance_core::datatypes::Schema::try_from(reader.schema().as_ref()).unwrap();

        let mut file_writer = FileWriter::try_new(
            writer,
            tmp_path.to_string(),
            lance_schema,
            FileWriterOptions::default(),
        )
        .unwrap();

        for batch in reader {
            file_writer.write_batch(&batch.unwrap()).await.unwrap();
        }
        file_writer.add_schema_metadata("foo", "bar");
        file_writer.finish().await.unwrap();
        // Tests asserting the contents of the written file are in reader.rs
    }

    #[tokio::test]
    async fn test_string_array_encoding() {
        // create table with string column
        let tmp_path = Path::parse("/home/ubuntu/test/strings.lance").unwrap();
        let obj_store = Arc::new(ObjectStore::local());
        let writer = obj_store.create(&tmp_path).await.unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::UInt32, false),
            Field::new("strings", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![1, 2, 3, 4, 5, 6])),
                Arc::new(StringArray::from(vec![
                    "abcd", "hello", "abcd", "apple", "hello", "abcd",
                ])),
            ],
        );

        let batches = vec![batch];
        let lance_schema = lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap();

        let mut file_writer = FileWriter::try_new(
            writer,
            tmp_path.to_string(),
            lance_schema,
            FileWriterOptions::default(),
        )
        .unwrap();

        for batch in batches {
            println!("Writing batch...");
            println!("{:?}", batch);
            file_writer.write_batch(&batch.unwrap()).await.unwrap();
        }

        file_writer.finish().await.unwrap();

        // let fs = FsFixture::default();
        let object_store = Arc::new(ObjectStore::local());
        let fs_scheduler = ScanScheduler::new(object_store.clone(), 8);
        let file_scheduler = fs_scheduler.open_file(&tmp_path).await.unwrap();

        let file_reader = FileReader::try_open(file_scheduler, None).await.unwrap();
        let schema = file_reader.schema();
        println!("Schema - {:?}", schema);

        let mut batch_stream = file_reader
            .read_stream(lance_io::ReadBatchParams::RangeFull, 1024, 16)
            .unwrap();

        // print length of batch stream
        let mut num_batches = 0;
        // print batches in stream
        while let Some(batch) = batch_stream.next().await {
            num_batches += 1;
            println!("{:?}", batch);
        }

        println!("Number of batches: {}", num_batches);
    }
}
