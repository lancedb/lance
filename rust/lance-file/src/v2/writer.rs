// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::RecordBatch;

use futures::stream::FuturesUnordered;
use futures::StreamExt;
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::{Error, Result};
use lance_encoding::encoder::{BatchEncoder, EncodeTask, EncodedPage, FieldEncoder};
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

        let encoder = BatchEncoder::try_new(&schema, cache_bytes_per_column)?;
        let num_columns = encoder.num_columns();

        let column_writers = encoder.field_encoders;
        let column_metadata = vec![pbfile::ColumnMetadata::default(); num_columns as usize];

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
        let mut buffer_offsets = Vec::with_capacity(encoded_page.array.buffers.len());
        let mut buffer_sizes = Vec::with_capacity(encoded_page.array.buffers.len());
        for buffer in encoded_page.array.buffers {
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
        let encoded_encoding = Any::from_msg(&encoded_page.array.encoding)?;
        let page = pbfile::column_metadata::Page {
            buffer_offsets,
            buffer_sizes,
            encoding: Some(pbfile::Encoding {
                style: Some(pbfile::encoding::Style::Direct(DirectEncoding {
                    encoding: Some(encoded_encoding),
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

    fn make_file_descriptor(&self) -> Result<pb::FileDescriptor> {
        let lance_schema = lance_core::datatypes::Schema::try_from(&self.schema)?;
        let fields_with_meta = FieldsWithMeta::from(&lance_schema);
        Ok(pb::FileDescriptor {
            schema: Some(pb::Schema {
                fields: fields_with_meta.fields.0,
                metadata: fields_with_meta.metadata,
            }),
            length: self.rows_written,
        })
    }

    async fn write_global_buffers(&mut self) -> Result<Vec<(u64, u64)>> {
        let file_descriptor = self.make_file_descriptor()?;
        let file_descriptor_bytes = file_descriptor.encode_to_vec();
        let file_descriptor_len = file_descriptor_bytes.len() as u64;
        let file_descriptor_position = self.writer.tell().await? as u64;
        self.writer.write_all(&file_descriptor_bytes).await?;
        Ok(vec![(file_descriptor_position, file_descriptor_len)])
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

        // 2. write the column metadatas
        let column_metadata_start = self.writer.tell().await? as u64;
        let metadata_positions = self.write_column_metadatas().await?;

        // 3. write the column metadata position table
        let cmo_table_start = self.writer.tell().await? as u64;
        for (meta_pos, meta_len) in metadata_positions {
            self.writer.write_u64_le(meta_pos).await?;
            self.writer.write_u64_le(meta_len).await?;
        }

        // 3. write global buffers (we write the schema here)
        let global_buffers_start = self.writer.tell().await? as u64;
        let global_buffer_offsets = self.write_global_buffers().await?;
        let num_global_buffers = global_buffer_offsets.len() as u32;

        // write global buffers offset table
        let gbo_table_start = self.writer.tell().await? as u64;
        for (gbo_pos, gbo_len) in global_buffer_offsets {
            self.writer.write_u64_le(gbo_pos).await?;
            self.writer.write_u64_le(gbo_len).await?;
        }

        // 6. write the footer
        self.writer.write_u64_le(column_metadata_start).await?;
        self.writer.write_u64_le(cmo_table_start).await?;
        self.writer.write_u64_le(global_buffers_start).await?;
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::types::Float64Type;
    use arrow_array::RecordBatchReader;
    use lance_datagen::{array, gen, BatchCount, RowCount};
    use lance_io::object_store::ObjectStore;
    use object_store::path::Path;

    use crate::v2::writer::{FileWriter, FileWriterOptions};

    #[tokio::test]
    async fn test_basic_write() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path: String = tmp_dir.path().to_str().unwrap().to_owned();
        let tmp_path = Path::parse(tmp_path).unwrap();
        let tmp_path = tmp_path.child("some_file.lance");
        let obj_store = Arc::new(ObjectStore::local());

        let reader = gen()
            .col(Some("score".to_string()), array::rand::<Float64Type>())
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
        file_writer.finish().await.unwrap();
    }
}
