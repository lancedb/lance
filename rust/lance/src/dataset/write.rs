// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchReader};
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::StreamExt;
use lance_core::{datatypes::Schema, Error, Result};
use lance_datafusion::chunker::chunk_stream;
use lance_datafusion::utils::reader_to_stream;
use lance_file::format::{MAJOR_VERSION, MINOR_VERSION_NEXT};
use lance_file::v2;
use lance_file::v2::writer::FileWriterOptions;
use lance_file::writer::{FileWriter, ManifestProvider};
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use lance_table::format::{DataFile, Fragment};
use lance_table::io::commit::CommitHandler;
use lance_table::io::manifest::ManifestDescribing;
use object_store::path::Path;
use snafu::{location, Location};
use tracing::instrument;
use uuid::Uuid;

use super::progress::{NoopFragmentWriteProgress, WriteFragmentProgress};
use super::DATA_DIR;

pub mod merge_insert;
pub mod update;

/// The mode to write dataset.
#[derive(Debug, Clone, Copy)]
pub enum WriteMode {
    /// Create a new dataset. Expect the dataset does not exist.
    Create,
    /// Append to an existing dataset.
    Append,
    /// Overwrite a dataset as a new version, or create new dataset if not exist.
    Overwrite,
}

impl TryFrom<&str> for WriteMode {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self> {
        match value.to_lowercase().as_str() {
            "create" => Ok(Self::Create),
            "append" => Ok(Self::Append),
            "overwrite" => Ok(Self::Overwrite),
            _ => Err(Error::IO {
                message: format!("Invalid write mode: {}", value),
                location: location!(),
            }),
        }
    }
}

/// Dataset Write Parameters
#[derive(Debug, Clone)]
pub struct WriteParams {
    /// Max number of records per file.
    pub max_rows_per_file: usize,

    /// Max number of rows per row group.
    pub max_rows_per_group: usize,

    /// Max file size in bytes.
    ///
    /// This is a soft limit. The actual file size may be larger than this value
    /// by a few megabytes, since once we detect we hit this limit, we still
    /// need to flush the footer.
    ///
    /// This limit is checked after writing each group, so if max_rows_per_group
    /// is set to a large value, this limit may be exceeded by a large amount.
    ///
    /// The default is 90 GB. If you are using an object store such as S3, we
    /// currently have a hard 100 GB limit.
    pub max_bytes_per_file: usize,

    /// Write mode
    pub mode: WriteMode,

    pub store_params: Option<ObjectStoreParams>,

    pub progress: Arc<dyn WriteFragmentProgress>,

    /// If present, dataset will use this to update the latest version
    ///
    /// If not set, the default will be based on the object store.  Generally this will
    /// be RenameCommitHandler unless the object store does not handle atomic renames (e.g. S3)
    ///
    /// If a custom object store is provided (via store_params.object_store) then this
    /// must also be provided.
    pub commit_handler: Option<Arc<dyn CommitHandler>>,

    /// If set to true then the Lance v2 writer will be used instead of the Lance v1 writer
    ///
    /// Unless you are intentionally testing the v2 writer, you should leave this as false
    /// as the v2 writer is still experimental and not fully implemented.
    pub use_experimental_writer: bool,
}

impl Default for WriteParams {
    fn default() -> Self {
        Self {
            max_rows_per_file: 1024 * 1024, // 1 million
            max_rows_per_group: 1024,
            // object-store has a 100GB limit, so we should at least make sure
            // we are under that.
            max_bytes_per_file: 90 * 1024 * 1024 * 1024, // 90 GB
            mode: WriteMode::Create,
            store_params: None,
            progress: Arc::new(NoopFragmentWriteProgress::new()),
            commit_handler: None,
            use_experimental_writer: false,
        }
    }
}

/// Writes the given data to the dataset and returns fragments.
///
/// NOTE: the fragments have not yet been assigned an ID. That must be done
/// by the caller. This is so this function can be called in parallel, and the
/// IDs can be assigned after writing is complete.
pub async fn write_fragments(
    dataset_uri: &str,
    data: impl RecordBatchReader + Send + 'static,
    params: WriteParams,
) -> Result<Vec<Fragment>> {
    let (object_store, base) = ObjectStore::from_uri_and_params(
        dataset_uri,
        &params.store_params.clone().unwrap_or_default(),
    )
    .await?;
    let (stream, schema) = reader_to_stream(Box::new(data)).await?;
    write_fragments_internal(Arc::new(object_store), &base, &schema, stream, params).await
}

/// Writes the given data to the dataset and returns fragments.
///
/// NOTE: the fragments have not yet been assigned an ID. That must be done
/// by the caller. This is so this function can be called in parallel, and the
/// IDs can be assigned after writing is complete.
///
/// This is a private variant that takes a `SendableRecordBatchStream` instead
/// of a reader. We don't expose the stream at our interface because it is a
/// DataFusion type.
#[instrument(level = "debug", skip_all)]
pub async fn write_fragments_internal(
    object_store: Arc<ObjectStore>,
    base_dir: &Path,
    schema: &Schema,
    data: SendableRecordBatchStream,
    mut params: WriteParams,
) -> Result<Vec<Fragment>> {
    // Make sure the max rows per group is not larger than the max rows per file
    params.max_rows_per_group = std::cmp::min(params.max_rows_per_group, params.max_rows_per_file);
    // TODO: When writing v2 we could consider skipping this chunking step.  However, leaving in
    // for now as it doesn't hurt anything
    let mut buffered_reader = chunk_stream(data, params.max_rows_per_group);

    let writer_generator = WriterGenerator::new(
        object_store,
        base_dir,
        schema,
        params.use_experimental_writer,
    );
    let mut writer: Option<Box<dyn GenericWriter>> = None;
    let mut num_rows_in_current_file = 0;
    let mut fragments = Vec::new();
    while let Some(batch_chunk) = buffered_reader.next().await {
        let batch_chunk = batch_chunk?;

        if writer.is_none() {
            let (new_writer, new_fragment) = writer_generator.new_writer().await?;
            // rustc has a hard time analyzing the lifetime of the &str returned
            // by multipart_id(), so we convert it to an owned value here.
            let multipart_id = new_writer.multipart_id().to_string();
            params.progress.begin(&new_fragment, &multipart_id).await?;
            writer = Some(new_writer);
            fragments.push(new_fragment);
        }

        writer.as_mut().unwrap().write(&batch_chunk).await?;
        for batch in batch_chunk {
            num_rows_in_current_file += batch.num_rows() as u32;
        }

        if num_rows_in_current_file >= params.max_rows_per_file as u32
            || writer.as_mut().unwrap().tell().await? >= params.max_bytes_per_file as u64
        {
            let (num_rows, data_file) = writer.take().unwrap().finish().await?;
            debug_assert_eq!(num_rows, num_rows_in_current_file);
            params.progress.complete(fragments.last().unwrap()).await?;
            let last_fragment = fragments.last_mut().unwrap();
            last_fragment.physical_rows = Some(num_rows as usize);
            last_fragment.files.push(data_file);
            num_rows_in_current_file = 0;
        }
    }

    // Complete the final writer
    if let Some(mut writer) = writer.take() {
        let (num_rows, data_file) = writer.finish().await?;
        let last_fragment = fragments.last_mut().unwrap();
        last_fragment.physical_rows = Some(num_rows as usize);
        last_fragment.files.push(data_file);
    }

    Ok(fragments)
}

#[async_trait::async_trait]
trait GenericWriter: Send {
    /// Get a unique id associated with the fragment being written
    ///
    /// This is used for progress reporting
    fn multipart_id(&self) -> &str;
    /// Write the given batches to the file
    async fn write(&mut self, batches: &[RecordBatch]) -> Result<()>;
    /// Get the current position in the file
    ///
    /// We use this to know when the file is too large and we need to start
    /// a new file
    async fn tell(&mut self) -> Result<u64>;
    /// Finish writing the file (flush the remaining data and write footer)
    async fn finish(&mut self) -> Result<(u32, DataFile)>;
}

#[async_trait::async_trait]
impl<M: ManifestProvider + Send + Sync> GenericWriter for (FileWriter<M>, String) {
    fn multipart_id(&self) -> &str {
        self.0.multipart_id()
    }
    async fn write(&mut self, batches: &[RecordBatch]) -> Result<()> {
        self.0.write(batches).await
    }
    async fn tell(&mut self) -> Result<u64> {
        Ok(self.0.tell().await? as u64)
    }
    async fn finish(&mut self) -> Result<(u32, DataFile)> {
        Ok((
            self.0.finish().await? as u32,
            DataFile::new_legacy(self.1.clone(), self.0.schema()),
        ))
    }
}

#[async_trait::async_trait]
impl GenericWriter for v2::writer::FileWriter {
    fn multipart_id(&self) -> &str {
        self.multipart_id()
    }
    async fn write(&mut self, batches: &[RecordBatch]) -> Result<()> {
        for batch in batches {
            self.write_batch(batch).await?;
        }
        Ok(())
    }
    async fn tell(&mut self) -> Result<u64> {
        Ok(self.tell().await? as u64)
    }
    async fn finish(&mut self) -> Result<(u32, DataFile)> {
        let field_ids = self
            .field_id_to_column_indices()
            .iter()
            .map(|(field_id, _)| *field_id)
            .collect::<Vec<_>>();
        let column_indices = self
            .field_id_to_column_indices()
            .iter()
            .map(|(_, column_index)| *column_index)
            .collect::<Vec<_>>();
        let data_file = DataFile::new(
            self.path(),
            field_ids,
            column_indices,
            MAJOR_VERSION as u32,
            MINOR_VERSION_NEXT as u32,
        );
        let num_rows = self.finish().await? as u32;
        Ok((num_rows, data_file))
    }
}

/// Creates new file writers for a given dataset.
struct WriterGenerator {
    object_store: Arc<ObjectStore>,
    base_dir: Path,
    schema: Schema,
    use_v2: bool,
}

impl WriterGenerator {
    pub fn new(
        object_store: Arc<ObjectStore>,
        base_dir: &Path,
        schema: &Schema,
        use_v2: bool,
    ) -> Self {
        Self {
            object_store,
            base_dir: base_dir.clone(),
            schema: schema.clone(),
            use_v2,
        }
    }

    pub async fn new_writer(&self) -> Result<(Box<dyn GenericWriter>, Fragment)> {
        let data_file_path = format!("{}.lance", Uuid::new_v4());

        // Use temporary ID 0; will assign ID later.
        let fragment = Fragment::new(0);

        let full_path = self.base_dir.child(DATA_DIR).child(data_file_path.clone());

        let writer = if self.use_v2 {
            let writer = self.object_store.create(&full_path).await?;
            Box::new(v2::writer::FileWriter::try_new(
                writer,
                data_file_path.to_string(),
                self.schema.clone(),
                FileWriterOptions::default(),
            )?) as Box<dyn GenericWriter>
        } else {
            let path = data_file_path.to_string();
            Box::new((
                FileWriter::<ManifestDescribing>::try_new(
                    self.object_store.as_ref(),
                    &full_path,
                    self.schema.clone(),
                    &Default::default(),
                )
                .await?,
                path,
            ))
        };
        Ok((writer, fragment))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Schema as ArrowSchema};
    use datafusion::{error::DataFusionError, physical_plan::stream::RecordBatchStreamAdapter};
    use futures::TryStreamExt;

    #[tokio::test]
    async fn test_chunking_large_batches() {
        // Create a stream of 3 batches of 10 rows
        let schema = Arc::new(ArrowSchema::new(vec![arrow::datatypes::Field::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from_iter(0..28))])
                .unwrap();
        let batches: Vec<RecordBatch> =
            vec![batch.slice(0, 10), batch.slice(10, 10), batch.slice(20, 8)];
        let stream = RecordBatchStreamAdapter::new(
            schema.clone(),
            futures::stream::iter(batches.into_iter().map(Ok::<_, DataFusionError>)),
        );

        // Chunk into a stream of 3 row batches
        let chunks: Vec<Vec<RecordBatch>> = chunk_stream(Box::pin(stream), 3)
            .try_collect()
            .await
            .unwrap();

        assert_eq!(chunks.len(), 10);
        assert_eq!(chunks[0].len(), 1);

        for (i, chunk) in chunks.iter().enumerate() {
            let num_rows = chunk.iter().map(|batch| batch.num_rows()).sum::<usize>();
            if i < chunks.len() - 1 {
                assert_eq!(num_rows, 3);
            } else {
                // Last chunk is shorter
                assert_eq!(num_rows, 1);
            }
        }

        // The fourth chunk is split along the boundary between the original first
        // two batches.
        assert_eq!(chunks[3].len(), 2);
        assert_eq!(chunks[3][0].num_rows(), 1);
        assert_eq!(chunks[3][1].num_rows(), 2);
    }

    #[tokio::test]
    async fn test_chunking_small_batches() {
        // Create a stream of 10 batches of 3 rows
        let schema = Arc::new(ArrowSchema::new(vec![arrow::datatypes::Field::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from_iter(0..30))])
                .unwrap();

        let batches: Vec<RecordBatch> = (0..10).map(|i| batch.slice(i * 3, 3)).collect();
        let stream = RecordBatchStreamAdapter::new(
            schema.clone(),
            futures::stream::iter(batches.into_iter().map(Ok::<_, DataFusionError>)),
        );

        // Chunk into a stream of 10 row batches
        let chunks: Vec<Vec<RecordBatch>> = chunk_stream(Box::pin(stream), 10)
            .try_collect()
            .await
            .unwrap();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 4);
        assert_eq!(chunks[0][0], batch.slice(0, 3));
        assert_eq!(chunks[0][1], batch.slice(3, 3));
        assert_eq!(chunks[0][2], batch.slice(6, 3));
        assert_eq!(chunks[0][3], batch.slice(9, 1));

        for chunk in &chunks {
            let num_rows = chunk.iter().map(|batch| batch.num_rows()).sum::<usize>();
            assert_eq!(num_rows, 10);
        }
    }

    #[tokio::test]
    async fn test_file_size() {
        let schema = Arc::new(ArrowSchema::new(vec![arrow::datatypes::Field::new(
            "a",
            DataType::Int32,
            false,
        )]));

        // Write 1024 rows and show they are split into two files
        // 512 * 4 bytes = 2KB
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter(0..1024))],
        )
        .unwrap();

        let write_params = WriteParams {
            max_rows_per_file: 1024 * 10, // Won't be limited by this
            max_rows_per_group: 512,
            max_bytes_per_file: 2 * 1024,
            mode: WriteMode::Create,
            ..Default::default()
        };

        let data_stream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            futures::stream::iter(std::iter::once(Ok(batch))),
        ));

        let schema = Schema::try_from(schema.as_ref()).unwrap();

        let object_store = Arc::new(ObjectStore::memory());
        let fragments = write_fragments_internal(
            object_store,
            &Path::from("test"),
            &schema,
            data_stream,
            write_params,
        )
        .await
        .unwrap();
        assert_eq!(fragments.len(), 2);
    }

    #[tokio::test]
    async fn test_file_write_v2() {
        let schema = Arc::new(ArrowSchema::new(vec![arrow::datatypes::Field::new(
            "a",
            DataType::Int32,
            false,
        )]));

        // Write 1024 rows
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter(0..1024))],
        )
        .unwrap();

        let write_params = WriteParams {
            use_experimental_writer: true,
            ..Default::default()
        };

        let data_stream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            futures::stream::iter(std::iter::once(Ok(batch))),
        ));

        let schema = Schema::try_from(schema.as_ref()).unwrap();

        let object_store = Arc::new(ObjectStore::memory());
        let fragments = write_fragments_internal(
            object_store,
            &Path::from("test"),
            &schema,
            data_stream,
            write_params,
        )
        .await
        .unwrap();
        assert_eq!(fragments.len(), 1);
        let fragment = &fragments[0];
        assert_eq!(fragment.files.len(), 1);
        assert_eq!(fragment.physical_rows, Some(1024));
        assert_eq!(fragment.files[0].file_minor_version, 3);
    }
}
