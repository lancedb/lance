// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchReader};
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::{StreamExt, TryStreamExt};
use lance_core::{datatypes::Schema, Error, Result};
use lance_datafusion::chunker::{break_stream, chunk_stream};
use lance_datafusion::utils::{peek_reader_schema, reader_to_stream};
use lance_file::format::{MAJOR_VERSION, MINOR_VERSION_NEXT};
use lance_file::v2;
use lance_file::v2::writer::FileWriterOptions;
use lance_file::writer::{FileWriter, ManifestProvider};
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use lance_table::format::{DataFile, Fragment};
use lance_table::io::commit::{CommitHandler, WrappingCommitHandler};
use lance_table::io::manifest::ManifestDescribing;
use object_store::path::Path;
use snafu::{location, Location};
use tracing::instrument;
use uuid::Uuid;

use crate::Dataset;

use super::builder::DatasetBuilder;
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
            _ => Err(Error::io(
                format!("Invalid write mode: {}", value),
                location!(),
            )),
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

    /// If present, will be used to wrap the commit handler implementation.
    /// 
    /// This can be used to augment the behavior of the commit of the commit handler implementation.
    pub commit_handler_wrapper: Option<Arc<dyn WrappingCommitHandler>>,

    /// If set to true then the Lance v1 writer will be used instead of the Lance v2 writer
    ///
    /// Unless you are intentionally testing the v2 writer, you should leave this as true
    /// as the v2 writer is still experimental and not fully implemented.
    pub use_legacy_format: bool,

    /// Experimental: if set to true, the writer will use move-stable row ids.
    /// These row ids are stable after compaction operations, but not after updates.
    /// This makes compaction more efficient, since with stable row ids no
    /// secondary indices need to be updated to point to new row ids.
    pub enable_move_stable_row_ids: bool,
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
            commit_handler_wrapper: None,
            use_legacy_format: true,
            enable_move_stable_row_ids: false,
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
    let (dataset, object_store, base) = if matches!(params.mode, WriteMode::Append) {
        match DatasetBuilder::from_uri(dataset_uri)
            .with_write_params(params.clone())
            .load()
            .await
        {
            Ok(dataset) => {
                let store = dataset.object_store().clone();
                let base = dataset.base.clone();
                (Some(dataset), store, base)
            }
            Err(Error::DatasetNotFound { .. }) => {
                let (object_store, base) = ObjectStore::from_uri_and_params(
                    dataset_uri,
                    &params.store_params.clone().unwrap_or_default(),
                )
                .await?;
                (None, object_store, base)
            }
            Err(err) => return Err(err),
        }
    } else {
        let (object_store, base) = ObjectStore::from_uri_and_params(
            dataset_uri,
            &params.store_params.clone().unwrap_or_default(),
        )
        .await?;
        (None, object_store, base)
    };

    let (data, schema) = peek_reader_schema(Box::new(data)).await?;
    let stream = reader_to_stream(data);
    write_fragments_internal(
        dataset.as_ref(),
        Arc::new(object_store),
        &base,
        &schema,
        stream,
        params,
    )
    .await
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
    dataset: Option<&Dataset>,
    object_store: Arc<ObjectStore>,
    base_dir: &Path,
    schema: &Schema,
    data: SendableRecordBatchStream,
    mut params: WriteParams,
) -> Result<Vec<Fragment>> {
    // Make sure the max rows per group is not larger than the max rows per file
    params.max_rows_per_group = std::cmp::min(params.max_rows_per_group, params.max_rows_per_file);

    let schema = if let Some(dataset) = dataset {
        if matches!(params.mode, WriteMode::Append) {
            // Append mode, so we need to check compatibility
            schema.check_compatible(dataset.schema(), &Default::default())?;
            // Use the schema from the dataset, because it has the correct
            // field ids.
            dataset.schema()
        } else {
            schema
        }
    } else {
        schema
    };

    let mut buffered_reader = if params.use_legacy_format {
        chunk_stream(data, params.max_rows_per_group)
    } else {
        // In v2 we don't care about group size but we do want to break
        // the stream on file boundaries
        break_stream(data, params.max_rows_per_file)
            .map_ok(|batch| vec![batch])
            .boxed()
    };

    let writer_generator =
        WriterGenerator::new(object_store, base_dir, schema, params.use_legacy_format);
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
pub trait GenericWriter: Send {
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

struct V2WriterAdapter {
    writer: v2::writer::FileWriter,
    path: String,
}

#[async_trait::async_trait]
impl GenericWriter for V2WriterAdapter {
    fn multipart_id(&self) -> &str {
        self.writer.multipart_id()
    }
    async fn write(&mut self, batches: &[RecordBatch]) -> Result<()> {
        for batch in batches {
            self.writer.write_batch(batch).await?;
        }
        Ok(())
    }
    async fn tell(&mut self) -> Result<u64> {
        Ok(self.writer.tell().await?)
    }
    async fn finish(&mut self) -> Result<(u32, DataFile)> {
        let field_ids = self
            .writer
            .field_id_to_column_indices()
            .iter()
            .map(|(field_id, _)| *field_id)
            .collect::<Vec<_>>();
        let column_indices = self
            .writer
            .field_id_to_column_indices()
            .iter()
            .map(|(_, column_index)| *column_index)
            .collect::<Vec<_>>();
        let data_file = DataFile::new(
            std::mem::take(&mut self.path),
            field_ids,
            column_indices,
            MAJOR_VERSION as u32,
            MINOR_VERSION_NEXT as u32,
        );
        let num_rows = self.writer.finish().await? as u32;
        Ok((num_rows, data_file))
    }
}

pub async fn open_writer(
    object_store: &ObjectStore,
    schema: &Schema,
    base_dir: &Path,
    use_legacy_format: bool,
) -> Result<Box<dyn GenericWriter>> {
    let filename = format!("{}.lance", Uuid::new_v4());

    let full_path = base_dir.child(DATA_DIR).child(filename.as_str());

    let writer = if use_legacy_format {
        Box::new((
            FileWriter::<ManifestDescribing>::try_new(
                object_store,
                &full_path,
                schema.clone(),
                &Default::default(),
            )
            .await?,
            filename,
        ))
    } else {
        let writer = object_store.create(&full_path).await?;
        let file_writer =
            v2::writer::FileWriter::try_new(writer, schema.clone(), FileWriterOptions::default())?;
        let writer_adapter = V2WriterAdapter {
            writer: file_writer,
            path: filename,
        };
        Box::new(writer_adapter) as Box<dyn GenericWriter>
    };
    Ok(writer)
}

/// Creates new file writers for a given dataset.
struct WriterGenerator {
    object_store: Arc<ObjectStore>,
    base_dir: Path,
    schema: Schema,
    use_legacy_format: bool,
}

impl WriterGenerator {
    pub fn new(
        object_store: Arc<ObjectStore>,
        base_dir: &Path,
        schema: &Schema,
        use_legacy_format: bool,
    ) -> Self {
        Self {
            object_store,
            base_dir: base_dir.clone(),
            schema: schema.clone(),
            use_legacy_format,
        }
    }

    pub async fn new_writer(&self) -> Result<(Box<dyn GenericWriter>, Fragment)> {
        // Use temporary ID 0; will assign ID later.
        let fragment = Fragment::new(0);

        let writer = open_writer(
            &self.object_store,
            &self.schema,
            &self.base_dir,
            self.use_legacy_format,
        )
        .await?;

        Ok((writer, fragment))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{Int32Array, StructArray};
    use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};
    use datafusion::{error::DataFusionError, physical_plan::stream::RecordBatchStreamAdapter};
    use futures::TryStreamExt;
    use lance_file::reader::FileReader;
    use lance_io::traits::Reader;

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
            None,
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
            use_legacy_format: false,
            // This parameter should be ignored
            max_rows_per_group: 1,
            ..Default::default()
        };

        let data_stream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            futures::stream::iter(std::iter::once(Ok(batch))),
        ));

        let schema = Schema::try_from(schema.as_ref()).unwrap();

        let object_store = Arc::new(ObjectStore::memory());
        let fragments = write_fragments_internal(
            None,
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

    #[tokio::test]
    async fn test_file_v1_schema_order() {
        // Create a schema where fields ids are not in order and contain holes.
        // Also first field id is a struct.
        let struct_fields = Fields::from(vec![ArrowField::new("b", DataType::Int32, false)]);
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("d", DataType::Int32, false),
            ArrowField::new("a", DataType::Struct(struct_fields.clone()), false),
        ]);
        let mut schema = Schema::try_from(&arrow_schema).unwrap();
        // Make schema:
        // 0: a
        // 1: a.b
        // (hole at 2)
        // 3: d
        schema.mut_field_by_id(0).unwrap().id = 3;
        schema.mut_field_by_id(1).unwrap().id = 0;
        schema.mut_field_by_id(2).unwrap().id = 1;

        let field_ids = schema.fields_pre_order().map(|f| f.id).collect::<Vec<_>>();
        assert_eq!(field_ids, vec![3, 0, 1]);

        let data = RecordBatch::try_new(
            Arc::new(arrow_schema.clone()),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StructArray::new(
                    struct_fields,
                    vec![Arc::new(Int32Array::from(vec![3, 4]))],
                    None,
                )),
            ],
        )
        .unwrap();

        let write_params = WriteParams {
            use_legacy_format: true,
            ..Default::default()
        };
        let data_stream = Box::pin(RecordBatchStreamAdapter::new(
            Arc::new(arrow_schema),
            futures::stream::iter(std::iter::once(Ok(data.clone()))),
        ));

        let object_store = Arc::new(ObjectStore::memory());
        let base_path = Path::from("test");
        let fragments = write_fragments_internal(
            None,
            object_store.clone(),
            &base_path,
            &schema,
            data_stream,
            write_params,
        )
        .await
        .unwrap();

        assert_eq!(fragments.len(), 1);
        let fragment = &fragments[0];
        assert_eq!(fragment.files.len(), 1);
        assert_eq!(fragment.files[0].fields, vec![0, 1, 3]);

        let path = base_path
            .child(DATA_DIR)
            .child(fragment.files[0].path.as_str());
        let file_reader: Arc<dyn Reader> = object_store.open(&path).await.unwrap().into();
        let reader = FileReader::try_new_from_reader(
            &path,
            file_reader,
            None,
            schema.clone(),
            0,
            0,
            3,
            None,
        )
        .await
        .unwrap();
        assert_eq!(reader.num_batches(), 1);
        let batch = reader.read_batch(0, .., &schema).await.unwrap();
        assert_eq!(batch, data);
    }
}
