// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchReader};
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::{StreamExt, TryStreamExt};
use lance_core::datatypes::{NullabilityComparison, SchemaCompareOptions, StorageClass};
use lance_core::{datatypes::Schema, Error, Result};
use lance_datafusion::chunker::{break_stream, chunk_stream};
use lance_file::v2;
use lance_file::v2::writer::FileWriterOptions;
use lance_file::version::LanceFileVersion;
use lance_file::writer::{FileWriter, ManifestProvider};
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use lance_table::format::{DataFile, Fragment};
use lance_table::io::commit::{commit_handler_from_url, CommitHandler};
use lance_table::io::manifest::ManifestDescribing;
use object_store::path::Path;
use snafu::{location, Location};
use tracing::instrument;
use uuid::Uuid;

use crate::session::Session;
use crate::Dataset;

use super::blob::BlobStreamExt;
use super::progress::{NoopFragmentWriteProgress, WriteFragmentProgress};
use super::transaction::Transaction;
use super::DATA_DIR;

mod commit;
mod insert;
pub mod merge_insert;
pub mod update;

pub use commit::CommitBuilder;
pub use insert::InsertBuilder;

/// The destination to write data to.
#[derive(Debug, Clone)]
pub enum WriteDestination<'a> {
    /// An existing dataset to write to.
    Dataset(Arc<Dataset>),
    /// A URI to write to.
    Uri(&'a str),
}

impl WriteDestination<'_> {
    pub fn dataset(&self) -> Option<&Dataset> {
        match self {
            WriteDestination::Dataset(dataset) => Some(dataset.as_ref()),
            WriteDestination::Uri(_) => None,
        }
    }
}

impl From<Arc<Dataset>> for WriteDestination<'_> {
    fn from(dataset: Arc<Dataset>) -> Self {
        WriteDestination::Dataset(dataset)
    }
}

impl<'a> From<&'a str> for WriteDestination<'a> {
    fn from(uri: &'a str) -> Self {
        WriteDestination::Uri(uri)
    }
}

impl<'a> From<&'a String> for WriteDestination<'a> {
    fn from(uri: &'a String) -> Self {
        WriteDestination::Uri(uri.as_str())
    }
}

impl<'a> From<&'a Path> for WriteDestination<'a> {
    fn from(path: &'a Path) -> Self {
        WriteDestination::Uri(path.as_ref())
    }
}

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
            _ => Err(Error::invalid_input(
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

    /// The format version to use when writing data.
    ///
    /// Newer versions are more efficient but the data can only be read by more recent versions
    /// of lance.
    ///
    /// If not specified then the latest stable version will be used.
    pub data_storage_version: Option<LanceFileVersion>,

    /// Experimental: if set to true, the writer will use move-stable row ids.
    /// These row ids are stable after compaction operations, but not after updates.
    /// This makes compaction more efficient, since with stable row ids no
    /// secondary indices need to be updated to point to new row ids.
    pub enable_move_stable_row_ids: bool,

    /// If set to true, and this is a new dataset, uses the new v2 manifest paths.
    /// These allow constant-time lookups for the latest manifest on object storage.
    /// This parameter has no effect on existing datasets. To migrate an existing
    /// dataset, use the [`super::Dataset::migrate_manifest_paths_v2`] method.
    /// Default is False.
    pub enable_v2_manifest_paths: bool,

    pub object_store_registry: Arc<ObjectStoreRegistry>,

    pub session: Option<Arc<Session>>,
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
            data_storage_version: None,
            enable_move_stable_row_ids: false,
            enable_v2_manifest_paths: false,
            object_store_registry: Arc::new(ObjectStoreRegistry::default()),
            session: None,
        }
    }
}

impl WriteParams {
    /// Create a new WriteParams with the given storage version.
    /// The other fields are set to their default values.
    pub fn with_storage_version(version: LanceFileVersion) -> Self {
        Self {
            data_storage_version: Some(version),
            ..Default::default()
        }
    }

    pub fn storage_version_or_default(&self) -> LanceFileVersion {
        self.data_storage_version.unwrap_or_default()
    }
}

/// Writes the given data to the dataset and returns fragments.
///
/// NOTE: the fragments have not yet been assigned an ID. That must be done
/// by the caller. This is so this function can be called in parallel, and the
/// IDs can be assigned after writing is complete.
#[deprecated(
    since = "0.20.0",
    note = "Use [`InsertBuilder::write_uncommitted_stream`] instead"
)]
pub async fn write_fragments(
    dest: impl Into<WriteDestination<'_>>,
    data: impl RecordBatchReader + Send + 'static,
    params: WriteParams,
) -> Result<Transaction> {
    InsertBuilder::new(dest.into())
        .with_params(&params)
        .execute_uncommitted_stream(Box::new(data))
        .await
}

pub async fn do_write_fragments(
    object_store: Arc<ObjectStore>,
    base_dir: &Path,
    schema: &Schema,
    data: SendableRecordBatchStream,
    params: WriteParams,
    storage_version: LanceFileVersion,
) -> Result<Vec<Fragment>> {
    let mut buffered_reader = if storage_version == LanceFileVersion::Legacy {
        // In v1 we split the stream into row group sized batches
        chunk_stream(data, params.max_rows_per_group)
    } else {
        // In v2 we don't care about group size but we do want to break
        // the stream on file boundaries
        break_stream(data, params.max_rows_per_file)
            .map_ok(|batch| vec![batch])
            .boxed()
    };

    let writer_generator = WriterGenerator::new(object_store, base_dir, schema, storage_version);
    let mut writer: Option<Box<dyn GenericWriter>> = None;
    let mut num_rows_in_current_file = 0;
    let mut fragments = Vec::new();
    while let Some(batch_chunk) = buffered_reader.next().await {
        let batch_chunk = batch_chunk?;

        if writer.is_none() {
            let (new_writer, new_fragment) = writer_generator.new_writer().await?;
            params.progress.begin(&new_fragment).await?;
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

pub struct WrittenFragments {
    /// The fragments written to the dataset (and the schema)
    pub default: (Vec<Fragment>, Schema),
    /// The fragments written to the blob dataset, if any
    pub blob: Option<(Vec<Fragment>, Schema)>,
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
    schema: Schema,
    data: SendableRecordBatchStream,
    mut params: WriteParams,
) -> Result<WrittenFragments> {
    // Make sure the max rows per group is not larger than the max rows per file
    params.max_rows_per_group = std::cmp::min(params.max_rows_per_group, params.max_rows_per_file);

    let (schema, storage_version) = if let Some(dataset) = dataset {
        match params.mode {
            WriteMode::Append | WriteMode::Create => {
                // Append mode, so we need to check compatibility
                schema.check_compatible(
                    dataset.schema(),
                    &SchemaCompareOptions {
                        // We don't care if the user claims their data is nullable / non-nullable.  We will
                        // verify against the actual data.
                        compare_nullability: NullabilityComparison::Ignore,
                        allow_missing_if_nullable: true,
                        ignore_field_order: true,
                        compare_dictionary: true,
                        ..Default::default()
                    },
                )?;
                // Project from the dataset schema, because it has the correct field ids.
                let write_schema = dataset.schema().project_by_schema(&schema)?;
                // Use the storage version from the dataset, ignoring any version from the user.
                let data_storage_version = dataset
                    .manifest()
                    .data_storage_format
                    .lance_file_version()?;
                (write_schema, data_storage_version)
            }
            WriteMode::Overwrite => {
                // Overwrite, use the schema from the data.  If the user specified
                // a storage version use that.  Otherwise use the version from the
                // dataset.
                let data_storage_version = params.data_storage_version.unwrap_or(
                    dataset
                        .manifest()
                        .data_storage_format
                        .lance_file_version()?,
                );
                (schema, data_storage_version)
            }
        }
    } else {
        // Brand new dataset, use the schema from the data and the storage version
        // from the user or the default.
        (schema, params.storage_version_or_default())
    };

    let data_schema = schema.project_by_schema(data.schema().as_ref())?;

    let (data, blob_data) = data.extract_blob_stream(&data_schema);

    // Some params we borrow from the normal write, some we override
    let blob_write_params = WriteParams {
        store_params: params.store_params.clone(),
        commit_handler: params.commit_handler.clone(),
        data_storage_version: params.data_storage_version,
        enable_move_stable_row_ids: true,
        // This shouldn't really matter since all commits are detached
        enable_v2_manifest_paths: true,
        object_store_registry: params.object_store_registry.clone(),
        max_bytes_per_file: params.max_bytes_per_file,
        max_rows_per_file: params.max_rows_per_file,
        ..Default::default()
    };

    if blob_data.is_some() && !params.enable_move_stable_row_ids {
        return Err(Error::invalid_input(
            "The blob storage class requires move stable row ids",
            location!(),
        ));
    }

    let frag_schema = schema.retain_storage_class(StorageClass::Default);
    let fragments_fut = do_write_fragments(
        object_store.clone(),
        base_dir,
        &frag_schema,
        data,
        params,
        storage_version,
    );

    let (default, blob) = if let Some(blob_data) = blob_data {
        let blob_schema = schema.retain_storage_class(StorageClass::Blob);
        let blobs_path = base_dir.child("_blobs");
        let blob_fut = do_write_fragments(
            object_store,
            &blobs_path,
            &blob_schema,
            blob_data,
            blob_write_params,
            storage_version,
        );
        let (fragments_res, blobs_res) = futures::join!(fragments_fut, blob_fut);
        let fragments = fragments_res?;
        let blobs = blobs_res?;
        ((fragments, frag_schema), Some((blobs, blob_schema)))
    } else {
        let fragments = fragments_fut.await?;
        ((fragments, frag_schema), None)
    };

    Ok(WrittenFragments { default, blob })
}

#[async_trait::async_trait]
pub trait GenericWriter: Send {
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
            .map(|(field_id, _)| *field_id as i32)
            .collect::<Vec<_>>();
        let column_indices = self
            .writer
            .field_id_to_column_indices()
            .iter()
            .map(|(_, column_index)| *column_index as i32)
            .collect::<Vec<_>>();
        let (major, minor) = self.writer.version().to_numbers();
        let data_file = DataFile::new(
            std::mem::take(&mut self.path),
            field_ids,
            column_indices,
            major,
            minor,
        );
        let num_rows = self.writer.finish().await? as u32;
        Ok((num_rows, data_file))
    }
}

pub async fn open_writer(
    object_store: &ObjectStore,
    schema: &Schema,
    base_dir: &Path,
    storage_version: LanceFileVersion,
) -> Result<Box<dyn GenericWriter>> {
    let filename = format!("{}.lance", Uuid::new_v4());

    let full_path = base_dir.child(DATA_DIR).child(filename.as_str());

    let writer = if storage_version == LanceFileVersion::Legacy {
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
        let file_writer = v2::writer::FileWriter::try_new(
            writer,
            schema.clone(),
            FileWriterOptions {
                format_version: Some(storage_version),
                ..Default::default()
            },
        )?;
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
    storage_version: LanceFileVersion,
}

impl WriterGenerator {
    pub fn new(
        object_store: Arc<ObjectStore>,
        base_dir: &Path,
        schema: &Schema,
        storage_version: LanceFileVersion,
    ) -> Self {
        Self {
            object_store,
            base_dir: base_dir.clone(),
            schema: schema.clone(),
            storage_version,
        }
    }

    pub async fn new_writer(&self) -> Result<(Box<dyn GenericWriter>, Fragment)> {
        // Use temporary ID 0; will assign ID later.
        let fragment = Fragment::new(0);

        let writer = open_writer(
            &self.object_store,
            &self.schema,
            &self.base_dir,
            self.storage_version,
        )
        .await?;

        Ok((writer, fragment))
    }
}

// Given input options resolve what the commit handler should be.
async fn resolve_commit_handler(
    uri: &str,
    commit_handler: Option<Arc<dyn CommitHandler>>,
    store_options: &Option<ObjectStoreParams>,
) -> Result<Arc<dyn CommitHandler>> {
    match commit_handler {
        None => {
            if store_options
                .as_ref()
                .map(|opts| opts.object_store.is_some())
                .unwrap_or_default()
            {
                return Err(Error::InvalidInput { source: "when creating a dataset with a custom object store the commit_handler must also be specified".into(), location: location!() });
            }
            commit_handler_from_url(uri, store_options).await
        }
        Some(commit_handler) => {
            if uri.starts_with("s3+ddb") {
                Err(Error::InvalidInput {
                    source: "`s3+ddb://` scheme and custom commit handler are mutually exclusive"
                        .into(),
                    location: location!(),
                })
            } else {
                Ok(commit_handler)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{Int32Array, StructArray};
    use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};
    use datafusion::{error::DataFusionError, physical_plan::stream::RecordBatchStreamAdapter};
    use futures::TryStreamExt;
    use lance_datagen::{array, gen, BatchCount, RowCount};
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
        let reader_to_frags = |data_reader: Box<dyn RecordBatchReader + Send>| {
            let schema = data_reader.schema();
            let data_reader =
                data_reader.map(|rb| rb.map_err(datafusion::error::DataFusionError::from));

            let data_stream = Box::pin(RecordBatchStreamAdapter::new(
                schema.clone(),
                futures::stream::iter(data_reader),
            ));

            let write_params = WriteParams {
                max_rows_per_file: 1024 * 1024, // Won't be limited by this
                max_bytes_per_file: 2 * 1024,
                mode: WriteMode::Create,
                ..Default::default()
            };

            async move {
                let schema = Schema::try_from(schema.as_ref()).unwrap();

                let object_store = Arc::new(ObjectStore::memory());
                write_fragments_internal(
                    None,
                    object_store,
                    &Path::from("test"),
                    schema,
                    data_stream,
                    write_params,
                )
                .await
            }
        };

        // The writer will not generate a new file until at enough data is *written* (not
        // just accumulated) to justify a new file.  Since the default page size is 8MiB
        // we actually need to generate quite a bit of data to trigger this.
        //
        // To avoid generating and writing millions of rows (which is a bit slow for a unit
        // test) we can use a large data type (1KiB binary)
        let data_reader = Box::new(
            gen()
                .anon_col(array::rand_fsb(1024))
                .into_reader_rows(RowCount::from(10 * 1024), BatchCount::from(2)),
        );

        let written = reader_to_frags(data_reader).await.unwrap();

        assert!(written.blob.is_none());
        let fragments = written.default.0;

        assert_eq!(fragments.len(), 2);
    }

    #[tokio::test]
    async fn test_file_write_version() {
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

        let versions = vec![
            LanceFileVersion::Legacy,
            LanceFileVersion::V2_0,
            LanceFileVersion::V2_1,
            LanceFileVersion::Stable,
            LanceFileVersion::Next,
        ];
        for version in versions {
            let (major, minor) = version.to_numbers();
            let write_params = WriteParams {
                data_storage_version: Some(version),
                // This parameter should be ignored
                max_rows_per_group: 1,
                ..Default::default()
            };

            let data_stream = Box::pin(RecordBatchStreamAdapter::new(
                schema.clone(),
                futures::stream::iter(std::iter::once(Ok(batch.clone()))),
            ));

            let schema = Schema::try_from(schema.as_ref()).unwrap();

            let object_store = Arc::new(ObjectStore::memory());
            let written = write_fragments_internal(
                None,
                object_store,
                &Path::from("test"),
                schema,
                data_stream,
                write_params,
            )
            .await
            .unwrap();

            assert!(written.blob.is_none());
            let fragments = written.default.0;

            assert_eq!(fragments.len(), 1);
            let fragment = &fragments[0];
            assert_eq!(fragment.files.len(), 1);
            assert_eq!(fragment.physical_rows, Some(1024));
            assert_eq!(
                fragment.files[0].file_major_version, major,
                "version: {}",
                version
            );
            assert_eq!(
                fragment.files[0].file_minor_version, minor,
                "version: {}",
                version
            );
        }
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
            data_storage_version: Some(LanceFileVersion::Legacy),
            ..Default::default()
        };
        let data_stream = Box::pin(RecordBatchStreamAdapter::new(
            Arc::new(arrow_schema),
            futures::stream::iter(std::iter::once(Ok(data.clone()))),
        ));

        let object_store = Arc::new(ObjectStore::memory());
        let base_path = Path::from("test");
        let written = write_fragments_internal(
            None,
            object_store.clone(),
            &base_path,
            schema.clone(),
            data_stream,
            write_params,
        )
        .await
        .unwrap();

        assert!(written.blob.is_none());
        let fragments = written.default.0;

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
