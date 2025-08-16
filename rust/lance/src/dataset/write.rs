// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::num::NonZero;
use std::sync::Arc;

use arrow_array::RecordBatch;
use chrono::TimeDelta;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::{Stream, StreamExt, TryStreamExt};
use lance_core::datatypes::{
    NullabilityComparison, OnMissing, OnTypeMismatch, SchemaCompareOptions, StorageClass,
};
use lance_core::error::LanceOptionExt;
use lance_core::utils::tracing::{AUDIT_MODE_CREATE, AUDIT_TYPE_DATA, TRACE_FILE_AUDIT};
use lance_core::{datatypes::Schema, Error, Result};
use lance_datafusion::chunker::{break_stream, chunk_stream};
use lance_datafusion::spill::{create_replay_spill, SpillReceiver, SpillSender};
use lance_datafusion::utils::StreamingWriteSource;
use lance_file::v2;
use lance_file::v2::writer::FileWriterOptions;
use lance_file::version::LanceFileVersion;
use lance_file::writer::{FileWriter, ManifestProvider};
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use lance_table::format::{DataFile, Fragment};
use lance_table::io::commit::{commit_handler_from_url, CommitHandler};
use lance_table::io::manifest::ManifestDescribing;
use object_store::path::Path;
use snafu::location;
use tracing::{info, instrument};
use uuid::Uuid;

use crate::session::Session;
use crate::Dataset;

use super::blob::BlobStreamExt;
use super::progress::{NoopFragmentWriteProgress, WriteFragmentProgress};
use super::transaction::Transaction;
use super::DATA_DIR;

mod commit;
pub mod delete;
mod insert;
pub mod merge_insert;
mod retry;
pub mod update;

pub use commit::CommitBuilder;
pub use delete::DeleteBuilder;
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

    pub fn uri(&self) -> String {
        match self {
            WriteDestination::Dataset(dataset) => dataset.uri.clone(),
            WriteDestination::Uri(uri) => uri.to_string(),
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

/// Auto cleanup parameters
#[derive(Debug, Clone)]
pub struct AutoCleanupParams {
    pub interval: usize,
    pub older_than: TimeDelta,
}

impl Default for AutoCleanupParams {
    fn default() -> Self {
        Self {
            interval: 20,
            older_than: TimeDelta::days(14),
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

    /// If set to true, and this is a new dataset, uses the new v3 manifest paths.
    /// These use reversed binary representation for S3 throughput optimization
    /// and include a _latest_manifest.json file for best-effort latest version tracking.
    /// This parameter has no effect on existing datasets.
    /// Default is False.
    pub enable_v3_manifest_paths: bool,

    pub session: Option<Arc<Session>>,

    /// If Some and this is a new dataset, old dataset versions will be
    /// automatically cleaned up according to the parameters set out in
    /// `AutoCleanupParams`. This parameter has no effect on existing datasets.
    /// To add autocleaning to an existing dataset, use Dataset::update_config
    /// to set lance.auto_cleanup.interval and lance.auto_cleanup.older_than.
    /// Both parameters must be set to invoke autocleaning.
    pub auto_cleanup: Option<AutoCleanupParams>,

    /// Batch size for loading head manifests when checking for concurrent writes
    /// in V3 manifest naming scheme. Smaller values can help reduce memory usage
    /// but may increase the number of object store requests.
    /// Default is 8.
    pub head_manifests_batch_size: usize,

    /// If true, skip auto cleanup during commits. This should be set to true
    /// for high frequency writes to improve performance. This is also useful
    /// if the writer does not have delete permissions and the clean up would
    /// just try and log a failure anyway. Default is false.
    pub skip_auto_cleanup: bool,

    /// Configuration key-value pairs for this write operation.
    /// This can include commit messages, engine information, etc.
    /// this properties map will be persisted as part of Transaction object.
    pub transaction_properties: Option<Arc<HashMap<String, String>>>,

    /// If true, create a detached commit that is not part of the mainline history.
    /// Detached commits will never show up in the dataset's history.
    /// This can be used to stage changes or to handle "secondary" datasets
    /// whose lineage is tracked elsewhere. Default is false.
    pub detached: bool,
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
            enable_v3_manifest_paths: false,
            session: None,
            auto_cleanup: Some(AutoCleanupParams::default()),
            head_manifests_batch_size: 8,
            skip_auto_cleanup: false,
            transaction_properties: None,
            detached: false,
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

    pub fn store_registry(&self) -> Arc<ObjectStoreRegistry> {
        self.session
            .as_ref()
            .map(|s| s.store_registry())
            .unwrap_or_default()
    }

    /// Set the properties for this WriteParams.
    pub fn with_transaction_properties(self, properties: HashMap<String, String>) -> Self {
        Self {
            transaction_properties: Some(Arc::new(properties)),
            ..self
        }
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
    data: impl StreamingWriteSource,
    params: WriteParams,
) -> Result<Transaction> {
    InsertBuilder::new(dest.into())
        .with_params(&params)
        .execute_uncommitted_stream(data)
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
            info!(target: TRACE_FILE_AUDIT, mode=AUDIT_MODE_CREATE, r#type=AUDIT_TYPE_DATA, path = &data_file.path);
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
        info!(target: TRACE_FILE_AUDIT, mode=AUDIT_MODE_CREATE, r#type=AUDIT_TYPE_DATA, path = &data_file.path);
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
                        compare_dictionary: dataset.is_legacy_storage(),
                        ..Default::default()
                    },
                )?;
                // Project from the dataset schema, because it has the correct field ids.
                let write_schema = dataset.schema().project_by_schema(
                    &schema,
                    OnMissing::Error,
                    OnTypeMismatch::Error,
                )?;
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

    let data_schema = schema.project_by_schema(
        data.schema().as_ref(),
        OnMissing::Error,
        OnTypeMismatch::Error,
    )?;

    let (data, blob_data) = data.extract_blob_stream(&data_schema);

    // Some params we borrow from the normal write, some we override
    let blob_write_params = WriteParams {
        store_params: params.store_params.clone(),
        commit_handler: params.commit_handler.clone(),
        data_storage_version: params.data_storage_version,
        enable_move_stable_row_ids: true,
        // This shouldn't really matter since all commits are detached
        enable_v2_manifest_paths: true,
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
        let size_bytes = self.0.tell().await?;
        Ok((
            self.0.finish().await? as u32,
            DataFile::new_legacy(
                self.1.clone(),
                self.0.schema(),
                NonZero::new(size_bytes as u64),
            ),
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
        let num_rows = self.writer.finish().await? as u32;
        let data_file = DataFile::new(
            std::mem::take(&mut self.path),
            field_ids,
            column_indices,
            major,
            minor,
            NonZero::new(self.writer.tell().await?),
        );
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
            #[allow(deprecated)]
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

/// Create an iterator of record batch streams from the given source.
///
/// If `enable_retries` is true, then the source will be saved either in memory
/// or spilled to disk to allow replaying the source in case of a failure. The
/// source will be kept in memory if either (1) the size hint shows that
/// there is only one batch or (2) the stream contains less than 100MB of
/// data. Otherwise, the source will be spilled to a temporary file on disk.
///
/// This is used to support retries on write operations.
async fn new_source_iter(
    source: SendableRecordBatchStream,
    enable_retries: bool,
) -> Result<Box<dyn Iterator<Item = SendableRecordBatchStream> + Send + 'static>> {
    if enable_retries {
        let schema = source.schema();

        // If size hint shows there is only one batch, spilling has no benefit, just keep that
        // in memory. (This is a pretty common case.)
        let size_hint = source.size_hint();
        if size_hint.0 == 1 && size_hint.1 == Some(1) {
            let batches: Vec<RecordBatch> = source.try_collect().await?;
            Ok(Box::new(std::iter::repeat_with(move || {
                Box::pin(RecordBatchStreamAdapter::new(
                    schema.clone(),
                    futures::stream::iter(batches.clone().into_iter().map(Ok)),
                )) as SendableRecordBatchStream
            })))
        } else {
            // Allow buffering up to 100MB in memory before spilling to disk.
            Ok(Box::new(
                SpillStreamIter::try_new(source, 100 * 1024 * 1024).await?,
            ))
        }
    } else {
        Ok(Box::new(std::iter::once(source)))
    }
}

struct SpillStreamIter {
    receiver: SpillReceiver,
    #[allow(dead_code)] // Exists to keep the SpillSender alive
    sender_handle: tokio::task::JoinHandle<SpillSender>,
    // This temp dir is used to store the spilled data. It is kept alive by
    // this struct. When this struct is dropped, the Drop implementation of
    // tempfile::TempDir will delete the temp dir.
    #[allow(dead_code)] // Exists to keep the temp dir alive
    tmp_dir: tempfile::TempDir,
}

impl SpillStreamIter {
    pub async fn try_new(
        mut source: SendableRecordBatchStream,
        memory_limit: usize,
    ) -> Result<Self> {
        let tmp_dir = tokio::task::spawn_blocking(|| {
            tempfile::tempdir().map_err(|e| Error::InvalidInput {
                source: format!("Failed to create temp dir: {}", e).into(),
                location: location!(),
            })
        })
        .await
        .ok()
        .expect_ok()??;

        let tmp_path = tmp_dir.path().join("spill.arrows");
        let (mut sender, receiver) = create_replay_spill(tmp_path, source.schema(), memory_limit);

        let sender_handle = tokio::task::spawn(async move {
            while let Some(res) = source.next().await {
                match res {
                    Ok(batch) => match sender.write(batch).await {
                        Ok(_) => {}
                        Err(e) => {
                            sender.send_error(e);
                            break;
                        }
                    },
                    Err(e) => {
                        sender.send_error(e);
                        break;
                    }
                }
            }

            if let Err(err) = sender.finish().await {
                sender.send_error(err);
            }
            sender
        });

        Ok(Self {
            receiver,
            tmp_dir,
            sender_handle,
        })
    }
}

impl Iterator for SpillStreamIter {
    type Item = SendableRecordBatchStream;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.receiver.read())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::types::{Float32Type, Int32Type};
    use arrow_array::{Int32Array, RecordBatchReader, StructArray};
    use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};
    use datafusion::{error::DataFusionError, physical_plan::stream::RecordBatchStreamAdapter};
    use futures::TryStreamExt;
    use lance_datafusion::datagen::DatafusionDatagenExt;
    use lance_datagen::{array, gen_batch, BatchCount, RowCount};
    use lance_file::reader::FileReader;
    use lance_io::traits::Reader;
    use lance_table::format::is_detached_version;
    use lance_table::io::commit::{read_latest_manifest_hint_best_effort, ManifestNamingScheme};

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
            gen_batch()
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

    #[tokio::test]
    async fn test_v3_manifest_naming_scheme_write_and_read() {
        let write_params = WriteParams {
            enable_v3_manifest_paths: true,
            max_rows_per_file: 100,
            ..Default::default()
        };

        let mut dataset = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col("value", lance_datagen::array::rand::<Float32Type>())
            .into_ram_dataset_with_params(
                FragmentCount::from(1),
                FragmentRowCount::from(50),
                Some(write_params.clone()),
            )
            .await
            .unwrap();

        // Verify V3 naming scheme is used
        assert_eq!(
            dataset.manifest_location.naming_scheme,
            ManifestNamingScheme::V3
        );
        assert_eq!(dataset.manifest.version, 1);

        // Verify manifest file exists with correct naming
        let expected_manifest_path = dataset
            .base
            .child("_versions")
            .child("1000000000000000000000000000000000000000000000000000000000000000.manifest");
        let manifest_exists = dataset
            .object_store
            .exists(&expected_manifest_path)
            .await
            .unwrap();
        assert!(
            manifest_exists,
            "V3 manifest file should exist at {:?}",
            expected_manifest_path
        );

        // Verify _latest_manifest_hint.json exists and contains correct version
        let latest_version =
            read_latest_manifest_hint_best_effort(&dataset.object_store, &dataset.base)
                .await
                .unwrap();
        assert_eq!(latest_version.unwrap().version, 1);

        // Create version 2
        let append_data1 = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::step_custom::<Int32Type>(100, 1))
            .col("value", lance_datagen::array::rand::<Float32Type>())
            .into_reader_rows(RowCount::from(25), BatchCount::from(1));

        dataset.append(append_data1, None).await.unwrap();
        assert_eq!(dataset.manifest.version, 2);

        // Verify the second manifest file exists with correct naming
        let expected_manifest_path_v2 = dataset
            .base
            .child("_versions")
            .child("0100000000000000000000000000000000000000000000000000000000000000.manifest");
        let manifest_v2_exists = dataset
            .object_store
            .exists(&expected_manifest_path_v2)
            .await
            .unwrap();
        assert!(
            manifest_v2_exists,
            "V3 manifest file for version 2 should exist at {:?}",
            expected_manifest_path_v2
        );

        // Verify _latest_manifest_hint.json is updated to version 2
        let latest_version =
            read_latest_manifest_hint_best_effort(&dataset.object_store, &dataset.base)
                .await
                .unwrap();
        assert_eq!(latest_version.map(|v| v.version), Some(2));

        // Create version 3
        let append_data2 = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::step_custom::<Int32Type>(125, 1))
            .col("value", lance_datagen::array::rand::<Float32Type>())
            .into_reader_rows(RowCount::from(25), BatchCount::from(1));

        dataset.append(append_data2, None).await.unwrap();
        assert_eq!(dataset.manifest.version, 3);

        // Verify third manifest file exists correct naming
        let expected_manifest_path_v3 = dataset
            .base
            .child("_versions")
            .child("1100000000000000000000000000000000000000000000000000000000000000.manifest");
        let manifest_v3_exists = dataset
            .object_store
            .exists(&expected_manifest_path_v3)
            .await
            .unwrap();
        assert!(
            manifest_v3_exists,
            "V3 manifest file for version 3 should exist at {:?}",
            expected_manifest_path_v3
        );

        // Verify _latest_manifest.json is updated to version 3
        let latest_version =
            read_latest_manifest_hint_best_effort(&dataset.object_store, &dataset.base)
                .await
                .unwrap();
        assert_eq!(latest_version.map(|v| v.version), Some(3));

        // Verify final dataset state and data integrity
        let scan = dataset.scan();
        let batches: Vec<RecordBatch> = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        // The initial data creates 50 rows
        // Then we append 25 rows twice
        assert_eq!(total_rows, 100); // After two append operations (50 + 25 + 25)
    }

    #[tokio::test]
    async fn test_v3_manifest_naming_scheme_concurrent_write() {
        let write_params = WriteParams {
            enable_v3_manifest_paths: true,
            max_rows_per_file: 100,
            ..Default::default()
        };

        // Create initial dataset with V3 naming scheme using in-memory storage
        let mut dataset = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col("value", lance_datagen::array::rand::<Float32Type>())
            .into_ram_dataset_with_params(
                FragmentCount::from(1),
                FragmentRowCount::from(50),
                Some(write_params.clone()),
            )
            .await
            .unwrap();

        // Verify V3 naming scheme is used
        assert_eq!(
            dataset.manifest_location.naming_scheme,
            ManifestNamingScheme::V3
        );
        assert_eq!(dataset.manifest.version, 1);

        // Check that _latest_manifest_hint.json exists
        let latest_manifest_path = dataset
            .base
            .child("_versions")
            .child("_latest_manifest_hint.json");
        assert!(dataset
            .object_store
            .exists(&latest_manifest_path)
            .await
            .unwrap());

        // Add 4 new versions using lightweight UpdateConfig transactions (simulating intermediate commits)
        for i in 1..=4 {
            dataset
                .update_config([(format!("test_key_{}", i), format!("test_value_{}", i))])
                .await
                .unwrap();
        }

        // Verify _latest_manifest_hint.json is updated to version 5
        let latest_version =
            read_latest_manifest_hint_best_effort(&dataset.object_store, &dataset.base)
                .await
                .unwrap();
        assert_eq!(latest_version.map(|v| v.version), Some(5));

        // Test Case 1: Concurrent write with default batch size
        let concurrent_dataset_1 = dataset.checkout_version(1).await.unwrap();
        assert_eq!(concurrent_dataset_1.manifest.version, 1);

        let concurrent_data_1 = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::step_custom::<Int32Type>(200, 1))
            .col("value", lance_datagen::array::rand::<Float32Type>())
            .into_df_stream(RowCount::from(25), BatchCount::from(1));

        let rebased_dataset_1 =
            InsertBuilder::new(WriteDestination::Dataset(Arc::new(concurrent_dataset_1)))
                .with_params(&WriteParams {
                    mode: WriteMode::Append,
                    ..write_params.clone()
                })
                .execute_stream(concurrent_data_1)
                .await
                .unwrap();

        // This should succeed and create version 6 (after rebasing over versions 2-5)
        assert_eq!(rebased_dataset_1.manifest.version, 6);

        // Verify read after write for test case 1
        let scan_1 = rebased_dataset_1.scan();
        let batches_1: Vec<RecordBatch> = scan_1
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let total_rows_1: usize = batches_1.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows_1, 75); // 50 (initial) + 25 (concurrent append 1)

        // Test Case 2: Concurrent write with small batch size (1)
        let concurrent_dataset_2 = dataset.checkout_version(2).await.unwrap();
        assert_eq!(concurrent_dataset_2.manifest.version, 2);

        let concurrent_data_2 = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::step_custom::<Int32Type>(300, 1))
            .col("value", lance_datagen::array::rand::<Float32Type>())
            .into_df_stream(RowCount::from(30), BatchCount::from(1));

        let rebased_dataset_2 =
            InsertBuilder::new(WriteDestination::Dataset(Arc::new(concurrent_dataset_2)))
                .with_params(&WriteParams {
                    mode: WriteMode::Append,
                    head_manifests_batch_size: 1, // Small batch size - forces multiple batches
                    ..write_params.clone()
                })
                .execute_stream(concurrent_data_2)
                .await
                .unwrap();

        // This should succeed and create version 7 (after rebasing over versions 3-6)
        assert_eq!(rebased_dataset_2.manifest.version, 7);

        // Verify read after write for test case 2
        let scan_2 = rebased_dataset_2.scan();
        let batches_2: Vec<RecordBatch> = scan_2
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let total_rows_2: usize = batches_2.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows_2, 105); // 50 (initial) + 25 (concurrent append 1) + 30 (concurrent append 2)

        // Test Case 3: Concurrent write with large batch size (20)
        let concurrent_dataset_3 = dataset.checkout_version(3).await.unwrap();
        assert_eq!(concurrent_dataset_3.manifest.version, 3);

        let concurrent_data_3 = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::step_custom::<Int32Type>(400, 1))
            .col("value", lance_datagen::array::rand::<Float32Type>())
            .into_df_stream(RowCount::from(20), BatchCount::from(1));

        let rebased_dataset_3 =
            InsertBuilder::new(WriteDestination::Dataset(Arc::new(concurrent_dataset_3)))
                .with_params(&WriteParams {
                    mode: WriteMode::Append,
                    head_manifests_batch_size: 20, // Large batch size - single batch
                    ..write_params.clone()
                })
                .execute_stream(concurrent_data_3)
                .await
                .unwrap();

        // This should succeed and create version 8 (after rebasing over versions 4-7)
        assert_eq!(rebased_dataset_3.manifest.version, 8);

        // Verify read after write for test case 3
        let scan_3 = rebased_dataset_3.scan();
        let batches_3: Vec<RecordBatch> = scan_3
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let total_rows_3: usize = batches_3.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows_3, 125); // 50 (initial) + 25 (concurrent append 1) + 30 (concurrent append 2) + 20 (concurrent append 3)

        // Verify _latest_manifest.json is updated to version 8
        let latest_version = read_latest_manifest_hint_best_effort(
            &rebased_dataset_3.object_store,
            &rebased_dataset_3.base,
        )
        .await
        .unwrap();
        assert_eq!(latest_version.map(|v| v.version), Some(8));

        // Use the latest dataset to verify final state
        let final_dataset = &rebased_dataset_3;
        assert_eq!(final_dataset.version().version, 8);

        // Verify all versions exist
        for version in 1..=8 {
            let version_dataset = final_dataset.checkout_version(version).await.unwrap();
            assert_eq!(version_dataset.manifest.version, version);
        }

        // Verify final dataset has all the data (read after write)
        let scan = final_dataset.scan();
        let batches: Vec<RecordBatch> = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        // Initial: 50, Concurrent append 1: 25, Concurrent append 2: 30, Concurrent append 3: 20
        assert_eq!(total_rows, 125); // 50 + 25 + 30 + 20

        // Test that the V3 batched existence checking works correctly
        // by verifying the latest version
        let latest_version_id = final_dataset.latest_version_id().await.unwrap();
        let fresh_dataset = final_dataset
            .checkout_version(latest_version_id)
            .await
            .unwrap();
        assert_eq!(fresh_dataset.manifest.version, 8);

        // Verify read after write works correctly
        let fresh_scan = fresh_dataset.scan();
        let fresh_batches: Vec<RecordBatch> = fresh_scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let fresh_total_rows: usize = fresh_batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(fresh_total_rows, 125);
    }

    #[tokio::test]
    async fn test_v3_manifest_naming_scheme_checkout_latest() {
        let write_params = WriteParams {
            enable_v3_manifest_paths: true,
            max_rows_per_file: 100,
            ..Default::default()
        };

        // Create initial dataset with V3 naming scheme using in-memory storage
        let mut dataset = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col("value", lance_datagen::array::rand::<Float32Type>())
            .into_ram_dataset_with_params(
                FragmentCount::from(1),
                FragmentRowCount::from(50),
                Some(write_params.clone()),
            )
            .await
            .unwrap();

        // Verify V3 naming scheme is used
        assert_eq!(
            dataset.manifest_location.naming_scheme,
            ManifestNamingScheme::V3
        );
        assert_eq!(dataset.manifest.version, 1);

        // Add several more versions to test checkout_latest
        for i in 2..=5 {
            dataset
                .update_config([(format!("version_{}", i), i.to_string())])
                .await
                .unwrap();
            assert_eq!(dataset.manifest.version, i);
        }

        // Test Case 1: Latest version hint is correctly pointing to the latest version
        {
            let latest_version =
                read_latest_manifest_hint_best_effort(&dataset.object_store, &dataset.base)
                    .await
                    .unwrap();
            assert_eq!(latest_version.map(|v| v.version), Some(5));

            // Checkout an older version and then checkout_latest
            let mut old_dataset = dataset.checkout_version(3).await.unwrap();
            assert_eq!(old_dataset.manifest.version, 3);

            // checkout_latest should bring us to version 5
            old_dataset.checkout_latest().await.unwrap();
            assert_eq!(old_dataset.manifest.version, 5);
        }

        // Test Case 2: Latest version hint is pointing to an old version
        {
            // Manually corrupt the latest version hint to point to an older version
            use lance_table::io::commit::LatestManifestHint;

            let corrupted_hint = LatestManifestHint {
                version: 3, // Point to older version instead of 5
                size: None,
                e_tag: None,
            };

            let hint_path = dataset
                .base
                .child("_versions")
                .child("_latest_manifest_hint.json");
            let hint_json = serde_json::to_vec(&corrupted_hint).unwrap();
            dataset
                .object_store
                .put(&hint_path, hint_json.as_slice())
                .await
                .unwrap();

            // Verify the hint now points to version 3
            let latest_version =
                read_latest_manifest_hint_best_effort(&dataset.object_store, &dataset.base)
                    .await
                    .unwrap();
            assert_eq!(latest_version.map(|v| v.version), Some(3));

            // Checkout an older version and then checkout_latest
            let mut old_dataset = dataset.checkout_version(2).await.unwrap();
            assert_eq!(old_dataset.manifest.version, 2);

            // checkout_latest should still find the actual latest version (5)
            // even though the hint points to version 3
            old_dataset.checkout_latest().await.unwrap();
            assert_eq!(old_dataset.manifest.version, 5);
        }

        // Test Case 3: Latest version hint does not exist
        {
            // Delete the latest version hint file
            let hint_path = dataset
                .base
                .child("_versions")
                .child("_latest_manifest_hint.json");
            dataset.object_store.delete(&hint_path).await.unwrap();

            // Verify the hint no longer exists
            let latest_version =
                read_latest_manifest_hint_best_effort(&dataset.object_store, &dataset.base)
                    .await
                    .unwrap();
            assert!(latest_version.is_none());

            // Checkout an older version and then checkout_latest
            let mut old_dataset = dataset.checkout_version(1).await.unwrap();
            assert_eq!(old_dataset.manifest.version, 1);

            // checkout_latest should still find the actual latest version (5)
            // by scanning all manifest files
            old_dataset.checkout_latest().await.unwrap();
            assert_eq!(old_dataset.manifest.version, 5);
        }
    }

    #[tokio::test]
    async fn test_v3_manifest_naming_scheme_checkout_version() {
        let write_params = WriteParams {
            enable_v3_manifest_paths: true,
            max_rows_per_file: 100,
            ..Default::default()
        };

        // Create initial dataset with V3 naming scheme using in-memory storage
        let mut dataset = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col("value", lance_datagen::array::rand::<Float32Type>())
            .into_ram_dataset_with_params(
                FragmentCount::from(1),
                FragmentRowCount::from(50),
                Some(write_params.clone()),
            )
            .await
            .unwrap();

        // Verify V3 naming scheme is used
        assert_eq!(
            dataset.manifest_location.naming_scheme,
            ManifestNamingScheme::V3
        );
        assert_eq!(dataset.manifest.version, 1);

        // Test Case 1: Checkout the first version after the dataset is created
        {
            let version_1_dataset = dataset.checkout_version(1).await.unwrap();
            assert_eq!(version_1_dataset.manifest.version, 1);

            // Verify the dataset content is accessible
            let scan = version_1_dataset.scan();
            let batches: Vec<RecordBatch> = scan
                .try_into_stream()
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total_rows, 50); // Initial dataset has 50 rows
        }

        // Create a few more versions using append operations
        for i in 2..=5 {
            let append_data = lance_datagen::gen_batch()
                .col(
                    "id",
                    lance_datagen::array::step_custom::<Int32Type>(i * 100, 1),
                )
                .col("value", lance_datagen::array::rand::<Float32Type>())
                .into_reader_rows(RowCount::from(10), BatchCount::from(1));

            dataset
                .append(append_data, Some(write_params.clone()))
                .await
                .unwrap();
            assert_eq!(dataset.manifest.version, i as u64);
        }

        // Now we have versions 1, 2, 3, 4, 5
        assert_eq!(dataset.manifest.version, 5);

        // Test Case 2: Checkout the first version
        {
            let version_1_dataset = dataset.checkout_version(1).await.unwrap();
            assert_eq!(version_1_dataset.manifest.version, 1);

            // Verify the content is only the original data
            let scan = version_1_dataset.scan();
            let batches: Vec<RecordBatch> = scan
                .try_into_stream()
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total_rows, 50); // Only original 50 rows
        }

        // Test Case 3: Checkout a middle version
        {
            let version_3_dataset = dataset.checkout_version(3).await.unwrap();
            assert_eq!(version_3_dataset.manifest.version, 3);

            // Verify the content includes original + 2 appends
            let scan = version_3_dataset.scan();
            let batches: Vec<RecordBatch> = scan
                .try_into_stream()
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total_rows, 70); // 50 (original) + 10 (v2) + 10 (v3)
        }

        // Test Case 4: Explicitly checkout the latest version
        {
            let version_5_dataset = dataset.checkout_version(5).await.unwrap();
            assert_eq!(version_5_dataset.manifest.version, 5);

            // Verify the content includes all data
            let scan = version_5_dataset.scan();
            let batches: Vec<RecordBatch> = scan
                .try_into_stream()
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total_rows, 90); // 50 (original) + 10 (v2) + 10 (v3) + 10 (v4) + 10 (v5)
        }

        // Test Case 5: Verify that all versions can be checked out sequentially
        for version in 1..=5 {
            let version_dataset = dataset.checkout_version(version).await.unwrap();
            assert_eq!(version_dataset.manifest.version, version);

            // Each version should have the expected number of rows
            let scan = version_dataset.scan();
            let batches: Vec<RecordBatch> = scan
                .try_into_stream()
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            let expected_rows = 50 + (version - 1) * 10; // 50 initial + 10 per additional version
            assert_eq!(total_rows, expected_rows as usize);
        }

        // Test Case 6: Verify checkout works from any version to any other version
        {
            let version_2_dataset = dataset.checkout_version(2).await.unwrap();
            assert_eq!(version_2_dataset.manifest.version, 2);

            // From version 2, checkout version 4
            let version_4_dataset = version_2_dataset.checkout_version(4).await.unwrap();
            assert_eq!(version_4_dataset.manifest.version, 4);

            // Verify content is correct for version 4
            let scan = version_4_dataset.scan();
            let batches: Vec<RecordBatch> = scan
                .try_into_stream()
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total_rows, 80); // 50 + 10 + 10 + 10
        }
    }

    #[tokio::test]
    async fn test_v3_manifest_naming_scheme_commit_checkout_detached() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        // Create initial dataset with V3 manifest scheme
        let write_params = WriteParams {
            enable_v3_manifest_paths: true,
            max_rows_per_file: 50,
            max_rows_per_group: 25,
            ..Default::default()
        };

        let initial_data = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col("value", lance_datagen::array::rand::<Float32Type>())
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));

        let mut dataset = Dataset::write(initial_data, test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        // Add a few normal versions
        for i in 1..=3 {
            let append_data = lance_datagen::gen_batch()
                .col(
                    "id",
                    lance_datagen::array::step_custom::<Int32Type>(100 * i, 1),
                )
                .col("value", lance_datagen::array::rand::<Float32Type>())
                .into_reader_rows(RowCount::from(20), BatchCount::from(1));

            dataset
                .append(
                    append_data,
                    Some(WriteParams {
                        mode: WriteMode::Append,
                        ..write_params.clone()
                    }),
                )
                .await
                .unwrap();
        }

        let current_version = dataset.version().version;
        assert_eq!(current_version, 4);
        assert_eq!(
            dataset.manifest_location.naming_scheme,
            ManifestNamingScheme::V3
        );

        // Test committing a detached version
        let detached_data = lance_datagen::gen_batch()
            .col(
                "id",
                lance_datagen::array::step_custom::<Int32Type>(1000, 1),
            )
            .col("value", lance_datagen::array::rand::<Float32Type>())
            .into_df_stream(RowCount::from(10), BatchCount::from(1));

        let detached_dataset = InsertBuilder::new(WriteDestination::Dataset(Arc::new(dataset)))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                detached: true,
                ..write_params.clone()
            })
            .execute_stream(detached_data)
            .await
            .unwrap();

        // Verify detached version properties
        let detached_version = detached_dataset.version().version;
        assert!(is_detached_version(detached_version));
        assert_eq!(
            detached_dataset.manifest_location.naming_scheme,
            ManifestNamingScheme::V3
        );

        // Verify V3 detached manifest file uses normal binary format (ends with "1" due to highest bit set)
        let manifest_path =
            ManifestNamingScheme::V3.manifest_path(&detached_dataset.base, detached_version);
        let filename = manifest_path.filename().unwrap();
        let stem = filename
            .strip_suffix(".manifest")
            .expect("filename should end with .manifest");
        assert!(
            stem.ends_with("1"),
            "V3 detached version binary should end with '1' due to DETACHED_VERSION_MASK, got: {}",
            stem
        );
        // Verify it's a 64-bit binary string
        assert_eq!(
            stem.len(),
            64,
            "V3 manifest filename should be 64-bit binary string, got length: {}",
            stem.len()
        );
        // Verify it only contains '0' and '1'
        assert!(
            stem.chars().all(|c| c == '0' || c == '1'),
            "V3 manifest filename should be binary, got: {}",
            stem
        );

        // Test checkout detached version by version number
        let checkout_detached = Dataset::open(test_uri)
            .await
            .unwrap()
            .checkout_version(detached_version)
            .await
            .unwrap();

        assert_eq!(checkout_detached.version().version, detached_version);
        assert_eq!(
            checkout_detached.manifest_location.naming_scheme,
            ManifestNamingScheme::V3
        );
        assert!(is_detached_version(checkout_detached.version().version));

        // Verify data integrity in detached version
        let detached_count = checkout_detached.count_rows(None).await.unwrap();
        let expected_count = 100 + 20 * 3 + 10; // initial + 3 appends + detached append
        assert_eq!(detached_count, expected_count);

        // Test that we can still checkout the main branch
        let main_dataset = Dataset::open(test_uri).await.unwrap();
        assert_eq!(main_dataset.version().version, current_version);
        assert!(!is_detached_version(main_dataset.version().version));

        let main_count = main_dataset.count_rows(None).await.unwrap();
        let expected_main_count = 100 + 20 * 3; // initial + 3 appends (no detached)
        assert_eq!(main_count, expected_main_count);

        // Test that checkout by version works for both detached and regular versions
        let checkout_v1 = Dataset::open(test_uri)
            .await
            .unwrap()
            .checkout_version(1)
            .await
            .unwrap();
        assert_eq!(checkout_v1.version().version, 1);
        let v1_count = checkout_v1.count_rows(None).await.unwrap();
        assert_eq!(v1_count, 100);
    }
}
