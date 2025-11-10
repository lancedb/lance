// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::RecordBatch;
use chrono::TimeDelta;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::{Stream, StreamExt, TryStreamExt};
use lance_core::datatypes::{
    BlobVersion, NullabilityComparison, OnMissing, OnTypeMismatch, SchemaCompareOptions,
};
use lance_core::error::LanceOptionExt;
use lance_core::utils::tempfile::TempDir;
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
use lance_table::format::{BasePath, DataFile, Fragment};
use lance_table::io::commit::{commit_handler_from_url, CommitHandler};
use lance_table::io::manifest::ManifestDescribing;
use object_store::path::Path;
use snafu::location;
use std::collections::HashMap;
use std::num::NonZero;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use tracing::{info, instrument};

use crate::session::Session;
use crate::Dataset;

use super::fragment::write::generate_random_filename;
use super::progress::{NoopFragmentWriteProgress, WriteFragmentProgress};
use super::transaction::Transaction;
use super::utils::SchemaAdapter;
use super::DATA_DIR;

fn blob_version_for(storage_version: LanceFileVersion) -> BlobVersion {
    if storage_version >= LanceFileVersion::V2_2 {
        BlobVersion::V2
    } else {
        BlobVersion::V1
    }
}

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

    /// Experimental: if set to true, the writer will use stable row ids.
    /// These row ids are stable after compaction operations, but not after updates.
    /// This makes compaction more efficient, since with stable row ids no
    /// secondary indices need to be updated to point to new row ids.
    pub enable_stable_row_ids: bool,

    /// If set to true, and this is a new dataset, uses the new v2 manifest paths.
    /// These allow constant-time lookups for the latest manifest on object storage.
    /// This parameter has no effect on existing datasets. To migrate an existing
    /// dataset, use the [`super::Dataset::migrate_manifest_paths_v2`] method.
    /// Default is False.
    pub enable_v2_manifest_paths: bool,

    pub session: Option<Arc<Session>>,

    /// If Some and this is a new dataset, old dataset versions will be
    /// automatically cleaned up according to the parameters set out in
    /// [`AutoCleanupParams`]. This parameter has no effect on existing datasets.
    /// To add auto-cleanup to an existing dataset, use [`Dataset::update_config`]
    /// to set `lance.auto_cleanup.interval` and `lance.auto_cleanup.older_than`.
    /// Both parameters must be set to invoke auto-cleanup.
    pub auto_cleanup: Option<AutoCleanupParams>,

    /// If true, skip auto cleanup during commits. This should be set to true
    /// for high frequency writes to improve performance. This is also useful
    /// if the writer does not have delete permissions and the clean up would
    /// just try and log a failure anyway. Default is false.
    pub skip_auto_cleanup: bool,

    /// Configuration key-value pairs for this write operation.
    /// This can include commit messages, engine information, etc.
    /// this properties map will be persisted as part of Transaction object.
    pub transaction_properties: Option<Arc<HashMap<String, String>>>,

    /// New base paths to register in the manifest during dataset creation.
    /// Each BasePath must have a properly assigned ID (non-zero).
    /// Only used in CREATE/OVERWRITE modes for manifest registration.
    /// IDs should be assigned by the caller before passing to WriteParams.
    pub initial_bases: Option<Vec<BasePath>>,

    /// Target base IDs for writing data files.
    /// When provided, all new data files will be written to bases with these IDs.
    /// Used in all modes (CREATE, APPEND, OVERWRITE) to specify where data should be written.
    /// The IDs must correspond to either:
    /// - IDs in initial_bases (for CREATE/OVERWRITE modes)
    /// - IDs already registered in the existing dataset manifest (for APPEND mode)
    pub target_bases: Option<Vec<u32>>,

    /// Target base names or paths as strings (unresolved).
    /// These will be resolved to IDs when the write operation executes.
    /// Resolution happens at builder execution time when dataset context is available.
    pub target_base_names_or_paths: Option<Vec<String>>,
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
            enable_stable_row_ids: false,
            enable_v2_manifest_paths: false,
            session: None,
            auto_cleanup: Some(AutoCleanupParams::default()),
            skip_auto_cleanup: false,
            transaction_properties: None,
            initial_bases: None,
            target_bases: None,
            target_base_names_or_paths: None,
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

    /// Set the initial_bases for this WriteParams.
    ///
    /// This specifies new base paths to register in the manifest during dataset creation.
    /// Each BasePath must have a properly assigned ID (non-zero) before calling this method.
    /// Only used in CREATE/OVERWRITE modes for manifest registration.
    pub fn with_initial_bases(self, bases: Vec<BasePath>) -> Self {
        Self {
            initial_bases: Some(bases),
            ..self
        }
    }

    /// Set the target_bases for this WriteParams.
    ///
    /// This specifies the base IDs where data files should be written.
    /// The IDs must correspond to either:
    /// - IDs in initial_bases (for CREATE/OVERWRITE modes)
    /// - IDs already registered in the existing dataset manifest (for APPEND mode)
    pub fn with_target_bases(self, base_ids: Vec<u32>) -> Self {
        Self {
            target_bases: Some(base_ids),
            ..self
        }
    }

    /// Store target base names or paths for deferred resolution.
    ///
    /// This method stores the references in `target_base_names_or_paths` field
    /// to be resolved later at execution time when the dataset manifest is available.
    ///
    /// Resolution will happen at write execution time and will try to match:
    /// 1. initial_bases by name
    /// 2. initial_bases by path
    /// 3. existing manifest by name
    /// 4. existing manifest by path
    ///
    /// # Arguments
    ///
    /// * `references` - Vector of base names or paths to be resolved later
    pub fn with_target_base_names_or_paths(self, references: Vec<String>) -> Self {
        Self {
            target_base_names_or_paths: Some(references),
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
    target_bases_info: Option<Vec<TargetBaseInfo>>,
) -> Result<Vec<Fragment>> {
    let adapter = SchemaAdapter::new(data.schema());
    let data = adapter.to_physical_stream(data);

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

    let writer_generator = WriterGenerator::new(
        object_store,
        base_dir,
        schema,
        storage_version,
        target_bases_info,
    );
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

pub async fn validate_and_resolve_target_bases(
    params: &mut WriteParams,
    existing_base_paths: Option<&HashMap<u32, BasePath>>,
) -> Result<Option<Vec<TargetBaseInfo>>> {
    // Step 1: Validations
    if !matches!(params.mode, WriteMode::Create) && params.initial_bases.is_some() {
        return Err(Error::invalid_input(
            format!(
                "Cannot register new bases in {:?} mode. Only CREATE mode can register new bases.",
                params.mode
            ),
            location!(),
        ));
    }

    if params.target_base_names_or_paths.is_some() && params.target_bases.is_some() {
        return Err(Error::invalid_input(
            "Cannot specify both target_base_names_or_paths and target_bases. Use one or the other.",
            location!(),
        ));
    }

    // Step 2: Assign IDs to initial_bases and add them to all_bases
    let mut all_bases: HashMap<u32, BasePath> = existing_base_paths.cloned().unwrap_or_default();
    if let Some(initial_bases) = &mut params.initial_bases {
        let mut next_id = all_bases.keys().max().map(|&id| id + 1).unwrap_or(1);

        for base_path in initial_bases.iter_mut() {
            if base_path.id == 0 {
                base_path.id = next_id;
                next_id += 1;
            }
            all_bases.insert(base_path.id, base_path.clone());
        }
    }

    // Step 3: Resolve target_base_names_or_paths to IDs
    let target_base_ids = if let Some(ref names_or_paths) = params.target_base_names_or_paths {
        let mut resolved_ids = Vec::new();
        for reference in names_or_paths {
            let ref_str = reference.as_str();
            let id = all_bases
                .iter()
                .find(|(_, base)| {
                    base.name.as_ref().map(|n| n == ref_str).unwrap_or(false)
                        || base.path == ref_str
                })
                .map(|(&id, _)| id)
                .ok_or_else(|| {
                    Error::invalid_input(
                        format!("Base reference '{}' not found in available bases", ref_str),
                        location!(),
                    )
                })?;

            resolved_ids.push(id);
        }
        Some(resolved_ids)
    } else {
        params.target_bases.clone()
    };

    // Step 4: Prepare TargetBaseInfo structs
    let store_registry = params
        .session
        .as_ref()
        .map(|s| s.store_registry())
        .unwrap_or_default();

    if let Some(target_bases) = &target_base_ids {
        let store_params = params.store_params.clone().unwrap_or_default();
        let mut bases_info = Vec::new();

        for &target_base_id in target_bases {
            let base_path = all_bases.get(&target_base_id).ok_or_else(|| {
                Error::invalid_input(
                    format!(
                        "Target base ID {} not found in available bases",
                        target_base_id
                    ),
                    location!(),
                )
            })?;

            let (target_object_store, extracted_path) = ObjectStore::from_uri_and_params(
                store_registry.clone(),
                &base_path.path,
                &store_params,
            )
            .await?;

            bases_info.push(TargetBaseInfo {
                base_id: target_base_id,
                object_store: target_object_store,
                base_dir: extracted_path,
                is_dataset_root: base_path.is_dataset_root,
            });
        }

        Ok(Some(bases_info))
    } else {
        Ok(None)
    }
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
    target_bases_info: Option<Vec<TargetBaseInfo>>,
) -> Result<(Vec<Fragment>, Schema)> {
    let adapter = SchemaAdapter::new(data.schema());

    let (data, converted_schema) = if adapter.requires_physical_conversion() {
        let data = adapter.to_physical_stream(data);
        // Update the schema to match the converted data
        let arrow_schema = data.schema();
        let converted_schema = Schema::try_from(arrow_schema.as_ref())?;
        (data, converted_schema)
    } else {
        // No conversion needed, use original schema to preserve dictionary info
        (data, schema)
    };

    // Make sure the max rows per group is not larger than the max rows per file
    params.max_rows_per_group = std::cmp::min(params.max_rows_per_group, params.max_rows_per_file);

    let allow_blob_version_change =
        dataset.is_none() || matches!(params.mode, WriteMode::Overwrite);
    let (mut schema, storage_version) = if let Some(dataset) = dataset {
        match params.mode {
            WriteMode::Append | WriteMode::Create => {
                // Append mode, so we need to check compatibility
                converted_schema.check_compatible(
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
                let write_schema = dataset.schema().project_by_schema(
                    &converted_schema,
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
                (converted_schema, data_storage_version)
            }
        }
    } else {
        // Brand new dataset, use the schema from the data and the storage version
        // from the user or the default.
        (converted_schema, params.storage_version_or_default())
    };

    let target_blob_version = blob_version_for(storage_version);
    schema.apply_blob_version(target_blob_version, allow_blob_version_change)?;

    let fragments = do_write_fragments(
        object_store,
        base_dir,
        &schema,
        data,
        params,
        storage_version,
        target_bases_info,
    )
    .await?;

    Ok((fragments, schema))
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

struct V1WriterAdapter<M>
where
    M: ManifestProvider + Send + Sync,
{
    writer: FileWriter<M>,
    path: String,
    base_id: Option<u32>,
}

#[async_trait::async_trait]
impl<M> GenericWriter for V1WriterAdapter<M>
where
    M: ManifestProvider + Send + Sync,
{
    async fn write(&mut self, batches: &[RecordBatch]) -> Result<()> {
        self.writer.write(batches).await
    }
    async fn tell(&mut self) -> Result<u64> {
        Ok(self.writer.tell().await? as u64)
    }
    async fn finish(&mut self) -> Result<(u32, DataFile)> {
        let size_bytes = self.writer.tell().await?;
        Ok((
            self.writer.finish().await? as u32,
            DataFile::new_legacy(
                self.path.clone(),
                self.writer.schema(),
                NonZero::new(size_bytes as u64),
                self.base_id,
            ),
        ))
    }
}

struct V2WriterAdapter {
    writer: v2::writer::FileWriter,
    path: String,
    base_id: Option<u32>,
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
            self.base_id,
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
    open_writer_with_options(object_store, schema, base_dir, storage_version, true, None).await
}

pub async fn open_writer_with_options(
    object_store: &ObjectStore,
    schema: &Schema,
    base_dir: &Path,
    storage_version: LanceFileVersion,
    add_data_dir: bool,
    base_id: Option<u32>,
) -> Result<Box<dyn GenericWriter>> {
    let filename = format!("{}.lance", generate_random_filename());

    let full_path = if add_data_dir {
        base_dir.child(DATA_DIR).child(filename.as_str())
    } else {
        base_dir.child(filename.as_str())
    };

    let writer = if storage_version == LanceFileVersion::Legacy {
        Box::new(V1WriterAdapter {
            writer: FileWriter::<ManifestDescribing>::try_new(
                object_store,
                &full_path,
                schema.clone(),
                &Default::default(),
            )
            .await?,
            path: filename,
            base_id,
        })
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
            base_id,
        };
        Box::new(writer_adapter) as Box<dyn GenericWriter>
    };
    Ok(writer)
}

/// Information about a target base for writing.
/// Contains the base ID, object store, directory path, and whether it's a dataset root.
pub struct TargetBaseInfo {
    pub base_id: u32,
    pub object_store: Arc<ObjectStore>,
    /// The base directory path (without /data subdirectory)
    pub base_dir: Path,
    /// Whether this base path is a dataset root.
    /// If true, /data will be added when creating file paths.
    /// If false, files will be written directly to base_dir.
    pub is_dataset_root: bool,
}

struct WriterGenerator {
    /// Default object store (used when no target bases specified)
    object_store: Arc<ObjectStore>,
    /// Default base directory (used when no target bases specified)
    base_dir: Path,
    schema: Schema,
    storage_version: LanceFileVersion,
    /// Target base information (if writing to specific bases)
    target_bases_info: Option<Vec<TargetBaseInfo>>,
    /// Counter for round-robin selection
    next_base_index: AtomicUsize,
}

impl WriterGenerator {
    pub fn new(
        object_store: Arc<ObjectStore>,
        base_dir: &Path,
        schema: &Schema,
        storage_version: LanceFileVersion,
        target_bases_info: Option<Vec<TargetBaseInfo>>,
    ) -> Self {
        Self {
            object_store,
            base_dir: base_dir.clone(),
            schema: schema.clone(),
            storage_version,
            target_bases_info,
            next_base_index: AtomicUsize::new(0),
        }
    }

    /// Select the next target base using round-robin strategy.
    /// TODO: In the future, we can develop different strategies for selecting target bases
    fn select_target_base(&self) -> Option<&TargetBaseInfo> {
        self.target_bases_info.as_ref().map(|bases| {
            let index = self
                .next_base_index
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            &bases[index % bases.len()]
        })
    }

    pub async fn new_writer(&self) -> Result<(Box<dyn GenericWriter>, Fragment)> {
        // Use temporary ID 0; will assign ID later.
        let fragment = Fragment::new(0);

        let writer = if let Some(base_info) = self.select_target_base() {
            open_writer_with_options(
                base_info.object_store.as_ref(),
                &self.schema,
                &base_info.base_dir,
                self.storage_version,
                base_info.is_dataset_root,
                Some(base_info.base_id),
            )
            .await?
        } else {
            open_writer(
                self.object_store.as_ref(),
                &self.schema,
                &self.base_dir,
                self.storage_version,
            )
            .await?
        };

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
                return Err(Error::InvalidInput { source: "when creating a dataset with a custom object store the commit_handler must also be specified".into(), location: Default::default() });
            }
            commit_handler_from_url(uri, store_options).await
        }
        Some(commit_handler) => {
            if uri.starts_with("s3+ddb") {
                Err(Error::InvalidInput {
                    source: "`s3+ddb://` scheme and custom commit handler are mutually exclusive"
                        .into(),
                    location: Default::default(),
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
    tmp_dir: TempDir,
}

impl SpillStreamIter {
    pub async fn try_new(
        mut source: SendableRecordBatchStream,
        memory_limit: usize,
    ) -> Result<Self> {
        let tmp_dir = tokio::task::spawn_blocking(|| {
            TempDir::try_new().map_err(|e| Error::InvalidInput {
                source: format!("Failed to create temp dir: {}", e).into(),
                location: location!(),
            })
        })
        .await
        .ok()
        .expect_ok()??;

        let tmp_path = tmp_dir.std_path().join("spill.arrows");
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

    use arrow_array::{Int32Array, RecordBatchReader, StructArray};
    use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};
    use datafusion::{error::DataFusionError, physical_plan::stream::RecordBatchStreamAdapter};
    use futures::TryStreamExt;
    use lance_datagen::{array, gen_batch, BatchCount, RowCount};
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
                    None,
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

        let (fragments, _) = reader_to_frags(data_reader).await.unwrap();

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
            let (fragments, _) = write_fragments_internal(
                None,
                object_store,
                &Path::from("test"),
                schema,
                data_stream,
                write_params,
                None,
            )
            .await
            .unwrap();

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
        let (fragments, _) = write_fragments_internal(
            None,
            object_store.clone(),
            &base_path,
            schema.clone(),
            data_stream,
            write_params,
            None,
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

    #[tokio::test]
    async fn test_explicit_data_file_bases_writer_generator() {
        use arrow::datatypes::{DataType, Field as ArrowField, Schema as ArrowSchema};
        use lance_io::object_store::ObjectStore;
        use std::sync::Arc;

        // Create test schema
        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();

        // Create in-memory object store
        let object_store = Arc::new(ObjectStore::memory());
        let base_dir = Path::from("test/bucket2");

        // Test WriterGenerator with explicit data file bases configuration
        let target_bases = vec![TargetBaseInfo {
            base_id: 2,
            object_store: object_store.clone(),
            base_dir: base_dir.clone(),
            is_dataset_root: false, // Test uses direct data directory
        }];
        let writer_generator = WriterGenerator::new(
            object_store.clone(),
            &base_dir,
            &schema,
            LanceFileVersion::Stable,
            Some(target_bases),
        );

        // Create a writer
        let (writer, fragment) = writer_generator.new_writer().await.unwrap();

        // Verify fragment is created
        assert_eq!(fragment.id, 0); // Temporary ID

        // Verify writer is created (we can't test much more without writing data)
        drop(writer); // Clean up
    }

    #[tokio::test]
    async fn test_writer_with_base_id() {
        use arrow::array::Int32Array;
        use arrow::datatypes::{DataType, Field as ArrowField, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        use lance_io::object_store::ObjectStore;
        use std::sync::Arc;

        // Create test data
        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();

        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();

        // Create in-memory object store and writer
        let object_store = Arc::new(ObjectStore::memory());
        let base_dir = Path::from("test/bucket2");

        let mut inner_writer = open_writer_with_options(
            object_store.as_ref(),
            &schema,
            &base_dir,
            LanceFileVersion::Stable,
            false, // Don't add /data
            None,
        )
        .await
        .unwrap();

        // Write data
        inner_writer.write(&[batch]).await.unwrap();

        // Finish and manually set base_id
        let base_id = 2u32;
        let (_num_rows, mut data_file) = inner_writer.finish().await.unwrap();
        data_file.base_id = Some(base_id);

        assert_eq!(data_file.base_id, Some(base_id));
        assert!(!data_file.path.is_empty());
    }

    #[tokio::test]
    async fn test_round_robin_target_base_selection() {
        use arrow::array::Int32Array;
        use arrow::datatypes::{DataType, Field as ArrowField, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        use lance_io::object_store::ObjectStore;
        use std::sync::Arc;

        // Create test schema
        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();

        // Create in-memory object stores for different bases
        let store1 = Arc::new(ObjectStore::memory());
        let store2 = Arc::new(ObjectStore::memory());
        let store3 = Arc::new(ObjectStore::memory());

        // Create WriterGenerator with multiple target bases
        let target_bases = vec![
            TargetBaseInfo {
                base_id: 1,
                object_store: store1.clone(),
                base_dir: Path::from("base1"),
                is_dataset_root: false,
            },
            TargetBaseInfo {
                base_id: 2,
                object_store: store2.clone(),
                base_dir: Path::from("base2"),
                is_dataset_root: false,
            },
            TargetBaseInfo {
                base_id: 3,
                object_store: store3.clone(),
                base_dir: Path::from("base3"),
                is_dataset_root: false,
            },
        ];

        let writer_generator = WriterGenerator::new(
            Arc::new(ObjectStore::memory()),
            &Path::from("default"),
            &schema,
            LanceFileVersion::Stable,
            Some(target_bases),
        );

        // Create test batch
        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();

        // Create multiple writers and verify round-robin selection
        let mut base_ids = Vec::new();
        for _ in 0..6 {
            let (mut writer, _fragment) = writer_generator.new_writer().await.unwrap();
            writer.write(std::slice::from_ref(&batch)).await.unwrap();
            let (_num_rows, data_file) = writer.finish().await.unwrap();
            base_ids.push(data_file.base_id.unwrap());
        }

        // Verify round-robin pattern: 1, 2, 3, 1, 2, 3
        assert_eq!(base_ids, vec![1, 2, 3, 1, 2, 3]);
    }

    #[tokio::test]
    async fn test_explicit_data_file_bases_path_parsing() {
        // Test URI parsing logic
        let test_cases = vec![
            ("s3://multi-path-test/test1/subBucket2", "test1/subBucket2"),
            ("gs://my-bucket/path/to/data", "path/to/data"),
            ("file:///tmp/test/bucket", "tmp/test/bucket"),
        ];

        for (uri, expected_path) in test_cases {
            let url = url::Url::parse(uri).unwrap();
            let parsed_path = url.path().trim_start_matches('/');
            assert_eq!(parsed_path, expected_path, "Failed for URI: {}", uri);
        }
    }

    #[tokio::test]
    async fn test_write_params_validation() {
        // Test CREATE mode validation
        let mut params = WriteParams {
            mode: WriteMode::Create,
            initial_bases: Some(vec![
                BasePath {
                    id: 1,
                    name: Some("bucket1".to_string()),
                    path: "s3://bucket1/path1".to_string(),
                    is_dataset_root: true,
                },
                BasePath {
                    id: 2,
                    name: Some("bucket2".to_string()),
                    path: "s3://bucket2/path2".to_string(),
                    is_dataset_root: true,
                },
            ]),
            target_bases: Some(vec![1]), // Use ID 1 which corresponds to bucket1
            ..Default::default()
        };

        // This should be valid
        let result = validate_write_params(&params);
        assert!(result.is_ok());

        // Test target_bases with ID not in initial_bases (should fail)
        params.target_bases = Some(vec![99]); // ID 99 doesn't exist
        let result = validate_write_params(&params);
        assert!(result.is_err());

        // Test CREATE mode with target_bases but no initial_bases (should fail)
        params.initial_bases = None;
        params.target_bases = Some(vec![1]);
        let result = validate_write_params(&params);
        assert!(result.is_err());
    }

    fn validate_write_params(params: &WriteParams) -> Result<()> {
        // Replicate the validation logic from the main write function
        if matches!(params.mode, WriteMode::Create) {
            if let Some(target_bases) = &params.target_bases {
                if target_bases.len() != 1 {
                    return Err(Error::invalid_input(
                        format!(
                            "target_bases with {} elements is not supported",
                            target_bases.len()
                        ),
                        Default::default(),
                    ));
                }
                let target_base_id = target_bases[0];
                if let Some(initial_bases) = &params.initial_bases {
                    if !initial_bases.iter().any(|bp| bp.id == target_base_id) {
                        return Err(Error::invalid_input(
                            format!(
                                "target_base_id {} must be one of the initial_bases in CREATE mode",
                                target_base_id
                            ),
                            Default::default(),
                        ));
                    }
                } else {
                    return Err(Error::invalid_input(
                        "initial_bases must be provided when target_bases is specified in CREATE mode",
                        Default::default(),
                    ));
                }
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_multi_base_create() {
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

        // Create dataset with multi-base configuration
        let test_uri = "memory://multi_base_test";
        let primary_uri = format!("{}/primary", test_uri);
        let base1_uri = format!("{}/base1", test_uri);
        let base2_uri = format!("{}/base2", test_uri);

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = crate::dataset::Dataset::write(
            data_gen.batch(5),
            &primary_uri,
            Some(WriteParams {
                mode: WriteMode::Create,
                initial_bases: Some(vec![
                    BasePath {
                        id: 1,
                        name: Some("base1".to_string()),
                        path: base1_uri.clone(),
                        is_dataset_root: true,
                    },
                    BasePath {
                        id: 2,
                        name: Some("base2".to_string()),
                        path: base2_uri.clone(),
                        is_dataset_root: true,
                    },
                ]),
                target_bases: Some(vec![1]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Verify dataset was created
        assert_eq!(dataset.count_rows(None).await.unwrap(), 5);

        // Verify base_paths are registered in manifest
        assert_eq!(dataset.manifest.base_paths.len(), 2);
        assert!(dataset
            .manifest
            .base_paths
            .values()
            .any(|bp| bp.name == Some("base1".to_string())));
        assert!(dataset
            .manifest
            .base_paths
            .values()
            .any(|bp| bp.name == Some("base2".to_string())));

        // Verify data was written to base1
        let fragments = dataset.get_fragments();
        assert!(!fragments.is_empty());
        for fragment in fragments {
            assert!(fragment
                .metadata
                .files
                .iter()
                .any(|file| file.base_id == Some(1)));
        }

        // Test validation: cannot specify both target_bases and target_base_names_or_paths
        let mut data_gen2 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let result = Dataset::write(
            data_gen2.batch(5),
            &format!("{}/test_validation", test_uri),
            Some(WriteParams {
                mode: WriteMode::Create,
                initial_bases: Some(vec![BasePath {
                    id: 1,
                    name: Some("base1".to_string()),
                    path: base1_uri.clone(),
                    is_dataset_root: true,
                }]),
                target_bases: Some(vec![1]),
                target_base_names_or_paths: Some(vec!["base1".to_string()]),
                ..Default::default()
            }),
        )
        .await;

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Cannot specify both target_base_names_or_paths and target_bases"));
    }

    #[tokio::test]
    async fn test_multi_base_overwrite() {
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

        // Create initial dataset
        let test_uri = "memory://overwrite_test";
        let primary_uri = format!("{}/primary", test_uri);
        let base1_uri = format!("{}/base1", test_uri);
        let base2_uri = format!("{}/base2", test_uri);
        let _base3_uri = format!("{}/base3", test_uri);

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = Dataset::write(
            data_gen.batch(3),
            &primary_uri,
            Some(WriteParams {
                mode: WriteMode::Create,
                initial_bases: Some(vec![
                    BasePath {
                        id: 1,
                        name: Some("base1".to_string()),
                        path: base1_uri.clone(),
                        is_dataset_root: true,
                    },
                    BasePath {
                        id: 2,
                        name: Some("base2".to_string()),
                        path: base2_uri.clone(),
                        is_dataset_root: true,
                    },
                ]),
                target_bases: Some(vec![1]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 3);

        // Overwrite - should inherit existing base configuration (base1, base2)
        // Write to base2
        let mut data_gen2 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = Dataset::write(
            data_gen2.batch(2),
            std::sync::Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                // No initial_bases - inherits existing base_paths
                target_bases: Some(vec![2]), // Write to base2
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Verify data was overwritten
        assert_eq!(dataset.count_rows(None).await.unwrap(), 2);

        // Verify base_paths were inherited (still base1 and base2)
        assert_eq!(dataset.manifest.base_paths.len(), 2);
        assert!(dataset
            .manifest
            .base_paths
            .values()
            .any(|bp| bp.name == Some("base1".to_string())));
        assert!(dataset
            .manifest
            .base_paths
            .values()
            .any(|bp| bp.name == Some("base2".to_string())));

        // Verify data was written to base2 (ID 2)
        let fragments = dataset.get_fragments();
        assert!(fragments.iter().all(|f| f
            .metadata
            .files
            .iter()
            .all(|file| file.base_id == Some(2))));

        // Test validation: cannot specify initial_bases in OVERWRITE mode
        let mut data_gen3 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let result = Dataset::write(
            data_gen3.batch(2),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                initial_bases: Some(vec![BasePath {
                    id: 3,
                    name: Some("base3".to_string()),
                    path: _base3_uri.clone(),
                    is_dataset_root: true,
                }]),
                target_bases: Some(vec![1]),
                ..Default::default()
            }),
        )
        .await;

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Cannot register new bases in Overwrite mode"));
    }

    #[tokio::test]
    async fn test_multi_base_append() {
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

        // Create initial dataset with multi-base configuration
        let test_uri = "memory://append_test";
        let primary_uri = format!("{}/primary", test_uri);
        let base1_uri = format!("{}/base1", test_uri);
        let base2_uri = format!("{}/base2", test_uri);

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = Dataset::write(
            data_gen.batch(3),
            &primary_uri,
            Some(WriteParams {
                mode: WriteMode::Create,
                initial_bases: Some(vec![
                    BasePath {
                        id: 1,
                        name: Some("base1".to_string()),
                        path: base1_uri.clone(),
                        is_dataset_root: true,
                    },
                    BasePath {
                        id: 2,
                        name: Some("base2".to_string()),
                        path: base2_uri.clone(),
                        is_dataset_root: true,
                    },
                ]),
                target_bases: Some(vec![1]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 3);

        // Append to base1 (same base as initial write)
        let mut data_gen2 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = Dataset::write(
            data_gen2.batch(2),
            std::sync::Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                target_bases: Some(vec![1]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 5);

        // Verify base_paths are still registered
        assert_eq!(dataset.manifest.base_paths.len(), 2);

        // Append to base2 (different base)
        let mut data_gen3 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = Dataset::write(
            data_gen3.batch(4),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                target_bases: Some(vec![2]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 9);

        // Verify data is distributed across both bases
        let fragments = dataset.get_fragments();
        let has_base1_data = fragments
            .iter()
            .any(|f| f.metadata.files.iter().any(|file| file.base_id == Some(1)));
        let has_base2_data = fragments
            .iter()
            .any(|f| f.metadata.files.iter().any(|file| file.base_id == Some(2)));

        assert!(has_base1_data, "Should have data in base1");
        assert!(has_base2_data, "Should have data in base2");

        // Test validation: cannot specify initial_bases in APPEND mode
        let mut data_gen4 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));
        let base3_uri = format!("{}/base3", test_uri);

        let result = Dataset::write(
            data_gen4.batch(2),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                initial_bases: Some(vec![BasePath {
                    id: 3,
                    name: Some("base3".to_string()),
                    path: base3_uri,
                    is_dataset_root: true,
                }]),
                target_bases: Some(vec![1]),
                ..Default::default()
            }),
        )
        .await;

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Cannot register new bases in Append mode"));
    }

    #[tokio::test]
    async fn test_multi_base_is_dataset_root_flag() {
        use lance_core::utils::tempfile::TempDir;
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

        // Create dataset with different is_dataset_root settings using tempdir
        let test_dir = TempDir::default();
        let primary_uri = test_dir.path_str();
        let base1_dir = test_dir.std_path().join("base1");
        let base2_dir = test_dir.std_path().join("base2");

        std::fs::create_dir_all(&base1_dir).unwrap();
        std::fs::create_dir_all(&base2_dir).unwrap();

        let base1_uri = format!("file://{}", base1_dir.display());
        let base2_uri = format!("file://{}", base2_dir.display());

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = Dataset::write(
            data_gen.batch(10),
            &primary_uri,
            Some(WriteParams {
                mode: WriteMode::Create,
                max_rows_per_file: 5, // Create multiple fragments
                initial_bases: Some(vec![
                    BasePath {
                        id: 1,
                        name: Some("base1".to_string()),
                        path: base1_uri.clone(),
                        is_dataset_root: true, // Files will go to base1/data/
                    },
                    BasePath {
                        id: 2,
                        name: Some("base2".to_string()),
                        path: base2_uri.clone(),
                        is_dataset_root: false, // Files will go directly to base2/
                    },
                ]),
                target_bases: Some(vec![1, 2]), // Write to both bases
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 10);

        // Verify base_paths configuration
        assert_eq!(dataset.manifest.base_paths.len(), 2);

        let base1 = dataset
            .manifest
            .base_paths
            .values()
            .find(|bp| bp.name == Some("base1".to_string()))
            .expect("base1 not found");
        let base2 = dataset
            .manifest
            .base_paths
            .values()
            .find(|bp| bp.name == Some("base2".to_string()))
            .expect("base2 not found");

        // Verify is_dataset_root flags are persisted correctly in manifest
        assert!(
            base1.is_dataset_root,
            "base1 should have is_dataset_root=true"
        );
        assert!(
            !base2.is_dataset_root,
            "base2 should have is_dataset_root=false"
        );

        // Verify data was written to both bases
        let fragments = dataset.get_fragments();
        assert!(!fragments.is_empty());

        let has_base1_data = fragments
            .iter()
            .any(|f| f.metadata.files.iter().any(|file| file.base_id == Some(1)));
        let has_base2_data = fragments
            .iter()
            .any(|f| f.metadata.files.iter().any(|file| file.base_id == Some(2)));

        assert!(has_base1_data, "Should have data in base1");
        assert!(has_base2_data, "Should have data in base2");

        // Verify actual file paths on disk
        // For base1 (is_dataset_root=true), files should be in base1/data/
        let base1_data_dir = base1_dir.join("data");
        assert!(base1_data_dir.exists(), "base1/data directory should exist");
        let base1_files: Vec<_> = std::fs::read_dir(&base1_data_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "lance")
                    .unwrap_or(false)
            })
            .collect();
        assert!(
            !base1_files.is_empty(),
            "base1/data should contain .lance files"
        );

        // For base2 (is_dataset_root=false), files should be directly in base2/
        let base2_files: Vec<_> = std::fs::read_dir(&base2_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "lance")
                    .unwrap_or(false)
            })
            .collect();
        assert!(
            !base2_files.is_empty(),
            "base2 should contain .lance files directly"
        );

        // Verify base2 does NOT have a data subdirectory with lance files
        let base2_data_dir = base2_dir.join("data");
        if base2_data_dir.exists() {
            let base2_data_files: Vec<_> = std::fs::read_dir(&base2_data_dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "lance")
                        .unwrap_or(false)
                })
                .collect();
            assert!(
                base2_data_files.is_empty(),
                "base2/data should NOT contain .lance files"
            );
        }
    }

    #[tokio::test]
    async fn test_multi_base_target_by_path_uri() {
        use lance_core::utils::tempfile::TempDir;
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

        // Create dataset with named bases
        let test_dir = TempDir::default();
        let primary_uri = test_dir.path_str();
        let base1_dir = test_dir.std_path().join("base1");
        let base2_dir = test_dir.std_path().join("base2");

        std::fs::create_dir_all(&base1_dir).unwrap();
        std::fs::create_dir_all(&base2_dir).unwrap();

        let base1_uri = format!("file://{}", base1_dir.display());
        let base2_uri = format!("file://{}", base2_dir.display());

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        // Create initial dataset writing to base1 using name
        let dataset = Dataset::write(
            data_gen.batch(10),
            &primary_uri,
            Some(WriteParams {
                mode: WriteMode::Create,
                max_rows_per_file: 5,
                initial_bases: Some(vec![
                    BasePath {
                        id: 1,
                        name: Some("base1".to_string()),
                        path: base1_uri.clone(),
                        is_dataset_root: true,
                    },
                    BasePath {
                        id: 2,
                        name: Some("base2".to_string()),
                        path: base2_uri.clone(),
                        is_dataset_root: true,
                    },
                ]),
                target_base_names_or_paths: Some(vec!["base1".to_string()]), // Use name
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 10);

        // Verify data was written to base1
        let fragments = dataset.get_fragments();
        assert!(fragments.iter().all(|f| f
            .metadata
            .files
            .iter()
            .all(|file| file.base_id == Some(1))));

        // Now append using the path URI instead of name
        let mut data_gen2 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = Dataset::write(
            data_gen2.batch(5),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                // Use the actual path URI instead of the name
                target_base_names_or_paths: Some(vec![base2_uri.clone()]),
                max_rows_per_file: 5,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 15);

        // Verify data is now in both bases
        let fragments = dataset.get_fragments();
        let has_base1_data = fragments
            .iter()
            .any(|f| f.metadata.files.iter().any(|file| file.base_id == Some(1)));
        let has_base2_data = fragments
            .iter()
            .any(|f| f.metadata.files.iter().any(|file| file.base_id == Some(2)));

        assert!(has_base1_data, "Should have data in base1");
        assert!(has_base2_data, "Should have data in base2");

        // Verify base2 has exactly 1 fragment (from the append)
        let base2_fragments: Vec<_> = fragments
            .iter()
            .filter(|f| f.metadata.files.iter().all(|file| file.base_id == Some(2)))
            .collect();
        assert_eq!(base2_fragments.len(), 1, "Should have 1 fragment in base2");
    }
}
