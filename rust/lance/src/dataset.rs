// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Dataset
//!

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::DataType;
use byteorder::{ByteOrder, LittleEndian};
use chrono::{prelude::*, Duration};
use deepsize::DeepSizeOf;
use futures::future::BoxFuture;
use futures::stream::{self, BoxStream, StreamExt, TryStreamExt};
use futures::{FutureExt, Stream};

use crate::dataset::metadata::UpdateFieldMetadataBuilder;
use crate::dataset::transaction::translate_schema_metadata_updates;
use crate::session::caches::{DSMetadataCache, ManifestKey, TransactionKey};
use crate::session::index_caches::DSIndexCache;
use itertools::Itertools;
use lance_core::datatypes::{Field, OnMissing, OnTypeMismatch, Projectable, Projection};
use lance_core::traits::DatasetTakeRows;
use lance_core::utils::address::RowAddress;
use lance_core::utils::tracing::{
    AUDIT_MODE_CREATE, AUDIT_TYPE_MANIFEST, DATASET_CLEANING_EVENT, DATASET_DELETING_EVENT,
    DATASET_DROPPING_COLUMN_EVENT, TRACE_DATASET_EVENTS, TRACE_FILE_AUDIT,
};
use lance_core::{ROW_ADDR, ROW_ADDR_FIELD, ROW_ID_FIELD};
use lance_datafusion::projection::ProjectionPlan;
use lance_file::datatypes::populate_schema_dictionary;
use lance_file::v2::reader::FileReaderOptions;
use lance_file::version::LanceFileVersion;
use lance_index::DatasetIndexExt;
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use lance_io::object_writer::{ObjectWriter, WriteResult};
use lance_io::traits::WriteExt;
use lance_io::utils::{read_last_block, read_metadata_offset, read_struct};
use lance_table::format::{
    DataFile, DataStorageFormat, DeletionFile, Fragment, IndexMetadata, Manifest, MAGIC,
    MAJOR_VERSION, MINOR_VERSION,
};
use lance_table::io::commit::{
    migrate_scheme_to_v2, CommitConfig, CommitError, CommitHandler, CommitLock, ManifestLocation,
    ManifestNamingScheme,
};
use lance_table::io::manifest::{read_manifest, write_manifest};
use object_store::path::Path;
use prost::Message;
use roaring::RoaringBitmap;
use rowids::get_row_id_index;
use serde::{Deserialize, Serialize};
use snafu::location;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Debug;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use take::row_offsets_to_row_addresses;
use tracing::{info, instrument};

mod blob;
mod branch_location;
pub mod builder;
pub mod cleanup;
pub mod delta;
pub mod fragment;
mod hash_joiner;
pub mod index;
mod metadata;
pub mod optimize;
pub mod progress;
pub mod refs;
pub(crate) mod rowids;
pub mod scanner;
mod schema_evolution;
pub mod sql;
pub mod statistics;
mod take;
pub mod transaction;
pub mod udtf;
pub mod updater;
mod utils;
mod write;

use self::builder::DatasetBuilder;
use self::cleanup::RemovalStats;
use self::fragment::FileFragment;
use self::refs::Refs;
use self::scanner::{DatasetRecordBatchStream, Scanner};
use self::transaction::{Operation, Transaction, TransactionBuilder, UpdateMapEntry};
use self::write::write_fragments_internal;
use crate::dataset::branch_location::BranchLocation;
use crate::dataset::cleanup::{CleanupPolicy, CleanupPolicyBuilder};
use crate::dataset::refs::{BranchContents, Branches, Tags};
use crate::dataset::sql::SqlQueryBuilder;
use crate::datatypes::Schema;
use crate::index::retain_supported_indices;
use crate::io::commit::{
    commit_detached_transaction, commit_new_dataset, commit_transaction,
    detect_overlapping_fragments, read_transaction_file,
};
use crate::session::Session;
use crate::utils::temporal::{timestamp_to_nanos, utc_now, SystemTime};
use crate::{Error, Result};
pub use blob::BlobFile;
use hash_joiner::HashJoiner;
use lance_core::box_error;
pub use lance_core::ROW_ID;
use lance_table::feature_flags::{apply_feature_flags, can_read_dataset};
pub use schema_evolution::{
    BatchInfo, BatchUDF, ColumnAlteration, NewColumnTransform, UDFCheckpointStore,
};
pub use take::TakeBuilder;
pub use write::merge_insert::{
    MergeInsertBuilder, MergeInsertJob, MergeStats, UncommittedMergeInsert, WhenMatched,
    WhenNotMatched, WhenNotMatchedBySource,
};

pub use write::update::{UpdateBuilder, UpdateJob};
#[allow(deprecated)]
pub use write::{
    write_fragments, AutoCleanupParams, CommitBuilder, DeleteBuilder, InsertBuilder,
    WriteDestination, WriteMode, WriteParams,
};

const INDICES_DIR: &str = "_indices";

pub const DATA_DIR: &str = "data";
// We default to 6GB for the index cache, since indices are often large but
// worth caching.
pub const DEFAULT_INDEX_CACHE_SIZE: usize = 6 * 1024 * 1024 * 1024;
// Default to 1 GiB for the metadata cache. Column metadata can be like 40MB,
// so this should be enough for a few hundred columns. Other metadata is much
// smaller.
pub const DEFAULT_METADATA_CACHE_SIZE: usize = 1024 * 1024 * 1024;

/// Lance Dataset
#[derive(Clone)]
pub struct Dataset {
    pub object_store: Arc<ObjectStore>,
    pub(crate) commit_handler: Arc<dyn CommitHandler>,
    /// Uri of the dataset.
    ///
    /// On cloud storage, we can not use [Dataset::base] to build the full uri because the
    /// `bucket` is swallowed in the inner [ObjectStore].
    uri: String,
    pub(crate) base: Path,
    pub manifest: Arc<Manifest>,
    // Path for the manifest that is loaded. Used to get additional information,
    // such as the index metadata.
    pub(crate) manifest_location: ManifestLocation,
    pub(crate) session: Arc<Session>,
    pub refs: Refs,

    // Bitmap of fragment ids in this dataset.
    pub(crate) fragment_bitmap: Arc<RoaringBitmap>,

    // These are references to session caches, but with the dataset URI as a prefix.
    pub(crate) index_cache: Arc<DSIndexCache>,
    pub(crate) metadata_cache: Arc<DSMetadataCache>,

    /// File reader options to use when reading data files.
    pub(crate) file_reader_options: Option<FileReaderOptions>,

    /// Object store parameters used when opening this dataset.
    /// These are used when creating object stores for additional base paths.
    pub(crate) store_params: Option<Box<ObjectStoreParams>>,
}

impl std::fmt::Debug for Dataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dataset")
            .field("uri", &self.uri)
            .field("base", &self.base)
            .field("version", &self.manifest.version)
            .field("cache_num_items", &self.session.approx_num_items())
            .finish()
    }
}

/// Dataset Version
#[derive(Deserialize, Serialize)]
pub struct Version {
    /// version number
    pub version: u64,

    /// Timestamp of dataset creation in UTC.
    pub timestamp: DateTime<Utc>,

    /// Key-value pairs of metadata.
    pub metadata: BTreeMap<String, String>,
}

/// Convert Manifest to Data Version.
impl From<&Manifest> for Version {
    fn from(m: &Manifest) -> Self {
        Self {
            version: m.version,
            timestamp: m.timestamp(),
            metadata: m.summary().into(),
        }
    }
}

/// Customize read behavior of a dataset.
#[derive(Clone, Debug)]
pub struct ReadParams {
    /// Size of the index cache in bytes. This cache stores index data in memory
    /// for faster lookups. The default is 6 GiB.
    pub index_cache_size_bytes: usize,

    /// Size of the metadata cache in bytes. This cache stores metadata in memory
    /// for faster open table and scans. The default is 1 GiB.
    pub metadata_cache_size_bytes: usize,

    /// If present, dataset will use this shared [`Session`] instead creating a new one.
    ///
    /// This is useful for sharing the same session across multiple datasets.
    pub session: Option<Arc<Session>>,

    pub store_options: Option<ObjectStoreParams>,

    /// If present, dataset will use this to resolve the latest version
    ///
    /// Lance needs to be able to make atomic updates to the manifest.  This involves
    /// coordination between readers and writers and we can usually rely on the filesystem
    /// to do this coordination for us.
    ///
    /// Some file systems (e.g. S3) do not support atomic operations.  In this case, for
    /// safety, we recommend an external commit mechanism (such as dynamodb) and, on the
    /// read path, we need to reach out to that external mechanism to figure out the latest
    /// version of the dataset.
    ///
    /// If this is not set then a default behavior is chosen that is appropriate for the
    /// filesystem.
    ///
    /// If a custom object store is provided (via store_params.object_store) then this
    /// must also be provided.
    pub commit_handler: Option<Arc<dyn CommitHandler>>,

    /// File reader options to use when reading data files.
    ///
    /// This allows control over features like caching repetition indices and validation.
    pub file_reader_options: Option<FileReaderOptions>,
}

impl ReadParams {
    /// Set the cache size for indices. Set to zero, to disable the cache.
    #[deprecated(
        since = "0.30.0",
        note = "Use `index_cache_size_bytes` instead, which accepts a size in bytes."
    )]
    pub fn index_cache_size(&mut self, cache_size: usize) -> &mut Self {
        let assumed_entry_size = 20 * 1024 * 1024; // 20 MiB per entry
        self.index_cache_size_bytes = cache_size * assumed_entry_size;
        self
    }

    pub fn index_cache_size_bytes(&mut self, cache_size: usize) -> &mut Self {
        self.index_cache_size_bytes = cache_size;
        self
    }

    /// Set the cache size for the file metadata. Set to zero to disable this cache.
    #[deprecated(
        since = "0.30.0",
        note = "Use `metadata_cache_size_bytes` instead, which accepts a size in bytes."
    )]
    pub fn metadata_cache_size(&mut self, cache_size: usize) -> &mut Self {
        let assumed_entry_size = 10 * 1024 * 1024; // 10 MiB per entry
        self.metadata_cache_size_bytes = cache_size * assumed_entry_size;
        self
    }

    /// Set the cache size for the file metadata in bytes.
    pub fn metadata_cache_size_bytes(&mut self, cache_size: usize) -> &mut Self {
        self.metadata_cache_size_bytes = cache_size;
        self
    }

    /// Set a shared session for the datasets.
    pub fn session(&mut self, session: Arc<Session>) -> &mut Self {
        self.session = Some(session);
        self
    }

    /// Use the explicit locking to resolve the latest version
    pub fn set_commit_lock<T: CommitLock + Send + Sync + 'static>(&mut self, lock: Arc<T>) {
        self.commit_handler = Some(Arc::new(lock));
    }

    /// Set the file reader options.
    pub fn file_reader_options(&mut self, options: FileReaderOptions) -> &mut Self {
        self.file_reader_options = Some(options);
        self
    }
}

impl Default for ReadParams {
    fn default() -> Self {
        Self {
            index_cache_size_bytes: DEFAULT_INDEX_CACHE_SIZE,
            metadata_cache_size_bytes: DEFAULT_METADATA_CACHE_SIZE,
            session: None,
            store_options: None,
            commit_handler: None,
            file_reader_options: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ProjectionRequest {
    Schema(Arc<Schema>),
    Sql(Vec<(String, String)>),
}

impl ProjectionRequest {
    pub fn from_columns(
        columns: impl IntoIterator<Item = impl AsRef<str>>,
        dataset_schema: &Schema,
    ) -> Self {
        let columns = columns
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect::<Vec<_>>();

        // Separate data columns from system columns
        // System columns need to be added to the schema manually since Schema::project
        // doesn't include them (they're virtual columns)
        let mut data_columns = Vec::new();
        let mut system_fields = Vec::new();

        for col in &columns {
            if lance_core::is_system_column(col) {
                // For now we only support _rowid and _rowaddr in projections
                if col == ROW_ID {
                    system_fields.push(Field::try_from(ROW_ID_FIELD.clone()).unwrap());
                } else if col == ROW_ADDR {
                    system_fields.push(Field::try_from(ROW_ADDR_FIELD.clone()).unwrap());
                }
                // Note: Other system columns like _rowoffset are handled differently
            } else {
                data_columns.push(col.as_str());
            }
        }

        // Project only the data columns
        let mut schema = dataset_schema.project(&data_columns).unwrap();

        // Add system fields in the order they appeared in the original columns list
        // We need to reconstruct the proper order
        let mut final_fields = Vec::new();
        for col in &columns {
            if lance_core::is_system_column(col) {
                // Find and add the system field
                if let Some(field) = system_fields.iter().find(|f| &f.name == col) {
                    final_fields.push(field.clone());
                }
            } else {
                // Find and add the data field
                if let Some(field) = schema.fields.iter().find(|f| &f.name == col) {
                    final_fields.push(field.clone());
                }
            }
        }

        schema.fields = final_fields;
        Self::Schema(Arc::new(schema))
    }

    pub fn from_schema(schema: Schema) -> Self {
        Self::Schema(Arc::new(schema))
    }

    /// Provide a list of projection with SQL transform.
    ///
    /// # Parameters
    /// - `columns`: A list of tuples where the first element is resulted column name and the second
    ///   element is the SQL expression.
    pub fn from_sql(
        columns: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        Self::Sql(
            columns
                .into_iter()
                .map(|(a, b)| (a.into(), b.into()))
                .collect(),
        )
    }

    pub fn into_projection_plan(self, dataset: Arc<Dataset>) -> Result<ProjectionPlan> {
        match self {
            Self::Schema(schema) => {
                // The schema might contain system columns (_rowid, _rowaddr) which are not
                // in the dataset schema. We handle these specially in ProjectionPlan::from_schema.
                let system_columns_present = schema
                    .fields
                    .iter()
                    .any(|f| lance_core::is_system_column(&f.name));

                if system_columns_present {
                    // If system columns are present, we can't use project_by_schema directly
                    // Just pass the schema to ProjectionPlan::from_schema which handles it
                    ProjectionPlan::from_schema(dataset, schema.as_ref())
                } else {
                    // No system columns, use normal path with validation
                    let projection = dataset.schema().project_by_schema(
                        schema.as_ref(),
                        OnMissing::Error,
                        OnTypeMismatch::Error,
                    )?;
                    ProjectionPlan::from_schema(dataset, &projection)
                }
            }
            Self::Sql(columns) => ProjectionPlan::from_expressions(dataset, &columns),
        }
    }
}

impl From<Arc<Schema>> for ProjectionRequest {
    fn from(schema: Arc<Schema>) -> Self {
        Self::Schema(schema)
    }
}

impl From<Schema> for ProjectionRequest {
    fn from(schema: Schema) -> Self {
        Self::from(Arc::new(schema))
    }
}

impl Dataset {
    /// Open an existing dataset.
    ///
    /// See also [DatasetBuilder].
    #[instrument]
    pub async fn open(uri: &str) -> Result<Self> {
        DatasetBuilder::from_uri(uri).load().await
    }

    /// Check out a dataset version with a ref
    pub async fn checkout_version(&self, version: impl Into<refs::Ref>) -> Result<Self> {
        let ref_: refs::Ref = version.into();
        match ref_ {
            refs::Ref::Version(branch, version_number) => {
                self.checkout_by_ref(version_number, branch).await
            }
            refs::Ref::Tag(tag_name) => {
                let tag_contents = self.tags().get(tag_name.as_str()).await?;
                self.checkout_by_ref(Some(tag_contents.version), tag_contents.branch)
                    .await
            }
        }
    }

    pub fn tags(&self) -> Tags<'_> {
        self.refs.tags()
    }

    pub fn branches(&self) -> Branches<'_> {
        self.refs.branches()
    }

    /// Check out the latest version of the dataset
    pub async fn checkout_latest(&mut self) -> Result<()> {
        let (manifest, manifest_location) = self.latest_manifest().await?;
        self.manifest = manifest;
        self.manifest_location = manifest_location;
        self.fragment_bitmap = Arc::new(
            self.manifest
                .fragments
                .iter()
                .map(|f| f.id as u32)
                .collect(),
        );
        Ok(())
    }

    /// Check out the latest version of the branch
    pub async fn checkout_branch(&self, branch: &str) -> Result<Self> {
        self.checkout_by_ref(None, Some(branch.to_string())).await
    }

    /// This is a two-phase operation:
    /// - Create the branch dataset by shallow cloning.
    /// - Create the branch metadata (a.k.a. `BranchContents`).
    ///
    /// These two phases are not atomic. We consider `BranchContents` as the source of truth
    /// for the branch.
    ///
    /// The cleanup procedure should:
    /// - Clean up zombie branch datasets that have no related `BranchContents`.
    /// - Delete broken `BranchContents` entries that have no related branch dataset.
    ///
    /// If `create_branch` stops at phase 1, it may leave a zombie branch dataset,
    /// which can be cleaned up later. Such a zombie dataset may cause a branch creation
    /// failure if we use the same name to `create_branch`. In that case, you need to call
    /// `force_delete_branch` to interactively clean up the zombie dataset.
    pub async fn create_branch(
        &mut self,
        branch: &str,
        version: impl Into<refs::Ref>,
        store_params: Option<ObjectStoreParams>,
    ) -> Result<Self> {
        let (source_branch, version_number) = self.resolve_reference(version.into()).await?;
        let branch_location = self.find_branch_location(branch)?;
        let clone_op = Operation::Clone {
            is_shallow: true,
            ref_name: source_branch.clone(),
            ref_version: version_number,
            ref_path: String::from(self.uri()),
            branch_name: Some(branch.to_string()),
        };
        let transaction = Transaction::new(version_number, clone_op, None);

        let builder = CommitBuilder::new(WriteDestination::Uri(branch_location.uri.as_str()))
            .with_store_params(store_params.unwrap_or_default())
            .with_object_store(Arc::new(self.object_store().clone()))
            .with_commit_handler(self.commit_handler.clone())
            .with_storage_format(self.manifest.data_storage_format.lance_file_version()?);
        let dataset = builder.execute(transaction).await?;

        // Create BranchContents after shallow_clone
        self.branches()
            .create(branch, version_number, source_branch.as_deref())
            .await?;
        Ok(dataset)
    }

    pub async fn delete_branch(&mut self, branch: &str) -> Result<()> {
        self.branches().delete(branch, false).await
    }

    /// Delete the branch even if the BranchContents is not found.
    /// This could be useful when we have zombie branches and want to clean them up immediately.
    pub async fn force_delete_branch(&mut self, branch: &str) -> Result<()> {
        self.branches().delete(branch, true).await
    }

    pub async fn list_branches(&self) -> Result<HashMap<String, BranchContents>> {
        self.branches().list().await
    }

    fn already_checked_out(
        &self,
        location: &ManifestLocation,
        branch_name: Option<String>,
    ) -> bool {
        // We check the e_tag here just in case it has been overwritten. This can
        // happen if the table has been dropped then re-created recently.
        self.manifest.branch == branch_name
            && self.manifest.version == location.version
            && self.manifest_location.naming_scheme == location.naming_scheme
            && location.e_tag.as_ref().is_some_and(|e_tag| {
                self.manifest_location
                    .e_tag
                    .as_ref()
                    .is_some_and(|current_e_tag| e_tag == current_e_tag)
            })
    }

    async fn checkout_by_ref(
        &self,
        version_number: Option<u64>,
        branch: Option<String>,
    ) -> Result<Self> {
        let new_location = if self.manifest.branch.as_ref() != branch.as_ref() {
            if let Some(branch_name) = branch.as_deref() {
                self.find_branch_location(branch_name)?
            } else {
                self.branch_location().find_main()?
            }
        } else {
            self.branch_location()
        };

        let manifest_location = if let Some(version_number) = version_number {
            self.commit_handler
                .resolve_version_location(
                    &new_location.path,
                    version_number,
                    &self.object_store.inner,
                )
                .await?
        } else {
            self.commit_handler
                .resolve_latest_location(&new_location.path, &self.object_store)
                .await?
        };

        if self.already_checked_out(&manifest_location, branch.clone()) {
            return Ok(self.clone());
        }

        let manifest = Self::load_manifest(
            self.object_store.as_ref(),
            &manifest_location,
            &new_location.uri,
            self.session.as_ref(),
        )
        .await?;
        Self::checkout_manifest(
            self.object_store.clone(),
            new_location.path,
            new_location.uri,
            Arc::new(manifest),
            manifest_location,
            self.session.clone(),
            self.commit_handler.clone(),
            self.file_reader_options.clone(),
            self.store_params.as_deref().cloned(),
        )
    }

    pub(crate) async fn load_manifest(
        object_store: &ObjectStore,
        manifest_location: &ManifestLocation,
        uri: &str,
        session: &Session,
    ) -> Result<Manifest> {
        let object_reader = if let Some(size) = manifest_location.size {
            object_store
                .open_with_size(&manifest_location.path, size as usize)
                .await
        } else {
            object_store.open(&manifest_location.path).await
        };
        let object_reader = object_reader.map_err(|e| match &e {
            Error::NotFound { uri, .. } => Error::DatasetNotFound {
                path: uri.clone(),
                source: box_error(e),
                location: location!(),
            },
            _ => e,
        })?;

        let last_block =
            read_last_block(object_reader.as_ref())
                .await
                .map_err(|err| match err {
                    object_store::Error::NotFound { path, source } => Error::DatasetNotFound {
                        path,
                        source,
                        location: location!(),
                    },
                    _ => Error::IO {
                        source: err.into(),
                        location: location!(),
                    },
                })?;
        let offset = read_metadata_offset(&last_block)?;

        // If manifest is in the last block, we can decode directly from memory.
        let manifest_size = object_reader.size().await?;
        let mut manifest = if manifest_size - offset <= last_block.len() {
            let manifest_len = manifest_size - offset;
            let offset_in_block = last_block.len() - manifest_len;
            let message_len =
                LittleEndian::read_u32(&last_block[offset_in_block..offset_in_block + 4]) as usize;
            let message_data = &last_block[offset_in_block + 4..offset_in_block + 4 + message_len];
            Manifest::try_from(lance_table::format::pb::Manifest::decode(message_data)?)
        } else {
            read_struct(object_reader.as_ref(), offset).await
        }?;

        if !can_read_dataset(manifest.reader_feature_flags) {
            let message = format!(
                "This dataset cannot be read by this version of Lance. \
                 Please upgrade Lance to read this dataset.\n Flags: {}",
                manifest.reader_feature_flags
            );
            return Err(Error::NotSupported {
                source: message.into(),
                location: location!(),
            });
        }

        // If indices were also the last block, we can take the opportunity to
        // decode them now and cache them.
        if let Some(index_offset) = manifest.index_section {
            if manifest_size - index_offset <= last_block.len() {
                let offset_in_block = last_block.len() - (manifest_size - index_offset);
                let message_len =
                    LittleEndian::read_u32(&last_block[offset_in_block..offset_in_block + 4])
                        as usize;
                let message_data =
                    &last_block[offset_in_block + 4..offset_in_block + 4 + message_len];
                let section = lance_table::format::pb::IndexSection::decode(message_data)?;
                let mut indices: Vec<IndexMetadata> = section
                    .indices
                    .into_iter()
                    .map(IndexMetadata::try_from)
                    .collect::<Result<Vec<_>>>()?;
                retain_supported_indices(&mut indices);
                let ds_index_cache = session.index_cache.for_dataset(uri);
                let metadata_key = crate::session::index_caches::IndexMetadataKey {
                    version: manifest_location.version,
                };
                ds_index_cache
                    .insert_with_key(&metadata_key, Arc::new(indices))
                    .await;
            }
        }

        if manifest.should_use_legacy_format() {
            populate_schema_dictionary(&mut manifest.schema, object_reader.as_ref()).await?;
        }

        Ok(manifest)
    }

    #[allow(clippy::too_many_arguments)]
    fn checkout_manifest(
        object_store: Arc<ObjectStore>,
        base_path: Path,
        uri: String,
        manifest: Arc<Manifest>,
        manifest_location: ManifestLocation,
        session: Arc<Session>,
        commit_handler: Arc<dyn CommitHandler>,
        file_reader_options: Option<FileReaderOptions>,
        store_params: Option<ObjectStoreParams>,
    ) -> Result<Self> {
        let refs = Refs::new(
            object_store.clone(),
            commit_handler.clone(),
            BranchLocation {
                path: base_path.clone(),
                uri: uri.clone(),
                branch: manifest.branch.clone(),
            },
        );
        let metadata_cache = Arc::new(session.metadata_cache.for_dataset(&uri));
        let index_cache = Arc::new(session.index_cache.for_dataset(&uri));
        let fragment_bitmap = Arc::new(manifest.fragments.iter().map(|f| f.id as u32).collect());
        Ok(Self {
            object_store,
            base: base_path,
            uri,
            manifest,
            manifest_location,
            commit_handler,
            session,
            refs,
            fragment_bitmap,
            metadata_cache,
            index_cache,
            file_reader_options,
            store_params: store_params.map(Box::new),
        })
    }

    /// Write to or Create a [Dataset] with a stream of [RecordBatch]s.
    ///
    /// `dest` can be a `&str`, `object_store::path::Path` or `Arc<Dataset>`.
    ///
    /// Returns the newly created [`Dataset`].
    /// Or Returns [Error] if the dataset already exists.
    ///
    pub async fn write(
        batches: impl RecordBatchReader + Send + 'static,
        dest: impl Into<WriteDestination<'_>>,
        params: Option<WriteParams>,
    ) -> Result<Self> {
        let mut builder = InsertBuilder::new(dest);
        if let Some(params) = &params {
            builder = builder.with_params(params);
        }
        Box::pin(builder.execute_stream(Box::new(batches) as Box<dyn RecordBatchReader + Send>))
            .await
    }

    /// Append to existing [Dataset] with a stream of [RecordBatch]s
    ///
    /// Returns void result or Returns [Error]
    pub async fn append(
        &mut self,
        batches: impl RecordBatchReader + Send + 'static,
        params: Option<WriteParams>,
    ) -> Result<()> {
        let write_params = WriteParams {
            mode: WriteMode::Append,
            ..params.unwrap_or_default()
        };

        let new_dataset = InsertBuilder::new(WriteDestination::Dataset(Arc::new(self.clone())))
            .with_params(&write_params)
            .execute_stream(Box::new(batches) as Box<dyn RecordBatchReader + Send>)
            .await?;

        *self = new_dataset;

        Ok(())
    }

    /// Get the fully qualified URI of this dataset.
    pub fn uri(&self) -> &str {
        &self.uri
    }

    pub fn branch_location(&self) -> BranchLocation {
        BranchLocation {
            path: self.base.clone(),
            uri: self.uri.clone(),
            branch: self.manifest.branch.clone(),
        }
    }

    pub fn find_branch_location(&self, branch_name: &str) -> Result<BranchLocation> {
        let current_location = BranchLocation {
            path: self.base.clone(),
            uri: self.uri.clone(),
            branch: self.manifest.branch.clone(),
        };
        current_location.find_branch(Some(branch_name.to_string()))
    }

    /// Get the full manifest of the dataset version.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn manifest_location(&self) -> &ManifestLocation {
        &self.manifest_location
    }

    /// Create a [`delta::DatasetDeltaBuilder`] to explore changes between dataset versions.
    ///
    /// # Example
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let delta = dataset.delta()
    ///     .compared_against_version(5)
    ///     .build()?;
    /// let inserted = delta.get_inserted_rows().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn delta(&self) -> delta::DatasetDeltaBuilder {
        delta::DatasetDeltaBuilder::new(self.clone())
    }

    // TODO: Cache this
    pub(crate) fn is_legacy_storage(&self) -> bool {
        self.manifest
            .data_storage_format
            .lance_file_version()
            .unwrap()
            == LanceFileVersion::Legacy
    }

    pub async fn latest_manifest(&self) -> Result<(Arc<Manifest>, ManifestLocation)> {
        let location = self
            .commit_handler
            .resolve_latest_location(&self.base, &self.object_store)
            .await?;

        // Check if manifest is in cache before reading from storage
        let manifest_key = ManifestKey {
            version: location.version,
            e_tag: location.e_tag.as_deref(),
        };
        let cached_manifest = self.metadata_cache.get_with_key(&manifest_key).await;
        if let Some(cached_manifest) = cached_manifest {
            return Ok((cached_manifest, location));
        }

        if self.already_checked_out(&location, self.manifest.branch.clone()) {
            return Ok((self.manifest.clone(), self.manifest_location.clone()));
        }
        let mut manifest = read_manifest(&self.object_store, &location.path, location.size).await?;
        if manifest.schema.has_dictionary_types() && manifest.should_use_legacy_format() {
            let reader = if let Some(size) = location.size {
                self.object_store
                    .open_with_size(&location.path, size as usize)
                    .await?
            } else {
                self.object_store.open(&location.path).await?
            };
            populate_schema_dictionary(&mut manifest.schema, reader.as_ref()).await?;
        }
        Ok((Arc::new(manifest), location))
    }

    /// Read the transaction file for this version of the dataset.
    ///
    /// If there was no transaction file written for this version of the dataset
    /// then this will return None.
    pub async fn read_transaction(&self) -> Result<Option<Transaction>> {
        let path = match &self.manifest.transaction_file {
            Some(path) => self.base.child("_transactions").child(path.as_str()),
            None => return Ok(None),
        };
        let data = self.object_store.inner.get(&path).await?.bytes().await?;
        let transaction = lance_table::format::pb::Transaction::decode(data)?;
        Transaction::try_from(transaction).map(Some)
    }

    /// Read the transaction file for this version of the dataset.
    ///
    /// If there was no transaction file written for this version of the dataset
    /// then this will return None.
    pub async fn read_transaction_by_version(&self, version: u64) -> Result<Option<Transaction>> {
        let dataset_version = self.checkout_version(version).await?;
        dataset_version.read_transaction().await
    }

    /// List transactions for the dataset, up to a maximum number.
    ///
    /// This method iterates through dataset versions, starting from the current version,
    /// and collects the transaction for each version. It stops when either `recent_transactions`
    /// is reached or there are no more versions.
    ///
    /// # Arguments
    ///
    /// * `recent_transactions` - Maximum number of transactions to return
    ///
    /// # Returns
    ///
    /// A vector of optional transactions. Each element corresponds to a version,
    /// and may be None if no transaction file exists for that version.
    pub async fn get_transactions(
        &self,
        recent_transactions: usize,
    ) -> Result<Vec<Option<Transaction>>> {
        let mut transactions = vec![];
        let mut dataset = self.clone();

        loop {
            let transaction = dataset.read_transaction().await?;
            transactions.push(transaction);

            if transactions.len() >= recent_transactions {
                break;
            } else {
                match dataset
                    .checkout_version(dataset.version().version - 1)
                    .await
                {
                    Ok(ds) => dataset = ds,
                    Err(Error::DatasetNotFound { .. }) => break,
                    Err(err) => return Err(err),
                }
            }
        }

        Ok(transactions)
    }

    /// Restore the currently checked out version of the dataset as the latest version.
    pub async fn restore(&mut self) -> Result<()> {
        let (latest_manifest, _) = self.latest_manifest().await?;
        let latest_version = latest_manifest.version;

        let transaction = Transaction::new(
            latest_version,
            Operation::Restore {
                version: self.manifest.version,
            },
            None,
        );

        self.apply_commit(transaction, &Default::default(), &Default::default())
            .await?;

        Ok(())
    }

    /// Removes old versions of the dataset from disk
    ///
    /// This function will remove all versions of the dataset that are older than the provided
    /// timestamp.  This function will not remove the current version of the dataset.
    ///
    /// Once a version is removed it can no longer be checked out or restored.  Any data unique
    /// to that version will be lost.
    ///
    /// # Arguments
    ///
    /// * `older_than` - Versions older than this will be deleted.
    /// * `delete_unverified` - If false (the default) then files will only be deleted if they
    ///                        are listed in at least one manifest.  Otherwise these files will
    ///                        be kept since they cannot be distinguished from an in-progress
    ///                        transaction.  Set to true to delete these files if you are sure
    ///                        there are no other in-progress dataset operations.
    ///
    /// # Returns
    ///
    /// * `RemovalStats` - Statistics about the removal operation
    #[instrument(level = "debug", skip(self))]
    pub fn cleanup_old_versions(
        &self,
        older_than: Duration,
        delete_unverified: Option<bool>,
        error_if_tagged_old_versions: Option<bool>,
    ) -> BoxFuture<'_, Result<RemovalStats>> {
        let mut builder = CleanupPolicyBuilder::default();
        builder = builder.before_timestamp(utc_now() - older_than);
        if let Some(v) = delete_unverified {
            builder = builder.delete_unverified(v);
        }
        if let Some(v) = error_if_tagged_old_versions {
            builder = builder.error_if_tagged_old_versions(v);
        }

        self.cleanup_with_policy(builder.build())
    }

    /// Removes old versions of the dataset from storage
    ///
    /// This function will remove all versions of the dataset that satisfies the given policy.
    /// This function will not remove the current version of the dataset.
    ///
    /// Once a version is removed it can no longer be checked out or restored.  Any data unique
    /// to that version will be lost.
    ///
    /// # Arguments
    ///
    /// * `policy` - `CleanupPolicy` determines the behaviour of cleanup.
    ///
    /// # Returns
    ///
    /// * `RemovalStats` - Statistics about the removal operation
    #[instrument(level = "debug", skip(self))]
    pub fn cleanup_with_policy(
        &self,
        policy: CleanupPolicy,
    ) -> BoxFuture<'_, Result<RemovalStats>> {
        info!(target: TRACE_DATASET_EVENTS, event=DATASET_CLEANING_EVENT, uri=&self.uri);
        cleanup::cleanup_old_versions(self, policy).boxed()
    }

    #[allow(clippy::too_many_arguments)]
    async fn do_commit(
        base_uri: WriteDestination<'_>,
        operation: Operation,
        read_version: Option<u64>,
        store_params: Option<ObjectStoreParams>,
        commit_handler: Option<Arc<dyn CommitHandler>>,
        session: Arc<Session>,
        enable_v2_manifest_paths: bool,
        detached: bool,
    ) -> Result<Self> {
        let read_version = read_version.map_or_else(
            || match operation {
                Operation::Overwrite { .. } | Operation::Restore { .. } => Ok(0),
                _ => Err(Error::invalid_input(
                    "read_version must be specified for this operation",
                    location!(),
                )),
            },
            Ok,
        )?;

        let transaction = Transaction::new(read_version, operation, None);

        let mut builder = CommitBuilder::new(base_uri)
            .enable_v2_manifest_paths(enable_v2_manifest_paths)
            .with_session(session)
            .with_detached(detached);

        if let Some(store_params) = store_params {
            builder = builder.with_store_params(store_params);
        }

        if let Some(commit_handler) = commit_handler {
            builder = builder.with_commit_handler(commit_handler);
        }

        builder.execute(transaction).await
    }

    /// Commit changes to the dataset
    ///
    /// This operation is not needed if you are using append/write/delete to manipulate the dataset.
    /// It is used to commit changes to the dataset that are made externally.  For example, a bulk
    /// import tool may import large amounts of new data and write the appropriate lance files
    /// directly instead of using the write function.
    ///
    /// This method can be used to commit this change to the dataset's manifest.  This method will
    /// not verify that the provided fragments exist and correct, that is the caller's responsibility.
    /// Some validation can be performed using the function
    /// [crate::dataset::transaction::validate_operation].
    ///
    /// If this commit is a change to an existing dataset then it will often need to be based on an
    /// existing version of the dataset.  For example, if this change is a `delete` operation then
    /// the caller will have read in the existing data (at some version) to determine which fragments
    /// need to be deleted.  The base version that the caller used should be supplied as the `read_version`
    /// parameter.  Some operations (e.g. Overwrite) do not depend on a previous version and `read_version`
    /// can be None.  An error will be returned if the `read_version` is needed for an operation and
    /// it is not specified.
    ///
    /// All operations except Overwrite will fail if the dataset does not already exist.
    ///
    /// # Arguments
    ///
    /// * `base_uri` - The base URI of the dataset
    /// * `operation` - A description of the change to commit
    /// * `read_version` - The version of the dataset that this change is based on
    /// * `store_params` Parameters controlling object store access to the manifest
    /// * `enable_v2_manifest_paths`: If set to true, and this is a new dataset, uses the new v2 manifest
    ///   paths. These allow constant-time lookups for the latest manifest on object storage.
    ///   This parameter has no effect on existing datasets. To migrate an existing
    ///   dataset, use the [`Self::migrate_manifest_paths_v2`] method. WARNING: turning
    ///   this on will make the dataset unreadable for older versions of Lance
    ///   (prior to 0.17.0). Default is False.
    pub async fn commit(
        dest: impl Into<WriteDestination<'_>>,
        operation: Operation,
        read_version: Option<u64>,
        store_params: Option<ObjectStoreParams>,
        commit_handler: Option<Arc<dyn CommitHandler>>,
        session: Arc<Session>,
        enable_v2_manifest_paths: bool,
    ) -> Result<Self> {
        Self::do_commit(
            dest.into(),
            operation,
            read_version,
            store_params,
            commit_handler,
            session,
            enable_v2_manifest_paths,
            /*detached=*/ false,
        )
        .await
    }

    /// Commits changes exactly the same as [`Self::commit`] but the commit will
    /// not be associated with the dataset lineage.
    ///
    /// The commit will not show up in the dataset's history and will never be
    /// the latest version of the dataset.
    ///
    /// This can be used to stage changes or to handle "secondary" datasets whose
    /// lineage is tracked elsewhere.
    pub async fn commit_detached(
        dest: impl Into<WriteDestination<'_>>,
        operation: Operation,
        read_version: Option<u64>,
        store_params: Option<ObjectStoreParams>,
        commit_handler: Option<Arc<dyn CommitHandler>>,
        session: Arc<Session>,
        enable_v2_manifest_paths: bool,
    ) -> Result<Self> {
        Self::do_commit(
            dest.into(),
            operation,
            read_version,
            store_params,
            commit_handler,
            session,
            enable_v2_manifest_paths,
            /*detached=*/ true,
        )
        .await
    }

    pub(crate) async fn apply_commit(
        &mut self,
        transaction: Transaction,
        write_config: &ManifestWriteConfig,
        commit_config: &CommitConfig,
    ) -> Result<()> {
        let (manifest, manifest_location) = commit_transaction(
            self,
            self.object_store(),
            self.commit_handler.as_ref(),
            &transaction,
            write_config,
            commit_config,
            self.manifest_location.naming_scheme,
            None,
        )
        .await?;

        self.manifest = Arc::new(manifest);
        self.manifest_location = manifest_location;
        self.fragment_bitmap = Arc::new(
            self.manifest
                .fragments
                .iter()
                .map(|f| f.id as u32)
                .collect(),
        );

        Ok(())
    }

    /// Create a Scanner to scan the dataset.
    pub fn scan(&self) -> Scanner {
        Scanner::new(Arc::new(self.clone()))
    }

    /// Count the number of rows in the dataset.
    ///
    /// It offers a fast path of counting rows by just computing via metadata.
    #[instrument(skip_all)]
    pub async fn count_rows(&self, filter: Option<String>) -> Result<usize> {
        // TODO: consolidate the count_rows into Scanner plan.
        if let Some(filter) = filter {
            let mut scanner = self.scan();
            scanner.filter(&filter)?;
            Ok(scanner
                .project::<String>(&[])?
                .with_row_id() // TODO: fix scan plan to not require row_id for count_rows.
                .count_rows()
                .await? as usize)
        } else {
            self.count_all_rows().await
        }
    }

    pub(crate) async fn count_all_rows(&self) -> Result<usize> {
        let cnts = stream::iter(self.get_fragments())
            .map(|f| async move { f.count_rows(None).await })
            .buffer_unordered(16)
            .try_collect::<Vec<_>>()
            .await?;
        Ok(cnts.iter().sum())
    }

    /// Take rows by indices.
    #[instrument(skip_all, fields(num_rows=row_indices.len()))]
    pub async fn take(
        &self,
        row_indices: &[u64],
        projection: impl Into<ProjectionRequest>,
    ) -> Result<RecordBatch> {
        take::take(self, row_indices, projection.into()).await
    }

    /// Take Rows by the internal ROW ids.
    ///
    /// In Lance format, each row has a unique `u64` id, which is used to identify the row globally.
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// # use tokio::runtime::Runtime;
    /// # use arrow_array::{RecordBatch, RecordBatchIterator, Int64Array};
    /// # use arrow_schema::{Schema, Field, DataType};
    /// # use lance::dataset::{WriteParams, Dataset, ProjectionRequest};
    /// #
    /// # let mut rt = Runtime::new().unwrap();
    /// # rt.block_on(async {
    /// # let test_dir = tempfile::tempdir().unwrap();
    /// # let uri = test_dir.path().to_str().unwrap().to_string();
    /// #
    /// # let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    /// # let write_params = WriteParams::default();
    /// # let array = Arc::new(Int64Array::from_iter(0..128));
    /// # let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
    /// # let reader = RecordBatchIterator::new(
    /// #    vec![batch].into_iter().map(Ok), schema
    /// # );
    /// # let dataset = Dataset::write(reader, &uri, Some(write_params)).await.unwrap();
    /// #
    /// let schema = dataset.schema().clone();
    /// let row_ids = vec![0, 4, 7];
    /// let rows = dataset.take_rows(&row_ids, schema).await.unwrap();
    ///
    /// // We can have more fine-grained control over the projection, i.e., SQL projection.
    /// let projection = ProjectionRequest::from_sql([("identity", "id * 2")]);
    /// let rows = dataset.take_rows(&row_ids, projection).await.unwrap();
    /// # });
    /// ```
    pub async fn take_rows(
        &self,
        row_ids: &[u64],
        projection: impl Into<ProjectionRequest>,
    ) -> Result<RecordBatch> {
        Arc::new(self.clone())
            .take_builder(row_ids, projection)?
            .execute()
            .await
    }

    pub fn take_builder(
        self: &Arc<Self>,
        row_ids: &[u64],
        projection: impl Into<ProjectionRequest>,
    ) -> Result<TakeBuilder> {
        TakeBuilder::try_new_from_ids(self.clone(), row_ids.to_vec(), projection.into())
    }

    /// Take [BlobFile] by row ids (row address).
    pub async fn take_blobs(
        self: &Arc<Self>,
        row_ids: &[u64],
        column: impl AsRef<str>,
    ) -> Result<Vec<BlobFile>> {
        blob::take_blobs(self, row_ids, column.as_ref()).await
    }

    /// Take [BlobFile] by row indices.
    ///
    pub async fn take_blobs_by_indices(
        self: &Arc<Self>,
        row_indices: &[u64],
        column: impl AsRef<str>,
    ) -> Result<Vec<BlobFile>> {
        let row_addrs = row_offsets_to_row_addresses(self, row_indices).await?;
        blob::take_blobs(self, &row_addrs, column.as_ref()).await
    }

    /// Get a stream of batches based on iterator of ranges of row numbers.
    ///
    /// This is an experimental API. It may change at any time.
    pub fn take_scan(
        &self,
        row_ranges: Pin<Box<dyn Stream<Item = Result<Range<u64>>> + Send>>,
        projection: Arc<Schema>,
        batch_readahead: usize,
    ) -> DatasetRecordBatchStream {
        take::take_scan(self, row_ranges, projection, batch_readahead)
    }

    /// Sample `n` rows from the dataset.
    pub(crate) async fn sample(&self, n: usize, projection: &Schema) -> Result<RecordBatch> {
        use rand::seq::IteratorRandom;
        let num_rows = self.count_rows(None).await?;
        let ids = (0..num_rows as u64).choose_multiple(&mut rand::rng(), n);
        self.take(&ids, projection.clone()).await
    }

    /// Delete rows based on a predicate.
    pub async fn delete(&mut self, predicate: &str) -> Result<()> {
        info!(target: TRACE_DATASET_EVENTS, event=DATASET_DELETING_EVENT, uri = &self.uri, predicate=predicate);
        write::delete::delete(self, predicate).await
    }

    /// Add new base paths to the dataset.
    ///
    /// This method allows you to register additional storage locations (buckets)
    /// that can be used for future data writes. The base paths are added to the
    /// dataset's manifest and can be referenced by name in subsequent write operations.
    ///
    /// # Arguments
    ///
    /// * `new_bases` - A vector of `lance_table::format::BasePath` objects representing the new storage
    ///   locations to add. Each base path should have a unique name and path.
    ///
    /// # Returns
    ///
    /// Returns a new `Dataset` instance with the updated manifest containing the
    /// new base paths.
    pub async fn add_bases(
        self: &Arc<Self>,
        new_bases: Vec<lance_table::format::BasePath>,
        transaction_properties: Option<HashMap<String, String>>,
    ) -> Result<Self> {
        let operation = Operation::UpdateBases { new_bases };

        let transaction = TransactionBuilder::new(self.manifest.version, operation)
            .transaction_properties(transaction_properties.map(Arc::new))
            .build();

        let new_dataset = CommitBuilder::new(self.clone())
            .execute(transaction)
            .await?;

        Ok(new_dataset)
    }

    pub async fn count_deleted_rows(&self) -> Result<usize> {
        futures::stream::iter(self.get_fragments())
            .map(|f| async move { f.count_deletions().await })
            .buffer_unordered(self.object_store.io_parallelism())
            .try_fold(0, |acc, x| futures::future::ready(Ok(acc + x)))
            .await
    }

    pub fn object_store(&self) -> &ObjectStore {
        &self.object_store
    }

    /// Returns the storage options used when opening this dataset, if any.
    pub fn storage_options(&self) -> Option<&HashMap<String, String>> {
        self.store_params
            .as_ref()
            .and_then(|params| params.storage_options.as_ref())
    }

    pub fn data_dir(&self) -> Path {
        self.base.child(DATA_DIR)
    }

    pub fn indices_dir(&self) -> Path {
        self.base.child(INDICES_DIR)
    }

    pub(crate) fn data_file_dir(&self, data_file: &DataFile) -> Result<Path> {
        match data_file.base_id.as_ref() {
            Some(base_id) => {
                let base_paths = &self.manifest.base_paths;
                let base_path = base_paths.get(base_id).ok_or_else(|| {
                    Error::invalid_input(
                        format!(
                            "base_path id {} not found for data_file {}",
                            base_id, data_file.path
                        ),
                        location!(),
                    )
                })?;
                let path = base_path.extract_path(self.session.store_registry())?;
                if base_path.is_dataset_root {
                    Ok(path.child(DATA_DIR))
                } else {
                    Ok(path)
                }
            }
            None => Ok(self.base.child(DATA_DIR)),
        }
    }

    /// Get the ObjectStore for a specific path based on base_id
    pub(crate) async fn object_store_for_base(&self, base_id: u32) -> Result<Arc<ObjectStore>> {
        let base_path = self.manifest.base_paths.get(&base_id).ok_or_else(|| {
            Error::invalid_input(
                format!("Dataset base path with ID {} not found", base_id),
                Default::default(),
            )
        })?;

        let (store, _) = ObjectStore::from_uri_and_params(
            self.session.store_registry(),
            &base_path.path,
            &self.store_params.as_deref().cloned().unwrap_or_default(),
        )
        .await?;

        Ok(store)
    }

    pub(crate) fn dataset_dir_for_deletion(&self, deletion_file: &DeletionFile) -> Result<Path> {
        match deletion_file.base_id.as_ref() {
            Some(base_id) => {
                let base_paths = &self.manifest.base_paths;
                let base_path = base_paths.get(base_id).ok_or_else(|| {
                    Error::invalid_input(
                        format!(
                            "base_path id {} not found for deletion_file {:?}",
                            base_id, deletion_file
                        ),
                        location!(),
                    )
                })?;

                if !base_path.is_dataset_root {
                    return Err(Error::Internal {
                        message: format!(
                            "base_path id {} is not a dataset root for deletion_file {:?}",
                            base_id, deletion_file
                        ),
                        location: location!(),
                    });
                }
                base_path.extract_path(self.session.store_registry())
            }
            None => Ok(self.base.clone()),
        }
    }

    /// Get the indices directory for a specific index, considering its base_id
    pub(crate) fn indice_files_dir(&self, index: &IndexMetadata) -> Result<Path> {
        match index.base_id.as_ref() {
            Some(base_id) => {
                let base_paths = &self.manifest.base_paths;
                let base_path = base_paths.get(base_id).ok_or_else(|| {
                    Error::invalid_input(
                        format!(
                            "base_path id {} not found for index {}",
                            base_id, index.uuid
                        ),
                        location!(),
                    )
                })?;
                let path = base_path.extract_path(self.session.store_registry())?;
                if base_path.is_dataset_root {
                    Ok(path.child(INDICES_DIR))
                } else {
                    // For non-dataset-root base paths, we assume the path already points to the indices directory
                    Ok(path)
                }
            }
            None => Ok(self.base.child(INDICES_DIR)),
        }
    }

    pub fn session(&self) -> Arc<Session> {
        self.session.clone()
    }

    pub fn version(&self) -> Version {
        Version::from(self.manifest.as_ref())
    }

    /// Get the number of entries currently in the index cache.
    pub async fn index_cache_entry_count(&self) -> usize {
        self.session.index_cache.size().await
    }

    /// Get cache hit ratio.
    pub async fn index_cache_hit_rate(&self) -> f32 {
        let stats = self.session.index_cache_stats().await;
        stats.hit_ratio()
    }

    pub fn cache_size_bytes(&self) -> u64 {
        self.session.deep_size_of() as u64
    }

    /// Get all versions.
    pub async fn versions(&self) -> Result<Vec<Version>> {
        let mut versions: Vec<Version> = self
            .commit_handler
            .list_manifest_locations(&self.base, &self.object_store, false)
            .try_filter_map(|location| async move {
                match read_manifest(&self.object_store, &location.path, location.size).await {
                    Ok(manifest) => Ok(Some(Version::from(&manifest))),
                    Err(e) => Err(e),
                }
            })
            .try_collect()
            .await?;

        // TODO: this API should support pagination
        versions.sort_by_key(|v| v.version);

        Ok(versions)
    }

    /// Get the latest version of the dataset
    /// This is meant to be a fast path for checking if a dataset has changed. This is why
    /// we don't return the full version struct.
    pub async fn latest_version_id(&self) -> Result<u64> {
        Ok(self
            .commit_handler
            .resolve_latest_location(&self.base, &self.object_store)
            .await?
            .version)
    }

    pub fn count_fragments(&self) -> usize {
        self.manifest.fragments.len()
    }

    /// Get the schema of the dataset
    pub fn schema(&self) -> &Schema {
        &self.manifest.schema
    }

    /// Similar to [Self::schema], but only returns fields that are not marked as blob columns
    /// Creates a new empty projection into the dataset schema
    pub fn empty_projection(self: &Arc<Self>) -> Projection {
        Projection::empty(self.clone())
    }

    /// Creates a projection that includes all columns in the dataset
    pub fn full_projection(self: &Arc<Self>) -> Projection {
        Projection::full(self.clone())
    }

    /// Get fragments.
    ///
    /// If `filter` is provided, only fragments with the given name will be returned.
    pub fn get_fragments(&self) -> Vec<FileFragment> {
        let dataset = Arc::new(self.clone());
        self.manifest
            .fragments
            .iter()
            .map(|f| FileFragment::new(dataset.clone(), f.clone()))
            .collect()
    }

    pub fn get_fragment(&self, fragment_id: usize) -> Option<FileFragment> {
        let dataset = Arc::new(self.clone());
        let fragment = self
            .manifest
            .fragments
            .iter()
            .find(|f| f.id == fragment_id as u64)?;
        Some(FileFragment::new(dataset, fragment.clone()))
    }

    pub fn fragments(&self) -> &Arc<Vec<Fragment>> {
        &self.manifest.fragments
    }

    // Gets a filtered list of fragments from ids in O(N) time instead of using
    // `get_fragment` which would require O(N^2) time.
    pub fn get_frags_from_ordered_ids(&self, ordered_ids: &[u32]) -> Vec<Option<FileFragment>> {
        let mut fragments = Vec::with_capacity(ordered_ids.len());
        let mut id_iter = ordered_ids.iter();
        let mut id = id_iter.next();
        // This field is just used to assert the ids are in order
        let mut last_id: i64 = -1;
        for frag in self.manifest.fragments.iter() {
            let mut the_id = if let Some(id) = id { *id } else { break };
            // Assert the given ids are, in fact, in order
            assert!(the_id as i64 > last_id);
            // For any IDs we've passed we can assume that no fragment exists any longer
            // with that ID.
            while the_id < frag.id as u32 {
                fragments.push(None);
                last_id = the_id as i64;
                id = id_iter.next();
                the_id = if let Some(id) = id { *id } else { break };
            }

            if the_id == frag.id as u32 {
                fragments.push(Some(FileFragment::new(
                    Arc::new(self.clone()),
                    frag.clone(),
                )));
                last_id = the_id as i64;
                id = id_iter.next();
            }
        }
        fragments
    }

    // This method filters deleted items from `addr_or_ids` using `addrs` as a reference
    async fn filter_addr_or_ids(&self, addr_or_ids: &[u64], addrs: &[u64]) -> Result<Vec<u64>> {
        if addrs.is_empty() {
            return Ok(Vec::new());
        }

        let mut perm = permutation::sort(addrs);
        // First we sort the addrs, then we transform from Vec<u64> to Vec<Option<u64>> and then
        // we un-sort and use the None values to filter `addr_or_ids`
        let sorted_addrs = perm.apply_slice(addrs);

        // Only collect deletion vectors for the fragments referenced by the given addrs
        let referenced_frag_ids = sorted_addrs
            .iter()
            .map(|addr| RowAddress::from(*addr).fragment_id())
            .dedup()
            .collect::<Vec<_>>();
        let frags = self.get_frags_from_ordered_ids(&referenced_frag_ids);
        let dv_futs = frags
            .iter()
            .map(|frag| {
                if let Some(frag) = frag {
                    frag.get_deletion_vector().boxed()
                } else {
                    std::future::ready(Ok(None)).boxed()
                }
            })
            .collect::<Vec<_>>();
        let dvs = stream::iter(dv_futs)
            .buffered(self.object_store.io_parallelism())
            .try_collect::<Vec<_>>()
            .await?;

        // Iterate through the sorted addresses and sorted fragments (and sorted deletion vectors)
        // and filter out addresses that have been deleted
        let mut filtered_sorted_addrs = Vec::with_capacity(sorted_addrs.len());
        let mut sorted_addr_iter = sorted_addrs.into_iter().map(RowAddress::from);
        let mut next_addr = sorted_addr_iter.next().unwrap();
        let mut exhausted = false;

        for frag_dv in frags.iter().zip(dvs).zip(referenced_frag_ids.iter()) {
            let ((frag, dv), frag_id) = frag_dv;
            if frag.is_some() {
                // Frag exists
                if let Some(dv) = dv.as_ref() {
                    // Deletion vector exists, scan DV
                    for deleted in dv.to_sorted_iter() {
                        while next_addr.fragment_id() == *frag_id
                            && next_addr.row_offset() < deleted
                        {
                            filtered_sorted_addrs.push(Some(u64::from(next_addr)));
                            if let Some(next) = sorted_addr_iter.next() {
                                next_addr = next;
                            } else {
                                exhausted = true;
                                break;
                            }
                        }
                        if exhausted {
                            break;
                        }
                        if next_addr.fragment_id() != *frag_id {
                            break;
                        }
                        if next_addr.row_offset() == deleted {
                            filtered_sorted_addrs.push(None);
                            if let Some(next) = sorted_addr_iter.next() {
                                next_addr = next;
                            } else {
                                exhausted = true;
                                break;
                            }
                        }
                    }
                }
                if exhausted {
                    break;
                }
                // Either no deletion vector, or we've exhausted it, keep everything else
                // in this frag
                while next_addr.fragment_id() == *frag_id {
                    filtered_sorted_addrs.push(Some(u64::from(next_addr)));
                    if let Some(next) = sorted_addr_iter.next() {
                        next_addr = next;
                    } else {
                        break;
                    }
                }
            } else {
                // Frag doesn't exist (possibly deleted), delete all items
                while next_addr.fragment_id() == *frag_id {
                    filtered_sorted_addrs.push(None);
                    if let Some(next) = sorted_addr_iter.next() {
                        next_addr = next;
                    } else {
                        break;
                    }
                }
            }
        }

        // filtered_sorted_ids is now a Vec with the same size as sorted_addrs, but with None
        // values where the corresponding address was deleted.  We now need to un-sort it and
        // filter out the deleted addresses.
        perm.apply_inv_slice_in_place(&mut filtered_sorted_addrs);
        Ok(addr_or_ids
            .iter()
            .zip(filtered_sorted_addrs)
            .filter_map(|(addr_or_id, maybe_addr)| maybe_addr.map(|_| *addr_or_id))
            .collect())
    }

    pub(crate) async fn filter_deleted_ids(&self, ids: &[u64]) -> Result<Vec<u64>> {
        let addresses = if let Some(row_id_index) = get_row_id_index(self).await? {
            let addresses = ids
                .iter()
                .filter_map(|id| row_id_index.get(*id).map(|address| address.into()))
                .collect::<Vec<_>>();
            Cow::Owned(addresses)
        } else {
            Cow::Borrowed(ids)
        };

        self.filter_addr_or_ids(ids, &addresses).await
    }

    /// Gets the number of files that are so small they don't even have a full
    /// group. These are considered too small because reading many of them is
    /// much less efficient than reading a single file because the separate files
    /// split up what would otherwise be single IO requests into multiple.
    pub async fn num_small_files(&self, max_rows_per_group: usize) -> usize {
        futures::stream::iter(self.get_fragments())
            .map(|f| async move { f.physical_rows().await })
            .buffered(self.object_store.io_parallelism())
            .try_filter(|row_count| futures::future::ready(*row_count < max_rows_per_group))
            .count()
            .await
    }

    pub async fn validate(&self) -> Result<()> {
        // All fragments have unique ids
        let id_counts =
            self.manifest
                .fragments
                .iter()
                .map(|f| f.id)
                .fold(HashMap::new(), |mut acc, id| {
                    *acc.entry(id).or_insert(0) += 1;
                    acc
                });
        for (id, count) in id_counts {
            if count > 1 {
                return Err(Error::corrupt_file(
                    self.base.clone(),
                    format!(
                        "Duplicate fragment id {} found in dataset {:?}",
                        id, self.base
                    ),
                    location!(),
                ));
            }
        }

        // Fragments are sorted in increasing fragment id order
        self.manifest
            .fragments
            .iter()
            .map(|f| f.id)
            .try_fold(0, |prev, id| {
                if id < prev {
                    Err(Error::corrupt_file(
                        self.base.clone(),
                        format!(
                            "Fragment ids are not sorted in increasing fragment-id order. Found {} after {} in dataset {:?}",
                            id, prev, self.base
                        ),
                        location!(),
                    ))
                } else {
                    Ok(id)
                }
            })?;

        // All fragments have equal lengths
        futures::stream::iter(self.get_fragments())
            .map(|f| async move { f.validate().await })
            .buffer_unordered(self.object_store.io_parallelism())
            .try_collect::<Vec<()>>()
            .await?;

        // Validate indices
        let indices = self.load_indices().await?;
        self.validate_indices(&indices)?;

        Ok(())
    }

    fn validate_indices(&self, indices: &[IndexMetadata]) -> Result<()> {
        // Make sure there are no duplicate ids
        let mut index_ids = HashSet::new();
        for index in indices.iter() {
            if !index_ids.insert(&index.uuid) {
                return Err(Error::corrupt_file(
                    self.manifest_location.path.clone(),
                    format!(
                        "Duplicate index id {} found in dataset {:?}",
                        &index.uuid, self.base
                    ),
                    location!(),
                ));
            }
        }

        // For each index name, make sure there is no overlap in fragment bitmaps
        if let Err(err) = detect_overlapping_fragments(indices) {
            let mut message = "Overlapping fragments detected in dataset.".to_string();
            for (index_name, overlapping_frags) in err.bad_indices {
                message.push_str(&format!(
                    "\nIndex {:?} has overlapping fragments: {:?}",
                    index_name, overlapping_frags
                ));
            }
            return Err(Error::corrupt_file(
                self.manifest_location.path.clone(),
                message,
                location!(),
            ));
        };

        Ok(())
    }

    /// Migrate the dataset to use the new manifest path scheme.
    ///
    /// This function will rename all V1 manifests to [ManifestNamingScheme::V2].
    /// These paths provide more efficient opening of datasets with many versions
    /// on object stores.
    ///
    /// This function is idempotent, and can be run multiple times without
    /// changing the state of the object store.
    ///
    /// However, it should not be run while other concurrent operations are happening.
    /// And it should also run until completion before resuming other operations.
    ///
    /// ```rust
    /// # use lance::dataset::Dataset;
    /// # use lance_table::io::commit::ManifestNamingScheme;
    /// # use lance_datagen::{array, RowCount, BatchCount};
    /// # use arrow_array::types::Int32Type;
    /// # let data = lance_datagen::gen_batch()
    /// #  .col("key", array::step::<Int32Type>())
    /// #  .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    /// # let fut = async {
    /// let mut dataset = Dataset::write(data, "memory://test", None).await.unwrap();
    /// assert_eq!(dataset.manifest_location().naming_scheme, ManifestNamingScheme::V1);
    ///
    /// dataset.migrate_manifest_paths_v2().await.unwrap();
    /// assert_eq!(dataset.manifest_location().naming_scheme, ManifestNamingScheme::V2);
    /// # };
    /// # tokio::runtime::Runtime::new().unwrap().block_on(fut);
    /// ```
    pub async fn migrate_manifest_paths_v2(&mut self) -> Result<()> {
        migrate_scheme_to_v2(self.object_store(), &self.base).await?;
        // We need to re-open.
        let latest_version = self.latest_version_id().await?;
        *self = self.checkout_version(latest_version).await?;
        Ok(())
    }

    /// Shallow clone the target version into a new dataset at target_path.
    /// 'target_path': the uri string to clone the dataset into.
    /// 'version': the version cloned from, could be a version number or tag.
    /// 'store_params': the object store params to use for the new dataset.
    pub async fn shallow_clone(
        &mut self,
        target_path: &str,
        version: impl Into<refs::Ref>,
        store_params: Option<ObjectStoreParams>,
    ) -> Result<Self> {
        let ref_ = version.into();
        let (ref_name, version_number) = self.resolve_reference(ref_).await?;
        let clone_op = Operation::Clone {
            is_shallow: true,
            ref_name,
            ref_version: version_number,
            ref_path: self.uri.clone(),
            branch_name: None,
        };
        let transaction = Transaction::new(version_number, clone_op, None);

        let builder = CommitBuilder::new(WriteDestination::Uri(target_path))
            .with_store_params(store_params.unwrap_or_default())
            .with_object_store(Arc::new(self.object_store().clone()))
            .with_commit_handler(self.commit_handler.clone())
            .with_storage_format(self.manifest.data_storage_format.lance_file_version()?);
        builder.execute(transaction).await
    }

    async fn resolve_reference(&self, reference: refs::Ref) -> Result<(Option<String>, u64)> {
        match reference {
            refs::Ref::Version(branch, version_number) => {
                if let Some(version_number) = version_number {
                    Ok((branch, version_number))
                } else {
                    let version_number = self
                        .commit_handler
                        .resolve_latest_location(&self.base, &self.object_store)
                        .await?
                        .version;
                    Ok((branch, version_number))
                }
            }
            refs::Ref::Tag(tag_name) => {
                let tag_contents = self.tags().get(tag_name.as_str()).await?;
                Ok((tag_contents.branch, tag_contents.version))
            }
        }
    }

    /// Run a SQL query against the dataset.
    /// The underlying SQL engine is DataFusion.
    /// Please refer to the DataFusion documentation for supported SQL syntax.
    pub fn sql(&mut self, sql: &str) -> SqlQueryBuilder {
        SqlQueryBuilder::new(self.clone(), sql)
    }

    /// Returns true if Lance supports writing this datatype with nulls.
    pub(crate) fn lance_supports_nulls(&self, datatype: &DataType) -> bool {
        match self
            .manifest()
            .data_storage_format
            .lance_file_version()
            .unwrap_or(LanceFileVersion::Legacy)
            .resolve()
        {
            LanceFileVersion::Legacy => matches!(
                datatype,
                DataType::Utf8
                    | DataType::LargeUtf8
                    | DataType::Binary
                    | DataType::List(_)
                    | DataType::FixedSizeBinary(_)
                    | DataType::FixedSizeList(_, _)
            ),
            LanceFileVersion::V2_0 => !matches!(datatype, DataType::Struct(..)),
            _ => true,
        }
    }
}

pub(crate) struct NewTransactionResult<'a> {
    pub dataset: BoxFuture<'a, Result<Dataset>>,
    pub new_transactions: BoxStream<'a, Result<(u64, Arc<Transaction>)>>,
}

pub(crate) fn load_new_transactions(dataset: &Dataset) -> NewTransactionResult<'_> {
    // Re-use the same list call for getting the latest manifest and the metadata
    // for all manifests in between.
    let io_parallelism = dataset.object_store().io_parallelism();
    let latest_version = dataset.manifest.version;
    let locations = dataset
        .commit_handler
        .list_manifest_locations(&dataset.base, dataset.object_store(), true)
        .try_take_while(move |location| {
            futures::future::ready(Ok(location.version > latest_version))
        });

    // Will send the latest manifest via a channel.
    let (latest_tx, latest_rx) = tokio::sync::oneshot::channel();
    let mut latest_tx = Some(latest_tx);

    let manifests = locations
        .map_ok(move |location| {
            let latest_tx = latest_tx.take();
            async move {
                let manifest_key = ManifestKey {
                    version: location.version,
                    e_tag: location.e_tag.as_deref(),
                };
                let manifest = if let Some(cached) =
                    dataset.metadata_cache.get_with_key(&manifest_key).await
                {
                    cached
                } else {
                    let loaded = Arc::new(
                        Dataset::load_manifest(
                            dataset.object_store(),
                            &location,
                            &dataset.uri,
                            dataset.session.as_ref(),
                        )
                        .await?,
                    );
                    dataset
                        .metadata_cache
                        .insert_with_key(&manifest_key, loaded.clone())
                        .await;
                    loaded
                };

                if let Some(latest_tx) = latest_tx {
                    // We ignore the error, since we don't care if the receiver is dropped.
                    let _ = latest_tx.send((manifest.clone(), location.clone()));
                }

                Ok((manifest, location))
            }
        })
        .try_buffer_unordered(io_parallelism / 2);
    let transactions = manifests
        .map_ok(move |(manifest, location)| async move {
            let manifest_copy = manifest.clone();
            let tx_key = TransactionKey {
                version: manifest.version,
            };
            let transaction =
                if let Some(cached) = dataset.metadata_cache.get_with_key(&tx_key).await {
                    cached
                } else {
                    let dataset_version = Dataset::checkout_manifest(
                        dataset.object_store.clone(),
                        dataset.base.clone(),
                        dataset.uri.clone(),
                        manifest_copy.clone(),
                        location,
                        dataset.session(),
                        dataset.commit_handler.clone(),
                        dataset.file_reader_options.clone(),
                        dataset.store_params.as_deref().cloned(),
                    )?;
                    let object_store = dataset_version.object_store();
                    let path = dataset_version
                        .manifest
                        .transaction_file
                        .as_ref()
                        .ok_or_else(|| Error::Internal {
                            message: format!(
                                "Dataset version {} does not have a transaction file",
                                manifest_copy.version
                            ),
                            location: location!(),
                        })?;
                    let loaded =
                        Arc::new(read_transaction_file(object_store, &dataset.base, path).await?);
                    dataset
                        .metadata_cache
                        .insert_with_key(&tx_key, loaded.clone())
                        .await;
                    loaded
                };
            Ok((manifest.version, transaction))
        })
        .try_buffer_unordered(io_parallelism / 2);

    let dataset = async move {
        if let Ok((latest_manifest, location)) = latest_rx.await {
            // If we got the latest manifest, we can checkout the dataset.
            Dataset::checkout_manifest(
                dataset.object_store.clone(),
                dataset.base.clone(),
                dataset.uri.clone(),
                latest_manifest,
                location,
                dataset.session(),
                dataset.commit_handler.clone(),
                dataset.file_reader_options.clone(),
                dataset.store_params.as_deref().cloned(),
            )
        } else {
            // If we didn't get the latest manifest, we can still return the dataset
            // with the current manifest.
            Ok(dataset.clone())
        }
    }
    .boxed();

    let new_transactions = transactions.boxed();

    NewTransactionResult {
        dataset,
        new_transactions,
    }
}

/// # Schema Evolution
///
/// Lance datasets support evolving the schema. Several operations are
/// supported that mirror common SQL operations:
///
/// - [Self::add_columns()]: Add new columns to the dataset, similar to `ALTER TABLE ADD COLUMN`.
/// - [Self::drop_columns()]: Drop columns from the dataset, similar to `ALTER TABLE DROP COLUMN`.
/// - [Self::alter_columns()]: Modify columns in the dataset, changing their name, type, or nullability.
///   Similar to `ALTER TABLE ALTER COLUMN`.
///
/// In addition, one operation is unique to Lance: [`merge`](Self::merge). This
/// operation allows inserting precomputed data into the dataset.
///
/// Because these operations change the schema of the dataset, they will conflict
/// with most other concurrent operations. Therefore, they should be performed
/// when no other write operations are being run.
impl Dataset {
    /// Append new columns to the dataset.
    pub async fn add_columns(
        &mut self,
        transforms: NewColumnTransform,
        read_columns: Option<Vec<String>>,
        batch_size: Option<u32>,
    ) -> Result<()> {
        schema_evolution::add_columns(self, transforms, read_columns, batch_size).await
    }

    /// Modify columns in the dataset, changing their name, type, or nullability.
    ///
    /// If only changing the name or nullability of a column, this is a zero-copy
    /// operation and any indices will be preserved. If changing the type of a
    /// column, the data for that column will be rewritten and any indices will
    /// be dropped. The old column data will not be immediately deleted. To remove
    /// it, call [optimize::compact_files()] and then
    /// [cleanup::cleanup_old_versions()] on the dataset.
    pub async fn alter_columns(&mut self, alterations: &[ColumnAlteration]) -> Result<()> {
        schema_evolution::alter_columns(self, alterations).await
    }

    /// Remove columns from the dataset.
    ///
    /// This is a metadata-only operation and does not remove the data from the
    /// underlying storage. In order to remove the data, you must subsequently
    /// call [optimize::compact_files()] to rewrite the data without the removed columns and
    /// then call [cleanup::cleanup_old_versions()] to remove the old files.
    pub async fn drop_columns(&mut self, columns: &[&str]) -> Result<()> {
        info!(target: TRACE_DATASET_EVENTS, event=DATASET_DROPPING_COLUMN_EVENT, uri = &self.uri, columns = columns.join(","));
        schema_evolution::drop_columns(self, columns).await
    }

    /// Drop columns from the dataset and return updated dataset. Note that this
    /// is a zero-copy operation and column is not physically removed from the
    /// dataset.
    /// Parameters:
    /// - `columns`: the list of column names to drop.
    #[deprecated(since = "0.9.12", note = "Please use `drop_columns` instead.")]
    pub async fn drop(&mut self, columns: &[&str]) -> Result<()> {
        self.drop_columns(columns).await
    }

    async fn merge_impl(
        &mut self,
        stream: Box<dyn RecordBatchReader + Send>,
        left_on: &str,
        right_on: &str,
    ) -> Result<()> {
        // Sanity check.
        if self.schema().field(left_on).is_none() && left_on != ROW_ID && left_on != ROW_ADDR {
            return Err(Error::invalid_input(
                format!("Column {} does not exist in the left side dataset", left_on),
                location!(),
            ));
        };
        let right_schema = stream.schema();
        if right_schema.field_with_name(right_on).is_err() {
            return Err(Error::invalid_input(
                format!(
                    "Column {} does not exist in the right side dataset",
                    right_on
                ),
                location!(),
            ));
        };
        for field in right_schema.fields() {
            if field.name() == right_on {
                // right_on is allowed to exist in the dataset, since it may be
                // the same as left_on.
                continue;
            }
            if self.schema().field(field.name()).is_some() {
                return Err(Error::invalid_input(
                    format!(
                        "Column {} exists in both sides of the dataset",
                        field.name()
                    ),
                    location!(),
                ));
            }
        }

        // Hash join
        let joiner = Arc::new(HashJoiner::try_new(stream, right_on).await?);
        // Final schema is union of current schema, plus the RHS schema without
        // the right_on key.
        let mut new_schema: Schema = self.schema().merge(joiner.out_schema().as_ref())?;
        new_schema.set_field_id(Some(self.manifest.max_field_id()));

        // Write new data file to each fragment. Parallelism is done over columns,
        // so no parallelism done at this level.
        let updated_fragments: Vec<Fragment> = stream::iter(self.get_fragments())
            .then(|f| {
                let joiner = joiner.clone();
                async move { f.merge(left_on, &joiner).await.map(|f| f.metadata) }
            })
            .try_collect::<Vec<_>>()
            .await?;

        let transaction = Transaction::new(
            self.manifest.version,
            Operation::Merge {
                fragments: updated_fragments,
                schema: new_schema,
            },
            None,
        );

        self.apply_commit(transaction, &Default::default(), &Default::default())
            .await?;

        Ok(())
    }

    /// Merge this dataset with another arrow Table / Dataset, and returns a new version of dataset.
    ///
    /// Parameters:
    ///
    /// - `stream`: the stream of [`RecordBatch`] to merge.
    /// - `left_on`: the column name to join on the left side (self).
    /// - `right_on`: the column name to join on the right side (stream).
    ///
    /// Returns: a new version of dataset.
    ///
    /// It performs a left-join on the two datasets.
    pub async fn merge(
        &mut self,
        stream: impl RecordBatchReader + Send + 'static,
        left_on: &str,
        right_on: &str,
    ) -> Result<()> {
        let stream = Box::new(stream);
        self.merge_impl(stream, left_on, right_on).await
    }
}

/// # Dataset metadata APIs
///
/// There are four kinds of metadata on datasets:
///
///  - **Schema metadata**: metadata about the data itself.
///  - **Field metadata**: metadata about the dataset itself.
///  - **Dataset metadata**: metadata about the dataset. For example, this could
///    store a created_at date.
///  - **Dataset config**: configuration values controlling how engines should
///    manage the dataset. This configures things like auto-cleanup.
///
/// You can get
impl Dataset {
    /// Get dataset metadata.
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.manifest.table_metadata
    }

    /// Get the dataset config from manifest
    pub fn config(&self) -> &HashMap<String, String> {
        &self.manifest.config
    }

    /// Delete keys from the config.
    #[deprecated(
        note = "Use the new update_config(values, replace) method - pass None values to delete keys"
    )]
    pub async fn delete_config_keys(&mut self, delete_keys: &[&str]) -> Result<()> {
        let updates = delete_keys.iter().map(|key| (*key, None));
        self.update_config(updates).await?;
        Ok(())
    }

    /// Update table metadata.
    ///
    /// Pass `None` for a value to remove that key.
    ///
    /// Use `.replace()` to replace the entire metadata map instead of merging.
    ///
    /// Returns the updated metadata map after the operation.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::transaction::UpdateMapEntry;
    /// # async fn test_update_metadata(dataset: &mut Dataset) -> Result<()> {
    /// // Update single key
    /// dataset.update_metadata([("key", "value")]).await?;
    ///
    /// // Remove a key
    /// dataset.update_metadata([("to_delete", None)]).await?;
    ///
    /// // Clear all metadata
    /// dataset.update_metadata([] as [UpdateMapEntry; 0]).replace().await?;
    ///
    /// // Replace full metadata
    /// dataset.update_metadata([("k1", "v1"), ("k2", "v2")]).replace().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn update_metadata(
        &mut self,
        values: impl IntoIterator<Item = impl Into<UpdateMapEntry>>,
    ) -> metadata::UpdateMetadataBuilder<'_> {
        metadata::UpdateMetadataBuilder::new(self, values, metadata::MetadataType::TableMetadata)
    }

    /// Update config.
    ///
    /// Pass `None` for a value to remove that key.
    ///
    /// Use `.replace()` to replace the entire config map instead of merging.
    ///
    /// Returns the updated config map after the operation.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::transaction::UpdateMapEntry;
    /// # async fn test_update_config(dataset: &mut Dataset) -> Result<()> {
    /// // Update single key
    /// dataset.update_config([("key", "value")]).await?;
    ///
    /// // Remove a key
    /// dataset.update_config([("to_delete", None)]).await?;
    ///
    /// // Clear all config
    /// dataset.update_config([] as [UpdateMapEntry; 0]).replace().await?;
    ///
    /// // Replace full config
    /// dataset.update_config([("k1", "v1"), ("k2", "v2")]).replace().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn update_config(
        &mut self,
        values: impl IntoIterator<Item = impl Into<UpdateMapEntry>>,
    ) -> metadata::UpdateMetadataBuilder<'_> {
        metadata::UpdateMetadataBuilder::new(self, values, metadata::MetadataType::Config)
    }

    /// Update schema metadata.
    ///
    /// Pass `None` for a value to remove that key.
    ///
    /// Use `.replace()` to replace the entire schema metadata map instead of merging.
    ///
    /// Returns the updated schema metadata map after the operation.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::transaction::UpdateMapEntry;
    /// # async fn test_update_schema_metadata(dataset: &mut Dataset) -> Result<()> {
    /// // Update single key
    /// dataset.update_schema_metadata([("key", "value")]).await?;
    ///
    /// // Remove a key
    /// dataset.update_schema_metadata([("to_delete", None)]).await?;
    ///
    /// // Clear all schema metadata
    /// dataset.update_schema_metadata([] as [UpdateMapEntry; 0]).replace().await?;
    ///
    /// // Replace full schema metadata
    /// dataset.update_schema_metadata([("k1", "v1"), ("k2", "v2")]).replace().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn update_schema_metadata(
        &mut self,
        values: impl IntoIterator<Item = impl Into<UpdateMapEntry>>,
    ) -> metadata::UpdateMetadataBuilder<'_> {
        metadata::UpdateMetadataBuilder::new(self, values, metadata::MetadataType::SchemaMetadata)
    }

    /// Update schema metadata
    #[deprecated(note = "Use the new update_schema_metadata(values).replace() instead")]
    pub async fn replace_schema_metadata(
        &mut self,
        new_values: impl IntoIterator<Item = (String, String)>,
    ) -> Result<()> {
        let new_values = new_values
            .into_iter()
            .map(|(k, v)| (k, Some(v)))
            .collect::<HashMap<_, _>>();
        self.update_schema_metadata(new_values).replace().await?;
        Ok(())
    }

    /// Update field metadata
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::transaction::UpdateMapEntry;
    /// # async fn test_update_field_metadata(dataset: &mut Dataset) -> Result<()> {
    /// // Update metadata by field path
    /// dataset.update_field_metadata()
    ///     .update("path.to_field", [("key", "value")])?
    ///     .await?;
    ///
    /// // Update metadata by field id
    /// dataset.update_field_metadata()
    ///     .update(12, [("key", "value")])?
    ///     .await?;
    ///
    /// // Clear field metadata
    /// dataset.update_field_metadata()
    ///     .replace("path.to_field", [] as [UpdateMapEntry; 0])?
    ///     .replace(12, [] as [UpdateMapEntry; 0])?
    ///     .await?;
    ///
    /// // Replace field metadata
    /// dataset.update_field_metadata()
    ///     .replace("field_name", [("k1", "v1"), ("k2", "v2")])?
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn update_field_metadata(&mut self) -> UpdateFieldMetadataBuilder<'_> {
        UpdateFieldMetadataBuilder::new(self)
    }

    /// Update field metadata
    pub async fn replace_field_metadata(
        &mut self,
        new_values: impl IntoIterator<Item = (u32, HashMap<String, String>)>,
    ) -> Result<()> {
        let new_values = new_values.into_iter().collect::<HashMap<_, _>>();
        let field_metadata_updates = new_values
            .into_iter()
            .map(|(field_id, metadata)| {
                (
                    field_id as i32,
                    translate_schema_metadata_updates(&metadata),
                )
            })
            .collect();
        metadata::execute_metadata_update(
            self,
            Operation::UpdateConfig {
                config_updates: None,
                table_metadata_updates: None,
                schema_metadata_updates: None,
                field_metadata_updates,
            },
        )
        .await
    }
}

#[async_trait::async_trait]
impl DatasetTakeRows for Dataset {
    fn schema(&self) -> &Schema {
        Self::schema(self)
    }

    async fn take_rows(&self, row_ids: &[u64], projection: &Schema) -> Result<RecordBatch> {
        Self::take_rows(self, row_ids, projection.clone()).await
    }
}

#[derive(Debug)]
pub(crate) struct ManifestWriteConfig {
    auto_set_feature_flags: bool,              // default true
    timestamp: Option<SystemTime>,             // default None
    use_stable_row_ids: bool,                  // default false
    use_legacy_format: Option<bool>,           // default None
    storage_format: Option<DataStorageFormat>, // default None
}

impl Default for ManifestWriteConfig {
    fn default() -> Self {
        Self {
            auto_set_feature_flags: true,
            timestamp: None,
            use_stable_row_ids: false,
            use_legacy_format: None,
            storage_format: None,
        }
    }
}

/// Commit a manifest file and create a copy at the latest manifest path.
pub(crate) async fn write_manifest_file(
    object_store: &ObjectStore,
    commit_handler: &dyn CommitHandler,
    base_path: &Path,
    manifest: &mut Manifest,
    indices: Option<Vec<IndexMetadata>>,
    config: &ManifestWriteConfig,
    naming_scheme: ManifestNamingScheme,
) -> std::result::Result<ManifestLocation, CommitError> {
    if config.auto_set_feature_flags {
        apply_feature_flags(manifest, config.use_stable_row_ids)?;
    }

    manifest.set_timestamp(timestamp_to_nanos(config.timestamp));

    manifest.update_max_fragment_id();

    commit_handler
        .commit(
            manifest,
            indices,
            base_path,
            object_store,
            write_manifest_file_to_path,
            naming_scheme,
        )
        .await
}

fn write_manifest_file_to_path<'a>(
    object_store: &'a ObjectStore,
    manifest: &'a mut Manifest,
    indices: Option<Vec<IndexMetadata>>,
    path: &'a Path,
) -> BoxFuture<'a, Result<WriteResult>> {
    Box::pin(async {
        let mut object_writer = ObjectWriter::new(object_store, path).await?;
        let pos = write_manifest(&mut object_writer, manifest, indices).await?;
        object_writer
            .write_magics(pos, MAJOR_VERSION, MINOR_VERSION, MAGIC)
            .await?;
        let res = object_writer.shutdown().await?;
        info!(target: TRACE_FILE_AUDIT, mode=AUDIT_MODE_CREATE, r#type=AUDIT_TYPE_MANIFEST, path = path.to_string());
        Ok(res)
    })
}

impl Projectable for Dataset {
    fn schema(&self) -> &Schema {
        self.schema()
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;
    use crate::dataset::optimize::{compact_files, CompactionOptions};
    use crate::dataset::transaction::DataReplacementGroup;
    use crate::dataset::WriteMode::Overwrite;
    use crate::index::vector::VectorIndexParams;
    use crate::utils::test::copy_test_data_to_tmp;
    use lance_arrow::FixedSizeListArrayExt;
    use mock_instant::thread_local::MockClock;

    use arrow::array::{as_struct_array, AsArray, GenericListBuilder, GenericStringBuilder};
    use arrow::compute::concat_batches;
    use arrow::datatypes::UInt64Type;
    use arrow_array::{
        builder::StringDictionaryBuilder,
        cast::as_string_array,
        types::{Float32Type, Int32Type},
        ArrayRef, DictionaryArray, Float32Array, Int32Array, Int64Array, Int8Array,
        Int8DictionaryArray, ListArray, RecordBatchIterator, StringArray, UInt16Array, UInt32Array,
    };
    use arrow_array::{
        Array, FixedSizeListArray, GenericStringArray, Int16Array, Int16DictionaryArray,
        LargeBinaryArray, StructArray, UInt64Array,
    };
    use arrow_ord::sort::sort_to_indices;
    use arrow_schema::{
        DataType, Field as ArrowField, Field, Fields as ArrowFields, Schema as ArrowSchema,
    };
    use lance_arrow::bfloat16::{self, BFLOAT16_EXT_NAME};
    use lance_arrow::{ARROW_EXT_META_KEY, ARROW_EXT_NAME_KEY, BLOB_META_KEY};
    use lance_core::utils::tempfile::{TempDir, TempStdDir, TempStrDir};
    use lance_datagen::{array, gen_batch, BatchCount, Dimension, RowCount};
    use lance_file::v2::writer::FileWriter;
    use lance_file::version::LanceFileVersion;
    use lance_index::scalar::inverted::{
        query::{BooleanQuery, MatchQuery, Occur, Operator, PhraseQuery},
        tokenizer::InvertedIndexParams,
    };
    use lance_index::scalar::FullTextSearchQuery;
    use lance_index::{scalar::ScalarIndexParams, vector::DIST_COL, IndexType};
    use lance_io::assert_io_eq;
    use lance_io::utils::tracking_store::IOTracker;
    use lance_io::utils::CachedFileSize;
    use lance_linalg::distance::MetricType;
    use lance_table::feature_flags;
    use lance_table::format::{DataFile, WriterVersion};

    use crate::datafusion::LanceTableProvider;
    use crate::dataset::refs::branch_contents_path;
    use datafusion::common::{assert_contains, assert_not_contains};
    use datafusion::prelude::SessionContext;
    use lance_arrow::json::ARROW_JSON_EXT_NAME;
    use lance_datafusion::datagen::DatafusionDatagenExt;
    use lance_datafusion::udf::register_functions;
    use lance_index::scalar::inverted::query::{FtsQuery, MultiMatchQuery};
    use lance_testing::datagen::generate_random_array;
    use pretty_assertions::assert_eq;
    use rand::seq::SliceRandom;
    use rand::Rng;
    use rstest::rstest;
    use std::cmp::Ordering;

    // Used to validate that futures returned are Send.
    fn require_send<T: Send>(t: T) -> T {
        t
    }

    async fn create_file(
        path: &std::path::Path,
        mode: WriteMode,
        data_storage_version: LanceFileVersion,
    ) {
        let fields = vec![
            ArrowField::new("i", DataType::Int32, false),
            ArrowField::new(
                "dict",
                DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
                false,
            ),
        ];
        let schema = Arc::new(ArrowSchema::new(fields));
        let dict_values = StringArray::from_iter_values(["a", "b", "c", "d", "e"]);
        let batches: Vec<RecordBatch> = (0..20)
            .map(|i| {
                let mut arrays =
                    vec![Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)) as ArrayRef];
                arrays.push(Arc::new(
                    DictionaryArray::try_new(
                        UInt16Array::from_iter_values((0_u16..20_u16).map(|v| v % 5)),
                        Arc::new(dict_values.clone()),
                    )
                    .unwrap(),
                ));
                RecordBatch::try_new(schema.clone(), arrays).unwrap()
            })
            .collect();
        let expected_batches = batches.clone();

        let test_uri = path.to_str().unwrap();
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            mode,
            data_storage_version: Some(data_storage_version),
            ..WriteParams::default()
        };
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let actual_ds = Dataset::open(test_uri).await.unwrap();
        assert_eq!(actual_ds.version().version, 1);
        assert_eq!(
            actual_ds.manifest.writer_version,
            Some(WriterVersion::default())
        );
        let actual_schema = ArrowSchema::from(actual_ds.schema());
        assert_eq!(&actual_schema, schema.as_ref());

        let actual_batches = actual_ds
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        // The batch size batches the group size.
        // (the v2 writer has no concept of group size)
        if data_storage_version == LanceFileVersion::Legacy {
            for batch in &actual_batches {
                assert_eq!(batch.num_rows(), 10);
            }
        }

        // sort
        let actual_batch = concat_batches(&schema, &actual_batches).unwrap();
        let idx_arr = actual_batch.column_by_name("i").unwrap();
        let sorted_indices = sort_to_indices(idx_arr, None, None).unwrap();
        let struct_arr: StructArray = actual_batch.into();
        let sorted_arr = arrow_select::take::take(&struct_arr, &sorted_indices, None).unwrap();

        let expected_struct_arr: StructArray =
            concat_batches(&schema, &expected_batches).unwrap().into();
        assert_eq!(&expected_struct_arr, as_struct_array(sorted_arr.as_ref()));

        // Each fragments has different fragment ID
        assert_eq!(
            actual_ds
                .fragments()
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            (0..10).collect::<Vec<_>>()
        )
    }

    #[rstest]
    #[lance_test_macros::test(tokio::test)]
    async fn test_create_dataset(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // Appending / Overwriting a dataset that does not exist is treated as Create
        for mode in [WriteMode::Create, WriteMode::Append, Overwrite] {
            let test_dir = TempStdDir::default();
            create_file(&test_dir, mode, data_storage_version).await
        }
    }

    #[rstest]
    #[lance_test_macros::test(tokio::test)]
    async fn test_create_and_fill_empty_dataset(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_uri = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let i32_array: ArrayRef = Arc::new(Int32Array::new(vec![].into(), None));
        let batch = RecordBatch::try_from_iter(vec![("i", i32_array)]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        // check schema of reader and original is same
        assert_eq!(schema.as_ref(), reader.schema().as_ref());
        let result = Dataset::write(
            reader,
            &test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // check dataset empty
        assert_eq!(result.count_rows(None).await.unwrap(), 0);
        // Since the dataset is empty, will return None.
        assert_eq!(result.manifest.max_fragment_id(), None);

        // append rows to dataset
        let mut write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        // We should be able to append even if the metadata doesn't exactly match.
        let schema_with_meta = Arc::new(
            schema
                .as_ref()
                .clone()
                .with_metadata([("key".to_string(), "value".to_string())].into()),
        );
        let batches = vec![RecordBatch::try_new(
            schema_with_meta,
            vec![Arc::new(Int32Array::from_iter_values(0..10))],
        )
        .unwrap()];
        write_params.mode = WriteMode::Append;
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, &test_uri, Some(write_params))
            .await
            .unwrap();

        let expected_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..10))],
        )
        .unwrap();

        // get actual dataset
        let actual_ds = Dataset::open(&test_uri).await.unwrap();
        // confirm schema is same
        let actual_schema = ArrowSchema::from(actual_ds.schema());
        assert_eq!(&actual_schema, schema.as_ref());
        // check num rows is 10
        assert_eq!(actual_ds.count_rows(None).await.unwrap(), 10);
        // Max fragment id is still 0 since we only have 1 fragment.
        assert_eq!(actual_ds.manifest.max_fragment_id(), Some(0));
        // check expected batch is correct
        let actual_batches = actual_ds
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        // sort
        let actual_batch = concat_batches(&schema, &actual_batches).unwrap();
        let idx_arr = actual_batch.column_by_name("i").unwrap();
        let sorted_indices = sort_to_indices(idx_arr, None, None).unwrap();
        let struct_arr: StructArray = actual_batch.into();
        let sorted_arr = arrow_select::take::take(&struct_arr, &sorted_indices, None).unwrap();
        let expected_struct_arr: StructArray = expected_batch.into();
        assert_eq!(&expected_struct_arr, as_struct_array(sorted_arr.as_ref()));
    }

    #[rstest]
    #[lance_test_macros::test(tokio::test)]
    async fn test_create_with_empty_iter(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_uri = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let reader = RecordBatchIterator::new(vec![].into_iter().map(Ok), schema.clone());
        // check schema of reader and original is same
        assert_eq!(schema.as_ref(), reader.schema().as_ref());
        let write_params = Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        });
        let result = Dataset::write(reader, &test_uri, write_params)
            .await
            .unwrap();

        // check dataset empty
        assert_eq!(result.count_rows(None).await.unwrap(), 0);
        // Since the dataset is empty, will return None.
        assert_eq!(result.manifest.max_fragment_id(), None);
    }

    #[tokio::test]
    async fn test_load_manifest_iops() {
        // Need to use in-memory for accurate IOPS tracking.
        let io_tracker = Arc::new(IOTracker::default());

        // Use consistent session so memory store can be reused.
        let session = Arc::new(Session::default());
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let _original_ds = Dataset::write(
            batches,
            "memory://test",
            Some(WriteParams {
                store_params: Some(ObjectStoreParams {
                    object_store_wrapper: Some(io_tracker.clone()),
                    ..Default::default()
                }),
                session: Some(session.clone()),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let _ = io_tracker.incremental_stats(); //reset

        let _dataset = DatasetBuilder::from_uri("memory://test")
            .with_read_params(ReadParams {
                store_options: Some(ObjectStoreParams {
                    object_store_wrapper: Some(io_tracker.clone()),
                    ..Default::default()
                }),
                session: Some(session),
                ..Default::default()
            })
            .load()
            .await
            .unwrap();

        // There should be only two IOPS:
        // 1. List _versions directory to get the latest manifest location
        // 2. Read the manifest file. (The manifest is small enough to be read in one go.
        //    Larger manifests would result in more IOPS.)
        let io_stats = io_tracker.incremental_stats();
        assert_io_eq!(io_stats, read_iops, 2);
    }

    #[rstest]
    #[tokio::test]
    async fn test_write_params(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        use fragment::FragReadConfig;

        let test_uri = TempStrDir::default();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let num_rows: usize = 1_000;
        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..num_rows as i32))],
        )
        .unwrap()];

        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

        let write_params = WriteParams {
            max_rows_per_file: 100,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let dataset = Dataset::write(batches, &test_uri, Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), num_rows);

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 10);
        assert_eq!(dataset.count_fragments(), 10);
        for fragment in &fragments {
            assert_eq!(fragment.count_rows(None).await.unwrap(), 100);
            let reader = fragment
                .open(dataset.schema(), FragReadConfig::default())
                .await
                .unwrap();
            // No group / batch concept in v2
            if data_storage_version == LanceFileVersion::Legacy {
                assert_eq!(reader.legacy_num_batches(), 10);
                for i in 0..reader.legacy_num_batches() as u32 {
                    assert_eq!(reader.legacy_num_rows_in_batch(i).unwrap(), 10);
                }
            }
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_write_manifest(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        use lance_table::feature_flags::FLAG_UNKNOWN;

        let test_uri = TempStrDir::default();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..20))],
        )
        .unwrap()];

        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let write_fut = Dataset::write(
            batches,
            &test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                auto_cleanup: None,
                ..Default::default()
            }),
        );
        let write_fut = require_send(write_fut);
        let mut dataset = write_fut.await.unwrap();

        // Check it has no flags
        let manifest = read_manifest(
            dataset.object_store(),
            &dataset
                .commit_handler
                .resolve_latest_location(&dataset.base, dataset.object_store())
                .await
                .unwrap()
                .path,
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            manifest.data_storage_format,
            DataStorageFormat::new(data_storage_version)
        );
        assert_eq!(manifest.reader_feature_flags, 0);

        // Create one with deletions
        dataset.delete("i < 10").await.unwrap();
        dataset.validate().await.unwrap();

        // Check it set the flag
        let mut manifest = read_manifest(
            dataset.object_store(),
            &dataset
                .commit_handler
                .resolve_latest_location(&dataset.base, dataset.object_store())
                .await
                .unwrap()
                .path,
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            manifest.writer_feature_flags,
            feature_flags::FLAG_DELETION_FILES
        );
        assert_eq!(
            manifest.reader_feature_flags,
            feature_flags::FLAG_DELETION_FILES
        );

        // Write with custom manifest
        manifest.writer_feature_flags |= FLAG_UNKNOWN; // Set another flag
        manifest.reader_feature_flags |= FLAG_UNKNOWN;
        manifest.version += 1;
        write_manifest_file(
            dataset.object_store(),
            dataset.commit_handler.as_ref(),
            &dataset.base,
            &mut manifest,
            None,
            &ManifestWriteConfig {
                auto_set_feature_flags: false,
                timestamp: None,
                use_stable_row_ids: false,
                use_legacy_format: None,
                storage_format: None,
            },
            dataset.manifest_location.naming_scheme,
        )
        .await
        .unwrap();

        // Check it rejects reading it
        let read_result = Dataset::open(&test_uri).await;
        assert!(matches!(read_result, Err(Error::NotSupported { .. })));

        // Check it rejects writing to it.
        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..20))],
        )
        .unwrap()];
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let write_result = Dataset::write(
            batches,
            &test_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await;

        assert!(matches!(write_result, Err(Error::NotSupported { .. })));
    }

    #[rstest]
    #[tokio::test]
    async fn append_dataset(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_uri = TempStrDir::default();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..20))],
        )
        .unwrap()];

        let mut write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, &test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(20..40))],
        )
        .unwrap()];
        write_params.mode = WriteMode::Append;
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, &test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let expected_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..40))],
        )
        .unwrap();

        let actual_ds = Dataset::open(&test_uri).await.unwrap();
        assert_eq!(actual_ds.version().version, 2);
        let actual_schema = ArrowSchema::from(actual_ds.schema());
        assert_eq!(&actual_schema, schema.as_ref());

        let actual_batches = actual_ds
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        // sort
        let actual_batch = concat_batches(&schema, &actual_batches).unwrap();
        let idx_arr = actual_batch.column_by_name("i").unwrap();
        let sorted_indices = sort_to_indices(idx_arr, None, None).unwrap();
        let struct_arr: StructArray = actual_batch.into();
        let sorted_arr = arrow_select::take::take(&struct_arr, &sorted_indices, None).unwrap();

        let expected_struct_arr: StructArray = expected_batch.into();
        assert_eq!(&expected_struct_arr, as_struct_array(sorted_arr.as_ref()));

        // Each fragments has different fragment ID
        assert_eq!(
            actual_ds
                .fragments()
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            (0..2).collect::<Vec<_>>()
        )
    }

    #[rstest]
    #[tokio::test]
    async fn test_shallow_clone_with_hybrid_paths(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStdDir::default();
        let base_dir = test_dir.join("base");
        let test_uri = base_dir.to_str().unwrap();
        let clone_dir = test_dir.join("clone");
        let cloned_uri = clone_dir.to_str().unwrap();

        // Generate consistent test data batches
        let generate_data = |prefix: &str, start_id: i32, row_count: u64| {
            gen_batch()
                .col("id", array::step_custom::<Int32Type>(start_id, 1))
                .col("value", array::fill_utf8(format!("{prefix}_data")))
                .into_reader_rows(RowCount::from(row_count), BatchCount::from(1))
        };

        // Reusable dataset writer with configurable mode
        async fn write_dataset(
            uri: &str,
            data_reader: impl RecordBatchReader + Send + 'static,
            mode: WriteMode,
            version: LanceFileVersion,
        ) -> Dataset {
            let params = WriteParams {
                max_rows_per_file: 100,
                max_rows_per_group: 20,
                data_storage_version: Some(version),
                mode,
                ..Default::default()
            };
            Dataset::write(data_reader, uri, Some(params))
                .await
                .unwrap()
        }

        // Unified dataset scanning and row counting
        async fn collect_rows(dataset: &Dataset) -> (usize, Vec<RecordBatch>) {
            let batches = dataset
                .scan()
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            (batches.iter().map(|b| b.num_rows()).sum(), batches)
        }

        // Create initial dataset
        let mut dataset = write_dataset(
            test_uri,
            generate_data("initial", 0, 50),
            WriteMode::Create,
            data_storage_version,
        )
        .await;

        // Store original state for comparison
        let original_version = dataset.version().version;
        let original_fragment_count = dataset.fragments().len();

        // Create tag and shallow clone
        dataset
            .tags()
            .create("test_tag", original_version)
            .await
            .unwrap();
        let cloned_dataset = dataset
            .shallow_clone(cloned_uri, "test_tag", None)
            .await
            .unwrap();

        // Verify cloned dataset state
        let (cloned_rows, _) = collect_rows(&cloned_dataset).await;
        assert_eq!(cloned_rows, 50);
        assert_eq!(cloned_dataset.version().version, original_version);

        // Append data to cloned dataset
        let updated_cloned = write_dataset(
            cloned_uri,
            generate_data("cloned_new", 50, 30),
            WriteMode::Append,
            data_storage_version,
        )
        .await;

        // Verify updated cloned dataset
        let (updated_cloned_rows, updated_batches) = collect_rows(&updated_cloned).await;
        assert_eq!(updated_cloned_rows, 80);
        assert_eq!(updated_cloned.version().version, original_version + 1);

        // Append data to original dataset
        let updated_original = write_dataset(
            test_uri,
            generate_data("original_new", 50, 25),
            WriteMode::Append,
            data_storage_version,
        )
        .await;

        // Verify updated original dataset
        let (original_rows, _) = collect_rows(&updated_original).await;
        assert_eq!(original_rows, 75);
        assert_eq!(updated_original.version().version, original_version + 1);

        // Final validations
        // Verify cloned dataset isolation
        let final_cloned = Dataset::open(cloned_uri).await.unwrap();
        let (final_cloned_rows, _) = collect_rows(&final_cloned).await;

        // Data integrity check
        let combined_batch =
            concat_batches(&updated_batches[0].schema(), &updated_batches).unwrap();
        assert_eq!(combined_batch.column_by_name("id").unwrap().len(), 80);
        assert_eq!(combined_batch.column_by_name("value").unwrap().len(), 80);

        // Fragment count validation
        assert_eq!(
            updated_original.fragments().len(),
            original_fragment_count + 1
        );
        assert_eq!(final_cloned.fragments().len(), original_fragment_count + 1);

        // Final assertions
        assert_eq!(final_cloned_rows, 80);
        assert_eq!(final_cloned.version().version, original_version + 1);
    }

    #[rstest]
    #[tokio::test]
    async fn test_shallow_clone_multiple_times(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_uri = TempStrDir::default();
        let append_row_count = 36;

        // Async dataset writer function
        async fn write_dataset(
            dest: impl Into<WriteDestination<'_>>,
            row_count: u64,
            mode: WriteMode,
            version: LanceFileVersion,
        ) -> Dataset {
            let data = gen_batch()
                .col("index", array::step::<Int32Type>())
                .col("category", array::fill_utf8("base".to_string()))
                .col("score", array::step_custom::<Float32Type>(1.0, 0.5));
            Dataset::write(
                data.into_reader_rows(RowCount::from(row_count), BatchCount::from(1)),
                dest,
                Some(WriteParams {
                    max_rows_per_file: 60,
                    max_rows_per_group: 12,
                    mode,
                    data_storage_version: Some(version),
                    ..Default::default()
                }),
            )
            .await
            .unwrap()
        }

        let mut current_dataset = write_dataset(
            &test_uri,
            append_row_count,
            WriteMode::Create,
            data_storage_version,
        )
        .await;

        let test_round = 3;
        // Generate clone paths
        let clone_paths = (1..=test_round)
            .map(|i| format!("{}/clone{}", test_uri, i))
            .collect::<Vec<_>>();
        let mut cloned_datasets = Vec::with_capacity(test_round);

        // Unified cloning procedure, write a fragment to each cloned dataset.
        for path in clone_paths.iter() {
            current_dataset
                .tags()
                .create("v1", current_dataset.latest_version_id().await.unwrap())
                .await
                .unwrap();

            current_dataset = current_dataset
                .shallow_clone(path, "v1", None)
                .await
                .unwrap();
            current_dataset = write_dataset(
                Arc::new(current_dataset),
                append_row_count,
                WriteMode::Append,
                data_storage_version,
            )
            .await;
            cloned_datasets.push(current_dataset.clone());
        }

        // Validation function
        async fn validate_dataset(
            dataset: &Dataset,
            expected_rows: usize,
            expected_fragments_count: usize,
            expected_base_paths_count: usize,
        ) {
            let batches = dataset
                .scan()
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();

            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total_rows, expected_rows);
            assert_eq!(dataset.fragments().len(), expected_fragments_count);
            assert_eq!(
                dataset.manifest().base_paths.len(),
                expected_base_paths_count
            );
        }

        // Verify cloned datasets row count, fragment count, base_path count
        for (i, ds) in cloned_datasets.iter().enumerate() {
            validate_dataset(ds, 36 * (i + 2), i + 2, i + 1).await;
        }

        // Verify original dataset row count, fragment count, base_path count
        let original = Dataset::open(&test_uri).await.unwrap();
        validate_dataset(&original, 36, 1, 0).await;
    }

    #[rstest]
    #[tokio::test]
    async fn test_self_dataset_append(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_uri = TempStrDir::default();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..20))],
        )
        .unwrap()];

        let mut write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let mut ds = Dataset::write(batches, &test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(20..40))],
        )
        .unwrap()];
        write_params.mode = WriteMode::Append;
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

        ds.append(batches, Some(write_params.clone()))
            .await
            .unwrap();

        let expected_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..40))],
        )
        .unwrap();

        let actual_ds = Dataset::open(&test_uri).await.unwrap();
        assert_eq!(actual_ds.version().version, 2);
        // validate fragment ids
        assert_eq!(actual_ds.fragments().len(), 2);
        assert_eq!(
            actual_ds
                .fragments()
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            (0..2).collect::<Vec<_>>()
        );

        let actual_schema = ArrowSchema::from(actual_ds.schema());
        assert_eq!(&actual_schema, schema.as_ref());

        let actual_batches = actual_ds
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        // sort
        let actual_batch = concat_batches(&schema, &actual_batches).unwrap();
        let idx_arr = actual_batch.column_by_name("i").unwrap();
        let sorted_indices = sort_to_indices(idx_arr, None, None).unwrap();
        let struct_arr: StructArray = actual_batch.into();
        let sorted_arr = arrow_select::take::take(&struct_arr, &sorted_indices, None).unwrap();

        let expected_struct_arr: StructArray = expected_batch.into();
        assert_eq!(&expected_struct_arr, as_struct_array(sorted_arr.as_ref()));

        actual_ds.validate().await.unwrap();
    }

    #[rstest]
    #[tokio::test]
    async fn test_self_dataset_append_schema_different(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_uri = TempStrDir::default();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..20))],
        )
        .unwrap()];

        let other_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int64,
            false,
        )]));
        let other_batches = vec![RecordBatch::try_new(
            other_schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(0..20))],
        )
        .unwrap()];

        let mut write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let mut ds = Dataset::write(batches, &test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        write_params.mode = WriteMode::Append;
        let other_batches =
            RecordBatchIterator::new(other_batches.into_iter().map(Ok), other_schema.clone());

        let result = ds.append(other_batches, Some(write_params.clone())).await;
        // Error because schema is different
        assert!(matches!(result, Err(Error::SchemaMismatch { .. })))
    }

    #[rstest]
    #[tokio::test]
    async fn append_dictionary(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // We store the dictionary as part of the schema, so we check that the
        // dictionary is consistent between appends.

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "x",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            false,
        )]));
        let dictionary = Arc::new(StringArray::from(vec!["a", "b"]));
        let indices = Int8Array::from(vec![0, 1, 0]);
        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(
                Int8DictionaryArray::try_new(indices, dictionary.clone()).unwrap(),
            )],
        )
        .unwrap()];

        let test_uri = TempStrDir::default();
        let mut write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, &test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        // create a new one with same dictionary
        let indices = Int8Array::from(vec![1, 0, 1]);
        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(
                Int8DictionaryArray::try_new(indices, dictionary).unwrap(),
            )],
        )
        .unwrap()];

        // Write to dataset (successful)
        write_params.mode = WriteMode::Append;
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, &test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        // Create a new one with *different* dictionary
        let dictionary = Arc::new(StringArray::from(vec!["d", "c"]));
        let indices = Int8Array::from(vec![1, 0, 1]);
        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(
                Int8DictionaryArray::try_new(indices, dictionary).unwrap(),
            )],
        )
        .unwrap()];

        // Try write to dataset (fails with legacy format)
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let result = Dataset::write(batches, &test_uri, Some(write_params)).await;
        if data_storage_version == LanceFileVersion::Legacy {
            assert!(result.is_err());
        } else {
            assert!(result.is_ok());
        }
    }

    #[rstest]
    #[tokio::test]
    async fn overwrite_dataset(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_uri = TempStrDir::default();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..20))],
        )
        .unwrap()];

        let mut write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let dataset = Dataset::write(batches, &test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(0));

        let new_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "s",
            DataType::Utf8,
            false,
        )]));
        let new_batches = vec![RecordBatch::try_new(
            new_schema.clone(),
            vec![Arc::new(StringArray::from_iter_values(
                (20..40).map(|v| v.to_string()),
            ))],
        )
        .unwrap()];
        write_params.mode = Overwrite;
        let new_batch_reader =
            RecordBatchIterator::new(new_batches.into_iter().map(Ok), new_schema.clone());
        let dataset = Dataset::write(new_batch_reader, &test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        // Fragment ids reset after overwrite.
        assert_eq!(fragments[0].id(), 0);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(0));

        let actual_ds = Dataset::open(&test_uri).await.unwrap();
        assert_eq!(actual_ds.version().version, 2);
        let actual_schema = ArrowSchema::from(actual_ds.schema());
        assert_eq!(&actual_schema, new_schema.as_ref());

        let actual_batches = actual_ds
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let actual_batch = concat_batches(&new_schema, &actual_batches).unwrap();

        assert_eq!(new_schema.clone(), actual_batch.schema());
        let arr = actual_batch.column_by_name("s").unwrap();
        assert_eq!(
            &StringArray::from_iter_values((20..40).map(|v| v.to_string())),
            as_string_array(arr)
        );
        assert_eq!(actual_ds.version().version, 2);

        // But we can still check out the first version
        let first_ver = DatasetBuilder::from_uri(&test_uri)
            .with_version(1)
            .load()
            .await
            .unwrap();
        assert_eq!(first_ver.version().version, 1);
        assert_eq!(&ArrowSchema::from(first_ver.schema()), schema.as_ref());
    }

    #[rstest]
    #[tokio::test]
    async fn test_fast_count_rows(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_uri = TempStrDir::default();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));

        let batches: Vec<RecordBatch> = (0..20)
            .map(|i| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20))],
                )
                .unwrap()
            })
            .collect();

        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, &test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(&test_uri).await.unwrap();
        dataset.validate().await.unwrap();
        assert_eq!(10, dataset.fragments().len());
        assert_eq!(400, dataset.count_rows(None).await.unwrap());
        assert_eq!(
            200,
            dataset
                .count_rows(Some("i < 200".to_string()))
                .await
                .unwrap()
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_create_index(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_uri = TempStrDir::default();

        let dimension = 16;
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "embeddings",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                dimension,
            ),
            false,
        )]));

        let float_arr = generate_random_array(512 * dimension as usize);
        let vectors = Arc::new(
            <arrow_array::FixedSizeListArray as FixedSizeListArrayExt>::try_new_from_values(
                float_arr, dimension,
            )
            .unwrap(),
        );
        let batches = vec![RecordBatch::try_new(schema.clone(), vec![vectors.clone()]).unwrap()];

        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(
            reader,
            &test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset.validate().await.unwrap();

        // Make sure valid arguments should create index successfully
        let params = VectorIndexParams::ivf_pq(10, 8, 2, MetricType::L2, 50);
        dataset
            .create_index(&["embeddings"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
        dataset.validate().await.unwrap();

        // The version should match the table version it was created from.
        let indices = dataset.load_indices().await.unwrap();
        let actual = indices.first().unwrap().dataset_version;
        let expected = dataset.manifest.version - 1;
        assert_eq!(actual, expected);
        let fragment_bitmap = indices.first().unwrap().fragment_bitmap.as_ref().unwrap();
        assert_eq!(fragment_bitmap.len(), 1);
        assert!(fragment_bitmap.contains(0));

        // Append should inherit index
        let write_params = WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = vec![RecordBatch::try_new(schema.clone(), vec![vectors.clone()]).unwrap()];
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let dataset = Dataset::write(reader, &test_uri, Some(write_params))
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let actual = indices.first().unwrap().dataset_version;
        let expected = dataset.manifest.version - 2;
        assert_eq!(actual, expected);
        dataset.validate().await.unwrap();
        // Fragment bitmap should show the original fragments, and not include
        // the newly appended fragment.
        let fragment_bitmap = indices.first().unwrap().fragment_bitmap.as_ref().unwrap();
        assert_eq!(fragment_bitmap.len(), 1);
        assert!(fragment_bitmap.contains(0));

        let actual_statistics: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("embeddings_idx").await.unwrap())
                .unwrap();
        let actual_statistics = actual_statistics.as_object().unwrap();
        assert_eq!(actual_statistics["index_type"].as_str().unwrap(), "IVF_PQ");

        let deltas = actual_statistics["indices"].as_array().unwrap();
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0]["metric_type"].as_str().unwrap(), "l2");
        assert_eq!(deltas[0]["num_partitions"].as_i64().unwrap(), 10);

        assert!(dataset.index_statistics("non-existent_idx").await.is_err());
        assert!(dataset.index_statistics("").await.is_err());

        // Overwrite should invalidate index
        let write_params = WriteParams {
            mode: WriteMode::Overwrite,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = vec![RecordBatch::try_new(schema.clone(), vec![vectors]).unwrap()];
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let dataset = Dataset::write(reader, &test_uri, Some(write_params))
            .await
            .unwrap();
        assert!(dataset.manifest.index_section.is_none());
        assert!(dataset.load_indices().await.unwrap().is_empty());
        dataset.validate().await.unwrap();

        let fragment_bitmap = indices.first().unwrap().fragment_bitmap.as_ref().unwrap();
        assert_eq!(fragment_bitmap.len(), 1);
        assert!(fragment_bitmap.contains(0));
    }

    #[rstest]
    #[tokio::test]
    async fn test_create_scalar_index(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] use_stable_row_id: bool,
    ) {
        let test_uri = TempStrDir::default();

        let data = gen_batch().col("int", array::step::<Int32Type>());
        // Write 64Ki rows.  We should get 16 4Ki pages
        let mut dataset = Dataset::write(
            data.into_reader_rows(RowCount::from(16 * 1024), BatchCount::from(4)),
            &test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                enable_stable_row_ids: use_stable_row_id,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let index_name = "my_index".to_string();

        dataset
            .create_index(
                &["int"],
                IndexType::Scalar,
                Some(index_name.clone()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        let indices = dataset.load_indices_by_name(&index_name).await.unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].dataset_version, 1);
        assert_eq!(indices[0].fields, vec![0]);
        assert_eq!(indices[0].name, index_name);

        dataset.index_statistics(&index_name).await.unwrap();
    }

    async fn create_bad_file(data_storage_version: LanceFileVersion) -> Result<Dataset> {
        let test_uri = TempStrDir::default();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a.b.c",
            DataType::Int32,
            false,
        )]));

        let batches: Vec<RecordBatch> = (0..20)
            .map(|i| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20))],
                )
                .unwrap()
            })
            .collect();
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(
            reader,
            &test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
    }

    #[tokio::test]
    async fn test_create_fts_index_with_empty_table() {
        let test_uri = TempStrDir::default();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "text",
            DataType::Utf8,
            false,
        )]));

        let batches: Vec<RecordBatch> = vec![];
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(reader, &test_uri, None)
            .await
            .expect("write dataset");

        let params = InvertedIndexParams::default();
        dataset
            .create_index(&["text"], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();

        let batch = dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new("lance".to_owned()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(batch.num_rows(), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn test_create_int8_index(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        use lance_testing::datagen::generate_random_int8_array;

        let test_uri = TempStrDir::default();

        let dimension = 16;
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "embeddings",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Int8, true)),
                dimension,
            ),
            false,
        )]));

        let int8_arr = generate_random_int8_array(512 * dimension as usize);
        let vectors = Arc::new(
            <arrow_array::FixedSizeListArray as FixedSizeListArrayExt>::try_new_from_values(
                int8_arr, dimension,
            )
            .unwrap(),
        );
        let batches = vec![RecordBatch::try_new(schema.clone(), vec![vectors.clone()]).unwrap()];

        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(
            reader,
            &test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset.validate().await.unwrap();

        // Make sure valid arguments should create index successfully
        let params = VectorIndexParams::ivf_pq(10, 8, 2, MetricType::L2, 50);
        dataset
            .create_index(&["embeddings"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
        dataset.validate().await.unwrap();

        // The version should match the table version it was created from.
        let indices = dataset.load_indices().await.unwrap();
        let actual = indices.first().unwrap().dataset_version;
        let expected = dataset.manifest.version - 1;
        assert_eq!(actual, expected);
        let fragment_bitmap = indices.first().unwrap().fragment_bitmap.as_ref().unwrap();
        assert_eq!(fragment_bitmap.len(), 1);
        assert!(fragment_bitmap.contains(0));

        // Append should inherit index
        let write_params = WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = vec![RecordBatch::try_new(schema.clone(), vec![vectors.clone()]).unwrap()];
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let dataset = Dataset::write(reader, &test_uri, Some(write_params))
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let actual = indices.first().unwrap().dataset_version;
        let expected = dataset.manifest.version - 2;
        assert_eq!(actual, expected);
        dataset.validate().await.unwrap();
        // Fragment bitmap should show the original fragments, and not include
        // the newly appended fragment.
        let fragment_bitmap = indices.first().unwrap().fragment_bitmap.as_ref().unwrap();
        assert_eq!(fragment_bitmap.len(), 1);
        assert!(fragment_bitmap.contains(0));

        let actual_statistics: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("embeddings_idx").await.unwrap())
                .unwrap();
        let actual_statistics = actual_statistics.as_object().unwrap();
        assert_eq!(actual_statistics["index_type"].as_str().unwrap(), "IVF_PQ");

        let deltas = actual_statistics["indices"].as_array().unwrap();
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0]["metric_type"].as_str().unwrap(), "l2");
        assert_eq!(deltas[0]["num_partitions"].as_i64().unwrap(), 10);

        assert!(dataset.index_statistics("non-existent_idx").await.is_err());
        assert!(dataset.index_statistics("").await.is_err());

        // Overwrite should invalidate index
        let write_params = WriteParams {
            mode: WriteMode::Overwrite,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = vec![RecordBatch::try_new(schema.clone(), vec![vectors]).unwrap()];
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let dataset = Dataset::write(reader, &test_uri, Some(write_params))
            .await
            .unwrap();
        assert!(dataset.manifest.index_section.is_none());
        assert!(dataset.load_indices().await.unwrap().is_empty());
        dataset.validate().await.unwrap();

        let fragment_bitmap = indices.first().unwrap().fragment_bitmap.as_ref().unwrap();
        assert_eq!(fragment_bitmap.len(), 1);
        assert!(fragment_bitmap.contains(0));
    }

    #[tokio::test]
    async fn test_create_fts_index_with_empty_strings() {
        let test_uri = TempStrDir::default();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "text",
            DataType::Utf8,
            false,
        )]));

        let batches: Vec<RecordBatch> = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec!["", "", ""]))],
        )
        .unwrap()];
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(reader, &test_uri, None)
            .await
            .expect("write dataset");

        let params = InvertedIndexParams::default();
        dataset
            .create_index(&["text"], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();

        let batch = dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new("lance".to_owned()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(batch.num_rows(), 0);
    }

    #[rstest]
    #[tokio::test]
    async fn test_bad_field_name(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // don't allow `.` in the field name
        assert!(create_bad_file(data_storage_version).await.is_err());
    }

    #[tokio::test]
    async fn test_open_dataset_not_found() {
        let result = Dataset::open(".").await;
        assert!(matches!(result.unwrap_err(), Error::DatasetNotFound { .. }));
    }

    fn assert_all_manifests_use_scheme(test_dir: &TempStdDir, scheme: ManifestNamingScheme) {
        let entries_names = test_dir
            .join("_versions")
            .read_dir()
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect::<Vec<_>>();
        assert!(
            entries_names
                .iter()
                .all(|name| ManifestNamingScheme::detect_scheme(name) == Some(scheme)),
            "Entries: {:?}",
            entries_names
        );
    }

    #[tokio::test]
    async fn test_v2_manifest_path_create() {
        // Can create a dataset, using V2 paths
        let data = lance_datagen::gen_batch()
            .col("key", array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(10))
            .unwrap();
        let test_dir = TempStdDir::default();
        let test_uri = test_dir.to_str().unwrap();
        Dataset::write(
            RecordBatchIterator::new([Ok(data.clone())], data.schema().clone()),
            test_uri,
            Some(WriteParams {
                enable_v2_manifest_paths: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_all_manifests_use_scheme(&test_dir, ManifestNamingScheme::V2);

        // Appending to it will continue to use those paths
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(data.clone())], data.schema().clone()),
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_all_manifests_use_scheme(&test_dir, ManifestNamingScheme::V2);

        UpdateBuilder::new(Arc::new(dataset))
            .update_where("key = 5")
            .unwrap()
            .set("key", "200")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        assert_all_manifests_use_scheme(&test_dir, ManifestNamingScheme::V2);
    }

    #[tokio::test]
    async fn test_v2_manifest_path_commit() {
        let schema = Schema::try_from(&ArrowSchema::new(vec![ArrowField::new(
            "x",
            DataType::Int32,
            false,
        )]))
        .unwrap();
        let operation = Operation::Overwrite {
            fragments: vec![],
            schema,
            config_upsert_values: None,
            initial_bases: None,
        };
        let test_dir = TempStdDir::default();
        let test_uri = test_dir.to_str().unwrap();
        let dataset = Dataset::commit(
            test_uri,
            operation,
            None,
            None,
            None,
            Default::default(),
            true, // enable_v2_manifest_paths
        )
        .await
        .unwrap();

        assert!(dataset.manifest_location.naming_scheme == ManifestNamingScheme::V2);

        assert_all_manifests_use_scheme(&test_dir, ManifestNamingScheme::V2);
    }

    #[tokio::test]
    async fn test_strict_overwrite() {
        let schema = Schema::try_from(&ArrowSchema::new(vec![ArrowField::new(
            "x",
            DataType::Int32,
            false,
        )]))
        .unwrap();
        let operation = Operation::Overwrite {
            fragments: vec![],
            schema,
            config_upsert_values: None,
            initial_bases: None,
        };
        let test_uri = TempStrDir::default();
        let read_version_0_transaction = Transaction::new(0, operation, None);
        let strict_builder = CommitBuilder::new(&test_uri).with_max_retries(0);
        let unstrict_builder = CommitBuilder::new(&test_uri).with_max_retries(1);
        strict_builder
            .clone()
            .execute(read_version_0_transaction.clone())
            .await
            .expect("Strict overwrite should succeed when writing a new dataset");
        strict_builder
            .clone()
            .execute(read_version_0_transaction.clone())
            .await
            .expect_err("Strict overwrite should fail when committing to a stale version");
        unstrict_builder
            .clone()
            .execute(read_version_0_transaction.clone())
            .await
            .expect("Unstrict overwrite should succeed when committing to a stale version");
    }

    #[rstest]
    #[tokio::test]
    async fn test_merge(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] use_stable_row_id: bool,
    ) {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, false),
            ArrowField::new("x", DataType::Float32, false),
        ]));
        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(Float32Array::from(vec![1.0, 2.0])),
            ],
        )
        .unwrap();
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![3, 2])),
                Arc::new(Float32Array::from(vec![3.0, 4.0])),
            ],
        )
        .unwrap();

        let test_uri = TempStrDir::default();

        let write_params = WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids: use_stable_row_id,
            ..Default::default()
        };

        let batches = RecordBatchIterator::new(vec![batch1].into_iter().map(Ok), schema.clone());
        Dataset::write(batches, &test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let batches = RecordBatchIterator::new(vec![batch2].into_iter().map(Ok), schema.clone());
        Dataset::write(batches, &test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let dataset = Dataset::open(&test_uri).await.unwrap();
        assert_eq!(dataset.fragments().len(), 2);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(1));

        let right_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i2", DataType::Int32, false),
            ArrowField::new("y", DataType::Utf8, true),
        ]));
        let right_batch1 = RecordBatch::try_new(
            right_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();

        let batches =
            RecordBatchIterator::new(vec![right_batch1].into_iter().map(Ok), right_schema.clone());
        let mut dataset = Dataset::open(&test_uri).await.unwrap();
        dataset.merge(batches, "i", "i2").await.unwrap();
        dataset.validate().await.unwrap();

        assert_eq!(dataset.version().version, 3);
        assert_eq!(dataset.fragments().len(), 2);
        assert_eq!(dataset.fragments()[0].files.len(), 2);
        assert_eq!(dataset.fragments()[1].files.len(), 2);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(1));

        let actual_batches = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let actual = concat_batches(&actual_batches[0].schema(), &actual_batches).unwrap();
        let expected = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                ArrowField::new("i", DataType::Int32, false),
                ArrowField::new("x", DataType::Float32, false),
                ArrowField::new("y", DataType::Utf8, true),
            ])),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 2])),
                Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0, 4.0])),
                Arc::new(StringArray::from(vec![
                    Some("a"),
                    Some("b"),
                    None,
                    Some("b"),
                ])),
            ],
        )
        .unwrap();

        assert_eq!(actual, expected);

        // Validate we can still read after re-instantiating dataset, which
        // clears the cache.
        let dataset = Dataset::open(&test_uri).await.unwrap();
        let actual_batches = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let actual = concat_batches(&actual_batches[0].schema(), &actual_batches).unwrap();
        assert_eq!(actual, expected);
    }

    #[rstest]
    #[tokio::test]
    async fn test_large_merge(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] use_stable_row_id: bool,
    ) {
        // Tests a merge that spans multiple batches within files

        // This test also tests "null filling" when merging (e.g. when keys do not match
        // we need to insert nulls)

        let data = lance_datagen::gen_batch()
            .col("key", array::step::<Int32Type>())
            .col("value", array::fill_utf8("value".to_string()))
            .into_reader_rows(RowCount::from(1_000), BatchCount::from(10));

        let test_uri = TempStrDir::default();

        let write_params = WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(data_storage_version),
            max_rows_per_file: 1024,
            max_rows_per_group: 150,
            enable_stable_row_ids: use_stable_row_id,
            ..Default::default()
        };
        Dataset::write(data, &test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let mut dataset = Dataset::open(&test_uri).await.unwrap();
        assert_eq!(dataset.fragments().len(), 10);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(9));

        let new_data = lance_datagen::gen_batch()
            .col("key2", array::step_custom::<Int32Type>(500, 1))
            .col("new_value", array::fill_utf8("new_value".to_string()))
            .into_reader_rows(RowCount::from(1_000), BatchCount::from(10));

        dataset.merge(new_data, "key", "key2").await.unwrap();
        dataset.validate().await.unwrap();
    }

    #[rstest]
    #[tokio::test]
    async fn test_merge_on_row_id(
        #[values(LanceFileVersion::Stable)] data_storage_version: LanceFileVersion,
        #[values(false, true)] use_stable_row_id: bool,
    ) {
        // Tests a merge on _rowid

        let data = lance_datagen::gen_batch()
            .col("key", array::step::<Int32Type>())
            .col("value", array::fill_utf8("value".to_string()))
            .into_reader_rows(RowCount::from(1_000), BatchCount::from(10));

        let write_params = WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(data_storage_version),
            max_rows_per_file: 1024,
            max_rows_per_group: 150,
            enable_stable_row_ids: use_stable_row_id,
            ..Default::default()
        };
        let mut dataset = Dataset::write(data, "memory://", Some(write_params.clone()))
            .await
            .unwrap();
        assert_eq!(dataset.fragments().len(), 10);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(9));

        let data = dataset.scan().with_row_id().try_into_batch().await.unwrap();
        let row_ids: Arc<dyn Array> = data[ROW_ID].clone();
        let key = data["key"].as_primitive::<Int32Type>();
        let new_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("rowid", DataType::UInt64, false),
            ArrowField::new("new_value", DataType::Int32, false),
        ]));
        let new_value = Arc::new(
            key.into_iter()
                .map(|v| v.unwrap() + 1)
                .collect::<arrow_array::Int32Array>(),
        );
        let len = new_value.len() as u32;
        let new_batch = RecordBatch::try_new(new_schema.clone(), vec![row_ids, new_value]).unwrap();
        // shuffle new_batch
        let mut rng = rand::rng();
        let mut indices: Vec<u32> = (0..len).collect();
        indices.shuffle(&mut rng);
        let indices = arrow_array::UInt32Array::from_iter_values(indices);
        let new_batch = arrow::compute::take_record_batch(&new_batch, &indices).unwrap();
        let new_data = RecordBatchIterator::new(vec![Ok(new_batch)], new_schema.clone());
        dataset.merge(new_data, ROW_ID, "rowid").await.unwrap();
        dataset.validate().await.unwrap();
        assert_eq!(dataset.schema().fields.len(), 3);
        assert!(dataset.schema().field("key").is_some());
        assert!(dataset.schema().field("value").is_some());
        assert!(dataset.schema().field("new_value").is_some());
        let batch = dataset.scan().try_into_batch().await.unwrap();
        let key = batch["key"].as_primitive::<Int32Type>();
        let new_value = batch["new_value"].as_primitive::<Int32Type>();
        for i in 0..key.len() {
            assert_eq!(key.value(i) + 1, new_value.value(i));
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_merge_on_row_addr(
        #[values(LanceFileVersion::Stable)] data_storage_version: LanceFileVersion,
        #[values(false, true)] use_stable_row_id: bool,
    ) {
        // Tests a merge on _rowaddr

        let data = lance_datagen::gen_batch()
            .col("key", array::step::<Int32Type>())
            .col("value", array::fill_utf8("value".to_string()))
            .into_reader_rows(RowCount::from(1_000), BatchCount::from(10));

        let write_params = WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(data_storage_version),
            max_rows_per_file: 1024,
            max_rows_per_group: 150,
            enable_stable_row_ids: use_stable_row_id,
            ..Default::default()
        };
        let mut dataset = Dataset::write(data, "memory://", Some(write_params.clone()))
            .await
            .unwrap();

        assert_eq!(dataset.fragments().len(), 10);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(9));

        let data = dataset
            .scan()
            .with_row_address()
            .try_into_batch()
            .await
            .unwrap();
        let row_addrs = data[ROW_ADDR].clone();
        let key = data["key"].as_primitive::<Int32Type>();
        let new_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("rowaddr", DataType::UInt64, false),
            ArrowField::new("new_value", DataType::Int32, false),
        ]));
        let new_value = Arc::new(
            key.into_iter()
                .map(|v| v.unwrap() + 1)
                .collect::<arrow_array::Int32Array>(),
        );
        let len = new_value.len() as u32;
        let new_batch =
            RecordBatch::try_new(new_schema.clone(), vec![row_addrs, new_value]).unwrap();
        // shuffle new_batch
        let mut rng = rand::rng();
        let mut indices: Vec<u32> = (0..len).collect();
        indices.shuffle(&mut rng);
        let indices = arrow_array::UInt32Array::from_iter_values(indices);
        let new_batch = arrow::compute::take_record_batch(&new_batch, &indices).unwrap();
        let new_data = RecordBatchIterator::new(vec![Ok(new_batch)], new_schema.clone());
        dataset.merge(new_data, ROW_ADDR, "rowaddr").await.unwrap();
        dataset.validate().await.unwrap();
        assert_eq!(dataset.schema().fields.len(), 3);
        assert!(dataset.schema().field("key").is_some());
        assert!(dataset.schema().field("value").is_some());
        assert!(dataset.schema().field("new_value").is_some());
        let batch = dataset.scan().try_into_batch().await.unwrap();
        let key = batch["key"].as_primitive::<Int32Type>();
        let new_value = batch["new_value"].as_primitive::<Int32Type>();
        for i in 0..key.len() {
            assert_eq!(key.value(i) + 1, new_value.value(i));
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_restore(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // Create a table
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::UInt32,
            false,
        )]));

        let test_uri = TempStrDir::default();

        let data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(UInt32Array::from_iter_values(0..100))],
        );
        let reader = RecordBatchIterator::new(vec![data.unwrap()].into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(
            reader,
            &test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.manifest.version, 1);
        let original_manifest = dataset.manifest.clone();

        // Delete some rows
        dataset.delete("i > 50").await.unwrap();
        assert_eq!(dataset.manifest.version, 2);

        // Checkout a previous version
        let mut dataset = dataset.checkout_version(1).await.unwrap();
        assert_eq!(dataset.manifest.version, 1);
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert_eq!(dataset.count_fragments(), 1);
        assert_eq!(fragments[0].metadata.deletion_file, None);
        assert_eq!(dataset.manifest, original_manifest);

        // Checkout latest and then go back.
        dataset.checkout_latest().await.unwrap();
        assert_eq!(dataset.manifest.version, 2);
        let mut dataset = dataset.checkout_version(1).await.unwrap();

        // Restore to a previous version
        dataset.restore().await.unwrap();
        assert_eq!(dataset.manifest.version, 3);
        assert_eq!(dataset.manifest.fragments, original_manifest.fragments);
        assert_eq!(dataset.manifest.schema, original_manifest.schema);

        // Delete some rows again (make sure we can still write as usual)
        dataset.delete("i > 30").await.unwrap();
        assert_eq!(dataset.manifest.version, 4);
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert_eq!(dataset.count_fragments(), 1);
        assert!(fragments[0].metadata.deletion_file.is_some());
    }

    #[rstest]
    #[tokio::test]
    async fn test_tag(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // Create a table
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::UInt32,
            false,
        )]));

        let test_uri = TempStrDir::default();

        let data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(UInt32Array::from_iter_values(0..100))],
        );
        let reader = RecordBatchIterator::new(vec![data.unwrap()].into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(
            reader,
            &test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.manifest.version, 1);

        // delete some rows
        dataset.delete("i > 50").await.unwrap();
        assert_eq!(dataset.manifest.version, 2);

        assert_eq!(dataset.tags().list().await.unwrap().len(), 0);

        let bad_tag_creation = dataset.tags().create("tag1", 3).await;
        assert_eq!(
            bad_tag_creation.err().unwrap().to_string(),
            "Version not found error: version Main::3 does not exist"
        );

        let bad_tag_deletion = dataset.tags().delete("tag1").await;
        assert_eq!(
            bad_tag_deletion.err().unwrap().to_string(),
            "Ref not found error: tag tag1 does not exist"
        );

        dataset.tags().create("tag1", 1).await.unwrap();

        assert_eq!(dataset.tags().list().await.unwrap().len(), 1);

        let another_bad_tag_creation = dataset.tags().create("tag1", 1).await;
        assert_eq!(
            another_bad_tag_creation.err().unwrap().to_string(),
            "Ref conflict error: tag tag1 already exists"
        );

        dataset.tags().delete("tag1").await.unwrap();

        assert_eq!(dataset.tags().list().await.unwrap().len(), 0);

        dataset.tags().create("tag1", 1).await.unwrap();
        dataset.tags().create("tag2", 1).await.unwrap();
        dataset.tags().create("v1.0.0-rc1", 2).await.unwrap();

        let default_order = dataset.tags().list_tags_ordered(None).await.unwrap();
        let default_names: Vec<_> = default_order.iter().map(|t| &t.0).collect();
        assert_eq!(
            default_names,
            ["v1.0.0-rc1", "tag1", "tag2"],
            "Default ordering mismatch"
        );

        let asc_order = dataset
            .tags()
            .list_tags_ordered(Some(Ordering::Less))
            .await
            .unwrap();
        let asc_names: Vec<_> = asc_order.iter().map(|t| &t.0).collect();
        assert_eq!(
            asc_names,
            ["tag1", "tag2", "v1.0.0-rc1"],
            "Ascending ordering mismatch"
        );

        let desc_order = dataset
            .tags()
            .list_tags_ordered(Some(Ordering::Greater))
            .await
            .unwrap();
        let desc_names: Vec<_> = desc_order.iter().map(|t| &t.0).collect();
        assert_eq!(
            desc_names,
            ["v1.0.0-rc1", "tag1", "tag2"],
            "Descending ordering mismatch"
        );

        assert_eq!(dataset.tags().list().await.unwrap().len(), 3);

        let bad_checkout = dataset.checkout_version("tag3").await;
        assert_eq!(
            bad_checkout.err().unwrap().to_string(),
            "Ref not found error: tag tag3 does not exist"
        );

        dataset = dataset.checkout_version("tag1").await.unwrap();
        assert_eq!(dataset.manifest.version, 1);

        let first_ver = DatasetBuilder::from_uri(&test_uri)
            .with_tag("tag1")
            .load()
            .await
            .unwrap();
        assert_eq!(first_ver.version().version, 1);

        // test update tag
        let bad_tag_update = dataset.tags().update("tag3", 1).await;
        assert_eq!(
            bad_tag_update.err().unwrap().to_string(),
            "Ref not found error: tag tag3 does not exist"
        );

        let another_bad_tag_update = dataset.tags().update("tag1", 3).await;
        assert_eq!(
            another_bad_tag_update.err().unwrap().to_string(),
            "Version not found error: version 3 does not exist"
        );

        dataset.tags().update("tag1", 2).await.unwrap();
        dataset = dataset.checkout_version("tag1").await.unwrap();
        assert_eq!(dataset.manifest.version, 2);

        dataset.tags().update("tag1", 1).await.unwrap();
        dataset = dataset.checkout_version("tag1").await.unwrap();
        assert_eq!(dataset.manifest.version, 1);
    }

    #[rstest]
    #[tokio::test]
    async fn test_search_empty(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // Create a table
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "vec",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                128,
            ),
            false,
        )]));

        let test_uri = TempStrDir::default();

        let vectors = Arc::new(
            <arrow_array::FixedSizeListArray as FixedSizeListArrayExt>::try_new_from_values(
                Float32Array::from_iter_values(vec![]),
                128,
            )
            .unwrap(),
        );

        let data = RecordBatch::try_new(schema.clone(), vec![vectors]);
        let reader = RecordBatchIterator::new(vec![data.unwrap()].into_iter().map(Ok), schema);
        let dataset = Dataset::write(
            reader,
            &test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let mut stream = dataset
            .scan()
            .nearest(
                "vec",
                &Float32Array::from_iter_values((0..128).map(|_| 0.1)),
                1,
            )
            .unwrap()
            .try_into_stream()
            .await
            .unwrap();

        while let Some(batch) = stream.next().await {
            let schema = batch.unwrap().schema();
            assert_eq!(schema.fields.len(), 2);
            assert_eq!(
                schema.field_with_name("vec").unwrap(),
                &ArrowField::new(
                    "vec",
                    DataType::FixedSizeList(
                        Arc::new(ArrowField::new("item", DataType::Float32, true)),
                        128
                    ),
                    false,
                )
            );
            assert_eq!(
                schema.field_with_name(DIST_COL).unwrap(),
                &ArrowField::new(DIST_COL, DataType::Float32, true)
            );
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_search_empty_after_delete(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] use_stable_row_id: bool,
    ) {
        // Create a table
        let test_uri = TempStrDir::default();

        let data = gen_batch().col("vec", array::rand_vec::<Float32Type>(Dimension::from(32)));
        let reader = data.into_reader_rows(RowCount::from(500), BatchCount::from(1));
        let mut dataset = Dataset::write(
            reader,
            &test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                enable_stable_row_ids: use_stable_row_id,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let params = VectorIndexParams::ivf_pq(1, 8, 1, MetricType::L2, 50);
        dataset
            .create_index(&["vec"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        dataset.delete("true").await.unwrap();

        // This behavior will be re-introduced once we work on empty vector index handling.
        // https://github.com/lancedb/lance/issues/4034
        // let indices = dataset.load_indices().await.unwrap();
        // // With the new retention behavior, indices are kept even when all fragments are deleted
        // // This allows the index configuration to persist through data changes
        // assert_eq!(indices.len(), 1);

        // // Verify the index has an empty effective fragment bitmap
        // let index = &indices[0];
        // let effective_bitmap = index
        //     .effective_fragment_bitmap(&dataset.fragment_bitmap)
        //     .unwrap();
        // assert!(effective_bitmap.is_empty());

        let mut stream = dataset
            .scan()
            .nearest(
                "vec",
                &Float32Array::from_iter_values((0..32).map(|_| 0.1)),
                1,
            )
            .unwrap()
            .try_into_stream()
            .await
            .unwrap();

        while let Some(batch) = stream.next().await {
            let schema = batch.unwrap().schema();
            assert_eq!(schema.fields.len(), 2);
            assert_eq!(
                schema.field_with_name("vec").unwrap(),
                &ArrowField::new(
                    "vec",
                    DataType::FixedSizeList(
                        Arc::new(ArrowField::new("item", DataType::Float32, true)),
                        32
                    ),
                    false,
                )
            );
            assert_eq!(
                schema.field_with_name(DIST_COL).unwrap(),
                &ArrowField::new(DIST_COL, DataType::Float32, true)
            );
        }

        // predicate with redundant whitespace
        dataset.delete(" True").await.unwrap();

        let mut stream = dataset
            .scan()
            .nearest(
                "vec",
                &Float32Array::from_iter_values((0..32).map(|_| 0.1)),
                1,
            )
            .unwrap()
            .try_into_stream()
            .await
            .unwrap();

        while let Some(batch) = stream.next().await {
            let batch = batch.unwrap();
            let schema = batch.schema();
            assert_eq!(schema.fields.len(), 2);
            assert_eq!(
                schema.field_with_name("vec").unwrap(),
                &ArrowField::new(
                    "vec",
                    DataType::FixedSizeList(
                        Arc::new(ArrowField::new("item", DataType::Float32, true)),
                        32
                    ),
                    false,
                )
            );
            assert_eq!(
                schema.field_with_name(DIST_COL).unwrap(),
                &ArrowField::new(DIST_COL, DataType::Float32, true)
            );
            assert_eq!(batch.num_rows(), 0, "Expected no results after delete");
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_num_small_files(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_uri = TempStrDir::default();
        let dimensions = 16;
        let column_name = "vec";
        let field = ArrowField::new(
            column_name,
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                dimensions,
            ),
            false,
        );

        let schema = Arc::new(ArrowSchema::new(vec![field]));

        let float_arr = generate_random_array(512 * dimensions as usize);
        let vectors =
            arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();

        let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();

        let reader =
            RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());

        let dataset = Dataset::write(
            reader,
            &test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset.validate().await.unwrap();

        assert!(dataset.num_small_files(1024).await > 0);
        assert!(dataset.num_small_files(512).await == 0);
    }

    #[tokio::test]
    async fn test_read_struct_of_dictionary_arrays() {
        let test_uri = TempStrDir::default();

        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "s",
            DataType::Struct(ArrowFields::from(vec![ArrowField::new(
                "d",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            )])),
            true,
        )]));

        let mut batches: Vec<RecordBatch> = Vec::new();
        for _ in 1..2 {
            let mut dict_builder = StringDictionaryBuilder::<Int32Type>::new();
            dict_builder.append("a").unwrap();
            dict_builder.append("b").unwrap();
            dict_builder.append("c").unwrap();
            dict_builder.append("d").unwrap();

            let struct_array = Arc::new(StructArray::from(vec![(
                Arc::new(ArrowField::new(
                    "d",
                    DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                    true,
                )),
                Arc::new(dict_builder.finish()) as ArrayRef,
            )]));

            let batch =
                RecordBatch::try_new(arrow_schema.clone(), vec![struct_array.clone()]).unwrap();
            batches.push(batch);
        }

        let batch_reader =
            RecordBatchIterator::new(batches.clone().into_iter().map(Ok), arrow_schema.clone());
        Dataset::write(batch_reader, &test_uri, Some(WriteParams::default()))
            .await
            .unwrap();

        let result = scan_dataset(&test_uri).await.unwrap();

        assert_eq!(batches, result);
    }

    async fn scan_dataset(uri: &str) -> Result<Vec<RecordBatch>> {
        let results = Dataset::open(uri)
            .await?
            .scan()
            .try_into_stream()
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        Ok(results)
    }

    #[rstest]
    #[tokio::test]
    async fn test_v0_7_5_migration() {
        // We migrate to add Fragment.physical_rows and DeletionFile.num_deletions
        // after this version.

        // Copy over table
        let test_dir = copy_test_data_to_tmp("v0.7.5/with_deletions").unwrap();
        let test_uri = test_dir.path_str();

        // Assert num rows, deletions, and physical rows are all correct.
        let dataset = Dataset::open(&test_uri).await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 90);
        assert_eq!(dataset.count_deleted_rows().await.unwrap(), 10);
        let total_physical_rows = futures::stream::iter(dataset.get_fragments())
            .then(|f| async move { f.physical_rows().await })
            .try_fold(0, |acc, x| async move { Ok(acc + x) })
            .await
            .unwrap();
        assert_eq!(total_physical_rows, 100);

        // Append 5 rows
        let schema = Arc::new(ArrowSchema::from(dataset.schema()));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(100..105))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let write_params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        let dataset = Dataset::write(batches, &test_uri, Some(write_params))
            .await
            .unwrap();

        // Assert num rows, deletions, and physical rows are all correct.
        assert_eq!(dataset.count_rows(None).await.unwrap(), 95);
        assert_eq!(dataset.count_deleted_rows().await.unwrap(), 10);
        let total_physical_rows = futures::stream::iter(dataset.get_fragments())
            .then(|f| async move { f.physical_rows().await })
            .try_fold(0, |acc, x| async move { Ok(acc + x) })
            .await
            .unwrap();
        assert_eq!(total_physical_rows, 105);

        dataset.validate().await.unwrap();

        // Scan data and assert it is as expected.
        let expected = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(
                (0..10).chain(20..105),
            ))],
        )
        .unwrap();
        let actual_batches = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let actual = concat_batches(&actual_batches[0].schema(), &actual_batches).unwrap();
        assert_eq!(actual, expected);
    }

    #[rstest]
    #[tokio::test]
    async fn test_fix_v0_8_0_broken_migration() {
        // The migration from v0.7.5 was broken in 0.8.0. This validates we can
        // automatically fix tables that have this problem.

        // Copy over table
        let test_dir = copy_test_data_to_tmp("v0.8.0/migrated_from_v0.7.5").unwrap();
        let test_uri = test_dir.path_str();
        let test_uri = &test_uri;

        // Assert num rows, deletions, and physical rows are all correct, even
        // though stats are bad.
        let dataset = Dataset::open(test_uri).await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 92);
        assert_eq!(dataset.count_deleted_rows().await.unwrap(), 10);
        let total_physical_rows = futures::stream::iter(dataset.get_fragments())
            .then(|f| async move { f.physical_rows().await })
            .try_fold(0, |acc, x| async move { Ok(acc + x) })
            .await
            .unwrap();
        assert_eq!(total_physical_rows, 102);

        // Append 5 rows to table.
        let schema = Arc::new(ArrowSchema::from(dataset.schema()));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(100..105))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let write_params = WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(LanceFileVersion::Legacy),
            ..Default::default()
        };
        let dataset = Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        // Assert statistics are all now correct.
        let physical_rows: Vec<_> = dataset
            .get_fragments()
            .iter()
            .map(|f| f.metadata.physical_rows)
            .collect();
        assert_eq!(physical_rows, vec![Some(100), Some(2), Some(5)]);
        let num_deletions: Vec<_> = dataset
            .get_fragments()
            .iter()
            .map(|f| {
                f.metadata
                    .deletion_file
                    .as_ref()
                    .and_then(|df| df.num_deleted_rows)
            })
            .collect();
        assert_eq!(num_deletions, vec![Some(10), None, None]);
        assert_eq!(dataset.count_rows(None).await.unwrap(), 97);

        // Scan data and assert it is as expected.
        let expected = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(
                (0..10).chain(20..100).chain(0..2).chain(100..105),
            ))],
        )
        .unwrap();
        let actual_batches = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let actual = concat_batches(&actual_batches[0].schema(), &actual_batches).unwrap();
        assert_eq!(actual, expected);
    }

    #[rstest]
    #[tokio::test]
    async fn test_v0_8_14_invalid_index_fragment_bitmap(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // Old versions of lance could create an index whose fragment bitmap was
        // invalid because it did not include fragments that were part of the index
        //
        // We need to make sure we do not rely on the fragment bitmap in these older
        // versions and instead fall back to a slower legacy behavior
        let test_dir = copy_test_data_to_tmp("v0.8.14/corrupt_index").unwrap();
        let test_uri = test_dir.path_str();
        let test_uri = &test_uri;

        let mut dataset = Dataset::open(test_uri).await.unwrap();

        // Uncomment to reproduce the issue.  The below query will panic
        // let mut scan = dataset.scan();
        // let query_vec = Float32Array::from(vec![0_f32; 128]);
        // let scan_fut = scan
        //     .nearest("vector", &query_vec, 2000)
        //     .unwrap()
        //     .nprobes(4)
        //     .prefilter(true)
        //     .try_into_stream()
        //     .await
        //     .unwrap()
        //     .try_collect::<Vec<_>>()
        //     .await
        //     .unwrap();

        // Add some data and recalculate the index, forcing a migration
        let mut scan = dataset.scan();
        let data = scan
            .limit(Some(10), None)
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let schema = data[0].schema();
        let data = RecordBatchIterator::new(data.into_iter().map(arrow::error::Result::Ok), schema);

        let broken_version = dataset.version().version;

        // Any transaction, no matter how simple, should trigger the fragment bitmap to be recalculated
        dataset
            .append(
                data,
                Some(WriteParams {
                    data_storage_version: Some(data_storage_version),
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

        for idx in dataset.load_indices().await.unwrap().iter() {
            // The corrupt fragment_bitmap does not contain 0 but the
            // restored one should
            assert!(idx.fragment_bitmap.as_ref().unwrap().contains(0));
        }

        let mut dataset = dataset.checkout_version(broken_version).await.unwrap();
        dataset.restore().await.unwrap();

        // Running compaction right away should work (this is verifying compaction
        // is not broken by the potentially malformed fragment bitmaps)
        compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();

        for idx in dataset.load_indices().await.unwrap().iter() {
            assert!(idx.fragment_bitmap.as_ref().unwrap().contains(0));
        }

        let mut scan = dataset.scan();
        let query_vec = Float32Array::from(vec![0_f32; 128]);
        let batches = scan
            .nearest("vector", &query_vec, 2000)
            .unwrap()
            .nprobes(4)
            .prefilter(true)
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let row_count = batches.iter().map(|batch| batch.num_rows()).sum::<usize>();
        assert_eq!(row_count, 1900);
    }

    #[tokio::test]
    async fn test_fix_v0_10_5_corrupt_schema() {
        // Schemas could be corrupted by successive calls to `add_columns` and
        // `drop_columns`. We should be able to detect this by checking for
        // duplicate field ids. We should be able to fix this in new commits
        // by dropping unused data files and re-writing the schema.

        // Copy over table
        let test_dir = copy_test_data_to_tmp("v0.10.5/corrupt_schema").unwrap();
        let test_uri = test_dir.path_str();
        let test_uri = &test_uri;

        let mut dataset = Dataset::open(test_uri).await.unwrap();

        let validate_res = dataset.validate().await;
        assert!(validate_res.is_err());

        // Force a migration.
        dataset.delete("false").await.unwrap();
        dataset.validate().await.unwrap();

        let data = dataset.scan().try_into_batch().await.unwrap();
        assert_eq!(
            data["b"]
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[0, 4, 8, 12]
        );
        assert_eq!(
            data["c"]
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[0, 5, 10, 15]
        );
    }

    #[tokio::test]
    async fn test_fix_v0_21_0_corrupt_fragment_bitmap() {
        // In v0.21.0 and earlier, delta indices had a bug where the fragment bitmap
        // could contain fragments that are part of other index deltas.

        // Copy over table
        let test_dir = copy_test_data_to_tmp("v0.21.0/bad_index_fragment_bitmap").unwrap();
        let test_uri = test_dir.path_str();
        let test_uri = &test_uri;

        let mut dataset = Dataset::open(test_uri).await.unwrap();

        let validate_res = dataset.validate().await;
        assert!(validate_res.is_err());
        assert_eq!(dataset.load_indices().await.unwrap()[0].name, "vector_idx");

        // Calling index statistics will force a migration
        let stats = dataset.index_statistics("vector_idx").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(stats["num_indexed_fragments"], 2);

        dataset.checkout_latest().await.unwrap();
        dataset.validate().await.unwrap();

        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 2);
        fn get_bitmap(meta: &IndexMetadata) -> Vec<u32> {
            meta.fragment_bitmap.as_ref().unwrap().iter().collect()
        }
        assert_eq!(get_bitmap(&indices[0]), vec![0]);
        assert_eq!(get_bitmap(&indices[1]), vec![1]);
    }

    #[tokio::test]
    async fn test_max_fragment_id_migration() {
        // v0.5.9 and earlier did not store the max fragment id in the manifest.
        // This test ensures that we can read such datasets and migrate them to
        // the latest version, which requires the max fragment id to be present.
        {
            let test_dir = copy_test_data_to_tmp("v0.5.9/no_fragments").unwrap();
            let test_uri = test_dir.path_str();
            let test_uri = &test_uri;
            let dataset = Dataset::open(test_uri).await.unwrap();

            assert_eq!(dataset.manifest.max_fragment_id, None);
            assert_eq!(dataset.manifest.max_fragment_id(), None);
        }

        {
            let test_dir = copy_test_data_to_tmp("v0.5.9/dataset_with_fragments").unwrap();
            let test_uri = test_dir.path_str();
            let test_uri = &test_uri;
            let dataset = Dataset::open(test_uri).await.unwrap();

            assert_eq!(dataset.manifest.max_fragment_id, None);
            assert_eq!(dataset.manifest.max_fragment_id(), Some(2));
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_bfloat16_roundtrip(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) -> Result<()> {
        let inner_field = Arc::new(
            ArrowField::new("item", DataType::FixedSizeBinary(2), true).with_metadata(
                [
                    (ARROW_EXT_NAME_KEY.into(), BFLOAT16_EXT_NAME.into()),
                    (ARROW_EXT_META_KEY.into(), "".into()),
                ]
                .into(),
            ),
        );
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "fsl",
            DataType::FixedSizeList(inner_field.clone(), 2),
            false,
        )]));

        let values = bfloat16::BFloat16Array::from_iter_values(
            (0..6).map(|i| i as f32).map(half::bf16::from_f32),
        );
        let vectors = FixedSizeListArray::new(inner_field, 2, Arc::new(values.into_inner()), None);

        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();

        let test_uri = TempStrDir::default();

        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone()),
            &test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await?;

        let data = dataset.scan().try_into_batch().await?;
        assert_eq!(batch, data);

        Ok(())
    }

    #[tokio::test]
    async fn test_overwrite_mixed_version() {
        let test_uri = TempStrDir::default();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let arr = Arc::new(Int32Array::from(vec![1, 2, 3]));

        let data = RecordBatch::try_new(schema.clone(), vec![arr]).unwrap();
        let reader =
            RecordBatchIterator::new(vec![data.clone()].into_iter().map(Ok), schema.clone());

        let dataset = Dataset::write(
            reader,
            &test_uri,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::Legacy),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(
            dataset
                .manifest
                .data_storage_format
                .lance_file_version()
                .unwrap(),
            LanceFileVersion::Legacy
        );

        let reader = RecordBatchIterator::new(vec![data].into_iter().map(Ok), schema);
        let dataset = Dataset::write(
            reader,
            &test_uri,
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(
            dataset
                .manifest
                .data_storage_format
                .lance_file_version()
                .unwrap(),
            LanceFileVersion::Legacy
        );
    }

    // Bug: https://github.com/lancedb/lancedb/issues/1223
    #[tokio::test]
    async fn test_open_nonexisting_dataset() {
        let temp_dir = TempStdDir::default();
        let dataset_dir = temp_dir.join("non_existing");
        let dataset_uri = dataset_dir.to_str().unwrap();

        let res = Dataset::open(dataset_uri).await;
        assert!(res.is_err());

        assert!(!dataset_dir.exists());
    }

    #[tokio::test]
    async fn test_manifest_partially_fits() {
        // This regresses a bug that occurred when the manifest file was over 4KiB but the manifest
        // itself was less than 4KiB (due to a dictionary).  4KiB is important here because that's the
        // block size we use when reading the "last block"

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "x",
            DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8)),
            false,
        )]));
        let dictionary = Arc::new(StringArray::from_iter_values(
            (0..1000).map(|i| i.to_string()),
        ));
        let indices = Int16Array::from_iter_values(0..1000);
        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(
                Int16DictionaryArray::try_new(indices, dictionary.clone()).unwrap(),
            )],
        )
        .unwrap()];

        let test_uri = TempStrDir::default();
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, &test_uri, None).await.unwrap();

        let dataset = Dataset::open(&test_uri).await.unwrap();
        assert_eq!(1000, dataset.count_rows(None).await.unwrap());
    }

    #[tokio::test]
    async fn test_dataset_uri_roundtrips() {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::Int32,
            false,
        )]));

        let test_uri = TempStrDir::default();
        let vectors = Arc::new(Int32Array::from_iter_values(vec![]));

        let data = RecordBatch::try_new(schema.clone(), vec![vectors]);
        let reader = RecordBatchIterator::new(vec![data.unwrap()].into_iter().map(Ok), schema);
        let dataset = Dataset::write(
            reader,
            &test_uri,
            Some(WriteParams {
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let uri = dataset.uri();
        assert_eq!(uri, test_uri.as_str());

        let ds2 = Dataset::open(uri).await.unwrap();
        assert_eq!(
            ds2.latest_version_id().await.unwrap(),
            dataset.latest_version_id().await.unwrap()
        );
    }

    #[tokio::test]
    async fn test_fts_fuzzy_query() {
        let params = InvertedIndexParams::default();
        let text_col = GenericStringArray::<i32>::from(vec![
            "fa", "fo", "fob", "focus", "foo", "food", "foul", // # spellchecker:disable-line
        ]);
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![arrow_schema::Field::new(
                "text",
                text_col.data_type().to_owned(),
                false,
            )])
            .into(),
            vec![Arc::new(text_col) as ArrayRef],
        )
        .unwrap();
        let schema = batch.schema();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        let test_uri = TempStrDir::default();
        let mut dataset = Dataset::write(batches, &test_uri, None).await.unwrap();
        dataset
            .create_index(&["text"], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();
        let results = dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new_fuzzy("foo".to_owned(), Some(1)))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 4);
        let texts = results["text"]
            .as_string::<i32>()
            .iter()
            .map(|s| s.unwrap().to_owned())
            .collect::<HashSet<_>>();
        assert_eq!(
            texts,
            vec![
                "foo".to_owned(),  // 0 edits
                "fo".to_owned(),   // 1 deletion        # spellchecker:disable-line
                "fob".to_owned(),  // 1 substitution    # spellchecker:disable-line
                "food".to_owned(), // 1 insertion       # spellchecker:disable-line
            ]
            .into_iter()
            .collect()
        );
    }

    #[tokio::test]
    async fn test_fts_on_multiple_columns() {
        let params = InvertedIndexParams::default();
        let title_col =
            GenericStringArray::<i32>::from(vec!["title common", "title hello", "title lance"]);
        let content_col = GenericStringArray::<i32>::from(vec![
            "content world",
            "content database",
            "content common",
        ]);
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![
                arrow_schema::Field::new("title", title_col.data_type().to_owned(), false),
                arrow_schema::Field::new("content", title_col.data_type().to_owned(), false),
            ])
            .into(),
            vec![
                Arc::new(title_col) as ArrayRef,
                Arc::new(content_col) as ArrayRef,
            ],
        )
        .unwrap();
        let schema = batch.schema();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        let test_uri = TempStrDir::default();
        let mut dataset = Dataset::write(batches, &test_uri, None).await.unwrap();
        dataset
            .create_index(&["title"], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();
        dataset
            .create_index(&["content"], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();

        let results = dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new("title".to_owned()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 3);

        let results = dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new("content".to_owned()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 3);

        let results = dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new("common".to_owned()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 2);

        let results = dataset
            .scan()
            .full_text_search(
                FullTextSearchQuery::new("common".to_owned())
                    .with_column("title".to_owned())
                    .unwrap(),
            )
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 1);

        let results = dataset
            .scan()
            .full_text_search(
                FullTextSearchQuery::new("common".to_owned())
                    .with_column("content".to_owned())
                    .unwrap(),
            )
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 1);
    }

    #[tokio::test]
    async fn test_fts_unindexed_data() {
        let params = InvertedIndexParams::default();
        let title_col = StringArray::from(vec!["title hello", "title lance", "title common"]);
        let content_col =
            StringArray::from(vec!["content world", "content database", "content common"]);
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![
                Field::new("title", title_col.data_type().to_owned(), false),
                Field::new("content", title_col.data_type().to_owned(), false),
            ])
            .into(),
            vec![
                Arc::new(title_col) as ArrayRef,
                Arc::new(content_col) as ArrayRef,
            ],
        )
        .unwrap();
        let schema = batch.schema();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(batches, "memory://test.lance", None)
            .await
            .unwrap();
        dataset
            .create_index(&["title"], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();

        let results = dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new("title".to_owned()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 3);

        // write new data
        let title_col = StringArray::from(vec!["new title"]);
        let content_col = StringArray::from(vec!["new content"]);
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![
                Field::new("title", title_col.data_type().to_owned(), false),
                Field::new("content", title_col.data_type().to_owned(), false),
            ])
            .into(),
            vec![
                Arc::new(title_col) as ArrayRef,
                Arc::new(content_col) as ArrayRef,
            ],
        )
        .unwrap();
        let schema = batch.schema();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        dataset.append(batches, None).await.unwrap();

        let results = dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new("title".to_owned()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 4);

        let results = dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new("new".to_owned()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 1);
    }

    #[tokio::test]
    async fn test_fts_unindexed_data_on_empty_index() {
        // Empty dataset with fts index
        let params = InvertedIndexParams::default();
        let title_col = StringArray::from(Vec::<&str>::new());
        let content_col = StringArray::from(Vec::<&str>::new());
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![
                Field::new("title", title_col.data_type().to_owned(), false),
                Field::new("content", title_col.data_type().to_owned(), false),
            ])
            .into(),
            vec![
                Arc::new(title_col) as ArrayRef,
                Arc::new(content_col) as ArrayRef,
            ],
        )
        .unwrap();
        let schema = batch.schema();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(batches, "memory://test.lance", None)
            .await
            .unwrap();
        dataset
            .create_index(&["title"], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();

        // Test fts search
        let results = dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Match(
                MatchQuery::new("title".to_owned()).with_column(Some("title".to_owned())),
            )))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 0);

        // write new data
        let title_col = StringArray::from(vec!["title hello", "title lance", "title common"]);
        let content_col =
            StringArray::from(vec!["content world", "content database", "content common"]);
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![
                Field::new("title", title_col.data_type().to_owned(), false),
                Field::new("content", title_col.data_type().to_owned(), false),
            ])
            .into(),
            vec![
                Arc::new(title_col) as ArrayRef,
                Arc::new(content_col) as ArrayRef,
            ],
        )
        .unwrap();
        let schema = batch.schema();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        dataset.append(batches, None).await.unwrap();

        let results = dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Match(
                MatchQuery::new("title".to_owned()).with_column(Some("title".to_owned())),
            )))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 3);
    }

    #[tokio::test]
    async fn test_fts_without_index() {
        // create table without index
        let title_col = StringArray::from(vec!["title hello", "title lance", "title common"]);
        let content_col =
            StringArray::from(vec!["content world", "content database", "content common"]);
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![
                Field::new("title", title_col.data_type().to_owned(), false),
                Field::new("content", title_col.data_type().to_owned(), false),
            ])
            .into(),
            vec![
                Arc::new(title_col) as ArrayRef,
                Arc::new(content_col) as ArrayRef,
            ],
        )
        .unwrap();
        let schema = batch.schema();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(batches, "memory://test.lance", None)
            .await
            .unwrap();

        // match query on title and content
        let results = dataset
            .scan()
            .full_text_search(
                FullTextSearchQuery::new("title".to_owned())
                    .with_columns(&["title".to_string(), "content".to_string()])
                    .unwrap(),
            )
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 3);

        // write new data
        let title_col = StringArray::from(vec!["new title"]);
        let content_col = StringArray::from(vec!["new content"]);
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![
                Field::new("title", title_col.data_type().to_owned(), false),
                Field::new("content", title_col.data_type().to_owned(), false),
            ])
            .into(),
            vec![
                Arc::new(title_col) as ArrayRef,
                Arc::new(content_col) as ArrayRef,
            ],
        )
        .unwrap();
        let schema = batch.schema();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        dataset.append(batches, None).await.unwrap();

        // match query on title and content
        let results = dataset
            .scan()
            .full_text_search(
                FullTextSearchQuery::new("title".to_owned())
                    .with_columns(&["title".to_string(), "content".to_string()])
                    .unwrap(),
            )
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 4);

        let results = dataset
            .scan()
            .full_text_search(
                FullTextSearchQuery::new("new".to_owned())
                    .with_columns(&["title".to_string(), "content".to_string()])
                    .unwrap(),
            )
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 1);
    }

    #[tokio::test]
    async fn test_fts_rank() {
        let params = InvertedIndexParams::default();
        let text_col =
            GenericStringArray::<i32>::from(vec!["score", "find score", "try to find score"]);
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![arrow_schema::Field::new(
                "text",
                text_col.data_type().to_owned(),
                false,
            )])
            .into(),
            vec![Arc::new(text_col) as ArrayRef],
        )
        .unwrap();
        let schema = batch.schema();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        let test_uri = TempStrDir::default();
        let mut dataset = Dataset::write(batches, &test_uri, None).await.unwrap();
        dataset
            .create_index(&["text"], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();

        let results = dataset
            .scan()
            .with_row_id()
            .full_text_search(FullTextSearchQuery::new("score".to_owned()))
            .unwrap()
            .limit(Some(3), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 3);
        let row_ids = results[ROW_ID].as_primitive::<UInt64Type>().values();
        assert_eq!(row_ids, &[0, 1, 2]);

        let results = dataset
            .scan()
            .with_row_id()
            .full_text_search(FullTextSearchQuery::new("score".to_owned()))
            .unwrap()
            .limit(Some(2), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 2);
        let row_ids = results[ROW_ID].as_primitive::<UInt64Type>().values();
        assert_eq!(row_ids, &[0, 1]);

        let results = dataset
            .scan()
            .with_row_id()
            .full_text_search(FullTextSearchQuery::new("score".to_owned()))
            .unwrap()
            .limit(Some(1), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 1);
        let row_ids = results[ROW_ID].as_primitive::<UInt64Type>().values();
        assert_eq!(row_ids, &[0]);
    }

    async fn create_fts_dataset<
        Offset: arrow::array::OffsetSizeTrait,
        ListOffset: arrow::array::OffsetSizeTrait,
    >(
        is_list: bool,
        with_position: bool,
        params: InvertedIndexParams,
    ) -> Dataset {
        let tempdir = TempStrDir::default();
        let uri = tempdir.to_owned();
        drop(tempdir);

        let params = params.with_position(with_position);
        let doc_col: Arc<dyn Array> = if is_list {
            let string_builder = GenericStringBuilder::<Offset>::new();
            let mut list_col = GenericListBuilder::<ListOffset, _>::new(string_builder);
            // Create a list of strings
            list_col.values().append_value("lance database the search"); // for testing phrase query
            list_col.append(true);
            list_col.values().append_value("lance database"); // for testing phrase query
            list_col.append(true);
            list_col.values().append_value("lance search");
            list_col.append(true);
            list_col.values().append_value("database");
            list_col.values().append_value("search");
            list_col.append(true);
            list_col.values().append_value("unrelated doc");
            list_col.append(true);
            list_col.values().append_value("unrelated");
            list_col.append(true);
            list_col.values().append_value("mots");
            list_col.values().append_value("accentués");
            list_col.append(true);
            list_col
                .values()
                .append_value("lance database full text search");
            list_col.append(true);

            // for testing null
            list_col.append(false);

            Arc::new(list_col.finish())
        } else {
            Arc::new(GenericStringArray::<Offset>::from(vec![
                "lance database the search",
                "lance database",
                "lance search",
                "database search",
                "unrelated doc",
                "unrelated",
                "mots accentués",
                "lance database full text search",
            ]))
        };
        let ids = UInt64Array::from_iter_values(0..doc_col.len() as u64);
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![
                arrow_schema::Field::new("doc", doc_col.data_type().to_owned(), true),
                arrow_schema::Field::new("id", DataType::UInt64, false),
            ])
            .into(),
            vec![Arc::new(doc_col) as ArrayRef, Arc::new(ids) as ArrayRef],
        )
        .unwrap();
        let schema = batch.schema();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(batches, &uri, None).await.unwrap();

        dataset
            .create_index(&["doc"], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();

        dataset
    }

    async fn test_fts_index<
        Offset: arrow::array::OffsetSizeTrait,
        ListOffset: arrow::array::OffsetSizeTrait,
    >(
        is_list: bool,
    ) {
        let ds = create_fts_dataset::<Offset, ListOffset>(
            is_list,
            false,
            InvertedIndexParams::default(),
        )
        .await;
        let result = ds
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new("lance".to_owned()).limit(Some(3)))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 3, "{:?}", result);
        let ids = result["id"].as_primitive::<UInt64Type>().values();
        assert!(ids.contains(&0), "{:?}", result);
        assert!(ids.contains(&1), "{:?}", result);
        assert!(ids.contains(&2), "{:?}", result);

        let result = ds
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new("database".to_owned()).limit(Some(3)))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 3);
        let ids = result["id"].as_primitive::<UInt64Type>().values();
        assert!(ids.contains(&0), "{:?}", result);
        assert!(ids.contains(&1), "{:?}", result);
        assert!(ids.contains(&3), "{:?}", result);

        let result = ds
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(
                FullTextSearchQuery::new_query(
                    MatchQuery::new("lance database".to_owned())
                        .with_operator(Operator::And)
                        .into(),
                )
                .limit(Some(5)),
            )
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 3, "{:?}", result);
        let ids = result["id"].as_primitive::<UInt64Type>().values();
        assert!(ids.contains(&0), "{:?}", result);
        assert!(ids.contains(&1), "{:?}", result);
        assert!(ids.contains(&7), "{:?}", result);

        let result = ds
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new("unknown null".to_owned()).limit(Some(3)))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 0);

        // test phrase query
        // for non-phrasal query, the order of the tokens doesn't matter
        // so there should be 4 documents that contain "database" or "lance"

        // we built the index without position, so the phrase query will not work
        let result = ds
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(
                FullTextSearchQuery::new_query(
                    PhraseQuery::new("lance database".to_owned()).into(),
                )
                .limit(Some(10)),
            )
            .unwrap()
            .try_into_batch()
            .await;
        let err = result.unwrap_err().to_string();
        assert!(err.contains("position is not found but required for phrase queries, try recreating the index with position"),"{}",err);

        // recreate the index with position
        let ds =
            create_fts_dataset::<Offset, ListOffset>(is_list, true, InvertedIndexParams::default())
                .await;
        let result = ds
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new("lance database".to_owned()).limit(Some(10)))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 5, "{:?}", result);
        let ids = result["id"].as_primitive::<UInt64Type>().values();
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
        assert!(ids.contains(&7));

        let result = ds
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(
                FullTextSearchQuery::new_query(
                    PhraseQuery::new("lance database".to_owned()).into(),
                )
                .limit(Some(10)),
            )
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let ids = result["id"].as_primitive::<UInt64Type>().values();
        assert_eq!(result.num_rows(), 3, "{:?}", ids);
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
        assert!(ids.contains(&7));

        let result = ds
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(
                FullTextSearchQuery::new_query(
                    PhraseQuery::new("database lance".to_owned()).into(),
                )
                .limit(Some(10)),
            )
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 0);

        let result = ds
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(
                FullTextSearchQuery::new_query(PhraseQuery::new("lance unknown".to_owned()).into())
                    .limit(Some(10)),
            )
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 0);

        let result = ds
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(
                FullTextSearchQuery::new_query(PhraseQuery::new("unknown null".to_owned()).into())
                    .limit(Some(3)),
            )
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 0);

        let result = ds
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(
                FullTextSearchQuery::new_query(PhraseQuery::new("lance search".to_owned()).into())
                    .limit(Some(3)),
            )
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 1);

        let result = ds
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(
                FullTextSearchQuery::new_query(
                    PhraseQuery::new("lance search".to_owned())
                        .with_slop(2)
                        .into(),
                )
                .limit(Some(3)),
            )
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 2);

        let result = ds
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(
                FullTextSearchQuery::new_query(
                    PhraseQuery::new("search lance".to_owned())
                        .with_slop(2)
                        .into(),
                )
                .limit(Some(3)),
            )
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 0);

        let result = ds
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(
                // must contain "lance" and "database", and may contain "search"
                FullTextSearchQuery::new_query(
                    BooleanQuery::new([
                        (
                            Occur::Should,
                            MatchQuery::new("search".to_owned())
                                .with_operator(Operator::And)
                                .into(),
                        ),
                        (
                            Occur::Must,
                            MatchQuery::new("lance database".to_owned())
                                .with_operator(Operator::And)
                                .into(),
                        ),
                    ])
                    .into(),
                )
                .limit(Some(3)),
            )
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 3, "{:?}", result);
        let ids = result["id"].as_primitive::<UInt64Type>().values();
        assert!(ids.contains(&0), "{:?}", result);
        assert!(ids.contains(&1), "{:?}", result);
        assert!(ids.contains(&7), "{:?}", result);

        let result = ds
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(
                // must contain "lance" and "database", and may contain "search"
                FullTextSearchQuery::new_query(
                    BooleanQuery::new([
                        (
                            Occur::Should,
                            MatchQuery::new("search".to_owned())
                                .with_operator(Operator::And)
                                .into(),
                        ),
                        (
                            Occur::Must,
                            MatchQuery::new("lance database".to_owned())
                                .with_operator(Operator::And)
                                .into(),
                        ),
                        (
                            Occur::MustNot,
                            MatchQuery::new("full text".to_owned()).into(),
                        ),
                    ])
                    .into(),
                )
                .limit(Some(3)),
            )
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 2, "{:?}", result);
        let ids = result["id"].as_primitive::<UInt64Type>().values();
        assert!(ids.contains(&0), "{:?}", result);
        assert!(ids.contains(&1), "{:?}", result);
    }

    #[tokio::test]
    async fn test_fts_index_with_string() {
        test_fts_index::<i32, i32>(false).await;
        test_fts_index::<i32, i32>(true).await;
        test_fts_index::<i32, i64>(true).await;
    }

    #[tokio::test]
    async fn test_fts_index_with_large_string() {
        test_fts_index::<i64, i32>(false).await;
        test_fts_index::<i64, i32>(true).await;
        test_fts_index::<i64, i64>(true).await;
    }

    #[tokio::test]
    async fn test_fts_accented_chars() {
        let ds = create_fts_dataset::<i32, i32>(false, false, InvertedIndexParams::default()).await;
        let result = ds
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new("accentués".to_owned()).limit(Some(3)))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 1);

        let result = ds
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new("accentues".to_owned()).limit(Some(3)))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 0);

        // with ascii folding enabled, the search should be accent-insensitive
        let ds = create_fts_dataset::<i32, i32>(
            false,
            false,
            InvertedIndexParams::default()
                .stem(false)
                .ascii_folding(true),
        )
        .await;
        let result = ds
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new("accentués".to_owned()).limit(Some(3)))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 1);

        let result = ds
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new("accentues".to_owned()).limit(Some(3)))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 1);
    }

    #[tokio::test]
    async fn test_fts_phrase_query() {
        let tmpdir = TempStrDir::default();
        let uri = tmpdir.to_owned();
        drop(tmpdir);

        let words = ["lance", "full", "text", "search"];
        let mut lance_search_count = 0;
        let mut full_text_count = 0;
        let mut doc_array = (0..4096)
            .map(|_| {
                let mut rng = rand::rng();
                let mut text = String::with_capacity(512);
                let len = rng.random_range(127..512);
                for i in 0..len {
                    if i > 0 {
                        text.push(' ');
                    }
                    text.push_str(words[rng.random_range(0..words.len())]);
                }
                if text.contains("lance search") {
                    lance_search_count += 1;
                }
                if text.contains("full text") {
                    full_text_count += 1;
                }
                text
            })
            .collect_vec();
        // Ensure at least one doc matches each phrase deterministically
        doc_array.push("lance search".to_owned());
        lance_search_count += 1;
        doc_array.push("full text".to_owned());
        full_text_count += 1;
        doc_array.push("position for phrase query".to_owned());

        // 1) Build index without positions and assert phrase query errors
        let params_no_pos = InvertedIndexParams::default().with_position(false);
        let doc_col: Arc<dyn Array> = Arc::new(GenericStringArray::<i32>::from(doc_array.clone()));
        let ids = UInt64Array::from_iter_values(0..doc_col.len() as u64);
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![
                arrow_schema::Field::new("doc", doc_col.data_type().to_owned(), true),
                arrow_schema::Field::new("id", DataType::UInt64, false),
            ])
            .into(),
            vec![Arc::new(doc_col) as ArrayRef, Arc::new(ids) as ArrayRef],
        )
        .unwrap();
        let schema = batch.schema();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(batches, &uri, None).await.unwrap();
        dataset
            .create_index(&["doc"], IndexType::Inverted, None, &params_no_pos, true)
            .await
            .unwrap();

        let err = dataset
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new_query(
                PhraseQuery::new("lance search".to_owned()).into(),
            ))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("position is not found but required for phrase queries, try recreating the index with position"), "{}", err);
        assert!(err.starts_with("Invalid user input: "), "{}", err);

        // 2) Recreate index with positions and assert phrase query works
        let params_with_pos = InvertedIndexParams::default().with_position(true);
        dataset
            .create_index(&["doc"], IndexType::Inverted, None, &params_with_pos, true)
            .await
            .unwrap();

        let result = dataset
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new_query(
                PhraseQuery::new("lance search".to_owned()).into(),
            ))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), lance_search_count);

        let result = dataset
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new_query(
                PhraseQuery::new("full text".to_owned()).into(),
            ))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), full_text_count);

        let result = dataset
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new_query(
                PhraseQuery::new("phrase query".to_owned()).into(),
            ))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 1);

        let result = dataset
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new_query(
                PhraseQuery::new("".to_owned()).into(),
            ))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 0);
    }

    #[tokio::test]
    async fn concurrent_create() {
        async fn write(uri: &str) -> Result<()> {
            let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "a",
                DataType::Int32,
                false,
            )]));
            let empty_reader = RecordBatchIterator::new(vec![], schema.clone());
            Dataset::write(empty_reader, uri, None).await?;
            Ok(())
        }

        for _ in 0..5 {
            let test_uri = TempStrDir::default();

            let (res1, res2) = tokio::join!(write(&test_uri), write(&test_uri));

            assert!(res1.is_ok() || res2.is_ok());
            if res1.is_err() {
                assert!(
                    matches!(res1, Err(Error::DatasetAlreadyExists { .. })),
                    "{:?}",
                    res1
                );
            } else if res2.is_err() {
                assert!(
                    matches!(res2, Err(Error::DatasetAlreadyExists { .. })),
                    "{:?}",
                    res2
                );
            } else {
                assert!(res1.is_ok() && res2.is_ok());
            }
        }
    }

    #[tokio::test]
    async fn test_read_transaction_properties() {
        const LANCE_COMMIT_MESSAGE_KEY: &str = "__lance_commit_message";
        // Create a test dataset
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("value", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();

        let test_uri = TempStrDir::default();

        // Create WriteParams with properties
        let mut properties1 = HashMap::new();
        properties1.insert(
            LANCE_COMMIT_MESSAGE_KEY.to_string(),
            "First commit".to_string(),
        );
        properties1.insert("custom_prop".to_string(), "custom_value".to_string());

        let write_params = WriteParams {
            transaction_properties: Some(Arc::new(properties1)),
            ..Default::default()
        };

        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], schema.clone()),
            &test_uri,
            Some(write_params),
        )
        .await
        .unwrap();

        let transaction = dataset.read_transaction_by_version(1).await.unwrap();
        assert!(transaction.is_some());
        let props = transaction.unwrap().transaction_properties.unwrap();
        assert_eq!(props.len(), 2);
        assert_eq!(
            props.get(LANCE_COMMIT_MESSAGE_KEY),
            Some(&"First commit".to_string())
        );
        assert_eq!(props.get("custom_prop"), Some(&"custom_value".to_string()));

        let mut properties2 = HashMap::new();
        properties2.insert(
            LANCE_COMMIT_MESSAGE_KEY.to_string(),
            "Second commit".to_string(),
        );
        properties2.insert("another_prop".to_string(), "another_value".to_string());

        let write_params = WriteParams {
            transaction_properties: Some(Arc::new(properties2)),
            mode: WriteMode::Append,
            ..Default::default()
        };

        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![4, 5])),
                Arc::new(StringArray::from(vec!["d", "e"])),
            ],
        )
        .unwrap();

        let mut dataset = dataset;
        dataset
            .append(
                RecordBatchIterator::new([Ok(batch2)], schema.clone()),
                Some(write_params),
            )
            .await
            .unwrap();

        let transaction = dataset.read_transaction_by_version(2).await.unwrap();
        assert!(transaction.is_some());
        let props = transaction.unwrap().transaction_properties.unwrap();
        assert_eq!(props.len(), 2);
        assert_eq!(
            props.get(LANCE_COMMIT_MESSAGE_KEY),
            Some(&"Second commit".to_string())
        );
        assert_eq!(
            props.get("another_prop"),
            Some(&"another_value".to_string())
        );

        let transaction = dataset.read_transaction_by_version(1).await.unwrap();
        assert!(transaction.is_some());
        let props = transaction.unwrap().transaction_properties.unwrap();
        assert_eq!(props.len(), 2);
        assert_eq!(
            props.get(LANCE_COMMIT_MESSAGE_KEY),
            Some(&"First commit".to_string())
        );
        assert_eq!(props.get("custom_prop"), Some(&"custom_value".to_string()));

        let result = dataset.read_transaction_by_version(999).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_insert_subschema() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("b", DataType::Int32, true),
        ]));
        let empty_reader = RecordBatchIterator::new(vec![], schema.clone());
        let mut dataset = Dataset::write(empty_reader, "memory://", None)
            .await
            .unwrap();
        dataset.validate().await.unwrap();

        // If missing columns that aren't nullable, will return an error
        // TODO: provide alternative default than null.
        let just_b = Arc::new(schema.project(&[1]).unwrap());
        let batch = RecordBatch::try_new(just_b.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
            .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], just_b.clone());
        let res = dataset.append(reader, None).await;
        assert!(
            matches!(res, Err(Error::SchemaMismatch { .. })),
            "Expected Error::SchemaMismatch, got {:?}",
            res
        );

        // If missing columns that are nullable, the write succeeds.
        let just_a = Arc::new(schema.project(&[0]).unwrap());
        let batch = RecordBatch::try_new(just_a.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
            .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], just_a.clone());
        dataset.append(reader, None).await.unwrap();
        dataset.validate().await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 1);

        // Looking at the fragments, there is no data file with the missing field
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].metadata.files.len(), 1);
        assert_eq!(&fragments[0].metadata.files[0].fields, &[0]);

        // When reading back, columns that are missing are null
        let data = dataset.scan().try_into_batch().await.unwrap();
        let expected = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Int32Array::from(vec![None])),
            ],
        )
        .unwrap();
        assert_eq!(data, expected);

        // Can still insert all columns
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![2])),
                Arc::new(Int32Array::from(vec![3])),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone());
        dataset.append(reader, None).await.unwrap();
        dataset.validate().await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 2);

        // When reading back, only missing data is null, otherwise is filled in
        let data = dataset.scan().try_into_batch().await.unwrap();
        let expected = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(Int32Array::from(vec![None, Some(3)])),
            ],
        )
        .unwrap();
        assert_eq!(data, expected);

        // Can run compaction. All files should now have all fields.
        compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();
        dataset.validate().await.unwrap();
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].metadata.files.len(), 1);
        assert_eq!(&fragments[0].metadata.files[0].fields, &[0, 1]);

        // Can scan and get expected data.
        let data = dataset.scan().try_into_batch().await.unwrap();
        assert_eq!(data, expected);
    }

    #[tokio::test]
    async fn test_insert_nested_subschemas() {
        // Test subschemas at struct level
        // Test different orders
        // Test the Dataset::write() path
        // Test Take across fragments with different field id sets
        let test_uri = TempStrDir::default();

        let field_a = Arc::new(ArrowField::new("a", DataType::Int32, true));
        let field_b = Arc::new(ArrowField::new("b", DataType::Int32, false));
        let field_c = Arc::new(ArrowField::new("c", DataType::Int32, true));
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "s",
            DataType::Struct(vec![field_a.clone(), field_b.clone(), field_c.clone()].into()),
            true,
        )]));
        let empty_reader = RecordBatchIterator::new(vec![], schema.clone());
        let dataset = Dataset::write(empty_reader, &test_uri, None).await.unwrap();
        dataset.validate().await.unwrap();

        let append_options = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        // Can insert b, a
        let just_b_a = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "s",
            DataType::Struct(vec![field_b.clone(), field_a.clone()].into()),
            true,
        )]));
        let batch = RecordBatch::try_new(
            just_b_a.clone(),
            vec![Arc::new(StructArray::from(vec![
                (
                    field_b.clone(),
                    Arc::new(Int32Array::from(vec![1])) as ArrayRef,
                ),
                (field_a.clone(), Arc::new(Int32Array::from(vec![2]))),
            ]))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], just_b_a.clone());
        let dataset = Dataset::write(reader, &test_uri, Some(append_options.clone()))
            .await
            .unwrap();
        dataset.validate().await.unwrap();
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].metadata.files.len(), 1);
        assert_eq!(&fragments[0].metadata.files[0].fields, &[0, 2, 1]);
        assert_eq!(&fragments[0].metadata.files[0].column_indices, &[0, 1, 2]);

        // Can insert c, b
        let just_c_b = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "s",
            DataType::Struct(vec![field_c.clone(), field_b.clone()].into()),
            true,
        )]));
        let batch = RecordBatch::try_new(
            just_c_b.clone(),
            vec![Arc::new(StructArray::from(vec![
                (
                    field_c.clone(),
                    Arc::new(Int32Array::from(vec![4])) as ArrayRef,
                ),
                (field_b.clone(), Arc::new(Int32Array::from(vec![3]))),
            ]))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], just_c_b.clone());
        let dataset = Dataset::write(reader, &test_uri, Some(append_options.clone()))
            .await
            .unwrap();
        dataset.validate().await.unwrap();
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        assert_eq!(fragments[1].metadata.files.len(), 1);
        assert_eq!(&fragments[1].metadata.files[0].fields, &[0, 3, 2]);
        assert_eq!(&fragments[1].metadata.files[0].column_indices, &[0, 1, 2]);

        // Can't insert a, c (b is non-nullable)
        let just_a_c = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "s",
            DataType::Struct(vec![field_a.clone(), field_c.clone()].into()),
            true,
        )]));
        let batch = RecordBatch::try_new(
            just_a_c.clone(),
            vec![Arc::new(StructArray::from(vec![
                (
                    field_a.clone(),
                    Arc::new(Int32Array::from(vec![5])) as ArrayRef,
                ),
                (field_c.clone(), Arc::new(Int32Array::from(vec![6]))),
            ]))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], just_a_c.clone());
        let res = Dataset::write(reader, &test_uri, Some(append_options)).await;
        assert!(
            matches!(res, Err(Error::SchemaMismatch { .. })),
            "Expected Error::SchemaMismatch, got {:?}",
            res
        );

        // Can scan and get all data
        let data = dataset.scan().try_into_batch().await.unwrap();
        let expected = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StructArray::from(vec![
                (
                    field_a.clone(),
                    Arc::new(Int32Array::from(vec![Some(2), None])) as ArrayRef,
                ),
                (field_b.clone(), Arc::new(Int32Array::from(vec![1, 3]))),
                (
                    field_c.clone(),
                    Arc::new(Int32Array::from(vec![None, Some(4)])),
                ),
            ]))],
        )
        .unwrap();
        assert_eq!(data, expected);

        // Can call take and get rows from all three back in one batch
        let result = dataset
            .take(&[1, 0], Arc::new(dataset.schema().clone()))
            .await
            .unwrap();
        let expected = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StructArray::from(vec![
                (
                    field_a.clone(),
                    Arc::new(Int32Array::from(vec![None, Some(2)])) as ArrayRef,
                ),
                (field_b.clone(), Arc::new(Int32Array::from(vec![3, 1]))),
                (
                    field_c.clone(),
                    Arc::new(Int32Array::from(vec![Some(4), None])),
                ),
            ]))],
        )
        .unwrap();
        assert_eq!(result, expected);
    }

    #[tokio::test]
    async fn test_insert_balanced_subschemas() {
        let test_uri = TempStrDir::default();

        let field_a = ArrowField::new("a", DataType::Int32, true);
        let field_b = ArrowField::new("b", DataType::LargeBinary, true);
        let schema = Arc::new(ArrowSchema::new(vec![
            field_a.clone(),
            field_b
                .clone()
                .with_metadata([(BLOB_META_KEY.to_string(), "true".to_string())].into()),
        ]));
        let empty_reader = RecordBatchIterator::new(vec![], schema.clone());
        let options = WriteParams {
            enable_stable_row_ids: true,
            enable_v2_manifest_paths: true,
            ..Default::default()
        };
        let mut dataset = Dataset::write(empty_reader, &test_uri, Some(options))
            .await
            .unwrap();
        dataset.validate().await.unwrap();

        // Insert left side
        let just_a = Arc::new(ArrowSchema::new(vec![field_a.clone()]));
        let batch = RecordBatch::try_new(just_a.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
            .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], just_a.clone());
        dataset.append(reader, None).await.unwrap();
        dataset.validate().await.unwrap();

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].metadata.files.len(), 1);
        assert_eq!(&fragments[0].metadata.files[0].fields, &[0]);

        // Insert right side
        let just_b = Arc::new(ArrowSchema::new(vec![field_b.clone()]));
        let batch = RecordBatch::try_new(
            just_b.clone(),
            vec![Arc::new(LargeBinaryArray::from_iter(vec![Some(vec![2u8])]))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], just_b.clone());
        dataset.append(reader, None).await.unwrap();
        dataset.validate().await.unwrap();

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        assert_eq!(fragments[1].metadata.files.len(), 1);
        assert_eq!(&fragments[1].metadata.files[0].fields, &[1]);

        let data = dataset
            .take(
                &[0, 1],
                ProjectionRequest::from_columns(["a"], dataset.schema()),
            )
            .await
            .unwrap();
        assert_eq!(data.num_rows(), 2);
        let a_column = data.column(0).as_primitive::<Int32Type>();
        assert_eq!(a_column.value(0), 1);
        assert!(a_column.is_null(1));

        let blob_batch = dataset
            .take(
                &[0, 1],
                ProjectionRequest::from_columns(["b"], dataset.schema()),
            )
            .await
            .unwrap();
        let blob_descriptions = blob_batch.column(0).as_struct();
        assert!(blob_descriptions.is_null(0));
        assert!(blob_descriptions.is_valid(1));
    }

    #[tokio::test]
    async fn test_datafile_replacement() {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::Int32,
            true,
        )]));
        let empty_reader = RecordBatchIterator::new(vec![], schema.clone());
        let dataset = Arc::new(
            Dataset::write(empty_reader, "memory://", None)
                .await
                .unwrap(),
        );
        dataset.validate().await.unwrap();

        // Test empty replacement should commit a new manifest and do nothing
        let mut dataset = Dataset::commit(
            WriteDestination::Dataset(dataset.clone()),
            Operation::DataReplacement {
                replacements: vec![],
            },
            Some(1),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await
        .unwrap();
        dataset.validate().await.unwrap();

        assert_eq!(dataset.version().version, 2);
        assert_eq!(dataset.get_fragments().len(), 0);

        // try the same thing on a non-empty dataset
        let vals: Int32Array = vec![1, 2, 3].into();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vals)]).unwrap();
        dataset
            .append(
                RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
                None,
            )
            .await
            .unwrap();

        let dataset = Dataset::commit(
            WriteDestination::Dataset(Arc::new(dataset)),
            Operation::DataReplacement {
                replacements: vec![],
            },
            Some(3),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await
        .unwrap();
        dataset.validate().await.unwrap();

        assert_eq!(dataset.version().version, 4);
        assert_eq!(dataset.get_fragments().len(), 1);

        let batch = dataset.scan().try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values(),
            &[1, 2, 3]
        );

        // write a new datafile
        let object_writer = dataset
            .object_store
            .create(&Path::from("data/test.lance"))
            .await
            .unwrap();
        let mut writer = FileWriter::try_new(
            object_writer,
            schema.as_ref().try_into().unwrap(),
            Default::default(),
        )
        .unwrap();

        let vals: Int32Array = vec![4, 5, 6].into();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vals)]).unwrap();
        writer.write_batch(&batch).await.unwrap();
        writer.finish().await.unwrap();

        // find the datafile we want to replace
        let frag = dataset.get_fragment(0).unwrap();
        let data_file = frag.data_file_for_field(0).unwrap();
        let mut new_data_file = data_file.clone();
        new_data_file.path = "test.lance".to_string();

        let dataset = Dataset::commit(
            WriteDestination::Dataset(Arc::new(dataset)),
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, new_data_file)],
            },
            Some(4),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await
        .unwrap();

        assert_eq!(dataset.version().version, 5);
        assert_eq!(dataset.get_fragments().len(), 1);
        assert_eq!(dataset.get_fragments()[0].metadata.files.len(), 1);

        let batch = dataset.scan().try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values(),
            &[4, 5, 6]
        );
    }

    #[tokio::test]
    async fn test_datafile_partial_replacement() {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::Int32,
            true,
        )]));
        let empty_reader = RecordBatchIterator::new(vec![], schema.clone());
        let mut dataset = Dataset::write(empty_reader, "memory://", None)
            .await
            .unwrap();
        dataset.validate().await.unwrap();

        let vals: Int32Array = vec![1, 2, 3].into();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vals)]).unwrap();
        dataset
            .append(
                RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
                None,
            )
            .await
            .unwrap();

        let fragment = dataset.get_fragments().pop().unwrap().metadata;

        let extended_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, true),
            ArrowField::new("b", DataType::Int32, true),
        ]));

        // add all null column
        let dataset = Dataset::commit(
            WriteDestination::Dataset(Arc::new(dataset)),
            Operation::Merge {
                fragments: vec![fragment],
                schema: extended_schema.as_ref().try_into().unwrap(),
            },
            Some(2),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await
        .unwrap();

        let partial_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "b",
            DataType::Int32,
            true,
        )]));

        // write a new datafile
        let object_writer = dataset
            .object_store
            .create(&Path::from("data/test.lance"))
            .await
            .unwrap();
        let mut writer = FileWriter::try_new(
            object_writer,
            partial_schema.as_ref().try_into().unwrap(),
            Default::default(),
        )
        .unwrap();

        let vals: Int32Array = vec![4, 5, 6].into();
        let batch = RecordBatch::try_new(partial_schema.clone(), vec![Arc::new(vals)]).unwrap();
        writer.write_batch(&batch).await.unwrap();
        writer.finish().await.unwrap();

        let (major, minor) = lance_file::version::LanceFileVersion::Stable.to_numbers();

        // find the datafile we want to replace
        let new_data_file = DataFile {
            path: "test.lance".to_string(),
            // the second column in the dataset
            fields: vec![1],
            // is located in the first column of this datafile
            column_indices: vec![0],
            file_major_version: major,
            file_minor_version: minor,
            file_size_bytes: CachedFileSize::unknown(),
            base_id: None,
        };

        let dataset = Dataset::commit(
            WriteDestination::Dataset(Arc::new(dataset)),
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, new_data_file)],
            },
            Some(3),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await
        .unwrap();

        assert_eq!(dataset.version().version, 4);
        assert_eq!(dataset.get_fragments().len(), 1);
        assert_eq!(dataset.get_fragments()[0].metadata.files.len(), 2);
        assert_eq!(dataset.get_fragments()[0].metadata.files[0].fields, vec![0]);
        assert_eq!(dataset.get_fragments()[0].metadata.files[1].fields, vec![1]);

        let batch = dataset.scan().try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values(),
            &[1, 2, 3]
        );
        assert_eq!(
            batch
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values(),
            &[4, 5, 6]
        );

        // do it again but on the first column
        // find the datafile we want to replace
        let new_data_file = DataFile {
            path: "test.lance".to_string(),
            // the first column in the dataset
            fields: vec![0],
            // is located in the first column of this datafile
            column_indices: vec![0],
            file_major_version: major,
            file_minor_version: minor,
            file_size_bytes: CachedFileSize::unknown(),
            base_id: None,
        };

        let dataset = Dataset::commit(
            WriteDestination::Dataset(Arc::new(dataset)),
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, new_data_file)],
            },
            Some(4),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await
        .unwrap();

        assert_eq!(dataset.version().version, 5);
        assert_eq!(dataset.get_fragments().len(), 1);
        assert_eq!(dataset.get_fragments()[0].metadata.files.len(), 2);

        let batch = dataset.scan().try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values(),
            &[4, 5, 6]
        );
        assert_eq!(
            batch
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values(),
            &[4, 5, 6]
        );
    }

    #[tokio::test]
    async fn test_datafile_replacement_error() {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::Int32,
            true,
        )]));
        let empty_reader = RecordBatchIterator::new(vec![], schema.clone());
        let mut dataset = Dataset::write(empty_reader, "memory://", None)
            .await
            .unwrap();
        dataset.validate().await.unwrap();

        let vals: Int32Array = vec![1, 2, 3].into();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vals)]).unwrap();
        dataset
            .append(
                RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
                None,
            )
            .await
            .unwrap();

        let fragment = dataset.get_fragments().pop().unwrap().metadata;

        let extended_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, true),
            ArrowField::new("b", DataType::Int32, true),
        ]));

        // add all null column
        let dataset = Dataset::commit(
            WriteDestination::Dataset(Arc::new(dataset)),
            Operation::Merge {
                fragments: vec![fragment],
                schema: extended_schema.as_ref().try_into().unwrap(),
            },
            Some(2),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await
        .unwrap();

        // find the datafile we want to replace
        let new_data_file = DataFile {
            path: "test.lance".to_string(),
            // the second column in the dataset
            fields: vec![1],
            // is located in the first column of this datafile
            column_indices: vec![0],
            file_major_version: 2,
            file_minor_version: 0,
            file_size_bytes: CachedFileSize::unknown(),
            base_id: None,
        };

        let new_data_file = DataFile {
            fields: vec![0, 1],
            ..new_data_file
        };

        let err = Dataset::commit(
            WriteDestination::Dataset(Arc::new(dataset.clone())),
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, new_data_file)],
            },
            Some(2),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("Expected to modify the fragment but no changes were made"),
            "Expected Error::DataFileReplacementError, got {:?}",
            err
        );
    }

    #[tokio::test]
    async fn test_replace_dataset() {
        let test_dir = TempDir::default();
        let test_uri = test_dir.path_str();
        let test_path = test_dir.obj_path();

        let data = gen_batch()
            .col("int", array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(20))
            .unwrap();
        let data1 = data.slice(0, 10);
        let data2 = data.slice(10, 10);
        let mut ds = InsertBuilder::new(&test_uri)
            .execute(vec![data1])
            .await
            .unwrap();

        ds.object_store().remove_dir_all(test_path).await.unwrap();

        let ds2 = InsertBuilder::new(&test_uri)
            .execute(vec![data2.clone()])
            .await
            .unwrap();

        ds.checkout_latest().await.unwrap();
        let roundtripped = ds.scan().try_into_batch().await.unwrap();
        assert_eq!(roundtripped, data2);

        ds.validate().await.unwrap();
        ds2.validate().await.unwrap();
        assert_eq!(ds.manifest.version, 1);
        assert_eq!(ds2.manifest.version, 1);
    }

    #[tokio::test]
    async fn test_session_store_registry() {
        // Create a session
        let session = Arc::new(Session::default());
        let registry = session.store_registry();
        assert!(registry.active_stores().is_empty());

        // Create a dataset with memory store
        let write_params = WriteParams {
            session: Some(session.clone()),
            ..Default::default()
        };
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "a",
                DataType::Int32,
                false,
            )])),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let dataset = InsertBuilder::new("memory://test")
            .with_params(&write_params)
            .execute(vec![batch.clone()])
            .await
            .unwrap();

        // Assert there is one active store.
        assert_eq!(registry.active_stores().len(), 1);

        // If we create another dataset also in memory, it should re-use the
        // existing store.
        let dataset2 = InsertBuilder::new("memory://test2")
            .with_params(&write_params)
            .execute(vec![batch.clone()])
            .await
            .unwrap();
        assert_eq!(registry.active_stores().len(), 1);
        assert_eq!(
            Arc::as_ptr(&dataset.object_store().inner),
            Arc::as_ptr(&dataset2.object_store().inner)
        );

        // If we create another with **different parameters**, it should create a new store.
        let write_params2 = WriteParams {
            session: Some(session.clone()),
            store_params: Some(ObjectStoreParams {
                block_size: Some(10_000),
                ..Default::default()
            }),
            ..Default::default()
        };
        let dataset3 = InsertBuilder::new("memory://test3")
            .with_params(&write_params2)
            .execute(vec![batch.clone()])
            .await
            .unwrap();
        assert_eq!(registry.active_stores().len(), 2);
        assert_ne!(
            Arc::as_ptr(&dataset.object_store().inner),
            Arc::as_ptr(&dataset3.object_store().inner)
        );

        // Remove both datasets
        drop(dataset3);
        assert_eq!(registry.active_stores().len(), 1);
        drop(dataset2);
        drop(dataset);
        assert_eq!(registry.active_stores().len(), 0);
    }

    #[tokio::test]
    async fn test_migrate_v2_manifest_paths() {
        let test_uri = TempStrDir::default();

        let data = lance_datagen::gen_batch()
            .col("key", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let mut dataset = Dataset::write(data, &test_uri, None).await.unwrap();
        assert_eq!(
            dataset.manifest_location().naming_scheme,
            ManifestNamingScheme::V1
        );

        dataset.migrate_manifest_paths_v2().await.unwrap();
        assert_eq!(
            dataset.manifest_location().naming_scheme,
            ManifestNamingScheme::V2
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_fragment_id_zero_not_reused() {
        // Test case 1: Fragment id zero isn't re-used
        // 1. Create a dataset with 1 fragment
        // 2. Delete all rows
        // 3. Append another fragment
        // 4. Assert new fragment has id 1 not 0

        let test_uri = TempStrDir::default();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::UInt32,
            false,
        )]));

        // Create dataset with 1 fragment
        let data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(UInt32Array::from_iter_values(0..10))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![data].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, &test_uri, None).await.unwrap();

        // Verify we have 1 fragment with id 0
        assert_eq!(dataset.get_fragments().len(), 1);
        assert_eq!(dataset.get_fragments()[0].id(), 0);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(0));

        // Delete all rows
        dataset.delete("true").await.unwrap();

        // After deletion, dataset should be empty but max_fragment_id preserved
        assert_eq!(dataset.get_fragments().len(), 0);
        assert_eq!(dataset.count_rows(None).await.unwrap(), 0);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(0));

        // Append another fragment
        let data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(UInt32Array::from_iter_values(20..30))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![data].into_iter().map(Ok), schema.clone());
        let write_params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        let dataset = Dataset::write(batches, &test_uri, Some(write_params))
            .await
            .unwrap();

        // Assert new fragment has id 1, not 0
        assert_eq!(dataset.get_fragments().len(), 1);
        assert_eq!(dataset.get_fragments()[0].id(), 1);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(1));
    }

    #[rstest]
    #[tokio::test]
    async fn test_fragment_id_never_reset() {
        // Test case 2: Fragment id is never reset, even if all rows are deleted
        // 1. Create dataset with N fragments
        // 2. Delete all rows
        // 3. Append more fragments
        // 4. Assert new fragments have ids >= N

        let test_uri = TempStrDir::default();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::UInt32,
            false,
        )]));

        // Create dataset with 3 fragments (N=3)
        let data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(UInt32Array::from_iter_values(0..30))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(data)], schema.clone());
        let write_params = WriteParams {
            max_rows_per_file: 10, // Force multiple fragments
            ..Default::default()
        };
        let mut dataset = Dataset::write(batches, &test_uri, Some(write_params))
            .await
            .unwrap();

        // Verify we have 3 fragments with ids 0, 1, 2
        assert_eq!(dataset.get_fragments().len(), 3);
        assert_eq!(dataset.get_fragments()[0].id(), 0);
        assert_eq!(dataset.get_fragments()[1].id(), 1);
        assert_eq!(dataset.get_fragments()[2].id(), 2);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(2));

        // Delete all rows
        dataset.delete("true").await.unwrap();

        // After deletion, dataset should be empty but max_fragment_id preserved
        assert_eq!(dataset.get_fragments().len(), 0);
        assert_eq!(dataset.count_rows(None).await.unwrap(), 0);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(2));

        // Append more fragments (2 new fragments)
        let data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(UInt32Array::from_iter_values(100..120))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(data)], schema.clone());
        let write_params = WriteParams {
            mode: WriteMode::Append,
            max_rows_per_file: 10, // Force multiple fragments
            ..Default::default()
        };
        let dataset = Dataset::write(batches, &test_uri, Some(write_params))
            .await
            .unwrap();

        // Assert new fragments have ids >= N (3, 4)
        assert_eq!(dataset.get_fragments().len(), 2);
        assert_eq!(dataset.get_fragments()[0].id(), 3);
        assert_eq!(dataset.get_fragments()[1].id(), 4);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(4));
    }

    #[tokio::test]
    async fn test_insert_skip_auto_cleanup() {
        let test_uri = TempStrDir::default();

        // Create initial dataset with aggressive auto cleanup (interval=1, older_than=1ms)
        let data = gen_batch()
            .col("id", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));

        let write_params = WriteParams {
            mode: WriteMode::Create,
            auto_cleanup: Some(AutoCleanupParams {
                interval: 1,
                older_than: chrono::TimeDelta::try_milliseconds(0).unwrap(), // Cleanup versions older than 0ms
            }),
            ..Default::default()
        };

        // Start at 1 second after epoch
        MockClock::set_system_time(std::time::Duration::from_secs(1));

        let dataset = Dataset::write(data, &test_uri, Some(write_params))
            .await
            .unwrap();
        assert_eq!(dataset.version().version, 1);

        // Advance time by 1 second
        MockClock::set_system_time(std::time::Duration::from_secs(2));

        // First append WITHOUT skip_auto_cleanup - should trigger cleanup
        let data1 = gen_batch()
            .col("id", array::step::<Int32Type>())
            .into_df_stream(RowCount::from(50), BatchCount::from(1));

        let write_params1 = WriteParams {
            mode: WriteMode::Append,
            skip_auto_cleanup: false,
            ..Default::default()
        };

        let dataset2 = InsertBuilder::new(WriteDestination::Dataset(Arc::new(dataset)))
            .with_params(&write_params1)
            .execute_stream(data1)
            .await
            .unwrap();

        assert_eq!(dataset2.version().version, 2);

        // Advance time
        MockClock::set_system_time(std::time::Duration::from_secs(3));

        // Need to do another commit for cleanup to take effect since cleanup runs on the old dataset
        let data1_extra = gen_batch()
            .col("id", array::step::<Int32Type>())
            .into_df_stream(RowCount::from(10), BatchCount::from(1));

        let dataset2_extra = InsertBuilder::new(WriteDestination::Dataset(Arc::new(dataset2)))
            .with_params(&write_params1)
            .execute_stream(data1_extra)
            .await
            .unwrap();

        assert_eq!(dataset2_extra.version().version, 3);

        // Version 1 should be cleaned up due to auto cleanup (cleanup runs every version)
        assert!(
            dataset2_extra.checkout_version(1).await.is_err(),
            "Version 1 should have been cleaned up"
        );
        // Version 2 should still exist
        assert!(
            dataset2_extra.checkout_version(2).await.is_ok(),
            "Version 2 should still exist"
        );

        // Advance time
        MockClock::set_system_time(std::time::Duration::from_secs(4));

        // Second append WITH skip_auto_cleanup - should NOT trigger cleanup
        let data2 = gen_batch()
            .col("id", array::step::<Int32Type>())
            .into_df_stream(RowCount::from(30), BatchCount::from(1));

        let write_params2 = WriteParams {
            mode: WriteMode::Append,
            skip_auto_cleanup: true, // Skip auto cleanup
            ..Default::default()
        };

        let dataset3 = InsertBuilder::new(WriteDestination::Dataset(Arc::new(dataset2_extra)))
            .with_params(&write_params2)
            .execute_stream(data2)
            .await
            .unwrap();

        assert_eq!(dataset3.version().version, 4);

        // Version 2 should still exist because skip_auto_cleanup was enabled
        assert!(
            dataset3.checkout_version(2).await.is_ok(),
            "Version 2 should still exist because skip_auto_cleanup was enabled"
        );
        // Version 3 should also still exist
        assert!(
            dataset3.checkout_version(3).await.is_ok(),
            "Version 3 should still exist"
        );
    }

    #[tokio::test]
    async fn test_nullable_struct_v2_1_issue_4385() {
        // Test for issue #4385: nullable struct should preserve null values in v2.1 format
        use arrow_array::cast::AsArray;
        use arrow_schema::Fields;

        // Create a struct field with nullable float field
        let struct_fields = Fields::from(vec![ArrowField::new("x", DataType::Float32, true)]);

        // Create outer struct with the nullable struct as a field (not root)
        let outer_fields = Fields::from(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("data", DataType::Struct(struct_fields.clone()), true),
        ]);
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "record",
            DataType::Struct(outer_fields.clone()),
            false,
        )]));

        // Create data with null struct
        let id_values = Int32Array::from(vec![1, 2, 3]);
        let x_values = Float32Array::from(vec![Some(1.0), Some(2.0), Some(3.0)]);
        let inner_struct_array = StructArray::new(
            struct_fields,
            vec![Arc::new(x_values) as ArrayRef],
            Some(vec![true, false, true].into()), // Second struct is null
        );

        let outer_struct_array = StructArray::new(
            outer_fields,
            vec![
                Arc::new(id_values) as ArrayRef,
                Arc::new(inner_struct_array.clone()) as ArrayRef,
            ],
            None, // Outer struct is not nullable
        );

        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(outer_struct_array)]).unwrap();

        // Write dataset with v2.1 format
        let test_uri = TempStrDir::default();

        let write_params = WriteParams {
            mode: WriteMode::Create,
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        };

        let batches = vec![batch.clone()];
        let batch_reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

        Dataset::write(batch_reader, &test_uri, Some(write_params))
            .await
            .unwrap();

        // Read back the dataset
        let dataset = Dataset::open(&test_uri).await.unwrap();
        let scanner = dataset.scan();
        let result_batches = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(result_batches.len(), 1);
        let result_batch = &result_batches[0];
        let read_outer_struct = result_batch.column(0).as_struct();
        let read_inner_struct = read_outer_struct.column(1).as_struct(); // "data" field

        // The bug: null struct is not preserved
        assert!(
            read_inner_struct.is_null(1),
            "Second struct should be null but it's not. Read value: {:?}",
            read_inner_struct
        );

        // Verify the null count is preserved
        assert_eq!(
            inner_struct_array.null_count(),
            read_inner_struct.null_count(),
            "Null count should be preserved"
        );
    }

    #[tokio::test]
    async fn test_issue_4902_packed_struct_v2_1_read_error() {
        use std::collections::HashMap;

        use arrow_array::{ArrayRef, Int32Array, RecordBatchIterator, StructArray, UInt32Array};
        use arrow_schema::{Field as ArrowField, Fields, Schema as ArrowSchema};

        let struct_fields = Fields::from(vec![
            ArrowField::new("x", DataType::UInt32, false),
            ArrowField::new("y", DataType::UInt32, false),
        ]);
        let mut packed_metadata = HashMap::new();
        packed_metadata.insert("packed".to_string(), "true".to_string());

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("int_col", DataType::Int32, false),
            ArrowField::new("struct_col", DataType::Struct(struct_fields.clone()), false)
                .with_metadata(packed_metadata),
        ]));

        let int_values = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]));
        let x_values = Arc::new(UInt32Array::from(vec![1, 4, 7, 10, 13, 16, 19, 22]));
        let y_values = Arc::new(UInt32Array::from(vec![2, 5, 8, 11, 14, 17, 20, 23]));
        let struct_array = Arc::new(StructArray::new(
            struct_fields,
            vec![x_values.clone() as ArrayRef, y_values.clone() as ArrayRef],
            None,
        ));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                int_values.clone() as ArrayRef,
                struct_array.clone() as ArrayRef,
            ],
        )
        .unwrap();

        let test_uri = TempStrDir::default();
        let write_params = WriteParams {
            mode: WriteMode::Create,
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        };
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone());
        Dataset::write(reader, &test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(&test_uri).await.unwrap();

        let result_batches = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(result_batches, vec![batch.clone()]);

        let struct_batches = dataset
            .scan()
            .project(&["struct_col"])
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(struct_batches.len(), 1);
        let read_struct = struct_batches[0].column(0).as_struct();
        assert_eq!(read_struct, struct_array.as_ref());
    }

    #[tokio::test]
    async fn test_issue_4429_nested_struct_encoding_v2_1_with_over_65k_structs() {
        // Regression test for miniblock 16KB limit with nested struct patterns
        // Tests encoding behavior when a nested struct<list<struct>> contains
        // large amounts of data that exceeds miniblock encoding limits

        // Create a struct with multiple fields that will trigger miniblock encoding
        // Each field is 4 bytes, making the struct narrow enough for miniblock
        let measurement_fields = vec![
            ArrowField::new("val_a", DataType::Float32, true),
            ArrowField::new("val_b", DataType::Float32, true),
            ArrowField::new("val_c", DataType::Float32, true),
            ArrowField::new("val_d", DataType::Float32, true),
            ArrowField::new("seq_high", DataType::Int32, true),
            ArrowField::new("seq_low", DataType::Int32, true),
        ];
        let measurement_type = DataType::Struct(measurement_fields.clone().into());

        // Create nested schema: struct<measurements: list<struct>>
        // This pattern can trigger encoding issues with large data volumes
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "data",
            DataType::Struct(
                vec![ArrowField::new(
                    "measurements",
                    DataType::List(Arc::new(ArrowField::new(
                        "item",
                        measurement_type.clone(),
                        true,
                    ))),
                    true,
                )]
                .into(),
            ),
            true,
        )]));

        // Create large number of measurements that will exceed encoding limits
        // Using 70,520 to match the exact problematic size
        const NUM_MEASUREMENTS: usize = 70_520;

        // Generate data for two full sets (rows 0 and 2 will have data, row 1 empty)
        const TOTAL_MEASUREMENTS: usize = NUM_MEASUREMENTS * 2;

        // Create arrays with realistic values
        let val_a_array = Float32Array::from_iter(
            (0..TOTAL_MEASUREMENTS).map(|i| Some(16.66 + (i as f32 * 0.0001))),
        );
        let val_b_array = Float32Array::from_iter(
            (0..TOTAL_MEASUREMENTS).map(|i| Some(-3.54 + (i as f32 * 0.0002))),
        );
        let val_c_array = Float32Array::from_iter(
            (0..TOTAL_MEASUREMENTS).map(|i| Some(2.94 + (i as f32 * 0.0001))),
        );
        let val_d_array =
            Float32Array::from_iter((0..TOTAL_MEASUREMENTS).map(|i| Some(((i % 50) + 10) as f32)));
        let seq_high_array =
            Int32Array::from_iter((0..TOTAL_MEASUREMENTS).map(|_| Some(1736962329)));
        let seq_low_array = Int32Array::from_iter(
            (0..TOTAL_MEASUREMENTS).map(|i| Some(304403000 + (i * 1000) as i32)),
        );

        // Create the struct array with all measurements
        let struct_array = StructArray::from(vec![
            (
                Arc::new(ArrowField::new("val_a", DataType::Float32, true)),
                Arc::new(val_a_array) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("val_b", DataType::Float32, true)),
                Arc::new(val_b_array) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("val_c", DataType::Float32, true)),
                Arc::new(val_c_array) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("val_d", DataType::Float32, true)),
                Arc::new(val_d_array) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("seq_high", DataType::Int32, true)),
                Arc::new(seq_high_array) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("seq_low", DataType::Int32, true)),
                Arc::new(seq_low_array) as ArrayRef,
            ),
        ]);

        // Create list array with pattern: [70520 items, 0 items, 70520 items]
        // This pattern triggers the issue with V2.1 encoding
        let offsets = vec![
            0i32,
            NUM_MEASUREMENTS as i32,       // End of row 0
            NUM_MEASUREMENTS as i32,       // End of row 1 (empty)
            (NUM_MEASUREMENTS * 2) as i32, // End of row 2
        ];
        let list_array = ListArray::try_new(
            Arc::new(ArrowField::new("item", measurement_type, true)),
            arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(offsets)),
            Arc::new(struct_array) as ArrayRef,
            None,
        )
        .unwrap();

        // Create the outer struct wrapping the list
        let data_struct = StructArray::from(vec![(
            Arc::new(ArrowField::new(
                "measurements",
                DataType::List(Arc::new(ArrowField::new(
                    "item",
                    DataType::Struct(measurement_fields.into()),
                    true,
                ))),
                true,
            )),
            Arc::new(list_array) as ArrayRef,
        )]);

        // Create the final record batch with 3 rows
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(data_struct) as ArrayRef]).unwrap();

        assert_eq!(batch.num_rows(), 3, "Should have exactly 3 rows");

        let test_uri = TempStrDir::default();

        // Test with V2.1 format which has different encoding behavior
        let batches = vec![batch];
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

        // V2.1 format triggers miniblock encoding for narrow structs
        let write_params = WriteParams {
            data_storage_version: Some(lance_file::version::LanceFileVersion::V2_1),
            ..Default::default()
        };

        // Write dataset - this will panic with miniblock 16KB assertion
        let dataset = Dataset::write(reader, &test_uri, Some(write_params))
            .await
            .unwrap();

        dataset.validate().await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 3);
    }

    async fn prepare_json_dataset() -> (Dataset, String) {
        let text_col = Arc::new(StringArray::from(vec![
            r#"{
              "Title": "HarryPotter Chapter One",
              "Content": "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say...",
              "Author": "J.K. Rowling",
              "Price": 128,
              "Language": ["english", "chinese"]
          }"#,
            r#"{
             "Title": "Fairy Talest",
             "Content": "Once upon a time, on a bitterly cold New Year's Eve, a little girl...",
             "Author": "ANDERSEN",
             "Price": 50,
             "Language": ["english", "chinese"]
          }"#,
        ]));
        let json_col = "json_field".to_string();

        // Prepare dataset
        let mut metadata = HashMap::new();
        metadata.insert(
            ARROW_EXT_NAME_KEY.to_string(),
            ARROW_JSON_EXT_NAME.to_string(),
        );
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![
                Field::new(&json_col, DataType::Utf8, false).with_metadata(metadata)
            ])
            .into(),
            vec![text_col.clone()],
        )
        .unwrap();
        let schema = batch.schema();
        let stream = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        let dataset = Dataset::write(stream, "memory://test/table", None)
            .await
            .unwrap();

        (dataset, json_col)
    }

    #[tokio::test]
    async fn test_json_inverted_fuzziness_query() {
        let (mut dataset, json_col) = prepare_json_dataset().await;

        // Create inverted index for json col
        dataset
            .create_index(
                &[&json_col],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default().lance_tokenizer("json".to_string()),
                true,
            )
            .await
            .unwrap();

        // Match query with fuzziness
        let query = FullTextSearchQuery {
            query: FtsQuery::Match(
                MatchQuery::new("Content,str,Dursley".to_string())
                    .with_column(Some(json_col.clone())),
            ),
            limit: None,
            wand_factor: None,
        };
        let batch = dataset
            .scan()
            .full_text_search(query)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(1, batch.num_rows());

        let query = FullTextSearchQuery {
            query: FtsQuery::Match(
                MatchQuery::new("Content,str,Bursley".to_string())
                    .with_column(Some(json_col.clone())),
            ),
            limit: None,
            wand_factor: None,
        };
        let batch = dataset
            .scan()
            .full_text_search(query)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(0, batch.num_rows());

        let query = FullTextSearchQuery {
            query: FtsQuery::Match(
                MatchQuery::new("Content,str,Bursley".to_string())
                    .with_column(Some(json_col.clone()))
                    .with_fuzziness(Some(1)),
            ),
            limit: None,
            wand_factor: None,
        };
        let batch = dataset
            .scan()
            .full_text_search(query)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(1, batch.num_rows());

        let query = FullTextSearchQuery {
            query: FtsQuery::Match(
                MatchQuery::new("Content,str,ABursley".to_string())
                    .with_column(Some(json_col.clone()))
                    .with_fuzziness(Some(1)),
            ),
            limit: None,
            wand_factor: None,
        };
        let batch = dataset
            .scan()
            .full_text_search(query)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(0, batch.num_rows());

        let query = FullTextSearchQuery {
            query: FtsQuery::Match(
                MatchQuery::new("Content,str,ABursley".to_string())
                    .with_column(Some(json_col.clone()))
                    .with_fuzziness(Some(2)),
            ),
            limit: None,
            wand_factor: None,
        };
        let batch = dataset
            .scan()
            .full_text_search(query)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(1, batch.num_rows());

        let query = FullTextSearchQuery {
            query: FtsQuery::Match(
                MatchQuery::new("Dontent,str,Bursley".to_string())
                    .with_column(Some(json_col.clone()))
                    .with_fuzziness(Some(2)),
            ),
            limit: None,
            wand_factor: None,
        };
        let batch = dataset
            .scan()
            .full_text_search(query)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(0, batch.num_rows());
    }

    #[tokio::test]
    async fn test_json_inverted_match_query() {
        let (mut dataset, json_col) = prepare_json_dataset().await;

        // Create inverted index for json col, with max token len 10 and enable stemming,
        // lower case, and remove stop words
        dataset
            .create_index(
                &[&json_col],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default()
                    .lance_tokenizer("json".to_string())
                    .max_token_length(Some(10))
                    .stem(true)
                    .lower_case(true)
                    .remove_stop_words(true),
                true,
            )
            .await
            .unwrap();

        // Match query with token length exceed max token length
        let query = FullTextSearchQuery {
            query: FtsQuery::Match(
                MatchQuery::new("Title,str,harrypotter".to_string())
                    .with_column(Some(json_col.clone())),
            ),
            limit: None,
            wand_factor: None,
        };
        let batch = dataset
            .scan()
            .full_text_search(query)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(0, batch.num_rows());

        // Match query with stemming
        let query = FullTextSearchQuery {
            query: FtsQuery::Match(
                MatchQuery::new("Content,str,onc".to_string()).with_column(Some(json_col.clone())),
            ),
            limit: None,
            wand_factor: None,
        };
        let batch = dataset
            .scan()
            .full_text_search(query)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(1, batch.num_rows());

        // Match query with lower case
        let query = FullTextSearchQuery {
            query: FtsQuery::Match(
                MatchQuery::new("Content,str,DURSLEY".to_string())
                    .with_column(Some(json_col.clone())),
            ),
            limit: None,
            wand_factor: None,
        };
        let batch = dataset
            .scan()
            .full_text_search(query)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(1, batch.num_rows());

        // Match query with stop word
        let query = FullTextSearchQuery {
            query: FtsQuery::Match(
                MatchQuery::new("Content,str,and".to_string()).with_column(Some(json_col.clone())),
            ),
            limit: None,
            wand_factor: None,
        };
        let batch = dataset
            .scan()
            .full_text_search(query)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(0, batch.num_rows());
    }

    #[tokio::test]
    async fn test_json_inverted_flat_match_query() {
        let (mut dataset, json_col) = prepare_json_dataset().await;

        // Create inverted index for json col
        dataset
            .create_index(
                &[&json_col],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default()
                    .lance_tokenizer("json".to_string())
                    .stem(false),
                true,
            )
            .await
            .unwrap();

        // Append data
        let text_col = Arc::new(StringArray::from(vec![
            r#"{
              "Title": "HarryPotter Chapter Two",
              "Content": "Nearly ten years had passed since the Dursleys had woken up...",
              "Author": "J.K. Rowling",
              "Price": 128,
              "Language": ["english", "chinese"]
            }"#,
        ]));

        let mut metadata = HashMap::new();
        metadata.insert(
            ARROW_EXT_NAME_KEY.to_string(),
            ARROW_JSON_EXT_NAME.to_string(),
        );
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![
                Field::new(&json_col, DataType::Utf8, false).with_metadata(metadata)
            ])
            .into(),
            vec![text_col.clone()],
        )
        .unwrap();
        let schema = batch.schema();
        let stream = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        dataset.append(stream, None).await.unwrap();

        // Test match query
        let query = FullTextSearchQuery {
            query: FtsQuery::Match(
                MatchQuery::new("Title,str,harrypotter".to_string())
                    .with_column(Some(json_col.clone())),
            ),
            limit: None,
            wand_factor: None,
        };
        let batch = dataset
            .scan()
            .full_text_search(query)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(2, batch.num_rows());
    }

    #[tokio::test]
    async fn test_json_inverted_phrase_query() {
        // Prepare json dataset
        let (mut dataset, json_col) = prepare_json_dataset().await;

        // Create inverted index for json col
        dataset
            .create_index(
                &[&json_col],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default()
                    .lance_tokenizer("json".to_string())
                    .stem(false)
                    .with_position(true),
                true,
            )
            .await
            .unwrap();

        // Test phrase query
        let query = FullTextSearchQuery {
            query: FtsQuery::Phrase(
                PhraseQuery::new("Title,str,harrypotter one chapter".to_string())
                    .with_column(Some(json_col.clone())),
            ),
            limit: None,
            wand_factor: None,
        };
        let batch = dataset
            .scan()
            .full_text_search(query)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(0, batch.num_rows());

        let query = FullTextSearchQuery {
            query: FtsQuery::Phrase(
                PhraseQuery::new("Title,str,harrypotter chapter one".to_string())
                    .with_column(Some(json_col.clone())),
            ),
            limit: None,
            wand_factor: None,
        };
        let batch = dataset
            .scan()
            .full_text_search(query)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(1, batch.num_rows());
    }

    #[tokio::test]
    async fn test_json_inverted_multimatch_query() {
        // Prepare json dataset
        let (mut dataset, json_col) = prepare_json_dataset().await;

        // Create inverted index for json col
        dataset
            .create_index(
                &[&json_col],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default()
                    .lance_tokenizer("json".to_string())
                    .stem(false),
                true,
            )
            .await
            .unwrap();

        // Test multi match query
        let query = FullTextSearchQuery {
            query: FtsQuery::MultiMatch(MultiMatchQuery {
                match_queries: vec![
                    MatchQuery::new("Title,str,harrypotter".to_string())
                        .with_column(Some(json_col.clone())),
                    MatchQuery::new("Language,str,english".to_string())
                        .with_column(Some(json_col.clone())),
                ],
            }),
            limit: None,
            wand_factor: None,
        };
        let batch = dataset
            .scan()
            .full_text_search(query)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(2, batch.num_rows());
    }

    #[tokio::test]
    async fn test_json_inverted_boolean_query() {
        // Prepare json dataset
        let (mut dataset, json_col) = prepare_json_dataset().await;

        // Create inverted index for json col
        dataset
            .create_index(
                &[&json_col],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default()
                    .lance_tokenizer("json".to_string())
                    .stem(false),
                true,
            )
            .await
            .unwrap();

        // Test boolean query
        let query = FullTextSearchQuery {
            query: FtsQuery::Boolean(BooleanQuery {
                should: vec![],
                must: vec![
                    FtsQuery::Match(
                        MatchQuery::new("Language,str,english".to_string())
                            .with_column(Some(json_col.clone())),
                    ),
                    FtsQuery::Match(
                        MatchQuery::new("Title,str,harrypotter".to_string())
                            .with_column(Some(json_col.clone())),
                    ),
                ],
                must_not: vec![],
            }),
            limit: None,
            wand_factor: None,
        };
        let batch = dataset
            .scan()
            .full_text_search(query)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(1, batch.num_rows());
    }

    #[tokio::test]
    async fn test_sql_contains_tokens() {
        let text_col = Arc::new(StringArray::from(vec![
            "a cat catch a fish",
            "a fish catch a cat",
            "a white cat catch a big fish",
            "cat catchup fish",
            "cat fish catch",
        ]));

        // Prepare dataset
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![Field::new("text", DataType::Utf8, false)]).into(),
            vec![text_col.clone()],
        )
        .unwrap();
        let schema = batch.schema();
        let stream = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(stream, "memory://test/table", None)
            .await
            .unwrap();

        // Test without fts index
        let results = execute_sql(
            "select * from foo where contains_tokens(text, 'cat catch fish')",
            "foo".to_string(),
            Arc::new(dataset.clone()),
        )
        .await
        .unwrap();

        assert_results(
            results,
            &StringArray::from(vec![
                "a cat catch a fish",
                "a fish catch a cat",
                "a white cat catch a big fish",
                "cat fish catch",
            ]),
        );

        // Verify plan, should not contain ScalarIndexQuery.
        let results = execute_sql(
            "explain select * from foo where contains_tokens(text, 'cat catch fish')",
            "foo".to_string(),
            Arc::new(dataset.clone()),
        )
        .await
        .unwrap();
        let plan = format!("{:?}", results);
        assert_not_contains!(&plan, "ScalarIndexQuery");

        // Test with unsuitable fts index
        dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default().base_tokenizer("raw".to_string()),
                true,
            )
            .await
            .unwrap();

        let results = execute_sql(
            "select * from foo where contains_tokens(text, 'cat catch fish')",
            "foo".to_string(),
            Arc::new(dataset.clone()),
        )
        .await
        .unwrap();

        assert_results(
            results,
            &StringArray::from(vec![
                "a cat catch a fish",
                "a fish catch a cat",
                "a white cat catch a big fish",
                "cat fish catch",
            ]),
        );

        // Verify plan, should not contain ScalarIndexQuery because fts index is not unsuitable.
        let results = execute_sql(
            "explain select * from foo where contains_tokens(text, 'cat catch fish')",
            "foo".to_string(),
            Arc::new(dataset.clone()),
        )
        .await
        .unwrap();
        let plan = format!("{:?}", results);
        assert_not_contains!(&plan, "ScalarIndexQuery");

        // Test with suitable fts index
        dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default()
                    .max_token_length(None)
                    .stem(false),
                true,
            )
            .await
            .unwrap();

        let results = execute_sql(
            "select * from foo where contains_tokens(text, 'cat catch fish')",
            "foo".to_string(),
            Arc::new(dataset.clone()),
        )
        .await
        .unwrap();

        assert_results(
            results,
            &StringArray::from(vec![
                "a cat catch a fish",
                "a fish catch a cat",
                "a white cat catch a big fish",
                "cat fish catch",
            ]),
        );

        // Verify plan, should contain ScalarIndexQuery.
        let results = execute_sql(
            "explain select * from foo where contains_tokens(text, 'cat catch fish')",
            "foo".to_string(),
            Arc::new(dataset.clone()),
        )
        .await
        .unwrap();
        let plan = format!("{:?}", results);
        assert_contains!(&plan, "ScalarIndexQuery");
    }

    async fn execute_sql(
        sql: &str,
        table: String,
        dataset: Arc<Dataset>,
    ) -> Result<Vec<RecordBatch>> {
        let ctx = SessionContext::new();
        ctx.register_table(
            table,
            Arc::new(LanceTableProvider::new(dataset, false, false)),
        )?;
        register_functions(&ctx);

        let df = ctx.sql(sql).await?;
        Ok(df
            .execute_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?)
    }

    fn assert_results<T: Array + PartialEq + 'static>(results: Vec<RecordBatch>, values: &T) {
        assert_eq!(results.len(), 1);
        let results = results.into_iter().next().unwrap();
        assert_eq!(results.num_columns(), 1);

        assert_eq!(
            results.column(0).as_any().downcast_ref::<T>().unwrap(),
            values
        )
    }

    #[tokio::test]
    async fn test_limit_pushdown_in_physical_plan() -> Result<()> {
        use tempfile::tempdir;
        let temp_dir = tempdir()?;

        let dataset_path = temp_dir.path().join("limit_pushdown_dataset");
        let values: Vec<i32> = (0..1000).collect();
        let array = Int32Array::from(values);
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)])?;

        let write_params = WriteParams {
            mode: WriteMode::Create,
            max_rows_per_file: 100,
            ..Default::default()
        };

        let batch_reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        Dataset::write(
            batch_reader,
            dataset_path.to_str().unwrap(),
            Some(write_params),
        )
        .await?;

        let mut dataset = Dataset::open(dataset_path.to_str().unwrap()).await?;

        dataset
            .create_index(
                &["value"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await?;

        // Test 1: No filter with limit
        {
            let mut scanner = dataset.scan();
            scanner.limit(Some(100), None)?;
            let plan = scanner.explain_plan(true).await?;

            assert!(plan.contains("range_before=Some(0..100)"));
            assert!(plan.contains("range_after=None"));

            let batches: Vec<RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(100, total_rows);
        }

        // Test 2: Indexed filter with limit
        {
            let mut scanner = dataset.scan();
            scanner.filter("value >= 500")?.limit(Some(50), None)?;
            let plan = scanner.explain_plan(true).await?;

            assert!(plan.contains("range_after=Some(0..50)"));
            assert!(plan.contains("range_before=None"));

            let batches: Vec<RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(50, total_rows);
        }

        // Test 3: Offset + Limit
        {
            let mut scanner = dataset.scan();
            scanner.filter("value < 500")?.limit(Some(30), Some(20))?;
            let plan = scanner.explain_plan(true).await?;

            assert!(plan.contains("GlobalLimitExec: skip=20, fetch=30"));
            assert!(plan.contains("range_after=Some(0..50)"));

            let batches: Vec<RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(30, total_rows);

            // Verify exact values (should be 20..50)
            let all_values: Vec<i32> = batches
                .iter()
                .flat_map(|batch| {
                    batch
                        .column_by_name("value")
                        .unwrap()
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .values()
                        .iter()
                        .copied()
                        .collect::<Vec<_>>()
                })
                .collect();
            assert_eq!(all_values, (20..50).collect::<Vec<i32>>());
        }

        // Test 4: Large limit exceeding data
        {
            let mut scanner = dataset.scan();
            scanner.limit(Some(5000), None)?;
            let plan = scanner.explain_plan(true).await?;

            assert!(plan.contains("range_before=Some(0..1000)"));

            let batches: Vec<RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(1000, total_rows);
        }

        // Test 5: Cross-fragment filter with limit
        {
            let mut scanner = dataset.scan();
            scanner
                .filter("value >= 95 AND value <= 205")?
                .limit(Some(50), None)?;
            let plan = scanner.explain_plan(true).await?;

            assert!(plan.contains("range_after=Some(0..50)"));

            let batches: Vec<RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(50, total_rows);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_index_take_batch_size() -> Result<()> {
        use tempfile::tempdir;
        let temp_dir = tempdir()?;

        let dataset_path = temp_dir.path().join("ints_dataset");
        let values: Vec<i32> = (0..1024).collect();
        let array = Int32Array::from(values);
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "ints",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)])?;
        let write_params = WriteParams {
            mode: WriteMode::Create,
            max_rows_per_file: 100,
            ..Default::default()
        };
        let batch_reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        Dataset::write(
            batch_reader,
            dataset_path.to_str().unwrap(),
            Some(write_params),
        )
        .await?;
        let mut dataset = Dataset::open(dataset_path.to_str().unwrap()).await?;
        dataset
            .create_index(
                &["ints"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await?;

        let mut scanner = dataset.scan();
        scanner.batch_size(50).filter("ints > 0")?.with_row_id();
        let batches: Vec<RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(1023, total_rows);
        assert_eq!(21, batches.len());

        let mut scanner = dataset.scan();
        scanner
            .batch_size(50)
            .filter("ints > 0")?
            .limit(Some(1024), None)?
            .with_row_id();
        let batches: Vec<RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(1023, total_rows);
        assert_eq!(21, batches.len());

        let dataset_path2 = temp_dir.path().join("strings_dataset");
        let strings: Vec<String> = (0..1024).map(|i| format!("string-{}", i)).collect();
        let string_array = StringArray::from(strings);
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "strings",
            DataType::Utf8,
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(string_array)])?;
        let write_params = WriteParams {
            mode: WriteMode::Create,
            max_rows_per_file: 100,
            ..Default::default()
        };
        let batch_reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        Dataset::write(
            batch_reader,
            dataset_path2.to_str().unwrap(),
            Some(write_params),
        )
        .await?;
        let mut dataset2 = Dataset::open(dataset_path2.to_str().unwrap()).await?;
        dataset2
            .create_index(
                &["strings"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await?;

        let mut scanner = dataset2.scan();
        scanner
            .batch_size(50)
            .filter("contains(strings, 'ing')")?
            .limit(Some(1024), None)?
            .with_row_id();
        let batches: Vec<RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(1024, total_rows);
        assert_eq!(21, batches.len());

        Ok(())
    }

    // This test covers
    // 1. Create branch from main, a branch and a global tag
    // 2. Write to each created branch and verify data
    // 2. Load branch from nested uris
    // 3. Checkout branch from main, a branch and a global tag
    // 4. List branches and verify branch metadata
    // 5. Delete branches
    // 6. Delete zombie branches
    #[tokio::test]
    async fn test_branch() {
        let tempdir = TempDir::default();
        let test_uri = tempdir.path_str();
        let data_storage_version = LanceFileVersion::Stable;

        // Generate consistent test data batches
        let generate_data = |prefix: &str, start_id: i32, row_count: u64| {
            gen_batch()
                .col("id", array::step_custom::<Int32Type>(start_id, 1))
                .col("value", array::fill_utf8(format!("{prefix}_data")))
                .into_reader_rows(RowCount::from(row_count), BatchCount::from(1))
        };

        // Reusable dataset writer with configurable mode
        async fn write_dataset(
            uri: &str,
            data_reader: impl RecordBatchReader + Send + 'static,
            mode: WriteMode,
            version: LanceFileVersion,
        ) -> Dataset {
            let params = WriteParams {
                max_rows_per_file: 100,
                max_rows_per_group: 20,
                data_storage_version: Some(version),
                mode,
                ..Default::default()
            };
            Dataset::write(data_reader, uri, Some(params))
                .await
                .unwrap()
        }

        // Unified dataset scanning and row counting
        async fn collect_rows(dataset: &Dataset) -> (usize, Vec<RecordBatch>) {
            let batches = dataset
                .scan()
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            (batches.iter().map(|b| b.num_rows()).sum(), batches)
        }

        // Phase 1: Create empty dataset, write data batch 1, create branch1 based on version_number, write data batch 2
        let mut dataset = write_dataset(
            &test_uri,
            generate_data("batch1", 0, 50),
            WriteMode::Create,
            data_storage_version,
        )
        .await;

        let original_version = dataset.version().version;
        assert_eq!(original_version, 1);

        // Create branch1 on the latest version and write data batch 2
        let mut branch1_dataset = dataset
            .create_branch("branch1", original_version, None)
            .await
            .unwrap();
        assert_eq!(branch1_dataset.uri, format!("{}/tree/branch1", test_uri));

        branch1_dataset = write_dataset(
            branch1_dataset.uri(),
            generate_data("batch2", 50, 30),
            WriteMode::Append,
            data_storage_version,
        )
        .await;

        // Phase 2: Create branch2 based on branch1's latest version_number, write data batch 3
        let mut branch2_dataset = branch1_dataset
            .create_branch(
                "dev/branch2",
                ("branch1", branch1_dataset.version().version),
                None,
            )
            .await
            .unwrap();
        assert_eq!(
            branch2_dataset.uri,
            format!("{}/tree/dev/branch2", test_uri)
        );

        branch2_dataset = write_dataset(
            branch2_dataset.uri(),
            generate_data("batch3", 80, 20),
            WriteMode::Append,
            data_storage_version,
        )
        .await;

        // Phase 3: Create a tag on branch2, the actual tag content is under root dataset
        // create branch3 based on that tag, write data batch 4
        branch2_dataset
            .tags()
            .create_on_branch(
                "tag1",
                branch2_dataset.version().version,
                Some("dev/branch2"),
            )
            .await
            .unwrap();

        let mut branch3_dataset = branch2_dataset
            .create_branch("feature/nathan/branch3", "tag1", None)
            .await
            .unwrap();
        assert_eq!(
            branch3_dataset.uri,
            format!("{}/tree/feature/nathan/branch3", test_uri)
        );

        branch3_dataset = write_dataset(
            branch3_dataset.uri(),
            generate_data("batch4", 100, 25),
            WriteMode::Append,
            data_storage_version,
        )
        .await;

        // Verify data correctness and independence of each branch
        // Main branch only has data 1 (50 rows)
        let main_dataset = Dataset::open(&test_uri).await.unwrap();
        let (main_rows, _) = collect_rows(&main_dataset).await;
        assert_eq!(main_rows, 50); // only batch1
        assert_eq!(main_dataset.version().version, 1);

        // branch1 has data 1 + 2 (80 rows)
        let updated_branch1 = Dataset::open(branch1_dataset.uri()).await.unwrap();
        let (branch1_rows, _) = collect_rows(&updated_branch1).await;
        assert_eq!(branch1_rows, 80); // batch1+batch2
        assert_eq!(updated_branch1.version().version, 2);

        // branch2 has data 1 + 2 + 3 (100 rows)
        let updated_branch2 = Dataset::open(branch2_dataset.uri()).await.unwrap();
        let (branch2_rows, _) = collect_rows(&updated_branch2).await;
        assert_eq!(branch2_rows, 100); // batch1+batch2+batch3
        assert_eq!(updated_branch2.version().version, 3);

        // branch3 has data 1 + 2 + 3 + 4 (125 rows)
        let updated_branch3 = Dataset::open(branch3_dataset.uri()).await.unwrap();
        let (branch3_rows, _) = collect_rows(&updated_branch3).await;
        assert_eq!(branch3_rows, 125); // batch1+batch2+batch3+batch4
        assert_eq!(updated_branch3.version().version, 4);

        // Use list_branches to get branch list and verify each field of branch_content
        let branches = dataset.list_branches().await.unwrap();
        assert_eq!(branches.len(), 3);
        assert!(branches.contains_key("branch1"));
        assert!(branches.contains_key("dev/branch2"));
        assert!(branches.contains_key("feature/nathan/branch3"));

        // Verify branch1 content
        let branch1_content = branches.get("branch1").unwrap();
        assert_eq!(branch1_content.parent_branch, None); // Created based on main branch
        assert_eq!(branch1_content.parent_version, 1);
        assert!(branch1_content.create_at > 0);
        assert!(branch1_content.manifest_size > 0);

        // Verify branch2 content
        let branch2_content = branches.get("dev/branch2").unwrap();
        assert_eq!(branch2_content.parent_branch.as_deref().unwrap(), "branch1");
        assert_eq!(branch2_content.parent_version, 2);
        assert!(branch2_content.create_at > 0);
        assert!(branch2_content.manifest_size > 0);
        assert!(branch2_content.create_at >= branch1_content.create_at);

        // Verify branch3 content
        let branch3_content = branches.get("feature/nathan/branch3").unwrap();
        // Created based on tag pointed to branch2
        assert_eq!(
            branch3_content.parent_branch.as_deref().unwrap(),
            "dev/branch2"
        );
        assert_eq!(branch3_content.parent_version, 3);
        assert!(branch3_content.create_at > 0);
        assert!(branch3_content.manifest_size > 0);
        assert!(branch3_content.create_at >= branch2_content.create_at);

        // Verify checkout_branch
        let checkout_branch1 = main_dataset.checkout_branch("branch1").await.unwrap();
        let checkout_branch2 = checkout_branch1
            .checkout_branch("dev/branch2")
            .await
            .unwrap();
        let checkout_branch2_tag = checkout_branch1.checkout_version("tag1").await.unwrap();
        let checkout_branch3 = checkout_branch2_tag
            .checkout_branch("feature/nathan/branch3")
            .await
            .unwrap();
        let checkout_branch3_at_version3 = checkout_branch2
            .checkout_version(("feature/nathan/branch3", 3))
            .await
            .unwrap();
        assert_eq!(checkout_branch3.version().version, 4);
        assert_eq!(checkout_branch3_at_version3.version().version, 3);
        assert_eq!(checkout_branch2.version().version, 3);
        assert_eq!(checkout_branch2_tag.version().version, 3);
        assert_eq!(checkout_branch1.version().version, 2);
        assert_eq!(checkout_branch3.count_rows(None).await.unwrap(), 125);
        assert_eq!(
            checkout_branch3_at_version3.count_rows(None).await.unwrap(),
            100
        );
        assert_eq!(checkout_branch2.count_rows(None).await.unwrap(), 100);
        assert_eq!(checkout_branch2_tag.count_rows(None).await.unwrap(), 100);
        assert_eq!(checkout_branch1.count_rows(None).await.unwrap(), 80);
        assert_eq!(
            checkout_branch3.manifest.branch.as_deref().unwrap(),
            "feature/nathan/branch3"
        );
        assert_eq!(
            checkout_branch3_at_version3
                .manifest
                .branch
                .as_deref()
                .unwrap(),
            "feature/nathan/branch3"
        );
        assert_eq!(
            checkout_branch2.manifest.branch.as_deref().unwrap(),
            "dev/branch2"
        );
        assert_eq!(
            checkout_branch2_tag.manifest.branch.as_deref().unwrap(),
            "dev/branch2"
        );
        assert_eq!(
            checkout_branch1.manifest.branch.as_deref().unwrap(),
            "branch1"
        );

        let mut dataset = main_dataset;
        // Finally delete all branches
        dataset.delete_branch("branch1").await.unwrap();
        dataset.delete_branch("dev/branch2").await.unwrap();
        // Test deleting zombie branch
        let root_location = dataset.refs.root().unwrap();
        let branch_file = branch_contents_path(&root_location.path, "feature/nathan/branch3");
        dataset.object_store.delete(&branch_file).await.unwrap();
        // Now "feature/nathan/branch3" is a zombie branch
        // Use delete_branch to verify if the directory is cleaned up
        dataset
            .force_delete_branch("feature/nathan/branch3")
            .await
            .unwrap();
        let cleaned_path = Path::parse(format!("{}/tree/feature", test_uri)).unwrap();
        assert!(!dataset.object_store.exists(&cleaned_path).await.unwrap());

        // Verify list_branches is empty
        let branches_after_delete = dataset.list_branches().await.unwrap();
        assert!(branches_after_delete.is_empty());

        // Verify branch directories are all deleted cleanly
        let test_path = tempdir.obj_path();
        let branches = dataset
            .object_store
            .read_dir(test_path.child("tree"))
            .await
            .unwrap();
        assert!(branches.is_empty());
    }

    #[tokio::test]
    async fn test_add_bases() {
        use lance_table::format::BasePath;
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32};
        use std::sync::Arc;

        // Create a test dataset
        let test_uri = "memory://add_bases_test";
        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = Dataset::write(
            data_gen.batch(5),
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Create,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let dataset = Arc::new(dataset);

        // Test adding new base paths
        let new_bases = vec![
            BasePath::new(
                0,
                "memory://bucket1".to_string(),
                Some("bucket1".to_string()),
                false,
            ),
            BasePath::new(
                0,
                "memory://bucket2".to_string(),
                Some("bucket2".to_string()),
                true,
            ),
        ];

        let updated_dataset = dataset.add_bases(new_bases, None).await.unwrap();

        // Verify the base paths were added
        assert_eq!(updated_dataset.manifest.base_paths.len(), 2);

        let bucket1 = updated_dataset
            .manifest
            .base_paths
            .values()
            .find(|bp| bp.name == Some("bucket1".to_string()))
            .expect("bucket1 not found");
        let bucket2 = updated_dataset
            .manifest
            .base_paths
            .values()
            .find(|bp| bp.name == Some("bucket2".to_string()))
            .expect("bucket2 not found");

        assert_eq!(bucket1.path, "memory://bucket1");
        assert!(!bucket1.is_dataset_root);
        assert_eq!(bucket2.path, "memory://bucket2");
        assert!(bucket2.is_dataset_root);

        let updated_dataset = Arc::new(updated_dataset);

        // Test conflict detection - try to add a base with the same name
        let conflicting_bases = vec![BasePath::new(
            0,
            "memory://bucket3".to_string(),
            Some("bucket1".to_string()),
            false,
        )];

        let result = updated_dataset.add_bases(conflicting_bases, None).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Conflict detected"));

        // Test conflict detection - try to add a base with the same path
        let conflicting_bases = vec![BasePath::new(
            0,
            "memory://bucket1".to_string(),
            Some("bucket3".to_string()),
            false,
        )];

        let result = updated_dataset.add_bases(conflicting_bases, None).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Conflict detected"));
    }

    #[tokio::test]
    async fn test_concurrent_add_bases_conflict() {
        use lance_table::format::BasePath;
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32};
        use std::sync::Arc;

        // Create a test dataset
        let test_uri = "memory://concurrent_add_bases_test";
        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = Dataset::write(
            data_gen.batch(5),
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Create,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Clone the dataset to simulate concurrent access
        let dataset = Arc::new(dataset);
        let dataset_clone = Arc::new(dataset.clone());

        // First transaction adds base1
        let new_bases1 = vec![BasePath::new(
            0,
            "memory://bucket1".to_string(),
            Some("base1".to_string()),
            false,
        )];

        let updated_dataset = dataset.add_bases(new_bases1, None).await.unwrap();

        // Second transaction tries to add a different base (base2)
        // This should succeed as there's no conflict
        let new_bases2 = vec![BasePath::new(
            0,
            "memory://bucket2".to_string(),
            Some("base2".to_string()),
            false,
        )];

        let result = dataset_clone.add_bases(new_bases2, None).await;
        assert!(result.is_ok());

        // Verify both bases are present after conflict resolution
        let mut final_dataset = updated_dataset;
        final_dataset.checkout_latest().await.unwrap();
        assert_eq!(final_dataset.manifest.base_paths.len(), 2);

        let base1 = final_dataset
            .manifest
            .base_paths
            .values()
            .find(|bp| bp.name == Some("base1".to_string()));
        let base2 = final_dataset
            .manifest
            .base_paths
            .values()
            .find(|bp| bp.name == Some("base2".to_string()));

        assert!(base1.is_some());
        assert!(base2.is_some());
    }

    #[tokio::test]
    async fn test_concurrent_add_bases_name_conflict() {
        use lance_table::format::BasePath;
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32};
        use std::sync::Arc;

        // Create a test dataset
        let test_uri = "memory://concurrent_name_conflict_test";
        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = Dataset::write(
            data_gen.batch(5),
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Create,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Clone the dataset to simulate concurrent access
        let dataset_clone = dataset.clone();
        let dataset = Arc::new(dataset);
        let dataset_clone = Arc::new(dataset_clone);

        // First transaction adds base with name "shared_base"
        let new_bases1 = vec![BasePath::new(
            0,
            "memory://bucket1".to_string(),
            Some("shared_base".to_string()),
            false,
        )];

        let _updated_dataset = dataset.add_bases(new_bases1, None).await.unwrap();

        // Second transaction tries to add a different base with same name
        // This should fail due to name conflict
        let new_bases2 = vec![BasePath::new(
            0,
            "memory://bucket2".to_string(),
            Some("shared_base".to_string()),
            false,
        )];

        let result = dataset_clone.add_bases(new_bases2, None).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("incompatible with concurrent transaction"));
    }

    #[tokio::test]
    async fn test_concurrent_add_bases_path_conflict() {
        use lance_table::format::BasePath;
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32};
        use std::sync::Arc;

        // Create a test dataset
        let test_uri = "memory://concurrent_path_conflict_test";
        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = Dataset::write(
            data_gen.batch(5),
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Create,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Clone the dataset to simulate concurrent access
        let dataset_clone = dataset.clone();
        let dataset = Arc::new(dataset);
        let dataset_clone = Arc::new(dataset_clone);

        // First transaction adds base with path "memory://shared_path"
        let new_bases1 = vec![BasePath::new(
            0,
            "memory://shared_path".to_string(),
            Some("base1".to_string()),
            false,
        )];

        let _updated_dataset = dataset.add_bases(new_bases1, None).await.unwrap();

        // Second transaction tries to add a different base with same path
        // This should fail due to path conflict
        let new_bases2 = vec![BasePath::new(
            0,
            "memory://shared_path".to_string(),
            Some("base2".to_string()),
            false,
        )];

        let result = dataset_clone.add_bases(new_bases2, None).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("incompatible with concurrent transaction"));
    }

    #[tokio::test]
    async fn test_concurrent_add_bases_with_data_write() {
        use lance_table::format::BasePath;
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32};
        use std::sync::Arc;

        // Create a test dataset
        let test_uri = "memory://concurrent_write_test";
        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = Dataset::write(
            data_gen.batch(5),
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Create,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Clone the dataset to simulate concurrent access
        let dataset_clone = dataset.clone();
        let dataset = Arc::new(dataset);

        // First transaction adds a new base
        let new_bases = vec![BasePath::new(
            0,
            "memory://bucket1".to_string(),
            Some("base1".to_string()),
            false,
        )];

        let updated_dataset = dataset.add_bases(new_bases, None).await.unwrap();

        // Concurrent transaction appends data
        // This should succeed as add_bases doesn't conflict with data writes
        let result = Dataset::write(
            data_gen.batch(5),
            WriteDestination::Dataset(Arc::new(dataset_clone)),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await;

        assert!(result.is_ok());

        // Verify both operations are reflected
        let mut final_dataset = updated_dataset;
        final_dataset.checkout_latest().await.unwrap();

        // Should have the new base
        assert_eq!(final_dataset.manifest.base_paths.len(), 1);
        assert!(final_dataset
            .manifest
            .base_paths
            .values()
            .any(|bp| bp.name == Some("base1".to_string())));

        // Should have both data writes (10 rows total)
        assert_eq!(final_dataset.count_rows(None).await.unwrap(), 10);
    }

    #[tokio::test]
    async fn test_auto_infer_lance_tokenizer() {
        let (mut dataset, json_col) = prepare_json_dataset().await;

        // Create inverted index for json col. Expect auto-infer 'json' for lance tokenizer.
        dataset
            .create_index(
                &[&json_col],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // Match query succeed only when lance tokenizer is 'json'
        let query = FullTextSearchQuery {
            query: FtsQuery::Match(
                MatchQuery::new("Content,str,once".to_string()).with_column(Some(json_col.clone())),
            ),
            limit: None,
            wand_factor: None,
        };
        let batch = dataset
            .scan()
            .full_text_search(query)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(1, batch.num_rows());
    }
}
