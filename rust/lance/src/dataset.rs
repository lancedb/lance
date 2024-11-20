// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Dataset
//!

use arrow_array::{RecordBatch, RecordBatchReader};
use byteorder::{ByteOrder, LittleEndian};
use chrono::{prelude::*, Duration};
use deepsize::DeepSizeOf;
use futures::future::BoxFuture;
use futures::stream::{self, StreamExt, TryStreamExt};
use futures::{FutureExt, Stream};
use itertools::Itertools;
use lance_core::traits::DatasetTakeRows;
use lance_core::utils::address::RowAddress;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_datafusion::projection::ProjectionPlan;
use lance_file::datatypes::populate_schema_dictionary;
use lance_file::version::LanceFileVersion;
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use lance_io::object_writer::ObjectWriter;
use lance_io::traits::WriteExt;
use lance_io::utils::{read_last_block, read_metadata_offset, read_struct};
use lance_table::format::{
    DataStorageFormat, Fragment, Index, Manifest, MAGIC, MAJOR_VERSION, MINOR_VERSION,
};
use lance_table::io::commit::{
    migrate_scheme_to_v2, CommitError, CommitHandler, CommitLock, ManifestLocation,
    ManifestNamingScheme,
};
use lance_table::io::manifest::{read_manifest, write_manifest};
use object_store::path::Path;
use prost::Message;
use rowids::get_row_id_index;
use serde::{Deserialize, Serialize};
use snafu::{location, Location};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use tracing::instrument;

mod blob;
pub mod builder;
pub mod cleanup;
pub mod fragment;
mod hash_joiner;
pub mod index;
pub mod optimize;
pub mod progress;
pub mod refs;
pub(crate) mod rowids;
pub mod scanner;
mod schema_evolution;
mod take;
pub mod transaction;
pub mod updater;
mod utils;
mod write;

use self::builder::DatasetBuilder;
use self::cleanup::RemovalStats;
use self::fragment::FileFragment;
use self::refs::Tags;
use self::scanner::{DatasetRecordBatchStream, Scanner};
use self::transaction::{Operation, Transaction};
use self::write::write_fragments_internal;
use crate::datatypes::Schema;
use crate::error::box_error;
use crate::io::commit::{commit_detached_transaction, commit_new_dataset, commit_transaction};
use crate::session::Session;
use crate::utils::temporal::{timestamp_to_nanos, utc_now, SystemTime};
use crate::{Error, Result};
pub use blob::BlobFile;
use hash_joiner::HashJoiner;
pub use lance_core::ROW_ID;
use lance_table::feature_flags::{apply_feature_flags, can_read_dataset};
pub use schema_evolution::{
    BatchInfo, BatchUDF, ColumnAlteration, NewColumnTransform, UDFCheckpointStore,
};
pub use take::TakeBuilder;
pub use write::merge_insert::{
    MergeInsertBuilder, MergeInsertJob, WhenMatched, WhenNotMatched, WhenNotMatchedBySource,
};
pub use write::update::{UpdateBuilder, UpdateJob};
#[allow(deprecated)]
pub use write::{
    write_fragments, CommitBuilder, InsertBuilder, WriteDestination, WriteMode, WriteParams,
};

const INDICES_DIR: &str = "_indices";

pub const DATA_DIR: &str = "data";
pub const BLOB_DIR: &str = "_blobs";
pub(crate) const DEFAULT_INDEX_CACHE_SIZE: usize = 256;
pub(crate) const DEFAULT_METADATA_CACHE_SIZE: usize = 256;

/// Lance Dataset
#[derive(Debug, Clone)]
pub struct Dataset {
    pub object_store: Arc<ObjectStore>,
    pub(crate) commit_handler: Arc<dyn CommitHandler>,
    /// Uri of the dataset.
    ///
    /// On cloud storage, we can not use [Dataset::base] to build the full uri because the
    /// `bucket` is swlloed in the inner [ObjectStore].
    uri: String,
    pub(crate) base: Path,
    pub(crate) manifest: Arc<Manifest>,
    pub(crate) session: Arc<Session>,
    pub tags: Tags,
    pub manifest_naming_scheme: ManifestNamingScheme,
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
            metadata: BTreeMap::default(),
        }
    }
}

/// Customize read behavior of a dataset.
#[derive(Clone, Debug)]
pub struct ReadParams {
    /// Cache size for index cache. If it is zero, index cache is disabled.
    ///
    pub index_cache_size: usize,

    /// Metadata cache size for the fragment metadata. If it is zero, metadata
    /// cache is disabled.
    pub metadata_cache_size: usize,

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

    pub object_store_registry: Arc<ObjectStoreRegistry>,
}

impl ReadParams {
    /// Set the cache size for indices. Set to zero, to disable the cache.
    pub fn index_cache_size(&mut self, cache_size: usize) -> &mut Self {
        self.index_cache_size = cache_size;
        self
    }

    /// Set the cache size for the file metadata. Set to zero to disable this cache.
    pub fn metadata_cache_size(&mut self, cache_size: usize) -> &mut Self {
        self.metadata_cache_size = cache_size;
        self
    }

    /// Set a shared session for the datasets.
    pub fn session(&mut self, session: Arc<Session>) -> &mut Self {
        self.session = Some(session);
        self
    }

    /// Provide an object store registry for custom object stores
    pub fn with_object_store_registry(
        &mut self,
        object_store_registry: Arc<ObjectStoreRegistry>,
    ) -> &mut Self {
        self.object_store_registry = object_store_registry;
        self
    }

    /// Use the explicit locking to resolve the latest version
    pub fn set_commit_lock<T: CommitLock + Send + Sync + 'static>(&mut self, lock: Arc<T>) {
        self.commit_handler = Some(Arc::new(lock));
    }
}

impl Default for ReadParams {
    fn default() -> Self {
        Self {
            index_cache_size: DEFAULT_INDEX_CACHE_SIZE,
            metadata_cache_size: DEFAULT_METADATA_CACHE_SIZE,
            session: None,
            store_options: None,
            commit_handler: None,
            object_store_registry: Arc::new(ObjectStoreRegistry::default()),
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
        let schema = dataset_schema.project(&columns).unwrap();
        Self::Schema(Arc::new(schema))
    }

    pub fn from_schema(schema: Schema) -> Self {
        Self::Schema(Arc::new(schema))
    }

    /// Provide a list of projection with SQL transform.
    ///
    /// # Parameters
    /// - `columns`: A list of tuples where the first element is resulted column name and the second
    ///              element is the SQL expression.
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

    pub fn into_projection_plan(self, dataset_schema: &Schema) -> Result<ProjectionPlan> {
        match self {
            Self::Schema(schema) => Ok(ProjectionPlan::new_empty(
                Arc::new(dataset_schema.project_by_schema(schema.as_ref())?),
                /*load_blobs=*/ false,
            )),
            Self::Sql(columns) => {
                ProjectionPlan::try_new(dataset_schema, &columns, /*load_blobs=*/ false)
            }
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
            refs::Ref::Version(version) => self.checkout_by_version_number(version).await,
            refs::Ref::Tag(tag) => self.checkout_by_tag(tag.as_str()).await,
        }
    }

    /// Check out the latest version of the dataset
    pub async fn checkout_latest(&mut self) -> Result<()> {
        self.manifest = Arc::new(self.latest_manifest().await?);
        Ok(())
    }

    async fn checkout_by_version_number(&self, version: u64) -> Result<Self> {
        let base_path = self.base.clone();
        let manifest_location = self
            .commit_handler
            .resolve_version_location(&base_path, version, &self.object_store.inner)
            .await?;
        let manifest = Self::load_manifest(self.object_store.as_ref(), &manifest_location).await?;
        Self::checkout_manifest(
            self.object_store.clone(),
            base_path,
            self.uri.clone(),
            manifest,
            self.session.clone(),
            self.commit_handler.clone(),
            manifest_location.naming_scheme,
        )
        .await
    }

    async fn checkout_by_tag(&self, tag: &str) -> Result<Self> {
        let version = self.tags.get_version(tag).await?;
        self.checkout_by_version_number(version).await
    }

    async fn load_manifest(
        object_store: &ObjectStore,
        manifest_location: &ManifestLocation,
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

        populate_schema_dictionary(&mut manifest.schema, object_reader.as_ref()).await?;

        Ok(manifest)
    }

    async fn checkout_manifest(
        object_store: Arc<ObjectStore>,
        base_path: Path,
        uri: String,
        manifest: Manifest,
        session: Arc<Session>,
        commit_handler: Arc<dyn CommitHandler>,
        manifest_naming_scheme: ManifestNamingScheme,
    ) -> Result<Self> {
        let tags = Tags::new(
            object_store.clone(),
            commit_handler.clone(),
            base_path.clone(),
        );
        Ok(Self {
            object_store,
            base: base_path,
            uri,
            manifest: Arc::new(manifest),
            commit_handler,
            session,
            tags,
            manifest_naming_scheme,
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
        builder.execute_stream(batches).await
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
            .execute_stream(batches)
            .await?;

        *self = new_dataset;

        Ok(())
    }

    /// Get the fully qualified URI of this dataset.
    pub fn uri(&self) -> &str {
        &self.uri
    }

    /// Get the full manifest of the dataset version.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    // TODO: Cache this
    pub async fn blobs_dataset(&self) -> Result<Option<Arc<Self>>> {
        if let Some(blobs_version) = self.manifest.blob_dataset_version {
            let blobs_path = self.base.child(BLOB_DIR);
            let blob_manifest_location = self
                .commit_handler
                .resolve_version_location(&blobs_path, blobs_version, &self.object_store.inner)
                .await?;
            let manifest = read_manifest(&self.object_store, &blob_manifest_location.path).await?;
            let blobs_dataset = Self::checkout_manifest(
                self.object_store.clone(),
                blobs_path,
                format!("{}/{}", self.uri, BLOB_DIR),
                manifest,
                self.session.clone(),
                self.commit_handler.clone(),
                ManifestNamingScheme::V2,
            )
            .await?;
            Ok(Some(Arc::new(blobs_dataset)))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn is_legacy_storage(&self) -> bool {
        self.manifest
            .data_storage_format
            .lance_file_version()
            .unwrap()
            == LanceFileVersion::Legacy
    }

    pub async fn latest_manifest(&self) -> Result<Manifest> {
        read_manifest(
            &self.object_store,
            &self
                .commit_handler
                .resolve_latest_version(&self.base, &self.object_store)
                .await?,
        )
        .await
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

    /// Restore the currently checked out version of the dataset as the latest version.
    pub async fn restore(&mut self) -> Result<()> {
        let latest_manifest = self.latest_manifest().await?;
        let latest_version = latest_manifest.version;

        let transaction = Transaction::new(
            latest_version,
            Operation::Restore {
                version: self.manifest.version,
            },
            /*blobs_op=*/ None,
            None,
        );

        self.manifest = Arc::new(
            commit_transaction(
                self,
                &self.object_store,
                self.commit_handler.as_ref(),
                &transaction,
                &Default::default(),
                &Default::default(),
                self.manifest_naming_scheme,
            )
            .await?,
        );

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
    pub fn cleanup_old_versions(
        &self,
        older_than: Duration,
        delete_unverified: Option<bool>,
        error_if_tagged_old_versions: Option<bool>,
    ) -> BoxFuture<Result<RemovalStats>> {
        let before = utc_now() - older_than;
        cleanup::cleanup_old_versions(
            self,
            before,
            delete_unverified,
            error_if_tagged_old_versions,
        )
        .boxed()
    }

    #[allow(clippy::too_many_arguments)]
    async fn do_commit(
        base_uri: WriteDestination<'_>,
        operation: Operation,
        blobs_op: Option<Operation>,
        read_version: Option<u64>,
        store_params: Option<ObjectStoreParams>,
        commit_handler: Option<Arc<dyn CommitHandler>>,
        object_store_registry: Arc<ObjectStoreRegistry>,
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

        let transaction = Transaction::new(read_version, operation, blobs_op, None);

        let mut builder = CommitBuilder::new(base_uri)
            .with_object_store_registry(object_store_registry)
            .enable_v2_manifest_paths(enable_v2_manifest_paths)
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
        object_store_registry: Arc<ObjectStoreRegistry>,
        enable_v2_manifest_paths: bool,
    ) -> Result<Self> {
        Self::do_commit(
            dest.into(),
            operation,
            // TODO: Allow blob operations to be specified? (breaking change?)
            /*blobs_op=*/
            None,
            read_version,
            store_params,
            commit_handler,
            object_store_registry,
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
        object_store_registry: Arc<ObjectStoreRegistry>,
        enable_v2_manifest_paths: bool,
    ) -> Result<Self> {
        Self::do_commit(
            dest.into(),
            operation,
            // TODO: Allow blob operations to be specified? (breaking change?)
            /*blobs_op=*/
            None,
            read_version,
            store_params,
            commit_handler,
            object_store_registry,
            enable_v2_manifest_paths,
            /*detached=*/ true,
        )
        .await
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
            .map(|f| async move { f.count_rows().await })
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

    pub async fn take_blobs(
        self: &Arc<Self>,
        row_ids: &[u64],
        column: impl AsRef<str>,
    ) -> Result<Vec<BlobFile>> {
        blob::take_blobs(self, row_ids, column.as_ref()).await
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
        let ids = (0..num_rows as u64).choose_multiple(&mut rand::thread_rng(), n);
        self.take(&ids, projection.clone()).await
    }

    /// Delete rows based on a predicate.
    pub async fn delete(&mut self, predicate: &str) -> Result<()> {
        let mut updated_fragments: Vec<Fragment> = Vec::new();
        let mut deleted_fragment_ids: Vec<u64> = Vec::new();
        stream::iter(self.get_fragments())
            .map(|f| async move {
                let old_fragment = f.metadata.clone();
                let new_fragment = f.delete(predicate).await?.map(|f| f.metadata);
                Ok((old_fragment, new_fragment))
            })
            .buffer_unordered(get_num_compute_intensive_cpus())
            // Drop the fragments that were deleted.
            .try_for_each(|(old_fragment, new_fragment)| {
                if let Some(new_fragment) = new_fragment {
                    if new_fragment != old_fragment {
                        updated_fragments.push(new_fragment);
                    }
                } else {
                    deleted_fragment_ids.push(old_fragment.id);
                }
                futures::future::ready(Ok::<_, crate::Error>(()))
            })
            .await?;

        let transaction = Transaction::new(
            self.manifest.version,
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids,
                predicate: predicate.to_string(),
            },
            // No change is needed to the blobs dataset.  The blobs are implicitly deleted since the
            // rows that reference them are deleted.
            /*blobs_op=*/
            None,
            None,
        );

        let manifest = commit_transaction(
            self,
            &self.object_store,
            self.commit_handler.as_ref(),
            &transaction,
            &Default::default(),
            &Default::default(),
            self.manifest_naming_scheme,
        )
        .await?;

        self.manifest = Arc::new(manifest);

        Ok(())
    }

    pub async fn count_deleted_rows(&self) -> Result<usize> {
        futures::stream::iter(self.get_fragments())
            .map(|f| async move { f.count_deletions().await })
            .buffer_unordered(self.object_store.io_parallelism())
            .try_fold(0, |acc, x| futures::future::ready(Ok(acc + x)))
            .await
    }

    pub(crate) fn object_store(&self) -> &ObjectStore {
        &self.object_store
    }

    pub(crate) async fn manifest_file(&self, version: u64) -> Result<Path> {
        self.commit_handler
            .resolve_version(&self.base, version, &self.object_store.inner)
            .await
    }

    pub(crate) fn data_dir(&self) -> Path {
        self.base.child(DATA_DIR)
    }

    pub(crate) fn indices_dir(&self) -> Path {
        self.base.child(INDICES_DIR)
    }

    pub fn session(&self) -> Arc<Session> {
        self.session.clone()
    }

    pub fn version(&self) -> Version {
        Version::from(self.manifest.as_ref())
    }

    /// Get the number of entries currently in the index cache.
    pub fn index_cache_entry_count(&self) -> usize {
        self.session.index_cache.get_size()
    }

    /// Get cache hit ratio.
    pub fn index_cache_hit_rate(&self) -> f32 {
        self.session.index_cache.hit_rate()
    }

    pub fn cache_size_bytes(&self) -> u64 {
        self.session.deep_size_of() as u64
    }

    /// Get all versions.
    pub async fn versions(&self) -> Result<Vec<Version>> {
        let mut versions: Vec<Version> = self
            .commit_handler
            .list_manifests(&self.base, &self.object_store.inner)
            .await?
            .try_filter_map(|path| async move {
                match read_manifest(&self.object_store, &path).await {
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
        self.commit_handler
            .resolve_latest_version_id(&self.base, &self.object_store)
            .await
    }

    pub fn count_fragments(&self) -> usize {
        self.manifest.fragments.len()
    }

    /// Get the schema of the dataset
    pub fn schema(&self) -> &Schema {
        &self.manifest.schema
    }

    /// Similar to [Self::schema], but only returns fields with the default storage class
    pub fn local_schema(&self) -> &Schema {
        &self.manifest.local_schema
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

    pub(crate) fn fragments(&self) -> &Arc<Vec<Fragment>> {
        &self.manifest.fragments
    }

    // Gets a filtered list of fragments from ids in O(N) time instead of using
    // `get_fragment` which would require O(N^2) time.
    fn get_frags_from_ordered_ids(&self, ordered_ids: &[u32]) -> Vec<Option<FileFragment>> {
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
        let mut filtered_sorted_ids = Vec::with_capacity(sorted_addrs.len());
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
                            filtered_sorted_ids.push(Some(u64::from(next_addr)));
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
                            filtered_sorted_ids.push(None);
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
                    filtered_sorted_ids.push(Some(u64::from(next_addr)));
                    if let Some(next) = sorted_addr_iter.next() {
                        next_addr = next;
                    } else {
                        break;
                    }
                }
            } else {
                // Frag doesn't exist (possibly deleted), delete all items
                while next_addr.fragment_id() == *frag_id {
                    filtered_sorted_ids.push(None);
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
        perm.apply_inv_slice_in_place(&mut filtered_sorted_ids);
        Ok(addr_or_ids
            .iter()
            .zip(filtered_sorted_ids)
            .filter_map(|(addr_or_id, maybe_addr)| maybe_addr.map(|_| *addr_or_id))
            .collect())
    }

    pub(crate) async fn filter_deleted_addresses(&self, addrs: &[u64]) -> Result<Vec<u64>> {
        self.filter_addr_or_ids(addrs, addrs).await
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
    /// # let data = lance_datagen::gen()
    /// #  .col("key", array::step::<Int32Type>())
    /// #  .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    /// # let fut = async {
    /// let mut dataset = Dataset::write(data, "memory://test", None).await.unwrap();
    /// assert_eq!(dataset.manifest_naming_scheme, ManifestNamingScheme::V1);
    ///
    /// dataset.migrate_manifest_paths_v2().await.unwrap();
    /// assert_eq!(dataset.manifest_naming_scheme, ManifestNamingScheme::V2);
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
}

/// # Schema Evolution
///
/// Lance datasets support evolving the schema. Several operations are
/// supported that mirror common SQL operations:
///
/// - [Self::add_columns()]: Add new columns to the dataset, similar to `ALTER TABLE ADD COLUMN`.
/// - [Self::drop_columns()]: Drop columns from the dataset, similar to `ALTER TABLE DROP COLUMN`.
/// - [Self::alter_columns()]: Modify columns in the dataset, changing their name, type, or nullability.
///                    Similar to `ALTER TABLE ALTER COLUMN`.
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
        if self.schema().field(left_on).is_none() {
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
            // It is not possible to add blob columns using merge
            /*blobs_op=*/
            None,
            None,
        );

        let manifest = commit_transaction(
            self,
            &self.object_store,
            self.commit_handler.as_ref(),
            &transaction,
            &Default::default(),
            &Default::default(),
            self.manifest_naming_scheme,
        )
        .await?;

        self.manifest = Arc::new(manifest);

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

    /// Update key-value pairs in config.
    pub async fn update_config(
        &mut self,
        upsert_values: impl IntoIterator<Item = (String, String)>,
    ) -> Result<()> {
        let transaction = Transaction::new(
            self.manifest.version,
            Operation::UpdateConfig {
                upsert_values: Some(HashMap::from_iter(upsert_values)),
                delete_keys: None,
            },
            /*blobs_op=*/ None,
            None,
        );

        let manifest = commit_transaction(
            self,
            &self.object_store,
            self.commit_handler.as_ref(),
            &transaction,
            &Default::default(),
            &Default::default(),
            self.manifest_naming_scheme,
        )
        .await?;

        self.manifest = Arc::new(manifest);

        Ok(())
    }

    /// Delete keys from the config.
    pub async fn delete_config_keys(&mut self, delete_keys: &[&str]) -> Result<()> {
        let transaction = Transaction::new(
            self.manifest.version,
            Operation::UpdateConfig {
                upsert_values: None,
                delete_keys: Some(Vec::from_iter(delete_keys.iter().map(ToString::to_string))),
            },
            /*blob_op=*/ None,
            None,
        );

        let manifest = commit_transaction(
            self,
            &self.object_store,
            self.commit_handler.as_ref(),
            &transaction,
            &Default::default(),
            &Default::default(),
            self.manifest_naming_scheme,
        )
        .await?;

        self.manifest = Arc::new(manifest);

        Ok(())
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
    use_move_stable_row_ids: bool,             // default false
    use_legacy_format: Option<bool>,           // default None
    storage_format: Option<DataStorageFormat>, // default None
}

impl Default for ManifestWriteConfig {
    fn default() -> Self {
        Self {
            auto_set_feature_flags: true,
            timestamp: None,
            use_move_stable_row_ids: false,
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
    indices: Option<Vec<Index>>,
    config: &ManifestWriteConfig,
    naming_scheme: ManifestNamingScheme,
) -> std::result::Result<(), CommitError> {
    if config.auto_set_feature_flags {
        apply_feature_flags(manifest, config.use_move_stable_row_ids)?;
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
        .await?;

    Ok(())
}

fn write_manifest_file_to_path<'a>(
    object_store: &'a ObjectStore,
    manifest: &'a mut Manifest,
    indices: Option<Vec<Index>>,
    path: &'a Path,
) -> BoxFuture<'a, Result<()>> {
    Box::pin(async {
        let mut object_writer = ObjectWriter::new(object_store, path).await?;
        let pos = write_manifest(&mut object_writer, manifest, indices).await?;
        object_writer
            .write_magics(pos, MAJOR_VERSION, MINOR_VERSION, MAGIC)
            .await?;
        object_writer.shutdown().await?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;
    use crate::arrow::FixedSizeListArrayExt;
    use crate::dataset::optimize::{compact_files, CompactionOptions};
    use crate::dataset::WriteMode::Overwrite;
    use crate::index::vector::VectorIndexParams;
    use crate::utils::test::TestDatasetGenerator;

    use arrow::array::as_struct_array;
    use arrow::compute::concat_batches;
    use arrow_array::{
        builder::StringDictionaryBuilder,
        cast::as_string_array,
        types::{Float32Type, Int32Type},
        ArrayRef, DictionaryArray, Float32Array, Int32Array, Int64Array, Int8Array,
        Int8DictionaryArray, RecordBatchIterator, StringArray, UInt16Array, UInt32Array,
    };
    use arrow_array::{
        Array, FixedSizeListArray, GenericStringArray, Int16Array, Int16DictionaryArray,
        StructArray,
    };
    use arrow_ord::sort::sort_to_indices;
    use arrow_schema::{
        DataType, Field as ArrowField, Fields as ArrowFields, Schema as ArrowSchema,
    };
    use lance_arrow::bfloat16::{self, ARROW_EXT_META_KEY, ARROW_EXT_NAME_KEY, BFLOAT16_EXT_NAME};
    use lance_core::datatypes::LANCE_STORAGE_CLASS_SCHEMA_META_KEY;
    use lance_datagen::{array, gen, BatchCount, Dimension, RowCount};
    use lance_file::version::LanceFileVersion;
    use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams};
    use lance_index::{scalar::ScalarIndexParams, vector::DIST_COL, DatasetIndexExt, IndexType};
    use lance_linalg::distance::MetricType;
    use lance_table::feature_flags;
    use lance_table::format::WriterVersion;
    use lance_table::io::commit::RenameCommitHandler;
    use lance_table::io::deletion::read_deletion_file;
    use lance_testing::datagen::generate_random_array;
    use pretty_assertions::assert_eq;
    use rstest::rstest;
    use tempfile::{tempdir, TempDir};
    use url::Url;

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
            let test_dir = tempdir().unwrap();
            create_file(test_dir.path(), mode, data_storage_version).await
        }
    }

    #[rstest]
    #[lance_test_macros::test(tokio::test)]
    async fn test_create_and_fill_empty_dataset(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
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
            test_uri,
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
        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let expected_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..10))],
        )
        .unwrap();

        // get actual dataset
        let actual_ds = Dataset::open(test_uri).await.unwrap();
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
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
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
        let result = Dataset::write(reader, test_uri, write_params)
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
        use crate::utils::test::IoTrackingStore;

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
        let dataset = Dataset::write(batches, "memory://test", None)
            .await
            .unwrap();

        // Then open with wrapping store.
        let memory_store = dataset.object_store.inner.clone();
        let (io_stats_wrapper, io_stats) = IoTrackingStore::new_wrapper();
        let _dataset = DatasetBuilder::from_uri("memory://test")
            .with_read_params(ReadParams {
                store_options: Some(ObjectStoreParams {
                    object_store_wrapper: Some(io_stats_wrapper),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .with_object_store(
                memory_store,
                Url::parse("memory://test").unwrap(),
                Arc::new(RenameCommitHandler),
            )
            .load()
            .await
            .unwrap();

        let get_iops = || io_stats.lock().unwrap().read_iops;

        // There should be only two IOPS:
        // 1. List _versions directory to get the latest manifest location
        // 2. Read the manifest file. (The manifest is small enough to be read in one go.
        //    Larger manifests would result in more IOPS.)
        assert_eq!(get_iops(), 2);
    }

    #[rstest]
    #[tokio::test]
    async fn test_write_params(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        use fragment::FragReadConfig;

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

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
        let dataset = Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), num_rows);

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 10);
        assert_eq!(dataset.count_fragments(), 10);
        for fragment in &fragments {
            assert_eq!(fragment.count_rows().await.unwrap(), 100);
            let reader = fragment
                .open(dataset.schema(), FragReadConfig::default(), None)
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

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

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
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
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
                .resolve_latest_version(&dataset.base, dataset.object_store())
                .await
                .unwrap(),
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
                .resolve_latest_version(&dataset.base, dataset.object_store())
                .await
                .unwrap(),
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
                use_move_stable_row_ids: false,
                use_legacy_format: None,
                storage_format: None,
            },
            dataset.manifest_naming_scheme,
        )
        .await
        .unwrap();

        // Check it rejects reading it
        let read_result = Dataset::open(test_uri).await;
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
            test_uri,
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
        let test_dir = tempdir().unwrap();

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

        let test_uri = test_dir.path().to_str().unwrap();
        let mut write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(20..40))],
        )
        .unwrap()];
        write_params.mode = WriteMode::Append;
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let expected_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..40))],
        )
        .unwrap();

        let actual_ds = Dataset::open(test_uri).await.unwrap();
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
    async fn test_self_dataset_append(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = tempdir().unwrap();

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

        let test_uri = test_dir.path().to_str().unwrap();
        let mut write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let mut ds = Dataset::write(batches, test_uri, Some(write_params.clone()))
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

        let actual_ds = Dataset::open(test_uri).await.unwrap();
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
        let test_dir = tempdir().unwrap();

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

        let test_uri = test_dir.path().to_str().unwrap();
        let mut write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let mut ds = Dataset::write(batches, test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        write_params.mode = WriteMode::Append;
        let other_batches =
            RecordBatchIterator::new(other_batches.into_iter().map(Ok), other_schema.clone());

        let result = ds.append(other_batches, Some(write_params.clone())).await;
        // Error because schema is different
        assert!(matches!(result, Err(Error::SchemaMismatch { .. })))
    }

    #[tokio::test]
    async fn append_dictionary() {
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

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params.clone()))
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
        Dataset::write(batches, test_uri, Some(write_params.clone()))
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

        // Try write to dataset (fail)
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let result = Dataset::write(batches, test_uri, Some(write_params)).await;
        assert!(result.is_err());
    }

    #[rstest]
    #[tokio::test]
    async fn overwrite_dataset(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = tempdir().unwrap();

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

        let test_uri = test_dir.path().to_str().unwrap();
        let mut write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let dataset = Dataset::write(batches, test_uri, Some(write_params.clone()))
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
        write_params.mode = WriteMode::Overwrite;
        let new_batch_reader =
            RecordBatchIterator::new(new_batches.into_iter().map(Ok), new_schema.clone());
        let dataset = Dataset::write(new_batch_reader, test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        // Fragment ids reset after overwrite.
        assert_eq!(fragments[0].id(), 0);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(0));

        let actual_ds = Dataset::open(test_uri).await.unwrap();
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
        let first_ver = DatasetBuilder::from_uri(test_uri)
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
        let test_dir = tempdir().unwrap();

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

        let test_uri = test_dir.path().to_str().unwrap();
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
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
        let test_dir = tempdir().unwrap();

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

        let test_uri = test_dir.path().to_str().unwrap();

        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

        let mut dataset = Dataset::write(
            reader,
            test_uri,
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
        let dataset = Dataset::write(reader, test_uri, Some(write_params))
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
        let dataset = Dataset::write(reader, test_uri, Some(write_params))
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
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let data = gen().col("int", array::step::<Int32Type>());
        // Write 64Ki rows.  We should get 16 4Ki pages
        let mut dataset = Dataset::write(
            data.into_reader_rows(RowCount::from(16 * 1024), BatchCount::from(4)),
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                enable_move_stable_row_ids: use_stable_row_id,
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
        let test_dir = tempdir().unwrap();

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
        let test_uri = test_dir.path().to_str().unwrap();
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
    }

    #[tokio::test]
    async fn test_create_fts_index_with_empty_table() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "text",
            DataType::Utf8,
            false,
        )]));

        let batches: Vec<RecordBatch> = vec![];
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(reader, test_uri, None)
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

    #[tokio::test]
    async fn test_create_fts_index_with_empty_strings() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

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
        let mut dataset = Dataset::write(reader, test_uri, None)
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

    fn assert_all_manifests_use_scheme(test_dir: &TempDir, scheme: ManifestNamingScheme) {
        let entries_names = test_dir
            .path()
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
        let data = lance_datagen::gen()
            .col("key", array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(10))
            .unwrap();
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
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
        };
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = Dataset::commit(
            test_uri,
            operation,
            None,
            None,
            None,
            Arc::new(ObjectStoreRegistry::default()),
            true, // enable_v2_manifest_paths
        )
        .await
        .unwrap();

        assert!(dataset.manifest_naming_scheme == ManifestNamingScheme::V2);

        assert_all_manifests_use_scheme(&test_dir, ManifestNamingScheme::V2);
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

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let write_params = WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(data_storage_version),
            enable_move_stable_row_ids: use_stable_row_id,
            ..Default::default()
        };

        let batches = RecordBatchIterator::new(vec![batch1].into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let batches = RecordBatchIterator::new(vec![batch2].into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
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
        let mut dataset = Dataset::open(test_uri).await.unwrap();
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
        let dataset = Dataset::open(test_uri).await.unwrap();
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

        let data = lance_datagen::gen()
            .col("key", array::step::<Int32Type>())
            .col("value", array::fill_utf8("value".to_string()))
            .into_reader_rows(RowCount::from(1_000), BatchCount::from(10));

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let write_params = WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(data_storage_version),
            max_rows_per_file: 1024,
            max_rows_per_group: 150,
            enable_move_stable_row_ids: use_stable_row_id,
            ..Default::default()
        };
        Dataset::write(data, test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let mut dataset = Dataset::open(test_uri).await.unwrap();
        assert_eq!(dataset.fragments().len(), 10);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(9));

        let new_data = lance_datagen::gen()
            .col("key2", array::step_custom::<Int32Type>(500, 1))
            .col("new_value", array::fill_utf8("new_value".to_string()))
            .into_reader_rows(RowCount::from(1_000), BatchCount::from(10));

        dataset.merge(new_data, "key", "key2").await.unwrap();
        dataset.validate().await.unwrap();
    }

    #[rstest]
    #[tokio::test]
    async fn test_delete(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] with_scalar_index: bool,
    ) {
        use std::collections::HashSet;

        fn sequence_data(range: Range<u32>) -> RecordBatch {
            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("i", DataType::UInt32, false),
                ArrowField::new("x", DataType::UInt32, false),
            ]));
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt32Array::from_iter_values(range.clone())),
                    Arc::new(UInt32Array::from_iter_values(range.map(|v| v * 2))),
                ],
            )
            .unwrap()
        }
        // Write a dataset
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::UInt32, false),
            ArrowField::new("x", DataType::UInt32, false),
        ]));
        let data = sequence_data(0..100);
        // Split over two files.
        let batches = vec![data.slice(0, 50), data.slice(50, 50)];
        let mut dataset = TestDatasetGenerator::new(batches, data_storage_version)
            .make_hostile(test_uri)
            .await;

        if with_scalar_index {
            dataset
                .create_index(
                    &["i"],
                    IndexType::Scalar,
                    Some("scalar_index".to_string()),
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
        }

        // Delete nothing
        dataset.delete("i < 0").await.unwrap();
        dataset.validate().await.unwrap();

        // We should not have any deletion file still
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        assert_eq!(dataset.count_fragments(), 2);
        assert_eq!(dataset.count_deleted_rows().await.unwrap(), 0);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(1));
        assert!(fragments[0].metadata.deletion_file.is_none());
        assert!(fragments[1].metadata.deletion_file.is_none());

        // Delete rows
        dataset.delete("i < 10 OR i >= 90").await.unwrap();
        dataset.validate().await.unwrap();

        // Verify result:
        // There should be a deletion file in the metadata
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        assert_eq!(dataset.count_fragments(), 2);
        assert!(fragments[0].metadata.deletion_file.is_some());
        assert!(fragments[1].metadata.deletion_file.is_some());
        assert_eq!(
            fragments[0]
                .metadata
                .deletion_file
                .as_ref()
                .unwrap()
                .num_deleted_rows,
            Some(10)
        );
        assert_eq!(
            fragments[1]
                .metadata
                .deletion_file
                .as_ref()
                .unwrap()
                .num_deleted_rows,
            Some(10)
        );

        // The deletion file should contain 20 rows
        assert_eq!(dataset.count_deleted_rows().await.unwrap(), 20);
        let store = dataset.object_store().clone();
        let path = Path::from_filesystem_path(test_uri).unwrap();
        // First fragment has 0..10 deleted
        let deletion_vector = read_deletion_file(&path, &fragments[0].metadata, &store)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(deletion_vector.len(), 10);
        assert_eq!(
            deletion_vector.into_iter().collect::<HashSet<_>>(),
            (0..10).collect::<HashSet<_>>()
        );
        // Second fragment has 90..100 deleted
        let deletion_vector = read_deletion_file(&path, &fragments[1].metadata, &store)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(deletion_vector.len(), 10);
        // The second fragment starts at 50, so 90..100 becomes 40..50 in local row ids.
        assert_eq!(
            deletion_vector.into_iter().collect::<HashSet<_>>(),
            (40..50).collect::<HashSet<_>>()
        );
        let second_deletion_file = fragments[1].metadata.deletion_file.clone().unwrap();

        // Delete more rows
        dataset.delete("i < 20").await.unwrap();
        dataset.validate().await.unwrap();

        // Verify result
        assert_eq!(dataset.count_deleted_rows().await.unwrap(), 30);
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        assert!(fragments[0].metadata.deletion_file.is_some());
        let deletion_vector = read_deletion_file(&path, &fragments[0].metadata, &store)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(deletion_vector.len(), 20);
        assert_eq!(
            deletion_vector.into_iter().collect::<HashSet<_>>(),
            (0..20).collect::<HashSet<_>>()
        );
        // Second deletion vector was not rewritten
        assert_eq!(
            fragments[1].metadata.deletion_file.as_ref().unwrap(),
            &second_deletion_file
        );

        // Delete full fragment
        dataset.delete("i >= 50").await.unwrap();
        dataset.validate().await.unwrap();

        // Verify second fragment is fully gone
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert_eq!(dataset.count_fragments(), 1);
        assert_eq!(fragments[0].id(), 0);

        // Verify the count_deleted_rows only contains the rows from the first fragment
        // i.e. - deleted_rows from the fragment that has been deleted are not counted
        assert_eq!(dataset.count_deleted_rows().await.unwrap(), 20);

        // Append after delete
        let data = sequence_data(0..100);
        let batches = RecordBatchIterator::new(vec![data].into_iter().map(Ok), schema.clone());
        let write_params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        let dataset = Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        dataset.validate().await.unwrap();

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        assert_eq!(dataset.count_fragments(), 2);
        // Fragment id picks up where we left off
        assert_eq!(fragments[0].id(), 0);
        assert_eq!(fragments[1].id(), 2);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(2));
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

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(UInt32Array::from_iter_values(0..100))],
        );
        let reader = RecordBatchIterator::new(vec![data.unwrap()].into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(
            reader,
            test_uri,
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
    async fn test_update_config() {
        // Create a table
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::UInt32,
            false,
        )]));

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(UInt32Array::from_iter_values(0..100))],
        );
        let reader = RecordBatchIterator::new(vec![data.unwrap()].into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        let mut desired_config = HashMap::new();
        desired_config.insert("lance:test".to_string(), "value".to_string());
        desired_config.insert("other-key".to_string(), "other-value".to_string());

        dataset.update_config(desired_config.clone()).await.unwrap();
        assert_eq!(dataset.manifest.config, desired_config);

        desired_config.remove("other-key");
        dataset.delete_config_keys(&["other-key"]).await.unwrap();
        assert_eq!(dataset.manifest.config, desired_config);
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

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(UInt32Array::from_iter_values(0..100))],
        );
        let reader = RecordBatchIterator::new(vec![data.unwrap()].into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(
            reader,
            test_uri,
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

        assert_eq!(dataset.tags.list().await.unwrap().len(), 0);

        let bad_tag_creation = dataset.tags.create("tag1", 3).await;
        assert_eq!(
            bad_tag_creation.err().unwrap().to_string(),
            "Version not found error: version 3 does not exist"
        );

        let bad_tag_deletion = dataset.tags.delete("tag1").await;
        assert_eq!(
            bad_tag_deletion.err().unwrap().to_string(),
            "Ref not found error: tag tag1 does not exist"
        );

        dataset.tags.create("tag1", 1).await.unwrap();

        assert_eq!(dataset.tags.list().await.unwrap().len(), 1);

        let another_bad_tag_creation = dataset.tags.create("tag1", 1).await;
        assert_eq!(
            another_bad_tag_creation.err().unwrap().to_string(),
            "Ref conflict error: tag tag1 already exists"
        );

        dataset.tags.delete("tag1").await.unwrap();

        assert_eq!(dataset.tags.list().await.unwrap().len(), 0);

        dataset.tags.create("tag1", 1).await.unwrap();
        dataset.tags.create("tag2", 1).await.unwrap();
        dataset.tags.create("v1.0.0-rc1", 1).await.unwrap();

        assert_eq!(dataset.tags.list().await.unwrap().len(), 3);

        let bad_checkout = dataset.checkout_version("tag3").await;
        assert_eq!(
            bad_checkout.err().unwrap().to_string(),
            "Ref not found error: tag tag3 does not exist"
        );

        dataset = dataset.checkout_version("tag1").await.unwrap();
        assert_eq!(dataset.manifest.version, 1);

        let first_ver = DatasetBuilder::from_uri(test_uri)
            .with_tag("tag1")
            .load()
            .await
            .unwrap();
        assert_eq!(first_ver.version().version, 1);

        // test update tag
        let bad_tag_update = dataset.tags.update("tag3", 1).await;
        assert_eq!(
            bad_tag_update.err().unwrap().to_string(),
            "Ref not found error: tag tag3 does not exist"
        );

        let another_bad_tag_update = dataset.tags.update("tag1", 3).await;
        assert_eq!(
            another_bad_tag_update.err().unwrap().to_string(),
            "Version not found error: version 3 does not exist"
        );

        dataset.tags.update("tag1", 2).await.unwrap();
        dataset = dataset.checkout_version("tag1").await.unwrap();
        assert_eq!(dataset.manifest.version, 2);

        dataset.tags.update("tag1", 1).await.unwrap();
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

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

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
            test_uri,
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
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let data = gen().col("vec", array::rand_vec::<Float32Type>(Dimension::from(32)));
        let reader = data.into_reader_rows(RowCount::from(1000), BatchCount::from(10));
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                enable_move_stable_row_ids: use_stable_row_id,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let params = VectorIndexParams::ivf_pq(10, 8, 2, MetricType::L2, 50);
        dataset
            .create_index(&["vec"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        dataset.delete("true").await.unwrap();

        let indices = dataset.load_indices().await.unwrap();
        // Indices should be gone if it's fragments are deleted
        assert_eq!(indices.len(), 0);

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

        // predicate with redundant whitespace
        dataset.delete(" True").await.unwrap();

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
    async fn test_num_small_files(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = tempdir().unwrap();
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

        let test_uri = test_dir.path().to_str().unwrap();
        let dataset = Dataset::write(
            reader,
            test_uri,
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
        let test_dir = tempdir().unwrap();

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

        let test_uri = test_dir.path().to_str().unwrap();

        let batch_reader =
            RecordBatchIterator::new(batches.clone().into_iter().map(Ok), arrow_schema.clone());
        Dataset::write(batch_reader, test_uri, Some(WriteParams::default()))
            .await
            .unwrap();

        let result = scan_dataset(test_uri).await.unwrap();

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

    fn copy_dir_all(
        src: impl AsRef<std::path::Path>,
        dst: impl AsRef<std::path::Path>,
    ) -> std::io::Result<()> {
        use std::fs;
        fs::create_dir_all(&dst)?;
        for entry in fs::read_dir(src)? {
            let entry = entry?;
            let ty = entry.file_type()?;
            if ty.is_dir() {
                copy_dir_all(entry.path(), dst.as_ref().join(entry.file_name()))?;
            } else {
                fs::copy(entry.path(), dst.as_ref().join(entry.file_name()))?;
            }
        }
        Ok(())
    }

    /// Copies a test dataset into a temporary directory, returning the tmpdir.
    ///
    /// The `table_path` should be relative to `test_data/` at the root of the
    /// repo.
    fn copy_test_data_to_tmp(table_path: &str) -> std::io::Result<TempDir> {
        use std::path::PathBuf;

        let mut src = PathBuf::new();
        src.push(env!("CARGO_MANIFEST_DIR"));
        src.push("../../test_data");
        src.push(table_path);

        let test_dir = tempdir().unwrap();

        copy_dir_all(src.as_path(), test_dir.path())?;

        Ok(test_dir)
    }

    #[rstest]
    #[tokio::test]
    async fn test_v0_7_5_migration() {
        // We migrate to add Fragment.physical_rows and DeletionFile.num_deletions
        // after this version.

        // Copy over table
        let test_dir = copy_test_data_to_tmp("v0.7.5/with_deletions").unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        // Assert num rows, deletions, and physical rows are all correct.
        let dataset = Dataset::open(test_uri).await.unwrap();
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
        let dataset = Dataset::write(batches, test_uri, Some(write_params))
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
        let test_uri = test_dir.path().to_str().unwrap();

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
        let test_uri = test_dir.path().to_str().unwrap();

        let mut dataset = Dataset::open(test_uri).await.unwrap();

        // Uncomment to reproduce the issue.  The below query will panic
        // let mut scan = dataset.scan();
        // let query_vec = Float32Array::from(vec![0_f32; 128]);
        // let scan_fut = scan
        //     .nearest("vector", &query_vec, 2000)
        //     .unwrap()
        //     .nprobs(4)
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
            .nprobs(4)
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
        let test_uri = test_dir.path().to_str().unwrap();

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

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone()),
            test_uri,
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
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

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
            test_uri,
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
            test_uri,
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
        let test_dir = tempdir().unwrap();
        let base_dir = test_dir.path();
        let dataset_dir = base_dir.join("non_existing");
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

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, None).await.unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        assert_eq!(1000, dataset.count_rows(None).await.unwrap());
    }

    #[tokio::test]
    async fn test_dataset_uri_roundtrips() {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::Int32,
            false,
        )]));

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let vectors = Arc::new(Int32Array::from_iter_values(vec![]));

        let data = RecordBatch::try_new(schema.clone(), vec![vectors]);
        let reader = RecordBatchIterator::new(vec![data.unwrap()].into_iter().map(Ok), schema);
        let dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let uri = dataset.uri();
        assert_eq!(uri, test_uri);

        let ds2 = Dataset::open(uri).await.unwrap();
        assert_eq!(
            ds2.latest_version_id().await.unwrap(),
            dataset.latest_version_id().await.unwrap()
        );
    }

    #[tokio::test]
    async fn test_fts_on_multiple_columns() {
        let tempdir = tempfile::tempdir().unwrap();

        let params = InvertedIndexParams::default();
        let title_col =
            GenericStringArray::<i32>::from(vec!["title hello", "title lance", "title common"]);
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
        let mut dataset = Dataset::write(batches, tempdir.path().to_str().unwrap(), None)
            .await
            .unwrap();
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
    }

    #[tokio::test]
    async fn test_fts_unindexed_data() {
        let tempdir = tempfile::tempdir().unwrap();

        let params = InvertedIndexParams::default();
        let title_col =
            GenericStringArray::<i32>::from(vec!["title hello", "title lance", "title common"]);
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
        let mut dataset = Dataset::write(batches, tempdir.path().to_str().unwrap(), None)
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
        let title_col = GenericStringArray::<i32>::from(vec!["new title"]);
        let content_col = GenericStringArray::<i32>::from(vec!["new content"]);
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
        let dataset = Dataset::write(
            batches,
            tempdir.path().to_str().unwrap(),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

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
            let test_dir = tempdir().unwrap();
            let test_uri = test_dir.path().to_str().unwrap();

            let (res1, res2) = tokio::join!(write(test_uri), write(test_uri));

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
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        dataset.append(reader, None).await.unwrap();
        dataset.validate().await.unwrap();

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
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let field_a = Arc::new(ArrowField::new("a", DataType::Int32, true));
        let field_b = Arc::new(ArrowField::new("b", DataType::Int32, false));
        let field_c = Arc::new(ArrowField::new("c", DataType::Int32, true));
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "s",
            DataType::Struct(vec![field_a.clone(), field_b.clone(), field_c.clone()].into()),
            true,
        )]));
        let empty_reader = RecordBatchIterator::new(vec![], schema.clone());
        let dataset = Dataset::write(empty_reader, test_uri, None).await.unwrap();
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
        let dataset = Dataset::write(reader, test_uri, Some(append_options.clone()))
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
        let dataset = Dataset::write(reader, test_uri, Some(append_options.clone()))
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
        let res = Dataset::write(reader, test_uri, Some(append_options)).await;
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
        // TODO: support this.
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let field_a = ArrowField::new("a", DataType::Int32, true);
        let field_b = ArrowField::new("b", DataType::Int64, true);
        let schema = Arc::new(ArrowSchema::new(vec![
            field_a.clone(),
            field_b.clone().with_metadata(
                [(
                    LANCE_STORAGE_CLASS_SCHEMA_META_KEY.to_string(),
                    "blob".to_string(),
                )]
                .into(),
            ),
        ]));
        let empty_reader = RecordBatchIterator::new(vec![], schema.clone());
        let options = WriteParams {
            enable_move_stable_row_ids: true,
            enable_v2_manifest_paths: true,
            ..Default::default()
        };
        let mut dataset = Dataset::write(empty_reader, test_uri, Some(options))
            .await
            .unwrap();
        dataset.validate().await.unwrap();

        // Insert left side
        let just_a = Arc::new(ArrowSchema::new(vec![field_a.clone()]));
        let batch = RecordBatch::try_new(just_a.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
            .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], just_a.clone());
        let result = dataset.append(reader, None).await;
        assert!(result.is_err());
        assert!(matches!(result, Err(Error::SchemaMismatch { .. })));

        // Insert right side
        let just_b = Arc::new(ArrowSchema::new(vec![field_b.clone()]));
        let batch = RecordBatch::try_new(just_b.clone(), vec![Arc::new(Int64Array::from(vec![2]))])
            .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], just_b.clone());
        let result = dataset.append(reader, None).await;
        assert!(result.is_err());
        assert!(matches!(result, Err(Error::SchemaMismatch { .. })));
    }
}
