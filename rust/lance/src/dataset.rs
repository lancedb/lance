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
use lance_core::{datatypes::SchemaCompareOptions, traits::DatasetTakeRows};
use lance_datafusion::utils::{peek_reader_schema, reader_to_stream};
use lance_file::datatypes::populate_schema_dictionary;
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use lance_io::object_writer::ObjectWriter;
use lance_io::traits::WriteExt;
use lance_io::utils::{read_last_block, read_metadata_offset, read_struct};
use lance_table::format::{Fragment, Index, Manifest, MAGIC, MAJOR_VERSION, MINOR_VERSION};
use lance_table::io::commit::{
    commit_handler_from_url, CommitError, CommitHandler, CommitLock, ManifestLocation,
};
use lance_table::io::manifest::{read_manifest, write_manifest};
use log::warn;
use object_store::path::Path;
use prost::Message;
use snafu::{location, Location};
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use tracing::instrument;

pub mod builder;
pub mod cleanup;
pub mod fragment;
mod hash_joiner;
pub mod index;
pub mod optimize;
pub mod progress;
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
use self::scanner::{DatasetRecordBatchStream, Scanner};
use self::transaction::{Operation, Transaction};
use self::write::write_fragments_internal;
use crate::datatypes::Schema;
use crate::error::box_error;
use crate::io::commit::{commit_new_dataset, commit_transaction};
use crate::session::Session;
use crate::utils::temporal::{timestamp_to_nanos, utc_now, SystemTime};
use crate::{Error, Result};
use hash_joiner::HashJoiner;
pub use lance_core::ROW_ID;
use lance_table::feature_flags::{
    apply_feature_flags, can_read_dataset, can_write_dataset, should_use_legacy_format,
    FLAG_USE_V2_FORMAT,
};
pub use schema_evolution::{
    BatchInfo, BatchUDF, ColumnAlteration, NewColumnTransform, UDFCheckpointStore,
};
pub use write::merge_insert::{
    MergeInsertBuilder, MergeInsertJob, WhenMatched, WhenNotMatched, WhenNotMatchedBySource,
};
pub use write::update::{UpdateBuilder, UpdateJob};
pub use write::{write_fragments, WriteMode, WriteParams};

const INDICES_DIR: &str = "_indices";

pub const DATA_DIR: &str = "data";
pub(crate) const DEFAULT_INDEX_CACHE_SIZE: usize = 256;
pub(crate) const DEFAULT_METADATA_CACHE_SIZE: usize = 256;

/// Lance Dataset
#[derive(Debug, Clone)]
pub struct Dataset {
    pub(crate) object_store: Arc<ObjectStore>,
    pub(crate) commit_handler: Arc<dyn CommitHandler>,
    /// Uri of the dataset.
    ///
    /// On cloud storage, we can not use [Dataset::base] to build the full uri because the
    /// `bucket` is swlloed in the inner [ObjectStore].
    uri: String,
    pub(crate) base: Path,
    pub(crate) manifest: Arc<Manifest>,
    pub(crate) session: Arc<Session>,
}

/// Dataset Version
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
        }
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

    async fn params_from_uri(
        uri: &str,
        commit_handler: &Option<Arc<dyn CommitHandler>>,
        store_options: &Option<ObjectStoreParams>,
    ) -> Result<(ObjectStore, Path, Arc<dyn CommitHandler>)> {
        let (mut object_store, base_path) = match store_options.as_ref() {
            Some(store_options) => ObjectStore::from_uri_and_params(uri, store_options).await?,
            None => ObjectStore::from_uri(uri).await?,
        };

        if let Some(block_size) = store_options.as_ref().and_then(|opts| opts.block_size) {
            object_store.set_block_size(block_size);
        }

        let commit_handler = match &commit_handler {
            None => {
                if store_options.is_some() && store_options.as_ref().unwrap().object_store.is_some()
                {
                    return Err(Error::InvalidInput { source: "when creating a dataset with a custom object store the commit_handler must also be specified".into(), location: location!() });
                }
                commit_handler_from_url(uri, store_options).await?
            }
            Some(commit_handler) => {
                if uri.starts_with("s3+ddb") {
                    return Err(Error::InvalidInput {
                        source:
                            "`s3+ddb://` scheme and custom commit handler are mutually exclusive"
                                .into(),
                        location: location!(),
                    });
                } else {
                    commit_handler.clone()
                }
            }
        };

        Ok((object_store, base_path, commit_handler))
    }

    /// Open a dataset with read params.
    #[deprecated(since = "0.8.17", note = "Please use `DatasetBuilder` instead.")]
    #[instrument(skip(params))]
    pub async fn open_with_params(uri: &str, params: &ReadParams) -> Result<Self> {
        DatasetBuilder::from_uri(uri)
            .with_read_params(params.clone())
            .load()
            .await
    }

    /// Check out a version of the dataset.
    #[deprecated(note = "Please use `DatasetBuilder` instead.")]
    pub async fn checkout(uri: &str, version: u64) -> Result<Self> {
        DatasetBuilder::from_uri(uri)
            .with_version(version)
            .load()
            .await
    }

    /// Check out a version of the dataset with read params.
    #[deprecated(note = "Please use `DatasetBuilder` instead.")]
    pub async fn checkout_with_params(
        uri: &str,
        version: u64,
        params: &ReadParams,
    ) -> Result<Self> {
        DatasetBuilder::from_uri(uri)
            .with_version(version)
            .with_read_params(params.clone())
            .load()
            .await
    }

    /// Check out the specified version of this dataset
    pub async fn checkout_version(&self, version: u64) -> Result<Self> {
        let base_path = self.base.clone();
        let manifest_file = self
            .commit_handler
            .resolve_version(&base_path, version, &self.object_store.inner)
            .await?;
        let manifest_location = ManifestLocation {
            version,
            path: manifest_file,
            size: None,
        };
        let manifest = Self::load_manifest(self.object_store.as_ref(), &manifest_location).await?;
        Self::checkout_manifest(
            self.object_store.clone(),
            base_path,
            self.uri.clone(),
            manifest,
            self.session.clone(),
            self.commit_handler.clone(),
        )
        .await
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
    ) -> Result<Self> {
        Ok(Self {
            object_store,
            base: base_path,
            uri,
            manifest: Arc::new(manifest),
            commit_handler,
            session,
        })
    }

    #[instrument(skip(batches, params))]
    async fn write_impl(
        batches: Box<dyn RecordBatchReader + Send>,
        uri: &str,
        params: Option<WriteParams>,
    ) -> Result<Self> {
        let mut params = params.unwrap_or_default();
        let (object_store, base, mut commit_handler) =
            Self::params_from_uri(uri, &params.commit_handler, &params.store_params).await?;

        if let Some(commit_handler_wrapper) = params.commit_handler_wrapper.as_ref() {
            commit_handler = commit_handler_wrapper.wrap(commit_handler);
        }

        // Read expected manifest path for the dataset
        let dataset_exists = match commit_handler
            .resolve_latest_version(&base, &object_store)
            .await
        {
            Ok(_) => true,
            Err(Error::NotFound { .. }) => false,
            Err(e) => return Err(e),
        };

        let (batches, schema) = peek_reader_schema(Box::new(batches)).await?;
        let stream = reader_to_stream(batches);

        // Running checks for the different write modes
        // create + dataset already exists = error
        if dataset_exists && matches!(params.mode, WriteMode::Create) {
            return Err(Error::DatasetAlreadyExists {
                uri: uri.to_owned(),
                location: location!(),
            });
        }

        // append + dataset doesn't already exists = warn + switch to create mode
        if !dataset_exists
            && (matches!(params.mode, WriteMode::Append)
                || matches!(params.mode, WriteMode::Overwrite))
        {
            warn!("No existing dataset at {uri}, it will be created");
            params = WriteParams {
                mode: WriteMode::Create,
                ..params
            };
        }

        let dataset = if matches!(params.mode, WriteMode::Create) {
            None
        } else {
            // pull the store params from write params because there might be creds in there
            Some(
                DatasetBuilder::from_uri(uri)
                    .with_read_params(ReadParams {
                        store_options: params.store_params.clone(),
                        commit_handler: params.commit_handler.clone(),
                        ..Default::default()
                    })
                    .load()
                    .await?,
            )
        };

        // append + input schema different from existing schema = error
        if matches!(params.mode, WriteMode::Append) {
            if let Some(d) = dataset.as_ref() {
                let m = d.manifest.as_ref();
                schema.check_compatible(
                    &m.schema,
                    &SchemaCompareOptions {
                        compare_dictionary: true,
                        ..Default::default()
                    },
                )?;
                params.use_legacy_format = should_use_legacy_format(m.writer_feature_flags);
            }
        }

        let params = params; // discard mut

        if let Some(d) = dataset.as_ref() {
            if !can_write_dataset(d.manifest.writer_feature_flags) {
                let message = format!(
                    "This dataset cannot be written by this version of Lance. \
                Please upgrade Lance to write to this dataset.\n Flags: {}",
                    d.manifest.writer_feature_flags
                );
                return Err(Error::NotSupported {
                    source: message.into(),
                    location: location!(),
                });
            }
        }

        let object_store = Arc::new(object_store);
        let fragments = write_fragments_internal(
            dataset.as_ref(),
            object_store.clone(),
            &base,
            &schema,
            stream,
            params.clone(),
        )
        .await?;

        let operation = match params.mode {
            WriteMode::Create | WriteMode::Overwrite => Operation::Overwrite { schema, fragments },
            WriteMode::Append => Operation::Append { fragments },
        };

        let transaction = Transaction::new(
            dataset.as_ref().map(|ds| ds.manifest.version).unwrap_or(0),
            operation,
            None,
        );

        let manifest_config = ManifestWriteConfig {
            use_move_stable_row_ids: params.enable_move_stable_row_ids,
            use_legacy_format: Some(params.use_legacy_format),
            ..Default::default()
        };
        let manifest = if let Some(dataset) = &dataset {
            commit_transaction(
                dataset,
                &object_store,
                commit_handler.as_ref(),
                &transaction,
                &manifest_config,
                &Default::default(),
            )
            .await?
        } else {
            commit_new_dataset(
                &object_store,
                commit_handler.as_ref(),
                &base,
                &transaction,
                &manifest_config,
            )
            .await?
        };

        Ok(Self {
            object_store,
            base,
            uri: uri.to_string(),
            manifest: Arc::new(manifest.clone()),
            session: Arc::new(Session::default()),
            commit_handler,
        })
    }

    /// Write to or Create a [Dataset] with a stream of [RecordBatch]s.
    ///
    /// Returns the newly created [`Dataset`].
    /// Or Returns [Error] if the dataset already exists.
    ///
    pub async fn write(
        batches: impl RecordBatchReader + Send + 'static,
        uri: &str,
        params: Option<WriteParams>,
    ) -> Result<Self> {
        // Box it so we don't monomorphize for every one. We take the generic
        // parameter for API ergonomics.
        let batches = Box::new(batches);
        Self::write_impl(batches, uri, params).await
    }

    async fn append_impl(
        &mut self,
        batches: Box<dyn RecordBatchReader + Send>,
        params: Option<WriteParams>,
    ) -> Result<()> {
        // Force append mode
        let params = WriteParams {
            mode: WriteMode::Append,
            ..params.unwrap_or_default()
        };

        if params.commit_handler.is_some() || params.store_params.is_some() {
            return Err(Error::InvalidInput {
                source: "commit_handler / store_params should not be specified when calling append"
                    .into(),
                location: location!(),
            });
        }

        let (batches, schema) = peek_reader_schema(Box::new(batches)).await?;
        let stream = reader_to_stream(batches);

        // Return Error if append and input schema differ
        self.manifest.schema.check_compatible(
            &schema,
            &SchemaCompareOptions {
                compare_dictionary: true,
                ..Default::default()
            },
        )?;

        let fragments = write_fragments_internal(
            Some(self),
            self.object_store.clone(),
            &self.base,
            &schema,
            stream,
            params.clone(),
        )
        .await?;

        let transaction =
            Transaction::new(self.manifest.version, Operation::Append { fragments }, None);

        let new_manifest = commit_transaction(
            self,
            &self.object_store,
            self.commit_handler.as_ref(),
            &transaction,
            &Default::default(),
            &Default::default(),
        )
        .await?;

        self.manifest = Arc::new(new_manifest);

        Ok(())
    }

    /// Append to existing [Dataset] with a stream of [RecordBatch]s
    ///
    /// Returns void result or Returns [Error]
    pub async fn append(
        &mut self,
        batches: impl RecordBatchReader + Send + 'static,
        params: Option<WriteParams>,
    ) -> Result<()> {
        // Box it so we don't monomorphize for every one. We take the generic
        // parameter for API ergonomics.
        let batches = Box::new(batches);
        self.append_impl(batches, params).await
    }

    /// Get the fully qualified URI of this dataset.
    pub fn uri(&self) -> &str {
        &self.uri
    }

    /// Get the full manifest of the dataset version.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
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
    ) -> BoxFuture<Result<RemovalStats>> {
        let before = utc_now() - older_than;
        cleanup::cleanup_old_versions(self, before, delete_unverified).boxed()
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
    pub async fn commit(
        base_uri: &str,
        operation: Operation,
        read_version: Option<u64>,
        store_params: Option<ObjectStoreParams>,
        commit_handler: Option<Arc<dyn CommitHandler>>,
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

        let (object_store, base, commit_handler) =
            Self::params_from_uri(base_uri, &commit_handler, &store_params).await?;

        // Test if the dataset exists
        let dataset_exists = match commit_handler
            .resolve_latest_version(&base, &object_store)
            .await
        {
            Ok(_) => true,
            Err(Error::NotFound { .. }) => false,
            Err(e) => return Err(e),
        };

        if !dataset_exists && !matches!(operation, Operation::Overwrite { .. }) {
            return Err(Error::DatasetNotFound {
                path: base.to_string(),
                source: "The dataset must already exist unless the operation is Overwrite".into(),
                location: location!(),
            });
        }

        let dataset = if dataset_exists {
            Some(
                DatasetBuilder::from_uri(base_uri)
                    .with_read_params(ReadParams {
                        store_options: store_params.clone(),
                        ..Default::default()
                    })
                    .load()
                    .await?,
            )
        } else {
            None
        };

        let transaction = Transaction::new(read_version, operation, None);

        let manifest = if let Some(dataset) = &dataset {
            commit_transaction(
                dataset,
                &object_store,
                commit_handler.as_ref(),
                &transaction,
                &Default::default(),
                &Default::default(),
            )
            .await?
        } else {
            commit_new_dataset(
                &object_store,
                commit_handler.as_ref(),
                &base,
                &transaction,
                &Default::default(),
            )
            .await?
        };

        Ok(Self {
            object_store: Arc::new(object_store),
            base,
            uri: base_uri.to_string(),
            manifest: Arc::new(manifest.clone()),
            session: Arc::new(Session::default()),
            commit_handler,
        })
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
            None,
        );

        let manifest = commit_transaction(
            self,
            &self.object_store,
            self.commit_handler.as_ref(),
            &transaction,
            &Default::default(),
            &Default::default(),
        )
        .await?;

        self.manifest = Arc::new(manifest);

        Ok(())
    }

    pub async fn merge(
        &mut self,
        stream: impl RecordBatchReader + Send + 'static,
        left_on: &str,
        right_on: &str,
    ) -> Result<()> {
        let stream = Box::new(stream);
        self.merge_impl(stream, left_on, right_on).await
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
            let cnts = stream::iter(self.get_fragments())
                .map(|f| async move { f.count_rows().await })
                .buffer_unordered(16)
                .try_collect::<Vec<_>>()
                .await?;
            Ok(cnts.iter().sum())
        }
    }

    #[instrument(skip_all, fields(num_rows=row_indices.len()))]
    pub async fn take(&self, row_indices: &[u64], projection: &Schema) -> Result<RecordBatch> {
        take::take(self, row_indices, projection).await
    }

    /// Take rows by the internal ROW ids.
    pub async fn take_rows(&self, row_ids: &[u64], projection: &Schema) -> Result<RecordBatch> {
        take::take_rows(self, row_ids, projection).await
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
        self.take(&ids, projection).await
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
            .buffer_unordered(num_cpus::get())
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
            None,
        );

        let manifest = commit_transaction(
            self,
            &self.object_store,
            self.commit_handler.as_ref(),
            &transaction,
            &Default::default(),
            &Default::default(),
        )
        .await?;

        self.manifest = Arc::new(manifest);

        Ok(())
    }

    pub async fn count_deleted_rows(&self) -> Result<usize> {
        futures::stream::iter(self.get_fragments())
            .map(|f| async move { f.count_deletions().await })
            .buffer_unordered(num_cpus::get() * 4)
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

    pub fn schema(&self) -> &Schema {
        &self.manifest.schema
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

    /// Gets the number of files that are so small they don't even have a full
    /// group. These are considered too small because reading many of them is
    /// much less efficient than reading a single file because the separate files
    /// split up what would otherwise be single IO requests into multiple.
    pub async fn num_small_files(&self, max_rows_per_group: usize) -> usize {
        futures::stream::iter(self.get_fragments())
            .map(|f| async move { f.physical_rows().await })
            .buffered(num_cpus::get() * 4)
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

        // All fragments have equal lengths
        futures::stream::iter(self.get_fragments())
            .map(|f| async move { f.validate().await })
            .buffer_unordered(num_cpus::get() * 4)
            .try_collect::<Vec<()>>()
            .await?;

        Ok(())
    }
}

/// # Schema Evolution
impl Dataset {
    /// Append new columns to the dataset.
    pub async fn add_columns(
        &mut self,
        transforms: NewColumnTransform,
        read_columns: Option<Vec<String>>,
    ) -> Result<()> {
        schema_evolution::add_columns(self, transforms, read_columns).await
    }

    /// Modify columns in the dataset, changing their name, type, or nullability.
    ///
    /// If a column has an index, it's index will be preserved.
    pub async fn alter_columns(&mut self, alterations: &[ColumnAlteration]) -> Result<()> {
        schema_evolution::alter_columns(self, alterations).await
    }

    /// Remove columns from the dataset.
    ///
    /// This is a metadata-only operation and does not remove the data from the
    /// underlying storage. In order to remove the data, you must subsequently
    /// call `compact_files` to rewrite the data without the removed columns and
    /// then call `cleanup_files` to remove the old files.
    pub async fn drop_columns(&mut self, columns: &[&str]) -> Result<()> {
        schema_evolution::drop_columns(self, columns).await
    }
}

#[async_trait::async_trait]
impl DatasetTakeRows for Dataset {
    fn schema(&self) -> &Schema {
        Self::schema(self)
    }

    async fn take_rows(&self, row_ids: &[u64], projection: &Schema) -> Result<RecordBatch> {
        Self::take_rows(self, row_ids, projection).await
    }
}

#[derive(Debug)]
pub(crate) struct ManifestWriteConfig {
    auto_set_feature_flags: bool,    // default true
    timestamp: Option<SystemTime>,   // default None
    use_move_stable_row_ids: bool,   // default false
    use_legacy_format: Option<bool>, // default None
}

impl Default for ManifestWriteConfig {
    fn default() -> Self {
        Self {
            auto_set_feature_flags: true,
            timestamp: None,
            use_move_stable_row_ids: false,
            use_legacy_format: None,
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
) -> std::result::Result<(), CommitError> {
    let was_using_legacy = should_use_legacy_format(manifest.writer_feature_flags);
    if config.auto_set_feature_flags {
        apply_feature_flags(manifest, config.use_move_stable_row_ids)?;
    }
    // For now, we don't auto-detect use_v2_format.  Instead, if the user
    // asks for it, we set it.  Otherwise we use what was there before.
    if let Some(use_legacy_format) = config.use_legacy_format {
        if !use_legacy_format {
            manifest.writer_feature_flags |= FLAG_USE_V2_FORMAT;
        }
    } else if was_using_legacy {
        manifest.writer_feature_flags &= !FLAG_USE_V2_FORMAT;
    } else {
        manifest.writer_feature_flags |= FLAG_USE_V2_FORMAT;
    }
    manifest.set_timestamp(timestamp_to_nanos(config.timestamp));

    manifest.update_max_fragment_id();

    commit_handler
        .commit(
            manifest,
            indices,
            base_path,
            &object_store.inner,
            write_manifest_file_to_path,
        )
        .await?;

    Ok(())
}

fn write_manifest_file_to_path<'a>(
    object_store: &'a dyn object_store::ObjectStore,
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
    use crate::index::scalar::ScalarIndexParams;
    use crate::index::vector::VectorIndexParams;
    use crate::utils::test::TestDatasetGenerator;

    use arrow::array::as_struct_array;
    use arrow::compute::concat_batches;
    use arrow_array::{
        builder::StringDictionaryBuilder, cast::as_string_array, types::Int32Type, ArrayRef,
        DictionaryArray, Float32Array, Int32Array, Int64Array, Int8Array, Int8DictionaryArray,
        RecordBatchIterator, StringArray, UInt16Array, UInt32Array,
    };
    use arrow_array::{FixedSizeListArray, Int16Array, Int16DictionaryArray, StructArray};
    use arrow_ord::sort::sort_to_indices;
    use arrow_schema::{
        DataType, Field as ArrowField, Fields as ArrowFields, Schema as ArrowSchema,
    };
    use lance_arrow::bfloat16::{self, ARROW_EXT_META_KEY, ARROW_EXT_NAME_KEY, BFLOAT16_EXT_NAME};
    use lance_datagen::{array, gen, BatchCount, RowCount};
    use lance_index::{vector::DIST_COL, DatasetIndexExt, IndexType};
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

    async fn create_file(path: &std::path::Path, mode: WriteMode, use_legacy_format: bool) {
        let mut fields = vec![ArrowField::new("i", DataType::Int32, false)];
        // TODO (GH-2347): currently the v2 writer does not support dictionary columns.
        if use_legacy_format {
            fields.push(ArrowField::new(
                "dict",
                DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
                false,
            ));
        }
        let schema = Arc::new(ArrowSchema::new(fields));
        let dict_values = StringArray::from_iter_values(["a", "b", "c", "d", "e"]);
        let batches: Vec<RecordBatch> = (0..20)
            .map(|i| {
                let mut arrays =
                    vec![Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)) as ArrayRef];
                if use_legacy_format {
                    arrays.push(Arc::new(
                        DictionaryArray::try_new(
                            UInt16Array::from_iter_values((0_u16..20_u16).map(|v| v % 5)),
                            Arc::new(dict_values.clone()),
                        )
                        .unwrap(),
                    ));
                }
                RecordBatch::try_new(schema.clone(), arrays).unwrap()
            })
            .collect();
        let expected_batches = batches.clone();

        let test_uri = path.to_str().unwrap();
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            mode,
            use_legacy_format,
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
        if use_legacy_format {
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
    async fn test_create_dataset(#[values(false, true)] use_legacy_format: bool) {
        // Appending / Overwriting a dataset that does not exist is treated as Create
        for mode in [WriteMode::Create, WriteMode::Append, Overwrite] {
            let test_dir = tempdir().unwrap();
            create_file(test_dir.path(), mode, use_legacy_format).await
        }
    }

    #[rstest]
    #[lance_test_macros::test(tokio::test)]
    async fn test_create_and_fill_empty_dataset(#[values(false, true)] use_legacy_format: bool) {
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
        let result = Dataset::write(reader, test_uri, None).await.unwrap();

        // check dataset empty
        assert_eq!(result.count_rows(None).await.unwrap(), 0);
        // Since the dataset is empty, will return None.
        assert_eq!(result.manifest.max_fragment_id(), None);

        // append rows to dataset
        let mut write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            use_legacy_format,
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
    async fn test_create_with_empty_iter(#[values(false, true)] use_legacy_format: bool) {
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
            use_legacy_format,
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
    async fn test_write_params(#[values(false, true)] use_legacy_format: bool) {
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
            use_legacy_format,
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
            let reader = fragment.open(dataset.schema(), false, false).await.unwrap();
            // No group / batch concept in v2
            if use_legacy_format {
                assert_eq!(reader.legacy_num_batches(), 10);
                for i in 0..reader.legacy_num_batches() as u32 {
                    assert_eq!(reader.legacy_num_rows_in_batch(i).unwrap(), 10);
                }
            }
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_write_manifest(#[values(false, true)] use_legacy_format: bool) {
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
                use_legacy_format,
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
        if !use_legacy_format {
            assert_eq!(manifest.writer_feature_flags, FLAG_USE_V2_FORMAT);
        }
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
            },
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
                use_legacy_format,
                ..Default::default()
            }),
        )
        .await;

        assert!(matches!(write_result, Err(Error::NotSupported { .. })));
    }

    #[rstest]
    #[tokio::test]
    async fn append_dataset(#[values(false, true)] use_legacy_format: bool) {
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
            use_legacy_format,
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
    async fn test_self_dataset_append(#[values(false, true)] use_legacy_format: bool) {
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
            use_legacy_format,
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
        #[values(false, true)] use_legacy_format: bool,
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
            use_legacy_format,
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
    async fn overwrite_dataset(#[values(false, true)] use_legacy_format: bool) {
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
            use_legacy_format,
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
    async fn test_fast_count_rows(#[values(false, true)] use_legacy_format: bool) {
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
            use_legacy_format,
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
    async fn test_create_index(#[values(false, true)] use_legacy_format: bool) {
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
                use_legacy_format,
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
            use_legacy_format,
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
        assert_eq!(actual_statistics["index_type"].as_str().unwrap(), "IVF");

        let deltas = actual_statistics["indices"].as_array().unwrap();
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0]["metric_type"].as_str().unwrap(), "l2");
        assert_eq!(deltas[0]["num_partitions"].as_i64().unwrap(), 10);

        assert!(dataset.index_statistics("non-existent_idx").await.is_err());
        assert!(dataset.index_statistics("").await.is_err());

        // Overwrite should invalidate index
        let write_params = WriteParams {
            mode: WriteMode::Overwrite,
            use_legacy_format,
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
        #[values(false, true)] use_legacy_format: bool,
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
                use_legacy_format,
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

    async fn create_bad_file(use_legacy_format: bool) -> Result<Dataset> {
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
                use_legacy_format,
                ..Default::default()
            }),
        )
        .await
    }

    #[rstest]
    #[tokio::test]
    async fn test_bad_field_name(#[values(false, true)] use_legacy_format: bool) {
        // don't allow `.` in the field name
        assert!(create_bad_file(use_legacy_format).await.is_err());
    }

    #[tokio::test]
    async fn test_open_dataset_not_found() {
        let result = Dataset::open(".").await;
        assert!(matches!(result.unwrap_err(), Error::DatasetNotFound { .. }));
    }

    #[rstest]
    #[tokio::test]
    async fn test_merge(
        #[values(false, true)] use_legacy_format: bool,
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
            use_legacy_format,
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
        #[values(false, true)] use_legacy_format: bool,
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
            use_legacy_format,
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
    async fn test_delete(#[values(false, true)] use_legacy_format: bool) {
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
        let mut dataset = TestDatasetGenerator::new(batches, use_legacy_format)
            .make_hostile(test_uri)
            .await;

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
    async fn test_restore(#[values(false, true)] use_legacy_format: bool) {
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
                use_legacy_format,
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
    async fn test_search_empty(#[values(false, true)] use_legacy_format: bool) {
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
                use_legacy_format,
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
        #[values(false, true)] use_legacy_format: bool,
        #[values(false, true)] use_stable_row_id: bool,
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
                Float32Array::from_iter_values((0..128).map(|_| 0.1_f32)),
                128,
            )
            .unwrap(),
        );

        let data = RecordBatch::try_new(schema.clone(), vec![vectors]);
        let reader = RecordBatchIterator::new(vec![data.unwrap()].into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                use_legacy_format,
                enable_move_stable_row_ids: use_stable_row_id,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset.delete("true").await.unwrap();

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
    async fn test_num_small_files(#[values(false, true)] use_legacy_format: bool) {
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
                use_legacy_format,
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
    async fn test_v0_7_5_migration(#[values(false, true)] use_legacy_format: bool) {
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
            use_legacy_format,
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
    async fn test_fix_v0_8_0_broken_migration(#[values(false, true)] use_legacy_format: bool) {
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
            use_legacy_format,
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
        #[values(false, true)] use_legacy_format: bool,
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
                    use_legacy_format,
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
    async fn test_bfloat16_roundtrip(#[values(false, true)] use_legacy_format: bool) -> Result<()> {
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
                use_legacy_format,
                ..Default::default()
            }),
        )
        .await?;

        let data = dataset.scan().try_into_batch().await?;
        assert_eq!(batch, data);

        Ok(())
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
}
