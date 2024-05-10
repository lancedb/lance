// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Dataset
//!

use arrow::compute::CastOptions;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::Array;
use arrow_array::{
    cast::as_struct_array, RecordBatch, RecordBatchReader, StructArray, UInt64Array,
};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow_select::interleave::interleave;
use arrow_select::{concat::concat_batches, take::take};
use chrono::{prelude::*, Duration};
use datafusion::error::DataFusionError;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::future::BoxFuture;
use futures::stream::{self, StreamExt, TryStreamExt};
use futures::{Future, FutureExt, Stream};
use lance_arrow::SchemaExt;
use lance_core::datatypes::{Field, SchemaCompareOptions};
use lance_datafusion::utils::{peek_reader_schema, reader_to_stream};
use lance_file::datatypes::populate_schema_dictionary;
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use lance_io::object_writer::ObjectWriter;
use lance_io::traits::WriteExt;
use lance_io::utils::{read_metadata_offset, read_struct};
use lance_table::format::{Fragment, Index, Manifest, MAGIC, MAJOR_VERSION, MINOR_VERSION};
use lance_table::io::commit::{commit_handler_from_url, CommitError, CommitHandler, CommitLock};
use lance_table::io::manifest::{read_manifest, write_manifest};
use log::warn;
use object_store::path::Path;
use prost::Message;
use snafu::{location, Location};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use tracing::instrument;

pub mod builder;
pub mod cleanup;
mod feature_flags;
pub mod fragment;
mod hash_joiner;
pub mod index;
pub mod optimize;
pub mod progress;
pub mod scanner;
pub mod transaction;
pub mod updater;
mod utils;
mod write;

use self::builder::DatasetBuilder;
use self::cleanup::RemovalStats;
use self::feature_flags::{apply_feature_flags, can_read_dataset, can_write_dataset};
use self::fragment::FileFragment;
use self::scanner::{DatasetRecordBatchStream, Scanner};
use self::transaction::{Operation, Transaction};
use self::write::write_fragments_internal;
use crate::datatypes::Schema;
use crate::error::box_error;
use crate::io::commit::{commit_new_dataset, commit_transaction};
use crate::io::exec::Planner;
use crate::session::Session;
use crate::utils::temporal::{timestamp_to_nanos, utc_now, SystemTime};
use crate::{Error, Result};
use hash_joiner::HashJoiner;
pub use lance_core::ROW_ID;
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
        let (object_store, base_path, commit_handler) =
            Self::params_from_uri(uri, &params.commit_handler, &params.store_options).await?;

        let latest_manifest = commit_handler
            .resolve_latest_version(&base_path, &object_store.inner)
            .await
            .map_err(|e| Error::DatasetNotFound {
                path: base_path.to_string(),
                source: Box::new(e),
                location: location!(),
            })?;

        let session = if let Some(session) = params.session.as_ref() {
            session.clone()
        } else {
            Arc::new(Session::new(
                params.index_cache_size,
                params.metadata_cache_size,
            ))
        };

        Self::checkout_manifest(
            Arc::new(object_store),
            base_path.clone(),
            &latest_manifest,
            session,
            commit_handler,
        )
        .await
    }

    /// Check out a version of the dataset.
    pub async fn checkout(uri: &str, version: u64) -> Result<Self> {
        let params = ReadParams::default();
        Self::checkout_with_params(uri, version, &params).await
    }

    /// Check out a version of the dataset with read params.
    pub async fn checkout_with_params(
        uri: &str,
        version: u64,
        params: &ReadParams,
    ) -> Result<Self> {
        let (object_store, base_path, commit_handler) =
            Self::params_from_uri(uri, &params.commit_handler, &params.store_options).await?;

        let manifest_file = commit_handler
            .resolve_version(&base_path, version, &object_store.inner)
            .await?;

        let session = if let Some(session) = params.session.as_ref() {
            session.clone()
        } else {
            Arc::new(Session::new(
                params.index_cache_size,
                params.metadata_cache_size,
            ))
        };
        Self::checkout_manifest(
            Arc::new(object_store),
            base_path,
            &manifest_file,
            session,
            commit_handler,
        )
        .await
    }

    /// Check out the specified version of this dataset
    pub async fn checkout_version(&self, version: u64) -> Result<Self> {
        let base_path = self.base.clone();
        let manifest_file = self
            .commit_handler
            .resolve_version(&base_path, version, &self.object_store.inner)
            .await?;
        Self::checkout_manifest(
            self.object_store.clone(),
            base_path,
            &manifest_file,
            self.session.clone(),
            self.commit_handler.clone(),
        )
        .await
    }

    async fn checkout_manifest(
        object_store: Arc<ObjectStore>,
        base_path: Path,
        manifest_path: &Path,
        session: Arc<Session>,
        commit_handler: Arc<dyn CommitHandler>,
    ) -> Result<Self> {
        let object_reader = object_store
            .open(manifest_path)
            .await
            .map_err(|e| match &e {
                Error::NotFound { uri, .. } => Error::DatasetNotFound {
                    path: uri.clone(),
                    source: box_error(e),
                    location: location!(),
                },
                _ => e,
            })?;
        // TODO: remove reference to inner.
        let get_result = object_store
            .inner
            .get(manifest_path)
            .await
            .map_err(|e| match e {
                object_store::Error::NotFound { path: _, source } => Error::DatasetNotFound {
                    path: base_path.to_string(),
                    source,
                    location: location!(),
                },
                _ => e.into(),
            })?;
        let bytes = get_result.bytes().await?;
        let offset = read_metadata_offset(&bytes)?;
        let mut manifest: Manifest = read_struct(object_reader.as_ref(), offset).await?;

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
        Ok(Self {
            object_store,
            base: base_path,
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
        let (object_store, base, commit_handler) =
            Self::params_from_uri(uri, &params.commit_handler, &params.store_params).await?;

        // Read expected manifest path for the dataset
        let dataset_exists = match commit_handler
            .resolve_latest_version(&base, &object_store.inner)
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
        let params = params; // discard mut

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
            }
        }

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
            object_store,
            base,
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

    /// Get the base URI of the dataset.
    pub fn uri(&self) -> &Path {
        &self.base
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
                .resolve_latest_version(&self.base, &self.object_store.inner)
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
        Transaction::try_from(&transaction).map(Some)
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
            .resolve_latest_version(&base, &object_store.inner)
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
        if row_indices.is_empty() {
            let schema = Arc::new(projection.into());
            return Ok(RecordBatch::new_empty(schema));
        }

        let mut sorted_indices: Vec<usize> = (0..row_indices.len()).collect();
        sorted_indices.sort_by_key(|&i| row_indices[i]);

        let fragments = self.get_fragments();

        // We will split into sub-requests for each fragment.
        let mut sub_requests: Vec<(&FileFragment, Range<usize>)> = Vec::new();
        // We will remap the row indices to the original row indices, using a pair
        // of (request position, position in request)
        let mut remap_index: Vec<(usize, usize)> = vec![(0, 0); row_indices.len()];
        let mut local_ids_buffer: Vec<u32> = Vec::with_capacity(row_indices.len());

        let mut fragments_iter = fragments.iter();
        let mut current_fragment = fragments_iter.next().ok_or_else(|| Error::InvalidInput {
            source: "Called take on an empty dataset.".to_string().into(),
            location: location!(),
        })?;
        let mut current_fragment_len = current_fragment.count_rows().await?;
        let mut curr_fragment_offset: u64 = 0;
        let mut current_fragment_end = current_fragment_len as u64;
        let mut start = 0;
        let mut end = 0;
        // We want to keep track of the previous row_index to detect duplicates
        // index takes. To start, we pick a value that is guaranteed to be different
        // from the first row_index.
        let mut previous_row_index: u64 = row_indices[sorted_indices[0]] + 1;
        let mut previous_sorted_index: usize = 0;

        for index in sorted_indices {
            // Get the index
            let row_index = row_indices[index];

            if previous_row_index == row_index {
                // If we have a duplicate index request we add a remap_index
                // entry that points to the original index request.
                remap_index[index] = remap_index[previous_sorted_index];
                continue;
            } else {
                previous_sorted_index = index;
                previous_row_index = row_index;
            }

            // If the row index is beyond the current fragment, iterate
            // until we find the fragment that contains it.
            while row_index >= current_fragment_end {
                // If we have a non-empty sub-request, add it to the list
                if end - start > 0 {
                    // If we have a non-empty sub-request, add it to the list
                    sub_requests.push((current_fragment, start..end));
                }

                start = end;

                current_fragment = fragments_iter.next().ok_or_else(|| Error::InvalidInput {
                    source: format!(
                        "Row index {} is beyond the range of the dataset.",
                        row_index
                    )
                    .into(),
                    location: location!(),
                })?;
                curr_fragment_offset += current_fragment_len as u64;
                current_fragment_len = current_fragment.count_rows().await?;
                current_fragment_end = curr_fragment_offset + current_fragment_len as u64;
            }

            // Note that we cast to u32 *after* subtracting the offset,
            // since it is possible for the global index to be larger than
            // u32::MAX.
            let local_index = (row_index - curr_fragment_offset) as u32;
            local_ids_buffer.push(local_index);

            remap_index[index] = (sub_requests.len(), end - start);

            end += 1;
        }

        // flush last batch
        if end - start > 0 {
            sub_requests.push((current_fragment, start..end));
        }

        let take_tasks = sub_requests
            .into_iter()
            .map(|(fragment, indices_range)| {
                let local_ids = &local_ids_buffer[indices_range];
                fragment.take(local_ids, projection)
            })
            .collect::<Vec<_>>();
        let batches = stream::iter(take_tasks)
            .buffered(num_cpus::get() * 4)
            .try_collect::<Vec<RecordBatch>>()
            .await?;

        let struct_arrs: Vec<StructArray> = batches.into_iter().map(StructArray::from).collect();
        let refs: Vec<_> = struct_arrs.iter().map(|x| x as &dyn Array).collect();
        let reordered = interleave(&refs, &remap_index)?;
        Ok(as_struct_array(&reordered).into())
    }

    /// Take rows by the internal ROW ids.
    pub async fn take_rows(&self, row_ids: &[u64], projection: &Schema) -> Result<RecordBatch> {
        if row_ids.is_empty() {
            return Ok(RecordBatch::new_empty(Arc::new(projection.into())));
        }

        let projection = Arc::new(projection.clone());
        let row_id_meta = check_row_ids(row_ids);

        // This method is mostly to annotate the send bound to avoid the
        // higher-order lifetime error.
        // manually implemented async for Send bound
        #[allow(clippy::manual_async_fn)]
        fn do_take(
            fragment: FileFragment,
            row_ids: Vec<u32>,
            projection: Arc<Schema>,
            with_row_id: bool,
        ) -> impl Future<Output = Result<RecordBatch>> + Send {
            async move {
                fragment
                    .take_rows(&row_ids, projection.as_ref(), with_row_id)
                    .await
            }
        }

        if row_id_meta.contiguous {
            // Fastest path: Can use `read_range` directly
            let start = row_ids.first().expect("empty range passed to take_rows");
            let fragment_id = (start >> 32) as usize;
            let range_start = *start as u32 as usize;
            let range_end =
                *row_ids.last().expect("empty range passed to take_rows") as u32 as usize;
            let range = range_start..(range_end + 1);

            let fragment = self.get_fragment(fragment_id).ok_or_else(|| {
                Error::invalid_input(
                    format!("row_id belongs to non-existant fragment: {start}"),
                    location!(),
                )
            })?;

            let reader = fragment.open(projection.as_ref(), false).await?;
            reader.legacy_read_range_as_batch(range).await
        } else if row_id_meta.sorted {
            // Don't need to re-arrange data, just concatenate

            let mut batches: Vec<_> = Vec::new();
            let mut current_fragment = row_ids[0] >> 32;
            let mut current_start = 0;
            let mut row_ids_iter = row_ids.iter().enumerate();
            'outer: loop {
                let (fragment_id, range) = loop {
                    if let Some((i, row_id)) = row_ids_iter.next() {
                        let fragment_id = row_id >> 32;
                        if fragment_id != current_fragment {
                            let next = (current_fragment, current_start..i);
                            current_fragment = fragment_id;
                            current_start = i;
                            break next;
                        }
                    } else if current_start != row_ids.len() {
                        let next = (current_fragment, current_start..row_ids.len());
                        current_start = row_ids.len();
                        break next;
                    } else {
                        break 'outer;
                    }
                };

                let fragment = self.get_fragment(fragment_id as usize).ok_or_else(|| {
                    Error::invalid_input(
                        format!(
                            "row_id {} belongs to non-existant fragment: {}",
                            row_ids[range.start], fragment_id
                        ),
                        location!(),
                    )
                })?;
                let row_ids: Vec<u32> = row_ids[range].iter().map(|x| *x as u32).collect();

                let batch_fut = do_take(fragment, row_ids, projection.clone(), false);
                batches.push(batch_fut);
            }
            let batches: Vec<RecordBatch> = futures::stream::iter(batches)
                .buffered(4 * num_cpus::get())
                .try_collect()
                .await?;
            Ok(concat_batches(&batches[0].schema(), &batches)?)
        } else {
            let projection_with_row_id = Schema::merge(
                projection.as_ref(),
                &ArrowSchema::new(vec![ArrowField::new(
                    ROW_ID,
                    arrow::datatypes::DataType::UInt64,
                    false,
                )]),
            )?;
            let schema_with_row_id = Arc::new(ArrowSchema::from(&projection_with_row_id));

            // Slow case: need to re-map data into expected order
            let mut sorted_row_ids = Vec::from(row_ids);
            sorted_row_ids.sort();
            // Group ROW Ids by the fragment
            let mut row_ids_per_fragment: BTreeMap<u64, Vec<u32>> = BTreeMap::new();
            sorted_row_ids.iter().for_each(|row_id| {
                let fragment_id = row_id >> 32;
                let offset = (row_id - (fragment_id << 32)) as u32;
                row_ids_per_fragment
                    .entry(fragment_id)
                    .and_modify(|v| v.push(offset))
                    .or_insert_with(|| vec![offset]);
            });

            let fragments = self.get_fragments();
            let fragment_and_indices = fragments.into_iter().filter_map(|f| {
                let local_row_ids = row_ids_per_fragment.remove(&(f.id() as u64))?;
                Some((f, local_row_ids))
            });

            let mut batches = stream::iter(fragment_and_indices)
                .map(|(fragment, indices)| do_take(fragment, indices, projection.clone(), true))
                .buffered(4 * num_cpus::get())
                .try_collect::<Vec<_>>()
                .await?;

            let one_batch = if batches.len() > 1 {
                concat_batches(&schema_with_row_id, &batches)?
            } else {
                batches.pop().unwrap()
            };
            // Note: one_batch may contains fewer rows than the number of requested
            // row ids because some rows may have been deleted. Because of this, we
            // get the results with row ids so that we can re-order the results
            // to match the requested order.

            let returned_row_ids = one_batch
                .column_by_name(ROW_ID)
                .ok_or_else(|| Error::Internal {
                    message: "ROW_ID column not found".into(),
                    location: location!(),
                })?
                .as_primitive::<UInt64Type>()
                .values();

            let remapping_index: UInt64Array = row_ids
                .iter()
                .filter_map(|o| {
                    returned_row_ids
                        .iter()
                        .position(|id| id == o)
                        .map(|pos| pos as u64)
                })
                .collect();

            debug_assert_eq!(remapping_index.len(), one_batch.num_rows());

            // Remove the row id column.
            let keep_indices = (0..one_batch.num_columns() - 1).collect::<Vec<_>>();
            let one_batch = one_batch.project(&keep_indices)?;
            let struct_arr: StructArray = one_batch.into();
            let reordered = take(&struct_arr, &remapping_index, None)?;
            Ok(as_struct_array(&reordered).into())
        }
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
        let arrow_schema = Arc::new(projection.as_ref().into());
        let dataset = Arc::new(self.clone());
        let batch_stream = row_ranges
            .map(move |res| {
                let dataset = dataset.clone();
                let projection = projection.clone();
                let fut = async move {
                    let range = res.map_err(|err| DataFusionError::External(Box::new(err)))?;
                    let row_pos: Vec<u64> = (range.start..range.end).collect();
                    dataset
                        .take(&row_pos, projection.as_ref())
                        .await
                        .map_err(|err| DataFusionError::External(Box::new(err)))
                };
                async move { tokio::task::spawn(fut).await.unwrap() }
            })
            .buffered(batch_readahead);

        DatasetRecordBatchStream::new(Box::pin(RecordBatchStreamAdapter::new(
            arrow_schema,
            batch_stream,
        )))
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
            .resolve_latest_version_id(&self.base, &self.object_store.inner)
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

#[derive(Debug, Clone, PartialEq)]
pub struct BatchInfo {
    pub fragment_id: u32,
    pub batch_index: usize,
}

/// A mechanism for saving UDF results.
///
/// This is used to determine if a UDF has already been run on a given input,
/// and to store the results of a UDF for future use.
pub trait UDFCheckpointStore: Send + Sync {
    fn get_batch(&self, info: &BatchInfo) -> Result<Option<RecordBatch>>;
    fn insert_batch(&self, info: BatchInfo, batch: RecordBatch) -> Result<()>;
    fn get_fragment(&self, fragment_id: u32) -> Result<Option<Fragment>>;
    fn insert_fragment(&self, fragment: Fragment) -> Result<()>;
}

pub struct BatchUDF {
    #[allow(clippy::type_complexity)]
    pub mapper: Box<dyn Fn(&RecordBatch) -> Result<RecordBatch> + Send + Sync>,
    /// The schema of the returned RecordBatch
    pub output_schema: Arc<ArrowSchema>,
    /// A checkpoint store for the UDF results
    pub result_checkpoint: Option<Arc<dyn UDFCheckpointStore>>,
}

/// A way to define one or more new columns in a dataset
pub enum NewColumnTransform {
    /// A UDF that takes a RecordBatch of existing data and returns a
    /// RecordBatch with the new columns for those corresponding rows. The returned
    /// batch must return the same number of rows as the input batch.
    BatchUDF(BatchUDF),
    /// A set of SQL expressions that define new columns.
    SqlExpressions(Vec<(String, String)>),
}

/// Definition of a change to a column in a dataset
pub struct ColumnAlteration {
    /// Path to the existing column to be altered.
    pub path: String,
    /// The new name of the column. If None, the column name will not be changed.
    pub rename: Option<String>,
    /// Whether the column is nullable. If None, the nullability will not be changed.
    pub nullable: Option<bool>,
    /// The new data type of the column. If None, the data type will not be changed.
    pub data_type: Option<DataType>,
}

impl ColumnAlteration {
    pub fn new(path: String) -> Self {
        Self {
            path,
            rename: None,
            nullable: None,
            data_type: None,
        }
    }

    pub fn rename(mut self, name: String) -> Self {
        self.rename = Some(name);
        self
    }

    pub fn set_nullable(mut self, nullable: bool) -> Self {
        self.nullable = Some(nullable);
        self
    }

    pub fn cast_to(mut self, data_type: DataType) -> Self {
        self.data_type = Some(data_type);
        self
    }
}

/// Limit casts to same type. This is mostly to filter out weird casts like
/// casting a string to a boolean or float to string.
fn is_upcast_downcast(from_type: &DataType, to_type: &DataType) -> bool {
    use DataType::*;
    match from_type {
        from_type if from_type.is_integer() => to_type.is_integer(),
        from_type if from_type.is_floating() => to_type.is_floating(),
        from_type if from_type.is_temporal() => to_type.is_temporal(),
        Boolean => matches!(to_type, Boolean),
        Utf8 | LargeUtf8 => matches!(to_type, Utf8 | LargeUtf8),
        Binary | LargeBinary => matches!(to_type, Binary | LargeBinary),
        Decimal128(_, _) | Decimal256(_, _) => {
            matches!(to_type, Decimal128(_, _) | Decimal256(_, _))
        }
        List(from_field) | LargeList(from_field) | FixedSizeList(from_field, _) => match to_type {
            List(to_field) | LargeList(to_field) | FixedSizeList(to_field, _) => {
                is_upcast_downcast(from_field.data_type(), to_field.data_type())
            }
            _ => false,
        },
        _ => false,
    }
}

// TODO: move all schema evolution methods to this impl and provide a dedicated
// docs section to describe the schema evolution methods.
impl Dataset {
    /// Append new columns to the dataset.
    pub async fn add_columns(
        &mut self,
        transforms: NewColumnTransform,
        read_columns: Option<Vec<String>>,
    ) -> Result<()> {
        // We just transform the SQL expression into a UDF backed by DataFusion
        // physical expressions.
        let (
            BatchUDF {
                mapper,
                output_schema,
                result_checkpoint,
            },
            read_columns,
        ) = match transforms {
            NewColumnTransform::BatchUDF(udf) => (udf, read_columns),
            NewColumnTransform::SqlExpressions(expressions) => {
                let arrow_schema = Arc::new(ArrowSchema::from(self.schema()));
                let planner = Planner::new(arrow_schema);
                let exprs = expressions
                    .into_iter()
                    .map(|(name, expr)| {
                        let expr = planner.parse_expr(&expr)?;
                        let expr = planner.optimize_expr(expr)?;
                        Ok((name, expr))
                    })
                    .collect::<Result<Vec<_>>>()?;

                let needed_columns = exprs
                    .iter()
                    .flat_map(|(_, expr)| Planner::column_names_in_expr(expr))
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                let read_schema = self.schema().project(&needed_columns)?;
                let read_schema = Arc::new(ArrowSchema::from(&read_schema));
                // Need to re-create the planner with the read schema because physical
                // expressions use positional column references.
                let planner = Planner::new(read_schema.clone());
                let exprs = exprs
                    .into_iter()
                    .map(|(name, expr)| {
                        let expr = planner.create_physical_expr(&expr)?;
                        Ok((name, expr))
                    })
                    .collect::<Result<Vec<_>>>()?;

                let output_schema = Arc::new(ArrowSchema::new(
                    exprs
                        .iter()
                        .map(|(name, expr)| {
                            Ok(ArrowField::new(
                                name,
                                expr.data_type(read_schema.as_ref())?,
                                expr.nullable(read_schema.as_ref())?,
                            ))
                        })
                        .collect::<Result<Vec<_>>>()?,
                ));

                let schema_ref = output_schema.clone();
                let mapper = move |batch: &RecordBatch| {
                    let num_rows = batch.num_rows();
                    let columns = exprs
                        .iter()
                        .map(|(_, expr)| Ok(expr.evaluate(batch)?.into_array(num_rows)?))
                        .collect::<Result<Vec<_>>>()?;

                    let batch = RecordBatch::try_new(schema_ref.clone(), columns)?;
                    Ok(batch)
                };
                let mapper = Box::new(mapper);

                let read_columns = Some(read_schema.field_names().into_iter().cloned().collect());
                (
                    BatchUDF {
                        mapper,
                        output_schema,
                        result_checkpoint: None,
                    },
                    read_columns,
                )
            }
        };

        {
            let new_names = output_schema.field_names();
            for field in &self.schema().fields {
                if new_names.contains(&&field.name) {
                    return Err(Error::invalid_input(
                        format!("Column {} already exists in the dataset", field.name),
                        location!(),
                    ));
                }
            }
        }

        let mut schema = self.schema().merge(output_schema.as_ref())?;
        schema.set_field_id(Some(self.manifest.max_field_id()));

        let fragments = self
            .add_columns_impl(read_columns, mapper, result_checkpoint, None)
            .await?;
        let operation = Operation::Merge { fragments, schema };
        let transaction = Transaction::new(self.manifest.version, operation, None);
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

    #[allow(clippy::type_complexity)]
    async fn add_columns_impl(
        &self,
        read_columns: Option<Vec<String>>,
        mapper: Box<dyn Fn(&RecordBatch) -> Result<RecordBatch> + Send + Sync>,
        result_cache: Option<Arc<dyn UDFCheckpointStore>>,
        schemas: Option<(Schema, Schema)>,
    ) -> Result<Vec<Fragment>> {
        let read_columns_ref = read_columns.as_deref();
        let mapper_ref = mapper.as_ref();
        let fragments = futures::stream::iter(self.get_fragments())
            .then(|fragment| {
                let cache_ref = result_cache.clone();
                let schemas_ref = &schemas;
                async move {
                    if let Some(cache) = &cache_ref {
                        let fragment_id = fragment.id() as u32;
                        let fragment = cache.get_fragment(fragment_id)?;
                        if let Some(fragment) = fragment {
                            return Ok(fragment);
                        }
                    }

                    let mut updater = fragment
                        .updater(read_columns_ref, schemas_ref.clone())
                        .await?;

                    let mut batch_index = 0;
                    // TODO: the structure of the updater prevents batch-level parallelism here,
                    //       but there is no reason why we couldn't do this in parallel.
                    while let Some(batch) = updater.next().await? {
                        let batch_info = BatchInfo {
                            fragment_id: fragment.id() as u32,
                            batch_index,
                        };

                        let new_batch = if let Some(cache) = &cache_ref {
                            if let Some(batch) = cache.get_batch(&batch_info)? {
                                batch
                            } else {
                                let new_batch = mapper_ref(batch)?;
                                cache.insert_batch(batch_info, new_batch.clone())?;
                                new_batch
                            }
                        } else {
                            mapper_ref(batch)?
                        };

                        updater.update(new_batch).await?;
                        batch_index += 1;
                    }

                    let fragment = updater.finish().await?;

                    if let Some(cache) = &cache_ref {
                        cache.insert_fragment(fragment.clone())?;
                    }

                    Ok::<_, Error>(fragment)
                }
            })
            .try_collect::<Vec<_>>()
            .await?;
        Ok(fragments)
    }

    /// Modify columns in the dataset, changing their name, type, or nullability.
    ///
    /// If a column has an index, it's index will be preserved.
    pub async fn alter_columns(&mut self, alterations: &[ColumnAlteration]) -> Result<()> {
        // Validate we aren't making nullable columns non-nullable and that all
        // the referenced columns actually exist.
        let mut new_schema = self.schema().clone();

        // Mapping of old to new fields that need to be casted.
        let mut cast_fields: Vec<(Field, Field)> = Vec::new();

        let mut next_field_id = self.manifest.max_field_id() + 1;

        for alteration in alterations {
            let field_src = self.schema().field(&alteration.path).ok_or_else(|| {
                Error::invalid_input(
                    format!(
                        "Column \"{}\" does not exist in the dataset",
                        alteration.path
                    ),
                    location!(),
                )
            })?;
            if let Some(nullable) = alteration.nullable {
                // TODO: in the future, we could check the values of the column to see if
                //       they are all non-null and thus the column could be made non-nullable.
                if field_src.nullable && !nullable {
                    return Err(Error::invalid_input(
                        format!(
                            "Column \"{}\" is already nullable and thus cannot be made non-nullable",
                            alteration.path
                        ),
                        location!(),
                    ));
                }
            }

            let field_dest = new_schema.mut_field_by_id(field_src.id).unwrap();
            if let Some(rename) = &alteration.rename {
                field_dest.name.clone_from(rename);
            }
            if let Some(nullable) = alteration.nullable {
                field_dest.nullable = nullable;
            }

            if let Some(data_type) = &alteration.data_type {
                if !(lance_arrow::cast::can_cast_types(&field_src.data_type(), data_type)
                    && is_upcast_downcast(&field_src.data_type(), data_type))
                {
                    return Err(Error::invalid_input(
                        format!(
                            "Cannot cast column \"{}\" from {:?} to {:?}",
                            alteration.path,
                            field_src.data_type(),
                            data_type
                        ),
                        location!(),
                    ));
                }

                let arrow_field = ArrowField::new(
                    field_dest.name.clone(),
                    data_type.clone(),
                    field_dest.nullable,
                );
                *field_dest = Field::try_from(&arrow_field)?;
                field_dest.set_id(field_src.parent_id, &mut next_field_id);

                cast_fields.push((field_src.clone(), field_dest.clone()));
            }
        }

        new_schema.validate()?;

        // If we aren't casting a column, we don't need to touch the fragments.
        let transaction = if cast_fields.is_empty() {
            Transaction::new(
                self.manifest.version,
                Operation::Project { schema: new_schema },
                None,
            )
        } else {
            // Otherwise, we need to re-write the relevant fields.
            let read_columns = cast_fields
                .iter()
                .map(|(old, _new)| {
                    let parts = self.schema().field_ancestry_by_id(old.id).unwrap();
                    let part_names = parts.iter().map(|p| p.name.clone()).collect::<Vec<_>>();
                    part_names.join(".")
                })
                .collect::<Vec<_>>();

            let new_ids = cast_fields
                .iter()
                .map(|(_old, new)| new.id)
                .collect::<Vec<_>>();
            // This schema contains the exact field ids we want to write the new fields with.
            let new_col_schema = new_schema.project_by_ids(&new_ids);

            let mapper = move |batch: &RecordBatch| {
                let mut fields = Vec::with_capacity(cast_fields.len());
                let mut columns = Vec::with_capacity(batch.num_columns());
                for (old, new) in &cast_fields {
                    let old_column = batch[&old.name].clone();
                    let new_column = lance_arrow::cast::cast_with_options(
                        &old_column,
                        &new.data_type(),
                        // Safe: false means it will error if the cast is lossy.
                        &CastOptions {
                            safe: false,
                            ..Default::default()
                        },
                    )?;
                    columns.push(new_column);
                    fields.push(Arc::new(ArrowField::from(new)));
                }
                let schema = Arc::new(ArrowSchema::new(fields));
                Ok(RecordBatch::try_new(schema, columns)?)
            };
            let mapper = Box::new(mapper);

            let fragments = self
                .add_columns_impl(
                    Some(read_columns),
                    mapper,
                    None,
                    Some((new_col_schema, new_schema.clone())),
                )
                .await?;

            // Some data files may no longer contain any columns in the dataset (e.g. if every
            // remaining column has been altered into a different data file) and so we remove them
            let schema_field_ids = new_schema.field_ids().into_iter().collect::<Vec<_>>();
            let fragments = fragments
                .into_iter()
                .map(|mut frag| {
                    frag.files.retain(|f| {
                        f.fields
                            .iter()
                            .any(|field| schema_field_ids.contains(field))
                    });
                    frag
                })
                .collect::<Vec<_>>();

            Transaction::new(
                self.manifest.version,
                Operation::Merge {
                    schema: new_schema,
                    fragments,
                },
                None,
            )
        };

        // TODO: adjust the indices here for the new schema

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

    /// Remove columns from the dataset.
    ///
    /// This is a metadata-only operation and does not remove the data from the
    /// underlying storage. In order to remove the data, you must subsequently
    /// call `compact_files` to rewrite the data without the removed columns and
    /// then call `cleanup_files` to remove the old files.
    pub async fn drop_columns(&mut self, columns: &[&str]) -> Result<()> {
        // Check if columns are present in the dataset and construct the new schema.
        for col in columns {
            if self.schema().field(col).is_none() {
                return Err(Error::invalid_input(
                    format!("Column {} does not exist in the dataset", col),
                    location!(),
                ));
            }
        }

        let columns_to_remove = self.manifest.schema.project(columns)?;
        let new_schema = self.manifest.schema.exclude(columns_to_remove)?;

        if new_schema.fields.is_empty() {
            return Err(Error::invalid_input(
                "Cannot drop all columns from a dataset",
                location!(),
            ));
        }

        let transaction = Transaction::new(
            self.manifest.version,
            Operation::Project { schema: new_schema },
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
}

#[derive(Debug)]
pub(crate) struct ManifestWriteConfig {
    auto_set_feature_flags: bool,  // default true
    timestamp: Option<SystemTime>, // default None
}

impl Default for ManifestWriteConfig {
    fn default() -> Self {
        Self {
            auto_set_feature_flags: true,
            timestamp: None,
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
    if config.auto_set_feature_flags {
        apply_feature_flags(manifest);
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

struct RowIdMeta {
    sorted: bool,
    contiguous: bool,
}

fn check_row_ids(row_ids: &[u64]) -> RowIdMeta {
    let mut sorted = true;
    let mut contiguous = true;

    if row_ids.is_empty() {
        return RowIdMeta { sorted, contiguous };
    }

    let mut last_id = row_ids[0];
    let first_fragment_id = row_ids[0] >> 32;

    for id in row_ids.iter().skip(1) {
        sorted &= *id > last_id;
        contiguous &= *id == last_id + 1;
        // Contiguous also requires the fragment ids are all the same
        contiguous &= (*id >> 32) == first_fragment_id;
        last_id = *id;
    }

    RowIdMeta { sorted, contiguous }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::vec;

    use super::*;
    use crate::arrow::FixedSizeListArrayExt;
    use crate::dataset::optimize::{compact_files, CompactionOptions};
    use crate::dataset::WriteMode::Overwrite;
    use crate::index::scalar::ScalarIndexParams;
    use crate::index::vector::VectorIndexParams;
    use crate::utils::test::TestDatasetGenerator;

    use arrow_array::types::Int64Type;
    use arrow_array::{
        builder::StringDictionaryBuilder, cast::as_string_array, types::Int32Type, ArrayRef,
        DictionaryArray, Float32Array, Int32Array, Int64Array, Int8Array, Int8DictionaryArray,
        RecordBatchIterator, StringArray, UInt16Array, UInt32Array,
    };
    use arrow_array::{FixedSizeListArray, Float16Array, Float64Array, ListArray};
    use arrow_ord::sort::sort_to_indices;
    use arrow_schema::{Field, Fields as ArrowFields, Schema as ArrowSchema};
    use half::f16;
    use lance_arrow::bfloat16::{self, ARROW_EXT_META_KEY, ARROW_EXT_NAME_KEY, BFLOAT16_EXT_NAME};
    use lance_datagen::{array, gen, BatchCount, RowCount};
    use lance_index::{vector::DIST_COL, DatasetIndexExt, IndexType};
    use lance_linalg::distance::MetricType;
    use lance_table::format::WriterVersion;
    use lance_table::io::deletion::read_deletion_file;
    use lance_testing::datagen::generate_random_array;
    use pretty_assertions::assert_eq;
    use tempfile::{tempdir, TempDir};
    use tests::scanner::test_dataset::TestVectorDataset;

    // Used to validate that futures returned are Send.
    fn require_send<T: Send>(t: T) -> T {
        t
    }

    async fn create_file(path: &std::path::Path, mode: WriteMode) {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new(
                "dict",
                DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
                false,
            ),
        ]));
        let dict_values = StringArray::from_iter_values(["a", "b", "c", "d", "e"]);
        let batches: Vec<RecordBatch> = (0..20)
            .map(|i| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)),
                        Arc::new(
                            DictionaryArray::try_new(
                                UInt16Array::from_iter_values((0_u16..20_u16).map(|v| v % 5)),
                                Arc::new(dict_values.clone()),
                            )
                            .unwrap(),
                        ),
                    ],
                )
                .unwrap()
            })
            .collect();
        let expected_batches = batches.clone();

        let test_uri = path.to_str().unwrap();
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            mode,
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
        for batch in &actual_batches {
            assert_eq!(batch.num_rows(), 10);
        }

        // sort
        let actual_batch = concat_batches(&schema, &actual_batches).unwrap();
        let idx_arr = actual_batch.column_by_name("i").unwrap();
        let sorted_indices = sort_to_indices(idx_arr, None, None).unwrap();
        let struct_arr: StructArray = actual_batch.into();
        let sorted_arr = take(&struct_arr, &sorted_indices, None).unwrap();

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

    #[lance_test_macros::test(tokio::test)]
    async fn test_create_dataset() {
        // Appending / Overwriting a dataset that does not exist is treated as Create
        for mode in [WriteMode::Create, WriteMode::Append, Overwrite] {
            let test_dir = tempdir().unwrap();
            create_file(test_dir.path(), mode).await
        }
    }

    #[lance_test_macros::test(tokio::test)]
    async fn test_create_and_fill_empty_dataset() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
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
        let sorted_arr = take(&struct_arr, &sorted_indices, None).unwrap();
        let expected_struct_arr: StructArray = expected_batch.into();
        assert_eq!(&expected_struct_arr, as_struct_array(sorted_arr.as_ref()));
    }

    #[lance_test_macros::test(tokio::test)]
    async fn test_create_with_empty_iter() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let reader = RecordBatchIterator::new(vec![].into_iter().map(Ok), schema.clone());
        // check schema of reader and original is same
        assert_eq!(schema.as_ref(), reader.schema().as_ref());
        let result = Dataset::write(reader, test_uri, None).await.unwrap();

        // check dataset empty
        assert_eq!(result.count_rows(None).await.unwrap(), 0);
        // Since the dataset is empty, will return None.
        assert_eq!(result.manifest.max_fragment_id(), None);
    }

    #[tokio::test]
    async fn test_write_params() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
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
            let reader = fragment.open(dataset.schema(), false).await.unwrap();
            assert_eq!(reader.legacy_num_batches(), 10);
            for i in 0..reader.legacy_num_batches() {
                assert_eq!(reader.legacy_num_rows_in_batch(i), 10);
            }
        }
    }

    #[tokio::test]
    async fn test_write_manifest() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
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
        let write_fut = Dataset::write(batches, test_uri, None);
        let write_fut = require_send(write_fut);
        let mut dataset = write_fut.await.unwrap();

        // Check it has no flags
        let manifest = read_manifest(
            dataset.object_store(),
            &dataset
                .commit_handler
                .resolve_latest_version(&dataset.base, &dataset.object_store().inner)
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(manifest.writer_feature_flags, 0);
        assert_eq!(manifest.reader_feature_flags, 0);

        // Create one with deletions
        dataset.delete("i < 10").await.unwrap();
        dataset.validate().await.unwrap();

        // Check it set the flag
        let mut manifest = read_manifest(
            dataset.object_store(),
            &dataset
                .commit_handler
                .resolve_latest_version(&dataset.base, &dataset.object_store().inner)
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
        manifest.writer_feature_flags = 5; // Set another flag
        manifest.reader_feature_flags = 5;
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
                ..Default::default()
            }),
        )
        .await;

        assert!(matches!(write_result, Err(Error::NotSupported { .. })));
    }

    #[tokio::test]
    async fn append_dataset() {
        let test_dir = tempdir().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
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
        let sorted_arr = take(&struct_arr, &sorted_indices, None).unwrap();

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

    #[tokio::test]
    async fn test_self_dataset_append() {
        let test_dir = tempdir().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
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
        let sorted_arr = take(&struct_arr, &sorted_indices, None).unwrap();

        let expected_struct_arr: StructArray = expected_batch.into();
        assert_eq!(&expected_struct_arr, as_struct_array(sorted_arr.as_ref()));

        actual_ds.validate().await.unwrap();
    }

    #[tokio::test]
    async fn test_self_dataset_append_schema_different() {
        let test_dir = tempdir().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..20))],
        )
        .unwrap()];

        let other_schema = Arc::new(ArrowSchema::new(vec![Field::new(
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

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
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

    #[tokio::test]
    async fn overwrite_dataset() {
        let test_dir = tempdir().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
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
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let dataset = Dataset::write(batches, test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(0));

        let new_schema = Arc::new(ArrowSchema::new(vec![Field::new(
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
        let first_ver = Dataset::checkout(test_uri, 1).await.unwrap();
        assert_eq!(first_ver.version().version, 1);
        assert_eq!(&ArrowSchema::from(first_ver.schema()), schema.as_ref());
    }

    #[tokio::test]
    async fn test_take() {
        let test_dir = tempdir().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        let batches: Vec<RecordBatch> = (0..20)
            .map(|i| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)),
                        Arc::new(StringArray::from_iter_values(
                            (i * 20..(i + 1) * 20).map(|i| format!("str-{i}")),
                        )),
                    ],
                )
                .unwrap()
            })
            .collect();
        let test_uri = test_dir.path().to_str().unwrap();
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
        let projection = Schema::try_from(schema.as_ref()).unwrap();
        let values = dataset
            .take(
                &[
                    200, // 200
                    199, // 199
                    39,  // 39
                    40,  // 40
                    199, // 40
                    40,  // 40
                    125, // 125
                ],
                &projection,
            )
            .await
            .unwrap();
        assert_eq!(
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values([
                        200, 199, 39, 40, 199, 40, 125
                    ])),
                    Arc::new(StringArray::from_iter_values(
                        [200, 199, 39, 40, 199, 40, 125]
                            .iter()
                            .map(|v| format!("str-{v}"))
                    )),
                ],
            )
            .unwrap(),
            values
        );
    }

    #[tokio::test]
    async fn test_take_rows_out_of_bound() {
        // a dataset with 1 fragment and 400 rows
        let test_ds = TestVectorDataset::new().await.unwrap();
        let ds = test_ds.dataset;

        // take the last row of first fragment
        // this triggeres the contiguous branch
        let indices = &[(1 << 32) - 1];
        let err = ds.take_rows(indices, ds.schema()).await.unwrap_err();
        assert!(
            err.to_string().contains("Invalid read params"),
            "{}",
            err.to_string()
        );

        // this triggeres the sorted branch, but not continguous
        let indices = &[(1 << 32) - 3, (1 << 32) - 1];
        let err = ds.take_rows(indices, ds.schema()).await.unwrap_err();
        assert!(
            err.to_string().contains("out of bounds"),
            "{}",
            err.to_string()
        );

        // this triggeres the catch all branch
        let indices = &[(1 << 32) - 1, (1 << 32) - 3];
        let err = ds.take_rows(indices, ds.schema()).await.unwrap_err();
        assert!(
            err.to_string().contains("out of bounds"),
            "{}",
            err.to_string()
        );
    }

    #[tokio::test]
    async fn test_take_rows() {
        let test_dir = tempdir().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        let batches: Vec<RecordBatch> = (0..20)
            .map(|i| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)),
                        Arc::new(StringArray::from_iter_values(
                            (i * 20..(i + 1) * 20).map(|i| format!("str-{i}")),
                        )),
                    ],
                )
                .unwrap()
            })
            .collect();
        let test_uri = test_dir.path().to_str().unwrap();
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
        let projection = Schema::try_from(schema.as_ref()).unwrap();
        let indices = &[
            5_u64 << 32,        // 200
            (4_u64 << 32) + 39, // 199
            39,                 // 39
            1_u64 << 32,        // 40
            (2_u64 << 32) + 20, // 100
        ];
        let values = dataset.take_rows(indices, &projection).await.unwrap();
        assert_eq!(
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values([200, 199, 39, 40, 100])),
                    Arc::new(StringArray::from_iter_values(
                        [200, 199, 39, 40, 100].iter().map(|v| format!("str-{v}"))
                    )),
                ],
            )
            .unwrap(),
            values
        );

        // Delete some rows from a fragment
        dataset.delete("i in (199, 100)").await.unwrap();
        dataset.validate().await.unwrap();
        let values = dataset.take_rows(indices, &projection).await.unwrap();
        assert_eq!(
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values([200, 39, 40])),
                    Arc::new(StringArray::from_iter_values(
                        [200, 39, 40].iter().map(|v| format!("str-{v}"))
                    )),
                ],
            )
            .unwrap(),
            values
        );

        // Take an empty selection.
        let values = dataset.take_rows(&[], &projection).await.unwrap();
        assert_eq!(RecordBatch::new_empty(schema.clone()), values);
    }

    #[tokio::test]
    async fn test_fast_count_rows() {
        let test_dir = tempdir().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
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

    #[tokio::test]
    async fn test_create_index() {
        let test_dir = tempdir().unwrap();

        let dimension = 16;
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "embeddings",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
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

        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();
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

    #[tokio::test]
    async fn test_create_scalar_index() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let data = gen().col("int", array::step::<Int32Type>());
        // Write 64Ki rows.  We should get 16 4Ki pages
        let mut dataset = Dataset::write(
            data.into_reader_rows(RowCount::from(16 * 1024), BatchCount::from(4)),
            test_uri,
            None,
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

    async fn create_bad_file() -> Result<Dataset> {
        let test_dir = tempdir().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
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
        Dataset::write(reader, test_uri, None).await
    }

    #[tokio::test]
    async fn test_bad_field_name() {
        // don't allow `.` in the field name
        assert!(create_bad_file().await.is_err());
    }

    #[tokio::test]
    async fn test_open_dataset_not_found() {
        let result = Dataset::open(".").await;
        assert!(matches!(result.unwrap_err(), Error::DatasetNotFound { .. }));
    }

    #[tokio::test]
    async fn test_drop_columns() -> Result<()> {
        let metadata: HashMap<String, String> = [("k1".into(), "v1".into())].into();

        let schema = Arc::new(ArrowSchema::new_with_metadata(
            vec![
                Field::new("i", DataType::Int32, false),
                Field::new(
                    "s",
                    DataType::Struct(ArrowFields::from(vec![
                        Field::new("d", DataType::Int32, true),
                        Field::new("l", DataType::Int32, true),
                    ])),
                    true,
                ),
                Field::new("x", DataType::Float32, false),
            ],
            metadata.clone(),
        ));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StructArray::from(vec![
                    (
                        Arc::new(ArrowField::new("d", DataType::Int32, true)),
                        Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
                    ),
                    (
                        Arc::new(ArrowField::new("l", DataType::Int32, true)),
                        Arc::new(Int32Array::from(vec![1, 2])),
                    ),
                ])),
                Arc::new(Float32Array::from(vec![1.0, 2.0])),
            ],
        )?;

        let test_dir = tempdir()?;
        let test_uri = test_dir.path().to_str().unwrap();

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await?;

        let lance_schema = dataset.schema().clone();
        let original_fragments = dataset.fragments().to_vec();

        dataset.drop_columns(&["x"]).await?;
        dataset.validate().await?;

        let expected_schema = lance_schema.project(&["i", "s"])?;
        assert_eq!(dataset.schema(), &expected_schema);

        assert_eq!(dataset.version().version, 2);
        assert_eq!(dataset.fragments().as_ref(), &original_fragments);

        dataset.drop_columns(&["s.d"]).await?;
        dataset.validate().await?;

        let expected_schema = expected_schema.project(&["i", "s.l"])?;
        assert_eq!(dataset.schema(), &expected_schema);

        let expected_data = RecordBatch::try_new(
            Arc::new(ArrowSchema::from(&expected_schema)),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StructArray::from(vec![(
                    Arc::new(ArrowField::new("l", DataType::Int32, true)),
                    Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
                )])),
            ],
        )?;
        let actual_data = dataset.scan().try_into_batch().await?;
        assert_eq!(actual_data, expected_data);

        assert_eq!(dataset.version().version, 3);
        assert_eq!(dataset.fragments().as_ref(), &original_fragments);

        Ok(())
    }

    #[tokio::test]
    async fn test_drop_add_columns() -> Result<()> {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1, 2]))])?;

        let test_dir = tempdir()?;
        let test_uri = test_dir.path().to_str().unwrap();

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await?;
        assert_eq!(dataset.manifest.max_field_id(), 0);

        // Test we can add 1 column, drop it, then add another column. Validate
        // the field ids are as expected.
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("x".into(), "i + 1".into())]),
                Some(vec!["i".into()]),
            )
            .await?;
        assert_eq!(dataset.manifest.max_field_id(), 1);

        dataset.drop_columns(&["x"]).await?;
        assert_eq!(dataset.manifest.max_field_id(), 0);

        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("y".into(), "2 * i".into())]),
                Some(vec!["i".into()]),
            )
            .await?;
        assert_eq!(dataset.manifest.max_field_id(), 1);

        let data = dataset.scan().try_into_batch().await?;
        let expected_data = RecordBatch::try_new(
            Arc::new(schema.try_with_column(Field::new("y", DataType::Int32, false))?),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(Int32Array::from(vec![2, 4])),
            ],
        )?;
        assert_eq!(data, expected_data);
        dataset.drop_columns(&["y"]).await?;
        assert_eq!(dataset.manifest.max_field_id(), 0);

        // Test we can add 2 columns, drop 1, then add another column. Validate
        // the field ids are as expected.
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![
                    ("a".into(), "i + 3".into()),
                    ("b".into(), "i + 7".into()),
                ]),
                Some(vec!["i".into()]),
            )
            .await?;
        assert_eq!(dataset.manifest.max_field_id(), 2);

        dataset.drop_columns(&["b"]).await?;
        // Even though we dropped a column, we still have the fragment with a and
        // b. So it should still act as if that field id is still in play.
        assert_eq!(dataset.manifest.max_field_id(), 2);

        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("c".into(), "i + 11".into())]),
                Some(vec!["i".into()]),
            )
            .await?;
        assert_eq!(dataset.manifest.max_field_id(), 3);

        let data = dataset.scan().try_into_batch().await?;
        let expected_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("a", DataType::Int32, false),
            Field::new("c", DataType::Int32, false),
        ]));
        let expected_data = RecordBatch::try_new(
            expected_schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(Int32Array::from(vec![4, 5])),
                Arc::new(Int32Array::from(vec![12, 13])),
            ],
        )?;
        assert_eq!(data, expected_data);

        Ok(())
    }

    #[tokio::test]
    async fn test_merge() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("x", DataType::Float32, false),
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
            Field::new("i2", DataType::Int32, false),
            Field::new("y", DataType::Utf8, true),
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
                Field::new("i", DataType::Int32, false),
                Field::new("x", DataType::Float32, false),
                Field::new("y", DataType::Utf8, true),
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

    #[tokio::test]
    async fn test_delete() {
        fn sequence_data(range: Range<u32>) -> RecordBatch {
            let schema = Arc::new(ArrowSchema::new(vec![
                Field::new("i", DataType::UInt32, false),
                Field::new("x", DataType::UInt32, false),
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
            Field::new("i", DataType::UInt32, false),
            Field::new("x", DataType::UInt32, false),
        ]));
        let data = sequence_data(0..100);
        // Split over two files.
        let batches = vec![data.slice(0, 50), data.slice(50, 50)];
        let mut dataset = TestDatasetGenerator::new(batches)
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

    #[tokio::test]
    async fn test_restore() {
        // Create a table
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
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

    #[tokio::test]
    async fn test_search_empty() {
        // Create a table
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "vec",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 128),
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
        let dataset = Dataset::write(reader, test_uri, None).await.unwrap();

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
                &Field::new(
                    "vec",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        128
                    ),
                    false,
                )
            );
            assert_eq!(
                schema.field_with_name(DIST_COL).unwrap(),
                &Field::new(DIST_COL, DataType::Float32, true)
            );
        }
    }

    #[tokio::test]
    async fn test_search_empty_after_delete() {
        // Create a table
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "vec",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 128),
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
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();
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
                &Field::new(
                    "vec",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        128
                    ),
                    false,
                )
            );
            assert_eq!(
                schema.field_with_name(DIST_COL).unwrap(),
                &Field::new(DIST_COL, DataType::Float32, true)
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
                &Field::new(
                    "vec",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        128
                    ),
                    false,
                )
            );
            assert_eq!(
                schema.field_with_name(DIST_COL).unwrap(),
                &Field::new(DIST_COL, DataType::Float32, true)
            );
        }
    }

    #[tokio::test]
    async fn test_num_small_files() {
        let test_dir = tempdir().unwrap();
        let dimensions = 16;
        let column_name = "vec";
        let field = Field::new(
            column_name,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
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
        let dataset = Dataset::write(reader, test_uri, None).await.unwrap();
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

    #[tokio::test]
    async fn take_scan_dataset() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("x", DataType::Float32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
                Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            ],
        )
        .unwrap();

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let write_params = WriteParams {
            max_rows_per_group: 2,
            ..Default::default()
        };

        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();

        let projection = Arc::new(dataset.schema().project(&["i"]).unwrap());
        let ranges = [0_u64..3, 1..4, 0..1];
        let range_stream = futures::stream::iter(ranges).map(Ok).boxed();
        let results = dataset
            .take_scan(range_stream, projection.clone(), 10)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let expected_schema = projection.as_ref().into();
        for batch in &results {
            assert_eq!(batch.schema().as_ref(), &expected_schema);
        }
        assert_eq!(results.len(), 3);
        assert_eq!(
            results[0].column(0).as_primitive::<Int32Type>().values(),
            &[1, 2, 3],
        );
        assert_eq!(
            results[1].column(0).as_primitive::<Int32Type>().values(),
            &[2, 3, 4],
        );
        assert_eq!(
            results[2].column(0).as_primitive::<Int32Type>().values(),
            &[1],
        );
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

    #[tokio::test]
    async fn test_v0_8_14_invalid_index_fragment_bitmap() {
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
        dataset.append(data, None).await.unwrap();

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

    #[tokio::test]
    async fn test_bfloat16_roundtrip() -> Result<()> {
        let inner_field = Arc::new(
            Field::new("item", DataType::FixedSizeBinary(2), true).with_metadata(
                [
                    (ARROW_EXT_NAME_KEY.into(), BFLOAT16_EXT_NAME.into()),
                    (ARROW_EXT_META_KEY.into(), "".into()),
                ]
                .into(),
            ),
        );
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
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
            None,
        )
        .await?;

        let data = dataset.scan().try_into_batch().await?;
        assert_eq!(batch, data);

        Ok(())
    }

    #[tokio::test]
    async fn test_append_columns_exprs() -> Result<()> {
        let num_rows = 5;
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..num_rows as i32))],
        )?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

        let test_dir = tempdir()?;
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = Dataset::write(reader, test_uri, None).await?;
        dataset.validate().await?;

        // Adding a duplicate column name will break
        let fut = dataset.add_columns(
            NewColumnTransform::SqlExpressions(vec![("id".into(), "id + 1".into())]),
            None,
        );
        // (Quick validation that the future is Send)
        let res = require_send(fut).await;
        assert!(matches!(res, Err(Error::InvalidInput { .. })));

        // Can add a column that is independent of any existing ones
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("value".into(), "2 * random()".into())]),
                None,
            )
            .await?;

        // Can add a column derived from an existing one.
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("double_id".into(), "2 * id".into())]),
                None,
            )
            .await?;

        // Can derive a column from existing ones across multiple data files.
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![(
                    "triple_id".into(),
                    "id + double_id".into(),
                )]),
                None,
            )
            .await?;

        // These can be read back, the dataset is valid
        dataset.validate().await?;

        let data = dataset.scan().try_into_batch().await?;
        let expected_schema = ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Float64, true),
            Field::new("double_id", DataType::Int32, false),
            Field::new("triple_id", DataType::Int32, false),
        ]);
        assert_eq!(data.schema().as_ref(), &expected_schema);
        assert_eq!(data.num_rows(), num_rows);

        Ok(())
    }

    #[tokio::test]
    async fn test_append_columns_udf() -> Result<()> {
        let num_rows = 5;
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..num_rows as i32))],
        )?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

        let test_dir = tempdir()?;
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = Dataset::write(reader, test_uri, None).await?;
        dataset.validate().await?;

        // Adding a duplicate column name will break
        let transforms = NewColumnTransform::BatchUDF(BatchUDF {
            mapper: Box::new(|_| unimplemented!()),
            output_schema: Arc::new(ArrowSchema::new(vec![Field::new(
                "id",
                DataType::Int32,
                false,
            )])),
            result_checkpoint: None,
        });
        let res = dataset.add_columns(transforms, None).await;
        assert!(matches!(res, Err(Error::InvalidInput { .. })));

        // Can add a column that independent (empty read_schema)
        let output_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "value",
            DataType::Float64,
            true,
        )]));
        let output_schema_ref = output_schema.clone();
        let mapper = move |batch: &RecordBatch| {
            Ok(RecordBatch::try_new(
                output_schema_ref.clone(),
                vec![Arc::new(Float64Array::from_iter_values(
                    (0..batch.num_rows()).map(|i| i as f64),
                ))],
            )?)
        };
        let transforms = NewColumnTransform::BatchUDF(BatchUDF {
            mapper: Box::new(mapper),
            output_schema,
            result_checkpoint: None,
        });
        dataset.add_columns(transforms, None).await?;

        // Can add a column that depends on another column (double id)
        let output_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "double_id",
            DataType::Int32,
            false,
        )]));
        let output_schema_ref = output_schema.clone();
        let mapper = move |batch: &RecordBatch| {
            let id = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            Ok(RecordBatch::try_new(
                output_schema_ref.clone(),
                vec![Arc::new(Int32Array::from_iter_values(
                    id.values().iter().map(|i| i * 2),
                ))],
            )?)
        };
        let transforms = NewColumnTransform::BatchUDF(BatchUDF {
            mapper: Box::new(mapper),
            output_schema,
            result_checkpoint: None,
        });
        dataset.add_columns(transforms, None).await?;
        // These can be read back, the dataset is valid
        dataset.validate().await?;

        let data = dataset.scan().try_into_batch().await?;
        let expected_schema = ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Float64, true),
            Field::new("double_id", DataType::Int32, false),
        ]);
        assert_eq!(data.schema().as_ref(), &expected_schema);
        assert_eq!(data.num_rows(), num_rows);

        Ok(())
    }

    #[tokio::test]
    async fn test_append_columns_udf_cache() -> Result<()> {
        let num_rows = 100;
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..num_rows))],
        )?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

        let test_dir = tempdir()?;
        let test_uri = test_dir.path().to_str().unwrap();
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 50,
                max_rows_per_group: 25,
                ..Default::default()
            }),
        )
        .await?;
        dataset.validate().await?;

        #[derive(Default)]
        struct RequestCounter {
            pub get_batch_requests: Mutex<Vec<BatchInfo>>,
            pub insert_batch_requests: Mutex<Vec<BatchInfo>>,
            pub get_fragment_requests: Mutex<Vec<u32>>,
            pub insert_fragment_requests: Mutex<Vec<u32>>,
        }

        impl UDFCheckpointStore for RequestCounter {
            fn get_batch(&self, info: &BatchInfo) -> Result<Option<RecordBatch>> {
                self.get_batch_requests.lock().unwrap().push(info.clone());

                if info.fragment_id == 1 && info.batch_index == 0 {
                    Ok(Some(RecordBatch::try_new(
                        Arc::new(ArrowSchema::new(vec![Field::new(
                            "double_id",
                            DataType::Int32,
                            false,
                        )])),
                        vec![Arc::new(Int32Array::from_iter_values(50..75))],
                    )?))
                } else {
                    Ok(None)
                }
            }

            fn insert_batch(&self, info: BatchInfo, _value: RecordBatch) -> Result<()> {
                self.insert_batch_requests.lock().unwrap().push(info);
                Ok(())
            }

            fn get_fragment(&self, fragment_id: u32) -> Result<Option<Fragment>> {
                self.get_fragment_requests.lock().unwrap().push(fragment_id);
                if fragment_id == 0 {
                    Ok(Some(Fragment {
                        files: vec![],
                        id: 0,
                        deletion_file: None,
                        physical_rows: Some(50),
                    }))
                } else {
                    Ok(None)
                }
            }

            fn insert_fragment(&self, fragment: Fragment) -> Result<()> {
                self.insert_fragment_requests
                    .lock()
                    .unwrap()
                    .push(fragment.id as u32);
                Ok(())
            }
        }

        let request_counter = Arc::new(RequestCounter::default());

        let output_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "double_id",
            DataType::Int32,
            false,
        )]));
        let output_schema_ref = output_schema.clone();
        let mapper = move |batch: &RecordBatch| {
            let id = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            Ok(RecordBatch::try_new(
                output_schema_ref.clone(),
                vec![Arc::new(Int32Array::from_iter_values(
                    id.values().iter().map(|i| i * 2),
                ))],
            )?)
        };
        let transforms = NewColumnTransform::BatchUDF(BatchUDF {
            mapper: Box::new(mapper),
            output_schema,
            result_checkpoint: Some(request_counter.clone()),
        });
        dataset.add_columns(transforms, None).await?;

        // Should have requested both fragments
        assert_eq!(
            request_counter
                .get_fragment_requests
                .lock()
                .unwrap()
                .as_slice(),
            &[0, 1]
        );
        // Should have only inserted the second fragment, since the first one was already cached
        assert_eq!(
            request_counter
                .insert_fragment_requests
                .lock()
                .unwrap()
                .as_slice(),
            &[1]
        );

        // Should have only requested the second two batches, since the first fragment was already cached
        assert_eq!(
            request_counter
                .get_batch_requests
                .lock()
                .unwrap()
                .as_slice(),
            &[
                BatchInfo {
                    fragment_id: 1,
                    batch_index: 0,
                },
                BatchInfo {
                    fragment_id: 1,
                    batch_index: 1,
                },
            ]
        );
        // Should have only saved the last batch, since the first batch of second fragment was already cached
        assert_eq!(
            request_counter
                .insert_batch_requests
                .lock()
                .unwrap()
                .as_slice(),
            &[BatchInfo {
                fragment_id: 1,
                batch_index: 1,
            },]
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_rename_columns() -> Result<()> {
        let metadata: HashMap<String, String> = [("k1".into(), "v1".into())].into();

        let schema = Arc::new(ArrowSchema::new_with_metadata(
            vec![
                Field::new("a", DataType::Int32, false),
                Field::new(
                    "b",
                    DataType::Struct(ArrowFields::from(vec![Field::new(
                        "c",
                        DataType::Int32,
                        true,
                    )])),
                    true,
                ),
            ],
            metadata.clone(),
        ));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StructArray::from(vec![(
                    Arc::new(ArrowField::new("c", DataType::Int32, true)),
                    Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
                )])),
            ],
        )?;

        let test_dir = tempdir()?;
        let test_uri = test_dir.path().to_str().unwrap();

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await?;

        let original_fragments = dataset.fragments().to_vec();

        // Rename a top-level column
        dataset
            .alter_columns(&[ColumnAlteration::new("a".into())
                .rename("x".into())
                .set_nullable(true)])
            .await?;
        dataset.validate().await?;
        assert_eq!(dataset.manifest.version, 2);
        assert_eq!(dataset.fragments().as_ref(), &original_fragments);

        let expected_schema = ArrowSchema::new_with_metadata(
            vec![
                Field::new("x", DataType::Int32, true),
                Field::new(
                    "b",
                    DataType::Struct(ArrowFields::from(vec![Field::new(
                        "c",
                        DataType::Int32,
                        true,
                    )])),
                    true,
                ),
            ],
            metadata.clone(),
        );
        assert_eq!(&ArrowSchema::from(dataset.schema()), &expected_schema);

        // Rename to duplicate name fails
        let err = dataset
            .alter_columns(&[ColumnAlteration::new("b".into()).rename("x".into())])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("Duplicate field name \"x\""));

        // Rename a nested column.
        dataset
            .alter_columns(&[ColumnAlteration::new("b.c".into()).rename("d".into())])
            .await?;
        dataset.validate().await?;
        assert_eq!(dataset.manifest.version, 3);
        assert_eq!(dataset.fragments().as_ref(), &original_fragments);

        let expected_schema = ArrowSchema::new_with_metadata(
            vec![
                Field::new("x", DataType::Int32, true),
                Field::new(
                    "b",
                    DataType::Struct(ArrowFields::from(vec![Field::new(
                        "d",
                        DataType::Int32,
                        true,
                    )])),
                    true,
                ),
            ],
            metadata.clone(),
        );
        assert_eq!(&ArrowSchema::from(dataset.schema()), &expected_schema);

        Ok(())
    }

    #[tokio::test]
    async fn test_cast_column() -> Result<()> {
        // Create a table with 2 scalar columns, 1 vector column
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("f", DataType::Float32, false),
            Field::new(
                "vec",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 128),
                false,
            ),
            Field::new("l", DataType::new_list(DataType::Int32, true), true),
        ]));

        let nrows = 512;
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..nrows)),
                Arc::new(Float32Array::from_iter_values((0..nrows).map(|i| i as f32))),
                Arc::new(
                    <arrow_array::FixedSizeListArray as FixedSizeListArrayExt>::try_new_from_values(
                        generate_random_array(128 * nrows as usize),
                        128,
                    )
                    .unwrap(),
                ),
                Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(
                    (0..nrows).map(|i| Some(vec![Some(i), Some(i + 1)])),
                )),
            ],
        )?;

        let test_dir = tempdir()?;
        let test_uri = test_dir.path().to_str().unwrap();

        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone()),
            test_uri,
            None,
        )
        .await?;

        let params = VectorIndexParams::ivf_pq(10, 8, 2, MetricType::L2, 50);
        dataset
            .create_index(&["vec"], IndexType::Vector, None, &params, false)
            .await?;
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await?;
        dataset.validate().await?;

        let indices = dataset.load_indices().await?;
        assert_eq!(indices.len(), 2);

        // Cast a scalar column to another type, nullability
        dataset
            .alter_columns(&[ColumnAlteration::new("f".into())
                .cast_to(DataType::Float16)
                .set_nullable(true)])
            .await?;
        dataset.validate().await?;
        let expected_schema = ArrowSchema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("f", DataType::Float16, true),
            Field::new(
                "vec",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 128),
                false,
            ),
            Field::new("l", DataType::new_list(DataType::Int32, true), true),
        ]);
        assert_eq!(&ArrowSchema::from(dataset.schema()), &expected_schema);

        // Each fragment gains a file with the new columns
        dataset.fragments().iter().for_each(|f| {
            assert_eq!(f.files.len(), 2);
        });

        // Cast scalar column with index, should not keep index (TODO: keep it)
        dataset
            .alter_columns(&[ColumnAlteration::new("i".into()).cast_to(DataType::Int64)])
            .await?;
        dataset.validate().await?;

        let expected_schema = ArrowSchema::new(vec![
            Field::new("i", DataType::Int64, false),
            Field::new("f", DataType::Float16, true),
            Field::new(
                "vec",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 128),
                false,
            ),
            Field::new("l", DataType::new_list(DataType::Int32, true), true),
        ]);
        assert_eq!(&ArrowSchema::from(dataset.schema()), &expected_schema);

        // We currently lose the index when casting a column
        let indices = dataset.load_indices().await?;
        assert_eq!(indices.len(), 1);

        // Each fragment gains a file with the new columns
        dataset.fragments().iter().for_each(|f| {
            assert_eq!(f.files.len(), 3);
        });

        // Cast vector column, should not keep index (TODO: keep it)
        dataset
            .alter_columns(&[
                ColumnAlteration::new("vec".into()).cast_to(DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float16, true)),
                    128,
                )),
            ])
            .await?;
        dataset.validate().await?;

        // Finally, case list column to show we can handle children.
        dataset
            .alter_columns(&[ColumnAlteration::new("l".into())
                .cast_to(DataType::new_list(DataType::Int64, true))])
            .await?;
        dataset.validate().await?;

        let expected_schema = ArrowSchema::new(vec![
            Field::new("i", DataType::Int64, false),
            Field::new("f", DataType::Float16, true),
            Field::new(
                "vec",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float16, true)), 128),
                false,
            ),
            Field::new("l", DataType::new_list(DataType::Int64, true), true),
        ]);
        assert_eq!(&ArrowSchema::from(dataset.schema()), &expected_schema);

        // We currently lose the index when casting a column
        let indices = dataset.load_indices().await?;
        assert_eq!(indices.len(), 0);

        // Each fragment gains a file with the new columns, but then the original file is dropped
        dataset.fragments().iter().for_each(|f| {
            assert_eq!(f.files.len(), 4);
        });

        let expected_data = RecordBatch::try_new(
            Arc::new(expected_schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..nrows as i64)),
                Arc::new(Float16Array::from_iter_values(
                    (0..nrows).map(|i| f16::from_f32(i as f32)),
                )),
                lance_arrow::cast::cast_with_options(
                    batch["vec"].as_ref(),
                    &DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float16, true)),
                        128,
                    ),
                    &Default::default(),
                )?,
                Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(
                    (0..nrows as i64).map(|i| Some(vec![Some(i), Some(i + 1)])),
                )),
            ],
        )?;
        let actual_data = dataset.scan().try_into_batch().await?;
        assert_eq!(actual_data, expected_data);

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
}
