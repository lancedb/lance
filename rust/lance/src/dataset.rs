// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Lance Dataset
//!

use std::collections::{BTreeMap, HashMap};
use std::default::Default;
use std::sync::Arc;

use arrow_array::{
    cast::as_struct_array, RecordBatch, RecordBatchReader, StructArray, UInt64Array,
};
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
use arrow_select::{concat::concat_batches, take::take};
use chrono::prelude::*;
use futures::future::BoxFuture;
use futures::stream::{self, StreamExt, TryStreamExt};
use log::warn;
use object_store::path::Path;

mod chunker;
pub mod cleanup;
mod feature_flags;
pub mod fragment;
mod hash_joiner;
pub mod progress;
pub mod scanner;
pub mod transaction;
pub mod updater;
mod write;

use self::feature_flags::{apply_feature_flags, can_read_dataset, can_write_dataset};
use self::fragment::FileFragment;
use self::scanner::Scanner;
use self::transaction::{Operation, Transaction};
use self::write::{reader_to_stream, write_fragments};
use crate::datatypes::Schema;
use crate::error::box_error;
use crate::format::{Fragment, Index, Manifest};
use crate::index::vector::open_index;
use crate::io::reader::read_manifest_indexes;
use crate::io::{
    commit::{commit_new_dataset, commit_transaction, CommitError},
    object_reader::read_struct,
    object_store::ObjectStoreParams,
    read_manifest, read_metadata_offset, write_manifest, ObjectStore,
};
use crate::session::Session;
use crate::utils::temporal::SystemTime;
use crate::{Error, Result};
use hash_joiner::HashJoiner;
pub use scanner::ROW_ID;
pub use write::{WriteMode, WriteParams};

const INDICES_DIR: &str = "_indices";
pub(crate) const DELETION_DIRS: &str = "_deletions";
const DATA_DIR: &str = "data";
pub(crate) const DEFAULT_INDEX_CACHE_SIZE: usize = 256;
pub(crate) const DEFAULT_METADATA_CACHE_SIZE: usize = 256;

/// Lance Dataset
#[derive(Debug, Clone)]
pub struct Dataset {
    pub(crate) object_store: Arc<ObjectStore>,
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
pub struct ReadParams {
    /// The block size passed to the underlying Object Store reader.
    ///
    /// This is used to control the minimal request size.
    pub block_size: Option<usize>,

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
}

impl Default for ReadParams {
    fn default() -> Self {
        Self {
            block_size: None,
            index_cache_size: DEFAULT_INDEX_CACHE_SIZE,
            metadata_cache_size: DEFAULT_METADATA_CACHE_SIZE,
            session: None,
            store_options: None,
        }
    }
}

impl Dataset {
    /// Open an existing dataset.
    pub async fn open(uri: &str) -> Result<Self> {
        let params = ReadParams::default();
        Self::open_with_params(uri, &params).await
    }

    /// Open a dataset with read params.
    pub async fn open_with_params(uri: &str, params: &ReadParams) -> Result<Self> {
        let (mut object_store, base_path) = match params.store_options.as_ref() {
            Some(store_options) => ObjectStore::from_uri_and_params(uri, store_options).await?,
            None => ObjectStore::from_uri(uri).await?,
        };

        if let Some(block_size) = params.block_size {
            object_store.set_block_size(block_size);
        }

        let latest_manifest = object_store
            .commit_handler
            .resolve_latest_version(&base_path, &object_store)
            .await
            .map_err(|e| Error::DatasetNotFound {
                path: base_path.to_string(),
                source: Box::new(e),
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
        let (mut object_store, base_path) = ObjectStore::from_uri(uri).await?;
        if let Some(block_size) = params.block_size {
            object_store.set_block_size(block_size);
        };

        let manifest_file = object_store
            .commit_handler
            .resolve_version(&base_path, version, &object_store)
            .await?;

        let session = if let Some(session) = params.session.as_ref() {
            session.clone()
        } else {
            Arc::new(Session::new(
                params.index_cache_size,
                params.metadata_cache_size,
            ))
        };
        Self::checkout_manifest(Arc::new(object_store), base_path, &manifest_file, session).await
    }

    /// Check out the specified version of this dataset
    pub async fn checkout_version(&self, version: u64) -> Result<Self> {
        let base_path = self.base.clone();
        let manifest_file = self
            .object_store
            .commit_handler
            .resolve_version(&base_path, version, &self.object_store)
            .await?;
        Self::checkout_manifest(
            self.object_store.clone(),
            base_path,
            &manifest_file,
            self.session.clone(),
        )
        .await
    }

    async fn checkout_manifest(
        object_store: Arc<ObjectStore>,
        base_path: Path,
        manifest_path: &Path,
        session: Arc<Session>,
    ) -> Result<Self> {
        let object_reader = object_store
            .open(manifest_path)
            .await
            .map_err(|e| match &e {
                Error::NotFound { uri } => Error::DatasetNotFound {
                    path: uri.clone(),
                    source: box_error(e),
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
            });
        }

        manifest
            .schema
            .load_dictionary(object_reader.as_ref())
            .await?;
        Ok(Self {
            object_store,
            base: base_path,
            manifest: Arc::new(manifest),
            session,
        })
    }

    async fn write_impl(
        batches: Box<dyn RecordBatchReader + Send>,
        uri: &str,
        params: Option<WriteParams>,
    ) -> Result<Self> {
        let mut params = params.unwrap_or_default();

        let (object_store, base) =
            ObjectStore::from_uri_and_params(uri, &params.store_params.clone().unwrap_or_default())
                .await?;

        // Read expected manifest path for the dataset
        let latest_manifest_path = object_store
            .commit_handler
            .resolve_latest_version(&base, &object_store)
            .await?;
        let flag_dataset_exists = object_store.exists(&latest_manifest_path).await?;

        let (stream, schema) = reader_to_stream(batches)?;

        // Running checks for the different write modes
        // create + dataset already exists = error
        if flag_dataset_exists && matches!(params.mode, WriteMode::Create) {
            return Err(Error::DatasetAlreadyExists {
                uri: uri.to_owned(),
            });
        }

        // append + dataset doesn't already exists = warn + switch to create mode
        if !flag_dataset_exists
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
                Self::open_with_params(
                    uri,
                    &ReadParams {
                        store_options: params.store_params.clone(),
                        ..Default::default()
                    },
                )
                .await?,
            )
        };

        // append + input schema different from existing schema = error
        if matches!(params.mode, WriteMode::Append) {
            if let Some(d) = dataset.as_ref() {
                let m = d.manifest.as_ref();
                if schema != m.schema {
                    return Err(Error::SchemaMismatch {
                        original: m.schema.clone(),
                        new: schema,
                    });
                }
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
                });
            }
        }

        let object_store = Arc::new(object_store);
        let fragments =
            write_fragments(object_store.clone(), &base, &schema, stream, params.clone()).await?;

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
                &transaction,
                &Default::default(),
                &Default::default(),
            )
            .await?
        } else {
            commit_new_dataset(&object_store, &base, &transaction, &Default::default()).await?
        };

        Ok(Self {
            object_store,
            base,
            manifest: Arc::new(manifest.clone()),
            session: Arc::new(Session::default()),
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

        // Need to include params here because it might include a commit mechanism.
        let object_store = Arc::new(
            self.object_store()
                .with_params(&params.store_params.clone().unwrap_or_default()),
        );

        let (stream, schema) = reader_to_stream(batches)?;

        // Return Error if append and input schema differ
        if self.manifest.schema != schema {
            return Err(Error::SchemaMismatch {
                original: self.manifest.schema.clone(),
                new: schema,
            });
        }

        let fragments = write_fragments(
            object_store.clone(),
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
            &object_store,
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

    async fn latest_manifest(&self) -> Result<Manifest> {
        read_manifest(
            &self.object_store,
            &self
                .object_store
                .commit_handler
                .resolve_latest_version(&self.base, &self.object_store)
                .await?,
        )
        .await
    }

    /// Restore the currently checked out version of the dataset as the latest version.
    ///
    /// Currently, `write_params` is just used to get additional store params.
    /// Other options are ignored.
    pub async fn restore(&mut self, write_params: Option<WriteParams>) -> Result<()> {
        let latest_manifest = self.latest_manifest().await?;
        let latest_version = latest_manifest.version;

        let transaction = Transaction::new(
            latest_version,
            Operation::Restore {
                version: self.manifest.version,
            },
            None,
        );

        let object_store =
            if let Some(store_params) = write_params.and_then(|params| params.store_params) {
                Arc::new(self.object_store.with_params(&store_params))
            } else {
                self.object_store.clone()
            };

        self.manifest = Arc::new(
            commit_transaction(
                self,
                &object_store,
                &transaction,
                &Default::default(),
                &Default::default(),
            )
            .await?,
        );

        Ok(())
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
    ) -> Result<Self> {
        let read_version = read_version.map_or_else(
            || match operation {
                Operation::Overwrite { .. } | Operation::Restore { .. } => Ok(0),
                _ => Err(Error::invalid_input(
                    "read_version must be specified for this operation",
                )),
            },
            Ok,
        )?;

        let (object_store, base) =
            ObjectStore::from_uri_and_params(base_uri, &store_params.clone().unwrap_or_default())
                .await?;

        // Test if the dataset exists
        let latest_manifest = object_store
            .commit_handler
            .resolve_latest_version(&base, &object_store)
            .await?;
        let flag_dataset_exists = object_store.exists(&latest_manifest).await?;

        if !flag_dataset_exists && !matches!(operation, Operation::Overwrite { .. }) {
            return Err(Error::DatasetNotFound {
                path: base.to_string(),
                source: "The dataset must already exist unless the operation is Overwrite".into(),
            });
        }

        let dataset = if flag_dataset_exists {
            Some(
                Self::open_with_params(
                    base_uri,
                    &ReadParams {
                        store_options: store_params.clone(),
                        ..Default::default()
                    },
                )
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
                &transaction,
                &Default::default(),
                &Default::default(),
            )
            .await?
        } else {
            commit_new_dataset(&object_store, &base, &transaction, &Default::default()).await?
        };

        Ok(Self {
            object_store: Arc::new(object_store),
            base,
            manifest: Arc::new(manifest.clone()),
            session: Arc::new(Session::default()),
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
            return Err(Error::invalid_input(format!(
                "Column {} does not exist in the left side dataset",
                left_on
            )));
        };
        let right_schema = stream.schema();
        if right_schema.field_with_name(right_on).is_err() {
            return Err(Error::invalid_input(format!(
                "Column {} does not exist in the right side dataset",
                right_on
            )));
        };
        for field in right_schema.fields() {
            if field.name() == right_on {
                // right_on is allowed to exist in the dataset, since it may be
                // the same as left_on.
                continue;
            }
            if self.schema().field(field.name()).is_some() {
                return Err(Error::invalid_input(format!(
                    "Column {} exists in both sides of the dataset",
                    field.name()
                )));
            }
        }

        // Hash join
        let joiner = Arc::new(HashJoiner::try_new(stream, right_on).await?);
        // Final schema is union of current schema, plus the RHS schema without
        // the right_on key.
        let new_schema: Schema = self.schema().merge(joiner.out_schema().as_ref())?;

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
    /// Create a Scanner to scan the dataset.
    pub fn scan(&self) -> Scanner {
        Scanner::new(Arc::new(self.clone()))
    }

    /// Count the number of rows in the dataset.
    ///
    /// It offers a fast path of counting rows by just computing via metadata.
    pub async fn count_rows(&self) -> Result<usize> {
        // Open file to read metadata.
        let counts = stream::iter(self.get_fragments())
            .map(|f| async move { f.count_rows().await })
            .buffer_unordered(16)
            .try_collect::<Vec<_>>()
            .await?;
        Ok(counts.iter().sum())
    }

    pub async fn take(&self, row_indices: &[usize], projection: &Schema) -> Result<RecordBatch> {
        let mut sorted_indices: Vec<u32> =
            Vec::from_iter(row_indices.iter().map(|indice| *indice as u32));
        sorted_indices.sort();

        let mut row_count = 0;
        let mut start = 0;
        let schema = Arc::new(ArrowSchema::from(projection));
        let mut batches = Vec::with_capacity(sorted_indices.len());
        for fragment in self.get_fragments().iter() {
            if start >= sorted_indices.len() {
                break;
            }

            let max_row_indices = row_count + fragment.count_rows().await? as u32;
            if sorted_indices[start] < max_row_indices {
                let mut end = start;
                sorted_indices[end] -= row_count;
                while end + 1 < sorted_indices.len() && sorted_indices[end + 1] < max_row_indices {
                    end += 1;
                    sorted_indices[end] -= row_count;
                }
                batches.push(
                    fragment
                        .take(&sorted_indices[start..end + 1], projection)
                        .await?,
                );

                // restore the row indices
                for indice in sorted_indices[start..end + 1].iter_mut() {
                    *indice += row_count;
                }

                start = end + 1;
            }
            row_count = max_row_indices;
        }

        let one_batch = concat_batches(&schema, &batches)?;
        let remapping_index: UInt64Array = row_indices
            .iter()
            .map(|o| sorted_indices.binary_search(&(*o as u32)).unwrap() as u64)
            .collect();
        let struct_arr: StructArray = one_batch.into();
        let reordered = take(&struct_arr, &remapping_index, None)?;
        Ok(as_struct_array(&reordered).into())
    }

    /// Take rows by the internal ROW ids.
    pub(crate) async fn take_rows(
        &self,
        row_ids: &[u64],
        projection: &Schema,
    ) -> Result<RecordBatch> {
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
        let fragment_and_indices = fragments.iter().filter_map(|f| {
            let local_row_ids = row_ids_per_fragment.get(&(f.id() as u64))?;
            Some((f, local_row_ids))
        });

        let projection_with_row_id = Schema::merge(
            projection,
            &ArrowSchema::new(vec![ArrowField::new(
                ROW_ID,
                arrow::datatypes::DataType::UInt64,
                false,
            )]),
        )?;
        let schema_with_row_id = Arc::new(ArrowSchema::from(&projection_with_row_id));

        let batches = stream::iter(fragment_and_indices)
            .then(|(fragment, indices)| fragment.take_rows(indices, projection, true))
            .try_collect::<Vec<_>>()
            .await?;

        let one_batch = concat_batches(&schema_with_row_id, &batches)?;
        // Note: one_batch may contains fewer rows than the number of requested
        // row ids because some rows may have been deleted. Because of this, we
        // get the results with row ids so that we can re-order the results
        // to match the requested order.

        let returned_row_ids = one_batch
            .column_by_name(ROW_ID)
            .ok_or_else(|| Error::Internal {
                message: "ROW_ID column not found".into(),
            })?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Internal {
                message: "ROW_ID column is not UInt64Array".into(),
            })?
            .values();

        let remapping_index: UInt64Array = row_ids
            .iter()
            .filter_map(|o| returned_row_ids.binary_search(o).ok().map(|pos| pos as u64))
            .collect();

        // Remove the row id column.
        let keep_indices = (0..one_batch.num_columns() - 1).collect::<Vec<_>>();
        let one_batch = one_batch.project(&keep_indices)?;
        let struct_arr: StructArray = one_batch.into();
        let reordered = take(&struct_arr, &remapping_index, None)?;
        Ok(as_struct_array(&reordered).into())
    }

    /// Sample `n` rows from the dataset.
    pub(crate) async fn sample(&self, n: usize, projection: &Schema) -> Result<RecordBatch> {
        use rand::seq::IteratorRandom;
        let num_rows = self.count_rows().await?;
        let ids = (0..num_rows).choose_multiple(&mut rand::thread_rng(), n);
        self.take(&ids[..], projection).await
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
            &transaction,
            &Default::default(),
            &Default::default(),
        )
        .await?;

        self.manifest = Arc::new(manifest);

        Ok(())
    }

    pub(crate) fn object_store(&self) -> &ObjectStore {
        &self.object_store
    }

    async fn manifest_file(&self, version: u64) -> Result<Path> {
        self.object_store
            .commit_handler
            .resolve_version(&self.base, version, &self.object_store)
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

    /// Get all versions.
    pub async fn versions(&self) -> Result<Vec<Version>> {
        let mut versions: Vec<Version> = self
            .object_store
            .commit_handler
            .list_manifests(&self.base, &self.object_store)
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

    /// Read all indices of this Dataset version.
    pub async fn load_indices(&self) -> Result<Vec<Index>> {
        let manifest_file = self.manifest_file(self.version().version).await?;
        read_manifest_indexes(&self.object_store, &manifest_file, &self.manifest).await
    }

    /// Find index with a given index_name and return its serialized statistics.
    pub async fn index_statistics(&self, index_name: &str) -> Result<Option<String>> {
        let index_uuid = self
            .load_indices()
            .await
            .unwrap()
            .iter()
            .find(|idx| idx.name.eq(index_name))
            .map(|idx| idx.uuid.to_string());

        if let Some(index_uuid) = index_uuid {
            let index_statistics = open_index(Arc::new(self.clone()), "vector", &index_uuid)
                .await?
                .statistics()
                .unwrap();
            Ok(Some(serde_json::to_string(&index_statistics).unwrap()))
        } else {
            Ok(None)
        }
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
    base_path: &Path,
    manifest: &mut Manifest,
    indices: Option<Vec<Index>>,
    config: &ManifestWriteConfig,
) -> std::result::Result<(), CommitError> {
    if config.auto_set_feature_flags {
        apply_feature_flags(manifest);
    }
    manifest.set_timestamp(config.timestamp);

    manifest.update_max_fragment_id();

    object_store
        .commit_handler
        .commit(
            manifest,
            indices,
            base_path,
            object_store,
            write_manifest_file_to_path,
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
        let mut object_writer = object_store.create(path).await?;
        let pos = write_manifest(&mut object_writer, manifest, indices).await?;
        object_writer.write_magics(pos).await?;
        object_writer.shutdown().await?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::ops::Range;
    use std::vec;

    use super::*;
    use crate::arrow::FixedSizeListArrayExt;
    use crate::datatypes::Schema;
    use crate::index::vector::MetricType;
    use crate::index::IndexType;
    use crate::index::{vector::VectorIndexParams, DatasetIndexExt};
    use crate::io::deletion::read_deletion_file;

    use crate::dataset::WriteMode::Overwrite;
    use arrow_array::{
        cast::{as_string_array, as_struct_array},
        DictionaryArray, Float32Array, Int32Array, Int64Array, Int8Array, Int8DictionaryArray,
        RecordBatch, RecordBatchIterator, StringArray, UInt16Array, UInt32Array,
    };
    use arrow_ord::sort::sort_to_indices;
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use arrow_select::take::take;
    use futures::stream::TryStreamExt;
    use lance_testing::datagen::generate_random_array;
    use tempfile::tempdir;

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

    #[tokio::test]
    async fn test_create_dataset() {
        // Appending / Overwriting a dataset that does not exist is treated as Create
        for mode in [WriteMode::Create, WriteMode::Append, Overwrite] {
            let test_dir = tempdir().unwrap();
            create_file(test_dir.path(), mode).await
        }
    }

    #[tokio::test]
    async fn test_create_and_fill_empty_dataset() {
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
        assert_eq!(result.count_rows().await.unwrap(), 0);
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
        assert_eq!(actual_ds.count_rows().await.unwrap(), 10);
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

        assert_eq!(dataset.count_rows().await.unwrap(), num_rows);

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 10);
        for fragment in &fragments {
            assert_eq!(fragment.count_rows().await.unwrap(), 100);
            let reader = fragment.open(dataset.schema()).await.unwrap();
            assert_eq!(reader.num_batches(), 10);
            for i in 0..reader.num_batches() {
                assert_eq!(reader.num_rows_in_batch(i), 10);
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
                .object_store()
                .commit_handler
                .resolve_latest_version(&dataset.base, dataset.object_store())
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
                .object_store()
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
        manifest.writer_feature_flags = 5; // Set another flag
        manifest.reader_feature_flags = 5;
        manifest.version += 1;
        write_manifest_file(
            dataset.object_store(),
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
        assert_eq!(dataset.count_rows().await.unwrap(), 400);
        let projection = Schema::try_from(schema.as_ref()).unwrap();
        let values = dataset
            .take(
                &[
                    200, // 200
                    199, // 199
                    39,  // 39
                    40,  // 40
                    100, // 100
                ],
                &projection,
            )
            .await
            .unwrap();
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

        assert_eq!(dataset.count_rows().await.unwrap(), 400);
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
        assert_eq!(10, dataset.fragments().len());
        assert_eq!(400, dataset.count_rows().await.unwrap());
        dataset.validate().await.unwrap();
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

        let dataset = Dataset::write(reader, test_uri, None).await.unwrap();
        dataset.validate().await.unwrap();

        // Make sure valid arguments should create index successfully
        let params = VectorIndexParams::ivf_pq(10, 8, 2, false, MetricType::L2, 50);
        let dataset = dataset
            .create_index(&["embeddings"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
        dataset.validate().await.unwrap();

        // The version should match the table version it was created from.
        let indices = dataset.load_indices().await.unwrap();
        let actual = indices.first().unwrap().dataset_version;
        let expected = dataset.manifest.version - 1;
        assert_eq!(actual, expected);

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

        let expected_statistics =
            "{\"index_type\":\"IVF\",\"metric_type\":\"l2\",\"num_partitions\":10";
        let actual_statistics = dataset
            .index_statistics("embeddings_idx")
            .await
            .unwrap()
            .unwrap();
        assert!(actual_statistics.starts_with(expected_statistics));

        assert_eq!(
            dataset.index_statistics("non-existent_idx").await.unwrap(),
            None
        );
        assert_eq!(dataset.index_statistics("").await.unwrap(), None);

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
    }

    #[tokio::test]
    async fn test_delete() {
        fn sequence_data(range: Range<u32>) -> RecordBatch {
            let schema = Arc::new(ArrowSchema::new(vec![Field::new(
                "i",
                DataType::UInt32,
                false,
            )]));
            RecordBatch::try_new(schema, vec![Arc::new(UInt32Array::from_iter_values(range))])
                .unwrap()
        }
        // Write a dataset
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "i",
            DataType::UInt32,
            false,
        )]));
        let data = sequence_data(0..100);
        let batches = RecordBatchIterator::new(vec![data].into_iter().map(Ok), schema.clone());
        let write_params = WriteParams {
            max_rows_per_file: 50, // Split over two files.
            ..Default::default()
        };
        let mut dataset = Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        // Delete nothing
        dataset.delete("i < 0").await.unwrap();
        dataset.validate().await.unwrap();

        // We should not have any deletion file still
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
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
        assert!(fragments[0].metadata.deletion_file.is_some());
        assert!(fragments[1].metadata.deletion_file.is_some());

        // The deletion file should contain 20 rows
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
        assert_eq!(fragments[0].id(), 0);

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
        assert_eq!(fragments[0].metadata.deletion_file, None);
        assert_eq!(dataset.manifest, original_manifest);

        // Restore to a previous version
        dataset.restore(None).await.unwrap();
        assert_eq!(dataset.manifest.version, 3);
        assert_eq!(dataset.manifest.fragments, original_manifest.fragments);
        assert_eq!(dataset.manifest.schema, original_manifest.schema);

        // Delete some rows again (make sure we can still write as usual)
        dataset.delete("i > 30").await.unwrap();
        assert_eq!(dataset.manifest.version, 4);
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert!(fragments[0].metadata.deletion_file.is_some());
    }
}
