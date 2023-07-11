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
use std::time::SystemTime;

use arrow_array::{
    cast::as_struct_array, RecordBatch, RecordBatchReader, StructArray, UInt64Array,
};
use arrow_schema::Schema as ArrowSchema;
use arrow_select::{concat::concat_batches, take::take};
use chrono::prelude::*;
use futures::stream::{self, StreamExt, TryStreamExt};
use log::warn;
use object_store::path::Path;
use uuid::Uuid;

mod feature_flags;
pub mod fragment;
mod hash_joiner;
pub mod scanner;
pub mod updater;
mod write;

use self::feature_flags::{apply_feature_flags, can_read_dataset, can_write_dataset};
use self::fragment::FileFragment;
use self::scanner::Scanner;
use crate::datatypes::Schema;
use crate::error::box_error;
use crate::format::{pb, Fragment, Index, Manifest};
use crate::io::object_store::ObjectStoreParams;
use crate::io::{
    object_reader::{read_message, read_struct},
    read_manifest, read_metadata_offset, write_manifest, FileWriter, ObjectStore,
};
use crate::session::Session;
use crate::{Error, Result};
use hash_joiner::HashJoiner;
pub use scanner::ROW_ID;
pub use write::*;

const LATEST_MANIFEST_NAME: &str = "_latest.manifest";
const VERSIONS_DIR: &str = "_versions";
const INDICES_DIR: &str = "_indices";
pub(crate) const DELETION_DIRS: &str = "_deletions";
const DATA_DIR: &str = "data";
pub(crate) const DEFAULT_INDEX_CACHE_SIZE: usize = 256;

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

/// Create a new [FileWriter] with the related `data_file_path` under `<DATA_DIR>`.
async fn new_file_writer(
    object_store: &ObjectStore,
    base_dir: &Path,
    data_file_path: &str,
    schema: &Schema,
) -> Result<FileWriter> {
    let full_path = base_dir.child(DATA_DIR).child(data_file_path);
    FileWriter::try_new(object_store, &full_path, schema.clone()).await
}

/// Get the manifest file path for a version.
fn manifest_path(base: &Path, version: u64) -> Path {
    base.child(VERSIONS_DIR)
        .child(format!("{version}.manifest"))
}

/// Get the latest manifest path
pub(crate) fn latest_manifest_path(base: &Path) -> Path {
    base.child(LATEST_MANIFEST_NAME)
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
        let (mut object_store, base_path) = match params.store_options.clone() {
            Some(store_options) => ObjectStore::from_uri_and_params(uri, store_options).await?,
            None => ObjectStore::from_uri(uri).await?,
        };

        if let Some(block_size) = params.block_size {
            object_store.set_block_size(block_size);
        }

        let latest_manifest_path = latest_manifest_path(&base_path);

        let session = if let Some(session) = params.session.as_ref() {
            session.clone()
        } else {
            Arc::new(Session::new(params.index_cache_size))
        };
        Self::checkout_manifest(
            Arc::new(object_store),
            base_path,
            &latest_manifest_path,
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

        let manifest_file = manifest_path(&base_path, version);
        let session = if let Some(session) = params.session.as_ref() {
            session.clone()
        } else {
            Arc::new(Session::new(params.index_cache_size))
        };
        Self::checkout_manifest(Arc::new(object_store), base_path, &manifest_file, session).await
    }

    /// Check out the specified version of this dataset
    pub async fn checkout_version(&self, version: u64) -> Result<Self> {
        let base_path = self.base.clone();
        let manifest_file = manifest_path(&base_path, version);
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

    pub async fn write_impl(
        batches: Box<dyn RecordBatchReader + Send>,
        uri: &str,
        params: Option<WriteParams>,
    ) -> Result<Self> {
        let mut params = params.unwrap_or_default();

        let (object_store, base) =
            ObjectStore::from_uri_and_params(uri, params.store_params.clone().unwrap_or_default())
                .await?;

        // Read expected manifest path for the dataset
        let latest_manifest_path = latest_manifest_path(&base);
        let flag_dataset_exists = object_store.exists(&latest_manifest_path).await?;

        let mut schema: Schema = Schema::try_from(batches.schema().as_ref())?;
        let mut peekable = batches.peekable();
        if let Some(batch) = peekable.peek() {
            if let Ok(b) = batch {
                schema.set_dictionary(b)?;
            } else {
                return Err(Error::from(batch.as_ref().unwrap_err()));
            }
        }
        schema.validate()?;

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
            Some(Self::open(uri).await?)
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

        let mut fragment_id = if matches!(params.mode, WriteMode::Append) {
            dataset.as_ref().map_or(0, |d| {
                d.manifest
                    .fragments
                    .iter()
                    .map(|f| f.id)
                    .max()
                    .map(|id| id + 1)
                    .unwrap_or(0)
            })
        } else {
            // Create or Overwrite.
            // Overwrite resets the fragment ID to zero.
            0
        };

        let mut fragments: Vec<Fragment> = if matches!(params.mode, WriteMode::Append) {
            dataset
                .as_ref()
                .map_or(vec![], |d| d.manifest.fragments.as_ref().clone())
        } else {
            // Create or Overwrite create new fragments.
            vec![]
        };

        let mut writer = None;
        let mut batches: Vec<RecordBatch> = Vec::new();
        let mut num_rows: usize = 0;
        for batch_result in peekable {
            let batch: RecordBatch = batch_result?;
            batches.push(batch.clone());
            num_rows += batch.num_rows();
            if num_rows >= params.max_rows_per_group {
                // TODO: the max rows per group boundary is not accurately calculated yet.
                if writer.is_none() {
                    writer = {
                        let file_path = format!("{}.lance", Uuid::new_v4());
                        let fragment = Fragment::with_file(fragment_id, &file_path, &schema);
                        fragments.push(fragment);
                        fragment_id += 1;
                        Some(new_file_writer(&object_store, &base, &file_path, &schema).await?)
                    }
                };

                writer.as_mut().unwrap().write(&batches).await?;
                batches = Vec::new();
                num_rows = 0;
            }
            if let Some(w) = writer.as_mut() {
                if w.len() >= params.max_rows_per_file {
                    w.finish().await?;
                    writer = None;
                }
            }
        }
        if num_rows > 0 {
            if writer.is_none() {
                writer = {
                    let file_path = format!("{}.lance", Uuid::new_v4());
                    let fragment = Fragment::with_file(fragment_id, &file_path, &schema);
                    fragments.push(fragment);
                    Some(new_file_writer(&object_store, &base, &file_path, &schema).await?)
                }
            };
            writer.as_mut().unwrap().write(&batches).await?;
        }
        if let Some(w) = writer.as_mut() {
            // Drop the last writer.
            w.finish().await?;
            drop(writer);
        };

        let mut manifest = Manifest::new(&schema, Arc::new(fragments));
        manifest.version = match dataset.as_ref() {
            Some(d) => d
                .latest_manifest()
                .await
                .map(|m| m.version + 1)
                .unwrap_or(1),
            None => 1,
        };
        // Inherit the index if we're just appending rows
        let indices = match (&dataset, &params.mode) {
            (Some(d), WriteMode::Append) => Some(d.load_indices().await?),
            _ => None,
        };
        write_manifest_file(
            &object_store,
            &base,
            &mut manifest,
            indices,
            Default::default(),
        )
        .await?;

        Ok(Self {
            object_store: Arc::new(object_store),
            base,
            manifest: Arc::new(manifest.clone()),
            session: Arc::new(Session::default()),
        })
    }

    /// Write to or Create a [Dataset] with a stream of [RecordBatch]s.
    ///
    /// Returns the newly created [`Dataset`]. Returns [Error] if the dataset already exists.
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

    /// Create a new version of [`Dataset`] from a collection of fragments.
    pub async fn commit(
        base_uri: &str,
        schema: &Schema,
        fragments: &[Fragment],
        mode: WriteMode,
    ) -> Result<Self> {
        let (object_store, base) = ObjectStore::from_uri(base_uri).await?;
        let latest_manifest = latest_manifest_path(&base);
        let mut indices = vec![];
        let mut version = 1;
        let schema = if object_store.exists(&latest_manifest).await? {
            let dataset = Self::open(base_uri).await?;
            version = dataset.version().version + 1;

            if matches!(mode, WriteMode::Append) {
                // Append mode: inherit indices from previous version.
                indices = dataset.load_indices().await?;
            }

            let dataset_schema = dataset.schema();
            let added_on_schema = schema.exclude(dataset_schema)?;
            dataset_schema.merge(&added_on_schema)?
        } else {
            schema.clone()
        };

        let mut manifest = Manifest::new(&schema, Arc::new(fragments.to_vec()));
        manifest.version = version;

        // Preserve indices.
        write_manifest_file(
            &object_store,
            &base,
            &mut manifest,
            Some(indices),
            Default::default(),
        )
        .await?;
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
    pub async fn merge_impl(
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

        // Inherit the index, since we are just adding columns.
        let indices = self.load_indices().await?;

        let mut manifest = Manifest::new(self.schema(), Arc::new(updated_fragments));
        manifest.version = self
            .latest_manifest()
            .await
            .map(|m| m.version + 1)
            .unwrap_or(1);
        manifest.schema = new_schema;

        write_manifest_file(
            &self.object_store,
            &self.base,
            &mut manifest,
            Some(indices),
            Default::default(),
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
        let schema = Arc::new(ArrowSchema::from(projection));
        let fragments = self.get_fragments();
        let fragment_and_indices = fragments
            .iter()
            .map(|f| {
                (
                    f,
                    row_ids_per_fragment.get(&(f.id() as u64)),
                    schema.clone(),
                )
            })
            .collect::<Vec<_>>();
        let batches = stream::iter(fragment_and_indices)
            .then(|(fragment, indices_opt, schema)| async move {
                let Some(indices) = indices_opt else {
                    return Ok(RecordBatch::new_empty(schema));
                };
                fragment.take(indices.as_slice(), projection).await
            })
            .try_collect::<Vec<_>>()
            .await?;
        let one_batch = concat_batches(&schema, &batches)?;

        let remapping_index: UInt64Array = row_ids
            .iter()
            .map(|o| sorted_row_ids.binary_search(o).unwrap() as u64)
            .collect();
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
        let mut updated_fragments = stream::iter(self.get_fragments())
            .map(|f| async move { f.delete(predicate).await.map(|f| f.map(|f| f.metadata)) })
            .buffer_unordered(num_cpus::get())
            // Drop the fragments that were deleted.
            .try_filter_map(|f| futures::future::ready(Ok(f)))
            .try_collect::<Vec<_>>()
            .await?;

        // Maintain the order of fragment IDs.
        updated_fragments.sort_by_key(|f| f.id);

        // Inherit the index, unless we deleted all the fragments.
        let indices = if updated_fragments.is_empty() {
            None
        } else {
            Some(self.load_indices().await?)
        };

        let mut manifest = Manifest::new(self.schema(), Arc::new(updated_fragments));
        manifest.version = self
            .latest_manifest()
            .await
            .map(|m| m.version + 1)
            .unwrap_or(1);

        write_manifest_file(
            &self.object_store,
            &self.base,
            &mut manifest,
            indices,
            Default::default(),
        )
        .await?;

        self.manifest = Arc::new(manifest);

        Ok(())
    }

    pub(crate) fn object_store(&self) -> &ObjectStore {
        &self.object_store
    }

    fn versions_dir(&self) -> Path {
        self.base.child(VERSIONS_DIR)
    }

    fn manifest_file(&self, version: u64) -> Path {
        self.versions_dir().child(format!("{version}.manifest"))
    }

    fn latest_manifest_path(&self) -> Path {
        latest_manifest_path(&self.base)
    }

    pub(crate) async fn latest_manifest(&self) -> Result<Manifest> {
        read_manifest(&self.object_store, &self.latest_manifest_path()).await
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
        let paths: Vec<Path> = self
            .object_store
            .inner
            .list_with_delimiter(Some(&self.versions_dir()))
            .await?
            .objects
            .iter()
            .filter(|&obj| obj.location.as_ref().ends_with(".manifest"))
            .map(|o| o.location.clone())
            .collect();
        let mut versions = vec![];
        for path in paths.iter() {
            let manifest = read_manifest(&self.object_store, path).await?;
            versions.push(Version::from(&manifest));
        }
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
        if let Some(pos) = self.manifest.index_section.as_ref() {
            let manifest_file = self.manifest_file(self.version().version);

            let reader = self.object_store.open(&manifest_file).await?;
            let section: pb::IndexSection = read_message(reader.as_ref(), *pos).await?;

            Ok(section
                .indices
                .iter()
                .map(Index::try_from)
                .collect::<Result<Vec<_>>>()?)
        } else {
            Ok(vec![])
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

/// Finish writing the manifest file, and commit the changes by linking the latest manifest file
/// to this version.
pub(crate) async fn write_manifest_file(
    object_store: &ObjectStore,
    base_path: &Path,
    manifest: &mut Manifest,
    indices: Option<Vec<Index>>,
    config: ManifestWriteConfig,
) -> Result<()> {
    if config.auto_set_feature_flags {
        apply_feature_flags(manifest);
    }
    manifest.set_timestamp(config.timestamp);

    let paths = vec![
        manifest_path(base_path, manifest.version),
        latest_manifest_path(base_path),
    ];

    for p in paths {
        write_manifest_file_to_path(object_store, manifest, indices.clone(), &p).await?
    }
    Ok(())
}

async fn write_manifest_file_to_path(
    object_store: &ObjectStore,
    manifest: &mut Manifest,
    indices: Option<Vec<Index>>,
    path: &Path,
) -> Result<()> {
    let mut object_writer = object_store.create(path).await?;
    let pos = write_manifest(&mut object_writer, manifest, indices).await?;
    object_writer.write_magics(pos).await?;
    object_writer.shutdown().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::ops::Range;
    use std::vec;

    use super::*;
    use crate::arrow::FixedSizeListArrayExt;
    use crate::index::vector::MetricType;
    use crate::index::IndexType;
    use crate::index::{vector::VectorIndexParams, DatasetIndexExt};
    use crate::io::deletion::read_deletion_file;
    use crate::{datatypes::Schema, utils::testing::generate_random_array};

    use crate::dataset::WriteMode::Overwrite;
    use arrow_array::{
        cast::{as_string_array, as_struct_array},
        DictionaryArray, Int32Array, RecordBatch, StringArray, UInt16Array,
    };
    use arrow_array::{Float32Array, RecordBatchIterator, UInt32Array};
    use arrow_ord::sort::sort_to_indices;
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use arrow_select::take::take;
    use futures::stream::TryStreamExt;
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
        let manifest = read_manifest(dataset.object_store(), &latest_manifest_path(&dataset.base))
            .await
            .unwrap();
        assert_eq!(manifest.writer_feature_flags, 0);
        assert_eq!(manifest.reader_feature_flags, 0);

        // Create one with deletions
        dataset.delete("i < 10").await.unwrap();
        dataset.validate().await.unwrap();

        // Check it set the flag
        let mut manifest =
            read_manifest(dataset.object_store(), &latest_manifest_path(&dataset.base))
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
        write_manifest_file(
            dataset.object_store(),
            &dataset.base,
            &mut manifest,
            None,
            ManifestWriteConfig {
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
        Dataset::write(batches, test_uri, Some(write_params.clone()))
            .await
            .unwrap();

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
        Dataset::write(new_batch_reader, test_uri, Some(write_params.clone()))
            .await
            .unwrap();

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
        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        assert_eq!(dataset.count_rows().await.unwrap(), 400);
        let projection = Schema::try_from(schema.as_ref()).unwrap();
        let values = dataset
            .take_rows(
                &[
                    5_u64 << 32,        // 200
                    (4_u64 << 32) + 39, // 199
                    39,                 // 39
                    1_u64 << 32,        // 40
                    (2_u64 << 32) + 20, // 100
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

        // Check the version is set correctly
        let indices = dataset.load_indices().await.unwrap();
        let actual = indices.first().unwrap().dataset_version;
        let expected = dataset.manifest.version;
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
        let expected = dataset.manifest.version - 1;
        assert_eq!(actual, expected);
        dataset.validate().await.unwrap();

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
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        // Delete nothing
        dataset.delete("i < 0").await.unwrap();
        dataset.validate().await.unwrap();

        // We should not have any deletion file still
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert!(fragments[0].metadata.deletion_file.is_none());

        // Delete rows
        dataset.delete("i < 10 OR i >= 90").await.unwrap();
        dataset.validate().await.unwrap();

        // Verify result:
        // There should be a deletion file in the metadata
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert!(fragments[0].metadata.deletion_file.is_some());

        // The deletion file should contain 20 rows
        let store = dataset.object_store().clone();
        let path = Path::from_filesystem_path(test_uri).unwrap();
        let deletion_vector = read_deletion_file(&path, &fragments[0].metadata, &store)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(deletion_vector.len(), 20);
        assert_eq!(
            deletion_vector.into_iter().collect::<HashSet<_>>(),
            (0..10).chain(90..100).collect::<HashSet<_>>()
        );

        // Delete more rows
        dataset.delete("i < 20").await.unwrap();
        dataset.validate().await.unwrap();

        // Verify result
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert!(fragments[0].metadata.deletion_file.is_some());
        let deletion_vector = read_deletion_file(&path, &fragments[0].metadata, &store)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(deletion_vector.len(), 30);
        assert_eq!(
            deletion_vector.into_iter().collect::<HashSet<_>>(),
            (0..20).chain(90..100).collect::<HashSet<_>>()
        );

        // Delete full fragment
        dataset.delete("i >= 20").await.unwrap();
        dataset.validate().await.unwrap();

        // Verify fragment is fully gone
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 0);
    }
}
