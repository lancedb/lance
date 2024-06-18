// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::{collections::HashMap, sync::Arc, time::Duration};

use lance_file::datatypes::populate_schema_dictionary;
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use lance_table::{
    format::Manifest,
    io::commit::{commit_handler_from_url, CommitHandler, ManifestLocation},
};
use object_store::{aws::AwsCredentialProvider, path::Path, DynObjectStore};
use prost::Message;
use snafu::{location, Location};
use tracing::instrument;
use url::Url;

use super::{ReadParams, WriteParams, DEFAULT_INDEX_CACHE_SIZE, DEFAULT_METADATA_CACHE_SIZE};
use crate::{
    error::{Error, Result},
    session::Session,
    Dataset,
};
/// builder for loading a [`Dataset`].
#[derive(Debug, Clone)]
pub struct DatasetBuilder {
    /// Cache size for index cache. If it is zero, index cache is disabled.
    index_cache_size: usize,
    /// Metadata cache size for the fragment metadata. If it is zero, metadata
    /// cache is disabled.
    metadata_cache_size: usize,
    /// Optional pre-loaded manifest to avoid loading it again.
    manifest: Option<Manifest>,
    session: Option<Arc<Session>>,
    commit_handler: Option<Arc<dyn CommitHandler>>,
    options: ObjectStoreParams,
    version: Option<u64>,
    table_uri: String,
    object_store_registry: Arc<ObjectStoreRegistry>,
}

impl DatasetBuilder {
    pub fn from_uri<T: AsRef<str>>(table_uri: T) -> Self {
        Self {
            index_cache_size: DEFAULT_INDEX_CACHE_SIZE,
            metadata_cache_size: DEFAULT_METADATA_CACHE_SIZE,
            table_uri: table_uri.as_ref().to_string(),
            options: ObjectStoreParams::default(),
            commit_handler: None,
            session: None,
            version: None,
            manifest: None,
            object_store_registry: Arc::new(ObjectStoreRegistry::default()),
        }
    }
}

// Much of this builder is directly inspired from the to delta-rs table builder implementation
// https://github.com/delta-io/delta-rs/main/crates/deltalake-core/src/table/builder.rs
impl DatasetBuilder {
    /// Set the cache size for indices. Set to zero, to disable the cache.
    pub fn with_index_cache_size(mut self, cache_size: usize) -> Self {
        self.index_cache_size = cache_size;
        self
    }

    /// Set the cache size for the file metadata. Set to zero to disable this cache.
    pub fn with_metadata_cache_size(mut self, cache_size: usize) -> Self {
        self.metadata_cache_size = cache_size;
        self
    }

    /// The block size passed to the underlying Object Store reader.
    ///
    /// This is used to control the minimal request size.
    /// Defaults to 4KB for local files and 64KB for others
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.options.block_size = Some(block_size);
        self
    }

    /// Sets `version` to the builder
    pub fn with_version(mut self, version: u64) -> Self {
        self.version = Some(version);
        self
    }

    pub fn with_commit_handler(mut self, commit_handler: Arc<dyn CommitHandler>) -> Self {
        self.commit_handler = Some(commit_handler);
        self
    }

    /// Sets the s3 credentials refresh.
    /// This only applies to s3 storage.
    pub fn with_s3_credentials_refresh_offset(mut self, offset: Duration) -> Self {
        self.options.s3_credentials_refresh_offset = offset;
        self
    }

    /// Sets the aws credentials provider.
    /// This only applies to aws object store.
    pub fn with_aws_credentials_provider(mut self, credentials: AwsCredentialProvider) -> Self {
        self.options.aws_credentials = Some(credentials);
        self
    }

    /// Directly set the object store to use.
    pub fn with_object_store(
        mut self,
        object_store: Arc<DynObjectStore>,
        location: Url,
        commit_handler: Arc<dyn CommitHandler>,
    ) -> Self {
        self.options.object_store = Some((object_store, location));
        self.commit_handler = Some(commit_handler);
        self
    }

    /// Use a serialized manifest instead of loading it from the object store.
    ///
    /// This is common when transferring a dataset across IPC boundaries.
    pub fn with_serialized_manifest(mut self, manifest: &[u8]) -> Result<Self> {
        let manifest = Manifest::try_from(lance_table::format::pb::Manifest::decode(manifest)?)?;
        self.manifest = Some(manifest);
        Ok(self)
    }

    /// Set options used to initialize storage backend
    ///
    /// Options may be passed in the HashMap or set as environment variables. See documentation of
    /// underlying object store implementation for details.
    ///
    /// - [Azure options](https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html#variants)
    /// - [S3 options](https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html#variants)
    /// - [Google options](https://docs.rs/object_store/latest/object_store/gcp/enum.GoogleConfigKey.html#variants)
    pub fn with_storage_options(mut self, storage_options: HashMap<String, String>) -> Self {
        self.options.storage_options = Some(storage_options);
        self
    }

    /// Set a single option used to initialize storage backend
    /// For example, to set the region for S3, you can use:
    ///
    /// ```ignore
    /// let builder = DatasetBuilder::from_uri("s3://bucket/path")
    ///     .with_storage_option("region", "us-east-1");
    /// ```
    pub fn with_storage_option(mut self, key: impl AsRef<str>, value: impl AsRef<str>) -> Self {
        let mut storage_options = self.options.storage_options.unwrap_or_default();
        storage_options.insert(key.as_ref().to_string(), value.as_ref().to_string());
        self.options.storage_options = Some(storage_options);
        self
    }

    /// Set options based on [ReadParams].
    pub fn with_read_params(mut self, read_params: ReadParams) -> Self {
        self = self
            .with_index_cache_size(read_params.index_cache_size)
            .with_metadata_cache_size(read_params.metadata_cache_size);

        if let Some(options) = read_params.store_options {
            self.options = options;
        }

        if let Some(session) = read_params.session {
            self.session = Some(session);
        }

        if let Some(commit_handler) = read_params.commit_handler {
            self.commit_handler = Some(commit_handler);
        }

        self.object_store_registry = read_params.object_store_registry.clone();

        self
    }

    /// Set options based on [WriteParams].
    pub fn with_write_params(mut self, write_params: WriteParams) -> Self {
        if let Some(options) = write_params.store_params {
            self.options = options;
        }

        if let Some(commit_handler) = write_params.commit_handler {
            self.commit_handler = Some(commit_handler);
        }

        self.object_store_registry = write_params.object_store_registry.clone();

        self
    }

    /// Re-use an existing session.
    ///
    /// The session holds caches for index and metadata.
    ///
    /// If this is set, then `with_index_cache_size` and `with_metadata_cache_size` are ignored.
    pub fn with_session(mut self, session: Arc<Session>) -> Self {
        self.session = Some(session);
        self
    }

    pub fn with_object_store_registry(mut self, registry: Arc<ObjectStoreRegistry>) -> Self {
        self.object_store_registry = registry;
        self
    }

    /// Build a lance object store for the given config
    pub async fn build_object_store(self) -> Result<(ObjectStore, Path, Arc<dyn CommitHandler>)> {
        let commit_handler = match self.commit_handler {
            Some(commit_handler) => Ok(commit_handler),
            None => commit_handler_from_url(&self.table_uri, &Some(self.options.clone())).await,
        }?;

        match &self.options.object_store {
            Some(store) => Ok((
                ObjectStore::new(
                    store.0.clone(),
                    store.1.clone(),
                    self.options.block_size,
                    self.options.object_store_wrapper,
                ),
                Path::from(store.1.path()),
                commit_handler,
            )),
            None => {
                let (store, path) = ObjectStore::from_uri_and_params(
                    self.object_store_registry.clone(),
                    &self.table_uri,
                    &self.options,
                )
                .await?;
                Ok((store, path, commit_handler))
            }
        }
    }

    #[instrument(skip_all)]
    pub async fn load(mut self) -> Result<Dataset> {
        let session = match self.session.take() {
            Some(session) => session,
            None => Arc::new(Session::new(
                self.index_cache_size,
                self.metadata_cache_size,
            )),
        };

        let version = self.version;
        let table_uri = self.table_uri.clone();

        let manifest = self.manifest.take();

        let (object_store, base_path, commit_handler) = self.build_object_store().await?;

        let manifest = if manifest.is_some() {
            let mut manifest = manifest.unwrap();
            if manifest.schema.has_dictionary_types() {
                let path = commit_handler
                    .resolve_version(&base_path, manifest.version, &object_store.inner)
                    .await?;
                let reader = object_store.open(&path).await?;
                populate_schema_dictionary(&mut manifest.schema, reader.as_ref()).await?;
            }
            manifest
        } else {
            let manifest_location = match version {
                Some(version) => {
                    let path = commit_handler
                        .resolve_version(&base_path, version, &object_store.inner)
                        .await?;
                    ManifestLocation {
                        version,
                        path,
                        size: None,
                    }
                }
                None => commit_handler
                    .resolve_latest_location(&base_path, &object_store)
                    .await
                    .map_err(|e| Error::DatasetNotFound {
                        source: Box::new(e),
                        path: base_path.to_string(),
                        location: location!(),
                    })?,
            };

            Dataset::load_manifest(&object_store, &manifest_location).await?
        };

        Dataset::checkout_manifest(
            Arc::new(object_store),
            base_path,
            table_uri,
            manifest,
            session,
            commit_handler,
        )
        .await
    }
}
