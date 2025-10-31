// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::{collections::HashMap, sync::Arc, time::Duration};

use super::refs::{Ref, Refs};
use super::{ReadParams, WriteParams, DEFAULT_INDEX_CACHE_SIZE, DEFAULT_METADATA_CACHE_SIZE};
use crate::dataset::branch_location::BranchLocation;
use crate::{session::Session, Dataset, Error, Result};
use futures::FutureExt;
use lance_core::utils::tracing::{DATASET_LOADING_EVENT, TRACE_DATASET_EVENTS};
use lance_file::datatypes::populate_schema_dictionary;
use lance_file::v2::reader::FileReaderOptions;
use lance_io::object_store::{
    LanceNamespaceStorageOptionsProvider, ObjectStore, ObjectStoreParams, StorageOptions,
    DEFAULT_CLOUD_IO_PARALLELISM,
};
use lance_namespace::models::DescribeTableRequest;
use lance_namespace::LanceNamespace;
use lance_table::{
    format::Manifest,
    io::commit::{commit_handler_from_url, CommitHandler},
};
#[cfg(feature = "aws")]
use object_store::aws::AwsCredentialProvider;
use object_store::{path::Path, DynObjectStore};
use prost::Message;
use snafu::location;
use tracing::{info, instrument};
use url::Url;

/// builder for loading a [`Dataset`].
#[derive(Clone)]
pub struct DatasetBuilder {
    /// Cache size for index cache. If it is zero, index cache is disabled.
    index_cache_size_bytes: usize,
    /// Metadata cache size for the fragment metadata. If it is zero, metadata
    /// cache is disabled.
    metadata_cache_size_bytes: usize,
    /// Optional pre-loaded manifest to avoid loading it again.
    manifest: Option<Manifest>,
    session: Option<Arc<Session>>,
    commit_handler: Option<Arc<dyn CommitHandler>>,
    options: ObjectStoreParams,
    version: Option<Ref>,
    table_uri: String,
    file_reader_options: Option<FileReaderOptions>,
    /// Storage options that override user-provided options (e.g., from namespace)
    storage_options_override: Option<HashMap<String, String>>,
}

impl std::fmt::Debug for DatasetBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DatasetBuilder")
            .field("index_cache_size_bytes", &self.index_cache_size_bytes)
            .field("metadata_cache_size_bytes", &self.metadata_cache_size_bytes)
            .field("manifest", &self.manifest.is_some())
            .field("session", &self.session.is_some())
            .field("commit_handler", &self.commit_handler.is_some())
            .field("version", &self.version)
            .field("table_uri", &self.table_uri)
            .field("file_reader_options", &self.file_reader_options)
            .field(
                "storage_options_override",
                &self.storage_options_override.is_some(),
            )
            .finish()
    }
}

impl DatasetBuilder {
    pub fn from_uri<T: AsRef<str>>(table_uri: T) -> Self {
        Self {
            index_cache_size_bytes: DEFAULT_INDEX_CACHE_SIZE,
            metadata_cache_size_bytes: DEFAULT_METADATA_CACHE_SIZE,
            table_uri: table_uri.as_ref().to_string(),
            options: ObjectStoreParams::default(),
            commit_handler: None,
            session: None,
            version: None,
            manifest: None,
            file_reader_options: None,
            storage_options_override: None,
        }
    }

    /// Create a DatasetBuilder from a LanceNamespace
    ///
    /// This will automatically fetch the table location and storage options from the namespace
    /// via `describe_table()`.
    ///
    /// Storage options from the namespace will override any user-provided storage options
    /// set via `.with_storage_options()`. This ensures the namespace is always the source
    /// of truth for storage options.
    ///
    /// # Arguments
    /// * `namespace` - The namespace implementation to fetch table info from
    /// * `table_id` - The table identifier (e.g., vec!["my_table"])
    /// * `ignore_namespace_table_storage_options` - If true, storage options returned from
    ///   the namespace's `describe_table()` will be ignored (treated as None). Defaults to false.
    ///
    /// # Example
    /// ```ignore
    /// use lance_namespace_impls::ConnectBuilder;
    /// use lance::dataset::DatasetBuilder;
    ///
    /// // Connect to a REST namespace
    /// let namespace = ConnectBuilder::new("rest")
    ///     .property("uri", "http://localhost:8080")
    ///     .connect()
    ///     .await?;
    ///
    /// // Load a dataset using storage options from namespace
    /// let dataset = DatasetBuilder::from_namespace(
    ///     namespace.clone(),
    ///     vec!["my_table".to_string()],
    ///     false,
    /// )
    /// .await?
    /// .load()
    /// .await?;
    ///
    /// // Load a dataset ignoring namespace storage options
    /// let dataset = DatasetBuilder::from_namespace(
    ///     namespace,
    ///     vec!["my_table".to_string()],
    ///     true,
    /// )
    /// .await?
    /// .load()
    /// .await?;
    /// ```
    pub async fn from_namespace(
        namespace: Arc<dyn LanceNamespace>,
        table_id: Vec<String>,
        ignore_namespace_table_storage_options: bool,
    ) -> Result<Self> {
        let request = DescribeTableRequest {
            id: Some(table_id.clone()),
            version: None,
        };

        let response = namespace
            .describe_table(request)
            .await
            .map_err(|e| Error::Namespace {
                source: Box::new(e),
                location: location!(),
            })?;

        let table_uri = response.location.ok_or_else(|| Error::Namespace {
            source: Box::new(std::io::Error::other(
                "Table location not found in namespace response",
            )),
            location: location!(),
        })?;

        let mut builder = Self::from_uri(table_uri);

        let namespace_storage_options = if ignore_namespace_table_storage_options {
            None
        } else {
            response.storage_options
        };

        builder.storage_options_override = namespace_storage_options.clone();

        if namespace_storage_options.is_some() {
            builder.options.storage_options_provider = Some(Arc::new(
                LanceNamespaceStorageOptionsProvider::new(namespace, table_id),
            ));
        }

        Ok(builder)
    }
}

// Much of this builder is directly inspired from the to delta-rs table builder implementation
// https://github.com/delta-io/delta-rs/main/crates/deltalake-core/src/table/builder.rs
impl DatasetBuilder {
    /// Set the cache size for indices. Set to zero, to disable the cache.
    pub fn with_index_cache_size_bytes(mut self, cache_size: usize) -> Self {
        self.index_cache_size_bytes = cache_size;
        self
    }

    /// Set the cache size for indices. Set to zero, to disable the cache.
    #[deprecated(since = "0.30.0", note = "Use `with_index_cache_size_bytes` instead")]
    pub fn with_index_cache_size(mut self, cache_size: usize) -> Self {
        let assumed_entry_size = 20 * 1024 * 1024; // 20 MiB per entry
        self.index_cache_size_bytes = cache_size * assumed_entry_size;
        self
    }

    /// Size of the metadata cache in bytes. This cache stores metadata in memory
    /// for faster open table and scans. The default is 1 GiB.
    pub fn with_metadata_cache_size_bytes(mut self, cache_size: usize) -> Self {
        self.metadata_cache_size_bytes = cache_size;
        self
    }

    /// Set the cache size for the file metadata. Set to zero to disable this cache.
    #[deprecated(
        since = "0.30.0",
        note = "Use `with_metadata_cache_size_bytes` instead"
    )]
    pub fn with_metadata_cache_size(mut self, cache_size: usize) -> Self {
        let assumed_entry_size = 10 * 1024 * 1024; // 10MB per entry
        self.metadata_cache_size_bytes = cache_size * assumed_entry_size;
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

    /// Sets `version` for the builder using a version number
    pub fn with_version(mut self, version: u64) -> Self {
        self.version = Some(Ref::from(version));
        self
    }

    /// Sets `version` for the builder using a branch and optional version number
    /// If version_number is null, checkout the latest version
    pub fn with_branch(mut self, branch: &str, version_number: Option<u64>) -> Self {
        self.version = Some(Ref::from((branch, version_number)));
        self
    }

    /// Sets `version` for the builder using a tag
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.version = Some(Ref::from(tag));
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
    #[cfg(feature = "aws")]
    pub fn with_aws_credentials_provider(mut self, credentials: AwsCredentialProvider) -> Self {
        self.options.aws_credentials = Some(credentials);
        self
    }

    /// Directly set the object store to use.
    #[deprecated(note = "Implement an ObjectStoreProvider instead")]
    #[allow(deprecated)]
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

    /// Enable credential vending from a LanceNamespace
    ///
    /// Credentials will be automatically refreshed from the namespace
    /// before they expire. The namespace should return `expires_at_millis`
    /// in the storage_options from `describe_table()`.
    ///
    /// Use `with_s3_credentials_refresh_offset()` to configure how early
    /// credentials should be refreshed before they expire (default is 5 minutes).
    ///
    /// # Arguments
    /// * `provider` - The storage options provider to fetch credentials from
    ///
    /// # Example
    /// ```ignore
    /// use std::sync::Arc;
    /// use std::time::Duration;
    /// use lance_namespace_impls::ConnectBuilder;
    /// use lance_io::object_store::{StorageOptionsProvider, LanceNamespaceStorageOptionsProvider};
    ///
    /// // Connect to a REST namespace
    /// let namespace = ConnectBuilder::new("rest")
    ///     .property("uri", "http://localhost:8080")
    ///     .connect()
    ///     .await?;
    ///
    /// // Create a storage options provider from namespace
    /// let provider = Arc::new(LanceNamespaceStorageOptionsProvider::new(
    ///     namespace,
    ///     vec!["my_table".to_string()],
    /// ));
    ///
    /// // With default settings (5 minute refresh offset)
    /// let dataset = DatasetBuilder::from_uri("s3://bucket/table.lance")
    ///     .with_storage_options_provider(provider)
    ///     .load()
    ///     .await?;
    /// ```
    ///
    /// // With custom refresh offset (refresh 10 minutes before expiration)
    /// let dataset = DatasetBuilder::from_uri("s3://bucket/table.lance")
    ///     .with_storage_options_provider(provider.clone())
    ///     .with_s3_credentials_refresh_offset(Duration::from_secs(600))
    ///     .load()
    ///     .await?;
    pub fn with_storage_options_provider(
        mut self,
        provider: Arc<dyn lance_io::object_store::StorageOptionsProvider>,
    ) -> Self {
        self.options.storage_options_provider = Some(provider);
        self
    }

    /// Set options based on [ReadParams].
    pub fn with_read_params(mut self, read_params: ReadParams) -> Self {
        self = self
            .with_index_cache_size_bytes(read_params.index_cache_size_bytes)
            .with_metadata_cache_size_bytes(read_params.metadata_cache_size_bytes);

        if let Some(options) = read_params.store_options {
            self.options = options;
        }

        if let Some(session) = read_params.session {
            self.session = Some(session);
        }

        if let Some(commit_handler) = read_params.commit_handler {
            self.commit_handler = Some(commit_handler);
        }

        if let Some(file_reader_options) = read_params.file_reader_options {
            self.file_reader_options = Some(file_reader_options);
        }

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

    /// Build a lance object store for the given config
    pub async fn build_object_store(
        self,
    ) -> Result<(Arc<ObjectStore>, Path, Arc<dyn CommitHandler>)> {
        let commit_handler = match self.commit_handler {
            Some(commit_handler) => Ok(commit_handler),
            None => commit_handler_from_url(&self.table_uri, &Some(self.options.clone())).await,
        }?;

        let storage_options = self
            .options
            .storage_options
            .clone()
            .map(StorageOptions::new)
            .unwrap_or_default();
        let download_retry_count = storage_options.download_retry_count();

        let store_registry = self
            .session
            .as_ref()
            .map(|s| s.store_registry())
            .unwrap_or_default();

        #[allow(deprecated)]
        match &self.options.object_store {
            Some(store) => Ok((
                Arc::new(ObjectStore::new(
                    store.0.clone(),
                    store.1.clone(),
                    self.options.block_size,
                    self.options.object_store_wrapper,
                    self.options.use_constant_size_upload_parts,
                    store.1.scheme() != "file",
                    // If user supplied an object store then we just assume it's probably
                    // cloud-like
                    DEFAULT_CLOUD_IO_PARALLELISM,
                    download_retry_count,
                    None, // No storage_options available here
                )),
                Path::from(store.1.path()),
                commit_handler,
            )),
            None => {
                let (store, path) = ObjectStore::from_uri_and_params(
                    store_registry,
                    &self.table_uri,
                    &self.options,
                )
                .await?;
                Ok((store, path, commit_handler))
            }
        }
    }

    #[instrument(skip_all)]
    pub async fn load(self) -> Result<Dataset> {
        let uri = self.table_uri.clone();
        let target_ref = self.version.clone();
        match self.load_impl().boxed().await {
            Ok(dataset) => {
                info!(target: TRACE_DATASET_EVENTS, event=DATASET_LOADING_EVENT, uri=uri, target_ref = ?target_ref, version=dataset.manifest.version, status="success");
                Ok(dataset)
            }
            Err(e) => {
                info!(target: TRACE_DATASET_EVENTS, event=DATASET_LOADING_EVENT, uri=uri, target_ref = ?target_ref, status="error");
                Err(e)
            }
        }
    }

    async fn load_impl(mut self) -> Result<Dataset> {
        // Apply storage_options_override last to ensure namespace options take precedence
        if let Some(override_opts) = self.storage_options_override.take() {
            let mut merged_opts = self.options.storage_options.clone().unwrap_or_default();
            // Override with namespace storage options - they take precedence
            merged_opts.extend(override_opts);
            self.options.storage_options = Some(merged_opts);
        }

        let session = match self.session.as_ref() {
            Some(session) => session.clone(),
            None => Arc::new(Session::new(
                self.index_cache_size_bytes,
                self.metadata_cache_size_bytes,
                Default::default(),
            )),
        };

        let target_ref = self.version.clone();
        let table_uri = self.table_uri.clone();

        // How do we detect which version scheme is in use?
        let manifest = self.manifest.take();

        let file_reader_options = self.file_reader_options.clone();
        let store_params = self.options.clone();
        let (object_store, base_path, commit_handler) = self.build_object_store().await?;

        // Two cases that need to check out after loading the manifest:
        // 1. If the target is configured as a branch, we need to check the branch field in the manifest
        // and reload the right branch in case the uri is not the right one.
        // 2. If the target is configured as a tag, and we don't find the tag under the table_uri,
        // we need to get the root_location after loading the manifest and get the right version.
        // In practice, we should try best to use the right uri and avoid double loading.
        let mut need_delay_checkout = false;
        let (mut branch, mut version_number) = match target_ref.clone() {
            Some(Ref::Version(branch, version_number)) => {
                if branch.is_some() {
                    need_delay_checkout = true;
                }
                (branch, version_number)
            }
            // Here we assume the uri and path is the root.
            // If tag not found, we need to delay checkout after loading by uri
            Some(Ref::Tag(tag_name)) => {
                let refs = Refs::new(
                    object_store.clone(),
                    commit_handler.clone(),
                    BranchLocation {
                        path: base_path.clone(),
                        uri: table_uri.clone(),
                        branch: None,
                    },
                );
                let tag_content = refs.tags().get(&tag_name).await;
                if let Ok(tag_content) = tag_content {
                    (tag_content.branch.clone(), Some(tag_content.version))
                } else {
                    need_delay_checkout = true;
                    (None, None)
                }
            }
            None => (None, None),
        };

        let dataset = Self::load_by_uri(
            session,
            manifest,
            file_reader_options,
            table_uri,
            version_number,
            object_store,
            base_path,
            commit_handler,
            Some(store_params),
        )
        .await?;

        if need_delay_checkout {
            if let Some(Ref::Tag(tag_name)) = target_ref {
                let tag_content = dataset.tags().get(tag_name.as_str()).await?;
                branch = tag_content.branch.clone();
                version_number = Some(tag_content.version);
            }

            if branch.as_deref() != dataset.manifest.branch.as_deref() {
                return dataset.checkout_version((branch, version_number)).await;
            }
        }
        if let Some(version_number) = version_number {
            if version_number != dataset.manifest.version {
                return Err(Error::VersionNotFound {
                    message: format!("version {} not found", version_number),
                });
            }
        }
        Ok(dataset)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn load_by_uri(
        session: Arc<Session>,
        manifest: Option<Manifest>,
        file_reader_options: Option<FileReaderOptions>,
        table_uri: String,
        version_number: Option<u64>,
        object_store: Arc<ObjectStore>,
        base_path: Path,
        commit_handler: Arc<dyn CommitHandler>,
        store_params: Option<ObjectStoreParams>,
    ) -> Result<Dataset> {
        let (manifest, location) = if let Some(mut manifest) = manifest {
            let location = commit_handler
                .resolve_version_location(&base_path, manifest.version, &object_store.inner)
                .await?;
            if manifest.schema.has_dictionary_types() && manifest.should_use_legacy_format() {
                let reader = object_store.open(&location.path).await?;
                populate_schema_dictionary(&mut manifest.schema, reader.as_ref()).await?;
            }
            (manifest, location)
        } else {
            let manifest_location = match version_number {
                Some(version) => {
                    let target_manifest_result = commit_handler
                        .resolve_version_location(&base_path, version, &object_store.inner)
                        .await;
                    // This may fail due to the uri is not the right branch
                    // In this case we should try to load the latest version and checkout the right branch and version_number
                    match target_manifest_result {
                        Ok(manifest_location) => manifest_location,
                        Err(e) => {
                            if let Error::VersionNotFound { message: _ } = e {
                                // If the version is not found, we need to try to load the latest version.
                                commit_handler
                                    .resolve_latest_location(&base_path, &object_store)
                                    .await?
                            } else {
                                return Err(e);
                            }
                        }
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
            let manifest = Dataset::load_manifest(
                &object_store,
                &manifest_location,
                &table_uri,
                session.as_ref(),
            )
            .await?;
            (manifest, manifest_location)
        };

        Dataset::checkout_manifest(
            object_store,
            base_path,
            table_uri,
            Arc::new(manifest),
            location,
            session,
            commit_handler,
            file_reader_options,
            store_params,
        )
    }
}
