use std::{collections::HashMap, sync::Arc, time::Duration};

use lance_core::io::{
    commit::CommitHandler,
    object_store::{ObjectStore, ObjectStoreParams},
};
use object_store::{aws::AwsCredentialProvider, DynObjectStore};
use snafu::{location, Location};
use url::Url;

use super::{DEFAULT_INDEX_CACHE_SIZE, DEFAULT_METADATA_CACHE_SIZE};
use crate::{
    error::{Error, Result},
    session::Session,
    Dataset,
};
/// builder for loading a [`Dataset`].
pub struct DatasetBuilder {
    /// Cache size for index cache. If it is zero, index cache is disabled.
    index_cache_size: usize,

    /// Metadata cache size for the fragment metadata. If it is zero, metadata
    /// cache is disabled.
    metadata_cache_size: usize,
    options: ObjectStoreParams,
    storage_options: Option<HashMap<String, String>>,
    version: Option<u64>,
    table_uri: String,
}

impl DatasetBuilder {
    pub fn from_uri<T: AsRef<str>>(table_uri: T) -> Self {
        Self {
            index_cache_size: DEFAULT_INDEX_CACHE_SIZE,
            metadata_cache_size: DEFAULT_METADATA_CACHE_SIZE,
            table_uri: table_uri.as_ref().to_string(),
            options: ObjectStoreParams::default(),
            storage_options: None,
            version: None,
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
        self.options.commit_handler = Some(commit_handler);
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
    pub fn with_object_store(mut self, object_store: Arc<DynObjectStore>, location: Url) -> Self {
        self.options.object_store = Some((object_store, location));
        self
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
        self.storage_options = Some(storage_options);
        self
    }

    /// Set a single option used to initialize storage backend
    pub fn with_storage_option(mut self, key: impl AsRef<str>, value: impl AsRef<str>) -> Self {
        let mut storage_options = self.storage_options.unwrap_or_default();
        storage_options.insert(key.as_ref().to_string(), value.as_ref().to_string());
        self.storage_options = Some(storage_options);
        self
    }

    /// Build a lance object store for the given config
    pub async fn build_object_store(self) -> Result<ObjectStore> {
        match &self.options.object_store {
            Some(store) => Ok(ObjectStore::new(
                store.0.clone(),
                store.1.clone(),
                self.options.block_size,
                self.options.commit_handler,
                self.options.object_store_wrapper,
            )),
            None => {
                ObjectStore::try_new(self.table_uri, self.storage_options.unwrap_or_default()).await
            }
        }
    }

    pub async fn load(self) -> Result<Dataset> {
        let session = Arc::new(Session::new(
            self.index_cache_size,
            self.metadata_cache_size,
        ));

        let version = self.version;

        let object_store = self.build_object_store().await?;
        let base_path = object_store.base_path();
        let manifest = match version {
            Some(version) => {
                object_store
                    .commit_handler
                    .resolve_version(base_path, version, &object_store.inner)
                    .await?
            }
            None => object_store
                .commit_handler
                .resolve_latest_version(base_path, &object_store.inner)
                .await
                .map_err(|e| Error::DatasetNotFound {
                    path: base_path.to_string(),
                    source: Box::new(e),
                    location: location!(),
                })?,
        };

        Dataset::checkout_manifest(
            Arc::new(object_store.clone()),
            base_path.clone(),
            &manifest,
            session,
        )
        .await
    }
}
