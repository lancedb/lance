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

//! Extend [object_store::ObjectStore] functionalities

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path as StdPath, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use aws_config::default_provider::credentials::DefaultCredentialsChain;
use aws_credential_types::provider::error::CredentialsError;
use aws_credential_types::provider::ProvideCredentials;
use chrono::{DateTime, Utc};
use futures::{future, stream::BoxStream, StreamExt, TryStreamExt};
use object_store::aws::{
    AmazonS3ConfigKey, AwsCredential as ObjectStoreAwsCredential, AwsCredentialProvider,
};
use object_store::azure::AzureConfigKey;
use object_store::gcp::GoogleConfigKey;
use object_store::{
    aws::AmazonS3Builder, local::LocalFileSystem, memory::InMemory, CredentialProvider,
    Error as ObjectStoreError, Result as ObjectStoreResult,
};
use object_store::{parse_url_opts, DynObjectStore};
use object_store::{path::Path, ObjectMeta, ObjectStore as OSObjectStore};
use shellexpand::tilde;
use snafu::{location, Location};
use tokio::{io::AsyncWriteExt, sync::RwLock};
use url::Url;

use super::local::LocalObjectReader;
#[cfg(feature = "dynamodb")]
use crate::io::commit::external_manifest::{ExternalManifestCommitHandler, ExternalManifestStore};

mod tracing;
use self::tracing::ObjectStoreTracingExt;
use crate::{
    error::{Error, Result},
    io::{
        commit::{CommitHandler, CommitLock, RenameCommitHandler, UnsafeCommitHandler},
        CloudObjectReader, ObjectWriter, Reader,
    },
};

#[async_trait]
pub trait ObjectStoreExt {
    /// Returns true if the file exists.
    async fn exists(&self, path: &Path) -> Result<bool>;

    /// Read all files (start from base directory) recursively
    ///
    /// unmodified_since can be specified to only return files that have not been modified since the given time.
    async fn read_dir_all(
        &self,
        dir_path: impl Into<&Path> + Send,
        unmodified_since: Option<DateTime<Utc>>,
    ) -> Result<BoxStream<Result<ObjectMeta>>>;
}

#[async_trait]
impl<O: OSObjectStore + ?Sized> ObjectStoreExt for O {
    async fn read_dir_all(
        &self,
        dir_path: impl Into<&Path> + Send,
        unmodified_since: Option<DateTime<Utc>>,
    ) -> Result<BoxStream<Result<ObjectMeta>>> {
        let mut output = self.list(Some(dir_path.into())).await?;
        if let Some(unmodified_since_val) = unmodified_since {
            output = output
                .try_filter(move |file| future::ready(file.last_modified < unmodified_since_val))
                .boxed();
        }
        Ok(output.map_err(|e| e.into()).boxed())
    }

    async fn exists(&self, path: &Path) -> Result<bool> {
        match self.head(path).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { path: _, source: _ }) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }
}

/// Wraps [ObjectStore](object_store::ObjectStore)
#[derive(Debug, Clone)]
pub struct ObjectStore {
    // Inner object store
    pub inner: Arc<dyn OSObjectStore>,
    scheme: String,
    base_path: Path,
    block_size: usize,
    pub commit_handler: Arc<dyn CommitHandler>,
}

impl std::fmt::Display for ObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ObjectStore({})", self.scheme)
    }
}

const AWS_CREDS_CACHE_KEY: &str = "aws_credentials";

/// Adapt an AWS SDK cred into object_store credentials
#[derive(Debug)]
struct AwsCredentialAdapter {
    pub inner: Arc<dyn ProvideCredentials>,

    // RefCell can't be shared accross threads, so we use HashMap
    cache: Arc<RwLock<HashMap<String, Arc<aws_credential_types::Credentials>>>>,

    // The amount of time before expiry to refresh credentials
    credentials_refresh_offset: Duration,
}

impl AwsCredentialAdapter {
    fn new(provider: Arc<dyn ProvideCredentials>, credentials_refresh_offset: Duration) -> Self {
        Self {
            inner: provider,
            cache: Arc::new(RwLock::new(HashMap::new())),
            credentials_refresh_offset,
        }
    }
}

/// Adapt an object_store credentials into AWS SDK creds
#[derive(Debug)]
struct OSObjectStoreToAwsCredAdaptor(AwsCredentialProvider);

impl ProvideCredentials for OSObjectStoreToAwsCredAdaptor {
    fn provide_credentials<'a>(
        &'a self,
    ) -> aws_credential_types::provider::future::ProvideCredentials<'a>
    where
        Self: 'a,
    {
        aws_credential_types::provider::future::ProvideCredentials::new(async {
            let creds = self
                .0
                .get_credential()
                .await
                .map_err(|e| CredentialsError::provider_error(Box::new(e)))?;
            Ok(aws_credential_types::Credentials::new(
                &creds.key_id,
                &creds.secret_key,
                creds.token.clone(),
                Some(
                    SystemTime::now()
                        .checked_add(Duration::from_secs(
                            60 * 10, //  10 min
                        ))
                        .expect("overflow"),
                ),
                "",
            ))
        })
    }
}

#[async_trait]
impl CredentialProvider for AwsCredentialAdapter {
    type Credential = ObjectStoreAwsCredential;

    async fn get_credential(&self) -> ObjectStoreResult<Arc<Self::Credential>> {
        let cached_creds = {
            let cache_value = self.cache.read().await.get(AWS_CREDS_CACHE_KEY).cloned();
            let expired = cache_value
                .clone()
                .map(|cred| {
                    cred.expiry()
                        .map(|exp| {
                            exp.checked_sub(self.credentials_refresh_offset)
                                .expect("this time should always be valid")
                                < SystemTime::now()
                        })
                        // no expiry is never expire
                        .unwrap_or(false)
                })
                .unwrap_or(true); // no cred is the same as expired;
            if expired {
                None
            } else {
                cache_value.clone()
            }
        };

        if let Some(creds) = cached_creds {
            Ok(Arc::new(Self::Credential {
                key_id: creds.access_key_id().to_string(),
                secret_key: creds.secret_access_key().to_string(),
                token: creds.session_token().map(|s| s.to_string()),
            }))
        } else {
            let refreshed_creds = Arc::new(self.inner.provide_credentials().await.unwrap());

            self.cache
                .write()
                .await
                .insert(AWS_CREDS_CACHE_KEY.to_string(), refreshed_creds.clone());

            Ok(Arc::new(Self::Credential {
                key_id: refreshed_creds.access_key_id().to_string(),
                secret_key: refreshed_creds.secret_access_key().to_string(),
                token: refreshed_creds.session_token().map(|s| s.to_string()),
            }))
        }
    }
}

/// Build AWS credentials
/// `credentials_refresh_offset` is the amount of time before expiry to refresh credentials.
async fn build_aws_credential(
    credentials_refresh_offset: Duration,
    credentials: Option<AwsCredentialProvider>,
    region: Option<String>,
) -> Result<(AwsCredentialProvider, String)> {
    use aws_config::meta::region::RegionProviderChain;
    const DEFAULT_REGION: &str = "us-west-2";
    let region_provider = RegionProviderChain::default_provider().or_else(DEFAULT_REGION);
    let region = region.unwrap_or(
        region_provider
            .region()
            .await
            .map(|r| r.as_ref().to_string())
            .unwrap_or(DEFAULT_REGION.to_string()),
    );

    let creds = match credentials {
        Some(creds) => creds,
        None => {
            let credentials_provider = DefaultCredentialsChain::builder()
                .region(region_provider.region().await)
                .build()
                .await;

            Arc::new(AwsCredentialAdapter::new(
                Arc::new(credentials_provider),
                credentials_refresh_offset,
            ))
        }
    };

    Ok((creds, region))
}

#[cfg(feature = "dynamodb")]
async fn build_dynamodb_external_store(
    table_name: &str,
    creds: AwsCredentialProvider,
    region: &str,
    app_name: &str,
) -> Result<Arc<dyn ExternalManifestStore>> {
    use std::env;

    use super::commit::dynamodb::DynamoDBExternalManifestStore;
    use aws_sdk_dynamodb::{config::Region, Client};

    let dynamodb_config = aws_sdk_dynamodb::config::Builder::new()
        .region(Some(Region::new(region.to_string())))
        .credentials_provider(OSObjectStoreToAwsCredAdaptor(creds));
    let dynamodb_config = match env::var("DYNAMODB_ENDPOINT") {
        Ok(endpoint) => dynamodb_config.endpoint_url(endpoint),
        _ => dynamodb_config,
    };

    let client = Client::from_conf(dynamodb_config.build());

    DynamoDBExternalManifestStore::new_external_store(client.into(), table_name, app_name).await
}

pub trait WrappingObjectStore: std::fmt::Debug + Send + Sync {
    fn wrap(&self, original: Arc<dyn OSObjectStore>) -> Arc<dyn OSObjectStore>;
}

/// Parameters to create an [ObjectStore]
///
#[derive(Debug, Clone)]
pub struct ObjectStoreParams {
    pub block_size: Option<usize>,
    pub object_store: Option<(Arc<DynObjectStore>, Url)>,
    pub commit_handler: Option<Arc<dyn CommitHandler>>,
    pub s3_credentials_refresh_offset: Duration,
    pub aws_credentials: Option<AwsCredentialProvider>,
    pub object_store_wrapper: Option<Arc<dyn WrappingObjectStore>>,
}

impl Default for ObjectStoreParams {
    fn default() -> Self {
        Self {
            object_store: None,
            commit_handler: None,
            block_size: None,
            s3_credentials_refresh_offset: Duration::from_secs(60),
            aws_credentials: None,
            object_store_wrapper: None,
        }
    }
}

impl ObjectStoreParams {
    /// Set a commit lock for the object store.
    pub fn set_commit_lock<T: CommitLock + Send + Sync + 'static>(&mut self, lock: Arc<T>) {
        self.commit_handler = Some(Arc::new(lock));
    }
}

static DDB_URL_QUERY_KEY: &str = "ddbTableName";

impl ObjectStore {
    /// Parse from a string URI.
    ///
    /// Returns the ObjectStore instance and the absolute path to the object.
    pub async fn from_uri(uri: &str) -> Result<(Self, Path)> {
        Self::from_uri_and_params(uri, &ObjectStoreParams::default()).await
    }

    /// Parse from a string URI.
    ///
    /// Returns the ObjectStore instance and the absolute path to the object.
    pub async fn from_uri_and_params(
        uri: &str,
        params: &ObjectStoreParams,
    ) -> Result<(Self, Path)> {
        let (object_store, base_path) = match Url::parse(uri) {
            Ok(url) if url.scheme().len() == 1 && cfg!(windows) => {
                // On Windows, the drive is parsed as a scheme
                Self::from_path(uri, params)
            }
            Ok(url) => {
                let store = Self::new_from_url(url.clone(), params.clone()).await?;
                let path = Path::from(url.path());
                Ok((store, path))
            }
            Err(_) => Self::from_path(uri, params),
        }?;

        Ok((
            Self {
                inner: params
                    .object_store_wrapper
                    .as_ref()
                    .map(|w| w.wrap(object_store.inner.clone()))
                    .unwrap_or(object_store.inner),
                ..object_store
            },
            base_path,
        ))
    }

    pub fn with_params(&self, params: &ObjectStoreParams) -> Self {
        Self {
            inner: params
                .object_store_wrapper
                .as_ref()
                .map(|w| w.wrap(self.inner.clone()))
                .unwrap_or_else(|| self.inner.clone()),
            commit_handler: params
                .commit_handler
                .clone()
                .unwrap_or(self.commit_handler.clone()),
            ..self.clone()
        }
    }

    pub fn from_path(str_path: &str, params: &ObjectStoreParams) -> Result<(Self, Path)> {
        let expanded = tilde(str_path).to_string();
        let expanded_path = StdPath::new(&expanded);

        if !expanded_path.try_exists()? {
            std::fs::create_dir_all(expanded_path)?;
        }

        let expanded_path = expanded_path.canonicalize()?;

        Ok((
            Self {
                inner: Arc::new(LocalFileSystem::new()).traced(),
                scheme: String::from("file"),
                base_path: Path::from_absolute_path(&expanded_path)?,
                block_size: 4 * 1024, // 4KB block size
                commit_handler: params
                    .commit_handler
                    .clone()
                    .unwrap_or_else(|| Arc::new(RenameCommitHandler)),
            },
            Path::from_filesystem_path(&expanded_path)?,
        ))
    }

    fn new_from_path(
        str_path: &str,
        commit_handler: Option<Arc<dyn CommitHandler>>,
    ) -> Result<(Self, Path)> {
        let expanded = tilde(str_path).to_string();
        let expanded_path = StdPath::new(&expanded);

        if !expanded_path.try_exists()? {
            std::fs::create_dir_all(expanded_path)?;
        }

        let expanded_path = expanded_path.canonicalize()?;

        Ok((
            Self {
                inner: Arc::new(LocalFileSystem::new()).traced(),
                scheme: String::from("file"),
                base_path: Path::from_absolute_path(&expanded_path)?,
                block_size: 4 * 1024, // 4KB block size
                commit_handler: commit_handler.unwrap_or_else(|| Arc::new(RenameCommitHandler)),
            },
            Path::from_filesystem_path(&expanded_path)?,
        ))
    }

    async fn new_from_url(url: Url, params: ObjectStoreParams) -> Result<Self> {
        let mut storage_options = StorageOptions::default();
        configure_store(url.as_str(), &mut storage_options, params).await
    }
    /// Local object store.
    pub fn local() -> Self {
        Self {
            inner: Arc::new(LocalFileSystem::new()).traced(),
            scheme: String::from("file"),
            base_path: Path::from("/"),
            block_size: 4 * 1024, // 4KB block size
            commit_handler: Arc::new(RenameCommitHandler),
        }
    }

    /// Create a in-memory object store directly for testing.
    pub fn memory() -> Self {
        Self {
            inner: Arc::new(InMemory::new()).traced(),
            scheme: String::from("memory"),
            base_path: Path::from("/"),
            block_size: 64 * 1024,
            commit_handler: Arc::new(RenameCommitHandler),
        }
    }

    /// Returns true if the object store pointed to a local file system.
    pub fn is_local(&self) -> bool {
        self.scheme == "file"
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn set_block_size(&mut self, new_size: usize) {
        self.block_size = new_size;
    }

    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Open a file for path.
    ///
    /// Parameters
    /// - ``path``: Absolute path to the file.
    pub async fn open(&self, path: &Path) -> Result<Box<dyn Reader>> {
        match self.scheme.as_str() {
            "file" => LocalObjectReader::open(path, self.block_size),
            _ => Ok(Box::new(CloudObjectReader::new(
                self.inner.clone(),
                path.clone(),
                self.block_size,
            )?)),
        }
    }

    /// Create an [ObjectWriter] from local [std::path::Path]
    pub async fn create_local_writer(path: &std::path::Path) -> Result<ObjectWriter> {
        let object_store = Self::local();
        let os_path = Path::from(path.to_str().unwrap());
        object_store.create(&os_path).await
    }

    /// Open an [Reader] from local [std::path::Path]
    pub async fn open_local(path: &std::path::Path) -> Result<Box<dyn Reader>> {
        let object_store = Self::local();
        let os_path = Path::from(path.to_str().unwrap());
        object_store.open(&os_path).await
    }

    /// Create a new file.
    pub async fn create(&self, path: &Path) -> Result<ObjectWriter> {
        ObjectWriter::new(self.inner.as_ref(), path).await
    }

    /// A helper function to create a file and write content to it.
    ///
    pub async fn put(&self, path: &Path, content: &[u8]) -> Result<()> {
        let mut writer = self.create(path).await?;
        writer.write_all(content).await?;
        writer.shutdown().await
    }

    pub async fn delete(&self, path: &Path) -> Result<()> {
        self.inner.delete(path).await?;
        Ok(())
    }

    pub async fn copy(&self, from: &Path, to: &Path) -> Result<()> {
        Ok(self.inner.copy(from, to).await?)
    }

    /// Read a directory (start from base directory) and returns all sub-paths in the directory.
    pub async fn read_dir(&self, dir_path: impl Into<Path>) -> Result<Vec<String>> {
        let path = dir_path.into();
        let path = Path::parse(&path)?;
        let output = self.inner.list_with_delimiter(Some(&path)).await?;
        Ok(output
            .common_prefixes
            .iter()
            .chain(output.objects.iter().map(|o| &o.location))
            .map(|s| s.filename().unwrap().to_string())
            .collect())
    }

    /// Read all files (start from base directory) recursively
    ///
    /// unmodified_since can be specified to only return files that have not been modified since the given time.
    pub async fn read_dir_all(
        &self,
        dir_path: impl Into<&Path> + Send,
        unmodified_since: Option<DateTime<Utc>>,
    ) -> Result<BoxStream<Result<ObjectMeta>>> {
        self.inner.read_dir_all(dir_path, unmodified_since).await
    }

    /// Remove a directory recursively.
    pub async fn remove_dir_all(&self, dir_path: impl Into<Path>) -> Result<()> {
        let path = dir_path.into();
        let path = Path::parse(&path)?;

        if self.is_local() {
            // Local file system needs to delete directories as well.
            return super::local::remove_dir_all(&path);
        }
        let sub_entries = self
            .inner
            .list(Some(&path))
            .await?
            .map(|m| m.map(|meta| meta.location))
            .boxed();
        self.inner
            .delete_stream(sub_entries)
            .try_collect::<Vec<_>>()
            .await?;
        Ok(())
    }

    pub fn remove_stream<'a>(
        &'a self,
        locations: BoxStream<'a, Result<Path>>,
    ) -> BoxStream<Result<Path>> {
        self.inner
            .delete_stream(locations.err_into::<ObjectStoreError>().boxed())
            .err_into::<Error>()
            .boxed()
    }

    /// Check a file exists.
    pub async fn exists(&self, path: &Path) -> Result<bool> {
        match self.inner.head(path).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { path: _, source: _ }) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    /// Get file size.
    pub async fn size(&self, path: &Path) -> Result<usize> {
        Ok(self.inner.head(path).await?.size)
    }
}
#[derive(Clone, Debug, Default)]
pub struct StorageOptions(pub HashMap<String, String>);

impl StorageOptions {
    /// Create a new instance of [`StorageOptions`]
    pub fn new(options: HashMap<String, String>) -> Self {
        let mut options = options;
        if let Ok(value) = std::env::var("AZURE_STORAGE_ALLOW_HTTP") {
            options.insert("allow_http".into(), value);
        }
        if let Ok(value) = std::env::var("AZURE_STORAGE_USE_HTTP") {
            options.insert("allow_http".into(), value);
        }
        if let Ok(value) = std::env::var("AWS_ALLOW_HTTP") {
            options.insert("allow_http".into(), value);
        }
        Self(options)
    }

    /// Add values from the environment to storage options
    pub fn with_env_azure(&mut self) {
        for (os_key, os_value) in std::env::vars_os() {
            if let (Some(key), Some(value)) = (os_key.to_str(), os_value.to_str()) {
                if let Ok(config_key) = AzureConfigKey::from_str(&key.to_ascii_lowercase()) {
                    if !self.0.contains_key(config_key.as_ref()) {
                        self.0
                            .insert(config_key.as_ref().to_string(), value.to_string());
                    }
                }
            }
        }
    }

    /// Add values from the environment to storage options
    pub fn with_env_gcs(&mut self) {
        for (os_key, os_value) in std::env::vars_os() {
            if let (Some(key), Some(value)) = (os_key.to_str(), os_value.to_str()) {
                if let Ok(config_key) = GoogleConfigKey::from_str(&key.to_ascii_lowercase()) {
                    if !self.0.contains_key(config_key.as_ref()) {
                        self.0
                            .insert(config_key.as_ref().to_string(), value.to_string());
                    }
                }
            }
        }
    }

    /// Add values from the environment to storage options
    pub fn with_env_s3(&mut self) {
        for (os_key, os_value) in std::env::vars_os() {
            if let (Some(key), Some(value)) = (os_key.to_str(), os_value.to_str()) {
                if let Ok(config_key) = AmazonS3ConfigKey::from_str(&key.to_ascii_lowercase()) {
                    if !self.0.contains_key(config_key.as_ref()) {
                        self.0
                            .insert(config_key.as_ref().to_string(), value.to_string());
                    }
                }
            }
        }
    }

    /// Denotes if unsecure connections via http are allowed
    pub fn allow_http(&self) -> bool {
        self.0.iter().any(|(key, value)| {
            key.to_ascii_lowercase().contains("allow_http") & str_is_truthy(value)
        })
    }

    /// Subset of options relevant for azure storage
    pub fn as_azure_options(&self) -> HashMap<AzureConfigKey, String> {
        self.0
            .iter()
            .filter_map(|(key, value)| {
                let az_key = AzureConfigKey::from_str(&key.to_ascii_lowercase()).ok()?;
                Some((az_key, value.clone()))
            })
            .collect()
    }

    /// Subset of options relevant for s3 storage
    pub fn as_s3_options(&self) -> HashMap<AmazonS3ConfigKey, String> {
        self.0
            .iter()
            .filter_map(|(key, value)| {
                let s3_key = AmazonS3ConfigKey::from_str(&key.to_ascii_lowercase()).ok()?;
                Some((s3_key, value.clone()))
            })
            .collect()
    }

    /// Subset of options relevant for gcs storage
    pub fn as_gcs_options(&self) -> HashMap<GoogleConfigKey, String> {
        self.0
            .iter()
            .filter_map(|(key, value)| {
                let gcs_key = GoogleConfigKey::from_str(&key.to_ascii_lowercase()).ok()?;
                Some((gcs_key, value.clone()))
            })
            .collect()
    }
}

impl From<HashMap<String, String>> for StorageOptions {
    fn from(value: HashMap<String, String>) -> Self {
        Self::new(value)
    }
}

async fn configure_store(
    url: &str,
    storage_options: &mut StorageOptions,
    options: ObjectStoreParams,
) -> Result<ObjectStore> {
    let mut url = ensure_table_uri(url)?;
    // Block size: On local file systems, we use 4KB block size. On cloud
    // object stores, we use 64KB block size. This is generally the largest
    // block size where we don't see a latency penalty.
    match url.scheme() {
        "s3" | "s3+ddb" => {
            storage_options.with_env_s3();

            if url.scheme() == "s3+ddb" && options.commit_handler.is_some() {
                return Err(Error::InvalidInput {
                    source: "`s3+ddb://` scheme and custom commit handler are mutually exclusive"
                        .into(),
                    location: location!(),
                });
            }

            let ddb_table_name = if url.scheme() == "s3" {
                None
            } else {
                if url.query_pairs().count() != 1 {
                    return Err(Error::InvalidInput {
                        source: "`s3+ddb://` scheme and expects exactly one query `ddbTableName`"
                            .into(),
                        location: location!(),
                    });
                }
                match url.query_pairs().next() {
                    Some((Cow::Borrowed(key), Cow::Borrowed(table_name)))
                        if key == DDB_URL_QUERY_KEY =>
                    {
                        if table_name.is_empty() {
                            return Err(Error::InvalidInput {
                                source: "`s3+ddb://` scheme requires non empty dynamodb table name"
                                    .into(),
                                location: location!(),
                            });
                        }
                        Some(table_name)
                    }
                    _ => {
                        return Err(Error::InvalidInput {
                            source:
                                "`s3+ddb://` scheme and expects exactly one query `ddbTableName`"
                                    .into(),
                            location: location!(),
                        });
                    }
                }
            };
            let storage_options = storage_options.as_s3_options();
            let region = storage_options
                .get(&AmazonS3ConfigKey::Region)
                .map(|s| s.to_string());

            let (aws_creds, region) = build_aws_credential(
                options.s3_credentials_refresh_offset,
                options.aws_credentials.clone(),
                region,
            )
            .await?;

            let commit_handler = match ddb_table_name {
                #[cfg(feature = "dynamodb")]
                Some(table_name) => Arc::new(ExternalManifestCommitHandler {
                    external_manifest_store: build_dynamodb_external_store(
                        table_name,
                        aws_creds.clone(),
                        &region,
                        "lancedb",
                    )
                    .await?,
                }),
                #[cfg(not(feature = "dynamodb"))]
                Some(_) => {
                    return Err(Error::InvalidInput {
                        source: "`s3+ddb://` scheme requires `dynamodb` feature to be enabled"
                            .into(),
                        location: location!(),
                    });
                }
                None => options
                    .commit_handler
                    .clone()
                    .unwrap_or_else(|| Arc::new(UnsafeCommitHandler)),
            };

            // before creating the OSObjectStore we need to rewrite the url to drop ddb related parts
            url.set_scheme("s3").map_err(|()| Error::Internal {
                message: "could not set scheme".into(),
                location: location!(),
            })?;

            url.set_query(None);

            // we can't use parse_url_opts here because we need to manually set the credentials provider
            let mut builder = AmazonS3Builder::new();
            for (key, value) in storage_options {
                builder = builder.with_config(key, value);
            }
            builder = builder
                .with_url(url.as_ref())
                .with_credentials(aws_creds)
                .with_region(region)
                .with_allow_http(true);
            let store = builder.build()?;

            Ok(ObjectStore {
                inner: Arc::new(store),
                scheme: String::from(url.scheme()),
                base_path: Path::from(url.path()),
                block_size: 64 * 1024,
                commit_handler,
            })
        }

        "gs" => {
            storage_options.with_env_gcs();
            let (store, _) = parse_url_opts(&url, storage_options.as_gcs_options())?;
            let store = Arc::new(store);
            Ok(ObjectStore {
                inner: store,
                scheme: String::from("gs"),
                base_path: Path::from(url.path()),
                block_size: 64 * 1024,
                commit_handler: options
                    .commit_handler
                    .clone()
                    .unwrap_or_else(|| Arc::new(RenameCommitHandler)),
            })
        }
        "az" => {
            storage_options.with_env_azure();

            let (store, _) = parse_url_opts(&url, storage_options.as_azure_options())?;
            let store = Arc::new(store);

            Ok(ObjectStore {
                inner: store,
                scheme: String::from("az"),
                base_path: Path::from(url.path()),
                block_size: 64 * 1024,
                commit_handler: options
                    .commit_handler
                    .clone()
                    .unwrap_or_else(|| Arc::new(RenameCommitHandler)),
            })
        }
        "file" => Ok(ObjectStore::new_from_path(url.path(), options.commit_handler)?.0),
        "memory" => Ok(ObjectStore {
            inner: Arc::new(InMemory::new()).traced(),
            scheme: String::from("memory"),
            base_path: Path::from(url.path()),
            block_size: 64 * 1024,
            commit_handler: options
                .commit_handler
                .clone()
                .unwrap_or_else(|| Arc::new(RenameCommitHandler)),
        }),
        s => Err(Error::IO {
            message: format!("Unsupported URI scheme: {}", s),
            location: location!(),
        }),
    }
}

impl ObjectStore {
    pub fn new(
        store: Arc<DynObjectStore>,
        location: Url,
        block_size: Option<usize>,
        commit_handler: Option<Arc<dyn CommitHandler>>,
        wrapper: Option<Arc<dyn WrappingObjectStore>>,
    ) -> Self {
        let scheme = location.scheme();
        let block_size = block_size.unwrap_or_else(|| infer_block_size(scheme));
        let commit_handler = commit_handler
            .unwrap_or_else(|| Arc::new(RenameCommitHandler) as Arc<dyn CommitHandler>);

        let store = match wrapper {
            Some(wrapper) => wrapper.wrap(store),
            None => store,
        };

        Self {
            inner: store,
            scheme: scheme.into(),
            base_path: location.path().into(),
            block_size,
            commit_handler,
        }
    }
    pub async fn try_new<T: AsRef<str>>(
        location: T,
        options: impl Into<StorageOptions> + Clone,
    ) -> Result<Self> {
        let mut storage_options = options.into();
        let options = ObjectStoreParams::default();
        configure_store(location.as_ref(), &mut storage_options, options).await
    }
}

fn infer_block_size(scheme: &str) -> usize {
    // Block size: On local file systems, we use 4KB block size. On cloud
    // object stores, we use 64KB block size. This is generally the largest
    // block size where we don't see a latency penalty.
    match scheme {
        "file" => 4 * 1024,
        _ => 64 * 1024,
    }
}

fn str_is_truthy(val: &str) -> bool {
    val.eq_ignore_ascii_case("1")
        | val.eq_ignore_ascii_case("true")
        | val.eq_ignore_ascii_case("on")
        | val.eq_ignore_ascii_case("yes")
        | val.eq_ignore_ascii_case("y")
}

/// Attempt to create a Url from given table location.
///
/// The location could be:
///  * A valid URL, which will be parsed and returned
///  * A path to a directory, which will be created and then converted to a URL.
///
/// If it is a local path, it will be created if it doesn't exist.
///
/// Extra slashes will be removed from the end path as well.
///
/// Will return an error if the location is not valid. For example,
pub fn ensure_table_uri(table_uri: impl AsRef<str>) -> Result<Url> {
    let table_uri = table_uri.as_ref();

    enum UriType {
        LocalPath(PathBuf),
        Url(Url),
    }
    let uri_type: UriType = if let Ok(url) = Url::parse(table_uri) {
        if url.scheme() == "file" {
            UriType::LocalPath(url.to_file_path().map_err(|err| {
                let msg = format!("Invalid table location: {}\nError: {:?}", table_uri, err);
                Error::InvalidTableLocation { message: msg }
            })?)
        // NOTE this check is required to support absolute windows paths which may properly parse as url
        } else if KNOWN_SCHEMES.contains(&url.scheme()) {
            UriType::Url(url)
        } else {
            UriType::LocalPath(PathBuf::from(table_uri))
        }
    } else {
        UriType::LocalPath(PathBuf::from(table_uri))
    };

    // If it is a local path, we need to create it if it does not exist.
    let mut url = match uri_type {
        UriType::LocalPath(path) => {
            let path = std::fs::canonicalize(path).map_err(|err| Error::DatasetNotFound {
                path: table_uri.to_string(),
                source: Box::new(err),
                location: location!(),
            })?;
            Url::from_directory_path(path).map_err(|_| {
                let msg = format!(
                    "Could not construct a URL from canonicalized path: {}.\n\
                  Something must be very wrong with the table path.",
                    table_uri
                );
                Error::InvalidTableLocation { message: msg }
            })?
        }
        UriType::Url(url) => url,
    };

    let trimmed_path = url.path().trim_end_matches('/').to_owned();
    url.set_path(&trimmed_path);
    Ok(url)
}

lazy_static::lazy_static! {
  static ref KNOWN_SCHEMES: Vec<&'static str> =
      Vec::from([
        "s3",
        "s3+ddb",
        "gs",
        "az",
        "file",
        "memory"
      ]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::data_type::AsBytes;
    use std::env::set_current_dir;
    use std::fs::{create_dir_all, write};
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Write test content to file.
    fn write_to_file(path_str: &str, contents: &str) -> std::io::Result<()> {
        let expanded = tilde(path_str).to_string();
        let path = StdPath::new(&expanded);
        std::fs::create_dir_all(path.parent().unwrap())?;
        write(path, contents)
    }

    async fn read_from_store(store: ObjectStore, path: &Path) -> Result<String> {
        let test_file_store = store.open(path).await.unwrap();
        let size = test_file_store.size().await.unwrap();
        let bytes = test_file_store.get_range(0..size).await.unwrap();
        let contents = String::from_utf8(bytes.to_vec()).unwrap();
        Ok(contents)
    }

    #[tokio::test]
    async fn test_absolute_paths() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path().to_str().unwrap().to_owned();
        write_to_file(
            &format!("{tmp_path}/bar/foo.lance/test_file"),
            "TEST_CONTENT",
        )
        .unwrap();

        // test a few variations of the same path
        for uri in &[
            format!("{tmp_path}/bar/foo.lance"),
            format!("{tmp_path}/./bar/foo.lance"),
            format!("{tmp_path}/bar/foo.lance/../foo.lance"),
        ] {
            let (store, path) = ObjectStore::from_uri(uri).await.unwrap();
            let contents = read_from_store(store, &path.child("test_file"))
                .await
                .unwrap();
            assert_eq!(contents, "TEST_CONTENT");
        }
    }

    #[tokio::test]
    async fn test_relative_paths() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path().to_str().unwrap().to_owned();
        write_to_file(
            &format!("{tmp_path}/bar/foo.lance/test_file"),
            "RELATIVE_URL",
        )
        .unwrap();

        set_current_dir(StdPath::new(&tmp_path)).expect("Error changing current dir");
        let (store, path) = ObjectStore::from_uri("./bar/foo.lance").await.unwrap();

        let contents = read_from_store(store, &path.child("test_file"))
            .await
            .unwrap();
        assert_eq!(contents, "RELATIVE_URL");
    }

    #[tokio::test]
    async fn test_tilde_expansion() {
        let uri = "~/foo.lance";
        write_to_file(&format!("{uri}/test_file"), "TILDE").unwrap();
        let (store, path) = ObjectStore::from_uri(uri).await.unwrap();
        let contents = read_from_store(store, &path.child("test_file"))
            .await
            .unwrap();
        assert_eq!(contents, "TILDE");
    }

    #[tokio::test]
    async fn test_read_directory() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path();
        create_dir_all(path.join("foo").join("bar")).unwrap();
        create_dir_all(path.join("foo").join("zoo")).unwrap();
        create_dir_all(path.join("foo").join("zoo").join("abc")).unwrap();
        write_to_file(
            path.join("foo").join("test_file").to_str().unwrap(),
            "read_dir",
        )
        .unwrap();
        let (store, base) = ObjectStore::from_uri(path.to_str().unwrap()).await.unwrap();

        let sub_dirs = store.read_dir(base.child("foo")).await.unwrap();
        assert_eq!(sub_dirs, vec!["bar", "zoo", "test_file"]);
    }

    #[tokio::test]
    async fn test_delete_directory() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path();
        create_dir_all(path.join("foo").join("bar")).unwrap();
        create_dir_all(path.join("foo").join("zoo")).unwrap();
        create_dir_all(path.join("foo").join("zoo").join("abc")).unwrap();
        write_to_file(
            path.join("foo")
                .join("bar")
                .join("test_file")
                .to_str()
                .unwrap(),
            "delete",
        )
        .unwrap();
        write_to_file(path.join("foo").join("top").to_str().unwrap(), "delete_top").unwrap();
        let (store, base) = ObjectStore::from_uri(path.to_str().unwrap()).await.unwrap();
        store.remove_dir_all(base.child("foo")).await.unwrap();

        assert!(!path.join("foo").exists());
    }

    #[derive(Debug)]
    struct TestWrapper {
        called: AtomicBool,

        return_value: Arc<dyn OSObjectStore>,
    }

    impl WrappingObjectStore for TestWrapper {
        fn wrap(&self, _original: Arc<dyn OSObjectStore>) -> Arc<dyn OSObjectStore> {
            self.called.store(true, Ordering::Relaxed);

            // return a mocked value so we can check if the final store is the one we expect
            self.return_value.clone()
        }
    }

    impl TestWrapper {
        fn called(&self) -> bool {
            self.called.load(Ordering::Relaxed)
        }
    }

    #[tokio::test]
    async fn test_wrapping_object_store_option_is_used() {
        // Make a store for the inner store first
        let mock_inner_store: Arc<dyn OSObjectStore> = Arc::new(InMemory::new());

        assert_eq!(Arc::strong_count(&mock_inner_store), 1);

        let wrapper = Arc::new(TestWrapper {
            called: AtomicBool::new(false),
            return_value: mock_inner_store.clone(),
        });

        let params = ObjectStoreParams {
            object_store_wrapper: Some(wrapper.clone()),
            ..ObjectStoreParams::default()
        };

        // not called yet
        assert!(!wrapper.called());

        let _ = ObjectStore::from_uri_and_params("memory:///", &params)
            .await
            .unwrap();

        // called after construction
        assert!(wrapper.called());

        // hard to compare two trait pointers as the point to vtables
        // using the ref count as a proxy to make sure that the store is correctly kept
        assert_eq!(Arc::strong_count(&mock_inner_store), 2);
    }

    #[derive(Debug, Default)]
    struct MockAwsCredentialsProvider {
        called: AtomicBool,
    }

    #[async_trait]
    impl CredentialProvider for MockAwsCredentialsProvider {
        type Credential = ObjectStoreAwsCredential;

        async fn get_credential(&self) -> ObjectStoreResult<Arc<Self::Credential>> {
            self.called.store(true, Ordering::Relaxed);
            Ok(Arc::new(Self::Credential {
                key_id: "".to_string(),
                secret_key: "".to_string(),
                token: None,
            }))
        }
    }

    #[tokio::test]
    async fn test_injected_aws_creds_option_is_used() {
        let mock_provider = Arc::new(MockAwsCredentialsProvider::default());

        let params = ObjectStoreParams {
            aws_credentials: Some(mock_provider.clone() as AwsCredentialProvider),
            ..ObjectStoreParams::default()
        };

        // Not called yet
        assert!(!mock_provider.called.load(Ordering::Relaxed));

        let (store, _) = ObjectStore::from_uri_and_params("s3://not-a-bucket", &params)
            .await
            .unwrap();

        // fails, but we don't care
        let _ = store
            .open(&Path::parse("/").unwrap())
            .await
            .unwrap()
            .get_range(0..1)
            .await;

        // Not called yet
        assert!(mock_provider.called.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn test_local_paths() {
        let temp_dir = tempfile::tempdir().unwrap();

        let file_path = temp_dir.path().join("test_file");
        let mut writer = ObjectStore::create_local_writer(file_path.as_path())
            .await
            .unwrap();
        writer.write_all(b"LOCAL").await.unwrap();
        writer.shutdown().await.unwrap();

        let reader = ObjectStore::open_local(file_path.as_path()).await.unwrap();
        let buf = reader.get_range(0..5).await.unwrap();
        assert_eq!(buf.as_bytes(), b"LOCAL");
    }

    #[tokio::test]
    #[cfg(windows)]
    async fn test_windows_paths() {
        use std::path::Component;
        use std::path::Prefix;
        use std::path::Prefix::*;

        fn get_path_prefix(path: &StdPath) -> Prefix {
            match path.components().next().unwrap() {
                Component::Prefix(prefix_component) => prefix_component.kind(),
                _ => panic!(),
            }
        }

        fn get_drive_letter(prefix: Prefix) -> String {
            match prefix {
                Disk(bytes) => String::from_utf8(vec![bytes]).unwrap(),
                _ => panic!(),
            }
        }

        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path();
        let prefix = get_path_prefix(tmp_path);
        let drive_letter = get_drive_letter(prefix);

        write_to_file(
            &(format!("{drive_letter}:/test_folder/test.lance") + "/test_file"),
            "WINDOWS",
        )
        .unwrap();

        for uri in &[
            format!("{drive_letter}:/test_folder/test.lance"),
            format!("{drive_letter}:\\test_folder\\test.lance"),
        ] {
            let (store, base) = ObjectStore::from_uri(uri).await.unwrap();
            let contents = read_from_store(store, &base.child("test_file"))
                .await
                .unwrap();
            assert_eq!(contents, "WINDOWS");
        }
    }
}
