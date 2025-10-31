// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, fmt, str::FromStr, sync::Arc, time::Duration};

#[cfg(test)]
use mock_instant::thread_local::{SystemTime, UNIX_EPOCH};

#[cfg(not(test))]
use std::time::{SystemTime, UNIX_EPOCH};

use object_store::ObjectStore as OSObjectStore;
use object_store_opendal::OpendalStore;
use opendal::{services::S3, Operator};

use aws_config::default_provider::credentials::DefaultCredentialsChain;
use aws_credential_types::provider::ProvideCredentials;
use object_store::{
    aws::{
        AmazonS3Builder, AmazonS3ConfigKey, AwsCredential as ObjectStoreAwsCredential,
        AwsCredentialProvider,
    },
    ClientOptions, CredentialProvider, Result as ObjectStoreResult, RetryConfig,
    StaticCredentialProvider,
};
use snafu::location;
use tokio::sync::RwLock;
use url::Url;

use crate::object_store::{
    ObjectStore, ObjectStoreParams, ObjectStoreProvider, StorageOptions, StorageOptionsProvider,
    DEFAULT_CLOUD_BLOCK_SIZE, DEFAULT_CLOUD_IO_PARALLELISM, DEFAULT_MAX_IOP_SIZE,
};
use lance_core::error::{Error, Result};

#[derive(Default, Debug)]
pub struct AwsStoreProvider;

impl AwsStoreProvider {
    async fn build_amazon_s3_store(
        &self,
        base_path: &mut Url,
        params: &ObjectStoreParams,
        storage_options: &StorageOptions,
        is_s3_express: bool,
    ) -> Result<Arc<dyn OSObjectStore>> {
        let max_retries = storage_options.client_max_retries();
        let retry_timeout = storage_options.client_retry_timeout();
        let retry_config = RetryConfig {
            backoff: Default::default(),
            max_retries,
            retry_timeout: Duration::from_secs(retry_timeout),
        };

        let mut s3_storage_options = storage_options.as_s3_options();
        let region = resolve_s3_region(base_path, &s3_storage_options).await?;
        let (aws_creds, region) = build_aws_credential(
            params.s3_credentials_refresh_offset,
            params.aws_credentials.clone(),
            Some(&s3_storage_options),
            region,
            params.storage_options_provider.clone(),
            storage_options.expires_at_millis(),
        )
        .await?;

        // Set S3Express flag if detected
        if is_s3_express {
            s3_storage_options.insert(AmazonS3ConfigKey::S3Express, true.to_string());
        }

        // before creating the OSObjectStore we need to rewrite the url to drop ddb related parts
        base_path.set_scheme("s3").unwrap();
        base_path.set_query(None);

        // we can't use parse_url_opts here because we need to manually set the credentials provider
        let mut builder = AmazonS3Builder::new();
        for (key, value) in s3_storage_options {
            builder = builder.with_config(key, value);
        }
        builder = builder
            .with_url(base_path.as_ref())
            .with_credentials(aws_creds)
            .with_retry(retry_config)
            .with_region(region);

        Ok(Arc::new(builder.build()?) as Arc<dyn OSObjectStore>)
    }

    async fn build_opendal_s3_store(
        &self,
        base_path: &Url,
        storage_options: &StorageOptions,
    ) -> Result<Arc<dyn OSObjectStore>> {
        let bucket = base_path
            .host_str()
            .ok_or_else(|| Error::invalid_input("S3 URL must contain bucket name", location!()))?
            .to_string();

        let prefix = base_path.path().trim_start_matches('/').to_string();

        // Start with all storage options as the config map
        // OpenDAL will handle environment variables through its default credentials chain
        let mut config_map: HashMap<String, String> = storage_options.0.clone();

        // Set required OpenDAL configuration
        config_map.insert("bucket".to_string(), bucket);

        if !prefix.is_empty() {
            config_map.insert("root".to_string(), "/".to_string());
        }

        let operator = Operator::from_iter::<S3>(config_map)
            .map_err(|e| {
                Error::invalid_input(
                    format!("Failed to create S3 operator: {:?}", e),
                    location!(),
                )
            })?
            .finish();

        Ok(Arc::new(OpendalStore::new(operator)) as Arc<dyn OSObjectStore>)
    }
}

#[async_trait::async_trait]
impl ObjectStoreProvider for AwsStoreProvider {
    async fn new_store(
        &self,
        mut base_path: Url,
        params: &ObjectStoreParams,
    ) -> Result<ObjectStore> {
        let block_size = params.block_size.unwrap_or(DEFAULT_CLOUD_BLOCK_SIZE);
        let mut storage_options =
            StorageOptions(params.storage_options.clone().unwrap_or_default());
        storage_options.with_env_s3();
        let download_retry_count = storage_options.download_retry_count();

        let use_opendal = storage_options
            .0
            .get("use_opendal")
            .map(|v| v == "true")
            .unwrap_or(false);

        // Determine S3 Express and constant size upload parts before building the store
        let is_s3_express = check_s3_express(&base_path, &storage_options);

        let use_constant_size_upload_parts = storage_options
            .0
            .get("aws_endpoint")
            .map(|endpoint| endpoint.contains("r2.cloudflarestorage.com"))
            .unwrap_or(false);

        let inner = if use_opendal {
            // Use OpenDAL implementation
            self.build_opendal_s3_store(&base_path, &storage_options)
                .await?
        } else {
            // Use default Amazon S3 implementation
            self.build_amazon_s3_store(&mut base_path, params, &storage_options, is_s3_express)
                .await?
        };

        Ok(ObjectStore {
            inner,
            scheme: String::from(base_path.scheme()),
            block_size,
            max_iop_size: *DEFAULT_MAX_IOP_SIZE,
            use_constant_size_upload_parts,
            list_is_lexically_ordered: !is_s3_express,
            io_parallelism: DEFAULT_CLOUD_IO_PARALLELISM,
            download_retry_count,
        })
    }
}

/// Check if the storage is S3 Express
fn check_s3_express(url: &Url, storage_options: &StorageOptions) -> bool {
    storage_options
        .0
        .get("s3_express")
        .map(|v| v == "true")
        .unwrap_or(false)
        || url.authority().ends_with("--x-s3")
}

/// Figure out the S3 region of the bucket.
///
/// This resolves in order of precedence:
/// 1. The region provided in the storage options
/// 2. (If endpoint is not set), the region returned by the S3 API for the bucket
///
/// It can return None if no region is provided and the endpoint is set.
async fn resolve_s3_region(
    url: &Url,
    storage_options: &HashMap<AmazonS3ConfigKey, String>,
) -> Result<Option<String>> {
    if let Some(region) = storage_options.get(&AmazonS3ConfigKey::Region) {
        Ok(Some(region.clone()))
    } else if storage_options.get(&AmazonS3ConfigKey::Endpoint).is_none() {
        // If no endpoint is set, we can assume this is AWS S3 and the region
        // can be resolved from the bucket.
        let bucket = url.host_str().ok_or_else(|| {
            Error::invalid_input(
                format!("Could not parse bucket from url: {}", url),
                location!(),
            )
        })?;

        let mut client_options = ClientOptions::default();
        for (key, value) in storage_options {
            if let AmazonS3ConfigKey::Client(client_key) = key {
                client_options = client_options.with_config(*client_key, value.clone());
            }
        }

        let bucket_region =
            object_store::aws::resolve_bucket_region(bucket, &client_options).await?;
        Ok(Some(bucket_region))
    } else {
        Ok(None)
    }
}

/// Build AWS credentials
///
/// This resolves credentials from the following sources in order:
/// 1. An explicit `storage_options_provider`
/// 2. An explicit `credentials` provider
/// 3. Explicit credentials in storage_options (as in `aws_access_key_id`,
///    `aws_secret_access_key`, `aws_session_token`)
/// 4. The default credential provider chain from AWS SDK.
///
/// # Initial Credentials with Storage Options Provider
///
/// When `storage_options_provider` is provided along with `storage_options` and
/// `expires_at_millis`, these serve as **initial values** to avoid redundant calls to
/// fetch new storage options. The provider will use these initial credentials until they
/// expire (based on `expires_at_millis`), then automatically fetch fresh credentials from
/// the provider. Once the initial credentials expire, the passed-in values are no longer
/// used - all future credentials come from the provider's `fetch_storage_options()` method.
///
/// `credentials_refresh_offset` is the amount of time before expiry to refresh credentials.
pub async fn build_aws_credential(
    credentials_refresh_offset: Duration,
    credentials: Option<AwsCredentialProvider>,
    storage_options: Option<&HashMap<AmazonS3ConfigKey, String>>,
    region: Option<String>,
    storage_options_provider: Option<Arc<dyn StorageOptionsProvider>>,
    expires_at_millis: Option<u64>,
) -> Result<(AwsCredentialProvider, String)> {
    // TODO: make this return no credential provider not using AWS
    use aws_config::meta::region::RegionProviderChain;
    const DEFAULT_REGION: &str = "us-west-2";

    let region = if let Some(region) = region {
        region
    } else {
        RegionProviderChain::default_provider()
            .or_else(DEFAULT_REGION)
            .region()
            .await
            .map(|r| r.as_ref().to_string())
            .unwrap_or(DEFAULT_REGION.to_string())
    };

    let storage_options_credentials = storage_options.and_then(extract_static_s3_credentials);
    if let Some(storage_options_provider) = storage_options_provider {
        let creds = build_aws_credential_with_storage_options_provider(
            storage_options_provider,
            credentials_refresh_offset,
            credentials,
            storage_options_credentials,
            expires_at_millis,
        )
        .await?;
        Ok((creds, region))
    } else if let Some(creds) = credentials {
        Ok((creds, region))
    } else if let Some(creds) = storage_options_credentials {
        Ok((Arc::new(creds), region))
    } else {
        let credentials_provider = DefaultCredentialsChain::builder().build().await;

        Ok((
            Arc::new(AwsCredentialAdapter::new(
                Arc::new(credentials_provider),
                credentials_refresh_offset,
            )),
            region,
        ))
    }
}

async fn build_aws_credential_with_storage_options_provider(
    storage_options_provider: Arc<dyn StorageOptionsProvider>,
    credentials_refresh_offset: Duration,
    credentials: Option<AwsCredentialProvider>,
    storage_options_credentials: Option<StaticCredentialProvider<ObjectStoreAwsCredential>>,
    expires_at_millis: Option<u64>,
) -> Result<AwsCredentialProvider> {
    match (expires_at_millis, credentials, storage_options_credentials) {
        // Case 1: provider + credentials + expiration time
        (Some(expires_at), Some(cred), _) => {
            Ok(Arc::new(
                DynamicStorageOptionsCredentialProvider::new_with_initial_credential(
                    storage_options_provider,
                    credentials_refresh_offset,
                    cred.get_credential().await?,
                    expires_at,
                ),
            ))
        }
        // Case 2: provider + storage_options (with valid credentials) + expiration time
        (Some(expires_at), None, Some(cred)) => {
            Ok(Arc::new(
                DynamicStorageOptionsCredentialProvider::new_with_initial_credential(
                    storage_options_provider,
                    credentials_refresh_offset,
                    cred.get_credential().await?,
                    expires_at,
                ),
            ))
        }
        // Case 3: provider + storage_options without expiration - FAIL
        (None, None, Some(_)) => Err(Error::IO {
            source: Box::new(std::io::Error::other(
                "expires_at_millis is required when using storage_options_provider with storage_options",
            )),
            location: location!(),
        }),
        // Case 4: provider + credentials without expiration - FAIL
        (None, Some(_), _) => Err(Error::IO {
            source: Box::new(std::io::Error::other(
                "expires_at_millis is required when using storage_options_provider with credentials",
            )),
            location: location!(),
        }),
        // Case 5: provider without credentials/storage_options, or with expiration but no creds/opts
        (_, None, None) => Ok(Arc::new(DynamicStorageOptionsCredentialProvider::new(
            storage_options_provider,
            credentials_refresh_offset,
        ))),
    }
}

fn extract_static_s3_credentials(
    options: &HashMap<AmazonS3ConfigKey, String>,
) -> Option<StaticCredentialProvider<ObjectStoreAwsCredential>> {
    let key_id = options.get(&AmazonS3ConfigKey::AccessKeyId).cloned();
    let secret_key = options.get(&AmazonS3ConfigKey::SecretAccessKey).cloned();
    let token = options.get(&AmazonS3ConfigKey::Token).cloned();
    match (key_id, secret_key, token) {
        (Some(key_id), Some(secret_key), token) => {
            Some(StaticCredentialProvider::new(ObjectStoreAwsCredential {
                key_id,
                secret_key,
                token,
            }))
        }
        _ => None,
    }
}

/// Adapt an AWS SDK cred into object_store credentials
#[derive(Debug)]
pub struct AwsCredentialAdapter {
    pub inner: Arc<dyn ProvideCredentials>,

    // RefCell can't be shared across threads, so we use HashMap
    cache: Arc<RwLock<HashMap<String, Arc<aws_credential_types::Credentials>>>>,

    // The amount of time before expiry to refresh credentials
    credentials_refresh_offset: Duration,
}

impl AwsCredentialAdapter {
    pub fn new(
        provider: Arc<dyn ProvideCredentials>,
        credentials_refresh_offset: Duration,
    ) -> Self {
        Self {
            inner: provider,
            cache: Arc::new(RwLock::new(HashMap::new())),
            credentials_refresh_offset,
        }
    }
}

const AWS_CREDS_CACHE_KEY: &str = "aws_credentials";

/// Convert std::time::SystemTime from AWS SDK to our mockable SystemTime
fn to_system_time(time: std::time::SystemTime) -> SystemTime {
    let duration_since_epoch = time
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time should be after UNIX_EPOCH");
    UNIX_EPOCH + duration_since_epoch
}

#[async_trait::async_trait]
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
                            to_system_time(exp)
                                .checked_sub(self.credentials_refresh_offset)
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
            let refreshed_creds = Arc::new(self.inner.provide_credentials().await.map_err(
                |e| Error::Internal {
                    message: format!("Failed to get AWS credentials: {:?}", e),
                    location: location!(),
                },
            )?);

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

impl StorageOptions {
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
}

impl ObjectStoreParams {
    /// Create a new instance of [`ObjectStoreParams`] based on the AWS credentials.
    pub fn with_aws_credentials(
        aws_credentials: Option<AwsCredentialProvider>,
        region: Option<String>,
    ) -> Self {
        Self {
            aws_credentials,
            storage_options: region
                .map(|region| [("region".into(), region)].iter().cloned().collect()),
            ..Default::default()
        }
    }
}

/// AWS Credential Provider that uses StorageOptionsProvider
///
/// This adapter converts our generic StorageOptionsProvider trait into
/// AWS-specific credentials that can be used with S3. It caches credentials
/// and automatically refreshes them before they expire.
///
/// # Future Work
///
/// TODO: Support AWS/GCP/Azure together in a unified credential provider.
/// Currently this is AWS-specific. Needs investigation of how GCP and Azure credential
/// refresh mechanisms work and whether they can be unified with AWS's approach.
///
/// See: <https://github.com/lancedb/lance/pull/4905#discussion_r2474605265>
pub struct DynamicStorageOptionsCredentialProvider {
    provider: Arc<dyn StorageOptionsProvider>,
    cache: Arc<RwLock<Option<CachedCredential>>>,
    refresh_offset: Duration,
}

impl fmt::Debug for DynamicStorageOptionsCredentialProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DynamicStorageOptionsCredentialProvider")
            .field("provider", &self.provider)
            .field("refresh_offset", &self.refresh_offset)
            .finish()
    }
}

#[derive(Debug, Clone)]
struct CachedCredential {
    credential: Arc<ObjectStoreAwsCredential>,
    expires_at_millis: Option<u64>,
}

impl DynamicStorageOptionsCredentialProvider {
    /// Create a new credential provider without initial credentials
    ///
    /// # Arguments
    /// * `provider` - The storage options provider
    /// * `refresh_offset` - Duration before expiry to refresh credentials
    pub fn new(provider: Arc<dyn StorageOptionsProvider>, refresh_offset: Duration) -> Self {
        Self {
            provider,
            cache: Arc::new(RwLock::new(None)),
            refresh_offset,
        }
    }

    /// Create a new credential provider with initial credentials from an explicit credential
    ///
    /// # Arguments
    /// * `provider` - The storage options provider
    /// * `refresh_offset` - Duration before expiry to refresh credentials
    /// * `credential` - Initial credential to cache
    /// * `expires_at_millis` - Expiration time in milliseconds since epoch (required for refresh)
    pub fn new_with_initial_credential(
        provider: Arc<dyn StorageOptionsProvider>,
        refresh_offset: Duration,
        credential: Arc<ObjectStoreAwsCredential>,
        expires_at_millis: u64,
    ) -> Self {
        Self {
            provider,
            cache: Arc::new(RwLock::new(Some(CachedCredential {
                credential,
                expires_at_millis: Some(expires_at_millis),
            }))),
            refresh_offset,
        }
    }

    fn needs_refresh(&self, cached: &Option<CachedCredential>) -> bool {
        match cached {
            None => true,
            Some(cached_cred) => {
                if let Some(expires_at_millis) = cached_cred.expires_at_millis {
                    let now_ms = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or(Duration::from_secs(0))
                        .as_millis() as u64;

                    // Refresh if we're within the refresh offset of expiration
                    let refresh_offset_millis = self.refresh_offset.as_millis() as u64;
                    now_ms + refresh_offset_millis >= expires_at_millis
                } else {
                    // No expiration means credentials never expire
                    false
                }
            }
        }
    }

    async fn do_get_credential(&self) -> ObjectStoreResult<Option<Arc<ObjectStoreAwsCredential>>> {
        // Check if we have valid cached credentials with read lock
        {
            let cached = self.cache.read().await;
            if !self.needs_refresh(&cached) {
                if let Some(cached_cred) = &*cached {
                    return Ok(Some(cached_cred.credential.clone()));
                }
            }
        }

        // Try to acquire write lock - if it fails, return None and let caller retry
        let Ok(mut cache) = self.cache.try_write() else {
            return Ok(None);
        };

        // Double-check if credentials are still stale after acquiring write lock
        // (another thread might have refreshed them)
        if !self.needs_refresh(&cache) {
            if let Some(cached_cred) = &*cache {
                return Ok(Some(cached_cred.credential.clone()));
            }
        }

        let storage_options_map = self
            .provider
            .fetch_storage_options()
            .await
            .map_err(|e| object_store::Error::Generic {
                store: "DynamicStorageOptionsCredentialProvider",
                source: Box::new(e),
            })?
            .ok_or_else(|| object_store::Error::Generic {
                store: "DynamicStorageOptionsCredentialProvider",
                source: "No storage options available".into(),
            })?;

        let storage_options = StorageOptions(storage_options_map);
        let expires_at_millis = storage_options.expires_at_millis();
        let s3_options = storage_options.as_s3_options();
        let static_creds = extract_static_s3_credentials(&s3_options).ok_or_else(|| {
            object_store::Error::Generic {
                store: "DynamicStorageOptionsCredentialProvider",
                source: "Missing required credentials in storage options".into(),
            }
        })?;

        let credential =
            static_creds
                .get_credential()
                .await
                .map_err(|e| object_store::Error::Generic {
                    store: "DynamicStorageOptionsCredentialProvider",
                    source: Box::new(e),
                })?;

        *cache = Some(CachedCredential {
            credential: credential.clone(),
            expires_at_millis,
        });

        Ok(Some(credential))
    }
}

#[async_trait::async_trait]
impl CredentialProvider for DynamicStorageOptionsCredentialProvider {
    type Credential = ObjectStoreAwsCredential;

    async fn get_credential(&self) -> ObjectStoreResult<Arc<Self::Credential>> {
        // Retry loop - if do_get_credential returns None (lock busy), retry from the beginning
        loop {
            match self.do_get_credential().await? {
                Some(cred) => return Ok(cred),
                None => {
                    // Lock was busy, wait 10ms before retrying
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    continue;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::object_store::ObjectStoreRegistry;
    use mock_instant::thread_local::MockClock;
    use object_store::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    #[derive(Debug, Default)]
    struct MockAwsCredentialsProvider {
        called: AtomicBool,
    }

    #[async_trait::async_trait]
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
        let registry = Arc::new(ObjectStoreRegistry::default());

        let params = ObjectStoreParams {
            aws_credentials: Some(mock_provider.clone() as AwsCredentialProvider),
            ..ObjectStoreParams::default()
        };

        // Not called yet
        assert!(!mock_provider.called.load(Ordering::Relaxed));

        let (store, _) = ObjectStore::from_uri_and_params(registry, "s3://not-a-bucket", &params)
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

    #[test]
    fn test_s3_path_parsing() {
        let provider = AwsStoreProvider;

        let cases = [
            ("s3://bucket/path/to/file", "path/to/file"),
            // for non ASCII string tests
            ("s3://bucket/测试path/to/file", "测试path/to/file"),
            ("s3://bucket/path/&to/file", "path/&to/file"),
            ("s3://bucket/path/=to/file", "path/=to/file"),
            (
                "s3+ddb://bucket/path/to/file?ddbTableName=test",
                "path/to/file",
            ),
        ];

        for (uri, expected_path) in cases {
            let url = Url::parse(uri).unwrap();
            let path = provider.extract_path(&url).unwrap();
            let expected_path = Path::from(expected_path);
            assert_eq!(path, expected_path)
        }
    }

    #[test]
    fn test_is_s3_express() {
        let cases = [
            (
                "s3://bucket/path/to/file",
                HashMap::from([("s3_express".to_string(), "true".to_string())]),
                true,
            ),
            (
                "s3://bucket/path/to/file",
                HashMap::from([("s3_express".to_string(), "false".to_string())]),
                false,
            ),
            ("s3://bucket/path/to/file", HashMap::from([]), false),
            (
                "s3://bucket--x-s3/path/to/file",
                HashMap::from([("s3_express".to_string(), "true".to_string())]),
                true,
            ),
            (
                "s3://bucket--x-s3/path/to/file",
                HashMap::from([("s3_express".to_string(), "false".to_string())]),
                true, // URL takes precedence
            ),
            ("s3://bucket--x-s3/path/to/file", HashMap::from([]), true),
        ];

        for (uri, storage_map, expected) in cases {
            let url = Url::parse(uri).unwrap();
            let storage_options = StorageOptions(storage_map);
            let is_s3_express = check_s3_express(&url, &storage_options);
            assert_eq!(is_s3_express, expected);
        }
    }

    #[tokio::test]
    async fn test_use_opendal_flag() {
        let provider = AwsStoreProvider;
        let url = Url::parse("s3://test-bucket/path").unwrap();
        let params_with_flag = ObjectStoreParams {
            storage_options: Some(HashMap::from([
                ("use_opendal".to_string(), "true".to_string()),
                ("region".to_string(), "us-west-2".to_string()),
            ])),
            ..Default::default()
        };

        let store = provider
            .new_store(url.clone(), &params_with_flag)
            .await
            .unwrap();
        assert_eq!(store.scheme, "s3");
    }

    #[derive(Debug)]
    struct MockStorageOptionsProvider {
        call_count: Arc<RwLock<usize>>,
        expires_in_millis: Option<u64>,
    }

    impl MockStorageOptionsProvider {
        fn new(expires_in_millis: Option<u64>) -> Self {
            Self {
                call_count: Arc::new(RwLock::new(0)),
                expires_in_millis,
            }
        }

        async fn get_call_count(&self) -> usize {
            *self.call_count.read().await
        }
    }

    #[async_trait::async_trait]
    impl StorageOptionsProvider for MockStorageOptionsProvider {
        async fn fetch_storage_options(&self) -> Result<Option<HashMap<String, String>>> {
            let count = {
                let mut c = self.call_count.write().await;
                *c += 1;
                *c
            };

            let mut options = HashMap::from([
                ("aws_access_key_id".to_string(), format!("AKID_{}", count)),
                (
                    "aws_secret_access_key".to_string(),
                    format!("SECRET_{}", count),
                ),
                ("aws_session_token".to_string(), format!("TOKEN_{}", count)),
            ]);

            if let Some(expires_in) = self.expires_in_millis {
                let now_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64;
                let expires_at = now_ms + expires_in;
                options.insert("expires_at_millis".to_string(), expires_at.to_string());
            }

            Ok(Some(options))
        }

        fn provider_id(&self) -> String {
            let ptr = Arc::as_ptr(&self.call_count) as usize;
            format!("MockStorageOptionsProvider {{ id: {} }}", ptr)
        }
    }

    #[tokio::test]
    async fn test_dynamic_credential_provider_with_initial_cache() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        let now_ms = MockClock::system_time().as_millis() as u64;

        // Create a mock provider that returns credentials expiring in 10 minutes
        let mock = Arc::new(MockStorageOptionsProvider::new(Some(
            600_000, // Expires in 10 minutes
        )));

        // Create credential provider with initial cached credentials that expire in 10 minutes
        let expires_at = now_ms + 600_000; // 10 minutes from now
        let initial_cred = Arc::new(ObjectStoreAwsCredential {
            key_id: "AKID_CACHED".to_string(),
            secret_key: "SECRET_CACHED".to_string(),
            token: Some("TOKEN_CACHED".to_string()),
        });

        let provider = DynamicStorageOptionsCredentialProvider::new_with_initial_credential(
            mock.clone(),
            Duration::from_secs(300), // 5 minute refresh offset
            initial_cred,
            expires_at,
        );

        // First call should use cached credentials (not expired yet)
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_CACHED");
        assert_eq!(cred.secret_key, "SECRET_CACHED");
        assert_eq!(cred.token, Some("TOKEN_CACHED".to_string()));

        // Should not have called the provider yet
        assert_eq!(mock.get_call_count().await, 0);
    }

    #[tokio::test]
    async fn test_dynamic_credential_provider_with_expired_cache() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        let now_ms = MockClock::system_time().as_millis() as u64;

        // Create a mock provider that returns credentials expiring in 10 minutes
        let mock = Arc::new(MockStorageOptionsProvider::new(Some(
            600_000, // Expires in 10 minutes
        )));

        // Create credential provider with initial cached credentials that expired 1 second ago
        let expired_time = now_ms - 1_000; // 1 second ago
        let initial_cred = Arc::new(ObjectStoreAwsCredential {
            key_id: "AKID_EXPIRED".to_string(),
            secret_key: "SECRET_EXPIRED".to_string(),
            token: None,
        });

        let provider = DynamicStorageOptionsCredentialProvider::new_with_initial_credential(
            mock.clone(),
            Duration::from_secs(300), // 5 minute refresh offset
            initial_cred,
            expired_time,
        );

        // First call should fetch new credentials because cached ones are expired
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_1");
        assert_eq!(cred.secret_key, "SECRET_1");
        assert_eq!(cred.token, Some("TOKEN_1".to_string()));

        // Should have called the provider once
        assert_eq!(mock.get_call_count().await, 1);
    }

    #[tokio::test]
    async fn test_dynamic_credential_provider_refresh_lead_time() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        // Create a mock provider that returns credentials expiring in 4 minutes
        let mock = Arc::new(MockStorageOptionsProvider::new(Some(
            240_000, // Expires in 4 minutes
        )));

        // Create credential provider with 5 minute refresh offset
        // This means credentials should be refreshed when they have less than 5 minutes left
        let provider = DynamicStorageOptionsCredentialProvider::new(
            mock.clone(),
            Duration::from_secs(300), // 5 minute refresh offset
        );

        // First call should fetch credentials from provider (no initial cache)
        // Credentials expire in 4 minutes, which is less than our 5 minute refresh offset,
        // so they should be considered "needs refresh" immediately
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_1");
        assert_eq!(mock.get_call_count().await, 1);

        // Second call should trigger refresh because credentials expire in 4 minutes
        // but our refresh lead time is 5 minutes (now + 5min > expires_at)
        // The mock will return new credentials (AKID_2) with the same expiration
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_2");
        assert_eq!(mock.get_call_count().await, 2);
    }

    #[tokio::test]
    async fn test_dynamic_credential_provider_no_initial_cache() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        // Create a mock provider that returns credentials expiring in 10 minutes
        let mock = Arc::new(MockStorageOptionsProvider::new(Some(
            600_000, // Expires in 10 minutes
        )));

        // Create credential provider without initial cache
        let provider = DynamicStorageOptionsCredentialProvider::new(
            mock.clone(),
            Duration::from_secs(300), // 5 minute refresh offset
        );

        // First call should fetch from provider (call count = 1)
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_1");
        assert_eq!(cred.secret_key, "SECRET_1");
        assert_eq!(cred.token, Some("TOKEN_1".to_string()));
        assert_eq!(mock.get_call_count().await, 1);

        // Second call should use cached credentials (not expired yet)
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_1");
        assert_eq!(mock.get_call_count().await, 1); // Still 1, didn't fetch again

        // Advance time to 6 minutes - should trigger refresh (within 5 min refresh offset)
        MockClock::set_system_time(Duration::from_secs(100_000 + 360));
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_2");
        assert_eq!(cred.secret_key, "SECRET_2");
        assert_eq!(cred.token, Some("TOKEN_2".to_string()));
        assert_eq!(mock.get_call_count().await, 2);

        // Advance time to 11 minutes total - should trigger another refresh
        MockClock::set_system_time(Duration::from_secs(100_000 + 660));
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_3");
        assert_eq!(cred.secret_key, "SECRET_3");
        assert_eq!(mock.get_call_count().await, 3);
    }

    #[tokio::test]
    async fn test_dynamic_credential_provider_with_initial_credential() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        let now_ms = MockClock::system_time().as_millis() as u64;

        // Create a mock provider that returns credentials expiring in 10 minutes
        let mock = Arc::new(MockStorageOptionsProvider::new(Some(
            600_000, // Expires in 10 minutes
        )));

        // Create an initial credential with expiration in 10 minutes
        let expires_at = now_ms + 600_000; // 10 minutes from now
        let initial_cred = Arc::new(ObjectStoreAwsCredential {
            key_id: "AKID_INITIAL".to_string(),
            secret_key: "SECRET_INITIAL".to_string(),
            token: Some("TOKEN_INITIAL".to_string()),
        });

        // Create credential provider with initial credential and expiration
        let provider = DynamicStorageOptionsCredentialProvider::new_with_initial_credential(
            mock.clone(),
            Duration::from_secs(300), // 5 minute refresh offset
            initial_cred,
            expires_at,
        );

        // First call should use the initial credential (not expired yet)
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_INITIAL");
        assert_eq!(cred.secret_key, "SECRET_INITIAL");
        assert_eq!(cred.token, Some("TOKEN_INITIAL".to_string()));

        // Should not have called the provider yet
        assert_eq!(mock.get_call_count().await, 0);

        // Advance time to 6 minutes - this should trigger a refresh
        // (5 minute refresh offset means we refresh 5 minutes before expiration)
        MockClock::set_system_time(Duration::from_secs(100_000 + 360));
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_1");
        assert_eq!(cred.secret_key, "SECRET_1");
        assert_eq!(cred.token, Some("TOKEN_1".to_string()));

        // Should have called the provider once
        assert_eq!(mock.get_call_count().await, 1);

        // Advance time to 11 minutes total - this should trigger another refresh
        MockClock::set_system_time(Duration::from_secs(100_000 + 660));
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_2");
        assert_eq!(cred.secret_key, "SECRET_2");
        assert_eq!(cred.token, Some("TOKEN_2".to_string()));

        // Should have called the provider twice
        assert_eq!(mock.get_call_count().await, 2);

        // Advance time to 16 minutes total - this should trigger yet another refresh
        MockClock::set_system_time(Duration::from_secs(100_000 + 960));
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_3");
        assert_eq!(cred.secret_key, "SECRET_3");
        assert_eq!(cred.token, Some("TOKEN_3".to_string()));

        // Should have called the provider three times
        assert_eq!(mock.get_call_count().await, 3);
    }

    #[tokio::test]
    async fn test_dynamic_credential_provider_concurrent_access() {
        // Create a mock provider with far future expiration
        let mock = Arc::new(MockStorageOptionsProvider::new(Some(9999999999999)));

        let provider = Arc::new(DynamicStorageOptionsCredentialProvider::new(
            mock.clone(),
            Duration::from_secs(300),
        ));

        // Spawn 10 concurrent tasks that all try to get credentials at the same time
        let mut handles = vec![];
        for i in 0..10 {
            let provider = provider.clone();
            let handle = tokio::spawn(async move {
                let cred = provider.get_credential().await.unwrap();
                // Verify we got the correct credentials (should all be AKID_1 from first fetch)
                assert_eq!(cred.key_id, "AKID_1");
                assert_eq!(cred.secret_key, "SECRET_1");
                assert_eq!(cred.token, Some("TOKEN_1".to_string()));
                i // Return task number for verification
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // Verify all 10 tasks completed successfully
        assert_eq!(results.len(), 10);
        for i in 0..10 {
            assert!(results.contains(&i));
        }

        // The provider should have been called exactly once (first request triggers fetch,
        // subsequent requests use cache)
        let call_count = mock.get_call_count().await;
        assert_eq!(
            call_count, 1,
            "Provider should be called exactly once despite concurrent access"
        );
    }

    #[tokio::test]
    async fn test_dynamic_credential_provider_concurrent_refresh() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        let now_ms = MockClock::system_time().as_millis() as u64;

        // Create initial credentials that expired in the past (1000 seconds ago)
        let expires_at = now_ms - 1_000_000;

        let initial_cred = Arc::new(ObjectStoreAwsCredential {
            key_id: "AKID_OLD".to_string(),
            secret_key: "SECRET_OLD".to_string(),
            token: Some("TOKEN_OLD".to_string()),
        });

        // Mock will return credentials expiring in 1 hour
        let mock = Arc::new(MockStorageOptionsProvider::new(Some(
            3_600_000, // Expires in 1 hour
        )));

        let provider = Arc::new(
            DynamicStorageOptionsCredentialProvider::new_with_initial_credential(
                mock.clone(),
                Duration::from_secs(300),
                initial_cred,
                expires_at,
            ),
        );

        // Spawn 20 concurrent tasks that all try to get credentials at the same time
        // Since the initial credential is expired, they'll all try to refresh
        let mut handles = vec![];
        for i in 0..20 {
            let provider = provider.clone();
            let handle = tokio::spawn(async move {
                let cred = provider.get_credential().await.unwrap();
                // All should get the new credentials (AKID_1 from first fetch)
                assert_eq!(cred.key_id, "AKID_1");
                assert_eq!(cred.secret_key, "SECRET_1");
                assert_eq!(cred.token, Some("TOKEN_1".to_string()));
                i
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // Verify all 20 tasks completed successfully
        assert_eq!(results.len(), 20);

        // The provider should have been called at least once, but possibly more times
        // due to the try_write mechanism and race conditions
        let call_count = mock.get_call_count().await;
        assert!(
            call_count >= 1,
            "Provider should be called at least once, was called {} times",
            call_count
        );

        // It shouldn't be called 20 times though - the lock should prevent most concurrent fetches
        assert!(
            call_count < 10,
            "Provider should not be called too many times due to lock contention, was called {} times",
            call_count
        );
    }
}
