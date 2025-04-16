// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::HashMap,
    str::FromStr,
    sync::Arc,
    time::{Duration, SystemTime},
};

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
    ObjectStore, ObjectStoreParams, ObjectStoreProvider, StorageOptions, DEFAULT_CLOUD_BLOCK_SIZE,
    DEFAULT_CLOUD_IO_PARALLELISM,
};
use lance_core::error::{Error, Result};

#[derive(Default, Debug)]
pub struct AwsStoreProvider;

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
        let download_retry_count = storage_options.download_retry_count();

        let max_retries = storage_options.client_max_retries();
        let retry_timeout = storage_options.client_retry_timeout();
        let retry_config = RetryConfig {
            backoff: Default::default(),
            max_retries,
            retry_timeout: Duration::from_secs(retry_timeout),
        };

        storage_options.with_env_s3();

        let mut storage_options = storage_options.as_s3_options();
        let region = resolve_s3_region(&base_path, &storage_options).await?;
        let (aws_creds, region) = build_aws_credential(
            params.s3_credentials_refresh_offset,
            params.aws_credentials.clone(),
            Some(&storage_options),
            region,
        )
        .await?;

        // This will be default in next version of object store.
        // https://github.com/apache/arrow-rs/pull/7181
        // We can do this when we upgrade to 0.12.
        storage_options
            .entry(AmazonS3ConfigKey::ConditionalPut)
            .or_insert_with(|| "etag".to_string());

        // Cloudflare does not support varying part sizes.
        let use_constant_size_upload_parts = storage_options
            .get(&AmazonS3ConfigKey::Endpoint)
            .map(|endpoint| endpoint.contains("r2.cloudflarestorage.com"))
            .unwrap_or(false);

        // before creating the OSObjectStore we need to rewrite the url to drop ddb related parts
        base_path.set_scheme("s3").unwrap();
        base_path.set_query(None);

        // we can't use parse_url_opts here because we need to manually set the credentials provider
        let mut builder = AmazonS3Builder::new();
        for (key, value) in storage_options {
            builder = builder.with_config(key, value);
        }
        builder = builder
            .with_url(base_path.as_ref())
            .with_credentials(aws_creds)
            .with_retry(retry_config)
            .with_region(region);
        let inner = Arc::new(builder.build()?);

        Ok(ObjectStore {
            inner,
            scheme: String::from(base_path.scheme()),
            block_size,
            use_constant_size_upload_parts,
            list_is_lexically_ordered: true,
            io_parallelism: DEFAULT_CLOUD_IO_PARALLELISM,
            download_retry_count,
        })
    }
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
/// 1. An explicit `credentials` provider
/// 2. Explicit credentials in storage_options (as in `aws_access_key_id`,
///    `aws_secret_access_key`, `aws_session_token`)
/// 3. The default credential provider chain from AWS SDK.
///
/// `credentials_refresh_offset` is the amount of time before expiry to refresh credentials.
pub async fn build_aws_credential(
    credentials_refresh_offset: Duration,
    credentials: Option<AwsCredentialProvider>,
    storage_options: Option<&HashMap<AmazonS3ConfigKey, String>>,
    region: Option<String>,
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

    if let Some(creds) = credentials {
        Ok((creds, region))
    } else if let Some(creds) = storage_options.and_then(extract_static_s3_credentials) {
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

fn extract_static_s3_credentials(
    options: &HashMap<AmazonS3ConfigKey, String>,
) -> Option<StaticCredentialProvider<ObjectStoreAwsCredential>> {
    let key_id = options
        .get(&AmazonS3ConfigKey::AccessKeyId)
        .map(|s| s.to_string());
    let secret_key = options
        .get(&AmazonS3ConfigKey::SecretAccessKey)
        .map(|s| s.to_string());
    let token = options
        .get(&AmazonS3ConfigKey::Token)
        .map(|s| s.to_string());
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
            let refreshed_creds = Arc::new(self.inner.provide_credentials().await.map_err(
                |e| Error::Internal {
                    message: format!("Failed to get AWS credentials: {}", e),
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use object_store::path::Path;

    use crate::object_store::ObjectStoreRegistry;

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
}
