// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, str::FromStr, sync::Arc, time::Duration};

#[cfg(test)]
use mock_instant::thread_local::{SystemTime, UNIX_EPOCH};

#[cfg(not(test))]
use std::time::{SystemTime, UNIX_EPOCH};

use object_store::ObjectStore as OSObjectStore;
use object_store_opendal::OpendalStore;
use opendal::{Operator, services::S3};

use aws_config::default_provider::credentials::DefaultCredentialsChain;
use aws_config::web_identity_token::{StaticConfiguration, WebIdentityTokenCredentialsProvider};
use aws_credential_types::provider::ProvideCredentials;
use object_store::{
    ClientOptions, CredentialProvider, Result as ObjectStoreResult, RetryConfig,
    StaticCredentialProvider,
    aws::{
        AmazonS3Builder, AmazonS3ConfigKey, AwsCredential as ObjectStoreAwsCredential,
        AwsCredentialProvider,
    },
};
use serde::Deserialize;
use tokio::sync::RwLock;
use url::Url;

use crate::object_store::{
    DEFAULT_CLOUD_BLOCK_SIZE, DEFAULT_CLOUD_IO_PARALLELISM, DEFAULT_MAX_IOP_SIZE, ObjectStore,
    ObjectStoreParams, ObjectStoreProvider, StorageOptions, StorageOptionsAccessor,
    dynamic_credentials::{NamespaceCredentialsProvider, build_dynamic_credential_provider},
    throttle::{AimdThrottleConfig, AimdThrottledStore},
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
        // Use a low retry count since the AIMD throttle layer handles
        // throttle recovery with its own retry loop.
        let retry_config = RetryConfig {
            backoff: Default::default(),
            max_retries: storage_options.client_max_retries(),
            retry_timeout: Duration::from_secs(storage_options.client_retry_timeout()),
        };

        let mut s3_storage_options = storage_options.as_s3_options();
        let region = resolve_s3_region(base_path, &s3_storage_options).await?;

        // Get accessor from params
        let accessor = params.get_accessor();

        let (aws_creds, region) = build_aws_credential(
            params.s3_credentials_refresh_offset,
            params.aws_credentials.clone(),
            Some(&s3_storage_options),
            region,
            accessor,
        )
        .await?;

        // Set S3Express flag if detected
        if is_s3_express {
            s3_storage_options.insert(AmazonS3ConfigKey::S3Express, true.to_string());
        }

        // Compute the metrics label before rewriting the url below, so it
        // matches the prefix the registry uses to key this store.
        #[cfg(feature = "metrics")]
        let store_prefix =
            self.calculate_object_store_prefix(base_path, Some(&storage_options.0))?;

        // before creating the OSObjectStore we need to rewrite the url to drop ddb related parts
        base_path.set_scheme("s3").unwrap();
        base_path.set_query(None);

        // we can't use parse_url_opts here because we need to manually set the credentials provider
        let mut builder =
            AmazonS3Builder::new().with_client_options(storage_options.client_options()?);
        for (key, value) in s3_storage_options {
            builder = builder.with_config(key, value);
        }
        builder = builder
            .with_url(base_path.as_ref())
            .with_credentials(aws_creds)
            .with_retry(retry_config)
            .with_region(region);

        #[cfg(feature = "metrics")]
        {
            builder = builder.with_http_connector(
                crate::object_store::metrics::MeteringHttpConnector::new(store_prefix),
            );
        }

        Ok(Arc::new(builder.build()?) as Arc<dyn OSObjectStore>)
    }

    async fn build_opendal_s3_store(
        &self,
        base_path: &Url,
        storage_options: &StorageOptions,
    ) -> Result<Arc<dyn OSObjectStore>> {
        let bucket = base_path
            .host_str()
            .ok_or_else(|| Error::invalid_input("S3 URL must contain bucket name"))?
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
            .map_err(|e| Error::invalid_input(format!("Failed to create S3 operator: {:?}", e)))?;

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
            StorageOptions::new(params.storage_options().cloned().unwrap_or_default());
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
        let throttle_config = AimdThrottleConfig::from_storage_options(params.storage_options())?;
        let inner = if throttle_config.is_disabled() {
            inner
        } else {
            Arc::new(AimdThrottledStore::new(inner, throttle_config)?) as Arc<dyn OSObjectStore>
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
            io_tracker: Default::default(),
            store_prefix: self
                .calculate_object_store_prefix(&base_path, params.storage_options())?,
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
            Error::invalid_input(format!("Could not parse bucket from url: {}", url))
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
/// 1. An explicit `storage_options_accessor` with a provider
/// 2. An explicit `credentials` provider
/// 3. Explicit credentials in storage_options (as in `aws_access_key_id`,
///    `aws_secret_access_key`, `aws_session_token`)
/// 4. IRSA (web identity token) if `storage_options` contains both
///    `aws_web_identity_token_file` and `aws_role_arn`
/// 5. ECS container credentials if `storage_options` contains
///    `aws_container_credentials_full_uri` or `aws_container_credentials_relative_uri`
/// 6. The default credential provider chain from AWS SDK.
///
/// # Storage Options Accessor
///
/// When `storage_options_accessor` is provided and has a dynamic provider,
/// credentials are fetched and cached by the accessor with automatic refresh
/// before expiration.
///
/// `credentials_refresh_offset` is the amount of time before expiry to refresh credentials.
pub async fn build_aws_credential(
    credentials_refresh_offset: Duration,
    credentials: Option<AwsCredentialProvider>,
    storage_options: Option<&HashMap<AmazonS3ConfigKey, String>>,
    region: Option<String>,
    storage_options_accessor: Option<Arc<StorageOptionsAccessor>>,
) -> Result<(AwsCredentialProvider, String)> {
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

    // If the user supplied their own credential provider that takes top priority
    if let Some(creds) = credentials {
        return Ok((creds, region));
    }

    // Otherwise, if the user provided a storage_options_accessor, try and use that
    if let Some(dynamic_creds) = build_dynamic_credential_provider::<ObjectStoreAwsCredential>(
        storage_options_accessor.clone(),
    )
    .await?
    {
        return Ok((dynamic_creds, region));
    }

    // If the user provided a storage_options_accessor, then it must not have matched AWS.
    // Log a message and ignore it.
    if storage_options_accessor
        .as_ref()
        .is_some_and(|a| a.has_provider())
    {
        log::debug!(
            "Storage options from provider do not contain explicit AWS credentials, \
             falling back to default AWS credentials chain."
        );
    }

    if let Some(opts) = storage_options {
        // Check for static credentials (access key & secret)
        if let Some(creds) = extract_static_s3_credentials(opts) {
            return Ok((Arc::new(creds), region));
        }

        // Check for explicitly provided IRSA (web identity token) credentials.
        if let (Some(token_file), Some(role_arn)) = (
            opts.get(&AmazonS3ConfigKey::WebIdentityTokenFile),
            opts.get(&AmazonS3ConfigKey::RoleArn),
        ) {
            let session_name = opts.get(&AmazonS3ConfigKey::RoleSessionName).cloned();
            let provider = build_irsa_provider(token_file, role_arn, session_name);
            return Ok((
                Arc::new(AwsCredentialAdapter::new(
                    Arc::new(provider),
                    credentials_refresh_offset,
                )),
                region,
            ));
        }

        // Check for explicitly provided ECS container credentials.
        if opts.contains_key(&AmazonS3ConfigKey::ContainerCredentialsFullUri)
            || opts.contains_key(&AmazonS3ConfigKey::ContainerCredentialsRelativeUri)
        {
            let full_uri = opts
                .get(&AmazonS3ConfigKey::ContainerCredentialsFullUri)
                .cloned();
            let relative_uri = opts
                .get(&AmazonS3ConfigKey::ContainerCredentialsRelativeUri)
                .cloned();
            let auth_token_file = opts
                .get(&AmazonS3ConfigKey::ContainerAuthorizationTokenFile)
                .cloned();

            let provider = EcsCredentialProvider::new(full_uri, relative_uri, auth_token_file)?;
            return Ok((
                Arc::new(AwsCredentialAdapter::new(
                    Arc::new(provider),
                    credentials_refresh_offset,
                )),
                region,
            ));
        }
    }

    let credentials_provider = DefaultCredentialsChain::builder().build().await;
    Ok((
        Arc::new(AwsCredentialAdapter::new(
            Arc::new(credentials_provider),
            credentials_refresh_offset,
        )),
        region,
    ))
}

/// Build an IRSA (web identity token) credentials provider from explicit configuration.
fn build_irsa_provider(
    token_file: &str,
    role_arn: &str,
    session_name: Option<String>,
) -> WebIdentityTokenCredentialsProvider {
    use std::path::PathBuf;
    let session_name = session_name.unwrap_or_else(|| "lance-session".to_string());
    WebIdentityTokenCredentialsProvider::builder()
        .static_configuration(StaticConfiguration {
            web_identity_token_file: PathBuf::from(token_file),
            role_arn: role_arn.to_string(),
            session_name,
        })
        .build()
}

/// Custom ECS/Pod identity credential provider that fetches credentials from a
/// container credential endpoint. Used when explicit URI storage options are set.
///
/// We cannot use aws_config::ecs::EcsCredentialProvider because it does not offer
/// any way to set the variables programmatically.  It _always_ pulls them from
/// environment variables.
#[derive(Debug)]
struct EcsCredentialProvider {
    uri: String,
    auth_token_file: Option<String>,
}

impl EcsCredentialProvider {
    fn new(
        full_uri: Option<String>,
        relative_uri: Option<String>,
        auth_token_file: Option<String>,
    ) -> Result<Self> {
        const ECS_CONTAINER_HOST: &str = "http://169.254.170.2";
        let uri = if let Some(full_uri) = full_uri {
            full_uri
        } else if let Some(relative_uri) = relative_uri {
            format!("{}{}", ECS_CONTAINER_HOST, relative_uri)
        } else {
            return Err(Error::invalid_input(
                "ECS credential provider requires at least one of \
                 aws_container_credentials_full_uri or aws_container_credentials_relative_uri",
            ));
        };
        Ok(Self {
            uri,
            auth_token_file,
        })
    }

    async fn fetch(&self) -> aws_credential_types::provider::Result {
        use aws_credential_types::provider::error::CredentialsError;

        #[derive(Deserialize)]
        struct EcsCredentialResponse {
            #[serde(rename = "AccessKeyId")]
            access_key_id: String,
            #[serde(rename = "SecretAccessKey")]
            secret_access_key: String,
            #[serde(rename = "Token")]
            token: Option<String>,
            #[serde(rename = "Expiration")]
            expiration: Option<String>,
        }

        let token = if let Some(file) = &self.auth_token_file {
            Some(
                tokio::fs::read_to_string(file)
                    .await
                    .map_err(|e| {
                        CredentialsError::provider_error(format!(
                            "Failed to read ECS auth token file '{}': {}",
                            file, e
                        ))
                    })?
                    .trim()
                    .to_string(),
            )
        } else {
            None
        };

        let client = reqwest::Client::new();
        let mut request = client.get(&self.uri);
        if let Some(token) = token {
            request = request.header("Authorization", token);
        }

        let response = request.send().await.map_err(|e| {
            CredentialsError::provider_error(format!(
                "Failed to fetch ECS credentials from '{}': {}",
                self.uri, e
            ))
        })?;

        let creds: EcsCredentialResponse = response.json().await.map_err(|e| {
            CredentialsError::provider_error(format!(
                "Failed to parse ECS credentials response from '{}': {}",
                self.uri, e
            ))
        })?;

        let expiry = creds.expiration.and_then(|e| {
            chrono::DateTime::parse_from_rfc3339(&e)
                .ok()
                .map(std::time::SystemTime::from)
        });

        Ok(aws_credential_types::Credentials::new(
            creds.access_key_id,
            creds.secret_access_key,
            creds.token,
            expiry,
            "EcsCredentialProvider",
        ))
    }
}

impl ProvideCredentials for EcsCredentialProvider {
    fn provide_credentials<'a>(
        &'a self,
    ) -> aws_credential_types::provider::future::ProvideCredentials<'a>
    where
        Self: 'a,
    {
        aws_credential_types::provider::future::ProvideCredentials::new(self.fetch())
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
            if expired { None } else { cache_value.clone() }
        };

        if let Some(creds) = cached_creds {
            Ok(Arc::new(Self::Credential {
                key_id: creds.access_key_id().to_string(),
                secret_key: creds.secret_access_key().to_string(),
                token: creds.session_token().map(|s| s.to_string()),
            }))
        } else {
            let refreshed_creds =
                Arc::new(self.inner.provide_credentials().await.map_err(|e| {
                    Error::internal(format!("Failed to get AWS credentials: {:?}", e))
                })?);

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
    /// Add values from the environment to storage options.
    ///
    /// Only adds keys that are not already present, so explicitly-set options
    /// (including empty-string sentinels) always take precedence over env vars.
    pub fn with_env_s3(&mut self) {
        for (os_key, os_value) in std::env::vars_os() {
            if let (Some(key), Some(value)) = (os_key.to_str(), os_value.to_str())
                && let Ok(config_key) = AmazonS3ConfigKey::from_str(&key.to_ascii_lowercase())
                && !self.0.contains_key(config_key.as_ref())
            {
                self.0
                    .insert(config_key.as_ref().to_string(), value.to_string());
            }
        }
    }

    /// Subset of options relevant for s3 storage.
    ///
    /// Empty-string values are excluded: setting a key to `""` is the way to
    /// suppress an env-var-injected value (the empty string blocks injection via
    /// `with_env_s3`, and this filter ensures it doesn't reach the builder or
    /// credential logic either).
    pub fn as_s3_options(&self) -> HashMap<AmazonS3ConfigKey, String> {
        self.0
            .iter()
            .filter_map(|(key, value)| {
                if value.is_empty() {
                    return None;
                }
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
        let storage_options_accessor = region.map(|region| {
            let opts: HashMap<String, String> =
                [("region".into(), region)].iter().cloned().collect();
            Arc::new(StorageOptionsAccessor::with_static_options(opts))
        });
        Self {
            aws_credentials,
            storage_options_accessor,
            ..Default::default()
        }
    }
}

pub type DynamicStorageOptionsCredentialProvider =
    NamespaceCredentialsProvider<ObjectStoreAwsCredential>;

#[cfg(test)]
mod tests {
    use crate::object_store::ObjectStoreRegistry;
    use crate::object_store::StorageOptionsProvider;
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
            // for non ASCII string tests: the URL encodes them, extract_path must decode back
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
            // extract_path decodes url.path(), so the Path stores the raw (decoded)
            // string. Path::parse keeps its input verbatim, matching that, whereas
            // Path::from would percent-encode non-ASCII bytes and not match.
            let expected_path = Path::parse(expected_path).unwrap();
            assert_eq!(path, expected_path)
        }
    }

    // Regression test for https://github.com/lance-format/lance/issues/6643
    // extract_path must NOT double-encode paths that contain non-ASCII characters.
    // url.path() returns a percent-encoded string; we must decode it back to raw
    // UTF-8 before storing it in a Path, so the object store HTTP client can apply
    // a single, correct percent-encoding when building the request URL.
    #[test]
    fn test_s3_non_ascii_path_no_double_encoding() {
        let provider = AwsStoreProvider;

        // "s3://bucket/中文路径" → url.path() == "/%E4%B8%AD%E6%96%87%E8%B7%AF%E5%BE%84".
        // The buggy Path::parse(url.path()) stored "%E4%B8%AD..." verbatim; the S3
        // client then percent-encodes the '%' again, yielding "%25E4%25B8%25AD...".
        // With Path::from_url_path the Path stores the decoded UTF-8 instead.
        let url = Url::parse("s3://bucket/中文路径").unwrap();
        let path = provider.extract_path(&url).unwrap();

        // The Path must hold the decoded UTF-8, not the percent-encoded form.
        assert_eq!(path.as_ref(), "中文路径");
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
        use crate::object_store::StorageOptionsAccessor;
        let provider = AwsStoreProvider;
        let url = Url::parse("s3://test-bucket/path").unwrap();
        let params_with_flag = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([
                    ("use_opendal".to_string(), "true".to_string()),
                    ("region".to_string(), "us-west-2".to_string()),
                ]),
            ))),
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

        // Create initial options with cached credentials that expire in 10 minutes
        let expires_at = now_ms + 600_000; // 10 minutes from now
        let initial_options = HashMap::from([
            ("aws_access_key_id".to_string(), "AKID_CACHED".to_string()),
            (
                "aws_secret_access_key".to_string(),
                "SECRET_CACHED".to_string(),
            ),
            ("aws_session_token".to_string(), "TOKEN_CACHED".to_string()),
            ("expires_at_millis".to_string(), expires_at.to_string()),
            ("refresh_offset_millis".to_string(), "300000".to_string()), // 5 minute refresh offset
        ]);

        let provider = DynamicStorageOptionsCredentialProvider::from_provider_with_initial(
            mock.clone(),
            initial_options,
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

        // Create initial options with credentials that expired 1 second ago
        let expired_time = now_ms - 1_000; // 1 second ago
        let initial_options = HashMap::from([
            ("aws_access_key_id".to_string(), "AKID_EXPIRED".to_string()),
            (
                "aws_secret_access_key".to_string(),
                "SECRET_EXPIRED".to_string(),
            ),
            ("expires_at_millis".to_string(), expired_time.to_string()),
            ("refresh_offset_millis".to_string(), "300000".to_string()), // 5 minute refresh offset
        ]);

        let provider = DynamicStorageOptionsCredentialProvider::from_provider_with_initial(
            mock.clone(),
            initial_options,
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

        // Create a mock provider that returns credentials expiring in 30 seconds
        let mock = Arc::new(MockStorageOptionsProvider::new(Some(
            30_000, // Expires in 30 seconds
        )));

        // Create credential provider with default 60 second refresh offset
        // This means credentials should be refreshed when they have less than 60 seconds left
        let provider = DynamicStorageOptionsCredentialProvider::from_provider(mock.clone());

        // First call should fetch credentials from provider (no initial cache)
        // Credentials expire in 30 seconds, which is less than our 60 second refresh offset,
        // so they should be considered "needs refresh" immediately
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_1");
        assert_eq!(mock.get_call_count().await, 1);

        // Second call should trigger refresh because credentials expire in 30 seconds
        // but our refresh lead time is 60 seconds (now + 60sec > expires_at)
        // The mock will return new credentials (AKID_2) with the same expiration
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_2");
        assert_eq!(mock.get_call_count().await, 2);
    }

    #[tokio::test]
    async fn test_dynamic_credential_provider_no_initial_cache() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        // Create a mock provider that returns credentials expiring in 2 minutes
        let mock = Arc::new(MockStorageOptionsProvider::new(Some(
            120_000, // Expires in 2 minutes
        )));

        // Create credential provider without initial cache, using default 60 second refresh offset
        let provider = DynamicStorageOptionsCredentialProvider::from_provider(mock.clone());

        // First call should fetch from provider (call count = 1)
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_1");
        assert_eq!(cred.secret_key, "SECRET_1");
        assert_eq!(cred.token, Some("TOKEN_1".to_string()));
        assert_eq!(mock.get_call_count().await, 1);

        // Second call should use cached credentials (not expired yet, still > 60 seconds remaining)
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_1");
        assert_eq!(mock.get_call_count().await, 1); // Still 1, didn't fetch again

        // Advance time to 90 seconds - should trigger refresh (within 60 sec refresh offset)
        // At this point, credentials expire in 30 seconds (< 60 sec offset)
        MockClock::set_system_time(Duration::from_secs(100_000 + 90));
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_2");
        assert_eq!(cred.secret_key, "SECRET_2");
        assert_eq!(cred.token, Some("TOKEN_2".to_string()));
        assert_eq!(mock.get_call_count().await, 2);

        // Advance time to 210 seconds total (90 + 120) - should trigger another refresh
        MockClock::set_system_time(Duration::from_secs(100_000 + 210));
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_3");
        assert_eq!(cred.secret_key, "SECRET_3");
        assert_eq!(mock.get_call_count().await, 3);
    }

    #[tokio::test]
    async fn test_dynamic_credential_provider_with_initial_options() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        let now_ms = MockClock::system_time().as_millis() as u64;

        // Create a mock provider that returns credentials expiring in 10 minutes
        let mock = Arc::new(MockStorageOptionsProvider::new(Some(
            600_000, // Expires in 10 minutes
        )));

        // Create initial options with expiration in 10 minutes
        let expires_at = now_ms + 600_000; // 10 minutes from now
        let initial_options = HashMap::from([
            ("aws_access_key_id".to_string(), "AKID_INITIAL".to_string()),
            (
                "aws_secret_access_key".to_string(),
                "SECRET_INITIAL".to_string(),
            ),
            ("aws_session_token".to_string(), "TOKEN_INITIAL".to_string()),
            ("expires_at_millis".to_string(), expires_at.to_string()),
            ("refresh_offset_millis".to_string(), "300000".to_string()), // 5 minute refresh offset
        ]);

        // Create credential provider with initial options
        let provider = DynamicStorageOptionsCredentialProvider::from_provider_with_initial(
            mock.clone(),
            initial_options,
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

        let provider = Arc::new(DynamicStorageOptionsCredentialProvider::from_provider(
            mock.clone(),
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

        // Create initial options with credentials that expired in the past (1000 seconds ago)
        let expires_at = now_ms - 1_000_000;
        let initial_options = HashMap::from([
            ("aws_access_key_id".to_string(), "AKID_OLD".to_string()),
            (
                "aws_secret_access_key".to_string(),
                "SECRET_OLD".to_string(),
            ),
            ("aws_session_token".to_string(), "TOKEN_OLD".to_string()),
            ("expires_at_millis".to_string(), expires_at.to_string()),
            ("refresh_offset_millis".to_string(), "300000".to_string()), // 5 minute refresh offset
        ]);

        // Mock will return credentials expiring in 1 hour
        let mock = Arc::new(MockStorageOptionsProvider::new(Some(
            3_600_000, // Expires in 1 hour
        )));

        let provider = Arc::new(
            DynamicStorageOptionsCredentialProvider::from_provider_with_initial(
                mock.clone(),
                initial_options,
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

    #[tokio::test]
    async fn test_explicit_aws_credentials_takes_precedence_over_accessor() {
        // Create a mock storage options provider that should NOT be called
        let mock_storage_provider = Arc::new(MockStorageOptionsProvider::new(Some(600_000)));

        // Create an accessor with the mock provider
        let accessor = Arc::new(StorageOptionsAccessor::with_provider(
            mock_storage_provider.clone(),
        ));

        // Create an explicit AWS credentials provider
        let explicit_cred_provider = Arc::new(MockAwsCredentialsProvider::default());

        // Build credentials with both aws_credentials AND accessor
        // The explicit aws_credentials should take precedence
        let (result, _region) = build_aws_credential(
            Duration::from_secs(300),
            Some(explicit_cred_provider.clone() as AwsCredentialProvider),
            None, // no storage_options
            Some("us-west-2".to_string()),
            Some(accessor),
        )
        .await
        .unwrap();

        // Get credential from the result
        let cred = result.get_credential().await.unwrap();

        // The explicit provider should have been called (it returns empty strings)
        assert!(explicit_cred_provider.called.load(Ordering::Relaxed));

        // The storage options provider should NOT have been called
        assert_eq!(
            mock_storage_provider.get_call_count().await,
            0,
            "Storage options provider should not be called when explicit aws_credentials is provided"
        );

        // Verify we got credentials from the explicit provider (empty strings)
        assert_eq!(cred.key_id, "");
        assert_eq!(cred.secret_key, "");
    }

    #[tokio::test]
    async fn test_accessor_used_when_no_explicit_aws_credentials() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        let now_ms = MockClock::system_time().as_millis() as u64;

        // Create a mock storage options provider
        let mock_storage_provider = Arc::new(MockStorageOptionsProvider::new(Some(600_000)));

        // Create initial options
        let expires_at = now_ms + 600_000; // 10 minutes from now
        let initial_options = HashMap::from([
            (
                "aws_access_key_id".to_string(),
                "AKID_FROM_ACCESSOR".to_string(),
            ),
            (
                "aws_secret_access_key".to_string(),
                "SECRET_FROM_ACCESSOR".to_string(),
            ),
            (
                "aws_session_token".to_string(),
                "TOKEN_FROM_ACCESSOR".to_string(),
            ),
            ("expires_at_millis".to_string(), expires_at.to_string()),
            ("refresh_offset_millis".to_string(), "300000".to_string()), // 5 minute refresh offset
        ]);

        // Create an accessor with initial options and provider
        let accessor = Arc::new(StorageOptionsAccessor::with_initial_and_provider(
            initial_options,
            mock_storage_provider.clone(),
        ));

        // Build credentials with accessor but NO explicit aws_credentials
        let (result, _region) = build_aws_credential(
            Duration::from_secs(300),
            None, // no explicit aws_credentials
            None, // no storage_options
            Some("us-west-2".to_string()),
            Some(accessor),
        )
        .await
        .unwrap();

        // Get credential - should use the initial accessor credentials
        let cred = result.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_FROM_ACCESSOR");
        assert_eq!(cred.secret_key, "SECRET_FROM_ACCESSOR");

        // Storage options provider should NOT have been called yet (using cached initial creds)
        assert_eq!(mock_storage_provider.get_call_count().await, 0);

        // Advance time to trigger refresh (past the 5 minute refresh offset)
        MockClock::set_system_time(Duration::from_secs(100_000 + 360));

        // Get credential again - should now fetch from provider
        let cred = result.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID_1");
        assert_eq!(cred.secret_key, "SECRET_1");

        // Storage options provider should have been called once
        assert_eq!(mock_storage_provider.get_call_count().await, 1);
    }

    // Test that explicitly providing IRSA keys selects the IRSA provider.
    // Provider construction succeeds immediately; the STS credential fetch happens lazily.
    #[tokio::test]
    async fn test_irsa_selected_when_explicit_keys_provided() {
        use tempfile::NamedTempFile;
        let token_file = NamedTempFile::new().unwrap();
        let token_path = token_file.path().to_str().unwrap().to_string();

        let opts = HashMap::from([
            (AmazonS3ConfigKey::WebIdentityTokenFile, token_path),
            (
                AmazonS3ConfigKey::RoleArn,
                "arn:aws:iam::123456789012:role/TestRole".to_string(),
            ),
        ]);

        let result = build_aws_credential(
            Duration::from_secs(300),
            None,
            Some(&opts),
            Some("us-east-1".to_string()),
            None,
        )
        .await;
        assert!(
            result.is_ok(),
            "IRSA provider should be built without error: {:?}",
            result.err()
        );
    }

    // Test that IRSA is NOT selected when the keys are absent (e.g., only present as env
    // vars, which with_env_s3() no longer injects for credential-selection keys).
    #[tokio::test]
    async fn test_irsa_not_selected_when_keys_absent() {
        let opts: HashMap<AmazonS3ConfigKey, String> = HashMap::new();

        let result = build_aws_credential(
            Duration::from_secs(300),
            None,
            Some(&opts),
            Some("us-east-1".to_string()),
            None,
        )
        .await;
        // Falls through to DefaultCredentialsChain without error
        assert!(result.is_ok());
    }

    // Test that ECS provider is selected when an explicit URI is provided.
    #[tokio::test]
    async fn test_ecs_selected_when_explicit_uri_provided() {
        let opts = HashMap::from([(
            AmazonS3ConfigKey::ContainerCredentialsFullUri,
            "http://169.254.170.2/credentials".to_string(),
        )]);

        let result = build_aws_credential(
            Duration::from_secs(300),
            None,
            Some(&opts),
            Some("us-east-1".to_string()),
            None,
        )
        .await;
        assert!(
            result.is_ok(),
            "ECS provider should be built without error: {:?}",
            result.err()
        );
    }

    // Test that IRSA takes precedence over ECS when both are explicitly provided.
    #[tokio::test]
    async fn test_irsa_takes_precedence_over_ecs() {
        use tempfile::NamedTempFile;
        let token_file = NamedTempFile::new().unwrap();
        let token_path = token_file.path().to_str().unwrap().to_string();

        let opts = HashMap::from([
            (AmazonS3ConfigKey::WebIdentityTokenFile, token_path),
            (
                AmazonS3ConfigKey::RoleArn,
                "arn:aws:iam::123456789012:role/TestRole".to_string(),
            ),
            (
                AmazonS3ConfigKey::ContainerCredentialsFullUri,
                "http://169.254.170.2/credentials".to_string(),
            ),
        ]);

        // Both IRSA and ECS set — IRSA should win (checked first in precedence order)
        let result = build_aws_credential(
            Duration::from_secs(300),
            None,
            Some(&opts),
            Some("us-east-1".to_string()),
            None,
        )
        .await;
        assert!(result.is_ok());
    }

    // Test that static access keys take precedence over IRSA.
    #[tokio::test]
    async fn test_static_keys_take_precedence_over_irsa() {
        use tempfile::NamedTempFile;
        let token_file = NamedTempFile::new().unwrap();
        let token_path = token_file.path().to_str().unwrap().to_string();

        let opts = HashMap::from([
            (AmazonS3ConfigKey::AccessKeyId, "AKID".to_string()),
            (AmazonS3ConfigKey::SecretAccessKey, "SECRET".to_string()),
            (AmazonS3ConfigKey::WebIdentityTokenFile, token_path),
            (
                AmazonS3ConfigKey::RoleArn,
                "arn:aws:iam::123456789012:role/TestRole".to_string(),
            ),
        ]);

        let (provider, _region) = build_aws_credential(
            Duration::from_secs(300),
            None,
            Some(&opts),
            Some("us-east-1".to_string()),
            None,
        )
        .await
        .unwrap();

        // Static creds should be used — key_id will be "AKID"
        let cred = provider.get_credential().await.unwrap();
        assert_eq!(cred.key_id, "AKID");
        assert_eq!(cred.secret_key, "SECRET");
    }

    // Test that setting credential-selection keys to "" in storage options blocks env-var
    // injection and that the empty sentinels are stripped by as_s3_options(), so the credential
    // logic sees only the explicitly-set non-empty keys (ECS URI in this case, not IRSA).
    #[tokio::test]
    async fn test_empty_string_masks_env_credential_keys() {
        use tempfile::NamedTempFile;

        // Simulate: IRSA env vars are present in the process env (e.g., pod has IRSA by default)
        // SAFETY: test-only env mutation; no other threads touch these vars.
        unsafe {
            std::env::set_var("AWS_WEB_IDENTITY_TOKEN_FILE", "/tmp/irsa-token");
            std::env::set_var("AWS_ROLE_ARN", "arn:aws:iam::123:role/IrsaRole");
        }

        // User explicitly wants ECS, not IRSA — set IRSA keys to "" to block injection
        let ecs_token_file = NamedTempFile::new().unwrap();
        let user_opts = HashMap::from([
            (
                "aws_container_credentials_full_uri".to_string(),
                "http://169.254.170.2/credentials".to_string(),
            ),
            (
                "aws_web_identity_token_file".to_string(),
                "".to_string(), // block IRSA env var
            ),
            (
                "aws_role_arn".to_string(),
                "".to_string(), // block IRSA env var
            ),
            (
                "aws_container_authorization_token_file".to_string(),
                ecs_token_file.path().to_str().unwrap().to_string(),
            ),
        ]);

        let mut storage_options = StorageOptions::new(user_opts);
        storage_options.with_env_s3();
        let s3_opts = storage_options.as_s3_options();

        // Restore env
        // SAFETY: test-only; restoring env vars set above.
        unsafe {
            std::env::remove_var("AWS_WEB_IDENTITY_TOKEN_FILE");
            std::env::remove_var("AWS_ROLE_ARN");
        }

        // IRSA keys must be absent — empty sentinels were stripped by as_s3_options()
        assert!(
            !s3_opts.contains_key(&AmazonS3ConfigKey::WebIdentityTokenFile),
            "IRSA token file should be absent after empty-string masking"
        );
        assert!(
            !s3_opts.contains_key(&AmazonS3ConfigKey::RoleArn),
            "IRSA role ARN should be absent after empty-string masking"
        );
        // ECS URI must be present
        assert!(
            s3_opts.contains_key(&AmazonS3ConfigKey::ContainerCredentialsFullUri),
            "ECS full URI should be present"
        );

        // build_aws_credential should select ECS, not IRSA
        let result = build_aws_credential(
            Duration::from_secs(300),
            None,
            Some(&s3_opts),
            Some("us-east-1".to_string()),
            None,
        )
        .await;
        assert!(
            result.is_ok(),
            "ECS provider should be selected and built without error: {:?}",
            result.err()
        );
    }
}
