// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, str::FromStr, sync::Arc, time::Duration};

use object_store::{
    gcp::{GcpCredential, GoogleCloudStorageBuilder, GoogleConfigKey},
    RetryConfig, StaticCredentialProvider,
};
use url::Url;

use crate::object_store::{
    ObjectStore, ObjectStoreParams, ObjectStoreProvider, StorageOptions, DEFAULT_CLOUD_BLOCK_SIZE,
    DEFAULT_CLOUD_IO_PARALLELISM,
};
use lance_core::error::Result;

#[derive(Default, Debug)]
pub struct GcsStoreProvider;

#[async_trait::async_trait]
impl ObjectStoreProvider for GcsStoreProvider {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore> {
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

        storage_options.with_env_gcs();
        let mut builder = GoogleCloudStorageBuilder::new()
            .with_url(base_path.as_ref())
            .with_retry(retry_config);
        for (key, value) in storage_options.as_gcs_options() {
            builder = builder.with_config(key, value);
        }
        let token_key = "google_storage_token";
        if let Some(storage_token) = storage_options.get(token_key) {
            let credential = GcpCredential {
                bearer: storage_token.to_string(),
            };
            let credential_provider = Arc::new(StaticCredentialProvider::new(credential)) as _;
            builder = builder.with_credentials(credential_provider);
        }
        let inner = Arc::new(builder.build()?);

        Ok(ObjectStore {
            inner,
            scheme: String::from("gs"),
            block_size,
            use_constant_size_upload_parts: false,
            list_is_lexically_ordered: true,
            io_parallelism: DEFAULT_CLOUD_IO_PARALLELISM,
            download_retry_count,
        })
    }
}

impl StorageOptions {
    /// Add values from the environment to storage options
    pub fn with_env_gcs(&mut self) {
        for (os_key, os_value) in std::env::vars_os() {
            if let (Some(key), Some(value)) = (os_key.to_str(), os_value.to_str()) {
                let lowercase_key = key.to_ascii_lowercase();
                let token_key = "google_storage_token";

                if let Ok(config_key) = GoogleConfigKey::from_str(&lowercase_key) {
                    if !self.0.contains_key(config_key.as_ref()) {
                        self.0
                            .insert(config_key.as_ref().to_string(), value.to_string());
                    }
                }
                // Check for GOOGLE_STORAGE_TOKEN until GoogleConfigKey supports storage token
                else if lowercase_key == token_key && !self.0.contains_key(token_key) {
                    self.0.insert(token_key.to_string(), value.to_string());
                }
            }
        }
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
