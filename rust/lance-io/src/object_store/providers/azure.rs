// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, str::FromStr, sync::Arc, time::Duration};

use object_store::ObjectStore as OSObjectStore;
use object_store_opendal::OpendalStore;
use opendal::{services::Azblob, Operator};
use snafu::location;

use object_store::{
    azure::{AzureConfigKey, MicrosoftAzureBuilder},
    RetryConfig,
};
use url::Url;

use crate::object_store::{
    ObjectStore, ObjectStoreParams, ObjectStoreProvider, StorageOptions, DEFAULT_CLOUD_BLOCK_SIZE,
    DEFAULT_CLOUD_IO_PARALLELISM, DEFAULT_MAX_IOP_SIZE,
};
use lance_core::error::{Error, Result};

#[derive(Default, Debug)]
pub struct AzureBlobStoreProvider;

impl AzureBlobStoreProvider {
    async fn build_opendal_azure_store(
        &self,
        base_path: &Url,
        storage_options: &StorageOptions,
    ) -> Result<Arc<dyn OSObjectStore>> {
        let container = base_path
            .host_str()
            .ok_or_else(|| {
                Error::invalid_input("Azure URL must contain container name", location!())
            })?
            .to_string();

        let prefix = base_path.path().trim_start_matches('/').to_string();

        // Start with all storage options as the config map
        // OpenDAL will handle environment variables through its default credentials chain
        let mut config_map: HashMap<String, String> = storage_options.0.clone();

        // Set required OpenDAL configuration
        config_map.insert("container".to_string(), container);

        if !prefix.is_empty() {
            config_map.insert("root".to_string(), format!("/{}", prefix));
        }

        let operator = Operator::from_iter::<Azblob>(config_map)
            .map_err(|e| {
                Error::invalid_input(
                    format!("Failed to create Azure Blob operator: {:?}", e),
                    location!(),
                )
            })?
            .finish();

        Ok(Arc::new(OpendalStore::new(operator)) as Arc<dyn OSObjectStore>)
    }

    async fn build_microsoft_azure_store(
        &self,
        base_path: &Url,
        storage_options: &StorageOptions,
    ) -> Result<Arc<dyn OSObjectStore>> {
        let max_retries = storage_options.client_max_retries();
        let retry_timeout = storage_options.client_retry_timeout();
        let retry_config = RetryConfig {
            backoff: Default::default(),
            max_retries,
            retry_timeout: Duration::from_secs(retry_timeout),
        };

        let mut builder = MicrosoftAzureBuilder::new()
            .with_url(base_path.as_ref())
            .with_retry(retry_config);
        for (key, value) in storage_options.as_azure_options() {
            builder = builder.with_config(key, value);
        }

        Ok(Arc::new(builder.build()?) as Arc<dyn OSObjectStore>)
    }
}

#[async_trait::async_trait]
impl ObjectStoreProvider for AzureBlobStoreProvider {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore> {
        let block_size = params.block_size.unwrap_or(DEFAULT_CLOUD_BLOCK_SIZE);
        let mut storage_options =
            StorageOptions(params.storage_options.clone().unwrap_or_default());
        storage_options.with_env_azure();
        let download_retry_count = storage_options.download_retry_count();

        let use_opendal = storage_options
            .0
            .get("use_opendal")
            .map(|v| v.as_str() == "true")
            .unwrap_or(false);

        let inner = if use_opendal {
            self.build_opendal_azure_store(&base_path, &storage_options)
                .await?
        } else {
            self.build_microsoft_azure_store(&base_path, &storage_options)
                .await?
        };

        Ok(ObjectStore {
            inner,
            scheme: String::from("az"),
            block_size,
            max_iop_size: *DEFAULT_MAX_IOP_SIZE,
            use_constant_size_upload_parts: false,
            list_is_lexically_ordered: true,
            io_parallelism: DEFAULT_CLOUD_IO_PARALLELISM,
            download_retry_count,
        })
    }
}

impl StorageOptions {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object_store::ObjectStoreParams;
    use std::collections::HashMap;

    #[test]
    fn test_azure_store_path() {
        let provider = AzureBlobStoreProvider;

        let url = Url::parse("az://bucket/path/to/file").unwrap();
        let path = provider.extract_path(&url).unwrap();
        let expected_path = object_store::path::Path::from("path/to/file");
        assert_eq!(path, expected_path);
    }

    #[tokio::test]
    async fn test_use_opendal_flag() {
        let provider = AzureBlobStoreProvider;
        let url = Url::parse("az://test-container/path").unwrap();
        let params_with_flag = ObjectStoreParams {
            storage_options: Some(HashMap::from([
                ("use_opendal".to_string(), "true".to_string()),
                ("account_name".to_string(), "test_account".to_string()),
                (
                    "endpoint".to_string(),
                    "https://test_account.blob.core.windows.net".to_string(),
                ),
                ("account_key".to_string(), "12345=".to_string()),
            ])),
            ..Default::default()
        };

        let store = provider
            .new_store(url.clone(), &params_with_flag)
            .await
            .unwrap();
        assert_eq!(store.scheme, "az");
    }
}
