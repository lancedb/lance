// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use object_store::ObjectStore as OSObjectStore;
use object_store_opendal::OpendalStore;
use opendal::{Operator, services::Cos};
use url::Url;

use crate::object_store::dynamic_opendal::DynamicOpenDalStore;
use crate::object_store::{
    DEFAULT_CLOUD_BLOCK_SIZE, DEFAULT_CLOUD_IO_PARALLELISM, DEFAULT_MAX_IOP_SIZE, ObjectStore,
    ObjectStoreParams, ObjectStoreProvider, StorageOptions,
};
use lance_core::error::{Error, Result};

#[derive(Default, Debug)]
pub struct TencentStoreProvider;

impl TencentStoreProvider {
    fn base_cos_options(
        base_path: &Url,
        storage_options: &StorageOptions,
    ) -> Result<HashMap<String, String>> {
        let bucket = base_path
            .host_str()
            .ok_or_else(|| Error::invalid_input("COS URL must contain bucket name"))?
            .to_string();

        let prefix = base_path.path().trim_start_matches('/').to_string();

        // Load COS/TENCENTCLOUD related config from environment variables as base
        let mut config_map: HashMap<String, String> = std::env::vars()
            .filter(|(key, _)| key.starts_with("COS_") || key.starts_with("TENCENTCLOUD_"))
            .map(|(key, value)| {
                let normalized_key = key
                    .to_lowercase()
                    .replace("cos_", "")
                    .replace("tencentcloud_", "");
                (normalized_key, value)
            })
            .collect();

        // Merge storage_options (user-provided options take priority over env vars)
        config_map.extend(storage_options.0.clone());

        config_map.insert("bucket".to_string(), bucket);
        if prefix.is_empty() {
            config_map.remove("root");
        } else {
            config_map.insert("root".to_string(), "/".to_string());
        }

        Ok(config_map)
    }

    /// Normalize COS config options by mapping alias keys to the standard keys
    /// required by the OpenDAL COS service, with support for temporary credentials (security_token).
    fn normalize_cos_config(options: &HashMap<String, String>) -> Result<HashMap<String, String>> {
        let mut config_map = options.clone();

        // Alias mapping: map user-friendly keys to OpenDAL COS standard keys
        let alias_groups: &[(&str, &[&str])] = &[
            ("endpoint", &["cos_endpoint"]),
            ("secret_id", &["cos_secret_id"]),
            ("secret_key", &["cos_secret_key"]),
            ("security_token", &["cos_security_token"]),
            ("bucket", &["cos_bucket"]),
            ("region", &["cos_region"]),
        ];

        for (canonical, aliases) in alias_groups {
            for alias in *aliases {
                if let Some(value) = config_map.remove(*alias) {
                    config_map.insert(canonical.to_string(), value);
                    break;
                }
            }
        }
        
        config_map
            .entry("disable_config_load".to_string())
            .or_insert_with(|| "false".to_string());

        if !config_map.contains_key("endpoint") {
            if let Some(region) = config_map
                .get("region")
                .filter(|v| !v.is_empty())
                .cloned()
            {
                let endpoint = format!("https://cos.{}.myqcloud.com", region);
                config_map.insert("endpoint".to_string(), endpoint);
            } else {
                return Err(Error::invalid_input(
                    "COS endpoint or region is required. Please provide 'cos_endpoint' or 'cos_region' in storage options, or set COS_ENDPOINT / TENCENTCLOUD_REGION environment variable",
                ));
            }
        }

        Ok(config_map)
    }

    /// Build a static OpendalStore from the normalized config
    fn build_cos_store(config_map: HashMap<String, String>) -> Result<OpendalStore> {
        let operator = Operator::from_iter::<Cos>(config_map)
            .map_err(|e| Error::invalid_input(format!("Failed to create COS operator: {:?}", e)))?
            .finish();

        Ok(OpendalStore::new(operator))
    }
}

#[async_trait::async_trait]
impl ObjectStoreProvider for TencentStoreProvider {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore> {
        let block_size = params.block_size.unwrap_or(DEFAULT_CLOUD_BLOCK_SIZE);
        let storage_options = StorageOptions(params.storage_options().cloned().unwrap_or_default());

        let base_options = Self::base_cos_options(&base_path, &storage_options)?;
        let accessor = params.get_accessor();

        // Two priority levels for store creation:
        // (1) First priority: check for external provider
        // (2) Second priority: user explicitly provided credentials (including temporary credentials), create a static store
        let inner: Arc<dyn OSObjectStore> =
            if let Some(accessor) = accessor.filter(|a| a.has_provider()) {
                Arc::new(
                    DynamicOpenDalStore::new(
                        format!("cos:{}", base_path),
                        base_options,
                        accessor,
                        Self::normalize_cos_config,
                        Self::build_cos_store,
                    )
                    .with_protected_keys(["bucket", "root"]),
                )
            } else {
                Arc::new(Self::build_cos_store(Self::normalize_cos_config(
                    &base_options,
                )?)?)
            };

        let mut url = base_path;
        if !url.path().ends_with('/') {
            url.set_path(&format!("{}/", url.path()));
        }

        Ok(ObjectStore {
            scheme: "cos".to_string(),
            inner,
            block_size,
            max_iop_size: *DEFAULT_MAX_IOP_SIZE,
            use_constant_size_upload_parts: params.use_constant_size_upload_parts,
            list_is_lexically_ordered: params.list_is_lexically_ordered.unwrap_or(true),
            io_parallelism: DEFAULT_CLOUD_IO_PARALLELISM,
            download_retry_count: storage_options.download_retry_count(),
            io_tracker: Default::default(),
            store_prefix: self.calculate_object_store_prefix(&url, params.storage_options())?,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::TencentStoreProvider;
    use crate::object_store::dynamic_opendal::DynamicOpenDalStore;
    use crate::object_store::test_utils::StaticMockStorageOptionsProvider;
    use crate::object_store::{ObjectStoreProvider, StorageOptionsAccessor};
    use url::Url;

    #[test]
    fn test_cos_store_path() {
        let provider = TencentStoreProvider;

        let url = Url::parse("cos://bucket/path/to/file").unwrap();
        let path = provider.extract_path(&url).unwrap();
        let expected_path = object_store::path::Path::from("path/to/file");
        assert_eq!(path, expected_path);
    }

    #[test]
    fn test_cosn_store_path() {
        let provider = TencentStoreProvider;

        let url = Url::parse("cosn://bucket/path/to/file").unwrap();
        let path = provider.extract_path(&url).unwrap();
        let expected_path = object_store::path::Path::from("path/to/file");
        assert_eq!(path, expected_path);
    }

    #[test]
    fn test_cos_alias_options_override_canonical_env_options() {
        let config = TencentStoreProvider::normalize_cos_config(&HashMap::from([
            (
                "endpoint".to_string(),
                "https://cos.ap-guangzhou.myqcloud.com".to_string(),
            ),
            (
                "cos_endpoint".to_string(),
                "https://cos.ap-shanghai.myqcloud.com".to_string(),
            ),
            ("secret_id".to_string(), "env-secret-id".to_string()),
            ("cos_secret_id".to_string(), "user-secret-id".to_string()),
            ("secret_key".to_string(), "env-secret-key".to_string()),
            ("cos_secret_key".to_string(), "user-secret-key".to_string()),
            ("security_token".to_string(), "env-token".to_string()),
            ("cos_security_token".to_string(), "user-token".to_string()),
            ("bucket".to_string(), "bucket".to_string()),
        ]))
        .unwrap();

        assert_eq!(
            config.get("endpoint").unwrap(),
            "https://cos.ap-shanghai.myqcloud.com"
        );
        assert_eq!(config.get("secret_id").unwrap(), "user-secret-id");
        assert_eq!(config.get("secret_key").unwrap(), "user-secret-key");
        assert_eq!(config.get("security_token").unwrap(), "user-token");
        assert!(!config.contains_key("cos_endpoint"));
        assert!(!config.contains_key("cos_secret_id"));
        assert!(!config.contains_key("cos_secret_key"));
        assert!(!config.contains_key("cos_security_token"));
    }

    #[test]
    fn test_cos_url_bucket_and_root_are_authoritative() {
        let storage_options = crate::object_store::StorageOptions(HashMap::from([
            (
                "cos_endpoint".to_string(),
                "https://cos.ap-guangzhou.myqcloud.com".to_string(),
            ),
            ("bucket".to_string(), "storage-options-bucket".to_string()),
            ("root".to_string(), "/storage-options-root".to_string()),
        ]));
        let base_options = TencentStoreProvider::base_cos_options(
            &Url::parse("cos://url-bucket/path").unwrap(),
            &storage_options,
        )
        .unwrap();
        let config = TencentStoreProvider::normalize_cos_config(&base_options).unwrap();

        // Bucket from URL has the highest priority
        assert_eq!(config.get("bucket").unwrap(), "url-bucket");
        assert_eq!(config.get("root").unwrap(), "/");
    }

    #[test]
    fn test_cos_empty_url_path_removes_storage_option_root() {
        let storage_options = crate::object_store::StorageOptions(HashMap::from([
            (
                "cos_endpoint".to_string(),
                "https://cos.ap-guangzhou.myqcloud.com".to_string(),
            ),
            ("root".to_string(), "/storage-options-root".to_string()),
        ]));
        let base_options = TencentStoreProvider::base_cos_options(
            &Url::parse("cos://url-bucket").unwrap(),
            &storage_options,
        )
        .unwrap();
        let config = TencentStoreProvider::normalize_cos_config(&base_options).unwrap();

        assert_eq!(config.get("bucket").unwrap(), "url-bucket");
        assert!(!config.contains_key("root"));
    }

    #[test]
    fn test_cos_region_generates_endpoint_automatically() {
        // When only region is provided (no endpoint), endpoint should be auto-generated
        let config = TencentStoreProvider::normalize_cos_config(&HashMap::from([
            ("region".to_string(), "ap-guangzhou".to_string()),
            ("bucket".to_string(), "test-bucket".to_string()),
        ]))
        .unwrap();

        assert_eq!(
            config.get("endpoint").unwrap(),
            "https://cos.ap-guangzhou.myqcloud.com"
        );
    }

    #[test]
    fn test_cos_region_alias_generates_endpoint() {
        // cos_region alias should also work
        let config = TencentStoreProvider::normalize_cos_config(&HashMap::from([
            ("cos_region".to_string(), "ap-shanghai".to_string()),
            ("bucket".to_string(), "test-bucket".to_string()),
        ]))
        .unwrap();

        assert_eq!(
            config.get("endpoint").unwrap(),
            "https://cos.ap-shanghai.myqcloud.com"
        );
    }

    #[test]
    fn test_cos_endpoint_takes_priority_over_region() {
        // When both endpoint and region are provided, endpoint should take priority
        let config = TencentStoreProvider::normalize_cos_config(&HashMap::from([
            (
                "endpoint".to_string(),
                "https://custom-endpoint.example.com".to_string(),
            ),
            ("region".to_string(), "ap-guangzhou".to_string()),
            ("bucket".to_string(), "test-bucket".to_string()),
        ]))
        .unwrap();

        assert_eq!(
            config.get("endpoint").unwrap(),
            "https://custom-endpoint.example.com"
        );
    }

    #[test]
    fn test_cos_no_endpoint_no_region_returns_error() {
        // When neither endpoint nor region is provided, should return error
        let result = TencentStoreProvider::normalize_cos_config(&HashMap::from([(
            "bucket".to_string(),
            "test-bucket".to_string(),
        )]));

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("endpoint or region is required"));
    }

    #[tokio::test]
    async fn test_dynamic_opendal_cos_store_uses_provider_credentials() {
        let accessor = Arc::new(StorageOptionsAccessor::with_provider(Arc::new(
            StaticMockStorageOptionsProvider {
                options: HashMap::from([
                    (
                        "cos_endpoint".to_string(),
                        "https://cos.ap-guangzhou.myqcloud.com".to_string(),
                    ),
                    ("cos_secret_id".to_string(), "akid".to_string()),
                    ("cos_secret_key".to_string(), "secret".to_string()),
                    ("cos_security_token".to_string(), "token".to_string()),
                ]),
            },
        )));

        let base_options = TencentStoreProvider::base_cos_options(
            &Url::parse("cos://bucket/path").unwrap(),
            &crate::object_store::StorageOptions(HashMap::new()),
        )
        .unwrap();

        let store = DynamicOpenDalStore::new(
            "cos",
            base_options,
            accessor,
            TencentStoreProvider::normalize_cos_config,
            TencentStoreProvider::build_cos_store,
        );

        let current_store = store
            .current_store()
            .await
            .expect("dynamic OpenDAL COS store should build");

        assert!(current_store.to_string().contains("Opendal"));
    }
}
