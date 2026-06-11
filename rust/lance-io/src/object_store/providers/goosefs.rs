// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use object_store::path::Path;
use object_store_opendal::OpendalStore;
use opendal::{Operator, services::GooseFs};
use url::Url;

use crate::object_store::{
    DEFAULT_CLOUD_BLOCK_SIZE, DEFAULT_CLOUD_IO_PARALLELISM, DEFAULT_MAX_IOP_SIZE, ObjectStore,
    ObjectStoreParams, ObjectStoreProvider, StorageOptions,
};
use lance_core::error::{Error, Result};

/// Default GooseFS Master gRPC port.
const DEFAULT_GOOSEFS_PORT: u16 = 9200;

/// GooseFS object store provider.
///
/// Uses OpenDAL's GooseFs service to access GooseFS via gRPC.
/// URL format: `goosefs://host:port/path`
///
/// Where:
/// - `host:port` is the GooseFS Master address (default port: 9200)
/// - `/path` is the filesystem path within GooseFS
///
/// Configuration priority: storage_options > environment variables > URL authority > defaults
#[derive(Default, Debug)]
pub struct GooseFsStoreProvider;

impl GooseFsStoreProvider {
    /// Resolve the GooseFS Master address from storage_options, environment, or URL.
    ///
    /// Priority:
    /// 1. `storage_options["goosefs_master_addr"]` (supports HA: "addr1:port,addr2:port")
    /// 2. `GOOSEFS_MASTER_ADDR` environment variable
    /// 3. URL authority (host:port from the URL)
    fn resolve_master_addr(url: &Url, storage_options: &StorageOptions) -> Result<String> {
        // 1. storage_options
        if let Some(addr) = storage_options
            .0
            .get("goosefs_master_addr")
            .filter(|v| !v.is_empty())
        {
            return Ok(addr.clone());
        }

        // 2. Environment variable
        if let Ok(addr) = std::env::var("GOOSEFS_MASTER_ADDR")
            && !addr.is_empty()
        {
            return Ok(addr);
        }

        // 3. URL authority
        let host = url.host_str().ok_or_else(|| {
            Error::invalid_input(
                "GooseFS URL must contain a master address (host), e.g. goosefs://host:port/path",
            )
        })?;

        let port = url.port().unwrap_or(DEFAULT_GOOSEFS_PORT);
        Ok(format!("{}:{}", host, port))
    }

    /// Resolve a storage option from storage_options or environment variable.
    fn resolve_option(
        storage_options: &StorageOptions,
        option_key: &str,
        env_key: &str,
    ) -> Option<String> {
        storage_options
            .0
            .get(option_key)
            .cloned()
            .or_else(|| std::env::var(env_key).ok())
            .filter(|v| !v.is_empty())
    }
}

#[async_trait::async_trait]
impl ObjectStoreProvider for GooseFsStoreProvider {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore> {
        let block_size = params.block_size.unwrap_or(DEFAULT_CLOUD_BLOCK_SIZE);
        let storage_options = StorageOptions(params.storage_options().cloned().unwrap_or_default());

        // Resolve master address
        let master_addr = Self::resolve_master_addr(&base_path, &storage_options)?;

        // Extract root path from URL
        let root = base_path.path().to_string();

        // Build OpenDAL config map
        let mut config_map: HashMap<String, String> = HashMap::new();
        config_map.insert("master_addr".to_string(), master_addr);

        if !root.is_empty() && root != "/" {
            config_map.insert("root".to_string(), root);
        }

        // Optional: write_type
        if let Some(wt) =
            Self::resolve_option(&storage_options, "goosefs_write_type", "GOOSEFS_WRITE_TYPE")
        {
            config_map.insert("write_type".to_string(), wt);
        }

        // Optional: block_size (for GooseFS, not Lance block_size)
        if let Some(bs) =
            Self::resolve_option(&storage_options, "goosefs_block_size", "GOOSEFS_BLOCK_SIZE")
        {
            config_map.insert("block_size".to_string(), bs);
        }

        // Optional: chunk_size
        if let Some(cs) =
            Self::resolve_option(&storage_options, "goosefs_chunk_size", "GOOSEFS_CHUNK_SIZE")
        {
            config_map.insert("chunk_size".to_string(), cs);
        }

        // Optional: auth_type (nosasl / simple)
        if let Some(at) =
            Self::resolve_option(&storage_options, "goosefs_auth_type", "GOOSEFS_AUTH_TYPE")
        {
            config_map.insert("auth_type".to_string(), at);
        }

        // Optional: auth_username (used in SIMPLE auth mode)
        if let Some(au) = Self::resolve_option(
            &storage_options,
            "goosefs_auth_username",
            "GOOSEFS_AUTH_USERNAME",
        ) {
            config_map.insert("auth_username".to_string(), au);
        }

        // Create OpenDAL Operator with GooseFS service
        let operator = Operator::from_iter::<GooseFs>(config_map)
            .map_err(|e| {
                Error::invalid_input(format!("Failed to create GooseFS operator: {:?}", e))
            })?
            .finish();

        // Wrap as object_store::ObjectStore via OpendalStore bridge
        let opendal_store = Arc::new(OpendalStore::new(operator));

        Ok(ObjectStore {
            scheme: "goosefs".to_string(),
            inner: opendal_store,
            block_size,
            max_iop_size: *DEFAULT_MAX_IOP_SIZE,
            use_constant_size_upload_parts: params.use_constant_size_upload_parts,
            list_is_lexically_ordered: params.list_is_lexically_ordered.unwrap_or(false),
            io_parallelism: DEFAULT_CLOUD_IO_PARALLELISM,
            download_retry_count: storage_options.download_retry_count(),
            io_tracker: Default::default(),
            store_prefix: self
                .calculate_object_store_prefix(&base_path, params.storage_options())?,
        })
    }

    /// Extract the path relative to the root of the GooseFS filesystem.
    ///
    /// For GooseFS, the entire URL path is set as the OpenDAL `root` in `new_store`,
    /// so the relative path returned here must be empty to avoid path duplication.
    ///
    /// `goosefs://host:port/data/file.lance` → root="/data/file.lance", extract_path=""
    fn extract_path(&self, _url: &Url) -> Result<Path> {
        Ok(Path::from(""))
    }

    /// Calculate the object store prefix for caching.
    ///
    /// Format: `goosefs$host:port`
    /// This ensures different GooseFS clusters get separate caches.
    fn calculate_object_store_prefix(
        &self,
        url: &Url,
        _storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        Ok(format!("{}${}", url.scheme(), url.authority()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goosefs_store_path() {
        let provider = GooseFsStoreProvider;

        let url = Url::parse("goosefs://10.0.0.1:9200/data/embeddings.lance").unwrap();
        let path = provider.extract_path(&url).unwrap();
        // extract_path returns empty because the full path is used as OpenDAL root
        assert_eq!(path.to_string(), "");
    }

    #[test]
    fn test_goosefs_store_root_path() {
        let provider = GooseFsStoreProvider;

        let url = Url::parse("goosefs://10.0.0.1:9200/").unwrap();
        let path = provider.extract_path(&url).unwrap();
        assert_eq!(path.to_string(), "");
    }

    #[test]
    fn test_goosefs_store_deep_path() {
        let provider = GooseFsStoreProvider;

        let url = Url::parse("goosefs://master:9200/a/b/c/d.lance").unwrap();
        let path = provider.extract_path(&url).unwrap();
        // All path components are in the OpenDAL root, extract_path is empty
        assert_eq!(path.to_string(), "");
    }

    #[test]
    fn test_calculate_object_store_prefix() {
        let provider = GooseFsStoreProvider;

        let url = Url::parse("goosefs://10.0.0.1:9200/data").unwrap();
        let prefix = provider.calculate_object_store_prefix(&url, None).unwrap();
        assert_eq!(prefix, "goosefs$10.0.0.1:9200");
    }

    #[test]
    fn test_calculate_object_store_prefix_with_hostname() {
        let provider = GooseFsStoreProvider;

        let url = Url::parse("goosefs://myhost:9200/data").unwrap();
        let prefix = provider.calculate_object_store_prefix(&url, None).unwrap();
        assert_eq!(prefix, "goosefs$myhost:9200");
    }

    #[test]
    fn test_resolve_master_addr_from_url() {
        let url = Url::parse("goosefs://10.0.0.1:9200/data").unwrap();
        let storage_options = StorageOptions(HashMap::new());
        let addr = GooseFsStoreProvider::resolve_master_addr(&url, &storage_options).unwrap();
        assert_eq!(addr, "10.0.0.1:9200");
    }

    #[test]
    fn test_resolve_master_addr_default_port() {
        let url = Url::parse("goosefs://10.0.0.1/data").unwrap();
        let storage_options = StorageOptions(HashMap::new());
        let addr = GooseFsStoreProvider::resolve_master_addr(&url, &storage_options).unwrap();
        assert_eq!(addr, "10.0.0.1:9200");
    }

    #[test]
    fn test_resolve_master_addr_from_storage_options() {
        let url = Url::parse("goosefs://10.0.0.1:9200/data").unwrap();
        let storage_options = StorageOptions(HashMap::from([(
            "goosefs_master_addr".to_string(),
            "10.0.0.2:9200,10.0.0.3:9200".to_string(),
        )]));
        let addr = GooseFsStoreProvider::resolve_master_addr(&url, &storage_options).unwrap();
        assert_eq!(addr, "10.0.0.2:9200,10.0.0.3:9200");
    }
}
