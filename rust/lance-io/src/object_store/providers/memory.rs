// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use object_store::{memory::InMemory, path::Path};
use url::Url;

use crate::object_store::{
    ObjectStore, ObjectStoreParams, ObjectStoreProvider, StorageOptions,
    DEFAULT_CLOUD_IO_PARALLELISM, DEFAULT_LOCAL_BLOCK_SIZE, DEFAULT_MAX_IOP_SIZE,
};
use lance_core::error::Result;

/// Provides a fresh in-memory object store for each call to `new_store`.
#[derive(Default, Debug)]
pub struct MemoryStoreProvider;

#[async_trait::async_trait]
impl ObjectStoreProvider for MemoryStoreProvider {
    async fn new_store(&self, _base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore> {
        let block_size = params.block_size.unwrap_or(DEFAULT_LOCAL_BLOCK_SIZE);
        let storage_options = StorageOptions(params.storage_options.clone().unwrap_or_default());
        let download_retry_count = storage_options.download_retry_count();
        Ok(ObjectStore {
            inner: Arc::new(InMemory::new()),
            scheme: String::from("memory"),
            block_size,
            max_iop_size: *DEFAULT_MAX_IOP_SIZE,
            use_constant_size_upload_parts: false,
            list_is_lexically_ordered: true,
            io_parallelism: DEFAULT_CLOUD_IO_PARALLELISM,
            download_retry_count,
        })
    }

    fn extract_path(&self, url: &Url) -> Path {
        let mut output = String::new();
        if let Some(domain) = url.domain() {
            output.push_str(domain);
        }
        output.push_str(url.path());
        Path::from(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_store_path() {
        let provider = MemoryStoreProvider;

        let url = Url::parse("memory://path/to/file").unwrap();
        let path = provider.extract_path(&url);
        let expected_path = Path::from("path/to/file");
        assert_eq!(path, expected_path);
    }
}
