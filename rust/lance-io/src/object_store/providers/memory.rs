// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use object_store::memory::InMemory;
use url::Url;

use crate::object_store::{
    tracing::ObjectStoreTracingExt, ObjectStore, ObjectStoreParams, ObjectStoreProvider,
    StorageOptions, DEFAULT_LOCAL_BLOCK_SIZE,
};
use lance_core::{error::Result, utils::tokio::get_num_compute_intensive_cpus};

#[derive(Default, Debug)]
pub struct MemoryStoreProvider;

#[async_trait::async_trait]
impl ObjectStoreProvider for MemoryStoreProvider {
    async fn new_store(&self, _base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore> {
        let block_size = params.block_size.unwrap_or(DEFAULT_LOCAL_BLOCK_SIZE);
        let storage_options = StorageOptions(params.storage_options.clone().unwrap_or_default());
        let download_retry_count = storage_options.download_retry_count();
        Ok(ObjectStore {
            inner: Arc::new(InMemory::new()).traced(),
            scheme: String::from("memory"),
            block_size,
            use_constant_size_upload_parts: false,
            list_is_lexically_ordered: true,
            io_parallelism: get_num_compute_intensive_cpus(),
            download_retry_count,
        })
    }
}
