// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use object_store::local::LocalFileSystem;
use url::Url;

use crate::object_store::{
    tracing::ObjectStoreTracingExt, ObjectStore, ObjectStoreParams, ObjectStoreProvider,
    StorageOptions, DEFAULT_LOCAL_BLOCK_SIZE, DEFAULT_LOCAL_IO_PARALLELISM,
};
use lance_core::error::Result;

#[derive(Default, Debug)]
pub struct FileStoreProvider;

#[async_trait::async_trait]
impl ObjectStoreProvider for FileStoreProvider {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore> {
        let block_size = params.block_size.unwrap_or(DEFAULT_LOCAL_BLOCK_SIZE);
        let storage_options = StorageOptions(params.storage_options.clone().unwrap_or_default());
        let download_retry_count = storage_options.download_retry_count();
        Ok(ObjectStore {
            inner: Arc::new(LocalFileSystem::new()).traced(),
            scheme: base_path.scheme().to_owned(),
            block_size,
            use_constant_size_upload_parts: false,
            list_is_lexically_ordered: false,
            io_parallelism: DEFAULT_LOCAL_IO_PARALLELISM,
            download_retry_count,
        })
    }
}
