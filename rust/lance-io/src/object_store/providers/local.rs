// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use object_store::{local::LocalFileSystem, path::Path};
use url::Url;

use crate::object_store::{
    ObjectStore, ObjectStoreParams, ObjectStoreProvider, StorageOptions, DEFAULT_LOCAL_BLOCK_SIZE,
    DEFAULT_LOCAL_IO_PARALLELISM, DEFAULT_MAX_IOP_SIZE,
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
            inner: Arc::new(LocalFileSystem::new()),
            scheme: base_path.scheme().to_owned(),
            block_size,
            max_iop_size: *DEFAULT_MAX_IOP_SIZE,
            use_constant_size_upload_parts: false,
            list_is_lexically_ordered: false,
            io_parallelism: DEFAULT_LOCAL_IO_PARALLELISM,
            download_retry_count,
        })
    }

    fn extract_path(&self, url: &Url) -> object_store::path::Path {
        url.to_file_path()
            .ok()
            .and_then(|p| Path::from_absolute_path(p).ok())
            .unwrap_or_else(|| Path::from(url.path()))
    }
}

#[cfg(test)]
mod tests {
    use crate::object_store::uri_to_url;

    use super::*;

    #[test]
    fn test_file_store_path() {
        let provider = FileStoreProvider;

        let cases = [
            ("file:///", ""),
            ("file:///usr/local/bin", "usr/local/bin"),
            ("file-object-store:///path/to/file", "path/to/file"),
            ("file:///path/to/foo/../bar", "path/to/bar"),
        ];

        for (uri, expected_path) in cases {
            let url = uri_to_url(uri).unwrap();
            let path = provider.extract_path(&url);
            assert_eq!(path.as_ref(), expected_path, "uri: '{}'", uri);
        }
    }

    #[test]
    #[cfg(windows)]
    fn test_file_store_path_windows() {
        let provider = FileStoreProvider;

        let cases = [
            (
                "C:\\Users\\ADMINI~1\\AppData\\Local\\",
                "C:/Users/ADMINI~1/AppData/Local",
            ),
            (
                "C:\\Users\\ADMINI~1\\AppData\\Local\\..\\",
                "C:/Users/ADMINI~1/AppData",
            ),
        ];

        for (uri, expected_path) in cases {
            let url = uri_to_url(uri).unwrap();
            let path = provider.extract_path(&url);
            assert_eq!(path.as_ref(), expected_path);
        }
    }
}
