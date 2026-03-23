// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, sync::Arc};

use crate::object_store::{
    DEFAULT_LOCAL_BLOCK_SIZE, DEFAULT_LOCAL_IO_PARALLELISM, DEFAULT_MAX_IOP_SIZE, ObjectStore,
    ObjectStoreParams, ObjectStoreProvider, StorageOptions,
};
use lance_core::Error;
use lance_core::error::Result;
use object_store::{local::LocalFileSystem, path::Path};
use url::Url;

#[derive(Default, Debug)]
pub struct FileStoreProvider;

#[async_trait::async_trait]
impl ObjectStoreProvider for FileStoreProvider {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore> {
        // By default, file:/// and file-object-store:/// do not honor the storage prefix in the creation URL.
        // So you could create an ObjectStore from the URL file:///foo, ask for file:///bar and get /bar rather than /foo/bar.
        // The storage option honor_local_prefix makes it work as expected.
        let honor_prefix = params
            .storage_options()
            .and_then(|options| options.get("honor_local_prefix"))
            .is_some_and(|value| value == "true");
        let inner = if honor_prefix {
            Arc::new(LocalFileSystem::new_with_prefix(base_path.path())?)
        } else {
            Arc::new(LocalFileSystem::new())
        };
        let block_size = params.block_size.unwrap_or(DEFAULT_LOCAL_BLOCK_SIZE);
        let storage_options = StorageOptions(params.storage_options().cloned().unwrap_or_default());
        let download_retry_count = storage_options.download_retry_count();
        Ok(ObjectStore {
            inner,
            scheme: base_path.scheme().to_owned(),
            block_size,
            max_iop_size: *DEFAULT_MAX_IOP_SIZE,
            use_constant_size_upload_parts: false,
            list_is_lexically_ordered: false,
            io_parallelism: DEFAULT_LOCAL_IO_PARALLELISM,
            download_retry_count,
            io_tracker: Default::default(),
            store_prefix: self
                .calculate_object_store_prefix(&base_path, params.storage_options())?,
        })
    }

    fn extract_path(&self, url: &Url) -> Result<Path> {
        if let Ok(file_path) = url.to_file_path()
            && let Ok(path) = Path::from_absolute_path(&file_path)
        {
            return Ok(path);
        }

        Path::parse(url.path()).map_err(|e| {
            Error::invalid_input(format!("Failed to parse path '{}': {}", url.path(), e))
        })
    }

    fn calculate_object_store_prefix(
        &self,
        url: &Url,
        _storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        Ok(url.scheme().to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Arc;

    use object_store::PutPayload;
    use tempfile::tempdir;

    use crate::object_store::StorageOptionsAccessor;
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
            let path = provider.extract_path(&url).unwrap();
            assert_eq!(path.as_ref(), expected_path, "uri: '{}'", uri);
        }
    }

    #[test]
    fn test_calculate_object_store_prefix() {
        let provider = FileStoreProvider;
        assert_eq!(
            "file",
            provider
                .calculate_object_store_prefix(&Url::parse("file:///etc").unwrap(), None)
                .unwrap()
        );
    }

    #[test]
    fn test_calculate_object_store_prefix_for_file_object_store() {
        let provider = FileStoreProvider;
        assert_eq!(
            "file-object-store",
            provider
                .calculate_object_store_prefix(
                    &Url::parse("file-object-store:///etc").unwrap(),
                    None
                )
                .unwrap()
        );
    }

    #[tokio::test]
    async fn test_new_store_honors_local_prefix_option() {
        let provider = FileStoreProvider;
        let prefix_dir = tempdir().unwrap();
        let base_path = Url::from_directory_path(prefix_dir.path()).unwrap();
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([("honor_local_prefix".to_string(), "true".to_string())]),
            ))),
            ..Default::default()
        };

        let store = provider.new_store(base_path, &params).await.unwrap();
        let location = Path::from("nested/file.txt");
        store
            .inner
            .put(&location, PutPayload::from_static(b"hello"))
            .await
            .unwrap();

        let file_path = prefix_dir.path().join("nested/file.txt");
        assert_eq!(fs::read(file_path).unwrap(), b"hello");
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
            (
                "file-object-store:///C:/Users/ADMINI~1/AppData/Local",
                "C:/Users/ADMINI~1/AppData/Local",
            ),
        ];

        for (uri, expected_path) in cases {
            let url = uri_to_url(uri).unwrap();
            let path = provider.extract_path(&url).unwrap();
            assert_eq!(path.as_ref(), expected_path);
        }
    }
}
