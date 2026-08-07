// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, path::PathBuf, sync::Arc};

#[cfg(windows)]
use std::path::{Component, Prefix};

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

#[derive(Debug)]
struct UncPath {
    root: PathBuf,
    relative_path: Path,
    store_prefix: String,
}

fn extract_unc_path(url: &Url) -> Result<Option<UncPath>> {
    #[cfg(not(windows))]
    {
        let _ = url;
        Ok(None)
    }

    #[cfg(windows)]
    {
        let Some(host) = url.host_str().filter(|host| *host != "localhost") else {
            return Ok(None);
        };
        let encoded_path = url.path().strip_prefix('/').unwrap_or(url.path());
        let (share, relative_path) = encoded_path.split_once('/').unwrap_or((encoded_path, ""));
        if share.is_empty() {
            return Err(Error::invalid_input(format!(
                "UNC URL '{}' is missing a share name",
                url
            )));
        }

        // `Url::to_file_path` only accepts authorities for the `file` scheme.
        // The other local schemes share the same filesystem path semantics.
        let mut file_url = url.clone();
        file_url.set_scheme("file").map_err(|_| {
            Error::invalid_input(format!("Failed to convert '{}' to a file URL", url))
        })?;
        let filesystem_path = file_url.to_file_path().map_err(|_| {
            Error::invalid_input(format!("Failed to convert UNC URL '{}' to a path", url))
        })?;
        let Some(Component::Prefix(prefix)) = filesystem_path.components().next() else {
            return Err(Error::invalid_input(format!(
                "UNC URL '{}' did not produce a UNC path",
                url
            )));
        };
        if !matches!(prefix.kind(), Prefix::UNC(_, _) | Prefix::VerbatimUNC(_, _)) {
            return Err(Error::invalid_input(format!(
                "UNC URL '{}' did not produce a UNC path",
                url
            )));
        }

        Ok(Some(UncPath {
            root: PathBuf::from(prefix.as_os_str()),
            relative_path: Path::from_url_path(relative_path).map_err(|error| {
                Error::invalid_input(format!(
                    "Failed to parse path '{}' from UNC URL '{}': {}",
                    relative_path, url, error
                ))
            })?,
            store_prefix: format!("{}${}/{}", url.scheme(), host, share),
        }))
    }
}

#[async_trait::async_trait]
impl ObjectStoreProvider for FileStoreProvider {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore> {
        let block_size = params.block_size.unwrap_or(DEFAULT_LOCAL_BLOCK_SIZE);
        let storage_options = StorageOptions(params.storage_options().cloned().unwrap_or_default());
        let download_retry_count = storage_options.download_retry_count();
        let unc_path = extract_unc_path(&base_path)?;
        let local_path_prefix = unc_path.as_ref().map(|path| path.root.clone());
        let inner = match &local_path_prefix {
            Some(prefix) => LocalFileSystem::new_with_prefix(prefix)?,
            None => LocalFileSystem::new(),
        };
        Ok(ObjectStore {
            inner: Arc::new(inner),
            scheme: base_path.scheme().to_owned(),
            local_path_prefix,
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
        if let Some(unc_path) = extract_unc_path(url)? {
            return Ok(unc_path.relative_path);
        }
        if let Ok(file_path) = url.to_file_path()
            && let Ok(path) = Path::from_absolute_path(&file_path)
        {
            return Ok(path);
        }

        Path::from_url_path(url.path()).map_err(|e| {
            Error::invalid_input(format!("Failed to parse path '{}': {}", url.path(), e))
        })
    }

    fn calculate_object_store_prefix(
        &self,
        url: &Url,
        _storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        Ok(extract_unc_path(url)?
            .map(|path| path.store_prefix)
            .unwrap_or_else(|| url.scheme().to_string()))
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
            (
                "file:///C:/Users/RUNNER~1/AppData/Local/Temp/tmpm49j_w0f",
                "C:/Users/RUNNER~1/AppData/Local/Temp/tmpm49j_w0f",
            ),
            (
                "file://192.168.0.1/My%20Share/data/my-dataset.lance",
                "data/my-dataset.lance",
            ),
        ];

        for (uri, expected_path) in cases {
            let url = uri_to_url(uri).unwrap();
            let path = provider.extract_path(&url).unwrap();
            assert_eq!(path.as_ref(), expected_path);
        }
    }

    #[test]
    #[cfg(windows)]
    fn test_unc_share_path() {
        let url = Url::parse("file://server/My%20Share/data/my-dataset.lance").unwrap();
        let unc_path = extract_unc_path(&url).unwrap().unwrap();

        assert_eq!(unc_path.root, PathBuf::from(r"\\server\My Share"));
        assert_eq!(unc_path.relative_path.as_ref(), "data/my-dataset.lance");
        assert_eq!(unc_path.store_prefix, "file$server/My%20Share");
    }
}
