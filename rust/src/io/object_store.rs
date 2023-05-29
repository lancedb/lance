// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Wraps [ObjectStore](object_store::ObjectStore)

use std::ops::Deref;
use std::path::Path as StdPath;
use std::sync::Arc;

use ::object_store::{
    aws::AmazonS3Builder, memory::InMemory, path::Path, ObjectStore as OSObjectStore,
};
use futures::{future, TryFutureExt};
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::local::LocalFileSystem;
use object_store::ClientOptions;
use reqwest::header::{HeaderMap, CACHE_CONTROL};
use shellexpand::tilde;
use url::Url;

use crate::error::{Error, Result};
use crate::io::object_reader::CloudObjectReader;
use crate::io::object_writer::ObjectWriter;

use super::local::LocalObjectReader;
use super::object_reader::ObjectReader;

/// Wraps [ObjectStore](object_store::ObjectStore)
#[derive(Debug, Clone)]
pub struct ObjectStore {
    // Inner object store
    pub inner: Arc<dyn OSObjectStore>,
    scheme: String,
    base_path: Path,
    block_size: usize,
}

impl std::fmt::Display for ObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ObjectStore({})", self.scheme)
    }
}

/// BUild S3 ObjectStore using default credential chain.
async fn build_s3_object_store(uri: &str) -> Result<Arc<dyn OSObjectStore>> {
    use aws_config::meta::region::RegionProviderChain;

    const DEFAULT_REGION: &str = "us-west-2";

    let region_provider = RegionProviderChain::default_provider().or_else(DEFAULT_REGION);
    Ok(Arc::new(
        AmazonS3Builder::from_env()
            .with_url(uri)
            .with_region(
                region_provider
                    .region()
                    .await
                    .map(|r| r.as_ref().to_string())
                    .unwrap_or(DEFAULT_REGION.to_string()),
            )
            .build()?,
    ))
}

async fn build_gcs_object_store(uri: &str) -> Result<Arc<dyn OSObjectStore>> {
    // GCS enables cache for public buckets, we disable to improve consistency
    let mut headers = HeaderMap::new();
    headers.insert(CACHE_CONTROL, "no-cache".parse().unwrap());
    Ok(Arc::new(
        GoogleCloudStorageBuilder::from_env()
            .with_url(uri)
            .with_client_options(ClientOptions::new().with_default_headers(headers))
            .build()?,
    ))
}

impl ObjectStore {
    /// Create a ObjectStore instance from a given URL.
    pub async fn new(uri: &str) -> Result<Self> {
        if uri == ":memory:" {
            return Ok(Self::memory());
        };

        // Try to parse the provided string as a Url, if that fails treat it as local FS
        future::ready(Url::parse(uri))
            .map_err(Error::from)
            .and_then(|url| Self::new_from_url(url))
            .or_else(|_| future::ready(Self::new_from_path(uri)))
            .await
    }

    fn new_from_path(str_path: &str) -> Result<Self> {
        let expanded = tilde(str_path).to_string();
        let expanded_path = StdPath::new(&expanded);

        if !expanded_path.try_exists()? {
            std::fs::create_dir_all(expanded_path.clone())?;
        } else if !expanded_path.is_dir() {
            return Err(Error::IO(format!("{} is not a lance directory", str_path)));
        }

        Ok(Self {
            inner: Arc::new(LocalFileSystem::new_with_prefix(expanded_path.deref())?),
            scheme: String::from("flle"),
            base_path: Path::from(object_store::path::DELIMITER),
            block_size: 4 * 1024, // 4KB block size
        })
    }

    async fn new_from_url(url: Url) -> Result<Self> {
        match url.scheme() {
            "s3" => Ok(Self {
                inner: build_s3_object_store(url.to_string().as_str()).await?,
                scheme: String::from("s3"),
                base_path: Path::from(url.path()),
                block_size: 64 * 1024,
            }),
            "gs" => Ok(Self {
                inner: build_gcs_object_store(url.to_string().as_str()).await?,
                scheme: String::from("gs"),
                base_path: Path::from(url.path()),
                block_size: 64 * 1024,
            }),
            "file" => Self::new_from_path(url.path()),
            s => Err(Error::IO(format!("Unsupported scheme {}", s))),
        }
    }

    /// Create a in-memory object store directly.
    pub(crate) fn memory() -> Self {
        Self {
            inner: Arc::new(InMemory::new()),
            scheme: String::from("memory"),
            base_path: Path::from("/"),
            block_size: 64 * 1024,
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn set_block_size(&mut self, new_size: usize) {
        self.block_size = new_size;
    }

    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Open a file for path
    pub async fn open(&self, path: &Path) -> Result<Box<dyn ObjectReader>> {
        match self.scheme.as_str() {
            "file" => LocalObjectReader::open(path, self.block_size),
            _ => Ok(Box::new(CloudObjectReader::new(
                self,
                path.clone(),
                self.block_size,
            )?)),
        }
    }

    /// Create a new file.
    pub async fn create(&self, path: &Path) -> Result<ObjectWriter> {
        ObjectWriter::new(self, path).await
    }

    /// Read a directory (start from base directory) and returns all sub-paths in the directory.
    pub async fn read_dir(&self, dir_path: impl Into<Path>) -> Result<Vec<String>> {
        let path = dir_path.into();
        let path = Path::parse(path.to_string())?;
        let output = self.inner.list_with_delimiter(Some(&path)).await?;
        output
            .common_prefixes
            .iter()
            .map(|cp| cp.filename().map(|s| s.to_string()).unwrap_or_default())
            .chain(output.objects.iter().map(|o| o.location.to_string()))
            .map(|s| Ok(Path::parse(s)?.filename().unwrap().to_string()))
            .collect::<Result<Vec<String>>>()
    }

    /// Check a file exists.
    pub async fn exists(&self, path: &Path) -> Result<bool> {
        match self.inner.head(path).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { path: _, source: _ }) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    /// Get file size.
    pub async fn size(&self, path: &Path) -> Result<usize> {
        Ok(self.inner.head(path).await?.size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::set_current_dir;
    use std::fs::{create_dir_all, write};

    /// Write test content to file.
    fn write_to_file(path_str: &str, contents: &str) -> std::io::Result<()> {
        let expanded = tilde(path_str).to_string();
        let path = StdPath::new(&expanded);
        std::fs::create_dir_all(path.parent().unwrap())?;
        write(path, contents)
    }

    async fn read_from_store(store: ObjectStore, path: &Path) -> Result<String> {
        let test_file_store = store.open(&path).await.unwrap();
        let size = test_file_store.size().await.unwrap();
        let bytes = test_file_store.get_range(0..size).await.unwrap();
        let contents = String::from_utf8(bytes.to_vec()).unwrap();
        Ok(contents)
    }

    #[tokio::test]
    async fn test_absolute_paths() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path().to_str().unwrap().to_owned();
        write_to_file(
            &(tmp_path.clone() + "/bar/foo.lance/test_file"),
            "TEST_CONTENT",
        )
        .unwrap();

        // test a few variations of the same path
        for uri in &[
            tmp_path.clone() + "/bar/foo.lance",
            tmp_path.clone() + "/./bar/foo.lance",
            tmp_path.clone() + "/bar/foo.lance/../foo.lance",
        ] {
            let store = ObjectStore::new(uri).await.unwrap();
            let contents = read_from_store(store, &Path::from("test_file"))
                .await
                .unwrap();
            assert_eq!(contents, "TEST_CONTENT");
        }
    }

    #[tokio::test]
    async fn test_relative_paths() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path().to_str().unwrap().to_owned();
        write_to_file(
            &(tmp_path.clone() + "/bar/foo.lance/test_file"),
            "RELATIVE_URL",
        )
        .unwrap();

        set_current_dir(StdPath::new(&tmp_path)).expect("Error changing current dir");
        let store = ObjectStore::new("./bar/foo.lance").await.unwrap();

        let contents = read_from_store(store, &Path::from("test_file"))
            .await
            .unwrap();
        assert_eq!(contents, "RELATIVE_URL");
    }

    #[tokio::test]
    async fn test_tilde_expansion() {
        let uri = "~/foo.lance";
        write_to_file(&(uri.to_string() + "/test_file"), "TILDE").unwrap();
        let store = ObjectStore::new(uri).await.unwrap();
        let contents = read_from_store(store, &Path::from("test_file"))
            .await
            .unwrap();
        assert_eq!(contents, "TILDE");
    }

    #[tokio::test]
    async fn test_read_directory() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path();
        create_dir_all(path.join("foo").join("bar")).unwrap();
        create_dir_all(path.join("foo").join("zoo")).unwrap();
        create_dir_all(path.join("foo").join("zoo").join("abc")).unwrap();
        write_to_file(
            path.join("foo").join("test_file").to_str().unwrap(),
            "read_dir",
        )
        .unwrap();
        let store = ObjectStore::new(path.to_str().unwrap()).await.unwrap();
        // tmp_dir

        let sub_dirs = store.read_dir("foo").await.unwrap();
        assert_eq!(sub_dirs, vec!["bar", "zoo", "test_file"]);
    }

    #[tokio::test]
    #[cfg(windows)]
    async fn test_windows_paths() {
        use std::path::Component;
        use std::path::Prefix;
        use std::path::Prefix::*;

        fn get_path_prefix(path: &StdPath) -> Prefix {
            match path.components().next().unwrap() {
                Component::Prefix(prefix_component) => prefix_component.kind(),
                _ => panic!(),
            }
        }

        fn get_drive_letter(prefix: Prefix) -> String {
            match prefix {
                Disk(bytes) => String::from_utf8(vec![bytes]).unwrap(),
                _ => panic!(),
            }
        }

        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path();
        let prefix = get_path_prefix(tmp_path);
        let drive_letter = get_drive_letter(prefix);

        write_to_file(
            format!("{drive_letter}:/test_folder/test.lance") + "/test_file",
            "WINDOWS".to_string(),
        )
        .unwrap();

        for uri in &[
            format!("{drive_letter}:/test_folder/test.lance"),
            format!("{drive_letter}:\\test_folder\\test.lance"),
        ] {
            let store = ObjectStore::new(uri).await.unwrap();
            let contents = read_from_store(store, &Path::from("test_file"))
                .await
                .unwrap();
            assert_eq!(contents, "WINDOWS");
        }
    }
}
