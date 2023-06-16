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

use std::path::{Path as StdPath, PathBuf};
use std::sync::Arc;

use ::object_store::{
    aws::AmazonS3Builder, gcp::GoogleCloudStorageBuilder, local::LocalFileSystem, memory::InMemory,
    path::Path, ClientOptions, ObjectStore as OSObjectStore,
};
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
    /// Parse from a string URI.
    ///
    /// Returns the ObjectStore instance and the absolute path to the object.
    pub async fn from_uri(uri: &str) -> Result<(Self, Path)> {
        match Url::parse(uri) {
            Ok(url) if url.scheme().len() == 1 && cfg!(windows) => {
                // On Windows, the drive is parsed as a scheme
                Self::new_from_path(uri)
            }
            Ok(url) => {
                let store = Self::new_from_url(url.clone()).await?;
                let path = Path::from(url.path());
                Ok((store, path))
            }
            Err(_) => Self::new_from_path(uri),
        }
    }

    fn new_from_path(str_path: &str) -> Result<(Self, Path)> {
        let expanded = tilde(str_path).to_string();
        let expanded_path = StdPath::new(&expanded);

        if !expanded_path.try_exists()? {
            std::fs::create_dir_all(expanded_path.clone())?;
        } else if !expanded_path.is_dir() {
            return Err(Error::IO {
                message: format!("{} is not a lance directory", str_path),
            });
        }
        let expanded_path = expanded_path.canonicalize()?;

        Ok((
            Self {
                inner: Arc::new(LocalFileSystem::new()),
                scheme: String::from("file"),
                base_path: Path::from_absolute_path(&expanded_path)?,
                block_size: 4 * 1024, // 4KB block size
            },
            Path::from_filesystem_path(&expanded_path)?,
        ))
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
            "file" => Ok(Self::new_from_path(url.path())?.0),
            "memory" => Ok(Self {
                inner: Arc::new(InMemory::new()),
                scheme: String::from("memory"),
                base_path: Path::from(url.path()),
                block_size: 64 * 1024,
            }),
            s => Err(Error::IO {
                message: format!("Unsupported URI scheme: {}", s),
            }),
        }
    }

    /// Create a in-memory object store directly for testing.
    pub fn memory() -> Self {
        Self {
            inner: Arc::new(InMemory::new()),
            scheme: String::from("memory"),
            base_path: Path::from("/"),
            block_size: 64 * 1024,
        }
    }

    /// Returns true if the object store pointed to a local file system.
    pub fn is_local(&self) -> bool {
        self.scheme == "file"
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

    /// Open a file for path.
    ///
    /// The [path] is absolute path.
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
        Ok(output
            .common_prefixes
            .iter()
            .chain(output.objects.iter().map(|o| &o.location))
            .map(|s| s.filename().unwrap().to_string())
            .collect())
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

    /// Deletes a file from the underlying storage
    pub async fn delete_file(&self, path: &Path) -> Result<()> {
        self.inner.delete(path).await?;
        Ok(())
    }

    /// Deletes a directory from the underlying storage, only supported in local file systems
    pub async fn remove_dir(&self, path: &Path) -> Result<()> {
        // Adapted object_storage::LocalFileSystem, since there is no public Path -> std::io::Path
        fn path_to_filesystem(location: &Path) -> Result<PathBuf> {
            let mut url = Url::parse("file:///").unwrap();
            url.path_segments_mut()
                .expect("url path")
                .pop_if_empty()
                .extend(location.parts());

            url.to_file_path().map_err(|_| {
                Error::IO {
                    message: format!("Invalid URL {}", url),
                }
                .into()
            })
        }

        // aws / gcloud removes empty folders so there is no need to explicitly delete them
        if self.is_local() {
            std::fs::remove_dir(path_to_filesystem(path)?)?
        }
        Ok(())
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
            let (store, path) = ObjectStore::from_uri(uri).await.unwrap();
            let contents = read_from_store(store, &path.child("test_file"))
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
        let (store, path) = ObjectStore::from_uri("./bar/foo.lance").await.unwrap();

        let contents = read_from_store(store, &path.child("test_file"))
            .await
            .unwrap();
        assert_eq!(contents, "RELATIVE_URL");
    }

    #[tokio::test]
    async fn test_tilde_expansion() {
        let uri = "~/foo.lance";
        write_to_file(&(uri.to_string() + "/test_file"), "TILDE").unwrap();
        let (store, path) = ObjectStore::from_uri(uri).await.unwrap();
        let contents = read_from_store(store, &path.child("test_file"))
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
        let (store, base) = ObjectStore::from_uri(path.to_str().unwrap()).await.unwrap();

        let sub_dirs = store.read_dir(base.child("foo")).await.unwrap();
        assert_eq!(sub_dirs, vec!["bar", "zoo", "test_file"]);
    }

    #[tokio::test]
    async fn test_delete() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path().to_str().unwrap();
        let (store, path) = ObjectStore::from_uri(tmp_path).await.unwrap();
        write_to_file(
            tmp_dir.path().join("test_file").to_str().unwrap(),
            "TEST_CONTENT",
        )
        .unwrap();

        let exists = store.exists(&path.child("test_file")).await;
        assert!(exists.ok().unwrap());
        store.delete_file(&path.child("test_file")).await.unwrap();
        let exists = store.exists(&path.child("test_file")).await;
        assert!(!exists.ok().unwrap());
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
            &(format!("{drive_letter}:/test_folder/test.lance") + "/test_file"),
            "WINDOWS",
        )
        .unwrap();

        for uri in &[
            format!("{drive_letter}:/test_folder/test.lance"),
            format!("{drive_letter}:\\test_folder\\test.lance"),
        ] {
            let (store, base) = ObjectStore::from_uri(uri).await.unwrap();
            let contents = read_from_store(store, &base.child("test_file"))
                .await
                .unwrap();
            assert_eq!(contents, "WINDOWS");
        }
    }
}
