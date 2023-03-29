// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Wraps [ObjectStore](object_store::ObjectStore)

use std::path::Path as StdPath;
use std::sync::Arc;

use ::object_store::{
    aws::AmazonS3Builder, memory::InMemory, path::Path, ObjectStore as OSObjectStore,
};
use futures::{future, TryFutureExt};
use object_store::local::LocalFileSystem;
use path_absolutize::Absolutize;
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
    prefetch_size: usize,
}

impl std::fmt::Display for ObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ObjectStore({})", self.scheme)
    }
}

/// BUild S3 ObjectStore using default credential chain.
async fn build_s3_object_store(uri: &str) -> Result<Arc<dyn OSObjectStore>> {
    use aws_config::meta::region::RegionProviderChain;
    use aws_credential_types::provider::ProvideCredentials;

    const DEFAULT_REGION: &str = "us-west-2";

    let region_provider = RegionProviderChain::default_provider().or_else(DEFAULT_REGION);
    let provider = aws_config::default_provider::credentials::default_provider().await;
    let credentials = provider.provide_credentials().await.unwrap();
    Ok(Arc::new(
        AmazonS3Builder::new()
            .with_url(uri)
            .with_access_key_id(credentials.access_key_id())
            .with_secret_access_key(credentials.secret_access_key())
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
        let absolute_path = StdPath::new(expanded.as_str()).absolutize()?;
        let absolute_path_str = absolute_path
            .to_str()
            .ok_or(Error::IO(format!("can't convert path {}", str_path)))?;
        let url = Url::from_file_path(absolute_path_str).unwrap();
        Ok(Self {
            inner: Arc::new(LocalFileSystem::new()),
            scheme: String::from("flle"),
            base_path: Path::from(url.path()),
            prefetch_size: 64 * 1024,
        })
    }

    async fn new_from_url(url: Url) -> Result<Self> {
        match url.scheme() {
            "s3" => Ok(Self {
                inner: build_s3_object_store(url.to_string().as_str()).await?,
                scheme: String::from("s3"),
                base_path: Path::from(url.path()),
                prefetch_size: 64 * 1024,
            }),
            "file" => Self::new_from_path(url.path()),
            s => Err(Error::IO(format!("Unknown scheme {}", s))),
        }
    }

    /// Create a in-memory object store directly.
    pub(crate) fn memory() -> Self {
        Self {
            inner: Arc::new(InMemory::new()),
            scheme: String::from("memory"),
            base_path: Path::from("/"),
            prefetch_size: 64 * 1024,
        }
    }

    pub fn prefetch_size(&self) -> usize {
        self.prefetch_size
    }

    pub fn set_prefetch_size(&mut self, new_size: usize) {
        self.prefetch_size = new_size;
    }

    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Open a file for path
    pub async fn open(&self, path: &Path) -> Result<Box<dyn ObjectReader>> {
        match self.scheme.as_str() {
            "file" => LocalObjectReader::open(path),
            _ => Ok(Box::new(CloudObjectReader::new(
                self,
                path.clone(),
                self.prefetch_size,
            )?)),
        }
    }

    /// Create a new file.
    pub async fn create(&self, path: &Path) -> Result<ObjectWriter> {
        ObjectWriter::new(self, path).await
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

    #[tokio::test]
    #[cfg(unix)]
    async fn test_uri_expansion() {
        // test tilde and absolute path expansion
        for uri in &["./bar/foo.lance", "../bar/foo.lance", "~/foo.lance"] {
            let store = ObjectStore::new(uri).await.unwrap();
            // NOTE: this is an optimistic check, since Path.as_ref() doesn't read back the leading slash
            // for an absolute path, we are assuming it takes at least 1 char more than the original uri.
            assert!(store.base_path().as_ref().len() > uri.len());
        }

        // absolute file system uri doesn't need expansion
        let uri = "/bar/foo.lance";
        let store = ObjectStore::new(uri).await.unwrap();
        // +1 for the leading slash Path.as_ref() doesn't read back
        assert!(store.base_path().as_ref().len() + 1 == uri.len());
    }

    #[tokio::test]
    async fn test_tilde_expansion() {
        let uri = "~/foo.lance";
        let store = ObjectStore::new(uri).await.unwrap();

        // dir uses the platform-specific path separators
        //    /home/eto/foo.lance
        //    C:\Users\Administrator\foo.lance
        let mut dir = dirs::home_dir().unwrap();
        dir.push("foo.lance");
        // dir_to_url always uses /
        //    file:///Users/eto/foo.lance
        //    file:///C:/Users/Administrator/foo.lance
        let dir_to_url = Url::from_file_path(dir).unwrap();
        // We are only interested in the path part of the URL, and we drop the first /
        //    Users/eto/foo.lance
        //    C:/Users/Administrator/foo.lance
        let expected_path = &dir_to_url.path()[1..];

        assert_eq!(store.base_path().to_string(), expected_path);
    }
}
