// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Extend [object_store::ObjectStore] functionalities

use std::collections::HashMap;
use std::ops::Range;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use deepsize::DeepSizeOf;
use futures::{future, stream::BoxStream, StreamExt, TryStreamExt};
use futures::{FutureExt, Stream};
use lance_core::error::LanceOptionExt;
use lance_core::utils::parse::str_is_truthy;
use list_retry::ListRetryStream;
#[cfg(feature = "aws")]
use object_store::aws::AwsCredentialProvider;
use object_store::DynObjectStore;
use object_store::Error as ObjectStoreError;
use object_store::{path::Path, ObjectMeta, ObjectStore as OSObjectStore};
use providers::local::FileStoreProvider;
use providers::memory::MemoryStoreProvider;
use shellexpand::tilde;
use snafu::location;
use tokio::io::AsyncWriteExt;
use url::Url;

use super::local::LocalObjectReader;
mod list_retry;
pub mod providers;
mod tracing;
use crate::object_reader::SmallReader;
use crate::object_writer::WriteResult;
use crate::{object_reader::CloudObjectReader, object_writer::ObjectWriter, traits::Reader};
use lance_core::{Error, Result};

// Local disks tend to do fine with a few threads
// Note: the number of threads here also impacts the number of files
// we need to read in some situations.  So keeping this at 8 keeps the
// RAM on our scanner down.
pub const DEFAULT_LOCAL_IO_PARALLELISM: usize = 8;
// Cloud disks often need many many threads to saturate the network
pub const DEFAULT_CLOUD_IO_PARALLELISM: usize = 64;

const DEFAULT_LOCAL_BLOCK_SIZE: usize = 4 * 1024; // 4KB block size
#[cfg(any(feature = "aws", feature = "gcp", feature = "azure"))]
const DEFAULT_CLOUD_BLOCK_SIZE: usize = 64 * 1024; // 64KB block size

pub static DEFAULT_MAX_IOP_SIZE: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| {
    std::env::var("LANCE_MAX_IOP_SIZE")
        .map(|val| val.parse().unwrap())
        .unwrap_or(16 * 1024 * 1024)
});

pub const DEFAULT_DOWNLOAD_RETRY_COUNT: usize = 3;

pub use providers::{ObjectStoreProvider, ObjectStoreRegistry};

#[async_trait]
pub trait ObjectStoreExt {
    /// Returns true if the file exists.
    async fn exists(&self, path: &Path) -> Result<bool>;

    /// Read all files (start from base directory) recursively
    ///
    /// unmodified_since can be specified to only return files that have not been modified since the given time.
    fn read_dir_all<'a, 'b>(
        &'a self,
        dir_path: impl Into<&'b Path> + Send,
        unmodified_since: Option<DateTime<Utc>>,
    ) -> BoxStream<'a, Result<ObjectMeta>>;
}

#[async_trait]
impl<O: OSObjectStore + ?Sized> ObjectStoreExt for O {
    fn read_dir_all<'a, 'b>(
        &'a self,
        dir_path: impl Into<&'b Path> + Send,
        unmodified_since: Option<DateTime<Utc>>,
    ) -> BoxStream<'a, Result<ObjectMeta>> {
        let output = self.list(Some(dir_path.into())).map_err(|e| e.into());
        if let Some(unmodified_since_val) = unmodified_since {
            output
                .try_filter(move |file| future::ready(file.last_modified < unmodified_since_val))
                .boxed()
        } else {
            output.boxed()
        }
    }

    async fn exists(&self, path: &Path) -> Result<bool> {
        match self.head(path).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { path: _, source: _ }) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }
}

/// Wraps [ObjectStore](object_store::ObjectStore)
#[derive(Debug, Clone)]
pub struct ObjectStore {
    // Inner object store
    pub inner: Arc<dyn OSObjectStore>,
    scheme: String,
    block_size: usize,
    max_iop_size: u64,
    /// Whether to use constant size upload parts for multipart uploads. This
    /// is only necessary for Cloudflare R2.
    pub use_constant_size_upload_parts: bool,
    /// Whether we can assume that the list of files is lexically ordered. This
    /// is true for object stores, but not for local filesystems.
    pub list_is_lexically_ordered: bool,
    io_parallelism: usize,
    /// Number of times to retry a failed download
    download_retry_count: usize,
}

impl DeepSizeOf for ObjectStore {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        // We aren't counting `inner` here which is problematic but an ObjectStore
        // shouldn't be too big.  The only exception might be the write cache but, if
        // the writer cache has data, it means we're using it somewhere else that isn't
        // a cache and so that doesn't really count.
        self.scheme.deep_size_of_children(context) + self.block_size.deep_size_of_children(context)
    }
}

impl std::fmt::Display for ObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ObjectStore({})", self.scheme)
    }
}

pub trait WrappingObjectStore: std::fmt::Debug + Send + Sync {
    fn wrap(&self, original: Arc<dyn OSObjectStore>) -> Arc<dyn OSObjectStore>;
}

#[derive(Debug, Clone)]
pub struct ChainedWrappingObjectStore {
    wrappers: Vec<Arc<dyn WrappingObjectStore>>,
}

impl ChainedWrappingObjectStore {
    pub fn new(wrappers: Vec<Arc<dyn WrappingObjectStore>>) -> Self {
        Self { wrappers }
    }

    pub fn add_wrapper(&mut self, wrapper: Arc<dyn WrappingObjectStore>) {
        self.wrappers.push(wrapper);
    }
}

impl WrappingObjectStore for ChainedWrappingObjectStore {
    fn wrap(&self, original: Arc<dyn OSObjectStore>) -> Arc<dyn OSObjectStore> {
        self.wrappers
            .iter()
            .fold(original, |acc, wrapper| wrapper.wrap(acc))
    }
}

/// Parameters to create an [ObjectStore]
///
#[derive(Debug, Clone)]
pub struct ObjectStoreParams {
    pub block_size: Option<usize>,
    #[deprecated(note = "Implement an ObjectStoreProvider instead")]
    pub object_store: Option<(Arc<DynObjectStore>, Url)>,
    pub s3_credentials_refresh_offset: Duration,
    #[cfg(feature = "aws")]
    pub aws_credentials: Option<AwsCredentialProvider>,
    pub object_store_wrapper: Option<Arc<dyn WrappingObjectStore>>,
    pub storage_options: Option<HashMap<String, String>>,
    /// Use constant size upload parts for multipart uploads. Only necessary
    /// for Cloudflare R2, which doesn't support variable size parts. When this
    /// is false, max upload size is 2.5TB. When this is true, the max size is
    /// 50GB.
    pub use_constant_size_upload_parts: bool,
    pub list_is_lexically_ordered: Option<bool>,
}

impl Default for ObjectStoreParams {
    fn default() -> Self {
        #[allow(deprecated)]
        Self {
            object_store: None,
            block_size: None,
            s3_credentials_refresh_offset: Duration::from_secs(60),
            #[cfg(feature = "aws")]
            aws_credentials: None,
            object_store_wrapper: None,
            storage_options: None,
            use_constant_size_upload_parts: false,
            list_is_lexically_ordered: None,
        }
    }
}

// We implement hash for caching
impl std::hash::Hash for ObjectStoreParams {
    #[allow(deprecated)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // For hashing, we use pointer values for ObjectStore, S3 credentials, and wrapper
        self.block_size.hash(state);
        if let Some((store, url)) = &self.object_store {
            Arc::as_ptr(store).hash(state);
            url.hash(state);
        }
        self.s3_credentials_refresh_offset.hash(state);
        #[cfg(feature = "aws")]
        if let Some(aws_credentials) = &self.aws_credentials {
            Arc::as_ptr(aws_credentials).hash(state);
        }
        if let Some(wrapper) = &self.object_store_wrapper {
            Arc::as_ptr(wrapper).hash(state);
        }
        if let Some(storage_options) = &self.storage_options {
            for (key, value) in storage_options {
                key.hash(state);
                value.hash(state);
            }
        }
        self.use_constant_size_upload_parts.hash(state);
        self.list_is_lexically_ordered.hash(state);
    }
}

// We implement eq for caching
impl Eq for ObjectStoreParams {}
impl PartialEq for ObjectStoreParams {
    #[allow(deprecated)]
    fn eq(&self, other: &Self) -> bool {
        // For equality, we use pointer comparison for ObjectStore, S3 credentials, and wrapper
        self.block_size == other.block_size
            && self
                .object_store
                .as_ref()
                .map(|(store, url)| (Arc::as_ptr(store), url))
                == other
                    .object_store
                    .as_ref()
                    .map(|(store, url)| (Arc::as_ptr(store), url))
            && self.s3_credentials_refresh_offset == other.s3_credentials_refresh_offset
            && self.aws_credentials.as_ref().map(Arc::as_ptr)
                == other.aws_credentials.as_ref().map(Arc::as_ptr)
            && self.object_store_wrapper.as_ref().map(Arc::as_ptr)
                == other.object_store_wrapper.as_ref().map(Arc::as_ptr)
            && self.storage_options == other.storage_options
            && self.use_constant_size_upload_parts == other.use_constant_size_upload_parts
            && self.list_is_lexically_ordered == other.list_is_lexically_ordered
    }
}

fn uri_to_url(uri: &str) -> Result<Url> {
    match Url::parse(uri) {
        Ok(url) if url.scheme().len() == 1 && cfg!(windows) => {
            // On Windows, the drive is parsed as a scheme
            local_path_to_url(uri)
        }
        Ok(url) => Ok(url),
        Err(_) => local_path_to_url(uri),
    }
}

fn expand_path(str_path: impl AsRef<str>) -> Result<std::path::PathBuf> {
    let expanded = tilde(str_path.as_ref()).to_string();

    let mut expanded_path = path_abs::PathAbs::new(expanded)
        .unwrap()
        .as_path()
        .to_path_buf();
    // path_abs::PathAbs::new(".") returns an empty string.
    if let Some(s) = expanded_path.as_path().to_str() {
        if s.is_empty() {
            expanded_path = std::env::current_dir()?;
        }
    }

    Ok(expanded_path)
}

fn local_path_to_url(str_path: &str) -> Result<Url> {
    let expanded_path = expand_path(str_path)?;

    Url::from_directory_path(expanded_path).map_err(|_| Error::InvalidInput {
        source: format!("Invalid table location: '{}'", str_path).into(),
        location: location!(),
    })
}

impl ObjectStore {
    /// Parse from a string URI.
    ///
    /// Returns the ObjectStore instance and the absolute path to the object.
    ///
    /// This uses the default [ObjectStoreRegistry] to find the object store. To
    /// allow for potential re-use of object store instances, it's recommended to
    /// create a shared [ObjectStoreRegistry] and pass that to [Self::from_uri_and_params].
    pub async fn from_uri(uri: &str) -> Result<(Arc<Self>, Path)> {
        let registry = Arc::new(ObjectStoreRegistry::default());

        Self::from_uri_and_params(registry, uri, &ObjectStoreParams::default()).await
    }

    /// Parse from a string URI.
    ///
    /// Returns the ObjectStore instance and the absolute path to the object.
    pub async fn from_uri_and_params(
        registry: Arc<ObjectStoreRegistry>,
        uri: &str,
        params: &ObjectStoreParams,
    ) -> Result<(Arc<Self>, Path)> {
        #[allow(deprecated)]
        if let Some((store, path)) = params.object_store.as_ref() {
            let mut inner = store.clone();
            if let Some(wrapper) = params.object_store_wrapper.as_ref() {
                inner = wrapper.wrap(inner);
            }
            let store = Self {
                inner,
                scheme: path.scheme().to_string(),
                block_size: params.block_size.unwrap_or(64 * 1024),
                max_iop_size: *DEFAULT_MAX_IOP_SIZE,
                use_constant_size_upload_parts: params.use_constant_size_upload_parts,
                list_is_lexically_ordered: params.list_is_lexically_ordered.unwrap_or_default(),
                io_parallelism: DEFAULT_CLOUD_IO_PARALLELISM,
                download_retry_count: DEFAULT_DOWNLOAD_RETRY_COUNT,
            };
            let path = Path::from(path.path());
            return Ok((Arc::new(store), path));
        }
        let url = uri_to_url(uri)?;
        let store = registry.get_store(url.clone(), params).await?;
        // We know the scheme is valid if we got a store back.
        let provider = registry.get_provider(url.scheme()).expect_ok()?;
        let path = provider.extract_path(&url);

        Ok((store, path))
    }

    #[deprecated(note = "Use `from_uri` instead")]
    pub fn from_path(str_path: &str) -> Result<(Arc<Self>, Path)> {
        Self::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            str_path,
            &Default::default(),
        )
        .now_or_never()
        .unwrap()
    }

    /// Local object store.
    pub fn local() -> Self {
        let provider = FileStoreProvider;
        provider
            .new_store(Url::parse("file:///").unwrap(), &Default::default())
            .now_or_never()
            .unwrap()
            .unwrap()
    }

    /// Create a in-memory object store directly for testing.
    pub fn memory() -> Self {
        let provider = MemoryStoreProvider;
        provider
            .new_store(Url::parse("memory:///").unwrap(), &Default::default())
            .now_or_never()
            .unwrap()
            .unwrap()
    }

    /// Returns true if the object store pointed to a local file system.
    pub fn is_local(&self) -> bool {
        self.scheme == "file"
    }

    pub fn is_cloud(&self) -> bool {
        self.scheme != "file" && self.scheme != "memory"
    }

    pub fn scheme(&self) -> &str {
        &self.scheme
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn max_iop_size(&self) -> u64 {
        self.max_iop_size
    }

    pub fn io_parallelism(&self) -> usize {
        std::env::var("LANCE_IO_THREADS")
            .map(|val| val.parse::<usize>().unwrap())
            .unwrap_or(self.io_parallelism)
    }

    /// Open a file for path.
    ///
    /// Parameters
    /// - ``path``: Absolute path to the file.
    pub async fn open(&self, path: &Path) -> Result<Box<dyn Reader>> {
        match self.scheme.as_str() {
            "file" => LocalObjectReader::open(path, self.block_size, None).await,
            _ => Ok(Box::new(CloudObjectReader::new(
                self.inner.clone(),
                path.clone(),
                self.block_size,
                None,
                self.download_retry_count,
            )?)),
        }
    }

    /// Open a reader for a file with known size.
    ///
    /// This size may either have been retrieved from a list operation or
    /// cached metadata. By passing in the known size, we can skip a HEAD / metadata
    /// call.
    pub async fn open_with_size(&self, path: &Path, known_size: usize) -> Result<Box<dyn Reader>> {
        // If we know the file is really small, we can read the whole thing
        // as a single request.
        if known_size <= self.block_size {
            return Ok(Box::new(SmallReader::new(
                self.inner.clone(),
                path.clone(),
                self.download_retry_count,
                known_size,
            )));
        }

        match self.scheme.as_str() {
            "file" => LocalObjectReader::open(path, self.block_size, Some(known_size)).await,
            _ => Ok(Box::new(CloudObjectReader::new(
                self.inner.clone(),
                path.clone(),
                self.block_size,
                Some(known_size),
                self.download_retry_count,
            )?)),
        }
    }

    /// Create an [ObjectWriter] from local [std::path::Path]
    pub async fn create_local_writer(path: &std::path::Path) -> Result<ObjectWriter> {
        let object_store = Self::local();
        let absolute_path = expand_path(path.to_string_lossy())?;
        let os_path = Path::from_absolute_path(absolute_path)?;
        object_store.create(&os_path).await
    }

    /// Open an [Reader] from local [std::path::Path]
    pub async fn open_local(path: &std::path::Path) -> Result<Box<dyn Reader>> {
        let object_store = Self::local();
        let absolute_path = expand_path(path.to_string_lossy())?;
        let os_path = Path::from_absolute_path(absolute_path)?;
        object_store.open(&os_path).await
    }

    /// Create a new file.
    pub async fn create(&self, path: &Path) -> Result<ObjectWriter> {
        ObjectWriter::new(self, path).await
    }

    /// A helper function to create a file and write content to it.
    pub async fn put(&self, path: &Path, content: &[u8]) -> Result<WriteResult> {
        let mut writer = self.create(path).await?;
        writer.write_all(content).await?;
        writer.shutdown().await
    }

    pub async fn delete(&self, path: &Path) -> Result<()> {
        self.inner.delete(path).await?;
        Ok(())
    }

    pub async fn copy(&self, from: &Path, to: &Path) -> Result<()> {
        if self.is_local() {
            // Use std::fs::copy for local filesystem to support cross-filesystem copies
            return super::local::copy_file(from, to);
        }
        Ok(self.inner.copy(from, to).await?)
    }

    /// Read a directory (start from base directory) and returns all sub-paths in the directory.
    pub async fn read_dir(&self, dir_path: impl Into<Path>) -> Result<Vec<String>> {
        let path = dir_path.into();
        let path = Path::parse(&path)?;
        let output = self.inner.list_with_delimiter(Some(&path)).await?;
        Ok(output
            .common_prefixes
            .iter()
            .chain(output.objects.iter().map(|o| &o.location))
            .map(|s| s.filename().unwrap().to_string())
            .collect())
    }

    pub fn list(
        &self,
        path: Option<Path>,
    ) -> Pin<Box<dyn Stream<Item = Result<ObjectMeta>> + Send>> {
        Box::pin(ListRetryStream::new(self.inner.clone(), path, 5).map(|m| m.map_err(|e| e.into())))
    }

    /// Read all files (start from base directory) recursively
    ///
    /// unmodified_since can be specified to only return files that have not been modified since the given time.
    pub fn read_dir_all<'a, 'b>(
        &'a self,
        dir_path: impl Into<&'b Path> + Send,
        unmodified_since: Option<DateTime<Utc>>,
    ) -> BoxStream<'a, Result<ObjectMeta>> {
        self.inner.read_dir_all(dir_path, unmodified_since)
    }

    /// Remove a directory recursively.
    pub async fn remove_dir_all(&self, dir_path: impl Into<Path>) -> Result<()> {
        let path = dir_path.into();
        let path = Path::parse(&path)?;

        if self.is_local() {
            // Local file system needs to delete directories as well.
            return super::local::remove_dir_all(&path);
        }
        let sub_entries = self
            .inner
            .list(Some(&path))
            .map(|m| m.map(|meta| meta.location))
            .boxed();
        self.inner
            .delete_stream(sub_entries)
            .try_collect::<Vec<_>>()
            .await?;
        Ok(())
    }

    pub fn remove_stream<'a>(
        &'a self,
        locations: BoxStream<'a, Result<Path>>,
    ) -> BoxStream<'a, Result<Path>> {
        self.inner
            .delete_stream(locations.err_into::<ObjectStoreError>().boxed())
            .err_into::<Error>()
            .boxed()
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
    pub async fn size(&self, path: &Path) -> Result<u64> {
        Ok(self.inner.head(path).await?.size)
    }

    /// Convenience function to open a reader and read all the bytes
    pub async fn read_one_all(&self, path: &Path) -> Result<Bytes> {
        let reader = self.open(path).await?;
        Ok(reader.get_all().await?)
    }

    /// Convenience function open a reader and make a single request
    ///
    /// If you will be making multiple requests to the path it is more efficient to call [`Self::open`]
    /// and then call [`Reader::get_range`] multiple times.
    pub async fn read_one_range(&self, path: &Path, range: Range<usize>) -> Result<Bytes> {
        let reader = self.open(path).await?;
        Ok(reader.get_range(range).await?)
    }
}

/// Options that can be set for multiple object stores
#[derive(PartialEq, Eq, Hash, Clone, Debug, Copy)]
pub enum LanceConfigKey {
    /// Number of times to retry a download that fails
    DownloadRetryCount,
}

impl FromStr for LanceConfigKey {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "download_retry_count" => Ok(Self::DownloadRetryCount),
            _ => Err(Error::InvalidInput {
                source: format!("Invalid LanceConfigKey: {}", s).into(),
                location: location!(),
            }),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StorageOptions(pub HashMap<String, String>);

impl StorageOptions {
    /// Create a new instance of [`StorageOptions`]
    pub fn new(options: HashMap<String, String>) -> Self {
        let mut options = options;
        if let Ok(value) = std::env::var("AZURE_STORAGE_ALLOW_HTTP") {
            options.insert("allow_http".into(), value);
        }
        if let Ok(value) = std::env::var("AZURE_STORAGE_USE_HTTP") {
            options.insert("allow_http".into(), value);
        }
        if let Ok(value) = std::env::var("AWS_ALLOW_HTTP") {
            options.insert("allow_http".into(), value);
        }
        if let Ok(value) = std::env::var("OBJECT_STORE_CLIENT_MAX_RETRIES") {
            options.insert("client_max_retries".into(), value);
        }
        if let Ok(value) = std::env::var("OBJECT_STORE_CLIENT_RETRY_TIMEOUT") {
            options.insert("client_retry_timeout".into(), value);
        }
        Self(options)
    }

    /// Denotes if unsecure connections via http are allowed
    pub fn allow_http(&self) -> bool {
        self.0.iter().any(|(key, value)| {
            key.to_ascii_lowercase().contains("allow_http") & str_is_truthy(value)
        })
    }

    /// Number of times to retry a download that fails
    pub fn download_retry_count(&self) -> usize {
        self.0
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case("download_retry_count"))
            .map(|(_, value)| value.parse::<usize>().unwrap_or(3))
            .unwrap_or(3)
    }

    /// Max retry times to set in RetryConfig for object store client
    pub fn client_max_retries(&self) -> usize {
        self.0
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case("client_max_retries"))
            .and_then(|(_, value)| value.parse::<usize>().ok())
            .unwrap_or(10)
    }

    /// Seconds of timeout to set in RetryConfig for object store client
    pub fn client_retry_timeout(&self) -> u64 {
        self.0
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case("client_retry_timeout"))
            .and_then(|(_, value)| value.parse::<u64>().ok())
            .unwrap_or(180)
    }

    pub fn get(&self, key: &str) -> Option<&String> {
        self.0.get(key)
    }
}

impl From<HashMap<String, String>> for StorageOptions {
    fn from(value: HashMap<String, String>) -> Self {
        Self::new(value)
    }
}

impl ObjectStore {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        store: Arc<DynObjectStore>,
        location: Url,
        block_size: Option<usize>,
        wrapper: Option<Arc<dyn WrappingObjectStore>>,
        use_constant_size_upload_parts: bool,
        list_is_lexically_ordered: bool,
        io_parallelism: usize,
        download_retry_count: usize,
    ) -> Self {
        let scheme = location.scheme();
        let block_size = block_size.unwrap_or_else(|| infer_block_size(scheme));

        let store = match wrapper {
            Some(wrapper) => wrapper.wrap(store),
            None => store,
        };

        Self {
            inner: store,
            scheme: scheme.into(),
            block_size,
            max_iop_size: *DEFAULT_MAX_IOP_SIZE,
            use_constant_size_upload_parts,
            list_is_lexically_ordered,
            io_parallelism,
            download_retry_count,
        }
    }
}

fn infer_block_size(scheme: &str) -> usize {
    // Block size: On local file systems, we use 4KB block size. On cloud
    // object stores, we use 64KB block size. This is generally the largest
    // block size where we don't see a latency penalty.
    match scheme {
        "file" => 4 * 1024,
        _ => 64 * 1024,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;
    use rstest::rstest;
    use std::env::set_current_dir;
    use std::fs::{create_dir_all, write};
    use std::path::Path as StdPath;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Write test content to file.
    fn write_to_file(path_str: &str, contents: &str) -> std::io::Result<()> {
        let expanded = tilde(path_str).to_string();
        let path = StdPath::new(&expanded);
        std::fs::create_dir_all(path.parent().unwrap())?;
        write(path, contents)
    }

    async fn read_from_store(store: &ObjectStore, path: &Path) -> Result<String> {
        let test_file_store = store.open(path).await.unwrap();
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
            &format!("{tmp_path}/bar/foo.lance/test_file"),
            "TEST_CONTENT",
        )
        .unwrap();

        // test a few variations of the same path
        for uri in &[
            format!("{tmp_path}/bar/foo.lance"),
            format!("{tmp_path}/./bar/foo.lance"),
            format!("{tmp_path}/bar/foo.lance/../foo.lance"),
        ] {
            let (store, path) = ObjectStore::from_uri(uri).await.unwrap();
            let contents = read_from_store(store.as_ref(), &path.child("test_file"))
                .await
                .unwrap();
            assert_eq!(contents, "TEST_CONTENT");
        }
    }

    #[tokio::test]
    async fn test_cloud_paths() {
        let uri = "s3://bucket/foo.lance";
        let (store, path) = ObjectStore::from_uri(uri).await.unwrap();
        assert_eq!(store.scheme, "s3");
        assert_eq!(path.to_string(), "foo.lance");

        let (store, path) = ObjectStore::from_uri("s3+ddb://bucket/foo.lance")
            .await
            .unwrap();
        assert_eq!(store.scheme, "s3");
        assert_eq!(path.to_string(), "foo.lance");

        let (store, path) = ObjectStore::from_uri("gs://bucket/foo.lance")
            .await
            .unwrap();
        assert_eq!(store.scheme, "gs");
        assert_eq!(path.to_string(), "foo.lance");
    }

    async fn test_block_size_used_test_helper(
        uri: &str,
        storage_options: Option<HashMap<String, String>>,
        default_expected_block_size: usize,
    ) {
        // Test the default
        let registry = Arc::new(ObjectStoreRegistry::default());
        let params = ObjectStoreParams {
            storage_options: storage_options.clone(),
            ..ObjectStoreParams::default()
        };
        let (store, _) = ObjectStore::from_uri_and_params(registry, uri, &params)
            .await
            .unwrap();
        assert_eq!(store.block_size, default_expected_block_size);

        // Ensure param is used
        let registry = Arc::new(ObjectStoreRegistry::default());
        let params = ObjectStoreParams {
            block_size: Some(1024),
            storage_options: storage_options.clone(),
            ..ObjectStoreParams::default()
        };
        let (store, _) = ObjectStore::from_uri_and_params(registry, uri, &params)
            .await
            .unwrap();
        assert_eq!(store.block_size, 1024);
    }

    #[rstest]
    #[case("s3://bucket/foo.lance", None)]
    #[case("gs://bucket/foo.lance", None)]
    #[case("az://account/bucket/foo.lance",
      Some(HashMap::from([
            (String::from("account_name"), String::from("account")),
            (String::from("container_name"), String::from("container"))
           ])))]
    #[tokio::test]
    async fn test_block_size_used_cloud(
        #[case] uri: &str,
        #[case] storage_options: Option<HashMap<String, String>>,
    ) {
        test_block_size_used_test_helper(uri, storage_options, 64 * 1024).await;
    }

    #[rstest]
    #[case("file")]
    #[case("file-object-store")]
    #[case("memory:///bucket/foo.lance")]
    #[tokio::test]
    async fn test_block_size_used_file(#[case] prefix: &str) {
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path().to_str().unwrap().to_owned();
        let path = format!("{tmp_path}/bar/foo.lance/test_file");
        write_to_file(&path, "URL").unwrap();
        let uri = format!("{prefix}:///{path}");
        test_block_size_used_test_helper(&uri, None, 4 * 1024).await;
    }

    #[tokio::test]
    async fn test_relative_paths() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path().to_str().unwrap().to_owned();
        write_to_file(
            &format!("{tmp_path}/bar/foo.lance/test_file"),
            "RELATIVE_URL",
        )
        .unwrap();

        set_current_dir(StdPath::new(&tmp_path)).expect("Error changing current dir");
        let (store, path) = ObjectStore::from_uri("./bar/foo.lance").await.unwrap();

        let contents = read_from_store(store.as_ref(), &path.child("test_file"))
            .await
            .unwrap();
        assert_eq!(contents, "RELATIVE_URL");
    }

    #[tokio::test]
    async fn test_tilde_expansion() {
        let uri = "~/foo.lance";
        write_to_file(&format!("{uri}/test_file"), "TILDE").unwrap();
        let (store, path) = ObjectStore::from_uri(uri).await.unwrap();
        let contents = read_from_store(store.as_ref(), &path.child("test_file"))
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
    async fn test_delete_directory() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path();
        create_dir_all(path.join("foo").join("bar")).unwrap();
        create_dir_all(path.join("foo").join("zoo")).unwrap();
        create_dir_all(path.join("foo").join("zoo").join("abc")).unwrap();
        write_to_file(
            path.join("foo")
                .join("bar")
                .join("test_file")
                .to_str()
                .unwrap(),
            "delete",
        )
        .unwrap();
        write_to_file(path.join("foo").join("top").to_str().unwrap(), "delete_top").unwrap();
        let (store, base) = ObjectStore::from_uri(path.to_str().unwrap()).await.unwrap();
        store.remove_dir_all(base.child("foo")).await.unwrap();

        assert!(!path.join("foo").exists());
    }

    #[derive(Debug)]
    struct TestWrapper {
        called: AtomicBool,

        return_value: Arc<dyn OSObjectStore>,
    }

    impl WrappingObjectStore for TestWrapper {
        fn wrap(&self, _original: Arc<dyn OSObjectStore>) -> Arc<dyn OSObjectStore> {
            self.called.store(true, Ordering::Relaxed);

            // return a mocked value so we can check if the final store is the one we expect
            self.return_value.clone()
        }
    }

    impl TestWrapper {
        fn called(&self) -> bool {
            self.called.load(Ordering::Relaxed)
        }
    }

    #[tokio::test]
    async fn test_wrapping_object_store_option_is_used() {
        // Make a store for the inner store first
        let mock_inner_store: Arc<dyn OSObjectStore> = Arc::new(InMemory::new());
        let registry = Arc::new(ObjectStoreRegistry::default());

        assert_eq!(Arc::strong_count(&mock_inner_store), 1);

        let wrapper = Arc::new(TestWrapper {
            called: AtomicBool::new(false),
            return_value: mock_inner_store.clone(),
        });

        let params = ObjectStoreParams {
            object_store_wrapper: Some(wrapper.clone()),
            ..ObjectStoreParams::default()
        };

        // not called yet
        assert!(!wrapper.called());

        let _ = ObjectStore::from_uri_and_params(registry, "memory:///", &params)
            .await
            .unwrap();

        // called after construction
        assert!(wrapper.called());

        // hard to compare two trait pointers as the point to vtables
        // using the ref count as a proxy to make sure that the store is correctly kept
        assert_eq!(Arc::strong_count(&mock_inner_store), 2);
    }

    #[tokio::test]
    async fn test_local_paths() {
        let temp_dir = tempfile::tempdir().unwrap();

        let file_path = temp_dir.path().join("test_file");
        let mut writer = ObjectStore::create_local_writer(file_path.as_path())
            .await
            .unwrap();
        writer.write_all(b"LOCAL").await.unwrap();
        writer.shutdown().await.unwrap();

        let reader = ObjectStore::open_local(file_path.as_path()).await.unwrap();
        let buf = reader.get_range(0..5).await.unwrap();
        assert_eq!(buf.as_ref(), b"LOCAL");
    }

    #[tokio::test]
    async fn test_read_one() {
        let temp_dir = tempfile::tempdir().unwrap();

        let file_path = temp_dir.path().join("test_file");
        let mut writer = ObjectStore::create_local_writer(file_path.as_path())
            .await
            .unwrap();
        writer.write_all(b"LOCAL").await.unwrap();
        writer.shutdown().await.unwrap();

        let file_path_os = object_store::path::Path::parse(file_path.to_str().unwrap()).unwrap();
        let obj_store = ObjectStore::local();
        let buf = obj_store.read_one_all(&file_path_os).await.unwrap();
        assert_eq!(buf.as_ref(), b"LOCAL");

        let buf = obj_store.read_one_range(&file_path_os, 0..5).await.unwrap();
        assert_eq!(buf.as_ref(), b"LOCAL");
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
            let contents = read_from_store(store.as_ref(), &base.child("test_file"))
                .await
                .unwrap();
            assert_eq!(contents, "WINDOWS");
        }
    }

    #[tokio::test]
    async fn test_cross_filesystem_copy() {
        // Create two temporary directories that simulate different filesystems
        let source_dir = tempfile::tempdir().unwrap();
        let dest_dir = tempfile::tempdir().unwrap();

        // Create a test file in the source directory
        let source_file_name = "test_file.txt";
        let source_file = source_dir.path().join(source_file_name);
        std::fs::write(&source_file, b"test content").unwrap();

        // Create ObjectStore for local filesystem
        let (store, base_path) = ObjectStore::from_uri(source_dir.path().to_str().unwrap())
            .await
            .unwrap();

        // Create paths relative to the ObjectStore base
        let from_path = base_path.child(source_file_name);

        // Use object_store::Path::parse for the destination
        let dest_file = dest_dir.path().join("copied_file.txt");
        let dest_str = dest_file.to_str().unwrap();
        let to_path = object_store::path::Path::parse(dest_str).unwrap();

        // Perform the copy operation
        store.copy(&from_path, &to_path).await.unwrap();

        // Verify the file was copied correctly
        assert!(dest_file.exists());
        let copied_content = std::fs::read(&dest_file).unwrap();
        assert_eq!(copied_content, b"test content");
    }

    #[tokio::test]
    async fn test_copy_creates_parent_directories() {
        let source_dir = tempfile::tempdir().unwrap();
        let dest_dir = tempfile::tempdir().unwrap();

        // Create a test file in the source directory
        let source_file_name = "test_file.txt";
        let source_file = source_dir.path().join(source_file_name);
        std::fs::write(&source_file, b"test content").unwrap();

        // Create ObjectStore for local filesystem
        let (store, base_path) = ObjectStore::from_uri(source_dir.path().to_str().unwrap())
            .await
            .unwrap();

        // Create paths
        let from_path = base_path.child(source_file_name);

        // Create destination with nested directories that don't exist yet
        let dest_file = dest_dir
            .path()
            .join("nested")
            .join("dirs")
            .join("copied_file.txt");
        let dest_str = dest_file.to_str().unwrap();
        let to_path = object_store::path::Path::parse(dest_str).unwrap();

        // Perform the copy operation - should create parent directories
        store.copy(&from_path, &to_path).await.unwrap();

        // Verify the file was copied correctly and directories were created
        assert!(dest_file.exists());
        assert!(dest_file.parent().unwrap().exists());
        let copied_content = std::fs::read(&dest_file).unwrap();
        assert_eq!(copied_content, b"test content");
    }
}
