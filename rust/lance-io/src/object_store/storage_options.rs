// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dynamic storage options object store wrapper
//!
//! This module provides an ObjectStore wrapper that automatically refreshes
//! storage options (such as credentials) from a StorageOptionsProvider.

use std::collections::HashMap;
use std::fmt;
use std::ops::Range;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::Mutex;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use lance_namespace::LanceNamespace;
use object_store::{
    path::Path, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore as OSObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    Result as ObjectStoreResult,
};
use snafu::location;

use crate::{Error, Result};

use super::WrappingObjectStore;

/// Trait for providing storage options with expiration tracking
///
/// Implementations can fetch storage options (including credentials) from various
/// sources (namespace servers, secret managers, etc.) and are usable from Python/Java via FFI.
#[async_trait]
pub trait StorageOptionsProvider: Send + Sync {
    /// Fetch fresh storage options
    ///
    /// Returns a tuple of (storage_options, expires_at_millis)
    /// where storage_options is a map of key-value pairs (e.g., AWS credentials)
    /// and expires_at_millis is the epoch time in milliseconds when the options expire
    async fn get_storage_options(&self) -> Result<(HashMap<String, String>, u64)>;
}

/// StorageOptionsProvider implementation that fetches options from a LanceNamespace
pub struct LanceNamespaceStorageOptionsProvider {
    namespace: Arc<dyn LanceNamespace>,
    table_id: Vec<String>,
}

impl LanceNamespaceStorageOptionsProvider {
    /// Create a new LanceNamespaceStorageOptionsProvider
    ///
    /// # Arguments
    /// * `namespace` - The namespace implementation to fetch storage options from
    /// * `table_id` - The table identifier
    pub fn new(namespace: Arc<dyn LanceNamespace>, table_id: Vec<String>) -> Self {
        Self {
            namespace,
            table_id,
        }
    }
}

#[async_trait]
impl StorageOptionsProvider for LanceNamespaceStorageOptionsProvider {
    async fn get_storage_options(&self) -> Result<(HashMap<String, String>, u64)> {
        use lance_namespace::models::DescribeTableRequest;

        let request = DescribeTableRequest {
            id: Some(self.table_id.clone()),
            version: None,
        };

        let response = self
            .namespace
            .describe_table(request)
            .await
            .map_err(|e| Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "Failed to fetch credentials: {}",
                    e
                ))),
                location: location!(),
            })?;

        let storage_options = response.storage_options.ok_or_else(|| Error::IO {
            source: Box::new(std::io::Error::other(
                "storage_options not found in describe_table response",
            )),
            location: location!(),
        })?;

        let expires_at_millis = storage_options
            .get("expires_at_millis")
            .and_then(|s| s.parse::<u64>().ok())
            .ok_or_else(|| Error::IO {
                source: Box::new(std::io::Error::other(
                    "expires_at_millis is required in storage_options",
                )),
                location: location!(),
            })?;

        Ok((storage_options, expires_at_millis))
    }
}

/// Configuration parameters for credential vending
#[derive(Debug, Clone)]
pub struct StorageOptionsProviderParams {
    /// How early to refresh credentials before expiration (in milliseconds)
    /// Default: 300,000 (5 minutes)
    pub refresh_lead_time_ms: u64,

    /// Initial storage options to use (avoids initial describe_table call)
    /// If provided, the wrapper will use these credentials immediately
    pub initial_storage_options: Option<HashMap<String, String>>,
}

impl Default for StorageOptionsProviderParams {
    fn default() -> Self {
        Self {
            refresh_lead_time_ms: 300_000, // 5 minutes
            initial_storage_options: None,
        }
    }
}

impl StorageOptionsProviderParams {
    /// Create new params with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the refresh lead time in milliseconds
    pub fn with_refresh_lead_time_ms(mut self, ms: u64) -> Self {
        self.refresh_lead_time_ms = ms;
        self
    }

    /// Set initial storage options to avoid first describe_table call
    pub fn with_initial_storage_options(mut self, options: HashMap<String, String>) -> Self {
        self.initial_storage_options = Some(options);
        self
    }
}

/// Cache for credentials with expiration tracking
#[derive(Debug, Clone)]
struct CredentialsCache {
    storage_options: HashMap<String, String>,
    expires_at_millis: u64,
    last_refresh: Instant,
    initialized: bool,
}

impl CredentialsCache {
    fn new() -> Self {
        Self {
            storage_options: HashMap::new(),
            expires_at_millis: 0,
            last_refresh: Instant::now(),
            initialized: false,
        }
    }
}

/// Wrapper that provides credential vending for ObjectStore
///
/// This wrapper automatically refreshes credentials from a StorageOptionsProvider
/// implementation before they expire.
#[derive(Clone)]
pub struct DynamicStorageOptionObjectStore {
    vendor: Arc<dyn StorageOptionsProvider>,
    params: StorageOptionsProviderParams,
    credentials: Arc<RwLock<CredentialsCache>>,
    refresh_lock: Arc<Mutex<()>>,
}

impl fmt::Debug for DynamicStorageOptionObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DynamicStorageOptionObjectStore")
            .field("params", &self.params)
            .finish()
    }
}

impl DynamicStorageOptionObjectStore {
    /// Create a new credential vending wrapper
    ///
    /// # Arguments
    /// * `vendor` - The credential vendor implementation to fetch credentials from
    /// * `params` - Configuration parameters for credential vending
    pub fn new(vendor: Arc<dyn StorageOptionsProvider>, params: StorageOptionsProviderParams) -> Self {
        let credentials = if let Some(initial_options) = &params.initial_storage_options {
            // Initialize with provided credentials - expires_at_millis is required
            let expires_at_millis = initial_options
                .get("expires_at_millis")
                .and_then(|s| s.parse::<u64>().ok())
                .expect("expires_at_millis is required in storage_options");

            Arc::new(RwLock::new(CredentialsCache {
                storage_options: initial_options.clone(),
                expires_at_millis,
                last_refresh: Instant::now(),
                initialized: true,
            }))
        } else {
            Arc::new(RwLock::new(CredentialsCache::new()))
        };

        Self {
            vendor,
            params,
            credentials,
            refresh_lock: Arc::new(Mutex::new(())),
        }
    }

    /// Refresh credentials from the vendor
    async fn refresh_credentials(&self) -> Result<()> {
        let (storage_options, expires_at_millis) = self.vendor.get_storage_options().await?;

        let mut cache = self.credentials.write().unwrap();
        cache.storage_options = storage_options;
        cache.expires_at_millis = expires_at_millis;
        cache.last_refresh = Instant::now();
        cache.initialized = true;

        Ok(())
    }

    /// Ensure credentials are fresh, refreshing if necessary
    async fn ensure_fresh_credentials(&self) -> Result<()> {
        // Minimum interval between refreshes (100ms) to prevent refresh storms
        const MIN_REFRESH_INTERVAL_MS: u64 = 100;

        // Quick read-lock check
        {
            let cache = self.credentials.read().unwrap();
            if cache.initialized {
                // Don't refresh if we just refreshed very recently
                let time_since_refresh = cache.last_refresh.elapsed().as_millis() as u64;
                if time_since_refresh < MIN_REFRESH_INTERVAL_MS {
                    return Ok(());
                }

                let now_millis = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or(Duration::from_secs(0))
                    .as_millis() as u64;

                // No need to refresh if not within lead time of expiration
                if now_millis + self.params.refresh_lead_time_ms < cache.expires_at_millis {
                    return Ok(());
                }
            }
        }

        // Need to refresh - acquire lock to ensure only one refresh happens
        let _guard = self.refresh_lock.lock().await;

        // Check again after acquiring lock - another thread might have refreshed
        {
            let cache = self.credentials.read().unwrap();
            if cache.initialized {
                // Don't refresh if we just refreshed very recently
                let time_since_refresh = cache.last_refresh.elapsed().as_millis() as u64;
                if time_since_refresh < MIN_REFRESH_INTERVAL_MS {
                    return Ok(());
                }

                let now_millis = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or(Duration::from_secs(0))
                    .as_millis() as u64;

                // No need to refresh if not within lead time of expiration
                if now_millis + self.params.refresh_lead_time_ms < cache.expires_at_millis {
                    return Ok(());
                }
            }
        }

        // Actually refresh now
        self.refresh_credentials().await?;
        Ok(())
    }
}

impl WrappingObjectStore for DynamicStorageOptionObjectStore {
    fn wrap(
        &self,
        original: Arc<dyn OSObjectStore>,
        _storage_options: Option<&HashMap<String, String>>,
    ) -> Arc<dyn OSObjectStore> {
        Arc::new(DelegatingObjectStore {
            wrapper: Arc::new(self.clone()),
            inner: original,
        })
    }
}

/// Delegating ObjectStore that auto-refreshes credentials
pub struct DelegatingObjectStore {
    wrapper: Arc<DynamicStorageOptionObjectStore>,
    inner: Arc<dyn OSObjectStore>,
}

impl fmt::Debug for DelegatingObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DelegatingObjectStore")
            .field("wrapper", &self.wrapper)
            .finish()
    }
}

impl fmt::Display for DelegatingObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DelegatingObjectStore(CredentialVending)")
    }
}

#[async_trait]
impl OSObjectStore for DelegatingObjectStore {
    async fn put(&self, location: &Path, payload: PutPayload) -> ObjectStoreResult<PutResult> {
        self.wrapper.ensure_fresh_credentials().await.map_err(|e| {
            object_store::Error::Generic {
                store: "CredentialVending",
                source: Box::new(e),
            }
        })?;
        self.inner.put(location, payload).await
    }

    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> ObjectStoreResult<PutResult> {
        self.wrapper.ensure_fresh_credentials().await.map_err(|e| {
            object_store::Error::Generic {
                store: "CredentialVending",
                source: Box::new(e),
            }
        })?;
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart(&self, location: &Path) -> ObjectStoreResult<Box<dyn MultipartUpload>> {
        self.wrapper.ensure_fresh_credentials().await.map_err(|e| {
            object_store::Error::Generic {
                store: "CredentialVending",
                source: Box::new(e),
            }
        })?;
        self.inner.put_multipart(location).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> ObjectStoreResult<Box<dyn MultipartUpload>> {
        self.wrapper.ensure_fresh_credentials().await.map_err(|e| {
            object_store::Error::Generic {
                store: "CredentialVending",
                source: Box::new(e),
            }
        })?;
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get(&self, location: &Path) -> ObjectStoreResult<GetResult> {
        self.wrapper.ensure_fresh_credentials().await.map_err(|e| {
            object_store::Error::Generic {
                store: "CredentialVending",
                source: Box::new(e),
            }
        })?;
        self.inner.get(location).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> ObjectStoreResult<GetResult> {
        self.wrapper.ensure_fresh_credentials().await.map_err(|e| {
            object_store::Error::Generic {
                store: "CredentialVending",
                source: Box::new(e),
            }
        })?;
        self.inner.get_opts(location, options).await
    }

    async fn get_range(&self, location: &Path, range: Range<u64>) -> ObjectStoreResult<Bytes> {
        self.wrapper.ensure_fresh_credentials().await.map_err(|e| {
            object_store::Error::Generic {
                store: "CredentialVending",
                source: Box::new(e),
            }
        })?;
        self.inner.get_range(location, range).await
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> ObjectStoreResult<Vec<Bytes>> {
        self.wrapper.ensure_fresh_credentials().await.map_err(|e| {
            object_store::Error::Generic {
                store: "CredentialVending",
                source: Box::new(e),
            }
        })?;
        self.inner.get_ranges(location, ranges).await
    }

    async fn head(&self, location: &Path) -> ObjectStoreResult<ObjectMeta> {
        self.wrapper.ensure_fresh_credentials().await.map_err(|e| {
            object_store::Error::Generic {
                store: "CredentialVending",
                source: Box::new(e),
            }
        })?;
        self.inner.head(location).await
    }

    async fn delete(&self, location: &Path) -> ObjectStoreResult<()> {
        self.wrapper.ensure_fresh_credentials().await.map_err(|e| {
            object_store::Error::Generic {
                store: "CredentialVending",
                source: Box::new(e),
            }
        })?;
        self.inner.delete(location).await
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, ObjectStoreResult<ObjectMeta>> {
        // Note: We can't easily refresh credentials here since list returns a stream
        // The credentials should be refreshed on the next operation if needed
        self.inner.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, ObjectStoreResult<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> ObjectStoreResult<ListResult> {
        self.wrapper.ensure_fresh_credentials().await.map_err(|e| {
            object_store::Error::Generic {
                store: "CredentialVending",
                source: Box::new(e),
            }
        })?;
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy(&self, from: &Path, to: &Path) -> ObjectStoreResult<()> {
        self.wrapper.ensure_fresh_credentials().await.map_err(|e| {
            object_store::Error::Generic {
                store: "CredentialVending",
                source: Box::new(e),
            }
        })?;
        self.inner.copy(from, to).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> ObjectStoreResult<()> {
        self.wrapper.ensure_fresh_credentials().await.map_err(|e| {
            object_store::Error::Generic {
                store: "CredentialVending",
                source: Box::new(e),
            }
        })?;
        self.inner.rename(from, to).await
    }

    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> ObjectStoreResult<()> {
        self.wrapper.ensure_fresh_credentials().await.map_err(|e| {
            object_store::Error::Generic {
                store: "CredentialVending",
                source: Box::new(e),
            }
        })?;
        self.inner.copy_if_not_exists(from, to).await
    }

    async fn rename_if_not_exists(&self, from: &Path, to: &Path) -> ObjectStoreResult<()> {
        self.wrapper.ensure_fresh_credentials().await.map_err(|e| {
            object_store::Error::Generic {
                store: "CredentialVending",
                source: Box::new(e),
            }
        })?;
        self.inner.rename_if_not_exists(from, to).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_namespace::models::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Mock credential vendor for testing
    struct MockStorageOptionsProvider {
        call_count: Arc<AtomicUsize>,
        expires_at_millis: u64,
    }

    #[async_trait]
    impl StorageOptionsProvider for MockStorageOptionsProvider {
        async fn get_storage_options(&self) -> Result<(HashMap<String, String>, u64)> {
            self.call_count.fetch_add(1, Ordering::SeqCst);

            let mut storage_options = HashMap::new();
            storage_options.insert(
                "expires_at_millis".to_string(),
                self.expires_at_millis.to_string(),
            );
            storage_options.insert("aws_access_key_id".to_string(), "test_key".to_string());

            Ok((storage_options, self.expires_at_millis))
        }
    }

    // Mock namespace for LanceNamespaceStorageOptionsProvider integration tests
    #[allow(dead_code)]
    struct MockNamespace {
        call_count: Arc<AtomicUsize>,
        expires_at_millis: u64,
    }

    #[async_trait]
    impl LanceNamespace for MockNamespace {
        async fn list_namespaces(
            &self,
            _request: ListNamespacesRequest,
        ) -> lance_core::Result<ListNamespacesResponse> {
            unimplemented!()
        }

        async fn describe_namespace(
            &self,
            _request: DescribeNamespaceRequest,
        ) -> lance_core::Result<DescribeNamespaceResponse> {
            unimplemented!()
        }

        async fn create_namespace(
            &self,
            _request: CreateNamespaceRequest,
        ) -> lance_core::Result<CreateNamespaceResponse> {
            unimplemented!()
        }

        async fn drop_namespace(
            &self,
            _request: DropNamespaceRequest,
        ) -> lance_core::Result<DropNamespaceResponse> {
            unimplemented!()
        }

        async fn namespace_exists(
            &self,
            _request: NamespaceExistsRequest,
        ) -> lance_core::Result<()> {
            unimplemented!()
        }

        async fn list_tables(
            &self,
            _request: ListTablesRequest,
        ) -> lance_core::Result<ListTablesResponse> {
            unimplemented!()
        }

        async fn describe_table(
            &self,
            _request: DescribeTableRequest,
        ) -> lance_core::Result<DescribeTableResponse> {
            self.call_count.fetch_add(1, Ordering::SeqCst);

            let mut storage_options = HashMap::new();
            storage_options.insert(
                "expires_at_millis".to_string(),
                self.expires_at_millis.to_string(),
            );
            storage_options.insert("aws_access_key_id".to_string(), "test_key".to_string());

            Ok(DescribeTableResponse {
                version: None,
                location: Some("/test/table".to_string()),
                schema: None,
                properties: None,
                storage_options: Some(storage_options),
            })
        }

        async fn register_table(
            &self,
            _request: RegisterTableRequest,
        ) -> lance_core::Result<RegisterTableResponse> {
            unimplemented!()
        }

        async fn table_exists(
            &self,
            _request: TableExistsRequest,
        ) -> lance_core::Result<()> {
            unimplemented!()
        }

        async fn drop_table(
            &self,
            _request: DropTableRequest,
        ) -> lance_core::Result<DropTableResponse> {
            unimplemented!()
        }

        async fn deregister_table(
            &self,
            _request: DeregisterTableRequest,
        ) -> lance_core::Result<DeregisterTableResponse> {
            unimplemented!()
        }

        async fn count_table_rows(
            &self,
            _request: CountTableRowsRequest,
        ) -> lance_core::Result<i64> {
            unimplemented!()
        }

        async fn create_table(
            &self,
            _request: CreateTableRequest,
            _request_data: Bytes,
        ) -> lance_core::Result<CreateTableResponse> {
            unimplemented!()
        }

        async fn create_empty_table(
            &self,
            _request: CreateEmptyTableRequest,
        ) -> lance_core::Result<CreateEmptyTableResponse> {
            unimplemented!()
        }

        async fn insert_into_table(
            &self,
            _request: InsertIntoTableRequest,
            _request_data: Bytes,
        ) -> lance_core::Result<InsertIntoTableResponse> {
            unimplemented!()
        }

        async fn merge_insert_into_table(
            &self,
            _request: MergeInsertIntoTableRequest,
            _request_data: Bytes,
        ) -> lance_core::Result<MergeInsertIntoTableResponse> {
            unimplemented!()
        }

        async fn update_table(
            &self,
            _request: UpdateTableRequest,
        ) -> lance_core::Result<UpdateTableResponse> {
            unimplemented!()
        }

        async fn delete_from_table(
            &self,
            _request: DeleteFromTableRequest,
        ) -> lance_core::Result<DeleteFromTableResponse> {
            unimplemented!()
        }

        async fn query_table(
            &self,
            _request: QueryTableRequest,
        ) -> lance_core::Result<Bytes> {
            unimplemented!()
        }

        async fn create_table_index(
            &self,
            _request: CreateTableIndexRequest,
        ) -> lance_core::Result<CreateTableIndexResponse> {
            unimplemented!()
        }

        async fn list_table_indices(
            &self,
            _request: ListTableIndicesRequest,
        ) -> lance_core::Result<ListTableIndicesResponse> {
            unimplemented!()
        }

        async fn describe_table_index_stats(
            &self,
            _request: DescribeTableIndexStatsRequest,
        ) -> lance_core::Result<DescribeTableIndexStatsResponse> {
            unimplemented!()
        }

        async fn describe_transaction(
            &self,
            _request: DescribeTransactionRequest,
        ) -> lance_core::Result<DescribeTransactionResponse> {
            unimplemented!()
        }

        async fn alter_transaction(
            &self,
            _request: AlterTransactionRequest,
        ) -> lance_core::Result<AlterTransactionResponse> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn test_credential_refresh() {
        let now_millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let call_count = Arc::new(AtomicUsize::new(0));
        let vendor = Arc::new(MockStorageOptionsProvider {
            call_count: call_count.clone(),
            expires_at_millis: now_millis + 1000, // Expires in 1 second
        });

        let params = StorageOptionsProviderParams::new().with_refresh_lead_time_ms(2000);
        let wrapper = DynamicStorageOptionObjectStore::new(vendor, params);

        // Should trigger refresh since we're within lead time
        wrapper.ensure_fresh_credentials().await.unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 1);

        // Should use cached credentials
        wrapper.ensure_fresh_credentials().await.unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_no_expiration() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let vendor = Arc::new(MockStorageOptionsProvider {
            call_count: call_count.clone(),
            expires_at_millis: 0, // Invalid, will be ignored
        });

        let params = StorageOptionsProviderParams::default();
        let wrapper = DynamicStorageOptionObjectStore::new(vendor, params);

        // First call should still refresh to get initial credentials
        wrapper.ensure_fresh_credentials().await.unwrap();

        // Since there's no valid expiration, subsequent calls won't refresh
        wrapper.ensure_fresh_credentials().await.unwrap();
        wrapper.ensure_fresh_credentials().await.unwrap();

        // Should only call once for initial load
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_initial_storage_options() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let vendor = Arc::new(MockStorageOptionsProvider {
            call_count: call_count.clone(),
            expires_at_millis: 0,
        });

        // Create initial storage options
        let mut initial_options = HashMap::new();
        initial_options.insert("aws_access_key_id".to_string(), "initial_key".to_string());
        initial_options.insert("expires_at_millis".to_string(), "9999999999999".to_string());

        let params =
            StorageOptionsProviderParams::default().with_initial_storage_options(initial_options);

        let wrapper = DynamicStorageOptionObjectStore::new(vendor, params);

        // Should not call get_storage_options since we have initial credentials
        wrapper.ensure_fresh_credentials().await.unwrap();
        wrapper.ensure_fresh_credentials().await.unwrap();
        wrapper.ensure_fresh_credentials().await.unwrap();

        // Should never call get_storage_options
        assert_eq!(call_count.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn test_different_tables_have_isolated_credentials() {
        // This test verifies that different vendors create different wrapper instances
        // and don't share credentials
        let call_count_a = Arc::new(AtomicUsize::new(0));
        let call_count_b = Arc::new(AtomicUsize::new(0));

        let vendor_a = Arc::new(MockStorageOptionsProvider {
            call_count: call_count_a.clone(),
            expires_at_millis: 9999999999999,
        });

        let vendor_b = Arc::new(MockStorageOptionsProvider {
            call_count: call_count_b.clone(),
            expires_at_millis: 9999999999999,
        });

        // Create two wrappers with different vendors
        let wrapper_a =
            DynamicStorageOptionObjectStore::new(vendor_a, StorageOptionsProviderParams::default());

        let wrapper_b =
            DynamicStorageOptionObjectStore::new(vendor_b, StorageOptionsProviderParams::default());

        // Fetch credentials for wrapper A
        wrapper_a.ensure_fresh_credentials().await.unwrap();
        assert_eq!(call_count_a.load(Ordering::SeqCst), 1);
        assert_eq!(call_count_b.load(Ordering::SeqCst), 0);

        // Fetch credentials for wrapper B
        wrapper_b.ensure_fresh_credentials().await.unwrap();
        assert_eq!(call_count_a.load(Ordering::SeqCst), 1);
        assert_eq!(call_count_b.load(Ordering::SeqCst), 1);

        // Verify wrapper A credentials are still cached and independent
        wrapper_a.ensure_fresh_credentials().await.unwrap();
        assert_eq!(call_count_a.load(Ordering::SeqCst), 1);
        assert_eq!(call_count_b.load(Ordering::SeqCst), 1);

        // Verify wrapper B credentials are still cached and independent
        wrapper_b.ensure_fresh_credentials().await.unwrap();
        assert_eq!(call_count_a.load(Ordering::SeqCst), 1);
        assert_eq!(call_count_b.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_wrapper_pointer_uniqueness() {
        // This test verifies that each wrapper instance has a unique pointer address
        // which is used for cache key differentiation
        let vendor = Arc::new(MockStorageOptionsProvider {
            call_count: Arc::new(AtomicUsize::new(0)),
            expires_at_millis: 9999999999999,
        });

        let wrapper_a = Arc::new(DynamicStorageOptionObjectStore::new(
            vendor.clone(),
            StorageOptionsProviderParams::default(),
        ));

        let wrapper_b = Arc::new(DynamicStorageOptionObjectStore::new(
            vendor,
            StorageOptionsProviderParams::default(),
        ));

        // Even though both wrappers use the same vendor, they should have
        // different pointer addresses, which ensures cache isolation
        let ptr_a = Arc::as_ptr(&wrapper_a);
        let ptr_b = Arc::as_ptr(&wrapper_b);

        assert_ne!(
            ptr_a, ptr_b,
            "Different wrapper instances must have different pointer addresses"
        );
    }
}
