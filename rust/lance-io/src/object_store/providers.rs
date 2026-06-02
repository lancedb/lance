// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, RwLock, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

use object_store::path::Path;
use url::Url;

use crate::object_store::WrappingObjectStore;
use crate::object_store::uri_to_url;

use super::{ObjectStore, ObjectStoreParams, tracing::ObjectStoreTracingExt};
use lance_core::error::{Error, LanceOptionExt, Result};

#[cfg(feature = "aws")]
pub mod aws;
#[cfg(feature = "azure")]
pub mod azure;
#[cfg(feature = "gcp")]
pub mod gcp;
#[cfg(feature = "huggingface")]
pub mod huggingface;
pub mod local;
pub mod memory;
#[cfg(feature = "oss")]
pub mod oss;
pub mod shared_memory;
#[cfg(feature = "tencent")]
pub mod tencent;

#[async_trait::async_trait]
pub trait ObjectStoreProvider: std::fmt::Debug + Sync + Send {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore>;

    /// Extract the path relative to the base of the store.
    ///
    /// For example, in S3 the path is relative to the bucket. So a URL of
    /// `s3://bucket/path/to/file` would return `path/to/file`.
    ///
    /// Meanwhile, for a file store, the path is relative to the filesystem root.
    /// So a URL of `file:///path/to/file` would return `/path/to/file`.
    fn extract_path(&self, url: &Url) -> Result<Path> {
        Path::parse(url.path())
            .map_err(|_| Error::invalid_input(format!("Invalid path in URL: {}", url.path())))
    }

    /// Calculate the unique prefix that should be used for this object store.
    ///
    /// For object stores that don't have the concept of buckets, this will just be something like
    /// 'file' or 'memory'.
    ///
    /// In object stores where all bucket names are unique, like s3, this will be
    /// simply 's3$my_bucket_name' or similar.
    ///
    /// In Azure, only the combination of (account name, container name) is unique, so
    /// this will be something like 'az$account_name@container'
    ///
    /// Providers should override this if they have special requirements like Azure's.
    fn calculate_object_store_prefix(
        &self,
        url: &Url,
        _storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        Ok(format!("{}${}", url.scheme(), url.authority()))
    }
}

/// Statistics for the object store registry cache.
#[derive(Debug, Clone, Default)]
pub struct ObjectStoreRegistryStats {
    /// Number of cache hits (store was already cached and reused).
    pub hits: u64,
    /// Number of cache misses (new store had to be created).
    pub misses: u64,
    /// Number of currently active object stores in the cache.
    pub active_stores: usize,
}

/// A cached object store's prefix plus the params it was built with.
type StoreCacheKey = (String, ObjectStoreParams);

/// Per-key lock that serializes concurrent cold builds of the same store.
type BuildLock = Arc<tokio::sync::Mutex<()>>;

/// A registry of object store providers.
///
/// Use [`Self::default()`] to create one with the available default providers.
/// This includes (depending on features enabled):
/// - `memory`: An in-memory object store.
/// - `file`: A local file object store, with optimized code paths.
/// - `file-object-store`: A local file object store that uses the ObjectStore API,
///   for all operations. Used for testing with ObjectStore wrappers.
/// - `file+uring`: A local file object store using io_uring (Linux only).
/// - `s3`: An S3 object store.
/// - `s3+ddb`: An S3 object store with DynamoDB for metadata.
/// - `az`: An Azure Blob Storage object store.
/// - `gs`: A Google Cloud Storage object store.
///
/// Use [`Self::empty()`] to create an empty registry, with no providers registered.
///
/// The registry also caches object stores that are currently in use. It holds
/// weak references to the object stores, so they are not held onto. If an object
/// store is no longer in use, it will be removed from the cache on the next
/// call to either [`Self::active_stores()`] or [`Self::get_store()`].
#[derive(Debug)]
pub struct ObjectStoreRegistry {
    providers: RwLock<HashMap<String, Arc<dyn ObjectStoreProvider>>>,
    // Cache of object stores currently in use. We use a weak reference so the
    // cache itself doesn't keep them alive if no object store is actually using
    // it.
    active_stores: RwLock<HashMap<StoreCacheKey, Weak<ObjectStore>>>,
    // Per-key locks that serialize cold builds for the same key. The
    // `active_stores` weak reference is only inserted after `new_store` returns,
    // so without this, tasks racing in before the first build finishes each
    // rebuild the same store. See issue #6838.
    building: Mutex<HashMap<StoreCacheKey, BuildLock>>,
    // Cache statistics
    hits: AtomicU64,
    misses: AtomicU64,
}

impl ObjectStoreRegistry {
    /// Create a new registry with no providers registered.
    ///
    /// Typically, you want to use [`Self::default()`] instead, so you get the
    /// default providers.
    pub fn empty() -> Self {
        Self {
            providers: RwLock::new(HashMap::new()),
            active_stores: RwLock::new(HashMap::new()),
            building: Mutex::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Get the object store provider for a given scheme.
    pub fn get_provider(&self, scheme: &str) -> Option<Arc<dyn ObjectStoreProvider>> {
        self.providers
            .read()
            .expect("ObjectStoreRegistry lock poisoned")
            .get(scheme)
            .cloned()
    }

    /// Get a list of all active object stores.
    ///
    /// Calling this will also clean up any weak references to object stores that
    /// are no longer valid.
    pub fn active_stores(&self) -> Vec<Arc<ObjectStore>> {
        let mut found_inactive = false;
        let output = self
            .active_stores
            .read()
            .expect("ObjectStoreRegistry lock poisoned")
            .values()
            .filter_map(|weak| match weak.upgrade() {
                Some(store) => Some(store),
                None => {
                    found_inactive = true;
                    None
                }
            })
            .collect();

        if found_inactive {
            // Clean up the cache by removing any weak references that are no longer valid
            let mut cache_lock = self
                .active_stores
                .write()
                .expect("ObjectStoreRegistry lock poisoned");
            cache_lock.retain(|_, weak| weak.upgrade().is_some());
        }
        output
    }

    /// Get cache statistics for monitoring and debugging.
    ///
    /// Returns the number of cache hits, misses, and currently active stores.
    /// This is useful for detecting configuration issues that cause excessive
    /// cache misses (e.g., storage options that vary per-request).
    pub fn stats(&self) -> ObjectStoreRegistryStats {
        let active_stores = self
            .active_stores
            .read()
            .map(|s| s.values().filter(|w| w.strong_count() > 0).count())
            .unwrap_or(0);
        ObjectStoreRegistryStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            active_stores,
        }
    }

    fn scheme_not_found_error(&self, scheme: &str) -> Error {
        let mut message = format!("No object store provider found for scheme: '{}'", scheme);
        if let Ok(providers) = self.providers.read() {
            let valid_schemes = providers.keys().cloned().collect::<Vec<_>>().join(", ");
            message.push_str(&format!("\nValid schemes: {}", valid_schemes));
        }
        Error::invalid_input(message)
    }

    /// Return a live cached store for `cache_key`, if one exists.
    ///
    /// A stale weak reference (the store was dropped since it was cached) is
    /// removed. Does not touch the hit/miss counters, so callers decide how to
    /// count the lookup.
    fn cached_store(&self, cache_key: &StoreCacheKey) -> Result<Option<Arc<ObjectStore>>> {
        let maybe_weak = self
            .active_stores
            .read()
            .ok()
            .expect_ok()?
            .get(cache_key)
            .cloned();
        let Some(weak) = maybe_weak else {
            return Ok(None);
        };
        if let Some(store) = weak.upgrade() {
            return Ok(Some(store));
        }
        // The store was dropped since we cached it. Remove the stale weak
        // reference so the slot can be rebuilt.
        let mut cache_lock = self.active_stores.write().ok().expect_ok()?;
        if let Some(weak) = cache_lock.get(cache_key)
            && weak.upgrade().is_none()
        {
            cache_lock.remove(cache_key);
        }
        Ok(None)
    }

    /// Drop our in-flight build marker, but only if it is still the one we
    /// registered. A later cold request may have installed a fresh lock, and we
    /// must not evict theirs.
    fn clear_build_lock(&self, cache_key: &StoreCacheKey, build_lock: &BuildLock) {
        let mut building = self
            .building
            .lock()
            .expect("ObjectStoreRegistry lock poisoned");
        if let Some(existing) = building.get(cache_key)
            && Arc::ptr_eq(existing, build_lock)
        {
            building.remove(cache_key);
        }
    }

    /// Get an object store for a given base path and parameters.
    ///
    /// If the object store is already in use, it will return a strong reference
    /// to the object store. If the object store is not in use, it will create a
    /// new object store and return a strong reference to it.
    ///
    /// Concurrent cold calls for the same key are serialized so the store is
    /// built once and the rest reuse the cached result.
    pub async fn get_store(
        &self,
        base_path: Url,
        params: &ObjectStoreParams,
    ) -> Result<Arc<ObjectStore>> {
        let scheme = base_path.scheme();
        let Some(provider) = self.get_provider(scheme) else {
            return Err(self.scheme_not_found_error(scheme));
        };

        let cache_path =
            provider.calculate_object_store_prefix(&base_path, params.storage_options())?;
        let cache_key = (cache_path.clone(), params.clone());

        // Fast path: return the cached store without taking the build lock.
        if let Some(store) = self.cached_store(&cache_key)? {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(store);
        }

        // Slow path: serialize cold builds for the same key behind one in-flight
        // `new_store()` so racing tasks reuse it instead of each rebuilding it.
        let build_lock = {
            let mut building = self
                .building
                .lock()
                .expect("ObjectStoreRegistry lock poisoned");
            building
                .entry(cache_key.clone())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                .clone()
        };
        let _build_guard = build_lock.lock().await;

        // Another task may have built the store while we waited for the lock.
        if let Some(store) = self.cached_store(&cache_key)? {
            self.hits.fetch_add(1, Ordering::Relaxed);
            self.clear_build_lock(&cache_key, &build_lock);
            return Ok(store);
        }

        self.misses.fetch_add(1, Ordering::Relaxed);

        let build_result = async {
            let mut store = provider.new_store(base_path, params).await?;

            store.inner = store.inner.traced();

            if let Some(wrapper) = &params.object_store_wrapper {
                store.inner = wrapper.wrap(&cache_path, store.inner);
            }

            // Always wrap with IO tracking
            store.inner = store.io_tracker.wrap("", store.inner);

            let store = Arc::new(store);

            // Insert the store into the cache
            let mut cache_lock = self.active_stores.write().ok().expect_ok()?;
            cache_lock.insert(cache_key.clone(), Arc::downgrade(&store));

            Ok(store)
        }
        .await;

        // Clear the in-flight marker on success or failure, so a failed build
        // leaves no stale lock entry behind.
        self.clear_build_lock(&cache_key, &build_lock);

        build_result
    }

    /// Calculate the datastore prefix based on the URI and the storage options.
    /// The data store prefix should uniquely identify the datastore.
    pub fn calculate_object_store_prefix(
        &self,
        uri: &str,
        storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        let url = uri_to_url(uri)?;
        match self.get_provider(url.scheme()) {
            None => {
                if url.scheme() == "file" || url.scheme().len() == 1 {
                    Ok("file".to_string())
                } else {
                    Err(self.scheme_not_found_error(url.scheme()))
                }
            }
            Some(provider) => provider.calculate_object_store_prefix(&url, storage_options),
        }
    }
}

impl Default for ObjectStoreRegistry {
    fn default() -> Self {
        let mut providers: HashMap<String, Arc<dyn ObjectStoreProvider>> = HashMap::new();

        providers.insert("memory".into(), Arc::new(memory::MemoryStoreProvider));
        providers.insert(
            "shared-memory".into(),
            Arc::new(shared_memory::SharedMemoryStoreProvider::default()),
        );
        providers.insert("file".into(), Arc::new(local::FileStoreProvider));
        // The "file" scheme has special optimized code paths that bypass
        // the ObjectStore API for better performance. However, this can make it
        // hard to test when using ObjectStore wrappers, such as IOTrackingStore.
        // So we provide a "file-object-store" scheme that uses the ObjectStore API.
        // The specialized code paths are differentiated by the scheme name.
        providers.insert(
            "file-object-store".into(),
            Arc::new(local::FileStoreProvider),
        );
        #[cfg(target_os = "linux")]
        providers.insert("file+uring".into(), Arc::new(local::FileStoreProvider));

        #[cfg(feature = "aws")]
        {
            let aws = Arc::new(aws::AwsStoreProvider);
            providers.insert("s3".into(), aws.clone());
            providers.insert("s3+ddb".into(), aws);
        }
        #[cfg(feature = "azure")]
        {
            let azure = Arc::new(azure::AzureBlobStoreProvider);
            providers.insert("az".into(), azure.clone());
            providers.insert("abfss".into(), azure);
        }
        #[cfg(feature = "gcp")]
        providers.insert("gs".into(), Arc::new(gcp::GcsStoreProvider));
        #[cfg(feature = "oss")]
        providers.insert("oss".into(), Arc::new(oss::OssStoreProvider));
        #[cfg(feature = "tencent")]
        providers.insert("cos".into(), Arc::new(tencent::TencentStoreProvider));
        #[cfg(feature = "huggingface")]
        providers.insert("hf".into(), Arc::new(huggingface::HuggingfaceStoreProvider));
        Self {
            providers: RwLock::new(providers),
            active_stores: RwLock::new(HashMap::new()),
            building: Mutex::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }
}

impl ObjectStoreRegistry {
    /// Add a new object store provider to the registry. The provider will be used
    /// in [`Self::get_store()`] when a URL is passed with a matching scheme.
    pub fn insert(&self, scheme: &str, provider: Arc<dyn ObjectStoreProvider>) {
        self.providers
            .write()
            .expect("ObjectStoreRegistry lock poisoned")
            .insert(scheme.into(), provider);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[derive(Debug)]
    struct DummyProvider;

    #[async_trait::async_trait]
    impl ObjectStoreProvider for DummyProvider {
        async fn new_store(
            &self,
            _base_path: Url,
            _params: &ObjectStoreParams,
        ) -> Result<ObjectStore> {
            unreachable!("This test doesn't create stores")
        }
    }

    #[test]
    fn test_calculate_object_store_prefix() {
        let provider = DummyProvider;
        let url = Url::parse("dummy://blah/path").unwrap();
        assert_eq!(
            "dummy$blah",
            provider.calculate_object_store_prefix(&url, None).unwrap()
        );
    }

    #[test]
    fn test_calculate_object_store_scheme_not_found() {
        let registry = ObjectStoreRegistry::empty();
        registry.insert("dummy", Arc::new(DummyProvider));
        let s = "Invalid user input: No object store provider found for scheme: 'dummy2'\nValid schemes: dummy";
        let result = registry
            .calculate_object_store_prefix("dummy2://mybucket/my/long/path", None)
            .expect_err("expected error")
            .to_string();
        assert_eq!(s, &result[..s.len()]);
    }

    // Test that paths without a scheme get treated as local paths.
    #[test]
    fn test_calculate_object_store_prefix_for_local() {
        let registry = ObjectStoreRegistry::empty();
        assert_eq!(
            "file",
            registry
                .calculate_object_store_prefix("/tmp/foobar", None)
                .unwrap()
        );
    }

    // Test that paths with a single-letter scheme that is not registered for anything get treated as local paths.
    #[test]
    fn test_calculate_object_store_prefix_for_local_windows_path() {
        let registry = ObjectStoreRegistry::empty();
        assert_eq!(
            "file",
            registry
                .calculate_object_store_prefix("c://dos/path", None)
                .unwrap()
        );
    }

    // Test that paths with a given scheme get mapped to that storage provider.
    #[test]
    fn test_calculate_object_store_prefix_for_dummy_path() {
        let registry = ObjectStoreRegistry::empty();
        registry.insert("dummy", Arc::new(DummyProvider));
        assert_eq!(
            "dummy$mybucket",
            registry
                .calculate_object_store_prefix("dummy://mybucket/my/long/path", None)
                .unwrap()
        );
    }

    #[tokio::test]
    async fn test_stats_hit_miss_tracking() {
        use crate::object_store::StorageOptionsAccessor;
        let registry = ObjectStoreRegistry::default();
        let url = Url::parse("memory://test").unwrap();

        let params1 = ObjectStoreParams::default();
        let params2 = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([("k".into(), "v".into())]),
            ))),
            ..Default::default()
        };

        // (hits, misses, active)
        let cases: &[(&ObjectStoreParams, (u64, u64, usize))] = &[
            (&params1, (0, 1, 1)), // miss: new params
            (&params1, (1, 1, 1)), // hit: same params
            (&params2, (1, 2, 2)), // miss: different storage_options
        ];

        let mut stores = vec![]; // retain the stores
        for (params, (hits, misses, active)) in cases {
            stores.push(registry.get_store(url.clone(), params).await.unwrap());
            let s = registry.stats();
            assert_eq!(
                (s.hits, s.misses, s.active_stores),
                (*hits, *misses, *active)
            );
        }

        // Same params returns same instance
        assert!(Arc::ptr_eq(&stores[0], &stores[1]));
    }

    /// Counts how many times it builds a store, and blocks inside the build
    /// until released, so a test can hold the build window open while concurrent
    /// callers race in.
    #[derive(Debug)]
    struct CountingProvider {
        builds: Arc<AtomicU64>,
        // Notified each time a build enters `new_store`.
        entered: Arc<tokio::sync::Notify>,
        // A build proceeds only once it acquires a permit. The test withholds
        // permits to hold the build window open, then grants enough that every
        // in-flight build can finish, so a regression with many concurrent
        // builds fails on the `builds` assertion instead of deadlocking.
        release: Arc<tokio::sync::Semaphore>,
    }

    #[async_trait::async_trait]
    impl ObjectStoreProvider for CountingProvider {
        async fn new_store(
            &self,
            base_path: Url,
            params: &ObjectStoreParams,
        ) -> Result<ObjectStore> {
            self.builds.fetch_add(1, Ordering::SeqCst);
            self.entered.notify_one();
            let _permit = self.release.acquire().await.expect("semaphore closed");
            memory::MemoryStoreProvider
                .new_store(base_path, params)
                .await
        }
    }

    // Concurrent cold opens for the same key must build the store once and share
    // it. Regression test for issue #6838.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_concurrent_cold_builds_share_one_store() {
        let builds = Arc::new(AtomicU64::new(0));
        let entered = Arc::new(tokio::sync::Notify::new());
        // Start with no permits so the first build blocks inside `new_store`.
        let release = Arc::new(tokio::sync::Semaphore::new(0));

        let registry = Arc::new(ObjectStoreRegistry::empty());
        registry.insert(
            "memory",
            Arc::new(CountingProvider {
                builds: builds.clone(),
                entered: entered.clone(),
                release: release.clone(),
            }),
        );

        let url = Url::parse("memory://race").unwrap();
        let params = ObjectStoreParams::default();

        const N: usize = 16;
        let handles: Vec<_> = (0..N)
            .map(|_| {
                let registry = registry.clone();
                let url = url.clone();
                let params = params.clone();
                tokio::spawn(async move { registry.get_store(url, &params).await.unwrap() })
            })
            .collect();

        // Wait until one task is inside the build, then let the rest pile up
        // behind the build lock before allowing the build to finish.
        entered.notified().await;
        for _ in 0..N {
            tokio::task::yield_now().await;
        }
        // Grant enough permits for every task. If serialization were broken and
        // all N entered the build, they could all finish and trip the `builds`
        // assertion instead of deadlocking.
        release.add_permits(N);

        let mut stores = Vec::with_capacity(N);
        for handle in handles {
            stores.push(handle.await.unwrap());
        }

        // The expensive build ran exactly once despite N concurrent cold opens.
        assert_eq!(builds.load(Ordering::SeqCst), 1);

        // Every caller observes the same cached instance.
        for store in &stores[1..] {
            assert!(Arc::ptr_eq(&stores[0], store));
        }

        // One miss for the single build. The rest are served as hits.
        let stats = registry.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, (N - 1) as u64);

        // The in-flight build marker is cleaned up after the build completes.
        assert!(
            registry
                .building
                .lock()
                .expect("ObjectStoreRegistry lock poisoned")
                .is_empty()
        );
    }
}
