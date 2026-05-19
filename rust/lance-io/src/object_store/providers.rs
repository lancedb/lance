// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::HashMap,
    sync::{
        Arc, RwLock, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

use object_store::path::Path;
use tokio::sync::Mutex as AsyncMutex;
use url::{Host, Url};

use crate::object_store::WrappingObjectStore;
use crate::object_store::uri_to_url;

use super::{ObjectStore, ObjectStoreParams, tracing::ObjectStoreTracingExt};
use lance_core::error::{Error, Result};

#[cfg(feature = "aws")]
pub mod aws;
#[cfg(feature = "azure")]
pub mod azure;
#[cfg(feature = "gcp")]
pub mod gcp;
#[cfg(feature = "goosefs")]
pub mod goosefs;
#[cfg(feature = "huggingface")]
pub mod huggingface;
pub mod local;
pub mod memory;
#[cfg(feature = "oss")]
pub mod oss;
pub mod shared_memory;
#[cfg(feature = "tencent")]
pub mod tencent;
#[cfg(feature = "tos")]
pub mod tos;

#[async_trait::async_trait]
pub trait ObjectStoreProvider: std::fmt::Debug + Sync + Send {
    /// Construct a new object store for the given base path and params.
    ///
    /// **Reentry warning**: implementations MUST NOT recursively call
    /// [`ObjectStoreRegistry::get_store`] for the same `(base_path, params)`
    /// key. The registry coalesces concurrent cold builds via a per-key
    /// async lock held across the call to `new_store`; a re-entrant call
    /// would deadlock waiting on the lock the current task already holds.
    /// Calling `get_store` for a *different* key (e.g. an underlying delegate
    /// store) is safe.
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore>;

    /// Extract the path relative to the base of the store.
    ///
    /// For example, in S3 the path is relative to the bucket. So a URL of
    /// `s3://bucket/path/to/file` would return `path/to/file`.
    ///
    /// Meanwhile, for a file store, the path is relative to the filesystem root.
    /// So a URL of `file:///path/to/file` would return `/path/to/file`.
    fn extract_path(&self, url: &Url) -> Result<Path> {
        // url.path() returns a percent-encoded string (per the WHATWG URL spec).
        // Path::from_url_path decodes it first so the Path internal representation
        // holds the raw UTF-8 string. This prevents double-encoding when the
        // object store client later percent-encodes the path for HTTP requests.
        Path::from_url_path(url.path()).map_err(|e| {
            Error::invalid_input(format!("Invalid path in URL '{}': {}", url.path(), e))
        })
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
    ///
    /// # Userinfo / IPv6 handling
    ///
    /// Overrides that build the prefix from URL authority MUST use
    /// [`sanitized_authority`] (or equivalent) instead of `Url::authority()` —
    /// the latter includes any embedded `userinfo@`, which would leak
    /// credentials into the cache key. The default impl below already does
    /// this. Azure is the only intentional exception: it parses `userinfo`
    /// position as the container name, not credentials.
    fn calculate_object_store_prefix(
        &self,
        url: &Url,
        _storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        Ok(format!("{}${}", url.scheme(), sanitized_authority(url)))
    }
}

/// Build a cache-key authority component from a URL with two safety properties:
///
/// 1. Strips any `userinfo@` segment so URL-embedded credentials never land in
///    the cache key (which is logged via `Debug` and observed by anything that
///    calls `calculate_object_store_prefix`).
/// 2. Wraps IPv6 hosts in brackets so `[::1]:9000` is unambiguous —
///    `Url::host_str` returns `"::1"` without brackets, which combined with a
///    port would produce the ambiguous string `"::1:9000"`.
///
/// Returns just `host[:port]`. Use this in any
/// [`ObjectStoreProvider::calculate_object_store_prefix`] override that would
/// otherwise reach for `Url::authority()`.
pub fn sanitized_authority(url: &Url) -> String {
    let host = match url.host() {
        Some(Host::Ipv6(addr)) => format!("[{}]", addr),
        Some(Host::Ipv4(addr)) => addr.to_string(),
        Some(Host::Domain(d)) => d.to_string(),
        None => String::new(),
    };
    match url.port() {
        Some(p) => format!("{}:{}", host, p),
        None => host,
    }
}

type CacheKey = (String, ObjectStoreParams);
type BuildLockMap = HashMap<CacheKey, Arc<AsyncMutex<()>>>;

/// Statistics for the object store registry cache.
///
/// Exposed for in-tree diagnostic consumers (e.g. the
/// `concurrent_open_bench` example, which prints hit/miss counts to prove
/// registry sharing across opens). `#[doc(hidden)]` because this is an
/// observability surface, not a stable API — field names and types may
/// change without a semver bump.
#[doc(hidden)]
#[derive(Debug, Clone, Default)]
pub struct ObjectStoreRegistryStats {
    /// Number of cache hits (store was already cached and reused).
    pub hits: u64,
    /// Number of cache misses (new store had to be created).
    pub misses: u64,
    /// Number of currently active object stores in the cache.
    pub active_stores: usize,
    /// Number of cache keys with an in-flight or recently-finished cold
    /// build whose RAII cleanup has not yet acquired the build-locks mutex.
    ///
    /// Steady-state should be 0 — non-zero indicates either an in-progress
    /// thundering herd (expected during cold start) or a stuck builder.
    pub build_locks: usize,
    /// Number of cold builds that returned an error from the underlying
    /// provider (`ObjectStoreProvider::new_store`).
    ///
    /// Per-call, not per-key: single-flight serializes failures via the
    /// per-key lock but does not cache the `Err` result. N waiters racing
    /// a failing upstream each retry in turn and bump this counter, so a
    /// stuck failure grows roughly linearly with attempts.
    pub build_failures: u64,
}

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
/// - `tos`: A Volcengine TOS object store.
///
/// Use [`Self::empty()`] to create an empty registry, with no providers registered.
///
/// The registry also caches object stores that are currently in use. It holds
/// weak references to the object stores, so they are not held onto. Stale
/// entries are reclaimed lazily:
/// - [`Self::active_stores()`] sweeps the whole map.
/// - [`Self::get_store()`] evicts the queried key on miss, and opportunistically
///   prunes other dead entries on every cold-build insert (bounded work, keyed
///   off in-use opens).
#[derive(Debug)]
pub struct ObjectStoreRegistry {
    providers: RwLock<HashMap<String, Arc<dyn ObjectStoreProvider>>>,
    // Cache of object stores currently in use. We use a weak reference so the
    // cache itself doesn't keep them alive if no object store is actually using
    // it.
    active_stores: RwLock<HashMap<CacheKey, Weak<ObjectStore>>>,
    // Per-key async build locks coalesce concurrent cold builds for the same
    // key into one `provider.new_store` call (avoids the thundering herd on
    // S3-style providers whose client construction is expensive).
    build_locks: std::sync::Mutex<BuildLockMap>,
    // Cache statistics
    hits: AtomicU64,
    misses: AtomicU64,
    build_failures: AtomicU64,
}

/// RAII session for the cold-build path: holds the per-key async lock guard
/// and reclaims the `build_locks` HashMap entry on drop, in that order.
///
/// `Drop::drop` explicitly releases `guard` *before* running
/// `cleanup_build_lock_if_idle`, so cleanup observes `strong_count == 1`
/// (only the HashMap entry) and can remove the lock entry safely. Encoding
/// the ordering in this Drop body — rather than relying on the declaration
/// order of two separate `let` bindings at the call site — keeps the
/// invariant locally checkable: a future refactor of the cold path cannot
/// accidentally invert it.
///
/// Drop runs on every exit (success, error, panic, cancellation), so the
/// `build_locks` entry can never leak if the cold-build path is interrupted
/// between acquiring the lock and finishing the build.
///
/// Owns the `CacheKey` (rather than borrowing) so the session's drop is not
/// tied to the lifetime of a local variable.
struct BuildLockSession<'a> {
    registry: &'a ObjectStoreRegistry,
    cache_key: CacheKey,
    guard: Option<tokio::sync::OwnedMutexGuard<()>>,
}

impl Drop for BuildLockSession<'_> {
    fn drop(&mut self) {
        // 1. Release the per-key async lock first by dropping the
        //    `OwnedMutexGuard`. This decrements the per-key
        //    `Arc<AsyncMutex<()>>` strong count.
        self.guard.take();
        // 2. Reclaim the `build_locks` entry. With the guard gone, only the
        //    HashMap entry holds an Arc, so cleanup observes count == 1 and
        //    safely removes it.
        self.registry.cleanup_build_lock_if_idle(&self.cache_key);
    }
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
            build_locks: std::sync::Mutex::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            build_failures: AtomicU64::new(0),
        }
    }

    /// Get the object store provider for a given scheme.
    pub fn get_provider(&self, scheme: &str) -> Option<Arc<dyn ObjectStoreProvider>> {
        self.providers
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get(scheme)
            .cloned()
    }

    /// Get a list of all active object stores.
    ///
    /// Calling this will also clean up any weak references to object stores that
    /// are no longer valid.
    ///
    /// **Caution**: the returned `Arc<ObjectStore>` instances may carry
    /// per-tenant credential state (S3 access keys, Azure SAS tokens, etc.).
    /// Callers that hand these stores out across trust boundaries are
    /// responsible for filtering — the registry itself does not know which
    /// store belongs to which tenant.
    pub fn active_stores(&self) -> Vec<Arc<ObjectStore>> {
        let mut found_inactive = false;
        let output = self
            .active_stores
            .read()
            .unwrap_or_else(|p| p.into_inner())
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
                .unwrap_or_else(|p| p.into_inner());
            cache_lock.retain(|_, weak| weak.upgrade().is_some());
        }
        output
    }

    /// Get cache statistics for monitoring and debugging.
    ///
    /// Returns counters for hits, misses, currently active stores,
    /// in-flight build locks, and accumulated build failures — see
    /// [`ObjectStoreRegistryStats`]. Useful for detecting configuration
    /// issues that cause excessive cache misses (e.g., storage options
    /// that vary per-request).
    ///
    /// `#[doc(hidden)]`: paired with [`ObjectStoreRegistryStats`] — this is
    /// an unstable observability surface kept `pub` only to support the
    /// in-tree `concurrent_open_bench` example.
    #[doc(hidden)]
    pub fn stats(&self) -> ObjectStoreRegistryStats {
        let active_stores = self
            .active_stores
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .values()
            .filter(|w| w.strong_count() > 0)
            .count();
        let build_locks = self
            .build_locks
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .len();
        ObjectStoreRegistryStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            active_stores,
            build_locks,
            build_failures: self.build_failures.load(Ordering::Relaxed),
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

    /// Get an object store for a given base path and parameters.
    ///
    /// If the object store is already in use, it will return a strong reference
    /// to the object store. If the object store is not in use, it will create a
    /// new object store and return a strong reference to it.
    ///
    /// Concurrent cold builds for the same key are coalesced via a per-key
    /// async lock: the first task builds the store, all others wait then
    /// re-check the cache and observe the freshly populated entry.
    ///
    /// On build *failure*, the cache is not populated, so each waiter retries
    /// in turn (serialized — not parallel-amplified, but not deduplicated
    /// either). Transient errors thus surface to operators rather than being
    /// masked by stale-error reuse.
    pub async fn get_store(
        &self,
        base_path: Url,
        params: &ObjectStoreParams,
    ) -> Result<Arc<ObjectStore>> {
        // Base-scoped storage options (`base_<id>.<key>`) are directives for
        // other registered base paths; resolve them away before building or
        // caching a store for this location. Params already resolved for a
        // base contain no scoped entries, so this is a no-op for them.
        let params = params.scoped_to_base(None);
        let params = params.as_ref();
        let scheme = base_path.scheme();
        let Some(provider) = self.get_provider(scheme) else {
            return Err(self.scheme_not_found_error(scheme));
        };

        let cache_path =
            provider.calculate_object_store_prefix(&base_path, params.storage_options())?;
        let cache_key = (cache_path.clone(), params.clone());

        // Fast path: cache hit avoids both the std mutex and the async lock.
        if let Some(store) = self.lookup_cached(&cache_key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(store);
        }

        // Cold path: per-key single-flight. The RAII session guarantees the
        // build_locks entry is GC'd on every exit (success, error, panic,
        // cancellation), preventing unbounded HashMap growth. Drop ordering
        // (guard released *before* cleanup) is encoded in
        // `BuildLockSession::drop`, so the cold path can't accidentally
        // invert it via `let` reordering.
        let lock = self.acquire_build_lock(&cache_key);
        let guard = lock.lock_owned().await;
        let _session = BuildLockSession {
            registry: self,
            cache_key: cache_key.clone(),
            guard: Some(guard),
        };

        // Re-check after acquiring the lock — coalesced waiters become hits.
        if let Some(store) = self.lookup_cached(&cache_key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            log::debug!(
                "ObjectStoreRegistry: coalesced wait hit for cache_key path={:?}",
                cache_key.0,
            );
            return Ok(store);
        }

        // `fetch_add` returns the prior value; `+ 1` is this attempt's
        // 1-indexed sequence number. Use it (not a separate counter) to
        // gate the opportunistic sweep below: cadence is approximate
        // because cold attempts that fail still bump `misses` but skip
        // the sweep block, so a failure at a multiple-of-N sequence
        // skips that sweep cycle entirely; the next candidate is the
        // next multiple-of-N cold attempt that succeeds. That's fine —
        // the sweep is a footprint trim, not a correctness invariant.
        // `wrapping_add` only silences debug-build overflow checks; at
        // 10M cold opens/sec the wrap is >58 millennia away.
        let cold_attempt_nth = self.misses.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
        log::debug!(
            "ObjectStoreRegistry: cold build starting for cache_key path={:?}",
            cache_key.0,
        );

        let mut store = match provider.new_store(base_path, params).await {
            Ok(s) => s,
            Err(e) => {
                self.build_failures.fetch_add(1, Ordering::Relaxed);
                // Intentionally no log here: provider error Display impls can
                // surface raw URLs whose username or query-string component
                // may carry credentials (SAS tokens, access keys), and `Url`'s
                // own `Display` only masks the password. The full error goes
                // to the caller via `Err(e)`; operators wanting an aggregate
                // signal can scrape `stats().build_failures`.
                return Err(e);
            }
        };
        store.inner = store.inner.traced();

        #[cfg(feature = "metrics")]
        {
            // Label metrics by the store's unique prefix (e.g. `s3$bucket`,
            // `az$container@account`) so multiple stores on one cloud differ.
            use crate::object_store::metrics::ObjectStoreMetricsExt;
            store.inner = store.inner.metered(cache_path.clone());
        }

        if let Some(wrapper) = &params.object_store_wrapper {
            store.inner = wrapper.wrap(&cache_path, store.inner);
        }
        store.inner = store.io_tracker.wrap("", store.inner);
        let cached = Arc::new(store);
        // Amortized opportunistic sweep: gate the O(n) `retain` behind a
        // mod-N counter so a burst of cold builds with distinct URIs costs
        // O(n) work in aggregate, not O(n²). N=64 is small enough that the
        // map's footprint stays roughly proportional to live opens, large
        // enough that bursty cold-open paths don't pay sweep cost on every
        // insert. Per-key `lookup_cached` and full sweep via `active_stores()`
        // remain unchanged; this is purely a per-insert cost lever.
        //
        // Cadence is "every 64th cold attempt" (1-indexed via the pre-bump
        // above), so the first sweep fires at the 64th, not the 0th, and
        // a near-empty map never pays the retain cost on its very first
        // cold build.
        const SWEEP_INTERVAL: u64 = 64;
        let should_sweep = cold_attempt_nth.is_multiple_of(SWEEP_INTERVAL);
        {
            let mut cache_lock = self
                .active_stores
                .write()
                .unwrap_or_else(|p| p.into_inner());
            // Safe under the write lock: no other writer can resurrect a weak
            // ref between `retain` and `insert`, and `insert` for `cache_key`
            // overwrites whatever `retain` left (live or dead) for that key.
            if should_sweep {
                cache_lock.retain(|_, weak| weak.upgrade().is_some());
            }
            cache_lock.insert(cache_key, Arc::downgrade(&cached));
        }
        Ok(cached)
    }

    /// Acquire (or create) the per-key async build lock for `cache_key`.
    ///
    /// On poison: recovers via `into_inner()` rather than bricking the
    /// registry. The protected data is just a HashMap of build-lock Arcs;
    /// any stale entry left by a panic is reclaimed by `BuildLockSession`.
    fn acquire_build_lock(&self, cache_key: &CacheKey) -> Arc<AsyncMutex<()>> {
        let mut locks = self
            .build_locks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        locks
            .entry(cache_key.clone())
            .or_insert_with(|| Arc::new(AsyncMutex::new(())))
            .clone()
    }

    /// Drop the build-lock entry for `cache_key` if no waiters remain.
    ///
    /// Strong-count == 1 means only the HashMap entry references the Arc, so
    /// no concurrent task is using or queued on the lock. The HashMap mutex
    /// makes this atomic: a new caller can't clone the Arc between the count
    /// read and the removal because `acquire_build_lock` takes the same mutex.
    /// A waiter parked inside `lock_owned()` holds its own Arc clone in the
    /// future, so it counts toward `strong_count` and keeps the entry alive
    /// until it wakes.
    ///
    /// Best-effort: if a task drops without calling this (e.g. between
    /// dropping the build lock and entering this function), the entry is
    /// reclaimed by the next caller for the same key, who finds count == 1
    /// and removes it.
    ///
    /// Bound: per-key cleanup is sufficient — no periodic full-map sweep is
    /// needed. The map's worst-case size is the number of distinct cache keys
    /// with an in-flight builder *plus* keys whose builder has finished but
    /// whose `BuildLockSession::drop` has not yet acquired the std mutex.
    /// Both terms are bounded by concurrently outstanding cold opens, which
    /// the underlying I/O concurrency limits already cap. RAII guarantees
    /// every successful `acquire_build_lock` is paired with exactly one
    /// `cleanup_build_lock_if_idle`, on every exit path.
    fn cleanup_build_lock_if_idle(&self, cache_key: &CacheKey) {
        let mut locks = self
            .build_locks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(entry) = locks.get(cache_key)
            && Arc::strong_count(entry) == 1
        {
            locks.remove(cache_key);
        }
    }

    #[cfg(test)]
    fn build_locks_len(&self) -> usize {
        self.build_locks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// Look up a cached store by key, evicting the entry if its weak ref is
    /// no longer upgradable.
    fn lookup_cached(&self, cache_key: &CacheKey) -> Option<Arc<ObjectStore>> {
        let maybe_weak = self
            .active_stores
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get(cache_key)
            .cloned();
        let weak = maybe_weak?;
        if let Some(store) = weak.upgrade() {
            return Some(store);
        }
        // Stale weak; evict under the write lock.
        let mut cache_lock = self
            .active_stores
            .write()
            .unwrap_or_else(|p| p.into_inner());
        if let Some(weak) = cache_lock.get(cache_key)
            && weak.upgrade().is_none()
        {
            cache_lock.remove(cache_key);
        }
        None
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
        #[cfg(feature = "goosefs")]
        providers.insert("goosefs".into(), Arc::new(goosefs::GooseFsStoreProvider));
        #[cfg(feature = "oss")]
        providers.insert("oss".into(), Arc::new(oss::OssStoreProvider));
        #[cfg(feature = "tencent")]
        providers.insert("cos".into(), Arc::new(tencent::TencentStoreProvider));
        #[cfg(feature = "huggingface")]
        providers.insert("hf".into(), Arc::new(huggingface::HuggingfaceStoreProvider));
        #[cfg(feature = "tos")]
        providers.insert("tos".into(), Arc::new(tos::TosStoreProvider));
        Self {
            providers: RwLock::new(providers),
            active_stores: RwLock::new(HashMap::new()),
            build_locks: std::sync::Mutex::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            build_failures: AtomicU64::new(0),
        }
    }
}

impl ObjectStoreRegistry {
    /// Add a new object store provider to the registry. The provider will be used
    /// in [`Self::get_store()`] when a URL is passed with a matching scheme.
    pub fn insert(&self, scheme: &str, provider: Arc<dyn ObjectStoreProvider>) {
        self.providers
            .write()
            .unwrap_or_else(|p| p.into_inner())
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

    #[tokio::test]
    async fn test_get_store_resolves_base_scoped_options() {
        use crate::object_store::StorageOptionsAccessor;

        let registry = ObjectStoreRegistry::default();
        let url = Url::parse("memory://test").unwrap();

        let with_scoped = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([
                    ("shared".to_string(), "value".to_string()),
                    ("base_1.account_key".to_string(), "base1-key".to_string()),
                ]),
            ))),
            ..Default::default()
        };
        let without_scoped = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([("shared".to_string(), "value".to_string())]),
            ))),
            ..Default::default()
        };

        // Base-scoped entries are resolved away before the store is built and
        // cached, so params with and without them yield the same cached store.
        let store_scoped = registry.get_store(url.clone(), &with_scoped).await.unwrap();
        let store_plain = registry.get_store(url, &without_scoped).await.unwrap();
        assert!(Arc::ptr_eq(&store_scoped, &store_plain));
    }

    /// `userinfo@host` (URL-embedded credentials) MUST be stripped from the
    /// cache-key prefix so the secret never lands in HashMap Debug logs or
    /// downstream state. Two URLs that differ only in `userinfo` collapse to
    /// the same prefix; multi-tenant isolation has to come from
    /// `ObjectStoreParams` (storage_options / accessor), not URL embedding.
    #[test]
    fn test_calculate_object_store_prefix_strips_userinfo() {
        let provider = DummyProvider;
        let with_creds = Url::parse("dummy://user:s3cret@blah/path").unwrap();
        let without = Url::parse("dummy://blah/path").unwrap();
        let with_prefix = provider
            .calculate_object_store_prefix(&with_creds, None)
            .unwrap();
        assert_eq!("dummy$blah", with_prefix);
        assert_eq!(
            with_prefix,
            provider
                .calculate_object_store_prefix(&without, None)
                .unwrap(),
        );
        // Defense in depth: assert the secret literally cannot appear in the
        // key, regardless of formatting changes.
        assert!(!with_prefix.contains('@'));
        assert!(!with_prefix.contains("s3cret"));
        assert!(!with_prefix.contains("user"));
    }

    #[test]
    fn test_calculate_object_store_prefix_keeps_port() {
        let provider = DummyProvider;
        let url = Url::parse("dummy://host:9000/path").unwrap();
        assert_eq!(
            "dummy$host:9000",
            provider.calculate_object_store_prefix(&url, None).unwrap()
        );
    }

    #[test]
    fn test_sanitized_authority_brackets_ipv6() {
        // Url::host_str returns "::1" without brackets; combined with a port,
        // the naive `format!("{}:{}", host, port)` would produce the
        // ambiguous string "::1:9000". sanitized_authority restores brackets.
        let url = Url::parse("dummy://[::1]:9000/path").unwrap();
        assert_eq!("[::1]:9000", sanitized_authority(&url));
        let url_no_port = Url::parse("dummy://[2001:db8::1]/path").unwrap();
        assert_eq!("[2001:db8::1]", sanitized_authority(&url_no_port));
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

        // The cache reused the entry for the second call (hits == 1 above);
        // when the same params hit the same cache slot, the returned Arcs
        // point to the same underlying ObjectStore.
        assert!(Arc::ptr_eq(&stores[0], &stores[1]));
    }

    /// A provider that counts `new_store` invocations and yields explicitly,
    /// so concurrent callers reliably contend on the registry's build lock.
    #[derive(Debug, Default)]
    struct CountingProvider {
        builds: AtomicU64,
    }

    #[async_trait::async_trait]
    impl ObjectStoreProvider for CountingProvider {
        async fn new_store(
            &self,
            base_path: Url,
            params: &ObjectStoreParams,
        ) -> Result<ObjectStore> {
            self.builds.fetch_add(1, Ordering::Relaxed);
            // Force a yield + delay so other tasks reach `get_store` and queue
            // on the build lock before this build completes.
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            memory::MemoryStoreProvider
                .new_store(base_path, params)
                .await
        }
    }

    /// Concurrent cold-misses for the same key must coalesce into a single
    /// `provider.new_store` invocation, with all callers receiving the same Arc.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_get_store_coalesces_concurrent_misses() {
        let provider = Arc::new(CountingProvider::default());
        let registry = Arc::new(ObjectStoreRegistry::empty());
        registry.insert("counting", provider.clone());

        let url = Url::parse("counting://test").unwrap();
        let params = ObjectStoreParams::default();

        let n = 16;
        let mut handles = Vec::with_capacity(n);
        for _ in 0..n {
            let registry = registry.clone();
            let url = url.clone();
            let params = params.clone();
            handles.push(tokio::spawn(async move {
                registry.get_store(url, &params).await.unwrap()
            }));
        }
        let mut stores = Vec::with_capacity(n);
        for h in handles {
            stores.push(h.await.unwrap());
        }

        assert_eq!(
            provider.builds.load(Ordering::Relaxed),
            1,
            "single-flight must collapse N concurrent misses into 1 build"
        );
        // All concurrent callers share the same cached Arc — single-flight
        // produced exactly one ObjectStore and the registry handed it out
        // to every waiter.
        for s in &stores[1..] {
            assert!(Arc::ptr_eq(&stores[0], s));
        }
        let s = registry.stats();
        assert_eq!((s.misses, s.hits as usize, s.active_stores), (1, n - 1, 1));
    }

    /// After a successful build, the per-key build lock must be removed so
    /// `build_locks` does not grow unbounded across many distinct keys.
    #[tokio::test]
    async fn test_get_store_cleans_build_locks_after_use() {
        let registry = ObjectStoreRegistry::default();
        let params = ObjectStoreParams::default();

        for i in 0..5 {
            let url = Url::parse(&format!("memory://cleanup-{}", i)).unwrap();
            let _store = registry.get_store(url, &params).await.unwrap();
        }
        assert_eq!(
            registry.build_locks_len(),
            0,
            "build_locks must be empty once all builds settle"
        );
    }

    /// A provider whose `new_store` panics, used to verify the RAII cleanup
    /// guard reclaims build_locks entries on panic.
    #[derive(Debug, Default)]
    struct PanickingProvider;

    #[async_trait::async_trait]
    impl ObjectStoreProvider for PanickingProvider {
        async fn new_store(
            &self,
            _base_path: Url,
            _params: &ObjectStoreParams,
        ) -> Result<ObjectStore> {
            panic!("boom from provider")
        }
    }

    /// A provider that returns an `Err` after a yield, simulating a normal
    /// (non-panic) build failure such as bad credentials or a network error.
    /// Counts invocations so the test can assert the single-flight collapsed
    /// N concurrent failures into 1 underlying call.
    #[derive(Debug, Default)]
    struct FailingProvider {
        builds: AtomicU64,
    }

    #[async_trait::async_trait]
    impl ObjectStoreProvider for FailingProvider {
        async fn new_store(
            &self,
            _base_path: Url,
            _params: &ObjectStoreParams,
        ) -> Result<ObjectStore> {
            self.builds.fetch_add(1, Ordering::Relaxed);
            // Yield + delay so concurrent callers reach the build lock and
            // queue behind us — this is the configuration that exercises
            // the coalesced-failure path in `get_store`.
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            Err(Error::invalid_input("simulated cold-build failure"))
        }
    }

    /// Build-Err (non-panic) path under coalesced contention asserts the
    /// shipped behavior of PR#1's single-flight: failures **serialize via
    /// the per-key lock but are not cached**. Each of N waiters acquires
    /// the lock in turn, re-checks the (still-empty) cache, and retries
    /// the build. The lock keeps failures from *fanning out in parallel*
    /// (so an upstream that costs T ms sees ~N·T total, not T concurrent),
    /// but it does not collapse them into one call.
    ///
    /// Concretely this test pins:
    ///   1. `provider.new_store` fires once per caller (no Err coalescing),
    ///   2. every concurrent caller surfaces an `Err`,
    ///   3. `build_failures` increments per failed call (not per key),
    ///   4. `active_stores` stays 0 — failed builds must never be cached,
    ///   5. `build_locks` drains back to 0 via the RAII cleanup, even on
    ///      the Err return path.
    ///
    /// Caching errors (with TTL or until-eviction) is a separate design
    /// point. This test exists to lock in the current contract so a future
    /// "let's cache Err for 100ms" change is forced to update this guard
    /// deliberately rather than land silently.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_get_store_serializes_build_failures_without_caching() {
        let provider = Arc::new(FailingProvider::default());
        let registry = Arc::new(ObjectStoreRegistry::empty());
        registry.insert("failing", provider.clone());

        let url = Url::parse("failing://x").unwrap();
        let params = ObjectStoreParams::default();

        let n = 8;
        let mut handles = Vec::with_capacity(n);
        for _ in 0..n {
            let registry = registry.clone();
            let url = url.clone();
            let params = params.clone();
            handles.push(tokio::spawn(async move {
                registry.get_store(url, &params).await
            }));
        }

        let mut err_count = 0;
        for h in handles {
            let result = h.await.expect("task must not panic");
            assert!(result.is_err(), "every waiter must surface Err");
            err_count += 1;
        }
        assert_eq!(err_count, n, "all N callers must observe an error");

        assert_eq!(
            provider.builds.load(Ordering::Relaxed) as usize,
            n,
            "shipped behavior: failures are not cached, so each waiter retries the build",
        );

        let s = registry.stats();
        assert_eq!(
            s.build_failures as usize, n,
            "build_failures counts every failed call (per-call, not per-key)",
        );
        assert_eq!(
            s.active_stores, 0,
            "failed builds must never populate active_stores",
        );
        assert_eq!(
            s.build_locks, 0,
            "RAII cleanup must drain build_locks even on Err path",
        );
    }

    /// If `new_store` panics, the RAII guard must still remove the build_locks
    /// entry — otherwise a single bad URI would leak a lock per failed open.
    ///
    /// Relies on the default `panic = "unwind"` profile: the panic unwinds
    /// the spawned task and `Drop` runs on the way out, which is what triggers
    /// `BuildLockSession`. With `panic = "abort"`, the process dies before
    /// cleanup can run — that's a stricter regime where this test (and the
    /// leak it guards against) becomes moot.
    #[tokio::test]
    async fn test_get_store_cleans_build_locks_after_panic() {
        let registry = Arc::new(ObjectStoreRegistry::empty());
        registry.insert("boom", Arc::new(PanickingProvider));
        let url = Url::parse("boom://x").unwrap();
        let params = ObjectStoreParams::default();

        let registry_for_task = registry.clone();
        let result =
            tokio::task::spawn(async move { registry_for_task.get_store(url, &params).await })
                .await;
        assert!(
            result.as_ref().err().is_some_and(|e| e.is_panic()),
            "task must propagate the panic"
        );
        assert_eq!(
            registry.build_locks_len(),
            0,
            "RAII guard must clean up the build_locks entry on panic"
        );
    }

    /// Regression: two callers resolving the same cache key must receive
    /// the same cached `Arc<ObjectStore>`. Single-flight correctness depends
    /// on this — see `test_get_store_coalesces_concurrent_misses` for the
    /// concurrent variant.
    #[tokio::test]
    async fn test_get_store_returns_shared_arc_on_hit() {
        let registry = ObjectStoreRegistry::default();
        let params = ObjectStoreParams::default();
        let url = Url::parse("memory://shared-arc").unwrap();

        let a = registry.get_store(url.clone(), &params).await.unwrap();
        let b = registry.get_store(url, &params).await.unwrap();

        let s = registry.stats();
        assert_eq!((s.misses, s.hits, s.active_stores), (1, 1, 1));
        assert!(
            Arc::ptr_eq(&a, &b),
            "cache hit must return the same Arc as the cold-build caller"
        );
    }
}
