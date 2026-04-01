// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance cache system.
//!
//! ## For cache users
//!
//! Use [`LanceCache`] (or [`WeakLanceCache`]) to store and retrieve typed
//! values. Define a [`CacheKey`] (or [`UnsizedCacheKey`] for trait objects) to
//! describe what you're caching and its type.
//!
//! ## For backend implementors
//!
//! Implement [`CacheBackend`] to provide a custom storage layer (disk, Redis,
//! etc.). Backends receive [`InternalCacheKey`] keys and type-erased
//! [`CacheEntry`] values — the typed wrapping is handled by [`LanceCache`].
//! See the [`backend`] module for details.

pub mod backend;
mod moka;

pub use backend::{CacheBackend, CacheEntry, InternalCacheKey};
pub use moka::MokaCacheBackend;

use std::borrow::Cow;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use futures::{Future, FutureExt};

use crate::Result;

pub use deepsize::{Context, DeepSizeOf};

// ---------------------------------------------------------------------------
// CacheKey / UnsizedCacheKey — typed key traits for cache users
// ---------------------------------------------------------------------------

/// Typed cache key for sized value types.
///
/// Implement this trait to define a new type of cached entry. [`LanceCache`]
/// uses the key string and type name to construct an [`InternalCacheKey`]
/// for the backend.
///
/// # Example
///
/// ```ignore
/// struct MyKey { id: u64 }
///
/// impl CacheKey for MyKey {
///     type ValueType = MyData;
///     fn key(&self) -> Cow<'_, str> { self.id.to_string().into() }
///     fn type_name() -> &'static str { "MyData" }
/// }
/// ```
pub trait CacheKey {
    type ValueType: 'static;

    fn key(&self) -> Cow<'_, str>;

    /// Short, stable string identifying this value type.
    ///
    /// Two `CacheKey` impls that store different `ValueType`s **must** return
    /// different type names; if they collide, gets will silently return `None`
    /// due to failed downcasts.
    ///
    /// Use a short literal (e.g. `"Vec<IndexMetadata>"`), not
    /// `std::any::type_name` — the latter is not guaranteed stable across
    /// compiler versions or build configurations.
    fn type_name() -> &'static str;
}

/// Like [`CacheKey`] but for unsized value types (e.g. `dyn Trait`).
///
/// The cache wraps values in an extra `Arc` layer internally; callers pass
/// and receive `Arc<T>` where `T: ?Sized`.
pub trait UnsizedCacheKey {
    type ValueType: 'static + ?Sized;

    fn key(&self) -> Cow<'_, str>;

    /// Short, stable string identifying this value type.
    /// See [`CacheKey::type_name`] for requirements.
    fn type_name() -> &'static str;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Size of a cached `Arc<T>`, accounting for the Arc overhead (two atomic counters).
fn cache_entry_size<T: DeepSizeOf + ?Sized>(value: &T) -> usize {
    value.deep_size_of() + std::mem::size_of::<std::sync::atomic::AtomicUsize>() * 2
}

/// Build an [`InternalCacheKey`] from a cache's prefix, a user key string,
/// and a type name.
fn build_key(prefix: &Arc<str>, key: &str, type_name: &'static str) -> InternalCacheKey {
    InternalCacheKey::new(prefix.clone(), Arc::from(key), type_name)
}

// ---------------------------------------------------------------------------
// LanceCache — typed wrapper around dyn CacheBackend
// ---------------------------------------------------------------------------

/// Typed cache wrapper that handles key construction and type safety.
///
/// Internally delegates to a [`CacheBackend`]. The default backend is
/// [`MokaCacheBackend`]; pass a custom backend via [`LanceCache::with_backend`].
#[derive(Clone)]
pub struct LanceCache {
    cache: Arc<dyn CacheBackend>,
    prefix: Arc<str>,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
}

impl std::fmt::Debug for LanceCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceCache")
            .field("cache", &self.cache)
            .finish()
    }
}

impl DeepSizeOf for LanceCache {
    fn deep_size_of_children(&self, _: &mut Context) -> usize {
        self.cache.approx_size_bytes()
    }
}

impl LanceCache {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: Arc::new(MokaCacheBackend::with_capacity(capacity)),
            prefix: Arc::from(""),
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Create a cache backed by a custom [`CacheBackend`].
    pub fn with_backend(backend: Arc<dyn CacheBackend>) -> Self {
        Self {
            cache: backend,
            prefix: Arc::from(""),
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn no_cache() -> Self {
        Self {
            cache: Arc::new(MokaCacheBackend::no_cache()),
            prefix: Arc::from(""),
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Create a cache with the given backend and an exact prefix string.
    /// Unlike `with_key_prefix`, this sets the prefix verbatim (no trailing slash added).
    pub fn with_backend_and_prefix(backend: Arc<dyn CacheBackend>, prefix: String) -> Self {
        Self {
            cache: backend,
            prefix: Arc::from(prefix),
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Appends a prefix to the cache key.
    pub fn with_key_prefix(&self, prefix: &str) -> Self {
        Self {
            cache: self.cache.clone(),
            prefix: Arc::from(format!("{}{}/", self.prefix, prefix)),
            hits: self.hits.clone(),
            misses: self.misses.clone(),
        }
    }

    /// Invalidate all entries whose prefix starts with the given string.
    pub async fn invalidate_prefix(&self, prefix: &str) {
        let full_prefix = format!("{}{}", self.prefix, prefix);
        self.cache.invalidate_prefix(&full_prefix).await;
    }

    pub async fn size(&self) -> usize {
        self.cache.num_entries().await
    }

    pub fn approx_size(&self) -> usize {
        self.cache.approx_num_entries()
    }

    pub async fn size_bytes(&self) -> usize {
        self.cache.size_bytes().await
    }

    // -- Sized insert/get (internal, shared by sized and unsized paths) --------

    async fn insert_with_id<T: DeepSizeOf + Send + Sync + 'static>(
        &self,
        key: &str,
        type_name: &'static str,
        metadata: Arc<T>,
    ) {
        let size = cache_entry_size(&*metadata);
        let cache_key = build_key(&self.prefix, key, type_name);
        self.cache.insert(&cache_key, metadata, size).await;
    }

    async fn get_with_id<T: Send + Sync + 'static>(
        &self,
        key: &str,
        type_name: &'static str,
    ) -> Option<Arc<T>> {
        let cache_key = build_key(&self.prefix, key, type_name);
        if let Some(entry) = self.cache.get(&cache_key).await {
            match entry.downcast::<T>() {
                Ok(val) => {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    Some(val)
                }
                Err(_) => {
                    // Type mismatch: the backend returned a different concrete
                    // type than expected (e.g. a disk cache may store
                    // intermediate state). Treat as a miss.
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    None
                }
            }
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    // -- Stats / clear --------------------------------------------------------

    pub async fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            num_entries: self.cache.num_entries().await,
            size_bytes: self.cache.size_bytes().await,
        }
    }

    pub async fn clear(&self) {
        self.cache.clear().await;
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    // -- CacheKey-based methods -----------------------------------------------

    pub async fn insert_with_key<K>(&self, cache_key: &K, metadata: Arc<K::ValueType>)
    where
        K: CacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
    {
        self.insert_with_id(&cache_key.key(), K::type_name(), metadata)
            .boxed()
            .await
    }

    pub async fn get_with_key<K>(&self, cache_key: &K) -> Option<Arc<K::ValueType>>
    where
        K: CacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
    {
        self.get_with_id::<K::ValueType>(&cache_key.key(), K::type_name())
            .boxed()
            .await
    }

    pub async fn get_or_insert_with_key<K, F, Fut>(
        &self,
        cache_key: K,
        loader: F,
    ) -> Result<Arc<K::ValueType>>
    where
        K: CacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<K::ValueType>> + Send,
    {
        let key = build_key(&self.prefix, &cache_key.key(), K::type_name());

        let typed_loader = Box::pin(async move {
            let value = loader().await?;
            let arc = Arc::new(value);
            let size = cache_entry_size(&*arc);
            Ok((arc as CacheEntry, size))
        });

        let (entry, was_cached) = self.cache.get_or_insert(&key, typed_loader).await?;

        if was_cached {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }

        Ok(entry.downcast::<K::ValueType>().unwrap())
    }

    pub async fn insert_unsized_with_key<K>(&self, cache_key: &K, metadata: Arc<K::ValueType>)
    where
        K: UnsizedCacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
    {
        self.insert_with_id(&cache_key.key(), K::type_name(), Arc::new(metadata))
            .boxed()
            .await
    }

    pub async fn get_unsized_with_key<K>(&self, cache_key: &K) -> Option<Arc<K::ValueType>>
    where
        K: UnsizedCacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
    {
        let outer = self
            .get_with_id::<Arc<K::ValueType>>(&cache_key.key(), K::type_name())
            .boxed()
            .await?;
        Some(outer.as_ref().clone())
    }
}

// ---------------------------------------------------------------------------
// WeakLanceCache
// ---------------------------------------------------------------------------

/// A weak reference to a LanceCache, used by indices to avoid circular references.
/// When the original cache is dropped, operations on this will gracefully no-op.
#[derive(Clone, Debug)]
pub struct WeakLanceCache {
    inner: std::sync::Weak<dyn CacheBackend>,
    prefix: Arc<str>,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
}

impl WeakLanceCache {
    pub fn from(cache: &LanceCache) -> Self {
        Self {
            inner: Arc::downgrade(&cache.cache),
            prefix: cache.prefix.clone(),
            hits: cache.hits.clone(),
            misses: cache.misses.clone(),
        }
    }

    pub fn with_key_prefix(&self, prefix: &str) -> Self {
        Self {
            inner: self.inner.clone(),
            prefix: Arc::from(format!("{}{}/", self.prefix, prefix)),
            hits: self.hits.clone(),
            misses: self.misses.clone(),
        }
    }

    /// The key prefix used for all entries in this cache.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    pub async fn get_with_key<K>(&self, cache_key: &K) -> Option<Arc<K::ValueType>>
    where
        K: CacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
    {
        let cache = self.inner.upgrade()?;
        let key = build_key(&self.prefix, &cache_key.key(), K::type_name());
        if let Some(entry) = cache.get(&key).await {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.downcast::<K::ValueType>().unwrap())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    pub async fn insert_with_key<K>(&self, cache_key: &K, value: Arc<K::ValueType>) -> bool
    where
        K: CacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
    {
        if let Some(cache) = self.inner.upgrade() {
            let size = cache_entry_size(&*value);
            let key = build_key(&self.prefix, &cache_key.key(), K::type_name());
            cache.insert(&key, value, size).await;
            true
        } else {
            log::warn!("WeakLanceCache: cache no longer available, unable to insert item");
            false
        }
    }

    /// Get or insert an item, computing it if necessary.
    ///
    /// Deduplication of concurrent loads is handled by the backend.
    pub async fn get_or_insert_with_key<K, F, Fut>(
        &self,
        cache_key: K,
        loader: F,
    ) -> Result<Arc<K::ValueType>>
    where
        K: CacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<K::ValueType>> + Send,
    {
        if let Some(cache) = self.inner.upgrade() {
            let key = build_key(&self.prefix, &cache_key.key(), K::type_name());
            let typed_loader = Box::pin(async move {
                let value = loader().await?;
                let arc = Arc::new(value);
                let size = cache_entry_size(&*arc);
                Ok((arc as CacheEntry, size))
            });
            let (entry, was_cached) = cache.get_or_insert(&key, typed_loader).await?;
            if was_cached {
                self.hits.fetch_add(1, Ordering::Relaxed);
            } else {
                self.misses.fetch_add(1, Ordering::Relaxed);
            }
            Ok(entry.downcast::<K::ValueType>().unwrap())
        } else {
            log::warn!("WeakLanceCache: cache no longer available, computing without caching");
            loader().await.map(Arc::new)
        }
    }

    pub async fn get_unsized_with_key<K>(&self, cache_key: &K) -> Option<Arc<K::ValueType>>
    where
        K: UnsizedCacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
    {
        let cache = self.inner.upgrade()?;
        let key = build_key(&self.prefix, &cache_key.key(), K::type_name());
        if let Some(entry) = cache.get(&key).await {
            entry
                .downcast::<Arc<K::ValueType>>()
                .ok()
                .map(|arc| arc.as_ref().clone())
        } else {
            None
        }
    }

    pub async fn insert_unsized_with_key<K>(&self, cache_key: &K, value: Arc<K::ValueType>)
    where
        K: UnsizedCacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
    {
        if let Some(cache) = self.inner.upgrade() {
            let wrapper = Arc::new(value);
            let size = cache_entry_size(&*wrapper);
            let key = build_key(&self.prefix, &cache_key.key(), K::type_name());
            cache.insert(&key, wrapper, size).await;
        } else {
            log::warn!("WeakLanceCache: cache no longer available, unable to insert unsized item");
        }
    }
}

// ---------------------------------------------------------------------------
// CacheStats
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of times `get`, `get_unsized`, or `get_or_insert` found an item in the cache.
    pub hits: u64,
    /// Number of times `get`, `get_unsized`, or `get_or_insert` did not find an item in the cache.
    pub misses: u64,
    /// Number of entries currently in the cache.
    pub num_entries: usize,
    /// Total size in bytes of all entries in the cache.
    pub size_bytes: usize,
}

impl CacheStats {
    pub fn hit_ratio(&self) -> f32 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f32 / (self.hits + self.misses) as f32
        }
    }

    pub fn miss_ratio(&self) -> f32 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.misses as f32 / (self.hits + self.misses) as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::marker::PhantomData;

    struct TestKey<T: 'static> {
        key: String,
        _phantom: PhantomData<T>,
    }

    impl<T: 'static> TestKey<T> {
        fn new(key: &str) -> Self {
            Self {
                key: key.to_string(),
                _phantom: PhantomData,
            }
        }
    }

    impl<T: 'static> CacheKey for TestKey<T> {
        type ValueType = T;
        fn key(&self) -> std::borrow::Cow<'_, str> {
            std::borrow::Cow::Borrowed(&self.key)
        }
        fn type_name() -> &'static str {
            std::any::type_name::<T>()
        }
    }

    /// Test helper: an UnsizedCacheKey for trait object values.
    struct TestUnsizedKey<T: 'static + ?Sized> {
        key: String,
        _phantom: PhantomData<T>,
    }

    impl<T: 'static + ?Sized> TestUnsizedKey<T> {
        fn new(key: &str) -> Self {
            Self {
                key: key.to_string(),
                _phantom: PhantomData,
            }
        }
    }

    impl<T: 'static + ?Sized> UnsizedCacheKey for TestUnsizedKey<T> {
        type ValueType = T;
        fn key(&self) -> std::borrow::Cow<'_, str> {
            std::borrow::Cow::Borrowed(&self.key)
        }
        fn type_name() -> &'static str {
            std::any::type_name::<T>()
        }
    }

    #[tokio::test]
    async fn test_cache_bytes() {
        let item = Arc::new(vec![1, 2, 3]);
        let item_size = item.deep_size_of();
        let capacity = 10 * item_size;
        let cache = LanceCache::with_capacity(capacity);

        cache
            .insert_with_key(&TestKey::<Vec<i32>>::new("key"), item.clone())
            .await;
        assert_eq!(cache.size().await, 1);

        let retrieved = cache
            .get_with_key(&TestKey::<Vec<i32>>::new("key"))
            .await
            .unwrap();
        assert_eq!(*retrieved, *item);

        for i in 0..20 {
            cache
                .insert_with_key(
                    &TestKey::<Vec<i32>>::new(&format!("key_{}", i)),
                    Arc::new(vec![i, i, i]),
                )
                .await;
        }
        assert!(cache.size_bytes().await <= capacity);
    }

    #[tokio::test]
    async fn test_cache_trait_objects() {
        #[derive(Debug, DeepSizeOf)]
        struct MyType(i32);

        trait MyTrait: DeepSizeOf + Send + Sync + std::any::Any {
            fn as_any(&self) -> &dyn std::any::Any;
        }

        impl MyTrait for MyType {
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }

        let item: Arc<dyn MyTrait> = Arc::new(MyType(42));
        let cache = LanceCache::with_capacity(1000);
        cache
            .insert_unsized_with_key(&TestUnsizedKey::<dyn MyTrait>::new("test"), item)
            .await;

        let retrieved = cache
            .get_unsized_with_key(&TestUnsizedKey::<dyn MyTrait>::new("test"))
            .await
            .unwrap();
        assert_eq!(retrieved.as_any().downcast_ref::<MyType>().unwrap().0, 42);
    }

    #[tokio::test]
    async fn test_cache_stats_basic() {
        let cache = LanceCache::with_capacity(1000);
        assert_eq!(cache.stats().await.hits, 0);

        // Miss
        assert!(
            cache
                .get_with_key(&TestKey::<Vec<i32>>::new("x"))
                .await
                .is_none()
        );
        assert_eq!(cache.stats().await.misses, 1);

        // Insert then hit
        cache
            .insert_with_key(&TestKey::new("k"), Arc::new(vec![1, 2, 3]))
            .await;
        assert!(
            cache
                .get_with_key(&TestKey::<Vec<i32>>::new("k"))
                .await
                .is_some()
        );
        assert_eq!(cache.stats().await.hits, 1);
    }

    #[tokio::test]
    async fn test_cache_stats_with_prefixes() {
        let base = LanceCache::with_capacity(1000);
        let prefixed = base.with_key_prefix("ns");

        assert!(
            prefixed
                .get_with_key(&TestKey::<Vec<i32>>::new("k"))
                .await
                .is_none()
        );
        assert_eq!(base.stats().await.misses, 1);

        prefixed
            .insert_with_key(&TestKey::new("k"), Arc::new(vec![1]))
            .await;
        assert!(
            prefixed
                .get_with_key(&TestKey::<Vec<i32>>::new("k"))
                .await
                .is_some()
        );
        assert_eq!(base.stats().await.hits, 1);
    }

    #[tokio::test]
    async fn test_cache_get_or_insert() {
        let cache = LanceCache::with_capacity(1000);

        let v: Arc<Vec<i32>> = cache
            .get_or_insert_with_key(TestKey::<Vec<i32>>::new("k"), || async {
                Ok(vec![1, 2, 3])
            })
            .await
            .unwrap();
        assert_eq!(*v, vec![1, 2, 3]);
        assert_eq!(cache.stats().await.misses, 1);
        assert_eq!(cache.stats().await.hits, 0);

        // Second call should not invoke loader and should be a hit
        let v: Arc<Vec<i32>> = cache
            .get_or_insert_with_key(TestKey::<Vec<i32>>::new("k"), || async {
                panic!("should not be called")
            })
            .await
            .unwrap();
        assert_eq!(*v, vec![1, 2, 3]);
        assert_eq!(cache.stats().await.hits, 1);
    }

    #[tokio::test]
    async fn test_custom_backend() {
        use async_trait::async_trait;
        use tokio::sync::Mutex;

        #[derive(Debug)]
        struct HashMapBackend {
            map: Mutex<HashMap<InternalCacheKey, (CacheEntry, usize)>>,
        }

        impl HashMapBackend {
            fn new() -> Self {
                Self {
                    map: Mutex::new(HashMap::new()),
                }
            }
        }

        #[async_trait]
        impl CacheBackend for HashMapBackend {
            async fn get(&self, key: &InternalCacheKey) -> Option<CacheEntry> {
                self.map.lock().await.get(key).map(|(e, _)| e.clone())
            }
            async fn insert(&self, key: &InternalCacheKey, entry: CacheEntry, size_bytes: usize) {
                self.map
                    .lock()
                    .await
                    .insert(key.clone(), (entry, size_bytes));
            }
            async fn get_or_insert<'a>(
                &self,
                key: &InternalCacheKey,
                loader: std::pin::Pin<
                    Box<dyn futures::Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>,
                >,
            ) -> Result<(CacheEntry, bool)> {
                if let Some((entry, _)) = self.map.lock().await.get(key) {
                    Ok((entry.clone(), true))
                } else {
                    let (entry, size) = loader.await?;
                    self.map
                        .lock()
                        .await
                        .insert(key.clone(), (entry.clone(), size));
                    Ok((entry, false))
                }
            }
            async fn invalidate_prefix(&self, prefix: &str) {
                self.map.lock().await.retain(|k, _| !k.starts_with(prefix));
            }
            async fn clear(&self) {
                self.map.lock().await.clear();
            }
            async fn num_entries(&self) -> usize {
                self.map.lock().await.len()
            }
            async fn size_bytes(&self) -> usize {
                self.map.lock().await.values().map(|(_, s)| *s).sum()
            }
        }

        let cache = LanceCache::with_backend(Arc::new(HashMapBackend::new()));

        cache
            .insert_with_key(&TestKey::new("k"), Arc::new(vec![1, 2, 3]))
            .await;
        assert!(
            cache
                .get_with_key(&TestKey::<Vec<i32>>::new("k"))
                .await
                .is_some()
        );
        // Different type at same key = miss
        assert!(
            cache
                .get_with_key(&TestKey::<Vec<u8>>::new("k"))
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_get_or_insert_dedup() {
        use std::sync::atomic::AtomicUsize;

        let load_count = Arc::new(AtomicUsize::new(0));
        let cache = LanceCache::with_capacity(10000);

        let (barrier_tx, _) = tokio::sync::broadcast::channel::<()>(1);
        let mut handles = Vec::new();
        for _ in 0..5 {
            let cache = cache.clone();
            let load_count = load_count.clone();
            let mut barrier_rx = barrier_tx.subscribe();
            handles.push(tokio::spawn(async move {
                barrier_rx.recv().await.ok();
                cache
                    .get_or_insert_with_key(TestKey::<Vec<i32>>::new("key"), || {
                        let load_count = load_count.clone();
                        async move {
                            load_count.fetch_add(1, Ordering::SeqCst);
                            tokio::task::yield_now().await;
                            Ok(vec![1, 2, 3])
                        }
                    })
                    .await
            }));
        }
        barrier_tx.send(()).unwrap();
        for h in handles {
            let result: Arc<Vec<i32>> = h.await.unwrap().unwrap();
            assert_eq!(*result, vec![1, 2, 3]);
        }

        assert_eq!(load_count.load(Ordering::SeqCst), 1);
    }
}
