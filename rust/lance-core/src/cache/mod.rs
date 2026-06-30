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
//! To make a value type serializable (so persistent backends can store it),
//! implement [`CacheCodecImpl`] on the type, then override [`CacheKey::codec`]:
//!
//! ```ignore
//! impl CacheCodecImpl for MyData {
//!     fn serialize(&self, w: &mut dyn Write) -> Result<()> { /* ... */ }
//!     fn deserialize(data: &Bytes) -> Result<Self> { /* ... */ }
//! }
//!
//! impl CacheKey for MyDataKey {
//!     type ValueType = MyData;
//!     fn key(&self) -> Cow<'_, str> { /* ... */ }
//!     fn type_name() -> &'static str { "MyData" }
//!     fn codec() -> Option<CacheCodec> {
//!         Some(CacheCodec::from_impl::<MyData>())
//!     }
//! }
//! ```
//!
//! ## For backend implementors
//!
//! Implement [`CacheBackend`] to provide a custom storage layer (disk, Redis,
//! etc.). Backends receive [`InternalCacheKey`] keys and type-erased
//! [`CacheEntry`] values — the typed wrapping is handled by [`LanceCache`].
//! See the [`backend`] module for details.
//!
//! ## Serialization flow
//!
//! When a [`CacheKey`] provides a codec via [`CacheKey::codec`]:
//!
//! 1. [`LanceCache`] wraps the [`CacheCodec`] and passes it to the backend
//!    alongside the entry on `insert` and `get` calls.
//! 2. In-memory backends (like [`MokaCacheBackend`]) ignore the codec.
//! 3. Persistent backends use `codec.serialize(entry, writer)` on insert and
//!    `codec.deserialize(reader)` on get to persist entries across restarts.

pub mod backend;
pub mod codec;
mod entry_io;
mod moka;

pub use backend::{
    CacheBackend, CacheBatchEntry, CacheBatchLoader, CacheEntry, CacheKeyIterator,
    CacheLoadedEntry, InternalCacheKey,
};
pub use codec::{
    CacheCodec, CacheCodecImpl, CacheDecode, CacheMissReason, MAGIC, has_cache_envelope,
};
pub use entry_io::{CacheEntryReader, CacheEntryWriter};
pub use moka::MokaCacheBackend;

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use futures::{Future, FutureExt};

use crate::{Error, Result};

pub use crate::deepsize::{Context, DeepSizeOf};

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

    /// Optional codec for serializing/deserializing this key's value type.
    ///
    /// Returns `None` by default. Cache backends that support persistence
    /// (e.g. disk-backed caches) use this to serialize entries on insert and
    /// deserialize on get. Types without a codec will only be stored in-memory.
    ///
    /// [`CacheCodec`] is `Copy` (two plain function pointers), so returning it
    /// by value is cheap — no allocation needed.
    fn codec() -> Option<CacheCodec> {
        None
    }
}

/// Like [`CacheKey`] but for unsized value types (e.g. `dyn Trait`).
///
/// The cache wraps values in an extra `Arc` layer internally; callers pass
/// and receive `Arc<T>` where `T: ?Sized`.
///
/// Unsized cache entries are always in-memory only (no serialization codec).
/// For serializable entries, use a sized [`CacheKey`] instead.
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

fn build_batch_keys<K>(
    prefix: &Arc<str>,
    cache_keys: &[K],
) -> Result<(Vec<InternalCacheKey>, HashMap<InternalCacheKey, K>)>
where
    K: CacheKey + Clone,
{
    let mut keys = Vec::with_capacity(cache_keys.len());
    let mut typed_keys = HashMap::with_capacity(cache_keys.len());

    for cache_key in cache_keys {
        let key = build_key(prefix, &cache_key.key(), K::type_name());
        if typed_keys.insert(key.clone(), cache_key.clone()).is_some() {
            return Err(Error::invalid_input(format!(
                "duplicate cache key in get_or_insert_with_key_batch: prefix='{}', key='{}', type='{}'",
                key.prefix(),
                key.key(),
                key.type_name()
            )));
        }
        keys.push(key);
    }

    Ok((keys, typed_keys))
}

/// Converts typed batch requests into the backend's type-erased batch API while
/// keeping cache coordination and single-flight ownership inside the backend.
async fn get_or_insert_batch_with_backend<K, F, Fut>(
    cache: &dyn CacheBackend,
    prefix: &Arc<str>,
    hits: &AtomicU64,
    misses: &AtomicU64,
    cache_keys: Vec<K>,
    loader: F,
) -> Result<Vec<CacheBatchValue<K::ValueType>>>
where
    K: CacheKey + Clone + Send + Sync,
    K::ValueType: DeepSizeOf + Send + Sync + 'static,
    F: Fn(Vec<K>) -> Fut + Send + Sync,
    Fut: Future<Output = Result<Vec<(K, K::ValueType)>>> + Send,
{
    let (keys, typed_keys) = build_batch_keys(prefix, &cache_keys)?;
    let typed_keys = Arc::new(typed_keys);
    let loader = Arc::new(loader);
    let prefix = prefix.clone();

    let typed_loader: CacheBatchLoader<'_> = Arc::new(move |owned_keys| {
        let typed_keys = typed_keys.clone();
        let loader = loader.clone();
        let prefix = prefix.clone();

        async move {
            let mut loader_keys = Vec::with_capacity(owned_keys.len());
            for key in &owned_keys {
                loader_keys.push(
                    typed_keys
                        .get(key)
                        .ok_or_else(|| {
                            Error::internal(format!(
                                "backend requested unknown cache key: prefix='{}', key='{}', type='{}'",
                                key.prefix(),
                                key.key(),
                                key.type_name()
                            ))
                        })?
                        .clone(),
                );
            }

            loader(loader_keys)
                .await?
                .into_iter()
                .map(|(cache_key, value)| {
                    let key = build_key(&prefix, &cache_key.key(), K::type_name());
                    let entry = Arc::new(value);
                    let size_bytes = cache_entry_size(&*entry);
                    Ok(CacheLoadedEntry {
                        key,
                        entry: entry as CacheEntry,
                        size_bytes,
                    })
                })
                .collect::<Result<Vec<_>>>()
        }
        .boxed()
    });

    let entries = cache
        .get_or_insert_many(keys, typed_loader, K::codec())
        .await?;

    let mut values = Vec::with_capacity(entries.len());
    for entry in entries {
        let CacheBatchEntry {
            key,
            entry,
            was_cached,
        } = entry;
        if was_cached {
            hits.fetch_add(1, Ordering::Relaxed);
        } else {
            misses.fetch_add(1, Ordering::Relaxed);
        }
        let value = entry.downcast::<K::ValueType>().map_err(|_| {
            Error::internal(format!(
                "cache entry type mismatch for key: prefix='{}', key='{}', type='{}'",
                key.prefix(),
                key.key(),
                key.type_name()
            ))
        })?;
        values.push(CacheBatchValue { value, was_cached });
    }

    Ok(values)
}

/// Preserves the typed batch contract when a WeakLanceCache has lost its
/// backend: no cache coordination, exact loader validation, input-order output.
async fn load_batch_without_cache<K, F, Fut>(
    prefix: &Arc<str>,
    cache_keys: Vec<K>,
    loader: F,
) -> Result<Vec<CacheBatchValue<K::ValueType>>>
where
    K: CacheKey + Clone + Send + Sync,
    K::ValueType: DeepSizeOf + Send + Sync + 'static,
    F: FnOnce(Vec<K>) -> Fut + Send,
    Fut: Future<Output = Result<Vec<(K, K::ValueType)>>> + Send,
{
    if cache_keys.is_empty() {
        return Ok(Vec::new());
    }

    let (keys, expected_keys) = build_batch_keys(prefix, &cache_keys)?;
    let mut loaded = HashMap::with_capacity(keys.len());
    for (cache_key, value) in loader(cache_keys).await? {
        let key = build_key(prefix, &cache_key.key(), K::type_name());
        if !expected_keys.contains_key(&key) {
            return Err(Error::invalid_input(format!(
                "batch cache loader returned unexpected key: prefix='{}', key='{}', type='{}'",
                key.prefix(),
                key.key(),
                key.type_name()
            )));
        }
        let loaded_key = key.clone();
        if loaded.insert(key, Arc::new(value)).is_some() {
            return Err(Error::invalid_input(format!(
                "batch cache loader returned duplicate keys: prefix='{}', key='{}', type='{}'",
                loaded_key.prefix(),
                loaded_key.key(),
                loaded_key.type_name()
            )));
        }
    }

    let mut values = Vec::with_capacity(keys.len());
    for key in keys {
        let value = loaded.remove(&key).ok_or_else(|| {
            Error::invalid_input(format!(
                "batch cache loader did not return expected key: prefix='{}', key='{}', type='{}'",
                key.prefix(),
                key.key(),
                key.type_name()
            ))
        })?;
        values.push(CacheBatchValue {
            value,
            was_cached: false,
        });
    }
    Ok(values)
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

    /// Return an iterator over keys currently stored under this cache's prefix.
    ///
    /// Returns `None` when the backend does not support key inventory. The
    /// iterator is intended for diagnostics and may be weakly consistent with
    /// concurrent cache mutations.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::{borrow::Cow, sync::Arc};
    /// # use lance_core::cache::{CacheKey, LanceCache};
    /// # struct MyKey;
    /// # impl CacheKey for MyKey {
    /// #     type ValueType = Vec<i32>;
    /// #     fn key(&self) -> Cow<'_, str> { Cow::Borrowed("my-key") }
    /// #     fn type_name() -> &'static str { "VecI32" }
    /// # }
    /// # async fn example() {
    /// let cache = LanceCache::with_capacity(1024);
    /// cache.insert_with_key(&MyKey, Arc::new(vec![1, 2, 3])).await;
    ///
    /// let mut keys = cache.keys().await.expect("Moka supports key inventory");
    /// assert_eq!(keys.next().unwrap().key(), "my-key");
    /// # }
    /// ```
    pub async fn keys(&self) -> Option<CacheKeyIterator<'_>> {
        Some(Box::new(
            self.cache
                .keys()
                .await?
                .filter(|key| key.starts_with(&self.prefix)),
        ))
    }

    // -- Sized insert/get (internal, shared by sized and unsized paths) --------

    async fn insert_with_id<T: DeepSizeOf + Send + Sync + 'static>(
        &self,
        key: &str,
        type_name: &'static str,
        codec: Option<CacheCodec>,
        metadata: Arc<T>,
    ) {
        let size = cache_entry_size(&*metadata);
        let cache_key = build_key(&self.prefix, key, type_name);
        self.cache.insert(&cache_key, metadata, size, codec).await;
    }

    async fn get_with_id<T: Send + Sync + 'static>(
        &self,
        key: &str,
        type_name: &'static str,
        codec: Option<CacheCodec>,
    ) -> Option<Arc<T>> {
        let cache_key = build_key(&self.prefix, key, type_name);
        if let Some(entry) = self.cache.get(&cache_key, codec).await {
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
        self.insert_with_id(&cache_key.key(), K::type_name(), K::codec(), metadata)
            .boxed()
            .await
    }

    pub async fn get_with_key<K>(&self, cache_key: &K) -> Option<Arc<K::ValueType>>
    where
        K: CacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
    {
        self.get_with_id::<K::ValueType>(&cache_key.key(), K::type_name(), K::codec())
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

        let (entry, was_cached) = self
            .cache
            .get_or_insert(&key, typed_loader, K::codec())
            .await?;

        if was_cached {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }

        entry.downcast::<K::ValueType>().map_err(|_| {
            Error::internal(format!(
                "cache entry type mismatch for key: prefix='{}', key='{}', type='{}'",
                key.prefix(),
                key.key(),
                key.type_name()
            ))
        })
    }

    /// Get or insert a batch of typed cache entries.
    ///
    /// `cache_keys` must be unique. The returned entries follow the same order
    /// as `cache_keys`. The loader receives only missing keys owned by this
    /// call, preserving their input order, and must return exactly one value
    /// for each received key.
    ///
    /// The loader is `Fn`, not `FnOnce`, because a backend may call it more
    /// than once during one batch request if keys need to be retried after
    /// another in-flight owner fails or is dropped. Custom backends that do
    /// not override [`CacheBackend::get_or_insert_many`] use a compatibility
    /// fallback that invokes the loader one key at a time.
    ///
    /// Use this when the loader can benefit from receiving multiple missing
    /// keys at once, for example by coalescing adjacent reads. This is not a
    /// general faster path: for isolated keys, cheap loaders, or disjoint
    /// concurrent batches, [`get_or_insert_with_key`](Self::get_or_insert_with_key)
    /// avoids the batch result maps and per-key coordination needed to preserve
    /// ordering and single-flight behavior.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::borrow::Cow;
    /// # use lance_core::{Result, cache::{CacheKey, LanceCache}};
    /// #
    /// # #[derive(Clone)]
    /// # struct PageKey(u32);
    /// #
    /// # impl CacheKey for PageKey {
    /// #     type ValueType = usize;
    /// #
    /// #     fn key(&self) -> Cow<'_, str> {
    /// #         Cow::Owned(self.0.to_string())
    /// #     }
    /// #
    /// #     fn type_name() -> &'static str {
    /// #         "PageKey"
    /// #     }
    /// # }
    /// #
    /// # async fn example(cache: LanceCache) -> Result<()> {
    /// let values = cache
    ///     .get_or_insert_with_key_batch(vec![PageKey(1), PageKey(2)], |keys| async move {
    ///         Ok(keys
    ///             .into_iter()
    ///             .map(|key| {
    ///                 let value = key.0 as usize;
    ///                 (key, value)
    ///             })
    ///             .collect())
    ///     })
    ///     .await?;
    ///
    /// assert_eq!(*values[0].value, 1);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_or_insert_with_key_batch<K, F, Fut>(
        &self,
        cache_keys: Vec<K>,
        loader: F,
    ) -> Result<Vec<CacheBatchValue<K::ValueType>>>
    where
        K: CacheKey + Clone + Send + Sync,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
        F: Fn(Vec<K>) -> Fut + Send + Sync,
        Fut: Future<Output = Result<Vec<(K, K::ValueType)>>> + Send,
    {
        get_or_insert_batch_with_backend(
            self.cache.as_ref(),
            &self.prefix,
            &self.hits,
            &self.misses,
            cache_keys,
            loader,
        )
        .await
    }

    pub async fn insert_unsized_with_key<K>(&self, cache_key: &K, metadata: Arc<K::ValueType>)
    where
        K: UnsizedCacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
    {
        self.insert_with_id(&cache_key.key(), K::type_name(), None, Arc::new(metadata))
            .boxed()
            .await
    }

    pub async fn get_unsized_with_key<K>(&self, cache_key: &K) -> Option<Arc<K::ValueType>>
    where
        K: UnsizedCacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
    {
        let outer = self
            .get_with_id::<Arc<K::ValueType>>(&cache_key.key(), K::type_name(), None)
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
        if let Some(entry) = cache.get(&key, K::codec()).await {
            match entry.downcast::<K::ValueType>() {
                Ok(value) => {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    Some(value)
                }
                Err(_) => {
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    log::warn!(
                        "cache entry type mismatch for key: prefix='{}', key='{}', type='{}'",
                        key.prefix(),
                        key.key(),
                        key.type_name()
                    );
                    None
                }
            }
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
            cache.insert(&key, value, size, K::codec()).await;
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
            let (entry, was_cached) = cache.get_or_insert(&key, typed_loader, K::codec()).await?;
            if was_cached {
                self.hits.fetch_add(1, Ordering::Relaxed);
            } else {
                self.misses.fetch_add(1, Ordering::Relaxed);
            }
            entry.downcast::<K::ValueType>().map_err(|_| {
                Error::internal(format!(
                    "cache entry type mismatch for key: prefix='{}', key='{}', type='{}'",
                    key.prefix(),
                    key.key(),
                    key.type_name()
                ))
            })
        } else {
            log::warn!("WeakLanceCache: cache no longer available, computing without caching");
            loader().await.map(Arc::new)
        }
    }

    /// Get or insert a batch of typed cache entries.
    ///
    /// If the backing cache is still available, this has the same semantics as
    /// [`LanceCache::get_or_insert_with_key_batch`]. If the backing cache has
    /// been dropped, the loader runs without caching and all returned entries
    /// are validated, reordered to match the input keys, and marked as not
    /// cached.
    pub async fn get_or_insert_with_key_batch<K, F, Fut>(
        &self,
        cache_keys: Vec<K>,
        loader: F,
    ) -> Result<Vec<CacheBatchValue<K::ValueType>>>
    where
        K: CacheKey + Clone + Send + Sync,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
        F: Fn(Vec<K>) -> Fut + Send + Sync,
        Fut: Future<Output = Result<Vec<(K, K::ValueType)>>> + Send,
    {
        if let Some(cache) = self.inner.upgrade() {
            get_or_insert_batch_with_backend(
                cache.as_ref(),
                &self.prefix,
                &self.hits,
                &self.misses,
                cache_keys,
                loader,
            )
            .await
        } else {
            log::warn!("WeakLanceCache: cache no longer available, computing without caching");
            load_batch_without_cache(&self.prefix, cache_keys, loader).await
        }
    }

    pub async fn get_unsized_with_key<K>(&self, cache_key: &K) -> Option<Arc<K::ValueType>>
    where
        K: UnsizedCacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
    {
        let cache = self.inner.upgrade()?;
        let key = build_key(&self.prefix, &cache_key.key(), K::type_name());
        if let Some(entry) = cache.get(&key, None).await {
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
            cache.insert(&key, wrapper, size, None).await;
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
    /// Number of cache operations satisfied without running this call's loader.
    ///
    /// For `get`/`get_unsized`, this means the entry was found in the cache.
    /// For `get_or_insert`/batch get-or-insert, this means the entry was
    /// either already cached or was loaded by another in-flight owner. A
    /// get-or-insert caller that initially missed but waited for another owner
    /// is counted here, not in `misses`.
    ///
    /// Batch get-or-insert calls count each returned key independently, not the
    /// batch request as a single operation.
    pub hits: u64,
    /// Number of cache operations that were not satisfied from cache/owner.
    ///
    /// For `get`/`get_unsized`, this means the entry was not found. For
    /// `get_or_insert`/batch get-or-insert, this means the current call
    /// executed the loader for that entry.
    ///
    /// Batch get-or-insert calls count each returned key independently, not the
    /// batch request as a single operation.
    pub misses: u64,
    /// Number of entries currently in the cache.
    pub num_entries: usize,
    /// Total size in bytes of all entries in the cache.
    pub size_bytes: usize,
}

/// A typed value returned by batch get-or-insert.
///
/// Includes both the cache value and whether this call avoided running its
/// loader for the key.
#[derive(Debug, Clone)]
pub struct CacheBatchValue<T> {
    /// Cached or loaded value.
    pub value: Arc<T>,
    /// True when this call did not run the loader for this key.
    ///
    /// This includes ordinary cache hits and values loaded by another
    /// in-flight owner.
    pub was_cached: bool,
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
    use rstest::rstest;
    use std::collections::{BTreeSet, HashMap};
    use std::marker::PhantomData;

    #[derive(Clone)]
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

    fn key_fields(keys: &[InternalCacheKey]) -> BTreeSet<(String, String, &'static str)> {
        keys.iter()
            .map(|key| {
                (
                    key.prefix().to_string(),
                    key.key().to_string(),
                    key.type_name(),
                )
            })
            .collect()
    }

    fn assert_invalid_input_contains(err: Error, snippets: &[&str]) {
        assert!(matches!(err, Error::InvalidInput { .. }));
        let message = err.to_string();
        for snippet in snippets {
            assert!(
                message.contains(snippet),
                "expected error message to contain '{snippet}', got: {message}"
            );
        }
    }

    #[derive(Debug)]
    struct HashMapBackend {
        map: tokio::sync::Mutex<HashMap<InternalCacheKey, (CacheEntry, usize)>>,
    }

    impl HashMapBackend {
        fn new() -> Self {
            Self {
                map: tokio::sync::Mutex::new(HashMap::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl CacheBackend for HashMapBackend {
        async fn get(
            &self,
            key: &InternalCacheKey,
            _codec: Option<CacheCodec>,
        ) -> Option<CacheEntry> {
            self.map.lock().await.get(key).map(|(e, _)| e.clone())
        }

        async fn insert(
            &self,
            key: &InternalCacheKey,
            entry: CacheEntry,
            size_bytes: usize,
            _codec: Option<CacheCodec>,
        ) {
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
            _codec: Option<CacheCodec>,
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
    async fn test_cache_keys_with_prefixes() {
        let base = LanceCache::with_capacity(1000);
        let prefixed = base.with_key_prefix("ns");
        let nested = prefixed.with_key_prefix("index");
        let other = base.with_key_prefix("ns-other");

        base.insert_with_key(&TestKey::new("root"), Arc::new(vec![0]))
            .await;
        prefixed
            .insert_with_key(&TestKey::new("child"), Arc::new(vec![1]))
            .await;
        nested
            .insert_with_key(&TestKey::new("nested"), Arc::new(vec![2]))
            .await;
        other
            .insert_with_key(&TestKey::new("other"), Arc::new(vec![3]))
            .await;

        let base_keys = base.keys().await.unwrap().collect::<Vec<_>>();
        assert_eq!(
            key_fields(&base_keys),
            BTreeSet::from([
                (
                    "".to_string(),
                    "root".to_string(),
                    TestKey::<Vec<i32>>::type_name()
                ),
                (
                    "ns/".to_string(),
                    "child".to_string(),
                    TestKey::<Vec<i32>>::type_name()
                ),
                (
                    "ns/index/".to_string(),
                    "nested".to_string(),
                    TestKey::<Vec<i32>>::type_name()
                ),
                (
                    "ns-other/".to_string(),
                    "other".to_string(),
                    TestKey::<Vec<i32>>::type_name()
                ),
            ])
        );

        let prefixed_keys = prefixed.keys().await.unwrap().collect::<Vec<_>>();
        assert_eq!(
            key_fields(&prefixed_keys),
            BTreeSet::from([
                (
                    "ns/".to_string(),
                    "child".to_string(),
                    TestKey::<Vec<i32>>::type_name()
                ),
                (
                    "ns/index/".to_string(),
                    "nested".to_string(),
                    TestKey::<Vec<i32>>::type_name()
                ),
            ])
        );
    }

    #[tokio::test]
    async fn test_cache_keys_reflect_invalidation_and_clear() {
        let base = LanceCache::with_capacity(1000);
        let prefixed = base.with_key_prefix("ns");
        let other = base.with_key_prefix("other");

        prefixed
            .insert_with_key(&TestKey::new("child"), Arc::new(vec![1]))
            .await;
        other
            .insert_with_key(&TestKey::new("other"), Arc::new(vec![2]))
            .await;
        assert_eq!(base.keys().await.unwrap().count(), 2);

        prefixed.invalidate_prefix("").await;
        let keys = base.keys().await.unwrap().collect::<Vec<_>>();
        assert_eq!(
            key_fields(&keys),
            BTreeSet::from([(
                "other/".to_string(),
                "other".to_string(),
                TestKey::<Vec<i32>>::type_name()
            )])
        );

        base.clear().await;
        assert_eq!(base.keys().await.unwrap().count(), 0);
    }

    #[tokio::test]
    async fn test_clear_does_not_cancel_in_flight_get_or_insert() {
        let cache = LanceCache::with_capacity(1000);
        let loader_started = Arc::new(tokio::sync::Notify::new());
        let finish_loader = Arc::new(tokio::sync::Notify::new());

        let task_cache = cache.clone();
        let loader_started_clone = loader_started.clone();
        let finish_loader_clone = finish_loader.clone();
        let load = tokio::spawn(async move {
            task_cache
                .get_or_insert_with_key(TestKey::<Vec<i32>>::new("k"), move || {
                    let loader_started = loader_started_clone;
                    let finish_loader = finish_loader_clone;
                    async move {
                        loader_started.notify_waiters();
                        finish_loader.notified().await;
                        Ok(vec![1, 2, 3])
                    }
                })
                .await
        });

        loader_started.notified().await;
        cache.clear().await;
        assert_eq!(cache.size().await, 0);

        finish_loader.notify_waiters();
        let loaded = load.await.unwrap().unwrap();
        assert_eq!(*loaded, vec![1, 2, 3]);

        let cached = cache
            .get_with_key(&TestKey::<Vec<i32>>::new("k"))
            .await
            .unwrap();
        assert_eq!(*cached, vec![1, 2, 3]);
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
        assert!(cache.keys().await.is_none());
    }

    #[tokio::test]
    async fn test_get_or_insert_batch_uses_backend_default_fallback() {
        let cache = LanceCache::with_backend(Arc::new(HashMapBackend::new()));
        let loader_calls = Arc::new(tokio::sync::Mutex::new(Vec::new()));

        let values = cache
            .get_or_insert_with_key_batch(
                vec![
                    TestKey::<Vec<i32>>::new("1"),
                    TestKey::<Vec<i32>>::new("2"),
                    TestKey::<Vec<i32>>::new("3"),
                ],
                {
                    let loader_calls = loader_calls.clone();
                    move |owned_keys| {
                        let loader_calls = loader_calls.clone();
                        async move {
                            loader_calls.lock().await.push(
                                owned_keys
                                    .iter()
                                    .map(|key| key.key.clone())
                                    .collect::<Vec<_>>(),
                            );
                            Ok(owned_keys
                                .into_iter()
                                .map(|key| {
                                    let value = key.key.parse::<i32>().unwrap();
                                    (key, vec![value])
                                })
                                .collect())
                        }
                    }
                },
            )
            .await
            .unwrap();

        assert_eq!(
            values
                .iter()
                .map(|entry| entry.value.as_ref().clone())
                .collect::<Vec<_>>(),
            vec![vec![1], vec![2], vec![3]]
        );
        assert_eq!(
            values
                .iter()
                .map(|entry| entry.was_cached)
                .collect::<Vec<_>>(),
            vec![false, false, false]
        );
        assert_eq!(
            *loader_calls.lock().await,
            vec![
                vec!["1".to_string()],
                vec!["2".to_string()],
                vec!["3".to_string()]
            ]
        );

        let cached_values = cache
            .get_or_insert_with_key_batch(
                vec![TestKey::<Vec<i32>>::new("2"), TestKey::<Vec<i32>>::new("3")],
                |_| async { panic!("cached fallback keys should not call loader") },
            )
            .await
            .unwrap();
        assert_eq!(
            cached_values
                .iter()
                .map(|entry| entry.was_cached)
                .collect::<Vec<_>>(),
            vec![true, true]
        );
    }

    #[tokio::test]
    async fn test_weak_get_or_insert_batch_empty_after_cache_drop_skips_loader() {
        let weak_cache = {
            let cache = LanceCache::with_capacity(10000);
            WeakLanceCache::from(&cache)
        };

        let values: Vec<CacheBatchValue<Vec<i32>>> = weak_cache
            .get_or_insert_with_key_batch(Vec::<TestKey<Vec<i32>>>::new(), |_| async {
                panic!("empty weak-cache batch should not call loader")
            })
            .await
            .unwrap();

        assert!(values.is_empty());
    }

    #[tokio::test]
    async fn test_weak_get_or_insert_batch_after_cache_drop_loads_without_cache() {
        let weak_cache = {
            let cache = LanceCache::with_capacity(10000);
            WeakLanceCache::from(&cache)
        };

        let values = weak_cache
            .get_or_insert_with_key_batch(
                vec![
                    TestKey::<Vec<i32>>::new("2"),
                    TestKey::<Vec<i32>>::new("1"),
                    TestKey::<Vec<i32>>::new("3"),
                ],
                |keys| async move {
                    assert_eq!(
                        keys.iter().map(|key| key.key.as_str()).collect::<Vec<_>>(),
                        vec!["2", "1", "3"]
                    );
                    Ok(vec![
                        (keys[2].clone(), vec![3]),
                        (keys[1].clone(), vec![1]),
                        (keys[0].clone(), vec![2]),
                    ])
                },
            )
            .await
            .unwrap();

        assert_eq!(
            values
                .iter()
                .map(|entry| entry.value.as_ref().clone())
                .collect::<Vec<_>>(),
            vec![vec![2], vec![1], vec![3]]
        );
        assert_eq!(
            values
                .iter()
                .map(|entry| entry.was_cached)
                .collect::<Vec<_>>(),
            vec![false, false, false]
        );

        let err = weak_cache
            .get_or_insert_with_key_batch(vec![TestKey::<Vec<i32>>::new("1")], |_| async {
                Ok(vec![(TestKey::<Vec<i32>>::new("2"), vec![2])])
            })
            .await
            .unwrap_err();

        assert_invalid_input_contains(err, &["unexpected key", "key='2'"]);

        let err = weak_cache
            .get_or_insert_with_key_batch(vec![TestKey::<Vec<i32>>::new("1")], |_| async {
                Ok(vec![
                    (TestKey::<Vec<i32>>::new("1"), vec![1]),
                    (TestKey::<Vec<i32>>::new("1"), vec![10]),
                ])
            })
            .await
            .unwrap_err();

        assert_invalid_input_contains(err, &["duplicate keys", "key='1'"]);
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

    #[tokio::test]
    async fn test_get_or_insert_batch_dedups_overlapping_keys() {
        use std::sync::atomic::AtomicUsize;
        use tokio::time::{Duration, sleep};

        let cache = LanceCache::with_capacity(10000);
        let load_counts = Arc::new(
            (0..5)
                .map(|_| AtomicUsize::new(0))
                .collect::<Vec<AtomicUsize>>(),
        );

        let (barrier_tx, _) = tokio::sync::broadcast::channel::<()>(1);
        let mut handles = Vec::new();
        for keys in [
            vec![
                TestKey::<Vec<i32>>::new("1"),
                TestKey::<Vec<i32>>::new("2"),
                TestKey::<Vec<i32>>::new("3"),
            ],
            vec![
                TestKey::<Vec<i32>>::new("2"),
                TestKey::<Vec<i32>>::new("3"),
                TestKey::<Vec<i32>>::new("4"),
            ],
        ] {
            let cache = cache.clone();
            let load_counts = load_counts.clone();
            let mut barrier_rx = barrier_tx.subscribe();
            handles.push(tokio::spawn(async move {
                barrier_rx.recv().await.ok();
                cache
                    .get_or_insert_with_key_batch(keys.clone(), move |owned_keys| {
                        let load_counts = load_counts.clone();
                        async move {
                            sleep(Duration::from_millis(20)).await;
                            Ok(owned_keys
                                .into_iter()
                                .map(|key| {
                                    let value = key.key.parse::<i32>().unwrap();
                                    load_counts[value as usize].fetch_add(1, Ordering::SeqCst);
                                    (key, vec![value])
                                })
                                .collect::<Vec<_>>())
                        }
                    })
                    .await
            }));
        }

        barrier_tx.send(()).unwrap();
        let first = handles.remove(0).await.unwrap().unwrap();
        let second = handles.remove(0).await.unwrap().unwrap();

        assert_eq!(
            first
                .iter()
                .map(|entry| entry.value.as_ref().clone())
                .collect::<Vec<_>>(),
            vec![vec![1], vec![2], vec![3]]
        );
        assert_eq!(
            second
                .iter()
                .map(|entry| entry.value.as_ref().clone())
                .collect::<Vec<_>>(),
            vec![vec![2], vec![3], vec![4]]
        );
        let mut cached_by_value = HashMap::new();
        for entry in first.iter().chain(second.iter()) {
            cached_by_value
                .entry(entry.value[0])
                .or_insert_with(Vec::new)
                .push(entry.was_cached);
        }
        assert_eq!(cached_by_value.get(&1).unwrap(), &vec![false]);
        assert_eq!(cached_by_value.get(&4).unwrap(), &vec![false]);
        assert_eq!(
            cached_by_value
                .get(&2)
                .unwrap()
                .iter()
                .filter(|was_cached| **was_cached)
                .count(),
            1
        );
        assert_eq!(
            cached_by_value
                .get(&3)
                .unwrap()
                .iter()
                .filter(|was_cached| **was_cached)
                .count(),
            1
        );
        for key in 1..=4 {
            assert_eq!(load_counts[key].load(Ordering::SeqCst), 1);
        }
        assert_eq!(cache.stats().await.misses, 4);
        assert_eq!(cache.stats().await.hits, 2);
    }

    #[tokio::test]
    async fn test_get_or_insert_batch_waits_on_in_flight_batch_owner() {
        use std::sync::atomic::AtomicUsize;

        let cache = LanceCache::with_capacity(10000);
        let first_load_count = Arc::new(AtomicUsize::new(0));
        let second_load_count = Arc::new(AtomicUsize::new(0));
        let owner_started = Arc::new(tokio::sync::Notify::new());
        let finish_owner = Arc::new(tokio::sync::Notify::new());
        let second_loader_started = Arc::new(tokio::sync::Notify::new());

        let first_cache = cache.clone();
        let first_load_count_clone = first_load_count.clone();
        let owner_started_clone = owner_started.clone();
        let finish_owner_clone = finish_owner.clone();
        let first = tokio::spawn(async move {
            first_cache
                .get_or_insert_with_key_batch(
                    vec![
                        TestKey::<Vec<i32>>::new("1"),
                        TestKey::<Vec<i32>>::new("2"),
                        TestKey::<Vec<i32>>::new("3"),
                    ],
                    move |owned_keys| {
                        let first_load_count = first_load_count_clone.clone();
                        let owner_started = owner_started_clone.clone();
                        let finish_owner = finish_owner_clone.clone();
                        async move {
                            assert_eq!(
                                owned_keys
                                    .iter()
                                    .map(|key| key.key.as_str())
                                    .collect::<Vec<_>>(),
                                vec!["1", "2", "3"]
                            );
                            first_load_count.fetch_add(owned_keys.len(), Ordering::SeqCst);
                            owner_started.notify_waiters();
                            finish_owner.notified().await;
                            Ok(owned_keys
                                .into_iter()
                                .map(|key| {
                                    let value = key.key.parse::<i32>().unwrap();
                                    (key, vec![value])
                                })
                                .collect::<Vec<_>>())
                        }
                    },
                )
                .await
        });

        owner_started.notified().await;

        let second_cache = cache.clone();
        let second_load_count_clone = second_load_count.clone();
        let second_loader_started_clone = second_loader_started.clone();
        let second = tokio::spawn(async move {
            second_cache
                .get_or_insert_with_key_batch(
                    vec![
                        TestKey::<Vec<i32>>::new("2"),
                        TestKey::<Vec<i32>>::new("3"),
                        TestKey::<Vec<i32>>::new("4"),
                    ],
                    move |owned_keys| {
                        let second_load_count = second_load_count_clone.clone();
                        let second_loader_started = second_loader_started_clone.clone();
                        async move {
                            second_loader_started.notify_waiters();
                            assert_eq!(
                                owned_keys
                                    .iter()
                                    .map(|key| key.key.as_str())
                                    .collect::<Vec<_>>(),
                                vec!["4"]
                            );
                            second_load_count.fetch_add(owned_keys.len(), Ordering::SeqCst);
                            Ok(owned_keys
                                .into_iter()
                                .map(|key| {
                                    let value = key.key.parse::<i32>().unwrap();
                                    (key, vec![value])
                                })
                                .collect::<Vec<_>>())
                        }
                    },
                )
                .await
        });

        second_loader_started.notified().await;
        finish_owner.notify_waiters();

        let first_values = first.await.unwrap().unwrap();
        let second_values = second.await.unwrap().unwrap();

        assert_eq!(
            first_values
                .iter()
                .map(|entry| entry.value.as_ref().clone())
                .collect::<Vec<_>>(),
            vec![vec![1], vec![2], vec![3]]
        );
        assert_eq!(
            first_values
                .iter()
                .map(|entry| entry.was_cached)
                .collect::<Vec<_>>(),
            vec![false, false, false]
        );
        assert_eq!(
            second_values
                .iter()
                .map(|entry| entry.value.as_ref().clone())
                .collect::<Vec<_>>(),
            vec![vec![2], vec![3], vec![4]]
        );
        assert_eq!(
            second_values
                .iter()
                .map(|entry| entry.was_cached)
                .collect::<Vec<_>>(),
            vec![true, true, false]
        );
        assert_eq!(first_load_count.load(Ordering::SeqCst), 3);
        assert_eq!(second_load_count.load(Ordering::SeqCst), 1);
        assert_eq!(cache.stats().await.misses, 4);
        assert_eq!(cache.stats().await.hits, 2);
    }

    #[tokio::test]
    async fn test_get_or_insert_batch_dedups_high_fanout_overlap() {
        use std::sync::atomic::AtomicUsize;
        use tokio::time::{Duration, sleep};

        let cache = LanceCache::with_capacity(10000);
        let key_count = 64;
        let concurrency = 8;
        let load_counts = Arc::new(
            (0..key_count)
                .map(|_| AtomicUsize::new(0))
                .collect::<Vec<_>>(),
        );
        let (barrier_tx, _) = tokio::sync::broadcast::channel::<()>(1);
        let mut handles = Vec::new();

        for _ in 0..concurrency {
            let cache = cache.clone();
            let load_counts = load_counts.clone();
            let mut barrier_rx = barrier_tx.subscribe();
            handles.push(tokio::spawn(async move {
                let keys = (0..key_count)
                    .map(|idx| TestKey::<Vec<i32>>::new(&idx.to_string()))
                    .collect::<Vec<_>>();
                barrier_rx.recv().await.ok();
                cache
                    .get_or_insert_with_key_batch(keys, move |owned_keys| {
                        let load_counts = load_counts.clone();
                        async move {
                            sleep(Duration::from_millis(10)).await;
                            Ok(owned_keys
                                .into_iter()
                                .map(|key| {
                                    let value = key.key.parse::<i32>().unwrap();
                                    load_counts[value as usize].fetch_add(1, Ordering::SeqCst);
                                    (key, vec![value])
                                })
                                .collect::<Vec<_>>())
                        }
                    })
                    .await
            }));
        }

        barrier_tx.send(()).unwrap();
        for handle in handles {
            let values = handle.await.unwrap().unwrap();
            assert_eq!(
                values
                    .iter()
                    .map(|entry| entry.value[0])
                    .collect::<Vec<_>>(),
                (0..key_count).collect::<Vec<_>>()
            );
        }

        for key in 0..key_count {
            assert_eq!(load_counts[key as usize].load(Ordering::SeqCst), 1);
        }
    }

    #[tokio::test]
    async fn test_get_or_insert_batch_shares_flights_with_single_key_get_or_insert() {
        use std::sync::atomic::AtomicUsize;

        let cache = LanceCache::with_capacity(10000);
        let batch_load_count = Arc::new(AtomicUsize::new(0));
        let single_load_count = Arc::new(AtomicUsize::new(0));
        let loader_started = Arc::new(tokio::sync::Notify::new());
        let finish_loader = Arc::new(tokio::sync::Notify::new());

        let batch_cache = cache.clone();
        let batch_load_count_clone = batch_load_count.clone();
        let loader_started_clone = loader_started.clone();
        let finish_loader_clone = finish_loader.clone();
        let batch = tokio::spawn(async move {
            batch_cache
                .get_or_insert_with_key_batch(
                    vec![TestKey::<Vec<i32>>::new("1"), TestKey::<Vec<i32>>::new("2")],
                    move |owned_keys| {
                        let batch_load_count = batch_load_count_clone.clone();
                        let loader_started = loader_started_clone.clone();
                        let finish_loader = finish_loader_clone.clone();
                        async move {
                            batch_load_count.fetch_add(owned_keys.len(), Ordering::SeqCst);
                            loader_started.notify_waiters();
                            finish_loader.notified().await;
                            Ok(owned_keys
                                .into_iter()
                                .map(|key| {
                                    let value = key.key.parse::<i32>().unwrap();
                                    (key, vec![value])
                                })
                                .collect::<Vec<_>>())
                        }
                    },
                )
                .await
        });

        loader_started.notified().await;

        let single_cache = cache.clone();
        let single_load_count_clone = single_load_count.clone();
        let single = tokio::spawn(async move {
            single_cache
                .get_or_insert_with_key(TestKey::<Vec<i32>>::new("2"), move || {
                    let single_load_count = single_load_count_clone;
                    async move {
                        single_load_count.fetch_add(1, Ordering::SeqCst);
                        Ok(vec![20])
                    }
                })
                .await
        });

        tokio::task::yield_now().await;
        assert_eq!(single_load_count.load(Ordering::SeqCst), 0);
        finish_loader.notify_waiters();

        let batch_values = batch.await.unwrap().unwrap();
        let single_value = single.await.unwrap().unwrap();

        assert_eq!(
            batch_values
                .iter()
                .map(|entry| entry.value.as_ref().clone())
                .collect::<Vec<_>>(),
            vec![vec![1], vec![2]]
        );
        assert_eq!(
            batch_values
                .iter()
                .map(|entry| entry.was_cached)
                .collect::<Vec<_>>(),
            vec![false, false]
        );
        assert_eq!(*single_value, vec![2]);
        assert_eq!(batch_load_count.load(Ordering::SeqCst), 2);
        assert_eq!(single_load_count.load(Ordering::SeqCst), 0);
        assert_eq!(cache.stats().await.misses, 2);
        assert_eq!(cache.stats().await.hits, 1);
    }

    #[tokio::test]
    async fn test_get_or_insert_single_key_shares_flights_with_batch_get_or_insert() {
        use std::sync::atomic::AtomicUsize;

        let cache = LanceCache::with_capacity(10000);
        let single_load_count = Arc::new(AtomicUsize::new(0));
        let batch_load_count = Arc::new(AtomicUsize::new(0));
        let loader_started = Arc::new(tokio::sync::Notify::new());
        let finish_loader = Arc::new(tokio::sync::Notify::new());
        let batch_loader_started = Arc::new(tokio::sync::Notify::new());

        let single_cache = cache.clone();
        let single_load_count_clone = single_load_count.clone();
        let loader_started_clone = loader_started.clone();
        let finish_loader_clone = finish_loader.clone();
        let single = tokio::spawn(async move {
            single_cache
                .get_or_insert_with_key(TestKey::<Vec<i32>>::new("2"), move || {
                    let single_load_count = single_load_count_clone.clone();
                    let loader_started = loader_started_clone.clone();
                    let finish_loader = finish_loader_clone;
                    async move {
                        single_load_count.fetch_add(1, Ordering::SeqCst);
                        loader_started.notify_waiters();
                        finish_loader.notified().await;
                        Ok(vec![2])
                    }
                })
                .await
        });

        loader_started.notified().await;

        let batch_cache = cache.clone();
        let batch_load_count_clone = batch_load_count.clone();
        let batch_loader_started_clone = batch_loader_started.clone();
        let batch = tokio::spawn(async move {
            batch_cache
                .get_or_insert_with_key_batch(
                    vec![TestKey::<Vec<i32>>::new("1"), TestKey::<Vec<i32>>::new("2")],
                    move |owned_keys| {
                        let batch_load_count = batch_load_count_clone.clone();
                        let batch_loader_started = batch_loader_started_clone.clone();
                        async move {
                            batch_loader_started.notify_waiters();
                            assert_eq!(
                                owned_keys
                                    .iter()
                                    .map(|key| key.key.as_str())
                                    .collect::<Vec<_>>(),
                                vec!["1"]
                            );
                            batch_load_count.fetch_add(owned_keys.len(), Ordering::SeqCst);
                            Ok(owned_keys
                                .into_iter()
                                .map(|key| {
                                    let value = key.key.parse::<i32>().unwrap();
                                    (key, vec![value])
                                })
                                .collect())
                        }
                    },
                )
                .await
        });

        batch_loader_started.notified().await;
        finish_loader.notify_waiters();

        let single_value = single.await.unwrap().unwrap();
        let batch_values = batch.await.unwrap().unwrap();

        assert_eq!(*single_value, vec![2]);
        assert_eq!(
            batch_values
                .iter()
                .map(|entry| entry.value.as_ref().clone())
                .collect::<Vec<_>>(),
            vec![vec![1], vec![2]]
        );
        assert_eq!(
            batch_values
                .iter()
                .map(|entry| entry.was_cached)
                .collect::<Vec<_>>(),
            vec![false, true]
        );
        assert_eq!(single_load_count.load(Ordering::SeqCst), 1);
        assert_eq!(batch_load_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_get_or_insert_batch_loader_receives_owned_keys_in_input_order() {
        let cache = LanceCache::with_capacity(10000);
        cache
            .insert_with_key(&TestKey::new("key-03"), Arc::new(vec![3]))
            .await;
        cache
            .insert_with_key(&TestKey::new("key-07"), Arc::new(vec![7]))
            .await;

        let keys = (0..16)
            .map(|idx| TestKey::<Vec<i32>>::new(&format!("key-{idx:02}")))
            .collect::<Vec<_>>();
        let expected_owned_keys = keys
            .iter()
            .filter(|key| key.key != "key-03" && key.key != "key-07")
            .map(|key| key.key.clone())
            .collect::<Vec<_>>();
        let observed_owned_keys = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed_owned_keys_clone = observed_owned_keys.clone();

        let values = cache
            .get_or_insert_with_key_batch(keys, move |owned_keys| {
                let observed_owned_keys = observed_owned_keys_clone.clone();
                async move {
                    *observed_owned_keys.lock().unwrap() =
                        owned_keys.iter().map(|key| key.key.clone()).collect();
                    Ok(owned_keys
                        .into_iter()
                        .map(|key| {
                            let value = key.key.strip_prefix("key-").unwrap().parse().unwrap();
                            (key, vec![value])
                        })
                        .collect())
                }
            })
            .await
            .unwrap();

        assert_eq!(*observed_owned_keys.lock().unwrap(), expected_owned_keys);
        assert_eq!(
            values
                .iter()
                .map(|entry| entry.value[0])
                .collect::<Vec<_>>(),
            (0..16).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn test_get_or_insert_batch_rejects_duplicate_input_keys() {
        let cache = LanceCache::with_capacity(10000);

        let err = cache
            .get_or_insert_with_key_batch(
                vec![TestKey::<Vec<i32>>::new("1"), TestKey::<Vec<i32>>::new("1")],
                |_| async { Ok(Vec::<(TestKey<Vec<i32>>, Vec<i32>)>::new()) },
            )
            .await
            .unwrap_err();

        assert_invalid_input_contains(err, &["duplicate cache key", "key='1'"]);
        assert_eq!(cache.stats().await.hits, 0);
        assert_eq!(cache.stats().await.misses, 0);
    }

    #[rstest]
    #[case::unexpected_key(
        vec![(TestKey::<Vec<i32>>::new("2"), vec![2])],
        &["unexpected key", "key='2'"]
    )]
    #[case::missing_key(
        Vec::<(TestKey<Vec<i32>>, Vec<i32>)>::new(),
        &["did not return expected key", "key='1'"]
    )]
    #[case::extra_key(vec![
        (TestKey::<Vec<i32>>::new("1"), vec![1]),
        (TestKey::<Vec<i32>>::new("2"), vec![2]),
    ], &["unexpected key", "key='2'"])]
    #[case::duplicate_key(vec![
        (TestKey::<Vec<i32>>::new("1"), vec![1]),
        (TestKey::<Vec<i32>>::new("1"), vec![10]),
    ], &["duplicate keys", "key='1'"])]
    #[tokio::test]
    async fn test_get_or_insert_batch_rejects_loader_key_validation(
        #[case] loaded: Vec<(TestKey<Vec<i32>>, Vec<i32>)>,
        #[case] expected_message: &[&str],
    ) {
        let cache = LanceCache::with_capacity(10000);

        let err = cache
            .get_or_insert_with_key_batch(vec![TestKey::<Vec<i32>>::new("1")], move |_| {
                let loaded = loaded.clone();
                async move { Ok(loaded) }
            })
            .await
            .unwrap_err();

        assert_invalid_input_contains(err, expected_message);
        assert!(
            cache
                .get_with_key(&TestKey::<Vec<i32>>::new("1"))
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_get_or_insert_single_key_waiter_retries_after_owner_error() {
        use std::sync::atomic::AtomicUsize;
        use tokio::time::{Duration, timeout};

        let cache = LanceCache::with_capacity(10000);
        let owner_started = Arc::new(tokio::sync::Notify::new());
        let fail_owner = Arc::new(tokio::sync::Notify::new());
        let retry_load_count = Arc::new(AtomicUsize::new(0));

        let owner_cache = cache.clone();
        let owner_started_clone = owner_started.clone();
        let fail_owner_clone = fail_owner.clone();
        let owner = tokio::spawn(async move {
            owner_cache
                .get_or_insert_with_key(TestKey::<Vec<i32>>::new("1"), move || async move {
                    owner_started_clone.notify_waiters();
                    fail_owner_clone.notified().await;
                    Err(Error::io("owner load failed"))
                })
                .await
        });

        owner_started.notified().await;

        let waiter_cache = cache.clone();
        let retry_load_count_clone = retry_load_count.clone();
        let waiter = tokio::spawn(async move {
            waiter_cache
                .get_or_insert_with_key(TestKey::<Vec<i32>>::new("1"), move || async move {
                    retry_load_count_clone.fetch_add(1, Ordering::SeqCst);
                    Ok(vec![1])
                })
                .await
        });

        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());
        fail_owner.notify_waiters();

        let owner_err = owner.await.unwrap().unwrap_err();
        assert!(matches!(owner_err, Error::IO { .. }));

        let waiter_value = timeout(Duration::from_secs(5), waiter)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(*waiter_value, vec![1]);
        assert_eq!(retry_load_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_get_or_insert_single_key_waiter_retries_after_owner_cancel() {
        use std::sync::atomic::AtomicUsize;
        use tokio::time::{Duration, sleep, timeout};

        let cache = LanceCache::with_capacity(10000);
        let owner_started = Arc::new(tokio::sync::Notify::new());
        let retry_load_count = Arc::new(AtomicUsize::new(0));

        let owner_cache = cache.clone();
        let owner_started_clone = owner_started.clone();
        let owner = tokio::spawn(async move {
            owner_cache
                .get_or_insert_with_key(TestKey::<Vec<i32>>::new("1"), move || async move {
                    owner_started_clone.notify_waiters();
                    std::future::pending::<Result<Vec<i32>>>().await
                })
                .await
        });

        owner_started.notified().await;

        let waiter_cache = cache.clone();
        let retry_load_count_clone = retry_load_count.clone();
        let waiter = tokio::spawn(async move {
            waiter_cache
                .get_or_insert_with_key(TestKey::<Vec<i32>>::new("1"), move || async move {
                    retry_load_count_clone.fetch_add(1, Ordering::SeqCst);
                    Ok(vec![1])
                })
                .await
        });

        sleep(Duration::from_millis(10)).await;
        assert!(!waiter.is_finished());
        owner.abort();

        let waiter_value = timeout(Duration::from_secs(5), waiter)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(*waiter_value, vec![1]);
        assert_eq!(retry_load_count.load(Ordering::SeqCst), 1);
        assert!(owner.await.unwrap_err().is_cancelled());
    }

    #[tokio::test]
    async fn test_get_or_insert_batch_waiter_retries_after_owner_error() {
        use std::sync::atomic::AtomicUsize;
        use tokio::time::{Duration, timeout};

        let cache = LanceCache::with_capacity(10000);
        let owner_started = Arc::new(tokio::sync::Notify::new());
        let fail_owner = Arc::new(tokio::sync::Notify::new());
        let retry_load_count = Arc::new(AtomicUsize::new(0));

        let owner_cache = cache.clone();
        let owner_started_clone = owner_started.clone();
        let fail_owner_clone = fail_owner.clone();
        let owner = tokio::spawn(async move {
            owner_cache
                .get_or_insert_with_key_batch(vec![TestKey::<Vec<i32>>::new("1")], move |_| {
                    let owner_started = owner_started_clone.clone();
                    let fail_owner = fail_owner_clone.clone();
                    async move {
                        owner_started.notify_waiters();
                        fail_owner.notified().await;
                        Err(Error::io("owner load failed"))
                    }
                })
                .await
        });

        owner_started.notified().await;

        let waiter_cache = cache.clone();
        let retry_load_count_clone = retry_load_count.clone();
        let waiter = tokio::spawn(async move {
            waiter_cache
                .get_or_insert_with_key_batch(vec![TestKey::<Vec<i32>>::new("1")], move |keys| {
                    let retry_load_count = retry_load_count_clone.clone();
                    async move {
                        retry_load_count.fetch_add(keys.len(), Ordering::SeqCst);
                        Ok(keys.into_iter().map(|key| (key, vec![1])).collect())
                    }
                })
                .await
        });

        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());
        fail_owner.notify_waiters();

        let owner_err = owner.await.unwrap().unwrap_err();
        assert!(matches!(owner_err, Error::IO { .. }));

        let waiter_values = timeout(Duration::from_secs(5), waiter)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(*waiter_values[0].value, vec![1]);
        assert_eq!(retry_load_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_get_or_insert_batch_waiter_retries_after_owner_cancel() {
        use std::sync::atomic::AtomicUsize;
        use tokio::time::{Duration, sleep, timeout};

        let cache = LanceCache::with_capacity(10000);
        let owner_started = Arc::new(tokio::sync::Notify::new());
        let retry_load_count = Arc::new(AtomicUsize::new(0));

        let owner_cache = cache.clone();
        let owner_started_clone = owner_started.clone();
        let owner = tokio::spawn(async move {
            owner_cache
                .get_or_insert_with_key_batch(vec![TestKey::<Vec<i32>>::new("1")], move |_| {
                    let owner_started = owner_started_clone.clone();
                    async move {
                        owner_started.notify_waiters();
                        std::future::pending::<Result<Vec<(TestKey<Vec<i32>>, Vec<i32>)>>>().await
                    }
                })
                .await
        });

        owner_started.notified().await;

        let waiter_cache = cache.clone();
        let retry_load_count_clone = retry_load_count.clone();
        let waiter = tokio::spawn(async move {
            waiter_cache
                .get_or_insert_with_key_batch(vec![TestKey::<Vec<i32>>::new("1")], move |keys| {
                    let retry_load_count = retry_load_count_clone.clone();
                    async move {
                        retry_load_count.fetch_add(keys.len(), Ordering::SeqCst);
                        Ok(keys.into_iter().map(|key| (key, vec![1])).collect())
                    }
                })
                .await
        });

        sleep(Duration::from_millis(10)).await;
        assert!(!waiter.is_finished());
        owner.abort();

        let waiter_values = timeout(Duration::from_secs(5), waiter)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(*waiter_values[0].value, vec![1]);
        assert_eq!(retry_load_count.load(Ordering::SeqCst), 1);
        assert!(owner.await.unwrap_err().is_cancelled());
    }
}
