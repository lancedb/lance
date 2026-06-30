// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Backend interface for cache implementors.
//!
//! This module defines the trait that custom cache backends must implement,
//! along with the key and entry types they operate on. Most callers should
//! use [`LanceCache`](super::LanceCache) instead of interacting with
//! backends directly.

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::{Future, future::BoxFuture};

use crate::{Error, Result};

use super::CacheCodec;

/// A type-erased cache entry.
pub type CacheEntry = Arc<dyn Any + Send + Sync>;

/// Iterator over cache keys currently known to a backend.
pub type CacheKeyIterator<'a> = Box<dyn Iterator<Item = InternalCacheKey> + Send + 'a>;

/// A cache entry loaded by a batch loader.
pub struct CacheLoadedEntry {
    /// Cache key for this loaded entry.
    pub key: InternalCacheKey,
    /// Loaded value.
    pub entry: CacheEntry,
    /// Entry weight in bytes for backend eviction accounting.
    pub size_bytes: usize,
}

/// A cache entry returned from a batch lookup.
pub struct CacheBatchEntry {
    /// Cache key for this entry.
    pub key: InternalCacheKey,
    /// Cached or loaded value.
    pub entry: CacheEntry,
    /// True when this call did not run the loader for this key.
    ///
    /// This includes ordinary cache hits and values loaded by another
    /// in-flight owner.
    pub was_cached: bool,
}

/// Loader used by [`CacheBackend::get_or_insert_many`].
///
/// Backends call this with the subset of missing keys owned by the current
/// call. The loader must return exactly one [`CacheLoadedEntry`] for each key
/// it receives and no other keys. A backend may call the loader more than
/// once during one batch request if keys need to be retried after another
/// in-flight owner fails or is dropped.
pub type CacheBatchLoader<'a> = Arc<
    dyn Fn(Vec<InternalCacheKey>) -> BoxFuture<'a, Result<Vec<CacheLoadedEntry>>>
        + Send
        + Sync
        + 'a,
>;

/// Structured cache key passed to [`CacheBackend`] methods.
///
/// CacheBackend impls receive these ready-made from [`LanceCache`](super::LanceCache)
/// — you do not construct them yourself. Composed of three parts:
/// - **prefix**: scopes the key to a dataset or index (e.g. `"s3://bucket/dataset/"`)
/// - **key**: identifies the specific entry (e.g. `"42"` for a version number)
/// - **type_name**: distinguishes different value types stored under the same
///   user key (e.g. `"Vec<IndexMetadata>"`)
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct InternalCacheKey {
    prefix: Arc<str>,
    key: Arc<str>,
    type_name: &'static str,
}

impl InternalCacheKey {
    pub fn new(prefix: Arc<str>, key: Arc<str>, type_name: &'static str) -> Self {
        Self {
            prefix,
            key,
            type_name,
        }
    }

    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// Returns true if this key's prefix starts with the given string.
    pub fn starts_with(&self, prefix: &str) -> bool {
        self.prefix.starts_with(prefix)
    }
}

/// Low-level pluggable cache backend.
///
/// Implementations store entries keyed by [`InternalCacheKey`] and return
/// type-erased [`CacheEntry`] values.
/// [`LanceCache`](super::LanceCache) handles key construction and type safety;
/// backend authors only need to implement storage and eviction.
#[async_trait]
pub trait CacheBackend: Send + Sync + std::fmt::Debug {
    /// Look up an entry by its key.
    ///
    /// `codec` is provided so that persistent backends can deserialize the
    /// entry from storage. In-memory backends can ignore it. When `codec`
    /// is `None`, the entry type does not support serialization yet and
    /// must be stored in-memory.
    ///
    /// The goal is for all cache entry types to eventually have codecs,
    /// at which point the `Option` will be removed.
    async fn get(&self, key: &InternalCacheKey, codec: Option<CacheCodec>) -> Option<CacheEntry>;

    /// Store an entry. `size_bytes` is used for eviction accounting.
    ///
    /// See [`get`](Self::get) for codec semantics.
    async fn insert(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size_bytes: usize,
        codec: Option<CacheCodec>,
    );

    /// Get an existing entry or compute it from `loader`.
    ///
    /// Implementations should deduplicate concurrent successful loads for the
    /// same key. If an in-flight owner fails or is dropped, a waiting caller
    /// may retry and invoke its own loader for that key.
    ///
    /// Returns `(entry, was_cached)` where `was_cached` is `true` if the entry
    /// was already present in the cache or was loaded by another in-flight
    /// owner, so this call did not invoke its loader.
    ///
    /// See [`get`](Self::get) for codec semantics.
    async fn get_or_insert<'a>(
        &self,
        key: &InternalCacheKey,
        loader: Pin<Box<dyn Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>>,
        codec: Option<CacheCodec>,
    ) -> Result<(CacheEntry, bool)>;

    /// Get existing entries or compute missing entries from `loader`.
    ///
    /// Input keys must be unique. The returned entries must follow the same
    /// order as `keys`. Implementations should deduplicate concurrent loads
    /// per key. The loader receives only keys this call owns, preserving their
    /// input order, and must return exactly one entry for each owned key.
    ///
    /// This API is intended for callers whose loader can load missing keys more
    /// efficiently as a batch. It is not a general replacement for
    /// [`get_or_insert`](Self::get_or_insert): for isolated keys, cheap loaders,
    /// or disjoint concurrent batches, the per-key coordination and result maps
    /// can make it slower than simpler cache access patterns. Use it when
    /// preserving batched loading is worth that overhead.
    ///
    /// The default implementation is a compatibility fallback: it calls
    /// [`get_or_insert`](Self::get_or_insert) one key at a time. It preserves
    /// the backend's single-key get-or-insert semantics, but it does not
    /// preserve coalesced batch I/O. Backends that can preserve true batch
    /// loading should override this method. The fallback is not atomic: if
    /// loading a later key fails, earlier keys may already have been inserted
    /// into the cache.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance_core::Result;
    /// # use lance_core::cache::{
    /// #     CacheBackend, CacheBatchEntry, CacheBatchLoader, CacheEntry,
    /// #     CacheLoadedEntry, InternalCacheKey,
    /// # };
    /// # async fn example(
    /// #     backend: &dyn CacheBackend,
    /// #     keys: Vec<InternalCacheKey>,
    /// # ) -> Result<Vec<CacheBatchEntry>> {
    /// let loader: CacheBatchLoader<'_> = Arc::new(|owned_keys| {
    ///     Box::pin(async move {
    ///         Ok(owned_keys
    ///             .into_iter()
    ///             .map(|key| CacheLoadedEntry {
    ///                 key,
    ///                 entry: Arc::new(42_usize) as CacheEntry,
    ///                 size_bytes: std::mem::size_of::<usize>(),
    ///             })
    ///             .collect())
    ///     })
    /// });
    ///
    /// let entries = backend.get_or_insert_many(keys, loader, None).await?;
    /// # Ok(entries)
    /// # }
    /// ```
    async fn get_or_insert_many<'a>(
        &self,
        keys: Vec<InternalCacheKey>,
        loader: CacheBatchLoader<'a>,
        codec: Option<CacheCodec>,
    ) -> Result<Vec<CacheBatchEntry>> {
        validate_unique_keys(&keys)?;

        let mut entries = Vec::with_capacity(keys.len());
        for key in keys {
            let loader = loader.clone();
            let loader_key = key.clone();
            let single_loader = Box::pin(async move {
                let loaded = loader(vec![loader_key.clone()]).await?;
                let mut loaded =
                    validate_loaded_entries(std::slice::from_ref(&loader_key), loaded)?;
                let loaded = loaded.remove(&loader_key).ok_or_else(|| {
                    Error::internal("validated batch loader result missing single fallback key")
                })?;
                Ok((loaded.entry, loaded.size_bytes))
            });
            let (entry, was_cached) = self.get_or_insert(&key, single_loader, codec).await?;
            entries.push(CacheBatchEntry {
                key,
                entry,
                was_cached,
            });
        }

        Ok(entries)
    }

    /// Remove all entries whose prefix starts with the given string.
    ///
    /// This only invalidates entries currently stored in the backend.
    /// Implementations are not required to cancel in-flight loaders, and an
    /// in-flight `get_or_insert` may insert a matching entry after this call
    /// returns.
    async fn invalidate_prefix(&self, prefix: &str);

    /// Remove all entries.
    ///
    /// This only invalidates entries currently stored in the backend.
    /// Implementations are not required to cancel in-flight loaders, and an
    /// in-flight `get_or_insert` may insert an entry after this call returns.
    async fn clear(&self);

    /// Return an iterator over cache keys currently known to this backend.
    ///
    /// Backends that cannot enumerate keys cheaply or accurately should return
    /// `None`. An empty iterator means key inventory is supported and the
    /// cache currently has no entries.
    async fn keys(&self) -> Option<CacheKeyIterator<'_>> {
        None
    }

    /// Number of entries currently stored (may flush pending operations).
    async fn num_entries(&self) -> usize;

    /// Total weighted size in bytes of all stored entries (may flush pending operations).
    async fn size_bytes(&self) -> usize;

    /// Approximate number of entries, callable from synchronous contexts.
    /// Backends that cannot provide this cheaply should return 0.
    fn approx_num_entries(&self) -> usize {
        0
    }

    /// Approximate weighted size in bytes, callable from synchronous contexts.
    /// Used by `DeepSizeOf` to report cache memory usage.
    /// Backends that cannot provide this cheaply should return 0.
    ///
    /// Assumes entries do not share underlying buffers; if they do, the
    /// returned total may overcount.
    fn approx_size_bytes(&self) -> usize {
        0
    }
}

pub(crate) fn validate_unique_keys(keys: &[InternalCacheKey]) -> Result<()> {
    let mut seen = HashSet::with_capacity(keys.len());
    for key in keys {
        if !seen.insert(key) {
            return Err(Error::invalid_input(format!(
                "duplicate cache key in get_or_insert_many: prefix='{}', key='{}', type='{}'",
                key.prefix(),
                key.key(),
                key.type_name()
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_loaded_entries(
    expected: &[InternalCacheKey],
    loaded: Vec<CacheLoadedEntry>,
) -> Result<HashMap<InternalCacheKey, CacheLoadedEntry>> {
    let expected_keys = expected.iter().cloned().collect::<HashSet<_>>();
    let mut loaded_by_key = HashMap::with_capacity(loaded.len());

    for entry in loaded {
        if !expected_keys.contains(&entry.key) {
            return Err(Error::invalid_input(format!(
                "batch cache loader returned unexpected key: prefix='{}', key='{}', type='{}'",
                entry.key.prefix(),
                entry.key.key(),
                entry.key.type_name()
            )));
        }
        let key = entry.key.clone();
        if loaded_by_key.insert(key.clone(), entry).is_some() {
            return Err(Error::invalid_input(format!(
                "batch cache loader returned duplicate keys: prefix='{}', key='{}', type='{}'",
                key.prefix(),
                key.key(),
                key.type_name()
            )));
        }
    }

    for key in expected {
        if !loaded_by_key.contains_key(key) {
            return Err(Error::invalid_input(format!(
                "batch cache loader did not return expected key: prefix='{}', key='{}', type='{}'",
                key.prefix(),
                key.key(),
                key.type_name()
            )));
        }
    }

    Ok(loaded_by_key)
}
