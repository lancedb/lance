// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Backend interface for cache implementors.
//!
//! This module defines the trait that custom cache backends must implement,
//! along with the key and entry types they operate on. Most callers should
//! use [`LanceCache`](super::LanceCache) instead of interacting with
//! backends directly.

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Future;

use crate::Result;

use super::CacheCodec;

/// A type-erased cache entry.
pub type CacheEntry = Arc<dyn Any + Send + Sync>;

/// Iterator over cache keys currently known to a backend.
pub type CacheKeyIterator<'a> = Box<dyn Iterator<Item = InternalCacheKey> + Send + 'a>;

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
    /// Implementations should deduplicate concurrent loads for the same key
    /// so the loader runs at most once.
    ///
    /// Returns `(entry, was_cached)` where `was_cached` is `true` if the entry
    /// was already present in the cache (the loader was not invoked).
    ///
    /// See [`get`](Self::get) for codec semantics.
    async fn get_or_insert<'a>(
        &self,
        key: &InternalCacheKey,
        loader: Pin<Box<dyn Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>>,
        codec: Option<CacheCodec>,
    ) -> Result<(CacheEntry, bool)>;

    /// Remove all entries whose prefix starts with the given string.
    async fn invalidate_prefix(&self, prefix: &str);

    /// Remove all entries within `prefix` whose `key` equals `key_prefix` or
    /// starts with `key_prefix` followed by `/`.
    ///
    /// Unlike [`invalidate_prefix`](Self::invalidate_prefix), which matches
    /// against an entry's namespace [`prefix`](InternalCacheKey::prefix)
    /// (e.g. a dataset URI), this matches against the per-entry
    /// [`key`](InternalCacheKey::key) (e.g. a version number or fragment id).
    /// This lets callers evict every cached entry associated with one
    /// version/fragment (a manifest, its transaction, its row address mask,
    /// etc.) without needing to know every optional suffix (e-tag, filter
    /// hash) a given key type may carry.
    ///
    /// The `/` boundary check prevents `key_prefix = "5"` from also matching
    /// `"50"` or `"500"`.
    ///
    /// Backends that cannot support this cheaply may leave it as a no-op;
    /// doing so only means those entries are evicted later by the normal
    /// capacity-based eviction policy instead of immediately.
    ///
    /// Defaults to a single-element call to
    /// [`invalidate_key_prefixes`](Self::invalidate_key_prefixes). Overriding
    /// [`invalidate_key_prefixes`](Self::invalidate_key_prefixes) is
    /// therefore the one to implement: it gives you both methods for free.
    /// Overriding only this method does *not* work the other way around --
    /// [`invalidate_key_prefixes`](Self::invalidate_key_prefixes)'s own
    /// default (a no-op) is unaffected, so any caller that invokes it
    /// directly (as [`LanceCache::invalidate_key_prefixes`](super::LanceCache::invalidate_key_prefixes)
    /// does) would silently evict nothing.
    async fn invalidate_key_prefix(&self, prefix: &str, key_prefix: &str) {
        self.invalidate_key_prefixes(prefix, std::slice::from_ref(&key_prefix.to_owned()))
            .await;
    }

    /// Like [`invalidate_key_prefix`](Self::invalidate_key_prefix), but
    /// removes entries matching *any* of `key_prefixes` in a single pass
    /// over the cache, instead of one pass per prefix.
    ///
    /// Prefer this over calling [`invalidate_key_prefix`](Self::invalidate_key_prefix)
    /// in a loop when invalidating many keys at once (e.g. several dataset
    /// versions' worth of cache entries after a cleanup run) — each call to
    /// either method registers its own scan over the backend's entries, so
    /// batching avoids `O(prefixes)` scans.
    ///
    /// Backends that cannot support this cheaply may leave it as a no-op;
    /// doing so only means those entries are evicted later by the normal
    /// capacity-based eviction policy instead of immediately.
    async fn invalidate_key_prefixes(&self, _prefix: &str, _key_prefixes: &[String]) {}

    /// Remove all entries.
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
