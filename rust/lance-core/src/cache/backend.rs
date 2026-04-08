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

    /// Remove all entries.
    async fn clear(&self);

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
