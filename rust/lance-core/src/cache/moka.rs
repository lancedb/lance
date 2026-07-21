// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use futures::Future;

use crate::Result;
use crate::error::CloneableError;

use super::backend::{CacheBackend, CacheEntry};
use super::{CacheCodec, InternalCacheKey};

/// Internal record stored in the moka cache.
#[derive(Clone, Debug)]
struct MokaCacheEntry {
    entry: CacheEntry,
    size_bytes: usize,
}

/// Per-entry key cost for eviction.
fn key_footprint(_key: &InternalCacheKey) -> usize {
    std::mem::size_of::<InternalCacheKey>()
}

fn entry_weight(key: &InternalCacheKey, size_bytes: usize) -> u32 {
    key_footprint(key)
        .saturating_add(size_bytes)
        .try_into()
        .unwrap_or(u32::MAX)
}

/// Default [`CacheBackend`] backed by a [moka](https://crates.io/crates/moka) cache.
///
/// Provides weighted-capacity eviction and concurrent-load deduplication
/// via moka's built-in `optionally_get_with`.
pub struct MokaCacheBackend {
    cache: moka::future::Cache<InternalCacheKey, MokaCacheEntry>,
}

impl std::fmt::Debug for MokaCacheBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MokaCacheBackend")
            .field("entry_count", &self.cache.entry_count())
            .finish()
    }
}

impl MokaCacheBackend {
    pub fn with_capacity(capacity: usize) -> Self {
        let cache = moka::future::Cache::builder()
            .max_capacity(capacity as u64)
            .weigher(|key: &InternalCacheKey, entry: &MokaCacheEntry| {
                entry_weight(key, entry.size_bytes)
            })
            .build();
        Self { cache }
    }

    pub fn no_cache() -> Self {
        Self {
            cache: moka::future::Cache::new(0),
        }
    }
}

#[async_trait]
impl CacheBackend for MokaCacheBackend {
    async fn get(&self, key: &InternalCacheKey, _codec: Option<CacheCodec>) -> Option<CacheEntry> {
        self.cache.get(key).await.map(|r| r.entry)
    }

    async fn insert(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size_bytes: usize,
        _codec: Option<CacheCodec>,
    ) {
        self.cache
            .insert(*key, MokaCacheEntry { entry, size_bytes })
            .await;
    }

    async fn get_or_insert<'a>(
        &self,
        key: &InternalCacheKey,
        loader: Pin<Box<dyn Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>>,
        _codec: Option<CacheCodec>,
    ) -> Result<(CacheEntry, bool)> {
        // Track whether the loader actually ran (= cache miss).
        let was_miss = Arc::new(AtomicBool::new(false));
        let was_miss_clone = was_miss.clone();

        let init = async move {
            was_miss_clone.store(true, Ordering::Relaxed);
            loader
                .await
                .map(|(entry, size_bytes)| MokaCacheEntry { entry, size_bytes })
                .map_err(CloneableError)
        };

        let owned_key = *key;
        match self.cache.try_get_with(owned_key, init).await {
            Ok(record) => {
                let was_cached = !was_miss.load(Ordering::Relaxed);
                Ok((record.entry, was_cached))
            }
            Err(error) => Err(Arc::unwrap_or_clone(error).0),
        }
    }

    async fn clear(&self) {
        self.cache.invalidate_all();
        self.cache.run_pending_tasks().await;
    }

    async fn num_entries(&self) -> usize {
        self.cache.run_pending_tasks().await;
        self.cache.entry_count() as usize
    }

    async fn size_bytes(&self) -> usize {
        self.cache.run_pending_tasks().await;
        self.cache.weighted_size() as usize
    }

    fn approx_num_entries(&self) -> usize {
        self.cache.entry_count() as usize
    }

    fn approx_size_bytes(&self) -> usize {
        // Iterate rather than using `weighted_size()` because moka's
        // weighted_size can be stale without `run_pending_tasks()`, which
        // is async and can't be called from this synchronous context.
        self.cache
            .iter()
            .map(|(key, entry)| key_footprint(key.as_ref()) + entry.size_bytes)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_weights_are_exact_and_saturate_at_moka_limit() {
        let key = InternalCacheKey::from_bytes([0; 16]);
        assert_eq!(entry_weight(&key, 7), 23);
        assert_eq!(entry_weight(&key, usize::MAX), u32::MAX);
    }
}
