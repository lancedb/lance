// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Experimental [`CacheBackend`] backed by [quick_cache](https://crates.io/crates/quick_cache).
//!
//! quick_cache records a hit by setting one atomic bit (CLOCK-style), with no
//! read-op channel or inline housekeeping, so warm hits stay cheap at high
//! read rates. Prototype for A/B against the moka backend; enable with
//! `LANCE_CACHE_BACKEND=quick`.

use std::pin::Pin;

use async_trait::async_trait;
use futures::Future;

use super::CacheCodec;
use super::backend::{CacheBackend, CacheEntry, InternalCacheKey};
use super::moka::key_footprint;
use crate::Result;

#[derive(Clone)]
struct QuickEntry {
    entry: CacheEntry,
    size_bytes: usize,
}

#[derive(Clone)]
struct EntryWeighter;

impl quick_cache::Weighter<InternalCacheKey, QuickEntry> for EntryWeighter {
    fn weight(&self, key: &InternalCacheKey, value: &QuickEntry) -> u64 {
        // Same accounting as the moka backend: key footprint + entry bytes.
        (key_footprint(key) + value.size_bytes).max(1) as u64
    }
}

pub struct QuickCacheBackend {
    cache: quick_cache::sync::Cache<InternalCacheKey, QuickEntry, EntryWeighter>,
}

impl std::fmt::Debug for QuickCacheBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuickCacheBackend")
            .field("entry_count", &self.cache.len())
            .finish()
    }
}

/// Minimum weight budget per internal shard. quick_cache splits its weight
/// capacity evenly across shards with no cross-shard borrowing, and an entry
/// heavier than ~its shard's budget is silently refused admission — so the
/// share must stay well above the largest cache entry. Large caches keep
/// many shards for read concurrency; small caches trade shards for
/// admissible entry size.
const TARGET_SHARD_SHARE: usize = 4 << 30;

impl QuickCacheBackend {
    pub fn with_capacity(capacity: usize) -> Self {
        // Round down to a power of two: quick_cache rounds the shard count
        // UP to one, which would silently halve the per-shard share.
        let shards = (capacity / TARGET_SHARD_SHARE)
            .next_power_of_two()
            .clamp(1, 1024);
        let shards = if shards * TARGET_SHARD_SHARE > capacity.max(TARGET_SHARD_SHARE) {
            (shards / 2).max(1)
        } else {
            shards
        };
        // Generous item estimate: pre-sizes the hash tables and keeps
        // quick_cache's own "at least 32 items per shard" heuristic from
        // shrinking the shard count below the explicit choice.
        let estimated_items = 1_000_000.max(shards * 32);
        let options = quick_cache::OptionsBuilder::new()
            .estimated_items_capacity(estimated_items)
            .weight_capacity(capacity as u64)
            .shards(shards)
            .build()
            .expect("quick_cache options");
        let cache = quick_cache::sync::Cache::with_options(
            options,
            EntryWeighter,
            Default::default(),
            Default::default(),
        );
        Self { cache }
    }
}

#[async_trait]
impl CacheBackend for QuickCacheBackend {
    async fn get(&self, key: &InternalCacheKey, _codec: Option<CacheCodec>) -> Option<CacheEntry> {
        self.cache.get(key).map(|v| v.entry)
    }

    async fn insert(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size_bytes: usize,
        _codec: Option<CacheCodec>,
    ) {
        self.cache
            .insert(key.clone(), QuickEntry { entry, size_bytes });
    }

    async fn get_or_insert<'a>(
        &self,
        key: &InternalCacheKey,
        loader: Pin<Box<dyn Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>>,
        _codec: Option<CacheCodec>,
    ) -> Result<(CacheEntry, bool)> {
        match self.cache.get_value_or_guard_async(key).await {
            Ok(value) => Ok((value.entry, true)),
            Err(guard) => {
                let (entry, size_bytes) = loader.await?;
                let _ = guard.insert(QuickEntry {
                    entry: entry.clone(),
                    size_bytes,
                });
                Ok((entry, false))
            }
        }
    }

    async fn invalidate_prefix(&self, prefix: &str) {
        let matching: Vec<InternalCacheKey> = self
            .cache
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(k, _)| k)
            .collect();
        for key in matching {
            self.cache.remove(&key);
        }
    }

    async fn clear(&self) {
        self.cache.clear();
    }

    async fn num_entries(&self) -> usize {
        self.cache.len()
    }

    async fn size_bytes(&self) -> usize {
        self.cache.weight() as usize
    }

    fn approx_num_entries(&self) -> usize {
        self.cache.len()
    }

    fn approx_size_bytes(&self) -> usize {
        self.cache.weight() as usize
    }
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::cache::{CacheKey, LanceCache};

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

    #[tokio::test]
    async fn test_quick_backend_roundtrip_singleflight_and_eviction() {
        // Capacity must be large relative to one entry: quick_cache shards
        // its weight budget, and an entry heavier than its shard's share is
        // not admitted at all.
        const CAPACITY: usize = 1 << 20;
        let item = Arc::new(vec![1u8, 2, 3]);
        let cache = LanceCache::with_backend(Arc::new(QuickCacheBackend::with_capacity(CAPACITY)));

        // insert + get roundtrip and weighted accounting
        cache
            .insert_with_key(&TestKey::<Vec<u8>>::new("a"), item.clone())
            .await;
        assert_eq!(
            cache
                .get_with_key(&TestKey::<Vec<u8>>::new("a"))
                .await
                .as_deref(),
            Some(&vec![1u8, 2, 3])
        );
        assert_eq!(cache.approx_size(), 1);
        assert!(cache.size_bytes().await > 0);

        // get_or_insert runs the loader only on a miss
        let loads = Arc::new(AtomicUsize::new(0));
        for _ in 0..2 {
            let loads = loads.clone();
            let value = cache
                .get_or_insert_with_key(TestKey::<Vec<u8>>::new("b"), || async move {
                    loads.fetch_add(1, Ordering::SeqCst);
                    Ok(vec![7u8])
                })
                .await
                .unwrap();
            assert_eq!(value.as_ref(), &vec![7u8]);
        }
        assert_eq!(loads.load(Ordering::SeqCst), 1);

        // capacity is enforced: overfill with 4x capacity of 16KiB entries
        // and confirm eviction kept the weighted size within budget
        for i in 0..256 {
            cache
                .insert_with_key(
                    &TestKey::<Vec<u8>>::new(&format!("fill-{i}")),
                    Arc::new(vec![0u8; 16 << 10]),
                )
                .await;
        }
        assert!(cache.size_bytes().await <= CAPACITY);
        assert!(cache.size().await < 258);

        cache.clear().await;
        assert_eq!(cache.size().await, 0);
    }
}
