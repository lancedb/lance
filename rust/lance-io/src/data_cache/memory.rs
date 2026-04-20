// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU32, AtomicU64, Ordering},
};
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use futures::future::BoxFuture;
use object_store::path::Path;
use tokio::sync::watch;

use lance_core::Result;

use super::{DataCache, DataCacheKey, file_ids::FileIds};

pub const DEFAULT_NUM_SHARDS: usize = 16;

const NUM_EVICTION_SAMPLES: usize = 10;

const EVICTION_PERCENTILE: usize = 80;

#[inline]
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[inline]
fn shard_idx(key: &DataCacheKey, shard_mask: u64) -> usize {
    let h = key
        .file_id
        .wrapping_mul(11_400_714_819_323_198_485_u64)
        .wrapping_add(key.offset.wrapping_mul(6_364_136_223_846_793_005_u64));
    (h & shard_mask) as usize
}

#[derive(Clone)]
enum LoadState {
    Loading,
    Loaded(Bytes),
    Failed,
}

struct CacheEntry {
    key: DataCacheKey,
    state_tx: watch::Sender<LoadState>,
    last_use_ms: AtomicU64,
    num_uses: AtomicU32,
    data_size: AtomicU64,
}

impl CacheEntry {
    fn new(key: DataCacheKey) -> Arc<Self> {
        let (state_tx, _rx) = watch::channel(LoadState::Loading);
        Arc::new(Self {
            key,
            state_tx,
            last_use_ms: AtomicU64::new(0),
            num_uses: AtomicU32::new(0),
            data_size: AtomicU64::new(0),
        })
    }

    fn eviction_score(&self, now: u64) -> u64 {
        let last = self.last_use_ms.load(Ordering::Relaxed);
        if last == 0 {
            return u64::MAX;
        }
        let age = now.saturating_sub(last);
        let uses = self.num_uses.load(Ordering::Relaxed) as u64;
        age / (1 + uses)
    }

    fn touch(&self) {
        self.last_use_ms.store(now_ms(), Ordering::Relaxed);
        self.num_uses.fetch_add(1, Ordering::Relaxed);
    }
}

struct CacheShardInner {
    entries: HashMap<DataCacheKey, Arc<CacheEntry>>,
    dense_ring: Vec<Option<Arc<CacheEntry>>>,
    empty_slots: Vec<usize>,
    clock_hand: usize,
    eviction_threshold: u64,
    loaded_bytes: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    stale_evictions: u64,
}

impl CacheShardInner {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            dense_ring: Vec::new(),
            empty_slots: Vec::new(),
            clock_hand: 0,
            eviction_threshold: 0,
            loaded_bytes: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            stale_evictions: 0,
        }
    }

    fn ring_insert(&mut self, entry: Arc<CacheEntry>) {
        if let Some(idx) = self.empty_slots.pop() {
            self.dense_ring[idx] = Some(entry);
        } else {
            self.dense_ring.push(Some(entry));
        }
    }

    fn ring_evict_slot(&mut self, idx: usize) {
        self.dense_ring[idx] = None;
        self.empty_slots.push(idx);
    }

    fn calibrate_threshold(&mut self) {
        let n = self.dense_ring.len();
        if n == 0 {
            self.eviction_threshold = 0;
            return;
        }
        let num_samples = NUM_EVICTION_SAMPLES.min(n);
        let step = (n / num_samples).max(1);
        let now = now_ms();

        let mut scores: Vec<u64> = (0..num_samples)
            .filter_map(|i| {
                let idx = (self.clock_hand + i * step) % n;
                self.dense_ring[idx]
                    .as_ref()
                    .filter(|e| e.data_size.load(Ordering::Relaxed) > 0)
                    .map(|e| e.eviction_score(now))
            })
            .collect();

        if scores.is_empty() {
            // All sampled slots are Loading or None — set threshold to 0 so
            // that any loaded entry qualifies for eviction.
            self.eviction_threshold = 0;
            return;
        }

        scores.sort_unstable();
        let idx = (scores.len() * EVICTION_PERCENTILE / 100).min(scores.len().saturating_sub(1));
        self.eviction_threshold = scores.get(idx).copied().unwrap_or(0);
    }

    fn evict(&mut self, target_bytes: u64) -> (u64, Vec<(DataCacheKey, Bytes)>) {
        let n = self.dense_ring.len();
        if n == 0 {
            return (0, Vec::new());
        }

        let now = now_ms();
        let mut freed = 0u64;
        let mut counter = 0usize;
        let mut num_checked = 0usize;
        let mut evicted: Vec<(DataCacheKey, Bytes)> = Vec::new();

        while counter < n {
            let idx = self.clock_hand % n;
            self.clock_hand = self.clock_hand.wrapping_add(1);
            counter += 1;

            if self.dense_ring[idx].is_none() {
                continue;
            }
            let entry = self.dense_ring[idx].as_ref().unwrap().clone();
            num_checked += 1;

            if self.eviction_threshold == 0 || num_checked > n / 8 {
                self.calibrate_threshold();
                num_checked = 0;
            }

            // 3 owners = HashMap + dense_ring + this clone. >3 means an
            // active caller holds a reference — skip to avoid evicting a
            // live entry.
            if Arc::strong_count(&entry) > 3 {
                continue;
            }

            let size = entry.data_size.load(Ordering::Relaxed);
            if size == 0 {
                continue;
            }

            let score = entry.eviction_score(now);
            if score < self.eviction_threshold {
                continue;
            }

            if self.entries.remove(&entry.key).is_some() {
                if let LoadState::Loaded(bytes) = entry.state_tx.borrow().clone() {
                    evicted.push((entry.key.clone(), bytes));
                }
                entry.data_size.store(0, Ordering::Relaxed);
                self.ring_evict_slot(idx);
                self.loaded_bytes = self.loaded_bytes.saturating_sub(size);
                self.evictions += 1;
                freed += size;

                if freed >= target_bytes {
                    break;
                }
            }
        }
        (freed, evicted)
    }
}

pub struct ShardStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub stale_evictions: u64,
}

struct CacheShard {
    inner: Mutex<CacheShardInner>,
    per_shard_limit: u64,
}

impl CacheShard {
    fn new(per_shard_limit: u64) -> Self {
        Self {
            inner: Mutex::new(CacheShardInner::new()),
            per_shard_limit,
        }
    }

    fn find_or_create(
        &self,
        key: &DataCacheKey,
        min_size: u64,
    ) -> (Arc<CacheEntry>, bool, u64, Vec<(DataCacheKey, Bytes)>) {
        let mut inner = self.inner.lock().unwrap();

        if let Some(entry) = inner.entries.get(key).cloned() {
            let cached_size = entry.data_size.load(Ordering::Relaxed);
            if cached_size > 0 && cached_size < min_size {
                if let Some(e) = inner.entries.remove(key) {
                    let sz = e.data_size.swap(0, Ordering::Relaxed);
                    if sz > 0 {
                        inner.loaded_bytes = inner.loaded_bytes.saturating_sub(sz);
                    }
                    inner.stale_evictions += 1;
                    inner.evictions += 1;
                }
            } else {
                inner.hits += 1;
                return (entry, false, 0, Vec::new());
            }
        }

        let batch_evict_bytes = self.per_shard_limit / 5;
        let (freed, evicted) = if inner.loaded_bytes >= self.per_shard_limit {
            let overage = inner.loaded_bytes - self.per_shard_limit;
            let to_free = overage.max(batch_evict_bytes);
            inner.evict(to_free)
        } else {
            (0, Vec::new())
        };

        inner.misses += 1;
        let entry = CacheEntry::new(key.clone());
        inner.entries.insert(key.clone(), entry.clone());
        inner.ring_insert(entry.clone());
        (entry, true, freed, evicted)
    }

    fn record_loaded(&self, size: u64) {
        self.inner.lock().unwrap().loaded_bytes += size;
    }

    fn remove(&self, key: &DataCacheKey) -> Option<u64> {
        let mut inner = self.inner.lock().unwrap();
        if let Some(entry) = inner.entries.remove(key) {
            let size = entry.data_size.load(Ordering::Relaxed);
            entry.data_size.store(0, Ordering::Relaxed);
            if size > 0 {
                inner.loaded_bytes = inner.loaded_bytes.saturating_sub(size);
            }
            Some(size)
        } else {
            None
        }
    }

    fn stats(&self) -> ShardStats {
        let inner = self.inner.lock().unwrap();
        ShardStats {
            hits: inner.hits,
            misses: inner.misses,
            evictions: inner.evictions,
            stale_evictions: inner.stale_evictions,
        }
    }
}

pub trait EvictionSink: Send + Sync + std::fmt::Debug {
    fn on_evicted(&self, entries: Vec<(DataCacheKey, Bytes)>, total_cache_bytes: u64);
}

#[derive(Debug, Default, Clone)]
pub struct MemoryCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub stale_evictions: u64,
    pub current_bytes: u64,
    pub max_bytes: u64,
}

pub struct MemoryCache {
    shards: Vec<CacheShard>,
    shard_mask: u64,
    max_bytes: u64,
    total_bytes: AtomicU64,
    eviction_sink: Option<Arc<dyn EvictionSink>>,
}

impl std::fmt::Debug for MemoryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryCache")
            .field("max_bytes", &self.max_bytes)
            .field("current_bytes", &self.total_bytes.load(Ordering::Relaxed))
            .finish()
    }
}

impl MemoryCache {
    pub fn new(max_bytes: u64) -> Arc<Self> {
        Self::new_with_shards(max_bytes, DEFAULT_NUM_SHARDS)
    }

    pub fn new_with_shards(max_bytes: u64, num_shards: usize) -> Arc<Self> {
        Self::with_eviction_sink(max_bytes, num_shards, None)
    }

    pub fn with_eviction_sink(
        max_bytes: u64,
        num_shards: usize,
        eviction_sink: Option<Arc<dyn EvictionSink>>,
    ) -> Arc<Self> {
        assert!(num_shards > 0, "num_shards must be positive");
        assert!(
            num_shards.is_power_of_two(),
            "num_shards must be a power of two, got {num_shards}"
        );
        let shard_mask = (num_shards as u64) - 1;
        let per_shard_limit = max_bytes / num_shards as u64;
        let shards = (0..num_shards)
            .map(|_| CacheShard::new(per_shard_limit))
            .collect();
        Arc::new(Self {
            shards,
            shard_mask,
            max_bytes,
            total_bytes: AtomicU64::new(0),
            eviction_sink,
        })
    }

    pub fn stats(&self) -> MemoryCacheStats {
        let (hits, misses, evictions, stale_evictions) =
            self.shards
                .iter()
                .fold((0u64, 0u64, 0u64, 0u64), |(h, m, e, s), shard| {
                    let st = shard.stats();
                    (
                        h + st.hits,
                        m + st.misses,
                        e + st.evictions,
                        s + st.stale_evictions,
                    )
                });
        MemoryCacheStats {
            hits,
            misses,
            evictions,
            stale_evictions,
            current_bytes: self.total_bytes.load(Ordering::Relaxed),
            max_bytes: self.max_bytes,
        }
    }

    pub async fn get_or_load(
        &self,
        key: DataCacheKey,
        length: u64,
        loader: BoxFuture<'_, Result<Bytes>>,
    ) -> Result<Bytes> {
        if self.max_bytes == 0 {
            return loader.await;
        }
        let (entry, is_new) = self.find_or_create(&key, length);
        if is_new {
            return self.load_exclusive(&key, length, &entry, loader).await;
        }
        let bytes = self
            .wait_for_entry(&key, length, &entry)
            .await
            .ok_or_else(|| {
                lance_core::Error::io(
                    "concurrent cache load failed; retry the operation".to_string(),
                )
            })?;
        Ok(bytes)
    }

    async fn load_exclusive(
        &self,
        key: &DataCacheKey,
        length: u64,
        entry: &Arc<CacheEntry>,
        loader: BoxFuture<'_, Result<Bytes>>,
    ) -> Result<Bytes> {
        match loader.await {
            Ok(bytes) => {
                let size = bytes.len() as u64;
                tracing::trace!(
                    file_id = key.file_id,
                    offset = key.offset,
                    size_bytes = size,
                    "memory cache miss — entry loaded and stored"
                );
                entry.data_size.store(size, Ordering::Release);
                entry.touch();
                entry
                    .state_tx
                    .send_replace(LoadState::Loaded(bytes.clone()));
                let shard = &self.shards[shard_idx(key, self.shard_mask)];
                shard.record_loaded(size);
                self.total_bytes.fetch_add(size, Ordering::Relaxed);
                Ok(bytes.slice(0..length.min(size) as usize))
            }
            Err(e) => {
                entry.state_tx.send_replace(LoadState::Failed);
                self.remove_entry(key);
                Err(e)
            }
        }
    }

    async fn wait_for_entry(
        &self,
        key: &DataCacheKey,
        length: u64,
        entry: &Arc<CacheEntry>,
    ) -> Option<Bytes> {
        let mut rx = entry.state_tx.subscribe();
        loop {
            let state = rx.borrow_and_update().clone();
            match state {
                LoadState::Loaded(bytes) => {
                    entry.touch();
                    tracing::trace!(
                        file_id = key.file_id,
                        offset = key.offset,
                        size_bytes = bytes.len(),
                        "memory cache hit"
                    );
                    return Some(bytes.slice(0..length.min(bytes.len() as u64) as usize));
                }
                LoadState::Failed => {
                    return None;
                }
                LoadState::Loading => {
                    if rx.changed().await.is_err() {
                        return None;
                    }
                }
            }
        }
    }

    fn find_or_create(&self, key: &DataCacheKey, min_size: u64) -> (Arc<CacheEntry>, bool) {
        let (entry, is_new, freed, evicted) =
            self.shards[shard_idx(key, self.shard_mask)].find_or_create(key, min_size);
        if freed > 0 {
            self.total_bytes
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                    Some(v.saturating_sub(freed))
                })
                .ok();
        }
        if !evicted.is_empty()
            && let Some(sink) = &self.eviction_sink
        {
            sink.on_evicted(evicted, self.total_bytes.load(Ordering::Relaxed));
        }
        (entry, is_new)
    }

    fn remove_entry(&self, key: &DataCacheKey) {
        let idx = shard_idx(key, self.shard_mask);
        if let Some(size) = self.shards[idx].remove(key)
            && size > 0
        {
            self.total_bytes
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                    Some(v.saturating_sub(size))
                })
                .ok();
        }
    }
}

pub struct StandaloneMemoryCache {
    inner: Arc<MemoryCache>,
    file_ids: Arc<FileIds>,
}

impl std::fmt::Debug for StandaloneMemoryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

impl StandaloneMemoryCache {
    pub fn new(max_bytes: u64) -> Arc<Self> {
        Arc::new(Self {
            inner: MemoryCache::new(max_bytes),
            file_ids: Arc::new(FileIds::new()),
        })
    }

    pub fn stats(&self) -> MemoryCacheStats {
        self.inner.stats()
    }
}

impl DataCache for StandaloneMemoryCache {
    fn intern_file(&self, path: &Path) -> u64 {
        self.file_ids.get_or_intern(path)
    }

    fn get_or_load_by_id<'a>(
        &'a self,
        file_id: u64,
        offset: u64,
        length: u64,
        loader: BoxFuture<'a, Result<Bytes>>,
    ) -> BoxFuture<'a, Result<Bytes>> {
        let key = DataCacheKey { file_id, offset };
        Box::pin(self.inner.get_or_load(key, length, loader))
    }

    fn cache_stats(&self) -> super::CacheStats {
        let s = self.inner.stats();
        super::CacheStats {
            memory_hits: s.hits,
            memory_misses: s.misses,
            memory_evictions: s.evictions,
            memory_current_bytes: s.current_bytes,
            memory_stale_evictions: s.stale_evictions,
            ssd_hits: 0,
            ssd_bytes_written: 0,
            ssd_stale_misses: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn key(file_id: u64, offset: u64) -> DataCacheKey {
        DataCacheKey { file_id, offset }
    }

    /// Verifies the default shard count matches DEFAULT_NUM_SHARDS.
    #[test]
    fn test_default_shard_count() {
        let cache = MemoryCache::new(1024 * 1024);
        assert_eq!(cache.shards.len(), DEFAULT_NUM_SHARDS);
    }

    /// Verifies shard count and mask are set correctly for all valid power-of-two counts.
    #[test]
    fn test_custom_shard_count_power_of_two() {
        for &n in &[1usize, 2, 4, 8, 32, 64] {
            let cache = MemoryCache::new_with_shards(1024 * 1024, n);
            assert_eq!(cache.shards.len(), n);
            assert_eq!(cache.shard_mask, (n as u64) - 1);
        }
    }

    /// Construction must panic if num_shards is not a power of two.
    #[test]
    #[should_panic(expected = "power of two")]
    fn test_non_power_of_two_shards_panics() {
        MemoryCache::new_with_shards(1024 * 1024, 3);
    }

    /// Construction must panic if num_shards is zero.
    #[test]
    #[should_panic(expected = "positive")]
    fn test_zero_shards_panics() {
        MemoryCache::new_with_shards(1024 * 1024, 0);
    }

    /// Smoke test for single-shard mode: a miss loads and a subsequent request hits.
    #[tokio::test]
    async fn test_single_shard_cache_works() {
        let cache = MemoryCache::new_with_shards(10 * 1024 * 1024, 1);
        let k = key(0, 0);
        let bytes = cache
            .get_or_load(
                k.clone(),
                2,
                Box::pin(async { Ok(Bytes::from_static(b"hi")) }),
            )
            .await
            .unwrap();
        assert_eq!(bytes, Bytes::from_static(b"hi"));
        let bytes2 = cache
            .get_or_load(k, 2, Box::pin(async { Ok(Bytes::from_static(b"miss")) }))
            .await
            .unwrap();
        assert_eq!(bytes2, Bytes::from_static(b"hi"));
        assert_eq!(cache.stats().hits, 1);
    }

    /// First request misses and invokes the loader; second request hits and
    /// returns the cached value without calling the loader again.
    #[tokio::test]
    async fn test_basic_hit_and_miss() {
        let cache = MemoryCache::new(10 * 1024 * 1024);
        let k = key(0, 0);

        let bytes = cache
            .get_or_load(
                k.clone(),
                5,
                Box::pin(async { Ok(Bytes::from_static(b"hello")) }),
            )
            .await
            .unwrap();
        assert_eq!(bytes, Bytes::from_static(b"hello"));

        let load_count = Arc::new(AtomicUsize::new(0));
        let lc = load_count.clone();
        let bytes2 = cache
            .get_or_load(
                k.clone(),
                5,
                Box::pin(async move {
                    lc.fetch_add(1, Ordering::Relaxed);
                    Ok(Bytes::from_static(b"should not be called"))
                }),
            )
            .await
            .unwrap();
        assert_eq!(bytes2, Bytes::from_static(b"hello"));
        assert_eq!(load_count.load(Ordering::Relaxed), 0);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    /// Eight concurrent requests for the same key must trigger the loader exactly
    /// once — all waiters coalesce on the in-flight load.
    #[tokio::test]
    async fn test_load_deduplication() {
        let cache = Arc::new(MemoryCache::new(10 * 1024 * 1024));
        let k = key(1, 0);
        let load_count = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let cache = cache.clone();
            let k = k.clone();
            let lc = load_count.clone();
            handles.push(tokio::spawn(async move {
                cache
                    .get_or_load(
                        k,
                        4,
                        Box::pin(async move {
                            lc.fetch_add(1, Ordering::Relaxed);
                            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                            Ok(Bytes::from_static(b"data"))
                        }),
                    )
                    .await
            }));
        }

        for h in handles {
            assert_eq!(h.await.unwrap().unwrap(), Bytes::from_static(b"data"));
        }
        assert_eq!(load_count.load(Ordering::Relaxed), 1);
    }

    /// A failed load removes the entry so the next request can retry with a
    /// fresh loader instead of being stuck with a permanent error.
    #[tokio::test]
    async fn test_loader_failure_allows_retry() {
        let cache = Arc::new(MemoryCache::new(10 * 1024 * 1024));
        let k = key(2, 0);

        let result = cache
            .get_or_load(
                k.clone(),
                8,
                Box::pin(async { Err(lance_core::Error::io("boom".to_string())) }),
            )
            .await;
        assert!(result.is_err());

        let bytes = cache
            .get_or_load(
                k,
                8,
                Box::pin(async { Ok(Bytes::from_static(b"retry ok")) }),
            )
            .await
            .unwrap();
        assert_eq!(bytes, Bytes::from_static(b"retry ok"));
    }

    /// Verifies that the cache evicts entries when memory is full.
    /// Loads 2x the cache capacity and checks that eviction fired
    /// and memory stayed bounded.
    #[tokio::test]
    async fn test_eviction_under_pressure() {
        let cache_cap = 4 * 1024 * 1024;
        let entry_size = 1024 * 1024;
        // Load 2x the cache capacity — eviction must fire to keep memory bounded.
        let num_entries = 8u64;

        let cache = MemoryCache::new_with_shards(cache_cap, 1);
        let chunk = Bytes::from(vec![0u8; entry_size]);

        for i in 0..num_entries {
            let data = chunk.clone();
            cache
                .get_or_load(
                    key(3, i),
                    entry_size as u64,
                    Box::pin(async move { Ok(data) }),
                )
                .await
                .unwrap();
        }

        let stats = cache.stats();
        assert!(stats.evictions > 0, "expected at least one eviction");
        assert!(
            stats.current_bytes <= stats.max_bytes * 2,
            "cache grew too large: {} bytes",
            stats.current_bytes
        );
    }

    /// When max_bytes is 0 the cache is disabled — every call invokes the
    /// loader directly with no caching or deduplication.
    #[tokio::test]
    async fn test_disabled_cache_bypasses() {
        let cache = MemoryCache::new(0);
        let k = key(4, 0);
        let count = Arc::new(AtomicUsize::new(0));

        for _ in 0..3 {
            let c = count.clone();
            cache
                .get_or_load(
                    k.clone(),
                    1,
                    Box::pin(async move {
                        c.fetch_add(1, Ordering::Relaxed);
                        Ok(Bytes::from_static(b"x"))
                    }),
                )
                .await
                .unwrap();
        }
        assert_eq!(count.load(Ordering::Relaxed), 3);
    }

    /// Three-phase lifecycle test:
    /// 1. Fill the cache to exactly capacity.
    /// 2. Re-request the same keys — all should hit, memory must stay bounded.
    /// 3. Request new keys beyond capacity — eviction must fire to make room.
    #[tokio::test]
    async fn test_replace_hits_and_evictions() {
        let cap = 4 * 1024 * 1024u64;
        let entry_size = 256 * 1024u64;
        let num_entries = cap / entry_size;
        let cache = MemoryCache::new_with_shards(cap, 1);

        for i in 0..num_entries {
            let data = Bytes::from(vec![(i % 256) as u8; entry_size as usize]);
            cache
                .get_or_load(
                    key(0, i * entry_size),
                    entry_size,
                    Box::pin(async move { Ok(data) }),
                )
                .await
                .unwrap();
        }

        for i in 0..num_entries {
            let data = Bytes::from(vec![(i % 256) as u8; entry_size as usize]);
            let _ = cache
                .get_or_load(
                    key(0, i * entry_size),
                    entry_size,
                    Box::pin(async move { Ok(data) }),
                )
                .await
                .unwrap();
        }

        let stats = cache.stats();
        assert!(stats.hits > 0, "expected cache hits on second pass, got 0");
        assert!(
            stats.current_bytes <= cap,
            "cache exceeded capacity: {} > {}",
            stats.current_bytes,
            cap
        );

        for i in num_entries..num_entries * 2 {
            let data = Bytes::from(vec![0u8; entry_size as usize]);
            cache
                .get_or_load(
                    key(0, i * entry_size),
                    entry_size,
                    Box::pin(async move { Ok(data) }),
                )
                .await
                .unwrap();
        }

        let stats2 = cache.stats();
        assert!(stats2.evictions > 0, "expected evictions beyond capacity");
    }

    /// Guards against u64 underflow in total_bytes. If eviction double-subtracts,
    /// total_bytes wraps to near u64::MAX (~18 exabytes). Loads 8x capacity to
    /// maximise eviction pressure and asserts the counter stayed sane.
    #[tokio::test]
    async fn test_accounting_invariant_under_eviction() {
        let cap = 2 * 1024 * 1024u64;
        let entry_size = 128 * 1024u64;
        let num_entries = (cap / entry_size) * 8;
        let cache = Arc::new(MemoryCache::new(cap));

        for i in 0..num_entries {
            let data = Bytes::from(vec![0u8; entry_size as usize]);
            cache
                .get_or_load(
                    key(1, i * entry_size),
                    entry_size,
                    Box::pin(async move { Ok(data) }),
                )
                .await
                .unwrap();
        }

        let stats = cache.stats();
        assert!(
            stats.current_bytes < cap * 2,
            "possible underflow: current_bytes={} (u64::MAX would indicate wrap)",
            stats.current_bytes
        );
        assert!(stats.evictions > 0, "expected evictions to fire");
    }

    /// When the exclusive loader fails, all concurrent waiters receive the error
    /// via the watch channel. A subsequent request after the failure succeeds.
    #[tokio::test]
    async fn test_concurrent_waiters_see_failure_and_retry() {
        let cache = Arc::new(MemoryCache::new(10 * 1024 * 1024));
        let k = key(5, 0);

        let (loading_tx, loading_rx) = tokio::sync::oneshot::channel::<()>();
        let (fail_tx, fail_rx) = tokio::sync::oneshot::channel::<()>();

        // Spawn the exclusive loader first and wait for it to signal it has started.
        let cache_clone = cache.clone();
        let k_clone = k.clone();
        let exclusive = tokio::spawn(async move {
            cache_clone
                .get_or_load(
                    k_clone,
                    2,
                    Box::pin(async move {
                        loading_tx.send(()).ok(); // entry is in ring, load is in-flight
                        fail_rx.await.ok(); // hold until test releases the failure
                        Err(lance_core::Error::io("injected failure".to_string()))
                    }),
                )
                .await
        });

        // Wait until the exclusive load is registered before spawning waiters.
        loading_rx.await.ok();

        // Spawn 7 waiters — all will subscribe to the Loading watch channel.
        // Their loaders must never be called (deduplication guarantee).
        let mut waiter_handles = Vec::new();
        for _ in 0..7 {
            let c = cache.clone();
            let k = k.clone();
            waiter_handles.push(tokio::spawn(async move {
                c.get_or_load(
                    k,
                    2,
                    Box::pin(async { panic!("waiter loader must not run") }),
                )
                .await
            }));
        }

        // Yield once so all 7 waiters run their synchronous setup (find_or_create +
        // subscribe) and park at rx.changed().await before the failure fires.
        // This is deterministic on the default single-threaded tokio runtime.
        tokio::task::yield_now().await;

        // Release the exclusive loader to fail.
        fail_tx.send(()).ok();

        assert!(
            exclusive.await.unwrap().is_err(),
            "exclusive loader should propagate the error"
        );
        for h in waiter_handles {
            assert!(
                h.await.unwrap().is_err(),
                "all waiters should see the failure"
            );
        }

        // A fresh request after cleanup must succeed.
        let bytes = cache
            .get_or_load(k, 5, Box::pin(async { Ok(Bytes::from_static(b"fresh")) }))
            .await
            .unwrap();
        assert_eq!(bytes, Bytes::from_static(b"fresh"));
    }

    /// 8 workers hammer a small key space for 500 ms. Verifies hits and misses
    /// both occur and memory never exceeds the cap under concurrent eviction.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_fuzz_concurrent_access() {
        use rand::Rng;

        let cap = 8 * 1024 * 1024u64;
        let entry_size = 64 * 1024u64;
        let num_files = 5u64;
        let offsets_per_file = 20u64;
        let cache = Arc::new(MemoryCache::new(cap));
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);

        let mut handles = Vec::new();
        for _worker in 0..8 {
            let cache = cache.clone();
            handles.push(tokio::spawn(async move {
                use rand::{SeedableRng, rngs::SmallRng};
                let mut rng = SmallRng::from_os_rng();
                while std::time::Instant::now() < deadline {
                    let file_id = rng.random_range(0..num_files);
                    let offset_idx = rng.random_range(0..offsets_per_file);
                    let offset = offset_idx * entry_size;
                    let k = key(file_id, offset);

                    let data = Bytes::from(vec![(file_id ^ offset_idx) as u8; entry_size as usize]);
                    let _ = cache
                        .get_or_load(k, entry_size, Box::pin(async move { Ok(data) }))
                        .await
                        .unwrap();
                }
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        let stats = cache.stats();
        assert!(stats.hits > 0, "expected hits in fuzz run");
        assert!(stats.misses > 0, "expected misses in fuzz run");
        assert!(
            stats.current_bytes <= cap,
            "cache exceeded capacity during fuzz: {} > {}",
            stats.current_bytes,
            cap
        );
    }

    /// Verifies hits, misses, and current_bytes counters are exact after
    /// N requests where only the first is a miss.
    #[tokio::test]
    async fn test_stats_accounting() {
        let cache = MemoryCache::new(10 * 1024 * 1024);
        let k = key(6, 0);
        let requests = 10usize;

        for _ in 0..requests {
            cache
                .get_or_load(
                    k.clone(),
                    1,
                    Box::pin(async { Ok(Bytes::from_static(b"x")) }),
                )
                .await
                .unwrap();
        }

        let stats = cache.stats();
        assert_eq!(stats.misses, 1, "only first request should miss");
        assert_eq!(
            stats.hits,
            (requests - 1) as u64,
            "remaining requests should hit"
        );
        assert_eq!(stats.current_bytes, 1);
    }

    /// 8 threads concurrently insert unique keys into a small cache, forcing
    /// continuous eviction. Checks that total_bytes never wraps (u64 underflow).
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_total_bytes_no_underflow_under_concurrent_eviction() {
        let cap = 512 * 1024u64;
        let entry_size = 64 * 1024u64;
        let cache = Arc::new(MemoryCache::new(cap));

        let mut handles = Vec::new();
        for thread_id in 0..8u64 {
            let cache = cache.clone();
            handles.push(tokio::spawn(async move {
                for i in 0..64u64 {
                    let offset = (thread_id * 1000 + i) * entry_size;
                    let data = Bytes::from(vec![0u8; entry_size as usize]);
                    cache
                        .get_or_load(
                            key(thread_id, offset),
                            entry_size,
                            Box::pin(async move { Ok(data) }),
                        )
                        .await
                        .unwrap();
                }
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        let stats = cache.stats();

        let one_tib = 1u64 << 40;
        assert!(
            stats.current_bytes < one_tib,
            "u64 underflow detected: total_bytes wrapped to {}",
            stats.current_bytes
        );
        assert!(
            stats.evictions > 0,
            "expected evictions under constant pressure"
        );
    }

    /// Eviction sweep must not crash or double-count when the only ring entry
    /// is still Loading (data_size == 0). Verifies accounting stays sane after
    /// the load completes.
    #[tokio::test]
    async fn test_eviction_graceful_when_all_entries_loading() {
        let cap = 128 * 1024u64;
        let entry_size = 128 * 1024u64;
        let cache = Arc::new(MemoryCache::new(cap));

        let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
        let cache_clone = cache.clone();

        let loading = tokio::spawn(async move {
            let _ = cache_clone
                .get_or_load(
                    key(99, 0),
                    entry_size,
                    Box::pin(async move {
                        started_tx.send(()).ok(); // entry is now in the ring
                        release_rx.await.ok();
                        Ok(Bytes::from(vec![0u8; entry_size as usize]))
                    }),
                )
                .await
                .unwrap();
        });

        // Wait until the Loading entry is registered in the ring before inserting
        // the second entry that triggers eviction. No sleep needed.
        started_rx.await.ok();

        let data = Bytes::from(vec![1u8; entry_size as usize]);
        let _ = cache
            .get_or_load(
                key(99, entry_size),
                entry_size,
                Box::pin(async move { Ok(data) }),
            )
            .await
            .unwrap();

        release_tx.send(()).ok();
        loading.await.unwrap();

        let one_tib = 1u64 << 40;
        assert!(cache.stats().current_bytes < one_tib);
    }

    /// Eviction must skip size-0 entries (Loading or stale-evicted) and must
    /// not double-account their bytes when they eventually complete.
    #[tokio::test]
    async fn test_eviction_skips_zero_size_entries() {
        let cap = 256 * 1024u64;
        let entry_size = 128 * 1024u64;
        let cache = MemoryCache::new(cap);

        for i in 0..2u64 {
            let data = Bytes::from(vec![i as u8; entry_size as usize]);
            cache
                .get_or_load(
                    key(0, i * entry_size),
                    entry_size,
                    Box::pin(async move { Ok(data) }),
                )
                .await
                .unwrap();
        }

        let before = cache.stats().current_bytes;

        for i in 2..4u64 {
            let data = Bytes::from(vec![i as u8; entry_size as usize]);
            cache
                .get_or_load(
                    key(0, i * entry_size),
                    entry_size,
                    Box::pin(async move { Ok(data) }),
                )
                .await
                .unwrap();
        }

        let after = cache.stats().current_bytes;

        assert!(
            after < cap * 4,
            "possible double-accounting: before={before} after={after}"
        );
        assert!(cache.stats().evictions > 0);
    }

    /// After sustained pressure (32x capacity), memory must converge to within
    /// 2x cap — the batched eviction policy must not let the cache grow unboundedly.
    #[tokio::test]
    async fn test_eviction_converges_to_cap() {
        let cap = 1024 * 1024u64;
        let entry_size = 128 * 1024u64;
        let cache = MemoryCache::new(cap);

        for i in 0..32u64 {
            let data = Bytes::from(vec![0u8; entry_size as usize]);
            cache
                .get_or_load(
                    key(0, i * entry_size),
                    entry_size,
                    Box::pin(async move { Ok(data) }),
                )
                .await
                .unwrap();
        }

        let data = Bytes::from(vec![0u8; entry_size as usize]);
        cache
            .get_or_load(key(1, 0), entry_size, Box::pin(async move { Ok(data) }))
            .await
            .unwrap();

        let stats = cache.stats();
        assert!(
            stats.current_bytes <= cap * 2,
            "eviction did not converge: current_bytes={} cap={}",
            stats.current_bytes,
            cap
        );
        assert!(stats.evictions > 0);
    }

    /// A cache miss invokes the loader exactly once and records one miss, zero hits.
    #[tokio::test]
    async fn test_find_miss() {
        let cache = MemoryCache::new(10 * 1024 * 1024);
        let k = key(10, 0);
        let load_count = Arc::new(AtomicUsize::new(0));
        let lc = load_count.clone();

        let bytes = cache
            .get_or_load(
                k.clone(),
                4,
                Box::pin(async move {
                    lc.fetch_add(1, Ordering::Relaxed);
                    Ok(Bytes::from_static(b"data"))
                }),
            )
            .await
            .unwrap();
        assert_eq!(bytes, Bytes::from_static(b"data"));
        assert_eq!(
            load_count.load(Ordering::Relaxed),
            1,
            "loader must run on miss"
        );
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
    }

    /// A cache hit returns the exact same bytes (same Arc pointer) as the
    /// original load — no copy, no data corruption.
    #[tokio::test]
    async fn test_find_hit_data_integrity() {
        let cache = MemoryCache::new(10 * 1024 * 1024);
        let pattern: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let original = Bytes::from(pattern.clone());
        let k = key(11, 0);

        let b1 = cache
            .get_or_load(k.clone(), 4096, Box::pin(async move { Ok(original) }))
            .await
            .unwrap();
        assert_eq!(
            b1.as_ref(),
            pattern.as_slice(),
            "loaded bytes must match pattern"
        );

        let b2 = cache
            .get_or_load(
                k,
                4096,
                Box::pin(async { panic!("loader must not be called on hit") }),
            )
            .await
            .unwrap();
        assert_eq!(
            b2.as_ref(),
            pattern.as_slice(),
            "hit bytes must match original"
        );
        assert_eq!(
            b1.as_ptr(),
            b2.as_ptr(),
            "hit must return the cached Arc, not a copy"
        );
        assert_eq!(cache.stats().hits, 1);
    }

    /// Verifies all stats fields (hits, misses, max_bytes, current_bytes) are
    /// exact after a known load + hit pattern.
    #[tokio::test]
    async fn test_cache_stats_fields() {
        let cache = MemoryCache::new_with_shards(2 * 1024 * 1024, 1);
        let entry_size = 256 * 1024u64;

        for i in 0u64..4 {
            let data = Bytes::from(vec![i as u8; entry_size as usize]);
            cache
                .get_or_load(
                    key(12, i * entry_size),
                    entry_size,
                    Box::pin(async move { Ok(data) }),
                )
                .await
                .unwrap();
        }

        for i in 0u64..4 {
            let data = Bytes::from(vec![i as u8; entry_size as usize]);
            cache
                .get_or_load(
                    key(12, i * entry_size),
                    entry_size,
                    Box::pin(async move { Ok(data) }),
                )
                .await
                .unwrap();
        }

        let stats = cache.stats();
        assert_eq!(stats.misses, 4, "misses");
        assert_eq!(stats.hits, 4, "hits");
        assert_eq!(stats.max_bytes, 2 * 1024 * 1024, "max_bytes");
        assert_eq!(stats.current_bytes, 4 * entry_size, "current_bytes");
    }

    /// While one task holds the exclusive load in-flight, a second task must
    /// wait on the watch channel and receive the same bytes without calling its
    /// own loader. Both results must point to the same Arc (zero-copy).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_exclusive_to_shared_transition() {
        let cache = Arc::new(MemoryCache::new(10 * 1024 * 1024));
        let k = key(13, 0);

        let (loading_tx, loading_rx) = tokio::sync::oneshot::channel::<()>();
        let (done_tx, _done_rx) = tokio::sync::oneshot::channel::<()>();

        let cache_a = cache.clone();
        let k_a = k.clone();
        let loader_task = tokio::spawn(async move {
            cache_a
                .get_or_load(
                    k_a,
                    9,
                    Box::pin(async move {
                        loading_tx.send(()).ok();
                        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                        done_tx.send(()).ok();
                        Ok(Bytes::from_static(b"exclusive"))
                    }),
                )
                .await
                .unwrap()
        });

        loading_rx.await.ok();

        let cache_b = cache.clone();
        let k_b = k.clone();
        let waiter_task = tokio::spawn(async move {
            cache_b
                .get_or_load(
                    k_b,
                    9,
                    Box::pin(async { panic!("waiter must not call its own loader") }),
                )
                .await
                .unwrap()
        });

        let a_result = loader_task.await.unwrap();
        let b_result = waiter_task.await.unwrap();

        assert_eq!(a_result, Bytes::from_static(b"exclusive"));
        assert_eq!(b_result, Bytes::from_static(b"exclusive"));
        assert_eq!(a_result.as_ptr(), b_result.as_ptr());
    }
}
