// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! In-memory cache tier — a Rust port of CacheShard / `AsyncDataCache`.
//!
//! Design mirrors exactly:
//!
//! * **16 independent shards** — `HashMap` + clock-hand eviction ring per shard,
//!   each protected by a `std::sync::Mutex`. Shards eliminate contention for the
//!   common case where different tasks access different files.
//!
//! * **Load deduplication** — a `tokio::sync::watch` channel replaces
//!   `folly::SharedPromise`. The first task to miss the cache transitions the
//!   entry from `Loading` to `Loaded(bytes)` or `Failed`; all concurrent tasks
//!   wait on the channel and receive the result without issuing a second fetch.
//!
//! * **Clock-hand eviction with percentile threshold** — follows
//!   `CacheShard::evict` / `calibrateThreshold` exactly. Every `ring_len/4`
//!   insertions the shard samples `NUM_EVICTION_SAMPLES` (10) entries, sorts
//!   their scores, and sets the eviction threshold to the
//!   `EVICTION_PERCENTILE`th (80th) percentile. Only entries scoring *above*
//!   the threshold are candidates, which prevents thrashing when the cache
//!   hovers at capacity.
//!
//! * **Score formula** — `(now_ms - last_use_ms) / (1 + num_uses)`. Older,
//!   less-frequently-accessed entries score higher and are evicted first.
//!   An entry that has never been accessed scores `u64::MAX` (evict immediately).

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

// ─── Constants (same as ) ───────────────────────────────────────────────

/// Default number of independent shards — must be a power of two.
/// Matches AsyncDataCache::kDefaultNumShards.
pub const DEFAULT_NUM_SHARDS: usize = 16;

/// Number of entries sampled when calibrating the eviction threshold.
const NUM_EVICTION_SAMPLES: usize = 10;

/// Only entries whose score is at or above this percentile are evicted.
const EVICTION_PERCENTILE: usize = 80;

// ─── Time ────────────────────────────────────────────────────────────────────

/// Milliseconds since the Unix epoch — cheap, ~1 ms resolution.
///
/// uses `folly::hardware_timestamp() >> 21` for ~1–2 ms resolution;
/// `SystemTime` gives the same order of magnitude with no unsafe code.
#[inline]
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ─── Shard selection ─────────────────────────────────────────────────────────

#[inline]
fn shard_idx(key: &DataCacheKey, shard_mask: u64) -> usize {
    // Fibonacci hashing — mixes file_id and offset uniformly across shards.
    let h = key
        .file_id
        .wrapping_mul(11_400_714_819_323_198_485_u64)
        .wrapping_add(key.offset.wrapping_mul(6_364_136_223_846_793_005_u64));
    (h & shard_mask) as usize
}

// ─── Entry ───────────────────────────────────────────────────────────────────

/// Possible states of a cache entry, broadcast via a `watch` channel.
///
/// Maps to numPins_ convention:
/// * `Loading` ≡ `numPins_ = kExclusive (-10000)`
/// * `Loaded(bytes)` ≡ `numPins_ = 1` (shared, data available)
/// * `Failed` ≡ entry removed from map; waiters must retry
#[derive(Clone)]
enum LoadState {
    Loading,
    Loaded(Bytes),
    Failed,
}

struct CacheEntry {
    key: DataCacheKey,
    /// The `Sender` half is owned here; receivers are created on demand by
    /// concurrent waiters. Sending a new state wakes *all* current waiters
    /// simultaneously — the same semantics as SharedPromise::setValue.
    state_tx: watch::Sender<LoadState>,
    /// Milliseconds since epoch of last access; 0 = never accessed.
    last_use_ms: AtomicU64,
    /// How many times this entry has been read (used in eviction scoring).
    num_uses: AtomicU32,
    /// Byte size of the cached payload; 0 while Loading or after failure.
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

    /// Eviction score. Higher = more worth evicting.
    ///
    /// Formula: `(now_ms - last_use_ms) / (1 + num_uses)` — identical to
    /// AccessStats::score.
    fn eviction_score(&self, now: u64) -> u64 {
        let last = self.last_use_ms.load(Ordering::Relaxed);
        if last == 0 {
            return u64::MAX; // never accessed → always evict first
        }
        let age = now.saturating_sub(last);
        let uses = self.num_uses.load(Ordering::Relaxed) as u64;
        age / (1 + uses)
    }

    /// Record an access (used when returning a cached hit).
    fn touch(&self) {
        self.last_use_ms.store(now_ms(), Ordering::Relaxed);
        self.num_uses.fetch_add(1, Ordering::Relaxed);
    }
}

// ─── Shard ───────────────────────────────────────────────────────────────────

/// Velox-style dense ring for the clock-hand eviction sweep.
///
/// Unlike a `Vec<Weak<CacheEntry>>`, this never accumulates dead pointers:
/// - Evicted slots are set to `None` and their index goes into `empty_slots`.
/// - New entries reuse a free slot (pop from `empty_slots`) or append to back.
/// - `clock_hand` advances through `dense_ring` — `None` slots are skipped
///   in O(1) with a plain `is_none()` check, no `Weak::upgrade()` needed.
struct CacheShardInner {
    /// O(1) key → entry lookup.
    entries: HashMap<DataCacheKey, Arc<CacheEntry>>,
    /// Dense ring — `None` means the slot is free for reuse.
    /// Size = high-water mark of concurrent entries ever held.
    dense_ring: Vec<Option<Arc<CacheEntry>>>,
    /// Indices of `None` slots available for reuse (Velox's `emptySlots_`).
    empty_slots: Vec<usize>,
    /// Clock-hand position in `dense_ring`.
    clock_hand: usize,
    /// 80th-percentile eviction score threshold.
    /// `0` = uncalibrated (everything evictable until first calibration).
    eviction_threshold: u64,
    /// Bytes of Loaded entries currently in this shard.
    loaded_bytes: u64,
    // Stats — plain u64 protected by the shard mutex.
    hits: u64,
    misses: u64,
    evictions: u64,
    /// Stale entries evicted because cached size < requested size (Velox stale-entry logic).
    stale_evictions: u64,
}

impl CacheShardInner {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            dense_ring: Vec::new(),
            empty_slots: Vec::new(),
            clock_hand: 0,
            eviction_threshold: 0, // everything evictable until calibration warms up
            loaded_bytes: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            stale_evictions: 0,
        }
    }

    /// Insert `entry` into the dense ring, reusing a free slot when available.
    fn ring_insert(&mut self, entry: Arc<CacheEntry>) {
        if let Some(idx) = self.empty_slots.pop() {
            self.dense_ring[idx] = Some(entry);
        } else {
            self.dense_ring.push(Some(entry));
        }
    }

    /// Remove entry at `idx` from the ring and mark the slot free.
    fn ring_evict_slot(&mut self, idx: usize) {
        self.dense_ring[idx] = None;
        self.empty_slots.push(idx);
    }

    /// Recompute the 80th-percentile eviction threshold.
    ///
    /// Samples up to `NUM_EVICTION_SAMPLES` (10) entries evenly from the
    /// dense ring. `None` slots score 0 — they don't bias the threshold
    /// (same as Velox's `element ? element->score(now) : 0`).
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
            .map(|i| {
                let idx = (self.clock_hand + i * step) % n;
                self.dense_ring[idx]
                    .as_ref()
                    .map(|e| e.eviction_score(now))
                    .unwrap_or(0)
            })
            .collect();

        scores.sort_unstable();
        let idx = (scores.len() * EVICTION_PERCENTILE / 100).min(scores.len().saturating_sub(1));
        self.eviction_threshold = scores.get(idx).copied().unwrap_or(0);
    }

    /// Clock-hand sweep following Velox's `CacheShard::evict` exactly.
    ///
    /// - `None` slots skipped with `is_none()` — no `Weak::upgrade()`.
    /// - Calibration fires **mid-sweep** every `n/8` live entries checked,
    ///   so the threshold adapts as cold entries are removed.
    /// - Evicted slots written to `empty_slots` for O(1) reuse on next insert.
    fn evict(&mut self, target_bytes: u64) -> (u64, Vec<(DataCacheKey, Bytes)>) {
        let n = self.dense_ring.len();
        if n == 0 {
            return (0, Vec::new());
        }

        let now = now_ms();
        let mut freed = 0u64;
        let mut counter = 0usize; // raw iterations (Velox's `counter`)
        let mut num_checked = 0usize; // live entries seen (Velox's `numChecked`)
        let mut evicted: Vec<(DataCacheKey, Bytes)> = Vec::new();

        while counter < n {
            let idx = self.clock_hand % n;
            self.clock_hand = self.clock_hand.wrapping_add(1);
            counter += 1;

            if self.dense_ring[idx].is_none() {
                continue; // empty slot — O(1) skip
            }
            let entry = self.dense_ring[idx].as_ref().unwrap().clone();
            num_checked += 1;

            // Mid-sweep recalibration (Velox: `numChecked > entries_.size() / 8`).
            // Threshold adapts as we evict, preventing over-eviction.
            if self.eviction_threshold == 0 || num_checked > n / 8 {
                self.calibrate_threshold();
                num_checked = 0;
            }

            // count == 3: entries map + dense_ring slot + our clone.
            // > 3 means an active external holder — skip.
            if Arc::strong_count(&entry) > 3 {
                continue;
            }

            let size = entry.data_size.load(Ordering::Relaxed);
            if size == 0 {
                continue; // Loading, Failed, or already evicted
            }

            let score = entry.eviction_score(now);
            if score < self.eviction_threshold {
                continue; // too hot
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

/// Per-shard stats snapshot.
pub struct ShardStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub stale_evictions: u64,
}

struct CacheShard {
    inner: Mutex<CacheShardInner>,
    /// Capacity limit for this shard in bytes.
    /// Set at construction as `max_bytes / num_shards`.
    per_shard_limit: u64,
}

impl CacheShard {
    fn new(per_shard_limit: u64) -> Self {
        Self {
            inner: Mutex::new(CacheShardInner::new()),
            per_shard_limit,
        }
    }

    /// Look up or atomically create an entry for `key`.
    ///
    /// `min_size`: if a cached entry's `data_size` is greater than 0 but less
    /// than `min_size`, the entry is stale (a previous smaller request cached
    /// fewer bytes than we now need). The stale entry is evicted and a fresh
    /// miss is returned so the caller can reload the full range — mirrors
    /// Velox's `lookupLocked` stale-entry eviction.
    ///
    /// Evicts BEFORE inserting when `loaded_bytes >= per_shard_limit` —
    /// same as Velox: allocation failure → makeSpace() → evict → retry.
    /// Eviction and insertion are atomic within the same lock hold.
    ///
    /// Returns `(entry, is_new, freed_bytes, evicted_entries)`.
    fn find_or_create(
        &self,
        key: &DataCacheKey,
        min_size: u64,
    ) -> (Arc<CacheEntry>, bool, u64, Vec<(DataCacheKey, Bytes)>) {
        let mut inner = self.inner.lock().unwrap();

        if let Some(entry) = inner.entries.get(key).cloned() {
            // Stale check: entry is loaded but smaller than requested.
            let cached_size = entry.data_size.load(Ordering::Relaxed);
            if cached_size > 0 && cached_size < min_size {
                // Evict stale entry — caller will reload the full range.
                if let Some(e) = inner.entries.remove(key) {
                    let sz = e.data_size.swap(0, Ordering::Relaxed);
                    if sz > 0 {
                        inner.loaded_bytes = inner.loaded_bytes.saturating_sub(sz);
                    }
                    inner.stale_evictions += 1;
                    inner.evictions += 1;
                }
                // Fall through to create a new entry (treated as miss below).
            } else {
                inner.hits += 1;
                return (entry, false, 0, Vec::new());
            }
        }

        // Evict before inserting if this shard is at or over its limit.
        // The new entry is not yet in the map so it won't be a candidate.
        //
        // Evict at least 20% of the shard's capacity rather than just the
        // overage — this batches evictions so we don't call evict() on every
        // single insert that nudges over the limit.  After eviction the shard
        // sits at ~80% capacity, giving a buffer before the next eviction call.
        // Reduces eviction call frequency by ~5x for workloads with many small
        // inserts.
        let batch_evict_bytes = self.per_shard_limit / 5; // 20%
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

    /// Record `size` bytes successfully loaded into this shard.
    fn record_loaded(&self, size: u64) {
        self.inner.lock().unwrap().loaded_bytes += size;
    }

    /// Remove an entry from the shard and return its byte size.
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

    /// Return a stats snapshot.
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

// ─── EvictionSink ────────────────────────────────────────────────────────────

/// Receives evicted cache entries for async persistence to the SSD tier.
///
/// Implementations accumulate entries and trigger batch writes when
/// configurable thresholds are exceeded — mirroring the threshold-based
/// trigger used in the reference design.
///
/// The trait is object-safe and sync so it can be called from within the
/// shard mutex without any async overhead.
pub trait EvictionSink: Send + Sync + std::fmt::Debug {
    /// Called from `maybe_evict` with all entries evicted in one sweep.
    ///
    /// `total_cache_bytes` is the current `total_bytes` counter — used to
    /// compute the ratio-based threshold.
    fn on_evicted(&self, entries: Vec<(DataCacheKey, Bytes)>, total_cache_bytes: u64);
}

// ─── MemoryCache ─────────────────────────────────────────────────────────────

/// Statistics snapshot for a [`MemoryCache`].
#[derive(Debug, Default, Clone)]
pub struct MemoryCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub stale_evictions: u64,
    pub current_bytes: u64,
    pub max_bytes: u64,
}

/// Sharded in-memory cache with -style clock-hand + percentile eviction.
///
/// The cache is logically split into `num_shards` independent shards (default
/// [`DEFAULT_NUM_SHARDS`] = 16, same as kDefaultNumShards).
/// Each shard owns its hash-map and eviction ring and is protected by its own
/// `std::sync::Mutex`, so concurrent tasks hitting different files (or
/// different offsets within the same file) almost never contend.
///
/// Async coordination (waiting for a concurrent load to finish) is done via
/// `tokio::sync::watch` *outside* of the shard mutex, so no tokio worker is
/// blocked while waiting.
pub struct MemoryCache {
    shards: Vec<CacheShard>,
    shard_mask: u64,
    max_bytes: u64,
    total_bytes: AtomicU64,
    /// Optional sink that receives evicted entries for SSD persistence.
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
    /// Create a new cache with the default shard count ([`DEFAULT_NUM_SHARDS`]).
    pub fn new(max_bytes: u64) -> Arc<Self> {
        Self::new_with_shards(max_bytes, DEFAULT_NUM_SHARDS)
    }

    /// Create a new cache with a custom shard count.
    ///
    /// `num_shards` must be a positive power of two (e.g. 4, 8, 16, 32).
    ///
    /// # Panics
    /// Panics if `num_shards` is zero or not a power of two.
    pub fn new_with_shards(max_bytes: u64, num_shards: usize) -> Arc<Self> {
        Self::with_eviction_sink(max_bytes, num_shards, None)
    }

    /// Create a cache that notifies `sink` when entries are evicted, allowing
    /// the SSD tier to persist them asynchronously using threshold-based batching.
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
        // Sweep all shards — each briefly acquires its own lock.
        // Locks are held for microseconds; stats reads are infrequent.
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

    /// Fetch bytes for `key`, calling `loader` on a cache miss.
    ///
    /// `length` is the number of bytes requested. If the cache holds a larger
    /// entry for the same `(file_id, offset)` the returned slice is trimmed to
    /// `length`. If the cached entry is *smaller* than `length` it is treated as
    /// stale — evicted and reloaded via `loader`.
    ///
    /// - **Exclusive path** (`is_new = true`): this task owns the entry and
    ///   calls `load_exclusive` to fetch and populate the cache.
    /// - **Waiter path** (`is_new = false`): another task is already loading;
    ///   `wait_for_entry` subscribes to the watch channel and returns when done.
    ///   If the concurrent load failed, returns an error — the caller should retry.
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

        // Entry exists — wait for the current owner to finish.
        // Three possible states when we check:
        //   Loaded  → bytes returned immediately, no suspend (fast path)
        //   Loading → suspend on watch channel until owner sends Loaded/Failed
        //   Failed  → owner's load failed; surface the error to the caller
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

    /// Exclusive load path: called when this task created the cache entry.
    ///
    /// Runs `loader`, stores the result in the entry, and wakes all waiters.
    /// On failure, signals waiters and removes the entry so the next caller
    /// can retry with a fresh miss.
    async fn load_exclusive(
        &self,
        key: &DataCacheKey,
        length: u64,
        entry: &Arc<CacheEntry>,
        loader: BoxFuture<'_, Result<Bytes>>,
    ) -> Result<Bytes> {
        // misses are already counted in find_or_create (inside the shard lock)
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
                // Transition to shared — wakes all waiting tasks.
                // Must use send_replace() not send(): send() silently drops
                // the value when there are no receivers (the initial _rx was
                // dropped in CacheEntry::new), leaving the channel stuck at
                // Loading and causing waiters to hang forever.
                entry
                    .state_tx
                    .send_replace(LoadState::Loaded(bytes.clone()));
                let shard = &self.shards[shard_idx(key, self.shard_mask)];
                shard.record_loaded(size);
                self.total_bytes.fetch_add(size, Ordering::Relaxed);
                // Return exactly `length` bytes — the loader may have returned
                // a larger buffer (e.g. aligned read). Cached bytes are kept in
                // full so that a future request for a larger range is a hit.
                Ok(bytes.slice(0..length.min(size) as usize))
            }
            Err(e) => {
                // Signal waiters and remove the entry so the next caller
                // gets a fresh miss and can retry with their own loader.
                entry.state_tx.send_replace(LoadState::Failed);
                self.remove_entry(key);
                Err(e)
            }
        }
    }

    /// Waiter path: subscribes to `entry`'s watch channel and waits until the
    /// loading task transitions it to `Loaded` or `Failed`.
    ///
    /// Returns `Some(bytes)` on success, `None` if the load failed (caller
    /// should retry as the new exclusive owner).
    async fn wait_for_entry(
        &self,
        key: &DataCacheKey,
        length: u64,
        entry: &Arc<CacheEntry>,
    ) -> Option<Bytes> {
        let mut rx = entry.state_tx.subscribe();
        loop {
            // Clone so the borrow on `rx` ends before the next `rx.changed()`
            // call (which also needs `&mut rx`).
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
                    // hits are counted in find_or_create (inside the shard lock)
                    return Some(bytes.slice(0..length.min(bytes.len() as u64) as usize));
                }
                LoadState::Failed => {
                    // Loading task failed — caller will retry as new owner.
                    return None;
                }
                LoadState::Loading => {
                    // Still in flight — yield until state changes.
                    if rx.changed().await.is_err() {
                        // Sender dropped unexpectedly; treat as failure.
                        return None;
                    }
                }
            }
        }
    }

    // ── Private helpers ──────────────────────────────────────────────────────

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

/// `MemoryCache` implements `DataCache` directly so it can be used standalone
/// without wrapping in `TieredDataCache`. File path interning is handled
/// internally via a `FileIds` registry owned by this instance.
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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::AtomicUsize;

    fn key(file_id: u64, offset: u64) -> DataCacheKey {
        DataCacheKey { file_id, offset }
    }

    // ── Shard configuration tests (mirrors numShardsDefault / numShardsInvalid) ─

    #[test]
    fn test_default_shard_count() {
        let cache = MemoryCache::new(1024 * 1024);
        assert_eq!(cache.shards.len(), DEFAULT_NUM_SHARDS);
    }

    #[test]
    fn test_custom_shard_count_power_of_two() {
        for &n in &[1usize, 2, 4, 8, 32, 64] {
            let cache = MemoryCache::new_with_shards(1024 * 1024, n);
            assert_eq!(cache.shards.len(), n);
            // shard_mask must be n-1
            assert_eq!(cache.shard_mask, (n as u64) - 1);
        }
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn test_non_power_of_two_shards_panics() {
        MemoryCache::new_with_shards(1024 * 1024, 3);
    }

    #[test]
    #[should_panic(expected = "positive")]
    fn test_zero_shards_panics() {
        MemoryCache::new_with_shards(1024 * 1024, 0);
    }

    #[tokio::test]
    async fn test_single_shard_cache_works() {
        // Degenerate case: 1 shard — all entries in one map, still correct.
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
        // Second call — hit.
        let bytes2 = cache
            .get_or_load(k, 2, Box::pin(async { Ok(Bytes::from_static(b"miss")) }))
            .await
            .unwrap();
        assert_eq!(bytes2, Bytes::from_static(b"hi"));
        assert_eq!(cache.stats().hits, 1);
    }

    #[tokio::test]
    async fn test_basic_hit_and_miss() {
        let cache = MemoryCache::new(10 * 1024 * 1024);
        let k = key(0, 0);

        // First access — miss, loader runs.
        let bytes = cache
            .get_or_load(
                k.clone(),
                5,
                Box::pin(async { Ok(Bytes::from_static(b"hello")) }),
            )
            .await
            .unwrap();
        assert_eq!(bytes, Bytes::from_static(b"hello"));

        // Second access — hit, loader should NOT run.
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

    #[tokio::test]
    async fn test_load_deduplication() {
        let cache = Arc::new(MemoryCache::new(10 * 1024 * 1024));
        let k = key(1, 0);
        let load_count = Arc::new(AtomicUsize::new(0));

        // Launch 8 concurrent requests for the same key.
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
                            // Small delay so other tasks arrive before load completes.
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
        // Only one loader should have run.
        assert_eq!(load_count.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_loader_failure_allows_retry() {
        let cache = Arc::new(MemoryCache::new(10 * 1024 * 1024));
        let k = key(2, 0);

        // First access fails.
        let result = cache
            .get_or_load(
                k.clone(),
                8,
                Box::pin(async { Err(lance_core::Error::io("boom".to_string())) }),
            )
            .await;
        assert!(result.is_err());

        // Second access should succeed — entry was removed after failure.
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

    #[tokio::test]
    async fn test_eviction_under_pressure() {
        // Single shard so all entries concentrate — per-shard eviction fires
        // when the shard exceeds its limit on the next insert.
        let cache = MemoryCache::new_with_shards(4 * 1024 * 1024, 1);
        let chunk = Bytes::from(vec![0u8; 1024 * 1024]);

        for i in 0..8u64 {
            let data = chunk.clone();
            cache
                .get_or_load(
                    key(3, i),
                    chunk.len() as u64,
                    Box::pin(async move { Ok(data) }),
                )
                .await
                .unwrap();
        }

        // Total bytes should be at or below max (eviction may lag slightly due
        // to the amortised nature of the clock sweep).
        let stats = cache.stats();
        assert!(
            stats.current_bytes <= stats.max_bytes * 2,
            "cache grew too large: {} bytes",
            stats.current_bytes
        );
        assert!(stats.evictions > 0, "expected at least one eviction");
    }

    #[tokio::test]
    async fn test_disabled_cache_bypasses() {
        // max_bytes == 0 means disabled; loader is called every time.
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

    // ── -inspired tests ──────────────────────────────────────────────

    /// replace test: fill the cache exactly to capacity,
    /// re-read the same keys, and verify hits occur and eviction fires when
    /// further entries are added beyond capacity.
    #[tokio::test]
    async fn test_replace_hits_and_evictions() {
        let cap = 4 * 1024 * 1024u64;
        let entry_size = 256 * 1024u64; // 256 KiB per entry
        let num_entries = cap / entry_size; // exactly fill the cache (16 entries)
        // Single shard — all entries go to same shard, per-shard eviction fires.
        let cache = MemoryCache::new_with_shards(cap, 1);

        // First pass — fill cache exactly to capacity (all misses).
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

        // Second pass over the SAME keys — should all hit the cache.
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

        // Now push beyond capacity — eviction must fire.
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

    /// staleEntry / double-eviction test: verify that
    /// evictions. A double-decrement bug would make `total_bytes` underflow
    /// causing `maybe_evict` to stop triggering.
    #[tokio::test]
    async fn test_accounting_invariant_under_eviction() {
        let cap = 2 * 1024 * 1024u64;
        let entry_size = 128 * 1024u64;
        // Load 8× capacity to force many evictions.
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
        // Primary invariant: no double-decrement. A u64 underflow wraps to
        // near u64::MAX. Allow up to 2× cap for natural amortisation overshoot
        // (the entry being inserted is pinned on the stack during maybe_evict
        // so it can't be evicted until the insertion returns).
        assert!(
            stats.current_bytes < cap * 2,
            "possible underflow: current_bytes={} (u64::MAX would indicate wrap)",
            stats.current_bytes
        );
        assert!(stats.evictions > 0, "expected evictions to fire");
    }

    /// findExclusiveWithWait + failure test: when a load
    /// fails, *all* concurrent waiters must be unblocked and the entry must be
    /// removed so the next caller can retry successfully.
    #[tokio::test]
    async fn test_concurrent_waiters_see_failure_and_retry() {
        let cache = Arc::new(MemoryCache::new(10 * 1024 * 1024));
        let k = key(5, 0);
        let load_count = Arc::new(AtomicUsize::new(0));

        // The FIRST loader call (whichever task wins find_or_create) fails.
        // Subsequent calls succeed. This is independent of task index.
        let first_call_done = Arc::new(AtomicBool::new(false));
        let barrier = Arc::new(tokio::sync::Barrier::new(8));
        let mut handles = Vec::new();

        for _ in 0..8usize {
            let cache = cache.clone();
            let k = k.clone();
            let lc = load_count.clone();
            let bar = barrier.clone();
            let first = first_call_done.clone();

            handles.push(tokio::spawn(async move {
                bar.wait().await;
                cache
                    .get_or_load(
                        k,
                        2,
                        Box::pin(async move {
                            let call_idx = lc.fetch_add(1, Ordering::SeqCst);
                            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                            if call_idx == 0 {
                                // First loader to run always fails.
                                first.store(true, Ordering::SeqCst);
                                Err(lance_core::Error::io("injected failure".to_string()))
                            } else {
                                Ok(Bytes::from_static(b"ok"))
                            }
                        }),
                    )
                    .await
            }));
        }

        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|h| h.unwrap())
            .collect();

        // Without the retry loop, ALL tasks see an error when the exclusive
        // load fails — the first because its loader returned Err, the rest
        // because wait_for_entry sees Failed and surfaces an error.
        // Callers are responsible for retrying at a higher level.
        assert!(
            first_call_done.load(Ordering::SeqCst),
            "first loader never ran"
        );
        assert!(
            results.iter().all(|r| r.is_err()),
            "all tasks should get an error when the exclusive load fails"
        );

        // A fresh get_or_load after all tasks have returned starts clean —
        // the failed entry was removed, so this succeeds as a new miss.
        let bytes = cache
            .get_or_load(k, 5, Box::pin(async { Ok(Bytes::from_static(b"fresh")) }))
            .await
            .unwrap();
        // No retry happened — entry was removed after failure.
        // This fresh call is a new miss and loads b"fresh".
        assert_eq!(bytes, Bytes::from_static(b"fresh"));
    }

    /// fuzz test: 8 concurrent tasks randomly reading
    /// different (file_id, offset) pairs for 500ms, verifying the cache
    /// never deadlocks, never panics, and never exceeds capacity.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_fuzz_concurrent_access() {
        use rand::Rng;

        let cap = 8 * 1024 * 1024u64;
        let entry_size = 64 * 1024u64; // 64 KiB
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

    /// Stats accounting: hits + misses == total requests (for single-key scenario).
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
        assert_eq!(stats.current_bytes, 1); // "x" = 1 byte
    }

    /// Regression test for the saturating_sub guard in maybe_evict and
    /// remove_entry. Without it, two concurrent evictions subtracting from
    /// total_bytes simultaneously could wrap a u64 to near u64::MAX, making
    /// the cache think it has effectively infinite free space and stop evicting.
    ///
    /// We verify that after heavy concurrent load total_bytes never approaches
    /// u64::MAX (which would indicate a wrap-around).
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_total_bytes_no_underflow_under_concurrent_eviction() {
        // Very small cap forces constant eviction pressure.
        let cap = 512 * 1024u64; // 512 KiB
        let entry_size = 64 * 1024u64; // 64 KiB — 8 entries fill the cache
        let cache = Arc::new(MemoryCache::new(cap));

        // 8 threads each load a distinct stream of keys, all competing for the
        // same tiny cache. Every insert triggers maybe_evict; concurrent calls
        // to fetch_sub on total_bytes used to be able to race and underflow.
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

        // If total_bytes wrapped, it would be close to u64::MAX (≥ 1 TiB).
        // A sane cache under 512 KiB cap should never report anywhere near that.
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

    // ── Missing tests ───────────────────────────────────────────────

    /// outOfCapacity: when all entries are actively loading
    /// (strong_count > 2), eviction must be a graceful no-op — nothing freed,
    /// no panic, no underflow.
    #[tokio::test]
    async fn test_eviction_graceful_when_all_entries_loading() {
        // Tiny cache — 1 entry fits.
        let cap = 128 * 1024u64;
        let entry_size = 128 * 1024u64;
        let cache = Arc::new(MemoryCache::new(cap));

        // Hold the Arc<CacheEntry> alive by keeping the loader suspended,
        // simulating a pinned / still-loading entry.
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let cache_clone = cache.clone();

        let loading = tokio::spawn(async move {
            let _ = cache_clone
                .get_or_load(
                    key(99, 0),
                    entry_size,
                    Box::pin(async move {
                        rx.await.ok(); // suspended — entry is in Loading state
                        Ok(Bytes::from(vec![0u8; entry_size as usize]))
                    }),
                )
                .await
                .unwrap();
        });

        // Give the loader task time to create the entry and suspend.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        // Try to load a second entry — cache is over capacity but the only
        // entry is loading (strong_count > 2). Eviction must not panic or
        // underflow.
        let data = Bytes::from(vec![1u8; entry_size as usize]);
        let _ = cache
            .get_or_load(
                key(99, entry_size),
                entry_size,
                Box::pin(async move { Ok(data) }),
            )
            .await
            .unwrap();

        // Unblock the first loader.
        tx.send(()).ok();
        loading.await.unwrap();

        // No underflow: total_bytes must be a sane value.
        let one_tib = 1u64 << 40;
        assert!(cache.stats().current_bytes < one_tib);
    }

    /// staleEntry: entries with `data_size == 0` (still
    /// loading or previously evicted) are skipped by the clock hand without
    /// touching byte counters — no double-accounting.
    #[tokio::test]
    async fn test_eviction_skips_zero_size_entries() {
        let cap = 256 * 1024u64;
        let entry_size = 128 * 1024u64;
        let cache = MemoryCache::new(cap);

        // Fill cache to capacity.
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

        // Load two more entries — forces eviction. Evicted entries get
        // data_size = 0. If the clock hand sweeps past them again and
        // double-subtracts, total_bytes wraps.
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

        // total_bytes must never wrap (would produce a value >> cap).
        assert!(
            after < cap * 4,
            "possible double-accounting: before={before} after={after}"
        );
        assert!(cache.stats().evictions > 0);
    }

    /// shrinkCache: after loading entries, eviction must
    /// bring total_bytes back to or below capacity when given enough pressure.
    #[tokio::test]
    async fn test_eviction_converges_to_cap() {
        let cap = 1024 * 1024u64; // 1 MiB
        let entry_size = 128 * 1024u64; // 128 KiB — 8 entries fill the cache
        let cache = MemoryCache::new(cap);

        // Load 4× capacity sequentially — forces many eviction rounds.
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

        // After the loading loop, all entry Arcs from is_new=true have dropped.
        // strong_count for each remaining entry is exactly 2 (map + ring) →
        // all are eviction candidates. One more load triggers final convergence.
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

    // ── Tests not ported from (with explanation) ───────────────────
    //
    // evictAccounting: tests interaction between cache eviction and a
    // custom MemoryPool allocator. Lance uses Bytes (Arc<[u8]>) with no
    // custom allocator, so pool-level accounting is not applicable.
    //
    // shrinkWithSsdWrite: Requires SCOPED_TESTVALUE_SET / TestValue hooks to
    // pause SSD writes at a specific code point. Our implementation has no
    // equivalent test-hook infrastructure.
    //
    // ttl: CacheTTLController expires entries based on when files were
    // opened. Lance datasets are immutable and versioned — cached data is
    // valid indefinitely for a given file path, so TTL is not implemented.
    //
    // makeEvictable: Tests explicit num_pins / CachePin management. We
    // deliberately omit CachePin (see TODO comment in evict()), relying on
    // Bytes (Arc<[u8]>) to keep data alive independently.
    //
    // dataRanges: Tests allocation-run API (tiny inline storage vs
    // MmapAllocator pages). Our entries are uniform Bytes (Arc<[u8]>);
    // there is no multi-run layout to verify.
    //
    // pin (partial): The full pin test exercises CachePin move semantics
    // and explicit numPins counting. The equivalent state-machine behaviour
    // (exclusive while loading, shared after, waiters unblocked on failure)
    // is already covered by test_concurrent_waiters_see_failure_and_retry
    // and test_load_deduplication.

    // ── Additional -inspired tests ──────────────────────────────────

    /// findMiss: looking up a key that was never inserted
    /// must return None (loader is called, not bypassed).
    #[tokio::test]
    async fn test_find_miss() {
        let cache = MemoryCache::new(10 * 1024 * 1024);
        let k = key(10, 0);
        let load_count = Arc::new(AtomicUsize::new(0));
        let lc = load_count.clone();

        // First access: miss — loader must be called.
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

    /// findHit: after a miss populates the cache, the next
    /// access must return exactly the same bytes without calling the loader.
    /// Verifies data integrity (byte-for-byte match) — equivalent to
    /// `checkContents(*entry)`.
    #[tokio::test]
    async fn test_find_hit_data_integrity() {
        let cache = MemoryCache::new(10 * 1024 * 1024);
        // Use a recognisable pattern so a stale-copy bug would be detectable.
        let pattern: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let original = Bytes::from(pattern.clone());
        let k = key(11, 0);

        // Miss — populate cache.
        let b1 = cache
            .get_or_load(k.clone(), 4096, Box::pin(async move { Ok(original) }))
            .await
            .unwrap();
        assert_eq!(
            b1.as_ref(),
            pattern.as_slice(),
            "loaded bytes must match pattern"
        );

        // Hit — must return identical bytes without calling loader.
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
        // Both should point to the same underlying allocation.
        assert_eq!(
            b1.as_ptr(),
            b2.as_ptr(),
            "hit must return the cached Arc, not a copy"
        );
        assert_eq!(cache.stats().hits, 1);
    }

    /// cacheStats: verifies that all stat counters are
    /// incremented correctly and reflect the true cache state.
    #[tokio::test]
    async fn test_cache_stats_fields() {
        // 4 entries × 256 KiB = 1 MiB. Use 2 MiB capacity + 1 shard so the
        // single shard's limit (2 MiB) fits all 4 entries with no eviction.
        let cache = MemoryCache::new_with_shards(2 * 1024 * 1024, 1);
        let entry_size = 256 * 1024u64;

        // 4 misses.
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

        // 4 more hits on the same keys (cache holds 2 MiB = 8 × 256 KiB entries).
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
        // current_bytes should reflect the 4 entries still in cache.
        assert_eq!(stats.current_bytes, 4 * entry_size, "current_bytes");
    }

    /// pin state-machine: while a load is in progress
    /// (exclusive / Loading state) concurrent callers must block; after the
    /// transition to Loaded all blocked callers receive the same data.
    /// This complements test_load_deduplication with an explicit timing check.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_exclusive_to_shared_transition() {
        let cache = Arc::new(MemoryCache::new(10 * 1024 * 1024));
        let k = key(13, 0);

        let (loading_tx, loading_rx) = tokio::sync::oneshot::channel::<()>();
        let (done_tx, _done_rx) = tokio::sync::oneshot::channel::<()>();

        // Task A: "exclusive" loader — signals when loading has started, then
        // sleeps to give Task B time to see the Loading state.
        let cache_a = cache.clone();
        let k_a = k.clone();
        let loader_task = tokio::spawn(async move {
            cache_a
                .get_or_load(
                    k_a,
                    9,
                    Box::pin(async move {
                        loading_tx.send(()).ok(); // signal: exclusive load started
                        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                        done_tx.send(()).ok();
                        Ok(Bytes::from_static(b"exclusive"))
                    }),
                )
                .await
                .unwrap()
        });

        // Wait until the loader has started (entry is in Loading state).
        loading_rx.await.ok();

        // Task B: concurrent waiter — should block until A completes.
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
        // Both should point to the same allocation (watch-channel clone).
        assert_eq!(a_result.as_ptr(), b_result.as_ptr());
    }
}
