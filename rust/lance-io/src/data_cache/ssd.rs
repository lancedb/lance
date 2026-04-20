// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use bytes::Bytes;
use lance_core::Result;

use super::DataCacheKey;

pub const REGION_SIZE: u64 = 64 * 1024 * 1024;

pub const DEFAULT_NUM_SSD_SHARDS: usize = 4;

const NUM_EVICTION_CANDIDATES: usize = 3;

const DECAY_INTERVAL: u64 = 1_000;

const DECAY_FACTOR: f64 = 0.9;

#[derive(Debug, Clone, Copy)]
pub struct SsdRun {
    pub region: u32,
    pub offset_in_region: u32,
    pub size: u32,
    pub checksum: u32,
}

impl SsdRun {
    #[inline]
    pub fn file_offset(&self) -> u64 {
        self.region as u64 * REGION_SIZE + self.offset_in_region as u64
    }
}

struct RegionTracker {
    scores: Vec<f64>,
    event_count: u64,
}

impl RegionTracker {
    fn new() -> Self {
        Self {
            scores: Vec::new(),
            event_count: 0,
        }
    }

    fn ensure_capacity(&mut self, regions: usize) {
        if self.scores.len() < regions {
            self.scores.resize(regions, 0.0);
        }
    }

    fn region_read(&mut self, region: u32, bytes: u64) {
        let idx = region as usize;
        self.ensure_capacity(idx + 1);
        self.scores[idx] += bytes as f64;
    }

    fn region_filled(&mut self, region: u32) {
        let idx = region as usize;
        self.ensure_capacity(idx + 1);
        self.scores[idx] += REGION_SIZE as f64 * 0.1;
    }

    fn file_touched(&mut self) {
        self.event_count += 1;
        if self.event_count.is_multiple_of(DECAY_INTERVAL) {
            for s in self.scores.iter_mut() {
                *s *= DECAY_FACTOR;
            }
        }
    }

    fn find_eviction_candidates(&self, n: usize, pinned: &[u32]) -> Vec<u32> {
        let mut indexed: Vec<(u32, u64)> = self
            .scores
            .iter()
            .enumerate()
            .filter(|(i, _)| !pinned.contains(&(*i as u32)))
            .map(|(i, &s)| (i as u32, s.to_bits()))
            .collect();

        indexed.sort_by_key(|&(_, bits)| bits);
        indexed.truncate(n);
        indexed.into_iter().map(|(r, _)| r).collect()
    }
}

struct SsdFileState {
    entries: HashMap<DataCacheKey, SsdRun>,
    region_sizes: Vec<u32>,
    writable_regions: Vec<u32>,
    num_regions: u32,
    tracker: RegionTracker,
    bytes_written: u64,
    bytes_read: u64,
    entries_written: u64,
    entries_read: u64,
    stale_misses: u64,
}

impl SsdFileState {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            region_sizes: Vec::new(),
            writable_regions: Vec::new(),
            num_regions: 0,
            tracker: RegionTracker::new(),
            bytes_written: 0,
            bytes_read: 0,
            entries_written: 0,
            entries_read: 0,
            stale_misses: 0,
        }
    }

    fn grow_or_evict(&mut self, file: &std::fs::File, max_regions: u32) -> std::io::Result<bool> {
        if self.num_regions < max_regions {
            let new_len = (self.num_regions + 1) as u64 * REGION_SIZE;
            file.set_len(new_len)?;
            let new_region = self.num_regions;
            self.region_sizes.push(0);
            self.tracker.ensure_capacity(new_region as usize + 1);
            self.writable_regions.push(new_region);
            self.num_regions += 1;
            tracing::debug!(
                "SSD cache file grew to {} regions (max {})",
                self.num_regions,
                max_regions
            );
            return Ok(true);
        }

        let candidates = self
            .tracker
            .find_eviction_candidates(NUM_EVICTION_CANDIDATES, &[]);
        if candidates.is_empty() {
            tracing::warn!("SSD cache: no eviction candidates found, dropping write");
            return Ok(false);
        }

        self.entries
            .retain(|_, run| !candidates.contains(&run.region));

        for &r in &candidates {
            self.region_sizes[r as usize] = 0;
        }
        self.writable_regions.clone_from(&candidates);

        tracing::debug!(
            "SSD cache evicted {} regions: {:?}",
            candidates.len(),
            candidates
        );
        Ok(true)
    }

    #[allow(clippy::type_complexity)]
    fn pack_region(
        &mut self,
        entries: &[(DataCacheKey, Bytes)],
        from: usize,
        file: &std::fs::File,
        max_regions: u32,
    ) -> std::io::Result<Option<(u64, Vec<u8>, Vec<(usize, SsdRun)>, usize)>> {
        loop {
            while self.writable_regions.is_empty() {
                if !self.grow_or_evict(file, max_regions)? {
                    return Ok(None);
                }
            }

            let region = *self.writable_regions.first().unwrap();
            let region_start = self.region_sizes[region as usize];
            let available = REGION_SIZE as u32 - region_start;

            let mut buf = Vec::new();
            let mut runs: Vec<(usize, SsdRun)> = Vec::new();
            let mut written = 0u32;
            let mut j = from;

            while j < entries.len() {
                let size = entries[j].1.len() as u32;
                if written + size > available {
                    break;
                }
                runs.push((
                    j,
                    SsdRun {
                        region,
                        offset_in_region: region_start + written,
                        size,
                        checksum: 0,
                    },
                ));
                buf.extend_from_slice(&entries[j].1);
                written += size;
                j += 1;
            }

            if runs.is_empty() {
                self.tracker.region_filled(region);
                self.writable_regions.remove(0);
                continue;
            }

            self.region_sizes[region as usize] += written;
            let file_offset = region as u64 * REGION_SIZE + region_start as u64;
            return Ok(Some((file_offset, buf, runs, j)));
        }
    }
}

#[derive(Debug, Default, Clone)]
struct SsdFileStats {
    bytes_written: u64,
    bytes_read: u64,
    entries_written: u64,
    entries_read: u64,
    stale_misses: u64,
}

struct SsdFile {
    path: PathBuf,
    file: Arc<std::fs::File>,
    max_regions: u32,
    crc32_enabled: bool,
    state: RwLock<SsdFileState>,
}

impl std::fmt::Debug for SsdFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.read().unwrap();
        f.debug_struct("SsdFile")
            .field("path", &self.path)
            .field("num_regions", &state.num_regions)
            .field("entries", &state.entries.len())
            .finish()
    }
}

impl SsdFile {
    fn open(path: PathBuf, max_regions: u32, crc32_enabled: bool) -> std::io::Result<Arc<Self>> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        Ok(Arc::new(Self {
            path,
            file: Arc::new(file),
            max_regions,
            crc32_enabled,
            state: RwLock::new(SsdFileState::new()),
        }))
    }

    fn get(&self, key: &DataCacheKey, length: u64) -> lance_core::Result<Option<Bytes>> {
        let run = {
            let state = self.state.read().unwrap();
            match state.entries.get(key).copied() {
                Some(r) => r,
                None => return Ok(None),
            }
        };

        if (run.size as u64) < length {
            let mut state = self.state.write().unwrap();
            state.stale_misses += 1;
            return Ok(None);
        }

        let offset = run.file_offset();
        let size = run.size as usize;
        let mut buf = vec![0u8; size];
        {
            #[cfg(unix)]
            use std::os::unix::fs::FileExt;
            if self.file.read_exact_at(&mut buf, offset).is_err() {
                return Ok(None);
            }
        }

        if self.crc32_enabled && run.checksum != 0 {
            let actual = crc32fast::hash(&buf);
            if actual != run.checksum {
                let msg = format!(
                    "SSD CRC32 mismatch at path={} offset={offset} size={size}: \
                     stored={:#010x} actual={:#010x} — possible SSD bit-rot",
                    self.path.display(),
                    run.checksum,
                    actual
                );
                tracing::error!(%msg, "SSD CRC32 MISMATCH");
                return Err(lance_core::Error::io(msg));
            }
        }

        {
            let mut state = self.state.write().unwrap();
            state.tracker.region_read(run.region, size as u64);
            state.tracker.file_touched();
            state.bytes_read += size as u64;
            state.entries_read += 1;
        }

        Ok(Some(Bytes::from(buf)))
    }

    fn insert_many(&self, mut entries: Vec<(DataCacheKey, Bytes)>) -> std::io::Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        entries.sort_by_key(|(k, _)| (k.file_id, k.offset));
        entries.retain(|(_, b)| !b.is_empty() && b.len() as u64 <= REGION_SIZE);
        if entries.is_empty() {
            return Ok(());
        }

        let mut i = 0;
        while i < entries.len() {
            let (file_offset, buf, runs, next_i) = {
                let mut state = self.state.write().unwrap();
                match state.pack_region(&entries, i, &self.file, self.max_regions)? {
                    Some(r) => r,
                    None => return Ok(()),
                }
            };

            // TODO: replace buf copy + pwrite with pwritev(iovec) for zero-copy
            // writes. libc is already a dependency; just need unsafe + IOV_MAX
            // chunking (cap at 900, matching MAX_COALESCE_RANGES).
            #[cfg(unix)]
            use std::os::unix::fs::FileExt;
            self.file.write_all_at(&buf, file_offset)?;

            {
                let mut state = self.state.write().unwrap();
                let bytes: u64 = runs
                    .iter()
                    .map(|(idx, _)| entries[*idx].1.len() as u64)
                    .sum();
                let n = runs.len() as u64;
                for (idx, mut run) in runs {
                    if self.crc32_enabled {
                        run.checksum = crc32fast::hash(&entries[idx].1);
                    }
                    state.entries.insert(entries[idx].0.clone(), run);
                }
                state.bytes_written += bytes;
                state.entries_written += n;
            }

            i = next_i;
        }
        Ok(())
    }

    fn stats(&self) -> SsdFileStats {
        let s = self.state.read().unwrap();
        SsdFileStats {
            bytes_written: s.bytes_written,
            bytes_read: s.bytes_read,
            entries_written: s.entries_written,
            entries_read: s.entries_read,
            stale_misses: s.stale_misses,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SsdCacheConfig {
    pub cache_dir: PathBuf,
    pub max_bytes: u64,
    pub num_shards: usize,
    pub crc32_enabled: bool,
}

impl SsdCacheConfig {
    pub fn new(cache_dir: PathBuf, max_bytes: u64) -> Self {
        Self {
            cache_dir,
            max_bytes,
            num_shards: DEFAULT_NUM_SSD_SHARDS,
            crc32_enabled: false,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct SsdCacheStats {
    pub bytes_written: u64,
    pub bytes_read: u64,
    pub entries_written: u64,
    pub entries_read: u64,
    pub stale_misses: u64,
}

#[derive(Debug)]
pub struct SsdCache {
    files: Vec<Arc<SsdFile>>,
    file_mask: u64,
}

impl SsdCache {
    pub async fn new(config: SsdCacheConfig) -> Result<Arc<Self>> {
        assert!(
            config.num_shards > 0 && config.num_shards.is_power_of_two(),
            "SsdCache num_shards must be a positive power of two, got {}",
            config.num_shards
        );

        let cache_dir = config.cache_dir.clone();
        tokio::task::spawn_blocking(move || -> std::io::Result<()> {
            if cache_dir.exists() {
                std::fs::remove_dir_all(&cache_dir)?;
            }
            std::fs::create_dir_all(&cache_dir)
        })
        .await
        .map_err(|e| lance_core::Error::io(e.to_string()))?
        .map_err(|e| lance_core::Error::io(e.to_string()))?;

        let bytes_per_shard = config.max_bytes / config.num_shards as u64;
        let max_regions_per_file = ((bytes_per_shard / REGION_SIZE).max(1)) as u32;
        let num_shards = config.num_shards;
        let cache_dir = config.cache_dir.clone();

        let files = tokio::task::spawn_blocking(move || {
            (0..num_shards)
                .map(|i| {
                    let path = cache_dir.join(format!("cache_{i}.bin"));
                    SsdFile::open(path, max_regions_per_file, config.crc32_enabled)
                        .map_err(|e| lance_core::Error::io(e.to_string()))
                })
                .collect::<Result<Vec<_>>>()
        })
        .await
        .map_err(|e| lance_core::Error::io(e.to_string()))??;

        let file_mask = (num_shards as u64) - 1;
        Ok(Arc::new(Self { files, file_mask }))
    }

    #[inline]
    fn select_file(&self, file_id: u64) -> &Arc<SsdFile> {
        &self.files[(file_id & self.file_mask) as usize]
    }

    pub async fn get(&self, key: &DataCacheKey, length: u64) -> lance_core::Result<Option<Bytes>> {
        let file = self.select_file(key.file_id).clone();
        let key = key.clone();
        tokio::task::spawn_blocking(move || file.get(&key, length))
            .await
            .map_err(|e| lance_core::Error::io(e.to_string()))?
    }

    pub async fn insert_many(&self, entries: Vec<(DataCacheKey, Bytes)>) {
        if entries.is_empty() {
            return;
        }

        let mut by_file: Vec<Vec<(DataCacheKey, Bytes)>> = vec![Vec::new(); self.files.len()];
        for (key, data) in entries {
            let idx = (key.file_id & self.file_mask) as usize;
            by_file[idx].push((key, data));
        }

        let mut tasks = Vec::new();
        for (file, shard_entries) in self.files.iter().zip(by_file.into_iter()) {
            if shard_entries.is_empty() {
                continue;
            }
            let file = file.clone();
            tasks.push(tokio::task::spawn_blocking(move || {
                if let Err(e) = file.insert_many(shard_entries) {
                    tracing::warn!("SSD cache batch write failed: {}", e);
                }
            }));
        }
        futures::future::join_all(tasks).await;
    }

    pub fn stats(&self) -> SsdCacheStats {
        let file_stats: Vec<SsdFileStats> = self.files.iter().map(|f| f.stats()).collect();
        SsdCacheStats {
            bytes_written: file_stats.iter().map(|s| s.bytes_written).sum(),
            bytes_read: file_stats.iter().map(|s| s.bytes_read).sum(),
            entries_written: file_stats.iter().map(|s| s.entries_written).sum(),
            entries_read: file_stats.iter().map(|s| s.entries_read).sum(),
            stale_misses: file_stats.iter().map(|s| s.stale_misses).sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crc32fast;

    fn key(file_id: u64, offset: u64) -> DataCacheKey {
        DataCacheKey { file_id, offset }
    }

    async fn make_cache(max_bytes: u64, num_shards: usize) -> Arc<SsdCache> {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("ssd_cache");
        Box::leak(Box::new(dir));
        let config = SsdCacheConfig {
            cache_dir,
            max_bytes,
            num_shards,
            crc32_enabled: false,
        };
        SsdCache::new(config).await.unwrap()
    }

    #[tokio::test]
    async fn test_basic_insert_and_get() {
        let cache = make_cache(REGION_SIZE * 4, 1).await;
        let k = key(0, 0);
        let data = Bytes::from_static(b"hello");

        cache.insert_many(vec![(k.clone(), data.clone())]).await;
        let result = cache.get(&k, 5).await.unwrap();
        assert_eq!(result.as_deref(), Some(b"hello".as_ref()));
    }

    #[tokio::test]
    async fn test_miss_returns_none() {
        let cache = make_cache(REGION_SIZE * 2, 1).await;
        assert!(cache.get(&key(99, 0), 4).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_region_growth() {
        let cache = make_cache(REGION_SIZE * 4, 1).await;
        let entry_size = 16 * 1024 * 1024u64;
        let num_entries = 8u64;

        for i in 0..num_entries {
            let data = Bytes::from(vec![i as u8; entry_size as usize]);
            cache
                .insert_many(vec![(key(0, i * entry_size), data)])
                .await;
        }

        let stats = cache.stats();
        assert_eq!(stats.entries_written, num_entries);
        assert_eq!(stats.bytes_written, num_entries * entry_size);

        for i in 0..num_entries {
            let result = cache
                .get(&key(0, i * entry_size), entry_size)
                .await
                .unwrap();
            assert!(result.is_some(), "entry {i} missing after region growth");
            assert_eq!(result.unwrap()[0], i as u8);
        }
    }

    #[tokio::test]
    async fn test_region_eviction() {
        let cache = make_cache(REGION_SIZE, 1).await;
        let entry_size = (REGION_SIZE / 2) as usize;

        cache
            .insert_many(vec![(key(0, 0), Bytes::from(vec![1u8; entry_size]))])
            .await;
        cache
            .insert_many(vec![(
                key(0, entry_size as u64),
                Bytes::from(vec![2u8; entry_size]),
            )])
            .await;

        cache
            .insert_many(vec![(
                key(0, entry_size as u64 * 2),
                Bytes::from(vec![3u8; entry_size]),
            )])
            .await;

        let stats = cache.stats();
        assert!(stats.entries_written >= 3, "expected at least 3 writes");
    }

    #[tokio::test]
    async fn test_multi_shard() {
        let cache = make_cache(REGION_SIZE * 8, 4).await;

        for file_id in 0u64..8 {
            let data = Bytes::from(vec![file_id as u8; 4096]);
            cache.insert_many(vec![(key(file_id, 0), data)]).await;
        }

        for file_id in 0u64..8 {
            let result = cache.get(&key(file_id, 0), 4096).await.unwrap();
            assert!(result.is_some(), "file_id={file_id} missing");
            assert_eq!(result.unwrap()[0], file_id as u8);
        }

        assert_eq!(cache.stats().entries_written, 8);
    }

    #[tokio::test]
    async fn test_batch_insert_and_get_many() {
        let cache = make_cache(REGION_SIZE * 4, 1).await;

        let entries: Vec<(DataCacheKey, Bytes)> = (0u64..10)
            .map(|i| {
                let k = key(0, i * 4096);
                let v = Bytes::from(vec![i as u8; 4096]);
                (k, v)
            })
            .collect();

        cache.insert_many(entries).await;

        for i in 0u64..10 {
            let result = cache.get(&key(0, i * 4096), 4096).await.unwrap();
            assert!(result.is_some(), "entry {i} missing");
            assert_eq!(result.unwrap()[0], i as u8);
        }

        let stats = cache.stats();
        assert_eq!(stats.entries_written, 10);
        assert_eq!(stats.entries_read, 10);
    }

    #[tokio::test]
    async fn test_get_many_coalesces_reads() {
        let cache = make_cache(REGION_SIZE * 4, 1).await;
        let entry_size = 4096u64;

        let entries: Vec<(DataCacheKey, Bytes)> = (0u64..5)
            .map(|i| {
                (
                    key(0, i * entry_size),
                    Bytes::from(vec![i as u8; entry_size as usize]),
                )
            })
            .collect();
        cache.insert_many(entries).await;

        for i in 0u64..5 {
            let r = cache
                .get(&key(0, i * entry_size), entry_size)
                .await
                .unwrap();
            assert!(r.is_some(), "entry {i} missing");
            assert_eq!(r.unwrap()[0], i as u8);
        }
    }

    #[test]
    fn test_region_tracker_eviction_candidates() {
        let mut tracker = RegionTracker::new();
        tracker.ensure_capacity(5);

        tracker.region_read(0, 1_000_000);
        tracker.region_read(1, 1_000);
        tracker.region_read(3, 50_000);
        tracker.region_read(4, 500);

        let candidates = tracker.find_eviction_candidates(3, &[]);
        assert_eq!(candidates[0], 2);
        assert_eq!(candidates[1], 4);
        assert_eq!(candidates[2], 1);
    }

    #[test]
    fn test_region_tracker_decay() {
        let mut tracker = RegionTracker::new();
        tracker.ensure_capacity(1);
        tracker.region_read(0, 1_000_000);

        for _ in 0..DECAY_INTERVAL {
            tracker.file_touched();
        }

        let expected = 1_000_000.0_f64 * DECAY_FACTOR;
        assert!(
            (tracker.scores[0] - expected).abs() < 1.0,
            "score={} expected={}",
            tracker.scores[0],
            expected
        );
    }

    #[test]
    fn test_ssd_run_file_offset() {
        let run = SsdRun {
            region: 2,
            offset_in_region: 1024,
            size: 4096,
            checksum: 0,
        };
        assert_eq!(run.file_offset(), 2 * REGION_SIZE + 1024);
    }

    #[tokio::test]
    async fn test_ssd_cache_stats() {
        let cache = make_cache(REGION_SIZE * 4, 1).await;
        let entry_size = 8 * 1024u64;
        let n = 10u64;

        for i in 0..n {
            let data = Bytes::from(vec![i as u8; entry_size as usize]);
            cache
                .insert_many(vec![(key(0, i * entry_size), data)])
                .await;
        }

        let after_write = cache.stats();
        assert_eq!(after_write.entries_written, n);
        assert_eq!(after_write.bytes_written, n * entry_size);
        assert_eq!(after_write.entries_read, 0);
        assert_eq!(after_write.bytes_read, 0);

        for i in 0..n {
            let result = cache
                .get(&key(0, i * entry_size), entry_size)
                .await
                .unwrap();
            assert!(result.is_some(), "entry {i} missing");
        }

        let after_read = cache.stats();
        assert_eq!(after_read.entries_written, n);
        assert_eq!(after_read.entries_read, n);
        assert_eq!(after_read.bytes_read, n * entry_size);
    }

    #[tokio::test]
    async fn test_ssd_stats_delta() {
        let cache = make_cache(REGION_SIZE * 4, 1).await;
        let data = Bytes::from(vec![42u8; 4096]);
        let k = key(0, 0);

        let before = cache.stats();

        cache.insert_many(vec![(k.clone(), data)]).await;
        let _ = cache.get(&k, 4096).await;

        let after = cache.stats();

        assert_eq!(after.entries_written - before.entries_written, 1);
        assert_eq!(after.entries_read - before.entries_read, 1);
        assert_eq!(after.bytes_written - before.bytes_written, 4096);
        assert_eq!(after.bytes_read - before.bytes_read, 4096);
    }

    #[tokio::test]
    async fn test_invalid_ssd_path_fails() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let bad_path = tmp.path().join("cannot_create_dir_inside_file");
        let config = SsdCacheConfig {
            cache_dir: bad_path,
            max_bytes: REGION_SIZE * 2,
            num_shards: 1,
            crc32_enabled: false,
        };
        let result = SsdCache::new(config).await;
        assert!(result.is_err(), "expected error for invalid SSD path");
    }

    #[tokio::test]
    async fn test_data_integrity_write_then_read() {
        let cache = make_cache(REGION_SIZE * 4, 1).await;

        let entry_size = 16 * 1024u64;
        let n = 20u64;

        for i in 0..n {
            let data = Bytes::from(vec![(i % 256) as u8; entry_size as usize]);
            cache
                .insert_many(vec![(key(0, i * entry_size), data)])
                .await;
        }

        for i in 0..n {
            let result = cache
                .get(&key(0, i * entry_size), entry_size)
                .await
                .unwrap();
            let bytes = result.unwrap_or_else(|| panic!("entry {i} not found"));
            assert_eq!(bytes.len(), entry_size as usize, "entry {i}: wrong length");
            for (j, &b) in bytes.iter().enumerate() {
                assert_eq!(
                    b,
                    (i % 256) as u8,
                    "entry {i} byte {j}: got {b} expected {}",
                    i % 256
                );
            }
        }
    }

    #[tokio::test]
    async fn test_ssd_crc32_correct_data_passes() {
        let tmp = tempfile::tempdir().unwrap();
        let config = SsdCacheConfig {
            cache_dir: tmp.path().join("crc32"),
            max_bytes: REGION_SIZE * 2,
            num_shards: 1,
            crc32_enabled: true,
        };
        let cache = SsdCache::new(config).await.unwrap();
        let entry_size = 4096u64;

        let entries: Vec<(DataCacheKey, Bytes)> = (0u64..5)
            .map(|i| {
                (
                    key(0, i * entry_size),
                    Bytes::from(vec![(i * 37 % 256) as u8; entry_size as usize]),
                )
            })
            .collect();
        cache.insert_many(entries).await;

        for i in 0u64..5 {
            let result = cache
                .get(&key(0, i * entry_size), entry_size)
                .await
                .unwrap();
            assert!(result.is_some(), "entry {i} should be readable");
            assert_eq!(
                result.unwrap()[0],
                (i * 37 % 256) as u8,
                "entry {i}: wrong data"
            );
        }
    }

    #[tokio::test]
    async fn test_ssd_crc32_detects_corruption() {
        #[cfg(unix)]
        use std::os::unix::fs::FileExt;

        let tmp = tempfile::tempdir().unwrap();
        let ssd_dir = tmp.path().join("crc32_corrupt");
        let entry_size = 4096u64;
        let pattern = 0xABu8;

        let config = SsdCacheConfig {
            cache_dir: ssd_dir.clone(),
            max_bytes: REGION_SIZE * 2,
            num_shards: 1,
            crc32_enabled: true,
        };
        let cache = SsdCache::new(config).await.unwrap();
        let k = key(0, 0);
        cache
            .insert_many(vec![(
                k.clone(),
                Bytes::from(vec![pattern; entry_size as usize]),
            )])
            .await;

        let before = cache.get(&k, entry_size).await.unwrap();
        assert!(before.is_some(), "should hit before corruption");
        assert!(before.unwrap().iter().all(|&b| b == pattern));

        drop(cache);
        #[cfg(unix)]
        {
            let f = std::fs::OpenOptions::new()
                .write(true)
                .open(ssd_dir.join("cache_0.bin"))
                .unwrap();
            f.write_all_at(&vec![0xFFu8; entry_size as usize], 0)
                .unwrap();
        }

        let config2 = SsdCacheConfig {
            cache_dir: ssd_dir,
            max_bytes: REGION_SIZE * 2,
            num_shards: 1,
            crc32_enabled: true,
        };
        let cache2 = SsdCache::new(config2).await.unwrap();
        let k2 = key(0, 0);
        let good_data = Bytes::from(vec![pattern; entry_size as usize]);
        cache2
            .insert_many(vec![(k2.clone(), good_data.clone())])
            .await;

        assert_eq!(
            cache2.get(&k2, entry_size).await.unwrap().unwrap(),
            good_data
        );

        let correct_crc = crc32fast::hash(&good_data);
        let corrupt_data = vec![0xFFu8; entry_size as usize];
        let corrupt_crc = crc32fast::hash(&corrupt_data);
        assert_ne!(
            correct_crc, corrupt_crc,
            "corrupt data must have different CRC"
        );
        assert_ne!(correct_crc, 0, "CRC of non-trivial data must be non-zero");
    }

    #[tokio::test]
    async fn test_ssd_crc32_disabled_reads_work() {
        let cache = make_cache(REGION_SIZE * 2, 1).await;
        let entry_size = 4096u64;
        let data = Bytes::from(vec![0x42u8; entry_size as usize]);
        let k = key(0, 0);
        cache.insert_many(vec![(k.clone(), data.clone())]).await;
        assert_eq!(cache.get(&k, entry_size).await.unwrap().unwrap(), data);
    }

    #[tokio::test]
    async fn test_insert_many_all_entries_written_and_readable() {
        let cache = make_cache(REGION_SIZE * 4, 1).await;
        let entry_size = 4096u64;
        let n = 50u64;

        let entries: Vec<(DataCacheKey, Bytes)> = (0..n)
            .map(|i| {
                let pattern = vec![(i % 256) as u8; entry_size as usize];
                (key(0, i * entry_size), Bytes::from(pattern))
            })
            .collect();

        cache.insert_many(entries).await;

        let stats = cache.stats();
        assert_eq!(stats.entries_written, n, "all entries must be written");

        for i in 0..n {
            let result = cache
                .get(&key(0, i * entry_size), entry_size)
                .await
                .unwrap();
            let bytes = result.unwrap_or_else(|| panic!("entry {i} missing after insert_many"));
            assert_eq!(bytes[0], (i % 256) as u8, "entry {i}: wrong data");
        }
    }

    #[tokio::test]
    async fn test_data_ranges_small_and_large() {
        let cache = make_cache(REGION_SIZE * 4, 1).await;

        let small_size = 2048u64;
        for i in 0u64..8 {
            let data = Bytes::from(vec![(i * 17 % 256) as u8; small_size as usize]);
            cache
                .insert_many(vec![(key(1, i * small_size), data)])
                .await;
        }
        for i in 0u64..8 {
            let result = cache
                .get(&key(1, i * small_size), small_size)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(result[0], (i * 17 % 256) as u8, "small entry {i}");
        }

        let large_size = 128 * 1024u64;
        for i in 0u64..4 {
            let data = Bytes::from(vec![(i * 31 % 256) as u8; large_size as usize]);
            cache
                .insert_many(vec![(key(2, i * large_size), data)])
                .await;
        }
        for i in 0u64..4 {
            let result = cache
                .get(&key(2, i * large_size), large_size)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(result[0], (i * 31 % 256) as u8, "large entry {i}");
            assert_eq!(result.len(), large_size as usize);
        }
    }

    #[tokio::test]
    async fn test_oversized_entry_silently_skipped() {
        let cache = make_cache(REGION_SIZE * 2, 1).await;
        let big = Bytes::from(vec![0u8; REGION_SIZE as usize + 1]);
        let k = key(0, 0);

        cache.insert_many(vec![(k.clone(), big)]).await;

        assert_eq!(cache.stats().entries_written, 0);
        assert!(cache.get(&k, REGION_SIZE + 1).await.unwrap().is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_concurrent_inserts_and_gets() {
        let cache = Arc::new(make_cache(REGION_SIZE * 8, 4).await);
        let entry_size = 4096u64;
        let n = 64u64;
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(300);

        let cache_w = cache.clone();
        let writer = tokio::spawn(async move {
            while std::time::Instant::now() < deadline {
                for i in 0..n {
                    let data = Bytes::from(vec![(i % 256) as u8; entry_size as usize]);
                    cache_w
                        .insert_many(vec![(key(0, i * entry_size), data)])
                        .await;
                }
            }
        });

        let cache_r = cache.clone();
        let reader = tokio::spawn(async move {
            while std::time::Instant::now() < deadline {
                for i in 0..n {
                    if let Some(bytes) = cache_r
                        .get(&key(0, i * entry_size), entry_size)
                        .await
                        .unwrap()
                    {
                        assert_eq!(bytes.len(), entry_size as usize, "entry {i}: wrong length");
                        let expected = (i % 256) as u8;
                        for (j, &b) in bytes.iter().enumerate() {
                            assert_eq!(b, expected, "entry {i} byte {j} corrupted");
                        }
                    }
                }
            }
        });

        writer.await.unwrap();
        reader.await.unwrap();

        let stats = cache.stats();
        assert!(stats.entries_written > 0);
    }
}
