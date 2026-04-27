// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[cfg(unix)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::{collections::HashMap, path::PathBuf, sync::Arc};

use bytes::Bytes;
use futures::future::BoxFuture;
use object_store::path::Path;

use lance_core::Result;

pub mod file_ids;
pub mod memory;
#[cfg(unix)]
pub mod ssd;

use file_ids::FileIds;
use memory::MemoryCache;
#[cfg(unix)]
use ssd::{SsdCache, SsdCacheConfig};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DataCacheKey {
    pub file_id: u64,
    pub offset: u64,
}

pub const DEFAULT_NUM_SSD_SHARDS: usize = 4;

#[derive(Debug, Clone)]
pub struct DataCacheConfig {
    pub max_memory_bytes: u64,

    pub num_shards: usize,

    pub ssd_enabled: bool,

    pub ssd_cache_dir: Option<PathBuf>,

    pub ssd_max_bytes: u64,

    pub ssd_num_shards: usize,

    pub verify: bool,

    pub ssd_crc32_enabled: bool,
}

impl DataCacheConfig {
    pub const KEY_ENABLED: &'static str = "data_cache_enabled";
    pub const KEY_MEMORY_BYTES: &'static str = "data_cache_memory_bytes";
    pub const KEY_SSD_ENABLED: &'static str = "data_cache_ssd_enabled";
    pub const KEY_SSD_DIR: &'static str = "data_cache_ssd_dir";
    pub const KEY_SSD_BYTES: &'static str = "data_cache_ssd_bytes";

    pub const KEY_MEMORY_SHARDS: &'static str = "data_cache_memory_shards";
    pub const KEY_SSD_SHARDS: &'static str = "data_cache_ssd_shards";
    pub const KEY_VERIFY: &'static str = "data_cache_check_rtt_enabled";
    pub const KEY_SSD_CRC32: &'static str = "data_cache_ssd_crc32_enabled";

    pub fn from_storage_options(opts: &HashMap<String, String>) -> Option<Self> {
        use lance_core::utils::parse::str_is_truthy;

        let enabled = opts
            .get(Self::KEY_ENABLED)
            .map(|v| str_is_truthy(v.trim()))
            .unwrap_or(false);

        if !enabled {
            return None;
        }

        let max_memory_bytes = opts
            .get(Self::KEY_MEMORY_BYTES)
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(256 * 1024 * 1024);

        let ssd_enabled = opts
            .get(Self::KEY_SSD_ENABLED)
            .map(|v| str_is_truthy(v.trim()))
            .unwrap_or(false);

        let ssd_cache_dir = opts.get(Self::KEY_SSD_DIR).map(PathBuf::from);

        let ssd_max_bytes = opts
            .get(Self::KEY_SSD_BYTES)
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);

        let num_shards = opts
            .get(Self::KEY_MEMORY_SHARDS)
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(memory::DEFAULT_NUM_SHARDS);

        let ssd_num_shards = opts
            .get(Self::KEY_SSD_SHARDS)
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_SSD_SHARDS);

        let verify = opts
            .get(Self::KEY_VERIFY)
            .map(|v| str_is_truthy(v.trim()))
            .unwrap_or(false);

        let ssd_crc32_enabled = opts
            .get(Self::KEY_SSD_CRC32)
            .map(|v| str_is_truthy(v.trim()))
            .unwrap_or(false);

        Some(Self {
            max_memory_bytes,
            num_shards,
            ssd_enabled,
            ssd_cache_dir,
            ssd_max_bytes,
            ssd_num_shards,
            verify,
            ssd_crc32_enabled,
        })
    }
}

#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub memory_hits: u64,
    pub memory_misses: u64,
    pub memory_evictions: u64,
    pub memory_current_bytes: u64,
    pub memory_stale_evictions: u64,
    pub ssd_hits: u64,
    pub ssd_bytes_written: u64,
    pub ssd_stale_misses: u64,
}

pub trait DataCache: Send + Sync + std::fmt::Debug {
    fn intern_file(&self, path: &Path) -> u64;

    fn get_or_load_by_id<'a>(
        &'a self,
        file_id: u64,
        offset: u64,
        length: u64,
        loader: BoxFuture<'a, Result<Bytes>>,
    ) -> BoxFuture<'a, Result<Bytes>>;

    fn get_or_load<'a>(
        &'a self,
        path: &'a Path,
        offset: u64,
        length: u64,
        loader: BoxFuture<'a, Result<Bytes>>,
    ) -> BoxFuture<'a, Result<Bytes>> {
        let file_id = self.intern_file(path);
        self.get_or_load_by_id(file_id, offset, length, loader)
    }

    fn cache_stats(&self) -> CacheStats;
}

#[derive(Debug)]
pub struct NoopDataCache;

impl DataCache for NoopDataCache {
    fn intern_file(&self, _path: &Path) -> u64 {
        0
    }

    fn get_or_load_by_id<'a>(
        &'a self,
        _file_id: u64,
        _offset: u64,
        _length: u64,
        loader: BoxFuture<'a, Result<Bytes>>,
    ) -> BoxFuture<'a, Result<Bytes>> {
        loader
    }

    fn cache_stats(&self) -> CacheStats {
        CacheStats::default()
    }
}

#[cfg(unix)]
#[derive(Debug)]
struct SsdWriter {
    ssd: Arc<SsdCache>,
    write_in_progress: Arc<AtomicBool>,
}

#[cfg(unix)]
const MAX_WRITE_RATIO: usize = 70;

/// Minimum number of L1 hits an entry must have accumulated before it is
/// eligible to be written to the SSD tier on eviction.  Entries below this
/// threshold are one-hit wonders and are dropped rather than polluting SSD.
#[cfg(unix)]
const SSD_MIN_HITS: u32 = 3;

#[cfg(unix)]
impl SsdWriter {
    fn new(ssd: Arc<SsdCache>) -> Arc<Self> {
        Arc::new(Self {
            ssd,
            write_in_progress: Arc::new(AtomicBool::new(false)),
        })
    }

    pub async fn wait_for_write(&self) {
        while self.write_in_progress.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    }
}

#[cfg(unix)]
impl memory::EvictionSink for SsdWriter {
    fn on_evicted(&self, entries: Vec<(DataCacheKey, Bytes, u32)>, _total_cache_bytes: u64) {
        if self
            .write_in_progress
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return;
        }

        // Only admit entries that have been accessed at least SSD_MIN_HITS times.
        // Entries below the threshold are one-hit wonders and are dropped to
        // prevent scan-once data from polluting the SSD tier.
        let admitted: Vec<(DataCacheKey, Bytes)> = entries
            .into_iter()
            .filter(|(_, _, num_uses)| *num_uses >= SSD_MIN_HITS)
            .map(|(key, bytes, _)| (key, bytes))
            .collect();

        let max = (admitted.len() * MAX_WRITE_RATIO / 100).max(1);
        let batch: Vec<_> = admitted.into_iter().take(max).collect();

        if batch.is_empty() {
            self.write_in_progress.store(false, Ordering::Release);
            return;
        }

        let ssd = self.ssd.clone();
        let flag = self.write_in_progress.clone();
        tokio::spawn(async move {
            ssd.insert_many(batch).await;
            flag.store(false, Ordering::Release);
        });
    }
}

#[derive(Debug)]
pub struct TieredDataCache {
    memory: Arc<MemoryCache>,
    #[cfg(unix)]
    pub ssd: Option<Arc<SsdCache>>,
    #[cfg(unix)]
    ssd_writer: Option<Arc<SsdWriter>>,
    file_ids: Arc<FileIds>,
}

impl TieredDataCache {
    pub async fn new(config: &DataCacheConfig) -> Result<Arc<Self>> {
        #[cfg(not(unix))]
        if config.ssd_enabled {
            return Err(lance_core::Error::invalid_input(
                "data_cache_ssd_enabled=true is not supported on Windows — \
                 SSD cache requires Unix pread/pwrite semantics",
            ));
        }

        #[cfg(unix)]
        let ssd = if config.ssd_enabled {
            match &config.ssd_cache_dir {
                Some(dir) => {
                    let ssd_config = SsdCacheConfig {
                        cache_dir: dir.clone(),
                        max_bytes: config.ssd_max_bytes,
                        num_shards: config.ssd_num_shards,
                        crc32_enabled: config.ssd_crc32_enabled,
                    };
                    Some(SsdCache::new(ssd_config).await?)
                }
                None => {
                    tracing::warn!(
                        "data_cache_ssd_enabled=true but data_cache_ssd_dir is not set \
                         — SSD tier disabled"
                    );
                    None
                }
            }
        } else {
            None
        };

        #[cfg(unix)]
        let ssd_writer: Option<Arc<SsdWriter>> =
            ssd.as_ref().map(|ssd_arc| SsdWriter::new(ssd_arc.clone()));

        #[cfg(unix)]
        let eviction_sink: Option<Arc<dyn memory::EvictionSink>> = ssd_writer
            .clone()
            .map(|w| w as Arc<dyn memory::EvictionSink>);
        #[cfg(not(unix))]
        let eviction_sink: Option<Arc<dyn memory::EvictionSink>> = None;

        let memory = memory::MemoryCache::with_eviction_sink(
            config.max_memory_bytes,
            config.num_shards,
            eviction_sink,
        );

        Ok(Arc::new(Self {
            memory,
            #[cfg(unix)]
            ssd,
            #[cfg(unix)]
            ssd_writer,
            file_ids: Arc::new(FileIds::new()),
        }))
    }

    pub fn memory_stats(&self) -> memory::MemoryCacheStats {
        self.memory.stats()
    }

    pub fn start_stats_logger(self: &Arc<Self>, interval_secs: u64) -> tokio::task::JoinHandle<()> {
        let cache = Arc::clone(self);
        let interval = std::time::Duration::from_secs(interval_secs);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                let s = cache.cache_stats();
                let hit_rate = if s.memory_hits + s.memory_misses > 0 {
                    s.memory_hits as f64 / (s.memory_hits + s.memory_misses) as f64 * 100.0
                } else {
                    0.0
                };
                tracing::info!(
                    memory_hits = s.memory_hits,
                    memory_misses = s.memory_misses,
                    memory_evictions = s.memory_evictions,
                    memory_current_bytes = s.memory_current_bytes,
                    memory_hit_rate_pct = hit_rate,
                    ssd_hits = s.ssd_hits,
                    ssd_bytes_written = s.ssd_bytes_written,
                    "cache stats"
                );
            }
        })
    }

    #[cfg(unix)]
    pub async fn flush_ssd(&self) {
        if let Some(writer) = &self.ssd_writer {
            writer.wait_for_write().await;
        }
    }
}

impl DataCache for TieredDataCache {
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

        #[cfg(unix)]
        let effective_loader: BoxFuture<'a, Result<Bytes>> = if let Some(ssd) = &self.ssd {
            let key_for_ssd = key.clone();
            Box::pin(async move {
                if let Some(bytes) = ssd.get(&key_for_ssd, length).await? {
                    return Ok(bytes);
                }
                loader.await
            })
        } else {
            loader
        };
        #[cfg(not(unix))]
        let effective_loader = loader;

        Box::pin(self.memory.get_or_load(key, length, effective_loader))
    }

    fn cache_stats(&self) -> CacheStats {
        let mem = self.memory.stats();
        #[cfg(unix)]
        let (ssd_hits, ssd_bytes_written, ssd_stale_misses) = {
            let ssd = self.ssd.as_ref().map(|s| s.stats()).unwrap_or_default();
            (ssd.entries_read, ssd.bytes_written, ssd.stale_misses)
        };
        #[cfg(not(unix))]
        let (ssd_hits, ssd_bytes_written, ssd_stale_misses) = (0u64, 0u64, 0u64);

        CacheStats {
            memory_hits: mem.hits,
            memory_misses: mem.misses,
            memory_evictions: mem.evictions,
            memory_current_bytes: mem.current_bytes,
            memory_stale_evictions: mem.stale_evictions,
            ssd_hits,
            ssd_bytes_written,
            ssd_stale_misses,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_absent_when_no_keys() {
        assert!(DataCacheConfig::from_storage_options(&HashMap::new()).is_none());
        let opts = HashMap::from([(
            DataCacheConfig::KEY_MEMORY_BYTES.to_string(),
            "1000000".to_string(),
        )]);
        assert!(DataCacheConfig::from_storage_options(&opts).is_none());
    }

    #[test]
    fn test_config_memory_only() {
        let opts = HashMap::from([
            (DataCacheConfig::KEY_ENABLED.to_string(), "true".to_string()),
            (
                DataCacheConfig::KEY_MEMORY_BYTES.to_string(),
                (512 * 1024 * 1024u64).to_string(),
            ),
        ]);
        let cfg = DataCacheConfig::from_storage_options(&opts).unwrap();
        assert_eq!(cfg.max_memory_bytes, 512 * 1024 * 1024);
        assert!(cfg.ssd_cache_dir.is_none());
    }

    #[test]
    fn test_config_full() {
        let ssd_bytes: u64 = 100_000 * 1024 * 1024;
        let opts = HashMap::from([
            (DataCacheConfig::KEY_ENABLED.to_string(), "true".to_string()),
            (
                DataCacheConfig::KEY_MEMORY_BYTES.to_string(),
                (1000 * 1024 * 1024u64).to_string(),
            ),
            (
                DataCacheConfig::KEY_SSD_ENABLED.to_string(),
                "true".to_string(),
            ),
            (
                DataCacheConfig::KEY_SSD_DIR.to_string(),
                "/mnt/nvme/cache".to_string(),
            ),
            (
                DataCacheConfig::KEY_SSD_BYTES.to_string(),
                ssd_bytes.to_string(),
            ),
        ]);
        let cfg = DataCacheConfig::from_storage_options(&opts).unwrap();
        assert_eq!(cfg.max_memory_bytes, 1000 * 1024 * 1024);
        assert_eq!(cfg.ssd_cache_dir, Some(PathBuf::from("/mnt/nvme/cache")));
        assert_eq!(cfg.ssd_max_bytes, ssd_bytes);
    }

    #[test]
    fn test_config_ssd_disabled_ignores_ssd_keys() {
        let opts = HashMap::from([
            (DataCacheConfig::KEY_ENABLED.to_string(), "true".to_string()),
            (
                DataCacheConfig::KEY_MEMORY_BYTES.to_string(),
                "1073741824".to_string(),
            ),
            (
                DataCacheConfig::KEY_SSD_ENABLED.to_string(),
                "false".to_string(),
            ),
            (
                DataCacheConfig::KEY_SSD_DIR.to_string(),
                "/mnt/nvme/cache".to_string(),
            ),
            (
                DataCacheConfig::KEY_SSD_BYTES.to_string(),
                "107374182400".to_string(),
            ),
        ]);
        let cfg = DataCacheConfig::from_storage_options(&opts).unwrap();
        assert!(!cfg.ssd_enabled, "ssd_enabled flag must be false");
        assert_eq!(cfg.ssd_cache_dir, Some(PathBuf::from("/mnt/nvme/cache")));
    }

    #[test]
    fn test_config_master_switch_false() {
        let opts = HashMap::from([
            (
                DataCacheConfig::KEY_ENABLED.to_string(),
                "false".to_string(),
            ),
            (
                DataCacheConfig::KEY_MEMORY_BYTES.to_string(),
                "1073741824".to_string(),
            ),
        ]);
        assert!(DataCacheConfig::from_storage_options(&opts).is_none());
    }

    #[tokio::test]
    async fn test_tiered_cache_memory_hit() {
        let config = DataCacheConfig {
            max_memory_bytes: 10 * 1024 * 1024,
            num_shards: memory::DEFAULT_NUM_SHARDS,
            ssd_enabled: false,
            ssd_cache_dir: None,
            ssd_max_bytes: 0,
            ssd_num_shards: DEFAULT_NUM_SSD_SHARDS,
            verify: false,
            ssd_crc32_enabled: false,
        };
        let cache = TieredDataCache::new(&config).await.unwrap();
        let path = Path::from("test/file.lance");

        let result = cache
            .get_or_load(
                &path,
                0,
                5,
                Box::pin(async { Ok(Bytes::from_static(b"hello")) }),
            )
            .await
            .unwrap();
        assert_eq!(result, Bytes::from_static(b"hello"));

        let result2 = cache
            .get_or_load(
                &path,
                0,
                5,
                Box::pin(async { panic!("loader should not be called on cache hit") }),
            )
            .await
            .unwrap();
        assert_eq!(result2, Bytes::from_static(b"hello"));

        assert_eq!(cache.memory_stats().hits, 1);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_two_tier_ssd_fallback_data_integrity() {
        let tmp = tempfile::tempdir().unwrap();
        let config = DataCacheConfig {
            max_memory_bytes: 512 * 1024,
            num_shards: memory::DEFAULT_NUM_SHARDS,
            ssd_enabled: true,
            ssd_cache_dir: Some(tmp.path().join("two_tier")),
            ssd_max_bytes: ssd::REGION_SIZE * 4,
            ssd_num_shards: 1,
            verify: false,
            ssd_crc32_enabled: false,
        };
        let cache = TieredDataCache::new(&config).await.unwrap();
        let path = Path::from("s3://bucket/data.lance");

        let entry_size = 256 * 1024u64;
        let n = 4u64;

        for i in 0..n {
            let pattern = Bytes::from(vec![(i * 37 % 256) as u8; entry_size as usize]);
            let p = pattern.clone();
            cache
                .get_or_load(
                    &path,
                    i * entry_size,
                    entry_size,
                    Box::pin(async move { Ok(p) }),
                )
                .await
                .unwrap();
        }

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let refetch_count = Arc::new(std::sync::atomic::AtomicU64::new(0));
        for i in 0..n {
            let expected = (i * 37 % 256) as u8;
            let rc = refetch_count.clone();
            let result = cache
                .get_or_load(
                    &path,
                    i * entry_size,
                    entry_size,
                    Box::pin(async move {
                        rc.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        Ok(Bytes::from(vec![(i * 37 % 256) as u8; entry_size as usize]))
                    }),
                )
                .await
                .unwrap();
            assert_eq!(result.len(), entry_size as usize, "entry {i}: wrong size");
            assert_eq!(result[0], expected, "entry {i}: data corruption detected");
        }
        let refetches = refetch_count.load(std::sync::atomic::Ordering::Relaxed);
        tracing::debug!("two-tier test: {refetches}/{n} re-fetches from object store");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_tiered_cache_stats_accumulate() {
        let tmp = tempfile::tempdir().unwrap();
        let config = DataCacheConfig {
            max_memory_bytes: 4 * 1024 * 1024,
            num_shards: memory::DEFAULT_NUM_SHARDS,
            ssd_enabled: true,
            ssd_cache_dir: Some(tmp.path().join("stats_test")),
            ssd_max_bytes: ssd::REGION_SIZE * 2,
            ssd_num_shards: 1,
            verify: false,
            ssd_crc32_enabled: false,
        };
        let cache = TieredDataCache::new(&config).await.unwrap();
        let path = Path::from("test.lance");

        for i in 0u64..5 {
            let data = Bytes::from(vec![i as u8; 4096]);
            cache
                .get_or_load(&path, i * 4096, 4096, Box::pin(async move { Ok(data) }))
                .await
                .unwrap();
        }

        let stats = cache.memory_stats();
        assert_eq!(stats.misses, 5);
        assert_eq!(stats.current_bytes, 5 * 4096);

        for i in 0u64..5 {
            cache
                .get_or_load(
                    &path,
                    i * 4096,
                    4096,
                    Box::pin(async { panic!("must hit") }),
                )
                .await
                .unwrap();
        }
        assert_eq!(cache.memory_stats().hits, 5);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_ssd_writer_fires_on_eviction() {
        let tmp = tempfile::tempdir().unwrap();
        let config = DataCacheConfig {
            max_memory_bytes: 2 * 1024 * 1024,
            num_shards: 1,
            ssd_enabled: true,
            ssd_cache_dir: Some(tmp.path().join("eviction_test")),
            ssd_max_bytes: ssd::REGION_SIZE * 4,
            ssd_num_shards: 1,
            verify: false,
            ssd_crc32_enabled: false,
        };
        let cache = TieredDataCache::new(&config).await.unwrap();
        let path = Path::from("s3://bucket/data.lance");
        let entry_size = 1024 * 1024u64;
        let n_load = 6u64;
        // L1 holds 2 entries (2 MiB limit, 1 MiB each). Load them first, then
        // re-access enough times to reach SSD_MIN_HITS, then overflow L1 so the
        // hot entries are evicted and admitted to SSD.
        let l1_capacity = 2u64;
        for i in 0..l1_capacity {
            let data = Bytes::from(vec![(i % 256) as u8; entry_size as usize]);
            cache
                .get_or_load(
                    &path,
                    i * entry_size,
                    entry_size,
                    Box::pin(async move { Ok(data) }),
                )
                .await
                .unwrap();
        }
        for _ in 0..(SSD_MIN_HITS - 1) {
            for i in 0..l1_capacity {
                cache
                    .get_or_load(
                        &path,
                        i * entry_size,
                        entry_size,
                        Box::pin(async { panic!("must hit L1") }),
                    )
                    .await
                    .unwrap();
            }
        }
        for i in l1_capacity..n_load {
            let data = Bytes::from(vec![(i % 256) as u8; entry_size as usize]);
            cache
                .get_or_load(
                    &path,
                    i * entry_size,
                    entry_size,
                    Box::pin(async move { Ok(data) }),
                )
                .await
                .unwrap();
        }

        cache.flush_ssd().await;

        let ssd_hits_before = cache.ssd.as_ref().unwrap().stats().entries_read;
        let refetch_count = Arc::new(std::sync::atomic::AtomicU64::new(0));

        for i in 0..n_load.saturating_sub(2) {
            let expected = (i % 256) as u8;
            let rc = refetch_count.clone();
            let result = cache
                .get_or_load(
                    &path,
                    i * entry_size,
                    entry_size,
                    Box::pin(async move {
                        rc.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        Ok(Bytes::from(vec![(i % 256) as u8; entry_size as usize]))
                    }),
                )
                .await
                .unwrap();
            assert_eq!(result[0], expected, "entry {i}: data corruption");
        }

        let ssd_hits_after = cache.ssd.as_ref().unwrap().stats().entries_read;
        assert!(
            ssd_hits_after > ssd_hits_before,
            "expected at least one SSD hit after flush_ssd()"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_ssd_corruption_silently_returned_without_checksum() {
        use std::os::unix::fs::FileExt;

        let tmp = tempfile::tempdir().unwrap();
        let ssd_dir = tmp.path().join("corrupt_test");
        let entry_size = 64 * 1024u64;
        let correct_pattern = 0xABu8;

        {
            let config = DataCacheConfig {
                max_memory_bytes: 128 * 1024,
                num_shards: 1,
                ssd_enabled: true,
                ssd_cache_dir: Some(ssd_dir.clone()),
                ssd_max_bytes: ssd::REGION_SIZE * 2,
                ssd_num_shards: 1,
                verify: false,
                ssd_crc32_enabled: false,
            };
            let cache = TieredDataCache::new(&config).await.unwrap();
            let path = Path::from("s3://bucket/data.lance");

            // L1 holds 2 entries (128 KiB / 64 KiB). Load them, re-hit enough
            // times to reach SSD_MIN_HITS, then overflow so they spill to SSD.
            let l1_capacity = 2u64;
            for i in 0..l1_capacity {
                let data = Bytes::from(vec![correct_pattern; entry_size as usize]);
                cache
                    .get_or_load(
                        &path,
                        i * entry_size,
                        entry_size,
                        Box::pin(async move { Ok(data) }),
                    )
                    .await
                    .unwrap();
            }
            for _ in 0..(SSD_MIN_HITS - 1) {
                for i in 0..l1_capacity {
                    cache
                        .get_or_load(
                            &path,
                            i * entry_size,
                            entry_size,
                            Box::pin(async { panic!("must hit L1") }),
                        )
                        .await
                        .unwrap();
                }
            }
            for i in l1_capacity..4u64 {
                let data = Bytes::from(vec![correct_pattern; entry_size as usize]);
                cache
                    .get_or_load(
                        &path,
                        i * entry_size,
                        entry_size,
                        Box::pin(async move { Ok(data) }),
                    )
                    .await
                    .unwrap();
            }
            cache.flush_ssd().await;
            assert!(cache.ssd.as_ref().unwrap().stats().entries_written > 0);
        }

        {
            let cache_file = ssd_dir.join("cache_0.bin");
            let f = std::fs::OpenOptions::new()
                .write(true)
                .open(&cache_file)
                .unwrap();
            f.write_all_at(&vec![0xFFu8; 4096], 0).unwrap();
        }

        let config = DataCacheConfig {
            max_memory_bytes: 128 * 1024,
            num_shards: 1,
            ssd_enabled: true,
            ssd_cache_dir: Some(ssd_dir),
            ssd_max_bytes: ssd::REGION_SIZE * 2,
            ssd_num_shards: 1,
            verify: false,
            ssd_crc32_enabled: false,
        };
        let cache = TieredDataCache::new(&config).await.unwrap();
        let path = Path::from("s3://bucket/data.lance");

        let result = cache
            .get_or_load(
                &path,
                0,
                entry_size,
                Box::pin(
                    async move { Ok(Bytes::from(vec![correct_pattern; entry_size as usize])) },
                ),
            )
            .await
            .unwrap();

        assert_eq!(
            result.len(),
            entry_size as usize,
            "SSD should return correct length"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_ssd_corruption_detected_by_checksum() {
        use std::os::unix::fs::FileExt;

        let tmp = tempfile::tempdir().unwrap();
        let ssd_dir = tmp.path().join("corrupt_test");
        let entry_size = 64 * 1024u64;
        let correct_pattern = 0xABu8;

        {
            let config = DataCacheConfig {
                max_memory_bytes: 128 * 1024,
                num_shards: 1,
                ssd_enabled: true,
                ssd_cache_dir: Some(ssd_dir.clone()),
                ssd_max_bytes: ssd::REGION_SIZE * 2,
                ssd_num_shards: 1,
                verify: false,
                ssd_crc32_enabled: false,
            };
            let cache = TieredDataCache::new(&config).await.unwrap();
            let path = Path::from("s3://bucket/data.lance");

            // L1 holds 2 entries (128 KiB / 64 KiB). Load them, re-hit enough
            // times to reach SSD_MIN_HITS, then overflow so they spill to SSD.
            let l1_capacity = 2u64;
            for i in 0..l1_capacity {
                let data = Bytes::from(vec![correct_pattern; entry_size as usize]);
                cache
                    .get_or_load(
                        &path,
                        i * entry_size,
                        entry_size,
                        Box::pin(async move { Ok(data) }),
                    )
                    .await
                    .unwrap();
            }
            for _ in 0..(SSD_MIN_HITS - 1) {
                for i in 0..l1_capacity {
                    cache
                        .get_or_load(
                            &path,
                            i * entry_size,
                            entry_size,
                            Box::pin(async { panic!("must hit L1") }),
                        )
                        .await
                        .unwrap();
                }
            }
            for i in l1_capacity..4u64 {
                let data = Bytes::from(vec![correct_pattern; entry_size as usize]);
                cache
                    .get_or_load(
                        &path,
                        i * entry_size,
                        entry_size,
                        Box::pin(async move { Ok(data) }),
                    )
                    .await
                    .unwrap();
            }
            cache.flush_ssd().await;

            let written = cache.ssd.as_ref().unwrap().stats().entries_written;
            assert!(written > 0, "no entries written to SSD");
        }

        {
            let cache_file = ssd_dir.join("cache_0.bin");
            let f = std::fs::OpenOptions::new()
                .write(true)
                .open(&cache_file)
                .unwrap();
            f.write_all_at(&vec![0xFFu8; 4096], 0).unwrap();
        }

        let config_verify = DataCacheConfig {
            max_memory_bytes: 128 * 1024,
            num_shards: 1,
            ssd_enabled: true,
            ssd_cache_dir: Some(ssd_dir),
            ssd_max_bytes: ssd::REGION_SIZE * 2,
            ssd_num_shards: 1,
            verify: true,
            ssd_crc32_enabled: false,
        };
        let cache_v = TieredDataCache::new(&config_verify).await.unwrap();
        let path = Path::from("s3://bucket/data.lance");

        let result = cache_v
            .get_or_load(
                &path,
                0,
                entry_size,
                Box::pin(
                    async move { Ok(Bytes::from(vec![correct_pattern; entry_size as usize])) },
                ),
            )
            .await
            .unwrap();

        assert_eq!(result.len(), entry_size as usize);
        assert!(
            result.iter().all(|&b| b == correct_pattern),
            "verify path returned corrupted bytes — should have fallen back to source"
        );
    }
}
