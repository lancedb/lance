// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use deepsize::DeepSizeOf;
use lance_index::{scalar::ScalarIndex, vector::VectorIndex};
use lance_table::format::Index;
use moka::sync::{Cache, ConcurrentCacheExt};

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Default, DeepSizeOf)]
struct CacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
}

impl CacheStats {
    fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Clone)]
pub struct IndexCache {
    // TODO: Can we merge these two caches into one for uniform memory management?
    scalar_cache: Arc<Cache<String, Arc<dyn ScalarIndex>>>,
    vector_cache: Arc<Cache<String, Arc<dyn VectorIndex>>>,

    /// Index metadata cache.
    ///
    /// The key is "{dataset_base_path}:{version}".
    /// Value is all the indies of a particular version of the dataset.
    metadata_cache: Arc<Cache<String, Arc<Vec<Index>>>>,

    cache_stats: Arc<CacheStats>,
}

impl DeepSizeOf for IndexCache {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.scalar_cache
            .iter()
            .map(|(_, v)| v.deep_size_of_children(context))
            .sum::<usize>()
            + self
                .vector_cache
                .iter()
                .map(|(_, v)| v.deep_size_of_children(context))
                .sum::<usize>()
            + self
                .metadata_cache
                .iter()
                .map(|(_, v)| v.deep_size_of_children(context))
                .sum::<usize>()
            + self.cache_stats.deep_size_of_children(context)
    }
}

impl IndexCache {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            scalar_cache: Arc::new(Cache::new(capacity as u64)),
            vector_cache: Arc::new(Cache::new(capacity as u64)),
            metadata_cache: Arc::new(Cache::new(capacity as u64)),
            cache_stats: Arc::new(CacheStats::default()),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn len_vector(&self) -> usize {
        self.vector_cache.sync();
        self.vector_cache.entry_count() as usize
    }

    pub(crate) fn get_size(&self) -> usize {
        self.scalar_cache.sync();
        self.vector_cache.sync();
        self.metadata_cache.sync();
        (self.scalar_cache.entry_count()
            + self.vector_cache.entry_count()
            + self.metadata_cache.entry_count()) as usize
    }

    /// Get an Index if present. Otherwise returns [None].
    pub(crate) fn get_scalar(&self, key: &str) -> Option<Arc<dyn ScalarIndex>> {
        if let Some(index) = self.scalar_cache.get(key) {
            self.cache_stats.record_hit();
            Some(index)
        } else {
            self.cache_stats.record_miss();
            None
        }
    }

    pub(crate) fn get_vector(&self, key: &str) -> Option<Arc<dyn VectorIndex>> {
        if let Some(index) = self.vector_cache.get(key) {
            self.cache_stats.record_hit();
            Some(index)
        } else {
            self.cache_stats.record_miss();
            None
        }
    }

    /// Insert a new entry into the cache.
    pub(crate) fn insert_scalar(&self, key: &str, index: Arc<dyn ScalarIndex>) {
        self.scalar_cache.insert(key.to_string(), index);
    }

    pub(crate) fn insert_vector(&self, key: &str, index: Arc<dyn VectorIndex>) {
        self.vector_cache.insert(key.to_string(), index);
    }

    /// Construct a key for index metadata arrays.
    fn metadata_key(dataset_uuid: &str, version: u64) -> String {
        format!("{}:{}", dataset_uuid, version)
    }

    /// Get all index metadata for a particular dataset version.
    pub(crate) fn get_metadata(&self, key: &str, version: u64) -> Option<Arc<Vec<Index>>> {
        let key = Self::metadata_key(key, version);
        if let Some(indices) = self.metadata_cache.get(&key) {
            self.cache_stats.record_hit();
            Some(indices)
        } else {
            self.cache_stats.record_miss();
            None
        }
    }

    pub(crate) fn insert_metadata(&self, key: &str, version: u64, indices: Arc<Vec<Index>>) {
        let key = Self::metadata_key(key, version);

        self.metadata_cache.insert(key, indices);
    }

    /// Get cache hit ratio.
    #[allow(dead_code)]
    pub(crate) fn hit_rate(&self) -> f32 {
        let hits = self.cache_stats.hits.load(Ordering::Relaxed) as f32;
        let misses = self.cache_stats.misses.load(Ordering::Relaxed) as f32;
        // Returns 1.0 if hits + misses == 0 and avoids division by zero.
        if (hits + misses) == 0.0 {
            1.0
        } else {
            hits / (hits + misses)
        }
    }
}
