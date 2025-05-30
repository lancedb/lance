// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use deepsize::DeepSizeOf;
use lance_index::vector::VectorIndexCacheEntry;
use lance_index::{
    scalar::{ScalarIndex, ScalarIndexType},
    vector::VectorIndex,
};
use lance_table::format::Index;
use moka::sync::Cache;

use lance_index::frag_reuse::FragReuseIndex;
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
    frag_reuse_cache: Arc<Cache<String, Arc<FragReuseIndex>>>,
    // this is for v3 index, sadly we can't use the same cache as the vector index for now
    vector_partition_cache: Arc<Cache<String, Arc<dyn VectorIndexCacheEntry>>>,

    /// Index metadata cache.
    ///
    /// The key is "{dataset_base_path}:{version}".
    /// Value is all the indies of a particular version of the dataset.
    metadata_cache: Arc<Cache<String, Arc<Vec<Index>>>>,

    /// Caches the ScalarIndexType for each index (it can be expensive to determine this
    /// in older indices that do not store index_details)
    type_cache: Arc<Cache<String, ScalarIndexType>>,

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
                .vector_partition_cache
                .iter()
                .map(|(_, v)| v.deep_size_of_children(context))
                .sum::<usize>()
            + self
                .frag_reuse_cache
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
            vector_partition_cache: Arc::new(Cache::new(capacity as u64)),
            // there is always 1 fragment reuse index that should be used
            frag_reuse_cache: Arc::new(Cache::new(1)),
            metadata_cache: Arc::new(Cache::new(capacity as u64)),
            type_cache: Arc::new(Cache::new(capacity as u64)),
            cache_stats: Arc::new(CacheStats::default()),
        }
    }

    /// Clear the cache
    #[cfg(test)]
    pub fn clear(&self) {
        self.scalar_cache.invalidate_all();
        self.scalar_cache.run_pending_tasks();

        self.vector_cache.invalidate_all();
        self.vector_cache.run_pending_tasks();

        self.vector_partition_cache.invalidate_all();
        self.vector_partition_cache.run_pending_tasks();

        self.frag_reuse_cache.invalidate_all();
        self.frag_reuse_cache.run_pending_tasks();

        self.metadata_cache.invalidate_all();
        self.metadata_cache.run_pending_tasks();

        self.type_cache.invalidate_all();
        self.type_cache.run_pending_tasks();
    }

    #[allow(dead_code)]
    pub(crate) fn len_vector(&self) -> usize {
        self.vector_cache.run_pending_tasks();
        self.vector_cache.entry_count() as usize
    }

    pub(crate) fn get_size(&self) -> usize {
        self.scalar_cache.run_pending_tasks();
        self.vector_cache.run_pending_tasks();
        self.vector_partition_cache.run_pending_tasks();
        self.frag_reuse_cache.run_pending_tasks();
        self.metadata_cache.run_pending_tasks();
        (self.scalar_cache.entry_count()
            + self.vector_cache.entry_count()
            + self.vector_partition_cache.entry_count()
            + self.frag_reuse_cache.entry_count()
            + self.metadata_cache.entry_count()) as usize
    }

    pub(crate) fn approx_size(&self) -> usize {
        (self.scalar_cache.entry_count()
            + self.vector_cache.entry_count()
            + self.vector_partition_cache.entry_count()
            + self.frag_reuse_cache.entry_count()
            + self.metadata_cache.entry_count()) as usize
    }

    pub(crate) fn get_type(&self, key: &str) -> Option<ScalarIndexType> {
        if let Some(index) = self.type_cache.get(key) {
            self.cache_stats.record_hit();
            Some(index)
        } else {
            self.cache_stats.record_miss();
            None
        }
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

    pub(crate) fn get_vector_partition(&self, key: &str) -> Option<Arc<dyn VectorIndexCacheEntry>> {
        if let Some(index) = self.vector_partition_cache.get(key) {
            self.cache_stats.record_hit();
            Some(index)
        } else {
            self.cache_stats.record_miss();
            None
        }
    }

    pub(crate) fn get_frag_reuse(&self, key: &str) -> Option<Arc<FragReuseIndex>> {
        if let Some(index) = self.frag_reuse_cache.get(key) {
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

    pub(crate) fn insert_frag_reuse(&self, key: &str, index: Arc<FragReuseIndex>) {
        self.frag_reuse_cache.insert(key.to_string(), index);
    }

    pub(crate) fn insert_vector_partition(&self, key: &str, index: Arc<dyn VectorIndexCacheEntry>) {
        self.vector_partition_cache.insert(key.to_string(), index);
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

    pub(crate) fn insert_type(&self, key: &str, index_type: ScalarIndexType) {
        self.type_cache.insert(key.to_string(), index_type);
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
