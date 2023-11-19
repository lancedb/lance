// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::mem;
use std::sync::Arc;

use lance_index::scalar::ScalarIndex;
use moka::sync::{Cache, ConcurrentCacheExt};

use super::vector::VectorIndex;

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Default)]
pub struct CacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
}

impl CacheStats {
    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Clone)]
pub struct IndexCache {
    scalar_cache: Arc<Cache<String, Arc<dyn ScalarIndex>>>,
    vector_cache: Arc<Cache<String, Arc<dyn VectorIndex>>>,
    cache_stats: Arc<CacheStats>,
}

impl IndexCache {
    pub(crate) fn new(capacity: usize) -> Self {
        let vector_cache = Cache::builder()
            .weigher(|k: &String, v: &Arc<dyn VectorIndex>| -> u32 { (k.len() + mem::size_of_val(v)) as u32 })
            .max_capacity(capacity as u64)
            .build();

        let scalar_cache = Cache::builder()
            .weigher(|k: &String, v: &Arc<dyn ScalarIndex>| -> u32 { (k.len() + mem::size_of_val(v)) as u32 })
            .max_capacity(capacity as u64)
            .build();

        Self {
            scalar_cache: Arc::new(scalar_cache),
            vector_cache: Arc::new(vector_cache),
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
        (self.scalar_cache.entry_count() + self.vector_cache.entry_count()) as usize
    }

    #[allow(dead_code)]
    pub(crate) fn get_byte_size(&self) -> u64 {
        self.scalar_cache.sync();
        self.vector_cache.sync();
        self.scalar_cache.weighted_size() + self.vector_cache.weighted_size()
    }

    /// Get an Index if present. Otherwise returns [None].
    pub(crate) fn get_scalar(&self, key: &str) -> Option<Arc<dyn ScalarIndex>> {
        self.scalar_cache.get(key)
    }

    pub(crate) fn get_vector(&self, key: &str) -> Option<Arc<dyn VectorIndex>> {
        if self.vector_cache.contains_key(key) || self.scalar_cache.contains_key(key) {
            self.cache_stats.record_hit();
        } else {
            self.cache_stats.record_miss();
        }
        self.vector_cache.get(key)
    }

    /// Insert a new entry into the cache.
    pub(crate) fn insert_scalar(&self, key: &str, index: Arc<dyn ScalarIndex>) {
        self.scalar_cache.insert(key.to_string(), index);
    }

    pub(crate) fn insert_vector(&self, key: &str, index: Arc<dyn VectorIndex>) {
        self.vector_cache.insert(key.to_string(), index);
    }

    /// Get cache hit ratio.
    #[allow(dead_code)]
    pub(crate) fn hit_rate(&self) -> f32 {
        let hits = self.cache_stats.hits.load(Ordering::Relaxed) as f32;
        let misses = self.cache_stats.misses.load(Ordering::Relaxed) as f32;
        // Returns 1.0 if hits + misses == 0 and avoids division by zero.
        if (hits + misses) == 0.0 {
            return 1.0;
        }
        hits / (hits + misses)
    }
}
