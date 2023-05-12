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

use std::sync::{Arc, Mutex};

use lru_time_cache::LruCache;

use super::vector::VectorIndex;

#[derive(Clone)]
pub(crate) struct IndexCache {
    /// The maximum number of indices to cache.
    capacity: usize,

    cache: Arc<Mutex<LruCache<String, Arc<dyn VectorIndex>>>>,
}

impl IndexCache {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            cache: Arc::new(Mutex::new(LruCache::with_capacity(capacity))),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        let cache = self.cache.lock().unwrap();
        cache.len()
    }

    /// Get an Index if present. Otherwise returns [None].
    pub(crate) fn get(&self, key: &str) -> Option<Arc<dyn VectorIndex>> {
        let mut cache = self.cache.lock().unwrap();
        let idx = cache.get(key);
        idx.map(|idx| idx.clone())
    }

    /// Insert a new entry into the cache.
    pub(crate) fn insert(&self, key: &str, index: Arc<dyn VectorIndex>) {
        if self.capacity == 0 {
            // Work-around. lru_time_cache panics if capacity is 0.
            return;
        }
        let mut cache = self.cache.lock().unwrap();
        cache.insert(key.to_string(), index);
    }
}
