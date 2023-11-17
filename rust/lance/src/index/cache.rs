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

use std::sync::Arc;

use lance_index::scalar::ScalarIndex;
use moka::sync::{Cache, ConcurrentCacheExt};

use super::vector::VectorIndex;

#[derive(Clone)]
pub struct IndexCache {
    scalar_cache: Arc<Cache<String, Arc<dyn ScalarIndex>>>,
    vector_cache: Arc<Cache<String, Arc<dyn VectorIndex>>>,
}

impl IndexCache {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            scalar_cache: Arc::new(Cache::new(capacity as u64)),
            vector_cache: Arc::new(Cache::new(capacity as u64)),
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
        self.scalar_cache.entry_count() as usize + self.vector_cache.entry_count() as usize
    }

    /// Get an Index if present. Otherwise returns [None].
    pub(crate) fn get_scalar(&self, key: &str) -> Option<Arc<dyn ScalarIndex>> {
        self.scalar_cache.get(key)
    }

    pub(crate) fn get_vector(&self, key: &str) -> Option<Arc<dyn VectorIndex>> {
        self.vector_cache.get(key)
    }

    /// Insert a new entry into the cache.
    pub(crate) fn insert_scalar(&self, key: &str, index: Arc<dyn ScalarIndex>) {
        self.scalar_cache.insert(key.to_string(), index);
    }

    pub(crate) fn insert_vector(&self, key: &str, index: Arc<dyn VectorIndex>) {
        self.vector_cache.insert(key.to_string(), index);
    }
}
