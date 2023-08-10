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

use moka::sync::{Cache, ConcurrentCacheExt};

use super::vector::VectorIndex;

#[derive(Clone)]
pub struct IndexCache {
    cache: Arc<Cache<String, Arc<dyn VectorIndex>>>,
}

impl IndexCache {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            cache: Arc::new(Cache::new(capacity as u64)),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.cache.sync();
        self.cache.entry_count() as usize
    }

    /// Get an Index if present. Otherwise returns [None].
    pub(crate) fn get(&self, key: &str) -> Option<Arc<dyn VectorIndex>> {
        self.cache.get(key)
    }

    /// Insert a new entry into the cache.
    pub(crate) fn insert(&self, key: &str, index: Arc<dyn VectorIndex>) {
        self.cache.insert(key.to_string(), index);
    }
}
