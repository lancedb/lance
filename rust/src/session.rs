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

use crate::dataset::DEFAULT_INDEX_CACHE_SIZE;
use crate::index::cache::IndexCache;

/// A user session tracks the runtime state.
#[derive(Clone)]
pub struct Session {
    /// Cache for opened indices.
    pub(crate) index_cache: IndexCache,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Session()")
    }
}

impl Session {
    /// Create a new session.
    ///
    /// Parameters:
    ///
    /// - ***index_cache_size***: the size of the index cache.
    pub fn new(index_cache_size: usize) -> Self {
        Self {
            index_cache: IndexCache::new(index_cache_size),
        }
    }
}

impl Default for Session {
    fn default() -> Self {
        Self {
            index_cache: IndexCache::new(DEFAULT_INDEX_CACHE_SIZE),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::index::vector::{
        pq::{PQIndex, ProductQuantizer},
        MetricType,
    };
    use std::sync::Arc;

    #[test]
    fn test_disable_index_cache() {
        let no_cache = Session::new(0);
        assert!(no_cache.index_cache.get("abc").is_none());

        let pq = Arc::new(ProductQuantizer::new(1, 8, 1));
        let idx = Arc::new(PQIndex::new(pq, MetricType::L2));
        no_cache.index_cache.insert("abc", idx);

        assert!(no_cache.index_cache.get("abc").is_none());
        assert_eq!(no_cache.index_cache.len(), 0);
    }
}
