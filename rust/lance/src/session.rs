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

use lance_core::cache::FileMetadataCache;

use crate::dataset::{DEFAULT_INDEX_CACHE_SIZE, DEFAULT_METADATA_CACHE_SIZE};
use crate::index::cache::IndexCache;

/// A user session tracks the runtime state.
#[derive(Clone)]
pub struct Session {
    /// Cache for opened indices.
    pub(crate) index_cache: IndexCache,

    /// Cache for file metadata
    pub(crate) file_metadata_cache: FileMetadataCache,
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
    pub fn new(index_cache_size: usize, metadata_cache_size: usize) -> Self {
        Self {
            index_cache: IndexCache::new(index_cache_size),
            file_metadata_cache: FileMetadataCache::new(metadata_cache_size),
        }
    }
}

impl Default for Session {
    fn default() -> Self {
        Self {
            index_cache: IndexCache::new(DEFAULT_INDEX_CACHE_SIZE),
            file_metadata_cache: FileMetadataCache::new(DEFAULT_METADATA_CACHE_SIZE),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::types::Float32Type;
    use std::sync::Arc;

    use crate::index::vector::pq::PQIndex;
    use lance_index::vector::pq::ProductQuantizerImpl;
    use lance_linalg::distance::MetricType;

    #[test]
    fn test_disable_index_cache() {
        let no_cache = Session::new(0, 0);
        assert!(no_cache.index_cache.get_vector("abc").is_none());
        let no_cache = Arc::new(no_cache);

        let pq = Arc::new(ProductQuantizerImpl::<Float32Type>::new(
            1,
            8,
            1,
            Arc::new(vec![0.0f32; 8].into()),
            MetricType::L2,
        ));
        let idx = Arc::new(PQIndex::new(pq, MetricType::L2));
        no_cache.index_cache.insert_vector("abc", idx);

        assert!(no_cache.index_cache.get_vector("abc").is_none());
        assert_eq!(no_cache.index_cache.len_vector(), 0);
    }

    #[test]
    fn test_basic() {
        let session = Session::new(10, 1);
        let session = Arc::new(session);

        let pq = Arc::new(ProductQuantizerImpl::<Float32Type>::new(
            1,
            8,
            1,
            Arc::new(vec![0.0f32; 8].into()),
            MetricType::L2,
        ));
        let idx = Arc::new(PQIndex::new(pq, MetricType::L2));
        session.index_cache.insert_vector("abc", idx.clone());

        let found = session.index_cache.get_vector("abc");
        assert!(found.is_some());
        assert_eq!(format!("{:?}", found.unwrap()), format!("{:?}", idx));
        assert!(session.index_cache.get_vector("abc").is_some());
        assert_eq!(session.index_cache.len_vector(), 1);

        for iter_idx in 0..100 {
            let pq_other = Arc::new(ProductQuantizerImpl::<Float32Type>::new(
                1,
                8,
                1,
                Arc::new(vec![0.0f32; 8].into()),
                MetricType::L2,
            ));
            let idx_other = Arc::new(PQIndex::new(pq_other, MetricType::L2));
            session
                .index_cache
                .insert_vector(format!("{iter_idx}").as_str(), idx_other.clone());
        }

        // Capacity is 10 so there should be at most 10 items
        assert_eq!(session.index_cache.len_vector(), 10);
    }
}
