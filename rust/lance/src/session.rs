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

use std::any::{Any, TypeId};
use std::sync::Arc;

use moka::sync::Cache;
use object_store::path::Path;

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

type ArcAny = Arc<dyn Any + Send + Sync>;

/// Cache for various metadata about files.
///
/// The cache is keyed by the file path and the type of metadata.
#[derive(Clone)]
pub struct FileMetadataCache {
    cache: Arc<Cache<(Path, TypeId), ArcAny>>,
}

impl FileMetadataCache {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            cache: Arc::new(Cache::new(capacity as u64)),
        }
    }

    pub(crate) fn get<T: Send + Sync + 'static>(&self, path: &Path) -> Option<Arc<T>> {
        self.cache
            .get(&(path.to_owned(), TypeId::of::<T>()))
            .map(|metadata| metadata.clone().downcast::<T>().unwrap())
    }

    pub(crate) fn insert<T: Send + Sync + 'static>(&self, path: Path, metadata: Arc<T>) {
        self.cache.insert((path, TypeId::of::<T>()), metadata);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use crate::index::vector::{
        pq::{PQIndex, ProductQuantizer},
        VectorIndex,
    };
    use arrow_array::{UInt64Array, UInt8Array};
    use lance_linalg::distance::MetricType;

    fn dummy_index(num_sub_vectors: usize) -> Arc<dyn VectorIndex> {
        let pq = Arc::new(ProductQuantizer::new(num_sub_vectors, 8, 1));
        let empty_code = Arc::new(UInt8Array::from_value(0, 0));
        let empty_row_ids = Arc::new(UInt64Array::from_value(0, 0));
        Arc::new(PQIndex::new(pq, MetricType::L2, empty_code, empty_row_ids))
    }

    #[test]
    fn test_disable_index_cache() {
        let no_cache = Session::new(0, 0);
        assert!(no_cache.index_cache.get("abc").is_none());
        let no_cache = Arc::new(no_cache);

        no_cache.index_cache.insert("abc", dummy_index(1));

        assert!(no_cache.index_cache.get("abc").is_none());
        assert_eq!(no_cache.index_cache.len(), 0);
    }

    #[test]
    fn test_basic() {
        let session = Session::new(10, 1);
        let session = Arc::new(session);

        session.index_cache.insert("abc", dummy_index(1));

        let found = session.index_cache.get("abc");
        assert!(found.is_some());
        assert_eq!(
            format!("{:?}", found.unwrap()),
            format!("{:?}", dummy_index(1))
        );
        assert!(session.index_cache.get("abc").is_some());
        assert_eq!(session.index_cache.len(), 1);

        for iter_idx in 0..100 {
            session
                .index_cache
                .insert(format!("{iter_idx}").as_str(), dummy_index(16));
        }

        // Capacity is 10 so there should be at most 10 items
        assert_eq!(session.index_cache.len(), 10);
    }
}
