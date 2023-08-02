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
use std::sync::{Arc, Mutex};

use lru_time_cache::LruCache;
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
    /// The maximum number of metadata to cache.
    capacity: usize,

    cache: Arc<Mutex<LruCache<(Path, TypeId), ArcAny>>>,
}

impl FileMetadataCache {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            cache: Arc::new(Mutex::new(LruCache::with_capacity(capacity))),
        }
    }

    pub(crate) fn get<T: Send + Sync + 'static>(&self, path: &Path) -> Option<Arc<T>> {
        let mut cache = self.cache.lock().unwrap();
        cache
            .get(&(path.to_owned(), TypeId::of::<T>()))
            .map(|metadata| metadata.clone().downcast::<T>().unwrap())
    }

    pub(crate) fn insert<T: Send + Sync + 'static>(&self, path: Path, metadata: Arc<T>) {
        if self.capacity == 0 {
            // Work-around. lru_time_cache panics if capacity is 0.
            return;
        }
        let mut cache = self.cache.lock().unwrap();
        cache.insert((path, TypeId::of::<T>()), metadata);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        datatypes::Schema,
        format::Manifest,
        index::vector::{
            pq::{PQIndex, ProductQuantizer},
            MetricType,
        },
        io::{deletion::LruDeletionVectorStore, ObjectStore},
    };
    use std::{collections::HashMap, sync::Arc};

    #[tokio::test]
    async fn test_disable_index_cache() {
        let no_cache = Session::new(0, 0);
        assert!(no_cache.index_cache.get("abc").is_none());
        let no_cache = Arc::new(no_cache);

        let schema = Schema {
            fields: vec![],
            metadata: HashMap::new(),
        };
        let manifest = Arc::new(Manifest::new(&schema, Arc::new(vec![])));
        let object_store = Arc::new(ObjectStore::from_uri("memory://").await.unwrap().0);
        let deletion_cache = Arc::new(LruDeletionVectorStore::new(
            no_cache.clone(),
            object_store.clone(),
            object_store.as_ref().base_path().clone(),
            manifest,
        ));

        let pq = Arc::new(ProductQuantizer::new(1, 8, 1));
        let idx = Arc::new(PQIndex::new(pq, MetricType::L2, deletion_cache));
        no_cache.index_cache.insert("abc", idx);

        assert!(no_cache.index_cache.get("abc").is_none());
        assert_eq!(no_cache.index_cache.len(), 0);
    }
}
