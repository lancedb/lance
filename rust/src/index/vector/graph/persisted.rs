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

use arrow_array::UInt32Array;
use lru_time_cache::LruCache;
use object_store::path::Path;

use super::Vertex;
use crate::io::{FileReader, ObjectStore};
use crate::Result;

const NEIGHBORS_COL: &str = "neighbors";
const VERTEX_COL: &str = "vertex";

/// Persisted graph on disk, stored in the file.
pub struct PersistedGraph<'a, V: Vertex> {
    reader: FileReader<'a>,

    /// LRU cache for vertices.
    cache: Arc<Mutex<LruCache<u32, Arc<V>>>>,

    /// LRU cache for neighbors.
    neighbors_cache: Arc<Mutex<LruCache<u32, UInt32Array>>>,
}

impl<'a, V: Vertex> PersistedGraph<'a, V> {
    /// Try open a persisted graph from a given URI.
    pub async fn try_new(
        object_store: &'a ObjectStore,
        path: &Path,
    ) -> Result<PersistedGraph<'a, V>> {
        let file_reader = FileReader::try_new(object_store, path).await?;
        Ok(Self {
            reader: file_reader,
            cache: Arc::new(Mutex::new(LruCache::with_capacity(1000000))),
            neighbors_cache: Arc::new(Mutex::new(LruCache::with_capacity(1000))),
        })
    }

    /// The number of vertices in the graph.
    pub fn len(&self) -> usize {
        self.reader.len()
    }

    /// Get the vertex specified by its id.
    pub async fn vertex(&self, id: u32) -> Result<Arc<V>> {
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(vertex) = cache.get(&id) {
                return Ok(vertex.clone());
            }
        }
        todo!()
    }

    /// Get the neighbors of a vertex, specified by its id.
    pub async fn neighbors(&self, id: u32) -> Result<&[u32]> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestVertex {
        row_id: u32,
        pq: Vec<u8>,
    }

    impl Vertex for TestVertex {
        fn bytes(&self) -> usize {
            return 20;
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[tokio::test]
    async fn test_persisted_graph() {
        let store = ObjectStore::memory();
        let path = Path::from("/graph");

        let graph = PersistedGraph::<TestVertex>::try_new(&store, &path)
            .await
            .unwrap();
    }
}
