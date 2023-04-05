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
use arrow_schema::DataType;
use lru_time_cache::LruCache;
use object_store::path::Path;

use super::Vertex;
use crate::arrow::as_fixed_size_binary_array;
use crate::datatypes::Schema;
use crate::io::{FileReader, ObjectStore};
use crate::{Error, Result};

const NEIGHBORS_COL: &str = "neighbors";
const VERTEX_COL: &str = "vertex";

/// Persisted graph on disk, stored in the file.
pub struct PersistedGraph<'a, V: Vertex> {
    reader: FileReader<'a>,

    /// Vertex size in bytes.
    vertex_size: usize,

    /// Projection of the vertex column.
    vertex_projection: Schema,

    /// LRU cache for vertices.
    cache: Arc<Mutex<LruCache<u32, Arc<V>>>>,

    /// LRU cache for neighbors.
    neighbors_cache: Arc<Mutex<LruCache<u32, UInt32Array>>>,

    prefetch_byte_size: usize,
}

impl<'a, V: Vertex> PersistedGraph<'a, V> {
    /// Try open a persisted graph from a given URI.
    pub async fn try_new(
        object_store: &'a ObjectStore,
        path: &Path,
    ) -> Result<PersistedGraph<'a, V>> {
        let file_reader = FileReader::try_new(object_store, path).await?;

        let schema = file_reader.schema();
        let vertex_projection = schema.project(&[VERTEX_COL])?;
        let vertex_size = if let Some(field) = vertex_projection.fields.get(0) {
            match field.data_type() {
                DataType::FixedSizeBinary(size) => size as usize,
                _ => {
                    return Err(Error::Index(format!(
                        "Vertex column must be of fixed size binary, got: {}",
                        field.data_type()
                    )))
                }
            }
        } else {
            return Err(Error::Index(
                "Vertex column does not exist in the graph".to_string(),
            ));
        };

        Ok(Self {
            reader: file_reader,
            vertex_size,
            vertex_projection,
            cache: Arc::new(Mutex::new(LruCache::with_capacity(1000000))),
            neighbors_cache: Arc::new(Mutex::new(LruCache::with_capacity(1000))),
            prefetch_byte_size: 8196, // 8MB
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
        let prefetch_size = self.prefetch_byte_size / self.vertex_size + 1;
        let batch = self
            .reader
            .read_range(
                id as usize..id as usize + prefetch_size,
                &self.vertex_projection,
            )
            .await?;
        assert_eq!(batch.num_rows(), prefetch_size);
        {
            let mut cache = self.cache.lock().unwrap();
            let array = as_fixed_size_binary_array(batch.column(0));
            for i in 0..prefetch_size {
                let vertex_bytes = array.value(i);
                let vertex = V::from_bytes(vertex_bytes)?;
                cache.insert(id + i as u32, Arc::new(vertex));
            }
            Ok(cache.get(&id).unwrap().clone())
        }
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
        fn byte_length(&self) -> usize {
            return 20;
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn from_bytes(data: &[u8]) -> Result<Self> {
            if data.len() != 20 {
                return Err(Error::Index(format!(
                    "Invalid vertex size, expected: 20, got: {}",
                    data.len()
                )));
            }
            let row_id = u32::from_le_bytes(data[0..4].try_into().unwrap());
            let pq = data[4..].to_vec();
            Ok(Self { row_id, pq })
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
