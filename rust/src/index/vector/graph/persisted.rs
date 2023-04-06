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

use arrow::array::{as_list_array, as_primitive_array};
use arrow_array::{
    builder::{FixedSizeBinaryBuilder, ListBuilder, UInt32Builder},
    Array, RecordBatch, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lru_time_cache::LruCache;
use object_store::path::Path;

use super::builder::GraphBuilder;
use super::Vertex;
use crate::arrow::as_fixed_size_binary_array;
use crate::datatypes::Schema;
use crate::io::{FileReader, FileWriter, ObjectStore};
use crate::{Error, Result};

const NEIGHBORS_COL: &str = "neighbors";
const VERTEX_COL: &str = "vertex";

/// Parameters for reading a persisted graph.
pub struct GraphReadParams {
    pub prefetch_byte_size: usize,

    pub vertex_cache_size: usize,

    pub neighbors_cache_size: usize,
}

impl Default for GraphReadParams {
    fn default() -> Self {
        Self {
            prefetch_byte_size: 8 * 1024,
            vertex_cache_size: 100_000,
            neighbors_cache_size: 1024,
        }
    }
}

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
    neighbors_cache: Arc<Mutex<LruCache<u32, Arc<UInt32Array>>>>,

    /// Projection of the neighbors column.
    neighbors_projection: Schema,

    /// Read parameters.
    params: GraphReadParams,
}

impl<'a, V: Vertex> PersistedGraph<'a, V> {
    /// Try open a persisted graph from a given URI.
    pub async fn try_new(
        object_store: &'a ObjectStore,
        path: &Path,
        params: GraphReadParams,
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
        let neighbors_projection = schema.project(&[NEIGHBORS_COL])?;

        Ok(Self {
            reader: file_reader,
            vertex_size,
            vertex_projection,
            cache: Arc::new(Mutex::new(LruCache::with_capacity(
                params.vertex_cache_size,
            ))),
            neighbors_cache: Arc::new(Mutex::new(LruCache::with_capacity(
                params.neighbors_cache_size,
            ))),
            neighbors_projection,
            params,
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
        let prefetch_size = self.params.prefetch_byte_size / self.vertex_size + 1;
        let end = std::cmp::min(self.len(), id as usize + prefetch_size);
        let batch = self
            .reader
            .read_range(id as usize..end, &self.vertex_projection)
            .await?;
        assert_eq!(batch.num_rows(), end - id as usize);
        {
            let mut cache = self.cache.lock().unwrap();
            let array = as_fixed_size_binary_array(batch.column(0));
            for (i, vertex_bytes) in array.iter().enumerate() {
                let vertex = V::from_bytes(vertex_bytes.unwrap())?;
                cache.insert(id + i as u32, Arc::new(vertex));
            }
            Ok(cache.get(&id).unwrap().clone())
        }
    }

    /// Get the neighbors of a vertex, specified by its id.
    pub async fn neighbors(&self, id: u32) -> Result<Arc<UInt32Array>> {
        {
            let mut cache = self.neighbors_cache.lock().unwrap();
            if let Some(neighbors) = cache.get(&id) {
                return Ok(neighbors.clone());
            }
        }
        let batch = self
            .reader
            .read_range(id as usize..(id + 1) as usize, &self.neighbors_projection)
            .await?;
        {
            let mut cache = self.neighbors_cache.lock().unwrap();

            let array = as_list_array(batch.column(0));
            if array.len() < 1 {
                return Err(Error::Index("Invalid graph".to_string()));
            }
            let value = array.value(0);
            let nb_array: &UInt32Array = as_primitive_array(value.as_ref());
            let neighbors = Arc::new(nb_array.clone());
            cache.insert(id, neighbors.clone());
            Ok(neighbors)
        }
    }
}

/// Parameters for writing the graph index.
pub struct WriteGraphParams {
    pub batch_size: usize,
}

impl Default for WriteGraphParams {
    fn default() -> Self {
        Self { batch_size: 10240 }
    }
}

/// Write the graph to a file.
pub async fn write_graph<V: Vertex>(
    graph: &GraphBuilder<V>,
    object_store: &ObjectStore,
    path: &Path,
    params: &WriteGraphParams,
) -> Result<()> {
    if graph.is_empty() {
        return Err(Error::Index("Invalid graph".to_string()));
    }
    let binary_size = graph.vertex(0).byte_length();
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new(
            "vertex",
            DataType::FixedSizeBinary(binary_size as i32),
            false,
        ),
        Field::new(
            "neighbors",
            DataType::List(Box::new(Field::new("item", DataType::UInt32, true))),
            false,
        ),
    ]));
    let schema = Schema::try_from(arrow_schema.as_ref())?;

    let mut writer = FileWriter::try_new(object_store, path, &schema).await?;
    for nodes in graph.nodes.as_slice().chunks(params.batch_size) {
        let mut vertex_builder =
            FixedSizeBinaryBuilder::with_capacity(nodes.len(), binary_size as i32);
        let total_neighbors = nodes.iter().map(|node| node.neighbors.len()).sum();
        let inner_builder = UInt32Builder::with_capacity(total_neighbors);
        let mut neighbors_builder = ListBuilder::with_capacity(inner_builder, nodes.len());
        for node in nodes {
            // Serialize the vertex metadata to fixed size binary bytes.
            vertex_builder.append_value(node.vertex.to_bytes())?;
            neighbors_builder
                .values()
                .append_slice(node.neighbors.as_slice());
            neighbors_builder.append(true);
        }
        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(vertex_builder.finish()),
                Arc::new(neighbors_builder.finish()),
            ],
        )?;

        writer.write(&[&batch]).await?;
    }

    writer.finish().await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FooVertex {
        row_id: u32,
        pq: Vec<u8>,
    }

    impl Vertex for FooVertex {
        fn byte_length(&self) -> usize {
            return 20;
        }

        fn to_bytes(&self) -> Vec<u8> {
            let mut bytes = Vec::with_capacity(20);
            bytes.extend_from_slice(&self.row_id.to_le_bytes());
            bytes.extend_from_slice(&self.pq[..16]);
            bytes
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

        let mut builder: GraphBuilder<FooVertex> = (0..100)
            .map(|v| FooVertex {
                row_id: v as u32,
                pq: vec![0; 16],
            })
            .collect();
        for i in 0..100 {
            for j in i..i + 10 {
                builder.add_edge(i, j);
            }
        }
        write_graph(&builder, &store, &path, &WriteGraphParams::default())
            .await
            .unwrap();

        let graph = PersistedGraph::<FooVertex>::try_new(&store, &path, GraphReadParams::default())
            .await
            .unwrap();
        let vertex = graph.vertex(77).await.unwrap();
        assert_eq!(vertex.row_id, 77);

        let vertex = graph.vertex(88).await.unwrap();
        assert_eq!(vertex.row_id, 88);
        let neighbors = graph.neighbors(88).await.unwrap();
        assert_eq!(
            neighbors.values(),
            &[88, 89, 90, 91, 92, 93, 94, 95, 96, 97]
        );
    }
}
