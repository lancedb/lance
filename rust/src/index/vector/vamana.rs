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

//! Vamana Graph, described in DiskANN (NeurIPS' 19) and its following papers.

use std::collections::{BTreeMap, BinaryHeap, HashSet};
use std::sync::Arc;

use arrow::array::as_primitive_array;
use arrow::datatypes::UInt64Type;
use arrow_array::Float32Array;
use async_trait::async_trait;
use futures::{stream, StreamExt, TryStreamExt};
use ordered_float::OrderedFloat;
use rand::distributions::Uniform;
use rand::Rng;

use super::graph::{Graph, Vertex, VertexWithDistance};
use crate::arrow::*;
use crate::dataset::{Dataset, ROW_ID};
use crate::utils::distance::l2::l2_distance;
use crate::{Error, Result};

#[derive(Debug)]
struct VemanaData {
    row_id: u64,
}

type VemanaVertex = Vertex<VemanaData>;

pub struct Vamana {
    dataset: Arc<Dataset>,

    column: String,

    vertices: Vec<Vertex<VemanaData>>,
}

impl Vamana {
    /// Randomly initialize the graph.
    ///
    /// Parameters
    /// ----------
    ///  - dataset: the dataset to index
    ///  - r: the number of neighbors to connect to.
    ///  - rng: the random number generator.
    ///
    async fn try_init(
        dataset: Arc<Dataset>,
        column: &str,
        r: usize,
        mut rng: impl Rng,
    ) -> Result<Self> {
        let total = dataset.count_rows().await?;
        let scanner = dataset
            .scan()
            .with_row_id()
            .try_into_stream()
            .await
            .unwrap();

        let batches = scanner.try_collect::<Vec<_>>().await?;
        let mut vertices: Vec<VemanaVertex> = Vec::new();
        let mut vertex_id = 0;
        for batch in batches {
            let row_id = as_primitive_array::<UInt64Type>(
                batch
                    .column_by_qualified_name(ROW_ID)
                    .ok_or(Error::Index("row_id not found".to_string()))?,
            );
            for i in 0..row_id.len() {
                vertices.push(Vertex {
                    id: vertex_id,
                    neighbors: vec![],
                    aux_data: VemanaData {
                        row_id: row_id.value(i),
                    },
                });
                vertex_id += 1;
            }
        }
        let distribution = Uniform::new(0, total);
        // Randomly connect to r neighbors.
        for i in 0..vertices.len() {
            let mut neighbor_ids: HashSet<u32> = {
                let v = vertices.get(i).unwrap();
                v.neighbors.iter().cloned().collect()
            };

            while neighbor_ids.len() < r {
                let neighbor_id = rng.sample(distribution);
                if neighbor_id != i {
                    neighbor_ids.insert(neighbor_id as u32);
                }
            }

            // Make bidirectional connections.
            {
                let v = vertices.get_mut(i).unwrap();
                v.neighbors = neighbor_ids.iter().copied().collect();
            }
            {
                for neighbor_id in neighbor_ids.iter() {
                    let neighbor = vertices.get_mut(*neighbor_id as usize).unwrap();
                    neighbor.neighbors.push(i as u32);
                }
            }
        }

        Ok(Self {
            dataset,
            column: column.to_string(),
            vertices,
        })
    }

    /// Build Vamana Graph from a dataset.
    pub async fn try_new(
        dataset: Arc<Dataset>,
        column: &str,
        r: usize,
        alpha: f32,
    ) -> Result<Self> {
        let mut graph = Self::try_init(dataset.clone(), column, r, rand::thread_rng()).await?;
        Ok(graph)
    }

    /// Get the vector at the given row id.
    async fn get_vector(&self, row_id: usize) -> Result<Arc<Float32Array>> {
        let projection = self.dataset.schema().project(&[&self.column])?;
        let rows = self
            .dataset
            .take_rows(&[row_id as u64], &projection)
            .await?;
        if rows.num_rows() != 1 {
            return Err(Error::Index(format!(
                "expected 1 row, got {}",
                rows.num_rows()
            )));
        }
        let array = rows.column(0);
        let fs_array = as_fixed_size_list_array(array);
        let float_array = fs_array.value(0);
        let float_array: &Float32Array = as_primitive_array(float_array.as_ref());
        Ok(Arc::new(float_array.clone()))
    }

    /// Distance from the query vector to the vector at the given idx.
    async fn distance_to(&self, query: &Float32Array, idx: usize) -> Result<f32> {
        let vector = self.get_vector(idx).await?;
        let dists = l2_distance(query, vector.as_ref(), query.len())?;
        Ok(dists.values()[0])
    }

    /// Greedy search.
    ///
    /// Algorithm 1 in the paper.
    async fn greedy_search(
        &self,
        start: usize,
        query: &Float32Array,
        k: usize,
        queue_size: usize, // L in the paper.
    ) -> Result<Vec<usize>> {
        let mut visited: HashSet<usize> = HashSet::new();

        let mut candidates: BTreeMap<OrderedFloat<f32>, usize> = BTreeMap::new();
        let mut heap: BinaryHeap<VertexWithDistance> = BinaryHeap::new();
        heap.push(VertexWithDistance {
            id: start,
            distance: OrderedFloat(0.0),
        });
        while !heap.is_empty() {
            let p = heap.pop().unwrap();
            if visited.contains(&p.id) {
                continue;
            }
            visited.insert(p.id);
            for neighbor_id in self.neighbors(p.id).await?.iter() {
                if visited.contains(&neighbor_id) {
                    // Already visited.
                    continue;
                }
                let dist = self.distance_to(query, *neighbor_id).await?;
                heap.push(VertexWithDistance {
                    id: *neighbor_id,
                    distance: OrderedFloat(dist),
                });
                candidates.insert(OrderedFloat(dist), *neighbor_id);
                if candidates.len() > queue_size {
                    candidates.pop_last();
                }
            }
        }

        Ok(candidates.iter().take(k).map(|(_, id)| *id).collect())
    }

    /// Algorithm 2 in the paper.
    async fn prune(
        &self,
        id: usize,
        visited: &mut HashSet<usize>,
        alpha: f32,
        r: usize,
    ) -> Result<()> {
        todo!()
    }
}

#[async_trait]
impl Graph for Vamana {
    async fn distance(&self, a: usize, b: usize) -> Result<f32> {
        let row_id_a = self.vertices[a].aux_data.row_id;
        let row_id_b = self.vertices[b].aux_data.row_id;
        let vector_a = self.get_vector(row_id_a as usize).await.unwrap();
        let vector_b = self.get_vector(row_id_b as usize).await.unwrap();

        let dist = l2_distance(&vector_a, &vector_b, vector_a.len())?;
        Ok(dist.values()[0])
    }

    async fn neighbors(&self, id: usize) -> Result<Vec<usize>> {
        Ok(self.vertices[id]
            .neighbors
            .iter()
            .map(|id| *id as usize)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchReader};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use tempfile;

    use crate::arrow::*;
    use crate::dataset::WriteParams;
    use crate::utils::testing::generate_random_array;

    async fn create_dataset(uri: &str, n: usize, dim: usize) -> Arc<Dataset> {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Box::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            true,
        )]));
        let data = generate_random_array(n * dim);
        let batches = RecordBatchBuffer::new(vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(
                FixedSizeListArray::try_new(&data, dim as i32).unwrap(),
            )],
        )
        .unwrap()]);

        let mut write_params = WriteParams::default();
        write_params.max_rows_per_file = 40;
        write_params.max_rows_per_group = 10;
        let mut batches: Box<dyn RecordBatchReader> = Box::new(batches);
        Dataset::write(&mut batches, uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(uri).await.unwrap();
        Arc::new(dataset)
    }

    #[tokio::test]
    async fn test_init() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();
        let dataset = create_dataset(uri, 200, 64).await;

        let rng = rand::thread_rng();
        let inited_graph = Vamana::try_init(dataset, "vector", 10, rng).await.unwrap();

        for (vertex, id) in inited_graph.vertices.iter().zip(0..) {
            // Statistically， each node should have 10 neighbors.
            assert!(vertex.neighbors.len() > 0);
            assert_eq!(vertex.id, id);
        }
    }
}
