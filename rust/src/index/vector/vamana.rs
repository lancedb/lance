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
use std::iter::repeat;
use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow::datatypes::{Float32Type, UInt64Type};
use arrow_arith::arithmetic::{add, divide_scalar};
use arrow_array::RecordBatch;
use arrow_array::{cast::as_primitive_array, Array, Float32Array};
use arrow_schema::DataType;
use arrow_select::concat::concat;
use async_trait::async_trait;
use futures::{stream, StreamExt, TryStreamExt};
use ordered_float::OrderedFloat;
use rand::distributions::Uniform;
use rand::seq::SliceRandom;
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

/// Vamana Graph, described in DiskANN (NeurIPS' 19) and its following papers.
///
#[async_trait]
pub(crate) trait Vamana: Graph {}

pub struct VamanaBuilder {
    dataset: Arc<Dataset>,

    column: String,

    vertices: Vec<Vertex<VemanaData>>,

    /// For simplicity, we load all vectors into memory in the beginning.
    batch: RecordBatch,
}

impl VamanaBuilder {
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
        let stream = dataset
            .scan()
            .project(&[column])?
            .with_row_id()
            .try_into_stream()
            .await
            .unwrap();

        let batches = stream.try_collect::<Vec<_>>().await?;
        let batch = concat_batches(&batches[0].schema(), &batches)?;

        let row_ids = as_primitive_array::<UInt64Type>(
            batch
                .column_by_qualified_name(ROW_ID)
                .ok_or(Error::Index("row_id not found".to_string()))?,
        );
        let mut vertices: Vec<VemanaVertex> = row_ids
            .values()
            .iter()
            .enumerate()
            .map(|(i, &row_id)| Vertex {
                id: i as u32,
                neighbors: vec![],
                aux_data: VemanaData { row_id },
            })
            .collect();

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
            batch,
        })
    }

    fn dimension(&self) -> Result<usize> {
        let schema = self.dataset.schema();
        let field = schema
            .field(&self.column)
            .ok_or_else(|| Error::Index(format!("column {} not found in schema", self.column)))?;
        match field.data_type() {
            DataType::FixedSizeList(_, s) => Ok(s as usize),
            _ => Err(Error::Index(format!(
                "column {} is not a vector column: {}",
                self.column,
                field.data_type()
            ))),
        }
    }

    /// Find the closest vertex ID to the centroids.
    async fn find_medoid(&self) -> Result<usize> {
        let mut stream = self
            .dataset
            .scan()
            .project(&[&self.column])?
            .try_into_stream()
            .await
            .unwrap();

        // compute the centroids.
        // Can we use sample here instead?
        let mut total: usize = 0;
        let dim = self.dimension()?;
        let mut centroids = Float32Array::from_iter(repeat(0.0).take(dim));

        while let Some(batch) = stream.try_next().await? {
            total += batch.num_rows();
            let vector_col = batch.column_by_name(&self.column).ok_or_else(|| {
                Error::Index(format!("column {} not found in schema", self.column))
            })?;
            let vectors = as_fixed_size_list_array(vector_col.as_ref());
            for i in 0..vectors.len() {
                let vector = vectors.value(i);
                centroids = add(&centroids, as_primitive_array(vector.as_ref()))?;
            }
        }
        centroids = divide_scalar(&centroids, total as f32)?;

        // Find the closest vertex to the centroid.
        let medoid_id = {
            let mut stream = self
                .dataset
                .scan()
                .project(&[&self.column])?
                .try_into_stream()
                .await
                .unwrap();

            let distances = stream
                .map(|b| async {
                    let b = b?;
                    let vector_col = b.column_by_name(&self.column).ok_or_else(|| {
                        Error::Index(format!("column {} not found in schema", self.column))
                    })?;
                    let column = as_fixed_size_list_array(vector_col.as_ref());
                    let vectors: &Float32Array = as_primitive_array(column.values().as_ref());
                    let dists = l2_distance(&centroids, vectors, dim)?;
                    Ok::<Arc<Float32Array>, Error>(dists)
                })
                .buffered(num_cpus::get())
                .try_collect::<Vec<_>>()
                .await?;
            // For 1B vectors, the `distances` array is about `sizeof(f32) * 1B = 4GB`.
            let mut distance_refs: Vec<&dyn Array> = vec![];
            for d in distances.iter() {
                distance_refs.push(d.as_ref());
            }

            let distances = concat(&distance_refs)?;
            argmin(as_primitive_array::<Float32Type>(distances.as_ref())).unwrap()
        };

        Ok(medoid_id as usize)
    }

    async fn index_pass(
        &mut self,
        medoid: usize,
        alpha: f32,
        r: usize,
        l: usize,
        mut rng: impl Rng,
    ) -> Result<()> {
        let mut ids = (0..self.vertices.len()).collect::<Vec<_>>();
        ids.shuffle(&mut rng);

        for id in ids {
            let vector = self.get_vector(id).await?;
            let (_, visited) = self.greedy_search(medoid, vector.as_ref(), 1, l).await?;
            let now = std::time::Instant::now();
            self.robust_prune(id, visited, alpha, r).await?;
            // println!("prune time: {:?}", now.elapsed());
            for neighbor_id in self.neighbors(id).await? {
                let mut neighbor_set = HashSet::new();
                neighbor_set.extend(self.neighbors(neighbor_id).await?);
                neighbor_set.insert(id);
                if neighbor_set.len() > r {
                    self.robust_prune(neighbor_id, neighbor_set, alpha, r)
                        .await?;
                } else {
                    self.vertices[neighbor_id].neighbors.push(id as u32);
                }
            }
        }

        Ok(())
    }

    /// Build Vamana Graph from a dataset.
    pub async fn try_new(
        dataset: Arc<Dataset>,
        column: &str,
        r: usize,
        alpha: f32,
        l: usize,
    ) -> Result<Self> {
        let stream = dataset
            .scan()
            .project(&[column])?
            .with_row_id()
            .try_into_stream()
            .await?;

        let now = std::time::Instant::now();
        let mut graph = Self::try_init(dataset.clone(), column, r, rand::thread_rng()).await?;
        println!("Init graph: {}ms", now.elapsed().as_millis());

        let now = std::time::Instant::now();
        let medoid = graph.find_medoid().await?;
        println!("Find medoid: {}ms", now.elapsed().as_millis());

        let rng = rand::thread_rng();
        // First pass.
        let now = std::time::Instant::now();
        graph.index_pass(medoid, 1.0, r, l, rng.clone()).await?;
        println!("First pass: {}ms", now.elapsed().as_millis());
        // Second pass.
        let now = std::time::Instant::now();
        graph.index_pass(medoid, alpha, r, l, rng).await?;
        println!("Second pass: {}ms", now.elapsed().as_millis());

        Ok(graph)
    }

    /// Get the vector at an index.
    async fn get_vector(&self, idx: usize) -> Result<Arc<Float32Array>> {
        let vector_column = self.batch.column_by_name(&self.column).ok_or_else(|| {
            Error::Index(format!("column {} not found in batch", self.column))
        })?;
        let vectors = as_fixed_size_list_array(vector_column.as_ref());
        let vector_value = vectors.value(idx);
        let float_array: &Float32Array = as_primitive_array(vector_value.as_ref());
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
    ) -> Result<(Vec<usize>, HashSet<usize>)> {
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

        Ok((
            candidates.iter().take(k).map(|(_, id)| *id).collect(),
            visited,
        ))
    }

    /// Algorithm 2 in the paper.
    async fn robust_prune(
        &mut self,
        id: usize,
        mut visited: HashSet<usize>,
        alpha: f32,
        r: usize,
    ) -> Result<()> {
        visited.remove(&id);
        let neighbors = self.neighbors(id).await?;
        visited.extend(neighbors.iter());

        let mut ios: usize = 0;

        let mut heap: BinaryHeap<VertexWithDistance> = BinaryHeap::new();
        for p in visited.iter() {
            let dist = self.distance(id, *p).await?;
            ios += 2;
            heap.push(VertexWithDistance {
                id: *p,
                distance: OrderedFloat(dist),
            });
        }

        let mut new_neighbours: Vec<usize> = vec![];
        while !visited.is_empty() {
            let mut p = heap.pop().unwrap();
            while !visited.contains(&p.id) {
                // Because we are using a heap for `argmin(Visited)` in the original
                // algorithm, we need to pop out the vertices that are not in `visited` anymore.
                p = heap.pop().unwrap();
            }

            new_neighbours.push(p.id);
            if new_neighbours.len() >= r {
                break;
            }

            let mut to_remove: HashSet<usize> = HashSet::new();
            for pv in visited.iter() {
                let dist_prime = self.distance(p.id, *pv).await?;
                let dist_query = self.distance(id, *pv).await?;
                ios += 4;
                if alpha * dist_prime <= dist_query {
                    to_remove.insert(*pv);
                }
            }
            for pv in to_remove.iter() {
                visited.remove(pv);
            }
        }
        // println!("ios: {}", ios);
        self.vertices[id].neighbors = new_neighbours.iter().map(|id| *id as u32).collect();
        Ok(())
    }
}

#[async_trait]
impl Graph for VamanaBuilder {
    async fn distance(&self, a: usize, b: usize) -> Result<f32> {
        let vector_a = self.get_vector(a).await.unwrap();
        let vector_b = self.get_vector(b).await.unwrap();

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

/// Vamana Graph implementation for Vamana.
impl Vamana for VamanaBuilder {}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchReader};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use tempfile;

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
        let inited_graph = VamanaBuilder::try_init(dataset, "vector", 10, rng)
            .await
            .unwrap();

        for (vertex, id) in inited_graph.vertices.iter().zip(0..) {
            // Statistically， each node should have 10 neighbors.
            assert!(vertex.neighbors.len() > 0);
            assert_eq!(vertex.id, id);
        }
    }

    #[tokio::test]
    async fn test_build_index() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let uri = tmp_dir.path().to_str().unwrap();
        let dataset = create_dataset(uri, 200, 64).await;

        let graph = VamanaBuilder::try_new(dataset, "vector", 50, 1.4, 100)
            .await
            .unwrap();
    }
}
