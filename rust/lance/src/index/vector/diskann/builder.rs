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

use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc;

use arrow_array::{
    cast::AsArray,
    types::{Float32Type, UInt64Type},
    UInt32Array,
};
use arrow_select::concat::concat_batches;
use futures::stream::{self, StreamExt, TryStreamExt};
use lance_arrow::*;
use lance_core::io::WriteExt;
use lance_linalg::kernels::argmin;
use lance_linalg::{
    distance::{cosine_distance_batch, dot_distance_batch, l2_distance, l2_distance_batch},
    matrix::MatrixView,
};
use ordered_float::OrderedFloat;
use rand::{distributions::Uniform, prelude::*, Rng, SeedableRng};
use snafu::{location, Location};

use crate::dataset::{Dataset, ROW_ID};
use crate::index::pb;
use crate::index::vector::diskann::row_vertex::RowVertexSerDe;
use crate::index::vector::diskann::DiskANNParams;
use crate::index::vector::graph::{
    builder::GraphBuilder, write_graph, VertexWithDistance, WriteGraphParams,
};
use crate::index::vector::graph::{Graph, Vertex};
use crate::index::vector::{MetricType, INDEX_FILE_NAME};
use crate::{Error, Result};

use super::row_vertex::RowVertex;
use super::search::greedy_search;

pub async fn build_diskann_index(
    dataset: &Dataset,
    column: &str,
    name: &str,
    uuid: &str,
    params: DiskANNParams,
) -> Result<()> {
    let rng = rand::rngs::SmallRng::from_entropy();

    // Randomly initialize the graph with r random neighbors for each vertex.
    let mut graph = init_graph(dataset, column, params.r, params.metric_type, rng.clone()).await?;

    // Find medoid
    let medoid = {
        let vectors = graph.data.clone();
        find_medoid(&vectors, params.metric_type).await?
    };

    // First pass.
    let now = std::time::Instant::now();
    index_once(&mut graph, medoid, 1.0, params.r, params.l, rng.clone()).await?;
    println!("DiskANN: first pass: {}s", now.elapsed().as_secs_f32());
    // Second pass.
    let now = std::time::Instant::now();
    index_once(
        &mut graph,
        medoid,
        params.alpha,
        params.r,
        params.l,
        rng.clone(),
    )
    .await?;
    println!("DiskANN: second pass: {}s", now.elapsed().as_secs_f32());

    let index_dir = dataset.indices_dir().child(uuid);
    let filename = "diskann_graph.lance";
    let graph_file = index_dir.child(filename);

    let write_params = WriteGraphParams {
        batch_size: 2048 * 10,
    };
    let serde = RowVertexSerDe {};

    write_graph(
        &graph,
        dataset.object_store(),
        &graph_file,
        &write_params,
        &serde,
    )
    .await?;

    write_index_file(
        dataset,
        column,
        name,
        uuid,
        graph.data.num_columns(),
        filename,
        &[medoid],
        params.metric_type,
        &params,
    )
    .await?;

    Ok(())
}

/// Randomly initialize the graph with r random neighbors for each vertex.
///
/// Parameters
/// ----------
///  - dataset: the dataset to index.
///  - column: the vector column to index.
///  - r: the number of neighbors to connect to.
///  - rng: the random number generator.
///
async fn init_graph(
    dataset: &Dataset,
    column: &str,
    r: usize,
    metric_type: MetricType,
    mut rng: impl Rng,
) -> Result<GraphBuilder<RowVertex>> {
    let stream = dataset
        .scan()
        .project(&[column])?
        .with_row_id()
        .try_into_stream()
        .await
        .unwrap();

    let batches = stream.try_collect::<Vec<_>>().await?;
    let batch = concat_batches(&batches[0].schema(), &batches)?;

    let row_ids = batch
        .column_by_qualified_name(ROW_ID)
        .ok_or(Error::Index {
            message: "row_id not found".to_string(),
            location: location!(),
        })?
        .as_primitive::<UInt64Type>();
    let vectors = batch
        .column_by_qualified_name(column)
        .ok_or(Error::Index {
            message: format!("column {} not found", column),
            location: location!(),
        })?
        .as_fixed_size_list();
    let matrix = MatrixView::<Float32Type>::try_from(vectors)?;
    let nodes = row_ids
        .values()
        .iter()
        .map(|&row_id| RowVertex::new(row_id, None))
        .collect::<Vec<_>>();
    let mut graph = GraphBuilder::new(&nodes, matrix, metric_type);

    let distribution = Uniform::new(0, batch.num_rows());
    // Randomly connect to r neighbors.
    for i in 0..graph.len() {
        let mut neighbor_ids: HashSet<u32> =
            graph.neighbors(i).await?.values().iter().copied().collect();

        while neighbor_ids.len() < r {
            let neighbor_id = rng.sample(distribution);
            if neighbor_id != i {
                neighbor_ids.insert(neighbor_id as u32);
            }
        }

        {
            let new_neighbors = Arc::new(UInt32Array::from_iter(neighbor_ids.iter().copied()));
            graph.set_neighbors(i, new_neighbors);
        }
    }

    Ok(graph)
}

/// Distance between two vectors in the matrix.
fn distance(matrix: &MatrixView<Float32Type>, i: usize, j: usize) -> Result<f32> {
    let vector_i = matrix.row(i).ok_or(Error::Index {
        message: "Invalid row index".to_string(),
        location: location!(),
    })?;
    let vector_j = matrix.row(j).ok_or(Error::Index {
        message: "Invalid row index".to_string(),
        location: location!(),
    })?;

    Ok(l2_distance(vector_i, vector_j))
}

/// Algorithm 2 in the paper.
async fn robust_prune<V: Vertex + Clone + Sync + Send>(
    graph: &GraphBuilder<V>,
    id: usize,
    mut visited: HashSet<usize>,
    alpha: f32,
    r: usize,
) -> Result<Vec<u32>> {
    visited.remove(&id);
    let neighbors = graph.neighbors(id).await?;
    visited.extend(neighbors.values().iter().map(|id| *id as usize));

    let mut heap: BinaryHeap<VertexWithDistance> = visited
        .iter()
        .map(|v| {
            let dist = distance(&graph.data, id, *v).unwrap();
            VertexWithDistance {
                id: *v,
                distance: OrderedFloat(dist),
            }
        })
        .collect();

    let matrix = graph.data.clone();
    let new_neighbours = tokio::task::spawn_blocking(move || {
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
                let dist_prime = distance(&matrix, p.id, *pv)?;
                let dist_query = distance(&matrix, id, *pv)?;
                if alpha * dist_prime <= dist_query {
                    to_remove.insert(*pv);
                }
            }
            for pv in to_remove.iter() {
                visited.remove(pv);
            }
        }
        Ok::<_, Error>(new_neighbours)
    })
    .await??;

    Ok(new_neighbours.iter().map(|id| *id as u32).collect())
}

/// Find the index of the medoid vector in all vectors.
async fn find_medoid(vectors: &MatrixView<Float32Type>, metric_type: MetricType) -> Result<usize> {
    let centroid = vectors.centroid().ok_or_else(|| Error::Index {
        message: "Cannot find the medoid of an empty matrix".to_string(),
        location: location!(),
    })?;

    // Find the closest vertex to the centroid.
    let centrodis = centroid.values();
    let data = vectors.data();
    let dim = vectors.ndim();
    let dists = match metric_type {
        MetricType::L2 => l2_distance_batch(centrodis, data.values(), dim),
        MetricType::Cosine => cosine_distance_batch(centrodis, data.values(), dim),
        MetricType::Dot => dot_distance_batch(centrodis, data.values(), dim),
    };
    let medoid_idx = argmin(dists).unwrap();
    Ok(medoid_idx as usize)
}

/// One pass of index building.
async fn index_once<V: Vertex + Clone + Sync + Send>(
    graph: &mut GraphBuilder<V>,
    medoid: usize,
    alpha: f32,
    r: usize,
    l: usize,
    mut rng: impl Rng,
) -> Result<()> {
    let mut ids = (0..graph.len()).collect::<Vec<_>>();
    ids.shuffle(&mut rng);

    for (i, &id) in ids.iter().enumerate() {
        let vector = graph.data.row(i).ok_or_else(|| Error::Index {
            message: format!("Cannot find vector with id {}", id),
            location: location!(),
        })?;

        let state = greedy_search(graph, medoid, vector, 1, l).await?;

        let neighbors = robust_prune(graph, id, state.visited, alpha, r).await?;
        graph.set_neighbors(
            id,
            Arc::new(neighbors.iter().copied().collect::<UInt32Array>()),
        );

        let fixed_graph: &GraphBuilder<V> = graph;
        let neighbours = stream::iter(neighbors)
            .map(|j| async move {
                let mut neighbor_set: HashSet<usize> = fixed_graph
                    .neighbors(j as usize)
                    .await?
                    .values()
                    .iter()
                    .map(|v| *v as usize)
                    .collect();
                neighbor_set.insert(id);
                if neighbor_set.len() + 1 > r {
                    let new_neighbours =
                        robust_prune(fixed_graph, j as usize, neighbor_set, alpha, r).await?;
                    Ok::<_, Error>((j as usize, new_neighbours))
                } else {
                    Ok::<_, Error>((
                        j as usize,
                        neighbor_set.iter().map(|n| *n as u32).collect::<Vec<_>>(),
                    ))
                }
            })
            .buffered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await?;
        for (j, nbs) in neighbours {
            graph.set_neighbors(j, Arc::new(nbs.into_iter().collect::<UInt32Array>()));
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn write_index_file(
    dataset: &Dataset,
    column: &str,
    index_name: &str,
    uuid: &str,
    dimension: usize,
    graph_file: &str,
    entries: &[usize],
    metric_type: MetricType,
    params: &DiskANNParams,
) -> Result<()> {
    let object_store = dataset.object_store();
    let path = dataset.indices_dir().child(uuid).child(INDEX_FILE_NAME);
    let mut writer = object_store.create(&path).await?;

    let stages: Vec<pb::VectorIndexStage> = vec![pb::VectorIndexStage {
        stage: Some(pb::vector_index_stage::Stage::Diskann(pb::DiskAnn {
            spec: 1,
            filename: graph_file.to_string(),
            r: params.r as u32,
            alpha: params.alpha,
            l: params.l as u32,
            entries: entries.iter().map(|v| *v as u64).collect(),
        })),
    }];
    let metadata = pb::Index {
        name: index_name.to_string(),
        columns: vec![column.to_string()],
        dataset_version: dataset.version().version,
        index_type: pb::IndexType::Vector.into(),
        implementation: Some(pb::index::Implementation::VectorIndex(pb::VectorIndex {
            spec_version: 1,
            dimension: dimension as u32,
            stages,
            metric_type: match metric_type {
                MetricType::L2 => pb::VectorMetricType::L2.into(),
                MetricType::Cosine => pb::VectorMetricType::Cosine.into(),
                MetricType::Dot => pb::VectorMetricType::Dot.into(),
            },
        })),
    };

    let pos = writer.write_protobuf(&metadata).await?;
    writer.write_magics(pos).await?;
    writer.shutdown().await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use tempfile;

    use crate::dataset::WriteParams;
    use lance_testing::datagen::generate_random_array;

    async fn create_dataset(uri: &str, n: usize, dim: usize) -> Arc<Dataset> {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            true,
        )]));
        let data = generate_random_array(n * dim);
        let batches = vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(
                FixedSizeListArray::try_new_from_values(data, dim as i32).unwrap(),
            )],
        )
        .unwrap()];

        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, uri, Some(write_params))
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
        let graph = init_graph(dataset.as_ref(), "vector", 10, MetricType::L2, rng)
            .await
            .unwrap();

        for (id, node) in graph.nodes.iter().enumerate() {
            // Statistically， each node should have 10 neighbors.
            assert!(!node.neighbors.is_empty());
            // For the first fragment, the IDs should match.
            if id < 40 {
                assert_eq!(node.vertex.row_id as usize, id);
            }
        }
    }
}
