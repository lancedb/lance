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

//! Graph in memory.

use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::UInt32Array;
use async_trait::async_trait;
use lance_linalg::distance::{DistanceFunc, MetricType};
use lance_linalg::matrix::MatrixView;

use super::{Graph, Vertex};
use crate::{Error, Result};

/// A graph node to hold the vertex data and its neighbors.
#[derive(Debug)]
pub struct Node<V: Vertex> {
    /// The vertex metadata. will be serialized into fixed size binary in the persisted graph.
    pub(crate) vertex: V,

    /// Neighbors are the ids of vertex in the graph.
    /// This id is not the same as the row_id in the original lance dataset.
    pub(crate) neighbors: Arc<UInt32Array>,
}

/// A Graph that allows dynamically build graph to be persisted later.
///
/// It requires all vertices to be of the same size.
pub struct GraphBuilder<V: Vertex + Clone + Sync + Send> {
    pub(crate) nodes: Vec<Node<V>>,

    /// Hold all vectors in memory for fast access at the moment.
    pub(crate) data: MatrixView<Float32Type>,

    /// Metric type.
    metric_type: MetricType,

    /// Distance function.
    distance_func: Arc<DistanceFunc>,
}

impl<V: Vertex + Clone + Sync + Send> GraphBuilder<V> {
    pub fn new(vertices: &[V], data: MatrixView<Float32Type>, metric_type: MetricType) -> Self {
        Self {
            nodes: vertices
                .iter()
                .map(|v| Node {
                    vertex: v.clone(),
                    neighbors: Arc::new(UInt32Array::from(vec![] as Vec<u32>)),
                })
                .collect(),
            data,
            metric_type,
            distance_func: metric_type.func(),
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn vertex(&self, id: usize) -> &V {
        &self.nodes[id].vertex
    }

    pub fn vertex_mut(&mut self, id: usize) -> &mut V {
        &mut self.nodes[id].vertex
    }

    /// Set neighbors of a node.
    pub fn set_neighbors(&mut self, id: usize, neighbors: Arc<UInt32Array>) {
        self.nodes[id].neighbors = neighbors;
    }
}

#[async_trait]
impl<V: Vertex + Clone + Sync + Send> Graph for GraphBuilder<V> {
    async fn distance(&self, a: usize, b: usize) -> Result<f32> {
        let vector_a = self.data.row(a).ok_or_else(|| Error::Index {
            message: format!(
                "Vector index is out of range: {} >= {}",
                a,
                self.data.num_rows()
            ),
        })?;

        let vector_b = self.data.row(b).ok_or_else(|| Error::Index {
            message: format!(
                "Vector index is out of range: {} >= {}",
                b,
                self.data.num_rows()
            ),
        })?;
        Ok((self.distance_func)(vector_a, vector_b))
    }

    async fn distance_to(&self, query: &[f32], idx: usize) -> Result<f32> {
        let vector = self.data.row(idx).ok_or_else(|| Error::Index {
            message: format!(
                "Attempt to access row {} in a matrix with {} rows",
                idx,
                self.data.num_rows()
            ),
        })?;
        Ok((self.distance_func)(query, vector))
    }

    async fn neighbors(&self, id: usize) -> Result<Arc<UInt32Array>> {
        Ok(self.nodes[id].neighbors.clone())
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[derive(Debug, Clone)]
    struct FooVertex {
        id: u32,
        val: f32,
    }

    impl Vertex for FooVertex {
        fn vector(&self) -> &[f32] {
            todo!()
        }

        fn as_any(&self) -> &dyn std::any::Any {
            todo!()
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            todo!()
        }
    }

    #[tokio::test]
    async fn test_construct_builder() {
        let nodes = (0..100)
            .map(|v| FooVertex {
                id: v as u32,
                val: v as f32 * 0.5,
            })
            .collect::<Vec<_>>();
        let mut builder = GraphBuilder::new(&nodes, MatrixView::random(100, 32), MetricType::L2);

        assert_eq!(builder.len(), 100);
        assert_eq!(builder.vertex(77).id, 77);
        assert_relative_eq!(builder.vertex(77).val, 38.5);
        assert!(builder.neighbors(55).await.unwrap().is_empty());

        builder.vertex_mut(88).val = 22.0;
        assert_relative_eq!(builder.vertex(88).val, 22.0);
    }
}
