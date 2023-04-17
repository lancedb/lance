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

use super::{Graph, Vertex};
use crate::arrow::linalg::MatrixView;
use crate::{Result, Error};
use crate::index::vector::MetricType;

/// A graph node to hold the vertex data and its neighbors.
#[derive(Debug)]
pub(crate) struct Node<V: Vertex> {
    /// The vertex metadata. will be serialized into fixed size binary in the persisted graph.
    pub(crate) vertex: V,

    /// Neighbors are the ids of vertex in the graph.
    /// This id is not the same as the row_id in the original lance dataset.
    pub(crate) neighbors: Vec<u32>,
}

/// A Graph that allows dynamically build graph to be persisted later.
///
/// It requires all vertices to be of the same size.
pub(crate) struct GraphBuilder<V: Vertex> {
    pub(crate) nodes: Vec<Node<V>>,

    /// Hold all vectors in memory for fast access at the moment.
    pub(crate) data: MatrixView,

    /// Metric type.
    metric_type: MetricType,

    /// Distance function.
    distance_func: Arc<dyn Fn(&[f32], &[f32]) -> f32>,
}

impl<V: Vertex> GraphBuilder<V> {
    pub fn new(nodes: Vec<V>, data: MatrixView, metric_type: MetricType) -> Self {
        Self {
            nodes: vec![],
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

    pub fn neighbors(&self, id: usize) -> &[u32] {
        self.nodes[id].neighbors.as_slice()
    }

    pub fn neighbors_mut(&mut self, id: usize) -> &mut Vec<u32> {
        &mut self.nodes[id].neighbors
    }

    /// Set neighbors of a node.
    pub fn set_neighbors(&mut self, id: usize, neighbors: impl Into<Vec<u32>>) {
        self.nodes[id].neighbors = neighbors.into();
    }

    /// Add a neighbor to a specific vertex.
    pub fn add_neighbor(&mut self, vertex: usize, neighbor: usize) {
        self.nodes[vertex].neighbors.push(neighbor as u32);
    }
}

impl<V: Vertex> Graph for GraphBuilder<V> {
    fn distance(&self, a: usize, b: usize) -> Result<f32> {
        let vector_a = self.data.row(a).ok_or_else(|| {
            Error::Index(format!(
                "Attempt to access row {} in a matrix with {} rows",
                a,
                self.data.num_rows()
            ))
        })?;

        let vector_b = self.data.row(b).ok_or_else(|| {
            format!(
                "Attempt to access row {} in a matrix with {} rows",
                b,
                self.data.num_rows()
            )
        })?;
        Ok(self.data.row(a).dot(&self.data.row(b)))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    struct FooVertex {
        id: u32,
        val: f32,
    }

    impl Vertex for FooVertex {}

    #[test]
    fn test_construct_builder() {
        let mut builder: GraphBuilder<FooVertex> = (0..100)
            .map(|v| FooVertex {
                id: v as u32,
                val: v as f32 * 0.5,
            })
            .collect();

        assert_eq!(builder.len(), 100);
        assert_eq!(builder.vertex(77).id, 77);
        assert_relative_eq!(builder.vertex(77).val, 38.5);
        assert!(builder.neighbors(55).is_empty());

        builder.vertex_mut(88).val = 22.0;
        assert_relative_eq!(builder.vertex(88).val, 22.0);
    }
}
