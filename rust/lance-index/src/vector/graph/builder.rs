// Copyright 2024 Lance Developers.
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

use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use lance_core::{Error, Result};
use snafu::{location, Location};

use super::OrderedNode;
use super::{memory::InMemoryVectorStorage, Graph, GraphNode, OrderedFloat};
use crate::vector::graph::storage::VectorStorage;

/// GraphNode during build.
#[derive(Debug)]
pub struct GraphBuilderNode {
    /// Node ID
    pub(crate) id: u32,

    /// Neighbors, sorted by the distance.
    pub(crate) neighbors: BinaryHeap<OrderedNode>,

    /// Pointer to the next level of graph, or acts as the idx
    pub pointer: u32,
}

impl GraphBuilderNode {
    fn new(id: u32) -> Self {
        Self {
            id,
            neighbors: BinaryHeap::new(),
            pointer: 0,
        }
    }

    fn add_neighbor(&mut self, distance: f32, id: u32) {
        self.neighbors.push(OrderedNode {
            dist: OrderedFloat(distance),
            id,
        });
    }

    /// Prune the node and only keep `max_edges` edges.
    ///
    /// Returns the ids of pruned neighbors.
    fn prune(&mut self, max_edges: usize) {
        while self.neighbors.len() > max_edges {
            self.neighbors.pop();
        }
    }
}

impl From<&GraphBuilderNode> for GraphNode<u32> {
    fn from(node: &GraphBuilderNode) -> Self {
        let neighbors = node
            .neighbors
            .clone()
            .into_sorted_vec()
            .into_iter()
            .map(|n| n.id)
            .collect::<Vec<_>>();
        Self {
            id: node.id,
            neighbors,
        }
    }
}

/// Graph Builder.
///
/// [GraphBuilder] is used to build a graph in memory.
///
pub struct GraphBuilder {
    pub(crate) nodes: HashMap<u32, GraphBuilderNode>,

    /// Storage for vectors.
    vectors: Arc<InMemoryVectorStorage>,
}

impl Graph for GraphBuilder {
    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn neighbors(&self, key: u32) -> Option<Box<dyn Iterator<Item = u32> + '_>> {
        let node = self.nodes.get(&key)?;
        Some(Box::new(
            node.neighbors
                .clone()
                .into_sorted_vec()
                .into_iter()
                .map(|n| n.id),
        ))
    }

    fn storage(&self) -> Arc<dyn VectorStorage> {
        self.vectors.clone() as Arc<dyn VectorStorage>
    }
}

impl GraphBuilder {
    /// Build from a [VectorStorage].
    pub fn new(vectors: Arc<InMemoryVectorStorage>) -> Self {
        Self {
            nodes: HashMap::new(),
            vectors,
        }
    }

    /// Insert a node into the graph.
    pub fn insert(&mut self, node: u32) {
        self.nodes.insert(node, GraphBuilderNode::new(node));
    }

    /// Connect from one node to another.
    pub fn connect(&mut self, from: u32, to: u32) -> Result<()> {
        let distance = self.vectors.distance_between(from, to);

        {
            let from_node = self.nodes.get_mut(&from).ok_or_else(|| Error::Index {
                message: format!("Node {} not found", from),
                location: location!(),
            })?;
            from_node.add_neighbor(distance, to)
        }

        {
            let to_node = self.nodes.get_mut(&to).ok_or_else(|| Error::Index {
                message: format!("Node {} not found", to),
                location: location!(),
            })?;
            to_node.add_neighbor(distance, from);
        }
        Ok(())
    }

    pub fn prune(&mut self, node: u32, max_edges: usize) -> Result<()> {
        let node = self.nodes.get_mut(&node).ok_or_else(|| Error::Index {
            message: format!("Node {} not found", node),
            location: location!(),
        })?;
        node.prune(max_edges);
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn stats(&self) -> GraphBuilderStats {
        let mut max_edges = 0;
        let mut total_edges = 0;
        let mut total_distance = 0.0;

        for node in self.nodes.values() {
            let edges = node.neighbors.len();
            total_edges += edges;
            max_edges = max_edges.max(edges);
            total_distance += node.neighbors.iter().map(|n| n.dist.0).sum::<f32>();
        }

        GraphBuilderStats {
            num_nodes: self.nodes.len(),
            max_edges,
            mean_edges: total_edges as f32 / self.nodes.len() as f32,
            mean_distance: total_distance / total_edges as f32,
        }
    }
}

#[derive(Debug)]
pub struct GraphBuilderStats {
    pub num_nodes: usize,
    pub max_edges: usize,
    pub mean_edges: f32,
    pub mean_distance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::{types::Float32Type, Float32Array};
    use lance_linalg::{distance::MetricType, MatrixView};

    #[test]
    fn test_builder() {
        let arr = Float32Array::from_iter_values((0..120).map(|v| v as f32));
        let mat = Arc::new(MatrixView::<Float32Type>::new(Arc::new(arr), 8));
        let store = Arc::new(InMemoryVectorStorage::new(mat, MetricType::L2));
        let mut builder = GraphBuilder::new(store.clone());
        builder.insert(0);
        builder.insert(1);
        builder.connect(0, 1).unwrap();
        assert_eq!(builder.len(), 2);

        assert_eq!(builder.neighbors(0).unwrap().collect::<Vec<_>>(), vec![1]);
        assert_eq!(builder.neighbors(1).unwrap().collect::<Vec<_>>(), vec![0]);
    }
}
