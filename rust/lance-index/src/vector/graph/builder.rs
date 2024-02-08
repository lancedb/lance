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

use std::collections::BTreeMap;

use lance_core::{Error, Result};
use lance_linalg::distance::{DistanceFunc, MetricType};
use snafu::{location, Location};

use super::{Graph, GraphNode, InMemoryGraph, OrderedFloat};
use crate::vector::graph::storage::VectorStorage;

/// GraphNode during build.
struct GraphBuilderNode {
    /// Node ID
    id: u32,
    /// Neighbors, sorted by the distance.
    neighbors: BTreeMap<OrderedFloat, u32>,
}

impl GraphBuilderNode {
    fn new(id: u32) -> Self {
        Self {
            id,
            neighbors: BTreeMap::new(),
        }
    }

    /// Prune the node and only keep `max_edges` edges.
    ///
    /// Returns the ids of pruned neighbors.
    fn prune(&mut self, max_edges: usize) -> Vec<u32> {
        if self.neighbors.len() <= max_edges {
            return vec![];
        }

        let mut pruned = Vec::with_capacity(self.neighbors.len() - max_edges);
        while self.neighbors.len() > max_edges {
            let (_, node) = self.neighbors.pop_last().unwrap();
            pruned.push(node)
        }
        pruned
    }
}

impl From<&GraphBuilderNode> for GraphNode<u32> {
    fn from(node: &GraphBuilderNode) -> Self {
        GraphNode {
            id: node.id,
            neighbors: node.neighbors.values().copied().collect(),
        }
    }
}

pub struct GraphBuilder<V: VectorStorage<f32>> {
    nodes: BTreeMap<u32, GraphBuilderNode>,
    vectors: V,

    dist_fn: Box<DistanceFunc>,
}

impl<V: VectorStorage<f32>> Graph<u32, f32> for GraphBuilder<V> {
    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn neighbors(&self, key: u32) -> Option<Box<dyn Iterator<Item = u32> + '_>> {
        let node = self.nodes.get(&key)?;
        Some(Box::new(node.neighbors.values().into_iter().copied()))
    }

    fn distance_to(&self, query: &[f32], key: u32) -> f32 {
        let vec = self.vectors.get(key as usize);
        (self.dist_fn)(query, vec)
    }

    fn distance_between(&self, a: u32, b: u32) -> f32 {
        let from_vec = self.vectors.get(a as usize);
        let to_vec = self.vectors.get(b as usize);
        (self.dist_fn)(from_vec, to_vec)
    }
}

impl<V: VectorStorage<f32>> GraphBuilder<V> {
    /// Build from a [VectorStorage].
    pub fn new(vectors: V) -> Self {
        Self {
            nodes: BTreeMap::new(),
            vectors,
            dist_fn: MetricType::L2.func().into(),
        }
    }

    /// Insert a node into the graph.
    pub fn insert(&mut self, node: u32) {
        self.nodes.insert(node, GraphBuilderNode::new(node));
    }

    /// Connect from one node to another.
    pub fn connect(&mut self, from: u32, to: u32) -> Result<()> {
        let distance: OrderedFloat = self.distance_between(from, to).into();

        {
            let from_node = self.nodes.get_mut(&from).ok_or_else(|| Error::Index {
                message: format!("Node {} not found", from),
                location: location!(),
            })?;
            from_node.neighbors.insert(distance, to);
        }

        {
            let to_node = self.nodes.get_mut(&to).ok_or_else(|| Error::Index {
                message: format!("Node {} not found", to),
                location: location!(),
            })?;
            to_node.neighbors.insert(distance, from);
        }
        Ok(())
    }

    /// Bidirectionally connect two nodes.
    pub fn bi_connect(&mut self, from: u32, to: u32) -> Result<()> {
        let distance: OrderedFloat = self.distance_between(from, to).into();

        self.connect(from, to)?;
        self.connect(to, from)
    }

    pub fn prune(&mut self, node: u32, max_edges: usize) -> Result<()> {
        let node = self.nodes.get_mut(&node).ok_or_else(|| Error::Index {
            message: format!("Node {} not found", node),
            location: location!(),
        })?;
        node.prune(max_edges);
        Ok(())
    }

    /// Build the Graph.
    pub fn build(self) -> Box<dyn Graph> {
        Box::new(InMemoryGraph::from_nodes(
            self.nodes
                .iter()
                .map(|(&id, node)| (id, node.into()))
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::{types::Float32Type, Float32Array};
    use lance_linalg::MatrixView;

    #[test]
    fn test_builder() {
        let arr = Float32Array::from_iter_values((0..120).map(|v| v as f32));
        let mat = MatrixView::<Float32Type>::new(Arc::new(arr), 8);
        let mut builder = GraphBuilder::new(mat);
        builder.insert(0);
        builder.insert(1);
        builder.connect(0, 1).unwrap();
        let graph = builder.build();
        assert_eq!(graph.len(), 2);
    }
}
