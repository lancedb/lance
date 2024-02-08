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

use super::{Graph, GraphNode, OrderedFloat};
use lance_core::Result;
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
    /// Returns the list of pruned neighbors.
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

impl From<GraphBuilderNode> for GraphNode<u32> {
    fn from(node: GraphBuilderNode) -> Self {
        GraphNode {
            id: node.id,
            neighbors: node.neighbors.values().copied().collect(),
        }
    }
}

pub struct GraphBuilder {
    nodes: BTreeMap<u32, GraphBuilderNode>,
    vectors: Box<dyn VectorStorage<f32>>
}

impl GraphBuilder {
    pub fn new(vectors: Box<dyn VectorStorage<f32>>) -> Self {
        Self {
            nodes: BTreeMap::new(),
            vectors,
        }
    }

    /// Insert a node into the graph.
    pub fn insert(&mut self, node: u32) {
        self.nodes.insert(node, GraphBuilderNode::new(node));
    }

    /// Connect from one node to another.
    pub fn connect(&mut self, from: u32, to: u32) -> Result<()> {
        let from = self.nodes.get_mut(&from).unwrap();
        
        from.neighbors.insert(to);
        Ok(())
    }

    /// Bidirectionally connect two nodes.
    pub fn bi_connect(&mut self, from: u32, to: u32) -> Result<()> {
        self.connect(from, to)?;
        self.connect(to, from)
    }

    pub fn prune(&mut self, node: u32, max_edges: usize) {}

    /// Build the Graph.
    pub fn build(self) -> Graph {
        Graph::from_nodes(self.nodes.iter().map(|(id, node)| (id, node.into())).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder() {
        let mut builder = GraphBuilder::new();
        builder.insert(0);
        builder.insert(1);
        builder.connect(0, 1);
        let graph = builder.build();
        assert_eq!(graph.len(), 2);
    }
}
