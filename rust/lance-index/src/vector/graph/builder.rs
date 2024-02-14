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

use arrow_array::UInt64Array;
use lance_core::{Error, Result};
use lance_linalg::distance::{DistanceFunc, MetricType};
use snafu::{location, Location};

use super::{Graph, InMemoryGraph, OrderedFloat};
use crate::vector::graph::storage::VectorStorage;

/// GraphNode during build.
pub(super) struct GraphBuilderNode {
    /// Node ID
    pub(super) id: u64,
    /// Neighbors, sorted by the distance.
    pub(super) neighbors: BTreeMap<OrderedFloat, u64>,

    pub(super) row_id: u64,
}

impl GraphBuilderNode {
    pub(crate) fn new(id: u64) -> Self {
        Self {
            id,
            neighbors: BTreeMap::new(),
            row_id: 0,
        }
    }

    /// Prune the node and only keep `max_edges` edges.
    ///
    /// Returns the ids of pruned neighbors.
    fn prune(&mut self, max_edges: usize) -> Vec<u64> {
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

/// Graph Builder.
///
///
pub struct GraphBuilder<V: VectorStorage<f32>> {
    nodes: BTreeMap<u64, GraphBuilderNode>,

    /// Storage for vectors.
    vectors: V,

    metric_type: MetricType,

    dist_fn: Box<DistanceFunc<f32>>,
}

impl<V: VectorStorage<f32>> Graph<f32> for GraphBuilder<V> {
    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn neighbors(&self, key: u64) -> Option<Arc<UInt64Array>> {
        let node = self.nodes.get(&key)?;
        Some(Arc::new(node.neighbors.values().copied()))
    }

    fn distance_to(&self, query: &[f32], key: u64) -> f32 {
        let vec = self.vectors.get(key as usize);
        (self.dist_fn)(query, vec)
    }

    fn distance_between(&self, a: u64, b: u64) -> f32 {
        let from_vec = self.vectors.get(a as usize);
        let to_vec = self.vectors.get(b as usize);
        (self.dist_fn)(from_vec, to_vec)
    }

    fn iter(&self) -> super::iter::Iter<'_, f32> {
        todo!()
    }
}

impl<V: VectorStorage<f32> + 'static> GraphBuilder<V> {
    /// Build from a [VectorStorage].
    pub fn new(vectors: V) -> Self {
        Self {
            nodes: BTreeMap::new(),
            vectors,
            dist_fn: Box::new(MetricType::L2.func::<f32>() as DistanceFunc<f32>),
            metric_type: MetricType::L2,
        }
    }

    /// Set metric type
    pub fn metric_type(mut self, metric_type: MetricType) -> Self {
        self.metric_type = metric_type;
        self.dist_fn = Box::new(metric_type.func() as DistanceFunc<f32>);
        self
    }

    /// Insert a node into the graph.
    pub fn insert(&mut self, node: u64) {
        self.nodes.insert(node, GraphBuilderNode::new(node));
    }

    /// Connect from one node to another.
    pub fn connect(&mut self, from: u64, to: u64) -> Result<()> {
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

    pub fn prune(&mut self, node: u64, max_edges: usize) -> Result<()> {
        let node = self.nodes.get_mut(&node).ok_or_else(|| Error::Index {
            message: format!("Node {} not found", node),
            location: location!(),
        })?;
        node.prune(max_edges);
        Ok(())
    }

    /// Build the Graph.
    pub fn build(&self) -> Box<dyn Graph> {
        Box::new(InMemoryGraph::from_builder(
            &self.nodes,
            self.vectors.clone(),
            self.metric_type,
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

        assert_eq!(graph.neighbors(0).unwrap().collect::<Vec<_>>(), vec![1]);
        assert_eq!(graph.neighbors(1).unwrap().collect::<Vec<_>>(), vec![0]);
    }
}
