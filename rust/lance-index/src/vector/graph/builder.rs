// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::BinaryHeap;

use super::OrderedFloat;
use super::OrderedNode;

/// GraphNode during build.
#[derive(Debug, Clone)]
pub struct GraphBuilderNode {
    /// Node ID
    pub(crate) id: u32,

    /// neighbors of each level of the node.
    pub(crate) level_neighbors: Vec<BinaryHeap<OrderedNode>>,
}

impl GraphBuilderNode {
    pub(crate) fn new(id: u32, max_level: usize) -> Self {
        Self {
            id,
            level_neighbors: vec![BinaryHeap::new(); max_level],
        }
    }

    pub(crate) fn add_neighbor(&mut self, v: u32, dist: OrderedFloat, level: u16) {
        self.level_neighbors[level as usize].push(OrderedNode { dist, id: v });
    }
}

// impl From<&GraphBuilderNode> for GraphNode<u32> {
//     fn from(node: &GraphBuilderNode) -> Self {
//         let neighbors = node
//             .level_neighbors
//             .clone()
//             .into_sorted_vec()
//             .into_iter()
//             .map(|n| n.id)
//             .collect::<Vec<_>>();
//         Self {
//             id: node.id,
//             neighbors,
//         }
//     }
// }

/// Graph Builder.
///
/// [GraphBuilder] is used to build a graph in memory.
///
// #[derive(Clone)]
// pub struct GraphBuilder {
//     nodes: Vec<GraphBuilderNode>,

//     /// Storage for vectors.
//     vectors: Arc<InMemoryVectorStorage>,
// }

// impl Graph for GraphBuilder {
//     fn len(&self) -> usize {
//         self.nodes.len()
//     }

//     fn neighbors(&self, node: u32) -> Option<Box<dyn Iterator<Item = u32> + '_>> {
//         let node = &self.nodes[node as usize];
//         Some(Box::new(
//             node.level_neighbors
//                 .clone()
//                 .into_sorted_vec()
//                 .into_iter()
//                 .map(|n| n.id),
//         ))
//     }

//     fn storage(&self) -> Arc<dyn VectorStorage> {
//         self.vectors.clone()
//     }
// }

// impl GraphBuilder {
//     /// Build from a [VectorStorage].
//     pub fn new(vectors: Arc<InMemoryVectorStorage>) -> Self {
//         Self {
//             nodes: Vec::new(),
//             vectors,
//         }
//     }

//     /// add a node into the graph.
//     pub fn add(&mut self, node: u32) {
//         self.nodes.push(node, GraphBuilderNode::new(node));
//     }

//     /// Connect from one node to another.
//     pub fn connect(&mut self, from: u32, to: u32, distance: Option<OrderedFloat>) -> Result<()> {
//         let distance = distance.unwrap_or_else(|| self.vectors.distance_between(from, to).into());

//         {
//             let from_node = self.nodes.get_mut(&from).ok_or_else(|| Error::Index {
//                 message: format!("Node {} not found", from),
//                 location: location!(),
//             })?;
//             from_node.add_neighbor(distance, to)
//         }

//         {
//             let to_node = self.nodes.get_mut(&to).ok_or_else(|| Error::Index {
//                 message: format!("Node {} not found", to),
//                 location: location!(),
//             })?;
//             to_node.add_neighbor(distance, from);
//         }
//         Ok(())
//     }

//     pub fn prune(&mut self, node: u32, max_edges: usize) -> Result<()> {
//         let vector = self.vectors.vector(node);

//         let neighbors = &self
//             .nodes
//             .get(&node)
//             .ok_or_else(|| Error::Index {
//                 message: format!("Node {} not found", node),
//                 location: location!(),
//             })?
//             .neighbors;

//         let pruned_neighbors =
//             select_neighbors_heuristic(self, vector, neighbors, max_edges, false).collect();

//         self.nodes
//             .entry(node)
//             .and_modify(|node| node.neighbors = pruned_neighbors);
//         Ok(())
//     }

//     #[allow(dead_code)]
//     pub(crate) fn stats(&self) -> GraphBuilderStats {
//         let mut max_edges = 0;
//         let mut total_edges = 0;
//         let mut total_distance = 0.0;

//         for node in self.nodes.values() {
//             let edges = node.neighbors.len();
//             total_edges += edges;
//             max_edges = max_edges.max(edges);
//             total_distance += node.neighbors.iter().map(|n| n.dist.0).sum::<f32>();
//         }

//         GraphBuilderStats {
//             num_nodes: self.nodes.len(),
//             max_edges,
//             mean_edges: total_edges as f32 / self.nodes.len() as f32,
//             mean_distance: total_distance / total_edges as f32,
//         }
//     }
// }

#[derive(Debug)]
pub struct GraphBuilderStats {
    #[allow(dead_code)]
    pub num_nodes: usize,
    #[allow(dead_code)]
    pub max_edges: usize,
    #[allow(dead_code)]
    pub mean_edges: f32,
    #[allow(dead_code)]
    pub mean_distance: f32,
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     use arrow_array::{types::Float32Type, Float32Array};
//     use lance_linalg::{distance::MetricType, MatrixView};

//     #[test]
//     fn test_builder() {
//         let arr = Float32Array::from_iter_values((0..120).map(|v| v as f32));
//         let mat = Arc::new(MatrixView::<Float32Type>::new(Arc::new(arr), 8));
//         let store = Arc::new(InMemoryVectorStorage::new(mat, MetricType::L2));
//         let mut builder = GraphBuilder::new(store.clone());
//         builder.insert(0);
//         builder.insert(1);
//         builder.connect(0, 1, None).unwrap();
//         assert_eq!(builder.len(), 2);

//         assert_eq!(builder.neighbors(0).unwrap().collect::<Vec<_>>(), vec![1]);
//         assert_eq!(builder.neighbors(1).unwrap().collect::<Vec<_>>(), vec![0]);

//         builder.insert(4);
//         builder.connect(0, 4, None).unwrap();
//         assert_eq!(builder.len(), 3);

//         assert_eq!(
//             builder.neighbors(0).unwrap().collect::<Vec<_>>(),
//             vec![1, 4]
//         );
//         assert_eq!(builder.neighbors(1).unwrap().collect::<Vec<_>>(), vec![0]);
//     }
// }
