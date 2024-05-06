// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::BinaryHeap;
use std::sync::RwLock;

use super::OrderedFloat;
use super::OrderedNode;

/// GraphNode during build.
///
/// WARNING: Internal API,  API stability is not guaranteed
#[derive(Debug)]
pub struct GraphBuilderNode {
    /// Node ID
    pub(crate) id: u32,

    /// neighbors of each level of the node.
    pub(crate) level_neighbors: Vec<RwLock<BinaryHeap<OrderedNode>>>,
}

impl GraphBuilderNode {
    pub(crate) fn new(id: u32, max_level: usize) -> Self {
        let level_neighbors = (0..max_level)
            .map(|_| RwLock::new(BinaryHeap::new()))
            .collect();
        Self {
            id,
            level_neighbors,
        }
    }

    pub(crate) fn add_neighbor(&self, v: u32, dist: OrderedFloat, level: u16) {
        self.level_neighbors[level as usize]
            .write()
            .unwrap()
            .push(OrderedNode { dist, id: v });
    }
}

#[derive(Debug, Clone, Default)]
pub struct GraphBuilderStats {
    /// Sum of the number of nodes.
    pub num_nodes: usize,

    /// Sum of the number of edges.
    pub num_edges: usize,

    /// Mean number of edges per node.
    pub mean_edges: f32,

    /// Mean distance between nodes.
    pub mean_distance: f32,
}
