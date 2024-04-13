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
