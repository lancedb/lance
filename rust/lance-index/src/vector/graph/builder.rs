// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::BinaryHeap;
use std::sync::RwLock;

use super::OrderedFloat;
use super::OrderedNode;

/// GraphNode during build.
/// 
/// WARNING: Internal API, do not use it directly.
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
