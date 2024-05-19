// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::OrderedFloat;
use super::OrderedNode;
use std::sync::Arc;

/// GraphNode during build.
///
/// WARNING: Internal API,  API stability is not guaranteed
#[derive(Debug, Clone)]
pub struct GraphBuilderNode {
    /// neighbors of each level of the node.
    pub(crate) bottom_neighbors: Arc<Vec<u32>>,
    pub(crate) level_neighbors: Vec<Arc<Vec<u32>>>,
    pub(crate) level_neighbors_ranked: Vec<Vec<OrderedNode>>,
}

impl GraphBuilderNode {
    pub(crate) fn new(_id: u32, max_level: usize) -> Self {
        let bottom_neighbors = Arc::new(Vec::new());
        let level_neighbors = (0..max_level).map(|_| Arc::new(Vec::new())).collect();
        let level_neighbors_ranked = (0..max_level).map(|_| Vec::new()).collect();
        Self {
            bottom_neighbors,
            level_neighbors,
            level_neighbors_ranked,
        }
    }

    pub(crate) fn add_neighbor(&mut self, v: u32, dist: OrderedFloat, level: u16) {
        self.level_neighbors_ranked[level as usize].push(OrderedNode { dist, id: v });
    }

    pub(crate) fn update_from_ranked_neighbors(&mut self, level: u16) {
        let level_index = level as usize;
        self.level_neighbors[level_index] = Arc::new(
            self.level_neighbors_ranked[level_index]
                .iter()
                .map(|ordered_node| ordered_node.id)
                .collect(),
        );
        if level == 0 {
            self.bottom_neighbors = self.level_neighbors[0].clone();
        }
    }

    pub(crate) fn cutoff(&self, level: u16, max_size: usize) -> OrderedFloat {
        let neighbors = &self.level_neighbors_ranked[level as usize];
        if neighbors.len() < max_size {
            OrderedFloat(f32::INFINITY)
        } else {
            neighbors.last().unwrap().dist
        }
    }

    /* TODO something like this should be useful for reads/writes
    pub(crate) fn encode(&self) -> RecordBatch {
        // Flattening the nested Vec structure into two Vecs for IDs and distances
        let mut neighbors = Vec::new();
        let mut dists = Vec::new();
        let mut lengths = Vec::new(); // For the lengths of each nested Vec

        for level in &self.level_neighbors_ranked {
            lengths.push(level.len() as u32);
            for edge in level {
                neighbors.push(edge.id);
                dists.push(edge.dist.into_inner());
            }
        }

        // Creating Arrow arrays for each flattened Vec
        let neighbor_array = UInt32Array::from(neighbors);
        let dist_array = Float32Array::from(dists);
        let lengths_array = Uint32Array::from(lengths);

        // Create a struct array from neighbor and dist
        let node_fields = vec![
            Field::new(
                "neighbor",
                DataType::List(Arc::new(Field::new(
                    "item",
                    neighbor_array.data_type().clone(),
                    true,
                ))),
                false,
            ),
            Field::new(
                "dist",
                DataType::List(Arc::new(Field::new(
                    "item",
                    dist_array.data_type().clone(),
                    true,
                ))),
                false,
            ),
            Field::new(
                "lengths",
                DataType::List(Arc::new(Field::new(
                    "item",
                    lengths_array.data_type().clone(),
                    true,
                ))),
                false,
            ),
        ];
        let node_type = DataType::Struct(node_fields);
        let node_struct_array = StructArray::from(vec![
            (
                Field::new("neighbor", DataType::UInt32, false),
                Arc::new(neighbor_array),
            ),
            (
                Field::new("dist", DataType::Float32, false),
                Arc::new(dist_array),
            ),
        ]);

        // Create a list array for the struct array using the recorded lengths
        let list_data_type = DataType::List(Box::new(Field::new("node", node_type, false)));
        let list_array = ListArray::from_iter_lengths(node_struct_array.iter(), lengths);

        // Creating the record batch
        let schema = Schema::new(vec![Field::new(
            "level_neighbors_ranked",
            list_data_type,
            false,
        )]);
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(list_array)]).unwrap()
    }
    */
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
