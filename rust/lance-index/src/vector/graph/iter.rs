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

use std::collections::{HashSet, VecDeque};

use num_traits::Float;

use super::Graph;

/// Breath-first search iterator.
///
/// Internal use only
pub struct Iter<'a, T: Float> {
    graph: &'a dyn Graph<T>,
    queue: VecDeque<u64>,
    visited: HashSet<u64>,
}

impl<'a, T: Float> Iter<'a, T> {
    pub(super) fn new(graph: &'a dyn Graph<T>, start: u64) -> Self {
        let mut visited = HashSet::new();
        visited.insert(start);
        Self {
            graph,
            queue: VecDeque::from([start]),
            visited,
        }
    }
}

impl<'a, T: Float> Iterator for Iter<'a, T> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.queue.pop_front()?;
        if let Some(neighbors) = self.graph.neighbors(node) {
            for &neighbor in neighbors {
                if self.visited.insert(neighbor) {
                    self.queue.push_back(neighbor);
                }
            }
        }
        Some(node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::{types::Float32Type, Float32Array};
    use lance_linalg::{distance::MetricType, MatrixView};

    use crate::vector::graph::builder::GraphBuilderNode;
    use crate::vector::graph::{InMemoryGraph, OrderedFloat};

    #[test]
    fn test_bfs_iterator() {
        let mut builder_nodes = (0..5).map(|i| GraphBuilderNode::new(i)).collect::<Vec<_>>();
        builder_nodes[0]
            .neighbors
            .extend([1, 2].map(|i| (OrderedFloat(i as f32), i as u64)));
        builder_nodes[1]
            .neighbors
            .extend([3, 4].map(|i| (OrderedFloat(i as f32), i as u64)));
        builder_nodes[2]
            .neighbors
            .extend([5, 6].map(|i| (OrderedFloat(i as f32), i as u64)));
        builder_nodes[3]
            .neighbors
            .extend([0, 7, 5].map(|i| (OrderedFloat(i as f32), i as u64)));
        builder_nodes[4]
            .neighbors
            .extend([8, 1, 2, 4, 9, 10].map(|i| (OrderedFloat(i as f32), i as u64)));

        let nodes = builder_nodes
            .into_iter()
            .enumerate()
            .map(|(id, n)| (id as u64, n))
            .collect();

        let mat =
            MatrixView::<Float32Type>::new(Arc::new(Float32Array::from(Vec::<f32>::new())), 8);
        let graph = InMemoryGraph::from_builder(&nodes, mat, MetricType::L2);

        let sorted_nodes = graph.iter().collect::<Vec<_>>();
        assert_eq!(sorted_nodes, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }
}
