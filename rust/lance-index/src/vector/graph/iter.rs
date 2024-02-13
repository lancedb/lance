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
use std::hash::Hash;

use num_traits::{Float, PrimInt};

use super::Graph;

/// Breath-first search iterator.
///
/// Internal use only
pub struct Iter<'a, K: PrimInt + Hash, T: Float> {
    graph: &'a dyn Graph<K, T>,
    queue: VecDeque<K>,
    visited: HashSet<K>,
}

impl<'a, K: PrimInt + Hash, T: Float> Iter<'a, K, T> {
    pub(super) fn new(graph: &'a dyn Graph<K, T>, start: K) -> Self {
        let mut visited = HashSet::new();
        visited.insert(start);
        Self {
            graph,
            queue: VecDeque::from([start]),
            visited,
        }
    }
}

impl<'a, K: PrimInt + Hash, T: Float> Iterator for Iter<'a, K, T> {
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.queue.pop_front()?;
        if let Some(neighbors) = self.graph.neighbors(node) {
            for neighbor in neighbors {
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

    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{types::Float32Type, Float32Array};
    use lance_linalg::{distance::MetricType, MatrixView};

    use crate::vector::graph::{GraphNode, InMemoryGraph};

    #[test]
    fn test_bfs_iterator() {
        let nodes = [
            GraphNode::new(0, vec![1, 2]),
            GraphNode::new(1, vec![3, 4]),
            GraphNode::new(2, vec![5, 6]),
            GraphNode::new(3, vec![0, 7, 5]),
            GraphNode::new(4, vec![8, 1, 2, 4, 9, 10]),
        ]
        .into_iter()
        .map(|n| (n.id, n))
        .collect::<HashMap<_, _>>();
        let mat =
            MatrixView::<Float32Type>::new(Arc::new(Float32Array::from(Vec::<f32>::new())), 8);
        let graph = InMemoryGraph::from_builder(nodes, mat, MetricType::L2);

        let sorted_nodes = graph.iter().collect::<Vec<_>>();
        assert_eq!(sorted_nodes, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }
}
