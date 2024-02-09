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

//! Generic Graph implementation.
//!

use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::Hash;

use lance_arrow::FloatToArrayType;
use lance_core::{Error, Result};
use lance_linalg::distance::{Cosine, DistanceFunc, Dot, MetricType, L2};
use num_traits::{AsPrimitive, Float, PrimInt};
use snafu::{location, Location};

pub(crate) mod builder;
pub(super) mod storage;

use storage::VectorStorage;

pub struct GraphNode<I = u32> {
    pub id: I,
    pub neighbors: Vec<I>,
}

impl<I> GraphNode<I> {
    pub fn new(id: I, neighbors: Vec<I>) -> Self {
        Self { id, neighbors }
    }
}

impl<I> From<I> for GraphNode<I> {
    fn from(id: I) -> Self {
        Self {
            id,
            neighbors: vec![],
        }
    }
}

#[async_trait::async_trait]
pub trait SerializeToLance {
    /// Serialize the object to one single Lance file.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///
    async fn to_lance(&self, path: &str) -> Result<()>;
}

/// A wrapper for f32 to make it ordered, so that we can put it into
/// a BTree or Heap
#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) struct OrderedFloat(pub f32);

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for OrderedFloat {}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

impl From<f32> for OrderedFloat {
    fn from(f: f32) -> Self {
        Self(f)
    }
}

impl From<OrderedFloat> for f32 {
    fn from(f: OrderedFloat) -> Self {
        f.0
    }
}

/// Graph trait.
///
/// Type parameters
/// ---------------
/// K: Vertex Index type
/// T: the data type of vector, i.e., ``f32`` or ``f16``.
pub trait Graph<K: PrimInt = u32, T: Float = f32> {
    /// Get the number of nodes in the graph.
    fn len(&self) -> usize;

    /// Returns true if the graph is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the neighbors of a graph node, identifyied by the index.
    fn neighbors(&self, key: K) -> Option<Box<dyn Iterator<Item = K> + '_>>;

    /// Distance from query vector to a node.
    fn distance_to(&self, query: &[T], key: K) -> f32;

    /// Distance between two nodes in the graph.
    ///
    /// Returns the distance between two nodes as float32.
    fn distance_between(&self, a: K, b: K) -> f32;
}

/// Serializable Graph.
///
/// Type parameters
/// ----------------
/// I : Vertex Index type
/// V : the data type of vector, i.e., ``f32`` or ``f16``.
struct InMemoryGraph<I: PrimInt + Hash, T: FloatToArrayType, V: storage::VectorStorage<T>> {
    pub nodes: HashMap<I, GraphNode<I>>,
    vectors: V,
    dist_fn: Box<DistanceFunc<T>>,
    phantom: std::marker::PhantomData<T>,
}

/// Beam search over a graph
///
/// This is the same as ``search-layer`` in HNSW.
///
/// Parameters
/// ----------
/// graph : Graph
///    The graph to search.
/// start : I
///   The index starting point.
/// query : &[f32]
///   The query vector.
///
/// Returns
/// -------
/// A sorted list of ``(dist, node_id)`` pairs.
///
pub(super) fn beam_search<I: PrimInt + Hash + std::fmt::Display>(
    graph: &dyn Graph<I>,
    start: &[I],
    query: &[f32],
    k: usize,
) -> Result<BTreeMap<OrderedFloat, I>> {
    let mut visited: HashSet<I> = start.iter().copied().collect();
    let mut candidates = start
        .iter()
        .map(|&i| (graph.distance_to(query, i).into(), i))
        .collect::<BTreeMap<_, _>>();
    let mut results = candidates.clone();

    while !candidates.is_empty() {
        let (dist, current) = candidates.pop_first().expect("candidates is empty");
        let furtherst = *results.last_key_value().expect("results set is empty").0;
        if dist < furtherst {
            break;
        }
        let neighbors = graph.neighbors(current).ok_or_else(|| Error::Index {
            message: format!("Node {} does not exist in the graph", current),
            location: location!(),
        })?;

        for neighbor in neighbors {
            if visited.contains(&neighbor) {
                continue;
            }
            visited.insert(neighbor);
            let dist = graph.distance_to(query, neighbor).into();
            if dist < furtherst || results.len() < k {
                results.insert(dist, neighbor);
                candidates.insert(dist, neighbor);
            }
        }
        // Maintain the size of the result set.
        while results.len() > k {
            results.pop_last();
        }
    }
    Ok(results)
}

impl<I: PrimInt + Hash + AsPrimitive<usize>, T: FloatToArrayType, V: storage::VectorStorage<T>>
    Graph<I, T> for InMemoryGraph<I, T, V>
{
    fn neighbors(&self, key: I) -> Option<Box<dyn Iterator<Item = I> + '_>> {
        self.nodes
            .get(&key)
            .map(|n| Box::new(n.neighbors.iter().copied()) as Box<dyn Iterator<Item = I>>)
    }

    fn distance_to(&self, query: &[T], idx: I) -> f32 {
        let vec = self.vectors.get(idx.as_());
        (self.dist_fn)(query, vec)
    }

    fn distance_between(&self, a: I, b: I) -> f32 {
        let from_vec = self.vectors.get(a.as_());
        self.distance_to(from_vec, b)
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }
}

impl<I: PrimInt + Hash, T: FloatToArrayType, V: VectorStorage<T>> InMemoryGraph<I, T, V>
where
    T::ArrowType: L2 + Cosine + Dot,
{
    fn from_builder(nodes: HashMap<I, GraphNode<I>>, vectors: V, metric_type: MetricType) -> Self {
        Self {
            nodes,
            vectors,
            dist_fn: metric_type.func::<T>().into(),
            phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {}
