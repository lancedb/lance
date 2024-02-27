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

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use arrow_schema::{DataType, Field};
use lance_core::{Error, Result};
use snafu::{location, Location};

pub(crate) mod builder;
pub mod memory;
pub(super) mod storage;

/// Vector storage to back a graph.
pub use storage::VectorStorage;

pub(crate) const NEIGHBORS_COL: &str = "__neighbors";

lazy_static::lazy_static! {
    /// NEIGHBORS field.
    pub static ref NEIGHBORS_FIELD: Field =
        Field::new(NEIGHBORS_COL, DataType::List(Field::new_list_field(DataType::UInt32, true).into()), true);
}

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

#[derive(Debug, Eq, PartialEq, Clone)]
pub(crate) struct OrderedNode {
    pub id: u32,
    pub dist: OrderedFloat,
}

impl PartialOrd for OrderedNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.dist.cmp(&other.dist))
    }
}

impl Ord for OrderedNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.cmp(&other.dist)
    }
}

impl From<(OrderedFloat, u32)> for OrderedNode {
    fn from((dist, id): (OrderedFloat, u32)) -> Self {
        Self { id, dist }
    }
}

impl From<OrderedNode> for (OrderedFloat, u32) {
    fn from(node: OrderedNode) -> Self {
        (node.dist, node.id)
    }
}

/// Distance calculator.
///
/// This trait is used to calculate a query vector to a stream of vector IDs.
///
pub trait DistanceCalculator {
    /// Compute distances between one query vector to all the vectors in the
    /// list of IDs.
    fn compute_distances(&self, ids: &[u32]) -> Box<dyn Iterator<Item = f32>>;
}

/// Graph trait.
///
/// Type parameters
/// ---------------
/// K: Vertex Index type
/// T: the data type of vector, i.e., ``f32`` or ``f16``.
pub trait Graph {
    /// Get the number of nodes in the graph.
    fn len(&self) -> usize;

    /// Returns true if the graph is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the neighbors of a graph node, identifyied by the index.
    fn neighbors(&self, key: u32) -> Option<Box<dyn Iterator<Item = u32> + '_>>;

    /// Access to underline storage
    fn storage(&self) -> Arc<dyn VectorStorage>;
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
pub(super) fn beam_search(
    graph: &dyn Graph,
    start: &[u32],
    query: &[f32],
    k: usize,
) -> Result<BTreeMap<OrderedFloat, u32>> {
    let mut visited: HashSet<_> = start.iter().copied().collect();
    let dist_calc = graph.storage().dist_calculator(query);
    let mut candidates: BTreeMap<OrderedFloat, _> = dist_calc
        .distance(start)
        .iter()
        .zip(start)
        .map(|(&dist, id)| (dist.into(), *id))
        .collect::<BTreeMap<_, _>>();
    let mut results = candidates.clone();

    while !candidates.is_empty() {
        let (dist, current) = candidates.pop_first().expect("candidates is empty");
        let furtherst = *results.last_key_value().expect("results set is empty").0;
        if dist > furtherst {
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
            let furtherst = *results.last_key_value().expect("results set is empty").0;
            let dist = dist_calc.distance(&[neighbor])[0].into();
            if dist < furtherst || results.len() < k {
                results.insert(dist, neighbor);
                candidates.insert(dist, neighbor);
                if results.len() > k {
                    results.pop_last();
                }
            }
        }
    }
    Ok(results)
}

#[cfg(test)]
mod tests {}
