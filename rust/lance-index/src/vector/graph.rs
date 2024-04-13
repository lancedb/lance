// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Generic Graph implementation.
//!

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc;

use arrow_schema::{DataType, Field};
use lance_core::{Error, Result};
use snafu::{location, Location};

pub(crate) mod builder;
pub mod memory;
pub(super) mod storage;

/// Vector storage to back a graph.
pub use storage::VectorStorage;

use self::storage::DistCalculator;

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
pub struct OrderedFloat(pub f32);

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
pub struct OrderedNode {
    pub id: u32,
    pub dist: OrderedFloat,
}

impl OrderedNode {
    pub fn new(id: u32, dist: OrderedFloat) -> Self {
        Self { id, dist }
    }
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
///  The graph to search.
/// start : &[OrderedNode]
///  The starting point.
/// query : &[f32]
///  The query vector.
/// k : usize
///  The number of results to return.
/// bitset : Option<&RoaringBitmap>
///  The bitset of node IDs to filter the results, bit 1 for the node to keep, and bit 0 for the node to discard.
///
/// Returns
/// -------
/// A descending sorted list of ``(dist, node_id)`` pairs.
///
pub(super) fn beam_search(
    graph: &dyn Graph,
    start: &[OrderedNode],
    query: &[f32],
    k: usize,
    dist_calc: Option<Arc<dyn DistCalculator>>,
    bitset: Option<&roaring::bitmap::RoaringBitmap>,
) -> Result<Vec<OrderedNode>> {
    let mut visited: HashSet<_> = start.iter().map(|node| node.id).collect();
    let dist_calc = dist_calc.unwrap_or_else(|| graph.storage().dist_calculator(query).into());
    let mut candidates = start
        .iter()
        .cloned()
        .map(Reverse)
        .collect::<BinaryHeap<_>>();
    let mut results = candidates
        .clone()
        .into_iter()
        .filter(|node| {
            bitset
                .map(|bitset| bitset.contains(node.0.id))
                .unwrap_or(true)
        })
        .map(|v| v.0)
        .collect::<BinaryHeap<_>>();

    while !candidates.is_empty() {
        let current = candidates.pop().expect("candidates is empty").0;
        let furthest = results
            .peek()
            .map(|node| node.dist)
            .unwrap_or(OrderedFloat(f32::INFINITY));
        if current.dist > furthest {
            break;
        }
        let neighbors = graph.neighbors(current.id).ok_or_else(|| Error::Index {
            message: format!("Node {} does not exist in the graph", current.id),
            location: location!(),
        })?;

        for neighbor in neighbors {
            if visited.contains(&neighbor) {
                continue;
            }
            visited.insert(neighbor);
            let furthest = results
                .peek()
                .map(|node| node.dist)
                .unwrap_or(OrderedFloat(f32::INFINITY));
            let dist = dist_calc.distance(&[neighbor])[0].into();
            if dist <= furthest || results.len() < k {
                if bitset
                    .map(|bitset| bitset.contains(neighbor))
                    .unwrap_or(true)
                {
                    results.push((dist, neighbor).into());
                    if results.len() > k {
                        results.pop();
                    }
                }
                candidates.push(Reverse((dist, neighbor).into()));
            }
        }
    }

    Ok(results.into_sorted_vec())
}

/// Greedy search over a graph
///
/// This searches for only one result, only used for finding the entry point
///
/// Parameters
/// ----------
/// graph : Graph
///    The graph to search.
/// start : u32
///   The index starting point.
/// query : &[f32]
///   The query vector.
///
/// Returns
/// -------
/// A ``(dist, node_id)`` pair.
///
pub(super) fn greedy_search(
    graph: &dyn Graph,
    start: OrderedNode,
    query: &[f32],
    dist_calc: Option<Arc<dyn DistCalculator>>,
) -> Result<OrderedNode> {
    let mut current = start.id;
    let mut closest_dist = start.dist.0;
    let dist_calc = dist_calc.unwrap_or_else(|| graph.storage().dist_calculator(query).into());
    loop {
        let neighbors: Vec<_> = graph
            .neighbors(current)
            .ok_or_else(|| Error::Index {
                message: format!("Node {} does not exist in the graph", current),
                location: location!(),
            })?
            .collect();
        let distances = dist_calc.distance(&neighbors);

        let mut next = None;
        for (neighbor, dist) in neighbors.into_iter().zip(distances) {
            if dist < closest_dist {
                closest_dist = dist;
                next = Some(neighbor);
            }
        }

        if let Some(next) = next {
            current = next;
        } else {
            break;
        }
    }

    Ok(OrderedNode::new(current, closest_dist.into()))
}

#[cfg(test)]
mod tests {}
