// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Generic Graph implementation.
//!

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use arrow_schema::{DataType, Field};
use bitvec::vec::BitVec;
use deepsize::DeepSizeOf;

pub mod builder;

use crate::vector::DIST_COL;

use crate::vector::storage::DistCalculator;

pub(crate) const NEIGHBORS_COL: &str = "__neighbors";

use std::sync::LazyLock;

/// NEIGHBORS field.
pub static NEIGHBORS_FIELD: LazyLock<Field> = LazyLock::new(|| {
    Field::new(
        NEIGHBORS_COL,
        DataType::List(Field::new_list_field(DataType::UInt32, true).into()),
        true,
    )
});
pub static DISTS_FIELD: LazyLock<Field> = LazyLock::new(|| {
    Field::new(
        DIST_COL,
        DataType::List(Field::new_list_field(DataType::Float32, true).into()),
        true,
    )
});

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
#[derive(Debug, PartialEq, Clone, Copy, DeepSizeOf)]
pub struct OrderedFloat(pub f32);

impl PartialOrd for OrderedFloat {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for OrderedFloat {}

impl Ord for OrderedFloat {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
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

#[derive(Debug, Eq, PartialEq, Clone, DeepSizeOf)]
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
    fn neighbors(&self, key: u32) -> Arc<Vec<u32>>;
}

/// Array-based visited list (faster than HashSet)
pub struct Visited<'a> {
    visited: &'a mut BitVec,
    recently_visited: Vec<u32>,
}

impl Visited<'_> {
    pub fn insert(&mut self, node_id: u32) {
        let node_id_usize = node_id as usize;
        if !self.visited[node_id_usize] {
            self.visited.set(node_id_usize, true);
            self.recently_visited.push(node_id);
        }
    }

    pub fn contains(&self, node_id: u32) -> bool {
        let node_id_usize = node_id as usize;
        self.visited[node_id_usize]
    }

    pub fn count_ones(&self) -> usize {
        self.visited.count_ones()
    }
}

impl Drop for Visited<'_> {
    fn drop(&mut self) {
        for node_id in self.recently_visited.iter() {
            self.visited.set(*node_id as usize, false);
        }
        self.recently_visited.clear();
    }
}

#[derive(Debug, Clone)]
pub struct VisitedGenerator {
    visited: BitVec,
    capacity: usize,
}

impl VisitedGenerator {
    pub fn new(capacity: usize) -> Self {
        Self {
            visited: BitVec::repeat(false, capacity),
            capacity,
        }
    }

    pub fn generate(&mut self, node_count: usize) -> Visited<'_> {
        if node_count > self.capacity {
            let new_capacity = self.capacity.max(node_count).next_power_of_two();
            self.visited.resize(new_capacity, false);
            self.capacity = new_capacity;
        }
        Visited {
            visited: &mut self.visited,
            recently_visited: Vec::new(),
        }
    }
}

fn process_neighbors_with_look_ahead<F>(
    neighbors: &[u32],
    mut process_neighbor: F,
    look_ahead: Option<usize>,
    dist_calc: &impl DistCalculator,
) where
    F: FnMut(u32),
{
    match look_ahead {
        Some(look_ahead) => {
            for i in 0..neighbors.len().saturating_sub(look_ahead) {
                dist_calc.prefetch(neighbors[i + look_ahead]);
                process_neighbor(neighbors[i]);
            }
            for neighbor in &neighbors[neighbors.len().saturating_sub(look_ahead)..] {
                process_neighbor(*neighbor);
            }
        }
        None => {
            for neighbor in neighbors.iter() {
                process_neighbor(*neighbor);
            }
        }
    }
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
/// WARNING: Internal API,  API stability is not guaranteed
///
/// TODO: This isn't actually beam search, function should probably be renamed
pub fn beam_search(
    graph: &dyn Graph,
    ep: &OrderedNode,
    k: usize,
    dist_calc: &impl DistCalculator,
    bitset: Option<&Visited>,
    prefetch_distance: Option<usize>,
    visited: &mut Visited,
) -> Vec<OrderedNode> {
    //let mut visited: HashSet<_> = HashSet::with_capacity(k);
    let mut candidates = BinaryHeap::with_capacity(k);
    visited.insert(ep.id);
    candidates.push(Reverse(ep.clone()));

    let mut results = BinaryHeap::with_capacity(k);
    if bitset.map(|bitset| bitset.contains(ep.id)).unwrap_or(true) {
        results.push(ep.clone());
    }

    while !candidates.is_empty() {
        let current = candidates.pop().expect("candidates is empty").0;
        let furthest = results
            .peek()
            .map(|node| node.dist)
            .unwrap_or(OrderedFloat(f32::INFINITY));

        // TODO: add an option to ignore the second condition for better performance.
        if current.dist > furthest && results.len() == k {
            break;
        }
        let neighbors = graph.neighbors(current.id);

        let furthest = results
            .peek()
            .map(|node| node.dist)
            .unwrap_or(OrderedFloat(f32::INFINITY));

        let unvisited_neighbors: Vec<_> = neighbors
            .iter()
            .filter(|&&neighbor| !visited.contains(neighbor))
            .copied()
            .collect();

        let process_neighbor = |neighbor: u32| {
            visited.insert(neighbor);
            let dist = dist_calc.distance(neighbor).into();
            if dist <= furthest || results.len() < k {
                if bitset
                    .map(|bitset| bitset.contains(neighbor))
                    .unwrap_or(true)
                {
                    if results.len() < k {
                        results.push((dist, neighbor).into());
                    } else if results.len() == k && dist < results.peek().unwrap().dist {
                        results.pop();
                        results.push((dist, neighbor).into());
                    }
                }
                candidates.push(Reverse((dist, neighbor).into()));
            }
        };
        process_neighbors_with_look_ahead(
            &unvisited_neighbors,
            process_neighbor,
            prefetch_distance,
            dist_calc,
        );
    }

    results.into_sorted_vec()
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
/// WARNING: Internal API,  API stability is not guaranteed
pub fn greedy_search(
    graph: &dyn Graph,
    start: OrderedNode,
    dist_calc: &impl DistCalculator,
    prefetch_distance: Option<usize>,
) -> OrderedNode {
    let mut current = start.id;
    let mut closest_dist = start.dist.0;
    loop {
        let neighbors = graph.neighbors(current);
        let mut next = None;

        let process_neighbor = |neighbor: u32| {
            let dist = dist_calc.distance(neighbor);
            if dist < closest_dist {
                closest_dist = dist;
                next = Some(neighbor);
            }
        };
        process_neighbors_with_look_ahead(
            &neighbors,
            process_neighbor,
            prefetch_distance,
            dist_calc,
        );

        if let Some(next) = next {
            current = next;
        } else {
            break;
        }
    }

    OrderedNode::new(current, closest_dist.into())
}

#[cfg(test)]
mod tests {}
