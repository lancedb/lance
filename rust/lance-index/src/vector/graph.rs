// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Generic Graph implementation.
//!

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::Arc;

use arrow_schema::{DataType, Field};
use lance_core::deepsize::DeepSizeOf;

use crate::vector::hnsw::builder::HnswQueryParams;

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
pub struct OrderedNode<T = u32>
where
    T: PartialEq + Eq,
{
    pub id: T,
    pub dist: OrderedFloat,
}

impl<T: PartialEq + Eq> OrderedNode<T> {
    pub fn new(id: T, dist: OrderedFloat) -> Self {
        Self { id, dist }
    }
}

impl<T: PartialEq + Eq> PartialOrd for OrderedNode<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: PartialEq + Eq> Ord for OrderedNode<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.cmp(&other.dist)
    }
}

impl<T: PartialEq + Eq> From<(OrderedFloat, T)> for OrderedNode<T> {
    fn from((dist, id): (OrderedFloat, T)) -> Self {
        Self { id, dist }
    }
}

impl<T: PartialEq + Eq> From<OrderedNode<T>> for (OrderedFloat, T) {
    fn from(node: OrderedNode<T>) -> Self {
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

pub trait BorrowingGraph {
    /// Get the number of nodes in the graph.
    fn len(&self) -> usize;

    /// Returns true if the graph is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the neighbors of a graph node, identified by the index.
    fn neighbors(&self, key: u32) -> &[u32];
}

const WORD_BITS: usize = usize::BITS as usize;

/// Compact visited list for graph traversals.
pub struct Visited<'a> {
    visited: &'a mut Vec<usize>,
    recently_visited: &'a mut Vec<u32>,
}

impl Visited<'_> {
    pub fn insert(&mut self, node_id: u32) {
        let node_id_usize = node_id as usize;
        let word_index = node_id_usize / WORD_BITS;
        let mask = 1usize << (node_id_usize % WORD_BITS);
        if self.visited[word_index] & mask == 0 {
            self.visited[word_index] |= mask;
            self.recently_visited.push(node_id);
        }
    }

    pub fn contains(&self, node_id: u32) -> bool {
        let node_id_usize = node_id as usize;
        let word_index = node_id_usize / WORD_BITS;
        let mask = 1usize << (node_id_usize % WORD_BITS);
        self.visited[word_index] & mask != 0
    }

    #[inline(always)]
    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.recently_visited
            .iter()
            .map(|node_id| *node_id as usize)
    }

    pub fn count_ones(&self) -> usize {
        self.recently_visited.len()
    }
}

impl Drop for Visited<'_> {
    fn drop(&mut self) {
        for node_id in self.recently_visited.iter().copied() {
            let node_id_usize = node_id as usize;
            let word_index = node_id_usize / WORD_BITS;
            let mask = 1usize << (node_id_usize % WORD_BITS);
            self.visited[word_index] &= !mask;
        }
        self.recently_visited.clear();
    }
}

#[derive(Debug, Clone)]
pub struct VisitedGenerator {
    visited: Vec<usize>,
    recently_visited: Vec<u32>,
    capacity: usize,
}

impl VisitedGenerator {
    pub fn new(capacity: usize) -> Self {
        Self {
            visited: vec![0; capacity.div_ceil(WORD_BITS)],
            recently_visited: Vec::new(),
            capacity,
        }
    }

    pub fn generate(&mut self, node_count: usize) -> Visited<'_> {
        if node_count > self.capacity {
            let new_capacity = self.capacity.max(node_count).next_power_of_two();
            self.visited.resize(new_capacity.div_ceil(WORD_BITS), 0);
            self.capacity = new_capacity;
        }
        Visited {
            visited: &mut self.visited,
            recently_visited: &mut self.recently_visited,
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

#[inline]
fn furthest_distance(results: &BinaryHeap<OrderedNode>) -> OrderedFloat {
    results
        .peek()
        .map(|node| node.dist)
        .unwrap_or(OrderedFloat(f32::INFINITY))
}

#[inline]
fn push_result(results: &mut BinaryHeap<OrderedNode>, candidate: OrderedNode, k: usize) {
    if results.len() < k {
        results.push(candidate);
    } else if candidate.dist < results.peek().unwrap().dist {
        results.pop();
        results.push(candidate);
    }
}

/// Test-only instrumentation: number of *full* 16-wide neighbor batches flushed
/// through [`DistCalculator::distance_batch_16`] by [`beam_search_loop`]. Lets the
/// AMX activation test prove the batched (accelerated) path is actually exercised
/// during graph construction, rather than every neighbor list being < 16 and
/// silently draining through the scalar tail.
#[cfg(test)]
pub(crate) static BEAM_BATCH16_FLUSHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Test-only instrumentation companion to [`BEAM_BATCH16_FLUSHES`]: number of
/// non-empty `< 16` remainder drains (the scalar tail). Confirms the remainder
/// path is exercised, not just the full-batch path.
#[cfg(test)]
pub(crate) static BEAM_BATCH16_DRAINS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Gate for the two counters above. Off by default: the counter `fetch_add`s are
/// a single shared atomic each, so leaving them live in the parallel build
/// benchmark would serialize every flush across all worker threads and dominate
/// the measurement. The (single-threaded) activation test flips this on around
/// its build; the benchmark leaves it off, so only a cheap uncontended relaxed
/// load remains on the hot path.
#[cfg(test)]
pub(crate) static BEAM_BATCH16_COUNT_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Apply the beam-search accept / threshold / push logic for one already-computed
/// neighbor distance. Extracted so the batched-flush path and the scalar-drain
/// path in [`beam_search_loop`] share a single implementation.
///
/// `furthest` is the per-candidate snapshot of the results-heap frontier — read
/// once when the current candidate was popped, exactly as the pre-batching loop
/// did — while `results.len()` (and `push_result`'s own `peek`) are read fresh.
/// Because distances are a pure function of (query, neighbor), independent of heap
/// state, computing a batch of them up-front and then applying this in buffer order
/// is byte-for-byte equivalent to the old interleaved per-neighbor loop.
#[inline]
fn apply_neighbor(
    candidates: &mut BinaryHeap<Reverse<OrderedNode>>,
    results: &mut BinaryHeap<OrderedNode>,
    k: usize,
    furthest: OrderedFloat,
    accepts: impl Fn(u32, OrderedFloat) -> bool,
    neighbor: u32,
    dist: OrderedFloat,
) {
    if dist <= furthest || results.len() < k {
        if accepts(neighbor, dist) {
            push_result(results, (dist, neighbor).into(), k);
        }
        candidates.push(Reverse((dist, neighbor).into()));
    }
}

macro_rules! beam_search_loop {
    (
        $candidates:ident,
        $results:ident,
        $visited:ident,
        $k:expr,
        $dist_calc:expr,
        $prefetch_distance:expr,
        $accepts_result:expr,
        |$current:ident, $process_neighbor:ident| $visit_neighbors:block
    ) => {{
        // Decided once per search — `dist_calc` is fixed for the whole traversal.
        // Only backends whose `distance_batch_16` is backed by a fused kernel take
        // the batched arm; everything else (PQ, SQ, BQ, flat f32/f64, and fp16 on
        // hosts without usable AMX-FP16) takes the `else` arm, which is the
        // original traversal, so they are untouched by this path.
        let use_batch16 = $dist_calc.has_batch_16_kernel();

        while !$candidates.is_empty() {
            let $current = $candidates.pop().expect("candidates is empty").0;
            let furthest = furthest_distance(&$results);

            if $current.dist > furthest && $results.len() == $k {
                break;
            }

            if use_batch16 {
                // Buffer freshly-visited neighbor ids and evaluate their distances
                // in batches of 16 via `distance_batch_16` (one fused AMX-FP16 tile
                // pass for flat fp16 Dot search) instead of one scalar call at a
                // time — mirrors FAISS #5235's `search_from_candidates_fixVT`.
                // Dedup / visited-marking still happens inline at first encounter,
                // exactly where the scalar loop marks it; only the distance
                // computation and its accept/push consequences are deferred to the
                // flush.
                let mut batch_ids = [0u32; 16];
                let mut batch_len = 0usize;

                let $process_neighbor = |neighbor: u32| {
                    if $visited.contains(neighbor) {
                        return;
                    }
                    $visited.insert(neighbor);
                    batch_ids[batch_len] = neighbor;
                    batch_len += 1;
                    if batch_len == 16 {
                        let mut batch_dists = [0f32; 16];
                        $dist_calc.distance_batch_16(&batch_ids, &mut batch_dists);
                        #[cfg(test)]
                        if $crate::vector::graph::BEAM_BATCH16_COUNT_ENABLED
                            .load(std::sync::atomic::Ordering::Relaxed)
                        {
                            $crate::vector::graph::BEAM_BATCH16_FLUSHES
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        for i in 0..16 {
                            apply_neighbor(
                                &mut $candidates,
                                &mut $results,
                                $k,
                                furthest,
                                &$accepts_result,
                                batch_ids[i],
                                batch_dists[i].into(),
                            );
                        }
                        batch_len = 0;
                    }
                };
                $visit_neighbors

                // Drain the < 16 remainder with scalar `distance` calls (FAISS's
                // tail loop). `distance_batch_16` needs exactly 16 valid ids, so a
                // partial buffer cannot use it without padding with bogus ids.
                #[cfg(test)]
                if batch_len > 0
                    && $crate::vector::graph::BEAM_BATCH16_COUNT_ENABLED
                        .load(std::sync::atomic::Ordering::Relaxed)
                {
                    $crate::vector::graph::BEAM_BATCH16_DRAINS
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                for i in 0..batch_len {
                    let neighbor = batch_ids[i];
                    let dist: OrderedFloat = $dist_calc.distance(neighbor).into();
                    apply_neighbor(
                        &mut $candidates,
                        &mut $results,
                        $k,
                        furthest,
                        &$accepts_result,
                        neighbor,
                        dist,
                    );
                }
            } else {
                // The pre-batching traversal, unchanged: evaluate each freshly
                // visited neighbor immediately. `apply_neighbor` holds the same
                // accept / threshold / push logic this loop used to inline.
                let $process_neighbor = |neighbor: u32| {
                    if $visited.contains(neighbor) {
                        return;
                    }
                    $visited.insert(neighbor);
                    let dist: OrderedFloat = $dist_calc.distance(neighbor).into();
                    apply_neighbor(
                        &mut $candidates,
                        &mut $results,
                        $k,
                        furthest,
                        &$accepts_result,
                        neighbor,
                        dist,
                    );
                };
                $visit_neighbors
            }
        }
    }};
}

macro_rules! greedy_search_loop {
    (
        $current:ident,
        $closest_dist:ident,
        $dist_calc:expr,
        $prefetch_distance:expr,
        |$process_neighbor:ident| $visit_neighbors:block
    ) => {{
        loop {
            let mut next = None;
            let $process_neighbor = |neighbor: u32| {
                let dist = $dist_calc.distance(neighbor);
                if dist < $closest_dist {
                    $closest_dist = dist;
                    next = Some(neighbor);
                }
            };
            $visit_neighbors

            if let Some(next) = next {
                $current = next;
            } else {
                break;
            }
        }
    }};
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
    params: &HnswQueryParams,
    dist_calc: &impl DistCalculator,
    bitset: Option<&Visited>,
    prefetch_distance: Option<usize>,
    visited: &mut Visited,
) -> Vec<OrderedNode> {
    let k = params.ef;
    let mut candidates = BinaryHeap::with_capacity(k);
    visited.insert(ep.id);
    candidates.push(Reverse(ep.clone()));

    let mut results = BinaryHeap::with_capacity(k);
    let no_filter =
        bitset.is_none() && params.lower_bound.is_none() && params.upper_bound.is_none();

    if no_filter {
        results.push(ep.clone());
        let accepts_result = |_: u32, _: OrderedFloat| true;
        beam_search_loop!(
            candidates,
            results,
            visited,
            k,
            dist_calc,
            prefetch_distance,
            accepts_result,
            |current, process_neighbor| {
                let neighbors = graph.neighbors(current.id);
                process_neighbors_with_look_ahead(
                    &neighbors,
                    process_neighbor,
                    prefetch_distance,
                    dist_calc,
                );
            }
        );
        return results.into_sorted_vec();
    }

    // add range search support
    let lower_bound: OrderedFloat = params.lower_bound.unwrap_or(f32::MIN).into();
    let upper_bound: OrderedFloat = params.upper_bound.unwrap_or(f32::MAX).into();

    if bitset.map(|bitset| bitset.contains(ep.id)).unwrap_or(true)
        && ep.dist >= lower_bound
        && ep.dist < upper_bound
    {
        results.push(ep.clone());
    }

    let accepts_result = |node_id: u32, dist: OrderedFloat| {
        bitset
            .map(|bitset| bitset.contains(node_id))
            .unwrap_or(true)
            && dist >= lower_bound
            && dist < upper_bound
    };
    beam_search_loop!(
        candidates,
        results,
        visited,
        k,
        dist_calc,
        prefetch_distance,
        accepts_result,
        |current, process_neighbor| {
            let neighbors = graph.neighbors(current.id);
            process_neighbors_with_look_ahead(
                &neighbors,
                process_neighbor,
                prefetch_distance,
                dist_calc,
            );
        }
    );
    results.into_sorted_vec()
}

pub fn beam_search_borrowed(
    graph: &impl BorrowingGraph,
    ep: &OrderedNode,
    params: &HnswQueryParams,
    dist_calc: &impl DistCalculator,
    bitset: Option<&Visited>,
    prefetch_distance: Option<usize>,
    visited: &mut Visited,
) -> Vec<OrderedNode> {
    let k = params.ef;
    let mut candidates = BinaryHeap::with_capacity(k);
    visited.insert(ep.id);
    candidates.push(Reverse(ep.clone()));

    let mut results = BinaryHeap::with_capacity(k);
    let no_filter =
        bitset.is_none() && params.lower_bound.is_none() && params.upper_bound.is_none();

    if no_filter {
        results.push(ep.clone());
        let accepts_result = |_: u32, _: OrderedFloat| true;
        beam_search_loop!(
            candidates,
            results,
            visited,
            k,
            dist_calc,
            prefetch_distance,
            accepts_result,
            |current, process_neighbor| {
                let neighbors = graph.neighbors(current.id);
                process_neighbors_with_look_ahead(
                    neighbors,
                    process_neighbor,
                    prefetch_distance,
                    dist_calc,
                );
            }
        );
        return results.into_sorted_vec();
    }

    let lower_bound: OrderedFloat = params.lower_bound.unwrap_or(f32::MIN).into();
    let upper_bound: OrderedFloat = params.upper_bound.unwrap_or(f32::MAX).into();

    if bitset.map(|bitset| bitset.contains(ep.id)).unwrap_or(true)
        && ep.dist >= lower_bound
        && ep.dist < upper_bound
    {
        results.push(ep.clone());
    }

    let accepts_result = |node_id: u32, dist: OrderedFloat| {
        bitset
            .map(|bitset| bitset.contains(node_id))
            .unwrap_or(true)
            && dist >= lower_bound
            && dist < upper_bound
    };
    beam_search_loop!(
        candidates,
        results,
        visited,
        k,
        dist_calc,
        prefetch_distance,
        accepts_result,
        |current, process_neighbor| {
            let neighbors = graph.neighbors(current.id);
            process_neighbors_with_look_ahead(
                neighbors,
                process_neighbor,
                prefetch_distance,
                dist_calc,
            );
        }
    );
    results.into_sorted_vec()
}

/// Number of mask-passing nodes used to seed [beam_search_acorn]'s frontier.
const ACORN_SEED_COUNT: usize = 16;

/// Cap on starved-frontier waypoint expansions in [beam_search_acorn],
/// as a multiple of `ef`.
const ACORN_BRIDGE_BUDGET_FACTOR: usize = 4;

/// Beam search over the mask-passing subgraph (ACORN-1).
///
/// Only nodes in `bitset` get distances. A filtered-out neighbor contributes
/// its own neighbors instead, expanded once via `expanded`. Deeper masked
/// chains are crossed through unscored waypoints under a budget. The frontier
/// starts from the entry point plus mask-sampled seeds. May return fewer than
/// `min(ef, passing)` results if the budget runs out, so callers needing a
/// guarantee must check the count.
#[allow(clippy::too_many_arguments)]
pub fn beam_search_acorn(
    graph: &impl BorrowingGraph,
    ep: &OrderedNode,
    params: &HnswQueryParams,
    dist_calc: &impl DistCalculator,
    bitset: &Visited,
    prefetch_distance: Option<usize>,
    visited: &mut Visited,
    expanded: &mut Visited,
) -> Vec<OrderedNode> {
    let ef = params.ef;
    let lower_bound: OrderedFloat = params.lower_bound.unwrap_or(f32::MIN).into();
    let upper_bound: OrderedFloat = params.upper_bound.unwrap_or(f32::MAX).into();
    let passing_total = bitset.count_ones();
    let mut candidates = BinaryHeap::with_capacity(ef);
    let mut results = BinaryHeap::with_capacity(ef);
    // collected per node before scoring so prefetch targets are the ids
    // that actually get distances
    let mut passing: Vec<u32> = Vec::with_capacity(64);
    // masked nodes seen two hops out, expandable if the frontier starves,
    // deduped against `expanded` at pop rather than at push
    let mut waypoints: VecDeque<u32> = VecDeque::new();
    let mut bridge_budget = ACORN_BRIDGE_BUDGET_FACTOR * ef;

    // the entry point seeds the traversal even if it fails the mask
    visited.insert(ep.id);
    candidates.push(Reverse(ep.clone()));
    if bitset.contains(ep.id) && ep.dist >= lower_bound && ep.dist < upper_bound {
        results.push(ep.clone());
    }

    let stride = (passing_total / ACORN_SEED_COUNT).max(1);
    for seed in bitset.iter_ones().step_by(stride).take(ACORN_SEED_COUNT) {
        let seed = seed as u32;
        if visited.contains(seed) {
            continue;
        }
        visited.insert(seed);
        let dist: OrderedFloat = dist_calc.distance(seed).into();
        if dist >= lower_bound && dist < upper_bound {
            push_result(&mut results, (dist, seed).into(), ef);
        }
        candidates.push(Reverse((dist, seed).into()));
    }

    loop {
        let Some(Reverse(current)) = candidates.pop() else {
            // frontier starved: burn bridge budget through masked waypoints
            // until a new passing node is found
            if results.len() >= ef.min(passing_total) {
                break;
            }
            let mut found = false;
            while let Some(waypoint) = waypoints.pop_front() {
                if bridge_budget == 0 {
                    break;
                }
                if expanded.contains(waypoint) {
                    continue;
                }
                expanded.insert(waypoint);
                bridge_budget -= 1;
                for &neighbor in graph.neighbors(waypoint) {
                    if bitset.contains(neighbor) {
                        if !visited.contains(neighbor) {
                            visited.insert(neighbor);
                            let dist: OrderedFloat = dist_calc.distance(neighbor).into();
                            if dist >= lower_bound && dist < upper_bound {
                                push_result(&mut results, (dist, neighbor).into(), ef);
                            }
                            candidates.push(Reverse((dist, neighbor).into()));
                            found = true;
                        }
                    } else if !expanded.contains(neighbor) {
                        waypoints.push_back(neighbor);
                    }
                }
                if found {
                    break;
                }
            }
            if !found {
                break;
            }
            continue;
        };
        if current.dist > furthest_distance(&results) && results.len() == ef {
            break;
        }

        passing.clear();
        for &neighbor in graph.neighbors(current.id) {
            if bitset.contains(neighbor) {
                if !visited.contains(neighbor) {
                    visited.insert(neighbor);
                    passing.push(neighbor);
                }
            } else if !expanded.contains(neighbor) {
                expanded.insert(neighbor);
                for &second_hop in graph.neighbors(neighbor) {
                    if bitset.contains(second_hop) {
                        if !visited.contains(second_hop) {
                            visited.insert(second_hop);
                            passing.push(second_hop);
                        }
                    } else if !expanded.contains(second_hop) {
                        waypoints.push_back(second_hop);
                    }
                }
            }
        }

        process_neighbors_with_look_ahead(
            &passing,
            |node| {
                let dist: OrderedFloat = dist_calc.distance(node).into();
                if dist <= furthest_distance(&results) || results.len() < ef {
                    if dist >= lower_bound && dist < upper_bound {
                        push_result(&mut results, (dist, node).into(), ef);
                    }
                    candidates.push(Reverse((dist, node).into()));
                }
            },
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
    greedy_search_loop!(
        current,
        closest_dist,
        dist_calc,
        prefetch_distance,
        |process_neighbor| {
            let neighbors = graph.neighbors(current);
            process_neighbors_with_look_ahead(
                &neighbors,
                process_neighbor,
                prefetch_distance,
                dist_calc,
            );
        }
    );
    OrderedNode::new(current, closest_dist.into())
}

pub fn greedy_search_borrowed(
    graph: &impl BorrowingGraph,
    start: OrderedNode,
    dist_calc: &impl DistCalculator,
    prefetch_distance: Option<usize>,
) -> OrderedNode {
    let mut current = start.id;
    let mut closest_dist = start.dist.0;
    greedy_search_loop!(
        current,
        closest_dist,
        dist_calc,
        prefetch_distance,
        |process_neighbor| {
            let neighbors = graph.neighbors(current);
            process_neighbors_with_look_ahead(
                neighbors,
                process_neighbor,
                prefetch_distance,
                dist_calc,
            );
        }
    );
    OrderedNode::new(current, closest_dist.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::{BEAM_BATCH16_COUNT_ENABLED, BEAM_BATCH16_FLUSHES};
    use crate::vector::flat::storage::FlatFloatStorage;
    use crate::vector::hnsw::HNSW;
    use crate::vector::hnsw::builder::{HnswBuildParams, HnswQueryParams};
    use crate::vector::storage::{DistCalculator, VectorStore};
    use crate::vector::v3::subindex::IvfSubIndex;
    use arrow_array::{ArrayRef, FixedSizeListArray, Float16Array};
    use half::f16;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_linalg::distance::DistanceType;
    use lance_linalg::distance::dot_f16::amx_fp16_available;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::sync::atomic::Ordering::Relaxed;

    struct ChainGraph {
        neighbors: Vec<Vec<u32>>,
    }

    impl BorrowingGraph for ChainGraph {
        fn len(&self) -> usize {
            self.neighbors.len()
        }

        fn neighbors(&self, key: u32) -> &[u32] {
            &self.neighbors[key as usize]
        }
    }

    struct ZeroDistance;

    impl DistCalculator for ZeroDistance {
        fn distance(&self, _id: u32) -> f32 {
            0.0
        }

        fn distance_all(&self, _k_hint: usize) -> Vec<f32> {
            Vec::new()
        }
    }

    /// Passing components joined only through chains of two masked nodes
    /// must still all be found (from review: without waypoint expansion
    /// only the seeded nodes return).
    #[test]
    fn test_acorn_reaches_across_masked_chains() {
        const PASSING_COUNT: usize = 20;
        const FAILING_COUNT: usize = (PASSING_COUNT - 1) * 2;
        let mut neighbors = vec![Vec::new(); PASSING_COUNT + FAILING_COUNT];
        for index in 0..PASSING_COUNT - 1 {
            let left = index as u32;
            let first_failing = (PASSING_COUNT + index * 2) as u32;
            let second_failing = first_failing + 1;
            let right = left + 1;

            neighbors[left as usize].push(first_failing);
            neighbors[first_failing as usize].extend([left, second_failing]);
            neighbors[second_failing as usize].extend([first_failing, right]);
            neighbors[right as usize].push(second_failing);
        }
        let graph = ChainGraph { neighbors };
        let params = HnswQueryParams {
            ef: 30,
            lower_bound: None,
            upper_bound: None,
            dist_q_c: 0.0,
            use_acorn: false,
        };
        let entry = OrderedNode::new(0, 0.0.into());

        let mut mask_generator = VisitedGenerator::new(graph.len());
        let mut mask = mask_generator.generate(graph.len());
        for id in 0..PASSING_COUNT as u32 {
            mask.insert(id);
        }

        let mut acorn_visited_generator = VisitedGenerator::new(graph.len());
        let mut acorn_expanded_generator = VisitedGenerator::new(graph.len());
        let acorn_results = beam_search_acorn(
            &graph,
            &entry,
            &params,
            &ZeroDistance,
            &mask,
            None,
            &mut acorn_visited_generator.generate(graph.len()),
            &mut acorn_expanded_generator.generate(graph.len()),
        );

        let mut basic_visited_generator = VisitedGenerator::new(graph.len());
        let basic_results = beam_search_borrowed(
            &graph,
            &entry,
            &params,
            &ZeroDistance,
            Some(&mask),
            None,
            &mut basic_visited_generator.generate(graph.len()),
        );

        assert_eq!(basic_results.len(), PASSING_COUNT);
        assert_eq!(acorn_results.len(), PASSING_COUNT);
        assert!(acorn_results.iter().all(|node| mask.contains(node.id)));
    }

    /// Deterministic random f32 vectors in `[-1, 1)`, flat `n * dim` layout.
    fn gen_vectors(seed: u64, n: usize, dim: usize) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n * dim)
            .map(|_| rng.random_range(-1.0f32..1.0))
            .collect()
    }

    // ===== Flat fp16 (AMX-FP16) HNSW query-path tests & benchmark =====
    //
    // The AMX-FP16 override lives on the flat fp16 distance calculator, so it
    // fires on the real user query path (table.search -> HNSW search ->
    // FlatFloatDistanceCalc::distance_batch_16).

    /// Flat (unquantized) fp16 storage over `values`, Dot metric: the symmetric
    /// fp16-vs-fp16 comparison an `IvfHnswFlat` index over an fp16 column runs.
    fn build_flat_f16_storage(values: &[f32], dim: usize) -> FlatFloatStorage {
        let f16vals = Float16Array::from_iter_values(values.iter().map(|&v| f16::from_f32(v)));
        let fsl = FixedSizeListArray::try_new_from_values(f16vals, dim as i32).unwrap();
        FlatFloatStorage::new(fsl, DistanceType::Dot)
    }

    fn flat_f16_hnsw_topk(
        hnsw: &HNSW,
        storage: &FlatFloatStorage,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Vec<u32> {
        let q = Arc::new(Float16Array::from_iter_values(
            query.iter().map(|&v| f16::from_f32(v)),
        )) as ArrayRef;
        let params = HnswQueryParams {
            ef,
            lower_bound: None,
            upper_bound: None,
            dist_q_c: 0.0,
            use_acorn: false,
        };
        hnsw.search_basic(q, k, &params, None, storage)
            .unwrap()
            .into_iter()
            .map(|n| n.id)
            .collect()
    }

    /// Exhaustive top-`k` under the same fp16 Dot distance (graph ground truth).
    fn brute_force_flat_f16_topk(storage: &FlatFloatStorage, query: &[f32], k: usize) -> Vec<u32> {
        let q = Arc::new(Float16Array::from_iter_values(
            query.iter().map(|&v| f16::from_f32(v)),
        )) as ArrayRef;
        let dc = storage.dist_calculator(q, 0.0);
        let mut all: Vec<(f32, u32)> = (0..storage.len() as u32)
            .map(|id| (dc.distance(id), id))
            .collect();
        all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        all.into_iter().take(k).map(|(_, id)| id).collect()
    }

    /// Recall parity for the real fp16 query path. Builds HNSW+Flat(fp16, Dot),
    /// runs queries, compares recall@k against brute-force fp16 ground truth, and
    /// proves both that the 16-wide batch flush path is exercised during *search*
    /// and (when not disabled) that it maps onto the AMX-FP16 tile kernel.
    ///
    /// Re-run with `LANCE_DISABLE_AMX=1` for the scalar/AVX-512-FP16 fallback
    /// recall (out-of-band comparison; the two should be ~identical — tile
    /// accumulation order only perturbs distances at the 1e-4 level).
    #[test]
    #[allow(clippy::print_stderr)]
    fn test_flat_f16_hnsw_recall_and_amx_activation() {
        let (n, dim, nq, k, ef) = (2000usize, 128usize, 100usize, 10usize, 100usize);
        let values = gen_vectors(1, n, dim);
        let storage = build_flat_f16_storage(&values, dim);

        // Pin construction to 1 thread so the build is deterministic here.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let hnsw = pool
            .install(|| HNSW::index_vectors(&storage, HnswBuildParams::default()))
            .unwrap();

        let qvals = gen_vectors(2, nq, dim);
        BEAM_BATCH16_FLUSHES.store(0, Relaxed);
        BEAM_BATCH16_COUNT_ENABLED.store(true, Relaxed);
        let mut hits = 0usize;
        for i in 0..nq {
            let q = &qvals[i * dim..(i + 1) * dim];
            let got = flat_f16_hnsw_topk(&hnsw, &storage, q, k, ef);
            let truth: HashSet<u32> = brute_force_flat_f16_topk(&storage, q, k)
                .into_iter()
                .collect();
            hits += got.iter().filter(|id| truth.contains(id)).count();
        }
        BEAM_BATCH16_COUNT_ENABLED.store(false, Relaxed);
        let flushes = BEAM_BATCH16_FLUSHES.load(Relaxed);
        let recall = hits as f64 / (nq * k) as f64;

        eprintln!(
            "[flat_f16] recall@{k}={recall:.4} search_flushes={flushes} amx_fp16_available={}",
            amx_fp16_available()
        );
        assert!(recall > 0.90, "fp16 HNSW recall unexpectedly low: {recall}");
        // The batched traversal must engage exactly when a fused kernel backs it.
        // With AMX-FP16 usable the 16-wide flush path has to be exercised (runtime
        // probe — the `kernel_support` cfg is set only for the lance-linalg crate,
        // so it cannot gate here). Without it (no hardware, or LANCE_DISABLE_AMX
        // set) the search must take the original per-neighbor traversal and never
        // buffer at all, which is what keeps this change free for backends and
        // hosts that cannot use the kernel.
        if amx_fp16_available() {
            assert!(
                flushes > 0,
                "16-wide batch flush path never hit during search"
            );
        } else {
            assert_eq!(
                flushes, 0,
                "neighbors were buffered even though no fused kernel is available"
            );
            eprintln!("[flat_f16] AMX-FP16 unavailable; search used the scalar traversal");
        }
    }

    /// Wall-clock HNSW+Flat(fp16) benchmark: construction time (all-core), then
    /// query throughput/latency over a batch of `search()` calls (the metric
    /// that matters, since AMX-FP16 fires on the query path) at each of three
    /// concurrency settings. `#[ignore]` -- run:
    ///   cargo test -p lance-index --release --features lance-linalg/amx \
    ///     flat_f16_search_bench -- --ignored --nocapture
    /// Without the `amx` feature this still runs, on the original per-neighbor
    /// traversal, which is the pre-patch baseline. Toggle AMX at runtime with
    /// `LANCE_DISABLE_AMX=1` (one setting per process — it is cached
    /// process-wide). Tune with `BENCH_N` / `BENCH_DIM` / `BENCH_NQ` /
    /// `BENCH_EF` / `BENCH_THREADS` (comma-separated; default `<ncpu>,32,1`).
    ///
    /// The index is built once (all-core) and then searched under each thread
    /// count via an explicit rayon pool, so a single process yields the whole
    /// concurrency sweep for one (dim, AMX) cell.
    #[test]
    #[ignore]
    #[allow(clippy::print_stderr)]
    fn hnsw_flat_f16_search_bench() {
        use rayon::prelude::*;
        use std::time::Instant;

        let env_usize = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d)
        };
        let n = env_usize("BENCH_N", 50_000);
        let dim = env_usize("BENCH_DIM", 128);
        let nq = env_usize("BENCH_NQ", 20_000);
        let ef = env_usize("BENCH_EF", 100);
        let k = 10usize;
        let ncpu = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let thread_counts: Vec<usize> = std::env::var("BENCH_THREADS")
            .ok()
            .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![ncpu, 32, 1]);

        let values = gen_vectors(7, n, dim);
        let storage = build_flat_f16_storage(&values, dim);

        // Build once, all-core, so construction time is independent of the
        // search-concurrency sweep. Both build and search take the AMX-FP16 path
        // when enabled.
        let build_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(ncpu)
            .build()
            .unwrap();
        let t0 = Instant::now();
        let hnsw = build_pool
            .install(|| HNSW::index_vectors(&storage, HnswBuildParams::default()))
            .unwrap();
        let build_dt = t0.elapsed();
        std::hint::black_box(&hnsw);
        eprintln!(
            "[flat_f16_bench] n={n} dim={dim} ef={ef} amx_fp16={} | build_allcore({ncpu}t)={:.3}s ({:.0} vec/s)",
            amx_fp16_available(),
            build_dt.as_secs_f64(),
            n as f64 / build_dt.as_secs_f64(),
        );

        let qvals = gen_vectors(11, nq, dim);
        let queries: Vec<&[f32]> = (0..nq).map(|i| &qvals[i * dim..(i + 1) * dim]).collect();

        // Each concurrency level is measured for a fixed wall-time budget
        // (repeating the query set), so single-thread and 384-thread runs take
        // comparable time and all report stable numbers regardless of dim.
        let budget = std::time::Duration::from_secs_f64(
            std::env::var("BENCH_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.5),
        );
        for &nthreads in &thread_counts {
            if nthreads == 0 || nthreads > ncpu {
                continue;
            }
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nthreads)
                .build()
                .unwrap();
            let run_pass = || -> usize {
                pool.install(|| {
                    queries
                        .par_iter()
                        .map(|q| flat_f16_hnsw_topk(&hnsw, &storage, q, k, ef).len())
                        .sum()
                })
            };
            run_pass(); // warm up (page-in, thread spin-up) before timing
            let t1 = Instant::now();
            let mut served = 0usize;
            let mut done = 0usize;
            while t1.elapsed() < budget {
                served += run_pass();
                done += nq;
            }
            let search_dt = t1.elapsed();
            std::hint::black_box(served);
            let qps = done as f64 / search_dt.as_secs_f64();
            let lat_us = search_dt.as_secs_f64() * 1e6 / done as f64;
            eprintln!(
                "[flat_f16_bench]   search_threads={nthreads:>3} queries={done:>8}: qps={qps:>9.0} lat={lat_us:>7.1}us/q",
            );
        }
    }
}
