// Copyright 2023 Lance Developers.
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

use std::{
    any::Any,
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, HashSet},
    sync::Arc,
};

use arrow_array::{
    builder::{Float32Builder, UInt64Builder},
    ArrayRef, Float32Array, RecordBatch, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt, TryStreamExt};
use object_store::path::Path;
use ordered_float::OrderedFloat;

use super::row_vertex::{RowVertex, RowVertexSerDe};
use crate::{
    dataset::{Dataset, ROW_ID},
    index::{
        vector::{
            graph::{GraphReadParams, PersistedGraph},
            SCORE_COL,
        },
        Index,
    },
    io::deletion::LruDeletionVectorStore,
    Result,
};
use crate::{
    index::{
        vector::VectorIndex,
        vector::{
            graph::{Graph, VertexWithDistance},
            Query,
        },
    },
    io::object_reader::ObjectReader,
    Error,
};

/// DiskANN search state.
pub(crate) struct SearchState {
    /// Visited vertices.
    pub visited: HashSet<usize>,

    /// Candidates. mapping: `<distance, vertex_id>`, ordered by
    /// the distance to the query vector.
    ///
    /// Different to the heap is that, candidates might contain visited vertices
    /// and unvisited vertices.
    candidates: BTreeMap<OrderedFloat<f32>, usize>,

    /// Heap maintains the unvisited vertices, ordered by the distance.
    heap: BinaryHeap<Reverse<VertexWithDistance>>,

    /// Track the ones that have been computed distance with and pushed to heap already.
    ///
    /// This is different to visited, mostly because visited has a different meaning in the
    /// paper, which is the one that has been popped from the heap.
    /// But we wanted to avoid repeatly computing `argmin` over the heap, so we need another
    /// meaning for visited.
    heap_visited: HashSet<usize>,

    /// Search size, `L` parameter in the paper. L must be greater or equal than k.
    l: Option<usize>,

    /// Number of results to return.
    //TODO: used during search.
    #[allow(dead_code)]
    k: Option<usize>,
}

impl SearchState {
    /// Creates a new search state.
    pub(crate) fn new(k: Option<usize>, l: Option<usize>) -> Self {
        Self {
            visited: HashSet::new(),
            candidates: BTreeMap::new(),
            heap: BinaryHeap::new(),
            heap_visited: HashSet::new(),
            k,
            l,
        }
    }

    /// Return the next unvisited vertex.
    fn pop(&mut self) -> Option<usize> {
        while let Some(vertex) = self.heap.pop() {
            // TODO: what if there are two who have the same distance??
            if !self.candidates.contains_key(&vertex.0.distance) {
                // The vertex has been removed from the candidate lists,
                // from [`push()`].
                continue;
            }

            self.visited.insert(vertex.0.id);
            return Some(vertex.0.id);
        }

        None
    }

    /// Drain up to N unvisited vertices
    fn drain(&mut self, n: usize) -> Vec<usize> {
        let mut result = Vec::with_capacity(n);
        while let Some(vertex) = self.heap.pop() {
            if !self.candidates.contains_key(&vertex.0.distance) {
                // The vertex has been removed from the candidate lists,
                // from [`push()`].
                continue;
            }

            self.visited.insert(vertex.0.id);
            result.push(vertex.0.id);
            if result.len() >= n {
                break;
            }
        }

        result
    }

    /// Push a new (unvisited) vertex into the search state.
    fn push(&mut self, vertex_id: usize, distance: f32) {
        assert!(!self.visited.contains(&vertex_id));
        self.heap_visited.insert(vertex_id);
        self.heap
            .push(Reverse(VertexWithDistance::new(vertex_id, distance)));
        self.candidates.insert(OrderedFloat(distance), vertex_id);
        if let Some(limit) = self.l {
            if self.candidates.len() > limit {
                self.candidates.pop_last();
            }
        }
    }

    /// Mark a vertex as visited.
    fn visit(&mut self, vertex_id: usize) {
        self.visited.insert(vertex_id);
    }

    /// Returns true if the vertex has been visited.
    fn is_visited(&self, vertex_id: usize) -> bool {
        self.visited.contains(&vertex_id) || self.heap_visited.contains(&vertex_id)
    }
}

/// Greedy search.
///
/// Algorithm 1 in the paper.
///
/// Parameters:
/// - start: The starting vertex.
/// - query: The query vector.
/// - k: The number of nearest neighbors to return.
/// - search_size: Search list size, L in the paper.
pub(crate) async fn greedy_search(
    graph: &(dyn Graph + Send + Sync),
    start: usize,
    query: &[f32],
    k: usize,
    search_size: usize, // L in the paper.
) -> Result<SearchState> {
    // L in the paper.
    // A map from distance to vertex id.
    let mut state = SearchState::new(Some(k), Some(search_size));

    let dist = graph.distance_to(query, start).await?;
    state.push(start, dist);
    while let Some(id) = state.pop() {
        state.visit(id);

        let neighbors = graph.neighbors(id).await?;
        for neighbor_id in neighbors.values() {
            let neighbor_id = *neighbor_id as usize;
            if state.is_visited(neighbor_id) {
                // Already visited.
                continue;
            }
            let dist = graph.distance_to(query, neighbor_id).await?;
            state.push(neighbor_id, dist);
        }
    }

    Ok(state)
}

async fn load_next_neighbors(
    graph: Arc<dyn Graph + Send + Sync>,
    state: &mut SearchState,
    query: &[f32],
    max_num_neighbors: usize,
) -> Result<()> {
    // Keep it to a max so that we don't have too much latency
    // Otherwise, each round of neighbors may get progressively larger until
    // we are processing thousands per call of this method.
    let current_round_count = std::cmp::min(state.heap.len(), max_num_neighbors);

    let ids = state.drain(current_round_count);
    for id in &ids {
        state.visit(*id);
    }
    let neighbors = futures::stream::iter(ids)
        .map(|id| {
            let graph = graph.clone();
            async move {
                match graph.neighbors(id.clone()).await {
                    Ok(neighbors) => Ok(neighbors
                        .into_iter()
                        .map(|maybe_id| {
                            maybe_id.ok_or(Error::Internal {
                                message: "Neighbor ID is null".to_string(),
                            })
                        })
                        .collect::<Vec<Result<u32>>>()),
                    Err(e) => Err(e),
                }
            }
        })
        .buffer_unordered(5)
        .map_ok(|neighbor_ids| futures::stream::iter(neighbor_ids))
        .try_flatten_unordered(10);
    let mut neighbor_distances = neighbors
        .map_ok(|id| {
            let graph = graph.clone();
            async move {
                let distance = graph.distance_to(query, id as usize).await?;
                Ok((id, distance))
            }
        })
        .try_buffer_unordered(10);

    while let Some((id, distance)) = neighbor_distances.try_next().await? {
        if state.is_visited(id as usize) {
            continue;
        }
        state.push(id as usize, distance);
    }

    Ok(())
}

/// Create a stream of search results.
///
/// The stream will yield a (row_id, distance) pair.
///
/// The stream will not be strictly ordered by distance, but will be approximately
/// ordered by distance.
pub(crate) async fn greedy_search_stream(
    graph: Arc<dyn Graph + Send + Sync>,
    start: usize,
    query: &[f32],
) -> Result<BoxStream<'static, Result<(usize, f32)>>> {
    let mut state = SearchState::new(None, None);

    let dist = graph.distance_to(query, start).await?;
    state.push(start, dist);

    let query: Arc<Vec<f32>> = Arc::new(query.to_vec());

    Ok(futures::stream::unfold(state, move |mut state| {
        let graph = graph.clone();
        let query = query.clone();
        async move {
            // If we have run out of candidates, try loading some more. We limit to
            // 30 neighbors at a time to avoid too much latency.
            if state.candidates.is_empty() {
                match load_next_neighbors(graph, &mut state, &query, 30).await {
                    Ok(_) => {}
                    Err(e) => {
                        return Some((Err(e), state));
                    }
                };
            }

            if let Some(candidate) = state.candidates.pop_first() {
                // If we have candidates, yield those
                Some((Ok((candidate.1, *candidate.0)), state))
            } else {
                None
            }
        }
    })
    .boxed())
}

pub struct DiskANNIndex {
    graph: PersistedGraph<RowVertex>,

    deletion_cache: Arc<LruDeletionVectorStore>,
}

impl std::fmt::Debug for DiskANNIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DiskANNIndex")
    }
}

impl DiskANNIndex {
    /// Creates a new DiskANN index.

    pub async fn try_new(
        dataset: Arc<Dataset>,
        index_column: &str,
        graph_path: &Path,
        deletion_cache: Arc<LruDeletionVectorStore>,
    ) -> Result<Self> {
        let params = GraphReadParams::default();
        let serde = Arc::new(RowVertexSerDe::new());
        let graph =
            PersistedGraph::try_new(dataset, index_column, graph_path, params, serde).await?;
        Ok(Self {
            graph,
            deletion_cache,
        })
    }
}

impl Index for DiskANNIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[async_trait]
impl VectorIndex for DiskANNIndex {
    async fn search(&self, query: &Query, k: usize) -> Result<RecordBatch> {
        let state = greedy_search(self.graph.as_ref(), 0, query.key.values(), k, k * 2).await?;
        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new(SCORE_COL, DataType::Float32, false),
        ]));

        let mut candidates = Vec::with_capacity(query.k);
        for (score, row) in state.candidates {
            if candidates.len() == query.k {
                break;
            }
            if !self.deletion_cache.as_ref().is_deleted(row as u64).await? {
                candidates.push((score, row));
            }
        }

        let row_ids: UInt64Array = candidates
            .iter()
            .take(k)
            .map(|(_, id)| *id as u64)
            .collect();
        let scores: Float32Array = candidates.iter().take(query.k).map(|(d, _)| **d).collect();

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(row_ids) as ArrayRef, Arc::new(scores) as ArrayRef],
        )?;
        Ok(batch)
    }

    async fn search_stream(
        &self,
        query: &Query,
    ) -> Result<BoxStream<'static, Result<RecordBatch>>> {
        let stream = greedy_search_stream(self.graph.clone(), 0, query.key.values()).await?;
        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new(SCORE_COL, DataType::Float32, false),
        ]));

        let stream = stream
            // TODO: make chunk size a parameter
            .chunks(20)
            .map(move |results| {
                let schema = schema.clone();

                let mut row_ids: UInt64Builder = UInt64Builder::with_capacity(results.len());
                let mut scores: Float32Builder = Float32Builder::with_capacity(results.len());

                for result in results {
                    match result {
                        // TODO: why is row_id a usize earlier?
                        Ok((row_id, score)) => {
                            row_ids.append_value(row_id as u64);
                            scores.append_value(score);
                        }
                        Err(e) => {
                            return Err(e);
                        }
                    }
                }

                let row_ids = row_ids.finish();
                let scores = scores.finish();

                let batch = RecordBatch::try_new(
                    schema,
                    vec![Arc::new(row_ids) as ArrayRef, Arc::new(scores) as ArrayRef],
                )?;
                Ok(batch)
            });

        Ok(stream.boxed())
    }

    fn is_loadable(&self) -> bool {
        false
    }

    async fn load(
        &self,
        _reader: &dyn ObjectReader,
        _offset: usize,
        _length: usize,
    ) -> Result<Arc<dyn VectorIndex>> {
        Err(Error::Index {
            message: "DiskANNIndex is not loadable".to_string(),
        })
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_search_state() {
        let k: usize = 10;
        let l: usize = 20;

        let mut state = SearchState::new(Some(k), Some(l));
        for i in (0..40).rev() {
            state.push(i, i as f32);
        }

        assert_eq!(state.visited.len(), 0);
        assert_eq!(state.heap.len(), 40);
        assert_eq!(state.candidates.len(), 20);

        let mut i = 0;
        while let Some(next) = state.pop() {
            state.visited.insert(next);
            assert_eq!(next, i);
            i += 1;
        }
        assert_eq!(i, 20);

        assert!(state.heap.is_empty());
        assert_eq!(state.candidates.len(), 20);
    }
}
