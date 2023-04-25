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
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, HashSet},
    sync::Arc,
};

use arrow_array::RecordBatch;
use async_trait::async_trait;
use object_store::path::Path;
use ordered_float::OrderedFloat;

use super::row_vertex::{RowVertex, RowVertexSerDe};
use crate::{
    index::vector::graph::{GraphReadParams, PersistedGraph},
    io::ObjectStore,
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

    /// Search size, `L` parameter in the paper. L must be greater or equal than k.
    l: usize,

    /// Number of results to return.
    //TODO: used during search.
    #[allow(dead_code)]
    k: usize,
}

impl SearchState {
    /// Creates a new search state.
    pub(crate) fn new(k: usize, l: usize) -> Self {
        Self {
            visited: HashSet::new(),
            candidates: BTreeMap::new(),
            heap: BinaryHeap::new(),
            k,
            l,
        }
    }

    /// Return the next unvisited vertex.
    fn pop(&mut self) -> Option<usize> {
        while let Some(vertex) = self.heap.pop() {
            // println!("Pop {} visited {:?}", vertex.0.id, self.visited);

            if self.is_visited(vertex.0.id) || !self.candidates.contains_key(&vertex.0.distance) {
                // The vertex has been removed from the candidate lists,
                // from [`push()`].
                continue;
            }
            self.visited.insert(vertex.0.id);
            return Some(vertex.0.id);
        }

        None
    }

    /// Push a new (unvisited) fvertex into the search state.
    fn push(&mut self, vertex_id: usize, distance: f32) {
        self.heap
            .push(Reverse(VertexWithDistance::new(vertex_id, distance)));
        self.candidates.insert(OrderedFloat(distance), vertex_id);
        if self.candidates.len() > self.l {
            self.candidates.pop_last();
        }
    }

    /// Mark a vertex as visited.
    fn visit(&mut self, vertex_id: usize) {
        self.visited.insert(vertex_id);
    }

    /// Returns true if the vertex has been visited.
    fn is_visited(&self, vertex_id: usize) -> bool {
        self.visited.contains(&vertex_id)
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
    graph: &dyn Graph,
    start: usize,
    query: &[f32],
    k: usize,
    search_size: usize, // L in the paper.
) -> Result<SearchState> {
    // L in the paper.
    // A map from distance to vertex id.
    let mut state = SearchState::new(k, search_size);

    let dist = graph.distance_to(query, start).await?;
    state.push(start, dist);
    while let Some(id) = state.pop() {
        state.visit(id);
        for neighbor_id in graph.neighbors(id).await?.iter() {
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

pub struct DiskANNIndex {
    graph: PersistedGraph<RowVertex>,
}

impl std::fmt::Debug for DiskANNIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DiskANNIndex")
    }
}

impl DiskANNIndex {
    /// Creates a new DiskANN index.

    pub async fn try_new(object_store: &ObjectStore, graph_path: &Path) -> Result<Self> {
        let params = GraphReadParams::default();
        let serde = Arc::new(RowVertexSerDe::new());
        let graph = PersistedGraph::try_new(object_store, graph_path, params, serde).await?;
        Ok(Self { graph })
    }
}

#[async_trait]
impl VectorIndex for DiskANNIndex {
    async fn search(&self, query: &Query) -> Result<RecordBatch> {
        let state = greedy_search(&self.graph, 0, query.key.values(), query.k, query.k * 2).await?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
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
        Err(Error::Index("DiskANNIndex is not loadable".to_string()))
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_search_state() {
        let k: usize = 10;
        let l: usize = 20;

        let mut state = SearchState::new(k, l);
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
