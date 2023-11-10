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
    collections::{BTreeMap, BinaryHeap, HashMap, HashSet},
    sync::Arc,
};

use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use lance_core::{io::Reader, Error, Result, ROW_ID_FIELD};
use lance_index::{
    vector::{Query, DIST_COL},
    Index, IndexType,
};
use object_store::path::Path;
use ordered_float::OrderedFloat;
use serde::Serialize;
use snafu::{location, Location};
use tracing::instrument;

use super::row_vertex::{RowVertex, RowVertexSerDe};
use crate::index::{
    vector::graph::{Graph, VertexWithDistance},
    vector::VectorIndex,
};
use crate::{
    dataset::Dataset,
    index::{
        prefilter::PreFilter,
        vector::graph::{GraphReadParams, PersistedGraph},
    },
};

/// DiskANN search state.
pub struct SearchState {
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
    heap_visisted: HashSet<usize>,

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
            heap_visisted: HashSet::new(),
            k,
            l,
        }
    }

    /// Return the next unvisited vertex.
    fn pop(&mut self) -> Option<usize> {
        while let Some(vertex) = self.heap.pop() {
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

    /// Push a new (unvisited) vertex into the search state.
    fn push(&mut self, vertex_id: usize, distance: f32) {
        assert!(!self.visited.contains(&vertex_id));
        self.heap_visisted.insert(vertex_id);
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
        self.visited.contains(&vertex_id) || self.heap_visisted.contains(&vertex_id)
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
pub async fn greedy_search(
    graph: &(dyn Graph + Send + Sync),
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

    pub async fn try_new(
        dataset: Arc<Dataset>,
        index_column: &str,
        graph_path: &Path,
    ) -> Result<Self> {
        let params = GraphReadParams::default();
        let serde = Arc::new(RowVertexSerDe::new());
        let graph =
            PersistedGraph::try_new(dataset, index_column, graph_path, params, serde).await?;
        Ok(Self { graph })
    }
}

#[derive(Serialize)]
pub struct DiskANNIndexStatistics {
    index_type: String,
    length: usize,
}

impl Index for DiskANNIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn index_type(&self) -> IndexType {
        IndexType::Vector
    }

    fn statistics(&self) -> Result<String> {
        Ok(serde_json::to_string(&DiskANNIndexStatistics {
            index_type: "DiskANNIndex".to_string(),
            length: self.graph.len(),
        })?)
    }
}

#[async_trait]
impl VectorIndex for DiskANNIndex {
    #[instrument(level = "debug", skip_all, name = "DiskANNIndex::search")]
    async fn search(&self, query: &Query, pre_filter: Arc<PreFilter>) -> Result<RecordBatch> {
        let state = greedy_search(
            &self.graph,
            0,
            query.key.as_primitive().values(),
            query.k,
            query.k * 2,
        )
        .await?;
        let schema = Arc::new(Schema::new(vec![
            ROW_ID_FIELD.clone(),
            Field::new(DIST_COL, DataType::Float32, true),
        ]));

        pre_filter.wait_for_ready().await?;

        let mut candidates = Vec::with_capacity(query.k);
        for (distance, row) in state.candidates {
            if candidates.len() == query.k {
                break;
            }
            if pre_filter.check_one(row as u64) {
                candidates.push((distance, row));
            }
        }

        let row_ids: UInt64Array = candidates
            .iter()
            .take(query.k)
            .map(|(_, id)| *id as u64)
            .collect();
        let distances: Float32Array = candidates.iter().take(query.k).map(|(d, _)| **d).collect();

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(row_ids) as ArrayRef,
                Arc::new(distances) as ArrayRef,
            ],
        )?;
        Ok(batch)
    }

    fn is_loadable(&self) -> bool {
        false
    }

    async fn load(
        &self,
        _reader: &dyn Reader,
        _offset: usize,
        _length: usize,
    ) -> Result<Box<dyn VectorIndex>> {
        Err(Error::Index {
            message: "DiskANNIndex is not loadable".to_string(),
            location: location!(),
        })
    }

    fn check_can_remap(&self) -> Result<()> {
        Err(Error::NotSupported {
            source: "DiskANNIndex does not yet support remap".into(),
            location: location!(),
        })
    }

    fn remap(&mut self, _mapping: &HashMap<u64, Option<u64>>) -> Result<()> {
        Err(Error::NotSupported {
            source: "DiskANNIndex does not yet support remap".into(),
            location: location!(),
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
