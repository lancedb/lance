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
use std::ops::Range;
use std::sync::Arc;

use arrow_array::UInt64Array;
use arrow_array::{
    builder::{ListBuilder, UInt64Builder},
    cast::AsArray,
    RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use lance_arrow::FloatToArrayType;
use lance_core::{Error, Result};
use lance_linalg::distance::{Cosine, DistanceFunc, Dot, MetricType, L2};
use num_traits::{AsPrimitive, Float};
use snafu::{location, Location};

pub(crate) mod builder;
mod io;
mod iter;
pub(super) mod storage;

use iter::Iter;
use storage::VectorStorage;

use self::builder::GraphBuilderNode;

const NEIGHBORS_COL: &str = "__neighbors";
const POINTER_COL: &str = "__pointer";

lazy_static::lazy_static! {
    /// NEIGHBORS field.
    pub static ref NEIGHBORS_FIELD: Field = Field::new(NEIGHBORS_COL, DataType::List(Arc::new(Field::new(
        "item", DataType::UInt64, true
    ))), true);
    /// POINTER field.
    pub static ref POINTER_FIELD: Field = Field::new(POINTER_COL, DataType::UInt64, true);
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
pub trait Graph<T: Float = f32> {
    /// Get the number of nodes in the graph.
    fn len(&self) -> usize;

    /// Returns true if the graph is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the neighbors of a graph node, identifyied by the index.
    fn neighbors<'a>(&'a self, key: u64) -> Option<Box<dyn Iterator<Item = &u64> + 'a>>;

    /// Distance from query vector to a node.
    fn distance_to(&self, query: &[T], key: u64) -> f32;

    /// Distance between two nodes in the graph.
    ///
    /// Returns the distance between two nodes as float32.
    fn distance_between(&self, a: u64, b: u64) -> f32;

    /// Create a BFS iterator.
    fn iter(&self) -> self::iter::Iter<'_, T>;
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
    start: &[u64],
    query: &[f32],
    k: usize,
) -> Result<BTreeMap<OrderedFloat, u64>> {
    let mut visited: HashSet<u64> = start.iter().copied().collect();
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

        for &neighbor in neighbors {
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

/// Serializable Graph.
///
/// Type parameters
/// ----------------
/// I : Vertex Index type
/// V : the data type of vector, i.e., ``f32`` or ``f16``.
struct InMemoryGraph<T: FloatToArrayType, V: storage::VectorStorage<T>> {
    /// Use a RecordBatch to store the graph.
    pub nodes: RecordBatch,
    neighbors: Arc<UInt64Array>,
    vectors: V,
    dist_fn: Box<DistanceFunc<T>>,
}

impl<T: FloatToArrayType, V: storage::VectorStorage<T>> Graph<T> for InMemoryGraph<T, V>
where
    T::ArrowType: L2 + Cosine + Dot,
{
    fn neighbors<'a>(&'a self, key: u64) -> Option<Box<dyn Iterator<Item = &u64> + 'a>> {
        let range = self.neighbor_range(key);
        Some(Box::new(self.neighbors.values()[range].iter()))
    }

    fn distance_to(&self, query: &[T], idx: u64) -> f32 {
        let vec = self.vectors.get(idx.as_());
        (self.dist_fn)(query, vec)
    }

    fn distance_between(&self, a: u64, b: u64) -> f32 {
        let from_vec = self.vectors.get(a.as_());
        self.distance_to(from_vec, b)
    }

    fn len(&self) -> usize {
        self.nodes.num_rows()
    }

    fn iter(&self) -> Iter<'_, T> {
        Iter::new(self as &dyn Graph<T>, 0)
    }
}

impl<T: FloatToArrayType, V: VectorStorage<T>> InMemoryGraph<T, V>
where
    T::ArrowType: L2 + Cosine + Dot,
{
    fn from_builder(
        nodes: &BTreeMap<u64, GraphBuilderNode>,
        vectors: V,
        metric_type: MetricType,
    ) -> Self {
        let mut neighbours_builder = ListBuilder::new(UInt64Builder::new());
        let mut pointers_builder = UInt64Builder::new();

        for (_, node) in nodes.iter() {
            pointers_builder.append_value(node.pointer as u64);
            neighbours_builder.append_value(node.neighbors.values().map(|&n| Some(n)));
        }

        let schema = Schema::new(vec![NEIGHBORS_FIELD.clone(), POINTER_FIELD.clone()]);

        let neighbors = Arc::new(neighbours_builder.finish());
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![neighbors.clone(), Arc::new(pointers_builder.finish())],
        )
        .unwrap();

        Self {
            nodes: batch,
            neighbors: neighbors.values().as_primitive().clone().into(),
            vectors,
            dist_fn: metric_type.func::<T>().into(),
        }
    }

    /// The range of neighbours on the var-length list array.
    fn neighbor_range(&self, k: u64) -> Range<usize> {
        let neighbors_col = self.nodes.column_by_name(NEIGHBORS_COL).unwrap();
        let neighbors = neighbors_col.as_list::<i32>();
        let start = neighbors.value_offsets()[k as usize] as usize;
        let end = start + neighbors.value_length(k as usize) as usize;
        start..end
    }
}

#[cfg(test)]
mod tests {}
