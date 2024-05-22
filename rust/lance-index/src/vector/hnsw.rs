// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! HNSW graph implementation.
//!
//! Hierarchical Navigable Small World (HNSW).
//!

pub mod builder;
use arrow_schema::{DataType, Field};
pub use builder::HNSW;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use self::builder::HnswBuildParams;

use super::graph::{OrderedFloat, OrderedNode, VectorStore};

const HNSW_TYPE: &str = "HNSW";
const VECTOR_ID_COL: &str = "__vector_id";
const POINTER_COL: &str = "__pointer";

lazy_static::lazy_static! {
    /// POINTER field.
    ///
    pub static ref POINTER_FIELD: Field = Field::new(POINTER_COL, DataType::UInt32, true);

    /// Id of the vector in the [VectorStorage].
    pub static ref VECTOR_ID_FIELD: Field = Field::new(VECTOR_ID_COL, DataType::UInt32, true);
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HnswMetadata {
    pub entry_point: u32,
    pub params: HnswBuildParams,
    pub level_offsets: Option<Vec<usize>>,
}

/// Select neighbors from the ordered candidate list.
///
/// Algorithm 3 in the HNSW paper.
///
/// WARNING: Internal API,  API stability is not guaranteed
pub fn select_neighbors(
    orderd_candidates: &[OrderedNode],
    k: usize,
) -> impl Iterator<Item = &OrderedNode> + '_ {
    orderd_candidates.iter().take(k)
}

/// Algorithm 4 in the HNSW paper.
///
/// NOTE: the result is not ordered
///
/// WARNING: Internal API,  API stability is not guaranteed
pub fn select_neighbors_heuristic(
    storage: &impl VectorStore,
    candidates: &[OrderedNode],
    k: usize,
) -> Vec<OrderedNode> {
    if candidates.len() <= k {
        return candidates.iter().cloned().collect_vec();
    }
    let mut candidates = candidates.to_vec();
    candidates.sort_unstable_by(|a, b| b.dist.partial_cmp(&a.dist).unwrap());

    let mut results: Vec<OrderedNode> = Vec::with_capacity(k);
    while !candidates.is_empty() && results.len() < k {
        let u = candidates.pop().unwrap();

        if results.is_empty()
            || results
                .iter()
                .all(|v| u.dist < OrderedFloat(storage.distance_between(u.id, v.id)))
        {
            results.push(u);
        }
    }
    results
}
