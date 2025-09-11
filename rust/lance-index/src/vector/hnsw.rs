// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! HNSW graph implementation.
//!
//! Hierarchical Navigable Small World (HNSW).
//!

use arrow_schema::{DataType, Field};
use deepsize::DeepSizeOf;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use self::builder::HnswBuildParams;
use super::graph::{OrderedFloat, OrderedNode};
use super::storage::VectorStore;

pub mod builder;
pub mod index;

pub use builder::HNSW;
pub use index::HNSWIndex;

const HNSW_TYPE: &str = "HNSW";
const VECTOR_ID_COL: &str = "__vector_id";
const POINTER_COL: &str = "__pointer";

use std::sync::LazyLock;

/// POINTER field.
///
pub static POINTER_FIELD: LazyLock<Field> =
    LazyLock::new(|| Field::new(POINTER_COL, DataType::UInt32, true));

/// Id of the vector in the [VectorStorage].
pub static VECTOR_ID_FIELD: LazyLock<Field> =
    LazyLock::new(|| Field::new(VECTOR_ID_COL, DataType::UInt32, true));

#[derive(Debug, Clone, Serialize, Deserialize, DeepSizeOf)]
pub struct HnswMetadata {
    pub entry_point: u32,
    pub params: HnswBuildParams,
    pub level_offsets: Vec<usize>,
}

impl Default for HnswMetadata {
    fn default() -> Self {
        let params = HnswBuildParams::default();
        let level_offsets = vec![0; params.max_level as usize];
        Self {
            entry_point: 0,
            params,
            level_offsets,
        }
    }
}

/// Algorithm 4 in the HNSW paper.
///
/// # NOTE
/// The results are not ordered.
fn select_neighbors_heuristic(
    storage: &impl VectorStore,
    candidates: &[OrderedNode],
    k: usize,
) -> Vec<OrderedNode> {
    if candidates.len() <= k {
        return candidates.iter().cloned().collect_vec();
    }
    let mut candidates = candidates.to_vec();
    candidates.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());

    let mut results: Vec<OrderedNode> = Vec::with_capacity(k);
    for u in candidates.iter() {
        if results.len() >= k {
            break;
        }

        if results.is_empty()
            || results
                .iter()
                .all(|v| u.dist < OrderedFloat(storage.dist_between(u.id, v.id)))
        {
            results.push(u.clone());
        }
    }
    results
}
