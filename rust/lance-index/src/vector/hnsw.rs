// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! HNSW graph implementation.
//!
//! Hierarchical Navigable Small World (HNSW).
//!

use arrow_schema::{DataType, Field};
use itertools::Itertools;
use lance_core::deepsize::DeepSizeOf;
use serde::{Deserialize, Serialize};

use self::builder::HnswBuildParams;
use super::graph::OrderedNode;
use super::storage::VectorStore;

pub mod builder;
pub mod index;
pub mod online;

pub use builder::HNSW;
pub use index::HNSWIndex;
pub use online::OnlineHnswBuilder;

const HNSW_TYPE: &str = "HNSW";
const VECTOR_ID_COL: &str = "__vector_id";
const POINTER_COL: &str = "__pointer";

use std::sync::LazyLock;

/// POINTER field.
///
pub static POINTER_FIELD: LazyLock<Field> =
    LazyLock::new(|| Field::new(POINTER_COL, DataType::UInt32, true));

/// Id of the vector in the `VectorStorage`.
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
/// This uses the paper's `extendCandidates = false` and
/// `keepPrunedConnections = false` configuration. Candidate extension can
/// substantially increase construction work on clustered data, so it is not
/// enabled implicitly; callers supply the complete candidate set that should
/// participate in this selection.
///
/// # NOTE
/// The results are not ordered.
pub(crate) fn select_neighbors_heuristic(
    storage: &impl VectorStore,
    candidates: &[OrderedNode],
    k: usize,
) -> Vec<OrderedNode> {
    if candidates.len() <= k {
        return candidates.iter().cloned().collect_vec();
    }

    select_neighbors_heuristic_owned(storage, candidates.to_vec(), k)
}

pub(crate) fn select_neighbors_heuristic_owned(
    storage: &impl VectorStore,
    mut candidates: Vec<OrderedNode>,
    k: usize,
) -> Vec<OrderedNode> {
    if candidates.len() <= k {
        return candidates;
    }

    candidates.sort_unstable();

    let mut results: Vec<OrderedNode> = Vec::with_capacity(k);
    for u in candidates.iter() {
        if results.len() >= k {
            break;
        }

        if results.is_empty() || storage.prefers_candidate(u, &results) {
            results.push(u.clone());
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use arrow_array::{FixedSizeListArray, Float32Array};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_linalg::distance::DistanceType;

    use super::select_neighbors_heuristic_owned;
    use crate::vector::flat::storage::FlatFloatStorage;
    use crate::vector::graph::OrderedNode;
    use crate::vector::storage::VectorStore;

    /// A reciprocal candidate must join the complete old-plus-new set before
    /// Algorithm 4 runs. A farther, directionally diverse connection can then
    /// replace closer connections that all point in the same direction.
    #[test]
    fn test_selection_retains_farther_diverse_reciprocal_candidate() {
        let vectors = Float32Array::from(vec![
            0.0, 0.0, // node receiving the reciprocal connection
            1.0, 0.0, // close candidate
            2.0, 0.0, // redundant candidate
            3.0, 0.0, // redundant candidate
            4.0, 0.0, // redundant candidate
            0.0, 5.0, // farther but directionally diverse candidate
        ]);
        let vectors = FixedSizeListArray::try_new_from_values(vectors, 2).unwrap();
        let storage = FlatFloatStorage::new(vectors, DistanceType::L2);
        let candidates = (1..=5)
            .map(|id| OrderedNode::new(id, storage.dist_between(0, id).into()))
            .collect();

        let selected = select_neighbors_heuristic_owned(&storage, candidates, 4);
        assert_eq!(
            selected.iter().map(|node| node.id).collect::<Vec<_>>(),
            vec![1, 5]
        );
    }
}
