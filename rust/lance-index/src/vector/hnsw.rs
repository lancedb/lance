// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! HNSW graph implementation.
//!
//! Hierarchical Navigable Small World (HNSW).
//!

use arrow_array::ArrayRef;
use arrow_schema::{DataType, Field};
use deepsize::DeepSizeOf;
use itertools::Itertools;
use lance_core::Result;
use lance_file::writer::FileWriter;
use lance_table::io::manifest::ManifestDescribing;
use serde::{Deserialize, Serialize};

use self::builder::HnswBuildParams;
use super::graph::{OrderedFloat, OrderedNode};
use super::storage::{DistCalculator, VectorStore};

pub mod builder;
pub mod index;

pub use builder::HNSW;
pub use index::HNSWIndex;

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

#[derive(Debug, Clone, Default, Serialize, Deserialize, DeepSizeOf)]
pub struct HnswMetadata {
    pub entry_point: u32,
    pub params: HnswBuildParams,
    pub level_offsets: Option<Vec<usize>>,
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
        let dist_cal = storage.dist_calculator_from_id(u.id);

        if results.is_empty()
            || results
                .iter()
                .all(|v| u.dist < OrderedFloat(dist_cal.distance(v.id)))
        {
            results.push(u.clone());
        }
    }
    results
}

/// Build and Write HNSW graph to a file.
pub async fn build_and_write_hnsw(
    params: HnswBuildParams,
    vectors: ArrayRef,
    mut writer: FileWriter<ManifestDescribing>,
) -> Result<usize> {
    let hnsw = params.build(vectors).await?;
    let length = hnsw.write(&mut writer).await?;
    Result::Ok(length)
}
