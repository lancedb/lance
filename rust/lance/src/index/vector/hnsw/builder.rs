// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::array::AsArray;
use arrow_array::Array;
use lance_core::Result;
use lance_index::vector::{
    flat::storage::FlatStorage,
    hnsw::builder::{HnswBuildParams, HNSW},
};
use lance_linalg::distance::DistanceType;

pub async fn build_hnsw_model(
    hnsw_params: HnswBuildParams,
    vectors: Arc<dyn Array>,
) -> Result<HNSW> {
    // We have normalized the vectors if the metric type is cosine, so we can use the L2 distance
    let vec_store = Arc::new(FlatStorage::new(
        vectors.as_fixed_size_list().clone(),
        DistanceType::L2,
    ));
    HNSW::build_with_storage(DistanceType::L2, hnsw_params, vec_store).await
}
