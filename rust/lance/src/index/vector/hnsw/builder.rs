// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::{array::AsArray, datatypes::Float32Type};
use arrow_array::Array;
use lance_core::Result;
use lance_index::vector::{
    graph::memory::InMemoryVectorStorage,
    hnsw::{builder::HnswBuildParams, HNSWBuilder, HNSW},
};
use lance_linalg::{distance::MetricType, MatrixView};

pub fn build_hnsw_model(hnsw_params: HnswBuildParams, vectors: Arc<dyn Array>) -> Result<HNSW> {
    let mat = Arc::new(MatrixView::<Float32Type>::try_from(
        vectors.as_fixed_size_list(),
    )?);

    // We have normalized the vectors if the metric type is cosine, so we can use the L2 distance
    let vec_store = Arc::new(InMemoryVectorStorage::new(mat.clone(), MetricType::L2));
    let mut hnsw_builder = HNSWBuilder::with_params(hnsw_params, vec_store);
    let hnsw = hnsw_builder.build()?;

    Ok(hnsw)
}
