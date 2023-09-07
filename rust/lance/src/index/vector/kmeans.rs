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

use std::sync::Arc;

use arrow_array::{builder::Float32Builder, Float32Array};
use rand::{seq::IteratorRandom, Rng};

use crate::index::vector::MetricType;
use crate::{
    utils::kmeans::{KMeans, KMeansParams},
    Result,
};

/// Train KMeans model and returns the centroids of each cluster.
#[allow(clippy::too_many_arguments)]
pub async fn train_kmeans(
    array: &Float32Array,
    centroids: Option<Arc<Float32Array>>,
    dimension: usize,
    k: usize,
    max_iterations: u32,
    redos: usize,
    mut rng: impl Rng,
    metric_type: MetricType,
) -> Result<Float32Array> {
    let num_rows = array.len() / dimension;
    if num_rows < k {
        return Err(crate::Error::Index{message: format!(
            "KMeans: can not train {k} centroids with {num_rows} vectors, choose a smaller K (< {num_rows}) instead"
        )});
    }
    // Ony sample 256 * num_clusters. See Faiss
    let data = if num_rows > 256 * k {
        println!(
            "Sample {} out of {} to train kmeans of {} dim, {} clusters",
            256 * k,
            array.len() / dimension,
            dimension,
            k,
        );
        let sample_size = 256 * k;
        let chosen = (0..num_rows).choose_multiple(&mut rng, sample_size);
        let mut builder = Float32Builder::with_capacity(sample_size * dimension);
        for idx in chosen.iter() {
            let s = array.slice(idx * dimension, dimension);
            builder.append_slice(s.values());
        }
        builder.finish()
    } else {
        array.clone()
    };

    let params = KMeansParams {
        max_iters: max_iterations,
        metric_type,
        centroids,
        redos,
        ..Default::default()
    };
    let model = KMeans::new_with_params(&data, dimension, k, &params).await?;
    Ok(model.centroids.as_ref().clone())
}
