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

use arrow_array::{Array, FixedSizeListArray};
use lance_arrow::{ArrowFloatType, FixedSizeListArrayExt, FloatArray};
use log::info;
use rand::{seq::IteratorRandom, Rng};
use snafu::{location, Location};
use std::sync::Arc;

use lance_core::{Error, Result};
use lance_linalg::{
    distance::{Cosine, Dot, MetricType, L2},
    kmeans::{KMeans, KMeansParams},
};

/// Train KMeans model and returns the centroids of each cluster.
#[allow(clippy::too_many_arguments)]
pub async fn train_kmeans<T: ArrowFloatType + Dot + L2 + Cosine>(
    array: &T::ArrayType,
    centroids: Option<Arc<T::ArrayType>>,
    dimension: usize,
    k: usize,
    max_iterations: u32,
    redos: usize,
    mut rng: impl Rng,
    metric_type: MetricType,
    sample_rate: usize,
) -> Result<T::ArrayType> {
    let num_rows = array.len() / dimension;
    if num_rows < k {
        return Err(Error::Index{message: format!(
            "KMeans: can not train {k} centroids with {num_rows} vectors, choose a smaller K (< {num_rows}) instead"
        ),location: location!()});
    }

    // Ony sample sample_rate * num_clusters. See Faiss
    let data = if num_rows > sample_rate * k {
        info!(
            "Sample {} out of {} to train kmeans of {} dim, {} clusters",
            sample_rate * k,
            array.len() / dimension,
            dimension,
            k,
        );
        let sample_size = sample_rate * k;
        let chosen = (0..num_rows).choose_multiple(&mut rng, sample_size);
        let mut builder = Vec::with_capacity(sample_size * dimension);
        for idx in chosen.iter() {
            let s = &array.as_slice()[idx * dimension..(idx + 1) * dimension];
            builder.extend_from_slice(s);
        }
        T::ArrayType::from(builder)
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
    let data = FixedSizeListArray::try_new_from_values(data, dimension as i32)?;
    let model = KMeans::<T>::new_with_params(&data, k, &params).await?;
    Ok(model.centroids.as_ref().clone())
}
