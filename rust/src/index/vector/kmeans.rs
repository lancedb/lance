// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use arrow_array::{
    builder::Float32Builder, cast::as_primitive_array, types::Float32Type, Array,
    FixedSizeListArray, Float32Array,
};
use rand::{seq::IteratorRandom, Rng};

use crate::{
    arrow::FixedSizeListArrayExt,
    utils::kmeans::{KMeans, KMeansParams},
    Result,
};

/// A fallback default implementation of KMeans, if not accelerator found.
async fn train_kmeans_fallback(
    array: &Float32Array,
    dimension: usize,
    k: u32,
    max_iters: u32,
) -> Result<Float32Array> {
    let data = Arc::new(FixedSizeListArray::try_new(array, dimension as i32)?);
    let mut params = KMeansParams::default();
    params.max_iters = max_iters;
    let model = KMeans::new_with_params(data, k, &params).await;
    let centroids = model.centroids.values();
    let floats: &Float32Array = as_primitive_array(centroids.as_ref());
    Ok(floats.clone())
}

/// Train kmeans model and returns the centroids of each cluster.
pub async fn train_kmeans(
    array: &Float32Array,
    dimension: usize,
    k: u32,
    max_iterations: u32,
    mut rng: impl Rng,
) -> Result<Float32Array> {
    let num_rows = array.len() / dimension;
    // Ony sample 256 * num_clusters. See Faiss
    let data = if num_rows > 256 * k as usize {
        println!(
            "Sample {} out of {} to train kmeans of {} dim, {} clusters",
            256 * k,
            array.len() / dimension,
            dimension,
            k,
        );
        let sample_size = 256 * k as usize;
        let chosen = (0..num_rows).choose_multiple(&mut rng, sample_size);
        let mut builder = Float32Builder::with_capacity(sample_size * dimension);
        for idx in chosen.iter() {
            let s = array.slice(idx * dimension, dimension);
            builder.append_slice(as_primitive_array::<Float32Type>(s.as_ref()).values());
        }
        builder.finish()
    } else {
        array.clone()
    };

    train_kmeans_fallback(&data, dimension, k, max_iterations).await
}
