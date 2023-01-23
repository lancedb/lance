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

use arrow_array::Float32Array;

use crate::Result;

#[cfg(feature = "faiss")]
fn train_kmeans_faiss(
    array: &Float32Array,
    dimension: usize,
    num_clusters: u32,
    max_iters: u32,
) -> Result<Vec<f32>> {
    use faiss::cluster::kmeans_clustering;

    let model = kmeans_clustering(dimension as u32, num_clusters, array.values()).unwrap();
    Ok(model.centroids)
}

/// Train kmean models and returns the centroids.
pub fn train_kmeans(
    array: &Float32Array,
    dimension: usize,
    num_clusters: u32,
    max_iterations: u32,
) -> Result<Vec<f32>> {
    #[cfg(feature = "faiss")]
    return train_kmeans(array, dimension, num_clusters, max_iterations);

    #[cfg(not(feature = "faiss"))]
    Ok(vec![])
}
