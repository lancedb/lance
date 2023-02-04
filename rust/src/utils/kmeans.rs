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

use std::collections::HashSet;
use std::sync::Arc;

use arrow_arith::arithmetic::{add, divide_scalar};
use arrow_array::{
    cast::as_primitive_array, new_empty_array, types::Float32Type, Array, FixedSizeListArray,
    Float32Array,
};
use arrow_schema::DataType;
use arrow_select::concat::concat;
use futures::stream::{self, repeat_with, StreamExt, TryStreamExt};
use rand::prelude::*;
use rand::{distributions::WeightedIndex, Rng, RngCore};

use super::distance::l2_distance;
use crate::arrow::*;
use crate::Result;

#[derive(Debug)]
pub struct KMeansParams {
    /// Max number of iterations.
    max_iters: u32,

    /// When the difference of mean distance to the centroids is less than this `tolerance`
    /// threshold, stop the training.
    tolerance: f32,
}

impl Default for KMeansParams {
    fn default() -> Self {
        Self {
            max_iters: 50,
            tolerance: 1e-4,
        }
    }
}

/// KMeans implementation for Apache Arrow Arrays.
#[derive(Debug)]
pub struct KMeans {
    /// Centroids for each of the k clusters.
    ///
    /// k * dimension.
    pub centroids: Arc<FixedSizeListArray>,

    /// The number of clusters
    pub k: u32,
}

/// Initialize using kmean++, and returns the centroids of k clusters.
async fn kmean_plusplus(
    data: Arc<FixedSizeListArray>,
    k: u32, /* dist_fn, rand_seed */
    rng: &mut dyn RngCore,
) -> KMeans {
    assert!(data.len() > k as usize);
    let dimension = data.value_length();

    let empty_array = new_empty_array(&DataType::Float32);
    let mut kmeans = KMeans {
        centroids: Arc::new(
            FixedSizeListArray::try_new(as_primitive_array::<Float32Type>(&empty_array), dimension)
                .unwrap(),
        ),
        k,
    };

    let first = rng.gen_range(0..data.len());
    println!("First node: {}", first);

    let vector = data.value(first);
    kmeans.centroids = FixedSizeListArray::try_new(vector, dimension)
        .unwrap()
        .into();

    let mut seen = HashSet::new();
    seen.insert(first);
    for _ in 1..k {
        let membership = kmeans.compute_membership(data.clone()).await;
        let weights = WeightedIndex::new(&membership.distances).unwrap();
        let mut chosen;
        loop {
            chosen = weights.sample(rng);
            if !seen.contains(&chosen) {
                seen.insert(chosen);
                break;
            }
        }
        let vector = data.value(chosen);
        let new_vector: &Float32Array = as_primitive_array(&vector);

        let centroids_array = kmeans.centroids.values();
        let centroids_f32_array: &Float32Array = as_primitive_array(&centroids_array);
        let new_centroid_values = Float32Array::from_iter_values(
            centroids_f32_array
                .values()
                .iter()
                .copied()
                .chain(new_vector.values().iter().copied()),
        );
        kmeans.centroids =
            Arc::new(FixedSizeListArray::try_new(new_centroid_values, dimension).unwrap());
    }
    kmeans
}

struct KMeanMembership {
    /// Reference to the input vectors.
    data: Arc<FixedSizeListArray>,

    /// Cluster Id for each vector.
    cluster_ids: Vec<u32>,

    /// Distance between each vector, to its corresponding centroids.
    distances: Vec<f32>,

    k: u32,
}

impl KMeanMembership {
    async fn to_kmean(&self) -> Result<KMeans> {
        let dimension = self.data.value_length();
        let cluster_ids = Arc::new(self.cluster_ids.clone());
        let data = self.data.clone();
        // New centroids for each cluster
        let means = stream::iter(0..self.k)
            .zip(repeat_with(|| (data.clone(), cluster_ids.clone())))
            .map(|(cluster, (data, cluster_ids))| async move {
                tokio::task::spawn_blocking(move || {
                    let mut sum = Float32Array::from_iter_values(
                        (0..dimension).map(|_| 0.0).collect::<Vec<_>>(),
                    );
                    let mut total = 0.0;
                    for i in 0..cluster_ids.len() {
                        if cluster_ids[i] == cluster {
                            sum = add(&sum, as_primitive_array(data.value(i).as_ref())).unwrap();
                            total += 1.0;
                        };
                    }
                    divide_scalar(&sum, total).unwrap()
                })
                .await
            })
            .buffered(16)
            .try_collect::<Vec<_>>()
            .await?;

        // TODO: concat requires `&[&dyn Array]`. Are there cheaper way to pass Vec<Float32Array> to `concat`?
        let mut mean_refs: Vec<&dyn Array> = vec![];
        for m in means.iter() {
            mean_refs.push(m);
        }
        let centroids = concat(&mean_refs).unwrap();
        Ok(KMeans {
            centroids: Arc::new(FixedSizeListArray::try_new(centroids, dimension)?),
            k: self.k,
        })
    }

    fn distance_sum(&self) -> f32 {
        self.distances.iter().sum()
    }

    /// Returns how many are here
    fn len(&self) -> usize {
        self.cluster_ids.len()
    }

    /// Histogram of the size of each cluster.
    fn histogram(&self) -> Vec<usize> {
        let mut hist: Vec<usize> = vec![0; self.k as usize];
        for cluster_id in self.cluster_ids.iter() {
            hist[*cluster_id as usize] += 1;
        }
        hist
    }

    fn stddev(&self) -> f32 {
        let mean: f32 = self.len() as f32 * 1.0 / self.k as f32;
        self.histogram()
            .iter()
            .map(|c| (*c as f32 - mean).powi(2))
            .sum::<f32>()
            / self.len() as f32
    }
}

impl KMeans {
    /// Train a KMean on data with `k` clusters.
    pub async fn new(data: Arc<FixedSizeListArray>, k: u32, max_iters: u32) -> Self {
        let mut params = KMeansParams::default();
        params.max_iters = max_iters;
        KMeans::new_with_params(data, k, &params).await
    }

    pub async fn new_with_params(
        data: Arc<FixedSizeListArray>,
        k: u32,
        params: &KMeansParams,
    ) -> Self {
        let mut kmeans = kmean_plusplus(data.clone(), k, &mut rand::thread_rng()).await;

        let mut last_membership = kmeans.compute_membership(data.clone()).await;
        for _ in 1..=params.max_iters {
            let new_kmeans: KMeans = last_membership.to_kmean().await.unwrap();
            let new_membership = new_kmeans.compute_membership(data.clone()).await;
            if (new_membership.distance_sum() - last_membership.distance_sum()).abs()
                / last_membership.distance_sum()
                < params.tolerance
            {
                kmeans = new_kmeans;
                last_membership = new_membership;
                break;
            }
            kmeans = new_kmeans;
            last_membership = new_membership;
        }
        println!(
            "kmean historam: {:?} stddev={}",
            last_membership.histogram(),
            last_membership.stddev(),
        );
        kmeans
    }

    async fn compute_membership(&self, data: Arc<FixedSizeListArray>) -> KMeanMembership {
        let cluster_with_distances = (0..data.len())
            .map(|idx| {
                let value_arr = data.value(idx);
                let vector: &Float32Array = as_primitive_array(&value_arr);
                let distances = l2_distance(vector, self.centroids.as_ref()).unwrap();
                let cluster_id = argmin(distances.as_ref()).unwrap();
                let distance = distances.value(cluster_id as usize);
                (cluster_id, distance)
            })
            .collect::<Vec<_>>();
        let k = self.k;
        KMeanMembership {
            data: data.clone(),
            cluster_ids: cluster_with_distances.iter().map(|(c, _)| *c).collect(),
            distances: cluster_with_distances.iter().map(|(_, d)| *d).collect(),
            k,
        }
    }
}

#[cfg(test)]
mod tests {}
