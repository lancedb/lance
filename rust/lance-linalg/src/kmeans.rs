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

use std::cmp::min;
use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{
    builder::Float32Builder, cast::as_primitive_array, new_empty_array, Array, Float32Array,
};
use arrow_schema::{ArrowError, DataType};
use arrow_select::concat::concat;
use futures::stream::{self, repeat_with, StreamExt, TryStreamExt};
use log::{info, warn};
use rand::prelude::*;
use rand::{distributions::WeightedIndex, Rng};

use crate::{
    distance::{Cosine, Dot, MetricType, L2},
    matrix::MatrixView,
};
use crate::{Error, Result};

/// KMean initialization method.
#[derive(Debug, PartialEq, Eq)]
pub enum KMeanInit {
    Random,
    KMeanPlusPlus,
}

/// KMean Training Parameters
#[derive(Debug)]
pub struct KMeansParams {
    /// Max number of iterations.
    pub max_iters: u32,

    /// When the difference of mean distance to the centroids is less than this `tolerance`
    /// threshold, stop the training.
    pub tolerance: f32,

    /// Run kmeans multiple times and pick the best (balanced) one.
    pub redos: usize,

    /// Init methods.
    pub init: KMeanInit,

    /// The metric to calculate distance.
    pub metric_type: MetricType,

    /// Centroids to continuous training. If present, it will continuously train
    /// from the given centroids. If None, it will initialize centroids via init method.
    pub centroids: Option<Arc<Float32Array>>,
}

impl Default for KMeansParams {
    fn default() -> Self {
        Self {
            max_iters: 50,
            tolerance: 1e-4,
            redos: 1,
            init: KMeanInit::Random,
            metric_type: MetricType::L2,
            centroids: None,
        }
    }
}

/// KMeans implementation for Apache Arrow Arrays.
#[derive(Debug, Clone)]
pub struct KMeans {
    /// Centroids for each of the k clusters.
    ///
    /// k * dimension.
    pub centroids: Arc<Float32Array>,

    /// Vector dimension.
    pub dimension: usize,

    /// The number of clusters
    pub k: usize,

    pub metric_type: MetricType,
}

/// Initialize using kmean++, and returns the centroids of k clusters.
async fn kmean_plusplus(
    data: Arc<Float32Array>,
    dimension: usize,
    k: usize,
    mut rng: impl Rng,
    metric_type: MetricType,
) -> KMeans {
    assert!(data.len() > k * dimension);
    let mut kmeans = KMeans::empty(k, dimension, metric_type);
    let first_idx = rng.gen_range(0..data.len() / dimension);
    let first_vector: Float32Array = data.slice(first_idx * dimension, dimension);
    kmeans.centroids = Arc::new(first_vector);

    let mut seen = HashSet::new();
    seen.insert(first_idx);

    for _ in 1..k {
        let membership = kmeans.compute_membership(data.clone()).await;
        let weights = WeightedIndex::new(&membership.distances).unwrap();
        let mut chosen;
        loop {
            chosen = weights.sample(&mut rng);
            if !seen.contains(&chosen) {
                seen.insert(chosen);
                break;
            }
        }

        let new_vector: Float32Array = data.slice(chosen * dimension, dimension);

        let new_centroid_values = Float32Array::from_iter_values(
            kmeans
                .centroids
                .as_ref()
                .values()
                .iter()
                .copied()
                .chain(new_vector.values().iter().copied()),
        );
        kmeans.centroids = Arc::new(new_centroid_values);
    }
    kmeans
}

/// Randomly initialize kmeans centroids.
///
///
async fn kmeans_random_init(
    data: &Float32Array,
    dimension: usize,
    k: usize,
    mut rng: impl Rng,
    metric_type: MetricType,
) -> Result<KMeans> {
    assert!(data.len() >= k * dimension);
    let chosen = (0..data.len() / dimension)
        .choose_multiple(&mut rng, k)
        .to_vec();
    let mut builder = Float32Builder::with_capacity(k * dimension);
    for i in chosen {
        builder.append_slice(&data.values()[i * dimension..(i + 1) * dimension]);
    }
    let mut kmeans = KMeans::empty(k, dimension, metric_type);
    kmeans.centroids = Arc::new(builder.finish());
    Ok(kmeans)
}

pub struct KMeanMembership {
    /// Previous centroids.
    ///
    /// `k * dimension` f32 matrix.
    centroids: Arc<Float32Array>,

    /// Reference to the input vectors, with dimension `dimension`.
    data: Arc<Float32Array>,

    dimension: usize,

    /// Cluster Id for each vector.
    cluster_ids: Vec<u32>,

    /// Distance between each vector, to its corresponding centroids.
    distances: Vec<f32>,

    /// Number of centroids.
    k: usize,

    metric_type: MetricType,
}

impl KMeanMembership {
    /// Reconstruct a KMeans model from the membership.
    async fn to_kmeans(&self) -> Result<KMeans> {
        let dimension = self.dimension;
        let cluster_ids = Arc::new(self.cluster_ids.clone());

        // New centroids for each cluster
        let means = stream::iter(0..self.k)
            .zip(repeat_with(|| {
                (
                    self.data.clone(),
                    cluster_ids.clone(),
                    self.centroids.clone(),
                )
            }))
            .map(
                |(cluster, (data, cluster_ids, prev_centroids))| async move {
                    tokio::task::spawn_blocking(move || {
                        let mut sum = vec![0.0; dimension];
                        let data = data.values();
                        let mut total = 0.0;
                        // Eager group-by cluster id.
                        for i in 0..cluster_ids.len() {
                            if cluster_ids[i] as usize == cluster {
                                // TODO: use simd ADD
                                for j in 0..dimension {
                                    sum[j] += data[i * dimension + j];
                                }
                                total += 1.0;
                            };
                        }
                        if total > 0.0 {
                            let s = Float32Array::from(
                                sum
                            );
                            s.unary_mut(|x| x / total).unwrap()
                        } else {
                            warn!("Warning: KMean: cluster {} has no value, does not change centroids.", cluster);
                            prev_centroids.slice(cluster * dimension, dimension)
                        }
                    })
                    .await
                },
            )
            .buffered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await.map_err(|e| {
                ArrowError::ComputeError(format!(
                    "KMeans: failed to compute new centroids: {}",
                    e
                ))
            })?;

        // TODO: concat requires `&[&dyn Array]`. Are there cheaper way to pass Vec<Float32Array> to `concat`?
        let mut mean_refs: Vec<&dyn Array> = vec![];
        for m in means.iter() {
            mean_refs.push(m);
        }
        let centroids = concat(&mean_refs).unwrap();
        Ok(KMeans {
            centroids: Arc::new(as_primitive_array(centroids.as_ref()).clone()),
            dimension,
            k: self.k,
            metric_type: self.metric_type,
        })
    }

    fn distance_sum(&self) -> f32 {
        self.distances.iter().sum()
    }

    /// Returns how many data points are here
    fn len(&self) -> usize {
        self.cluster_ids.len()
    }

    /// Histogram of the size of each cluster.
    fn histogram(&self) -> Vec<usize> {
        let mut hist: Vec<usize> = vec![0; self.k];
        for cluster_id in self.cluster_ids.iter() {
            hist[*cluster_id as usize] += 1;
        }
        hist
    }

    /// Std deviation of the histogram / cluster distribution.
    fn hist_stddev(&self) -> f32 {
        let mean: f32 = self.len() as f32 * 1.0 / self.k as f32;
        (self
            .histogram()
            .iter()
            .map(|c| (*c as f32 - mean).powi(2))
            .sum::<f32>()
            / self.len() as f32)
            .sqrt()
    }
}

impl KMeans {
    fn empty(k: usize, dimension: usize, metric_type: MetricType) -> Self {
        let empty_array = new_empty_array(&DataType::Float32);
        Self {
            centroids: Arc::new(as_primitive_array(empty_array.as_ref()).clone()),
            dimension,
            k,
            metric_type,
        }
    }

    /// Create a [`KMeans`] with existing centroids.
    /// It is useful for continuing training.
    fn with_centroids(
        centroids: Arc<Float32Array>,
        k: usize,
        dimension: usize,
        metric_type: MetricType,
    ) -> Self {
        Self {
            centroids,
            dimension,
            k,
            metric_type,
        }
    }

    /// Initialize a [`KMeans`] with random centroids.
    ///
    /// Parameters
    /// - *data*: training data. provided to do samplings.
    /// - *k*: the number of clusters.
    /// - *metric_type*: the metric type to calculate distance.
    /// - *rng*: random generator.
    pub async fn init_random(
        data: &MatrixView,
        k: usize,
        metric_type: MetricType,
        rng: impl Rng,
    ) -> Result<Self> {
        kmeans_random_init(&data.data(), data.num_columns(), k, rng, metric_type).await
    }

    /// Train a KMeans model on data with `k` clusters.
    pub async fn new(
        data: &Float32Array,
        dimension: usize,
        k: usize,
        max_iters: u32,
    ) -> Result<Self> {
        let params = KMeansParams {
            max_iters,
            metric_type: MetricType::L2,
            ..Default::default()
        };
        Self::new_with_params(data, dimension, k, &params).await
    }

    /// Train a [`KMeans`] model with full parameters.
    pub async fn new_with_params(
        data: &Float32Array,
        dimension: usize,
        k: usize,
        params: &KMeansParams,
    ) -> Result<Self> {
        let n = data.len() / dimension;
        if n < k {
            return Err(ArrowError::ComputeError(
                format!(
                    "KMeans: training does not have sufficient data points: n({}) is smaller than k({})",
                    n, k
                )
            ));
        }

        // TODO: refactor kmeans to work with reference instead of Arc?
        let data = Arc::new(data.clone());
        let mut best_kmeans = Self::empty(k, dimension, params.metric_type);
        let mut best_stddev = f32::MAX;

        let rng = rand::rngs::SmallRng::from_entropy();
        let mat = MatrixView::new(data.clone(), dimension);
        for redo in 1..=params.redos {
            let mut kmeans = if let Some(centroids) = params.centroids.as_ref() {
                // Use existing centroids.
                Self::with_centroids(centroids.clone(), k, dimension, params.metric_type)
            } else {
                match params.init {
                    KMeanInit::Random => {
                        Self::init_random(&mat, k, params.metric_type, rng.clone()).await?
                    }
                    KMeanInit::KMeanPlusPlus => {
                        kmean_plusplus(data.clone(), dimension, k, rng.clone(), params.metric_type)
                            .await
                    }
                }
            };

            let mut dist_sum: f32 = f32::MAX;
            let mut stddev: f32 = f32::MAX;
            for i in 1..=params.max_iters {
                if i % 10 == 0 {
                    info!(
                        "KMeans training: iteration {} / {}, redo={}",
                        i, params.max_iters, redo
                    );
                };
                let last_membership = kmeans.train_once(&mat).await;
                let last_dist_sum = last_membership.distance_sum();
                stddev = last_membership.hist_stddev();
                kmeans = last_membership.to_kmeans().await.unwrap();
                if (dist_sum - last_dist_sum).abs() / last_dist_sum < params.tolerance {
                    info!(
                        "KMeans training: converged at iteration {} / {}, redo={}",
                        i, params.max_iters, redo
                    );
                    break;
                }
                dist_sum = last_dist_sum;
            }
            // Optimize for balanced clusters instead of minimal distance.
            if stddev < best_stddev {
                best_kmeans = kmeans;
                best_stddev = stddev;
            }
        }

        Ok(best_kmeans)
    }

    /// Train for one iteration.
    ///
    /// Parameters
    ///
    /// - *data*: training data / samples.
    ///
    /// Returns a new KMeans
    ///
    /// ```rust,ignore
    /// for i in 0..max_iters {
    ///   let membership = kmeans.train_once(&mat).await;
    ///   let kmeans = membership.to_kmeans();
    /// }
    /// ```
    pub async fn train_once(&self, data: &MatrixView) -> KMeanMembership {
        self.compute_membership(data.data().clone()).await
    }

    /// Recompute the membership of each vector.
    ///
    /// Parameters:
    ///
    /// - *data*: a `N * dimension` float32 array.
    /// - *dist_fn*: the function to compute distances.
    async fn compute_membership(&self, data: Arc<Float32Array>) -> KMeanMembership {
        let dimension = self.dimension;
        let n = data.len() / self.dimension;
        let metric_type = self.metric_type;
        const CHUNK_SIZE: usize = 1024;
        let cluster_with_distances = stream::iter((0..n).step_by(CHUNK_SIZE))
            // make tiles of input data to split between threads.
            .zip(repeat_with(|| (data.clone(), self.centroids.clone())))
            .map(|(start_idx, (data, centroids))| async move {
                let data = tokio::task::spawn_blocking(move || {
                    let array = data.values();
                    let centroids_array = centroids.values();

                    (start_idx..min(start_idx + CHUNK_SIZE, n))
                        .map(|idx| {
                            let vector = &array[idx * dimension..(idx + 1) * dimension];
                            let mut min = std::f32::MAX;
                            let mut min_idx = 0;
                            for (idx, other) in centroids_array.chunks_exact(dimension).enumerate()
                            {
                                // We've found about 40% performance improvement by using static dispatch instead
                                // of dynamic dispatch.
                                //
                                // NOTE: Please make sure run benchmark when changing the following code.
                                // `RUSTFLAGS="-C target-cpu=native" cargo bench --bench ivf_pq`
                                let dist = match metric_type {
                                    MetricType::L2 => vector.l2(other),
                                    MetricType::Cosine => vector.cosine(other),
                                    MetricType::Dot => vector.dot(other),
                                };
                                if dist < min {
                                    min = dist;
                                    min_idx = idx;
                                }
                            }
                            (min_idx as u32, min)
                        })
                        .collect::<Vec<_>>()
                })
                .await
                .map_err(|e| {
                    ArrowError::ComputeError(format!("KMeans: failed to compute membership: {}", e))
                })?;
                Ok::<Vec<_>, Error>(data)
            })
            .buffered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        KMeanMembership {
            centroids: self.centroids.clone(),
            data,
            dimension,
            cluster_ids: cluster_with_distances
                .iter()
                .flatten()
                .map(|(c, _)| *c)
                .collect(),
            distances: cluster_with_distances
                .iter()
                .flatten()
                .map(|(_, d)| *d)
                .collect(),
            k: self.k,
            metric_type: self.metric_type,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::Float32Array;

    #[tokio::test]
    async fn test_train_with_small_dataset() {
        let data = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        match KMeans::new(&data, 2, 128, 5).await {
            Ok(_) => panic!("Should fail to train KMeans"),
            Err(e) => {
                assert!(e.to_string().contains("smaller than"));
            }
        }
    }
}
