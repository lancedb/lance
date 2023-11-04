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
use std::vec;

use arrow_array::{
    cast::AsArray, new_empty_array, types::Float32Type, Array, FixedSizeListArray, Float32Array,
};
use arrow_schema::{ArrowError, DataType};
use futures::stream::{self, repeat_with, StreamExt, TryStreamExt};
use log::{info, warn};
use rand::prelude::*;
use rand::{distributions::WeightedIndex, Rng};
use tracing::instrument;

use crate::kernels::argmin_value_float;
use crate::{
    distance::{dot_distance, l2_distance_batch, Cosine, MetricType, Normalize, L2},
    kernels::{argmin, argmin_value},
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
        let membership = kmeans.compute_membership(data.clone(), None).await;
        let weights =
            WeightedIndex::new(membership.cluster_id_and_distances.iter().map(|(_, d)| d)).unwrap();
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
    let mut builder: Vec<f32> = Vec::with_capacity(k * dimension);
    for i in chosen {
        builder.extend(data.values()[i * dimension..(i + 1) * dimension].iter());
    }
    let mut kmeans = KMeans::empty(k, dimension, metric_type);
    kmeans.centroids = Arc::new(builder.into());
    Ok(kmeans)
}

pub struct KMeanMembership {
    /// Reference to the input vectors, with dimension `dimension`.
    data: Arc<Float32Array>,

    dimension: usize,

    /// Cluster Id and distances for each vector.
    pub cluster_id_and_distances: Vec<(u32, f32)>,

    /// Number of centroids.
    k: usize,

    metric_type: MetricType,
}

impl KMeanMembership {
    /// Reconstruct a KMeans model from the membership.
    async fn to_kmeans(&self) -> Result<KMeans> {
        let dimension = self.dimension;

        let mut cluster_cnts = vec![0_usize; self.k];
        let mut new_centroids = vec![0.0_f32; self.k * dimension];
        self.data
            .values()
            .chunks_exact(dimension)
            .zip(self.cluster_id_and_distances.iter().map(|(c, _)| c))
            .for_each(|(vector, cluster_id)| {
                cluster_cnts[*cluster_id as usize] += 1;
                // TODO: simd
                for (old, new) in new_centroids
                    [*cluster_id as usize * dimension..(1 + *cluster_id as usize) * dimension]
                    .iter_mut()
                    .zip(vector)
                {
                    *old += new;
                }
            });
        cluster_cnts.iter().enumerate().for_each(|(i, &cnt)| {
            if cnt == 0 {
                warn!("KMeans: cluster {} is empty", i);
            } else {
                // TODO: simd
                new_centroids[i * dimension..(i + 1) * dimension]
                    .iter_mut()
                    .for_each(|v| *v /= cnt as f32);
            }
        });

        Ok(KMeans {
            centroids: Arc::new(new_centroids.into()),
            dimension,
            k: self.k,
            metric_type: self.metric_type,
        })
    }

    fn distance_sum(&self) -> f32 {
        self.cluster_id_and_distances.iter().map(|(_, d)| d).sum()
    }

    /// Returns how many data points are here
    fn len(&self) -> usize {
        self.cluster_id_and_distances.len()
    }

    /// Histogram of the size of each cluster.
    fn histogram(&self) -> Vec<usize> {
        let mut hist: Vec<usize> = vec![0; self.k];
        for (cluster_id, _) in self.cluster_id_and_distances.iter() {
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
            centroids: Arc::new(empty_array.as_primitive().clone()),
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
        data: &MatrixView<Float32Type>,
        k: usize,
        metric_type: MetricType,
        rng: impl Rng,
    ) -> Result<Self> {
        kmeans_random_init(&data.data(), data.num_columns(), k, rng, metric_type).await
    }

    /// Train a KMeans model on data with `k` clusters.
    pub async fn new(data: &FixedSizeListArray, k: usize, max_iters: u32) -> Result<Self> {
        let params = KMeansParams {
            max_iters,
            metric_type: MetricType::L2,
            ..Default::default()
        };
        Self::new_with_params(data, k, &params).await
    }

    /// Train a [`KMeans`] model with full parameters.
    pub async fn new_with_params(
        data: &FixedSizeListArray,
        k: usize,
        params: &KMeansParams,
    ) -> Result<Self> {
        let dimension = data.value_length() as usize;
        let n = data.len();
        if n < k {
            return Err(ArrowError::InvalidArgumentError(
                format!(
                    "KMeans: training does not have sufficient data points: n({}) is smaller than k({})",
                    n, k
                )
            ));
        }

        if !matches!(data.value_type(), DataType::Float32) {
            return Err(ArrowError::InvalidArgumentError(format!(
                "KMeans: data must be Float32, got: {}",
                data.value_type()
            )));
        }
        let values: &Float32Array = data.values().as_primitive();

        // TODO: refactor kmeans to work with reference instead of Arc?
        let data = Arc::new(values.clone());
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
    #[instrument(level = "debug", skip_all)]
    pub async fn train_once(&self, data: &MatrixView<Float32Type>) -> KMeanMembership {
        match self.metric_type {
            MetricType::Cosine => self.train_cosine_once(data).await,
            _ => self.compute_membership(data.data().clone(), None).await,
        }
    }

    async fn train_cosine_once(&self, data: &MatrixView<Float32Type>) -> KMeanMembership {
        let norm_data = Some(Arc::new(
            data.iter()
                .map(|v| v.norm_l2())
                .collect::<Vec<f32>>()
                .into(),
        ));
        self.compute_membership(data.data().clone(), norm_data)
            .await
    }

    /// Recompute the membership of each vector.
    ///
    /// Parameters:
    ///
    /// - *data*: a `N * dimension` float32 array.
    /// - *dist_fn*: the function to compute distances.
    pub async fn compute_membership(
        &self,
        data: Arc<Float32Array>,
        norm_data: Option<Arc<Float32Array>>,
    ) -> KMeanMembership {
        let dimension = self.dimension;
        let n = data.len() / self.dimension;
        let metric_type = self.metric_type;
        const CHUNK_SIZE: usize = 1024;

        // Normalized centroids for fast cosine. cosine(A, B) = A * B / (|A| * |B|).
        // So here, norm_centroids = |B| for each centroid B.
        let norm_centroids = if matches!(metric_type, MetricType::Cosine) {
            Arc::new(Some(
                self.centroids
                    .values()
                    .chunks_exact(dimension)
                    .map(|centroid| centroid.norm_l2())
                    .collect::<Vec<f32>>(),
            ))
        } else {
            Arc::new(None)
        };

        let cluster_with_distances = stream::iter((0..n).step_by(CHUNK_SIZE))
            // make tiles of input data to split between threads.
            .zip(repeat_with(|| {
                (
                    data.clone(),
                    self.centroids.clone(),
                    norm_centroids.clone(),
                    norm_data.clone(),
                )
            }))
            .map(
                |(start_idx, (data, centroids, norms, norm_data))| async move {
                    let data = tokio::task::spawn_blocking(move || {
                        let last_idx = min(start_idx + CHUNK_SIZE, n);

                        let centroids_array = centroids.values();
                        let values = &data.values()[start_idx * dimension..last_idx * dimension];

                        if metric_type == MetricType::L2 {
                            return compute_partitions_l2_f32(centroids_array, values, dimension)
                                .collect();
                        }

                        values
                            .chunks_exact(dimension)
                            .enumerate()
                            .map(|(idx, vector)| {
                                let centroid_stream = centroids_array.chunks_exact(dimension);
                                match metric_type {
                                    MetricType::L2 => {
                                        argmin_value(centroid_stream.map(|cent| vector.l2(cent)))
                                    }
                                    MetricType::Cosine => {
                                        let centroid_norms = norms.as_ref().as_ref().unwrap();
                                        if let Some(norm_vectors) = norm_data.as_ref() {
                                            let norm_vec = norm_vectors.values()[idx];
                                            argmin_value(
                                                centroid_stream.zip(centroid_norms.iter()).map(
                                                    |(cent, &cent_norm)| {
                                                        cent.cosine_with_norms(
                                                            cent_norm, norm_vec, vector,
                                                        )
                                                    },
                                                ),
                                            )
                                        } else {
                                            argmin_value(
                                                centroid_stream.zip(centroid_norms.iter()).map(
                                                    |(cent, &cent_norm)| {
                                                        cent.cosine_fast(cent_norm, vector)
                                                    },
                                                ),
                                            )
                                        }
                                    }
                                    crate::distance::DistanceType::Dot => argmin_value(
                                        centroid_stream.map(|cent| dot_distance(vector, cent)),
                                    ),
                                }
                                .unwrap()
                            })
                            .collect::<Vec<_>>()
                    })
                    .await
                    .map_err(|e| {
                        ArrowError::ComputeError(format!(
                            "KMeans: failed to compute membership: {}",
                            e
                        ))
                    })?;
                    Ok::<Vec<_>, Error>(data)
                },
            )
            .buffered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        KMeanMembership {
            data,
            dimension,
            cluster_id_and_distances: cluster_with_distances.iter().flatten().copied().collect(),
            k: self.k,
            metric_type: self.metric_type,
        }
    }
}

/// Return a slice of `data[x,y..y+strip]`.
#[inline]
fn get_slice(data: &[f32], x: usize, y: usize, dim: usize, strip: usize) -> &[f32] {
    &data[x * dim + y..x * dim + y + strip]
}

fn compute_partitions_l2_f32_small<'a>(
    centroids: &'a [f32],
    data: &'a [f32],
    dim: usize,
) -> Box<dyn Iterator<Item = (u32, f32)> + 'a> {
    Box::new(
        data.chunks(dim)
            .map(move |row| argmin_value_float(l2_distance_batch(row, centroids, dim))),
    )
}

/// Fast partition computation for L2 distance.
fn compute_partitions_l2_f32<'a>(
    centroids: &'a [f32],
    data: &'a [f32],
    dim: usize,
) -> Box<dyn Iterator<Item = (u32, f32)> + 'a> {
    if std::mem::size_of_val(centroids) <= 16 * 1024 {
        return compute_partitions_l2_f32_small(centroids, data, dim);
    }

    const STRIPE_SIZE: usize = 128;
    const TILE_SIZE: usize = 16;

    // 128 * 4bytes * 16 = 8KB for centroid and data respectively, so both of them can
    // stay in L1 cache.
    let num_centroids = centroids.len() / dim;

    // Read a tile of data, `data[idx..idx+TILE_SIZE]`
    let stream = data.chunks(TILE_SIZE * dim).flat_map(move |data_tile| {
        // Loop over each strip.
        // s is the index of value in each vector.
        let num_rows_in_tile = data_tile.len() / dim;
        let mut min_dists = vec![f32::MAX; num_rows_in_tile];
        let mut partitions = vec![0_u32; num_rows_in_tile];

        for centroid_start in (0..num_centroids).step_by(TILE_SIZE) {
            // 4B * 16 * 16 = 1 KB
            let mut dists = vec![0_f32; TILE_SIZE * TILE_SIZE];
            let num_centroids_in_tile = min(TILE_SIZE, num_centroids - centroid_start);
            for s in (0..dim).step_by(STRIPE_SIZE) {
                // Calculate L2 within each TILE * STRIP
                let slice_len = min(STRIPE_SIZE, dim - s);
                for di in 0..num_rows_in_tile {
                    let data_slice = get_slice(data_tile, di, s, dim, slice_len);
                    for ci in centroid_start..centroid_start + num_centroids_in_tile {
                        // Get a slice of `data[di][s..s+STRIP_SIZE]`.
                        let cent_slice = get_slice(centroids, ci, s, dim, slice_len);
                        let dist = data_slice.l2(cent_slice);
                        dists[di * TILE_SIZE + (ci - centroid_start)] += dist;
                    }
                }
            }

            for i in 0..num_rows_in_tile {
                let (part_id, dist) = argmin_value(
                    dists[i * TILE_SIZE..(i * TILE_SIZE + num_centroids_in_tile)]
                        .iter()
                        .copied(),
                )
                .unwrap();
                if dist < min_dists[i] {
                    min_dists[i] = dist;
                    partitions[i] = centroid_start as u32 + part_id;
                }
            }
        }
        partitions.into_iter().zip(min_dists)
    });
    Box::new(stream)
}

fn compute_partitions_cosine(centroids: &[f32], data: &[f32], dimension: usize) -> Vec<u32> {
    let centroid_norms = centroids
        .chunks(dimension)
        .map(|centroid| centroid.norm_l2())
        .collect::<Vec<_>>();
    data.chunks(dimension)
        .map(|row| {
            argmin(
                centroids
                    .chunks(dimension)
                    .zip(centroid_norms.iter())
                    .map(|(centroid, &norm)| centroid.cosine_fast(norm, row)),
            )
            .unwrap()
        })
        .collect()
}

fn compute_partitions_dot(centroids: &[f32], data: &[f32], dimension: usize) -> Vec<u32> {
    data.chunks(dimension)
        .map(|row| {
            argmin(
                centroids
                    .chunks(dimension)
                    .map(|centroid| dot_distance(row, centroid)),
            )
            .unwrap()
        })
        .collect()
}

#[inline]
pub fn compute_partitions(
    centroids: &[f32],
    data: &[f32],
    dimension: usize,
    metric_type: MetricType,
) -> Vec<u32> {
    match metric_type {
        MetricType::L2 => compute_partitions_l2_f32(centroids, data, dimension)
            .map(|(c, _)| c)
            .collect(),
        MetricType::Cosine => compute_partitions_cosine(centroids, data, dimension),
        MetricType::Dot => compute_partitions_dot(centroids, data, dimension),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::Float32Array;
    use lance_arrow::*;
    use lance_testing::datagen::generate_random_array;

    #[tokio::test]
    async fn test_train_with_small_dataset() {
        let data = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        let data = FixedSizeListArray::try_new_from_values(data, 2).unwrap();
        match KMeans::new(&data, 128, 5).await {
            Ok(_) => panic!("Should fail to train KMeans"),
            Err(e) => {
                assert!(e.to_string().contains("smaller than"));
            }
        }
    }

    #[test]
    fn test_compute_partitions() {
        const DIM: usize = 256;
        let centroids = generate_random_array(DIM * 18);
        let data = generate_random_array(DIM * 20);

        let expected = data
            .values()
            .chunks(DIM)
            .map(|row| {
                argmin(
                    centroids
                        .values()
                        .chunks(DIM)
                        .map(|centroid| row.l2(centroid)),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let actual = compute_partitions(centroids.values(), data.values(), DIM, MetricType::L2);
        assert_eq!(expected, actual);
    }
}
