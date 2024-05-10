// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! KMeans implementation for Apache Arrow Arrays.
//!
//! Support ``l2``, ``cosine`` and ``dot`` distances, see [MetricType].
//!
//! ``Cosine`` distance are calculated by normalizing the vectors to unit length,
//! and run ``l2`` distance on the unit vectors.
//!

use std::cmp::min;
use std::ops::DivAssign;
use std::sync::Arc;
use std::vec;

use arrow_array::{Array, FixedSizeListArray, Float32Array, UInt32Array};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::ArrowError;
use futures::stream::{self, repeat_with, StreamExt, TryStreamExt};
use lance_arrow::{ArrowFloatType, FloatArray};
use log::{info, warn};
use num_traits::{AsPrimitive, Float, FromPrimitive, Num, Zero};
use rand::prelude::*;
use tracing::instrument;

use crate::distance::norm_l2::Normalize;
use crate::distance::{dot_distance_batch, DistanceType};
use crate::kernels::{argmax, argmin_value_float};
use crate::{
    distance::{
        dot_distance,
        l2::{l2, l2_distance_batch, L2},
        Dot, MetricType,
    },
    kernels::argmin_value,
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
    pub tolerance: f64,

    /// Run kmeans multiple times and pick the best (balanced) one.
    pub redos: usize,

    /// Init methods.
    pub init: KMeanInit,

    /// The metric to calculate distance.
    pub distance_type: DistanceType,
}

impl Default for KMeansParams {
    fn default() -> Self {
        Self {
            max_iters: 50,
            tolerance: 1e-4,
            redos: 1,
            init: KMeanInit::Random,
            distance_type: DistanceType::L2,
        }
    }
}

impl KMeansParams {
    /// Create a new KMeansParams with cosine distance.
    #[allow(dead_code)]
    fn cosine() -> Self {
        Self {
            distance_type: DistanceType::Cosine,
            ..Default::default()
        }
    }
}

/// KMeans implementation for Apache Arrow Arrays.
#[derive(Debug, Clone)]
pub struct KMeans<T: ArrowFloatType>
where
    T::Native: L2 + Dot,
{
    /// Centroids for each of the k clusters.
    ///
    /// k * dimension.
    pub centroids: Arc<T::ArrayType>,

    /// Vector dimension.
    pub dimension: usize,

    /// The number of clusters
    pub k: usize,

    pub distance_type: DistanceType,
}

/// Randomly initialize kmeans centroids.
///
///
fn kmeans_random_init<T: ArrowFloatType>(
    data: &T::ArrayType,
    dimension: usize,
    k: usize,
    mut rng: impl Rng,
    metric_type: MetricType,
) -> Result<KMeans<T>>
where
    T::Native: AsPrimitive<f32> + L2 + Dot + Normalize,
{
    assert!(data.len() >= k * dimension);
    let chosen = (0..data.len() / dimension)
        .choose_multiple(&mut rng, k)
        .to_vec();
    let mut builder: Vec<T::Native> = Vec::with_capacity(k * dimension);
    for i in chosen {
        builder.extend(data.as_slice()[i * dimension..(i + 1) * dimension].iter());
    }
    let mut kmeans = KMeans::empty(k, dimension, metric_type);
    kmeans.centroids = Arc::new(builder.into());
    Ok(kmeans)
}

pub struct KMeanMembership {
    dimension: usize,

    /// Cluster Id and distance for each vector.
    ///
    /// If it is None, means the assignment is not valid, i.e., input vectors might
    /// be all `NaN`.
    pub cluster_id_and_distances: Vec<Option<(u32, f32)>>,

    /// Number of centroids.
    k: usize,

    distance_type: DistanceType,
}

/// Split one big cluster into two smaller clusters. After split, each
/// cluster has approximately half of the vectors.
fn split_clusters<T: Float + DivAssign>(cnts: &mut [u64], centroids: &mut [T], dimension: usize) {
    for i in 0..cnts.len() {
        if cnts[i] == 0 {
            let largest_idx = argmax(cnts.iter().copied()).unwrap() as usize;
            cnts[i] = cnts[largest_idx] / 2;
            cnts[largest_idx] /= 2;
            for j in 0..dimension {
                centroids[i * dimension + j] =
                    centroids[largest_idx * dimension + j] * (T::one() + T::epsilon());
                centroids[largest_idx * dimension + j] /= T::one() + T::epsilon();
            }
        }
    }
}

impl KMeanMembership {
    /// Reconstruct a KMeans model from the membership.
    fn to_kmeans<T: ArrowFloatType>(&self, data: &[T::Native]) -> Result<KMeans<T>>
    where
        T::Native: L2 + Dot + Normalize,
    {
        let dimension = self.dimension;

        let mut cluster_cnts = vec![0_u64; self.k];
        let mut new_centroids = vec![T::Native::zero(); self.k * dimension];
        data.chunks_exact(dimension)
            .zip(self.cluster_id_and_distances.iter())
            .filter_map(|(vec, cd)| {
                if cd.is_some() {
                    Some((vec, cd.unwrap()))
                } else {
                    None
                }
            })
            .for_each(|(vector, (cluster_id, _))| {
                cluster_cnts[cluster_id as usize] += 1;
                // TODO: simd
                for (old, &new) in new_centroids
                    [cluster_id as usize * dimension..(1 + cluster_id as usize) * dimension]
                    .iter_mut()
                    .zip(vector)
                {
                    *old += new;
                }
            });

        let mut empty_clusters = 0;

        cluster_cnts.iter().enumerate().for_each(|(i, &cnt)| {
            if cnt == 0 {
                empty_clusters += 1;
                new_centroids[i * dimension..(i + 1) * dimension]
                    .iter_mut()
                    .for_each(|v| *v = T::Native::nan());
                // TODO: find the largest one to split.
            } else {
                // TODO: simd
                new_centroids[i * dimension..(i + 1) * dimension]
                    .iter_mut()
                    .for_each(|v| *v /= T::Native::from_u64(cnt).unwrap());
            }
        });

        if empty_clusters as f32 / self.k as f32 > 0.1 {
            warn!(
                "KMeans: more than 10% of clusters are empty: {} of {}.\nHelp: this could mean your dataset \
                is too small to have a meaningful index (less than 5000 vectors) or has many duplicate vectors.",
                empty_clusters, self.k
            );
        }

        split_clusters(&mut cluster_cnts, &mut new_centroids, dimension);

        Ok(KMeans {
            centroids: Arc::new(new_centroids.into()),
            dimension,
            k: self.k,
            distance_type: self.distance_type,
        })
    }

    fn distance_sum(&self) -> f64 {
        self.cluster_id_and_distances
            .iter()
            .flat_map(|cd| {
                if let Some((_, d)) = cd {
                    Some(*d as f64)
                } else {
                    None
                }
            })
            .sum::<f64>()
    }

    /// Returns how many data points are here
    fn len(&self) -> usize {
        self.cluster_id_and_distances.len()
    }

    /// Histogram of the size of each cluster.
    fn histogram(&self) -> Vec<usize> {
        let mut hist: Vec<usize> = vec![0; self.k];
        self.cluster_id_and_distances.iter().for_each(|cd| {
            if let Some((cluster_id, _)) = cd {
                hist[*cluster_id as usize] += 1;
            }
        });

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

impl<T: ArrowFloatType> KMeans<T>
where
    T::Native: AsPrimitive<f32> + L2 + Dot + Normalize,
{
    fn empty(k: usize, dimension: usize, distance_type: DistanceType) -> Self {
        Self {
            centroids: T::empty_array().into(),
            dimension,
            k,
            distance_type,
        }
    }

    /// Create a [`KMeans`] with existing centroids.
    /// It is useful for continuing training.
    pub fn with_centroids(
        centroids: Arc<T::ArrayType>,
        dimension: usize,
        distance_type: DistanceType,
    ) -> Self {
        let k = centroids.len() / dimension;
        Self {
            centroids,
            dimension,
            k,
            distance_type,
        }
    }

    /// Initialize a [`KMeans`] with random centroids.
    ///
    /// Parameters
    /// - *data*: training data. provided to do samplings.
    /// - *k*: the number of clusters.
    /// - *metric_type*: the metric type to calculate distance.
    /// - *rng*: random generator.
    pub fn init_random(
        data: &MatrixView<T>,
        k: usize,
        metric_type: MetricType,
        rng: impl Rng,
    ) -> Result<Self> {
        kmeans_random_init(
            data.data().as_ref(),
            data.num_columns(),
            k,
            rng,
            metric_type,
        )
    }

    /// Train a KMeans model on data with `k` clusters.
    pub async fn new(data: &FixedSizeListArray, k: usize, max_iters: u32) -> Result<Self> {
        let params = KMeansParams {
            max_iters,
            distance_type: MetricType::L2,
            ..Default::default()
        };
        Self::new_with_params(data, k, &params).await
    }

    /// Train a [`KMeans`] model with full parameters.
    ///
    /// If the MetricType is `Cosine`, the input vectors will be normalized with each iteration.
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

        if !data.value_type().is_floating() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "KMeans: data must be floating number, got: {}",
                data.value_type()
            )));
        }

        let data = data
            .values()
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .ok_or(Error::InvalidArgumentError(format!(
                "KMeans: data must be floating number, got: {}",
                data.value_type()
            )))?;

        let mat = MatrixView::<T>::new(Arc::new(data.clone()), dimension);
        // TODO: refactor kmeans to work with reference instead of Arc?
        let mut best_kmeans = Self::empty(k, dimension, params.distance_type);
        let mut best_stddev = f32::MAX;

        // TODO: use seed for Rng.
        let rng = SmallRng::from_entropy();
        for redo in 1..=params.redos {
            let mut kmeans = match params.init {
                KMeanInit::Random => Self::init_random(&mat, k, params.distance_type, rng.clone())?,
                KMeanInit::KMeanPlusPlus => {
                    unimplemented!()
                }
            };

            let mut dist_sum = f64::MAX;
            let mut stddev = f32::MAX;
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
                kmeans = last_membership.to_kmeans(data.as_slice()).unwrap();
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
    async fn train_once(&self, data: &MatrixView<T>) -> KMeanMembership {
        self.compute_membership(data.data().clone()).await
    }

    /// Recompute the membership of each vector.
    ///
    /// Parameters:
    ///
    /// - *data*: a `N * dimension` floating array. Not necessarily normalized.
    ///
    pub async fn compute_membership(&self, data: Arc<T::ArrayType>) -> KMeanMembership {
        let dimension = self.dimension;
        let n = data.len() / self.dimension;
        let distance_type = self.distance_type;
        const CHUNK_SIZE: usize = 1024;

        let cluster_with_distances = stream::iter((0..n).step_by(CHUNK_SIZE))
            // make tiles of input data to split between threads.
            .zip(repeat_with(|| (data.clone(), self.centroids.clone())))
            .map(|(start_idx, (data, centroids))| async move {
                let data = tokio::task::spawn_blocking(move || {
                    let last_idx = min(start_idx + CHUNK_SIZE, n);

                    let centroids_array = centroids.as_slice();
                    let values = &data.as_slice()[start_idx * dimension..last_idx * dimension];

                    match distance_type {
                        DistanceType::L2 => {
                            return compute_partitions_l2(centroids_array, values, dimension)
                                .collect();
                        }
                        DistanceType::Dot => values
                            .chunks_exact(dimension)
                            .map(|vector| {
                                let centroid_stream = centroids_array.chunks_exact(dimension);
                                argmin_value(centroid_stream.map(|cent| dot_distance(vector, cent)))
                            })
                            .collect::<Vec<_>>(),
                        _ => {
                            panic!("KMeans: distance type {} is not supported", distance_type);
                        }
                    }
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
            dimension,
            cluster_id_and_distances: cluster_with_distances.iter().flatten().copied().collect(),
            k: self.k,
            distance_type: self.distance_type,
        }
    }

    pub fn find_partitions(&self, query: &[T::Native], nprobes: usize) -> Result<UInt32Array> {
        if query.len() != self.dimension {
            return Err(Error::InvalidArgumentError(format!(
                "KMeans::find_partitions: query dimension mismatch: {} != {}",
                query.len(),
                self.dimension
            )));
        };

        let dists: Vec<f32> = match self.distance_type {
            MetricType::L2 => {
                l2_distance_batch(query, self.centroids.as_slice(), self.dimension).collect()
            }
            MetricType::Dot => {
                dot_distance_batch(query, self.centroids.as_slice(), self.dimension).collect()
            }
            _ => {
                panic!(
                    "KMeans::find_partitions: {} is not supported",
                    self.distance_type
                );
            }
        };

        let dists_arr = Float32Array::from(dists);
        sort_to_indices(&dists_arr, None, Some(nprobes))
    }
}

/// Return a slice of `data[x,y..y+strip]`.
#[inline]
fn get_slice<T: Num>(data: &[T], x: usize, y: usize, dim: usize, strip: usize) -> &[T] {
    &data[x * dim + y..x * dim + y + strip]
}

/// Compute L2 kmeans partitions with small number of centroids, while the tiling overhead is big.
fn compute_partitions_l2_small<'a, T: L2>(
    centroids: &'a [T],
    data: &'a [T],
    dim: usize,
) -> impl Iterator<Item = Option<(u32, f32)>> + 'a {
    data.chunks(dim)
        .map(move |row| argmin_value_float(l2_distance_batch(row, centroids, dim)))
}

/// Fast partition computation for L2 distance.
///
/// Parameters
/// ----------
/// *centroids*: the flat array of the centroids to run against to.
/// *data*: the flat array of the vectors.
/// *dim*: the dimension of centroids / vectors.
///
/// Returns
/// -------
/// An iterator of ``(partition_id, dist)`` pairs.
///
/// If the distance is not valid, returns ``None`` as placeholder.
///
fn compute_partitions_l2<'a, T: L2>(
    centroids: &'a [T],
    data: &'a [T],
    dim: usize,
) -> Box<dyn Iterator<Item = Option<(u32, f32)>> + 'a> {
    if std::mem::size_of_val(centroids) <= 16 * 1024 {
        return Box::new(compute_partitions_l2_small(centroids, data, dim));
    }

    const STRIPE_SIZE: usize = 128;
    const TILE_SIZE: usize = 16;

    // 128 * 4bytes * 16 = 8KB for centroid and data respectively, so both of them can
    // stay in L1 cache.
    let num_centroids = centroids.len() / dim;

    // Read a tile of vectors, `data[idx..idx+TILE_SIZE]`
    let stream = data.chunks(TILE_SIZE * dim).flat_map(move |data_tile| {
        // Loop over each strip.
        // s is the index of value in each vector.
        let num_rows_in_tile = data_tile.len() / dim;
        let mut min_dists = vec![f32::infinity(); num_rows_in_tile];
        let mut partitions: Vec<Option<u32>> = vec![None; num_rows_in_tile];

        for centroid_start in (0..num_centroids).step_by(TILE_SIZE) {
            // 4B * 16 * 16 = 1 KB
            let mut dists = [0.0; TILE_SIZE * TILE_SIZE];
            let num_centroids_in_tile = min(TILE_SIZE, num_centroids - centroid_start);
            for s in (0..dim).step_by(STRIPE_SIZE) {
                // Calculate L2 within each TILE * STRIP
                let slice_len = min(STRIPE_SIZE, dim - s);
                for di in 0..num_rows_in_tile {
                    let data_slice = get_slice(data_tile, di, s, dim, slice_len);
                    for ci in centroid_start..centroid_start + num_centroids_in_tile {
                        // Get a slice of `data[di][s..s+STRIP_SIZE]`.
                        let cent_slice = get_slice(centroids, ci, s, dim, slice_len);
                        let dist = l2(data_slice, cent_slice);
                        dists[di * TILE_SIZE + (ci - centroid_start)] += dist;
                    }
                }
            }

            for i in 0..num_rows_in_tile {
                if let Some((part_id, dist)) = argmin_value(
                    dists[i * TILE_SIZE..(i * TILE_SIZE + num_centroids_in_tile)]
                        .iter()
                        .copied(),
                ) {
                    if dist < min_dists[i] {
                        min_dists[i] = dist;
                        partitions[i] = Some(centroid_start as u32 + part_id);
                    }
                }
            }
        }
        partitions
            .into_iter()
            .zip(min_dists)
            .map(|(p, d)| p.map(|p| (p, d)))
    });
    Box::new(stream)
}

/// Compute partition ID of each vector in the KMeans.
///
/// If returns `None`, means the vector is not valid, i.e., all `NaN`.
pub async fn compute_partitions<T: ArrowFloatType>(
    centroids: Arc<T::ArrayType>,
    vectors: Arc<T::ArrayType>,
    dimension: usize,
    metric_type: MetricType,
) -> Vec<Option<u32>>
where
    T::Native: L2 + Dot + Normalize,
{
    let kmeans: KMeans<T> = KMeans::with_centroids(centroids, dimension, metric_type);
    let membership = kmeans.compute_membership(vectors).await;
    membership
        .cluster_id_and_distances
        .iter()
        .map(|cluster_and_dist| cluster_and_dist.map(|(cluster, _)| cluster))
        .collect()
}

#[cfg(test)]
mod tests {

    use std::iter::repeat;

    use super::*;

    use arrow_array::types::Float32Type;
    use lance_arrow::*;
    use lance_testing::datagen::generate_random_array;

    use crate::kernels::argmin;

    #[tokio::test]
    async fn test_train_with_small_dataset() {
        let data = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        let data = FixedSizeListArray::try_new_from_values(data, 2).unwrap();
        match KMeans::<Float32Type>::new(&data, 128, 5).await {
            Ok(_) => panic!("Should fail to train KMeans"),
            Err(e) => {
                assert!(e.to_string().contains("smaller than"));
            }
        }
    }

    #[tokio::test]
    async fn test_compute_partitions() {
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
                        .map(|centroid| l2(row, centroid)),
                )
            })
            .collect::<Vec<_>>();
        let actual = compute_partitions::<Float32Type>(
            Arc::new(centroids),
            Arc::new(data),
            DIM,
            MetricType::L2,
        )
        .await;
        assert_eq!(expected, actual);
    }

    #[tokio::test]
    async fn test_l2_with_nans() {
        const DIM: usize = 8;
        const K: usize = 32;
        const NUM_CENTROIDS: usize = 16 * 2048;
        let centroids = generate_random_array(DIM * NUM_CENTROIDS);
        let values = Float32Array::from_iter_values(repeat(f32::NAN).take(DIM * K));

        compute_partitions_l2(centroids.values(), values.values(), DIM).for_each(|cd| {
            assert!(cd.is_none());
        });
    }

    #[tokio::test]
    async fn test_train_l2_kmeans_with_nans() {
        const DIM: usize = 8;
        const K: usize = 32;
        const NUM_CENTROIDS: usize = 16 * 2048;
        let centroids = generate_random_array(DIM * NUM_CENTROIDS);
        let values = Float32Array::from_iter_values(repeat(f32::NAN).take(DIM * K));

        let kmeans = KMeans::<Float32Type>::with_centroids(centroids.into(), DIM, MetricType::L2);
        let membership = kmeans.compute_membership(values.into()).await;

        membership
            .cluster_id_and_distances
            .iter()
            .for_each(|cd| assert!(cd.is_none()));
    }
}
