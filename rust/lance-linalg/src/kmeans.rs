// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! KMeans implementation for Apache Arrow Arrays.
//!
//! Support ``l2``, ``cosine`` and ``dot`` distances, see [DistanceType].
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
use lance_arrow::{ArrowFloatType, FloatArray};
use log::{info, warn};
use num_traits::{AsPrimitive, Float, FromPrimitive, Num, Zero};
use rand::prelude::*;
use rayon::prelude::*;

use crate::distance::norm_l2::Normalize;
use crate::distance::{dot_distance_batch, DistanceType};
use crate::kernels::{argmax, argmin_value_float};
use crate::{
    clustering::Clustering,
    distance::{
        l2::{l2, l2_distance_batch, L2},
        Dot,
    },
    kernels::argmin_value,
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
    data: &[T::Native],
    dimension: usize,
    k: usize,
    mut rng: impl Rng,
    distance_type: DistanceType,
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
        builder.extend(data[i * dimension..(i + 1) * dimension].iter());
    }
    let mut kmeans = KMeans::empty(k, dimension, distance_type);
    kmeans.centroids = Arc::new(builder.into());
    Ok(kmeans)
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
    /// - *distance_type*: the distance type to calculate distance.
    /// - *rng*: random generator.
    pub fn init_random(
        data: &[T::Native],
        dimension: usize,
        k: usize,
        distance_type: DistanceType,
        rng: impl Rng,
    ) -> Result<Self> {
        kmeans_random_init(data, dimension, k, rng, distance_type)
    }

    /// Train a KMeans model on data with `k` clusters.
    pub fn new(data: &FixedSizeListArray, k: usize, max_iters: u32) -> Result<Self> {
        let params = KMeansParams {
            max_iters,
            distance_type: DistanceType::L2,
            ..Default::default()
        };
        Self::new_with_params(data, k, &params)
    }

    /// Train a [`KMeans`] model with full parameters.
    ///
    /// If the DistanceType is `Cosine`, the input vectors will be normalized with each iteration.
    pub fn new_with_params(
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

        let mut best_kmeans = Self::empty(k, dimension, params.distance_type);

        // TODO: use seed for Rng.
        let rng = SmallRng::from_entropy();
        for redo in 1..=params.redos {
            let kmeans = match params.init {
                KMeanInit::Random => Self::init_random(
                    data.as_slice(),
                    dimension,
                    k,
                    params.distance_type,
                    rng.clone(),
                )?,
                KMeanInit::KMeanPlusPlus => {
                    unimplemented!()
                }
            };

            let mut loss = f64::MAX;
            for i in 1..=params.max_iters {
                if i % 10 == 0 {
                    info!(
                        "KMeans training: iteration {} / {}, redo={}",
                        i, params.max_iters, redo
                    );
                };
                let (last_membership, last_loss) =
                    kmeans.compute_membership_and_loss(data.as_slice());
                let kmeans = Self::to_kmeans(
                    data.as_slice(),
                    dimension,
                    k,
                    &last_membership,
                    params.distance_type,
                )
                .unwrap();
                if (loss - last_loss).abs() / last_loss < params.tolerance {
                    info!(
                        "KMeans training: converged at iteration {} / {}, redo={}",
                        i, params.max_iters, redo
                    );
                    break;
                }
                loss = last_loss;
                best_kmeans = kmeans;
            }
        }

        Ok(best_kmeans)
    }

    /// Recompute the membership of each vector.
    ///
    /// Parameters:
    ///
    /// - *data*: a `N * dimension` floating array. Not necessarily normalized.
    ///
    fn compute_membership_and_loss(&self, data: &[T::Native]) -> (Vec<Option<u32>>, f64) {
        let centroids = self.centroids.as_ref().as_slice();
        let cluster_and_dists = data
            .par_chunks(self.dimension)
            .map(|vec| {
                argmin_value(match self.distance_type {
                    DistanceType::L2 => l2_distance_batch(vec, centroids, self.dimension),
                    DistanceType::Dot => dot_distance_batch(vec, centroids, self.dimension),
                    _ => {
                        panic!(
                            "KMeans::find_partitions: {} is not supported",
                            self.distance_type
                        );
                    }
                })
            })
            .collect::<Vec<_>>();
        (
            cluster_and_dists
                .iter()
                .map(|cd| cd.map(|(c, _)| c))
                .collect::<Vec<_>>(),
            cluster_and_dists
                .iter()
                .map(|cd| cd.map(|(_, d)| d).unwrap_or_default() as f64)
                .sum(),
        )
    }

    fn to_kmeans(
        data: &[T::Native],
        dimension: usize,
        k: usize,
        membership: &[Option<u32>],
        distance_type: DistanceType,
    ) -> Result<Self>
    where
        T::Native: L2 + Dot + Normalize,
    {
        let mut cluster_cnts = vec![0_u64; k];
        let mut new_centroids = vec![T::Native::zero(); k * dimension];
        data.chunks_exact(dimension)
            .zip(membership.iter())
            .for_each(|(vector, cluster_id)| {
                if let Some(&cluster_id) = cluster_id.as_ref() {
                    cluster_cnts[cluster_id as usize] += 1;
                    // TODO: simd
                    for (old, &new) in new_centroids
                        [cluster_id as usize * dimension..(1 + cluster_id as usize) * dimension]
                        .iter_mut()
                        .zip(vector)
                    {
                        *old += new;
                    }
                }
            });

        let mut empty_clusters = 0;

        cluster_cnts.iter().enumerate().for_each(|(i, &cnt)| {
            if cnt == 0 {
                empty_clusters += 1;
                new_centroids[i * dimension..(i + 1) * dimension]
                    .iter_mut()
                    .for_each(|v| *v = T::Native::nan());
            } else {
                new_centroids[i * dimension..(i + 1) * dimension]
                    .iter_mut()
                    .for_each(|v| *v /= T::Native::from_u64(cnt).unwrap());
            }
        });

        if empty_clusters as f32 / k as f32 > 0.1 {
            warn!(
                "KMeans: more than 10% of clusters are empty: {} of {}.\nHelp: this could mean your dataset \
                is too small to have a meaningful index (less than 5000 vectors) or has many duplicate vectors.",
                empty_clusters, k
            );
        }

        split_clusters(&mut cluster_cnts, &mut new_centroids, dimension);

        Ok(Self {
            centroids: Arc::new(new_centroids.into()),
            dimension,
            k,
            distance_type,
        })
    }
}

impl<T: ArrowFloatType> Clustering<T::Native> for KMeans<T>
where
    T::Native: L2 + Dot,
{
    fn deminsion(&self) -> u32 {
        self.dimension as u32
    }

    fn num_clusters(&self) -> u32 {
        self.k as u32
    }

    fn find_partitions(&self, query: &[T::Native], nprobes: usize) -> Result<UInt32Array> {
        assert_eq!(query.len(), self.dimension);

        let dists: Vec<f32> = match self.distance_type {
            DistanceType::L2 => {
                l2_distance_batch(query, self.centroids.as_slice(), self.dimension).collect()
            }
            DistanceType::Dot => {
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

    fn compute_membership(&self, data: &[T::Native], _nprobes: Option<usize>) -> Vec<Option<u32>> {
        assert!(_nprobes.is_none());

        let centroids = self.centroids.as_ref().as_slice();
        data.par_chunks(self.dimension)
            .map(|vec| {
                argmin_value(match self.distance_type {
                    DistanceType::L2 => l2_distance_batch(vec, centroids, self.dimension),
                    DistanceType::Dot => dot_distance_batch(vec, centroids, self.dimension),
                    _ => {
                        panic!(
                            "KMeans::find_partitions: {} is not supported",
                            self.distance_type
                        );
                    }
                })
                .map(|(idx, _)| idx)
            })
            .collect::<Vec<_>>()
    }
}

/// Compute partition ID of each vector in the KMeans.
///
/// If returns `None`, means the vector is not valid, i.e., all `NaN`.
pub fn compute_partitions<T: ArrowFloatType>(
    centroids: Arc<T::ArrayType>,
    vectors: Arc<T::ArrayType>,
    dimension: usize,
    distance_type: DistanceType,
) -> Vec<Option<u32>>
where
    T::Native: L2 + Dot + Normalize,
{
    let kmeans = KMeans::<T>::with_centroids(centroids, dimension, distance_type);
    kmeans.compute_membership(vectors.as_slice(), None)
}

#[cfg(test)]
mod tests {
    use std::iter::repeat;

    use arrow_array::types::Float32Type;
    use lance_arrow::*;
    use lance_testing::datagen::generate_random_array;

    use super::*;
    use crate::kernels::argmin;

    #[test]
    fn test_train_with_small_dataset() {
        let data = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        let data = FixedSizeListArray::try_new_from_values(data, 2).unwrap();
        match KMeans::<Float32Type>::new(&data, 128, 5) {
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
                        .map(|centroid| l2(row, centroid)),
                )
            })
            .collect::<Vec<_>>();
        let actual = compute_partitions::<Float32Type>(
            Arc::new(centroids),
            Arc::new(data),
            DIM,
            DistanceType::L2,
        );
        assert_eq!(expected, actual);
    }

    #[tokio::test]
    async fn test_compute_membership_and_loss() {
        const DIM: usize = 256;
        let centroids = generate_random_array(DIM * 18);
        let data = generate_random_array(DIM * 20);

        let kmeans = KMeans::<Float32Type>::with_centroids(centroids.into(), DIM, DistanceType::L2);
        let (membership, loss) = kmeans.compute_membership_and_loss(data.values());
        assert!(loss > 0.0, "loss is not zero: {}", loss);
        membership.iter().for_each(|cd| {
            assert!(cd.is_some());
        });
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
        let values = repeat(f32::NAN).take(DIM * K).collect::<Vec<_>>();

        let kmeans = KMeans::<Float32Type>::with_centroids(centroids.into(), DIM, DistanceType::L2);
        let (membership, _) = kmeans.compute_membership_and_loss(&values);

        membership.iter().for_each(|cd| assert!(cd.is_none()));
    }
}
