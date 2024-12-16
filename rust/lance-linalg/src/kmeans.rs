// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! KMeans implementation for Apache Arrow Arrays.
//!
//! Support ``l2``, ``cosine`` and ``dot`` distances, see [DistanceType].
//!
//! ``Cosine`` distance are calculated by normalizing the vectors to unit length,
//! and run ``l2`` distance on the unit vectors.
//!

use std::collections::HashMap;
use std::ops::{AddAssign, DivAssign};
use std::sync::Arc;
use std::vec;

use arrow_array::{
    cast::AsArray,
    types::{ArrowPrimitiveType, Float16Type, Float32Type, Float64Type, UInt8Type},
    Array, ArrayRef, FixedSizeListArray, Float32Array, PrimitiveArray, UInt32Array,
};
use arrow_array::{ArrowNumericType, UInt8Array};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{ArrowError, DataType};
use bitvec::prelude::*;
use log::{info, warn};
use num_traits::{AsPrimitive, Float, FromPrimitive, Num, Zero};
use rand::prelude::*;
use rayon::prelude::*;

use crate::distance::hamming::{hamming, hamming_distance_batch};
use crate::distance::{dot_distance_batch, DistanceType};
use crate::kernels::{argmax, argmin_value_float};
use crate::{
    distance::{
        l2::{l2_distance_batch, L2},
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
pub struct KMeans {
    /// Flattened array of centroids.
    ///
    /// dimension * k of floating number.
    pub centroids: ArrayRef,

    /// The dimension of each vector.
    pub dimension: usize,

    /// How to calculate distance between two vectors.
    pub distance_type: DistanceType,
}

/// Randomly initialize kmeans centroids.
///
///
fn kmeans_random_init<T: ArrowPrimitiveType>(
    data: &[T::Native],
    dimension: usize,
    k: usize,
    mut rng: impl Rng,
    distance_type: DistanceType,
) -> KMeans {
    assert!(data.len() >= k * dimension);
    let chosen = (0..data.len() / dimension).choose_multiple(&mut rng, k);
    let centroids = PrimitiveArray::<T>::from_iter_values(
        chosen
            .iter()
            .flat_map(|&i| data[i * dimension..(i + 1) * dimension].iter())
            .copied(),
    );
    KMeans {
        centroids: Arc::new(centroids),
        dimension,
        distance_type,
    }
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

fn histogram(k: usize, membership: &[Option<u32>]) -> Vec<usize> {
    let mut hist: Vec<usize> = vec![0; k];
    membership.iter().for_each(|cd| {
        if let Some(cluster_id) = cd {
            hist[*cluster_id as usize] += 1;
        }
    });

    hist
}

/// Std deviation of the histogram / cluster distribution.
fn hist_stddev(k: usize, membership: &[Option<u32>]) -> f32 {
    let mean: f32 = membership.len() as f32 * 1.0 / k as f32;
    let len = membership.len();
    (histogram(k, membership)
        .par_iter()
        .map(|c| (*c as f32 - mean).powi(2))
        .sum::<f32>()
        / len as f32)
        .sqrt()
}

pub trait KMeansAlgo<T: Num> {
    /// Recompute the membership of each vector.
    ///
    /// Parameters:
    ///
    /// - *data*: a `N * dimension` floating array. Not necessarily normalized.
    ///
    fn compute_membership_and_loss(
        centroids: &[T],
        data: &[T],
        dimension: usize,
        distance_type: DistanceType,
    ) -> (Vec<Option<u32>>, f64);

    /// Construct a new KMeans model.
    fn to_kmeans(
        data: &[T],
        dimension: usize,
        k: usize,
        membership: &[Option<u32>],
        distance_type: DistanceType,
    ) -> KMeans;
}

pub struct KMeansAlgoFloat<T: ArrowNumericType>
where
    T::Native: Float + Num,
{
    phantom_data: std::marker::PhantomData<T>,
}

impl<T: ArrowNumericType> KMeansAlgo<T::Native> for KMeansAlgoFloat<T>
where
    T::Native: Float + Dot + L2 + DivAssign + AddAssign + FromPrimitive + Sync,
{
    fn compute_membership_and_loss(
        centroids: &[T::Native],
        data: &[T::Native],
        dimension: usize,
        distance_type: DistanceType,
    ) -> (Vec<Option<u32>>, f64) {
        let cluster_and_dists = match distance_type {
            DistanceType::L2 => data
                .par_chunks(dimension)
                .map(|vec| argmin_value_float(l2_distance_batch(vec, centroids, dimension)))
                .collect::<Vec<_>>(),
            DistanceType::Dot => data
                .par_chunks(dimension)
                .map(|vec| argmin_value_float(dot_distance_batch(vec, centroids, dimension)))
                .collect::<Vec<_>>(),
            _ => {
                panic!(
                    "KMeans::find_partitions: {} is not supported",
                    distance_type
                );
            }
        };
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
    ) -> KMeans {
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

        KMeans {
            centroids: Arc::new(PrimitiveArray::<T>::from_iter_values(new_centroids)),
            dimension,
            distance_type,
        }
    }
}

struct KModeAlgo {}

impl KMeansAlgo<u8> for KModeAlgo {
    fn compute_membership_and_loss(
        centroids: &[u8],
        data: &[u8],
        dimension: usize,
        distance_type: DistanceType,
    ) -> (Vec<Option<u32>>, f64) {
        assert_eq!(distance_type, DistanceType::Hamming);
        let cluster_and_dists = data
            .par_chunks(dimension)
            .map(|vec| {
                argmin_value(
                    centroids
                        .par_chunks(dimension)
                        .map(|c| hamming(vec, c))
                        .collect::<Vec<f32>>()
                        .into_iter(),
                )
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
        data: &[u8],
        dimension: usize,
        k: usize,
        membership: &[Option<u32>],
        distance_type: DistanceType,
    ) -> KMeans {
        assert_eq!(distance_type, DistanceType::Hamming);

        let mut clusters = HashMap::<u32, Vec<usize>>::new();
        membership.iter().enumerate().for_each(|(i, part_id_opt)| {
            if let Some(part_id) = part_id_opt {
                clusters.entry(*part_id).or_default().push(i);
            }
        });
        let centroids = (0..k as u32)
            .into_par_iter()
            .flat_map(|part_id| {
                if let Some(vecs) = clusters.get(&part_id) {
                    let mut ones = vec![0_u32; dimension * 8];
                    let cnt = vecs.len() as u32;
                    vecs.iter().for_each(|&i| {
                        let vec = &data[i * dimension..(i + 1) * dimension];
                        ones.iter_mut()
                            .zip(vec.view_bits::<Lsb0>())
                            .for_each(|(c, v)| {
                                if *v.as_ref() {
                                    *c += 1;
                                }
                            });
                    });

                    let bits = ones.iter().map(|&c| c * 2 > cnt).collect::<BitVec<u8>>();
                    bits.as_raw_slice()
                        .iter()
                        .copied()
                        .map(Some)
                        .collect::<Vec<_>>()
                } else {
                    vec![None; dimension]
                }
            })
            .collect::<Vec<_>>();

        KMeans {
            centroids: Arc::new(UInt8Array::from(centroids)),
            dimension,
            distance_type,
        }
    }
}

impl KMeans {
    fn empty(dimension: usize, distance_type: DistanceType) -> Self {
        Self {
            centroids: arrow_array::array::new_empty_array(&DataType::Float32),
            dimension,
            distance_type,
        }
    }

    /// Create a [`KMeans`] with existing centroids.
    /// It is useful for continuing training.
    pub fn with_centroids(
        centroids: ArrayRef,
        dimension: usize,
        distance_type: DistanceType,
    ) -> Self {
        assert!(matches!(
            centroids.data_type(),
            DataType::Float16 | DataType::Float32 | DataType::Float64 | DataType::UInt8
        ));
        Self {
            centroids,
            dimension,
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
    fn init_random<T: ArrowPrimitiveType>(
        data: &[T::Native],
        dimension: usize,
        k: usize,
        rng: impl Rng,
        distance_type: DistanceType,
    ) -> Self {
        kmeans_random_init::<T>(data, dimension, k, rng, distance_type)
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

    fn train_kmeans<T: ArrowNumericType, Algo: KMeansAlgo<T::Native>>(
        data: &FixedSizeListArray,
        k: usize,
        params: &KMeansParams,
    ) -> Result<Self>
    where
        T::Native: Num,
    {
        let dimension = data.value_length() as usize;

        let data = data
            .values()
            .as_primitive_opt::<T>()
            .ok_or(Error::InvalidArgumentError(format!(
                "KMeans: data must be {}, got: {}",
                T::DATA_TYPE,
                data.value_type()
            )))?;

        let mut best_kmeans = Self::empty(dimension, params.distance_type);
        let mut best_stddev = f32::MAX;

        // TODO: use seed for Rng.
        let rng = SmallRng::from_entropy();
        for redo in 1..=params.redos {
            let mut kmeans: Self = match params.init {
                KMeanInit::Random => Self::init_random::<T>(
                    data.values(),
                    dimension,
                    k,
                    rng.clone(),
                    params.distance_type,
                ),
                KMeanInit::KMeanPlusPlus => {
                    unimplemented!()
                }
            };

            let mut loss = f64::MAX;
            let mut last_membership: Option<Vec<Option<u32>>> = None;
            for i in 1..=params.max_iters {
                if i % 10 == 0 {
                    info!(
                        "KMeans training: iteration {} / {}, redo={}",
                        i, params.max_iters, redo
                    );
                };
                let (membership, last_loss) = Algo::compute_membership_and_loss(
                    kmeans.centroids.as_primitive::<T>().values(),
                    data.values(),
                    dimension,
                    params.distance_type,
                );
                kmeans = Algo::to_kmeans(
                    data.values(),
                    dimension,
                    k,
                    &membership,
                    params.distance_type,
                );
                last_membership = Some(membership);
                if (loss - last_loss).abs() / last_loss < params.tolerance {
                    info!(
                        "KMeans training: converged at iteration {} / {}, redo={}, loss={}, last_loss={}, loss_diff={}",
                        i, params.max_iters, redo, loss, last_loss, (loss - last_loss).abs() / last_loss
                    );
                    break;
                }
                loss = last_loss;
            }
            let stddev = hist_stddev(
                k,
                last_membership
                    .as_ref()
                    .expect("Last membership should already set"),
            );
            if stddev < best_stddev {
                best_stddev = stddev;
                best_kmeans = kmeans;
            }
        }

        Ok(best_kmeans)
    }

    /// Train a [`KMeans`] model with full parameters.
    ///
    /// If the DistanceType is `Cosine`, the input vectors will be normalized with each iteration.
    pub fn new_with_params(
        data: &FixedSizeListArray,
        k: usize,
        params: &KMeansParams,
    ) -> Result<Self> {
        let n = data.len();
        if n < k {
            return Err(ArrowError::InvalidArgumentError(
                format!(
                    "KMeans: training does not have sufficient data points: n({}) is smaller than k({})",
                    n, k
                )
            ));
        }

        match (data.value_type(), params.distance_type) {
            (DataType::Float16, _) => {
                Self::train_kmeans::<Float16Type, KMeansAlgoFloat<Float16Type>>(data, k, params)
            }

            (DataType::Float32, _) => {
                Self::train_kmeans::<Float32Type, KMeansAlgoFloat<Float32Type>>(data, k, params)
            }
            (DataType::Float64, _) => {
                Self::train_kmeans::<Float64Type, KMeansAlgoFloat<Float64Type>>(data, k, params)
            }
            (DataType::UInt8, DistanceType::Hamming) => {
                Self::train_kmeans::<UInt8Type, KModeAlgo>(data, k, params)
            }
            _ => Err(ArrowError::InvalidArgumentError(format!(
                "KMeans: can not train data type {} with distance type: {}",
                data.value_type(),
                params.distance_type
            ))),
        }
    }
}

pub fn kmeans_find_partitions_arrow_array(
    centroids: &FixedSizeListArray,
    query: &dyn Array,
    nprobes: usize,
    distance_type: DistanceType,
) -> Result<UInt32Array> {
    if centroids.value_length() as usize != query.len() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "Centroids and vectors have different dimensions: {} != {}",
            centroids.value_length(),
            query.len()
        )));
    }

    match (centroids.value_type(), query.data_type()) {
        (DataType::Float16, DataType::Float16) => Ok(kmeans_find_partitions(
            centroids.values().as_primitive::<Float16Type>().values(),
            query.as_primitive::<Float16Type>().values(),
            nprobes,
            distance_type,
        )?),
        (DataType::Float32, DataType::Float32) => Ok(kmeans_find_partitions(
            centroids.values().as_primitive::<Float32Type>().values(),
            query.as_primitive::<Float32Type>().values(),
            nprobes,
            distance_type,
        )?),
        (DataType::Float64, DataType::Float64) => Ok(kmeans_find_partitions(
            centroids.values().as_primitive::<Float64Type>().values(),
            query.as_primitive::<Float64Type>().values(),
            nprobes,
            distance_type,
        )?),
        (DataType::UInt8, DataType::UInt8) => kmeans_find_partitions_binary(
            centroids.values().as_primitive::<UInt8Type>().values(),
            query.as_primitive::<UInt8Type>().values(),
            nprobes,
            distance_type,
        ),
        _ => Err(ArrowError::InvalidArgumentError(format!(
            "Centroids and vectors have different types: {} != {}",
            centroids.value_type(),
            query.data_type()
        ))),
    }
}

/// KMeans finds N nearest partitions.
///
/// Parameters:
/// - *centroids*: a `k * dimension` floating array.
/// - *query*: a `dimension` floating array.
/// - *nprobes*: the number of partitions to find.
/// - *distance_type*: the distance type to calculate distance.
///
/// This function allows to conduct kmeans search without constructing
/// `Arrow Array` or `Vec<Float>` types.
///
pub fn kmeans_find_partitions<T: Float + L2 + Dot>(
    centroids: &[T],
    query: &[T],
    nprobes: usize,
    distance_type: DistanceType,
) -> Result<UInt32Array> {
    let dists: Vec<f32> = match distance_type {
        DistanceType::L2 => l2_distance_batch(query, centroids, query.len()).collect(),
        DistanceType::Dot => dot_distance_batch(query, centroids, query.len()).collect(),
        _ => {
            panic!(
                "KMeans::find_partitions: {} is not supported",
                distance_type
            );
        }
    };

    // TODO: use heap to just keep nprobes smallest values.
    let dists_arr = Float32Array::from(dists);
    sort_to_indices(&dists_arr, None, Some(nprobes))
}

pub fn kmeans_find_partitions_binary(
    centroids: &[u8],
    query: &[u8],
    nprobes: usize,
    distance_type: DistanceType,
) -> Result<UInt32Array> {
    let dists: Vec<f32> = match distance_type {
        DistanceType::Hamming => hamming_distance_batch(query, centroids, query.len()).collect(),
        _ => {
            panic!(
                "KMeans::find_partitions: {} is not supported",
                distance_type
            );
        }
    };

    // TODO: use heap to just keep nprobes smallest values.
    let dists_arr = Float32Array::from(dists);
    sort_to_indices(&dists_arr, None, Some(nprobes))
}

/// Compute partitions from Arrow FixedSizeListArray.
pub fn compute_partitions_arrow_array(
    centroids: &FixedSizeListArray,
    vectors: &FixedSizeListArray,
    distance_type: DistanceType,
) -> Result<Vec<Option<u32>>> {
    if centroids.value_length() != vectors.value_length() {
        return Err(ArrowError::InvalidArgumentError(
            "Centroids and vectors have different dimensions".to_string(),
        ));
    }
    match (centroids.value_type(), vectors.value_type()) {
        (DataType::Float16, DataType::Float16) => Ok(compute_partitions::<
            Float16Type,
            KMeansAlgoFloat<Float16Type>,
        >(
            centroids.values().as_primitive(),
            vectors.values().as_primitive(),
            centroids.value_length(),
            distance_type,
        )),
        (DataType::Float32, DataType::Float32) => Ok(compute_partitions::<
            Float32Type,
            KMeansAlgoFloat<Float32Type>,
        >(
            centroids.values().as_primitive(),
            vectors.values().as_primitive(),
            centroids.value_length(),
            distance_type,
        )),
        (DataType::Float64, DataType::Float64) => Ok(compute_partitions::<
            Float64Type,
            KMeansAlgoFloat<Float64Type>,
        >(
            centroids.values().as_primitive(),
            vectors.values().as_primitive(),
            centroids.value_length(),
            distance_type,
        )),
        (DataType::UInt8, DataType::UInt8) => Ok(compute_partitions::<UInt8Type, KModeAlgo>(
            centroids.values().as_primitive(),
            vectors.values().as_primitive(),
            centroids.value_length(),
            distance_type,
        )),
        _ => Err(ArrowError::InvalidArgumentError(
            "Centroids and vectors have different types".to_string(),
        )),
    }
}

/// Compute partition ID of each vector in the KMeans.
///
/// If returns `None`, means the vector is not valid, i.e., all `NaN`.
pub fn compute_partitions<T: ArrowNumericType, K: KMeansAlgo<T::Native>>(
    centroids: &PrimitiveArray<T>,
    vectors: &PrimitiveArray<T>,
    dimension: impl AsPrimitive<usize>,
    distance_type: DistanceType,
) -> Vec<Option<u32>>
where
    T::Native: Num,
{
    let dimension = dimension.as_();
    let (membership, _) = K::compute_membership_and_loss(
        centroids.values(),
        vectors.values(),
        dimension,
        distance_type,
    );
    membership
}

#[inline]
pub fn compute_partition<T: Float + L2 + Dot>(
    centroids: &[T],
    vector: &[T],
    distance_type: DistanceType,
) -> Option<u32> {
    match distance_type {
        DistanceType::L2 => {
            argmin_value_float(l2_distance_batch(vector, centroids, vector.len())).map(|c| c.0)
        }
        DistanceType::Dot => {
            argmin_value_float(dot_distance_batch(vector, centroids, vector.len())).map(|c| c.0)
        }
        _ => {
            panic!(
                "KMeans::compute_partition: distance type {} is not supported",
                distance_type
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter::repeat;

    use lance_arrow::*;
    use lance_testing::datagen::generate_random_array;

    use super::*;
    use crate::distance::l2;
    use crate::kernels::argmin;

    #[test]
    fn test_train_with_small_dataset() {
        let data = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        let data = FixedSizeListArray::try_new_from_values(data, 2).unwrap();
        match KMeans::new(&data, 128, 5) {
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
        let actual = compute_partitions::<Float32Type, KMeansAlgoFloat<Float32Type>>(
            &centroids,
            &data,
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

        let (membership, loss) = KMeansAlgoFloat::<Float32Type>::compute_membership_and_loss(
            centroids.as_slice(),
            data.values(),
            DIM,
            DistanceType::L2,
        );
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

        compute_partitions::<Float32Type, KMeansAlgoFloat<Float32Type>>(
            &centroids,
            &values,
            DIM,
            DistanceType::L2,
        )
        .iter()
        .for_each(|cd| {
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

        let (membership, _) = KMeansAlgoFloat::<Float32Type>::compute_membership_and_loss(
            centroids.as_slice(),
            &values,
            DIM,
            DistanceType::L2,
        );

        membership.iter().for_each(|cd| assert!(cd.is_none()));
    }

    #[tokio::test]
    async fn test_train_kmode() {
        const DIM: usize = 16;
        const K: usize = 32;
        const NUM_VALUES: usize = 256 * K;

        let mut rng = SmallRng::from_entropy();
        let values =
            UInt8Array::from_iter_values((0..NUM_VALUES * DIM).map(|_| rng.gen_range(0..255)));

        let fsl = FixedSizeListArray::try_new_from_values(values, DIM as i32).unwrap();

        let params = KMeansParams {
            distance_type: DistanceType::Hamming,
            ..Default::default()
        };
        let kmeans = KMeans::new_with_params(&fsl, K, &params).unwrap();
        assert_eq!(kmeans.centroids.len(), K * DIM);
        assert_eq!(kmeans.dimension, DIM);
        assert_eq!(kmeans.centroids.data_type(), &DataType::UInt8);
    }
}
