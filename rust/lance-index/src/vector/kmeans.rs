// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! KMeans implementation for Apache Arrow Arrays.
//!
//! Support ``l2``, ``cosine`` and ``dot`` distances, see [DistanceType].
//!
//! ``Cosine`` distance are calculated by normalizing the vectors to unit length,
//! and run ``l2`` distance on the unit vectors.
//!

use std::ops::{AddAssign, DivAssign};
use std::sync::Arc;
use std::vec;
use std::{collections::HashMap, ops::MulAssign};

use arrow_array::{
    cast::AsArray,
    types::{ArrowPrimitiveType, Float16Type, Float32Type, Float64Type, UInt8Type},
    Array, ArrayRef, FixedSizeListArray, Float32Array, PrimitiveArray, UInt32Array,
};
use arrow_array::{ArrowNumericType, UInt8Array};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{ArrowError, DataType};
use bitvec::prelude::*;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use log::{info, warn};
use num_traits::One;
use num_traits::{AsPrimitive, Float, FromPrimitive, Num, Zero};
use rand::prelude::*;
use rayon::prelude::*;
use snafu::location;

use crate::vector::flat::storage::FlatFloatStorage;
use crate::vector::utils::SimpleIndex;
use crate::{Error, Result};
use lance_linalg::distance::hamming::{hamming, hamming_distance_batch};
use lance_linalg::distance::{dot_distance_batch, DistanceType, Normalize};
use lance_linalg::kernels::argmin_value_float;
use {
    lance_linalg::distance::{
        l2::{l2_distance_batch, L2},
        Dot,
    },
    lance_linalg::kernels::argmin_value,
};

/// KMean initialization method.
#[derive(Debug, PartialEq)]
pub enum KMeanInit {
    Random,
    Incremental(Arc<FixedSizeListArray>),
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
    pub fn new(
        centroids: Option<Arc<FixedSizeListArray>>,
        max_iters: u32,
        redos: usize,
        distance_type: DistanceType,
    ) -> Self {
        let init = match centroids {
            Some(centroids) => KMeanInit::Incremental(centroids),
            None => KMeanInit::Random,
        };
        Self {
            max_iters,
            redos,
            distance_type,
            init,
            ..Default::default()
        }
    }
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
        loss: f64::MAX,
    }
}

/// Split one big cluster into two smaller clusters. After split, each
/// cluster has approximately half of the vectors.
fn split_clusters<T: Float + MulAssign>(
    n: usize,
    cnts: &mut [u64],
    centroids: &mut [T],
    dim: usize,
) {
    let eps = T::from(1.0 / 1024.0).unwrap();
    let mut rng = SmallRng::from_entropy();
    for i in 0..cnts.len() {
        if cnts[i] == 0 {
            let mut j = 0;
            loop {
                let p = (cnts[j] as f32 - 1.0) / (n - cnts.len()) as f32;
                if rng.gen::<f32>() < p {
                    break;
                }
                j += 1;
                j %= cnts.len();
            }

            cnts[i] = cnts[j] / 2;
            cnts[j] -= cnts[i];
            for k in 0..dim {
                if k % 2 == 0 {
                    centroids[i * dim + k] = centroids[j * dim + k] * (T::one() + eps);
                    centroids[j * dim + k] *= T::one() - eps;
                } else {
                    centroids[i * dim + k] = centroids[j * dim + k] * (T::one() - eps);
                    centroids[j * dim + k] *= T::one() + eps;
                }
            }
        }
    }
}

fn histogram(k: usize, membership: &[Option<u32>]) -> Vec<usize> {
    let mut hist: Vec<usize> = vec![0; k];
    membership.iter().for_each(|cd| {
        if let Some(cd) = cd {
            hist[*cd as usize] += 1;
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
        index: Option<&SimpleIndex>,
    ) -> (Vec<Option<u32>>, f64);

    /// Construct a new KMeans model.
    fn to_kmeans(
        data: &[T],
        dimension: usize,
        k: usize,
        membership: &[Option<u32>],
        distance_type: DistanceType,
        loss: f64,
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
    T::Native: Float + Dot + L2 + MulAssign + DivAssign + AddAssign + FromPrimitive + Sync,
    PrimitiveArray<T>: From<Vec<T::Native>>,
{
    fn compute_membership_and_loss(
        centroids: &[T::Native],
        data: &[T::Native],
        dimension: usize,
        distance_type: DistanceType,
        index: Option<&SimpleIndex>,
    ) -> (Vec<Option<u32>>, f64) {
        let cluster_and_dists = match index {
            Some(index) => data
                .par_chunks(dimension)
                .map(|vec| {
                    let query = PrimitiveArray::<T>::from_iter_values(vec.iter().copied());
                    index
                        .search(query)
                        .map(|(id, dist)| Some((id, dist)))
                        .unwrap()
                })
                .collect::<Vec<_>>(),
            None => match distance_type {
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
            },
        };

        (
            cluster_and_dists
                .iter()
                .map(|cd| cd.map(|(c, _)| c))
                .collect::<Vec<_>>(),
            cluster_and_dists
                .iter()
                .map(|cd| cd.map(|(_, d)| d as f64).unwrap_or_default())
                .sum(),
        )
    }

    fn to_kmeans(
        data: &[T::Native],
        dimension: usize,
        k: usize,
        membership: &[Option<u32>],
        distance_type: DistanceType,
        loss: f64,
    ) -> KMeans {
        let mut cluster_cnts = vec![0_u64; k];
        let mut centroids = vec![T::Native::zero(); k * dimension];

        let mut num_cpus = get_num_compute_intensive_cpus();
        if k < num_cpus || k < 16 {
            num_cpus = 1;
        }
        let chunk_size = k / num_cpus;

        centroids
            .par_chunks_mut(dimension * chunk_size)
            .zip(cluster_cnts.par_chunks_mut(chunk_size))
            .enumerate()
            .with_max_len(1)
            .for_each(|(i, (centroids, cnts))| {
                let start = i * chunk_size;
                let end = ((i + 1) * chunk_size).min(k);
                data.chunks(dimension)
                    .zip(membership.iter())
                    .filter_map(|(vector, cluster_id)| {
                        cluster_id.map(|cluster_id| (vector, cluster_id as usize))
                    })
                    .for_each(|(vector, cluster_id)| {
                        if start <= cluster_id && cluster_id < end {
                            let local_id = cluster_id - start;
                            cnts[local_id] += 1;
                            let centroid =
                                &mut centroids[local_id * dimension..(local_id + 1) * dimension];
                            centroid.iter_mut().zip(vector).for_each(|(c, v)| *c += *v);
                        }
                    });
            });

        centroids
            .par_chunks_mut(dimension)
            .zip(cluster_cnts.par_iter())
            .for_each(|(centroid, &cnt)| {
                if cnt > 0 {
                    let norm = T::Native::one() / T::Native::from_u64(cnt).unwrap();
                    centroid.iter_mut().for_each(|v| *v *= norm);
                }
            });

        let empty_clusters = cluster_cnts.iter().filter(|&cnt| *cnt == 0).count();
        if empty_clusters as f32 / k as f32 > 0.1 {
            warn!(
                "KMeans: more than 10% of clusters are empty: {} of {}.\nHelp: this could mean your dataset \
                is too small to have a meaningful index (less than 5000 vectors) or has many duplicate vectors.",
                empty_clusters, k
            );
        }

        split_clusters(
            data.len() / dimension,
            &mut cluster_cnts,
            &mut centroids,
            dimension,
        );

        KMeans {
            centroids: Arc::new(PrimitiveArray::<T>::from(centroids)),
            dimension,
            distance_type,
            loss,
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
        _: Option<&SimpleIndex>,
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
                .map(|cd| cd.map(|(_, d)| d as f64).unwrap_or_default())
                .sum(),
        )
    }

    fn to_kmeans(
        data: &[u8],
        dimension: usize,
        k: usize,
        membership: &[Option<u32>],
        distance_type: DistanceType,
        loss: f64,
    ) -> KMeans {
        assert_eq!(distance_type, DistanceType::Hamming);

        let mut clusters = HashMap::<u32, Vec<usize>>::new();
        membership.iter().enumerate().for_each(|(i, part_id)| {
            if let Some(part_id) = part_id {
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
            loss,
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

    /// The loss of the last training.
    pub loss: f64,
}

impl KMeans {
    fn empty(dimension: usize, distance_type: DistanceType) -> Self {
        Self {
            centroids: arrow_array::array::new_empty_array(&DataType::Float32),
            dimension,
            distance_type,
            loss: f64::MAX,
        }
    }

    /// Create a [`KMeans`] with existing centroids.
    /// It is useful for continuing training.
    pub fn with_centroids(
        centroids: ArrayRef,
        dimension: usize,
        distance_type: DistanceType,
        loss: f64,
    ) -> Self {
        assert!(matches!(
            centroids.data_type(),
            DataType::Float16 | DataType::Float32 | DataType::Float64 | DataType::UInt8
        ));
        Self {
            centroids,
            dimension,
            distance_type,
            loss,
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
    pub fn new(data: &FixedSizeListArray, k: usize, max_iters: u32) -> arrow::error::Result<Self> {
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
    ) -> arrow::error::Result<Self>
    where
        T::Native: Num,
    {
        let dimension = data.value_length() as usize;

        let data =
            data.values()
                .as_primitive_opt::<T>()
                .ok_or(ArrowError::InvalidArgumentError(format!(
                    "KMeans: data must be {}, got: {}",
                    T::DATA_TYPE,
                    data.value_type()
                )))?;

        let mut best_kmeans = Self::empty(dimension, params.distance_type);
        let mut best_stddev = f32::MAX;

        // TODO: use seed for Rng.
        let rng = SmallRng::from_entropy();
        for redo in 1..=params.redos {
            let mut kmeans: Self = match &params.init {
                KMeanInit::Random => Self::init_random::<T>(
                    data.values(),
                    dimension,
                    k,
                    rng.clone(),
                    params.distance_type,
                ),
                KMeanInit::Incremental(centroids) => Self::with_centroids(
                    centroids.values().clone(),
                    dimension,
                    params.distance_type,
                    f64::MAX,
                ),
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

                let index = Self::may_train_index(
                    kmeans.centroids.clone(),
                    kmeans.dimension,
                    kmeans.distance_type,
                )?;
                let (membership, last_loss) = Algo::compute_membership_and_loss(
                    kmeans.centroids.as_primitive::<T>().values(),
                    data.values(),
                    dimension,
                    params.distance_type,
                    index.as_ref(),
                );
                kmeans = Algo::to_kmeans(
                    data.values(),
                    dimension,
                    k,
                    &membership,
                    params.distance_type,
                    last_loss,
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
    ) -> arrow::error::Result<Self> {
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

    // train an HNSW over the centroids to speed up finding the nearest clusters,
    // only train if all conditions are met:
    //  - the centroids are float32s or uint8s
    //  - `num_centroids * dimension >= 1_000_000`
    //      we benchmarked that it's 2x faster in the case of 1024 centroids and 1024 dimensions,
    //      so set the threshold to 1_000_000.
    fn may_train_index(
        centroids: ArrayRef,
        dimension: usize,
        distance_type: DistanceType,
    ) -> Result<Option<SimpleIndex>> {
        if centroids.len() < 1_000_000 {
            // the centroids are stored in a flat array,
            // the length of the centroids is `num_centroids * dimension`
            return Ok(None);
        }

        match centroids.data_type() {
            DataType::Float32 => {
                let fsl =
                    FixedSizeListArray::try_new_from_values(centroids.clone(), dimension as i32)?;
                let store = FlatFloatStorage::new(fsl, distance_type);
                SimpleIndex::try_new(store).map(|index| Some(index))
            }
            _ => return Ok(None),
        }
    }
}

pub fn kmeans_find_partitions_arrow_array(
    centroids: &FixedSizeListArray,
    query: &dyn Array,
    nprobes: usize,
    distance_type: DistanceType,
) -> arrow::error::Result<UInt32Array> {
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
) -> arrow::error::Result<UInt32Array> {
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
) -> arrow::error::Result<UInt32Array> {
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
) -> arrow::error::Result<(Vec<Option<u32>>, f64)> {
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
        (DataType::Float32, DataType::Int8) => Ok(compute_partitions::<
            Float32Type,
            KMeansAlgoFloat<Float32Type>,
        >(
            centroids.values().as_primitive(),
            vectors.convert_to_floating_point()?.values().as_primitive(),
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
            "Centroids and vectors have incompatible types".to_string(),
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
) -> (Vec<Option<u32>>, f64)
where
    T::Native: Num,
{
    let dimension = dimension.as_();
    K::compute_membership_and_loss(
        centroids.values(),
        vectors.values(),
        dimension,
        distance_type,
        None,
    )
}

#[inline]
pub fn compute_partition<T: Float + L2 + Dot>(
    centroids: &[T],
    vector: &[T],
    distance_type: DistanceType,
) -> Option<u32> {
    match distance_type {
        DistanceType::L2 => {
            argmin_value_float(l2_distance_batch(vector, centroids, vector.len())).map(|(c, _)| c)
        }
        DistanceType::Dot => {
            argmin_value_float(dot_distance_batch(vector, centroids, vector.len())).map(|(c, _)| c)
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
    use std::iter::repeat_n;

    use lance_arrow::*;
    use lance_testing::datagen::generate_random_array;

    use super::*;
    use lance_linalg::distance::l2;
    use lance_linalg::kernels::argmin;

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
        let (actual, _) = compute_partitions::<Float32Type, KMeansAlgoFloat<Float32Type>>(
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
            None,
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
        let values = Float32Array::from_iter_values(repeat_n(f32::NAN, DIM * K));

        compute_partitions::<Float32Type, KMeansAlgoFloat<Float32Type>>(
            &centroids,
            &values,
            DIM,
            DistanceType::L2,
        )
        .0
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
        let values = repeat_n(f32::NAN, DIM * K).collect::<Vec<_>>();

        let (membership, _) = KMeansAlgoFloat::<Float32Type>::compute_membership_and_loss(
            centroids.as_slice(),
            &values,
            DIM,
            DistanceType::L2,
            None,
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

/// Train KMeans model and returns the centroids of each cluster.
///
/// Parameters
/// ----------
/// - *centroids*: initial centroids, use the random initialization if None
/// - *array*: a flatten floating number array of vectors
/// - *dimension*: dimension of the vector
/// - *k*: number of clusters
/// - *max_iterations*: maximum number of iterations
/// - *redos*: number of times to redo the k-means clustering
/// - *distance_type*: distance type to compute pair-wise vector distance
/// - *sample_rate*: sample rate to select the data for training
#[allow(clippy::too_many_arguments)]
pub fn train_kmeans<T: ArrowPrimitiveType>(
    centroids: Option<Arc<FixedSizeListArray>>,
    array: &PrimitiveArray<T>,
    dimension: usize,
    k: usize,
    max_iterations: u32,
    redos: usize,
    distance_type: DistanceType,
    sample_rate: usize,
) -> Result<KMeans>
where
    T::Native: Dot + L2 + Normalize,
    PrimitiveArray<T>: From<Vec<T::Native>>,
{
    let num_rows = array.len() / dimension;
    if num_rows < k {
        return Err(Error::Index{message: format!(
            "KMeans: can not train {k} centroids with {num_rows} vectors, choose a smaller K (< {num_rows}) instead"
        ),location: location!()});
    }

    // Only sample sample_rate * num_clusters. See Faiss
    let data = if num_rows > sample_rate * k {
        info!(
            "Sample {} out of {} to train kmeans of {} dim, {} clusters",
            sample_rate * k,
            array.len() / dimension,
            dimension,
            k,
        );
        let sample_size = sample_rate * k;
        array.slice(0, sample_size * dimension)
    } else {
        array.clone()
    };

    let params = KMeansParams::new(centroids, max_iterations, redos, distance_type);
    let data = FixedSizeListArray::try_new_from_values(data, dimension as i32)?;
    let model = KMeans::new_with_params(&data, k, &params)?;
    Ok(model)
}
