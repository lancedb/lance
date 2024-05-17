// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Product Quantization
//!

use std::any::Any;
use std::sync::Arc;

use arrow_array::{cast::AsArray, Array, FixedSizeListArray, UInt8Array};
use arrow_array::{ArrayRef, Float32Array};
use lance_arrow::*;
use lance_core::{Error, Result};
use lance_linalg::distance::{
    dot_distance_batch, l2_distance_batch, DistanceType, Dot, Normalize, L2,
};
use lance_linalg::kernels::{argmin, argmin_value_float};
use lance_linalg::{distance::MetricType, MatrixView};
use lance_linalg::{kmeans::KMeans, Clustering};
use rayon::prelude::*;
use snafu::{location, Location};

pub mod builder;
mod distance;
pub mod storage;
pub mod transform;
pub(crate) mod utils;

use self::distance::{build_distance_table_l2, compute_l2_distance};
pub use self::utils::num_centroids;
use super::pb;
pub use builder::PQBuildParams;
use utils::get_sub_vector_centroids;

/// Product Quantization
pub trait ProductQuantizer: Send + Sync + std::fmt::Debug {
    fn as_any(&self) -> &dyn Any;

    /// Compute the distance between query vector to the PQ code.
    ///
    fn compute_distances(&self, query: &dyn Array, code: &UInt8Array) -> Result<Float32Array>;

    fn transform(&self, data: &dyn Array) -> Result<ArrayRef>;

    /// Number of sub-vectors
    fn num_sub_vectors(&self) -> usize;

    fn num_bits(&self) -> u32;

    fn dimension(&self) -> usize;

    // TODO: move to pub(crate) once the refactor of lance::index to lance-index is done.
    fn codebook_as_fsl(&self) -> FixedSizeListArray;

    /// Whether to use residual as input or not.
    fn use_residual(&self) -> bool;
}

/// Product Quantization, optimized for [Apache Arrow] buffer memory layout.
///
//
// TODO: move this to be pub(crate) once we have a better way to test it.
#[derive(Debug, Clone)]
pub struct ProductQuantizerImpl<T: ArrowFloatType>
where
    T::Native: Dot + L2 + Normalize,
{
    /// Number of bits for the centroids.
    ///
    /// Only support 8, as one of `u8` byte now.
    pub num_bits: u32,

    /// Number of sub-vectors.
    pub num_sub_vectors: usize,

    /// Vector dimension.
    pub dimension: usize,

    /// Distance type.
    pub metric_type: MetricType,

    /// PQ codebook
    ///
    /// ```((2 ^ nbits) * num_subvector * sub_vector_length)``` of `f32`
    ///
    /// Use a layout that is cache / SIMD friendly to compute centroid.
    /// But not sure how to make distance lookup via PQ code lookup
    /// be cache friendly tho.
    ///
    /// Layout:
    ///
    ///  - *row*: all centroids for the same sub-vector.
    ///  - *column*: the centroid value of the n-th sub-vector.
    ///
    /// ```text
    /// // Centroids for a sub-vector.
    /// Codebook[sub_vector_id][pq_code]
    /// ```
    pub codebook: Arc<T::ArrayType>,
}

impl<T: ArrowFloatType> ProductQuantizerImpl<T>
where
    T::Native: Dot + L2 + Normalize,
{
    /// Create a [`ProductQuantizer`] with pre-trained codebook.
    pub fn new(
        m: usize,
        nbits: u32,
        dimension: usize,
        codebook: Arc<T::ArrayType>,
        metric_type: MetricType,
    ) -> Self {
        assert_ne!(
            metric_type,
            MetricType::Cosine,
            "Product quantization does not support cosine, use normalized L2 instead"
        );
        assert_eq!(nbits, 8, "nbits can only be 8");
        Self {
            num_bits: nbits,
            num_sub_vectors: m,
            dimension,
            codebook,
            metric_type,
        }
    }

    pub fn num_centroids(num_bits: u32) -> usize {
        2_usize.pow(num_bits)
    }

    /// Calculate codebook length.
    pub fn codebook_length(num_bits: u32, num_sub_vectors: usize) -> usize {
        Self::num_centroids(num_bits) * num_sub_vectors
    }

    /// Get the centroids for one sub-vector.
    ///
    /// Returns a flatten `num_centroids * sub_vector_width` f32 array.
    pub fn centroids(&self, sub_vector_idx: usize) -> &[T::Native] {
        get_sub_vector_centroids(
            self.codebook.as_slice(),
            self.dimension,
            self.num_bits,
            self.num_sub_vectors,
            sub_vector_idx,
        )
    }

    /// Reconstruct a vector from its PQ code.
    ///
    /// It only supports U8 PQ code for now.
    #[allow(dead_code)]
    pub(crate) fn reconstruct(&self, code: &[u8]) -> Arc<T::ArrayType> {
        assert_eq!(code.len(), self.num_sub_vectors);
        let mut builder = Vec::with_capacity(self.dimension);
        let sub_vector_dim = self.dimension / self.num_sub_vectors;
        for (i, sub_code) in code.iter().enumerate() {
            let centroids = self.centroids(i);
            builder.extend_from_slice(
                &centroids[*sub_code as usize * sub_vector_dim
                    ..(*sub_code as usize + 1) * sub_vector_dim],
            );
        }
        Arc::new(T::ArrayType::from(builder))
    }

    /// Compute the quantization distortion (E).
    ///
    /// Quantization distortion is the difference between the centroids
    /// from the PQ code to the actual vector.
    ///
    /// This method is just for debugging purpose.
    #[allow(dead_code)]
    pub(crate) async fn distortion(
        &self,
        data: &MatrixView<T>,
        distance_type: DistanceType,
    ) -> Result<f64> {
        let sub_vector_width = self.dimension / self.num_sub_vectors;
        let total_distortion = data
            .iter()
            .map(|vector| {
                vector
                    .chunks_exact(sub_vector_width)
                    .enumerate()
                    .map(|(sub_vector_idx, sub_vec)| {
                        let centroids = self.centroids(sub_vector_idx);
                        let distances = match distance_type {
                            DistanceType::L2 => {
                                l2_distance_batch(sub_vec, centroids, sub_vector_width)
                            }
                            DistanceType::Dot => {
                                dot_distance_batch(sub_vec, centroids, sub_vector_width)
                            }
                            _ => {
                                panic!(
                                    "ProductQuantization: distance type {} is not supported",
                                    distance_type
                                );
                            }
                        };
                        argmin_value_float(distances).map(|(_, v)| v).unwrap_or(0.0)
                    })
                    .sum::<f32>() as f64
            })
            .sum::<f64>();
        Ok(total_distortion / data.num_rows() as f64)
    }

    fn build_l2_distance_table(&self, key: &dyn Array) -> Result<Vec<f32>> {
        let key: &T::ArrayType = key.as_any().downcast_ref().ok_or(Error::Index {
            message: format!(
                "Build L2 distance table, type mismatch: {}",
                key.data_type()
            ),
            location: Default::default(),
        })?;
        Ok(build_distance_table_l2(
            self.codebook.as_slice(),
            self.num_bits,
            self.num_sub_vectors,
            key.as_slice(),
        ))
    }

    /// Compute L2 distance from the query to all code.
    ///
    /// Type parameters
    /// ---------------
    /// - C: the tile size of code-book to run at once.
    /// - V: the tile size of PQ code to run at once.
    ///
    /// Parameters
    /// ----------
    /// - distance_table: the pre-computed L2 distance table.
    ///   It is a flatten array of [num_sub_vectors, num_centroids] f32.
    /// - code: the PQ code to be used to compute the distances.
    ///
    /// Returns
    /// -------
    ///  The squared L2 distance.
    #[inline]
    fn compute_l2_distance<const C: usize, const V: usize>(
        &self,
        distance_table: &[f32],
        code: &[u8],
    ) -> Float32Array {
        Float32Array::from(compute_l2_distance::<C, V>(
            distance_table,
            self.num_bits,
            self.num_sub_vectors,
            code,
        ))
    }

    /// Pre-compute L2 distance from the query to all code.
    ///
    /// It returns the squared L2 distance.
    fn l2_distances(&self, key: &dyn Array, code: &UInt8Array) -> Result<Float32Array> {
        let distance_table = self.build_l2_distance_table(key)?;

        #[cfg(target_feature = "avx512f")]
        {
            Ok(self.compute_l2_distance::<16, 64>(&distance_table, code.values()))
        }
        #[cfg(not(target_feature = "avx512f"))]
        {
            Ok(self.compute_l2_distance::<8, 64>(&distance_table, code.values()))
        }
    }

    /// Parameters
    /// ----------
    ///  - query: the query vector, with shape (dimension, )
    ///  - code: the PQ code in one partition.
    ///
    fn dot_distances(&self, key: &dyn Array, code: &UInt8Array) -> Result<Float32Array> {
        let key: &T::ArrayType = key.as_any().downcast_ref().ok_or(Error::Index {
            message: format!(
                "Build Dot distance table, type mismatch: {}",
                key.data_type()
            ),
            location: Default::default(),
        })?;

        // Distance table: `[f32: num_sub_vectors(row) * num_centroids(column)]`.
        let capacity = self.num_sub_vectors * num_centroids(self.num_bits);
        let mut distance_table = Vec::with_capacity(capacity);

        let sub_vector_length = self.dimension / self.num_sub_vectors;
        key.as_slice()
            .chunks_exact(sub_vector_length)
            .enumerate()
            .for_each(|(sub_vec_id, sub_vec)| {
                let subvec_centroids = self.centroids(sub_vec_id);
                let distances = dot_distance_batch(sub_vec, subvec_centroids, sub_vector_length);
                distance_table.extend(distances);
            });

        // Compute distance from the pre-compute table.
        Ok(Float32Array::from_iter_values(
            code.values().chunks_exact(self.num_sub_vectors).map(|c| {
                c.iter()
                    .enumerate()
                    .map(|(sub_vec_idx, centroid)| {
                        distance_table[sub_vec_idx * 256 + *centroid as usize]
                    })
                    .sum::<f32>()
            }),
        ))
    }
}

impl<T: ArrowFloatType + 'static> ProductQuantizer for ProductQuantizerImpl<T>
where
    T::Native: Dot + L2 + Normalize,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn transform(&self, data: &dyn Array) -> Result<ArrayRef> {
        let fsl = data
            .as_fixed_size_list_opt()
            .ok_or(Error::Index {
                message: format!(
                    "Expect to be a FixedSizeList<float> vector array, got: {:?} array",
                    data.data_type()
                ),
                location: location!(),
            })?
            .clone();

        let num_sub_vectors = self.num_sub_vectors;
        let dim = self.dimension;
        let num_bits = self.num_bits;
        let codebook = self.codebook.clone();

        let metric_type = self.metric_type;

        let kmeans = (0..num_sub_vectors)
            .map(|idx| {
                let centroids = get_sub_vector_centroids(
                    codebook.as_slice(),
                    dim,
                    num_bits,
                    num_sub_vectors,
                    idx,
                );
                KMeans::with_centroids(
                    Arc::new(T::ArrayType::from(centroids.to_vec())),
                    dim / num_sub_vectors,
                    metric_type,
                )
            })
            .collect::<Vec<_>>();
        let flatten_data =
            fsl.values()
                .as_any()
                .downcast_ref::<T::ArrayType>()
                .ok_or(Error::Index {
                    message: format!(
                        "Expect to be a float vector array, got: {:?}",
                        fsl.value_type()
                    ),
                    location: location!(),
                })?;

        let values = flatten_data
            .as_slice()
            .par_chunks(dim)
            .map(|sub_vec| {
                sub_vec
                    .chunks_exact(dim / num_sub_vectors)
                    .enumerate()
                    .flat_map(|(sub_idx, sub_vector)| {
                        let kmean: &KMeans<T> = &kmeans[sub_idx];
                        let code = kmean.find_partitions(sub_vector, 1).unwrap();
                        code.values().iter().map(|&v| v as u8)
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect::<Vec<_>>();

        Ok(Arc::new(FixedSizeListArray::try_new_from_values(
            UInt8Array::from(values),
            self.num_sub_vectors as i32,
        )?))
    }

    fn compute_distances(&self, query: &dyn Array, code: &UInt8Array) -> Result<Float32Array> {
        match self.metric_type {
            MetricType::L2 => self.l2_distances(query, code),
            MetricType::Cosine => {
                // L2 over normalized vectors:  ||x - y|| = x^2 + y^2 - 2 * xy = 1 + 1 - 2 * xy = 2 * (1 - xy)
                // Cosine distance: 1 - |xy| / (||x|| * ||y||) = 1 - xy / (x^2 * y^2) = 1 - xy / (1 * 1) = 1 - xy
                // Therefore, Cosine = L2 / 2
                let l2_dists = self.l2_distances(query, code)?;
                Ok(l2_dists.values().iter().map(|v| *v / 2.0).collect())
            }
            MetricType::Dot => self.dot_distances(query, code),
            _ => panic!(
                "ProductQuantization: metric type {} not supported",
                self.metric_type
            ),
        }
    }

    fn num_sub_vectors(&self) -> usize {
        self.num_sub_vectors
    }

    fn num_bits(&self) -> u32 {
        self.num_bits
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn codebook_as_fsl(&self) -> FixedSizeListArray {
        FixedSizeListArray::try_new_from_values(
            self.codebook.as_ref().clone(),
            self.dimension as i32,
        )
        .unwrap()
    }

    fn use_residual(&self) -> bool {
        matches!(self.metric_type, MetricType::L2 | MetricType::Cosine)
    }
}

#[allow(clippy::fallible_impl_from)]
impl TryFrom<&dyn ProductQuantizer> for pb::Pq {
    type Error = Error;

    fn try_from(pq: &dyn ProductQuantizer) -> Result<Self> {
        let fsl = pq.codebook_as_fsl();
        let tensor = pb::Tensor::try_from(&fsl)?;
        Ok(Self {
            num_bits: pq.num_bits(),
            num_sub_vectors: pq.num_sub_vectors() as u32,
            dimension: pq.dimension() as u32,
            codebook: vec![],
            codebook_tensor: Some(tensor),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::iter::repeat;

    use approx::assert_relative_eq;
    use arrow::datatypes::UInt8Type;
    use arrow_array::{
        types::{Float16Type, Float32Type},
        Float16Array,
    };
    use half::f16;
    use lance_testing::datagen::generate_random_array;
    use num_traits::Zero;

    #[test]
    fn test_f16_pq_to_protobuf() {
        let pq = ProductQuantizerImpl::<Float16Type> {
            num_bits: 8,
            num_sub_vectors: 4,
            dimension: 16,
            codebook: Arc::new(Float16Array::from_iter_values(
                repeat(f16::zero()).take(256 * 16),
            )),
            metric_type: MetricType::L2,
        };
        let proto: pb::Pq = pb::Pq::try_from(&pq as &dyn ProductQuantizer).unwrap();
        assert_eq!(proto.num_bits, 8);
        assert_eq!(proto.num_sub_vectors, 4);
        assert_eq!(proto.dimension, 16);
        assert!(proto.codebook.is_empty());
        assert!(proto.codebook_tensor.is_some());

        let tensor = proto.codebook_tensor.as_ref().unwrap();
        assert_eq!(tensor.data_type, pb::tensor::DataType::Float16 as i32);
        assert_eq!(tensor.shape, vec![256, 16]);
    }

    #[tokio::test]
    async fn test_l2_distance() {
        const DIM: usize = 512;
        const TOTAL: usize = 66; // 64 + 2 to make sure reminder is handled correctly.
        let codebook = Arc::new(generate_random_array(256 * DIM));
        let pq = ProductQuantizerImpl::<Float32Type> {
            num_bits: 8,
            num_sub_vectors: 16,
            dimension: DIM,
            codebook: codebook.clone(),
            metric_type: MetricType::L2,
        };
        let pq_code = UInt8Array::from_iter_values((0..16 * TOTAL).map(|v| v as u8));
        let query = generate_random_array(DIM);

        let dists = pq.compute_distances(&query, &pq_code).unwrap();

        let sub_vec_len = DIM / 16;
        let expected = pq_code
            .values()
            .chunks(16)
            .map(|code| {
                code.iter()
                    .enumerate()
                    .flat_map(|(sub_idx, c)| {
                        let subvec_centroids = pq.centroids(sub_idx);
                        let subvec =
                            &query.values()[sub_idx * sub_vec_len..(sub_idx + 1) * sub_vec_len];
                        l2_distance_batch(
                            subvec,
                            &subvec_centroids
                                [*c as usize * sub_vec_len..(*c as usize + 1) * sub_vec_len],
                            sub_vec_len,
                        )
                    })
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        dists
            .values()
            .iter()
            .zip(expected.iter())
            .for_each(|(v, e)| {
                assert_relative_eq!(*v, *e, epsilon = 1e-4);
            });
    }

    #[tokio::test]
    async fn test_pq_transform() {
        const DIM: usize = 16;
        const TOTAL: usize = 64;
        let codebook = generate_random_array(DIM * 256);
        let pq = ProductQuantizerImpl::<Float32Type> {
            num_bits: 8,
            num_sub_vectors: 4,
            dimension: DIM,
            codebook: Arc::new(codebook),
            metric_type: MetricType::L2,
        };

        let vectors = generate_random_array(DIM * TOTAL);
        let fsl = FixedSizeListArray::try_new_from_values(vectors.clone(), DIM as i32).unwrap();
        let pq_code = pq.transform(&fsl).await.unwrap();

        let mut expected = Vec::with_capacity(TOTAL * 4);
        vectors.values().chunks_exact(DIM).for_each(|vec| {
            vec.chunks_exact(DIM / 4)
                .enumerate()
                .for_each(|(sub_idx, sub_vec)| {
                    let centroids = pq.centroids(sub_idx);
                    let dists = l2_distance_batch(sub_vec, centroids, DIM / 4);
                    let code = argmin(dists).unwrap() as u8;
                    expected.push(code);
                });
        });

        assert_eq!(pq_code.len(), TOTAL);
        assert_eq!(
            &expected,
            pq_code
                .as_fixed_size_list()
                .values()
                .as_primitive::<UInt8Type>()
                .values()
        );
    }
}
