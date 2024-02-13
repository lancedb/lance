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

//! Product Quantization
//!

use std::any::Any;
use std::sync::Arc;

use arrow_array::{cast::AsArray, Array, FixedSizeListArray, UInt8Array};
use arrow_array::{ArrayRef, Float32Array};
use async_trait::async_trait;
use lance_arrow::floats::FloatArray;
use lance_arrow::*;
use lance_core::{Error, Result};
use lance_linalg::distance::{
    cosine_distance_batch, dot_distance_batch, l2_distance_batch, Cosine, Dot, L2,
};
use lance_linalg::kernels::{argmin, argmin_value_float, normalize};
use lance_linalg::{distance::MetricType, MatrixView};
use snafu::{location, Location};
pub mod builder;
pub mod transform;
pub(crate) mod utils;

pub use self::utils::num_centroids;
use super::pb;
pub use builder::PQBuildParams;

/// Product Quantization

#[async_trait::async_trait]
pub trait ProductQuantizer: Send + Sync + std::fmt::Debug {
    fn as_any(&self) -> &dyn Any;

    /// Transform a vector column to PQ code column.
    ///
    /// Parameters
    /// ----------
    /// *data*: vector array, must be a `FixedSizeListArray`
    ///
    /// Returns
    /// -------
    ///   PQ code column
    async fn transform(&self, data: &dyn Array) -> Result<ArrayRef>;

    /// Build the distance lookup in `f32`.
    fn build_distance_table(&self, query: &dyn Array, code: &UInt8Array) -> Result<Float32Array>;

    /// Get the centroids for one sub-vector.
    fn num_bits(&self) -> u32;

    /// Number of sub-vectors
    fn num_sub_vectors(&self) -> usize;

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
#[derive(Debug)]
pub struct ProductQuantizerImpl<T: ArrowFloatType + Cosine + Dot + L2> {
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

fn get_sub_vector_centroids<T: FloatToArrayType>(
    codebook: &[T],
    dimension: usize,
    num_bits: impl Into<u32>,
    num_sub_vectors: usize,
    sub_vector_idx: usize,
) -> &[T] {
    assert!(sub_vector_idx < num_sub_vectors);

    let num_centroids = num_centroids(num_bits);
    let sub_vector_width = dimension / num_sub_vectors;
    &codebook[sub_vector_idx * num_centroids * sub_vector_width
        ..(sub_vector_idx + 1) * num_centroids * sub_vector_width]
}

impl<T: ArrowFloatType + Cosine + Dot + L2> ProductQuantizerImpl<T> {
    /// Create a [`ProductQuantizer`] with pre-trained codebook.
    pub fn new(
        m: usize,
        nbits: u32,
        dimension: usize,
        codebook: Arc<T::ArrayType>,
        metric_type: MetricType,
    ) -> Self {
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
        metric_type: MetricType,
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
                        let distances = match metric_type {
                            lance_linalg::distance::DistanceType::L2 => {
                                l2_distance_batch(sub_vec, centroids, sub_vector_width)
                            }
                            lance_linalg::distance::DistanceType::Cosine => {
                                cosine_distance_batch(sub_vec, centroids, sub_vector_width)
                            }
                            lance_linalg::distance::DistanceType::Dot => {
                                dot_distance_batch(sub_vec, centroids, sub_vector_width)
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

        let mut distance_table = vec![];

        let sub_vector_length = self.dimension / self.num_sub_vectors;
        key.as_slice()
            .chunks_exact(sub_vector_length)
            .enumerate()
            .for_each(|(i, sub_vec)| {
                let subvec_centroids = self.centroids(i);
                let distances = l2_distance_batch(sub_vec, subvec_centroids, sub_vector_length);
                distance_table.extend(distances);
            });
        Ok(distance_table)
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
        let num_centroids = num_centroids(self.num_bits);

        let iter = code.chunks_exact(self.num_sub_vectors * V);
        let distances = iter.clone().flat_map(|c| {
            let mut sums = [0.0_f32; V];
            for i in (0..self.num_sub_vectors).step_by(C) {
                for (vec_idx, sum) in sums.iter_mut().enumerate() {
                    let vec_start = vec_idx * self.num_sub_vectors;
                    #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
                    {
                        use std::arch::x86_64::*;
                        let mut offsets = [(i * num_centroids) as i32; C];
                        for j in 0..C {
                            offsets[j] += c[vec_start + j] as i32;
                        }
                        unsafe {
                            let simd_offsets = _mm512_loadu_epi32(offsets.as_ptr());
                            let v = _mm512_i32gather_ps(
                                simd_offsets,
                                distance_table.as_ptr() as *const u8,
                                4,
                            );
                            *sum += _mm512_reduce_add_ps(v);
                        }
                    }
                    #[cfg(not(all(feature = "nightly", target_feature = "avx512f")))]
                    {
                        let s = c[vec_start..]
                            .iter()
                            .take(C)
                            .enumerate()
                            .map(|(k, c)| distance_table[(i + k) * 256 + *c as usize])
                            .sum::<f32>();
                        *sum += s;
                    }
                }
            }
            sums.into_iter()
        });
        // Remainder
        let remainder = iter.remainder().chunks(self.num_sub_vectors).map(|c| {
            c.iter()
                .enumerate()
                .map(|(sub_vec_idx, code)| {
                    distance_table[sub_vec_idx * num_centroids + *code as usize]
                })
                .sum::<f32>()
        });
        Float32Array::from_iter_values(distances.chain(remainder))
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

    /// Pre-compute dot product to each sub-centroids.
    /// Parameters
    ///  - query: the query vector, with shape (dimension, )
    ///  - code: the PQ code in one partition.
    ///
    fn dot_distance_table(&self, key: &dyn Array, code: &UInt8Array) -> Result<Float32Array> {
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

#[async_trait]
impl<T: ArrowFloatType + Cosine + Dot + L2 + 'static> ProductQuantizer for ProductQuantizerImpl<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn transform(&self, data: &dyn Array) -> Result<ArrayRef> {
        let fsl = data
            .as_fixed_size_list_opt()
            .ok_or(Error::Index {
                message: format!(
                    "Expect to be a float vector array, got: {:?}",
                    data.data_type()
                ),
                location: location!(),
            })?
            .clone();

        let fsl = if self.metric_type == MetricType::Cosine {
            // Normalize cosine vectors to unit length.
            let values = fsl
                .values()
                .as_any()
                .downcast_ref::<T::ArrayType>()
                .ok_or(Error::Index {
                    message: format!(
                        "Expect to be a float vector array, got: {:?}",
                        fsl.value_type()
                    ),
                    location: location!(),
                })?
                .as_slice()
                .chunks(self.dimension)
                .flat_map(normalize)
                .collect::<Vec<_>>();
            let data = T::ArrayType::from(values);
            FixedSizeListArray::try_new_from_values(data, self.dimension as i32)?
        } else {
            fsl
        };

        let num_sub_vectors = self.num_sub_vectors;
        let dim = self.dimension;
        let num_rows = fsl.len();
        let num_bits = self.num_bits;
        let codebook = self.codebook.clone();

        let metric_type = self.metric_type;
        let values = tokio::task::spawn_blocking(move || {
            let all_centroids = (0..num_sub_vectors)
                .map(|idx| {
                    get_sub_vector_centroids(
                        codebook.as_slice(),
                        dim,
                        num_bits,
                        num_sub_vectors,
                        idx,
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

            let flatten_values = flatten_data.as_slice();
            let capacity = num_sub_vectors * num_rows;
            let mut builder: Vec<u8> = vec![0; capacity];
            // Dimension of each sub-vector.
            let sub_dim = dim / num_sub_vectors;
            for i in 0..num_rows {
                let row_offset = i * dim;

                for sub_idx in 0..num_sub_vectors {
                    let offset = row_offset + sub_idx * sub_dim;
                    let sub_vector = &flatten_values[offset..offset + sub_dim];
                    let centroids = all_centroids[sub_idx];

                    let dist_iter = match metric_type {
                        MetricType::L2 | MetricType::Cosine => {
                            l2_distance_batch(sub_vector, centroids, sub_dim)
                        }
                        MetricType::Dot => dot_distance_batch(sub_vector, centroids, sub_dim),
                    };
                    let code = argmin(dist_iter).ok_or(Error::Index {
                        message: format!(
                            "Failed to assign PQ code: {}, sub-vector={:#?}",
                            "it is likely that distance is NaN or Inf", sub_vector
                        ),
                        location: location!(),
                    })? as u8;
                    builder[i * num_sub_vectors + sub_idx] = code as u8;
                }
            }
            Ok::<UInt8Array, Error>(UInt8Array::from(builder))
        })
        .await??;

        Ok(Arc::new(FixedSizeListArray::try_new_from_values(
            values,
            self.num_sub_vectors as i32,
        )?))
    }

    fn build_distance_table(&self, query: &dyn Array, code: &UInt8Array) -> Result<Float32Array> {
        match self.metric_type {
            MetricType::L2 => self.l2_distances(query, code),
            MetricType::Cosine => {
                let query: &T::ArrayType = query.as_any().downcast_ref().ok_or(Error::Index {
                    message: format!(
                        "Build cosine distance table, type mismatch: {}",
                        query.data_type()
                    ),
                    location: Default::default(),
                })?;

                // Normalized query vector.
                let query = T::ArrayType::from(normalize(query.as_slice()).collect::<Vec<_>>());
                // L2 over normalized vectors:  ||x - y|| = x^2 + y^2 - 2 * xy = 1 + 1 - 2 * xy = 2 * (1 - xy)
                // Cosine distance: 1 - |xy| / (||x|| * ||y||) = 1 - xy / (x^2 * y^2) = 1 - xy / (1 * 1) = 1 - xy
                // Therefore, Cosine = L2 / 2
                let l2_dists = self.l2_distances(&query, code)?;
                Ok(l2_dists.values().iter().map(|v| *v / 2.0).collect())
            }
            MetricType::Dot => self.dot_distance_table(query, code),
        }
    }

    fn num_bits(&self) -> u32 {
        self.num_bits
    }

    fn num_sub_vectors(&self) -> usize {
        self.num_sub_vectors
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
        self.metric_type != MetricType::Cosine
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

    use arrow_array::{
        types::{Float16Type, Float32Type},
        Float16Array, Float32Array,
    };
    use half::f16;
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

    #[test]
    fn test_cosine_pq_does_not_use_residual() {
        let pq = ProductQuantizerImpl::<Float32Type> {
            num_bits: 8,
            num_sub_vectors: 4,
            dimension: 16,
            codebook: Arc::new(Float32Array::from_iter_values(repeat(0.0).take(128))),
            metric_type: MetricType::Cosine,
        };
        assert!(!pq.use_residual());

        let pq = ProductQuantizerImpl::<Float32Type> {
            num_bits: 8,
            num_sub_vectors: 4,
            dimension: 16,
            codebook: Arc::new(Float32Array::from_iter_values(repeat(0.0).take(128))),
            metric_type: MetricType::L2,
        };
        assert!(pq.use_residual());
    }

    #[tokio::test]
    async fn test_empty_dist_iter() {
        let pq = ProductQuantizerImpl::<Float32Type> {
            num_bits: 8,
            num_sub_vectors: 4,
            dimension: 16,
            codebook: Arc::new(Float32Array::from_iter_values(
                (0..256 * 16).map(|v| v as f32),
            )),
            metric_type: MetricType::Cosine,
        };

        let data = Float32Array::from_iter_values(repeat(0.0).take(16));
        let data = FixedSizeListArray::try_new_from_values(data, 16).unwrap();
        let rst = pq.transform(&data).await;
        assert!(rst.is_err());
        assert!(rst
            .unwrap_err()
            .to_string()
            .contains("it is likely that distance is NaN"));
    }
}
