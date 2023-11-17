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
    cosine_distance_batch, dot_distance_batch, l2_distance_batch, norm_l2, Cosine, Dot, L2,
};
use lance_linalg::kernels::{argmin, argmin_value_float};
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
    async fn transform(&self, data: &dyn Array) -> Result<ArrayRef>;

    /// Build the distance lookup in `f32`.
    fn build_distance_table(&self, query: &dyn Array, code: &UInt8Array) -> Result<ArrayRef>;

    /// Get the centroids for one sub-vector.
    fn num_bits(&self) -> u32;

    /// Number of sub-vectors
    fn num_sub_vectors(&self) -> usize;

    fn dimension(&self) -> usize;

    // TODO: move to pub(crate) once the refactor of lance::index to lance-index is done.
    fn codebook_as_fsl(&self) -> FixedSizeListArray;
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
                        argmin_value_float(distances).1
                    })
                    .sum::<f32>() as f64
            })
            .sum::<f64>();
        Ok(total_distortion / data.num_rows() as f64)
    }

    fn l2_distance_table(&self, key: &dyn Array, code: &UInt8Array) -> Result<ArrayRef> {
        let key: &T::ArrayType = key.as_any().downcast_ref().ok_or(Error::Index {
            message: format!(
                "Build L2 distance table, type mismatch: {}",
                key.data_type()
            ),
            location: Default::default(),
        })?;

        // Build distance table for each sub-centroid to the query key.
        //
        // Distance table: `[T::Native: num_sub_vectors(row) * num_centroids(column)]`.
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

        Ok(Arc::new(Float32Array::from_iter_values(
            code.values().chunks_exact(self.num_sub_vectors).map(|c| {
                c.iter()
                    .enumerate()
                    .map(|(sub_vec_idx, centroid)| {
                        distance_table[sub_vec_idx * 256 + *centroid as usize]
                    })
                    .sum()
            }),
        )))
    }

    /// Pre-compute dot product to each sub-centroids.
    /// Parameters
    ///  - query: the query vector, with shape (dimension, )
    ///  - code: the PQ code in one partition.
    ///
    fn dot_distance_table(&self, key: &dyn Array, code: &UInt8Array) -> Result<ArrayRef> {
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
        Ok(Arc::new(Float32Array::from_iter_values(
            code.values().chunks_exact(self.num_sub_vectors).map(|c| {
                c.iter()
                    .enumerate()
                    .map(|(sub_vec_idx, centroid)| {
                        distance_table[sub_vec_idx * 256 + *centroid as usize]
                    })
                    .sum::<f32>()
            }),
        )))
    }

    /// Pre-compute cosine distance to each sub-centroids.
    ///
    /// Parameters
    ///  - query: the query vector, with shape (dimension, )
    ///  - code: the PQ code in one partition.
    ///
    fn cosine_distances(&self, key: &dyn Array, code: &UInt8Array) -> Result<ArrayRef> {
        let query: &T::ArrayType = key.as_any().downcast_ref().ok_or(Error::Index {
            message: format!(
                "Build Dot distance table, type mismatch: {}",
                key.data_type()
            ),
            location: Default::default(),
        })?;

        // Build two tables for cosine distance.
        //
        // xy table: `[f32: num_sub_vectors(row) * num_centroids(column)]`.
        // y_norm table: `[f32: num_sub_vectors(row) * num_centroids(column)]`.
        let num_centroids = num_centroids(self.num_bits);
        let mut xy_table: Vec<f32> = Vec::with_capacity(self.num_sub_vectors * num_centroids);
        let mut y2_table: Vec<f32> = Vec::with_capacity(self.num_sub_vectors * num_centroids);

        let x_norm = norm_l2(query.as_slice());
        let sub_vector_length = self.dimension / self.num_sub_vectors;
        query
            .as_slice()
            .chunks_exact(sub_vector_length)
            .enumerate()
            .for_each(|(i, sub_vector)| {
                let sub_vector_centroids = self.centroids(i);
                xy_table.extend(dot_distance_batch(
                    sub_vector,
                    sub_vector_centroids,
                    sub_vector_length,
                ));
                y2_table.extend(
                    sub_vector_centroids
                        .chunks_exact(sub_vector_length)
                        .map(|cent| norm_l2(cent).powi(2)),
                );
            });

        // Compute distance from the pre-compute table.
        Ok(Arc::new(Float32Array::from_iter_values(
            code.values().chunks_exact(self.num_sub_vectors).map(|c| {
                let xy = c
                    .iter()
                    .enumerate()
                    .map(|(sub_vec_idx, centroid)| {
                        let idx = sub_vec_idx * num_centroids + *centroid as usize;
                        xy_table[idx]
                    })
                    .sum::<f32>();
                let y2 = c
                    .iter()
                    .enumerate()
                    .map(|(sub_vec_idx, centroid)| {
                        let idx = sub_vec_idx * num_centroids + *centroid as usize;
                        y2_table[idx]
                    })
                    .sum::<f32>();
                1.0 - xy / (x_norm * y2.sqrt())
            }),
        )))
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
                        lance_linalg::distance::DistanceType::L2 => {
                            l2_distance_batch(sub_vector, centroids, sub_dim)
                        }
                        lance_linalg::distance::DistanceType::Cosine => {
                            cosine_distance_batch(sub_vector, centroids, sub_dim)
                        }
                        lance_linalg::distance::DistanceType::Dot => {
                            dot_distance_batch(sub_vector, centroids, sub_dim)
                        }
                    };
                    let code = argmin(dist_iter).unwrap();
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

    fn build_distance_table(&self, query: &dyn Array, code: &UInt8Array) -> Result<ArrayRef> {
        match self.metric_type {
            MetricType::Cosine => self.cosine_distances(query, code),
            MetricType::Dot => self.dot_distance_table(query, code),
            MetricType::L2 => self.l2_distance_table(query, code),
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

    use arrow_array::{types::Float16Type, Float16Array};
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
        let proto: pb::Pq = pb::Pq::try_from(&pq as &dyn ProductQuantizerExt).unwrap();
        assert_eq!(proto.num_bits, 8);
        assert_eq!(proto.num_sub_vectors, 4);
        assert_eq!(proto.dimension, 16);
        assert!(proto.codebook.is_empty());
        assert!(proto.codebook_tensor.is_some());

        let tensor = proto.codebook_tensor.as_ref().unwrap();
        assert_eq!(tensor.data_type, pb::tensor::DataType::Float16 as i32);
        assert_eq!(tensor.shape, vec![256, 16]);
    }
}
