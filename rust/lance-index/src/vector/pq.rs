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

use std::sync::Arc;

use arrow_array::{
    builder::Float32Builder, cast::AsArray, types::Float32Type, Array, FixedSizeListArray,
    Float32Array, UInt8Array,
};
use futures::{stream, stream::repeat_with, StreamExt, TryStreamExt};
use lance_arrow::*;
use lance_core::{Error, Result};
use lance_linalg::distance::{cosine_distance_batch, dot_distance_batch, l2_distance_batch};
use lance_linalg::kernels::argmin;
use lance_linalg::{distance::MetricType, MatrixView};
use rand::SeedableRng;
pub mod transform;

use super::kmeans::train_kmeans;
use super::pb;

/// Parameters for building product quantization.
#[derive(Debug, Clone)]
pub struct PQBuildParams {
    /// Number of subvectors to build PQ code
    pub num_sub_vectors: usize,

    /// The number of bits to present one PQ centroid.
    pub num_bits: usize,

    /// Metric type, L2 or Cosine.
    pub metric_type: MetricType,

    /// Train as optimized product quantization.
    pub use_opq: bool,

    /// The max number of iterations for kmeans training.
    pub max_iters: usize,

    /// Max number of iterations to train OPQ, if `use_opq` is true.
    pub max_opq_iters: usize,

    /// User provided codebook.
    pub codebook: Option<Arc<Float32Array>>,

    pub sample_rate: usize,
}

impl Default for PQBuildParams {
    fn default() -> Self {
        Self {
            num_sub_vectors: 16,
            num_bits: 8,
            metric_type: MetricType::L2,
            use_opq: false,
            max_iters: 50,
            max_opq_iters: 50,
            codebook: None,
            sample_rate: 256,
        }
    }
}

impl PQBuildParams {
    pub fn new(num_sub_vectors: usize, num_bits: usize) -> Self {
        Self {
            num_sub_vectors,
            num_bits,
            ..Default::default()
        }
    }

    pub fn with_codebook(
        num_sub_vectors: usize,
        num_bits: usize,
        codebook: Arc<Float32Array>,
    ) -> Self {
        Self {
            num_sub_vectors,
            num_bits,
            codebook: Some(codebook),
            ..Default::default()
        }
    }
}

/// Divide a 2D vector in [`FixedSizeListArray`] to `m` sub-vectors.
///
/// For example, for a `[1024x1M]` matrix, when `n = 8`, this function divides
/// the matrix into  `[128x1M; 8]` vector of matrix.
fn divide_to_subvectors(data: &MatrixView<Float32Type>, m: usize) -> Vec<Arc<FixedSizeListArray>> {
    assert!(!data.num_rows() > 0);

    let sub_vector_length = data.num_columns() / m;
    let capacity = data.num_rows() * sub_vector_length;
    let mut subarrays = vec![];

    // TODO: very intensive memory copy involved!!! But this is on the write path.
    // Optimize for memory copy later.
    for i in 0..m {
        let mut builder = Float32Builder::with_capacity(capacity);
        for j in 0..data.num_rows() {
            let row = data.row(j).unwrap();
            let start = i * sub_vector_length;
            builder.append_slice(&row[start..start + sub_vector_length]);
        }
        let values = builder.finish();
        let sub_array = Arc::new(
            FixedSizeListArray::try_new_from_values(values, sub_vector_length as i32).unwrap(),
        );
        subarrays.push(sub_array);
    }
    subarrays
}

/// Product Quantization, optimized for [Apache Arrow] buffer memory layout.
///
#[derive(Debug)]
pub struct ProductQuantizer {
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
    pub codebook: Arc<Float32Array>,
}

fn get_sub_vector_centroids(
    codebook: &Float32Array,
    dimension: usize,
    num_bits: impl Into<u32>,
    num_sub_vectors: usize,
    sub_vector_idx: usize,
) -> &[f32] {
    assert!(sub_vector_idx < num_sub_vectors);

    let num_centroids = ProductQuantizer::num_centroids(num_bits.into());
    let sub_vector_width = dimension / num_sub_vectors;
    &codebook.as_slice()[sub_vector_idx * num_centroids * sub_vector_width
        ..(sub_vector_idx + 1) * num_centroids * sub_vector_width]
}

impl ProductQuantizer {
    /// Create a [`ProductQuantizer`] with pre-trained codebook.
    pub fn new(
        m: usize,
        nbits: u32,
        dimension: usize,
        codebook: Arc<Float32Array>,
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
    pub fn centroids(&self, sub_vector_idx: usize) -> &[f32] {
        get_sub_vector_centroids(
            self.codebook.as_ref(),
            self.dimension,
            self.num_bits,
            self.num_sub_vectors,
            sub_vector_idx,
        )
    }

    /// Reconstruct a vector from its PQ code.
    ///
    /// It only supports U8 PQ code for now.
    pub fn reconstruct(&self, code: &[u8]) -> Arc<Float32Array> {
        assert_eq!(code.len(), self.num_sub_vectors);
        let mut builder = Float32Builder::with_capacity(self.dimension);
        let sub_vector_dim = self.dimension / self.num_sub_vectors;
        for (i, sub_code) in code.iter().enumerate() {
            let centroids = self.centroids(i);
            builder.append_slice(
                &centroids[*sub_code as usize * sub_vector_dim
                    ..(*sub_code as usize + 1) * sub_vector_dim],
            );
        }
        Arc::new(builder.finish())
    }

    /// Compute the quantization distortion (E).
    ///
    /// Quantization distortion is the difference between the centroids
    /// from the PQ code to the actual vector.
    pub async fn distortion(
        &self,
        data: &MatrixView<Float32Type>,
        metric_type: MetricType,
    ) -> Result<f64> {
        let sub_vectors = divide_to_subvectors(data, self.num_sub_vectors);
        debug_assert_eq!(sub_vectors.len(), self.num_sub_vectors);

        let vectors = sub_vectors.to_vec();
        let codebook = self.codebook.clone();
        let dimension = self.dimension;
        let num_bits = self.num_bits;
        let num_sub_vectors = self.num_sub_vectors;
        let distortion = stream::iter(vectors)
            .zip(repeat_with(|| codebook.clone()))
            .enumerate()
            .map(|(sub_idx, (vec, codebook))| async move {
                let cb = codebook.clone();
                tokio::task::spawn_blocking(move || {
                    let centroid = get_sub_vector_centroids(
                        cb.as_ref(),
                        dimension,
                        num_bits,
                        num_sub_vectors,
                        sub_idx,
                    );
                    (0..vec.len())
                        .map(|i| {
                            let value = vec.value(i);
                            let vector: &Float32Array = value.as_primitive();
                            let distances = match metric_type {
                                lance_linalg::distance::DistanceType::L2 => {
                                    l2_distance_batch(vector.values(), centroid, dimension)
                                }
                                lance_linalg::distance::DistanceType::Cosine => {
                                    cosine_distance_batch(vector.values(), centroid, dimension)
                                }
                                lance_linalg::distance::DistanceType::Dot => {
                                    dot_distance_batch(vector.values(), centroid, dimension)
                                }
                            };
                            distances.fold(f32::INFINITY, |a, b| a.min(b))
                        })
                        .sum::<f32>() as f64 // in case of overflow
                })
                .await
            })
            .buffered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await?
            .iter()
            .sum::<f64>();
        Ok(distortion / data.num_rows() as f64)
    }

    /// Transform the vector array to PQ code array.
    pub async fn transform(&self, data: &MatrixView<Float32Type>) -> Result<FixedSizeListArray> {
        let flatten_data = data.data();
        let num_sub_vectors = self.num_sub_vectors;
        let dim = self.dimension;
        let num_rows = data.num_rows();
        let num_bits = self.num_bits;
        let codebook = self.codebook.clone();

        let metric_type = self.metric_type;
        let values = tokio::task::spawn_blocking(move || {
            let all_centroids = (0..num_sub_vectors)
                .map(|idx| {
                    get_sub_vector_centroids(codebook.as_ref(), dim, num_bits, num_sub_vectors, idx)
                })
                .collect::<Vec<_>>();
            let flatten_values = flatten_data.values();
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

        Ok(FixedSizeListArray::try_new_from_values(
            values,
            self.num_sub_vectors as i32,
        )?)
    }

    /// Train [`ProductQuantizer`] using vectors.
    pub async fn train(data: &MatrixView<Float32Type>, params: &PQBuildParams) -> Result<Self> {
        if data.num_columns() % params.num_sub_vectors != 0 {
            return Err(Error::Index {
                message: format!(
                    "Dimension {} cannot be divided by {}",
                    data.num_columns(),
                    params.num_sub_vectors
                ),
            });
        }
        assert_eq!(data.data().null_count(), 0);

        let sub_vectors = divide_to_subvectors(data, params.num_sub_vectors);
        let num_centroids = 2_usize.pow(params.num_bits as u32);
        let dimension = data.num_columns();
        let sub_vector_dimension = dimension / params.num_sub_vectors;

        let mut codebook_builder = Float32Builder::with_capacity(num_centroids * dimension);

        const REDOS: usize = 1;
        // TODO: parallel training.
        let d = stream::iter(sub_vectors.into_iter())
            .map(|sub_vec| async move {
                let rng = rand::rngs::SmallRng::from_entropy();

                // Centroids for one sub vector.
                let sub_vec = sub_vec.clone();
                let flatten_array: &Float32Array = sub_vec.values().as_primitive();
                let centroids = train_kmeans(
                    flatten_array,
                    None,
                    sub_vector_dimension,
                    num_centroids,
                    params.max_iters as u32,
                    REDOS,
                    rng.clone(),
                    params.metric_type,
                    params.sample_rate,
                )
                .await;
                centroids
            })
            .buffered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await?;

        for centroid in d.iter() {
            unsafe {
                codebook_builder.append_trusted_len_iter(centroid.values().iter().copied());
            }
        }

        let pd_centroids = codebook_builder.finish();

        Ok(Self::new(
            params.num_sub_vectors,
            params.num_bits as u32,
            dimension,
            Arc::new(pd_centroids),
            params.metric_type,
        ))
    }

    /// Reset the centroids from the OPQ training.
    pub fn reset_centroids(
        &mut self,
        data: &MatrixView<Float32Type>,
        pq_code: &FixedSizeListArray,
    ) -> Result<()> {
        assert_eq!(data.num_rows(), pq_code.len());

        let num_centroids = 2_usize.pow(self.num_bits);
        let mut builder = Float32Builder::with_capacity(num_centroids * self.dimension);
        let sub_vector_dim = self.dimension / self.num_sub_vectors;
        let mut sum = vec![0.0_f32; self.dimension * num_centroids];
        // Counts of each subvector x centroids.
        // counts[sub_vector][centroid]
        let mut counts = vec![0; self.num_sub_vectors * num_centroids];

        let sum_stride = sub_vector_dim * num_centroids;

        for i in 0..data.num_rows() {
            let code_arr = pq_code.value(i);
            let code: &UInt8Array = code_arr.as_primitive();
            for sub_vec_id in 0..code.len() {
                let centroid = code.value(sub_vec_id) as usize;
                let sub_vector: Float32Array = data.data().slice(
                    i * self.dimension + sub_vec_id * sub_vector_dim,
                    sub_vector_dim,
                );
                counts[sub_vec_id * num_centroids + centroid] += 1;
                for k in 0..sub_vector.len() {
                    sum[sub_vec_id * sum_stride + centroid * sub_vector_dim + k] +=
                        sub_vector.value(k);
                }
            }
        }
        for (i, cnt) in counts.iter().enumerate() {
            if *cnt > 0 {
                let s = sum[i * sub_vector_dim..(i + 1) * sub_vector_dim].as_mut();
                for v in s.iter_mut() {
                    *v /= *cnt as f32;
                }
                builder.append_slice(s);
            } else {
                builder.append_slice(vec![f32::MAX; sub_vector_dim].as_slice());
            }
        }

        let pd_centroids = builder.finish();
        self.codebook = Arc::new(pd_centroids);

        Ok(())
    }
}

#[allow(clippy::fallible_impl_from)]
impl From<&ProductQuantizer> for pb::Pq {
    fn from(pq: &ProductQuantizer) -> Self {
        Self {
            num_bits: pq.num_bits,
            num_sub_vectors: pq.num_sub_vectors as u32,
            dimension: pq.dimension as u32,
            codebook: pq.codebook.values().to_vec(),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use approx::relative_eq;

    #[test]
    fn test_divide_to_subvectors() {
        let values = Float32Array::from_iter((0..320).map(|v| v as f32));
        // A [10, 32] array.
        let mat = MatrixView::new(values.into(), 32);
        let sub_vectors = divide_to_subvectors(&mat, 4);
        assert_eq!(sub_vectors.len(), 4);
        assert_eq!(sub_vectors[0].len(), 10);
        assert_eq!(sub_vectors[0].value_length(), 8);

        assert_eq!(
            sub_vectors[0].as_ref(),
            &FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                (0..10)
                    .map(|i| {
                        Some(
                            (i * 32..i * 32 + 8)
                                .map(|v| Some(v as f32))
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>(),
                8
            )
        );
    }

    #[ignore]
    #[tokio::test]
    async fn test_train_pq_iteratively() {
        let mut params = PQBuildParams::new(2, 8);
        params.max_iters = 1;

        let values = Float32Array::from_iter((0..16000).map(|v| v as f32));
        // A 16-dim array.
        let dim = 16;
        let mat = MatrixView::new(values.into(), dim);
        let pq = ProductQuantizer::train(&mat, &params).await.unwrap();

        // Init centroids
        let centroids = pq.codebook.clone();

        // Keep training 10 times
        let mut actual_pq = ProductQuantizer {
            num_bits: 8,
            num_sub_vectors: 2,
            dimension: dim,
            codebook: centroids,
            metric_type: MetricType::L2,
        };
        // Iteratively train for 10 times.
        for _ in 0..10 {
            let code = actual_pq.transform(&mat).await.unwrap();
            actual_pq.reset_centroids(&mat, &code).unwrap();
            params.codebook = Some(actual_pq.codebook.clone());
            actual_pq = ProductQuantizer::train(&mat, &params).await.unwrap();
        }

        let result = pq.codebook;
        let expected = actual_pq.codebook;
        result
            .values()
            .iter()
            .zip(expected.values())
            .for_each(|(&r, &e)| {
                assert!(relative_eq!(r, e));
            });
    }
}
