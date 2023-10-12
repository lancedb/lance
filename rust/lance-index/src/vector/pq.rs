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
use std::sync::Arc;

use arrow_array::{
    builder::Float32Builder, cast::as_primitive_array, types::Float32Type, Array,
    FixedSizeListArray, Float32Array, UInt8Array,
};
use futures::stream;
use lance_arrow::*;
use lance_core::{Error, Result};
use lance_linalg::kernels::argmin_opt;
use lance_linalg::{distance::MetricType, MatrixView};
use rand::SeedableRng;

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
            sample_rate: 1024,
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
    pub codebook: Option<Arc<Float32Array>>,
}

impl ProductQuantizer {
    /// Build a Product quantizer with `m` sub-vectors, and `nbits` to present centroids.
    pub fn new(m: usize, nbits: u32, dimension: usize) -> Self {
        assert_eq!(nbits, 8, "nbits can only be 8");
        Self {
            num_bits: nbits,
            num_sub_vectors: m,
            dimension,
            codebook: None,
        }
    }

    /// Create a [`ProductQuantizer`] with pre-trained codebook.
    pub fn new_with_codebook(
        m: usize,
        nbits: u32,
        dimension: usize,
        codebook: Arc<Float32Array>,
    ) -> Self {
        assert!(nbits == 8, "nbits can only be 8");
        Self {
            num_bits: nbits,
            num_sub_vectors: m,
            dimension,
            codebook: Some(codebook),
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
    pub fn centroids(&self, sub_vector_idx: usize) -> Option<Arc<Float32Array>> {
        assert!(sub_vector_idx < self.num_sub_vectors);
        self.codebook.as_ref()?;

        let num_centroids = Self::num_centroids(self.num_bits);
        let sub_vector_width = self.dimension / self.num_sub_vectors;
        let codebook = self.codebook.as_ref().unwrap();
        let arr = codebook.slice(
            sub_vector_idx * num_centroids * sub_vector_width,
            num_centroids * sub_vector_width,
        );
        Some(Arc::new(arr))
    }

    /// Reconstruct a vector from its PQ code.
    ///
    /// It only supports U8 PQ code for now.
    pub fn reconstruct(&self, code: &[u8]) -> Arc<Float32Array> {
        assert_eq!(code.len(), self.num_sub_vectors);
        let mut builder = Float32Builder::with_capacity(self.dimension);
        let sub_vector_dim = self.dimension / self.num_sub_vectors;
        for (i, sub_code) in code.iter().enumerate() {
            let centroids = self.centroids(i).unwrap();
            builder.append_slice(
                &centroids.values()[*sub_code as usize * sub_vector_dim
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
        let all_centroids = (0..sub_vectors.len())
            .map(|idx| self.centroids(idx).unwrap())
            .collect::<Vec<_>>();
        let distortion = stream::iter(vectors)
            .zip(stream::iter(all_centroids))
            .map(|(vec, centroid)| async move {
                tokio::task::spawn_blocking(move || {
                    let dist_func = metric_type.batch_func();
                    (0..vec.len())
                        .map(|i| {
                            let value = vec.value(i);
                            let vector: &Float32Array = as_primitive_array(value.as_ref());
                            let distances =
                                dist_func(vector.values(), centroid.values(), vector.len());
                            min(distances.as_ref()).unwrap_or(0.0)
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
    pub async fn transform(
        &self,
        data: &MatrixView<Float32Type>,
        metric_type: MetricType,
    ) -> Result<FixedSizeListArray> {
        let all_centroids = (0..self.num_sub_vectors)
            .map(|idx| self.centroids(idx).unwrap())
            .collect::<Vec<_>>();
        let dist_func = metric_type.batch_func();

        let flatten_data = data.data();
        let num_sub_vectors = self.num_sub_vectors;
        let dim = self.dimension;
        let num_rows = data.num_rows();
        let values = tokio::task::spawn_blocking(move || {
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
                    let centroids = all_centroids[sub_idx].as_ref();
                    // TODO(lei): use kmeans.compute_membership()
                    let code =
                        argmin_opt(dist_func(sub_vector, centroids.values(), sub_dim).iter())
                            .unwrap();
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
    pub async fn train(
        &mut self,
        data: &MatrixView<Float32Type>,
        params: &PQBuildParams,
    ) -> Result<()> {
        assert!(data.num_columns() % self.num_sub_vectors == 0);
        assert_eq!(data.data().null_count(), 0);

        let sub_vectors = divide_to_subvectors(data, self.num_sub_vectors);
        let num_centroids = 2_usize.pow(self.num_bits);
        let dimension = data.num_columns();
        let sub_vector_dimension = dimension / self.num_sub_vectors;

        let mut codebook_builder = Float32Builder::with_capacity(num_centroids * dimension);
        let rng = rand::rngs::SmallRng::from_entropy();

        const REDOS: usize = 1;
        // TODO: parallel training.
        for (i, sub_vec) in sub_vectors.iter().enumerate() {
            // Centroids for one sub vector.
            let values = sub_vec.values();
            let flatten_array: &Float32Array = as_primitive_array(&values);
            let prev_centroids = self.centroids(i);
            let centroids = train_kmeans(
                flatten_array,
                prev_centroids,
                sub_vector_dimension,
                num_centroids,
                params.max_iters as u32,
                REDOS,
                rng.clone(),
                params.metric_type,
                params.sample_rate,
            )
            .await?;
            // TODO: COPIED COPIED COPIED
            unsafe {
                codebook_builder.append_trusted_len_iter(centroids.values().iter().copied());
            }
        }
        let pd_centroids = codebook_builder.finish();
        self.codebook = Some(Arc::new(pd_centroids));

        Ok(())
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
            let code: &UInt8Array = as_primitive_array(code_arr.as_ref());
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
        self.codebook = Some(Arc::new(pd_centroids));

        Ok(())
    }
}
