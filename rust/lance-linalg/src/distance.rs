// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Distance metrics
//!
//! This module provides distance metrics for vectors.
//!
//! - `bf16, f16, f32, f64` types are supported.
//! - SIMD is used when available, on `x86_64`, `aarch64` and `loongarch64`
//!   architectures.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float16Type, Float32Type, Float64Type, UInt8Type};
use arrow_array::{Array, ArrowPrimitiveType, FixedSizeListArray, Float32Array, ListArray};
use arrow_schema::{ArrowError, DataType};

pub mod cosine;
pub mod cosine_u8;
pub mod dot;
pub mod dot_u8;
pub mod hamming;
pub mod l2;
pub mod l2_u8;
pub mod norm_l2;

/// Number of distances computed per call into a runtime-selected batch kernel.
///
/// Keeping a small output buffer amortizes the `#[target_feature]` call while
/// avoiding the allocation and full-batch materialization that dominate the
/// common dimension-8 case.
#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx2", target_feature = "fma"))
))]
const BATCH_BUFFER_SIZE: usize = 64;

#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx2", target_feature = "fma"))
))]
pub(crate) type BatchKernel = unsafe fn(&[f32], &[f32], usize, &mut [f32]);

/// Runtime-selected target-feature tier for a batch kernel.
#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx2", target_feature = "fma"))
))]
#[derive(Clone, Copy)]
pub(crate) enum BatchKind {
    Scalar,
    Avx,
    AvxFma,
    Avx512,
}

/// Per-vector operations used when a batch consumer can be folded directly.
#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx2", target_feature = "fma"))
))]
pub(crate) trait BatchOperation {
    fn fold_scalar<B, F>(key: &[f32], batch: &[f32], dimension: usize, init: B, f: F) -> B
    where
        F: FnMut(B, f32) -> B;

    unsafe fn fold_avx<B, F>(key: &[f32], batch: &[f32], dimension: usize, init: B, f: F) -> B
    where
        F: FnMut(B, f32) -> B;

    unsafe fn fold_avx_fma<B, F>(key: &[f32], batch: &[f32], dimension: usize, init: B, f: F) -> B
    where
        F: FnMut(B, f32) -> B;

    unsafe fn fold_avx512<B, F>(key: &[f32], batch: &[f32], dimension: usize, init: B, f: F) -> B
    where
        F: FnMut(B, f32) -> B;
}

/// Allocation-free iterator over a runtime-selected f32 distance kernel.
#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx2", target_feature = "fma"))
))]
pub(crate) struct BatchIter<'a, O> {
    key: &'a [f32],
    batch: &'a [f32],
    dimension: usize,
    kernel: BatchKernel,
    kind: BatchKind,
    buffer: [f32; BATCH_BUFFER_SIZE],
    buffer_index: usize,
    buffer_len: usize,
    operation: std::marker::PhantomData<O>,
}

#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx2", target_feature = "fma"))
))]
impl<'a, O> BatchIter<'a, O> {
    /// Creates an iterator after the caller has verified `kernel`'s CPU feature
    /// requirements.
    ///
    /// # Safety
    /// The host must support every target feature required by `kernel`.
    #[inline]
    pub(crate) unsafe fn new(
        key: &'a [f32],
        batch: &'a [f32],
        dimension: usize,
        kernel: BatchKernel,
        kind: BatchKind,
    ) -> Self {
        // Match `chunks_exact` validation before the buffered iterator performs
        // division by the dimension.
        let _ = batch.chunks_exact(dimension);
        Self {
            key,
            batch,
            dimension,
            kernel,
            kind,
            buffer: [0.0; BATCH_BUFFER_SIZE],
            buffer_index: 0,
            buffer_len: 0,
            operation: std::marker::PhantomData,
        }
    }

    #[inline]
    fn refill(&mut self) -> bool {
        let num_vectors = (self.batch.len() / self.dimension).min(BATCH_BUFFER_SIZE);
        if num_vectors == 0 {
            return false;
        }

        let num_values = num_vectors * self.dimension;
        let (input, remaining) = self.batch.split_at(num_values);
        // SAFETY: `new` requires the caller to verify the selected kernel's
        // target features before constructing the iterator.
        unsafe {
            (self.kernel)(
                self.key,
                input,
                self.dimension,
                &mut self.buffer[..num_vectors],
            );
        }
        self.batch = remaining;
        self.buffer_index = 0;
        self.buffer_len = num_vectors;
        true
    }
}

#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx2", target_feature = "fma"))
))]
impl<O: BatchOperation> Iterator for BatchIter<'_, O> {
    type Item = f32;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.buffer_index == self.buffer_len && !self.refill() {
            return None;
        }
        let value = self.buffer[self.buffer_index];
        self.buffer_index += 1;
        Some(value)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    /// Processes each filled buffer directly so consumers such as `sum` do not
    /// pay a refill check for every distance.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let accumulator = self.buffer[self.buffer_index..self.buffer_len]
            .iter()
            .copied()
            .fold(init, &mut f);

        // SAFETY: `new` requires the caller to verify the selected tier's
        // target features. Each helper runs the complete remaining loop in
        // that target-feature context, avoiding intermediate output writes.
        match self.kind {
            BatchKind::Scalar => {
                O::fold_scalar(self.key, self.batch, self.dimension, accumulator, f)
            }
            BatchKind::Avx => unsafe {
                O::fold_avx(self.key, self.batch, self.dimension, accumulator, f)
            },
            BatchKind::AvxFma => unsafe {
                O::fold_avx_fma(self.key, self.batch, self.dimension, accumulator, f)
            },
            BatchKind::Avx512 => unsafe {
                O::fold_avx512(self.key, self.batch, self.dimension, accumulator, f)
            },
        }
    }

    #[inline]
    fn for_each<F>(self, mut f: F)
    where
        F: FnMut(Self::Item),
    {
        self.fold((), |(), value| f(value));
    }
}

#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx2", target_feature = "fma"))
))]
impl<O: BatchOperation> ExactSizeIterator for BatchIter<'_, O> {
    #[inline]
    fn len(&self) -> usize {
        self.buffer_len - self.buffer_index + self.batch.len() / self.dimension
    }
}

pub use cosine::*;
pub use dot::*;
pub use hamming::{
    BinaryHashValues, Cluster, ClusteringResult, PairwiseResult, UnionFind, cluster_edges,
    cluster_pairwise_result, extract_binary_hashes_from_fixed_list, extract_hashes_from_fixed_list,
    hamming_distance_arrow_batch, hamming_u64, pairwise_hamming_distance,
    pairwise_hamming_distance_binary, pairwise_hamming_distance_binary_parallel,
    pairwise_hamming_distance_parallel,
};
pub use l2::*;
use lance_core::deepsize::DeepSizeOf;
pub use norm_l2::*;

use crate::Result;

/// Distance metrics type.
#[derive(Debug, Copy, Clone, PartialEq, DeepSizeOf)]
pub enum DistanceType {
    L2,
    Cosine,
    /// Dot Product
    Dot,
    /// Hamming Distance
    Hamming,
}

/// For backwards compatibility.
pub type MetricType = DistanceType;

pub type DistanceFunc<T> = fn(&[T], &[T]) -> f32;
pub type BatchDistanceFunc = fn(&[f32], &[f32], usize) -> Arc<Float32Array>;
pub type ArrowBatchDistanceFunc = fn(&dyn Array, &FixedSizeListArray) -> Result<Arc<Float32Array>>;

impl DistanceType {
    /// Compute the distance from one vector to a batch of vectors.
    ///
    /// This propagates nulls to the output.
    pub fn arrow_batch_func(&self) -> ArrowBatchDistanceFunc {
        match self {
            Self::L2 => l2_distance_arrow_batch,
            Self::Cosine => cosine_distance_arrow_batch,
            Self::Dot => dot_distance_arrow_batch,
            Self::Hamming => hamming_distance_arrow_batch,
        }
    }

    /// Returns the distance function between two vectors.
    pub fn func<T: L2 + Cosine + Dot>(&self) -> DistanceFunc<T> {
        match self {
            Self::L2 => l2,
            Self::Cosine => cosine_distance,
            Self::Dot => dot_distance,
            Self::Hamming => todo!(),
        }
    }
}

impl std::fmt::Display for DistanceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::L2 => "l2",
                Self::Cosine => "cosine",
                Self::Dot => "dot",
                Self::Hamming => "hamming",
            }
        )
    }
}

impl TryFrom<&str> for DistanceType {
    type Error = ArrowError;

    fn try_from(s: &str) -> std::result::Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "l2" | "euclidean" => Ok(Self::L2),
            "cosine" => Ok(Self::Cosine),
            "dot" => Ok(Self::Dot),
            "hamming" => Ok(Self::Hamming),
            _ => Err(ArrowError::InvalidArgumentError(format!(
                "Metric type '{s}' is not supported"
            ))),
        }
    }
}

pub fn multivec_distance(
    query: &dyn Array,
    vectors: &ListArray,
    distance_type: DistanceType,
) -> Result<Vec<f32>> {
    let dim = if let DataType::FixedSizeList(_, dim) = vectors.value_type() {
        dim as usize
    } else {
        return Err(ArrowError::InvalidArgumentError(
            "vectors must be a list of fixed size list".to_string(),
        ));
    };

    // check the query vectors type first
    // because we don't want to check the vectors type for each vector
    match query.data_type() {
        DataType::Float16 | DataType::Float32 | DataType::Float64 | DataType::UInt8 => {}
        _ => {
            return Err(ArrowError::InvalidArgumentError(
                "query must be a float array or binary array".to_string(),
            ));
        }
    }

    let mut dists = Vec::with_capacity(vectors.len());
    for v in vectors.iter() {
        match v {
            None => dists.push(f32::NAN),
            Some(v) => {
                let multivector = v.as_fixed_size_list();
                if multivector.len() == 0 {
                    dists.push(f32::NAN);
                    continue;
                }

                let sim = match distance_type {
                    DistanceType::Hamming => {
                        let query = query.as_primitive::<UInt8Type>().values();
                        query
                            .chunks_exact(dim)
                            .map(|q| {
                                multivector
                                    .values()
                                    .as_primitive::<UInt8Type>()
                                    .values()
                                    .chunks_exact(dim)
                                    .map(|v| hamming::hamming(q, v))
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .unwrap()
                            })
                            .sum()
                    }
                    _ => match query.data_type() {
                        DataType::Float16 => multivec_distance_impl::<Float16Type>(
                            query,
                            multivector,
                            dim,
                            distance_type,
                        ),
                        DataType::Float32 => multivec_distance_impl::<Float32Type>(
                            query,
                            multivector,
                            dim,
                            distance_type,
                        ),
                        DataType::Float64 => multivec_distance_impl::<Float64Type>(
                            query,
                            multivector,
                            dim,
                            distance_type,
                        ),
                        _ => unreachable!("missed to check query type"),
                    },
                };

                dists.push(1.0 - sim);
            }
        }
    }
    Ok(dists)
}

fn multivec_distance_impl<T: ArrowPrimitiveType>(
    query: &dyn Array,
    multivector: &FixedSizeListArray,
    dim: usize,
    distance_type: DistanceType,
) -> f32
where
    T::Native: L2 + Cosine + Dot,
{
    let query = query.as_primitive::<T>().values();
    query
        .chunks_exact(dim)
        .map(|q| {
            multivector
                .values()
                .as_primitive::<T>()
                .values()
                .chunks_exact(dim)
                .map(|v| 1.0 - distance_type.func()(q, v))
                .max_by(|a, b| a.total_cmp(b))
                .unwrap()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::types::Float32Type;
    use arrow_array::{Float32Array, ListArray};
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::Field;

    #[test]
    fn test_multivec_distance_empty_row_is_nan() {
        let query: Arc<dyn Array> = Arc::new(Float32Array::from_iter_values([1.0_f32, 2.0]));

        let dim = 2;
        let values = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec![Some(vec![Some(1.0_f32), Some(2.0)])],
            dim,
        );

        // Two rows: first is empty list, second has one sub-vector.
        let offsets = OffsetBuffer::from_lengths([0_usize, 1]);
        let field = Arc::new(Field::new("item", values.data_type().clone(), true));
        let vectors = ListArray::try_new(field, offsets, Arc::new(values), None).unwrap();

        let dists = multivec_distance(query.as_ref(), &vectors, DistanceType::Dot).unwrap();
        assert_eq!(dists.len(), 2);
        assert!(dists[0].is_nan());
        assert_eq!(dists[1], -4.0);
    }
}
