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
use arrow_array::types::{Float32Type, UInt8Type};
use arrow_array::{Array, FixedSizeListArray, Float32Array, ListArray};
use arrow_schema::{ArrowError, DataType};

pub mod cosine;
pub mod dot;
pub mod hamming;
pub mod l2;
pub mod norm_l2;

pub use cosine::*;
use deepsize::DeepSizeOf;
pub use dot::*;
use hamming::hamming_distance_arrow_batch;
pub use l2::*;
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

    let dists = vectors
        .iter()
        .map(|v| {
            v.map(|v| {
                let multivector = v.as_fixed_size_list();
                match distance_type {
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
                    _ => {
                        let query = query.as_primitive::<Float32Type>().values();
                        query
                            .chunks_exact(dim)
                            .map(|q| {
                                multivector
                                    .values()
                                    .as_primitive::<Float32Type>()
                                    .values()
                                    .chunks_exact(dim)
                                    .map(|v| distance_type.func()(q, v))
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .unwrap()
                            })
                            .sum()
                    }
                }
            })
            .unwrap_or(f32::NAN)
        })
        .collect();
    Ok(dists)
}
