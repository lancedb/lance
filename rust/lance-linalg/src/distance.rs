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

//! Distance metrics
//!
//! This module provides distance metrics for vectors.
//!
//! - `bf16, f16, f32, f64` types are supported.
//! - SIMD is used when available, on `x86_64` and `aarch64` architectures.

use std::sync::Arc;

use arrow_array::{FixedSizeListArray, Float32Array};
use arrow_schema::ArrowError;

pub mod cosine;
pub mod dot;
pub mod l2;
pub mod norm_l2;

pub use cosine::*;
pub use dot::*;
pub use l2::*;
pub use norm_l2::*;

/// Distance metrics type.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum DistanceType {
    L2,
    Cosine,
    Dot, // Dot product
}

/// For backwards compatibility.
pub type MetricType = DistanceType;

pub type DistanceFunc = fn(&[f32], &[f32]) -> f32;
pub type BatchDistanceFunc = fn(&[f32], &[f32], usize) -> Arc<Float32Array>;
pub type ArrowBatchDistanceFunc = fn(&[f32], &FixedSizeListArray) -> Arc<Float32Array>;

impl DistanceType {
    /// Compute the distance from one vector to a batch of vectors.
    ///
    /// This propagates nulls to the output.
    pub fn arrow_batch_func(&self) -> ArrowBatchDistanceFunc {
        match self {
            Self::L2 => l2_distance_arrow_batch,
            Self::Cosine => cosine_distance_arrow_batch,
            Self::Dot => dot_distance_arrow_batch,
        }
    }

    /// Returns the distance function between two vectors.
    pub fn func(&self) -> DistanceFunc {
        match self {
            Self::L2 => l2_distance,
            Self::Cosine => cosine_distance,
            Self::Dot => dot_distance,
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
            _ => Err(ArrowError::InvalidArgumentError(format!(
                "Metric type '{s}' is not supported"
            ))),
        }
    }
}
