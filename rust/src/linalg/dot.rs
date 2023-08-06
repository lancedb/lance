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

//! Dot product.

use std::iter::Sum;
use std::sync::Arc;

use arrow_array::Float32Array;
use half::{bf16, f16};
use num_traits::real::Real;

#[inline]
pub fn dot<T: Real + Sum>(from: &[T], to: &[T]) -> T {
    from.iter().zip(to.iter()).map(|(x, y)| x.mul(*y)).sum()
}

pub trait Dot {
    type Output;

    /// Dot product.
    fn dot(&self, other: &Self) -> Self::Output;
}

impl Dot for [bf16] {
    type Output = bf16;

    fn dot(&self, other: &[bf16]) -> bf16 {
        dot(self, other)
    }
}

impl Dot for [f16] {
    type Output = f16;

    fn dot(&self, other: &[f16]) -> f16 {
        dot(self, other)
    }
}

impl Dot for [f32] {
    type Output = f32;

    fn dot(&self, other: &[f32]) -> f32 {
        dot(self, other)
    }
}

impl Dot for [f64] {
    type Output = f64;

    fn dot(&self, other: &[f64]) -> f64 {
        dot(self, other)
    }
}

pub fn dot_distance_batch(from: &[f32], to: &[f32], dimension: usize) -> Arc<Float32Array> {
    debug_assert_eq!(from.len(), dimension);
    debug_assert_eq!(to.len() % dimension, 0);

    let dists = unsafe {
        Float32Array::from_trusted_len_iter(to.chunks_exact(dimension).map(|v| Some(from.dot(v))))
    };
    Arc::new(dists)
}

pub fn dot_distance(from: &[f32], to: &[f32]) -> f32 {
    from.dot(to)
}
