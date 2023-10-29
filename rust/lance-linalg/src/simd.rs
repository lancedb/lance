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

//! Poor-man's SIMD
//!
//! The difference between this implementation and [std::simd] is that
//! this implementation holds the SIMD register directly, thus it exposes more
//! optimization opportunity to use wide range of instructions available.
//!
//! Also, it gives us more control, for example, it is likely that we will have
//! f16/bf16 support before the standard library does.
//!
//! The API are close to [std::simd] to make migration easier in the future.

pub mod f32;

use num_traits::Float;

pub trait SIMD<T: Float>: std::fmt::Debug {
    fn splat(val: T) -> Self;

    /// Load aligned data from memory.
    fn load(ptr: *const T) -> Self;

    /// Load unaligned data from memory.
    fn load_unaligned(ptr: *const T) -> Self;

    fn store(&self, ptr: *mut T);

    fn store_unaligned(&self, ptr: *mut T);

    /// fused multiply-add
    ///
    /// c = a * b + c
    fn multiply_add(&mut self, a: Self, b: Self);

    fn argmin(&self) -> u32;

    fn argmax(&self) -> u32;

    fn reduce_sum(&self) -> T;
}
