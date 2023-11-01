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

use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

pub mod f32;

use num_traits::Float;

/// Lance SIMD lib
///
pub trait SIMD<T: Float>:
    std::fmt::Debug + AddAssign + Add + Mul + Sub + SubAssign + Copy + Clone + Sized
{
    /// Create a new instance with all lanes set to `val`.
    fn splat(val: T) -> Self;

    /// Create a new instance with all lanes set to zero.
    fn zeros() -> Self;

    /// Load aligned data from aligned memory.
    ///
    /// # Safety
    ///
    /// It crashes if the ptr is not aligned.
    unsafe fn load(ptr: *const T) -> Self;

    /// Load unaligned data from memory.
    ///
    /// # Safety
    unsafe fn load_unaligned(ptr: *const T) -> Self;

    /// Store the values to aligned memory.
    ///
    /// # Safety
    ///
    /// It crashes if the ptr is not aligned
    unsafe fn store(&self, ptr: *mut T);

    /// Store the values to unaligned memory.
    ///
    /// # Safety
    unsafe fn store_unaligned(&self, ptr: *mut T);

    /// fused multiply-add
    ///
    /// c = a * b + c
    fn multiply_add(&mut self, a: Self, b: Self);

    /// Calculate the sum across this vector.
    fn reduce_sum(&self) -> T;
}
