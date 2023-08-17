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

//! Testing utilities

use num_traits::real::Real;
use num_traits::FromPrimitive;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::iter::repeat_with;

use arrow_array::{ArrowNumericType, Float32Array, NativeAdapter, PrimitiveArray};

/// Create a random float32 array.
pub fn generate_random_array_with_seed<T: ArrowNumericType>(
    n: usize,
    seed: [u8; 32],
) -> PrimitiveArray<T>
where
    T::Native: Real + FromPrimitive,
    NativeAdapter<T>: From<T::Native>,
{
    let mut rng = StdRng::from_seed(seed);

    PrimitiveArray::<T>::from_iter(repeat_with(|| T::Native::from_f32(rng.gen::<f32>())).take(n))
}

/// Create a random float32 array.
pub fn generate_random_array(n: usize) -> Float32Array {
    let mut rng = rand::thread_rng();
    Float32Array::from(
        repeat_with(|| rng.gen::<f32>())
            .take(n)
            .collect::<Vec<f32>>(),
    )
}
