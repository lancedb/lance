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

//! Floats Array

use arrow_array::{Array, ArrowPrimitiveType, PrimitiveArray};
use half::bf16;
use num_traits::{Float, FromPrimitive};

use super::bfloat16::BFloat16Array;

/// [FloatArray] is a trait that is implemented by all float type arrays.
pub trait FloatArray: Array + From<Vec<Self::Native>> {
    type Native: Float + FromPrimitive;

    /// Returns a reference to the underlying data as a slice.
    fn as_slice(&self) -> &[Self::Native];
}

impl FloatArray for BFloat16Array {
    type Native = bf16;

    fn as_slice(&self) -> &[Self::Native] {
        // TODO: apache/arrow-rs#4820
        todo!()
    }
}

impl<T: ArrowPrimitiveType> FloatArray for PrimitiveArray<T>
where
    T::Native: Float + FromPrimitive,
    Self: From<Vec<T::Native>>,
{
    type Native = T::Native;

    fn as_slice(&self) -> &[Self::Native] {
        self.values()
    }
}
