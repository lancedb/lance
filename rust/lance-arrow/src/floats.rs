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

use arrow_array::{Array, Float32Array, Float16Array, Float64Array};
use half::{bf16, f16};
use num_traits::{Float, FromPrimitive};

use super::bfloat16::BFloat16Array;

/// Float data type.
///
/// This helps differentiate between the different float types,
/// because bf16 is not officially supported [DataType] in arrow-rs.
pub enum FloatType {
    BFloat16,
    Float16,
    Float32,
    Float64,
}

/// [FloatArray] is a trait that is implemented by all float type arrays.
pub trait FloatArray: Array + From<Vec<Self::Native>> {
    type Native: Float + FromPrimitive;

    /// Float type
    const DATA_TYPE: FloatType;

    /// Returns a reference to the underlying data as a slice.
    fn as_slice(&self) -> &[Self::Native];
}

impl FloatArray for BFloat16Array {
    type Native = bf16;

    const DATA_TYPE: FloatType = FloatType::BFloat16;

    fn as_slice(&self) -> &[Self::Native] {
        // TODO: apache/arrow-rs#4820
        todo!()
    }
}

impl FloatArray for Float16Array {
    type Native = f16;

    const DATA_TYPE: FloatType = FloatType::Float16;

    fn as_slice(&self) -> &[Self::Native] {
        self.values()
    }
}

impl FloatArray for Float32Array {
    type Native = f32;

    const DATA_TYPE: FloatType = FloatType::Float32;

    fn as_slice(&self) -> &[Self::Native] {
        self.values()
    }
}

impl FloatArray for Float64Array {
    type Native = f64;

    const DATA_TYPE: FloatType = FloatType::Float64;

    fn as_slice(&self) -> &[Self::Native] {
        self.values()
    }
}