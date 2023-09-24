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

use crate::bfloat16::BFloat16Type;
use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use arrow_array::{Array, ArrowPrimitiveType, Float16Array, Float32Array, Float64Array};
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

pub trait LanceArrowFloatType {}

impl LanceArrowFloatType for BFloat16Type {}

impl<T: ArrowPrimitiveType> LanceArrowFloatType for T where T::Native: Float + FromPrimitive {}

/// [FloatArray] is a trait that is implemented by all float type arrays.
pub trait FloatArray: Array + Clone + From<Vec<Self::Native>> + 'static{
    type Native: Float + FromPrimitive;

    type ArrowFloatType: LanceArrowFloatType;

    /// Float type
    const FLOAT_TYPE: FloatType;

    /// Returns a reference to the underlying data as a slice.
    fn as_slice(&self) -> &[Self::Native];
}

impl FloatArray for BFloat16Array {
    type Native = bf16;

    type ArrowFloatType = BFloat16Type;

    const FLOAT_TYPE: FloatType = FloatType::BFloat16;

    fn as_slice(&self) -> &[Self::Native] {
        // TODO: apache/arrow-rs#4820
        todo!()
    }
}

impl FloatArray for Float16Array {
    type Native = f16;

    type ArrowFloatType = Float16Type;

    const FLOAT_TYPE: FloatType = FloatType::Float16;

    fn as_slice(&self) -> &[Self::Native] {
        self.values()
    }
}

impl FloatArray for Float32Array {
    type Native = f32;

    type ArrowFloatType = Float32Type;

    const FLOAT_TYPE: FloatType = FloatType::Float32;

    fn as_slice(&self) -> &[Self::Native] {
        self.values()
    }
}

impl FloatArray for Float64Array {
    type Native = f64;

    type ArrowFloatType = Float64Type;

    const FLOAT_TYPE: FloatType = FloatType::Float64;

    fn as_slice(&self) -> &[Self::Native] {
        self.values()
    }
}
