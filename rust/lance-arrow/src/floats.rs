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
use arrow_array::{Array, Float16Array, Float32Array, Float64Array};
use half::{bf16, f16};
use num_traits::{Float, FromPrimitive};
use std::fmt::Formatter;

use super::bfloat16::BFloat16Array;

/// Float data type.
///
/// This helps differentiate between the different float types,
/// because bf16 is not officially supported [DataType] in arrow-rs.
#[derive(Debug)]
pub enum FloatType {
    BFloat16,
    Float16,
    Float32,
    Float64,
}

impl std::fmt::Display for FloatType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BFloat16 => write!(f, "bfloat16"),
            Self::Float16 => write!(f, "float16"),
            Self::Float32 => write!(f, "float32"),
            Self::Float64 => write!(f, "float64"),
        }
    }
}

pub trait ArrowFloatType {
    type Native: Float + FromPrimitive;

    const FLOAT_TYPE: FloatType;

    type ArrayType: FloatArray<Self>;
}

impl ArrowFloatType for BFloat16Type {
    type Native = bf16;

    const FLOAT_TYPE: FloatType = FloatType::BFloat16;

    type ArrayType = BFloat16Array;
}

impl ArrowFloatType for Float16Type {
    type Native = f16;

    const FLOAT_TYPE: FloatType = FloatType::Float16;

    type ArrayType = Float16Array;
}

impl ArrowFloatType for Float32Type {
    type Native = f32;

    const FLOAT_TYPE: FloatType = FloatType::Float32;

    type ArrayType = Float32Array;
}

impl ArrowFloatType for Float64Type {
    type Native = f64;

    const FLOAT_TYPE: FloatType = FloatType::Float64;

    type ArrayType = Float64Array;
}

/// [FloatArray] is a trait that is implemented by all float type arrays.
pub trait FloatArray<T: ArrowFloatType + ?Sized>:
    Array + Clone + From<Vec<T::Native>> + 'static
{
    type FloatType: ArrowFloatType;

    /// Returns a reference to the underlying data as a slice.
    fn as_slice(&self) -> &[T::Native];
}

impl FloatArray<BFloat16Type> for BFloat16Array {
    type FloatType = BFloat16Type;

    fn as_slice(&self) -> &[<BFloat16Type as ArrowFloatType>::Native] {
        // TODO: apache/arrow-rs#4820
        todo!()
    }
}

impl FloatArray<Float16Type> for Float16Array {
    type FloatType = Float16Type;

    fn as_slice(&self) -> &[<Float16Type as ArrowFloatType>::Native] {
        self.values()
    }
}

impl FloatArray<Float32Type> for Float32Array {
    type FloatType = Float32Type;

    fn as_slice(&self) -> &[<Float32Type as ArrowFloatType>::Native] {
        self.values()
    }
}

impl FloatArray<Float64Type> for Float64Array {
    type FloatType = Float64Type;

    fn as_slice(&self) -> &[<Float64Type as ArrowFloatType>::Native] {
        self.values()
    }
}
