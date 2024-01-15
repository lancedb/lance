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

use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::{
    fmt::Formatter,
    ops::{AddAssign, DivAssign},
};

use arrow_array::{
    types::{Float16Type, Float32Type, Float64Type},
    Array, Float16Array, Float32Array, Float64Array,
};
use arrow_schema::{DataType, Field};
use half::{bf16, f16};
use num_traits::{AsPrimitive, Bounded, Float, FromPrimitive};

use super::bfloat16::{BFloat16Array, BFloat16Type};
use crate::bfloat16::is_bfloat16_field;
use crate::Result;

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

/// Try to convert a [DataType] to a [FloatType]. To support bfloat16, always
/// prefer using the `TryFrom<&Field>` implementation.
impl TryFrom<&DataType> for FloatType {
    type Error = crate::ArrowError;

    fn try_from(value: &DataType) -> Result<Self> {
        match *value {
            DataType::Float16 => Ok(Self::Float16),
            DataType::Float32 => Ok(Self::Float32),
            DataType::Float64 => Ok(Self::Float64),
            _ => Err(crate::ArrowError::InvalidArgumentError(format!(
                "{:?} is not a floating type",
                value
            ))),
        }
    }
}

impl TryFrom<&Field> for FloatType {
    type Error = crate::ArrowError;

    fn try_from(field: &Field) -> Result<Self> {
        match field.data_type() {
            DataType::FixedSizeBinary(2) if is_bfloat16_field(field) => Ok(Self::BFloat16),
            _ => Self::try_from(field.data_type()),
        }
    }
}

/// Trait for float types used in Arrow Array.
///
pub trait ArrowFloatType: Debug {
    type Native: FromPrimitive
        + FloatToArrayType<ArrowType = Self>
        + AsPrimitive<f32>
        + Debug
        + Display;

    const FLOAT_TYPE: FloatType;

    /// Arrow Float Array Type.
    type ArrayType: FloatArray<Self>;

    /// Returns empty array of this type.
    fn empty_array() -> Self::ArrayType {
        Vec::<Self::Native>::new().into()
    }
}

pub trait FloatToArrayType:
    Float
    + Bounded
    + Sum
    + AddAssign<Self>
    + AsPrimitive<f64>
    + AsPrimitive<f32>
    + DivAssign
    + Send
    + Sync
    + Copy
{
    type ArrowType: ArrowFloatType<Native = Self>;
}

impl FloatToArrayType for bf16 {
    type ArrowType = BFloat16Type;
}

impl FloatToArrayType for f16 {
    type ArrowType = Float16Type;
}

impl FloatToArrayType for f32 {
    type ArrowType = Float32Type;
}

impl FloatToArrayType for f64 {
    type ArrowType = Float64Type;
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

/// Convert a float32 array to another float array.
pub fn coerce_float_vector(input: &Float32Array, float_type: FloatType) -> Result<Box<dyn Array>> {
    match float_type {
        FloatType::BFloat16 => Ok(Box::new(BFloat16Array::from_iter_values(
            input.values().iter().map(|v| bf16::from_f32(*v)),
        ))),
        FloatType::Float16 => Ok(Box::new(Float16Array::from_iter_values(
            input.values().iter().map(|v| f16::from_f32(*v)),
        ))),
        FloatType::Float32 => Ok(Box::new(input.clone())),
        FloatType::Float64 => Ok(Box::new(Float64Array::from_iter_values(
            input.values().iter().map(|v| *v as f64),
        ))),
    }
}
