// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Binary Quantization (BQ)

use std::iter::once;
use std::str::FromStr;
use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{cast::AsArray, Array, ArrayRef, UInt8Array};
use lance_core::{Error, Result};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use snafu::location;

use crate::vector::quantizer::QuantizerBuildParams;

pub mod builder;
pub mod rotation;
pub mod storage;
pub mod transform;

#[derive(Clone, Default)]
pub struct BinaryQuantization {}

impl BinaryQuantization {
    /// Transform an array of float vectors to binary vectors.
    pub fn transform(&self, data: &dyn Array) -> Result<ArrayRef> {
        let fsl = data
            .as_fixed_size_list_opt()
            .ok_or(Error::Index {
                message: format!(
                    "Expect to be a float vector array, got: {:?}",
                    data.data_type()
                ),
                location: location!(),
            })?
            .clone();

        let data = fsl
            .values()
            .as_primitive_opt::<Float32Type>()
            .ok_or(Error::Index {
                message: format!(
                    "Expect to be a float32 vector array, got: {:?}",
                    fsl.values().data_type()
                ),
                location: location!(),
            })?;
        let dim = fsl.value_length() as usize;
        let code = data
            .values()
            .chunks_exact(dim)
            .flat_map(binary_quantization)
            .collect::<Vec<_>>();

        Ok(Arc::new(UInt8Array::from(code)))
    }
}

/// Binary quantization.
///
/// Use the sign bit of the float vector to represent the binary vector.
fn binary_quantization<T: Float>(data: &[T]) -> impl Iterator<Item = u8> + '_ {
    let iter = data.chunks_exact(8);
    iter.clone()
        .map(|c| {
            // Auto vectorized.
            // Before changing this code, please check the assembly output.
            let mut bits: u8 = 0;
            c.iter().enumerate().for_each(|(idx, v)| {
                bits |= (v.is_sign_positive() as u8) << idx;
            });
            bits
        })
        .chain(once(0).map(move |_| {
            let mut bits: u8 = 0;
            iter.remainder().iter().enumerate().for_each(|(idx, v)| {
                bits |= (v.is_sign_positive() as u8) << idx;
            });
            bits
        }))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RQRotationType {
    #[default]
    Fast,
    Matrix,
}

impl FromStr for RQRotationType {
    type Err = Error;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "fast" | "fht_kac" | "fht-kac" => Ok(Self::Fast),
            "matrix" | "dense" => Ok(Self::Matrix),
            _ => Err(Error::invalid_input(
                format!(
                    "Unknown RQ rotation type: {}. Expected one of: fast, matrix",
                    value
                ),
                location!(),
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RQBuildParams {
    pub num_bits: u8,
    pub rotation_type: RQRotationType,
}

impl RQBuildParams {
    pub fn new(num_bits: u8) -> Self {
        Self {
            num_bits,
            rotation_type: RQRotationType::default(),
        }
    }

    pub fn with_rotation_type(num_bits: u8, rotation_type: RQRotationType) -> Self {
        Self {
            num_bits,
            rotation_type,
        }
    }
}

impl QuantizerBuildParams for RQBuildParams {
    fn sample_size(&self) -> usize {
        0
    }
}

impl Default for RQBuildParams {
    fn default() -> Self {
        Self {
            num_bits: 1,
            rotation_type: RQRotationType::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use half::{bf16, f16};

    fn test_bq<T: Float>() {
        let data: Vec<T> = [1.0, -1.0, 1.0, -5.0, -7.0, -1.0, 1.0, -1.0, -0.2, 1.2, 3.2]
            .iter()
            .map(|&v| T::from(v).unwrap())
            .collect();
        let expected = vec![0b01000101, 0b00000110];
        let result = binary_quantization(&data).collect::<Vec<_>>();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_binary_quantization() {
        test_bq::<bf16>();
        test_bq::<f16>();
        test_bq::<f32>();
        test_bq::<f64>();
    }

    #[test]
    fn test_rotation_type_parse() {
        assert_eq!(
            "fast".parse::<RQRotationType>().unwrap(),
            RQRotationType::Fast
        );
        assert_eq!(
            "matrix".parse::<RQRotationType>().unwrap(),
            RQRotationType::Matrix
        );
        assert!("invalid".parse::<RQRotationType>().is_err());
    }
}
