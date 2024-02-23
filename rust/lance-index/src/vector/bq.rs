// Copyright 2024 Lance Developers.
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

//! Binary Quantization (BQ)

use std::iter::once;
use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{cast::AsArray, Array, ArrayRef, UInt8Array};
use lance_core::{Error, Result};
use snafu::{location, Location};

#[derive(Clone, Default)]
pub struct BinaryQuantization {}

impl BinaryQuantization {
    /// Transform an array of float vectors to binary vectors.
    pub async fn transform(&self, data: &dyn Array) -> Result<ArrayRef> {
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
fn binary_quantization(data: &[f32]) -> impl Iterator<Item=u8> + '_ {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_quantization() {
        let data = vec![1.0, -1.0, 1.0, -5.0, -7.0, -1.0, 1.0, -1.0, -0.2, 1.2, 3.2];
        let expected = vec![0b01000101, 0b00000110];
        let result = binary_quantization(&data).collect::<Vec<_>>();
        assert_eq!(result, expected);
    }
}
