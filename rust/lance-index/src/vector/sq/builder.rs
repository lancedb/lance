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

use arrow::{
    array::AsArray,
    datatypes::{Float16Type, Float32Type, Float64Type},
};
use arrow_array::Array;

use arrow_schema::DataType;
use lance_core::{Error, Result};
use lance_linalg::distance::MetricType;
use snafu::{location, Location};

use super::ScalarQuantizer;

#[derive(Debug, Clone)]
pub struct SQBuildParams {
    /// Number of bits of scaling range.
    pub num_bits: u16,

    /// Sample rate for training.
    pub sample_rate: usize,
}

impl Default for SQBuildParams {
    fn default() -> Self {
        Self {
            num_bits: 8,
            sample_rate: 256,
        }
    }
}

impl SQBuildParams {
    pub fn build(&self, data: &dyn Array, metric_type: MetricType) -> Result<ScalarQuantizer> {
        let fsl = data.as_fixed_size_list_opt().ok_or(Error::Index {
            message: format!(
                "SQ builder: input is not a FixedSizeList: {}",
                data.data_type()
            ),
            location: location!(),
        })?;

        let mut quantizer =
            ScalarQuantizer::new(self.num_bits, fsl.value_length() as usize, metric_type);

        match fsl.value_type() {
            DataType::Float16 => {
                quantizer.update_bounds::<Float16Type>(fsl)?;
            }
            DataType::Float32 => {
                quantizer.update_bounds::<Float32Type>(fsl)?;
            }
            DataType::Float64 => {
                quantizer.update_bounds::<Float64Type>(fsl)?;
            }
            _ => {
                return Err(Error::Index {
                    message: format!("SQ builder: unsupported data type: {}", fsl.value_type()),
                    location: location!(),
                })
            }
        }

        Ok(quantizer)
    }
}
