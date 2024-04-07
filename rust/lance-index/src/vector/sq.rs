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

use std::{ops::Range, sync::Arc};

use arrow::array::AsArray;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, UInt8Array};

use itertools::Itertools;
use lance_arrow::*;
use lance_core::{Error, Result};
use lance_linalg::distance::MetricType;
use num_traits::*;
use snafu::{location, Location};

pub mod builder;
pub mod storage;
pub mod transform;

/// Scalar Quantization, optimized for [Apache Arrow] buffer memory layout.
///
//
// TODO: move this to be pub(crate) once we have a better way to test it.
#[derive(Debug, Clone)]
pub struct ScalarQuantizer {
    /// Number of bits for the centroids.
    ///
    /// Only support 8, as one of `u8` byte now.
    pub num_bits: u16,

    /// Original dimension of the vectors.
    pub dim: usize,

    /// Distance type.
    pub metric_type: MetricType,

    pub bounds: Range<f64>,
}

impl ScalarQuantizer {
    pub fn new(num_bits: u16, dim: usize, metric_type: MetricType) -> Self {
        Self {
            num_bits,
            dim,
            metric_type,
            bounds: Range::<f64> {
                start: f64::MAX,
                end: f64::MIN,
            },
        }
    }

    pub fn with_bounds(
        num_bits: u16,
        dim: usize,
        metric_type: MetricType,
        bounds: Range<f64>,
    ) -> Self {
        let mut sq = Self::new(num_bits, dim, metric_type);
        sq.bounds = bounds;
        sq
    }

    pub fn num_bits(&self) -> u16 {
        self.num_bits
    }

    pub fn update_bounds<T: ArrowFloatType>(
        &mut self,
        vectors: &FixedSizeListArray,
    ) -> Result<Range<f64>> {
        let data = vectors
            .values()
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .ok_or(Error::Index {
                message: format!(
                    "Expect to be a float vector array, got: {:?}",
                    vectors.value_type()
                ),
                location: location!(),
            })?
            .as_slice();

        self.bounds = data.iter().fold(self.bounds.clone(), |f, v| {
            f.start.min(v.to_f64().unwrap())..f.end.max(v.to_f64().unwrap())
        });

        Ok(self.bounds.clone())
    }

    pub fn transform<T: ArrowFloatType>(&self, data: &dyn Array) -> Result<ArrayRef> {
        let fsl = data
            .as_fixed_size_list_opt()
            .ok_or(Error::Index {
                message: format!(
                    "Expect to be a FixedSizeList<float> vector array, got: {:?} array",
                    data.data_type()
                ),
                location: location!(),
            })?
            .clone();
        let data = fsl
            .values()
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .ok_or(Error::Index {
                message: format!(
                    "Expect to be a float vector array, got: {:?}",
                    fsl.value_type()
                ),
                location: location!(),
            })?
            .as_slice();

        // TODO: support SQ4
        let builder: Vec<u8> = scale_to_u8::<T>(data, self.bounds.clone());

        Ok(Arc::new(FixedSizeListArray::try_new_from_values(
            UInt8Array::from(builder),
            fsl.value_length(),
        )?))
    }

    pub fn bounds(&self) -> Range<f64> {
        self.bounds.clone()
    }

    /// Whether to use residual as input or not.
    pub fn use_residual(&self) -> bool {
        false
    }
}

pub(crate) fn scale_to_u8<T: ArrowFloatType>(values: &[T::Native], bounds: Range<f64>) -> Vec<u8> {
    let range = bounds.end - bounds.start;
    values
        .iter()
        .map(|&v| {
            let v = v.to_f64().unwrap();
            match v {
                v if v < bounds.start => 0,
                v if v > bounds.end => 255,
                _ => ((v - bounds.start) * f64::from_u32(255).unwrap() / range)
                    .round()
                    .to_u8()
                    .unwrap(),
            }
        })
        .collect_vec()
}
#[cfg(test)]
mod tests {
    use arrow::datatypes::{Float16Type, Float32Type, Float64Type};
    use arrow_array::{Float16Array, Float32Array, Float64Array};
    use half::f16;

    use super::*;

    #[tokio::test]
    async fn test_f16_sq8() {
        let float_values = Vec::from_iter((0..16).map(|v| f16::from_usize(v).unwrap()));
        let float_array = Float16Array::from_iter_values(float_values.clone());
        let vectors =
            FixedSizeListArray::try_new_from_values(float_array, float_values.len() as i32)
                .unwrap();
        let mut sq = ScalarQuantizer::new(8, float_values.len(), MetricType::L2);

        sq.update_bounds::<Float16Type>(&vectors).unwrap();
        assert_eq!(sq.bounds.start, float_values[0].to_f64());
        assert_eq!(
            sq.bounds.end,
            float_values.last().cloned().unwrap().to_f64()
        );

        let sq_code = sq.transform::<Float16Type>(&vectors).unwrap();
        let sq_values = sq_code
            .as_fixed_size_list()
            .values()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap();

        sq_values.values().iter().enumerate().for_each(|(i, v)| {
            assert_eq!(*v, (i * 17) as u8);
        });
    }

    #[tokio::test]
    async fn test_f32_sq8() {
        let float_values = Vec::from_iter((0..16).map(|v| v as f32));
        let float_array = Float32Array::from_iter_values(float_values.clone());
        let vectors =
            FixedSizeListArray::try_new_from_values(float_array, float_values.len() as i32)
                .unwrap();
        let mut sq = ScalarQuantizer::new(8, float_values.len(), MetricType::L2);

        sq.update_bounds::<Float32Type>(&vectors).unwrap();
        assert_eq!(sq.bounds.start, float_values[0].to_f64().unwrap());
        assert_eq!(
            sq.bounds.end,
            float_values.last().cloned().unwrap().to_f64().unwrap()
        );

        let sq_code = sq.transform::<Float32Type>(&vectors).unwrap();
        let sq_values = sq_code
            .as_fixed_size_list()
            .values()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap();

        sq_values.values().iter().enumerate().for_each(|(i, v)| {
            assert_eq!(*v, (i * 17) as u8,);
        });
    }

    #[tokio::test]
    async fn test_f64_sq8() {
        let float_values = Vec::from_iter((0..16).map(|v| v as f64));
        let float_array = Float64Array::from_iter_values(float_values.clone());
        let vectors =
            FixedSizeListArray::try_new_from_values(float_array, float_values.len() as i32)
                .unwrap();
        let mut sq = ScalarQuantizer::new(8, float_values.len(), MetricType::L2);

        sq.update_bounds::<Float64Type>(&vectors).unwrap();
        assert_eq!(sq.bounds.start, float_values[0]);
        assert_eq!(sq.bounds.end, float_values.last().cloned().unwrap());

        let sq_code = sq.transform::<Float64Type>(&vectors).unwrap();
        let sq_values = sq_code
            .as_fixed_size_list()
            .values()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap();

        sq_values.values().iter().enumerate().for_each(|(i, v)| {
            assert_eq!(*v, (i * 17) as u8,);
        });
    }
}
