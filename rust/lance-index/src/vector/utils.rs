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

use arrow_array::{Array, FixedSizeListArray};
use arrow_schema::DataType;
use lance_arrow::{ArrowFloatType, FloatType};
use lance_core::{Error, Result};
use lance_linalg::MatrixView;
use rand::distributions::Standard;
use rand::prelude::*;

use super::pb;

fn to_pb_data_type<T: ArrowFloatType>() -> pb::tensor::DataType {
    match T::FLOAT_TYPE {
        FloatType::BFloat16 => pb::tensor::DataType::Bfloat16,
        FloatType::Float16 => pb::tensor::DataType::Float16,
        FloatType::Float32 => pb::tensor::DataType::Float32,
        FloatType::Float64 => pb::tensor::DataType::Float64,
    }
}

fn arrow_to_pb_data_type(dt: &DataType) -> Result<pb::tensor::DataType> {
    match dt {
        DataType::UInt8 => Ok(pb::tensor::DataType::Uint8),
        DataType::UInt16 => Ok(pb::tensor::DataType::Uint16),
        DataType::UInt32 => Ok(pb::tensor::DataType::Uint32),
        DataType::UInt64 => Ok(pb::tensor::DataType::Uint64),
        DataType::Float16 => Ok(pb::tensor::DataType::Float16),
        DataType::Float32 => Ok(pb::tensor::DataType::Float32),
        DataType::Float64 => Ok(pb::tensor::DataType::Float64),
        _ => Err(Error::Index {
            message: format!("pb tensor type not supported: {:?}", dt),
        }),
    }
}

impl<T: ArrowFloatType> From<&MatrixView<T>> for pb::Tensor
where
    Standard: Distribution<<T as ArrowFloatType>::Native>,
{
    fn from(mat: &MatrixView<T>) -> Self {
        let mut tensor = pb::Tensor::default();
        tensor.data_type = to_pb_data_type::<T>() as i32;
        tensor.shape = vec![mat.num_rows() as u32, mat.num_columns() as u32];
        let flat_array = mat.data().as_ref().clone();
        tensor.data = flat_array.into_data().buffers()[0].to_vec();
        tensor
    }
}

impl TryFrom<&FixedSizeListArray> for pb::Tensor {
    type Error = Error;

    fn try_from(array: &FixedSizeListArray) -> Result<Self> {
        let mut tensor = pb::Tensor::default();
        tensor.data_type = arrow_to_pb_data_type(&array.value_type())? as i32;
        tensor.shape = vec![array.len() as u32, array.value_length() as u32];
        let flat_array = array.values();
        tensor.data = flat_array.into_data().buffers()[0].to_vec();
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::{types::*, Float16Array, Float32Array, Float64Array};
    use half::f16;
    use lance_arrow::bfloat16::BFloat16Type;
    use lance_arrow::FixedSizeListArrayExt;
    use num_traits::identities::Zero;

    #[test]
    fn test_to_pb_data_type() {
        assert_eq!(
            to_pb_data_type::<Float32Type>(),
            pb::tensor::DataType::Float32
        );
        assert_eq!(
            to_pb_data_type::<Float64Type>(),
            pb::tensor::DataType::Float64
        );
        assert_eq!(
            to_pb_data_type::<Float16Type>(),
            pb::tensor::DataType::Float16
        );
        assert_eq!(
            to_pb_data_type::<BFloat16Type>(),
            pb::tensor::DataType::Bfloat16
        );
    }

    #[test]
    fn test_matrix_to_tensor() {
        let mat = MatrixView::<Float32Type>::new(Arc::new(vec![0.0; 20].into()), 5);
        let tensor = pb::Tensor::from(&mat);
        assert_eq!(tensor.data_type, pb::tensor::DataType::Float32 as i32);
        assert_eq!(tensor.shape, vec![4, 5]);
        assert_eq!(tensor.data.len(), 20 * 4);

        let mat = MatrixView::<Float64Type>::new(Arc::new(vec![0.0; 20].into()), 5);
        let tensor = pb::Tensor::from(&mat);
        assert_eq!(tensor.data_type, pb::tensor::DataType::Float64 as i32);
        assert_eq!(tensor.shape, vec![4, 5]);
        assert_eq!(tensor.data.len(), 20 * 8);
    }

    #[test]
    fn test_fsl_to_tensor() {
        let fsl =
            FixedSizeListArray::try_new_from_values(Float16Array::from(vec![f16::zero(); 20]), 5)
                .unwrap();
        let tensor = pb::Tensor::try_from(&fsl).unwrap();
        assert_eq!(tensor.data_type, pb::tensor::DataType::Float16 as i32);
        assert_eq!(tensor.shape, vec![4, 5]);
        assert_eq!(tensor.data.len(), 20 * 2);

        let fsl =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0; 20]), 5).unwrap();
        let tensor = pb::Tensor::try_from(&fsl).unwrap();
        assert_eq!(tensor.data_type, pb::tensor::DataType::Float32 as i32);
        assert_eq!(tensor.shape, vec![4, 5]);
        assert_eq!(tensor.data.len(), 20 * 4);

        let fsl =
            FixedSizeListArray::try_new_from_values(Float64Array::from(vec![0.0; 20]), 5).unwrap();
        let tensor = pb::Tensor::try_from(&fsl).unwrap();
        assert_eq!(tensor.data_type, pb::tensor::DataType::Float64 as i32);
        assert_eq!(tensor.shape, vec![4, 5]);
        assert_eq!(tensor.data.len(), 20 * 8);
    }
}
