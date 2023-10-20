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

use arrow_array::Array;
use lance_arrow::{ArrowFloatType, FloatType};
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::types::*;
    use lance_arrow::bfloat16::BFloat16Type;

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
}
