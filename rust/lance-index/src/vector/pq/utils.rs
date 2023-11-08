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

use std::sync::Arc;

use arrow_array::{FixedSizeListArray, types::Float32Type, Float32Array};
use lance_linalg::MatrixView;
use lance_arrow::FixedSizeListArrayExt;

/// Divide a 2D vector in [`FixedSizeListArray`] to `m` sub-vectors.
///
/// For example, for a `[1024x1M]` matrix, when `n = 8`, this function divides
/// the matrix into  `[128x1M; 8]` vector of matrix.
pub(super) fn divide_to_subvectors(data: &MatrixView<Float32Type>, m: usize) -> Vec<Arc<FixedSizeListArray>> {
    assert!(!data.num_rows() > 0);

    let sub_vector_length = data.num_columns() / m;
    let capacity = data.num_rows() * sub_vector_length;
    let mut subarrays = vec![];

    // TODO: very intensive memory copy involved!!! But this is on the write path.
    // Optimize for memory copy later.
    for i in 0..m {
        let mut builder = Vec::with_capacity(capacity);
        for j in 0..data.num_rows() {
            let row = data.row(j).unwrap();
            let start = i * sub_vector_length;
            builder.extend_from_slice(&row[start..start + sub_vector_length]);
        }
        let values = Float32Array::from(builder);
        let sub_array = Arc::new(
            FixedSizeListArray::try_new_from_values(values, sub_vector_length as i32).unwrap(),
        );
        subarrays.push(sub_array);
    }
    subarrays
}
