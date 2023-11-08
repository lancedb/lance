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

use lance_arrow::ArrowFloatType;
use lance_linalg::MatrixView;

/// Divide a 2D vector in [`T::Array`] to `m` sub-vectors.
///
/// For example, for a `[1024x1M]` matrix, when `n = 8`, this function divides
/// the matrix into  `[128x1M; 8]` vector of matrix.
pub(super) fn divide_to_subvectors<T: ArrowFloatType>(
    data: &MatrixView<T>,
    m: usize,
) -> Vec<Arc<T::ArrayType>> {
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
        let values = T::ArrayType::from(builder);
        subarrays.push(Arc::new(values));
    }
    subarrays
}

/// Number of PQ centroids, for the corresponding number of PQ bits.
///
// TODO: pub(crate)
pub fn num_centroids(num_bits: impl Into<u32>) -> usize {
    2_usize.pow(num_bits.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{types::Float32Type, Float32Array};

    #[test]
    fn test_divide_to_subvectors() {
        let values = Float32Array::from_iter((0..320).map(|v| v as f32));
        // A [10, 32] array.
        let mat = MatrixView::new(values.into(), 32);
        let sub_vectors = divide_to_subvectors::<Float32Type>(&mat, 4);
        assert_eq!(sub_vectors.len(), 4);
        assert_eq!(sub_vectors[0].len(), 10 * 8);

        assert_eq!(
            sub_vectors[0].as_ref(),
            &Float32Array::from_iter_values(
                (0..10).flat_map(|i| (0..8).map(move |c| 32.0 * i as f32 + c as f32))
            )
        );
    }
}
