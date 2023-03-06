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

//! Linear Algebra on Arrow Array
//!
//!

use std::cmp::min;

use arrow::array::{as_primitive_array, Float32Builder};
use arrow_array::{Array, FixedSizeListArray, Float32Array};
use arrow_schema::DataType;

use crate::{arrow::FixedSizeListArrayExt, Error, Result};

/// Single Value Decomposition.
///
/// https://en.wikipedia.org/wiki/Singular_value_decomposition
pub trait SingularValueDecomposition {
    /// Matrix type
    type Matrix;
    ///
    type Sigma;

    /// Compute Singular Value Decomposition over 2-D matrix.
    ///
    /// Returns: `(U, Sigma, Vt)`
    fn svd(&self) -> Result<(Self::Matrix, Self::Sigma, Self::Matrix)>;
}

impl SingularValueDecomposition for FixedSizeListArray {
    type Matrix = Self;
    type Sigma = Float32Array;

    fn svd(&self) -> Result<(Self::Matrix, Self::Sigma, Self::Matrix)> {
        use lapacke::{sgesvd, Layout};

        if !matches!(self.data_type(), DataType::Float32) {
            return Err(Error::Arrow(format!(
                "SVD only supports f32 type, got {}",
                self.data_type()
            )));
        }

        let m = self.len() as i32;
        let n = self.value_length();
        let values = self.values();
        let flatten_values: &Float32Array = as_primitive_array(values.as_ref());
        // Solving: A = U * Sigma * Vt
        //
        // TODO: Lapacke requires a mutable reference of `A`.
        // How can we get mutable reference without copying data?
        let mut a = flatten_values.values().to_vec();
        // f32 array builder for matrix U.
        let mut u_builder = Float32Builder::with_capacity((m * m) as usize);
        // f32 array builder for matrix V_T
        let mut vt_builder = Float32Builder::with_capacity((n * n) as usize);
        let mut sigma_builder = Float32Builder::with_capacity(n as usize);
        let mut superb = vec![0_f32; min(m, n) as usize - 1];

        unsafe {
            sgesvd(
                Layout::RowMajor,
                b'A',
                b'A',
                m,
                n,
                a.as_mut_slice(),
                m as i32,
                sigma_builder.values_slice_mut(),
                u_builder.values_slice_mut(),
                n,
                vt_builder.values_slice_mut(),
                n,
                superb.as_mut_slice(),
            );
        }

        let u_values = u_builder.finish();
        let u = FixedSizeListArray::try_new(&u_values, m)?;
        let vt_values = vt_builder.finish();
        let vt = FixedSizeListArray::try_new(&vt_values, n)?;
        let sigma = sigma_builder.finish();
        Ok((u, sigma, vt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svd() {}
}
