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
        #[cfg(target_os = "macos")]
        use accelerate_src;
        use lapack::sgesvd;

        if !matches!(self.value_type(), DataType::Float32) {
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
        let mut u = vec![0.0; (m * m) as usize];
        // f32 array builder for matrix V_T
        let mut vt = vec![0.0; (n*n) as usize];
        let mut sigma = vec![0.0;n as usize];

        let mut work = vec![0_f32; 1];
        let lwork: i32 = -1;
        let mut info: i32 = -1;

        unsafe {
            sgesvd(
                b'A',
                b'A',
                m,
                n,
                a.as_mut_slice(),
                m,
                sigma.as_mut_slice(),
                u.as_mut_slice(),
                m,
                vt.as_mut_slice(),
                n,
                work.as_mut_slice(),
                lwork,
                &mut info,
            );
        }
        println!("Info value: {info} work={:?}", work);
        if info > 0 {
            println!("Failed to compute sgesvd");
        }

        let u_values = Float32Array::from_iter_values(u);
        let u = FixedSizeListArray::try_new(&u_values, m)?;
        let vt_values = Float32Array::from_iter_values(vt);
        let vt = FixedSizeListArray::try_new(&vt_values, n)?;
        let sigma = Float32Array::from_iter_values(sigma);
        Ok((u, sigma, vt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svd() {
        let values = Float32Array::from_iter_values([
            8.79, 9.93, 9.83, 5.45, 3.16, 6.11, 6.91, 5.04, -0.27, 7.98, -9.15, -7.93, 4.86, 4.85,
            3.01, 9.57, 1.64, 8.83, 0.74, 5.80, -3.49, 4.02, 9.80, 10.00, 4.27, 9.84, 0.15, -8.99,
            -6.02, -5.31,
        ]);
        let a = FixedSizeListArray::try_new(&values, 5).unwrap();

        let (u, sigma, vt) = a.svd().unwrap();
        let expected_u = Float32Array::from_iter_values([
            -0.59114238,
            0.26316781,
            0.35543017,
            0.31426436,
            0.22993832,
            0.55075318,
            -0.39756679,
            0.24379903,
            -0.22239,
            -0.75346615,
            -0.36358969,
            0.18203479,
            -0.03347897,
            -0.60027258,
            -0.45083927,
            0.23344966,
            -0.30547573,
            0.53617327,
            -0.4297069,
            0.23616681,
            -0.68586286,
            0.33186002,
            0.16492763,
            -0.38966287,
            -0.46974792,
            -0.3508914,
            0.3874446,
            0.15873556,
            -0.51825744,
            -0.46077223,
            0.29335876,
            0.57626212,
            -0.02085292,
            0.37907767,
            -0.6525516,
            0.10910681,
        ]);
        assert_eq!(as_primitive_array(u.values().as_ref()), &expected_u);
    }
}
