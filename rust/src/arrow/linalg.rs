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
use std::sync::Arc;

use arrow::array::{as_primitive_array, Float32Builder};
use arrow_array::{Array, FixedSizeListArray, Float32Array};
use arrow_schema::DataType;

#[allow(unused_imports)]
#[cfg(target_os = "macos")]
use accelerate_src;

#[allow(unused_imports)]
#[cfg(target_os = "linux")]
use openblas_src;

use crate::{arrow::FixedSizeListArrayExt, Error, Result};

/// Transpose a matrix.
fn transpose(input: &[f32], dimension: usize) -> Vec<f32> {
    let n = input.len() / dimension;
    let mut mat = vec![0_f32; input.len()];
    for row in 0..dimension {
        for col in 0..n {
            mat[row * n + col] = input[col * dimension + row];
        }
    }

    mat
}

/// A 2-D matrix view on top of Arrow Arrays.
///
/// [MatrixView] does not own the data.
#[derive(Debug)]
pub struct MatrixView {
    /// Underneath data array.
    pub data: Arc<Float32Array>,

    /// The number of
    pub num_columns: usize,

    /// Is this matrix transposed or not.
    pub transpose: bool,
}

impl MatrixView {
    /// Create a MatrixView from a
    pub fn new(data: Arc<Float32Array>, num_columns: usize) -> Self {
        Self {
            data,
            num_columns,
            transpose: false,
        }
    }

    /// Number of rows in the matrix
    pub fn num_rows(&self) -> usize {
        if self.transpose {
            self.num_columns
        } else {
            self.data.len() / self.num_columns
        }
    }

    pub fn num_columns(&self) -> usize {
        if self.transpose {
            self.data.len() / self.num_columns
        } else {
            self.num_columns
        }
    }

    /// (Lazy) transpose of the matrix.
    ///
    pub fn transpose(&self) -> Self {
        Self {
            data: self.data.clone(),
            num_columns: self.num_columns,
            transpose: !self.transpose,
        }
    }

    /// Dot multiply
    pub fn dot(&self, rhs: &Self) -> Result<Self> {
        use cblas::{sgemm, Layout, Transpose};

        let m = self.num_rows() as i32;
        let k = self.num_columns() as i32;
        let n = rhs.num_columns() as i32;
        if self.num_columns() != rhs.num_rows() {
            return Err(Error::Arrow(format!(
                "MatMul dimension mismatch: A({m}x{k}) * B({}x{n}",
                rhs.num_rows()
            )));
        }

        let mut c_builder = Float32Builder::with_capacity((m * n) as usize);
        unsafe { c_builder.append_trusted_len_iter((0..n * m).map(|_| 0.0)) }

        let (trans_a, lda) = if self.transpose {
            (Transpose::Ordinary, m)
        } else {
            (Transpose::None, k)
        };
        let (trans_b, ldb) = if rhs.transpose {
            (Transpose::Ordinary, k)
        } else {
            (Transpose::None, n)
        };
        unsafe {
            sgemm(
                Layout::RowMajor,
                trans_a,
                trans_b,
                m,
                n,
                k,
                1.0,
                self.data.as_ref().values(),
                lda,
                rhs.data.as_ref().values(),
                ldb,
                0.0,
                c_builder.values_slice_mut(),
                n,
            )
        }

        let data = Arc::new(c_builder.finish());
        Ok(Self {
            data,
            num_columns: n as usize,
            transpose: false,
        })
    }
}

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
        /// Sadly that the Accelerate Framework on macOS does not have LAPACKE(C)
        /// so we have to use the Fortran one which is column-major matrix.
        use lapack::sgesdd;

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
        let mut a = transpose(flatten_values.values(), n as usize);
        // f32 array builder for matrix U.
        let mut u = vec![0.0; (m * m) as usize];
        // f32 array builder for matrix V_T
        let mut vt = vec![0.0; (n * n) as usize];
        let mut sigma = vec![0.0; n as usize];

        let mut work = vec![0_f32; 1];
        // Length of the workspace
        let lwork: i32 = -1;
        let mut iwork = vec![0; 8 * min(m, m) as usize];
        //  return value.
        let mut info: i32 = -1;

        // Query the optimal workspace size, will be stored in `work[0]`.
        unsafe {
            sgesdd(
                b'A',
                m,
                n,
                a.as_mut_slice(),
                m,
                sigma.as_mut_slice(),
                u.as_mut_slice(),
                m,
                vt.as_mut_slice(),
                m,
                work.as_mut_slice(),
                lwork,
                iwork.as_mut_slice(),
                &mut info,
            );
        }
        if info > 0 {
            return Err(Error::Arrow("Failed to compute SVD".to_string()));
        }

        let lwork = work[0] as i32;
        let mut work = vec![0_f32; lwork as usize];
        unsafe {
            sgesdd(
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
                iwork.as_mut_slice(),
                &mut info,
            );
        }
        if info != 0 {
            return Err(Error::Arrow("Failed to compute SVD".to_string()));
        }

        let u_values = Float32Array::from_iter_values(transpose(&u, m as usize));
        let u = Self::try_new(&u_values, m)?;
        let vt_values = Float32Array::from_iter_values(transpose(&vt, n as usize));
        let vt = Self::try_new(&vt_values, n)?;
        let sigma = Float32Array::from_iter_values(sigma);
        Ok((u, sigma, vt))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use arrow::datatypes::Float32Type;

    use super::*;

    #[test]
    fn test_svd() {
        // A 6 x 5 matrix, from
        // https://www.intel.com/content/www/us/en/develop/documentation/onemkl-lapack-examples/top/least-squares-and-eigenvalue-problems/singular-value-decomposition/gesvd-function/sgesvd-example/lapacke-sgesvd-example-c-row.html
        let values = Float32Array::from_iter_values([
            8.79, 9.93, 9.83, 5.45, 3.16, 6.11, 6.91, 5.04, -0.27, 7.98, -9.15, -7.93, 4.86, 4.85,
            3.01, 9.57, 1.64, 8.83, 0.74, 5.80, -3.49, 4.02, 9.80, 10.00, 4.27, 9.84, 0.15, -8.99,
            -6.02, -5.31,
        ]);
        let a = FixedSizeListArray::try_new(&values, 5).unwrap();

        let (u, sigma, vt) = a.svd().unwrap();
        // Results obtained from `numpy.linalg.svd()`.
        let expected_u = vec![
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
        ];
        assert_relative_eq!(
            as_primitive_array::<Float32Type>(u.values().as_ref()).values(),
            expected_u.as_slice(),
            epsilon = 0.0001,
        );

        assert_relative_eq!(
            sigma.values(),
            vec![27.46873242, 22.64318501, 8.55838823, 5.9857232, 2.01489966].as_slice(),
            epsilon = 0.0001,
        );

        // Obtained from `numpy.linagl.svd()`.
        let expected_vt = vec![
            -0.25138279,
            -0.39684555,
            -0.69215101,
            -0.36617044,
            -0.40763524,
            0.81483669,
            0.3586615,
            -0.24888801,
            -0.36859354,
            -0.09796257,
            -0.26061851,
            0.70076821,
            -0.22081145,
            0.38593848,
            -0.49325014,
            0.39672378,
            -0.45071124,
            0.25132115,
            0.4342486,
            -0.62268407,
            -0.21802776,
            0.14020995,
            0.58911945,
            -0.62652825,
            -0.43955169,
        ];
        assert_relative_eq!(
            as_primitive_array::<Float32Type>(vt.values().as_ref()).values(),
            expected_vt.as_slice(),
            epsilon = 0.0001,
        )
    }

    #[test]
    fn test_matrix_dot() {
        // A[2,3]
        let a_data = Arc::new(Float32Array::from_iter((1..=6).map(|v| v as f32)));
        let a = MatrixView::new(a_data, 3);

        // B[3,2]
        let b_data = Arc::new(Float32Array::from_iter_values([
            2.0, 3.0, 6.0, 7.0, 10.0, 11.0,
        ]));
        let b = MatrixView::new(b_data, 2);

        let c = a.dot(&b).unwrap();
        assert_relative_eq!(
            c.data.as_ref().values(),
            vec![44.0, 50.0, 98.0, 113.0].as_slice(),
        );
    }

    #[test]
    fn test_dot_on_transposed_mat() {
        // A[2,3]
        let a_data = Arc::new(Float32Array::from_iter((1..=6).map(|v| v as f32)));
        let a = MatrixView::new(a_data, 3);

        // B[3,2]
        let b_data = Arc::new(Float32Array::from_iter_values([
            2.0, 3.0, 6.0, 7.0, 10.0, 11.0,
        ]));
        let b = MatrixView::new(b_data, 2);

        let c_t = b.transpose().dot(&a.transpose()).unwrap();
        assert_relative_eq!(
            c_t.data.as_ref().values(),
            vec![44.0, 98.0, 50.0, 113.0].as_slice(),
        );
    }
}
