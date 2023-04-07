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

use std::sync::Arc;

use arrow::{
    array::{as_primitive_array, Float32Builder},
    datatypes::Float32Type,
};
use arrow_array::{Array, FixedSizeListArray, Float32Array};
use arrow_schema::DataType;
use rand::{distributions::Standard, rngs::SmallRng, seq::IteratorRandom, Rng, SeedableRng};

#[allow(unused_imports)]
#[cfg(target_os = "macos")]
use accelerate_src;

#[allow(unused_imports)]
#[cfg(any(target_os = "linux", target_os = "windows"))]
use openblas_src;

use crate::{Error, Result};

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
#[derive(Debug, Clone)]
pub struct MatrixView {
    /// Underneath data array.
    data: Arc<Float32Array>,

    /// The number of
    num_columns: usize,

    /// Is this matrix transposed or not.
    pub transpose: bool,
}

impl MatrixView {
    /// Create a MatrixView from a f32 data.
    pub fn new(data: Arc<Float32Array>, num_columns: usize) -> Self {
        Self {
            data,
            num_columns,
            transpose: false,
        }
    }

    /// Randomly initialize a matrix of shape `(num_rows, num_columns)`.
    pub fn random(num_rows: usize, num_columns: usize) -> Self {
        let mut rng = SmallRng::from_entropy();
        let data = Arc::new(Float32Array::from_iter(
            (&mut rng)
                .sample_iter(Standard)
                .take(num_columns * num_rows)
                .collect::<Vec<f32>>(),
        ));
        Self {
            data,
            num_columns,
            transpose: false,
        }
    }

    /// Create a identity matrix, with number of rows `n`.
    pub fn identity(n: usize) -> Self {
        let mut builder = Float32Builder::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                builder.append_value(if i == j { 1.0 } else { 0.0 });
            }
        }
        Self {
            data: Arc::new(builder.finish()),
            num_columns: n,
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

    pub fn data(&self) -> Arc<Float32Array> {
        if self.transpose {
            Arc::new(transpose(self.data.values(), self.num_rows()).into())
        } else {
            self.data.clone()
        }
    }

    pub fn row(&self, i: usize) -> Option<Float32Array> {
        if i >= self.num_rows() {
            None
        } else {
            let slice_arr = self.data.slice(i * self.num_columns(), self.num_columns());
            Some(as_primitive_array(slice_arr.as_ref()).clone())
        }
    }

    /// Compute the centroid of all the rows. Returns None if the matrix is empty.
    ///
    /// # Panics if the matrix is transposed.
    pub fn centroid(&self) -> Option<Float32Array> {
        assert!(
            !self.transpose,
            "Centroid is not defined for transposed matrix."
        );
        if self.num_rows() == 0 {
            return None;
        }
        // Scale to f64 to reduce the chance of overflow.
        let dim = self.num_columns();
        let mut sum = vec![0_f64; dim];
        // TODO: can SIMD work better here?
        // This seems to be memory-throughput bound computation.
        self.data.values().chunks(dim).for_each(|row| {
            row.iter().enumerate().for_each(|(i, v)| {
                sum[i] += *v as f64;
            })
        });
        let total = self.num_rows() as f64;
        let arr = Float32Array::from_iter(sum.iter().map(|v| (v / total) as f32));
        Some(arr)
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
                self.data.values(),
                lda,
                rhs.data.values(),
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

    /// Sample `n` rows from the matrix.
    pub fn sample(&self, n: usize) -> Self {
        let rng = SmallRng::from_entropy();
        self.sample_with(n, rng)
    }

    /// Sample `n` rows with a random generator.
    pub fn sample_with(&self, n: usize, mut rng: impl Rng) -> Self {
        assert_eq!(
            self.transpose, false,
            "Does not support sampling on transposed matrix"
        );
        if n > self.num_rows() {
            return self.clone();
        }
        let chosen = (0..self.num_rows()).choose_multiple(&mut rng, n);
        let dim = self.num_columns();
        let mut builder = Float32Builder::with_capacity(n * dim);
        for idx in chosen.iter() {
            let s = self.data.slice(idx * dim, dim);
            builder.append_slice(as_primitive_array::<Float32Type>(s.as_ref()).values());
        }
        let data = Arc::new(builder.finish());
        Self {
            data,
            num_columns: self.num_columns,
            transpose: false,
        }
    }
}

impl TryFrom<&FixedSizeListArray> for MatrixView {
    type Error = Error;

    fn try_from(fsl: &FixedSizeListArray) -> Result<Self> {
        if !matches!(fsl.value_type(), DataType::Float32) {
            return Err(Error::Arrow(format!(
                "Only support convert f32 FixedSizeListArray to MatrixView, got {}",
                fsl.data_type()
            )));
        }
        let values = fsl.values();
        Ok(Self {
            data: Arc::new(as_primitive_array(values.as_ref()).clone()),
            num_columns: fsl.value_length() as usize,
            transpose: false,
        })
    }
}

/// Single Value Decomposition.
///
/// <https://en.wikipedia.org/wiki/Singular_value_decomposition>
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

impl SingularValueDecomposition for MatrixView {
    type Matrix = Self;
    type Sigma = Float32Array;

    fn svd(&self) -> Result<(Self::Matrix, Self::Sigma, Self::Matrix)> {
        /// Sadly that the Accelerate Framework on macOS does not have LAPACKE(C)
        /// so we have to use the Fortran one which is column-major matrix.
        use lapack::sgesdd;
        use std::cmp::min;

        let m = self.num_rows() as i32;
        let n = self.num_columns() as i32;

        // Solving: A = U * Sigma * Vt
        //
        // TODO: Lapacke requires a mutable reference of `A`.
        // How can we get mutable reference without copying data?
        // Convert A to column-major matrix.
        let mut a = self.transpose().data().values().to_vec();
        // f32 array builder for matrix U.
        let mut u = vec![0.0; (m * m) as usize];
        // f32 array builder for matrix V_T
        let mut vt = vec![0.0; (n * n) as usize];
        let mut sigma = vec![0.0; n as usize];

        let mut work = vec![0_f32; 1];
        // Length of the workspace
        let lwork: i32 = -1;
        let mut iwork = vec![0; 8 * min(m, m) as usize];
        // Return value.
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

        let u_values = Arc::new(Float32Array::from_iter_values(u));
        let u = MatrixView::new(u_values, m as usize).transpose();
        let vt_values = Arc::new(Float32Array::from_iter_values(vt));
        let vt = MatrixView::new(vt_values, n as usize).transpose();
        let sigma = Float32Array::from_iter_values(sigma);
        Ok((u, sigma, vt))
    }
}



#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use approx::assert_relative_eq;

    use super::*;

    #[test]
    #[cfg(unix)]
    fn test_svd() {
        // A 6 x 5 matrix, from
        // https://www.intel.com/content/www/us/en/develop/documentation/onemkl-lapack-examples/top/least-squares-and-eigenvalue-problems/singular-value-decomposition/gesvd-function/sgesvd-example/lapacke-sgesvd-example-c-row.html
        let values = Arc::new(Float32Array::from_iter_values([
            8.79, 9.93, 9.83, 5.45, 3.16, 6.11, 6.91, 5.04, -0.27, 7.98, -9.15, -7.93, 4.86, 4.85,
            3.01, 9.57, 1.64, 8.83, 0.74, 5.80, -3.49, 4.02, 9.80, 10.00, 4.27, 9.84, 0.15, -8.99,
            -6.02, -5.31,
        ]));
        let a = MatrixView::new(values, 5);

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
        assert_relative_eq!(u.data().values(), expected_u.as_slice(), epsilon = 0.0001,);

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
        assert_relative_eq!(vt.data().values(), expected_vt.as_slice(), epsilon = 0.0001,)
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
        assert_relative_eq!(c.data.values(), vec![44.0, 50.0, 98.0, 113.0].as_slice(),);
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
        assert_relative_eq!(c_t.data.values(), vec![44.0, 98.0, 50.0, 113.0].as_slice(),);
    }

    #[test]
    fn test_sample_matrix() {
        let a_data = Arc::new(Float32Array::from_iter((1..=20).map(|v| v as f32)));
        let a = MatrixView::new(a_data, 2);

        let samples = a.sample(5);
        assert_eq!(samples.num_rows(), 5);

        let all_values: HashSet<i64> = samples.data.values().iter().map(|v| *v as i64).collect();
        assert_eq!(all_values.len(), 5 * 2);
    }

    #[test]
    fn test_transpose() {
        let a_data = Arc::new(Float32Array::from_iter((1..=12).map(|v| v as f32)));
        let a = MatrixView::new(a_data, 3);
        assert_eq!(a.num_rows(), 4);
        assert_eq!(a.num_columns(), 3);
        assert_eq!(
            a.data().values(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );

        let a_t = a.transpose();
        assert_eq!(a_t.num_rows(), 3);
        assert_eq!(a_t.num_columns(), 4);
        assert_eq!(
            a_t.data().values(),
            &[1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0]
        );
    }

    #[test]
    fn test_centroids() {
        let data = Arc::new(Float32Array::from_iter((0..500).map(|v| v as f32)));
        let mat = MatrixView::new(data, 10);
        let centroids = mat.centroid().unwrap();
        assert_eq!(
            centroids.values(),
            (245..255).map(|v| v as f32).collect::<Vec<_>>().as_slice(),
        );
    }
}
