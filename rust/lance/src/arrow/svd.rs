use std::sync::Arc;

use lance_linalg::matrix::MatrixView;
use arrow_array::Float32Array;

#[allow(unused_imports)]
#[cfg(target_os = "macos")]
use accelerate_src;

#[allow(unused_imports)]
#[cfg(any(target_os = "linux", target_os = "windows"))]
use openblas_src;

use crate::{Error, Result};

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
            return Err(Error::Arrow {
                message: "Failed to compute SVD".to_string(),
            });
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
            return Err(Error::Arrow {
                message: "Failed to compute SVD".to_string(),
            });
        }

        let u_values = Arc::new(Float32Array::from_iter_values(u));
        let u = Self::new(u_values, m as usize).transpose();
        let vt_values = Arc::new(Float32Array::from_iter_values(vt));
        let vt = Self::new(vt_values, n as usize).transpose();
        let sigma = Float32Array::from_iter_values(sigma);
        Ok((u, sigma, vt))
    }
}
#[cfg(test)]
mod tests {
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
        let expected_u = [
            -0.59114238,
            0.263_167_8,
            0.35543017,
            0.31426436,
            0.22993832,
            0.550_753_2,
            -0.397_566_8,
            0.24379903,
            -0.22239,
            -0.753_466_1,
            -0.363_589_7,
            0.18203479,
            -0.03347897,
            -0.600_272_6,
            -0.45083927,
            0.23344966,
            -0.30547573,
            0.536_173_3,
            -0.4297069,
            0.236_166_8,
            -0.68586286,
            0.331_86,
            0.16492763,
            -0.38966287,
            -0.46974792,
            -0.3508914,
            0.3874446,
            0.15873556,
            -0.51825744,
            -0.46077223,
            0.29335876,
            0.576_262_1,
            -0.02085292,
            0.37907767,
            -0.6525516,
            0.10910681,
        ];
        u.data()
            .values()
            .iter()
            .zip(expected_u.iter())
            .for_each(|(a, b)| {
                assert_relative_eq!(a, b, epsilon = 0.0001);
            });

        let expected = vec![27.468_733, 22.643_185, 8.558_388, 5.985_723, 2.014_899_7];
        sigma.values().iter().zip(expected).for_each(|(&a, b)| {
            assert_relative_eq!(a, b, epsilon = 0.0001);
        });

        // Obtained from `numpy.linagl.svd()`.
        let expected_vt = vec![
            -0.251_382_8,
            -0.39684555,
            -0.692_151,
            -0.36617044,
            -0.40763524,
            0.814_836_7,
            0.3586615,
            -0.24888801,
            -0.36859354,
            -0.09796257,
            -0.260_618_5,
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
        vt.data()
            .values()
            .iter()
            .zip(expected_vt)
            .for_each(|(&a, b)| {
                assert_relative_eq!(a, b, epsilon = 0.0001);
            });
    }
}
