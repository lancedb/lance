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

use std::sync::Arc;

use arrow::array::{as_primitive_array, Float32Builder};
use arrow_array::{Array, FixedSizeListArray, Float32Array};
use arrow_schema::DataType;
use rand::{distributions::Standard, rngs::SmallRng, seq::IteratorRandom, Rng, SeedableRng};

#[allow(unused_imports)]
#[cfg(all(feature = "opq", target_os = "macos"))]
use accelerate_src;

#[allow(unused_imports)]
#[cfg(all(feature = "opq", any(target_os = "linux", target_os = "windows")))]
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
        let data = Arc::new(Float32Array::from_iter_values(
            (&mut rng)
                .sample_iter(Standard)
                .take(num_columns * num_rows),
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

    /// Returns a row at index `i`. Returns `None` if the index is out of bound.
    ///
    /// # Panics if the matrix is transposed.
    pub fn row(&self, i: usize) -> Option<&[f32]> {
        assert!(
            !self.transpose,
            "Centroid is not defined for transposed matrix."
        );
        if i >= self.num_rows() {
            None
        } else {
            let dim = self.num_columns();
            Some(&self.data.values()[i * dim..(i + 1) * dim])
        }
    }

    /// Compute the centroid from all the rows. Returns `None` if this matrix is empty.
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
        // Add all rows with only one memory allocation.
        let mut sum = vec![0_f64; dim];
        // TODO: can SIMD work better here? Is it memory-bandwidth bound?.
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
    #[cfg(feature = "opq")]
    pub fn dot(&self, rhs: &Self) -> Result<Self> {
        use cblas::{sgemm, Layout, Transpose};

        let m = self.num_rows() as i32;
        let k = self.num_columns() as i32;
        let n = rhs.num_columns() as i32;
        if self.num_columns() != rhs.num_rows() {
            return Err(Error::Arrow {
                message: format!(
                    "MatMul dimension mismatch: A({m}x{k}) * B({}x{n}",
                    rhs.num_rows()
                ),
            });
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
        assert!(
            !self.transpose,
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
            builder.append_slice(s.values());
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
            return Err(Error::Arrow {
                message: format!(
                    "Only support convert f32 FixedSizeListArray to MatrixView, got {}",
                    fsl.data_type()
                ),
            });
        }
        let values = fsl.values();
        Ok(Self {
            data: Arc::new(as_primitive_array(values.as_ref()).clone()),
            num_columns: fsl.value_length() as usize,
            transpose: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    #[cfg(feature = "opq")]
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    #[cfg(feature = "opq")]
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
        let expected = vec![44.0, 50.0, 98.0, 113.0];
        c.data.values().iter().zip(expected).for_each(|(&a, b)| {
            assert_relative_eq!(a, b, epsilon = 0.0001);
        });
    }

    #[test]
    #[cfg(feature = "opq")]
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
        let expected = vec![44.0, 98.0, 50.0, 113.0];
        c_t.data.values().iter().zip(expected).for_each(|(&a, b)| {
            assert_relative_eq!(a, b, epsilon = 0.0001);
        });
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
