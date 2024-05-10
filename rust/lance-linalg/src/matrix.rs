// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Linear Algebra on Arrow Array
//!

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, ArrowPrimitiveType, FixedSizeListArray};
use arrow_schema::{ArrowError, DataType};
use lance_arrow::{ArrowFloatType, FixedSizeListArrayExt, FloatArray, FloatType};
use num_traits::{AsPrimitive, Float, FromPrimitive, ToPrimitive};
use rand::{distributions::Standard, rngs::SmallRng, seq::IteratorRandom, Rng, SeedableRng};

use crate::kernels::normalize;
use crate::{Error, Result};

/// Transpose a matrix.
fn transpose<T: Float>(input: &[T], dimension: usize) -> Vec<T> {
    let n = input.len() / dimension;
    let mut mat = vec![T::zero(); input.len()];
    for row in 0..dimension {
        for col in 0..n {
            mat[row * n + col] = input[col * dimension + row];
        }
    }

    mat
}

/// A 2-D dense matrix on top of Arrow Arrays.
///
#[derive(Debug)]
pub struct MatrixView<T: ArrowFloatType> {
    /// Underneath data array.
    data: Arc<T::ArrayType>,

    /// The number of columns in the matrix.
    num_columns: usize,

    /// Is this matrix transposed or not.
    pub transpose: bool,
}

impl<T: ArrowFloatType> Clone for MatrixView<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            num_columns: self.num_columns,
            transpose: self.transpose,
        }
    }
}

impl<T: ArrowFloatType> MatrixView<T> {
    /// Create a MatrixView from a f32 data.
    pub fn new(data: Arc<T::ArrayType>, num_columns: impl AsPrimitive<usize>) -> Self {
        Self {
            data,
            num_columns: num_columns.as_(),
            transpose: false,
        }
    }

    /// Randomly initialize a matrix of shape `(num_rows, num_columns)`.
    pub fn random(num_rows: usize, num_columns: usize) -> Self {
        let mut rng = SmallRng::from_entropy();
        let data = Arc::new(T::ArrayType::from(
            (&mut rng)
                .sample_iter(Standard)
                .take(num_columns * num_rows)
                .collect::<Vec<f32>>()
                .iter()
                .map(|&v| T::Native::from_f32(v).unwrap())
                .collect::<Vec<_>>(),
        ));
        Self {
            data,
            num_columns,
            transpose: false,
        }
    }

    /// Create a identity matrix, with number of rows `n`.
    pub fn identity(n: usize) -> Self {
        let mut builder = Vec::<T::Native>::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                builder.push(if i == j {
                    T::Native::from_f32(1.0).unwrap()
                } else {
                    T::Native::from_f32(0.0).unwrap()
                });
            }
        }
        Self {
            data: Arc::new(builder.into()),
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

    /// Number of the columns (dimension) in the matrix.
    pub fn num_columns(&self) -> usize {
        if self.transpose {
            self.data.len() / self.num_columns
        } else {
            self.num_columns
        }
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.num_columns()
    }

    pub fn data(&self) -> Arc<T::ArrayType> {
        if self.transpose {
            Arc::new(transpose(self.data.as_slice(), self.num_rows()).into())
        } else {
            self.data.clone()
        }
    }

    /// Returns a row at index `i`. Returns `None` if the index is out of bound.
    ///
    /// # Panics if the matrix is transposed.
    pub fn row_ref(&self, i: usize) -> Option<&[T::Native]> {
        assert!(
            !self.transpose,
            "Centroid is not defined for transposed matrix."
        );
        if i >= self.num_rows() {
            None
        } else {
            let dim = self.num_columns();
            Some(&self.data.as_slice()[i * dim..(i + 1) * dim])
        }
    }

    pub fn row(&self, i: usize) -> Option<ArrayRef> {
        assert!(
            !self.transpose,
            "Centroid is not defined for transposed matrix."
        );
        if i >= self.num_rows() {
            None
        } else {
            let dim = self.num_columns();
            Some(self.data.slice(i * dim, dim))
        }
    }

    /// Compute the centroid from all the rows. Returns `None` if this matrix is empty.
    ///
    /// # Panics if the matrix is transposed.
    pub fn centroid(&self) -> Option<T::ArrayType> {
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
        self.data.as_slice().chunks(dim).for_each(|row| {
            row.iter().enumerate().for_each(|(i, v)| {
                sum[i] += v.to_f64().unwrap();
            })
        });
        let total = self.num_rows() as f64;
        let arr: T::ArrayType = sum
            .iter()
            .map(|v| T::Native::from_f64(v / total).unwrap())
            .collect::<Vec<T::Native>>()
            .into();

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

    pub fn normalize(&self) -> Self {
        let data = self
            .data
            .as_slice()
            .chunks(self.num_columns)
            .flat_map(normalize)
            .collect::<Vec<_>>();
        let data = Arc::new(data.into());
        Self {
            data,
            num_columns: self.num_columns,
            transpose: self.transpose,
        }
        // todo!("normalize")
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
            return Self {
                data: self.data.clone(),
                num_columns: self.num_columns,
                transpose: self.transpose,
            };
        }
        let chosen = (0..self.num_rows()).choose_multiple(&mut rng, n);
        let dim = self.num_columns();
        let mut builder = Vec::with_capacity(n * dim);
        for idx in chosen.iter() {
            builder.extend(self.data.as_slice()[idx * dim..(idx + 1) * dim].iter());
        }
        let data = Arc::new(builder.into());
        Self {
            data,
            num_columns: self.num_columns,
            transpose: false,
        }
    }

    pub fn iter(&self) -> MatrixRowIter<T> {
        MatrixRowIter {
            data: self,
            cur_idx: 0,
        }
    }
}

impl<T: ArrowFloatType> TryFrom<MatrixView<T>> for FixedSizeListArray {
    type Error = Error;

    fn try_from(value: MatrixView<T>) -> Result<Self> {
        Self::try_new_from_values(value.data.as_ref().clone(), value.num_columns as i32)
    }
}

impl<T: ArrowFloatType + ArrowPrimitiveType> TryFrom<&FixedSizeListArray> for MatrixView<T> {
    type Error = Error;

    fn try_from(value: &FixedSizeListArray) -> Result<Self> {
        // TODO: move this check to FloatType?
        match (value.value_type(), T::FLOAT_TYPE) {
            (DataType::Float16, FloatType::Float16)
            | (DataType::Float32, FloatType::Float32)
            | (DataType::Float64, FloatType::Float64) => {}
            _ => {
                return Err(ArrowError::CastError(format!(
                    "Can not convert from {}",
                    value.value_type(),
                )))
            }
        }
        // Use downcast instead of `as_primitive()` to get away from the type check.
        let data: Arc<T::ArrayType> = Arc::new(
            value
                .values()
                .as_any()
                .downcast_ref::<T::ArrayType>()
                .ok_or(Error::CastError(format!(
                    "Can not cast {} to {}",
                    value.value_type(),
                    T::FLOAT_TYPE,
                )))?
                .clone(),
        );
        Ok(Self {
            data,
            num_columns: value.value_length() as usize,
            transpose: false,
        })
    }
}

/// Iterator over the matrix one row at a time.
pub struct MatrixRowIter<'a, T: ArrowFloatType> {
    data: &'a MatrixView<T>,
    cur_idx: usize,
}

impl<'a, T: ArrowFloatType> Iterator for MatrixRowIter<'a, T> {
    type Item = &'a [T::Native];

    fn next(&mut self) -> Option<Self::Item> {
        let cur_idx = self.cur_idx;
        self.cur_idx += 1;
        self.data.row_ref(cur_idx)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use arrow_array::types::{Float32Type, Float64Type};
    use arrow_array::Float32Array;

    use super::*;

    #[test]
    fn test_sample_matrix() {
        let a_data = Arc::new(Float32Array::from_iter((1..=20).map(|v| v as f32)));
        let a: MatrixView<Float32Type> = MatrixView::new(a_data, 2);

        let samples = a.sample(5);
        assert_eq!(samples.num_rows(), 5);

        let all_values: HashSet<i64> = samples.data.values().iter().map(|v| *v as i64).collect();
        assert_eq!(all_values.len(), 5 * 2);
    }

    #[test]
    fn test_transpose() {
        let a_data = Arc::new(Float32Array::from_iter((1..=12).map(|v| v as f32)));
        let a = MatrixView::<Float32Type>::new(a_data, 3);
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
        let mat: MatrixView<Float32Type> = MatrixView::new(data, 10);
        let centroids = mat.centroid().unwrap();
        assert_eq!(
            centroids.values(),
            (245..255).map(|v| v as f32).collect::<Vec<_>>().as_slice(),
        );
    }

    #[test]
    fn test_from_fsl() {
        let data = Float32Array::from_iter((0..500).map(|v| v as f32));
        let fsl = FixedSizeListArray::try_new_from_values(data.clone(), 50).unwrap();
        let mat = MatrixView::<Float32Type>::try_from(&fsl).unwrap();
        assert_eq!(mat.num_rows(), 10);
        assert_eq!(mat.num_columns(), 50);
        assert_eq!(mat.data().values(), data.values());

        assert!(MatrixView::<Float64Type>::try_from(&fsl).is_err());
    }
}
