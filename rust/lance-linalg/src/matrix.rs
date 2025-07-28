// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;
use std::ops::Mul;

use arrow_array::{ArrowNativeTypeOp, ArrowPrimitiveType, PrimitiveArray};

/// A matrix is a 2D array of elements.
/// It's a thin wrapper around a vector of elements.
pub struct Matrix<'a, T: ArrowPrimitiveType> {
    data: Cow<'a, [T::Native]>,
    // (row, col) without considering transposed
    // transposing won't change the internal shape, but method `shape()` will return the transposed shape
    shape: (usize, usize),
    transposed: bool,
}

impl<'a, T: ArrowPrimitiveType> Matrix<'a, T> {
    pub fn new(data: &'a [T::Native], shape: (usize, usize)) -> Self {
        Self {
            data: Cow::Borrowed(data),
            shape,
            transposed: false,
        }
    }

    pub fn new_owned(data: Vec<T::Native>, shape: (usize, usize)) -> Self {
        Self {
            data: Cow::Owned(data),
            shape,
            transposed: false,
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        if self.transposed {
            (self.shape.1, self.shape.0)
        } else {
            self.shape
        }
    }

    pub fn len(&self) -> usize {
        self.shape.0 * self.shape.1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn row(&self, i: usize) -> impl Iterator<Item = &T::Native> {
        if self.transposed {
            self.col_iter(i)
        } else {
            self.row_iter(i)
        }
    }

    pub fn col(&self, j: usize) -> impl Iterator<Item = &T::Native> {
        if self.transposed {
            self.row_iter(j)
        } else {
            self.col_iter(j)
        }
    }

    // iterate over the i-th row without considering transposed
    fn row_iter(&self, i: usize) -> MatrixVectorIter<'_, T> {
        MatrixVectorIter::new(
            self.data.as_ref(),
            i * self.shape.1,
            (i + 1) * self.shape.1,
            1,
        )
    }

    // iterate over the j-th column without considering transposed
    fn col_iter(&self, j: usize) -> MatrixVectorIter<'_, T> {
        MatrixVectorIter::new(self.data.as_ref(), j, self.len(), self.shape.1)
    }

    /// Transpose the matrix.
    /// This is a cheap operation, it doesn't copy the data.
    /// It only changes the shape and the direction of the iteration.
    pub fn transpose(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape,
            transposed: !self.transposed,
        }
    }
}

pub struct MatrixVectorIter<'a, T: ArrowPrimitiveType> {
    data: &'a [T::Native],
    current: usize,
    end: usize,
    step: usize,
}

impl<'a, T: ArrowPrimitiveType> MatrixVectorIter<'a, T> {
    pub fn new(data: &'a [T::Native], current: usize, end: usize, step: usize) -> Self {
        Self {
            data,
            current,
            end,
            step,
        }
    }
}

impl<'a, T: ArrowPrimitiveType> Iterator for MatrixVectorIter<'a, T> {
    type Item = &'a T::Native;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.end {
            None
        } else {
            let item = &self.data[self.current];
            self.current += self.step;
            Some(item)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.end - self.current) / self.step;
        (remaining, Some(remaining))
    }
}

pub trait ArrayAsMatrix<T: ArrowPrimitiveType> {
    fn as_row_vector(&self) -> Matrix<'_, T>;
    fn as_column_vector(&self) -> Matrix<'_, T>;
}

impl<T: ArrowPrimitiveType> ArrayAsMatrix<T> for PrimitiveArray<T> {
    fn as_row_vector(&self) -> Matrix<'_, T> {
        Matrix::new(self.values(), (self.len(), 1))
    }

    fn as_column_vector(&self) -> Matrix<'_, T> {
        Matrix::new(self.values(), (1, self.len()))
    }
}

impl<'a, T: ArrowPrimitiveType> Mul<&Matrix<'a, T>> for &Matrix<'a, T> {
    type Output = Matrix<'a, T>;

    fn mul(self, rhs: &Matrix<'a, T>) -> Self::Output {
        debug_assert_eq!(self.shape().1, rhs.shape().0);

        let mut result = Vec::with_capacity(self.len());
        let row = self.shape().0;
        let col = rhs.shape().1;
        for i in 0..row {
            for j in 0..col {
                let mut sum = T::Native::default();
                let lhs_row = self.row(i);
                let rhs_col = rhs.col(j);
                for (lhs, rhs) in lhs_row.zip(rhs_col) {
                    sum = sum.add_wrapping(lhs.mul_wrapping(*rhs));
                }
                result.push(sum);
            }
        }

        Matrix::new_owned(result, (row, col))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::types::UInt32Type;

    use super::*;

    #[test]
    fn test_mul() {
        // 2x3 * 3x4 -> 2x4
        // 1 2 3
        // 4 5 6
        // x
        // 7 8 9 10
        // 11 12 13 14
        // 15 16 17 18
        // =
        // 74 80 86 92
        // 173 188 203 218
        let lhs = Matrix::<UInt32Type>::new_owned(vec![1, 2, 3, 4, 5, 6], (2, 3));
        let rhs = Matrix::new_owned(vec![7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], (3, 4));
        let result = lhs.mul(&rhs);
        assert_eq!(result.data.as_ref(), &[74, 80, 86, 92, 173, 188, 203, 218]);
        assert_eq!(result.shape(), (2, 4));

        // 4x3 * 3x2 -> 4x2
        let lhs = lhs.transpose();
        let rhs = rhs.transpose();
        let result = rhs.mul(&lhs);
        assert_eq!(result.shape(), (4, 2));
        assert_eq!(result.data.as_ref(), &[74, 173, 80, 188, 86, 203, 92, 218]);
    }

    #[test]
    fn test_transpose() {
        // 1 2 3
        // 4 5 6
        let m = Matrix::<UInt32Type>::new_owned(vec![1, 2, 3, 4, 5, 6], (2, 3));
        // 1 4
        // 2 5
        // 3 6
        let transposed = m.transpose();
        let (row, col) = transposed.shape();
        assert_eq!(row, 3);
        assert_eq!(col, 2);
        for i in 0..row {
            assert_eq!(
                transposed.row(i).collect::<Vec<_>>(),
                m.col(i).collect::<Vec<_>>()
            );
        }
        for j in 0..col {
            assert_eq!(
                transposed.col(j).collect::<Vec<_>>(),
                m.row(j).collect::<Vec<_>>()
            );
        }
    }
}
