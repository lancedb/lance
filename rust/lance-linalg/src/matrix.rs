// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;
use std::ops::Mul;

use arrow_array::{ArrowNativeTypeOp, ArrowPrimitiveType, PrimitiveArray};

pub struct Matrix<'a, T: ArrowPrimitiveType> {
    data: Cow<'a, [T::Native]>,
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
            self.data[i..self.len()].iter().step_by(self.shape.1)
        } else {
            self.data[i * self.shape.1..(i + 1) * self.shape.1]
                .iter()
                .step_by(1) // call step_by(1) to make the type match
        }
    }

    pub fn col(&self, j: usize) -> impl Iterator<Item = &T::Native> {
        if self.transposed {
            self.data[j * self.shape.1..(j + 1) * self.shape.1]
                .iter()
                .step_by(1)
        } else {
            self.data[j..self.len()].iter().step_by(self.shape.1)
        }
    }

    pub fn transpose(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape,
            transposed: !self.transposed,
        }
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
