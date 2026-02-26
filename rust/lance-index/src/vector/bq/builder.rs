// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::{Float16Type, Float32Type, Float64Type};
use arrow_array::{Array, ArrayRef, FixedSizeListArray, UInt8Array};
use arrow_schema::{DataType, Field};
use bitvec::prelude::{BitVec, Lsb0};
use deepsize::DeepSizeOf;
use lance_arrow::{ArrowFloatType, FixedSizeListArrayExt, FloatArray, FloatType};
use lance_core::{Error, Result};
use ndarray::{s, Axis, ShapeBuilder};
use num_traits::{AsPrimitive, FromPrimitive};
use rand_distr::Distribution;
use rayon::prelude::*;
use snafu::location;

use crate::vector::bq::storage::{
    RabitQuantizationMetadata, RabitQuantizationStorage, RABIT_CODE_COLUMN, RABIT_METADATA_KEY,
};
use crate::vector::bq::transform::{ADD_FACTORS_FIELD, SCALE_FACTORS_FIELD};
use crate::vector::bq::{
    rotation::{apply_fast_rotation, random_fast_rotation_signs},
    RQBuildParams, RQRotationType,
};
use crate::vector::quantizer::{Quantization, Quantizer, QuantizerBuildParams};

/// Build parameters for RabitQuantizer.
///
/// num_bits: the number of bits per dimension.
pub struct RabitBuildParams {
    pub num_bits: u8,
    pub rotation_type: RQRotationType,
}

impl Default for RabitBuildParams {
    fn default() -> Self {
        Self {
            num_bits: 1,
            rotation_type: RQRotationType::default(),
        }
    }
}

impl QuantizerBuildParams for RabitBuildParams {
    fn sample_size(&self) -> usize {
        // RabitQ doesn't need to sample any data
        0
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
pub struct RabitQuantizer {
    metadata: RabitQuantizationMetadata,
}

#[inline]
fn pack_sign_bits(codes: &mut [u8], rotated: &[f32]) {
    codes.fill(0);
    for (bit_idx, value) in rotated.iter().enumerate() {
        if value.is_sign_positive() {
            codes[bit_idx / u8::BITS as usize] |= 1u8 << (bit_idx % u8::BITS as usize);
        }
    }
}

impl RabitQuantizer {
    pub fn new<T: ArrowFloatType>(num_bits: u8, dim: i32) -> Self {
        Self::new_with_rotation::<T>(num_bits, dim, RQRotationType::default())
    }

    pub fn new_with_rotation<T: ArrowFloatType>(
        num_bits: u8,
        dim: i32,
        rotation_type: RQRotationType,
    ) -> Self {
        let code_dim = (dim * num_bits as i32) as usize;
        let metadata = match rotation_type {
            RQRotationType::Matrix => {
                // we don't need to calculate the inverse of P, just take generated Q as P^{-1}
                let rotate_mat = random_orthogonal::<T>(code_dim);
                let (rotate_mat, _) = rotate_mat.into_raw_vec_and_offset();
                let rotate_mat = match T::FLOAT_TYPE {
                    FloatType::Float16 | FloatType::Float32 | FloatType::Float64 => {
                        let rotate_mat = <T::ArrayType as FloatArray<T>>::from_values(rotate_mat);
                        FixedSizeListArray::try_new_from_values(rotate_mat, code_dim as i32)
                            .unwrap()
                    }
                    _ => unimplemented!("RabitQ does not support data type: {:?}", T::FLOAT_TYPE),
                };
                RabitQuantizationMetadata {
                    rotate_mat: Some(rotate_mat),
                    rotate_mat_position: None,
                    fast_rotation_signs: None,
                    rotation_type,
                    code_dim: code_dim as u32,
                    num_bits,
                    packed: false,
                }
            }
            RQRotationType::Fast => RabitQuantizationMetadata {
                rotate_mat: None,
                rotate_mat_position: None,
                fast_rotation_signs: Some(random_fast_rotation_signs(code_dim)),
                rotation_type,
                code_dim: code_dim as u32,
                num_bits,
                packed: false,
            },
        };
        Self { metadata }
    }

    pub fn num_bits(&self) -> u8 {
        self.metadata.num_bits
    }

    pub fn rotation_type(&self) -> RQRotationType {
        self.metadata.rotation_type
    }

    #[inline]
    fn fast_rotation_signs(&self) -> &[u8] {
        self.metadata
            .fast_rotation_signs
            .as_ref()
            .expect("RabitQ fast rotation signs missing")
            .as_slice()
    }

    #[inline]
    fn rotate_mat_flat<T: ArrowFloatType>(&self) -> &[T::Native] {
        let rotate_mat = self.metadata.rotate_mat.as_ref().unwrap();
        rotate_mat
            .values()
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .unwrap()
            .as_slice()
    }

    #[inline]
    fn rotate_mat<T: ArrowFloatType>(&'_ self) -> ndarray::ArrayView2<'_, T::Native> {
        let code_dim = self.code_dim();
        ndarray::ArrayView2::from_shape((code_dim, code_dim), self.rotate_mat_flat::<T>()).unwrap()
    }

    fn rotate_vectors<T: ArrowFloatType>(
        &self,
        vectors: ndarray::ArrayView2<'_, T::Native>,
    ) -> ndarray::Array2<f32>
    where
        T::Native: AsPrimitive<f32>,
    {
        let dim = vectors.nrows();
        let code_dim = self.code_dim();
        match self.rotation_type() {
            RQRotationType::Matrix => {
                let rotate_mat = self.rotate_mat::<T>();
                let rotate_mat = rotate_mat.slice(s![.., 0..dim]);
                rotate_mat.dot(&vectors).mapv(|v| v.as_())
            }
            RQRotationType::Fast => {
                let signs = self.fast_rotation_signs();
                let ncols = vectors.ncols();
                let mut rotated_data = vec![0.0f32; code_dim * ncols];
                rotated_data
                    .par_chunks_mut(code_dim)
                    .enumerate()
                    .for_each_init(
                        || vec![0.0f32; code_dim],
                        |scratch, (col_idx, dst)| {
                            let column = vectors.column(col_idx);
                            let input = column
                                .as_slice()
                                .expect("RabitQ input vectors should be contiguous");
                            apply_fast_rotation(input, scratch, signs);
                            dst.copy_from_slice(scratch);
                        },
                    );

                ndarray::Array2::from_shape_vec((code_dim, ncols).f(), rotated_data).unwrap()
            }
        }
    }

    pub fn dim(&self) -> usize {
        self.code_dim() / self.metadata.num_bits as usize
    }

    // compute the dot product of v_q * v_r
    pub fn codes_res_dot_dists<T: ArrowFloatType>(
        &self,
        residual_vectors: &FixedSizeListArray,
    ) -> Result<Vec<f32>>
    where
        T::Native: AsPrimitive<f32> + Sync,
    {
        let dim = self.dim();
        if residual_vectors.value_length() as usize != dim {
            return Err(Error::invalid_input(
                format!(
                    "Vector dimension mismatch: {} != {}",
                    residual_vectors.value_length(),
                    dim
                ),
                location!(),
            ));
        }

        let sqrt_dim = (dim as f32 * self.metadata.num_bits as f32).sqrt();
        let values = residual_vectors
            .values()
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .unwrap()
            .as_slice();

        match self.rotation_type() {
            RQRotationType::Matrix => {
                // convert the vector to a dxN matrix
                let vec_mat =
                    ndarray::ArrayView2::from_shape((residual_vectors.len(), dim), values)
                        .map_err(|e| Error::invalid_input(e.to_string(), location!()))?;
                let vec_mat = vec_mat.t();
                let rotated_vectors = self.rotate_vectors::<T>(vec_mat);
                let norm_dists = rotated_vectors.mapv(f32::abs).sum_axis(Axis(0)) / sqrt_dim;
                debug_assert_eq!(norm_dists.len(), residual_vectors.len());
                Ok(norm_dists.to_vec())
            }
            RQRotationType::Fast => {
                let code_dim = self.code_dim();
                let signs = self.fast_rotation_signs();
                let mut norm_dists = vec![0.0f32; residual_vectors.len()];
                norm_dists
                    .par_iter_mut()
                    .zip(values.par_chunks_exact(dim))
                    .for_each_init(
                        || vec![0.0f32; code_dim],
                        |scratch, (dst, input)| {
                            apply_fast_rotation(input, scratch, signs);
                            *dst = scratch.iter().map(|v| v.abs()).sum::<f32>() / sqrt_dim;
                        },
                    );
                Ok(norm_dists)
            }
        }
    }

    fn transform<T: ArrowFloatType>(
        &self,
        residual_vectors: &FixedSizeListArray,
    ) -> Result<ArrayRef>
    where
        T::Native: AsPrimitive<f32> + Sync,
    {
        // we don't need to normalize the residual vectors,
        // because the sign of P^{-1} * v_r is the same as P^{-1} * v_r / ||v_r||
        let n = residual_vectors.len();
        let dim = self.dim();
        debug_assert_eq!(residual_vectors.values().len(), n * dim);
        let values = residual_vectors
            .values()
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .unwrap()
            .as_slice();
        let code_dim = self.code_dim();
        let code_bytes = code_dim / u8::BITS as usize;

        match self.rotation_type() {
            RQRotationType::Matrix => {
                let vectors = ndarray::ArrayView2::from_shape((n, dim), values)
                    .map_err(|e| Error::invalid_input(e.to_string(), location!()))?;
                let vectors = vectors.t();
                let rotated_vectors = self.rotate_vectors::<T>(vectors);

                let quantized_vectors = rotated_vectors.t().mapv(|v| v.is_sign_positive());
                let bv: BitVec<u8, Lsb0> = BitVec::from_iter(quantized_vectors);

                let codes = UInt8Array::from(bv.into_vec());
                debug_assert_eq!(codes.len(), n * code_bytes);
                Ok(Arc::new(FixedSizeListArray::try_new_from_values(
                    codes,
                    code_bytes as i32, // num_bits -> num_bytes
                )?))
            }
            RQRotationType::Fast => {
                let signs = self.fast_rotation_signs();
                let mut encoded_codes = vec![0u8; n * code_bytes];
                encoded_codes
                    .par_chunks_mut(code_bytes)
                    .zip(values.par_chunks_exact(dim))
                    .for_each_init(
                        || vec![0.0f32; code_dim],
                        |scratch, (code_dst, input)| {
                            apply_fast_rotation(input, scratch, signs);
                            pack_sign_bits(code_dst, scratch);
                        },
                    );
                let codes = UInt8Array::from(encoded_codes);
                debug_assert_eq!(codes.len(), n * code_bytes);
                Ok(Arc::new(FixedSizeListArray::try_new_from_values(
                    codes,
                    code_bytes as i32,
                )?))
            }
        }
    }
}

impl Quantization for RabitQuantizer {
    type BuildParams = RQBuildParams;
    type Metadata = RabitQuantizationMetadata;
    type Storage = RabitQuantizationStorage;

    fn build(
        data: &dyn Array,
        _: lance_linalg::distance::DistanceType,
        params: &Self::BuildParams,
    ) -> Result<Self> {
        let dim = data.as_fixed_size_list().value_length() as usize;
        if !dim.is_multiple_of(u8::BITS as usize) {
            return Err(Error::invalid_input(
                "vector dimension must be divisible by 8 for IVF_RQ",
                location!(),
            ));
        }

        let q = match data.as_fixed_size_list().value_type() {
            DataType::Float16 => Self::new_with_rotation::<Float16Type>(
                params.num_bits,
                data.as_fixed_size_list().value_length(),
                params.rotation_type,
            ),
            DataType::Float32 => Self::new_with_rotation::<Float32Type>(
                params.num_bits,
                data.as_fixed_size_list().value_length(),
                params.rotation_type,
            ),
            DataType::Float64 => Self::new_with_rotation::<Float64Type>(
                params.num_bits,
                data.as_fixed_size_list().value_length(),
                params.rotation_type,
            ),
            dt => {
                return Err(Error::invalid_input(
                    format!("Unsupported data type: {:?}", dt),
                    location!(),
                ))
            }
        };
        Ok(q)
    }

    fn retrain(&mut self, _data: &dyn Array) -> Result<()> {
        Ok(())
    }

    fn code_dim(&self) -> usize {
        if self.metadata.code_dim > 0 {
            self.metadata.code_dim as usize
        } else {
            self.metadata
                .rotate_mat
                .as_ref()
                .map(|rotate_mat| rotate_mat.len())
                .unwrap_or(0)
        }
    }

    fn column(&self) -> &'static str {
        RABIT_CODE_COLUMN
    }

    fn use_residual(_: lance_linalg::distance::DistanceType) -> bool {
        true
    }

    fn quantize(&self, vectors: &dyn Array) -> Result<arrow_array::ArrayRef> {
        let vectors = vectors.as_fixed_size_list();
        match vectors.value_type() {
            DataType::Float16 => self.transform::<Float16Type>(vectors),
            DataType::Float32 => self.transform::<Float32Type>(vectors),
            DataType::Float64 => self.transform::<Float64Type>(vectors),
            value_type => Err(Error::invalid_input(
                format!("Unsupported data type: {:?}", value_type),
                location!(),
            )),
        }
    }

    fn metadata_key() -> &'static str {
        RABIT_METADATA_KEY
    }

    fn quantization_type() -> crate::vector::quantizer::QuantizationType {
        crate::vector::quantizer::QuantizationType::Rabit
    }

    fn metadata(
        &self,
        args: Option<crate::vector::quantizer::QuantizationMetadata>,
    ) -> Self::Metadata {
        let mut metadata = self.metadata.clone();
        metadata.packed = args.map(|args| args.transposed).unwrap_or_default();
        metadata
    }

    fn from_metadata(
        metadata: &Self::Metadata,
        _: lance_linalg::distance::DistanceType,
    ) -> Result<Quantizer> {
        Ok(Quantizer::Rabit(Self {
            metadata: metadata.clone(),
        }))
    }

    fn field(&self) -> Field {
        Field::new(
            RABIT_CODE_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                self.code_dim() as i32 / u8::BITS as i32, // num_bits -> num_bytes
            ),
            true,
        )
    }

    fn extra_fields(&self) -> Vec<Field> {
        vec![ADD_FACTORS_FIELD.clone(), SCALE_FACTORS_FIELD.clone()]
    }
}

impl TryFrom<Quantizer> for RabitQuantizer {
    type Error = Error;

    fn try_from(quantizer: Quantizer) -> Result<Self> {
        match quantizer {
            Quantizer::Rabit(quantizer) => Ok(quantizer),
            _ => Err(Error::invalid_input(
                "Cannot convert non-RabitQuantizer to RabitQuantizer",
                location!(),
            )),
        }
    }
}

impl From<RabitQuantizer> for Quantizer {
    fn from(quantizer: RabitQuantizer) -> Self {
        Self::Rabit(quantizer)
    }
}

fn random_normal_matrix(n: usize) -> ndarray::Array2<f64> {
    let mut rng = rand::rng();
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    ndarray::Array2::from_shape_simple_fn((n, n), || normal.sample(&mut rng))
}

// implement the householder qr decomposition referenced from https://en.wikipedia.org/wiki/Householder_transformation#QR_decomposition
fn householder_qr(a: ndarray::Array2<f64>) -> (ndarray::Array2<f64>, ndarray::Array2<f64>) {
    let (m, n) = a.dim();
    let mut q = ndarray::Array2::eye(m);
    let mut r = a;

    for k in 0..n.min(m - 1) {
        let mut x = r.slice(s![k.., k]).to_owned();
        let x_norm = x.dot(&x).sqrt();

        if x_norm < f64::EPSILON {
            continue;
        }

        // Create Householder vector
        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        x[0] += sign * x_norm;
        let u = &x / x.dot(&x).sqrt();

        // Apply Householder transformation to R
        // Compute outer product manually
        let mut u_outer = ndarray::Array2::zeros((m - k, m - k));
        for i in 0..(m - k) {
            for j in 0..(m - k) {
                u_outer[[i, j]] = u[i] * u[j];
            }
        }
        let h = ndarray::Array2::eye(m - k) - 2.0 * u_outer;

        // Apply transformation to R
        let r_block = r.slice(s![k.., k..]).to_owned();
        let h_r = h.dot(&r_block);
        r.slice_mut(s![k.., k..]).assign(&h_r);

        // Apply transformation to Q
        let q_block = q.slice(s![.., k..]).to_owned();
        let q_h = q_block.dot(&h);
        q.slice_mut(s![.., k..]).assign(&q_h);
    }

    (q, r)
}

fn random_orthogonal<T: ArrowFloatType>(n: usize) -> ndarray::Array2<T::Native>
where
    T::Native: FromPrimitive,
{
    let a = random_normal_matrix(n);
    let (q, _) = householder_qr(a);

    // cast f64 matrix to T::Native matrix
    q.mapv(|v| T::Native::from_f64(v).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use arrow::datatypes::Float32Type;
    use arrow_array::{FixedSizeListArray, Float32Array};
    use lance_linalg::distance::DistanceType;
    use rstest::rstest;

    #[rstest]
    #[case(8)]
    #[case(16)]
    #[case(32)]
    fn test_householder_qr(#[case] n: usize) {
        let a = random_normal_matrix(n);
        let (m, n) = a.dim();

        let (q, r) = householder_qr(a.clone());

        // Check Q is orthogonal: Q^T * Q should be identity
        let q_t_q = q.t().dot(&q);
        for i in 0..m {
            for j in 0..m {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(q_t_q[[i, j]], expected, epsilon = 1e-5);
            }
        }

        // Check QR decomposition: Q * R should equal original matrix
        let qr = q.dot(&r);
        for i in 0..m {
            for j in 0..n {
                assert_relative_eq!(qr[[i, j]], a[[i, j]], epsilon = 1e-5);
            }
        }

        // Check R is upper triangular
        for i in 1..n.min(m) {
            for j in 0..i {
                assert_relative_eq!(r[[i, j]], 0.0, epsilon = 1e-5);
            }
        }

        // Additional check: Q should have shape (m, m) and R should have shape (m, n)
        assert_eq!(q.dim(), (m, m));
        assert_eq!(r.dim(), (m, n));
    }

    #[test]
    fn test_rabit_quantizer_rotation_modes() {
        let fast_q = RabitQuantizer::new_with_rotation::<Float32Type>(1, 128, RQRotationType::Fast);
        assert_eq!(fast_q.rotation_type(), RQRotationType::Fast);
        assert_eq!(fast_q.dim(), 128);

        let matrix_q =
            RabitQuantizer::new_with_rotation::<Float32Type>(1, 128, RQRotationType::Matrix);
        assert_eq!(matrix_q.rotation_type(), RQRotationType::Matrix);
        assert_eq!(matrix_q.dim(), 128);
    }

    #[test]
    fn test_rabit_quantizer_requires_dim_divisible_by_8() {
        let vectors = Float32Array::from(vec![0.0f32; 4 * 30]);
        let fsl = FixedSizeListArray::try_new_from_values(vectors, 30).unwrap();
        let params = RQBuildParams::new(1);

        let err = RabitQuantizer::build(&fsl, DistanceType::L2, &params).unwrap_err();
        assert!(
            err.to_string()
                .contains("vector dimension must be divisible by 8 for IVF_RQ"),
            "{}",
            err
        );
    }
}
