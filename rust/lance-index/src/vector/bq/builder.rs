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
use ndarray::{s, Axis};
use num_traits::{AsPrimitive, FromPrimitive};
use rand_distr::Distribution;
use snafu::location;

use crate::vector::bq::storage::{
    RabitQuantizationMetadata, RabitQuantizationStorage, RABIT_CODE_COLUMN, RABIT_METADATA_KEY,
};
use crate::vector::bq::transform::{ADD_FACTORS_FIELD, SCALE_FACTORS_FIELD};
use crate::vector::bq::RQBuildParams;
use crate::vector::quantizer::{Quantization, Quantizer, QuantizerBuildParams};

/// Build parameters for RabitQuantizer.
///
/// num_bits: the number of bits per dimension.
pub struct RabitBuildParams {
    pub num_bits: u8,
}

impl Default for RabitBuildParams {
    fn default() -> Self {
        Self { num_bits: 1 }
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

impl RabitQuantizer {
    pub fn new<T: ArrowFloatType>(num_bits: u8, dim: i32) -> Self {
        // we don't need to calculate the inverse of P,
        // because the inverse of an orthogonal matrix is still an orthogonal matrix,
        // we just generate the inverse of P here.
        let inv_p = random_orthogonal::<T>(dim as usize);
        let (inv_p, _) = inv_p.into_raw_vec_and_offset();

        let inv_p = match T::FLOAT_TYPE {
            FloatType::Float16 => {
                let inv_p = T::ArrayType::from(inv_p);
                FixedSizeListArray::try_new_from_values(inv_p, dim).unwrap()
            }
            FloatType::Float32 => {
                let inv_p = T::ArrayType::from(inv_p);
                FixedSizeListArray::try_new_from_values(inv_p, dim).unwrap()
            }
            FloatType::Float64 => {
                let inv_p = T::ArrayType::from(inv_p);
                FixedSizeListArray::try_new_from_values(inv_p, dim).unwrap()
            }
            _ => unimplemented!("RabitQ does not support data type: {:?}", T::FLOAT_TYPE),
        };

        let metadata = RabitQuantizationMetadata {
            inv_p: Some(inv_p),
            inv_p_position: 0,
            num_bits,
            packed: false,
        };
        Self { metadata }
    }

    fn inv_p<T: ArrowFloatType>(&self) -> ndarray::ArrayView2<T::Native> {
        let inv_p = self.metadata.inv_p.as_ref().unwrap();
        let dim = inv_p.len();

        let inv_p = inv_p
            .values()
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .unwrap()
            .as_slice();
        ndarray::ArrayView2::from_shape((dim, dim), inv_p).unwrap()
    }

    pub fn dim(&self) -> usize {
        self.metadata
            .inv_p
            .as_ref()
            .map(|inv_p| inv_p.len())
            .unwrap_or(0)
    }

    // compute the dot product of v_q * v_r
    pub fn codes_res_dot_dists<T: ArrowFloatType>(
        &self,
        residual_vectors: &FixedSizeListArray,
    ) -> Result<Vec<f32>>
    where
        T::Native: AsPrimitive<f32>,
    {
        if residual_vectors.value_length() as usize != self.dim() {
            return Err(Error::invalid_input(
                format!(
                    "Vector dimension mismatch: {} != {}",
                    residual_vectors.value_length(),
                    self.dim()
                ),
                location!(),
            ));
        }

        // convert the vector to a dxN matrix
        let vec_mat = ndarray::ArrayView2::from_shape(
            (residual_vectors.len(), self.dim()),
            residual_vectors
                .values()
                .as_any()
                .downcast_ref::<T::ArrayType>()
                .unwrap()
                .as_slice(),
        )
        .map_err(|e| Error::invalid_input(e.to_string(), location!()))?;
        let vec_matrix = vec_mat.t();

        let transformed_vectors = self.inv_p::<T>().dot(&vec_matrix);
        let sqrt_dim = (self.dim() as f32).sqrt();
        let norm_dists = transformed_vectors
            .mapv(|v| v.as_().abs())
            .sum_axis(Axis(0))
            / sqrt_dim;
        debug_assert_eq!(norm_dists.len(), residual_vectors.len());
        Ok(norm_dists.to_vec())
    }

    // compute the dot product of v_q * c
    // v_q * c = v_q * (v-v_r) = v_q * v - v_q * v_r
    // we already know v_q * v_r, so we just need to compute v_q * v
    // this avoids copying centroids
    // pub fn codes_dot_centroids(
    //     &self,
    //     codes: &FixedSizeListArray,
    //     vectors: &FixedSizeListArray,
    //     dot_vq_res: &PrimitiveArray<Float32Type>,
    // ) -> Result<ArrayRef> {
    //     // dot(v_q, v) = dot(P^{-1} * v_q, P^{-1} * v)
    //     // P^{-1} * v_q is just the vector v_h where:
    //     // [v_h]_i = (2*codes[i] - 1) / sqrt(dim)
    //     let n = vectors.len();
    //     let inv_sqrt_dim = 1.0 / (self.dim() as f32).sqrt();
    //     // we now only support 1 bit per dimension for RQ
    //     debug_assert_eq!(self.dim(), self.code_dim());
    //     let mut vecs_h = Vec::with_capacity(n * self.dim());
    //     codes
    //         .values()
    //         .as_primitive::<UInt8Type>()
    //         .values()
    //         .chunks_exact(self.code_dim() / 8)
    //         .for_each(|vec| {
    //             for &byte in vec.iter() {
    //                 for j in 0..8 {
    //                     let bit = (byte >> j) & 1;
    //                     if bit == 1 {
    //                         vecs_h.push(inv_sqrt_dim);
    //                     } else {
    //                         vecs_h.push(-inv_sqrt_dim);
    //                     }
    //                 }
    //             }
    //         });

    //     debug_assert_eq!(vecs_h.len(), n * self.dim());
    //     let mut vecs_h = ndarray::Array2::from_shape_vec((n, self.dim()), vecs_h)
    //         .map_err(|e| Error::invalid_input(e.to_string(), location!()))?;

    //     let vectors = ndarray::ArrayView2::from_shape(
    //         (n, self.dim()),
    //         vectors.values().as_primitive::<Float32Type>().values(),
    //     )
    //     .map_err(|e| Error::invalid_input(e.to_string(), location!()))?;
    //     let vectors = vectors.t();
    //     let rotated_vectors = self.metadata.inv_p.dot(&vectors);
    //     let rotated_vectors = rotated_vectors.t();

    //     vecs_h.zip_mut_with(&rotated_vectors, |f, &v| *f = *f * v);
    //     debug_assert_eq!(vecs_h.shape(), &[n, self.dim()]);
    //     let dot_vq_v = vecs_h.sum_axis(Axis(1));
    //     debug_assert_eq!(dot_vq_v.len(), n);
    //     let dot_vq_v = Float32Array::from(dot_vq_v.to_vec());
    //     let dot_vq_c = arrow_arith::numeric::sub(&dot_vq_v, dot_vq_res)?;
    //     Ok(dot_vq_c)
    // }

    fn transform<T: ArrowFloatType>(
        &self,
        residual_vectors: &FixedSizeListArray,
    ) -> Result<ArrayRef>
    where
        T::Native: AsPrimitive<f32>,
    {
        debug_assert_eq!(
            self.metadata.num_bits, 1,
            "RQ only supports 1 bit per element for now"
        );
        debug_assert_eq!(residual_vectors.value_length(), self.dim() as i32);
        debug_assert_eq!(self.code_dim(), self.dim());

        // we don't need to normalize the residual vectors,
        // because the signal of P^{-1} x v_r is the same as P^{-1} x v_r / ||v_r||
        let n = residual_vectors.len();
        let vectors = ndarray::ArrayView2::from_shape(
            (n, self.dim()),
            residual_vectors
                .values()
                .as_any()
                .downcast_ref::<T::ArrayType>()
                .unwrap()
                .as_slice(),
        )
        .map_err(|e| Error::invalid_input(e.to_string(), location!()))?;
        // let vectors = vectors.mapv(|v| v.as_());
        let vectors = vectors.t();
        let rotated_vectors = self.inv_p::<T>().dot(&vectors);

        let quantized_vectors = rotated_vectors.t().mapv(|v| v.as_().is_sign_positive());
        let bv: BitVec<u8, Lsb0> = BitVec::from_iter(quantized_vectors);

        let codes = UInt8Array::from(bv.into_vec());
        debug_assert_eq!(codes.len(), n * self.code_dim() / 8);
        Ok(Arc::new(FixedSizeListArray::try_new_from_values(
            codes,
            self.code_dim() as i32 / 8, // num_bits -> num_bytes
        )?))
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
        let q = match data.as_fixed_size_list().value_type() {
            DataType::Float16 => {
                Self::new::<Float16Type>(params.num_bits, data.as_fixed_size_list().value_length())
            }
            DataType::Float32 => {
                Self::new::<Float32Type>(params.num_bits, data.as_fixed_size_list().value_length())
            }
            DataType::Float64 => {
                Self::new::<Float64Type>(params.num_bits, data.as_fixed_size_list().value_length())
            }
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
        self.dim() * self.metadata.num_bits as usize
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
                self.code_dim() as i32 / 8, // num_bits -> num_bytes
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

fn random_normal_matrix(n: usize) -> ndarray::Array2<f32> {
    let mut rng = rand::rng();
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    ndarray::Array2::from_shape_simple_fn((n, n), || normal.sample(&mut rng))
}

fn householder_qr(a: ndarray::Array2<f32>) -> (ndarray::Array2<f32>, ndarray::Array2<f32>) {
    let (m, n) = a.dim();
    let mut q = ndarray::Array2::eye(m);
    let mut r = a.clone();

    for k in 0..n.min(m - 1) {
        let mut x = r.slice(s![k.., k]).to_owned();
        let x_norm = x.dot(&x).sqrt();

        if x_norm < f32::EPSILON {
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

    // cast f32 matrix to T::Native matrix
    q.mapv(|v| T::Native::from_f32(v).unwrap())
}
