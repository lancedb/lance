// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::{Float16Type, Float32Type, Float64Type};
use arrow_array::{Array, ArrayRef, ArrowPrimitiveType, FixedSizeListArray, UInt8Array};
use arrow_schema::{DataType, Field};
use bitvec::prelude::{BitVec, Lsb0};
use deepsize::DeepSizeOf;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use num_traits::AsPrimitive;
use rand_distr::Distribution;
use snafu::location;

use crate::vector::bq::storage::{
    RabbitQuantizationMetadata, RabbitQuantizationStorage, RABBIT_CODE_COLUMN, RABBIT_METADATA_KEY,
};
use crate::vector::bq::RQBuildParams;
use crate::vector::quantizer::{Quantization, Quantizer, QuantizerBuildParams};

/// Build parameters for RabbitQuantizer.
///
/// num_bits: the number of bits per dimension.
pub struct RabbitBuildParams {
    pub num_bits: u8,
}

impl Default for RabbitBuildParams {
    fn default() -> Self {
        Self { num_bits: 1 }
    }
}

impl QuantizerBuildParams for RabbitBuildParams {
    fn sample_size(&self) -> usize {
        // RabbitQ doesn't need to sample any data
        0
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
pub struct RabbitQuantizer {
    metadata: RabbitQuantizationMetadata,
}

impl RabbitQuantizer {
    pub fn new(num_bits: u8, dim: i32) -> Self {
        // we don't need to calculate the inverse of P,
        // because the inverse of an orthogonal matrix is still an orthogonal matrix,
        // we just generate the inverse of P here.
        let inv_p = random_orthogonal(dim as usize);
        let metadata = RabbitQuantizationMetadata {
            inv_p: inv_p.into(),
            inv_p_position: 0,
            num_bits,
        };
        Self { metadata }
    }

    pub fn dim(&self) -> usize {
        self.metadata.inv_p.nrows()
    }

    fn transform<T: ArrowPrimitiveType>(&self, vectors: &FixedSizeListArray) -> Result<ArrayRef>
    where
        T::Native: AsPrimitive<f32>,
    {
        debug_assert_eq!(
            self.metadata.num_bits, 1,
            "RQ only supports 1 bit per element for now"
        );
        debug_assert_eq!(vectors.value_length(), self.dim() as i32);
        debug_assert_eq!(self.code_dim(), self.dim());
        let mut bv = BitVec::<u8, Lsb0>::with_capacity(vectors.len() * self.code_dim());
        vectors
            .values()
            .as_primitive::<T>()
            .values()
            .chunks_exact(self.dim())
            .for_each(|vec| {
                let col_vec = ndarray::Array2::from_shape_vec(
                    (self.dim(), 1),
                    vec.into_iter().map(|&x| x.as_()).collect(),
                )
                .unwrap();
                let col_vec: ndarray::Array2<f32> = self.metadata.inv_p.dot(&col_vec);

                // quantize this vector to a binary vector,
                // the i-th bit is 1 if the i-th element of the vector is greater than 0, otherwise 0.
                // TODO: support more than 1 bit per element
                for &v in col_vec.iter() {
                    bv.push(v > 0.0);
                }
            });

        let codes = UInt8Array::from(bv.into_vec());
        debug_assert_eq!(codes.len(), vectors.len() * self.code_dim() / 8);
        Ok(Arc::new(FixedSizeListArray::try_new_from_values(
            codes,
            self.code_dim() as i32 / 8, // num_bits -> num_bytes
        )?))
    }
}

impl Quantization for RabbitQuantizer {
    type BuildParams = RQBuildParams;
    type Metadata = RabbitQuantizationMetadata;
    type Storage = RabbitQuantizationStorage;

    fn build(
        data: &dyn Array,
        _: lance_linalg::distance::DistanceType,
        params: &Self::BuildParams,
    ) -> Result<Self> {
        Ok(Self::new(
            params.num_bits,
            data.as_fixed_size_list().value_length(),
        ))
    }

    fn retrain(&mut self, _data: &dyn Array) -> Result<()> {
        Ok(())
    }

    fn code_dim(&self) -> usize {
        self.dim() * self.metadata.num_bits as usize
    }

    fn column(&self) -> &'static str {
        RABBIT_CODE_COLUMN
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
        RABBIT_METADATA_KEY
    }

    fn quantization_type() -> crate::vector::quantizer::QuantizationType {
        crate::vector::quantizer::QuantizationType::Rabbit
    }

    fn metadata(
        &self,
        _: Option<crate::vector::quantizer::QuantizationMetadata>,
    ) -> Self::Metadata {
        self.metadata.clone()
    }

    fn from_metadata(
        metadata: &Self::Metadata,
        _: lance_linalg::distance::DistanceType,
    ) -> Result<Quantizer> {
        Ok(Quantizer::Rabbit(Self {
            metadata: metadata.clone(),
        }))
    }

    fn field(&self) -> Field {
        Field::new(
            RABBIT_CODE_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                self.code_dim() as i32 / 8, // num_bits -> num_bytes
            ),
            true,
        )
    }
}

impl TryFrom<Quantizer> for RabbitQuantizer {
    type Error = Error;

    fn try_from(quantizer: Quantizer) -> Result<Self> {
        match quantizer {
            Quantizer::Rabbit(quantizer) => Ok(quantizer),
            _ => Err(Error::invalid_input(
                "Cannot convert non-RabbitQuantizer to RabbitQuantizer",
                location!(),
            )),
        }
    }
}

impl From<RabbitQuantizer> for Quantizer {
    fn from(quantizer: RabbitQuantizer) -> Self {
        Quantizer::Rabbit(quantizer)
    }
}

fn random_normal_matrix_f32(n: usize) -> ndarray::Array2<f32> {
    let mut rng = rand::rng();
    let normal = rand_distr::Normal::new(0.0f32, 1.0f32).unwrap();
    ndarray::Array2::from_shape_simple_fn((n, n), || normal.sample(&mut rng))
}

fn gram_schmidt(a: ndarray::Array2<f32>) -> ndarray::Array2<f32> {
    let (n, m) = a.dim();
    let mut q = ndarray::Array2::<f32>::zeros((n, m));

    for j in 0..m {
        let mut v = a.column(j).to_owned();

        for i in 0..j {
            let qi = q.column(i);
            let proj = qi.dot(&v) * &qi;
            v = &v - &proj;
        }

        let norm = v.dot(&v).sqrt();
        if norm > f32::EPSILON {
            q.column_mut(j).assign(&(&v / norm));
        }
    }

    q
}

fn random_orthogonal(n: usize) -> ndarray::Array2<f32> {
    let a = random_normal_matrix_f32(n);
    gram_schmidt(a)
}
