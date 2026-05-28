// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Sub;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::{Float16Type, Float32Type, Float64Type, UInt8Type, UInt64Type};
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, UInt8Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, SchemaRef};
use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use deepsize::DeepSizeOf;
use itertools::Itertools;
use lance_arrow::{ArrowFloatType, FixedSizeListArrayExt, FloatArray, RecordBatchExt};
use lance_core::{Error, ROW_ID, Result};
use lance_file::previous::reader::FileReader as PreviousFileReader;
use lance_linalg::distance::{DistanceType, Dot};
use lance_linalg::simd::{
    self,
    dist_table::{BATCH_SIZE, PERM0, PERM0_INVERSE},
};
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "loongarch64"
))]
use lance_linalg::simd::{SIMD, f32::f32x16};
use lance_table::utils::LanceIteratorExtension;
use num_traits::AsPrimitive;
use prost::Message;
use serde::{Deserialize, Serialize};

use crate::frag_reuse::FragReuseIndex;
use crate::pb;
use crate::vector::bq::RQRotationType;
use crate::vector::bq::rotation::{apply_fast_rotation, apply_fast_rotation_in_place};
use crate::vector::bq::transform::{ADD_FACTORS_COLUMN, SCALE_FACTORS_COLUMN};
use crate::vector::pq::storage::transpose;
use crate::vector::quantizer::{QuantizerMetadata, QuantizerStorage};
use crate::vector::storage::{DistCalculator, QueryResidual, VectorStore};

pub const RABIT_METADATA_KEY: &str = "lance:rabit";
pub const RABIT_CODE_COLUMN: &str = "_rabit_codes";
pub const SEGMENT_LENGTH: usize = 4;
pub const SEGMENT_NUM_CODES: usize = 1 << SEGMENT_LENGTH;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RabitQuantizationMetadata {
    // this rotate matrix is large, and lance index would store all metadata in schema metadata,
    // which is in JSON format, so we skip it in serialization and deserialization, and store it
    // in the global buffer, which is a binary format (protobuf for now) for efficiency.
    #[serde(skip)]
    pub rotate_mat: Option<FixedSizeListArray>,
    #[serde(default)]
    pub rotate_mat_position: Option<u32>,
    #[serde(default)]
    pub fast_rotation_signs: Option<Vec<u8>>,
    #[serde(default = "default_rotation_type_compat")]
    pub rotation_type: RQRotationType,
    #[serde(default)]
    pub code_dim: u32,
    pub num_bits: u8,
    pub packed: bool,
}

fn default_rotation_type_compat() -> RQRotationType {
    // Older metadata does not have this field and always used dense matrices.
    RQRotationType::Matrix
}

impl RabitQuantizationMetadata {
    fn code_dim(&self) -> usize {
        if self.code_dim > 0 {
            self.code_dim as usize
        } else {
            self.rotate_mat
                .as_ref()
                .map(|rotate_mat| rotate_mat.len())
                .unwrap_or_default()
        }
    }

    fn rotate_vector_with_residual_into(
        &self,
        vector: &dyn Array,
        residual_centroid: Option<&dyn Array>,
        output: &mut [f32],
    ) {
        debug_assert_eq!(output.len(), self.code_dim());
        match self.rotation_type {
            RQRotationType::Matrix => {
                let rotate_mat = self
                    .rotate_mat
                    .as_ref()
                    .expect("RabitQ dense rotation metadata not loaded");

                match rotate_mat.value_type() {
                    DataType::Float16 => {
                        RabitQuantizationStorage::rotate_query_vector_dense_into::<Float16Type>(
                            rotate_mat,
                            vector,
                            residual_centroid,
                            output,
                        )
                    }
                    DataType::Float32 => {
                        RabitQuantizationStorage::rotate_query_vector_dense_into::<Float32Type>(
                            rotate_mat,
                            vector,
                            residual_centroid,
                            output,
                        )
                    }
                    DataType::Float64 => {
                        RabitQuantizationStorage::rotate_query_vector_dense_into::<Float64Type>(
                            rotate_mat,
                            vector,
                            residual_centroid,
                            output,
                        )
                    }
                    dt => unimplemented!("RabitQ does not support data type: {}", dt),
                }
            }
            RQRotationType::Fast => {
                let signs = self
                    .fast_rotation_signs
                    .as_ref()
                    .expect("RabitQ fast rotation metadata not loaded");
                match vector.data_type() {
                    DataType::Float16 => RabitQuantizationStorage::rotate_query_vector_fast_into::<
                        Float16Type,
                    >(
                        signs, vector, residual_centroid, output
                    ),
                    DataType::Float32 => {
                        RabitQuantizationStorage::rotate_query_vector_fast_f32_into(
                            signs,
                            vector,
                            residual_centroid,
                            output,
                        )
                    }
                    DataType::Float64 => RabitQuantizationStorage::rotate_query_vector_fast_into::<
                        Float64Type,
                    >(
                        signs, vector, residual_centroid, output
                    ),
                    dt => unimplemented!("RabitQ does not support data type: {}", dt),
                }
            }
        }
    }
}

impl DeepSizeOf for RabitQuantizationMetadata {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.rotate_mat
            .as_ref()
            .map(|inv_p| inv_p.get_array_memory_size())
            .unwrap_or(0)
            + self
                .fast_rotation_signs
                .as_ref()
                .map(|signs| signs.len())
                .unwrap_or(0)
    }
}

#[async_trait]
impl QuantizerMetadata for RabitQuantizationMetadata {
    fn buffer_index(&self) -> Option<u32> {
        match self.rotation_type {
            RQRotationType::Matrix => self.rotate_mat_position,
            RQRotationType::Fast => None,
        }
    }

    fn set_buffer_index(&mut self, index: u32) {
        self.rotate_mat_position = Some(index);
    }

    fn parse_buffer(&mut self, bytes: Bytes) -> Result<()> {
        if self.rotation_type != RQRotationType::Matrix {
            return Ok(());
        }
        debug_assert!(!bytes.is_empty());
        let codebook_tensor: pb::Tensor = pb::Tensor::decode(bytes)?;
        self.rotate_mat = Some(FixedSizeListArray::try_from(&codebook_tensor)?);
        if self.code_dim == 0 {
            self.code_dim = self
                .rotate_mat
                .as_ref()
                .map(|rotate_mat| rotate_mat.len() as u32)
                .unwrap_or(0);
        }
        Ok(())
    }

    fn extra_metadata(&self) -> Result<Option<Bytes>> {
        match self.rotation_type {
            RQRotationType::Matrix => {
                if let Some(inv_p) = &self.rotate_mat {
                    let inv_p_tensor = pb::Tensor::try_from(inv_p)?;
                    let mut bytes = BytesMut::new();
                    inv_p_tensor.encode(&mut bytes)?;
                    Ok(Some(bytes.freeze()))
                } else {
                    Ok(None)
                }
            }
            RQRotationType::Fast => Ok(None),
        }
    }

    async fn load(reader: &PreviousFileReader) -> Result<Self> {
        let metadata_str = reader
            .schema()
            .metadata
            .get(RABIT_METADATA_KEY)
            .ok_or(Error::index(format!(
                "Reading Rabit metadata: metadata key {} not found",
                RABIT_METADATA_KEY
            )))?;
        serde_json::from_str(metadata_str)
            .map_err(|_| Error::index(format!("Failed to parse index metadata: {}", metadata_str)))
    }
}

#[derive(Debug, Clone)]
pub struct RabitQuantizationStorage {
    metadata: RabitQuantizationMetadata,
    batch: RecordBatch,
    distance_type: DistanceType,

    // helper fields
    row_ids: UInt64Array,
    codes: FixedSizeListArray,
    add_factors: Float32Array,
    scale_factors: Float32Array,
}

impl DeepSizeOf for RabitQuantizationStorage {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.metadata.deep_size_of_children(context) + self.batch.get_array_memory_size()
    }
}

impl RabitQuantizationStorage {
    fn code_dim(&self) -> usize {
        self.metadata.code_dim()
    }

    fn query_factor(&self, dist_q_c: f32) -> f32 {
        match self.distance_type {
            DistanceType::L2 => dist_q_c,
            DistanceType::Cosine | DistanceType::Dot => dist_q_c - 1.0,
            _ => unimplemented!(
                "RabitQ does not support distance type: {}",
                self.distance_type
            ),
        }
    }

    fn distance_calculator_from_parts<'a>(
        &'a self,
        dim: usize,
        dist_q_c: f32,
        dist_table: Cow<'a, [f32]>,
        sum_q: f32,
    ) -> RabitDistCalculator<'a> {
        RabitDistCalculator::new(
            dim,
            self.metadata.num_bits,
            dist_table,
            sum_q,
            self.codes.values().as_primitive::<UInt8Type>().values(),
            self.add_factors.values(),
            self.scale_factors.values(),
            self.query_factor(dist_q_c),
        )
    }

    fn rotate_query_vector(&self, code_dim: usize, qr: &dyn Array) -> Vec<f32> {
        let mut output = vec![0.0f32; code_dim];
        self.rotate_query_vector_into(code_dim, qr, None, &mut output);
        output
    }

    fn rotate_query_vector_into(
        &self,
        code_dim: usize,
        qr: &dyn Array,
        residual_centroid: Option<&dyn Array>,
        output: &mut [f32],
    ) {
        debug_assert_eq!(output.len(), code_dim);
        self.metadata
            .rotate_vector_with_residual_into(qr, residual_centroid, output);
    }

    fn rotate_query_vector_dense_into<T: ArrowFloatType>(
        rotate_mat: &FixedSizeListArray,
        qr: &dyn Array,
        residual_centroid: Option<&dyn Array>,
        output: &mut [f32],
    ) where
        T::Native: AsPrimitive<f32> + Dot + Sub<Output = T::Native>,
    {
        let d = qr.len();
        let code_dim = rotate_mat.len();
        debug_assert_eq!(output.len(), code_dim);
        let rotate_mat = rotate_mat
            .values()
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .unwrap()
            .as_slice();

        let qr = qr
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .unwrap()
            .as_slice();

        if let Some(residual_centroid) = residual_centroid {
            let residual_centroid = residual_centroid
                .as_any()
                .downcast_ref::<T::ArrayType>()
                .unwrap()
                .as_slice();
            debug_assert_eq!(residual_centroid.len(), d);
            for (chunk, out) in rotate_mat.chunks_exact(code_dim).zip(output.iter_mut()) {
                let mut sum = 0.0;
                for idx in 0..d {
                    let residual = qr[idx] - residual_centroid[idx];
                    sum += chunk[idx].as_() * residual.as_();
                }
                *out = sum;
            }
        } else {
            rotate_mat
                .chunks_exact(code_dim)
                .zip(output.iter_mut())
                .for_each(|(chunk, out)| {
                    *out = lance_linalg::distance::dot(&chunk[..d], qr);
                });
        }
    }

    fn rotate_query_vector_fast_into<T: ArrowFloatType>(
        signs: &[u8],
        qr: &dyn Array,
        residual_centroid: Option<&dyn Array>,
        output: &mut [f32],
    ) where
        T::Native: AsPrimitive<f32> + Sub<Output = T::Native>,
    {
        let qr = qr
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .unwrap()
            .as_slice();

        if let Some(residual_centroid) = residual_centroid {
            let residual_centroid = residual_centroid
                .as_any()
                .downcast_ref::<T::ArrayType>()
                .unwrap()
                .as_slice();
            let input_len = qr.len().min(output.len());
            debug_assert!(residual_centroid.len() >= input_len);
            for idx in 0..input_len {
                output[idx] = (qr[idx] - residual_centroid[idx]).as_();
            }
            if input_len < output.len() {
                output[input_len..].fill(0.0);
            }
            apply_fast_rotation_in_place(output, signs);
        } else {
            apply_fast_rotation(qr, output, signs);
        }
    }

    fn rotate_query_vector_fast_f32_into(
        signs: &[u8],
        qr: &dyn Array,
        residual_centroid: Option<&dyn Array>,
        output: &mut [f32],
    ) {
        let qr = qr.as_any().downcast_ref::<Float32Array>().unwrap().values();

        if let Some(residual_centroid) = residual_centroid {
            let residual_centroid = residual_centroid
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .values();
            copy_subtract_f32(qr, residual_centroid, output);
            apply_fast_rotation_in_place(output, signs);
        } else {
            apply_fast_rotation(qr, output, signs);
        }
    }
}

#[inline]
fn copy_subtract_f32(lhs: &[f32], rhs: &[f32], output: &mut [f32]) {
    let input_len = lhs.len().min(output.len());
    debug_assert!(rhs.len() >= input_len);

    #[cfg(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "loongarch64"
    ))]
    let simd_len = input_len / f32x16::LANES * f32x16::LANES;
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "loongarch64"
    )))]
    let simd_len = 0;

    #[cfg(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "loongarch64"
    ))]
    for idx in (0..simd_len).step_by(f32x16::LANES) {
        let lhs = f32x16::from(&lhs[idx..]);
        let rhs = f32x16::from(&rhs[idx..]);
        let result = lhs - rhs;
        unsafe {
            result.store_unaligned(output.as_mut_ptr().add(idx));
        }
    }

    for idx in simd_len..input_len {
        output[idx] = lhs[idx] - rhs[idx];
    }
    if input_len < output.len() {
        output[input_len..].fill(0.0);
    }
}

pub struct RabitDistCalculator<'a> {
    dim: usize,
    // num_bits is the number of bits per dimension,
    // it's always 1 for now
    num_bits: u8,
    // n * d * num_bits / 8 bytes
    codes: &'a [u8],
    // this is a flattened 2D array of size d/4 * 16,
    // we split the query codes into d/4 chunks, each chunk is with 4 elements,
    // then dist_table[i][j] is the distance between the i-th query code and the code j
    dist_table: Cow<'a, [f32]>,
    add_factors: &'a [f32],
    scale_factors: &'a [f32],
    query_factor: f32,

    sum_q: f32,
    sqrt_d: f32,
}

impl<'a> RabitDistCalculator<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        num_bits: u8,
        dist_table: Cow<'a, [f32]>,
        sum_q: f32,
        codes: &'a [u8],
        add_factors: &'a [f32],
        scale_factors: &'a [f32],
        query_factor: f32,
    ) -> Self {
        Self {
            dim,
            num_bits,
            codes,
            dist_table,
            add_factors,
            scale_factors,
            query_factor,
            sqrt_d: (dim as f32 * num_bits as f32).sqrt(),
            sum_q,
        }
    }
}

#[inline]
fn lowbit(x: usize) -> usize {
    1 << x.trailing_zeros()
}

#[inline]
pub fn build_dist_table_direct<T: ArrowFloatType>(qc: &[T::Native]) -> Vec<f32>
where
    T::Native: AsPrimitive<f32>,
{
    // every 4 bits (SEGMENT_LENGTH) is a segment, and we need to compute the distance between the segment and all the codes
    // so there are dim/4 segments, and the number of codes is 16 (2^{SEGMENT_LENGTH}),
    // so we have dim/4 * 16 = dim * 4 elements in the dist_table
    let mut dist_table = vec![0.0; qc.len() * 4];
    build_dist_table_direct_into::<T>(qc, &mut dist_table);
    dist_table
}

fn build_dist_table_direct_into<T: ArrowFloatType>(qc: &[T::Native], dist_table: &mut [f32])
where
    T::Native: AsPrimitive<f32>,
{
    debug_assert_eq!(dist_table.len(), qc.len() * 4);
    qc.chunks_exact(SEGMENT_LENGTH)
        .zip(dist_table.chunks_exact_mut(SEGMENT_NUM_CODES))
        .for_each(|(sub_vec, dist_table)| {
            dist_table[0] = 0.0;
            build_dist_table_for_subvec::<T>(sub_vec, dist_table);
        });
}

#[inline(always)]
fn build_dist_table_for_subvec<T: ArrowFloatType>(sub_vec: &[T::Native], dist_table: &mut [f32])
where
    T::Native: AsPrimitive<f32>,
{
    // skip 0 because it's always 0
    (1..SEGMENT_NUM_CODES).for_each(|j| {
        // this is a little bit tricky,
        // j represents a subset of 4 bits, that if the i-th bit of `j` is 1,
        // then we need to add the distance of the i-th dim of the segment.
        // but we don't need to check all bits of `j`,
        // because `j` = `j - lowbit(j)` + `lowbit(j)`,
        // where `j-lowbit(j)` is less than `j`,
        // which means dist_table[j-lowbit(j)] is already computed,
        // and we can use it to compute dist_table[j]
        // for example, if j = 0b1010, then j - lowbit(j) = 0b1000,
        // and dist_table[0b1000] is already computed,
        // so dist_table[0b1010] = dist_table[0b1000] + sub_vec[LOWBIT_IDX[0b1010]];
        // where lowbit(0b1010) = 0b10, LOWBIT_IDX[0b1010] = LOWBIT_IDX[0b10] = 1.
        dist_table[j] = dist_table[j - lowbit(j)] + sub_vec[LOWBIT_IDX[j]].as_();
    })
}

// Quantize the distance table into a caller-owned buffer.
#[inline]
fn quantize_dist_table_into(dist_table: &[f32], quantized_dist_table: &mut Vec<u8>) -> (f32, f32) {
    let (qmin, qmax) = dist_table
        .iter()
        .cloned()
        .minmax_by(|a, b| a.total_cmp(b))
        .into_option()
        .unwrap();
    // this happens if the query is all zeros
    if qmin == qmax {
        quantized_dist_table.clear();
        quantized_dist_table.resize(dist_table.len(), 0);
        return (qmin, qmax);
    }
    let factor = 255.0 / (qmax - qmin);
    quantized_dist_table.clear();
    quantized_dist_table.reserve(dist_table.len());
    let spare = quantized_dist_table.spare_capacity_mut();
    for (quantized, &d) in spare[..dist_table.len()].iter_mut().zip(dist_table.iter()) {
        quantized.write(((d - qmin) * factor).round() as u8);
    }
    // SAFETY: every element in the reserved range was initialized in the loop above.
    unsafe {
        quantized_dist_table.set_len(dist_table.len());
    }

    (qmin, qmax)
}

impl DistCalculator for RabitDistCalculator<'_> {
    #[inline(always)]
    fn distance(&self, id: u32) -> f32 {
        let id = id as usize;
        let code_len = self.dim * (self.num_bits as usize) / u8::BITS as usize;
        let num_vectors = self.codes.len() / code_len;
        let dist =
            compute_single_rq_distance(self.codes, id, num_vectors, code_len, &self.dist_table);

        // distance between quantized vector and query vector
        let dist_vq_qr = (2.0 * dist - self.sum_q) / self.sqrt_d;
        dist_vq_qr * self.scale_factors[id] + self.add_factors[id] + self.query_factor
    }

    #[inline(always)]
    fn distance_all(&self, _: usize) -> Vec<f32> {
        let mut dists = Vec::new();
        let mut quantized_dists = Vec::new();
        let mut quantized_dists_table = Vec::new();
        self.distance_all_with_scratch(
            0,
            &mut dists,
            &mut quantized_dists,
            &mut quantized_dists_table,
        );
        dists
    }

    #[inline(always)]
    #[allow(clippy::uninit_vec)]
    fn distance_all_with_scratch(
        &self,
        _: usize,
        dists: &mut Vec<f32>,
        quantized_dists: &mut Vec<u16>,
        quantized_dists_table: &mut Vec<u8>,
    ) {
        let code_len = self.dim * (self.num_bits as usize) / u8::BITS as usize;
        let n = self.codes.len() / code_len;
        if n == 0 {
            dists.clear();
            quantized_dists.clear();
            return;
        }

        let (qmin, qmax) = quantize_dist_table_into(&self.dist_table, quantized_dists_table);
        let remainder = n % BATCH_SIZE;
        let simd_len = n - remainder;
        quantized_dists.clear();
        quantized_dists.reserve(simd_len);
        // SAFETY: sum_4bit_dist_table overwrites each element in the SIMD batch range.
        unsafe {
            quantized_dists.set_len(simd_len);
        }
        simd::dist_table::sum_4bit_dist_table(
            simd_len,
            code_len,
            self.codes,
            quantized_dists_table,
            quantized_dists,
        );

        let range = (qmax - qmin) / 255.0;
        let num_tables = quantized_dists_table.len() / 16;
        let sum_min = num_tables as f32 * qmin;
        dists.clear();
        dists.reserve(n);
        // SAFETY: the SIMD section below writes [0, simd_len), and the
        // remainder section writes [simd_len, n).
        unsafe {
            dists.set_len(n);
        }
        let (simd_dists, remainder_dists) = dists.split_at_mut(simd_len);
        simd_dists
            .iter_mut()
            .zip(quantized_dists.iter())
            .enumerate()
            .for_each(|(id, (dist, q_dist))| {
                let dist_vq = (*q_dist as f32) * range + sum_min;
                let dist_vq_qr = (2.0 * dist_vq - self.sum_q) / self.sqrt_d;
                *dist =
                    dist_vq_qr * self.scale_factors[id] + self.add_factors[id] + self.query_factor;
            });

        remainder_dists
            .iter_mut()
            .enumerate()
            .for_each(|(id, dist)| {
                *dist = self.distance((simd_len + id) as u32);
            });
    }
}

impl VectorStore for RabitQuantizationStorage {
    type DistanceCalculator<'a> = RabitDistCalculator<'a>;

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> &SchemaRef {
        self.batch.schema_ref()
    }

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch> + Send> {
        Ok(std::iter::once(self.batch.clone()))
    }

    fn append_batch(&self, _batch: RecordBatch, _vector_column: &str) -> Result<Self> {
        unimplemented!("RabitQ does not support append_batch")
    }

    fn len(&self) -> usize {
        self.batch.num_rows()
    }

    fn row_id(&self, id: u32) -> u64 {
        self.row_ids.value(id as usize)
    }

    fn row_ids(&self) -> impl Iterator<Item = &u64> {
        self.row_ids.values().iter()
    }

    fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    // qr = (q-c)
    #[inline(never)]
    fn dist_calculator(&self, qr: Arc<dyn Array>, dist_q_c: f32) -> Self::DistanceCalculator<'_> {
        let code_dim = self.code_dim();
        let rotated_qr = self.rotate_query_vector(code_dim, &qr);
        let dist_table = build_dist_table_direct::<Float32Type>(&rotated_qr);
        let sum_q = rotated_qr.into_iter().sum();

        self.distance_calculator_from_parts(qr.len(), dist_q_c, Cow::Owned(dist_table), sum_q)
    }

    // qr = (q-c)
    #[inline(never)]
    fn dist_calculator_with_scratch<'a>(
        &'a self,
        qr: Arc<dyn Array>,
        dist_q_c: f32,
        residual: Option<QueryResidual<'_>>,
        f32_scratch: &'a mut Vec<f32>,
    ) -> Self::DistanceCalculator<'a> {
        let code_dim = self.code_dim();
        let dist_table_len = code_dim * 4;
        f32_scratch.resize(code_dim + dist_table_len, 0.0);

        let sum_q = {
            let (rotated_qr, dist_table) = f32_scratch.split_at_mut(code_dim);
            match residual {
                Some(QueryResidual::Centroid(residual_centroid)) => {
                    self.rotate_query_vector_into(
                        code_dim,
                        &qr,
                        Some(residual_centroid),
                        rotated_qr,
                    );
                }
                None => {
                    self.rotate_query_vector_into(code_dim, &qr, None, rotated_qr);
                }
            }
            build_dist_table_direct_into::<Float32Type>(rotated_qr, dist_table);
            rotated_qr.iter().copied().sum()
        };

        self.distance_calculator_from_parts(
            qr.len(),
            dist_q_c,
            Cow::Borrowed(&f32_scratch[code_dim..code_dim + dist_table_len]),
            sum_q,
        )
    }

    // TODO: implement this
    // This method is required for HNSW, we can't support HNSW_RABIT before this is implemented
    fn dist_calculator_from_id(&self, _: u32) -> Self::DistanceCalculator<'_> {
        unimplemented!("RabitQ does not support dist_calculator_from_id")
    }
}

const LOWBIT_IDX: [usize; 16] = {
    let mut array = [0; 16];
    let mut i = 1;
    while i < 16 {
        array[i] = i.trailing_zeros() as usize;
        i += 1;
    }
    array
};

fn get_column(
    quantization_code: &[u8],
    code_len: usize,
    row: usize,
    col_idx: usize,
    codes: &mut [u8; 32],
) {
    for (i, code) in codes.iter_mut().enumerate() {
        let vec_idx = row + i;
        *code = quantization_code[vec_idx * code_len + col_idx];
    }
}

pub fn pack_codes(codes: &FixedSizeListArray) -> FixedSizeListArray {
    let code_len = codes.value_length() as usize;

    // round up num of vectors to multiple of batch size (32)
    let num_blocks = codes.len() / BATCH_SIZE;
    let num_packed_vectors = num_blocks * BATCH_SIZE;

    // calculate total size for packed blocks
    // we pack each 32 vectors into a block, each block contains 2 codes (1byte) of each vector
    // so every 32 vectors would produce code_len blocks
    // the low 16 bytes of each block is the codes for the low 4 bits of each vector
    // the high 16 bytes of each block is the codes for the high 4 bits of each vector
    let mut blocks = vec![0u8; codes.values().len()];

    let codes_values = codes
        .slice(0, num_packed_vectors)
        .values()
        .as_primitive::<UInt8Type>()
        .clone();
    let codes_values = codes_values.values();

    // Pack codes batch by batch
    // Each batch contains codes for 32 vectors
    let mut col = [0u8; 32];
    let mut col_0 = [0u8; 32]; // lower 4 bits
    let mut col_1 = [0u8; 32]; // higher 4 bits
    for row in (0..num_packed_vectors).step_by(BATCH_SIZE) {
        // Get quantization codes for each column for each batch
        // i.e., we get the codes for 8 dims of 32 vectors and reorganize the data layout
        // based on the shuffle SIMD instruction used during querying
        for i in 0..code_len {
            get_column(codes_values, code_len, row, i, &mut col);

            for j in 0..32 {
                col_0[j] = col[j] & 0xF;
                col_1[j] = col[j] >> 4;
            }

            let block_offset = (row / BATCH_SIZE) * code_len * BATCH_SIZE + i * BATCH_SIZE;
            for j in 0..16 {
                // The lower 4 bits represent vector 0 to 15
                // The upper 4 bits represent vector 16 to 31
                let val0 = col_0[PERM0[j]] | (col_0[PERM0[j] + 16] << 4);
                let val1 = col_1[PERM0[j]] | (col_1[PERM0[j] + 16] << 4);
                blocks[block_offset + j] = val0;
                blocks[block_offset + j + 16] = val1;
            }
        }
    }

    // for the left codes, transpose them for better cache locality
    let transposed_codes = transpose(
        &codes.values().as_primitive::<UInt8Type>().slice(
            num_packed_vectors * code_len,
            (codes.len() - num_packed_vectors) * code_len,
        ),
        codes.len() - num_packed_vectors,
        code_len,
    );

    let offset = codes.values().len() - transposed_codes.len();
    for (i, v) in transposed_codes.values().iter().enumerate() {
        blocks[offset + i] = *v;
    }

    assert_eq!(blocks.len(), codes.values().len());
    FixedSizeListArray::try_new_from_values(UInt8Array::from(blocks), code_len as i32).unwrap()
}

// Inverse of pack_codes
pub fn unpack_codes(codes: &FixedSizeListArray) -> FixedSizeListArray {
    let code_len = codes.value_length() as usize;
    let num_vectors = codes.len();

    // Calculate number of complete batches
    let num_blocks = num_vectors / BATCH_SIZE;
    let num_packed_vectors = num_blocks * BATCH_SIZE;

    let mut unpacked = vec![0u8; codes.values().len()];

    let codes_values = codes.values().as_primitive::<UInt8Type>().values();

    // Unpack complete batches
    for batch_idx in 0..num_blocks {
        let block_start = batch_idx * code_len * BATCH_SIZE;

        for i in 0..code_len {
            let block_offset = block_start + i * BATCH_SIZE;
            let block = &codes_values[block_offset..block_offset + BATCH_SIZE];

            // Reverse the permutation
            for j in 0..16 {
                let val0 = block[j];
                let val1 = block[j + 16];

                let low_0 = val0 & 0xF;
                let high_0 = val0 >> 4;
                let low_1 = val1 & 0xF;
                let high_1 = val1 >> 4;

                let vec_idx_0 = batch_idx * BATCH_SIZE + PERM0[j];
                let vec_idx_1 = batch_idx * BATCH_SIZE + PERM0[j] + 16;

                unpacked[vec_idx_0 * code_len + i] = low_0 | (low_1 << 4);
                unpacked[vec_idx_1 * code_len + i] = high_0 | (high_1 << 4);
            }
        }
    }

    // Transpose back the remainder
    if num_packed_vectors < num_vectors {
        let remainder = num_vectors - num_packed_vectors;
        let offset = num_packed_vectors * code_len;
        let transposed_data = &codes_values[offset..];

        // Transpose from column-major back to row-major
        for row in 0..remainder {
            for col in 0..code_len {
                unpacked[offset + row * code_len + col] = transposed_data[col * remainder + row];
            }
        }
    }

    FixedSizeListArray::try_new_from_values(UInt8Array::from(unpacked), code_len as i32).unwrap()
}

#[async_trait]
impl QuantizerStorage for RabitQuantizationStorage {
    type Metadata = RabitQuantizationMetadata;

    fn try_from_batch(
        batch: RecordBatch,
        metadata: &Self::Metadata,
        distance_type: DistanceType,
        _fri: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>().clone();
        let codes = batch[RABIT_CODE_COLUMN].as_fixed_size_list().clone();
        let add_factors = batch[ADD_FACTORS_COLUMN]
            .as_primitive::<Float32Type>()
            .clone();
        let scale_factors = batch[SCALE_FACTORS_COLUMN]
            .as_primitive::<Float32Type>()
            .clone();

        let (batch, codes) = if !metadata.packed {
            let codes = pack_codes(&codes);
            let batch = batch.replace_column_by_name(RABIT_CODE_COLUMN, Arc::new(codes))?;
            let codes = batch[RABIT_CODE_COLUMN].as_fixed_size_list().clone();
            (batch, codes)
        } else {
            (batch, codes)
        };

        let mut metadata = metadata.clone();
        metadata.packed = true;

        Ok(Self {
            metadata,
            batch,
            distance_type,
            row_ids,
            codes,
            add_factors,
            scale_factors,
        })
    }

    fn metadata(&self) -> &Self::Metadata {
        &self.metadata
    }

    async fn load_partition(
        reader: &PreviousFileReader,
        range: std::ops::Range<usize>,
        distance_type: DistanceType,
        metadata: &Self::Metadata,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let schema = reader.schema();
        let batch = reader.read_range(range, schema).await?;
        Self::try_from_batch(batch, metadata, distance_type, frag_reuse_index)
    }

    fn remap(&self, mapping: &HashMap<u64, Option<u64>>) -> Result<Self> {
        let num_vectors = self.codes.len();
        let num_code_bytes = self.codes.value_length() as usize;
        let codes = self.codes.values().as_primitive::<UInt8Type>().values();
        let mut indices = Vec::with_capacity(num_vectors);
        let mut new_row_ids = Vec::with_capacity(num_vectors);
        let mut new_codes = Vec::with_capacity(codes.len());

        let row_ids = self.row_ids.values();
        for (i, row_id) in row_ids.iter().enumerate() {
            match mapping.get(row_id) {
                Some(Some(new_id)) => {
                    indices.push(i as u32);
                    new_row_ids.push(*new_id);
                    new_codes.extend(get_rq_code(codes, i, num_vectors, num_code_bytes));
                }
                Some(None) => {}
                None => {
                    indices.push(i as u32);
                    new_row_ids.push(*row_id);
                    new_codes.extend(get_rq_code(codes, i, num_vectors, num_code_bytes));
                }
            }
        }

        let new_row_ids = UInt64Array::from(new_row_ids);
        let new_codes = FixedSizeListArray::try_new_from_values(
            UInt8Array::from(new_codes),
            num_code_bytes as i32,
        )?;
        let batch = if new_row_ids.is_empty() {
            RecordBatch::new_empty(self.schema().clone())
        } else {
            let codes = Arc::new(pack_codes(&new_codes));
            self.batch
                .take(&UInt32Array::from(indices))?
                .replace_column_by_name(ROW_ID, Arc::new(new_row_ids.clone()))?
                .replace_column_by_name(RABIT_CODE_COLUMN, codes)?
        };
        let codes = batch[RABIT_CODE_COLUMN].as_fixed_size_list().clone();

        Ok(Self {
            metadata: self.metadata.clone(),
            distance_type: self.distance_type,
            batch,
            codes,
            add_factors: self.add_factors.clone(),
            scale_factors: self.scale_factors.clone(),
            row_ids: new_row_ids,
        })
    }
}

/// Compute the raw distance for a single vector without allocating.
///
/// Fuses code extraction from the packed layout with distance accumulation
/// in a single pass, avoiding the intermediate `Vec` allocation that
/// `get_rq_code` + iterator would require.
#[inline]
fn compute_single_rq_distance(
    codes: &[u8],
    id: usize,
    num_vectors: usize,
    num_code_bytes: usize,
    dist_table: &[f32],
) -> f32 {
    let remainder = num_vectors % BATCH_SIZE;
    let mut dist_table_iter = dist_table.chunks_exact(SEGMENT_NUM_CODES).tuples();

    if id < num_vectors - remainder {
        let batch_codes = &codes[id / BATCH_SIZE * BATCH_SIZE * num_code_bytes
            ..(id / BATCH_SIZE + 1) * BATCH_SIZE * num_code_bytes];

        let id_in_batch = id % BATCH_SIZE;
        let idx = PERM0_INVERSE[id_in_batch % 16];
        let is_lower = id_in_batch < 16;

        let mut dist = 0.0f32;
        for block in batch_codes.chunks_exact(BATCH_SIZE) {
            let code_byte = if is_lower {
                (block[idx] & 0xF) | (block[idx + 16] << 4)
            } else {
                (block[idx] >> 4) | (block[idx + 16] & 0xF0)
            };
            if let Some((current_dt, next_dt)) = dist_table_iter.next() {
                let current_code = (code_byte & 0x0F) as usize;
                let next_code = (code_byte >> 4) as usize;
                dist += current_dt[current_code] + next_dt[next_code];
            }
        }
        dist
    } else {
        let offset_id = id - (num_vectors - remainder);
        let remainder_codes = &codes[(num_vectors - remainder) * num_code_bytes..];

        let mut dist = 0.0f32;
        for &code_byte in remainder_codes.iter().skip(offset_id).step_by(remainder) {
            if let Some((current_dt, next_dt)) = dist_table_iter.next() {
                let current_code = (code_byte & 0x0F) as usize;
                let next_code = (code_byte >> 4) as usize;
                dist += current_dt[current_code] + next_dt[next_code];
            }
        }
        dist
    }
}

#[inline]
fn get_rq_code(
    codes: &[u8],
    id: usize,
    num_vectors: usize,
    num_code_bytes: usize,
) -> impl Iterator<Item = u8> + '_ {
    let remainder = num_vectors % BATCH_SIZE;

    if id < num_vectors - remainder {
        // the codes are packed
        let codes = &codes[id / BATCH_SIZE * BATCH_SIZE * num_code_bytes
            ..(id / BATCH_SIZE + 1) * BATCH_SIZE * num_code_bytes];

        let id_in_batch = id % BATCH_SIZE;
        if id_in_batch < 16 {
            let idx = PERM0_INVERSE[id_in_batch];
            codes
                .chunks_exact(BATCH_SIZE)
                .map(|block| (block[idx] & 0xF) | (block[idx + 16] << 4))
                .exact_size(num_code_bytes)
                .collect_vec()
                .into_iter()
        } else {
            let idx = PERM0_INVERSE[id_in_batch - 16];
            codes
                .chunks_exact(BATCH_SIZE)
                .map(|block| (block[idx] >> 4) | (block[idx + 16] & 0xF0))
                .exact_size(num_code_bytes)
                .collect_vec()
                .into_iter()
        }
    } else {
        let id = id - (num_vectors - remainder);
        let codes = &codes[(num_vectors - remainder) * num_code_bytes..];
        codes
            .iter()
            .skip(id)
            .step_by(remainder)
            .copied()
            .exact_size(num_code_bytes)
            .collect_vec()
            .into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use arrow_array::{ArrayRef, Float32Array, Float64Array, UInt64Array};
    use lance_core::ROW_ID;
    use lance_linalg::distance::DistanceType;

    use crate::vector::bq::{RQRotationType, builder::RabitQuantizer};
    use crate::vector::quantizer::{Quantization, QuantizerStorage};

    fn build_dist_table_not_optimized<T: ArrowFloatType>(
        sub_vec: &[T::Native],
        dist_table: &mut [f32],
    ) where
        T::Native: AsPrimitive<f32>,
    {
        for (j, dist) in dist_table.iter_mut().enumerate().take(SEGMENT_NUM_CODES) {
            for (k, v) in sub_vec.iter().enumerate().take(SEGMENT_LENGTH) {
                if j & (1 << k) != 0 {
                    *dist += v.as_();
                }
            }
        }
    }

    #[test]
    fn test_build_dist_table_not_optimized() {
        let sub_vec = vec![1.0, 2.0, 3.0, 4.0];
        let mut expected = vec![0.0; SEGMENT_NUM_CODES];
        build_dist_table_not_optimized::<Float32Type>(&sub_vec, &mut expected);
        let mut dist_table = vec![0.0; SEGMENT_NUM_CODES];
        build_dist_table_for_subvec::<Float32Type>(&sub_vec, &mut dist_table);
        assert_eq!(dist_table, expected);
    }

    #[test]
    fn test_dist_calculator_with_scratch_matches_owned_and_reuses_buffer() {
        let code_dim = 64;
        let original_codes = make_test_codes(50, code_dim);
        let metadata = make_test_metadata(original_codes.value_length() as usize * 8);
        let storage = RabitQuantizationStorage::try_from_batch(
            make_test_batch(original_codes),
            &metadata,
            DistanceType::L2,
            None,
        )
        .unwrap();
        let query = Arc::new(Float32Array::from_iter_values(
            (0..code_dim).map(|idx| idx as f32 / code_dim as f32),
        )) as ArrayRef;

        let expected = storage.dist_calculator(query.clone(), 0.25).distance_all(0);
        let expected_scratch_len = code_dim as usize + code_dim as usize * 4;
        let mut scratch = Vec::with_capacity(expected_scratch_len);
        let initial_ptr = scratch.as_ptr();
        {
            let calc =
                storage.dist_calculator_with_scratch(query.clone(), 0.25, None, &mut scratch);
            assert_eq!(calc.distance_all(0), expected);
        }
        assert_eq!(scratch.len(), expected_scratch_len);
        assert_eq!(scratch.as_ptr(), initial_ptr);

        scratch.fill(f32::NAN);
        {
            let calc = storage.dist_calculator_with_scratch(query, 0.25, None, &mut scratch);
            assert_eq!(calc.distance_all(0), expected);
        }
        assert_eq!(scratch.as_ptr(), initial_ptr);
    }

    #[test]
    fn test_dist_calculator_with_scratch_applies_residual_centroid_without_residual_array() {
        let code_dim = 64usize;
        let original_codes = make_test_codes(50, code_dim as i32);
        let metadata = make_test_metadata(original_codes.value_length() as usize * 8);
        let storage = RabitQuantizationStorage::try_from_batch(
            make_test_batch(original_codes),
            &metadata,
            DistanceType::L2,
            None,
        )
        .unwrap();
        let query_values = (0..code_dim)
            .map(|idx| idx as f32 / code_dim as f32)
            .collect::<Vec<_>>();
        let centroid_values = (0..code_dim)
            .map(|idx| (idx % 7) as f32 / code_dim as f32)
            .collect::<Vec<_>>();
        let residual_values = query_values
            .iter()
            .zip(centroid_values.iter())
            .map(|(query, centroid)| query - centroid)
            .collect::<Vec<_>>();
        let query = Arc::new(Float32Array::from(query_values)) as ArrayRef;
        let centroid = Arc::new(Float32Array::from(centroid_values)) as ArrayRef;
        let residual = Arc::new(Float32Array::from(residual_values)) as ArrayRef;

        let expected = storage.dist_calculator(residual, 0.25).distance_all(0);
        let mut scratch = Vec::new();
        let calc = storage.dist_calculator_with_scratch(
            query.clone(),
            0.25,
            Some(QueryResidual::Centroid(centroid.as_ref())),
            &mut scratch,
        );

        assert_eq!(calc.distance_all(0), expected);
    }

    #[test]
    fn test_dist_calculator_with_scratch_applies_float64_residual_before_f32_cast() {
        let code_dim = 64usize;
        let original_codes = make_test_codes(50, code_dim as i32);
        let metadata = make_test_metadata(original_codes.value_length() as usize * 8);
        let storage = RabitQuantizationStorage::try_from_batch(
            make_test_batch(original_codes),
            &metadata,
            DistanceType::L2,
            None,
        )
        .unwrap();
        let query_values = (0..code_dim)
            .map(|idx| 1.0 + idx as f64 * 1.0e-9)
            .collect::<Vec<_>>();
        let centroid_values = vec![1.0; code_dim];
        let residual_values = query_values
            .iter()
            .zip(centroid_values.iter())
            .map(|(query, centroid)| query - centroid)
            .collect::<Vec<_>>();
        let query = Arc::new(Float64Array::from(query_values)) as ArrayRef;
        let centroid = Arc::new(Float64Array::from(centroid_values)) as ArrayRef;
        let residual = Arc::new(Float64Array::from(residual_values)) as ArrayRef;

        let expected = storage.dist_calculator(residual, 0.25).distance_all(0);
        let mut scratch = Vec::new();
        let calc = storage.dist_calculator_with_scratch(
            query,
            0.25,
            Some(QueryResidual::Centroid(centroid.as_ref())),
            &mut scratch,
        );

        assert_eq!(calc.distance_all(0), expected);
    }

    #[test]
    fn test_pack_unpack_codes() {
        // Test with multiple batch sizes to cover both packed and transposed sections
        for num_vectors in [10, 32, 50, 64, 100] {
            let code_len = 8;

            // Create test data with known pattern
            let mut codes_data = Vec::new();
            for i in 0..num_vectors {
                for j in 0..code_len {
                    codes_data.push((i * code_len + j) as u8);
                }
            }

            let original_codes = FixedSizeListArray::try_new_from_values(
                UInt8Array::from(codes_data.clone()),
                code_len,
            )
            .unwrap();

            // Pack and then unpack
            let packed = pack_codes(&original_codes);
            let unpacked = unpack_codes(&packed);

            // Verify they match
            assert_eq!(original_codes.len(), unpacked.len());
            assert_eq!(original_codes.value_length(), unpacked.value_length());

            let original_values = original_codes.values().as_primitive::<UInt8Type>().values();
            let unpacked_values = unpacked.values().as_primitive::<UInt8Type>().values();

            assert_eq!(
                original_values, unpacked_values,
                "Mismatch for num_vectors={}",
                num_vectors
            );
        }
    }

    fn make_test_codes(num_vectors: usize, code_dim: i32) -> FixedSizeListArray {
        let quantizer =
            RabitQuantizer::new_with_rotation::<Float32Type>(1, code_dim, RQRotationType::Fast);
        let values = Float32Array::from_iter_values(
            (0..num_vectors * code_dim as usize).map(|idx| idx as f32 / code_dim as f32),
        );
        let vectors = FixedSizeListArray::try_new_from_values(values, code_dim).unwrap();
        quantizer
            .quantize(&vectors)
            .unwrap()
            .as_fixed_size_list()
            .clone()
    }

    fn make_test_metadata(code_dim: usize) -> RabitQuantizationMetadata {
        RabitQuantizer::new_with_rotation::<Float32Type>(1, code_dim as i32, RQRotationType::Fast)
            .metadata(None)
    }

    fn make_test_batch(codes: FixedSizeListArray) -> RecordBatch {
        let num_rows = codes.len();
        RecordBatch::try_from_iter(vec![
            (
                ROW_ID,
                Arc::new(UInt64Array::from_iter_values(0..num_rows as u64)) as ArrayRef,
            ),
            (RABIT_CODE_COLUMN, Arc::new(codes) as ArrayRef),
            (
                ADD_FACTORS_COLUMN,
                Arc::new(Float32Array::from_iter_values(
                    (0..num_rows).map(|v| v as f32),
                )) as ArrayRef,
            ),
            (
                SCALE_FACTORS_COLUMN,
                Arc::new(Float32Array::from_iter_values(
                    (0..num_rows).map(|v| v as f32 + 0.5),
                )) as ArrayRef,
            ),
        ])
        .unwrap()
    }

    fn assert_codes_eq(actual: &FixedSizeListArray, expected: &FixedSizeListArray) {
        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual.value_length(), expected.value_length());
        assert_eq!(
            actual.values().as_primitive::<UInt8Type>().values(),
            expected.values().as_primitive::<UInt8Type>().values()
        );
    }

    #[test]
    fn test_try_from_batch_canonicalizes_rq_codes_to_packed_layout() {
        let original_codes = make_test_codes(50, 64);
        let metadata = make_test_metadata(original_codes.value_length() as usize * 8);
        assert!(!metadata.packed);

        let storage = RabitQuantizationStorage::try_from_batch(
            make_test_batch(original_codes.clone()),
            &metadata,
            DistanceType::L2,
            None,
        )
        .unwrap();

        assert!(storage.metadata().packed);
        let stored_batch = storage.to_batches().unwrap().next().unwrap();
        let stored_codes = stored_batch[RABIT_CODE_COLUMN].as_fixed_size_list();
        let expected_codes = pack_codes(&original_codes);
        assert_codes_eq(stored_codes, &expected_codes);
    }

    #[test]
    fn test_remap_preserves_packed_rq_storage_layout() {
        let original_codes = make_test_codes(50, 64);
        let metadata = make_test_metadata(original_codes.value_length() as usize * 8);
        let storage = RabitQuantizationStorage::try_from_batch(
            make_test_batch(original_codes.clone()),
            &metadata,
            DistanceType::L2,
            None,
        )
        .unwrap();

        let mut mapping = HashMap::new();
        mapping.insert(1, Some(101));
        mapping.insert(3, None);
        mapping.insert(4, Some(104));

        let remapped = storage.remap(&mapping).unwrap();
        assert!(remapped.metadata().packed);

        let remapped_batch = remapped.to_batches().unwrap().next().unwrap();
        let remapped_row_ids = remapped_batch[ROW_ID].as_primitive::<UInt64Type>().values();
        let expected_row_ids = UInt64Array::from_iter_values(
            [0, 101, 2, 104]
                .into_iter()
                .chain(5..original_codes.len() as u64),
        );
        assert_eq!(remapped_row_ids, expected_row_ids.values());

        let remapped_codes = remapped_batch[RABIT_CODE_COLUMN].as_fixed_size_list();
        let repacked = pack_codes(&unpack_codes(remapped_codes));
        assert_codes_eq(remapped_codes, &repacked);
    }
}
