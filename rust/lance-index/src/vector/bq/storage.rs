// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::{Float16Type, Float32Type, Float64Type, UInt64Type, UInt8Type};
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_schema::{DataType, SchemaRef};
use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use deepsize::DeepSizeOf;
use itertools::Itertools;
use lance_arrow::{ArrowFloatType, FixedSizeListArrayExt, FloatArray, RecordBatchExt};
use lance_core::{Error, Result, ROW_ID};
use lance_file::reader::FileReader;
use lance_linalg::distance::{DistanceType, Dot};
use lance_linalg::simd::dist_table::{BATCH_SIZE, PERM0, PERM0_INVERSE};
use lance_linalg::simd::{self};
use lance_table::utils::LanceIteratorExtension;
use num_traits::AsPrimitive;
use prost::Message;
use serde::{Deserialize, Serialize};
use snafu::location;

use crate::frag_reuse::FragReuseIndex;
use crate::pb;
use crate::vector::bq::transform::{ADD_FACTORS_COLUMN, SCALE_FACTORS_COLUMN};
use crate::vector::pq::storage::transpose;
use crate::vector::quantizer::{QuantizerMetadata, QuantizerStorage};
use crate::vector::storage::{DistCalculator, VectorStore};

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
    pub rotate_mat_position: u32,
    pub num_bits: u8,
    pub packed: bool,
}

impl DeepSizeOf for RabitQuantizationMetadata {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.rotate_mat
            .as_ref()
            .map(|inv_p| inv_p.get_array_memory_size())
            .unwrap_or(0)
    }
}

#[async_trait]
impl QuantizerMetadata for RabitQuantizationMetadata {
    fn buffer_index(&self) -> Option<u32> {
        Some(self.rotate_mat_position)
    }

    fn set_buffer_index(&mut self, index: u32) {
        self.rotate_mat_position = index;
    }

    fn parse_buffer(&mut self, bytes: Bytes) -> Result<()> {
        debug_assert!(!bytes.is_empty());
        let codebook_tensor: pb::Tensor = pb::Tensor::decode(bytes)?;
        self.rotate_mat = Some(FixedSizeListArray::try_from(&codebook_tensor)?);
        Ok(())
    }

    fn extra_metadata(&self) -> Result<Option<Bytes>> {
        if let Some(inv_p) = &self.rotate_mat {
            let inv_p_tensor = pb::Tensor::try_from(inv_p)?;
            let mut bytes = BytesMut::new();
            inv_p_tensor.encode(&mut bytes)?;
            Ok(Some(bytes.freeze()))
        } else {
            Ok(None)
        }
    }

    async fn load(reader: &FileReader) -> Result<Self> {
        let metadata_str =
            reader
                .schema()
                .metadata
                .get(RABIT_METADATA_KEY)
                .ok_or(Error::Index {
                    message: format!(
                        "Reading Rabit metadata: metadata key {} not found",
                        RABIT_METADATA_KEY
                    ),
                    location: location!(),
                })?;
        serde_json::from_str(metadata_str).map_err(|_| Error::Index {
            message: format!("Failed to parse index metadata: {}", metadata_str),
            location: location!(),
        })
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
    fn rotate_query_vector<T: ArrowFloatType>(
        rotate_mat: &FixedSizeListArray,
        qr: &dyn Array,
    ) -> Vec<f32>
    where
        T::Native: Dot,
    {
        let d = qr.len();
        let code_dim = rotate_mat.len();
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

        rotate_mat
            .chunks_exact(code_dim)
            .map(|chunk| lance_linalg::distance::dot(&chunk[..d], qr))
            .collect()
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
    dist_table: Vec<f32>,
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
        dist_table: Vec<f32>,
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
    qc.chunks_exact(SEGMENT_LENGTH)
        .zip(dist_table.chunks_exact_mut(SEGMENT_NUM_CODES))
        .for_each(|(sub_vec, dist_table)| build_dist_table_for_subvec::<T>(sub_vec, dist_table));
    dist_table
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

// Quantize the distance table to u8, map distance `d` to `(d-qmin) * 255 / (qmax-qmin)`
#[inline]
fn quantize_dist_table(dist_table: &[f32]) -> (f32, f32, Vec<u8>) {
    let (qmin, qmax) = dist_table
        .iter()
        .cloned()
        .minmax_by(|a, b| a.total_cmp(b))
        .into_option()
        .unwrap();
    // this happens if the query is all zeros
    if qmin == qmax {
        return (qmin, qmax, vec![0; dist_table.len()]);
    }
    let factor = 255.0 / (qmax - qmin);
    let quantized_dist_table = dist_table
        .iter()
        .map(|&d| ((d - qmin) * factor).round() as u8)
        .collect();

    (qmin, qmax, quantized_dist_table)
}

#[inline]
fn compute_rq_distance_flat(
    dist_table: &[f32],
    codes: &[u8],
    offset: usize,
    length: usize,
    dists: &mut [f32],
) {
    let d = dist_table.len() / 4;
    let code_len = d / u8::BITS as usize;
    let codes = &codes[offset * code_len..(offset + length) * code_len];
    let dists = &mut dists[offset..offset + length];

    for (sub_vec_idx, codes) in codes.chunks_exact(length).enumerate() {
        let current_dist_table = &dist_table
            [sub_vec_idx * 2 * SEGMENT_NUM_CODES..(sub_vec_idx * 2 + 1) * SEGMENT_NUM_CODES];
        let next_dist_table = &dist_table
            [(sub_vec_idx * 2 + 1) * SEGMENT_NUM_CODES..(sub_vec_idx * 2 + 2) * SEGMENT_NUM_CODES];

        codes.iter().zip(dists.iter_mut()).for_each(|(code, dist)| {
            let current_code = (code & 0x0F) as usize;
            let next_code = (code >> 4) as usize;
            *dist += current_dist_table[current_code] + next_dist_table[next_code];
        });
    }
}

impl DistCalculator for RabitDistCalculator<'_> {
    #[inline(always)]
    fn distance(&self, id: u32) -> f32 {
        let id = id as usize;
        let code_len = self.dim * (self.num_bits as usize) / u8::BITS as usize;
        let num_vectors = self.codes.len() / code_len;
        let code = get_rq_code(self.codes, id, num_vectors, code_len);
        let dist = code
            .zip(self.dist_table.chunks_exact(SEGMENT_NUM_CODES).tuples())
            .map(|(code_byte, (dist_table, next_dist_table))| {
                // code is a bit vector, we iterate over 8 bits at a time,
                // every 4 bits is a sub-vector, we need to extract the bits
                let current_code = (code_byte & 0x0F) as usize;
                let next_code = (code_byte >> 4) as usize;
                dist_table[current_code] + next_dist_table[next_code]
            })
            .sum::<f32>();

        // distance between quantized vector and query vector
        let dist_vq_qr = (2.0 * dist - self.sum_q) / self.sqrt_d;
        dist_vq_qr * self.scale_factors[id] + self.add_factors[id] + self.query_factor
    }

    #[inline(always)]
    fn distance_all(&self, _: usize) -> Vec<f32> {
        let code_len = self.dim * (self.num_bits as usize) / u8::BITS as usize;
        let n = self.codes.len() / code_len;
        if n == 0 {
            return Vec::new();
        }

        let mut dists = vec![0.0; n];

        let (qmin, qmax, quantized_dists_table) = quantize_dist_table(&self.dist_table);
        let mut quantized_dists = vec![0; n];

        let remainder = n % BATCH_SIZE;
        simd::dist_table::sum_4bit_dist_table(
            n - remainder,
            code_len,
            self.codes,
            &quantized_dists_table,
            &mut quantized_dists,
        );
        if remainder > 0 {
            compute_rq_distance_flat(
                &self.dist_table,
                self.codes,
                n - remainder,
                remainder,
                &mut dists,
            );
        }

        let range = (qmax - qmin) / 255.0;
        let num_tables = quantized_dists_table.len() / 16;
        let sum_min = num_tables as f32 * qmin;
        dists
            .iter_mut()
            .take(n - remainder)
            .zip(quantized_dists.into_iter().take(n - remainder))
            .for_each(|(dist, q_dist)| {
                *dist = (q_dist as f32) * range + sum_min;
            });

        dists
            .into_iter()
            .enumerate()
            .map(|(id, dist)| {
                let dist_vq_qr = (2.0 * dist - self.sum_q) / self.sqrt_d;
                dist_vq_qr * self.scale_factors[id] + self.add_factors[id] + self.query_factor
            })
            .collect()
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
        let codes = self.codes.values().as_primitive::<UInt8Type>().values();
        let rotate_mat = self
            .metadata
            .rotate_mat
            .as_ref()
            .expect("RabitQ metadata not loaded");

        let rotated_qr = match rotate_mat.value_type() {
            DataType::Float16 => Self::rotate_query_vector::<Float16Type>(rotate_mat, &qr),
            DataType::Float32 => Self::rotate_query_vector::<Float32Type>(rotate_mat, &qr),
            DataType::Float64 => Self::rotate_query_vector::<Float64Type>(rotate_mat, &qr),
            dt => unimplemented!("RabitQ does not support data type: {}", dt),
        };

        let dist_table = build_dist_table_direct::<Float32Type>(&rotated_qr);
        let sum_q = rotated_qr.into_iter().sum();

        let q_factor = match self.distance_type {
            DistanceType::L2 => dist_q_c,
            DistanceType::Cosine | DistanceType::Dot => dist_q_c - 1.0,
            _ => unimplemented!(
                "RabitQ does not support distance type: {}",
                self.distance_type
            ),
        };
        RabitDistCalculator::new(
            qr.len(),
            self.metadata.num_bits,
            dist_table,
            sum_q,
            codes,
            self.add_factors.values(),
            self.scale_factors.values(),
            q_factor,
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
        reader: &FileReader,
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
}
