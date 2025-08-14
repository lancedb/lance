// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::cmp::{max, min};
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
use lance_linalg::distance::DistanceType;
use lance_linalg::simd::u8::u8x16;
use lance_linalg::simd::{Shuffle, SIMD};
use lance_table::utils::LanceIteratorExtension;
use num_traits::AsPrimitive;
use prost::Message;
use serde::{Deserialize, Serialize};
use snafu::location;

use crate::frag_reuse::FragReuseIndex;
use crate::pb;
use crate::vector::bq::transform::{
    ADD_FACTORS_COLUMN, CODE_BITCOUNT_COLUMN, SCALE_FACTORS_COLUMN,
};
use crate::vector::pq::storage::transpose;
use crate::vector::quantizer::{QuantizerMetadata, QuantizerStorage};
use crate::vector::sq::scale_to_u8;
use crate::vector::storage::{DistCalculator, VectorStore};

pub const RABIT_METADATA_KEY: &str = "lance:rabit";
pub const RABIT_CODE_COLUMN: &str = "_rabit_codes";
pub const SEGMENT_LENGTH: usize = 4;
pub const SEGMENT_NUM_CODES: usize = 1 << SEGMENT_LENGTH;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RabitQuantizationMetadata {
    #[serde(skip)]
    pub inv_p: Option<FixedSizeListArray>,
    pub inv_p_position: u32,
    pub num_bits: u8,
    pub transposed: bool,
}

impl DeepSizeOf for RabitQuantizationMetadata {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.inv_p
            .as_ref()
            .map(|inv_p| inv_p.get_array_memory_size())
            .unwrap_or(0)
    }
}

#[async_trait]
impl QuantizerMetadata for RabitQuantizationMetadata {
    fn buffer_index(&self) -> Option<u32> {
        Some(self.inv_p_position)
    }

    fn set_buffer_index(&mut self, index: u32) {
        self.inv_p_position = index;
    }

    fn parse_buffer(&mut self, bytes: Bytes) -> Result<()> {
        debug_assert!(!bytes.is_empty());
        let codebook_tensor: pb::Tensor = pb::Tensor::decode(bytes)?;
        self.inv_p = Some(FixedSizeListArray::try_from(&codebook_tensor)?);
        Ok(())
    }

    fn extra_metadata(&self) -> Result<Option<Bytes>> {
        if let Some(inv_p) = &self.inv_p {
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
    code_bitcounts: Float32Array,
    add_factors: Float32Array,
    scale_factors: Float32Array,
}

impl DeepSizeOf for RabitQuantizationStorage {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.metadata.deep_size_of_children(context) + self.batch.get_array_memory_size()
    }
}

impl RabitQuantizationStorage {
    fn quantize_query_vector<T: ArrowFloatType>(
        inv_p: &dyn Array,
        qr: &dyn Array,
    ) -> (Vec<u8>, f32, f32)
    where
        T::Native: num_traits::Float + AsPrimitive<f64> + AsPrimitive<f32>,
    {
        let d = qr.len();
        // convert to matrix
        let inv_p = inv_p
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .unwrap()
            .as_slice();
        let inv_p = ndarray::ArrayView2::from_shape((d, d), inv_p).unwrap();
        let qr = qr
            .as_any()
            .downcast_ref::<T::ArrayType>()
            .unwrap()
            .as_slice();
        let qr = ndarray::ArrayView2::from_shape((d, 1), qr).unwrap();

        // rotate query vector
        let rotated_qr = inv_p.dot(&qr);
        let rotated_qr = rotated_qr.as_slice().unwrap();
        let (min, max) = rotated_qr
            .iter()
            .copied()
            .minmax()
            .into_option()
            .expect("failed to get min and max of query vector");
        let bounds = min.as_()..max.as_();
        let query_codes = scale_to_u8::<T>(rotated_qr, &bounds);

        let (min, max): (f32, f32) = (min.as_(), max.as_());
        (query_codes, min, (max - min) / 255.0)
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
    dist_table: Vec<u32>,
    code_bitcounts: &'a [f32],
    add_factors: &'a [f32],
    scale_factors: &'a [f32],
    query_factor: f32,

    sq_min: f32,
    sq_scale: f32,
    sq_sum: f32,
    sqrt_d: f32,
}

impl<'a> RabitDistCalculator<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        query_codes: Vec<u8>,
        sq_min: f32,
        sq_scale: f32,
        codes: &'a [u8],
        code_bitcounts: &'a [f32],
        add_factors: &'a [f32],
        scale_factors: &'a [f32],
        query_factor: f32,
    ) -> Self {
        let dim = query_codes.len();
        let sq_sum = query_codes.iter().map(|&v| v as u32).sum::<u32>() as f32;

        let dist_table = build_dist_table(query_codes);
        Self {
            dim,
            num_bits: 1,
            codes,
            dist_table,
            code_bitcounts,
            add_factors,
            scale_factors,
            query_factor,
            sq_min,
            sq_scale,
            sq_sum,
            sqrt_d: (dim as f32).sqrt(),
        }
    }
}

pub fn build_dist_table(query_codes: Vec<u8>) -> Vec<u32> {
    // divide query codes into segments of 4 bytes,
    // and calculate the distance between the segment and binary vector (4 bits),
    // dist_table[i][j] is the distance between the i-th query code segment and the code j
    query_codes
        .chunks_exact(SEGMENT_LENGTH)
        .flat_map(|sub_vec| {
            debug_assert_eq!(sub_vec.len(), SEGMENT_LENGTH);
            (0..SEGMENT_NUM_CODES).map(|j| {
                let mut dist = 0;
                for (b, &v) in sub_vec.iter().enumerate() {
                    if (j >> b) & 0x1 != 0 {
                        dist += v as u32;
                    }
                }
                dist
            })
        })
        .exact_size(query_codes.len() / SEGMENT_LENGTH * SEGMENT_NUM_CODES)
        .collect()
}

#[inline]
fn quantize_dist_table(dist_table: &[u32], qmax: u32) -> (u32, Vec<u8>) {
    let qmin = dist_table.iter().cloned().min().unwrap();
    let factor = 255.0 / (qmax - qmin) as f32;
    let quantized_dist_table = dist_table
        .iter()
        .map(|&d| ((d - qmin) as f32 * factor).round() as u8)
        .collect();

    (qmin, quantized_dist_table)
}

#[inline]
fn compute_rq_distance_flat(
    dist_table: &[u32],
    n: usize,
    codes: &[u8],
    offset: usize,
    length: usize,
    dists: &mut [u32],
) {
    for (sub_vec_idx, codes) in codes.chunks_exact(n).enumerate() {
        let codes = &codes[offset..offset + length];
        let dists = &mut dists[offset..offset + length];
        debug_assert_eq!(codes.len(), n);
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
        let code_len = self.dim * (self.num_bits as usize) / 8;
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
            .sum::<u32>();

        let vec_bitcount = self.code_bitcounts[id];
        let dist = dist as f32;

        // distance between quantized vector and query vector
        let dist_vq_qr = (2.0 * self.sq_scale * dist + 2.0 * self.sq_min * vec_bitcount
            - self.sq_scale * self.sq_sum
            - self.dim as f32 * self.sq_min)
            / self.sqrt_d;

        dist_vq_qr * self.scale_factors[id] + self.add_factors[id] + self.query_factor
    }

    #[inline(never)]
    fn distance_all(&self, k_hint: usize) -> Vec<f32> {
        let code_len = self.dim * (self.num_bits as usize) / 8;
        let n = self.codes.len() / code_len;
        if n == 0 {
            return Vec::new();
        }

        let mut dists = vec![0; n];

        const FLAT_NUM_RQ: usize = 200;
        let k_hint = min(k_hint, n);
        let flat_num = max(FLAT_NUM_RQ, k_hint).min(n);
        compute_rq_distance_flat(&self.dist_table, n, self.codes, 0, flat_num, &mut dists);

        let qmax = dists.iter().take(flat_num).copied().max().unwrap();
        let (qmin, quantized_dists_table) = quantize_dist_table(&self.dist_table, qmax);

        let mut quantized_dists = vec![0_u8; n];

        let remainder = n % SEGMENT_NUM_CODES;
        for i in (0..n - remainder).step_by(SEGMENT_NUM_CODES) {
            let mut block_distances = u8x16::zeros();
            for (sub_vec_idx, codes) in self.codes.chunks_exact(n).enumerate() {
                let dist_table = unsafe {
                    u8x16::load_unaligned(
                        quantized_dists_table
                            .as_ptr()
                            .add(sub_vec_idx * 2 * SEGMENT_NUM_CODES),
                    )
                };
                let next_dist_table = unsafe {
                    u8x16::load_unaligned(
                        quantized_dists_table
                            .as_ptr()
                            .add((sub_vec_idx * 2 + 1) * SEGMENT_NUM_CODES),
                    )
                };

                let codes = unsafe { u8x16::load_unaligned(codes.as_ptr().add(i)) };
                let current_indices = codes.bit_and(0x0F);
                block_distances += dist_table.shuffle(current_indices);
                let next_indices = codes.right_shift::<4>();
                block_distances += next_dist_table.shuffle(next_indices);
            }
            unsafe {
                block_distances.store_unaligned(quantized_dists.as_mut_ptr().add(i));
            }
        }

        if remainder > 0 {
            let offset = max(n - remainder, flat_num);
            compute_rq_distance_flat(
                &self.dist_table,
                n,
                self.codes,
                offset,
                n - offset,
                &mut dists,
            );
        }

        let range = qmax - qmin;
        dists
            .iter_mut()
            .take(n - remainder)
            .skip(flat_num)
            .zip(
                quantized_dists
                    .into_iter()
                    .take(n - remainder)
                    .skip(flat_num),
            )
            .for_each(|(dist, q_dist)| {
                *dist = (q_dist as u32) * range / 255 + qmin;
            });

        dists
            .into_iter()
            .enumerate()
            .map(|(id, dist)| {
                let dist = dist as f32;
                let vec_bitcount = self.code_bitcounts[id];
                let dist_vq_qr = (2.0 * self.sq_scale * dist + 2.0 * self.sq_min * vec_bitcount
                    - self.sq_scale * self.sq_sum
                    - self.dim as f32 * self.sq_min)
                    / self.sqrt_d;
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
    fn dist_calculator(&self, qr: Arc<dyn Array>, dist_q_c: f32) -> Self::DistanceCalculator<'_> {
        let codes = self.codes.values().as_primitive::<UInt8Type>().values();
        let inv_p = self
            .metadata
            .inv_p
            .as_ref()
            .map(|inv_p| inv_p.values())
            .expect("RabitQ metadata not loaded");

        let (query_codes, sq_min, sq_scale) = match inv_p.data_type() {
            DataType::Float16 => Self::quantize_query_vector::<Float16Type>(&inv_p, &qr),
            DataType::Float32 => Self::quantize_query_vector::<Float32Type>(&inv_p, &qr),
            DataType::Float64 => Self::quantize_query_vector::<Float64Type>(&inv_p, &qr),
            dt => unimplemented!("RabitQ does not support data type: {}", dt),
        };
        let q_factor = match self.distance_type {
            DistanceType::L2 => dist_q_c,
            DistanceType::Dot => dist_q_c - 1.0,
            _ => unimplemented!(
                "RabitQ does not support distance type: {}",
                self.distance_type
            ),
        };
        RabitDistCalculator::new(
            query_codes,
            sq_min,
            sq_scale,
            codes,
            self.code_bitcounts.values(),
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
        let code_bitcounts = batch[CODE_BITCOUNT_COLUMN]
            .as_primitive::<Float32Type>()
            .clone();
        let add_factors = batch[ADD_FACTORS_COLUMN]
            .as_primitive::<Float32Type>()
            .clone();
        let scale_factors = batch[SCALE_FACTORS_COLUMN]
            .as_primitive::<Float32Type>()
            .clone();

        let (batch, codes) = if !metadata.transposed {
            let num_code_bytes = codes.value_length();
            let codes = transpose(
                codes.values().as_primitive::<UInt8Type>(),
                row_ids.len(),
                num_code_bytes as usize,
            );
            let codes = FixedSizeListArray::try_new_from_values(codes, num_code_bytes)?;
            let batch = batch.replace_column_by_name(RABIT_CODE_COLUMN, Arc::new(codes))?;
            let codes = batch[RABIT_CODE_COLUMN].as_fixed_size_list().clone();
            (batch, codes)
        } else {
            (batch, codes)
        };

        let mut metadata = metadata.clone();
        metadata.transposed = true;

        Ok(Self {
            metadata,
            batch,
            distance_type,
            row_ids,
            codes,
            code_bitcounts,
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
        let new_codes = UInt8Array::from(new_codes);
        let batch = if new_row_ids.is_empty() {
            RecordBatch::new_empty(self.schema().clone())
        } else {
            let codes = transpose(&new_codes, new_row_ids.len(), num_code_bytes);
            let codes = Arc::new(FixedSizeListArray::try_new_from_values(
                codes,
                num_code_bytes as i32,
            )?);

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
            code_bitcounts: self.code_bitcounts.clone(),
            add_factors: self.add_factors.clone(),
            scale_factors: self.scale_factors.clone(),
            row_ids: new_row_ids,
        })
    }
}

fn get_rq_code(
    codes: &[u8],
    id: usize,
    num_vectors: usize,
    num_code_bytes: usize,
) -> impl Iterator<Item = u8> + '_ {
    codes
        .iter()
        .skip(id)
        .step_by(num_vectors)
        .copied()
        .exact_size(num_code_bytes)
}
