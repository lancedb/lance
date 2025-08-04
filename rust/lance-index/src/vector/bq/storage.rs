// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::{Float16Type, Float32Type, Float64Type, UInt64Type, UInt8Type};
use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, SchemaRef};
use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use deepsize::DeepSizeOf;
use itertools::Itertools;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result, ROW_ID};
use lance_file::reader::FileReader;
use lance_linalg::distance::DistanceType;
use lance_linalg::kernels::{self, normalize_arrow};
use lance_table::utils::LanceIteratorExtension;
use num_traits::AsPrimitive;
use prost::Message;
use serde::{Deserialize, Serialize};
use snafu::location;

use crate::frag_reuse::FragReuseIndex;
use crate::pb;
use crate::vector::bq::transform::{CODE_BITCOUNT_COLUMN, IP_RQ_CENTROID_COLUMN, IP_RQ_RES_COLUMN};
use crate::vector::quantizer::{QuantizerMetadata, QuantizerStorage};
use crate::vector::sq::scale_to_u8;
use crate::vector::storage::{DistCalculator, VectorStore};
use crate::vector::CENTROID_DIST_COLUMN;

pub const RABIT_METADATA_KEY: &str = "lance:rabit";
pub const RABIT_CODE_COLUMN: &str = "_rabit_codes";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RabitQuantizationMetadata {
    #[serde(skip)]
    pub inv_p: ndarray::ArcArray2<f32>,
    pub inv_p_position: u32,
    pub num_bits: u8,
}

impl DeepSizeOf for RabitQuantizationMetadata {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.inv_p.len() * std::mem::size_of::<f32>()
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
        let inv_p = FixedSizeListArray::try_from(&codebook_tensor)?;
        let shape = (inv_p.len(), inv_p.len());
        let inv_p = inv_p
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        let inv_p = ndarray::ArcArray2::from_shape_vec(shape, inv_p).map_err(|e| Error::Index {
            message: format!("failed to parse matrix: {}", e),
            location: location!(),
        })?;
        self.inv_p = inv_p;
        Ok(())
    }

    fn extra_metadata(&self) -> Result<Option<Bytes>> {
        let inv_p_flatten = Float32Array::from_iter_values(self.inv_p.iter().copied());
        let inv_p_fsl =
            FixedSizeListArray::try_new_from_values(inv_p_flatten, self.inv_p.nrows() as i32)?;
        let inv_p_tensor = pb::Tensor::try_from(&inv_p_fsl)?;
        let mut bytes = BytesMut::new();
        inv_p_tensor.encode(&mut bytes)?;
        Ok(Some(bytes.freeze()))
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
    centroid_dists: Float32Array,
    codes: FixedSizeListArray,
    code_bitcounts: Float32Array,
    ip_rq_res: Float32Array,
    ip_rq_centroid: Float32Array,
}

impl DeepSizeOf for RabitQuantizationStorage {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.metadata.deep_size_of_children(context) + self.batch.get_array_memory_size()
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
    centroid_dists: &'a [f32],
    code_bitcounts: &'a [f32],
    ip_rq_res: &'a [f32],
    ip_rq_centroid: &'a [f32],

    // factors for all computing distances to all vectors
    // |v-c|^2  + 2 * |v-c|^2 * ip_rq_centroid[i] / ip_rq_res[i]
    // add_factor: f32,
    // -2 * |v-c| / ip_rq_res[i] * |v-c| = -2 * |v-c|^2 / ip_rq_res[i]
    // scale_factor: f32,

    // factors for query vector
    // -(2^B - 1)/2 * sum(q') where B is the number of bits per dimension for RQ (not SQ)
    // it's 0 for 1 bit RQ
    // kbit_x_sq_sum: f32,
    // |(q-c)| ^ 2
    // qr_norm_square: f32,
    qc_norm_square: f32,

    sq_min: f32,
    sq_delta: f32,
    sq_scale: f32,
    sq_sum: f32,
    sqrt_d: f32,
}

impl<'a> RabitDistCalculator<'a> {
    pub fn new(
        qr: &dyn Array,
        // centroid: &dyn Array,
        codes: &'a [u8],
        centroid_dists: &'a [f32],
        code_bitcounts: &'a [f32],
        ip_rq_res: &'a [f32],
        ip_rq_centroid: &'a [f32],
        inv_p: ndarray::ArrayView2<'a, f32>,
    ) -> Self {
        // let qr = arrow_arith::numeric::sub(&q, &centroid).unwrap();

        let qr = qr.as_primitive::<Float32Type>().values();
        let q_norm_square = qr.iter().map(|&v| v * v).sum::<f32>();
        let qr = ndarray::ArrayView2::from_shape((qr.len(), 1), qr).unwrap();
        let rotated_qr = inv_p.dot(&qr);
        let rotated_qr = rotated_qr.as_slice().unwrap();
        let (min, max) = rotated_qr
            .iter()
            .copied()
            .minmax()
            .into_option()
            .expect("failed to get min and max of query vector");
        let bounds = min.as_()..max.as_();
        let query_codes = scale_to_u8::<Float32Type>(&rotated_qr, &bounds);
        let sq_sum = query_codes.iter().map(|&v| v as u32).sum::<u32>() as f32;

        let dist_table = Self::build_dist_table(query_codes);
        Self {
            dim: qr.len(),
            num_bits: 1,
            codes,
            dist_table,
            centroid_dists,
            code_bitcounts,
            ip_rq_res,
            ip_rq_centroid,
            qc_norm_square: q_norm_square,
            sq_min: min,
            sq_delta: (max - min),
            sq_scale: (max - min) / 255.0,
            sq_sum,
            sqrt_d: (qr.len() as f32).sqrt(),
        }
    }

    fn build_dist_table(query_codes: Vec<u8>) -> Vec<u32> {
        // TODO: optimize this with SIMD
        query_codes
            .chunks_exact(4)
            .flat_map(|sub_vec| {
                (0..16).map(|j| {
                    let mut dist = 0;
                    for (b, &v) in sub_vec.iter().enumerate() {
                        dist += (v * ((j >> b) & 0x1)) as u32;
                    }
                    dist
                })
            })
            .exact_size(query_codes.len() * 4)
            .collect()
    }
}

// TODO: optimize this with SIMD
impl DistCalculator for RabitDistCalculator<'_> {
    fn distance(&self, id: u32) -> f32 {
        let mut dist = 0;
        let id = id as usize;
        let code_len = self.dim * (self.num_bits as usize) / 8;
        let code = &self.codes[id * code_len..(id + 1) * code_len];
        for (i, code_byte) in code.iter().enumerate() {
            // code is a bit vector, we iterate over 8 bits at a time,
            // every 4 bits is a sub-vector, we need to extract the bits
            let dist_table = &self.dist_table[2 * i * 16..(2 * i + 1) * 16];
            let next_dist_table = &self.dist_table[(2 * i + 1) * 16..(2 * i + 2) * 16];
            let current_code = (code_byte & 0x0F) as usize;
            let next_code = (code_byte >> 4) as usize;
            dist += dist_table[current_code];
            dist += next_dist_table[next_code];
        }

        let vec_bitcount = self.code_bitcounts[id];
        let dist = dist as f32;

        // distance between quantized vector and query vector
        let dist_vq_qr = (2.0 * self.sq_scale * dist + 2.0 * self.sq_min * vec_bitcount
            - self.sq_scale * self.sq_sum
            - self.dim as f32 * self.sq_min)
            / self.sqrt_d;

        let dist_v_q = dist_vq_qr / self.ip_rq_res[id];
        let vr_norm_square = self.centroid_dists[id];
        vr_norm_square + self.qc_norm_square - 2.0 * vr_norm_square * dist_v_q
    }

    fn distance_all(&self, _: usize) -> Vec<f32> {
        let code_len = self.dim * (self.num_bits as usize) / 8;
        let n = self.codes.len() / code_len;
        let mut dists = vec![0.0; n];
        for (i, dist) in dists.iter_mut().enumerate() {
            *dist = self.distance(i as u32);
        }
        dists
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
    fn dist_calculator(&self, qr: Arc<dyn Array>) -> Self::DistanceCalculator<'_> {
        let codes = self.codes.values().as_primitive::<UInt8Type>().values();
        RabitDistCalculator::new(
            qr.as_ref(),
            codes,
            self.centroid_dists.values(),
            self.code_bitcounts.values(),
            self.ip_rq_res.values(),
            self.ip_rq_centroid.values(),
            self.metadata.inv_p.view(),
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
        let centroid_dists = batch[CENTROID_DIST_COLUMN]
            .as_primitive::<Float32Type>()
            .clone();
        let code_bitcounts = batch[CODE_BITCOUNT_COLUMN]
            .as_primitive::<Float32Type>()
            .clone();
        let ip_rq_res = batch[IP_RQ_RES_COLUMN]
            .as_primitive::<Float32Type>()
            .clone();
        let ip_rq_centroid = batch[IP_RQ_CENTROID_COLUMN]
            .as_primitive::<Float32Type>()
            .clone();
        Ok(Self {
            metadata: metadata.clone(),
            batch,
            distance_type,
            row_ids,
            codes,
            centroid_dists,
            code_bitcounts,
            ip_rq_res,
            ip_rq_centroid,
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
}
