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
use lance_table::utils::LanceIteratorExtension;
use num_traits::AsPrimitive;
use prost::Message;
use serde::{Deserialize, Serialize};
use snafu::location;

use crate::frag_reuse::FragReuseIndex;
use crate::pb;
use crate::vector::quantizer::{QuantizerMetadata, QuantizerStorage};
use crate::vector::sq::scale_to_u8;
use crate::vector::storage::{DistCalculator, VectorStore};

pub const RABBIT_METADATA_KEY: &str = "lance:rabbit";
pub const RABBIT_CODE_COLUMN: &str = "_rabbit_codes";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RabbitQuantizationMetadata {
    #[serde(skip)]
    pub inv_p: ndarray::ArcArray2<f32>,
    pub inv_p_position: u32,
    pub num_bits: u8,
}

impl DeepSizeOf for RabbitQuantizationMetadata {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.inv_p.len() * std::mem::size_of::<f32>()
    }
}

#[async_trait]
impl QuantizerMetadata for RabbitQuantizationMetadata {
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
                .get(RABBIT_METADATA_KEY)
                .ok_or(Error::Index {
                    message: format!(
                        "Reading Rabbit metadata: metadata key {} not found",
                        RABBIT_METADATA_KEY
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
pub struct RabbitQuantizationStorage {
    metadata: RabbitQuantizationMetadata,
    batch: RecordBatch,
    distance_type: DistanceType,

    // helper fields
    row_ids: UInt64Array,
    codes: FixedSizeListArray,
}

impl DeepSizeOf for RabbitQuantizationStorage {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.metadata.deep_size_of_children(context) + self.batch.get_array_memory_size()
    }
}

pub struct RabbitDistCalculator<'a> {
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
}

impl<'a> RabbitDistCalculator<'a> {
    // norm_qr = (q-c)/|q-c|
    pub fn new(norm_qr: &dyn Array, codes: &'a [u8]) -> Self {
        let query_codes = match norm_qr.data_type() {
            DataType::Float16 => {
                let norm_qr = norm_qr.as_primitive::<Float16Type>().values();
                let (min, max) = norm_qr
                    .iter()
                    .copied()
                    .minmax()
                    .into_option()
                    .expect("failed to get min and max of query vector");
                let bounds = min.as_()..max.as_();
                scale_to_u8::<Float16Type>(norm_qr, &bounds)
            }
            DataType::Float32 => {
                let norm_qr = norm_qr.as_primitive::<Float32Type>().values();
                let (min, max) = norm_qr
                    .iter()
                    .copied()
                    .minmax()
                    .into_option()
                    .expect("failed to get min and max of query vector");
                let bounds = min.as_()..max.as_();
                scale_to_u8::<Float32Type>(norm_qr, &bounds)
            }
            DataType::Float64 => {
                let norm_qr = norm_qr.as_primitive::<Float64Type>().values();
                let (min, max) = norm_qr
                    .iter()
                    .copied()
                    .minmax()
                    .into_option()
                    .expect("failed to get min and max of query vector");
                let bounds = min..max;
                scale_to_u8::<Float64Type>(norm_qr, &bounds)
            }
            _ => {
                unimplemented!("unsupported data type for RabbitQ: {}", norm_qr.data_type());
            }
        };

        let dist_table = Self::build_dist_table(query_codes);
        Self {
            dim: norm_qr.len(),
            num_bits: 1,
            codes,
            dist_table,
        }
    }

    fn build_dist_table(query_codes: Vec<u8>) -> Vec<u32> {
        // TODO: optimize this with SIMD
        query_codes
            .chunks_exact(4)
            .flat_map(|sub_vec| {
                (0..16).map(|j| {
                    let mut dist = 0;
                    for (b, v) in sub_vec.iter().enumerate() {
                        dist += (*v * ((j >> b) & 0x1)) as u32;
                    }
                    dist
                })
            })
            .exact_size(query_codes.len() * 4)
            .collect()
    }
}

// TODO: optimize this with SIMD
impl DistCalculator for RabbitDistCalculator<'_> {
    fn distance(&self, id: u32) -> f32 {
        let mut dist = 0;
        let id = id as usize;
        let code_len = self.dim * (self.num_bits as usize) / 8;
        let code = &self.codes[id * code_len..(id + 1) * code_len];
        for code_byte in code.iter() {
            // code is a bit vector, we need to extract the bits
            let current_code = (code_byte & 0x0F) as usize;
            let next_code = (code_byte >> 4) as usize;
            dist += self.dist_table[current_code];
            dist += self.dist_table[next_code];
        }
        dist as f32
    }

    fn distance_all(&self, _: usize) -> Vec<f32> {
        let code_len = self.dim * (self.num_bits as usize) / 8;
        let len = self.codes.len() / code_len;
        let mut dists = vec![0.0; len];
        for (i, dist) in dists.iter_mut().enumerate() {
            *dist = self.distance(i as u32);
        }
        dists
    }
}

impl VectorStore for RabbitQuantizationStorage {
    type DistanceCalculator<'a> = RabbitDistCalculator<'a>;

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
        unimplemented!("RabbitQ does not support append_batch")
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

    fn dist_calculator(&self, norm_qr: Arc<dyn Array>) -> Self::DistanceCalculator<'_> {
        let codes = self.codes.values().as_primitive::<UInt8Type>().values();
        RabbitDistCalculator::new(norm_qr.as_ref(), codes)
    }

    // TODO: implement this
    // This method is required for HNSW, we can't support HNSW_RABBIT before this is implemented
    fn dist_calculator_from_id(&self, _: u32) -> Self::DistanceCalculator<'_> {
        unimplemented!("RabbitQ does not support dist_calculator_from_id")
    }
}

#[async_trait]
impl QuantizerStorage for RabbitQuantizationStorage {
    type Metadata = RabbitQuantizationMetadata;

    fn try_from_batch(
        batch: RecordBatch,
        metadata: &Self::Metadata,
        distance_type: DistanceType,
        _fri: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>().clone();
        let codes = batch[RABBIT_CODE_COLUMN].as_fixed_size_list().clone();
        Ok(Self {
            metadata: metadata.clone(),
            batch,
            distance_type,
            row_ids,
            codes,
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
