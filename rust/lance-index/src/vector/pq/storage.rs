// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Product Quantization storage
//!
//! Used as storage backend for Graph based algorithms.

use std::{cmp::min, collections::HashMap, sync::Arc};

use arrow::datatypes::{self, UInt8Type};
use arrow_array::{
    cast::AsArray,
    types::{Float32Type, UInt64Type},
    FixedSizeListArray, RecordBatch, UInt64Array, UInt8Array,
};
use arrow_array::{Array, ArrayRef, ArrowPrimitiveType, PrimitiveArray};
use arrow_schema::{DataType, SchemaRef};
use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use deepsize::DeepSizeOf;
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_core::{Error, Result, ROW_ID};
use lance_file::{reader::FileReader, writer::FileWriter};
use lance_io::{object_store::ObjectStore, utils::read_message};
use lance_linalg::distance::{DistanceType, Dot, L2};
use lance_table::utils::LanceIteratorExtension;
use lance_table::{format::SelfDescribingFileReader, io::manifest::ManifestDescribing};
use object_store::path::Path;
use prost::Message;
use serde::{Deserialize, Serialize};
use snafu::location;

use super::distance::{build_distance_table_dot, build_distance_table_l2, compute_pq_distance};
use super::ProductQuantizer;
use crate::frag_reuse::FragReuseIndex;
use crate::{
    pb,
    vector::{
        pq::transform::PQTransformer,
        quantizer::{QuantizerMetadata, QuantizerStorage},
        storage::{DistCalculator, VectorStore},
        transform::Transformer,
        PQ_CODE_COLUMN,
    },
    IndexMetadata, INDEX_METADATA_SCHEMA_KEY,
};

pub const PQ_METADATA_KEY: &str = "lance:pq";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizationMetadata {
    pub codebook_position: usize,
    pub nbits: u32,
    pub num_sub_vectors: usize,
    pub dimension: usize,

    #[serde(skip)]
    pub codebook: Option<FixedSizeListArray>,

    // empty for v1 format
    // used for v3 format
    // deprecated in later version
    pub codebook_tensor: Vec<u8>,
    pub transposed: bool,
}

impl DeepSizeOf for ProductQuantizationMetadata {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.codebook
            .as_ref()
            .map(|codebook| codebook.get_array_memory_size())
            .unwrap_or(0)
    }
}

impl PartialEq for ProductQuantizationMetadata {
    fn eq(&self, other: &Self) -> bool {
        self.num_sub_vectors == other.num_sub_vectors
            && self.nbits == other.nbits
            && self.dimension == other.dimension
            && self.codebook == other.codebook
    }
}

#[async_trait]
impl QuantizerMetadata for ProductQuantizationMetadata {
    fn buffer_index(&self) -> Option<u32> {
        if self.codebook_position > 0 {
            // the global buffer index starts from 1
            Some(self.codebook_position as u32)
        } else {
            None
        }
    }

    fn set_buffer_index(&mut self, index: u32) {
        self.codebook_position = index as usize;
    }

    fn parse_buffer(&mut self, bytes: Bytes) -> Result<()> {
        debug_assert!(!bytes.is_empty());
        debug_assert!(self.codebook.is_none());
        let codebook_tensor: pb::Tensor = pb::Tensor::decode(bytes)?;
        self.codebook = Some(FixedSizeListArray::try_from(&codebook_tensor)?);
        Ok(())
    }

    fn extra_metadata(&self) -> Result<Option<Bytes>> {
        debug_assert!(self.codebook.is_some());
        let codebook_tensor: pb::Tensor = pb::Tensor::try_from(self.codebook.as_ref().unwrap())?;
        let mut bytes = BytesMut::new();
        codebook_tensor.encode(&mut bytes)?;
        Ok(Some(bytes.freeze()))
    }

    async fn load(reader: &FileReader) -> Result<Self> {
        let metadata = reader
            .schema()
            .metadata
            .get(PQ_METADATA_KEY)
            .ok_or(Error::Index {
                message: format!(
                    "Reading PQ storage: metadata key {} not found",
                    PQ_METADATA_KEY
                ),
                location: location!(),
            })?;
        let mut metadata: Self = serde_json::from_str(metadata).map_err(|_| Error::Index {
            message: format!("Failed to parse PQ metadata: {}", metadata),
            location: location!(),
        })?;

        debug_assert!(metadata.codebook.is_none());
        debug_assert!(metadata.codebook_tensor.is_empty());

        let codebook_tensor: pb::Tensor =
            read_message(reader.object_reader.as_ref(), metadata.codebook_position).await?;
        metadata.codebook = Some(FixedSizeListArray::try_from(&codebook_tensor)?);
        Ok(metadata)
    }
}

/// Product Quantization Storage
///
/// It stores PQ code, as well as the row ID to the original vectors.
///
/// It is possible to store additional metadata to accelerate filtering later.
#[derive(Clone, Debug)]
pub struct ProductQuantizationStorage {
    metadata: ProductQuantizationMetadata,
    distance_type: DistanceType,
    batch: RecordBatch,

    // For easy access
    pq_code: Arc<UInt8Array>,
    row_ids: Arc<UInt64Array>,
}

impl DeepSizeOf for ProductQuantizationStorage {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.batch.get_array_memory_size()
            + self
                .metadata
                .codebook
                .as_ref()
                .map(|codebook| codebook.get_array_memory_size())
                .unwrap_or(0)
    }
}

impl PartialEq for ProductQuantizationStorage {
    fn eq(&self, other: &Self) -> bool {
        self.distance_type == other.distance_type
            && self.metadata.eq(&other.metadata)
            && self.batch.columns().eq(other.batch.columns())
    }
}

impl ProductQuantizationStorage {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        codebook: FixedSizeListArray,
        mut batch: RecordBatch,
        num_bits: u32,
        num_sub_vectors: usize,
        dimension: usize,
        distance_type: DistanceType,
        transposed: bool,
        fri: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        if batch.num_columns() != 2 {
            log::warn!(
                "PQ storage should have 2 columns, but got {} columns: {}",
                batch.num_columns(),
                batch.schema(),
            );
            batch = batch.project(&[
                batch.schema().index_of(ROW_ID)?,
                batch.schema().index_of(PQ_CODE_COLUMN)?,
            ])?;
        }

        let Some(row_ids) = batch.column_by_name(ROW_ID) else {
            return Err(Error::Index {
                message: "Row ID column not found from PQ storage".to_string(),
                location: location!(),
            });
        };
        let row_ids: Arc<UInt64Array> = row_ids
            .as_primitive_opt::<UInt64Type>()
            .ok_or(Error::Index {
                message: "Row ID column is not of type UInt64".to_string(),
                location: location!(),
            })?
            .clone()
            .into();

        if !transposed {
            let num_sub_vectors_in_byte = if num_bits == 4 {
                num_sub_vectors / 2
            } else {
                num_sub_vectors
            };
            let pq_col = batch[PQ_CODE_COLUMN].as_fixed_size_list();
            let transposed_code = transpose(
                pq_col.values().as_primitive::<UInt8Type>(),
                row_ids.len(),
                num_sub_vectors_in_byte,
            );
            let pq_code_fsl = Arc::new(FixedSizeListArray::try_new_from_values(
                transposed_code,
                num_sub_vectors_in_byte as i32,
            )?);
            batch = batch.replace_column_by_name(PQ_CODE_COLUMN, pq_code_fsl)?;
        }

        let mut pq_code: Arc<UInt8Array> = batch[PQ_CODE_COLUMN]
            .as_fixed_size_list()
            .values()
            .as_primitive()
            .clone()
            .into();

        if let Some(fri_ref) = fri.as_ref() {
            let transposed_codes = pq_code.values();
            let mut new_row_ids = Vec::with_capacity(row_ids.len());
            let mut new_codes = Vec::with_capacity(row_ids.len() * num_sub_vectors);

            let row_ids_values = row_ids.values();
            for (i, row_id) in row_ids_values.iter().enumerate() {
                if let Some(mapped_value) = fri_ref.remap_row_id(*row_id) {
                    new_row_ids.push(mapped_value);
                    new_codes.extend(get_pq_code(
                        transposed_codes,
                        num_bits,
                        num_sub_vectors,
                        i as u32,
                    ));
                }
            }

            let new_row_ids = Arc::new(UInt64Array::from(new_row_ids));
            let new_codes = UInt8Array::from(new_codes);
            batch = if new_row_ids.is_empty() {
                RecordBatch::new_empty(batch.schema())
            } else {
                let num_bytes_in_code = new_codes.len() / new_row_ids.len();
                let new_transposed_codes =
                    transpose(&new_codes, new_row_ids.len(), num_bytes_in_code);
                let codes_fsl = Arc::new(FixedSizeListArray::try_new_from_values(
                    new_transposed_codes,
                    num_bytes_in_code as i32,
                )?);
                RecordBatch::try_new(batch.schema(), vec![new_row_ids, codes_fsl])?
            };
            pq_code = batch[PQ_CODE_COLUMN]
                .as_fixed_size_list()
                .values()
                .as_primitive::<UInt8Type>()
                .clone()
                .into();
        }

        let distance_type = match distance_type {
            DistanceType::Cosine => DistanceType::L2,
            _ => distance_type,
        };
        let metadata = ProductQuantizationMetadata {
            codebook_position: 0,
            nbits: num_bits,
            num_sub_vectors,
            dimension,
            codebook: Some(codebook),
            codebook_tensor: Vec::new(), // empty for v1 format
            transposed: true,
        };
        Ok(Self {
            metadata,
            distance_type,
            batch,
            pq_code,
            row_ids,
        })
    }

    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    /// Build a PQ storage from ProductQuantizer and a RecordBatch.
    ///
    /// Parameters
    /// ----------
    /// quantizer: ProductQuantizer
    ///    The quantizer used to transform the vectors.
    /// batch: RecordBatch
    ///   The batch of vectors to be transformed.
    /// vector_col: &str
    ///   The name of the column containing the vectors.
    pub async fn build(
        quantizer: ProductQuantizer,
        batch: &RecordBatch,
        vector_col: &str,
        fri: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let codebook = quantizer.codebook.clone();
        let num_bits = quantizer.num_bits;
        let dimension = quantizer.dimension;
        let num_sub_vectors = quantizer.num_sub_vectors;
        let metric_type = quantizer.distance_type;
        let transform = PQTransformer::new(quantizer, vector_col, PQ_CODE_COLUMN);
        let batch = transform.transform(batch)?;
        Self::new(
            codebook,
            batch,
            num_bits,
            num_sub_vectors,
            dimension,
            metric_type,
            false,
            fri,
        )
    }

    pub fn codebook(&self) -> &FixedSizeListArray {
        self.metadata.codebook.as_ref().unwrap()
    }

    /// Load full PQ storage from disk.
    ///
    /// Parameters
    /// ----------
    /// object_store: &ObjectStore
    ///   The object store to load the storage from.
    /// path: &Path
    ///  The path to the storage.
    ///
    /// Returns
    /// --------
    /// Self
    ///
    /// Currently it loads everything in memory.
    /// TODO: support lazy loading later.
    pub async fn load(
        object_store: &ObjectStore,
        path: &Path,
        fri: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let reader = FileReader::try_new_self_described(object_store, path, None).await?;
        let schema = reader.schema();

        let metadata_str = schema
            .metadata
            .get(INDEX_METADATA_SCHEMA_KEY)
            .ok_or(Error::Index {
                message: format!(
                    "Reading PQ storage: index key {} not found",
                    INDEX_METADATA_SCHEMA_KEY
                ),
                location: location!(),
            })?;
        let index_metadata: IndexMetadata =
            serde_json::from_str(metadata_str).map_err(|_| Error::Index {
                message: format!("Failed to parse index metadata: {}", metadata_str),
                location: location!(),
            })?;
        let distance_type: DistanceType =
            DistanceType::try_from(index_metadata.distance_type.as_str())?;

        let metadata = ProductQuantizationMetadata::load(&reader).await?;
        Self::load_partition(&reader, 0..reader.len(), distance_type, &metadata, fri).await
    }

    pub fn schema(&self) -> SchemaRef {
        self.batch.schema()
    }

    pub fn get_row_ids(&self, ids: &[u32]) -> Vec<u64> {
        ids.iter()
            .map(|&id| self.row_ids.value(id as usize))
            .collect()
    }

    /// Write the PQ storage as a Lance partition to disk,
    /// and returns the number of rows written.
    ///
    pub async fn write_partition(
        &self,
        writer: &mut FileWriter<ManifestDescribing>,
    ) -> Result<usize> {
        let batch_size: usize = 10240; // TODO: make it configurable
        for offset in (0..self.batch.num_rows()).step_by(batch_size) {
            let length = min(batch_size, self.batch.num_rows() - offset);
            let slice = self.batch.slice(offset, length);
            writer.write(&[slice]).await?;
        }
        Ok(self.batch.num_rows())
    }
}

pub fn transpose<T: ArrowPrimitiveType>(
    original: &PrimitiveArray<T>,
    num_rows: usize,
    num_columns: usize,
) -> PrimitiveArray<T>
where
    PrimitiveArray<T>: From<Vec<T::Native>>,
{
    if original.is_empty() {
        return original.clone();
    }

    let mut transposed_codes = vec![T::default_value(); original.len()];
    for (vec_idx, codes) in original.values().chunks_exact(num_columns).enumerate() {
        for (sub_vec_idx, code) in codes.iter().enumerate() {
            transposed_codes[sub_vec_idx * num_rows + vec_idx] = *code;
        }
    }

    transposed_codes.into()
}

#[async_trait]
impl QuantizerStorage for ProductQuantizationStorage {
    type Metadata = ProductQuantizationMetadata;

    fn try_from_batch(
        batch: RecordBatch,
        metadata: &Self::Metadata,
        distance_type: DistanceType,
        fri: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        let distance_type = match distance_type {
            DistanceType::Cosine => DistanceType::L2,
            _ => distance_type,
        };

        // now it supports only Float32Type
        let codebook = match &metadata.codebook {
            Some(codebook) => codebook.clone(),
            None => {
                // legacy format would contains codebook tensor but not codebook
                debug_assert!(!metadata.codebook_tensor.is_empty());
                let codebook_tensor = pb::Tensor::decode(metadata.codebook_tensor.as_slice())?;
                FixedSizeListArray::try_from(&codebook_tensor)?
            }
        };

        Self::new(
            codebook,
            batch,
            metadata.nbits,
            metadata.num_sub_vectors,
            metadata.dimension,
            distance_type,
            metadata.transposed,
            fri,
        )
    }

    fn metadata(&self) -> &Self::Metadata {
        &self.metadata
    }

    // we can't use the default implementation of remap,
    // because PQ Storage transposed the PQ codes
    fn remap(&self, mapping: &HashMap<u64, Option<u64>>) -> Result<Self> {
        let transposed_codes = self.pq_code.values();
        let mut new_row_ids = Vec::with_capacity(self.len());
        let mut new_codes = Vec::with_capacity(self.len() * self.metadata.num_sub_vectors);

        let row_ids = self.row_ids.values();
        for (i, row_id) in row_ids.iter().enumerate() {
            match mapping.get(row_id) {
                Some(Some(new_id)) => {
                    new_row_ids.push(*new_id);
                    new_codes.extend(get_pq_code(
                        transposed_codes,
                        self.metadata.nbits,
                        self.metadata.num_sub_vectors,
                        i as u32,
                    ));
                }
                Some(None) => {}
                None => {
                    new_row_ids.push(*row_id);
                    new_codes.extend(get_pq_code(
                        transposed_codes,
                        self.metadata.nbits,
                        self.metadata.num_sub_vectors,
                        i as u32,
                    ));
                }
            }
        }

        let new_row_ids = Arc::new(UInt64Array::from(new_row_ids));
        let new_codes = UInt8Array::from(new_codes);
        let batch = if new_row_ids.is_empty() {
            RecordBatch::new_empty(self.schema())
        } else {
            let num_bytes_in_code = new_codes.len() / new_row_ids.len();
            let new_transposed_codes = transpose(&new_codes, new_row_ids.len(), num_bytes_in_code);
            let codes_fsl = Arc::new(FixedSizeListArray::try_new_from_values(
                new_transposed_codes,
                num_bytes_in_code as i32,
            )?);
            RecordBatch::try_new(self.schema(), vec![new_row_ids.clone(), codes_fsl])?
        };
        let transposed_codes = batch[PQ_CODE_COLUMN]
            .as_fixed_size_list()
            .values()
            .as_primitive::<UInt8Type>()
            .clone();

        Ok(Self {
            metadata: self.metadata.clone(),
            distance_type: self.distance_type,
            batch,
            pq_code: Arc::new(transposed_codes),
            row_ids: new_row_ids,
        })
    }

    /// Load a partition of PQ storage from disk.
    ///
    /// Parameters
    /// ----------
    /// - *reader: &FileReader
    async fn load_partition(
        reader: &FileReader,
        range: std::ops::Range<usize>,
        distance_type: DistanceType,
        metadata: &Self::Metadata,
        fri: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        // Hard coded to float32 for now
        let codebook = metadata
            .codebook
            .as_ref()
            .ok_or(Error::Index {
                message: "Codebook not found in PQ metadata".to_string(),
                location: location!(),
            })?
            .values()
            .as_primitive::<Float32Type>()
            .clone();

        let codebook =
            FixedSizeListArray::try_new_from_values(codebook, metadata.dimension as i32)?;

        let schema = reader.schema();
        let batch = reader.read_range(range, schema).await?;

        Self::new(
            codebook,
            batch,
            metadata.nbits,
            metadata.num_sub_vectors,
            metadata.dimension,
            distance_type,
            metadata.transposed,
            fri,
        )
    }
}

impl VectorStore for ProductQuantizationStorage {
    type DistanceCalculator<'a> = PQDistCalculator;

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch>> {
        Ok(std::iter::once(self.batch.clone()))
    }

    fn append_batch(&self, _batch: RecordBatch, _vector_column: &str) -> Result<Self> {
        unimplemented!()
    }

    fn schema(&self) -> &SchemaRef {
        self.batch.schema_ref()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn len(&self) -> usize {
        self.batch.num_rows()
    }

    fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    fn row_id(&self, id: u32) -> u64 {
        self.row_ids.values()[id as usize]
    }

    fn row_ids(&self) -> impl Iterator<Item = &u64> {
        self.row_ids.values().iter()
    }

    fn dist_calculator(&self, query: ArrayRef) -> Self::DistanceCalculator<'_> {
        let codebook = self.metadata.codebook.as_ref().unwrap();
        match codebook.value_type() {
            DataType::Float16 => PQDistCalculator::new(
                codebook
                    .values()
                    .as_primitive::<datatypes::Float16Type>()
                    .values(),
                self.metadata.nbits,
                self.metadata.num_sub_vectors,
                self.pq_code.clone(),
                query.as_primitive::<datatypes::Float16Type>().values(),
                self.distance_type,
            ),
            DataType::Float32 => PQDistCalculator::new(
                codebook
                    .values()
                    .as_primitive::<datatypes::Float32Type>()
                    .values(),
                self.metadata.nbits,
                self.metadata.num_sub_vectors,
                self.pq_code.clone(),
                query.as_primitive::<datatypes::Float32Type>().values(),
                self.distance_type,
            ),
            DataType::Float64 => PQDistCalculator::new(
                codebook
                    .values()
                    .as_primitive::<datatypes::Float64Type>()
                    .values(),
                self.metadata.nbits,
                self.metadata.num_sub_vectors,
                self.pq_code.clone(),
                query.as_primitive::<datatypes::Float64Type>().values(),
                self.distance_type,
            ),
            _ => unimplemented!("Unsupported data type: {:?}", codebook.value_type()),
        }
    }

    fn dist_calculator_from_id(&self, id: u32) -> Self::DistanceCalculator<'_> {
        let codes = get_pq_code(
            self.pq_code.values(),
            self.metadata.nbits,
            self.metadata.num_sub_vectors,
            id,
        );
        let codebook = self.metadata.codebook.as_ref().unwrap();
        match codebook.value_type() {
            DataType::Float16 => {
                let codebook = codebook
                    .values()
                    .as_primitive::<datatypes::Float16Type>()
                    .values();
                let query = get_centroids(
                    codebook,
                    self.metadata.nbits,
                    self.metadata.num_sub_vectors,
                    self.metadata.dimension,
                    codes,
                );
                PQDistCalculator::new(
                    codebook,
                    self.metadata.nbits,
                    self.metadata.num_sub_vectors,
                    self.pq_code.clone(),
                    &query,
                    self.distance_type,
                )
            }
            DataType::Float32 => {
                let codebook = codebook
                    .values()
                    .as_primitive::<datatypes::Float32Type>()
                    .values();
                let query = get_centroids(
                    codebook,
                    self.metadata.nbits,
                    self.metadata.num_sub_vectors,
                    self.metadata.dimension,
                    codes,
                );
                PQDistCalculator::new(
                    codebook,
                    self.metadata.nbits,
                    self.metadata.num_sub_vectors,
                    self.pq_code.clone(),
                    &query,
                    self.distance_type,
                )
            }
            DataType::Float64 => {
                let codebook = codebook
                    .values()
                    .as_primitive::<datatypes::Float64Type>()
                    .values();
                let query = get_centroids(
                    codebook,
                    self.metadata.nbits,
                    self.metadata.num_sub_vectors,
                    self.metadata.dimension,
                    codes,
                );
                PQDistCalculator::new(
                    codebook,
                    self.metadata.nbits,
                    self.metadata.num_sub_vectors,
                    self.pq_code.clone(),
                    &query,
                    self.distance_type,
                )
            }
            _ => unimplemented!("Unsupported data type: {:?}", codebook.value_type()),
        }
    }

    fn dist_between(&self, u: u32, v: u32) -> f32 {
        // this is a fast way to compute distance between two vectors in the same storage.
        // it doesn't construct the distance table.
        let pq_codes = self.pq_code.values();
        let u_codes = get_pq_code(
            pq_codes,
            self.metadata.nbits,
            self.metadata.num_sub_vectors,
            u,
        );
        let v_codes = get_pq_code(
            pq_codes,
            self.metadata.nbits,
            self.metadata.num_sub_vectors,
            v,
        );
        let codebook = self.metadata.codebook.as_ref().unwrap();

        match codebook.value_type() {
            DataType::Float16 => {
                let qu = get_centroids(
                    codebook
                        .values()
                        .as_primitive::<datatypes::Float16Type>()
                        .values(),
                    self.metadata.nbits,
                    self.metadata.num_sub_vectors,
                    self.metadata.dimension,
                    u_codes,
                );
                let qv = get_centroids(
                    codebook
                        .values()
                        .as_primitive::<datatypes::Float16Type>()
                        .values(),
                    self.metadata.nbits,
                    self.metadata.num_sub_vectors,
                    self.metadata.dimension,
                    v_codes,
                );
                self.distance_type.func()(&qu, &qv)
            }
            DataType::Float32 => {
                let qu = get_centroids(
                    codebook
                        .values()
                        .as_primitive::<datatypes::Float32Type>()
                        .values(),
                    self.metadata.nbits,
                    self.metadata.num_sub_vectors,
                    self.metadata.dimension,
                    u_codes,
                );
                let qv = get_centroids(
                    codebook
                        .values()
                        .as_primitive::<datatypes::Float32Type>()
                        .values(),
                    self.metadata.nbits,
                    self.metadata.num_sub_vectors,
                    self.metadata.dimension,
                    v_codes,
                );
                self.distance_type.func()(&qu, &qv)
            }
            DataType::Float64 => {
                let qu = get_centroids(
                    codebook
                        .values()
                        .as_primitive::<datatypes::Float64Type>()
                        .values(),
                    self.metadata.nbits,
                    self.metadata.num_sub_vectors,
                    self.metadata.dimension,
                    u_codes,
                );
                let qv = get_centroids(
                    codebook
                        .values()
                        .as_primitive::<datatypes::Float64Type>()
                        .values(),
                    self.metadata.nbits,
                    self.metadata.num_sub_vectors,
                    self.metadata.dimension,
                    v_codes,
                );
                self.distance_type.func()(&qu, &qv)
            }
            _ => unimplemented!("Unsupported data type: {:?}", codebook.value_type()),
        }
    }
}

/// Distance calculator backed by PQ code.
pub struct PQDistCalculator {
    distance_table: Vec<f32>,
    pq_code: Arc<UInt8Array>,
    num_sub_vectors: usize,
    num_bits: u32,
    distance_type: DistanceType,
}

impl PQDistCalculator {
    fn new<T: L2 + Dot>(
        codebook: &[T],
        num_bits: u32,
        num_sub_vectors: usize,
        pq_code: Arc<UInt8Array>,
        query: &[T],
        distance_type: DistanceType,
    ) -> Self {
        let distance_table = match distance_type {
            DistanceType::L2 | DistanceType::Cosine => {
                build_distance_table_l2(codebook, num_bits, num_sub_vectors, query)
            }
            DistanceType::Dot => {
                build_distance_table_dot(codebook, num_bits, num_sub_vectors, query)
            }
            _ => unimplemented!("DistanceType is not supported: {:?}", distance_type),
        };
        Self {
            distance_table,
            num_sub_vectors,
            pq_code,
            num_bits,
            distance_type,
        }
    }

    fn get_pq_code(&self, id: u32) -> impl Iterator<Item = usize> + '_ {
        get_pq_code(
            self.pq_code.values(),
            self.num_bits,
            self.num_sub_vectors,
            id,
        )
        .map(|v| v as usize)
    }
}

impl DistCalculator for PQDistCalculator {
    fn distance(&self, id: u32) -> f32 {
        let num_centroids = 2_usize.pow(self.num_bits);
        let pq_code = self.get_pq_code(id);
        let diff = self.num_sub_vectors as f32 - 1.0;
        let dist = if self.num_bits == 4 {
            pq_code
                .enumerate()
                .map(|(i, c)| {
                    let current_idx = c & 0x0F;
                    let next_idx = c >> 4;

                    self.distance_table[2 * i * num_centroids + current_idx]
                        + self.distance_table[(2 * i + 1) * num_centroids + next_idx]
                })
                .sum()
        } else {
            pq_code
                .enumerate()
                .map(|(i, c)| self.distance_table[i * num_centroids + c])
                .sum()
        };

        if self.distance_type == DistanceType::Dot {
            dist - diff
        } else {
            dist
        }
    }

    fn distance_all(&self, k_hint: usize) -> Vec<f32> {
        match self.distance_type {
            DistanceType::L2 => compute_pq_distance(
                &self.distance_table,
                self.num_bits,
                self.num_sub_vectors,
                self.pq_code.values(),
                k_hint,
            ),
            DistanceType::Cosine => {
                // it seems we implemented cosine distance at some version,
                // but from now on, we should use normalized L2 distance.
                debug_assert!(
                    false,
                    "cosine distance should be converted to normalized L2 distance"
                );
                // L2 over normalized vectors:  ||x - y|| = x^2 + y^2 - 2 * xy = 1 + 1 - 2 * xy = 2 * (1 - xy)
                // Cosine distance: 1 - |xy| / (||x|| * ||y||) = 1 - xy / (x^2 * y^2) = 1 - xy / (1 * 1) = 1 - xy
                // Therefore, Cosine = L2 / 2
                let l2_dists = compute_pq_distance(
                    &self.distance_table,
                    self.num_bits,
                    self.num_sub_vectors,
                    self.pq_code.values(),
                    k_hint,
                );
                l2_dists.into_iter().map(|v| v / 2.0).collect()
            }
            DistanceType::Dot => {
                let dot_dists = compute_pq_distance(
                    &self.distance_table,
                    self.num_bits,
                    self.num_sub_vectors,
                    self.pq_code.values(),
                    k_hint,
                );
                let diff = self.num_sub_vectors as f32 - 1.0;
                dot_dists.into_iter().map(|v| v - diff).collect()
            }
            _ => unimplemented!("distance type is not supported: {:?}", self.distance_type),
        }
    }
}

fn get_pq_code(
    pq_code: &[u8],
    num_bits: u32,
    num_sub_vectors: usize,
    id: u32,
) -> impl Iterator<Item = u8> + '_ {
    let num_bytes = if num_bits == 4 {
        num_sub_vectors / 2
    } else {
        num_sub_vectors
    };

    let num_vectors = pq_code.len() / num_bytes;
    pq_code
        .iter()
        .skip(id as usize)
        .step_by(num_vectors)
        .copied()
        .exact_size(num_bytes)
}

fn get_centroids<T: Clone>(
    codebook: &[T],
    num_bits: u32,
    num_sub_vectors: usize,
    dimension: usize,
    codes: impl Iterator<Item = u8>,
) -> Vec<T> {
    // codebook[i][j] is the j-th centroid of the i-th sub-vector.
    // the codebook is stored as a flat array, codebook[i * num_centroids + j] = codebook[i][j]

    if num_bits == 4 {
        return get_centroids_4bit(codebook, num_sub_vectors, dimension, codes);
    }

    let num_centroids: usize = 2_usize.pow(8);
    let sub_vector_width = dimension / num_sub_vectors;
    let mut centroids = Vec::with_capacity(dimension);
    for (sub_vec_idx, centroid_idx) in codes.enumerate() {
        let centroid_idx = centroid_idx as usize;
        let centroid = &codebook[sub_vec_idx * num_centroids * sub_vector_width
            + centroid_idx * sub_vector_width
            ..sub_vec_idx * num_centroids * sub_vector_width
                + (centroid_idx + 1) * sub_vector_width];
        centroids.extend_from_slice(centroid);
    }
    centroids
}

fn get_centroids_4bit<T: Clone>(
    codebook: &[T],
    num_sub_vectors: usize,
    dimension: usize,
    codes: impl Iterator<Item = u8>,
) -> Vec<T> {
    let num_centroids: usize = 16;
    let sub_vector_width = dimension / num_sub_vectors;
    let mut centroids = Vec::with_capacity(dimension);
    for (sub_vec_idx, centroid_idx) in codes.into_iter().enumerate() {
        let current_idx = (centroid_idx & 0x0F) as usize;
        let offset = 2 * sub_vec_idx * num_centroids * sub_vector_width;
        let current_centroid = &codebook[offset + current_idx * sub_vector_width
            ..offset + (current_idx + 1) * sub_vector_width];
        centroids.extend_from_slice(current_centroid);

        let next_idx = (centroid_idx >> 4) as usize;
        let offset = (2 * sub_vec_idx + 1) * num_centroids * sub_vector_width;
        let next_centroid = &codebook
            [offset + next_idx * sub_vector_width..offset + (next_idx + 1) * sub_vector_width];
        centroids.extend_from_slice(next_centroid);
    }
    centroids
}

#[cfg(test)]
mod tests {
    use crate::vector::storage::StorageBuilder;

    use super::*;

    use arrow_array::{Float32Array, UInt32Array};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::ROW_ID_FIELD;
    use rand::Rng;

    const DIM: usize = 32;
    const TOTAL: usize = 512;
    const NUM_SUB_VECTORS: usize = 16;

    async fn create_pq_storage() -> ProductQuantizationStorage {
        let codebook = Float32Array::from_iter_values((0..256 * DIM).map(|_| rand::random()));
        let codebook = FixedSizeListArray::try_new_from_values(codebook, DIM as i32).unwrap();
        let pq = ProductQuantizer::new(NUM_SUB_VECTORS, 8, DIM, codebook, DistanceType::Dot);

        let schema = ArrowSchema::new(vec![
            Field::new(
                "vec",
                DataType::FixedSizeList(
                    Field::new_list_field(DataType::Float32, true).into(),
                    DIM as i32,
                ),
                true,
            ),
            ROW_ID_FIELD.clone(),
        ]);
        let vectors = Float32Array::from_iter_values((0..TOTAL * DIM).map(|_| rand::random()));
        let row_ids = UInt64Array::from_iter_values((0..TOTAL).map(|v| v as u64));
        let fsl = FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap();
        let batch =
            RecordBatch::try_new(schema.into(), vec![Arc::new(fsl), Arc::new(row_ids)]).unwrap();

        StorageBuilder::new("vec".to_owned(), pq.distance_type, pq, None)
            .unwrap()
            .build(vec![batch])
            .unwrap()
    }

    async fn create_pq_storage_with_extra_column() -> ProductQuantizationStorage {
        let codebook = Float32Array::from_iter_values((0..256 * DIM).map(|_| rand::random()));
        let codebook = FixedSizeListArray::try_new_from_values(codebook, DIM as i32).unwrap();
        let pq = ProductQuantizer::new(NUM_SUB_VECTORS, 8, DIM, codebook, DistanceType::Dot);

        let schema = ArrowSchema::new(vec![
            Field::new(
                "vec",
                DataType::FixedSizeList(
                    Field::new_list_field(DataType::Float32, true).into(),
                    DIM as i32,
                ),
                true,
            ),
            ROW_ID_FIELD.clone(),
            Field::new("extra", DataType::UInt32, true),
        ]);
        let vectors = Float32Array::from_iter_values((0..TOTAL * DIM).map(|_| rand::random()));
        let row_ids = UInt64Array::from_iter_values((0..TOTAL).map(|v| v as u64));
        let extra_column = UInt32Array::from_iter_values((0..TOTAL).map(|v| v as u32));
        let fsl = FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap();
        let batch = RecordBatch::try_new(
            schema.into(),
            vec![Arc::new(fsl), Arc::new(row_ids), Arc::new(extra_column)],
        )
        .unwrap();

        StorageBuilder::new("vec".to_owned(), pq.distance_type, pq, None)
            .unwrap()
            .assert_num_columns(false)
            .build(vec![batch])
            .unwrap()
    }

    #[tokio::test]
    async fn test_build_pq_storage() {
        let storage = create_pq_storage().await;
        assert_eq!(storage.len(), TOTAL);
        assert_eq!(storage.metadata.num_sub_vectors, NUM_SUB_VECTORS);
        assert_eq!(
            storage.metadata.codebook.as_ref().unwrap().values().len(),
            256 * DIM
        );
        assert_eq!(storage.pq_code.len(), TOTAL * NUM_SUB_VECTORS);
        assert_eq!(storage.row_ids.len(), TOTAL);
    }

    #[tokio::test]
    async fn test_distance_all() {
        let storage = create_pq_storage().await;
        let query = Arc::new(Float32Array::from_iter_values((0..DIM).map(|v| v as f32)));
        let dist_calc = storage.dist_calculator(query);
        let expected = (0..storage.len())
            .map(|id| dist_calc.distance(id as u32))
            .collect::<Vec<_>>();
        let distances = dist_calc.distance_all(100);
        assert_eq!(distances, expected);
    }

    #[tokio::test]
    async fn test_dist_between() {
        let mut rng = rand::thread_rng();
        let storage = create_pq_storage().await;
        let u = rng.gen_range(0..storage.len() as u32);
        let v = rng.gen_range(0..storage.len() as u32);
        let dist1 = storage.dist_between(u, v);
        let dist2 = storage.dist_between(v, u);
        assert_eq!(dist1, dist2);
    }

    #[tokio::test]
    async fn test_remap_with_extra_column() {
        let storage = create_pq_storage_with_extra_column().await;
        let mut mapping = HashMap::new();
        for i in 0..TOTAL / 2 {
            mapping.insert(i as u64, Some((TOTAL + i) as u64));
        }
        for i in TOTAL / 2..TOTAL {
            mapping.insert(i as u64, None);
        }
        let new_storage = storage.remap(&mapping).unwrap();
        assert_eq!(new_storage.len(), TOTAL / 2);
        assert_eq!(new_storage.row_ids.len(), TOTAL / 2);
        for (i, row_id) in new_storage.row_ids().enumerate() {
            assert_eq!(*row_id, (TOTAL + i) as u64);
        }
        assert_eq!(new_storage.batch.num_columns(), 2);
        assert!(new_storage.batch.column_by_name(ROW_ID).is_some());
        assert!(new_storage.batch.column_by_name(PQ_CODE_COLUMN).is_some());
    }
}
