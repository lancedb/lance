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
use deepsize::DeepSizeOf;
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_core::{Error, Result, ROW_ID};
use lance_file::{reader::FileReader, writer::FileWriter};
use lance_io::{
    object_store::ObjectStore,
    traits::{WriteExt, Writer},
    utils::read_message,
};
use lance_linalg::distance::{DistanceType, Dot, L2};
use lance_table::{format::SelfDescribingFileReader, io::manifest::ManifestDescribing};
use object_store::path::Path;
use prost::Message;
use serde::{Deserialize, Serialize};
use snafu::{location, Location};

use super::distance::{build_distance_table_dot, compute_l2_distance};
use super::distance::{build_distance_table_l2, compute_dot_distance};
use super::ProductQuantizer;
use crate::vector::storage::STORAGE_METADATA_KEY;
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
    pub num_bits: u32,
    pub num_sub_vectors: usize,
    pub dimension: usize,

    #[serde(skip)]
    pub codebook: Option<FixedSizeListArray>,

    // empty for old format
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

#[async_trait]
impl QuantizerMetadata for ProductQuantizationMetadata {
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
///
/// TODO: support f16/f64 later.
#[derive(Clone, Debug)]
pub struct ProductQuantizationStorage {
    codebook: FixedSizeListArray,
    batch: RecordBatch,

    // Metadata
    num_bits: u32,
    num_sub_vectors: usize,
    dimension: usize,
    distance_type: DistanceType,

    // For easy access
    pq_code: Arc<UInt8Array>,
    row_ids: Arc<UInt64Array>,
}

impl DeepSizeOf for ProductQuantizationStorage {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.codebook.get_array_memory_size()
            + self.batch.get_array_memory_size()
            + self.pq_code.get_array_memory_size()
            + self.row_ids.get_array_memory_size()
    }
}

impl PartialEq for ProductQuantizationStorage {
    fn eq(&self, other: &Self) -> bool {
        self.distance_type.eq(&other.distance_type)
            && self.codebook.eq(&other.codebook)
            && self.num_bits.eq(&other.num_bits)
            && self.num_sub_vectors.eq(&other.num_sub_vectors)
            && self.dimension.eq(&other.dimension)
            // Ignore the schema because they might have different metadata.
            && self.batch.columns().eq(other.batch.columns())
    }
}

impl ProductQuantizationStorage {
    pub fn new(
        codebook: FixedSizeListArray,
        mut batch: RecordBatch,
        num_bits: u32,
        num_sub_vectors: usize,
        dimension: usize,
        distance_type: DistanceType,
        transposed: bool,
    ) -> Result<Self> {
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

        let pq_code = batch[PQ_CODE_COLUMN]
            .as_fixed_size_list()
            .values()
            .as_primitive()
            .clone()
            .into();

        let distance_type = match distance_type {
            DistanceType::Cosine => DistanceType::L2,
            _ => distance_type,
        };
        Ok(Self {
            codebook,
            batch,
            pq_code,
            row_ids,
            num_sub_vectors,
            num_bits,
            dimension,
            distance_type,
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
        )
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
    pub async fn load(object_store: &ObjectStore, path: &Path) -> Result<Self> {
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
        Self::load_partition(&reader, 0..reader.len(), distance_type, &metadata).await
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

    /// Write the PQ storage to disk.
    pub async fn write_full(&self, writer: &mut FileWriter<ManifestDescribing>) -> Result<()> {
        let pos = writer.object_writer.tell().await?;
        let codebook_tensor = pb::Tensor::try_from(&self.codebook)?;
        writer
            .object_writer
            .write_protobuf(&codebook_tensor)
            .await?;

        self.write_partition(writer).await?;

        let metadata = ProductQuantizationMetadata {
            codebook_position: pos,
            num_bits: self.num_bits,
            num_sub_vectors: self.num_sub_vectors,
            dimension: self.dimension,
            codebook: None,
            codebook_tensor: Vec::new(),
            transposed: true,
        };

        let index_metadata = IndexMetadata {
            index_type: "PQ".to_string(),
            distance_type: self.distance_type.to_string(),
        };

        let mut schema_metadata = HashMap::new();
        schema_metadata.insert(
            PQ_METADATA_KEY.to_string(),
            serde_json::to_string(&metadata)?,
        );
        schema_metadata.insert(
            INDEX_METADATA_SCHEMA_KEY.to_string(),
            serde_json::to_string(&index_metadata)?,
        );
        writer.finish_with_metadata(&schema_metadata).await?;
        Ok(())
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
            metadata.num_bits,
            metadata.num_sub_vectors,
            metadata.dimension,
            distance_type,
            metadata.transposed,
        )
    }
}

impl VectorStore for ProductQuantizationStorage {
    type DistanceCalculator<'a> = PQDistCalculator;

    fn try_from_batch(batch: RecordBatch, distance_type: DistanceType) -> Result<Self>
    where
        Self: Sized,
    {
        let metadata_json = batch
            .schema_ref()
            .metadata()
            .get(STORAGE_METADATA_KEY)
            .ok_or(Error::Index {
                message: "Metadata not found in schema".to_string(),
                location: location!(),
            })?;
        let metadata: ProductQuantizationMetadata = serde_json::from_str(metadata_json)?;

        // now it supports only Float32Type
        let codebook_tensor = pb::Tensor::decode(metadata.codebook_tensor.as_slice())?;
        let codebook = FixedSizeListArray::try_from(&codebook_tensor)?;

        Self::new(
            codebook,
            batch,
            metadata.num_bits,
            metadata.num_sub_vectors,
            metadata.dimension,
            distance_type,
            metadata.transposed,
        )
    }

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch>> {
        let codebook = pb::Tensor::try_from(&self.codebook)?.encode_to_vec();
        let metadata = ProductQuantizationMetadata {
            codebook_position: 0, // deprecated in new format
            num_bits: self.num_bits,
            num_sub_vectors: self.num_sub_vectors,
            dimension: self.dimension,
            codebook: None,
            codebook_tensor: codebook,
            transposed: true, // we always transpose the pq codes for efficiency
        };

        let metadata_json = serde_json::to_string(&metadata)?;
        let metadata = HashMap::from_iter(vec![(STORAGE_METADATA_KEY.to_string(), metadata_json)]);
        Ok([self.batch.with_metadata(metadata)?].into_iter())
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
        match self.codebook.value_type() {
            DataType::Float16 => PQDistCalculator::new(
                self.codebook
                    .values()
                    .as_primitive::<datatypes::Float16Type>()
                    .values(),
                self.num_bits,
                self.num_sub_vectors,
                self.pq_code.clone(),
                query.as_primitive::<datatypes::Float16Type>().values(),
                self.distance_type,
            ),
            DataType::Float32 => PQDistCalculator::new(
                self.codebook
                    .values()
                    .as_primitive::<datatypes::Float32Type>()
                    .values(),
                self.num_bits,
                self.num_sub_vectors,
                self.pq_code.clone(),
                query.as_primitive::<datatypes::Float32Type>().values(),
                self.distance_type,
            ),
            DataType::Float64 => PQDistCalculator::new(
                self.codebook
                    .values()
                    .as_primitive::<datatypes::Float64Type>()
                    .values(),
                self.num_bits,
                self.num_sub_vectors,
                self.pq_code.clone(),
                query.as_primitive::<datatypes::Float64Type>().values(),
                self.distance_type,
            ),
            _ => unimplemented!("Unsupported data type: {:?}", self.codebook.value_type()),
        }
    }

    fn dist_calculator_from_id(&self, _: u32) -> Self::DistanceCalculator<'_> {
        todo!("distance_between not implemented for PQ storage")
    }

    fn distance_between(&self, _: u32, _: u32) -> f32 {
        todo!("distance_between not implemented for PQ storage")
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

    fn get_pq_code(&self, id: u32) -> Vec<usize> {
        let num_sub_vectors_in_byte = if self.num_bits == 4 {
            self.num_sub_vectors / 2
        } else {
            self.num_sub_vectors
        };
        let num_vectors = self.pq_code.len() / num_sub_vectors_in_byte;
        self.pq_code
            .values()
            .iter()
            .skip(id as usize)
            .step_by(num_vectors)
            .map(|&c| c as usize)
            .collect()
    }
}

impl DistCalculator for PQDistCalculator {
    fn distance(&self, id: u32) -> f32 {
        let num_centroids = 2_usize.pow(self.num_bits);
        let pq_code = self.get_pq_code(id);

        if self.num_bits == 4 {
            pq_code
                .into_iter()
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
                .into_iter()
                .enumerate()
                .map(|(i, c)| self.distance_table[i * num_centroids + c])
                .sum()
        }
    }

    fn distance_all(&self) -> Vec<f32> {
        match self.distance_type {
            DistanceType::L2 => compute_l2_distance(
                &self.distance_table,
                self.num_bits,
                self.num_sub_vectors,
                self.pq_code.values(),
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
                let l2_dists = compute_l2_distance(
                    &self.distance_table,
                    self.num_bits,
                    self.num_sub_vectors,
                    self.pq_code.values(),
                );
                l2_dists.into_iter().map(|v| v / 2.0).collect()
            }
            DistanceType::Dot => compute_dot_distance(
                &self.distance_table,
                self.num_bits,
                self.num_sub_vectors,
                self.pq_code.values(),
            ),
            _ => unimplemented!("distance type is not supported: {:?}", self.distance_type),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::vector::storage::StorageBuilder;

    use super::*;

    use arrow_array::Float32Array;
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::datatypes::Schema;
    use lance_core::ROW_ID_FIELD;

    const DIM: usize = 32;
    const TOTAL: usize = 512;
    const NUM_SUB_VECTORS: usize = 16;

    async fn create_pq_storage() -> ProductQuantizationStorage {
        let codebook = Float32Array::from_iter_values((0..256 * DIM).map(|v| v as f32));
        let codebook = FixedSizeListArray::try_new_from_values(codebook, DIM as i32).unwrap();
        let pq = ProductQuantizer::new(NUM_SUB_VECTORS, 8, DIM, codebook, DistanceType::L2);

        let schema = ArrowSchema::new(vec![
            Field::new(
                "vectors",
                DataType::FixedSizeList(
                    Field::new_list_field(DataType::Float32, true).into(),
                    DIM as i32,
                ),
                true,
            ),
            ROW_ID_FIELD.clone(),
        ]);
        let vectors = Float32Array::from_iter_values((0..TOTAL * DIM).map(|v| v as f32));
        let row_ids = UInt64Array::from_iter_values((0..TOTAL).map(|v| v as u64));
        let fsl = FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap();
        let batch =
            RecordBatch::try_new(schema.into(), vec![Arc::new(fsl), Arc::new(row_ids)]).unwrap();

        StorageBuilder::new("vectors".to_owned(), pq.distance_type, pq)
            .build(&batch)
            .unwrap()
    }

    #[tokio::test]
    async fn test_build_pq_storage() {
        let storage = create_pq_storage().await;
        assert_eq!(storage.len(), TOTAL);
        assert_eq!(storage.num_sub_vectors, NUM_SUB_VECTORS);
        assert_eq!(storage.codebook.values().len(), 256 * DIM);
        assert_eq!(storage.pq_code.len(), TOTAL * NUM_SUB_VECTORS);
        assert_eq!(storage.row_ids.len(), TOTAL);
    }

    #[tokio::test]
    async fn test_read_write_pq_storage() {
        let storage = create_pq_storage().await;

        let store = ObjectStore::memory();
        let path = Path::from("pq_storage");
        let schema = Schema::try_from(storage.schema().as_ref()).unwrap();
        let mut file_writer = FileWriter::<ManifestDescribing>::try_new(
            &store,
            &path,
            schema.clone(),
            &Default::default(),
        )
        .await
        .unwrap();

        storage.write_full(&mut file_writer).await.unwrap();

        let storage2 = ProductQuantizationStorage::load(&store, &path)
            .await
            .unwrap();

        assert_eq!(storage, storage2);
    }

    #[tokio::test]
    async fn test_distance_all() {
        let storage = create_pq_storage().await;
        let query = Arc::new(Float32Array::from_iter_values((0..DIM).map(|v| v as f32)));
        let dist_calc = storage.dist_calculator(query);
        let expected = (0..storage.len())
            .map(|id| dist_calc.distance(id as u32))
            .collect::<Vec<_>>();
        let distances = dist_calc.distance_all();
        assert_eq!(distances, expected);
    }
}
