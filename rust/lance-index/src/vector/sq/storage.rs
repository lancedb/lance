// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::{BTreeMap, HashMap},
    ops::{Bound::Included, Range},
    sync::Arc,
};

use arrow_array::{
    cast::AsArray,
    types::{Float32Type, UInt64Type, UInt8Type},
    ArrayRef, RecordBatch, UInt64Array, UInt8Array,
};
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use lance_core::{Error, Result, ROW_ID};
use lance_file::reader::FileReader;
use lance_io::object_store::ObjectStore;
use lance_linalg::distance::{l2_distance_uint_scalar, DistanceType};
use lance_table::format::SelfDescribingFileReader;
use object_store::path::Path;
use serde::{Deserialize, Serialize};
use snafu::{location, Location};

use crate::{
    vector::{
        quantizer::{QuantizerMetadata, QuantizerStorage},
        transform::Transformer,
        v3::storage::{DistCalculator, VectorStore},
        SQ_CODE_COLUMN,
    },
    IndexMetadata, INDEX_METADATA_SCHEMA_KEY,
};

use super::{scale_to_u8, ScalarQuantizer};

pub const SQ_METADATA_KEY: &str = "lance:sq";

#[derive(Clone, Serialize, Deserialize)]
pub struct ScalarQuantizationMetadata {
    pub dim: usize,
    pub num_bits: u16,
    pub bounds: Range<f64>,
}

impl DeepSizeOf for ScalarQuantizationMetadata {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        0
    }
}

#[async_trait]
impl QuantizerMetadata for ScalarQuantizationMetadata {
    async fn load(reader: &FileReader) -> Result<Self> {
        let metadata_str = reader
            .schema()
            .metadata
            .get(SQ_METADATA_KEY)
            .ok_or(Error::Index {
                message: format!(
                    "Reading SQ metadata: metadata key {} not found",
                    SQ_METADATA_KEY
                ),
                location: location!(),
            })?;
        serde_json::from_str(metadata_str).map_err(|_| Error::Index {
            message: format!("Failed to parse index metadata: {}", metadata_str),
            location: location!(),
        })
    }
}

/// An immutable chunk of SclarQuantizationStorage.
#[derive(Clone)]
struct SQStorageChunk {
    batch: RecordBatch,

    dim: usize,

    // Helper fields, references to the batch
    // These fields share the `Arc` pointer to the columns in batch,
    // so it does not take more memory.
    row_ids: UInt64Array,
    sq_codes: UInt8Array,
}

impl SQStorageChunk {
    // Create a new chunk from a RecordBatch.
    fn new(batch: RecordBatch) -> Result<Self> {
        let row_ids = batch
            .column_by_name(ROW_ID)
            .ok_or(Error::Index {
                message: "Row ID column not found in the batch".to_owned(),
                location: location!(),
            })?
            .as_primitive::<UInt64Type>()
            .clone();
        let fsl = batch
            .column_by_name(SQ_CODE_COLUMN)
            .ok_or(Error::Index {
                message: "SQ code column not found in the batch".to_owned(),
                location: location!(),
            })?
            .as_fixed_size_list();
        let dim = fsl.value_length() as usize;
        let sq_codes = fsl
            .values()
            .as_primitive_opt::<UInt8Type>()
            .ok_or(Error::Index {
                message: "SQ code column is not FixedSizeList<u8>".to_owned(),
                location: location!(),
            })?
            .clone();
        Ok(Self {
            batch,
            dim,
            row_ids,
            sq_codes,
        })
    }

    /// Returns vector dimension
    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.row_ids.len()
    }

    fn schema(&self) -> &SchemaRef {
        self.batch.schema_ref()
    }

    fn row_id(&self, id: u32) -> u64 {
        self.row_ids.value(id as usize)
    }

    /// Get a slice of SQ code for id
    fn sq_code_slice(&self, id: u32) -> &[u8] {
        assert!(id < self.len() as u32);
        &self.sq_codes.values()[id as usize * self.dim..(id + 1) as usize * self.dim]
    }
}

impl DeepSizeOf for SQStorageChunk {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.batch.get_array_memory_size()
    }
}

#[derive(Clone)]
pub struct ScalarQuantizationStorage {
    quantizer: ScalarQuantizer,

    distance_type: DistanceType,

    /// Chunks of storage
    /// A ordered map of `<start_id, chunk>`
    chunks: BTreeMap<u32, SQStorageChunk>,
}

impl DeepSizeOf for ScalarQuantizationStorage {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.chunks
            .iter()
            .map(|c| c.deep_size_of_children(context))
            .sum()
    }
}

impl ScalarQuantizationStorage {
    pub fn new(
        num_bits: u16,
        distance_type: DistanceType,
        bounds: Range<f64>,
        batch: RecordBatch,
    ) -> Result<Self> {
        let chunk = SQStorageChunk::new(batch)?;

        let quantizer = ScalarQuantizer::with_bounds(num_bits, chunk.dim(), bounds);
        Ok(Self {
            quantizer,
            distance_type,
            chunks: [(0, chunk)].into_iter().collect(),
        })
    }

    /// Get the chunk that covers the id.
    ///
    /// Returns:
    /// `(offset, chunk)`
    ///
    /// We did not check out of range in this call. But the out of range will
    /// panic once you access the data in the last [SQStorageChunk].
    #[inline]
    fn chunk(&self, id: u32) -> (&u32, &SQStorageChunk) {
        self.chunks
            .range((Included(&0), Included(&id)))
            .next_back()
            .unwrap()
    }

    pub async fn load(object_store: &ObjectStore, path: &Path) -> Result<Self> {
        let reader = FileReader::try_new_self_described(object_store, path, None).await?;
        let schema = reader.schema();

        let metadata_str = schema
            .metadata
            .get(INDEX_METADATA_SCHEMA_KEY)
            .ok_or(Error::Index {
                message: format!(
                    "Reading SQ storage: index key {} not found",
                    INDEX_METADATA_SCHEMA_KEY
                ),
                location: location!(),
            })?;
        let index_metadata: IndexMetadata =
            serde_json::from_str(metadata_str).map_err(|_| Error::Index {
                message: format!("Failed to parse index metadata: {}", metadata_str),
                location: location!(),
            })?;
        let distance_type = DistanceType::try_from(index_metadata.distance_type.as_str())?;
        let metadata = ScalarQuantizationMetadata::load(&reader).await?;

        Self::load_partition(&reader, 0..reader.len(), distance_type, &metadata).await
    }
}

#[async_trait]
impl QuantizerStorage for ScalarQuantizationStorage {
    type Metadata = ScalarQuantizationMetadata;
    /// Load a partition of SQ storage from disk.
    ///
    /// Parameters
    /// ----------
    /// - *reader: file reader
    /// - *range: row range of the partition
    /// - *metric_type: metric type of the vectors
    /// - *metadata: scalar quantization metadata
    async fn load_partition(
        reader: &FileReader,
        range: std::ops::Range<usize>,
        distance_type: DistanceType,
        metadata: &Self::Metadata,
    ) -> Result<Self> {
        let schema = reader.schema();
        let batch = reader.read_range(range, schema).await?;

        Self::new(
            metadata.num_bits,
            distance_type,
            metadata.bounds.clone(),
            batch,
        )
    }
}

impl VectorStore for ScalarQuantizationStorage {
    type DistanceCalculator<'a> = SQDistCalculator<'a>;

    fn try_from_batch(
        batch: RecordBatch,
        distance_type: lance_linalg::distance::DistanceType,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        let metadata_json = batch
            .schema_ref()
            .metadata()
            .get("metadata")
            .ok_or(Error::Schema {
                message: "metadata not found".to_string(),
                location: location!(),
            })?;
        let metadata: ScalarQuantizationMetadata = serde_json::from_str(metadata_json)?;

        Self::new(metadata.num_bits, distance_type, metadata.bounds, batch)
    }

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch>> {
        let metadata = ScalarQuantizationMetadata {
            dim: self.chunks.first_key_value().map(|(_, c)| c.dim()).unwrap(),
            num_bits: self.quantizer.num_bits,
            bounds: self.quantizer.bounds.clone(),
        };
        let metadata_json = serde_json::to_string(&metadata)?;
        let metadata = HashMap::from_iter(vec![("metadata".to_owned(), metadata_json)]);

        let schema = self
            .chunks
            .first_key_value()
            .map(|(_, c)| c.schema().as_ref().clone())
            .unwrap()
            .with_metadata(metadata);
        let schema = Arc::new(schema);
        Ok(self.chunks.values().map(move |chunk| {
            chunk
                .batch
                .clone()
                .with_schema(schema.clone())
                .expect("attach schema")
        }))
    }

    fn append_batch(&self, batch: RecordBatch, vector_column: &str) -> Result<Self> {
        // TODO: use chunked storage
        let transformer = super::transform::SQTransformer::new(
            self.quantizer.clone(),
            vector_column.to_string(),
            SQ_CODE_COLUMN.to_string(),
        );

        let new_batch = transformer.transform(&batch)?;

        // self.quantizer.transform(data)
        let mut storage = self.clone();
        let offset = self.len() as u32;
        let new_chunk = SQStorageChunk::new(new_batch)?;
        storage.chunks.insert(offset, new_chunk);

        Ok(storage)
    }

    fn schema(&self) -> &SchemaRef {
        self.chunks
            .first_key_value()
            .map(|(_, c)| c.schema())
            .unwrap()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn len(&self) -> usize {
        self.chunks
            .last_key_value()
            .map(|(&offset, batch)| offset as usize + batch.len())
            .unwrap_or_default()
    }

    /// Return the [DistanceType] of the vectors.
    fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    fn row_id(&self, id: u32) -> u64 {
        let (offset, chunk) = self.chunk(id);
        chunk.row_id(id - offset)
    }

    fn row_ids(&self) -> impl Iterator<Item = &u64> {
        self.chunks.values().flat_map(|c| c.row_ids.values())
    }

    /// Create a [DistCalculator] to compute the distance between the query.
    ///
    /// Using dist calcualtor can be more efficient as it can pre-compute some
    /// values.
    fn dist_calculator(&self, query: ArrayRef) -> Self::DistanceCalculator<'_> {
        SQDistCalculator::new(query, self, self.quantizer.bounds.clone())
    }

    fn dist_calculator_from_id(&self, id: u32) -> Self::DistanceCalculator<'_> {
        let (offset, chunk) = self.chunk(id);
        let query_sq_code = chunk.sq_code_slice(id - offset).to_vec();
        SQDistCalculator {
            query_sq_code,
            storage: self,
        }
    }

    fn distance_between(&self, a: u32, b: u32) -> f32 {
        let (offset_a, chunk_a) = self.chunk(a);
        let (offset_b, chunk_b) = self.chunk(b);
        l2_distance_uint_scalar(
            chunk_a.sq_code_slice(a - offset_a),
            chunk_b.sq_code_slice(b - offset_b),
        )
    }
}

pub struct SQDistCalculator<'a> {
    query_sq_code: Vec<u8>,
    storage: &'a ScalarQuantizationStorage,
}

impl<'a> SQDistCalculator<'a> {
    fn new(query: ArrayRef, storage: &'a ScalarQuantizationStorage, bounds: Range<f64>) -> Self {
        let query_sq_code =
            scale_to_u8::<Float32Type>(query.as_primitive::<Float32Type>().values(), bounds);
        Self {
            query_sq_code,
            storage,
        }
    }
}

impl<'a> DistCalculator for SQDistCalculator<'a> {
    fn distance(&self, id: u32) -> f32 {
        let (offset, chunk) = self.storage.chunk(id);
        let sq_code = chunk.sq_code_slice(id - offset);
        l2_distance_uint_scalar(sq_code, &self.query_sq_code)
    }

    #[allow(unused_variables)]
    fn prefetch(&self, id: u32) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            const CACHE_LINE_SIZE: usize = 64;

            let (offset, chunk) = self.storage.chunk(id);
            let dim = chunk.dim();
            let base_ptr = chunk.sq_code_slice(id - offset).as_ptr();

            unsafe {
                // Loop over the sq_code to prefetch each cache line
                for offset in (0..self.dim).step_by(CACHE_LINE_SIZE) {
                    {
                        use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                        _mm_prefetch(base_ptr.add(offset) as *const i8, _MM_HINT_T0);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::FixedSizeListArray;
    use arrow_schema::{DataType, Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;
    use rand::prelude::*;

    #[test]
    fn test_get_chunks() {
        const DIM: usize = 64;

        let mut rng = rand::thread_rng();
        let row_ids = UInt64Array::from_iter_values(0..100);
        let sq_code = UInt8Array::from_iter_values((0..100 * DIM).map(|_| rng.gen::<u8>()));
        let fsl = FixedSizeListArray::try_new_from_values(sq_code, DIM as i32).unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new(
                SQ_CODE_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::UInt8, true)),
                    DIM as i32,
                ),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(row_ids), Arc::new(fsl)]).unwrap();

        let storage =
            ScalarQuantizationStorage::new(8, DistanceType::L2, -0.7..0.7, batch).unwrap();

        assert_eq!(storage.len(), 100);

        let (offset, chunk) = storage.chunk(0);
        assert_eq!(*offset, 0);
        assert_eq!(chunk.row_id(20), 20);

        let (offset, _) = storage.chunk(50);
        assert_eq!(*offset, 0);
    }
}
