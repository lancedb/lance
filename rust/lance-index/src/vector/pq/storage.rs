// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Product Quantization storage
//!
//! Used as storage backend for Graph based algorithms.

use std::{cmp::min, collections::HashMap, sync::Arc};

use arrow_array::{
    cast::AsArray,
    types::{Float32Type, UInt64Type, UInt8Type},
    FixedSizeListArray, Float32Array, RecordBatch, UInt64Array, UInt8Array,
};
use arrow_schema::SchemaRef;
use lance_core::{Error, Result, ROW_ID};
use lance_file::{reader::FileReader, writer::FileWriter};
use lance_io::{
    object_store::ObjectStore,
    traits::{WriteExt, Writer},
    utils::read_message,
};
use lance_linalg::{distance::MetricType, MatrixView};
use lance_table::{format::SelfDescribingFileReader, io::manifest::ManifestDescribing};
use object_store::path::Path;
use serde::{Deserialize, Serialize};
use snafu::{location, Location};

use super::{distance::build_distance_table_l2, num_centroids, ProductQuantizerImpl};
use crate::{
    pb,
    vector::{
        graph::storage::{DistCalculator, VectorStorage},
        pq::transform::PQTransformer,
        transform::Transformer,
        PQ_CODE_COLUMN,
    },
    IndexMetadata, INDEX_METADATA_SCHEMA_KEY,
};

const PQ_METADTA_KEY: &str = "lance:pq";

#[derive(Serialize, Deserialize)]
struct ProductQuantizationMetadata {
    codebook_position: usize,
    num_bits: u32,
    num_sub_vectors: usize,
    dimension: usize,
}

/// Product Quantization Storage
///
/// It stores PQ code, as well as the row ID to the orignal vectors.
///
/// It is possible to store additonal metadata to accelerate filtering later.
///
/// TODO: support f16/f64 later.
#[derive(Clone, Debug)]
pub struct ProductQuantizationStorage {
    codebook: Arc<Float32Array>,
    batch: RecordBatch,

    // Metadata
    num_bits: u32,
    num_sub_vectors: usize,
    dimension: usize,
    metric_type: MetricType,

    // For easy access
    pq_code: Arc<UInt8Array>,
    row_ids: Arc<UInt64Array>,
}

impl PartialEq for ProductQuantizationStorage {
    fn eq(&self, other: &Self) -> bool {
        self.metric_type.eq(&other.metric_type)
            &&        self.codebook.eq(&other.codebook)
            && self.num_bits.eq(&other.num_bits)
            && self.num_sub_vectors.eq(&other.num_sub_vectors)
            && self.dimension.eq(&other.dimension)
            // Ignore the schema because they might have different metadata.
            && self.batch.columns().eq(other.batch.columns())
    }
}

#[allow(dead_code)]
impl ProductQuantizationStorage {
    pub fn new(
        codebook: Arc<Float32Array>,
        batch: RecordBatch,
        num_bits: u32,
        num_sub_vectors: usize,
        dimension: usize,
        metric_type: MetricType,
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

        let Some(pq_col) = batch.column_by_name(PQ_CODE_COLUMN) else {
            return Err(Error::Index {
                message: format!("{PQ_CODE_COLUMN} column not found from PQ storage"),
                location: location!(),
            });
        };
        let pq_code_fsl = pq_col.as_fixed_size_list_opt().ok_or(Error::Index {
            message: format!(
                "{PQ_CODE_COLUMN} column is not of type UInt8: {}",
                pq_col.data_type()
            ),
            location: location!(),
        })?;
        let pq_code: Arc<UInt8Array> = pq_code_fsl
            .values()
            .as_primitive_opt::<UInt8Type>()
            .ok_or(Error::Index {
                message: format!(
                    "{PQ_CODE_COLUMN} column is not of type UInt8: {}",
                    pq_col.data_type()
                ),
                location: location!(),
            })?
            .clone()
            .into();

        Ok(Self {
            codebook,
            batch,
            pq_code,
            row_ids,
            num_sub_vectors,
            num_bits,
            dimension,
            metric_type,
        })
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
        quantizer: Arc<ProductQuantizerImpl<Float32Type>>,
        batch: &RecordBatch,
        vector_col: &str,
    ) -> Result<Self> {
        let codebook = quantizer.codebook.clone();
        let num_bits = quantizer.num_bits;
        let dimension = quantizer.dimension;
        let num_sub_vectors = quantizer.num_sub_vectors;
        let metric_type = quantizer.metric_type;
        let transform = PQTransformer::new(quantizer, vector_col, PQ_CODE_COLUMN);
        let batch = transform.transform(batch).await?;

        Self::new(
            codebook,
            batch,
            num_bits,
            num_sub_vectors,
            dimension,
            metric_type,
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
        let metric_type: MetricType = MetricType::try_from(index_metadata.metric_type.as_str())?;

        let metadata = schema.metadata.get(PQ_METADTA_KEY).ok_or(Error::Index {
            message: format!(
                "Reading PQ storage: metadata key {} not found",
                PQ_METADTA_KEY
            ),
            location: location!(),
        })?;
        let pq_matadata: ProductQuantizationMetadata =
            serde_json::from_str(metadata).map_err(|_| Error::Index {
                message: format!("Failed to parse PQ metadata: {}", metadata),
                location: location!(),
            })?;
        let codebook_tensor: pb::Tensor =
            read_message(reader.object_reader.as_ref(), pq_matadata.codebook_position).await?;
        let fsl = FixedSizeListArray::try_from(&codebook_tensor)?;

        // Hard coded to float32 for now
        let codebook = Arc::new(fsl.values().as_primitive::<Float32Type>().clone());

        let total = reader.len();
        let batch = reader.read_range(0..total, schema, None).await?;

        Self::new(
            codebook,
            batch,
            pq_matadata.num_bits,
            pq_matadata.num_sub_vectors,
            pq_matadata.dimension,
            metric_type,
        )
    }

    pub fn schema(&self) -> SchemaRef {
        self.batch.schema()
    }

    #[allow(dead_code)]
    pub fn get_row_ids(&self, ids: &[u32]) -> Vec<u64> {
        ids.iter()
            .map(|&id| self.row_ids.value(id as usize))
            .collect()
    }

    /// Write the PQ storage to disk.
    pub async fn write(&self, writer: &mut FileWriter<ManifestDescribing>) -> Result<()> {
        let pos = writer.object_writer.tell().await?;
        let mat = MatrixView::<Float32Type>::new(self.codebook.clone(), self.dimension);
        let codebook_tensor = pb::Tensor::from(&mat);
        writer
            .object_writer
            .write_protobuf(&codebook_tensor)
            .await?;

        let metadata = ProductQuantizationMetadata {
            codebook_position: pos,
            num_bits: self.num_bits,
            num_sub_vectors: self.num_sub_vectors,
            dimension: self.dimension,
        };

        let batch_size: usize = 10240; // TODO: make it configurable
        for i in (0..self.batch.num_rows()).step_by(batch_size) {
            let offset = i * batch_size;
            let length = min(batch_size, self.batch.num_rows() - offset);
            let slice = self.batch.slice(offset, length);
            writer.write(&[slice]).await?;
        }

        let index_metadata = IndexMetadata {
            index_type: "PQ".to_string(),
            metric_type: self.metric_type.to_string(),
        };

        let mut schema_metadata = HashMap::new();
        schema_metadata.insert(
            PQ_METADTA_KEY.to_string(),
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

impl VectorStorage for ProductQuantizationStorage {
    fn len(&self) -> usize {
        self.batch.num_rows()
    }

    fn metric_type(&self) -> MetricType {
        self.metric_type
    }

    fn dist_calculator(&self, query: &[f32]) -> Box<dyn DistCalculator> {
        Box::new(PQDistCalculator::new(
            self.codebook.values(),
            self.num_bits,
            self.num_sub_vectors,
            self.pq_code.clone(),
            query,
            MetricType::L2,
        ))
    }
}

/// Distance calculator backed by PQ code.
struct PQDistCalculator {
    distance_table: Vec<f32>,
    pq_code: Arc<UInt8Array>,
    num_sub_vectors: usize,
    num_centroids: usize,
}

impl PQDistCalculator {
    fn new(
        codebook: &[f32],
        num_bits: u32,
        num_sub_vectors: usize,
        pq_code: Arc<UInt8Array>,
        query: &[f32],
        metric_type: MetricType,
    ) -> Self {
        let distance_table = if matches!(metric_type, MetricType::Cosine | MetricType::L2) {
            build_distance_table_l2(codebook, num_bits, num_sub_vectors, query)
        } else {
            unimplemented!("Metric type not supported: {:?}", metric_type);
        };
        Self {
            distance_table,
            num_sub_vectors,
            pq_code,
            num_centroids: num_centroids(num_bits),
        }
    }

    fn get_pq_code(&self, id: u32) -> &[u8] {
        let start = id as usize * self.num_sub_vectors;
        let end = start + self.num_sub_vectors;
        &self.pq_code.values()[start..end]
    }
}

impl DistCalculator for PQDistCalculator {
    fn distance(&self, ids: &[u32]) -> Vec<f32> {
        ids.iter()
            .map(|&id| {
                let pq_code = self.get_pq_code(id);
                pq_code
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| self.distance_table[i * self.num_centroids + c as usize])
                    .sum()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::FixedSizeListArray;
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::{datatypes::Schema, ROW_ID_FIELD};

    const DIM: usize = 32;
    const TOTAL: usize = 512;
    const NUM_SUB_VECTORS: usize = 16;

    async fn create_pq_storage() -> ProductQuantizationStorage {
        let codebook = Arc::new(Float32Array::from_iter_values(
            (0..256 * DIM).map(|v| v as f32),
        ));
        let pq = Arc::new(ProductQuantizerImpl::<Float32Type>::new(
            NUM_SUB_VECTORS,
            8,
            DIM,
            codebook,
            MetricType::L2,
        ));

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

        ProductQuantizationStorage::build(pq.clone(), &batch, "vectors")
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_build_pq_storage() {
        let storage = create_pq_storage().await;
        assert_eq!(storage.len(), TOTAL);
        assert_eq!(storage.num_sub_vectors, NUM_SUB_VECTORS);
        assert_eq!(storage.codebook.len(), 256 * DIM);
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

        storage.write(&mut file_writer).await.unwrap();

        let storage2 = ProductQuantizationStorage::load(&store, &path)
            .await
            .unwrap();

        assert_eq!(storage, storage2);
    }
}
