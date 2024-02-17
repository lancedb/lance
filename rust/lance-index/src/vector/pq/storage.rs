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

use std::sync::Arc;

use arrow_array::{
    cast::AsArray,
    types::{Float32Type, UInt64Type, UInt8Type},
    Float32Array, RecordBatch, UInt64Array, UInt8Array,
};
use arrow_schema::SchemaRef;
use lance_core::{Error, Result, ROW_ID};
use lance_linalg::distance::MetricType;
use snafu::{location, Location};

use super::{build_distance_table_l2, ProductQuantizerImpl};
use crate::vector::{
    graph::storage::{DistCalculator, VectorStorage},
    pq::transform::PQTransformer,
    transform::Transformer,
    PQ_CODE_COLUMN,
};

/// Product Quantization Storage
///
/// It stores PQ code, as well as the row ID to the orignal vectors.
///
/// It is possible to store additonal metadata to accelerate filtering later.
///
#[derive(Clone)]
pub struct ProductQuantizationStorage {
    codebook: Arc<Float32Array>,
    batch: RecordBatch,

    // Metadata
    num_bits: u32,
    num_sub_vectors: usize,
    dimension: usize,

    // For easy access
    pq_code: Arc<UInt8Array>,
    row_ids: Arc<UInt64Array>,
}

impl ProductQuantizationStorage {
    pub async fn new(
        quantizer: Arc<ProductQuantizerImpl<Float32Type>>,
        batch: &RecordBatch,
        vector_col: &str,
    ) -> Result<Self> {
        let codebook = quantizer.codebook.clone();
        let num_bits = quantizer.num_bits;
        let dimension = quantizer.dimension;
        let num_sub_vectors = quantizer.num_sub_vectors;
        let transform = PQTransformer::new(quantizer, vector_col, PQ_CODE_COLUMN);
        let batch = transform.transform(batch).await?;

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
        })
    }

    pub fn schema(&self) -> SchemaRef {
        self.batch.schema()
    }
}

struct PQDistCalculator {
    distance_table: Vec<f32>,
}

impl PQDistCalculator {
    fn new(
        codebook: &[f32],
        num_bits: u32,
        num_sub_vectors: usize,
        dimension: usize,
        query: &[f32],
    ) -> Self {
        let distance_table = build_distance_table_l2::<Float32Type>(
            codebook,
            dimension,
            num_bits,
            num_sub_vectors,
            query,
        );
        Self { distance_table }
    }
}

impl VectorStorage<f32> for ProductQuantizationStorage {
    fn len(&self) -> usize {
        self.batch.num_rows()
    }

    fn dist_calculator(&self, query: &[f32], metric_type: MetricType) -> Box<dyn DistCalculator> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::FixedSizeListArray;
    use arrow_schema::{DataType, Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::ROW_ID_FIELD;

    #[tokio::test]
    async fn test_build_pq_storage() {
        const DIM: usize = 32;
        const TOTAL: usize = 512;
        const NUM_SUB_VECTORS: usize = 16;
        let codebook = Arc::new(Float32Array::from_iter_values(
            (0..256 * DIM).map(|v| v as f32),
        ));
        let pq = Arc::new(ProductQuantizerImpl::<Float32Type>::new(
            NUM_SUB_VECTORS,
            8,
            DIM,
            codebook.clone(),
            MetricType::L2,
        ));
        let schema = Schema::new(vec![
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

        let storage = ProductQuantizationStorage::new(pq.clone(), &batch, "vectors")
            .await
            .unwrap();
        assert_eq!(storage.len(), TOTAL);
        assert_eq!(storage.num_sub_vectors, NUM_SUB_VECTORS);
        assert_eq!(storage.codebook.len(), 256 * DIM);
        assert_eq!(storage.pq_code.len(), TOTAL * NUM_SUB_VECTORS);
        assert_eq!(storage.row_ids.len(), TOTAL);
    }
}
