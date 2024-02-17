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
    types::{Float32Type, UInt8Type},
    FixedSizeListArray, Float32Array, RecordBatch, UInt64Array, UInt8Array,
};
use arrow_schema::SchemaRef;
use lance_core::Result;
use lance_linalg::distance::MetricType;

use super::{ProductQuantizer, ProductQuantizerImpl};
use crate::vector::{pq::transform::PQTransformer, transform::Transformer, PQ_CODE_COLUMN};

/// Product Quantization Storage
pub struct ProductQuantizationStorage {
    codebook: Arc<Float32Array>,
    batch: RecordBatch,
    // pq_code: Arc<UInt8Array>,
    // row_ids: Arc<UInt64Array>,
}

impl ProductQuantizationStorage {
    pub async fn new(
        quantizer: Arc<ProductQuantizerImpl<Float32Type>>,
        batch: &RecordBatch,
        vector_col: &str,
    ) -> Result<Self> {
        let codebook = quantizer.codebook.clone();
        let transform = PQTransformer::new(quantizer, vector_col, PQ_CODE_COLUMN);
        let batch = transform.transform(batch).await?;
        Ok(Self {
            codebook,
            batch,
            // pq_code: pq_code.into(),
            // row_ids,
        })
    }

    pub fn schema(&self) -> SchemaRef {
        self.batch.schema()
    }
}

#[cfg(test)]
mod tests {
    use crate::vector::graph::storage;

    use super::*;

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

        let storage = ProductQuantizationStorage::new(pq, &batch, "vectors")
            .await
            .unwrap();
    }
}
