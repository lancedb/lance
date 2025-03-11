// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use arrow::datatypes::UInt8Type;
use arrow_array::FixedSizeListArray;
use arrow_array::{cast::AsArray, Array, RecordBatch};
use arrow_schema::Field;
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_core::{Error, Result};
use snafu::location;
use tracing::instrument;

use super::storage::{transpose, ProductQuantizationMetadata};
use super::ProductQuantizer;
use crate::vector::quantizer::Quantization;
use crate::vector::storage::STORAGE_METADATA_KEY;
use crate::vector::transform::Transformer;
use crate::vector::PQ_CODE_COLUMN;

/// Product Quantizer Transformer
///
/// It transforms a column of vectors into a column of PQ codes.
pub struct PQTransformer {
    quantizer: ProductQuantizer,
    input_column: String,
    output_column: String,
}

impl PQTransformer {
    pub fn new(quantizer: ProductQuantizer, input_column: &str, output_column: &str) -> Self {
        Self {
            quantizer,
            input_column: input_column.to_owned(),
            output_column: output_column.to_owned(),
        }
    }
}

impl Debug for PQTransformer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PQTransformer(input={}, output={})",
            self.input_column, self.output_column
        )
    }
}

impl Transformer for PQTransformer {
    #[instrument(name = "PQTransformer::transform", level = "debug", skip_all)]
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let input_arr = batch
            .column_by_name(&self.input_column)
            .ok_or(Error::Index {
                message: format!(
                    "PQ Transform: column {} not found in batch",
                    self.input_column
                ),
                location: location!(),
            })?;
        let data = input_arr.as_fixed_size_list_opt().ok_or(Error::Index {
            message: format!(
                "PQ Transform: column {} is not a fixed size list, got {}",
                self.input_column,
                input_arr.data_type(),
            ),
            location: location!(),
        })?;
        let pq_code = self.quantizer.quantize(&data)?;
        let pq_field = Field::new(&self.output_column, pq_code.data_type().clone(), false);
        let batch = batch.try_with_column(pq_field, Arc::new(pq_code))?;
        let batch = batch.drop_column(&self.input_column)?;
        Ok(batch)
    }
}

// this transpose transformer would transform the PQ codes back to original codes,
// we need this because if the PQ codes are stored in a transposed way,
// then we can't directly concat the PQ codes from different batches.
#[derive(Debug)]
pub struct TransposeTransformer {
    metadata_json: String,
    metadata: ProductQuantizationMetadata,
}

impl TransposeTransformer {
    pub fn new(metadata_json: String) -> Result<Self> {
        let metadata: ProductQuantizationMetadata = serde_json::from_str(&metadata_json)?;
        Ok(Self {
            metadata_json,
            metadata,
        })
    }
}

impl Transformer for TransposeTransformer {
    #[instrument(name = "TransposeTransformer::transform", level = "debug", skip_all)]
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let is_transposed = batch
            .metadata()
            .get(STORAGE_METADATA_KEY)
            .map(|v| serde_json::from_str::<ProductQuantizationMetadata>(v))
            .transpose()
            .unwrap_or_default()
            .is_some_and(|meta| meta.transposed);
        if !is_transposed {
            return Ok(batch.with_metadata(HashMap::new())?); // clear the metadata
        }

        let num_sub_vectors_in_byte = if self.metadata.nbits == 4 {
            self.metadata.num_sub_vectors / 2
        } else {
            self.metadata.num_sub_vectors
        };
        let codes = &batch[PQ_CODE_COLUMN];
        let transposed_codes = transpose(
            codes
                .as_fixed_size_list()
                .values()
                .as_primitive::<UInt8Type>(),
            num_sub_vectors_in_byte,
            batch.num_rows(),
        );
        let transposed_codes = FixedSizeListArray::try_new_from_values(
            transposed_codes,
            num_sub_vectors_in_byte as i32,
        )?;
        let batch = batch
            .replace_column_by_name(PQ_CODE_COLUMN, Arc::new(transposed_codes))?
            .with_metadata(HashMap::new())?; // clear the metadata
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{FixedSizeListArray, Float32Array, Int32Array};
    use arrow_schema::{DataType, Schema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_linalg::distance::DistanceType;

    use crate::vector::pq::PQBuildParams;

    #[tokio::test]
    async fn test_pq_transform() {
        let values = Float32Array::from_iter((0..16000).map(|v| v as f32));
        let dim = 16;
        let arr = Arc::new(FixedSizeListArray::try_new_from_values(values, 16).unwrap());
        let params = PQBuildParams::new(1, 8);
        let pq = ProductQuantizer::build(arr.as_ref(), DistanceType::L2, &params).unwrap();

        let schema = Schema::new(vec![
            Field::new(
                "vec",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                true,
            ),
            Field::new("other", DataType::Int32, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![arr, Arc::new(Int32Array::from_iter_values(0..1000))],
        )
        .unwrap();

        let transformer = PQTransformer::new(pq, "vec", "pq_code");
        let batch = transformer.transform(&batch).unwrap();
        assert!(batch.column_by_name("vec").is_none());
        assert!(batch.column_by_name("pq_code").is_some());
        assert!(batch.column_by_name("other").is_some());
        assert_eq!(batch.num_rows(), 1000)
    }
}
