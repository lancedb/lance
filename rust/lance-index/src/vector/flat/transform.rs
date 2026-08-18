// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::RecordBatch;
use arrow_schema::Field;
use lance_arrow::RecordBatchExt;
use lance_core::Error;
use tracing::instrument;

use crate::vector::transform::Transformer;

use super::storage::FLAT_COLUMN;

#[derive(Debug)]
pub struct FlatTransformer {
    input_column: String,
}

impl FlatTransformer {
    pub fn new(input_column: impl AsRef<str>) -> Self {
        Self {
            input_column: input_column.as_ref().to_owned(),
        }
    }
}

impl Transformer for FlatTransformer {
    #[instrument(name = "FlatTransformer::transform", level = "debug", skip_all)]
    fn transform(&self, batch: &RecordBatch) -> lance_core::Result<RecordBatch> {
        let input_arr = batch
            .column_by_name(&self.input_column)
            .ok_or(Error::index(format!(
                "FlatTransform: column {} not found in batch",
                self.input_column
            )))?;
        let field = Field::new(
            FLAT_COLUMN,
            input_arr.data_type().clone(),
            input_arr.is_nullable(),
        );
        // rename the column to FLAT_COLUMN
        let batch = batch
            .drop_column(&self.input_column)?
            .try_with_column(field, input_arr.clone())?;
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::{Array, FixedSizeListArray, Float32Array, Int32Array};
    use arrow_schema::{DataType, Schema};
    use lance_arrow::FixedSizeListArrayExt;

    const DIM: i32 = 4;
    const ROWS: usize = 8;

    fn batch() -> RecordBatch {
        let values = Float32Array::from_iter((0..(DIM as usize * ROWS)).map(|v| v as f32));
        let vectors = Arc::new(FixedSizeListArray::try_new_from_values(values, DIM).unwrap());
        let schema = Schema::new(vec![
            Field::new("vec", vectors.data_type().clone(), true),
            Field::new("other", DataType::Int32, false),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                vectors,
                Arc::new(Int32Array::from_iter_values(0..ROWS as i32)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_flat_transform_renames_vector_column() {
        let input = batch();
        let output = FlatTransformer::new("vec").transform(&input).unwrap();

        assert!(output.column_by_name("vec").is_none());
        assert!(
            output.column_by_name("other").is_some(),
            "unrelated columns should survive"
        );
        assert_eq!(output.num_rows(), ROWS);

        let flat = output.column_by_name(FLAT_COLUMN).unwrap();
        assert_eq!(
            flat.data_type(),
            input.column_by_name("vec").unwrap().data_type()
        );
        assert_eq!(flat.as_ref(), input.column_by_name("vec").unwrap().as_ref());
    }

    #[test]
    fn test_flat_transform_preserves_nullability() {
        // The renamed field copies the source column's nullability rather than
        // hardcoding it, so a non-nullable vector column stays non-nullable.
        let values = Float32Array::from_iter((0..(DIM as usize * ROWS)).map(|v| v as f32));
        let vectors = Arc::new(FixedSizeListArray::try_new_from_values(values, DIM).unwrap());
        let schema = Schema::new(vec![
            Field::new("vec", vectors.data_type().clone(), false),
            Field::new("other", DataType::Int32, false),
        ]);
        let input = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                vectors,
                Arc::new(Int32Array::from_iter_values(0..ROWS as i32)),
            ],
        )
        .unwrap();

        let output = FlatTransformer::new("vec").transform(&input).unwrap();
        let field = output
            .schema()
            .field_with_name(FLAT_COLUMN)
            .unwrap()
            .clone();
        assert!(!field.is_nullable());
    }

    #[test]
    fn test_flat_transform_reports_missing_column() {
        let message = FlatTransformer::new("absent")
            .transform(&batch())
            .unwrap_err()
            .to_string();
        assert!(message.contains("column absent"), "{message}");
    }
}
