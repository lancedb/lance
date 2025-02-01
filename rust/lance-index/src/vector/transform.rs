// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector Transforms
//!

use std::fmt::Debug;
use std::sync::Arc;

use arrow::datatypes::UInt64Type;
use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use arrow_array::UInt64Array;
use arrow_array::{cast::AsArray, Array, ArrowPrimitiveType, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use lance_arrow::RecordBatchExt;
use num_traits::Float;
use snafu::location;

use lance_core::{Error, Result, ROW_ID, ROW_ID_FIELD};
use lance_linalg::kernels::normalize_fsl;
use tracing::instrument;

/// Transform of a Vector Matrix.
///
///
pub trait Transformer: Debug + Send + Sync {
    /// Transform a [`RecordBatch`] of vectors
    ///
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch>;
}

/// Normalize Transformer
///
/// L2 Normalize each vector.
#[derive(Debug)]
pub struct NormalizeTransformer {
    input_column: String,
    output_column: Option<String>,
}

impl NormalizeTransformer {
    pub fn new(column: impl AsRef<str>) -> Self {
        Self {
            input_column: column.as_ref().to_owned(),
            output_column: None,
        }
    }

    /// Create Normalize output transform that will be stored in a different column.
    ///
    pub fn new_with_output(input_column: impl AsRef<str>, output_column: impl AsRef<str>) -> Self {
        Self {
            input_column: input_column.as_ref().to_owned(),
            output_column: Some(output_column.as_ref().to_owned()),
        }
    }
}

impl Transformer for NormalizeTransformer {
    #[instrument(name = "NormalizeTransformer::transform", level = "debug", skip_all)]
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let arr = batch
            .column_by_name(&self.input_column)
            .ok_or(Error::Index {
                message: format!(
                    "Normalize Transform: column {} not found in RecordBatch",
                    self.input_column
                ),
                location: location!(),
            })?;

        let data = arr.as_fixed_size_list();
        let norm = normalize_fsl(data)?;
        let transformed = Arc::new(norm);

        if let Some(output_column) = &self.output_column {
            let field = Field::new(output_column, transformed.data_type().clone(), true);
            Ok(batch.try_with_column(field, transformed)?)
        } else {
            Ok(batch.replace_column_by_name(&self.input_column, transformed)?)
        }
    }
}

/// Only keep the vectors that is finite number, filter out NaN and Inf.
#[derive(Debug)]
pub(crate) struct KeepFiniteVectors {
    column: String,
}

impl KeepFiniteVectors {
    pub fn new(column: impl AsRef<str>) -> Self {
        Self {
            column: column.as_ref().to_owned(),
        }
    }
}

fn is_all_finite<T: ArrowPrimitiveType>(arr: &dyn Array) -> bool
where
    T::Native: Float,
{
    !arr.as_primitive::<T>()
        .values()
        .iter()
        .any(|&v| !v.is_finite())
}

impl Transformer for KeepFiniteVectors {
    #[instrument(name = "KeepFiniteVectors::transform", level = "debug", skip_all)]
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let arr = batch.column_by_name(&self.column).ok_or(Error::Index {
            message: format!(
                "KeepFiniteVectors: column {} not found in RecordBatch",
                self.column
            ),
            location: location!(),
        })?;

        let data = match arr.data_type() {
            DataType::FixedSizeList(_, _) => arr.as_fixed_size_list(),
            DataType::List(_) => arr.as_list::<i32>().values().as_fixed_size_list(),
            _ => {
                return Err(Error::Index {
                    message: format!(
                        "KeepFiniteVectors: column {} is not a fixed size list: {}",
                        self.column,
                        arr.data_type()
                    ),
                    location: location!(),
                })
            }
        };

        let mut valid = Vec::with_capacity(batch.num_rows());
        data.iter().enumerate().for_each(|(idx, arr)| {
            if let Some(data) = arr {
                let is_valid = match data.data_type() {
                    DataType::Float16 => is_all_finite::<Float16Type>(&data),
                    DataType::Float32 => is_all_finite::<Float32Type>(&data),
                    DataType::Float64 => is_all_finite::<Float64Type>(&data),
                    _ => false,
                };
                if is_valid {
                    valid.push(idx as u32);
                }
            };
        });
        if valid.len() < batch.num_rows() {
            let indices = UInt32Array::from(valid);
            Ok(batch.take(&indices)?)
        } else {
            Ok(batch.clone())
        }
    }
}

#[derive(Debug)]
pub struct DropColumn {
    column: String,
}

impl DropColumn {
    pub fn new(column: &str) -> Self {
        Self {
            column: column.to_owned(),
        }
    }
}

impl Transformer for DropColumn {
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        Ok(batch.drop_column(&self.column)?)
    }
}

#[derive(Debug)]
pub struct Flatten {
    column: String,
}

impl Flatten {
    pub fn new(column: &str) -> Self {
        Self {
            column: column.to_owned(),
        }
    }
}

impl Transformer for Flatten {
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let arr = batch.column_by_name(&self.column).ok_or(Error::Index {
            message: format!("Flatten: column {} not found in RecordBatch", self.column),
            location: location!(),
        })?;
        match arr.data_type() {
            DataType::FixedSizeList(_, _) => {
                // do nothing
                Ok(batch.clone())
            }
            DataType::List(_) => {
                let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
                let vectors = arr.as_list::<i32>();

                let row_ids = row_ids.values().iter().zip(vectors.iter()).flat_map(
                    |(row_id, multivector)| {
                        std::iter::repeat(*row_id)
                            .take(multivector.map(|multivec| multivec.len()).unwrap_or(0))
                    },
                );
                let row_ids = UInt64Array::from_iter_values(row_ids);
                let vectors = vectors.values().as_fixed_size_list().clone();
                let schema = Arc::new(Schema::new(vec![
                    ROW_ID_FIELD.clone(),
                    Field::new(self.column.as_str(), vectors.data_type().clone(), true),
                ]));
                let batch =
                    RecordBatch::try_new(schema, vec![Arc::new(row_ids), Arc::new(vectors)])?;
                Ok(batch)
            }
            _ => Err(Error::Index {
                message: format!(
                    "Flatten: column {} is not a vector: {}",
                    self.column,
                    arr.data_type()
                ),
                location: location!(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use arrow_array::{FixedSizeListArray, Float16Array, Float32Array, Int32Array};
    use arrow_schema::Schema;
    use half::f16;
    use lance_arrow::*;

    #[tokio::test]
    async fn test_normalize_transformer_f32() {
        let data = Float32Array::from_iter_values([1.0, 1.0, 2.0, 2.0].into_iter());
        let fsl = FixedSizeListArray::try_new_from_values(data, 2).unwrap();
        let schema = Schema::new(vec![Field::new(
            "v",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
            true,
        )]);
        let batch = RecordBatch::try_new(schema.into(), vec![Arc::new(fsl)]).unwrap();
        let transformer = NormalizeTransformer::new("v");
        let output = transformer.transform(&batch).unwrap();
        let actual = output.column_by_name("v").unwrap();
        let act_fsl = actual.as_fixed_size_list();
        assert_eq!(act_fsl.len(), 2);
        assert_relative_eq!(
            act_fsl.value(0).as_primitive::<Float32Type>().values()[..],
            [1.0 / 2.0_f32.sqrt(); 2]
        );
        assert_relative_eq!(
            act_fsl.value(1).as_primitive::<Float32Type>().values()[..],
            [2.0 / 8.0_f32.sqrt(); 2]
        );
    }

    #[tokio::test]
    async fn test_normalize_transformer_16() {
        let data =
            Float16Array::from_iter_values([1.0_f32, 1.0, 2.0, 2.0].into_iter().map(f16::from_f32));
        let fsl = FixedSizeListArray::try_new_from_values(data, 2).unwrap();
        let schema = Schema::new(vec![Field::new(
            "v",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float16, true)), 2),
            true,
        )]);
        let batch = RecordBatch::try_new(schema.into(), vec![Arc::new(fsl)]).unwrap();
        let transformer = NormalizeTransformer::new("v");
        let output = transformer.transform(&batch).unwrap();
        let actual = output.column_by_name("v").unwrap();
        let act_fsl = actual.as_fixed_size_list();
        assert_eq!(act_fsl.len(), 2);
        let expect_1 = [f16::from_f32_const(1.0) / f16::from_f32_const(2.0).sqrt(); 2];
        act_fsl
            .value(0)
            .as_primitive::<Float16Type>()
            .values()
            .iter()
            .zip(expect_1.iter())
            .for_each(|(a, b)| assert!(a - b <= f16::epsilon()));

        let expect_2 = [f16::from_f32_const(2.0) / f16::from_f32_const(8.0).sqrt(); 2];
        act_fsl
            .value(1)
            .as_primitive::<Float16Type>()
            .values()
            .iter()
            .zip(expect_2.iter())
            .for_each(|(a, b)| assert!(a - b <= f16::epsilon()));
    }

    #[tokio::test]
    async fn test_normalize_transformer_with_output_column() {
        let data = Float32Array::from_iter_values([1.0, 1.0, 2.0, 2.0].into_iter());
        let fsl = FixedSizeListArray::try_new_from_values(data, 2).unwrap();
        let schema = Schema::new(vec![Field::new(
            "v",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
            true,
        )]);
        let batch = RecordBatch::try_new(schema.into(), vec![Arc::new(fsl.clone())]).unwrap();
        let transformer = NormalizeTransformer::new_with_output("v", "o");
        let output = transformer.transform(&batch).unwrap();
        let input = output.column_by_name("v").unwrap();
        assert_eq!(input.as_ref(), &fsl);
        let actual = output.column_by_name("o").unwrap();
        let act_fsl = actual.as_fixed_size_list();
        assert_eq!(act_fsl.len(), 2);
        assert_relative_eq!(
            act_fsl.value(0).as_primitive::<Float32Type>().values()[..],
            [1.0 / 2.0_f32.sqrt(); 2]
        );
        assert_relative_eq!(
            act_fsl.value(1).as_primitive::<Float32Type>().values()[..],
            [2.0 / 8.0_f32.sqrt(); 2]
        );
    }

    #[tokio::test]
    async fn test_drop_column() {
        let i32_array = Int32Array::from_iter_values([1, 2].into_iter());
        let data = Float32Array::from_iter_values([1.0, 1.0, 2.0, 2.0].into_iter());
        let fsl = FixedSizeListArray::try_new_from_values(data, 2).unwrap();
        let schema = Schema::new(vec![
            Field::new("i32", DataType::Int32, false),
            Field::new(
                "v",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
                true,
            ),
        ]);
        let batch =
            RecordBatch::try_new(schema.into(), vec![Arc::new(i32_array), Arc::new(fsl)]).unwrap();
        let transformer = DropColumn::new("v");
        let output = transformer.transform(&batch).unwrap();
        assert!(output.column_by_name("v").is_none());

        let dup_drop_result = transformer.transform(&output);
        assert!(dup_drop_result.is_ok());
    }
}
