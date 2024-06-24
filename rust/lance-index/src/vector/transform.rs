// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector Transforms
//!

use std::fmt::Debug;
use std::sync::Arc;

use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use arrow_array::{cast::AsArray, Array, ArrowPrimitiveType, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field};
use lance_arrow::RecordBatchExt;
use num_traits::Float;
use snafu::{location, Location};

use lance_core::{Error, Result};
use lance_linalg::kernels::normalize_fsl;

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
        let data = arr.as_fixed_size_list_opt().ok_or(Error::Index {
            message: format!(
                "Normalize Transform: column {} is not a fixed size list: {}",
                self.input_column,
                arr.data_type()
            ),
            location: location!(),
        })?;
        let norm = normalize_fsl(data)?;
        if let Some(output_column) = &self.output_column {
            let field = Field::new(output_column, norm.data_type().clone(), true);
            Ok(batch.try_with_column(field, Arc::new(norm))?)
        } else {
            Ok(batch.replace_column_by_name(&self.input_column, Arc::new(norm))?)
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
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let arr = batch.column_by_name(&self.column).ok_or(Error::Index {
            message: format!(
                "KeepFiniteVectors: column {} not found in RecordBatch",
                self.column
            ),
            location: location!(),
        })?;
        let data = arr.as_fixed_size_list_opt().ok_or(Error::Index {
            message: format!(
                "KeepFiniteVectors: column {} is not a fixed size list: {}",
                self.column,
                arr.data_type()
            ),
            location: location!(),
        })?;

        let valid = data
            .iter()
            .enumerate()
            .filter_map(|(idx, arr)| {
                arr.and_then(|data| {
                    let is_valid = match data.data_type() {
                        DataType::Float16 => is_all_finite::<Float16Type>(&data),
                        DataType::Float32 => is_all_finite::<Float32Type>(&data),
                        DataType::Float64 => is_all_finite::<Float64Type>(&data),
                        DataType::Int8 => true,
                        _ => false,
                    };
                    if is_valid {
                        Some(idx as u32)
                    } else {
                        None
                    }
                })
            })
            .collect::<Vec<_>>();
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
