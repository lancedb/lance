// Copyright 2023 Lance Developers.
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

//! Vector Transforms
//!

use std::fmt::Debug;
use std::sync::Arc;

use arrow_array::{cast::AsArray, Array, RecordBatch};
use arrow_schema::Field;
use async_trait::async_trait;
use lance_arrow::RecordBatchExt;
use snafu::{location, Location};

use lance_core::{Error, Result};
use lance_linalg::kernels::normalize_fsl;

/// Transform of a Vector Matrix.
///
///
#[async_trait]
pub trait Transformer: Debug + Sync + Send {
    /// Transform a [`RecordBatch`] of vectors
    ///
    async fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch>;
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

#[async_trait]
impl Transformer for NormalizeTransformer {
    async fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
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
            return Ok(batch.try_with_column(field, Arc::new(norm))?);
        } else {
            return Ok(batch.replace_column_by_name(&self.input_column, Arc::new(norm))?);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use arrow_array::types::Float16Type;
    use arrow_array::{
        cast::AsArray, types::Float32Type, Array, FixedSizeListArray, Float16Array, Float32Array,
    };
    use arrow_schema::{DataType, Field, Schema};
    use half::f16;
    use lance_arrow::*;
    use num_traits::Float;

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
        let output = transformer.transform(&batch).await.unwrap();
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
        let data = Float16Array::from_iter_values(
            [1.0_f32, 1.0, 2.0, 2.0]
                .into_iter()
                .map(|v| f16::from_f32(v)),
        );
        let fsl = FixedSizeListArray::try_new_from_values(data, 2).unwrap();
        let schema = Schema::new(vec![Field::new(
            "v",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float16, true)), 2),
            true,
        )]);
        let batch = RecordBatch::try_new(schema.into(), vec![Arc::new(fsl)]).unwrap();
        let transformer = NormalizeTransformer::new("v");
        let output = transformer.transform(&batch).await.unwrap();
        let actual = output.column_by_name("v").unwrap();
        let act_fsl = actual.as_fixed_size_list();
        assert_eq!(act_fsl.len(), 2);
        let expect_1 = vec![f16::from_f32_const(1.0) / f16::from_f32_const(2.0).sqrt(); 2];
        act_fsl
            .value(0)
            .as_primitive::<Float16Type>()
            .values()
            .iter()
            .zip(expect_1.iter())
            .for_each(|(a, b)| assert!(a - b <= f16::epsilon()));

        let expect_2 = vec![f16::from_f32_const(2.0) / f16::from_f32_const(8.0).sqrt(); 2];
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
        let output = transformer.transform(&batch).await.unwrap();
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
}
