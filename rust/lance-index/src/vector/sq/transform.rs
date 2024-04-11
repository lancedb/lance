// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    fmt::{Debug, Formatter},
    marker::PhantomData,
    sync::Arc,
};

use arrow_array::RecordBatch;
use arrow_schema::Field;
use snafu::{location, Location};

use crate::vector::transform::Transformer;

use lance_arrow::{ArrowFloatType, RecordBatchExt};
use lance_core::{Error, Result};

use super::ScalarQuantizer;

pub struct SQTransformer<T: ArrowFloatType> {
    quantizer: ScalarQuantizer,
    input_column: String,
    output_column: String,

    _mark: PhantomData<fn() -> T>,
}

impl<T: ArrowFloatType> SQTransformer<T> {
    pub fn new(quantizer: ScalarQuantizer, input_column: String, output_column: String) -> Self {
        Self {
            quantizer,
            input_column,
            output_column,
            _mark: PhantomData,
        }
    }
}

impl<T: ArrowFloatType> Debug for SQTransformer<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SQTransformer(input={}, output={})",
            self.input_column, self.output_column
        )
    }
}

#[async_trait::async_trait]
impl<T: ArrowFloatType> Transformer for SQTransformer<T> {
    async fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let input = batch
            .column_by_name(&self.input_column)
            .ok_or(Error::Index {
                message: format!(
                    "SQ Transform: column {} not found in batch",
                    self.input_column
                ),
                location: location!(),
            })?;
        let batch = batch.drop_column(&self.input_column)?;

        let sq_code = self.quantizer.transform::<T>(input)?;
        let sq_field = Field::new(&self.output_column, sq_code.data_type().clone(), false);
        let batch = batch.try_with_column(sq_field, Arc::new(sq_code))?;
        Ok(batch)
    }
}
