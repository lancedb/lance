// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt::{Debug, Formatter};
use std::sync::{Arc, LazyLock};

use arrow::array::AsArray;
use arrow::datatypes::Float32Type;
use arrow_array::{Float32Array, RecordBatch};
use lance_arrow::RecordBatchExt;
use lance_core::{Error, Result};
use snafu::location;
use tracing::instrument;

use crate::vector::bq::builder::RabbitQuantizer;
use crate::vector::bq::storage::RABBIT_CODE_COLUMN;
use crate::vector::quantizer::Quantization;
use crate::vector::transform::Transformer;

// the inner product of quantized vector and the normalized residual vector.
pub const NORM_DIST_COLUMN: &str = "__norm_dist";
pub static NORM_DIST_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(NORM_DIST_COLUMN, arrow_schema::DataType::Float32, true)
});

pub struct RQTransformer {
    rq: RabbitQuantizer,
    vector_column: String,
}

impl RQTransformer {
    pub fn new(rq: RabbitQuantizer, vector_column: impl Into<String>) -> Self {
        Self {
            rq,
            vector_column: vector_column.into(),
        }
    }
}

impl Debug for RQTransformer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "RabbitTransformer(vector_column={})", self.vector_column)
    }
}

impl Transformer for RQTransformer {
    #[instrument(name = "RQTransformer::transform", level = "debug", skip_all)]
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        if batch.column_by_name(RABBIT_CODE_COLUMN).is_some() {
            return Ok(batch.clone());
        }
        let input_arr = batch
            .column_by_name(&self.vector_column)
            .ok_or(Error::Index {
                message: format!(
                    "RQ Transform: column {} not found in batch",
                    self.vector_column
                ),
                location: location!(),
            })?;
        let data = input_arr.as_fixed_size_list_opt().ok_or(Error::Index {
            message: format!(
                "RQ Transform: column {} is not a fixed size list, got {}",
                self.vector_column,
                input_arr.data_type(),
            ),
            location: location!(),
        })?;
        let rq_code = self.rq.quantize(&data)?;
        let norm_dists = self.rq.norm_dists::<Float32Type>(&data)?;
        let norm_dists = Float32Array::from(norm_dists);
        debug_assert_eq!(rq_code.len(), batch.num_rows());
        let batch = batch.try_with_column(self.rq.field(), Arc::new(rq_code))?;
        let batch = batch.try_with_column(NORM_DIST_FIELD.clone(), Arc::new(norm_dists))?;
        let batch = batch.drop_column(&self.vector_column)?;
        Ok(batch)
    }
}
