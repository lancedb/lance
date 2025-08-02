// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt::{Debug, Formatter};
use std::sync::{Arc, LazyLock};

use arrow::array::AsArray;
use arrow::datatypes::{Float32Type, UInt32Type, UInt8Type};
use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch};
use lance_arrow::RecordBatchExt;
use lance_core::{Error, Result};
use snafu::location;
use tracing::instrument;

use crate::vector::bq::builder::RabitQuantizer;
use crate::vector::bq::storage::RABIT_CODE_COLUMN;
use crate::vector::quantizer::Quantization;
use crate::vector::residual::RESIDUAL_COLUMN;
use crate::vector::transform::Transformer;
use crate::vector::PART_ID_COLUMN;

pub const CODE_BITCOUNT_COLUMN: &str = "__code_bitcount";
// the inner product of quantized vector and the residual vector.
pub const IP_RQ_RES_COLUMN: &str = "__ip_rq_res";
// the inner product of quantized vector and the centroid vector.
pub const IP_RQ_CENTROID_COLUMN: &str = "__ip_rq_centroid";

pub static CODE_BITCOUNT_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(CODE_BITCOUNT_COLUMN, arrow_schema::DataType::Float32, true)
});
pub static IP_RQ_RES_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(IP_RQ_RES_COLUMN, arrow_schema::DataType::Float32, true)
});
pub static IP_RQ_CENTROID_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(IP_RQ_CENTROID_COLUMN, arrow_schema::DataType::Float32, true)
});

pub struct RQTransformer {
    rq: RabitQuantizer,
    vector_column: String,
}

impl RQTransformer {
    pub fn new(rq: RabitQuantizer, vector_column: impl Into<String>) -> Self {
        Self {
            rq,
            vector_column: vector_column.into(),
        }
    }
}

impl Debug for RQTransformer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "RabitTransformer(vector_column={})", self.vector_column)
    }
}

impl Transformer for RQTransformer {
    #[instrument(name = "RQTransformer::transform", level = "debug", skip_all)]
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        if batch.column_by_name(RABIT_CODE_COLUMN).is_some() {
            return Ok(batch.clone());
        }
        let vectors = batch
            .column_by_name(&self.vector_column)
            .ok_or(Error::Index {
                message: format!(
                    "RQ Transform: column {} not found in batch",
                    self.vector_column
                ),
                location: location!(),
            })?;
        let vectors = vectors.as_fixed_size_list_opt().ok_or(Error::Index {
            message: format!(
                "RQ Transform: column {} is not a fixed size list, got {}",
                self.vector_column,
                vectors.data_type()
            ),
            location: location!(),
        })?;

        let residual_vectors = batch.column_by_name(RESIDUAL_COLUMN).ok_or(Error::Index {
            message: format!(
                "RQ Transform: column {} not found in batch",
                RESIDUAL_COLUMN
            ),
            location: location!(),
        })?;
        let residual_vectors = residual_vectors
            .as_fixed_size_list_opt()
            .ok_or(Error::Index {
                message: format!(
                    "RQ Transform: column {} is not a fixed size list, got {}",
                    RESIDUAL_COLUMN,
                    residual_vectors.data_type(),
                ),
                location: location!(),
            })?;

        let rq_code = self.rq.quantize(&residual_vectors)?;
        let code_bitcount =
            Float32Array::from_iter_values(rq_code.as_fixed_size_list().iter().map(|v| {
                v.map(|v| {
                    v.as_primitive::<UInt8Type>()
                        .values()
                        .iter()
                        .map(|v| v.count_ones())
                        .sum::<u32>() as f32
                })
                .unwrap_or_default()
            }));
        let ip_rq_res_dists = Float32Array::from(
            self.rq
                .codes_res_dot_dists::<Float32Type>(&residual_vectors)?,
        );
        debug_assert_eq!(rq_code.len(), batch.num_rows());
        let ip_rq_centroid_dists = self.rq.codes_dot_centroids(
            rq_code.as_fixed_size_list(),
            &vectors,
            &ip_rq_res_dists,
        )?;
        debug_assert_eq!(ip_rq_centroid_dists.len(), batch.num_rows());

        let batch = batch.try_with_column(self.rq.field(), Arc::new(rq_code))?;
        let batch = batch.try_with_column(CODE_BITCOUNT_FIELD.clone(), Arc::new(code_bitcount))?;
        let batch = batch.try_with_column(IP_RQ_RES_FIELD.clone(), Arc::new(ip_rq_res_dists))?;
        let batch =
            batch.try_with_column(IP_RQ_CENTROID_FIELD.clone(), Arc::new(ip_rq_centroid_dists))?;
        let batch = batch.drop_column(&self.vector_column)?;
        Ok(batch)
    }
}
