// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt::{Debug, Formatter};
use std::sync::{Arc, LazyLock};

use arrow::array::AsArray;
use arrow::datatypes::{Float32Type, UInt32Type, UInt8Type};
use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch};
use lance_arrow::RecordBatchExt;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use snafu::location;
use tracing::instrument;

use crate::vector::bq::builder::RabitQuantizer;
use crate::vector::bq::storage::RABIT_CODE_COLUMN;
use crate::vector::quantizer::Quantization;
use crate::vector::residual::RESIDUAL_COLUMN;
use crate::vector::transform::Transformer;
use crate::vector::{CENTROID_DIST_COLUMN, PART_ID_COLUMN};

pub const CODE_BITCOUNT_COLUMN: &str = "__code_bitcount";
// the inner product of quantized vector and the residual vector.
pub const ADD_FACTORS_COLUMN: &str = "__add_factors";
// the inner product of quantized vector and the centroid vector.
pub const SCALE_FACTORS_COLUMN: &str = "__scale_factors";

pub static CODE_BITCOUNT_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(CODE_BITCOUNT_COLUMN, arrow_schema::DataType::Float32, true)
});
pub static ADD_FACTORS_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(ADD_FACTORS_COLUMN, arrow_schema::DataType::Float32, true)
});
pub static SCALE_FACTORS_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(SCALE_FACTORS_COLUMN, arrow_schema::DataType::Float32, true)
});

pub struct RQTransformer {
    rq: RabitQuantizer,
    distance_type: DistanceType,
    centroids_norm_square: Option<Float32Array>,
    vector_column: String,
}

impl RQTransformer {
    pub fn new(
        rq: RabitQuantizer,
        distance_type: DistanceType,
        centroids: FixedSizeListArray,
        vector_column: impl Into<String>,
    ) -> Self {
        // for dot product, the add factor is `1 - v*c + |c|^2`, so we need to compute |c|^2
        let centroids_norm_square = (distance_type == DistanceType::Dot).then(|| {
            Float32Array::from(
                centroids
                    .iter()
                    .map(|v| {
                        v.map(|v| {
                            v.as_primitive::<Float32Type>()
                                .values()
                                .iter()
                                .map(|v| v * v)
                                .sum::<f32>()
                        })
                        .unwrap_or_default()
                    })
                    .collect::<Vec<_>>(),
            )
        });

        Self {
            rq,
            distance_type,
            centroids_norm_square,
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

        let dist_v_c = batch
            .column_by_name(CENTROID_DIST_COLUMN)
            .ok_or(Error::Index {
                message: format!(
                    "RQ Transform: column {} not found in batch",
                    CENTROID_DIST_COLUMN
                ),
                location: location!(),
            })?;
        let dist_v_c = dist_v_c.as_primitive::<Float32Type>();

        let res_norm_square = match self.distance_type {
            // for L2, |v-c|^2 is just the distance to the centroid
            DistanceType::L2 => dist_v_c.clone(),
            DistanceType::Dot => Float32Array::from(
                residual_vectors
                    .iter()
                    .map(|v| {
                        v.map(|v| {
                            v.as_primitive::<Float32Type>()
                                .values()
                                .iter()
                                .map(|v| v * v)
                                .sum::<f32>()
                        })
                    })
                    .collect::<Vec<_>>(),
            ),
            _ => {
                return Err(Error::Index {
                    message: format!(
                        "RQ Transform: distance type {} not supported",
                        self.distance_type
                    ),
                    location: location!(),
                });
            }
        };

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
        let ip_rq_res = Float32Array::from(
            self.rq
                .codes_res_dot_dists::<Float32Type>(&residual_vectors)?,
        );
        debug_assert_eq!(rq_code.len(), batch.num_rows());

        let add_factors = match self.distance_type {
            DistanceType::L2 => res_norm_square.clone(),
            DistanceType::Dot => {
                // for dot, the add factor is `1 - v*c + |c|^2 = dist_v_c + |c|^2`
                let part_ids = &batch[PART_ID_COLUMN];
                let part_ids = part_ids.as_primitive::<UInt32Type>();
                let centroids_norm_square =
                    self.centroids_norm_square.as_ref().ok_or(Error::Index {
                        message: format!("RQ Transform: centroids norm square not found"),
                        location: location!(),
                    })?;
                let centroids_norm_square =
                    arrow::compute::take(centroids_norm_square, part_ids, None)?;
                let centroids_norm_square = centroids_norm_square.as_primitive::<Float32Type>();
                Float32Array::from_iter_values(
                    dist_v_c
                        .values()
                        .iter()
                        .zip(centroids_norm_square.values().iter())
                        .map(|(dist_v_c, centroids_norm_square)| dist_v_c + centroids_norm_square),
                )
            }
            _ => {
                return Err(Error::Index {
                    message: format!(
                        "RQ Transform: distance type {} not supported",
                        self.distance_type
                    ),
                    location: location!(),
                });
            }
        };

        let scale_factors = match self.distance_type {
            DistanceType::L2 => Float32Array::from_iter_values(
                res_norm_square
                    .values()
                    .iter()
                    .zip(ip_rq_res.values())
                    .map(|(res_norm_square, ip_rq_res)| -2.0 * res_norm_square / ip_rq_res),
            ),
            DistanceType::Dot => Float32Array::from_iter_values(
                res_norm_square
                    .values()
                    .iter()
                    .zip(ip_rq_res.values())
                    .map(|(res_norm_square, ip_rq_res)| -res_norm_square / ip_rq_res),
            ),
            _ => {
                return Err(Error::Index {
                    message: format!(
                        "RQ Transform: distance type {} not supported",
                        self.distance_type
                    ),
                    location: location!(),
                });
            }
        };

        let batch = batch.try_with_column(self.rq.field(), Arc::new(rq_code))?;
        let batch = batch.try_with_column(CODE_BITCOUNT_FIELD.clone(), Arc::new(code_bitcount))?;
        let batch = batch
            .try_with_column(ADD_FACTORS_FIELD.clone(), Arc::new(add_factors))?
            .drop_column(CENTROID_DIST_COLUMN)?;
        let batch = batch.try_with_column(SCALE_FACTORS_FIELD.clone(), Arc::new(scale_factors))?;
        let batch = batch.drop_column(&self.vector_column)?;
        Ok(batch)
    }
}
