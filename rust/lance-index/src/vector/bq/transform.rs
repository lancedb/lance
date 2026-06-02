// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt::{Debug, Formatter};
use std::sync::{Arc, LazyLock};

use arrow::array::AsArray;
use arrow::datatypes::{Float16Type, Float32Type, Float64Type, UInt32Type};
use arrow_array::{Array, ArrowNativeTypeOp, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::DataType;
use lance_arrow::RecordBatchExt;
use lance_core::{Error, Result};
use lance_linalg::distance::{DistanceType, norm_squared_fsl};
use tracing::instrument;

use crate::vector::bq::builder::RabitQuantizer;
use crate::vector::bq::storage::{RABIT_CODE_COLUMN, RABIT_EX_CODE_COLUMN};
use crate::vector::quantizer::Quantization;
use crate::vector::transform::Transformer;
use crate::vector::{CENTROID_DIST_COLUMN, PART_ID_COLUMN};

// the inner product of quantized vector and the residual vector.
pub const ADD_FACTORS_COLUMN: &str = "__add_factors";
// the inner product of quantized vector and the centroid vector.
pub const SCALE_FACTORS_COLUMN: &str = "__scale_factors";
pub const EX_SCALE_FACTORS_COLUMN: &str = "__scale_factors_ex";

pub static ADD_FACTORS_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(ADD_FACTORS_COLUMN, arrow_schema::DataType::Float32, true)
});
pub static SCALE_FACTORS_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(SCALE_FACTORS_COLUMN, arrow_schema::DataType::Float32, true)
});
pub static EX_SCALE_FACTORS_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(
        EX_SCALE_FACTORS_COLUMN,
        arrow_schema::DataType::Float32,
        true,
    )
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
        let centroids_norm_square = (distance_type == DistanceType::Dot)
            .then(|| Float32Array::from(norm_squared_fsl(&centroids)));

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
        let has_split_codes = self.rq.num_bits() == 1
            || (batch.column_by_name(RABIT_EX_CODE_COLUMN).is_some()
                && batch.column_by_name(EX_SCALE_FACTORS_COLUMN).is_some());
        if batch.column_by_name(RABIT_CODE_COLUMN).is_some() && has_split_codes {
            return Ok(batch.clone());
        }

        let residual_vectors = batch
            .column_by_name(&self.vector_column)
            .ok_or(Error::index(format!(
                "RQ Transform: column {} not found in batch",
                self.vector_column
            )))?;
        let residual_vectors = residual_vectors
            .as_fixed_size_list_opt()
            .ok_or(Error::index(format!(
                "RQ Transform: column {} is not a fixed size list, got {}",
                self.vector_column,
                residual_vectors.data_type(),
            )))?;

        let dist_v_c = batch
            .column_by_name(CENTROID_DIST_COLUMN)
            .ok_or(Error::index(format!(
                "RQ Transform: column {} not found in batch",
                CENTROID_DIST_COLUMN
            )))?;
        let dist_v_c = dist_v_c.as_primitive::<Float32Type>();

        let res_norm_square = match self.distance_type {
            // for L2, |v-c|^2 is just the distance to the centroid
            DistanceType::L2 => dist_v_c.clone(),
            DistanceType::Dot => Float32Array::from(norm_squared_fsl(residual_vectors)),
            _ => {
                return Err(Error::index(format!(
                    "RQ Transform: distance type {} not supported",
                    self.distance_type
                )));
            }
        };

        let rq_codes = self.rq.quantize_split(residual_vectors)?;
        let codes_fsl = rq_codes.binary_codes.as_fixed_size_list();

        let ip_rq_res = match residual_vectors.value_type() {
            DataType::Float16 => Float32Array::from(
                self.rq
                    .codes_res_dot_dists::<Float16Type>(residual_vectors)?,
            ),
            DataType::Float32 => Float32Array::from(
                self.rq
                    .codes_res_dot_dists::<Float32Type>(residual_vectors)?,
            ),
            DataType::Float64 => Float32Array::from(
                self.rq
                    .codes_res_dot_dists::<Float64Type>(residual_vectors)?,
            ),
            _ => {
                return Err(Error::index(format!(
                    "RQ Transform: unsupported residual vector data type: {}",
                    residual_vectors.data_type()
                )));
            }
        };
        debug_assert_eq!(codes_fsl.len(), batch.num_rows());

        let add_factors = match self.distance_type {
            DistanceType::L2 => res_norm_square.clone(),
            DistanceType::Dot => {
                // for dot, the add factor is `1 - v*c + |c|^2 = dist_v_c + |c|^2`
                let part_ids = &batch[PART_ID_COLUMN];
                let part_ids = part_ids.as_primitive::<UInt32Type>();
                let centroids_norm_square = self.centroids_norm_square.as_ref().ok_or(
                    Error::index("RQ Transform: centroids norm square not found".to_string()),
                )?;
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
                return Err(Error::index(format!(
                    "RQ Transform: distance type {} not supported",
                    self.distance_type
                )));
            }
        };

        let scale_factors = match self.distance_type {
            DistanceType::L2 => Float32Array::from_iter_values(
                res_norm_square.values().iter().zip(ip_rq_res.values()).map(
                    |(res_norm_square, ip_rq_res)| {
                        (-2.0 * res_norm_square)
                            .div_checked(*ip_rq_res)
                            .unwrap_or_default()
                    },
                ),
            ),
            DistanceType::Dot => Float32Array::from_iter_values(
                res_norm_square.values().iter().zip(ip_rq_res.values()).map(
                    |(res_norm_square, ip_rq_res)| {
                        -res_norm_square.div_checked(*ip_rq_res).unwrap_or_default()
                    },
                ),
            ),
            _ => {
                return Err(Error::index(format!(
                    "RQ Transform: distance type {} not supported",
                    self.distance_type
                )));
            }
        };

        let batch = batch.try_with_column(self.rq.field(), rq_codes.binary_codes)?;
        let batch = batch.try_with_column(ADD_FACTORS_FIELD.clone(), Arc::new(add_factors))?;
        let mut batch =
            batch.try_with_column(SCALE_FACTORS_FIELD.clone(), Arc::new(scale_factors))?;

        if let (Some(ex_codes), Some(ex_res_dot_dists)) =
            (rq_codes.ex_codes, rq_codes.ex_res_dot_dists)
        {
            // Lance's IVF_RQ estimator uses residual queries, so the ex-code
            // path shares the additive factor and only needs a separate scale.
            let ex_scale_factors = match self.distance_type {
                DistanceType::L2 => Float32Array::from_iter_values(
                    res_norm_square
                        .values()
                        .iter()
                        .zip(ex_res_dot_dists.iter())
                        .map(|(res_norm_square, ex_res_dot)| {
                            (-2.0 * res_norm_square)
                                .div_checked(*ex_res_dot)
                                .unwrap_or_default()
                        }),
                ),
                DistanceType::Dot => Float32Array::from_iter_values(
                    res_norm_square
                        .values()
                        .iter()
                        .zip(ex_res_dot_dists.iter())
                        .map(|(res_norm_square, ex_res_dot)| {
                            -res_norm_square.div_checked(*ex_res_dot).unwrap_or_default()
                        }),
                ),
                _ => {
                    return Err(Error::index(format!(
                        "RQ Transform: distance type {} not supported",
                        self.distance_type
                    )));
                }
            };
            batch = batch
                .try_with_column(
                    crate::vector::bq::storage::rabit_ex_code_field(
                        self.rq.dim(),
                        self.rq.num_bits(),
                    )?
                    .expect("ex-code field should exist for num_bits > 1"),
                    ex_codes,
                )?
                .try_with_column(EX_SCALE_FACTORS_FIELD.clone(), Arc::new(ex_scale_factors))?;
        }

        let batch = batch
            .drop_column(&self.vector_column)?
            .drop_column(CENTROID_DIST_COLUMN)?;
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::AsArray;
    use arrow::datatypes::{Float32Type, UInt8Type};
    use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt32Array};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_linalg::distance::DistanceType;

    use crate::vector::bq::RQRotationType;
    use crate::vector::bq::builder::RabitQuantizer;
    use crate::vector::bq::storage::RABIT_EX_CODE_COLUMN;
    use crate::vector::transform::Transformer;
    use crate::vector::{CENTROID_DIST_COLUMN, PART_ID_COLUMN};

    use super::{ADD_FACTORS_COLUMN, EX_SCALE_FACTORS_COLUMN, RQTransformer};

    #[test]
    fn test_rq_transformer_writes_multi_bit_ex_scale_factors() {
        let rq = RabitQuantizer::new_with_rotation::<Float32Type>(4, 8, RQRotationType::Fast);
        let centroids =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0f32; 8]), 8)
                .unwrap();
        let transformer = RQTransformer::new(rq.clone(), DistanceType::L2, centroids, "vector");

        let residual_vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![
                1.0, -2.0, 3.0, -4.0, 1.5, -2.5, 3.5, -4.5, 0.5, -1.0, 1.5, -2.0, 2.5, -3.0, 3.5,
                -4.0,
            ]),
            8,
        )
        .unwrap();
        let res_norm_square = Float32Array::from(vec![73.0f32, 47.0]);
        let batch = RecordBatch::try_from_iter(vec![
            ("vector", Arc::new(residual_vectors.clone()) as ArrayRef),
            (
                PART_ID_COLUMN,
                Arc::new(UInt32Array::from(vec![0, 0])) as ArrayRef,
            ),
            (
                CENTROID_DIST_COLUMN,
                Arc::new(res_norm_square.clone()) as ArrayRef,
            ),
        ])
        .unwrap();

        let transformed = transformer.transform(&batch).unwrap();
        assert!(transformed.column_by_name(RABIT_EX_CODE_COLUMN).is_some());
        assert_eq!(
            transformed[RABIT_EX_CODE_COLUMN]
                .as_fixed_size_list()
                .value_length(),
            3
        );
        assert!(
            transformed[RABIT_EX_CODE_COLUMN]
                .as_fixed_size_list()
                .values()
                .as_primitive::<UInt8Type>()
                .values()
                .iter()
                .any(|value| *value != 0)
        );
        let expected_ex_dots = rq
            .quantize_split(&residual_vectors)
            .unwrap()
            .ex_res_dot_dists
            .unwrap();
        let ex_scale_factors = transformed[EX_SCALE_FACTORS_COLUMN].as_primitive::<Float32Type>();
        for ((actual, norm), ex_dot) in ex_scale_factors
            .values()
            .iter()
            .zip(res_norm_square.values())
            .zip(expected_ex_dots)
        {
            let expected = if ex_dot == 0.0 {
                0.0
            } else {
                -2.0 * norm / ex_dot
            };
            assert!((actual - expected).abs() < 1e-6);
        }
        assert!(transformed.column_by_name("vector").is_none());
        assert!(transformed.column_by_name(CENTROID_DIST_COLUMN).is_none());
        assert!(transformed.column_by_name(ADD_FACTORS_COLUMN).is_some());
    }
}
