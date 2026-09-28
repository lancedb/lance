// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt::{Debug, Formatter};
use std::sync::{Arc, LazyLock};

use arrow::array::AsArray;
use arrow::datatypes::{Float16Type, Float32Type, Float64Type, UInt32Type};
use arrow_array::{
    Array, ArrowNativeTypeOp, FixedSizeListArray, Float32Array, RecordBatch, UInt8Array,
    UInt32Array,
};
use arrow_schema::DataType;
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_core::{Error, Result};
use lance_linalg::distance::{DistanceType, norm_squared_fsl};
use rayon::prelude::*;
use tracing::instrument;

use crate::vector::bq::builder::RabitQuantizer;
use crate::vector::bq::rabit_ex_bits;
use crate::vector::bq::storage::{
    RABIT_BLOCKED_EX_CODE_COLUMN, RABIT_CODE_COLUMN, RabitQueryEstimator,
};
use crate::vector::bq::sym::sym_bin_code_bytes;
use crate::vector::quantizer::Quantization;
use crate::vector::transform::Transformer;
use crate::vector::{CENTROID_DIST_COLUMN, PART_ID_COLUMN};

// the inner product of quantized vector and the residual vector.
pub const ADD_FACTORS_COLUMN: &str = "__add_factors";
// the inner product of quantized vector and the centroid vector.
pub const SCALE_FACTORS_COLUMN: &str = "__scale_factors";
pub const EX_ADD_FACTORS_COLUMN: &str = "__add_factors_ex";
pub const EX_SCALE_FACTORS_COLUMN: &str = "__scale_factors_ex";
pub const ERROR_FACTORS_COLUMN: &str = "__error_factors";
pub const SYM_BIN_CODES_COLUMN: &str = "__sym_bin_codes";
pub const SYM_RHO_COLUMN: &str = "__sym_rho";
pub const SYM_GAMMA_COLUMN: &str = "__sym_gamma";
pub const SYM_UNORM_COLUMN: &str = "__sym_unorm";
pub const SYM_IP_CENT_COLUMN: &str = "__sym_ip_cent";

pub const SYM_COLUMNS: [&str; 5] = [
    SYM_BIN_CODES_COLUMN,
    SYM_RHO_COLUMN,
    SYM_GAMMA_COLUMN,
    SYM_UNORM_COLUMN,
    SYM_IP_CENT_COLUMN,
];

const RABIT_ERROR_EPSILON: f32 = 1.9;

pub static ADD_FACTORS_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(ADD_FACTORS_COLUMN, arrow_schema::DataType::Float32, true)
});
pub static SCALE_FACTORS_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(SCALE_FACTORS_COLUMN, arrow_schema::DataType::Float32, true)
});
pub static EX_ADD_FACTORS_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(EX_ADD_FACTORS_COLUMN, arrow_schema::DataType::Float32, true)
});
pub static EX_SCALE_FACTORS_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(
        EX_SCALE_FACTORS_COLUMN,
        arrow_schema::DataType::Float32,
        true,
    )
});
pub static ERROR_FACTORS_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(ERROR_FACTORS_COLUMN, arrow_schema::DataType::Float32, true)
});
pub static SYM_RHO_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(SYM_RHO_COLUMN, arrow_schema::DataType::Float32, true)
});
pub static SYM_GAMMA_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(SYM_GAMMA_COLUMN, arrow_schema::DataType::Float32, true)
});
pub static SYM_UNORM_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(SYM_UNORM_COLUMN, arrow_schema::DataType::Float32, true)
});
pub static SYM_IP_CENT_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(SYM_IP_CENT_COLUMN, arrow_schema::DataType::Float32, true)
});

pub fn sym_bin_codes_field(code_dim: usize) -> arrow_schema::Field {
    arrow_schema::Field::new(
        SYM_BIN_CODES_COLUMN,
        DataType::FixedSizeList(
            Arc::new(arrow_schema::Field::new("item", DataType::UInt8, true)),
            crate::vector::bq::sym::sym_bin_code_bytes(code_dim) as i32,
        ),
        true,
    )
}

fn has_sym_columns(batch: &RecordBatch) -> bool {
    SYM_COLUMNS
        .iter()
        .all(|name| batch.column_by_name(name).is_some())
}

pub struct RQTransformer {
    rq: RabitQuantizer,
    distance_type: DistanceType,
    centroids_norm_square: Option<Float32Array>,
    rotated_centroids: Option<Vec<f32>>,
    vector_column: String,
}

impl RQTransformer {
    fn transform_raw_query(
        &self,
        batch: &RecordBatch,
        residual_vectors: &FixedSizeListArray,
        res_norm_square: &Float32Array,
    ) -> Result<RecordBatch> {
        let part_ids = batch
            .column_by_name(PART_ID_COLUMN)
            .ok_or_else(|| Error::index("RQ Transform: partition ids not found"))?
            .as_primitive::<UInt32Type>();
        let rotated_centroids = self.rotated_centroids.as_ref().ok_or_else(|| {
            Error::internal("RabitQ raw-query transformer is missing rotated centroids")
        })?;
        let dim = self.rq.dim();
        let num_centroids = rotated_centroids.len() / dim;
        for &part_id in part_ids.values() {
            if part_id as usize >= num_centroids {
                return Err(Error::invalid_input(format!(
                    "RQ Transform: partition id {part_id} out of range for {num_centroids} rotated centroids"
                )));
            }
        }
        let ex_bits = rabit_ex_bits(self.rq.num_bits())?;
        let num_rows = residual_vectors.len();
        let mut add = vec![0.0; num_rows];
        let mut scale = vec![0.0; num_rows];
        let mut error = vec![0.0; num_rows];
        let ex_rows = if ex_bits == 0 { 0 } else { num_rows };
        let mut ex_add = vec![0.0; ex_rows];
        let mut ex_scale = vec![0.0; ex_rows];
        let extra = if ex_bits == 0 {
            rayon::iter::Either::Left((0..num_rows).into_par_iter().map(|_| None))
        } else {
            rayon::iter::Either::Right(ex_add.par_iter_mut().zip(ex_scale.par_iter_mut()).map(Some))
        };
        // SymRaBitQ columns are accumulated in the same quantization pass.
        let with_sym = self.rq.metadata_ref().uses_sym_columns();
        let sym_rows = if with_sym { num_rows } else { 0 };
        let sym_bin_bytes = sym_bin_code_bytes(dim);
        let mut sym_bins = vec![0u8; sym_rows * sym_bin_bytes];
        let mut sym_rho = vec![0.0; sym_rows];
        let mut sym_gamma = vec![0.0; sym_rows];
        let mut sym_unorm = vec![0.0; sym_rows];
        let mut sym_ip_cent = vec![0.0; sym_rows];
        let centroid_norm_sq: Vec<f32> = rotated_centroids
            .chunks_exact(dim)
            .map(|centroid| centroid.iter().map(|value| value * value).sum())
            .collect();
        let sym_dests = if with_sym {
            rayon::iter::Either::Right(
                sym_bins
                    .par_chunks_mut(sym_bin_bytes)
                    .zip(sym_rho.par_iter_mut())
                    .zip(sym_gamma.par_iter_mut())
                    .zip(sym_unorm.par_iter_mut())
                    .zip(sym_ip_cent.par_iter_mut())
                    .map(Some),
            )
        } else {
            rayon::iter::Either::Left((0..num_rows).into_par_iter().map(|_| None))
        };
        let destinations = (&mut add, &mut scale, &mut error)
            .into_par_iter()
            .zip(extra)
            .zip(sym_dests);
        let num_bits = self.rq.num_bits();
        let codes = self.rq.quantize_with_factors(
            residual_vectors,
            destinations,
            |row, (((add, scale, error), extra), sym)| {
                let part = part_ids.value(row) as usize;
                let start = part * dim;
                RabitRowFactors {
                    distance_type: self.distance_type,
                    norm_square: res_norm_square.value(row),
                    centroid: &rotated_centroids[start..start + dim],
                    ex_bits,
                    ex_code_bias: -((1u32 << ex_bits) as f32 - 0.5),
                    binary_res_dot: 0.0,
                    binary_cent_dot: 0.0,
                    ex_cent_dot: 0.0,
                    residual_centroid_dot: 0.0,
                    add,
                    scale,
                    error,
                    extra,
                    sym: sym.map(|((((bin_dst, rho), gamma), unorm), ip_cent)| {
                        SymRowFactors::new(
                            &rotated_centroids[start..start + dim],
                            centroid_norm_sq[part],
                            num_bits,
                            ex_bits,
                            bin_dst,
                            rho,
                            gamma,
                            unorm,
                            ip_cent,
                        )
                    }),
                }
            },
        )?;
        let mut batch = batch
            .try_with_column(self.rq.field(), codes.binary_codes)?
            .try_with_column(ADD_FACTORS_FIELD.clone(), Arc::new(Float32Array::from(add)))?
            .try_with_column(
                SCALE_FACTORS_FIELD.clone(),
                Arc::new(Float32Array::from(scale)),
            )?
            .try_with_column(
                ERROR_FACTORS_FIELD.clone(),
                Arc::new(Float32Array::from(error)),
            )?;
        if let Some(ex_codes) = codes.ex_codes {
            let field = crate::vector::bq::storage::rabit_ex_code_field(dim, self.rq.num_bits())?
                .ok_or_else(|| Error::internal("Missing RabitQ extra-code field"))?;
            batch = batch
                .try_with_column(field, ex_codes)?
                .try_with_column(
                    EX_ADD_FACTORS_FIELD.clone(),
                    Arc::new(Float32Array::from(ex_add)),
                )?
                .try_with_column(
                    EX_SCALE_FACTORS_FIELD.clone(),
                    Arc::new(Float32Array::from(ex_scale)),
                )?;
        }
        if with_sym {
            let sym_bin_codes = FixedSizeListArray::try_new_from_values(
                UInt8Array::from(sym_bins),
                sym_bin_bytes as i32,
            )?;
            batch = batch
                .try_with_column(sym_bin_codes_field(dim), Arc::new(sym_bin_codes))?
                .try_with_column(SYM_RHO_FIELD.clone(), Arc::new(Float32Array::from(sym_rho)))?
                .try_with_column(
                    SYM_GAMMA_FIELD.clone(),
                    Arc::new(Float32Array::from(sym_gamma)),
                )?
                .try_with_column(
                    SYM_UNORM_FIELD.clone(),
                    Arc::new(Float32Array::from(sym_unorm)),
                )?
                .try_with_column(
                    SYM_IP_CENT_FIELD.clone(),
                    Arc::new(Float32Array::from(sym_ip_cent)),
                )?;
        }
        Ok(batch
            .drop_column(&self.vector_column)?
            .drop_column(CENTROID_DIST_COLUMN)?)
    }

    pub fn new(
        rq: RabitQuantizer,
        distance_type: DistanceType,
        centroids: FixedSizeListArray,
        vector_column: impl Into<String>,
    ) -> Result<Self> {
        // for dot product, the add factor is `1 - v*c + |c|^2`, so we need to compute |c|^2
        let centroids_norm_square = (distance_type == DistanceType::Dot)
            .then(|| Float32Array::from(norm_squared_fsl(&centroids)));
        let rotated_centroids = (rq.metadata_ref().query_estimator
            == RabitQueryEstimator::RawQuery)
            .then(|| rq.rotate_fsl_to_f32(&centroids))
            .transpose()?;

        Ok(Self {
            rq,
            distance_type,
            centroids_norm_square,
            rotated_centroids,
            vector_column: vector_column.into(),
        })
    }
}

#[doc(hidden)]
pub struct RabitRawQueryFactors {
    pub add_factors: Float32Array,
    pub scale_factors: Float32Array,
    pub error_factors: Float32Array,
    pub ex_add_factors: Option<Float32Array>,
    pub ex_scale_factors: Option<Float32Array>,
}

pub(crate) struct RabitRowFactors<'c, 'o> {
    distance_type: DistanceType,
    norm_square: f32,
    centroid: &'c [f32],
    ex_bits: u8,
    ex_code_bias: f32,
    binary_res_dot: f32,
    binary_cent_dot: f32,
    ex_cent_dot: f32,
    residual_centroid_dot: f32,
    add: &'o mut f32,
    scale: &'o mut f32,
    error: &'o mut f32,
    extra: Option<(&'o mut f32, &'o mut f32)>,
    sym: Option<SymRowFactors<'c, 'o>>,
}

/// SymRaBitQ encode factors for one row, accumulated in the same quantization
/// pass as [`RabitRowFactors`] so rotated residuals are never materialized.
pub(crate) struct SymRowFactors<'c, 'o> {
    centroid: &'c [f32],
    centroid_norm_sq: f32,
    ex_bits: u8,
    code_bias: f32,
    bin_dst: &'o mut [u8],
    rho_sq: f32,
    unorm_sq: f32,
    ip: f32,
    ip_cent: f32,
    rho: &'o mut f32,
    gamma: &'o mut f32,
    unorm: &'o mut f32,
    ip_cent_dst: &'o mut f32,
}

impl<'c, 'o> SymRowFactors<'c, 'o> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        centroid: &'c [f32],
        centroid_norm_sq: f32,
        num_bits: u8,
        ex_bits: u8,
        bin_dst: &'o mut [u8],
        rho: &'o mut f32,
        gamma: &'o mut f32,
        unorm: &'o mut f32,
        ip_cent_dst: &'o mut f32,
    ) -> Self {
        Self {
            centroid,
            centroid_norm_sq,
            ex_bits,
            code_bias: ((1u32 << num_bits) - 1) as f32 / 2.0,
            bin_dst,
            rho_sq: 0.0,
            unorm_sq: 0.0,
            ip: 0.0,
            ip_cent: 0.0,
            rho,
            gamma,
            unorm,
            ip_cent_dst,
        }
    }

    fn observe(&mut self, dim: usize, residual: f32, extra: u8) {
        // SymRaBitQ's sign is `residual > 0`, which differs from the binary
        // code's `is_sign_positive` at exactly +0.0 (e.g. padded dims).
        let sign = u32::from(residual > 0.0);
        if sign == 1 {
            // `pack_binary_be` layout: bit-endian 64-dim words, LE bytes.
            let bit = 63 - dim % 64;
            self.bin_dst[dim / 64 * 8 + bit / 8] |= 1u8 << (bit % 8);
        }
        let centered = ((sign << self.ex_bits) + extra as u32) as f32 - self.code_bias;
        self.rho_sq += residual * residual;
        self.unorm_sq += centered * centered;
        self.ip += residual * centered;
        self.ip_cent += residual * self.centroid[dim];
    }

    fn finish(self) {
        let rho = self.rho_sq.sqrt();
        let unorm = self.unorm_sq.sqrt();
        *self.rho = rho;
        *self.ip_cent_dst = self.ip_cent + 0.5 * self.centroid_norm_sq;
        if rho == 0.0 || unorm == 0.0 {
            *self.gamma = 1.0;
            *self.unorm = 1.0;
        } else {
            *self.gamma = self.ip / (rho * unorm);
            *self.unorm = unorm;
        }
    }
}

impl RabitRowFactors<'_, '_> {
    pub(crate) fn observe(&mut self, dim: usize, residual: f32, sign: u8, extra: u8) {
        let centroid = self.centroid[dim];
        let binary_factor = if sign == 1 { 0.5 } else { -0.5 };
        // Keep each accumulator's original dimension order while consuming the
        // code directly from quantization, before scratch is reused.
        self.binary_res_dot += residual * binary_factor;
        self.binary_cent_dot += centroid * binary_factor;
        if self.extra.is_some() {
            let ex_factor =
                (((sign as u32) << self.ex_bits) + extra as u32) as f32 + self.ex_code_bias;
            self.ex_cent_dot += centroid * ex_factor;
        }
        if self.distance_type == DistanceType::Dot {
            self.residual_centroid_dot += residual * centroid;
        }
        if let Some(sym) = &mut self.sym {
            sym.observe(dim, residual, extra);
        }
    }

    pub(crate) fn finish(self, ex_res_dot: f32) {
        let binary_correction =
            factor_ratio(self.norm_square * self.binary_cent_dot, self.binary_res_dot);
        *self.error = error_factor_value(
            self.distance_type,
            self.norm_square,
            self.binary_res_dot,
            self.centroid.len(),
        );
        match self.distance_type {
            DistanceType::L2 => {
                *self.add = self.norm_square + 2.0 * binary_correction;
                *self.scale = factor_ratio(-2.0 * self.norm_square, self.binary_res_dot);
                if let Some((add, scale)) = self.extra {
                    let correction = factor_ratio(self.norm_square * self.ex_cent_dot, ex_res_dot);
                    *add = self.norm_square + 2.0 * correction;
                    *scale = factor_ratio(-2.0 * self.norm_square, ex_res_dot);
                }
            }
            DistanceType::Dot => {
                let dot_base = 1.0 - self.residual_centroid_dot;
                *self.add = dot_base + binary_correction;
                *self.scale = factor_ratio(-self.norm_square, self.binary_res_dot);
                if let Some((add, scale)) = self.extra {
                    let correction = factor_ratio(self.norm_square * self.ex_cent_dot, ex_res_dot);
                    *add = dot_base + correction;
                    *scale = factor_ratio(-self.norm_square, ex_res_dot);
                }
            }
            _ => unreachable!(),
        }
        if let Some(sym) = self.sym {
            sym.finish();
        }
    }
}

#[inline]
fn factor_ratio(numerator: f32, denominator: f32) -> f32 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

#[inline]
fn binary_factor_value(rotated_residual: f32) -> f32 {
    if rotated_residual.is_sign_positive() {
        0.5
    } else {
        -0.5
    }
}

#[inline]
fn error_factor_value(
    distance_type: DistanceType,
    norm_square: f32,
    binary_res_dot: f32,
    code_dim: usize,
) -> f32 {
    if code_dim <= 1 || norm_square <= 0.0 || binary_res_dot == 0.0 {
        return 0.0;
    }

    let code_norm_square = code_dim as f32 * 0.25;
    let alignment = norm_square * code_norm_square / binary_res_dot.powi(2);
    let angular_error = ((alignment - 1.0).max(0.0) / (code_dim as f32 - 1.0)).sqrt();
    let error = norm_square.sqrt() * RABIT_ERROR_EPSILON * angular_error;
    match distance_type {
        DistanceType::L2 => 2.0 * error,
        DistanceType::Dot => error,
        _ => unreachable!(),
    }
}

#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub fn compute_raw_query_factors(
    distance_type: DistanceType,
    res_norm_square: &Float32Array,
    rotated_residuals: &[f32],
    rotated_centroids: &[f32],
    part_ids: &UInt32Array,
    ex_code_values: Option<&[u8]>,
    ex_res_dot_dists: Option<&[f32]>,
    ex_bits: u8,
    code_dim: usize,
) -> Result<RabitRawQueryFactors> {
    if !matches!(distance_type, DistanceType::L2 | DistanceType::Dot) {
        return Err(Error::index(format!(
            "RQ Transform: distance type {} not supported",
            distance_type
        )));
    }

    let num_rows = res_norm_square.len();
    debug_assert_eq!(rotated_residuals.len(), num_rows * code_dim);
    if let Some(ex_code_values) = ex_code_values {
        debug_assert_eq!(ex_code_values.len(), num_rows * code_dim);
    }
    if let Some(ex_res_dot_dists) = ex_res_dot_dists {
        debug_assert_eq!(ex_res_dot_dists.len(), num_rows);
    }

    let has_ex_codes = ex_bits != 0;
    let ex_code_bias = -((1u32 << ex_bits) as f32 - 0.5);
    let mut add_factors = Vec::with_capacity(num_rows);
    let mut scale_factors = Vec::with_capacity(num_rows);
    let mut error_factors = Vec::with_capacity(num_rows);
    let mut ex_add_factors = has_ex_codes.then(|| Vec::with_capacity(num_rows));
    let mut ex_scale_factors = has_ex_codes.then(|| Vec::with_capacity(num_rows));

    for (row_idx, &norm_square) in res_norm_square.values().iter().enumerate() {
        let part_id = part_ids.value(row_idx) as usize;
        let centroid_start = part_id.checked_mul(code_dim).ok_or_else(|| {
            Error::invalid_input(format!(
                "RQ Transform: partition id {} overflows code_dim {}",
                part_id, code_dim
            ))
        })?;
        let centroid_end = centroid_start.checked_add(code_dim).ok_or_else(|| {
            Error::invalid_input(format!(
                "RQ Transform: partition id {} plus code_dim {} overflows",
                part_id, code_dim
            ))
        })?;
        if centroid_end > rotated_centroids.len() {
            return Err(Error::invalid_input(format!(
                "RQ Transform: partition id {} out of range for {} rotated centroids",
                part_id,
                rotated_centroids.len() / code_dim
            )));
        }

        let row_start = row_idx * code_dim;
        let row_end = row_start + code_dim;
        let residual = &rotated_residuals[row_start..row_end];
        let centroid = &rotated_centroids[centroid_start..centroid_end];
        let ex_values = ex_code_values.map(|values| &values[row_start..row_end]);

        let mut binary_res_dot = 0.0f32;
        let mut binary_cent_dot = 0.0f32;
        let mut ex_cent_dot = 0.0f32;
        let mut residual_centroid_dot = 0.0f32;
        for (dim_idx, (&residual_value, &centroid_value)) in
            residual.iter().zip(centroid.iter()).enumerate()
        {
            let residual_value: f32 = residual_value;
            let centroid_value: f32 = centroid_value;
            let binary_code = if residual_value.is_sign_positive() {
                1u32
            } else {
                0u32
            };
            let binary_factor = binary_factor_value(residual_value);

            binary_res_dot += residual_value * binary_factor;
            binary_cent_dot += centroid_value * binary_factor;
            if let Some(ex_values) = ex_values {
                let ex_code_value = ex_values[dim_idx];
                let ex_factor =
                    ((binary_code << ex_bits) + ex_code_value as u32) as f32 + ex_code_bias;
                ex_cent_dot += centroid_value * ex_factor;
            }
            residual_centroid_dot += residual_value * centroid_value;
        }

        let binary_correction = factor_ratio(norm_square * binary_cent_dot, binary_res_dot);
        let ex_res_dot = ex_res_dot_dists
            .map(|values| values[row_idx])
            .unwrap_or_default();
        let ex_correction = factor_ratio(norm_square * ex_cent_dot, ex_res_dot);
        error_factors.push(error_factor_value(
            distance_type,
            norm_square,
            binary_res_dot,
            code_dim,
        ));

        match distance_type {
            DistanceType::L2 => {
                add_factors.push(norm_square + 2.0 * binary_correction);
                scale_factors.push(factor_ratio(-2.0 * norm_square, binary_res_dot));
                if let Some(ex_add_factors) = ex_add_factors.as_mut() {
                    ex_add_factors.push(norm_square + 2.0 * ex_correction);
                }
                if let Some(ex_scale_factors) = ex_scale_factors.as_mut() {
                    ex_scale_factors.push(factor_ratio(-2.0 * norm_square, ex_res_dot));
                }
            }
            DistanceType::Dot => {
                let dot_base = 1.0 - residual_centroid_dot;
                add_factors.push(dot_base + binary_correction);
                scale_factors.push(factor_ratio(-norm_square, binary_res_dot));
                if let Some(ex_add_factors) = ex_add_factors.as_mut() {
                    ex_add_factors.push(dot_base + ex_correction);
                }
                if let Some(ex_scale_factors) = ex_scale_factors.as_mut() {
                    ex_scale_factors.push(factor_ratio(-norm_square, ex_res_dot));
                }
            }
            _ => unreachable!(),
        }
    }

    Ok(RabitRawQueryFactors {
        add_factors: Float32Array::from(add_factors),
        scale_factors: Float32Array::from(scale_factors),
        error_factors: Float32Array::from(error_factors),
        ex_add_factors: ex_add_factors.map(Float32Array::from),
        ex_scale_factors: ex_scale_factors.map(Float32Array::from),
    })
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
            || (batch.column_by_name(RABIT_BLOCKED_EX_CODE_COLUMN).is_some()
                && batch.column_by_name(EX_ADD_FACTORS_COLUMN).is_some()
                && batch.column_by_name(EX_SCALE_FACTORS_COLUMN).is_some());
        if batch.column_by_name(RABIT_CODE_COLUMN).is_some() && has_split_codes {
            if !self.rq.metadata_ref().uses_sym_columns() || has_sym_columns(batch) {
                return Ok(batch.clone());
            }
            return Err(Error::index(
                "RQ Transform: batch has split codes but is missing SymRaBitQ columns".to_string(),
            ));
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

        if self.rq.metadata_ref().query_estimator == RabitQueryEstimator::RawQuery {
            return self.transform_raw_query(batch, residual_vectors, &res_norm_square);
        }

        let rq_codes = self.rq.quantize_split(residual_vectors)?;
        let codes_fsl = rq_codes.binary_codes.as_fixed_size_list();
        debug_assert_eq!(codes_fsl.len(), batch.num_rows());

        let mut batch = batch.try_with_column(self.rq.field(), rq_codes.binary_codes)?;
        if self.rq.metadata_ref().query_estimator == RabitQueryEstimator::ResidualQuery {
            // Preserve the released residual-query estimator and factor layout.
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
                            .map(|(dist_v_c, centroids_norm_square)| {
                                dist_v_c + centroids_norm_square
                            }),
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

            batch = batch
                .try_with_column(ADD_FACTORS_FIELD.clone(), Arc::new(add_factors))?
                .try_with_column(SCALE_FACTORS_FIELD.clone(), Arc::new(scale_factors))?;
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
    use arrow_array::{
        Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt32Array,
    };
    use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
    use lance_linalg::distance::DistanceType;
    use rstest::rstest;

    use crate::vector::bq::RQRotationType;
    use crate::vector::bq::builder::RabitQuantizer;
    use crate::vector::bq::ex_dot::blocked_ex_code_bytes;
    use crate::vector::bq::storage::{RABIT_BLOCKED_EX_CODE_COLUMN, RABIT_CODE_COLUMN};
    use crate::vector::bq::sym::{compute_sym_factors, pack_binary_be, sym_bin_code_bytes};
    use crate::vector::transform::Transformer;
    use crate::vector::{CENTROID_DIST_COLUMN, PART_ID_COLUMN};

    use super::{
        ADD_FACTORS_COLUMN, ERROR_FACTORS_COLUMN, EX_ADD_FACTORS_COLUMN, EX_SCALE_FACTORS_COLUMN,
        RQTransformer, SYM_BIN_CODES_COLUMN, SYM_COLUMNS, SYM_GAMMA_COLUMN, SYM_IP_CENT_COLUMN,
        SYM_RHO_COLUMN, SYM_UNORM_COLUMN, compute_raw_query_factors, error_factor_value,
    };

    #[rstest]
    #[case::fast_small(RQRotationType::Fast, 8, 16)]
    #[case::fast_parallel(RQRotationType::Fast, 768, 129)]
    #[case::matrix(RQRotationType::Matrix, 8, 16)]
    fn test_fused_quantization_matches_split_pipeline(
        #[case] rotation: RQRotationType,
        #[case] dim: usize,
        #[case] num_rows: usize,
        #[values(1, 3, 5, 9)] bits: u8,
        #[values(DistanceType::L2, DistanceType::Dot)] distance: DistanceType,
        #[values(
            arrow_schema::DataType::Float16,
            arrow_schema::DataType::Float32,
            arrow_schema::DataType::Float64
        )]
        value_type: arrow_schema::DataType,
    ) {
        let rq = match value_type {
            arrow_schema::DataType::Float16 => RabitQuantizer::new_with_rotation::<
                arrow::datatypes::Float16Type,
            >(bits, dim as i32, rotation),
            arrow_schema::DataType::Float64 => RabitQuantizer::new_with_rotation::<
                arrow::datatypes::Float64Type,
            >(bits, dim as i32, rotation),
            _ => RabitQuantizer::new_with_rotation::<Float32Type>(bits, dim as i32, rotation),
        };
        let centroid_values = Float32Array::from(
            (0..3 * dim)
                .map(|i| ((i * 17 % 103) as f32 - 51.0) / 64.0)
                .collect::<Vec<_>>(),
        );
        let centroids = FixedSizeListArray::try_new_from_values(
            arrow::compute::cast(&centroid_values, &value_type).unwrap(),
            dim as i32,
        )
        .unwrap();
        let transformer = RQTransformer::new(rq.clone(), distance, centroids, "vector").unwrap();
        let values = Float32Array::from(
            (0..(num_rows + 2) * dim)
                .map(|i| {
                    if i / dim == 2 {
                        0.0
                    } else if i / dim == 3 {
                        ((i * 31 % 107) as f32 - 53.0) * 1.0e-20
                    } else {
                        ((i * 31 % 107) as f32 - 53.0) / 64.0
                    }
                })
                .collect::<Vec<_>>(),
        );
        let values = arrow::compute::cast(&values, &value_type).unwrap();
        let residuals = FixedSizeListArray::try_new_from_values(values, dim as i32)
            .unwrap()
            .slice(1, num_rows);
        let norm = Float32Array::from(lance_linalg::distance::norm_squared_fsl(&residuals));
        let parts = UInt32Array::from((0..num_rows).map(|i| (i % 3) as u32).collect::<Vec<_>>());
        let batch = RecordBatch::try_from_iter(vec![
            ("vector", Arc::new(residuals.clone()) as ArrayRef),
            (CENTROID_DIST_COLUMN, Arc::new(norm.clone()) as ArrayRef),
            (PART_ID_COLUMN, Arc::new(parts.clone()) as ArrayRef),
        ])
        .unwrap();
        let split = rq.quantize_split(&residuals).unwrap();
        let expected = compute_raw_query_factors(
            distance,
            &norm,
            split.rotated_residuals.as_ref().unwrap(),
            transformer.rotated_centroids.as_ref().unwrap(),
            &parts,
            split.ex_code_values.as_deref(),
            split.ex_res_dot_dists.as_deref(),
            bits - 1,
            dim,
        )
        .unwrap();
        let actual = transformer.transform(&batch).unwrap();
        assert_eq!(
            actual
                .column_by_name(crate::vector::bq::storage::RABIT_CODE_COLUMN)
                .unwrap()
                .to_data(),
            split.binary_codes.to_data()
        );
        if let Some(ex_codes) = split.ex_codes {
            assert_eq!(
                actual
                    .column_by_name(RABIT_BLOCKED_EX_CODE_COLUMN)
                    .unwrap()
                    .to_data(),
                ex_codes.to_data()
            );
        }
        let mut expected_columns = vec![
            (ADD_FACTORS_COLUMN, expected.add_factors),
            (super::SCALE_FACTORS_COLUMN, expected.scale_factors),
            (ERROR_FACTORS_COLUMN, expected.error_factors),
        ];
        if let (Some(add), Some(scale)) = (expected.ex_add_factors, expected.ex_scale_factors) {
            expected_columns.push((EX_ADD_FACTORS_COLUMN, add));
            expected_columns.push((EX_SCALE_FACTORS_COLUMN, scale));
        }
        for (name, expected) in expected_columns {
            let actual = actual
                .column_by_name(name)
                .unwrap()
                .as_primitive::<Float32Type>();
            assert_eq!(
                actual
                    .values()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .values()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "{name}"
            );
        }
        assert_eq!(
            transformer
                .transform(&batch.slice(0, 0))
                .unwrap()
                .num_rows(),
            0
        );
        let bad_parts = batch
            .drop_column(PART_ID_COLUMN)
            .unwrap()
            .try_with_column(
                arrow_schema::Field::new(PART_ID_COLUMN, arrow_schema::DataType::UInt32, false),
                Arc::new(UInt32Array::from(vec![u32::MAX; num_rows])),
            )
            .unwrap();
        let error = transformer.transform(&bad_parts).unwrap_err();
        assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
        assert!(error.to_string().contains("out of range"));
    }

    fn pack_sign_bits_le(rotated: &[f32]) -> Vec<u8> {
        let mut codes = vec![0u8; rotated.len().div_ceil(8)];
        for (bit_idx, value) in rotated.iter().enumerate() {
            if value.is_sign_positive() {
                codes[bit_idx / 8] |= 1u8 << (bit_idx % 8);
            }
        }
        codes
    }

    fn make_residual_batch(
        residuals: Vec<f32>,
        dim: i32,
        part_ids: Vec<u32>,
        centroid_dists: Vec<f32>,
    ) -> RecordBatch {
        let residual_vectors =
            FixedSizeListArray::try_new_from_values(Float32Array::from(residuals), dim).unwrap();
        RecordBatch::try_from_iter(vec![
            ("vector", Arc::new(residual_vectors) as ArrayRef),
            (
                PART_ID_COLUMN,
                Arc::new(UInt32Array::from(part_ids)) as ArrayRef,
            ),
            (
                CENTROID_DIST_COLUMN,
                Arc::new(Float32Array::from(centroid_dists)) as ArrayRef,
            ),
        ])
        .unwrap()
    }

    #[test]
    fn test_rq_transformer_writes_multi_bit_ex_factors() {
        let rq = RabitQuantizer::new_with_rotation::<Float32Type>(4, 8, RQRotationType::Fast);
        let centroids =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0f32; 8]), 8)
                .unwrap();
        let transformer =
            RQTransformer::new(rq.clone(), DistanceType::L2, centroids, "vector").unwrap();

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
        assert!(
            transformed
                .column_by_name(RABIT_BLOCKED_EX_CODE_COLUMN)
                .is_some()
        );
        assert_eq!(
            transformed[RABIT_BLOCKED_EX_CODE_COLUMN]
                .as_fixed_size_list()
                .value_length(),
            blocked_ex_code_bytes(8, 3) as i32
        );
        assert!(
            transformed[RABIT_BLOCKED_EX_CODE_COLUMN]
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
        let ex_add_factors = transformed[EX_ADD_FACTORS_COLUMN].as_primitive::<Float32Type>();
        assert_eq!(ex_add_factors.values(), res_norm_square.values());
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
        assert!(transformed.column_by_name(ERROR_FACTORS_COLUMN).is_some());
    }

    #[test]
    fn test_rq_transformer_omits_sym_columns_for_one_bit() {
        let rq = RabitQuantizer::new_with_rotation::<Float32Type>(1, 8, RQRotationType::Fast);
        let centroids =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.25f32; 8]), 8)
                .unwrap();
        let transformer = RQTransformer::new(rq, DistanceType::L2, centroids, "vector").unwrap();
        let batch = make_residual_batch(
            vec![1.0, -2.0, 3.0, -4.0, 0.0, -0.0, 1.5, -2.5],
            8,
            vec![0],
            vec![30.5],
        );
        let transformed = transformer.transform(&batch).unwrap();
        for name in SYM_COLUMNS {
            assert!(
                transformed.column_by_name(name).is_none(),
                "1-bit RQ must not write {name}"
            );
        }
    }

    #[test]
    fn test_rq_transformer_writes_sym_columns_for_multi_bit() {
        let dim = 8;
        let num_bits = 4u8;
        let mut rq =
            RabitQuantizer::new_with_rotation::<Float32Type>(num_bits, dim, RQRotationType::Fast);
        rq.metadata.with_sym_columns = true;
        let centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.25, -0.5, 0.75, -1.0, 0.0, 1.25, -0.25, 0.5]),
            dim,
        )
        .unwrap();
        let residuals = vec![1.0, -2.0, 3.0, -4.0, 0.0, -0.0, 1.5, -2.5];
        let residual_fsl =
            FixedSizeListArray::try_new_from_values(Float32Array::from(residuals.clone()), dim)
                .unwrap();
        let transformer =
            RQTransformer::new(rq.clone(), DistanceType::L2, centroids.clone(), "vector").unwrap();
        let batch = make_residual_batch(residuals, dim, vec![0], vec![30.5]);
        let transformed = transformer.transform(&batch).unwrap();

        for name in SYM_COLUMNS {
            assert!(
                transformed.column_by_name(name).is_some(),
                "multi-bit RQ must write {name}"
            );
        }

        let split = rq.quantize_split(&residual_fsl).unwrap();
        let rotated = split.rotated_residuals.as_ref().unwrap();
        let rotated_centroid = rq.rotate_fsl_to_f32(&centroids).unwrap();
        let ex = split
            .ex_codes
            .as_ref()
            .unwrap()
            .as_fixed_size_list()
            .value(0)
            .as_primitive::<UInt8Type>()
            .values()
            .to_vec();
        let mut expected_be = vec![0u8; sym_bin_code_bytes(dim as usize)];
        pack_binary_be(rotated, &mut expected_be);
        let expected_factors =
            compute_sym_factors(rotated, &rotated_centroid, &expected_be, &ex, num_bits);

        let sym_bin = transformed[SYM_BIN_CODES_COLUMN].as_fixed_size_list();
        assert_eq!(sym_bin.value_length(), expected_be.len() as i32);
        assert_eq!(
            sym_bin
                .value(0)
                .as_primitive::<UInt8Type>()
                .values()
                .as_ref(),
            expected_be.as_slice()
        );

        let le_codes = transformed[RABIT_CODE_COLUMN].as_fixed_size_list();
        assert_eq!(le_codes.value_length(), (dim as usize / 8) as i32);
        assert_eq!(
            le_codes
                .value(0)
                .as_primitive::<UInt8Type>()
                .values()
                .as_ref(),
            pack_sign_bits_le(rotated).as_slice()
        );

        let rho = transformed[SYM_RHO_COLUMN]
            .as_primitive::<Float32Type>()
            .value(0);
        let gamma = transformed[SYM_GAMMA_COLUMN]
            .as_primitive::<Float32Type>()
            .value(0);
        let unorm = transformed[SYM_UNORM_COLUMN]
            .as_primitive::<Float32Type>()
            .value(0);
        let ip_cent = transformed[SYM_IP_CENT_COLUMN]
            .as_primitive::<Float32Type>()
            .value(0);
        for (name, value) in [
            ("rho", rho),
            ("gamma", gamma),
            ("unorm", unorm),
            ("ip_cent", ip_cent),
        ] {
            assert!(value.is_finite(), "{name} must be finite");
        }
        assert!((rho - expected_factors.rho).abs() <= 1e-5 * expected_factors.rho.abs().max(1.0));
        assert!(
            (gamma - expected_factors.gamma).abs() <= 1e-5 * expected_factors.gamma.abs().max(1.0)
        );
        assert!(
            (unorm - expected_factors.unorm).abs() <= 1e-5 * expected_factors.unorm.abs().max(1.0)
        );
        assert!(
            (ip_cent - expected_factors.ip_cent).abs()
                <= 1e-5 * expected_factors.ip_cent.abs().max(1.0)
        );
    }

    #[test]
    fn test_rq_transformer_writes_sym_columns_for_one_bit() {
        let dim = 8;
        let num_bits = 1u8;
        let mut rq =
            RabitQuantizer::new_with_rotation::<Float32Type>(num_bits, dim, RQRotationType::Fast);
        rq.metadata.with_sym_columns = true;
        let centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.25, -0.5, 0.75, -1.0, 0.0, 1.25, -0.25, 0.5]),
            dim,
        )
        .unwrap();
        let residuals = vec![1.0, -2.0, 3.0, -4.0, 0.0, -0.0, 1.5, -2.5];
        let residual_fsl =
            FixedSizeListArray::try_new_from_values(Float32Array::from(residuals.clone()), dim)
                .unwrap();
        let transformer =
            RQTransformer::new(rq.clone(), DistanceType::L2, centroids, "vector").unwrap();
        let batch = make_residual_batch(residuals, dim, vec![0], vec![30.5]);
        let transformed = transformer.transform(&batch).unwrap();

        for name in SYM_COLUMNS {
            assert!(
                transformed.column_by_name(name).is_some(),
                "1-bit sym RQ must write {name}"
            );
        }
        // 1-bit has no ex codes at all.
        assert!(
            transformed
                .column_by_name(RABIT_BLOCKED_EX_CODE_COLUMN)
                .is_none()
        );

        let split = rq.quantize_split(&residual_fsl).unwrap();
        let rotated = split.rotated_residuals.as_ref().unwrap();
        let mut expected_be = vec![0u8; sym_bin_code_bytes(dim as usize)];
        pack_binary_be(rotated, &mut expected_be);
        let sym_bin = transformed[SYM_BIN_CODES_COLUMN].as_fixed_size_list();
        assert_eq!(
            sym_bin
                .value(0)
                .as_primitive::<UInt8Type>()
                .values()
                .as_ref(),
            expected_be.as_slice()
        );

        // 1-bit centered codes are all ±0.5, so ‖û‖ = sqrt(dim) / 2.
        let unorm = transformed[SYM_UNORM_COLUMN]
            .as_primitive::<Float32Type>()
            .value(0);
        let want_unorm = (dim as f32).sqrt() / 2.0;
        assert!(
            (unorm - want_unorm).abs() <= 1e-5 * want_unorm,
            "1-bit unorm: {unorm} != {want_unorm}"
        );
    }

    #[test]
    fn test_rq_transformer_rejects_split_batch_missing_sym_columns() {
        let dim = 8;
        let mut rq = RabitQuantizer::new_with_rotation::<Float32Type>(4, dim, RQRotationType::Fast);
        rq.metadata.with_sym_columns = true;
        let centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.25, -0.5, 0.75, -1.0, 0.0, 1.25, -0.25, 0.5]),
            dim,
        )
        .unwrap();
        let transformer = RQTransformer::new(rq, DistanceType::L2, centroids, "vector").unwrap();
        let batch = make_residual_batch(
            vec![1.0, -2.0, 3.0, -4.0, 0.0, -0.0, 1.5, -2.5],
            dim,
            vec![0],
            vec![30.5],
        );
        let encoded = transformer.transform(&batch).unwrap();
        let mut missing_sym = encoded;
        for name in SYM_COLUMNS {
            missing_sym = missing_sym.drop_column(name).unwrap();
        }
        // Residual still present must not fall through into a re-encode:
        // try_with_column would then reject the existing _rabit_codes.
        let residual = batch.column_by_name("vector").unwrap().clone();
        let with_vector = missing_sym
            .try_with_column(
                arrow_schema::Field::new("vector", residual.data_type().clone(), true),
                residual,
            )
            .unwrap();
        let err = transformer.transform(&with_vector).unwrap_err();
        assert!(
            err.to_string().contains("missing SymRaBitQ columns"),
            "{err}"
        );
    }

    #[test]
    fn test_rq_transformer_caches_rotated_centroids_for_raw_query() {
        let centroids =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0f32; 8]), 8)
                .unwrap();
        let raw_query_rq =
            RabitQuantizer::new_with_rotation::<Float32Type>(1, 8, RQRotationType::Fast);
        let raw_query_transformer =
            RQTransformer::new(raw_query_rq, DistanceType::L2, centroids.clone(), "vector")
                .unwrap();
        assert_eq!(
            raw_query_transformer
                .rotated_centroids
                .as_ref()
                .unwrap()
                .len(),
            8
        );

        let multi_bit_rq =
            RabitQuantizer::new_with_rotation::<Float32Type>(4, 8, RQRotationType::Fast);
        let multi_bit_transformer =
            RQTransformer::new(multi_bit_rq, DistanceType::L2, centroids, "vector").unwrap();
        assert_eq!(
            multi_bit_transformer
                .rotated_centroids
                .as_ref()
                .unwrap()
                .len(),
            8
        );
    }

    #[test]
    fn test_raw_query_factors_match_reference_l2_formula() {
        let res_norm_square = Float32Array::from(vec![5.0f32, 7.0]);
        let rotated_residuals = vec![2.0, -1.0, 0.0, 0.0];
        let rotated_centroids = vec![3.0, 4.0];
        let part_ids = UInt32Array::from(vec![0, 0]);
        let ex_code_values = vec![1, 0, 0, 0];
        let ex_res_dot_dists = vec![4.5, 0.0];

        let factors = compute_raw_query_factors(
            DistanceType::L2,
            &res_norm_square,
            &rotated_residuals,
            &rotated_centroids,
            &part_ids,
            Some(&ex_code_values),
            Some(&ex_res_dot_dists),
            1,
            2,
        )
        .unwrap();

        assert!((factors.add_factors.value(0) - 1.6666667).abs() < 1e-5);
        assert!((factors.scale_factors.value(0) + 6.6666665).abs() < 1e-5);
        let expected_error = error_factor_value(DistanceType::L2, 5.0, 1.5, 2);
        assert!((factors.error_factors.value(0) - expected_error).abs() < 1e-5);
        let ex_add_factors = factors.ex_add_factors.unwrap();
        let ex_scale_factors = factors.ex_scale_factors.unwrap();
        assert!((ex_add_factors.value(0) - 1.6666667).abs() < 1e-5);
        assert!((ex_scale_factors.value(0) + 2.2222223).abs() < 1e-5);
        assert_eq!(factors.add_factors.value(1), 7.0);
        assert_eq!(factors.scale_factors.value(1), 0.0);
        assert_eq!(factors.error_factors.value(1), 0.0);
        assert_eq!(ex_add_factors.value(1), 7.0);
        assert_eq!(ex_scale_factors.value(1), 0.0);
    }

    #[test]
    fn test_raw_query_factors_match_reference_dot_formula() {
        let res_norm_square = Float32Array::from(vec![5.0f32]);
        let rotated_residuals = vec![2.0, -1.0];
        let rotated_centroids = vec![3.0, 4.0];
        let part_ids = UInt32Array::from(vec![0]);
        let ex_code_values = vec![1, 0];
        let ex_res_dot_dists = vec![4.5];

        let factors = compute_raw_query_factors(
            DistanceType::Dot,
            &res_norm_square,
            &rotated_residuals,
            &rotated_centroids,
            &part_ids,
            Some(&ex_code_values),
            Some(&ex_res_dot_dists),
            1,
            2,
        )
        .unwrap();

        assert!((factors.add_factors.value(0) + 2.6666667).abs() < 1e-5);
        assert!((factors.scale_factors.value(0) + 3.3333333).abs() < 1e-5);
        let expected_error = error_factor_value(DistanceType::Dot, 5.0, 1.5, 2);
        assert!((factors.error_factors.value(0) - expected_error).abs() < 1e-5);
        let ex_add_factors = factors.ex_add_factors.unwrap();
        let ex_scale_factors = factors.ex_scale_factors.unwrap();
        assert!((ex_add_factors.value(0) + 2.6666667).abs() < 1e-5);
        assert!((ex_scale_factors.value(0) + 1.1111112).abs() < 1e-5);
    }
}
