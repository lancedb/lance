// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Distributed KMeans primitives.
//!
//! Pure-function, Arrow-native primitives that let an external scheduler
//! (Spark, Ray, custom RPC) drive distributed IVF centroid training. See
//! `docs/superpowers/specs/2026-06-10-distributed-centroid-training-abstraction-design.md`.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;

use arrow_array::{
    Array, ArrowPrimitiveType, FixedSizeListArray, Float16Array, Float32Array, Float64Array,
    PrimitiveArray, RecordBatch, UInt32Array, UInt64Array,
    builder::{FixedSizeListBuilder, Float64Builder},
    cast::AsArray,
    types::{Float16Type, Float32Type, Float64Type, Int8Type},
};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use half::f16;
use lance_arrow::FixedSizeListArrayExt;
use lance_linalg::distance::DistanceType;

use crate::vector::kmeans::{KMeans, KMeansParams, train_kmeans};
use crate::{Error, Result};

pub const PARTIAL_STATS_VERSION: &str = "1";
pub const META_VERSION: &str = "lance.partial_stats.version";
pub const META_K: &str = "lance.partial_stats.k";
pub const META_DIM: &str = "lance.partial_stats.dim";
pub const META_DT: &str = "lance.partial_stats.distance_type";
pub const META_FP: &str = "lance.partial_stats.centroids_fingerprint";

pub const COL_CLUSTER_ID: &str = "cluster_id";
pub const COL_COUNT: &str = "count";
pub const COL_SUM: &str = "sum";
pub const COL_SQ_NORM_SUM: &str = "sq_norm_sum";
pub const COL_LOSS: &str = "loss";
pub const COL_RADIUS: &str = "radius";

/// One worker's contribution to a kmeans round.
///
/// Wraps a fixed-schema [`RecordBatch`] of `k` rows so the wire format is just
/// Arrow IPC. See module docs for the schema contract.
#[derive(Debug, Clone)]
pub struct PartialStats {
    pub(crate) batch: RecordBatch,
}

fn dt_metadata_value(dt: DistanceType) -> &'static str {
    match dt {
        DistanceType::L2 => "l2",
        DistanceType::Dot => "dot",
        DistanceType::Cosine => "cosine",
        DistanceType::Hamming => "hamming",
    }
}

fn fingerprint_to_hex(fp: &[u8; 8]) -> String {
    let mut s = String::with_capacity(16);
    for byte in fp {
        // Write into a String never errors.
        let _ = write!(s, "{:02x}", byte);
    }
    s
}

fn fingerprint_from_hex(s: &str) -> Result<[u8; 8]> {
    if s.len() != 16 {
        return Err(Error::index(format!(
            "invalid fingerprint hex length: {}",
            s.len()
        )));
    }
    let mut out = [0u8; 8];
    for i in 0..8 {
        out[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
            .map_err(|e| Error::index(format!("invalid fingerprint hex: {}", e)))?;
    }
    Ok(out)
}

pub(crate) fn build_schema(k: usize, dim: usize, dt: DistanceType, fp: [u8; 8]) -> SchemaRef {
    // The inner Float64 field is nullable so it round-trips through the default
    // [`FixedSizeListBuilder`] (whose primitive builder produces a nullable
    // child by default).
    let sum_field = DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float64, true)),
        dim as i32,
    );
    let mut metadata = HashMap::new();
    metadata.insert(META_VERSION.into(), PARTIAL_STATS_VERSION.into());
    metadata.insert(META_K.into(), k.to_string());
    metadata.insert(META_DIM.into(), dim.to_string());
    metadata.insert(META_DT.into(), dt_metadata_value(dt).into());
    metadata.insert(META_FP.into(), fingerprint_to_hex(&fp));
    Arc::new(
        Schema::new(vec![
            Field::new(COL_CLUSTER_ID, DataType::UInt32, false),
            Field::new(COL_COUNT, DataType::UInt64, false),
            Field::new(COL_SUM, sum_field, false),
            Field::new(COL_SQ_NORM_SUM, DataType::Float64, false),
            Field::new(COL_LOSS, DataType::Float64, false),
            Field::new(COL_RADIUS, DataType::Float32, false),
        ])
        .with_metadata(metadata),
    )
}

impl PartialStats {
    /// Build a zero-filled stats buffer for `k` clusters of `dim` dimension.
    /// `fingerprint` identifies the centroids these stats are computed against.
    pub fn empty(k: usize, dim: usize, dt: DistanceType, fingerprint: [u8; 8]) -> Self {
        let schema = build_schema(k, dim, dt, fingerprint);

        let cluster_id: UInt32Array = (0..k as u32).collect();
        let count = UInt64Array::from(vec![0u64; k]);
        let sq_norm = Float64Array::from(vec![0.0f64; k]);
        let loss = Float64Array::from(vec![0.0f64; k]);
        let radius = Float32Array::from(vec![0.0f32; k]);

        let mut sum_builder = FixedSizeListBuilder::new(Float64Builder::new(), dim as i32);
        for _ in 0..k {
            for _ in 0..dim {
                sum_builder.values().append_value(0.0);
            }
            sum_builder.append(true);
        }
        let sum = sum_builder.finish();

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(cluster_id),
                Arc::new(count),
                Arc::new(sum),
                Arc::new(sq_norm),
                Arc::new(loss),
                Arc::new(radius),
            ],
        )
        .expect("partial stats schema is internally consistent");
        Self { batch }
    }

    /// Validate and adopt an externally-built RecordBatch.
    pub fn from_record_batch(batch: RecordBatch) -> Result<Self> {
        let schema = batch.schema();
        let md = schema.metadata();
        let version = md.get(META_VERSION).map(String::as_str).unwrap_or("");
        if version != PARTIAL_STATS_VERSION {
            return Err(Error::index(format!(
                "PartialStats version mismatch: got {:?}, expected {}",
                version, PARTIAL_STATS_VERSION
            )));
        }
        // Spot-check schema columns; full structural check happens in merge/finalize.
        for (idx, name) in [
            COL_CLUSTER_ID,
            COL_COUNT,
            COL_SUM,
            COL_SQ_NORM_SUM,
            COL_LOSS,
            COL_RADIUS,
        ]
        .iter()
        .enumerate()
        {
            let field = schema.field(idx);
            if field.name() != *name {
                return Err(Error::index(format!(
                    "PartialStats schema mismatch at col {}: got {}, expected {}",
                    idx,
                    field.name(),
                    name
                )));
            }
        }
        Ok(Self { batch })
    }

    pub fn into_record_batch(self) -> RecordBatch {
        self.batch
    }

    pub fn record_batch(&self) -> &RecordBatch {
        &self.batch
    }

    pub fn k(&self) -> usize {
        self.batch.num_rows()
    }

    pub fn dim(&self) -> usize {
        match self.batch.schema().field(2).data_type() {
            DataType::FixedSizeList(_, n) => *n as usize,
            _ => 0,
        }
    }

    pub fn distance_type(&self) -> DistanceType {
        match self
            .batch
            .schema()
            .metadata()
            .get(META_DT)
            .map(String::as_str)
            .unwrap_or("l2")
        {
            "dot" => DistanceType::Dot,
            "cosine" => DistanceType::Cosine,
            "hamming" => DistanceType::Hamming,
            _ => DistanceType::L2,
        }
    }

    pub fn centroids_fingerprint(&self) -> [u8; 8] {
        self.batch
            .schema()
            .metadata()
            .get(META_FP)
            .and_then(|s| fingerprint_from_hex(s).ok())
            .unwrap_or([0u8; 8])
    }

    pub fn total_count(&self) -> u64 {
        let counts = self
            .batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("count column is UInt64");
        counts.values().iter().sum()
    }

    pub fn total_loss(&self) -> f64 {
        let losses = self
            .batch
            .column(4)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("loss column is Float64");
        losses.values().iter().sum()
    }
}

/// 8-byte SHA-256 prefix over the raw bytes of a centroids buffer.
///
/// Used as `lance.partial_stats.centroids_fingerprint` so reducers can detect
/// mixing partials from different training rounds.
pub(crate) fn compute_centroids_fingerprint(centroids: &FixedSizeListArray) -> [u8; 8] {
    use sha2::{Digest, Sha256};
    let buffers = centroids.values().to_data().buffers().to_vec();
    let mut hasher = Sha256::new();
    for buf in buffers {
        hasher.update(buf.as_slice());
    }
    let digest = hasher.finalize();
    let mut out = [0u8; 8];
    out.copy_from_slice(&digest[..8]);
    out
}

fn arrow_error_to_lance(e: ArrowError) -> Error {
    Error::index(e.to_string())
}

/// E-step: assign every row of `data` to its closest centroid and accumulate
/// `count`/`sum`/`sq_norm_sum`/`loss`/`radius` per cluster.
///
/// `data` and `centroids` must share dtype, except that `Int8` data is up-cast
/// to `Float32` before assignment, matching `train_ivf_kmeans_step` in
/// `rust/lance/src/index/vector/ivf.rs`.
///
/// Hamming / `UInt8` is intentionally not supported here.
pub fn compute_partial_stats(
    centroids: &FixedSizeListArray,
    data: &FixedSizeListArray,
    distance_type: DistanceType,
) -> Result<PartialStats> {
    if matches!(distance_type, DistanceType::Hamming) {
        return Err(Error::index(
            "distributed Hamming kmeans is not supported in v1",
        ));
    }
    if centroids.value_length() != data.value_length() {
        return Err(Error::index(format!(
            "centroids dim {} does not match data dim {}",
            centroids.value_length(),
            data.value_length()
        )));
    }

    let dim = centroids.value_length() as usize;
    let k = centroids.len();
    let fingerprint = compute_centroids_fingerprint(centroids);
    let stats_empty = PartialStats::empty(k, dim, distance_type, fingerprint);
    if data.len() == 0 {
        return Ok(stats_empty);
    }

    // Up-cast Int8 -> Float32 (matches `train_ivf_kmeans_step`).
    let data_for_assignment: FixedSizeListArray = match data.value_type() {
        DataType::Int8 => convert_int8_to_f32(data)?,
        DataType::Float16 | DataType::Float32 | DataType::Float64 => data.clone(),
        other => {
            return Err(Error::index(format!(
                "unsupported data dtype for distributed kmeans: {}",
                other
            )));
        }
    };
    let centroids_for_assignment: FixedSizeListArray =
        if centroids.value_type() == data_for_assignment.value_type() {
            centroids.clone()
        } else {
            return Err(Error::index(format!(
                "centroids dtype {} does not match data dtype {}",
                centroids.value_type(),
                data_for_assignment.value_type()
            )));
        };

    // Cosine: caller (Layer 2) is responsible for normalizing both centroids
    // and data before reaching here, so we run the assignment as L2.
    let kmeans = KMeans::with_centroids(
        centroids_for_assignment.values().clone(),
        dim,
        match distance_type {
            DistanceType::Cosine => DistanceType::L2,
            other => other,
        },
        f64::MAX,
    );
    let (membership, distances) = kmeans
        .compute_membership_and_distances(&data_for_assignment)
        .map_err(arrow_error_to_lance)?;

    let mut counts_vec = vec![0u64; k];
    let mut sum_vec = vec![0.0f64; k * dim];
    let mut sq_norm = vec![0.0f64; k];
    let mut loss = vec![0.0f64; k];
    let mut radius = vec![0.0f32; k];

    let value_array = data_for_assignment.values();
    match value_array.data_type() {
        DataType::Float32 => accumulate::<Float32Type>(
            value_array.as_primitive::<Float32Type>().values(),
            dim,
            &membership,
            &distances,
            &mut counts_vec,
            &mut sum_vec,
            &mut sq_norm,
            &mut loss,
            &mut radius,
            |v| v as f64,
        ),
        DataType::Float16 => accumulate::<Float16Type>(
            value_array.as_primitive::<Float16Type>().values(),
            dim,
            &membership,
            &distances,
            &mut counts_vec,
            &mut sum_vec,
            &mut sq_norm,
            &mut loss,
            &mut radius,
            |v| v.to_f64(),
        ),
        DataType::Float64 => accumulate::<Float64Type>(
            value_array.as_primitive::<Float64Type>().values(),
            dim,
            &membership,
            &distances,
            &mut counts_vec,
            &mut sum_vec,
            &mut sq_norm,
            &mut loss,
            &mut radius,
            |v| v,
        ),
        other => {
            return Err(Error::index(format!(
                "unsupported assignment dtype: {}",
                other
            )));
        }
    }

    let new_count = Arc::new(UInt64Array::from(counts_vec));
    let mut sum_builder = FixedSizeListBuilder::new(Float64Builder::new(), dim as i32);
    for ci in 0..k {
        for d in 0..dim {
            sum_builder.values().append_value(sum_vec[ci * dim + d]);
        }
        sum_builder.append(true);
    }
    let new_sum = Arc::new(sum_builder.finish());
    let new_sq = Arc::new(Float64Array::from(sq_norm));
    let new_loss = Arc::new(Float64Array::from(loss));
    let new_radius = Arc::new(Float32Array::from(radius));

    let schema = stats_empty.batch.schema();
    let cluster_id_col = stats_empty.batch.column(0).clone();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            cluster_id_col,
            new_count,
            new_sum,
            new_sq,
            new_loss,
            new_radius,
        ],
    )
    .map_err(arrow_error_to_lance)?;
    Ok(PartialStats { batch })
}

#[allow(clippy::too_many_arguments)]
fn accumulate<T: ArrowPrimitiveType>(
    values: &[T::Native],
    dim: usize,
    membership: &[Option<u32>],
    distances: &[Option<f32>],
    counts: &mut [u64],
    sums: &mut [f64],
    sq_norm: &mut [f64],
    loss: &mut [f64],
    radius: &mut [f32],
    to_f64: impl Fn(T::Native) -> f64,
) where
    T::Native: Copy,
{
    for (row_idx, (&m, &d)) in membership.iter().zip(distances.iter()).enumerate() {
        let (Some(c), Some(dist)) = (m, d) else {
            continue;
        };
        let ci = c as usize;
        counts[ci] += 1;
        loss[ci] += dist as f64;
        if dist > radius[ci] {
            radius[ci] = dist;
        }
        let row = &values[row_idx * dim..(row_idx + 1) * dim];
        let mut row_sq = 0.0f64;
        for (offset, &v) in row.iter().enumerate() {
            let v64 = to_f64(v);
            sums[ci * dim + offset] += v64;
            row_sq += v64 * v64;
        }
        sq_norm[ci] += row_sq;
    }
}

fn convert_int8_to_f32(data: &FixedSizeListArray) -> Result<FixedSizeListArray> {
    let values = data
        .values()
        .as_any()
        .downcast_ref::<PrimitiveArray<Int8Type>>()
        .ok_or_else(|| Error::index("expected Int8 values"))?;
    let f32_values: Float32Array = values.iter().map(|v| v.map(|x| x as f32)).collect();
    FixedSizeListArray::try_new_from_values(f32_values, data.value_length())
        .map_err(arrow_error_to_lance)
}

/// Combine two partial stats produced against the same centroids.
pub fn merge_partial_stats(a: PartialStats, b: PartialStats) -> Result<PartialStats> {
    if a.batch.schema().metadata().get(META_VERSION)
        != b.batch.schema().metadata().get(META_VERSION)
    {
        return Err(Error::index("PartialStats version mismatch"));
    }
    if a.k() != b.k() || a.dim() != b.dim() {
        return Err(Error::index(format!(
            "PartialStats shape mismatch: ({},{}) vs ({},{})",
            a.k(),
            a.dim(),
            b.k(),
            b.dim()
        )));
    }
    if a.distance_type() != b.distance_type() {
        return Err(Error::index("PartialStats distance_type mismatch"));
    }
    if a.centroids_fingerprint() != b.centroids_fingerprint() {
        return Err(Error::index("PartialStats centroids_fingerprint mismatch"));
    }

    let k = a.k();
    let dim = a.dim();
    let counts = pairwise_sum_u64(a.batch.column(1), b.batch.column(1));
    let sums = pairwise_sum_fsl_f64(a.batch.column(2), b.batch.column(2), k, dim);
    let sq_norm = pairwise_sum_f64(a.batch.column(3), b.batch.column(3));
    let loss = pairwise_sum_f64(a.batch.column(4), b.batch.column(4));
    let radius = pairwise_max_f32(a.batch.column(5), b.batch.column(5));

    let schema = a.batch.schema();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            a.batch.column(0).clone(),
            Arc::new(UInt64Array::from(counts)),
            Arc::new(sums),
            Arc::new(Float64Array::from(sq_norm)),
            Arc::new(Float64Array::from(loss)),
            Arc::new(Float32Array::from(radius)),
        ],
    )
    .map_err(arrow_error_to_lance)?;
    Ok(PartialStats { batch })
}

/// Fold an iterator of partial stats. Returns `Err` if the iterator is empty.
pub fn reduce_partial_stats<I: IntoIterator<Item = PartialStats>>(iter: I) -> Result<PartialStats> {
    let mut iter = iter.into_iter();
    let mut acc = iter
        .next()
        .ok_or_else(|| Error::index("reduce_partial_stats: empty iterator"))?;
    for next in iter {
        acc = merge_partial_stats(acc, next)?;
    }
    Ok(acc)
}

fn pairwise_sum_u64(a: &dyn Array, b: &dyn Array) -> Vec<u64> {
    let a = a.as_any().downcast_ref::<UInt64Array>().unwrap();
    let b = b.as_any().downcast_ref::<UInt64Array>().unwrap();
    a.values()
        .iter()
        .zip(b.values().iter())
        .map(|(x, y)| x + y)
        .collect()
}

fn pairwise_sum_f64(a: &dyn Array, b: &dyn Array) -> Vec<f64> {
    let a = a.as_any().downcast_ref::<Float64Array>().unwrap();
    let b = b.as_any().downcast_ref::<Float64Array>().unwrap();
    a.values()
        .iter()
        .zip(b.values().iter())
        .map(|(x, y)| x + y)
        .collect()
}

fn pairwise_max_f32(a: &dyn Array, b: &dyn Array) -> Vec<f32> {
    let a = a.as_any().downcast_ref::<Float32Array>().unwrap();
    let b = b.as_any().downcast_ref::<Float32Array>().unwrap();
    a.values()
        .iter()
        .zip(b.values().iter())
        .map(|(x, y)| x.max(*y))
        .collect()
}

fn pairwise_sum_fsl_f64(a: &dyn Array, b: &dyn Array, k: usize, dim: usize) -> FixedSizeListArray {
    let a = a.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
    let b = b.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
    let av = a.values().as_primitive::<Float64Type>().values();
    let bv = b.values().as_primitive::<Float64Type>().values();
    let mut builder = FixedSizeListBuilder::new(Float64Builder::new(), dim as i32);
    for ci in 0..k {
        for d in 0..dim {
            builder
                .values()
                .append_value(av[ci * dim + d] + bv[ci * dim + d]);
        }
        builder.append(true);
    }
    builder.finish()
}

/// Compute new centroids from accumulated stats.
///
/// `prev` provides the dtype and the fallback for empty clusters.
pub fn finalize_centroids(
    stats: &PartialStats,
    prev: &FixedSizeListArray,
) -> Result<FixedSizeListArray> {
    if stats.k() != prev.len() {
        return Err(Error::index(format!(
            "stats.k {} != prev.len {}",
            stats.k(),
            prev.len()
        )));
    }
    if stats.dim() != prev.value_length() as usize {
        return Err(Error::index(format!(
            "stats.dim {} != prev.dim {}",
            stats.dim(),
            prev.value_length()
        )));
    }
    if stats.total_count() == 0 {
        return Err(Error::index("no training data assigned"));
    }

    let k = stats.k();
    let dim = stats.dim();
    let counts = stats
        .batch
        .column(1)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    let sums_arr = stats
        .batch
        .column(2)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    let sums = sums_arr.values().as_primitive::<Float64Type>().values();

    match prev.value_type() {
        DataType::Float32 => {
            let prev_vals = prev.values().as_primitive::<Float32Type>().values();
            let mut out = prev_vals.to_vec();
            for ci in 0..k {
                let n = counts.value(ci);
                if n == 0 {
                    continue;
                }
                for d in 0..dim {
                    let s = sums[ci * dim + d];
                    if !s.is_finite() {
                        return Err(Error::index(format!(
                            "non-finite sum at cluster {}, dim {}",
                            ci, d
                        )));
                    }
                    out[ci * dim + d] = (s / n as f64) as f32;
                }
            }
            FixedSizeListArray::try_new_from_values(Float32Array::from(out), dim as i32)
                .map_err(arrow_error_to_lance)
        }
        DataType::Float16 => {
            let prev_vals = prev.values().as_primitive::<Float16Type>().values();
            let mut out: Vec<f16> = prev_vals.to_vec();
            for ci in 0..k {
                let n = counts.value(ci);
                if n == 0 {
                    continue;
                }
                for d in 0..dim {
                    let s = sums[ci * dim + d];
                    if !s.is_finite() {
                        return Err(Error::index(format!(
                            "non-finite sum at cluster {}, dim {}",
                            ci, d
                        )));
                    }
                    out[ci * dim + d] = f16::from_f64(s / n as f64);
                }
            }
            FixedSizeListArray::try_new_from_values(Float16Array::from_iter_values(out), dim as i32)
                .map_err(arrow_error_to_lance)
        }
        DataType::Float64 => {
            let prev_vals = prev.values().as_primitive::<Float64Type>().values();
            let mut out = prev_vals.to_vec();
            for ci in 0..k {
                let n = counts.value(ci);
                if n == 0 {
                    continue;
                }
                for d in 0..dim {
                    let s = sums[ci * dim + d];
                    if !s.is_finite() {
                        return Err(Error::index(format!(
                            "non-finite sum at cluster {}, dim {}",
                            ci, d
                        )));
                    }
                    out[ci * dim + d] = s / n as f64;
                }
            }
            FixedSizeListArray::try_new_from_values(Float64Array::from(out), dim as i32)
                .map_err(arrow_error_to_lance)
        }
        other => Err(Error::index(format!(
            "finalize_centroids: unsupported prev dtype {}",
            other
        ))),
    }
}

const RESERVOIR_VEC_COL: &str = "vec";

fn samples_schema(value_type: DataType, dim: usize) -> SchemaRef {
    let item = DataType::FixedSizeList(Arc::new(Field::new("item", value_type, true)), dim as i32);
    Arc::new(Schema::new(vec![Field::new(
        RESERVOIR_VEC_COL,
        item,
        false,
    )]))
}

/// Algorithm-R reservoir sample of `target` rows from `data`.
///
/// Output schema: `vec: FixedSizeList<element_type, dim>`.
/// If `data.len() <= target`, returns all rows verbatim.
pub fn local_reservoir_sample(
    data: &FixedSizeListArray,
    target: usize,
    rng_seed: u64,
) -> Result<RecordBatch> {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    let dim = data.value_length() as usize;
    let n = data.len();
    let take = target.min(n);
    let mut rng = StdRng::seed_from_u64(rng_seed);

    let indices: Vec<usize> = if n <= target {
        (0..n).collect()
    } else {
        let mut chosen: Vec<usize> = (0..target).collect();
        for i in target..n {
            let j = rng.random_range(0..=i);
            if j < target {
                chosen[j] = i;
            }
        }
        chosen
    };

    let schema = samples_schema(data.value_type(), dim);
    if take == 0 {
        let empty = arrow_array::array::new_empty_array(schema.field(0).data_type());
        return RecordBatch::try_new(schema, vec![empty]).map_err(arrow_error_to_lance);
    }
    let take_array = UInt32Array::from(indices.iter().map(|&i| i as u32).collect::<Vec<_>>());
    let taken = arrow::compute::take(data, &take_array, None).map_err(arrow_error_to_lance)?;
    RecordBatch::try_new(schema, vec![taken]).map_err(arrow_error_to_lance)
}

fn concat_samples(samples: Vec<RecordBatch>) -> Result<FixedSizeListArray> {
    if samples.is_empty() {
        return Err(Error::index(
            "select/bootstrap requires at least one sample batch",
        ));
    }
    let arrays: Vec<&dyn Array> = samples.iter().map(|b| b.column(0).as_ref()).collect();
    let combined = arrow::compute::concat(&arrays).map_err(arrow_error_to_lance)?;
    combined
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .cloned()
        .ok_or_else(|| Error::index("concatenated samples are not FixedSizeList"))
}

/// Driver-side: pick `k` rows uniformly at random from the union of worker samples.
pub fn select_initial_centroids(
    samples: Vec<RecordBatch>,
    k: usize,
    rng_seed: u64,
) -> Result<FixedSizeListArray> {
    use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

    let combined = concat_samples(samples)?;
    if combined.len() < k {
        return Err(Error::index(format!(
            "not enough samples ({}) to select {} centroids",
            combined.len(),
            k
        )));
    }
    let mut rng = StdRng::seed_from_u64(rng_seed);
    let mut idx: Vec<u32> = (0..combined.len() as u32).collect();
    idx.shuffle(&mut rng);
    let take = UInt32Array::from(idx[..k].to_vec());
    let chosen = arrow::compute::take(&combined, &take, None).map_err(arrow_error_to_lance)?;
    chosen
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .cloned()
        .ok_or_else(|| Error::index("select_initial_centroids: take returned non-FSL"))
}

/// Driver-side: run single-machine `train_kmeans` over the union of worker samples
/// to obtain a high-quality initial set of centroids. For `k > 256`, the existing
/// kmeans engine automatically falls back to hierarchical kmeans (see
/// `train_kmeans` in this crate).
pub fn bootstrap_centroids(
    samples: Vec<RecordBatch>,
    k: usize,
    distance_type: DistanceType,
    rng_seed: u64,
) -> Result<FixedSizeListArray> {
    let combined = concat_samples(samples)?;
    let dim = combined.value_length() as usize;
    let params = KMeansParams::default()
        .with_distance_type(distance_type)
        .with_seed(rng_seed);
    match combined.value_type() {
        DataType::Float32 => {
            let arr = combined.values().as_primitive::<Float32Type>().clone();
            let model = train_kmeans::<Float32Type>(&arr, params, dim, k, 256)?;
            FixedSizeListArray::try_new_from_values(
                model.centroids.as_primitive::<Float32Type>().clone(),
                dim as i32,
            )
            .map_err(arrow_error_to_lance)
        }
        DataType::Float16 => {
            let arr = combined.values().as_primitive::<Float16Type>().clone();
            let model = train_kmeans::<Float16Type>(&arr, params, dim, k, 256)?;
            FixedSizeListArray::try_new_from_values(
                model.centroids.as_primitive::<Float16Type>().clone(),
                dim as i32,
            )
            .map_err(arrow_error_to_lance)
        }
        DataType::Float64 => {
            let arr = combined.values().as_primitive::<Float64Type>().clone();
            let model = train_kmeans::<Float64Type>(&arr, params, dim, k, 256)?;
            FixedSizeListArray::try_new_from_values(
                model.centroids.as_primitive::<Float64Type>().clone(),
                dim as i32,
            )
            .map_err(arrow_error_to_lance)
        }
        other => Err(Error::index(format!(
            "bootstrap_centroids: unsupported dtype {}",
            other
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::FixedSizeListArray;
    use arrow_schema::DataType;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_linalg::distance::DistanceType;

    #[test]
    fn test_partial_stats_empty_schema() {
        let stats = PartialStats::empty(64, 16, DistanceType::L2, [0u8; 8]);
        let batch = stats.into_record_batch();

        assert_eq!(batch.num_rows(), 64);
        let schema = batch.schema();
        assert_eq!(schema.field(0).name(), "cluster_id");
        assert_eq!(schema.field(0).data_type(), &DataType::UInt32);
        assert_eq!(schema.field(1).name(), "count");
        assert_eq!(schema.field(1).data_type(), &DataType::UInt64);
        assert!(matches!(
            schema.field(2).data_type(),
            DataType::FixedSizeList(_, 16)
        ));
        let md = schema.metadata();
        assert_eq!(
            md.get("lance.partial_stats.version").map(String::as_str),
            Some("1")
        );
        assert_eq!(
            md.get("lance.partial_stats.k").map(String::as_str),
            Some("64")
        );
        assert_eq!(
            md.get("lance.partial_stats.dim").map(String::as_str),
            Some("16")
        );
        assert_eq!(
            md.get("lance.partial_stats.distance_type")
                .map(String::as_str),
            Some("l2")
        );
    }

    #[test]
    fn test_from_record_batch_rejects_wrong_version() {
        let stats = PartialStats::empty(4, 8, DistanceType::L2, [0u8; 8]);
        let batch = stats.into_record_batch();
        let mut md = batch.schema().metadata().clone();
        md.insert(META_VERSION.into(), "999".into());
        let new_schema = Arc::new((*batch.schema()).clone().with_metadata(md));
        let bad = RecordBatch::try_new(new_schema, batch.columns().to_vec()).unwrap();
        assert!(PartialStats::from_record_batch(bad).is_err());
    }

    #[test]
    fn test_fingerprint_is_stable_and_dtype_independent() {
        let f32_centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![1.0_f32, 2.0, 3.0, 4.0]),
            2,
        )
        .unwrap();
        let fp1 = compute_centroids_fingerprint(&f32_centroids);
        let fp2 = compute_centroids_fingerprint(&f32_centroids);
        assert_eq!(fp1, fp2, "fingerprint must be deterministic");

        let mutated = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![1.0_f32, 2.0, 3.0, 5.0]),
            2,
        )
        .unwrap();
        assert_ne!(
            compute_centroids_fingerprint(&mutated),
            fp1,
            "different bytes must produce different fingerprint"
        );
    }

    #[test]
    fn test_compute_partial_stats_l2_basic() {
        // 4 centroids in 2-D, 6 vectors; manually-computed expected counts.
        let centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.0_f32, 0.0, 10.0, 0.0, 0.0, 10.0, 10.0, 10.0]),
            2,
        )
        .unwrap();
        let data = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![
                0.1_f32, 0.1, // -> cluster 0
                10.0, 0.1, // -> cluster 1
                0.0, 9.9, // -> cluster 2
                9.9, 9.9, // -> cluster 3
                10.1, 0.0, // -> cluster 1
                0.0, 0.0, // -> cluster 0
            ]),
            2,
        )
        .unwrap();

        let stats = compute_partial_stats(&centroids, &data, DistanceType::L2).unwrap();
        let counts = stats
            .record_batch()
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(counts, vec![2, 2, 1, 1]);
        assert_eq!(stats.total_count(), 6);
        assert!(stats.total_loss() >= 0.0);
    }

    #[test]
    fn test_compute_partial_stats_rejects_hamming() {
        let centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.0_f32, 0.0, 1.0, 1.0]),
            2,
        )
        .unwrap();
        let data =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0_f32, 0.0]), 2)
                .unwrap();
        assert!(compute_partial_stats(&centroids, &data, DistanceType::Hamming).is_err());
    }

    #[test]
    fn test_compute_partial_stats_all_nan_returns_empty() {
        let centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.0_f32, 0.0, 10.0, 10.0]),
            2,
        )
        .unwrap();
        let data = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![f32::NAN, f32::NAN, f32::NAN, f32::NAN]),
            2,
        )
        .unwrap();
        let stats = compute_partial_stats(&centroids, &data, DistanceType::L2).unwrap();
        assert_eq!(stats.total_count(), 0, "all NaN -> no assignments");
    }

    #[test]
    fn test_merge_partial_stats_simple_sum() {
        let centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.0_f32, 0.0, 10.0, 10.0]),
            2,
        )
        .unwrap();
        let d1 = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.0_f32, 0.0, 1.0, 1.0]),
            2,
        )
        .unwrap();
        let d2 = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![10.0_f32, 10.0, 9.0, 9.0]),
            2,
        )
        .unwrap();
        let s1 = compute_partial_stats(&centroids, &d1, DistanceType::L2).unwrap();
        let s2 = compute_partial_stats(&centroids, &d2, DistanceType::L2).unwrap();

        let merged = merge_partial_stats(s1, s2).unwrap();
        let counts = merged
            .record_batch()
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(counts, vec![2, 2]);
        assert_eq!(merged.total_count(), 4);
    }

    #[test]
    fn test_merge_rejects_fingerprint_mismatch() {
        let c1 = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.0_f32, 0.0, 1.0, 1.0]),
            2,
        )
        .unwrap();
        let c2 = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.0_f32, 0.0, 5.0, 5.0]),
            2,
        )
        .unwrap();
        let d = FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.5_f32, 0.5]), 2)
            .unwrap();
        let s1 = compute_partial_stats(&c1, &d, DistanceType::L2).unwrap();
        let s2 = compute_partial_stats(&c2, &d, DistanceType::L2).unwrap();
        assert!(merge_partial_stats(s1, s2).is_err());
    }

    #[test]
    fn test_merge_empty_is_identity() {
        let centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.0_f32, 0.0, 1.0, 1.0]),
            2,
        )
        .unwrap();
        let d = FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.1_f32, 0.1]), 2)
            .unwrap();
        let s = compute_partial_stats(&centroids, &d, DistanceType::L2).unwrap();
        let fp = compute_centroids_fingerprint(&centroids);
        let empty = PartialStats::empty(2, 2, DistanceType::L2, fp);
        let merged = merge_partial_stats(s.clone(), empty).unwrap();
        assert_eq!(merged.total_count(), s.total_count());
        assert_eq!(merged.total_loss(), s.total_loss());
    }

    #[test]
    fn test_finalize_centroids_basic_f32() {
        let prev = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.0_f32, 0.0, 100.0, 100.0]),
            2,
        )
        .unwrap();
        let data = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.0_f32, 0.0, 2.0, 2.0, 99.0, 99.0, 101.0, 101.0]),
            2,
        )
        .unwrap();
        let stats = compute_partial_stats(&prev, &data, DistanceType::L2).unwrap();
        let new = finalize_centroids(&stats, &prev).unwrap();
        let v = new.values().as_primitive::<Float32Type>().values().to_vec();
        assert!((v[0] - 1.0).abs() < 1e-5);
        assert!((v[1] - 1.0).abs() < 1e-5);
        assert!((v[2] - 100.0).abs() < 1e-5);
        assert!((v[3] - 100.0).abs() < 1e-5);
    }

    #[test]
    fn test_finalize_centroids_empty_cluster_keeps_prev() {
        let prev = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.0_f32, 0.0, 100.0, 100.0]),
            2,
        )
        .unwrap();
        let data = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.0_f32, 0.0, 0.5, 0.5]),
            2,
        )
        .unwrap();
        let stats = compute_partial_stats(&prev, &data, DistanceType::L2).unwrap();
        let new = finalize_centroids(&stats, &prev).unwrap();
        let v = new.values().as_primitive::<Float32Type>().values().to_vec();
        // cluster 1 has 0 assignments and must retain prev[1]
        assert!((v[2] - 100.0).abs() < 1e-6);
        assert!((v[3] - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_finalize_centroids_all_empty_errors() {
        let prev = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.0_f32, 0.0, 100.0, 100.0]),
            2,
        )
        .unwrap();
        let fp = compute_centroids_fingerprint(&prev);
        let empty = PartialStats::empty(2, 2, DistanceType::L2, fp);
        assert!(finalize_centroids(&empty, &prev).is_err());
    }

    fn random_fsl_f32(seed: u64, rows: usize, dim: usize) -> FixedSizeListArray {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(seed);
        let total = rows * dim;
        let v: Vec<f32> = (0..total)
            .map(|_| rng.random_range(-10.0..10.0_f32))
            .collect();
        FixedSizeListArray::try_new_from_values(Float32Array::from(v), dim as i32).unwrap()
    }

    fn assert_partial_stats_close(a: &PartialStats, b: &PartialStats, eps: f64) {
        assert_eq!(a.k(), b.k());
        let ca = a
            .record_batch()
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values()
            .to_vec();
        let cb = b
            .record_batch()
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(ca, cb, "counts must match exactly");

        let sa = a
            .record_batch()
            .column(2)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap()
            .values()
            .as_primitive::<Float64Type>()
            .values()
            .to_vec();
        let sb = b
            .record_batch()
            .column(2)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap()
            .values()
            .as_primitive::<Float64Type>()
            .values()
            .to_vec();
        for (x, y) in sa.iter().zip(sb.iter()) {
            assert!((x - y).abs() < eps, "sum mismatch: {} vs {}", x, y);
        }
    }

    #[test]
    fn test_distributed_equals_single_three_way_split() {
        let centroids = random_fsl_f32(1, 16, 8);
        let data = random_fsl_f32(2, 600, 8);

        let single = compute_partial_stats(&centroids, &data, DistanceType::L2).unwrap();

        let dim = 8;
        let values = data.values().as_primitive::<Float32Type>().values();
        let split_a = FixedSizeListArray::try_new_from_values(
            Float32Array::from(values[..200 * dim].to_vec()),
            dim as i32,
        )
        .unwrap();
        let split_b = FixedSizeListArray::try_new_from_values(
            Float32Array::from(values[200 * dim..400 * dim].to_vec()),
            dim as i32,
        )
        .unwrap();
        let split_c = FixedSizeListArray::try_new_from_values(
            Float32Array::from(values[400 * dim..].to_vec()),
            dim as i32,
        )
        .unwrap();

        let distributed = reduce_partial_stats(vec![
            compute_partial_stats(&centroids, &split_a, DistanceType::L2).unwrap(),
            compute_partial_stats(&centroids, &split_b, DistanceType::L2).unwrap(),
            compute_partial_stats(&centroids, &split_c, DistanceType::L2).unwrap(),
        ])
        .unwrap();

        assert_partial_stats_close(&single, &distributed, 1e-6);
    }

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn merge_is_commutative(seed in 0u64..1024, k in 4usize..16, dim in 2usize..8) {
            let centroids = random_fsl_f32(seed, k, dim);
            let d1 = random_fsl_f32(seed.wrapping_add(1), 50, dim);
            let d2 = random_fsl_f32(seed.wrapping_add(2), 50, dim);
            let s1 = compute_partial_stats(&centroids, &d1, DistanceType::L2).unwrap();
            let s2 = compute_partial_stats(&centroids, &d2, DistanceType::L2).unwrap();
            let ab = merge_partial_stats(s1.clone(), s2.clone()).unwrap();
            let ba = merge_partial_stats(s2, s1).unwrap();
            assert_partial_stats_close(&ab, &ba, 1e-9);
        }

        #[test]
        fn merge_is_associative(seed in 0u64..1024, k in 4usize..16, dim in 2usize..8) {
            let centroids = random_fsl_f32(seed, k, dim);
            let d1 = random_fsl_f32(seed.wrapping_add(11), 30, dim);
            let d2 = random_fsl_f32(seed.wrapping_add(22), 30, dim);
            let d3 = random_fsl_f32(seed.wrapping_add(33), 30, dim);
            let s1 = compute_partial_stats(&centroids, &d1, DistanceType::L2).unwrap();
            let s2 = compute_partial_stats(&centroids, &d2, DistanceType::L2).unwrap();
            let s3 = compute_partial_stats(&centroids, &d3, DistanceType::L2).unwrap();
            let lhs = merge_partial_stats(
                merge_partial_stats(s1.clone(), s2.clone()).unwrap(),
                s3.clone(),
            )
            .unwrap();
            let rhs = merge_partial_stats(s1, merge_partial_stats(s2, s3).unwrap()).unwrap();
            assert_partial_stats_close(&lhs, &rhs, 1e-9);
        }
    }

    #[test]
    fn test_local_reservoir_sample_size_and_seed() {
        let data = random_fsl_f32(7, 1000, 4);
        let s1 = local_reservoir_sample(&data, 64, 42).unwrap();
        let s2 = local_reservoir_sample(&data, 64, 42).unwrap();
        let s3 = local_reservoir_sample(&data, 64, 43).unwrap();

        assert_eq!(s1.num_rows(), 64);
        assert_eq!(s1.schema().field(0).name(), "vec");
        assert_eq!(s1, s2, "same seed -> same output (I7)");
        assert_ne!(s1, s3, "different seed -> different output");
    }

    #[test]
    fn test_local_reservoir_sample_smaller_than_target() {
        let data = random_fsl_f32(8, 10, 4);
        let s = local_reservoir_sample(&data, 64, 1).unwrap();
        assert_eq!(s.num_rows(), 10, "less data than target -> return all rows");
    }

    #[test]
    fn test_select_initial_centroids_picks_k_rows() {
        let s1 = local_reservoir_sample(&random_fsl_f32(1, 200, 6), 100, 9).unwrap();
        let s2 = local_reservoir_sample(&random_fsl_f32(2, 200, 6), 100, 10).unwrap();
        let centroids = select_initial_centroids(vec![s1, s2], 32, 7).unwrap();
        assert_eq!(centroids.len(), 32);
        assert_eq!(centroids.value_length(), 6);
    }

    #[test]
    fn test_bootstrap_centroids_runs_kmeans() {
        let s = local_reservoir_sample(&random_fsl_f32(3, 5_000, 8), 4_000, 11).unwrap();
        let centroids = bootstrap_centroids(vec![s], 32, DistanceType::L2, 13).unwrap();
        assert_eq!(centroids.len(), 32);
        assert_eq!(centroids.value_length(), 8);
    }
}
