// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark of building PQ distance table.

use std::time::Duration;

use arrow::datatypes::UInt64Type;
use arrow_array::types::Float32Type;
use arrow_schema::DataType;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use lance_arrow::fixed_size_list_type;
use lance_core::ROW_ID;
use lance_datagen::array::rand_type;
use lance_datagen::{BatchGeneratorBuilder, RowCount};
use lance_index::vector::bq::RQRotationType;
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::ex_dot::{
    build_ex_query, ex_dot_code_bytes, ex_dot_kernel, needs_plane_repack, packed_ex_code_value,
    plane_pack_row,
};
use lance_index::vector::bq::storage::*;
use lance_index::vector::bq::transform::{ADD_FACTORS_COLUMN, SCALE_FACTORS_COLUMN};
use lance_index::vector::quantizer::{Quantization, QuantizerStorage};
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_linalg::distance::DistanceType;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const DIM: usize = 128;
const TOTAL: usize = 16 * 1000;

fn mock_rq_storage(num_bits: u8, rotation_type: RQRotationType) -> RabitQuantizationStorage {
    // generate random rq codes
    let rq = RabitQuantizer::new_with_rotation::<Float32Type>(num_bits, DIM as i32, rotation_type);
    let builder = BatchGeneratorBuilder::new()
        .col(ROW_ID, lance_datagen::array::step::<UInt64Type>())
        .col(
            RABIT_CODE_COLUMN,
            rand_type(&fixed_size_list_type(
                (DIM * num_bits as usize / u8::BITS as usize) as i32,
                DataType::UInt8,
            )),
        )
        .col(ADD_FACTORS_COLUMN, rand_type(&DataType::Float32))
        .col(SCALE_FACTORS_COLUMN, rand_type(&DataType::Float32));
    RabitQuantizationStorage::try_from_batch(
        builder
            .into_batch_rows(RowCount::from(TOTAL as u64))
            .unwrap(),
        &rq.metadata(None),
        DistanceType::L2,
        None,
    )
    .unwrap()
}

fn construct_dist_table(c: &mut Criterion) {
    let rotation_types = [RQRotationType::Fast, RQRotationType::Matrix];
    for num_bits in 1..=1 {
        for rotation_type in rotation_types {
            let rq = mock_rq_storage(num_bits, rotation_type);
            let query = rand_type(&DataType::Float32)
                .generate_default(RowCount::from(DIM as u64))
                .unwrap();
            c.bench_function(
                format!(
                    "RQ{}({:?}): construct_dist_table: {},DIM={}",
                    num_bits,
                    rotation_type,
                    DistanceType::L2,
                    DIM
                )
                .as_str(),
                |b| {
                    b.iter(|| {
                        black_box(rq.dist_calculator(query.clone(), 0.0));
                    })
                },
            );
        }
    }
}

fn compute_distances(c: &mut Criterion) {
    let rotation_types = [RQRotationType::Fast, RQRotationType::Matrix];
    for num_bits in 1..=1 {
        for rotation_type in rotation_types {
            let rq = mock_rq_storage(num_bits, rotation_type);
            let query = rand_type(&DataType::Float32)
                .generate_default(RowCount::from(DIM as u64))
                .unwrap();
            let dist_calc = rq.dist_calculator(query.clone(), 0.0);

            c.bench_function(
                format!(
                    "RQ{}({:?}): compute_distances: {},DIM={}",
                    num_bits, rotation_type, TOTAL, DIM
                )
                .as_str(),
                |b| {
                    b.iter(|| {
                        black_box(dist_calc.distance_all(0));
                    })
                },
            );

            c.bench_function(
                format!(
                    "RQ{}({:?}): compute_distances_single: {},DIM={}",
                    num_bits, rotation_type, TOTAL, DIM
                )
                .as_str(),
                |b| {
                    b.iter(|| {
                        for i in 0..TOTAL {
                            black_box(dist_calc.distance(i as u32));
                        }
                    })
                },
            );
        }
    }
}

/// The table-gather ex distance used before the dedicated ex-dot kernels,
/// kept here as the baseline: per dim, extract the packed code and gather
/// `query[d] * code` from a `dim * 2^ex_bits` table.
fn gather_ex_distance(row_codes: &[u8], dim: usize, ex_bits: u8, ex_dist_table: &[f32]) -> f32 {
    let entries_per_dim = 1usize << ex_bits;
    (0..dim)
        .map(|dim_idx| {
            let code = packed_ex_code_value(row_codes, dim_idx, ex_bits) as usize;
            ex_dist_table[dim_idx * entries_per_dim + code]
        })
        .sum()
}

fn ex_dot_kernels(c: &mut Criterion) {
    for ex_dim in [1536usize, 2048] {
        ex_dot_kernels_for_dim(c, ex_dim);
    }
}

fn ex_dot_kernels_for_dim(c: &mut Criterion, ex_dim: usize) {
    const NUM_ROWS: usize = 1024;

    let mut rng = SmallRng::seed_from_u64(42);
    let query = (0..ex_dim)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect::<Vec<_>>();

    for ex_bits in 1..=8u8 {
        let max_code = ((1u16 << ex_bits) - 1) as u8;
        let seq_code_len = (ex_dim * ex_bits as usize).div_ceil(8);
        let mut seq_codes = vec![0u8; NUM_ROWS * seq_code_len];
        for row in seq_codes.chunks_exact_mut(seq_code_len) {
            for dim in 0..ex_dim {
                let value = rng.random_range(0..=max_code);
                let bit_offset = dim * ex_bits as usize;
                let bits = (value as u16) << (bit_offset % 8);
                row[bit_offset / 8] |= bits as u8;
                if bits >> 8 != 0 {
                    row[bit_offset / 8 + 1] |= (bits >> 8) as u8;
                }
            }
        }

        let kernel_code_len = ex_dot_code_bytes(ex_dim, ex_bits);
        let kernel_codes = if needs_plane_repack(ex_bits) {
            let mut out = vec![0u8; NUM_ROWS * kernel_code_len];
            for (seq_row, plane_row) in seq_codes
                .chunks_exact(seq_code_len)
                .zip(out.chunks_exact_mut(kernel_code_len))
            {
                plane_pack_row(seq_row, ex_dim, ex_bits, plane_row);
            }
            out
        } else {
            seq_codes.clone()
        };

        let ex_query = build_ex_query(&query, ex_bits);
        let kernel = ex_dot_kernel(ex_bits);
        c.bench_function(
            format!("RQ ex_dot kernel: ex_bits={ex_bits}, DIM={ex_dim}, rows={NUM_ROWS}").as_str(),
            |b| {
                b.iter(|| {
                    let mut sum = 0.0f32;
                    for row in kernel_codes.chunks_exact(kernel_code_len) {
                        sum += kernel(&ex_query, row);
                    }
                    black_box(sum)
                })
            },
        );

        let entries_per_dim = 1usize << ex_bits;
        let mut ex_dist_table = vec![0.0f32; ex_dim * entries_per_dim];
        for (dim, table) in ex_dist_table.chunks_exact_mut(entries_per_dim).enumerate() {
            for (code, value) in table.iter_mut().enumerate() {
                *value = query[dim] * code as f32;
            }
        }
        c.bench_function(
            format!("RQ ex_dot table-gather: ex_bits={ex_bits}, DIM={ex_dim}, rows={NUM_ROWS}")
                .as_str(),
            |b| {
                b.iter(|| {
                    let mut sum = 0.0f32;
                    for row in seq_codes.chunks_exact(seq_code_len) {
                        sum += gather_ex_distance(row, ex_dim, ex_bits, &ex_dist_table);
                    }
                    black_box(sum)
                })
            },
        );
    }
}

criterion_group!(
    name=benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = construct_dist_table, compute_distances, ex_dot_kernels);

criterion_main!(benches);
