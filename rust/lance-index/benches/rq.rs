// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark of building PQ distance table.

use std::time::Duration;

use arrow::datatypes::UInt64Type;
use arrow_array::types::Float32Type;
use arrow_schema::DataType;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lance_arrow::fixed_size_list_type;
use lance_core::ROW_ID;
use lance_datagen::array::rand_type;
use lance_datagen::{BatchGeneratorBuilder, RowCount};
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::storage::*;
use lance_index::vector::bq::transform::{ADD_FACTORS_COLUMN, SCALE_FACTORS_COLUMN};
use lance_index::vector::bq::RQRotationType;
use lance_index::vector::quantizer::{Quantization, QuantizerStorage};
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_linalg::distance::DistanceType;

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

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = construct_dist_table, compute_distances);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = construct_dist_table, compute_distances);

criterion_main!(benches);
