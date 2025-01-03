// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark of building PQ distance table.

use std::iter::repeat;

use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, UInt8Array};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lance_arrow::FixedSizeListArrayExt;
use lance_index::vector::pq::distance::{build_distance_table_dot, build_distance_table_l2};
use lance_index::vector::pq::ProductQuantizer;
use lance_linalg::distance::DistanceType;
use lance_testing::datagen::generate_random_array_with_seed;
use rand::{prelude::StdRng, Rng, SeedableRng};

#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

const PQ: usize = 96;
const DIM: usize = 1536;
const TOTAL: usize = 16 * 1000;

fn construct_dist_table(c: &mut Criterion) {
    let codebook = generate_random_array_with_seed::<Float32Type>(256 * DIM, [88; 32]);
    let query = generate_random_array_with_seed::<Float32Type>(DIM, [32; 32]);

    c.bench_function(
        format!(
            "construct_dist_table: {},PQ={}x{},DIM={}",
            DistanceType::L2,
            PQ,
            4,
            DIM
        )
        .as_str(),
        |b| {
            b.iter(|| {
                black_box(build_distance_table_l2(
                    codebook.values(),
                    4,
                    PQ,
                    query.values(),
                ));
            })
        },
    );

    c.bench_function(
        format!(
            "construct_dist_table: {},PQ={}x{},DIM={}",
            DistanceType::Dot,
            PQ,
            4,
            DIM
        )
        .as_str(),
        |b| {
            b.iter(|| {
                black_box(build_distance_table_dot(
                    codebook.values(),
                    4,
                    PQ,
                    query.values(),
                ));
            })
        },
    );
}

fn compute_distances(c: &mut Criterion) {
    let codebook = generate_random_array_with_seed::<Float32Type>(256 * DIM, [88; 32]);
    let query = generate_random_array_with_seed::<Float32Type>(DIM, [32; 32]);

    let mut rnd = StdRng::from_seed([32; 32]);
    let code = UInt8Array::from_iter_values(repeat(rnd.gen::<u8>()).take(TOTAL * PQ));

    for dt in [DistanceType::L2, DistanceType::Cosine, DistanceType::Dot].iter() {
        let pq = ProductQuantizer::new(
            PQ,
            4,
            DIM,
            FixedSizeListArray::try_new_from_values(codebook.clone(), DIM as i32).unwrap(),
            *dt,
        );

        c.bench_function(
            format!("{},{},PQ={}x{},DIM={}", TOTAL, dt, PQ, 4, DIM).as_str(),
            |b| {
                b.iter(|| {
                    black_box(pq.compute_distances(&query, &code).unwrap());
                })
            },
        );
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = construct_dist_table, compute_distances);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = construct_dist_table, compute_distances);

criterion_main!(benches);
