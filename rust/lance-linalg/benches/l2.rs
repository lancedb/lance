// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::iter::repeat_with;

use arrow_array::{
    types::{Float16Type, Float32Type, Float64Type},
    Float32Array, UInt8Array,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_traits::{AsPrimitive, Float};

#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use lance_arrow::{ArrowFloatType, FloatArray};
use lance_linalg::distance::{l2::l2, l2_distance_batch, L2};
use lance_testing::datagen::generate_random_array_with_seed;
use rand::Rng;

const DIMENSION: usize = 1024;
const TOTAL: usize = 1024 * 1024; // 1M vectors

/// Naive scalar implementation of L2 distance.
fn l2_scalar<T: Float + AsPrimitive<f32>>(x: &[T], y: &[T]) -> T {
    let mut sum = T::zero();
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        sum = sum + (xi - yi).powi(2);
    }
    sum.sqrt()
}

fn run_bench<T: ArrowFloatType + L2>(c: &mut Criterion) {
    const DIMENSION: usize = 1024;
    const TOTAL: usize = 1024 * 1024; // 1M vectors

    let key = generate_random_array_with_seed::<T>(DIMENSION, [0; 32]);
    // 1M of 1024 D vectors
    let target = generate_random_array_with_seed::<T>(TOTAL * DIMENSION, [42; 32]);

    let type_name = std::any::type_name::<T::Native>();

    c.bench_function(format!("L2({type_name}, scalar)").as_str(), |b| {
        b.iter(|| {
            Float32Array::from_iter_values(
                target
                    .as_slice()
                    .chunks(DIMENSION)
                    .map(|arr| l2_scalar(key.as_slice(), arr).as_()),
            );
        });
    });

    c.bench_function(
        format!("L2({type_name}, auto-vectorization)").as_str(),
        |b| {
            b.iter(|| unsafe {
                Float32Array::from_trusted_len_iter(
                    target
                        .as_slice()
                        .chunks(DIMENSION)
                        .map(|y| Some(black_box(l2(key.as_slice(), y)))),
                );
            });
        },
    );
}

fn bench_distance(c: &mut Criterion) {
    run_bench::<Float16Type>(c);
    run_bench::<Float32Type>(c);
    let key: Float32Array = generate_random_array_with_seed::<Float32Type>(DIMENSION, [0; 32]);
    let target: Float32Array =
        generate_random_array_with_seed::<Float32Type>(TOTAL * DIMENSION, [42; 32]);
    c.bench_function("L2(f32, simd)", |b| {
        b.iter(|| {
            black_box(l2_distance_batch(key.values(), target.values(), DIMENSION).count());
        })
    });

    run_bench::<Float64Type>(c);
}

fn bench_small_distance(c: &mut Criterion) {
    let key = generate_random_array_with_seed::<Float32Type>(8, [5; 32]);
    // 1M of 1024 D vectors. 4GB in memory.
    let target = generate_random_array_with_seed::<Float32Type>(TOTAL * 8, [7; 32]);
    c.bench_function("L2(simd,f32x8)", |b| {
        b.iter(|| {
            black_box(l2_distance_batch(key.values(), target.values(), 8).count());
        })
    });
}

fn l2_distance_uint_scalar(key: &[u8], target: &[u8]) -> f32 {
    key.iter()
        .zip(target.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).pow(2))
        .sum::<i16>() as f32
}

fn bench_uint_distance(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let key = repeat_with(|| rng.gen::<u8>())
        .take(DIMENSION)
        .collect::<Vec<_>>();
    let target = repeat_with(|| rng.gen::<u8>())
        .take(TOTAL * DIMENSION)
        .collect::<Vec<_>>();
    println!("Target size: {}", target.len());

    c.bench_function("L2(uint8, scalar)", |b| {
        b.iter(|| {
            black_box(
                target
                    .chunks_exact(DIMENSION)
                    .map(|tgt| l2_distance_uint_scalar(&key, tgt))
                    .collect::<Vec<_>>(),
            );
        });
    });
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_distance, bench_small_distance, bench_uint_distance);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_distance, bench_small_distance, bench_uint_distance);
criterion_main!(benches);
