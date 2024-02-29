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
    Float16Array, Float32Array,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use half::bf16;
use num_traits::Float;
use rand::Rng;

use lance_arrow::{ArrowFloatType, FloatArray};
use lance_linalg::distance::norm_l2;
use lance_linalg::distance::norm_l2::Normalize;
use lance_testing::datagen::generate_random_array_with_seed;

#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

const DIMENSION: usize = 1024;
const TOTAL: usize = 1024 * 1024; // 1M vectors

fn run_bench<T: ArrowFloatType>(c: &mut Criterion)
where
    T::Native: Float,
{
    // 1M of 1024 D vectors
    let target = generate_random_array_with_seed::<T>(TOTAL * DIMENSION, [42; 32]);

    let type_name = std::any::type_name::<T::Native>();

    c.bench_function(format!("NormL2({type_name}, scalar)").as_str(), |b| {
        b.iter(|| {
            target
                .as_slice()
                .chunks(DIMENSION)
                .map(|arr| arr.iter().map(|&x| x * x).sum::<T::Native>().sqrt())
                .collect::<Vec<_>>()
        });
    });

    c.bench_function(
        format!("NormL2({type_name}, auto-vectorization)").as_str(),
        |b| {
            b.iter(|| {
                black_box(
                    target
                        .as_slice()
                        .chunks_exact(DIMENSION)
                        .map(norm_l2)
                        .collect::<Vec<_>>(),
                );
            });
        },
    );
}

fn bench_distance(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let target = repeat_with(|| rng.gen::<u16>())
        .map(bf16::from_bits)
        .take(TOTAL * DIMENSION)
        .collect::<Vec<_>>();
    c.bench_function("norm_l2(bf16, auto-vectorization)", |b| {
        b.iter(|| {
            black_box(
                target
                    .chunks(DIMENSION)
                    .map(|x| x.norm_l2())
                    .collect::<Vec<_>>(),
            )
        });
    });

    run_bench::<Float16Type>(c);

    let target: Float16Array =
        generate_random_array_with_seed::<Float16Type>(TOTAL * DIMENSION, [42; 32]);
    c.bench_function("norm_l2(f16, SIMD)", |b| {
        b.iter(|| {
            black_box(
                target
                    .values()
                    .chunks_exact(DIMENSION)
                    .map(|arr| arr.norm_l2())
                    .collect::<Vec<_>>(),
            )
        });
    });

    run_bench::<Float32Type>(c);
    // 1M of 1024 D vectors. 4GB in memory.
    let target: Float32Array =
        generate_random_array_with_seed::<Float32Type>(TOTAL * DIMENSION, [42; 32]);
    c.bench_function("norm_l2(f32, SIMD)", |b| {
        b.iter(|| {
            black_box(
                target
                    .values()
                    .chunks_exact(DIMENSION)
                    .map(|arr| arr.norm_l2())
                    .collect::<Vec<_>>(),
            )
        });
    });

    run_bench::<Float64Type>(c);
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_distance);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_distance);

criterion_main!(benches);
