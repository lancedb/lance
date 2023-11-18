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

use arrow_array::{
    types::{Float16Type, Float32Type, Float64Type},
    Float32Array,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lance_arrow::{ArrowFloatType, FloatArray};
use num_traits::Float;

use lance_linalg::distance::cosine::{cosine_distance_batch, Cosine};
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use lance_testing::datagen::generate_random_array_with_seed;

fn cosine_scalar<T: Float>(x: &[T], y: &[T], dim: usize) -> Vec<T> {
    y.chunks_exact(dim)
        .map(|vec| {
            let mut dot = T::zero();
            let mut x_norm = T::zero();
            let mut y_norm = T::zero();

            for (&xi, &yi) in x.iter().zip(vec.iter()) {
                dot = dot + xi * yi;
                x_norm = x_norm + xi * xi;
                y_norm = y_norm + yi * yi;
            }
            dot / (x_norm * y_norm).sqrt()
        })
        .collect()
}

fn run_bench<T: ArrowFloatType + Cosine>(c: &mut Criterion) {
    const DIMENSION: usize = 1024;
    const TOTAL: usize = 1024 * 1024; // 1M vectors

    let type_name = std::any::type_name::<T::Native>();
    let key = generate_random_array_with_seed::<T>(DIMENSION, [0; 32]);
    let target = generate_random_array_with_seed::<T>(TOTAL * DIMENSION, [42; 32]);

    c.bench_function(format!("Cosine({}, scalar)", type_name).as_str(), |b| {
        b.iter(|| {
            black_box(cosine_scalar(key.as_slice(), target.as_slice(), DIMENSION));
        })
    });

    c.bench_function(
        format!("Cosine({}, auto-vectorized)", type_name).as_str(),
        |b| {
            b.iter(|| {
                black_box(
                    cosine_distance_batch::<T::Native>(
                        key.as_slice(),
                        target.as_slice(),
                        DIMENSION,
                    )
                    .collect::<Vec<_>>(),
                );
            })
        },
    );
}

fn bench_distance(c: &mut Criterion) {
    // run_bench::<BFloat16Type>(c);
    run_bench::<Float16Type>(c);
    run_bench::<Float32Type>(c);
    run_bench::<Float64Type>(c);

    let key: Float32Array = generate_random_array_with_seed::<Float32Type>(8, [0; 32]);
    let target: Float32Array =
        generate_random_array_with_seed::<Float32Type>(1024 * 1024 * 8, [42; 32]);

    c.bench_function("Cosine(simd,f32x8) rng seed", |b| {
        b.iter(|| {
            black_box(cosine_distance_batch(key.values(), target.values(), 8).collect::<Vec<_>>())
        })
    });
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
