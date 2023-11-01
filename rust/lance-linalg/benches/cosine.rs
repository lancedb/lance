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

use arrow_array::Float32Array;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use lance_linalg::distance::cosine::cosine_distance_batch;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use lance_testing::datagen::generate_random_array_with_seed;

fn bench_distance(c: &mut Criterion) {
    const DIMENSION: usize = 1024;
    const TOTAL: usize = 1024 * 1024; // 1M vectors

    let key: Float32Array = generate_random_array_with_seed(DIMENSION, [0; 32]);
    // 1M of 1024 D vectors. 4GB in memory.
    let target: Float32Array = generate_random_array_with_seed(TOTAL * DIMENSION, [42; 32]);

    c.bench_function("Cosine(simd)", |b| {
        b.iter(|| {
            black_box(
                cosine_distance_batch(key.values(), target.values(), DIMENSION).collect::<Vec<_>>(),
            );
        })
    });

    let key: Float32Array = generate_random_array_with_seed(DIMENSION, [5; 32]);
    // 1M of 1024 D vectors. 4GB in memory.
    let target: Float32Array = generate_random_array_with_seed(TOTAL * DIMENSION, [7; 32]);

    c.bench_function("Cosine(simd) second rng seed", |b| {
        b.iter(|| {
            black_box(
                cosine_distance_batch(key.values(), target.values(), DIMENSION).collect::<Vec<_>>(),
            )
        })
    });

    let key: Float32Array = generate_random_array_with_seed(8, [0; 32]);
    let target: Float32Array = generate_random_array_with_seed(TOTAL * 8, [42; 32]);

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
