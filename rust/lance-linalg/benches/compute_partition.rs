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

use std::sync::Arc;

use arrow_array::types::Float32Type;
use criterion::{criterion_group, criterion_main, Criterion};
use lance_linalg::{distance::MetricType, kmeans::compute_partitions};
use lance_testing::datagen::generate_random_array_with_seed;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

fn bench_compute_partitions(c: &mut Criterion) {
    const K: usize = 1024 * 4;
    const DIMENSION: usize = 1536;
    const INPUT_SIZE: usize = 10240;
    const SEED: [u8; 32] = [42; 32];

    let centroids = Arc::new(generate_random_array_with_seed::<Float32Type>(
        K * DIMENSION,
        SEED,
    ));
    let input = Arc::new(generate_random_array_with_seed::<Float32Type>(
        INPUT_SIZE * DIMENSION,
        SEED,
    ));

    c.bench_function("compute_centroids(L2)", |b| {
        b.iter(|| {
            compute_partitions::<Float32Type>(
                centroids.clone(),
                input.clone(),
                DIMENSION,
                MetricType::L2,
            )
        })
    });

    c.bench_function("compute_centroids(Cosine)", |b| {
        b.iter(|| {
            compute_partitions::<Float32Type>(
                centroids.clone(),
                input.clone(),
                DIMENSION,
                MetricType::Cosine,
            )
        })
    });
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default()
    .sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_compute_partitions);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().sample_size(10);
    targets = bench_compute_partitions);

criterion_main!(benches);
