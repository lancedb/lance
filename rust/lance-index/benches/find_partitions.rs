// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::types::Float32Type;
use arrow_array::Float32Array;
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use lance_index::vector::ivf::{Ivf, IvfImpl};
use lance_linalg::{distance::MetricType, MatrixView};
use lance_testing::datagen::generate_random_array_with_seed;

fn bench_partitions(c: &mut Criterion) {
    const DIMENSION: usize = 1536;
    const SEED: [u8; 32] = [42; 32];

    let query: Float32Array = generate_random_array_with_seed::<Float32Type>(DIMENSION, SEED);

    for num_centroids in &[10240, 65536] {
        let centroids = Arc::new(generate_random_array_with_seed::<Float32Type>(
            num_centroids * DIMENSION,
            SEED,
        ));
        let matrix = MatrixView::<Float32Type>::new(centroids.clone(), DIMENSION);

        for k in &[1, 10, 50] {
            let ivf = IvfImpl::new(matrix.clone(), MetricType::L2, "vector", vec![], None);

            c.bench_function(format!("IVF{},k={},L2", num_centroids, k).as_str(), |b| {
                b.iter(|| {
                    let _ = ivf.find_partitions(&query, *k);
                })
            });

            let ivf = IvfImpl::new(matrix.clone(), MetricType::Cosine, "vector", vec![], None);
            c.bench_function(
                format!("IVF{},k={},Cosine", num_centroids, k).as_str(),
                |b| {
                    b.iter(|| {
                        let _ = ivf.find_partitions(&query, *k);
                    })
                },
            );
        }
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets =bench_partitions);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_partitions);

criterion_main!(benches);
