// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::hint::black_box;

use arrow_array::types::Float32Type;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lance_index::vector::kmeans::{KMeansAlgo, KMeansAlgoFloat};
use lance_linalg::distance::DistanceType;
use lance_testing::datagen::generate_random_array;

fn bench_recompute_centroids(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_recompute_centroids");

    let (num_vectors, dimension, k) = (128 * 1024, 128, 256);
    let data = generate_random_array(num_vectors * dimension);
    let membership = (0..num_vectors)
        .map(|row| Some((row % k) as u32))
        .collect::<Vec<_>>();
    let cluster_sizes = vec![num_vectors / k; k];

    group.bench_with_input(
        BenchmarkId::new(format!("{dimension}d_{k}k"), num_vectors),
        &num_vectors,
        |b, _| {
            b.iter(|| {
                let mut cluster_sizes = cluster_sizes.clone();
                black_box(KMeansAlgoFloat::<Float32Type>::to_kmeans(
                    black_box(data.values()),
                    dimension,
                    k,
                    black_box(&membership),
                    &mut cluster_sizes,
                    DistanceType::L2,
                    0.0,
                ))
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_recompute_centroids);
criterion_main!(benches);
