// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::FixedSizeListArray;
use criterion::{criterion_group, criterion_main, Criterion};

use lance_arrow::FixedSizeListArrayExt;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use lance_linalg::{
    distance::DistanceType,
    kmeans::{compute_partitions_arrow_array, KMeans},
};
use lance_testing::datagen::generate_random_array;

fn bench_train(c: &mut Criterion) {
    let params = [
        (64 * 1024, 8),    // training PQ
        (64 * 1024, 128),  // training IVF with small vectors (1M rows)
        (64 * 1024, 1024), // training IVF with large vectors (1M rows)
    ];
    for (n, dimension) in params {
        let k = n / 256;

        let values = generate_random_array(n * dimension as usize);
        let data = FixedSizeListArray::try_new_from_values(values, dimension).unwrap();

        let values = generate_random_array(k * dimension as usize);
        let centroids = FixedSizeListArray::try_new_from_values(values, dimension).unwrap();

        c.bench_function(&format!("train_{}d_{}k", dimension, n / 1024), |b| {
            b.iter(|| {
                KMeans::new(&data, k, 50).ok().unwrap();
            })
        });

        c.bench_function(
            &format!("compute_membership_{}d_{}k", dimension, n / 1024),
            |b| b.iter(|| compute_partitions_arrow_array(&centroids, &data, DistanceType::L2)),
        );
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
    .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_train);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_train);
criterion_main!(benches);
