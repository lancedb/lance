// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{types::Float32Type, FixedSizeListArray};
use criterion::{criterion_group, criterion_main, Criterion};

use lance_arrow::{FixedSizeListArrayExt, FloatArray};
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use lance_linalg::{kmeans::KMeans, Clustering};
use lance_testing::datagen::generate_random_array;

fn bench_train(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();

    let dimension: i32 = 128;
    let values = generate_random_array(1024 * 4 * dimension as usize);
    let array = FixedSizeListArray::try_new_from_values(values, dimension).unwrap();

    c.bench_function("train_128d_4k", |b| {
        b.to_async(&rt).iter(|| async {
            KMeans::<Float32Type>::new(&array, 25, 50).ok().unwrap();
        })
    });

    let values = generate_random_array(1024 * 64 * dimension as usize);
    let array = FixedSizeListArray::try_new_from_values(values, dimension).unwrap();
    c.bench_function("train_128d_65535", |b| {
        b.to_async(&rt).iter(|| async {
            KMeans::<Float32Type>::new(&array, 25, 50).ok().unwrap();
        })
    });

    let values = generate_random_array(1024 * 64 * dimension as usize);
    let array = FixedSizeListArray::try_new_from_values(values.clone(), dimension).unwrap();
    c.bench_function("compute_membership_128d_65535", |b| {
        let kmeans = KMeans::<Float32Type>::new(&array, 25, 50).ok().unwrap();

        b.to_async(&rt)
            .iter(|| async { kmeans.compute_membership(values.as_slice(), None) })
    });

    let dimension = 8;
    let values = generate_random_array(1024 * 64 * dimension as usize);
    let array = FixedSizeListArray::try_new_from_values(values, dimension).unwrap();
    c.bench_function("train_8d_65535", |b| {
        b.to_async(&rt).iter(|| async {
            KMeans::<Float32Type>::new(&array, 25, 50).ok().unwrap();
        })
    });
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
