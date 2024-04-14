// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark of HNSW graph.
//!
//!

use std::{collections::HashSet, sync::Arc, time::Duration};

use arrow_array::types::Float32Type;
use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use lance_index::vector::{
    graph::memory::InMemoryVectorStorage,
    hnsw::builder::{HNSWBuilder, HnswBuildParams},
};
use lance_linalg::{distance::MetricType, MatrixView};
use lance_testing::datagen::generate_random_array_with_seed;

fn bench_hnsw(c: &mut Criterion) {
    const DIMENSION: usize = 1024;
    const TOTAL: usize = 1024 * 1024;
    const SEED: [u8; 32] = [42; 32];
    const K: usize = 10;

    let data = generate_random_array_with_seed::<Float32Type>(TOTAL * DIMENSION, SEED);
    let mat = Arc::new(MatrixView::<Float32Type>::new(data.into(), DIMENSION));
    let vectors = Arc::new(InMemoryVectorStorage::new(mat.clone(), MetricType::L2));

    let query = mat.row(0).unwrap();
    c.bench_function(
        format!("create_hnsw({TOTAL}x1024,levels=6)").as_str(),
        |b| {
            // b.iter(|| {
            //     let hnsw = HNSWBuilder::with_params(
            //         HnswBuildParams::default().max_level(6),
            //         vectors.clone(),
            //     )
            //     .build()
            //     .unwrap();
            //     let uids: HashSet<u32> = hnsw
            //         .search(query, K, 300, None)
            //         .unwrap()
            //         .iter()
            //         .map(|node| node.id)
            //         .collect();

            //     assert_eq!(uids.len(), K);
            // })
        },
    );
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_hnsw);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10);
    targets = bench_hnsw);

criterion_main!(benches);
