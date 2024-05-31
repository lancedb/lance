// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark of HNSW graph.
//!
//!

use std::{collections::HashSet, sync::Arc, time::Duration};

use arrow_array::{types::Float32Type, FixedSizeListArray};
use criterion::{criterion_group, criterion_main, Criterion};
use lance_arrow::FixedSizeListArrayExt;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use lance_index::vector::{
    flat::storage::FlatStorage,
    hnsw::builder::{HnswBuildParams, HNSW},
};
use lance_linalg::distance::DistanceType;
use lance_testing::datagen::generate_random_array_with_seed;

fn bench_hnsw(c: &mut Criterion) {
    const DIMENSION: usize = 512;
    const TOTAL: usize = 10 * 1024;
    const SEED: [u8; 32] = [42; 32];
    const K: usize = 10;

    let rt = tokio::runtime::Runtime::new().unwrap();

    let data = generate_random_array_with_seed::<Float32Type>(TOTAL * DIMENSION, SEED);
    let fsl = FixedSizeListArray::try_new_from_values(data, DIMENSION as i32).unwrap();
    let vectors = Arc::new(FlatStorage::new(fsl.clone(), DistanceType::L2));

    let query = fsl.value(0);
    c.bench_function(
        format!("create_hnsw({TOTAL}x{DIMENSION},levels=6)").as_str(),
        |b| {
            b.to_async(&rt).iter(|| async {
                let hnsw = HNSW::build_with_storage(
                    DistanceType::L2,
                    HnswBuildParams::default().max_level(6),
                    vectors.clone(),
                )
                .await
                .unwrap();
                let uids: HashSet<u32> = hnsw
                    .search_basic(query.clone(), K, 300, None, vectors.as_ref())
                    .unwrap()
                    .iter()
                    .map(|node| node.id)
                    .collect();

                assert_eq!(uids.len(), K);
            })
        },
    );

    let hnsw = rt
        .block_on(HNSW::build_with_storage(
            DistanceType::L2,
            HnswBuildParams::default().max_level(6),
            vectors.clone(),
        ))
        .unwrap();
    c.bench_function(
        format!("search_hnsw{TOTAL}x{DIMENSION}, levels=6").as_str(),
        |b| {
            b.to_async(&rt).iter(|| async {
                let uids: HashSet<u32> = hnsw
                    .search_basic(query.clone(), K, 300, None, vectors.as_ref())
                    .unwrap()
                    .iter()
                    .map(|node| node.id)
                    .collect();

                assert_eq!(uids.len(), K);
            })
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
