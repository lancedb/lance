// Copyright 2024 Lance Developers.
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

//! Benchmark of HNSW graph.
//!
//!

use std::{collections::HashSet, time::Duration};

use arrow_array::types::Float32Type;
use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use lance_index::vector::hnsw::builder::HNSWBuilder;
use lance_linalg::MatrixView;
use lance_testing::datagen::generate_random_array_with_seed;

fn bench_hnsw(c: &mut Criterion) {
    const DIMENSION: usize = 1024;
    const TOTAL: usize = 65536;
    const SEED: [u8; 32] = [42; 32];
    const K: usize = 10;

    let data = generate_random_array_with_seed::<Float32Type>(TOTAL * DIMENSION, SEED);
    let mat = MatrixView::<Float32Type>::new(data.into(), DIMENSION);

    let query = mat.row(0).unwrap();
    c.bench_function("create_hnsw(65535x1024,levels=4)", |b| {
        b.iter(|| {
            let hnsw = HNSWBuilder::new(mat.clone()).max_level(4).build().unwrap();
            let uids: HashSet<u32> = hnsw
                .search(query, K, 300)
                .unwrap()
                .iter()
                .map(|(i, _)| *i)
                .collect();

            assert_eq!(uids.len(), K);
        })
    });
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(32)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_hnsw);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(32);
    targets = bench_hnsw);

criterion_main!(benches);
