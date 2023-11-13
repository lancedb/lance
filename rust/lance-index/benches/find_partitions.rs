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

    let query: Float32Array = generate_random_array_with_seed(DIMENSION, SEED);

    for num_centroids in &[10240, 65536] {
        let centroids = Arc::new(generate_random_array_with_seed::<Float32Type>(
            num_centroids * DIMENSION,
            SEED,
        ));
        let matrix = MatrixView::<Float32Type>::new(centroids.clone(), DIMENSION);

        for k in &[1, 10, 50] {
            let ivf = IvfImpl::new(matrix.clone(), MetricType::L2, vec![], None);

            c.bench_function(format!("IVF{},k={},L2", num_centroids, k).as_str(), |b| {
                b.iter(|| {
                    let _ = ivf.find_partitions(&query, *k);
                })
            });

            let ivf = IvfImpl::new(matrix.clone(), MetricType::Cosine, vec![], None);
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
