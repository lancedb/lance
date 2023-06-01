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

use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use lance::dataset::WriteMode;
use lance::index::vector::MetricType;
use lance::utils::testing::{create_file, create_vector_index};

fn bench_ivf_pq_index_build(c: &mut Criterion) {
    let dims = 1024;
    let num_rows = 100_000;
    let path = "./vec_data.lance";

    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("Ivf_PQ_Build(d=1024, n=100_000)", |b| {
        b.iter_with_setup(
            || {
                rt.block_on(async {
                    create_file(
                        std::path::Path::new(path),
                        WriteMode::Overwrite,
                        dims,
                        num_rows,
                        10_000,
                        [42; 32],
                    )
                    .await
                })
            },
            |()| {
                rt.block_on(async {
                    create_vector_index(
                        path,
                        "vector",
                        "ivf_pq_index",
                        32,
                        8,
                        16,
                        false,
                        MetricType::L2,
                    )
                    .await;
                })
            },
        )
    });
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default()
    .measurement_time(Duration::from_secs(180))
    .significance_level(0.1)
    .sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_ivf_pq_index_build);
// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_ivf_pq_index_build);

criterion_main!(benches);
