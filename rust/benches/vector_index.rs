// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow_array::{cast::as_primitive_array, Float32Array};
use criterion::{criterion_group, criterion_main, Criterion};
use futures::TryStreamExt;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use rand::{self, Rng};

use lance::dataset::WriteMode;
use lance::index::vector::MetricType;
use lance::utils::testing::{create_file, create_vector_index};
use lance::{arrow::as_fixed_size_list_array, dataset::Dataset};

fn bench_ivf_pq_index(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();
    let path = "./vec_data.lance";
    rt.block_on(async {
        create_file(
            std::path::Path::new(path),
            WriteMode::Overwrite,
            128,
            100_000,
            10_000,
            [42; 32],
        )
        .await;
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
    });
    let dataset = rt.block_on(async { Dataset::open(path).await.unwrap() });
    let first_batch = rt.block_on(async {
        dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_next()
            .await
            .unwrap()
            .unwrap()
    });

    let mut rng = rand::thread_rng();
    let vector_column = first_batch.column_by_name("vector").unwrap();
    let value =
        as_fixed_size_list_array(&vector_column).value(rng.gen_range(0..vector_column.len()));
    let q: &Float32Array = as_primitive_array(&value);

    c.bench_function(
        format!("Flat_Index(d={},top_k=10,nprobes=10)", q.len()).as_str(),
        |b| {
            b.to_async(&rt).iter(|| async {
                let results = dataset
                    .scan()
                    .nearest("vector", q, 10)
                    .unwrap()
                    .nprobs(10)
                    .try_into_stream()
                    .await
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
                assert!(results.len() >= 1);
            })
        },
    );

    c.bench_function(
        format!("Ivf_PQ_Refine(d={},top_k=10,nprobes=10, refine=2)", q.len()).as_str(),
        |b| {
            b.to_async(&rt).iter(|| async {
                let results = dataset
                    .scan()
                    .nearest("vector", q, 10)
                    .unwrap()
                    .nprobs(10)
                    .refine(2)
                    .try_into_stream()
                    .await
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
                assert!(results.len() >= 1);
            })
        },
    );
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(32)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_ivf_pq_index);
// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_ivf_pq_index);
criterion_main!(benches);
