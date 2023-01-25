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

use criterion::{criterion_group, criterion_main, Criterion};
use futures::stream::TryStreamExt;
use pprof::criterion::{Output, PProfProfiler};

use lance::dataset::Dataset;

fn bench_scan(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();

    let dataset = rt.block_on(async { Dataset::open("./test.lance").await.unwrap() });

    c.bench_function("Scan datasets", |b| {
        b.to_async(&rt).iter(|| async {
            let count = dataset
                .scan()
                .into_stream()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            assert!(count.len() >= 1);
        })
    });
}

criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_scan);
criterion_main!(benches);
