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

use rand::Rng;
use std::collections::HashSet;
use std::env::current_dir;

use arrow_array::cast::as_primitive_array;
use arrow_array::types::UInt64Type;
use arrow_array::RecordBatch;
use criterion::{criterion_group, criterion_main, Criterion};

use futures::StreamExt;
use lance::arrow::*;
use lance::dataset::Dataset;
use lance::index::ann::{FlatIndex, IvfPQIndex, SearchParams};
use pprof::criterion::{Output, PProfProfiler};

fn compute_recall(predicts: &RecordBatch, ground_truth: &RecordBatch) -> f32 {
    let pred_rows = predicts.column_with_name("_rowid").unwrap();
    let gt_rows = ground_truth.column_with_name("_rowid").unwrap();
    let pred_row_vec = as_primitive_array::<UInt64Type>(&pred_rows).values();
    let gt_row_vec = as_primitive_array::<UInt64Type>(&gt_rows).values();

    let gt_row_set: HashSet<u64, _> = HashSet::<u64>::from_iter(gt_row_vec.iter().copied());
    let true_positive = pred_row_vec
        .iter()
        .filter(|p| gt_row_set.contains(p))
        .count() as f32;
    true_positive / (gt_row_set.len() as f32)
}

fn bench_search(c: &mut Criterion) {
    const NUM_THREADS: usize = 16;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(NUM_THREADS)
        .build()
        .unwrap();

    c.bench_function("vec-ivf_pq-index_top_100", move |b| {
        let mut rand = rand::thread_rng();
        let column = "vector".to_string();

        let dataset_uri = current_dir().unwrap().join("vec_data");
        let dataset = runtime.block_on(async {
            Dataset::open(dataset_uri.as_path().to_str().unwrap())
                .await
                .unwrap()
        });

        let (first_batch, index) = runtime.block_on(async {
            let mut stream = dataset.scan().project(&[&column]).unwrap().into_stream();
            let first_batch = stream.next().await.unwrap().unwrap();

            (
                first_batch.column(0).clone(),
                IvfPQIndex::open(&dataset, &column).await.unwrap(),
            )
        });

        let key =
            as_fixed_size_list_array(&first_batch).value(rand.gen_range(0..first_batch.len()));
        let key = as_primitive_array(&key).clone();

        let params = SearchParams {
            key,
            k: 100,
            nprob: 60,
        };

        b.to_async(&runtime).iter(|| async {
            let b = index.search(&params).await.unwrap();
            assert_eq!(b.num_rows(), 100);
        });

        // Compute recall
        runtime.block_on(async {
            let flat_index = FlatIndex::new(&dataset, column.clone());
            let ivf_results = index.search(&params).await.unwrap();
            let flat_results = flat_index.search(&params).await.unwrap();
            println!("Recall: {}", compute_recall(&ivf_results, &flat_results))
        });
    });
}

criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10).with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_search);
criterion_main!(benches);
