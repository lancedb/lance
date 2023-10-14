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

use std::sync::Arc;

use arrow_array::{Array, UInt8Array};
use criterion::{criterion_group, criterion_main, Criterion};

use lance::index::{
    prefilter::PreFilter,
    vector::{pq::PQIndex, Query, VectorIndex},
};
use lance_index::vector::pq::{PQBuildParams, ProductQuantizer};
use lance_linalg::{distance::MetricType, MatrixView};
use lance_testing::datagen::{generate_random_array, generate_random_indices};
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

fn bench_pq_search(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();

    let data = MatrixView::random(100_000, 1536);

    println!("Training PQ");
    let pq = Arc::new(
        rt.block_on(ProductQuantizer::train(&data, &PQBuildParams::new(192, 8)))
            .unwrap(),
    );
    println!("Done training PQ");

    for size in vec![1000, 10000, 100000, 1000000] {
        let data = MatrixView::random(size, 1536);
        let codes = rt.block_on(pq.transform(&data)).unwrap();
        let mut pq_index = PQIndex::new(pq.clone(), MetricType::L2);
        let code_arr = UInt8Array::from_iter_values(
            codes
                .values()
                .as_any()
                .downcast_ref::<UInt8Array>()
                .unwrap()
                .values()
                .iter()
                .map(|x| *x),
        );
        pq_index.code = Some(Arc::new(code_arr));
        let row_ids = generate_random_indices(size);
        pq_index.row_ids = Some(Arc::new(row_ids));

        c.bench_function(
            format!("PQ192, 1536D, {} rows, 10 neighbors", size).as_str(),
            |b| {
                b.iter(|| {
                    let query = Query {
                        column: "".to_string(),
                        /// The vector to be searched.
                        key: Arc::new(generate_random_array(1536)),
                        k: 10,
                        // doesn't matter in pq
                        nprobes: 0,
                        // doesn't matter in pq
                        refine_factor: None,
                        // doesn't matter in pq
                        metric_type: MetricType::L2,
                        // doesn't matter in pq
                        use_index: true,
                    };
                    rt.block_on(pq_index.search(&query, &PreFilter::new_empty()))
                        .unwrap();
                })
            },
        );
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_pq_search);

criterion_main!(benches);
