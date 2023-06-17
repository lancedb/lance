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

use arrow_array::{FixedSizeListArray, RecordBatch};
use arrow_schema::{DataType, Field, FieldRef, Schema};
use criterion::{criterion_group, criterion_main, Criterion};
use lance::{arrow::*, dataset::WriteMode};
use pprof::criterion::{Output, PProfProfiler};

async fn create_file(path: &std::path::Path, dim: usize, mode: WriteMode) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(
            FieldRef::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
        ),
        false,
    )]));

    let num_rows = 100_000;
    let batch_size = 10_000;
    let batches = RecordBatchBuffer::new(
        (0..(num_rows / batch_size) as i32)
            .map(|_| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(
                        FixedSizeListArray::try_new(create_float32_array(num_rows * 128), 128)
                            .unwrap(),
                    )],
                )
                .unwrap()
            })
            .collect(),
    );
}

fn bench_ivf_pq_index(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();

    rt.block_on(async {
        create_file(
            std::path::Path::new("./ivf_pq_768d.lance"),
            768,
            WriteMode::Create,
        )
        .await
    });
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_ivf_pq_index);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_ivf_pq_index);

criterion_main!(benches);
