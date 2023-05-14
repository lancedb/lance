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

//! Before running the dataset, prepare a "test.lance" dataset, in the
//! `lance/rust` directory. There is no limitation in the dataset size,
//! schema, or content.
//!
//! Run benchmark.
//! ```
//! cargo bench --bench scan
//! ```.
//!
//! TODO: Take parameterized input to specify dataset URI from command line.

use arrow_array::{
    BinaryArray, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchReader,
    StringArray,
};
use arrow_schema::{DataType, Field, FieldRef, Schema as ArrowSchema};
use criterion::{criterion_group, criterion_main, Criterion};
use futures::stream::TryStreamExt;
use lance::arrow::FixedSizeListArrayExt;
use lance::arrow::RecordBatchBuffer;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use std::sync::Arc;

use lance::dataset::{Dataset, WriteMode, WriteParams};

fn bench_scan(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        create_file(std::path::Path::new("./test.lance"), WriteMode::Create).await
    });
    let dataset = rt.block_on(async { Dataset::open("./test.lance").await.unwrap() });

    c.bench_function("Scan full dataset", |b| {
        b.to_async(&rt).iter(|| async {
            let count = dataset
                .scan()
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            assert!(count.len() >= 1);
        })
    });

    std::fs::remove_dir_all("./test.lance").unwrap();
}

async fn create_file(path: &std::path::Path, mode: WriteMode) {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("i", DataType::Int32, false),
        Field::new("f", DataType::Float32, false),
        Field::new("s", DataType::Utf8, false),
        Field::new(
            "fsl",
            DataType::FixedSizeList(
                FieldRef::new(Field::new("item", DataType::Float32, true)),
                2,
            ),
            false,
        ),
        Field::new("blob", DataType::Binary, false),
    ]));
    let num_rows = 100_000;
    let batch_size = 10;
    let batches = RecordBatchBuffer::new(
        (0..(num_rows / batch_size) as i32)
            .map(|i| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from_iter_values(
                            i * batch_size..(i + 1) * batch_size,
                        )),
                        Arc::new(Float32Array::from_iter_values(
                            (i * batch_size..(i + 1) * batch_size)
                                .map(|x| x as f32)
                                .collect::<Vec<_>>(),
                        )),
                        Arc::new(StringArray::from_iter_values(
                            (i * batch_size..(i + 1) * batch_size)
                                .map(|x| format!("s-{}", x).to_string())
                                .collect::<Vec<_>>(),
                        )),
                        Arc::new(
                            FixedSizeListArray::try_new(
                                Float32Array::from_iter_values(
                                    (i * batch_size..(i + 2) * batch_size)
                                        .map(|x| (batch_size + (x - batch_size) / 2) as f32)
                                        .collect::<Vec<_>>(),
                                ),
                                2,
                            )
                            .unwrap(),
                        ),
                        Arc::new(BinaryArray::from_iter_values(
                            (i * batch_size..(i + 1) * batch_size)
                                .map(|x| format!("blob-{}", x).to_string().into_bytes())
                                .collect::<Vec<_>>(),
                        )),
                    ],
                )
                .unwrap()
            })
            .collect(),
    );

    let test_uri = path.to_str().unwrap();
    let mut write_params = WriteParams::default();
    write_params.max_rows_per_file = 40;
    write_params.max_rows_per_group = 10;
    write_params.mode = mode;
    let mut reader: Box<dyn RecordBatchReader> = Box::new(batches);
    Dataset::write(&mut reader, test_uri, Some(write_params))
        .await
        .unwrap();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_scan);
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_scan);
criterion_main!(benches);
