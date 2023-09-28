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

use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, FieldRef, Schema};
use criterion::{criterion_group, criterion_main, Criterion};
use lance::{
    arrow::*,
    dataset::{WriteMode, WriteParams},
    index::{vector::VectorIndexParams, DatasetIndexExt, IndexType},
    Dataset,
};

use lance_linalg::distance::MetricType;
use lance_testing::datagen::generate_random_array;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

async fn create_dataset(path: &std::path::Path, dim: usize, mode: WriteMode) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(
            FieldRef::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
        ),
        false,
    )]));

    let num_rows = 1_000_000;
    let batch_size = 10_000;
    let batches: Vec<RecordBatch> = (0..(num_rows / batch_size) as i32)
        .map(|_| {
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        generate_random_array(batch_size * dim),
                        dim as i32,
                    )
                    .unwrap(),
                )],
            )
            .unwrap()
        })
        .collect();

    let write_params = WriteParams {
        max_rows_per_file: num_rows,
        max_rows_per_group: batch_size,
        mode,
        ..Default::default()
    };

    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    Dataset::write(reader, path.to_str().unwrap(), Some(write_params))
        .await
        .unwrap();
}

fn bench_ivf_pq_index(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();

    const DIM: usize = 768;
    let uri = format!("./ivf_pq_{}d.lance", DIM);
    std::fs::remove_dir_all(&uri).map_or_else(|_| println!("{} not exists", uri), |_| {});

    rt.block_on(async { create_dataset(std::path::Path::new(&uri), DIM, WriteMode::Create).await });

    let dataset = rt.block_on(async { Dataset::open(&uri).await.unwrap() });

    let ivf_partition = 256;
    let pq = 96;

    c.bench_function(
        format!("CreateIVF{},PQ{}(d={})", ivf_partition, pq, DIM).as_str(),
        |b| {
            b.to_async(&rt).iter(|| async {
                let params =
                    VectorIndexParams::ivf_pq(ivf_partition, 8, pq, false, MetricType::L2, 50);

                dataset
                    .clone()
                    .create_index(
                        vec!["vector"].as_slice(),
                        IndexType::Vector,
                        None,
                        &params,
                        true,
                    )
                    .await
                    .unwrap();
            });
        },
    );
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
    config = Criterion::default().significance_level(0.1).sample_size(5);
    targets = bench_ivf_pq_index);

criterion_main!(benches);
