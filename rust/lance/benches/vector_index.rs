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

use std::sync::Arc;

use arrow_array::{
    cast::as_primitive_array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
};
use arrow_schema::{DataType, Field, FieldRef, Schema as ArrowSchema};
use codspeed_criterion_compat::{criterion_group, criterion_main, Criterion};
use futures::TryStreamExt;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use rand::{self, Rng};

use lance::dataset::{WriteMode, WriteParams};
use lance::index::vector::ivf::IvfBuildParams;
use lance::index::vector::pq::PQBuildParams;
use lance::index::vector::VectorIndexParams;
use lance::index::{DatasetIndexExt, IndexType};
use lance::{arrow::as_fixed_size_list_array, dataset::Dataset};
use lance_arrow::FixedSizeListArrayExt;
use lance_linalg::distance::MetricType;

fn bench_ivf_pq_index(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        create_file(std::path::Path::new("./vec_data.lance"), WriteMode::Create).await
    });
    let dataset = rt.block_on(async { Dataset::open("./vec_data.lance").await.unwrap() });
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
                assert!(!results.is_empty());
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
                assert!(!results.is_empty());
            })
        },
    );
}

async fn create_file(path: &std::path::Path, mode: WriteMode) {
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(
            FieldRef::new(Field::new("item", DataType::Float32, true)),
            128,
        ),
        false,
    )]));

    let num_rows = 100_000;
    let batch_size = 10000;
    let batches: Vec<RecordBatch> = (0..(num_rows / batch_size))
        .map(|_| {
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        create_float32_array(num_rows * 128),
                        128,
                    )
                    .unwrap(),
                )],
            )
            .unwrap()
        })
        .collect();

    let test_uri = path.to_str().unwrap();
    std::fs::remove_dir_all(test_uri).map_or_else(|_| println!("{} not exists", test_uri), |_| {});
    let write_params = WriteParams {
        max_rows_per_file: num_rows as usize,
        max_rows_per_group: batch_size as usize,
        mode,
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();
    let ivf_params = IvfBuildParams {
        num_partitions: 32,
        ..Default::default()
    };
    let pq_params = PQBuildParams {
        num_bits: 8,
        num_sub_vectors: 16,
        use_opq: false,
        ..Default::default()
    };
    let m_type = MetricType::L2;
    let params = VectorIndexParams::with_ivf_pq_params(m_type, ivf_params, pq_params);
    dataset
        .create_index(
            vec!["vector"].as_slice(),
            IndexType::Vector,
            Some("ivf_pq_index".to_string()),
            &params,
            true,
        )
        .await
        .unwrap();
}

fn create_float32_array(num_elements: i32) -> Float32Array {
    // generate an Arrow Float32Array with 10000*128 elements randomly
    let mut rng = rand::thread_rng();
    let mut values = Vec::with_capacity(num_elements as usize);
    for _ in 0..num_elements {
        values.push(rng.gen_range(0.0..1.0));
    }
    Float32Array::from(values)
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
