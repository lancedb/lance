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

use arrow_array::types::{Float32Type, UInt32Type};
use arrow_array::{
    cast::as_primitive_array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchReader,
};
use arrow_array::{PrimitiveArray, UInt32Array};
use arrow_schema::{DataType, Field, FieldRef, Schema as ArrowSchema};
use arrow_select::concat::concat_batches;
use criterion::{criterion_group, criterion_main, Criterion};
use futures::TryStreamExt;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use rand::{self, Rng};
use std::collections::HashSet;
use std::sync::Arc;

use lance::arrow::{FixedSizeListArrayExt, RecordBatchBuffer};
use lance::dataset::{WriteMode, WriteParams};
use lance::index::vector::ivf::IvfBuildParams;
use lance::index::vector::pq::PQBuildParams;
use lance::index::vector::{MetricType, VectorIndexParams};
use lance::index::{DatasetIndexExt, IndexType};
use lance::utils::testing::generate_random_array_with_seed;
use lance::{arrow::as_fixed_size_list_array, dataset::Dataset};

// Benchmarks on a single file, comparing flat and ivf_pq index.
fn bench_ivf_pq_index(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();
    let dataset = rt.block_on(async {
        create_dataset(
            std::path::Path::new("./vec_data.lance"),
            WriteMode::Create,
            Default::default(),
        )
        .await
    });
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
                let results = vector_query(&dataset, q, 10, 10, 2).await;
                assert!(results.len() >= 1);
            })
        },
    );
}

async fn vector_query(
    dataset: &Dataset,
    q: &PrimitiveArray<Float32Type>,
    top_k: usize,
    nprobes: usize,
    refine: u32,
) -> Vec<RecordBatch> {
    dataset
        .scan()
        .nearest("vector", q, top_k)
        .unwrap()
        .nprobs(nprobes)
        .refine(refine)
        .with_row_id()
        .project(&["id", "vector"])
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap()
}

// Benchmarks on multiple files, comparing different amounts of deletions.
fn bench_ivf_pq_index_deletions(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    // Create a multi-file dataset
    let config = DatasetConfig {
        num_batches: 100,
        batch_size: 10_000,
        num_files: 10,
        ..Default::default()
    };
    let mut dataset = rt.block_on(async {
        create_dataset(
            std::path::Path::new("./vec_data.lance"),
            WriteMode::Create,
            config.clone(),
        )
        .await
    });

    // Choose a particular vector we will search for
    let q = rt.block_on(async {
        let q = dataset.take(&[42000], dataset.schema()).await.unwrap()["vector"].clone();
        let q = as_fixed_size_list_array(&q).value(0);
        as_primitive_array(&q).clone()
    });

    async fn run_query(dataset: &Dataset, q: &PrimitiveArray<Float32Type>) -> Vec<RecordBatch> {
        vector_query(&dataset, &q, 10, 10, 2).await
    }

    // Figure out which ids are in the result, so we know if we delete whether
    // we are deleting ones in the result set or not.
    let results = rt.block_on(async { run_query(&dataset, &q).await });
    let results = concat_batches(&results[0].schema(), results.iter()).unwrap();
    let id_col: &PrimitiveArray<UInt32Type> = as_primitive_array(results["id"].as_ref());
    let id_col: HashSet<u32> = id_col.iter().map(|id| id.unwrap()).collect();

    // Run query
    c.bench_function(format!("Ivf_PQ_Refine/deletions/none").as_str(), |b| {
        b.to_async(&rt).iter(|| run_query(&dataset, &q))
    });

    // Delete some rows, only in one fragment
    let id_col_ref = &id_col;
    let mut dataset = rt.block_on(async move {
        let ids_to_delete = (0..config.batch_size as u32)
            .filter(|id| !id_col_ref.contains(id))
            .map(|id| format!("{}", id))
            .take(20)
            .collect::<Vec<String>>()
            .join(", ");

        // Delete a few random values
        dataset
            .delete(&format!("id in ({ids_to_delete})"))
            .await
            .unwrap();

        dataset
    });

    // Run query again
    c.bench_function(
        format!("Ivf_PQ_Refine/deletions/irrelevant").as_str(),
        |b| {
            b.to_async(&rt).iter(|| async {
                let results = vector_query(&dataset, &q, 10, 10, 2).await;
                assert!(results.len() >= 1);
            })
        },
    );

    // Delete some rows inside the query
    let dataset = rt.block_on(async move {
        // Delete a few random values
        let ids_to_delete = id_col
            .iter()
            .take(10)
            .map(|id| format!("{}", id))
            .collect::<Vec<String>>()
            .join(", ");

        dbg!(&ids_to_delete);
        // Delete a few random values
        dataset
            .delete(&format!("id in ({ids_to_delete})"))
            .await
            .unwrap();

        dataset
    });

    // Run query again
    c.bench_function(format!("Ivf_PQ_Refine/deletions/few").as_str(), |b| {
        b.to_async(&rt).iter(|| async {
            let results = vector_query(&dataset, &q, 10, 10, 2).await;
            assert!(results.len() >= 1);
        })
    });
}

#[derive(Clone)]
struct DatasetConfig {
    num_batches: usize,
    batch_size: usize,
    num_files: usize,
    seed: [u8; 32],
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            num_batches: 10,
            batch_size: 100_000,
            num_files: 1,
            seed: [42; 32],
        }
    }
}

async fn create_dataset(path: &std::path::Path, mode: WriteMode, config: DatasetConfig) -> Dataset {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(
            "vector",
            DataType::FixedSizeList(
                FieldRef::new(Field::new("item", DataType::Float32, true)),
                128,
            ),
            false,
        ),
        Field::new("id", DataType::UInt32, false),
    ]));

    let batches = RecordBatchBuffer::new(
        (0..config.num_batches as u32)
            .map(|batch_i| {
                let vectors = FixedSizeListArray::try_new(
                    generate_random_array_with_seed(config.batch_size * 128, config.seed),
                    128,
                )
                .unwrap();
                let ids = (batch_i * config.batch_size as u32
                    ..(batch_i + 1) * config.batch_size as u32)
                    .collect::<UInt32Array>();
                RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors), Arc::new(ids)])
                    .unwrap()
            })
            .collect(),
    );

    let test_uri = path.to_str().unwrap();
    std::fs::remove_dir_all(test_uri).map_or_else(|_| println!("{} not exists", test_uri), |_| {});
    let mut write_params = WriteParams::default();
    write_params.max_rows_per_file =
        config.batch_size * config.num_batches / config.num_files as usize;
    write_params.max_rows_per_group = config.batch_size as usize;
    write_params.mode = mode;
    let mut reader: Box<dyn RecordBatchReader> = Box::new(batches);
    let dataset = Dataset::write(&mut reader, test_uri, Some(write_params))
        .await
        .unwrap();
    let mut ivf_params = IvfBuildParams::default();
    ivf_params.num_partitions = 32;
    let mut pq_params = PQBuildParams::default();
    pq_params.num_bits = 8;
    pq_params.num_sub_vectors = 16;
    pq_params.use_opq = false;
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
        .unwrap()
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_ivf_pq_index, bench_ivf_pq_index_deletions);
// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_ivf_pq_index, bench_ivf_pq_index_deletions);
criterion_main!(benches);
