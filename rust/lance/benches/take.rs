// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{
    BinaryArray, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchIterator,
};
use arrow_schema::{DataType, Field, FieldRef, Schema as ArrowSchema};
use criterion::{criterion_group, criterion_main, Criterion};

use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance::{arrow::FixedSizeListArrayExt, dataset::ProjectionRequest};
use lance_file::version::LanceFileVersion;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use rand::Rng;
use std::sync::Arc;
#[cfg(target_os = "linux")]
use std::time::Duration;

const BATCH_SIZE: u64 = 1024;

fn gen_ranges(num_rows: u64, file_size: u64, n: usize) -> Vec<u64> {
    let mut rng = rand::thread_rng();
    let mut ranges = Vec::with_capacity(n);
    for i in 0..n {
        ranges.push(rng.gen_range(1..num_rows));
        ranges[i] = ((ranges[i] / file_size) << 32) | (ranges[i] % file_size);
    }

    ranges
}

fn bench_take_for_version(
    c: &mut Criterion,
    rt: &tokio::runtime::Runtime,
    file_size: usize,
    num_batches: usize,
    version: LanceFileVersion,
    version_name: &str,
) {
    let dataset = rt.block_on(create_dataset(
        "memory://test.lance",
        version,
        num_batches as i32,
        file_size as i32,
    ));
    let schema = Arc::new(dataset.schema().clone());

    for num_rows in [1, 10, 100, 1000] {
        c.bench_function(
            &format!(
                "{} Random Take ({} file size, {} batches, {} rows per take)",
                version_name, file_size, num_batches, num_rows
            ),
            |b| {
                b.to_async(rt).iter(|| async {
                    let rows =
                        gen_ranges(num_batches as u64 * BATCH_SIZE, file_size as u64, num_rows);
                    let batch = dataset
                        .take_rows(&rows, ProjectionRequest::Schema(schema.clone()))
                        .await
                        .unwrap_or_else(|_| panic!("rows: {:?}", rows));
                    assert_eq!(batch.num_rows(), num_rows);
                })
            },
        );
    }
}

fn bench_random_take(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let num_batches = 1024;

    for file_size in [1024 * 1024, 1024] {
        bench_take_for_version(
            c,
            &rt,
            file_size,
            num_batches,
            LanceFileVersion::V2_0,
            "V2_0",
        );
        bench_take_for_version(
            c,
            &rt,
            file_size,
            num_batches,
            LanceFileVersion::V2_1,
            "V2_1",
        );
    }
}

async fn create_dataset(
    path: &str,
    data_storage_version: LanceFileVersion,
    num_batches: i32,
    file_size: i32,
) -> Dataset {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("i", DataType::Int32, false),
        Field::new("f", DataType::Float32, false),
        Field::new("s", DataType::Binary, false),
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
    let batch_size = BATCH_SIZE as i32;
    let batches: Vec<RecordBatch> = (0..num_batches)
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
                    Arc::new(BinaryArray::from_iter_values(
                        (i * batch_size..(i + 1) * batch_size)
                            .map(|x| format!("blob-{}", x).into_bytes()),
                    )),
                    Arc::new(
                        FixedSizeListArray::try_new_from_values(
                            Float32Array::from_iter_values(
                                (i * batch_size..(i + 2) * batch_size)
                                    .map(|x| (batch_size + (x - batch_size) / 2) as f32),
                            ),
                            2,
                        )
                        .unwrap(),
                    ),
                    Arc::new(BinaryArray::from_iter_values(
                        (i * batch_size..(i + 1) * batch_size)
                            .map(|x| format!("blob-{}", x).into_bytes()),
                    )),
                ],
            )
            .unwrap()
        })
        .collect();

    let write_params = WriteParams {
        max_rows_per_file: file_size as usize,
        max_rows_per_group: batch_size as usize,
        mode: WriteMode::Create,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    Dataset::write(reader, path, Some(write_params))
        .await
        .unwrap()
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .significance_level(0.01)
        .sample_size(10000)
        .warm_up_time(Duration::from_secs_f32(3.0))
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_random_take);
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_random_take);
criterion_main!(benches);
