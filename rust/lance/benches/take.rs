// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{
    BinaryArray, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchIterator,
};
use arrow_schema::{DataType, Field, FieldRef, Schema as ArrowSchema};
use criterion::{criterion_group, criterion_main, Criterion};

use lance::{
    arrow::FixedSizeListArrayExt,
    dataset::{builder::DatasetBuilder, ProjectionRequest},
};
use lance_file::version::LanceFileVersion;
use lance_table::io::commit::RenameCommitHandler;
use object_store::ObjectStore;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use rand::Rng;
use std::sync::Arc;
#[cfg(target_os = "linux")]
use std::time::Duration;
use url::Url;

use lance::dataset::{Dataset, WriteMode, WriteParams};

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

fn bench_random_take(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();
    let num_batches = 1024;

    for file_size in [1024 * 1024, 1024] {
        let dataset = rt.block_on(create_dataset(
            "memory://test.lance",
            LanceFileVersion::Legacy,
            num_batches,
            file_size,
        ));
        let schema = Arc::new(dataset.schema().clone());

        for num_rows in [1, 10, 100, 1000] {
            c.bench_function(&format!(
                "V1 Random Take ({file_size} file size, {num_batches} batches, {num_rows} rows per take)"
            ), |b| {
                b.to_async(&rt).iter(|| async {
                    let rows = gen_ranges(num_batches as u64 * BATCH_SIZE, file_size as u64, num_rows);
                    let batch = dataset
                        .take_rows(&rows, ProjectionRequest::Schema(schema.clone()))
                        .await
                        .unwrap_or_else(|_| panic!("rows: {:?}", rows));
                    assert_eq!(batch.num_rows(), num_rows);
                })
            });
        }

        let dataset = rt.block_on(create_dataset(
            "memory://test.lance",
            LanceFileVersion::Stable,
            num_batches,
            file_size,
        ));
        let schema = Arc::new(dataset.schema().clone());
        for num_rows in [1, 10, 100, 1000] {
            c.bench_function(&format!(
                "V2 Random Take ({file_size} file size, {num_batches} batches, {num_rows} rows per take)"
            ), |b| {
                b.to_async(&rt).iter(|| async {
                    let batch = dataset
                        .take_rows(&gen_ranges(num_batches as u64 * BATCH_SIZE, file_size as u64, num_rows), ProjectionRequest::Schema(schema.clone()))
                        .await
                        .unwrap();
                    assert_eq!(batch.num_rows(), num_rows);
                })
            });
        }
    }
}

async fn create_dataset(
    path: &str,
    data_storage_version: LanceFileVersion,
    num_batches: i32,
    file_size: i32,
) -> Dataset {
    let store = create_file(
        std::path::Path::new(path),
        WriteMode::Create,
        data_storage_version,
        num_batches,
        file_size,
    )
    .await;

    DatasetBuilder::from_uri(path)
        .with_object_store(
            store,
            Url::parse(path).unwrap(),
            Arc::new(RenameCommitHandler),
        )
        .load()
        .await
        .unwrap()
}

async fn create_file(
    path: &std::path::Path,
    mode: WriteMode,
    data_storage_version: LanceFileVersion,
    num_batches: i32,
    file_size: i32,
) -> Arc<dyn ObjectStore> {
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

    let test_uri = path.to_str().unwrap();
    let write_params = WriteParams {
        max_rows_per_file: file_size as usize,
        max_rows_per_group: batch_size as usize,
        mode,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let ds = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();
    ds.object_store.inner.clone()
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
