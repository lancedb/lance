// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{
    BinaryArray, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchIterator,
    UInt32Array,
};
use arrow_schema::{DataType, Field, FieldRef, Schema as ArrowSchema};
use criterion::{criterion_group, criterion_main, Criterion};

use futures::StreamExt;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance::{arrow::FixedSizeListArrayExt, dataset::ProjectionRequest};
use lance_core::cache::LanceCache;
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::v2::reader::{FileReader, FileReaderOptions};
use lance_file::v2::LanceEncodingsIo;
use lance_file::version::LanceFileVersion;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use lance_io::ReadBatchParams;
use object_store::path::Path;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use rand::Rng;
use std::sync::Arc;
#[cfg(target_os = "linux")]
use std::time::Duration;

const ENV_OBJECT_STORE_SCHEME: &str = "OBJECT_STORE_SCHEME";

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

fn bench_random_take_with_dataset(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let num_batches = 1024;

    for file_size in [1024 * 1024, 1024] {
        dataset_take(
            c,
            &rt,
            file_size,
            num_batches,
            LanceFileVersion::V2_0,
            "V2_0",
        );
        dataset_take(
            c,
            &rt,
            file_size,
            num_batches,
            LanceFileVersion::V2_1,
            "V2_1",
        );
    }
}

fn dataset_take(
    c: &mut Criterion,
    rt: &tokio::runtime::Runtime,
    file_size: usize,
    num_batches: usize,
    version: LanceFileVersion,
    version_name: &str,
) {
    let (dataset_path, _temp_dir) = generate_lance_dataset_path();
    let dataset = rt.block_on(create_dataset(
        &dataset_path,
        version,
        num_batches as i32,
        file_size as i32,
    ));
    let schema = Arc::new(dataset.schema().clone());

    for num_rows in [1, 10, 100, 1000] {
        c.bench_function(
            &format!(
                "{version_name} Random Take Dataset({file_size} file size, {num_batches} batches, {num_rows} rows per take)",
            ), |b| {
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

fn bench_random_single_take_with_file_reader(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();
    let num_batches: u64 = 1024;
    let file_size: u64 = num_batches * BATCH_SIZE + 1;
    let rows_gen = Box::new(|num_batches, file_size, num_rows| {
        let rows = gen_ranges(num_batches * BATCH_SIZE, file_size, num_rows);
        let mut rows_list: Vec<Vec<u32>> = Vec::with_capacity(rows.len());
        for row in rows {
            rows_list.push(vec![row as u32]);
        }
        rows_list
    });

    file_reader_take(
        c,
        &rt,
        file_size,
        num_batches,
        LanceFileVersion::V2_0,
        "V2_0 Single",
        rows_gen.clone(),
    );
    file_reader_take(
        c,
        &rt,
        file_size,
        num_batches,
        LanceFileVersion::V2_1,
        "V2_1 Single",
        rows_gen,
    );
}

fn bench_random_batch_take_with_file_reader(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();
    let num_batches: u64 = 1024;
    let file_size: u64 = num_batches * BATCH_SIZE + 1;
    let rows_gen = Box::new(|num_batches, file_size, num_rows| {
        let rows = gen_ranges(num_batches * BATCH_SIZE, file_size, num_rows);
        let mut rows: Vec<u32> = rows.iter().map(|&x| x as u32).collect();
        rows.sort();
        vec![rows]
    });

    file_reader_take(
        c,
        &rt,
        file_size,
        num_batches,
        LanceFileVersion::V2_0,
        "V2_0 Batch",
        rows_gen.clone(),
    );
    file_reader_take(
        c,
        &rt,
        file_size,
        num_batches,
        LanceFileVersion::V2_1,
        "V2_1 Batch",
        rows_gen,
    );
}

fn file_reader_take(
    c: &mut Criterion,
    rt: &tokio::runtime::Runtime,
    file_size: u64,
    num_batches: u64,
    version: LanceFileVersion,
    version_name: &str,
    rows_gen: Box<dyn Fn(u64, u64, usize) -> Vec<Vec<u32>>>,
) {
    let (dataset, file_path) = rt.block_on(async {
        let (dataset_path, _temp_dir) = generate_lance_dataset_path();
        // Make sure there is only one fragment.
        let dataset =
            create_dataset(&dataset_path, version, num_batches as i32, file_size as i32).await;

        assert_eq!(dataset.get_fragments().len(), 1);
        let fragments = dataset.get_fragments();
        let fragment = fragments.first().unwrap();
        assert_eq!(fragment.num_data_files(), 1);
        let file = fragment.metadata().files.first().unwrap();
        let file_path = dataset.data_dir().child(file.path.as_str());

        (dataset, file_path)
    });

    // Bench random take.
    for num_rows in [1, 10, 100, 1000] {
        c.bench_function(&format!(
            "{version_name} Random Take FileReader({file_size} file size, {num_batches} batches, {num_rows} rows per take)"
        ), |b| {
            b.to_async(rt).iter(|| async {
                let file_reader = create_file_reader(&dataset, &file_path).await;

                let rows_list = rows_gen(num_batches, file_size, num_rows);
                for rows in rows_list {
                    let rows = ReadBatchParams::Indices(UInt32Array::from(rows));
                    let stream = file_reader
                        .read_stream(
                            rows,
                            1024,
                            16,
                            FilterExpression::no_filter(),
                        )
                        .unwrap();
                    stream.fold(Vec::new(), |mut acc, item| async move {
                        acc.push(item);
                        acc
                    }).await;
                }
            })
        });
    }
}

async fn create_file_reader(dataset: &Dataset, file_path: &Path) -> FileReader {
    // Create file reader v2.
    let scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig {
            io_buffer_size_bytes: 2 * 1024 * 1024 * 1024,
        },
    );
    let file = scheduler
        .open_file(file_path, &CachedFileSize::unknown())
        .await
        .unwrap();
    let file_metadata = FileReader::read_all_metadata(&file).await.unwrap();

    FileReader::try_open_with_file_metadata(
        Arc::new(LanceEncodingsIo(file.clone())),
        file_path.clone(),
        None,
        Arc::<DecoderPlugins>::default(),
        Arc::new(file_metadata),
        &LanceCache::no_cache(),
        FileReaderOptions::default(),
    )
    .await
    .unwrap()
}

fn bench_random_single_take_with_file_fragment(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();
    let num_batches: u64 = 1024;
    // Make sure there is only one fragment.
    let file_size: u64 = num_batches * BATCH_SIZE + 1;
    let rows_gen = Box::new(|num_batches, file_size, num_rows| {
        let rows = gen_ranges(num_batches * BATCH_SIZE, file_size, num_rows);
        let mut rows_list: Vec<Vec<u32>> = Vec::with_capacity(rows.len());
        for row in rows {
            rows_list.push(vec![row as u32]);
        }
        rows_list
    });

    fragment_take(
        c,
        &rt,
        file_size,
        num_batches,
        LanceFileVersion::V2_0,
        "V2_0 Single",
        rows_gen.clone(),
    );
    fragment_take(
        c,
        &rt,
        file_size,
        num_batches,
        LanceFileVersion::V2_1,
        "V2_1 Single",
        rows_gen,
    );
}

fn bench_random_batch_take_with_file_fragment(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();
    let num_batches: u64 = 1024;
    // Make sure there is only one fragment.
    let file_size: u64 = num_batches * BATCH_SIZE + 1;
    let rows_gen = Box::new(|num_batches, file_size, num_rows| {
        let rows = gen_ranges(num_batches * BATCH_SIZE, file_size, num_rows);
        let mut rows: Vec<u32> = rows.iter().map(|&x| x as u32).collect();
        rows.sort();
        vec![rows]
    });

    fragment_take(
        c,
        &rt,
        file_size,
        num_batches,
        LanceFileVersion::V2_0,
        "V2_0 Batch",
        rows_gen.clone(),
    );
    fragment_take(
        c,
        &rt,
        file_size,
        num_batches,
        LanceFileVersion::V2_1,
        "V2_1 Batch",
        rows_gen,
    );
}

fn fragment_take(
    c: &mut Criterion,
    rt: &tokio::runtime::Runtime,
    file_size: u64,
    num_batches: u64,
    version: LanceFileVersion,
    version_name: &str,
    rows_gen: Box<dyn Fn(u64, u64, usize) -> Vec<Vec<u32>>>,
) {
    let (dataset_path, _temp_dir) = generate_lance_dataset_path();
    let dataset = rt.block_on(create_dataset(
        &dataset_path,
        version,
        num_batches as i32,
        file_size as i32,
    ));

    assert_eq!(dataset.get_fragments().len(), 1);
    let fragments = dataset.get_fragments();
    let fragment = fragments.first().unwrap();

    // Bench random take.
    for num_rows in [1, 10, 100, 1000] {
        c.bench_function(&format!(
            "{version_name} Random Take Fragment({file_size} file size, {num_batches} batches, {num_rows} rows per take)"
        ), |b| {
            b.to_async(rt).iter(|| async {
                let rows_list = rows_gen(num_batches, file_size, num_rows);
                for rows in rows_list {
                    let _ = fragment.take(rows.as_slice(), dataset.schema()).await;
                }
            })
        });
    }
}

fn generate_lance_dataset_path() -> (String, Option<tempfile::TempDir>) {
    let object_store_scheme =
        std::env::var(ENV_OBJECT_STORE_SCHEME).unwrap_or("memory".to_string());
    match object_store_scheme.as_str() {
        "memory" => ("memory://test.lance".to_string(), None),
        "local" => {
            let uuid = uuid::Uuid::new_v4().to_string();
            let temp_dir = tempfile::tempdir().unwrap();
            let file_path = temp_dir.path().join(uuid);
            (
                format!("file://{}.lance", file_path.to_str().unwrap()),
                Some(temp_dir),
            )
        }
        _ => {
            panic!("Unknown object store scheme: {}", object_store_scheme)
        }
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
    targets = bench_random_take_with_dataset, bench_random_single_take_with_file_fragment, bench_random_single_take_with_file_reader, bench_random_batch_take_with_file_fragment, bench_random_batch_take_with_file_reader);
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_random_take_with_dataset, bench_random_single_take_with_file_fragment, bench_random_single_take_with_file_reader, bench_random_batch_take_with_file_fragment, bench_random_batch_take_with_file_reader);
criterion_main!(benches);
