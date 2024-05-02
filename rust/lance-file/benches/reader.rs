// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::sync::Arc;

use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, Criterion};
use futures::StreamExt;
use lance_file::v2::{
    reader::FileReader,
    writer::{FileWriter, FileWriterOptions},
};
use lance_io::{object_store::ObjectStore, scheduler::StoreScheduler};

fn bench_reader(c: &mut Criterion) {
    let mut group = c.benchmark_group("reader");
    let data = lance_datagen::gen()
        .anon_col(lance_datagen::array::rand_type(&DataType::Int32))
        .into_batch_rows(lance_datagen::RowCount::from(1024 * 1024))
        .unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();

    let tempdir = tempfile::tempdir().unwrap();
    let test_path = tempdir.path();
    let (object_store, base_path) =
        ObjectStore::from_path(test_path.as_os_str().to_str().unwrap()).unwrap();
    let file_path = base_path.child("foo.lance");
    let object_writer = rt.block_on(object_store.create(&file_path)).unwrap();
    let schema = data.schema().as_ref().clone();

    let mut writer = FileWriter::try_new(
        object_writer,
        file_path.to_string(),
        data.schema().as_ref().try_into().unwrap(),
        FileWriterOptions::default(),
    )
    .unwrap();
    rt.block_on(writer.write_batch(&data)).unwrap();
    rt.block_on(writer.finish()).unwrap();
    group.throughput(criterion::Throughput::Bytes(
        data.get_array_memory_size() as u64
    ));
    group.bench_function("decode", |b| {
        b.iter(|| {
            let object_store = &object_store;
            let file_path = &file_path;
            let schema = &schema;
            let data = &data;
            rt.block_on(async move {
                let store_scheduler = StoreScheduler::new(Arc::new(object_store.clone()), 8);
                let scheduler = store_scheduler.open_file(file_path).await.unwrap();
                let reader = FileReader::try_open(scheduler.clone(), schema.clone())
                    .await
                    .unwrap();
                let mut stream = reader
                    .read_stream(lance_io::ReadBatchParams::RangeFull, 16 * 1024, 16)
                    .unwrap();
                let mut row_count = 0;
                while let Some(batch) = stream.next().await {
                    row_count += batch.unwrap().num_rows();
                }
                assert_eq!(data.num_rows(), row_count);
            });
        })
    });
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(pprof::criterion::PProfProfiler::new(100, pprof::criterion::Output::Flamegraph(None)));
    targets = bench_reader);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_reader);
criterion_main!(benches);
