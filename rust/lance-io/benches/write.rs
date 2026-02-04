// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
#![allow(clippy::print_stdout)]

use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use rand::RngCore;
use std::fs::File;
use tokio::{io::AsyncWriteExt, runtime::Runtime};

use criterion::{criterion_group, criterion_main, Criterion};

fn generate_data(num_bytes: u64) -> Vec<u8> {
    let mut data = vec![0; num_bytes as usize];
    rand::rng().fill_bytes(&mut data);
    data
}

fn write_basic(data: &[u8], path: &std::path::Path) {
    let mut f = File::create(path).unwrap();
    let writer = std::io::BufWriter::new(&mut f);
    std::io::Write::write_all(&mut writer.into_inner().unwrap(), data).unwrap();
}

async fn write_lance(data: &[u8], obj_store: &ObjectStore, path: &Path) {
    let mut writer = obj_store.create(path).await.unwrap();
    writer.write_all(data).await.unwrap();
}

const DATA_SIZE: u64 = 128 * 1024 * 1024;

fn bench_basic_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("write");

    group.throughput(criterion::Throughput::Bytes(DATA_SIZE));

    let runtime = Runtime::new().unwrap();
    let temp_path = std::env::temp_dir().join("lance_io_bench_write");
    std::fs::create_dir_all(&temp_path).unwrap();

    let data = generate_data(DATA_SIZE);

    group.bench_function("basic_write", |b| {
        let path = temp_path.join("basic_write.file");
        b.iter(|| {
            write_basic(&data, &path);
        });
    });

    let obj_store = ObjectStore::local();
    group.bench_function("lance_write", |b| {
        let path = Path::from_absolute_path(temp_path.join("lance_write.file")).unwrap();
        b.iter(|| {
            runtime.block_on(write_lance(&data, &obj_store, &path));
        });
    });
}

criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_basic_write);

criterion_main!(benches);
