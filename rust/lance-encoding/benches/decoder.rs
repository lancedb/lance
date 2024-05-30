// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::sync::Arc;

use arrow_schema::{DataType, Field, TimeUnit};
use criterion::{criterion_group, criterion_main, Criterion};
use lance_encoding::encoder::{encode_batch, CoreFieldEncodingStrategy};

const PRIMITIVE_TYPES: &[DataType] = &[
    DataType::Date32,
    DataType::Date64,
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::UInt8,
    DataType::UInt16,
    DataType::UInt32,
    DataType::UInt64,
    DataType::Float16,
    DataType::Float32,
    DataType::Float64,
    DataType::Decimal128(10, 10),
    DataType::Decimal256(10, 10),
    DataType::Timestamp(TimeUnit::Nanosecond, None),
    DataType::Time32(TimeUnit::Second),
    DataType::Time64(TimeUnit::Nanosecond),
    DataType::Duration(TimeUnit::Second),
    // The Interval type is supported by the reader but the writer works with Lance schema
    // at the moment and Lance schema can't parse interval
    // DataType::Interval(IntervalUnit::DayTime),
];

// Some types are supported by the encoder/decoder but Lance
// schema doesn't yet parse them in the context of a fixed size list.
const PRIMITIVE_TYPES_FOR_FSL: &[DataType] = &[
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::UInt8,
    DataType::UInt16,
    DataType::UInt32,
    DataType::UInt64,
    DataType::Float16,
    DataType::Float32,
    DataType::Float64,
];

fn bench_decode(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_primitive");
    for data_type in PRIMITIVE_TYPES {
        let data = lance_datagen::gen()
            .anon_col(lance_datagen::array::rand_type(&DataType::Int32))
            .into_batch_rows(lance_datagen::RowCount::from(1024 * 1024))
            .unwrap();
        let input_bytes = data.get_array_memory_size();
        group.throughput(criterion::Throughput::Bytes(input_bytes as u64));
        let encoding_strategy = CoreFieldEncodingStrategy::default();
        let encoded = rt
            .block_on(encode_batch(&data, &encoding_strategy, 1024 * 1024))
            .unwrap();
        let func_name = format!("{:?}", data_type).to_lowercase();
        group.bench_function(func_name, |b| {
            b.iter(|| {
                let batch = rt
                    .block_on(lance_encoding::decoder::decode_batch(&encoded))
                    .unwrap();
                assert_eq!(data.num_rows(), batch.num_rows());
            })
        });
    }
}
fn bench_decode_fsl(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_primitive_fsl");
    for data_type in PRIMITIVE_TYPES_FOR_FSL {
        let data = lance_datagen::gen()
            .anon_col(lance_datagen::array::rand_type(&DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Int32, true)),
                1024,
            )))
            .into_batch_rows(lance_datagen::RowCount::from(1024))
            .unwrap();
        let input_bytes = data.get_array_memory_size();
        group.throughput(criterion::Throughput::Bytes(input_bytes as u64));
        let encoding_strategy = CoreFieldEncodingStrategy::default();
        let encoded = rt
            .block_on(encode_batch(&data, &encoding_strategy, 1024 * 1024))
            .unwrap();
        let func_name = format!("{:?}", data_type).to_lowercase();
        group.bench_function(func_name, |b| {
            b.iter(|| {
                let batch = rt
                    .block_on(lance_encoding::decoder::decode_batch(&encoded))
                    .unwrap();
                assert_eq!(data.num_rows(), batch.num_rows());
            })
        });
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(pprof::criterion::PProfProfiler::new(100, pprof::criterion::Output::Flamegraph(None)));
    targets = bench_decode, bench_decode_fsl);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_decode, bench_decode_fsl);
criterion_main!(benches);
