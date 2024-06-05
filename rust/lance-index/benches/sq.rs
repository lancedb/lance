// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Scalar Quantization Benchmarks

use std::{iter::repeat_with, ops::Range, sync::Arc, time::Duration};

use arrow_array::{FixedSizeListArray, RecordBatch, UInt64Array, UInt8Array};
use arrow_schema::{DataType, Field, Schema};
use criterion::{criterion_group, criterion_main, Criterion};
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_core::ROW_ID;
use lance_index::vector::{
    sq::storage::ScalarQuantizationStorage, v3::storage::VectorStore, SQ_CODE_COLUMN,
};
use lance_linalg::distance::DistanceType;
use lance_testing::datagen::generate_random_array;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use rand::prelude::*;

fn create_full_batch(range: Range<u64>, dim: usize) -> RecordBatch {
    let mut rng = rand::thread_rng();
    let row_ids = UInt64Array::from_iter_values(range.clone().into_iter());
    let sq_code =
        UInt8Array::from_iter_values(repeat_with(|| rng.gen::<u8>()).take(row_ids.len() * dim));
    let sq_code_fsl = FixedSizeListArray::try_new_from_values(sq_code, dim as i32).unwrap();

    let vector_data = generate_random_array(row_ids.len() * dim);
    let vector_fsl = FixedSizeListArray::try_new_from_values(vector_data, dim as i32).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID, DataType::UInt64, false),
        Field::new(
            SQ_CODE_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                dim as i32,
            ),
            false,
        ),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(row_ids),
            Arc::new(sq_code_fsl),
            Arc::new(vector_fsl),
        ],
    )
    .unwrap()
}

fn create_sq_batch(row_id_range: Range<u64>, dim: usize) -> RecordBatch {
    let batch = create_full_batch(row_id_range, dim);
    batch.drop_column("vector").unwrap()
}

fn bench_storge(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    const TOTAL: usize = 8 * 1024 * 1024; // 8M rows

    for num_chunks in [1, 32, 128, 1024] {
        let storage = ScalarQuantizationStorage::try_new(
            8,
            DistanceType::L2,
            -1.0..1.0,
            repeat_with(|| create_sq_batch(0..(TOTAL / num_chunks) as u64, 512)).take(num_chunks),
        )
        .unwrap();
        c.bench_function(
            format!("ScalarQuantizationStorage,chunks={}x10K", num_chunks).as_str(),
            |b| {
                let total = storage.len();
                b.iter(|| {
                    let a = rng.gen_range(0..total as u32);
                    let b = rng.gen_range(0..total as u32);
                    storage.distance_between(a, b)
                });
            },
        );
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_storge);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10);
    targets = bench_storge);

criterion_main!(benches);
