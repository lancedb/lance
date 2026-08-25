// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{hint::black_box, sync::Arc, time::Duration};

use arrow_array::{RecordBatch, RecordBatchOptions, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use futures::{FutureExt, StreamExt, TryStreamExt, future, stream};
use lance_io::ReadBatchParams;
use lance_table::{
    rowids::RowIdSequence,
    utils::stream::{
        ReadBatchTask, ReadBatchTaskStream, RowIdAndDeletesConfig, wrap_with_row_id_and_delete,
    },
};

fn make_batch(batch_size: usize, has_payload: bool) -> RecordBatch {
    if has_payload {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "value",
                DataType::UInt64,
                false,
            )])),
            vec![Arc::new(UInt64Array::from(vec![0; batch_size]))],
        )
        .unwrap()
    } else {
        RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            Vec::new(),
            &RecordBatchOptions::new().with_row_count(Some(batch_size)),
        )
        .unwrap()
    }
}

fn make_tasks(batch: RecordBatch, total_rows: usize, batch_size: usize) -> ReadBatchTaskStream {
    let tasks = (0..total_rows).step_by(batch_size).map(move |offset| {
        let num_rows = batch_size.min(total_rows - offset);
        let batch = if num_rows == batch.num_rows() {
            batch.clone()
        } else {
            batch.slice(0, num_rows)
        };
        ReadBatchTask {
            task: future::ready(Ok(batch)).boxed(),
            num_rows: num_rows as u32,
        }
    });
    stream::iter(tasks).boxed()
}

fn make_config(
    total_rows: usize,
    sequence: Arc<RowIdSequence>,
    with_all_system_columns: bool,
) -> RowIdAndDeletesConfig {
    RowIdAndDeletesConfig {
        params: ReadBatchParams::RangeFull,
        with_row_id: true,
        with_row_addr: with_all_system_columns,
        with_row_last_updated_at_version: with_all_system_columns,
        with_row_created_at_version: with_all_system_columns,
        deletion_vector: None,
        row_id_sequence: Some(sequence),
        last_updated_at_sequence: None,
        created_at_sequence: None,
        make_deletions_null: false,
        total_num_rows: total_rows as u32,
    }
}

fn bench_stream_row_ids(c: &mut Criterion) {
    let total_rows = std::env::var("BENCH_SYSTEM_ROWS")
        .map(|value| value.parse().unwrap())
        .unwrap_or(100_000_usize);
    let batch_size = std::env::var("BENCH_SYSTEM_BATCH_SIZE")
        .map(|value| value.parse().unwrap())
        .unwrap_or(1_024_usize)
        .min(total_rows);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("stream_row_ids");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for hole_stride in [2_u64, 17] {
        let sequence = Arc::new(
            RowIdSequence::try_from_iter(
                (0_u64..)
                    .filter(|value| value % hole_stride != 0)
                    .take(total_rows),
            )
            .unwrap(),
        );
        for has_payload in [false, true] {
            for with_all_system_columns in [false, true] {
                let batch = make_batch(batch_size, has_payload);
                let system_columns = if with_all_system_columns {
                    "all"
                } else {
                    "row_id"
                };
                let parameter =
                    format!("holes_{hole_stride}/payload_{has_payload}/system_{system_columns}");
                group.bench_with_input(
                    BenchmarkId::new("shape", parameter),
                    &has_payload,
                    |b, _| {
                        b.iter_batched(
                            || {
                                (
                                    make_tasks(batch.clone(), total_rows, batch_size),
                                    make_config(
                                        total_rows,
                                        sequence.clone(),
                                        with_all_system_columns,
                                    ),
                                )
                            },
                            |(tasks, config)| {
                                let batches = runtime
                                    .block_on(
                                        wrap_with_row_id_and_delete(tasks, 0, config)
                                            .buffered(8)
                                            .try_collect::<Vec<_>>(),
                                    )
                                    .unwrap();
                                black_box(batches);
                            },
                            BatchSize::SmallInput,
                        );
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_stream_row_ids);
criterion_main!(benches);
