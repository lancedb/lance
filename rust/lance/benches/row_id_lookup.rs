// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compare the existing in-memory reverse index with an ordinary `_rowid` BTree.
//!
//! Run with `cargo bench -p lance --profile release-with-debug --bench row_id_lookup`.
//! `ROW_ID_LOOKUP_ROWS` sets the row count (default 100,000, at least 8).
//! Append-only and globally shuffled tables use the same keys and local store.
//! Cold means a fresh Lance session (OS caches are not flushed); dataset open is
//! outside the timed region. Warm means the reverse index / BTree pages have
//! already been read. Each result is checked against the other implementation.
//! The report also includes BTree bytes/row and the reverse index's retained
//! deep size; these are different resources, not a memory savings comparison.

#![allow(clippy::print_stdout)]

use std::sync::Arc;
use std::time::Duration;

use arrow_array::{RecordBatchIterator, UInt64Array, record_batch};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use lance::Dataset;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::dataset::rowids::{
    INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, SPILL_ROW_LINEAGE_CONFIG_KEY, get_row_id_index,
    row_ids_to_row_addrs,
};
use lance::dataset::{CommitBuilder, InsertBuilder, WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance_core::ROW_ID;
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::tempfile::TempStrDir;
use lance_index::{IndexType, scalar::ScalarIndexParams};
use lance_table::format::RowIdMeta;
use lance_table::rowids::{RowIdSequence, write_row_ids};
use lance_table::transaction::{Operation, RewriteGroup, TransactionBuilder};
use rand::{Rng, SeedableRng, rngs::SmallRng, seq::SliceRandom};
use tokio::runtime::Runtime;

async fn build_table(uri: &str, rows: usize, shuffled: bool) -> Dataset {
    let batch = record_batch!(("key", UInt64, (0..rows as u64).collect::<Vec<_>>())).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    let params = WriteParams {
        enable_stable_row_ids: true,
        max_rows_per_file: rows.div_ceil(8),
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, uri, Some(params.clone()))
        .await
        .unwrap();
    if shuffled {
        let mut ids = (0..rows as u64).collect::<Vec<_>>();
        ids.shuffle(&mut SmallRng::seed_from_u64(42));
        let batch = record_batch!(("key", UInt64, ids.clone())).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let dataset_ref = Arc::new(dataset);
        let staged = InsertBuilder::new(dataset_ref.clone())
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..params
            })
            .execute_uncommitted_stream(reader)
            .await
            .unwrap();
        let Operation::Append { mut fragments } = staged.operation else {
            panic!("expected an uncommitted append");
        };
        let mut offset = 0;
        for fragment in &mut fragments {
            let len = fragment.physical_rows.unwrap();
            fragment.row_id_meta = Some(RowIdMeta::Inline(
                write_row_ids(&RowIdSequence::from(&ids[offset..offset + len])).into(),
            ));
            offset += len;
        }
        let transaction = TransactionBuilder::new(
            dataset_ref.version().version,
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments: dataset_ref.fragments().as_ref().clone(),
                    new_fragments: fragments,
                }],
                rewritten_indices: vec![],
                frag_reuse_index: None,
            },
        )
        .build();
        dataset = CommitBuilder::new(dataset_ref)
            .execute(transaction)
            .await
            .unwrap();
        dataset
            .update_config([
                (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
                (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
            ])
            .await
            .unwrap();
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: rows,
                defer_index_remap: false,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
    }
    let index = dataset
        .create_index(
            &[ROW_ID],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let bytes: u64 = index
        .files
        .as_ref()
        .unwrap()
        .iter()
        .map(|file| file.size_bytes)
        .sum();
    let reverse = get_row_id_index(&dataset).await.unwrap().unwrap();
    println!(
        "layout={} rows={rows} btree_bytes={bytes} btree_bytes_per_row={:.3} reverse_index_deep_size_bytes={}",
        if shuffled { "shuffled" } else { "append" },
        bytes as f64 / rows as f64,
        reverse.deep_size_of()
    );
    dataset
}

fn bench_row_id_lookup(criterion: &mut Criterion) {
    let rows = std::env::var("ROW_ID_LOOKUP_ROWS").map_or(100_000, |value| {
        value
            .parse::<usize>()
            .expect("ROW_ID_LOOKUP_ROWS must be an integer")
    });
    assert!(rows >= 8, "ROW_ID_LOOKUP_ROWS must be at least 8");
    let runtime = Runtime::new().unwrap();
    for shuffled in [false, true] {
        let directory = TempStrDir::default();
        let dataset = runtime.block_on(build_table(&directory, rows, shuffled));
        let layout = if shuffled { "shuffled" } else { "append" };
        let mut group = criterion.benchmark_group(format!("row_id_lookup/{layout}"));
        group.sample_size(10);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(2));
        for key_count in [1, 1000] {
            let mut random = SmallRng::seed_from_u64(7);
            let ids = (0..key_count)
                .map(|_| random.random_range(0..rows as u64))
                .collect::<Vec<_>>();
            let keys = UInt64Array::from(ids.clone());
            let reverse = runtime
                .block_on(get_row_id_index(&dataset))
                .unwrap()
                .unwrap();
            let expected = reverse.get_many(&ids).unwrap();
            assert_eq!(
                runtime
                    .block_on(row_ids_to_row_addrs(&dataset, &keys))
                    .unwrap(),
                expected
            );
            group.bench_with_input(
                BenchmarkId::new("reverse_warm", key_count),
                &ids,
                |bench, ids| {
                    bench.iter(|| {
                        std::hint::black_box(runtime.block_on(async {
                            get_row_id_index(&dataset)
                                .await
                                .unwrap()
                                .unwrap()
                                .get_many(ids)
                                .unwrap()
                        }))
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new("btree_warm", key_count),
                &keys,
                |bench, keys| {
                    bench.iter(|| {
                        std::hint::black_box(
                            runtime
                                .block_on(row_ids_to_row_addrs(&dataset, keys))
                                .unwrap(),
                        )
                    });
                },
            );
            for btree in [false, true] {
                group.bench_function(
                    BenchmarkId::new(if btree { "btree_cold" } else { "reverse_cold" }, key_count),
                    |bench| {
                        bench.iter_batched(
                            || {
                                runtime
                                    .block_on(DatasetBuilder::from_uri(&directory).load())
                                    .unwrap()
                            },
                            |cold| {
                                runtime.block_on(async {
                                    let actual = if btree {
                                        row_ids_to_row_addrs(&cold, &keys).await.unwrap()
                                    } else {
                                        get_row_id_index(&cold)
                                            .await
                                            .unwrap()
                                            .unwrap()
                                            .get_many(&ids)
                                            .unwrap()
                                    };
                                    std::hint::black_box(actual)
                                })
                            },
                            BatchSize::SmallInput,
                        );
                    },
                );
            }
        }
        group.finish();
    }
}

criterion_group!(benches, bench_row_id_lookup);
criterion_main!(benches);
