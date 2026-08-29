// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Tests for #7502: the v2 write path coalesces small input batches in the
//! stream chunker before passing them to the writer, so compaction no longer
//! produces one page per source fragment.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::dataset::optimize::{CompactionMode, CompactionOptions, compact_files};
use lance_core::cache::LanceCache;
use lance_core::utils::tempfile::TempStrDir;
use lance_encoding::decoder::DecoderPlugins;
use lance_file::reader::FileReader;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;

/// Build a batch with a single wide binary column so each row is large enough
/// that a single batch is already several megabytes.
fn make_wide_batch(num_rows: usize) -> RecordBatch {
    let value_bytes = 2 * 1024 * 1024; // 2 MiB per row
    let values: Vec<Vec<u8>> = (0..num_rows)
        .map(|i| vec![(i % 251) as u8; value_bytes])
        .collect();
    let array = Arc::new(arrow_array::LargeBinaryArray::from_iter_values(values));
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "payload",
            DataType::LargeBinary,
            true,
        )])),
        vec![array as ArrayRef],
    )
    .unwrap()
}

async fn count_pages(dataset: &Dataset, file_name: &str) -> usize {
    let object_store = dataset.object_store(None).await.unwrap();
    let path = dataset.data_dir().join(file_name);
    let scheduler = ScanScheduler::new(
        object_store,
        SchedulerConfig::max_bandwidth(dataset.object_store(None).await.unwrap().as_ref()),
    );
    let file_scheduler = scheduler
        .open_file(&path, &CachedFileSize::unknown())
        .await
        .unwrap();
    let reader = FileReader::try_open(
        file_scheduler,
        None,
        Arc::<DecoderPlugins>::default(),
        &LanceCache::no_cache(),
        Default::default(),
    )
    .await
    .unwrap();
    reader.metadata().column_infos[0].page_infos.len()
}

/// When multiple small fragments are written individually, each fragment's
/// file has 1 page because the stream chunker only operates within a single
/// write, not across separate fragment writes.
#[tokio::test]
async fn test_write_many_small_fragments_one_page_each() {
    let tmp = TempStrDir::default();
    let uri = tmp.as_str();

    let num_fragments = 8;
    let batch = make_wide_batch(5);
    let batches: Vec<RecordBatch> = (0..num_fragments).map(|_| batch.clone()).collect();
    let schema = batches[0].schema();
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);

    let dataset = Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            max_rows_per_file: 5,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), num_fragments);

    for fragment in dataset.get_fragments() {
        let file = &fragment.metadata().files[0];
        let pages = count_pages(&dataset, &file.path).await;
        assert_eq!(
            pages,
            1,
            "fragment {} should have 1 page, got {}",
            fragment.id(),
            pages
        );
    }
}

/// A single oversized batch is emitted as one chunk by the byte-bounded
/// chunker and therefore written as one page.  This documents the current
/// behavior.
#[tokio::test]
async fn test_write_single_large_batch_one_page() {
    let tmp = TempStrDir::default();
    let uri = tmp.as_str();

    let batch = make_wide_batch(40);
    let schema = batch.schema();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);

    let dataset = Dataset::write(reader, uri, Some(WriteParams::default()))
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 1);

    let fragment = &dataset.get_fragments()[0];
    let file = &fragment.metadata().files[0];
    let pages = count_pages(&dataset, &file.path).await;
    assert_eq!(
        pages, 1,
        "single 80 MiB batch currently produces 1 page, got {pages}"
    );
}

/// Build a batch with a per-row unique `id` and a wide `payload` column whose
/// first bytes encode the id, so a row's provenance is verifiable after a
/// compaction merges and reorders batches.
fn make_batch_with_ids(num_rows: usize) -> RecordBatch {
    let ids: Vec<i64> = (0..num_rows).map(|i| i as i64).collect();
    let payloads: Vec<Vec<u8>> = (0..num_rows)
        .map(|i| {
            // First 8 bytes are the little-endian id, rest is a distinctive fill.
            let mut v = vec![0xAB; 2 * 1024 * 1024];
            v[..8].copy_from_slice(&(i as u64).to_le_bytes());
            v
        })
        .collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("payload", DataType::LargeBinary, true),
        ])),
        vec![
            Arc::new(arrow_array::Int64Array::from(ids)),
            Arc::new(arrow_array::LargeBinaryArray::from_iter_values(
                payloads.iter().map(|v| v.as_slice()),
            )) as ArrayRef,
        ],
    )
    .unwrap()
}

/// After compaction, every row survives intact: the `id` column keeps its
/// value and each `payload` still begins with its id's little-endian bytes.
/// This guards against the coalescing rewrite dropping or miss-ordering rows.
#[tokio::test]
async fn test_compaction_preserves_data_integrity() {
    let tmp = TempStrDir::default();
    let uri = tmp.as_str();

    let num_fragments = 8;
    let rows_per_fragment = 5;
    let batch = make_batch_with_ids(rows_per_fragment);
    let batches: Vec<RecordBatch> = (0..num_fragments).map(|_| batch.clone()).collect();
    let schema = batches[0].schema();
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);

    let mut dataset = Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            max_rows_per_file: rows_per_fragment,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), num_fragments);

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: rows_per_fragment * num_fragments,
            compaction_mode: Some(CompactionMode::Reencode),
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 1);

    // Read back the full table and verify every row's id and payload agree.
    let table = dataset
        .scan()
        .try_into_batch()
        .await
        .expect("scan should succeed");
    assert_eq!(table.num_rows(), num_fragments * rows_per_fragment);

    let ids = table
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("id should be Int64");
    let payloads = table
        .column_by_name("payload")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow_array::LargeBinaryArray>()
        .expect("payload should be LargeBinary");

    for row in 0..table.num_rows() {
        let id = ids.value(row);
        let payload = payloads.value(row);
        assert!(
            payload.len() >= 8,
            "row {row}: payload too short: {} bytes",
            payload.len()
        );
        let encoded = u64::from_le_bytes(payload[..8].try_into().unwrap());
        assert_eq!(
            encoded, id as u64,
            "row {row}: payload id {encoded} does not match column id {id}"
        );
        assert!(
            payload[8..].iter().all(|&b| b == 0xAB),
            "row {row}: payload fill corrupted"
        );
    }
}

/// After compaction, the merged file no longer keeps one page per source
/// fragment.  The stream chunker coalesces small input batches across
/// fragment boundaries before they reach the writer, so the page count is
/// independent of the number of source fragments.
#[tokio::test]
async fn test_compaction_coalesces_pages() {
    let tmp = TempStrDir::default();
    let uri = tmp.as_str();

    let num_fragments = 8;
    let batch = make_wide_batch(5);
    let batches: Vec<RecordBatch> = (0..num_fragments).map(|_| batch.clone()).collect();
    let schema = batches[0].schema();
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);

    let mut dataset = Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            max_rows_per_file: 5,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), num_fragments);

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 5 * num_fragments,
            compaction_mode: Some(CompactionMode::Reencode),
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 1);

    let fragment = &dataset.get_fragments()[0];
    let file = &fragment.metadata().files[0];
    let pages = count_pages(&dataset, &file.path).await;
    assert!(
        pages < num_fragments,
        "compaction should coalesce pages: got {pages} pages for {num_fragments} source fragments"
    );
}
