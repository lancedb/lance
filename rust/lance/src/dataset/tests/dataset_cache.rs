// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Integration tests verifying that the two-tier data cache produces cache
//! hits on repeated scans of the same Lance dataset.
//!
//! Key property being tested: the decoder always requests the same byte
//! ranges for the same file + projection, so the second scan should be
//! served entirely from the memory cache.

use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance_io::data_cache::{DataCacheConfig, TieredDataCache, ssd};

use crate::Dataset;
use crate::dataset::builder::DatasetBuilder;
use crate::dataset::write::WriteParams;
use crate::session::Session;
use lance_file::version::LanceFileVersion;

/// Write a small Lance dataset with N rows of (id: i32, value: i32).
async fn create_test_dataset(uri: &str, n_rows: usize) -> Dataset {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..n_rows as i32)),
            Arc::new(Int32Array::from_iter_values(
                (0..n_rows as i32).map(|x| x * 2),
            )),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    // Explicitly request v2 format to ensure the FileScheduler + cache path is used.
    let write_params = WriteParams {
        data_storage_version: Some(LanceFileVersion::V2_1),
        ..Default::default()
    };
    Dataset::write(reader, uri, Some(write_params)).await.unwrap()
}

/// Open `uri` with the given `TieredDataCache` wired into the session.
async fn open_with_cache(uri: &str, cache: Arc<TieredDataCache>) -> Dataset {
    let session = Arc::new(Session::default().with_data_cache(cache));
    DatasetBuilder::from_uri(uri)
        .with_session(session)
        .load()
        .await
        .unwrap()
}

#[tokio::test]
async fn test_cache_hit_rate_on_repeated_scans() {
    // Write to a real temp directory — memory:// doesn't share state between
    // different ObjectStore instances opened by DatasetBuilder.
    let tmp = tempfile::tempdir().unwrap();
    let uri = tmp.path().join("test.lance");
    let uri = uri.to_str().unwrap();

    create_test_dataset(uri, 10_000).await;

    // Memory-only cache, single shard (all entries go to same shard).
    let config = DataCacheConfig {
        max_memory_bytes: 64 * 1024 * 1024, // 64 MiB — easily fits the dataset
        num_shards: 1,
        ssd_enabled: false,
        ssd_cache_dir: None,
        ssd_max_bytes: 0,
        ssd_num_shards: ssd::DEFAULT_NUM_SSD_SHARDS,
        verify: false,
        ssd_crc32_enabled: false,
    };
    let cache = TieredDataCache::new(&config).await.unwrap();
    let dataset = open_with_cache(uri, cache.clone()).await;

    // Verify the cache is wired into the dataset session.
    assert!(
        dataset.session.data_cache.is_some(),
        "session must have data_cache set"
    );

    // ── First scan (cold) ─────────────────────────────────────────────────
    let batches: Vec<RecordBatch> = dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert!(!batches.is_empty(), "first scan produced no batches");
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    println!("First scan: {} batches, {} rows", batches.len(), total_rows);

    let stats_cold = cache.memory_stats();
    println!(
        "After scan 1 (cold):  misses={}, hits={}, bytes={}",
        stats_cold.misses, stats_cold.hits, stats_cold.current_bytes
    );
    assert!(stats_cold.misses > 0, "expected misses on cold scan");
    assert_eq!(stats_cold.hits, 0, "expected zero hits on cold scan");

    // ── Second scan (warm) ────────────────────────────────────────────────
    // The decoder requests the same byte ranges — should hit the cache.
    let batches2: Vec<RecordBatch> = dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(batches.len(), batches2.len(), "batch count mismatch");

    let stats_warm = cache.memory_stats();
    let new_hits = stats_warm.hits;
    let new_misses = stats_warm.misses - stats_cold.misses;
    let hit_rate = new_hits as f64 / (new_hits + new_misses).max(1) as f64;

    println!(
        "After scan 2 (warm):  misses={}, hits={}, bytes={}",
        stats_warm.misses, stats_warm.hits, stats_warm.current_bytes
    );
    println!(
        "Second scan hit rate: {:.1}%  ({} hits, {} new misses)",
        hit_rate * 100.0,
        new_hits,
        new_misses
    );

    assert!(stats_warm.hits > 0, "expected cache hits on warm scan");
    assert!(
        hit_rate >= 0.9,
        "expected ≥90% cache hit rate on warm scan, got {:.1}%",
        hit_rate * 100.0
    );
}

#[tokio::test]
async fn test_cache_data_integrity_across_scans() {
    // Verify cached bytes produce byte-for-byte identical results.
    let tmp = tempfile::tempdir().unwrap();
    let uri = tmp.path().join("integrity.lance");
    let uri = uri.to_str().unwrap();

    create_test_dataset(uri, 5_000).await;

    let config = DataCacheConfig {
        max_memory_bytes: 64 * 1024 * 1024,
        num_shards: 1,
        ssd_enabled: false,
        ssd_cache_dir: None,
        ssd_max_bytes: 0,
        ssd_num_shards: ssd::DEFAULT_NUM_SSD_SHARDS,
        verify: false,
        ssd_crc32_enabled: false,
    };
    let cache = TieredDataCache::new(&config).await.unwrap();
    let dataset = open_with_cache(uri, cache.clone()).await;

    // Scan 1 — cold.
    let scan1: Vec<RecordBatch> = dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();

    // Scan 2 — warm (from cache).
    let scan2: Vec<RecordBatch> = dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();

    assert_eq!(scan1.len(), scan2.len(), "batch count mismatch");
    for (i, (b1, b2)) in scan1.iter().zip(scan2.iter()).enumerate() {
        assert_eq!(b1.schema(), b2.schema(), "batch {i}: schema mismatch");
        assert_eq!(b1.num_rows(), b2.num_rows(), "batch {i}: row count mismatch");
        for col in 0..b1.num_columns() {
            assert_eq!(
                b1.column(col).as_ref(),
                b2.column(col).as_ref(),
                "batch {i} col {col}: data mismatch — cached bytes corrupted"
            );
        }
    }

    let stats = cache.memory_stats();
    let total_rows: usize = scan2.iter().map(|b| b.num_rows()).sum();
    println!(
        "Data integrity: {} batches × {} rows match. hits={}, misses={}",
        scan2.len(), total_rows, stats.hits, stats.misses
    );
    assert!(stats.hits > 0, "warm scan must have cache hits");
}
