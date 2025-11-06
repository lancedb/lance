// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Tests to validate IndexCache memory accounting by creating real indices,
//! prewarming them to populate the cache, then evicting entries one-by-one
//! and verifying that memory reduction matches DeepSizeOf estimates.
//!
//! This approach tests DeepSizeOf in realistic conditions where items are
//! actually cached, catching issues like:
//! - Arc sharing and double-counting
//! - Trait object DeepSizeOf under-counting
//! - Cache overhead miscalculations

use super::utils::AllocTracker;
use arrow::datatypes::{UInt32Type, UInt8Type};
use lance::dataset::InsertBuilder;
use lance_core::cache::LanceCache;
use lance_datafusion::datagen::DatafusionDatagenExt;
use lance_datagen::{array, gen_batch, BatchCount, RowCount};
use lance_index::scalar::InvertedIndexParams;
use lance_index::DatasetIndexExt;
use lance_index::{scalar::ScalarIndexParams, IndexType};
use rand::seq::SliceRandom;
use std::sync::Arc;

/// Test framework that validates DeepSizeOf by creating an index, prewarming cache,
/// then evicting entries one-by-one and verifying memory reduction
///
/// # Arguments
/// * `cache` - The cache instance to test
/// * `prewarm_fn` - Function to call that populates the cache
/// * `test_name` - Name of the test for error messages
/// * `tolerance_per_entry` - Acceptable deviation in bytes per cache entry
async fn test_cache_accounting<F, Fut>(
    cache: LanceCache,
    prewarm_fn: F,
    test_name: &str,
    tolerance_per_entry: usize,
) where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    prewarm_fn().await;
    if cache.size().await == 0 {
        panic!("{}: Cache is empty after prewarm!", test_name);
    }

    let mut entries = cache.entries().await;
    entries.shuffle(&mut rand::rng());
    drop(cache);

    for ((key, _), entry) in entries {
        assert_eq!(
            Arc::strong_count(&entry.record),
            1,
            "{}: Entry for key {:?} has unexpected strong count {}",
            test_name,
            key,
            Arc::strong_count(&entry.record)
        );
        let expected_freed = deepsize::DeepSizeOf::deep_size_of(&entry);
        let type_name = entry.type_name;

        let tracker = AllocTracker::new();
        {
            let _guard = tracker.enter();
            // Evict the entry - this should free memory
            drop(entry);
        }
        let stats = tracker.stats();

        // Actual memory freed = deallocations - allocations during eviction
        let actual_freed = stats
            .total_bytes_deallocated
            .saturating_sub(stats.total_bytes_allocated);

        let deviation = (expected_freed as isize - actual_freed).abs();
        dbg!((type_name, expected_freed, actual_freed, deviation));
        assert!(
            deviation <= tolerance_per_entry as isize,
            "{}: Entry (key: {:?}, type: {}): Expected to free {} bytes, but actually freed {} bytes (deviation: {}, tolerance: {}). Stats: alloc={}, dealloc={}",
            test_name,
            key,
            type_name,
            expected_freed,
            actual_freed,
            deviation,
            tolerance_per_entry,
            stats.total_bytes_allocated,
            stats.total_bytes_deallocated,
        );
    }
}

#[tokio::test]
async fn test_label_list_index_cache_accounting() {
    AllocTracker::init();

    // Create a dataset with a label list (inverted) index
    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp_path = tmp_dir.path().to_str().unwrap();

    // Create test data - list of uint8 values
    // Using larger dataset to get bigger cache entries: ~50MB
    let batch_size = 100_000;
    let num_batches = BatchCount::from(50);
    let data = gen_batch()
        .col(
            "labels",
            array::rand_list_any(array::cycle::<UInt8Type>(vec![1u8, 2]), false),
        )
        .into_df_stream(RowCount::from(batch_size), num_batches);

    InsertBuilder::new(tmp_path)
        .execute_stream(data)
        .await
        .unwrap();

    // Build label list index
    let mut dataset = lance::dataset::Dataset::open(tmp_path).await.unwrap();
    dataset
        .create_index_builder(
            &["labels"],
            IndexType::Scalar,
            &ScalarIndexParams::new("labellist".to_string()),
        )
        .await
        .unwrap();

    // Reload dataset to get fresh index with cache
    let dataset = lance::dataset::Dataset::open(tmp_path).await.unwrap();

    // Access the index cache (now public)
    let cache = (*dataset.index_cache).clone();

    // Test cache accounting by prewarming the index
    test_cache_accounting(
        cache,
        || async {
            dataset.prewarm_index("labels_idx").await.unwrap();
            drop(dataset);
        },
        "LabelListIndex",
        5_000, // 5KB tolerance per entry
    )
    .await;
}

#[tokio::test]
async fn test_btree_index_cache_accounting() {
    AllocTracker::init();

    let batch_size = 100_000;
    let num_batches = BatchCount::from(50);
    let data = gen_batch()
        .col("values", array::step::<UInt32Type>())
        .into_df_stream(RowCount::from(batch_size), num_batches);

    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp_path = tmp_dir.path().to_str().unwrap();
    InsertBuilder::new(tmp_path)
        .execute_stream(data)
        .await
        .unwrap();

    let mut dataset = lance::dataset::Dataset::open(tmp_path).await.unwrap();
    dataset
        .create_index_builder(
            &["values"],
            IndexType::Scalar,
            &ScalarIndexParams::new("btree".to_string()),
        )
        .await
        .unwrap();

    // Reload dataset to get fresh index with cache
    let dataset = lance::dataset::Dataset::open(tmp_path).await.unwrap();

    // Access the index cache (now public)
    let cache = (*dataset.index_cache).clone();

    // Test cache accounting by prewarming the index
    test_cache_accounting(
        cache,
        || async {
            dataset.prewarm_index("values_idx").await.unwrap();
            drop(dataset);
        },
        "BTreeIndex",
        5_000, // 5KB tolerance per entry
    )
    .await;
}

#[tokio::test]
async fn test_fts_index_cache_accounting() {
    AllocTracker::init();

    let batch_size = 10_000;
    let num_batches = BatchCount::from(50);
    // TODO: generate more realistic text data
    let data = gen_batch()
        .col(
            "text",
            array::rand_type(&arrow::datatypes::DataType::LargeUtf8),
        )
        .into_df_stream(RowCount::from(batch_size), num_batches);

    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp_path = tmp_dir.path().to_str().unwrap();
    InsertBuilder::new(tmp_path)
        .execute_stream(data)
        .await
        .unwrap();

    let params = InvertedIndexParams::default();
    let mut dataset = lance::dataset::Dataset::open(tmp_path).await.unwrap();
    dataset
        .create_index_builder(
            &["text"],
            IndexType::Scalar,
            &ScalarIndexParams::new("inverted".to_string()).with_params(&params),
        )
        .await
        .unwrap();

    // Reload dataset to get fresh index with cache
    let dataset = lance::dataset::Dataset::open(tmp_path).await.unwrap();

    // Access the index cache (now public)
    let cache = (*dataset.index_cache).clone();

    // Test cache accounting by prewarming the index
    test_cache_accounting(
        cache,
        || async {
            dataset.prewarm_index("text_idx").await.unwrap();
            drop(dataset);
        },
        "FTSIndex",
        20_000, // 20KB tolerance per entry - FTS indices are larger and more complex
    )
    .await;
}
