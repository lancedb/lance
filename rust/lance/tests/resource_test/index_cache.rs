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
use arrow::datatypes::UInt8Type;
use arrow_schema::{DataType, Field};
use lance::dataset::InsertBuilder;
use lance_core::cache::LanceCache;
use lance_datafusion::datagen::DatafusionDatagenExt;
use lance_datagen::{array, gen_batch, BatchCount, RowCount};
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
        dbg!((expected_freed, actual_freed));
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

// fn test_deep_size_of(value: impl deepsize::DeepSizeOf) {
//     let tracker = AllocTracker::new();
//     let reported_size = deepsize::DeepSizeOf::deep_size_of(&value);
//     {
//         let _guard = tracker.enter();
//         drop(value);
//     }
//     let stats = tracker.stats();
//     let actual_freed = stats.total_bytes_deallocated.saturating_sub(stats.total_bytes_allocated) as usize;
//     assert_eq!(reported_size, actual_freed);
// }

// #[test]
// fn test_deep_size_of_label_list_index() {
//     AllocTracker::init();
//     LabelListIndex::nemw
//     let value = todo!();
//     test_deep_size_of(value);
// }

#[tokio::test]
async fn test_label_list_index_cache_accounting() {
    AllocTracker::init();

    // Create a dataset with a label list (inverted) index
    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp_path = tmp_dir.path().to_str().unwrap();

    // Create test data - list of uint8 values
    // Using larger dataset to get bigger cache entries: ~50MB
    let batch_size = 1_000_000;
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
        // TODO: if we impl DeepSizeOf for FileReader, then we should be able to reduce this tolerance
        60_000, // 60KB tolerance per entry - accounts for cache overhead
    )
    .await;
}
