use std::sync::Arc;

use super::utils::AllocTracker;
use all_asserts::assert_le;
use arrow_schema::{DataType, Field};
use lance::dataset::InsertBuilder;
use lance_datafusion::datagen::DatafusionDatagenExt;
use lance_datagen::{array, gen_batch, BatchCount, RowCount};
use lance_index::DatasetIndexExt;
use lance_index::{scalar::ScalarIndexParams, IndexType};

// Key things to test
// - Getting index stats requires reading only the metadata (no data read)
// -

// Ops with index
// - Build
// - Load
// - get stats
// -

#[tokio::test]
async fn test_label_list_lifecycle() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp_path = tmp_dir.path().to_str().unwrap();
    // Create a stream of 100MB of data, in batches
    {
        // 12 bytes per list entry, average 5 entries per list -> 60 bytes per row
        // 1MB / 60 = ~16k rows per batch
        let batch_size = 16_000;
        let num_batches = BatchCount::from(100);
        let data = gen_batch()
            .col(
                "value",
                array::rand_type(&DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::UInt8,
                    false,
                )))),
            )
            .into_df_stream(RowCount::from(batch_size), num_batches);
        let _ = InsertBuilder::new(tmp_path)
            .execute_stream(data)
            .await
            .unwrap();
    }

    // Build index on column
    // let io_tracking = todo!();
    let alloc_tracker = AllocTracker::new();
    {
        let _guard = alloc_tracker.enter();
        let mut dataset = lance::dataset::Dataset::open(tmp_path).await.unwrap();

        dataset
            .create_index_builder(
                &["value"],
                IndexType::Scalar,
                &ScalarIndexParams::new("labellist".to_string()),
            )
            .await
            .unwrap();
    }

    let mem_stats = alloc_tracker.stats();
    assert_le!(
        mem_stats.max_bytes_allocated,
        70 * 1024 * 1024,
        "Memory usage too high"
    );
    assert_le!(
        mem_stats.total_bytes_allocated,
        400 * 1024 * 1024,
        "Total memory allocation too high"
    );
    assert_eq!(mem_stats.net_bytes_allocated(), 0, "memory leak");

    // Drop everything, assert no leak

    // Call load index

    // assert minimal IO and memory usage done

    // Drop everything, assert no leak

    // Call get stats
    // Assert IO and memory are small

    // Drop everything, assert no leak
}
