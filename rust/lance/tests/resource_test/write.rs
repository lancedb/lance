use super::utils::AllocTracker;
use all_asserts::assert_le;
use arrow_schema::DataType;
use lance::dataset::InsertBuilder;
use lance_datafusion::datagen::DatafusionDatagenExt;
use lance_datagen::{array, gen_batch, BatchCount, ByteCount, RoundingBehavior};

// TODO: also add IO

#[tokio::test]
async fn test_insert_memory() {
    // Create a stream of 100MB of data, in batches
    let batch_size = 1024 * 1024; // 1MB
    let num_batches = BatchCount::from(100);
    let data = gen_batch()
        .col("a", array::rand_type(&DataType::Int32))
        .into_df_stream_bytes(
            ByteCount::from(batch_size),
            num_batches,
            RoundingBehavior::RoundDown,
        )
        .unwrap();

    let alloc_tracker = AllocTracker::new();
    {
        let _guard = alloc_tracker.enter();

        // write out to temporary directory
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path().to_str().unwrap();
        let _dataset = InsertBuilder::new(tmp_path)
            .execute_stream(data)
            .await
            .unwrap();
    }

    let stats = alloc_tracker.stats();
    // Allow for 15x the batch size to account for:
    // - Allocator metadata overhead (wrapped_size vs object_size)
    // - Internal buffering and temporary allocations
    // - Arrow array overhead
    // The key test is that we don't load all 100MB into memory at once
    assert_le!(
        stats.max_bytes_allocated,
        (batch_size * 15) as isize,
        "Max memory usage exceeded"
    );
}
