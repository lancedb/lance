// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use all_asserts::assert_gt;
use arrow_schema::DataType;
use lance::dataset::InsertBuilder;
use lance_datafusion::datagen::DatafusionDatagenExt;
use lance_datagen::{array, gen_batch, BatchCount, ByteCount, RoundingBehavior};

use crate::resource_test::utils::{get_alloc_stats, reset_alloc_stats};

#[tokio::test]
async fn test_insert_memory() {
    reset_alloc_stats();
    // Create a stream of 100MB of data, in batches
    let batch_size = 10 * 1024 * 1024; // 10MB
    let num_batches = BatchCount::from(10);
    let data = gen_batch()
        .col("a", array::rand_type(&DataType::Int32))
        .into_df_stream_bytes(
            ByteCount::from(batch_size),
            num_batches,
            RoundingBehavior::RoundDown,
        )
        .unwrap();

    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp_path = tmp_dir.path().to_str().unwrap();
    let _dataset = InsertBuilder::new(tmp_path)
        .execute_stream(data)
        .await
        .unwrap();

    let stats = get_alloc_stats();

    assert_gt!(stats.total_bytes_allocated, 100 * 1024 * 1024);

    // The key test: we shouldn't load all 100MB at once
    // Allow 2x the batch size to account for overhead and buffering
    let max_allowed_bytes = (batch_size * 2) as isize;

    let peak_mb = stats.max_bytes_allocated as f64 / (1024.0 * 1024.0);

    assert!(
        stats.max_bytes_allocated <= max_allowed_bytes,
        "Peak memory {:.2} MB exceeded limit {:.2} MB",
        peak_mb,
        max_allowed_bytes as f64 / (1024.0 * 1024.0)
    );
}
