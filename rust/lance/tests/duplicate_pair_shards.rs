// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{Float32Array, RecordBatch, RecordBatchIterator, UInt64Array, record_batch};
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::TryStreamExt;
use lance::Dataset;
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance::index::vector::dedup::{DuplicatePairsOptions, find_duplicate_pairs_with_options};
use lance_index::IndexType;
use lance_linalg::distance::DistanceType;

async fn pairs(mut stream: SendableRecordBatchStream) -> Vec<(u64, u64, u32)> {
    let mut pairs = Vec::new();
    while let Some(batch) = stream.try_next().await.unwrap() {
        let a = batch["row_id_a"]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let b = batch["row_id_b"]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let d = batch["distance"]
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        pairs.extend((0..batch.num_rows()).map(|i| (a.value(i), b.value(i), d.value(i).to_bits())));
    }
    pairs.sort_unstable();
    pairs
}

#[tokio::test]
async fn shards_preserve_cross_block_pairs_and_deletions() {
    let ids = record_batch!(("id", Int32, (0..17).collect::<Vec<_>>())).unwrap();
    let vectors = arrow_array::FixedSizeListArray::from_iter_primitive::<
        arrow_array::types::Float32Type,
        _,
        _,
    >(
        (0..17).map(|i| Some(vec![Some((i % 5) as f32), Some(1.0)])),
        2,
    );
    let batch = RecordBatch::try_from_iter([
        ("id", ids.column(0).clone()),
        ("vector", Arc::new(vectors) as arrow_array::ArrayRef),
    ])
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        None,
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            None,
            &VectorIndexParams::ivf_flat(1, DistanceType::L2),
            false,
        )
        .await
        .unwrap();
    dataset.delete("id = 3 OR id = 12").await.unwrap();
    let dataset = Arc::new(dataset);
    let expected = pairs(
        find_duplicate_pairs_with_options(
            dataset.clone(),
            "vector",
            0.0,
            DuplicatePairsOptions::default(),
        )
        .await
        .unwrap(),
    )
    .await;
    assert!(!expected.is_empty());
    for memory in [0, 1024 * 1024] {
        for rows in [3, 8, 16] {
            let mut actual = Vec::new();
            for shard in 0..4 {
                actual.extend(
                    pairs(
                        find_duplicate_pairs_with_options(
                            dataset.clone(),
                            "vector",
                            0.0,
                            DuplicatePairsOptions::default()
                                .with_memory_limit(memory)
                                .with_max_concurrency(2)
                                .with_sharding(rows, shard, 4),
                        )
                        .await
                        .unwrap(),
                    )
                    .await,
                );
            }
            actual.sort_unstable();
            assert_eq!(actual, expected, "memory={memory} rows={rows}");
        }
    }
    for (rows, index, count) in [(0, 0, 1), (32, 0, 0), (32, 2, 2)] {
        let error = find_duplicate_pairs_with_options(
            dataset.clone(),
            "vector",
            0.0,
            DuplicatePairsOptions::default().with_sharding(rows, index, count),
        )
        .await
        .err()
        .unwrap();
        assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
        assert!(error.to_string().contains("shard"));
    }
}
