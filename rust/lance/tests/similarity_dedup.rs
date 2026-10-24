// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
    UInt64Array,
};
use futures::TryStreamExt;
use lance::Dataset;
use lance::index::vector::similarity_dedup::{
    SimilaritySearchOptions, find_similar_pairs_in_partition,
};
use lance::index::vector::similarity_dedup::{
    SimilaritySearchSharding, find_similar_pairs_in_partition_with_sharding,
};
use lance::index::{DatasetIndexExt, vector::VectorIndexParams};
use lance_arrow::FixedSizeListArrayExt;
use lance_index::{IndexType, vector::ivf::IvfBuildParams};
use lance_linalg::distance::DistanceType;

async fn dataset(values: Vec<f32>, centers: Vec<f32>) -> Arc<Dataset> {
    let vectors = FixedSizeListArray::try_new_from_values(Float32Array::from(values), 1).unwrap();
    let batch = RecordBatch::try_from_iter([("vector", Arc::new(vectors) as ArrayRef)]).unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        None,
    )
    .await
    .unwrap();
    let centers =
        FixedSizeListArray::try_new_from_values(Float32Array::from(centers.clone()), 1).unwrap();
    let ivf = IvfBuildParams::try_with_centroids(centers.len(), Arc::new(centers)).unwrap();
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            None,
            &VectorIndexParams::with_ivf_flat_params(DistanceType::L2, ivf),
            false,
        )
        .await
        .unwrap();
    Arc::new(dataset)
}

async fn pairs(
    dataset: Arc<Dataset>,
    options: SimilaritySearchOptions,
    threshold: f32,
) -> Vec<(u64, u64)> {
    let meta = dataset.load_indices().await.unwrap();
    let segment = meta[0].uuid;
    let count = if dataset.count_rows(None).await.unwrap() > 100 {
        1
    } else {
        2
    };
    let mut pairs = Vec::new();
    for partition in 0..count {
        let mut stream = find_similar_pairs_in_partition(
            dataset.clone(),
            "vector",
            segment,
            partition,
            threshold,
            options,
        )
        .await
        .unwrap();
        while let Some(batch) = stream.try_next().await.unwrap() {
            let a = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let b = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            pairs.extend(a.values().iter().zip(b.values()).map(|(&a, &b)| (a, b)));
        }
    }
    pairs
}

#[tokio::test]
async fn auto_probes_cross_centroid_boundary() {
    let dataset = dataset(vec![-0.01, 0.01, -1.0, 1.0], vec![-1.0, 1.0]).await;
    let fixed = pairs(
        dataset.clone(),
        SimilaritySearchOptions {
            nprobes: Some(1),
            max_nprobes: None,
        },
        0.001,
    )
    .await;
    assert!(fixed.is_empty());
    let auto = pairs(dataset, SimilaritySearchOptions::default(), 0.001).await;
    assert_eq!(
        auto.into_iter().collect::<BTreeSet<_>>(),
        BTreeSet::from([(0, 1)])
    );
}

#[tokio::test]
async fn radius_has_no_top_k_truncation() {
    let dataset = dataset(vec![1.0; 151], vec![1.0]).await;
    let pairs = pairs(dataset, SimilaritySearchOptions::default(), 0.0).await;
    assert_eq!(pairs.len(), 151 * 150);
    assert_eq!(
        pairs.into_iter().collect::<BTreeSet<_>>().len(),
        151 * 150 / 2
    );
    assert!(
        SimilaritySearchOptions {
            nprobes: Some(0),
            max_nprobes: None
        }
        .validate()
        .is_err()
    );
    assert!(
        SimilaritySearchOptions {
            nprobes: Some(3),
            max_nprobes: Some(2)
        }
        .validate()
        .is_err()
    );
}

#[tokio::test]
async fn source_and_target_shards_preserve_all_cross_block_edges() {
    for quantized in [false, true] {
        let vectors =
            FixedSizeListArray::from_iter_primitive::<arrow_array::types::Float32Type, _, _>(
                (0..19).map(|i| {
                    let mut vector = vec![Some(1.0); 8];
                    vector[0] = Some((i % 5) as f32);
                    Some(vector)
                }),
                8,
            );
        let batch =
            RecordBatch::try_from_iter([("vector", Arc::new(vectors) as ArrayRef)]).unwrap();
        let mut ds = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            "memory://",
            None,
        )
        .await
        .unwrap();
        let params = if quantized {
            VectorIndexParams::ivf_rq(1, 4, DistanceType::L2)
        } else {
            VectorIndexParams::ivf_flat(1, DistanceType::L2)
        };
        ds.create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();
        let ds = Arc::new(ds);
        let segment = ds.load_indices().await.unwrap()[0].uuid;
        let mut expected = Vec::new();
        let mut stream = find_similar_pairs_in_partition(
            ds.clone(),
            "vector",
            segment,
            0,
            1000.0,
            SimilaritySearchOptions::default(),
        )
        .await
        .unwrap();
        while let Some(batch) = stream.try_next().await.unwrap() {
            let a = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let b = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            expected.extend(a.values().iter().zip(b.values()).map(|(&a, &b)| (a, b)));
        }
        expected.sort_unstable();
        assert_eq!(expected.len(), 19 * 18);
        for rows in [1, 3, 8] {
            let mut actual = Vec::new();
            for worker in 0..4 {
                let mut stream = find_similar_pairs_in_partition_with_sharding(
                    ds.clone(),
                    "vector",
                    segment,
                    0,
                    1000.0,
                    SimilaritySearchOptions::default(),
                    SimilaritySearchSharding {
                        rows,
                        index: worker,
                        count: 4,
                    },
                )
                .await
                .unwrap();
                while let Some(batch) = stream.try_next().await.unwrap() {
                    let a = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .unwrap();
                    let b = batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .unwrap();
                    actual.extend(a.values().iter().zip(b.values()).map(|(&a, &b)| (a, b)));
                }
            }
            actual.sort_unstable();
            assert_eq!(actual, expected, "quantized={quantized},rows={rows}");
        }
        assert!(
            SimilaritySearchSharding {
                rows: 0,
                index: 0,
                count: 1
            }
            .validate()
            .is_err()
        );
        assert!(
            SimilaritySearchSharding {
                rows: 8,
                index: 4,
                count: 4
            }
            .validate()
            .is_err()
        );
    }
}
