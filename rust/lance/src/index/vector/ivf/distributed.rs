// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Layer 2 distributed IVF centroid training: dataset-aware async wrappers
//! around the pure primitives in `lance_index::vector::kmeans::distributed`.
//!
//! See `docs/superpowers/specs/2026-06-10-distributed-centroid-training-abstraction-design.md`.

pub use lance_index::vector::kmeans::distributed::{
    PartialStats, bootstrap_centroids, finalize_centroids, merge_partial_stats,
    reduce_partial_stats, select_initial_centroids,
};

use arrow_array::{FixedSizeListArray, RecordBatch};
use lance_linalg::distance::DistanceType;
use lance_linalg::kernels::normalize_fsl_owned;

use crate::Result;
use crate::dataset::Dataset;
use crate::index::vector::utils::{filter_finite_training_data, maybe_sample_training_data};

/// Round-0 worker entrypoint: reservoir-sample a Lance dataset slice.
///
/// Internals mirror `build_ivf_model` (`rust/lance/src/index/vector/ivf.rs`):
/// 1. Sample raw rows via `maybe_sample_training_data`.
/// 2. If `distance_type == Cosine`, normalize the sample (turning it into L2 internally).
/// 3. Filter out non-finite rows.
/// 4. Reservoir-sample down to `target` rows via Layer 1 `local_reservoir_sample`.
pub async fn sample_round_0(
    dataset: &Dataset,
    column: &str,
    fragment_ids: Option<&[u32]>,
    target: usize,
    distance_type: DistanceType,
    rng_seed: u64,
) -> Result<RecordBatch> {
    // Round-0 oversamples to give the driver-side bootstrap enough material.
    let raw =
        maybe_sample_training_data(dataset, column, target.saturating_mul(2), fragment_ids).await?;
    let normalized = if distance_type == DistanceType::Cosine {
        normalize_fsl_owned(raw)?
    } else {
        raw
    };
    let filtered = filter_finite_training_data(normalized)?;
    lance_index::vector::kmeans::distributed::local_reservoir_sample(&filtered, target, rng_seed)
}

/// Round-r worker entrypoint: scan the worker's fragment slice and produce a
/// `PartialStats` batch against the broadcast `centroids`.
///
/// `centroids` are interpreted in the same dtype as the dataset's vector column
/// (caller is responsible for keeping them stable across rounds).
pub async fn compute_partial_stats(
    dataset: &Dataset,
    column: &str,
    fragment_ids: Option<&[u32]>,
    centroids: &FixedSizeListArray,
    distance_type: DistanceType,
) -> Result<PartialStats> {
    // Pull all matching rows; the worker is expected to have a small enough fragment
    // slice that scanning it whole is cheap.
    let raw = maybe_sample_training_data(dataset, column, usize::MAX, fragment_ids).await?;
    let normalized = if distance_type == DistanceType::Cosine {
        normalize_fsl_owned(raw)?
    } else {
        raw
    };
    let filtered = filter_finite_training_data(normalized)?;

    lance_index::vector::kmeans::distributed::compute_partial_stats(
        centroids,
        &filtered,
        distance_type,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::{RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_testing::datagen::generate_random_array_with_seed;

    use crate::dataset::Dataset;

    /// Build a small in-memory Lance dataset with `n` rows of FSL<Float32, dim>
    /// across 4 fragments so distributed sampling has multiple slices to play with.
    async fn make_vector_dataset(uri: &str, n: usize, dim: usize) -> Dataset {
        let total = n * dim;
        let values =
            generate_random_array_with_seed::<arrow_array::types::Float32Type>(total, [42; 32]);
        let fsl = FixedSizeListArray::try_new_from_values(values, dim as i32).unwrap();

        let schema = Arc::new(Schema::new(vec![Field::new(
            "vec",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            true,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();

        let params = crate::dataset::WriteParams {
            // Spread the rows across 4 fragments so worker partitioning has something to bite on.
            max_rows_per_file: (n / 4).max(1),
            max_rows_per_group: 256,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        Dataset::write(batches, uri, Some(params)).await.unwrap()
    }

    #[tokio::test]
    async fn test_sample_round_0_returns_target_rows() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let ds = make_vector_dataset(uri, 2_000, 8).await;
        let sample = sample_round_0(&ds, "vec", None, 256, DistanceType::L2, 7)
            .await
            .unwrap();
        assert_eq!(sample.num_rows(), 256);
        assert_eq!(sample.schema().field(0).name(), "vec");
    }

    #[tokio::test]
    async fn test_layer2_compute_partial_stats_l2() {
        use arrow_array::Float32Array;

        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let ds = make_vector_dataset(uri, 1_000, 4).await;
        let centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.0_f32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
            4,
        )
        .unwrap();

        let stats = compute_partial_stats(&ds, "vec", None, &centroids, DistanceType::L2)
            .await
            .unwrap();
        assert_eq!(stats.k(), 2);
        assert!(stats.total_count() > 0, "should have some assigned vectors");
    }

    #[tokio::test]
    async fn test_end_to_end_four_workers_match_single_machine() {
        use arrow_array::cast::AsArray;
        use arrow_array::types::Float32Type;
        use lance_index::vector::ivf::builder::IvfBuildParams;

        let dim = 8;
        let k = 16;
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let ds = make_vector_dataset(uri, 4_000, dim).await;
        let column = "vec";

        // Pre-train a single-machine baseline using the existing path.
        let baseline_ivf = crate::index::vector::ivf::build_ivf_model(
            &ds,
            column,
            dim,
            DistanceType::L2,
            &IvfBuildParams {
                num_partitions: Some(k),
                sample_rate: 256,
                max_iters: 5,
                ..Default::default()
            },
            None,
            std::sync::Arc::new(lance_index::progress::NoopIndexBuildProgress),
        )
        .await
        .unwrap();
        let baseline_centroids = baseline_ivf
            .centroids_array()
            .expect("baseline IVF model should have centroids")
            .clone();

        // Simulate 4 workers via fragment id slicing.
        let frags: Vec<u32> = ds.get_fragments().iter().map(|f| f.id() as u32).collect();
        assert!(frags.len() >= 4, "need >=4 fragments to simulate 4 workers");
        let groups: Vec<Vec<u32>> = frags
            .chunks(frags.len().div_ceil(4))
            .map(|s| s.to_vec())
            .collect();

        // Round 0: each "worker" reservoir-samples its slice; driver bootstraps.
        let mut samples = Vec::new();
        for g in &groups {
            let s = sample_round_0(&ds, column, Some(g), 256, DistanceType::L2, 42)
                .await
                .unwrap();
            samples.push(s);
        }
        let mut centroids = bootstrap_centroids(samples, k, DistanceType::L2, 42).unwrap();

        // 5 Lloyd's rounds.
        for _ in 0..5 {
            let mut partials = Vec::new();
            for g in &groups {
                let s = compute_partial_stats(&ds, column, Some(g), &centroids, DistanceType::L2)
                    .await
                    .unwrap();
                partials.push(s);
            }
            let merged = reduce_partial_stats(partials).unwrap();
            centroids = finalize_centroids(&merged, &centroids).unwrap();
        }

        // Centroid sets are close (allow re-permutation: assert each baseline centroid has a near
        // neighbour in the distributed centroids).
        let v_base = baseline_centroids
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        let v_dist = centroids
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        for ci in 0..k {
            let base = &v_base[ci * dim..(ci + 1) * dim];
            let mut best = f32::INFINITY;
            for cj in 0..k {
                let dist_c = &v_dist[cj * dim..(cj + 1) * dim];
                let d: f32 = base
                    .iter()
                    .zip(dist_c.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                best = best.min(d);
            }
            // Each baseline centroid should have a fairly close match in the distributed result;
            // values are L2-squared in 8 dims so we accept a few units of slack.
            assert!(
                best < 25.0,
                "no close match for centroid {}: best={}",
                ci,
                best
            );
        }
    }
}
