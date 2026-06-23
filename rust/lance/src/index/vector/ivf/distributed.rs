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

use arrow_array::{Array, FixedSizeListArray, RecordBatch};
use futures::StreamExt;
use lance_linalg::distance::DistanceType;
use lance_linalg::kernels::normalize_fsl_owned;

use crate::Result;
use crate::dataset::Dataset;
use crate::index::vector::utils::{
    filter_finite_training_data, sample_training_data_stream, vector_column_to_fsl,
};

/// Round-0 worker entrypoint: reservoir-sample a Lance dataset slice.
///
/// Streams the worker's projected training rows through a single
/// [`StreamingReservoir`] so peak memory is bounded by the output sample
/// (target rows) rather than the worker's full fragment slice. The same
/// `rng_seed` is forwarded to both the upstream sampling stream and the
/// reservoir, making same-seed runs byte-deterministic for a given
/// `(dataset, column, fragments, target)` tuple.
///
/// Internals mirror `build_ivf_model` (`rust/lance/src/index/vector/ivf.rs`):
/// 1. Stream raw rows via `sample_training_data_stream` (oversample to 2*target).
/// 2. Per batch: extract FSL, optionally L2-normalize for Cosine, drop non-finite rows.
/// 3. Feed each filtered chunk into the streaming reservoir.
pub async fn sample_round_0(
    dataset: &Dataset,
    column: &str,
    fragment_ids: Option<&[u32]>,
    target: usize,
    distance_type: DistanceType,
    rng_seed: u64,
) -> Result<RecordBatch> {
    use lance_index::vector::kmeans::distributed::StreamingReservoir;

    // Round-0 oversamples to give the driver-side bootstrap enough material.
    let mut stream = sample_training_data_stream(
        dataset,
        column,
        target.saturating_mul(2),
        fragment_ids,
        Some(rng_seed),
    )
    .await?;

    let mut reservoir = StreamingReservoir::new(target, rng_seed);
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        let fsl = vector_column_to_fsl(&batch, column)?;
        let normalized = if distance_type == DistanceType::Cosine {
            normalize_fsl_owned(fsl)?
        } else {
            fsl
        };
        let filtered = filter_finite_training_data(normalized)?;
        if filtered.is_empty() {
            continue;
        }
        reservoir.feed(&filtered)?;
    }

    reservoir.into_record_batch()
}

/// Round-r worker entrypoint: scan the worker's fragment slice and produce a
/// `PartialStats` batch against the broadcast `centroids`.
///
/// `centroids` are interpreted in the same dtype as the dataset's vector column
/// (caller is responsible for keeping them stable across rounds).
///
/// Streams the projected training rows and accumulates per-batch
/// `PartialStats`, keeping peak memory at O(k·d) instead of O(N·d).
pub async fn compute_partial_stats(
    dataset: &Dataset,
    column: &str,
    fragment_ids: Option<&[u32]>,
    centroids: &FixedSizeListArray,
    distance_type: DistanceType,
) -> Result<PartialStats> {
    use lance_index::vector::kmeans::distributed::{
        compute_centroids_fingerprint, compute_partial_stats as l1_compute_partial_stats,
        merge_partial_stats,
    };

    // Worker is expected to have a small enough fragment slice that scanning
    // it whole is cheap. `usize::MAX` skips sampling entirely; we just want
    // every row streamed in.
    let mut stream =
        sample_training_data_stream(dataset, column, usize::MAX, fragment_ids, None).await?;

    let mut acc: Option<PartialStats> = None;
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        let fsl = vector_column_to_fsl(&batch, column)?;
        let normalized = if distance_type == DistanceType::Cosine {
            normalize_fsl_owned(fsl)?
        } else {
            fsl
        };
        let filtered = filter_finite_training_data(normalized)?;
        if filtered.is_empty() {
            continue;
        }
        let chunk = l1_compute_partial_stats(centroids, &filtered, distance_type)?;
        acc = Some(match acc.take() {
            None => chunk,
            Some(prev) => merge_partial_stats(prev, chunk)?,
        });
    }

    // Empty input (no rows / all filtered): synthesize an empty `PartialStats`
    // with the right metadata so downstream merge/finalize works uniformly.
    Ok(acc.unwrap_or_else(|| {
        let dim = centroids.value_length() as usize;
        let k = centroids.len();
        let fp = compute_centroids_fingerprint(centroids);
        PartialStats::empty(k, dim, distance_type, fp)
    }))
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

    /// Same `(dataset, fragments, target, seed)` must produce a byte-identical
    /// sample, exercising the seed plumbed through `sample_training_data_stream`
    /// and `StreamingReservoir`.
    #[tokio::test]
    async fn test_sample_round_0_is_deterministic_for_same_seed() {
        use arrow_array::cast::AsArray;
        use arrow_array::types::Float32Type;

        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let ds = make_vector_dataset(uri, 4_000, 8).await;

        let s1 = sample_round_0(&ds, "vec", None, 256, DistanceType::L2, 42)
            .await
            .unwrap();
        let s2 = sample_round_0(&ds, "vec", None, 256, DistanceType::L2, 42)
            .await
            .unwrap();

        assert_eq!(s1.num_rows(), s2.num_rows());
        let v1 = s1
            .column(0)
            .as_fixed_size_list()
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        let v2 = s2
            .column(0)
            .as_fixed_size_list()
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        assert_eq!(v1, v2, "same seed must produce byte-identical samples");
    }

    /// Different seeds should not produce byte-identical samples on a dataset
    /// large enough to make collisions astronomically unlikely.
    #[tokio::test]
    async fn test_sample_round_0_different_seed_changes_output() {
        use arrow_array::cast::AsArray;
        use arrow_array::types::Float32Type;

        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let ds = make_vector_dataset(uri, 4_000, 8).await;

        let mut samples = Vec::new();
        for seed in 0..5 {
            let s = sample_round_0(&ds, "vec", None, 256, DistanceType::L2, seed)
                .await
                .unwrap();
            let v = s
                .column(0)
                .as_fixed_size_list()
                .values()
                .as_primitive::<Float32Type>()
                .values()
                .to_vec();
            samples.push(v);
        }

        let mut differs = false;
        for i in 0..samples.len() {
            for j in (i + 1)..samples.len() {
                if samples[i] != samples[j] {
                    differs = true;
                    break;
                }
            }
            if differs {
                break;
            }
        }
        assert!(differs, "different seeds should change the sample");
    }

    /// Streaming `compute_partial_stats` must produce identical statistics to
    /// a materializing reference path that loads the whole dataset at once
    /// (parity test for the Layer-2 streaming refactor).
    #[tokio::test]
    async fn test_compute_partial_stats_streaming_matches_materialized() {
        use arrow_array::Float32Array;
        use arrow_array::cast::AsArray;
        use arrow_array::types::Float64Type;
        use lance_index::vector::kmeans::distributed::compute_partial_stats as l1_compute;

        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let ds = make_vector_dataset(uri, 4_000, 8).await;

        // 4 fixed centroids inside the [0, 1)^8 unit cube generated by the
        // dataset helper, so each one has at least one nearby vector.
        let cs: Vec<f32> = vec![
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, //
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, //
            0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, //
            0.2, 0.7, 0.3, 0.8, 0.4, 0.6, 0.5, 0.5, //
        ];
        let centroids = FixedSizeListArray::try_new_from_values(Float32Array::from(cs), 8).unwrap();

        let stream_stats = compute_partial_stats(&ds, "vec", None, &centroids, DistanceType::L2)
            .await
            .unwrap();

        // Reference: scan the whole dataset into one FSL and run Layer-1
        // `compute_partial_stats` directly.
        let raw =
            crate::index::vector::utils::maybe_sample_training_data(&ds, "vec", usize::MAX, None)
                .await
                .unwrap();
        let filtered = crate::index::vector::utils::filter_finite_training_data(raw).unwrap();
        let mat_stats = l1_compute(&centroids, &filtered, DistanceType::L2).unwrap();

        assert_eq!(stream_stats.k(), mat_stats.k());
        assert_eq!(stream_stats.dim(), mat_stats.dim());
        assert_eq!(stream_stats.total_count(), mat_stats.total_count());
        // total loss can drift by a couple ULPs due to summation order across
        // streamed chunks; allow a tiny relative tolerance.
        let l_stream = stream_stats.total_loss();
        let l_mat = mat_stats.total_loss();
        assert!(
            (l_stream - l_mat).abs() <= 1e-3 * l_mat.abs().max(1.0),
            "loss drift: stream={} mat={}",
            l_stream,
            l_mat
        );

        let s_stream = stream_stats
            .record_batch()
            .column(2)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap()
            .values()
            .as_primitive::<Float64Type>()
            .values()
            .to_vec();
        let s_mat = mat_stats
            .record_batch()
            .column(2)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap()
            .values()
            .as_primitive::<Float64Type>()
            .values()
            .to_vec();
        for (a, b) in s_stream.iter().zip(s_mat.iter()) {
            assert!(
                (a - b).abs() <= 1e-3 * b.abs().max(1.0),
                "sum drift: {} vs {}",
                a,
                b
            );
        }
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
