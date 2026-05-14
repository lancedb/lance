// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::array::AsArray;
use arrow::datatypes::Float32Type;
use arrow_array::FixedSizeListArray;
use criterion::{Criterion, criterion_group, criterion_main};

use lance_arrow::FixedSizeListArrayExt;
use lance_index::vector::utils::SimpleIndex;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use lance_index::vector::kmeans::{
    KMeans, KMeansAlgo, KMeansAlgoFloat, KMeansParams, compute_partitions_arrow_array,
};
use lance_linalg::distance::DistanceType;
use lance_testing::datagen::generate_random_array;

fn bench_train(c: &mut Criterion) {
    let params = [
        // (64 * 1024, 8),      // training PQ
        // (64 * 1024, 128),    // training IVF with small vectors (1M rows)
        // (64 * 1024, 1024),   // training IVF with large vectors (1M rows)
        // (256 * 1024, 1024),  // hit the threshold for using HNSW to speed up
        // (256 * 2048, 1024),  // hit the threshold for using HNSW to speed up
        // (256 * 4096, 1024),  // hit the threshold for using HNSW to speed up
        (256 * 16384, 1024), // hit the threshold for using HNSW to speed up
    ];
    for (n, dimension) in params {
        let k = n / 256;

        let values = generate_random_array(n * dimension as usize);
        let data = FixedSizeListArray::try_new_from_values(values, dimension).unwrap();

        let values = generate_random_array(k * dimension as usize);
        let centroids = FixedSizeListArray::try_new_from_values(values, dimension).unwrap();

        c.bench_function(&format!("train_{}d_{}k", dimension, k), |b| {
            let params = KMeansParams::default().with_hierarchical_k(0);
            b.iter(|| {
                KMeans::new_with_params(&data, k, &params).ok().unwrap();
            })
        });

        if k > 256 {
            for hierarchical_k in [4, 8, 16, 24, 32] {
                let params = KMeansParams::default().with_hierarchical_k(hierarchical_k);
                c.bench_function(
                    &format!(
                        "train_{}d_{}k_hierarchical_{}",
                        dimension, k, hierarchical_k
                    ),
                    |b| {
                        b.iter(|| KMeans::new_with_params(&data, k, &params).ok().unwrap());
                    },
                );
            }
        }

        let mut group = c.benchmark_group(format!("compute_membership_{}d_{}k", dimension, k));

        group.bench_function("flat", |b| {
            b.iter(|| compute_partitions_arrow_array(&centroids, &data, DistanceType::L2))
        });

        if k * dimension as usize >= 1_000_000 {
            let index = SimpleIndex::may_train_index(
                centroids.values().clone(),
                dimension as usize,
                DistanceType::L2,
            )
            .unwrap()
            .unwrap();
            group.bench_function("with_index", |b| {
                b.iter(|| {
                    KMeansAlgoFloat::<Float32Type>::compute_membership_and_loss(
                        centroids.values().as_primitive::<Float32Type>().values(),
                        data.values().as_primitive::<Float32Type>().values(),
                        dimension as usize,
                        DistanceType::L2,
                        0.0,
                        None,
                        Some(&index),
                    )
                })
            });
        }
    }
}

/// End-to-end train_kmeans: scalar vs SGEMM at PQ-typical configuration.
///
/// This is the correct level to measure the SGEMM benefit: a single call amortises
/// the N×K inner-matrix allocation across all K-means iterations, rather than
/// paying the allocator overhead once per benchmark sample.
///
/// Config mirrors standard PQ sub-vector training:
///   N = 50k rows, dim = 8 (sub_dim = 128 / 16 = 8), K = 256 centroids, max 50 iters.
///
/// SGEMM is now activated automatically when N×K×4 ≤ LANCE_SGEMM_THRESHOLD (default 64 MiB).
/// The "scalar" variant uses LANCE_SGEMM_THRESHOLD=0 to force the scalar path.
fn bench_sgemm_train_kmeans(c: &mut Criterion) {
    const N: usize = 50_000;
    const SUB_DIM: usize = 8;
    const K: usize = 256;

    let data_arr = generate_random_array(N * SUB_DIM);
    let data_fsl = FixedSizeListArray::try_new_from_values(data_arr, SUB_DIM as i32).unwrap();

    let mut group = c.benchmark_group(format!("sgemm_train_kmeans_dim{}_n{}_k{}", SUB_DIM, N, K));
    group.sample_size(10);

    group.bench_function("scalar", |b| {
        let params = KMeansParams::new(None, 50, 1, DistanceType::L2);
        // Force scalar path via env var (threshold=0 disables SGEMM).
        // SAFETY: single-threaded benchmark context.
        unsafe { std::env::set_var("LANCE_SGEMM_THRESHOLD", "0") };
        b.iter(|| KMeans::new_with_params(&data_fsl, K, &params).ok().unwrap())
    });
    unsafe { std::env::remove_var("LANCE_SGEMM_THRESHOLD") };

    group.bench_function("sgemm", |b| {
        let params = KMeansParams::new(None, 50, 1, DistanceType::L2);
        // Default: SGEMM activates automatically via memory budget.
        b.iter(|| KMeans::new_with_params(&data_fsl, K, &params).ok().unwrap())
    });
}

/// Micro-benchmark: single assignment call, scalar vs SGEMM, sweep sub_dim.
///
/// With the memory-budget gate (`N×K×4 ≤ LANCE_SGEMM_THRESHOLD`, default 64 MiB),
/// SGEMM activates for all sub_dim values at K=256 and N=50k
/// (50k × 256 × 4 = 51 MB < 64 MB). The "scalar" variant uses
/// LANCE_SGEMM_THRESHOLD=0 to force the scalar path for comparison.
///
/// Interpret with care: the per-call N×K allocation overhead is amortised in
/// train_kmeans but dominates here at small sample counts.
fn bench_sgemm_assignment(c: &mut Criterion) {
    const N: usize = 50_000;
    const K: usize = 256;
    for sub_dim in [8_usize, 16, 32, 64, 128] {
        let data_arr = generate_random_array(N * sub_dim);
        let cents_arr = generate_random_array(K * sub_dim);
        let data: &[f32] = data_arr.values();
        let cents: &[f32] = cents_arr.values();

        let mut group = c.benchmark_group(format!("sgemm_assign_dim{}_n{}_k{}", sub_dim, N, K));
        group.sample_size(10);

        group.bench_function("scalar", |b| {
            // SAFETY: single-threaded benchmark context.
            unsafe { std::env::set_var("LANCE_SGEMM_THRESHOLD", "0") };
            b.iter(|| {
                KMeansAlgoFloat::<Float32Type>::compute_membership_and_dist(
                    cents,
                    data,
                    sub_dim,
                    DistanceType::L2,
                    0.0,
                    None,
                    None,
                )
            })
        });
        unsafe { std::env::remove_var("LANCE_SGEMM_THRESHOLD") };

        group.bench_function("sgemm", |b| {
            // Default: SGEMM activates automatically via memory budget.
            b.iter(|| {
                KMeansAlgoFloat::<Float32Type>::compute_membership_and_dist(
                    cents,
                    data,
                    sub_dim,
                    DistanceType::L2,
                    0.0,
                    None,
                    None,
                )
            })
        });
    }
}

/// Budget-vs-performance sweep: vary N (and thus matrix_bytes = N × K × 4) to find
/// where SGEMM stops being faster than scalar.
///
/// When `matrix_bytes ≤ L3_cache`, the SGEMM `inner[N,K]` matrix fits in cache and
/// the argmin pass is fast.  Once it exceeds L3, the argmin generates streaming cache
/// misses and the allocation overhead dominates, causing scalar to win.
///
/// This bench uses the same `LANCE_SGEMM_THRESHOLD` trick to compare paths at each N.
/// Run with: `cargo bench -p lance-index --bench kmeans -- sgemm_budget`
fn bench_sgemm_budget_vs_cache(c: &mut Criterion) {
    const DIM: usize = 8; // typical PQ sub_dim
    const K: usize = 256;

    // N sweep to characterise performance as the inner[N,K] matrix grows.
    // matrix_bytes = N × 256 × 4:
    //   N=10k  →   9 MB  (fits in most L3 caches)
    //   N=50k  →  48 MB  (default 64 MB threshold is near here)
    //   N=100k →  97 MB  (well above L3, tests streaming behaviour)
    //   N=200k → 195 MB  (far above L3 — validates M-series still benefits from SGEMM)
    for n in [10_000_usize, 50_000, 100_000, 200_000] {
        let matrix_bytes = n * K * std::mem::size_of::<f32>();
        let data_arr = generate_random_array(n * DIM);
        let cents_arr = generate_random_array(K * DIM);
        let data: &[f32] = data_arr.values();
        let cents: &[f32] = cents_arr.values();

        let label = format!(
            "sgemm_budget_n{}_k{}_matrix{}MB",
            n,
            K,
            matrix_bytes / (1024 * 1024)
        );
        let mut group = c.benchmark_group(label);
        group.sample_size(10);

        group.bench_function("scalar", |b| {
            // SAFETY: single-threaded benchmark context.
            unsafe { std::env::set_var("LANCE_SGEMM_THRESHOLD", "0") };
            b.iter(|| {
                KMeansAlgoFloat::<Float32Type>::compute_membership_and_dist(
                    cents,
                    data,
                    DIM,
                    DistanceType::L2,
                    0.0,
                    None,
                    None,
                )
            })
        });
        unsafe { std::env::remove_var("LANCE_SGEMM_THRESHOLD") };

        group.bench_function("sgemm", |b| {
            // Force SGEMM on regardless of default threshold (raise to 1 GiB).
            unsafe { std::env::set_var("LANCE_SGEMM_THRESHOLD", "1073741824") };
            b.iter(|| {
                KMeansAlgoFloat::<Float32Type>::compute_membership_and_dist(
                    cents,
                    data,
                    DIM,
                    DistanceType::L2,
                    0.0,
                    None,
                    None,
                )
            })
        });
        unsafe { std::env::remove_var("LANCE_SGEMM_THRESHOLD") };
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
    .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_train, bench_sgemm_train_kmeans, bench_sgemm_assignment,
              bench_sgemm_budget_vs_cache);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_train, bench_sgemm_train_kmeans, bench_sgemm_assignment,
              bench_sgemm_budget_vs_cache);
criterion_main!(benches);
