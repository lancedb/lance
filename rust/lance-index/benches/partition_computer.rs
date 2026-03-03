// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark IVF partition assignment across partition counts and dimensions.
//!
//! Partition counts span sqrt(1_000) ≈ 32 to sqrt(10^9) ≈ 31_623,
//! matching real-world IVF configurations for datasets from 1K to 10B rows.
//! Vector count per batch is 4× the partition count.
//! Dimensions: 128, 768, 1536, 4096.

use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use lance_arrow::FixedSizeListArrayExt;
use lance_index::vector::ivf::IvfTransformer;
use lance_index::vector::transform::Transformer;
use lance_linalg::distance::DistanceType;
use lance_testing::datagen::generate_random_array_with_seed;
use std::sync::Arc;
use std::time::Duration;

#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

const SEED: [u8; 32] = [42; 32];
const VECTOR_COLUMN: &str = "vector";

const DIMENSIONS: &[usize] = &[128, 768, 1536, 4096];

/// Build a RecordBatch with a single FixedSizeList vector column.
fn make_input_batch(num_vectors: usize, dim: usize) -> RecordBatch {
    let flat = generate_random_array_with_seed::<Float32Type>(num_vectors * dim, SEED);
    let fsl = FixedSizeListArray::try_new_from_values(flat, dim as i32).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        VECTOR_COLUMN,
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
        ),
        false,
    )]));
    RecordBatch::try_new(schema, vec![Arc::new(fsl)]).unwrap()
}

fn bench_partition_computer(c: &mut Criterion) {
    let mut group = c.benchmark_group("partition_computer");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));

    // sqrt(1_000) ≈ 32, sqrt(10_000) = 100, sqrt(100_000) ≈ 316,
    // sqrt(1_000_000) = 1000, sqrt(10_000_000) ≈ 3162,
    // sqrt(100_000_000) = 10000, sqrt(1_000_000_000) ≈ 31623
    let partition_counts: Vec<usize> = [
        1_000u64, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000,
    ]
    .iter()
    .map(|&n| (n as f64).sqrt() as usize)
    .collect();

    for &dim in DIMENSIONS {
        for &num_partitions in &partition_counts {
            let num_vectors = num_partitions * 4;

            // Skip configurations that would allocate > ~4GB total
            // (centroids + vectors, 4 bytes per f32)
            let total_floats = (num_partitions + num_vectors) * dim;
            if total_floats > 1_000_000_000 {
                continue;
            }

            let centroids_flat =
                generate_random_array_with_seed::<Float32Type>(num_partitions * dim, SEED);
            let centroids =
                FixedSizeListArray::try_new_from_values(centroids_flat, dim as i32).unwrap();

            let batch = make_input_batch(num_vectors, dim);

            let ivf = IvfTransformer::new_partition_transformer(
                centroids,
                DistanceType::L2,
                VECTOR_COLUMN,
            );

            group.bench_with_input(
                BenchmarkId::new(
                    format!("d={}", dim),
                    format!("k={}/n={}", num_partitions, num_vectors),
                ),
                &num_partitions,
                |b, _| {
                    b.iter(|| {
                        ivf.transform(&batch).unwrap();
                    });
                },
            );
        }
    }

    group.finish();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_partition_computer
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1));
    targets = bench_partition_computer
);

criterion_main!(benches);
