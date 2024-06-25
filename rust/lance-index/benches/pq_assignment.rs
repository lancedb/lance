// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark of Building PQ code from Dense Vectors.

use std::sync::Arc;

use arrow_array::{types::Float32Type, FixedSizeListArray};
use criterion::{criterion_group, criterion_main, Criterion};
use lance_arrow::FixedSizeListArrayExt;
use lance_index::vector::pq::{ProductQuantizer, ProductQuantizerImpl};
use lance_linalg::distance::DistanceType;
use lance_testing::datagen::generate_random_array_with_seed;

const PQ: usize = 96;
const DIM: usize = 1536;
const TOTAL: usize = 32 * 1024;

fn pq_transform(c: &mut Criterion) {
    let codebook = Arc::new(generate_random_array_with_seed::<Float32Type>(
        256 * DIM,
        [88; 32],
    ));

    let vectors = generate_random_array_with_seed::<Float32Type>(DIM * TOTAL, [3; 32]);
    let fsl = FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap();

    for dt in [DistanceType::L2, DistanceType::Dot].iter() {
        let pq = ProductQuantizerImpl::<Float32Type>::new(PQ, 8, DIM, codebook.clone(), *dt);

        c.bench_function(format!("{},{}", dt, TOTAL).as_str(), |b| {
            b.iter(|| {
                let _ = pq.transform(&fsl).unwrap();
            })
        });
    }
}

criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = pq_transform);

criterion_main!(benches);
