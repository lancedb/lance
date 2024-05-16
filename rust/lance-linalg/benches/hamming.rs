// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::iter::repeat_with;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lance_linalg::distance::hamming::{hamming, hamming_scalar};
use rand::Rng;

const DIMENSION: usize = 1024;
const TOTAL: usize = 1024 * 1024; // 1M vectors

fn bench_hamming(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let key = repeat_with(|| rng.gen::<u8>())
        .take(DIMENSION)
        .collect::<Vec<_>>();
    let target = repeat_with(|| rng.gen::<u8>())
        .take(TOTAL * DIMENSION)
        .collect::<Vec<_>>();

    c.bench_function("hamming,scalar", |b| {
        b.iter(|| {
            black_box(
                target
                    .chunks_exact(DIMENSION)
                    .map(|tgt| hamming_scalar(&key, tgt))
                    .sum::<f32>(),
            );
        })
    });

    c.bench_function("hamming,auto_vec", |b| {
        b.iter(|| {
            black_box(
                target
                    .chunks_exact(DIMENSION)
                    .map(|tgt| hamming(&key, tgt))
                    .sum::<f32>(),
            );
        })
    });
}

criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_hamming);
criterion_main!(benches);
