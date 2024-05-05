// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::iter::repeat_with;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lance_linalg::distance::hamming::hamming;
use rand::Rng;

const DIMENSION: usize = 1024;
const TOTAL: usize = 1024 * 1024; // 1M vectors

#[inline]
fn hamming_autovec(x: &[u8], y: &[u8]) -> f32 {
    let x_chunk = x.chunks_exact(32);
    let y_chunk = y.chunks_exact(32);
    let sum = x_chunk
        .remainder()
        .iter()
        .zip(y_chunk.remainder())
        .map(|(&a, &b)| (a ^ b).count_ones())
        .sum::<u32>();
    (sum + x_chunk
        .zip(y_chunk)
        .map(|(x, y)| {
            x.iter().zip(y.iter()).map(|(&a, &b)| (a ^ b).count_ones()).sum::<u32>()
        })
        .sum::<u32>()) as f32
}

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
                    .map(|tgt| hamming(&key, tgt))
                    .sum::<f32>(),
            );
        })
    });

    c.bench_function("hamming,auto_vec", |b| {
        b.iter(|| {
            black_box(
                target
                    .chunks_exact(DIMENSION)
                    .map(|tgt| hamming_autovec(&key, tgt))
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
