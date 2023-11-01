// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use arrow_arith::{aggregate::sum, numeric::mul};
use arrow_array::{cast::AsArray, types::Float32Type, Float32Array};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use lance_linalg::distance::norm_l2::Normalize;
use lance_testing::datagen::generate_random_array_with_seed;

#[inline]
fn norm_l2_arrow(x: &Float32Array) -> f32 {
    let m = mul(&x, &x).unwrap();
    sum(m.as_primitive::<Float32Type>()).unwrap()
}

#[inline]
fn norm_l2_auto_vectorization(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>()
}

fn bench_distance(c: &mut Criterion) {
    const DIMENSION: usize = 1024;
    const TOTAL: usize = 1024 * 1024; // 1M vectors

    // 1M of 1024 D vectors. 4GB in memory.
    let target = generate_random_array_with_seed(TOTAL * DIMENSION, [42; 32]);

    c.bench_function("norm_l2(arrow)", |b| {
        b.iter(|| unsafe {
            Float32Array::from_trusted_len_iter((0..target.len() / DIMENSION).map(|idx| {
                let arr = target.slice(idx * DIMENSION, DIMENSION);
                Some(norm_l2_arrow(&arr))
            }))
        });
    });

    c.bench_function("norm_l2(auto-vectorization)", |b| {
        b.iter(|| unsafe {
            Float32Array::from_trusted_len_iter((0..target.len() / DIMENSION).map(|idx| {
                let arr = target.slice(idx * DIMENSION, DIMENSION);
                Some(norm_l2_auto_vectorization(arr.values()))
            }))
        });
    });

    c.bench_function("norm_l2(SIMD)", |b| {
        b.iter(|| unsafe {
            Float32Array::from_trusted_len_iter((0..target.len() / DIMENSION).map(|idx| {
                let arr = &target.values()[idx * DIMENSION..(idx + 1) * DIMENSION];
                Some(arr.norm_l2())
            }))
        });
    });
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_distance);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_distance);

criterion_main!(benches);
