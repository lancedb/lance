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

use arrow_arith::{
    aggregate::sum,
    arity::binary,
    numeric::{mul, sub},
};
use arrow_array::{
    cast::{as_primitive_array, AsArray},
    types::Float32Type,
    Float32Array,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use lance_linalg::distance::{l2_distance_batch, l2_scalar};
use lance_testing::datagen::generate_random_array_with_seed;

#[inline]
fn l2_arrow(x: &Float32Array, y: &Float32Array) -> f32 {
    let s = sub(x, y).unwrap();
    let m = mul(&s, &s).unwrap();
    sum(m.as_primitive::<Float32Type>()).unwrap()
}

#[inline]
fn l2_arrow_arity(x: &Float32Array, y: &Float32Array) -> f32 {
    let m: Float32Array = binary(x, y, |a, b| (a - b).powi(2)).unwrap();
    sum(&m).unwrap()
}

fn bench_distance(c: &mut Criterion) {
    const DIMENSION: usize = 1024;
    const TOTAL: usize = 1024 * 1024; // 1M vectors

    let key = generate_random_array_with_seed(DIMENSION, [0; 32]);
    // 1M of 1024 D vectors. 4GB in memory.
    let target = generate_random_array_with_seed(TOTAL * DIMENSION, [42; 32]);

    c.bench_function("L2(simd)", |b| {
        b.iter(|| {
            black_box(l2_distance_batch(key.values(), target.values(), DIMENSION).count());
        })
    });

    c.bench_function("L2(arrow)", |b| {
        b.iter(|| unsafe {
            Float32Array::from_trusted_len_iter((0..target.len() / DIMENSION).map(|idx| {
                let arr = target.slice(idx * DIMENSION, DIMENSION);
                Some(l2_arrow(&key, &arr))
            }))
        });
    });

    c.bench_function("L2(arrow_artiy)", |b| {
        b.iter(|| unsafe {
            Float32Array::from_trusted_len_iter((0..target.len() / DIMENSION).map(|idx| {
                let arr = target.slice(idx * DIMENSION, DIMENSION);
                Some(l2_arrow_arity(&key, as_primitive_array(&arr)))
            }))
        });
    });

    c.bench_function("L2(auto-vectorization)", |b| {
        let x = key.values();
        b.iter(|| unsafe {
            Float32Array::from_trusted_len_iter((0..target.len() / DIMENSION).map(|idx| {
                let y = target.values()[idx * DIMENSION..(idx + 1) * DIMENSION].as_ref();
                Some(l2_scalar::<f32, 16>(x, y))
            }))
        });
    });

    let key = generate_random_array_with_seed(DIMENSION, [5; 32]);
    // 1M of 1024 D vectors. 4GB in memory.
    let target = generate_random_array_with_seed(TOTAL * DIMENSION, [7; 32]);

    c.bench_function("L2(simd) second rng seed", |b| {
        b.iter(|| {
            black_box(l2_distance_batch(key.values(), target.values(), DIMENSION).count());
        })
    });

    c.bench_function("L2(arrow) second rng seed", |b| {
        b.iter(|| unsafe {
            Float32Array::from_trusted_len_iter((0..target.len() / DIMENSION).map(|idx| {
                let arr = target.slice(idx * DIMENSION, DIMENSION);
                Some(l2_arrow(&key, &arr))
            }))
        });
    });

    c.bench_function("L2(arrow_artiy) second rng seed", |b| {
        b.iter(|| unsafe {
            Float32Array::from_trusted_len_iter((0..target.len() / DIMENSION).map(|idx| {
                let arr = target.slice(idx * DIMENSION, DIMENSION);
                Some(l2_arrow_arity(&key, as_primitive_array(&arr)))
            }))
        });
    });

    c.bench_function("L2(auto-vectorization) second rng seed", |b| {
        let x = key.values();
        b.iter(|| unsafe {
            Float32Array::from_trusted_len_iter((0..target.len() / DIMENSION).map(|idx| {
                let y = target.values()[idx * DIMENSION..(idx + 1) * DIMENSION].as_ref();
                Some(l2_scalar::<f32, 16>(x, y))
            }))
        });
    });

    let key: Float32Array = generate_random_array_with_seed(8, [5; 32]);
    // 1M of 1024 D vectors. 4GB in memory.
    let target: Float32Array = generate_random_array_with_seed(TOTAL * 8, [7; 32]);
    c.bench_function("L2(simd,f32x8)", |b| {
        b.iter(|| {
            black_box(l2_distance_batch(key.values(), target.values(), 8).count());
        })
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
