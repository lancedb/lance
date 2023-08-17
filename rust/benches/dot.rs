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

use arrow_arith::aggregate::sum;
use arrow_arith::arithmetic::{multiply, subtract};
use arrow_array::cast::as_primitive_array;
use arrow_array::types::{Float16Type, Float32Type};
use arrow_array::{Float16Array, Float32Array};
use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use lance::linalg::dot::dot;
use lance::utils::testing::generate_random_array_with_seed;

#[inline]
fn dot_arrow_artiy(x: &Float32Array, y: &Float32Array) -> f32 {
    let m = multiply(x, y).unwrap();
    sum(&m).unwrap()
}

fn bench_distance(c: &mut Criterion) {
    const DIMENSION: usize = 1024;
    const TOTAL: usize = 1024 * 1024; // 1M vectors

    let key = generate_random_array_with_seed::<Float32Type>(DIMENSION, [0; 32]);
    // 1M of 1024 D vectors. 4GB in memory.
    let target = generate_random_array_with_seed::<Float32Type>(TOTAL * DIMENSION, [42; 32]);

    let f16_key = generate_random_array_with_seed::<Float16Type>(DIMENSION, [0; 32]);
    let f16_target = generate_random_array_with_seed::<Float16Type>(DIMENSION, [42; 32]);

    c.bench_function("Dot(f32, arrow_artiy)", |b| {
        b.iter(|| unsafe {
            Float32Array::from_trusted_len_iter((0..target.len() / DIMENSION).map(|idx| {
                let arr = target.slice(idx * DIMENSION, DIMENSION);
                Some(dot_arrow_artiy(&key, as_primitive_array(&arr)))
            }))
        });
    });

    c.bench_function("Dot(f32)", |b| {
        let x = key.values();
        b.iter(|| unsafe {
            Float32Array::from_trusted_len_iter((0..target.len() / DIMENSION).map(|idx| {
                let y = target.values()[idx * DIMENSION..(idx + 1) * DIMENSION].as_ref();
                Some(dot(x, y))
            }))
        });
    });

    c.bench_function("Dot(f16)", |b| {
        let x = f16_key.values();
        b.iter(|| unsafe {
            Float16Array::from_trusted_len_iter((0..target.len() / DIMENSION).map(|idx| {
                let y = f16_target.values()[idx * DIMENSION..(idx + 1) * DIMENSION].as_ref();
                Some(dot(x, y))
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
