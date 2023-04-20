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
use arrow_arith::arithmetic::{multiply_dyn, subtract_dyn};
use arrow_arith::arity::binary;
use arrow_array::cast::as_primitive_array;
use arrow_array::types::Float32Type;
use arrow_array::{Array, Float32Array};
use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use lance::linalg::l2::l2_distance_batch;
use lance::utils::testing::generate_random_array;

#[inline]
fn l2_arrow(x: &Float32Array, y: &Float32Array) -> f32 {
    0.0
}

#[inline]
fn l2_arrow_arity(x: &Float32Array, y: &Float32Array) -> f32 {
    0.0
}

fn bench_distance(c: &mut Criterion) {
    const DIMENSION: usize = 1024;
    const TOTAL: usize = 1024 * 1024; // 1M vectors

    let key = generate_random_array(DIMENSION);
    // 1M of 1024 D vectors. 4GB in memory.
    let target = generate_random_array(TOTAL * DIMENSION);

    c.bench_function("L2 distance", |b| {
        b.iter(|| {
            l2_distance_batch(key.values(), target.values(), DIMENSION);
        })
    });

    c.bench_function("L2_distance_arrow", |b| {
        b.iter(|| unsafe {
            Float32Array::from_trusted_len_iter(
                (0..target.len() / DIMENSION)
                    .map(|idx| {
                        let arr = target.slice(idx * DIMENSION, DIMENSION);
                        l2_arrow(&key, as_primitive_array(arr.as_ref()))
                    })
                    .map(|d| Some(d)),
            )
        });
    });

    c.bench_function("L2_distance_arrow_arith", |b| {
        b.iter(|| unsafe {
            Float32Array::from_trusted_len_iter(
                (0..target.len() / DIMENSION)
                    .map(|idx| {
                        let arr = target.slice(idx * DIMENSION, DIMENSION);
                        let sub = subtract_dyn(arr.as_ref(), &key).unwrap();
                        let mul = multiply_dyn(&sub, &sub).unwrap();
                        sum(as_primitive_array::<Float32Type>(&mul)).unwrap_or(0.0)
                    })
                    .map(|d| Some(d)),
            )
        });
    });
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_distance);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_distance);
criterion_main!(benches);
