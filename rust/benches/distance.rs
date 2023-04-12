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

use std::iter::Sum;

use arrow_arith::aggregate::sum;
use arrow_arith::arithmetic::{multiply_dyn, subtract_dyn};
use arrow_array::cast::as_primitive_array;
use arrow_array::types::Float32Type;
use arrow_array::{Array, Float32Array};
use criterion::{criterion_group, criterion_main, Criterion};
use num_traits::real::Real;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use lance::linalg::l2::*;
use lance::utils::testing::generate_random_array;

#[inline]
fn l2_scalar<T: Real + Sum>(from: &[T], to: &[T]) -> T {
    from.iter()
        .zip(to.iter())
        .map(|(a, b)| (a.sub(*b).powi(2)))
        .sum::<T>()
        .sqrt()
}

fn bench_distance(c: &mut Criterion) {
    const DIMENSION: usize = 1024;

    let key = generate_random_array(DIMENSION as usize);
    // 1M of 1024 D vectors. 4GB in memory.
    let target = generate_random_array(1024 * 1024 * DIMENSION as usize);

    c.bench_function("L2 distance", |b| b.iter(|| key.l2(&target)));

    c.bench_function("L2 distance(auto-vectorization)", |b| {
        b.iter(|| l2_scalar(key.values(), target.values()))
    });

    c.bench_function("L2 distance (loop)", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for i in 0..key.len() {
                let diff = key.value(i) - target.value(i);
                sum += diff * diff;
            }
            sum.sqrt()
        })
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

criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
        // .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_distance);
criterion_main!(benches);
