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
use std::ops::AddAssign;

use arrow_arith::{aggregate::sum, numeric::mul};
use arrow_array::{
    cast::AsArray,
    types::{Float16Type, Float32Type, Float64Type},
    ArrowNumericType, Float16Array, Float32Array, NativeAdapter, PrimitiveArray,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_traits::{real::Real, FromPrimitive};

#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use lance_linalg::distance::dot::{dot, dot_distance, Dot};
use lance_testing::datagen::generate_random_array_with_seed;

#[inline]
fn dot_arrow_artiy<T: ArrowNumericType>(x: &PrimitiveArray<T>, y: &PrimitiveArray<T>) -> T::Native {
    let m = mul(x, y).unwrap();
    sum(m.as_primitive::<T>()).unwrap()
}

fn run_bench<T: ArrowNumericType>(c: &mut Criterion)
where
    T::Native: Real + FromPrimitive + Sum + AddAssign,
    NativeAdapter<T>: From<T::Native>,
{
    const DIMENSION: usize = 1024;
    const TOTAL: usize = 1024 * 1024; // 1M vectors

    let key: PrimitiveArray<T> = generate_random_array_with_seed(DIMENSION, [0; 32]);
    // 1M of 1024 D vectors
    let target: PrimitiveArray<T> = generate_random_array_with_seed(TOTAL * DIMENSION, [42; 32]);

    let type_name = std::any::type_name::<T::Native>();

    c.bench_function(format!("Dot({type_name}, arrow_artiy)").as_str(), |b| {
        b.iter(|| unsafe {
            PrimitiveArray::<T>::from_trusted_len_iter((0..target.len() / DIMENSION).map(|idx| {
                let arr = target.slice(idx * DIMENSION, DIMENSION);
                Some(dot_arrow_artiy(&key, &arr))
            }));
        });
    });

    c.bench_function(
        format!("Dot({type_name}, auto-vectorization)").as_str(),
        |b| {
            let x = key.values();
            b.iter(|| unsafe {
                PrimitiveArray::<T>::from_trusted_len_iter((0..target.len() / 1024).map(|idx| {
                    let y = target.values()[idx * DIMENSION..(idx + 1) * DIMENSION].as_ref();
                    Some(black_box(dot(x, y)))
                }));
            });
        },
    );

    // TODO: SIMD needs generic specialization
}

fn bench_distance(c: &mut Criterion) {
    const DIMENSION: usize = 1024;
    const TOTAL: usize = 1024 * 1024; // 1M vectors

    run_bench::<Float16Type>(c);
    c.bench_function("Dot(f16, SIMD)", |b| {
        let key: Float16Array = generate_random_array_with_seed(DIMENSION, [0; 32]);
        // 1M of 1024 D vectors
        let target: Float16Array = generate_random_array_with_seed(TOTAL * DIMENSION, [42; 32]);
        b.iter(|| unsafe {
            let x = key.values().as_ref();
            Float16Array::from_trusted_len_iter((0..target.len() / DIMENSION).map(|idx| {
                let y = target.values()[idx * DIMENSION..(idx + 1) * DIMENSION].as_ref();
                Some(dot_distance(x, y))
            }))
        });
    });

    run_bench::<Float32Type>(c);
    c.bench_function("Dot(f32, SIMD)", |b| {
        let key: Float32Array = generate_random_array_with_seed(DIMENSION, [0; 32]);
        // 1M of 1024 D vectors
        let target: Float32Array = generate_random_array_with_seed(TOTAL * DIMENSION, [42; 32]);
        b.iter(|| unsafe {
            let x = key.values().as_ref();
            Float32Array::from_trusted_len_iter((0..target.len() / DIMENSION).map(|idx| {
                let y = target.values()[idx * DIMENSION..(idx + 1) * DIMENSION].as_ref();
                Some(Float32Type::dot(x, y))
            }))
        });
    });

    run_bench::<Float64Type>(c);
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
