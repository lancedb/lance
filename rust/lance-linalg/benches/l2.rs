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

use arrow_arith::{aggregate::sum, arity::binary};
use arrow_array::{
    types::{Float16Type, Float32Type, Float64Type},
    ArrowNumericType, Float32Array, NativeAdapter, PrimitiveArray,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use num_traits::{Float, FromPrimitive};
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use lance_arrow::FloatToArrayType;
use lance_linalg::distance::{l2::l2, l2_distance_batch, L2};
use lance_testing::datagen::generate_random_array_with_seed;

const TOTAL: usize = 1024 * 1024; // 1M vectors

#[inline]
fn l2_arrow_arity<T: ArrowNumericType>(x: &PrimitiveArray<T>, y: &PrimitiveArray<T>) -> T::Native
where
    T::Native: Float,
{
    let m: PrimitiveArray<T> = binary(x, y, |a, b| (a - b).powi(2)).unwrap();
    sum(&m).unwrap()
}

fn run_bench<T: ArrowNumericType>(c: &mut Criterion)
where
    T::Native: FromPrimitive + FloatToArrayType,
    NativeAdapter<T>: From<T::Native>,
    <T::Native as FloatToArrayType>::ArrowType: L2,
{
    const DIMENSION: usize = 1024;
    const TOTAL: usize = 1024 * 1024; // 1M vectors

    let key: PrimitiveArray<T> = generate_random_array_with_seed(DIMENSION, [0; 32]);
    // 1M of 1024 D vectors
    let target: PrimitiveArray<T> = generate_random_array_with_seed(TOTAL * DIMENSION, [42; 32]);

    let type_name = std::any::type_name::<T::Native>();

    c.bench_function(format!("L2({type_name}, arrow_artiy)").as_str(), |b| {
        b.iter(|| unsafe {
            PrimitiveArray::<T>::from_trusted_len_iter((0..target.len() / DIMENSION).map(|idx| {
                let arr = target.slice(idx * DIMENSION, DIMENSION);
                Some(l2_arrow_arity(&key, &arr))
            }));
        });
    });

    c.bench_function(
        format!("L2({type_name}, auto-vectorization)").as_str(),
        |b| {
            let x = &key.values()[..];
            b.iter(|| unsafe {
                PrimitiveArray::<T>::from_trusted_len_iter((0..target.len() / 1024).map(|idx| {
                    let y = target.values()[idx * DIMENSION..(idx + 1) * DIMENSION].as_ref();
                    Some(black_box(l2(x, y)))
                }));
            });
        },
    );
}

fn bench_distance(c: &mut Criterion) {
    const DIMENSION: usize = 1024;

    run_bench::<Float32Type>(c);

    let key: Float32Array = generate_random_array_with_seed(DIMENSION, [0; 32]);
    let target: Float32Array = generate_random_array_with_seed(TOTAL * DIMENSION, [42; 32]);
    c.bench_function("L2(f32, simd)", |b| {
        b.iter(|| {
            black_box(l2_distance_batch(key.values(), target.values(), DIMENSION).count());
        })
    });

    run_bench::<Float16Type>(c);
    run_bench::<Float64Type>(c);
}

fn bench_small_distance(c: &mut Criterion) {
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
    targets = bench_distance, bench_small_distance);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_distance, bench_small_distance);
criterion_main!(benches);
