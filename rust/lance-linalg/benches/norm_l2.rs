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
use arrow_array::types::{Float16Type, Float64Type};
use arrow_array::{
    cast::AsArray, types::Float32Type, ArrowNumericType, Float32Array, NativeAdapter,
    PrimitiveArray,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lance_arrow::FloatToArrayType;
use num_traits::FromPrimitive;

#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use lance_linalg::distance::norm_l2::{norm_l2, Normalize};
use lance_testing::datagen::generate_random_array_with_seed;

#[inline]
fn norm_l2_arrow<T: ArrowNumericType>(x: &PrimitiveArray<T>) -> T::Native {
    let m = mul(&x, &x).unwrap();
    sum(m.as_primitive::<T>()).unwrap()
}

const DIMENSION: usize = 1024;
const TOTAL: usize = 1024 * 1024; // 1M vectors

fn run_bench<T: ArrowNumericType>(c: &mut Criterion)
where
    T::Native: FromPrimitive + FloatToArrayType,
    NativeAdapter<T>: From<T::Native>,
    for<'a> &'a [T::Native]: Normalize<T::Native>,
{
    // 1M of 1024 D vectors
    let target: PrimitiveArray<T> = generate_random_array_with_seed(TOTAL * DIMENSION, [42; 32]);

    let type_name = std::any::type_name::<T::Native>();

    c.bench_function(format!("NormL2({type_name}, arrow)").as_str(), |b| {
        b.iter(|| unsafe {
            PrimitiveArray::<T>::from_trusted_len_iter((0..target.len() / DIMENSION).map(|idx| {
                let arr = target.slice(idx * DIMENSION, DIMENSION);
                Some(norm_l2_arrow(&arr))
            }))
        });
    });

    c.bench_function(
        format!("NormL2({type_name}, auto-vectorization)").as_str(),
        |b| {
            b.iter(|| {
                black_box(
                    target
                        .values()
                        .chunks_exact(DIMENSION)
                        .map(|arr| norm_l2(arr))
                        .collect::<Vec<_>>(),
                );
            });
        },
    );
}

fn bench_distance(c: &mut Criterion) {
    run_bench::<Float16Type>(c);
    run_bench::<Float32Type>(c);

    // 1M of 1024 D vectors. 4GB in memory.
    let target: Float32Array = generate_random_array_with_seed(TOTAL * DIMENSION, [42; 32]);
    c.bench_function("norm_l2(f32, SIMD)", |b| {
        b.iter(|| {
            target.values().chunks_exact(DIMENSION).for_each(|arr| {
                let _ = arr.norm_l2();
            })
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
