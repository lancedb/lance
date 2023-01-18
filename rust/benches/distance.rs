// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow_arith::aggregate::sum;
use arrow_arith::arithmetic::{multiply_dyn, subtract_dyn};
use arrow_array::cast::as_primitive_array;
use arrow_array::types::Float32Type;
use arrow_array::{Array, FixedSizeListArray, Float32Array};
use criterion::{criterion_group, criterion_main, Criterion};
use pprof::criterion::{Output, PProfProfiler};

use lance::utils::distance::{l2_distance, l2_distance_arrow};
use lance::{arrow::FixedSizeListArrayExt, utils::testing::generate_random_array};

fn bench_distance(c: &mut Criterion) {
    const DIMENSION: i32 = 1024;

    let key = generate_random_array(DIMENSION as usize);
    // 1M of 1024 D vectors. 4GB in memory.
    let values = generate_random_array(1024 * 1024 * DIMENSION as usize);
    let target = FixedSizeListArray::try_new(values, DIMENSION).unwrap();

    c.bench_function("L2 distance", |b| {
        b.iter(|| {
            l2_distance(&key, &target).unwrap();
        })
    });

    c.bench_function("L2_distance_arrow", |b| {
        b.iter(|| unsafe {
            Float32Array::from_trusted_len_iter(
                (0..target.len())
                    .map(|idx| {
                        let left = target.value(idx);
                        let arr = left.as_any().downcast_ref::<Float32Array>().unwrap();
                        l2_distance_arrow(&key, arr)
                    })
                    .map(|d| Some(d)),
            )
        });
    });

    c.bench_function("L2_distance_arrow_arith", |b| {
        b.iter(|| unsafe {
            Float32Array::from_trusted_len_iter(
                (0..target.len())
                    .map(|idx| {
                        let left = target.value(idx);
                        let sub = subtract_dyn(left.as_ref(), &key).unwrap();
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
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_distance);
criterion_main!(benches);
