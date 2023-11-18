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

use std::{sync::Arc, time::Duration};

use arrow_array::types::Float32Type;
use arrow_array::{Float32Array, UInt32Array};
use criterion::{criterion_group, criterion_main, Criterion};
use lance_linalg::kernels::argmin_opt;
use lance_testing::datagen::generate_random_array_with_seed;
use num_traits::Float;

#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

#[inline]
fn argmin_arrow(x: &Float32Array) -> u32 {
    argmin_opt(x.iter()).unwrap()
}

fn argmin_arrow_batch(x: &Float32Array, dimension: usize) -> Arc<UInt32Array> {
    assert_eq!(x.len() % dimension, 0);

    let idxs = unsafe {
        UInt32Array::from_trusted_len_iter(
            (0..x.len())
                .step_by(dimension)
                .map(|start| Some(argmin_arrow(&x.slice(start, dimension)))),
        )
    };
    Arc::new(idxs)
}

fn bench_argmin(c: &mut Criterion) {
    const DIMENSION: usize = 1024 * 8;
    const TOTAL: usize = 1024;
    const SEED: [u8; 32] = [42; 32];

    let target = generate_random_array_with_seed::<Float32Type>(TOTAL * DIMENSION, SEED);

    c.bench_function("argmin(arrow)", |b| {
        b.iter(|| {
            argmin_arrow_batch(&target, DIMENSION);
        })
    });
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(32)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_argmin);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(32);
    targets = bench_argmin);

criterion_main!(benches);
