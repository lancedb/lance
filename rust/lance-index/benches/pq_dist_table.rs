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

//! Benchmark of building PQ distance table.

use std::iter::repeat;
use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::UInt8Array;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lance_index::vector::pq::{ProductQuantizer, ProductQuantizerImpl};
use lance_linalg::distance::MetricType;
use lance_testing::datagen::generate_random_array_with_seed;
use rand::{prelude::StdRng, Rng, SeedableRng};

#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

const PQ: usize = 96;
const DIM: usize = 1536;
const TOTAL: usize = 5 * 1024 * 1024;

fn dist_table(c: &mut Criterion) {
    let codebook = Arc::new(generate_random_array_with_seed::<Float32Type>(
        256 * DIM,
        [88; 32],
    ));
    let query = generate_random_array_with_seed::<Float32Type>(DIM, [32; 32]);

    let mut rnd = StdRng::from_seed([32; 32]);
    let code = UInt8Array::from_iter_values(repeat(rnd.gen::<u8>()).take(TOTAL * PQ));

    let l2_pq =
        ProductQuantizerImpl::<Float32Type>::new(PQ, 8, DIM, codebook.clone(), MetricType::L2);

    c.bench_function(
        format!("{},L2,PQ={},DIM={}", TOTAL, PQ, DIM).as_str(),
        |b| {
            b.iter(|| {
                black_box(l2_pq.build_distance_table(&query, &code).unwrap().len());
            })
        },
    );

    let cosine_pq =
        ProductQuantizerImpl::<Float32Type>::new(PQ, 8, DIM, codebook.clone(), MetricType::Cosine);

    c.bench_function(
        format!("{},Cosine,PQ={},DIM={}", TOTAL, PQ, DIM).as_str(),
        |b| {
            b.iter(|| {
                black_box(cosine_pq.build_distance_table(&query, &code).unwrap());
            })
        },
    );
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = dist_table);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = dist_table);

criterion_main!(benches);
