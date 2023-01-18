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

use criterion::{criterion_group, criterion_main, Criterion};

use arrow_array::FixedSizeListArray;

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
        b.iter(|| {
            l2_distance(&key, &target).unwrap();
        })
    });
}

criterion_group!(benches, bench_distance);
criterion_main!(benches);
