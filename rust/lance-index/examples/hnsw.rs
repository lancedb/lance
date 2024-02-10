// Copyright 2024 Lance Developers.
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

//! Run recall benchmarks for HNSW.
//!
//! run with `cargo run --release --example hnsw`

use std::collections::HashSet;

use arrow_array::types::Float32Type;
use lance_index::vector::hnsw::HNSWBuilder;
use lance_linalg::MatrixView;
use lance_testing::datagen::generate_random_array_with_seed;

fn ground_truth(mat: &MatrixView<Float32Type>, query: &[f32], k: usize) -> HashSet<u32> {
    let mut dists = vec![];
    for i in 0..mat.num_rows() {
        let dist = lance_linalg::distance::l2_distance(query, mat.row(i).unwrap());
        dists.push((dist, i as u32));
    }
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    dists.truncate(k);
    dists.into_iter().map(|(_, i)| i).collect()
}

fn main() {
    const TOTAL: usize = 65536;
    const DIMENSION: usize = 1024;
    const SEED: [u8; 32] = [42; 32];

    let data = generate_random_array_with_seed::<Float32Type>(TOTAL * DIMENSION, SEED);
    let mat = MatrixView::<Float32Type>::new(data.into(), DIMENSION);

    let q = mat.row(0).unwrap();
    let k = 10;
    let gt = ground_truth(&mat, q, k);

    for level in [4, 8, 16, 32] {
        for ef_construction in [50, 100, 200, 400] {
            let hnsw = HNSWBuilder::new(mat.clone())
                .max_level(level)
                .ef_construction(ef_construction)
                .build()
                .unwrap();
            let results: HashSet<u32> = hnsw
                .search(q, k, 100)
                .unwrap()
                .iter()
                .map(|(i, _)| *i)
                .collect();
            println!(
                "level={}, ef_construct={}, ef={} recall={}",
                level,
                ef_construction,
                100,
                results.intersection(&gt).count() as f32 / k as f32
            );
        }
    }
}
