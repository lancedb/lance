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
use std::sync::Arc;

use arrow::array::AsArray;
use arrow_array::types::Float32Type;
use arrow_select::concat::concat;
use clap::Parser;
use futures::StreamExt;
use lance::Dataset;
use lance_index::vector::{graph::memory::InMemoryVectorStorage, hnsw::HNSWBuilder};
use lance_linalg::{distance::MetricType, MatrixView};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Dataset URI
    uri: String,

    /// Vector column name
    #[arg(short, long, value_name = "NAME", default_value = "vector")]
    column: Option<String>,

    #[arg(long, default_value = "100")]
    ef: usize,

    /// Max number of edges of each node.
    #[arg(long, default_value = "30")]
    max_edges: usize,

    #[arg(long, default_value = "7")]
    max_level: u16,
}

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

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let dataset = Dataset::open(&args.uri)
        .await
        .expect("Failed to open dataset");
    println!("Dataset schema: {:#?}", dataset.schema());
    let column = args.column.as_deref().unwrap_or("vector");
    let batches = dataset
        .scan()
        .project(&[column])
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .then(|batch| async move { batch.unwrap().column_by_name(column).unwrap().clone() })
        .collect::<Vec<_>>()
        .await;
    let arrs = batches.iter().map(|b| b.as_ref()).collect::<Vec<_>>();
    let fsl = concat(&arrs).unwrap();
    let mat = Arc::new(MatrixView::<Float32Type>::try_from(fsl.as_fixed_size_list()).unwrap());
    println!("Loaded {:?} batches", mat.num_rows());

    let vector_store = Arc::new(InMemoryVectorStorage::new(mat.clone(), MetricType::L2));

    let q = mat.row(0).unwrap();
    let k = 10;
    let gt = ground_truth(&mat, q, k);

    for ef_construction in [15, 30, 50] {
        let now = std::time::Instant::now();
        let hnsw = HNSWBuilder::new(vector_store.clone())
            .max_level(args.max_level)
            .num_edges(15)
            .max_num_edges(args.max_edges)
            .ef_construction(ef_construction)
            .build()
            .unwrap();
        let construct_time = now.elapsed().as_secs_f32();
        let now = std::time::Instant::now();
        let results: HashSet<u32> = hnsw
            .search(q, k, args.ef, None)
            .unwrap()
            .iter()
            .map(|(i, _)| *i)
            .collect();
        let search_time = now.elapsed().as_micros();
        println!(
            "level={}, ef_construct={}, ef={} recall={}: construct={:.3}s search={:.3} us",
            args.max_level,
            ef_construction,
            args.ef,
            results.intersection(&gt).count() as f32 / k as f32,
            construct_time,
            search_time
        );
    }
}
