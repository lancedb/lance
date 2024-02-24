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
use clap::Parser;
use lance::Dataset;
use lance_index::vector::{
    graph::memory::InMemoryVectorStorage,
    hnsw::HNSWBuilder,
    pq::{storage::ProductQuantizationStorage, PQBuildParams},
};
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
    #[arg(long, default_value = "64")]
    max_edges: usize,

    /// Number of PQ sub-vectors
    #[arg(short, long, default_value = "96")]
    pq: Option<usize>,

    /// Metric type
    #[arg(short, long, default_value = "l2")]
    metric_type: String,
}

fn ground_truth(mat: &MatrixView<Float32Type>, query: &[f32], k: usize) -> HashSet<u32> {
    let mut dists = vec![];
    for i in 0..mat.num_rows() {
        let dist = lance_linalg::distance::cosine_distance(query, mat.row(i).unwrap());
        dists.push((dist, i as u32));
    }
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    dists.truncate(k);
    dists.into_iter().map(|(_, i)| i).collect()
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let mt = MetricType::try_from(args.metric_type.as_str()).expect("Unknown metric type");
    let dataset = Dataset::open(&args.uri)
        .await
        .expect("Failed to open dataset");
    let column = args.column.as_deref().unwrap_or("vector");
    println!("Dataset schema: {:#?}", dataset.schema());
    let batch = dataset
        .scan()
        .with_row_id()
        .project(&[column])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let fsl = batch.column_by_name(column).unwrap();
    let mat = Arc::new(MatrixView::<Float32Type>::try_from(fsl.as_fixed_size_list()).unwrap());
    println!("Loaded {:?} vectors", mat.num_rows());

    let vector_store = Arc::new(InMemoryVectorStorage::new(mat.clone(), mt));

    let q = mat.row(0).unwrap();
    let k = 10;
    let now = std::time::Instant::now();
    let gt = ground_truth(&mat, q, k);
    println!("Build GT: {} seconds", now.elapsed().as_secs_f32());

    let num_sub_vectors = args.pq.unwrap_or(mat.num_columns() / 8);
    let pq_param = PQBuildParams::new(num_sub_vectors, 8);
    let sampled_data = mat.sample(256 * 256);
    let now = std::time::Instant::now();
    let pq = pq_param.build_from_matrix(&sampled_data, mt).await.unwrap();
    println!("Build PQ: {} s", now.elapsed().as_secs_f32());

    let pq_storage = Arc::new(
        ProductQuantizationStorage::build(pq, &batch, column)
            .await
            .unwrap(),
    );

    let level = 8;

    for ef_construction in [50, 100, 200, 400] {
        let now = std::time::Instant::now();
        let hnsw = HNSWBuilder::new(vector_store.clone())
            .max_level(level)
            .max_num_edges(args.max_edges)
            .ef_construction(ef_construction)
            // .build_with(pq_storage.clone())
            .build()
            .unwrap();
        let construct_time = now.elapsed().as_secs_f32();
        let now = std::time::Instant::now();
        let results: HashSet<u32> = hnsw
            .search(q, k, args.ef)
            .unwrap()
            .iter()
            .map(|(i, _)| *i)
            .collect();
        let search_time = now.elapsed().as_micros();
        println!(
            "level={}, ef_construct={}, ef={} recall={}: construct={:.3}s search={:.3} us",
            level,
            ef_construction,
            args.ef,
            results.intersection(&gt).count() as f32 / k as f32,
            construct_time,
            search_time
        );
    }
}
