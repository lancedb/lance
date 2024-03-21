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
use arrow_array::PrimitiveArray;
use arrow_select::concat::concat;
use clap::Parser;
use futures::{StreamExt, TryStreamExt};
use lance::index::vector::VectorIndexParams;
use lance::Dataset;
use lance_index::vector::hnsw::{builder::HnswBuildParams, HNSWBuilder};
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::{DatasetIndexExt, IndexType};
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

    #[arg(long, default_value = "true")]
    replace: bool,

    #[arg(long, default_value = "1")]
    nprobe: usize,

    #[arg(short, default_value = "10")]
    k: usize,

    #[arg(long, default_value = "false")]
    create_index: bool,
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
    env_logger::init();
    let args = Args::parse();

    let dataset = Dataset::open(&args.uri)
        .await
        .expect("Failed to open dataset");
    println!("Dataset schema: {:#?}", dataset.schema());

    let column = args.column.as_deref().unwrap_or("vector");

    let mut ivf_params = IvfBuildParams::new(128);
    ivf_params.sample_rate = 20480;
    let pq_params = PQBuildParams::default();
    let hnsw_params = HnswBuildParams::default()
        .ef_construction(100)
        .num_edges(15)
        .max_num_edges(30);
    let params = VectorIndexParams::with_ivf_hnsw_pq_params(
        MetricType::Cosine,
        ivf_params,
        hnsw_params,
        pq_params,
    );
    println!("{:?}", params);

    if args.create_index {
        let now = std::time::Instant::now();
        dataset
            .create_index(&[column], IndexType::Vector, None, &params, args.replace)
            .await
            .unwrap();
        let build_time = now.elapsed().as_secs_f32();
        println!("build={:.3}s", build_time);
    }

    println!("Loaded {} batches", dataset.count_rows().await.unwrap());

    let q = dataset
        .take(&[0], &dataset.schema().project(&[column]).unwrap())
        .await
        .unwrap()
        .column(0)
        .as_fixed_size_list()
        .values()
        .as_primitive::<Float32Type>()
        .clone();
    let now = std::time::Instant::now();
    let results = dataset
        .scan()
        .nearest(column, &q, args.k)
        .unwrap()
        .nprobs(args.nprobe)
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let search_time = now.elapsed().as_micros();
    println!(
        "level={}, search={:.3} us",
        args.max_level,
        // hnsw_params,
        // args.ef,
        // results.intersection(&gt).count() as f32 / k as f32,
        // build_time,
        search_time
    );

    let now = std::time::Instant::now();
    let results = dataset
        .scan()
        .nearest(column, &q, args.k)
        .unwrap()
        .nprobs(args.nprobe)
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let search_time = now.elapsed().as_micros();
    println!(
        "warm up: level={}, search={:.3} us",
        args.max_level,
        // hnsw_params,
        // args.ef,
        // results.intersection(&gt).count() as f32 / k as f32,
        // build_time,
        search_time
    );
}
