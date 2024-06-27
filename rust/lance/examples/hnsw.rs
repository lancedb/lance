// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Run recall benchmarks for HNSW.
//!
//! run with `cargo run --release --example hnsw`

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow_array::{types::Float32Type, Array, FixedSizeListArray};
use arrow_select::concat::concat;
use clap::Parser;
use futures::StreamExt;
use lance::Dataset;
use lance_index::vector::v3::subindex::IvfSubIndex;
use lance_index::vector::{
    flat::storage::FlatStorage,
    hnsw::{builder::HnswBuildParams, HNSW},
};
use lance_linalg::distance::DistanceType;

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

fn ground_truth(fsl: &FixedSizeListArray, query: &[f32], k: usize) -> HashSet<u32> {
    let mut dists = vec![];
    for i in 0..fsl.len() {
        let dist = lance_linalg::distance::l2_distance(
            query,
            fsl.value(i).as_primitive::<Float32Type>().values(),
        );
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
    let fsl = concat(&arrs).unwrap().as_fixed_size_list().clone();
    println!("Loaded {:?} batches", fsl.len());

    let vector_store = Arc::new(FlatStorage::new(fsl.clone(), DistanceType::L2));

    let q = fsl.value(0);
    let k = 10;
    let gt = ground_truth(&fsl, q.as_primitive::<Float32Type>().values(), k);

    for ef_construction in [15, 30, 50] {
        let now = std::time::Instant::now();
        let hnsw = HNSW::index_vectors(
            vector_store.as_ref(),
            HnswBuildParams::default()
                .max_level(args.max_level)
                .num_edges(15)
                .ef_construction(ef_construction),
        )
        .unwrap();
        let construct_time = now.elapsed().as_secs_f32();
        let now = std::time::Instant::now();
        let results: HashSet<u32> = hnsw
            .search_basic(q.clone(), k, args.ef, None, vector_store.as_ref())
            .unwrap()
            .iter()
            .map(|node| node.id)
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
