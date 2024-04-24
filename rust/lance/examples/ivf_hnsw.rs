// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Run recall benchmarks for HNSW.
//!
//! run with `cargo run --release --example hnsw`

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::compute::concat;
use arrow::datatypes::UInt64Type;
use arrow_array::types::Float32Type;
use clap::Parser;
use futures::{StreamExt, TryStreamExt};
use itertools::Itertools;
use lance::index::vector::VectorIndexParams;
use lance::Dataset;
use lance_core::ROW_ID;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_index::{DatasetIndexExt, IndexType};
use lance_linalg::distance::{DistanceType, MetricType};
use lance_linalg::MatrixView;
use rand::{Rng, SeedableRng};

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

    #[arg(long, default_value = "20")]
    num_edges: usize,

    /// Max number of edges of each node.
    #[arg(long, default_value = "40")]
    max_edges: usize,

    #[arg(long, default_value = "7")]
    max_level: u16,

    #[arg(long, default_value = "1")]
    nprobe: usize,

    #[arg(short, default_value = "10")]
    k: usize,

    #[arg(long, default_value = "false")]
    create_index: bool,

    #[arg(long, default_value = "cosine")]
    metric_type: String,
}

fn ground_truth(
    mat: &MatrixView<Float32Type>,
    query: &[f32],
    k: usize,
    distance_type: DistanceType,
) -> HashSet<u32> {
    let mut dists = vec![];
    for i in 0..mat.num_rows() {
        let dist = distance_type.func()(query, mat.row(i).unwrap());
        dists.push((dist, i as u32));
    }
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    dists.truncate(k);
    dists.into_iter().map(|(_, i)| i).collect()
}

async fn search(args: &Args, dataset: &Dataset, column: &str, row_id: u64) -> f32 {
    let q = dataset
        .take(&[row_id], &dataset.schema().project(&[column]).unwrap())
        .await
        .unwrap()
        .column(0)
        .as_fixed_size_list()
        .values()
        .as_primitive::<Float32Type>()
        .clone();

    let columns: &[&str] = &[];
    let row_ids = dataset
        .scan()
        .project(columns)
        .unwrap()
        .with_row_id()
        .nearest(column, &q, args.k)
        .unwrap()
        .use_index(false)
        .try_into_stream()
        .await
        .unwrap()
        .map(|batch| {
            let column = batch
                .unwrap()
                .column_by_name(ROW_ID)
                .unwrap()
                .as_primitive::<UInt64Type>()
                .clone();
            let row_ids: &[u64] = column.values();
            Vec::from(row_ids)
        })
        .collect::<Vec<_>>()
        .await;
    let gt = row_ids.into_iter().flatten().collect::<HashSet<_>>();

    let mut scan = dataset.scan();
    let plan = scan
        .project(columns)
        .unwrap()
        .with_row_id()
        .nearest(column, &q, args.k)
        .unwrap()
        .nprobs(args.nprobe)
        .ef(args.ef);
    println!("{:?}", plan.explain_plan(true).await.unwrap());

    let now = std::time::Instant::now();
    let results = plan
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert!(results.len() == 1);
    let row_ids = results
        .into_iter()
        .map(|b| {
            let row_ids: &[u64] = b
                .column_by_name(ROW_ID)
                .unwrap()
                .as_primitive::<UInt64Type>()
                .values();
            Vec::from(row_ids).into_iter()
        })
        .flatten()
        .collect::<HashSet<u64>>();
    let recall = row_ids.intersection(&gt).count() as f32 / args.k as f32;
    println!(
        "level={}, nprobe={}, k={}, search={:?}, recall: {}",
        args.max_level,
        args.nprobe,
        args.k,
        now.elapsed(),
        recall,
    );

    recall
}

#[tokio::main]
async fn main() {
    env_logger::init();
    let args = Args::parse();

    let mut dataset = Dataset::open(&args.uri)
        .await
        .expect("Failed to open dataset");
    println!("Dataset schema: {:#?}", dataset.schema());

    let column = args.column.as_deref().unwrap_or("vector");
    let metric_type = MetricType::try_from(args.metric_type.as_str()).unwrap();

    let mut ivf_params = IvfBuildParams::new(16);
    ivf_params.sample_rate = 20480;
    let hnsw_params = HnswBuildParams::default()
        .ef_construction(args.ef)
        .num_edges(args.num_edges)
        .max_num_edges(args.max_edges);
    // .extend_candidates(true);
    let mut sq_params = SQBuildParams::default();
    sq_params.sample_rate = 5120;
    let params =
        VectorIndexParams::with_ivf_hnsw_sq_params(metric_type, ivf_params, hnsw_params, sq_params);
    println!("{:?}", params);

    if args.create_index {
        let now = std::time::Instant::now();
        dataset
            .create_index(&[column], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
        println!("build={:.3}s", now.elapsed().as_secs_f32());
    }

    let num_rows = dataset.count_rows(None).await.unwrap() as u64;
    println!("Loaded {} records", num_rows);

    let mut validate_row_ids = Vec::new();
    let mut rng = rand::rngs::StdRng::from_seed([13; 32]);
    while validate_row_ids.len() < 20 {
        let row_id = rng.gen::<u64>() % num_rows;
        validate_row_ids.push(row_id);
    }
    println!("validate row ids: {:?}", validate_row_ids);
    let mut avg_recall = 0.0;
    for row_id in validate_row_ids.iter() {
        avg_recall += search(&args, &dataset, column, *row_id).await;
    }
    println!("avg_recall: {}", avg_recall / validate_row_ids.len() as f32);

    // let now = std::time::Instant::now();
    // for _ in 0..10 {
    //     plan.try_into_stream()
    //         .await
    //         .unwrap()
    //         .try_collect::<Vec<_>>()
    //         .await
    //         .unwrap();
    // }
    // println!(
    //     "warm up: level={}, nprobe={}, k={}, search={:?}",
    //     args.max_level,
    //     args.nprobe,
    //     args.k,
    //     now.elapsed().div_f32(10.0),
    // );
}
