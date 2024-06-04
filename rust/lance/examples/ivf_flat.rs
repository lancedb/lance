// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Run recall benchmarks for HNSW.
//!
//! run with `cargo run --release --example hnsw`

use std::collections::HashMap;
use std::ops::{Range, RangeInclusive};
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::UInt8Type;
use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;
use futures::TryStreamExt;
use lance::index::vector::VectorIndexParams;
use lance::Dataset;
use lance_arrow::FixedSizeListArrayExt;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_index::{DatasetIndexExt, IndexType};
use lance_linalg::distance::{DistanceType, MetricType};
use lance_testing::datagen::generate_random_u8_array_with_range;
use tempfile::tempdir;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Dataset URI
    uri: String,

    /// Vector column name
    #[arg(short, long, value_name = "NAME", default_value = "vector")]
    column: Option<String>,

    #[arg(long, default_value = "1")]
    nprobe: usize,

    #[arg(short, default_value = "100")]
    k: usize,

    #[arg(long, default_value = "false")]
    create_index: bool,
}

#[cfg(test)]
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

async fn generate_u8_test_dataset(
    test_uri: &str,
    dim: usize,
    range: Range<u8>,
) -> (Dataset, Arc<FixedSizeListArray>) {
    let vectors = generate_random_u8_array_with_range(5_000_000 * dim, range);
    let metadata: HashMap<String, String> = vec![("test".to_string(), "ivf_pq".to_string())]
        .into_iter()
        .collect();

    let schema: Arc<_> = Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::UInt8, true)),
            dim as i32,
        ),
        true,
    )])
    .with_metadata(metadata)
    .into();
    let array = Arc::new(FixedSizeListArray::try_new_from_values(vectors, dim as i32).unwrap());
    let batch = RecordBatch::try_new(schema.clone(), vec![array.clone()]).unwrap();

    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
    let dataset = Dataset::write(batches, test_uri, None).await.unwrap();
    (dataset, array)
}

#[tokio::main]
async fn main() {
    env_logger::init();
    let args = Args::parse();

    // let (mut dataset, _) = generate_u8_test_dataset(&args.uri, 512 / 8, 0..255).await;

    let mut dataset = Dataset::open(&args.uri)
        .await
        .expect("Failed to open dataset");
    println!("Dataset schema: {:#?}", dataset.schema());

    let column = args.column.as_deref().unwrap_or("vector");
    let params = VectorIndexParams::ivf_flat(128, DistanceType::Hamming);
    println!("{:?}", params);

    if args.create_index {
        let now = std::time::Instant::now();
        dataset
            .create_index(&[column], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
        println!("build={:.3}s", now.elapsed().as_secs_f32());
    }

    println!("Loaded {} records", dataset.count_rows(None).await.unwrap());

    let q = dataset
        .take(&[0], &dataset.schema().project(&[column]).unwrap())
        .await
        .unwrap()
        .column(0)
        .as_fixed_size_list()
        .values()
        .as_primitive::<UInt8Type>()
        .clone();

    // let columns: &[&str] = &[];
    // let mut scan = dataset.scan();
    // let plan = scan
    //     .project(columns)
    //     .unwrap()
    //     .with_row_id()
    //     .nearest(column, &q, args.k)
    //     .unwrap()
    //     .nprobs(args.nprobe);
    // println!("{:?}", plan.explain_plan(true).await.unwrap());

    // let now = std::time::Instant::now();
    // plan.try_into_stream()
    //     .await
    //     .unwrap()
    //     .try_collect::<Vec<_>>()
    //     .await
    //     .unwrap();
    // println!(
    //     "nprobe={}, k={}, search={:?}",
    //     args.nprobe,
    //     args.k,
    //     now.elapsed(),
    // );

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
    //     "warm up: nprobe={}, k={}, search={:?}",
    //     args.nprobe,
    //     args.k,
    //     now.elapsed().div_f32(10.0),
    // );
}
