// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
//! Benchmark a local distributed `IVF_PQ` build end to end.
//!
//! This benchmark:
//! - creates a local dataset with multiple fragments,
//! - trains shared IVF/PQ parameters once,
//! - builds shard-local partial indices concurrently,
//! - finalizes the distributed merge, and
//! - commits the merged index.
//!
//! Run it with:
//!
//! ```bash
//! cargo run --manifest-path benchmarks/distributed_ivf_pq/Cargo.toml --release -- \
//!   --fragments 8 \
//!   --rows-per-fragment 262144 \
//!   --shards 8 \
//!   --dim 128 \
//!   --num-partitions 32768 \
//!   --num-sub-vectors 16 \
//!   --max-iters 10 \
//!   --sample-rate 8 \
//!   --cleanup
//! ```
//!
//! The benchmark prints timing information as JSON so it can be used in
//! scripts or copied into benchmark notes.
#![allow(clippy::print_stdout)]

use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::types::Float32Type;
use arrow::array::{
    ArrayRef, AsArray, FixedSizeListArray, FixedSizeListBuilder, Float32Builder, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use clap::Parser;
use lance::Dataset;
use lance::dataset::{WriteMode, WriteParams};
use lance::index::vector::VectorIndexParams;
use lance::index::vector::ivf::finalize_distributed_merge;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::kmeans::{KMeansParams, train_kmeans};
use lance_index::vector::pq::PQBuildParams;
use lance_index::{DatasetIndexExt, IndexType};
use lance_linalg::distance::DistanceType;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use tokio::task::JoinSet;
use uuid::Uuid;

type DynError = Box<dyn Error + Send + Sync + 'static>;
type DynResult<T> = Result<T, DynError>;

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Benchmark local distributed IVF_PQ build with shard merge"
)]
struct Args {
    /// Dataset URI. If omitted, a temporary local directory under /tmp is used.
    #[arg(long)]
    uri: Option<String>,

    /// Optional JSON output path.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Number of fragments to generate.
    #[arg(long, default_value_t = 16)]
    fragments: usize,

    /// Number of rows per fragment.
    #[arg(long, default_value_t = 4_096)]
    rows_per_fragment: usize,

    /// Number of shards to split fragments into.
    #[arg(long, default_value_t = 4)]
    shards: usize,

    /// Vector dimension.
    #[arg(long, default_value_t = 128)]
    dim: i32,

    /// Number of IVF partitions.
    #[arg(long, default_value_t = 64)]
    num_partitions: usize,

    /// Number of PQ sub-vectors.
    #[arg(long, default_value_t = 16)]
    num_sub_vectors: usize,

    /// Number of bits per PQ code.
    #[arg(long, default_value_t = 8)]
    num_bits: usize,

    /// KMeans / PQ training max iterations.
    #[arg(long, default_value_t = 20)]
    max_iters: usize,

    /// Sample rate for IVF/PQ training.
    #[arg(long, default_value_t = 256)]
    sample_rate: usize,

    /// Index name to commit.
    #[arg(long, default_value = "dist_bench_idx")]
    index_name: String,

    /// Vector column name.
    #[arg(long, default_value = "vector")]
    vector_column: String,

    /// Delete generated dataset after the benchmark completes.
    #[arg(long, default_value_t = false)]
    cleanup: bool,
}

#[derive(Debug, Serialize)]
struct ShardTiming {
    shard_id: usize,
    fragment_ids: Vec<u32>,
    build_ms: u128,
}

#[derive(Debug, Serialize)]
struct BenchResult {
    dataset_uri: String,
    index_name: String,
    vector_column: String,
    shared_uuid: String,
    rows: usize,
    fragments: usize,
    shards: usize,
    rows_per_fragment: usize,
    dim: i32,
    num_partitions: usize,
    num_sub_vectors: usize,
    num_bits: usize,
    dataset_prepare_ms: u128,
    train_shared_params_ms: u128,
    shard_build_ms: u128,
    finalize_ms: u128,
    commit_ms: u128,
    total_index_build_ms: u128,
    total_pipeline_ms: u128,
    shard_timings: Vec<ShardTiming>,
}

fn default_dataset_uri() -> String {
    let path = std::env::temp_dir().join(format!("lance-dist-bench-{}", Uuid::new_v4()));
    path.to_string_lossy().into_owned()
}

fn validate_args(args: &Args) -> DynResult<()> {
    if args.fragments == 0 {
        return Err("fragments must be positive".into());
    }
    if args.rows_per_fragment == 0 {
        return Err("rows_per_fragment must be positive".into());
    }
    if args.shards == 0 {
        return Err("shards must be positive".into());
    }
    if args.fragments < args.shards {
        return Err(format!(
            "fragments {} must be >= shards {}",
            args.fragments, args.shards
        )
        .into());
    }
    if args.dim <= 0 {
        return Err(format!("dim must be positive, got {}", args.dim).into());
    }
    if args.num_sub_vectors == 0 {
        return Err("num_sub_vectors must be positive".into());
    }
    if args.dim as usize % args.num_sub_vectors != 0 {
        return Err(format!(
            "dim {} must be divisible by num_sub_vectors {}",
            args.dim, args.num_sub_vectors
        )
        .into());
    }
    Ok(())
}

fn create_batch(
    start_row_id: u64,
    num_rows: usize,
    dim: i32,
    vector_column: &str,
    seed: u64,
) -> DynResult<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new(
            vector_column,
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
            false,
        ),
    ]));

    let ids = Arc::new(UInt64Array::from_iter_values(
        (0..num_rows).map(|offset| start_row_id + offset as u64),
    )) as ArrayRef;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut list_builder = FixedSizeListBuilder::new(Float32Builder::new(), dim);
    for _ in 0..num_rows {
        for _ in 0..dim {
            list_builder.values().append_value(rng.random::<f32>());
        }
        list_builder.append(true);
    }
    let vectors = Arc::new(list_builder.finish()) as ArrayRef;

    Ok(RecordBatch::try_new(schema, vec![ids, vectors])?)
}

async fn create_fragmented_dataset(args: &Args, uri: &str) -> DynResult<Dataset> {
    let write_params = WriteParams {
        max_rows_per_file: args.rows_per_fragment,
        mode: WriteMode::Create,
        ..Default::default()
    };

    let first_batch = create_batch(0, args.rows_per_fragment, args.dim, &args.vector_column, 0)?;
    let first_schema = first_batch.schema();
    let first_reader = RecordBatchIterator::new(vec![Ok(first_batch)].into_iter(), first_schema);
    let mut dataset = Dataset::write(first_reader, uri, Some(write_params.clone())).await?;

    for fragment_id in 1..args.fragments {
        let batch = create_batch(
            (fragment_id * args.rows_per_fragment) as u64,
            args.rows_per_fragment,
            args.dim,
            &args.vector_column,
            fragment_id as u64,
        )?;
        let schema = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
        let append_params = WriteParams {
            mode: WriteMode::Append,
            max_rows_per_file: args.rows_per_fragment,
            ..Default::default()
        };
        dataset.append(reader, Some(append_params)).await?;
    }

    Ok(dataset)
}

async fn prepare_global_ivf_pq(
    dataset: &Dataset,
    vector_column: &str,
    num_partitions: usize,
    num_sub_vectors: usize,
    num_bits: usize,
    max_iters: usize,
    sample_rate: usize,
) -> DynResult<(IvfBuildParams, PQBuildParams)> {
    let batch = dataset
        .scan()
        .project(&[vector_column.to_string()])?
        .try_into_batch()
        .await?;
    let vectors = batch
        .column_by_name(vector_column)
        .ok_or_else(|| format!("vector column '{vector_column}' does not exist"))?
        .as_fixed_size_list();

    let dim = vectors.value_length() as usize;
    let values = vectors.values().as_primitive::<Float32Type>();
    let kmeans_params = KMeansParams::new(
        None,
        max_iters
            .try_into()
            .map_err(|_| format!("max_iters {} does not fit into u32", max_iters))?,
        1,
        DistanceType::L2,
    );
    let kmeans =
        train_kmeans::<Float32Type>(values, kmeans_params, dim, num_partitions, sample_rate)?;

    let centroids_flat = kmeans.centroids.as_primitive::<Float32Type>().clone();
    let centroids = Arc::new(FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
        Arc::new(centroids_flat),
        None,
    )?);
    let mut ivf_params = IvfBuildParams::try_with_centroids(num_partitions, centroids)?;
    ivf_params.max_iters = max_iters;
    ivf_params.sample_rate = sample_rate;

    let mut pq_train_params = PQBuildParams::new(num_sub_vectors, num_bits);
    pq_train_params.max_iters = max_iters;
    pq_train_params.sample_rate = sample_rate;

    let pq = pq_train_params.build(vectors, DistanceType::L2)?;
    let codebook = Arc::new(pq.codebook.values().as_primitive::<Float32Type>().clone()) as ArrayRef;
    let mut pq_params = PQBuildParams::with_codebook(num_sub_vectors, num_bits, codebook);
    pq_params.max_iters = max_iters;
    pq_params.sample_rate = sample_rate;

    Ok((ivf_params, pq_params))
}

fn split_fragment_groups(fragment_ids: &[u32], num_shards: usize) -> Vec<Vec<u32>> {
    let group_size = fragment_ids.len().div_ceil(num_shards);
    fragment_ids
        .chunks(group_size.max(1))
        .map(|group| group.to_vec())
        .collect()
}

async fn build_shards(
    uri: &str,
    fragment_groups: Vec<Vec<u32>>,
    vector_column: &str,
    index_name: &str,
    shared_uuid: &str,
    params: VectorIndexParams,
) -> DynResult<Vec<ShardTiming>> {
    let mut join_set = JoinSet::new();

    for (shard_id, fragment_ids) in fragment_groups.into_iter().enumerate() {
        let uri = uri.to_string();
        let vector_column = vector_column.to_string();
        let index_name = index_name.to_string();
        let shared_uuid = shared_uuid.to_string();
        let params = params.clone();
        join_set.spawn(async move {
            let mut dataset = Dataset::open(&uri).await?;
            let start = Instant::now();
            let columns = [vector_column.as_str()];
            let mut builder = dataset.create_index_builder(&columns, IndexType::Vector, &params);
            builder = builder
                .name(index_name)
                .fragments(fragment_ids.clone())
                .index_uuid(shared_uuid);
            builder.execute_uncommitted().await?;
            Ok::<_, DynError>(ShardTiming {
                shard_id,
                fragment_ids,
                build_ms: start.elapsed().as_millis(),
            })
        });
    }

    let mut shard_timings = Vec::new();
    while let Some(result) = join_set.join_next().await {
        shard_timings.push(result??);
    }
    shard_timings.sort_by_key(|timing| timing.shard_id);
    Ok(shard_timings)
}

fn maybe_write_output(path: &Option<PathBuf>, result: &BenchResult) -> DynResult<()> {
    if let Some(path) = path {
        let json = serde_json::to_vec_pretty(result)?;
        std::fs::write(path, json)?;
    }
    Ok(())
}

#[tokio::main]
async fn main() -> DynResult<()> {
    let args = Args::parse();
    validate_args(&args)?;

    let dataset_uri = args.uri.clone().unwrap_or_else(default_dataset_uri);
    let total_start = Instant::now();

    let prepare_start = Instant::now();
    let mut dataset = create_fragmented_dataset(&args, &dataset_uri).await?;
    let dataset_prepare_ms = prepare_start.elapsed().as_millis();

    let fragment_ids = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<Vec<_>>();
    let fragment_groups = split_fragment_groups(&fragment_ids, args.shards);

    let train_start = Instant::now();
    let (ivf_params, pq_params) = prepare_global_ivf_pq(
        &dataset,
        &args.vector_column,
        args.num_partitions,
        args.num_sub_vectors,
        args.num_bits,
        args.max_iters,
        args.sample_rate,
    )
    .await?;
    let train_shared_params_ms = train_start.elapsed().as_millis();

    let shared_uuid = Uuid::new_v4();
    let params = VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params);

    let build_start = Instant::now();
    let shard_timings = build_shards(
        &dataset_uri,
        fragment_groups,
        &args.vector_column,
        &args.index_name,
        &shared_uuid.to_string(),
        params,
    )
    .await?;
    let shard_build_ms = build_start.elapsed().as_millis();

    let finalize_start = Instant::now();
    let index_dir = dataset.indices_dir().child(shared_uuid.to_string());
    finalize_distributed_merge(dataset.object_store(), &index_dir, Some(IndexType::IvfPq)).await?;
    let finalize_ms = finalize_start.elapsed().as_millis();

    let commit_start = Instant::now();
    dataset
        .commit_existing_index(&args.index_name, &args.vector_column, shared_uuid)
        .await?;
    let commit_ms = commit_start.elapsed().as_millis();

    let total_index_build_ms = train_shared_params_ms + shard_build_ms + finalize_ms + commit_ms;
    let result = BenchResult {
        dataset_uri: dataset_uri.clone(),
        index_name: args.index_name.clone(),
        vector_column: args.vector_column.clone(),
        shared_uuid: shared_uuid.to_string(),
        rows: args.fragments * args.rows_per_fragment,
        fragments: args.fragments,
        shards: args.shards,
        rows_per_fragment: args.rows_per_fragment,
        dim: args.dim,
        num_partitions: args.num_partitions,
        num_sub_vectors: args.num_sub_vectors,
        num_bits: args.num_bits,
        dataset_prepare_ms,
        train_shared_params_ms,
        shard_build_ms,
        finalize_ms,
        commit_ms,
        total_index_build_ms,
        total_pipeline_ms: total_start.elapsed().as_millis(),
        shard_timings,
    };

    maybe_write_output(&args.output, &result)?;
    println!("{}", serde_json::to_string_pretty(&result)?);

    if args.cleanup {
        let (store, base) = lance::io::ObjectStore::from_uri(&dataset_uri).await?;
        store.remove_dir_all(base).await?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_args() -> Args {
        Args {
            uri: None,
            output: None,
            fragments: 4,
            rows_per_fragment: 8,
            shards: 2,
            dim: 8,
            num_partitions: 2,
            num_sub_vectors: 2,
            num_bits: 8,
            max_iters: 1,
            sample_rate: 1,
            index_name: "dist_bench_idx".to_string(),
            vector_column: "vector".to_string(),
            cleanup: false,
        }
    }

    #[test]
    fn test_create_batch_uses_configured_vector_column() {
        let batch = create_batch(0, 4, 8, "embedding", 0).unwrap();
        assert!(batch.column_by_name("embedding").is_some());
        assert!(batch.column_by_name("vector").is_none());
    }

    #[test]
    fn test_validate_args_rejects_zero_shards() {
        let mut args = test_args();
        args.shards = 0;
        let err = validate_args(&args).unwrap_err();
        assert_eq!(err.to_string(), "shards must be positive");
    }

    #[test]
    fn test_validate_args_rejects_zero_num_sub_vectors() {
        let mut args = test_args();
        args.num_sub_vectors = 0;
        let err = validate_args(&args).unwrap_err();
        assert_eq!(err.to_string(), "num_sub_vectors must be positive");
    }

    #[tokio::test]
    async fn test_create_fragmented_dataset_does_not_overwrite_existing_dataset() {
        let args = test_args();
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = temp_dir.path().join("bench.lance");
        let uri = uri.to_string_lossy().into_owned();

        create_fragmented_dataset(&args, &uri).await.unwrap();
        let err = create_fragmented_dataset(&args, &uri).await.unwrap_err();
        assert!(err.to_string().contains("already exists"));
    }

    #[tokio::test]
    async fn test_create_fragmented_dataset_uses_configured_vector_column() {
        let mut args = test_args();
        args.vector_column = "embedding".to_string();

        let temp_dir = tempfile::tempdir().unwrap();
        let uri = temp_dir.path().join("bench.lance");
        let uri = uri.to_string_lossy().into_owned();

        let dataset = create_fragmented_dataset(&args, &uri).await.unwrap();
        assert!(dataset.schema().field(&args.vector_column).is_some());
        assert!(dataset.schema().field("vector").is_none());
    }
}
