// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{ArrayRef, FixedSizeListArray, RecordBatch, RecordBatchIterator};
use arrow_array::{cast::AsArray, types::Float32Type};
use arrow_schema::{DataType, Field, FieldRef, Schema as ArrowSchema};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use serde::Serialize;

use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance::index::{DatasetIndexExt, vector::VectorIndexParams};
use lance_arrow::FixedSizeListArrayExt;
use lance_index::vector::kmeans::{KMeansParams, train_kmeans};
use lance_index::{
    IndexType,
    vector::{ivf::IvfBuildParams, pq::PQBuildParams},
};
use lance_linalg::distance::DistanceType;
use lance_table::format::IndexMetadata;
use lance_testing::datagen::generate_random_array;
use tokio::runtime::Runtime;

const NUM_FRAGMENTS: usize = 128;
const ROWS_PER_FRAGMENT: usize = 1024;
const DIM: i32 = 128;
const NUM_SUB_VECTORS: usize = 16;
const NUM_BITS: usize = 8;
const MAX_ITERS: usize = 20;
const SAMPLE_RATE: usize = 8;
const AUXILIARY_FILE_NAME: &str = "auxiliary.idx";

#[derive(Clone, Copy, Debug)]
struct BenchCase {
    num_shards: usize,
    num_partitions: usize,
}

impl BenchCase {
    fn label(&self) -> String {
        format!(
            "pq_shards_{}_partitions_{}",
            self.num_shards, self.num_partitions
        )
    }
}

#[derive(Clone, Debug)]
struct MergeFixture {
    segments: Vec<IndexMetadata>,
    segment_aux_bytes: u64,
    segment_count: usize,
}

#[derive(Debug, Serialize)]
struct CaseMetadata {
    label: String,
    num_shards: usize,
    num_partitions: usize,
    segment_count: usize,
    segment_aux_bytes: u64,
    segment_aux_bytes_per_shard: u64,
    total_rows: usize,
    rows_per_shard: usize,
}

fn dataset_root() -> PathBuf {
    std::env::temp_dir().join(format!(
        "lance_bench_distributed_build_{}_{}_{}",
        NUM_FRAGMENTS, ROWS_PER_FRAGMENT, DIM
    ))
}

fn dataset_uri() -> String {
    format!("file://{}", dataset_root().display())
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

fn criterion_group_root() -> PathBuf {
    workspace_root()
        .join("target")
        .join("criterion")
        .join("distributed_merge_only_ivf_pq")
}

fn bench_cases() -> [BenchCase; 6] {
    [
        BenchCase {
            num_shards: 8,
            num_partitions: 256,
        },
        BenchCase {
            num_shards: 32,
            num_partitions: 256,
        },
        BenchCase {
            num_shards: 128,
            num_partitions: 256,
        },
        BenchCase {
            num_shards: 8,
            num_partitions: 1024,
        },
        BenchCase {
            num_shards: 32,
            num_partitions: 1024,
        },
        BenchCase {
            num_shards: 128,
            num_partitions: 1024,
        },
    ]
}

fn create_batches() -> (Arc<ArrowSchema>, Vec<RecordBatch>) {
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(
            FieldRef::new(Field::new("item", DataType::Float32, true)),
            DIM,
        ),
        false,
    )]));

    let batches = (0..NUM_FRAGMENTS)
        .map(|_| {
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        generate_random_array(ROWS_PER_FRAGMENT * DIM as usize),
                        DIM,
                    )
                    .unwrap(),
                )],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    (schema, batches)
}

async fn create_or_open_dataset() -> Dataset {
    let uri = dataset_uri();
    if let Ok(dataset) = Dataset::open(&uri).await
        && dataset.get_fragments().len() == NUM_FRAGMENTS
    {
        return dataset;
    }

    let dataset_path = dataset_root();
    if dataset_path.exists() {
        fs::remove_dir_all(&dataset_path).unwrap();
    }

    let (schema, batches) = create_batches();
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    let write_params = WriteParams {
        max_rows_per_file: ROWS_PER_FRAGMENT,
        max_rows_per_group: ROWS_PER_FRAGMENT,
        mode: WriteMode::Overwrite,
        ..Default::default()
    };

    let dataset = Dataset::write(reader, &uri, Some(write_params))
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), NUM_FRAGMENTS);
    dataset
}

async fn train_shared_ivf_pq(
    dataset: &Dataset,
    num_partitions: usize,
) -> (IvfBuildParams, PQBuildParams) {
    let batch = dataset
        .scan()
        .project(&["vector".to_string()])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let vectors = batch.column_by_name("vector").unwrap().as_fixed_size_list();
    let dim = vectors.value_length() as usize;
    let values = vectors.values().as_primitive::<Float32Type>();

    let kmeans = train_kmeans::<Float32Type>(
        values,
        KMeansParams::new(None, MAX_ITERS as u32, 1, DistanceType::L2),
        dim,
        num_partitions,
        SAMPLE_RATE,
    )
    .unwrap();

    let centroids = Arc::new(
        FixedSizeListArray::try_new_from_values(
            kmeans.centroids.as_primitive::<Float32Type>().clone(),
            dim as i32,
        )
        .unwrap(),
    );
    let mut ivf_params = IvfBuildParams::try_with_centroids(num_partitions, centroids).unwrap();
    ivf_params.max_iters = MAX_ITERS;
    ivf_params.sample_rate = SAMPLE_RATE;

    let mut pq_train_params = PQBuildParams::new(NUM_SUB_VECTORS, NUM_BITS);
    pq_train_params.max_iters = MAX_ITERS;
    pq_train_params.sample_rate = SAMPLE_RATE;

    let pq = pq_train_params.build(vectors, DistanceType::L2).unwrap();
    let codebook: ArrayRef = Arc::new(pq.codebook.values().as_primitive::<Float32Type>().clone());

    let mut pq_params = PQBuildParams::with_codebook(NUM_SUB_VECTORS, NUM_BITS, codebook);
    pq_params.max_iters = MAX_ITERS;
    pq_params.sample_rate = SAMPLE_RATE;

    (ivf_params, pq_params)
}

fn contiguous_fragment_groups(dataset: &Dataset, num_shards: usize) -> Vec<Vec<u32>> {
    assert_eq!(NUM_FRAGMENTS % num_shards, 0);
    let fragments = dataset.get_fragments();
    let group_size = fragments.len() / num_shards;
    fragments
        .chunks(group_size)
        .map(|group| {
            group
                .iter()
                .map(|frag| frag.id() as u32)
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Auxiliary-file size of an uncommitted segment on disk.
fn segment_auxiliary_bytes(segment: &IndexMetadata) -> u64 {
    let aux_path = dataset_root()
        .join("_indices")
        .join(segment.uuid.to_string())
        .join(AUXILIARY_FILE_NAME);
    fs::metadata(aux_path).map(|m| m.len()).unwrap_or(0)
}

async fn build_partial_fixture(dataset: &mut Dataset, bench_case: BenchCase) -> MergeFixture {
    let fragment_groups = contiguous_fragment_groups(dataset, bench_case.num_shards);
    let (ivf_params, pq_params) =
        Box::pin(train_shared_ivf_pq(dataset, bench_case.num_partitions)).await;
    let params = VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params);

    // Build one uncommitted segment per shard. Segment UUIDs are generated by
    // Lance; collect the returned metadata to feed the merge.
    let mut segments = Vec::with_capacity(fragment_groups.len());
    for fragments in fragment_groups {
        let mut builder = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &params)
            .name("distributed_merge_only".to_string())
            .fragments(fragments);
        let segment = Box::pin(builder.execute_uncommitted()).await.unwrap();
        segments.push(segment);
    }

    let segment_count = segments.len();
    let segment_aux_bytes = segments.iter().map(segment_auxiliary_bytes).sum();

    MergeFixture {
        segments,
        segment_aux_bytes,
        segment_count,
    }
}

fn write_case_metadata(fixtures: &[(BenchCase, MergeFixture)]) {
    let output_dir = criterion_group_root();
    fs::create_dir_all(&output_dir).unwrap();
    let metadata = fixtures
        .iter()
        .map(|(bench_case, fixture)| CaseMetadata {
            label: bench_case.label(),
            num_shards: bench_case.num_shards,
            num_partitions: bench_case.num_partitions,
            segment_count: fixture.segment_count,
            segment_aux_bytes: fixture.segment_aux_bytes,
            segment_aux_bytes_per_shard: fixture.segment_aux_bytes / fixture.segment_count as u64,
            total_rows: NUM_FRAGMENTS * ROWS_PER_FRAGMENT,
            rows_per_shard: (NUM_FRAGMENTS * ROWS_PER_FRAGMENT) / bench_case.num_shards,
        })
        .collect::<Vec<_>>();
    let payload = serde_json::to_vec_pretty(&metadata).unwrap();
    fs::write(output_dir.join("case_metadata.json"), payload).unwrap();
}

fn bench_distributed_merge_only(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut dataset = rt.block_on(create_or_open_dataset());
    let mut fixtures = Vec::new();

    for bench_case in bench_cases() {
        fixtures.push((
            bench_case,
            rt.block_on(build_partial_fixture(&mut dataset, bench_case)),
        ));
    }
    write_case_metadata(&fixtures);

    let dataset = Arc::new(dataset);
    let mut group = c.benchmark_group("distributed_merge_only_ivf_pq");
    group.sample_size(10);

    for (bench_case, fixture) in fixtures {
        group.throughput(Throughput::Bytes(fixture.segment_aux_bytes));
        group.bench_with_input(
            BenchmarkId::new("finalize_only", bench_case.label()),
            &bench_case,
            |b, _| {
                let dataset = dataset.clone();
                let segments = fixture.segments.clone();
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let segments = segments.clone();
                        let start = Instant::now();
                        let merged = rt
                            .block_on(dataset.merge_existing_index_segments(segments))
                            .unwrap();
                        total += start.elapsed();

                        let merged_dir = dataset_root()
                            .join("_indices")
                            .join(merged.uuid.to_string());
                        let _ = fs::remove_dir_all(merged_dir);
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name = benches;
    config = Criterion::default()
        .significance_level(0.1)
        .sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_distributed_merge_only
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_distributed_merge_only
);

criterion_main!(benches);
