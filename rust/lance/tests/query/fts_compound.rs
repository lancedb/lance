// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[path = "../../benches/compound_fts/workload.rs"]
mod workload;

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchIterator, StringArray, UInt64Array};
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::index::DatasetIndexExt;
use lance_datafusion::exec::OUTPUT_ROWS_METRIC;
use lance_index::IndexType;
use lance_index::metrics::{
    FTS_CANDIDATES_SCORED_METRIC, FTS_CANDIDATES_VISITED_METRIC, FTS_PHRASE_POSITION_CHECKS_METRIC,
    FTS_POSTING_BLOCKS_DECODED_METRIC,
};
use lance_index::scalar::{InvertedIndexParams, ScalarIndexParams};
use rstest::rstest;
use tempfile::TempDir;
use workload::{
    DatasetKind, FILTER_COLUMN, QueryExecution, TEXT_COLUMN, WorkloadSpec, assert_exact_top_k,
    execute_query, exhaustive_top_k, workload_specs,
};

const ROWS_PER_FRAGMENT: usize = 64;
const INDEXED_FRAGMENTS: usize = 4;

fn fixture_text(row: usize) -> String {
    let local_row = row % ROWS_PER_FRAGMENT;
    if local_row < 4 {
        return ["corpus".to_string()]
            .into_iter()
            .chain((0..8).map(|term| format!("high{term:03}")))
            .collect::<Vec<_>>()
            .join(" ");
    }

    let mut tokens = vec!["corpus".to_string()];
    for term in 0..128 {
        if !(row.wrapping_mul(31) + term * 17).is_multiple_of(4) {
            tokens.push(format!("high{term:03}"));
        }
        if (row.wrapping_mul(13) + term * 7).is_multiple_of(64) {
            tokens.push(format!("low{term:03}"));
        }
    }
    if !row.is_multiple_of(3) {
        for term in 0..8 {
            tokens.push(format!("must{term:03}"));
        }
    }
    for term in 0..16 {
        if !(row + term).is_multiple_of(3) {
            tokens.push(format!("nested{term:03}"));
        }
    }
    if row.is_multiple_of(5) {
        tokens.extend([
            "phraserequired".to_string(),
            "quick".to_string(),
            "brown".to_string(),
        ]);
    }
    if row % 3 != 1 {
        tokens.push("boostpositive".to_string());
    }
    if row.is_multiple_of(2) {
        tokens.push("negativecommon".to_string());
    }
    if row.is_multiple_of(97) {
        tokens.push("negativerare".to_string());
    }
    tokens.join(" ")
}

fn category(row: usize) -> &'static str {
    if row.is_multiple_of(16) {
        "keep"
    } else {
        "drop"
    }
}

fn rich_batch(start_row: usize) -> RecordBatch {
    let ids = Arc::new(UInt64Array::from_iter_values(
        start_row as u64..(start_row + ROWS_PER_FRAGMENT) as u64,
    ));
    let text = Arc::new(StringArray::from_iter_values(
        (start_row..start_row + ROWS_PER_FRAGMENT).map(fixture_text),
    ));
    let category = Arc::new(StringArray::from_iter_values(
        (start_row..start_row + ROWS_PER_FRAGMENT).map(category),
    ));
    RecordBatch::try_from_iter(vec![
        ("id", ids as ArrayRef),
        (TEXT_COLUMN, text as ArrayRef),
        (FILTER_COLUMN, category as ArrayRef),
    ])
    .unwrap()
}

fn fragment_ids(dataset: &Dataset) -> Vec<u32> {
    dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect()
}

async fn append_batch(dataset: &mut Dataset, batch: RecordBatch) {
    let reader = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
    dataset.append(reader, None).await.unwrap();
}

async fn build_rich_dataset(segment_count: usize, reverse_segments: bool) -> (TempDir, Dataset) {
    let tempdir = TempDir::new().unwrap();
    let uri = tempdir.path().join("rich").to_string_lossy().into_owned();
    let batches = (0..INDEXED_FRAGMENTS)
        .map(|fragment| Ok(rich_batch(fragment * ROWS_PER_FRAGMENT)))
        .collect::<Vec<_>>();
    let schema = batches[0].as_ref().unwrap().schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(batches, schema),
        &uri,
        Some(WriteParams {
            max_rows_per_file: ROWS_PER_FRAGMENT,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let ids = fragment_ids(&dataset);
    assert_eq!(ids.len(), INDEXED_FRAGMENTS);
    let params = InvertedIndexParams::default()
        .with_position(true)
        .num_workers(1);
    let mut segments = Vec::new();
    for group in ids.chunks(ids.len().div_ceil(segment_count)) {
        segments.push(
            dataset
                .create_index_builder(&[TEXT_COLUMN], IndexType::Inverted, &params)
                .name("compound_fts".to_string())
                .fragments(group.to_vec())
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    if reverse_segments {
        segments.reverse();
    }
    dataset
        .commit_existing_index_segments("compound_fts", TEXT_COLUMN, segments)
        .await
        .unwrap();

    append_batch(
        &mut dataset,
        rich_batch(INDEXED_FRAGMENTS * ROWS_PER_FRAGMENT),
    )
    .await;
    dataset
        .create_index(
            &[FILTER_COLUMN],
            IndexType::BTree,
            Some("compound_category".to_string()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    (tempdir, dataset)
}

fn wide_batch(start_row: usize) -> RecordBatch {
    let ids = Arc::new(UInt64Array::from_iter_values(
        start_row as u64..(start_row + ROWS_PER_FRAGMENT) as u64,
    ));
    let mut columns = vec![("id".to_string(), ids as ArrayRef)];
    for field in 0..4 {
        let values = Arc::new(StringArray::from_iter_values(
            (start_row..start_row + ROWS_PER_FRAGMENT).map(move |row| {
                if !(row + field).is_multiple_of(5) {
                    "widematch"
                } else {
                    "other"
                }
            }),
        ));
        columns.push((format!("field_{field:03}"), values as ArrayRef));
    }
    RecordBatch::try_from_iter(columns).unwrap()
}

async fn build_wide_dataset() -> (TempDir, Dataset) {
    let tempdir = TempDir::new().unwrap();
    let uri = tempdir.path().join("wide").to_string_lossy().into_owned();
    let batches = (0..2)
        .map(|fragment| Ok(wide_batch(fragment * ROWS_PER_FRAGMENT)))
        .collect::<Vec<_>>();
    let schema = batches[0].as_ref().unwrap().schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(batches, schema),
        &uri,
        Some(WriteParams {
            max_rows_per_file: ROWS_PER_FRAGMENT,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let params = InvertedIndexParams::default().num_workers(1);
    for field in 0..4 {
        let column = format!("field_{field:03}");
        dataset
            .create_index(
                &[column.as_str()],
                IndexType::Inverted,
                Some(format!("wide_{field:03}")),
                &params,
                true,
            )
            .await
            .unwrap();
    }
    append_batch(&mut dataset, wide_batch(2 * ROWS_PER_FRAGMENT)).await;
    (tempdir, dataset)
}

fn metric(execution: &QueryExecution, name: &str) -> usize {
    execution
        .stats
        .all_counts
        .get(name)
        .copied()
        .unwrap_or_default()
}

async fn verify_workload(dataset: &Dataset, workload: &WorkloadSpec) -> Vec<QueryExecution> {
    let exhaustive = execute_query(dataset, workload, None).await.unwrap();
    let mut optimized = Vec::new();
    for k in [10, 100] {
        let expected = exhaustive_top_k(exhaustive.rows.clone(), k);
        let actual = execute_query(dataset, workload, Some(k)).await.unwrap();
        assert_exact_top_k(&workload.name, &expected, &actual.rows).unwrap();
        optimized.push(actual);
    }
    optimized
}

#[test]
fn compound_workload_matrix_covers_acceptance_dimensions() {
    let workloads = workload_specs(&[8, 32, 128], &[2, 4, 8], &[2, 4], &[4, 32, 256, 500]).unwrap();
    let names = workloads
        .iter()
        .map(|workload| workload.name.as_str())
        .collect::<BTreeSet<_>>();
    let families = workloads
        .iter()
        .map(|workload| workload.family)
        .collect::<BTreeSet<_>>();

    for expected in [
        "boolean_should_8_high_df",
        "boolean_should_32_low_df",
        "boolean_should_128_high_df",
        "boolean_must_2",
        "boolean_must_4",
        "boolean_must_8",
        "boolean_must_should_common_must_not",
        "boolean_must_should_rare_must_not",
        "boolean_nested_depth_2",
        "boolean_nested_depth_4",
        "boolean_nested_phrase",
        "boost_common_negative",
        "boost_rare_negative",
        "multi_match_4_fields",
        "multi_match_32_fields",
        "multi_match_256_fields",
        "multi_match_500_fields",
        "boolean_should_8_high_df_selective_prefilter",
    ] {
        assert!(names.contains(expected), "missing workload {expected}");
    }
    assert_eq!(
        families,
        BTreeSet::from([
            "boolean_mixed",
            "boolean_must",
            "boolean_should",
            "boost",
            "multi_match",
            "nested_boolean",
            "nested_phrase",
            "selective_prefilter",
        ])
    );
    assert!(workloads.iter().all(|workload| {
        workload.exercises_fresh_overlay || workload.family == "nested_phrase"
    }));
}

#[rstest]
#[case::single_segment(1, false)]
#[case::multiple_segments(2, false)]
#[case::reversed_segments(2, true)]
#[tokio::test]
async fn compound_rich_queries_match_exhaustive_oracle(
    #[case] segment_count: usize,
    #[case] reverse_segments: bool,
) {
    let (_tempdir, dataset) = build_rich_dataset(segment_count, reverse_segments).await;
    let workloads = workload_specs(&[8], &[2], &[2], &[]).unwrap();
    let workloads = workloads
        .iter()
        .filter(|workload| workload.dataset_kind == DatasetKind::RichText)
        .collect::<Vec<_>>();
    let mut candidates_visited = 0;
    let mut candidates_scored = 0;
    let mut posting_blocks_decoded = 0;
    let mut phrase_position_checks = 0;
    let mut rows_materialized = 0;
    let mut measured_elapsed = Duration::ZERO;

    for workload in workloads {
        let optimized = verify_workload(&dataset, workload).await;
        for execution in &optimized {
            candidates_visited += metric(execution, FTS_CANDIDATES_VISITED_METRIC);
            candidates_scored += metric(execution, FTS_CANDIDATES_SCORED_METRIC);
            posting_blocks_decoded += metric(execution, FTS_POSTING_BLOCKS_DECODED_METRIC);
            phrase_position_checks += metric(execution, FTS_PHRASE_POSITION_CHECKS_METRIC);
            rows_materialized += metric(execution, OUTPUT_ROWS_METRIC);
            measured_elapsed += execution.elapsed;
        }
    }

    assert!(candidates_visited > 0);
    assert!(candidates_scored > 0);
    assert!(posting_blocks_decoded > 0);
    assert!(phrase_position_checks > 0);
    assert!(rows_materialized > 0);
    assert!(measured_elapsed > Duration::ZERO);

    let tie_workload = workload_specs(&[8], &[], &[], &[])
        .unwrap()
        .into_iter()
        .find(|workload| workload.name == "boolean_should_8_high_df")
        .unwrap();
    let first = execute_query(&dataset, &tie_workload, Some(10))
        .await
        .unwrap();
    let second = execute_query(&dataset, &tie_workload, Some(10))
        .await
        .unwrap();
    assert_eq!(first.rows, second.rows);
    assert!(
        first
            .rows
            .windows(2)
            .any(|pair| pair[0].score == pair[1].score),
        "fixture should exercise a score tie"
    );
}

#[tokio::test]
async fn compound_multi_match_matches_exhaustive_oracle() {
    let (_tempdir, dataset) = build_wide_dataset().await;
    let workload = workload_specs(&[], &[], &[], &[4])
        .unwrap()
        .into_iter()
        .find(|workload| workload.dataset_kind == DatasetKind::WideFields)
        .unwrap();
    verify_workload(&dataset, &workload).await;
}
