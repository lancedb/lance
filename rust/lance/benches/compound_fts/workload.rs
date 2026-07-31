// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arrow::array::AsArray;
use arrow::datatypes::{Float32Type, UInt64Type};
use lance::Dataset;
use lance_datafusion::exec::{ExecutionStatsCallback, ExecutionSummaryCounts};
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::SCORE_COL;
use lance_index::scalar::inverted::query::{
    BooleanQuery, BoostQuery, FtsQuery, MatchQuery, MultiMatchQuery, Occur, Operator, PhraseQuery,
};

pub const TEXT_COLUMN: &str = "text";
pub const FILTER_COLUMN: &str = "category";
pub const SELECTIVE_FILTER: &str = "category = 'keep'";
pub const SCORE_ABS_TOLERANCE: f32 = 1.0e-5;
pub const SCORE_REL_TOLERANCE: f32 = 1.0e-5;

pub type BenchError = Box<dyn std::error::Error + Send + Sync>;
pub type BenchResult<T> = Result<T, BenchError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DatasetKind {
    RichText,
    WideFields,
}

#[derive(Clone, Debug)]
pub struct WorkloadSpec {
    pub name: String,
    pub family: &'static str,
    pub dataset_kind: DatasetKind,
    pub query: FtsQuery,
    pub filter: Option<&'static str>,
    pub exercises_fresh_overlay: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredRow {
    pub row_id: u64,
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct QueryExecution {
    pub rows: Vec<ScoredRow>,
    pub elapsed: Duration,
    pub stats: ExecutionSummaryCounts,
}

fn match_query(column: &str, term: impl Into<String>) -> FtsQuery {
    MatchQuery::new(term.into())
        .with_column(Some(column.to_string()))
        .with_operator(Operator::And)
        .into()
}

fn boolean_query(
    should: impl IntoIterator<Item = FtsQuery>,
    must: impl IntoIterator<Item = FtsQuery>,
    must_not: impl IntoIterator<Item = FtsQuery>,
) -> FtsQuery {
    let clauses = should
        .into_iter()
        .map(|query| (Occur::Should, query))
        .chain(must.into_iter().map(|query| (Occur::Must, query)))
        .chain(must_not.into_iter().map(|query| (Occur::MustNot, query)));
    BooleanQuery::new(clauses).into()
}

fn term_queries(prefix: &str, count: usize) -> Vec<FtsQuery> {
    (0..count)
        .map(|index| match_query(TEXT_COLUMN, format!("{prefix}{index:03}")))
        .collect()
}

fn nested_boolean(depth: usize, seed: usize) -> FtsQuery {
    if depth == 0 {
        return match_query(TEXT_COLUMN, format!("nested{:03}", seed % 16));
    }
    boolean_query(
        [nested_boolean(depth - 1, seed + 1)],
        [match_query(TEXT_COLUMN, format!("nested{:03}", seed % 4))],
        [],
    )
}

pub fn workload_specs(
    should_clause_counts: &[usize],
    must_clause_counts: &[usize],
    nested_depths: &[usize],
    multi_match_field_counts: &[usize],
) -> BenchResult<Vec<WorkloadSpec>> {
    let mut workloads = Vec::new();

    for &count in should_clause_counts {
        for (frequency, prefix) in [("high_df", "high"), ("low_df", "low")] {
            workloads.push(WorkloadSpec {
                name: format!("boolean_should_{count}_{frequency}"),
                family: "boolean_should",
                dataset_kind: DatasetKind::RichText,
                query: boolean_query(term_queries(prefix, count), [], []),
                filter: None,
                exercises_fresh_overlay: true,
            });
        }
    }

    for &count in must_clause_counts {
        workloads.push(WorkloadSpec {
            name: format!("boolean_must_{count}"),
            family: "boolean_must",
            dataset_kind: DatasetKind::RichText,
            query: boolean_query([], term_queries("must", count), []),
            filter: None,
            exercises_fresh_overlay: true,
        });
    }

    for (name, must_not_term) in [
        ("common_must_not", "negativecommon"),
        ("rare_must_not", "negativerare"),
    ] {
        workloads.push(WorkloadSpec {
            name: format!("boolean_must_should_{name}"),
            family: "boolean_mixed",
            dataset_kind: DatasetKind::RichText,
            query: boolean_query(
                term_queries("high", 8),
                term_queries("must", 2),
                [match_query(TEXT_COLUMN, must_not_term)],
            ),
            filter: None,
            exercises_fresh_overlay: true,
        });
    }

    workloads.push(WorkloadSpec {
        name: "boolean_should_8_high_df_selective_prefilter".to_string(),
        family: "selective_prefilter",
        dataset_kind: DatasetKind::RichText,
        query: boolean_query(term_queries("high", 8), [], []),
        filter: Some(SELECTIVE_FILTER),
        exercises_fresh_overlay: true,
    });

    for &depth in nested_depths {
        workloads.push(WorkloadSpec {
            name: format!("boolean_nested_depth_{depth}"),
            family: "nested_boolean",
            dataset_kind: DatasetKind::RichText,
            query: nested_boolean(depth, 0),
            filter: None,
            exercises_fresh_overlay: true,
        });
    }

    let phrase = PhraseQuery::new("quick brown".to_string())
        .with_column(Some(TEXT_COLUMN.to_string()))
        .into();
    workloads.push(WorkloadSpec {
        name: "boolean_nested_phrase".to_string(),
        family: "nested_phrase",
        dataset_kind: DatasetKind::RichText,
        query: boolean_query([phrase], [match_query(TEXT_COLUMN, "phraserequired")], []),
        filter: None,
        // The indexed phrase executor intentionally has no flat fallback for
        // unindexed fragments.
        exercises_fresh_overlay: false,
    });

    for (frequency, negative_term) in [
        ("common_negative", "negativecommon"),
        ("rare_negative", "negativerare"),
    ] {
        workloads.push(WorkloadSpec {
            name: format!("boost_{frequency}"),
            family: "boost",
            dataset_kind: DatasetKind::RichText,
            query: BoostQuery::new(
                match_query(TEXT_COLUMN, "boostpositive"),
                match_query(TEXT_COLUMN, negative_term),
                Some(0.25),
            )
            .into(),
            filter: None,
            exercises_fresh_overlay: true,
        });
    }

    for &field_count in multi_match_field_counts {
        let columns = (0..field_count)
            .map(|index| format!("field_{index:03}"))
            .collect::<Vec<_>>();
        workloads.push(WorkloadSpec {
            name: format!("multi_match_{field_count}_fields"),
            family: "multi_match",
            dataset_kind: DatasetKind::WideFields,
            query: MultiMatchQuery::try_new("widematch".to_string(), columns)?.into(),
            filter: None,
            exercises_fresh_overlay: true,
        });
    }

    Ok(workloads)
}

fn canonical_cmp(left: &ScoredRow, right: &ScoredRow) -> std::cmp::Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.row_id.cmp(&right.row_id))
}

pub fn exhaustive_top_k(mut rows: Vec<ScoredRow>, k: usize) -> Vec<ScoredRow> {
    rows.sort_unstable_by(canonical_cmp);
    rows.truncate(k);
    rows
}

pub fn assert_exact_top_k(
    workload_name: &str,
    expected: &[ScoredRow],
    actual: &[ScoredRow],
) -> BenchResult<()> {
    if actual
        .windows(2)
        .any(|pair| canonical_cmp(&pair[0], &pair[1]).is_gt())
    {
        return Err(format!(
            "{workload_name}: optimized results are not ordered by score desc, row_id asc"
        )
        .into());
    }
    if expected.len() != actual.len() {
        return Err(format!(
            "{workload_name}: expected {} rows, got {}",
            expected.len(),
            actual.len()
        )
        .into());
    }
    for (rank, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        if expected.row_id != actual.row_id {
            return Err(format!(
                "{workload_name}: row-id mismatch at rank {rank}: expected {}, got {}",
                expected.row_id, actual.row_id
            )
            .into());
        }
        let difference = (expected.score - actual.score).abs();
        let scale = expected.score.abs().max(actual.score.abs());
        let tolerance = SCORE_ABS_TOLERANCE.max(SCORE_REL_TOLERANCE * scale);
        if difference > tolerance {
            return Err(format!(
                "{workload_name}: score mismatch at rank {rank}, row {}: expected {}, got {}, \
                 difference {} exceeds tolerance {}",
                expected.row_id, expected.score, actual.score, difference, tolerance
            )
            .into());
        }
    }
    Ok(())
}

pub async fn execute_query(
    dataset: &Dataset,
    workload: &WorkloadSpec,
    limit: Option<usize>,
) -> BenchResult<QueryExecution> {
    let collected_stats = Arc::new(Mutex::new(None));
    let stats_sink = collected_stats.clone();
    let stats_callback: ExecutionStatsCallback = Arc::new(move |stats| {
        *stats_sink.lock().expect("scan stats mutex poisoned") = Some(stats.clone());
    });

    let mut scanner = dataset.scan();
    scanner
        .scan_stats_callback(stats_callback)
        .prefilter(true)
        .full_text_search(
            FullTextSearchQuery::new_query(workload.query.clone())
                .limit(limit.map(|limit| limit as i64)),
        )?
        .project(&[lance_core::ROW_ID, SCORE_COL])?;
    if let Some(filter) = workload.filter {
        scanner.filter(filter)?;
    }
    if let Some(limit) = limit {
        scanner.limit(Some(limit as i64), None)?;
    }

    let started = Instant::now();
    let batch = scanner.try_into_batch().await?;
    let elapsed = started.elapsed();
    let row_ids = batch[lance_core::ROW_ID].as_primitive::<UInt64Type>();
    let scores = batch[SCORE_COL].as_primitive::<Float32Type>();
    let rows = row_ids
        .values()
        .iter()
        .copied()
        .zip(scores.values().iter().copied())
        .map(|(row_id, score)| ScoredRow { row_id, score })
        .collect();
    let stats = collected_stats
        .lock()
        .expect("scan stats mutex poisoned")
        .take()
        .unwrap_or_default();
    Ok(QueryExecution {
        rows,
        elapsed,
        stats,
    })
}
