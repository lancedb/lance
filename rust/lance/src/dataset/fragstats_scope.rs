// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Planning-time fragment scope from the fragment column statistics index.
//!
//! Given a query filter, computes the set of fragments that provably contain
//! no matching row, using the per-fragment min/max/null records of any
//! `fragment_column_stats` indices on the filtered columns. The resolution is
//! subtractive: a fragment is only removed from a scan on proof, so fragments
//! without statistics (or with stale statistics) always stay candidates.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::DataType;
use datafusion::logical_expr::Expr;
use lance_core::Result;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::SargableQuery;
use lance_index::scalar::expression::{
    IndexInformationProvider, MultiQueryParser, SargableQueryParser, ScalarIndexExpr,
    apply_scalar_indices,
};
use lance_index::scalar::fragstats::FragmentColumnStatsIndex;
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;

use super::Dataset;
use super::overlay::{collect_overlay_stale_frags, overlaid_fragments};
use crate::index::DatasetIndexExt;
use crate::index::DatasetIndexInternalExt;

const FRAGSTATS_DETAILS_SUFFIX: &str = "FragmentColumnStatsIndexDetails";

struct FragStatsIndexInfo {
    columns: HashMap<String, (DataType, MultiQueryParser)>,
}

impl IndexInformationProvider for FragStatsIndexInfo {
    fn get_index(&self, col: &str) -> Option<(&DataType, &MultiQueryParser)> {
        self.columns
            .get(col)
            .map(|(data_type, parser)| (data_type, parser))
    }
}

/// One opened statistics segment plus the fragments whose statistics are
/// invalidated by newer column overlays (validity: same fragment id, changed
/// column data means the old record must not prune).
struct LoadedSegment {
    index: Arc<dyn lance_index::scalar::ScalarIndex>,
    /// The segment's claimed coverage. Records outside it are coverage that
    /// central maintenance withdrew without rewriting the file (in-place
    /// column rewrites) and must never prune.
    coverage: RoaringBitmap,
    stale_fragments: RoaringBitmap,
}

/// Compute the fragments provably excluded by `full_expr`.
///
/// Returns an empty bitmap when no fragment column statistics index applies.
/// Only fragments with valid, complete statistics records can ever appear in
/// the result.
pub(crate) async fn fragment_stats_excluded(
    dataset: &Dataset,
    full_expr: &Expr,
) -> Result<RoaringBitmap> {
    let indices = dataset.load_indices().await?;
    // column -> statistics index segments for that column
    let mut segments_by_column: HashMap<String, Vec<IndexMetadata>> = HashMap::new();
    let mut info = FragStatsIndexInfo {
        columns: HashMap::new(),
    };
    for index in indices.iter() {
        let Some(details) = &index.index_details else {
            continue;
        };
        if !details.type_url.ends_with(FRAGSTATS_DETAILS_SUFFIX) {
            continue;
        }
        let Some(field_id) = index.keyed_field() else {
            continue;
        };
        let Ok(field_path) = dataset.schema().field_path(field_id) else {
            continue;
        };
        let Some(data_type) = dataset.schema().field(&field_path).map(|f| f.data_type()) else {
            continue;
        };
        info.columns.entry(field_path.clone()).or_insert_with(|| {
            (
                data_type,
                MultiQueryParser::single(Box::new(SargableQueryParser::new(
                    index.name.clone(),
                    "FragmentColumnStats".to_string(),
                    true,
                ))),
            )
        });
        segments_by_column
            .entry(field_path)
            .or_default()
            .push(index.clone());
    }
    if info.columns.is_empty() {
        return Ok(RoaringBitmap::new());
    }

    // Reuse the shared predicate decomposition so the statistics see exactly
    // the sargable pieces of the filter; anything it cannot parse simply
    // provides no exclusion proof.
    let indexed = apply_scalar_indices(full_expr.clone(), &info)?;
    let Some(query_tree) = indexed.scalar_query else {
        return Ok(RoaringBitmap::new());
    };

    // Open the referenced segments (warm after the first query via the index
    // cache) and compute overlay staleness per segment.
    let overlaid = overlaid_fragments(&dataset.manifest.fragments);
    let mut loaded: HashMap<String, Vec<LoadedSegment>> = HashMap::new();
    for (column, segments) in segments_by_column {
        let mut loaded_segments = Vec::with_capacity(segments.len());
        for meta in &segments {
            let Ok(index) = dataset
                .open_scalar_index(&column, &meta.uuid, &NoOpMetricsCollector)
                .await
            else {
                continue;
            };
            if index
                .as_any()
                .downcast_ref::<FragmentColumnStatsIndex>()
                .is_none()
            {
                continue;
            }
            // No bitmap means unknown coverage: the segment proves nothing.
            let Some(coverage) = meta.fragment_bitmap.clone() else {
                continue;
            };
            let mut stale_fragments = RoaringBitmap::new();
            if collect_overlay_stale_frags(meta, &overlaid, &mut stale_fragments, dataset.schema())
                .is_err()
            {
                // Unknown staleness means the whole segment cannot prove
                // anything: skip it entirely.
                continue;
            }
            loaded_segments.push(LoadedSegment {
                index,
                coverage,
                stale_fragments,
            });
        }
        loaded.insert(column, loaded_segments);
    }

    excluded_by_expr(&query_tree, &loaded)
}

impl Dataset {
    /// Of `fragment_ids`, the fragments that may contain rows matching
    /// `filter` (a SQL predicate), in input order.
    ///
    /// Drops only fragments whose valid fragment column statistics PROVE no
    /// live row can match. Fragments without statistics, with stale or
    /// withdrawn coverage, or under unsupported predicates are always
    /// retained: the result is a candidate set, not a match set, and every
    /// returned fragment still needs predicate evaluation.
    pub async fn candidate_fragments(
        &self,
        fragment_ids: impl IntoIterator<Item = u64>,
        filter: &str,
    ) -> Result<Vec<u64>> {
        let schema = Arc::new(arrow_schema::Schema::from(self.schema()));
        let planner = lance_datafusion::planner::Planner::new(schema);
        let expr = planner.parse_filter(filter)?;
        let excluded = fragment_stats_excluded(self, &expr).await?;
        Ok(fragment_ids
            .into_iter()
            .filter(|id| u32::try_from(*id).map_or(true, |id32| !excluded.contains(id32)))
            .collect())
    }
}

/// Fold the parsed predicate tree into an excluded-fragments bitmap.
///
/// Combinators keep the subtractive discipline sound:
/// - AND: a row must satisfy both sides, so a proof from either side excludes
///   the fragment (union).
/// - OR: both branches must be impossible (intersection).
/// - NOT: no exclusion proof is derived (empty).
fn excluded_by_expr(
    expr: &ScalarIndexExpr,
    loaded: &HashMap<String, Vec<LoadedSegment>>,
) -> Result<RoaringBitmap> {
    match expr {
        ScalarIndexExpr::Not(_) => Ok(RoaringBitmap::new()),
        ScalarIndexExpr::And(lhs, rhs) => {
            Ok(excluded_by_expr(lhs, loaded)? | excluded_by_expr(rhs, loaded)?)
        }
        ScalarIndexExpr::Or(lhs, rhs) => {
            Ok(excluded_by_expr(lhs, loaded)? & excluded_by_expr(rhs, loaded)?)
        }
        ScalarIndexExpr::Query(search) => {
            let Some(query) = search.query.as_any().downcast_ref::<SargableQuery>() else {
                return Ok(RoaringBitmap::new());
            };
            let Some(segments) = loaded.get(&search.column) else {
                return Ok(RoaringBitmap::new());
            };
            let mut excluded = RoaringBitmap::new();
            for segment in segments {
                let Some(stats) = segment
                    .index
                    .as_any()
                    .downcast_ref::<FragmentColumnStatsIndex>()
                else {
                    continue;
                };
                let mut segment_excluded = stats.excluded_fragments(query)?;
                segment_excluded &= &segment.coverage;
                segment_excluded -= &segment.stale_fragments;
                excluded |= segment_excluded;
            }
            Ok(excluded)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::dataset::{WriteMode, WriteParams};
    use crate::index::DatasetIndexExt;
    use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType as ArrowDataType, Field, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_index::IndexType;
    use lance_index::scalar::ScalarIndexParams;
    use std::sync::Arc;

    /// v1 lifecycle: create the index first (register-only), then append;
    /// coverage arrives with the appends through same-commit registration.
    /// Fragments 0..1 (0..200) predate the index and stay unknown forever.
    async fn stats_dataset(test_uri: &str) -> crate::Dataset {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "i",
            ArrowDataType::Int64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(0..200))],
        )
        .unwrap();
        let params = WriteParams {
            max_rows_per_file: 100,
            ..Default::default()
        };
        let mut dataset = crate::Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema.clone()),
            test_uri,
            Some(params),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["i"],
                IndexType::FragmentColumnStats,
                None,
                &ScalarIndexParams::new("FragmentColumnStats".to_string()),
                true,
            )
            .await
            .unwrap();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(200..400))],
        )
        .unwrap();
        let append = WriteParams {
            mode: WriteMode::Append,
            max_rows_per_file: 100,
            ..Default::default()
        };
        crate::Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            Arc::new(dataset),
            Some(append),
        )
        .await
        .unwrap()
    }

    async fn collect_values(dataset: &crate::Dataset, filter: &str, use_stats: bool) -> Vec<i64> {
        let mut scanner = dataset.scan();
        scanner.filter(filter).unwrap();
        scanner.use_fragment_stats(use_stats);
        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let mut values: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                b.column_by_name("i")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        values.sort_unstable();
        values
    }

    #[tokio::test]
    async fn test_scan_prunes_covered_fragments_and_keeps_results_identical() {
        let test_uri = TempStrDir::default();
        let dataset = stats_dataset(&test_uri).await;

        for filter in [
            "i >= 350",
            "i < 42",
            "i >= 120 AND i < 260",
            "i < 50 OR i >= 380",
            "i = 205",
            "i >= 9999",
        ] {
            let with_stats = collect_values(&dataset, filter, true).await;
            let without_stats = collect_values(&dataset, filter, false).await;
            assert_eq!(
                with_stats, without_stats,
                "results must be identical for filter {filter}"
            );
        }

        // Fragments 0,1 are pre-index (unknown, always scanned); fragments
        // 2 (200..300) and 3 (300..400) are covered. i >= 350 excludes 2.
        let mut scanner = dataset.scan();
        scanner.filter("i >= 350").unwrap();
        let analyzed = scanner.analyze_plan().await.unwrap();
        assert!(
            analyzed.contains("num_fragments=3"),
            "expected num_fragments=3 in plan:\n{analyzed}"
        );

        let mut scanner = dataset.scan();
        scanner.filter("i >= 350").unwrap();
        scanner.use_fragment_stats(false);
        let analyzed = scanner.analyze_plan().await.unwrap();
        assert!(
            analyzed.contains("num_fragments=4"),
            "expected num_fragments=4 in plan:\n{analyzed}"
        );

        // A filter nothing can match still scans the unknown fragments.
        let mut scanner = dataset.scan();
        scanner.filter("i >= 9999").unwrap();
        let analyzed = scanner.analyze_plan().await.unwrap();
        assert!(
            analyzed.contains("num_fragments=2"),
            "expected num_fragments=2 in plan:\n{analyzed}"
        );
    }

    #[tokio::test]
    async fn test_scan_after_delete_stays_correct() {
        let test_uri = TempStrDir::default();
        let mut dataset = stats_dataset(&test_uri).await;
        dataset.delete("i >= 350").await.unwrap();

        let with_stats = collect_values(&dataset, "i >= 300", true).await;
        let without_stats = collect_values(&dataset, "i >= 300", false).await;
        assert_eq!(with_stats, without_stats);
        assert_eq!(with_stats.len(), 50);
    }

    /// P1-2 regression: the per-plan exclusion must be recomputed when the
    /// scanner is reused with a different filter.
    #[tokio::test]
    async fn test_scanner_reuse_recomputes_exclusion() {
        let test_uri = TempStrDir::default();
        let dataset = stats_dataset(&test_uri).await;

        let mut scanner = dataset.scan();
        scanner.filter("i >= 350").unwrap();
        let first: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let first_rows: usize = first.iter().map(|b| b.num_rows()).sum();
        assert_eq!(first_rows, 50);

        // Same scanner, broader filter: fragment 2 (200..300) must come back.
        scanner.filter("i >= 250").unwrap();
        let second: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let second_rows: usize = second.iter().map(|b| b.num_rows()).sum();
        assert_eq!(second_rows, 150);
    }

    /// P0-1 / P0-2 regression: an in-place column rewrite withdraws the
    /// fragment's coverage (bitmap shrinks, file keeps the record); the
    /// reader must stop pruning by the stale record, and the next append's
    /// copy-on-write must not resurrect it.
    #[tokio::test]
    async fn test_in_place_column_rewrite_withdraws_pruning() {
        use crate::dataset::{MergeInsertBuilder, WhenMatched, WhenNotMatched};

        let test_uri = TempStrDir::default();
        // Three columns so the update's [k, i] source schema is a proper
        // subset and the merge takes the in-place column rewrite path.
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("k", ArrowDataType::Int64, false),
            Field::new("i", ArrowDataType::Int64, false),
            Field::new("pad", ArrowDataType::Int64, false),
        ]));
        let make_batch = |range: std::ops::Range<i64>| {
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from_iter_values(range.clone())),
                    Arc::new(Int64Array::from_iter_values(range.clone())),
                    Arc::new(Int64Array::from_iter_values(range)),
                ],
            )
            .unwrap()
        };
        let params = WriteParams {
            max_rows_per_file: 100,
            ..Default::default()
        };
        let mut dataset = crate::Dataset::write(
            RecordBatchIterator::new([Ok(make_batch(0..100))], schema.clone()),
            &test_uri,
            Some(params.clone()),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["i"],
                IndexType::FragmentColumnStats,
                None,
                &ScalarIndexParams::new("FragmentColumnStats".to_string()),
                true,
            )
            .await
            .unwrap();
        let append = WriteParams {
            mode: WriteMode::Append,
            max_rows_per_file: 100,
            ..Default::default()
        };
        let dataset = crate::Dataset::write(
            RecordBatchIterator::new([Ok(make_batch(100..300))], schema.clone()),
            Arc::new(dataset),
            Some(append.clone()),
        )
        .await
        .unwrap();

        // In-place rewrite of `i` for ten keys inside covered fragment 1
        // (100..200): new values far outside the fragment's recorded bounds.
        let update_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("k", ArrowDataType::Int64, false),
            Field::new("i", ArrowDataType::Int64, false),
        ]));
        let update = RecordBatch::try_new(
            update_schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(150..160)),
                Arc::new(Int64Array::from_iter_values(9000..9010)),
            ],
        )
        .unwrap();
        let job = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["k".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .try_build()
            .unwrap();
        let (dataset, _) = job
            .execute_reader(Box::new(RecordBatchIterator::new(
                [Ok(update)],
                update_schema,
            )))
            .await
            .unwrap();

        let filter = "i >= 9000";
        let with_stats = collect_values(&dataset, filter, true).await;
        let without_stats = collect_values(&dataset, filter, false).await;
        assert_eq!(with_stats, without_stats);
        assert_eq!(with_stats.len(), 10);

        // A later append's bucket copy-on-write must not re-legitimize the
        // withdrawn record.
        let dataset = crate::Dataset::write(
            RecordBatchIterator::new([Ok(make_batch(300..400))], schema),
            dataset.clone(),
            Some(append),
        )
        .await
        .unwrap();
        let with_stats = collect_values(&dataset, filter, true).await;
        assert_eq!(with_stats, without_stats);
    }

    /// Public candidate API: input minus provably-non-matching, order kept,
    /// unknown ids retained.
    #[tokio::test]
    async fn test_candidate_fragments_api() {
        let test_uri = TempStrDir::default();
        let dataset = stats_dataset(&test_uri).await;
        // Fragments: 0,1 pre-index (unknown), 2 = 200..300, 3 = 300..400.
        let candidates = dataset
            .candidate_fragments([0u64, 1, 2, 3, 99], "i >= 350")
            .await
            .unwrap();
        assert_eq!(candidates, vec![0, 1, 3, 99]);
        let candidates = dataset
            .candidate_fragments([3u64, 2, 1], "i >= 9999")
            .await
            .unwrap();
        assert_eq!(candidates, vec![1]);
    }

    #[tokio::test]
    async fn test_vector_search_consumes_stats_scope() {
        use crate::index::vector::VectorIndexParams;
        use arrow_array::{FixedSizeListArray, Float32Array};
        use lance_arrow::FixedSizeListArrayExt;
        use lance_linalg::distance::MetricType;

        const DIM: usize = 8;
        let test_uri = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("i", ArrowDataType::Int64, false),
            Field::new(
                "vec",
                ArrowDataType::FixedSizeList(
                    Arc::new(Field::new("item", ArrowDataType::Float32, true)),
                    DIM as i32,
                ),
                true,
            ),
        ]));
        let make_batch = |start: i64, end: i64| {
            let values = Float32Array::from_iter_values(
                (start..end).flat_map(|v| std::iter::repeat_n(v as f32, DIM)),
            );
            let vecs = FixedSizeListArray::try_new_from_values(values, DIM as i32).unwrap();
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from_iter_values(start..end)),
                    Arc::new(vecs),
                ],
            )
            .unwrap()
        };

        // Fragments 0,1 (0..200): written first, covered by the vector index.
        let params = WriteParams {
            max_rows_per_file: 100,
            ..Default::default()
        };
        let mut dataset = crate::Dataset::write(
            RecordBatchIterator::new([Ok(make_batch(0, 200))], schema.clone()),
            &test_uri,
            Some(params),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                None,
                &VectorIndexParams::ivf_flat(1, MetricType::L2),
                true,
            )
            .await
            .unwrap();
        dataset
            .create_index(
                &["i"],
                IndexType::FragmentColumnStats,
                None,
                &ScalarIndexParams::new("FragmentColumnStats".to_string()),
                true,
            )
            .await
            .unwrap();

        // Fragments 2,3 (200..400): appended after both indices; covered by
        // statistics via same-commit registration, NOT by the vector index.
        let append = WriteParams {
            mode: WriteMode::Append,
            max_rows_per_file: 100,
            ..Default::default()
        };
        let dataset = crate::Dataset::write(
            RecordBatchIterator::new([Ok(make_batch(200, 400))], schema.clone()),
            Arc::new(dataset),
            Some(append),
        )
        .await
        .unwrap();

        async fn knn_ids(dataset: &crate::Dataset, filter: &str, use_stats: bool) -> Vec<i64> {
            const DIM: usize = 8;
            let query = Float32Array::from_iter_values(std::iter::repeat_n(205.0f32, DIM));
            let mut scanner = dataset.scan();
            scanner.nearest("vec", &query, 5).unwrap();
            scanner.prefilter(true);
            scanner.filter(filter).unwrap();
            scanner.use_fragment_stats(use_stats);
            let batches: Vec<RecordBatch> = scanner
                .try_into_stream()
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            let mut ids: Vec<i64> = batches
                .iter()
                .flat_map(|b| {
                    b.column_by_name("i")
                        .unwrap()
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .values()
                        .to_vec()
                })
                .collect();
            ids.sort_unstable();
            ids
        }

        for filter in ["i >= 350", "i >= 150 AND i < 260", "i >= 200"] {
            assert_eq!(
                knn_ids(&dataset, filter, true).await,
                knn_ids(&dataset, filter, false).await,
                "top-k must match for filter {filter}"
            );
        }

        // Candidates entirely in the statistics-covered, vector-unindexed
        // tail: the ANN branch disappears (its only segment covers 0,1).
        let query = Float32Array::from_iter_values(std::iter::repeat_n(205.0f32, 8));
        let mut scanner = dataset.scan();
        scanner.nearest("vec", &query, 5).unwrap();
        scanner.prefilter(true);
        scanner.filter("i >= 350").unwrap();
        let analyzed = scanner.analyze_plan().await.unwrap();
        assert!(
            !analyzed.contains("ANNIvfPartition"),
            "ANN segment should be skipped when its fragments are excluded:\n{analyzed}"
        );

        let mut scanner = dataset.scan();
        scanner.nearest("vec", &query, 5).unwrap();
        scanner.prefilter(true);
        scanner.filter("i >= 350").unwrap();
        scanner.use_fragment_stats(false);
        let analyzed = scanner.analyze_plan().await.unwrap();
        assert!(
            analyzed.contains("ANNIvfPartition"),
            "baseline keeps the ANN branch:\n{analyzed}"
        );
    }
}
