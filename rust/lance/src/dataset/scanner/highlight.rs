// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::physical_plan::{ExecutionPlan, metrics::ExecutionPlanMetricsSet};
use lance_core::datatypes::LogicalType;
use lance_index::IndexCriteria;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::highlight::{
    HighlightLeaf, HighlightOptions, HighlightTerms, PreparedHighlightQuery,
};
use lance_index::scalar::inverted::query::{
    FtsQuery, FtsSearchParams, Operator, try_collect_query_tokens,
};
use lance_index::scalar::inverted::tokenizer::document_tokenizer::{DocType, LanceTokenizer};
use lance_index::scalar::inverted::{DocumentGranularity, InvertedIndex};
use roaring::RoaringBitmap;

use super::{FtsOverlayPlan, Scanner};
use crate::index::scalar::inverted::{load_segment_details, load_segments, resolve_fts_field};
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
use crate::io::exec::fts::{default_text_tokenizer, open_fts_segments, tokenizer_for_match_query};
use crate::io::exec::highlight::{HighlightExec, HighlightSource};
use crate::io::exec::utils::IndexMetrics;
use crate::{Error, Result};
use lance_core::datatypes::OnMissing;

struct PreparedField {
    source: HighlightSource,
    tokenizer: Arc<dyn LanceTokenizer>,
    indices: Vec<Arc<InvertedIndex>>,
}

impl Scanner {
    pub(super) async fn highlight_plan(
        &self,
        input: Arc<dyn ExecutionPlan>,
        query: &FullTextSearchQuery,
        mut options: HighlightOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let granularity = self.fts_document_granularity(&query.query)?;
        let canonical = |column: &str| {
            Ok(resolve_fts_field(self.dataset.schema(), column, granularity)?.canonical_path)
        };
        let query = canonicalize_fields(query.query.clone(), &canonical)?;
        options.fields = options
            .fields
            .iter()
            .map(|field| canonical(field))
            .collect::<Result<_>>()?;
        let mut queries = Vec::new();
        collect_leaves(&query, &mut queries);
        let mut fields = HashMap::<String, PreparedField>::new();
        let mut leaves = Vec::with_capacity(queries.len());
        for query in queries {
            let (columns, text, operator, slop, params) = match &query {
                FtsQuery::Match(query) => (
                    vec![query.column.clone().ok_or_else(|| {
                        Error::internal("highlight Match column was not resolved")
                    })?],
                    query.terms.as_str(),
                    query.operator,
                    None,
                    FtsSearchParams::new()
                        .with_fuzziness(query.fuzziness)
                        .with_prefix_length(query.prefix_length)
                        .with_max_expansions(query.max_expansions),
                ),
                FtsQuery::Phrase(query) => (
                    vec![query.column.clone().ok_or_else(|| {
                        Error::internal("highlight Phrase column was not resolved")
                    })?],
                    query.terms.as_str(),
                    Operator::And,
                    Some(query.slop),
                    FtsSearchParams::new(),
                ),
                FtsQuery::CombinedFields(query) => (
                    query.column_names().map(str::to_owned).collect(),
                    query.terms(),
                    query.operator(),
                    None,
                    FtsSearchParams::new(),
                ),
                _ => return Err(Error::internal("highlight expected a lexical leaf")),
            };
            for column in &columns {
                if !fields.contains_key(column) {
                    fields.insert(
                        column.clone(),
                        self.prepare_highlight_field(column, granularity).await?,
                    );
                }
            }
            let first = columns
                .iter()
                .filter_map(|column| fields.get(column))
                .find(|field| {
                    !matches!(query, FtsQuery::CombinedFields(_)) || !field.indices.is_empty()
                })
                .ok_or_else(|| Error::internal("highlight query has no source fields"))?;
            // CombinedFields uses its first indexed field's analyzer even for
            // fields with no index. Other leaves on those fields may use the
            // default analyzer, so replay is shared per field and analyzer.
            let tokenizer = first.tokenizer.clone();
            let mut query_tokenizer = tokenizer.box_clone();
            let flat_tokens = try_collect_query_tokens(text, &mut query_tokenizer)?;
            let flat = HighlightTerms {
                tokens: Arc::new(flat_tokens),
                has_all_query_positions: true,
            };
            let indexed = if matches!(query, FtsQuery::Match(_)) && !first.indices.is_empty() {
                let mut query_tokenizer =
                    tokenizer_for_match_query(&first.indices[0], params.fuzziness);
                let tokens = try_collect_query_tokens(text, &mut query_tokenizer)?;
                HighlightTerms::for_indices(&first.indices, tokens, &params)?
            } else {
                flat.clone()
            };
            leaves.push(HighlightLeaf {
                fields: columns,
                tokenizer,
                flat,
                indexed,
                operator,
                phrase_slop: slop,
            });
        }
        let mut sources = fields
            .into_values()
            .map(|field| field.source)
            .collect::<Vec<_>>();
        sources.sort_by(|a, b| a.field.canonical_path.cmp(&b.field.canonical_path));
        let roots = sources.iter().map(|source| {
            lance_core::datatypes::format_field_path(&[source.field.root_column.as_str()])
        });
        let projection = self
            .dataset
            .empty_projection()
            .with_row_addr()
            .union_columns(roots, OnMissing::Error)?;
        let input = self.take(input, projection)?;
        let query = PreparedHighlightQuery::try_new(query, leaves, options)?;
        Ok(Arc::new(HighlightExec::new(
            input,
            Arc::new(query),
            Arc::new(sources),
        )))
    }

    async fn prepare_highlight_field(
        &self,
        column: &str,
        granularity: DocumentGranularity,
    ) -> Result<PreparedField> {
        let field = resolve_fts_field(self.dataset.schema(), column, granularity)?;
        if self
            .dataset
            .schema()
            .field_by_id(field.final_field_id)
            .is_some_and(|field| field.logical_type == LogicalType::from("json"))
        {
            return Err(Error::not_supported_source(
                format!("highlight does not support JSON source field '{column}'").into(),
            ));
        }
        let target = self
            .fragments
            .as_deref()
            .unwrap_or_else(|| self.dataset.fragments());
        let mut indexed_fragments = RoaringBitmap::new();
        let mut stale_rows = HashMap::new();
        let mut tokenizer: Arc<dyn LanceTokenizer> = default_text_tokenizer().into();
        let mut indices = Vec::new();
        if let Some(segments) = load_segments(&self.dataset, column, granularity).await? {
            load_segment_details(&self.dataset, column, &segments).await?;
            indices = open_fts_segments(
                &self.dataset,
                column,
                &segments,
                &IndexMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
            )
            .await?;
            if let Some(first) = indices.first() {
                tokenizer = first.tokenizer().into();
            }
            let index = self
                .dataset
                .load_scalar_index(
                    IndexCriteria::default()
                        .for_column(column)
                        .supports_fts()
                        .with_fts_document_granularity(granularity),
                )
                .await?
                .ok_or_else(|| Error::internal("highlight selected FTS index disappeared"))?;
            let unindexed = self
                .dataset
                .unindexed_fragments(&index.name)
                .await?
                .iter()
                .map(|fragment| fragment.id as u32)
                .collect::<RoaringBitmap>();
            indexed_fragments = target
                .iter()
                .map(|fragment| fragment.id as u32)
                .filter(|id| !unindexed.contains(*id))
                .collect();
            match self.fts_overlay_plan(column, granularity, target).await? {
                FtsOverlayPlan::Unchanged(_) => {}
                FtsOverlayPlan::RowLevel {
                    stale_rows: stale, ..
                } => stale_rows = stale,
                FtsOverlayPlan::FullScan => indexed_fragments.clear(),
            }
        }
        if matches!(tokenizer.doc_type(), DocType::Json) {
            return Err(Error::not_supported_source(
                format!("highlight does not support JSON analyzer for '{column}'").into(),
            ));
        }
        Ok(PreparedField {
            source: HighlightSource {
                field,
                indexed_fragments,
                stale_rows,
            },
            tokenizer,
            indices,
        })
    }
}

fn canonicalize_fields(
    mut query: FtsQuery,
    canonical: &impl Fn(&str) -> Result<String>,
) -> Result<FtsQuery> {
    match &mut query {
        FtsQuery::Match(query) => {
            query.column = query.column.as_deref().map(canonical).transpose()?
        }
        FtsQuery::Phrase(query) => {
            query.column = query.column.as_deref().map(canonical).transpose()?
        }
        FtsQuery::MultiMatch(query) => {
            for query in &mut query.match_queries {
                query.column = query.column.as_deref().map(canonical).transpose()?;
            }
        }
        FtsQuery::Boost(query) => {
            *query.positive = canonicalize_fields(*query.positive.clone(), canonical)?;
        }
        FtsQuery::Boolean(query) => {
            for child in query
                .must
                .iter_mut()
                .chain(&mut query.should)
                .chain(&mut query.must_not)
            {
                *child = canonicalize_fields(child.clone(), canonical)?;
            }
        }
        FtsQuery::CombinedFields(query) => {
            *query = lance_index::scalar::inverted::query::CombinedFieldsQuery::try_new(
                query.terms().to_owned(),
                query.column_names().map(canonical).collect::<Result<_>>()?,
            )?
            .with_operator(query.operator())
            .try_with_boosts(query.weighted_columns().map(|(_, weight)| weight).collect())?;
        }
    }
    Ok(query)
}

fn collect_leaves(query: &FtsQuery, leaves: &mut Vec<FtsQuery>) {
    match query {
        FtsQuery::Match(_) | FtsQuery::Phrase(_) | FtsQuery::CombinedFields(_) => {
            leaves.push(query.clone())
        }
        FtsQuery::MultiMatch(query) => {
            leaves.extend(query.match_queries.iter().cloned().map(FtsQuery::Match))
        }
        FtsQuery::Boost(query) => collect_leaves(&query.positive, leaves),
        FtsQuery::Boolean(query) => {
            for child in query
                .must
                .iter()
                .chain(&query.should)
                .chain(&query.must_not)
            {
                collect_leaves(child, leaves);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dataset;
    use crate::dataset::{WriteMode, WriteParams};
    use arrow_array::{Array, RecordBatch, RecordBatchIterator, cast::AsArray, record_batch};
    use lance_index::IndexType;
    use lance_index::scalar::inverted::InvertedIndexParams;
    use lance_index::scalar::inverted::highlight::HIGHLIGHTS_COL;
    use lance_index::scalar::inverted::query::{
        BooleanQuery, BoostQuery, CombinedFieldsQuery, MatchQuery, MultiMatchQuery, Occur,
        PhraseQuery,
    };
    use rstest::rstest;

    async fn write(batch: RecordBatch, params: Option<WriteParams>) -> Dataset {
        Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            "memory://",
            params,
        )
        .await
        .unwrap()
    }

    fn text_match(text: &str, field: &str) -> FtsQuery {
        MatchQuery::new(text.into())
            .with_column(Some(field.into()))
            .with_fuzziness(Some(0))
            .into()
    }

    fn hits(batch: &RecordBatch, row: usize) -> Vec<String> {
        let fragments = batch[HIGHLIGHTS_COL].as_list::<i32>().value(row);
        let fragments = fragments.as_struct();
        let text = fragments.column_by_name("text").unwrap().as_string::<i32>();
        let matches = fragments
            .column_by_name("matches")
            .unwrap()
            .as_list::<i32>();
        (0..fragments.len())
            .flat_map(|i| {
                let spans = matches.value(i);
                let spans = spans.as_struct();
                let starts = spans
                    .column_by_name("start")
                    .unwrap()
                    .as_primitive::<arrow_array::types::UInt64Type>();
                let ends = spans
                    .column_by_name("end")
                    .unwrap()
                    .as_primitive::<arrow_array::types::UInt64Type>();
                (0..spans.len())
                    .map(|j| {
                        text.value(i)[starts.value(j) as usize..ends.value(j) as usize].to_owned()
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[rstest]
    #[case::flat(false)]
    #[case::indexed(true)]
    #[tokio::test]
    async fn query_shapes_preserve_results_and_hidden_projection(#[case] indexed: bool) {
        let batch = record_batch!(
            ("id", Int32, [0, 1, 2, 3]),
            ("title", Utf8, ["beta", "none", "beta", "alpha"]),
            (
                "body",
                Utf8,
                ["alpha", "alpha gamma", "new fast york running café", "beta"]
            )
        )
        .unwrap();
        let mut dataset = write(
            batch,
            Some(WriteParams {
                max_rows_per_file: 2,
                max_rows_per_group: 2,
                ..Default::default()
            }),
        )
        .await;
        if indexed {
            for field in ["title", "body"] {
                dataset
                    .create_index(
                        &[field],
                        IndexType::Inverted,
                        None,
                        &InvertedIndexParams::default().with_position(true),
                        true,
                    )
                    .await
                    .unwrap();
            }
        }
        let queries = vec![
            text_match("alpha", "body"),
            PhraseQuery::new("new york".into())
                .with_column(Some("body".into()))
                .with_slop(1)
                .into(),
            CombinedFieldsQuery::try_new("alpha beta".into(), vec!["title".into(), "body".into()])
                .unwrap()
                .with_operator(Operator::And)
                .into(),
            MultiMatchQuery::try_new("alpha".into(), vec!["title".into(), "body".into()])
                .unwrap()
                .into(),
            BoostQuery::new(
                text_match("alpha", "body"),
                text_match("gamma", "body"),
                None,
            )
            .into(),
            BooleanQuery::new([
                (
                    Occur::Should,
                    BooleanQuery::new([
                        (Occur::Must, text_match("alpha", "body")),
                        (Occur::Must, text_match("beta", "title")),
                    ])
                    .into(),
                ),
                (Occur::Should, text_match("gamma", "body")),
            ])
            .into(),
        ];
        for query in queries {
            let requires_index = matches!(query, FtsQuery::CombinedFields(_));
            let mut scan = dataset.scan();
            scan.full_text_search(FullTextSearchQuery::new_query(query))
                .unwrap()
                .project(&["id"])
                .unwrap()
                .with_row_id()
                .limit(Some(3), None)
                .unwrap();
            if !indexed && requires_index {
                let error = scan.try_into_batch().await.unwrap_err();
                assert!(matches!(error, Error::InvalidInput { .. }));
                assert!(error.to_string().contains("requires an inverted index"));
                continue;
            }
            let baseline = scan.try_into_batch().await.unwrap();
            scan.with_highlight(HighlightOptions::new(["body"]))
                .unwrap();
            let highlighted = scan.try_into_batch().await.unwrap();
            assert_eq!(baseline.num_rows(), highlighted.num_rows());
            assert!(highlighted.column_by_name("body").is_none());
            for field in baseline.schema().fields() {
                assert_eq!(
                    baseline.column_by_name(field.name()),
                    highlighted.column_by_name(field.name())
                );
            }
            assert!(highlighted.column_by_name(HIGHLIGHTS_COL).is_some());
        }
    }

    #[tokio::test]
    async fn mixed_fuzzy_uses_each_retrieval_paths_vocabulary() {
        let batch = record_batch!(("body", Utf8, ["bat cat"])).unwrap();
        let mut dataset = write(batch.clone(), None).await;
        dataset
            .create_index(
                &["body"],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        dataset
            .append(
                RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                Some(WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        let query = MatchQuery::new("cat".into())
            .with_column(Some("body".into()))
            .with_fuzziness(Some(1))
            .with_max_expansions(1);
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap()
            .with_row_id();
        let baseline = scan.try_into_batch().await.unwrap();
        let highlighted = scan
            .with_highlight(HighlightOptions::new(["body"]))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(baseline.num_rows(), 2);
        assert_eq!(
            baseline.column_by_name("_score"),
            highlighted.column_by_name("_score")
        );
        let mut words = (0..highlighted.num_rows())
            .map(|row| hits(&highlighted, row))
            .collect::<Vec<_>>();
        words.sort();
        assert_eq!(words, vec![vec!["bat"], vec!["cat"]]);
    }

    #[tokio::test]
    async fn hidden_boolean_fields_filter_and_pagination() {
        let batch = record_batch!(
            ("id", Int32, [0, 1, 2]),
            ("title", Utf8, ["beta", "none", "beta"]),
            ("body", Utf8, ["alpha", "alpha gamma", "alpha"])
        )
        .unwrap();
        let dataset = write(batch, None).await;
        let query = BooleanQuery::new([
            (
                Occur::Should,
                BooleanQuery::new([
                    (Occur::Must, text_match("alpha", "body")),
                    (Occur::Must, text_match("beta", "title")),
                ])
                .into(),
            ),
            (Occur::Should, text_match("gamma", "body")),
        ]);
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap()
            .filter("id = 1")
            .unwrap()
            .project(&["id"])
            .unwrap();
        let highlighted = scan
            .with_highlight(HighlightOptions::new(["body"]))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(highlighted.num_rows(), 1);
        assert_eq!(hits(&highlighted, 0), vec!["gamma"]);
    }
    #[rstest]
    #[case::row(DocumentGranularity::Row)]
    #[case::element(DocumentGranularity::ListElement)]
    #[tokio::test]
    async fn nested_arrays_preserve_coordinates(#[case] granularity: DocumentGranularity) {
        use arrow_array::builder::{ListBuilder, StringBuilder};
        use arrow_schema::{Field, Schema};
        let mut values = ListBuilder::new(ListBuilder::new(StringBuilder::new()));
        values.values().values().append_value("new");
        values.values().values().append_null();
        values.values().values().append_value("york");
        values.values().append(true);
        values.values().values().append_value("new york");
        values.values().append(true);
        values.append(true);
        values.append(false);
        values.append(true);
        let array = Arc::new(values.finish());
        let schema = Arc::new(Schema::new(vec![Field::new(
            "body",
            array.data_type().clone(),
            true,
        )]));
        let batch = RecordBatch::try_new(schema, vec![array]).unwrap();
        let mut dataset = write(
            batch,
            Some(WriteParams {
                max_rows_per_file: 2,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await;
        dataset
            .create_index(
                &["body"],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default()
                    .with_position(true)
                    .document_granularity(granularity),
                true,
            )
            .await
            .unwrap();
        let query = PhraseQuery::new("new york".into())
            .with_column(Some("body".into()))
            .with_document_granularity(granularity);
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap()
            .with_row_id();
        let baseline = scan.try_into_batch().await.unwrap();
        let highlighted = scan
            .with_highlight(HighlightOptions::new(["body"]).with_max_fragments(3))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        for field in baseline.schema().fields() {
            assert_eq!(
                baseline.column_by_name(field.name()),
                highlighted.column_by_name(field.name())
            );
        }
        assert_eq!(highlighted.num_rows(), 1);
        let fragments = highlighted[HIGHLIGHTS_COL].as_list::<i32>().value(0);
        let coordinates = fragments
            .as_struct()
            .column_by_name("value_index")
            .unwrap()
            .as_list::<i32>();
        let coordinates = (0..coordinates.len())
            .map(|i| {
                coordinates
                    .value(i)
                    .as_primitive::<arrow_array::types::UInt32Type>()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        if granularity == DocumentGranularity::Row {
            assert_eq!(coordinates, vec![vec![0, 0], vec![0, 2], vec![1, 0]]);
            assert_eq!(hits(&highlighted, 0), vec!["new", "york", "new york"]);
        } else {
            assert_eq!(coordinates, vec![vec![1, 0]]);
            assert_eq!(hits(&highlighted, 0), vec!["new york"]);
        }
    }

    #[rstest]
    #[case::physical(false)]
    #[case::stable(true)]
    #[tokio::test]
    async fn updates_deletes_and_pinned_snapshot(#[case] stable: bool) {
        use crate::dataset::UpdateBuilder;
        let batch = record_batch!(
            ("id", Int32, [0, 1, 2]),
            ("body", Utf8, ["old needle", "delete needle", "keep needle"])
        )
        .unwrap();
        let mut dataset = write(
            batch,
            Some(WriteParams {
                max_rows_per_file: 2,
                enable_stable_row_ids: stable,
                ..Default::default()
            }),
        )
        .await;
        dataset
            .create_index(
                &["body"],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        let mut pinned = dataset.scan();
        pinned
            .full_text_search(FullTextSearchQuery::new_query(text_match("needle", "body")))
            .unwrap()
            .with_highlight(HighlightOptions::new(["body"]))
            .unwrap();
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 0")
            .unwrap()
            .set("body", "'new needle'")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        let mut updated = result.new_dataset.as_ref().clone();
        updated.delete("id = 1").await.unwrap();
        let before = pinned.try_into_batch().await.unwrap();
        assert_eq!(before.num_rows(), 3);
        assert!(
            before["body"]
                .as_string::<i32>()
                .iter()
                .flatten()
                .any(|value| value == "old needle")
        );
        let mut scan = updated.scan();
        scan.full_text_search(FullTextSearchQuery::new_query(text_match("needle", "body")))
            .unwrap()
            .with_row_id();
        let baseline = scan.try_into_batch().await.unwrap();
        let highlighted = scan
            .with_highlight(HighlightOptions::new(["body"]))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(highlighted.num_rows(), 2);
        for field in baseline.schema().fields() {
            assert_eq!(
                baseline.column_by_name(field.name()),
                highlighted.column_by_name(field.name())
            );
        }
        for row in 0..highlighted.num_rows() {
            assert_eq!(hits(&highlighted, row), vec!["needle"]);
        }
    }

    #[tokio::test]
    async fn limits_apply_only_after_final_selection() {
        let batch = record_batch!(
            ("id", Int32, [0, 1, 2]),
            (
                "body",
                Utf8,
                ["needle", "needle needle padding padding", "needle"]
            )
        )
        .unwrap();
        let dataset = write(batch, None).await;
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new_query(text_match("needle", "body")))
            .unwrap()
            .filter("id != 1")
            .unwrap()
            .limit(Some(1), Some(1))
            .unwrap();
        let baseline = scan.try_into_batch().await.unwrap();
        let highlighted = scan
            .with_highlight(HighlightOptions::new(["body"]).with_max_analyzed_bytes(6))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(highlighted.num_rows(), 1);
        assert_eq!(
            baseline.column_by_name("id"),
            highlighted.column_by_name("id")
        );
        assert_eq!(hits(&highlighted, 0), vec!["needle"]);
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new_query(text_match("needle", "body")))
            .unwrap()
            .with_highlight(HighlightOptions::new(["body"]).with_max_analyzed_bytes(6))
            .unwrap()
            .filter("id = 1")
            .unwrap();
        let error = scan.try_into_batch().await.unwrap_err();
        assert!(error.to_string().contains("max_analyzed_bytes=6"));
    }

    #[rstest]
    #[case::stem("simple", "run cafe", "Running café", true, true)]
    #[case::code("code", "http", "myHTTPClient", false, false)]
    #[case::grapheme("simple", "cafe", "🙂 cafe\u{301}", false, false)]
    #[tokio::test]
    async fn persisted_analyzers_preserve_source(
        #[case] tokenizer: &str,
        #[case] query: &str,
        #[case] source: &str,
        #[case] stem: bool,
        #[case] fold: bool,
    ) {
        let batch = record_batch!(("body", Utf8, [source])).unwrap();
        let mut dataset = write(batch, None).await;
        let params = InvertedIndexParams::default()
            .base_tokenizer(tokenizer.to_owned())
            .split_identifiers(tokenizer == "code")
            .stem(stem)
            .ascii_folding(fold);
        dataset
            .create_index(&["body"], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new_query(text_match(query, "body")))
            .unwrap()
            .with_highlight(HighlightOptions::new(["body"]))
            .unwrap();
        let highlighted = scan.try_into_batch().await.unwrap();
        assert_eq!(highlighted.num_rows(), 1);
        let expected = if stem {
            vec!["Running", "café"]
        } else if tokenizer == "code" {
            vec!["HTTP"]
        } else {
            vec!["cafe\u{301}"]
        };
        assert_eq!(hits(&highlighted, 0), expected);
    }
    #[tokio::test]
    async fn combined_fields_reuses_indexed_analyzer_for_unindexed_fields() {
        let batch =
            record_batch!(("title", Utf8, ["Running"]), ("body", Utf8, ["unrelated"])).unwrap();
        let mut dataset = write(batch, None).await;
        dataset
            .create_index(
                &["body"],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default().stem(true),
                true,
            )
            .await
            .unwrap();
        let combined =
            CombinedFieldsQuery::try_new("run".into(), vec!["title".into(), "body".into()])
                .unwrap();
        let query = BooleanQuery::new([
            (Occur::Should, text_match("run", "title")),
            (Occur::Should, combined.into()),
        ]);
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap();
        let baseline = scan.try_into_batch().await.unwrap();
        assert_eq!(baseline.num_rows(), 1);
        let highlighted = scan
            .with_highlight(HighlightOptions::new(["TITLE"]))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(hits(&highlighted, 0), vec!["Running"]);
        assert_eq!(
            highlighted.column_by_name("_score"),
            baseline.column_by_name("_score")
        );
    }
    #[tokio::test]
    async fn flat_highlight_respects_the_retrieval_analyzer() {
        let dataset = write(record_batch!(("body", Utf8, ["Rust run"])).unwrap(), None).await;
        let query = BooleanQuery::new([
            (Occur::Should, text_match("rust", "body")),
            (Occur::Should, text_match("run", "body")),
        ]);
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap()
            .with_highlight(HighlightOptions::new(["body"]))
            .unwrap();
        let highlighted = scan.try_into_batch().await.unwrap();
        assert_eq!(highlighted.num_rows(), 1);
        assert_eq!(hits(&highlighted, 0), vec!["run"]);
    }
}
