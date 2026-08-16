// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Full-text search, including hybrid queries that combine it with vector search.

use arrow_array::RecordBatch;
use std::sync::Arc;

use crate::dataset::{Dataset, Scanner};

use super::harness::*;

// ---------------------------------------------------------------------------------------------
// Full-text search
// ---------------------------------------------------------------------------------------------

/// Text with enough shared vocabulary that boolean and boost queries actually select subsets.
pub(super) fn fts_text(row: i32) -> String {
    const WORDS: [&str; 4] = ["hello", "world", "lance", "search"];
    let mut terms = vec![format!("doc{row}")];
    for (index, word) in WORDS.iter().enumerate() {
        if (row as usize).is_multiple_of(index + 2) {
            terms.push((*word).to_string());
        }
    }
    terms.join(" ")
}

/// `[i, s, vec]` — the vector column is here so the FTS/vector `query_filter` combinations can be
/// tested on the same fixture. Values are a deterministic function of the row so both planning
/// paths see identical data.
pub(super) fn fts_data(start: i32, count: i32) -> Box<dyn arrow_array::RecordBatchReader + Send> {
    use arrow_array::{
        FixedSizeListArray, Float32Array, Int32Array, RecordBatchIterator, StringArray,
    };
    use arrow_schema::{DataType, Field, Schema};

    let schema = Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int32, false),
        Field::new("s", DataType::Utf8, false),
        Field::new(
            "vec",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            true,
        ),
    ]));
    let rows = start..start + count;
    let vectors = Float32Array::from(
        rows.clone()
            .flat_map(|row| (0..DIM).map(move |dim| (row + dim as i32) as f32 % 17.0))
            .collect::<Vec<_>>(),
    );
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(rows.clone().collect::<Vec<_>>())),
            Arc::new(StringArray::from(rows.map(fts_text).collect::<Vec<_>>())),
            Arc::new(
                FixedSizeListArray::try_new(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    DIM as i32,
                    Arc::new(vectors),
                    None,
                )
                .unwrap(),
            ),
        ],
    )
    .unwrap();
    Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema))
}

/// A two-fragment dataset with an inverted index covering everything.
pub(super) async fn fts_dataset() -> Dataset {
    use crate::dataset::WriteParams;
    use crate::index::DatasetIndexExt;
    use lance_index::IndexType;
    use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;

    let mut dataset = Dataset::write(
        fts_data(0, 200),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 100,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    // Positions are required for phrase queries; stop words are kept so short tokens stay
    // searchable, matching the scanner's own FTS fixtures.
    let params = InvertedIndexParams::default()
        .with_position(true)
        .remove_stop_words(false);
    dataset
        .create_index(&["s"], IndexType::Inverted, None, &params, true)
        .await
        .unwrap();
    dataset
}

/// The same dataset with rows appended after the index was built.
pub(super) async fn partially_indexed_fts_dataset() -> Dataset {
    use crate::dataset::{WriteMode, WriteParams};

    let mut dataset = fts_dataset().await;
    dataset
        .append(
            fts_data(1000, 60),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    dataset
}

/// `with_fragments` narrower than the index's coverage. The index still scores every fragment it
/// covers, so the restriction is enforced by the take above the search.
#[tokio::test]
pub(super) async fn test_paths_agree_on_fts_with_a_fragment_restriction() {
    let dataset = fts_dataset().await;
    // Unordered: every document in this fixture scores identically on `hello`, and neither path
    // breaks the tie deterministically. Row-set equality is the strongest honest bar here.
    assert_paths_agree_unordered(&dataset, |scan| {
        let first = vec![dataset_fragments(scan)[0].clone()];
        scan.project(&["s"])?
            .with_row_id()
            .with_fragments(first)
            .full_text_search(match_query("hello"))
    })
    .await
    .unwrap();
}

/// `Scanner::fragments` is private to the scanner module, and a `ScanConfig` closure has no other
/// handle on the dataset, so read the list off the scanner itself.
pub(super) fn dataset_fragments(scan: &Scanner) -> &[lance_table::format::Fragment] {
    scan.dataset.fragments()
}

/// A `list<utf8>` text column with a list-element-granularity inverted index.
///
/// The distinct thing about this shape is the schema: hits are `(row, element)` pairs, so the FTS
/// nodes carry a `_doc_index` column and the same row can appear more than once.
pub(super) async fn list_element_fts_dataset() -> Dataset {
    use crate::dataset::WriteParams;
    use crate::index::DatasetIndexExt;
    use arrow_array::builder::{ListBuilder, StringBuilder};
    use arrow_array::{Int32Array, RecordBatchIterator};
    use lance_index::IndexType;
    use lance_index::scalar::inverted::DocumentGranularity;
    use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;

    let mut tags = ListBuilder::new(StringBuilder::new());
    for row in 0..200 {
        for element in 0..(row % 3) + 1 {
            tags.values().append_value(fts_text(row + element));
        }
        tags.append(true);
    }
    let batch = RecordBatch::try_from_iter(vec![
        (
            "i",
            Arc::new(Int32Array::from((0..200).collect::<Vec<_>>())) as arrow_array::ArrayRef,
        ),
        ("tags", Arc::new(tags.finish()) as arrow_array::ArrayRef),
    ])
    .unwrap();
    let schema = batch.schema();

    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 100,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["tags"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default()
                .with_position(true)
                .remove_stop_words(false)
                .document_granularity(DocumentGranularity::ListElement),
            true,
        )
        .await
        .unwrap();
    dataset
}

pub(super) fn list_element_match_query(terms: &str) -> lance_index::scalar::FullTextSearchQuery {
    use lance_index::scalar::FullTextSearchQuery;
    use lance_index::scalar::inverted::DocumentGranularity;
    use lance_index::scalar::inverted::query::MatchQuery;

    FullTextSearchQuery::new_query(
        MatchQuery::new(terms.to_owned())
            .with_column(Some("tags".to_owned()))
            .with_document_granularity(DocumentGranularity::ListElement)
            .into(),
    )
}

#[tokio::test]
pub(super) async fn test_paths_agree_on_list_element_fts() {
    let dataset = list_element_fts_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["i"])?
            .with_row_id()
            .full_text_search(list_element_match_query("hello"))
    })
    .await
    .unwrap();
}

#[tokio::test]
pub(super) async fn test_paths_agree_on_list_element_fts_prefilter() {
    let dataset = list_element_fts_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["i"])?
            .with_row_id()
            .prefilter(true)
            .filter("i > 10")?
            .full_text_search(list_element_match_query("hello"))
    })
    .await
    .unwrap();
}

#[tokio::test]
pub(super) async fn test_paths_agree_on_list_element_fts_phrase() {
    use lance_index::scalar::FullTextSearchQuery;
    use lance_index::scalar::inverted::DocumentGranularity;
    use lance_index::scalar::inverted::query::PhraseQuery;

    let dataset = list_element_fts_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["i"])?
            .with_row_id()
            .full_text_search(FullTextSearchQuery::new_query(
                PhraseQuery::new("hello world".to_owned())
                    .with_column(Some("tags".to_owned()))
                    .with_document_granularity(DocumentGranularity::ListElement)
                    .into(),
            ))
    })
    .await
    .unwrap();
}

/// A vector query for use as a `QueryFilter::Vector`. `Query` has no builder, so every field is
/// spelled out; `use_index` is false because there is no vector index on the FTS fixture.
pub(super) fn vector_filter_query() -> lance_index::vector::Query {
    use lance_index::vector::Query;
    use lance_linalg::distance::DistanceType;

    Query {
        column: "vec".to_string(),
        key: Arc::new(query_vector()),
        k: 20,
        lower_bound: None,
        upper_bound: None,
        minimum_nprobes: 1,
        maximum_nprobes: None,
        ef: None,
        refine_factor: None,
        metric_type: Some(DistanceType::L2),
        use_index: false,
        query_parallelism: 0,
        dist_q_c: 0.0,
        approx_mode: Default::default(),
    }
}

/// As [`match_query`], but against the `text` column of the overlay fixtures.
pub(super) fn text_match_query(terms: &str) -> lance_index::scalar::FullTextSearchQuery {
    use lance_index::scalar::FullTextSearchQuery;
    use lance_index::scalar::inverted::query::MatchQuery;
    FullTextSearchQuery::new_query(
        MatchQuery::new(terms.to_owned())
            .with_column(Some("text".to_owned()))
            .into(),
    )
}

pub(super) fn match_query(terms: &str) -> lance_index::scalar::FullTextSearchQuery {
    use lance_index::scalar::FullTextSearchQuery;
    use lance_index::scalar::inverted::query::MatchQuery;
    FullTextSearchQuery::new_query(
        MatchQuery::new(terms.to_owned())
            .with_column(Some("s".to_owned()))
            .into(),
    )
}

#[tokio::test]
pub(super) async fn test_fts_match_uses_the_index() {
    let dataset = fts_dataset().await;

    // The leaf lowered straight to a `MatchQuery` exec with no prefilter child, and the take
    // above it is the ordinary late materialization every search gets.
    assert_logical_plan(
        &dataset,
        |scan| scan.project(&["s"])?.full_text_search(match_query("hello")),
        "ProjectionExec: expr=[s@2 as s, _score@1 as _score]
  LanceRead: uri=..., projection=[s], source=stream(_rowid)
    MatchQuery: column=s, query=[hello]",
    )
    .await
    .unwrap();
}

#[tokio::test]
pub(super) async fn test_paths_agree_on_fts_match() {
    let dataset = fts_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s"])?
            .with_row_id()
            .full_text_search(match_query("hello"))
    })
    .await
    .unwrap();
}

#[tokio::test]
pub(super) async fn test_paths_agree_on_fts_phrase() {
    use lance_index::scalar::FullTextSearchQuery;
    use lance_index::scalar::inverted::query::PhraseQuery;

    let dataset = fts_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s"])?
            .with_row_id()
            .full_text_search(FullTextSearchQuery::new_query(
                PhraseQuery::new("hello world".to_owned())
                    .with_column(Some("s".to_owned()))
                    .into(),
            ))
    })
    .await
    .unwrap();
}

#[tokio::test]
pub(super) async fn test_paths_agree_on_fts_boost() {
    use lance_index::scalar::FullTextSearchQuery;
    use lance_index::scalar::inverted::query::{BoostQuery, MatchQuery};

    let dataset = fts_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        let positive = MatchQuery::new("hello".to_owned()).with_column(Some("s".to_owned()));
        let negative = MatchQuery::new("world".to_owned()).with_column(Some("s".to_owned()));
        scan.project(&["s"])?
            .with_row_id()
            .full_text_search(FullTextSearchQuery::new_query(
                BoostQuery::new(positive.into(), negative.into(), Some(1.0)).into(),
            ))
    })
    .await
    .unwrap();
}

#[tokio::test]
pub(super) async fn test_paths_agree_on_fts_boolean() {
    use lance_index::scalar::FullTextSearchQuery;
    use lance_index::scalar::inverted::query::{BooleanQuery, MatchQuery, Occur};

    let dataset = fts_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        let must = MatchQuery::new("hello".to_owned()).with_column(Some("s".to_owned()));
        let should = MatchQuery::new("lance".to_owned()).with_column(Some("s".to_owned()));
        let excluded = MatchQuery::new("search".to_owned()).with_column(Some("s".to_owned()));
        let query = BooleanQuery::new(vec![
            (Occur::Must, must.into()),
            (Occur::Should, should.into()),
            (Occur::MustNot, excluded.into()),
        ]);
        scan.project(&["s"])?
            .with_row_id()
            .full_text_search(FullTextSearchQuery::new_query(query.into()))
    })
    .await
    .unwrap();
}

#[tokio::test]
pub(super) async fn test_paths_agree_on_fts_multi_match() {
    use lance_index::scalar::FullTextSearchQuery;
    use lance_index::scalar::inverted::query::MultiMatchQuery;

    let dataset = fts_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s"])?
            .with_row_id()
            .full_text_search(FullTextSearchQuery::new_query(
                MultiMatchQuery::try_new("hello".to_owned(), vec!["s".to_owned()])
                    .unwrap()
                    .into(),
            ))
    })
    .await
    .unwrap();
}

/// The compound-scorer fast path is a rule that collapses a whole subtree — leaves, and each
/// leaf's copy of the prefilter — into one posting-list scorer.
#[tokio::test]
pub(super) async fn test_fts_boost_collapses_to_a_compound_scorer() {
    use lance_index::scalar::FullTextSearchQuery;
    use lance_index::scalar::inverted::query::{BoostQuery, MatchQuery};

    let dataset = fts_dataset().await;
    let plan = logical_plan_for(&dataset, |scan| {
        let positive = MatchQuery::new("hello".to_owned()).with_column(Some("s".to_owned()));
        let negative = MatchQuery::new("world".to_owned()).with_column(Some("s".to_owned()));
        scan.project(&["s"])?
            .full_text_search(FullTextSearchQuery::new_query(
                BoostQuery::new(positive.into(), negative.into(), Some(1.0)).into(),
            ))
    })
    .await
    .unwrap();
    let text = format!(
        "{}",
        datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
    );
    assert!(text.contains("CompoundFtsScorer"), "{text}");
    assert!(!text.contains("BoostQuery"), "{text}");
}

/// The same structural claim the vector path makes: a predicate below the search becomes the
/// prefilter source, visible here as the `MatchQuery`'s child.
#[tokio::test]
pub(super) async fn test_fts_prefilter_feeds_the_index() {
    let dataset = fts_dataset().await;

    assert_logical_plan(
        &dataset,
        |scan| {
            scan.project(&["s"])?
                .prefilter(true)
                .filter("i > 10")?
                .full_text_search(match_query("hello"))
        },
        "ProjectionExec: expr=[s@2 as s, _score@1 as _score]
  LanceRead: uri=..., projection=[s], source=stream(_rowid)
    MatchQuery: column=s, query=[hello]
      LanceRead: uri=..., projection=[], num_fragments=2, range_before=None, range_after=None, \
      row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=i > Int32(10)",
    )
    .await
    .unwrap();
}

#[tokio::test]
pub(super) async fn test_paths_agree_on_fts_prefilter() {
    let dataset = fts_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s"])?
            .with_row_id()
            .prefilter(true)
            .filter("i > 10")?
            .full_text_search(match_query("hello"))
    })
    .await
    .unwrap();
}

#[tokio::test]
pub(super) async fn test_paths_agree_on_fts_postfilter() {
    let dataset = fts_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s"])?
            .with_row_id()
            .prefilter(false)
            .filter("i > 10")?
            .full_text_search(match_query("hello"))
    })
    .await
    .unwrap();
}

#[tokio::test]
pub(super) async fn test_paths_agree_on_fts_limit() {
    let dataset = fts_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s"])?
            .with_row_id()
            .full_text_search(match_query("doc7"))?
            .limit(Some(3), None)
    })
    .await
    .unwrap();
}

/// An index that covers only some fragments splits into an indexed branch and a flat branch,
/// merged by a stock `Union` + `Sort` + `Limit`.
#[tokio::test]
pub(super) async fn test_partially_indexed_fts_splits() {
    let dataset = partially_indexed_fts_dataset().await;

    let plan = logical_plan_for(&dataset, |scan| {
        scan.project(&["s"])?.full_text_search(match_query("hello"))
    })
    .await
    .unwrap();
    let text = format!(
        "{}",
        datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
    );
    assert!(text.contains("UnionExec"), "no union in plan:\n{text}");
    assert!(text.contains("MatchQuery:"), "no indexed branch:\n{text}");
    assert!(text.contains("FlatMatchQuery:"), "no flat branch:\n{text}");
}

#[tokio::test]
pub(super) async fn test_paths_agree_on_partially_indexed_fts() {
    let dataset = partially_indexed_fts_dataset().await;
    // Unordered: the coverage split unions an index search over the indexed fragments with a
    // brute-force search over the rest, and a union imposes no order between its branches. Which
    // branch's rows land first varies run to run.
    assert_paths_agree_unordered(&dataset, |scan| {
        scan.project(&["s"])?
            .with_row_id()
            .full_text_search(match_query("hello"))
    })
    .await
    .unwrap();
}

#[tokio::test]
pub(super) async fn test_paths_agree_on_partially_indexed_fts_prefilter() {
    let dataset = partially_indexed_fts_dataset().await;
    // Unordered: the coverage split unions an index search over the indexed fragments with a
    // brute-force search over the rest, and a union imposes no order between its branches. Which
    // branch's rows land first varies run to run.
    assert_paths_agree_unordered(&dataset, |scan| {
        scan.project(&["s"])?
            .with_row_id()
            .prefilter(true)
            .filter("i > 10")?
            .full_text_search(match_query("hello"))
    })
    .await
    .unwrap();
}

/// `fast_search` drops the flat branch entirely rather than merging it.
#[tokio::test]
pub(super) async fn test_fast_search_skips_the_flat_branch() {
    let dataset = partially_indexed_fts_dataset().await;

    let plan = logical_plan_for(&dataset, |scan| {
        scan.project(&["s"])?
            .fast_search()
            .full_text_search(match_query("hello"))
    })
    .await
    .unwrap();
    let text = format!(
        "{}",
        datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
    );
    assert!(text.contains("MatchQuery:"), "{text}");
    assert!(!text.contains("FlatMatchQuery:"), "{text}");
}

#[tokio::test]
pub(super) async fn test_paths_agree_on_fts_without_an_index() {
    use crate::dataset::WriteParams;

    let dataset = Dataset::write(
        fts_data(0, 200),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 100,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s"])?
            .with_row_id()
            .full_text_search(match_query("hello"))
    })
    .await
    .unwrap();
}

/// A `query_filter` is only legal alongside the *other* kind of search, and it obeys the same
/// prefilter/postfilter switch: below the search it supplies the candidates, above it trims the
/// results. All four combinations are here because each lowers to a different node.
#[tokio::test]
pub(super) async fn test_paths_agree_on_fts_filter_postfiltering_a_vector_search() {
    use crate::dataset::scanner::QueryFilter;

    let dataset = fts_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s"])?
            .with_row_id()
            .prefilter(false)
            .nearest("vec", &query_vector(), 20)?
            .filter_query(QueryFilter::Fts(match_query("hello")))
    })
    .await
    .unwrap();
}

#[tokio::test]
pub(super) async fn test_paths_agree_on_fts_filter_prefiltering_a_vector_search() {
    use crate::dataset::scanner::QueryFilter;

    let dataset = fts_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s"])?
            .with_row_id()
            .prefilter(true)
            .nearest("vec", &query_vector(), 20)?
            .filter_query(QueryFilter::Fts(match_query("hello")))
    })
    .await
    .unwrap();
}

#[tokio::test]
pub(super) async fn test_paths_agree_on_vector_filter_postfiltering_an_fts_search() {
    use crate::dataset::scanner::QueryFilter;

    let dataset = fts_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s"])?
            .with_row_id()
            .prefilter(false)
            .full_text_search(match_query("hello"))?
            .filter_query(QueryFilter::Vector(vector_filter_query()))
    })
    .await
    .unwrap();
}

#[tokio::test]
pub(super) async fn test_paths_agree_on_vector_filter_prefiltering_an_fts_search() {
    use crate::dataset::scanner::QueryFilter;

    let dataset = fts_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s"])?
            .with_row_id()
            .prefilter(true)
            .full_text_search(match_query("hello"))?
            .filter_query(QueryFilter::Vector(vector_filter_query()))
    })
    .await
    .unwrap();
}
