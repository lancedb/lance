// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The equivalence oracle for the logical-plan spike.
//!
//! Two kinds of check, deliberately:
//!
//! * [`assert_logical_plan`] pins the physical plan the logical path emits, so a change in
//!   DataFusion's rule behavior shows up as a test diff rather than a silent replan.
//! * [`assert_paths_agree`] executes *both* paths and compares the rows. This is the check that
//!   actually matters — the two paths are expected to emit different plan strings (the logical
//!   path gets DataFusion's projection pushdown for free, which the imperative builder does not
//!   do), so plan-string equality would be the wrong bar.

use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow_array::RecordBatch;
use datafusion::physical_plan::ExecutionPlan;
use futures::TryStreamExt;
use lance_datafusion::exec::{LanceExecutionOptions, execute_plan};
use lance_datagen::{BatchCount, ByteCount, Dimension, RowCount, array, gen_batch};

use crate::Result;
use crate::dataset::scanner::ColumnOrdering;
use crate::dataset::{Dataset, Scanner};
use crate::utils::test::assert_plan_node_equals;

/// A scanner-configuring closure. Generic rather than a `fn` pointer so a case can close over
/// its query vector or filter string, and taken by reference internally because every oracle
/// applies it twice — once per path.
trait ScanConfig: Fn(&mut Scanner) -> Result<&mut Scanner> {}
impl<F: Fn(&mut Scanner) -> Result<&mut Scanner>> ScanConfig for F {}

/// Sort a result batch by `_rowid` so two plans can be compared as sets.
fn sorted_by_row_id(batch: &RecordBatch) -> Result<RecordBatch> {
    use arrow::compute::{SortColumn, lexsort_to_indices, take};

    let row_id = batch
        .column_by_name(lance_core::ROW_ID)
        .expect("results must carry _rowid for set comparison");
    let indices = lexsort_to_indices(
        &[SortColumn {
            values: row_id.clone(),
            options: None,
        }],
        None,
    )?;
    let columns = batch
        .columns()
        .iter()
        .map(|column| take(column.as_ref(), &indices, None))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(RecordBatch::try_new(batch.schema(), columns)?)
}

/// Build a scanner, apply `config`, and plan it through the logical path.
///
/// `target_parallelism(1)` pins `EnforceDistribution`'s output so plan strings do not depend on
/// the machine's CPU count, matching the convention in the scanner's own plan tests.
async fn logical_plan_for(
    dataset: &Dataset,
    config: impl ScanConfig,
) -> Result<Arc<dyn ExecutionPlan>> {
    let mut scan = dataset.scan();
    scan.target_parallelism(1);
    config(&mut scan)?;
    super::create_plan(&scan).await
}

async fn imperative_plan_for(
    dataset: &Dataset,
    config: impl ScanConfig,
) -> Result<Arc<dyn ExecutionPlan>> {
    let mut scan = dataset.scan();
    scan.target_parallelism(1);
    config(&mut scan)?;
    scan.create_plan().await
}

async fn assert_logical_plan(
    dataset: &Dataset,
    config: impl ScanConfig,
    expected: &str,
) -> Result<()> {
    let plan = logical_plan_for(dataset, config).await?;
    assert_plan_node_equals(plan, expected).await
}

async fn run(plan: Arc<dyn ExecutionPlan>) -> Result<RecordBatch> {
    let schema = plan.schema();
    let batches = execute_plan(plan, LanceExecutionOptions::default())?
        .try_collect::<Vec<_>>()
        .await?;
    Ok(concat_batches(&schema, &batches)?)
}

/// Execute both planning paths and assert they produce the same rows, in the same order.
///
/// This is the default, and deliberately so: row order is observable, so a path that returns the
/// right rows in a different order has still changed what a caller sees. Reach for
/// [`assert_paths_agree_unordered`] only where the order is genuinely not part of the contract,
/// and say why at the call site.
async fn assert_paths_agree(dataset: &Dataset, config: impl ScanConfig) -> Result<()> {
    let expected = run(imperative_plan_for(dataset, &config).await?).await?;
    let actual = run(logical_plan_for(dataset, &config).await?).await?;

    assert_eq!(
        expected.schema(),
        actual.schema(),
        "logical path produced a different output schema"
    );
    assert_eq!(
        expected.num_rows(),
        actual.num_rows(),
        "logical path produced a different row count"
    );
    assert_eq!(expected, actual, "logical path produced different rows");
    Ok(())
}

/// As [`assert_paths_agree`], but comparing row *sets* rather than row sequences.
///
/// Only two shapes need this, and both are cases where nothing in either path establishes an
/// order: an all-ties FTS score sort, and a union of coverage branches. Each call site states
/// which. A new use is a claim that order does not matter — check that before making it.
async fn assert_paths_agree_unordered(dataset: &Dataset, config: impl ScanConfig) -> Result<()> {
    let expected = sorted_by_row_id(&run(imperative_plan_for(dataset, &config).await?).await?)?;
    let actual = sorted_by_row_id(&run(logical_plan_for(dataset, &config).await?).await?)?;

    assert_eq!(
        expected.schema(),
        actual.schema(),
        "logical path produced a different output schema"
    );
    assert_eq!(
        expected.num_rows(),
        actual.num_rows(),
        "logical path produced a different row count"
    );
    assert_eq!(expected, actual, "logical path produced different rows");
    Ok(())
}

const DIM: u32 = 32;

/// Tag a reader's schema with dataset-level metadata.
///
/// The imperative suite's own fixtures do this deliberately — `scanner.rs:6489` says "so it tests
/// all paths that re-construct the schema along the way". The oracle needs it for the same reason:
/// a lowering step that rebuilds an output schema and drops its metadata is invisible against a
/// fixture that has none.
fn tagged(reader: impl arrow_array::RecordBatchReader) -> impl arrow_array::RecordBatchReader {
    use arrow::datatypes::Schema as ArrowSchema;
    use arrow_array::{RecordBatch, RecordBatchIterator};

    let schema = Arc::new(ArrowSchema::new_with_metadata(
        reader.schema().fields().clone(),
        [("dataset".to_string(), "logical".to_string())].into(),
    ));
    let batches = reader
        .map(|batch| {
            let batch = batch?;
            RecordBatch::try_new(schema.clone(), batch.columns().to_vec())
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();
    RecordBatchIterator::new(batches.into_iter().map(Ok), schema)
}

async fn test_dataset() -> Dataset {
    use crate::dataset::WriteParams;
    use arrow::datatypes::Int32Type;

    let data = gen_batch()
        .col("i", array::step::<Int32Type>())
        .col("s", array::rand_utf8(ByteCount::from(8), false))
        .into_reader_rows(RowCount::from(100), BatchCount::from(2));
    Dataset::write(
        tagged(data),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 100,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

/// Written through a real (in-memory) object store rather than `into_ram_dataset`, so index
/// creation and `io_stats_incremental` both work.
async fn vector_dataset() -> Dataset {
    use crate::dataset::WriteParams;
    use arrow::datatypes::{Float32Type, Int32Type};

    // 1024 rows so PQ has enough to train on, split into two fragments so plans exercise the
    // multi-fragment path.
    let data = gen_batch()
        .col("i", array::step::<Int32Type>())
        .col("s", array::rand_utf8(ByteCount::from(8), false))
        .col("vec", array::rand_vec::<Float32Type>(Dimension::from(DIM)))
        .into_reader_rows(RowCount::from(256), BatchCount::from(4));
    Dataset::write(
        tagged(data),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 512,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

fn query_vector() -> arrow_array::Float32Array {
    arrow_array::Float32Array::from((0..DIM).map(|v| v as f32).collect::<Vec<_>>())
}

/// A vector dataset with an IVF_PQ index covering every fragment.
async fn indexed_vector_dataset() -> Dataset {
    use crate::index::DatasetIndexExt;
    use crate::index::vector::VectorIndexParams;
    use lance_index::IndexType;
    use lance_linalg::distance::DistanceType;

    let mut dataset = vector_dataset().await;
    let params = VectorIndexParams::ivf_pq(2, 8, 2, DistanceType::L2, 2);
    dataset
        .create_index(&["vec"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    dataset
}

#[tokio::test]
async fn test_filtered_scan_plan() {
    let dataset = test_dataset().await;

    // The whole plan is a single LanceRead: DataFusion pushed both the projection and the
    // predicate into the scan leaf, and the leaf claims exact filter pushdown, so no FilterExec
    // and no ProjectionExec survive.
    assert_logical_plan(
        &dataset,
        |scan| scan.project(&["s"])?.filter("i > 10 and i < 20"),
        "LanceRead: uri=..., projection=[s], num_fragments=2, range_before=None, range_after=None, \
         row_id=false, row_addr=false, full_filter=i > Int32(10) AND i < Int32(20), \
         refine_filter=i > Int32(10) AND i < Int32(20)",
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn test_paths_agree_on_filtered_scan() {
    let dataset = test_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s"])?.filter("i > 10 and i < 20")
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn test_paths_agree_on_full_scan() {
    let dataset = test_dataset().await;
    assert_paths_agree(&dataset, |scan| Ok(scan)).await.unwrap();
}

#[tokio::test]
async fn test_paths_agree_on_limit() {
    let dataset = test_dataset().await;
    assert_paths_agree(&dataset, |scan| scan.limit(Some(10), Some(5)))
        .await
        .unwrap();
}

#[tokio::test]
async fn test_flat_knn_plan() {
    let dataset = vector_dataset().await;

    // The projection below the search is narrowed to `[vec, _rowid]` by DataFusion's
    // `optimize_projections` reaching through the extension node via `necessary_children_exprs`;
    // `i` and `s` are fetched afterwards by the take.
    //
    // The outer `SortExec` restates the distance ordering above the take. It survives because
    // `FilteredReadExec` advertises no output ordering, so DataFusion cannot see that the take
    // already emits rows in the order the top-k produced them.
    assert_logical_plan(
        &dataset,
        |scan| scan.nearest("vec", &query_vector(), 5),
        "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@1 as _distance]
  SortExec: expr=[_distance@1 ASC NULLS LAST, _rowid@0 ASC NULLS LAST], preserve_partitioning=[false]
    LanceRead: uri=..., projection=[i, s, vec], source=stream(_rowid)
      ProjectionExec: expr=[_rowid@1 as _rowid, _distance@2 as _distance]
        FilterExec: _distance@2 IS NOT NULL
          SortExec: TopK(fetch=5), expr=[_distance@2 ASC NULLS LAST, _rowid@1 ASC NULLS LAST], preserve_partitioning=[false]
            KNNVectorDistance: metric=l2
              LanceRead: uri=..., projection=[vec], num_fragments=2, range_before=None, range_after=None, \
              row_id=true, row_addr=false, full_filter=--, refine_filter=--",
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn test_paths_agree_on_flat_knn() {
    let dataset = vector_dataset().await;
    assert_paths_agree(&dataset, |scan| scan.nearest("vec", &query_vector(), 5))
        .await
        .unwrap();
}

#[tokio::test]
async fn test_paths_agree_on_flat_knn_with_projection() {
    let dataset = vector_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["i"])?.nearest("vec", &query_vector(), 7)
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn test_ann_plan_uses_the_index() {
    let dataset = indexed_vector_dataset().await;

    // Same logical node as the flat case; the only difference is what `ResolveVectorAccessPath`
    // found in the planning context.
    assert_logical_plan(
        &dataset,
        |scan| scan.nearest("vec", &query_vector(), 5),
        "ProjectionExec: ...
  LanceRead: ...
    ProjectionExec: expr=[_rowid@1 as _rowid, _distance@0 as _distance]
      SortExec: TopK(fetch=5), expr=[_distance@0 ASC NULLS LAST, _rowid@1 ASC NULLS LAST], ...
        ANNSubIndex: name=..., k=5, deltas=1, metric=L2
          ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1",
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn test_paths_agree_on_ann() {
    let dataset = indexed_vector_dataset().await;
    assert_paths_agree(&dataset, |scan| scan.nearest("vec", &query_vector(), 5))
        .await
        .unwrap();
}

/// `use_index(false)` is a semantic downgrade to exact search, so the rule must not pick the
/// index even though one exists.
#[tokio::test]
async fn test_exact_accuracy_forces_flat() {
    let dataset = indexed_vector_dataset().await;
    let mut scan = dataset.scan();
    scan.target_parallelism(1);
    scan.nearest("vec", &query_vector(), 5).unwrap();
    scan.use_index(false);

    let plan = super::create_plan(&scan).await.unwrap();
    let text = format!(
        "{}",
        datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
    );
    assert!(text.contains("KNNVectorDistance"), "{text}");
    assert!(!text.contains("ANNSubIndex"), "{text}");
}

/// Context collection is the only stage that reads storage, and what it reads is cached. So
/// planning the same query twice must issue no reads the second time — the same invariant
/// `test_scan_planning_io` pins for the imperative path.
#[tokio::test]
async fn test_planning_is_io_free_once_warm() {
    use lance_io::assert_io_eq;

    let dataset = indexed_vector_dataset().await;
    logical_plan_for(&dataset, |scan| scan.nearest("vec", &query_vector(), 5))
        .await
        .unwrap();
    dataset.object_store.as_ref().io_stats_incremental();

    logical_plan_for(&dataset, |scan| scan.nearest("vec", &query_vector(), 5))
        .await
        .unwrap();
    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_eq!(io_stats, read_iops, 0);
}

/// Doc case (d): the predicate sits below the search, so it restricts the candidate set. It
/// reaches the index as a `PreFilterSource` — visible here as the second child of `ANNSubIndex`.
#[tokio::test]
async fn test_ann_prefilter_feeds_the_index() {
    let dataset = indexed_vector_dataset().await;

    assert_logical_plan(
        &dataset,
        |scan| {
            scan.prefilter(true)
                .filter("i > 10")?
                .nearest("vec", &query_vector(), 5)
        },
        "ProjectionExec: ...
        ANNSubIndex: name=..., k=5, deltas=1, metric=L2
          ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
          LanceRead: uri=..., projection=[], num_fragments=2, range_before=None, range_after=None, \
          row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=i > Int32(10)",
    )
    .await
    .unwrap();
}

/// Doc case (e): the same predicate above the search trims the results instead. The index sees
/// no prefilter, and a `FilterExec` survives above the take.
#[tokio::test]
async fn test_ann_postfilter_stays_above_the_search() {
    let dataset = indexed_vector_dataset().await;

    let plan = logical_plan_for(&dataset, |scan| {
        scan.prefilter(false)
            .filter("i > 10")?
            .nearest("vec", &query_vector(), 5)
    })
    .await
    .unwrap();
    let text = format!(
        "{}",
        datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
    );

    let filter_line = text
        .lines()
        .position(|line| line.contains("FilterExec: i@"))
        .unwrap_or_else(|| panic!("no postfilter in plan:\n{text}"));
    let search_line = text
        .lines()
        .position(|line| line.contains("ANNSubIndex"))
        .unwrap_or_else(|| panic!("no ANN search in plan:\n{text}"));

    // The load-bearing invariant: `PushDownFilter` must treat the search node as opaque. If it
    // ever learns to push through, a postfilter silently becomes a prefilter and the query
    // quietly changes meaning.
    assert!(
        filter_line < search_line,
        "postfilter sank below the search:\n{text}"
    );
}

#[tokio::test]
async fn test_paths_agree_on_ann_prefilter() {
    let dataset = indexed_vector_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.prefilter(true)
            .filter("i > 10")?
            .nearest("vec", &query_vector(), 5)
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn test_paths_agree_on_flat_knn_prefilter() {
    let dataset = vector_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.prefilter(true)
            .filter("i > 10")?
            .nearest("vec", &query_vector(), 5)
    })
    .await
    .unwrap();
}

/// Appendix A: an index that covers only some fragments.
async fn partially_indexed_dataset() -> Dataset {
    use crate::dataset::WriteParams;
    use arrow::datatypes::{Float32Type, Int32Type};

    let mut dataset = indexed_vector_dataset().await;
    let more = gen_batch()
        .col("i", array::step_custom::<Int32Type>(10_000, 1))
        .col("s", array::rand_utf8(ByteCount::from(8), false))
        .col("vec", array::rand_vec::<Float32Type>(Dimension::from(DIM)))
        .into_reader_rows(RowCount::from(64), BatchCount::from(1));
    dataset
        .append(
            more,
            Some(WriteParams {
                mode: crate::dataset::WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    dataset
}

#[tokio::test]
async fn test_combined_knn_ann_splits_into_two_branches() {
    let dataset = partially_indexed_dataset().await;

    let plan = logical_plan_for(&dataset, |scan| scan.nearest("vec", &query_vector(), 5))
        .await
        .unwrap();
    let text = format!(
        "{}",
        datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
    );

    assert!(text.contains("UnionExec"), "no union in plan:\n{text}");
    assert!(text.contains("ANNSubIndex"), "no indexed branch:\n{text}");
    // Two brute-force nodes: the unindexed branch, and the exact re-rank above the union.
    assert_eq!(
        text.matches("KNNVectorDistance").count(),
        2,
        "expected a flat branch and a re-rank:\n{text}"
    );
}

#[tokio::test]
async fn test_paths_agree_on_combined_knn_ann() {
    let dataset = partially_indexed_dataset().await;
    assert_paths_agree(&dataset, |scan| scan.nearest("vec", &query_vector(), 5))
        .await
        .unwrap();
}

/// The stress case from Appendix A: one logical `Filter` has to reach both branches — as a
/// prefilter source on the indexed side and as an ordinary predicate on the flat side.
#[tokio::test]
async fn test_paths_agree_on_combined_knn_ann_prefilter() {
    let dataset = partially_indexed_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.prefilter(true)
            .filter("i > 10")?
            .nearest("vec", &query_vector(), 5)
    })
    .await
    .unwrap();
}

// ---------------------------------------------------------------------------------------------
// Full-text search
// ---------------------------------------------------------------------------------------------

/// Text with enough shared vocabulary that boolean and boost queries actually select subsets.
fn fts_text(row: i32) -> String {
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
fn fts_data(start: i32, count: i32) -> Box<dyn arrow_array::RecordBatchReader + Send> {
    use arrow_array::{
        FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchIterator, StringArray,
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
async fn fts_dataset() -> Dataset {
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
async fn partially_indexed_fts_dataset() -> Dataset {
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
async fn test_paths_agree_on_fts_with_a_fragment_restriction() {
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
fn dataset_fragments(scan: &Scanner) -> &[lance_table::format::Fragment] {
    scan.dataset.fragments()
}

/// A `list<utf8>` text column with a list-element-granularity inverted index.
///
/// The distinct thing about this shape is the schema: hits are `(row, element)` pairs, so the FTS
/// nodes carry a `_doc_index` column and the same row can appear more than once.
async fn list_element_fts_dataset() -> Dataset {
    use crate::dataset::WriteParams;
    use crate::index::DatasetIndexExt;
    use arrow_array::builder::{ListBuilder, StringBuilder};
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
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

fn list_element_match_query(terms: &str) -> lance_index::scalar::FullTextSearchQuery {
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
async fn test_paths_agree_on_list_element_fts() {
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
async fn test_paths_agree_on_list_element_fts_prefilter() {
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
async fn test_paths_agree_on_list_element_fts_phrase() {
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
fn vector_filter_query() -> lance_index::vector::Query {
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
fn text_match_query(terms: &str) -> lance_index::scalar::FullTextSearchQuery {
    use lance_index::scalar::FullTextSearchQuery;
    use lance_index::scalar::inverted::query::MatchQuery;
    FullTextSearchQuery::new_query(
        MatchQuery::new(terms.to_owned())
            .with_column(Some("text".to_owned()))
            .into(),
    )
}

fn match_query(terms: &str) -> lance_index::scalar::FullTextSearchQuery {
    use lance_index::scalar::FullTextSearchQuery;
    use lance_index::scalar::inverted::query::MatchQuery;
    FullTextSearchQuery::new_query(
        MatchQuery::new(terms.to_owned())
            .with_column(Some("s".to_owned()))
            .into(),
    )
}

#[tokio::test]
async fn test_fts_match_uses_the_index() {
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
async fn test_paths_agree_on_fts_match() {
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
async fn test_paths_agree_on_fts_phrase() {
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
async fn test_paths_agree_on_fts_boost() {
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
async fn test_paths_agree_on_fts_boolean() {
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
async fn test_paths_agree_on_fts_multi_match() {
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
async fn test_fts_boost_collapses_to_a_compound_scorer() {
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
async fn test_fts_prefilter_feeds_the_index() {
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
async fn test_paths_agree_on_fts_prefilter() {
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
async fn test_paths_agree_on_fts_postfilter() {
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
async fn test_paths_agree_on_fts_limit() {
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
async fn test_partially_indexed_fts_splits() {
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
async fn test_paths_agree_on_partially_indexed_fts() {
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
async fn test_paths_agree_on_partially_indexed_fts_prefilter() {
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
async fn test_fast_search_skips_the_flat_branch() {
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
async fn test_paths_agree_on_fts_without_an_index() {
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
async fn test_paths_agree_on_fts_filter_postfiltering_a_vector_search() {
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
async fn test_paths_agree_on_fts_filter_prefiltering_a_vector_search() {
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
async fn test_paths_agree_on_vector_filter_postfiltering_an_fts_search() {
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
async fn test_paths_agree_on_vector_filter_prefiltering_an_fts_search() {
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

#[tokio::test]
async fn test_unsupported_shape_is_rejected() {
    let dataset = test_dataset().await;
    let mut scan = dataset.scan();
    scan.with_row_id().include_deleted_rows();

    let err = super::create_plan(&scan).await.unwrap_err();
    assert!(
        matches!(err, crate::Error::NotSupported { .. }),
        "expected NotSupported, got {err:?}"
    );
    assert!(err.to_string().contains("include_deleted_rows"), "{err}");
}

#[tokio::test]
async fn test_paths_agree_on_ordering() {
    let dataset = test_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s", "i"])?
            .order_by(Some(vec![ColumnOrdering::desc_nulls_last("i".into())]))
    })
    .await
    .unwrap();
}

/// A limit must take the first rows *of the ordering*, which is only true if the sort is below the
/// limit. It also blocks the scan-range pushdown a bare limit would get.
#[tokio::test]
async fn test_paths_agree_on_ordering_with_limit() {
    let dataset = test_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s", "i"])?
            .limit(Some(7), Some(3))?
            .order_by(Some(vec![ColumnOrdering::desc_nulls_last("i".into())]))
    })
    .await
    .unwrap();
}

/// The ordering column is not projected, so it has to be read and then dropped again.
#[tokio::test]
async fn test_paths_agree_on_ordering_by_an_unprojected_column() {
    let dataset = test_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s"])?
            .filter("i > 20")?
            .order_by(Some(vec![ColumnOrdering::asc_nulls_first("i".into())]))
    })
    .await
    .unwrap();
}

/// A vector `query_filter` under a *compound* FTS query takes the other rerank branch: the FTS
/// query is planned independently and joined to the vector results on `_rowid`. That join is the
/// only stock DataFusion join in the whole plan, and its physical form decides the output order —
/// so this asserts ordering, not just row membership.
#[tokio::test]
async fn test_paths_agree_on_vector_filter_prefiltering_a_compound_fts_search() {
    use crate::dataset::scanner::QueryFilter;
    use lance_index::scalar::FullTextSearchQuery;
    use lance_index::scalar::inverted::query::PhraseQuery;

    let dataset = fts_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.project(&["s"])?
            .with_row_id()
            .prefilter(true)
            .full_text_search(FullTextSearchQuery::new_query(
                PhraseQuery::new("hello world".to_owned())
                    .with_column(Some("s".to_owned()))
                    .into(),
            ))?
            .filter_query(QueryFilter::Vector(vector_filter_query()))
    })
    .await
    .unwrap();
}

/// An empty fragment selection: the imperative path recognizes it up front and emits an
/// `EmptyExec`, while the logical path plans the search normally and lets the take — restricted to
/// zero fragments — return nothing. The plan shapes differ; the answers must not.
#[tokio::test]
async fn test_paths_agree_on_an_empty_fragment_selection() {
    let dataset = fts_dataset().await;
    assert_paths_agree(&dataset, |scan| {
        scan.with_fragments(Vec::new());
        scan.project(&["s"])?
            .with_row_id()
            .full_text_search(match_query("hello"))
    })
    .await
    .unwrap();
}

/// A top-k whose result spans more than one output batch, at real parallelism. Regression test for
/// the take losing the search's distance ordering: `FilteredReadExec` advertises no output
/// ordering, so without an explicit sort above it `execute_plan` merges partitions with a plain
/// coalesce and the global order is scrambled.
#[tokio::test]
async fn test_flat_knn_large_limit_stays_globally_ordered() {
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow::datatypes::Float32Type;
    use arrow_array::cast::AsArray;
    use lance_index::vector::DIST_COL;

    let dataset = gen_batch()
        .col("vec", array::rand_vec::<Float32Type>(Dimension::from(16)))
        .into_ram_dataset(FragmentCount::from(4), FragmentRowCount::from(5_000))
        .await
        .unwrap();
    let query = arrow_array::Float32Array::from(vec![0.0_f32; 16]);

    // BATCH_SIZE_FALLBACK is 8192, so 12k results span batches. The scrambling is a scheduling
    // race, hence the repeats.
    for _ in 0..5 {
        let mut scan = dataset.scan();
        scan.nearest("vec", &query, 12_000).unwrap();
        scan.target_parallelism(8);
        let batch = run(super::create_plan(&scan).await.unwrap()).await.unwrap();

        assert_eq!(batch.num_rows(), 12_000);
        let distances = batch[DIST_COL].as_primitive::<Float32Type>();
        for pair in distances.values().windows(2) {
            assert!(
                pair[0] <= pair[1],
                "results must be globally sorted by distance, found {} before {}",
                pair[0],
                pair[1]
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Data overlays: index coverage at row granularity
// ---------------------------------------------------------------------------------------------
//
// A data overlay committed after an index was built, touching a field that index covers, leaves
// the index describing values that no longer exist. Those rows are a coverage gap in exactly the
// sense a fragment the index never saw is one, and `SplitOnIndexCoverage` fills both the same way.
// The exhaustive behavioral coverage lives in `dataset::tests::dataset_overlay_index_masking`,
// which runs against whichever path the flag selects; these assert the two paths agree.

use crate::dataset::tests::dataset_overlay_index_masking as overlay_fixtures;

#[tokio::test]
async fn test_paths_agree_on_a_vector_search_over_an_overlay() {
    for stable_row_ids in [false, true] {
        let dataset = overlay_fixtures::create_vector_overlay_dataset(stable_row_ids).await;
        let query = arrow_array::Float32Array::from(overlay_fixtures::vec_query());
        assert_paths_agree(&dataset, |scan| {
            scan.nearest("vec", &query, 3)?
                .minimum_nprobes(1)
                .with_row_id()
                .project(&["id"])
        })
        .await
        .unwrap();
    }
}

/// `fast_search` blocks the stale rows but does not re-score them, so the two paths have to agree
/// on dropping the stale hit *and* on not surfacing the moved-on match.
#[tokio::test]
async fn test_paths_agree_on_a_fast_search_over_an_overlay() {
    let dataset = overlay_fixtures::create_vector_overlay_dataset(false).await;
    let query = arrow_array::Float32Array::from(overlay_fixtures::vec_query());
    assert_paths_agree(&dataset, |scan| {
        scan.nearest("vec", &query, 3)?
            .minimum_nprobes(1)
            .fast_search()
            .with_row_id()
            .project(&["id"])
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn test_paths_agree_on_an_fts_search_over_an_overlay() {
    let mut dataset = overlay_fixtures::create_text_dataset(false).await;
    overlay_fixtures::build_text_fts_index(&mut dataset).await;
    let dataset = overlay_fixtures::commit_overlay(
        dataset,
        "logical_fts_overlay",
        0,
        &[1],
        lance_table::format::overlay::OverlayCoverage::dense(roaring::RoaringBitmap::from_iter([
            1,
        ])),
        vec![std::sync::Arc::new(arrow_array::StringArray::from(vec![
            "banana bread",
        ]))],
    )
    .await;

    assert_paths_agree(&dataset, |scan| {
        scan.project(&["id"])?
            .with_row_id()
            .full_text_search(text_match_query("apple"))
    })
    .await
    .unwrap();
}

/// The coverage split the shared rule cannot reach: a scalar index query is derived inside the
/// scan leaf, not carried by a logical node, so its stale rows are handled there.
#[tokio::test]
async fn test_paths_agree_on_a_scalar_index_over_an_overlay() {
    for stable_row_ids in [false, true] {
        let mut dataset = overlay_fixtures::create_base_dataset_with(stable_row_ids).await;
        overlay_fixtures::build_age_index(&mut dataset).await;
        let dataset = overlay_fixtures::commit_overlay(
            dataset,
            "logical_scalar_overlay",
            0,
            &[1],
            lance_table::format::overlay::OverlayCoverage::dense(
                roaring::RoaringBitmap::from_iter([1]),
            ),
            vec![overlay_fixtures::i32_array([Some(50)])],
        )
        .await;

        // `age = 50` now matches the overlaid row (whose index entry says 10) and the untouched
        // row that always said 50, so both the block and the re-read have to be right.
        for filter in ["age = 50", "age = 10", "age = 20"] {
            // Unordered: the coverage split makes this a union of an index lookup and a re-read of
            // the overlaid rows, and a union imposes no order between its branches. The two paths
            // build that union in different places — `scan_impl` versus `SplitOnIndexCoverage` —
            // so they interleave the branches differently while selecting the same rows.
            assert_paths_agree_unordered(&dataset, |scan| {
                scan.filter(filter)?.with_row_id().project(&["id"])
            })
            .await
            .unwrap();
        }
    }
}

/// The oracle compares the two paths to each other, so a schema fact both of them lose is
/// invisible to it. This asserts the fact against the dataset instead.
#[tokio::test]
async fn test_output_schema_keeps_dataset_metadata() -> Result<()> {
    let dataset = test_dataset().await;
    let expected = dataset.schema().metadata.clone();
    assert!(
        !expected.is_empty(),
        "fixture must carry schema metadata for this test to mean anything"
    );

    for (path, plan) in [
        (
            "imperative",
            imperative_plan_for(&dataset, |scan| scan.filter("i > 50")?.project(&["s"])).await?,
        ),
        (
            "logical",
            logical_plan_for(&dataset, |scan| scan.filter("i > 50")?.project(&["s"])).await?,
        ),
    ] {
        assert_eq!(
            plan.schema().metadata(),
            &expected,
            "{path} path's plan schema dropped the dataset's schema metadata"
        );
        // Checked separately from the plan schema because they can disagree, and it is the batch
        // the caller actually receives.
        let batches = execute_plan(plan, LanceExecutionOptions::default())?
            .try_collect::<Vec<_>>()
            .await?;
        assert_eq!(
            batches[0].schema().metadata(),
            &expected,
            "{path} path's output batches dropped the dataset's schema metadata"
        );
    }
    Ok(())
}

/// The check that replaced the deleted idempotence markers: a search whose access path was never
/// resolved is rejected as non-executable rather than lowered to a silent brute-force fallback.
///
/// DataFusion runs this check at the end of the analyzer, which is the stage the resolving rules
/// live in, so a rule that fails to fire surfaces as a planning error.
#[tokio::test]
async fn test_an_unresolved_search_is_not_executable() {
    use datafusion::logical_expr::InvariantLevel;

    let dataset = vector_dataset().await;
    let mut scan = dataset.scan();
    scan.nearest("vec", &query_vector(), 5).unwrap();

    let prepared = super::prepare::PreparedQueries::resolve(&scan)
        .await
        .unwrap();
    let plan = super::builder::build(&scan, &prepared).unwrap();

    let err = plan
        .check_invariants(InvariantLevel::Executable)
        .expect_err("an unresolved search must not pass the executable check");
    assert!(
        err.to_string().contains("no access path resolved"),
        "unexpected error: {err}"
    );
}
