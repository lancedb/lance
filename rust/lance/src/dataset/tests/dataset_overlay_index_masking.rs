// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end tests for data-overlay index masking: a scalar index masks data overlay files so that
//! queries stay correct while overlays remain (stale index hits are dropped and new
//! matches are added by re-evaluating overlay-covered rows on the flat path).

use std::sync::Arc;

use futures::TryStreamExt;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{ArrayRef, Int32Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_index::IndexType;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::ScalarIndexParams;
use lance_index::scalar::inverted::InvertedIndexParams;
use lance_io::utils::CachedFileSize;
use lance_linalg::distance::MetricType;
use lance_table::format::DataFile;
use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};
use roaring::RoaringBitmap;
use rstest::rstest;

use lance_file::writer::{FileWriter, FileWriterOptions};

use crate::Dataset;
use crate::dataset::transaction::{DataOverlayGroup, Operation};
use crate::dataset::{WriteDestination, WriteParams};
use crate::index::DatasetIndexExt;
use crate::index::vector::VectorIndexParams;

/// Two-fragment Int32 dataset: `id` (field 0) = 0..12 and `age` (field 1) = id * 10,
/// six rows per file (fragments 0 and 1). In-memory store so overlay files can be written
/// with a store-relative `data/<name>.lance` path and committed against the dataset.
async fn create_base_dataset() -> Dataset {
    create_base_dataset_with(false).await
}

async fn create_base_dataset_with(stable_row_ids: bool) -> Dataset {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, true),
        ArrowField::new("age", DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..12)),
            Arc::new(Int32Array::from_iter_values((0..12).map(|v| v * 10))),
        ],
    )
    .unwrap();
    let write_params = WriteParams {
        max_rows_per_file: 6,
        enable_stable_row_ids: stable_row_ids,
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    Dataset::write(reader, "memory://", Some(write_params))
        .await
        .unwrap()
}

async fn build_age_index(dataset: &mut Dataset) {
    dataset
        .create_index(
            &["age"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
}

/// Write an overlay file covering `fields` of `fragment_id` with `coverage` and the given
/// per-field value columns, then commit it as a `DataOverlay` transaction. `name` makes
/// the overlay file unique.
async fn commit_overlay(
    dataset: Dataset,
    name: &str,
    fragment_id: u64,
    fields: &[i32],
    coverage: OverlayCoverage,
    columns: Vec<ArrayRef>,
) -> Dataset {
    let read_version = dataset.version().version;
    let overlay_schema = dataset.schema().project_by_ids(fields, true);

    let filename = format!("{name}.lance");
    // Use dataset.base so the path is absolute for file:// stores.
    // to_local_path() prepends '/' to the object_store path, so a bare
    // "data/foo.lance" would resolve to /data/foo.lance (root fs). With
    // base we get e.g. tmp/lance-bench/data/foo.lance → /tmp/lance-bench/data/foo.lance.
    // For memory:// stores base is empty so the result is the same as before.
    let path = dataset.base.clone().join("data").join(filename.as_str());
    let obj_writer = dataset.object_store.create(&path).await.unwrap();
    let mut writer =
        FileWriter::try_new(obj_writer, overlay_schema, FileWriterOptions::default()).unwrap();
    let (major, minor) = writer.version().to_numbers();
    for (i, array) in columns.into_iter().enumerate() {
        writer.write_column(i, array).await.unwrap();
    }
    let summary = writer.finish().await.unwrap();

    let mut data_file = DataFile::new_unstarted(filename, major, minor);
    data_file.fields = writer
        .field_id_to_column_indices()
        .iter()
        .map(|(field_id, _)| *field_id as i32)
        .collect::<Vec<_>>()
        .into();
    data_file.column_indices = writer
        .field_id_to_column_indices()
        .iter()
        .map(|(_, column_index)| *column_index as i32)
        .collect::<Vec<_>>()
        .into();
    data_file.file_size_bytes = CachedFileSize::new(summary.size_bytes);

    let overlay = DataOverlayFile {
        data_file,
        coverage,
        committed_version: 0,
    };
    Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset)),
        Operation::DataOverlay {
            groups: vec![DataOverlayGroup {
                fragment_id,
                overlays: vec![overlay],
            }],
        },
        Some(read_version),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap()
}

/// Sorted `id` values returned by a filtered scan.
async fn ids_matching(dataset: &Dataset, filter: &str) -> Vec<i32> {
    ids_matching_opts(dataset, filter, false).await
}

/// Like [`ids_matching`] but lets a test enable `fast_search()`, which skips unindexed
/// fragments. Overlay masking on indexed fragments must still apply regardless.
async fn ids_matching_opts(dataset: &Dataset, filter: &str, fast_search: bool) -> Vec<i32> {
    let mut scanner = dataset.scan();
    scanner.filter(filter).unwrap().project(&["id"]).unwrap();
    if fast_search {
        scanner.fast_search();
    }
    let batch = scanner.try_into_batch().await.unwrap();
    let mut ids = ids_from_batches(std::slice::from_ref(&batch));
    ids.sort_unstable();
    ids
}

/// Concatenate the `id` (Int32) column from each batch, in batch order.
fn ids_from_batches(batches: &[RecordBatch]) -> Vec<i32> {
    batches
        .iter()
        .flat_map(|b| {
            b.column_by_name("id")
                .unwrap()
                .as_primitive::<Int32Type>()
                .values()
                .to_vec()
        })
        .collect()
}

fn i32_array(values: impl IntoIterator<Item = Option<i32>>) -> ArrayRef {
    Arc::new(Int32Array::from_iter(values))
}

fn fsl(rows: Vec<Vec<f32>>, dim: i32) -> ArrayRef {
    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    let item = Arc::new(ArrowField::new("item", DataType::Float32, true));
    Arc::new(
        arrow_array::FixedSizeListArray::try_new(
            item,
            dim,
            Arc::new(arrow_array::Float32Array::from(flat)),
            None,
        )
        .unwrap(),
    )
}

/// A newer overlay on the indexed field drops stale index hits (the old value no longer
/// matches) and surfaces new matches (the new value is found even though the index never
/// saw it). Mirrors the spec's Bob 25 -> 26 worked example.
///
/// Parametrized over `stable_row_ids` to cover the address-based stale-Take path under both
/// row-id schemes.
#[rstest]
#[tokio::test]
async fn test_overlay_stale_drop_and_new_match(#[values(false, true)] stable_row_ids: bool) {
    let mut dataset = create_base_dataset_with(stable_row_ids).await;
    build_age_index(&mut dataset).await;

    // Fragment 0, offset 1 is id=1, age=10. The overlay (committed after the index)
    // changes its age to 999.
    let dataset = commit_overlay(
        dataset,
        "age_overlay",
        0,
        &[1],
        OverlayCoverage::dense(RoaringBitmap::from_iter([1])),
        vec![i32_array([Some(999)])],
    )
    .await;

    // Stale-drop: the index still holds age=10 for id=1, but its current value is 999,
    // so it must not be returned.
    assert_eq!(ids_matching(&dataset, "age = 10").await, Vec::<i32>::new());
    // New-match: the index never saw age=999, but re-evaluation finds it.
    assert_eq!(ids_matching(&dataset, "age = 999").await, vec![1]);
    // An untouched indexed value is unaffected.
    assert_eq!(ids_matching(&dataset, "age = 20").await, vec![2]);
}

/// Row-level BTree precision: when one row in a covered fragment is stale, only that row is
/// blocked from the index result and re-evaluated on the stale-Take path. Non-stale rows in
/// the same fragment (including one that matches the predicate) remain on the indexed path.
///
/// Setup: fragment 0 has id=5 → age=50 (not stale). Overlay id=1 → age=50 (stale).
/// After the overlay two rows in fragment 0 have age=50. The row-level optimization must
/// return both: id=5 from the index and id=1 from the stale-Take path.
///
/// Parametrized over `stable_row_ids`: with stable row ids enabled the stale-Take path must
/// identify rows by physical address, not `_rowid`, or it would take the wrong rows.
#[rstest]
#[tokio::test]
async fn test_btree_overlay_row_level_precision(#[values(false, true)] stable_row_ids: bool) {
    let mut dataset = create_base_dataset_with(stable_row_ids).await;
    build_age_index(&mut dataset).await;

    // Fragment 0: ids 0-5, ages 0,10,20,30,40,50. Overlay offset 1 (id=1): age 10→50.
    // After this both id=1 and id=5 have age=50, in the same fragment.
    let dataset = commit_overlay(
        dataset,
        "age_row_level",
        0,
        &[1],
        OverlayCoverage::dense(RoaringBitmap::from_iter([1])),
        vec![i32_array([Some(50)])],
    )
    .await;

    // Stale drop: id=1's old age=10 entry must not appear.
    assert_eq!(ids_matching(&dataset, "age = 10").await, Vec::<i32>::new());

    // id=5 via index + id=1 via stale-Take path — both in fragment 0.
    assert_eq!(ids_matching(&dataset, "age = 50").await, vec![1, 5]);

    // Non-stale rows in the same fragment still return correctly.
    assert_eq!(ids_matching(&dataset, "age = 20").await, vec![2]);
    assert_eq!(ids_matching(&dataset, "age = 30").await, vec![3]);
}

/// `fast_search` skips *unindexed fragments*, but overlay masking on indexed fragments must
/// still apply: the drop-stale block and the stale-Take re-eval both run regardless of
/// `fast_search` on the scalar path. A regression that gated overlay masking behind
/// `!fast_search` would leak id=1's stale age=10 hit here.
#[tokio::test]
async fn test_btree_overlay_masked_under_fast_search() {
    let mut dataset = create_base_dataset().await;
    build_age_index(&mut dataset).await;

    // Fragment 0, offset 1 is id=1, age=10. Overlay (committed after the index) → age=999.
    let dataset = commit_overlay(
        dataset,
        "age_fast_search",
        0,
        &[1],
        OverlayCoverage::dense(RoaringBitmap::from_iter([1])),
        vec![i32_array([Some(999)])],
    )
    .await;

    // Stale hit dropped even under fast_search — the block is not gated by fast_search.
    assert_eq!(
        ids_matching_opts(&dataset, "age = 10", true).await,
        Vec::<i32>::new()
    );
    // The scalar re-eval path is likewise not gated, so the new value is still surfaced.
    assert_eq!(
        ids_matching_opts(&dataset, "age = 999", true).await,
        vec![1]
    );
    // An untouched indexed value on the same fragment is unaffected.
    assert_eq!(ids_matching_opts(&dataset, "age = 20", true).await, vec![2]);
}

/// An overlay touching only a non-indexed field excludes nothing from the index on `age`.
#[tokio::test]
async fn test_overlay_on_unrelated_field_excludes_nothing() {
    let mut dataset = create_base_dataset().await;
    build_age_index(&mut dataset).await;

    // Overlay field 0 (`id`), not the indexed `age`. The age index stays fully trusted.
    let dataset = commit_overlay(
        dataset,
        "id_overlay",
        0,
        &[0],
        OverlayCoverage::dense(RoaringBitmap::from_iter([1])),
        vec![i32_array([Some(777)])],
    )
    .await;

    // The age index is still trusted: age=10 finds the offset-1 row, whose id now reads
    // through the overlay as 777. The fragment was not routed to the flat path on account
    // of an overlay that touches no indexed field.
    assert_eq!(ids_matching(&dataset, "age = 10").await, vec![777]);
    // An untouched row is unaffected.
    assert_eq!(ids_matching(&dataset, "age = 20").await, vec![2]);
    // The overlaid id is the new value on read, and the old one is gone.
    assert_eq!(ids_matching(&dataset, "id = 777").await, vec![777]);
    assert_eq!(ids_matching(&dataset, "id = 1").await, Vec::<i32>::new());
}

/// An overlay whose `committed_version <= index.dataset_version` is already incorporated by
/// the index (the index was built reading merged values) and is not excluded.
#[tokio::test]
async fn test_overlay_older_than_index_not_excluded() {
    let dataset = create_base_dataset().await;

    // Commit the overlay first (age of id=1 becomes 999), then build the index on top.
    let mut dataset = commit_overlay(
        dataset,
        "age_overlay_old",
        0,
        &[1],
        OverlayCoverage::dense(RoaringBitmap::from_iter([1])),
        vec![i32_array([Some(999)])],
    )
    .await;
    build_age_index(&mut dataset).await;

    // The index incorporates the overlay, so it returns the merged value directly.
    assert_eq!(ids_matching(&dataset, "age = 999").await, vec![1]);
    assert_eq!(ids_matching(&dataset, "age = 10").await, Vec::<i32>::new());
}

/// A covered offset whose overlay value is NULL overrides the cell to NULL, so the stale
/// index hit for its old value is dropped.
#[tokio::test]
async fn test_overlay_null_override() {
    let mut dataset = create_base_dataset().await;
    build_age_index(&mut dataset).await;

    // id=1 (age=10) is overridden to NULL.
    let dataset = commit_overlay(
        dataset,
        "age_overlay_null",
        0,
        &[1],
        OverlayCoverage::dense(RoaringBitmap::from_iter([1])),
        vec![i32_array([None])],
    )
    .await;

    assert_eq!(ids_matching(&dataset, "age = 10").await, Vec::<i32>::new());
    assert_eq!(ids_matching(&dataset, "age IS NULL").await, vec![1]);
}

/// Overlays on a non-first fragment are masked correctly, and a query spanning both
/// fragments returns the right rows.
///
/// Parametrized over `stable_row_ids`, and crucially overlays fragment 1 (ids 6..12), where a
/// physical address diverges from the stable row id — so this exercises the address-vs-row-id
/// distinction that a fragment-0 overlay cannot.
#[rstest]
#[tokio::test]
async fn test_overlay_multi_fragment(#[values(false, true)] stable_row_ids: bool) {
    let mut dataset = create_base_dataset_with(stable_row_ids).await;
    build_age_index(&mut dataset).await;

    // Fragment 1 holds ids 6..12 (ages 60..110). Offset 2 within fragment 1 is id=8,
    // age=80; change it to 60 (a value that also legitimately exists at id=6).
    let dataset = commit_overlay(
        dataset,
        "age_overlay_frag1",
        1,
        &[1],
        OverlayCoverage::dense(RoaringBitmap::from_iter([2])),
        vec![i32_array([Some(60)])],
    )
    .await;

    // id=8 no longer has age=80 (stale-drop on fragment 1).
    assert_eq!(ids_matching(&dataset, "age = 80").await, Vec::<i32>::new());
    // Both id=6 (base) and id=8 (overlay) now have age=60 (new-match added to base hit).
    assert_eq!(ids_matching(&dataset, "age = 60").await, vec![6, 8]);
    // A value in the untouched fragment 0 is still served correctly.
    assert_eq!(ids_matching(&dataset, "age = 30").await, vec![3]);
}

const VEC_DIM: i32 = 8;

fn vec_query() -> Vec<f32> {
    vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

/// 64-row two-fragment vector dataset with a single-partition IVF_FLAT index, then an overlay
/// on fragment 1 that moves id=35 (offset 3) onto `far` (away from the query) and id=40
/// (offset 8) onto the query. Built before the overlay, the index still believes id=35 is the
/// query and has never seen id=40 near it. Every other base vector is orthogonal to the query.
///
/// Overlaying fragment 1 (ids 32..64) is deliberate: a physical address diverges from the
/// stable row id there, so both the ANN prefilter block and the flat re-score take must operate
/// in the row-id domain when `stable_row_ids` is enabled.
async fn create_vector_overlay_dataset(stable_row_ids: bool) -> Dataset {
    let query = vec_query();
    let far = vec![0.0_f32, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(64);
    for i in 0..64 {
        if i == 35 {
            vectors.push(query.clone());
        } else {
            let mut v = vec![0.0_f32; VEC_DIM as usize];
            v[1] = (i + 2) as f32; // orthogonal to the query, distinct, far
            vectors.push(v);
        }
    }

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, true),
        ArrowField::new(
            "vec",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                VEC_DIM,
            ),
            true,
        ),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..64)),
            fsl(vectors, VEC_DIM),
        ],
    )
    .unwrap();
    let write_params = WriteParams {
        max_rows_per_file: 32,
        enable_stable_row_ids: stable_row_ids,
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let mut dataset = Dataset::write(reader, "memory://", Some(write_params))
        .await
        .unwrap();

    // Single-partition IVF_FLAT: the ANN searches every indexed row with exact distances.
    let params = VectorIndexParams::ivf_flat(1, MetricType::L2);
    dataset
        .create_index(&["vec"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    commit_overlay(
        dataset,
        "vec_overlay",
        1,
        &[1],
        OverlayCoverage::dense(RoaringBitmap::from_iter([3, 8])),
        vec![fsl(vec![far, query], VEC_DIM)],
    )
    .await
}

/// Run a top-`k` ANN search for the standard query vector and return the returned `id`s,
/// optionally with `fast_search()` enabled.
async fn vector_query_ids(dataset: &Dataset, k: usize, fast_search: bool) -> Vec<i32> {
    let mut scanner = dataset.scan();
    scanner
        .nearest("vec", &arrow_array::Float32Array::from(vec_query()), k)
        .unwrap()
        .minimum_nprobes(1)
        .project(&["id"])
        .unwrap();
    if fast_search {
        scanner.fast_search();
    }
    let results = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    ids_from_batches(&results)
}

/// A vector index masks overlays: a row whose vector was moved (by a newer overlay) away
/// from the query is dropped from results, and a row moved *onto* the query is found by
/// re-scoring its current vector on the flat path — even though the index never saw it.
///
/// Parametrized over `stable_row_ids` to cover the row-id domain for both block and re-score.
#[rstest]
#[tokio::test]
async fn test_vector_index_rescore_on_overlay(#[values(false, true)] stable_row_ids: bool) {
    let dataset = create_vector_overlay_dataset(stable_row_ids).await;
    let ids = vector_query_ids(&dataset, 3, false).await;

    // id=40 was moved onto the query and is found by re-scoring (new-match recall).
    assert!(
        ids.contains(&40),
        "expected id=40 (re-scored to query) in {ids:?}"
    );
    // id=35's stale index entry (the query) must not resurface: its current vector is far.
    assert!(
        !ids.contains(&35),
        "stale vector for id=35 should be dropped, got {ids:?}"
    );
}

/// The ANN prefilter block that drops stale overlay rows runs regardless of `fast_search`;
/// only the flat re-score is gated by it. So under `fast_search` id=35's stale hit must still
/// be dropped, while id=40 (moved onto the query) is intentionally not re-scored — the same
/// recall tradeoff `fast_search` already makes for unindexed data. A regression that moved the
/// `overlay_block` computation inside the `!fast_search` guard would leak id=35's stale vector.
#[tokio::test]
async fn test_vector_overlay_stale_dropped_under_fast_search() {
    let dataset = create_vector_overlay_dataset(false).await;
    let ids = vector_query_ids(&dataset, 3, true).await;

    // Correctness: the stale index hit is dropped even though the re-score is skipped.
    assert!(
        !ids.contains(&35),
        "stale vector for id=35 must be dropped under fast_search, got {ids:?}"
    );
    // Recall tradeoff: fast_search skips the flat re-score, so the moved-on match is not surfaced.
    assert!(
        !ids.contains(&40),
        "fast_search skips re-score, so id=40 should be absent, got {ids:?}"
    );
}

/// A compound boolean predicate (age AND id) exercises the ScalarIndexExpr tree-walk in
/// `overlay_stale_index_rows`. An overlay on `age` marks fragment 0 stale from the `age`
/// index's perspective, so the compound query must re-evaluate fragment 0 on the flat path.
#[tokio::test]
async fn test_overlay_stale_with_compound_index_expression() {
    let mut dataset = create_base_dataset().await;
    // Build BTree indexes on both columns so a compound filter can use both.
    build_age_index(&mut dataset).await;
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    // Fragment 0 covers id=0..5, age=0..50. Overlay changes id=1's age from 10 to 999.
    let dataset = commit_overlay(
        dataset,
        "age_compound",
        0,
        &[1],
        OverlayCoverage::dense(RoaringBitmap::from_iter([1])),
        vec![i32_array([Some(999)])],
    )
    .await;

    // Compound query: both the `age` and `id` index are involved. The overlay on `age`
    // makes fragment 0 stale for the `age` index; it falls to the flat path, which uses
    // the merged (overlay) value. Result: the stale age=10 hit is gone, age=999 appears.
    assert_eq!(ids_matching(&dataset, "age = 10").await, Vec::<i32>::new());
    assert_eq!(ids_matching(&dataset, "age = 999").await, vec![1]);
    // A pure `id` query on an unaffected fragment still works correctly.
    assert_eq!(ids_matching(&dataset, "id = 2").await, vec![2]);
}

/// Text dataset: two fragments, 6 rows each. Schema: id (Int32), text (Utf8).
/// Texts are unique tokens so each row can be identified by its term.
async fn create_text_dataset() -> Dataset {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, true),
        ArrowField::new("text", DataType::Utf8, true),
    ]));
    let texts: Vec<&str> = vec![
        "apple pie",
        "apple banana", // row 1, fragment 0 — will be overlaid in tests
        "cherry cake",
        "banana split",
        "orange juice",
        "grape vine",
        "mango sorbet", // fragment 1 starts here
        "pear tart",
        "lemon curd",
        "peach cobbler",
        "plum pudding",
        "fig newton",
    ];
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..12)),
            Arc::new(StringArray::from(texts)),
        ],
    )
    .unwrap();
    let write_params = WriteParams {
        max_rows_per_file: 6,
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    Dataset::write(reader, "memory://", Some(write_params))
        .await
        .unwrap()
}

async fn build_text_fts_index(dataset: &mut Dataset) {
    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default(),
            true,
        )
        .await
        .unwrap();
}

/// FTS index with token positions stored, required for phrase queries.
async fn build_text_fts_index_with_positions(dataset: &mut Dataset) {
    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default().with_position(true),
            true,
        )
        .await
        .unwrap();
}

/// Collect sorted IDs of rows returned by an FTS query on `text`.
async fn fts_ids(dataset: &Dataset, query: FullTextSearchQuery) -> Vec<i32> {
    let results = dataset
        .scan()
        .full_text_search(query)
        .unwrap()
        .project(&["id"])
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let mut ids = ids_from_batches(&results);
    ids.sort_unstable();
    ids
}

async fn fts_ids_matching(dataset: &Dataset, term: &str) -> Vec<i32> {
    fts_ids(dataset, FullTextSearchQuery::new(term.to_owned())).await
}

async fn fts_phrase_ids_matching(dataset: &Dataset, phrase: &str) -> Vec<i32> {
    use lance_index::scalar::inverted::query::{FtsQuery, PhraseQuery};

    let query = FullTextSearchQuery::new_query(FtsQuery::Phrase(
        PhraseQuery::new(phrase.to_owned()).with_column(Some("text".to_owned())),
    ));
    fts_ids(dataset, query).await
}

/// An overlay committed after the FTS index is built replaces a row's text. Searching for
/// the old term must not return the stale row; searching for the new term must find it.
#[tokio::test]
async fn test_fts_overlay_stale_drop_and_new_match() {
    let mut dataset = create_text_dataset().await;
    build_text_fts_index(&mut dataset).await;

    // fragment 0, row offset 1 (id=1): "apple banana" → "cherry mango"
    // field ID 1 is the `text` column.
    let dataset = commit_overlay(
        dataset,
        "text_overlay",
        0,
        &[1],
        OverlayCoverage::dense(RoaringBitmap::from_iter([1])),
        vec![Arc::new(StringArray::from(vec![Some("cherry mango")]))],
    )
    .await;

    // "apple" now matches only id=0 ("apple pie"); id=1's stale index entry must be dropped.
    assert_eq!(fts_ids_matching(&dataset, "apple").await, vec![0]);

    // "banana" matched id=1 and id=3 before; after overlay id=1's stale entry must be gone.
    assert_eq!(fts_ids_matching(&dataset, "banana").await, vec![3]);

    // "cherry" now matches id=1 (via flat path on stale fragment) and id=2 ("cherry cake").
    let cherry_ids = fts_ids_matching(&dataset, "cherry").await;
    assert!(
        cherry_ids.contains(&1),
        "id=1 overlay→cherry mango should be found: {cherry_ids:?}"
    );
    assert!(
        cherry_ids.contains(&2),
        "id=2 cherry cake should still be found: {cherry_ids:?}"
    );

    // "mango" now matches id=1 (overlay) and id=6 ("mango sorbet" in fragment 1).
    let mango_ids = fts_ids_matching(&dataset, "mango").await;
    assert!(
        mango_ids.contains(&1),
        "id=1 overlay→cherry mango should be found: {mango_ids:?}"
    );
    assert!(
        mango_ids.contains(&6),
        "id=6 mango sorbet should still be found: {mango_ids:?}"
    );
}

/// A phrase query must not return a stale hit for an overlaid FTS-indexed row. Phrase queries
/// have no flat re-evaluation path, so the fragment is excluded from the indexed phrase search
/// (like an unindexed fragment) rather than re-scored — the point of this test is that the
/// pre-overlay phrase hit is dropped, not that the new value is found.
#[tokio::test]
async fn test_fts_phrase_overlay_stale_drop() {
    let mut dataset = create_text_dataset().await;
    build_text_fts_index_with_positions(&mut dataset).await;

    // Before any overlay the phrase "apple banana" matches only id=1.
    assert_eq!(
        fts_phrase_ids_matching(&dataset, "apple banana").await,
        vec![1]
    );

    // Overlay id=1's text (field 1) so the phrase no longer applies to its current value.
    let dataset = commit_overlay(
        dataset,
        "phrase_overlay",
        0,
        &[1],
        OverlayCoverage::dense(RoaringBitmap::from_iter([1])),
        vec![Arc::new(StringArray::from(vec![Some("cherry mango")]))],
    )
    .await;

    // The stale inverted-index positions for "apple banana" on id=1 must not be returned.
    assert_eq!(
        fts_phrase_ids_matching(&dataset, "apple banana").await,
        Vec::<i32>::new()
    );
}

/// An overlay on a non-FTS field must not exclude the fragment from phrase search.
#[tokio::test]
async fn test_fts_phrase_overlay_unrelated_field_not_excluded() {
    let mut dataset = create_text_dataset().await;
    build_text_fts_index_with_positions(&mut dataset).await;

    // Overlay field 0 (`id`), not the FTS-indexed `text` column: phrase coverage is untouched.
    let dataset = commit_overlay(
        dataset,
        "id_overlay",
        0,
        &[0],
        OverlayCoverage::dense(RoaringBitmap::from_iter([1])),
        vec![i32_array([Some(777)])],
    )
    .await;

    assert_eq!(
        fts_phrase_ids_matching(&dataset, "apple banana").await,
        vec![777]
    );
}

/// An overlay on a field the FTS index does NOT cover must not exclude anything.
#[tokio::test]
async fn test_fts_overlay_unrelated_field_not_excluded() {
    let mut dataset = create_text_dataset().await;
    build_text_fts_index(&mut dataset).await;

    // Overlay field 0 (id) — not covered by the FTS index on `text`.
    let dataset = commit_overlay(
        dataset,
        "id_overlay_for_fts",
        0,
        &[0],
        OverlayCoverage::dense(RoaringBitmap::from_iter([1])),
        vec![i32_array([Some(999)])],
    )
    .await;

    // FTS coverage must be unchanged — both rows containing "apple" are still returned.
    // The `id` overlay changes row offset 1's id from 1 to 999, so the projected id column
    // reflects the overlay even though the FTS index correctly returned that row.
    assert_eq!(fts_ids_matching(&dataset, "apple").await, vec![0, 999]);
    assert_eq!(fts_ids_matching(&dataset, "banana").await, vec![3, 999]);
}

/// Benchmark: measure query latency for BTree, FTS, and vector ANN with 0/4/16 overlay layers.
///
/// Run with: cargo test -p lance --lib --release -- overlay_index_masking::bench --ignored --nocapture
#[tokio::test]
#[ignore = "benchmark"]
#[allow(clippy::print_stdout)]
async fn bench_index_query_overlay_overhead() {
    use std::time::Instant;

    use arrow_array::Float32Array;

    const DIM: i32 = 32;
    const ROWS: i32 = 1_000_000;
    const ROWS_PER_FRAG: i32 = 100_000; // 10 fragments
    const ITERS: u32 = 10; // large scans — 10 is enough for stable averages

    // Fixed disk path so timings are comparable across runs. Deleted and recreated fresh.
    let uri = "/tmp/lance-bench-overlay-oss1325";
    if std::path::Path::new(uri).exists() {
        std::fs::remove_dir_all(uri).unwrap();
    }

    // --- Build 1M-row dataset on local disk --------------------------------
    // Schema: id(0), age(1), vec(2) — 3 top-level fields.
    // Lance field IDs (depth-first): id=0, age=1, vec=2, vec.item=3.

    println!("Building {ROWS}-row dataset at {uri} (this takes ~30 s)...");

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("age", DataType::Int32, false),
        ArrowField::new(
            "vec",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                DIM,
            ),
            false,
        ),
    ]));

    let row_ids: Vec<i32> = (0..ROWS).collect();
    let ages: Vec<i32> = row_ids.iter().map(|&i| i * 10).collect();
    // Build the 128 MB flat float array directly (avoids 1M per-row Vec allocations).
    let flat_vecs: Vec<f32> = (0..(ROWS as usize * DIM as usize))
        .map(|j| (j / DIM as usize) as f32 % 1000.0)
        .collect();
    let vec_col = Arc::new(
        arrow_array::FixedSizeListArray::try_new(
            Arc::new(ArrowField::new("item", DataType::Float32, true)),
            DIM,
            Arc::new(Float32Array::from(flat_vecs)),
            None,
        )
        .unwrap(),
    );

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(row_ids)),
            Arc::new(Int32Array::from(ages)),
            vec_col,
        ],
    )
    .unwrap();

    let write_params = WriteParams {
        max_rows_per_file: ROWS_PER_FRAG as usize,
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let mut dataset = Dataset::write(reader, uri, Some(write_params))
        .await
        .unwrap();

    println!("Building BTree index on age...");
    dataset
        .create_index(
            &["age"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    println!("Building IVF_FLAT(1 partition) index on vec...");
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

    println!("Indexes built.\n");

    // --- Timing helper ---------------------------------------------------

    async fn timeit<F, Fut>(iters: u32, mut f: F) -> f64
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        f().await; // warmup
        let t0 = Instant::now();
        for _ in 0..iters {
            f().await;
        }
        t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
    }

    // === Scenario A: BTree query overhead ================================
    //
    // Overlay on `age` (field 1), covering only offset 0 of fragment 0.
    // Fragment granularity: the entire fragment 0 (100k rows) falls to flat-scan.
    //
    // btree_cold: `age = 420` → id=42 → in fragment 0 (rows 0..99999).
    //   With overlays: 100k-row flat scan + per-overlay merge instead of index lookup.
    //   Without overlays: O(log n) BTree lookup.
    //
    // btree_warm: `age = 1000420` → id=100042 → in fragment 1 (rows 100000..199999).
    //   Always served by the BTree index regardless of overlay count on fragment 0.
    //   This isolates the index-lookup baseline.
    println!("=== Scenario A: BTree (overlay on `age`, fragment 0 becomes stale) ===");
    println!(
        "{:>10}  {:>14}  {:>14}",
        "overlays", "cold_frag0_ms", "warm_frag1_ms"
    );

    let mut committed_a = 0u32;
    for num_overlays in [0u32, 1, 4, 16] {
        // Commit only the delta since the last iteration.
        for layer in committed_a..num_overlays {
            dataset = commit_overlay(
                dataset,
                &format!("age_ol{layer}"),
                0,    // fragment 0
                &[1], // field 1 = age
                OverlayCoverage::dense(RoaringBitmap::from_iter([0u32])),
                vec![i32_array([Some(999)])],
            )
            .await;
        }
        committed_a = num_overlays;

        let ds = Arc::new(dataset.clone());

        // Cold path: stale fragment falls to flat scan when overlays > 0.
        let ds2 = ds.clone();
        let cold_ms = timeit(ITERS, || {
            let ds = ds2.clone();
            async move {
                ds.scan()
                    .filter("age = 420")
                    .unwrap()
                    .project(&["age"])
                    .unwrap()
                    .try_into_batch()
                    .await
                    .unwrap();
            }
        })
        .await;

        // Warm path: fragment 1 never stale, always index-served.
        let ds2 = ds.clone();
        let warm_ms = timeit(ITERS, || {
            let ds = ds2.clone();
            async move {
                ds.scan()
                    .filter("age = 1000420")
                    .unwrap()
                    .project(&["age"])
                    .unwrap()
                    .try_into_batch()
                    .await
                    .unwrap();
            }
        })
        .await;

        println!("{num_overlays:>10}  {cold_ms:>14.1}  {warm_ms:>14.1}");
    }

    // === Scenario B: Vector ANN overhead =================================
    //
    // Overlay on `vec` (field 2), covering only offset 0 of fragment 0.
    // The field-aware check means the 16 age overlays from Scenario A do NOT affect
    // the vector index (they touch field 1, not field 2). Only a vec overlay (field 2)
    // marks fragment 0 stale for the vector index.
    //
    // With a vec overlay: 100k rows of fragment 0 are excluded from ANN prefilter
    // bitmaps and re-scored brute-force (O(100k × DIM) distance computations).
    println!("\n=== Scenario B: Vector ANN (overlay on `vec`, 100k rows brute-forced) ===");
    println!("{:>12}  {:>10}", "vec_overlays", "ann_ms");

    let query_vec = Float32Array::from(vec![0.5f32; DIM as usize]);

    for num_vec_overlays in [0u32, 1] {
        if num_vec_overlays == 1 {
            dataset = commit_overlay(
                dataset,
                "vec_ol0",
                0,    // fragment 0
                &[2], // field 2 = vec (FixedSizeList top-level field)
                OverlayCoverage::dense(RoaringBitmap::from_iter([0u32])),
                vec![fsl(vec![vec![0.0f32; DIM as usize]], DIM)],
            )
            .await;
        }

        let ds = Arc::new(dataset.clone());
        let ds2 = ds.clone();
        let qv = query_vec.clone();
        let ann_ms = timeit(ITERS, || {
            let ds = ds2.clone();
            let q = qv.clone();
            async move {
                ds.scan()
                    .nearest("vec", &q, 10)
                    .unwrap()
                    .minimum_nprobes(1)
                    .project(&["id"])
                    .unwrap()
                    .try_into_batch()
                    .await
                    .unwrap();
            }
        })
        .await;

        println!("{num_vec_overlays:>12}  {ann_ms:>10.1}");
    }
}
