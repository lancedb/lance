// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

// Deterministic top-k tiebreak (score DESC, row_id ASC).
//
// A BM25 top-k that breaks ties arbitrarily is not a stable prefix across `k`:
// top-k1 is not a prefix of top-k2, so paginating a tied-score query (a broad
// term where many documents share `(tf, doc_len)`) duplicates and skips rows.
// Four coupled pieces fix that, all keyed on the row address:
//   1. `ScoredDoc::cmp` is the total order used for the final sort.
//   2. `wand::TopKCollector` evicts on the full key when the walk knows row
//      addresses, and retains the whole k-th-score tie band when it does not.
//   3. `wand::admit_ties_floor` lowers the WAND threshold one ULP so the
//      score-only prune kernels keep, rather than drop, that band, including a
//      slower partition's ties behind the shared cross-partition floor.
//   4. `ModernCandidates::push` keeps the band across partitions and
//      `rank_resolved_documents` re-selects once addresses are resolved.
//
// The tests below cover a single partition, a post-compaction address order that
// diverges from DocId order, several partitions, and a tie band wide enough to
// overflow the collector's buffer and force the address-ordered retry.

struct MaskPreFilter {
    mask: Arc<RowAddrMask>,
}

#[cfg_attr(coverage, coverage(off))]
#[async_trait]
impl PreFilter for MaskPreFilter {
    async fn wait_for_ready(&self) -> Result<()> {
        Ok(())
    }

    fn is_empty(&self) -> bool {
        false
    }

    fn mask(&self) -> Arc<RowAddrMask> {
        self.mask.clone()
    }

    fn filter_row_ids<'a>(&self, row_ids: Box<dyn Iterator<Item = &'a u64> + 'a>) -> Vec<u64> {
        self.mask.selected_indices(row_ids)
    }
}

/// Build one partition whose documents all score identically: term "alpha"
/// once, doc length 1. `row_ids` is indexed by DocId, so passing an unsorted
/// slice models a post-compaction remap.
async fn build_tied_partition(store: &Arc<LanceIndexStore>, partition_id: u64, row_ids: &[u64]) {
    let mut builder = InnerBuilder::new(partition_id, false, TokenSetFormat::default());
    builder.tokens.add("alpha".to_owned());
    builder.posting_lists.push(PostingListBuilder::new(false));
    for (doc_id, &row_id) in row_ids.iter().enumerate() {
        builder.posting_lists[0].add(doc_id as u32, PositionRecorder::Count(1));
        builder.docs.append(row_id, 1);
    }
    builder.write(store.as_ref()).await.unwrap();
}

async fn load_tied_index(partitions: &[(u64, Vec<u64>)]) -> (TempObjDir, Arc<InvertedIndex>) {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    for (partition_id, row_ids) in partitions {
        build_tied_partition(&store, *partition_id, row_ids).await;
    }
    write_test_metadata(
        &store,
        partitions.iter().map(|(id, _)| *id).collect(),
        InvertedIndexParams::default(),
    )
    .await;
    let cache = LanceCache::with_capacity(64 * 1024 * 1024);
    let index = InvertedIndex::load(store, None, &cache).await.unwrap();
    (tmpdir, index)
}

async fn search_alpha_with(
    index: &InvertedIndex,
    limit: usize,
    prefilter: Arc<dyn PreFilter>,
) -> Vec<u64> {
    let (row_ids, scores) = index
        .bm25_search(
            Arc::new(Tokens::new(vec!["alpha".to_owned()], DocType::Text)),
            Arc::new(FtsSearchParams::new().with_limit(Some(limit))),
            Operator::Or,
            prefilter,
            Arc::new(NoOpMetricsCollector),
            None,
        )
        .await
        .unwrap();
    // Every document ties, so the premise of these tests is that the returned
    // scores are all identical; without that the row_id order proves nothing.
    for window in scores.windows(2) {
        assert!(
            (window[0] - window[1]).abs() < 1e-6,
            "premise: scores must be tied, got {scores:?}"
        );
    }
    row_ids
}

async fn search_alpha(index: &InvertedIndex, limit: usize) -> Vec<u64> {
    search_alpha_with(index, limit, Arc::new(NoFilter)).await
}

#[tokio::test]
async fn test_fts_topk_tied_scores_stable_prefix_single_partition() {
    let (_tmpdir, index) =
        load_tied_index(&[(0, vec![100, 101, 102, 103, 104, 105, 106, 107])]).await;

    let top3 = search_alpha(&index, 3).await;
    let top5 = search_alpha(&index, 5).await;
    let top8 = search_alpha(&index, 8).await;
    assert_eq!(
        top3,
        vec![100, 101, 102],
        "top-3 must be the lowest 3 row_ids in order"
    );
    assert_eq!(top5, vec![100, 101, 102, 103, 104]);
    assert_eq!(top8, vec![100, 101, 102, 103, 104, 105, 106, 107]);
    // Stable prefix: a smaller k is an exact ordered prefix of a larger k, so
    // paginating never duplicates or skips a row.
    assert_eq!(top3, top5[..3], "top-3 must be a prefix of top-5");
    assert_eq!(top5, top8[..5], "top-5 must be a prefix of top-8");
}

#[tokio::test]
async fn test_fts_topk_tied_scores_resolve_by_row_id_not_doc_id() {
    // Row addresses no longer ascend with DocId, as after a compaction remap.
    // Postings are still walked in DocId order, so a collector that kept the
    // first documents it saw would return 105, 100, 107 instead of 100, 101, 102.
    let (_tmpdir, index) =
        load_tied_index(&[(0, vec![105, 100, 107, 102, 104, 101, 106, 103])]).await;

    let top3 = search_alpha(&index, 3).await;
    let top5 = search_alpha(&index, 5).await;
    assert_eq!(
        top3,
        vec![100, 101, 102],
        "the tie band must resolve by row_id, not by DocId or encounter order"
    );
    assert_eq!(top5, vec![100, 101, 102, 103, 104]);
    assert_eq!(top3, top5[..3], "top-3 must be a prefix of top-5");
}

#[tokio::test]
async fn test_fts_topk_tied_scores_stable_prefix_across_partitions() {
    // Tied documents split across partitions with interleaved row_ids, so the
    // shared cross-partition WAND floor and the merge must resolve ties by
    // row_id no matter which partition finishes first (`buffer_unordered`).
    let (_tmpdir, index) =
        load_tied_index(&[(0, vec![100, 102, 104, 106]), (1, vec![101, 103, 105, 107])]).await;

    let top3 = search_alpha(&index, 3).await;
    let top6 = search_alpha(&index, 6).await;
    assert_eq!(
        top3,
        vec![100, 101, 102],
        "cross-partition ties must resolve by row_id"
    );
    assert_eq!(top6, vec![100, 101, 102, 103, 104, 105]);
    assert_eq!(top3, top6[..3], "top-3 must be a prefix of top-6");
}

#[tokio::test]
async fn test_fts_topk_tied_scores_are_stable_across_repeated_searches() {
    // Twelve partitions all tied at one score, with row_ids striped across them
    // so the correct top-6 draws one document from each of six partitions. The
    // partitions complete in whatever order `buffer_unordered` yields, so
    // repeating the query checks directly that the result does not depend on it.
    const PARTITIONS: u64 = 12;
    let partitions = (0..PARTITIONS)
        .map(|partition_id| {
            let row_ids = (0..3)
                .map(|offset| partition_id + offset * PARTITIONS)
                .collect::<Vec<_>>();
            (partition_id, row_ids)
        })
        .collect::<Vec<_>>();
    let (_tmpdir, index) = load_tied_index(&partitions).await;

    let expected_top6 = vec![0, 1, 2, 3, 4, 5];
    for _ in 0..8 {
        assert_eq!(search_alpha(&index, 3).await, expected_top6[..3]);
        assert_eq!(search_alpha(&index, 6).await, expected_top6);
    }
}

#[tokio::test]
async fn test_fts_topk_tied_scores_survive_a_prefilter() {
    // An explicit allow-list routes the search off the unfiltered fast path.
    // The partition then loads its addresses to build visibility, so the walk
    // settles the tie itself instead of deferring it to the merge.
    let row_ids = [105u64, 100, 107, 102, 104, 101, 106, 103];
    let (_tmpdir, index) = load_tied_index(&[(0, row_ids.to_vec())]).await;
    let allow_all = || -> Arc<dyn PreFilter> {
        Arc::new(MaskPreFilter {
            mask: Arc::new(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(
                row_ids.iter().copied(),
            ))),
        })
    };

    let top3 = search_alpha_with(&index, 3, allow_all()).await;
    let top5 = search_alpha_with(&index, 5, allow_all()).await;
    assert_eq!(
        top3,
        vec![100, 101, 102],
        "the filtered path must resolve ties by row_id too"
    );
    assert_eq!(top5, vec![100, 101, 102, 103, 104]);
    assert_eq!(top3, top5[..3], "top-3 must be a prefix of top-5");
}

#[tokio::test]
async fn test_fts_topk_tied_scores_wider_than_the_collector_buffer() {
    // The tie band is far wider than the collector's buffer, and row addresses
    // descend with DocId, so the winners sit at the very end of the walk. A
    // collector that truncated its band would answer with the highest row_ids;
    // the overflow retry reloads the partition's addresses and orders exactly.
    const NUM_DOCS: u64 = 512;
    const BASE: u64 = 10_000;
    let row_ids = (0..NUM_DOCS)
        .map(|doc_id| BASE + NUM_DOCS - doc_id)
        .collect::<Vec<_>>();
    let (_tmpdir, index) = load_tied_index(&[(0, row_ids)]).await;

    let top3 = search_alpha(&index, 3).await;
    let top5 = search_alpha(&index, 5).await;
    assert_eq!(
        top3,
        vec![BASE + 1, BASE + 2, BASE + 3],
        "the exact lowest row_ids, which the walk only reaches at its last DocIds"
    );
    assert_eq!(top5, vec![BASE + 1, BASE + 2, BASE + 3, BASE + 4, BASE + 5]);
    assert_eq!(top3, top5[..3], "top-3 must be a prefix of top-5");
}
