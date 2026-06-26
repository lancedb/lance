// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Deref;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock};
use std::{
    cell::UnsafeCell,
    collections::{BinaryHeap, VecDeque},
};
use std::{cmp::Reverse, fmt::Debug};

use arrow::array::AsArray;
use arrow::datatypes::Int32Type;
use arrow_array::Array;
use itertools::Itertools;
use lance_core::Result;
use lance_core::utils::address::RowAddress;
use lance_select::RowAddrMask;

use crate::metrics::MetricsCollector;

use super::{
    CompressedPositionStorage,
    query::Operator,
    scorer::{K1, idf},
};
use super::{
    CompressedPostingList, DocSet, PostingList, RawDocInfo,
    builder::ScoredDoc,
    encoding::{
        decode_position_stream_block, decompress_positions, decompress_posting_block,
        decompress_posting_remainder,
    },
    query::FtsSearchParams,
    scorer::Scorer,
};
use super::{DocInfo, builder::BLOCK_SIZE};

const TERMINATED_DOC_ID: u64 = u64::MAX;
pub static FLAT_SEARCH_PERCENT_THRESHOLD: LazyLock<u64> = LazyLock::new(|| {
    std::env::var("LANCE_FLAT_SEARCH_PERCENT_THRESHOLD")
        .unwrap_or_else(|_| "10".to_string())
        .parse::<u64>()
        .unwrap_or(10)
});

pub struct PostingIterator {
    token: String,
    token_id: u32,
    position: u32,
    query_weight: f32,
    list: PostingList,
    // the index of current doc, this can be changed only by `next()`
    index: usize,
    // the index of current block, this can be changed by `next() and shallow_next()`
    block_idx: usize,
    approximate_upper_bound: f32,

    // for compressed posting list
    compressed: Option<UnsafeCell<CompressedState>>,
}

#[derive(Clone)]
struct CompressedState {
    block_idx: usize,
    doc_ids: Vec<u32>,
    freqs: Vec<u32>,
    buffer: Box<[u32; BLOCK_SIZE]>,
    position_block_idx: Option<usize>,
    position_values: Vec<u32>,
    position_offsets: Vec<usize>,
    block_max_window: BlockMaxWindow,
}

impl CompressedState {
    fn new() -> Self {
        Self {
            block_idx: 0,
            doc_ids: Vec::with_capacity(BLOCK_SIZE),
            freqs: Vec::with_capacity(BLOCK_SIZE),
            buffer: Box::new([0; BLOCK_SIZE]),
            position_block_idx: None,
            position_values: Vec::new(),
            position_offsets: Vec::new(),
            block_max_window: BlockMaxWindow::new(),
        }
    }

    #[inline]
    fn decompress(
        &mut self,
        block: &[u8],
        block_idx: usize,
        num_blocks: usize,
        length: u32,
        tail_codec: super::PostingTailCodec,
    ) {
        self.doc_ids.clear();
        self.freqs.clear();

        let remainder = length as usize % BLOCK_SIZE;
        if block_idx + 1 == num_blocks && remainder != 0 {
            decompress_posting_remainder(
                block,
                remainder,
                tail_codec,
                &mut self.doc_ids,
                &mut self.freqs,
            );
        } else {
            decompress_posting_block(block, &mut self.buffer, &mut self.doc_ids, &mut self.freqs);
        }
        self.block_idx = block_idx;
        self.position_block_idx = None;
        self.position_values.clear();
        self.position_offsets.clear();
    }
}

#[derive(Clone)]
struct BlockMaxWindow {
    // Sliding block range used for Lucene-style getMaxScore(upTo). The deque is
    // monotonic by score and covers blocks in [start_block_idx, next_block_idx).
    start_block_idx: usize,
    next_block_idx: usize,
    max_scores: VecDeque<(usize, f32)>,
}

struct BlockMaxScore {
    score: f32,
    blocks_scanned: usize,
}

impl BlockMaxWindow {
    fn new() -> Self {
        Self {
            start_block_idx: 0,
            next_block_idx: 0,
            max_scores: VecDeque::new(),
        }
    }

    fn reset(&mut self, start_block_idx: usize) {
        self.start_block_idx = start_block_idx;
        self.next_block_idx = start_block_idx;
        self.max_scores.clear();
    }

    fn max_score_up_to(
        &mut self,
        list: &CompressedPostingList,
        start_block_idx: usize,
        up_to: u64,
    ) -> BlockMaxScore {
        if start_block_idx >= list.blocks.len() {
            self.reset(start_block_idx);
            return BlockMaxScore {
                score: 0.0,
                blocks_scanned: 0,
            };
        }
        if start_block_idx < self.start_block_idx || start_block_idx > self.next_block_idx {
            self.reset(start_block_idx);
        }
        self.start_block_idx = start_block_idx;
        while matches!(self.max_scores.front(), Some((block_idx, _)) if *block_idx < start_block_idx)
        {
            self.max_scores.pop_front();
        }

        if list.block_least_doc_id(start_block_idx) as u64 > up_to {
            self.reset(start_block_idx);
            return BlockMaxScore {
                score: 0.0,
                blocks_scanned: 0,
            };
        }

        self.next_block_idx = self.next_block_idx.max(start_block_idx);
        let mut blocks_scanned = 0;
        while self.next_block_idx < list.blocks.len()
            && list.block_least_doc_id(self.next_block_idx) as u64 <= up_to
        {
            let score = list.block_max_score(self.next_block_idx);
            while matches!(self.max_scores.back(), Some((_, old_score)) if *old_score <= score) {
                self.max_scores.pop_back();
            }
            self.max_scores.push_back((self.next_block_idx, score));
            self.next_block_idx += 1;
            blocks_scanned += 1;
        }

        let score = self
            .max_scores
            .front()
            .map(|(_, score)| *score)
            .unwrap_or(0.0);
        BlockMaxScore {
            score,
            blocks_scanned,
        }
    }
}

impl Debug for PostingIterator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostingIterator")
            .field(
                "doc",
                &self
                    .doc()
                    .map(|doc| doc.doc_id())
                    .unwrap_or(TERMINATED_DOC_ID),
            )
            .field("approximate_upper_bound", &self.approximate_upper_bound)
            .field("token_id", &self.token_id)
            .finish()
    }
}

impl PartialEq for PostingIterator {
    fn eq(&self, other: &Self) -> bool {
        self.token_id == other.token_id && self.position == other.position
    }
}

impl Eq for PostingIterator {}

impl PartialOrd for PostingIterator {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PostingIterator {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.doc(), other.doc()) {
            (Some(doc1), Some(doc2)) => doc1
                .cmp(&doc2)
                .then(
                    self.approximate_upper_bound
                        .total_cmp(&other.approximate_upper_bound),
                )
                .then(self.token_id.cmp(&other.token_id))
                .then(self.position.cmp(&other.position)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => self
                .approximate_upper_bound
                .total_cmp(&other.approximate_upper_bound)
                .then(self.token_id.cmp(&other.token_id))
                .then(self.position.cmp(&other.position)),
        }
    }
}

impl PostingIterator {
    #[inline]
    fn compressed_state_ptr(&self) -> *mut CompressedState {
        debug_assert!(self.compressed.is_some());
        // this method is called very frequently, so we prefer to use `UnsafeCell` instead of
        // `RefCell` to avoid the overhead of runtime borrow checking
        self.compressed.as_ref().unwrap().get()
    }

    #[inline]
    fn ensure_compressed_block_ptr(
        &self,
        list: &CompressedPostingList,
        block_idx: usize,
    ) -> *mut CompressedState {
        let compressed = unsafe { &mut *self.compressed_state_ptr() };
        if compressed.block_idx != block_idx || compressed.doc_ids.is_empty() {
            let block = list.blocks.value(block_idx);
            compressed.decompress(
                block,
                block_idx,
                list.blocks.len(),
                list.length,
                list.posting_tail_codec,
            );
        }
        compressed as *mut CompressedState
    }

    #[cfg(test)]
    pub(crate) fn new(
        token: String,
        token_id: u32,
        position: u32,
        list: PostingList,
        num_doc: usize,
    ) -> Self {
        Self::with_query_weight(token, token_id, position, 1.0, list, num_doc)
    }

    pub(crate) fn with_query_weight(
        token: String,
        token_id: u32,
        position: u32,
        query_weight: f32,
        list: PostingList,
        num_doc: usize,
    ) -> Self {
        let approximate_upper_bound = match list.max_score() {
            Some(max_score) => max_score,
            None => idf(list.len(), num_doc) * (K1 + 1.0),
        };

        let is_compressed = matches!(list, PostingList::Compressed(_));

        Self {
            token,
            token_id,
            position,
            query_weight,
            list,
            index: 0,
            block_idx: 0,
            approximate_upper_bound,
            compressed: is_compressed.then(|| UnsafeCell::new(CompressedState::new())),
        }
    }

    #[inline]
    pub(crate) fn term_index(&self) -> u32 {
        self.position
    }

    #[inline]
    pub(crate) fn token(&self) -> &str {
        &self.token
    }

    #[inline]
    fn approximate_upper_bound(&self) -> f32 {
        self.approximate_upper_bound
    }

    #[inline]
    fn score<S: Scorer>(&self, scorer: &S, freq: u32, doc_length: u32) -> f32 {
        self.query_weight * scorer.doc_weight(freq, doc_length)
    }

    #[inline]
    fn cost(&self) -> usize {
        self.list.len()
    }

    #[inline]
    fn empty(&self) -> bool {
        self.index >= self.list.len()
    }

    #[inline]
    fn doc(&self) -> Option<DocInfo> {
        if self.empty() {
            return None;
        }

        match self.list {
            PostingList::Compressed(ref list) => {
                let block_idx = self.index / BLOCK_SIZE;
                let block_offset = self.index % BLOCK_SIZE;
                let compressed = unsafe { &mut *self.ensure_compressed_block_ptr(list, block_idx) };

                // Read from the decompressed block
                let doc_id = compressed.doc_ids[block_offset];
                let frequency = compressed.freqs[block_offset];
                let doc = DocInfo::Raw(RawDocInfo { doc_id, frequency });
                Some(doc)
            }
            PostingList::Plain(ref list) => Some(DocInfo::Located(list.doc(self.index))),
        }
    }

    fn position_cursor(&self) -> Option<PositionCursor<'_>> {
        match self.list {
            PostingList::Plain(ref list) => list.positions.as_ref().map(|positions| {
                let start = positions.value_offsets()[self.index] as usize;
                let end = positions.value_offsets()[self.index + 1] as usize;
                PositionCursor::new(
                    PositionValues::Owned(
                        positions.values().as_primitive::<Int32Type>().values()[start..end]
                            .iter()
                            .map(|value| *value as u32)
                            .collect(),
                    ),
                    self.position as i32,
                )
            }),
            PostingList::Compressed(ref list) => match list.positions.as_ref()? {
                CompressedPositionStorage::LegacyPerDoc(positions) => {
                    let positions = positions.value(self.index);
                    let positions = decompress_positions(positions.as_binary());
                    Some(PositionCursor::new(
                        PositionValues::Owned(positions),
                        self.position as i32,
                    ))
                }
                CompressedPositionStorage::SharedStream(stream) => {
                    let block_idx = self.index / BLOCK_SIZE;
                    let block_offset = self.index % BLOCK_SIZE;
                    let compressed =
                        unsafe { &mut *self.ensure_compressed_block_ptr(list, block_idx) };
                    if compressed.position_block_idx != Some(block_idx) {
                        decode_position_stream_block(
                            stream.block(block_idx),
                            compressed.freqs.as_slice(),
                            stream.codec(),
                            &mut compressed.position_values,
                        )
                        .expect("shared position stream decoding should succeed");
                        compressed.position_offsets.clear();
                        compressed
                            .position_offsets
                            .reserve(compressed.freqs.len() + 1);
                        compressed.position_offsets.push(0);
                        let mut offset = 0usize;
                        for &freq in &compressed.freqs {
                            offset += freq as usize;
                            compressed.position_offsets.push(offset);
                        }
                        compressed.position_block_idx = Some(block_idx);
                    }
                    let start = compressed.position_offsets[block_offset];
                    let end = compressed.position_offsets[block_offset + 1];
                    Some(PositionCursor::new(
                        PositionValues::Borrowed(&compressed.position_values[start..end]),
                        self.position as i32,
                    ))
                }
            },
        }
    }

    // move to the next doc id that is greater than or equal to least_id
    fn next(&mut self, least_id: u64) {
        match self.list {
            PostingList::Compressed(ref list) => {
                debug_assert!(least_id <= u32::MAX as u64);
                let least_id = least_id as u32;
                let mut block_idx = self.index / BLOCK_SIZE;
                while block_idx + 1 < list.blocks.len()
                    && list.block_least_doc_id(block_idx + 1) <= least_id
                {
                    block_idx += 1;
                }
                self.index = self.index.max(block_idx * BLOCK_SIZE);
                let length = list.length as usize;
                while self.index < length {
                    let block_idx = self.index / BLOCK_SIZE;
                    let block_offset = self.index % BLOCK_SIZE;
                    let compressed =
                        unsafe { &mut *self.ensure_compressed_block_ptr(list, block_idx) };
                    let in_block = &compressed.doc_ids[block_offset..];
                    let offset_in_block = in_block.partition_point(|&doc_id| doc_id < least_id);
                    let new_offset = block_offset + offset_in_block;
                    if new_offset < compressed.doc_ids.len() {
                        self.index = block_idx * BLOCK_SIZE + new_offset;
                        break;
                    }
                    if block_idx + 1 >= list.blocks.len() {
                        self.index = length;
                        break;
                    }
                    self.index = (block_idx + 1) * BLOCK_SIZE;
                }
                self.block_idx = self.index / BLOCK_SIZE;
            }
            PostingList::Plain(ref list) => {
                self.index += list.row_ids[self.index..].partition_point(|&id| id < least_id);
            }
        }
    }

    fn shallow_next(&mut self, least_id: u64) {
        match self.list {
            PostingList::Compressed(ref list) => {
                debug_assert!(least_id <= u32::MAX as u64);
                let least_id = least_id as u32;
                while self.block_idx + 1 < list.blocks.len()
                    && list.block_least_doc_id(self.block_idx + 1) <= least_id
                {
                    self.block_idx += 1;
                }
            }
            PostingList::Plain(_) => {
                // we don't have block max score for legacy index,
                // and no compression, so just do nothing
            }
        }
    }

    #[inline]
    fn block_max_score(&self) -> f32 {
        match self.list {
            PostingList::Compressed(ref list) => list.block_max_score(self.block_idx),
            PostingList::Plain(_) => self.approximate_upper_bound,
        }
    }

    #[inline]
    fn block_max_score_up_to_with_stats(&mut self, up_to: u64) -> BlockMaxScore {
        match self.list {
            PostingList::Compressed(ref list) => {
                let compressed = unsafe { &mut *self.compressed_state_ptr() };
                compressed
                    .block_max_window
                    .max_score_up_to(list, self.block_idx, up_to)
            }
            PostingList::Plain(_) => BlockMaxScore {
                score: self.approximate_upper_bound,
                blocks_scanned: 0,
            },
        }
    }

    #[inline]
    fn is_compressed(&self) -> bool {
        matches!(self.list, PostingList::Compressed(_))
    }

    #[inline]
    fn has_next_compressed_block(&self) -> bool {
        match self.list {
            PostingList::Compressed(ref list) => self.block_idx + 1 < list.blocks.len(),
            PostingList::Plain(_) => false,
        }
    }

    fn block_first_doc(&self) -> Option<u64> {
        match self.list {
            PostingList::Compressed(ref list) => {
                Some(list.block_least_doc_id(self.block_idx) as u64)
            }
            PostingList::Plain(ref plain) => plain.row_ids.get(self.index).cloned(),
        }
    }

    #[inline]
    fn next_block_first_doc(&self) -> Option<u64> {
        match self.list {
            PostingList::Compressed(ref list) => {
                if self.block_idx + 1 >= list.blocks.len() {
                    return None;
                }
                Some(list.block_least_doc_id(self.block_idx + 1) as u64)
            }
            PostingList::Plain(ref plain) => plain.row_ids.get(self.index + 1).cloned(),
        }
    }
}

/// How wand identified a candidate: either it already had the real
/// row_id (DocSet carried row_ids), or only the partition-local
/// doc_id (deferred-row_id path; the caller must resolve via
/// [`super::lazy_docset::LazyDocSet::resolve_row_ids`]).
#[derive(Debug, Clone, Copy)]
pub enum CandidateAddr {
    RowId(u64),
    Pending(u32),
}

#[derive(Debug)]
pub struct DocCandidate {
    pub addr: CandidateAddr,
    /// The document key used by the posting lists: doc_id for compressed
    /// postings, row_id for legacy plain postings.
    pub posting_doc_id: u64,
    /// (term_index, freq)
    pub freqs: Vec<(u32, u32)>,
    pub doc_length: u32,
}

struct HeadPosting {
    // Iterators that are already positioned on or after the next candidate doc.
    // The heap is ordered by smallest doc id so the top element determines
    // the next target doc to consider.
    doc_id: u64,
    posting: Box<PostingIterator>,
}

impl HeadPosting {
    fn new(posting: Box<PostingIterator>) -> Self {
        let doc_id = posting
            .doc()
            .map(|doc| doc.doc_id())
            .unwrap_or(TERMINATED_DOC_ID);
        Self { doc_id, posting }
    }

    fn doc_id(&self) -> u64 {
        self.doc_id
    }
}

impl PartialEq for HeadPosting {
    fn eq(&self, other: &Self) -> bool {
        self.doc_id == other.doc_id
            && self.posting.approximate_upper_bound().to_bits()
                == other.posting.approximate_upper_bound().to_bits()
            && self.posting.token_id == other.posting.token_id
            && self.posting.position == other.posting.position
    }
}

impl Eq for HeadPosting {}

impl PartialOrd for HeadPosting {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeadPosting {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .doc_id
            .cmp(&self.doc_id)
            .then_with(|| {
                self.posting
                    .approximate_upper_bound()
                    .total_cmp(&other.posting.approximate_upper_bound())
            })
            .then_with(|| self.posting.token_id.cmp(&other.posting.token_id))
            .then_with(|| self.posting.position.cmp(&other.posting.position))
    }
}

struct TailPosting {
    // Iterators that lag behind the current target doc but may still help the
    // target beat the threshold if advanced to that doc.
    upper_bound: f32,
    // Used as a tie-breaker when upper bounds are equal. Lower-cost iterators
    // are cheaper to advance, so they are preferred.
    cost: usize,
    posting: Box<PostingIterator>,
}

impl TailPosting {
    fn new(upper_bound: f32, cost: usize, posting: Box<PostingIterator>) -> Self {
        Self {
            upper_bound,
            cost,
            posting,
        }
    }
}

impl PartialEq for TailPosting {
    fn eq(&self, other: &Self) -> bool {
        self.upper_bound.to_bits() == other.upper_bound.to_bits()
            && self.cost == other.cost
            && self.posting.token_id == other.posting.token_id
            && self.posting.position == other.posting.position
    }
}

#[derive(Default)]
struct AndWindowStats {
    windows_wide: usize,
    windows_narrow: usize,
    windows_skipped: usize,
    range_blocks_scanned: usize,
    candidates_returned: usize,
}

#[derive(Default)]
struct AndSearchStats {
    pruned_before_return_start: usize,
    candidates_seen: usize,
    full_scores: usize,
    freqs_collected: usize,
}

impl Eq for TailPosting {}

impl PartialOrd for TailPosting {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TailPosting {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.upper_bound
            .total_cmp(&other.upper_bound)
            .then_with(|| other.cost.cmp(&self.cost))
            .then_with(|| other.posting.token_id.cmp(&self.posting.token_id))
            .then_with(|| other.posting.position.cmp(&self.posting.position))
    }
}

pub struct Wand<'a, S: Scorer> {
    threshold: f32, // multiple of factor and the minimum score of the top-k documents
    operator: Operator,
    num_terms: usize,
    // Posting iterators whose current doc id is >= the next target doc.
    // The heap top gives the smallest current doc id.
    head: BinaryHeap<HeadPosting>,
    #[allow(clippy::vec_box)]
    // Posting iterators that already match the current target doc.
    // Only these iterators participate in scoring / phrase checks for the
    // current candidate.
    lead: Vec<Box<PostingIterator>>,
    // Posting iterators that are behind the current target doc but still kept
    // in play because their score upper bound could affect the decision for the
    // current candidate.
    tail: BinaryHeap<TailPosting>,
    // Sum of upper bounds for all iterators currently held in `tail`.
    // This lets us cheaply decide whether the current candidate can still beat
    // the threshold before fully advancing every lagging iterator.
    tail_max_score: f32,
    // Block-max scores are valid for all candidate docs up to this doc id.
    // `None` means the window has not been initialized yet and the next
    // candidate must refresh block-max state before making pruning decisions.
    up_to: Option<u64>,
    // For conjunctions, this is the maximum attainable score for the current
    // block-max window `[target, up_to]`.
    and_max_score: f32,
    // Last conjunction doc returned to the caller. The next conjunction search
    // resumes strictly after this doc, like Lucene's `nextDoc()/advance()`.
    and_last_doc: Option<u64>,
    and_window_stats: AndWindowStats,
    and_candidates_pruned_before_return: usize,
    docs: &'a DocSet,
    scorer: S,
    // Shared cross-partition top-k floor. Each partition publishes its local
    // k-th score (`atomic_store_max_f32`) and prunes against the running value
    // -- a lower bound on the global k-th, so it never drops a real top-k doc.
    shared_threshold: Option<Arc<AtomicU32>>,
}

/// Monotonically raise an f32 stored in an `AtomicU32` to `val`. CAS loop (not a
/// bit-max) so it stays correct for negative scores -- BM25 idf can go negative.
fn atomic_store_max_f32(slot: &AtomicU32, val: f32) {
    let mut cur = slot.load(Ordering::Relaxed);
    while val > f32::from_bits(cur) {
        match slot.compare_exchange_weak(cur, val.to_bits(), Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => cur = actual,
        }
    }
}

// we were using row id as doc id in the past, which is u64,
// but now we are using the index as doc id, which is u32.
// so here WAND is a generic struct that can be used for both u32 and u64 doc ids.
impl<'a, S: Scorer> Wand<'a, S> {
    pub(crate) fn new(
        operator: Operator,
        postings: impl Iterator<Item = PostingIterator>,
        docs: &'a DocSet,
        scorer: S,
    ) -> Self {
        let mut head = BinaryHeap::new();
        let mut lead = Vec::new();
        for posting in postings {
            if posting.doc().is_none() {
                continue;
            }
            let posting = Box::new(posting);
            if operator == Operator::And {
                lead.push(posting);
            } else {
                head.push(HeadPosting::new(posting));
            }
        }
        if operator == Operator::And {
            lead.sort_unstable_by_key(|posting| posting.cost());
        }

        Self {
            threshold: 0.0,
            operator,
            num_terms: if operator == Operator::And {
                lead.len()
            } else {
                head.len()
            },
            head,
            lead,
            tail: BinaryHeap::new(),
            tail_max_score: 0.0,
            up_to: None,
            and_max_score: f32::INFINITY,
            and_last_doc: None,
            and_window_stats: AndWindowStats::default(),
            and_candidates_pruned_before_return: 0,
            docs,
            scorer,
            shared_threshold: None,
        }
    }

    /// Share one cross-partition top-k floor across a query's partitions.
    pub(crate) fn with_shared_threshold(mut self, shared: Arc<AtomicU32>) -> Self {
        self.shared_threshold = Some(shared);
        self
    }

    /// Set the pruning threshold from this partition's k-th best, raised to the
    /// shared cross-partition floor when one is attached.
    fn update_threshold(&mut self, local_kth: f32, wand_factor: f32) {
        let mut t = local_kth * wand_factor;
        if let Some(shared) = self.shared_threshold.as_ref() {
            atomic_store_max_f32(shared, local_kth);
            let g = f32::from_bits(shared.load(Ordering::Relaxed)) * wand_factor;
            if g > t {
                t = g;
            }
        }
        self.threshold = t;
    }

    /// Raise the local threshold to the shared cross-partition floor, picking up
    /// updates published by sibling partitions.
    fn raise_to_shared_floor(&mut self, wand_factor: f32) {
        if let Some(shared) = self.shared_threshold.as_ref() {
            let g = f32::from_bits(shared.load(Ordering::Relaxed)) * wand_factor;
            if g > self.threshold {
                self.threshold = g;
            }
        }
    }

    // search the top-k documents that contain the query
    // returns the row_id, frequency and doc length
    pub(crate) fn search(
        &mut self,
        params: &FtsSearchParams,
        mask: Arc<RowAddrMask>,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<DocCandidate>> {
        let limit = params.limit.unwrap_or(usize::MAX);
        if limit == 0 {
            return Ok(vec![]);
        }

        match (mask.max_len(), mask.iter_addrs()) {
            (Some(num_rows_matched), Some(row_ids))
                if self.operator == Operator::Or
                    && num_rows_matched * 100
                        <= FLAT_SEARCH_PERCENT_THRESHOLD.deref() * self.docs.len() as u64 =>
            {
                return self.flat_search(params, row_ids, metrics);
            }
            _ => {}
        }

        // Deferred-row_id path: when the DocSet was built without
        // row_ids, wand emits candidates carrying just the
        // partition-local doc_id; the outer caller resolves them to
        // row_ids post-wand.
        let docs_has_row_ids = self.docs.has_row_ids();

        let mut candidates = BinaryHeap::with_capacity(std::cmp::min(limit, BLOCK_SIZE * 10));
        let mut num_comparisons = 0;
        let mut and_search_stats = (self.operator == Operator::And).then_some(AndSearchStats {
            pruned_before_return_start: self.and_candidates_pruned_before_return,
            ..Default::default()
        });
        loop {
            self.raise_to_shared_floor(params.wand_factor);
            let Some((doc, mut score)) = self.next()? else {
                break;
            };
            num_comparisons += 1;
            if let Some(and_stats) = and_search_stats.as_mut() {
                and_stats.candidates_seen += 1;
            }

            // Either a real row_id (so we can run the mask check
            // inline) or the doc_id widened to u64 (deferred path;
            // the outer caller will resolve it post-wand).
            let posting_doc_id = doc.doc_id();
            let row_id = match &doc {
                DocInfo::Raw(doc) => {
                    if docs_has_row_ids {
                        self.docs.row_id(doc.doc_id)
                    } else {
                        doc.doc_id as u64
                    }
                }
                DocInfo::Located(doc) => doc.row_id,
            };
            // Skip docs the fragment-reuse remap deleted. They are tombstoned
            // in the DocSet (slot kept so posting-list doc_ids stay aligned)
            // and must not surface in results.
            if docs_has_row_ids && row_id == RowAddress::TOMBSTONE_ROW {
                if self.operator == Operator::Or {
                    self.push_back_leads(doc.doc_id() + 1);
                }
                continue;
            }
            if docs_has_row_ids && !mask.selected(row_id) {
                if self.operator == Operator::Or {
                    self.push_back_leads(doc.doc_id() + 1);
                }
                continue;
            }

            let doc_length = match &doc {
                DocInfo::Raw(doc) => self.docs.num_tokens(doc.doc_id),
                DocInfo::Located(doc) => self.docs.num_tokens_by_row_id(doc.row_id),
            };

            let score = if self.operator == Operator::Or {
                self.advance_all_tail(doc.doc_id(), Some(doc_length), Some(&mut score));
                if params.phrase_slop.is_some()
                    && !self.check_positions(params.phrase_slop.unwrap() as i32)
                {
                    self.push_back_leads(doc.doc_id() + 1);
                    continue;
                }
                score
            } else {
                self.advance_all_tail(doc.doc_id(), None, None);
                if params.phrase_slop.is_some()
                    && !self.check_positions(params.phrase_slop.unwrap() as i32)
                {
                    continue;
                }
                if let Some(and_stats) = and_search_stats.as_mut() {
                    and_stats.full_scores += 1;
                }
                self.score(doc_length)
            };

            if candidates.len() < limit {
                let freqs = self.iter_term_freqs().collect();
                if let Some(and_stats) = and_search_stats.as_mut() {
                    and_stats.freqs_collected += 1;
                }
                candidates.push(Reverse((
                    ScoredDoc::new(row_id, score),
                    freqs,
                    doc_length,
                    posting_doc_id,
                )));
                if candidates.len() == limit {
                    let kth = candidates.peek().unwrap().0.0.score.0;
                    self.update_threshold(kth, params.wand_factor);
                }
            } else if score > candidates.peek().unwrap().0.0.score.0 {
                let freqs = self.iter_term_freqs().collect();
                if let Some(and_stats) = and_search_stats.as_mut() {
                    and_stats.freqs_collected += 1;
                }
                candidates.pop();
                candidates.push(Reverse((
                    ScoredDoc::new(row_id, score),
                    freqs,
                    doc_length,
                    posting_doc_id,
                )));
                let kth = candidates.peek().unwrap().0.0.score.0;
                self.update_threshold(kth, params.wand_factor);
            }
            if self.operator == Operator::Or {
                self.push_back_leads(doc.doc_id() + 1);
            }
        }
        if self.operator == Operator::And {
            tracing::debug!(
                and_windows_wide = self.and_window_stats.windows_wide,
                and_windows_narrow = self.and_window_stats.windows_narrow,
                and_windows_skipped = self.and_window_stats.windows_skipped,
                and_range_blocks_scanned = self.and_window_stats.range_blocks_scanned,
                and_candidates_returned = self.and_window_stats.candidates_returned,
                "fts conjunction block-max window stats"
            );
        }
        metrics.record_comparisons(num_comparisons);
        if let Some(and_stats) = and_search_stats {
            let and_candidates_pruned_before_return = self
                .and_candidates_pruned_before_return
                .saturating_sub(and_stats.pruned_before_return_start);
            metrics.record_and_candidates_seen(and_stats.candidates_seen);
            metrics.record_and_candidates_pruned_before_return(and_candidates_pruned_before_return);
            metrics.record_and_full_scores(and_stats.full_scores);
            metrics.record_freqs_collected(and_stats.freqs_collected);
        }

        // The heap entry's `row_id` slot is either a real row_id
        // (DocSet had row_ids) or the doc_id widened to u64
        // (deferred). Tag it accordingly so the caller can match
        // rather than guess.
        let to_addr = |row_id_slot: u64| {
            if docs_has_row_ids {
                CandidateAddr::RowId(row_id_slot)
            } else {
                CandidateAddr::Pending(row_id_slot as u32)
            }
        };
        Ok(candidates
            .into_iter()
            .map(
                |Reverse((doc, freqs, doc_length, posting_doc_id))| DocCandidate {
                    addr: to_addr(doc.row_id),
                    posting_doc_id,
                    freqs,
                    doc_length,
                },
            )
            .collect())
    }

    fn flat_search(
        &mut self,
        params: &FtsSearchParams,
        row_ids: Box<dyn Iterator<Item = RowAddress> + '_>,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<DocCandidate>> {
        let limit = params.limit.unwrap_or(usize::MAX);
        if limit == 0 {
            return Ok(vec![]);
        }

        // we need to map the row ids to doc ids, and sort them,
        // because WAND PostingIterator can't go back to the previous doc id.
        // A list column maps one row id to several doc ids, so expand every
        // document the row owns — keying on a single doc id would drop matches
        // at non-last list positions (lancedb#3352).
        let doc_ids = row_ids
            .flat_map(|row_addr| {
                let row_id: u64 = row_addr.into();
                self.docs
                    .doc_ids(row_id)
                    .map(move |doc_id| (doc_id, row_id))
            })
            .sorted_unstable()
            .collect::<Vec<_>>();
        let is_compressed = self
            .head
            .peek()
            .map(|posting| matches!(posting.posting.list, PostingList::Compressed(_)))
            .or_else(|| {
                self.lead
                    .first()
                    .map(|posting| matches!(posting.list, PostingList::Compressed(_)))
            })
            .unwrap_or(false);

        let mut num_comparisons = 0;
        let mut candidates = BinaryHeap::new();
        for (doc_id, row_id) in doc_ids {
            num_comparisons += 1;
            self.move_head_before_target_to_tail(doc_id);
            self.move_head_doc_to_lead(doc_id);

            if self.lead.is_empty() && self.tail.is_empty() {
                continue;
            }

            if !self.can_target_beat_threshold(doc_id) {
                self.advance_tail_and_lead_to_head(doc_id + 1);
                continue;
            }

            self.collect_tail_matches(doc_id);

            if self.operator == Operator::And && self.lead.len() < self.num_terms {
                self.advance_lead_to_head(doc_id + 1);
                continue;
            }

            // check positions
            if params.phrase_slop.is_some()
                && !self.check_positions(params.phrase_slop.unwrap() as i32)
            {
                self.advance_lead_to_head(doc_id + 1);
                continue;
            }

            // score the doc
            let doc_length = match is_compressed {
                true => self.docs.num_tokens(doc_id as u32),
                false => self.docs.num_tokens_by_row_id(row_id),
            };
            if self.operator == Operator::Or && !self.refine_or_candidate(doc_id, doc_length) {
                // `flat_search` evaluates an explicit allow-list of doc ids. Unlike the
                // regular WAND path, skipping to the next block boundary is unsafe here
                // because later doc ids from the same block may still be present in the
                // allow-list and need to be evaluated individually.
                self.advance_tail_and_lead_to_head(doc_id + 1);
                continue;
            }

            self.collect_tail_matches(doc_id);
            let score = self.score(doc_length);

            if candidates.len() < limit {
                let freqs = self.iter_term_freqs().collect();
                candidates.push(Reverse((
                    ScoredDoc::new(row_id, score),
                    freqs,
                    doc_length,
                    doc_id,
                )));
                if candidates.len() == limit {
                    let kth = candidates.peek().unwrap().0.0.score.0;
                    self.update_threshold(kth, params.wand_factor);
                }
            } else if score > candidates.peek().unwrap().0.0.score.0 {
                let freqs = self.iter_term_freqs().collect();
                candidates.pop();
                candidates.push(Reverse((
                    ScoredDoc::new(row_id, score),
                    freqs,
                    doc_length,
                    doc_id,
                )));
                let kth = candidates.peek().unwrap().0.0.score.0;
                self.update_threshold(kth, params.wand_factor);
            }

            self.advance_lead_to_head(doc_id + 1);
        }
        metrics.record_comparisons(num_comparisons);

        // flat_search is driven by an explicit row_ids iterator, so
        // every candidate already has a real row_id.
        Ok(candidates
            .into_iter()
            .map(
                |Reverse((doc, freqs, doc_length, posting_doc_id))| DocCandidate {
                    addr: CandidateAddr::RowId(doc.row_id),
                    posting_doc_id,
                    freqs,
                    doc_length,
                },
            )
            .collect())
    }

    // calculate the score of the current document
    fn score(&self, doc_length: u32) -> f32 {
        let mut score = 0.0;
        for posting in &self.lead {
            if let Some(doc) = posting.doc() {
                score += posting.score(&self.scorer, doc.frequency(), doc_length);
            }
        }
        score
    }

    // iterate over all the preceding terms and collect the term index and frequency
    fn iter_term_freqs(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.lead.iter().filter_map(|posting| {
            posting
                .doc()
                .map(|doc| (posting.term_index(), doc.frequency()))
        })
    }

    fn and_candidate_cannot_beat_threshold(&self, doc_length: u32) -> bool {
        if self.operator != Operator::And
            || self.threshold <= 0.0
            || self.num_terms < 2
            || self.lead.len() != self.num_terms
        {
            return false;
        }

        let Some((first, remaining)) = self.lead.split_first() else {
            return false;
        };
        let Some(doc) = first.doc() else {
            return false;
        };

        let remaining_upper_bound = remaining
            .iter()
            .map(|posting| posting.block_max_score())
            .sum::<f32>();
        first.score(&self.scorer, doc.frequency(), doc_length) + remaining_upper_bound
            <= self.threshold
    }

    // find the next doc candidate
    // Find the next term-level candidate doc. The returned score is the exact
    // contribution from the current `lead` set; additional score can still come
    // from `tail` iterators that are advanced to the same doc later.
    fn next(&mut self) -> Result<Option<(DocInfo, f32)>> {
        if self.operator == Operator::And {
            let candidate = self.next_and_candidate();
            if candidate.is_some() {
                self.and_window_stats.candidates_returned += 1;
            }
            return Ok(candidate.map(|doc| (doc, 0.0)));
        }

        loop {
            let Some(target) = self.head_doc() else {
                if self.advance_tail_to_next_or_window() {
                    continue;
                }
                return Ok(None);
            };

            if self.up_to.is_none_or(|up_to| target > up_to) {
                self.update_max_scores(target);
            }
            self.move_head_doc_to_lead(target);
            if self.lead.is_empty() {
                continue;
            }

            // Block-Max WAND pruning: skip the whole window when its score upper
            // bound cannot reach the top-k threshold.
            if self.threshold > 0.0 && self.or_block_window_max() <= self.threshold {
                // On the final block `up_to` is the `u64::MAX` sentinel; step once
                // there to avoid seeking past the valid doc id range.
                let skip_to = match self.up_to {
                    Some(up_to) if up_to < u32::MAX as u64 => up_to + 1,
                    _ => target + 1,
                };
                self.push_back_leads(skip_to);
                continue;
            }

            let Some(first_doc) = self.lead.first().and_then(|posting| posting.doc()) else {
                self.push_back_leads(target + 1);
                continue;
            };
            let doc_length = match &first_doc {
                DocInfo::Raw(doc) => self.docs.num_tokens(doc.doc_id),
                DocInfo::Located(doc) => self.docs.num_tokens_by_row_id(doc.row_id),
            };
            let mut lead_score = 0.0;
            if let Some(first_posting) = self.lead.first() {
                lead_score += first_posting.score(&self.scorer, first_doc.frequency(), doc_length);
            }
            for posting in self.lead.iter().skip(1) {
                if let Some(lead_doc) = posting.doc() {
                    lead_score += posting.score(&self.scorer, lead_doc.frequency(), doc_length);
                }
            }

            while lead_score <= self.threshold {
                if lead_score + self.tail_max_score <= self.threshold {
                    self.push_back_leads(first_doc.doc_id() + 1);
                    break;
                }
                if !self.advance_tail_top(target, doc_length, &mut lead_score) {
                    self.push_back_leads(first_doc.doc_id() + 1);
                    break;
                }
            }

            if !self.lead.is_empty() {
                return Ok(Some((first_doc, lead_score)));
            }
        }
    }
    fn next_and_candidate(&mut self) -> Option<DocInfo> {
        if self.lead.len() < self.num_terms {
            return None;
        }
        if let Some(last_doc) = self.and_last_doc
            && self
                .lead
                .first()
                .and_then(|posting| posting.doc())
                .map(|doc| doc.doc_id())
                == Some(last_doc)
        {
            let next_target = self.and_advance_target(last_doc + 1);
            if next_target == TERMINATED_DOC_ID {
                return None;
            }
            self.lead[0].next(next_target);
        }

        'advance_head: loop {
            let doc = self
                .lead
                .first()
                .and_then(|posting| posting.doc())?
                .doc_id();
            if self.up_to.is_none_or(|up_to| doc > up_to) {
                let next_target = self.and_advance_target(doc);
                if next_target == TERMINATED_DOC_ID {
                    return None;
                }
                if next_target != doc {
                    self.lead[0].next(next_target);
                    continue;
                }
            }

            for posting in self.lead.iter_mut().skip(1) {
                if posting.doc()?.doc_id() < doc {
                    posting.next(doc);
                }
                let next = posting.doc()?.doc_id();
                if next > doc {
                    let next_target = self.and_advance_target(next);
                    if next_target == TERMINATED_DOC_ID {
                        return None;
                    }
                    self.lead[0].next(next_target);
                    continue 'advance_head;
                }
            }

            let lead_doc = self.lead.first().and_then(|posting| posting.doc())?;
            let doc_length = match &lead_doc {
                DocInfo::Raw(doc) => self.docs.num_tokens(doc.doc_id),
                DocInfo::Located(doc) => self.docs.num_tokens_by_row_id(doc.row_id),
            };
            if self.and_candidate_cannot_beat_threshold(doc_length) {
                self.and_candidates_pruned_before_return += 1;
                let next_target = self.and_advance_target(doc.saturating_add(1));
                if next_target == TERMINATED_DOC_ID {
                    return None;
                }
                self.lead[0].next(next_target);
                continue;
            }

            self.and_last_doc = Some(doc);
            return Some(lead_doc);
        }
    }

    fn posting_block_up_to(posting: &PostingIterator, target: u64) -> u64 {
        posting
            .next_block_first_doc()
            .map(|doc| doc.saturating_sub(1))
            .unwrap_or(TERMINATED_DOC_ID)
            .max(target)
    }

    fn and_move_to_next_block(&mut self, target: u64) {
        if self.threshold <= 0.0 {
            self.up_to = Some(target);
            self.and_max_score = f32::INFINITY;
            return;
        }

        if self.lead.is_empty() {
            self.up_to = Some(TERMINATED_DOC_ID);
            self.and_max_score = 0.0;
            return;
        }

        for posting in &mut self.lead {
            posting.shallow_next(target);
        }

        let narrow_up_to = self
            .lead
            .iter()
            .map(|posting| Self::posting_block_up_to(posting, target))
            .min()
            .unwrap_or(TERMINATED_DOC_ID);
        let narrow_max_score = self
            .lead
            .iter()
            .map(|posting| posting.block_max_score())
            .sum::<f32>();

        if narrow_max_score >= self.threshold {
            self.up_to = Some(narrow_up_to);
            self.and_max_score = narrow_max_score;
            self.and_window_stats.windows_narrow += 1;
            return;
        }

        let lead_up_to = self
            .lead
            .first()
            .map(|posting| Self::posting_block_up_to(posting, target))
            .unwrap_or(TERMINATED_DOC_ID);
        let can_try_wide = lead_up_to > narrow_up_to
            && lead_up_to != TERMINATED_DOC_ID
            && self.lead.iter().all(|posting| posting.is_compressed());

        if can_try_wide {
            let mut wide_max_score = 0.0;
            let mut range_blocks_scanned = 0;
            for posting in &mut self.lead {
                let block_max = posting.block_max_score_up_to_with_stats(lead_up_to);
                wide_max_score += block_max.score;
                range_blocks_scanned += block_max.blocks_scanned;
            }
            self.and_window_stats.range_blocks_scanned += range_blocks_scanned;

            if wide_max_score < self.threshold {
                self.up_to = Some(lead_up_to);
                self.and_max_score = wide_max_score;
                self.and_window_stats.windows_wide += 1;
                return;
            }
        }

        self.up_to = Some(narrow_up_to);
        self.and_max_score = narrow_max_score;
        self.and_window_stats.windows_narrow += 1;
    }

    fn and_advance_target(&mut self, mut target: u64) -> u64 {
        if self.up_to.is_none_or(|up_to| target > up_to) {
            self.and_move_to_next_block(target);
        }

        loop {
            let Some(up_to) = self.up_to else {
                return TERMINATED_DOC_ID;
            };
            if self.and_max_score >= self.threshold {
                return target;
            }
            self.and_window_stats.windows_skipped += 1;
            if up_to == TERMINATED_DOC_ID {
                return TERMINATED_DOC_ID;
            }
            target = up_to + 1;
            self.and_move_to_next_block(target);
        }
    }

    #[allow(clippy::vec_box)]
    fn head_doc(&self) -> Option<u64> {
        self.head.peek().map(HeadPosting::doc_id)
    }

    fn push_head(&mut self, posting: Box<PostingIterator>) {
        if posting.doc().is_some() {
            self.head.push(HeadPosting::new(posting));
        }
    }

    fn move_head_doc_to_lead(&mut self, target: u64) {
        while self.head_doc() == Some(target) {
            if let Some(posting) = self.head.pop() {
                self.lead.push(posting.posting);
            }
        }
    }

    // Move all head iterators that are already known to be behind `target`
    // into `tail`, possibly overflowing low-value entries back into `head`.
    fn move_head_before_target_to_tail(&mut self, target: u64) {
        while matches!(self.head_doc(), Some(doc_id) if doc_id < target) {
            if let Some(posting) = self.head.pop() {
                let upper_bound = posting.posting.approximate_upper_bound();
                if let Some(mut evicted) =
                    self.insert_tail_with_overflow(posting.posting, upper_bound)
                {
                    evicted.next(target);
                    self.push_head(evicted);
                }
            }
        }
    }

    /// Upper bound on the score of any document in the window `[target, up_to]`
    /// for a disjunction. Sums the block-max of every overlapping iterator:
    /// `lead`, `head` (later docs still in the window, which
    /// `can_target_beat_threshold` omits), and the tail via `tail_max_score`.
    fn or_block_window_max(&self) -> f32 {
        let lead: f32 = self
            .lead
            .iter()
            .map(|posting| posting.block_max_score())
            .sum();
        let head: f32 = self
            .head
            .iter()
            .map(|posting| posting.posting.block_max_score())
            .sum();
        lead + head + self.tail_max_score
    }

    fn can_target_beat_threshold(&mut self, target: u64) -> bool {
        if self.up_to.is_none_or(|up_to| target > up_to) {
            self.update_max_scores(target);
        }

        let mut sum = self
            .lead
            .iter()
            .map(|posting| posting.block_max_score())
            .sum::<f32>();
        let mut possible_matches = self.lead.len();
        for posting in &self.tail {
            if matches!(posting.posting.block_first_doc(), Some(block_doc) if block_doc <= target) {
                sum += posting.posting.block_max_score();
                possible_matches += 1;
            }
        }

        match self.operator {
            Operator::And => possible_matches >= self.num_terms && sum > self.threshold,
            Operator::Or => sum > self.threshold,
        }
    }

    fn update_max_scores(&mut self, target: u64) {
        // Refresh the block-max window for the current target. The resulting
        // `up_to` is the furthest doc id for which this block-max view remains
        // valid.
        let lead_cost = self
            .lead
            .iter()
            .map(|posting| posting.cost())
            .min()
            .unwrap_or(usize::MAX);
        let mut up_to = TERMINATED_DOC_ID;
        for posting in &mut self.lead {
            posting.shallow_next(target);
            let block_end = posting
                .next_block_first_doc()
                .map(|doc| doc.saturating_sub(1))
                .unwrap_or(TERMINATED_DOC_ID);
            up_to = up_to.min(block_end);
        }
        let head = std::mem::take(&mut self.head);
        let mut rebuilt_head = BinaryHeap::with_capacity(head.len());
        for mut posting in head.into_vec() {
            if posting.posting.cost() <= lead_cost {
                posting.posting.shallow_next(posting.doc_id());
                let block_end = posting
                    .posting
                    .next_block_first_doc()
                    .map(|doc| doc.saturating_sub(1))
                    .unwrap_or(TERMINATED_DOC_ID);
                up_to = up_to.min(block_end);
            }
            rebuilt_head.push(posting);
        }
        self.head = rebuilt_head;
        if up_to == TERMINATED_DOC_ID
            && let Some(top) = self.tail.peek()
            && top.cost <= lead_cost
        {
            let block_end = top
                .posting
                .next_block_first_doc()
                .map(|doc| doc.saturating_sub(1))
                .unwrap_or(TERMINATED_DOC_ID);
            up_to = up_to.min(block_end.max(target));
        }
        self.up_to = Some(up_to);

        let tail = std::mem::take(&mut self.tail);
        self.tail_max_score = 0.0;
        for mut tail_posting in tail.into_vec() {
            tail_posting.posting.shallow_next(target);
            let upper_bound = match tail_posting.posting.block_first_doc() {
                Some(block_doc) if block_doc <= target => {
                    tail_posting
                        .posting
                        .block_max_score_up_to_with_stats(up_to)
                        .score
                }
                _ => 0.0,
            };
            if let Some(mut evicted) =
                self.insert_tail_with_overflow(tail_posting.posting, upper_bound)
            {
                evicted.next(target);
                self.push_head(evicted);
            }
        }
    }

    fn refine_or_candidate(&mut self, target: u64, doc_length: u32) -> bool {
        if self.threshold <= 0.0 {
            return true;
        }

        let mut lead_score = self
            .lead
            .iter()
            .filter_map(|posting| {
                posting
                    .doc()
                    .map(|doc| posting.score(&self.scorer, doc.frequency(), doc_length))
            })
            .sum::<f32>();

        while lead_score <= self.threshold {
            if lead_score + self.tail_max_score <= self.threshold {
                return false;
            }
            if !self.advance_tail_top(target, doc_length, &mut lead_score) {
                return false;
            }
        }

        true
    }

    fn collect_tail_matches(&mut self, target: u64) {
        let mut remaining = Vec::with_capacity(self.tail.len());
        let tail = std::mem::take(&mut self.tail);
        self.tail_max_score = 0.0;
        for tail_posting in tail.into_vec() {
            let mut posting = tail_posting.posting;
            posting.next(target);
            match posting.doc().map(|doc| doc.doc_id()) {
                Some(doc_id) if doc_id == target => self.lead.push(posting),
                Some(_) => remaining.push(posting),
                None => {}
            }
        }

        for posting in remaining {
            self.push_head(posting);
        }
    }

    fn advance_tail_and_lead_to_head(&mut self, least_id: u64) {
        let mut postings = Vec::with_capacity(self.tail.len() + self.lead.len());
        while let Some(tail) = self.tail.pop() {
            postings.push(tail.posting);
        }
        self.tail_max_score = 0.0;
        postings.append(&mut self.lead);
        for mut posting in postings {
            posting.next(least_id);
            self.push_head(posting);
        }
    }

    fn advance_lead_to_head(&mut self, least_id: u64) {
        let lead = std::mem::take(&mut self.lead);
        for mut posting in lead {
            posting.next(least_id);
            self.push_head(posting);
        }
        // In the flat-search path this is only called after `collect_tail_matches`,
        // which drains the current tail into either `lead` or `head`. At this
        // point `tail` is expected to be empty, so clearing it is a no-op that
        // just resets the cached `tail_max_score`.
        debug_assert!(self.tail.is_empty());
        self.clear_tail();
    }

    fn clear_tail(&mut self) {
        self.tail.clear();
        self.tail_max_score = 0.0;
    }

    fn insert_tail(&mut self, posting: Box<PostingIterator>, upper_bound: f32) {
        self.tail_max_score += upper_bound;
        self.tail
            .push(TailPosting::new(upper_bound, posting.cost(), posting));
    }

    fn insert_tail_with_overflow(
        &mut self,
        posting: Box<PostingIterator>,
        upper_bound: f32,
    ) -> Option<Box<PostingIterator>> {
        // Keep only the lagging iterators that are most useful for deciding the
        // current candidate. If a stronger tail entry arrives, evict the weakest
        // one back to the caller so it can be advanced into `head`.
        if self.threshold <= 0.0 || upper_bound <= 0.0 {
            return Some(posting);
        }

        if self.tail_max_score + upper_bound < self.threshold {
            self.insert_tail(posting, upper_bound);
            return None;
        }

        if self.tail.is_empty() {
            return Some(posting);
        }

        let candidate = TailPosting::new(upper_bound, posting.cost(), posting);
        if let Some(top) = self.tail.peek()
            && top > &candidate
        {
            let evicted = self.tail.pop().expect("peeked tail posting should exist");
            self.tail_max_score = self.tail_max_score - evicted.upper_bound + upper_bound;
            self.tail.push(candidate);
            return Some(evicted.posting);
        }

        Some(candidate.posting)
    }

    fn lead_to_tail_upper_bound(&self, posting: &PostingIterator, target: u64) -> f32 {
        if self.operator == Operator::Or
            && posting.is_compressed()
            && self.up_to.is_some_and(|up_to| target <= up_to)
        {
            posting.block_max_score()
        } else {
            posting.approximate_upper_bound()
        }
    }

    fn advance_tail_to_next_or_window(&mut self) -> bool {
        if self.operator != Operator::Or || self.tail.is_empty() {
            return false;
        }

        let Some(up_to) = self.up_to else {
            return false;
        };
        if up_to >= u32::MAX as u64 {
            return false;
        }
        if !self
            .tail
            .iter()
            .any(|tail| tail.posting.has_next_compressed_block())
        {
            return false;
        }

        // A low-scoring tail can be the only iterator left in the current
        // window. Move to the next window so a later high-scoring block is still
        // reachable instead of ending the disjunction early.
        self.update_max_scores(up_to + 1);
        true
    }

    fn push_back_leads(&mut self, target: u64) {
        // After finishing a candidate doc, convert the aligned iterators back
        // into lagging iterators. Entries that do not stay in `tail` are
        // advanced to `target` and returned to `head`.
        // pop() drains in place, keeping self.lead's capacity for reuse.
        while let Some(posting) = self.lead.pop() {
            let upper_bound = self.lead_to_tail_upper_bound(&posting, target);
            if let Some(mut evicted) = self.insert_tail_with_overflow(posting, upper_bound) {
                evicted.next(target);
                self.push_head(evicted);
            }
        }
    }

    fn advance_tail_top(&mut self, target: u64, doc_length: u32, lead_score: &mut f32) -> bool {
        // Advance the most promising lagging iterator to the current target.
        // If it lands on the target, fold its exact contribution into
        // `lead_score`; otherwise put it back into `head`.
        let Some(TailPosting {
            upper_bound,
            cost: _,
            mut posting,
        }) = self.tail.pop()
        else {
            return false;
        };
        self.tail_max_score -= upper_bound;
        posting.next(target);
        match posting.doc() {
            Some(doc) if doc.doc_id() == target => {
                *lead_score += posting.score(&self.scorer, doc.frequency(), doc_length);
                self.lead.push(posting);
            }
            Some(_) => self.push_head(posting),
            None => {}
        }
        true
    }

    fn advance_all_tail(
        &mut self,
        target: u64,
        doc_length: Option<u32>,
        mut score: Option<&mut f32>,
    ) {
        // Materialize all remaining lagging iterators for `target`. This is
        // only done once we have already decided to fully score / validate the
        // candidate.
        let tail = std::mem::take(&mut self.tail);
        self.tail_max_score = 0.0;
        for tail_posting in tail.into_vec() {
            let mut posting = tail_posting.posting;
            posting.next(target);
            match posting.doc() {
                Some(doc) if doc.doc_id() == target => {
                    if let (Some(doc_length), Some(score)) = (doc_length, score.as_deref_mut()) {
                        *score += posting.score(&self.scorer, doc.frequency(), doc_length);
                    }
                    self.lead.push(posting)
                }
                Some(_) => self.push_head(posting),
                None => {}
            }
        }
    }

    fn current_doc_postings(&self) -> Vec<&PostingIterator> {
        if !self.lead.is_empty() {
            return self.lead.iter().map(|posting| posting.as_ref()).collect();
        }

        let Some(target) = self.head_doc() else {
            return Vec::new();
        };
        self.head
            .iter()
            .filter(|posting| posting.doc_id() == target)
            .map(|posting| posting.posting.as_ref())
            .collect()
    }

    fn check_positions(&self, slop: i32) -> bool {
        if slop == 0 {
            return self.check_exact_positions();
        }

        let mut position_iters = self
            .current_doc_postings()
            .into_iter()
            .map(|posting| posting.position_cursor().expect("positions must exist"))
            .collect::<Vec<_>>();
        position_iters.sort_unstable_by_key(|iter| iter.position_in_query);

        loop {
            let mut max_relative_pos = None;
            let mut all_same = true;
            for window in position_iters.windows(2) {
                let last = window[0].relative_position();
                let next = window[1].relative_position();
                let (Some(last), Some(next)) = (last, next) else {
                    return false;
                };

                let move_to = if last > next {
                    last
                } else {
                    std::cmp::max(last + 1, next - slop)
                };
                max_relative_pos = max_relative_pos.max(Some(move_to));
                if !(last <= next && next <= last + slop) {
                    all_same = false;
                    break;
                }
            }

            if all_same {
                return true;
            }

            position_iters.iter_mut().for_each(|iter| {
                iter.advance_to_relative(max_relative_pos.unwrap());
            });
        }
    }

    fn check_exact_positions(&self) -> bool {
        let mut position_iters = self
            .current_doc_postings()
            .into_iter()
            .map(|posting| posting.position_cursor().expect("positions must exist"))
            .collect::<Vec<_>>();
        position_iters.sort_unstable_by_key(|iter| iter.len());
        let Some(lead) = position_iters.first() else {
            return false;
        };
        let lead_position = lead.position_in_query;

        loop {
            let Some(anchor) = position_iters[0].absolute_position() else {
                return false;
            };
            let Some(base) = anchor.checked_sub(lead_position as u32) else {
                position_iters[0].advance_next();
                continue;
            };

            let mut next_lead_relative = None;
            let mut matched = true;
            for follower in position_iters.iter_mut().skip(1) {
                let Some(target) = base.checked_add(follower.position_in_query as u32) else {
                    return false;
                };
                let Some(position) = follower.advance_to_absolute(target) else {
                    return false;
                };
                if position != target {
                    next_lead_relative = Some(position as i32 - follower.position_in_query);
                    matched = false;
                    break;
                }
            }

            if matched {
                return true;
            }

            position_iters[0].advance_to_relative(next_lead_relative.unwrap());
        }
    }
}

#[derive(Debug)]
enum PositionValues<'a> {
    Borrowed(&'a [u32]),
    Owned(Vec<u32>),
}

impl<'a> PositionValues<'a> {
    fn as_slice(&self) -> &[u32] {
        match self {
            Self::Borrowed(values) => values,
            Self::Owned(values) => values.as_slice(),
        }
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }
}

#[derive(Debug)]
struct PositionCursor<'a> {
    positions: PositionValues<'a>,
    pub position_in_query: i32,
    index: usize,
}

impl<'a> PositionCursor<'a> {
    fn new(positions: PositionValues<'a>, position_in_query: i32) -> Self {
        Self {
            positions,
            position_in_query,
            index: 0,
        }
    }

    fn len(&self) -> usize {
        self.positions.len()
    }

    fn absolute_position(&self) -> Option<u32> {
        self.positions.as_slice().get(self.index).copied()
    }

    fn relative_position(&self) -> Option<i32> {
        self.positions
            .as_slice()
            .get(self.index)
            .map(|position| *position as i32 - self.position_in_query)
    }

    fn advance_to_relative(&mut self, least_relative_pos: i32) {
        if self.index >= self.len() {
            return;
        }
        let least_pos = least_relative_pos + self.position_in_query;
        let least_pos = least_pos.max(0) as u32;
        let values = self.positions.as_slice();
        self.index += values[self.index..].partition_point(|&pos| pos < least_pos);
    }

    fn advance_to_absolute(&mut self, least_pos: u32) -> Option<u32> {
        if self.index >= self.len() {
            return None;
        }
        let values = self.positions.as_slice();
        self.index += values[self.index..].partition_point(|&pos| pos < least_pos);
        self.absolute_position()
    }

    fn advance_next(&mut self) {
        self.index = self.index.saturating_add(1).min(self.len());
    }
}

#[cfg(test)]
mod tests {
    use arrow::buffer::ScalarBuffer;
    use rstest::rstest;

    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::scalar::inverted::scorer::IndexBM25Scorer;
    use crate::{
        metrics::{MetricsCollector, NoOpMetricsCollector},
        scalar::inverted::{
            CompressedPostingList, PlainPostingList, PostingListBuilder, builder::PositionRecorder,
            encoding::compress_posting_list,
        },
    };

    struct UnitScorer;

    impl Scorer for UnitScorer {
        fn query_weight(&self, _token: &str) -> f32 {
            1.0
        }

        fn doc_weight(&self, freq: u32, _doc_tokens: u32) -> f32 {
            freq as f32
        }
    }

    struct PanicQueryWeightScorer;

    impl Scorer for PanicQueryWeightScorer {
        fn query_weight(&self, _token: &str) -> f32 {
            panic!("query_weight should be precomputed before WAND construction");
        }

        fn doc_weight(&self, freq: u32, _doc_tokens: u32) -> f32 {
            freq as f32
        }
    }

    struct InverseDocLengthScorer;

    impl Scorer for InverseDocLengthScorer {
        fn query_weight(&self, _token: &str) -> f32 {
            1.0
        }

        fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32 {
            freq as f32 / doc_tokens as f32
        }
    }

    // Inverse-doc-length scorer that counts scored documents, so a test can
    // assert that block-max pruning skipped blocks.
    struct CountingScorer {
        scored: Arc<AtomicUsize>,
    }

    impl Scorer for CountingScorer {
        fn query_weight(&self, _token: &str) -> f32 {
            1.0
        }

        fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32 {
            self.scored.fetch_add(1, Ordering::Relaxed);
            freq as f32 / doc_tokens as f32
        }
    }

    #[derive(Default)]
    struct CountAndSearchStats {
        comparisons: AtomicUsize,
        candidates_seen: AtomicUsize,
        candidates_pruned_before_return: AtomicUsize,
        full_scores: AtomicUsize,
        freqs_collected: AtomicUsize,
    }

    impl MetricsCollector for CountAndSearchStats {
        fn record_parts_loaded(&self, _: usize) {}

        fn record_index_loads(&self, _: usize) {}

        fn record_comparisons(&self, n: usize) {
            self.comparisons.fetch_add(n, Ordering::Relaxed);
        }

        fn record_and_candidates_seen(&self, n: usize) {
            self.candidates_seen.fetch_add(n, Ordering::Relaxed);
        }

        fn record_and_candidates_pruned_before_return(&self, n: usize) {
            self.candidates_pruned_before_return
                .fetch_add(n, Ordering::Relaxed);
        }

        fn record_and_full_scores(&self, n: usize) {
            self.full_scores.fetch_add(n, Ordering::Relaxed);
        }

        fn record_freqs_collected(&self, n: usize) {
            self.freqs_collected.fetch_add(n, Ordering::Relaxed);
        }
    }

    struct PanicOnAndMetrics {
        comparisons: AtomicUsize,
    }

    impl PanicOnAndMetrics {
        fn new() -> Self {
            Self {
                comparisons: AtomicUsize::new(0),
            }
        }
    }

    impl MetricsCollector for PanicOnAndMetrics {
        fn record_parts_loaded(&self, _: usize) {}

        fn record_index_loads(&self, _: usize) {}

        fn record_comparisons(&self, n: usize) {
            self.comparisons.fetch_add(n, Ordering::Relaxed);
        }

        fn record_and_candidates_seen(&self, _: usize) {
            panic!("OR search should not record AND candidate metrics");
        }

        fn record_and_candidates_pruned_before_return(&self, _: usize) {
            panic!("OR search should not record AND prune metrics");
        }

        fn record_and_full_scores(&self, _: usize) {
            panic!("OR search should not record AND scoring metrics");
        }

        fn record_freqs_collected(&self, _: usize) {
            panic!("OR search should not record AND frequency metrics");
        }
    }

    fn generate_posting_list(
        doc_ids: Vec<u32>,
        max_score: f32,
        block_max_scores: Option<Vec<f32>>,
        is_compressed: bool,
    ) -> PostingList {
        let freqs = vec![1; doc_ids.len()];
        generate_posting_list_with_freqs(doc_ids, freqs, max_score, block_max_scores, is_compressed)
    }

    fn generate_posting_list_with_freqs(
        doc_ids: Vec<u32>,
        freqs: Vec<u32>,
        max_score: f32,
        block_max_scores: Option<Vec<f32>>,
        is_compressed: bool,
    ) -> PostingList {
        assert_eq!(doc_ids.len(), freqs.len());
        let block_max_scores = block_max_scores.unwrap_or_else(|| vec![max_score; doc_ids.len()]);
        if is_compressed {
            let blocks = compress_posting_list(
                doc_ids.len(),
                doc_ids.iter(),
                freqs.iter(),
                block_max_scores.into_iter(),
            )
            .unwrap();
            PostingList::Compressed(CompressedPostingList::new(
                blocks,
                max_score,
                doc_ids.len() as u32,
                crate::scalar::inverted::PostingTailCodec::VarintDelta,
                None,
            ))
        } else {
            PostingList::Plain(PlainPostingList::new(
                ScalarBuffer::from_iter(doc_ids.iter().map(|id| *id as u64)),
                ScalarBuffer::from_iter(freqs.iter().map(|freq| *freq as f32)),
                Some(max_score),
                None,
            ))
        }
    }

    fn generate_posting_list_with_positions(
        doc_ids: Vec<u32>,
        positions_by_doc: Vec<Vec<u32>>,
        max_score: f32,
        is_compressed: bool,
    ) -> PostingList {
        let freqs = positions_by_doc
            .iter()
            .map(|positions| positions.len() as u32)
            .collect::<Vec<_>>();
        if is_compressed {
            let mut builder = PostingListBuilder::new(true);
            for (doc_id, positions) in doc_ids.iter().copied().zip(positions_by_doc) {
                builder.add(doc_id, PositionRecorder::Position(positions.into()));
            }
            let batch = builder
                .to_batch(vec![max_score; doc_ids.len().div_ceil(BLOCK_SIZE)])
                .unwrap();
            PostingList::from_batch(&batch, Some(max_score), Some(doc_ids.len() as u32)).unwrap()
        } else {
            let mut position_builder =
                arrow::array::ListBuilder::new(arrow::array::Int32Builder::new());
            for positions in positions_by_doc {
                for position in positions {
                    position_builder.values().append_value(position as i32);
                }
                position_builder.append(true);
            }
            PostingList::Plain(PlainPostingList::new(
                ScalarBuffer::from_iter(doc_ids.iter().map(|id| *id as u64)),
                ScalarBuffer::from_iter(freqs.iter().map(|freq| *freq as f32)),
                Some(max_score),
                Some(position_builder.finish()),
            ))
        }
    }

    fn sorted_candidate_row_ids(candidates: Vec<DocCandidate>) -> Vec<u64> {
        let mut row_ids = candidates
            .into_iter()
            .map(|candidate| match candidate.addr {
                CandidateAddr::RowId(row_id) => row_id,
                CandidateAddr::Pending(doc_id) => doc_id as u64,
            })
            .collect::<Vec<_>>();
        row_ids.sort_unstable();
        row_ids
    }

    #[rstest]
    #[tokio::test]
    async fn test_wand(#[values(false, true)] is_compressed: bool) {
        let mut docs = DocSet::default();
        for i in 0..2 * BLOCK_SIZE {
            docs.append(i as u64, 1);
        }

        // when the pivot is greater than 0, and the first posting list is exhausted after shallow_next
        let postings = vec![
            PostingIterator::new(
                String::from("test"),
                0,
                0,
                generate_posting_list(
                    Vec::from_iter(0..=BLOCK_SIZE as u32 + 1),
                    1.0,
                    None,
                    is_compressed,
                ),
                docs.len(),
            ),
            PostingIterator::new(
                String::from("full"),
                1,
                1,
                generate_posting_list(vec![BLOCK_SIZE as u32 + 2], 1.0, None, is_compressed),
                docs.len(),
            ),
        ];

        let bm25 = IndexBM25Scorer::new(std::iter::empty());
        let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, bm25);
        // This should trigger the bug when the second posting list becomes empty
        let result = wand
            .search(
                &FtsSearchParams::default(),
                Arc::new(RowAddrMask::default()),
                &NoOpMetricsCollector,
            )
            .unwrap();
        assert_eq!(result.len(), 0); // Should not panic
    }

    /// The shared floor prunes partitions that can't reach the global top-k: a
    /// high-scoring partition sets the floor and the rest skip their blocks.
    /// (Result correctness is covered by the FTS search tests, since sharing is
    /// always on.)
    #[test]
    fn cross_partition_threshold_sharing_prunes() {
        use crate::metrics::MetricsCollector;
        use std::sync::atomic::AtomicUsize;

        #[derive(Default)]
        struct CountComparisons(AtomicUsize);
        impl MetricsCollector for CountComparisons {
            fn record_parts_loaded(&self, _: usize) {}
            fn record_index_loads(&self, _: usize) {}
            fn record_comparisons(&self, n: usize) {
                self.0.fetch_add(n, Ordering::Relaxed);
            }
        }

        let params = FtsSearchParams::default().with_limit(Some(10));
        let part_docs = 4 * BLOCK_SIZE as u32;
        // One high-scoring partition (weight 10) then 7 low-scoring ones.
        let parts: Vec<(f32, std::ops::Range<u32>)> = std::iter::once((10.0, 0..part_docs))
            .chain((1..8).map(|i| (1.0, i * part_docs..(i + 1) * part_docs)))
            .collect();

        let new_floor = || Arc::new(AtomicU32::new(f32::NEG_INFINITY.to_bits()));

        // Total comparisons across all partitions. `Some(floor)` makes every
        // partition share that one floor; `None` gives each its own.
        let total_comparisons = |shared_floor: Option<&Arc<AtomicU32>>| -> usize {
            let metrics = CountComparisons::default();
            for (qw, rows) in &parts {
                let mut docs = DocSet::default();
                for d in rows.clone() {
                    docs.append(d as u64, 1);
                }
                let postings = vec![PostingIterator::with_query_weight(
                    String::from("t"),
                    0,
                    0,
                    *qw,
                    generate_posting_list(rows.clone().collect(), *qw, None, false),
                    docs.len(),
                )];
                let floor = shared_floor.cloned().unwrap_or_else(new_floor);
                Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer)
                    .with_shared_threshold(floor)
                    .search(&params, Arc::new(RowAddrMask::default()), &metrics)
                    .unwrap();
            }
            metrics.0.load(Ordering::Relaxed)
        };

        let one_floor = new_floor();
        let with_shared_floor = total_comparisons(Some(&one_floor));
        let with_private_floors = total_comparisons(None);
        assert!(
            with_shared_floor < with_private_floors,
            "shared floor should prune comparisons: \
             shared={with_shared_floor} private={with_private_floors}"
        );
    }

    #[test]
    fn test_posting_iterator_next_compressed_partition_point() {
        let mut docs = DocSet::default();
        let num_docs = (BLOCK_SIZE * 2 + 5) as u32;
        for i in 0..num_docs {
            docs.append(i as u64, 1);
        }

        let doc_ids = (0..num_docs).collect::<Vec<_>>();
        let posting = generate_posting_list(doc_ids, 1.0, None, true);
        let mut iter = PostingIterator::new(String::from("term"), 0, 0, posting, docs.len());

        iter.next(10);
        assert_eq!(iter.doc().unwrap().doc_id(), 10);

        let target = BLOCK_SIZE as u64 + 3;
        iter.next(target);
        assert_eq!(iter.doc().unwrap().doc_id(), target);

        iter.next(num_docs as u64 + 10);
        assert!(iter.doc().is_none());
    }

    #[test]
    fn test_wand_skip_to_next_block() {
        let mut docs = DocSet::default();
        for i in 0..201 {
            docs.append(i as u64, 1);
        }

        let large_posting_docs1: Vec<u32> = (0..=200).collect();

        let postings = vec![
            PostingIterator::new(
                String::from("full"),
                0,
                0,
                generate_posting_list(large_posting_docs1, 1.0, Some(vec![0.5, 0.5]), true),
                docs.len(),
            ),
            PostingIterator::new(
                String::from("text"),
                1,
                1,
                generate_posting_list(vec![0], 1.0, Some(vec![0.5]), true),
                docs.len(),
            ),
        ];

        let bm25 = IndexBM25Scorer::new(std::iter::empty());
        let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, bm25);

        // set a threshold that the sum of max scores can hit,
        // but the sum of block max scores is less than the threshold,
        wand.threshold = 1.5;

        let result = wand.search(
            &FtsSearchParams::default(),
            Arc::new(RowAddrMask::default()),
            &NoOpMetricsCollector,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_or_single_term_block_skip_matches_and() {
        // Hot docs occupy the middle block; the flanking blocks score far below
        // the threshold. A single-term disjunction must skip them yet return what
        // the conjunctive path returns.
        let total = 3 * BLOCK_SIZE as u32;
        let hot = BLOCK_SIZE as u32..BLOCK_SIZE as u32 + 12;

        let mut docs = DocSet::default();
        for row_id in 0..total {
            // hot docs get distinct scores 1/1..1/12; the rest score 0.001
            let doc_tokens = if hot.contains(&row_id) {
                row_id - hot.start + 1
            } else {
                1000
            };
            docs.append(row_id as u64, doc_tokens);
        }

        let params = FtsSearchParams::new().with_limit(Some(10));
        let run = |operator| {
            let scored = Arc::new(AtomicUsize::new(0));
            let posting = PostingIterator::with_query_weight(
                String::from("term"),
                0,
                0,
                1.0,
                generate_posting_list(
                    (0..total).collect(),
                    1.0,
                    Some(vec![0.001, 1.0, 0.001]),
                    true,
                ),
                docs.len(),
            );
            let mut wand = Wand::new(
                operator,
                std::iter::once(posting),
                &docs,
                CountingScorer {
                    scored: scored.clone(),
                },
            );
            let hits = wand
                .search(
                    &params,
                    Arc::new(RowAddrMask::default()),
                    &NoOpMetricsCollector,
                )
                .unwrap();
            let mut row_ids = hits
                .iter()
                .map(|hit| match hit.addr {
                    CandidateAddr::RowId(r) => r,
                    CandidateAddr::Pending(_) => panic!("row_id should be set in this path"),
                })
                .collect::<Vec<_>>();
            row_ids.sort_unstable();
            (row_ids, scored.load(Ordering::Relaxed))
        };

        let (or_hits, or_scored) = run(Operator::Or);
        let (and_hits, _) = run(Operator::And);

        let expected = (hot.start..hot.start + 10)
            .map(u64::from)
            .collect::<Vec<_>>();
        assert_eq!(or_hits, expected, "OR must return the top-k");
        assert_eq!(or_hits, and_hits, "OR and AND must agree for a single term");
        // Without pruning OR scores all `total` docs; with it the cold blocks are skipped.
        assert!(
            or_scored <= 2 * BLOCK_SIZE,
            "expected pruning to skip a block, but scored {or_scored} of {total}",
        );
    }

    #[rstest]
    fn test_or_search_does_not_record_and_metrics(#[values(false, true)] is_compressed: bool) {
        let mut docs = DocSet::default();
        for row_id in 0..6 {
            docs.append(row_id, 1);
        }

        let postings = vec![
            PostingIterator::with_query_weight(
                String::from("alpha"),
                0,
                0,
                1.0,
                generate_posting_list(vec![0, 1, 4], 1.0, None, is_compressed),
                docs.len(),
            ),
            PostingIterator::with_query_weight(
                String::from("beta"),
                1,
                1,
                1.0,
                generate_posting_list(vec![1, 2, 5], 1.0, None, is_compressed),
                docs.len(),
            ),
        ];

        let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer);
        let metrics = PanicOnAndMetrics::new();
        let candidates = wand
            .search(
                &FtsSearchParams::default(),
                Arc::new(RowAddrMask::default()),
                &metrics,
            )
            .unwrap();

        assert_eq!(sorted_candidate_row_ids(candidates), vec![0, 1, 2, 4, 5]);
        assert!(metrics.comparisons.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_wand_new_uses_precomputed_query_weight() {
        let mut docs = DocSet::default();
        docs.append(1, 1);

        let postings = vec![PostingIterator::with_query_weight(
            String::from("term"),
            0,
            0,
            2.0,
            generate_posting_list(vec![0], 1.0, None, false),
            docs.len(),
        )];

        let wand = Wand::new(
            Operator::Or,
            postings.into_iter(),
            &docs,
            PanicQueryWeightScorer,
        );
        assert_eq!(wand.head.len(), 1);
    }

    #[test]
    fn test_and_search_terminates_for_disjoint_postings() {
        let mut docs = DocSet::default();
        for i in 0..6 {
            docs.append(i, 1);
        }

        let postings = vec![
            PostingIterator::with_query_weight(
                String::from("a"),
                0,
                0,
                1.0,
                generate_posting_list(vec![0, 2, 4], 1.0, None, false),
                docs.len(),
            ),
            PostingIterator::with_query_weight(
                String::from("b"),
                1,
                1,
                1.0,
                generate_posting_list(vec![1, 3, 5], 1.0, None, false),
                docs.len(),
            ),
        ];

        let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer);
        assert!(wand.next().unwrap().is_none());
    }

    #[test]
    fn test_up_to_refreshes_on_first_candidate() {
        let mut docs = DocSet::default();
        for i in 0..=(BLOCK_SIZE as u64 + 1) {
            docs.append(i, 1);
        }

        let postings = vec![PostingIterator::with_query_weight(
            String::from("term"),
            0,
            0,
            1.0,
            generate_posting_list(
                (0..=(BLOCK_SIZE as u32 + 1)).collect(),
                1.0,
                Some(vec![1.0, 1.0]),
                true,
            ),
            docs.len(),
        )];

        let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer);
        assert!(wand.up_to.is_none());
        let _ = wand.next().unwrap();
        assert!(wand.up_to.is_some());
    }

    #[test]
    fn test_or_push_back_lead_uses_current_block_max_for_tail_bound() {
        let total = 2 * BLOCK_SIZE as u32;
        let mut docs = DocSet::default();
        for doc_id in 0..total {
            docs.append(doc_id as u64, 1);
        }

        let postings = vec![PostingIterator::with_query_weight(
            String::from("term"),
            0,
            0,
            1.0,
            generate_posting_list((0..total).collect(), 10.0, Some(vec![1.0, 10.0]), true),
            docs.len(),
        )];
        let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer);
        wand.threshold = 1.5;

        wand.update_max_scores(0);
        wand.move_head_doc_to_lead(0);
        assert_eq!(wand.up_to, Some((BLOCK_SIZE - 1) as u64));

        wand.push_back_leads(1);

        assert_eq!(wand.tail.len(), 1);
        assert!(
            (wand.tail_max_score - 1.0).abs() < 1e-6,
            "tail should use the current block max, got {}",
            wand.tail_max_score
        );
        assert!(wand.head_doc().is_none());
    }

    #[test]
    fn test_or_push_back_lead_falls_back_after_block_window_expires() {
        let total = 2 * BLOCK_SIZE as u32;
        let mut docs = DocSet::default();
        for doc_id in 0..total {
            docs.append(doc_id as u64, 1);
        }

        let freqs = (0..total)
            .map(|doc_id| if doc_id >= BLOCK_SIZE as u32 { 10 } else { 1 })
            .collect::<Vec<_>>();
        let mut posting = PostingIterator::with_query_weight(
            String::from("term"),
            0,
            0,
            1.0,
            generate_posting_list_with_freqs(
                (0..total).collect(),
                freqs,
                10.0,
                Some(vec![1.0, 10.0]),
                true,
            ),
            docs.len(),
        );
        posting.next((BLOCK_SIZE - 1) as u64);
        let mut wand = Wand::new(Operator::Or, std::iter::once(posting), &docs, UnitScorer);
        wand.threshold = 1.5;

        let block_end = (BLOCK_SIZE - 1) as u64;
        wand.update_max_scores(block_end);
        wand.move_head_doc_to_lead(block_end);
        assert_eq!(wand.up_to, Some(block_end));

        wand.push_back_leads(BLOCK_SIZE as u64);

        assert!(wand.tail.is_empty());
        assert_eq!(wand.head_doc(), Some(BLOCK_SIZE as u64));
        let candidate = wand.next().unwrap().unwrap();
        assert_eq!(candidate.0.doc_id(), BLOCK_SIZE as u64);
    }

    #[test]
    fn test_or_plain_tail_does_not_advance_headless_window() {
        let mut docs = DocSet::default();
        for doc_id in 0..4 {
            docs.append(doc_id, 1);
        }

        let postings = vec![PostingIterator::with_query_weight(
            String::from("term"),
            0,
            0,
            1.0,
            generate_posting_list(vec![0, 1, 2, 3], 1.0, None, false),
            docs.len(),
        )];
        let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer);
        wand.threshold = 2.0;

        wand.update_max_scores(0);
        wand.move_head_doc_to_lead(0);
        wand.push_back_leads(1);

        assert_eq!(wand.tail.len(), 1);
        assert!(wand.head_doc().is_none());
        assert!(!wand.advance_tail_to_next_or_window());
    }

    #[test]
    fn test_or_headless_tail_window_scans_past_final_top_tail() {
        let total = 3 * BLOCK_SIZE as u32;
        let mut docs = DocSet::default();
        for doc_id in 0..total {
            docs.append(doc_id as u64, 1);
        }

        let future_docs = (0..BLOCK_SIZE as u32)
            .chain(2 * BLOCK_SIZE as u32..3 * BLOCK_SIZE as u32)
            .collect::<Vec<_>>();
        let mut future_freqs = vec![1; future_docs.len()];
        future_freqs[0] = 4;
        future_freqs[BLOCK_SIZE] = 20;
        let postings = vec![
            PostingIterator::with_query_weight(
                String::from("future"),
                0,
                0,
                1.0,
                generate_posting_list_with_freqs(
                    future_docs,
                    future_freqs,
                    20.0,
                    Some(vec![4.0, 20.0]),
                    true,
                ),
                docs.len(),
            ),
            PostingIterator::with_query_weight(
                String::from("final_tail"),
                1,
                1,
                1.0,
                generate_posting_list_with_freqs(vec![0], vec![6], 6.0, Some(vec![6.0]), true),
                docs.len(),
            ),
            PostingIterator::with_query_weight(
                String::from("booster"),
                2,
                2,
                1.0,
                generate_posting_list_with_freqs(vec![0], vec![7], 7.0, Some(vec![7.0]), true),
                docs.len(),
            ),
        ];

        let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer);
        let result = wand
            .search(
                &FtsSearchParams::new().with_limit(Some(1)),
                Arc::new(RowAddrMask::default()),
                &NoOpMetricsCollector,
            )
            .unwrap();

        assert_eq!(
            sorted_candidate_row_ids(result),
            vec![(2 * BLOCK_SIZE) as u64]
        );
    }

    #[test]
    fn test_and_search_prunes_with_threshold_and_keeps_candidate() {
        let mut docs = DocSet::default();
        for i in 0..(2 * BLOCK_SIZE as u64) {
            let doc_tokens = if i < BLOCK_SIZE as u64 { 100 } else { 1 };
            docs.append(i, doc_tokens);
        }
        let all_docs = (0..2 * BLOCK_SIZE as u32).collect::<Vec<_>>();

        let postings = vec![
            PostingIterator::with_query_weight(
                String::from("a"),
                0,
                0,
                1.0,
                generate_posting_list(all_docs.clone(), 1.0, Some(vec![0.02, 1.0]), true),
                docs.len(),
            ),
            PostingIterator::with_query_weight(
                String::from("b"),
                1,
                1,
                1.0,
                generate_posting_list(all_docs, 1.0, Some(vec![0.02, 1.0]), true),
                docs.len(),
            ),
        ];

        let mut wand = Wand::new(
            Operator::And,
            postings.into_iter(),
            &docs,
            InverseDocLengthScorer,
        );
        wand.threshold = 0.5;

        let candidate = wand.next().unwrap().unwrap();
        assert_eq!(candidate.0.doc_id(), BLOCK_SIZE as u64);
    }

    #[test]
    fn test_and_advance_falls_back_to_narrow_when_range_max_loosens_bound() {
        let total = 4 * BLOCK_SIZE as u32;
        let mut docs = DocSet::default();
        for i in 0..total {
            docs.append(i as u64, 1);
        }

        let lead_docs = (0..total).step_by(2).collect::<Vec<_>>();
        let follower_docs = (0..total).collect::<Vec<_>>();
        let postings = vec![
            PostingIterator::with_query_weight(
                String::from("lead"),
                0,
                0,
                1.0,
                generate_posting_list(lead_docs, 1.0, Some(vec![1.0, 1.0]), true),
                docs.len(),
            ),
            PostingIterator::with_query_weight(
                String::from("follower"),
                1,
                1,
                1.0,
                generate_posting_list(follower_docs, 10.0, Some(vec![0.1, 10.0, 0.1, 0.1]), true),
                docs.len(),
            ),
        ];

        let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer);
        wand.threshold = 5.0;

        let target = wand.and_advance_target(0);

        assert_eq!(target, BLOCK_SIZE as u64);
        assert_eq!(wand.up_to, Some((2 * BLOCK_SIZE - 1) as u64));
        assert!(
            (wand.and_max_score - 11.0).abs() < 1e-6,
            "expected the second narrow window to include the high follower block, got {}",
            wand.and_max_score
        );
        assert_eq!(wand.and_window_stats.windows_wide, 0);
        assert_eq!(wand.and_window_stats.windows_narrow, 2);
        assert_eq!(wand.and_window_stats.windows_skipped, 1);
    }

    #[test]
    fn test_and_advance_uses_narrow_window_for_candidate_ranges() {
        let total = 4 * BLOCK_SIZE as u32;
        let mut docs = DocSet::default();
        for i in 0..total {
            docs.append(i as u64, 1);
        }

        let lead_docs = (0..total).step_by(2).collect::<Vec<_>>();
        let follower_docs = (0..total).collect::<Vec<_>>();
        let postings = vec![
            PostingIterator::with_query_weight(
                String::from("lead"),
                0,
                0,
                1.0,
                generate_posting_list(lead_docs, 1.0, Some(vec![1.0, 1.0]), true),
                docs.len(),
            ),
            PostingIterator::with_query_weight(
                String::from("follower"),
                1,
                1,
                1.0,
                generate_posting_list(follower_docs, 1.0, Some(vec![1.0, 1.0, 1.0, 1.0]), true),
                docs.len(),
            ),
        ];

        let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer);
        wand.threshold = 1.5;

        let target = wand.and_advance_target(0);

        assert_eq!(target, 0);
        assert_eq!(wand.up_to, Some((BLOCK_SIZE - 1) as u64));
        assert!((wand.and_max_score - 2.0).abs() < 1e-6);
        assert_eq!(wand.and_window_stats.windows_wide, 0);
        assert_eq!(wand.and_window_stats.windows_narrow, 1);
        assert_eq!(wand.and_window_stats.range_blocks_scanned, 0);
    }

    #[test]
    fn test_and_wide_window_only_skips_and_does_not_return_candidates() {
        let total = 4 * BLOCK_SIZE as u32;
        let mut docs = DocSet::default();
        for i in 0..total {
            docs.append(i as u64, 1);
        }

        let lead_docs = (0..total).step_by(2).collect::<Vec<_>>();
        let follower_docs = (0..total).collect::<Vec<_>>();
        let postings = vec![
            PostingIterator::with_query_weight(
                String::from("lead"),
                0,
                0,
                1.0,
                generate_posting_list(lead_docs, 3.0, Some(vec![1.0, 3.0]), true),
                docs.len(),
            ),
            PostingIterator::with_query_weight(
                String::from("follower"),
                1,
                1,
                1.0,
                generate_posting_list(follower_docs, 3.0, Some(vec![0.1, 0.1, 3.0, 3.0]), true),
                docs.len(),
            ),
        ];

        let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer);
        wand.threshold = 2.0;

        let candidate = wand.next().unwrap().unwrap();

        assert_eq!(candidate.0.doc_id(), (2 * BLOCK_SIZE) as u64);
        assert_eq!(wand.up_to, Some((3 * BLOCK_SIZE - 1) as u64));
        assert_eq!(wand.and_window_stats.windows_wide, 1);
        assert_eq!(wand.and_window_stats.windows_skipped, 1);
        assert_eq!(wand.and_window_stats.windows_narrow, 1);
        assert_eq!(wand.and_window_stats.candidates_returned, 1);
    }

    #[test]
    fn test_and_range_max_preserves_exact_top_k() {
        let total = 4 * BLOCK_SIZE as u32;
        let hot = BLOCK_SIZE as u32..BLOCK_SIZE as u32 + 16;
        let mut docs = DocSet::default();
        for doc_id in 0..total {
            let doc_tokens = if hot.contains(&doc_id) { 1 } else { 1000 };
            docs.append(doc_id as u64, doc_tokens);
        }

        let params = FtsSearchParams::new().with_limit(Some(8));
        let run = |is_compressed: bool| {
            let lead_docs = (0..total).step_by(2).collect::<Vec<_>>();
            let follower_docs = (0..total).collect::<Vec<_>>();
            let lead_scores = is_compressed.then_some(vec![1.0, 0.001]);
            let follower_scores = is_compressed.then_some(vec![0.001, 1.0, 0.001, 0.001]);
            let postings = vec![
                PostingIterator::with_query_weight(
                    String::from("lead"),
                    0,
                    0,
                    1.0,
                    generate_posting_list(lead_docs, 1.0, lead_scores, is_compressed),
                    docs.len(),
                ),
                PostingIterator::with_query_weight(
                    String::from("follower"),
                    1,
                    1,
                    1.0,
                    generate_posting_list(follower_docs, 1.0, follower_scores, is_compressed),
                    docs.len(),
                ),
            ];
            let mut wand = Wand::new(
                Operator::And,
                postings.into_iter(),
                &docs,
                InverseDocLengthScorer,
            );
            sorted_candidate_row_ids(
                wand.search(
                    &params,
                    Arc::new(RowAddrMask::default()),
                    &NoOpMetricsCollector,
                )
                .unwrap(),
            )
        };

        let compressed = run(true);
        let plain = run(false);
        let expected = hot.step_by(2).map(u64::from).collect::<Vec<_>>();
        assert_eq!(compressed, expected);
        assert_eq!(compressed, plain);
    }

    #[test]
    fn test_block_max_score_up_to_slides_and_expires_old_max() {
        let total = 5 * BLOCK_SIZE as u32;
        let posting = generate_posting_list(
            (0..total).collect(),
            5.0,
            Some(vec![1.0, 4.0, 2.0, 5.0, 3.0]),
            true,
        );
        let mut posting = PostingIterator::new(String::from("term"), 0, 0, posting, total as usize);

        posting.shallow_next(0);
        assert_eq!(
            posting
                .block_max_score_up_to_with_stats((3 * BLOCK_SIZE - 1) as u64)
                .score,
            4.0
        );

        posting.shallow_next((2 * BLOCK_SIZE) as u64);
        assert_eq!(
            posting
                .block_max_score_up_to_with_stats((4 * BLOCK_SIZE - 1) as u64)
                .score,
            5.0
        );

        posting.shallow_next((4 * BLOCK_SIZE) as u64);
        assert_eq!(
            posting
                .block_max_score_up_to_with_stats((5 * BLOCK_SIZE - 1) as u64)
                .score,
            3.0
        );
    }

    #[test]
    fn test_and_candidate_prune_scores_first_term_before_full_score() {
        let total_docs = 2 * BLOCK_SIZE as u32 + 1;
        let mut docs = DocSet::default();
        for doc_id in 0..total_docs {
            let doc_tokens = if doc_id == 0 { 1 } else { 1000 };
            docs.append(doc_id as u64, doc_tokens);
        }

        let first_docs = (0..2 * BLOCK_SIZE as u32).collect::<Vec<_>>();
        let second_docs = (0..total_docs).collect::<Vec<_>>();
        let postings = vec![
            PostingIterator::with_query_weight(
                String::from("a"),
                0,
                0,
                1.0,
                generate_posting_list(first_docs, 1.0, Some(vec![1.0, 0.001]), true),
                docs.len(),
            ),
            PostingIterator::with_query_weight(
                String::from("b"),
                1,
                1,
                1.0,
                generate_posting_list(second_docs, 1.0, Some(vec![1.0, 0.001, 0.001]), true),
                docs.len(),
            ),
        ];

        let scored = Arc::new(AtomicUsize::new(0));
        let mut wand = Wand::new(
            Operator::And,
            postings.into_iter(),
            &docs,
            CountingScorer {
                scored: scored.clone(),
            },
        );

        let result = wand
            .search(
                &FtsSearchParams::new().with_limit(Some(1)),
                Arc::new(RowAddrMask::default()),
                &NoOpMetricsCollector,
            )
            .unwrap();

        let addrs = result.into_iter().map(|doc| doc.addr).collect::<Vec<_>>();
        assert!(matches!(addrs.as_slice(), [CandidateAddr::RowId(0)]));
        let scored = scored.load(Ordering::Relaxed);
        assert!(
            scored <= BLOCK_SIZE + 1,
            "expected candidate pruning to avoid full scoring in the first block, scored {scored}"
        );
    }

    #[test]
    fn test_and_candidate_prune_records_scoring_counters() {
        let total_docs = 2 * BLOCK_SIZE as u32 + 1;
        let mut docs = DocSet::default();
        for doc_id in 0..total_docs {
            let doc_tokens = if doc_id == 0 { 1 } else { 1000 };
            docs.append(doc_id as u64, doc_tokens);
        }

        let first_docs = (0..2 * BLOCK_SIZE as u32).collect::<Vec<_>>();
        let second_docs = (0..total_docs).collect::<Vec<_>>();
        let postings = vec![
            PostingIterator::with_query_weight(
                String::from("a"),
                0,
                0,
                1.0,
                generate_posting_list(first_docs, 1.0, Some(vec![1.0, 0.001]), true),
                docs.len(),
            ),
            PostingIterator::with_query_weight(
                String::from("b"),
                1,
                1,
                1.0,
                generate_posting_list(second_docs, 1.0, Some(vec![1.0, 0.001, 0.001]), true),
                docs.len(),
            ),
        ];

        let mut wand = Wand::new(
            Operator::And,
            postings.into_iter(),
            &docs,
            InverseDocLengthScorer,
        );
        let metrics = CountAndSearchStats::default();
        let result = wand
            .search(
                &FtsSearchParams::new().with_limit(Some(1)),
                Arc::new(RowAddrMask::default()),
                &metrics,
            )
            .unwrap();

        let addrs = result.into_iter().map(|doc| doc.addr).collect::<Vec<_>>();
        assert!(matches!(addrs.as_slice(), [CandidateAddr::RowId(0)]));

        let candidates_seen = metrics.candidates_seen.load(Ordering::Relaxed);
        let candidates_pruned_before_return = metrics
            .candidates_pruned_before_return
            .load(Ordering::Relaxed);
        let full_scores = metrics.full_scores.load(Ordering::Relaxed);
        assert_eq!(metrics.comparisons.load(Ordering::Relaxed), 1);
        assert_eq!(candidates_seen, 1);
        assert!(candidates_pruned_before_return > 0);
        assert_eq!(full_scores, 1);
        assert_eq!(metrics.freqs_collected.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_and_candidate_prune_keeps_later_high_score_candidate() {
        let mut docs = DocSet::default();
        for doc_id in 0..3 {
            docs.append(doc_id, 1);
        }

        let postings = vec![
            PostingIterator::with_query_weight(
                String::from("a"),
                0,
                0,
                1.0,
                generate_posting_list_with_freqs(
                    vec![0, 1],
                    vec![10, 1],
                    10.0,
                    Some(vec![10.0]),
                    true,
                ),
                docs.len(),
            ),
            PostingIterator::with_query_weight(
                String::from("b"),
                1,
                1,
                1.0,
                generate_posting_list_with_freqs(
                    vec![0, 1, 2],
                    vec![1, 20, 1],
                    20.0,
                    Some(vec![20.0]),
                    true,
                ),
                docs.len(),
            ),
        ];

        let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer);
        let result = wand
            .search(
                &FtsSearchParams::new().with_limit(Some(1)),
                Arc::new(RowAddrMask::default()),
                &NoOpMetricsCollector,
            )
            .unwrap();

        let addrs = result.into_iter().map(|doc| doc.addr).collect::<Vec<_>>();
        assert!(matches!(addrs.as_slice(), [CandidateAddr::RowId(1)]));
    }

    #[rstest]
    fn test_wand_batches_lagging_iterators(#[values(false, true)] is_compressed: bool) {
        let mut docs = DocSet::default();
        for i in 0..16 {
            docs.append(i as u64, 1);
        }

        let postings = vec![
            PostingIterator::new(
                String::from("a"),
                0,
                0,
                generate_posting_list(vec![1, 10], 1.0, None, is_compressed),
                docs.len(),
            ),
            PostingIterator::new(
                String::from("b"),
                1,
                1,
                generate_posting_list(vec![2, 10], 1.0, None, is_compressed),
                docs.len(),
            ),
            PostingIterator::new(
                String::from("c"),
                2,
                2,
                generate_posting_list(vec![10], 1.0, None, is_compressed),
                docs.len(),
            ),
        ];

        let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer);
        wand.threshold = 2.5;

        let candidate = wand.next().unwrap().unwrap();
        assert_eq!(candidate.0.doc_id(), 10);
        assert_eq!(wand.lead.len(), 3);
    }

    #[test]
    fn test_flat_search_or_keeps_masked_docs_in_same_block() {
        let mut docs = DocSet::default();
        for i in 0..=(BLOCK_SIZE as u64 + 1) {
            let doc_tokens = if i == 1 { 100 } else { 1 };
            docs.append(i, doc_tokens);
        }

        let posting = PostingIterator::with_query_weight(
            String::from("term"),
            0,
            0,
            1.0,
            generate_posting_list(
                (1..=(BLOCK_SIZE as u32 + 1)).collect(),
                1.0,
                Some(vec![1.0, 1.0]),
                true,
            ),
            docs.len(),
        );

        let mut wand = Wand::new(
            Operator::Or,
            vec![posting].into_iter(),
            &docs,
            InverseDocLengthScorer,
        );
        wand.threshold = 0.5;

        let selected = vec![RowAddress::from(1_u64), RowAddress::from(2_u64)];
        let result = wand
            .flat_search(
                &FtsSearchParams::default(),
                Box::new(selected.into_iter()),
                &NoOpMetricsCollector,
            )
            .unwrap();

        let matched = result
            .into_iter()
            .map(|doc| match doc.addr {
                CandidateAddr::RowId(r) => r,
                CandidateAddr::Pending(_) => panic!("row_id should be set in this path"),
            })
            .collect::<Vec<_>>();
        assert_eq!(matched, vec![2]);
    }

    #[test]
    fn test_doc_ids_resolves_every_document_a_row_owns() {
        // A list<string> column indexes each element as its own document, so
        // one row id owns several doc ids. row 100 -> {0, 1}, row 101 -> {2}.
        let row_id_col = arrow_array::UInt64Array::from(vec![100_u64, 100, 101]);
        let num_tokens_col = arrow_array::UInt32Array::from(vec![1_u32, 1, 1]);
        let docs = DocSet::from_columns(&row_id_col, &num_tokens_col, false, None).unwrap();

        assert_eq!(docs.doc_ids(100).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(docs.doc_ids(101).collect::<Vec<_>>(), vec![2]);
        assert!(docs.doc_ids(999).next().is_none());

        // legacy shape (row id == doc id) still resolves to a single document.
        let mut legacy = DocSet::default();
        legacy.append(7, 1);
        assert_eq!(legacy.doc_ids(7).collect::<Vec<_>>(), vec![7]);
        assert!(legacy.doc_ids(8).next().is_none());
    }

    #[rstest]
    fn test_flat_search_finds_list_row_with_match_at_non_last_position(
        #[values(false, true)] is_compressed: bool,
    ) {
        // row 100 owns two element-documents (doc 0, doc 1) that share its row
        // id; row 101 owns doc 2. The query term lives only in doc 0 — the
        // *non-last* element of row 100. Resolving the row to a single doc id
        // would evaluate doc 1, miss the term, and drop the row (lancedb#3352).
        let row_id_col = arrow_array::UInt64Array::from(vec![100_u64, 100, 101]);
        let num_tokens_col = arrow_array::UInt32Array::from(vec![1_u32, 1, 1]);
        let docs = DocSet::from_columns(&row_id_col, &num_tokens_col, false, None).unwrap();

        let posting = PostingIterator::with_query_weight(
            String::from("needle"),
            0,
            0,
            1.0,
            generate_posting_list(vec![0], 1.0, None, is_compressed),
            docs.len(),
        );

        let mut wand = Wand::new(
            Operator::Or,
            vec![posting].into_iter(),
            &docs,
            InverseDocLengthScorer,
        );
        wand.threshold = 0.5;

        let selected = vec![RowAddress::from(100_u64)];
        let result = wand
            .flat_search(
                &FtsSearchParams::default(),
                Box::new(selected.into_iter()),
                &NoOpMetricsCollector,
            )
            .unwrap();

        // flat_search resolves the prefilter against the DocSet, so the single
        // match comes back as a concrete RowId(100) rather than a deferred
        // Pending addr. Asserting on the whole result avoids a never-taken
        // match arm that would otherwise read as uncovered.
        let addrs = result.into_iter().map(|doc| doc.addr).collect::<Vec<_>>();
        assert!(
            matches!(addrs.as_slice(), [CandidateAddr::RowId(100)]),
            "expected exactly row 100, got {addrs:?}"
        );
    }

    #[test]
    fn test_block_max_score_matches_stored_value() {
        let doc_ids = vec![0_u32];
        let block_max_scores = vec![0.7_f32];
        let posting_list = generate_posting_list(doc_ids, 0.7, Some(block_max_scores), true);
        let expected = match &posting_list {
            PostingList::Compressed(list) => list.block_max_score(0),
            PostingList::Plain(_) => unreachable!("expected compressed posting list"),
        };

        let posting = PostingIterator::new(String::from("test"), 0, 0, posting_list, 1);

        let actual = posting.block_max_score();
        assert!(
            (actual - expected).abs() < 1e-6,
            "block max score should match stored value"
        );
    }

    #[rstest]
    fn test_exact_phrase_with_repeated_terms(#[values(false, true)] is_compressed: bool) {
        let mut docs = DocSet::default();
        docs.append(0, 16);

        let token_a_positions = vec![vec![1_u32, 3, 10]];
        let token_b_positions = vec![vec![2_u32, 11]];
        let postings = vec![
            PostingIterator::new(
                String::from("a"),
                0,
                0,
                generate_posting_list_with_positions(
                    vec![0],
                    token_a_positions.clone(),
                    1.0,
                    is_compressed,
                ),
                docs.len(),
            ),
            PostingIterator::new(
                String::from("b"),
                1,
                1,
                generate_posting_list_with_positions(
                    vec![0],
                    token_b_positions,
                    1.0,
                    is_compressed,
                ),
                docs.len(),
            ),
            PostingIterator::new(
                String::from("a"),
                2,
                2,
                generate_posting_list_with_positions(
                    vec![0],
                    token_a_positions,
                    1.0,
                    is_compressed,
                ),
                docs.len(),
            ),
        ];

        let bm25 = IndexBM25Scorer::new(std::iter::empty());
        let wand = Wand::new(Operator::And, postings.into_iter(), &docs, bm25);
        assert!(wand.check_exact_positions());
        assert!(wand.check_positions(0));
    }

    #[rstest]
    fn test_exact_phrase_respects_query_position_gaps(#[values(false, true)] is_compressed: bool) {
        let mut docs = DocSet::default();
        docs.append(0, 16);

        let postings = vec![
            PostingIterator::new(
                String::from("want"),
                0,
                0,
                generate_posting_list_with_positions(
                    vec![0],
                    vec![vec![0_u32]],
                    1.0,
                    is_compressed,
                ),
                docs.len(),
            ),
            PostingIterator::new(
                String::from("apple"),
                1,
                2,
                generate_posting_list_with_positions(
                    vec![0],
                    vec![vec![2_u32]],
                    1.0,
                    is_compressed,
                ),
                docs.len(),
            ),
        ];

        let bm25 = IndexBM25Scorer::new(std::iter::empty());
        let wand = Wand::new(Operator::And, postings.into_iter(), &docs, bm25);
        assert!(wand.check_exact_positions());
        assert!(wand.check_positions(0));
    }

    #[rstest]
    fn test_and_phrase_miss_advances_to_next_candidate(#[values(false, true)] is_compressed: bool) {
        let mut docs = DocSet::default();
        docs.append(0, 8);
        docs.append(1, 8);

        let postings = vec![
            PostingIterator::new(
                String::from("a"),
                0,
                0,
                generate_posting_list_with_positions(
                    vec![0, 1],
                    vec![vec![1_u32], vec![10_u32]],
                    1.0,
                    is_compressed,
                ),
                docs.len(),
            ),
            PostingIterator::new(
                String::from("b"),
                1,
                1,
                generate_posting_list_with_positions(
                    vec![0, 1],
                    vec![vec![3_u32], vec![11_u32]],
                    1.0,
                    is_compressed,
                ),
                docs.len(),
            ),
        ];

        let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer);
        let first = wand.next().unwrap().unwrap();
        assert_eq!(first.0.doc_id(), 0);
        assert!(!wand.check_positions(0));

        wand.threshold = 1.5;
        let second = wand.next().unwrap().unwrap();
        assert_eq!(second.0.doc_id(), 1);
        assert!(wand.check_positions(0));
    }
}
