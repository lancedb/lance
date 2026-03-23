// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Deref;
use std::sync::{Arc, LazyLock};
use std::{cell::UnsafeCell, collections::BinaryHeap};
use std::{cmp::Reverse, fmt::Debug};

use arrow::array::AsArray;
use arrow::datatypes::Int32Type;
use arrow_array::Array;
use itertools::Itertools;
use lance_core::Result;
use lance_core::utils::address::RowAddress;
use lance_core::utils::mask::RowAddrMask;

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

#[derive(Debug)]
pub struct DocCandidate {
    pub row_id: u64,
    /// (term_index, freq)
    pub freqs: Vec<(u32, u32)>,
    pub doc_length: u32,
}

struct HeadPosting {
    // Iterators that are already positioned on or after the next candidate doc.
    // The heap is ordered by smallest doc id so the top element determines
    // the next target doc to consider.
    posting: Box<PostingIterator>,
}

impl HeadPosting {
    fn new(posting: Box<PostingIterator>) -> Self {
        Self { posting }
    }

    fn doc_id(&self) -> u64 {
        self.posting
            .doc()
            .map(|doc| doc.doc_id())
            .unwrap_or(TERMINATED_DOC_ID)
    }
}

impl PartialEq for HeadPosting {
    fn eq(&self, other: &Self) -> bool {
        self.doc_id() == other.doc_id()
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
            .doc_id()
            .cmp(&self.doc_id())
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
    docs: &'a DocSet,
    scorer: S,
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
            docs,
            scorer,
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

        let mut candidates = BinaryHeap::with_capacity(std::cmp::min(limit, BLOCK_SIZE * 10));
        let mut num_comparisons = 0;
        while let Some((doc, mut score)) = self.next()? {
            num_comparisons += 1;

            let row_id = match &doc {
                DocInfo::Raw(doc) => {
                    // if the doc is not located, we need to find the row id
                    self.docs.row_id(doc.doc_id)
                }
                DocInfo::Located(doc) => doc.row_id,
            };
            if !mask.selected(row_id) {
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
                self.score(doc_length)
            };

            let freqs = self.iter_term_freqs().collect();
            if candidates.len() < limit {
                candidates.push(Reverse((ScoredDoc::new(row_id, score), freqs, doc_length)));
                if candidates.len() == limit {
                    self.threshold = candidates.peek().unwrap().0.0.score.0 * params.wand_factor;
                }
            } else if score > candidates.peek().unwrap().0.0.score.0 {
                candidates.pop();
                candidates.push(Reverse((ScoredDoc::new(row_id, score), freqs, doc_length)));
                self.threshold = candidates.peek().unwrap().0.0.score.0 * params.wand_factor;
            }
            if self.operator == Operator::Or {
                self.push_back_leads(doc.doc_id() + 1);
            }
        }
        metrics.record_comparisons(num_comparisons);

        Ok(candidates
            .into_iter()
            .map(|Reverse((doc, freqs, doc_length))| DocCandidate {
                row_id: doc.row_id,
                freqs,
                doc_length,
            })
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
        // because WAND PostingIterator can't go back to the previous doc id
        let doc_ids = row_ids
            .filter_map(|row_addr| {
                let row_id: u64 = row_addr.into();
                self.docs.doc_id(row_id).map(|doc_id| (doc_id, row_id))
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
            let freqs = self.iter_term_freqs().collect();

            if candidates.len() < limit {
                candidates.push(Reverse((ScoredDoc::new(row_id, score), freqs, doc_length)));
                if candidates.len() == limit {
                    self.threshold = candidates.peek().unwrap().0.0.score.0 * params.wand_factor;
                }
            } else if score > candidates.peek().unwrap().0.0.score.0 {
                candidates.pop();
                candidates.push(Reverse((ScoredDoc::new(row_id, score), freqs, doc_length)));
                self.threshold = candidates.peek().unwrap().0.0.score.0 * params.wand_factor;
            }

            self.advance_lead_to_head(doc_id + 1);
        }
        metrics.record_comparisons(num_comparisons);

        Ok(candidates
            .into_iter()
            .map(|Reverse((doc, freqs, doc_length))| DocCandidate {
                row_id: doc.row_id,
                freqs,
                doc_length,
            })
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

    // find the next doc candidate
    // Find the next term-level candidate doc. The returned score is the exact
    // contribution from the current `lead` set; additional score can still come
    // from `tail` iterators that are advanced to the same doc later.
    fn next(&mut self) -> Result<Option<(DocInfo, f32)>> {
        if self.operator == Operator::And {
            return Ok(self.next_and_candidate().map(|doc| (doc, 0.0)));
        }

        while let Some(target) = self.head_doc() {
            if self.up_to.is_none_or(|up_to| target > up_to) {
                self.update_max_scores(target);
            }
            self.move_head_doc_to_lead(target);
            if self.lead.is_empty() {
                continue;
            }

            let Some(doc) = self.lead.first().and_then(|posting| posting.doc()) else {
                self.push_back_leads(target + 1);
                continue;
            };
            let doc_length = match &doc {
                DocInfo::Raw(doc) => self.docs.num_tokens(doc.doc_id),
                DocInfo::Located(doc) => self.docs.num_tokens_by_row_id(doc.row_id),
            };
            let mut lead_score = self
                .lead
                .iter()
                .filter_map(|posting| {
                    posting.doc().map(|lead_doc| {
                        posting.score(&self.scorer, lead_doc.frequency(), doc_length)
                    })
                })
                .sum::<f32>();

            while lead_score <= self.threshold {
                if lead_score + self.tail_max_score <= self.threshold {
                    self.push_back_leads(doc.doc_id() + 1);
                    break;
                }
                if !self.advance_tail_top(target, doc_length, &mut lead_score) {
                    self.push_back_leads(doc.doc_id() + 1);
                    break;
                }
            }

            if !self.lead.is_empty() {
                return Ok(self
                    .lead
                    .first()
                    .and_then(|posting| posting.doc())
                    .map(|doc| (doc, lead_score)));
            }
        }

        Ok(None)
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

            self.and_last_doc = Some(doc);
            return self.lead.first().and_then(|posting| posting.doc());
        }
    }

    fn and_move_to_next_block(&mut self, target: u64) {
        if self.threshold <= 0.0 {
            self.up_to = Some(target);
            self.and_max_score = f32::INFINITY;
            return;
        }

        let mut up_to = TERMINATED_DOC_ID;
        let mut max_score = 0.0;
        for posting in &mut self.lead {
            posting.shallow_next(target);
            let block_end = posting
                .next_block_first_doc()
                .map(|doc| doc.saturating_sub(1))
                .unwrap_or(TERMINATED_DOC_ID);
            up_to = up_to.min(block_end.max(target));
            max_score += posting.block_max_score();
        }
        self.up_to = Some(up_to);
        self.and_max_score = max_score;
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
                Some(block_doc) if block_doc <= target => tail_posting.posting.block_max_score(),
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

    fn push_back_leads(&mut self, target: u64) {
        // After finishing a candidate doc, convert the aligned iterators back
        // into lagging iterators. Entries that do not stay in `tail` are
        // advanced to `target` and returned to `head`.
        let leads = std::mem::take(&mut self.lead);
        for posting in leads {
            let upper_bound = posting.approximate_upper_bound();
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
        match posting.doc().map(|doc| doc.doc_id()) {
            Some(doc_id) if doc_id == target => {
                let frequency = posting.doc().expect("posting must exist").frequency();
                *lead_score += posting.score(&self.scorer, frequency, doc_length);
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
            match posting.doc().map(|doc| doc.doc_id()) {
                Some(doc_id) if doc_id == target => {
                    if let (Some(doc_length), Some(score)) = (doc_length, score.as_deref_mut()) {
                        let frequency = posting
                            .doc()
                            .expect("posting moved to target should have doc")
                            .frequency();
                        *score += posting.score(&self.scorer, frequency, doc_length);
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

    use super::*;
    use crate::scalar::inverted::scorer::IndexBM25Scorer;
    use crate::{
        metrics::NoOpMetricsCollector,
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

    fn generate_posting_list(
        doc_ids: Vec<u32>,
        max_score: f32,
        block_max_scores: Option<Vec<f32>>,
        is_compressed: bool,
    ) -> PostingList {
        let freqs = vec![1; doc_ids.len()];
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

        let matched = result.into_iter().map(|doc| doc.row_id).collect::<Vec<_>>();
        assert_eq!(matched, vec![2]);
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
