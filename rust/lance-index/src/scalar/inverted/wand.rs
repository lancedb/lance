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

    pub(crate) fn new(
        token: String,
        token_id: u32,
        position: u32,
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

pub struct Wand<'a, S: Scorer> {
    threshold: f32, // multiple of factor and the minimum score of the top-k documents
    operator: Operator,
    num_terms: usize,
    // we need to sort the posting iterators frequently,
    // so wrap them in `Box` to avoid the cost of copying
    #[allow(clippy::vec_box)]
    postings: Vec<Box<PostingIterator>>,
    docs: &'a DocSet,
    scorer: S,
    cur_doc: Option<DocInfo>,
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
        let mut posting_lists = postings.collect::<Vec<_>>();
        posting_lists.sort_unstable();

        Self {
            threshold: 0.0,
            operator,
            num_terms: posting_lists.len(),
            postings: posting_lists.into_iter().map(Box::new).collect(),
            docs,
            scorer,
            cur_doc: None,
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
                if num_rows_matched * 100
                    <= FLAT_SEARCH_PERCENT_THRESHOLD.deref() * self.docs.len() as u64 =>
            {
                return self.flat_search(params, row_ids, metrics);
            }
            _ => {}
        }

        let mut candidates = BinaryHeap::with_capacity(std::cmp::min(limit, BLOCK_SIZE * 10));
        let mut num_comparisons = 0;
        while let Some((pivot, doc)) = self.next()? {
            if let Some(cur_doc) = self.cur_doc
                && cur_doc.doc_id() >= doc.doc_id()
            {
                continue;
            }
            self.cur_doc = Some(doc);
            num_comparisons += 1;

            let row_id = match &doc {
                DocInfo::Raw(doc) => {
                    // if the doc is not located, we need to find the row id
                    self.docs.row_id(doc.doc_id)
                }
                DocInfo::Located(doc) => doc.row_id,
            };
            if !mask.selected(row_id) {
                self.move_preceding(pivot, doc.doc_id() + 1);
                continue;
            }

            if params.phrase_slop.is_some()
                && !self.check_positions(params.phrase_slop.unwrap() as i32)
            {
                self.move_preceding(pivot, doc.doc_id() + 1);
                continue;
            }

            let doc_length = match &doc {
                DocInfo::Raw(doc) => self.docs.num_tokens(doc.doc_id),
                DocInfo::Located(doc) => self.docs.num_tokens_by_row_id(doc.row_id),
            };
            let score = self.score(pivot, doc_length);
            let freqs = self.iter_term_freqs(pivot).collect();
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
            self.move_preceding(pivot, doc.doc_id() + 1);
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
        let is_compressed = matches!(self.postings[0].list, PostingList::Compressed(_));

        let mut num_comparisons = 0;
        let mut candidates = BinaryHeap::new();
        let mut current_doc = 0;
        for (doc_id, row_id) in doc_ids {
            num_comparisons += 1;

            if doc_id < current_doc {
                continue;
            }
            current_doc = doc_id;

            // even we already know the candidate doc id, we still need to know how many terms are required to hit the threshold
            let mut pivot = 0;
            let mut approximate_upper_bound = self.postings[0].approximate_upper_bound();
            while pivot + 1 < self.postings.len() && approximate_upper_bound < self.threshold {
                approximate_upper_bound += self.postings[pivot + 1].approximate_upper_bound();
                pivot += 1;
            }

            if let Some(least_id) = self.postings[0].block_first_doc()
                && least_id > doc_id
            {
                current_doc = least_id;
                continue;
            }
            let mut max_pivot = 0;
            while max_pivot + 1 < self.postings.len() {
                self.postings[max_pivot + 1].shallow_next(doc_id);
                match self.postings[max_pivot + 1].block_first_doc() {
                    Some(block_doc_id) if block_doc_id <= doc_id => {
                        max_pivot += 1;
                    }
                    _ => break,
                }
            }

            if !self.check_block_max(max_pivot, doc_id) {
                // the current block max score is less than the threshold,
                // which means we have to skip at least the current block
                let (_, least_id) = self.get_new_candidate(max_pivot);
                if least_id == TERMINATED_DOC_ID {
                    break;
                }
                current_doc = std::cmp::max(doc_id, least_id);
                self.move_preceding(max_pivot, least_id);
                continue;
            }

            // move all postings to this doc id
            if !self.check_pivot_aligned(pivot, doc_id) {
                if self.postings.is_empty() {
                    break;
                } else {
                    continue;
                }
            }

            max_pivot = 0;
            while max_pivot + 1 < self.postings.len()
                && self.postings[max_pivot + 1].doc().map(|d| d.doc_id()) == Some(doc_id)
            {
                max_pivot += 1;
            }

            // check positions
            if params.phrase_slop.is_some()
                && !self.check_positions(params.phrase_slop.unwrap() as i32)
            {
                continue;
            }

            // score the doc
            let doc_length = match is_compressed {
                true => self.docs.num_tokens(doc_id as u32),
                false => self.docs.num_tokens_by_row_id(row_id),
            };

            let score = self.score(max_pivot, doc_length);
            let freqs = self.iter_term_freqs(max_pivot).collect();

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
    fn score(&self, pivot: usize, doc_length: u32) -> f32 {
        let mut score = 0.0;
        for (token, freq) in self.iter_token_freqs(pivot) {
            score += self.scorer.score(token, freq, doc_length);
        }
        score
    }

    // iterate over all the preceding terms and collect the token and frequency
    fn iter_token_freqs(&self, pivot: usize) -> impl Iterator<Item = (&str, u32)> + '_ {
        self.postings[..=pivot].iter().filter_map(|posting| {
            posting
                .doc()
                .map(|doc| (posting.token.as_str(), doc.frequency()))
        })
    }

    // iterate over all the preceding terms and collect the term index and frequency
    fn iter_term_freqs(&self, pivot: usize) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.postings[..=pivot].iter().filter_map(|posting| {
            posting
                .doc()
                .map(|doc| (posting.term_index(), doc.frequency()))
        })
    }

    // find the next doc candidate
    fn next(&mut self) -> Result<Option<(usize, DocInfo)>> {
        while let Some((pivot, max_pivot)) = self.find_pivot_term() {
            let posting = &self.postings[pivot];
            let doc = posting.doc().unwrap();
            let doc_id = doc.doc_id();

            if !self.check_block_max(max_pivot, doc_id) {
                // the current block max score is less than the threshold,
                // which means we have to skip at least the current block
                let (picked_term, least_id) = self.get_new_candidate(max_pivot);
                if least_id == TERMINATED_DOC_ID {
                    return Ok(None);
                }
                self.move_term(picked_term, least_id);
                continue;
            }

            if !self.check_pivot_aligned(pivot, doc_id) {
                continue;
            }

            // all the posting iterators preceding pivot have reached this doc id,
            // this means the sum of upper bound of all terms is not less than the threshold,
            // this document is a candidate, but we still need to check filters, positions, etc.
            return Ok(Some((max_pivot, doc)));
        }
        Ok(None)
    }

    fn check_block_max(&mut self, pivot: usize, pivot_doc: u64) -> bool {
        let mut sum = 0.0;
        for posting in self.postings[..=pivot].iter_mut() {
            posting.shallow_next(pivot_doc);
            sum += posting.block_max_score();
        }
        sum > self.threshold
    }

    // find the term and new doc_id to move / move to,
    // the term should be the one with the maximum score,
    // the new doc_id should be the one is the minimum among:
    // 1. for the terms preceding the pivot, the next block first doc id
    // 2. for the terms after the pivot, the doc id of the term
    fn get_new_candidate(&self, pivot: usize) -> (usize, u64) {
        let mut picked_term = pivot;
        let mut max_score = self.postings[pivot].approximate_upper_bound();
        let mut least_id = self.postings[pivot]
            .next_block_first_doc()
            .unwrap_or(TERMINATED_DOC_ID);
        for (i, posting) in self.postings[..pivot].iter().enumerate().rev() {
            let next_block_first_doc = posting.next_block_first_doc().unwrap_or(TERMINATED_DOC_ID);
            if next_block_first_doc < least_id {
                least_id = next_block_first_doc;
            }
            if posting.approximate_upper_bound() > max_score {
                max_score = posting.approximate_upper_bound();
                picked_term = i;
            }
        }

        for posting in self.postings[pivot + 1..].iter() {
            let doc = posting
                .doc()
                .map(|d| d.doc_id())
                .unwrap_or(TERMINATED_DOC_ID);
            if doc < least_id {
                least_id = doc;
            }
        }

        (picked_term, least_id)
    }

    // find the first term that the sum of upper bound of all preceding terms and itself,
    // are greater than or equal to the threshold.
    // returns the least pivot and the max index of the terms that have the same doc id.
    fn find_pivot_term(&self) -> Option<(usize, usize)> {
        if self.operator == Operator::And {
            // for AND query, we always require all terms to be present in the document,
            // so the pivot is always the last term as long as no posting list is exhausted
            if self.postings.len() == self.num_terms {
                return Some((self.num_terms - 1, self.num_terms - 1));
            }
            return None;
        }

        let mut acc = 0.0;
        let mut pivot = None;
        for (idx, posting) in self.postings.iter().enumerate() {
            acc += posting.approximate_upper_bound();
            if acc >= self.threshold {
                pivot = Some(idx);
                break;
            }
        }
        let pivot = pivot?;
        let mut max_pivot = pivot;
        let doc_id = self.postings[pivot].doc().unwrap().doc_id();
        while max_pivot + 1 < self.postings.len()
            && self.postings[max_pivot + 1].doc().unwrap().doc_id() == doc_id
        {
            max_pivot += 1;
        }

        Some((pivot, max_pivot))
    }

    // pick the term that has the maximum upper bound and the current doc id is less than the given doc id
    // so that we can move the posting iterator to the next doc id that is possible to be candidate
    fn move_term(&mut self, picked_term: usize, least_id: u64) {
        self.postings[picked_term].next(least_id);
        let doc_id = self.postings[picked_term]
            .doc()
            .map(|d| d.doc_id())
            .unwrap_or(TERMINATED_DOC_ID);
        if doc_id == TERMINATED_DOC_ID {
            self.postings.swap_remove(picked_term);
        }
        self.bubble_up(picked_term);
    }

    fn check_pivot_aligned(&mut self, pivot: usize, pivot_doc: u64) -> bool {
        for i in (0..=pivot).rev() {
            self.postings[i].next(pivot_doc);
            let doc_id = self.postings[i]
                .doc()
                .map(|d| d.doc_id())
                .unwrap_or(TERMINATED_DOC_ID);
            if doc_id != pivot_doc {
                if doc_id == TERMINATED_DOC_ID {
                    self.postings.swap_remove(i);
                }
                self.bubble_up(i);
                return false;
            } else {
                self.bubble_up(i);
            }
        }
        true
    }

    fn move_preceding(&mut self, pivot: usize, least_id: u64) {
        for i in 0..=pivot {
            self.postings[i].next(least_id);
        }

        let mut i = 0;
        while i < self.postings.len() {
            if self.postings[i].doc().is_none() {
                self.postings.swap_remove(i);
            } else {
                i += 1;
            }
        }
        self.postings.sort_unstable();
    }

    fn bubble_up(&mut self, index: usize) {
        if index >= self.postings.len() {
            return;
        }

        for i in index + 1..self.postings.len() {
            if self.postings[i].cmp(&self.postings[i - 1]) >= std::cmp::Ordering::Equal {
                break;
            }
            self.postings.swap(i - 1, i);
        }
    }

    fn check_positions(&self, slop: i32) -> bool {
        if slop == 0 {
            return self.check_exact_positions();
        }

        let mut position_iters = self
            .postings
            .iter()
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
            .postings
            .iter()
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
}
