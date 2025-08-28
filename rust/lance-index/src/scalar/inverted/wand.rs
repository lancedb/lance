// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Deref;
use std::sync::{Arc, LazyLock};
use std::{cell::UnsafeCell, collections::BinaryHeap};
use std::{cmp::Reverse, fmt::Debug};

use arrow::array::AsArray;
use arrow::datatypes::{Int32Type, UInt32Type};
use arrow_array::{Array, UInt32Array};
use arrow_schema::DataType;
use itertools::Itertools;
use lance_core::utils::address::RowAddress;
use lance_core::utils::mask::RowIdMask;
use lance_core::Result;

use crate::metrics::MetricsCollector;

use super::{
    builder::ScoredDoc,
    encoding::{decompress_positions, decompress_posting_block, decompress_posting_remainder},
    query::FtsSearchParams,
    scorer::Scorer,
    DocSet, PostingList, RawDocInfo,
};
use super::{builder::BLOCK_SIZE, DocInfo};
use super::{
    query::Operator,
    scorer::{idf, K1},
};

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
    index: usize,
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
}

impl CompressedState {
    fn new() -> Self {
        Self {
            block_idx: 0,
            doc_ids: Vec::with_capacity(BLOCK_SIZE),
            freqs: Vec::with_capacity(BLOCK_SIZE),
            buffer: Box::new([0; BLOCK_SIZE]),
        }
    }

    #[inline]
    fn decompress(&mut self, block: &[u8], block_idx: usize, num_blocks: usize, length: u32) {
        self.doc_ids.clear();
        self.freqs.clear();

        let remainder = length as usize % BLOCK_SIZE;
        if block_idx + 1 == num_blocks && remainder != 0 {
            decompress_posting_remainder(block, remainder, &mut self.doc_ids, &mut self.freqs);
        } else {
            decompress_posting_block(block, &mut self.buffer, &mut self.doc_ids, &mut self.freqs);
        }
        self.block_idx = block_idx;
    }
}

impl Debug for PostingIterator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostingIterator")
            .field("token_id", &self.token_id)
            .field("position", &self.position)
            .field("index", &self.index)
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
            approximate_upper_bound,
            compressed: is_compressed.then(|| UnsafeCell::new(CompressedState::new())),
        }
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
                debug_assert!(self.compressed.is_some());
                // this method is called very frequently, so we prefer to use `UnsafeCell` instead of `RefCell`
                // to avoid the overhead of runtime borrow checking
                let compressed = unsafe {
                    let compressed = self.compressed.as_ref().unwrap();
                    &mut *compressed.get()
                };
                let block_idx = self.index / BLOCK_SIZE;
                let block_offset = self.index % BLOCK_SIZE;
                if compressed.block_idx != block_idx || compressed.doc_ids.is_empty() {
                    let block = list.blocks.value(block_idx);
                    compressed.decompress(block, block_idx, list.blocks.len(), list.length);
                }

                // Read from the decompressed block
                let doc_id = compressed.doc_ids[block_offset];
                let frequency = compressed.freqs[block_offset];
                let doc = DocInfo::Raw(RawDocInfo { doc_id, frequency });
                Some(doc)
            }
            PostingList::Plain(ref list) => Some(DocInfo::Located(list.doc(self.index))),
        }
    }

    fn positions(&self) -> Option<Arc<dyn Array>> {
        match self.list {
            PostingList::Plain(ref list) => list.positions(self.index),
            PostingList::Compressed(ref list) => list.positions.as_ref().map(|p| {
                let positions = p.value(self.index);
                let positions = decompress_positions(positions.as_binary());
                Arc::new(UInt32Array::from(positions)) as Arc<dyn Array>
            }),
        }
    }

    // move to the next doc id that is greater than or equal to least_id
    fn next(&mut self, least_id: u64) {
        match self.list {
            PostingList::Compressed(ref mut list) => {
                debug_assert!(least_id <= u32::MAX as u64);
                let least_id = least_id as u32;
                let mut block_idx = self.index / BLOCK_SIZE;
                while block_idx + 1 < list.blocks.len()
                    && list.block_least_doc_id(block_idx + 1) <= least_id
                {
                    block_idx += 1;
                }
                self.index = self.index.max(block_idx * BLOCK_SIZE);
                let length = self.list.len();
                while self.index < length && (self.doc().unwrap().doc_id() as u32) < least_id {
                    self.index += 1;
                }
            }
            PostingList::Plain(ref list) => {
                self.index += list.row_ids[self.index..].partition_point(|&id| id < least_id);
            }
        }
    }

    fn shallow_next(&mut self, least_id: u64) {
        match self.list {
            PostingList::Compressed(ref mut list) => {
                debug_assert!(least_id <= u32::MAX as u64);
                let least_id = least_id as u32;
                let mut block_idx = self.index / BLOCK_SIZE;
                while block_idx + 1 < list.blocks.len()
                    && list.block_least_doc_id(block_idx + 1) <= least_id
                {
                    block_idx += 1;
                }
                self.index = self.index.max(block_idx * BLOCK_SIZE);
            }
            PostingList::Plain(ref list) => {
                // we don't have block max score for legacy index,
                // and no compression, so just do the same as `next(least_id)`
                self.index += list.row_ids[self.index..].partition_point(|&id| id < least_id);
            }
        }
    }

    #[inline]
    fn block_max_score(&self) -> f32 {
        match self.list {
            PostingList::Compressed(ref list) => list.block_max_score(self.index / BLOCK_SIZE),
            PostingList::Plain(_) => self.approximate_upper_bound,
        }
    }

    fn block_first_doc(&self) -> Option<u64> {
        match self.list {
            PostingList::Compressed(ref list) => {
                let block_idx = self.index / BLOCK_SIZE;
                Some(list.block_least_doc_id(block_idx) as u64)
            }
            PostingList::Plain(ref plain) => plain.row_ids.get(self.index).cloned(),
        }
    }

    #[inline]
    fn next_block_first_doc(&self) -> Option<u64> {
        match self.list {
            PostingList::Compressed(ref list) => {
                let next_block = self.index / BLOCK_SIZE + 1;
                if next_block >= list.blocks.len() {
                    return None;
                }
                Some(list.block_least_doc_id(next_block) as u64)
            }
            PostingList::Plain(ref plain) => plain.row_ids.get(self.index + 1).cloned(),
        }
    }
}

pub struct DocCandidate {
    pub row_id: u64,
    pub freqs: Vec<(String, u32)>,
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
        mask: Arc<RowIdMask>,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<DocCandidate>> {
        let limit = params.limit.unwrap_or(usize::MAX);
        if limit == 0 {
            return Ok(vec![]);
        }

        match (mask.max_len(), mask.iter_ids()) {
            (Some(num_rows_matched), Some(row_ids))
                if num_rows_matched * 100
                    <= FLAT_SEARCH_PERCENT_THRESHOLD.deref() * self.docs.len() as u64 =>
            {
                return self.flat_search(params, row_ids, metrics);
            }
            _ => {}
        }

        let mut candidates = BinaryHeap::new();
        let mut num_comparisons = 0;
        while let Some((pivot, doc)) = self.next()? {
            if let Some(cur_doc) = self.cur_doc {
                if cur_doc.doc_id() >= doc.doc_id() {
                    continue;
                }
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
            let freqs = self
                .iter_token_freqs(pivot)
                .map(|(token, freq)| (token.to_owned(), freq))
                .collect();
            if candidates.len() < limit {
                candidates.push(Reverse((ScoredDoc::new(row_id, score), freqs, doc_length)));
                if candidates.len() == limit {
                    self.threshold = candidates.peek().unwrap().0 .0.score.0 * params.wand_factor;
                }
            } else if score > candidates.peek().unwrap().0 .0.score.0 {
                candidates.pop();
                candidates.push(Reverse((ScoredDoc::new(row_id, score), freqs, doc_length)));
                self.threshold = candidates.peek().unwrap().0 .0.score.0 * params.wand_factor;
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

            if let Some(least_id) = self.postings[0].block_first_doc() {
                if least_id > doc_id {
                    current_doc = least_id;
                    continue;
                }
            }
            self.move_shallow(self.postings.len() - 1, doc_id);
            let mut pivot = 0;
            while pivot + 1 < self.postings.len() {
                if let Some(block_doc_id) = self.postings[pivot + 1].block_first_doc() {
                    if block_doc_id > doc_id {
                        break;
                    }
                } else {
                    break;
                }
                pivot += 1;
            }

            if !self.check_block_max(pivot) {
                // the current block max score is less than the threshold,
                // which means we have to skip at least the current block
                if let Some(least_id) = self.get_new_candidate(pivot) {
                    current_doc = std::cmp::max(doc_id, least_id);
                    continue;
                } else {
                    break;
                }
            }

            // move all postings to this doc id
            if !self.check_pivot_aligned(pivot, doc_id) {
                if self.postings.is_empty() {
                    break;
                } else {
                    continue;
                }
            }

            pivot = 0;
            while pivot + 1 < self.postings.len()
                && self.postings[pivot + 1].doc().map(|d| d.doc_id()) == Some(doc_id)
            {
                pivot += 1;
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

            let score = self.score(pivot, doc_length);
            let freqs = self
                .iter_token_freqs(pivot)
                .map(|(token, freq)| (token.to_owned(), freq))
                .collect();

            if candidates.len() < limit {
                candidates.push(Reverse((ScoredDoc::new(row_id, score), freqs, doc_length)));
                if candidates.len() == limit {
                    self.threshold = candidates.peek().unwrap().0 .0.score.0 * params.wand_factor;
                }
            } else if score > candidates.peek().unwrap().0 .0.score.0 {
                candidates.pop();
                candidates.push(Reverse((ScoredDoc::new(row_id, score), freqs, doc_length)));
                self.threshold = candidates.peek().unwrap().0 .0.score.0 * params.wand_factor;
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

    // find the next doc candidate
    fn next(&mut self) -> Result<Option<(usize, DocInfo)>> {
        while let Some((pivot, max_pivot)) = self.find_pivot_term() {
            let posting = &self.postings[pivot];
            let doc = posting.doc().unwrap();
            let doc_id = doc.doc_id();

            self.move_shallow(max_pivot, doc_id);

            if !self.check_block_max(max_pivot) {
                // the current block max score is less than the threshold,
                // which means we have to skip at least the current block
                if let Some(least_id) = self.get_new_candidate(max_pivot) {
                    self.move_term(least_id);
                    continue;
                } else {
                    // no more candidates, so we can stop
                    return Ok(None);
                }
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

    fn check_block_max(&self, pivot: usize) -> bool {
        let mut sum = 0.0;
        for posting in self.postings[..=pivot].iter() {
            sum += posting.block_max_score();
        }
        sum > self.threshold
    }

    fn get_new_candidate(&self, pivot: usize) -> Option<u64> {
        let mut least_id = None;
        for posting in self.postings[..=pivot].iter() {
            if let Some(doc) = posting.next_block_first_doc() {
                if least_id.is_none() || doc < least_id.unwrap() {
                    least_id = Some(doc);
                }
            }
        }
        least_id
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
    fn move_term(&mut self, least_id: u64) {
        let picked = self.pick_term(least_id);
        self.postings[picked].next(least_id);
        let doc_id = self.postings[picked]
            .doc()
            .map(|d| d.doc_id())
            .unwrap_or(TERMINATED_DOC_ID);
        if doc_id == TERMINATED_DOC_ID {
            self.postings.swap_remove(picked);
        }
        self.bubble_up(picked);
    }

    // move the posting iterators preceding the pivot to the block that contains the least_id
    fn move_shallow(&mut self, pivot: usize, least_id: u64) {
        for posting in self.postings[..=pivot].iter_mut() {
            posting.shallow_next(least_id);
        }
    }

    fn check_pivot_aligned(&mut self, pivot: usize, least_id: u64) -> bool {
        for i in (0..=pivot).rev() {
            self.postings[i].next(least_id);
            let doc_id = self.postings[i]
                .doc()
                .map(|d| d.doc_id())
                .unwrap_or(TERMINATED_DOC_ID);
            if doc_id > least_id {
                if doc_id == TERMINATED_DOC_ID {
                    self.postings.swap_remove(i);
                }
                self.bubble_up(i);
                return false;
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

        let doc_id = self.postings[index].doc().unwrap().doc_id();
        for i in index + 1..self.postings.len() {
            if self.postings[i].doc().unwrap().doc_id() >= doc_id {
                break;
            }
            self.postings.swap(i - 1, i);
        }
    }

    fn pick_term(&self, least_id: u64) -> usize {
        let mut least_length = usize::MAX;
        let mut pick_index = 0;
        for (i, posting) in self.postings.iter().enumerate() {
            if let Some(doc) = posting.doc() {
                if doc.doc_id() >= least_id {
                    continue;
                }
            }
            // a shorter posting list means this term is rare and more likely to skip more documents,
            // so we prefer the term with a shorter posting list.
            if posting.list.len() < least_length {
                least_length = posting.list.len();
                pick_index = i;
            }
        }
        pick_index
    }

    fn check_positions(&self, slop: i32) -> bool {
        let mut position_iters = self
            .postings
            .iter()
            .map(|posting| {
                PositionIterator::new(
                    posting.positions().expect("positions must exist"),
                    posting.position as i32,
                )
            })
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
                iter.next(max_relative_pos.unwrap());
            });
        }
    }
}

#[derive(Debug)]
struct PositionIterator {
    // It's Int32Array for legacy index,
    // UInt32Array for new index
    positions: Arc<dyn Array>,
    pub position_in_query: i32,
    index: usize,
}

impl PositionIterator {
    fn new(positions: Arc<dyn Array>, position_in_query: i32) -> Self {
        let mut iter = Self {
            positions,
            position_in_query,
            index: 0,
        };
        iter.next(0);
        iter
    }

    // get the current relative position
    fn relative_position(&self) -> Option<i32> {
        if self.index < self.positions.len() {
            match self.positions.data_type() {
                DataType::Int32 => Some(
                    self.positions.as_primitive::<Int32Type>().value(self.index)
                        - self.position_in_query,
                ),
                DataType::UInt32 => Some(
                    self.positions
                        .as_primitive::<UInt32Type>()
                        .value(self.index) as i32
                        - self.position_in_query,
                ),
                _ => {
                    unreachable!("position iterator only supports Int32 and UInt32");
                }
            }
        } else {
            None
        }
    }

    // move to the next position that the relative position is greater than or equal to least_pos
    fn next(&mut self, least_relative_pos: i32) {
        let least_pos = least_relative_pos + self.position_in_query;
        self.index = match self.positions.data_type() {
            DataType::Int32 => self
                .positions
                .as_primitive::<Int32Type>()
                .values()
                .partition_point(|&pos| pos < least_pos),
            DataType::UInt32 => self
                .positions
                .as_primitive::<UInt32Type>()
                .values()
                .partition_point(|&pos| (pos as i32) < least_pos),
            _ => unreachable!("position iterator only supports Int32 and UInt32"),
        };
    }
}

#[cfg(test)]
mod tests {
    use arrow::buffer::ScalarBuffer;
    use rstest::rstest;

    use crate::{
        metrics::NoOpMetricsCollector,
        scalar::inverted::{
            encoding::compress_posting_list, scorer::BM25Scorer, CompressedPostingList,
            PlainPostingList,
        },
    };

    use super::*;

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

        let bm25 = BM25Scorer::new(std::iter::empty());
        let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, bm25);
        // This should trigger the bug when the second posting list becomes empty
        let result = wand
            .search(
                &FtsSearchParams::default(),
                Arc::new(RowIdMask::default()),
                &NoOpMetricsCollector,
            )
            .unwrap();
        assert_eq!(result.len(), 0); // Should not panic
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

        let bm25 = BM25Scorer::new(std::iter::empty());
        let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, bm25);

        // set a threshold that the sum of max scores can hit,
        // but the sum of block max scores is less than the threshold,
        wand.threshold = 1.5;

        let result = wand.search(
            &FtsSearchParams::default(),
            Arc::new(RowIdMask::default()),
            &NoOpMetricsCollector,
        );

        assert!(result.is_ok());
    }
}
