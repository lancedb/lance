// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::{cell::UnsafeCell, collections::BinaryHeap};
use std::{cmp::Reverse, fmt::Debug};

use arrow::array::AsArray;
use arrow::datatypes::{Int32Type, UInt32Type};
use arrow_array::{Array, UInt32Array};
use arrow_schema::DataType;
use lance_core::utils::mask::RowIdMask;
use lance_core::Result;
use tracing::instrument;

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

    fn positions(&self, index: usize) -> Option<Arc<dyn Array>> {
        match self.list {
            PostingList::Plain(ref list) => list.positions(index),
            PostingList::Compressed(ref list) => list.positions.as_ref().map(|p| {
                let positions = p.value(index);
                let positions = decompress_positions(positions.as_binary());
                Arc::new(UInt32Array::from(positions)) as Arc<dyn Array>
            }),
        }
    }

    // move to the next doc id that is greater than or equal to least_id
    #[instrument(level = "debug", name = "posting_iter_next", skip(self))]
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

pub struct Wand<'a, S: Scorer> {
    threshold: f32, // multiple of factor and the minimum score of the top-k documents
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
        let threshold = match operator {
            Operator::Or => 0.0,
            Operator::And => posting_lists
                .iter()
                .map(|posting| posting.approximate_upper_bound())
                .sum::<f32>(),
        };

        Self {
            threshold,
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
    ) -> Result<Vec<(u64, u32, u32)>> {
        let limit = params.limit.unwrap_or(usize::MAX);
        if limit == 0 {
            return Ok(vec![]);
        }

        let mut candidates = BinaryHeap::new();
        let mut num_comparisons = 0;
        while let Some((pivot, doc)) = self.next()? {
            self.cur_doc = Some(doc);
            num_comparisons += 1;
            if params.phrase_slop.is_some()
                && !self.check_positions(params.phrase_slop.unwrap() as i32)
            {
                self.move_preceding(pivot, doc.doc_id() + 1);
                continue;
            }

            // if the doc is not located, we need to find the row id
            let row_id = match &doc {
                DocInfo::Raw(doc) => {
                    // if the doc is not located, we need to find the row id
                    // in the doc set. This is a bit slow, but it should be rare.
                    self.docs.row_id(doc.doc_id)
                }
                DocInfo::Located(doc) => doc.row_id,
            };
            if !mask.selected(row_id) {
                self.move_preceding(pivot, doc.doc_id() + 1);
                continue;
            }
            let doc_length = match &doc {
                DocInfo::Raw(doc) => self.docs.num_tokens(doc.doc_id),
                DocInfo::Located(doc) => self.docs.num_tokens_by_row_id(doc.row_id),
            };
            let score = self.score(pivot, doc.doc_id(), doc.frequency(), doc_length);
            if candidates.len() < limit {
                candidates.push(Reverse((
                    ScoredDoc::new(row_id, score),
                    doc.frequency(),
                    doc_length,
                )));
            } else if score > candidates.peek().unwrap().0 .0.score.0 {
                candidates.pop();
                candidates.push(Reverse((
                    ScoredDoc::new(row_id, score),
                    doc.frequency(),
                    doc_length,
                )));
                self.threshold = candidates.peek().unwrap().0 .0.score.0 * params.wand_factor;
            }
            self.move_preceding(pivot, doc.doc_id() + 1);
        }
        metrics.record_comparisons(num_comparisons);

        Ok(candidates
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse((doc, freq, length))| (doc.row_id, freq, length))
            .collect())
    }

    // calculate the score of the document
    fn score(&self, pivot: usize, doc_id: u64, freq: u32, doc_length: u32) -> f32 {
        let mut score = 0.0;
        for posting in self.postings[..=pivot].iter() {
            score += self.scorer.score(&posting.token, freq, doc_length);
        }
        for posting in self.postings[pivot + 1..].iter() {
            let doc = posting.doc().unwrap();
            if doc.doc_id() > doc_id {
                // the posting iterator is sorted by doc id,
                // so we can stop here
                break;
            }
            debug_assert_eq!(doc.doc_id(), doc_id);
            score += self.scorer.score(&posting.token, freq, doc_length);
        }
        score
    }

    // find the next doc candidate
    fn next(&mut self) -> Result<Option<(usize, DocInfo)>> {
        while let Some(pivot) = self.find_pivot_term() {
            let posting = &self.postings[pivot];
            let doc = posting.doc().unwrap();
            let doc_id = doc.doc_id();

            for posting in self.postings[..pivot].iter_mut() {
                posting.shallow_next(doc_id);
            }

            if self.check_block_max(pivot) {
                if self.postings[0].doc().unwrap().doc_id() == doc_id {
                    // all the posting iterators preceding pivot have reached this doc id,
                    // so that means the sum of upper bound of all terms is not less than the threshold,
                    // this document is a candidate
                    return Ok(Some((pivot, doc)));
                } else {
                    self.move_term(doc_id);
                }
            } else {
                // the current block max score is less than the threshold,
                // which means we have to skip at least the current block
                if let Some(least_id) = self.get_new_candidate(pivot) {
                    self.move_term(least_id);
                } else {
                    // no more candidates, so we can stop
                    return Ok(None);
                }
            }
        }
        Ok(None)
    }

    fn check_block_max(&self, pivot: usize) -> bool {
        let mut sum = 0.0;
        for posting in self.postings[..=pivot].iter() {
            sum += posting.block_max_score();
        }
        sum >= self.threshold
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
    // are greater than or equal to the threshold
    #[instrument(level = "debug", skip_all)]
    fn find_pivot_term(&self) -> Option<usize> {
        let mut acc = 0.0;
        let mut pivot = None;
        for (idx, posting) in self.postings.iter().enumerate() {
            acc += posting.approximate_upper_bound();
            if acc >= self.threshold {
                pivot = Some(idx);
                break;
            }
        }
        let mut pivot = pivot?;
        let doc_id = self.postings[pivot].doc().unwrap().doc_id();
        while pivot + 1 < self.postings.len()
            && self.postings[pivot + 1].doc().unwrap().doc_id() == doc_id
        {
            pivot += 1;
        }
        Some(pivot)
    }

    // pick the term that has the maximum upper bound and the current doc id is less than the given doc id
    // so that we can move the posting iterator to the next doc id that is possible to be candidate
    #[instrument(level = "debug", skip_all)]
    fn move_term(&mut self, least_id: u64) {
        let picked = self.pick_term(least_id);
        self.postings[picked].next(least_id);
        if self.postings[picked].empty() {
            self.postings.remove(picked);
        }
        self.postings.sort_unstable();
    }

    fn move_preceding(&mut self, pivot: usize, least_id: u64) {
        for posting in self.postings[..=pivot].iter_mut() {
            posting.next(least_id);
        }

        self.postings.sort_unstable();
        while let Some(posting) = self.postings.last() {
            if posting.empty() {
                self.postings.pop();
            } else {
                break;
            }
        }
    }

    fn pick_term(&self, least_id: u64) -> usize {
        let mut least_length = usize::MAX;
        let mut pick_index = 0;
        for (i, posting) in self.postings.iter().enumerate() {
            let doc = posting.doc().unwrap();
            if doc.doc_id() >= least_id {
                continue;
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
                    posting
                        .positions(posting.index)
                        .expect("positions must exist"),
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
                max_relative_pos = Some(std::cmp::max(last, next) - slop);
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
