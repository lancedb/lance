// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::BinaryHeap;
use std::{
    cell::{Cell, RefCell},
    sync::Arc,
};
use std::{cmp::Reverse, rc::Rc};

use arrow::array::AsArray;
use arrow::datatypes::{Int32Type, UInt32Type};
use arrow_array::{Array, UInt32Array};
use arrow_schema::DataType;
use lance_core::utils::mask::RowIdMask;
use lance_core::Result;
use tracing::instrument;

use super::{
    builder::OrderedDoc,
    encoding::{decompress_positions, decompress_posting_block, decompress_posting_remainder},
    DocSet, PostingList, RawDocInfo,
};
use super::{builder::BLOCK_SIZE, DocInfo};
use super::{
    query::Operator,
    scorer::{idf, K1},
};

#[derive(Clone)]
pub struct PostingIterator {
    token_id: u32,
    position: u32,
    list: PostingList,
    index: usize,
    approximate_upper_bound: f32,

    // for compressed posting list
    // the posting iterator won't be accessed by multiple threads at the same time,
    // but we need to use `Mutex` to modify the buffer and the decompressed data,
    // and `Rc` to clone.
    block_idx: Cell<usize>,
    decompressed_doc_ids: RefCell<Option<Vec<u32>>>,
    decompressed_frequencies: RefCell<Option<Vec<u32>>>,
    buffer: RefCell<Option<Box<[u32; BLOCK_SIZE]>>>,
}

impl PartialEq for PostingIterator {
    fn eq(&self, other: &Self) -> bool {
        self.token_id == other.token_id
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
            (Some(doc1), Some(doc2)) => doc1.cmp(&doc2).then(
                self.approximate_upper_bound
                    .total_cmp(&other.approximate_upper_bound),
            ),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        }
    }
}

impl PostingIterator {
    pub(crate) fn new(token_id: u32, position: u32, list: PostingList, num_doc: usize) -> Self {
        let approximate_upper_bound = match list.max_score() {
            Some(max_score) => max_score,
            None => idf(list.len(), num_doc) * (K1 + 1.0),
        };

        let (decompressed_doc_ids, decompressed_frequencies, buffer) =
            if matches!(list, PostingList::Compressed(_)) {
                let decompressed_doc_ids = Some(Vec::with_capacity(BLOCK_SIZE));
                let decompressed_frequencies = Some(Vec::with_capacity(BLOCK_SIZE));
                let buffer = Some(Box::new([0; BLOCK_SIZE]));
                (decompressed_doc_ids, decompressed_frequencies, buffer)
            } else {
                (None, None, None)
            };

        Self {
            token_id,
            position,
            list,
            index: 0,
            approximate_upper_bound,
            block_idx: Cell::new(0),
            decompressed_doc_ids: RefCell::new(decompressed_doc_ids),
            decompressed_frequencies: RefCell::new(decompressed_frequencies),
            buffer: RefCell::new(buffer),
        }
    }

    #[inline]
    fn approximate_upper_bound(&self) -> f32 {
        self.approximate_upper_bound
    }

    #[instrument(level = "info", name = "posting_iter_doc", skip_all)]
    fn doc(&self) -> Option<DocInfo> {
        if self.index >= self.list.len() {
            return None;
        }

        match self.list {
            PostingList::Plain(ref list) => Some(DocInfo::Located(list.doc(self.index))),
            PostingList::Compressed(ref list) => {
                let block_idx = self.index / BLOCK_SIZE;
                let block_offset = self.index % BLOCK_SIZE;

                let mut doc_ids = self.decompressed_doc_ids.borrow_mut();
                let doc_ids = doc_ids.as_mut().unwrap();
                let mut freqs = self.decompressed_frequencies.borrow_mut();
                let freqs = freqs.as_mut().unwrap();
                let mut buffer = self.buffer.borrow_mut();
                let buffer = buffer.as_mut().unwrap();
                if self.block_idx.get() != block_idx || doc_ids.len() == 0 {
                    let block = list.blocks.value(block_idx);
                    doc_ids.clear();
                    freqs.clear();

                    let remainder = list.length as usize % BLOCK_SIZE;
                    if block_idx + 1 == list.blocks.len() && remainder != 0 {
                        // the last block is not full
                        decompress_posting_remainder(block, remainder, doc_ids, freqs);
                    } else {
                        decompress_posting_block(block, buffer, doc_ids, freqs);
                    }

                    self.block_idx.set(block_idx);
                }
                let doc_id = doc_ids[block_offset];
                let frequency = freqs[block_offset];
                Some(DocInfo::Raw(RawDocInfo { doc_id, frequency }))
            }
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
    fn next(&mut self, least_id: u64) -> Option<(u64, usize)> {
        match self.list {
            PostingList::Plain(ref list) => {
                self.index += list.row_ids[self.index..].partition_point(|&id| id < least_id);
            }
            PostingList::Compressed(ref mut list) => {
                let least_id = least_id as u32;
                let mut block_idx = self.index / BLOCK_SIZE;
                while block_idx + 1 < list.blocks.len()
                    && list.block_least_doc_id(block_idx + 1) < least_id
                {
                    block_idx += 1;
                }
                self.index = self.index.max(block_idx * BLOCK_SIZE);
                let length = list.length as usize;
                while self.index < length && (self.doc().unwrap().doc_id() as u32) < least_id {
                    self.index += 1;
                }
            }
        }
        None
    }
}

pub struct Wand<'a> {
    threshold: f32, // multiple of factor and the minimum score of the top-k documents
    cur_doc: Option<u64>,
    postings: Vec<PostingIterator>,
    docs: &'a DocSet,
}

// we were using row id as doc id in the past, which is u64,
// but now we are using the index as doc id, which is u32.
// so here WAND is a generic struct that can be used for both u32 and u64 doc ids.
impl<'a> Wand<'a> {
    pub(crate) fn new(
        operator: Operator,
        postings: impl Iterator<Item = PostingIterator>,
        docs: &'a DocSet,
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
            cur_doc: None,
            postings: posting_lists
                .into_iter()
                .filter(|posting| posting.doc().is_some())
                .collect(),
            docs,
        }
    }

    // search the top-k documents that contain the query
    // returns the row ids and their scores
    pub(crate) fn search(
        &mut self,
        is_phrase_query: bool,
        limit: usize,
        mask: Arc<RowIdMask>,
        factor: f32,
        scorer: impl Fn(&DocInfo) -> f32,
    ) -> Result<(Vec<u64>, Vec<f32>)> {
        if limit == 0 {
            return Ok((vec![], vec![]));
        }

        let mut candidates = BinaryHeap::new();

        while let Some(doc) = self.next()? {
            if is_phrase_query && !self.check_positions() {
                continue;
            }

            // if the doc is not located, we need to find the row id
            let row_id = match &doc {
                DocInfo::Located(doc) => doc.row_id,
                DocInfo::Raw(doc) => {
                    // if the doc is not located, we need to find the row id
                    // in the doc set. This is a bit slow, but it should be rare.
                    self.docs.row_id(doc.doc_id as u32)
                }
            };
            if !mask.selected(row_id) {
                continue;
            }
            let score = self.score(&doc, &scorer);
            if candidates.len() < limit {
                candidates.push(Reverse(OrderedDoc::new(row_id, score)));
            } else if score > candidates.peek().unwrap().0.score.0 {
                candidates.pop();
                candidates.push(Reverse(OrderedDoc::new(row_id, score)));
                self.threshold = candidates.peek().unwrap().0.score.0 * factor;
            }
        }

        Ok(candidates
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse(doc)| (doc.row_id, doc.score.0))
            .unzip())
    }

    // calculate the score of the document
    fn score(&self, doc: &DocInfo, scorer: &impl Fn(&DocInfo) -> f32) -> f32 {
        let mut score = 0.0;
        for posting in &self.postings {
            let cur_doc = posting.doc().unwrap();
            if cur_doc.doc_id() > doc.doc_id() {
                // the posting list is sorted by its current doc id,
                // so we can break early once we find the current doc id is less than the doc id we are looking for
                break;
            }
            debug_assert!(cur_doc.doc_id() == doc.doc_id());
            let idf = idf(posting.list.len(), self.docs.len());
            score += idf * scorer(doc);
        }
        score
    }

    // find the next doc candidate
    #[instrument(level = "info", name = "wand_next", skip_all)]
    fn next(&mut self) -> Result<Option<DocInfo>> {
        while let Some(pivot_posting) = self.find_pivot_term() {
            let doc = pivot_posting
                .doc()
                .expect("pivot posting should have at least one document");

            let cur_doc = self.cur_doc.unwrap_or_default();
            if self.cur_doc.is_some() && doc.doc_id() <= cur_doc {
                self.move_term(cur_doc + 1);
            } else if self.postings[0].doc().unwrap().doc_id() == doc.doc_id() {
                // all the posting iterators preceding pivot have reached this doc id,
                // so that means the sum of upper bound of all terms is not less than the threshold,
                // this document is a candidate
                self.cur_doc = Some(doc.doc_id());
                return Ok(Some(doc));
            } else {
                // some posting iterators haven't reached this doc id,
                // so move such terms to the doc id
                self.move_term(doc.doc_id());
            }
        }
        Ok(None)
    }

    // find the first term that the sum of upper bound of all preceding terms and itself,
    // are greater than or equal to the threshold
    #[instrument(level = "debug", skip_all)]
    fn find_pivot_term(&self) -> Option<&PostingIterator> {
        let mut acc = 0.0;
        for posting in self.postings.iter() {
            acc += posting.approximate_upper_bound();
            if acc >= self.threshold {
                return Some(posting);
            }
        }
        None
    }

    // pick the term that has the maximum upper bound and the current doc id is less than the given doc id
    // so that we can move the posting iterator to the next doc id that is possible to be candidate
    #[instrument(level = "debug", skip_all)]
    fn move_term(&mut self, least_id: u64) {
        let picked = self.pick_term(least_id);
        self.postings[picked].next(least_id);

        self.postings.sort_unstable();
        while let Some(last) = self.postings.last() {
            if last.doc().is_none() {
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
                break;
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

    fn check_positions(&self) -> bool {
        let mut position_iters = self
            .postings
            .iter()
            .map(|posting| {
                PositionIterator::new(
                    posting
                        .positions(posting.index)
                        .expect("positions must exist"),
                    posting.position,
                )
            })
            .collect::<Vec<_>>();

        loop {
            let mut max_pos = None;
            let mut all_same = true;
            for iter in &position_iters {
                match (iter.relative_position(), max_pos) {
                    (Some(pos), None) => {
                        max_pos = Some(pos);
                    }
                    (Some(pos), Some(max)) => {
                        if pos > max {
                            max_pos = Some(pos);
                        }
                        if pos != max {
                            all_same = false;
                        }
                    }
                    _ => return false,
                }
            }

            if all_same {
                return true;
            }

            position_iters.iter_mut().for_each(|iter| {
                iter.next(max_pos.unwrap());
            });
        }
    }
}

#[derive(Debug)]
struct PositionIterator {
    // It's Int32Array for legacy index,
    // UInt32Array for new index
    positions: Arc<dyn Array>,
    pub position_in_query: u32,
    index: usize,
}

impl PositionIterator {
    fn new(positions: Arc<dyn Array>, position_in_query: u32) -> Self {
        let mut iter = Self {
            positions,
            position_in_query,
            index: 0,
        };
        iter.next(0);
        iter
    }

    // get the current relative position
    fn relative_position(&self) -> Option<u32> {
        if self.index < self.positions.len() {
            match self.positions.data_type() {
                &DataType::Int32 => Some(
                    self.positions.as_primitive::<Int32Type>().value(self.index) as u32
                        - self.position_in_query,
                ),
                &DataType::UInt32 => Some(
                    self.positions
                        .as_primitive::<UInt32Type>()
                        .value(self.index)
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
    fn next(&mut self, least_relative_pos: u32) {
        let least_pos = least_relative_pos + self.position_in_query;
        self.index = match self.positions.data_type() {
            &DataType::Int32 => self
                .positions
                .as_primitive::<Int32Type>()
                .values()
                .partition_point(|&pos| (pos as u32) < least_pos),
            &DataType::UInt32 => self
                .positions
                .as_primitive::<UInt32Type>()
                .values()
                .partition_point(|&pos| pos < least_pos),
            _ => unreachable!("position iterator only supports Int32 and UInt32"),
        };
    }
}
