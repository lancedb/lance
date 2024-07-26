// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes;
use arrow_array::RecordBatch;
use itertools::Itertools;
use lance_core::utils::mask::RowIdMask;
use lance_core::{Result, ROW_ID};

use crate::vector::graph::OrderedFloat;

use super::builder::OrderedDoc;
use super::index::{idf, PostingListReader, FREQUENCY_COL, K1};

// WAND parameters
pub const MIN_BLOCK_SIZE: usize = 256;
pub const BLOCK_SIZE_SQUARE: usize = MIN_BLOCK_SIZE * MIN_BLOCK_SIZE;

#[inline]
pub fn block_size(length: usize) -> usize {
    match length {
        ..=BLOCK_SIZE_SQUARE => MIN_BLOCK_SIZE,
        _ => (length as f32).sqrt().ceil() as usize,
    }
}

#[derive(Clone)]
pub struct PostingIterator {
    token_id: u32,
    list: PostingListReader,
    index: usize,
    doc: (u64, f32),
    mask: Arc<RowIdMask>,
    approximate_upper_bound: f32,
    // cache the current block
    block_id: Option<usize>,
    block: Option<RecordBatch>,
}

impl PartialEq for PostingIterator {
    fn eq(&self, other: &Self) -> bool {
        self.token_id == other.token_id && self.index == other.index
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
        match self.doc.0.cmp(&other.doc.0) {
            std::cmp::Ordering::Equal => self.token_id.cmp(&other.token_id),
            ord => ord,
        }
    }
}

impl PostingIterator {
    pub(crate) fn new(
        token_id: u32,
        list: PostingListReader,
        num_doc: usize,
        mask: Arc<RowIdMask>,
    ) -> Self {
        let first_block = list.block_last_element(0);
        let approximate_upper_bound = idf(list.len(), num_doc) * (K1 + 1.0);
        Self {
            token_id,
            list,
            index: 0,
            doc: first_block,
            mask,
            approximate_upper_bound,
            block_id: None,
            block: None,
        }
    }

    #[inline]
    fn approximate_upper_bound(&self) -> f32 {
        self.approximate_upper_bound
    }

    fn doc(&self) -> Option<(u64, f32)> {
        if self.index < self.list.len() {
            Some(self.doc)
        } else {
            None
        }
    }

    // move to the next row id that is greater than or equal to least_id
    async fn next(&mut self, least_id: u64) -> Result<Option<(u64, usize)>> {
        // skip blocks
        let block_row_ids = self.list.block_row_ids();
        let block_size = block_size(self.list.len());
        let mut current_block = self.index / block_size;
        while current_block + 1 < self.list.num_blocks()
            && block_row_ids[current_block + 1] <= least_id
        {
            current_block += 1;
            self.index = current_block * block_size;
            if self.index < self.list.len() {
                self.doc = self.list.block_last_element(current_block);
            }
        }

        // if the first row id of the current block is greater than or equal to least_id
        // return it directly to avoid IO
        if self.index == current_block * block_size && block_row_ids[current_block] >= least_id {
            return Ok(Some((block_row_ids[current_block], self.index)));
        }

        // read the current block all into memory and do linear search
        let mut batch = self.read_block(current_block).await?;
        let mut row_ids = batch[ROW_ID]
            .as_primitive::<datatypes::UInt64Type>()
            .values();

        let mut block_offset = current_block * block_size;
        while self.index < self.list.len() {
            let row_id = row_ids[self.index - block_offset];
            if row_id >= least_id && self.mask.selected(row_id) {
                let freq = batch[FREQUENCY_COL]
                    .as_primitive::<datatypes::Float32Type>()
                    .values()[self.index - block_offset];
                self.doc = (row_id, freq);
                return Ok(Some((row_id, self.index)));
            }
            self.index += 1;
            if self.index == block_offset + block_size {
                current_block += 1;
                block_offset += block_size;
                batch = self.read_block(current_block).await?;
                row_ids = batch[ROW_ID]
                    .as_primitive::<datatypes::UInt64Type>()
                    .values();
            }
        }
        Ok(None)
    }

    async fn read_block(&mut self, block_id: usize) -> Result<RecordBatch> {
        match self.block_id {
            Some(id) if id == block_id => Ok(self.block.as_ref().unwrap().clone()),
            _ => {
                let block = self.list.read_block(block_id).await?;
                self.block_id = Some(block_id);
                self.block = Some(block.clone());
                Ok(block)
            }
        }
    }
}

pub struct Wand {
    factor: f32,
    threshold: f32, // multiple of factor and the minimum score of the top-k documents
    cur_doc: Option<u64>,
    postings: BTreeSet<PostingIterator>,
    candidates: BinaryHeap<Reverse<OrderedDoc>>,
}

impl Wand {
    pub(crate) fn new(postings: impl Iterator<Item = PostingIterator>) -> Self {
        Self {
            factor: 1.0,
            threshold: 0.0,
            cur_doc: None,
            postings: postings.collect(),
            candidates: BinaryHeap::new(),
        }
    }

    // search the top-k documents that contain the query
    pub(crate) async fn search(
        &mut self,
        k: usize,
        scorer: impl Fn(u64, f32) -> f32,
    ) -> Result<Vec<(u64, f32)>> {
        if k == 0 {
            return Ok(vec![]);
        }

        while let Some(doc) = self.next().await? {
            let score = self.score(doc, &scorer).await?;
            if self.candidates.len() < k {
                self.candidates.push(Reverse(OrderedDoc::new(doc, score)));
            } else if score > self.threshold {
                self.candidates.pop();
                self.candidates.push(Reverse(OrderedDoc::new(doc, score)));
                self.threshold = self.candidates.peek().unwrap().0.score.0 * self.factor;
            }
        }

        Ok(self
            .candidates
            .iter()
            .map(|doc| (doc.0.row_id, doc.0.score))
            .sorted_unstable()
            .map(|(row_id, score)| (row_id, score.0))
            .collect())
    }

    // calculate the score of the document
    async fn score(&self, doc: u64, scorer: &impl Fn(u64, f32) -> f32) -> Result<f32> {
        let mut score = 0.0;
        for posting in &self.postings {
            let (cur_doc, freq) = posting.doc().expect("empty posting list was removed");
            if cur_doc > doc {
                // the posting list is sorted by its current doc id,
                // so we can break early once we find the current doc id is greater than the doc id we are looking for
                break;
            }
            debug_assert!(cur_doc == doc);

            score += posting.approximate_upper_bound() * scorer(doc, freq);
        }
        Ok(score)
    }

    // find the next doc candidate
    async fn next(&mut self) -> Result<Option<u64>> {
        while let Some((index, pivot_posting)) = self.find_pivot_term() {
            let (doc, _) = pivot_posting
                .doc()
                .expect("pivot posting should have at least one document");

            let cur_doc = self.cur_doc.unwrap_or(0);
            if self.cur_doc.is_some() && doc <= cur_doc {
                // the pivot doc id is less than the current doc id,
                // that means this doc id has been processed before, so skip it
                let posting = self.pick_term(cur_doc + 1, self.postings.iter().take(index + 1));
                let mut posting = self
                    .postings
                    .take(&posting)
                    .expect("we just found it in the previous step");
                if posting.next(cur_doc + 1).await?.is_some() {
                    self.postings.insert(posting);
                }
            } else if self
                .postings
                .first()
                .and_then(|posting| posting.doc().map(|(d, _)| d))
                .expect("the postings can't be empty")
                == doc
            {
                // all the posting iterators have reached this doc id,
                // so that means the sum of upper bound of all terms is not less than the threshold,
                // this document is a candidate
                self.cur_doc = Some(doc);
                return Ok(Some(doc));
            } else {
                // some posting iterators haven't reached this doc id,
                // so pick one of such term(s) and move it to the doc id
                let posting = self.pick_term(doc, self.postings.iter().take(index));
                let mut posting = self
                    .postings
                    .take(&posting)
                    .expect("we just found it in the previous step");
                if posting.next(doc).await?.is_some() {
                    self.postings.insert(posting);
                }
            }
        }
        Ok(None)
    }

    // find the first term that the sum of upper bound of all preceding terms and itself,
    // are greater than or equal to the threshold
    fn find_pivot_term(&self) -> Option<(usize, PostingIterator)> {
        let mut acc = 0.0;
        for (i, iter) in self.postings.iter().enumerate() {
            acc += iter.approximate_upper_bound();
            if acc >= self.threshold {
                return Some((i, iter.clone()));
            }
        }
        None
    }

    // pick the term that has the maximum upper bound and the current doc id is less than the given doc id
    // so that we can move the posting iterator to the next doc id that is possible to be candidate
    fn pick_term<'b>(
        &self,
        doc: u64,
        postings: impl Iterator<Item = &'b PostingIterator>,
    ) -> PostingIterator {
        postings
            .filter(|posting| posting.doc().unwrap().0 < doc)
            .max_by_key(|posting| OrderedFloat(posting.approximate_upper_bound()))
            .expect("at least one posting list")
            .clone()
    }
}
