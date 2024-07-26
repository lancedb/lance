// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::mem::size_of;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes;
use arrow_array::RecordBatch;
use itertools::Itertools;
use lance_core::utils::mask::RowIdMask;
use lance_core::{Result, ROW_ID};
use lazy_static::lazy_static;
use tracing::instrument;

use super::builder::OrderedDoc;
use super::index::{idf, PostingListReader, FREQUENCY_COL, K1};

// WAND parameters
// One block consists of rows of a posting list (row id (u64) and frequency (f32)),
// Increasing the block size can decrease the memory usage, but also decrease the probability of skipping blocks.
lazy_static! {
    pub static ref BLOCK_SIZE: usize = std::env::var("BLOCK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
}

// we might change the block size in the future
// and it could be a function of the total number of documents
#[inline]
pub fn block_size(_length: usize) -> usize {
    *BLOCK_SIZE
}

#[inline]
pub fn num_blocks_to_read(block_size: usize) -> usize {
    const MIN_IO_SIZE: usize = 64 * 1024;
    MIN_IO_SIZE / (block_size * (size_of::<u64>() + size_of::<f32>()))
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
    block_id: usize,
    blocks: Vec<RecordBatch>,

    // stats
    skiped_blocks: usize,
    read_block_miss: usize,
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
            (Some((doc1, _)), Some((doc2, _))) => doc1.cmp(&doc2),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
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
        let first_block = list.block_head_element(0);
        let approximate_upper_bound = idf(list.len(), num_doc) * (K1 + 1.0);
        Self {
            token_id,
            list,
            index: 0,
            doc: first_block,
            mask,
            approximate_upper_bound,
            block_id: 0,
            blocks: Vec::new(),
            skiped_blocks: 0,
            read_block_miss: 0,
        }
    }

    #[inline]
    fn approximate_upper_bound(&self) -> f32 {
        match self.doc() {
            Some(_) => self.approximate_upper_bound,
            None => 0.0,
        }
    }

    fn doc(&self) -> Option<(u64, f32)> {
        if self.index < self.list.len() {
            Some(self.doc)
        } else {
            None
        }
    }

    // move to the next row id that is greater than or equal to least_id
    #[instrument(level = "debug", name = "posting_iter_next", skip(self))]
    async fn next(&mut self, least_id: u64) -> Result<Option<(u64, usize)>> {
        // skip blocks
        let block_row_ids = self.list.block_row_ids();
        let block_size = block_size(self.list.len());

        // the binary search version of skipping blocks,
        // I didn't see obvious performance improvement, so commented it out,
        // might be useful after benchmarking with large datasets
        //
        // let start = self.index / block_size;
        // let current_block =
        //     start + block_row_ids[start..].partition_point(|&row_id| row_id <= least_id) - 1;
        // if current_block > start {
        //     self.index = current_block * block_size;
        //     self.doc = self.list.block_head_element(current_block);
        //     // if the first row id of the current block is greater than or equal to least_id,
        //     // and it's not filtered out,
        //     // return it directly to avoid IO
        //     if self.doc.0 >= least_id && self.mask.selected(self.doc.0) {
        //         return Ok(Some((self.doc.0, self.index)));
        //     }
        // }

        // if the next block is with head_row_id <= least_id,
        // then we can skip the current block
        let start_block = self.index / block_size;
        let mut current_block = start_block;
        while current_block + 1 < self.list.num_blocks()
            && block_row_ids[current_block + 1] <= least_id
        {
            current_block += 1;
            self.skiped_blocks += 1;
        }
        if current_block > start_block {
            self.index = current_block * block_size;
            self.doc = self.list.block_head_element(current_block);
            // if the first row id of the current block is greater than or equal to least_id,
            // and it's not filtered out,
            // return it directly to avoid IO
            if self.doc.0 >= least_id && self.mask.selected(self.doc.0) {
                return Ok(Some((self.doc.0, self.index)));
            }
        }

        // read the current block all into memory and do linear search
        let mut batch = self.read_block(current_block).await?;
        let mut row_ids = batch[ROW_ID]
            .as_primitive::<datatypes::UInt64Type>()
            .values();

        let mut block_offset = current_block * block_size;
        loop {
            self.index += 1;
            if self.index >= self.list.len() {
                return Ok(None);
            }

            // switch to the next block
            if self.index == block_offset + block_size {
                self.doc = self.list.block_head_element(current_block + 1);
                if self.mask.selected(self.doc.0) {
                    // the next block must be with first row id greater than least_id
                    // so return it directly
                    return Ok(Some((self.doc.0, self.index)));
                }

                current_block += 1;
                block_offset += block_size;
                batch = self.read_block(current_block).await?;
                row_ids = batch[ROW_ID]
                    .as_primitive::<datatypes::UInt64Type>()
                    .values();
            }

            let row_id = row_ids[self.index - block_offset];
            if row_id >= least_id && self.mask.selected(row_id) {
                let freq = batch[FREQUENCY_COL]
                    .as_primitive::<datatypes::Float32Type>()
                    .values()[self.index - block_offset];
                self.doc = (row_id, freq);
                return Ok(Some((row_id, self.index)));
            }
        }
    }

    #[instrument(level = "debug", skip(self))]
    async fn read_block(&mut self, block_id: usize) -> Result<RecordBatch> {
        if self.blocks.len() > 0
            && self.block_id <= block_id
            && block_id < self.block_id + self.blocks.len()
        {
            return Ok(self.blocks[block_id - self.block_id].clone());
        }

        self.read_block_miss += 1;
        self.block_id = block_id;
        self.blocks = self.list.read_blocks(block_id).await?;
        Ok(self.blocks[0].clone())
    }
}

pub struct Wand {
    factor: f32,
    threshold: f32, // multiple of factor and the minimum score of the top-k documents
    cur_doc: Option<u64>,
    postings: Vec<PostingIterator>,
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
            let score = self.score(doc, &scorer);
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
    fn score(&self, doc: u64, scorer: &impl Fn(u64, f32) -> f32) -> f32 {
        let mut score = 0.0;
        for posting in &self.postings {
            if let Some((cur_doc, freq)) = posting.doc() {
                if cur_doc > doc {
                    // the posting list is sorted by its current doc id,
                    // so we can break early once we find the current doc id is less than the doc id we are looking for
                    break;
                }
                debug_assert!(cur_doc == doc);

                score += posting.approximate_upper_bound() * scorer(doc, freq);
            } else {
                // the posting list is exhausted
                break;
            }
        }
        score
    }

    // find the next doc candidate
    #[instrument(level = "debug", name = "wand_next", skip_all)]
    async fn next(&mut self) -> Result<Option<u64>> {
        self.postings.sort_unstable();
        while let Some(pivot_posting) = self.find_pivot_term() {
            let (doc, _) = pivot_posting
                .doc()
                .expect("pivot posting should have at least one document");

            let cur_doc = self.cur_doc.unwrap_or(0);
            if self.cur_doc.is_some() && doc <= cur_doc {
                // the pivot doc id is less than the current doc id,
                // that means this doc id has been processed before, so skip it
                self.move_terms(cur_doc + 1).await?;
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
                // so move such terms to the doc id
                self.move_terms(doc).await?;
            }
        }
        Ok(None)
    }

    // find the first term that the sum of upper bound of all preceding terms and itself,
    // are greater than or equal to the threshold
    #[instrument(level = "debug", skip_all)]
    fn find_pivot_term(&self) -> Option<&PostingIterator> {
        let mut acc = 0.0;
        for iter in self.postings.iter() {
            acc += iter.approximate_upper_bound();
            if acc >= self.threshold && acc > 0. {
                return Some(iter);
            }
        }
        None
    }

    // pick the term that has the maximum upper bound and the current doc id is less than the given doc id
    // so that we can move the posting iterator to the next doc id that is possible to be candidate
    #[instrument(level = "debug", skip_all)]
    async fn move_terms<'b>(&mut self, least_id: u64) -> Result<()> {
        let mut move_tasks = Vec::new();
        for posting in &mut self.postings {
            match posting.doc() {
                Some((d, _)) if d < least_id => {
                    move_tasks.push(posting.next(least_id));
                }
                _ => break,
            }
        }
        futures::future::try_join_all(move_tasks).await?;

        self.postings.sort_unstable();

        Ok(())
    }
}
