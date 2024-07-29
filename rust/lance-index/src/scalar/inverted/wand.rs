// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::cmp::{min, Reverse};
use std::collections::BinaryHeap;
use std::sync::Arc;

use itertools::Itertools;
use lance_core::utils::mask::RowIdMask;
use lance_core::Result;
use lazy_static::lazy_static;
use tracing::instrument;

use super::builder::OrderedDoc;
use super::index::{idf, PostingListReader, K1};
use super::Block;

// WAND parameters
// One block consists of rows of a posting list (row id (u64) and frequency (f32)),
// Increasing the block size can decrease the memory usage, but also decrease the probability of skipping blocks.
lazy_static! {
    pub static ref POSTING_BLOCK_SIZE: usize = std::env::var("POSTING_BLOCK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    static ref MIN_IO_SIZE: usize = std::env::var("MIN_IO_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024 * 1024); // 1MiB
    static ref FACTOR: f32 = std::env::var("WAND_FACTOR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
}

// we might change the block size in the future
// and it could be a function of the total number of documents
#[inline]
pub fn block_size(_length: usize) -> usize {
    *POSTING_BLOCK_SIZE
}

// the number of blocks to read in one IO operation
#[inline]
pub fn num_blocks_to_read(num_blocks: usize, block_size: usize) -> usize {
    min(num_blocks, *MIN_IO_SIZE / (block_size * 8)) // the widest column is u64, so 8 bytes
}

#[derive(Clone)]
pub struct PostingIterator {
    token_id: u32,
    list: PostingListReader,
    index: usize,
    doc: (u64, f32),
    mask: Arc<RowIdMask>,
    approximate_upper_bound: f32,
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
    #[instrument(level = "debug", name = "posting_iter_next", skip(self))]
    async fn next(&mut self, least_id: u64) -> Result<Option<(u64, usize)>> {
        // skip blocks
        let block_row_ids = self.list.block_row_ids();
        let block_size = block_size(self.list.len());

        // the binary search version of skipping blocks,
        // I didn't see obvious performance improvement, so commented it out,
        // might be useful after benchmarking with large datasets
        //
        let span = tracing::span!(tracing::Level::DEBUG, "skip_blocks");
        let _guard = span.enter();
        let start = self.index / block_size;
        let mut current_block =
            start + block_row_ids[start..].partition_point(|&row_id| row_id <= least_id) - 1;
        if current_block > start {
            self.index = current_block * block_size;
            self.doc = self.list.block_head_element(current_block);
            // if the first row id of the current block is greater than or equal to least_id,
            // and it's not filtered out,
            // return it directly to avoid IO
            if self.doc.0 >= least_id && self.mask.selected(self.doc.0) {
                return Ok(Some((self.doc.0, self.index)));
            }
        }

        // if the next block is with head_row_id <= least_id,
        // then we can skip the current block
        // let start_block = self.index / block_size;
        // let mut current_block = start_block;
        // while current_block + 1 < self.list.num_blocks()
        //     && block_row_ids[current_block + 1] <= least_id
        // {
        //     current_block += 1;
        // }
        // if current_block > start_block {
        //     self.index = current_block * block_size;
        //     self.doc = self.list.block_head_element(current_block);
        //     // if the first row id of the current block is greater than or equal to least_id,
        //     // and it's not filtered out,
        //     // return it directly to avoid IO
        //     if self.doc.0 >= least_id && self.mask.selected(self.doc.0) {
        //         return Ok(Some((self.doc.0, self.index)));
        //     }
        // }
        std::mem::drop(_guard);

        // read the current block all into memory and do linear search
        let span = tracing::span!(tracing::Level::DEBUG, "search_blocks");
        let _guard = span.enter();

        self.list.try_fetch_blocks(current_block).await?;
        let block = self.list.block(current_block);
        // do binary search in the first block
        let block_offset = current_block * block_size;
        let in_block_index =
            block.row_ids[self.index - block_offset..].partition_point(|&row_id| row_id < least_id);
        if let Some((index, doc)) = self.go_through_block(block, in_block_index) {
            self.index = index;
            self.doc = doc;
            return Ok(Some((doc.0, index)));
        }

        // the remaining blocks must be with first row id greater than least_id
        // so just do linear search to find the first row id that is not filtered out
        self.index = block_offset + block.len();
        while self.index < self.list.len() {
            current_block += 1;
            self.doc = self.list.block_head_element(current_block);
            if self.mask.selected(self.doc.0) {
                // the next block must be with first row id greater than least_id
                // so return it directly
                return Ok(Some((self.doc.0, self.index)));
            }

            self.list.try_fetch_blocks(current_block).await?;
            let block = self.list.block(current_block);
            if let Some((index, doc)) = self.go_through_block(block, in_block_index) {
                self.index = index;
                self.doc = doc;
                return Ok(Some((doc.0, index)));
            }
            self.index += block.len();
        }
        Ok(None)
    }

    #[instrument(level = "debug", skip_all)]
    fn go_through_block(
        &self,
        block: &Block,
        mut block_index: usize,
    ) -> Option<(usize, (u64, f32))> {
        let mut index = self.index;
        while block_index < block.len() {
            let doc = block.doc(block_index);
            if self.mask.selected(doc.0) {
                return Some((index, doc));
            }

            block_index += 1;
            index += 1;
        }
        None
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
            factor: *FACTOR,
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
            let (cur_doc, freq) = posting.doc().unwrap();
            if cur_doc > doc {
                // the posting list is sorted by its current doc id,
                // so we can break early once we find the current doc id is less than the doc id we are looking for
                break;
            }
            debug_assert!(cur_doc == doc);

            score += posting.approximate_upper_bound() * scorer(doc, freq);
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
    async fn move_terms<'b>(&mut self, least_id: u64) -> Result<()> {
        // let mut move_tasks = Vec::new();
        // for posting in &mut self.postings {
        //     match posting.doc() {
        //         Some((d, _)) if d < least_id => {
        //             move_tasks.push(posting.next(least_id));
        //         }
        //         _ => break,
        //     }
        // }
        // futures::future::try_join_all(move_tasks).await?;
        for posting in self.postings.iter_mut() {
            match posting.doc() {
                Some((d, _)) if d < least_id => {
                    posting.next(least_id).await?;
                }
                _ => break,
            }
        }

        self.postings.sort_unstable();
        while let Some(last) = self.postings.last() {
            if last.doc().is_none() {
                self.postings.pop();
            } else {
                break;
            }
        }

        Ok(())
    }
}
