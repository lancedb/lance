// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};
use std::sync::Arc;

use itertools::Itertools;
use lance_core::utils::mask::RowIdMask;

use crate::vector::graph::OrderedFloat;

use super::{idf, OrderedDoc, PostingList, K1};

#[derive(Clone)]
pub(crate) struct PostingIterator<'a> {
    token_id: u32,
    list: &'a PostingList,
    index: usize,
    mask: Arc<RowIdMask>,
    approximate_upper_bound: f32,
}

impl PartialEq for PostingIterator<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.token_id == other.token_id && self.index == other.index
    }
}

impl Eq for PostingIterator<'_> {}

impl PartialOrd for PostingIterator<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PostingIterator<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.doc().map(|d| d.0).cmp(&other.doc().map(|d| d.0)) {
            std::cmp::Ordering::Equal => self.token_id.cmp(&other.token_id),
            ord => ord,
        }
    }
}

impl<'a> PostingIterator<'a> {
    pub(crate) fn new(
        token_id: u32,
        list: &'a PostingList,
        num_doc: usize,
        mask: Arc<RowIdMask>,
    ) -> Option<Self> {
        Some(Self {
            token_id,
            list,
            index: 0,
            mask,
            approximate_upper_bound: idf(list.len(), num_doc) * (K1 + 1.0),
        })
    }

    #[inline]
    fn approximate_upper_bound(&self) -> f32 {
        self.approximate_upper_bound
    }

    fn doc(&self) -> Option<(u64, f32)> {
        if self.index < self.list.len() {
            Some((
                self.list.row_ids[self.index],
                self.list.frequencies[self.index],
            ))
        } else {
            None
        }
    }

    // move to the next row id that is greater than or equal to least_id
    fn next(&mut self, least_id: u64) -> Option<(u64, usize)> {
        let block_size = ((self.list.len() - self.index) as f32).sqrt().ceil() as usize;
        // skip blocks
        while self.index + block_size < self.list.len()
            && self.list.row_ids[self.index + block_size] < least_id
        {
            self.index += block_size;
        }
        // linear search
        while self.index < self.list.len() {
            let row_id = self.list.row_ids[self.index];
            if row_id >= least_id && self.mask.selected(row_id) {
                return Some((row_id, self.index));
            }
            self.index += 1;
        }
        None
    }
}

pub(crate) struct Wand<'a> {
    factor: f32,
    threshold: f32, // multiple of factor and the minimum score of the top-k documents
    cur_doc: Option<u64>,
    postings: BTreeSet<PostingIterator<'a>>,
    candidates: BinaryHeap<Reverse<OrderedDoc>>,
}

impl<'a> Wand<'a> {
    pub(crate) fn new(postings: impl Iterator<Item = PostingIterator<'a>>) -> Self {
        Self {
            factor: 1.0,
            threshold: 0.0,
            cur_doc: None,
            postings: postings.collect(),
            candidates: BinaryHeap::new(),
        }
    }

    // search the top-k documents that contain the query
    pub(crate) fn search(&mut self, k: usize, scorer: impl Fn(u64, f32) -> f32) -> Vec<(u64, f32)> {
        if k == 0 {
            return vec![];
        }

        while let Some(doc) = self.next() {
            let score = self.score(doc, &scorer);
            if self.candidates.len() < k {
                self.candidates.push(Reverse(OrderedDoc::new(doc, score)));
            } else if score > self.threshold {
                self.candidates.pop();
                self.candidates.push(Reverse(OrderedDoc::new(doc, score)));
                self.threshold = self.candidates.peek().unwrap().0.score.0 * self.factor;
            }
        }

        self.candidates
            .iter()
            .map(|doc| (doc.0.row_id, doc.0.score))
            .sorted_unstable()
            .map(|(row_id, score)| (row_id, score.0))
            .collect()
    }

    // calculate the score of the document
    fn score(&self, doc: u64, scorer: &impl Fn(u64, f32) -> f32) -> f32 {
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
        score
    }

    // find the next doc candidate
    fn next(&mut self) -> Option<u64> {
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
                if posting.next(cur_doc + 1).is_some() {
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
                return Some(doc);
            } else {
                // some posting iterators haven't reached this doc id,
                // so pick one of such term(s) and move it to the doc id
                let posting = self.pick_term(doc, self.postings.iter().take(index));
                let mut posting = self
                    .postings
                    .take(&posting)
                    .expect("we just found it in the previous step");
                if posting.next(doc).is_some() {
                    self.postings.insert(posting);
                }
            }
        }
        None
    }

    // find the first term that the sum of upper bound of all preceding terms and itself,
    // are greater than or equal to the threshold
    fn find_pivot_term(&self) -> Option<(usize, PostingIterator<'a>)> {
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
        postings: impl Iterator<Item = &'b PostingIterator<'a>>,
    ) -> PostingIterator<'a>
    where
        'a: 'b,
    {
        postings
            .filter(|posting| posting.doc().unwrap().0 < doc)
            .max_by_key(|posting| OrderedFloat(posting.approximate_upper_bound()))
            .expect("at least one posting list")
            .clone()
    }
}
