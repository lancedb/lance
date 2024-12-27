// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use arrow::datatypes::Int32Type;
use arrow_array::PrimitiveArray;
use itertools::Itertools;
use lance_core::utils::mask::RowIdMask;
use lance_core::Result;
use tracing::instrument;

use super::builder::OrderedDoc;
use super::index::{idf, K1};
use super::{DocInfo, PostingList};

#[derive(Clone)]
pub struct PostingIterator {
    token_id: u32,
    position: i32,
    list: PostingList,
    index: usize,
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
            (Some(doc1), Some(doc2)) => doc1.cmp(&doc2),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        }
    }
}

impl PostingIterator {
    pub(crate) fn new(
        token_id: u32,
        position: i32,
        list: PostingList,
        num_doc: usize,
        mask: Arc<RowIdMask>,
    ) -> Self {
        let approximate_upper_bound = match list.max_score() {
            Some(max_score) => max_score,
            None => idf(list.len(), num_doc) * (K1 + 1.0),
        };

        // move the iterator to the first selected document. This is important
        // because caller might directly call `doc()` without calling `next()`.
        let mut index = 0;
        while index < list.len() && !mask.selected(list.row_id(index)) {
            index += 1;
        }

        Self {
            token_id,
            position,
            list,
            index,
            mask,
            approximate_upper_bound,
        }
    }

    #[inline]
    fn approximate_upper_bound(&self) -> f32 {
        self.approximate_upper_bound
    }

    fn doc(&self) -> Option<DocInfo> {
        if self.index < self.list.len() {
            Some(self.list.doc(self.index))
        } else {
            None
        }
    }

    fn positions(&self, row_id: u64) -> Option<PrimitiveArray<Int32Type>> {
        self.list.positions(row_id)
    }

    // move to the next row id that is greater than or equal to least_id
    #[instrument(level = "debug", name = "posting_iter_next", skip(self))]
    fn next(&mut self, least_id: u64) -> Option<(u64, usize)> {
        self.index += self.list.row_ids[self.index..].partition_point(|&id| id < least_id);
        while self.index < self.list.len() {
            let row_id = self.list.row_id(self.index);
            if self.mask.selected(self.list.row_id(self.index)) {
                return Some((row_id, self.index));
            }
            self.index += 1;
        }
        None
    }
}

pub struct Wand {
    threshold: f32, // multiple of factor and the minimum score of the top-k documents
    cur_doc: Option<u64>,
    num_docs: usize,
    postings: Vec<PostingIterator>,
    candidates: BinaryHeap<Reverse<OrderedDoc>>,
}

impl Wand {
    pub(crate) fn new(num_docs: usize, postings: impl Iterator<Item = PostingIterator>) -> Self {
        Self {
            threshold: 0.0,
            cur_doc: None,
            num_docs,
            postings: postings.filter(|posting| posting.doc().is_some()).collect(),
            candidates: BinaryHeap::new(),
        }
    }

    // search the top-k documents that contain the query
    pub(crate) async fn search(
        &mut self,
        is_phrase_query: bool,
        limit: usize,
        factor: f32,
        scorer: impl Fn(u64, f32) -> f32,
    ) -> Result<Vec<(u64, f32)>> {
        if limit == 0 {
            return Ok(vec![]);
        }

        let num_query_tokens = self.postings.len();

        while let Some(doc) = self.next().await? {
            if is_phrase_query {
                // all the tokens should be in the same document cause it's a phrase query
                if self.postings.len() != num_query_tokens {
                    break;
                }
                if let Some(last) = self.postings.last() {
                    if last.doc().unwrap().row_id != doc {
                        continue;
                    }
                }

                if !self.check_positions() {
                    continue;
                }
            }
            let score = self.score(doc, &scorer);
            if self.candidates.len() < limit {
                self.candidates.push(Reverse(OrderedDoc::new(doc, score)));
            } else if score > self.candidates.peek().unwrap().0.score.0 {
                self.candidates.pop();
                self.candidates.push(Reverse(OrderedDoc::new(doc, score)));
                self.threshold = self.candidates.peek().unwrap().0.score.0 * factor;
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
    fn score(&self, doc_id: u64, scorer: &impl Fn(u64, f32) -> f32) -> f32 {
        let mut score = 0.0;
        for posting in &self.postings {
            let cur_doc = posting.doc().unwrap();
            if cur_doc.row_id > doc_id {
                // the posting list is sorted by its current doc id,
                // so we can break early once we find the current doc id is less than the doc id we are looking for
                break;
            }
            debug_assert!(cur_doc.row_id == doc_id);
            let idf = idf(posting.list.len(), self.num_docs);
            score += idf * (K1 + 1.0) * scorer(doc_id, cur_doc.frequency);
        }
        score
    }

    // find the next doc candidate
    #[instrument(level = "debug", name = "wand_next", skip_all)]
    async fn next(&mut self) -> Result<Option<u64>> {
        self.postings.sort_unstable();
        while let Some(pivot_posting) = self.find_pivot_term() {
            let doc = pivot_posting
                .doc()
                .expect("pivot posting should have at least one document");

            let cur_doc = self.cur_doc.unwrap_or(0);
            if self.cur_doc.is_some() && doc.row_id <= cur_doc {
                self.move_term(cur_doc + 1);
            } else if self.postings[0].doc().unwrap().row_id == doc.row_id {
                // all the posting iterators have reached this doc id,
                // so that means the sum of upper bound of all terms is not less than the threshold,
                // this document is a candidate
                self.cur_doc = Some(doc.row_id);
                return Ok(Some(doc.row_id));
            } else {
                // some posting iterators haven't reached this doc id,
                // so move such terms to the doc id
                self.move_term(doc.row_id);
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
            if doc.row_id >= least_id {
                break;
            }
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
                        .positions(posting.doc().unwrap().row_id)
                        .expect("positions must exist"),
                    posting.position,
                )
            })
            .collect::<Vec<_>>();

        loop {
            let mut max_pos = None;
            let mut all_same = true;
            for iter in &position_iters {
                match (iter.position(), max_pos) {
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

struct PositionIterator {
    positions: PrimitiveArray<Int32Type>,
    pub position_in_query: i32,
    index: usize,
}

impl PositionIterator {
    fn new(positions: PrimitiveArray<Int32Type>, position_in_query: i32) -> Self {
        Self {
            positions,
            position_in_query,
            index: 0,
        }
    }

    fn position(&self) -> Option<i32> {
        if self.index < self.positions.len() {
            Some(self.positions.value(self.index) - self.position_in_query)
        } else {
            None
        }
    }

    fn next(&mut self, least_pos: i32) -> Option<i32> {
        let least_pos = least_pos + self.position_in_query;
        self.index = self
            .positions
            .values()
            .partition_point(|&pos| pos < least_pos);
        self.position()
    }
}
