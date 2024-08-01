// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use itertools::Itertools;
use lance_core::utils::mask::RowIdMask;
use lance_core::Result;
use tracing::instrument;

use super::builder::OrderedDoc;
use super::index::{idf, K1};
use super::PostingList;

#[derive(Clone)]
pub struct PostingIterator {
    token_id: u32,
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
        list: PostingList,
        num_doc: usize,
        mask: Arc<RowIdMask>,
    ) -> Self {
        let approximate_upper_bound = idf(list.len(), num_doc) * (K1 + 1.0);
        Self {
            token_id,
            list,
            index: 0,
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
            Some(self.list.doc(self.index))
        } else {
            None
        }
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
    postings: Vec<PostingIterator>,
    candidates: BinaryHeap<Reverse<OrderedDoc>>,
}

impl Wand {
    pub(crate) fn new(postings: impl Iterator<Item = PostingIterator>) -> Self {
        Self {
            threshold: 0.0,
            cur_doc: None,
            postings: postings.collect(),
            candidates: BinaryHeap::new(),
        }
    }

    // search the top-k documents that contain the query
    pub(crate) async fn search(
        &mut self,
        limit: usize,
        factor: f32,
        scorer: impl Fn(u64, f32) -> f32,
    ) -> Result<Vec<(u64, f32)>> {
        if limit == 0 {
            return Ok(vec![]);
        }

        while let Some(doc) = self.next().await? {
            let score = self.score(doc, &scorer);
            if self.candidates.len() < limit {
                self.candidates.push(Reverse(OrderedDoc::new(doc, score)));
            } else if score > self.threshold {
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
                self.move_terms(cur_doc + 1);
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
                self.move_terms(doc);
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
    fn move_terms<'b>(&mut self, least_id: u64) {
        for posting in self.postings.iter_mut() {
            match posting.doc() {
                Some((d, _)) if d < least_id => {
                    posting.next(least_id);
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
    }
}
