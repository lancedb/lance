// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{DocSet, InvertedPartition, PlainPostingList, TokenSet};

// the Scorer trait is used to calculate the score of a token in a document
// in general, the score is calculated as:
// sum over all query_weight(query_token) * doc_weight(freq, doc_tokens)
pub trait Scorer: Send + Sync {
    fn query_weight(&self, token: &str) -> f32;
    fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32;
    // calculate the contribution of the token in the document
    // token: the token to score
    // freq: the frequency of the token in the document
    // doc_tokens: the number of tokens in the document
    fn score(&self, token: &str, freq: u32, doc_tokens: u32) -> f32 {
        self.query_weight(token) * self.doc_weight(freq, doc_tokens)
    }
}

// BM25 parameters
pub const K1: f32 = 1.2;
pub const B: f32 = 0.75;

pub enum BM25Scorer<'a> {
    Index(IndexBM25Scorer<'a>),
    Memory(MemoryBM25Scorer),
    Union(Box<BM25Scorer<'a>>, Box<BM25Scorer<'a>>),
}

impl<'a> BM25Scorer<'a> {
    pub fn num_docs(&self) -> usize {
        match self {
            BM25Scorer::Index(scorer) => scorer.num_docs(),
            BM25Scorer::Memory(scorer) => scorer.num_docs(),
            BM25Scorer::Union(left, right) => left.num_docs() + right.num_docs(),
        }
    }

    pub fn avgdl(&self) -> f32 {
        match self {
            BM25Scorer::Index(scorer) => scorer.avgdl(),
            BM25Scorer::Memory(scorer) => scorer.avgdl(),
            BM25Scorer::Union(left, right) => {
                (left.avgdl() * left.num_docs() as f32 + right.avgdl() * right.num_docs() as f32)
                    / (left.num_docs() + right.num_docs()) as f32
            }
        }
    }

    // the number of documents that contain the token
    pub fn nq(&self, token: &str) -> usize {
        match self {
            BM25Scorer::Index(scorer) => scorer.nq(token),
            BM25Scorer::Memory(scorer) => scorer.nq(token),
            BM25Scorer::Union(left, right) => left.nq(token) + right.nq(token),
        }
    }
}

impl Scorer for BM25Scorer<'_> {
    fn query_weight(&self, token: &str) -> f32 {
        let nq = self.nq(token);
        if nq == 0 {
            return 0.0;
        }
        idf(nq, self.num_docs())
    }

    fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32 {
        let freq = freq as f32;
        let doc_tokens = doc_tokens as f32;
        let doc_norm = K1 * (1.0 - B + B * doc_tokens / self.avgdl());
        (K1 + 1.0) * freq / (freq + doc_norm)
    }
}

pub struct MemoryBM25Scorer {
    pub posting_lists: Vec<PlainPostingList>,
    pub tokens_set: TokenSet,
    pub doc_set: DocSet,
}

impl MemoryBM25Scorer {
    pub fn new(
        posting_lists: Vec<PlainPostingList>,
        tokens_set: TokenSet,
        doc_set: DocSet,
    ) -> Self {
        Self {
            posting_lists,
            tokens_set,
            doc_set,
        }
    }

    pub fn num_docs(&self) -> usize {
        self.doc_set.len()
    }

    pub fn avgdl(&self) -> f32 {
        self.doc_set.average_length()
    }

    pub fn nq(&self, token: &str) -> usize {
        if let Some(token_id) = self.tokens_set.get(token) {
            self.posting_lists
                .get(token_id as usize)
                .unwrap()
                .len()
                .max(1)
        } else {
            0
        }
    }
}

pub struct IndexBM25Scorer<'a> {
    partitions: Vec<&'a InvertedPartition>,
    num_docs: usize,
    avgdl: f32,
}

impl<'a> IndexBM25Scorer<'a> {
    pub fn new(partitions: impl Iterator<Item = &'a InvertedPartition>) -> Self {
        let partitions = partitions.collect::<Vec<_>>();
        let num_docs = partitions.iter().map(|p| p.docs.len()).sum();
        let total_tokens = partitions
            .iter()
            .map(|part| part.docs.total_tokens_num())
            .sum::<u64>();
        let avgdl = total_tokens as f32 / num_docs as f32;
        Self {
            partitions,
            num_docs,
            avgdl,
        }
    }

    pub fn num_docs(&self) -> usize {
        self.num_docs
    }

    pub fn avgdl(&self) -> f32 {
        self.avgdl
    }

    pub fn nq(&self, token: &str) -> usize {
        self.partitions
            .iter()
            .map(|part| {
                if let Some(token_id) = part.tokens.get(token) {
                    part.inverted_list.posting_len(token_id)
                } else {
                    0
                }
            })
            .sum()
    }
}

impl Scorer for IndexBM25Scorer<'_> {
    fn query_weight(&self, token: &str) -> f32 {
        let nq = self.nq(token);
        if nq == 0 {
            return 0.0;
        }
        idf(nq, self.num_docs)
    }

    fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32 {
        let freq = freq as f32;
        let doc_tokens = doc_tokens as f32;
        let doc_norm = K1 * (1.0 - B + B * doc_tokens / self.avgdl);
        (K1 + 1.0) * freq / (freq + doc_norm)
    }
}

#[inline]
pub fn idf(nq: usize, num_docs: usize) -> f32 {
    let num_docs = num_docs as f32;
    ((num_docs - nq as f32 + 0.5) / (nq as f32 + 0.5) + 1.0).ln()
}
