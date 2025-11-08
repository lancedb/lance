// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::InvertedPartition;
use crate::scalar::inverted::query::Tokens;
use std::collections::HashMap;

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

#[derive(Debug, Clone)]
pub struct MemBM25Scorer {
    pub total_tokens: u64,
    pub num_docs: usize,
    pub token_docs: HashMap<String, usize>,
}

impl MemBM25Scorer {
    pub fn new(total_tokens: u64, num_docs: usize, token_docs: HashMap<String, usize>) -> Self {
        Self {
            total_tokens,
            num_docs,
            token_docs,
        }
    }

    /// Incremental update bm25 scorer with one new document.
    ///
    /// # Arguments
    /// * `tokens` - The tokens of the new document.
    pub fn update(&mut self, tokens: &Tokens) {
        self.total_tokens += tokens.len() as u64;
        self.num_docs += 1;
        for token in tokens {
            *self.token_docs.entry(token.clone()).or_insert(0) += 1;
        }
    }

    pub fn num_docs(&self) -> usize {
        self.num_docs
    }

    pub fn avg_doc_length(&self) -> f32 {
        (self.total_tokens / self.num_docs as u64) as f32
    }

    pub fn num_docs_containing_token(&self, token: &str) -> usize {
        match self.token_docs.get(token) {
            Some(nq) => *nq,
            None => 0,
        }
    }
}

pub struct IndexBM25Scorer<'a> {
    partitions: Vec<&'a InvertedPartition>,
    num_docs: usize,
    avg_doc_length: f32,
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
            avg_doc_length: avgdl,
        }
    }

    pub fn num_docs(&self) -> usize {
        self.num_docs
    }

    pub fn avg_doc_length(&self) -> f32 {
        self.avg_doc_length
    }

    pub fn num_docs_containing_token(&self, token: &str) -> usize {
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
        let token_docs = self.num_docs_containing_token(token);
        if token_docs == 0 {
            return 0.0;
        }
        idf(token_docs, self.num_docs)
    }

    fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32 {
        let freq = freq as f32;
        let doc_tokens = doc_tokens as f32;
        let doc_norm = K1 * (1.0 - B + B * doc_tokens / self.avg_doc_length);
        (K1 + 1.0) * freq / (freq + doc_norm)
    }
}

#[inline]
pub fn idf(token_docs: usize, num_docs: usize) -> f32 {
    let num_docs = num_docs as f32;
    ((num_docs - token_docs as f32 + 0.5) / (token_docs as f32 + 0.5) + 1.0).ln()
}
