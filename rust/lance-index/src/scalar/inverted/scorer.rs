// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

// the Scorer trait is used to calculate the score of a token in a document
// in general, the score is calculated as:
// sum over all query_weight(query_token) * doc_weight(freq, doc_tokens)
pub trait Scorer: Send + Sync {
    fn query_weight(&self, token: &str) -> f32;
    fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32;
}

impl<T: Scorer + ?Sized> Scorer for Arc<T> {
    fn query_weight(&self, token: &str) -> f32 {
        self.as_ref().query_weight(token)
    }

    fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32 {
        self.as_ref().doc_weight(freq, doc_tokens)
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
    /// * `tokens` - The tokens of the new document that are also in the query
    /// * `num_tokens` - The total number of tokens in the document
    pub fn update(&mut self, doc_token_count: &HashMap<String, usize>, num_tokens: u64) {
        self.total_tokens += num_tokens;
        self.num_docs += 1;
        for (token, count) in doc_token_count {
            if let Some(old_count) = self.token_docs.get_mut(token) {
                *old_count += *count;
            } else {
                // This shouldn't happen because `tokens` should only contain tokens that are in the query
                // and we should have already initialized this with query tokens.  Still, log a warning just in case.
                log::warn!("Token {} not found in token_docs", token);
            }
        }
    }

    pub fn num_docs(&self) -> usize {
        self.num_docs
    }

    pub fn avg_doc_length(&self) -> f32 {
        self.total_tokens as f32 / self.num_docs as f32
    }

    pub fn num_docs_containing_token(&self, token: &str) -> usize {
        match self.token_docs.get(token) {
            Some(nq) => *nq,
            None => 0,
        }
    }
}

impl Scorer for MemBM25Scorer {
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
        let doc_norm = K1 * (1.0 - B + B * doc_tokens / self.avg_doc_length());
        (K1 + 1.0) * freq / (freq + doc_norm)
    }
}

#[inline]
pub fn idf(token_docs: usize, num_docs: usize) -> f32 {
    let num_docs = num_docs as f32;
    ((num_docs - token_docs as f32 + 0.5) / (token_docs as f32 + 0.5) + 1.0).ln()
}
