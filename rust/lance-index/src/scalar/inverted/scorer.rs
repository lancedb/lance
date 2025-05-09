// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::InvertedIndex;

pub trait Scorer {
    // calculate the contribution of the token in the document
    // token: the token to score
    // freq: the frequency of the token in the document
    // doc_tokens: the number of tokens in the document
    #[allow(dead_code)]
    fn score(&self, token: &str, freq: u32, doc_tokens: u32) -> f32;
}

// BM25 parameters
pub const K1: f32 = 1.2;
pub const B: f32 = 0.75;

pub struct BM25Scorer<'a> {
    index: &'a InvertedIndex,
    num_docs: usize,
    total_tokens: u64,
}

impl<'a> BM25Scorer<'a> {
    pub fn new(index: &'a InvertedIndex) -> Self {
        let num_docs = index.partitions.iter().map(|p| p.docs.len()).sum();
        let total_tokens = index
            .partitions
            .iter()
            .map(|part| part.docs.total_tokens_num())
            .sum();
        Self {
            index,
            num_docs,
            total_tokens,
        }
    }

    pub fn num_docs(&self) -> usize {
        self.num_docs
    }

    // the number of documents that contain the token
    pub fn nq(&self, token: &str) -> usize {
        self.index
            .partitions
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

    pub fn avgdl(&self) -> f32 {
        self.total_tokens as f32 / self.num_docs as f32
    }
}

impl<'a> Scorer for BM25Scorer<'a> {
    fn score(&self, token: &str, freq: u32, doc_tokens: u32) -> f32 {
        let freq = freq as f32;
        let doc_tokens = doc_tokens as f32;
        let doc_norm = K1 * (1.0 - B + B * doc_tokens / self.avgdl());
        return idf(self.nq(token), self.num_docs) * (K1 + 1.0) * freq / (freq + doc_norm);
    }
}

#[inline]
pub fn idf(nq: usize, num_docs: usize) -> f32 {
    let num_docs = num_docs as f32;
    ((num_docs - nq as f32 + 0.5) / (nq as f32 + 0.5) + 1.0).ln()
}
