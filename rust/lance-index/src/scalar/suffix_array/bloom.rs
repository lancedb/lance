// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Simple bloom filter for n-gram existence checks.
//!
//! Used to quickly determine if a segment *definitely does not* contain
//! a query substring, avoiding expensive full-segment loads from S3.
//!
//! Each segment stores a bloom filter of all unique 4-grams (or shorter
//! for small texts). The filter is ~128 KB per segment — about 1000×
//! smaller than the full segment data.

use std::hash::{BuildHasher, Hasher};

/// Number of bytes per n-gram inserted into the bloom filter.
/// 4-grams provide a good balance: specific enough to filter out
/// non-matching segments, short enough to cover most queries.
pub const NGRAM_SIZE: usize = 4;

/// A simple bloom filter using double hashing.
///
/// Uses two independent hash functions (SipHash with different seeds)
/// to generate `num_hashes` bit positions per item. This avoids needing
/// multiple independent hash functions while maintaining good distribution.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    /// The bit array, packed into u64 words.
    bits: Vec<u64>,
    /// Total number of bits in the filter.
    num_bits: u64,
    /// Number of hash functions (bit positions per item).
    num_hashes: u32,
}

impl BloomFilter {
    /// Create a new bloom filter sized for the expected number of items.
    ///
    /// Uses ~10 bits per item with 7 hash functions for ~0.8% false positive rate.
    /// The filter size is clamped to [1 KB, 256 KB].
    pub fn with_capacity(expected_items: usize) -> Self {
        // ~10 bits per item, 7 hash functions → ~0.8% FPR
        let bits_needed = (expected_items as u64 * 10).max(8192); // min 1 KB
        let num_bits = bits_needed.min(2_097_152); // max 256 KB
        let num_words = ((num_bits + 63) / 64) as usize;
        Self {
            bits: vec![0u64; num_words],
            num_bits: num_words as u64 * 64,
            num_hashes: 7,
        }
    }

    /// Number of bits in the filter (for diagnostics).
    pub fn num_bits(&self) -> u64 {
        self.num_bits
    }

    /// Insert an item (byte slice) into the bloom filter.
    pub fn insert(&mut self, item: &[u8]) {
        let (h1, h2) = self.double_hash(item);
        for i in 0..self.num_hashes {
            let pos = self.bit_position(h1, h2, i);
            let word = (pos / 64) as usize;
            let bit = pos % 64;
            self.bits[word] |= 1u64 << bit;
        }
    }

    /// Check if an item might be in the filter.
    ///
    /// Returns `false` if the item is *definitely not* present.
    /// Returns `true` if the item *might* be present (possible false positive).
    pub fn might_contain(&self, item: &[u8]) -> bool {
        if self.bits.is_empty() {
            return false;
        }
        let (h1, h2) = self.double_hash(item);
        for i in 0..self.num_hashes {
            let pos = self.bit_position(h1, h2, i);
            let word = (pos / 64) as usize;
            let bit = pos % 64;
            if self.bits[word] & (1u64 << bit) == 0 {
                return false;
            }
        }
        true
    }

    /// Check if a query string might exist in the segment's text.
    ///
    /// Extracts all NGRAM_SIZE-grams from the query and checks each against
    /// the bloom filter. If any n-gram is definitely absent, the full query
    /// cannot appear in the segment.
    ///
    /// For queries shorter than NGRAM_SIZE, checks the query as a single item.
    pub fn might_contain_substring(&self, query: &[u8]) -> bool {
        self.might_contain_substring_aligned(query, 1)
    }

    /// Check if a query might exist, respecting token alignment.
    ///
    /// `token_width` is the number of bytes per token (1 for text, 2/4 for token-level).
    /// N-grams are checked at token-aligned boundaries.
    pub fn might_contain_substring_aligned(&self, query: &[u8], token_width: usize) -> bool {
        if query.is_empty() || self.bits.is_empty() {
            return true; // empty query matches everything; empty filter = no data
        }

        let tw = token_width.max(1);
        let ngram_bytes = NGRAM_SIZE * tw;

        if query.len() < ngram_bytes {
            // For short queries, check the exact bytes.
            // We also inserted all n-grams shorter than NGRAM_SIZE during build.
            return self.might_contain(query);
        }

        // Check all token-aligned n-grams of the query
        let mut pos = 0;
        while pos + ngram_bytes <= query.len() {
            if !self.might_contain(&query[pos..pos + ngram_bytes]) {
                return false;
            }
            pos += tw;
        }
        true
    }

    /// Serialize the bloom filter to bytes.
    ///
    /// Format: [num_hashes: u32 LE] [num_bits: u64 LE] [bits: packed u64 LE...]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + 8 + self.bits.len() * 8);
        out.extend_from_slice(&self.num_hashes.to_le_bytes());
        out.extend_from_slice(&self.num_bits.to_le_bytes());
        for &word in &self.bits {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out
    }

    /// Deserialize a bloom filter from bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 12 {
            return None;
        }
        let num_hashes = u32::from_le_bytes(data[0..4].try_into().ok()?);
        let num_bits = u64::from_le_bytes(data[4..12].try_into().ok()?);
        let num_words = ((num_bits + 63) / 64) as usize;
        if data.len() < 12 + num_words * 8 {
            return None;
        }
        let bits: Vec<u64> = data[12..]
            .chunks_exact(8)
            .take(num_words)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        Some(Self {
            bits,
            num_bits,
            num_hashes,
        })
    }

    /// Build a bloom filter from a text buffer.
    ///
    /// Inserts all unique NGRAM_SIZE-grams from the text.
    /// Also inserts shorter n-grams (1, 2, 3 bytes) so that short
    /// queries can be checked directly.
    pub fn from_text(text: &[u8]) -> Self {
        if text.is_empty() {
            return Self {
                bits: Vec::new(),
                num_bits: 0,
                num_hashes: 7,
            };
        }

        // Estimate unique n-gram count (rough upper bound)
        let estimated_ngrams = text.len().min(1_000_000); // cap estimate
        let mut filter = Self::with_capacity(estimated_ngrams);

        // Insert all NGRAM_SIZE-grams
        if text.len() >= NGRAM_SIZE {
            for window in text.windows(NGRAM_SIZE) {
                filter.insert(window);
            }
        }

        // Also insert shorter n-grams so short queries work
        for n in 1..NGRAM_SIZE {
            if text.len() >= n {
                for window in text.windows(n) {
                    filter.insert(window);
                }
            }
        }

        filter
    }

    /// Build a bloom filter from a text buffer, for token-level indices.
    ///
    /// Like `from_text`, but n-grams are aligned to `token_width` boundaries.
    pub fn from_text_token_aligned(text: &[u8], token_width: usize) -> Self {
        if text.is_empty() || token_width == 0 {
            return Self {
                bits: Vec::new(),
                num_bits: 0,
                num_hashes: 7,
            };
        }

        let estimated_ngrams = (text.len() / token_width).min(1_000_000);
        let mut filter = Self::with_capacity(estimated_ngrams);

        // Insert token-aligned n-grams
        for n_tokens in 1..=NGRAM_SIZE {
            let window_bytes = n_tokens * token_width;
            if text.len() >= window_bytes {
                let mut pos = 0;
                while pos + window_bytes <= text.len() {
                    filter.insert(&text[pos..pos + window_bytes]);
                    pos += token_width;
                }
            }
        }

        filter
    }

    // ── Internal helpers ──

    fn double_hash(&self, item: &[u8]) -> (u64, u64) {
        // Hash 1: default hasher with seed 0
        let h1 = {
            let bh = std::hash::BuildHasherDefault::<std::collections::hash_map::DefaultHasher>::default();
            let mut h = bh.build_hasher();
            h.write(item);
            h.finish()
        };
        // Hash 2: mix in a constant to get a different hash
        let h2 = {
            let bh = std::hash::BuildHasherDefault::<std::collections::hash_map::DefaultHasher>::default();
            let mut h = bh.build_hasher();
            h.write(item);
            h.write_u64(0x517cc1b727220a95); // golden ratio constant
            h.finish()
        };
        (h1, h2)
    }

    fn bit_position(&self, h1: u64, h2: u64, i: u32) -> u64 {
        h1.wrapping_add(h2.wrapping_mul(i as u64)) % self.num_bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_basic() {
        let mut bf = BloomFilter::with_capacity(100);
        bf.insert(b"hello");
        bf.insert(b"world");

        assert!(bf.might_contain(b"hello"));
        assert!(bf.might_contain(b"world"));
        // Very unlikely to be a false positive for a simple test
        assert!(!bf.might_contain(b"xyzzy"));
    }

    #[test]
    fn test_bloom_from_text() {
        let text = b"the quick brown fox jumps over the lazy dog";
        let bf = BloomFilter::from_text(text);

        // All 4-grams from the text should be present
        assert!(bf.might_contain_substring(b"quic"));
        assert!(bf.might_contain_substring(b"quick"));
        assert!(bf.might_contain_substring(b"the q"));
        assert!(bf.might_contain_substring(b"the"));

        // Substrings not in the text
        assert!(!bf.might_contain_substring(b"xyzzyplugh"));
        assert!(!bf.might_contain_substring(b"zzzz"));
    }

    #[test]
    fn test_bloom_short_query() {
        let text = b"abcdef";
        let bf = BloomFilter::from_text(text);

        // Short queries should work
        assert!(bf.might_contain_substring(b"a"));
        assert!(bf.might_contain_substring(b"ab"));
        assert!(bf.might_contain_substring(b"abc"));
        assert!(!bf.might_contain_substring(b"x"));
        assert!(!bf.might_contain_substring(b"xy"));
    }

    #[test]
    fn test_bloom_serialization() {
        let text = b"hello world test data";
        let bf = BloomFilter::from_text(text);

        let bytes = bf.to_bytes();
        let bf2 = BloomFilter::from_bytes(&bytes).unwrap();

        assert!(bf2.might_contain_substring(b"hello"));
        assert!(bf2.might_contain_substring(b"world"));
        assert!(!bf2.might_contain_substring(b"xyzzy"));
    }

    #[test]
    fn test_bloom_empty_text() {
        let bf = BloomFilter::from_text(b"");
        // Empty filter should say "might contain" for empty query
        assert!(bf.might_contain_substring(b""));
        // But should say "might contain" is true because filter is empty
        // (we return true for empty filter since we can't prove absence)
        assert!(bf.might_contain_substring(b"anything"));
    }
}
