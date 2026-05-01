// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Query operations on the suffix array.
//!
//! Provides binary search-based lookup and counting of n-gram patterns,
//! as well as language-modeling queries (conditional probability,
//! next-byte distribution, and infinity-gram with backoff).

use std::any::Any;
use std::cmp::Ordering;
use std::collections::HashMap;

use datafusion_common::Column;
use datafusion_expr::Expr;

use crate::scalar::AnyQuery;

/// Query types supported by the suffix array index.
#[derive(Debug, Clone, PartialEq)]
pub enum SuffixArrayQuery {
    /// Count the number of occurrences of a byte pattern in the corpus.
    Count { query_bytes: Vec<u8> },
    /// Search for documents containing a byte pattern, returning up to max_results.
    Search {
        query_bytes: Vec<u8>,
        max_results: usize,
    },
    /// Conditional probability: P(continuation | prompt) = count(prompt+cont) / count(prompt).
    Prob {
        prompt_bytes: Vec<u8>,
        continuation_bytes: Vec<u8>,
    },
    /// Next-byte distribution: for each possible next byte after `prompt`,
    /// returns the count and probability. Uses divide-and-conquer over the SA range.
    NextByteDistribution {
        prompt_bytes: Vec<u8>,
        /// Maximum number of entries to scan before switching to approximate mode.
        max_support: Option<u64>,
    },
    /// Infinity-gram probability with backoff.
    /// Finds the longest suffix of `prompt` with nonzero count, then computes
    /// P(continuation | longest_suffix).
    InfgramProb {
        prompt_bytes: Vec<u8>,
        continuation_bytes: Vec<u8>,
    },
}

impl AnyQuery for SuffixArrayQuery {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn format(&self, col: &str) -> String {
        match self {
            Self::Count { query_bytes } => {
                format!("sa_count({col}, [{} bytes])", query_bytes.len())
            }
            Self::Search {
                query_bytes,
                max_results,
            } => {
                format!(
                    "sa_search({col}, [{} bytes], max={})",
                    query_bytes.len(),
                    max_results
                )
            }
            Self::Prob {
                prompt_bytes,
                continuation_bytes,
            } => {
                format!(
                    "sa_prob({col}, [{} bytes], [{} bytes])",
                    prompt_bytes.len(),
                    continuation_bytes.len()
                )
            }
            Self::NextByteDistribution {
                prompt_bytes,
                max_support,
            } => {
                format!(
                    "sa_ntd({col}, [{} bytes], max_support={:?})",
                    prompt_bytes.len(),
                    max_support
                )
            }
            Self::InfgramProb {
                prompt_bytes,
                continuation_bytes,
            } => {
                format!(
                    "sa_infgram_prob({col}, [{} bytes], [{} bytes])",
                    prompt_bytes.len(),
                    continuation_bytes.len()
                )
            }
        }
    }

    fn to_expr(&self, col: String) -> Expr {
        // Suffix array queries do not map to standard SQL expressions.
        // Return a placeholder literal.
        Expr::Column(Column::new_unqualified(col))
    }

    fn dyn_eq(&self, other: &dyn AnyQuery) -> bool {
        match other.as_any().downcast_ref::<Self>() {
            Some(o) => self == o,
            None => false,
        }
    }
}

/// Find the suffix array range `[lo, hi)` where all suffixes start with `query`.
///
/// Uses a two-phase binary search: first finds any match, then refines
/// the left and right boundaries.
///
/// Returns `(lo, hi)` such that `hi - lo` equals the count of occurrences.
pub fn sa_find(
    tokenized: &[u8],
    sa: &[u8],
    ptr_width: usize,
    total_entries: u64,
    query: &[u8],
) -> (u64, u64) {
    let n = total_entries;
    if n == 0 || query.is_empty() {
        return (0, n);
    }

    // Phase 1: find any matching position via standard binary search
    let mut lo: u64 = 0;
    let mut hi: u64 = n;
    let mut found = false;
    let mut mid: u64 = 0;

    // Save original bounds for phase 2
    let orig_lo = lo;
    let orig_hi = hi;

    while lo < hi {
        mid = lo + (hi - lo) / 2;
        let ptr = read_pointer(sa, mid, ptr_width);
        let cmp = compare_suffix(tokenized, ptr, query);
        match cmp {
            Ordering::Less => lo = mid + 1,
            Ordering::Greater => hi = mid,
            Ordering::Equal => {
                found = true;
                break;
            }
        }
    }

    if !found {
        return (lo, lo); // Not found, count = 0
    }

    // Phase 2: find exact boundaries using two separate binary searches

    // Left boundary: find the first match in [orig_lo, mid]
    let left = {
        let mut l = orig_lo;
        let mut r = mid;
        while l < r {
            let m = l + (r - l) / 2;
            let ptr = read_pointer(sa, m, ptr_width);
            if compare_suffix(tokenized, ptr, query) == Ordering::Less {
                l = m + 1;
            } else {
                r = m;
            }
        }
        l
    };

    // Right boundary: find the first non-match in [mid+1, orig_hi]
    let right = {
        let mut l = mid + 1;
        let mut r = orig_hi;
        while l < r {
            let m = l + (r - l) / 2;
            let ptr = read_pointer(sa, m, ptr_width);
            if compare_suffix(tokenized, ptr, query) == Ordering::Greater {
                r = m;
            } else {
                l = m + 1;
            }
        }
        l
    };

    (left, right)
}

/// Count occurrences of a byte pattern in the corpus.
pub fn count(
    tokenized: &[u8],
    sa: &[u8],
    ptr_width: usize,
    total_entries: u64,
    query: &[u8],
) -> u64 {
    let (lo, hi) = sa_find(tokenized, sa, ptr_width, total_entries, query);
    hi - lo
}

// ─── Language modeling queries ─────────────────────────────────────────────────

/// Result of a conditional probability query.
#[derive(Debug, Clone)]
pub struct ProbResult {
    /// Number of times the prompt appears in the corpus.
    pub prompt_cnt: u64,
    /// Number of times prompt + continuation appears.
    pub cont_cnt: u64,
    /// Conditional probability: cont_cnt / prompt_cnt (0.0 if prompt_cnt == 0).
    pub prob: f64,
}

/// Result of an infinity-gram probability query.
#[derive(Debug, Clone)]
pub struct InfgramProbResult {
    /// The conditional probability result using the effective suffix.
    pub prob_result: ProbResult,
    /// Length of the effective suffix used (longest suffix of prompt with nonzero count).
    pub effective_suffix_len: usize,
}

/// Entry in a next-byte distribution.
#[derive(Debug, Clone)]
pub struct NtdEntry {
    /// The byte value.
    pub byte_value: u8,
    /// Number of times this byte follows the prompt.
    pub count: u64,
    /// Probability: count / prompt_count.
    pub prob: f64,
}

/// Result of a next-byte distribution query.
#[derive(Debug, Clone)]
pub struct NtdResult {
    /// Number of times the prompt appears.
    pub prompt_cnt: u64,
    /// Distribution over possible next bytes, sorted by count descending.
    pub distribution: Vec<NtdEntry>,
    /// Whether the result is approximate (when exceeding max_support).
    pub approximate: bool,
}

/// Compute conditional probability P(continuation | prompt).
///
/// This is `count(prompt + continuation) / count(prompt)`.
/// If the prompt has zero occurrences, the probability is 0.
pub fn prob(
    tokenized: &[u8],
    sa: &[u8],
    ptr_width: usize,
    total_entries: u64,
    prompt: &[u8],
    continuation: &[u8],
) -> ProbResult {
    let prompt_cnt = count(tokenized, sa, ptr_width, total_entries, prompt);
    if prompt_cnt == 0 {
        return ProbResult {
            prompt_cnt: 0,
            cont_cnt: 0,
            prob: 0.0,
        };
    }

    // Concatenate prompt + continuation
    let mut full_query = Vec::with_capacity(prompt.len() + continuation.len());
    full_query.extend_from_slice(prompt);
    full_query.extend_from_slice(continuation);

    let cont_cnt = count(tokenized, sa, ptr_width, total_entries, &full_query);
    let probability = cont_cnt as f64 / prompt_cnt as f64;

    ProbResult {
        prompt_cnt,
        cont_cnt,
        prob: probability,
    }
}

/// Compute the next-byte distribution after a prompt.
///
/// Uses a divide-and-conquer approach over the suffix array range:
/// if all entries in a range have the same next byte, count them as one group.
/// Otherwise, split the range and recurse.
pub fn next_byte_distribution(
    tokenized: &[u8],
    sa: &[u8],
    ptr_width: usize,
    total_entries: u64,
    prompt: &[u8],
    max_support: Option<u64>,
) -> NtdResult {
    let (lo, hi) = sa_find(tokenized, sa, ptr_width, total_entries, prompt);
    let prompt_cnt = hi - lo;

    if prompt_cnt == 0 {
        return NtdResult {
            prompt_cnt: 0,
            distribution: Vec::new(),
            approximate: false,
        };
    }

    // Check if we should use approximate mode
    let approximate = max_support.is_some_and(|max| prompt_cnt > max);

    // Collect byte -> count using divide-and-conquer
    let mut byte_counts: HashMap<u8, u64> = HashMap::new();
    let prompt_len = prompt.len();

    if approximate {
        // Approximate mode: sample uniformly from the range
        let max = max_support.unwrap();
        let step = prompt_cnt / max;
        for i in 0..max {
            let rank = lo + i * step;
            let pos = read_pointer(sa, rank, ptr_width) as usize;
            let next_pos = pos + prompt_len;
            if next_pos < tokenized.len() {
                let next_byte = tokenized[next_pos];
                *byte_counts.entry(next_byte).or_insert(0) += step;
            }
        }
    } else {
        // Exact mode: divide-and-conquer over the SA range
        ntd_divide_and_conquer(
            tokenized,
            sa,
            ptr_width,
            prompt_len,
            lo,
            hi,
            &mut byte_counts,
        );
    }

    // Build sorted distribution
    let mut distribution: Vec<NtdEntry> = byte_counts
        .into_iter()
        .map(|(byte_value, cnt)| NtdEntry {
            byte_value,
            count: cnt,
            prob: cnt as f64 / prompt_cnt as f64,
        })
        .collect();
    distribution.sort_by(|a, b| b.count.cmp(&a.count));

    NtdResult {
        prompt_cnt,
        distribution,
        approximate,
    }
}

/// Divide-and-conquer helper for next-byte distribution.
///
/// If the first and last entries in `[lo, hi)` have the same next byte,
/// the entire range has that byte. Otherwise, split at the midpoint and recurse.
fn ntd_divide_and_conquer(
    tokenized: &[u8],
    sa: &[u8],
    ptr_width: usize,
    prompt_len: usize,
    lo: u64,
    hi: u64,
    byte_counts: &mut HashMap<u8, u64>,
) {
    if lo >= hi {
        return;
    }

    let first_byte = get_next_byte(tokenized, sa, ptr_width, prompt_len, lo);
    if hi - lo == 1 {
        if let Some(b) = first_byte {
            *byte_counts.entry(b).or_insert(0) += 1;
        }
        return;
    }

    let last_byte = get_next_byte(tokenized, sa, ptr_width, prompt_len, hi - 1);

    if first_byte == last_byte {
        // All entries in this range have the same next byte
        if let Some(b) = first_byte {
            *byte_counts.entry(b).or_insert(0) += hi - lo;
        }
    } else {
        // Split and recurse
        let mid = lo + (hi - lo) / 2;
        ntd_divide_and_conquer(tokenized, sa, ptr_width, prompt_len, lo, mid, byte_counts);
        ntd_divide_and_conquer(tokenized, sa, ptr_width, prompt_len, mid, hi, byte_counts);
    }
}

/// Get the byte immediately following the prompt at a given SA rank.
/// Returns None if the suffix is too short (prompt is at the end of the corpus).
#[inline]
fn get_next_byte(
    tokenized: &[u8],
    sa: &[u8],
    ptr_width: usize,
    prompt_len: usize,
    rank: u64,
) -> Option<u8> {
    let pos = read_pointer(sa, rank, ptr_width) as usize;
    let next_pos = pos + prompt_len;
    if next_pos < tokenized.len() {
        Some(tokenized[next_pos])
    } else {
        None
    }
}

/// Infinity-gram probability with backoff.
///
/// Finds the longest suffix of `prompt` that has a non-zero count in the corpus,
/// then computes `prob(longest_suffix, continuation)`.
///
/// Uses binary lifting (doubling) for efficiency:
/// 1. Try suffix lengths 1, 2, 4, 8, 16, ... until count drops to 0
/// 2. Binary search between the last non-zero and first zero length
/// 3. Compute prob() using the longest suffix with non-zero count
pub fn infgram_prob(
    tokenized: &[u8],
    sa: &[u8],
    ptr_width: usize,
    total_entries: u64,
    prompt: &[u8],
    continuation: &[u8],
) -> InfgramProbResult {
    let prompt_len = prompt.len();

    if prompt_len == 0 {
        // No prompt, just count the continuation
        let result = prob(tokenized, sa, ptr_width, total_entries, &[], continuation);
        return InfgramProbResult {
            prob_result: result,
            effective_suffix_len: 0,
        };
    }

    // Phase 1: Binary lifting — find where count drops to 0
    let mut good_len = 0usize; // last length with nonzero count
    let mut bad_len = prompt_len + 1; // first length known to have zero count
    let mut power = 1usize;

    while power <= prompt_len {
        let suffix_start = prompt_len.saturating_sub(power);
        let suffix = &prompt[suffix_start..];
        let cnt = count(tokenized, sa, ptr_width, total_entries, suffix);
        if cnt > 0 {
            good_len = power;
            power *= 2;
        } else {
            bad_len = power;
            break;
        }
    }

    // If the entire prompt has nonzero count, use it
    if good_len == prompt_len || power > prompt_len {
        if good_len == 0 {
            // Even single-byte suffix has zero count
            return InfgramProbResult {
                prob_result: ProbResult {
                    prompt_cnt: 0,
                    cont_cnt: 0,
                    prob: 0.0,
                },
                effective_suffix_len: 0,
            };
        }
        // Check if the full prompt has nonzero count
        let full_cnt = count(tokenized, sa, ptr_width, total_entries, prompt);
        if full_cnt > 0 {
            good_len = prompt_len;
        }
    }

    // Phase 2: Binary search between good_len and bad_len
    while good_len + 1 < bad_len && bad_len <= prompt_len {
        let mid = good_len + (bad_len - good_len) / 2;
        let suffix_start = prompt_len - mid;
        let suffix = &prompt[suffix_start..];
        let cnt = count(tokenized, sa, ptr_width, total_entries, suffix);
        if cnt > 0 {
            good_len = mid;
        } else {
            bad_len = mid;
        }
    }

    // Phase 3: Compute prob using the effective suffix
    if good_len == 0 {
        return InfgramProbResult {
            prob_result: ProbResult {
                prompt_cnt: 0,
                cont_cnt: 0,
                prob: 0.0,
            },
            effective_suffix_len: 0,
        };
    }

    let suffix_start = prompt_len - good_len;
    let effective_suffix = &prompt[suffix_start..];
    let result = prob(
        tokenized,
        sa,
        ptr_width,
        total_entries,
        effective_suffix,
        continuation,
    );

    InfgramProbResult {
        prob_result: result,
        effective_suffix_len: good_len,
    }
}

/// Read a compact pointer from the suffix array at the given rank.
///
/// The pointer is stored as `ptr_width` bytes in little-endian format.
#[inline]
pub fn read_pointer(sa: &[u8], rank: u64, ptr_width: usize) -> u64 {
    let offset = (rank as usize) * ptr_width;
    let mut buf = [0u8; 8];
    buf[..ptr_width].copy_from_slice(&sa[offset..offset + ptr_width]);
    u64::from_le_bytes(buf)
}

/// Compare the suffix at `pos` in the tokenized data against the query.
///
/// Only compares up to `query.len()` bytes of the suffix (prefix comparison).
#[inline]
fn compare_suffix(tokenized: &[u8], pos: u64, query: &[u8]) -> Ordering {
    let pos = pos as usize;
    let available = tokenized.len().saturating_sub(pos);
    let cmp_len = available.min(query.len());
    let suffix_prefix = &tokenized[pos..pos + cmp_len];
    let cmp = suffix_prefix.cmp(&query[..cmp_len]);
    if cmp != Ordering::Equal {
        return cmp;
    }
    // If the suffix is shorter than the query, it is "less" because
    // it cannot fully match the query prefix.
    if available < query.len() {
        Ordering::Less
    } else {
        Ordering::Equal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::suffix_array::builder::{build_suffix_array, compact_suffix_array};

    fn build_test_index(data: &[u8]) -> (Vec<u8>, u64) {
        let sa = build_suffix_array(data);
        let n = sa.len() as u64;
        let compact = compact_suffix_array(&sa, 4).unwrap();
        (compact, n)
    }

    #[test]
    fn test_count_banana() {
        let data = b"banana";
        let (sa, n) = build_test_index(data);

        assert_eq!(count(data, &sa, 4, n, b"a"), 3);
        assert_eq!(count(data, &sa, 4, n, b"an"), 2);
        assert_eq!(count(data, &sa, 4, n, b"ana"), 2);
        assert_eq!(count(data, &sa, 4, n, b"ban"), 1);
        assert_eq!(count(data, &sa, 4, n, b"banana"), 1);
        assert_eq!(count(data, &sa, 4, n, b"xyz"), 0);
        assert_eq!(count(data, &sa, 4, n, b"bananaa"), 0);
        assert_eq!(count(data, &sa, 4, n, b"na"), 2);
        assert_eq!(count(data, &sa, 4, n, b"n"), 2);
    }

    #[test]
    fn test_count_empty_query() {
        let data = b"banana";
        let (sa, n) = build_test_index(data);
        // Empty query matches everything
        assert_eq!(count(data, &sa, 4, n, b""), n);
    }

    #[test]
    fn test_count_empty_data() {
        let data = b"";
        let (sa, n) = build_test_index(data);
        assert_eq!(count(data, &sa, 4, n, b"a"), 0);
    }

    #[test]
    fn test_count_single_char() {
        let data = b"a";
        let (sa, n) = build_test_index(data);
        assert_eq!(count(data, &sa, 4, n, b"a"), 1);
        assert_eq!(count(data, &sa, 4, n, b"b"), 0);
    }

    #[test]
    fn test_count_repeated() {
        let data = b"aaaa";
        let (sa, n) = build_test_index(data);
        assert_eq!(count(data, &sa, 4, n, b"a"), 4);
        assert_eq!(count(data, &sa, 4, n, b"aa"), 3);
        assert_eq!(count(data, &sa, 4, n, b"aaa"), 2);
        assert_eq!(count(data, &sa, 4, n, b"aaaa"), 1);
        assert_eq!(count(data, &sa, 4, n, b"aaaaa"), 0);
    }

    #[test]
    fn test_sa_find_boundaries() {
        let data = b"abracadabra";
        let (sa, n) = build_test_index(data);
        let (lo, hi) = sa_find(data, &sa, 4, n, b"abra");
        assert_eq!(hi - lo, 2); // "abra" appears twice
    }

    #[test]
    fn test_count_with_compact_pointers() {
        let data = b"banana";
        let sa = build_suffix_array(data);
        let n = sa.len() as u64;
        // Use 2-byte pointers (sufficient for small data)
        let compact = compact_suffix_array(&sa, 2).unwrap();

        assert_eq!(count(data, &compact, 2, n, b"a"), 3);
        assert_eq!(count(data, &compact, 2, n, b"an"), 2);
        assert_eq!(count(data, &compact, 2, n, b"ban"), 1);
    }

    #[test]
    fn test_read_pointer_various_widths() {
        let val: u32 = 0x05060708;
        let bytes = val.to_le_bytes();

        // 1 byte
        assert_eq!(read_pointer(&bytes[..1], 0, 1), 0x08);
        // 2 bytes
        assert_eq!(read_pointer(&bytes[..2], 0, 2), 0x0708);
        // 4 bytes
        assert_eq!(read_pointer(&bytes, 0, 4), 0x05060708);
    }

    #[test]
    fn test_suffix_array_query_equality() {
        let q1 = SuffixArrayQuery::Count {
            query_bytes: vec![1, 2, 3],
        };
        let q2 = SuffixArrayQuery::Count {
            query_bytes: vec![1, 2, 3],
        };
        let q3 = SuffixArrayQuery::Count {
            query_bytes: vec![4, 5],
        };

        assert!(q1.dyn_eq(&q2));
        assert!(!q1.dyn_eq(&q3));
    }

    // ─── Language modeling query tests ──────────────────────────────────────────

    #[test]
    fn test_prob_basic() {
        // "abcabcabc" — "abc" appears 3 times, "abca" appears 2 times
        let data = b"abcabcabc";
        let (sa, n) = build_test_index(data);

        let result = prob(data, &sa, 4, n, b"abc", b"a");
        assert_eq!(result.prompt_cnt, 3); // "abc" appears 3 times
        assert_eq!(result.cont_cnt, 2); // "abca" appears 2 times
        assert!((result.prob - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_prob_zero_prompt() {
        let data = b"hello world";
        let (sa, n) = build_test_index(data);

        let result = prob(data, &sa, 4, n, b"xyz", b"a");
        assert_eq!(result.prompt_cnt, 0);
        assert_eq!(result.cont_cnt, 0);
        assert_eq!(result.prob, 0.0);
    }

    #[test]
    fn test_prob_zero_continuation() {
        let data = b"hello world";
        let (sa, n) = build_test_index(data);

        let result = prob(data, &sa, 4, n, b"hello", b"xyz");
        assert_eq!(result.prompt_cnt, 1);
        assert_eq!(result.cont_cnt, 0);
        assert_eq!(result.prob, 0.0);
    }

    #[test]
    fn test_prob_certain() {
        // "aaaa" — "aa" appears 3 times, "aaa" appears 2 times
        let data = b"aaaa";
        let (sa, n) = build_test_index(data);

        let result = prob(data, &sa, 4, n, b"aaa", b"a");
        assert_eq!(result.prompt_cnt, 2); // "aaa" appears 2 times
        assert_eq!(result.cont_cnt, 1); // "aaaa" appears 1 time
        assert!((result.prob - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_ntd_basic() {
        // "abacad" — after "a": b, c, d appear once each
        let data = b"abacad";
        let (sa, n) = build_test_index(data);

        let result = next_byte_distribution(data, &sa, 4, n, b"a", None);
        assert_eq!(result.prompt_cnt, 3); // "a" appears 3 times
        assert!(!result.approximate);
        assert_eq!(result.distribution.len(), 3); // b, c, d

        // Verify all three bytes are present
        let bytes: Vec<u8> = result.distribution.iter().map(|e| e.byte_value).collect();
        assert!(bytes.contains(&b'b'));
        assert!(bytes.contains(&b'c'));
        assert!(bytes.contains(&b'd'));

        // Each has count 1
        for entry in &result.distribution {
            assert_eq!(entry.count, 1);
            assert!((entry.prob - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ntd_single_next() {
        // "banana" — after "an": only 'a' follows (twice)
        let data = b"banana";
        let (sa, n) = build_test_index(data);

        let result = next_byte_distribution(data, &sa, 4, n, b"an", None);
        assert_eq!(result.prompt_cnt, 2);
        assert_eq!(result.distribution.len(), 1);
        assert_eq!(result.distribution[0].byte_value, b'a');
        assert_eq!(result.distribution[0].count, 2);
        assert!((result.distribution[0].prob - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ntd_no_match() {
        let data = b"hello";
        let (sa, n) = build_test_index(data);

        let result = next_byte_distribution(data, &sa, 4, n, b"xyz", None);
        assert_eq!(result.prompt_cnt, 0);
        assert!(result.distribution.is_empty());
    }

    #[test]
    fn test_ntd_approximate() {
        // Use a larger corpus so we can set a small max_support
        let data = b"abababababababababab"; // "ab" repeated 10 times (minus last char)
        let (sa, n) = build_test_index(data);

        let result = next_byte_distribution(data, &sa, 4, n, b"a", Some(3));
        // "a" appears 10 times in this string, max_support=3 triggers approximate mode
        assert!(result.approximate);
    }

    #[test]
    fn test_ntd_end_of_corpus() {
        // The last "a" in "banana" has no following byte
        let data = b"banana";
        let (sa, n) = build_test_index(data);

        let result = next_byte_distribution(data, &sa, 4, n, b"na", None);
        // "na" appears twice: "nan" and "na$" (end of string)
        assert_eq!(result.prompt_cnt, 2);
        // Only one entry should be present (for 'n'), the other "na" is at end
        // "na" at pos 2 → next byte 'a', "na" at pos 4 → end of string (no next byte)
        assert_eq!(result.distribution.len(), 1);
        assert_eq!(result.distribution[0].byte_value, b'n');
        // wait — "na" at pos 2 is followed by 'n' ("nan") and "na" at pos 4 is end of string
        // Actually pos 2 is "nana" so next byte is 'n', and pos 4 is "na" at end
    }

    #[test]
    fn test_infgram_prob_full_prompt_found() {
        // "abcabcabc" — full prompt "abc" exists
        let data = b"abcabcabc";
        let (sa, n) = build_test_index(data);

        let result = infgram_prob(data, &sa, 4, n, b"abc", b"a");
        assert_eq!(result.effective_suffix_len, 3); // Full prompt "abc" is found
        assert_eq!(result.prob_result.prompt_cnt, 3);
        assert_eq!(result.prob_result.cont_cnt, 2);
    }

    #[test]
    fn test_infgram_prob_backoff() {
        // Corpus that has "bc" but not "abc"
        let data = b"xbcybcy";
        let (sa, n) = build_test_index(data);

        // "abc" is not in the corpus, but "bc" is (2 times)
        let result = infgram_prob(data, &sa, 4, n, b"abc", b"y");
        assert_eq!(result.effective_suffix_len, 2); // Backed off to "bc"
        assert_eq!(result.prob_result.prompt_cnt, 2); // "bc" appears 2 times
        assert_eq!(result.prob_result.cont_cnt, 2); // "bcy" appears 2 times
    }

    #[test]
    fn test_infgram_prob_no_match() {
        let data = b"hello";
        let (sa, n) = build_test_index(data);

        // Nothing in the prompt exists in the corpus as a suffix
        let result = infgram_prob(data, &sa, 4, n, b"xyz", b"a");
        assert_eq!(result.effective_suffix_len, 0);
        assert_eq!(result.prob_result.prompt_cnt, 0);
        assert_eq!(result.prob_result.prob, 0.0);
    }

    #[test]
    fn test_infgram_prob_empty_prompt() {
        let data = b"hello";
        let (sa, n) = build_test_index(data);

        let result = infgram_prob(data, &sa, 4, n, b"", b"h");
        assert_eq!(result.effective_suffix_len, 0);
        // With empty prompt, count("") = n, count("h") = 1
        assert_eq!(result.prob_result.cont_cnt, 1);
    }

    #[test]
    fn test_infgram_prob_single_byte_backoff() {
        // "abcdef" — prompt "xyz" doesn't exist, but "z" doesn't either
        let data = b"abcdef";
        let (sa, n) = build_test_index(data);

        let result = infgram_prob(data, &sa, 4, n, b"xyz", b"a");
        // None of x, y, z exist in "abcdef" — wait, no.
        // Actually let's check: does 'z' exist? No. Does 'y' exist? No. Does 'x' exist? No.
        assert_eq!(result.effective_suffix_len, 0);
        assert_eq!(result.prob_result.prob, 0.0);
    }

    #[test]
    fn test_prob_consistency_with_count() {
        // Verify that prob() results are consistent with count()
        let data = b"the cat sat on the mat the cat";
        let (sa, n) = build_test_index(data);

        let prompt = b"the ";
        let continuation = b"cat";
        let result = prob(data, &sa, 4, n, prompt, continuation);

        let prompt_cnt = count(data, &sa, 4, n, prompt);
        let mut full = Vec::new();
        full.extend_from_slice(prompt);
        full.extend_from_slice(continuation);
        let cont_cnt = count(data, &sa, 4, n, &full);

        assert_eq!(result.prompt_cnt, prompt_cnt);
        assert_eq!(result.cont_cnt, cont_cnt);
        assert!((result.prob - cont_cnt as f64 / prompt_cnt as f64).abs() < 1e-10);
    }

    #[test]
    fn test_ntd_consistency_with_count() {
        // Verify that ntd distribution counts sum to prompt_cnt (minus end-of-corpus entries)
        let data = b"abcabdabe";
        let (sa, n) = build_test_index(data);

        let result = next_byte_distribution(data, &sa, 4, n, b"ab", None);
        assert_eq!(result.prompt_cnt, 3); // "ab" appears 3 times

        // The sum of distribution counts should equal prompt_cnt
        // (assuming none of the "ab" occurrences are at the very end of the corpus)
        let total: u64 = result.distribution.iter().map(|e| e.count).sum();
        assert_eq!(total, 3); // All 3 "ab" occurrences have a following byte

        // Verify each entry matches count()
        for entry in &result.distribution {
            let mut query = Vec::from(b"ab".as_slice());
            query.push(entry.byte_value);
            let cnt = count(data, &sa, 4, n, &query);
            assert_eq!(cnt, entry.count);
        }
    }
}
