// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Suffix array construction using the SA-IS algorithm.
//!
//! Implements the Suffix Array by Induced Sorting (SA-IS) algorithm from
//! Nong, Zhang & Chan (2009). This runs in O(N) time and O(N) space,
//! matching the approach used by the original infini-gram engine.
//!
//! The algorithm works on a virtual string T$, where $ is a sentinel character
//! smaller than any character in the alphabet. The sentinel is implicit — we
//! never actually append it. Instead, the last position is always treated as
//! an LMS suffix of its own, sorted before everything else in bucket 0 (or
//! whatever character it holds).

use lance_core::{Error, Result};

/// Alphabet size for byte-level indexing.
const ALPHABET_SIZE: usize = 256;

/// Build a suffix array for byte data using the SA-IS algorithm.
///
/// Returns an array of starting positions sorted by their corresponding suffixes.
/// Runs in O(N) time and O(N) space.
pub fn build_suffix_array(data: &[u8]) -> Vec<u64> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    // SA-IS works with u32 positions. For inputs > 4GB, fall back to naive sort.
    if n <= u32::MAX as usize {
        // Append a sentinel byte (0) that is guaranteed to be <= any character.
        // This is the standard SA-IS approach: the input is T$ where $ < all
        // characters. We use 0 as sentinel since our byte alphabet is 0..=255,
        // but we place the sentinel at rank -1 by giving it special handling.
        //
        // Actually, the cleanest approach: build SA on an integer array where
        // we map each byte to byte+1, then append a 0 sentinel. This ensures
        // the sentinel is strictly smaller than all characters.
        let mut text: Vec<u32> = Vec::with_capacity(n + 1);
        for &b in data {
            text.push(b as u32 + 1); // shift alphabet to 1..=256
        }
        text.push(0); // sentinel, strictly smallest

        let sa = sais_int(&text, ALPHABET_SIZE + 1); // alphabet 0..=256

        // Remove the sentinel from the result (it's always at SA[0])
        let mut result: Vec<u64> = Vec::with_capacity(n);
        for &pos in &sa {
            if (pos as usize) < n {
                result.push(pos as u64);
            }
        }
        result
    } else {
        build_suffix_array_naive(data)
    }
}

/// Naive O(N log²N) suffix array construction (fallback and test reference).
fn build_suffix_array_naive(data: &[u8]) -> Vec<u64> {
    let n = data.len();
    let mut sa: Vec<u64> = (0..n as u64).collect();
    sa.sort_by(|&a, &b| data[a as usize..].cmp(&data[b as usize..]));
    sa
}

/// SA-IS for integer alphabets. Input values must be in 0..alphabet_size.
/// The last character MUST be a sentinel (value 0) that appears exactly once
/// and is the smallest character.
fn sais_int(text: &[u32], alphabet_size: usize) -> Vec<u32> {
    let n = text.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }
    if n == 2 {
        // text[0] > text[1] always (since text[1] is sentinel = 0)
        return vec![1, 0];
    }

    // Step 1: Classify each position as S-type or L-type.
    // S-type: suffix[i] < suffix[i+1] lexicographically
    // L-type: suffix[i] > suffix[i+1]
    // The sentinel (last position) is always S-type.
    let mut is_s = vec![false; n];
    is_s[n - 1] = true; // sentinel is S-type

    for i in (0..n - 1).rev() {
        is_s[i] = if text[i] < text[i + 1] {
            true
        } else if text[i] > text[i + 1] {
            false
        } else {
            is_s[i + 1]
        };
    }

    // Step 2: Find LMS (Left-Most S-type) positions.
    // Position i is LMS if is_s[i] && !is_s[i-1].
    // The sentinel position (n-1) is always LMS.
    let mut is_lms = vec![false; n];
    let mut lms_count = 0usize;
    is_lms[n - 1] = true; // sentinel
    lms_count += 1;
    for i in 1..n - 1 {
        if is_s[i] && !is_s[i - 1] {
            is_lms[i] = true;
            lms_count += 1;
        }
    }

    // Step 3: Compute bucket boundaries.
    let bucket_sizes = compute_bucket_sizes(text, alphabet_size);

    // Step 4: Place LMS suffixes and do induced sorting.
    let mut sa = vec![u32::MAX; n];

    // Place LMS suffixes at bucket tails (right-to-left scan of text)
    {
        let mut tails = get_bucket_tails(&bucket_sizes);
        // Place sentinel first
        let c = text[n - 1] as usize;
        sa[tails[c]] = (n - 1) as u32;
        tails[c] = tails[c].wrapping_sub(1);
        // Place other LMS suffixes right-to-left
        for i in (1..n - 1).rev() {
            if is_lms[i] {
                let c = text[i] as usize;
                sa[tails[c]] = i as u32;
                tails[c] = tails[c].wrapping_sub(1);
            }
        }
    }

    // Induce L-type suffixes
    induce_l(&mut sa, text, &is_s, &bucket_sizes);
    // Induce S-type suffixes
    induce_s(&mut sa, text, &is_s, &bucket_sizes);

    // Step 5: Compact sorted LMS suffixes and assign names.
    if lms_count <= 1 {
        return sa;
    }

    // Collect sorted LMS positions from SA
    let mut sorted_lms: Vec<u32> = Vec::with_capacity(lms_count);
    for &pos in &sa {
        if pos != u32::MAX && is_lms[pos as usize] {
            sorted_lms.push(pos);
        }
    }

    // Assign names to LMS substrings
    let mut name_map = vec![u32::MAX; n]; // position -> name
    let mut current_name: u32 = 0;
    name_map[sorted_lms[0] as usize] = current_name;

    for i in 1..sorted_lms.len() {
        if !lms_substrings_equal(text, &is_s, &is_lms, sorted_lms[i - 1], sorted_lms[i]) {
            current_name += 1;
        }
        name_map[sorted_lms[i] as usize] = current_name;
    }

    let num_unique = current_name as usize + 1;

    // Step 6: Get the correct LMS order.
    // The first pass gives us the correct relative order of LMS suffixes
    // (in sorted_lms), but the full SA might not be correct.
    // If names are not all unique, recurse on the reduced string to
    // determine the correct order. Otherwise, sorted_lms already has it.
    let sorted_lms_final = if num_unique < lms_count {
        // Build reduced string: names of LMS substrings in text order
        let mut reduced: Vec<u32> = Vec::with_capacity(lms_count);
        let mut lms_positions: Vec<u32> = Vec::with_capacity(lms_count);
        for i in 0..n {
            if is_lms[i] {
                lms_positions.push(i as u32);
                reduced.push(name_map[i]);
            }
        }

        // Recursively sort
        let reduced_sa = sais_int(&reduced, num_unique);

        // Reorder LMS positions according to recursive result
        let mut result: Vec<u32> = Vec::with_capacity(lms_count);
        for &rank in &reduced_sa {
            result.push(lms_positions[rank as usize]);
        }
        result
    } else {
        // All names unique — sorted_lms from the first pass is already correct
        sorted_lms
    };

    // Step 7: Redo induced sorting with the correct LMS order.
    // This ALWAYS runs — the first pass only determines LMS order,
    // not the final SA.
    sa.fill(u32::MAX);
    {
        let mut tails = get_bucket_tails(&bucket_sizes);
        // Place LMS in correct order (right-to-left for stability)
        for &pos in sorted_lms_final.iter().rev() {
            let c = text[pos as usize] as usize;
            sa[tails[c]] = pos;
            tails[c] = tails[c].wrapping_sub(1);
        }
    }
    induce_l(&mut sa, text, &is_s, &bucket_sizes);
    induce_s(&mut sa, text, &is_s, &bucket_sizes);

    sa
}

// ─── Bucket helpers ─────────────────────────────────────────────────────────

fn compute_bucket_sizes(text: &[u32], alphabet_size: usize) -> Vec<usize> {
    let mut sizes = vec![0usize; alphabet_size];
    for &c in text {
        sizes[c as usize] += 1;
    }
    sizes
}

/// Get bucket head positions (start of each bucket).
fn get_bucket_heads(sizes: &[usize]) -> Vec<usize> {
    let mut heads = Vec::with_capacity(sizes.len());
    let mut offset = 0usize;
    for &s in sizes {
        heads.push(offset);
        offset += s;
    }
    heads
}

/// Get bucket tail positions (last valid index in each bucket).
/// For empty buckets, returns usize::MAX as a sentinel (should never be accessed).
fn get_bucket_tails(sizes: &[usize]) -> Vec<usize> {
    let mut tails = Vec::with_capacity(sizes.len());
    let mut offset = 0usize;
    for &s in sizes {
        offset += s;
        tails.push(if s > 0 { offset - 1 } else { usize::MAX });
    }
    tails
}

// ─── Induced sorting ────────────────────────────────────────────────────────

/// Induce L-type suffixes: scan SA left-to-right.
fn induce_l(sa: &mut [u32], text: &[u32], is_s: &[bool], bucket_sizes: &[usize]) {
    let n = text.len();
    let mut heads = get_bucket_heads(bucket_sizes);

    for i in 0..n {
        if sa[i] == u32::MAX {
            continue;
        }
        let pos = sa[i] as usize;
        if pos == 0 {
            continue;
        }
        let j = pos - 1;
        if !is_s[j] {
            let c = text[j] as usize;
            sa[heads[c]] = j as u32;
            heads[c] += 1;
        }
    }
}

/// Induce S-type suffixes: scan SA right-to-left.
fn induce_s(sa: &mut [u32], text: &[u32], is_s: &[bool], bucket_sizes: &[usize]) {
    let n = text.len();
    let mut tails = get_bucket_tails(bucket_sizes);

    for i in (0..n).rev() {
        if sa[i] == u32::MAX {
            continue;
        }
        let pos = sa[i] as usize;
        if pos == 0 {
            continue;
        }
        let j = pos - 1;
        if is_s[j] {
            let c = text[j] as usize;
            sa[tails[c]] = j as u32;
            tails[c] = tails[c].wrapping_sub(1);
        }
    }
}

// ─── LMS substring comparison ───────────────────────────────────────────────

/// Check if two LMS substrings are equal.
/// An LMS substring starts at an LMS position and extends through the next LMS
/// position (inclusive).
fn lms_substrings_equal(
    text: &[u32],
    is_s: &[bool],
    is_lms: &[bool],
    pos1: u32,
    pos2: u32,
) -> bool {
    let n = text.len();
    let mut i = pos1 as usize;
    let mut j = pos2 as usize;
    let mut first = true;

    loop {
        // Compare character and S/L type at current position
        if text[i] != text[j] || is_s[i] != is_s[j] {
            return false;
        }
        // After confirming equality, check if both reached the next LMS boundary.
        // Skip this check on the first character (which is the starting LMS position).
        if !first {
            let i_is_lms = is_lms[i];
            let j_is_lms = is_lms[j];
            if i_is_lms && j_is_lms {
                return true; // Same length, all characters matched
            }
            if i_is_lms != j_is_lms {
                return false; // Different lengths
            }
        }
        first = false;
        i += 1;
        j += 1;
        if i >= n || j >= n {
            return i >= n && j >= n;
        }
    }
}

// ─── Compact suffix array and pointer width (unchanged) ─────────────────────

/// Compact a suffix array into variable-width pointers.
///
/// Each pointer is stored in `pointer_width` bytes using little-endian encoding.
/// This reduces memory usage when the corpus is smaller than what 8-byte pointers
/// would require.
pub fn compact_suffix_array(sa: &[u64], pointer_width: usize) -> Result<Vec<u8>> {
    if pointer_width == 0 || pointer_width > 8 {
        return Err(Error::invalid_input(format!(
            "pointer_width must be between 1 and 8, got {pointer_width}"
        )));
    }

    let max_value = if pointer_width == 8 {
        u64::MAX
    } else {
        (1u64 << (pointer_width * 8)) - 1
    };

    let mut result = Vec::with_capacity(sa.len() * pointer_width);
    for &ptr in sa {
        if ptr > max_value {
            return Err(Error::invalid_input(format!(
                "suffix array pointer value {ptr} exceeds maximum for {pointer_width}-byte width (max {max_value})"
            )));
        }
        let bytes = ptr.to_le_bytes();
        result.extend_from_slice(&bytes[..pointer_width]);
    }
    Ok(result)
}

/// Calculate the minimum pointer width needed to address `corpus_bytes` positions.
///
/// Returns the number of bytes needed to represent any offset within the corpus.
pub fn compute_pointer_width(corpus_bytes: u64) -> usize {
    if corpus_bytes == 0 {
        return 1;
    }
    let max_val = corpus_bytes.saturating_sub(1);
    let bits_needed = 64 - max_val.leading_zeros() as usize;
    let bytes_needed = bits_needed.div_ceil(8);
    bytes_needed.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_suffix_array_basic() {
        // "banana" -> suffixes sorted: "a", "ana", "anana", "banana", "na", "nana"
        // positions:                     5,   3,     1,       0,       4,    2
        let data = b"banana";
        let sa = build_suffix_array(data);
        assert_eq!(sa.len(), 6);
        assert_eq!(sa, vec![5, 3, 1, 0, 4, 2]);
    }

    #[test]
    fn test_build_suffix_array_empty() {
        let sa = build_suffix_array(b"");
        assert!(sa.is_empty());
    }

    #[test]
    fn test_build_suffix_array_single_char() {
        let sa = build_suffix_array(b"a");
        assert_eq!(sa, vec![0]);
    }

    #[test]
    fn test_build_suffix_array_repeated() {
        let data = b"aaa";
        let sa = build_suffix_array(data);
        assert_eq!(sa, vec![2, 1, 0]);
    }

    #[test]
    fn test_build_suffix_array_mississippi() {
        let data = b"mississippi";
        let sa = build_suffix_array(data);
        for i in 1..sa.len() {
            let s1 = &data[sa[i - 1] as usize..];
            let s2 = &data[sa[i] as usize..];
            assert!(s1 < s2, "SA not sorted at {i}: {:?} >= {:?}", s1, s2);
        }
        assert_eq!(sa, vec![10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2]);
    }

    #[test]
    fn test_build_suffix_array_all_same() {
        let data = b"aaaaaaa";
        let sa = build_suffix_array(data);
        assert_eq!(sa, vec![6, 5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn test_build_suffix_array_two_chars() {
        let data = b"ab";
        let sa = build_suffix_array(data);
        assert_eq!(sa, vec![0, 1]);
    }

    #[test]
    fn test_build_suffix_array_reverse_sorted() {
        let data = b"dcba";
        let sa = build_suffix_array(data);
        assert_eq!(sa, vec![3, 2, 1, 0]);
    }

    #[test]
    fn test_build_suffix_array_natural_text() {
        let data = b"the cat sat on the mat";
        let sa = build_suffix_array(data);
        for i in 1..sa.len() {
            let s1 = &data[sa[i - 1] as usize..];
            let s2 = &data[sa[i] as usize..];
            assert!(s1 < s2, "SA not sorted at {i}");
        }
    }

    #[test]
    fn test_build_suffix_array_binary_data() {
        let data: Vec<u8> = (0..=255).collect();
        let sa = build_suffix_array(&data);
        assert_eq!(sa, (0..256u64).collect::<Vec<_>>());
    }

    #[test]
    fn test_sais_correctness_medium() {
        let data = b"abcabcabcabc";
        let sa = build_suffix_array(data);
        assert_eq!(sa.len(), data.len());
        for i in 1..sa.len() {
            let s1 = &data[sa[i - 1] as usize..];
            let s2 = &data[sa[i] as usize..];
            assert!(s1 < s2, "SA not sorted at position {i}");
        }
    }

    #[test]
    fn test_sais_vs_naive() {
        let test_cases: Vec<&[u8]> = vec![
            b"banana",
            b"mississippi",
            b"abracadabra",
            b"aaa",
            b"abcdef",
            b"fedcba",
            b"the quick brown fox jumps over the lazy dog",
            b"ababababab",
        ];

        for data in test_cases {
            let sais_result = build_suffix_array(data);
            let naive_result = build_suffix_array_naive(data);
            assert_eq!(
                sais_result,
                naive_result,
                "SA-IS and naive differ for {:?}",
                std::str::from_utf8(data).unwrap_or("<binary>")
            );
        }
    }

    #[test]
    fn test_sais_large_random() {
        // Test with a larger pseudo-random input
        let mut data = Vec::with_capacity(10_000);
        let mut seed: u32 = 12345;
        for _ in 0..10_000 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            data.push(((seed >> 16) & 0xFF) as u8);
        }

        let sa = build_suffix_array(&data);
        assert_eq!(sa.len(), data.len());
        for i in 1..sa.len() {
            let s1 = &data[sa[i - 1] as usize..];
            let s2 = &data[sa[i] as usize..];
            assert!(s1 < s2, "SA not sorted at position {i}");
        }
    }

    #[test]
    fn test_compact_suffix_array_roundtrip() {
        let sa = vec![0u64, 100, 200, 300];
        let compact = compact_suffix_array(&sa, 2).unwrap();
        assert_eq!(compact.len(), 8);

        for (i, &expected) in sa.iter().enumerate() {
            let offset = i * 2;
            let mut buf = [0u8; 8];
            buf[..2].copy_from_slice(&compact[offset..offset + 2]);
            let val = u64::from_le_bytes(buf);
            assert_eq!(val, expected);
        }
    }

    #[test]
    fn test_compact_suffix_array_overflow() {
        let sa = vec![256u64];
        let result = compact_suffix_array(&sa, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_pointer_width() {
        assert_eq!(compute_pointer_width(0), 1);
        assert_eq!(compute_pointer_width(1), 1);
        assert_eq!(compute_pointer_width(255), 1);
        assert_eq!(compute_pointer_width(256), 1);
        assert_eq!(compute_pointer_width(257), 2);
        assert_eq!(compute_pointer_width(65536), 2);
        assert_eq!(compute_pointer_width(65537), 3);
        assert_eq!(compute_pointer_width(1 << 32), 4);
        assert_eq!(compute_pointer_width((1 << 32) + 1), 5);
        assert_eq!(compute_pointer_width(1 << 40), 5);
        assert_eq!(compute_pointer_width((1 << 40) + 1), 6);
    }
}
