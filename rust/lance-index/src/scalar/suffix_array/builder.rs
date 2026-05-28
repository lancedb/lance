// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Suffix array construction using libsais (Ilya Grebnov).
//!
//! Uses the vendored libsais C library for O(N) suffix array construction.
//! libsais implements the SA-IS algorithm with extensive SIMD optimizations
//! and cache-aware memory access patterns, achieving state-of-the-art
//! performance (~3-5× faster than a naive Rust SA-IS implementation).
//!
//! This is the same class of algorithm used by the original infini-gram
//! paper (which used libdivsufsort). libsais is the modern successor,
//! typically 2× faster than libdivsufsort.
//!
//! Memory usage: approximately 5×N for an N-byte input:
//!   - input data: N bytes (borrowed, not copied)
//!   - SA array (i32): 4N bytes

use lance_core::{Error, Result};

// FFI binding to vendored libsais C library (compiled via build.rs)
unsafe extern "C" {
    /// Constructs the suffix array of a given byte string.
    /// T: input bytes [0..n-1]
    /// SA: output suffix array [0..n-1+fs]
    /// n: length of input
    /// fs: extra space at end of SA (0 is enough for most cases)
    /// freq: optional output frequency table (NULL if not needed)
    /// Returns 0 on success, -1 or -2 on error.
    fn libsais(
        T: *const u8,
        SA: *mut i32,
        n: i32,
        fs: i32,
        freq: *mut i32,
    ) -> i32;
}

/// Build a suffix array for byte data using libsais.
///
/// Uses Ilya Grebnov's libsais library — the fastest known implementation
/// of the SA-IS algorithm with SIMD optimizations. Runs in O(N) time.
///
/// Returns an array of u32 starting positions sorted by their corresponding
/// suffixes. Since segments are limited to < 2 GB (i32::MAX), u32 pointers suffice.
pub fn build_suffix_array(data: &[u8]) -> Vec<u32> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    assert!(
        n <= i32::MAX as usize,
        "build_suffix_array: input must be < 2 GB for libsais (i32 addressing), got {n} bytes"
    );

    let mut sa: Vec<i32> = vec![0i32; n];

    let ret = unsafe {
        libsais(
            data.as_ptr(),
            sa.as_mut_ptr(),
            n as i32,
            0,      // no extra space needed
            std::ptr::null_mut(), // no frequency table needed
        )
    };

    assert!(
        ret == 0,
        "libsais failed with error code {ret} for input of {n} bytes"
    );

    // Convert i32 -> u32 (all values are non-negative positions 0..n-1)
    // Safety: i32 and u32 have the same size and alignment, and all values
    // are non-negative, so this is a safe transmute.
    let sa_u32: Vec<u32> = sa.into_iter().map(|x| x as u32).collect();
    sa_u32
}

/// Naive O(N log N) suffix array construction (test reference).
#[cfg(test)]
fn build_suffix_array_naive(data: &[u8]) -> Vec<u32> {
    let n = data.len();
    let mut sa: Vec<u32> = (0..n as u32).collect();
    sa.sort_by(|&a, &b| data[a as usize..].cmp(&data[b as usize..]));
    sa
}

/// Compact a suffix array (u32) into variable-width pointers.
///
/// Each pointer is stored in `pointer_width` bytes using little-endian encoding.
pub fn compact_suffix_array(sa: &[u32], pointer_width: usize) -> Result<Vec<u8>> {
    if pointer_width == 0 || pointer_width > 4 {
        return Err(Error::invalid_input(format!(
            "pointer_width must be between 1 and 4, got {pointer_width}"
        )));
    }

    let max_value = if pointer_width == 4 {
        u32::MAX
    } else {
        (1u32 << (pointer_width * 8)) - 1
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
pub fn compute_pointer_width(corpus_bytes: u64) -> usize {
    if corpus_bytes == 0 {
        return 1;
    }
    let max_val = corpus_bytes.saturating_sub(1);
    if max_val > u32::MAX as u64 {
        4
    } else {
        let bits_needed = 32 - (max_val as u32).leading_zeros() as usize;
        let bytes_needed = bits_needed.div_ceil(8);
        bytes_needed.max(1)
    }
}

/// Release freed memory back to the OS.
pub fn release_memory_to_os() {
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        unsafe {
            malloc_trim(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_suffix_array_basic() {
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
        assert_eq!(sa, (0..256u32).collect::<Vec<_>>());
    }

    #[test]
    fn test_libsais_correctness_medium() {
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
    fn test_libsais_vs_naive() {
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
            let libsais_result = build_suffix_array(data);
            let naive_result = build_suffix_array_naive(data);
            assert_eq!(
                libsais_result,
                naive_result,
                "libsais and naive differ for {:?}",
                std::str::from_utf8(data).unwrap_or("<binary>")
            );
        }
    }

    #[test]
    fn test_libsais_large_random() {
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
        let sa = vec![0u32, 100, 200, 255];
        let compact = compact_suffix_array(&sa, 2).unwrap();
        assert_eq!(compact.len(), 8);

        for (i, &expected) in sa.iter().enumerate() {
            let offset = i * 2;
            let mut buf = [0u8; 4];
            buf[..2].copy_from_slice(&compact[offset..offset + 2]);
            let val = u32::from_le_bytes(buf);
            assert_eq!(val, expected);
        }
    }

    #[test]
    fn test_compact_suffix_array_overflow() {
        let sa = vec![256u32];
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
    }

    #[test]
    fn test_libsais_100kb_random() {
        // Test with 100KB of random data to exercise libsais on a larger input
        let mut data = Vec::with_capacity(100_000);
        let mut seed: u64 = 42;
        for _ in 0..100_000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            data.push(((seed >> 33) & 0xFF) as u8);
        }

        let sa = build_suffix_array(&data);
        assert_eq!(sa.len(), data.len());
        // Verify sorted order for a sample of positions
        for i in 1..sa.len().min(1000) {
            let s1 = &data[sa[i - 1] as usize..];
            let s2 = &data[sa[i] as usize..];
            assert!(s1 < s2, "SA not sorted at position {i}");
        }
    }
}
