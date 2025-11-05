// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[allow(unused_imports)]
use lance_core::utils::cpu::{SimdSupport, SIMD_SUPPORT};

pub const PERM0: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];
pub const PERM0_INVERSE: [usize; 16] = [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15];
pub const BATCH_SIZE: usize = 32;

// This function is used to sum the distance table for 4-bit codes.
// the distance table is a 2D array, that dist_table[i][j] is the distance between the i-th subvector and the code j,
// the distance table is stored as a flat array for better cache locality and SIMD instruction usage.
//
// The codes are organized in the order of PERM0:
// +----------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
// | address  |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 |
// | (bytes)  |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
// +----------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
// | bits 0..3|  0 |  8 |  1 |  9 |  2 | 10 |  3 | 11 |  4 | 12 |  5 | 13 |  6 | 14 |  7 | 15 |
// | bits 4..7| 16 | 24 | 17 | 25 | 18 | 26 | 19 | 27 | 20 | 28 | 21 | 29 | 22 | 30 | 23 | 31 |
// +----------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
// so that we can use SIMD instruction (especially _mm256_shuffle_epi8) to do the summation.
#[inline]
pub fn sum_4bit_dist_table(
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    dists: &mut [u16],
) {
    debug_assert!(n.is_multiple_of(BATCH_SIZE));

    match *SIMD_SUPPORT {
        #[cfg(all(kernel_support = "avx512", target_arch = "x86_64"))]
        SimdSupport::Avx512 | SimdSupport::Avx512FP16 => unsafe {
            for i in (0..n).step_by(BATCH_SIZE) {
                let codes = &codes[i * code_len..(i + BATCH_SIZE) * code_len];
                sum_4bit_dist_table_32bytes_batch_avx512(
                    codes.as_ptr(),
                    codes.len(),
                    dist_table.as_ptr(),
                    dists[i..i + BATCH_SIZE].as_mut_ptr(),
                )
            }
        },
        #[cfg(target_arch = "x86_64")]
        SimdSupport::Avx2 => unsafe {
            for i in (0..n).step_by(BATCH_SIZE) {
                sum_dist_table_32bytes_batch_avx2(
                    &codes[i * code_len..(i + BATCH_SIZE) * code_len],
                    dist_table,
                    &mut dists[i..i + BATCH_SIZE],
                )
            }
        },
        _ => sum_4bit_dist_table_scalar(code_len, codes, dist_table, dists),
    }
}

#[inline]
#[allow(unused)]
fn sum_4bit_dist_table_scalar(code_len: usize, codes: &[u8], dist_table: &[u8], dists: &mut [u16]) {
    for (vec_block_idx, blocks) in codes.chunks_exact(BATCH_SIZE * code_len).enumerate() {
        for (sub_vec_idx, block) in blocks.chunks_exact(BATCH_SIZE).enumerate() {
            let current_dist_table = &dist_table[sub_vec_idx * 2 * 16..(sub_vec_idx * 2 + 1) * 16];
            let next_dist_table =
                &dist_table[(sub_vec_idx * 2 + 1) * 16..(sub_vec_idx * 2 + 2) * 16];

            for j in 0..16 {
                let low_current_code = (block[j] & 0x0F) as usize;
                let high_current_code = (block[j] >> 4) as usize;
                let low_next_code = (block[j + 16] & 0x0F) as usize;
                let high_next_code = (block[j + 16] >> 4) as usize;

                let lower_id = vec_block_idx * BATCH_SIZE + PERM0[j];
                let higher_id = vec_block_idx * BATCH_SIZE + PERM0[j] + 16;
                dists[lower_id] = dists[lower_id]
                    .saturating_add(current_dist_table[low_current_code] as u16)
                    .saturating_add(next_dist_table[low_next_code] as u16);
                dists[higher_id] = dists[higher_id]
                    .saturating_add(current_dist_table[high_current_code] as u16)
                    .saturating_add(next_dist_table[high_next_code] as u16);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unused)]
unsafe fn sum_dist_table_32bytes_batch_avx2(codes: &[u8], dist_table: &[u8], dists: &mut [u16]) {
    let mut c = _mm256_undefined_si256();
    let mut lo = _mm256_undefined_si256();
    let mut hi = _mm256_undefined_si256();
    let mut lut_vec = _mm256_undefined_si256();
    let mut res_lo = _mm256_undefined_si256();
    let mut res_hi = _mm256_undefined_si256();

    let mut accu0 = _mm256_setzero_si256();
    let mut accu1 = _mm256_setzero_si256();
    let mut accu2 = _mm256_setzero_si256();
    let mut accu3 = _mm256_setzero_si256();
    let low_mask = _mm256_set1_epi8(0x0f);

    for i in (0..codes.len()).step_by(64) {
        // load 32 * 2 codes (we pack 2 codes into 1 byte)
        c = _mm256_loadu_si256(codes.as_ptr().add(i) as *const __m256i);
        lut_vec = _mm256_loadu_si256(dist_table.as_ptr().add(i) as *const __m256i);

        // split the first 4 bits and the second 4 bits
        lo = _mm256_and_si256(c, low_mask);
        hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        // lookup the lut
        res_lo = _mm256_shuffle_epi8(lut_vec, lo);
        res_hi = _mm256_shuffle_epi8(lut_vec, hi);

        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));

        // load the left 32 bytes of codes and lut
        c = _mm256_loadu_si256(codes.as_ptr().add(i + 32) as *const __m256i);
        lut_vec = _mm256_loadu_si256(dist_table.as_ptr().add(i + 32) as *const __m256i);

        lo = _mm256_and_si256(c, low_mask);
        hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        res_lo = _mm256_shuffle_epi8(lut_vec, lo);
        res_hi = _mm256_shuffle_epi8(lut_vec, hi);

        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));
    }

    // merge the low 4 bits
    accu0 = _mm256_sub_epi16(accu0, _mm256_slli_epi16(accu1, 8));
    let dis0 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu0, accu1, 0x21),
        _mm256_blend_epi32(accu0, accu1, 0xF0),
    );
    _mm256_storeu_si256(dists.as_mut_ptr() as *mut __m256i, dis0);

    // merge the high 4 bits
    accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));
    let dis1 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu2, accu3, 0x21),
        _mm256_blend_epi32(accu2, accu3, 0xF0),
    );

    _mm256_storeu_si256(dists.as_mut_ptr().add(16) as *mut __m256i, dis1);
}

// We implement the AVX512 version in C because AVX512 is not stable yet in Rust,
// implement it in Rust once we upgrade rust to 1.89.0.
extern "C" {
    #[cfg(all(kernel_support = "avx512", target_arch = "x86_64"))]
    pub fn sum_4bit_dist_table_32bytes_batch_avx512(
        codes: *const u8,
        code_length: usize,
        dist_table: *const u8,
        dists: *mut u16,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_4bit_dist_table_basic() {
        // we have 32 vectors
        let n = 32;

        // each code is 2 bytes (16 dim), so code_len = 2
        let code_len = 2;

        let codes = [
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, // codes[0..8]
            0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, // codes[8..16]
            0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, // codes[16..24]
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, // codes[24..32]
        ];
        let codes = codes.repeat(n * code_len / codes.len());

        let mut dist_table = vec![0u8; 16 * 4];
        for (i, dist) in dist_table.iter_mut().enumerate() {
            *dist = (i % 16 + 1) as u8;
        }

        // Test the function
        let mut dists = vec![0u16; n];
        sum_4bit_dist_table(n, code_len, &codes, &dist_table, &mut dists);

        // Compare with reference implementation
        let mut expected_dists = vec![0u16; n];
        sum_4bit_dist_table_scalar(code_len, &codes, &dist_table, &mut expected_dists);

        assert_eq!(dists, expected_dists);
        // the vector 1's code is the low 4bits of codes[PERM0_INVERSE[1]] = codes[2],
        // the first 4 bits are the low 4 bits of codes[2], so it's 0x6,
        // the second 4 bits are the low 4 bits of codes[2 + 16], so it's 0xb,
        // the third 4 bits are the same as the first 4 bits, so it's 0x6,
        // the fourth 4 bits are the same as the second 4 bits, so it's 0xb,

        // so the distance is 2 * (dist_table[0x6] + dist_table[0xb + 16]) = 2*(7 + 12) = 38
        assert_eq!(dists[1], 38);
    }
}
