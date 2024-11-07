// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub fn is_pwr_two(n: u64) -> bool {
    n & (n - 1) == 0
}

pub fn pad_bytes<const ALIGN: usize>(n: usize) -> usize {
    debug_assert!(is_pwr_two(ALIGN as u64));
    (ALIGN - (n & (ALIGN - 1))) & (ALIGN - 1)
}

pub fn pad_bytes_u64<const ALIGN: u64>(n: u64) -> u64 {
    debug_assert!(is_pwr_two(ALIGN));
    (ALIGN - (n & (ALIGN - 1))) & (ALIGN - 1)
}
