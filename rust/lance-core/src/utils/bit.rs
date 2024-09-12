// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub fn is_pwr_two(n: u64) -> bool {
    n & (n - 1) == 0
}
