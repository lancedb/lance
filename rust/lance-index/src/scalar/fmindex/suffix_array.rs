// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Safe suffix-array construction using induced sorting.
//!
//! The implementation follows the SA-IS algorithm used by AtCoder Library,
//! which is available under CC0-1.0.

/// Builds the suffix array for `text` in linear time.
pub(super) fn build_suffix_array(text: &[u8]) -> Vec<usize> {
    sa_is(text, u8::MAX as usize)
}

trait Symbol: Copy + Ord {
    fn rank(self) -> usize;
}

impl Symbol for u8 {
    fn rank(self) -> usize {
        self as usize
    }
}

impl Symbol for usize {
    fn rank(self) -> usize {
        self
    }
}

fn sa_naive<T: Symbol>(symbols: &[T]) -> Vec<usize> {
    let mut suffix_array: Vec<_> = (0..symbols.len()).collect();
    suffix_array.sort_unstable_by(|&left, &right| symbols[left..].cmp(&symbols[right..]));
    suffix_array
}

fn sa_doubling<T: Symbol>(symbols: &[T]) -> Vec<usize> {
    let len = symbols.len();
    let mut suffix_array: Vec<_> = (0..len).collect();
    let mut ranks: Vec<_> = symbols.iter().map(|symbol| symbol.rank()).collect();
    let mut next_ranks = vec![0; len];
    let mut width = 1;

    while width < len {
        let rank_pair = |suffix: usize| {
            (
                ranks[suffix],
                suffix
                    .checked_add(width)
                    .filter(|&i| i < len)
                    .map(|i| ranks[i]),
            )
        };
        suffix_array.sort_unstable_by_key(|&suffix| rank_pair(suffix));
        next_ranks[suffix_array[0]] = 0;
        for pair in suffix_array.windows(2) {
            next_ranks[pair[1]] =
                next_ranks[pair[0]] + usize::from(rank_pair(pair[0]) < rank_pair(pair[1]));
        }
        std::mem::swap(&mut ranks, &mut next_ranks);
        width = width.saturating_mul(2);
    }
    suffix_array
}

fn sa_is<T: Symbol>(symbols: &[T], alphabet_max: usize) -> Vec<usize> {
    let len = symbols.len();
    match len {
        0 => return Vec::new(),
        1 => return vec![0],
        2 => {
            return if symbols[0] < symbols[1] {
                vec![0, 1]
            } else {
                vec![1, 0]
            };
        }
        3..10 => return sa_naive(symbols),
        10..40 => return sa_doubling(symbols),
        _ => {}
    }

    let mut suffix_array = vec![0; len];
    let mut is_s_type = vec![false; len];
    for index in (0..len - 1).rev() {
        is_s_type[index] = if symbols[index] == symbols[index + 1] {
            is_s_type[index + 1]
        } else {
            symbols[index] < symbols[index + 1]
        };
    }

    let mut bucket_l = vec![0; alphabet_max + 1];
    let mut bucket_s = vec![0; alphabet_max + 1];
    for index in 0..len {
        let rank = symbols[index].rank();
        if is_s_type[index] {
            bucket_l[rank + 1] += 1;
        } else {
            bucket_s[rank] += 1;
        }
    }
    for rank in 0..=alphabet_max {
        bucket_s[rank] += bucket_l[rank];
        if rank < alphabet_max {
            bucket_l[rank + 1] += bucket_s[rank];
        }
    }

    let induce = |suffix_array: &mut [usize], lms_suffixes: &[usize]| {
        suffix_array.fill(0);
        let mut buckets = bucket_s.clone();
        for &suffix in lms_suffixes {
            if suffix == len {
                continue;
            }
            let rank = symbols[suffix].rank();
            suffix_array[buckets[rank]] = suffix + 1;
            buckets[rank] += 1;
        }

        buckets.copy_from_slice(&bucket_l);
        let last_rank = symbols[len - 1].rank();
        suffix_array[buckets[last_rank]] = len;
        buckets[last_rank] += 1;
        for index in 0..len {
            let suffix = suffix_array[index];
            if suffix >= 2 && !is_s_type[suffix - 2] {
                let rank = symbols[suffix - 2].rank();
                suffix_array[buckets[rank]] = suffix - 1;
                buckets[rank] += 1;
            }
        }

        buckets.copy_from_slice(&bucket_l);
        for index in (0..len).rev() {
            let suffix = suffix_array[index];
            if suffix >= 2 && is_s_type[suffix - 2] {
                let rank = symbols[suffix - 2].rank() + 1;
                buckets[rank] -= 1;
                suffix_array[buckets[rank]] = suffix - 1;
            }
        }
    };

    // Stored positions are offset by one so zero can represent an empty slot.
    let mut lms_indices = vec![0; len + 1];
    let mut lms_count = 0;
    for index in 1..len {
        if !is_s_type[index - 1] && is_s_type[index] {
            lms_indices[index] = lms_count + 1;
            lms_count += 1;
        }
    }
    let lms_suffixes: Vec<_> = (1..len)
        .filter(|&index| !is_s_type[index - 1] && is_s_type[index])
        .collect();
    induce(&mut suffix_array, &lms_suffixes);

    if lms_count > 0 {
        let mut sorted_lms = Vec::with_capacity(lms_count);
        for &suffix in &suffix_array {
            if suffix > 0 && lms_indices[suffix - 1] != 0 {
                sorted_lms.push(suffix - 1);
            }
        }

        let mut reduced_symbols = vec![0; lms_count];
        let mut reduced_alphabet_max = 0;
        reduced_symbols[lms_indices[sorted_lms[0]] - 1] = 0;
        for index in 1..lms_count {
            let mut left = sorted_lms[index - 1];
            let mut right = sorted_lms[index];
            let left_end = if lms_indices[left] < lms_count {
                lms_suffixes[lms_indices[left]]
            } else {
                len
            };
            let right_end = if lms_indices[right] < lms_count {
                lms_suffixes[lms_indices[right]]
            } else {
                len
            };
            let mut is_same = left_end - left == right_end - right;
            while is_same && left < left_end {
                if symbols[left] != symbols[right] {
                    is_same = false;
                    break;
                }
                left += 1;
                right += 1;
            }
            if is_same && (left == len || symbols[left] != symbols[right]) {
                is_same = false;
            }
            if !is_same {
                reduced_alphabet_max += 1;
            }
            reduced_symbols[lms_indices[sorted_lms[index]] - 1] = reduced_alphabet_max;
        }

        let reduced_suffix_array = sa_is(&reduced_symbols, reduced_alphabet_max);
        for index in 0..lms_count {
            sorted_lms[index] = lms_suffixes[reduced_suffix_array[index]];
        }
        induce(&mut suffix_array, &sorted_lms);
    }

    for suffix in &mut suffix_array {
        *suffix -= 1;
    }
    suffix_array
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected_suffix_array(text: &[u8]) -> Vec<usize> {
        let mut suffix_array: Vec<_> = (0..text.len()).collect();
        suffix_array.sort_unstable_by(|&left, &right| text[left..].cmp(&text[right..]));
        suffix_array
    }

    #[test]
    fn test_suffix_array_edge_cases() {
        for text in [
            &b""[..],
            &b"a"[..],
            &b"banana"[..],
            &b"mississippi"[..],
            &b"aaaaaaaaaa"[..],
            &[0, 255, 0, 255, 0, 0, 255],
        ] {
            assert_eq!(build_suffix_array(text), expected_suffix_array(text));
        }
    }

    #[test]
    fn test_suffix_array_varied_binary_inputs() {
        let mut state = 0x9E37_79B9_u32;
        for len in 0..256 {
            let text: Vec<_> = (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    state as u8
                })
                .collect();
            assert_eq!(
                build_suffix_array(&text),
                expected_suffix_array(&text),
                "suffix array mismatch for random input length {len}"
            );

            let repetitive_text: Vec<_> = text.iter().map(|byte| byte % 5).collect();
            assert_eq!(
                build_suffix_array(&repetitive_text),
                expected_suffix_array(&repetitive_text),
                "suffix array mismatch for repetitive input length {len}"
            );
        }
    }
}
