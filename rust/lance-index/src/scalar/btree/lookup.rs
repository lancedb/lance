// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Derived UInt64 lookup representations. Persistent BTree pages stay unchanged.

use arrow_array::{Array, RecordBatch, UInt32Array, UInt64Array, cast::AsArray, types::UInt64Type};
use lance_core::utils::address::RowAddress;

#[derive(Clone, Copy, Debug)]
pub(super) struct RegularU64Pages {
    first: u64,
    last: u64,
    width: u64,
    start: usize,
}

impl RegularU64Pages {
    pub(super) fn try_new(batch: &RecordBatch, start: usize) -> Option<Self> {
        let mins = batch.column(0).as_primitive_opt::<UInt64Type>()?;
        let maxs = batch.column(1).as_primitive_opt::<UInt64Type>()?;
        if start >= batch.num_rows() || mins.is_null(start) || maxs.is_null(start) {
            return None;
        }
        let first = mins.value(start);
        let width = maxs.value(start).checked_sub(first)?.checked_add(1)?;
        for row in start..batch.num_rows() {
            if mins.is_null(row) || maxs.is_null(row) {
                return None;
            }
            let expected_min = first.checked_add(width.checked_mul((row - start) as u64)?)?;
            let expected_max = expected_min.checked_add(width - 1)?;
            let max = maxs.value(row);
            if mins.value(row) != expected_min
                || max < expected_min
                || max > expected_max
                || (row + 1 != batch.num_rows() && max != expected_max)
            {
                return None;
            }
        }
        Some(Self {
            first,
            last: maxs.value(batch.num_rows() - 1),
            width,
            start,
        })
    }

    pub(super) fn candidate_pages(&self, keys: &UInt64Array, page_ids: &[u32]) -> Vec<u32> {
        let mut pages = Vec::with_capacity(keys.len());
        for key in keys.iter().flatten() {
            if key >= self.first && key <= self.last {
                let position = self.start + ((key - self.first) / self.width) as usize;
                pages.push(page_ids[position]);
            }
        }
        pages.sort_unstable();
        pages.dedup();
        pages
    }
}

#[derive(Debug)]
enum U64Positions {
    Range {
        first_address: u64,
    },
    Dense(UInt32Array),
    Sparse {
        words: Box<[u64]>,
        ranks: Box<[u32]>,
        positions: UInt32Array,
    },
}

#[derive(Debug)]
pub(super) struct U64Lookup {
    first: u64,
    last: u64,
    positions: U64Positions,
}

impl U64Lookup {
    /// Charge the maximum bitmap/rank allocation before lazy initialization.
    /// The position array shares the existing value-order buffer.
    pub(super) fn extra_size(rows: usize) -> usize {
        rows.saturating_mul(2).div_ceil(64)
            * (std::mem::size_of::<u64>() + std::mem::size_of::<u32>())
    }

    pub(super) fn try_new(
        values: &UInt64Array,
        addresses: &UInt64Array,
        positions: UInt32Array,
    ) -> Option<Self> {
        let order = positions.values();
        let first = values.value(*order.first()? as usize);
        let last = values.value(*order.last()? as usize);
        // A duplicate key must retain every candidate, including stale addresses.
        // Those pages continue through the generic multi-match lookup.
        if !order
            .windows(2)
            .all(|pair| values.value(pair[0] as usize) < values.value(pair[1] as usize))
        {
            return None;
        }
        let span = last.checked_sub(first)?.checked_add(1)?;
        let positions = if span == order.len() as u64 {
            let first_address = addresses.value(order[0] as usize);
            let base = RowAddress::from(first_address);
            let consecutive = base
                .row_offset()
                .checked_add((order.len() - 1) as u32)
                .is_some()
                && order.iter().enumerate().all(|(offset, &position)| {
                    first_address.checked_add(offset as u64)
                        == Some(addresses.value(position as usize))
                });
            if consecutive {
                U64Positions::Range { first_address }
            } else {
                U64Positions::Dense(positions)
            }
        } else if span <= (order.len() as u64).saturating_mul(2) {
            // Bound memory by row count, never by an arbitrary UInt64 key span.
            let mut words = vec![0_u64; (span as usize).div_ceil(64)].into_boxed_slice();
            for &position in order {
                let offset = (values.value(position as usize) - first) as usize;
                words[offset / 64] |= 1_u64 << (offset % 64);
            }
            let mut rank = 0;
            let ranks = words
                .iter()
                .map(|word| {
                    let before = rank;
                    rank += word.count_ones();
                    before
                })
                .collect();
            U64Positions::Sparse {
                words,
                ranks,
                positions,
            }
        } else {
            return None;
        };
        Some(Self {
            first,
            last,
            positions,
        })
    }

    pub(super) fn first(&self) -> u64 {
        self.first
    }

    pub(super) fn last(&self) -> u64 {
        self.last
    }

    pub(super) fn get(&self, key: u64, addresses: &UInt64Array) -> Option<u64> {
        if key < self.first || key > self.last {
            return None;
        }
        let offset = key - self.first;
        let position = match &self.positions {
            U64Positions::Range { first_address } => return first_address.checked_add(offset),
            U64Positions::Dense(positions) => positions.value(offset as usize),
            U64Positions::Sparse {
                words,
                ranks,
                positions,
            } => {
                let word_index = (offset / 64) as usize;
                let bit = 1_u64 << (offset % 64);
                let word = words[word_index];
                if word & bit == 0 {
                    return None;
                }
                let rank = ranks[word_index] + (word & (bit - 1)).count_ones();
                positions.value(rank as usize)
            }
        };
        Some(addresses.value(position as usize))
    }
}
