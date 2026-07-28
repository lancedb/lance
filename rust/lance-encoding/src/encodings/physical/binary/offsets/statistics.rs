// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use crate::compression::{BlockValueType, block::MAX_DICTIONARY_ITEMS};

#[derive(Debug, Clone)]
pub struct SequenceStats {
    pub value_type: BlockValueType,
    pub len: u64,
    pub first: Option<u64>,
    pub min: u64,
    pub max: u64,
    pub is_non_decreasing: bool,
    pub arithmetic_step: Option<u64>,
    pub run_count: u64,
    pub sample: Option<Arc<[u8]>>,
}

impl SequenceStats {
    pub fn raw_bytes(&self) -> u64 {
        self.len
            .saturating_mul(self.value_type.bytes_per_value() as u64)
    }

    #[cfg(feature = "bitpacking")]
    pub fn required_bits(&self) -> u64 {
        if self.max == 0 {
            1
        } else {
            u64::from(u64::BITS - self.max.leading_zeros())
        }
    }

    pub fn sample_bytes(&self) -> &[u8] {
        self.sample.as_deref().unwrap_or_default()
    }
}

pub struct SequenceStatsBuilder {
    value_type: BlockValueType,
    len: u64,
    first: Option<u64>,
    previous: Option<u64>,
    min: u64,
    max: u64,
    is_non_decreasing: bool,
    step: Option<u64>,
    is_arithmetic: bool,
    run_count: u64,
    sample: Vec<u8>,
    sample_limit: usize,
}

impl SequenceStatsBuilder {
    pub fn without_sample(value_type: BlockValueType) -> Self {
        Self::with_sample_limit(value_type, 0)
    }

    pub fn with_sample_limit(value_type: BlockValueType, sample_limit: usize) -> Self {
        Self {
            value_type,
            len: 0,
            first: None,
            previous: None,
            min: u64::MAX,
            max: 0,
            is_non_decreasing: true,
            step: None,
            is_arithmetic: true,
            run_count: 0,
            sample: Vec::new(),
            sample_limit,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, value: u64) {
        if self.first.is_none() {
            self.first = Some(value);
            self.run_count = 1;
        }
        if let Some(previous) = self.previous {
            self.is_non_decreasing &= value >= previous;
            if value != previous {
                self.run_count += 1;
            }
            let difference = value.checked_sub(previous);
            match (self.step, difference) {
                (None, Some(difference)) if self.len == 1 => self.step = Some(difference),
                (Some(step), Some(difference)) if step == difference => {}
                _ => self.is_arithmetic = false,
            }
        }
        self.previous = Some(value);
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.len += 1;

        if self.sample.len() < self.sample_limit {
            let bytes = value.to_le_bytes();
            let width = self.value_type.bytes_per_value();
            let remaining = self.sample_limit - self.sample.len();
            self.sample
                .extend_from_slice(&bytes[..width.min(remaining)]);
        }
    }

    pub fn finish(self) -> SequenceStats {
        let arithmetic_step = if self.len >= 2 && self.is_arithmetic {
            self.step
        } else {
            None
        };
        SequenceStats {
            value_type: self.value_type,
            len: self.len,
            first: self.first,
            min: if self.len == 0 { 0 } else { self.min },
            max: self.max,
            is_non_decreasing: self.is_non_decreasing,
            arithmetic_step,
            run_count: self.run_count,
            sample: (!self.sample.is_empty()).then(|| Arc::from(self.sample)),
        }
    }
}

pub struct BoundedDistinctStatsBuilder {
    items: Option<Vec<u64>>,
}

impl BoundedDistinctStatsBuilder {
    pub fn new() -> Self {
        Self {
            items: Some(Vec::new()),
        }
    }

    #[inline(always)]
    pub fn push(&mut self, value: u64) {
        let Some(items) = self.items.as_mut() else {
            return;
        };
        match items.binary_search(&value) {
            Ok(_) => {}
            Err(_) if items.len() == MAX_DICTIONARY_ITEMS => self.items = None,
            Err(position) => {
                items.insert(position, value);
            }
        }
    }

    pub fn finish(self) -> Option<Arc<[u64]>> {
        self.items.map(Arc::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_collection_stops_at_the_dictionary_bound() {
        let mut bounded = BoundedDistinctStatsBuilder::new();
        for value in 0..MAX_DICTIONARY_ITEMS as u64 {
            bounded.push(value);
        }
        assert_eq!(bounded.finish().unwrap().len(), MAX_DICTIONARY_ITEMS);

        let mut oversized = BoundedDistinctStatsBuilder::new();
        for value in 0..=MAX_DICTIONARY_ITEMS as u64 {
            oversized.push(value);
        }
        assert!(oversized.finish().is_none());
    }
}
