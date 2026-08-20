// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Primitive predicates used by encoded compute.

use arrow_buffer::BooleanBufferBuilder;
use lance_core::{Error, Result};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    buffer::LanceBuffer,
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Comparison operators supported by primitive encoded compute.
pub enum ComparisonOperator {
    /// Equal.
    Equal,
    /// Not equal.
    NotEqual,
    /// Less than.
    LessThan,
    /// Less than or equal.
    LessThanOrEqual,
    /// Greater than.
    GreaterThan,
    /// Greater than or equal.
    GreaterThanOrEqual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Literal values supported by primitive encoded compute.
pub enum PrimitiveLiteral {
    /// A signed 32-bit literal.
    Int32(i32),
    /// An unsigned 32-bit literal.
    UInt32(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// A one-column comparison evaluated by the current-format decoder.
pub struct PrimitivePredicate {
    /// Top-level column name.
    pub column: String,
    /// Comparison operator.
    pub operator: ComparisonOperator,
    /// Typed literal.
    pub literal: PrimitiveLiteral,
}

static DIRECT_VALUES: AtomicU64 = AtomicU64::new(0);
static FALLBACK_VALUES: AtomicU64 = AtomicU64::new(0);
static OUTPUT_BYTES: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// Process-wide counters for primitive encoded predicate evaluation.
pub struct PrimitivePredicateStats {
    /// Values evaluated by an encoding-specific kernel.
    pub direct_values: u64,
    /// Values evaluated after falling back to a decoded `DataBlock`.
    pub fallback_values: u64,
    /// Bytes emitted for Boolean predicate bitmaps.
    pub output_bytes: u64,
}

/// Returns a snapshot of process-wide primitive predicate counters.
pub fn primitive_predicate_stats() -> PrimitivePredicateStats {
    PrimitivePredicateStats {
        direct_values: DIRECT_VALUES.load(Ordering::Relaxed),
        fallback_values: FALLBACK_VALUES.load(Ordering::Relaxed),
        output_bytes: OUTPUT_BYTES.load(Ordering::Relaxed),
    }
}

pub(crate) fn record_direct_values(num_values: u64) {
    DIRECT_VALUES.fetch_add(num_values, Ordering::Relaxed);
    OUTPUT_BYTES.fetch_add(num_values.div_ceil(8), Ordering::Relaxed);
}

pub(crate) fn record_fallback_values(num_values: u64) {
    FALLBACK_VALUES.fetch_add(num_values, Ordering::Relaxed);
    OUTPUT_BYTES.fetch_add(num_values.div_ceil(8), Ordering::Relaxed);
}

impl PrimitivePredicate {
    /// Evaluates this predicate against a decoded fixed-width block.
    pub fn evaluate_block(&self, data: &DataBlock) -> Result<DataBlock> {
        let fixed = data.as_fixed_width_ref().ok_or_else(|| {
            Error::not_supported_source(
                format!(
                    "Encoded predicate expected fixed-width data, got {}",
                    data.name()
                )
                .into(),
            )
        })?;
        if fixed.bits_per_value != 32 {
            return Err(Error::not_supported_source(
                format!(
                    "Encoded predicate expected 32-bit data, got {} bits per value",
                    fixed.bits_per_value
                )
                .into(),
            ));
        }

        let values = fixed.data.borrow_to_typed_slice::<u32>();
        self.evaluate_u32_values(values.iter().copied(), fixed.num_values)
    }

    pub(crate) fn evaluate_u32_values(
        &self,
        values: impl IntoIterator<Item = u32>,
        num_values: u64,
    ) -> Result<DataBlock> {
        let mut matches = BooleanBufferBuilder::new(num_values as usize);
        for value in values {
            matches.append(self.matches_u32(value));
        }
        if matches.len() != num_values as usize {
            return Err(Error::internal(format!(
                "Encoded predicate received {} values, expected {num_values}",
                matches.len()
            )));
        }
        Ok(boolean_block(matches, num_values))
    }

    pub(crate) fn evaluate_u32_runs(
        &self,
        runs: impl IntoIterator<Item = (u32, usize)>,
        num_values: u64,
    ) -> Result<DataBlock> {
        let expected = num_values as usize;
        let mut matches = BooleanBufferBuilder::new(expected);
        if expected == 0 {
            return Ok(boolean_block(matches, num_values));
        }
        for (value, length) in runs {
            if length == 0 {
                return Err(Error::invalid_input_source(
                    "RLE predicate encountered a zero run length".into(),
                ));
            }
            let remaining = expected.saturating_sub(matches.len());
            matches.append_n(length.min(remaining), self.matches_u32(value));
            if matches.len() == expected {
                break;
            }
        }
        if matches.len() != expected {
            return Err(Error::invalid_input_source(
                format!(
                    "RLE predicate produced {} values, expected {num_values}",
                    matches.len()
                )
                .into(),
            ));
        }
        Ok(boolean_block(matches, num_values))
    }

    pub(crate) fn matches_u32(&self, value: u32) -> bool {
        match self.literal {
            PrimitiveLiteral::Int32(literal) => compare(value as i32, literal, self.operator),
            PrimitiveLiteral::UInt32(literal) => compare(value, literal, self.operator),
        }
    }
}

fn boolean_block(mut matches: BooleanBufferBuilder, num_values: u64) -> DataBlock {
    let matches = matches.finish();
    DataBlock::FixedWidth(FixedWidthDataBlock {
        data: LanceBuffer::from(matches.into_inner()),
        bits_per_value: 1,
        num_values,
        block_info: BlockInfo::new(),
    })
}

fn compare<T: PartialEq + PartialOrd>(value: T, literal: T, operator: ComparisonOperator) -> bool {
    match operator {
        ComparisonOperator::Equal => value == literal,
        ComparisonOperator::NotEqual => value != literal,
        ComparisonOperator::LessThan => value < literal,
        ComparisonOperator::LessThanOrEqual => value <= literal,
        ComparisonOperator::GreaterThan => value > literal,
        ComparisonOperator::GreaterThanOrEqual => value >= literal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_buffer::BooleanBuffer;

    fn fixed_i32(values: Vec<i32>) -> DataBlock {
        DataBlock::FixedWidth(FixedWidthDataBlock {
            num_values: values.len() as u64,
            data: LanceBuffer::reinterpret_vec(values),
            bits_per_value: 32,
            block_info: BlockInfo::new(),
        })
    }

    #[test]
    fn evaluates_signed_values() {
        let predicate = PrimitivePredicate {
            column: "value".to_string(),
            operator: ComparisonOperator::LessThan,
            literal: PrimitiveLiteral::Int32(0),
        };
        let actual = predicate
            .evaluate_block(&fixed_i32(vec![-2, 0, 3]))
            .unwrap();
        let actual = actual.as_fixed_width_ref().unwrap();
        let actual = BooleanBuffer::new(actual.data.clone().into_buffer(), 0, 3);
        assert_eq!(actual.iter().collect::<Vec<_>>(), vec![true, false, false]);
    }
}
