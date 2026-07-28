// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared types and decoder construction for unsigned block sequences.
//!
//! As with mini-block compression, selectors return concrete codecs. Codecs
//! own their child codecs, framing, and validation.

use lance_core::{Error, Result};

pub(crate) const MAX_DICTIONARY_ITEMS: usize = 4096;
#[cfg(feature = "bitpacking")]
pub(crate) const BITPACK_CHUNK_VALUES: u64 = 1024;

/// Typed unsigned role expected from a block descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockValueType {
    /// Unsigned 8-bit values.
    UInt8,
    /// Unsigned 16-bit values.
    UInt16,
    /// Unsigned 32-bit values.
    UInt32,
    /// Unsigned 64-bit values.
    UInt64,
}

impl BlockValueType {
    pub(crate) fn from_bits(bits_per_value: u64) -> Result<Self> {
        match bits_per_value {
            8 => Ok(Self::UInt8),
            16 => Ok(Self::UInt16),
            32 => Ok(Self::UInt32),
            64 => Ok(Self::UInt64),
            _ => Err(Error::invalid_input(format!(
                "Block sequence only supports 8, 16, 32, or 64-bit values, got {bits_per_value}"
            ))),
        }
    }

    /// Returns the fixed width of each value.
    pub fn bits_per_value(self) -> u64 {
        match self {
            Self::UInt8 => 8,
            Self::UInt16 => 16,
            Self::UInt32 => 32,
            Self::UInt64 => 64,
        }
    }

    pub(crate) fn bytes_per_value(self) -> usize {
        (self.bits_per_value() / 8) as usize
    }

    pub(crate) fn max_value(self) -> u64 {
        match self {
            Self::UInt8 => u8::MAX as u64,
            Self::UInt16 => u16::MAX as u64,
            Self::UInt32 => u32::MAX as u64,
            Self::UInt64 => u64::MAX,
        }
    }
}

mod factory;
pub(crate) mod fixed;

#[cfg(test)]
pub(crate) use factory::encode_scalar;
pub(crate) use factory::{
    create_block_decompressor, infer_block_value_type, validate_fixed_payload_len,
};
#[cfg(feature = "bitpacking")]
pub(crate) use factory::{validate_inline_bitpacking_payload, validate_out_of_line_payload};
#[cfg(test)]
pub(crate) use fixed::fixed_from_u64_values;
#[cfg(any(test, feature = "bitpacking"))]
pub(crate) use fixed::visit_unsigned_values;
pub(crate) use fixed::{fixed_block, read_unsigned_values};

#[cfg(test)]
mod tests;
