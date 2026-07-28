// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Routines for compressing and decompressing constant-encoded data

#[cfg(test)]
use crate::compression::{BlockCompressor, block::validate_fixed_payload_len};
use crate::{
    buffer::LanceBuffer,
    compression::{
        BlockDecompressor, BlockValueType, FixedPerValueDecompressor, require_no_block_payload,
    },
    data::{AllNullDataBlock, BlockInfo, ConstantDataBlock, DataBlock, FixedWidthDataBlock},
    encodings::physical::try_vec_with_capacity,
};

use lance_core::{Error, Result};

/// A decompressor for constant-encoded data
#[derive(Debug)]
pub struct ConstantDecompressor {
    scalar: Option<LanceBuffer>,
}

impl ConstantDecompressor {
    pub fn new(scalar: Option<LanceBuffer>) -> Self {
        Self { scalar }
    }
}

impl BlockDecompressor for ConstantDecompressor {
    fn decompress(&self, _data: Option<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        if let Some(scalar) = self.scalar.clone() {
            Ok(DataBlock::Constant(ConstantDataBlock {
                data: scalar,
                num_values,
            }))
        } else {
            Ok(DataBlock::AllNull(AllNullDataBlock { num_values }))
        }
    }
}

impl FixedPerValueDecompressor for ConstantDecompressor {
    fn decompress(&self, _data: FixedWidthDataBlock, num_values: u64) -> Result<DataBlock> {
        if let Some(scalar) = self.scalar.clone() {
            Ok(DataBlock::Constant(ConstantDataBlock {
                data: scalar,
                num_values,
            }))
        } else {
            Ok(DataBlock::AllNull(AllNullDataBlock { num_values }))
        }
    }

    fn bits_per_value(&self) -> u64 {
        self.scalar
            .as_ref()
            .map(|s| s.len() as u64 * 8)
            .unwrap_or(0)
    }
}

/// Metadata-only fixed-width constant (or typed empty) block compressor.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct ConstantBlockCompressor {
    value_type: BlockValueType,
    value: Option<u64>,
}

#[cfg(test)]
impl ConstantBlockCompressor {
    pub(crate) fn new(value_type: BlockValueType, value: Option<u64>) -> Self {
        Self { value_type, value }
    }
}

#[cfg(test)]
impl BlockCompressor for ConstantBlockCompressor {
    fn compress(&self, data: DataBlock) -> Result<Option<LanceBuffer>> {
        let DataBlock::FixedWidth(data) = data else {
            return Err(Error::invalid_input(
                "Constant block compression requires fixed-width data",
            ));
        };
        if data.bits_per_value != self.value_type.bits_per_value() {
            return Err(Error::invalid_input(format!(
                "Constant block compressor expects {}-bit values, got {}",
                self.value_type.bits_per_value(),
                data.bits_per_value
            )));
        }
        validate_fixed_payload_len(
            &data.data,
            self.value_type,
            data.num_values,
            "Constant block input",
        )?;
        match self.value {
            None => {
                if data.num_values != 0 {
                    return Err(Error::invalid_input(
                        "Typed empty block compressor received non-empty data",
                    ));
                }
            }
            Some(expected) => {
                if data.num_values == 0 {
                    return Err(Error::invalid_input(
                        "Constant block compressor received an empty sequence",
                    ));
                }
                macro_rules! check_values {
                    ($ty:ty) => {
                        for (index, actual) in
                            data.data.borrow_to_typed_slice::<$ty>().iter().enumerate()
                        {
                            if u64::from(*actual) != expected {
                                return Err(Error::invalid_input(format!(
                                    "Constant block expects {expected}, got {actual} at index {index}"
                                )));
                            }
                        }
                    };
                }
                match self.value_type {
                    BlockValueType::UInt8 => check_values!(u8),
                    BlockValueType::UInt16 => check_values!(u16),
                    BlockValueType::UInt32 => check_values!(u32),
                    BlockValueType::UInt64 => check_values!(u64),
                }
            }
        }
        Ok(None)
    }
}

/// Metadata-only fixed-width constant (or typed empty) block decompressor.
#[derive(Debug)]
pub(crate) struct ConstantBlockDecompressor {
    value_type: BlockValueType,
    value: Option<u64>,
}

impl ConstantBlockDecompressor {
    pub(crate) fn new(value_type: BlockValueType, value: Option<u64>) -> Self {
        Self { value_type, value }
    }
}

impl BlockDecompressor for ConstantBlockDecompressor {
    fn decompress(&self, data: Option<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        require_no_block_payload(data, "Constant block")?;
        let output = match self.value {
            None => {
                if num_values != 0 {
                    return Err(Error::invalid_input(format!(
                        "Typed empty block cannot represent {num_values} values"
                    )));
                }
                LanceBuffer::empty()
            }
            Some(value) => {
                if num_values == 0 {
                    return Err(Error::invalid_input(
                        "Non-empty Constant descriptor cannot represent an empty block",
                    ));
                }
                macro_rules! repeat {
                    ($ty:ty) => {{
                        let mut values =
                            try_vec_with_capacity::<$ty>(num_values, "Constant block output")?;
                        values.resize(num_values as usize, value as $ty);
                        LanceBuffer::reinterpret_vec(values)
                    }};
                }
                match self.value_type {
                    BlockValueType::UInt8 => repeat!(u8),
                    BlockValueType::UInt16 => repeat!(u16),
                    BlockValueType::UInt32 => repeat!(u32),
                    BlockValueType::UInt64 => repeat!(u64),
                }
            }
        };
        Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
            data: output,
            bits_per_value: self.value_type.bits_per_value(),
            num_values,
            block_info: BlockInfo::new(),
        }))
    }
}
