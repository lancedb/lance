// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::{
    buffer::LanceBuffer,
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
};

pub fn fixed_block(value_type: BlockValueType, num_values: u64, data: LanceBuffer) -> DataBlock {
    DataBlock::FixedWidth(FixedWidthDataBlock {
        bits_per_value: value_type.bits_per_value(),
        data,
        num_values,
        block_info: BlockInfo::default(),
    })
}

pub fn fixed_from_u64_values(
    values: &[u64],
    value_type: BlockValueType,
    label: &str,
) -> Result<FixedWidthDataBlock> {
    let data = match value_type {
        BlockValueType::UInt8 => LanceBuffer::reinterpret_vec(
            values
                .iter()
                .map(|value| {
                    u8::try_from(*value).map_err(|_| {
                        Error::invalid_input(format!("{label} value {value} exceeds u8::MAX"))
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        ),
        BlockValueType::UInt16 => LanceBuffer::reinterpret_vec(
            values
                .iter()
                .map(|value| {
                    u16::try_from(*value).map_err(|_| {
                        Error::invalid_input(format!("{label} value {value} exceeds u16::MAX"))
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        ),
        BlockValueType::UInt32 => LanceBuffer::reinterpret_vec(
            values
                .iter()
                .map(|value| {
                    u32::try_from(*value).map_err(|_| {
                        Error::invalid_input(format!("{label} value {value} exceeds u32::MAX"))
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        ),
        BlockValueType::UInt64 => LanceBuffer::reinterpret_vec(values.to_vec()),
    };
    Ok(FixedWidthDataBlock {
        bits_per_value: value_type.bits_per_value(),
        data,
        num_values: values.len() as u64,
        block_info: BlockInfo::default(),
    })
}

pub fn visit_unsigned_values(
    block: &FixedWidthDataBlock,
    value_type: BlockValueType,
    mut visit: impl FnMut(u64) -> Result<()>,
) -> Result<()> {
    validate_fixed_payload_len(&block.data, value_type, block.num_values, "Block input")?;
    match value_type {
        BlockValueType::UInt8 => {
            for value in block.data.iter().copied() {
                visit(u64::from(value))?;
            }
        }
        BlockValueType::UInt16 => {
            for value in block.data.borrow_to_typed_view::<u16>().iter().copied() {
                visit(u64::from(value))?;
            }
        }
        BlockValueType::UInt32 => {
            for value in block.data.borrow_to_typed_view::<u32>().iter().copied() {
                visit(u64::from(value))?;
            }
        }
        BlockValueType::UInt64 => {
            for value in block.data.borrow_to_typed_view::<u64>().iter().copied() {
                visit(value)?;
            }
        }
    }
    Ok(())
}

pub fn read_unsigned_values(
    block: &FixedWidthDataBlock,
    value_type: BlockValueType,
) -> Result<Vec<u64>> {
    validate_fixed_payload_len(
        &block.data,
        value_type,
        block.num_values,
        "Fixed-width block",
    )?;
    Ok(match value_type {
        BlockValueType::UInt8 => block.data.iter().map(|value| u64::from(*value)).collect(),
        BlockValueType::UInt16 => block
            .data
            .borrow_to_typed_slice::<u16>()
            .iter()
            .map(|value| u64::from(*value))
            .collect(),
        BlockValueType::UInt32 => block
            .data
            .borrow_to_typed_slice::<u32>()
            .iter()
            .map(|value| u64::from(*value))
            .collect(),
        BlockValueType::UInt64 => block.data.borrow_to_typed_slice::<u64>().to_vec(),
    })
}
