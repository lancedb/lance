// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Delta transform for non-decreasing unsigned block sequences.

use super::try_vec_with_capacity;
use crate::{
    buffer::LanceBuffer,
    compression::BlockDecompressor,
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
};
use lance_core::{Error, Result};

/// Converts a non-decreasing u32/u64 block into adjacent differences.
#[cfg(test)]
pub(crate) fn encode_deltas(
    data: FixedWidthDataBlock,
    expected_base: u64,
) -> Result<FixedWidthDataBlock> {
    if !matches!(data.bits_per_value, 32 | 64) {
        return Err(Error::invalid_input(format!(
            "Delta only supports 32 or 64-bit values, got {}",
            data.bits_per_value
        )));
    }
    if data.num_values < 2 {
        return Err(Error::invalid_input(format!(
            "Delta requires at least 2 values, got {}",
            data.num_values
        )));
    }

    match data.bits_per_value {
        32 => {
            let values = checked_values::<u32>(&data, "Delta")?;
            if u64::from(values[0]) != expected_base {
                return Err(Error::invalid_input(format!(
                    "Delta base mismatch: codec expects {expected_base}, input starts with {}",
                    values[0]
                )));
            }
            let mut deltas = Vec::with_capacity(values.len() - 1);
            for (index, pair) in values.windows(2).enumerate() {
                deltas.push(pair[1].checked_sub(pair[0]).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Delta input decreases at index {}: {} -> {}",
                        index + 1,
                        pair[0],
                        pair[1]
                    ))
                })?);
            }
            Ok(FixedWidthDataBlock {
                bits_per_value: 32,
                data: LanceBuffer::reinterpret_vec(deltas),
                num_values: data.num_values - 1,
                block_info: BlockInfo::default(),
            })
        }
        64 => {
            let values = checked_values::<u64>(&data, "Delta")?;
            if values[0] != expected_base {
                return Err(Error::invalid_input(format!(
                    "Delta base mismatch: codec expects {expected_base}, input starts with {}",
                    values[0]
                )));
            }
            let mut deltas = Vec::with_capacity(values.len() - 1);
            for (index, pair) in values.windows(2).enumerate() {
                deltas.push(pair[1].checked_sub(pair[0]).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Delta input decreases at index {}: {} -> {}",
                        index + 1,
                        pair[0],
                        pair[1]
                    ))
                })?);
            }
            Ok(FixedWidthDataBlock {
                bits_per_value: 64,
                data: LanceBuffer::reinterpret_vec(deltas),
                num_values: data.num_values - 1,
                block_info: BlockInfo::default(),
            })
        }
        _ => unreachable!("delta width was validated above"),
    }
}

/// Reconstructs a delta sequence after its child has been decoded.
#[derive(Debug)]
pub(crate) struct DeltaDecompressor {
    bits_per_value: u64,
    base: u64,
    child: Box<dyn BlockDecompressor>,
}

impl DeltaDecompressor {
    pub(crate) fn new(bits_per_value: u64, base: u64, child: Box<dyn BlockDecompressor>) -> Self {
        Self {
            bits_per_value,
            base,
            child,
        }
    }
}

impl BlockDecompressor for DeltaDecompressor {
    fn decompress(&self, data: Option<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        if num_values < 2 {
            return Err(Error::invalid_input(format!(
                "Delta requires at least 2 values, got {num_values}"
            )));
        }
        if self.bits_per_value == 32 && self.base > u32::MAX as u64 {
            return Err(Error::invalid_input(format!(
                "Delta base {} exceeds u32::MAX",
                self.base
            )));
        }
        if !matches!(self.bits_per_value, 32 | 64) {
            return Err(Error::invalid_input(format!(
                "Delta only supports 32 or 64-bit values, got {}",
                self.bits_per_value
            )));
        }

        let child = self.child.decompress(data, num_values - 1)?;
        reconstruct_deltas(child, self.bits_per_value, self.base, num_values)
    }
}

pub(crate) fn reconstruct_deltas(
    child: DataBlock,
    bits_per_value: u64,
    base: u64,
    num_values: u64,
) -> Result<DataBlock> {
    if num_values < 2 {
        return Err(Error::invalid_input(format!(
            "Delta requires at least 2 values, got {num_values}"
        )));
    }
    let DataBlock::FixedWidth(child) = child else {
        return Err(Error::invalid_input(
            "Delta child decoded to a non fixed-width block",
        ));
    };
    if child.bits_per_value != bits_per_value || child.num_values != num_values - 1 {
        return Err(Error::invalid_input(format!(
            "Delta child decoded {} {}-bit values, expected {} {}-bit values",
            child.num_values,
            child.bits_per_value,
            num_values - 1,
            bits_per_value
        )));
    }

    let data = match bits_per_value {
        32 => {
            let deltas = checked_values::<u32>(&child, "Delta child")?;
            let mut values = try_vec_with_capacity::<u32>(num_values, "Delta output")?;
            let mut current = u32::try_from(base)
                .map_err(|_| Error::invalid_input(format!("Delta base {base} exceeds u32::MAX")))?;
            values.push(current);
            for (index, delta) in deltas.iter().enumerate() {
                current = current.checked_add(*delta).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Delta prefix sum overflows u32 at index {}",
                        index + 1
                    ))
                })?;
                values.push(current);
            }
            LanceBuffer::reinterpret_vec(values)
        }
        64 => {
            let deltas = checked_values::<u64>(&child, "Delta child")?;
            let mut values = try_vec_with_capacity::<u64>(num_values, "Delta output")?;
            let mut current = base;
            values.push(current);
            for (index, delta) in deltas.iter().enumerate() {
                current = current.checked_add(*delta).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Delta prefix sum overflows u64 at index {}",
                        index + 1
                    ))
                })?;
                values.push(current);
            }
            LanceBuffer::reinterpret_vec(values)
        }
        _ => {
            return Err(Error::invalid_input(format!(
                "Delta only supports 32 or 64-bit values, got {bits_per_value}"
            )));
        }
    };

    Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
        bits_per_value,
        data,
        num_values,
        block_info: BlockInfo::default(),
    }))
}

fn checked_values<T: arrow_buffer::ArrowNativeType>(
    data: &FixedWidthDataBlock,
    label: &str,
) -> Result<arrow_buffer::ScalarBuffer<T>> {
    let expected = usize::try_from(data.num_values)
        .ok()
        .and_then(|len| len.checked_mul(std::mem::size_of::<T>()))
        .ok_or_else(|| Error::invalid_input(format!("{label} byte length overflows usize")))?;
    if data.data.len() != expected {
        return Err(Error::invalid_input(format!(
            "{label} has {} bytes, expected {expected}",
            data.data.len()
        )));
    }
    Ok(data.data.borrow_to_typed_slice::<T>())
}

#[cfg(test)]
mod tests {
    use crate::compression::BlockCompressor;
    use crate::encodings::physical::value::{ValueDecompressor, ValueEncoder};
    use crate::format::pb21::Flat;

    use super::*;

    #[test]
    fn delta_round_trip_with_zero_delta() {
        let input = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 64,
            data: LanceBuffer::reinterpret_vec(vec![5_u64, 5, 9, 12]),
            num_values: 4,
            block_info: BlockInfo::default(),
        });
        let DataBlock::FixedWidth(input) = input else {
            unreachable!()
        };
        let deltas = encode_deltas(input, 5).unwrap();
        let payload = ValueEncoder::default()
            .compress(DataBlock::FixedWidth(deltas))
            .unwrap();
        let decoded = DeltaDecompressor::new(
            64,
            5,
            Box::new(ValueDecompressor::from_flat(&Flat {
                bits_per_value: 64,
                data: None,
            })),
        )
        .decompress(payload, 4)
        .unwrap()
        .as_fixed_width()
        .unwrap();
        assert_eq!(
            decoded.data.borrow_to_typed_slice::<u64>().as_ref(),
            &[5, 5, 9, 12]
        );
    }

    #[test]
    fn delta_rejects_decreasing_input() {
        let input = FixedWidthDataBlock {
            bits_per_value: 32,
            data: LanceBuffer::reinterpret_vec(vec![2_u32, 1]),
            num_values: 2,
            block_info: BlockInfo::default(),
        };
        let error = encode_deltas(input, 2).unwrap_err();
        assert!(error.to_string().contains("decreases"));
    }
}
