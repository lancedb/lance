// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Packed encoding
//!
//! These encodings take struct data and compress it in a way that all fields are collected
//! together.
//!
//! This encoding can be transparent or opaque.  In order to be transparent we must use transparent
//! compression on all children.  Then we can zip together the compressed children.

use std::{convert::TryInto, sync::Arc};

use arrow_array::types::UInt64Type;

use lance_core::{datatypes::Field, Error, Result};
use snafu::location;

use crate::{
    buffer::LanceBuffer,
    compression::{
        DefaultCompressionStrategy, FixedPerValueDecompressor, MiniBlockDecompressor,
        VariablePerValueDecompressor,
    },
    data::{
        BlockInfo, DataBlock, DataBlockBuilder, FixedWidthDataBlock, StructDataBlock,
        VariableWidthBlock,
    },
    encodings::logical::primitive::{
        fullzip::{PerValueCompressor, PerValueDataBlock},
        miniblock::{MiniBlockCompressed, MiniBlockCompressor},
    },
    format::{
        pb21::{compressive_encoding::Compression, CompressiveEncoding, PackedStruct},
        ProtobufUtils21,
    },
    statistics::{GetStat, Stat},
};

use super::value::{ValueDecompressor, ValueEncoder};

// Transforms a `StructDataBlock` into a row major `FixedWidthDataBlock`.
// Only fields with fixed-width fields are supported for now, and the
// assumption that all fields has `bits_per_value % 8 == 0` is made.
fn struct_data_block_to_fixed_width_data_block(
    struct_data_block: StructDataBlock,
    bits_per_values: &[u64],
) -> DataBlock {
    let data_size = struct_data_block.expect_single_stat::<UInt64Type>(Stat::DataSize);
    let mut output = Vec::with_capacity(data_size as usize);
    let num_values = struct_data_block.children[0].num_values();

    for i in 0..num_values as usize {
        for (j, child) in struct_data_block.children.iter().enumerate() {
            let bytes_per_value = (bits_per_values[j] / 8) as usize;
            let this_data = child
                .as_fixed_width_ref()
                .unwrap()
                .data
                .slice_with_length(bytes_per_value * i, bytes_per_value);
            output.extend_from_slice(&this_data);
        }
    }

    DataBlock::FixedWidth(FixedWidthDataBlock {
        bits_per_value: bits_per_values.iter().copied().sum(),
        data: LanceBuffer::from(output),
        num_values,
        block_info: BlockInfo::default(),
    })
}

#[derive(Debug, Default)]
pub struct PackedStructFixedWidthMiniBlockEncoder {}

impl MiniBlockCompressor for PackedStructFixedWidthMiniBlockEncoder {
    fn compress(&self, data: DataBlock) -> Result<(MiniBlockCompressed, CompressiveEncoding)> {
        match data {
            DataBlock::Struct(struct_data_block) => {
                let bits_per_values = struct_data_block.children.iter().map(|data_block| data_block.as_fixed_width_ref().unwrap().bits_per_value).collect::<Vec<_>>();

                // transform struct datablock to fixed-width data block.
                let data_block = struct_data_block_to_fixed_width_data_block(struct_data_block, &bits_per_values);

                // store and transformed fixed-width data block.
                let value_miniblock_compressor = Box::new(ValueEncoder::default()) as Box<dyn MiniBlockCompressor>;
                let (value_miniblock_compressed, value_array_encoding) =
                value_miniblock_compressor.compress(data_block)?;

                Ok((
                    value_miniblock_compressed,
                    ProtobufUtils21::packed_struct(value_array_encoding, bits_per_values),
                ))
            }
            _ => Err(Error::InvalidInput {
                source: format!(
                    "Cannot compress a data block of type {} with PackedStructFixedWidthBlockEncoder",
                    data.name()
                )
                .into(),
                location: location!(),
            }),
        }
    }
}

#[derive(Debug)]
pub struct PackedStructFixedWidthMiniBlockDecompressor {
    bits_per_values: Vec<u64>,
    array_encoding: Box<dyn MiniBlockDecompressor>,
}

impl PackedStructFixedWidthMiniBlockDecompressor {
    pub fn new(description: &PackedStruct) -> Self {
        let array_encoding: Box<dyn MiniBlockDecompressor> = match description.values.as_ref().unwrap().compression.as_ref().unwrap() {
            Compression::Flat(flat) => Box::new(ValueDecompressor::from_flat(flat)),
            _ => panic!("Currently only `ArrayEncoding::Flat` is supported in packed struct encoding in Lance 2.1."),
        };
        Self {
            bits_per_values: description.bits_per_value.clone(),
            array_encoding,
        }
    }
}

impl MiniBlockDecompressor for PackedStructFixedWidthMiniBlockDecompressor {
    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        assert_eq!(data.len(), 1);
        let encoded_data_block = self.array_encoding.decompress(data, num_values)?;
        let DataBlock::FixedWidth(encoded_data_block) = encoded_data_block else {
            panic!("ValueDecompressor should output FixedWidth DataBlock")
        };

        let bytes_per_values = self
            .bits_per_values
            .iter()
            .map(|bits_per_value| *bits_per_value as usize / 8)
            .collect::<Vec<_>>();

        assert!(encoded_data_block.bits_per_value % 8 == 0);
        let encoded_bytes_per_row = (encoded_data_block.bits_per_value / 8) as usize;

        // use a prefix_sum vector as a helper to reconstruct to `StructDataBlock`.
        let mut prefix_sum = vec![0; self.bits_per_values.len()];
        for i in 0..(self.bits_per_values.len() - 1) {
            prefix_sum[i + 1] = prefix_sum[i] + bytes_per_values[i];
        }

        let mut children_data_block = vec![];
        for i in 0..self.bits_per_values.len() {
            let child_buf_size = bytes_per_values[i] * num_values as usize;
            let mut child_buf: Vec<u8> = Vec::with_capacity(child_buf_size);

            for j in 0..num_values as usize {
                // the start of the data at this row is `j * encoded_bytes_per_row`, and the offset for this field is `prefix_sum[i]`, this field has length `bytes_per_values[i]`.
                let this_value = encoded_data_block.data.slice_with_length(
                    prefix_sum[i] + (j * encoded_bytes_per_row),
                    bytes_per_values[i],
                );

                child_buf.extend_from_slice(&this_value);
            }

            let child = DataBlock::FixedWidth(FixedWidthDataBlock {
                data: LanceBuffer::from(child_buf),
                bits_per_value: self.bits_per_values[i],
                num_values,
                block_info: BlockInfo::default(),
            });
            children_data_block.push(child);
        }
        Ok(DataBlock::Struct(StructDataBlock {
            children: children_data_block,
            block_info: BlockInfo::default(),
            validity: None,
        }))
    }
}

#[derive(Debug)]
enum VariablePackedFieldData {
    Fixed {
        block: FixedWidthDataBlock,
    },
    Variable {
        block: VariableWidthBlock,
        bits_per_length: u64,
    },
}

impl VariablePackedFieldData {
    fn append_row_bytes(&self, row_idx: usize, output: &mut Vec<u8>) -> Result<()> {
        match self {
            Self::Fixed { block } => {
                let bits_per_value = block.bits_per_value;
                if bits_per_value % 8 != 0 {
                    return Err(Error::invalid_input(
                        "Packed struct variable encoding requires byte-aligned fixed-width children",
                        location!(),
                    ));
                }
                let bytes_per_value = (bits_per_value / 8) as usize;
                let start = row_idx.checked_mul(bytes_per_value).ok_or_else(|| {
                    Error::invalid_input("Packed struct row size overflow", location!())
                })?;
                let end = start + bytes_per_value;
                let data = block.data.as_ref();
                if end > data.len() {
                    return Err(Error::invalid_input(
                        "Packed struct fixed child out of bounds",
                        location!(),
                    ));
                }
                output.extend_from_slice(&data[start..end]);
                Ok(())
            }
            Self::Variable {
                block,
                bits_per_length,
            } => {
                if bits_per_length % 8 != 0 {
                    return Err(Error::invalid_input(
                        "Packed struct variable children must have byte-aligned length prefixes",
                        location!(),
                    ));
                }
                let prefix_bytes = (*bits_per_length / 8) as usize;
                if !(prefix_bytes == 4 || prefix_bytes == 8) {
                    return Err(Error::invalid_input(
                        "Packed struct variable children must use 32 or 64-bit length prefixes",
                        location!(),
                    ));
                }
                match block.bits_per_offset {
                    32 => {
                        let offsets = block.offsets.borrow_to_typed_slice::<u32>();
                        let start = offsets[row_idx] as usize;
                        let end = offsets[row_idx + 1] as usize;
                        if end > block.data.len() {
                            return Err(Error::invalid_input(
                                "Packed struct variable child offsets out of bounds",
                                location!(),
                            ));
                        }
                        let len = (end - start) as u32;
                        if prefix_bytes != std::mem::size_of::<u32>() {
                            return Err(Error::invalid_input(
                                "Packed struct variable child length prefix mismatch",
                                location!(),
                            ));
                        }
                        output.extend_from_slice(&len.to_le_bytes());
                        output.extend_from_slice(&block.data[start..end]);
                        Ok(())
                    }
                    64 => {
                        let offsets = block.offsets.borrow_to_typed_slice::<u64>();
                        let start = offsets[row_idx] as usize;
                        let end = offsets[row_idx + 1] as usize;
                        if end > block.data.len() {
                            return Err(Error::invalid_input(
                                "Packed struct variable child offsets out of bounds",
                                location!(),
                            ));
                        }
                        let len = (end - start) as u64;
                        if prefix_bytes != std::mem::size_of::<u64>() {
                            return Err(Error::invalid_input(
                                "Packed struct variable child length prefix mismatch",
                                location!(),
                            ));
                        }
                        output.extend_from_slice(&len.to_le_bytes());
                        output.extend_from_slice(&block.data[start..end]);
                        Ok(())
                    }
                    _ => Err(Error::invalid_input(
                        "Packed struct variable child must use 32 or 64-bit offsets",
                        location!(),
                    )),
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct PackedStructVariablePerValueEncoder {
    strategy: DefaultCompressionStrategy,
    fields: Vec<Field>,
}

impl PackedStructVariablePerValueEncoder {
    pub fn new(strategy: DefaultCompressionStrategy, fields: Vec<Field>) -> Self {
        Self { strategy, fields }
    }
}

impl PerValueCompressor for PackedStructVariablePerValueEncoder {
    fn compress(&self, data: DataBlock) -> Result<(PerValueDataBlock, CompressiveEncoding)> {
        let DataBlock::Struct(struct_block) = data else {
            return Err(Error::invalid_input(
                "Packed struct encoder requires Struct data block",
                location!(),
            ));
        };

        if struct_block.children.is_empty() {
            return Err(Error::invalid_input(
                "Packed struct encoder requires at least one child field",
                location!(),
            ));
        }
        if struct_block.children.len() != self.fields.len() {
            return Err(Error::invalid_input(
                "Struct field metadata does not match number of children",
                location!(),
            ));
        }

        let num_values = struct_block.children[0].num_values();
        for child in struct_block.children.iter() {
            if child.num_values() != num_values {
                return Err(Error::invalid_input(
                    "Packed struct children must have matching value counts",
                    location!(),
                ));
            }
        }

        let mut field_data = Vec::with_capacity(self.fields.len());
        let mut field_metadata = Vec::with_capacity(self.fields.len());

        for (field, child_block) in self.fields.iter().zip(struct_block.children.into_iter()) {
            let compressor = crate::compression::CompressionStrategy::create_per_value(
                &self.strategy,
                field,
                &child_block,
            )?;
            let (compressed, encoding) = compressor.compress(child_block)?;
            match compressed {
                PerValueDataBlock::Fixed(block) => {
                    field_metadata.push(ProtobufUtils21::packed_struct_field_fixed(
                        encoding,
                        block.bits_per_value,
                    ));
                    field_data.push(VariablePackedFieldData::Fixed { block });
                }
                PerValueDataBlock::Variable(block) => {
                    let bits_per_length = block.bits_per_offset as u64;
                    field_metadata.push(ProtobufUtils21::packed_struct_field_variable(
                        encoding,
                        bits_per_length,
                    ));
                    field_data.push(VariablePackedFieldData::Variable {
                        block,
                        bits_per_length,
                    });
                }
            }
        }

        let mut row_buffers: Vec<Vec<u8>> = Vec::with_capacity(num_values as usize);
        let mut total_bytes: usize = 0;
        let mut max_row_len: usize = 0;
        for row in 0..num_values as usize {
            let mut row_bytes = Vec::new();
            for field in &field_data {
                field.append_row_bytes(row, &mut row_bytes)?;
            }
            max_row_len = max_row_len.max(row_bytes.len());
            total_bytes = total_bytes.checked_add(row_bytes.len()).ok_or_else(|| {
                Error::invalid_input("Packed struct row data size overflow", location!())
            })?;
            row_buffers.push(row_bytes);
        }

        let use_u32_offsets = total_bytes <= u32::MAX as usize && max_row_len <= u32::MAX as usize;
        let bits_per_offset = if use_u32_offsets { 32 } else { 64 };
        let mut row_builder = DataBlockBuilder::with_capacity_estimate(total_bytes as u64);

        for row_bytes in row_buffers {
            let row_len = row_bytes.len();
            let data_buffer = LanceBuffer::from(row_bytes);
            if use_u32_offsets {
                let row_block = DataBlock::VariableWidth(VariableWidthBlock {
                    data: data_buffer,
                    bits_per_offset,
                    offsets: LanceBuffer::reinterpret_vec(vec![0_u32, row_len as u32]),
                    num_values: 1,
                    block_info: BlockInfo::new(),
                });
                row_builder.append(&row_block, 0..1);
            } else {
                let row_block = DataBlock::VariableWidth(VariableWidthBlock {
                    data: data_buffer,
                    bits_per_offset,
                    offsets: LanceBuffer::reinterpret_vec(vec![0_u64, row_len as u64]),
                    num_values: 1,
                    block_info: BlockInfo::new(),
                });
                row_builder.append(&row_block, 0..1);
            }
        }

        let DataBlock::VariableWidth(data_block) = row_builder.finish() else {
            panic!("Packed struct variable encoder expected variable-width block")
        };

        Ok((
            PerValueDataBlock::Variable(data_block),
            ProtobufUtils21::packed_struct_variable(field_metadata),
        ))
    }
}

#[derive(Debug)]
pub(crate) enum VariablePackedStructFieldKind {
    Fixed {
        bits_per_value: u64,
        decompressor: Arc<dyn FixedPerValueDecompressor>,
    },
    Variable {
        bits_per_length: u64,
        decompressor: Arc<dyn VariablePerValueDecompressor>,
    },
}

#[derive(Debug)]
pub(crate) struct VariablePackedStructFieldDecoder {
    pub(crate) kind: VariablePackedStructFieldKind,
}

#[derive(Debug)]
pub struct PackedStructVariablePerValueDecompressor {
    fields: Vec<VariablePackedStructFieldDecoder>,
}

impl PackedStructVariablePerValueDecompressor {
    pub(crate) fn new(fields: Vec<VariablePackedStructFieldDecoder>) -> Self {
        Self { fields }
    }
}

enum FieldAccumulator {
    Fixed {
        builder: DataBlockBuilder,
        bits_per_value: u64,
    },
    Variable32 {
        builder: DataBlockBuilder,
    },
    Variable64 {
        builder: DataBlockBuilder,
    },
}

impl VariablePerValueDecompressor for PackedStructVariablePerValueDecompressor {
    fn decompress(&self, data: VariableWidthBlock) -> Result<DataBlock> {
        let num_values = data.num_values;
        let offsets_u64 = match data.bits_per_offset {
            32 => data
                .offsets
                .borrow_to_typed_slice::<u32>()
                .iter()
                .map(|v| *v as u64)
                .collect::<Vec<_>>(),
            64 => data
                .offsets
                .borrow_to_typed_slice::<u64>()
                .as_ref()
                .to_vec(),
            _ => {
                return Err(Error::invalid_input(
                    "Packed struct row offsets must be 32 or 64 bits",
                    location!(),
                ))
            }
        };

        if offsets_u64.len() != num_values as usize + 1 {
            return Err(Error::invalid_input(
                "Packed struct row offsets length mismatch",
                location!(),
            ));
        }

        let mut accumulators = Vec::with_capacity(self.fields.len());
        for field in &self.fields {
            match &field.kind {
                VariablePackedStructFieldKind::Fixed { bits_per_value, .. } => {
                    if bits_per_value % 8 != 0 {
                        return Err(Error::invalid_input(
                            "Packed struct fixed child must be byte-aligned",
                            location!(),
                        ));
                    }
                    let bytes_per_value = bits_per_value.checked_div(8).ok_or_else(|| {
                        Error::invalid_input(
                            "Invalid bits per value for packed struct field",
                            location!(),
                        )
                    })?;
                    let estimate = bytes_per_value.checked_mul(num_values).ok_or_else(|| {
                        Error::invalid_input(
                            "Packed struct fixed child allocation overflow",
                            location!(),
                        )
                    })?;
                    accumulators.push(FieldAccumulator::Fixed {
                        builder: DataBlockBuilder::with_capacity_estimate(estimate),
                        bits_per_value: *bits_per_value,
                    });
                }
                VariablePackedStructFieldKind::Variable {
                    bits_per_length, ..
                } => match bits_per_length {
                    32 => accumulators.push(FieldAccumulator::Variable32 {
                        builder: DataBlockBuilder::with_capacity_estimate(data.data.len() as u64),
                    }),
                    64 => accumulators.push(FieldAccumulator::Variable64 {
                        builder: DataBlockBuilder::with_capacity_estimate(data.data.len() as u64),
                    }),
                    _ => {
                        return Err(Error::invalid_input(
                            "Packed struct variable child must use 32 or 64-bit length prefixes",
                            location!(),
                        ))
                    }
                },
            }
        }

        for row_idx in 0..num_values as usize {
            let row_start = offsets_u64[row_idx] as usize;
            let row_end = offsets_u64[row_idx + 1] as usize;
            if row_end > data.data.len() || row_start > row_end {
                return Err(Error::invalid_input(
                    "Packed struct row bounds exceed buffer",
                    location!(),
                ));
            }
            let mut cursor = row_start;
            for (field, accumulator) in self.fields.iter().zip(accumulators.iter_mut()) {
                match (&field.kind, accumulator) {
                    (
                        VariablePackedStructFieldKind::Fixed { bits_per_value, .. },
                        FieldAccumulator::Fixed {
                            builder,
                            bits_per_value: acc_bits,
                        },
                    ) => {
                        debug_assert_eq!(bits_per_value, acc_bits);
                        let bytes_per_value = (bits_per_value / 8) as usize;
                        let end = cursor + bytes_per_value;
                        if end > row_end {
                            return Err(Error::invalid_input(
                                "Packed struct fixed child exceeds row bounds",
                                location!(),
                            ));
                        }
                        let value_block = DataBlock::FixedWidth(FixedWidthDataBlock {
                            data: LanceBuffer::from(data.data[cursor..end].to_vec()),
                            bits_per_value: *bits_per_value,
                            num_values: 1,
                            block_info: BlockInfo::new(),
                        });
                        builder.append(&value_block, 0..1);
                        cursor = end;
                    }
                    (
                        VariablePackedStructFieldKind::Variable {
                            bits_per_length, ..
                        },
                        FieldAccumulator::Variable32 { builder },
                    ) => {
                        if *bits_per_length != 32 {
                            return Err(Error::invalid_input(
                                "Packed struct length prefix size mismatch",
                                location!(),
                            ));
                        }
                        let end = cursor + std::mem::size_of::<u32>();
                        if end > row_end {
                            return Err(Error::invalid_input(
                                "Packed struct variable child length prefix out of bounds",
                                location!(),
                            ));
                        }
                        let len = u32::from_le_bytes(
                            data.data[cursor..end]
                                .try_into()
                                .expect("slice has exact length"),
                        ) as usize;
                        cursor = end;
                        let value_end = cursor + len;
                        if value_end > row_end {
                            return Err(Error::invalid_input(
                                "Packed struct variable child exceeds row bounds",
                                location!(),
                            ));
                        }
                        let value_block = DataBlock::VariableWidth(VariableWidthBlock {
                            data: LanceBuffer::from(data.data[cursor..value_end].to_vec()),
                            bits_per_offset: 32,
                            offsets: LanceBuffer::reinterpret_vec(vec![0_u32, len as u32]),
                            num_values: 1,
                            block_info: BlockInfo::new(),
                        });
                        builder.append(&value_block, 0..1);
                        cursor = value_end;
                    }
                    (
                        VariablePackedStructFieldKind::Variable {
                            bits_per_length, ..
                        },
                        FieldAccumulator::Variable64 { builder },
                    ) => {
                        if *bits_per_length != 64 {
                            return Err(Error::invalid_input(
                                "Packed struct length prefix size mismatch",
                                location!(),
                            ));
                        }
                        let end = cursor + std::mem::size_of::<u64>();
                        if end > row_end {
                            return Err(Error::invalid_input(
                                "Packed struct variable child length prefix out of bounds",
                                location!(),
                            ));
                        }
                        let len = u64::from_le_bytes(
                            data.data[cursor..end]
                                .try_into()
                                .expect("slice has exact length"),
                        ) as usize;
                        cursor = end;
                        let value_end = cursor + len;
                        if value_end > row_end {
                            return Err(Error::invalid_input(
                                "Packed struct variable child exceeds row bounds",
                                location!(),
                            ));
                        }
                        let value_block = DataBlock::VariableWidth(VariableWidthBlock {
                            data: LanceBuffer::from(data.data[cursor..value_end].to_vec()),
                            bits_per_offset: 64,
                            offsets: LanceBuffer::reinterpret_vec(vec![0_u64, len as u64]),
                            num_values: 1,
                            block_info: BlockInfo::new(),
                        });
                        builder.append(&value_block, 0..1);
                        cursor = value_end;
                    }
                    _ => {
                        return Err(Error::invalid_input(
                            "Packed struct accumulator kind mismatch",
                            location!(),
                        ))
                    }
                }
            }
            if cursor != row_end {
                return Err(Error::invalid_input(
                    "Packed struct row parsing did not consume full row",
                    location!(),
                ));
            }
        }

        let mut children = Vec::with_capacity(self.fields.len());
        for (field, accumulator) in self.fields.iter().zip(accumulators.into_iter()) {
            match (field, accumulator) {
                (
                    VariablePackedStructFieldDecoder {
                        kind: VariablePackedStructFieldKind::Fixed { decompressor, .. },
                    },
                    FieldAccumulator::Fixed { builder, .. },
                ) => {
                    let DataBlock::FixedWidth(block) = builder.finish() else {
                        panic!("Expected fixed-width datablock from builder");
                    };
                    let decoded = decompressor.decompress(block, num_values)?;
                    children.push(decoded);
                }
                (
                    VariablePackedStructFieldDecoder {
                        kind:
                            VariablePackedStructFieldKind::Variable {
                                bits_per_length,
                                decompressor,
                            },
                    },
                    FieldAccumulator::Variable32 { builder },
                ) => {
                    let DataBlock::VariableWidth(mut block) = builder.finish() else {
                        panic!("Expected variable-width datablock from builder");
                    };
                    debug_assert_eq!(block.bits_per_offset, 32);
                    block.bits_per_offset = (*bits_per_length) as u8;
                    let decoded = decompressor.decompress(block)?;
                    children.push(decoded);
                }
                (
                    VariablePackedStructFieldDecoder {
                        kind:
                            VariablePackedStructFieldKind::Variable {
                                bits_per_length,
                                decompressor,
                            },
                    },
                    FieldAccumulator::Variable64 { builder },
                ) => {
                    let DataBlock::VariableWidth(mut block) = builder.finish() else {
                        panic!("Expected variable-width datablock from builder");
                    };
                    debug_assert_eq!(block.bits_per_offset, 64);
                    block.bits_per_offset = (*bits_per_length) as u8;
                    let decoded = decompressor.decompress(block)?;
                    children.push(decoded);
                }
                _ => {
                    return Err(Error::invalid_input(
                        "Packed struct accumulator mismatch during finalize",
                        location!(),
                    ))
                }
            }
        }

        Ok(DataBlock::Struct(StructDataBlock {
            children,
            block_info: BlockInfo::new(),
            validity: None,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        compression::{DefaultCompressionStrategy, DefaultDecompressionStrategy},
        statistics::ComputeStat,
    };
    use arrow_schema::{DataType, Field as ArrowField, Fields};

    #[test]
    fn variable_packed_struct_round_trip() -> Result<()> {
        let arrow_fields: Fields = vec![
            ArrowField::new("id", DataType::UInt32, false),
            ArrowField::new("name", DataType::Utf8, true),
        ]
        .into();
        let arrow_struct = ArrowField::new("item", DataType::Struct(arrow_fields), false);
        let struct_field = Field::try_from(&arrow_struct)?;

        let ids = vec![1_u32, 2, 42];
        let id_bytes = ids
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let mut id_block = FixedWidthDataBlock {
            data: LanceBuffer::reinterpret_vec(ids),
            bits_per_value: 32,
            num_values: 3,
            block_info: BlockInfo::new(),
        };
        id_block.compute_stat();
        let id_block = DataBlock::FixedWidth(id_block);

        let name_offsets = vec![0_i32, 1, 4, 4];
        let name_bytes = b"abcz".to_vec();
        let mut name_block = VariableWidthBlock {
            data: LanceBuffer::from(name_bytes.clone()),
            bits_per_offset: 32,
            offsets: LanceBuffer::reinterpret_vec(name_offsets.clone()),
            num_values: 3,
            block_info: BlockInfo::new(),
        };
        name_block.compute_stat();
        let name_block = DataBlock::VariableWidth(name_block);

        let struct_block = StructDataBlock {
            children: vec![id_block, name_block],
            block_info: BlockInfo::new(),
            validity: None,
        };

        let data_block = DataBlock::Struct(struct_block);

        let compression_strategy = DefaultCompressionStrategy::new();
        let compressor = crate::compression::CompressionStrategy::create_per_value(
            &compression_strategy,
            &struct_field,
            &data_block,
        )?;
        let (compressed, encoding) = compressor.compress(data_block)?;

        let PerValueDataBlock::Variable(zipped) = compressed else {
            panic!("expected variable-width packed struct output");
        };

        let decompression_strategy = DefaultDecompressionStrategy::default();
        let decompressor =
            crate::compression::DecompressionStrategy::create_variable_per_value_decompressor(
                &decompression_strategy,
                &encoding,
            )?;
        let decoded = decompressor.decompress(zipped)?;

        let DataBlock::Struct(decoded_struct) = decoded else {
            panic!("expected struct datablock after decode");
        };

        let decoded_id = decoded_struct.children[0].as_fixed_width_ref().unwrap();
        assert_eq!(decoded_id.bits_per_value, 32);
        assert_eq!(decoded_id.data.as_ref(), id_bytes.as_slice());

        let decoded_name = decoded_struct.children[1].as_variable_width_ref().unwrap();
        assert_eq!(decoded_name.bits_per_offset, 32);
        let decoded_offsets = decoded_name.offsets.borrow_to_typed_slice::<i32>();
        assert_eq!(decoded_offsets.as_ref(), name_offsets.as_slice());
        assert_eq!(decoded_name.data.as_ref(), name_bytes.as_slice());

        Ok(())
    }
}
