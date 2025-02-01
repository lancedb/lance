// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::datatypes::UInt64Type;

use lance_core::{Error, Result};
use snafu::location;

use crate::{
    buffer::LanceBuffer,
    data::{BlockInfo, DataBlock, FixedWidthDataBlock, StructDataBlock},
    decoder::MiniBlockDecompressor,
    encoder::{MiniBlockCompressed, MiniBlockCompressor},
    format::{
        pb::{self},
        ProtobufUtils,
    },
    statistics::{GetStat, Stat},
};

use super::value::{ValueDecompressor, ValueEncoder};

// Transforms a `StructDataBlock` into a row major `FixedWidthDataBlock`.
// Only fields with fixed-width fields are supported for now, and the
// assumption that all fields has `bits_per_value % 8 == 0` is made.
fn struct_data_block_to_fixed_width_data_block(
    struct_data_block: StructDataBlock,
    bits_per_values: &[u32],
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
        bits_per_value: bits_per_values
            .iter()
            .map(|bits_per_value| *bits_per_value as u64)
            .sum(),
        data: LanceBuffer::Owned(output),
        num_values,
        block_info: BlockInfo::default(),
    })
}

#[derive(Debug, Default)]
pub struct PackedStructFixedWidthMiniBlockEncoder {}

impl MiniBlockCompressor for PackedStructFixedWidthMiniBlockEncoder {
    fn compress(
        &self,
        data: DataBlock,
    ) -> Result<(MiniBlockCompressed, crate::format::pb::ArrayEncoding)> {
        match data {
            DataBlock::Struct(struct_data_block) => {
                let bits_per_values = struct_data_block.children.iter().map(|data_block| data_block.as_fixed_width_ref().unwrap().bits_per_value as u32).collect::<Vec<_>>();

                // transform struct datablock to fixed-width data block.
                let data_block = struct_data_block_to_fixed_width_data_block(struct_data_block, &bits_per_values);

                // store and transformed fixed-width data block.
                let value_miniblock_compressor = Box::new(ValueEncoder::default()) as Box<dyn MiniBlockCompressor>;
                let (value_miniblock_compressed, value_array_encoding) =
                value_miniblock_compressor.compress(data_block)?;

                Ok((
                    value_miniblock_compressed,
                    ProtobufUtils::packed_struct_fixed_width_mini_block(value_array_encoding, bits_per_values),
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
    bits_per_values: Vec<u32>,
    array_encoding: Box<dyn MiniBlockDecompressor>,
}

impl PackedStructFixedWidthMiniBlockDecompressor {
    pub fn new(description: &pb::PackedStructFixedWidthMiniBlock) -> Self {
        let array_encoding: Box<dyn MiniBlockDecompressor> = match description
            .flat
            .as_ref()
            .unwrap()
            .array_encoding
            .as_ref()
            .unwrap()
        {
            pb::array_encoding::ArrayEncoding::Flat(flat) => Box::new(ValueDecompressor::new(flat)),
            _ => panic!("Currently only `ArrayEncoding::Flat` is supported in packed struct encoding in Lance 2.1."),
        };
        Self {
            bits_per_values: description.bits_per_values.clone(),
            array_encoding,
        }
    }
}

impl MiniBlockDecompressor for PackedStructFixedWidthMiniBlockDecompressor {
    fn decompress(&self, data: LanceBuffer, num_values: u64) -> Result<DataBlock> {
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
                data: LanceBuffer::Owned(child_buf),
                bits_per_value: self.bits_per_values[i] as u64,
                num_values,
                block_info: BlockInfo::default(),
            });
            children_data_block.push(child);
        }
        Ok(DataBlock::Struct(StructDataBlock {
            children: children_data_block,
            block_info: BlockInfo::default(),
        }))
    }
}
