// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! FSST encoding
//!
//! FSST is a lightweight encoding for variable width data.  This module includes
//! adapters for both miniblock and per-value encoding.
//!
//! FSST encoding creates a small symbol table that is needed for decoding.  Currently
//! we create one symbol table per disk page and store it in the description.
//!
//! TODO: This seems to be potentially limiting.  Perhaps we should create one symbol
//! table per mini-block chunk?  In the per-value compression it may even make sense to
//! create multiple symbol tables for a single value!
//!
//! FSST encoding is transparent.

use lance_core::{Error, Result};
use snafu::location;

use crate::{
    buffer::LanceBuffer,
    compression::{MiniBlockDecompressor, VariablePerValueDecompressor},
    data::{BlockInfo, DataBlock, VariableWidthBlock},
    encodings::logical::primitive::{
        fullzip::{PerValueCompressor, PerValueDataBlock},
        miniblock::{MiniBlockCompressed, MiniBlockCompressor},
    },
    format::{
        pb::{self},
        ProtobufUtils,
    },
};

use super::binary::{BinaryMiniBlockDecompressor, BinaryMiniBlockEncoder};

struct FsstCompressed {
    data: VariableWidthBlock,
    symbol_table: Vec<u8>,
}

impl FsstCompressed {
    fn fsst_compress(data: DataBlock) -> Result<Self> {
        match data {
            DataBlock::VariableWidth(mut variable_width) => {
                let offsets = variable_width.offsets.borrow_to_typed_slice::<i32>();
                let offsets_slice = offsets.as_ref();
                let bytes_data = variable_width.data.into_buffer();

                // prepare compression output buffer
                let mut dest_offsets = vec![0_i32; offsets_slice.len() * 2];
                let mut dest_values = vec![0_u8; bytes_data.len() * 2];
                let mut symbol_table = vec![0_u8; fsst::fsst::FSST_SYMBOL_TABLE_SIZE];

                // fsst compression
                fsst::fsst::compress(
                    &mut symbol_table,
                    bytes_data.as_slice(),
                    offsets_slice,
                    &mut dest_values,
                    &mut dest_offsets,
                )?;

                // construct `DataBlock` for BinaryMiniBlockEncoder, we may want some `DataBlock` construct methods later
                let compressed = VariableWidthBlock {
                    data: LanceBuffer::reinterpret_vec(dest_values),
                    bits_per_offset: 32,
                    offsets: LanceBuffer::reinterpret_vec(dest_offsets),
                    num_values: variable_width.num_values,
                    block_info: BlockInfo::new(),
                };

                Ok(Self {
                    data: compressed,
                    symbol_table,
                })
            }
            _ => Err(Error::InvalidInput {
                source: format!(
                    "Cannot compress a data block of type {} with FsstEncoder",
                    data.name()
                )
                .into(),
                location: location!(),
            }),
        }
    }
}

#[derive(Debug, Default)]
pub struct FsstMiniBlockEncoder {}

impl MiniBlockCompressor for FsstMiniBlockEncoder {
    fn compress(
        &self,
        data: DataBlock,
    ) -> Result<(MiniBlockCompressed, crate::format::pb::ArrayEncoding)> {
        let compressed = FsstCompressed::fsst_compress(data)?;

        let data_block = DataBlock::VariableWidth(compressed.data);

        // compress the fsst compressed data using `BinaryMiniBlockEncoder`
        let binary_compressor =
            Box::new(BinaryMiniBlockEncoder::default()) as Box<dyn MiniBlockCompressor>;

        let (binary_miniblock_compressed, binary_array_encoding) =
            binary_compressor.compress(data_block)?;

        Ok((
            binary_miniblock_compressed,
            ProtobufUtils::fsst(binary_array_encoding, compressed.symbol_table),
        ))
    }
}

#[derive(Debug)]
pub struct FsstPerValueEncoder {
    inner: Box<dyn PerValueCompressor>,
}

impl FsstPerValueEncoder {
    pub fn new(inner: Box<dyn PerValueCompressor>) -> Self {
        Self { inner }
    }
}

impl PerValueCompressor for FsstPerValueEncoder {
    fn compress(&self, data: DataBlock) -> Result<(PerValueDataBlock, pb::ArrayEncoding)> {
        let compressed = FsstCompressed::fsst_compress(data)?;

        let data_block = DataBlock::VariableWidth(compressed.data);

        let (binary_compressed, binary_array_encoding) = self.inner.compress(data_block)?;

        Ok((
            binary_compressed,
            ProtobufUtils::fsst(binary_array_encoding, compressed.symbol_table),
        ))
    }
}

#[derive(Debug)]
pub struct FsstPerValueDecompressor {
    symbol_table: LanceBuffer,
    inner_decompressor: Box<dyn VariablePerValueDecompressor>,
}

impl FsstPerValueDecompressor {
    pub fn new(
        symbol_table: LanceBuffer,
        inner_decompressor: Box<dyn VariablePerValueDecompressor>,
    ) -> Self {
        Self {
            symbol_table,
            inner_decompressor,
        }
    }
}

impl VariablePerValueDecompressor for FsstPerValueDecompressor {
    fn decompress(&self, data: VariableWidthBlock) -> Result<DataBlock> {
        // Step 1. Run inner decompressor
        let mut compressed_variable_data = self
            .inner_decompressor
            .decompress(data)?
            .as_variable_width()
            .unwrap();

        // Step 2. FSST decompress
        let bytes = compressed_variable_data.data.borrow_to_typed_slice::<u8>();
        let bytes = bytes.as_ref();
        let offsets = compressed_variable_data
            .offsets
            .borrow_to_typed_slice::<i32>();
        let offsets = offsets.as_ref();
        let num_values = compressed_variable_data.num_values;

        // The data will expand at most 8 times
        // The offsets will be the same size because we have the same # of strings
        let mut decompress_bytes_buf = vec![0u8; bytes.len() * 8];
        let mut decompress_offset_buf = vec![0i32; offsets.len()];
        fsst::fsst::decompress(
            &self.symbol_table,
            bytes,
            offsets,
            &mut decompress_bytes_buf,
            &mut decompress_offset_buf,
        )?;

        Ok(DataBlock::VariableWidth(VariableWidthBlock {
            data: LanceBuffer::Owned(decompress_bytes_buf),
            offsets: LanceBuffer::reinterpret_vec(decompress_offset_buf),
            bits_per_offset: 32,
            num_values,
            block_info: BlockInfo::new(),
        }))
    }
}

#[derive(Debug)]
pub struct FsstMiniBlockDecompressor {
    symbol_table: LanceBuffer,
}

impl FsstMiniBlockDecompressor {
    pub fn new(description: &pb::Fsst) -> Self {
        Self {
            symbol_table: LanceBuffer::from_bytes(description.symbol_table.clone(), 1),
        }
    }
}

impl MiniBlockDecompressor for FsstMiniBlockDecompressor {
    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        // Step 1. decompress data use `BinaryMiniBlockDecompressor`
        // TODO: Support 64 bits for FSST compressor
        let binary_decompressor =
            Box::new(BinaryMiniBlockDecompressor::new(32)) as Box<dyn MiniBlockDecompressor>;
        let compressed_data_block = binary_decompressor.decompress(data, num_values)?;
        let DataBlock::VariableWidth(mut compressed_data_block) = compressed_data_block else {
            panic!("BinaryMiniBlockDecompressor should output VariableWidth DataBlock")
        };

        // Step 2. FSST decompress
        let bytes = compressed_data_block.data.borrow_to_typed_slice::<u8>();
        let bytes = bytes.as_ref();
        let offsets = compressed_data_block.offsets.borrow_to_typed_slice::<i32>();
        let offsets = offsets.as_ref();

        // FSST decompression output buffer, the `MiniBlock` has a size limit of `4 KiB` and
        // the FSST decompression algorithm output is at most `8 * input_size`
        // Since `MiniBlock Size` <= 4 KiB and `offsets` are type `i32, it has number of `offsets` <= 1024.
        let mut decompress_bytes_buf = vec![0u8; 4 * 1024 * 8];
        let mut decompress_offset_buf = vec![0i32; 1024];
        fsst::fsst::decompress(
            &self.symbol_table,
            bytes,
            offsets,
            &mut decompress_bytes_buf,
            &mut decompress_offset_buf,
        )?;

        Ok(DataBlock::VariableWidth(VariableWidthBlock {
            data: LanceBuffer::Owned(decompress_bytes_buf),
            offsets: LanceBuffer::reinterpret_vec(decompress_offset_buf),
            bits_per_offset: 32,
            num_values,
            block_info: BlockInfo::new(),
        }))
    }
}

#[cfg(test)]
mod tests {

    use std::collections::HashMap;

    use lance_datagen::{ByteCount, RowCount};

    use crate::{
        testing::{check_round_trip_encoding_of_data, TestCases},
        version::LanceFileVersion,
    };

    #[test_log::test(tokio::test)]
    async fn test_fsst() {
        let arr = lance_datagen::gen()
            .anon_col(lance_datagen::array::rand_utf8(ByteCount::from(32), false))
            .into_batch_rows(RowCount::from(1_000_000))
            .unwrap()
            .column(0)
            .clone();
        check_round_trip_encoding_of_data(
            vec![arr],
            &TestCases::default().with_file_version(LanceFileVersion::V2_1),
            HashMap::new(),
        )
        .await;

        let arr = lance_datagen::gen()
            .anon_col(lance_datagen::array::rand_utf8(ByteCount::from(64), false))
            .into_batch_rows(RowCount::from(1_000_000))
            .unwrap()
            .column(0)
            .clone();
        check_round_trip_encoding_of_data(
            vec![arr],
            &TestCases::default().with_file_version(LanceFileVersion::V2_1),
            HashMap::new(),
        )
        .await;
    }
}
