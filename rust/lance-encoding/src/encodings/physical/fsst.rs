// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{ops::Range, sync::Arc};

use arrow_buffer::ScalarBuffer;
use arrow_schema::DataType;
use futures::{future::BoxFuture, FutureExt};

use lance_core::{Error, Result};
use snafu::{location, Location};

use crate::{
    buffer::LanceBuffer,
    data::{BlockInfo, DataBlock, NullableDataBlock, VariableWidthBlock},
    decoder::{MiniBlockDecompressor, PageScheduler, PrimitivePageDecoder},
    encoder::{ArrayEncoder, EncodedArray},
    encoder::{MiniBlockCompressed, MiniBlockCompressor},
    format::pb::{self},
    format::ProtobufUtils,
    EncodingsIo,
};

use super::binary::{BinaryMiniBlockDecompressor, BinaryMiniBlockEncoder};

#[derive(Debug)]
pub struct FsstPageScheduler {
    inner_scheduler: Box<dyn PageScheduler>,
    symbol_table: Vec<u8>,
}

impl FsstPageScheduler {
    pub fn new(inner_scheduler: Box<dyn PageScheduler>, symbol_table: Vec<u8>) -> Self {
        Self {
            inner_scheduler,
            symbol_table,
        }
    }
}

impl PageScheduler for FsstPageScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[Range<u64>],
        scheduler: &Arc<dyn EncodingsIo>,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PrimitivePageDecoder>>> {
        let inner_decoder = self
            .inner_scheduler
            .schedule_ranges(ranges, scheduler, top_level_row);
        let symbol_table = self.symbol_table.clone();

        async move {
            let inner_decoder = inner_decoder.await?;
            Ok(Box::new(FsstPageDecoder {
                inner_decoder,
                symbol_table,
            }) as Box<dyn PrimitivePageDecoder>)
        }
        .boxed()
    }
}

struct FsstPageDecoder {
    inner_decoder: Box<dyn PrimitivePageDecoder>,
    symbol_table: Vec<u8>,
}

impl PrimitivePageDecoder for FsstPageDecoder {
    fn decode(&self, rows_to_skip: u64, num_rows: u64) -> Result<DataBlock> {
        let compressed_data = self.inner_decoder.decode(rows_to_skip, num_rows)?;
        let (string_data, nulls) = match compressed_data {
            DataBlock::Nullable(nullable) => {
                let data = nullable.data.as_variable_width().unwrap();
                Result::Ok((data, Some(nullable.nulls)))
            }
            DataBlock::VariableWidth(variable) => Ok((variable, None)),
            _ => panic!("Received non-variable width data from inner decoder"),
        }?;

        let offsets = ScalarBuffer::<i32>::from(string_data.offsets.into_buffer());
        let bytes = string_data.data.into_buffer();

        let mut decompressed_offsets = vec![0_i32; offsets.len()];
        let mut decompressed_bytes = vec![0_u8; bytes.len() * 8];
        // Safety: Exposes uninitialized memory but we're about to clobber it
        unsafe {
            decompressed_bytes.set_len(decompressed_bytes.capacity());
        }
        fsst::fsst::decompress(
            &self.symbol_table,
            &bytes,
            &offsets,
            &mut decompressed_bytes,
            &mut decompressed_offsets,
        )?;

        // TODO: Change PrimitivePageDecoder to use Vec instead of BytesMut
        // since there is no way to get BytesMut from Vec but these copies should be avoidable
        // This is not the first time this has happened
        let mut offsets_as_bytes_mut = Vec::with_capacity(decompressed_offsets.len());
        let decompressed_offsets = ScalarBuffer::<i32>::from(decompressed_offsets);
        offsets_as_bytes_mut.extend_from_slice(decompressed_offsets.inner().as_slice());

        let mut bytes_as_bytes_mut = Vec::with_capacity(decompressed_bytes.len());
        bytes_as_bytes_mut.extend_from_slice(&decompressed_bytes);

        let new_string_data = DataBlock::VariableWidth(VariableWidthBlock {
            bits_per_offset: 32,
            data: LanceBuffer::from(bytes_as_bytes_mut),
            num_values: num_rows,
            offsets: LanceBuffer::from(offsets_as_bytes_mut),
            block_info: BlockInfo::new(),
        });

        if let Some(nulls) = nulls {
            Ok(DataBlock::Nullable(NullableDataBlock {
                data: Box::new(new_string_data),
                nulls,
                block_info: BlockInfo::new(),
            }))
        } else {
            Ok(new_string_data)
        }
    }
}

#[derive(Debug)]
pub struct FsstArrayEncoder {
    inner_encoder: Box<dyn ArrayEncoder>,
}

impl FsstArrayEncoder {
    pub fn new(inner_encoder: Box<dyn ArrayEncoder>) -> Self {
        Self { inner_encoder }
    }
}

impl ArrayEncoder for FsstArrayEncoder {
    fn encode(
        &self,
        data: DataBlock,
        data_type: &DataType,
        buffer_index: &mut u32,
    ) -> lance_core::Result<EncodedArray> {
        let (mut data, nulls) = match data {
            DataBlock::Nullable(nullable) => {
                let data = nullable.data.as_variable_width().unwrap();
                (data, Some(nullable.nulls))
            }
            DataBlock::VariableWidth(variable) => (variable, None),
            _ => panic!("Expected variable width data block"),
        };
        assert_eq!(data.bits_per_offset, 32);
        let num_values = data.num_values;
        let offsets = data.offsets.borrow_to_typed_slice::<i32>();
        let offsets_slice = offsets.as_ref();
        let bytes_data = data.data.into_buffer();

        let mut dest_offsets = vec![0_i32; offsets_slice.len() * 2];
        let mut dest_values = vec![0_u8; bytes_data.len() * 2];
        let mut symbol_table = vec![0_u8; fsst::fsst::FSST_SYMBOL_TABLE_SIZE];

        fsst::fsst::compress(
            &mut symbol_table,
            bytes_data.as_slice(),
            offsets_slice,
            &mut dest_values,
            &mut dest_offsets,
        )?;

        let dest_offset = LanceBuffer::reinterpret_vec(dest_offsets);
        let dest_values = LanceBuffer::Owned(dest_values);
        let dest_data = DataBlock::VariableWidth(VariableWidthBlock {
            bits_per_offset: 32,
            data: dest_values,
            num_values,
            offsets: dest_offset,
            block_info: BlockInfo::new(),
        });

        let data_block = if let Some(nulls) = nulls {
            DataBlock::Nullable(NullableDataBlock {
                data: Box::new(dest_data),
                nulls,
                block_info: BlockInfo::new(),
            })
        } else {
            dest_data
        };

        let inner_encoded = self
            .inner_encoder
            .encode(data_block, data_type, buffer_index)?;

        let encoding = ProtobufUtils::fsst(inner_encoded.encoding, symbol_table);

        Ok(EncodedArray {
            data: inner_encoded.data,
            encoding,
        })
    }
}

#[derive(Debug, Default)]
pub struct FsstMiniBlockEncoder {}

impl MiniBlockCompressor for FsstMiniBlockEncoder {
    fn compress(
        &self,
        data: DataBlock,
    ) -> Result<(MiniBlockCompressed, crate::format::pb::ArrayEncoding)> {
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
                let data_block = DataBlock::VariableWidth(VariableWidthBlock {
                    data: LanceBuffer::reinterpret_vec(dest_values),
                    bits_per_offset: 32,
                    offsets: LanceBuffer::reinterpret_vec(dest_offsets),
                    num_values: variable_width.num_values,
                    block_info: BlockInfo::new(),
                });

                // compress the fsst compressed data using `BinaryMiniBlockEncoder`
                let binary_compressor =
                    Box::new(BinaryMiniBlockEncoder::default()) as Box<dyn MiniBlockCompressor>;

                let (binary_miniblock_compressed, binary_array_encoding) =
                    binary_compressor.compress(data_block)?;

                Ok((
                    binary_miniblock_compressed,
                    ProtobufUtils::fsst_mini_block(binary_array_encoding, symbol_table),
                ))
            }
            _ => Err(Error::InvalidInput {
                source: format!(
                    "Cannot compress a data block of type {} with BinaryMiniBlockEncoder",
                    data.name()
                )
                .into(),
                location: location!(),
            }),
        }
    }
}

#[derive(Debug)]
pub struct FsstMiniBlockDecompressor {
    symbol_table: Vec<u8>,
}

impl FsstMiniBlockDecompressor {
    pub fn new(description: &pb::FsstMiniBlock) -> Self {
        Self {
            symbol_table: description.symbol_table.clone(),
        }
    }
}

impl MiniBlockDecompressor for FsstMiniBlockDecompressor {
    fn decompress(&self, data: LanceBuffer, num_values: u64) -> Result<DataBlock> {
        // Step 1. decompress data use `BinaryMiniBlockDecompressor`
        let binary_decompressor =
            Box::new(BinaryMiniBlockDecompressor::default()) as Box<dyn MiniBlockDecompressor>;
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
