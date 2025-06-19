// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Basic encodings for variable width data
//!
//! These are not compression but represent the "leaf" encodings for variable length data
//! where we simply match the data with the rules of the structural encoding.
//!
//! These encodings are transparent since we aren't actually doing any compression.  No information
//! is needed in the encoding description.

use bytemuck::{cast_slice, try_cast_slice};
use byteorder::{ByteOrder, LittleEndian};
use core::panic;
use lance_core::utils::bit::pad_bytes;
use snafu::location;

use crate::compression::{
    BlockCompressor, BlockDecompressor, MiniBlockDecompressor, VariablePerValueDecompressor,
};

use crate::buffer::LanceBuffer;
use crate::data::{BlockInfo, DataBlock, VariableWidthBlock};
use crate::encodings::logical::primitive::fullzip::{PerValueCompressor, PerValueDataBlock};
use crate::encodings::logical::primitive::miniblock::{
    MiniBlockChunk, MiniBlockCompressed, MiniBlockCompressor,
};
use crate::format::{pb, ProtobufUtils};

use lance_core::{Error, Result};

#[derive(Debug, Default)]
pub struct BinaryMiniBlockEncoder {}

const AIM_MINICHUNK_SIZE: u32 = 4 * 1024;
const BINARY_MINIBLOCK_CHUNK_ALIGNMENT: usize = 4;

// search for the next offset index to cut the values into a chunk.
// this function incrementally peek the number of values in a chunk,
// each time multiplies the number of values by 2.
// It returns the offset_idx in `offsets` that belongs to this chunk.
fn search_next_offset_idx(offsets: &[u32], last_offset_idx: usize) -> usize {
    let mut num_values = 1;
    let mut new_num_values = num_values * 2;
    loop {
        if last_offset_idx + new_num_values >= offsets.len() {
            if (offsets[offsets.len() - 1] - offsets[last_offset_idx])
                + (offsets.len() - last_offset_idx) as u32 * 4
                <= AIM_MINICHUNK_SIZE
            {
                // case 1: can fit the rest of all data into a miniblock
                return offsets.len() - 1;
            } else {
                // case 2: can only fit the last tried `num_values` into a miniblock
                return last_offset_idx + num_values;
            }
        }
        if ((offsets[last_offset_idx + new_num_values] - offsets[last_offset_idx])
            + ((new_num_values + 1) * 4) as u32)
            <= AIM_MINICHUNK_SIZE
        {
            num_values = new_num_values;
            new_num_values *= 2;
        } else {
            break;
        }
    }
    last_offset_idx + num_values
}

impl BinaryMiniBlockEncoder {
    // put binary data into chunks, every chunk is less than or equal to `AIM_MINICHUNK_SIZE`.
    // In each chunk, offsets are put first then followed by binary bytes data, each chunk is padded to 8 bytes.
    // the offsets in the chunk points to the bytes offset in this chunk.
    fn chunk_data(
        &self,
        mut data: VariableWidthBlock,
    ) -> (MiniBlockCompressed, crate::format::pb::ArrayEncoding) {
        assert!(data.bits_per_offset == 32);

        let offsets = data.offsets.borrow_to_typed_slice::<u32>();
        let offsets = offsets.as_ref();

        assert!(offsets.len() > 1);

        #[derive(Debug)]
        struct ChunkInfo {
            chunk_start_offset_in_orig_idx: usize,
            chunk_last_offset_in_orig_idx: usize,
            // the bytes in every chunk starts at `chunk.bytes_start_offset`
            bytes_start_offset: usize,
            // every chunk is padded to 8 bytes.
            // we need to interpret every chunk as &[u32] so we need it to padded at least to 4 bytes,
            // this field can actually be eliminated and I can use `num_bytes` in `MiniBlockChunk` to compute
            // the `output_total_bytes`.
            padded_chunk_size: usize,
        }

        let mut chunks_info = vec![];
        let mut chunks = vec![];
        let mut last_offset_in_orig_idx = 0;
        const CHUNK_PAD_BUFFER: [u8; BINARY_MINIBLOCK_CHUNK_ALIGNMENT] =
            [72; BINARY_MINIBLOCK_CHUNK_ALIGNMENT];
        loop {
            let this_last_offset_in_orig_idx =
                search_next_offset_idx(offsets, last_offset_in_orig_idx);

            // case 1: last chunk
            if this_last_offset_in_orig_idx == offsets.len() - 1 {
                let num_values_in_this_chunk =
                    this_last_offset_in_orig_idx - last_offset_in_orig_idx;

                let this_chunk_size = (num_values_in_this_chunk + 1) * 4
                    + (offsets[offsets.len() - 1] - offsets[last_offset_in_orig_idx]) as usize;

                let padded_chunk_size = this_chunk_size.next_multiple_of(4);

                // the bytes are put after the offsets
                let this_chunk_bytes_start_offset = (num_values_in_this_chunk + 1) * 4;
                chunks_info.push(ChunkInfo {
                    chunk_start_offset_in_orig_idx: last_offset_in_orig_idx,
                    chunk_last_offset_in_orig_idx: this_last_offset_in_orig_idx,
                    bytes_start_offset: this_chunk_bytes_start_offset,
                    padded_chunk_size,
                });
                chunks.push(MiniBlockChunk {
                    log_num_values: 0,
                    buffer_sizes: vec![padded_chunk_size as u16],
                });
                break;
            } else {
                // case 2: not the last chunk
                let num_values_in_this_chunk =
                    this_last_offset_in_orig_idx - last_offset_in_orig_idx;

                let this_chunk_size = (num_values_in_this_chunk + 1) * 4
                    + (offsets[this_last_offset_in_orig_idx] - offsets[last_offset_in_orig_idx])
                        as usize;

                let padded_chunk_size = this_chunk_size.next_multiple_of(4);

                // the bytes are put after the offsets
                let this_chunk_bytes_start_offset = (num_values_in_this_chunk + 1) * 4;

                chunks_info.push(ChunkInfo {
                    chunk_start_offset_in_orig_idx: last_offset_in_orig_idx,
                    chunk_last_offset_in_orig_idx: this_last_offset_in_orig_idx,
                    bytes_start_offset: this_chunk_bytes_start_offset,
                    padded_chunk_size,
                });

                chunks.push(MiniBlockChunk {
                    log_num_values: num_values_in_this_chunk.trailing_zeros() as u8,
                    buffer_sizes: vec![padded_chunk_size as u16],
                });

                last_offset_in_orig_idx = this_last_offset_in_orig_idx;
            }
        }
        let output_total_bytes = chunks_info
            .iter()
            .map(|chunk_info| chunk_info.padded_chunk_size)
            .sum::<usize>();

        let mut output: Vec<u8> = Vec::with_capacity(output_total_bytes);
        for chunk in chunks_info {
            // `this_chunk_offsets` are offsets that points to bytes in this chunk,
            let this_chunk_offsets = offsets
                [chunk.chunk_start_offset_in_orig_idx..chunk.chunk_last_offset_in_orig_idx + 1]
                .iter()
                .map(|offset| {
                    offset - offsets[chunk.chunk_start_offset_in_orig_idx]
                        + chunk.bytes_start_offset as u32
                })
                .collect::<Vec<_>>();

            output.extend_from_slice(cast_slice(&this_chunk_offsets));

            let start_in_orig = offsets[chunk.chunk_start_offset_in_orig_idx];
            let end_in_orig = offsets[chunk.chunk_last_offset_in_orig_idx];

            output.extend_from_slice(&data.data[start_in_orig as usize..end_in_orig as usize]);

            // pad this chunk to make it align to 4 bytes.
            output.extend_from_slice(
                &CHUNK_PAD_BUFFER[..pad_bytes::<BINARY_MINIBLOCK_CHUNK_ALIGNMENT>(output.len())],
            );
        }

        (
            MiniBlockCompressed {
                data: vec![LanceBuffer::reinterpret_vec(output)],
                chunks,
                num_values: data.num_values,
            },
            ProtobufUtils::variable(/*bits_per_value=*/ 32),
        )
    }
}

impl MiniBlockCompressor for BinaryMiniBlockEncoder {
    fn compress(&self, data: DataBlock) -> Result<(MiniBlockCompressed, pb::ArrayEncoding)> {
        match data {
            DataBlock::VariableWidth(variable_width) => Ok(self.chunk_data(variable_width)),
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

#[derive(Debug, Default)]
pub struct BinaryMiniBlockDecompressor {}

impl MiniBlockDecompressor for BinaryMiniBlockDecompressor {
    // decompress a MiniBlock of binary data, the num_values must be less than or equal
    // to the number of values this MiniBlock has, BinaryMiniBlock doesn't store `the number of values`
    // it has so assertion can not be done here and the caller of `decompress` must ensure `num_values` <= number of values in the chunk.
    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        assert_eq!(data.len(), 1);
        let data = data.into_iter().next().unwrap();
        assert!(data.len() >= 8);
        let offsets: &[u32] = try_cast_slice(&data)
            .expect("casting buffer failed during BinaryMiniBlock decompression");

        let result_offsets = offsets[0..(num_values + 1) as usize]
            .iter()
            .map(|offset| offset - offsets[0])
            .collect::<Vec<u32>>();

        Ok(DataBlock::VariableWidth(VariableWidthBlock {
            data: LanceBuffer::Owned(
                data[offsets[0] as usize..offsets[num_values as usize] as usize].to_vec(),
            ),
            offsets: LanceBuffer::reinterpret_vec(result_offsets),
            bits_per_offset: 32,
            num_values,
            block_info: BlockInfo::new(),
        }))
    }
}

/// Most basic encoding for variable-width data which does no compression at all
#[derive(Debug, Default)]
pub struct VariableEncoder {}

impl BlockCompressor for VariableEncoder {
    fn compress(&self, data: DataBlock) -> Result<LanceBuffer> {
        let num_values: u32 = data
            .num_values()
            .try_into()
            .expect("The Maximum number of values BinaryBlockEncoder can work with is u32::MAX");

        match data {
            DataBlock::VariableWidth(mut variable_width_data) => {
                if variable_width_data.bits_per_offset != 32 {
                    panic!("BinaryBlockEncoder only works with 32 bits per offset VariableWidth DataBlock.");
                }
                let offsets = variable_width_data.offsets.borrow_to_typed_slice::<u32>();
                let offsets = offsets.as_ref();
                // the first 4 bytes store the number of values, then 4 bytes for bytes_start_offset,
                // then offsets data, then bytes data.
                let bytes_start_offset = 4 + 4 + std::mem::size_of_val(offsets) as u32;

                let output_total_bytes =
                    bytes_start_offset as usize + variable_width_data.data.len();
                let mut output: Vec<u8> = Vec::with_capacity(output_total_bytes);

                // store `num_values` in the first 4 bytes of output buffer
                output.extend_from_slice(&(num_values).to_le_bytes());

                // store `bytes_start_offset` in the next 4 bytes of output buffer
                output.extend_from_slice(&(bytes_start_offset).to_le_bytes());

                // store offsets
                output.extend_from_slice(cast_slice(offsets));

                // store bytes
                output.extend_from_slice(&variable_width_data.data);

                Ok(LanceBuffer::Owned(output))
            }
            _ => {
                panic!("BinaryBlockEncoder can only work with Variable Width DataBlock.");
            }
        }
    }
}

impl PerValueCompressor for VariableEncoder {
    fn compress(&self, data: DataBlock) -> Result<(PerValueDataBlock, pb::ArrayEncoding)> {
        let DataBlock::VariableWidth(variable) = data else {
            panic!("BinaryPerValueCompressor can only work with Variable Width DataBlock.");
        };

        let encoding = ProtobufUtils::variable(variable.bits_per_offset);
        Ok((PerValueDataBlock::Variable(variable), encoding))
    }
}

#[derive(Debug, Default)]
pub struct VariableDecoder {}

impl VariablePerValueDecompressor for VariableDecoder {
    fn decompress(&self, data: VariableWidthBlock) -> Result<DataBlock> {
        Ok(DataBlock::VariableWidth(data))
    }
}

#[derive(Debug, Default)]
pub struct BinaryBlockDecompressor {}

impl BlockDecompressor for BinaryBlockDecompressor {
    fn decompress(&self, data: LanceBuffer, num_values: u64) -> Result<DataBlock> {
        // the first 4 bytes in the BinaryBlock compressed buffer stores the num_values this block has.
        debug_assert_eq!(num_values, LittleEndian::read_u32(&data[..4]) as u64);

        // the next 4 bytes in the BinaryBlock compressed buffer stores the bytes_start_offset.
        let bytes_start_offset = LittleEndian::read_u32(&data[4..8]);

        // the next `bytes_start_offset - 8` stores the offsets.
        let offsets = data.slice_with_length(8, bytes_start_offset as usize - 8);

        // the rest are the binary bytes.
        let data = data.slice_with_length(
            bytes_start_offset as usize,
            data.len() - bytes_start_offset as usize,
        );

        Ok(DataBlock::VariableWidth(VariableWidthBlock {
            data,
            offsets,
            bits_per_offset: 32,
            num_values,
            block_info: BlockInfo::new(),
        }))
    }
}

#[cfg(test)]
pub mod tests {
    use arrow_array::{
        builder::{LargeStringBuilder, StringBuilder},
        ArrayRef, StringArray,
    };
    use arrow_schema::{DataType, Field};

    use lance_core::datatypes::{
        COMPRESSION_META_KEY, STRUCTURAL_ENCODING_FULLZIP, STRUCTURAL_ENCODING_META_KEY,
        STRUCTURAL_ENCODING_MINIBLOCK,
    };
    use rstest::rstest;
    use std::{collections::HashMap, sync::Arc, vec};

    use crate::{
        testing::{check_round_trip_encoding_of_data, check_round_trip_encoding_random, TestCases},
        version::LanceFileVersion,
    };

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_utf8_binary(
        #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
    ) {
        let field = Field::new("", DataType::Utf8, false);
        check_round_trip_encoding_random(field, version).await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_binary(
        #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        #[values(STRUCTURAL_ENCODING_MINIBLOCK, STRUCTURAL_ENCODING_FULLZIP)]
        structural_encoding: &str,
        #[values(DataType::Utf8, DataType::Binary)] data_type: DataType,
    ) {
        use lance_core::datatypes::STRUCTURAL_ENCODING_META_KEY;

        let mut field_metadata = HashMap::new();
        field_metadata.insert(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            structural_encoding.into(),
        );

        let field = Field::new("", data_type, false).with_metadata(field_metadata);
        check_round_trip_encoding_random(field, version).await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_binary_fsst(
        #[values(STRUCTURAL_ENCODING_MINIBLOCK, STRUCTURAL_ENCODING_FULLZIP)]
        structural_encoding: &str,
    ) {
        let mut field_metadata = HashMap::new();
        field_metadata.insert(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            structural_encoding.into(),
        );
        field_metadata.insert(COMPRESSION_META_KEY.to_string(), "fsst".into());

        let field = Field::new("", DataType::Utf8, true).with_metadata(field_metadata);
        check_round_trip_encoding_random(field, LanceFileVersion::V2_1).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_large_binary() {
        let field = Field::new("", DataType::LargeBinary, true);
        check_round_trip_encoding_random(field, LanceFileVersion::V2_0).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_large_utf8() {
        let field = Field::new("", DataType::LargeUtf8, true);
        check_round_trip_encoding_random(field, LanceFileVersion::V2_0).await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_simple_binary(
        #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        #[values(STRUCTURAL_ENCODING_MINIBLOCK, STRUCTURAL_ENCODING_FULLZIP)]
        structural_encoding: &str,
        #[values(DataType::Utf8, DataType::Binary)] data_type: DataType,
    ) {
        use lance_core::datatypes::STRUCTURAL_ENCODING_META_KEY;

        let string_array = StringArray::from(vec![Some("abc"), None, Some("pqr"), None, Some("m")]);
        let string_array = arrow_cast::cast(&string_array, &data_type).unwrap();

        let mut field_metadata = HashMap::new();
        field_metadata.insert(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            structural_encoding.into(),
        );

        let test_cases = TestCases::default()
            .with_range(0..2)
            .with_range(0..3)
            .with_range(1..3)
            .with_indices(vec![0, 1, 3, 4])
            .with_file_version(version);
        check_round_trip_encoding_of_data(
            vec![Arc::new(string_array)],
            &test_cases,
            field_metadata,
        )
        .await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_sliced_utf8(
        #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
    ) {
        let string_array = StringArray::from(vec![Some("abc"), Some("de"), None, Some("fgh")]);
        let string_array = string_array.slice(1, 3);

        let test_cases = TestCases::default()
            .with_range(0..1)
            .with_range(0..2)
            .with_range(1..2)
            .with_file_version(version);
        check_round_trip_encoding_of_data(
            vec![Arc::new(string_array)],
            &test_cases,
            HashMap::new(),
        )
        .await;
    }

    #[test_log::test(tokio::test)]
    async fn test_bigger_than_max_page_size() {
        // Create an array with one single 32MiB string
        let big_string = String::from_iter((0..(32 * 1024 * 1024)).map(|_| '0'));
        let string_array = StringArray::from(vec![
            Some(big_string),
            Some("abc".to_string()),
            None,
            None,
            Some("xyz".to_string()),
        ]);

        // Drop the max page size to 1MiB
        let test_cases = TestCases::default().with_max_page_size(1024 * 1024);

        check_round_trip_encoding_of_data(
            vec![Arc::new(string_array)],
            &test_cases,
            HashMap::new(),
        )
        .await;

        // This is a regression testing the case where a page with X rows is split into Y parts
        // where the number of parts is not evenly divisible by the number of rows.  In this
        // case we are splitting 90 rows into 4 parts.
        let big_string = String::from_iter((0..(1000 * 1000)).map(|_| '0'));
        let string_array = StringArray::from_iter_values((0..90).map(|_| big_string.clone()));

        check_round_trip_encoding_of_data(
            vec![Arc::new(string_array)],
            &TestCases::default(),
            HashMap::new(),
        )
        .await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_empty_strings(
        #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
    ) {
        // Scenario 1: Some strings are empty

        let values = [Some("abc"), Some(""), None];
        // Test empty list at beginning, middle, and end
        for order in [[0, 1, 2], [1, 0, 2], [2, 0, 1]] {
            let mut string_builder = StringBuilder::new();
            for idx in order {
                string_builder.append_option(values[idx]);
            }
            let string_array = Arc::new(string_builder.finish());
            let test_cases = TestCases::default()
                .with_indices(vec![1])
                .with_indices(vec![0])
                .with_indices(vec![2])
                .with_indices(vec![0, 1])
                .with_file_version(version);
            check_round_trip_encoding_of_data(
                vec![string_array.clone()],
                &test_cases,
                HashMap::new(),
            )
            .await;
            let test_cases = test_cases.with_batch_size(1);
            check_round_trip_encoding_of_data(vec![string_array], &test_cases, HashMap::new())
                .await;
        }

        // Scenario 2: All strings are empty

        // When encoding an array of empty strings there are no bytes to encode
        // which is strange and we want to ensure we handle it
        let string_array = Arc::new(StringArray::from(vec![Some(""), None, Some("")]));

        let test_cases = TestCases::default().with_range(0..2).with_indices(vec![1]);
        check_round_trip_encoding_of_data(vec![string_array.clone()], &test_cases, HashMap::new())
            .await;
        let test_cases = test_cases.with_batch_size(1);
        check_round_trip_encoding_of_data(vec![string_array], &test_cases, HashMap::new()).await;
    }

    #[test_log::test(tokio::test)]
    #[ignore] // This test is quite slow in debug mode
    async fn test_jumbo_string() {
        // This is an overflow test.  We have a list of lists where each list
        // has 1Mi items.  We encode 5000 of these lists and so we have over 4Gi in the
        // offsets range
        let mut string_builder = LargeStringBuilder::new();
        // a 1 MiB string
        let giant_string = String::from_iter((0..(1024 * 1024)).map(|_| '0'));
        for _ in 0..5000 {
            string_builder.append_option(Some(&giant_string));
        }
        let giant_array = Arc::new(string_builder.finish()) as ArrayRef;
        let arrs = vec![giant_array];

        // // We can't validate because our validation relies on concatenating all input arrays
        let test_cases = TestCases::default().without_validation();
        check_round_trip_encoding_of_data(arrs, &test_cases, HashMap::new()).await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_binary_dictionary_encoding(
        #[values(true, false)] with_nulls: bool,
        #[values(100, 500, 35000)] dict_size: u32,
    ) {
        let test_cases = TestCases::default().with_file_version(LanceFileVersion::V2_1);
        let strings = (0..dict_size)
            .map(|i| i.to_string())
            .collect::<Vec<String>>();

        let repeated_strings: Vec<_> = strings
            .iter()
            .cycle()
            .take(70000)
            .enumerate()
            .map(|(i, s)| {
                if with_nulls && i % 7 == 0 {
                    None
                } else {
                    Some(s.to_string())
                }
            })
            .collect();
        let string_array = Arc::new(StringArray::from(repeated_strings)) as ArrayRef;
        check_round_trip_encoding_of_data(vec![string_array], &test_cases, HashMap::new()).await;
    }
}
