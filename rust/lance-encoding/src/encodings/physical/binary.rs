// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Basic encodings for variable width data
//!
//! These are not compression but represent the "leaf" encodings for variable length data
//! where we simply match the data with the rules of the structural encoding.
//!
//! These encodings are transparent since we aren't actually doing any compression.  No information
//! is needed in the encoding description.

use arrow_array::OffsetSizeTrait;
use bytemuck::{cast_slice, try_cast_slice};
use byteorder::{ByteOrder, LittleEndian};
use core::panic;
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

use lance_core::utils::bit::pad_bytes_to;
use lance_core::{Error, Result};

#[derive(Debug, Default)]
pub struct BinaryMiniBlockEncoder {}

const AIM_MINICHUNK_SIZE: i64 = 4 * 1024;

// Make it to support both u32 and u64
fn chunk_offsets<N: OffsetSizeTrait>(
    offsets: &[N],
    data: &[u8],
    alignment: usize,
) -> (Vec<LanceBuffer>, Vec<MiniBlockChunk>) {
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

    let byte_width: usize = N::get_byte_width();
    let mut chunks_info = vec![];
    let mut chunks = vec![];
    let mut last_offset_in_orig_idx = 0;
    loop {
        let this_last_offset_in_orig_idx = search_next_offset_idx(offsets, last_offset_in_orig_idx);

        let num_values_in_this_chunk = this_last_offset_in_orig_idx - last_offset_in_orig_idx;
        let chunk_bytes = offsets[this_last_offset_in_orig_idx] - offsets[last_offset_in_orig_idx];
        let this_chunk_size =
            (num_values_in_this_chunk + 1) * byte_width + chunk_bytes.to_usize().unwrap();

        let padded_chunk_size = this_chunk_size.next_multiple_of(alignment);

        let this_chunk_bytes_start_offset = (num_values_in_this_chunk + 1) * byte_width;
        chunks_info.push(ChunkInfo {
            chunk_start_offset_in_orig_idx: last_offset_in_orig_idx,
            chunk_last_offset_in_orig_idx: this_last_offset_in_orig_idx,
            bytes_start_offset: this_chunk_bytes_start_offset,
            padded_chunk_size,
        });
        chunks.push(MiniBlockChunk {
            log_num_values: if this_last_offset_in_orig_idx == offsets.len() - 1 {
                0
            } else {
                num_values_in_this_chunk.trailing_zeros() as u8
            },
            buffer_sizes: vec![padded_chunk_size as u16],
        });
        if this_last_offset_in_orig_idx == offsets.len() - 1 {
            break;
        }
        last_offset_in_orig_idx = this_last_offset_in_orig_idx;
    }

    let output_total_bytes = chunks_info
        .iter()
        .map(|chunk_info| chunk_info.padded_chunk_size)
        .sum::<usize>();

    let mut output: Vec<u8> = Vec::with_capacity(output_total_bytes);

    for chunk in chunks_info {
        let this_chunk_offsets: Vec<N> = offsets
            [chunk.chunk_start_offset_in_orig_idx..=chunk.chunk_last_offset_in_orig_idx]
            .iter()
            .map(|offset| {
                *offset - offsets[chunk.chunk_start_offset_in_orig_idx]
                    + N::from_usize(chunk.bytes_start_offset).unwrap()
            })
            .collect();

        let this_chunk_offsets = LanceBuffer::reinterpret_vec(this_chunk_offsets);
        output.extend_from_slice(&this_chunk_offsets);

        let start_in_orig = offsets[chunk.chunk_start_offset_in_orig_idx]
            .to_usize()
            .unwrap();
        let end_in_orig = offsets[chunk.chunk_last_offset_in_orig_idx]
            .to_usize()
            .unwrap();
        output.extend_from_slice(&data[start_in_orig..end_in_orig]);

        // pad this chunk to make it align to desired bytes.
        const PAD_BYTE: u8 = 72;
        let pad_len = pad_bytes_to(output.len(), alignment);

        // Compare with usize literal to avoid type mismatch with N
        if pad_len > 0_usize {
            output.extend(std::iter::repeat_n(PAD_BYTE, pad_len));
        }
    }
    (vec![LanceBuffer::reinterpret_vec(output)], chunks)
}

// search for the next offset index to cut the values into a chunk.
// this function incrementally peek the number of values in a chunk,
// each time multiplies the number of values by 2.
// It returns the offset_idx in `offsets` that belongs to this chunk.
fn search_next_offset_idx<N: OffsetSizeTrait>(offsets: &[N], last_offset_idx: usize) -> usize {
    let mut num_values = 1;
    let mut new_num_values = num_values * 2;
    loop {
        if last_offset_idx + new_num_values >= offsets.len() {
            let existing_bytes = offsets[offsets.len() - 1] - offsets[last_offset_idx];
            // existing bytes plus the new offset size
            let new_size = existing_bytes
                + N::from_usize((offsets.len() - last_offset_idx) * N::get_byte_width()).unwrap();
            if new_size.to_i64().unwrap() <= AIM_MINICHUNK_SIZE {
                // case 1: can fit the rest of all data into a miniblock
                return offsets.len() - 1;
            } else {
                // case 2: can only fit the last tried `num_values` into a miniblock
                return last_offset_idx + num_values;
            }
        }
        let existing_bytes = offsets[last_offset_idx + new_num_values] - offsets[last_offset_idx];
        let new_size =
            existing_bytes + N::from_usize((new_num_values + 1) * N::get_byte_width()).unwrap();
        if new_size.to_i64().unwrap() <= AIM_MINICHUNK_SIZE {
            num_values = new_num_values;
            new_num_values *= 2;
        } else {
            break;
        }
    }
    last_offset_idx + new_num_values
}

impl BinaryMiniBlockEncoder {
    // put binary data into chunks, every chunk is less than or equal to `AIM_MINICHUNK_SIZE`.
    // In each chunk, offsets are put first then followed by binary bytes data, each chunk is padded to 8 bytes.
    // the offsets in the chunk points to the bytes offset in this chunk.
    fn chunk_data(
        &self,
        mut data: VariableWidthBlock,
    ) -> (MiniBlockCompressed, crate::format::pb::ArrayEncoding) {
        match data.bits_per_offset {
            32 => {
                let offsets = data.offsets.borrow_to_typed_slice::<i32>();
                let (buffers, chunks) = chunk_offsets(offsets.as_ref(), &data.data, 4);
                (
                    MiniBlockCompressed {
                        data: buffers,
                        chunks,
                        num_values: data.num_values,
                    },
                    ProtobufUtils::variable(32),
                )
            }
            64 => {
                let offsets = data.offsets.borrow_to_typed_slice::<i64>();
                let (buffers, chunks) = chunk_offsets(offsets.as_ref(), &data.data, 8);
                (
                    MiniBlockCompressed {
                        data: buffers,
                        chunks,
                        num_values: data.num_values,
                    },
                    ProtobufUtils::variable(64),
                )
            }
            _ => panic!("Unsupported bits_per_offset"),
        }
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

#[derive(Debug)]
pub struct BinaryMiniBlockDecompressor {
    bits_per_offset: u8,
}

impl BinaryMiniBlockDecompressor {
    pub fn new(bits_per_offset: u8) -> Self {
        assert!(bits_per_offset == 32 || bits_per_offset == 64);
        Self { bits_per_offset }
    }

    pub fn from_variable(variable: &pb::Variable) -> Self {
        Self {
            bits_per_offset: variable.bits_per_offset as u8,
        }
    }
}

impl MiniBlockDecompressor for BinaryMiniBlockDecompressor {
    // decompress a MiniBlock of binary data, the num_values must be less than or equal
    // to the number of values this MiniBlock has, BinaryMiniBlock doesn't store `the number of values`
    // it has so assertion can not be done here and the caller of `decompress` must ensure `num_values` <= number of values in the chunk.
    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        assert_eq!(data.len(), 1);
        let data = data.into_iter().next().unwrap();

        if self.bits_per_offset == 64 {
            // offset and at least one value
            assert!(data.len() >= 16);

            let offsets: &[u64] = try_cast_slice(&data)
                .expect("casting buffer failed during BinaryMiniBlock decompression");

            let result_offsets = offsets[0..(num_values + 1) as usize]
                .iter()
                .map(|offset| offset - offsets[0])
                .collect::<Vec<u64>>();
            Ok(DataBlock::VariableWidth(VariableWidthBlock {
                data: LanceBuffer::Owned(
                    data[offsets[0] as usize..offsets[num_values as usize] as usize].to_vec(),
                ),
                offsets: LanceBuffer::reinterpret_vec(result_offsets),
                bits_per_offset: 64,
                num_values,
                block_info: BlockInfo::new(),
            }))
        } else {
            // offset and at least one value
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
        testing::{
            check_round_trip_encoding_generated, check_round_trip_encoding_of_data,
            check_round_trip_encoding_random, FnArrayGeneratorProvider, TestCases,
        },
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

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_large_binary(
        #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
    ) {
        let field = Field::new("", DataType::LargeBinary, true);
        check_round_trip_encoding_random(field, version).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_large_utf8() {
        let field = Field::new("", DataType::LargeUtf8, true);
        check_round_trip_encoding_random(field, LanceFileVersion::V2_0).await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_small_strings(
        #[values(STRUCTURAL_ENCODING_MINIBLOCK, STRUCTURAL_ENCODING_FULLZIP)]
        structural_encoding: &str,
    ) {
        let mut field_metadata = HashMap::new();
        field_metadata.insert(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            structural_encoding.into(),
        );
        let field = Field::new("", DataType::Utf8, true).with_metadata(field_metadata);
        check_round_trip_encoding_generated(
            field,
            Box::new(FnArrayGeneratorProvider::new(move || {
                lance_datagen::array::utf8_prefix_plus_counter("user_", /*is_large=*/ false)
            })),
            LanceFileVersion::V2_1,
        )
        .await;
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
