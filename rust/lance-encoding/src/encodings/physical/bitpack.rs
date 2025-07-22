// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bitpacking encodings
//!
//! These encodings look for unused higher order bits and discard them.  For example, if we
//! have a u32 array and all values are between 0 and 5000 then we only need 12 bits to store
//! each value.  The encoding will discard the upper 20 bits and only store 12 bits.
//!
//! This is a simple encoding that works well for data that has a small range.
//!
//! In order to decode the values we need to know the bit width of the values.  This can be stored
//! inline with the data (miniblock) or in the encoding description, out of line (full zip).
//!
//! The encoding is transparent because the output has a fixed width (just like the input) and
//! we can easily jump to the correct value.

use arrow::datatypes::UInt64Type;
use arrow_array::{Array, PrimitiveArray};
use arrow_buffer::ArrowNativeType;
use byteorder::{ByteOrder, LittleEndian};
use snafu::location;

use lance_core::{Error, Result};

use crate::buffer::LanceBuffer;
use crate::compression::{
    BlockCompressor, BlockDecompressor, FixedPerValueDecompressor, MiniBlockDecompressor,
};
use crate::compression_algo::fastlanes::BitPacking;
use crate::data::BlockInfo;
use crate::data::{DataBlock, FixedWidthDataBlock};
use crate::encodings::logical::primitive::fullzip::{PerValueCompressor, PerValueDataBlock};
use crate::encodings::logical::primitive::miniblock::{
    MiniBlockChunk, MiniBlockCompressed, MiniBlockCompressor,
};
use crate::format::{pb, ProtobufUtils};
use crate::statistics::{GetStat, Stat};
use bytemuck::{cast_slice, AnyBitPattern};

const LOG_ELEMS_PER_CHUNK: u8 = 10;
const ELEMS_PER_CHUNK: u64 = 1 << LOG_ELEMS_PER_CHUNK;

#[derive(Debug, Default)]
pub struct InlineBitpacking {
    uncompressed_bit_width: u64,
}

impl InlineBitpacking {
    pub fn new(uncompressed_bit_width: u64) -> Self {
        Self {
            uncompressed_bit_width,
        }
    }

    pub fn from_description(description: &pb::InlineBitpacking) -> Self {
        Self {
            uncompressed_bit_width: description.uncompressed_bits_per_value,
        }
    }

    pub fn min_size_bytes(bit_width: u64) -> u64 {
        (ELEMS_PER_CHUNK * bit_width).div_ceil(8)
    }

    /// Bitpacks a FixedWidthDataBlock into compressed chunks of 1024 values
    ///
    /// Each chunk can have a different bit width
    ///
    /// Each chunk has the compressed bit width stored inline in the chunk itself.
    fn bitpack_chunked<T: ArrowNativeType + BitPacking>(
        mut data: FixedWidthDataBlock,
    ) -> MiniBlockCompressed {
        let data_buffer = data.data.borrow_to_typed_slice::<T>();
        let data_buffer = data_buffer.as_ref();

        let bit_widths = data.expect_stat(Stat::BitWidth);
        let bit_widths_array = bit_widths
            .as_any()
            .downcast_ref::<PrimitiveArray<UInt64Type>>()
            .unwrap();

        let (packed_chunk_sizes, total_size) = bit_widths_array
            .values()
            .iter()
            .map(|&bit_width| {
                let chunk_size = ((1024 * bit_width) / data.bits_per_value) as usize;
                (chunk_size, chunk_size + 1)
            })
            .fold(
                (Vec::with_capacity(bit_widths_array.len()), 0),
                |(mut sizes, total), (size, inc)| {
                    sizes.push(size);
                    (sizes, total + inc)
                },
            );

        let mut output: Vec<T> = Vec::with_capacity(total_size);
        let mut chunks = Vec::with_capacity(bit_widths_array.len());

        for (i, packed_chunk_size) in packed_chunk_sizes
            .iter()
            .enumerate()
            .take(bit_widths_array.len() - 1)
        {
            let start_elem = i * ELEMS_PER_CHUNK as usize;
            let bit_width = bit_widths_array.value(i) as usize;
            output.push(T::from_usize(bit_width).unwrap());
            let output_len = output.len();
            unsafe {
                output.set_len(output_len + *packed_chunk_size);
                BitPacking::unchecked_pack(
                    bit_width,
                    &data_buffer[start_elem..][..ELEMS_PER_CHUNK as usize],
                    &mut output[output_len..][..*packed_chunk_size],
                );
            }
            chunks.push(MiniBlockChunk {
                buffer_sizes: vec![((1 + *packed_chunk_size) * std::mem::size_of::<T>()) as u16],
                log_num_values: LOG_ELEMS_PER_CHUNK,
            });
        }

        // Handle the last chunk
        let last_chunk_elem_num = if data.num_values % ELEMS_PER_CHUNK == 0 {
            1024
        } else {
            data.num_values % ELEMS_PER_CHUNK
        };
        let mut last_chunk: Vec<T> = vec![T::from_usize(0).unwrap(); ELEMS_PER_CHUNK as usize];
        last_chunk[..last_chunk_elem_num as usize].clone_from_slice(
            &data_buffer[data.num_values as usize - last_chunk_elem_num as usize..],
        );
        let bit_width = bit_widths_array.value(bit_widths_array.len() - 1) as usize;
        output.push(T::from_usize(bit_width).unwrap());
        let output_len = output.len();
        unsafe {
            output.set_len(output_len + packed_chunk_sizes[bit_widths_array.len() - 1]);
            BitPacking::unchecked_pack(
                bit_width,
                &last_chunk,
                &mut output[output_len..][..packed_chunk_sizes[bit_widths_array.len() - 1]],
            );
        }
        chunks.push(MiniBlockChunk {
            buffer_sizes: vec![
                ((1 + packed_chunk_sizes[bit_widths_array.len() - 1]) * std::mem::size_of::<T>())
                    as u16,
            ],
            log_num_values: 0,
        });

        MiniBlockCompressed {
            data: vec![LanceBuffer::reinterpret_vec(output)],
            chunks,
            num_values: data.num_values,
        }
    }

    fn chunk_data(
        &self,
        data: FixedWidthDataBlock,
    ) -> (MiniBlockCompressed, crate::format::pb::ArrayEncoding) {
        assert!(data.bits_per_value % 8 == 0);
        assert_eq!(data.bits_per_value, self.uncompressed_bit_width);
        let bits_per_value = data.bits_per_value;
        let compressed = match bits_per_value {
            8 => Self::bitpack_chunked::<u8>(data),
            16 => Self::bitpack_chunked::<u16>(data),
            32 => Self::bitpack_chunked::<u32>(data),
            64 => Self::bitpack_chunked::<u64>(data),
            _ => unreachable!(),
        };
        (compressed, ProtobufUtils::inline_bitpacking(bits_per_value))
    }

    fn unchunk<T: ArrowNativeType + BitPacking + AnyBitPattern>(
        data: LanceBuffer,
        num_values: u64,
    ) -> Result<DataBlock> {
        assert!(data.len() >= 8);
        assert!(num_values <= ELEMS_PER_CHUNK);

        // This macro decompresses a chunk(1024 values) of bitpacked values.
        let uncompressed_bit_width = std::mem::size_of::<T>() * 8;
        let mut decompressed = vec![T::from_usize(0).unwrap(); ELEMS_PER_CHUNK as usize];

        // Copy for memory alignment
        let chunk_in_u8: Vec<u8> = data.to_vec();
        let bit_width_bytes = &chunk_in_u8[..std::mem::size_of::<T>()];
        let bit_width_value = LittleEndian::read_uint(bit_width_bytes, std::mem::size_of::<T>());
        let chunk = cast_slice(&chunk_in_u8[std::mem::size_of::<T>()..]);

        // The bit-packed chunk should have number of bytes (bit_width_value * ELEMS_PER_CHUNK / 8)
        assert!(std::mem::size_of_val(chunk) == (bit_width_value * ELEMS_PER_CHUNK) as usize / 8);

        unsafe {
            BitPacking::unchecked_unpack(bit_width_value as usize, chunk, &mut decompressed);
        }

        decompressed.truncate(num_values as usize);
        Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::reinterpret_vec(decompressed),
            bits_per_value: uncompressed_bit_width as u64,
            num_values,
            block_info: BlockInfo::new(),
        }))
    }
}

impl MiniBlockCompressor for InlineBitpacking {
    fn compress(
        &self,
        chunk: DataBlock,
    ) -> Result<(MiniBlockCompressed, crate::format::pb::ArrayEncoding)> {
        match chunk {
            DataBlock::FixedWidth(fixed_width) => Ok(self.chunk_data(fixed_width)),
            _ => Err(Error::InvalidInput {
                source: format!(
                    "Cannot compress a data block of type {} with BitpackMiniBlockEncoder",
                    chunk.name()
                )
                .into(),
                location: location!(),
            }),
        }
    }
}

impl BlockCompressor for InlineBitpacking {
    fn compress(&self, data: DataBlock) -> Result<LanceBuffer> {
        let fixed_width = data.as_fixed_width().unwrap();
        let (chunked, _) = self.chunk_data(fixed_width);
        Ok(chunked.data.into_iter().next().unwrap())
    }
}

impl MiniBlockDecompressor for InlineBitpacking {
    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        assert_eq!(data.len(), 1);
        let data = data.into_iter().next().unwrap();
        match self.uncompressed_bit_width {
            8 => Self::unchunk::<u8>(data, num_values),
            16 => Self::unchunk::<u16>(data, num_values),
            32 => Self::unchunk::<u32>(data, num_values),
            64 => Self::unchunk::<u64>(data, num_values),
            _ => unimplemented!("Bitpacking word size must be 8, 16, 32, or 64"),
        }
    }
}

impl BlockDecompressor for InlineBitpacking {
    fn decompress(&self, data: LanceBuffer, num_values: u64) -> Result<DataBlock> {
        match self.uncompressed_bit_width {
            8 => Self::unchunk::<u8>(data, num_values),
            16 => Self::unchunk::<u16>(data, num_values),
            32 => Self::unchunk::<u32>(data, num_values),
            64 => Self::unchunk::<u64>(data, num_values),
            _ => unimplemented!("Bitpacking word size must be 8, 16, 32, or 64"),
        }
    }
}

/// Bitpacks a FixedWidthDataBlock with a given bit width
///
/// This function is simpler as it does not do any chunking, but slightly less efficient.
/// The compressed bits per value is constant across the entire buffer.
///
/// Note: even though we are not strictly "chunking" we are still operating on chunks of
/// 1024 values because that's what the bitpacking primitives expect.  They just don't
/// have a unique bit width for each chunk.
fn bitpack_out_of_line<T: ArrowNativeType + BitPacking>(
    mut data: FixedWidthDataBlock,
    bit_width: usize,
) -> LanceBuffer {
    let data_buffer = data.data.borrow_to_typed_slice::<T>();
    let data_buffer = data_buffer.as_ref();

    let num_chunks = data_buffer.len().div_ceil(ELEMS_PER_CHUNK as usize);
    let last_chunk_is_runt = data_buffer.len() % ELEMS_PER_CHUNK as usize != 0;
    let words_per_chunk =
        (ELEMS_PER_CHUNK as usize * bit_width).div_ceil(data.bits_per_value as usize);
    #[allow(clippy::uninit_vec)]
    let mut output: Vec<T> = Vec::with_capacity(num_chunks * words_per_chunk);
    #[allow(clippy::uninit_vec)]
    unsafe {
        output.set_len(num_chunks * words_per_chunk);
    }

    let num_whole_chunks = if last_chunk_is_runt {
        num_chunks - 1
    } else {
        num_chunks
    };

    // Simple case for complete chunks
    for i in 0..num_whole_chunks {
        let input_start = i * ELEMS_PER_CHUNK as usize;
        let input_end = input_start + ELEMS_PER_CHUNK as usize;
        let output_start = i * words_per_chunk;
        let output_end = output_start + words_per_chunk;
        unsafe {
            BitPacking::unchecked_pack(
                bit_width,
                &data_buffer[input_start..input_end],
                &mut output[output_start..output_end],
            );
        }
    }

    if !last_chunk_is_runt {
        return LanceBuffer::reinterpret_vec(output);
    }

    // If we get here then the last chunk needs to be padded with zeros
    let remaining_items = data_buffer.len() % ELEMS_PER_CHUNK as usize;
    let last_chunk_start = num_whole_chunks * ELEMS_PER_CHUNK as usize;

    let mut last_chunk: Vec<T> = vec![T::from_usize(0).unwrap(); ELEMS_PER_CHUNK as usize];
    last_chunk[..remaining_items].clone_from_slice(&data_buffer[last_chunk_start..]);
    let output_start = num_whole_chunks * words_per_chunk;
    unsafe {
        BitPacking::unchecked_pack(bit_width, &last_chunk, &mut output[output_start..]);
    }

    LanceBuffer::reinterpret_vec(output)
}

/// Unpacks a FixedWidthDataBlock that has been bitpacked with a constant bit width
fn unpack_out_of_line<T: ArrowNativeType + BitPacking>(
    mut data: FixedWidthDataBlock,
    num_values: usize,
    bits_per_value: usize,
) -> FixedWidthDataBlock {
    let words_per_chunk =
        (ELEMS_PER_CHUNK as usize * bits_per_value).div_ceil(data.bits_per_value as usize);
    let compressed_words = data.data.borrow_to_typed_slice::<T>();

    let num_chunks = data.num_values as usize / words_per_chunk;
    debug_assert_eq!(data.num_values as usize % words_per_chunk, 0);

    // This is slightly larger than num_values because the last chunk has some padding, we will truncate at the end
    #[allow(clippy::uninit_vec)]
    let mut decompressed = Vec::with_capacity(num_chunks * ELEMS_PER_CHUNK as usize);
    #[allow(clippy::uninit_vec)]
    unsafe {
        decompressed.set_len(num_chunks * ELEMS_PER_CHUNK as usize);
    }

    for chunk_idx in 0..num_chunks {
        let input_start = chunk_idx * words_per_chunk;
        let input_end = input_start + words_per_chunk;
        let output_start = chunk_idx * ELEMS_PER_CHUNK as usize;
        let output_end = output_start + ELEMS_PER_CHUNK as usize;
        unsafe {
            BitPacking::unchecked_unpack(
                bits_per_value,
                &compressed_words[input_start..input_end],
                &mut decompressed[output_start..output_end],
            );
        }
    }

    decompressed.truncate(num_values);

    FixedWidthDataBlock {
        data: LanceBuffer::reinterpret_vec(decompressed),
        bits_per_value: data.bits_per_value,
        num_values: num_values as u64,
        block_info: BlockInfo::new(),
    }
}

/// A transparent compressor that bit packs data
///
/// In order for the encoding to be transparent we must have a fixed bit width
/// across the entire array.  Chunking within the buffer is not supported.  This
/// means that we will be slightly less efficient than something like the mini-block
/// approach.
///
/// WARNING: DO NOT USE YET.
///
/// This was an interesting experiment but it can't be used as a per-value compressor
/// at the moment.  The resulting data IS transparent but it's not quite so simple.  We
/// compress in blocks of 1024 and each block has a fixed size but also has some padding.
///
/// In other words, if we try the simple math to access the item at index `i` we will be
/// out of luck because `bits_per_value * i` is not the location.  What we need is something
/// like:
///
/// ```ignore
/// let chunk_idx = i / 1024;
/// let chunk_offset = i % 1024;
/// bits_per_chunk * chunk_idx + bits_per_value * chunk_offset
/// ```
///
/// However, this logic isn't expressible with the per-value traits we have today.  We can
/// enhance these traits should we need to support it at some point in the future.
#[derive(Debug)]
pub struct OutOfLineBitpacking {
    compressed_bit_width: usize,
}

impl PerValueCompressor for OutOfLineBitpacking {
    fn compress(&self, data: DataBlock) -> Result<(PerValueDataBlock, pb::ArrayEncoding)> {
        let fixed_width = data.as_fixed_width().unwrap();
        let num_values = fixed_width.num_values;
        let word_size = fixed_width.bits_per_value;
        let compressed = match word_size {
            8 => bitpack_out_of_line::<u8>(fixed_width, self.compressed_bit_width),
            16 => bitpack_out_of_line::<u16>(fixed_width, self.compressed_bit_width),
            32 => bitpack_out_of_line::<u32>(fixed_width, self.compressed_bit_width),
            64 => bitpack_out_of_line::<u64>(fixed_width, self.compressed_bit_width),
            _ => panic!("Bitpacking word size must be 8,16,32,64"),
        };
        let compressed = FixedWidthDataBlock {
            data: compressed,
            bits_per_value: self.compressed_bit_width as u64,
            num_values,
            block_info: BlockInfo::new(),
        };
        let encoding =
            ProtobufUtils::out_of_line_bitpacking(word_size, self.compressed_bit_width as u64);
        Ok((PerValueDataBlock::Fixed(compressed), encoding))
    }
}

impl FixedPerValueDecompressor for OutOfLineBitpacking {
    fn decompress(&self, data: FixedWidthDataBlock, num_values: u64) -> Result<DataBlock> {
        let unpacked = match data.bits_per_value {
            8 => unpack_out_of_line::<u8>(data, num_values as usize, self.compressed_bit_width),
            16 => unpack_out_of_line::<u16>(data, num_values as usize, self.compressed_bit_width),
            32 => unpack_out_of_line::<u32>(data, num_values as usize, self.compressed_bit_width),
            64 => unpack_out_of_line::<u64>(data, num_values as usize, self.compressed_bit_width),
            _ => panic!("Bitpacking word size must be 8,16,32,64"),
        };
        Ok(DataBlock::FixedWidth(unpacked))
    }

    fn bits_per_value(&self) -> u64 {
        self.compressed_bit_width as u64
    }
}

#[cfg(test)]
mod test {
    use std::{collections::HashMap, sync::Arc};

    use arrow_array::{Int64Array, Int8Array};

    use arrow_schema::DataType;

    use arrow_array::Array;

    use crate::{
        testing::{check_round_trip_encoding_of_data, TestCases},
        version::LanceFileVersion,
    };

    #[test_log::test(tokio::test)]
    async fn test_miniblock_bitpack() {
        let test_cases = TestCases::default().with_file_version(LanceFileVersion::V2_1);

        let mut metadata = HashMap::new();
        metadata.insert(
            "lance-encoding:compression".to_string(),
            "bitpacking".to_string(),
        );

        let arrays = vec![
            Arc::new(Int8Array::from(vec![100; 1024])) as Arc<dyn Array>,
            Arc::new(Int8Array::from(vec![1; 1024])) as Arc<dyn Array>,
            Arc::new(Int8Array::from(vec![16; 1024])) as Arc<dyn Array>,
            Arc::new(Int8Array::from(vec![-1; 1024])) as Arc<dyn Array>,
            Arc::new(Int8Array::from(vec![5; 1])) as Arc<dyn Array>,
        ];
        check_round_trip_encoding_of_data(arrays, &test_cases, metadata).await;

        for data_type in [DataType::Int16, DataType::Int32, DataType::Int64] {
            let int64_arrays = vec![
                Int64Array::from(vec![3; 1024]),
                Int64Array::from(vec![8; 1024]),
                Int64Array::from(vec![16; 1024]),
                Int64Array::from(vec![100; 1024]),
                Int64Array::from(vec![512; 1024]),
                Int64Array::from(vec![1000; 1024]),
                Int64Array::from(vec![2000; 1024]),
                Int64Array::from(vec![-1; 10]),
            ];

            let mut arrays = vec![];
            for int64_array in int64_arrays {
                arrays.push(arrow_cast::cast(&int64_array, &data_type).unwrap());
            }

            let mut metadata = HashMap::new();
            metadata.insert(
                "lance-encoding:compression".to_string(),
                "bitpacking".to_string(),
            );
            check_round_trip_encoding_of_data(arrays, &test_cases, metadata).await;
        }
    }
}
