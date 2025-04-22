use std::io::Write;

use arrow::{array::LargeBinaryBuilder, buffer::Buffer};
use delta::delta_encode;
use lance_core::Result;
use lance_encoding::{
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
    encoder::BlockCompressor,
    encodings::physical::bitpack_fastlanes::InlineBitpacking,
};

use super::builder::BLOCK_SIZE;

pub mod delta;

// compress the posting list to multiple blocks of fixed number of elements (BLOCK_SIZE),
// returns a LargeBinaryArray, where each binary is a compressed block (128 row ids + 128 frequencies),
// the first binary is
pub fn compress_posting_list(
    row_ids: &mut [u64],
    frequencies: &mut [u32],
) -> Result<arrow::array::LargeBinaryArray> {
    let mut builder = LargeBinaryBuilder::new();
    let bitpack = InlineBitpacking::new(64);
    let row_id_chunks = row_ids.chunks_exact_mut(BLOCK_SIZE);
    let frequency_chunks = frequencies.chunks_exact_mut(BLOCK_SIZE);
    for (row_id_chunk, freq_chunk) in std::iter::zip(row_id_chunks, frequency_chunks) {
        let (first, delta_encoded) = delta_encode(row_id_chunk);
        let compressed_row_ids = bitpack.compress(DataBlock::FixedWidth(FixedWidthDataBlock {
            data: lance_encoding::buffer::LanceBuffer::Borrowed(Buffer::from_vec(delta_encoded)),
            bits_per_value: 64,
            num_values: BLOCK_SIZE as u64,
            block_info: BlockInfo::new(),
        }))?;
        builder.write(first.to_le_bytes().as_ref())?;
        builder.write(&compressed_row_ids);

        let compressed_freqs = bitpack.compress(DataBlock::FixedWidth(FixedWidthDataBlock {
            data: lance_encoding::buffer::LanceBuffer::Borrowed(Buffer::from_slice_ref(freq_chunk)),
            bits_per_value: 32,
            num_values: BLOCK_SIZE as u64,
            block_info: BlockInfo::new(),
        }))?;
        builder.append_value(compressed_freqs);
    }

    let remainder = row_ids.len() % BLOCK_SIZE;
    if remainder > 0 {
        let row_id_chunk = &mut row_ids[row_ids.len() - remainder..];
        let freq_chunk = &mut frequencies[frequencies.len() - remainder..];
        let (first, delta_encoded) = delta_encode(row_id_chunk);
        let compressed_row_ids = bitpack.compress(DataBlock::FixedWidth(FixedWidthDataBlock {
            data: lance_encoding::buffer::LanceBuffer::Borrowed(Buffer::from_vec(delta_encoded)),
            bits_per_value: 64,
            num_values: remainder as u64,
            block_info: BlockInfo::new(),
        }))?;
        builder.write(first.to_le_bytes().as_ref())?;
        builder.write(&compressed_row_ids);

        let compressed_freqs = bitpack.compress(DataBlock::FixedWidth(FixedWidthDataBlock {
            data: lance_encoding::buffer::LanceBuffer::Borrowed(Buffer::from_slice_ref(freq_chunk)),
            bits_per_value: 32,
            num_values: remainder as u64,
            block_info: BlockInfo::new(),
        }))?;
        builder.append_value(compressed_freqs);
    }
    Ok(builder.finish())
}

/// decompress the posting list from a LargeBinaryArray
/// returns a vector of (row_id, frequency) tuples
pub fn decompress_posting_list(
    posting_list: &arrow::array::LargeBinaryArray,
) -> Result<Vec<(u64, u32)>> {
    let mut result = Vec::with_capacity(posting_list.len() * BLOCK_SIZE);
    let bitpack = InlineBitpacking::new(64);
    for i in 0..posting_list.len() {
        let block = posting_list.value(i);
        let (first, compressed_row_ids) = block.split_at(std::mem::size_of::<u64>());
        let first = u64::from_le_bytes(first.try_into().unwrap());
        let row_ids = bitpack.decompress(compressed_row_ids)?;
        let frequencies = bitpack.decompress(posting_list.value(i + 1))?;
        for j in 0..BLOCK_SIZE {
            result.push((first + row_ids[j], frequencies[j]));
        }
    }
    Ok(result)
}
