// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::io::Write;

use super::builder::BLOCK_SIZE;
use arrow::array::{AsArray, LargeBinaryBuilder};
use arrow::array::{ListBuilder, UInt32Builder};
use arrow_array::{Array, ListArray};
use bitpacking::{BitPacker, BitPacker4x};
use lance_core::Result;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub(crate) enum Compression {
    // delta-encode + bitpack for doc ids
    // bitpack for frequencies
    // and plain for the remainder
    Bitpack,
}

#[derive(Debug, PartialEq, Clone)]
pub struct CompressedPostingListHeader {
    pub(crate) compression: Compression,
    pub num_docs: u32,
    pub max_score: f32,
}

impl CompressedPostingListHeader {
    pub(crate) fn new(compression: Compression, num_docs: u32, max_score: f32) -> Self {
        Self {
            compression,
            num_docs,
            max_score,
        }
    }

    pub fn num_bytes() -> usize {
        std::mem::size_of::<Compression>() + std::mem::size_of::<u32>() + std::mem::size_of::<f32>()
    }

    pub fn write(&self, builder: &mut LargeBinaryBuilder) -> Result<()> {
        builder.write(&[self.compression as u8])?;
        builder.write(&self.num_docs.to_le_bytes())?;
        builder.write(&self.max_score.to_le_bytes())?;
        builder.append_value("");
        Ok(())
    }
}

// compress the posting list to multiple blocks of fixed number of elements (BLOCK_SIZE),
// returns a LargeBinaryArray, where each binary is a compressed block (128 row ids + 128 frequencies),
// the first binary is
pub fn compress_posting_list(
    doc_ids: &[u32],
    frequencies: &[u32],
) -> Result<arrow::array::LargeBinaryArray> {
    let mut builder = LargeBinaryBuilder::with_capacity(
        doc_ids.len().div_ceil(BLOCK_SIZE) + 1,
        doc_ids.len() * 4,
    );
    let doc_id_chunks = doc_ids.chunks_exact(BLOCK_SIZE);
    let frequency_chunks = frequencies.chunks_exact(BLOCK_SIZE);
    let mut buffer = [0u8; BLOCK_SIZE * 4 + 5];
    for (doc_id_chunk, freq_chunk) in std::iter::zip(doc_id_chunks, frequency_chunks) {
        // delta encoding + bitpacking for doc ids
        compress_sorted_block(doc_id_chunk, &mut buffer, &mut builder)?;
        // bitpacking for frequencies
        compress_block(freq_chunk, &mut buffer, &mut builder)?;
        builder.append_value("");
    }

    // we don't compress the last block if it is not full
    let length = doc_ids.len();
    let remainder = length % BLOCK_SIZE;
    if remainder > 0 {
        compress_remainder(&doc_ids[length - remainder..], &mut builder)?;
        compress_remainder(&frequencies[length - remainder..], &mut builder)?;
        builder.append_value("");
    }
    Ok(builder.finish())
}

fn compress_sorted_block(
    data: &[u32],
    buffer: &mut [u8],
    builder: &mut LargeBinaryBuilder,
) -> Result<()> {
    let compressor = BitPacker4x::new();
    let num_bits = compressor.num_bits_sorted(data[0], data);
    let num_bytes = compressor.compress_sorted(data[0], data, buffer, num_bits);
    builder.write(data[0].to_le_bytes().as_ref())?;
    builder.write(&[num_bits])?;
    builder.write(&buffer[..num_bytes])?;
    Ok(())
}

fn compress_block(data: &[u32], buffer: &mut [u8], builder: &mut LargeBinaryBuilder) -> Result<()> {
    let compressor = BitPacker4x::new();
    let num_bits = compressor.num_bits(data);
    let num_bytes = compressor.compress(data, buffer, num_bits);
    builder.write(&[num_bits])?;
    builder.write(&buffer[..num_bytes])?;
    Ok(())
}

fn compress_remainder(data: &[u32], builder: &mut LargeBinaryBuilder) -> Result<()> {
    for value in data.iter() {
        builder.write(value.to_le_bytes().as_ref())?;
    }
    Ok(())
}

pub fn compress_positions(positions: &[u32]) -> Result<arrow::array::LargeBinaryArray> {
    let mut builder = LargeBinaryBuilder::with_capacity(
        positions.len().div_ceil(BLOCK_SIZE),
        positions.len() * 4,
    );
    // record the number of positions in the first binary
    let num_positions = positions.len() as u32;
    builder.append_value(&num_positions.to_le_bytes().as_ref());

    let position_chunks = positions.chunks_exact(BLOCK_SIZE);
    let mut buffer = [0u8; BLOCK_SIZE * 4 + 5];
    for position_chunk in position_chunks {
        // delta encoding + bitpacking for positions
        compress_sorted_block(position_chunk, &mut buffer, &mut builder)?;
        builder.append_value("");
    }

    // we don't compress the last block if it is not full
    let length = positions.len();
    let remainder = length % BLOCK_SIZE;
    if remainder > 0 {
        compress_remainder(&positions[length - remainder..], &mut builder)?;
        builder.append_value("");
    }
    Ok(builder.finish())
}

/// decompress the posting list from a LargeBinaryArray
/// returns a vector of (row_id, frequency) tuples
pub fn decompress_posting_list(
    num_docs: u32,
    posting_list: &arrow::array::LargeBinaryArray,
) -> Result<(Vec<u32>, Vec<u32>)> {
    let mut doc_ids: Vec<u32> = Vec::with_capacity(num_docs as usize);
    let mut frequencies: Vec<u32> = Vec::with_capacity(num_docs as usize);

    let mut buffer = [0u32; BLOCK_SIZE];
    let bitpacking_blocks = num_docs as usize / BLOCK_SIZE;
    for compressed in posting_list.iter().take(bitpacking_blocks) {
        let compressed = compressed.unwrap();
        decompress_posting_block(compressed, &mut buffer, &mut doc_ids, &mut frequencies);
    }

    let remainder = num_docs as usize % BLOCK_SIZE;
    if remainder > 0 {
        let compressed = posting_list.value(bitpacking_blocks);
        decompress_remainder(compressed, remainder, &mut doc_ids);
        decompress_remainder(&compressed[remainder * 4..], remainder, &mut frequencies);
    }

    Ok((doc_ids, frequencies))
}

pub fn decompress_positions(compressed: &arrow::array::LargeBinaryArray) -> Result<Vec<u32>> {
    let num_positions = read_num_positions(compressed);
    let mut positions: Vec<u32> = Vec::with_capacity(num_positions as usize);

    let mut buffer = [0u32; BLOCK_SIZE];
    let num_blocks = num_positions as usize / BLOCK_SIZE;
    for compressed in compressed.iter().skip(1).take(num_blocks) {
        let compressed = compressed.unwrap();
        decompress_sorted_block(compressed, &mut buffer, &mut positions);
    }

    let remainder = num_positions as usize % BLOCK_SIZE;
    if remainder > 0 {
        let compressed = compressed.value(num_blocks + 1);
        decompress_remainder(compressed, remainder, &mut positions);
    }

    Ok(positions)
}

// decompress the positions list from a ListArray of binary
// to a ListArray of u32
pub fn decompress_positions_list(compressed: &ListArray) -> Result<ListArray> {
    let mut builder = ListBuilder::with_capacity(UInt32Builder::new(), compressed.len());
    for i in 0..compressed.len() {
        let compressed = compressed.value(i);
        let compressed = compressed.as_binary::<i64>();
        let positions = decompress_positions(compressed)?;
        builder.values().append_slice(&positions);
        builder.append(true);
    }
    Ok(builder.finish())
}

pub fn read_num_positions(posting_list: &arrow::array::LargeBinaryArray) -> u32 {
    u32::from_le_bytes(posting_list.values().as_ref()[..4].try_into().unwrap())
}

pub fn decompress_posting_block(
    block: &[u8],
    buffer: &mut [u32; BLOCK_SIZE],
    doc_ids: &mut Vec<u32>,
    frequencies: &mut Vec<u32>,
) {
    let num_bytes = decompress_sorted_block(block, buffer, doc_ids);
    decompress_block(&block[num_bytes..], buffer, frequencies);
}

pub fn decompress_sorted_block(
    block: &[u8],
    buffer: &mut [u32; BLOCK_SIZE],
    res: &mut Vec<u32>,
) -> usize {
    let compressor = BitPacker4x::new();
    let initial = u32::from_le_bytes(block[0..4].try_into().unwrap());
    let num_bits = block[4];
    let num_bytes = compressor.decompress_sorted(initial, &block[5..], buffer, num_bits);
    res.extend_from_slice(&buffer[..]);
    5 + num_bytes
}

fn decompress_block(block: &[u8], buffer: &mut [u32; BLOCK_SIZE], res: &mut Vec<u32>) {
    let compressor = BitPacker4x::new();
    let num_bits = block[0];
    compressor.decompress(&block[1..], buffer, num_bits);
    res.extend_from_slice(&buffer[..]);
}

pub fn decompress_remainder(compressed: &[u8], n: usize, dest: &mut Vec<u32>) {
    for bytes in compressed.chunks_exact(4).take(n) {
        let data = u32::from_le_bytes(bytes.try_into().unwrap());
        dest.push(data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use itertools::Itertools;
    use rand::Rng;

    #[test]
    fn test_compress_posting_list() -> Result<()> {
        let num_rows: usize = BLOCK_SIZE * 1024 - 7;
        let mut rng = rand::thread_rng();
        let doc_ids: Vec<u32> = (0..num_rows).map(|_| rng.gen()).sorted_unstable().collect();
        let frequencies: Vec<u32> = (0..num_rows).map(|_| rng.gen_range(1..=u32::MAX)).collect();
        let posting_list = compress_posting_list(&doc_ids, &frequencies)?;
        assert_eq!(posting_list.len(), num_rows.div_ceil(BLOCK_SIZE));
        let compressed_size =
            posting_list.value_data().len() + posting_list.value_offsets().len() * 8;
        let original_size = 2 * num_rows * 4;
        assert!(
            compressed_size < original_size,
            "compressed size {} should be less than original size {}",
            compressed_size,
            original_size
        );

        let (decompressed_doc_ids, decompressed_frequencies) =
            decompress_posting_list(num_rows as u32, &posting_list)?;
        assert_eq!(doc_ids, decompressed_doc_ids);
        assert_eq!(frequencies, decompressed_frequencies);
        Ok(())
    }

    #[test]
    fn test_compress_positions() -> Result<()> {
        let num_positions: usize = BLOCK_SIZE * 2 - 7;
        let mut rng = rand::thread_rng();
        let positions: Vec<u32> = (0..num_positions)
            .map(|_| rng.gen())
            .sorted_unstable()
            .collect();
        let compressed = compress_positions(&positions)?;
        assert_eq!(compressed.len(), num_positions.div_ceil(BLOCK_SIZE) + 1);
        let compressed_size = compressed.value_data().len() + compressed.value_offsets().len() * 8;
        let original_size = 2 * num_positions * 4;
        assert!(
            compressed_size < original_size,
            "compressed size {} should be less than original size {}",
            compressed_size,
            original_size
        );

        let decompressed_positions = decompress_positions(&compressed)?;
        assert_eq!(positions, decompressed_positions);
        assert_eq!(positions.len(), num_positions);
        Ok(())
    }
}
