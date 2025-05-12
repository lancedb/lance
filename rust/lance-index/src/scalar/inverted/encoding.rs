// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::io::Write;

use super::builder::BLOCK_SIZE;
use arrow::array::{AsArray, LargeBinaryBuilder};
use arrow::array::{ListBuilder, UInt32Builder};
use arrow_array::{Array, ListArray};
use bitpacking::{BitPacker, BitPacker4x};
use lance_core::Result;
use tracing::instrument;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
#[allow(dead_code)]
pub enum Compression {
    // each block is:
    // - 4 bytes for the first doc id
    // - 1 byte for the number of bits used for the doc ids
    // - delta encoded & bitpacked binary for the doc ids
    // - 1 byte for the number of bits used for the frequencies
    // - bitpacked binary for the frequencies
    // and plain for the remainder
    Bitpack,

    // each block is:
    // - 4 bytes for the first doc id
    // - 4 bytes for max score of the block
    // - 1 byte for the number of bits used for the doc ids
    // - delta encoded & bitpacked binary for the doc ids
    // - 1 byte for the number of bits used for the frequencies
    // - bitpacked binary for the frequencies
    // and plain for the remainder
    BlockMaxBitpack,
}

// compress the posting list to multiple blocks of fixed number of elements (BLOCK_SIZE),
// returns a LargeBinaryArray, where each binary is a compressed block (128 row ids + 128 frequencies)
#[allow(dead_code)]
pub fn compress_posting_list(
    doc_ids: &[u32],
    frequencies: &[u32],
) -> Result<arrow::array::LargeBinaryArray> {
    let length = doc_ids.len();
    if length < BLOCK_SIZE {
        // directly do remainder compression to avoid overhead of creating buffer
        let mut builder = LargeBinaryBuilder::with_capacity(1, length * 4 * 2);
        compress_remainder(doc_ids, &mut builder)?;
        compress_remainder(frequencies, &mut builder)?;
        builder.append_value("");
        return Ok(builder.finish());
    }

    let mut builder =
        LargeBinaryBuilder::with_capacity(doc_ids.len().div_ceil(BLOCK_SIZE), length * 3);
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
    let remainder = length % BLOCK_SIZE;
    if remainder > 0 {
        compress_remainder(&doc_ids[length - remainder..], &mut builder)?;
        compress_remainder(&frequencies[length - remainder..], &mut builder)?;
        builder.append_value("");
    }
    Ok(builder.finish())
}

pub fn compress_posting_slices<'a>(
    length: usize,
    doc_ids: impl Iterator<Item = &'a [u32]>,
    frequencies: impl Iterator<Item = &'a [u32]>,
) -> Result<arrow::array::LargeBinaryArray> {
    if length < BLOCK_SIZE {
        // directly do remainder compression to avoid overhead of creating buffer
        let mut builder = LargeBinaryBuilder::with_capacity(1, length * 4 * 2);
        compress_remainder(
            doc_ids.flatten().copied().collect::<Vec<_>>().as_slice(),
            &mut builder,
        )?;
        compress_remainder(
            frequencies
                .flatten()
                .copied()
                .collect::<Vec<_>>()
                .as_slice(),
            &mut builder,
        )?;
        builder.append_value("");
        return Ok(builder.finish());
    }

    let mut builder = LargeBinaryBuilder::with_capacity(length.div_ceil(BLOCK_SIZE), length * 3);
    let mut buffer = [0u8; BLOCK_SIZE * 4 + 5];
    let mut doc_id_buffer = Vec::with_capacity(BLOCK_SIZE);
    let mut freq_buffer = Vec::with_capacity(BLOCK_SIZE);
    for (doc_id_chunk, freq_chunk) in std::iter::zip(doc_ids, frequencies) {
        let (doc_id_chunk, freq_chunk) =
            if doc_id_buffer.is_empty() && doc_id_chunk.len() == BLOCK_SIZE {
                (doc_id_chunk, freq_chunk) // no need to copy
            } else {
                doc_id_buffer.extend_from_slice(doc_id_chunk);
                freq_buffer.extend_from_slice(freq_chunk);
                (doc_id_buffer.as_slice(), freq_buffer.as_slice())
            };

        // this is a hack, that the ExpLinkedList would always return a slice of BLOCK_SIZE
        // after consuming the first blocks that cap is less than BLOCK_SIZE
        if doc_id_chunk.len() < BLOCK_SIZE {
            continue;
        }
        assert_eq!(doc_id_chunk.len(), BLOCK_SIZE);

        // delta encoding + bitpacking for doc ids
        compress_sorted_block(doc_id_chunk, &mut buffer, &mut builder)?;
        // bitpacking for frequencies
        compress_block(freq_chunk, &mut buffer, &mut builder)?;
        builder.append_value("");
        doc_id_buffer.clear();
        freq_buffer.clear();
    }

    // we don't compress the last block if it is not full
    if !doc_id_buffer.is_empty() {
        compress_remainder(&doc_id_buffer, &mut builder)?;
        compress_remainder(&freq_buffer, &mut builder)?;
        builder.append_value("");
    }
    Ok(builder.finish())
}

#[inline]
fn compress_sorted_block(
    data: &[u32],
    buffer: &mut [u8],
    builder: &mut LargeBinaryBuilder,
) -> Result<()> {
    let compressor = BitPacker4x::new();
    let num_bits = compressor.num_bits_sorted(data[0], data);
    let num_bytes = compressor.compress_sorted(data[0], data, buffer, num_bits);
    let _ = builder.write(data[0].to_le_bytes().as_ref())?;
    let _ = builder.write(&[num_bits])?;
    let _ = builder.write(&buffer[..num_bytes])?;
    Ok(())
}

#[inline]
fn compress_block(data: &[u32], buffer: &mut [u8], builder: &mut LargeBinaryBuilder) -> Result<()> {
    let compressor = BitPacker4x::new();
    let num_bits = compressor.num_bits(data);
    let num_bytes = compressor.compress(data, buffer, num_bits);
    let _ = builder.write(&[num_bits])?;
    let _ = builder.write(&buffer[..num_bytes])?;
    Ok(())
}

#[inline]
fn compress_remainder(data: &[u32], builder: &mut LargeBinaryBuilder) -> Result<()> {
    for value in data.iter() {
        builder.write_all(value.to_le_bytes().as_ref())?;
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
    builder.append_value(num_positions.to_le_bytes().as_ref());

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
#[allow(dead_code)]
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
        decompress_posting_remainder(compressed, remainder, &mut doc_ids, &mut frequencies);
    }

    Ok((doc_ids, frequencies))
}

pub fn decompress_positions(compressed: &arrow::array::LargeBinaryArray) -> Vec<u32> {
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

    positions
}

// decompress the positions list from a ListArray of binary
// to a ListArray of u32
#[allow(dead_code)]
pub fn decompress_positions_list(compressed: &ListArray) -> Result<ListArray> {
    let mut builder = ListBuilder::with_capacity(UInt32Builder::new(), compressed.len());
    for i in 0..compressed.len() {
        let compressed = compressed.value(i);
        let compressed = compressed.as_binary::<i64>();
        let positions = decompress_positions(compressed);
        builder.values().append_slice(&positions);
        builder.append(true);
    }
    Ok(builder.finish())
}

pub fn read_num_positions(posting_list: &arrow::array::LargeBinaryArray) -> u32 {
    u32::from_le_bytes(posting_list.values().as_ref()[..4].try_into().unwrap())
}

#[instrument(level = "info", name = "decompress_posting_block", skip_all)]
pub fn decompress_posting_block(
    block: &[u8],
    buffer: &mut [u32; BLOCK_SIZE],
    doc_ids: &mut Vec<u32>,
    frequencies: &mut Vec<u32>,
) {
    let num_bytes = decompress_sorted_block(block, buffer, doc_ids);
    decompress_block(&block[num_bytes..], buffer, frequencies);
}

#[instrument(level = "info", name = "decompress_posting_remainder", skip_all)]
pub fn decompress_posting_remainder(
    block: &[u8],
    n: usize,
    doc_ids: &mut Vec<u32>,
    frequencies: &mut Vec<u32>,
) {
    decompress_remainder(block, n, doc_ids);
    decompress_remainder(&block[n * 4..], n, frequencies);
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

        let decompressed_positions = decompress_positions(&compressed);
        assert_eq!(positions, decompressed_positions);
        assert_eq!(positions.len(), num_positions);
        Ok(())
    }
}
