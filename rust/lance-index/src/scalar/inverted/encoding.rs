// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::io::Write;

use super::builder::BLOCK_SIZE;
use arrow::{
    array::{AsArray, LargeBinaryBuilder},
    datatypes::ToByteSlice,
    ipc::LargeBinary,
};
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
    let compressor = BitPacker4x::new();
    let doc_id_chunks = doc_ids.chunks_exact(BLOCK_SIZE);
    let frequency_chunks = frequencies.chunks_exact(BLOCK_SIZE);
    let mut buffer = [0u8; BLOCK_SIZE * 4 + 4];
    for (doc_id_chunk, freq_chunk) in std::iter::zip(doc_id_chunks, frequency_chunks) {
        // delta encoding + bitpacking for doc ids
        let num_bits = compressor.num_bits_sorted(doc_id_chunk[0], &doc_id_chunk);
        let num_bytes =
            compressor.compress_sorted(doc_id_chunk[0], &doc_id_chunk, &mut buffer, num_bits);
        builder.write(doc_id_chunk[0].to_le_bytes().as_ref())?;
        builder.write(&[num_bits])?;
        builder.write(&buffer[..num_bytes])?;

        // bitpacking for frequencies
        let num_bits = compressor.num_bits(&freq_chunk);
        let num_bytes = compressor.compress(&freq_chunk, &mut buffer, num_bits);
        builder.write(&[num_bits])?;
        builder.append_value(&buffer[..num_bytes]);
    }

    // we don't compress the last block if it is not full
    let length = doc_ids.len();
    let remainder = length % BLOCK_SIZE;
    if remainder > 0 {
        let row_id_chunk = &doc_ids[length - remainder..];
        let freq_chunk = &frequencies[length - remainder..];
        for value in row_id_chunk.iter().chain(freq_chunk) {
            builder.write(value.to_le_bytes().as_ref())?;
        }
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
        decompress_block(compressed, &mut buffer, &mut doc_ids, &mut frequencies);
    }

    if num_docs as usize % BLOCK_SIZE > 0 {
        let compressed = posting_list.value(bitpacking_blocks);
        decompress_remainder(compressed, &mut doc_ids, &mut frequencies);
    }

    Ok((doc_ids, frequencies))
}

pub fn decompress_block(
    block: &[u8],
    buffer: &mut [u32; BLOCK_SIZE],
    doc_ids: &mut Vec<u32>,
    frequencies: &mut Vec<u32>,
) {
    let compressor = BitPacker4x::new();
    let initial = u32::from_le_bytes(block[0..4].try_into().unwrap());
    let num_bits = block[4];
    let num_bytes = compressor.decompress_sorted(initial, &block[5..], buffer, num_bits);
    doc_ids.extend_from_slice(&buffer[..]);

    let num_bits = block[5 + num_bytes];
    compressor.decompress(&block[6 + num_bytes..], buffer, num_bits);
    frequencies.extend_from_slice(&buffer[..]);
}

pub fn decompress_remainder(block: &[u8], doc_ids: &mut Vec<u32>, frequencies: &mut Vec<u32>) {
    // let reader = std::io::Cursor::new(block);
    // let reader = arrow::ipc::reader::FileReader::try_new(reader, None).unwrap();
    // for batch in reader {
    //     let batch = batch.unwrap();
    //     let doc_id_array = batch["doc_ids"].as_primitive::<UInt32Type>();
    //     let freq_array = batch["frequencies"].as_primitive::<UInt32Type>();

    //     doc_ids.extend(doc_id_array.values());
    //     frequencies.extend(freq_array.values());
    // }
    let num_docs = block.len() / 4 / 2;
    for i in 0..num_docs {
        let doc_id = u32::from_le_bytes(block[i * 4..(i + 1) * 4].try_into().unwrap());
        let freq = u32::from_le_bytes(
            block[(num_docs + i) * 4..(num_docs + i + 1) * 4]
                .try_into()
                .unwrap(),
        );
        doc_ids.push(doc_id);
        frequencies.push(freq);
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
        let num_rows: usize = BLOCK_SIZE * 4 - 1;
        let mut rng = rand::thread_rng();
        let doc_ids: Vec<u32> = (0..num_rows).map(|_| rng.gen()).sorted_unstable().collect();
        let frequencies: Vec<u32> = (0..num_rows).map(|_| rng.gen_range(1..255)).collect();
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
}
