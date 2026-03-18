// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::io::Write;

use super::builder::BLOCK_SIZE;
use super::index::{PositionStreamCodec, PostingTailCodec};
use arrow::array::LargeBinaryBuilder;
use bitpacking::{BitPacker, BitPacker4x};
use lance_core::{Error, Result};

// we compress the posting list to multiple blocks of fixed number of elements (BLOCK_SIZE),
// returns a LargeBinaryArray, where each binary is a compressed block (128 row ids + 128 frequencies)
// each block is:
// - 4 bytes for the max block score
// - 4 bytes for the first doc id
// - 1 byte for the number of bits used to pack the doc ids
// - n bytes for the packed doc ids
// - 1 byte for the number of bits used to pack the frequencies
// - n bytes for the packed frequencies
// if the block is not full (the last block), we encode the remainder separately
// using the configured remainder codec.

// compress the posting list to multiple blocks of fixed number of elements (BLOCK_SIZE),
// returns a LargeBinaryArray, where each binary is a compressed block (128 row ids + 128 frequencies)
#[cfg(test)]
pub fn compress_posting_list<'a>(
    length: usize,
    doc_ids: impl Iterator<Item = &'a u32>,
    frequencies: impl Iterator<Item = &'a u32>,
    block_max_scores: impl Iterator<Item = f32>,
) -> Result<arrow::array::LargeBinaryArray> {
    compress_posting_list_with_tail_codec(
        length,
        doc_ids,
        frequencies,
        block_max_scores,
        PostingTailCodec::VarintDelta,
    )
}

#[cfg(test)]
pub fn compress_posting_list_with_tail_codec<'a>(
    length: usize,
    doc_ids: impl Iterator<Item = &'a u32>,
    frequencies: impl Iterator<Item = &'a u32>,
    mut block_max_scores: impl Iterator<Item = f32>,
    tail_codec: PostingTailCodec,
) -> Result<arrow::array::LargeBinaryArray> {
    if length < BLOCK_SIZE {
        // directly do remainder compression to avoid overhead of creating buffer
        let mut builder = LargeBinaryBuilder::with_capacity(1, length * 4 * 2 + 1);
        // write the max score of the block
        let max_score = block_max_scores.next().unwrap();
        let _ = builder.write(max_score.to_le_bytes().as_ref())?;
        compress_posting_remainder(
            doc_ids.copied().collect::<Vec<_>>().as_slice(),
            frequencies.copied().collect::<Vec<_>>().as_slice(),
            tail_codec,
            &mut builder,
        )?;
        builder.append_value("");
        return Ok(builder.finish());
    }

    let mut builder = LargeBinaryBuilder::with_capacity(length.div_ceil(BLOCK_SIZE), length * 3);
    let mut buffer = [0u8; BLOCK_SIZE * 4 + 5];
    let mut doc_id_buffer = Vec::with_capacity(BLOCK_SIZE);
    let mut freq_buffer = Vec::with_capacity(BLOCK_SIZE);
    for (doc_id, freq) in std::iter::zip(doc_ids, frequencies) {
        doc_id_buffer.push(*doc_id);
        freq_buffer.push(*freq);

        if doc_id_buffer.len() < BLOCK_SIZE {
            continue;
        }

        assert_eq!(doc_id_buffer.len(), BLOCK_SIZE);

        // write the max score of the block
        let max_score = block_max_scores.next().unwrap();
        let _ = builder.write(max_score.to_le_bytes().as_ref())?;
        // delta encoding + bitpacking for doc ids
        compress_sorted_block(&doc_id_buffer, &mut buffer, &mut builder)?;
        // bitpacking for frequencies
        compress_block(&freq_buffer, &mut buffer, &mut builder)?;
        builder.append_value("");
        doc_id_buffer.clear();
        freq_buffer.clear();
    }

    // we don't compress the last block if it is not full
    if !doc_id_buffer.is_empty() {
        // write the max score of the block
        let max_score = block_max_scores.next().unwrap();
        let _ = builder.write(max_score.to_le_bytes().as_ref())?;
        compress_posting_remainder(&doc_id_buffer, &freq_buffer, tail_codec, &mut builder)?;
        builder.append_value("");
    }
    Ok(builder.finish())
}

pub fn encode_full_posting_block_into(
    doc_ids: &[u32],
    frequencies: &[u32],
    block: &mut Vec<u8>,
) -> Result<()> {
    debug_assert_eq!(doc_ids.len(), BLOCK_SIZE);
    debug_assert_eq!(frequencies.len(), BLOCK_SIZE);
    block.extend_from_slice(&0f32.to_le_bytes());
    let mut buffer = [0u8; BLOCK_SIZE * 4 + 5];
    compress_sorted_block(doc_ids, &mut buffer, block)?;
    compress_block(frequencies, &mut buffer, block)?;
    Ok(())
}

pub fn encode_remainder_posting_block_into(
    doc_ids: &[u32],
    frequencies: &[u32],
    codec: PostingTailCodec,
    block: &mut Vec<u8>,
) -> Result<()> {
    debug_assert_eq!(doc_ids.len(), frequencies.len());
    block.extend_from_slice(&0f32.to_le_bytes());
    compress_posting_remainder(doc_ids, frequencies, codec, block)?;
    Ok(())
}

#[inline]
fn compress_sorted_block(data: &[u32], buffer: &mut [u8], builder: &mut impl Write) -> Result<()> {
    let compressor = BitPacker4x::new();
    let num_bits = compressor.num_bits_sorted(data[0], data);
    let num_bytes = compressor.compress_sorted(data[0], data, buffer, num_bits);
    let _ = builder.write(data[0].to_le_bytes().as_ref())?;
    let _ = builder.write(&[num_bits])?;
    let _ = builder.write(&buffer[..num_bytes])?;
    Ok(())
}

#[inline]
fn compress_block(data: &[u32], buffer: &mut [u8], builder: &mut impl Write) -> Result<()> {
    let compressor = BitPacker4x::new();
    let num_bits = compressor.num_bits(data);
    let num_bytes = compressor.compress(data, buffer, num_bits);
    let _ = builder.write(&[num_bits])?;
    let _ = builder.write(&buffer[..num_bytes])?;
    Ok(())
}

#[inline]
fn compress_raw_remainder(data: &[u32], builder: &mut impl Write) -> Result<()> {
    for value in data.iter() {
        let _ = builder.write(value.to_le_bytes().as_ref())?;
    }
    Ok(())
}

#[inline]
fn write_varint_u32(builder: &mut impl Write, mut value: u32) -> Result<()> {
    let mut bytes = [0u8; 5];
    let mut len = 0usize;
    while value >= 0x80 {
        bytes[len] = (value as u8) | 0x80;
        value >>= 7;
        len += 1;
    }
    bytes[len] = value as u8;
    len += 1;
    let _ = builder.write(&bytes[..len])?;
    Ok(())
}

#[inline]
fn compress_posting_remainder(
    doc_ids: &[u32],
    frequencies: &[u32],
    codec: PostingTailCodec,
    builder: &mut impl Write,
) -> Result<()> {
    debug_assert_eq!(doc_ids.len(), frequencies.len());
    match codec {
        PostingTailCodec::Fixed32 => {
            compress_raw_remainder(doc_ids, builder)?;
            compress_raw_remainder(frequencies, builder)?;
        }
        PostingTailCodec::VarintDelta => {
            let mut previous = 0u32;
            for (index, &doc_id) in doc_ids.iter().enumerate() {
                let delta = if index == 0 {
                    doc_id
                } else {
                    doc_id.checked_sub(previous).ok_or_else(|| {
                        Error::index(format!(
                            "doc ids must be sorted within a posting tail block, got {} after {}",
                            doc_id, previous
                        ))
                    })?
                };
                write_varint_u32(builder, delta)?;
                previous = doc_id;
            }
            for &frequency in frequencies {
                write_varint_u32(builder, frequency)?;
            }
        }
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
        compress_raw_remainder(&positions[length - remainder..], &mut builder)?;
        builder.append_value("");
    }

    Ok(builder.finish())
}

#[inline]
fn encode_varint_u32(dst: &mut Vec<u8>, mut value: u32) {
    while value >= 0x80 {
        dst.push((value as u8) | 0x80);
        value >>= 7;
    }
    dst.push(value as u8);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PositionBlockBuilder {
    codec: PositionStreamCodec,
    encoded_bytes: Vec<u8>,
    pending_deltas: Vec<u32>,
}

impl Default for PositionBlockBuilder {
    fn default() -> Self {
        Self::new(PositionStreamCodec::PackedDelta)
    }
}

impl PositionBlockBuilder {
    pub(super) fn new(codec: PositionStreamCodec) -> Self {
        Self {
            codec,
            encoded_bytes: Vec::new(),
            pending_deltas: Vec::new(),
        }
    }

    pub(super) fn size(&self) -> usize {
        self.encoded_bytes.capacity() + self.pending_deltas.capacity() * std::mem::size_of::<u32>()
    }

    pub(super) fn append_doc_positions(&mut self, positions: &[u32]) -> Result<()> {
        let mut previous = 0u32;
        for (index, &position) in positions.iter().enumerate() {
            let delta = if index == 0 {
                position
            } else {
                position.checked_sub(previous).ok_or_else(|| {
                    Error::index(format!(
                        "positions must be sorted within a document, got {} after {}",
                        position, previous
                    ))
                })?
            };
            self.push_delta(delta)?;
            previous = position;
        }
        Ok(())
    }

    pub(super) fn append_position(
        &mut self,
        position: u32,
        previous_in_doc: Option<u32>,
    ) -> Result<()> {
        let delta = match previous_in_doc {
            Some(previous) => position.checked_sub(previous).ok_or_else(|| {
                Error::index(format!(
                    "positions must be sorted within a document, got {} after {}",
                    position, previous
                ))
            })?,
            None => position,
        };
        self.push_delta(delta)
    }

    pub(super) fn finish(self) -> Vec<u8> {
        let mut bytes = self.encoded_bytes;
        match self.codec {
            PositionStreamCodec::VarintDocDelta | PositionStreamCodec::PackedDelta => {
                for delta in self.pending_deltas {
                    encode_varint_u32(&mut bytes, delta);
                }
            }
        }
        bytes
    }

    pub(super) fn decode_into(&self, frequencies: &[u32], dst: &mut Vec<u32>) -> Result<()> {
        let bytes = self.clone().finish();
        decode_position_stream_block(bytes.as_slice(), frequencies, self.codec, dst)
    }

    fn push_delta(&mut self, delta: u32) -> Result<()> {
        match self.codec {
            PositionStreamCodec::VarintDocDelta => {
                encode_varint_u32(&mut self.encoded_bytes, delta);
            }
            PositionStreamCodec::PackedDelta => {
                self.pending_deltas.push(delta);
                if self.pending_deltas.len() == BLOCK_SIZE {
                    let mut packed_buffer = [0u8; BLOCK_SIZE * 4 + 1];
                    compress_block(
                        self.pending_deltas.as_slice(),
                        &mut packed_buffer,
                        &mut self.encoded_bytes,
                    )?;
                    self.pending_deltas.clear();
                }
            }
        }
        Ok(())
    }
}

#[inline]
fn decode_varint_u32(src: &[u8], offset: &mut usize) -> Result<u32> {
    let mut value = 0u32;
    let mut shift = 0u32;
    while *offset < src.len() {
        let byte = src[*offset];
        *offset += 1;
        value |= u32::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift >= 35 {
            return Err(Error::index(
                "invalid u32 varint in position stream".to_owned(),
            ));
        }
    }
    Err(Error::index(
        "unexpected EOF while decoding position stream".to_owned(),
    ))
}

#[cfg(test)]
fn encode_position_stream_varint_block_into(
    positions: &[u32],
    frequencies: &[u32],
    dst: &mut Vec<u8>,
) -> Result<()> {
    let mut offset = 0usize;
    for &frequency in frequencies {
        let frequency = frequency as usize;
        let end = offset
            .checked_add(frequency)
            .ok_or_else(|| Error::index("position block length overflow".to_owned()))?;
        if end > positions.len() {
            return Err(Error::index(format!(
                "position block has {} positions but frequencies require at least {}",
                positions.len(),
                end
            )));
        }
        let mut previous = 0u32;
        for (index, &position) in positions[offset..end].iter().enumerate() {
            let delta = if index == 0 {
                position
            } else {
                position.checked_sub(previous).ok_or_else(|| {
                    Error::index(format!(
                        "positions must be sorted within a document, got {} after {}",
                        position, previous
                    ))
                })?
            };
            encode_varint_u32(dst, delta);
            previous = position;
        }
        offset = end;
    }
    if offset != positions.len() {
        return Err(Error::index(format!(
            "position block has {} trailing positions after consuming {} frequencies",
            positions.len() - offset,
            frequencies.len()
        )));
    }
    Ok(())
}

fn decode_position_stream_varint_block(
    src: &[u8],
    frequencies: &[u32],
    dst: &mut Vec<u32>,
) -> Result<()> {
    let mut offset = 0usize;
    for &frequency in frequencies {
        let mut previous = 0u32;
        for index in 0..frequency as usize {
            let delta = decode_varint_u32(src, &mut offset)?;
            let position = if index == 0 {
                delta
            } else {
                previous.checked_add(delta).ok_or_else(|| {
                    Error::index("position stream overflow while decoding".to_owned())
                })?
            };
            dst.push(position);
            previous = position;
        }
    }
    if offset != src.len() {
        return Err(Error::index(format!(
            "position stream has {} trailing bytes after decoding block",
            src.len() - offset
        )));
    }
    Ok(())
}

#[cfg(test)]
fn encode_position_stream_packed_block_into(
    positions: &[u32],
    frequencies: &[u32],
    dst: &mut Vec<u8>,
) -> Result<()> {
    let mut delta_buffer = [0u32; BLOCK_SIZE];
    let mut delta_count = 0usize;
    let mut packed_buffer = [0u8; BLOCK_SIZE * 4 + 1];
    let mut offset = 0usize;

    for &frequency in frequencies {
        let frequency = frequency as usize;
        let end = offset
            .checked_add(frequency)
            .ok_or_else(|| Error::index("position block length overflow".to_owned()))?;
        if end > positions.len() {
            return Err(Error::index(format!(
                "position block has {} positions but frequencies require at least {}",
                positions.len(),
                end
            )));
        }
        let mut previous = 0u32;
        for (index, &position) in positions[offset..end].iter().enumerate() {
            let delta = if index == 0 {
                position
            } else {
                position.checked_sub(previous).ok_or_else(|| {
                    Error::index(format!(
                        "positions must be sorted within a document, got {} after {}",
                        position, previous
                    ))
                })?
            };
            delta_buffer[delta_count] = delta;
            delta_count += 1;
            if delta_count == BLOCK_SIZE {
                compress_block(&delta_buffer, &mut packed_buffer, dst)?;
                delta_count = 0;
            }
            previous = position;
        }
        offset = end;
    }

    if offset != positions.len() {
        return Err(Error::index(format!(
            "position block has {} trailing positions after consuming {} frequencies",
            positions.len() - offset,
            frequencies.len()
        )));
    }

    for delta in &delta_buffer[..delta_count] {
        encode_varint_u32(dst, *delta);
    }
    Ok(())
}

fn decode_position_stream_packed_block(
    src: &[u8],
    frequencies: &[u32],
    dst: &mut Vec<u32>,
) -> Result<()> {
    let total_positions = frequencies.iter().try_fold(0usize, |total, &frequency| {
        total.checked_add(frequency as usize).ok_or_else(|| {
            Error::index("position stream length overflow while decoding".to_owned())
        })
    })?;

    let full_delta_blocks = total_positions / BLOCK_SIZE;
    let tail_len = total_positions % BLOCK_SIZE;

    let compressor = BitPacker4x::new();
    let mut packed_offset = 0usize;
    let mut packed_values = [0u32; BLOCK_SIZE];
    let mut deltas = Vec::with_capacity(total_positions);

    for _ in 0..full_delta_blocks {
        if packed_offset >= src.len() {
            return Err(Error::index(
                "unexpected EOF while decoding packed position stream".to_owned(),
            ));
        }
        let num_bits = src[packed_offset];
        packed_offset += 1;
        let consumed = compressor.decompress(&src[packed_offset..], &mut packed_values, num_bits);
        packed_offset += consumed;
        deltas.extend_from_slice(&packed_values);
    }

    for _ in 0..tail_len {
        deltas.push(decode_varint_u32(src, &mut packed_offset)?);
    }

    if packed_offset != src.len() {
        return Err(Error::index(format!(
            "position stream has {} trailing bytes after decoding block",
            src.len() - packed_offset
        )));
    }

    let mut delta_offset = 0usize;
    for &frequency in frequencies {
        let mut previous = 0u32;
        for index in 0..frequency as usize {
            let delta = deltas[delta_offset];
            delta_offset += 1;
            let position = if index == 0 {
                delta
            } else {
                previous.checked_add(delta).ok_or_else(|| {
                    Error::index("position stream overflow while decoding".to_owned())
                })?
            };
            dst.push(position);
            previous = position;
        }
    }
    debug_assert_eq!(delta_offset, deltas.len());
    Ok(())
}

#[cfg(test)]
pub fn encode_position_stream_block_into(
    positions: &[u32],
    frequencies: &[u32],
    codec: PositionStreamCodec,
    dst: &mut Vec<u8>,
) -> Result<()> {
    match codec {
        PositionStreamCodec::VarintDocDelta => {
            encode_position_stream_varint_block_into(positions, frequencies, dst)
        }
        PositionStreamCodec::PackedDelta => {
            encode_position_stream_packed_block_into(positions, frequencies, dst)
        }
    }
}

pub fn decode_position_stream_block(
    src: &[u8],
    frequencies: &[u32],
    codec: PositionStreamCodec,
    dst: &mut Vec<u32>,
) -> Result<()> {
    match codec {
        PositionStreamCodec::VarintDocDelta => {
            decode_position_stream_varint_block(src, frequencies, dst)
        }
        PositionStreamCodec::PackedDelta => {
            decode_position_stream_packed_block(src, frequencies, dst)
        }
    }
}

/// decompress the posting list from a LargeBinaryArray
/// returns a vector of (row_id, frequency) tuples
#[cfg(test)]
pub fn decompress_posting_list(
    num_docs: u32,
    posting_list: &arrow::array::LargeBinaryArray,
) -> Result<(Vec<u32>, Vec<u32>)> {
    decompress_posting_list_with_tail_codec(num_docs, posting_list, PostingTailCodec::VarintDelta)
}

#[cfg(test)]
pub fn decompress_posting_list_with_tail_codec(
    num_docs: u32,
    posting_list: &arrow::array::LargeBinaryArray,
    tail_codec: PostingTailCodec,
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
        decompress_posting_remainder(
            compressed,
            remainder,
            tail_codec,
            &mut doc_ids,
            &mut frequencies,
        );
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
        let compressed_block = compressed.value(num_blocks + 1);
        decompress_raw_remainder(compressed_block, remainder, &mut positions);
    }

    positions
}

pub fn read_num_positions(compressed: &arrow::array::LargeBinaryArray) -> u32 {
    u32::from_le_bytes(compressed.value(0).try_into().unwrap())
}

pub fn decompress_posting_block(
    block: &[u8],
    buffer: &mut [u32; BLOCK_SIZE],
    doc_ids: &mut Vec<u32>,
    frequencies: &mut Vec<u32>,
) {
    // skip the first 4 bytes for the max block score
    let block = &block[4..];
    let num_bytes = decompress_sorted_block(block, buffer, doc_ids);
    decompress_block(&block[num_bytes..], buffer, frequencies);
}

pub fn decompress_posting_remainder(
    block: &[u8],
    n: usize,
    codec: PostingTailCodec,
    doc_ids: &mut Vec<u32>,
    frequencies: &mut Vec<u32>,
) {
    let block = &block[4..];
    match codec {
        PostingTailCodec::Fixed32 => {
            decompress_raw_remainder(block, n, doc_ids);
            decompress_raw_remainder(&block[n * 4..], n, frequencies);
        }
        PostingTailCodec::VarintDelta => {
            let mut offset = 0usize;
            let mut previous = 0u32;
            for index in 0..n {
                let delta = decode_varint_u32(block, &mut offset)
                    .expect("posting tail doc ids should contain valid varints");
                let doc_id = if index == 0 {
                    delta
                } else {
                    previous
                        .checked_add(delta)
                        .expect("posting tail doc id delta should not overflow")
                };
                doc_ids.push(doc_id);
                previous = doc_id;
            }
            for _ in 0..n {
                let frequency = decode_varint_u32(block, &mut offset)
                    .expect("posting tail frequencies should contain valid varints");
                frequencies.push(frequency);
            }
            assert_eq!(
                offset,
                block.len(),
                "posting tail block has {} trailing bytes after decoding",
                block.len() - offset
            );
        }
    }
}

pub fn decode_full_posting_block(block: &[u8], doc_ids: &mut Vec<u32>, frequencies: &mut Vec<u32>) {
    let mut buffer = [0u32; BLOCK_SIZE];
    decompress_posting_block(block, &mut buffer, doc_ids, frequencies);
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

pub fn decompress_raw_remainder(compressed: &[u8], n: usize, dest: &mut Vec<u32>) {
    for bytes in compressed.chunks_exact(4).take(n) {
        let data = u32::from_le_bytes(bytes.try_into().unwrap());
        dest.push(data);
    }
}

pub fn read_posting_tail_first_doc(block: &[u8], codec: PostingTailCodec) -> u32 {
    match codec {
        PostingTailCodec::Fixed32 => u32::from_le_bytes(block[4..8].try_into().unwrap()),
        PostingTailCodec::VarintDelta => {
            let mut offset = 4usize;
            decode_varint_u32(block, &mut offset)
                .expect("posting tail block should contain a valid first doc id")
        }
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
        let mut rng = rand::rng();
        let doc_ids: Vec<u32> = (0..num_rows)
            .map(|_| rng.random())
            .sorted_unstable()
            .collect();
        let frequencies: Vec<u32> = (0..num_rows)
            .map(|_| rng.random_range(1..=u32::MAX))
            .collect();
        let block_max_scores =
            (0..num_rows.div_ceil(BLOCK_SIZE)).map(|_| rng.random_range(0.0..1.0));
        let posting_list = compress_posting_list(
            doc_ids.len(),
            doc_ids.iter(),
            frequencies.iter(),
            block_max_scores,
        )?;
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
    fn test_compress_posting_list_fixed32_tail_still_roundtrips() -> Result<()> {
        let doc_ids = vec![3_u32, 10_u32, 24_u32];
        let frequencies = vec![1_u32, 7_u32, 2_u32];
        let posting_list = compress_posting_list_with_tail_codec(
            doc_ids.len(),
            doc_ids.iter(),
            frequencies.iter(),
            std::iter::once(1.0_f32),
            PostingTailCodec::Fixed32,
        )?;
        let (decoded_doc_ids, decoded_frequencies) = decompress_posting_list_with_tail_codec(
            doc_ids.len() as u32,
            &posting_list,
            PostingTailCodec::Fixed32,
        )?;
        assert_eq!(decoded_doc_ids, doc_ids);
        assert_eq!(decoded_frequencies, frequencies);
        Ok(())
    }

    #[test]
    fn test_compress_positions() -> Result<()> {
        let num_positions: usize = BLOCK_SIZE * 2 - 7;
        let mut rng = rand::rng();
        let positions: Vec<u32> = (0..num_positions)
            .map(|_| rng.random())
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

    #[test]
    fn test_encode_position_stream_block_roundtrip() -> Result<()> {
        let frequencies = vec![1, 3, 2, 4];
        let positions = vec![7, 1, 3, 8, 2, 100, 0, 4, 9, 25];
        for codec in [
            PositionStreamCodec::VarintDocDelta,
            PositionStreamCodec::PackedDelta,
        ] {
            let mut encoded = Vec::new();
            encode_position_stream_block_into(&positions, &frequencies, codec, &mut encoded)?;
            let mut decoded = Vec::new();
            decode_position_stream_block(&encoded, &frequencies, codec, &mut decoded)?;
            assert_eq!(decoded, positions);
            assert!(encoded.len() < positions.len() * std::mem::size_of::<u32>());
        }
        Ok(())
    }

    #[test]
    fn test_incremental_position_block_builder_matches_batch_encoder() -> Result<()> {
        let frequencies = vec![1, 3, 2, 4, 1, 5];
        let positions = vec![7, 1, 3, 8, 2, 100, 0, 4, 9, 25, 11, 2, 6, 7, 10, 15];

        let mut builder = PositionBlockBuilder::new(PositionStreamCodec::PackedDelta);
        let mut offset = 0usize;
        for &frequency in &frequencies {
            let end = offset + frequency as usize;
            builder.append_doc_positions(&positions[offset..end])?;
            offset = end;
        }

        let incremental = builder.finish();
        let mut batch = Vec::new();
        encode_position_stream_block_into(
            &positions,
            &frequencies,
            PositionStreamCodec::PackedDelta,
            &mut batch,
        )?;
        assert_eq!(incremental, batch);
        Ok(())
    }
}
