// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Classic ALP (Adaptive Lossless floating-Point) miniblock encoding.
//!
//! # Buffer layout
//!
//! This is a **miniblock** physical encoding (see `encodings::logical::primitive::miniblock`).
//! The encoder emits **four value buffers per page**. For each chunk, `MiniBlockChunk.buffer_sizes`
//! contains four entries matching these buffers in order.
//!
//! - **Buffer 0 (payload):** bitpacked integer deltas
//!   - values are `encoded[i] - min`, stored using the same layout as
//!     `encodings::physical::bitpacking::bitpack_out_of_line`
//! - **Buffer 1 (header):** per-chunk header
//!   - `min`: i32 for f32 or i64 for f64 (little-endian)
//!   - `bit_width`: u8 (number of bits per delta in buffer 0)
//! - **Buffer 2 (exceptions positions):** `u16` positions (little-endian), relative to the chunk
//! - **Buffer 3 (exceptions values):** original IEEE754 bit patterns for exceptions
//!   - f32: `u32` little-endian bits, one per exception
//!   - f64: `u64` little-endian bits, one per exception
//!
//! # Why this design
//!
//! - **Bitwise lossless:** exceptions store the original IEEE754 bit patterns. This preserves
//!   `-0.0` and NaN payloads, which cannot be guaranteed by float equality alone.
//! - **Miniblock-friendly random access:** decoding happens at chunk granularity. Positions use
//!   `u16` because chunk sizes are limited to 1024 (f32) / 512 (f64), keeping the exception index
//!   overhead small.
//! - **Plays well with `GeneralMiniBlockCompressor`:** it only compresses the *first* buffer. Placing
//!   the ALP main payload (bitpacked deltas) in buffer 0 maximizes the benefit while leaving the
//!   usually-small header/exception side buffers untouched.
//! - **Robustness:** if a page is not compressible (or a chunk cannot be encoded), the implementation
//!   falls back to the `ValueEncoder` (flat encoding) instead of forcing a larger representation.

use std::fmt::Debug;

use snafu::location;

use crate::buffer::LanceBuffer;
use crate::compression::MiniBlockDecompressor;
use crate::data::{BlockInfo, DataBlock, FixedWidthDataBlock};
use crate::encodings::logical::primitive::miniblock::{
    MiniBlockChunk, MiniBlockCompressed, MiniBlockCompressor, MAX_MINIBLOCK_BYTES,
};
#[cfg(feature = "bitpacking")]
use crate::encodings::physical::bitpacking::{bitpack_out_of_line_vec, unpack_out_of_line};
use crate::encodings::physical::value::ValueEncoder;
use crate::format::pb21::CompressiveEncoding;
use crate::format::{pb21, ProtobufUtils21};

use lance_core::{Error, Result};

#[derive(Debug, Clone, Copy)]
struct Exponents {
    e: u8,
    f: u8,
}

#[inline]
fn bits_required_u64(v: u64) -> u8 {
    if v == 0 {
        0
    } else {
        (64 - v.leading_zeros()) as u8
    }
}

/// Estimate the number of bytes produced by `bitpack_out_of_line` for `num_values`.
///
/// This models the exact layout decision used by `bitpack_out_of_line`:
/// values are packed in blocks of 1024, and the tail is either padded+packed or
/// appended raw depending on a simple cost comparison.
fn estimate_bitpack_out_of_line_bytes(
    uncompressed_bits_per_value: usize,
    num_values: usize,
    compressed_bits_per_value: usize,
) -> usize {
    const ELEMS_PER_CHUNK: usize = 1024;

    if num_values == 0 {
        return 0;
    }
    debug_assert!(uncompressed_bits_per_value % 8 == 0);

    let words_per_chunk =
        (ELEMS_PER_CHUNK * compressed_bits_per_value).div_ceil(uncompressed_bits_per_value);

    let num_chunks = num_values.div_ceil(ELEMS_PER_CHUNK);
    let last_chunk_is_runt = num_values % ELEMS_PER_CHUNK != 0;
    let num_whole_chunks = if last_chunk_is_runt {
        num_chunks.saturating_sub(1)
    } else {
        num_chunks
    };

    let mut words = num_whole_chunks * words_per_chunk;
    if last_chunk_is_runt {
        let remaining_items = num_values - (num_whole_chunks * ELEMS_PER_CHUNK);
        let tail_bit_savings =
            uncompressed_bits_per_value.saturating_sub(compressed_bits_per_value);
        let padding_cost = compressed_bits_per_value * (ELEMS_PER_CHUNK - remaining_items);
        let tail_pack_savings = tail_bit_savings.saturating_mul(remaining_items);
        if padding_cost < tail_pack_savings {
            words += words_per_chunk;
        } else {
            words += remaining_items;
        }
    }

    words * (uncompressed_bits_per_value / 8)
}

/// Pick sample indices for exponent search.
///
/// ALP exponent selection is relatively expensive. For large pages we use a small deterministic
/// sample (up to 32 values) to estimate the best `(e, f)` without scanning the whole input.
fn sample_positions(num_values: usize, sample_size: usize) -> Vec<usize> {
    if num_values <= sample_size {
        return (0..num_values).collect();
    }
    let step = num_values / sample_size;
    (0..sample_size).map(|i| i * step).collect()
}

/// Exhaustively search ALP exponents for `f32` using a small sample.
///
/// We try all `0 <= f < e <= 10` and pick the pair with the smallest estimated encoded size.
/// The estimate uses the classic ALP output (`encode_f32`) and models the downstream integer
/// packing as `min + bitpack_out_of_line(deltas)` plus exception overhead. This keeps selection
/// stable and fast while tracking the same storage layout used by actual chunk encoding.
fn find_best_exponents_f32(values: &[f32]) -> Exponents {
    let positions = sample_positions(values.len(), 32);
    let sample = positions.into_iter().map(|i| values[i]).collect::<Vec<_>>();

    let mut best = Exponents { e: 0, f: 0 };
    let mut best_bytes = usize::MAX;
    for e in (0u8..=10).rev() {
        for f in 0u8..e {
            let exp = Exponents { e, f };
            let (encoded, patch_count) = encode_f32(&sample, exp);
            let bytes = estimate_alp_size_i64(&encoded, patch_count, 4);
            if bytes < best_bytes || (bytes == best_bytes && e - f < best.e - best.f) {
                best = exp;
                best_bytes = bytes;
            }
        }
    }
    best
}

/// Exhaustively search ALP exponents for `f64` using a small sample.
///
/// Same strategy as [`find_best_exponents_f32`], but with the `f64` exponent range (`e <= 18`).
fn find_best_exponents_f64(values: &[f64]) -> Exponents {
    let positions = sample_positions(values.len(), 32);
    let sample = positions.into_iter().map(|i| values[i]).collect::<Vec<_>>();

    let mut best = Exponents { e: 0, f: 0 };
    let mut best_bytes = usize::MAX;
    for e in (0u8..=18).rev() {
        for f in 0u8..e {
            let exp = Exponents { e, f };
            let (encoded, patch_count) = encode_f64(&sample, exp);
            let bytes = estimate_alp_size_i64(&encoded, patch_count, 8);
            if bytes < best_bytes || (bytes == best_bytes && e - f < best.e - best.f) {
                best = exp;
                best_bytes = bytes;
            }
        }
    }
    best
}

fn estimate_alp_size_i64(encoded: &[i64], patch_count: usize, value_bytes: usize) -> usize {
    if encoded.is_empty() {
        return 0;
    }
    let mut min = i64::MAX;
    let mut max = i64::MIN;
    for &v in encoded {
        min = min.min(v);
        max = max.max(v);
    }
    let range = max.wrapping_sub(min) as u64;
    let bit_width = bits_required_u64(range) as usize;
    let packed = estimate_bitpack_out_of_line_bytes(value_bytes * 8, encoded.len(), bit_width);
    let header = value_bytes + 1; // min + bit_width (stored in a separate header buffer)
    let exceptions = patch_count * (2 + value_bytes); // pos(u16) + value bits
    header + packed + exceptions
}

#[inline]
fn fast_round_f32(v: f32) -> f32 {
    const SWEET: f32 = (1u32 << 23) as f32 + (1u32 << 22) as f32;
    (v + SWEET) - SWEET
}

#[inline]
fn fast_round_f64(v: f64) -> f64 {
    const SWEET: f64 = (1u64 << 52) as f64 + (1u64 << 51) as f64;
    (v + SWEET) - SWEET
}

const F10_F32: [f32; 11] = [
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
    1000000.0,
    10000000.0,
    100000000.0,
    1000000000.0,
    10000000000.0,
];
const IF10_F32: [f32; 11] = [
    1.0,
    0.1,
    0.01,
    0.001,
    0.0001,
    0.00001,
    0.000001,
    0.0000001,
    0.00000001,
    0.000000001,
    0.0000000001,
];

const F10_F64: [f64; 24] = [
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
    1000000.0,
    10000000.0,
    100000000.0,
    1000000000.0,
    10000000000.0,
    100000000000.0,
    1000000000000.0,
    10000000000000.0,
    100000000000000.0,
    1000000000000000.0,
    10000000000000000.0,
    100000000000000000.0,
    1000000000000000000.0,
    10000000000000000000.0,
    100000000000000000000.0,
    1000000000000000000000.0,
    10000000000000000000000.0,
    100000000000000000000000.0,
];
const IF10_F64: [f64; 24] = [
    1.0,
    0.1,
    0.01,
    0.001,
    0.0001,
    0.00001,
    0.000001,
    0.0000001,
    0.00000001,
    0.000000001,
    0.0000000001,
    0.00000000001,
    0.000000000001,
    0.0000000000001,
    0.00000000000001,
    0.000000000000001,
    0.0000000000000001,
    0.00000000000000001,
    0.000000000000000001,
    0.0000000000000000001,
    0.00000000000000000001,
    0.000000000000000000001,
    0.0000000000000000000001,
    0.00000000000000000000001,
];

/// Classic ALP encode for `f32`, used for exponent selection only.
///
/// This function computes the transformed integer stream and counts how many values must be
/// stored as exceptions when requiring **bitwise** round-tripping (`to_bits()` equality).
///
/// For size estimation we replace exception slots with a "fill" value (the first encodable value).
/// This matches the behavior of the real chunk encoder: reducing the integer range improves
/// downstream delta+bitpacking even when exceptions exist.
fn encode_f32(values: &[f32], exp: Exponents) -> (Vec<i64>, usize) {
    let mut encoded = Vec::with_capacity(values.len());
    let mut patches = 0usize;
    let mut fill: Option<i32> = None;

    for &v in values {
        if !v.is_finite() {
            encoded.push(0);
            patches += 1;
            continue;
        }
        let scaled = v * F10_F32[exp.e as usize] * IF10_F32[exp.f as usize];
        if !scaled.is_finite() {
            encoded.push(0);
            patches += 1;
            continue;
        }
        let rounded = fast_round_f32(scaled);
        let enc = rounded as i32;
        let decoded = (enc as f32) * F10_F32[exp.f as usize] * IF10_F32[exp.e as usize];
        if decoded.to_bits() == v.to_bits() {
            if fill.is_none() {
                fill = Some(enc);
            }
            encoded.push(enc as i64);
        } else {
            encoded.push(0);
            patches += 1;
        }
    }

    if let Some(fill) = fill {
        for (i, &v) in values.iter().enumerate() {
            if !v.is_finite() {
                encoded[i] = fill as i64;
                continue;
            }
            let scaled = v * F10_F32[exp.e as usize] * IF10_F32[exp.f as usize];
            if !scaled.is_finite() {
                encoded[i] = fill as i64;
                continue;
            }
            let rounded = fast_round_f32(scaled);
            let enc = rounded as i32;
            let decoded = (enc as f32) * F10_F32[exp.f as usize] * IF10_F32[exp.e as usize];
            if decoded.to_bits() != v.to_bits() {
                encoded[i] = fill as i64;
            }
        }
    }

    (encoded, patches)
}

/// Classic ALP encode for `f64`, used for exponent selection only.
///
/// See [`encode_f32`] for the design. The core difference is the exponent range and integer type.
fn encode_f64(values: &[f64], exp: Exponents) -> (Vec<i64>, usize) {
    let mut encoded = Vec::with_capacity(values.len());
    let mut patches = 0usize;
    let mut fill: Option<i64> = None;

    for &v in values {
        if !v.is_finite() {
            encoded.push(0);
            patches += 1;
            continue;
        }
        let scaled = v * F10_F64[exp.e as usize] * IF10_F64[exp.f as usize];
        if !scaled.is_finite() {
            encoded.push(0);
            patches += 1;
            continue;
        }
        let rounded = fast_round_f64(scaled);
        let enc = rounded as i64;
        let decoded = (enc as f64) * F10_F64[exp.f as usize] * IF10_F64[exp.e as usize];
        if decoded.to_bits() == v.to_bits() {
            if fill.is_none() {
                fill = Some(enc);
            }
            encoded.push(enc);
        } else {
            encoded.push(0);
            patches += 1;
        }
    }

    if let Some(fill) = fill {
        for (i, &v) in values.iter().enumerate() {
            if !v.is_finite() {
                encoded[i] = fill;
                continue;
            }
            let scaled = v * F10_F64[exp.e as usize] * IF10_F64[exp.f as usize];
            if !scaled.is_finite() {
                encoded[i] = fill;
                continue;
            }
            let rounded = fast_round_f64(scaled);
            let enc = rounded as i64;
            let decoded = (enc as f64) * F10_F64[exp.f as usize] * IF10_F64[exp.e as usize];
            if decoded.to_bits() != v.to_bits() {
                encoded[i] = fill;
            }
        }
    }

    (encoded, patches)
}

/// Encodes f32/f64 values using the ALP miniblock layout described in the module docs.
///
/// This is an *opt-in* encoder (selected via compression metadata/config) and is expected to be
/// version-gated by the caller (Lance file version >= 2.2). The encoder may fall back to
/// `ValueEncoder` if ALP is ineffective for the given page.
#[derive(Debug, Clone)]
pub struct AlpMiniBlockEncoder {
    bits_per_value: u64,
}

impl AlpMiniBlockEncoder {
    pub fn new(bits_per_value: u64) -> Self {
        assert!(bits_per_value == 32 || bits_per_value == 64);
        Self { bits_per_value }
    }

    fn max_chunk_size(&self) -> usize {
        match self.bits_per_value {
            32 => 1024,
            64 => 512,
            _ => unreachable!(),
        }
    }
}

impl MiniBlockCompressor for AlpMiniBlockEncoder {
    fn compress(&self, page: DataBlock) -> Result<(MiniBlockCompressed, CompressiveEncoding)> {
        let DataBlock::FixedWidth(data) = page else {
            return Err(Error::invalid_input(
                "ALP encoding only supports FixedWidth data blocks",
                location!(),
            ));
        };

        if data.bits_per_value != self.bits_per_value {
            return Err(Error::invalid_input(
                "ALP bits_per_value mismatch",
                location!(),
            ));
        }

        if data.num_values == 0 {
            let encoding = ProtobufUtils21::alp(self.bits_per_value as u32, 1, 0);
            return Ok((
                MiniBlockCompressed {
                    data: vec![
                        LanceBuffer::empty(),
                        LanceBuffer::empty(),
                        LanceBuffer::empty(),
                        LanceBuffer::empty(),
                    ],
                    chunks: vec![],
                    num_values: 0,
                },
                encoding,
            ));
        }

        let max_chunk = self.max_chunk_size();
        let bytes_per_value = (self.bits_per_value / 8) as usize;
        let raw_size = data.num_values as usize * bytes_per_value;

        match self.bits_per_value {
            32 => {
                let words = data.data.borrow_to_typed_slice::<u32>();
                let floats = words
                    .as_ref()
                    .iter()
                    .map(|b| f32::from_bits(*b))
                    .collect::<Vec<_>>();
                let exponents = find_best_exponents_f32(&floats);

                let mut buf0: Vec<u32> = Vec::new();
                let mut buf1: Vec<u8> = Vec::new();
                let mut buf2: Vec<u8> = Vec::new();
                let mut buf3: Vec<u8> = Vec::new();
                let mut chunks = Vec::new();

                let mut offset = 0usize;
                let bytes = data.data.as_ref();
                while offset < bytes.len() {
                    let remaining_values = (bytes.len() - offset) / bytes_per_value;
                    let chunk_values = remaining_values.min(max_chunk);
                    let chunk_bytes_len = chunk_values * bytes_per_value;
                    let chunk_bytes = &bytes[offset..offset + chunk_bytes_len];

                    let words = bytemuck::try_cast_slice::<u8, u32>(chunk_bytes).map_err(|_| {
                        Error::invalid_input("invalid f32 buffer alignment", location!())
                    })?;
                    let floats = words.iter().map(|b| f32::from_bits(*b)).collect::<Vec<_>>();
                    let Ok(encoded) = encode_chunk_f32(&floats, exponents) else {
                        return ValueEncoder::default().compress(DataBlock::FixedWidth(data));
                    };

                    let delta_bytes = encoded
                        .packed_deltas
                        .len()
                        .checked_mul(std::mem::size_of::<u32>())
                        .ok_or_else(|| {
                            Error::invalid_input("ALP chunk size overflow", location!())
                        })? as u32;
                    let header_bytes = encoded.header.len() as u32;
                    let pos_bytes = (encoded.exception_positions.len() * 2) as u32;
                    let exc_bytes = encoded.exception_bits.len() as u32;

                    let total_value_bytes = delta_bytes as u64
                        + header_bytes as u64
                        + pos_bytes as u64
                        + exc_bytes as u64;
                    if total_value_bytes > MAX_MINIBLOCK_BYTES {
                        return ValueEncoder::default().compress(DataBlock::FixedWidth(data));
                    }

                    buf0.extend(encoded.packed_deltas);
                    buf1.extend_from_slice(&encoded.header);
                    for pos in encoded.exception_positions {
                        buf2.extend_from_slice(&pos.to_le_bytes());
                    }
                    buf3.extend_from_slice(&encoded.exception_bits);

                    let log_num_values = if offset + chunk_bytes_len == bytes.len() {
                        0
                    } else {
                        (chunk_values as u64).ilog2() as u8
                    };
                    chunks.push(MiniBlockChunk {
                        buffer_sizes: vec![delta_bytes, header_bytes, pos_bytes, exc_bytes],
                        log_num_values,
                    });

                    offset += chunk_bytes_len;
                }

                let compressed_size =
                    buf0.len() * std::mem::size_of::<u32>() + buf1.len() + buf2.len() + buf3.len();
                if compressed_size >= raw_size {
                    return ValueEncoder::default().compress(DataBlock::FixedWidth(data));
                }

                let encoding = ProtobufUtils21::alp(
                    self.bits_per_value as u32,
                    exponents.e as u32,
                    exponents.f as u32,
                );
                Ok((
                    MiniBlockCompressed {
                        data: vec![
                            LanceBuffer::reinterpret_vec(buf0),
                            LanceBuffer::from(buf1),
                            LanceBuffer::from(buf2),
                            LanceBuffer::from(buf3),
                        ],
                        chunks,
                        num_values: data.num_values,
                    },
                    encoding,
                ))
            }
            64 => {
                let words = data.data.borrow_to_typed_slice::<u64>();
                let floats = words
                    .as_ref()
                    .iter()
                    .map(|b| f64::from_bits(*b))
                    .collect::<Vec<_>>();
                let exponents = find_best_exponents_f64(&floats);

                let mut buf0: Vec<u64> = Vec::new();
                let mut buf1: Vec<u8> = Vec::new();
                let mut buf2: Vec<u8> = Vec::new();
                let mut buf3: Vec<u8> = Vec::new();
                let mut chunks = Vec::new();

                let mut offset = 0usize;
                let bytes = data.data.as_ref();
                while offset < bytes.len() {
                    let remaining_values = (bytes.len() - offset) / bytes_per_value;
                    let chunk_values = remaining_values.min(max_chunk);
                    let chunk_bytes_len = chunk_values * bytes_per_value;
                    let chunk_bytes = &bytes[offset..offset + chunk_bytes_len];

                    let words = bytemuck::try_cast_slice::<u8, u64>(chunk_bytes).map_err(|_| {
                        Error::invalid_input("invalid f64 buffer alignment", location!())
                    })?;
                    let floats = words.iter().map(|b| f64::from_bits(*b)).collect::<Vec<_>>();
                    let Ok(encoded) = encode_chunk_f64(&floats, exponents) else {
                        return ValueEncoder::default().compress(DataBlock::FixedWidth(data));
                    };

                    let delta_bytes = encoded
                        .packed_deltas
                        .len()
                        .checked_mul(std::mem::size_of::<u64>())
                        .ok_or_else(|| {
                            Error::invalid_input("ALP chunk size overflow", location!())
                        })? as u32;
                    let header_bytes = encoded.header.len() as u32;
                    let pos_bytes = (encoded.exception_positions.len() * 2) as u32;
                    let exc_bytes = encoded.exception_bits.len() as u32;

                    let total_value_bytes = delta_bytes as u64
                        + header_bytes as u64
                        + pos_bytes as u64
                        + exc_bytes as u64;
                    if total_value_bytes > MAX_MINIBLOCK_BYTES {
                        return ValueEncoder::default().compress(DataBlock::FixedWidth(data));
                    }

                    buf0.extend(encoded.packed_deltas);
                    buf1.extend_from_slice(&encoded.header);
                    for pos in encoded.exception_positions {
                        buf2.extend_from_slice(&pos.to_le_bytes());
                    }
                    buf3.extend_from_slice(&encoded.exception_bits);

                    let log_num_values = if offset + chunk_bytes_len == bytes.len() {
                        0
                    } else {
                        (chunk_values as u64).ilog2() as u8
                    };
                    chunks.push(MiniBlockChunk {
                        buffer_sizes: vec![delta_bytes, header_bytes, pos_bytes, exc_bytes],
                        log_num_values,
                    });

                    offset += chunk_bytes_len;
                }

                let compressed_size =
                    buf0.len() * std::mem::size_of::<u64>() + buf1.len() + buf2.len() + buf3.len();
                if compressed_size >= raw_size {
                    return ValueEncoder::default().compress(DataBlock::FixedWidth(data));
                }

                let encoding = ProtobufUtils21::alp(
                    self.bits_per_value as u32,
                    exponents.e as u32,
                    exponents.f as u32,
                );
                Ok((
                    MiniBlockCompressed {
                        data: vec![
                            LanceBuffer::reinterpret_vec(buf0),
                            LanceBuffer::from(buf1),
                            LanceBuffer::from(buf2),
                            LanceBuffer::from(buf3),
                        ],
                        chunks,
                        num_values: data.num_values,
                    },
                    encoding,
                ))
            }
            _ => unreachable!(),
        }
    }
}

struct EncodedChunk32 {
    packed_deltas: Vec<u32>,
    header: Vec<u8>,
    exception_positions: Vec<u16>,
    exception_bits: Vec<u8>,
}

struct EncodedChunk64 {
    packed_deltas: Vec<u64>,
    header: Vec<u8>,
    exception_positions: Vec<u16>,
    exception_bits: Vec<u8>,
}

#[cfg(feature = "bitpacking")]
fn encode_chunk_f32(values: &[f32], exp: Exponents) -> Result<EncodedChunk32> {
    let mut encoded = Vec::with_capacity(values.len());
    let mut exception_positions = Vec::new();
    let mut exception_bits = Vec::new();
    let mut fill: Option<i32> = None;

    for (i, &v) in values.iter().enumerate() {
        let bits = v.to_bits();
        if !v.is_finite() || bits == 0x8000_0000 {
            exception_positions.push(i as u16);
            exception_bits.extend_from_slice(&bits.to_le_bytes());
            encoded.push(0i32);
            continue;
        }

        let scaled = v * F10_F32[exp.e as usize] * IF10_F32[exp.f as usize];
        if !scaled.is_finite() {
            exception_positions.push(i as u16);
            exception_bits.extend_from_slice(&bits.to_le_bytes());
            encoded.push(0i32);
            continue;
        }
        let rounded = fast_round_f32(scaled);
        let enc = rounded as i32;
        let decoded = (enc as f32) * F10_F32[exp.f as usize] * IF10_F32[exp.e as usize];
        if decoded.to_bits() == bits {
            if fill.is_none() {
                fill = Some(enc);
            }
            encoded.push(enc);
        } else {
            exception_positions.push(i as u16);
            exception_bits.extend_from_slice(&bits.to_le_bytes());
            encoded.push(0i32);
        }
    }

    let Some(fill) = fill else {
        return Err(Error::invalid_input(
            "ALP chunk has no encodable values",
            location!(),
        ));
    };
    for &pos in &exception_positions {
        encoded[pos as usize] = fill;
    }

    let mut min = i32::MAX;
    let mut max = i32::MIN;
    for &v in &encoded {
        min = min.min(v);
        max = max.max(v);
    }
    let range = (max as i64 - min as i64) as u64;
    let bit_width = bits_required_u64(range);

    let deltas = encoded
        .iter()
        .map(|&v| v.wrapping_sub(min) as u32)
        .collect::<Vec<_>>();
    let deltas_block = FixedWidthDataBlock {
        data: LanceBuffer::reinterpret_vec(deltas),
        bits_per_value: 32,
        num_values: values.len() as u64,
        block_info: BlockInfo::new(),
    };
    let packed_deltas = bitpack_out_of_line_vec::<u32>(deltas_block, bit_width as usize);
    let mut header = Vec::with_capacity(5);
    header.extend_from_slice(&min.to_le_bytes());
    header.push(bit_width);

    Ok(EncodedChunk32 {
        packed_deltas,
        header,
        exception_positions,
        exception_bits,
    })
}

#[cfg(not(feature = "bitpacking"))]
fn encode_chunk_f32(_values: &[f32], _exp: Exponents) -> Result<EncodedChunk32> {
    Err(Error::NotSupported {
        source: "ALP requires bitpacking support".into(),
        location: location!(),
    })
}

#[cfg(feature = "bitpacking")]
fn encode_chunk_f64(values: &[f64], exp: Exponents) -> Result<EncodedChunk64> {
    let mut encoded = Vec::with_capacity(values.len());
    let mut exception_positions = Vec::new();
    let mut exception_bits = Vec::new();
    let mut fill: Option<i64> = None;

    for (i, &v) in values.iter().enumerate() {
        let bits = v.to_bits();
        if !v.is_finite() || bits == 0x8000_0000_0000_0000 {
            exception_positions.push(i as u16);
            exception_bits.extend_from_slice(&bits.to_le_bytes());
            encoded.push(0i64);
            continue;
        }
        let scaled = v * F10_F64[exp.e as usize] * IF10_F64[exp.f as usize];
        if !scaled.is_finite() {
            exception_positions.push(i as u16);
            exception_bits.extend_from_slice(&bits.to_le_bytes());
            encoded.push(0i64);
            continue;
        }
        let rounded = fast_round_f64(scaled);
        let enc = rounded as i64;
        let decoded = (enc as f64) * F10_F64[exp.f as usize] * IF10_F64[exp.e as usize];
        if decoded.to_bits() == bits {
            if fill.is_none() {
                fill = Some(enc);
            }
            encoded.push(enc);
        } else {
            exception_positions.push(i as u16);
            exception_bits.extend_from_slice(&bits.to_le_bytes());
            encoded.push(0i64);
        }
    }

    let Some(fill) = fill else {
        return Err(Error::invalid_input(
            "ALP chunk has no encodable values",
            location!(),
        ));
    };
    for &pos in &exception_positions {
        encoded[pos as usize] = fill;
    }

    let mut min = i64::MAX;
    let mut max = i64::MIN;
    for &v in &encoded {
        min = min.min(v);
        max = max.max(v);
    }
    let range = max.wrapping_sub(min) as u64;
    let bit_width = bits_required_u64(range);

    let deltas = encoded
        .iter()
        .map(|&v| v.wrapping_sub(min) as u64)
        .collect::<Vec<_>>();
    let deltas_block = FixedWidthDataBlock {
        data: LanceBuffer::reinterpret_vec(deltas),
        bits_per_value: 64,
        num_values: values.len() as u64,
        block_info: BlockInfo::new(),
    };
    let packed_deltas = bitpack_out_of_line_vec::<u64>(deltas_block, bit_width as usize);
    let mut header = Vec::with_capacity(9);
    header.extend_from_slice(&min.to_le_bytes());
    header.push(bit_width);

    Ok(EncodedChunk64 {
        packed_deltas,
        header,
        exception_positions,
        exception_bits,
    })
}

#[cfg(not(feature = "bitpacking"))]
fn encode_chunk_f64(_values: &[f64], _exp: Exponents) -> Result<EncodedChunk64> {
    Err(Error::NotSupported {
        source: "ALP requires bitpacking support".into(),
        location: location!(),
    })
}

/// Decodes values encoded by `AlpMiniBlockEncoder`.
///
/// See the module-level "Buffer layout" section for the expected buffer ordering and
/// per-chunk structure.
#[derive(Debug)]
pub struct AlpMiniBlockDecompressor {
    bits_per_value: u64,
    exp: Exponents,
}

impl AlpMiniBlockDecompressor {
    pub fn from_description(desc: &pb21::Alp) -> Result<Self> {
        let bits_per_value = desc.bits_per_value as u64;
        if bits_per_value != 32 && bits_per_value != 64 {
            return Err(Error::invalid_input(
                "ALP bits_per_value must be 32 or 64",
                location!(),
            ));
        }
        let exp = Exponents {
            e: desc.exponent_e as u8,
            f: desc.exponent_f as u8,
        };

        if exp.f >= exp.e {
            return Err(Error::invalid_input(
                "ALP requires exponent_f < exponent_e",
                location!(),
            ));
        }

        match bits_per_value {
            32 => {
                if exp.e as usize >= F10_F32.len() || exp.f as usize >= F10_F32.len() {
                    return Err(Error::invalid_input(
                        "ALP f32 exponents out of range",
                        location!(),
                    ));
                }
            }
            64 => {
                if exp.e as usize >= F10_F64.len() || exp.f as usize >= F10_F64.len() {
                    return Err(Error::invalid_input(
                        "ALP f64 exponents out of range",
                        location!(),
                    ));
                }
            }
            _ => unreachable!(),
        }

        Ok(Self {
            bits_per_value,
            exp,
        })
    }
}

#[cfg(feature = "bitpacking")]
impl MiniBlockDecompressor for AlpMiniBlockDecompressor {
    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        if num_values == 0 {
            return Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
                data: LanceBuffer::empty(),
                bits_per_value: self.bits_per_value,
                num_values: 0,
                block_info: BlockInfo::new(),
            }));
        }

        if data.len() != 4 {
            return Err(Error::invalid_input(
                format!("ALP decompression expects 4 buffers, got {}", data.len()),
                location!(),
            ));
        }

        let n = usize::try_from(num_values)
            .map_err(|_| Error::invalid_input("ALP chunk too large for usize", location!()))?;

        let mut iter = data.into_iter();
        let packed_deltas = iter.next().unwrap();
        let header = iter.next().unwrap();
        let exception_positions = iter.next().unwrap();
        let exception_bits = iter.next().unwrap();

        let header_bytes = header.as_ref();
        let (min, bit_width) = match self.bits_per_value {
            32 => {
                if header_bytes.len() != 5 {
                    return Err(Error::invalid_input(
                        "ALP header buffer size mismatch",
                        location!(),
                    ));
                }
                let min = i32::from_le_bytes([
                    header_bytes[0],
                    header_bytes[1],
                    header_bytes[2],
                    header_bytes[3],
                ]) as i64;
                let bit_width = header_bytes[4];
                if bit_width > 32 {
                    return Err(Error::invalid_input(
                        "ALP bit width out of range for f32",
                        location!(),
                    ));
                }
                (min, bit_width)
            }
            64 => {
                if header_bytes.len() != 9 {
                    return Err(Error::invalid_input(
                        "ALP header buffer size mismatch",
                        location!(),
                    ));
                }
                let min = i64::from_le_bytes([
                    header_bytes[0],
                    header_bytes[1],
                    header_bytes[2],
                    header_bytes[3],
                    header_bytes[4],
                    header_bytes[5],
                    header_bytes[6],
                    header_bytes[7],
                ]);
                let bit_width = header_bytes[8];
                if bit_width > 64 {
                    return Err(Error::invalid_input(
                        "ALP bit width out of range for f64",
                        location!(),
                    ));
                }
                (min, bit_width)
            }
            _ => unreachable!(),
        };

        let mut out_bytes = Vec::with_capacity(n * (self.bits_per_value as usize / 8));
        match self.bits_per_value {
            32 => {
                let packed_block = FixedWidthDataBlock {
                    data: packed_deltas,
                    bits_per_value: 32,
                    num_values,
                    block_info: BlockInfo::new(),
                };
                let decompressed = unpack_out_of_line::<u32>(packed_block, n, bit_width as usize);
                let deltas = decompressed.data.borrow_to_typed_slice::<u32>();
                for &d in deltas.as_ref() {
                    let enc = (min + d as i64) as i32;
                    let decoded =
                        (enc as f32) * F10_F32[self.exp.f as usize] * IF10_F32[self.exp.e as usize];
                    out_bytes.extend_from_slice(&decoded.to_bits().to_le_bytes());
                }
            }
            64 => {
                let packed_block = FixedWidthDataBlock {
                    data: packed_deltas,
                    bits_per_value: 64,
                    num_values,
                    block_info: BlockInfo::new(),
                };
                let decompressed = unpack_out_of_line::<u64>(packed_block, n, bit_width as usize);
                let deltas = decompressed.data.borrow_to_typed_slice::<u64>();
                for &d in deltas.as_ref() {
                    let enc = (min as i128 + d as i128) as i64;
                    let decoded =
                        (enc as f64) * F10_F64[self.exp.f as usize] * IF10_F64[self.exp.e as usize];
                    out_bytes.extend_from_slice(&decoded.to_bits().to_le_bytes());
                }
            }
            _ => unreachable!(),
        }

        let pos_bytes = exception_positions.as_ref();
        let exc_bytes = exception_bits.as_ref();

        if pos_bytes.len() % 2 != 0 {
            return Err(Error::invalid_input(
                "ALP exception positions not u16-aligned",
                location!(),
            ));
        }
        let exception_count = pos_bytes.len() / 2;
        let value_bytes = (self.bits_per_value / 8) as usize;
        if exc_bytes.len() != exception_count * value_bytes {
            return Err(Error::invalid_input(
                "ALP exception values length mismatch",
                location!(),
            ));
        }

        for i in 0..exception_count {
            let pos = u16::from_le_bytes([pos_bytes[i * 2], pos_bytes[i * 2 + 1]]) as usize;
            if pos >= n {
                return Err(Error::invalid_input(
                    "ALP exception position out of range",
                    location!(),
                ));
            }
            let start = i * value_bytes;
            let end = start + value_bytes;
            out_bytes[pos * value_bytes..(pos + 1) * value_bytes]
                .copy_from_slice(&exc_bytes[start..end]);
        }

        Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(out_bytes),
            bits_per_value: self.bits_per_value,
            num_values,
            block_info: BlockInfo::new(),
        }))
    }
}

#[cfg(not(feature = "bitpacking"))]
impl MiniBlockDecompressor for AlpMiniBlockDecompressor {
    fn decompress(&self, _data: Vec<LanceBuffer>, _num_values: u64) -> Result<DataBlock> {
        Err(Error::NotSupported {
            source: "ALP requires bitpacking support".into(),
            location: location!(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::ComputeStat;

    fn round_trip_f32(values: &[f32]) {
        let bytes = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let mut block = FixedWidthDataBlock {
            data: LanceBuffer::from(bytes),
            bits_per_value: 32,
            num_values: values.len() as u64,
            block_info: BlockInfo::new(),
        };
        block.compute_stat();
        let encoder = AlpMiniBlockEncoder::new(32);
        let (compressed, encoding) = encoder.compress(DataBlock::FixedWidth(block)).unwrap();
        let pb21::compressive_encoding::Compression::Alp(desc) = encoding.compression.unwrap()
        else {
            panic!("expected ALP encoding")
        };
        assert_eq!(compressed.data.len(), 4);
        let decompressor = AlpMiniBlockDecompressor::from_description(&desc).unwrap();

        let mut vals_in_prev = 0u64;
        let mut buffer_offsets = vec![0usize; compressed.data.len()];
        let mut out = Vec::new();
        for chunk in &compressed.chunks {
            let chunk_vals = chunk.num_values(vals_in_prev, compressed.num_values);
            vals_in_prev += chunk_vals;
            let buffers = chunk
                .buffer_sizes
                .iter()
                .zip(compressed.data.iter().zip(buffer_offsets.iter_mut()))
                .map(|(sz, (buf, off))| {
                    let start = *off;
                    let end = start + *sz as usize;
                    *off = end;
                    buf.slice_with_length(start, *sz as usize)
                })
                .collect::<Vec<_>>();

            let decoded = decompressor.decompress(buffers, chunk_vals).unwrap();
            let DataBlock::FixedWidth(decoded) = decoded else {
                panic!("expected fixed width")
            };
            let words = decoded.data.borrow_to_typed_slice::<u32>();
            out.extend(words.as_ref().iter().map(|b| f32::from_bits(*b)));
        }

        assert_eq!(out.len(), values.len());
        for (a, b) in out.iter().zip(values.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    fn round_trip_f64(values: &[f64]) {
        let bytes = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let mut block = FixedWidthDataBlock {
            data: LanceBuffer::from(bytes),
            bits_per_value: 64,
            num_values: values.len() as u64,
            block_info: BlockInfo::new(),
        };
        block.compute_stat();
        let encoder = AlpMiniBlockEncoder::new(64);
        let (compressed, encoding) = encoder.compress(DataBlock::FixedWidth(block)).unwrap();
        let pb21::compressive_encoding::Compression::Alp(desc) = encoding.compression.unwrap()
        else {
            panic!("expected ALP encoding")
        };
        assert_eq!(compressed.data.len(), 4);
        let decompressor = AlpMiniBlockDecompressor::from_description(&desc).unwrap();

        let mut vals_in_prev = 0u64;
        let mut buffer_offsets = vec![0usize; compressed.data.len()];
        let mut out = Vec::new();
        for chunk in &compressed.chunks {
            let chunk_vals = chunk.num_values(vals_in_prev, compressed.num_values);
            vals_in_prev += chunk_vals;
            let buffers = chunk
                .buffer_sizes
                .iter()
                .zip(compressed.data.iter().zip(buffer_offsets.iter_mut()))
                .map(|(sz, (buf, off))| {
                    let start = *off;
                    let end = start + *sz as usize;
                    *off = end;
                    buf.slice_with_length(start, *sz as usize)
                })
                .collect::<Vec<_>>();

            let decoded = decompressor.decompress(buffers, chunk_vals).unwrap();
            let DataBlock::FixedWidth(decoded) = decoded else {
                panic!("expected fixed width")
            };
            let words = decoded.data.borrow_to_typed_slice::<u64>();
            out.extend(words.as_ref().iter().map(|b| f64::from_bits(*b)));
        }

        assert_eq!(out.len(), values.len());
        for (a, b) in out.iter().zip(values.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn test_round_trip_with_exceptions_f32() {
        let mut values = (0..1024).map(|v| v as f32).collect::<Vec<_>>();
        values[3] = -0.0;
        values[7] = f32::from_bits(0x7FC0_0001);
        values[11] = f32::INFINITY;
        round_trip_f32(&values);
    }

    #[test]
    fn test_round_trip_with_exceptions_f64() {
        let mut values = (0..512).map(|v| v as f64).collect::<Vec<_>>();
        values[2] = -0.0;
        values[5] = f64::from_bits(0x7FF8_0000_0000_0001);
        values[9] = f64::NEG_INFINITY;
        round_trip_f64(&values);
    }

    #[test]
    fn test_fallback_when_not_beneficial() {
        let values = vec![-0.0f32; 1024];
        let bytes = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let mut block = FixedWidthDataBlock {
            data: LanceBuffer::from(bytes),
            bits_per_value: 32,
            num_values: values.len() as u64,
            block_info: BlockInfo::new(),
        };
        block.compute_stat();
        let encoder = AlpMiniBlockEncoder::new(32);
        let (_compressed, encoding) = encoder.compress(DataBlock::FixedWidth(block)).unwrap();
        assert!(matches!(
            encoding.compression.unwrap(),
            pb21::compressive_encoding::Compression::Flat(_)
        ));
    }
}
