// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Split Block Bloom Filter (SBBF) implementation for Lance
//!
//! Based on the Apache Arrow Parquet SBBF implementation but with public APIs
//! for use in Lance indexing. This implementation follows the Parquet spec
//! https://github.com/apache/arrow-rs/blob/main/parquet/src/bloom_filter/mod.rs
//! for SBBF as described in https://github.com/apache/parquet-format/blob/master/BloomFilter.md
//! FIXME: Make the upstream SBBF implementation public so that this file could be
//! removed from Lance.
//! https://github.com/apache/arrow-rs/issues/8277

use crate::scalar::bloomfilter::as_bytes::AsBytes;
use std::error::Error;
use std::fmt;
use std::io::Write;
use twox_hash::XxHash64;

#[derive(Debug)]
pub enum SbbfError {
    InvalidFpp { fpp: f64 },
    WriteError { source: std::io::Error },
    InvalidData { message: String },
}

impl fmt::Display for SbbfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SbbfError::InvalidFpp { fpp } => {
                write!(
                    f,
                    "False positive probability must be between 0.0 and 1.0, got {}",
                    fpp
                )
            }
            SbbfError::WriteError { source } => {
                write!(f, "Failed to write bloom filter: {}", source)
            }
            SbbfError::InvalidData { message } => {
                write!(f, "Invalid bloom filter data: {}", message)
            }
        }
    }
}

impl Error for SbbfError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            SbbfError::WriteError { source } => Some(source),
            _ => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, SbbfError>;

/// Salt as defined in the Parquet spec
const SALT: [u32; 8] = [
    0x47b6137b_u32,
    0x44974d91_u32,
    0x8824ad5b_u32,
    0xa2b7289d_u32,
    0x705495c7_u32,
    0x2df1424b_u32,
    0x9efc4947_u32,
    0x5c6bfb31_u32,
];

/// Each block is 256 bits, broken up into eight contiguous "words", each consisting of 32 bits.
/// Each word is thought of as an array of bits; each bit is either "set" or "not set".
#[derive(Debug, Copy, Clone)]
struct Block([u32; 8]);

impl Block {
    const ZERO: Block = Block([0; 8]);

    /// Takes as its argument a single unsigned 32-bit integer and returns a block in which each
    /// word has exactly one bit set.
    fn mask(x: u32) -> Self {
        let mut result = [0_u32; 8];
        for i in 0..8 {
            // wrapping instead of checking for overflow
            let y = x.wrapping_mul(SALT[i]);
            let y = y >> 27;
            result[i] = 1 << y;
        }
        Self(result)
    }

    #[inline]
    #[cfg(target_endian = "little")]
    fn to_le_bytes(self) -> [u8; 32] {
        self.to_ne_bytes()
    }

    #[inline]
    #[cfg(not(target_endian = "little"))]
    fn to_le_bytes(self) -> [u8; 32] {
        self.swap_bytes().to_ne_bytes()
    }

    #[inline]
    fn to_ne_bytes(self) -> [u8; 32] {
        // SAFETY: [u32; 8] and [u8; 32] have the same size and neither has invalid bit patterns.
        unsafe { std::mem::transmute(self.0) }
    }

    #[inline]
    #[cfg(not(target_endian = "little"))]
    fn swap_bytes(mut self) -> Self {
        self.0.iter_mut().for_each(|x| *x = x.swap_bytes());
        self
    }

    /// Setting every bit in the block that was also set in the result from mask
    fn insert(&mut self, hash: u32) {
        let mask = Self::mask(hash);
        for i in 0..8 {
            self[i] |= mask[i];
        }
    }

    /// Returns true when every bit that is set in the result of mask is also set in the block.
    fn check(&self, hash: u32) -> bool {
        let mask = Self::mask(hash);
        for i in 0..8 {
            if self[i] & mask[i] == 0 {
                return false;
            }
        }
        true
    }
}

impl std::ops::Index<usize> for Block {
    type Output = u32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl std::ops::IndexMut<usize> for Block {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

/// Minimum and maximum bitset lengths
pub const BITSET_MIN_LENGTH: usize = 32;
pub const BITSET_MAX_LENGTH: usize = 128 * 1024 * 1024;

#[inline]
fn optimal_num_of_bytes(num_bytes: usize) -> usize {
    let num_bytes = num_bytes.min(BITSET_MAX_LENGTH);
    let num_bytes = num_bytes.max(BITSET_MIN_LENGTH);
    num_bytes.next_power_of_two()
}

/// Calculate number of bits needed for given NDV and FPP
/// Formula: m = -k * n / ln(1 - fpp^(1/k))
/// Where k=8 (number of hash functions), n=NDV, fpp=false positive probability
#[inline]
fn num_of_bits_from_ndv_fpp(ndv: u64, fpp: f64) -> usize {
    let num_bits = -8.0 * ndv as f64 / (1.0 - fpp.powf(1.0 / 8.0)).ln();
    num_bits as usize
}

/// A Split Block Bloom Filter (SBBF) implementation
///
/// This is a high-performance bloom filter optimized for SIMD operations,
/// compatible with the Parquet specification.
#[derive(Debug, Clone)]
pub struct Sbbf {
    blocks: Vec<Block>,
}

impl Sbbf {
    /// Create a new SBBF from raw bitset data
    pub fn new(bitset: &[u8]) -> Result<Self> {
        if bitset.len() % 32 != 0 {
            return Err(SbbfError::InvalidData {
                message: format!(
                    "Bitset length must be a multiple of 32, got {}",
                    bitset.len()
                ),
            });
        }

        let data = bitset
            .chunks_exact(4 * 8)
            .map(|chunk| {
                let mut block = Block::ZERO;
                for (i, word) in chunk.chunks_exact(4).enumerate() {
                    block[i] = u32::from_le_bytes(word.try_into().unwrap());
                }
                block
            })
            .collect::<Vec<Block>>();

        Ok(Self { blocks: data })
    }

    /// Create a new empty SBBF with the given number of bytes
    /// The actual size will be adjusted to the next power of two within bounds
    pub fn with_num_bytes(num_bytes: usize) -> Self {
        let num_bytes = optimal_num_of_bytes(num_bytes);
        let bitset = vec![0_u8; num_bytes];
        // unwrap is safe because we know the size is valid
        Self::new(&bitset).unwrap()
    }

    /// Create a new SBBF with given number of distinct values and false positive probability
    pub fn with_ndv_fpp(ndv: u64, fpp: f64) -> Result<Self> {
        if !(0.0..1.0).contains(&fpp) {
            return Err(SbbfError::InvalidFpp { fpp });
        }
        let num_bits = num_of_bits_from_ndv_fpp(ndv, fpp);
        Ok(Self::with_num_bytes(num_bits / 8))
    }

    /// Get the hash-to-block-index for a given hash
    #[inline]
    fn hash_to_block_index(&self, hash: u64) -> usize {
        (((hash >> 32).saturating_mul(self.blocks.len() as u64)) >> 32) as usize
    }

    /// Insert an AsBytes value into the filter
    pub fn insert<T: AsBytes + ?Sized>(&mut self, value: &T) {
        self.insert_hash(hash_as_bytes(value));
    }

    /// Insert a hash into the filter
    pub fn insert_hash(&mut self, hash: u64) {
        let block_index = self.hash_to_block_index(hash);
        self.blocks[block_index].insert(hash as u32)
    }

    /// Check if an AsBytes value is probably present or definitely absent in the filter
    pub fn check<T: AsBytes + ?Sized>(&self, value: &T) -> bool {
        self.check_hash(hash_as_bytes(value))
    }

    /// Check if a hash is in the filter. May return
    /// true for values that were never inserted ("false positive")
    /// but will always return false if a hash has not been inserted.
    pub fn check_hash(&self, hash: u64) -> bool {
        let block_index = self.hash_to_block_index(hash);
        self.blocks[block_index].check(hash as u32)
    }

    /// Write the bitset in serialized form to the writer
    pub fn write_bitset<W: Write>(&self, mut writer: W) -> Result<()> {
        for block in &self.blocks {
            writer
                .write_all(block.to_le_bytes().as_slice())
                .map_err(|source| SbbfError::WriteError { source })?;
        }
        Ok(())
    }

    /// Get the raw bitset as bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(self.blocks.len() * 32);
        for block in &self.blocks {
            result.extend_from_slice(&block.to_le_bytes());
        }
        result
    }

    /// Get the number of blocks in this filter
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get the size in bytes of this filter
    pub fn size_bytes(&self) -> usize {
        self.blocks.len() * 32
    }

    /// Return the total in memory size of this bloom filter in bytes
    pub fn estimated_memory_size(&self) -> usize {
        self.blocks.capacity() * std::mem::size_of::<Block>()
    }
}

// Per spec we use xxHash with seed=0
const SEED: u64 = 0;

#[inline]
fn hash_as_bytes<A: AsBytes + ?Sized>(value: &A) -> u64 {
    XxHash64::oneshot(SEED, &value.as_bytes())
}

/// Builder for creating SBBF instances with a fluent API
pub struct SbbfBuilder {
    ndv: Option<u64>,
    fpp: Option<f64>,
    num_bytes: Option<usize>,
}

impl SbbfBuilder {
    /// Create a new SBBF builder
    pub fn new() -> Self {
        Self {
            ndv: None,
            fpp: None,
            num_bytes: None,
        }
    }

    /// Set the expected number of distinct values
    pub fn expected_items(mut self, ndv: u64) -> Self {
        self.ndv = Some(ndv);
        self
    }

    /// Set the desired false positive probability
    pub fn false_positive_probability(mut self, fpp: f64) -> Self {
        self.fpp = Some(fpp);
        self
    }

    /// Set the number of bytes directly
    pub fn num_bytes(mut self, num_bytes: usize) -> Self {
        self.num_bytes = Some(num_bytes);
        self
    }

    /// Build the SBBF
    pub fn build(self) -> Result<Sbbf> {
        if let Some(num_bytes) = self.num_bytes {
            Ok(Sbbf::with_num_bytes(num_bytes))
        } else if let (Some(ndv), Some(fpp)) = (self.ndv, self.fpp) {
            Sbbf::with_ndv_fpp(ndv, fpp)
        } else {
            Err(SbbfError::InvalidData {
                message: "Must specify either num_bytes or both ndv and fpp".to_string(),
            })
        }
    }
}

impl Default for SbbfBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_bytes() {
        assert_eq!(hash_as_bytes(""), 17241709254077376921);
    }

    #[test]
    fn test_mask_set_quick_check() {
        for i in 0..1_000 {
            let result = Block::mask(i);
            assert!(result.0.iter().all(|&x| x.is_power_of_two()));
        }
    }

    #[test]
    fn test_block_insert_and_check() {
        for i in 0..1_000 {
            let mut block = Block::ZERO;
            block.insert(i);
            assert!(block.check(i));
        }
    }

    #[test]
    fn test_sbbf_insert_and_check() {
        let mut sbbf = Sbbf::with_num_bytes(1024);
        for i in 0..1_000 {
            sbbf.insert(&i);
            assert!(sbbf.check(&i));
        }
    }

    #[test]
    fn test_sbbf_builder() {
        let sbbf = SbbfBuilder::new()
            .expected_items(1000)
            .false_positive_probability(0.01)
            .build()
            .unwrap();

        assert!(sbbf.num_blocks() > 0);
    }

    #[test]
    fn test_sbbf_string_types() {
        let mut sbbf = SbbfBuilder::new()
            .expected_items(100)
            .false_positive_probability(0.01)
            .build()
            .unwrap();

        // Test different string types
        let string_val = "hello";
        let str_val = "world";
        let bytes_val = b"bytes";

        sbbf.insert(string_val);
        sbbf.insert(str_val);
        sbbf.insert(&bytes_val[..]);

        assert!(sbbf.check(string_val));
        assert!(sbbf.check(str_val));
        assert!(sbbf.check(&bytes_val[..]));
        assert!(!sbbf.check("not_inserted"));
    }

    #[test]
    fn test_sbbf_numeric_types() {
        let mut sbbf = SbbfBuilder::new()
            .expected_items(100)
            .false_positive_probability(0.01)
            .build()
            .unwrap();

        // Test different numeric types
        let i32_val = 42i32;
        let i64_val = 12345i64;
        let f64_val = 3.14f64;
        let bool_val = true;

        sbbf.insert(&i32_val);
        sbbf.insert(&i64_val);
        sbbf.insert(&f64_val);
        sbbf.insert(&bool_val);

        assert!(sbbf.check(&i32_val));
        assert!(sbbf.check(&i64_val));
        assert!(sbbf.check(&f64_val));
        assert!(sbbf.check(&bool_val));
        assert!(!sbbf.check(&999i32));
    }

    #[test]
    fn test_optimal_num_of_bytes() {
        for (input, expected) in &[
            (0, 32),
            (9, 32),
            (31, 32),
            (32, 32),
            (33, 64),
            (99, 128),
            (1024, 1024),
            (999_000_000, 128 * 1024 * 1024),
        ] {
            assert_eq!(*expected, optimal_num_of_bytes(*input));
        }
    }

    #[test]
    fn test_num_of_bits_from_ndv_fpp() {
        for (fpp, ndv, num_bits) in &[
            (0.1, 10, 57),
            (0.01, 10, 96),
            (0.001, 10, 146),
            (0.1, 100, 577),
            (0.01, 100, 968),
            (0.001, 100, 1460),
            (0.1, 1000, 5772),
            (0.01, 1000, 9681),
            (0.001, 1000, 14607),
        ] {
            assert_eq!(*num_bits, num_of_bits_from_ndv_fpp(*ndv, *fpp) as u64);
        }
    }

    #[test]
    fn test_serialization() {
        let mut sbbf = SbbfBuilder::new()
            .expected_items(100)
            .false_positive_probability(0.01)
            .build()
            .unwrap();

        // Insert some values
        for i in 0..50 {
            sbbf.insert(&i);
        }

        // Serialize to bytes
        let bytes = sbbf.to_bytes();
        assert!(!bytes.is_empty());
        assert_eq!(bytes.len(), sbbf.size_bytes());

        // Deserialize from bytes
        let sbbf2 = Sbbf::new(&bytes).unwrap();
        assert_eq!(sbbf.num_blocks(), sbbf2.num_blocks());

        // Check that deserialized filter works
        for i in 0..50 {
            assert!(sbbf2.check(&i));
        }
        assert!(!sbbf2.check(&999));
    }
}
