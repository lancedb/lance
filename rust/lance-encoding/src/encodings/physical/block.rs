// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Encodings based on traditional block compression schemes
//!
//! Traditional compressors take in a buffer and return a smaller buffer.  All encoding
//! description is shoved into the compressed buffer and the entire buffer is needed to
//! decompress any of the data.
//!
//! These encodings are not transparent, which limits our ability to use them.  In addition
//! they are often quite expensive in CPU terms.
//!
//! However, they are effective and useful for some cases.  For example, when working with large
//! variable length values (e.g. source code files) they can be very effective.
//!
//! The module introduces the `[BufferCompressor]` trait which describes the interface for a
//! traditional block compressor.  It is implemented for the most common compression schemes
//! (zstd, lz4, etc).
//!
//! There is not yet a mini-block variant of this compressor (but could easily be one) and the
//! full zip variant works by applying compression on a per-value basis (which allows it to be
//! transparent).

use arrow_buffer::ArrowNativeType;
use lance_core::{Error, Result};
use snafu::location;

use std::str::FromStr;

use crate::format::pb21::{self, CompressiveEncoding};
use crate::format::ProtobufUtils21;
use crate::{
    buffer::LanceBuffer,
    compression::VariablePerValueDecompressor,
    data::{BlockInfo, DataBlock, VariableWidthBlock},
    encodings::logical::primitive::fullzip::{PerValueCompressor, PerValueDataBlock},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressionConfig {
    pub(crate) scheme: CompressionScheme,
    pub(crate) level: Option<i32>,
}

impl CompressionConfig {
    pub(crate) fn new(scheme: CompressionScheme, level: Option<i32>) -> Self {
        Self { scheme, level }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            scheme: CompressionScheme::Lz4,
            level: Some(0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionScheme {
    None,
    Fsst,
    Zstd,
    Lz4,
}

impl TryFrom<CompressionScheme> for pb21::CompressionScheme {
    type Error = Error;

    fn try_from(scheme: CompressionScheme) -> Result<Self> {
        match scheme {
            CompressionScheme::Lz4 => Ok(Self::CompressionAlgorithmLz4),
            CompressionScheme::Zstd => Ok(Self::CompressionAlgorithmZstd),
            _ => Err(Error::invalid_input(
                format!("Unsupported compression scheme: {:?}", scheme),
                location!(),
            )),
        }
    }
}

impl TryFrom<pb21::CompressionScheme> for CompressionScheme {
    type Error = Error;

    fn try_from(scheme: pb21::CompressionScheme) -> Result<Self> {
        match scheme {
            pb21::CompressionScheme::CompressionAlgorithmLz4 => Ok(Self::Lz4),
            pb21::CompressionScheme::CompressionAlgorithmZstd => Ok(Self::Zstd),
            _ => Err(Error::invalid_input(
                format!("Unsupported compression scheme: {:?}", scheme),
                location!(),
            )),
        }
    }
}

impl std::fmt::Display for CompressionScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let scheme_str = match self {
            Self::Fsst => "fsst",
            Self::Zstd => "zstd",
            Self::None => "none",
            Self::Lz4 => "lz4",
        };
        write!(f, "{}", scheme_str)
    }
}

impl FromStr for CompressionScheme {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "none" => Ok(Self::None),
            "fsst" => Ok(Self::Fsst),
            "zstd" => Ok(Self::Zstd),
            "lz4" => Ok(Self::Lz4),
            _ => Err(Error::invalid_input(
                format!("Unknown compression scheme: {}", s),
                location!(),
            )),
        }
    }
}

pub trait BufferCompressor: std::fmt::Debug + Send + Sync {
    fn compress(&self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()>;
    fn decompress(&self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()>;
    fn config(&self) -> CompressionConfig;
}

#[cfg(feature = "zstd")]
mod zstd {
    use std::io::{Cursor, Write};

    use super::*;

    use ::zstd::bulk::decompress_to_buffer;
    use ::zstd::stream::copy_decode;

    #[derive(Debug, Default)]
    pub struct ZstdBufferCompressor {
        compression_level: i32,
    }

    impl ZstdBufferCompressor {
        pub fn new(compression_level: i32) -> Self {
            Self { compression_level }
        }

        // https://datatracker.ietf.org/doc/html/rfc8878
        fn is_raw_stream_format(&self, input_buf: &[u8]) -> bool {
            if input_buf.len() < 8 {
                return true; // can't be length prefixed format if less than 8 bytes
            }
            // read the first 4 bytes as the magic number
            let mut magic_buf = [0u8; 4];
            magic_buf.copy_from_slice(&input_buf[..4]);
            let magic = u32::from_le_bytes(magic_buf);

            // see RFC 8878, section 3.1.1. Zstandard Frames, which defines the magic number
            const ZSTD_MAGIC_NUMBER: u32 = 0xFD2FB528;
            if magic == ZSTD_MAGIC_NUMBER {
                // the compressed buffer starts like a Zstd frame.
                // Per RFC 8878, the reserved bit (with Bit Number 3, the 4th bit) in the FHD (frame header descriptor) MUST be 0
                // see section 3.1.1.1.1. 'Frame_Header_Descriptor' and section 3.1.1.1.1.4. 'Reserved Bit' for details
                const FHD_BYTE_INDEX: usize = 4;
                let fhd_byte = input_buf[FHD_BYTE_INDEX];
                const FHD_RESERVED_BIT_MASK: u8 = 0b0001_0000;
                let reserved_bit = fhd_byte & FHD_RESERVED_BIT_MASK;

                if reserved_bit != 0 {
                    // this bit is 1. This is NOT a valid zstd frame.
                    // therefore, it must be length prefixed format where the length coincidentally
                    // started with the magic number
                    false
                } else {
                    // the reserved bit is 0. This is consistent with a valid Zstd frame.
                    // treat it as raw stream format
                    true
                }
            } else {
                // doesn't start with the magic number, so it can't be the raw stream format
                false
            }
        }

        fn decompress_length_prefixed_zstd(
            &self,
            input_buf: &[u8],
            output_buf: &mut Vec<u8>,
        ) -> Result<()> {
            const LENGTH_PREFIX_SIZE: usize = 8;
            let mut len_buf = [0u8; LENGTH_PREFIX_SIZE];
            len_buf.copy_from_slice(&input_buf[..LENGTH_PREFIX_SIZE]);

            let uncompressed_len = u64::from_le_bytes(len_buf) as usize;

            let start = output_buf.len();
            output_buf.resize(start + uncompressed_len, 0);

            let compressed_data = &input_buf[LENGTH_PREFIX_SIZE..];
            decompress_to_buffer(compressed_data, &mut output_buf[start..])?;
            Ok(())
        }
    }

    impl BufferCompressor for ZstdBufferCompressor {
        fn compress(&self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()> {
            output_buf.write_all(&(input_buf.len() as u64).to_le_bytes())?;
            let mut encoder = ::zstd::stream::Encoder::new(output_buf, self.compression_level)?;

            encoder.write_all(input_buf)?;
            match encoder.finish() {
                Ok(_) => Ok(()),
                Err(e) => Err(e.into()),
            }
        }

        fn decompress(&self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()> {
            if input_buf.is_empty() {
                return Ok(());
            }

            let is_raw_stream_format = self.is_raw_stream_format(input_buf);
            if is_raw_stream_format {
                copy_decode(Cursor::new(input_buf), output_buf)?;
            } else {
                self.decompress_length_prefixed_zstd(input_buf, output_buf)?;
            }

            Ok(())
        }

        fn config(&self) -> CompressionConfig {
            CompressionConfig {
                scheme: CompressionScheme::Zstd,
                level: Some(self.compression_level),
            }
        }
    }
}

#[cfg(feature = "lz4")]
mod lz4 {
    use super::*;

    #[derive(Debug, Default)]
    pub struct Lz4BufferCompressor {}

    impl BufferCompressor for Lz4BufferCompressor {
        fn compress(&self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()> {
            // Remember the starting position
            let start_pos = output_buf.len();

            // LZ4 needs space for the compressed data
            let max_size = ::lz4::block::compress_bound(input_buf.len())?;
            // Resize to ensure we have enough space (including 4 bytes for size header)
            output_buf.resize(start_pos + max_size + 4, 0);

            let compressed_size = ::lz4::block::compress_to_buffer(
                input_buf,
                None,
                true,
                &mut output_buf[start_pos..],
            )
            .map_err(|err| Error::Internal {
                message: format!("LZ4 compression error: {}", err),
                location: location!(),
            })?;

            // Truncate to actual size
            output_buf.truncate(start_pos + compressed_size);
            Ok(())
        }

        fn decompress(&self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()> {
            // When prepend_size is true, LZ4 stores the uncompressed size in the first 4 bytes
            // We can read this to know exactly how much space we need
            if input_buf.len() < 4 {
                return Err(Error::Internal {
                    message: "LZ4 compressed data too short".to_string(),
                    location: location!(),
                });
            }

            // Read the uncompressed size from the first 4 bytes (little-endian)
            let uncompressed_size =
                u32::from_le_bytes([input_buf[0], input_buf[1], input_buf[2], input_buf[3]])
                    as usize;

            // Remember the starting position
            let start_pos = output_buf.len();

            // Resize to ensure we have the exact space needed
            output_buf.resize(start_pos + uncompressed_size, 0);

            // Now decompress directly into the buffer slice
            let decompressed_size =
                ::lz4::block::decompress_to_buffer(input_buf, None, &mut output_buf[start_pos..])
                    .map_err(|err| Error::Internal {
                    message: format!("LZ4 decompression error: {}", err),
                    location: location!(),
                })?;

            // Truncate to actual decompressed size (should be same as uncompressed_size)
            output_buf.truncate(start_pos + decompressed_size);

            Ok(())
        }

        fn config(&self) -> CompressionConfig {
            CompressionConfig {
                scheme: CompressionScheme::Lz4,
                level: None,
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct NoopBufferCompressor {}

impl BufferCompressor for NoopBufferCompressor {
    fn compress(&self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()> {
        output_buf.extend_from_slice(input_buf);
        Ok(())
    }

    fn decompress(&self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()> {
        output_buf.extend_from_slice(input_buf);
        Ok(())
    }

    fn config(&self) -> CompressionConfig {
        CompressionConfig {
            scheme: CompressionScheme::None,
            level: None,
        }
    }
}

pub struct GeneralBufferCompressor {}

impl GeneralBufferCompressor {
    pub fn get_compressor(
        compression_config: CompressionConfig,
    ) -> Result<Box<dyn BufferCompressor>> {
        match compression_config.scheme {
            // FSST has its own compression path and isn't implemented as a generic buffer compressor
            CompressionScheme::Fsst => Err(Error::InvalidInput {
                source: "fsst is not usable as a general buffer compressor".into(),
                location: location!(),
            }),
            CompressionScheme::Zstd => {
                #[cfg(feature = "zstd")]
                {
                    Ok(Box::new(zstd::ZstdBufferCompressor::new(
                        compression_config.level.unwrap_or(0),
                    )))
                }
                #[cfg(not(feature = "zstd"))]
                {
                    Err(Error::InvalidInput {
                        source: "package was not built with zstd support".into(),
                        location: location!(),
                    })
                }
            }
            CompressionScheme::Lz4 => {
                #[cfg(feature = "lz4")]
                {
                    Ok(Box::new(lz4::Lz4BufferCompressor::default()))
                }
                #[cfg(not(feature = "lz4"))]
                {
                    Err(Error::InvalidInput {
                        source: "package was not built with lz4 support".into(),
                        location: location!(),
                    })
                }
            }
            CompressionScheme::None => Ok(Box::new(NoopBufferCompressor {})),
        }
    }
}

// An encoder which uses generic compression, such as zstd/lz4 to encode buffers
#[derive(Debug)]
pub struct CompressedBufferEncoder {
    pub(crate) compressor: Box<dyn BufferCompressor>,
}

impl Default for CompressedBufferEncoder {
    fn default() -> Self {
        // Pick zstd if available, otherwise lz4, otherwise none
        #[cfg(feature = "zstd")]
        let (scheme, level) = (CompressionScheme::Zstd, Some(0));
        #[cfg(all(feature = "lz4", not(feature = "zstd")))]
        let (scheme, level) = (CompressionScheme::Lz4, None);
        #[cfg(not(any(feature = "zstd", feature = "lz4")))]
        let (scheme, level) = (CompressionScheme::None, None);

        let compressor =
            GeneralBufferCompressor::get_compressor(CompressionConfig { scheme, level }).unwrap();
        Self { compressor }
    }
}

impl CompressedBufferEncoder {
    pub fn try_new(compression_config: CompressionConfig) -> Result<Self> {
        let compressor = GeneralBufferCompressor::get_compressor(compression_config)?;
        Ok(Self { compressor })
    }

    pub fn from_scheme(scheme: pb21::CompressionScheme) -> Result<Self> {
        let scheme = CompressionScheme::try_from(scheme)?;
        Ok(Self {
            compressor: GeneralBufferCompressor::get_compressor(CompressionConfig {
                scheme,
                level: Some(0),
            })?,
        })
    }
}

impl CompressedBufferEncoder {
    pub fn per_value_compress<T: ArrowNativeType>(
        &self,
        data: &[u8],
        offsets: &[T],
        compressed: &mut Vec<u8>,
    ) -> Result<LanceBuffer> {
        let mut new_offsets: Vec<T> = Vec::with_capacity(offsets.len());
        new_offsets.push(T::from_usize(0).unwrap());

        for off in offsets.windows(2) {
            let start = off[0].as_usize();
            let end = off[1].as_usize();
            self.compressor.compress(&data[start..end], compressed)?;
            new_offsets.push(T::from_usize(compressed.len()).unwrap());
        }

        Ok(LanceBuffer::reinterpret_vec(new_offsets))
    }

    pub fn per_value_decompress<T: ArrowNativeType>(
        &self,
        data: &[u8],
        offsets: &[T],
        decompressed: &mut Vec<u8>,
    ) -> Result<LanceBuffer> {
        let mut new_offsets: Vec<T> = Vec::with_capacity(offsets.len());
        new_offsets.push(T::from_usize(0).unwrap());

        for off in offsets.windows(2) {
            let start = off[0].as_usize();
            let end = off[1].as_usize();
            self.compressor
                .decompress(&data[start..end], decompressed)?;
            new_offsets.push(T::from_usize(decompressed.len()).unwrap());
        }

        Ok(LanceBuffer::reinterpret_vec(new_offsets))
    }
}

impl PerValueCompressor for CompressedBufferEncoder {
    fn compress(&self, data: DataBlock) -> Result<(PerValueDataBlock, CompressiveEncoding)> {
        let data_type = data.name();
        let data = data.as_variable_width().ok_or(Error::Internal {
            message: format!(
                "Attempt to use CompressedBufferEncoder on data of type {}",
                data_type
            ),
            location: location!(),
        })?;

        let data_bytes = &data.data;
        let mut compressed = Vec::with_capacity(data_bytes.len());

        let new_offsets = match data.bits_per_offset {
            32 => self.per_value_compress::<u32>(
                data_bytes,
                &data.offsets.borrow_to_typed_slice::<u32>(),
                &mut compressed,
            )?,
            64 => self.per_value_compress::<u64>(
                data_bytes,
                &data.offsets.borrow_to_typed_slice::<u64>(),
                &mut compressed,
            )?,
            _ => unreachable!(),
        };

        let compressed = PerValueDataBlock::Variable(VariableWidthBlock {
            bits_per_offset: data.bits_per_offset,
            data: LanceBuffer::from(compressed),
            offsets: new_offsets,
            num_values: data.num_values,
            block_info: BlockInfo::new(),
        });

        // TODO: Support setting the level
        // TODO: Support underlying compression of data (e.g. defer to binary encoding for offset bitpacking)
        let encoding = ProtobufUtils21::wrapped(
            self.compressor.config(),
            ProtobufUtils21::variable(
                ProtobufUtils21::flat(data.bits_per_offset as u64, None),
                None,
            ),
        )?;

        Ok((compressed, encoding))
    }
}

impl VariablePerValueDecompressor for CompressedBufferEncoder {
    fn decompress(&self, data: VariableWidthBlock) -> Result<DataBlock> {
        let data_bytes = &data.data;
        let mut decompressed = Vec::with_capacity(data_bytes.len() * 2);

        let new_offsets = match data.bits_per_offset {
            32 => self.per_value_decompress(
                data_bytes,
                &data.offsets.borrow_to_typed_slice::<u32>(),
                &mut decompressed,
            )?,
            64 => self.per_value_decompress(
                data_bytes,
                &data.offsets.borrow_to_typed_slice::<u32>(),
                &mut decompressed,
            )?,
            _ => unreachable!(),
        };
        Ok(DataBlock::VariableWidth(VariableWidthBlock {
            bits_per_offset: data.bits_per_offset,
            data: LanceBuffer::from(decompressed),
            offsets: new_offsets,
            num_values: data.num_values,
            block_info: BlockInfo::new(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    use crate::encodings::physical::block::zstd::ZstdBufferCompressor;

    #[test]
    fn test_compression_scheme_from_str() {
        assert_eq!(
            CompressionScheme::from_str("none").unwrap(),
            CompressionScheme::None
        );
        assert_eq!(
            CompressionScheme::from_str("zstd").unwrap(),
            CompressionScheme::Zstd
        );
    }

    #[test]
    fn test_compression_scheme_from_str_invalid() {
        assert!(CompressionScheme::from_str("invalid").is_err());
    }

    #[cfg(feature = "zstd")]
    mod zstd {
        use std::io::Write;

        use super::*;

        #[test]
        fn test_compress_zstd_with_length_prefixed() {
            let compressor = ZstdBufferCompressor::new(0);
            let input_data = b"Hello, world!";
            let mut compressed_data = Vec::new();

            compressor
                .compress(input_data, &mut compressed_data)
                .unwrap();
            let mut decompressed_data = Vec::new();
            compressor
                .decompress(&compressed_data, &mut decompressed_data)
                .unwrap();
            assert_eq!(input_data, decompressed_data.as_slice());
        }

        #[test]
        fn test_zstd_compress_decompress_multiple_times() {
            let compressor = ZstdBufferCompressor::new(0);
            let (input_data_1, input_data_2) = (b"Hello ", b"World");
            let mut compressed_data = Vec::new();

            compressor
                .compress(input_data_1, &mut compressed_data)
                .unwrap();
            let compressed_length_1 = compressed_data.len();

            compressor
                .compress(input_data_2, &mut compressed_data)
                .unwrap();

            let mut decompressed_data = Vec::new();
            compressor
                .decompress(
                    &compressed_data[..compressed_length_1],
                    &mut decompressed_data,
                )
                .unwrap();

            compressor
                .decompress(
                    &compressed_data[compressed_length_1..],
                    &mut decompressed_data,
                )
                .unwrap();

            // the output should contain both input_data_1 and input_data_2
            assert_eq!(
                decompressed_data.len(),
                input_data_1.len() + input_data_2.len()
            );
            assert_eq!(
                &decompressed_data[..input_data_1.len()],
                input_data_1,
                "First part of decompressed data should match input_1"
            );
            assert_eq!(
                &decompressed_data[input_data_1.len()..],
                input_data_2,
                "Second part of decompressed data should match input_2"
            );
        }

        #[test]
        fn test_compress_zstd_raw_stream_format_and_decompress_with_length_prefixed() {
            let compressor = ZstdBufferCompressor::new(0);
            let input_data = b"Hello, world!";
            let mut compressed_data = Vec::new();

            // compress using raw stream format
            let mut encoder = ::zstd::Encoder::new(&mut compressed_data, 0).unwrap();
            encoder.write_all(input_data).unwrap();
            encoder.finish().expect("failed to encode data with zstd");

            // decompress using length prefixed format
            let mut decompressed_data = Vec::new();
            compressor
                .decompress(&compressed_data, &mut decompressed_data)
                .unwrap();
            assert_eq!(input_data, decompressed_data.as_slice());
        }
    }

    #[cfg(feature = "lz4")]
    mod lz4 {
        use super::*;

        use crate::encodings::physical::block::lz4::Lz4BufferCompressor;

        #[test]
        fn test_lz4_compress_decompress() {
            let compressor = Lz4BufferCompressor::default();
            let input_data = b"Hello, world!";
            let mut compressed_data = Vec::new();

            compressor
                .compress(input_data, &mut compressed_data)
                .unwrap();
            let mut decompressed_data = Vec::new();
            compressor
                .decompress(&compressed_data, &mut decompressed_data)
                .unwrap();
            assert_eq!(input_data, decompressed_data.as_slice());
        }
    }
}
