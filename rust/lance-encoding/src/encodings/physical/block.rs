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
use snafu::location;
use std::{
    io::{Cursor, Write},
    str::FromStr,
};

use lance_core::{Error, Result};

use crate::{
    buffer::LanceBuffer,
    compression::VariablePerValueDecompressor,
    data::{BlockInfo, DataBlock, VariableWidthBlock},
    encodings::logical::primitive::fullzip::{PerValueCompressor, PerValueDataBlock},
    format::{pb, ProtobufUtils},
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
            "zstd" => Ok(Self::Zstd),
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
    fn name(&self) -> &str;
}

#[derive(Debug, Default)]
pub struct ZstdBufferCompressor {
    compression_level: i32,
}

impl ZstdBufferCompressor {
    pub fn new(compression_level: i32) -> Self {
        Self { compression_level }
    }
}

impl BufferCompressor for ZstdBufferCompressor {
    fn compress(&self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()> {
        let mut encoder = zstd::Encoder::new(output_buf, self.compression_level)?;
        encoder.write_all(input_buf)?;
        match encoder.finish() {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    fn decompress(&self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()> {
        let source = Cursor::new(input_buf);
        zstd::stream::copy_decode(source, output_buf)?;
        Ok(())
    }

    fn name(&self) -> &str {
        "zstd"
    }
}

#[derive(Debug, Default)]
pub struct Lz4BufferCompressor {}

impl BufferCompressor for Lz4BufferCompressor {
    fn compress(&self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()> {
        lz4::block::compress_to_buffer(input_buf, None, true, output_buf)
            .map_err(|err| Error::Internal {
                message: format!("LZ4 compression error: {}", err),
                location: location!(),
            })
            .map(|_| ())
    }

    fn decompress(&self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()> {
        lz4::block::decompress_to_buffer(input_buf, None, output_buf)
            .map_err(|err| Error::Internal {
                message: format!("LZ4 decompression error: {}", err),
                location: location!(),
            })
            .map(|_| ())
    }

    fn name(&self) -> &str {
        "zstd"
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

    fn name(&self) -> &str {
        "none"
    }
}

pub struct GeneralBufferCompressor {}

impl GeneralBufferCompressor {
    pub fn get_compressor(compression_config: CompressionConfig) -> Box<dyn BufferCompressor> {
        match compression_config.scheme {
            // FSST has its own compression path and isn't implemented as a generic buffer compressor
            CompressionScheme::Fsst => unimplemented!(),
            CompressionScheme::Zstd => Box::new(ZstdBufferCompressor::new(
                compression_config.level.unwrap_or(0),
            )),
            CompressionScheme::Lz4 => Box::new(Lz4BufferCompressor::default()),
            CompressionScheme::None => Box::new(NoopBufferCompressor {}),
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
        Self {
            compressor: GeneralBufferCompressor::get_compressor(CompressionConfig {
                scheme: CompressionScheme::Zstd,
                level: Some(0),
            }),
        }
    }
}

impl CompressedBufferEncoder {
    pub fn new(compression_config: CompressionConfig) -> Self {
        let compressor = GeneralBufferCompressor::get_compressor(compression_config);
        Self { compressor }
    }

    pub fn from_scheme(scheme: &str) -> Result<Self> {
        let scheme = CompressionScheme::from_str(scheme)?;
        Ok(Self {
            compressor: GeneralBufferCompressor::get_compressor(CompressionConfig {
                scheme,
                level: Some(0),
            }),
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
    fn compress(&self, data: DataBlock) -> Result<(PerValueDataBlock, pb::ArrayEncoding)> {
        let data_type = data.name();
        let mut data = data.as_variable_width().ok_or(Error::Internal {
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

        let encoding = ProtobufUtils::block(self.compressor.name());

        Ok((compressed, encoding))
    }
}

impl VariablePerValueDecompressor for CompressedBufferEncoder {
    fn decompress(&self, mut data: VariableWidthBlock) -> Result<DataBlock> {
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
}
