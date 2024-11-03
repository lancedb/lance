// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_schema::DataType;
use snafu::{location, Location};
use std::{
    io::{Cursor, Write},
    str::FromStr,
};

use lance_core::{Error, Result};

use crate::{
    data::{BlockInfo, DataBlock, OpaqueBlock},
    encoder::{ArrayEncoder, EncodedArray},
    format::ProtobufUtils,
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
            scheme: CompressionScheme::Zstd,
            level: Some(0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionScheme {
    None,
    Zstd,
}

impl std::fmt::Display for CompressionScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let scheme_str = match self {
            Self::Zstd => "zstd",
            Self::None => "none",
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
}

pub struct GeneralBufferCompressor {}

impl GeneralBufferCompressor {
    pub fn get_compressor(compression_config: CompressionConfig) -> Box<dyn BufferCompressor> {
        match compression_config.scheme {
            CompressionScheme::Zstd => Box::new(ZstdBufferCompressor::new(
                compression_config.level.unwrap_or(0),
            )),
            CompressionScheme::None => Box::new(NoopBufferCompressor {}),
        }
    }
}

// An encoder which uses generic compression, such as zstd/lz4 to encode buffers
#[derive(Debug)]
pub struct CompressedBufferEncoder {
    compressor: Box<dyn BufferCompressor>,
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
}

impl ArrayEncoder for CompressedBufferEncoder {
    fn encode(
        &self,
        data: DataBlock,
        _data_type: &DataType,
        buffer_index: &mut u32,
    ) -> Result<EncodedArray> {
        let uncompressed_data = data.as_fixed_width().unwrap();

        let mut compressed_buf = Vec::with_capacity(uncompressed_data.data.len());
        self.compressor
            .compress(&uncompressed_data.data, &mut compressed_buf)?;

        let compressed_data = DataBlock::Opaque(OpaqueBlock {
            buffers: vec![compressed_buf.into()],
            num_values: uncompressed_data.num_values,
            block_info: BlockInfo::new(),
        });

        let comp_buf_index = *buffer_index;
        *buffer_index += 1;

        let encoding = ProtobufUtils::flat_encoding(
            uncompressed_data.bits_per_value,
            comp_buf_index,
            Some(CompressionConfig::new(CompressionScheme::Zstd, None)),
        );

        Ok(EncodedArray {
            data: compressed_data,
            encoding,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::LanceBuffer;
    use crate::data::FixedWidthDataBlock;
    use arrow_schema::DataType;
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

    #[test]
    fn test_compressed_buffer_encoder() {
        let encoder = CompressedBufferEncoder::default();
        let data = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 64,
            data: LanceBuffer::reinterpret_vec(vec![0, 1, 2, 3, 4, 5, 6, 7]),
            num_values: 8,
            block_info: BlockInfo::new(),
        });

        let mut buffer_index = 0;
        let encoded_array_result = encoder.encode(data, &DataType::Int64, &mut buffer_index);
        assert!(encoded_array_result.is_ok(), "{:?}", encoded_array_result);
        let encoded_array = encoded_array_result.unwrap();
        assert_eq!(encoded_array.data.num_values(), 8);
        let buffers = encoded_array.data.into_buffers();
        assert_eq!(buffers.len(), 1);
        assert!(buffers[0].len() < 64 * 8);
    }
}
