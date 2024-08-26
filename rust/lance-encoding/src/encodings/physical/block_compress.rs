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
    data::{DataBlock, OpaqueBlock},
    encoder::{ArrayEncoder, EncodedArray},
    format::ProtobufUtils,
};

pub const COMPRESSION_META_KEY: &str = "lance:compression";

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
pub struct ZstdBufferCompressor {}

impl BufferCompressor for ZstdBufferCompressor {
    fn compress(&self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()> {
        let mut encoder = zstd::Encoder::new(output_buf, 0)?;
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

pub struct GeneralBufferCompressor {}

impl GeneralBufferCompressor {
    pub fn get_compressor(compression_type: &str) -> Box<dyn BufferCompressor> {
        match compression_type {
            "" => Box::<ZstdBufferCompressor>::default(),
            "zstd" => Box::<ZstdBufferCompressor>::default(),
            _ => panic!("Unsupported compression type: {}", compression_type),
        }
    }
}

// An encoder which uses lightweight compression, such as zstd/lz4 to encode buffers
#[derive(Debug)]
pub struct CompressedBufferEncoder {
    compressor: Box<dyn BufferCompressor>,
}

impl Default for CompressedBufferEncoder {
    fn default() -> Self {
        Self {
            compressor: GeneralBufferCompressor::get_compressor("zstd"),
        }
    }
}

impl CompressedBufferEncoder {
    pub fn new(compression_type: &str) -> Self {
        let compressor = GeneralBufferCompressor::get_compressor(compression_type);
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
        let uncompressed_data = data.as_fixed_width()?;

        let mut compressed_buf = Vec::with_capacity(uncompressed_data.data.len());
        self.compressor
            .compress(&uncompressed_data.data, &mut compressed_buf)?;

        let compressed_data = DataBlock::Opaque(OpaqueBlock {
            buffers: vec![compressed_buf.into()],
            num_values: uncompressed_data.num_values,
        });

        let comp_buf_index = *buffer_index;
        *buffer_index += 1;

        let encoding = ProtobufUtils::flat_encoding(
            uncompressed_data.bits_per_value,
            comp_buf_index,
            Some(CompressionScheme::Zstd),
        );

        Ok(EncodedArray {
            data: compressed_data,
            encoding,
        })
    }
}
