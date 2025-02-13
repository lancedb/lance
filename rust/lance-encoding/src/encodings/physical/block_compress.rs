// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_buffer::ArrowNativeType;
use arrow_schema::DataType;
use snafu::{location, Location};
use std::{
    io::{Cursor, Write},
    str::FromStr,
};

use lance_core::{Error, Result};

use crate::{
    buffer::LanceBuffer,
    data::{BlockInfo, DataBlock, OpaqueBlock, VariableWidthBlock},
    decoder::VariablePerValueDecompressor,
    encoder::{ArrayEncoder, EncodedArray, PerValueCompressor, PerValueDataBlock},
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
            scheme: CompressionScheme::Zstd,
            level: Some(0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionScheme {
    None,
    Fsst,
    Zstd,
}

impl std::fmt::Display for CompressionScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let scheme_str = match self {
            Self::Fsst => "fsst",
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
