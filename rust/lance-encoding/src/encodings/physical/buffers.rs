// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::io::{Cursor, Write};
use std::sync::Arc;

use arrow_array::{cast::AsArray, Array, ArrayRef};
use arrow_buffer::{BooleanBufferBuilder, Buffer};
use arrow_schema::DataType;

use lance_arrow::DataTypeExt;
use lance_core::Result;

use crate::{
    data::{BlockEncodedDataBlock, EncodedDataBlock, FixedWidthEncodedDataBlock},
    encoder::{BufferEncoder, EncodedBuffer, EncodedBufferMeta}, format::pb,
};

use super::value::CompressionScheme;

#[derive(Debug, Default)]
pub struct FlatBufferEncoder {}

impl BufferEncoder for FlatBufferEncoder {
    fn encode(&self, arrays: &[ArrayRef], buffer_index: u32) -> Result<Box<dyn EncodedDataBlock>> {
        let parts = arrays
            .iter()
            .map(|arr| arr.to_data().buffers()[0].clone())
            .collect::<Vec<_>>();
        let data_type = arrays[0].data_type();
        let bits_per_value = (data_type.byte_width() * 8) as u64; 
        Ok(Box::new(FixedWidthEncodedDataBlock {
            encoding: Arc::new(flat_buffer_encoding(bits_per_value, None, buffer_index)),
            data: parts,
            bits_per_value,
        }))
        /*
        Ok((
            EncodedBuffer { parts },
            EncodedBufferMeta {
                bits_per_value: (data_type.byte_width() * 8) as u64,
                bitpacked_bits_per_value: None,
                compression_scheme: None,
            },
        ))
        */
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

impl BufferEncoder for CompressedBufferEncoder {
    fn encode(&self, arrays: &[ArrayRef], buffer_index: u32) -> Result<Box<dyn EncodedDataBlock>> {
        let mut parts = Vec::with_capacity(arrays.len());
        for arr in arrays {
            let buffer = arr.to_data().buffers()[0].clone();
            let buffer_len = buffer.len();
            let buffer_data = buffer.as_slice();
            let mut compressed = Vec::with_capacity(buffer_len);
            self.compressor.compress(buffer_data, &mut compressed)?;
            parts.push(Buffer::from(compressed));
        }


        let data_type = arrays[0].data_type();
        let bits_per_value = (data_type.byte_width() * 8) as u64;
        Ok(Box::new(BlockEncodedDataBlock {
            encoding: Arc::new(flat_buffer_encoding(bits_per_value, Some(CompressionScheme::Zstd), buffer_index)),
            data: parts,
        }))

        /*
        Ok((
            EncodedBuffer { parts },
            EncodedBufferMeta {
                bits_per_value: (data_type.byte_width() * 8) as u64,
                bitpacked_bits_per_value: None,
                compression_scheme: Some(CompressionScheme::Zstd),
            },
        ))
        */
    }
}

// Encoder for writing boolean arrays as dense bitmaps
#[derive(Debug, Default)]
pub struct BitmapBufferEncoder {}

impl BufferEncoder for BitmapBufferEncoder {
    fn encode(&self, arrays: &[ArrayRef], buffer_index: u32) -> Result<Box<dyn EncodedDataBlock>> {
        debug_assert!(arrays
            .iter()
            .all(|arr| *arr.data_type() == DataType::Boolean));
        let num_rows: u32 = arrays.iter().map(|arr| arr.len() as u32).sum();
        // Empty pages don't make sense, this should be prevented before we
        // get here
        debug_assert_ne!(num_rows, 0);
        // We can't just write the inner value buffers one after the other because
        // bitmaps can have junk padding at the end (e.g. a boolean array with 12
        // values will be 2 bytes but the last four bits of the second byte are
        // garbage).  So we go ahead and pay the cost of a copy (we could avoid this
        // if we really needed to, at the expense of more complicated code and a slightly
        // larger encoded size but writer cost generally doesn't matter as much as reader cost)
        let mut builder = BooleanBufferBuilder::new(num_rows as usize);
        for arr in arrays {
            let bool_arr = arr.as_boolean();
            builder.append_buffer(bool_arr.values());
        }
        let buffer = builder.finish().into_inner();
        let parts = vec![buffer];
        
        Ok(Box::new(FixedWidthEncodedDataBlock {
            encoding: Arc::new(flat_buffer_encoding(1, None, buffer_index)),
            bits_per_value: 1,
            data: parts,
        }))

        /*
        let buffer = EncodedBuffer { parts };
        Ok((
            buffer,
            EncodedBufferMeta {
                bits_per_value: 1,
                bitpacked_bits_per_value: None,
                compression_scheme: None,
            },
        ))
        */
    }
}

// helper function for creating flat buffer encoder
fn flat_buffer_encoding(
    bits_per_value: u64,
    compression: Option<CompressionScheme>,
    buffer_index: u32,
) -> pb::ArrayEncoding {
    pb::ArrayEncoding{
        array_encoding: Some(pb::array_encoding::ArrayEncoding::Flat(pb::Flat {
            bits_per_value,
            buffer: Some(pb::Buffer {
                buffer_index,
                buffer_type: pb::buffer::BufferType::Page as i32,
            }),
            compression: compression.map(|c| pb::Compression {
                scheme: c.to_string()
            })
        }))
    }
}
