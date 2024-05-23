// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt;
use arrow_array::ArrayRef;
use arrow_schema::DataType;
use bytes::Bytes;
use futures::{future::BoxFuture, FutureExt};
use lance_arrow::DataTypeExt;
use log::trace;
use snafu::{location, Location};
use std::ops::Range;
use std::sync::{Arc, Mutex};

use crate::{
    decoder::{PhysicalPageDecoder, PhysicalPageScheduler},
    encoder::{ArrayEncoder, BufferEncoder, EncodedArray, EncodedArrayBuffer},
    format::pb,
    EncodingsIo,
};

use lance_core::{Error, Result};

use super::buffers::{
    BitmapBufferEncoder, CompressedBufferEncoder, FlatBufferEncoder, GeneralBufferCompressor,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionScheme {
    None,
    Zstd,
}

impl fmt::Display for CompressionScheme {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let scheme_str = match self {
            CompressionScheme::Zstd => "zstd",
            CompressionScheme::None => "none",
        };
        write!(f, "{}", scheme_str)
    }
}

/// Scheduler for a simple encoding where buffers of fixed-size items are stored as-is on disk
#[derive(Debug, Clone, Copy)]
pub struct ValuePageScheduler {
    // TODO: do we really support values greater than 2^32 bytes per value?
    // I think we want to, in theory, but will need to test this case.
    bytes_per_value: u64,
    buffer_offset: u64,
    buffer_size: u64,
    compression_scheme: CompressionScheme,
}

impl ValuePageScheduler {
    pub fn new(
        bytes_per_value: u64,
        buffer_offset: u64,
        buffer_size: u64,
        compression_scheme: CompressionScheme,
    ) -> Self {
        Self {
            bytes_per_value,
            buffer_offset,
            buffer_size,
            compression_scheme,
        }
    }
}

impl PhysicalPageScheduler for ValuePageScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u32>],
        scheduler: &dyn EncodingsIo,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>> {
        let (mut min, mut max) = (u64::MAX, 0);
        let byte_ranges = if self.compression_scheme == CompressionScheme::None {
            ranges
                .iter()
                .map(|range| {
                    let start = self.buffer_offset + (range.start as u64 * self.bytes_per_value);
                    let end = self.buffer_offset + (range.end as u64 * self.bytes_per_value);
                    min = min.min(start);
                    max = max.max(end);
                    start..end
                })
                .collect::<Vec<_>>()
        } else {
            min = self.buffer_offset;
            max = self.buffer_offset + self.buffer_size;
            // for compressed page, the ranges are always the entire page,
            // and it is guaranteed that only one range is passed
            vec![Range {
                start: min,
                end: max,
            }]
        };

        trace!(
            "Scheduling I/O for {} ranges spread across byte range {}..{}",
            byte_ranges.len(),
            min,
            max
        );
        let bytes = scheduler.submit_request(byte_ranges, top_level_row);
        let bytes_per_value = self.bytes_per_value;

        let range_offsets = if self.compression_scheme != CompressionScheme::None {
            ranges
                .iter()
                .map(|range| {
                    let start = (range.start as u64 * bytes_per_value) as usize;
                    let end = (range.end as u64 * bytes_per_value) as usize;
                    start..end
                })
                .collect::<Vec<_>>()
        } else {
            vec![]
        };

        async move {
            let bytes = bytes.await?;

            Ok(Box::new(ValuePageDecoder {
                bytes_per_value,
                data: bytes,
                uncompressed_data: Arc::new(Mutex::new(None)),
                uncompressed_range_offsets: range_offsets,
            }) as Box<dyn PhysicalPageDecoder>)
        }
        .boxed()
    }
}

struct ValuePageDecoder {
    bytes_per_value: u64,
    data: Vec<Bytes>,
    uncompressed_data: Arc<Mutex<Option<Vec<Bytes>>>>,
    uncompressed_range_offsets: Vec<std::ops::Range<usize>>,
}

impl ValuePageDecoder {
    fn decompress(&self) -> Result<Vec<Bytes>> {
        // for compressed page, it is guaranteed that only one range is passed
        let bytes_u8: Vec<u8> = self.data[0].to_vec();
        let buffer_compressor = GeneralBufferCompressor::get_compressor("");
        let mut uncompressed_bytes: Vec<u8> = Vec::new();
        buffer_compressor.decompress(&bytes_u8, &mut uncompressed_bytes)?;

        let mut bytes_in_ranges: Vec<Bytes> = Vec::new();
        bytes_in_ranges.reserve(self.uncompressed_range_offsets.len());
        for range in &self.uncompressed_range_offsets {
            let start = range.start;
            let end = range.end;
            bytes_in_ranges.push(Bytes::from(uncompressed_bytes[start..end].to_vec()));
        }
        Ok(bytes_in_ranges)
    }

    fn get_uncompressed_bytes(&self) -> Result<Arc<Mutex<Option<Vec<Bytes>>>>> {
        let mut uncompressed_bytes = self.uncompressed_data.lock().unwrap();
        if uncompressed_bytes.is_none() {
            *uncompressed_bytes = Some(self.decompress()?);
        }
        Ok(Arc::clone(&self.uncompressed_data))
    }

    fn is_compressed(&self) -> bool {
        !self.uncompressed_range_offsets.is_empty()
    }

    fn decode_buffer(
        &self,
        buf: &Bytes,
        bytes_to_skip: &mut u64,
        bytes_to_take: &mut u64,
        dest: &mut bytes::BytesMut,
    ) {
        let buf_len = buf.len() as u64;
        if *bytes_to_skip > buf_len {
            *bytes_to_skip -= buf_len;
        } else {
            let bytes_to_take_here = (buf_len - *bytes_to_skip).min(*bytes_to_take);
            *bytes_to_take -= bytes_to_take_here;
            let start = *bytes_to_skip as usize;
            let end = start + bytes_to_take_here as usize;
            dest.extend_from_slice(&buf.slice(start..end));
            *bytes_to_skip = 0;
        }
    }
}

impl PhysicalPageDecoder for ValuePageDecoder {
    fn update_capacity(
        &self,
        _rows_to_skip: u32,
        num_rows: u32,
        buffers: &mut [(u64, bool)],
        _all_null: &mut bool,
    ) {
        buffers[0].0 = self.bytes_per_value * num_rows as u64;
        buffers[0].1 = true;
    }

    fn decode_into(
        &self,
        rows_to_skip: u32,
        num_rows: u32,
        dest_buffers: &mut [bytes::BytesMut],
    ) -> Result<()> {
        let mut bytes_to_skip = rows_to_skip as u64 * self.bytes_per_value;
        let mut bytes_to_take = num_rows as u64 * self.bytes_per_value;

        let dest = &mut dest_buffers[0];

        debug_assert!(dest.capacity() as u64 >= bytes_to_take);

        if self.is_compressed() {
            let decoding_data = self.get_uncompressed_bytes()?;
            for buf in decoding_data.lock().unwrap().as_ref().unwrap() {
                self.decode_buffer(buf, &mut bytes_to_skip, &mut bytes_to_take, dest);
            }
        } else {
            for buf in &self.data {
                self.decode_buffer(buf, &mut bytes_to_skip, &mut bytes_to_take, dest);
            }
        }
        Ok(())
    }

    fn num_buffers(&self) -> u32 {
        1
    }
}

#[derive(Debug)]
pub struct ValueEncoder {
    buffer_encoder: Box<dyn BufferEncoder>,
    compression_scheme: CompressionScheme,
}

impl ValueEncoder {
    pub fn try_new(data_type: &DataType, compression_scheme: CompressionScheme) -> Result<Self> {
        if *data_type == DataType::Boolean {
            Ok(Self {
                buffer_encoder: Box::<BitmapBufferEncoder>::default(),
                compression_scheme,
            })
        } else if data_type.is_fixed_stride() {
            Ok(Self {
                buffer_encoder: if compression_scheme != CompressionScheme::None {
                    Box::<CompressedBufferEncoder>::default()
                } else {
                    Box::<FlatBufferEncoder>::default()
                },
                compression_scheme,
            })
        } else {
            Err(Error::invalid_input(
                format!("Cannot use ValueEncoder to encode {}", data_type),
                location!(),
            ))
        }
    }
}

impl ArrayEncoder for ValueEncoder {
    fn encode(&self, arrays: &[ArrayRef], buffer_index: &mut u32) -> Result<EncodedArray> {
        let index = *buffer_index;
        *buffer_index += 1;

        let encoded_buffer = self.buffer_encoder.encode(arrays)?;
        let array_bufs = vec![EncodedArrayBuffer {
            parts: encoded_buffer.parts,
            index,
        }];

        let data_type = arrays[0].data_type();
        let bits_per_value = match data_type {
            DataType::Boolean => 1,
            _ => 8 * data_type.byte_width() as u64,
        };
        let flat_encoding = pb::ArrayEncoding {
            array_encoding: Some(pb::array_encoding::ArrayEncoding::Flat(pb::Flat {
                bits_per_value,
                buffer: Some(pb::Buffer {
                    buffer_index: index,
                    buffer_type: pb::buffer::BufferType::Page as i32,
                }),
                compression: if self.compression_scheme != CompressionScheme::None {
                    Some(pb::Compression {
                        scheme: self.compression_scheme.to_string(),
                    })
                } else {
                    None
                },
            })),
        };

        Ok(EncodedArray {
            buffers: array_bufs,
            encoding: flat_encoding,
        })
    }
}

// public tests module because we share the PRIMITIVE_TYPES constant with fixed_size_list
#[cfg(test)]
pub(crate) mod tests {
    use arrow_schema::{DataType, Field, TimeUnit};

    use crate::testing::check_round_trip_encoding_random;

    const PRIMITIVE_TYPES: &[DataType] = &[
        DataType::FixedSizeBinary(2),
        DataType::Date32,
        DataType::Date64,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
        DataType::Float16,
        DataType::Float32,
        DataType::Float64,
        DataType::Decimal128(10, 10),
        DataType::Decimal256(10, 10),
        DataType::Timestamp(TimeUnit::Nanosecond, None),
        DataType::Time32(TimeUnit::Second),
        DataType::Time64(TimeUnit::Nanosecond),
        DataType::Duration(TimeUnit::Second),
        // The Interval type is supported by the reader but the writer works with Lance schema
        // at the moment and Lance schema can't parse interval
        // DataType::Interval(IntervalUnit::DayTime),
    ];

    #[test_log::test(tokio::test)]
    async fn test_value_primitive() {
        for data_type in PRIMITIVE_TYPES {
            let field = Field::new("", data_type.clone(), false);
            check_round_trip_encoding_random(field).await;
        }
    }
}
