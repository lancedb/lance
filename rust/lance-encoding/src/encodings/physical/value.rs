// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_schema::DataType;
use bytes::Bytes;
use futures::{future::BoxFuture, FutureExt};
use log::trace;
use snafu::{location, Location};
use std::ops::Range;
use std::sync::{Arc, Mutex};

use crate::buffer::LanceBuffer;
use crate::data::{DataBlock, FixedWidthDataBlock};
use crate::format::ProtobufUtils;
use crate::{
    decoder::{PageScheduler, PrimitivePageDecoder},
    encoder::{ArrayEncoder, EncodedArray},
    EncodingsIo,
};

use lance_core::{Error, Result};

use super::block_compress::{CompressionScheme, GeneralBufferCompressor};

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

impl PageScheduler for ValuePageScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u64>],
        scheduler: &Arc<dyn EncodingsIo>,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PrimitivePageDecoder>>> {
        let (mut min, mut max) = (u64::MAX, 0);
        let byte_ranges = if self.compression_scheme == CompressionScheme::None {
            ranges
                .iter()
                .map(|range| {
                    let start = self.buffer_offset + (range.start * self.bytes_per_value);
                    let end = self.buffer_offset + (range.end * self.bytes_per_value);
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
                    let start = (range.start * bytes_per_value) as usize;
                    let end = (range.end * bytes_per_value) as usize;
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
            }) as Box<dyn PrimitivePageDecoder>)
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

        let mut bytes_in_ranges: Vec<Bytes> =
            Vec::with_capacity(self.uncompressed_range_offsets.len());
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

    fn decode_buffers<'a>(
        &'a self,
        buffers: impl IntoIterator<Item = &'a Bytes>,
        mut bytes_to_skip: u64,
        mut bytes_to_take: u64,
    ) -> LanceBuffer {
        let mut dest: Option<Vec<u8>> = None;

        for buf in buffers.into_iter() {
            let buf_len = buf.len() as u64;
            if bytes_to_skip > buf_len {
                bytes_to_skip -= buf_len;
            } else {
                let bytes_to_take_here = (buf_len - bytes_to_skip).min(bytes_to_take);
                bytes_to_take -= bytes_to_take_here;
                let start = bytes_to_skip as usize;
                let end = start + bytes_to_take_here as usize;
                let slice = buf.slice(start..end);
                match (&mut dest, bytes_to_take) {
                    (None, 0) => {
                        // The entire request is contained in one buffer so we can maybe zero-copy
                        // if the slice is aligned properly
                        return LanceBuffer::from_bytes(slice, self.bytes_per_value);
                    }
                    (None, _) => {
                        dest.replace(Vec::with_capacity(bytes_to_take as usize));
                    }
                    _ => {}
                }
                dest.as_mut().unwrap().extend_from_slice(&slice);
                bytes_to_skip = 0;
            }
        }
        LanceBuffer::from(dest.unwrap_or_default())
    }
}

impl PrimitivePageDecoder for ValuePageDecoder {
    fn decode(&self, rows_to_skip: u64, num_rows: u64) -> Result<DataBlock> {
        let bytes_to_skip = rows_to_skip * self.bytes_per_value;
        let bytes_to_take = num_rows * self.bytes_per_value;

        let data_buffer = if self.is_compressed() {
            let decoding_data = self.get_uncompressed_bytes()?;
            let buffers = decoding_data.lock().unwrap();
            self.decode_buffers(buffers.as_ref().unwrap(), bytes_to_skip, bytes_to_take)
        } else {
            self.decode_buffers(&self.data, bytes_to_skip, bytes_to_take)
        };
        Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: self.bytes_per_value * 8,
            data: data_buffer,
            num_values: num_rows,
        }))
    }
}

#[derive(Debug, Default)]
pub struct ValueEncoder {}

impl ArrayEncoder for ValueEncoder {
    fn encode(
        &self,
        data: DataBlock,
        _data_type: &DataType,
        buffer_index: &mut u32,
    ) -> Result<EncodedArray> {
        let index = *buffer_index;
        *buffer_index += 1;

        let encoding = match &data {
            DataBlock::FixedWidth(fixed_width) => Ok(ProtobufUtils::flat_encoding(
                fixed_width.bits_per_value,
                index,
                None,
            )),
            _ => Err(Error::InvalidInput {
                source: format!(
                    "Cannot encode a data block of type {} with ValueEncoder",
                    data.name()
                )
                .into(),
                location: location!(),
            }),
        }?;
        Ok(EncodedArray { data, encoding })
    }
}

// public tests module because we share the PRIMITIVE_TYPES constant with fixed_size_list
#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;

    use arrow_schema::{DataType, Field, TimeUnit};

    use crate::testing::check_round_trip_encoding_random;

    const PRIMITIVE_TYPES: &[DataType] = &[
        DataType::Null,
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
            log::info!("Testing encoding for {:?}", data_type);
            let field = Field::new("", data_type.clone(), false);
            check_round_trip_encoding_random(field, HashMap::new()).await;
        }
    }
}
