// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use bytes::Bytes;
use futures::{future::BoxFuture, FutureExt};
use lance_arrow::DataTypeExt;
use log::trace;
use snafu::{location, Location};

use crate::{
    decoder::{PhysicalPageDecoder, PhysicalPageScheduler},
    encoder::{ArrayEncoder, BufferEncoder, EncodedArray, EncodedArrayBuffer},
    format::pb,
    EncodingsIo,
};

use lance_core::{Error, Result};

use super::buffers::{BitmapBufferEncoder, FlatBufferEncoder};

/// Scheduler for a simple encoding where buffers of fixed-size items are stored as-is on disk
#[derive(Debug, Clone, Copy)]
pub struct ValuePageScheduler {
    // TODO: do we really support values greater than 2^32 bytes per value?
    // I think we want to, in theory, but will need to test this case.
    bytes_per_value: u64,
    buffer_offset: u64,
}

impl ValuePageScheduler {
    pub fn new(bytes_per_value: u64, buffer_offset: u64) -> Self {
        Self {
            bytes_per_value,
            buffer_offset,
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
        let mut min = u64::MAX;
        let mut max = 0;
        let byte_ranges = ranges
            .iter()
            .map(|range| {
                let start = self.buffer_offset + (range.start as u64 * self.bytes_per_value);
                let end = self.buffer_offset + (range.end as u64 * self.bytes_per_value);
                min = min.min(start);
                max = max.max(end);
                start..end
            })
            .collect::<Vec<_>>();

        trace!(
            "Scheduling I/O for {} ranges spread across byte range {}..{}",
            byte_ranges.len(),
            min,
            max
        );
        let bytes = scheduler.submit_request(byte_ranges, top_level_row);
        let bytes_per_value = self.bytes_per_value;

        async move {
            let bytes = bytes.await?;
            Ok(Box::new(ValuePageDecoder {
                bytes_per_value,
                data: bytes,
            }) as Box<dyn PhysicalPageDecoder>)
        }
        .boxed()
    }
}

struct ValuePageDecoder {
    bytes_per_value: u64,
    data: Vec<Bytes>,
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

    fn decode_into(&self, rows_to_skip: u32, num_rows: u32, dest_buffers: &mut [bytes::BytesMut]) {
        let mut bytes_to_skip = rows_to_skip as u64 * self.bytes_per_value;
        let mut bytes_to_take = num_rows as u64 * self.bytes_per_value;

        let dest = &mut dest_buffers[0];

        debug_assert!(dest.capacity() as u64 >= bytes_to_take);

        for buf in &self.data {
            let buf_len = buf.len() as u64;
            if bytes_to_skip > buf_len {
                bytes_to_skip -= buf_len;
            } else {
                let bytes_to_take_here = (buf_len - bytes_to_skip).min(bytes_to_take);
                bytes_to_take -= bytes_to_take_here;
                let start = bytes_to_skip as usize;
                let end = start + bytes_to_take_here as usize;
                dest.extend_from_slice(&buf.slice(start..end));
                bytes_to_skip = 0;
            }
        }
    }

    fn num_buffers(&self) -> u32 {
        1
    }
}

#[derive(Debug)]
pub struct ValueEncoder {
    buffer_encoder: Box<dyn BufferEncoder>,
}

impl ValueEncoder {
    pub fn try_new(data_type: &DataType) -> Result<Self> {
        if *data_type == DataType::Boolean {
            Ok(Self {
                buffer_encoder: Box::<BitmapBufferEncoder>::default(),
            })
        } else if data_type.is_fixed_stride() {
            Ok(Self {
                buffer_encoder: Box::<FlatBufferEncoder>::default(),
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
