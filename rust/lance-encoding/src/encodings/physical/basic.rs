use arrow_array::ArrayRef;
use arrow_schema::DataType;
use futures::{future::BoxFuture, FutureExt};
use lance_arrow::DataTypeExt;

use crate::{
    decoder::{PhysicalPageDecoder, PhysicalPageScheduler},
    encoder::{ArrayEncoder, BufferEncoder, EncodedBuffer, EncodedPage},
    format::pb,
    EncodingsIo,
};

use lance_core::Result;

use super::{bitmap::BitmapEncoder, value::ValueEncoder};

enum DataValidity {
    NoNull,
    SomeNull(Box<dyn PhysicalPageDecoder>),
}

/// A physical scheduler for "basic" fields.  These are fields that have an optional
/// validity bitmap and some kind of values buffer.
///
/// No actual decoding happens here, we are simply aggregating the two buffers.
///
// TODO: A validity bitmap is also not needed if everything is null.  Refactor
// DataValidity to be
//
// NoNull(values decoder)
// SomeNull(validity decoder, values decoder)
// AllNull
#[derive(Debug)]
pub struct BasicPageScheduler {
    validity_decoder: PageValidity,
    values_decoder: Box<dyn PhysicalPageScheduler>,
}

impl BasicPageScheduler {
    /// Creates a new instance that expect a validity bitmap
    pub fn new_nullable(
        validity_decoder: Box<dyn PhysicalPageScheduler>,
        values_decoder: Box<dyn PhysicalPageScheduler>,
    ) -> Self {
        Self {
            validity_decoder: PageValidity::SomeNull(validity_decoder),
            values_decoder,
        }
    }

    /// Create a new instance that does not need a validity bitmap because no item is null
    pub fn new_non_nullable(values_decoder: Box<dyn PhysicalPageScheduler>) -> Self {
        Self {
            validity_decoder: PageValidity::NoNull,
            values_decoder,
        }
    }
}

impl PhysicalPageScheduler for BasicPageScheduler {
    fn schedule_range(
        &self,
        range: std::ops::Range<u32>,
        scheduler: &dyn EncodingsIo,
    ) -> BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>> {
        let validity_future = match &self.validity_decoder {
            PageValidity::NoNull => None,
            PageValidity::SomeNull(validity_decoder) => {
                Some(validity_decoder.schedule_range(range.clone(), scheduler))
            }
        };

        let values_future = self.values_decoder.schedule_range(range, scheduler);

        async move {
            let validity = match validity_future {
                None => DataValidity::NoNull,
                Some(fut) => DataValidity::SomeNull(fut.await?),
            };
            let values = values_future.await?;
            Ok(Box::new(BasicPageDecoder { validity, values }) as Box<dyn PhysicalPageDecoder>)
        }
        .boxed()
    }
}

struct BasicPageDecoder {
    validity: DataValidity,
    values: Box<dyn PhysicalPageDecoder>,
}

impl PhysicalPageDecoder for BasicPageDecoder {
    fn update_capacity(&self, rows_to_skip: u32, num_rows: u32, buffers: &mut [(u64, bool)]) {
        // No need to look at the validity decoder to know the dest buffer size since it is boolean
        buffers[0].0 = arrow_buffer::bit_util::ceil(num_rows as usize, 8) as u64;
        // The validity buffer is only required if we have some nulls
        buffers[0].1 = match self.validity {
            DataValidity::NoNull => false,
            DataValidity::SomeNull(_) => true,
        };
        self.values
            .update_capacity(rows_to_skip, num_rows, &mut buffers[1..]);
    }

    fn decode_into(&self, rows_to_skip: u32, num_rows: u32, dest_buffers: &mut [bytes::BytesMut]) {
        if let DataValidity::SomeNull(validity_decoder) = &self.validity {
            validity_decoder.decode_into(rows_to_skip, num_rows, &mut dest_buffers[..1]);
        }
        self.values
            .decode_into(rows_to_skip, num_rows, &mut dest_buffers[1..]);
    }
}

#[derive(Debug)]
enum PageValidity {
    NoNull,
    SomeNull(Box<dyn PhysicalPageScheduler>),
}

#[derive(Debug)]
pub struct BasicEncoder {
    column_index: u32,
}

impl BasicEncoder {
    pub fn new(column_index: u32) -> Self {
        Self { column_index }
    }

    fn encode_values(arrays: &[ArrayRef]) -> Result<(EncodedBuffer, pb::BufferEncoding)> {
        if *arrays[0].data_type() == DataType::Boolean {
            let values = BitmapEncoder::default().encode(arrays)?;
            Ok((
                values,
                pb::BufferEncoding {
                    buffer_encoding: Some(pb::buffer_encoding::BufferEncoding::Bitmap(
                        pb::Bitmap { buffer: None },
                    )),
                },
            ))
        } else {
            let bytes_per_value = arrays[0].data_type().byte_width() as u64;
            debug_assert!(bytes_per_value > 0);
            let values = ValueEncoder::default().encode(arrays)?;
            Ok((
                values,
                pb::BufferEncoding {
                    buffer_encoding: Some(pb::buffer_encoding::BufferEncoding::Value(pb::Value {
                        buffer: None,
                        bytes_per_value,
                    })),
                },
            ))
        }
    }
}

impl ArrayEncoder for BasicEncoder {
    fn encode(&self, arrays: &[ArrayRef]) -> Result<Vec<EncodedPage>> {
        let (null_count, row_count) = arrays
            .iter()
            .map(|arr| (arr.null_count() as u32, arr.len() as u32))
            .fold((0, 0), |acc, val| (acc.0 + val.0, acc.1 + val.1));
        let (buffers, nullability) = if null_count == 0 {
            let (values_buffer, values_encoding) = Self::encode_values(arrays)?;
            let encoding = pb::basic::Nullability::NoNulls(pb::basic::NoNull {
                values: Some(values_encoding),
            });
            (vec![values_buffer], encoding)
        } else if null_count == row_count {
            let encoding = pb::basic::Nullability::AllNulls(pb::basic::AllNull {});
            (vec![], encoding)
        } else {
            let validity_encoding = pb::BufferEncoding {
                buffer_encoding: Some(pb::buffer_encoding::BufferEncoding::Bitmap(pb::Bitmap {
                    buffer: None,
                })),
            };
            let (values_buffer, values_encoding) = Self::encode_values(arrays)?;
            let encoding = pb::basic::Nullability::SomeNulls(pb::basic::SomeNull {
                validity: Some(validity_encoding),
                values: Some(values_encoding),
            });
            let validity = BitmapEncoder::default().encode(arrays)?;
            (vec![validity, values_buffer], encoding)
        };

        Ok(vec![EncodedPage {
            buffers,
            encoding: pb::ArrayEncoding {
                array_encoding: Some(pb::array_encoding::ArrayEncoding::Basic(pb::Basic {
                    nullability: Some(nullability),
                })),
            },
            num_rows: row_count,
            column_idx: self.column_index,
        }])
    }
}
