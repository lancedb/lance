use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray};
use arrow_buffer::BooleanBuffer;
use futures::{future::BoxFuture, FutureExt};
use log::trace;

use crate::{
    decoder::{PhysicalPageDecoder, PhysicalPageScheduler},
    encoder::{ArrayEncoder, BufferEncoder, EncodedArray, EncodedArrayBuffer},
    format::pb,
    EncodingsIo,
};

use lance_core::Result;

use super::buffers::BitmapBufferEncoder;

enum DataValidity {
    NoNull,
    SomeNull(Box<dyn PhysicalPageDecoder>),
}

/// A physical scheduler for "basic" fields.  These are fields that have an optional
/// validity bitmap and some kind of values buffer.
///
/// No actual decoding happens here, we are simply aggregating the two buffers.
///
/// If everything is null then there are no data buffers at all.
// TODO: Add support/tests for primitive nulls
// TODO: Add tests for the all-null case
//
// Right now this is always present on primitive fields.  In the future we may use a
// sentinel encoding instead.
#[derive(Debug)]
pub struct BasicPageScheduler {
    validity_decoder: PageValidity,
    values_decoder: Box<dyn PhysicalPageScheduler>,
}

impl BasicPageScheduler {
    /// Creates a new instance that expects a validity bitmap
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
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u32>],
        scheduler: &dyn EncodingsIo,
    ) -> BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>> {
        let validity_future = match &self.validity_decoder {
            PageValidity::NoNull => None,
            PageValidity::SomeNull(validity_decoder) => {
                trace!("Scheduling ranges {:?} from validity", ranges);
                Some(validity_decoder.schedule_ranges(ranges, scheduler))
            }
        };

        trace!("Scheduling range {:?} from values", ranges);
        let values_future = self.values_decoder.schedule_ranges(ranges, scheduler);

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

    fn num_buffers(&self) -> u32 {
        1 + self.values.num_buffers()
    }
}

#[derive(Debug)]
enum PageValidity {
    NoNull,
    SomeNull(Box<dyn PhysicalPageScheduler>),
}

#[derive(Debug)]
pub struct BasicEncoder {
    values_encoder: Box<dyn ArrayEncoder>,
}

impl BasicEncoder {
    pub fn new(values_encoder: Box<dyn ArrayEncoder>) -> Self {
        Self { values_encoder }
    }
}

impl ArrayEncoder for BasicEncoder {
    fn encode(&self, arrays: &[ArrayRef], buffer_index: &mut u32) -> Result<EncodedArray> {
        let (null_count, row_count) = arrays
            .iter()
            .map(|arr| (arr.null_count() as u32, arr.len() as u32))
            .fold((0, 0), |acc, val| (acc.0 + val.0, acc.1 + val.1));
        let (buffers, nullability) = if null_count == 0 {
            let arr_encoding = self.values_encoder.encode(arrays, buffer_index)?;
            let encoding = pb::nullable::Nullability::NoNulls(Box::new(pb::nullable::NoNull {
                values: Some(Box::new(arr_encoding.encoding)),
            }));
            (arr_encoding.buffers, encoding)
        } else if null_count == row_count {
            let encoding = pb::nullable::Nullability::AllNulls(pb::nullable::AllNull {});
            (vec![], encoding)
        } else {
            let validity_as_arrays = arrays
                .iter()
                .map(|arr| {
                    if let Some(nulls) = arr.nulls() {
                        Arc::new(BooleanArray::new(nulls.inner().clone(), None)) as ArrayRef
                    } else {
                        let buff = BooleanBuffer::new_set(arr.len());
                        Arc::new(BooleanArray::new(buff, None)) as ArrayRef
                    }
                })
                .collect::<Vec<_>>();

            let validity_buffer_index = *buffer_index;
            *buffer_index += 1;
            let validity = BitmapBufferEncoder::default().encode(
                &validity_as_arrays,
                validity_buffer_index,
                pb::buffer::BufferType::Page,
            )?;

            let arr_encoding = self.values_encoder.encode(arrays, buffer_index)?;
            let encoding = pb::nullable::Nullability::SomeNulls(Box::new(pb::nullable::SomeNull {
                validity: Some(validity.encoding),
                values: Some(Box::new(arr_encoding.encoding)),
            }));

            let mut buffers = arr_encoding.buffers;
            buffers.push(EncodedArrayBuffer {
                parts: validity.parts,
                index: validity_buffer_index,
            });
            (buffers, encoding)
        };

        Ok(EncodedArray {
            buffers,
            encoding: pb::ArrayEncoding {
                array_encoding: Some(pb::array_encoding::ArrayEncoding::Nullable(Box::new(
                    pb::Nullable {
                        nullability: Some(nullability),
                    },
                ))),
            },
        })
    }
}
