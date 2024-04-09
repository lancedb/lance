// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

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

struct DataDecoders {
    validity: Box<dyn PhysicalPageDecoder>,
    values: Box<dyn PhysicalPageDecoder>,
}

enum DataValidity {
    // Neither validity nor values
    AllNull,
    // Values only
    NoNull(Box<dyn PhysicalPageDecoder>),
    // Validity and values
    SomeNull(DataDecoders),
}

impl DataValidity {
    fn values_decoder(&self) -> Option<&Box<dyn PhysicalPageDecoder>> {
        match self {
            DataValidity::AllNull => None,
            DataValidity::SomeNull(decoders) => Some(&decoders.values),
            DataValidity::NoNull(values) => Some(values),
        }
    }
}

#[derive(Debug)]
struct DataSchedulers {
    validity: Box<dyn PhysicalPageScheduler>,
    values: Box<dyn PhysicalPageScheduler>,
}

#[derive(Debug)]
enum SchedulerValidity {
    // Values only
    NoNull(Box<dyn PhysicalPageScheduler>),
    // Validity and values
    SomeNull(DataSchedulers),
    // Neither validity nor values
    AllNull,
}

impl SchedulerValidity {
    fn values_scheduler(&self) -> Option<&Box<dyn PhysicalPageScheduler>> {
        match self {
            SchedulerValidity::AllNull => None,
            SchedulerValidity::NoNull(values) => Some(values),
            SchedulerValidity::SomeNull(schedulers) => Some(&schedulers.values),
        }
    }
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
    mode: SchedulerValidity,
}

impl BasicPageScheduler {
    /// Creates a new instance that expects a validity bitmap
    pub fn new_nullable(
        validity_decoder: Box<dyn PhysicalPageScheduler>,
        values_decoder: Box<dyn PhysicalPageScheduler>,
    ) -> Self {
        Self {
            mode: SchedulerValidity::SomeNull(DataSchedulers {
                validity: validity_decoder,
                values: values_decoder,
            }),
        }
    }

    /// Create a new instance that does not need a validity bitmap because no item is null
    pub fn new_non_nullable(values_decoder: Box<dyn PhysicalPageScheduler>) -> Self {
        Self {
            mode: SchedulerValidity::NoNull(values_decoder),
        }
    }

    /// Create a new instance where all values are null
    ///
    /// It may seem strange we need `values_decoder` here but Arrow requires that value
    /// buffers still be allocated / sized even if everything is null.  So we need the value
    /// decoder to calculate the capcity of the garbage buffer.
    pub fn new_all_null() -> Self {
        Self {
            mode: SchedulerValidity::AllNull,
        }
    }
}

impl PhysicalPageScheduler for BasicPageScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u32>],
        scheduler: &dyn EncodingsIo,
    ) -> BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>> {
        let validity_future = match &self.mode {
            SchedulerValidity::NoNull(_) | SchedulerValidity::AllNull => None,
            SchedulerValidity::SomeNull(schedulers) => {
                trace!("Scheduling ranges {:?} from validity", ranges);
                Some(schedulers.validity.schedule_ranges(ranges, scheduler))
            }
        };

        let values_future = if let Some(values_scheduler) = self.mode.values_scheduler() {
            trace!("Scheduling range {:?} from values", ranges);
            Some(values_scheduler.schedule_ranges(ranges, scheduler).boxed())
        } else {
            trace!("No values fetch needed since values all null");
            None
        };

        async move {
            let mode = match (values_future, validity_future) {
                (None, None) => DataValidity::AllNull,
                (Some(values_future), None) => DataValidity::NoNull(values_future.await?),
                (Some(values_future), Some(validity_future)) => {
                    DataValidity::SomeNull(DataDecoders {
                        values: values_future.await?,
                        validity: validity_future.await?,
                    })
                }
                _ => unreachable!(),
            };
            Ok(Box::new(BasicPageDecoder { mode }) as Box<dyn PhysicalPageDecoder>)
        }
        .boxed()
    }
}

struct BasicPageDecoder {
    mode: DataValidity,
}

impl PhysicalPageDecoder for BasicPageDecoder {
    fn update_capacity(
        &self,
        rows_to_skip: u32,
        num_rows: u32,
        buffers: &mut [(u64, bool)],
        all_null: &mut bool,
    ) {
        // No need to look at the validity decoder to know the dest buffer size since it is boolean
        buffers[0].0 = arrow_buffer::bit_util::ceil(num_rows as usize, 8) as u64;
        // The validity buffer is only required if we have some nulls
        buffers[0].1 = matches!(self.mode, DataValidity::SomeNull(_));
        if let Some(values) = self.mode.values_decoder() {
            values.update_capacity(rows_to_skip, num_rows, &mut buffers[1..], all_null);
        } else {
            *all_null = true;
        }
    }

    fn decode_into(&self, rows_to_skip: u32, num_rows: u32, dest_buffers: &mut [bytes::BytesMut]) {
        match &self.mode {
            DataValidity::SomeNull(decoders) => {
                decoders
                    .validity
                    .decode_into(rows_to_skip, num_rows, &mut dest_buffers[..1]);
                decoders
                    .values
                    .decode_into(rows_to_skip, num_rows, &mut dest_buffers[1..]);
            }
            // Either dest_buffers[0] is empty, in which case these are no-ops, or one of the
            // other pages needed the buffer, in which case we need to fill our section
            DataValidity::AllNull => {
                dest_buffers[0].fill(0);
            }
            DataValidity::NoNull(values) => {
                dest_buffers[0].fill(1);
                values.decode_into(rows_to_skip, num_rows, &mut dest_buffers[1..]);
            }
        }
    }

    fn num_buffers(&self) -> u32 {
        1 + self
            .mode
            .values_decoder()
            .map(|val| val.num_buffers())
            .unwrap_or(0)
    }
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
