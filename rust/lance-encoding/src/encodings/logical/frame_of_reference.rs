use std::ops::Range;
use std::sync::Arc;

use arrow::array::downcast_array;
use arrow_array::{make_array, ArrayRef, UInt64Array};
use arrow_schema::DataType;
use futures::future::{BoxFuture, FutureExt};
use lance_core::Result;

use crate::buffer::LanceBuffer;
use crate::data::{DataBlock, DataBlockExt, FixedWidthDataBlock, NullableDataBlock};
use crate::decoder::{PageScheduler, PrimitivePageDecoder};
use crate::encoder::{ArrayEncoder, EncodedArray};
use crate::format::pb;
use crate::EncodingsIo;

pub fn get_frame_of_reference(arr: ArrayRef) -> Option<u64> {
    let tmp_as_u64: UInt64Array = downcast_array(arr.as_ref());
    arrow::compute::min(&tmp_as_u64)
}

#[derive(Debug)]
pub struct FrameOfReferenceEncoder {
    inner_encoder: Box<dyn ArrayEncoder>,
}

impl FrameOfReferenceEncoder {
    pub fn new(inner_encoder: Box<dyn ArrayEncoder>) -> Self {
        Self { inner_encoder }
    }
}

impl ArrayEncoder for FrameOfReferenceEncoder {
    fn encode(&self, arrays: &[ArrayRef], buffer_index: &mut u32) -> Result<EncodedArray> {
        let frame_of_reference = arrays
            .iter()
            .map(|e| get_frame_of_reference(e.clone()))
            .collect::<Vec<_>>();

        for opt in &frame_of_reference {
            if opt.is_none() {
                println!("returning early");
                return self.inner_encoder.encode(arrays, buffer_index);
            }
        }

        // TODO safe to unwrap?
        let frame_of_reference = frame_of_reference.iter().map(|e| e.unwrap()).min().unwrap();
            
        let mut new_arrs = vec![];
        for arr in arrays {
            let tmp_arr = arr.clone();
            let arr_as_u64: UInt64Array = downcast_array(tmp_arr.as_ref());
            let frame_of_reference = UInt64Array::new_scalar(frame_of_reference);
            let new_arr = arrow::compute::kernels::numeric::sub(
                &arr_as_u64,
                &frame_of_reference
            ).unwrap(); // TODO safe to unwarp?
            new_arrs.push(Arc::new(new_arr) as ArrayRef);
        }

        let inner = self.inner_encoder.encode(&new_arrs, buffer_index)?;

        let array_encoding =
            pb::array_encoding::ArrayEncoding::FrameOfReference(Box::new(pb::FrameOfReference {
                inner: Some(Box::new(inner.encoding)),
                frame_of_reference: frame_of_reference as i64,
                negative: false,
            }));
        
        Ok(EncodedArray {
            buffers: inner.buffers,
            encoding: pb::ArrayEncoding {
                array_encoding: Some(array_encoding),
            },
        })
    }
}

#[derive(Debug)]
pub struct FrameOfReferencePageScheduler {
    // TODO handle negative
    frame_of_reference: u64,
    inner_scheduler: Box<dyn PageScheduler>,
}

impl FrameOfReferencePageScheduler {
    pub fn new(frame_of_reference: u64, inner_scheduler: Box<dyn PageScheduler>) -> Self {
        Self {
            frame_of_reference,
            inner_scheduler,
        }
    }
}

impl PageScheduler for FrameOfReferencePageScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[Range<u64>],
        scheduler: &Arc<dyn EncodingsIo>,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PrimitivePageDecoder>>> {
        // TODO -- need to handle case where page is compressed?

        let frame_of_reference = self.frame_of_reference;
        let values_scheduler_future =
            self.inner_scheduler
                .schedule_ranges(ranges, scheduler, top_level_row);
        async move {
            let values_decoder = values_scheduler_future.await?;
            Ok(Box::new(FrameOfReferenceDecoder {
                frame_of_reference,
                inner_decoder: values_decoder,
            }) as Box<dyn PrimitivePageDecoder>)
        }
        .boxed()
    }
}

pub struct FrameOfReferenceDecoder {
    frame_of_reference: u64,
    inner_decoder: Box<dyn PrimitivePageDecoder>,
}

impl PrimitivePageDecoder for FrameOfReferenceDecoder {
    fn decode(&self, rows_to_skip: u64, num_rows: u64) -> Result<Box<dyn DataBlock>> {
        let inner_data = self.inner_decoder.decode(rows_to_skip, num_rows)?;
        let inner_data = inner_data.as_any_box();
        let (fixed_width_data, nulls) = match inner_data.downcast::<NullableDataBlock>() {
            Ok(nullable) => {
                let data = nullable.data.try_into_layout::<FixedWidthDataBlock>()?;
                Result::Ok((data, Some(nullable.nulls)))
            }
            Err(data) => {
                let data = data.downcast::<FixedWidthDataBlock>().unwrap();
                Ok((data, None))
            }
        }?;

        let num_values = fixed_width_data.num_values;
        let bits_per_value = fixed_width_data.bits_per_value;

        // convert to array
        let arrow_arr = fixed_width_data.into_arrow(DataType::UInt64, true)?;

        let arr = make_array(arrow_arr);

        // add FOR
        let frame_of_ref = UInt64Array::new_scalar(self.frame_of_reference);
        let new_arr = arrow::compute::kernels::numeric::add(&arr, &frame_of_ref)?;
        let array_data = new_arr.to_data();

        // convert to lance_buffer
        let buffers = array_data.buffers();
        debug_assert_eq!(buffers.len(), 1);
        let buffer = buffers[0].clone();
        let lance_buffer = LanceBuffer::Borrowed(buffer);

        let new_data = Box::new(FixedWidthDataBlock {
            bits_per_value,
            num_values,
            data: lance_buffer,
        });

        if let Some(nulls) = nulls {
            Ok(Box::new(NullableDataBlock {
                data: new_data,
                nulls,
            }))
        } else {
            Ok(new_data)
        }
    }
}
