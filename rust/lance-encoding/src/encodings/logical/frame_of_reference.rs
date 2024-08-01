use std::ops::Range;
use std::sync::Arc;

use arrow::array::downcast_array;
use arrow_array::{downcast_primitive, make_array, ArrayRef, PrimitiveArray, UInt64Array};
use arrow_buffer::Buffer;
use arrow_schema::DataType;
use futures::future::{BoxFuture, FutureExt};
use lance_core::Result;

use crate::buffer::LanceBuffer;
use crate::EncodingsIo;
use crate::decoder::{PageScheduler, PrimitivePageDecoder};
use crate::data::{DataBlock, DataBlockExt, FixedWidthDataBlock};
use crate::encoder::{ArrayEncoder, ArrayEncodingStrategy, EncodedArray};
use crate::format::pb;

pub fn get_frame_of_reference(arr: ArrayRef) -> u64 {
    let tmp_as_u64: UInt64Array = downcast_array(arr.as_ref());
    arrow::compute::min(&tmp_as_u64).unwrap()
}

#[derive(Debug)]
pub struct FrameOfReferenceEncoder {
    inner_encoder: Box<dyn ArrayEncoder>
}


impl FrameOfReferenceEncoder {
    pub fn new(
        inner_encoder: Box<dyn ArrayEncoder>
    ) -> Self {
        Self {
            inner_encoder
        }
    }
}

impl ArrayEncoder for FrameOfReferenceEncoder {
    fn encode(&self, arrays: &[ArrayRef], buffer_index: &mut u32) -> Result<EncodedArray> {
        let frame_of_reference: u64 = arrays.iter().map(|e| get_frame_of_reference(e.clone())).sum();

        let tmp_arr = arrays[0].clone();
        let tmp_as_u64: UInt64Array = downcast_array(tmp_arr.as_ref());

        // TODO use subtraction here
//        let g = tmp_as_u64.clone().into_builder().unwrap().values_slice().into_iter().map(|e| {
//            e - frame_of_reference
//        }).collect::<Vec<u64>>();
        let g2 = UInt64Array::new_scalar(frame_of_reference);
        let booby = arrow::compute::kernels::numeric::sub(&tmp_as_u64, &g2)?;
        println!("{:?}", booby);

        //let new_arr = UInt64Array::from(g);

        //let new_arrs = vec![Arc::new(new_arr) as ArrayRef];
        let new_arrs = vec![Arc::new(booby) as ArrayRef];

        let inner = self.inner_encoder.encode(&new_arrs, buffer_index)?;


        let array_encoding = pb::array_encoding::ArrayEncoding::FrameOfReference(Box::new(pb::FrameOfReference {
            inner: Some(Box::new(inner.encoding)),
            frame_of_reference: frame_of_reference as i64,
            negative: false,
        }));

        return Ok(EncodedArray {
            buffers: inner.buffers,
            encoding: pb::ArrayEncoding  {
                array_encoding: Some(array_encoding)
            }
        })
        
    }
}

#[derive(Debug)]
pub struct FrameOfReferencePageScheduler {
    // TODO handle negative
    frame_of_reference: u64,
    inner_scheduler: Box<dyn PageScheduler>
}

impl FrameOfReferencePageScheduler {
    pub fn new(
        frame_of_reference: u64, 
        inner_scheduler: Box<dyn PageScheduler>,
    ) -> Self {
        Self {
            frame_of_reference,
            inner_scheduler
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
        let values_scheduler_future = self.inner_scheduler.schedule_ranges(ranges, scheduler, top_level_row);
        async move {
            let values_decoder = values_scheduler_future.await?;
            Ok(Box::new(FrameOfReferenceDecoder {
                frame_of_reference,
                inner_decoder: values_decoder,
            }) as Box<dyn PrimitivePageDecoder>)

        }.boxed()
    }
}


pub struct FrameOfReferenceDecoder {
    frame_of_reference: u64,
    inner_decoder: Box<dyn PrimitivePageDecoder> 
}

impl PrimitivePageDecoder for FrameOfReferenceDecoder {

    fn decode(&self, rows_to_skip: u64, num_rows: u64) -> Result<Box<dyn DataBlock>> {
        let data_block = self.inner_decoder.decode(rows_to_skip, num_rows)?;
        let fl_db = data_block.try_into_layout::<FixedWidthDataBlock>()?;
        let num_values = fl_db.num_values;
        let bits_per_value = fl_db.bits_per_value;

        // convert to array
        let arrow_arr = fl_db.into_arrow(DataType::UInt64, true)?;
        let arr = make_array(arrow_arr);

        // add FOR
        let frame_of_ref = UInt64Array::new_scalar(self.frame_of_reference);
        let new_arr = arrow::compute::kernels::numeric::add(&arr, &frame_of_ref)?;
        let array_data = new_arr.to_data();

        // convert to lance_buffer
        let buffers= array_data.buffers();
        let buffer = buffers[0].clone();
        println!("The number of buffers is {:?}", buffers.len());
        let lance_buffer = LanceBuffer::Borrowed(buffer);

        Ok(Box::new(FixedWidthDataBlock {
            bits_per_value,
            num_values,
            data: lance_buffer,

        }))

    }
}


