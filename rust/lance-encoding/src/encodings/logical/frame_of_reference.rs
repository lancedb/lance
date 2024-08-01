use std::ops::Range;
use std::sync::Arc;

use arrow::array::downcast_array;
use arrow_array::{ArrayRef, downcast_primitive, PrimitiveArray, UInt64Array};
use arrow_schema::DataType;
use futures::future::BoxFuture;
use lance_core::Result;

use crate::EncodingsIo;
use crate::decoder::{PageScheduler, PrimitivePageDecoder};
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
    
}

impl PageScheduler for FrameOfReferencePageScheduler {
    fn schedule_ranges(
            &self,
            ranges: &[Range<u64>],
            scheduler: &Arc<dyn EncodingsIo>,
            top_level_row: u64,
        ) -> BoxFuture<'static, Result<Box<dyn PrimitivePageDecoder>>> {
 
        println!("I'll try to create FOR page scheduler");
        todo!()
    }
}



