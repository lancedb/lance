use std::sync::Arc;

use arrow::array::downcast_array;
use arrow_array::{ArrayRef, downcast_primitive, PrimitiveArray, UInt64Array};
use arrow_schema::DataType;
use lance_core::Result;

use crate::encoder::{ArrayEncoder, ArrayEncodingStrategy, EncodedArray};

fn get_frame_of_reference(arr: ArrayRef) -> u64 {
    let tmp_as_u64: UInt64Array = downcast_array(arr.as_ref());
    arrow::compute::max(&tmp_as_u64).unwrap()
}

#[derive(Debug)]
struct FrameOfReferenceEncoder {
    array_encoding_strategy: Arc<dyn ArrayEncodingStrategy>,
    frame_of_reference: u64,
    negative: bool,
}

impl FrameOfReferenceEncoder {
    pub fn new(
        array_encoding_strategy: Arc<dyn ArrayEncodingStrategy>,
        frame_of_reference: u64,
        negative: bool,
    ) -> Self {
        Self {
            array_encoding_strategy,
            frame_of_reference,
            negative
        }
    }
}

impl ArrayEncoder for FrameOfReferenceEncoder {
    fn encode(&self, arrays: &[ArrayRef], buffer_index: &mut u32) -> Result<EncodedArray> {
        let tmp_arr = arrays[0].clone();
        let tmp_as_u64: UInt64Array = downcast_array(tmp_arr.as_ref());

        let g = tmp_as_u64.clone().into_builder().unwrap().values_slice().into_iter().map(|e| {
            e - self.frame_of_reference
        }).collect::<Vec<u64>>();

        let new_arr = UInt64Array::from(g);

        let new_arrs = vec![Arc::new(new_arr) as ArrayRef];
        let inner_encoder = self.array_encoding_strategy.create_array_encoder(&arrays)?;

        return inner_encoder.encode(&new_arrs, buffer_index);
    }
}
