use arrow_array::{cast::AsArray, ArrayRef};

use arrow_buffer::BooleanBufferBuilder;
use arrow_schema::DataType;
use lance_arrow::DataTypeExt;
use lance_core::Result;

use crate::{
    encoder::{BufferEncoder, EncodedBuffer},
    format::pb,
};

#[derive(Debug, Default)]
pub struct FlatBufferEncoder {}

impl BufferEncoder for FlatBufferEncoder {
    fn encode(
        &self,
        arrays: &[ArrayRef],
        buffer_index: u32,
        buffer_type: pb::buffer::BufferType,
    ) -> Result<EncodedBuffer> {
        let bytes_per_value = arrays[0].data_type().byte_width() as u64;
        let parts = arrays
            .iter()
            .map(|arr| arr.to_data().buffers()[0].clone())
            .collect::<Vec<_>>();
        Ok(EncodedBuffer {
            parts,
            encoding: pb::BufferEncoding {
                buffer_encoding: Some(pb::buffer_encoding::BufferEncoding::Flat(pb::Flat {
                    buffer: Some(pb::Buffer {
                        buffer_index,
                        buffer_type: buffer_type as i32,
                    }),
                    bytes_per_value,
                })),
            },
        })
    }
}

// Encoder for writing boolean arrays as dense bitmaps
#[derive(Debug, Default)]
pub struct BitmapBufferEncoder {}

impl BufferEncoder for BitmapBufferEncoder {
    fn encode(
        &self,
        arrays: &[ArrayRef],
        buffer_index: u32,
        buffer_type: pb::buffer::BufferType,
    ) -> Result<EncodedBuffer> {
        debug_assert!(arrays
            .iter()
            .all(|arr| *arr.data_type() == DataType::Boolean));
        let num_rows: u32 = arrays.iter().map(|arr| arr.len() as u32).sum();
        // Empty pages don't make sense, this should be prevented before we
        // get here
        debug_assert_ne!(num_rows, 0);
        // We can't just write the inner value buffers one after the other because
        // bitmaps can have junk padding at the end (e.g. a boolean array with 12
        // values will be 2 bytes but the last four bits of the second byte are
        // garbage).  So we go ahead and pay the cost of a copy (we could avoid this
        // if we really needed to, at the expense of more complicated code and a slightly
        // larger encoded size but writer cost generally doesn't matter as much as reader cost)
        let mut builder = BooleanBufferBuilder::new(num_rows as usize);
        for arr in arrays {
            let bool_arr = arr.as_boolean();
            builder.append_buffer(bool_arr.values());
        }
        let buffer = builder.finish().into_inner();
        let parts = vec![buffer];
        let buffer = EncodedBuffer {
            parts,
            encoding: pb::BufferEncoding {
                buffer_encoding: Some(pb::buffer_encoding::BufferEncoding::Bitmap(pb::Bitmap {
                    buffer: Some(pb::Buffer {
                        buffer_index,
                        buffer_type: buffer_type as i32,
                    }),
                })),
            },
        };
        Ok(buffer)
    }
}
