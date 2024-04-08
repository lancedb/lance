use crate::{decoder::PhysicalPageScheduler, format::pb};

use self::{
    basic::BasicPageScheduler, bitmap::DenseBitmapScheduler, fixed_size_list::FixedListScheduler,
    value::ValuePageScheduler,
};

pub mod basic;
pub mod bitmap;
pub mod buffers;
pub mod fixed_size_list;
pub mod value;

/// These contain the file buffers shared across the entire file
#[derive(Clone, Copy, Debug)]
pub struct FileBuffers<'a> {
    pub positions: &'a Vec<u64>,
}

/// These contain the file buffers and also buffers specific to a column
#[derive(Clone, Copy, Debug)]
pub struct ColumnBuffers<'a, 'b> {
    pub file_buffers: FileBuffers<'a>,
    pub positions: &'b Vec<u64>,
}
/// These contain the file & column buffers and also buffers specific to a page
#[derive(Clone, Copy, Debug)]
pub struct PageBuffers<'a, 'b, 'c> {
    pub column_buffers: ColumnBuffers<'a, 'b>,
    pub positions: &'c Vec<u64>,
}

// Translate a protobuf buffer description into a position in the file.  This could be a page
// buffer, a column buffer, or a file buffer.
fn get_buffer(buffer_desc: &pb::Buffer, buffers: &PageBuffers) -> u64 {
    match pb::buffer::BufferType::try_from(buffer_desc.buffer_type).unwrap() {
        pb::buffer::BufferType::Page => buffers.positions[buffer_desc.buffer_index as usize],
        pb::buffer::BufferType::Column => {
            buffers.column_buffers.positions[buffer_desc.buffer_index as usize]
        }
        pb::buffer::BufferType::File => {
            buffers.column_buffers.file_buffers.positions[buffer_desc.buffer_index as usize]
        }
    }
}

/// Convert a protobuf buffer encoding into a physical page scheduler
fn decoder_from_buffer_encoding(
    encoding: &pb::BufferEncoding,
    buffers: &PageBuffers,
) -> Box<dyn PhysicalPageScheduler> {
    match encoding.buffer_encoding.as_ref().unwrap() {
        pb::buffer_encoding::BufferEncoding::Flat(flat) => Box::new(ValuePageScheduler::new(
            flat.bytes_per_value,
            get_buffer(flat.buffer.as_ref().unwrap(), buffers),
        )),
        pb::buffer_encoding::BufferEncoding::Bitmap(bitmap) => Box::new(DenseBitmapScheduler::new(
            get_buffer(bitmap.buffer.as_ref().unwrap(), buffers),
        )),
    }
}

/// Convert a protobuf array encoding into a physical page scheduler
pub fn decoder_from_array_encoding(
    encoding: &pb::ArrayEncoding,
    buffers: &PageBuffers,
) -> Box<dyn PhysicalPageScheduler> {
    match encoding.array_encoding.as_ref().unwrap() {
        pb::array_encoding::ArrayEncoding::Nullable(basic) => {
            match basic.nullability.as_ref().unwrap() {
                pb::nullable::Nullability::NoNulls(no_nulls) => {
                    Box::new(BasicPageScheduler::new_non_nullable(
                        decoder_from_array_encoding(no_nulls.values.as_ref().unwrap(), buffers),
                    ))
                }
                pb::nullable::Nullability::SomeNulls(some_nulls) => {
                    Box::new(BasicPageScheduler::new_nullable(
                        decoder_from_buffer_encoding(
                            some_nulls.validity.as_ref().unwrap(),
                            buffers,
                        ),
                        decoder_from_array_encoding(some_nulls.values.as_ref().unwrap(), buffers),
                    ))
                }
                pb::nullable::Nullability::AllNulls(_) => todo!(),
            }
        }
        pb::array_encoding::ArrayEncoding::Value(value) => {
            decoder_from_buffer_encoding(value.buffer.as_ref().unwrap(), buffers)
        }
        pb::array_encoding::ArrayEncoding::FixedSizeList(fixed_size_list) => {
            let item_encoding = fixed_size_list.items.as_ref().unwrap();
            let item_scheduler = decoder_from_array_encoding(item_encoding, buffers);
            Box::new(FixedListScheduler::new(
                item_scheduler,
                fixed_size_list.dimension,
            ))
        }
        // This is a column containing the list offsets.  This wrapper is superfluous at the moment
        // since we know it is a list based on the schema.  In the future there may be different ways
        // of storing the list offsets.
        pb::array_encoding::ArrayEncoding::List(list) => {
            decoder_from_array_encoding(list.offsets.as_ref().unwrap(), buffers)
        }
        // Currently there is no way to encode struct nullability and structs are encoded with a "header" column
        // (that has no data).  We never actually decode that column and so this branch is never actually encountered.
        //
        // This will change in the future when we add support for struct nullability.
        pb::array_encoding::ArrayEncoding::Struct(_) => unreachable!(),
    }
}
