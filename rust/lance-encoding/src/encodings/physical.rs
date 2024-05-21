// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

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
    pub sizes: &'a Vec<u64>,
}

/// These contain the file buffers and also buffers specific to a column
#[derive(Clone, Copy, Debug)]
pub struct ColumnBuffers<'a, 'b> {
    pub file_buffers: FileBuffers<'a>,
    pub positions: &'b Vec<u64>,
    pub sizes: &'b Vec<u64>,
}

/// These contain the file & column buffers and also buffers specific to a page
#[derive(Clone, Copy, Debug)]
pub struct PageBuffers<'a, 'b, 'c> {
    pub column_buffers: ColumnBuffers<'a, 'b>,
    pub positions: &'c Vec<u64>,
    pub sizes: &'c Vec<u64>,
}

// Translate a protobuf buffer description into a position in the file.  This could be a page
// buffer, a column buffer, or a file buffer.
fn get_buffer(buffer_desc: &pb::Buffer, buffers: &PageBuffers) -> (u64, u64) {
    fn get_from_buffers<'a>(
        positions: &'a Vec<u64>,
        sizes: &'a Vec<u64>,
        index: usize,
    ) -> (u64, u64) {
        (positions[index], sizes[index])
    }

    let index = buffer_desc.buffer_index as usize;

    match pb::buffer::BufferType::try_from(buffer_desc.buffer_type).unwrap() {
        pb::buffer::BufferType::Page => get_from_buffers(buffers.positions, buffers.sizes, index),
        pb::buffer::BufferType::Column => get_from_buffers(
            buffers.column_buffers.positions,
            buffers.column_buffers.sizes,
            index,
        ),
        pb::buffer::BufferType::File => get_from_buffers(
            buffers.column_buffers.file_buffers.positions,
            buffers.column_buffers.file_buffers.sizes,
            index,
        ),
    }
}

/// Convert a protobuf buffer encoding into a physical page scheduler
fn get_buffer_decoder(
    encoding: &pb::Flat,
    buffers: &PageBuffers,
) -> Box<dyn PhysicalPageScheduler> {
    let (buffer_offset, buffer_size) = get_buffer(encoding.buffer.as_ref().unwrap(), buffers);
    match encoding.bits_per_value {
        1 => Box::new(DenseBitmapScheduler::new(buffer_offset)),
        bits_per_value => {
            if bits_per_value % 8 != 0 {
                todo!("bits_per_value that are not multiples of 8");
            }
            Box::new(ValuePageScheduler::new(
                bits_per_value / 8,
                buffer_offset,
                buffer_size,
            ))
        }
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
                        decoder_from_array_encoding(some_nulls.validity.as_ref().unwrap(), buffers),
                        decoder_from_array_encoding(some_nulls.values.as_ref().unwrap(), buffers),
                    ))
                }
                pb::nullable::Nullability::AllNulls(_) => {
                    Box::new(BasicPageScheduler::new_all_null())
                }
            }
        }
        pb::array_encoding::ArrayEncoding::Flat(flat) => get_buffer_decoder(flat, buffers),
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
