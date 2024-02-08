use arrow_array::ArrayRef;
use arrow_buffer::Buffer;

use arrow_schema::{DataType, Field, Schema};
use futures::future::BoxFuture;
use lance_core::Result;

use std::sync::Arc;

use crate::{
    encodings::{
        logical::{list::ListFieldEncoder, primitive::PrimitiveFieldEncoder},
        physical::basic::BasicEncoder,
    },
    format::pb,
};

/// An encoded buffer
pub struct EncodedBuffer {
    /// If true, the buffer should be stored as "data"
    /// If false, the buffer should be stored as "metadata"
    ///
    /// Metadata buffers are typically small buffers that should be cached.  For example,
    /// this might be a small dictionary when data has been dictionary encoded.  Or it might
    /// contain a skip block when data has been RLE encoded.
    pub is_data: bool,
    /// Buffers that make up the encoded buffer
    ///
    /// All of these buffers should be written to the file as one contiguous buffer
    ///
    /// This is a Vec to allow for zero-copy
    ///
    /// For example, if we are asked to write 3 primitive arrays of 1000 rows and we can write them all
    /// as one page then this will be the value buffers from the 3 primitive arrays
    pub parts: Vec<Buffer>,
}

pub struct EncodedPage {
    /// The encoded buffers
    pub buffers: Vec<EncodedBuffer>,
    /// A description of the encoding used to encode the column
    pub encoding: pb::ArrayEncoding,
    /// The logical length of the encoded page
    pub num_rows: u32,
    /// The index of the column
    pub column_idx: u32,
}

pub trait BufferEncoder: std::fmt::Debug + Send + Sync {
    /// Encode data
    ///
    /// This method may receive multiple chunks and should encode them all into
    /// a single encoded buffer.
    fn encode(&self, arrays: &[ArrayRef]) -> Result<EncodedBuffer>;
}

/// Encodes data from Arrow format into some kind of on-disk format
///
/// The encoder is responsible for looking at the incoming data and determining
/// which encoding is most appropriate.  It then needs to actually encode that
/// data according to the chosen encoding.
///
/// The array encoder must be Send + Sync.  Encoding is always done on its own
/// thread task in the background and there could potentially be multiple encode
/// tasks running for a column at once (TODO: Not entirely sure this is true)
pub trait ArrayEncoder: std::fmt::Debug + Send + Sync {
    /// Encode data
    ///
    /// This method may receive multiple chunks and should encode them all into
    /// a single encoded array.
    ///
    /// The result should contain the encoded buffers and a description of the
    /// encoding that was chosen.  This can be used to decode the data later.
    fn encode(&self, arrays: &[ArrayRef]) -> Result<Vec<EncodedPage>>;
}

pub trait FieldEncoder {
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
    ) -> Option<BoxFuture<'static, Result<Vec<EncodedPage>>>>;
    fn flush(&mut self) -> Option<BoxFuture<'static, Result<Vec<EncodedPage>>>>;
    fn num_columns(&self) -> u32;
}

pub struct BatchEncoder {
    pub field_encoders: Vec<Box<dyn FieldEncoder>>,
}

impl BatchEncoder {
    fn get_encoder_for_field(
        field: &Field,
        cache_bytes_per_column: u64,
        col_idx: &mut u32,
    ) -> Result<Box<dyn FieldEncoder>> {
        match field.data_type() {
            DataType::Boolean
            | DataType::Date32
            | DataType::Date64
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
            | DataType::Duration(_)
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Int8
            | DataType::Interval(_)
            | DataType::Null
            | DataType::RunEndEncoded(_, _)
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::UInt8 => {
                let my_col_idx = *col_idx;
                *col_idx += 1;
                Ok(Box::new(PrimitiveFieldEncoder::new(
                    cache_bytes_per_column,
                    Arc::new(BasicEncoder::new(my_col_idx)),
                )))
            }
            DataType::List(inner_type) => {
                let my_col_idx = *col_idx;
                *col_idx += 1;
                let inner_encoding =
                    Self::get_encoder_for_field(inner_type, cache_bytes_per_column, col_idx)?;
                Ok(Box::new(ListFieldEncoder::new(
                    inner_encoding,
                    cache_bytes_per_column,
                    my_col_idx,
                )))
            }
            _ => todo!(),
        }
    }

    pub fn try_new(schema: &Schema, cache_bytes_per_column: u64) -> Result<Self> {
        let mut col_idx = 0;
        let field_encoders = schema
            .fields
            .iter()
            .map(|field| Self::get_encoder_for_field(field, cache_bytes_per_column, &mut col_idx))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { field_encoders })
    }

    pub fn num_columns(&self) -> u32 {
        self.field_encoders
            .iter()
            .map(|field_encoder| field_encoder.num_columns())
            .sum::<u32>()
    }
}

fn fill_in_buffer_locations_buffer(
    encoding: &mut pb::BufferEncoding,
    buffers_iter: &mut impl Iterator<Item = (u64, u64)>,
) {
    match encoding.buffer_encoding.as_mut().unwrap() {
        pb::buffer_encoding::BufferEncoding::Value(value) => {
            let next_buf = buffers_iter.next().unwrap();
            value.buffer = Some(pb::Buffer {
                file_offset: next_buf.0,
                buffer_size: next_buf.1,
            })
        }
        pb::buffer_encoding::BufferEncoding::Bitmap(bitmap) => {
            let next_buf = buffers_iter.next().unwrap();
            bitmap.buffer = Some(pb::Buffer {
                file_offset: next_buf.0,
                buffer_size: next_buf.1,
            })
        }
    }
}

/// Updates `encoding` with buffer locations
///
/// When a page is originally encoded as part of [`Self::encode`] we don't
/// know where the page (and its buffers) will actually be located on disk.
/// Once the page has been persisted we update the metadata with the buffer
/// locations.
fn fill_in_buffer_locations_array(
    encoding: &mut pb::ArrayEncoding,
    buffers_iter: &mut impl Iterator<Item = (u64, u64)>,
) {
    let arr_encoding = encoding.array_encoding.as_mut().unwrap();
    match arr_encoding {
        pb::array_encoding::ArrayEncoding::Basic(basic) => {
            match basic.nullability.as_mut().unwrap() {
                pb::basic::Nullability::AllNulls(_) => {}
                pb::basic::Nullability::NoNulls(no_nulls) => {
                    fill_in_buffer_locations_buffer(
                        no_nulls.values.as_mut().unwrap(),
                        buffers_iter,
                    );
                }
                pb::basic::Nullability::SomeNulls(some_nulls) => {
                    fill_in_buffer_locations_buffer(
                        some_nulls.validity.as_mut().unwrap(),
                        buffers_iter,
                    );
                    fill_in_buffer_locations_buffer(
                        some_nulls.values.as_mut().unwrap(),
                        buffers_iter,
                    );
                }
            }
        }
        pb::array_encoding::ArrayEncoding::List(list) => {
            fill_in_buffer_locations_array(list.offsets.as_mut().unwrap(), buffers_iter)
        }
        pb::array_encoding::ArrayEncoding::Struct(_) => {}
    };
}

pub fn fill_in_buffer_locations(
    encoding: &mut pb::ArrayEncoding,
    buffers: impl IntoIterator<Item = (u64, u64)>,
) {
    let mut buffers_iter = buffers.into_iter();
    fill_in_buffer_locations_array(encoding, &mut buffers_iter);
}
