use arrow_array::ArrayRef;
use arrow_buffer::Buffer;
use arrow_schema::{DataType, Field, Schema};
use futures::future::BoxFuture;
use lance_core::Result;

use crate::{
    encodings::logical::{
        list::ListFieldEncoder, primitive::PrimitiveFieldEncoder, r#struct::StructFieldEncoder,
        utf8::Utf8FieldEncoder,
    },
    format::pb,
};

/// An encoded buffer
pub struct EncodedBuffer {
    /// Buffers that make up the encoded buffer
    ///
    /// All of these buffers should be written to the file as one contiguous buffer
    ///
    /// This is a Vec to allow for zero-copy
    ///
    /// For example, if we are asked to write 3 primitive arrays of 1000 rows and we can write them all
    /// as one page then this will be the value buffers from the 3 primitive arrays
    pub parts: Vec<Buffer>,
    /// A description of the encoding used to encode the buffer
    pub encoding: pb::BufferEncoding,
}

// Custom impl because buffers shouldn't be included in debug output
impl std::fmt::Debug for EncodedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodedBuffer")
            .field("len", &self.parts.iter().map(|p| p.len()).sum::<usize>())
            .finish()
    }
}

pub struct EncodedArrayBuffer {
    /// The data making up the buffer
    pub parts: Vec<Buffer>,
    /// The index of the buffer in the page
    pub index: u32,
}

// Custom impl because buffers shouldn't be included in debug output
impl std::fmt::Debug for EncodedArrayBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodedBuffer")
            .field("len", &self.parts.iter().map(|p| p.len()).sum::<usize>())
            .field("index", &self.index)
            .finish()
    }
}

/// An encoded array
///
/// Maps to a single Arrow array
///
/// This may contain multiple buffers.  For example, a nullable int32 array will contain two buffers,
/// one for the null bitmap and one for the values
#[derive(Debug)]
pub struct EncodedArray {
    /// The encoded buffers
    pub buffers: Vec<EncodedArrayBuffer>,
    /// A description of the encoding used to encode the array
    pub encoding: pb::ArrayEncoding,
}

/// An encoded page of data
///
/// Maps to a top-level array
///
/// For example, FixedSizeList<Int32> will have two EncodedArray instances and one EncodedPage
#[derive(Debug)]
pub struct EncodedPage {
    // The encoded array data
    pub array: EncodedArray,
    /// The number of rows in the encoded page
    pub num_rows: u32,
    /// The index of the column
    pub column_idx: u32,
}

/// Encodes data into a single buffer
pub trait BufferEncoder: std::fmt::Debug + Send + Sync {
    /// Encode data
    ///
    /// This method may receive multiple chunks and should encode them all into
    /// a single EncodedBuffer (though that buffer may have multiple parts).  All
    /// parts will be written to the file as one contiguous block.
    fn encode(
        &self,
        arrays: &[ArrayRef],
        buffer_index: u32,
        buffer_type: pb::buffer::BufferType,
    ) -> Result<EncodedBuffer>;
}

/// Encodes data from Arrow format into some kind of on-disk format
///
/// The encoder is responsible for looking at the incoming data and determining
/// which encoding is most appropriate.  This may involve calculating statistics,
/// etc.  It then needs to actually encode that data according to the chosen encoding.
///
/// The encoder may even encode the statistics as well (typically in the column
/// metadata) so that the statistics can be used for filtering later.
///
/// The array encoder must be Send + Sync.  Encoding is always done on its own
/// thread task in the background and there could potentially be multiple encode
/// tasks running for a column at once.
///
/// Note: not all Arrow arrays can be encoded using an ArrayEncoder.  Some arrays
/// will be econded into several Lance columns.  For example, a list array or a
/// struct array.  See [FieldEncoder] for the top-level encoding entry point
pub trait ArrayEncoder: std::fmt::Debug + Send + Sync {
    /// Encode data
    ///
    /// This method may receive multiple chunks and should encode them into a
    /// single EncodedPage.
    ///
    /// The result should contain a description of the encoding that was chosen.
    /// This can be used to decode the data later.
    fn encode(&self, arrays: &[ArrayRef], buffer_index: &mut u32) -> Result<EncodedArray>;
}

/// Top level encoding trait to code any Arrow array type into one or more pages.
///
/// The field encoder implements buffering and encoding of a single input column
/// but it may map to multiple output columns.  For example, a list array or struct
/// array will be encoded into multiple columns.
///
/// Also, fields may be encoded at different speeds.  For example, given a struct
/// column with three fields (a boolean field, an int32 field, and a 4096-dimension
/// tensor field) the tensor field is likely to emit encoded pages much more frequently
/// than the boolean field.
pub trait FieldEncoder: Send {
    /// Buffer the data and, if there is enough data in the buffer to form a page, return
    /// an encoding task to encode the data.
    ///
    /// This may return more than one task because a single column may be mapped to multiple
    /// output columns.  For example, if encoding a struct column with three children then
    /// up to three tasks may be returned from each call to maybe_encode.
    ///
    /// It may also return multiple tasks for a single column if the input array is larger
    /// than a single disk page.
    ///
    /// It could also return an empty Vec if there is not enough data yet to encode any pages.
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
    ) -> Result<Vec<BoxFuture<'static, Result<EncodedPage>>>>;
    /// Flush any remaining data from the buffers into encoding tasks
    fn flush(&mut self) -> Result<Vec<BoxFuture<'static, Result<EncodedPage>>>>;
    /// The number of output columns this encoding will create
    fn num_columns(&self) -> u32;
}

pub struct BatchEncoder {
    pub field_encoders: Vec<Box<dyn FieldEncoder>>,
}

impl BatchEncoder {
    pub(crate) fn get_encoder_for_field(
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
            | DataType::UInt8
            | DataType::FixedSizeList(_, _) => {
                let my_col_idx = *col_idx;
                *col_idx += 1;
                Ok(Box::new(PrimitiveFieldEncoder::try_new(
                    cache_bytes_per_column,
                    field.data_type(),
                    my_col_idx,
                )?))
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
            DataType::Struct(fields) => {
                let header_col_idx = *col_idx;
                *col_idx += 1;
                let children_encoders = fields
                    .iter()
                    .map(|field| {
                        Self::get_encoder_for_field(field, cache_bytes_per_column, col_idx)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Box::new(StructFieldEncoder::new(
                    children_encoders,
                    header_col_idx,
                )))
            }
            DataType::Utf8 => {
                let my_col_idx = *col_idx;
                *col_idx += 2;
                Ok(Box::new(Utf8FieldEncoder::new(
                    cache_bytes_per_column,
                    my_col_idx,
                )))
            }
            _ => todo!("Implement encoding for field type: {:?}", field.data_type()),
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
