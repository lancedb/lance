// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::{collections::HashMap, env, sync::Arc};

use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_buffer::Buffer;
use arrow_schema::DataType;
use bytes::{Bytes, BytesMut};
use futures::future::BoxFuture;
use lance_arrow::DataTypeExt;
use lance_core::datatypes::{Field, Schema};
use lance_core::Result;

use crate::encodings::logical::r#struct::StructFieldEncoder;
use crate::encodings::physical::bitpack::{num_compressed_bits, BitpackingBufferEncoder};
use crate::encodings::physical::buffers::{
    BitmapBufferEncoder, CompressedBufferEncoder, FlatBufferEncoder,
};
use crate::encodings::physical::fsst::FsstArrayEncoder;
use crate::encodings::physical::packed_struct::PackedStructEncoder;
use crate::encodings::physical::value::{parse_compression_scheme, CompressionScheme};
use crate::{
    decoder::{ColumnInfo, PageInfo},
    encodings::{
        logical::{list::ListFieldEncoder, primitive::PrimitiveFieldEncoder},
        physical::{
            basic::BasicEncoder, binary::BinaryEncoder, dictionary::DictionaryEncoder,
            fixed_size_list::FslEncoder, value::ValueEncoder,
        },
    },
    format::pb,
};

use hyperloglogplus::{HyperLogLog, HyperLogLogPlus};
use std::collections::hash_map::RandomState;

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
}

// Custom impl because buffers shouldn't be included in debug output
impl std::fmt::Debug for EncodedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodedBuffer")
            .field("len", &self.parts.iter().map(|p| p.len()).sum::<usize>())
            .finish()
    }
}

#[derive(Clone)]
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
/// This may contain multiple EncodedArrayBuffers.  For example, a nullable int32 array will contain two buffers,
/// one for the null bitmap and one for the values
#[derive(Debug, Clone)]
pub struct EncodedArray {
    /// The encoded buffers
    pub buffers: Vec<EncodedArrayBuffer>,
    /// A description of the encoding used to encode the array
    pub encoding: pb::ArrayEncoding,
}

impl EncodedArray {
    pub fn into_parts(mut self) -> (Vec<EncodedBuffer>, pb::ArrayEncoding) {
        self.buffers.sort_by_key(|b| b.index);
        (
            self.buffers
                .into_iter()
                .map(|b| EncodedBuffer { parts: b.parts })
                .collect(),
            self.encoding,
        )
    }
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
    pub num_rows: u64,
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
    fn encode(&self, arrays: &[ArrayRef]) -> Result<(EncodedBuffer, EncodedBufferMeta)>;
}

pub struct EncodedBufferMeta {
    pub bits_per_value: u64,

    pub bitpacked_bits_per_value: Option<u64>,

    pub compression_scheme: Option<CompressionScheme>,
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

pub fn values_column_encoding() -> pb::ColumnEncoding {
    pb::ColumnEncoding {
        column_encoding: Some(pb::column_encoding::ColumnEncoding::Values(())),
    }
}

pub struct EncodedColumn {
    pub column_buffers: Vec<EncodedBuffer>,
    pub encoding: pb::ColumnEncoding,
    pub final_pages: Vec<EncodedPage>,
}

impl Default for EncodedColumn {
    fn default() -> Self {
        Self {
            column_buffers: Default::default(),
            encoding: pb::ColumnEncoding {
                column_encoding: Some(pb::column_encoding::ColumnEncoding::Values(())),
            },
            final_pages: Default::default(),
        }
    }
}

/// A task to create a page of data
pub type EncodeTask = BoxFuture<'static, Result<EncodedPage>>;

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
    fn maybe_encode(&mut self, array: ArrayRef) -> Result<Vec<EncodeTask>>;
    /// Flush any remaining data from the buffers into encoding tasks
    ///
    /// This may be called intermittently throughout encoding but will always be called
    /// once at the end of encoding just before calling finish
    fn flush(&mut self) -> Result<Vec<EncodeTask>>;
    /// Finish encoding and return column metadata
    ///
    /// This is called only once, after all encode tasks have completed
    ///
    /// This returns a Vec because a single field may have created multiple columns
    fn finish(&mut self) -> BoxFuture<'_, Result<Vec<EncodedColumn>>>;

    /// The number of output columns this encoding will create
    fn num_columns(&self) -> u32;
}

/// A trait to pick which encoding strategy to use for a single page
/// of data
///
/// Presumably, implementations will make encoding decisions based on
/// array statistics.
pub trait ArrayEncodingStrategy: Send + Sync + std::fmt::Debug {
    fn create_array_encoder(&self, arrays: &[ArrayRef]) -> Result<Box<dyn ArrayEncoder>>;
}

/// The core array encoding strategy is a set of basic encodings that
/// are generally applicable in most scenarios.
#[derive(Debug, Default)]
pub struct CoreArrayEncodingStrategy;

fn get_compression_scheme() -> CompressionScheme {
    let compression_scheme = std::env::var("LANCE_PAGE_COMPRESSION").unwrap_or("none".to_string());
    parse_compression_scheme(&compression_scheme).unwrap_or(CompressionScheme::None)
}

impl CoreArrayEncodingStrategy {
    fn can_use_fsst(data_type: &DataType, data_size: u64) -> bool {
        std::env::var("LANCE_USE_FSST").is_ok()
            && matches!(data_type, DataType::Utf8 | DataType::Binary)
            && data_size > 4 * 1024 * 1024
    }

    fn array_encoder_from_type(
        data_type: &DataType,
        data_size: u64,
        use_dict_encoding: bool,
    ) -> Result<Box<dyn ArrayEncoder>> {
        match data_type {
            DataType::FixedSizeList(inner, dimension) => {
                Ok(Box::new(BasicEncoder::new(Box::new(FslEncoder::new(
                    Self::array_encoder_from_type(inner.data_type(), data_size, use_dict_encoding)?,
                    *dimension as u32,
                )))))
            }
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => {
                if use_dict_encoding {
                    let dict_indices_encoder =
                        Self::array_encoder_from_type(&DataType::UInt8, data_size, false)?;
                    let dict_items_encoder =
                        Self::array_encoder_from_type(&DataType::Utf8, data_size, false)?;

                    Ok(Box::new(DictionaryEncoder::new(
                        dict_indices_encoder,
                        dict_items_encoder,
                    )))
                } else {
                    let bin_indices_encoder =
                        Self::array_encoder_from_type(&DataType::UInt64, data_size, false)?;
                    let bin_bytes_encoder =
                        Self::array_encoder_from_type(&DataType::UInt8, data_size, false)?;

                    let bin_encoder =
                        Box::new(BinaryEncoder::new(bin_indices_encoder, bin_bytes_encoder));
                    if Self::can_use_fsst(data_type, data_size) {
                        Ok(Box::new(FsstArrayEncoder::new(bin_encoder)))
                    } else {
                        Ok(bin_encoder)
                    }
                }
            }
            DataType::Struct(fields) => {
                let num_fields = fields.len();
                let mut inner_encoders = Vec::new();

                for i in 0..num_fields {
                    let inner_datatype = fields[i].data_type();
                    let inner_encoder = Self::array_encoder_from_type(
                        inner_datatype,
                        data_size,
                        use_dict_encoding,
                    )?;
                    inner_encoders.push(inner_encoder);
                }

                Ok(Box::new(PackedStructEncoder::new(inner_encoders)))
            }
            _ => Ok(Box::new(BasicEncoder::new(Box::new(
                ValueEncoder::try_new(Arc::new(CoreBufferEncodingStrategy {
                    compression_scheme: get_compression_scheme(),
                }))?,
            )))),
        }
    }
}

fn get_dict_encoding_threshold() -> u64 {
    env::var("LANCE_DICT_ENCODING_THRESHOLD")
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(100)
}

// check whether we want to use dictionary encoding or not
// by applying a threshold on cardinality
// returns true if cardinality < threshold but false if the total number of rows is less than the threshold
// The choice to use 100 is just a heuristic for now
// hyperloglog is used for cardinality estimation
// error rate = 1.04 / sqrt(2^p), where p is the precision
// and error rate is 1.04 / sqrt(2^12) = 1.56%
fn check_dict_encoding(arrays: &[ArrayRef], threshold: u64) -> bool {
    let num_total_rows = arrays.iter().map(|arr| arr.len()).sum::<usize>();
    if num_total_rows < threshold as usize {
        return false;
    }
    const PRECISION: u8 = 12;

    let mut hll: HyperLogLogPlus<String, RandomState> =
        HyperLogLogPlus::new(PRECISION, RandomState::new()).unwrap();

    for arr in arrays {
        let string_array = arrow_array::cast::as_string_array(arr);
        for value in string_array.iter().flatten() {
            hll.insert(value);
            let estimated_cardinality = hll.count() as u64;
            if estimated_cardinality >= threshold {
                return false;
            }
        }
    }

    true
}

impl ArrayEncodingStrategy for CoreArrayEncodingStrategy {
    fn create_array_encoder(&self, arrays: &[ArrayRef]) -> Result<Box<dyn ArrayEncoder>> {
        let data_size = arrays
            .iter()
            .map(|arr| arr.get_buffer_memory_size() as u64)
            .sum::<u64>();
        let data_type = arrays[0].data_type();
        let use_dict_encoding = data_type == &DataType::Utf8
            && check_dict_encoding(arrays, get_dict_encoding_threshold());
        Self::array_encoder_from_type(data_type, data_size, use_dict_encoding)
    }
}

/// A trait to pick which encoding strategy will be used for a single buffer of data
pub trait BufferEncodingStrategy: Send + Sync + std::fmt::Debug {
    fn create_buffer_encoder(&self, arrays: &[ArrayRef]) -> Result<Box<dyn BufferEncoder>>;
}

#[derive(Debug)]
pub struct CoreBufferEncodingStrategy {
    pub compression_scheme: CompressionScheme,
}

impl Default for CoreBufferEncodingStrategy {
    fn default() -> Self {
        Self {
            compression_scheme: CompressionScheme::None,
        }
    }
}

impl CoreBufferEncodingStrategy {
    fn try_bitpacked_encoding(&self, arrays: &[ArrayRef]) -> Option<BitpackingBufferEncoder> {
        if std::env::var("LANCE_USE_BITPACKING").is_err() {
            return None;
        }

        // calculate the number of bits to compress array items into
        let mut num_bits = 0;
        for arr in arrays {
            match num_compressed_bits(arr.clone()) {
                Some(arr_max) => num_bits = num_bits.max(arr_max),
                None => return None,
            }
        }

        // check that the number of bits in the compressed array is less than the
        // number of bits in the native type. Otherwise there's no point to bitpacking
        let data_type = arrays[0].data_type();
        let native_num_bits = 8 * data_type.byte_width() as u64;
        if num_bits >= native_num_bits {
            return None;
        }

        Some(BitpackingBufferEncoder::new(num_bits))
    }
}

impl BufferEncodingStrategy for CoreBufferEncodingStrategy {
    fn create_buffer_encoder(&self, arrays: &[ArrayRef]) -> Result<Box<dyn BufferEncoder>> {
        let data_type = arrays[0].data_type();
        if *data_type == DataType::Boolean {
            return Ok(Box::<BitmapBufferEncoder>::default());
        }

        if self.compression_scheme != CompressionScheme::None {
            return Ok(Box::<CompressedBufferEncoder>::default());
        }

        if let Some(bitpacking_encoder) = self.try_bitpacked_encoding(arrays) {
            return Ok(Box::new(bitpacking_encoder));
        }

        Ok(Box::<FlatBufferEncoder>::default())
    }
}

/// Keeps track of the current column index and makes a mapping
/// from field id to column index
#[derive(Default)]
pub struct ColumnIndexSequence {
    current_index: u32,
    mapping: Vec<(i32, i32)>,
}

impl ColumnIndexSequence {
    pub fn next_column_index(&mut self, field_id: i32) -> u32 {
        let idx = self.current_index;
        self.current_index += 1;
        self.mapping.push((field_id, idx as i32));
        idx
    }

    pub fn skip(&mut self) {
        self.current_index += 1;
    }
}

/// A trait to pick which kind of field encoding to use for a field
///
/// Unlike the ArrayEncodingStrategy, the field encoding strategy is
/// chosen before any data is generated and the same field encoder is
/// used for all data in the field.
pub trait FieldEncodingStrategy: Send + Sync + std::fmt::Debug {
    /// Choose and create an appropriate field encoder for the given
    /// field.
    ///
    /// The field encoder can be chosen on the data type as well as
    /// any metadata that is attached to the field.
    ///
    /// The `encoding_strategy_root` is the encoder that should be
    /// used to encode any inner data in struct / list / etc. fields.
    ///
    /// Initially it is the same as `self` and generally should be
    /// forwarded to any inner encoding strategy.
    fn create_field_encoder(
        &self,
        encoding_strategy_root: &dyn FieldEncodingStrategy,
        field: &Field,
        column_index: &mut ColumnIndexSequence,
        cache_bytes_per_column: u64,
        keep_original_array: bool,
        config: &HashMap<String, String>,
    ) -> Result<Box<dyn FieldEncoder>>;
}

/// The core field encoding strategy is a set of basic encodings that
/// are generally applicable in most scenarios.
#[derive(Debug)]
pub struct CoreFieldEncodingStrategy {
    array_encoding_strategy: Arc<dyn ArrayEncodingStrategy>,
}

impl Default for CoreFieldEncodingStrategy {
    fn default() -> Self {
        Self {
            array_encoding_strategy: Arc::new(CoreArrayEncodingStrategy),
        }
    }
}

impl FieldEncodingStrategy for CoreFieldEncodingStrategy {
    fn create_field_encoder(
        &self,
        encoding_strategy_root: &dyn FieldEncodingStrategy,
        field: &Field,
        column_index: &mut ColumnIndexSequence,
        cache_bytes_per_column: u64,
        keep_original_array: bool,
        _config: &HashMap<String, String>,
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
            | DataType::FixedSizeBinary(_)
            | DataType::FixedSizeList(_, _)
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::Utf8
            | DataType::LargeUtf8 => Ok(Box::new(PrimitiveFieldEncoder::try_new(
                cache_bytes_per_column,
                keep_original_array,
                self.array_encoding_strategy.clone(),
                column_index.next_column_index(field.id),
            )?)),
            DataType::List(child) => {
                let list_idx = column_index.next_column_index(field.id);
                let inner_encoding = encoding_strategy_root.create_field_encoder(
                    encoding_strategy_root,
                    &field.children[0],
                    column_index,
                    cache_bytes_per_column,
                    keep_original_array,
                    child.metadata(),
                )?;
                Ok(Box::new(ListFieldEncoder::new(
                    inner_encoding,
                    cache_bytes_per_column,
                    keep_original_array,
                    list_idx,
                )))
            }
            DataType::Struct(_) => {
                let field_metadata = &field.metadata;
                if field_metadata
                    .get("packed")
                    .map(|v| v == "true")
                    .unwrap_or(false)
                {
                    Ok(Box::new(PrimitiveFieldEncoder::try_new(
                        cache_bytes_per_column,
                        keep_original_array,
                        self.array_encoding_strategy.clone(),
                        column_index.next_column_index(field.id),
                    )?))
                } else {
                    let header_idx = column_index.next_column_index(field.id);
                    let children_encoders = field
                        .children
                        .iter()
                        .map(|field| {
                            self.create_field_encoder(
                                encoding_strategy_root,
                                field,
                                column_index,
                                cache_bytes_per_column,
                                keep_original_array,
                                &field.metadata,
                            )
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok(Box::new(StructFieldEncoder::new(
                        children_encoders,
                        header_idx,
                    )))
                }
            }
            _ => todo!("Implement encoding for field {}", field),
        }
    }
}

/// A batch encoder that encodes RecordBatch objects by delegating
/// to field encoders for each top-level field in the batch.
pub struct BatchEncoder {
    pub field_encoders: Vec<Box<dyn FieldEncoder>>,
    pub field_id_to_column_index: Vec<(i32, i32)>,
}

impl BatchEncoder {
    pub fn try_new(
        schema: &Schema,
        strategy: &dyn FieldEncodingStrategy,
        cache_bytes_per_column: u64,
        keep_original_array: bool,
    ) -> Result<Self> {
        let mut col_idx = 0;
        let mut col_idx_sequence = ColumnIndexSequence::default();
        let field_encoders = schema
            .fields
            .iter()
            .map(|field| {
                let encoder = strategy.create_field_encoder(
                    strategy,
                    field,
                    &mut col_idx_sequence,
                    cache_bytes_per_column,
                    keep_original_array,
                    &field.metadata,
                )?;
                col_idx += encoder.as_ref().num_columns();
                Ok(encoder)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            field_encoders,
            field_id_to_column_index: col_idx_sequence.mapping,
        })
    }

    pub fn num_columns(&self) -> u32 {
        self.field_encoders
            .iter()
            .map(|field_encoder| field_encoder.num_columns())
            .sum::<u32>()
    }
}

/// An encoded batch of data and a page table describing it
///
/// This is returned by [`crate::encoder::encode_batch`]
pub struct EncodedBatch {
    pub data: Bytes,
    pub page_table: Vec<Arc<ColumnInfo>>,
    pub schema: Arc<Schema>,
    pub top_level_columns: Vec<u32>,
    pub num_rows: u64,
}

fn write_page_to_data_buffer(page: EncodedPage, data_buffer: &mut BytesMut) -> PageInfo {
    let mut buffers = page.array.buffers;
    buffers.sort_by_key(|b| b.index);
    let mut buffer_offsets_and_sizes = Vec::new();
    for buffer in buffers {
        let buffer_offset = data_buffer.len() as u64;
        for part in buffer.parts {
            data_buffer.extend_from_slice(&part);
        }
        let size = data_buffer.len() as u64 - buffer_offset;
        buffer_offsets_and_sizes.push((buffer_offset, size));
    }
    PageInfo {
        buffer_offsets_and_sizes: Arc::from(buffer_offsets_and_sizes),
        encoding: page.array.encoding,
        num_rows: page.num_rows,
    }
}

/// Helper method to encode a batch of data into memory
///
/// This is primarily for testing and benchmarking but could be useful in other
/// niche situations like IPC.
pub async fn encode_batch(
    batch: &RecordBatch,
    schema: Arc<Schema>,
    encoding_strategy: &dyn FieldEncodingStrategy,
    cache_bytes_per_column: u64,
) -> Result<EncodedBatch> {
    let mut data_buffer = BytesMut::new();
    let lance_schema = Schema::try_from(batch.schema().as_ref())?;
    let batch_encoder = BatchEncoder::try_new(
        &lance_schema,
        encoding_strategy,
        cache_bytes_per_column,
        true,
    )?;
    let mut page_table = Vec::new();
    let mut col_idx_offset = 0;
    for (arr, mut encoder) in batch.columns().iter().zip(batch_encoder.field_encoders) {
        let mut tasks = encoder.maybe_encode(arr.clone())?;
        tasks.extend(encoder.flush()?);
        let mut pages = HashMap::<u32, Vec<PageInfo>>::new();
        for task in tasks {
            let encoded_page = task.await?;
            pages
                .entry(encoded_page.column_idx)
                .or_default()
                .push(write_page_to_data_buffer(encoded_page, &mut data_buffer));
        }
        let encoded_columns = encoder.finish().await?;
        let num_columns = encoded_columns.len();
        for (col_idx, encoded_column) in encoded_columns.into_iter().enumerate() {
            let col_idx = col_idx + col_idx_offset;
            let mut col_buffer_offsets_and_sizes = Vec::new();
            for buffer in encoded_column.column_buffers {
                let buffer_offset = data_buffer.len() as u64;
                for part in buffer.parts {
                    data_buffer.extend_from_slice(&part);
                }
                let size = data_buffer.len() as u64 - buffer_offset;
                col_buffer_offsets_and_sizes.push((buffer_offset, size));
            }
            for page in encoded_column.final_pages {
                pages
                    .entry(page.column_idx)
                    .or_default()
                    .push(write_page_to_data_buffer(page, &mut data_buffer));
            }
            let col_pages = std::mem::take(pages.entry(col_idx as u32).or_default());
            page_table.push(Arc::new(ColumnInfo {
                index: col_idx as u32,
                buffer_offsets_and_sizes: Arc::from(
                    col_buffer_offsets_and_sizes.into_boxed_slice(),
                ),
                page_infos: Arc::from(col_pages.into_boxed_slice()),
                encoding: encoded_column.encoding,
            }))
        }
        col_idx_offset += num_columns;
    }
    let top_level_columns = batch_encoder
        .field_id_to_column_index
        .iter()
        .map(|(_, idx)| *idx as u32)
        .collect();
    Ok(EncodedBatch {
        data: data_buffer.freeze(),
        top_level_columns,
        page_table,
        schema,
        num_rows: batch.num_rows() as u64,
    })
}

#[cfg(test)]
pub mod tests {
    use arrow_array::{ArrayRef, StringArray};
    use std::sync::Arc;

    use super::check_dict_encoding;

    fn is_dict_encoding_applicable(arr: Vec<Option<&str>>, threshold: u64) -> bool {
        let arr = StringArray::from(arr);
        let arr = Arc::new(arr) as ArrayRef;
        check_dict_encoding(&[arr], threshold)
    }

    #[test]
    fn test_dict_encoding_should_be_applied_if_cardinality_less_than_threshold() {
        assert!(is_dict_encoding_applicable(
            vec![Some("a"), Some("b"), Some("a"), Some("b")],
            3,
        ));
    }

    #[test]
    fn test_dict_encoding_should_not_be_applied_if_cardinality_larger_than_threshold() {
        assert!(!is_dict_encoding_applicable(
            vec![Some("a"), Some("b"), Some("c"), Some("d")],
            3,
        ));
    }

    #[test]
    fn test_dict_encoding_should_not_be_applied_if_cardinality_equal_to_threshold() {
        assert!(!is_dict_encoding_applicable(
            vec![Some("a"), Some("b"), Some("c"), Some("a")],
            3,
        ));
    }

    #[test]
    fn test_dict_encoding_should_not_be_applied_for_empty_arrays() {
        assert!(!is_dict_encoding_applicable(vec![], 3));
    }

    #[test]
    fn test_dict_encoding_should_not_be_applied_for_smaller_than_threshold_arrays() {
        assert!(!is_dict_encoding_applicable(vec![Some("a"), Some("a")], 3));
    }
}
