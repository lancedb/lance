// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::{collections::HashMap, env, sync::Arc};

use arrow::array::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch, UInt8Array};
use arrow_schema::DataType;
use bytes::{Bytes, BytesMut};
use futures::future::BoxFuture;
use lance_core::datatypes::{Field, Schema};
use lance_core::{Error, Result};
use snafu::{location, Location};

use crate::buffer::LanceBuffer;
use crate::data::DataBlock;
use crate::encodings::logical::r#struct::StructFieldEncoder;
use crate::encodings::physical::bitpack_fastlanes::compute_compressed_bit_width_for_non_neg;
use crate::encodings::physical::bitpack_fastlanes::BitpackedForNonNegArrayEncoder;
use crate::encodings::physical::block_compress::CompressionScheme;
use crate::encodings::physical::dictionary::AlreadyDictionaryEncoder;
use crate::encodings::physical::fsst::FsstArrayEncoder;
use crate::encodings::physical::packed_struct::PackedStructEncoder;
use crate::version::LanceFileVersion;
use crate::{
    decoder::{ColumnInfo, PageInfo},
    encodings::{
        logical::{list::ListFieldEncoder, primitive::PrimitiveFieldEncoder},
        physical::{
            basic::BasicEncoder, binary::BinaryEncoder, dictionary::DictionaryEncoder,
            fixed_size_binary::FixedSizeBinaryEncoder, fixed_size_list::FslEncoder,
            value::ValueEncoder,
        },
    },
    format::pb,
};

use hyperloglogplus::{HyperLogLog, HyperLogLogPlus};
use std::collections::hash_map::RandomState;

pub const COMPRESSION_META_KEY: &str = "lance-encoding:compression";

/// An encoded array
///
/// Maps to a single Arrow array
///
/// This contains the encoded data as well as a description of the encoding that was applied which
/// can be used to decode the data later.
#[derive(Debug)]
pub struct EncodedArray {
    /// The encoded buffers
    pub data: DataBlock,
    /// A description of the encoding used to encode the array
    pub encoding: pb::ArrayEncoding,
}

impl EncodedArray {
    pub fn new(data: DataBlock, encoding: pb::ArrayEncoding) -> Self {
        Self { data, encoding }
    }

    pub fn into_buffers(self) -> (Vec<LanceBuffer>, pb::ArrayEncoding) {
        let buffers = self.data.into_buffers();
        (buffers, self.encoding)
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

#[derive(Debug)]
pub struct EncodedBufferMeta {
    pub bits_per_value: u64,

    pub bitpacking: Option<BitpackingBufferMeta>,

    pub compression_scheme: Option<CompressionScheme>,
}

#[derive(Debug)]
pub struct BitpackingBufferMeta {
    pub bits_per_value: u64,

    pub signed: bool,
}

/// Encodes data from one format to another (hopefully more compact or useful) format
///
/// The array encoder must be Send + Sync.  Encoding is always done on its own
/// thread task in the background and there could potentially be multiple encode
/// tasks running for a column at once.
pub trait ArrayEncoder: std::fmt::Debug + Send + Sync {
    /// Encode data
    ///
    /// The result should contain a description of the encoding that was chosen.
    /// This can be used to decode the data later.
    fn encode(
        &self,
        data: DataBlock,
        data_type: &DataType,
        buffer_index: &mut u32,
    ) -> Result<EncodedArray>;
}

pub fn values_column_encoding() -> pb::ColumnEncoding {
    pb::ColumnEncoding {
        column_encoding: Some(pb::column_encoding::ColumnEncoding::Values(())),
    }
}

pub struct EncodedColumn {
    pub column_buffers: Vec<LanceBuffer>,
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
    /// Each encode task produces a single page.  The order of these pages will be maintained
    /// in the file (we do not worry about order between columns but all pages in the same
    /// column should maintain order)
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
    fn create_array_encoder(
        &self,
        arrays: &[ArrayRef],
        field: &Field,
    ) -> Result<Box<dyn ArrayEncoder>>;
}

/// The core array encoding strategy is a set of basic encodings that
/// are generally applicable in most scenarios.
#[derive(Debug, Default)]
pub struct CoreArrayEncodingStrategy {
    pub version: LanceFileVersion,
}

const BINARY_DATATYPES: [DataType; 4] = [
    DataType::Binary,
    DataType::LargeBinary,
    DataType::Utf8,
    DataType::LargeUtf8,
];

impl CoreArrayEncodingStrategy {
    fn can_use_fsst(data_type: &DataType, data_size: u64, version: LanceFileVersion) -> bool {
        version >= LanceFileVersion::V2_1
            && matches!(data_type, DataType::Utf8 | DataType::Binary)
            && data_size > 4 * 1024 * 1024
    }

    fn get_field_compression(field_meta: &HashMap<String, String>) -> Option<CompressionScheme> {
        let compression = field_meta.get(COMPRESSION_META_KEY)?;
        Some(compression.parse::<CompressionScheme>().unwrap())
    }

    fn default_binary_encoder(
        arrays: &[ArrayRef],
        data_type: &DataType,
        field_meta: Option<&HashMap<String, String>>,
        data_size: u64,
        version: LanceFileVersion,
    ) -> Result<Box<dyn ArrayEncoder>> {
        let bin_indices_encoder =
            Self::choose_array_encoder(arrays, &DataType::UInt64, data_size, false, version, None)?;

        let compression = field_meta.and_then(Self::get_field_compression);

        let bin_encoder = Box::new(BinaryEncoder::new(bin_indices_encoder, compression));
        if compression.is_none() && Self::can_use_fsst(data_type, data_size, version) {
            Ok(Box::new(FsstArrayEncoder::new(bin_encoder)))
        } else {
            Ok(bin_encoder)
        }
    }

    fn choose_array_encoder(
        arrays: &[ArrayRef],
        data_type: &DataType,
        data_size: u64,
        use_dict_encoding: bool,
        version: LanceFileVersion,
        field_meta: Option<&HashMap<String, String>>,
    ) -> Result<Box<dyn ArrayEncoder>> {
        match data_type {
            DataType::FixedSizeList(inner, dimension) => {
                Ok(Box::new(BasicEncoder::new(Box::new(FslEncoder::new(
                    Self::choose_array_encoder(
                        arrays,
                        inner.data_type(),
                        data_size,
                        use_dict_encoding,
                        version,
                        None,
                    )?,
                    *dimension as u32,
                )))))
            }
            DataType::Dictionary(key_type, value_type) => {
                let key_encoder =
                    Self::choose_array_encoder(arrays, key_type, data_size, false, version, None)?;
                let value_encoder = Self::choose_array_encoder(
                    arrays, value_type, data_size, false, version, None,
                )?;

                Ok(Box::new(AlreadyDictionaryEncoder::new(
                    key_encoder,
                    value_encoder,
                )))
            }
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => {
                if use_dict_encoding {
                    let dict_indices_encoder = Self::choose_array_encoder(
                        // We need to pass arrays to this method to figure out what kind of compression to
                        // use but we haven't actually calculated the indices yet.  For now, we just assume
                        // worst case and use the full range.  In the future maybe we can pass in statistics
                        // instead of the actual data
                        &[Arc::new(UInt8Array::from_iter_values(0_u8..255_u8))],
                        &DataType::UInt8,
                        data_size,
                        false,
                        version,
                        None,
                    )?;
                    let dict_items_encoder = Self::choose_array_encoder(
                        arrays,
                        &DataType::Utf8,
                        data_size,
                        false,
                        version,
                        None,
                    )?;

                    Ok(Box::new(DictionaryEncoder::new(
                        dict_indices_encoder,
                        dict_items_encoder,
                    )))
                }
                // The parent datatype should be binary or utf8 to use the fixed size encoding
                // The variable 'data_type' is passed through recursion so comparing with it would be incorrect
                else if BINARY_DATATYPES.contains(arrays[0].data_type()) {
                    if let Some(byte_width) = check_fixed_size_encoding(arrays, version) {
                        // use FixedSizeBinaryEncoder
                        let bytes_encoder = Self::choose_array_encoder(
                            arrays,
                            &DataType::UInt8,
                            data_size,
                            false,
                            version,
                            None,
                        )?;

                        Ok(Box::new(BasicEncoder::new(Box::new(
                            FixedSizeBinaryEncoder::new(bytes_encoder, byte_width as usize),
                        ))))
                    } else {
                        Self::default_binary_encoder(
                            arrays, data_type, field_meta, data_size, version,
                        )
                    }
                } else {
                    Self::default_binary_encoder(arrays, data_type, field_meta, data_size, version)
                }
            }
            DataType::Struct(fields) => {
                let num_fields = fields.len();
                let mut inner_encoders = Vec::new();

                for i in 0..num_fields {
                    let inner_datatype = fields[i].data_type();
                    let inner_encoder = Self::choose_array_encoder(
                        arrays,
                        inner_datatype,
                        data_size,
                        use_dict_encoding,
                        version,
                        None,
                    )?;
                    inner_encoders.push(inner_encoder);
                }

                Ok(Box::new(PackedStructEncoder::new(inner_encoders)))
            }
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                if version >= LanceFileVersion::V2_1 && arrays[0].data_type() == data_type {
                    let compressed_bit_width = compute_compressed_bit_width_for_non_neg(arrays);
                    Ok(Box::new(BitpackedForNonNegArrayEncoder::new(
                        compressed_bit_width as usize,
                        data_type.clone(),
                    )))
                } else {
                    Ok(Box::new(BasicEncoder::new(Box::new(
                        ValueEncoder::default(),
                    ))))
                }
            }

            // TODO: for signed integers, I intend to make it a cascaded encoding, a sparse array for the negative values and very wide(bit-width) values,
            // then a bitpacked array for the narrow(bit-width) values, I need `BitpackedForNeg` to be merged first, I am
            // thinking about putting this sparse array in the metadata so bitpacking remain using one page buffer only.
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
                if version >= LanceFileVersion::V2_1 && arrays[0].data_type() == data_type {
                    let compressed_bit_width = compute_compressed_bit_width_for_non_neg(arrays);
                    Ok(Box::new(BitpackedForNonNegArrayEncoder::new(
                        compressed_bit_width as usize,
                        data_type.clone(),
                    )))
                } else {
                    Ok(Box::new(BasicEncoder::new(Box::new(
                        ValueEncoder::default(),
                    ))))
                }
            }
            _ => Ok(Box::new(BasicEncoder::new(Box::new(
                ValueEncoder::default(),
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

fn check_fixed_size_encoding(arrays: &[ArrayRef], version: LanceFileVersion) -> Option<u64> {
    if version < LanceFileVersion::V2_1 || arrays.is_empty() {
        return None;
    }

    // make sure no array has an empty string
    if !arrays.iter().all(|arr| {
        if let Some(arr) = arr.as_string_opt::<i32>() {
            arr.iter().flatten().all(|s| !s.is_empty())
        } else if let Some(arr) = arr.as_binary_opt::<i32>() {
            arr.iter().flatten().all(|s| !s.is_empty())
        } else if let Some(arr) = arr.as_string_opt::<i64>() {
            arr.iter().flatten().all(|s| !s.is_empty())
        } else if let Some(arr) = arr.as_binary_opt::<i64>() {
            arr.iter().flatten().all(|s| !s.is_empty())
        } else {
            panic!("wrong dtype");
        }
    }) {
        return None;
    }

    let lengths = arrays
        .iter()
        .flat_map(|arr| {
            if let Some(arr) = arr.as_string_opt::<i32>() {
                let offsets = arr.offsets().inner();
                offsets
                    .windows(2)
                    .map(|w| (w[1] - w[0]) as u64)
                    .collect::<Vec<_>>()
            } else if let Some(arr) = arr.as_binary_opt::<i32>() {
                let offsets = arr.offsets().inner();
                offsets
                    .windows(2)
                    .map(|w| (w[1] - w[0]) as u64)
                    .collect::<Vec<_>>()
            } else if let Some(arr) = arr.as_string_opt::<i64>() {
                let offsets = arr.offsets().inner();
                offsets
                    .windows(2)
                    .map(|w| (w[1] - w[0]) as u64)
                    .collect::<Vec<_>>()
            } else if let Some(arr) = arr.as_binary_opt::<i64>() {
                let offsets = arr.offsets().inner();
                offsets
                    .windows(2)
                    .map(|w| (w[1] - w[0]) as u64)
                    .collect::<Vec<_>>()
            } else {
                panic!("wrong dtype");
            }
        })
        .collect::<Vec<_>>();

    // find first non-zero value in lengths
    let first_non_zero = lengths.iter().position(|&x| x != 0);
    if let Some(first_non_zero) = first_non_zero {
        // make sure all lengths are equal to first_non_zero length or zero
        if !lengths
            .iter()
            .all(|&x| x == 0 || x == lengths[first_non_zero])
        {
            return None;
        }

        // set the byte width
        Some(lengths[first_non_zero])
    } else {
        None
    }
}

impl ArrayEncodingStrategy for CoreArrayEncodingStrategy {
    fn create_array_encoder(
        &self,
        arrays: &[ArrayRef],
        field: &Field,
    ) -> Result<Box<dyn ArrayEncoder>> {
        let data_size = arrays
            .iter()
            .map(|arr| arr.get_buffer_memory_size() as u64)
            .sum::<u64>();
        let data_type = arrays[0].data_type();

        let use_dict_encoding = data_type == &DataType::Utf8
            && check_dict_encoding(arrays, get_dict_encoding_threshold());

        Self::choose_array_encoder(
            arrays,
            data_type,
            data_size,
            use_dict_encoding,
            self.version,
            Some(&field.metadata),
        )
    }
}
/// Keeps track of the current column index and makes a mapping
/// from field id to column index
#[derive(Default)]
pub struct ColumnIndexSequence {
    current_index: u32,
    mapping: Vec<(u32, u32)>,
}

impl ColumnIndexSequence {
    pub fn next_column_index(&mut self, field_id: u32) -> u32 {
        let idx = self.current_index;
        self.current_index += 1;
        self.mapping.push((field_id, idx));
        idx
    }

    pub fn skip(&mut self) {
        self.current_index += 1;
    }
}

/// Options that control the encoding process
pub struct EncodingOptions {
    /// How much data (in bytes) to cache in-memory before writing a page
    ///
    /// This cache is applied on a per-column basis
    pub cache_bytes_per_column: u64,
    /// The maximum size of a page in bytes, if a single array would create
    /// a page larger than this then it will be split into multiple pages
    pub max_page_bytes: u64,
    /// If false (the default) then arrays will be copied (deeply) before
    /// being cached.  This ensures any data kept alive by the array can
    /// be discarded safely and helps avoid writer accumulation.  However,
    /// there is an associated cost.
    pub keep_original_array: bool,
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
        options: &EncodingOptions,
    ) -> Result<Box<dyn FieldEncoder>>;
}

/// The core field encoding strategy is a set of basic encodings that
/// are generally applicable in most scenarios.
#[derive(Debug)]
pub struct CoreFieldEncodingStrategy {
    pub array_encoding_strategy: Arc<dyn ArrayEncodingStrategy>,
    pub version: LanceFileVersion,
}

// For some reason clippy has a false negative and thinks this can be derived but
// it can't because ArrayEncodingStrategy has no default implementation
#[allow(clippy::derivable_impls)]
impl Default for CoreFieldEncodingStrategy {
    fn default() -> Self {
        Self {
            array_encoding_strategy: Arc::<CoreArrayEncodingStrategy>::default(),
            version: LanceFileVersion::default(),
        }
    }
}

impl CoreFieldEncodingStrategy {
    fn is_primitive_type(data_type: &DataType) -> bool {
        matches!(
            data_type,
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
                | DataType::LargeUtf8,
        )
    }
}

impl FieldEncodingStrategy for CoreFieldEncodingStrategy {
    fn create_field_encoder(
        &self,
        encoding_strategy_root: &dyn FieldEncodingStrategy,
        field: &Field,
        column_index: &mut ColumnIndexSequence,
        options: &EncodingOptions,
    ) -> Result<Box<dyn FieldEncoder>> {
        let data_type = field.data_type();
        if Self::is_primitive_type(&data_type) {
            Ok(Box::new(PrimitiveFieldEncoder::try_new(
                options,
                self.array_encoding_strategy.clone(),
                column_index.next_column_index(field.id as u32),
                field.clone(),
            )?))
        } else {
            match data_type {
                DataType::List(_child) | DataType::LargeList(_child) => {
                    let list_idx = column_index.next_column_index(field.id as u32);
                    let inner_encoding = encoding_strategy_root.create_field_encoder(
                        encoding_strategy_root,
                        &field.children[0],
                        column_index,
                        options,
                    )?;
                    let offsets_encoder =
                        Arc::new(BasicEncoder::new(Box::new(ValueEncoder::default())));
                    Ok(Box::new(ListFieldEncoder::new(
                        inner_encoding,
                        offsets_encoder,
                        options.cache_bytes_per_column,
                        options.keep_original_array,
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
                            options,
                            self.array_encoding_strategy.clone(),
                            column_index.next_column_index(field.id as u32),
                            field.clone(),
                        )?))
                    } else {
                        let header_idx = column_index.next_column_index(field.id as u32);
                        let children_encoders = field
                            .children
                            .iter()
                            .map(|field| {
                                self.create_field_encoder(
                                    encoding_strategy_root,
                                    field,
                                    column_index,
                                    options,
                                )
                            })
                            .collect::<Result<Vec<_>>>()?;
                        Ok(Box::new(StructFieldEncoder::new(
                            children_encoders,
                            header_idx,
                        )))
                    }
                }
                DataType::Dictionary(_, value_type) => {
                    // A dictionary of primitive is, itself, primitive
                    if Self::is_primitive_type(&value_type) {
                        Ok(Box::new(PrimitiveFieldEncoder::try_new(
                            options,
                            self.array_encoding_strategy.clone(),
                            column_index.next_column_index(field.id as u32),
                            field.clone(),
                        )?))
                    } else {
                        // A dictionary of logical is, itself, logical and we don't support that today
                        // It could be possible (e.g. store indices in one column and values in remaining columns)
                        // but would be a significant amount of work
                        //
                        // An easier fallback implementation would be to decode-on-write and encode-on-read
                        Err(Error::NotSupported { source: format!("cannot encode a dictionary column whose value type is a logical type ({})", value_type).into(), location: location!() })
                    }
                }
                _ => todo!("Implement encoding for field {}", field),
            }
        }
    }
}

/// A batch encoder that encodes RecordBatch objects by delegating
/// to field encoders for each top-level field in the batch.
pub struct BatchEncoder {
    pub field_encoders: Vec<Box<dyn FieldEncoder>>,
    pub field_id_to_column_index: Vec<(u32, u32)>,
}

impl BatchEncoder {
    pub fn try_new(
        schema: &Schema,
        strategy: &dyn FieldEncodingStrategy,
        options: &EncodingOptions,
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
                    options,
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
#[derive(Debug)]
pub struct EncodedBatch {
    pub data: Bytes,
    pub page_table: Vec<Arc<ColumnInfo>>,
    pub schema: Arc<Schema>,
    pub top_level_columns: Vec<u32>,
    pub num_rows: u64,
}

fn write_page_to_data_buffer(page: EncodedPage, data_buffer: &mut BytesMut) -> PageInfo {
    let (buffers, encoding) = page.array.into_buffers();
    let mut buffer_offsets_and_sizes = Vec::with_capacity(buffers.len());
    for buffer in buffers {
        let buffer_offset = data_buffer.len() as u64;
        data_buffer.extend_from_slice(&buffer);
        let size = data_buffer.len() as u64 - buffer_offset;
        buffer_offsets_and_sizes.push((buffer_offset, size));
    }
    PageInfo {
        buffer_offsets_and_sizes: Arc::from(buffer_offsets_and_sizes),
        encoding,
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
    options: &EncodingOptions,
) -> Result<EncodedBatch> {
    let mut data_buffer = BytesMut::new();
    let lance_schema = Schema::try_from(batch.schema().as_ref())?;
    let options = EncodingOptions {
        keep_original_array: true,
        ..*options
    };
    let batch_encoder = BatchEncoder::try_new(&lance_schema, encoding_strategy, &options)?;
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
                data_buffer.extend_from_slice(&buffer);
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
        .map(|(_, idx)| *idx)
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

    use crate::version::LanceFileVersion;

    use super::check_dict_encoding;
    use super::check_fixed_size_encoding;

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

    fn is_fixed_size_encoding_applicable(
        arrays: Vec<Vec<Option<&str>>>,
        version: LanceFileVersion,
    ) -> bool {
        let mut final_arrays = Vec::new();
        for arr in arrays {
            let arr = StringArray::from(arr);
            let arr = Arc::new(arr) as ArrayRef;
            final_arrays.push(arr);
        }

        check_fixed_size_encoding(&final_arrays.clone(), version).is_some()
    }

    #[test]
    fn test_fixed_size_binary_encoding_applicable() {
        assert!(!is_fixed_size_encoding_applicable(
            vec![vec![]],
            LanceFileVersion::V2_1
        ));

        assert!(is_fixed_size_encoding_applicable(
            vec![vec![Some("a"), Some("b")]],
            LanceFileVersion::V2_1
        ));

        assert!(!is_fixed_size_encoding_applicable(
            vec![vec![Some("abc"), Some("de")]],
            LanceFileVersion::V2_1
        ));

        assert!(is_fixed_size_encoding_applicable(
            vec![vec![Some("pqr"), None]],
            LanceFileVersion::V2_1
        ));

        assert!(!is_fixed_size_encoding_applicable(
            vec![vec![Some("pqr"), Some("")]],
            LanceFileVersion::V2_1
        ));

        assert!(!is_fixed_size_encoding_applicable(
            vec![vec![Some(""), Some("")]],
            LanceFileVersion::V2_1
        ));
    }

    #[test]
    fn test_fixed_size_binary_encoding_applicable_multiple_arrays() {
        assert!(is_fixed_size_encoding_applicable(
            vec![vec![Some("a"), Some("b")], vec![Some("c"), Some("d")]],
            LanceFileVersion::V2_1
        ));

        assert!(!is_fixed_size_encoding_applicable(
            vec![vec![Some("ab"), Some("bc")], vec![Some("c"), Some("d")]],
            LanceFileVersion::V2_1
        ));

        assert!(!is_fixed_size_encoding_applicable(
            vec![vec![Some("ab"), None], vec![None, Some("d")]],
            LanceFileVersion::V2_1
        ));

        assert!(is_fixed_size_encoding_applicable(
            vec![vec![Some("a"), None], vec![None, Some("d")]],
            LanceFileVersion::V2_1
        ));

        assert!(!is_fixed_size_encoding_applicable(
            vec![vec![Some(""), None], vec![None, Some("")]],
            LanceFileVersion::V2_1
        ));

        assert!(!is_fixed_size_encoding_applicable(
            vec![vec![None, None], vec![None, None]],
            LanceFileVersion::V2_1
        ));
    }
}
