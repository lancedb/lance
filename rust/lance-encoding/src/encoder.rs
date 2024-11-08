// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::{collections::HashMap, env, sync::Arc};

use arrow::array::AsArray;
use arrow::datatypes::UInt64Type;
use arrow_array::PrimitiveArray;
use arrow_array::{Array, ArrayRef, RecordBatch, UInt8Array};
use arrow_schema::DataType;
use bytes::{Bytes, BytesMut};
use futures::future::BoxFuture;
use lance_arrow::DataTypeExt;
use lance_core::datatypes::{
    Field, Schema, BLOB_DESC_FIELD, BLOB_META_KEY, COMPRESSION_LEVEL_META_KEY,
    COMPRESSION_META_KEY, PACKED_STRUCT_LEGACY_META_KEY, PACKED_STRUCT_META_KEY,
};
use lance_core::utils::bit::{is_pwr_two, pad_bytes_to};
use lance_core::{Error, Result};
use snafu::{location, Location};

use crate::buffer::LanceBuffer;
use crate::data::{DataBlock, FixedWidthDataBlock, VariableWidthBlock};
use crate::decoder::PageEncoding;
use crate::encodings::logical::blob::BlobFieldEncoder;
use crate::encodings::logical::primitive::PrimitiveStructuralEncoder;
use crate::encodings::logical::r#struct::StructFieldEncoder;
use crate::encodings::logical::r#struct::StructStructuralEncoder;
use crate::encodings::physical::binary::BinaryMiniBlockEncoder;
use crate::encodings::physical::bitpack_fastlanes::BitpackedForNonNegArrayEncoder;
use crate::encodings::physical::bitpack_fastlanes::{
    compute_compressed_bit_width_for_non_neg, BitpackMiniBlockEncoder,
};
use crate::encodings::physical::block_compress::{CompressionConfig, CompressionScheme};
use crate::encodings::physical::dictionary::AlreadyDictionaryEncoder;
use crate::encodings::physical::fsst::FsstArrayEncoder;
use crate::encodings::physical::packed_struct::PackedStructEncoder;
use crate::format::ProtobufUtils;
use crate::repdef::RepDefBuilder;
use crate::statistics::{GetStat, Stat};
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
    // The encoded page buffers
    pub data: Vec<LanceBuffer>,
    // A description of the encoding used to encode the page
    pub description: PageEncoding,
    /// The number of rows in the encoded page
    pub num_rows: u64,
    /// The top-level row number of the first row in the page
    ///
    /// Generally the number of "top-level" rows and the number of rows are the same.  However,
    /// when there is repetition (list/fixed-size-list) there will be more or less items than rows.
    ///
    /// A top-level row can never be split across a page boundary.
    pub row_number: u64,
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

pub const MAX_MINIBLOCK_BYTES: u64 = 8 * 1024 - 6;
pub const MAX_MINIBLOCK_VALUES: u64 = 4096;

/// Page data that has been compressed into a series of chunks put into
/// a single buffer.
pub struct MiniBlockCompressed {
    /// The buffer of compressed data
    pub data: LanceBuffer,
    /// Describes the size of each chunk
    pub chunks: Vec<MiniBlockChunk>,
    /// The number of values in the entire page
    pub num_values: u64,
}

/// Describes the size of a mini-block chunk of data
///
/// Mini-block chunks are designed to be small (just a few disk sectors)
/// and contain a power-of-two number of values (except for the last chunk)
///
/// To enforce this we limit a chunk to 4Ki values and slightly less than
/// 8KiB of compressed data.  This means that even in the extreme case
/// where we have 4 bytes of rep/def then we will have at most 24KiB of
/// data (values, repetition, and definition) per mini-block.
#[derive(Debug)]
pub struct MiniBlockChunk {
    // The number of bytes that make up the chunk
    //
    // This value must be less than or equal to 8Ki - 6 (8188)
    pub num_bytes: u16,
    // The log (base 2) of the number of values in the chunk.  If this is the final chunk
    // then this should be 0 (the number of values will be calculated by subtracting the
    // size of all other chunks from the total size of the page)
    //
    // For example, 1 would mean there are 2 values in the chunk and 12 would mean there
    // are 4Ki values in the chunk.
    //
    // This must be <= 12 (i.e. <= 4096 values)
    pub log_num_values: u8,
}

impl MiniBlockChunk {
    /// Gets the number of values in this block
    ///
    /// This requires `vals_in_prev_blocks` and `total_num_values` because the
    /// last block in a page is a special case which stores 0 for log_num_values
    /// and, in that case, the number of values is determined by subtracting
    /// `vals_in_prev_blocks` from `total_num_values`
    pub fn num_values(&self, vals_in_prev_blocks: u64, total_num_values: u64) -> u64 {
        if self.log_num_values == 0 {
            total_num_values - vals_in_prev_blocks
        } else {
            1 << self.log_num_values
        }
    }
}

/// Trait for compression algorithms that are suitable for use in the miniblock structural encoding
///
/// These compression algorithms should be capable of encoding the data into small chunks
/// where each chunk (except the last) has 2^N values (N can vary between chunks)
pub trait MiniBlockCompressor: std::fmt::Debug + Send + Sync {
    /// Compress a `page` of data into multiple chunks
    ///
    /// See [`MiniBlockCompressed`] for details on how chunks should be sized.
    ///
    /// This method also returns a description of the encoding applied that will be
    /// used at decode time to read the data.
    fn compress(&self, page: DataBlock) -> Result<(MiniBlockCompressed, pb::ArrayEncoding)>;
}

/// Trait for compression algorithms that are suitable for use in the zipped structural encoding
///
/// Compared to [`VariablePerValueCompressor`], these compressors are capable of compressing the data
/// so that every value has the exact same number of bits per value.  For example, this is useful
/// for encoding vector embeddings where every value has a fixed size but the values themselves are
/// too large to use mini-block.
///
/// The advantage of a fixed-bytes-per-value is that we can do random access in 1 IOP instead of 2
/// and do not need a repetition index.
pub trait FixedPerValueCompressor: std::fmt::Debug + Send + Sync {
    /// Compress the data into a single buffer where each value is encoded with the same number of bits
    ///
    /// Also returns a description of the compression that can be used to decompress when reading the data back
    fn compress(&self, data: DataBlock) -> Result<(FixedWidthDataBlock, pb::ArrayEncoding)>;
}

/// Trait for compression algorithms that are suitable for use in the zipped structural encoding
///
/// This encoding is useful for non-short strings, binary, and variable length lists
/// (i.e. when the average value is >= 128 bytes)
///
/// These compressors can be extremely generic.  They only need to produce one buffer of bytes
/// and another buffer of offsets into the bytes, one offset for each value.  Both of these buffers
/// will be stored.
///
/// Note: It is perfectly legal for a value to have 0 bytes.  However, we still need to store the
/// offset itself.  This means that this compressor, when implemented by something like RLE will not
/// be as efficient (space-wise) as a block version (which could skip the offsets for runs).
///
/// Accessing this data will require 2 IOPS and accessing in a random-access fashion will require
/// a repetition index.
pub trait VariablePerValueCompressor: std::fmt::Debug + Send + Sync {
    /// Compress the data into a single buffer where each value is encoded with a different size
    ///
    /// Also returns a description of the compression that can be used to decompress when reading the data back
    fn compress(&self, data: DataBlock) -> Result<(VariableWidthBlock, pb::ArrayEncoding)>;
}

/// Trait for compression algorithms that compress an entire block of data into one opaque
/// and self-described chunk.
///
/// This is the most general type of compression.  There are no constraints on the method
/// of compression it is assumed that the entire block of data will be present at decompression.
///
/// This is the least appropriate strategy for random access because we must load the entire
/// block to access any single value.  This should only be used for cases where random access is never
/// required (e.g. when encoding metadata buffers like a dictionary or for encoding rep/def
/// mini-block chunks)
pub trait BlockCompressor: std::fmt::Debug + Send + Sync {
    /// Compress the data into a single buffer
    ///
    /// Also returns a description of the compression that can be used to decompress
    /// when reading the data back
    fn compress(&self, data: DataBlock) -> Result<LanceBuffer>;
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

/// A tool to reserve space for buffers that are not in-line with the data
///
/// In most cases, buffers are stored in the page and referred to in the encoding
/// metadata by their index in the page.  This keeps all buffers within a page together.
/// As a result, most encoders should not need to use this structure.
///
/// In some cases (currently only the large binary encoding) there is a need to access
/// buffers that are not in the page (because storing the position / offset of every page
/// in the page metadata would be too expensive).
///
/// To do this you can add a buffer with `add_buffer` and then use the returned position
/// in some way (in the large binary encoding the returned position is stored in the page
/// data as a position / size array).
pub struct OutOfLineBuffers {
    position: u64,
    buffer_alignment: u64,
    buffers: Vec<LanceBuffer>,
}

impl OutOfLineBuffers {
    pub fn new(base_position: u64, buffer_alignment: u64) -> Self {
        Self {
            position: base_position,
            buffer_alignment,
            buffers: Vec::new(),
        }
    }

    pub fn add_buffer(&mut self, buffer: LanceBuffer) -> u64 {
        let position = self.position;
        self.position += buffer.len() as u64;
        self.position += pad_bytes_to(buffer.len(), self.buffer_alignment as usize) as u64;
        self.buffers.push(buffer);
        position
    }

    pub fn take_buffers(self) -> Vec<LanceBuffer> {
        self.buffers
    }

    pub fn reset_position(&mut self, position: u64) {
        self.position = position;
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
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
        external_buffers: &mut OutOfLineBuffers,
        repdef: RepDefBuilder,
        row_number: u64,
    ) -> Result<Vec<EncodeTask>>;
    /// Flush any remaining data from the buffers into encoding tasks
    ///
    /// Each encode task produces a single page.  The order of these pages will be maintained
    /// in the file (we do not worry about order between columns but all pages in the same
    /// column should maintain order)
    ///
    /// This may be called intermittently throughout encoding but will always be called
    /// once at the end of encoding just before calling finish
    fn flush(&mut self, external_buffers: &mut OutOfLineBuffers) -> Result<Vec<EncodeTask>>;
    /// Finish encoding and return column metadata
    ///
    /// This is called only once, after all encode tasks have completed
    ///
    /// This returns a Vec because a single field may have created multiple columns
    fn finish(
        &mut self,
        external_buffers: &mut OutOfLineBuffers,
    ) -> BoxFuture<'_, Result<Vec<EncodedColumn>>>;

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

/// A trait to pick which compression to use for given data
///
/// There are several different kinds of compression.
///
/// - Block compression is the most generic, but most difficult to use efficiently
/// - Fixed-per-value compression results in a fixed number of bits for each value
///     It is used for wide fixed-width types like vector embeddings.
/// - Variable-per-value compression results in two buffers, one buffer of offsets
///     and one buffer of data bytes.  It is used for wide variable-width types
///     like strings, variable-length lists, binary, etc.
/// - Mini-block compression results in a small block of opaque data for chunks
///     of rows.  Each block is somewhere between 0 and 16KiB in size.  This is
///     used for narrow data types (both fixed and variable length) where we can
///     fit many values into an 16KiB block.
pub trait CompressionStrategy: Send + Sync + std::fmt::Debug {
    /// Create a block compressor for the given data
    fn create_block_compressor(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<(Box<dyn BlockCompressor>, pb::ArrayEncoding)>;

    /// Create a fixed-per-value compressor for the given data
    fn create_fixed_per_value(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn FixedPerValueCompressor>>;

    /// Create a variable-per-value compressor for the given data
    fn create_variable_per_value(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn VariablePerValueCompressor>>;

    /// Create a mini-block compressor for the given data
    fn create_miniblock_compressor(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn MiniBlockCompressor>>;
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

    fn get_field_compression(field_meta: &HashMap<String, String>) -> Option<CompressionConfig> {
        let compression = field_meta.get(COMPRESSION_META_KEY)?;
        let compression_scheme = compression.parse::<CompressionScheme>();
        match compression_scheme {
            Ok(compression_scheme) => Some(CompressionConfig::new(
                compression_scheme,
                field_meta
                    .get(COMPRESSION_LEVEL_META_KEY)
                    .and_then(|level| level.parse().ok()),
            )),
            Err(_) => None,
        }
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

impl CompressionStrategy for CoreArrayEncodingStrategy {
    fn create_miniblock_compressor(
        &self,
        _field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn MiniBlockCompressor>> {
        if let DataBlock::FixedWidth(ref fixed_width_data) = data {
            let bit_widths = data
                .get_stat(Stat::BitWidth)
                .expect("FixedWidthDataBlock should have valid bit width statistics");
            // Temporary hack to work around https://github.com/lancedb/lance/issues/3102
            // Ideally we should still be able to bit-pack here (either to 0 or 1 bit per value)
            let has_all_zeros = bit_widths
                .as_primitive::<UInt64Type>()
                .values()
                .iter()
                .any(|v| *v == 0);
            if !has_all_zeros
                && (fixed_width_data.bits_per_value == 8
                    || fixed_width_data.bits_per_value == 16
                    || fixed_width_data.bits_per_value == 32
                    || fixed_width_data.bits_per_value == 64)
            {
                return Ok(Box::new(BitpackMiniBlockEncoder::default()));
            }
        }
        if let DataBlock::VariableWidth(ref variable_width_data) = data {
            if variable_width_data.bits_per_offset == 32 {
                return Ok(Box::new(BinaryMiniBlockEncoder::default()));
            }
        }
        Ok(Box::new(ValueEncoder::default()))
    }

    fn create_fixed_per_value(
        &self,
        field: &Field,
        _data: &DataBlock,
    ) -> Result<Box<dyn FixedPerValueCompressor>> {
        // Right now we only need block compressors for rep/def which is u16.  Will need to expand
        // this if we need block compression of other types.
        assert!(field.data_type().byte_width() > 0);
        Ok(Box::new(ValueEncoder::default()))
    }

    fn create_variable_per_value(
        &self,
        _field: &Field,
        _data: &DataBlock,
    ) -> Result<Box<dyn VariablePerValueCompressor>> {
        todo!()
    }

    fn create_block_compressor(
        &self,
        _field: &Field,
        data: &DataBlock,
    ) -> Result<(Box<dyn BlockCompressor>, pb::ArrayEncoding)> {
        match data {
            DataBlock::FixedWidth(fixed_width) => {
                let encoder = Box::new(ValueEncoder::default());
                let encoding = ProtobufUtils::flat_encoding(fixed_width.bits_per_value, 0, None);
                Ok((encoder, encoding))
            }
            _ => unreachable!(),
        }
    }
}
/// Keeps track of the current column index and makes a mapping
/// from field id to column index
#[derive(Debug, Default)]
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
    /// The alignment that the writer is applying to buffers
    ///
    /// The encoder needs to know this so it figures the position of out-of-line
    /// buffers correctly
    pub buffer_alignment: u64,
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

pub fn default_encoding_strategy(version: LanceFileVersion) -> Box<dyn FieldEncodingStrategy> {
    match version.resolve() {
        LanceFileVersion::Legacy => panic!(),
        LanceFileVersion::V2_0 => Box::new(CoreFieldEncodingStrategy::default()),
        _ => Box::new(StructuralEncodingStrategy::default()),
    }
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
            let column_index = column_index.next_column_index(field.id as u32);
            if field.metadata.contains_key(BLOB_META_KEY) {
                let mut packed_meta = HashMap::new();
                packed_meta.insert(PACKED_STRUCT_META_KEY.to_string(), "true".to_string());
                let desc_field =
                    Field::try_from(BLOB_DESC_FIELD.clone().with_metadata(packed_meta)).unwrap();
                let desc_encoder = Box::new(PrimitiveFieldEncoder::try_new(
                    options,
                    self.array_encoding_strategy.clone(),
                    column_index,
                    desc_field,
                )?);
                Ok(Box::new(BlobFieldEncoder::new(desc_encoder)))
            } else {
                Ok(Box::new(PrimitiveFieldEncoder::try_new(
                    options,
                    self.array_encoding_strategy.clone(),
                    column_index,
                    field.clone(),
                )?))
            }
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
                        .get(PACKED_STRUCT_LEGACY_META_KEY)
                        .map(|v| v == "true")
                        .unwrap_or(field_metadata.contains_key(PACKED_STRUCT_META_KEY))
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

/// An encoding strategy used for 2.1+ files
#[derive(Debug)]
pub struct StructuralEncodingStrategy {
    pub compression_strategy: Arc<dyn CompressionStrategy>,
    pub version: LanceFileVersion,
}

// For some reason, clippy thinks we can add Default to the above derive but
// rustc doesn't agree (no default for Arc<dyn Trait>)
#[allow(clippy::derivable_impls)]
impl Default for StructuralEncodingStrategy {
    fn default() -> Self {
        Self {
            compression_strategy: Arc::<CoreArrayEncodingStrategy>::default(),
            version: LanceFileVersion::default(),
        }
    }
}

impl StructuralEncodingStrategy {
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

impl FieldEncodingStrategy for StructuralEncodingStrategy {
    fn create_field_encoder(
        &self,
        _encoding_strategy_root: &dyn FieldEncodingStrategy,
        field: &Field,
        column_index: &mut ColumnIndexSequence,
        options: &EncodingOptions,
    ) -> Result<Box<dyn FieldEncoder>> {
        let data_type = field.data_type();
        if Self::is_primitive_type(&data_type) {
            Ok(Box::new(PrimitiveStructuralEncoder::try_new(
                options,
                self.compression_strategy.clone(),
                column_index.next_column_index(field.id as u32),
                field.clone(),
            )?))
        } else {
            match data_type {
                DataType::List(_child) | DataType::LargeList(_child) => {
                    todo!()
                }
                DataType::Struct(_) => {
                    let field_metadata = &field.metadata;
                    if field_metadata
                        .get("packed")
                        .map(|v| v == "true")
                        .unwrap_or(false)
                    {
                        Ok(Box::new(PrimitiveStructuralEncoder::try_new(
                            options,
                            self.compression_strategy.clone(),
                            column_index.next_column_index(field.id as u32),
                            field.clone(),
                        )?))
                    } else {
                        let children_encoders = field
                            .children
                            .iter()
                            .map(|field| {
                                self.create_field_encoder(
                                    _encoding_strategy_root,
                                    field,
                                    column_index,
                                    options,
                                )
                            })
                            .collect::<Result<Vec<_>>>()?;
                        Ok(Box::new(StructStructuralEncoder::new(children_encoders)))
                    }
                }
                DataType::Dictionary(_, value_type) => {
                    // A dictionary of primitive is, itself, primitive
                    if Self::is_primitive_type(&value_type) {
                        Ok(Box::new(PrimitiveStructuralEncoder::try_new(
                            options,
                            self.compression_strategy.clone(),
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
    let buffers = page.data;
    let mut buffer_offsets_and_sizes = Vec::with_capacity(buffers.len());
    for buffer in buffers {
        let buffer_offset = data_buffer.len() as u64;
        data_buffer.extend_from_slice(&buffer);
        let size = data_buffer.len() as u64 - buffer_offset;
        buffer_offsets_and_sizes.push((buffer_offset, size));
    }

    PageInfo {
        buffer_offsets_and_sizes: Arc::from(buffer_offsets_and_sizes),
        encoding: page.description,
        num_rows: page.num_rows,
        priority: page.row_number,
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
    if !is_pwr_two(options.buffer_alignment) || options.buffer_alignment < 8 {
        return Err(Error::InvalidInput {
            source: "buffer_alignment must be a power of two and at least 8".into(),
            location: location!(),
        });
    }

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
        let mut external_buffers =
            OutOfLineBuffers::new(data_buffer.len() as u64, options.buffer_alignment);
        let repdef = RepDefBuilder::default();
        let encoder = encoder.as_mut();
        let mut tasks = encoder.maybe_encode(arr.clone(), &mut external_buffers, repdef, 0)?;
        tasks.extend(encoder.flush(&mut external_buffers)?);
        for buffer in external_buffers.take_buffers() {
            data_buffer.extend_from_slice(&buffer);
        }
        let mut pages = HashMap::<u32, Vec<PageInfo>>::new();
        for task in tasks {
            let encoded_page = task.await?;
            // Write external buffers first
            pages
                .entry(encoded_page.column_idx)
                .or_default()
                .push(write_page_to_data_buffer(encoded_page, &mut data_buffer));
        }
        let mut external_buffers =
            OutOfLineBuffers::new(data_buffer.len() as u64, options.buffer_alignment);
        let encoded_columns = encoder.finish(&mut external_buffers).await?;
        for buffer in external_buffers.take_buffers() {
            data_buffer.extend_from_slice(&buffer);
        }
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
    use crate::version::LanceFileVersion;
    use arrow_array::{ArrayRef, StringArray};
    use arrow_schema::Field;
    use lance_core::datatypes::{COMPRESSION_LEVEL_META_KEY, COMPRESSION_META_KEY};
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::check_fixed_size_encoding;
    use super::{check_dict_encoding, ArrayEncodingStrategy, CoreArrayEncodingStrategy};

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

    fn verify_array_encoder(
        array: ArrayRef,
        field_meta: Option<HashMap<String, String>>,
        version: LanceFileVersion,
        expected_encoder: &str,
    ) {
        let encoding_strategy = CoreArrayEncodingStrategy { version };
        let mut field = Field::new("test_field", array.data_type().clone(), true);
        if let Some(field_meta) = field_meta {
            field.set_metadata(field_meta);
        }
        let lance_field = lance_core::datatypes::Field::try_from(field).unwrap();
        let encoder_result = encoding_strategy.create_array_encoder(&[array], &lance_field);
        assert!(encoder_result.is_ok());
        let encoder = encoder_result.unwrap();
        assert_eq!(format!("{:?}", encoder).as_str(), expected_encoder);
    }

    #[test]
    fn test_choose_encoder_for_zstd_compressed_string_field() {
        verify_array_encoder(Arc::new(StringArray::from(vec!["a", "bb", "ccc"])),
                             Some(HashMap::from([(COMPRESSION_META_KEY.to_string(), "zstd".to_string())])),
                             LanceFileVersion::V2_1,
                             "BinaryEncoder { indices_encoder: BasicEncoder { values_encoder: ValueEncoder }, compression_config: Some(CompressionConfig { scheme: Zstd, level: None }), buffer_compressor: Some(ZstdBufferCompressor { compression_level: 0 }) }");
    }

    #[test]
    fn test_choose_encoder_for_zstd_compression_level() {
        verify_array_encoder(Arc::new(StringArray::from(vec!["a", "bb", "ccc"])),
                             Some(HashMap::from([
                                 (COMPRESSION_META_KEY.to_string(), "zstd".to_string()),
                                 (COMPRESSION_LEVEL_META_KEY.to_string(), "22".to_string())
                             ])),
                             LanceFileVersion::V2_1,
                             "BinaryEncoder { indices_encoder: BasicEncoder { values_encoder: ValueEncoder }, compression_config: Some(CompressionConfig { scheme: Zstd, level: Some(22) }), buffer_compressor: Some(ZstdBufferCompressor { compression_level: 22 }) }");
    }
}
