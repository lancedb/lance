// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Data layouts to represent encoded data in a sub-Arrow format
//!
//! These [`DataBlock`] structures represent physical layouts.  They fill a gap somewhere
//! between [`arrow_data::data::ArrayData`] (which, as a collection of buffers, is too
//! generic because it doesn't give us enough information about what those buffers represent)
//! and [`arrow_array::array::Array`] (which is too specific, because it cares about the
//! logical data type).
//!
//! In addition, the layouts represented here are slightly stricter than Arrow's layout rules.
//! For example, offset buffers MUST start with 0.  These additional restrictions impose a
//! slight penalty on encode (to normalize arrow data) but make the development of encoders
//! and decoders easier (since they can rely on a normalized representation)

use std::{ops::Range, sync::Arc};

use arrow::array::{ArrayData, ArrayDataBuilder, AsArray};
use arrow_array::{new_null_array, Array, ArrayRef, UInt64Array};
use arrow_buffer::{ArrowNativeType, BooleanBuffer, BooleanBufferBuilder, NullBuffer};
use arrow_schema::DataType;
use lance_arrow::DataTypeExt;
use snafu::{location, Location};

use lance_core::{Error, Result};

use crate::buffer::LanceBuffer;

/// A data block with no buffers where everything is null
///
/// Note: this data block should not be used for future work.  It will be deprecated
/// in the 2.1 version of the format where nullability will be handled by the structural
/// encoders.
#[derive(Debug)]
pub struct AllNullDataBlock {
    /// The number of values represented by this block
    pub num_values: u64,
}

impl AllNullDataBlock {
    fn into_arrow(self, data_type: DataType, _validate: bool) -> Result<ArrayData> {
        Ok(ArrayData::new_null(&data_type, self.num_values as usize))
    }

    fn into_buffers(self) -> Vec<LanceBuffer> {
        vec![]
    }

    fn borrow_and_clone(&mut self) -> Self {
        Self {
            num_values: self.num_values,
        }
    }

    fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            num_values: self.num_values,
        })
    }
}

/// Wraps a data block and adds nullability information to it
///
/// Note: this data block should not be used for future work.  It will be deprecated
/// in the 2.1 version of the format where nullability will be handled by the structural
/// encoders.
#[derive(Debug)]
pub struct NullableDataBlock {
    /// The underlying data
    pub data: Box<DataBlock>,
    /// A bitmap of validity for each value
    pub nulls: LanceBuffer,
}

impl NullableDataBlock {
    fn into_arrow(self, data_type: DataType, validate: bool) -> Result<ArrayData> {
        let nulls = self.nulls.into_buffer();
        let data = self.data.into_arrow(data_type, validate)?.into_builder();
        let data = data.null_bit_buffer(Some(nulls));
        if validate {
            Ok(data.build()?)
        } else {
            Ok(unsafe { data.build_unchecked() })
        }
    }

    fn into_buffers(self) -> Vec<LanceBuffer> {
        let mut buffers = vec![self.nulls];
        buffers.extend(self.data.into_buffers());
        buffers
    }

    fn borrow_and_clone(&mut self) -> Self {
        Self {
            data: Box::new(self.data.borrow_and_clone()),
            nulls: self.nulls.borrow_and_clone(),
        }
    }

    fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            data: Box::new(self.data.try_clone()?),
            nulls: self.nulls.try_clone()?,
        })
    }
}

/// A data block for a single buffer of data where each element has a fixed number of bits
#[derive(Debug)]
pub struct FixedWidthDataBlock {
    /// The data buffer
    pub data: LanceBuffer,
    /// The number of bits per value
    pub bits_per_value: u64,
    /// The number of values represented by this block
    pub num_values: u64,
}

impl FixedWidthDataBlock {
    fn do_into_arrow(
        self,
        data_type: DataType,
        num_values: u64,
        validate: bool,
    ) -> Result<ArrayData> {
        let builder = match &data_type {
            DataType::FixedSizeList(child_field, dim) => {
                let child_len = num_values * *dim as u64;
                let child_data =
                    self.do_into_arrow(child_field.data_type().clone(), child_len, validate)?;
                ArrayDataBuilder::new(data_type)
                    .add_child_data(child_data)
                    .len(num_values as usize)
                    .null_count(0)
            }
            _ => {
                let data_buffer = self.data.into_buffer();
                ArrayDataBuilder::new(data_type)
                    .add_buffer(data_buffer)
                    .len(num_values as usize)
                    .null_count(0)
            }
        };
        if validate {
            Ok(builder.build()?)
        } else {
            Ok(unsafe { builder.build_unchecked() })
        }
    }

    fn into_arrow(self, data_type: DataType, validate: bool) -> Result<ArrayData> {
        let root_num_values = self.num_values;
        self.do_into_arrow(data_type, root_num_values, validate)
    }

    fn into_buffers(self) -> Vec<LanceBuffer> {
        vec![self.data]
    }

    fn borrow_and_clone(&mut self) -> Self {
        Self {
            data: self.data.borrow_and_clone(),
            bits_per_value: self.bits_per_value,
            num_values: self.num_values,
        }
    }

    fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.try_clone()?,
            bits_per_value: self.bits_per_value,
            num_values: self.num_values,
        })
    }
}

/// A data block for variable-width data (e.g. strings, packed rows, etc.)
#[derive(Debug)]
pub struct VariableWidthBlock {
    /// The data buffer
    pub data: LanceBuffer,
    /// The offsets buffer (contains num_values + 1 offsets)
    ///
    /// Offsets MUST start at 0
    pub offsets: LanceBuffer,
    /// The number of bits per offset
    pub bits_per_offset: u8,
    /// The number of values represented by this block
    pub num_values: u64,
}

impl VariableWidthBlock {
    fn into_arrow(self, data_type: DataType, validate: bool) -> Result<ArrayData> {
        let data_buffer = self.data.into_buffer();
        let offsets_buffer = self.offsets.into_buffer();
        let builder = ArrayDataBuilder::new(data_type)
            .add_buffer(offsets_buffer)
            .add_buffer(data_buffer)
            .len(self.num_values as usize)
            .null_count(0);
        if validate {
            Ok(builder.build()?)
        } else {
            Ok(unsafe { builder.build_unchecked() })
        }
    }

    fn into_buffers(self) -> Vec<LanceBuffer> {
        vec![self.offsets, self.data]
    }

    fn borrow_and_clone(&mut self) -> Self {
        Self {
            data: self.data.borrow_and_clone(),
            offsets: self.offsets.borrow_and_clone(),
            bits_per_offset: self.bits_per_offset,
            num_values: self.num_values,
        }
    }

    fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.try_clone()?,
            offsets: self.offsets.try_clone()?,
            bits_per_offset: self.bits_per_offset,
            num_values: self.num_values,
        })
    }
}

/// A data block representing a struct
#[derive(Debug)]
pub struct StructDataBlock {
    /// The child arrays
    pub children: Vec<DataBlock>,
}

impl StructDataBlock {
    fn into_arrow(self, data_type: DataType, validate: bool) -> Result<ArrayData> {
        if let DataType::Struct(fields) = &data_type {
            let mut builder = ArrayDataBuilder::new(DataType::Struct(fields.clone()));
            let mut num_rows = 0;
            for (field, child) in fields.iter().zip(self.children) {
                let child_data = child.into_arrow(field.data_type().clone(), validate)?;
                num_rows = child_data.len();
                builder = builder.add_child_data(child_data);
            }
            let builder = builder.null_count(0).len(num_rows);
            if validate {
                Ok(builder.build()?)
            } else {
                Ok(unsafe { builder.build_unchecked() })
            }
        } else {
            Err(Error::Internal {
                message: format!("Expected Struct, got {:?}", data_type),
                location: location!(),
            })
        }
    }

    fn into_buffers(self) -> Vec<LanceBuffer> {
        self.children
            .into_iter()
            .flat_map(|c| c.into_buffers())
            .collect()
    }

    fn borrow_and_clone(&mut self) -> Self {
        Self {
            children: self
                .children
                .iter_mut()
                .map(|c| c.borrow_and_clone())
                .collect(),
        }
    }

    fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            children: self
                .children
                .iter()
                .map(|c| c.try_clone())
                .collect::<Result<_>>()?,
        })
    }
}

/// A data block for dictionary encoded data
///
/// Note that, unlike Arrow, there is only one canonical place to store nulls, and that is
/// in the dictionary itself.  This simplifies the representation of dictionary encoded data
/// and makes it more efficient to encode and decode.
#[derive(Debug)]
pub struct DictionaryDataBlock {
    /// The indices buffer
    pub indices: FixedWidthDataBlock,
    /// The dictionary itself
    pub dictionary: Box<DataBlock>,
}

impl DictionaryDataBlock {
    fn into_arrow(self, data_type: DataType, validate: bool) -> Result<ArrayData> {
        let (key_type, value_type) = if let DataType::Dictionary(key_type, value_type) = &data_type
        {
            (key_type.as_ref().clone(), value_type.as_ref().clone())
        } else {
            return Err(Error::Internal {
                message: format!("Expected Dictionary, got {:?}", data_type),
                location: location!(),
            });
        };

        let indices = self.indices.into_arrow(key_type, validate)?;
        let dictionary = self.dictionary.into_arrow(value_type, validate)?;

        let builder = indices
            .into_builder()
            .add_child_data(dictionary)
            .data_type(data_type);

        if validate {
            Ok(builder.build()?)
        } else {
            Ok(unsafe { builder.build_unchecked() })
        }
    }

    fn into_buffers(self) -> Vec<LanceBuffer> {
        let mut buffers = self.indices.into_buffers();
        buffers.extend(self.dictionary.into_buffers());
        buffers
    }

    fn borrow_and_clone(&mut self) -> Self {
        Self {
            indices: self.indices.borrow_and_clone(),
            dictionary: Box::new(self.dictionary.borrow_and_clone()),
        }
    }

    fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            indices: self.indices.try_clone()?,
            dictionary: Box::new(self.dictionary.try_clone()?),
        })
    }
}

/// A DataBlock is a collection of buffers that represents an "array" of data in very generic terms
///
/// The output of each decoder is a DataBlock.  Decoders can be chained together to transform
/// one DataBlock into a different kind of DataBlock.
///
/// The DataBlock is somewhere in between Arrow's ArrayData and Array and represents a physical
/// layout of the data.
///
/// A DataBlock can be converted into an Arrow ArrayData (and then Array) for a given array type.
/// For example, a FixedWidthDataBlock can be converted into any primitive type or a fixed size
/// list of a primitive type.  This is a zero-copy operation.
///
/// In addition, a DataBlock can be created from an Arrow array or arrays.  This is not a zero-copy
/// operation as some normalization may be required.
#[derive(Debug)]
pub enum DataBlock {
    AllNull(AllNullDataBlock),
    Nullable(NullableDataBlock),
    FixedWidth(FixedWidthDataBlock),
    VariableWidth(VariableWidthBlock),
    Struct(StructDataBlock),
    Dictionary(DictionaryDataBlock),
}

impl DataBlock {
    /// Convert self into an Arrow ArrayData
    pub fn into_arrow(self, data_type: DataType, validate: bool) -> Result<ArrayData> {
        match self {
            Self::AllNull(inner) => inner.into_arrow(data_type, validate),
            Self::Nullable(inner) => inner.into_arrow(data_type, validate),
            Self::FixedWidth(inner) => inner.into_arrow(data_type, validate),
            Self::VariableWidth(inner) => inner.into_arrow(data_type, validate),
            Self::Struct(inner) => inner.into_arrow(data_type, validate),
            Self::Dictionary(inner) => inner.into_arrow(data_type, validate),
        }
    }

    /// Convert the data block into a collection of buffers for serialization
    ///
    /// The order matters and will be used to reconstruct the data block at read time.
    pub fn into_buffers(self) -> Vec<LanceBuffer> {
        match self {
            Self::AllNull(inner) => inner.into_buffers(),
            Self::Nullable(inner) => inner.into_buffers(),
            Self::FixedWidth(inner) => inner.into_buffers(),
            Self::VariableWidth(inner) => inner.into_buffers(),
            Self::Struct(inner) => inner.into_buffers(),
            Self::Dictionary(inner) => inner.into_buffers(),
        }
    }

    /// Converts the data buffers into borrowed mode and clones the block
    ///
    /// This is a zero-copy operation but requires a mutable reference to self and, afterwards,
    /// all buffers will be in Borrowed mode.
    pub fn borrow_and_clone(&mut self) -> Self {
        match self {
            Self::AllNull(inner) => Self::AllNull(inner.borrow_and_clone()),
            Self::Nullable(inner) => Self::Nullable(inner.borrow_and_clone()),
            Self::FixedWidth(inner) => Self::FixedWidth(inner.borrow_and_clone()),
            Self::VariableWidth(inner) => Self::VariableWidth(inner.borrow_and_clone()),
            Self::Struct(inner) => Self::Struct(inner.borrow_and_clone()),
            Self::Dictionary(inner) => Self::Dictionary(inner.borrow_and_clone()),
        }
    }

    /// Try and clone the block
    ///
    /// This will fail if any buffers are in owned mode.  You can call borrow_and_clone() to
    /// ensure that all buffers are in borrowed mode before calling this method.
    pub fn try_clone(&self) -> Result<Self> {
        match self {
            Self::AllNull(inner) => Ok(Self::AllNull(inner.try_clone()?)),
            Self::Nullable(inner) => Ok(Self::Nullable(inner.try_clone()?)),
            Self::FixedWidth(inner) => Ok(Self::FixedWidth(inner.try_clone()?)),
            Self::VariableWidth(inner) => Ok(Self::VariableWidth(inner.try_clone()?)),
            Self::Struct(inner) => Ok(Self::Struct(inner.try_clone()?)),
            Self::Dictionary(inner) => Ok(Self::Dictionary(inner.try_clone()?)),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::AllNull(_) => "AllNull",
            Self::Nullable(_) => "Nullable",
            Self::FixedWidth(_) => "FixedWidth",
            Self::VariableWidth(_) => "VariableWidth",
            Self::Struct(_) => "Struct",
            Self::Dictionary(_) => "Dictionary",
        }
    }

    pub fn num_values(&self) -> u64 {
        match self {
            Self::AllNull(inner) => inner.num_values,
            Self::Nullable(inner) => inner.data.num_values(),
            Self::FixedWidth(inner) => inner.num_values,
            Self::VariableWidth(inner) => inner.num_values,
            Self::Struct(inner) => inner.children[0].num_values(),
            Self::Dictionary(inner) => inner.indices.num_values,
        }
    }
}

macro_rules! as_type {
    ($fn_name:ident, $inner:tt, $inner_type:ident) => {
        pub fn $fn_name(self) -> Result<$inner_type> {
            match self {
                Self::$inner(inner) => Ok(inner),
                _ => Err(Error::Internal {
                    message: format!("Expected {}, got {}", stringify!($inner), self.name()),
                    location: location!(),
                }),
            }
        }
    };
}

// Cast implementations
impl DataBlock {
    as_type!(as_all_null, AllNull, AllNullDataBlock);
    as_type!(as_nullable, Nullable, NullableDataBlock);
    as_type!(as_fixed_width, FixedWidth, FixedWidthDataBlock);
    as_type!(as_variable_width, VariableWidth, VariableWidthBlock);
    as_type!(as_struct, Struct, StructDataBlock);
    as_type!(as_dictionary, Dictionary, DictionaryDataBlock);
}

// Methods to convert from Arrow -> DataBlock

fn get_byte_range<T: ArrowNativeType>(offsets: &mut LanceBuffer) -> Range<usize> {
    let offsets = offsets.borrow_to_typed_slice::<T>();
    if offsets.as_ref().is_empty() {
        0..0
    } else {
        offsets.as_ref().first().unwrap().as_usize()..offsets.as_ref().last().unwrap().as_usize()
    }
}

// Given multiple offsets arrays [0, 5, 10], [0, 3, 7], etc. stitch
// them together to get [0, 5, 10, 13, 20, ...]
//
// Also returns the data range referenced by each offset array (may
// not be 0..len if there is slicing involved)
fn stitch_offsets<T: ArrowNativeType + std::ops::Add<Output = T> + std::ops::Sub<Output = T>>(
    offsets: Vec<LanceBuffer>,
) -> (LanceBuffer, Vec<Range<usize>>) {
    if offsets.is_empty() {
        return (LanceBuffer::empty(), Vec::default());
    }
    let len = offsets.iter().map(|b| b.len()).sum::<usize>();
    // Note: we are making a copy here, even if there is only one input, because we want to
    // normalize that input if it doesn't start with zero.  This could be micro-optimized out
    // if needed.
    let mut dest = Vec::with_capacity(len);
    let mut byte_ranges = Vec::with_capacity(offsets.len());

    // We insert one leading 0 before processing any of the inputs
    dest.push(T::from_usize(0).unwrap());

    for mut o in offsets.into_iter() {
        if !o.is_empty() {
            let last_offset = *dest.last().unwrap();
            let o = o.borrow_to_typed_slice::<T>();
            let start = *o.as_ref().first().unwrap();
            // First, we skip the first offset
            // Then, we subtract that first offset from each remaining offset
            //
            // This gives us a 0-based offset array (minus the leading 0)
            //
            // Then we add the last offset from the previous array to each offset
            // which shifts our offset array to the correct position
            //
            // For example, let's assume the last offset from the previous array
            // was 10 and we are given [13, 17, 22].  This means we have two values with
            // length 4 (17 - 13) and 5 (22 - 17).  The output from this step will be
            // [14, 19].  Combined with our last offset of 10, this gives us [10, 14, 19]
            // which is our same two values of length 4 and 5.
            dest.extend(o.as_ref()[1..].iter().map(|&x| x + last_offset - start));
        }
        byte_ranges.push(get_byte_range::<T>(&mut o));
    }
    (LanceBuffer::reinterpret_vec(dest), byte_ranges)
}

fn arrow_binary_to_data_block(
    arrays: &[ArrayRef],
    num_values: u64,
    bits_per_offset: u8,
) -> DataBlock {
    let datas = arrays.iter().map(|arr| arr.to_data()).collect::<Vec<_>>();
    let bytes_per_offset = bits_per_offset as usize / 8;
    let offsets = datas
        .iter()
        .map(|d| {
            LanceBuffer::Borrowed(
                d.buffers()[0]
                    .slice_with_length(d.offset(), (d.len() + 1) * bytes_per_offset)
                    .clone(),
            )
        })
        .collect::<Vec<_>>();
    let (offsets, data_ranges) = if bits_per_offset == 32 {
        stitch_offsets::<i32>(offsets)
    } else {
        stitch_offsets::<i64>(offsets)
    };
    let data = datas
        .iter()
        .zip(data_ranges)
        .map(|(d, byte_range)| {
            LanceBuffer::Borrowed(
                d.buffers()[1]
                    .slice_with_length(byte_range.start, byte_range.end - byte_range.start),
            )
        })
        .collect::<Vec<_>>();
    let data = LanceBuffer::concat_into_one(data);
    DataBlock::VariableWidth(VariableWidthBlock {
        data,
        offsets,
        bits_per_offset,
        num_values,
    })
}

fn encode_flat_data(arrays: &[ArrayRef], num_values: u64) -> LanceBuffer {
    let bytes_per_value = arrays[0].data_type().byte_width();
    let mut buffer = Vec::with_capacity(num_values as usize * bytes_per_value);
    for arr in arrays {
        let data = arr.to_data();
        buffer.extend_from_slice(data.buffers()[0].as_slice());
    }
    LanceBuffer::Owned(buffer)
}

fn do_encode_bitmap_data(bitmaps: &[BooleanBuffer], num_values: u64) -> LanceBuffer {
    let mut builder = BooleanBufferBuilder::new(num_values as usize);

    for buf in bitmaps {
        builder.append_buffer(buf);
    }

    let buffer = builder.finish().into_inner();
    LanceBuffer::Borrowed(buffer)
}

fn encode_bitmap_data(arrays: &[ArrayRef], num_values: u64) -> LanceBuffer {
    let bitmaps = arrays
        .iter()
        .map(|arr| arr.as_boolean().values().clone())
        .collect::<Vec<_>>();
    do_encode_bitmap_data(&bitmaps, num_values)
}

// Concatenate dictionary arrays.  This is a bit tricky because we might overflow the
// index type.  If we do, we need to upscale the indices to a larger type.
fn concat_dict_arrays(arrays: &[ArrayRef]) -> ArrayRef {
    let value_type = arrays[0].as_any_dictionary().values().data_type();
    let array_refs = arrays.iter().map(|arr| arr.as_ref()).collect::<Vec<_>>();
    match arrow_select::concat::concat(&array_refs) {
        Ok(array) => array,
        Err(arrow_schema::ArrowError::DictionaryKeyOverflowError { .. }) => {
            // Slow, but hopefully a corner case.  Optimize later
            let upscaled = array_refs
                .iter()
                .map(|arr| {
                    match arrow_cast::cast(
                        *arr,
                        &DataType::Dictionary(
                            Box::new(DataType::UInt32),
                            Box::new(value_type.clone()),
                        ),
                    ) {
                        Ok(arr) => arr,
                        Err(arrow_schema::ArrowError::DictionaryKeyOverflowError { .. }) => {
                            // Technically I think this means the input type was u64 already
                            unimplemented!("Dictionary arrays with more than 2^32 unique values")
                        }
                        err => err.unwrap(),
                    }
                })
                .collect::<Vec<_>>();
            let array_refs = upscaled.iter().map(|arr| arr.as_ref()).collect::<Vec<_>>();
            // Can still fail if concat pushes over u32 boundary
            match arrow_select::concat::concat(&array_refs) {
                Ok(array) => array,
                Err(arrow_schema::ArrowError::DictionaryKeyOverflowError { .. }) => {
                    unimplemented!("Dictionary arrays with more than 2^32 unique values")
                }
                err => err.unwrap(),
            }
        }
        // Shouldn't be any other possible errors in concat
        err => err.unwrap(),
    }
}

fn max_index_val(index_type: &DataType) -> u64 {
    match index_type {
        DataType::Int8 => i8::MAX as u64,
        DataType::Int16 => i16::MAX as u64,
        DataType::Int32 => i32::MAX as u64,
        DataType::Int64 => i64::MAX as u64,
        DataType::UInt8 => u8::MAX as u64,
        DataType::UInt16 => u16::MAX as u64,
        DataType::UInt32 => u32::MAX as u64,
        DataType::UInt64 => u64::MAX,
        _ => panic!("Invalid dictionary index type"),
    }
}

// If we get multiple dictionary arrays and they don't all have the same dictionary
// then we need to normalize the indices.  Otherwise we might have something like:
//
// First chunk ["hello", "foo"], [0, 0, 1, 1, 1]
// Second chunk ["bar", "world"], [0, 1, 0, 1, 1]
//
// If we simply encode as ["hello", "foo", "bar", "world"], [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]
// then we will get the wrong answer because the dictionaries were not merged and the indices
// were not remapped.
//
// A simple way to do this today is to just concatenate all the arrays.  This is because
// arrow's dictionary concatenation function already has the logic to merge dictionaries.
//
// TODO: We could be more efficient here by checking if the dictionaries are the same
//       Also, if they aren't, we can possibly do something cheaper than concatenating
//
// In addition, we want to normalize the representation of nulls.  The cheapest thing to
// do (space-wise) is to put the nulls in the dictionary.
fn arrow_dictionary_to_data_block(arrays: &[ArrayRef], validity: Option<NullBuffer>) -> DataBlock {
    let array = concat_dict_arrays(arrays);
    let array_dict = array.as_any_dictionary();
    let mut indices = array_dict.keys();
    let num_values = indices.len() as u64;
    let mut values = array_dict.values().clone();
    // Placeholder, if we need to upcast, we will initialize this and set `indices` to refer to it
    let mut upcast = None;

    // TODO: Should we just always normalize indices to u32?  That would make logic simpler
    // and we're going to bitpack them soon anyways

    let indices_block = if let Some(validity) = validity {
        // If there is validity then we find the first invalid index in the dictionary values, inserting
        // a new value if we need to.  Then we change all indices to point to that value.  This way we
        // never need to store nullability of the indices.
        let mut first_invalid_index = None;
        if let Some(values_validity) = values.nulls() {
            first_invalid_index = (!values_validity.inner()).set_indices().next();
        }
        let first_invalid_index = first_invalid_index.unwrap_or_else(|| {
            let null_arr = new_null_array(values.data_type(), 1);
            values = arrow_select::concat::concat(&[values.as_ref(), null_arr.as_ref()]).unwrap();
            let null_index = values.len() - 1;
            let max_index_val = max_index_val(indices.data_type());
            if null_index as u64 > max_index_val {
                // Widen the index type
                if max_index_val >= u32::MAX as u64 {
                    unimplemented!("Dictionary arrays with 2^32 unique value (or more) and a null")
                }
                upcast = Some(arrow_cast::cast(indices, &DataType::UInt32).unwrap());
                indices = upcast.as_ref().unwrap();
            }
            null_index
        });
        // This can't fail since we already checked for fit
        let null_index_arr = arrow_cast::cast(
            &UInt64Array::from(vec![first_invalid_index as u64]),
            indices.data_type(),
        )
        .unwrap();

        let bytes_per_index = indices.data_type().byte_width();
        let bits_per_index = bytes_per_index as u64 * 8;

        let null_index_arr = null_index_arr.into_data();
        let null_index_bytes = &null_index_arr.buffers()[0];
        // Need to make a copy here since indices isn't mutable, could be avoided in theory
        let mut indices_bytes = indices.to_data().buffers()[0].to_vec();
        for invalid_idx in (!validity.inner()).set_indices() {
            indices_bytes[invalid_idx * bytes_per_index..(invalid_idx + 1) * bytes_per_index]
                .copy_from_slice(null_index_bytes.as_slice());
        }
        FixedWidthDataBlock {
            data: LanceBuffer::Owned(indices_bytes),
            bits_per_value: bits_per_index,
            num_values,
        }
    } else {
        FixedWidthDataBlock {
            data: LanceBuffer::Borrowed(indices.to_data().buffers()[0].clone()),
            bits_per_value: indices.data_type().byte_width() as u64 * 8,
            num_values,
        }
    };

    let items = DataBlock::from(values);
    DataBlock::Dictionary(DictionaryDataBlock {
        indices: indices_block,
        dictionary: Box::new(items),
    })
}

enum Nullability {
    None,
    All,
    Some(NullBuffer),
}

impl Nullability {
    fn to_option(&self) -> Option<NullBuffer> {
        match self {
            Self::Some(nulls) => Some(nulls.clone()),
            _ => None,
        }
    }
}

fn extract_nulls(arrays: &[ArrayRef], num_values: u64) -> Nullability {
    let mut has_nulls = false;
    let nulls_and_lens = arrays
        .iter()
        .map(|arr| {
            let nulls = arr.logical_nulls();
            has_nulls |= nulls.is_some();
            (nulls, arr.len())
        })
        .collect::<Vec<_>>();
    if !has_nulls {
        return Nullability::None;
    }
    let mut builder = BooleanBufferBuilder::new(num_values as usize);
    let mut num_nulls = 0;
    for (null, len) in nulls_and_lens {
        if let Some(null) = null {
            num_nulls += null.null_count();
            builder.append_buffer(&null.into_inner());
        } else {
            builder.append_n(len, true);
        }
    }
    if num_nulls == num_values as usize {
        Nullability::All
    } else {
        Nullability::Some(NullBuffer::new(builder.finish()))
    }
}

impl DataBlock {
    pub fn from_arrays(arrays: &[ArrayRef], num_values: u64) -> Self {
        if arrays.is_empty() || num_values == 0 {
            return Self::AllNull(AllNullDataBlock { num_values: 0 });
        }

        let data_type = arrays[0].data_type();
        let nulls = extract_nulls(arrays, num_values);

        if let Nullability::All = nulls {
            return Self::AllNull(AllNullDataBlock { num_values });
        }

        let encoded = match data_type {
            DataType::Binary | DataType::Utf8 => arrow_binary_to_data_block(arrays, num_values, 32),
            DataType::BinaryView | DataType::Utf8View => {
                todo!()
            }
            DataType::LargeBinary | DataType::LargeUtf8 => {
                arrow_binary_to_data_block(arrays, num_values, 64)
            }
            DataType::Boolean => {
                let data = encode_bitmap_data(arrays, num_values);
                Self::FixedWidth(FixedWidthDataBlock {
                    data,
                    bits_per_value: 1,
                    num_values,
                })
            }
            DataType::Date32
            | DataType::Date64
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
            | DataType::Duration(_)
            | DataType::FixedSizeBinary(_)
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Int8
            | DataType::Interval(_)
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::UInt8 => {
                let data = encode_flat_data(arrays, num_values);
                Self::FixedWidth(FixedWidthDataBlock {
                    data,
                    bits_per_value: data_type.byte_width() as u64 * 8,
                    num_values,
                })
            }
            DataType::Null => Self::AllNull(AllNullDataBlock { num_values }),
            DataType::Dictionary(_, _) => arrow_dictionary_to_data_block(arrays, nulls.to_option()),
            DataType::Struct(fields) => {
                let structs = arrays.iter().map(|arr| arr.as_struct()).collect::<Vec<_>>();
                let mut children = Vec::with_capacity(fields.len());
                for child_idx in 0..fields.len() {
                    let childs = structs
                        .iter()
                        .map(|s| s.column(child_idx).clone())
                        .collect::<Vec<_>>();
                    children.push(Self::from_arrays(&childs, num_values));
                }
                Self::Struct(StructDataBlock { children })
            }
            DataType::FixedSizeList(_, dim) => {
                let children = arrays
                    .iter()
                    .map(|arr| arr.as_fixed_size_list().values().clone())
                    .collect::<Vec<_>>();
                let child_block = Self::from_arrays(&children, num_values * *dim as u64);
                match child_block {
                    Self::FixedWidth(inner) => {
                        Self::FixedWidth(FixedWidthDataBlock {
                        data: inner.data,
                        bits_per_value: inner.bits_per_value * *dim as u64,
                        num_values,
                    })},
                    _ => panic!("FSL of something that is not fixed-width cannot be converted to data block"),
                }
            }
            DataType::LargeList(_)
            | DataType::List(_)
            | DataType::ListView(_)
            | DataType::LargeListView(_)
            | DataType::Map(_, _)
            | DataType::RunEndEncoded(_, _)
            | DataType::Union(_, _) => {
                panic!(
                    "Field with data type {} cannot be converted to data block",
                    data_type
                )
            }
        };
        if !matches!(data_type, DataType::Dictionary(_, _)) {
            match nulls {
                Nullability::None => encoded,
                Nullability::Some(nulls) => Self::Nullable(NullableDataBlock {
                    data: Box::new(encoded),
                    nulls: LanceBuffer::Borrowed(nulls.into_inner().into_inner()),
                }),
                _ => unreachable!(),
            }
        } else {
            // Dictionaries already insert the nulls into the dictionary items
            encoded
        }
    }

    pub fn from_array<T: Array + 'static>(array: T) -> Self {
        let num_values = array.len();
        Self::from_arrays(&[Arc::new(array)], num_values as u64)
    }
}

impl From<ArrayRef> for DataBlock {
    fn from(array: ArrayRef) -> Self {
        let num_values = array.len() as u64;
        Self::from_arrays(&[array], num_values)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::Int8Type;
    use arrow_array::{
        ArrayRef, DictionaryArray, Int8Array, LargeBinaryArray, StringArray, UInt8Array,
    };
    use arrow_buffer::{BooleanBuffer, NullBuffer};

    use crate::buffer::LanceBuffer;

    use super::DataBlock;

    #[test]
    fn test_string_to_data_block() {
        // Converting string arrays that contain nulls to DataBlock
        let strings1 = StringArray::from(vec![Some("hello"), None, Some("world")]);
        let strings2 = StringArray::from(vec![Some("a"), Some("b")]);
        let strings3 = StringArray::from(vec![Option::<&'static str>::None, None]);

        let arrays = &[strings1, strings2, strings3]
            .iter()
            .map(|arr| Arc::new(arr.clone()) as ArrayRef)
            .collect::<Vec<_>>();

        let block = DataBlock::from_arrays(arrays, 7);

        assert_eq!(block.num_values(), 7);
        let block = block.as_nullable().unwrap();

        assert_eq!(block.nulls, LanceBuffer::Owned(vec![0b00011101]));

        let data = block.data.as_variable_width().unwrap();
        assert_eq!(
            data.offsets,
            LanceBuffer::reinterpret_vec(vec![0, 5, 5, 10, 11, 12, 12, 12])
        );

        assert_eq!(data.data, LanceBuffer::copy_slice(b"helloworldab"));

        // Converting string arrays that do not contain nulls to DataBlock
        let strings1 = StringArray::from(vec![Some("a"), Some("bc")]);
        let strings2 = StringArray::from(vec![Some("def")]);

        let arrays = &[strings1, strings2]
            .iter()
            .map(|arr| Arc::new(arr.clone()) as ArrayRef)
            .collect::<Vec<_>>();

        let block = DataBlock::from_arrays(arrays, 3);

        assert_eq!(block.num_values(), 3);
        // Should be no nullable wrapper
        let data = block.as_variable_width().unwrap();
        assert_eq!(data.offsets, LanceBuffer::reinterpret_vec(vec![0, 1, 3, 6]));
        assert_eq!(data.data, LanceBuffer::copy_slice(b"abcdef"));
    }

    #[test]
    fn test_string_sliced() {
        let check = |arr: Vec<StringArray>, expected_off: Vec<i32>, expected_data: &[u8]| {
            let arrs = arr
                .into_iter()
                .map(|a| Arc::new(a) as ArrayRef)
                .collect::<Vec<_>>();
            let num_rows = arrs.iter().map(|a| a.len()).sum::<usize>() as u64;
            let data = DataBlock::from_arrays(&arrs, num_rows);

            assert_eq!(data.num_values(), num_rows);

            let data = data.as_variable_width().unwrap();
            assert_eq!(data.offsets, LanceBuffer::reinterpret_vec(expected_off));
            assert_eq!(data.data, LanceBuffer::copy_slice(expected_data));
        };

        let string = StringArray::from(vec![Some("hello"), Some("world")]);
        check(vec![string.slice(1, 1)], vec![0, 5], b"world");
        check(vec![string.slice(0, 1)], vec![0, 5], b"hello");
        check(
            vec![string.slice(0, 1), string.slice(1, 1)],
            vec![0, 5, 10],
            b"helloworld",
        );

        let string2 = StringArray::from(vec![Some("foo"), Some("bar")]);
        check(
            vec![string.slice(0, 1), string2.slice(0, 1)],
            vec![0, 5, 8],
            b"hellofoo",
        );
    }

    #[test]
    fn test_large() {
        let arr = LargeBinaryArray::from_vec(vec![b"hello", b"world"]);
        let data = DataBlock::from_array(arr);

        assert_eq!(data.num_values(), 2);
        let data = data.as_variable_width().unwrap();
        assert_eq!(data.bits_per_offset, 64);
        assert_eq!(data.num_values, 2);
        assert_eq!(data.data, LanceBuffer::copy_slice(b"helloworld"));
        assert_eq!(
            data.offsets,
            LanceBuffer::reinterpret_vec(vec![0_u64, 5, 10])
        );
    }

    #[test]
    fn test_dictionary_indices_normalized() {
        let arr1 = DictionaryArray::<Int8Type>::from_iter([Some("a"), Some("a"), Some("b")]);
        let arr2 = DictionaryArray::<Int8Type>::from_iter([Some("b"), Some("c")]);

        let data = DataBlock::from_arrays(&[Arc::new(arr1), Arc::new(arr2)], 5);

        assert_eq!(data.num_values(), 5);
        let data = data.as_dictionary().unwrap();
        let indices = data.indices;
        assert_eq!(indices.bits_per_value, 8);
        assert_eq!(indices.num_values, 5);
        assert_eq!(
            indices.data,
            // You might expect 0, 0, 1, 1, 2 but it seems that arrow's dictionary concat does
            // not actually collapse dictionaries.  This is an arrow problem however, and we don't
            // need to fix it here.
            LanceBuffer::reinterpret_vec::<i8>(vec![0, 0, 1, 2, 3])
        );

        let items = data.dictionary.as_variable_width().unwrap();
        assert_eq!(items.bits_per_offset, 32);
        assert_eq!(items.num_values, 4);
        assert_eq!(items.data, LanceBuffer::copy_slice(b"abbc"));
        assert_eq!(
            items.offsets,
            LanceBuffer::reinterpret_vec(vec![0, 1, 2, 3, 4],)
        );
    }

    #[test]
    fn test_dictionary_nulls() {
        // Test both ways of encoding nulls

        // By default, nulls get encoded into the indices
        let arr1 = DictionaryArray::<Int8Type>::from_iter([None, Some("a"), Some("b")]);
        let arr2 = DictionaryArray::<Int8Type>::from_iter([Some("c"), None]);

        let data = DataBlock::from_arrays(&[Arc::new(arr1), Arc::new(arr2)], 5);

        let check_common = |data: DataBlock| {
            assert_eq!(data.num_values(), 5);
            let dict = data.as_dictionary().unwrap();

            let nullable_items = dict.dictionary.as_nullable().unwrap();
            assert_eq!(nullable_items.nulls, LanceBuffer::Owned(vec![0b00000111]));
            assert_eq!(nullable_items.data.num_values(), 4);

            let items = nullable_items.data.as_variable_width().unwrap();
            assert_eq!(items.bits_per_offset, 32);
            assert_eq!(items.num_values, 4);
            assert_eq!(items.data, LanceBuffer::copy_slice(b"abc"));
            assert_eq!(
                items.offsets,
                LanceBuffer::reinterpret_vec(vec![0, 1, 2, 3, 3],)
            );

            let indices = dict.indices;
            assert_eq!(indices.bits_per_value, 8);
            assert_eq!(indices.num_values, 5);
            assert_eq!(
                indices.data,
                LanceBuffer::reinterpret_vec::<i8>(vec![3, 0, 1, 2, 3])
            );
        };
        println!("Check one");
        check_common(data);

        // However, we can manually create a dictionary where nulls are in the dictionary
        let items = StringArray::from(vec![Some("a"), Some("b"), Some("c"), None]);
        let indices = Int8Array::from(vec![Some(3), Some(0), Some(1), Some(2), Some(3)]);
        let dict = DictionaryArray::new(indices, Arc::new(items));

        let data = DataBlock::from_array(dict);

        println!("Check two");
        check_common(data);
    }

    #[test]
    fn test_dictionary_cannot_add_null() {
        // 256 unique strings
        let items = StringArray::from(
            (0..256)
                .map(|i| Some(String::from_utf8(vec![0; i]).unwrap()))
                .collect::<Vec<_>>(),
        );
        // 257 indices, covering the whole range, plus one null
        let indices = UInt8Array::from(
            (0..=256)
                .map(|i| if i == 256 { None } else { Some(i as u8) })
                .collect::<Vec<_>>(),
        );
        // We want to normalize this by pushing nulls into the dictionary, but we cannot because
        // the dictionary is too large for the index type
        let dict = DictionaryArray::new(indices, Arc::new(items));
        let data = DataBlock::from_array(dict);

        assert_eq!(data.num_values(), 257);

        let dict = data.as_dictionary().unwrap();

        assert_eq!(dict.indices.bits_per_value, 32);
        assert_eq!(
            dict.indices.data,
            LanceBuffer::reinterpret_vec((0_u32..257).collect::<Vec<_>>())
        );

        let nullable_items = dict.dictionary.as_nullable().unwrap();
        let null_buffer = NullBuffer::new(BooleanBuffer::new(
            nullable_items.nulls.into_buffer(),
            0,
            257,
        ));
        for i in 0..256 {
            assert!(!null_buffer.is_null(i));
        }
        assert!(null_buffer.is_null(256));

        assert_eq!(
            nullable_items.data.as_variable_width().unwrap().data.len(),
            32640
        );
    }

    #[test]
    fn test_dictionary_cannot_concatenate() {
        // 256 unique strings
        let items = StringArray::from(
            (0..256)
                .map(|i| Some(String::from_utf8(vec![0; i]).unwrap()))
                .collect::<Vec<_>>(),
        );
        // 256 different unique strings
        let other_items = StringArray::from(
            (0..256)
                .map(|i| Some(String::from_utf8(vec![1; i + 1]).unwrap()))
                .collect::<Vec<_>>(),
        );
        let indices = UInt8Array::from_iter_values(0..=255);
        let dict1 = DictionaryArray::new(indices.clone(), Arc::new(items));
        let dict2 = DictionaryArray::new(indices, Arc::new(other_items));
        let data = DataBlock::from_arrays(&[Arc::new(dict1), Arc::new(dict2)], 512);
        assert_eq!(data.num_values(), 512);

        let dict = data.as_dictionary().unwrap();

        assert_eq!(dict.indices.bits_per_value, 32);
        assert_eq!(
            dict.indices.data,
            LanceBuffer::reinterpret_vec::<u32>((0..512).collect::<Vec<_>>())
        );
        // What fun: 0 + 1 + .. + 255 + 1 + 2 + .. + 256 = 2^16
        assert_eq!(
            dict.dictionary.as_variable_width().unwrap().data.len(),
            65536
        );
    }
}
