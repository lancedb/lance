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

use std::{
    collections::HashSet,
    ops::Range,
    sync::{Arc, RwLock},
};

use arrow::array::{ArrayData, ArrayDataBuilder, AsArray};
use arrow_array::{new_empty_array, new_null_array, Array, ArrayRef, UInt64Array};
use arrow_buffer::{ArrowNativeType, BooleanBuffer, BooleanBufferBuilder, NullBuffer};
use arrow_schema::DataType;
use lance_arrow::DataTypeExt;
use snafu::{location, Location};

use lance_core::{Error, Result};

use crate::{
    buffer::LanceBuffer,
    statistics::{ComputeStat, Stat},
};

/// `Encoding` enum serves as a encoding registration center.
///
/// All the encodings added to Lance should register here, and
/// these encodings can be dynamically selected during encoding,
/// users can also specify the particular encoding they want to use in the field metadata.
#[derive(Eq, Hash, PartialEq, Debug)]
pub enum Encoding {
    Bitpack,
    Fsst,
    FixedSizeBinary,
}
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

use std::collections::HashMap;

// `BlockInfo` stores the statistics of this `DataBlock`, such as `NullCount` for `NullableDataBlock`,
// `BitWidth` for `FixedWidthDataBlock`, `Cardinality` for all `DataBlock`
#[derive(Debug, Clone)]
pub struct BlockInfo(pub Arc<RwLock<HashMap<Stat, Arc<dyn Array>>>>);

impl Default for BlockInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockInfo {
    pub fn new() -> Self {
        Self(Arc::new(RwLock::new(HashMap::new())))
    }
}

impl PartialEq for BlockInfo {
    fn eq(&self, other: &Self) -> bool {
        let self_info = self.0.read().unwrap();
        let other_info = other.0.read().unwrap();
        *self_info == *other_info
    }
}
// `UsedEncoding` is used to record the encodings that has applied to a `DataBlock`
#[derive(Debug, Clone)]
pub struct UsedEncoding(Arc<RwLock<HashSet<Encoding>>>);

impl Default for UsedEncoding {
    fn default() -> Self {
        Self::new()
    }
}

impl UsedEncoding {
    pub fn new() -> Self {
        Self(Arc::new(RwLock::new(HashSet::new())))
    }
}

impl PartialEq for UsedEncoding {
    fn eq(&self, other: &Self) -> bool {
        let self_used = self.0.read().unwrap();
        let other_used = other.0.read().unwrap();
        *self_used == *other_used
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

    pub block_info: BlockInfo,

    pub used_encoding: UsedEncoding,
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
            block_info: self.block_info.clone(),
            used_encoding: self.used_encoding.clone(),
        }
    }

    fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            data: Box::new(self.data.try_clone()?),
            nulls: self.nulls.try_clone()?,
            block_info: self.block_info.clone(),
            used_encoding: self.used_encoding.clone(),
        })
    }

    pub fn data_size(&self) -> u64 {
        self.data.data_size() + self.nulls.len() as u64
    }
}

/// A block representing the same constant value repeated many times
#[derive(Debug, PartialEq)]
pub struct ConstantDataBlock {
    /// Data buffer containing the value
    pub data: LanceBuffer,
    /// The number of values
    pub num_values: u64,
}

impl ConstantDataBlock {
    fn into_buffers(self) -> Vec<LanceBuffer> {
        vec![self.data]
    }

    fn into_arrow(self, _data_type: DataType, _validate: bool) -> Result<ArrayData> {
        // We don't need this yet but if we come up with some way of serializing
        // scalars to/from bytes then we could implement it.
        todo!()
    }

    pub fn borrow_and_clone(&mut self) -> Self {
        Self {
            data: self.data.borrow_and_clone(),
            num_values: self.num_values,
        }
    }

    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.try_clone()?,
            num_values: self.num_values,
        })
    }

    pub fn data_size(&self) -> u64 {
        self.data.len() as u64
    }
}

/// A data block for a single buffer of data where each element has a fixed number of bits
#[derive(Debug, PartialEq)]
pub struct FixedWidthDataBlock {
    /// The data buffer
    pub data: LanceBuffer,
    /// The number of bits per value
    pub bits_per_value: u64,
    /// The number of values represented by this block
    pub num_values: u64,

    pub block_info: BlockInfo,

    pub used_encoding: UsedEncoding,
}

impl FixedWidthDataBlock {
    fn do_into_arrow(
        self,
        data_type: DataType,
        num_values: u64,
        validate: bool,
    ) -> Result<ArrayData> {
        let data_buffer = self.data.into_buffer();
        let builder = ArrayDataBuilder::new(data_type)
            .add_buffer(data_buffer)
            .len(num_values as usize)
            .null_count(0);
        if validate {
            Ok(builder.build()?)
        } else {
            Ok(unsafe { builder.build_unchecked() })
        }
    }

    pub fn into_arrow(self, data_type: DataType, validate: bool) -> Result<ArrayData> {
        let root_num_values = self.num_values;
        self.do_into_arrow(data_type, root_num_values, validate)
    }

    pub fn into_buffers(self) -> Vec<LanceBuffer> {
        vec![self.data]
    }

    pub fn borrow_and_clone(&mut self) -> Self {
        Self {
            data: self.data.borrow_and_clone(),
            bits_per_value: self.bits_per_value,
            num_values: self.num_values,
            block_info: self.block_info.clone(),
            used_encoding: self.used_encoding.clone(),
        }
    }

    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.try_clone()?,
            bits_per_value: self.bits_per_value,
            num_values: self.num_values,
            block_info: self.block_info.clone(),
            used_encoding: self.used_encoding.clone(),
        })
    }

    pub fn data_size(&self) -> u64 {
        self.data.len() as u64
    }
}

pub struct VariableWidthDataBlockBuilder {
    offsets: Vec<u32>,
    bytes: Vec<u8>,
}

impl VariableWidthDataBlockBuilder {
    fn new(estimated_size_bytes: u64) -> Self {
        Self {
            offsets: vec![0u32],
            bytes: Vec::with_capacity(estimated_size_bytes as usize),
        }
    }
}

impl DataBlockBuilderImpl for VariableWidthDataBlockBuilder {
    fn append(&mut self, data_block: &mut DataBlock, selection: Range<u64>) {
        let block = data_block.as_variable_width_mut_ref().unwrap();
        assert!(block.bits_per_offset == 32);

        let offsets = block.offsets.borrow_to_typed_slice::<u32>();
        let offsets = offsets.as_ref();

        let start_offset = offsets[selection.start as usize];
        let end_offset = offsets[selection.end as usize];
        let mut previous_len = self.bytes.len();

        self.bytes
            .extend_from_slice(&block.data[start_offset as usize..end_offset as usize]);

        self.offsets.extend(
            offsets[selection.start as usize..selection.end as usize]
                .iter()
                .zip(&offsets[selection.start as usize + 1..=selection.end as usize])
                .map(|(&current, &next)| {
                    let this_value_len = next - current;
                    previous_len += this_value_len as usize;
                    previous_len as u32
                }),
        );
    }

    fn finish(self: Box<Self>) -> DataBlock {
        let num_values = (self.offsets.len() - 1) as u64;
        DataBlock::VariableWidth(VariableWidthBlock {
            data: LanceBuffer::Owned(self.bytes),
            offsets: LanceBuffer::reinterpret_vec(self.offsets),
            bits_per_offset: 32,
            num_values,
            block_info: BlockInfo::new(),
            used_encodings: UsedEncoding::new(),
        })
    }
}

struct FixedWidthDataBlockBuilder {
    bits_per_value: u64,
    bytes_per_value: u64,
    values: Vec<u8>,
}

impl FixedWidthDataBlockBuilder {
    fn new(bits_per_value: u64, estimated_size_bytes: u64) -> Self {
        assert!(bits_per_value % 8 == 0);
        Self {
            bits_per_value,
            bytes_per_value: bits_per_value / 8,
            values: Vec::with_capacity(estimated_size_bytes as usize),
        }
    }
}

impl DataBlockBuilderImpl for FixedWidthDataBlockBuilder {
    fn append(&mut self, data_block: &mut DataBlock, selection: Range<u64>) {
        let block = data_block.as_fixed_width_ref().unwrap();
        assert_eq!(self.bits_per_value, block.bits_per_value);
        let start = selection.start as usize * self.bytes_per_value as usize;
        let end = selection.end as usize * self.bytes_per_value as usize;
        self.values.extend_from_slice(&block.data[start..end]);
    }

    fn finish(self: Box<Self>) -> DataBlock {
        let num_values = (self.values.len() / self.bytes_per_value as usize) as u64;
        DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::Owned(self.values),
            bits_per_value: self.bits_per_value,
            num_values,
            block_info: BlockInfo::new(),
            used_encoding: UsedEncoding::new(),
        })
    }
}

/// A data block to represent a fixed size list
#[derive(Debug)]
pub struct FixedSizeListBlock {
    /// The child data block
    pub child: Box<DataBlock>,
    /// The number of items in each list
    pub dimension: u64,
}

impl FixedSizeListBlock {
    fn borrow_and_clone(&mut self) -> Self {
        Self {
            child: Box::new(self.child.borrow_and_clone()),
            dimension: self.dimension,
        }
    }

    fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            child: Box::new(self.child.try_clone()?),
            dimension: self.dimension,
        })
    }

    fn remove_validity(self) -> Self {
        Self {
            child: Box::new(self.child.remove_validity()),
            dimension: self.dimension,
        }
    }

    fn num_values(&self) -> u64 {
        self.child.num_values() / self.dimension
    }

    /// Try to flatten a FixedSizeListBlock into a FixedWidthDataBlock
    ///
    /// Returns None if any children are nullable
    pub fn try_into_flat(self) -> Option<FixedWidthDataBlock> {
        match *self.child {
            // Cannot flatten a nullable child
            DataBlock::Nullable(_) => None,
            DataBlock::FixedSizeList(inner) => {
                let mut flat = inner.try_into_flat()?;
                flat.bits_per_value *= self.dimension;
                flat.num_values /= self.dimension;
                Some(flat)
            }
            DataBlock::FixedWidth(mut inner) => {
                inner.bits_per_value *= self.dimension;
                inner.num_values /= self.dimension;
                Some(inner)
            }
            _ => panic!(
                "Expected FixedSizeList or FixedWidth data block but found {:?}",
                self
            ),
        }
    }

    /// Convert a flattened values block into a FixedSizeListBlock
    pub fn from_flat(data: FixedWidthDataBlock, data_type: &DataType) -> DataBlock {
        match data_type {
            DataType::FixedSizeList(child_field, dimension) => {
                let mut data = data;
                data.bits_per_value /= *dimension as u64;
                data.num_values *= *dimension as u64;
                let child_data = Self::from_flat(data, child_field.data_type());
                DataBlock::FixedSizeList(Self {
                    child: Box::new(child_data),
                    dimension: *dimension as u64,
                })
            }
            // Base case, we've hit a non-list type
            _ => DataBlock::FixedWidth(data),
        }
    }

    fn into_arrow(self, data_type: DataType, validate: bool) -> Result<ArrayData> {
        let num_values = self.num_values();
        let builder = match &data_type {
            DataType::FixedSizeList(child_field, _) => {
                let child_data = self
                    .child
                    .into_arrow(child_field.data_type().clone(), validate)?;
                ArrayDataBuilder::new(data_type)
                    .add_child_data(child_data)
                    .len(num_values as usize)
                    .null_count(0)
            }
            _ => panic!("Expected FixedSizeList data type and got {:?}", data_type),
        };
        if validate {
            Ok(builder.build()?)
        } else {
            Ok(unsafe { builder.build_unchecked() })
        }
    }

    fn into_buffers(self) -> Vec<LanceBuffer> {
        self.child.into_buffers()
    }

    fn data_size(&self) -> u64 {
        self.child.data_size()
    }
}

struct FixedSizeListBlockBuilder {
    inner: Box<dyn DataBlockBuilderImpl>,
    dimension: u64,
}

impl FixedSizeListBlockBuilder {
    fn new(inner: Box<dyn DataBlockBuilderImpl>, dimension: u64) -> Self {
        Self { inner, dimension }
    }
}

impl DataBlockBuilderImpl for FixedSizeListBlockBuilder {
    fn append(&mut self, data_block: &mut DataBlock, selection: Range<u64>) {
        let selection = selection.start * self.dimension..selection.end * self.dimension;
        let fsl = data_block.as_fixed_size_list_mut_ref().unwrap();
        self.inner.append(fsl.child.as_mut(), selection);
    }

    fn finish(self: Box<Self>) -> DataBlock {
        let inner_block = self.inner.finish();
        DataBlock::FixedSizeList(FixedSizeListBlock {
            child: Box::new(inner_block),
            dimension: self.dimension,
        })
    }
}

/// A data block with no regular structure.  There is no available spot to attach
/// validity / repdef information and it cannot be converted to Arrow without being
/// decoded
#[derive(Debug)]
pub struct OpaqueBlock {
    pub buffers: Vec<LanceBuffer>,
    pub num_values: u64,
    pub block_info: BlockInfo,
    pub used_encoding: UsedEncoding,
}

impl OpaqueBlock {
    fn borrow_and_clone(&mut self) -> Self {
        Self {
            buffers: self
                .buffers
                .iter_mut()
                .map(|b| b.borrow_and_clone())
                .collect(),
            num_values: self.num_values,
            block_info: self.block_info.clone(),
            used_encoding: self.used_encoding.clone(),
        }
    }

    fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            buffers: self
                .buffers
                .iter()
                .map(|b| b.try_clone())
                .collect::<Result<_>>()?,
            num_values: self.num_values,
            block_info: self.block_info.clone(),
            used_encoding: self.used_encoding.clone(),
        })
    }

    pub fn data_size(&self) -> u64 {
        self.buffers.iter().map(|b| b.len() as u64).sum()
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

    pub block_info: BlockInfo,

    pub used_encodings: UsedEncoding,
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
            block_info: self.block_info.clone(),
            used_encodings: self.used_encodings.clone(),
        }
    }

    fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.try_clone()?,
            offsets: self.offsets.try_clone()?,
            bits_per_offset: self.bits_per_offset,
            num_values: self.num_values,
            block_info: self.block_info.clone(),
            used_encodings: self.used_encodings.clone(),
        })
    }

    pub fn data_size(&self) -> u64 {
        (self.data.len() + self.offsets.len()) as u64
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

    fn remove_validity(self) -> Self {
        Self {
            children: self
                .children
                .into_iter()
                .map(|c| c.remove_validity())
                .collect(),
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
    Empty(),
    Constant(ConstantDataBlock),
    AllNull(AllNullDataBlock),
    Nullable(NullableDataBlock),
    FixedWidth(FixedWidthDataBlock),
    FixedSizeList(FixedSizeListBlock),
    VariableWidth(VariableWidthBlock),
    Opaque(OpaqueBlock),
    Struct(StructDataBlock),
    Dictionary(DictionaryDataBlock),
}

impl DataBlock {
    /// Convert self into an Arrow ArrayData
    pub fn into_arrow(self, data_type: DataType, validate: bool) -> Result<ArrayData> {
        match self {
            Self::Empty() => Ok(new_empty_array(&data_type).to_data()),
            Self::Constant(inner) => inner.into_arrow(data_type, validate),
            Self::AllNull(inner) => inner.into_arrow(data_type, validate),
            Self::Nullable(inner) => inner.into_arrow(data_type, validate),
            Self::FixedWidth(inner) => inner.into_arrow(data_type, validate),
            Self::FixedSizeList(inner) => inner.into_arrow(data_type, validate),
            Self::VariableWidth(inner) => inner.into_arrow(data_type, validate),
            Self::Struct(inner) => inner.into_arrow(data_type, validate),
            Self::Dictionary(inner) => inner.into_arrow(data_type, validate),
            Self::Opaque(_) => Err(Error::Internal {
                message: "Cannot convert OpaqueBlock to Arrow".to_string(),
                location: location!(),
            }),
        }
    }

    /// Convert the data block into a collection of buffers for serialization
    ///
    /// The order matters and will be used to reconstruct the data block at read time.
    pub fn into_buffers(self) -> Vec<LanceBuffer> {
        match self {
            Self::Empty() => Vec::default(),
            Self::Constant(inner) => inner.into_buffers(),
            Self::AllNull(inner) => inner.into_buffers(),
            Self::Nullable(inner) => inner.into_buffers(),
            Self::FixedWidth(inner) => inner.into_buffers(),
            Self::FixedSizeList(inner) => inner.into_buffers(),
            Self::VariableWidth(inner) => inner.into_buffers(),
            Self::Struct(inner) => inner.into_buffers(),
            Self::Dictionary(inner) => inner.into_buffers(),
            Self::Opaque(inner) => inner.buffers,
        }
    }

    /// Converts the data buffers into borrowed mode and clones the block
    ///
    /// This is a zero-copy operation but requires a mutable reference to self and, afterwards,
    /// all buffers will be in Borrowed mode.
    pub fn borrow_and_clone(&mut self) -> Self {
        match self {
            Self::Empty() => Self::Empty(),
            Self::Constant(inner) => Self::Constant(inner.borrow_and_clone()),
            Self::AllNull(inner) => Self::AllNull(inner.borrow_and_clone()),
            Self::Nullable(inner) => Self::Nullable(inner.borrow_and_clone()),
            Self::FixedWidth(inner) => Self::FixedWidth(inner.borrow_and_clone()),
            Self::FixedSizeList(inner) => Self::FixedSizeList(inner.borrow_and_clone()),
            Self::VariableWidth(inner) => Self::VariableWidth(inner.borrow_and_clone()),
            Self::Struct(inner) => Self::Struct(inner.borrow_and_clone()),
            Self::Dictionary(inner) => Self::Dictionary(inner.borrow_and_clone()),
            Self::Opaque(inner) => Self::Opaque(inner.borrow_and_clone()),
        }
    }

    /// Try and clone the block
    ///
    /// This will fail if any buffers are in owned mode.  You can call borrow_and_clone() to
    /// ensure that all buffers are in borrowed mode before calling this method.
    pub fn try_clone(&self) -> Result<Self> {
        match self {
            Self::Empty() => Ok(Self::Empty()),
            Self::Constant(inner) => Ok(Self::Constant(inner.try_clone()?)),
            Self::AllNull(inner) => Ok(Self::AllNull(inner.try_clone()?)),
            Self::Nullable(inner) => Ok(Self::Nullable(inner.try_clone()?)),
            Self::FixedWidth(inner) => Ok(Self::FixedWidth(inner.try_clone()?)),
            Self::FixedSizeList(inner) => Ok(Self::FixedSizeList(inner.try_clone()?)),
            Self::VariableWidth(inner) => Ok(Self::VariableWidth(inner.try_clone()?)),
            Self::Struct(inner) => Ok(Self::Struct(inner.try_clone()?)),
            Self::Dictionary(inner) => Ok(Self::Dictionary(inner.try_clone()?)),
            Self::Opaque(inner) => Ok(Self::Opaque(inner.try_clone()?)),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Constant(_) => "Constant",
            Self::Empty() => "Empty",
            Self::AllNull(_) => "AllNull",
            Self::Nullable(_) => "Nullable",
            Self::FixedWidth(_) => "FixedWidth",
            Self::FixedSizeList(_) => "FixedSizeList",
            Self::VariableWidth(_) => "VariableWidth",
            Self::Struct(_) => "Struct",
            Self::Dictionary(_) => "Dictionary",
            Self::Opaque(_) => "Opaque",
        }
    }

    pub fn num_values(&self) -> u64 {
        match self {
            Self::Empty() => 0,
            Self::Constant(inner) => inner.num_values,
            Self::AllNull(inner) => inner.num_values,
            Self::Nullable(inner) => inner.data.num_values(),
            Self::FixedWidth(inner) => inner.num_values,
            Self::FixedSizeList(inner) => inner.num_values(),
            Self::VariableWidth(inner) => inner.num_values,
            Self::Struct(inner) => inner.children[0].num_values(),
            Self::Dictionary(inner) => inner.indices.num_values,
            Self::Opaque(inner) => inner.num_values,
        }
    }

    pub fn data_size(&self) -> u64 {
        match self {
            Self::Empty() => 0,
            Self::Constant(inner) => inner.data_size(),
            Self::AllNull(_) => 0,
            Self::Nullable(inner) => inner.data_size(),
            Self::FixedWidth(inner) => inner.data_size(),
            Self::FixedSizeList(inner) => inner.data_size(),
            Self::VariableWidth(inner) => inner.data_size(),
            Self::Struct(_) => {
                todo!("the data_size method for StructDataBlock is not implemented yet")
            }
            Self::Dictionary(_) => {
                todo!("the data_size method for DictionaryDataBlock is not implemented yet")
            }
            Self::Opaque(inner) => inner.data_size(),
        }
    }

    /// Removes any validity information from the block
    ///
    /// This does not filter the block (e.g. remove rows).  It only removes
    /// the validity bitmaps (if present).  Any garbage masked by null bits
    /// will now appear as proper values.
    pub fn remove_validity(self) -> Self {
        match self {
            Self::Empty() => Self::Empty(),
            Self::Constant(inner) => Self::Constant(inner),
            Self::AllNull(_) => panic!("Cannot remove validity on all-null data"),
            Self::Nullable(inner) => *inner.data,
            Self::FixedWidth(inner) => Self::FixedWidth(inner),
            Self::FixedSizeList(inner) => Self::FixedSizeList(inner.remove_validity()),
            Self::VariableWidth(inner) => Self::VariableWidth(inner),
            Self::Struct(inner) => Self::Struct(inner.remove_validity()),
            Self::Dictionary(inner) => Self::FixedWidth(inner.indices),
            Self::Opaque(inner) => Self::Opaque(inner),
        }
    }

    pub fn make_builder(&self, estimated_size_bytes: u64) -> Box<dyn DataBlockBuilderImpl> {
        match self {
            Self::FixedWidth(inner) => Box::new(FixedWidthDataBlockBuilder::new(
                inner.bits_per_value,
                estimated_size_bytes,
            )),
            Self::VariableWidth(inner) => {
                if inner.bits_per_offset == 32 {
                    Box::new(VariableWidthDataBlockBuilder::new(estimated_size_bytes))
                } else {
                    todo!()
                }
            }
            Self::FixedSizeList(inner) => {
                let inner_builder = inner.child.make_builder(estimated_size_bytes);
                Box::new(FixedSizeListBlockBuilder::new(
                    inner_builder,
                    inner.dimension,
                ))
            }
            _ => todo!(),
        }
    }
}

macro_rules! as_type {
    ($fn_name:ident, $inner:tt, $inner_type:ident) => {
        pub fn $fn_name(self) -> Option<$inner_type> {
            match self {
                Self::$inner(inner) => Some(inner),
                _ => None,
            }
        }
    };
}

macro_rules! as_type_ref {
    ($fn_name:ident, $inner:tt, $inner_type:ident) => {
        pub fn $fn_name(&self) -> Option<&$inner_type> {
            match self {
                Self::$inner(inner) => Some(inner),
                _ => None,
            }
        }
    };
}

macro_rules! as_type_ref_mut {
    ($fn_name:ident, $inner:tt, $inner_type:ident) => {
        pub fn $fn_name(&mut self) -> Option<&mut $inner_type> {
            match self {
                Self::$inner(inner) => Some(inner),
                _ => None,
            }
        }
    };
}

// Cast implementations
impl DataBlock {
    as_type!(as_all_null, AllNull, AllNullDataBlock);
    as_type!(as_nullable, Nullable, NullableDataBlock);
    as_type!(as_fixed_width, FixedWidth, FixedWidthDataBlock);
    as_type!(as_fixed_size_list, FixedSizeList, FixedSizeListBlock);
    as_type!(as_variable_width, VariableWidth, VariableWidthBlock);
    as_type!(as_struct, Struct, StructDataBlock);
    as_type!(as_dictionary, Dictionary, DictionaryDataBlock);
    as_type_ref!(as_all_null_ref, AllNull, AllNullDataBlock);
    as_type_ref!(as_nullable_ref, Nullable, NullableDataBlock);
    as_type_ref!(as_fixed_width_ref, FixedWidth, FixedWidthDataBlock);
    as_type_ref!(as_fixed_size_list_ref, FixedSizeList, FixedSizeListBlock);
    as_type_ref!(as_variable_width_ref, VariableWidth, VariableWidthBlock);
    as_type_ref!(as_struct_ref, Struct, StructDataBlock);
    as_type_ref!(as_dictionary_ref, Dictionary, DictionaryDataBlock);
    as_type_ref_mut!(as_all_null_mut_ref, AllNull, AllNullDataBlock);
    as_type_ref_mut!(as_nullable_mut_ref, Nullable, NullableDataBlock);
    as_type_ref_mut!(as_fixed_width_mut_ref, FixedWidth, FixedWidthDataBlock);
    as_type_ref_mut!(
        as_fixed_size_list_mut_ref,
        FixedSizeList,
        FixedSizeListBlock
    );
    as_type_ref_mut!(as_variable_width_mut_ref, VariableWidth, VariableWidthBlock);
    as_type_ref_mut!(as_struct_mut_ref, Struct, StructDataBlock);
    as_type_ref_mut!(as_dictionary_mut_ref, Dictionary, DictionaryDataBlock);
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
    let data_vec = arrays.iter().map(|arr| arr.to_data()).collect::<Vec<_>>();
    let bytes_per_offset = bits_per_offset as usize / 8;
    let offsets = data_vec
        .iter()
        .map(|d| {
            LanceBuffer::Borrowed(
                d.buffers()[0].slice_with_length(d.offset(), (d.len() + 1) * bytes_per_offset),
            )
        })
        .collect::<Vec<_>>();
    let (offsets, data_ranges) = if bits_per_offset == 32 {
        stitch_offsets::<i32>(offsets)
    } else {
        stitch_offsets::<i64>(offsets)
    };
    let data = data_vec
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
        block_info: BlockInfo::new(),
        used_encodings: UsedEncoding::new(),
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
            block_info: BlockInfo::new(),
            used_encoding: UsedEncoding::new(),
        }
    } else {
        FixedWidthDataBlock {
            data: LanceBuffer::Borrowed(indices.to_data().buffers()[0].clone()),
            bits_per_value: indices.data_type().byte_width() as u64 * 8,
            num_values,
            block_info: BlockInfo::new(),
            used_encoding: UsedEncoding::new(),
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

        let mut encoded = match data_type {
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
                    block_info: BlockInfo::new(),
                    used_encoding: UsedEncoding::new(),
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
                    block_info: BlockInfo::new(),
                    used_encoding: UsedEncoding::new(),
                })
            }
            DataType::Null => Self::AllNull(AllNullDataBlock { num_values }),
            DataType::Dictionary(_, _) => arrow_dictionary_to_data_block(arrays, nulls.to_option()),
            DataType::Struct(fields) => {
                let structs = arrays.iter().map(|arr| arr.as_struct()).collect::<Vec<_>>();
                let mut children = Vec::with_capacity(fields.len());
                for child_idx in 0..fields.len() {
                    let child_vec = structs
                        .iter()
                        .map(|s| s.column(child_idx).clone())
                        .collect::<Vec<_>>();
                    children.push(Self::from_arrays(&child_vec, num_values));
                }
                Self::Struct(StructDataBlock { children })
            }
            DataType::FixedSizeList(_, dim) => {
                let children = arrays
                    .iter()
                    .map(|arr| arr.as_fixed_size_list().values().clone())
                    .collect::<Vec<_>>();
                let child_block = Self::from_arrays(&children, num_values * *dim as u64);
                Self::FixedSizeList(FixedSizeListBlock {
                    child: Box::new(child_block),
                    dimension: *dim as u64,
                })
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

        // compute statistics
        encoded.compute_stat();

        if !matches!(data_type, DataType::Dictionary(_, _)) {
            match nulls {
                Nullability::None => encoded,
                Nullability::Some(nulls) => Self::Nullable(NullableDataBlock {
                    data: Box::new(encoded),
                    nulls: LanceBuffer::Borrowed(nulls.into_inner().into_inner()),
                    block_info: BlockInfo::new(),
                    used_encoding: UsedEncoding::new(),
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

pub trait DataBlockBuilderImpl {
    fn append(&mut self, data_block: &mut DataBlock, selection: Range<u64>);
    fn finish(self: Box<Self>) -> DataBlock;
}

pub struct DataBlockBuilder {
    estimated_size_bytes: u64,
    builder: Option<Box<dyn DataBlockBuilderImpl>>,
}

impl DataBlockBuilder {
    pub fn with_capacity_estimate(estimated_size_bytes: u64) -> Self {
        Self {
            estimated_size_bytes,
            builder: None,
        }
    }

    fn get_builder(&mut self, block: &DataBlock) -> &mut dyn DataBlockBuilderImpl {
        if self.builder.is_none() {
            self.builder = Some(block.make_builder(self.estimated_size_bytes));
        }
        self.builder.as_mut().unwrap().as_mut()
    }

    pub fn append(&mut self, data_block: &mut DataBlock, selection: Range<u64>) {
        self.get_builder(data_block).append(data_block, selection);
    }

    pub fn finish(self) -> DataBlock {
        let builder = self.builder.expect("DataBlockBuilder didn't see any data");
        builder.finish()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{Int32Type, Int8Type};
    use arrow_array::{
        make_array, new_null_array, ArrayRef, DictionaryArray, Int8Array, LargeBinaryArray,
        StringArray, UInt8Array,
    };
    use arrow_buffer::{BooleanBuffer, NullBuffer};

    use arrow_schema::{DataType, Field, Fields};
    use lance_datagen::{array, ArrayGeneratorExt, RowCount, DEFAULT_SEED};
    use rand::SeedableRng;

    use crate::buffer::LanceBuffer;

    use super::{AllNullDataBlock, DataBlock};

    use arrow::compute::concat;
    use arrow_array::Array;
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
        check_common(data);

        // However, we can manually create a dictionary where nulls are in the dictionary
        let items = StringArray::from(vec![Some("a"), Some("b"), Some("c"), None]);
        let indices = Int8Array::from(vec![Some(3), Some(0), Some(1), Some(2), Some(3)]);
        let dict = DictionaryArray::new(indices, Arc::new(items));

        let data = DataBlock::from_array(dict);

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
    fn test_all_null() {
        for data_type in [
            DataType::UInt32,
            DataType::FixedSizeBinary(2),
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, true))),
            DataType::Struct(Fields::from(vec![Field::new("a", DataType::UInt32, true)])),
        ] {
            let block = DataBlock::AllNull(AllNullDataBlock { num_values: 10 });
            let arr = block.into_arrow(data_type.clone(), true).unwrap();
            let arr = make_array(arr);
            let expected = new_null_array(&data_type, 10);
            assert_eq!(&arr, &expected);
        }
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

    #[test]
    fn test_data_size() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(DEFAULT_SEED.0);
        // test data_size() when input has no nulls
        let mut gen = array::rand::<Int32Type>().with_nulls(&[false, false, false]);

        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let block = DataBlock::from_array(arr.clone());
        assert!(block.data_size() == arr.get_buffer_memory_size() as u64);

        let arr = gen.generate(RowCount::from(400), &mut rng).unwrap();
        let block = DataBlock::from_array(arr.clone());
        assert!(block.data_size() == arr.get_buffer_memory_size() as u64);

        // test data_size() when input has nulls
        let mut gen = array::rand::<Int32Type>().with_nulls(&[false, true, false]);
        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let block = DataBlock::from_array(arr.clone());

        let array_data = arr.to_data();
        let total_buffer_size: usize = array_data.buffers().iter().map(|buffer| buffer.len()).sum();
        // the NullBuffer.len() returns the length in bits so we divide_round_up by 8
        let array_nulls_size_in_bytes = (arr.nulls().unwrap().len() + 7) / 8;
        assert!(block.data_size() == (total_buffer_size + array_nulls_size_in_bytes) as u64);

        let arr = gen.generate(RowCount::from(400), &mut rng).unwrap();
        let block = DataBlock::from_array(arr.clone());

        let array_data = arr.to_data();
        let total_buffer_size: usize = array_data.buffers().iter().map(|buffer| buffer.len()).sum();
        let array_nulls_size_in_bytes = (arr.nulls().unwrap().len() + 7) / 8;
        assert!(block.data_size() == (total_buffer_size + array_nulls_size_in_bytes) as u64);

        let mut gen = array::rand::<Int32Type>().with_nulls(&[true, true, false]);
        let arr = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let block = DataBlock::from_array(arr.clone());

        let array_data = arr.to_data();
        let total_buffer_size: usize = array_data.buffers().iter().map(|buffer| buffer.len()).sum();
        let array_nulls_size_in_bytes = (arr.nulls().unwrap().len() + 7) / 8;
        assert!(block.data_size() == (total_buffer_size + array_nulls_size_in_bytes) as u64);

        let arr = gen.generate(RowCount::from(400), &mut rng).unwrap();
        let block = DataBlock::from_array(arr.clone());

        let array_data = arr.to_data();
        let total_buffer_size: usize = array_data.buffers().iter().map(|buffer| buffer.len()).sum();
        let array_nulls_size_in_bytes = (arr.nulls().unwrap().len() + 7) / 8;
        assert!(block.data_size() == (total_buffer_size + array_nulls_size_in_bytes) as u64);

        let mut gen = array::rand::<Int32Type>().with_nulls(&[false, true, false]);
        let arr1 = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let arr2 = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let arr3 = gen.generate(RowCount::from(3), &mut rng).unwrap();
        let block = DataBlock::from_arrays(&[arr1.clone(), arr2.clone(), arr3.clone()], 9);

        let concatenated_array = concat(&[
            &*Arc::new(arr1.clone()) as &dyn Array,
            &*Arc::new(arr2.clone()) as &dyn Array,
            &*Arc::new(arr3.clone()) as &dyn Array,
        ])
        .unwrap();
        let total_buffer_size: usize = concatenated_array
            .to_data()
            .buffers()
            .iter()
            .map(|buffer| buffer.len())
            .sum();

        let total_nulls_size_in_bytes = (concatenated_array.nulls().unwrap().len() + 7) / 8;
        assert!(block.data_size() == (total_buffer_size + total_nulls_size_in_bytes) as u64);
    }
}
