// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Data layouts to represent encoded data in a sub-Arrow format

use std::any::Any;

use arrow::array::{ArrayData, ArrayDataBuilder};
use arrow_buffer::Buffer;
use arrow_schema::DataType;
use snafu::{location, Location};

use lance_core::{Error, Result};

use crate::{buffer::LanceBuffer, encoder::EncodedArray};

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
/// list of a primitive type.
pub trait DataBlock: Any + std::fmt::Debug + Send + Sync {
    /// Get a reference to the Any trait object
    fn as_any(&self) -> &dyn Any;
    /// Convert self into a Box<dyn Any>
    fn as_any_box(self: Box<Self>) -> Box<dyn Any>;
    /// Convert self into an Arrow ArrayData
    fn into_arrow(self: Box<Self>, data_type: DataType, validate: bool) -> Result<ArrayData>;
    /// Converts the data buffers into borrowed mode and clones the block
    ///
    /// This is a zero-copy operation but requires a mutable reference to self and, afterwards,
    /// all buffers will be in Borrowed mode.
    fn borrow_and_clone(&mut self) -> Box<dyn DataBlock>;
    /// Try and clone the block
    ///
    /// This will fail if any buffers are in owned mode.  You can call borrow_and_clone() to
    /// ensure that all buffers are in borrowed mode before calling this method.
    fn try_clone(&self) -> Result<Box<dyn DataBlock>>;
}

/// Extension trait for DataBlock
pub trait DataBlockExt {
    /// Try to convert a DataBlock into a specific layout
    fn try_into_layout<T: DataBlock>(self) -> Result<Box<T>>;
}

impl DataBlockExt for Box<dyn DataBlock> {
    fn try_into_layout<T: DataBlock>(self) -> Result<Box<T>> {
        self.as_any_box()
            .downcast::<T>()
            .map_err(|_| Error::Internal {
                message: "Couldn't convert to expected layout".to_string(),
                location: location!(),
            })
    }
}

/// A data block with no buffers where everything is null
#[derive(Debug)]
pub struct AllNullDataBlock {
    /// The number of values represented by this block
    pub num_values: u64,
}

impl DataBlock for AllNullDataBlock {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_box(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn into_arrow(self: Box<Self>, data_type: DataType, _validate: bool) -> Result<ArrayData> {
        Ok(ArrayData::new_null(&data_type, self.num_values as usize))
    }

    fn borrow_and_clone(&mut self) -> Box<dyn DataBlock> {
        Box::new(Self {
            num_values: self.num_values,
        })
    }

    fn try_clone(&self) -> Result<Box<dyn DataBlock>> {
        Ok(Box::new(Self {
            num_values: self.num_values,
        }))
    }
}

/// Wraps a data block and adds nullability information to it
#[derive(Debug)]
pub struct NullableDataBlock {
    /// The underlying data
    pub data: Box<dyn DataBlock>,
    /// A bitmap of validity for each value
    pub nulls: LanceBuffer,
}

impl DataBlock for NullableDataBlock {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_box(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn into_arrow(self: Box<Self>, data_type: DataType, validate: bool) -> Result<ArrayData> {
        let nulls = self.nulls.into_buffer();
        let data = self.data.into_arrow(data_type, validate)?.into_builder();
        let data = data.null_bit_buffer(Some(nulls));
        if validate {
            Ok(data.build()?)
        } else {
            Ok(unsafe { data.build_unchecked() })
        }
    }

    fn borrow_and_clone(&mut self) -> Box<dyn DataBlock> {
        Box::new(Self {
            data: self.data.borrow_and_clone(),
            nulls: self.nulls.borrow_and_clone(),
        })
    }

    fn try_clone(&self) -> Result<Box<dyn DataBlock>> {
        Ok(Box::new(Self {
            data: self.data.try_clone()?,
            nulls: self.nulls.try_clone()?,
        }))
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
        self: Box<Self>,
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
}

impl DataBlock for FixedWidthDataBlock {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_box(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn into_arrow(self: Box<Self>, data_type: DataType, validate: bool) -> Result<ArrayData> {
        let root_num_values = self.num_values;
        self.do_into_arrow(data_type, root_num_values, validate)
    }

    fn borrow_and_clone(&mut self) -> Box<dyn DataBlock> {
        Box::new(Self {
            data: self.data.borrow_and_clone(),
            bits_per_value: self.bits_per_value,
            num_values: self.num_values,
        })
    }

    fn try_clone(&self) -> Result<Box<dyn DataBlock>> {
        Ok(Box::new(Self {
            data: self.data.try_clone()?,
            bits_per_value: self.bits_per_value,
            num_values: self.num_values,
        }))
    }
}

/// A data block for variable-width data (e.g. strings, packed rows, etc.)
#[derive(Debug)]
pub struct VariableWidthBlock {
    /// The data buffer
    pub data: LanceBuffer,
    /// The offsets buffer (contains num_values + 1 offsets)
    pub offsets: LanceBuffer,
    /// The number of bits per offset
    pub bits_per_offset: u8,
    /// The number of values represented by this block
    pub num_values: u64,
}

impl DataBlock for VariableWidthBlock {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_box(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn into_arrow(self: Box<Self>, data_type: DataType, validate: bool) -> Result<ArrayData> {
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

    fn borrow_and_clone(&mut self) -> Box<dyn DataBlock> {
        Box::new(Self {
            data: self.data.borrow_and_clone(),
            offsets: self.offsets.borrow_and_clone(),
            bits_per_offset: self.bits_per_offset,
            num_values: self.num_values,
        })
    }

    fn try_clone(&self) -> Result<Box<dyn DataBlock>> {
        Ok(Box::new(Self {
            data: self.data.try_clone()?,
            offsets: self.offsets.try_clone()?,
            bits_per_offset: self.bits_per_offset,
            num_values: self.num_values,
        }))
    }
}

/// A data block representing a struct
#[derive(Debug)]
pub struct StructDataBlock {
    /// The child arrays
    pub children: Vec<Box<dyn DataBlock>>,
}

impl DataBlock for StructDataBlock {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_box(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn into_arrow(self: Box<Self>, data_type: DataType, validate: bool) -> Result<ArrayData> {
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

    fn borrow_and_clone(&mut self) -> Box<dyn DataBlock> {
        Box::new(Self {
            children: self
                .children
                .iter_mut()
                .map(|c| c.borrow_and_clone())
                .collect(),
        })
    }

    fn try_clone(&self) -> Result<Box<dyn DataBlock>> {
        Ok(Box::new(Self {
            children: self
                .children
                .iter()
                .map(|c| c.try_clone())
                .collect::<Result<_>>()?,
        }))
    }
}

/// A data block for dictionary encoded data
#[derive(Debug)]
pub struct DictionaryDataBlock {
    /// The indices buffer
    pub indices: Box<dyn DataBlock>,
    /// The dictionary itself
    pub dictionary: Box<dyn DataBlock>,
}

impl DataBlock for DictionaryDataBlock {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_box(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn into_arrow(self: Box<Self>, data_type: DataType, validate: bool) -> Result<ArrayData> {
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

    fn borrow_and_clone(&mut self) -> Box<dyn DataBlock> {
        Box::new(Self {
            indices: self.indices.borrow_and_clone(),
            dictionary: self.dictionary.borrow_and_clone(),
        })
    }

    fn try_clone(&self) -> Result<Box<dyn DataBlock>> {
        Ok(Box::new(Self {
            indices: self.indices.try_clone()?,
            dictionary: self.dictionary.try_clone()?,
        }))
    }
}

pub trait EncodedDataBlock: Any + std::fmt::Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn as_any_box(self: Box<Self>) -> Box<dyn Any>;

    fn into_parts(self: Box<Self>) -> Vec<Buffer>;
}

pub trait EncodedDataBlockExt {
    fn try_into_layout<T: EncodedDataBlock>(self) -> Result<Box<T>>;
}

// TODO this could probably be combined into a single trait with DataBlockExt ?
impl EncodedDataBlockExt for Box<dyn EncodedDataBlock> {
    fn try_into_layout<T: EncodedDataBlock>(self) -> Result<Box<T>> {
        self.as_any_box()
            .downcast::<T>()
            .map_err(|_| Error::Internal {
                message: "Couldn't convert to expected layout".to_string(),
                location: location!(),
            })
    }
}

/// TODO better comment
/// Each item is encoded with a fixed width
/// e.g. basic flat, bitpacked, bitmap, FSL
#[derive(Debug)]
pub struct FixedWidthEncodedDataBlock {
  pub data: Vec<Buffer>, // TODO change this to data block

  pub bits_per_value: u64,
}

impl EncodedDataBlock for FixedWidthEncodedDataBlock {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_box(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn into_parts(self: Box<Self>) -> Vec<Buffer> {
        self.data
    }
}

/// TODO better comment
/// Each item is encoded as a block of contiguous data
/// e.g. general compression
#[derive(Debug)]
pub struct BlockEncodedDataBlock {
    pub data: Vec<Buffer>,
}

impl EncodedDataBlock for BlockEncodedDataBlock {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_box(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn into_parts(self: Box<Self>) -> Vec<Buffer> {
        self.data
    }
}
