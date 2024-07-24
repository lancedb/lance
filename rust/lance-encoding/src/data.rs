// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;

use arrow::array::{ArrayData, ArrayDataBuilder};
use arrow_schema::DataType;
use snafu::{location, Location};

use lance_core::{Error, Result};

use crate::buffer::LanceBuffer;

pub trait DataBlock: Any {
    fn as_any(&self) -> &dyn Any;
    fn as_any_box(self: Box<Self>) -> Box<dyn Any>;
    fn into_arrow(self: Box<Self>, data_type: DataType, validate: bool) -> Result<ArrayData>;
}

pub trait DataBlockExt {
    fn try_into_layout<T: DataBlock>(self) -> Result<Box<T>>;
}

impl DataBlockExt for Box<dyn DataBlock> {
    fn try_into_layout<T: DataBlock>(self) -> Result<Box<T>> {
        self.as_any_box()
            .downcast::<T>()
            .map_err(|_| Error::Internal {
                message: format!("Couldn't convert to expected layout"),
                location: location!(),
            })
    }
}

pub struct AllNullDataBlock {
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
}

pub struct NullableDataBlock {
    pub data: Box<dyn DataBlock>,
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
}

pub struct FixedWidthDataBlock {
    pub data: LanceBuffer,
    pub bits_per_value: u64,
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
}

pub struct VariableWidthBlock {
    pub data: LanceBuffer,
    pub offsets: LanceBuffer,
    pub bits_per_offset: u32,
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
}

pub struct StructDataBlock {
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
            return Err(Error::Internal {
                message: format!("Expected Struct, got {:?}", data_type),
                location: location!(),
            });
        }
    }
}
