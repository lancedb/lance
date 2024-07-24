// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{
    new_null_array,
    types::{
        ArrowPrimitiveType, ByteArrayType, Date32Type, Date64Type, Decimal128Type, Decimal256Type,
        DurationMicrosecondType, DurationMillisecondType, DurationNanosecondType,
        DurationSecondType, Float16Type, Float32Type, Float64Type, GenericBinaryType,
        GenericStringType, Int16Type, Int32Type, Int64Type, Int8Type, IntervalDayTimeType,
        IntervalMonthDayNanoType, IntervalYearMonthType, Time32MillisecondType, Time32SecondType,
        Time64MicrosecondType, Time64NanosecondType, TimestampMicrosecondType,
        TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt16Type,
        UInt32Type, UInt64Type, UInt8Type,
    },
    ArrayRef, BooleanArray, FixedSizeBinaryArray, FixedSizeListArray, GenericByteArray,
    PrimitiveArray, StructArray,
};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, IntervalUnit, TimeUnit};
use bytes::BytesMut;
use snafu::{location, Location};

use lance_core::{Error, Result};

pub fn new_primitive_array<T: ArrowPrimitiveType>(
    buffers: Vec<BytesMut>,
    num_rows: u64,
    data_type: &DataType,
) -> ArrayRef {
    let mut buffer_iter = buffers.into_iter();
    let null_buffer = buffer_iter.next().unwrap();
    let null_buffer = if null_buffer.is_empty() {
        None
    } else {
        let null_buffer = null_buffer.freeze().into();
        Some(NullBuffer::new(BooleanBuffer::new(
            Buffer::from_bytes(null_buffer),
            0,
            num_rows as usize,
        )))
    };

    let data_buffer = buffer_iter.next().unwrap().freeze();
    let data_buffer = Buffer::from_bytes(data_buffer.into());
    let data_buffer = ScalarBuffer::<T::Native>::new(data_buffer, 0, num_rows as usize);

    // The with_data_type is needed here to recover the parameters for types like Decimal/Timestamp
    Arc::new(PrimitiveArray::<T>::new(data_buffer, null_buffer).with_data_type(data_type.clone()))
}

pub fn new_generic_byte_array<T: ByteArrayType>(buffers: Vec<BytesMut>, num_rows: u64) -> ArrayRef {
    // iterate over buffers to get offsets and then bytes
    let mut buffer_iter = buffers.into_iter();

    let null_buffer = buffer_iter.next().unwrap();
    let null_buffer = if null_buffer.is_empty() {
        None
    } else {
        let null_buffer = null_buffer.freeze().into();
        Some(NullBuffer::new(BooleanBuffer::new(
            Buffer::from_bytes(null_buffer),
            0,
            num_rows as usize,
        )))
    };

    let indices_bytes = buffer_iter.next().unwrap().freeze();
    let indices_buffer = Buffer::from_bytes(indices_bytes.into());
    let indices_buffer = ScalarBuffer::<T::Offset>::new(indices_buffer, 0, num_rows as usize + 1);

    let offsets = OffsetBuffer::new(indices_buffer.clone());

    // Decoding the bytes creates 2 buffers, the first one is empty since
    // validity is stored in an earlier buffer
    buffer_iter.next().unwrap();

    let bytes_buffer = buffer_iter.next().unwrap().freeze();
    let bytes_buffer = Buffer::from_bytes(bytes_buffer.into());
    let bytes_buffer_len = bytes_buffer.len();
    let bytes_buffer = ScalarBuffer::<u8>::new(bytes_buffer, 0, bytes_buffer_len);

    let bytes_array = Arc::new(
        PrimitiveArray::<UInt8Type>::new(bytes_buffer, None).with_data_type(DataType::UInt8),
    );

    Arc::new(GenericByteArray::<T>::new(
        offsets,
        bytes_array.values().into(),
        null_buffer,
    ))
}

pub fn bytes_to_validity(bytes: BytesMut, num_rows: u64) -> Option<NullBuffer> {
    if bytes.is_empty() {
        None
    } else {
        let null_buffer = bytes.freeze().into();
        Some(NullBuffer::new(BooleanBuffer::new(
            Buffer::from_bytes(null_buffer),
            0,
            num_rows as usize,
        )))
    }
}

pub fn primitive_array_from_buffers(
    data_type: &DataType,
    buffers: Vec<BytesMut>,
    num_rows: u64,
) -> Result<ArrayRef> {
    match data_type {
        DataType::Boolean => {
            let mut buffer_iter = buffers.into_iter();
            let null_buffer = buffer_iter.next().unwrap();
            let null_buffer = bytes_to_validity(null_buffer, num_rows);

            let data_buffer = buffer_iter.next().unwrap().freeze();
            let data_buffer = Buffer::from(data_buffer);
            let data_buffer = BooleanBuffer::new(data_buffer, 0, num_rows as usize);

            Ok(Arc::new(BooleanArray::new(data_buffer, null_buffer)))
        }
        DataType::Date32 => Ok(new_primitive_array::<Date32Type>(
            buffers, num_rows, data_type,
        )),
        DataType::Date64 => Ok(new_primitive_array::<Date64Type>(
            buffers, num_rows, data_type,
        )),
        DataType::Decimal128(_, _) => Ok(new_primitive_array::<Decimal128Type>(
            buffers, num_rows, data_type,
        )),
        DataType::Decimal256(_, _) => Ok(new_primitive_array::<Decimal256Type>(
            buffers, num_rows, data_type,
        )),
        DataType::Duration(units) => Ok(match units {
            TimeUnit::Second => {
                new_primitive_array::<DurationSecondType>(buffers, num_rows, data_type)
            }
            TimeUnit::Microsecond => {
                new_primitive_array::<DurationMicrosecondType>(buffers, num_rows, data_type)
            }
            TimeUnit::Millisecond => {
                new_primitive_array::<DurationMillisecondType>(buffers, num_rows, data_type)
            }
            TimeUnit::Nanosecond => {
                new_primitive_array::<DurationNanosecondType>(buffers, num_rows, data_type)
            }
        }),
        DataType::Float16 => Ok(new_primitive_array::<Float16Type>(
            buffers, num_rows, data_type,
        )),
        DataType::Float32 => Ok(new_primitive_array::<Float32Type>(
            buffers, num_rows, data_type,
        )),
        DataType::Float64 => Ok(new_primitive_array::<Float64Type>(
            buffers, num_rows, data_type,
        )),
        DataType::Int16 => Ok(new_primitive_array::<Int16Type>(
            buffers, num_rows, data_type,
        )),
        DataType::Int32 => Ok(new_primitive_array::<Int32Type>(
            buffers, num_rows, data_type,
        )),
        DataType::Int64 => Ok(new_primitive_array::<Int64Type>(
            buffers, num_rows, data_type,
        )),
        DataType::Int8 => Ok(new_primitive_array::<Int8Type>(
            buffers, num_rows, data_type,
        )),
        DataType::Interval(unit) => Ok(match unit {
            IntervalUnit::DayTime => {
                new_primitive_array::<IntervalDayTimeType>(buffers, num_rows, data_type)
            }
            IntervalUnit::MonthDayNano => {
                new_primitive_array::<IntervalMonthDayNanoType>(buffers, num_rows, data_type)
            }
            IntervalUnit::YearMonth => {
                new_primitive_array::<IntervalYearMonthType>(buffers, num_rows, data_type)
            }
        }),
        DataType::Null => Ok(new_null_array(data_type, num_rows as usize)),
        DataType::Time32(unit) => match unit {
            TimeUnit::Millisecond => Ok(new_primitive_array::<Time32MillisecondType>(
                buffers, num_rows, data_type,
            )),
            TimeUnit::Second => Ok(new_primitive_array::<Time32SecondType>(
                buffers, num_rows, data_type,
            )),
            _ => Err(Error::io(
                format!("invalid time unit {:?} for 32-bit time type", unit),
                location!(),
            )),
        },
        DataType::Time64(unit) => match unit {
            TimeUnit::Microsecond => Ok(new_primitive_array::<Time64MicrosecondType>(
                buffers, num_rows, data_type,
            )),
            TimeUnit::Nanosecond => Ok(new_primitive_array::<Time64NanosecondType>(
                buffers, num_rows, data_type,
            )),
            _ => Err(Error::io(
                format!("invalid time unit {:?} for 64-bit time type", unit),
                location!(),
            )),
        },
        DataType::Timestamp(unit, _) => Ok(match unit {
            TimeUnit::Microsecond => {
                new_primitive_array::<TimestampMicrosecondType>(buffers, num_rows, data_type)
            }
            TimeUnit::Millisecond => {
                new_primitive_array::<TimestampMillisecondType>(buffers, num_rows, data_type)
            }
            TimeUnit::Nanosecond => {
                new_primitive_array::<TimestampNanosecondType>(buffers, num_rows, data_type)
            }
            TimeUnit::Second => {
                new_primitive_array::<TimestampSecondType>(buffers, num_rows, data_type)
            }
        }),
        DataType::UInt16 => Ok(new_primitive_array::<UInt16Type>(
            buffers, num_rows, data_type,
        )),
        DataType::UInt32 => Ok(new_primitive_array::<UInt32Type>(
            buffers, num_rows, data_type,
        )),
        DataType::UInt64 => Ok(new_primitive_array::<UInt64Type>(
            buffers, num_rows, data_type,
        )),
        DataType::UInt8 => Ok(new_primitive_array::<UInt8Type>(
            buffers, num_rows, data_type,
        )),
        DataType::FixedSizeBinary(dimension) => {
            let mut buffers_iter = buffers.into_iter();
            let fsb_validity = buffers_iter.next().unwrap();
            let fsb_nulls = bytes_to_validity(fsb_validity, num_rows);

            let fsb_values = buffers_iter.next().unwrap();
            let fsb_values = Buffer::from_bytes(fsb_values.freeze().into());
            Ok(Arc::new(FixedSizeBinaryArray::new(
                *dimension, fsb_values, fsb_nulls,
            )))
        }
        DataType::FixedSizeList(items, dimension) => {
            let mut buffers_iter = buffers.into_iter();
            let fsl_validity = buffers_iter.next().unwrap();
            let fsl_nulls = bytes_to_validity(fsl_validity, num_rows);

            let remaining_buffers = buffers_iter.collect::<Vec<_>>();
            let items_array = primitive_array_from_buffers(
                items.data_type(),
                remaining_buffers,
                num_rows * (*dimension as u64),
            )?;
            Ok(Arc::new(FixedSizeListArray::new(
                items.clone(),
                *dimension,
                items_array,
                fsl_nulls,
            )))
        }
        DataType::Utf8 => Ok(new_generic_byte_array::<GenericStringType<i32>>(
            buffers, num_rows,
        )),
        DataType::LargeUtf8 => Ok(new_generic_byte_array::<GenericStringType<i64>>(
            buffers, num_rows,
        )),
        DataType::Binary => Ok(new_generic_byte_array::<GenericBinaryType<i32>>(
            buffers, num_rows,
        )),
        DataType::LargeBinary => Ok(new_generic_byte_array::<GenericBinaryType<i64>>(
            buffers, num_rows,
        )),
        DataType::Struct(fields) => {
            let mut field_arrays = Vec::new();

            for (field_index, field) in fields.iter().enumerate() {
                let null_bytes = BytesMut::default();
                let mut final_buffers = vec![null_bytes];

                // Pushes a null buffer for inner field of the FSL
                // Right now this works only if inner fields of the FSL are nullable
                if matches!(field.data_type(), DataType::FixedSizeList(_, _)) {
                    final_buffers.push(BytesMut::default());
                }

                final_buffers.push(buffers[field_index].clone());

                let field_array =
                    primitive_array_from_buffers(field.data_type(), final_buffers, num_rows)?;

                field_arrays.push(field_array);
            }

            let struct_array = StructArray::try_new(fields.clone(), field_arrays, None).unwrap();
            Ok(Arc::new(struct_array))
        }
        _ => Err(Error::io(
            format!(
                "The data type {} cannot be decoded from a primitive encoding",
                data_type
            ),
            location!(),
        )),
    }
}
