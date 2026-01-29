// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Byte serialization and deserialization for Scalar values.
//!
//! Format for primitives:
//! ```text
//! | is_null (1 byte) | value bytes (if not null) |
//! ```
//!
//! Format for variable-length types (Utf8, Binary, Utf8View, BinaryView, FixedSizeBinary):
//! ```text
//! | is_null (1 byte) | u32_le length | value bytes |
//! ```
//!
//! Format for large variable-length types (LargeUtf8, LargeBinary):
//! ```text
//! | is_null (1 byte) | u64_le length | value bytes |
//! ```

use arrow_buffer::{i256, Buffer};
use arrow_schema::{ArrowError, DataType, IntervalUnit};
use half::f16;

use crate::Scalar;

type Result<T> = std::result::Result<T, ArrowError>;

const NULL_MARKER: u8 = 0;
const NON_NULL_MARKER: u8 = 1;

/// Serializes a scalar value to bytes.
impl Scalar {
    /// Converts this scalar to a byte representation.
    ///
    /// The format is designed for simple serialization of scalar values,
    /// primarily for use in indexing and storage scenarios.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.write_bytes(&mut buf);
        buf
    }

    fn write_bytes(&self, buf: &mut Vec<u8>) {
        use Scalar::*;

        match self {
            Null => buf.push(NULL_MARKER),
            Boolean(v) => write_opt_primitive(buf, v.map(u8::from)),
            Int8(v) => write_opt_primitive(buf, v.map(|x| x as u8)),
            Int16(v) => write_opt_le(buf, *v),
            Int32(v) => write_opt_le(buf, *v),
            Int64(v) => write_opt_le(buf, *v),
            UInt8(v) => write_opt_primitive(buf, *v),
            UInt16(v) => write_opt_le(buf, *v),
            UInt32(v) => write_opt_le(buf, *v),
            UInt64(v) => write_opt_le(buf, *v),
            Float16(v) => write_opt_le(buf, v.map(|f| f.to_bits())),
            Float32(v) => write_opt_le(buf, v.map(|f| f.to_bits())),
            Float64(v) => write_opt_le(buf, v.map(|f| f.to_bits())),
            Decimal128(v, _, _) => write_opt_le(buf, *v),
            Decimal256(v, _, _) => match v {
                None => buf.push(NULL_MARKER),
                Some(val) => {
                    buf.push(NON_NULL_MARKER);
                    buf.extend_from_slice(&val.to_le_bytes());
                }
            },
            Utf8(v) | Utf8View(v) | Binary(v) | BinaryView(v) => write_opt_bytes(buf, v.as_deref()),
            LargeUtf8(v) | LargeBinary(v) => write_opt_bytes_large(buf, v.as_deref()),
            FixedSizeBinary(_, v) => write_opt_bytes(buf, v.as_deref()),
            Date32(v) => write_opt_le(buf, *v),
            Date64(v) => write_opt_le(buf, *v),
            Time32(v, _) => write_opt_le(buf, *v),
            Time64(v, _) => write_opt_le(buf, *v),
            Timestamp(v, _, _) => write_opt_le(buf, *v),
            Duration(v, _) => write_opt_le(buf, *v),
            IntervalYearMonth(v) => write_opt_le(buf, *v),
            IntervalDayTime(v) => write_opt_le(buf, *v),
            IntervalMonthDayNano(v) => write_opt_le(buf, *v),
            List(_)
            | LargeList(_)
            | FixedSizeList(_)
            | Struct(_, _)
            | Map(_)
            | Dictionary(_, _) => {
                panic!(
                    "Complex types (List, Struct, Map, Dictionary) do not support byte serialization"
                );
            }
        }
    }

    /// Deserializes a scalar from bytes given its data type.
    pub fn from_bytes(data_type: &DataType, bytes: &[u8]) -> Result<Self> {
        let mut offset = 0;
        Self::read_bytes(data_type, bytes, &mut offset)
    }

    fn read_bytes(data_type: &DataType, bytes: &[u8], offset: &mut usize) -> Result<Self> {
        match data_type {
            DataType::Null => {
                read_null_marker(bytes, offset)?;
                Ok(Self::Null)
            }
            DataType::Boolean => {
                let v = read_opt_primitive(bytes, offset)?;
                Ok(Self::Boolean(v.map(|b| b != 0)))
            }
            DataType::Int8 => {
                let v = read_opt_primitive(bytes, offset)?;
                Ok(Self::Int8(v.map(|b| b as i8)))
            }
            DataType::Int16 => {
                let v = read_opt_le::<i16>(bytes, offset)?;
                Ok(Self::Int16(v))
            }
            DataType::Int32 => {
                let v = read_opt_le::<i32>(bytes, offset)?;
                Ok(Self::Int32(v))
            }
            DataType::Int64 => {
                let v = read_opt_le::<i64>(bytes, offset)?;
                Ok(Self::Int64(v))
            }
            DataType::UInt8 => {
                let v = read_opt_primitive(bytes, offset)?;
                Ok(Self::UInt8(v))
            }
            DataType::UInt16 => {
                let v = read_opt_le::<u16>(bytes, offset)?;
                Ok(Self::UInt16(v))
            }
            DataType::UInt32 => {
                let v = read_opt_le::<u32>(bytes, offset)?;
                Ok(Self::UInt32(v))
            }
            DataType::UInt64 => {
                let v = read_opt_le::<u64>(bytes, offset)?;
                Ok(Self::UInt64(v))
            }
            DataType::Float16 => {
                let v = read_opt_le::<u16>(bytes, offset)?;
                Ok(Self::Float16(v.map(f16::from_bits)))
            }
            DataType::Float32 => {
                let v = read_opt_le::<u32>(bytes, offset)?;
                Ok(Self::Float32(v.map(f32::from_bits)))
            }
            DataType::Float64 => {
                let v = read_opt_le::<u64>(bytes, offset)?;
                Ok(Self::Float64(v.map(f64::from_bits)))
            }
            DataType::Decimal128(precision, scale) => {
                let v = read_opt_le::<i128>(bytes, offset)?;
                Ok(Self::Decimal128(v, *precision, *scale))
            }
            DataType::Decimal256(precision, scale) => {
                let is_null = read_null_marker(bytes, offset)?;
                if is_null {
                    Ok(Self::Decimal256(None, *precision, *scale))
                } else {
                    let val_bytes = read_exact(bytes, offset, 32)?;
                    let val = i256::from_le_bytes(val_bytes.try_into().unwrap());
                    Ok(Self::Decimal256(Some(val), *precision, *scale))
                }
            }
            DataType::Utf8 => {
                let v = read_opt_bytes(bytes, offset)?;
                if let Some(b) = &v {
                    std::str::from_utf8(b).map_err(|e| {
                        ArrowError::InvalidArgumentError(format!("Invalid UTF-8: {}", e))
                    })?;
                }
                Ok(Self::Utf8(v.map(Buffer::from)))
            }
            DataType::LargeUtf8 => {
                let v = read_opt_bytes_large(bytes, offset)?;
                if let Some(b) = &v {
                    std::str::from_utf8(b).map_err(|e| {
                        ArrowError::InvalidArgumentError(format!("Invalid UTF-8: {}", e))
                    })?;
                }
                Ok(Self::LargeUtf8(v.map(Buffer::from)))
            }
            DataType::Utf8View => {
                let v = read_opt_bytes(bytes, offset)?;
                if let Some(b) = &v {
                    std::str::from_utf8(b).map_err(|e| {
                        ArrowError::InvalidArgumentError(format!("Invalid UTF-8: {}", e))
                    })?;
                }
                Ok(Self::Utf8View(v.map(Buffer::from)))
            }
            DataType::Binary => {
                let v = read_opt_bytes(bytes, offset)?;
                Ok(Self::Binary(v.map(Buffer::from)))
            }
            DataType::LargeBinary => {
                let v = read_opt_bytes_large(bytes, offset)?;
                Ok(Self::LargeBinary(v.map(Buffer::from)))
            }
            DataType::BinaryView => {
                let v = read_opt_bytes(bytes, offset)?;
                Ok(Self::BinaryView(v.map(Buffer::from)))
            }
            DataType::FixedSizeBinary(size) => {
                let v = read_opt_bytes(bytes, offset)?;
                Ok(Self::FixedSizeBinary(*size, v.map(Buffer::from)))
            }
            DataType::Date32 => {
                let v = read_opt_le::<i32>(bytes, offset)?;
                Ok(Self::Date32(v))
            }
            DataType::Date64 => {
                let v = read_opt_le::<i64>(bytes, offset)?;
                Ok(Self::Date64(v))
            }
            DataType::Time32(unit) => {
                let v = read_opt_le::<i32>(bytes, offset)?;
                Ok(Self::Time32(v, *unit))
            }
            DataType::Time64(unit) => {
                let v = read_opt_le::<i64>(bytes, offset)?;
                Ok(Self::Time64(v, *unit))
            }
            DataType::Timestamp(unit, tz) => {
                let v = read_opt_le::<i64>(bytes, offset)?;
                Ok(Self::Timestamp(v, *unit, tz.clone()))
            }
            DataType::Duration(unit) => {
                let v = read_opt_le::<i64>(bytes, offset)?;
                Ok(Self::Duration(v, *unit))
            }
            DataType::Interval(IntervalUnit::YearMonth) => {
                let v = read_opt_le::<i32>(bytes, offset)?;
                Ok(Self::IntervalYearMonth(v))
            }
            DataType::Interval(IntervalUnit::DayTime) => {
                let v = read_opt_le::<i64>(bytes, offset)?;
                Ok(Self::IntervalDayTime(v))
            }
            DataType::Interval(IntervalUnit::MonthDayNano) => {
                let v = read_opt_le::<i128>(bytes, offset)?;
                Ok(Self::IntervalMonthDayNano(v))
            }
            _ => Err(ArrowError::NotYetImplemented(format!(
                "Byte deserialization not implemented for {:?}",
                data_type
            ))),
        }
    }
}

// Helper functions for writing

fn write_opt_primitive(buf: &mut Vec<u8>, v: Option<u8>) {
    match v {
        None => buf.push(NULL_MARKER),
        Some(val) => {
            buf.push(NON_NULL_MARKER);
            buf.push(val);
        }
    }
}

fn write_opt_le<T: ToLeBytes>(buf: &mut Vec<u8>, v: Option<T>) {
    match v {
        None => buf.push(NULL_MARKER),
        Some(val) => {
            buf.push(NON_NULL_MARKER);
            buf.extend_from_slice(&val.to_le_bytes_vec());
        }
    }
}

fn write_opt_bytes(buf: &mut Vec<u8>, v: Option<&[u8]>) {
    match v {
        None => buf.push(NULL_MARKER),
        Some(bytes) => {
            buf.push(NON_NULL_MARKER);
            buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(bytes);
        }
    }
}

fn write_opt_bytes_large(buf: &mut Vec<u8>, v: Option<&[u8]>) {
    match v {
        None => buf.push(NULL_MARKER),
        Some(bytes) => {
            buf.push(NON_NULL_MARKER);
            buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(bytes);
        }
    }
}

// Helper functions for reading

fn read_null_marker(bytes: &[u8], offset: &mut usize) -> Result<bool> {
    if *offset >= bytes.len() {
        return Err(ArrowError::InvalidArgumentError(
            "Unexpected end of bytes".to_string(),
        ));
    }
    let marker = bytes[*offset];
    *offset += 1;
    Ok(marker == NULL_MARKER)
}

fn read_exact<'a>(bytes: &'a [u8], offset: &mut usize, len: usize) -> Result<&'a [u8]> {
    if *offset + len > bytes.len() {
        return Err(ArrowError::InvalidArgumentError(
            "Unexpected end of bytes".to_string(),
        ));
    }
    let slice = &bytes[*offset..*offset + len];
    *offset += len;
    Ok(slice)
}

fn read_opt_primitive(bytes: &[u8], offset: &mut usize) -> Result<Option<u8>> {
    let is_null = read_null_marker(bytes, offset)?;
    if is_null {
        Ok(None)
    } else {
        if *offset >= bytes.len() {
            return Err(ArrowError::InvalidArgumentError(
                "Unexpected end of bytes".to_string(),
            ));
        }
        let val = bytes[*offset];
        *offset += 1;
        Ok(Some(val))
    }
}

fn read_opt_le<T: FromLeBytes>(bytes: &[u8], offset: &mut usize) -> Result<Option<T>> {
    let is_null = read_null_marker(bytes, offset)?;
    if is_null {
        Ok(None)
    } else {
        let size = std::mem::size_of::<T>();
        let val_bytes = read_exact(bytes, offset, size)?;
        Ok(Some(T::from_le_bytes_slice(val_bytes)))
    }
}

fn read_opt_bytes(bytes: &[u8], offset: &mut usize) -> Result<Option<Vec<u8>>> {
    let is_null = read_null_marker(bytes, offset)?;
    if is_null {
        Ok(None)
    } else {
        let len_bytes = read_exact(bytes, offset, 4)?;
        let len = u32::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
        let val_bytes = read_exact(bytes, offset, len)?;
        Ok(Some(val_bytes.to_vec()))
    }
}

fn read_opt_bytes_large(bytes: &[u8], offset: &mut usize) -> Result<Option<Vec<u8>>> {
    let is_null = read_null_marker(bytes, offset)?;
    if is_null {
        Ok(None)
    } else {
        let len_bytes = read_exact(bytes, offset, 8)?;
        let len = u64::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
        let val_bytes = read_exact(bytes, offset, len)?;
        Ok(Some(val_bytes.to_vec()))
    }
}

// Traits for generic le bytes conversion

trait ToLeBytes {
    fn to_le_bytes_vec(&self) -> Vec<u8>;
}

macro_rules! impl_to_le_bytes {
    ($($t:ty),*) => {
        $(
            impl ToLeBytes for $t {
                fn to_le_bytes_vec(&self) -> Vec<u8> {
                    self.to_le_bytes().to_vec()
                }
            }
        )*
    };
}

impl_to_le_bytes!(i16, i32, i64, i128, u16, u32, u64);

trait FromLeBytes: Sized {
    fn from_le_bytes_slice(bytes: &[u8]) -> Self;
}

macro_rules! impl_from_le_bytes {
    ($($t:ty),*) => {
        $(
            impl FromLeBytes for $t {
                fn from_le_bytes_slice(bytes: &[u8]) -> Self {
                    let arr: [u8; std::mem::size_of::<$t>()] = bytes.try_into().unwrap();
                    Self::from_le_bytes(arr)
                }
            }
        )*
    };
}

impl_from_le_bytes!(i16, i32, i64, i128, u16, u32, u64);

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::TimeUnit;
    use rstest::rstest;
    use std::sync::Arc;

    #[rstest]
    #[case::null(Scalar::Null, DataType::Null)]
    #[case::bool_true(Scalar::Boolean(Some(true)), DataType::Boolean)]
    #[case::bool_false(Scalar::Boolean(Some(false)), DataType::Boolean)]
    #[case::bool_null(Scalar::Boolean(None), DataType::Boolean)]
    #[case::int32(Scalar::Int32(Some(42)), DataType::Int32)]
    #[case::int32_neg(Scalar::Int32(Some(-42)), DataType::Int32)]
    #[case::int32_null(Scalar::Int32(None), DataType::Int32)]
    #[case::int64(Scalar::Int64(Some(1234567890123)), DataType::Int64)]
    #[case::float32(Scalar::Float32(Some(3.14)), DataType::Float32)]
    #[case::float64(Scalar::Float64(Some(2.718281828)), DataType::Float64)]
    #[case::float64_nan(Scalar::Float64(Some(f64::NAN)), DataType::Float64)]
    #[case::utf8(Scalar::Utf8(Some(Buffer::from(b"hello".as_ref()))), DataType::Utf8)]
    #[case::utf8_empty(Scalar::Utf8(Some(Buffer::from(b"".as_ref()))), DataType::Utf8)]
    #[case::utf8_null(Scalar::Utf8(None), DataType::Utf8)]
    #[case::large_utf8(Scalar::LargeUtf8(Some(Buffer::from(b"hello large".as_ref()))), DataType::LargeUtf8)]
    #[case::large_utf8_null(Scalar::LargeUtf8(None), DataType::LargeUtf8)]
    #[case::binary(Scalar::Binary(Some(Buffer::from(vec![1u8, 2, 3]))), DataType::Binary)]
    #[case::large_binary(Scalar::LargeBinary(Some(Buffer::from(vec![4u8, 5, 6]))), DataType::LargeBinary)]
    #[case::large_binary_null(Scalar::LargeBinary(None), DataType::LargeBinary)]
    #[case::date32(Scalar::Date32(Some(19000)), DataType::Date32)]
    fn test_round_trip(#[case] scalar: Scalar, #[case] data_type: DataType) {
        let bytes = scalar.to_bytes();
        let decoded = Scalar::from_bytes(&data_type, &bytes).unwrap();

        // For floats, compare bit patterns to handle NaN
        match (&scalar, &decoded) {
            (Scalar::Float32(Some(a)), Scalar::Float32(Some(b))) => {
                assert_eq!(a.to_bits(), b.to_bits());
            }
            (Scalar::Float64(Some(a)), Scalar::Float64(Some(b))) => {
                assert_eq!(a.to_bits(), b.to_bits());
            }
            _ => {
                assert_eq!(scalar, decoded);
            }
        }
    }

    #[test]
    fn test_decimal128_round_trip() {
        let scalar = Scalar::Decimal128(Some(12345678901234567890), 38, 10);
        let bytes = scalar.to_bytes();
        let decoded = Scalar::from_bytes(&DataType::Decimal128(38, 10), &bytes).unwrap();
        assert_eq!(scalar, decoded);
    }

    #[test]
    fn test_timestamp_round_trip() {
        let scalar = Scalar::Timestamp(
            Some(1234567890123456789),
            TimeUnit::Nanosecond,
            Some(Arc::from("UTC")),
        );
        let bytes = scalar.to_bytes();
        let decoded = Scalar::from_bytes(
            &DataType::Timestamp(TimeUnit::Nanosecond, Some(Arc::from("UTC"))),
            &bytes,
        )
        .unwrap();
        assert_eq!(scalar, decoded);
    }
}
