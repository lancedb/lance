// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Display formatting for Scalar values.

use std::fmt::{Display, Formatter, Result};

use crate::Scalar;

impl Display for Scalar {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        use Scalar::*;

        match self {
            Null => write!(f, "NULL"),
            Boolean(None) => write!(f, "NULL"),
            Boolean(Some(v)) => write!(f, "{}", v),
            Int8(None) => write!(f, "NULL"),
            Int8(Some(v)) => write!(f, "{}", v),
            Int16(None) => write!(f, "NULL"),
            Int16(Some(v)) => write!(f, "{}", v),
            Int32(None) => write!(f, "NULL"),
            Int32(Some(v)) => write!(f, "{}", v),
            Int64(None) => write!(f, "NULL"),
            Int64(Some(v)) => write!(f, "{}", v),
            UInt8(None) => write!(f, "NULL"),
            UInt8(Some(v)) => write!(f, "{}", v),
            UInt16(None) => write!(f, "NULL"),
            UInt16(Some(v)) => write!(f, "{}", v),
            UInt32(None) => write!(f, "NULL"),
            UInt32(Some(v)) => write!(f, "{}", v),
            UInt64(None) => write!(f, "NULL"),
            UInt64(Some(v)) => write!(f, "{}", v),
            Float16(None) => write!(f, "NULL"),
            Float16(Some(v)) => write!(f, "{}", v),
            Float32(None) => write!(f, "NULL"),
            Float32(Some(v)) => write!(f, "{}", v),
            Float64(None) => write!(f, "NULL"),
            Float64(Some(v)) => write!(f, "{}", v),
            Decimal128(None, _, _) => write!(f, "NULL"),
            Decimal128(Some(v), precision, scale) => write_decimal(f, *v, *precision, *scale),
            Decimal256(None, _, _) => write!(f, "NULL"),
            Decimal256(Some(v), _precision, scale) => {
                // Convert i256 to string representation
                let s = format!("{}", v);
                // Apply scale
                if *scale > 0 {
                    let scale = *scale as usize;
                    if s.len() <= scale {
                        write!(f, "0.{:0>width$}", s, width = scale)
                    } else {
                        let (int_part, frac_part) = s.split_at(s.len() - scale);
                        write!(f, "{}.{}", int_part, frac_part)
                    }
                } else {
                    write!(f, "{}", s)
                }
            }
            Utf8(None) | LargeUtf8(None) | Utf8View(None) => write!(f, "NULL"),
            Utf8(Some(v)) | LargeUtf8(Some(v)) | Utf8View(Some(v)) => {
                let s = std::str::from_utf8(v).expect("Utf8 scalar must contain valid UTF-8");
                write!(f, "\"{}\"", s)
            }
            Binary(None) | LargeBinary(None) | BinaryView(None) => write!(f, "NULL"),
            Binary(Some(v)) | LargeBinary(Some(v)) | BinaryView(Some(v)) => write_hex(f, v),
            FixedSizeBinary(_, None) => write!(f, "NULL"),
            FixedSizeBinary(_, Some(v)) => write_hex(f, v),
            Date32(None) => write!(f, "NULL"),
            Date32(Some(days)) => {
                // Days since Unix epoch
                let date = chrono_date_from_days(*days as i64);
                write!(f, "{}", date)
            }
            Date64(None) => write!(f, "NULL"),
            Date64(Some(ms)) => {
                // Milliseconds since Unix epoch
                let date = chrono_date_from_ms(*ms);
                write!(f, "{}", date)
            }
            Time32(None, _) => write!(f, "NULL"),
            Time32(Some(v), unit) => {
                let (h, m, s, ns) = time_parts_from_unit(*v as i64, unit);
                write!(f, "{:02}:{:02}:{:02}.{:09}", h, m, s, ns)
            }
            Time64(None, _) => write!(f, "NULL"),
            Time64(Some(v), unit) => {
                let (h, m, s, ns) = time_parts_from_unit(*v, unit);
                write!(f, "{:02}:{:02}:{:02}.{:09}", h, m, s, ns)
            }
            Timestamp(None, _, _) => write!(f, "NULL"),
            Timestamp(Some(v), unit, tz) => {
                let ns = match unit {
                    arrow_schema::TimeUnit::Second => *v * 1_000_000_000,
                    arrow_schema::TimeUnit::Millisecond => *v * 1_000_000,
                    arrow_schema::TimeUnit::Microsecond => *v * 1_000,
                    arrow_schema::TimeUnit::Nanosecond => *v,
                };
                let secs = ns / 1_000_000_000;
                let subsec_ns = (ns % 1_000_000_000) as u32;

                if let Some(tz) = tz {
                    write!(f, "{}T{:09} {}", secs, subsec_ns, tz)
                } else {
                    write!(f, "{}T{:09}", secs, subsec_ns)
                }
            }
            Duration(None, _) => write!(f, "NULL"),
            Duration(Some(v), unit) => {
                let label = match unit {
                    arrow_schema::TimeUnit::Second => "s",
                    arrow_schema::TimeUnit::Millisecond => "ms",
                    arrow_schema::TimeUnit::Microsecond => "us",
                    arrow_schema::TimeUnit::Nanosecond => "ns",
                };
                write!(f, "{}{}", v, label)
            }
            IntervalYearMonth(None) => write!(f, "NULL"),
            IntervalYearMonth(Some(v)) => {
                let years = v / 12;
                let months = v % 12;
                write!(f, "{}y{}m", years, months)
            }
            IntervalDayTime(None) => write!(f, "NULL"),
            IntervalDayTime(Some(v)) => {
                let days = (*v & 0xFFFFFFFF) as i32;
                let ms = (*v >> 32) as i32;
                write!(f, "{}d{}ms", days, ms)
            }
            IntervalMonthDayNano(None) => write!(f, "NULL"),
            IntervalMonthDayNano(Some(v)) => {
                let months = (*v & 0xFFFFFFFF) as i32;
                let days = ((*v >> 32) & 0xFFFFFFFF) as i32;
                let ns = (*v >> 64) as i64;
                write!(f, "{}m{}d{}ns", months, days, ns)
            }
            List(arr) => {
                if arr.is_null(0) {
                    write!(f, "NULL")
                } else {
                    write!(f, "[list]")
                }
            }
            LargeList(arr) => {
                if arr.is_null(0) {
                    write!(f, "NULL")
                } else {
                    write!(f, "[large_list]")
                }
            }
            FixedSizeList(arr) => {
                if arr.is_null(0) {
                    write!(f, "NULL")
                } else {
                    write!(f, "[fixed_size_list]")
                }
            }
            Struct(fields, values) => {
                if values.is_empty() {
                    write!(f, "NULL")
                } else {
                    write!(f, "{{")?;
                    for (i, (field, value)) in fields.iter().zip(values.iter()).enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}: {}", field.name(), value)?;
                    }
                    write!(f, "}}")
                }
            }
            Map(arr) => {
                if arr.is_null(0) {
                    write!(f, "NULL")
                } else {
                    write!(f, "[map]")
                }
            }
            Dictionary(_, value) => write!(f, "{}", value),
        }
    }
}

fn write_decimal(f: &mut Formatter<'_>, value: i128, _precision: u8, scale: i8) -> Result {
    if scale <= 0 {
        write!(f, "{}", value)
    } else {
        let scale = scale as usize;
        let is_neg = value < 0;
        let abs_value = value.unsigned_abs();
        let s = format!("{}", abs_value);

        let result = if s.len() <= scale {
            format!("0.{:0>width$}", s, width = scale)
        } else {
            let (int_part, frac_part) = s.split_at(s.len() - scale);
            format!("{}.{}", int_part, frac_part)
        };

        if is_neg {
            write!(f, "-{}", result)
        } else {
            write!(f, "{}", result)
        }
    }
}

fn write_hex(f: &mut Formatter<'_>, bytes: &[u8]) -> Result {
    write!(f, "0x")?;
    for b in bytes {
        write!(f, "{:02x}", b)?;
    }
    Ok(())
}

fn chrono_date_from_days(days: i64) -> String {
    // Unix epoch is 1970-01-01
    // Simple calculation without chrono dependency
    let epoch_days = 719_468i64; // Days from year 0 to 1970-01-01
    let total_days = epoch_days + days;

    // Algorithm from https://howardhinnant.github.io/date_algorithms.html
    let era = if total_days >= 0 {
        total_days / 146097
    } else {
        (total_days - 146096) / 146097
    };
    let doe = (total_days - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };

    format!("{:04}-{:02}-{:02}", year, m, d)
}

fn chrono_date_from_ms(ms: i64) -> String {
    let days = ms / (24 * 60 * 60 * 1000);
    chrono_date_from_days(days)
}

fn time_parts_from_unit(value: i64, unit: &arrow_schema::TimeUnit) -> (u32, u32, u32, u32) {
    let ns = match unit {
        arrow_schema::TimeUnit::Second => value * 1_000_000_000,
        arrow_schema::TimeUnit::Millisecond => value * 1_000_000,
        arrow_schema::TimeUnit::Microsecond => value * 1_000,
        arrow_schema::TimeUnit::Nanosecond => value,
    };

    let total_secs = ns / 1_000_000_000;
    let subsec_ns = (ns % 1_000_000_000) as u32;

    let h = (total_secs / 3600) as u32;
    let m = ((total_secs % 3600) / 60) as u32;
    let s = (total_secs % 60) as u32;

    (h, m, s, subsec_ns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_buffer::Buffer;

    #[test]
    fn test_display_primitives() {
        assert_eq!(format!("{}", Scalar::Null), "NULL");
        assert_eq!(format!("{}", Scalar::Boolean(Some(true))), "true");
        assert_eq!(format!("{}", Scalar::Boolean(Some(false))), "false");
        assert_eq!(format!("{}", Scalar::Boolean(None)), "NULL");
        assert_eq!(format!("{}", Scalar::Int32(Some(42))), "42");
        assert_eq!(format!("{}", Scalar::Int32(Some(-42))), "-42");
        assert_eq!(format!("{}", Scalar::Int32(None)), "NULL");
    }

    #[test]
    fn test_display_floats() {
        assert_eq!(format!("{}", Scalar::Float64(Some(3.14))), "3.14");
        assert_eq!(format!("{}", Scalar::Float64(None)), "NULL");
    }

    #[test]
    fn test_display_strings() {
        assert_eq!(
            format!("{}", Scalar::Utf8(Some(Buffer::from("hello".as_bytes())))),
            "\"hello\""
        );
        assert_eq!(format!("{}", Scalar::Utf8(None)), "NULL");
    }

    #[test]
    fn test_display_binary() {
        assert_eq!(
            format!("{}", Scalar::Binary(Some(Buffer::from(vec![0xABu8, 0xCD])))),
            "0xabcd"
        );
    }

    #[test]
    fn test_display_decimal() {
        assert_eq!(
            format!("{}", Scalar::Decimal128(Some(12345), 10, 2)),
            "123.45"
        );
        assert_eq!(
            format!("{}", Scalar::Decimal128(Some(-12345), 10, 2)),
            "-123.45"
        );
        assert_eq!(format!("{}", Scalar::Decimal128(Some(45), 10, 2)), "0.45");
    }

    #[test]
    fn test_display_date() {
        // 2020-01-01 is 18262 days since Unix epoch
        assert_eq!(format!("{}", Scalar::Date32(Some(18262))), "2020-01-01");
    }
}
