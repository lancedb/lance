// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Comparison trait implementations for Scalar.
//!
//! This module provides `PartialEq`, `Eq`, `PartialOrd`, `Ord`, and `Hash`
//! implementations for `Scalar` values.
//!
//! Key semantics:
//! - NULL == NULL (for equality purposes)
//! - NaN == NaN (using total_cmp semantics for floats)
//! - Nulls sort first (less than all non-null values)
//! - Floats use total_cmp() for ordering

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use crate::Scalar;

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        use Scalar::*;
        match (self, other) {
            (Null, Null) => true,
            (Boolean(a), Boolean(b)) => a == b,
            (Int8(a), Int8(b)) => a == b,
            (Int16(a), Int16(b)) => a == b,
            (Int32(a), Int32(b)) => a == b,
            (Int64(a), Int64(b)) => a == b,
            (UInt8(a), UInt8(b)) => a == b,
            (UInt16(a), UInt16(b)) => a == b,
            (UInt32(a), UInt32(b)) => a == b,
            (UInt64(a), UInt64(b)) => a == b,
            (Float16(a), Float16(b)) => match (a, b) {
                (Some(x), Some(y)) => x.to_bits() == y.to_bits(),
                (None, None) => true,
                _ => false,
            },
            (Float32(a), Float32(b)) => match (a, b) {
                (Some(x), Some(y)) => x.to_bits() == y.to_bits(),
                (None, None) => true,
                _ => false,
            },
            (Float64(a), Float64(b)) => match (a, b) {
                (Some(x), Some(y)) => x.to_bits() == y.to_bits(),
                (None, None) => true,
                _ => false,
            },
            (Decimal128(a, p1, s1), Decimal128(b, p2, s2)) => a == b && p1 == p2 && s1 == s2,
            (Decimal256(a, p1, s1), Decimal256(b, p2, s2)) => a == b && p1 == p2 && s1 == s2,
            (Utf8(a), Utf8(b)) => a == b,
            (LargeUtf8(a), LargeUtf8(b)) => a == b,
            (Utf8View(a), Utf8View(b)) => a == b,
            (Binary(a), Binary(b)) => a == b,
            (LargeBinary(a), LargeBinary(b)) => a == b,
            (BinaryView(a), BinaryView(b)) => a == b,
            (FixedSizeBinary(s1, a), FixedSizeBinary(s2, b)) => s1 == s2 && a == b,
            (Date32(a), Date32(b)) => a == b,
            (Date64(a), Date64(b)) => a == b,
            (Time32(a, u1), Time32(b, u2)) => a == b && u1 == u2,
            (Time64(a, u1), Time64(b, u2)) => a == b && u1 == u2,
            (Timestamp(a, u1, tz1), Timestamp(b, u2, tz2)) => a == b && u1 == u2 && tz1 == tz2,
            (Duration(a, u1), Duration(b, u2)) => a == b && u1 == u2,
            (IntervalYearMonth(a), IntervalYearMonth(b)) => a == b,
            (IntervalDayTime(a), IntervalDayTime(b)) => a == b,
            (IntervalMonthDayNano(a), IntervalMonthDayNano(b)) => a == b,
            (List(a), List(b)) => a == b,
            (LargeList(a), LargeList(b)) => a == b,
            (FixedSizeList(a), FixedSizeList(b)) => a == b,
            (Struct(f1, v1), Struct(f2, v2)) => f1 == f2 && v1 == v2,
            (Map(a), Map(b)) => a == b,
            (Dictionary(k1, v1), Dictionary(k2, v2)) => k1 == k2 && v1 == v2,
            _ => false,
        }
    }
}

impl Eq for Scalar {}

impl PartialOrd for Scalar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Scalar {
    fn cmp(&self, other: &Self) -> Ordering {
        use Scalar::*;

        // Helper macro for comparing Option<T> where T: Ord
        macro_rules! cmp_opt {
            ($a:expr, $b:expr) => {
                match ($a, $b) {
                    (None, None) => Ordering::Equal,
                    (None, Some(_)) => Ordering::Less,
                    (Some(_), None) => Ordering::Greater,
                    (Some(x), Some(y)) => x.cmp(y),
                }
            };
        }

        // Helper macro for comparing Option<f*> using total_cmp
        macro_rules! cmp_float {
            ($a:expr, $b:expr) => {
                match ($a, $b) {
                    (None, None) => Ordering::Equal,
                    (None, Some(_)) => Ordering::Less,
                    (Some(_), None) => Ordering::Greater,
                    (Some(x), Some(y)) => x.total_cmp(y),
                }
            };
        }

        // Helper macro for comparing Option<Buffer> (Buffer doesn't impl Ord)
        macro_rules! cmp_opt_buf {
            ($a:expr, $b:expr) => {
                match ($a, $b) {
                    (None, None) => Ordering::Equal,
                    (None, Some(_)) => Ordering::Less,
                    (Some(_), None) => Ordering::Greater,
                    (Some(x), Some(y)) => x.as_slice().cmp(y.as_slice()),
                }
            };
        }

        match (self, other) {
            (Null, Null) => Ordering::Equal,
            (Null, _) => {
                if other.is_null() {
                    Ordering::Equal
                } else {
                    Ordering::Less
                }
            }
            (_, Null) => {
                if self.is_null() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            }

            (Boolean(a), Boolean(b)) => cmp_opt!(a, b),

            (Int8(a), Int8(b)) => cmp_opt!(a, b),
            (Int16(a), Int16(b)) => cmp_opt!(a, b),
            (Int32(a), Int32(b)) => cmp_opt!(a, b),
            (Int64(a), Int64(b)) => cmp_opt!(a, b),

            (UInt8(a), UInt8(b)) => cmp_opt!(a, b),
            (UInt16(a), UInt16(b)) => cmp_opt!(a, b),
            (UInt32(a), UInt32(b)) => cmp_opt!(a, b),
            (UInt64(a), UInt64(b)) => cmp_opt!(a, b),

            (Float16(a), Float16(b)) => cmp_float!(a, b),
            (Float32(a), Float32(b)) => cmp_float!(a, b),
            (Float64(a), Float64(b)) => cmp_float!(a, b),

            (Decimal128(a, p1, s1), Decimal128(b, p2, s2)) => {
                if p1 != p2 || s1 != s2 {
                    panic!(
                        "Cannot compare Decimal128 with different precision/scale: ({}, {}) vs ({}, {})",
                        p1, s1, p2, s2
                    );
                }
                cmp_opt!(a, b)
            }
            (Decimal256(a, p1, s1), Decimal256(b, p2, s2)) => {
                if p1 != p2 || s1 != s2 {
                    panic!(
                        "Cannot compare Decimal256 with different precision/scale: ({}, {}) vs ({}, {})",
                        p1, s1, p2, s2
                    );
                }
                cmp_opt!(a, b)
            }

            (Utf8(a), Utf8(b)) => cmp_opt_buf!(a, b),
            (LargeUtf8(a), LargeUtf8(b)) => cmp_opt_buf!(a, b),
            (Utf8View(a), Utf8View(b)) => cmp_opt_buf!(a, b),
            // Allow comparing different string types
            (Utf8(a) | LargeUtf8(a) | Utf8View(a), Utf8(b) | LargeUtf8(b) | Utf8View(b)) => {
                cmp_opt_buf!(a, b)
            }

            (Binary(a), Binary(b)) => cmp_opt_buf!(a, b),
            (LargeBinary(a), LargeBinary(b)) => cmp_opt_buf!(a, b),
            (BinaryView(a), BinaryView(b)) => cmp_opt_buf!(a, b),
            // Allow comparing different binary types
            (
                Binary(a) | LargeBinary(a) | BinaryView(a),
                Binary(b) | LargeBinary(b) | BinaryView(b),
            ) => cmp_opt_buf!(a, b),

            (FixedSizeBinary(s1, a), FixedSizeBinary(s2, b)) => {
                if s1 != s2 {
                    panic!(
                        "Cannot compare FixedSizeBinary with different sizes: {} vs {}",
                        s1, s2
                    );
                }
                cmp_opt_buf!(a, b)
            }

            (Date32(a), Date32(b)) => cmp_opt!(a, b),
            (Date64(a), Date64(b)) => cmp_opt!(a, b),

            (Time32(a, u1), Time32(b, u2)) => {
                if u1 != u2 {
                    panic!(
                        "Cannot compare Time32 with different units: {:?} vs {:?}",
                        u1, u2
                    );
                }
                cmp_opt!(a, b)
            }
            (Time64(a, u1), Time64(b, u2)) => {
                if u1 != u2 {
                    panic!(
                        "Cannot compare Time64 with different units: {:?} vs {:?}",
                        u1, u2
                    );
                }
                cmp_opt!(a, b)
            }

            (Timestamp(a, u1, _), Timestamp(b, u2, _)) => {
                if u1 != u2 {
                    panic!(
                        "Cannot compare Timestamp with different units: {:?} vs {:?}",
                        u1, u2
                    );
                }
                cmp_opt!(a, b)
            }

            (Duration(a, u1), Duration(b, u2)) => {
                if u1 != u2 {
                    panic!(
                        "Cannot compare Duration with different units: {:?} vs {:?}",
                        u1, u2
                    );
                }
                cmp_opt!(a, b)
            }

            (IntervalYearMonth(a), IntervalYearMonth(b)) => cmp_opt!(a, b),
            (IntervalDayTime(a), IntervalDayTime(b)) => cmp_opt!(a, b),
            (IntervalMonthDayNano(a), IntervalMonthDayNano(b)) => cmp_opt!(a, b),

            // Complex types - compare by array equality or panic
            (List(a), List(b)) => {
                if a == b {
                    Ordering::Equal
                } else {
                    panic!("Cannot order List scalars")
                }
            }
            (LargeList(a), LargeList(b)) => {
                if a == b {
                    Ordering::Equal
                } else {
                    panic!("Cannot order LargeList scalars")
                }
            }
            (FixedSizeList(a), FixedSizeList(b)) => {
                if a == b {
                    Ordering::Equal
                } else {
                    panic!("Cannot order FixedSizeList scalars")
                }
            }
            (Struct(f1, v1), Struct(f2, v2)) => {
                if f1 != f2 {
                    panic!("Cannot compare Struct with different fields");
                }
                for (a, b) in v1.iter().zip(v2.iter()) {
                    match a.cmp(b) {
                        Ordering::Equal => continue,
                        ord => return ord,
                    }
                }
                Ordering::Equal
            }
            (Map(a), Map(b)) => {
                if a == b {
                    Ordering::Equal
                } else {
                    panic!("Cannot order Map scalars")
                }
            }
            (Dictionary(_, v1), Dictionary(_, v2)) => v1.cmp(v2),

            // Mismatched types
            (a, b) => panic!(
                "Cannot compare scalars of different types: {:?} vs {:?}",
                a.data_type(),
                b.data_type()
            ),
        }
    }
}

impl Hash for Scalar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        use Scalar::*;
        std::mem::discriminant(self).hash(state);
        match self {
            Null => {}
            Boolean(v) => v.hash(state),
            Int8(v) => v.hash(state),
            Int16(v) => v.hash(state),
            Int32(v) => v.hash(state),
            Int64(v) => v.hash(state),
            UInt8(v) => v.hash(state),
            UInt16(v) => v.hash(state),
            UInt32(v) => v.hash(state),
            UInt64(v) => v.hash(state),
            Float16(v) => v.map(|f| f.to_bits()).hash(state),
            Float32(v) => v.map(|f| f.to_bits()).hash(state),
            Float64(v) => v.map(|f| f.to_bits()).hash(state),
            Decimal128(v, p, s) => {
                v.hash(state);
                p.hash(state);
                s.hash(state);
            }
            Decimal256(v, p, s) => {
                // i256 doesn't implement Hash, so we hash its bytes
                if let Some(val) = v {
                    val.to_le_bytes().hash(state);
                } else {
                    0u8.hash(state);
                }
                p.hash(state);
                s.hash(state);
            }
            Utf8(v) | LargeUtf8(v) | Utf8View(v) | Binary(v) | LargeBinary(v) | BinaryView(v) => {
                v.as_ref().map(|b| b.as_slice()).hash(state)
            }
            FixedSizeBinary(s, v) => {
                s.hash(state);
                v.as_ref().map(|b| b.as_slice()).hash(state);
            }
            Date32(v) => v.hash(state),
            Date64(v) => v.hash(state),
            Time32(v, u) => {
                v.hash(state);
                u.hash(state);
            }
            Time64(v, u) => {
                v.hash(state);
                u.hash(state);
            }
            Timestamp(v, u, tz) => {
                v.hash(state);
                u.hash(state);
                tz.hash(state);
            }
            Duration(v, u) => {
                v.hash(state);
                u.hash(state);
            }
            IntervalYearMonth(v) => v.hash(state),
            IntervalDayTime(v) => v.hash(state),
            IntervalMonthDayNano(v) => v.hash(state),
            // For complex types, we hash their array data
            List(arr) | LargeList(arr) | FixedSizeList(arr) | Map(arr) => {
                arr.to_data().buffers().iter().for_each(|b| {
                    b.as_slice().hash(state);
                });
            }
            Struct(fields, values) => {
                fields.len().hash(state);
                values.iter().for_each(|v| v.hash(state));
            }
            Dictionary(k, v) => {
                k.hash(state);
                v.hash(state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_buffer::Buffer;
    use std::collections::HashSet;

    #[test]
    fn test_eq_nulls() {
        assert_eq!(Scalar::Null, Scalar::Null);
        assert_eq!(Scalar::Int32(None), Scalar::Int32(None));
        assert_ne!(Scalar::Null, Scalar::Int32(None));
    }

    #[test]
    fn test_eq_floats_nan() {
        let nan1 = Scalar::Float64(Some(f64::NAN));
        let nan2 = Scalar::Float64(Some(f64::NAN));
        assert_eq!(nan1, nan2);

        let neg_zero = Scalar::Float64(Some(-0.0));
        let pos_zero = Scalar::Float64(Some(0.0));
        assert_ne!(neg_zero, pos_zero);
    }

    #[test]
    fn test_ord_nulls_first() {
        let null = Scalar::Int32(None);
        let one = Scalar::Int32(Some(1));
        let two = Scalar::Int32(Some(2));

        assert!(null < one);
        assert!(null < two);
        assert!(one < two);
    }

    #[test]
    fn test_ord_floats_nan() {
        let nan = Scalar::Float64(Some(f64::NAN));
        let inf = Scalar::Float64(Some(f64::INFINITY));
        let neg_inf = Scalar::Float64(Some(f64::NEG_INFINITY));
        let one = Scalar::Float64(Some(1.0));

        // NaN should be greater than everything in total_cmp
        assert!(nan > inf);
        assert!(nan > one);
        assert!(nan > neg_inf);
    }

    #[test]
    fn test_hash_consistency() {
        use std::hash::DefaultHasher;

        fn hash_scalar(s: &Scalar) -> u64 {
            let mut hasher = DefaultHasher::new();
            s.hash(&mut hasher);
            hasher.finish()
        }

        let a = Scalar::Int32(Some(42));
        let b = Scalar::Int32(Some(42));
        assert_eq!(hash_scalar(&a), hash_scalar(&b));

        let nan1 = Scalar::Float64(Some(f64::NAN));
        let nan2 = Scalar::Float64(Some(f64::NAN));
        assert_eq!(hash_scalar(&nan1), hash_scalar(&nan2));
    }

    #[test]
    fn test_hash_set() {
        let mut set = HashSet::new();
        set.insert(Scalar::Int32(Some(1)));
        set.insert(Scalar::Int32(Some(2)));
        set.insert(Scalar::Int32(Some(1)));
        assert_eq!(set.len(), 2);

        set.insert(Scalar::Int32(None));
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_ord_strings() {
        let a = Scalar::Utf8(Some(Buffer::from(b"aaa".as_ref())));
        let b = Scalar::Utf8(Some(Buffer::from(b"bbb".as_ref())));
        assert!(a < b);
    }

    #[test]
    #[should_panic(expected = "Cannot compare scalars of different types")]
    fn test_ord_different_types_panics() {
        let a = Scalar::Int32(Some(1));
        let b = Scalar::Int64(Some(1));
        let _ = a.cmp(&b);
    }
}
