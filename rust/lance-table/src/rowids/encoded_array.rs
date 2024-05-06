// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;

/// Encoded array of u64 values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodedU64Array {
    /// u64 values represented as u16 offset from a base value.
    ///
    /// Useful when the min and max value are within u16 range (0..65535).
    /// Only space saving when there are more than 2 values.
    U16 { base: u64, offsets: Vec<u16> },
    /// u64 values represented as u32 offset from a base value.
    ///
    /// Useful when the min and max value are within u32 range (0..~4 billion).
    U32 { base: u64, offsets: Vec<u32> },
    /// Just a plain vector of u64 values.
    ///
    /// For when the values cover a wide range.
    U64(Vec<u64>),
}

impl EncodedU64Array {
    pub fn len(&self) -> usize {
        match self {
            Self::U16 { offsets, .. } => offsets.len(),
            Self::U32 { offsets, .. } => offsets.len(),
            Self::U64(values) => values.len(),
        }
    }

    pub fn iter(&self) -> Box<dyn DoubleEndedIterator<Item = u64> + '_> {
        match self {
            Self::U16 { base, offsets } => {
                Box::new(offsets.iter().cloned().map(move |o| base + o as u64))
            }
            Self::U32 { base, offsets } => {
                Box::new(offsets.iter().cloned().map(move |o| base + o as u64))
            }
            Self::U64(values) => Box::new(values.iter().cloned()),
        }
    }

    pub fn get(&self, i: usize) -> Option<u64> {
        match self {
            Self::U16 { base, offsets } => {
                if i < offsets.len() {
                    Some(*base + offsets[i] as u64)
                } else {
                    None
                }
            }
            Self::U32 { base, offsets } => {
                if i < offsets.len() {
                    Some(*base + offsets[i] as u64)
                } else {
                    None
                }
            }
            Self::U64(values) => values.get(i).copied(),
        }
    }

    pub fn min(&self) -> Option<u64> {
        match self {
            Self::U16 { base, offsets } => {
                if offsets.is_empty() {
                    None
                } else {
                    Some(*base)
                }
            }
            Self::U32 { base, offsets } => {
                if offsets.is_empty() {
                    None
                } else {
                    Some(*base)
                }
            }
            Self::U64(values) => values.iter().copied().min(),
        }
    }

    pub fn max(&self) -> Option<u64> {
        match self {
            Self::U16 { base, offsets } => {
                if offsets.is_empty() {
                    None
                } else {
                    Some(*base + *offsets.last().unwrap() as u64)
                }
            }
            Self::U32 { base, offsets } => {
                if offsets.is_empty() {
                    None
                } else {
                    Some(*base + *offsets.last().unwrap() as u64)
                }
            }
            Self::U64(values) => values.iter().copied().max(),
        }
    }

    pub fn first(&self) -> Option<u64> {
        match self {
            Self::U16 { base, offsets } => {
                if offsets.is_empty() {
                    None
                } else {
                    Some(*base + *offsets.first().unwrap() as u64)
                }
            }
            Self::U32 { base, offsets } => {
                if offsets.is_empty() {
                    None
                } else {
                    Some(*base + *offsets.first().unwrap() as u64)
                }
            }
            Self::U64(values) => values.first().copied(),
        }
    }

    pub fn last(&self) -> Option<u64> {
        match self {
            Self::U16 { base, offsets } => {
                if offsets.is_empty() {
                    None
                } else {
                    Some(*base + *offsets.last().unwrap() as u64)
                }
            }
            Self::U32 { base, offsets } => {
                if offsets.is_empty() {
                    None
                } else {
                    Some(*base + *offsets.last().unwrap() as u64)
                }
            }
            Self::U64(values) => values.last().copied(),
        }
    }

    pub fn binary_search(&self, val: u64) -> std::result::Result<usize, usize> {
        match self {
            Self::U16 { base, offsets } => {
                let u16 = val as u16;
                let base = *base as u16;
                offsets.binary_search(&(u16 - base))
            }
            Self::U32 { base, offsets } => {
                let u32 = val as u32;
                let base = *base as u32;
                offsets.binary_search(&(u32 - base))
            }
            Self::U64(values) => values.binary_search(&val),
        }
    }

    pub fn slice(&self, offset: usize, len: usize) -> Self {
        match self {
            Self::U16 { base, offsets } => offsets[offset..(offset + len)]
                .iter()
                .map(|o| *base + *o as u64)
                .collect(),
            Self::U32 { base, offsets } => offsets[offset..(offset + len)]
                .iter()
                .map(|o| *base + *o as u64)
                .collect(),
            Self::U64(values) => {
                let values = values[offset..(offset + len)].to_vec();
                Self::U64(values)
            }
        }
    }
}

impl From<Vec<u64>> for EncodedU64Array {
    fn from(values: Vec<u64>) -> Self {
        let min = values.iter().copied().min().unwrap_or(0);
        let max = values.iter().copied().max().unwrap_or(0);
        let range = max - min;
        if range < u16::MAX as u64 {
            let base = min;
            let offsets = values.iter().map(|v| (*v - base) as u16).collect();
            Self::U16 { base, offsets }
        } else if range < u32::MAX as u64 {
            let base = min;
            let offsets = values.iter().map(|v| (*v - base) as u32).collect();
            Self::U32 { base, offsets }
        } else {
            Self::U64(values)
        }
    }
}

impl From<Range<u64>> for EncodedU64Array {
    fn from(range: Range<u64>) -> Self {
        let min = range.start;
        let max = range.end;
        let range = max - min;
        if range < u16::MAX as u64 {
            let base = min;
            let offsets = (0..range as u16).collect();
            Self::U16 { base, offsets }
        } else if range < u32::MAX as u64 {
            let base = min;
            let offsets = (0..range as u32).collect();
            Self::U32 { base, offsets }
        } else {
            Self::U64((min..max).collect())
        }
    }
}

impl FromIterator<u64> for EncodedU64Array {
    fn from_iter<I: IntoIterator<Item = u64>>(iter: I) -> Self {
        let values: Vec<u64> = iter.into_iter().collect();
        let min = values.iter().copied().min().unwrap_or(0);
        let max = values.iter().copied().max().unwrap_or(0);
        let range = max - min;
        if range < u16::MAX as u64 {
            let base = min;
            let offsets = values.iter().map(|v| (*v - base) as u16).collect();
            Self::U16 { base, offsets }
        } else if range < u32::MAX as u64 {
            let base = min;
            let offsets = values.iter().map(|v| (*v - base) as u32).collect();
            Self::U32 { base, offsets }
        } else {
            Self::U64(values)
        }
    }
}

impl IntoIterator for EncodedU64Array {
    type Item = u64;
    type IntoIter = Box<dyn DoubleEndedIterator<Item = u64>>;
    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::U16 { base, offsets } => {
                Box::new(offsets.into_iter().map(move |o| base + o as u64))
            }
            Self::U32 { base, offsets } => {
                Box::new(offsets.into_iter().map(move |o| base + o as u64))
            }
            Self::U64(values) => Box::new(values.into_iter()),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_encoded_array_from_vec() {
        // u16 version
        let values = [42, 43, 99, u16::MAX as u64]
            .map(|v| v + 2 * u16::MAX as u64)
            .to_vec();
        let encoded = EncodedU64Array::from(values.clone());
        let expected_base = 42 + 2 * u16::MAX as u64;
        assert!(matches!(
            encoded,
            EncodedU64Array::U16 {
                base,
                ..
            } if base == expected_base
        ));
        let roundtripped = encoded.into_iter().collect::<Vec<_>>();
        assert_eq!(values, roundtripped);

        // u32 version
        let values = [42, 43, 99, u32::MAX as u64]
            .map(|v| v + 2 * u32::MAX as u64)
            .to_vec();
        let encoded = EncodedU64Array::from(values.clone());
        let expected_base = 42 + 2 * u32::MAX as u64;
        assert!(matches!(
            encoded,
            EncodedU64Array::U32 {
                base,
                ..
            } if base == expected_base
        ));
        let roundtripped = encoded.into_iter().collect::<Vec<_>>();
        assert_eq!(values, roundtripped);

        // u64 version
        let values = [42, 43, 99, u64::MAX].to_vec();
        let encoded = EncodedU64Array::from(values.clone());
        assert!(matches!(encoded, EncodedU64Array::U64(_)));
        let roundtripped = encoded.into_iter().collect::<Vec<_>>();
        assert_eq!(values, roundtripped);

        // empty one
        let values = Vec::<u64>::new();
        let encoded = EncodedU64Array::from(values.clone());
        assert_eq!(encoded.len(), 0);
        let roundtripped = encoded.into_iter().collect::<Vec<_>>();
        assert_eq!(values, roundtripped);
    }

    #[test]
    fn test_encoded_array_from_range() {
        // u16 version
        let range = (2 * u16::MAX as u64)..(40 + 2 * u16::MAX as u64);
        let encoded = EncodedU64Array::from(range.clone());
        let expected_base = 2 * u16::MAX as u64;
        assert!(
            matches!(
                encoded,
                EncodedU64Array::U16 {
                    base,
                    ..
                } if base == expected_base
            ),
            "{:?}",
            encoded
        );
        let roundtripped = encoded.into_iter().collect::<Vec<_>>();
        assert_eq!(range.collect::<Vec<_>>(), roundtripped);

        // u32 version
        let range = (2 * u32::MAX as u64)..(u16::MAX as u64 + 10 + 2 * u32::MAX as u64);
        let encoded = EncodedU64Array::from(range.clone());
        let expected_base = 2 * u32::MAX as u64;
        assert!(matches!(
            encoded,
            EncodedU64Array::U32 {
                base,
                ..
            } if base == expected_base
        ));
        let roundtripped = encoded.into_iter().collect::<Vec<_>>();
        assert_eq!(range.collect::<Vec<_>>(), roundtripped);

        // We'll skip u64 since it would take a lot of memory.

        // Empty one
        let range = 42..42;
        let encoded = EncodedU64Array::from(range.clone());
        assert_eq!(encoded.len(), 0);
    }
}
