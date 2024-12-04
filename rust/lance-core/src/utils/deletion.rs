// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashSet, ops::Range};

use arrow_array::BooleanArray;
use deepsize::{Context, DeepSizeOf};
use roaring::RoaringBitmap;

/// Threshold for when a DeletionVector::Set should be promoted to a DeletionVector::Bitmap.
const BITMAP_THRESDHOLD: usize = 5_000;
// TODO: Benchmark to find a better value.

/// Represents a set of deleted row offsets in a single fragment.
#[derive(Debug, Clone)]
pub enum DeletionVector {
    NoDeletions,
    Set(HashSet<u32>),
    Bitmap(RoaringBitmap),
}

impl DeepSizeOf for DeletionVector {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        match self {
            Self::NoDeletions => 0,
            Self::Set(set) => set.deep_size_of_children(context),
            // Inexact but probably close enough
            Self::Bitmap(bitmap) => bitmap.serialized_size(),
        }
    }
}

impl DeletionVector {
    #[allow(dead_code)] // Used in tests
    pub fn len(&self) -> usize {
        match self {
            Self::NoDeletions => 0,
            Self::Set(set) => set.len(),
            Self::Bitmap(bitmap) => bitmap.len() as usize,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, i: u32) -> bool {
        match self {
            Self::NoDeletions => false,
            Self::Set(set) => set.contains(&i),
            Self::Bitmap(bitmap) => bitmap.contains(i),
        }
    }

    pub fn contains_range(&self, mut range: Range<u32>) -> bool {
        match self {
            Self::NoDeletions => range.is_empty(),
            Self::Set(set) => range.all(|i| set.contains(&i)),
            Self::Bitmap(bitmap) => bitmap.contains_range(range),
        }
    }

    pub fn iter(&self) -> Box<dyn Iterator<Item = u32> + Send + '_> {
        match self {
            Self::NoDeletions => Box::new(std::iter::empty()),
            Self::Set(set) => Box::new(set.iter().copied()),
            Self::Bitmap(bitmap) => Box::new(bitmap.iter()),
        }
    }

    pub fn into_sorted_iter(self) -> Box<dyn Iterator<Item = u32> + Send + 'static> {
        match self {
            Self::NoDeletions => Box::new(std::iter::empty()),
            Self::Set(set) => {
                // If we're using a set we shouldn't have too many values
                // and so this conversion should be affordable.
                let mut values = Vec::from_iter(set);
                values.sort();
                Box::new(values.into_iter())
            }
            // Bitmaps always iterate in sorted order
            Self::Bitmap(bitmap) => Box::new(bitmap.into_iter()),
        }
    }

    /// Create an iterator that iterates over the values in the deletion vector in sorted order.
    pub fn to_sorted_iter<'a>(&'a self) -> Box<dyn Iterator<Item = u32> + Send + 'a> {
        match self {
            Self::NoDeletions => Box::new(std::iter::empty()),
            // We have to make a clone when we're using a set
            // but sets should be relatively small.
            Self::Set(_) => self.clone().into_sorted_iter(),
            Self::Bitmap(bitmap) => Box::new(bitmap.iter()),
        }
    }

    // Note: deletion vectors are based on 32-bit offsets.  However, this function works
    // even when given 64-bit row addresses.  That is because `id as u32` returns the lower
    // 32 bits (the row offset) and the upper 32 bits are ignored.
    pub fn build_predicate(&self, row_ids: std::slice::Iter<u64>) -> Option<BooleanArray> {
        match self {
            Self::Bitmap(bitmap) => Some(
                row_ids
                    .map(|&id| !bitmap.contains(id as u32))
                    .collect::<Vec<_>>(),
            ),
            Self::Set(set) => Some(
                row_ids
                    .map(|&id| !set.contains(&(id as u32)))
                    .collect::<Vec<_>>(),
            ),
            Self::NoDeletions => None,
        }
        .map(BooleanArray::from)
    }
}

impl Default for DeletionVector {
    fn default() -> Self {
        Self::NoDeletions
    }
}

impl From<&DeletionVector> for RoaringBitmap {
    fn from(value: &DeletionVector) -> Self {
        match value {
            DeletionVector::Bitmap(bitmap) => bitmap.clone(),
            DeletionVector::Set(set) => Self::from_iter(set.iter()),
            DeletionVector::NoDeletions => Self::new(),
        }
    }
}

impl PartialEq for DeletionVector {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::NoDeletions, Self::NoDeletions) => true,
            (Self::Set(set1), Self::Set(set2)) => set1 == set2,
            (Self::Bitmap(bitmap1), Self::Bitmap(bitmap2)) => bitmap1 == bitmap2,
            (Self::Set(set), Self::Bitmap(bitmap)) | (Self::Bitmap(bitmap), Self::Set(set)) => {
                let set = set.iter().copied().collect::<RoaringBitmap>();
                set == *bitmap
            }
            _ => false,
        }
    }
}

impl Extend<u32> for DeletionVector {
    fn extend<T: IntoIterator<Item = u32>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        // The mem::replace allows changing the variant of Self when we only
        // have &mut Self.
        *self = match (std::mem::take(self), iter.size_hint()) {
            (Self::NoDeletions, (_, Some(0))) => Self::NoDeletions,
            (Self::NoDeletions, (lower, _)) if lower >= BITMAP_THRESDHOLD => {
                let bitmap = iter.collect::<RoaringBitmap>();
                Self::Bitmap(bitmap)
            }
            (Self::NoDeletions, (_, Some(upper))) if upper < BITMAP_THRESDHOLD => {
                let set = iter.collect::<HashSet<_>>();
                Self::Set(set)
            }
            (Self::NoDeletions, _) => {
                // We don't know the size, so just try as a set and move to bitmap
                // if it ends up being big.
                let set = iter.collect::<HashSet<_>>();
                if set.len() > BITMAP_THRESDHOLD {
                    let bitmap = set.into_iter().collect::<RoaringBitmap>();
                    Self::Bitmap(bitmap)
                } else {
                    Self::Set(set)
                }
            }
            (Self::Set(mut set), _) => {
                set.extend(iter);
                if set.len() > BITMAP_THRESDHOLD {
                    let bitmap = set.drain().collect::<RoaringBitmap>();
                    Self::Bitmap(bitmap)
                } else {
                    Self::Set(set)
                }
            }
            (Self::Bitmap(mut bitmap), _) => {
                bitmap.extend(iter);
                Self::Bitmap(bitmap)
            }
        };
    }
}

// TODO: impl methods for DeletionVector
/// impl DeletionVector {
///     pub fn get(i: u32) -> bool { ... }
/// }
/// impl BitAnd for DeletionVector { ... }
impl IntoIterator for DeletionVector {
    type IntoIter = Box<dyn Iterator<Item = Self::Item> + Send>;
    type Item = u32;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::NoDeletions => Box::new(std::iter::empty()),
            Self::Set(set) => {
                // In many cases, it's much better if this is sorted. It's
                // guaranteed to be small, so the cost is low.
                let mut sorted = set.into_iter().collect::<Vec<_>>();
                sorted.sort();
                Box::new(sorted.into_iter())
            }
            Self::Bitmap(bitmap) => Box::new(bitmap.into_iter()),
        }
    }
}

impl FromIterator<u32> for DeletionVector {
    fn from_iter<T: IntoIterator<Item = u32>>(iter: T) -> Self {
        let mut deletion_vector = Self::default();
        deletion_vector.extend(iter);
        deletion_vector
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_deletion_vector() {
        let set = HashSet::from_iter(0..100);
        let bitmap = RoaringBitmap::from_iter(0..100);

        let set_dv = DeletionVector::Set(set);
        let bitmap_dv = DeletionVector::Bitmap(bitmap);

        assert_eq!(set_dv, bitmap_dv);
    }

    #[test]
    fn test_threshold() {
        let dv = DeletionVector::from_iter(0..(BITMAP_THRESDHOLD as u32));
        assert!(matches!(dv, DeletionVector::Bitmap(_)));
    }
}
