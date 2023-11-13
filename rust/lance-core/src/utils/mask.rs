// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use arrow_array::{Array, BinaryArray, GenericBinaryArray};
use arrow_buffer::{Buffer, NullBuffer, OffsetBuffer};
use roaring::RoaringTreemap;

use crate::Result;

/// A row id mask to select or deselect particular row ids
///
/// If both the allow_list and the block_list are Some then the only selected
/// row ids are those that are in the allow_list but not in the block_list
/// (the block_list takes precedence)
///
/// If both the allow_list and the block_list are None (the default) then
/// all row ids are selected
#[derive(Clone, Debug, Default)]
pub struct RowIdMask {
    /// If Some then only these row ids are selected
    pub allow_list: Option<RoaringTreemap>,
    /// If Some then these row ids are not selected.
    pub block_list: Option<RoaringTreemap>,
}

impl RowIdMask {
    // Create a mask allowing all rows, this is an alias for [default]
    pub fn all_rows() -> Self {
        Self::default()
    }

    // Create a mask that doesn't allow anything
    pub fn allow_nothing() -> Self {
        Self {
            allow_list: Some(RoaringTreemap::new()),
            block_list: None,
        }
    }

    // Create a mask from an allow list
    pub fn from_allowed(allow_list: RoaringTreemap) -> Self {
        Self {
            allow_list: Some(allow_list),
            block_list: None,
        }
    }

    // Create a mask from a block list
    pub fn from_block(block_list: RoaringTreemap) -> Self {
        Self {
            allow_list: None,
            block_list: Some(block_list),
        }
    }

    /// True if the row_id is selected by the mask, false otherwise
    pub fn selected(&self, row_id: u64) -> bool {
        match (&self.allow_list, &self.block_list) {
            (None, None) => true,
            (Some(allow_list), None) => allow_list.contains(row_id),
            (None, Some(block_list)) => !block_list.contains(row_id),
            (Some(allow_list), Some(block_list)) => {
                allow_list.contains(row_id) && !block_list.contains(row_id)
            }
        }
    }

    /// Return the indices of the input row ids that were valid
    pub fn selected_indices(&self, row_ids: &[u64]) -> Vec<u64> {
        let enumerated_ids = row_ids.iter().enumerate();
        match (&self.block_list, &self.allow_list) {
            (Some(block_list), Some(allow_list)) => {
                // Only take rows that are both in the allow list and not in the block list
                enumerated_ids
                    .filter(|(_, row_id)| {
                        !block_list.contains(**row_id) && allow_list.contains(**row_id)
                    })
                    .map(|(idx, _)| idx as u64)
                    .collect()
            }
            (Some(block_list), None) => {
                // Take rows that are not in the block list
                enumerated_ids
                    .filter(|(_, row_id)| !block_list.contains(**row_id))
                    .map(|(idx, _)| idx as u64)
                    .collect()
            }
            (None, Some(allow_list)) => {
                // Take rows that are in the allow list
                enumerated_ids
                    .filter(|(_, row_id)| allow_list.contains(**row_id))
                    .map(|(idx, _)| idx as u64)
                    .collect()
            }
            (None, None) => {
                // We should not encounter this case because callers should
                // check is_empty first.
                panic!("filter_row_ids called but prefilter has nothing to filter with")
            }
        }
    }

    /// Also block the given ids
    pub fn also_block(self, block_list: RoaringTreemap) -> Self {
        if let Some(existing) = self.block_list {
            Self {
                block_list: Some(existing | block_list),
                allow_list: self.allow_list,
            }
        } else {
            Self {
                block_list: Some(block_list),
                allow_list: self.allow_list,
            }
        }
    }

    /// Also allow the given ids
    pub fn also_allow(self, allow_list: RoaringTreemap) -> Self {
        if let Some(existing) = self.allow_list {
            Self {
                block_list: self.block_list,
                allow_list: Some(existing | allow_list),
            }
        } else {
            Self {
                block_list: self.block_list,
                // allow_list = None means "all rows allowed" and so allowing
                //              more rows is meaningless
                allow_list: None,
            }
        }
    }

    /// Converst a mask into an arrow array
    ///
    /// A row id mask is not very arrow-compatible.  We can't make it a batch with
    /// two columns because the block list and allow list will have different lengths.  Also,
    /// there is no Arrow type for compressed bitmaps.
    ///
    /// However, we need to shove it into some kind of Arrow container to pass it along the
    /// datafusion stream.  Perhaps, in the future, we can add row id masks as first class
    /// types in datafusion, and this can be passed along as a mask / selection vector.
    ///
    /// We serialize this as a variable length binary array with two items.  The first item
    /// is the block list and the second item is the allow list.
    pub fn into_arrow(&self) -> Result<BinaryArray> {
        let block_list_length = self
            .block_list
            .as_ref()
            .map(|bl| bl.serialized_size())
            .unwrap_or(0);
        let allow_list_length = self
            .allow_list
            .as_ref()
            .map(|al| al.serialized_size())
            .unwrap_or(0);
        let lengths = vec![block_list_length, allow_list_length];
        let offsets = OffsetBuffer::from_lengths(lengths);
        let mut value_bytes = vec![0; block_list_length + allow_list_length];
        let mut validity = vec![false, false];
        if let Some(block_list) = &self.block_list {
            validity[0] = true;
            block_list.serialize_into(&mut value_bytes[0..])?;
        }
        if let Some(allow_list) = &self.allow_list {
            validity[1] = true;
            allow_list.serialize_into(&mut value_bytes[block_list_length..])?;
        }
        let values = Buffer::from(value_bytes);
        let nulls = NullBuffer::from(validity);
        Ok(BinaryArray::try_new(offsets, values, Some(nulls))?)
    }

    /// Deserialize a row id mask from Arrow
    pub fn from_arrow(array: &GenericBinaryArray<i32>) -> Result<Self> {
        let block_list = if array.is_null(0) {
            None
        } else {
            Some(RoaringTreemap::deserialize_from(array.value(0)))
        }
        .transpose()?;

        let allow_list = if array.is_null(1) {
            None
        } else {
            Some(RoaringTreemap::deserialize_from(array.value(1)))
        }
        .transpose()?;
        Ok(Self {
            block_list,
            allow_list,
        })
    }
}

impl std::ops::Not for RowIdMask {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self {
            block_list: self.allow_list,
            allow_list: self.block_list,
        }
    }
}

impl std::ops::BitAnd for RowIdMask {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        let block_list = match (self.block_list, rhs.block_list) {
            (None, None) => None,
            (Some(lhs), None) => Some(lhs),
            (None, Some(rhs)) => Some(rhs),
            (Some(lhs), Some(rhs)) => Some(lhs | rhs),
        };
        let allow_list = match (self.allow_list, rhs.allow_list) {
            (None, None) => None,
            (Some(lhs), None) => Some(lhs),
            (None, Some(rhs)) => Some(rhs),
            (Some(lhs), Some(rhs)) => Some(lhs & rhs),
        };
        Self {
            block_list,
            allow_list,
        }
    }
}

impl std::ops::BitOr for RowIdMask {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        let block_list = match (self.block_list, rhs.block_list) {
            (None, None) => None,
            (Some(lhs), None) => Some(lhs),
            (None, Some(rhs)) => Some(rhs),
            (Some(lhs), Some(rhs)) => Some(lhs & rhs),
        };
        let allow_list = match (self.allow_list, rhs.allow_list) {
            (None, None) => None,
            // Remember that an allow list of None means "all rows" and
            // so "all rows" | "some rows" is always "all rows"
            (Some(_), None) => None,
            (None, Some(_)) => None,
            (Some(lhs), Some(rhs)) => Some(lhs | rhs),
        };
        Self {
            block_list,
            allow_list,
        }
    }
}

#[cfg(test)]
mod tests {
    use roaring::RoaringTreemap;

    use super::RowIdMask;

    #[test]
    fn test_ops() {
        let mask = RowIdMask::default();
        assert!(mask.selected(1));
        assert!(mask.selected(5));
        let block_list = mask.also_block(RoaringTreemap::from_iter(&[0, 5, 15]));
        assert!(block_list.selected(1));
        assert!(!block_list.selected(5));
        let allow_list = RowIdMask::from_allowed(RoaringTreemap::from_iter(&[0, 2, 5]));
        assert!(!allow_list.selected(1));
        assert!(allow_list.selected(5));
        let combined = block_list & allow_list;
        assert!(combined.selected(2));
        assert!(!combined.selected(0));
        assert!(!combined.selected(5));
        let other = RowIdMask::from_allowed(RoaringTreemap::from_iter(&[3]));
        let combined = combined | other;
        assert!(combined.selected(2));
        assert!(combined.selected(3));
        assert!(!combined.selected(0));
        assert!(!combined.selected(5));

        let block_list = RowIdMask::from_block(RoaringTreemap::from_iter(&[0]));
        let allow_list = RowIdMask::from_allowed(RoaringTreemap::from_iter(&[3]));
        let combined = block_list | allow_list;
        assert!(combined.selected(1));
    }
}
