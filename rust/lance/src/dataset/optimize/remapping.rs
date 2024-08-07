// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for remapping row ids. Necessary before move-stable row ids.
//!

use async_trait::async_trait;
use lance_core::utils::address::RowAddress;
use lance_table::format::Fragment;
use roaring::RoaringTreemap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::Dataset;
use crate::Result;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemappedIndex {
    pub original: Uuid,
    pub new: Uuid,
}

impl RemappedIndex {
    pub fn new(original: Uuid, new: Uuid) -> Self {
        Self { original, new }
    }
}

/// When compaction runs the row ids will change.  This typically means that
/// indices will need to be remapped.  The details of how this happens are not
/// a part of the compaction process and so a trait is defined here to allow
/// for inversion of control.
#[async_trait]
pub trait IndexRemapper: Send + Sync {
    async fn remap_indices(
        &self,
        index_map: HashMap<u64, Option<u64>>,
        affected_fragment_ids: &[u64],
    ) -> Result<Vec<RemappedIndex>>;
}

/// Options for creating an [IndexRemapper]
///
/// Currently we don't have any options but we may need options in the future and so we
/// want to keep a placeholder
pub trait IndexRemapperOptions: Send + Sync {
    fn create_remapper(&self, dataset: &Dataset) -> Result<Box<dyn IndexRemapper>>;
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct IgnoreRemap {}

#[async_trait]
impl IndexRemapper for IgnoreRemap {
    async fn remap_indices(
        &self,
        _: HashMap<u64, Option<u64>>,
        _: &[u64],
    ) -> Result<Vec<RemappedIndex>> {
        Ok(Vec::new())
    }
}

impl IndexRemapperOptions for IgnoreRemap {
    fn create_remapper(&self, _: &Dataset) -> Result<Box<dyn IndexRemapper>> {
        Ok(Box::new(Self {}))
    }
}

/// Iterator that yields row_ids that are in the given fragments but not in
/// the given row_ids iterator.
struct MissingIds<'a, I: Iterator<Item = u64>> {
    row_ids: I,
    expected_row_id: u64,
    current_fragment_idx: usize,
    last: Option<u64>,
    fragments: &'a Vec<Fragment>,
}

impl<'a, I: Iterator<Item = u64>> MissingIds<'a, I> {
    /// row_ids must be sorted in the same order in which the rows would be
    /// found by scanning fragments in the order they are presented in.
    /// fragments is not guaranteed to be sorted by id.
    fn new(row_ids: I, fragments: &'a Vec<Fragment>) -> Self {
        assert!(!fragments.is_empty());
        let first_frag = &fragments[0];
        Self {
            row_ids,
            expected_row_id: first_frag.id * RowAddress::FRAGMENT_SIZE,
            current_fragment_idx: 0,
            last: None,
            fragments,
        }
    }
}

impl<'a, I: Iterator<Item = u64>> Iterator for MissingIds<'a, I> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_fragment_idx >= self.fragments.len() {
                return None;
            }
            let val = if let Some(last) = self.last {
                self.last = None;
                last
            } else {
                // If we've exhausted row_ids but we aren't done then use 0 which
                // is guaranteed to not match because that would mean that row_ids
                // was empty and we check for that earlier.
                self.row_ids.next().unwrap_or(0)
            };

            let current_fragment = &self.fragments[self.current_fragment_idx];
            let frag = val / RowAddress::FRAGMENT_SIZE;
            let expected_row_id = self.expected_row_id;
            self.expected_row_id += 1;
            // We validate before this we should have physical rows recorded
            let current_physical_rows = current_fragment
                .physical_rows
                .expect("Fragment doesn't have physical rows recorded");
            if (self.expected_row_id % RowAddress::FRAGMENT_SIZE) == current_physical_rows as u64 {
                self.current_fragment_idx += 1;
                if self.current_fragment_idx < self.fragments.len() {
                    self.expected_row_id =
                        self.fragments[self.current_fragment_idx].id * RowAddress::FRAGMENT_SIZE;
                }
            }
            if frag != current_fragment.id {
                self.last = Some(val);
                return Some(expected_row_id);
            }
            if val != expected_row_id {
                self.last = Some(val);
                return Some(expected_row_id);
            }
        }
    }
}

pub fn transpose_row_ids(
    row_ids: RoaringTreemap,
    old_fragments: &Vec<Fragment>,
    new_fragments: &[Fragment],
) -> HashMap<u64, Option<u64>> {
    let new_ids = new_fragments.iter().flat_map(|frag| {
        (0..frag.physical_rows.unwrap() as u32).map(|offset| {
            Some(u64::from(RowAddress::new_from_parts(
                frag.id as u32,
                offset,
            )))
        })
    });
    // The hashmap will have an entry for each row id to map plus all rows that
    // were deleted.
    let expected_size = row_ids.len() as usize
        + old_fragments
            .iter()
            .map(|frag| {
                frag.deletion_file
                    .as_ref()
                    .and_then(|d| d.num_deleted_rows)
                    .unwrap_or(0)
            })
            .sum::<usize>();
    // We expect row ids to be unique, so we should already not get many collisions.
    // The default hasher is designed to be resistance to DoS attacks, which is
    // more than we need for this use case.
    let mut mapping: HashMap<u64, Option<u64>> = HashMap::with_capacity(expected_size);
    mapping.extend(row_ids.iter().zip(new_ids));
    MissingIds::new(row_ids.into_iter(), old_fragments).for_each(|id| {
        mapping.insert(id, None);
    });
    mapping
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_indices() {
        // Sanity test to make sure MissingIds works.  Does not test actual functionality so
        // feel free to remove if it becomes inconvenient
        let frags = vec![
            Fragment {
                id: 0,
                files: Vec::new(),
                deletion_file: None,
                row_id_meta: None,
                physical_rows: Some(5),
            },
            Fragment {
                id: 3,
                files: Vec::new(),
                deletion_file: None,
                row_id_meta: None,
                physical_rows: Some(3),
            },
        ];
        let rows = [(0, 1), (0, 3), (0, 4), (3, 0), (3, 2)]
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into());

        let missing = MissingIds::new(rows, &frags).collect::<Vec<_>>();
        let expected_missing = [(0, 0), (0, 2), (3, 1)]
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into())
            .collect::<Vec<u64>>();
        assert_eq!(missing, expected_missing);
    }

    #[test]
    fn test_missing_ids() {
        // test with missing first row
        // test with missing last row
        // test fragment ids out of order

        let fragments = vec![
            Fragment {
                id: 0,
                files: Vec::new(),
                deletion_file: None,
                row_id_meta: None,
                physical_rows: Some(5),
            },
            Fragment {
                id: 3,
                files: Vec::new(),
                deletion_file: None,
                row_id_meta: None,
                physical_rows: Some(3),
            },
            Fragment {
                id: 1,
                files: Vec::new(),
                deletion_file: None,
                row_id_meta: None,
                physical_rows: Some(3),
            },
        ];

        // Written as pairs of (fragment_id, offset)
        let row_ids = vec![
            (0, 1),
            (0, 3),
            (0, 4),
            (3, 0),
            (3, 2),
            (1, 0),
            (1, 1),
            (1, 2),
        ];
        let row_ids = row_ids
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into());
        let result = MissingIds::new(row_ids, &fragments).collect::<Vec<_>>();

        let expected = vec![(0, 0), (0, 2), (3, 1)];
        let expected = expected
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into())
            .collect::<Vec<u64>>();
        assert_eq!(result, expected);
    }
}
