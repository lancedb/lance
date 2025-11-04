// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for remapping row ids. Necessary before stable row ids.
//!

use crate::dataset::transaction::{Operation, Transaction};
use crate::index::frag_reuse::{load_frag_reuse_index_details, open_frag_reuse_index};
use crate::Result;
use crate::{index, Dataset};
use async_trait::async_trait;
use lance_core::utils::address::RowAddress;
use lance_core::Error;
use lance_index::frag_reuse::{FragDigest, FRAG_REUSE_INDEX_NAME};
use lance_index::DatasetIndexExt;
use lance_table::format::{Fragment, IndexMetadata};
use lance_table::io::manifest::read_manifest_indexes;
use roaring::RoaringTreemap;
use serde::{Deserialize, Serialize};
use snafu::location;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// The result of remapping an index
#[derive(Debug, Clone, PartialEq)]
pub enum RemapResult {
    // Index could not be remapped, drop it
    Drop,
    // No remapping is needed, keep the index as-is
    Keep(Uuid),
    // Index was remapped, return the new index
    Remapped(RemappedIndex),
}

/// A remapped index
#[derive(Debug, Clone, PartialEq)]
pub struct RemappedIndex {
    pub old_id: Uuid,
    pub new_id: Uuid,
    pub index_details: prost_types::Any,
    pub index_version: u32,
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

/// Iterator that yields row_addrs that are in the given fragments but not in
/// the given row_addrs iterator.
struct MissingAddrs<'a, I: Iterator<Item = u64>> {
    row_addrs: I,
    expected_row_addr: u64,
    current_fragment_idx: usize,
    last: Option<u64>,
    fragments: &'a Vec<FragDigest>,
}

impl<'a, I: Iterator<Item = u64>> MissingAddrs<'a, I> {
    /// row_addrs must be sorted in the same order in which the rows would be
    /// found by scanning fragments in the order they are presented in.
    /// fragments is not guaranteed to be sorted by id.
    fn new(row_addrs: I, fragments: &'a Vec<FragDigest>) -> Self {
        assert!(!fragments.is_empty());
        let first_frag = &fragments[0];
        Self {
            row_addrs,
            expected_row_addr: first_frag.id * RowAddress::FRAGMENT_SIZE,
            current_fragment_idx: 0,
            last: None,
            fragments,
        }
    }
}

impl<I: Iterator<Item = u64>> Iterator for MissingAddrs<'_, I> {
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
                // If we've exhausted row_addrs but we aren't done then use 0 which
                // is guaranteed to not match because that would mean that row_addrs
                // was empty and we check for that earlier.
                self.row_addrs.next().unwrap_or(0)
            };

            let current_fragment = &self.fragments[self.current_fragment_idx];
            let frag = val / RowAddress::FRAGMENT_SIZE;
            let expected_row_addr = self.expected_row_addr;
            self.expected_row_addr += 1;

            let current_physical_rows = current_fragment.physical_rows;
            if (self.expected_row_addr % RowAddress::FRAGMENT_SIZE) == current_physical_rows as u64
            {
                self.current_fragment_idx += 1;
                if self.current_fragment_idx < self.fragments.len() {
                    self.expected_row_addr =
                        self.fragments[self.current_fragment_idx].id * RowAddress::FRAGMENT_SIZE;
                }
            }
            if frag != current_fragment.id {
                self.last = Some(val);
                return Some(expected_row_addr);
            }
            if val != expected_row_addr {
                self.last = Some(val);
                return Some(expected_row_addr);
            }
        }
    }
}

pub fn transpose_row_addrs(
    row_addrs: RoaringTreemap,
    old_fragments: &[Fragment],
    new_fragments: &[Fragment],
) -> HashMap<u64, Option<u64>> {
    let old_frag_digests: Vec<FragDigest> = old_fragments.iter().map(|frag| frag.into()).collect();
    let new_frag_digests: Vec<FragDigest> = new_fragments.iter().map(|frag| frag.into()).collect();
    transpose_row_ids_from_digest(row_addrs, &old_frag_digests, &new_frag_digests)
}

pub fn transpose_row_ids_from_digest(
    row_addrs: RoaringTreemap,
    old_fragments: &Vec<FragDigest>,
    new_fragments: &[FragDigest],
) -> HashMap<u64, Option<u64>> {
    let new_addrs = new_fragments.iter().flat_map(|frag| {
        (0..frag.physical_rows as u32).map(|offset| {
            Some(u64::from(RowAddress::new_from_parts(
                frag.id as u32,
                offset,
            )))
        })
    });
    // The hashmap will have an entry for each row addr to map plus all rows that
    // were deleted.
    let expected_size = row_addrs.len() as usize
        + old_fragments
            .iter()
            .map(|frag| frag.num_deleted_rows)
            .sum::<usize>();
    // We expect row addrs to be unique, so we should already not get many collisions.
    // The default hasher is designed to be resistance to DoS attacks, which is
    // more than we need for this use case.
    let mut mapping: HashMap<u64, Option<u64>> = HashMap::with_capacity(expected_size);
    mapping.extend(row_addrs.iter().zip(new_addrs));
    MissingAddrs::new(row_addrs.into_iter(), old_fragments).for_each(|addr| {
        mapping.insert(addr, None);
    });
    mapping
}

/// Remap a given index using the fragment reuse index if possible.
/// If the frag reuse index does not exist, the operation fails with [Error::NotSupported]
/// If the frag reuse index exists but is empty, the operation succeeds without a commit.
async fn remap_index(dataset: &mut Dataset, index_id: &Uuid) -> Result<()> {
    let indices = dataset.load_indices().await.unwrap();
    let frag_reuse_index_meta = match indices.iter().find(|idx| idx.name == FRAG_REUSE_INDEX_NAME) {
        None => Err(Error::NotSupported {
            source: "Fragment reuse index not found, cannot remap an index post compaction".into(),
            location: location!(),
        }),
        Some(frag_reuse_index_meta) => Ok(frag_reuse_index_meta),
    }?;

    let frag_reuse_details = load_frag_reuse_index_details(dataset, frag_reuse_index_meta)
        .await
        .unwrap();
    let frag_reuse_index =
        open_frag_reuse_index(frag_reuse_index_meta.uuid, frag_reuse_details.as_ref())
            .await
            .unwrap();

    if frag_reuse_index.row_id_maps.is_empty() {
        return Ok(());
    }

    // Sequentially apply the row addr maps from oldest to latest
    let mut curr_index_id = *index_id;
    for (i, row_id_map) in frag_reuse_index.row_id_maps.iter().enumerate() {
        let version = &frag_reuse_index.details.versions[i];
        // load on-disk index metadata before auto-remap
        let curr_index_meta = read_manifest_indexes(
            &dataset.object_store,
            &dataset.manifest_location,
            &dataset.manifest,
        )
        .await?
        .into_iter()
        .find(|idx| idx.uuid == curr_index_id)
        .unwrap();

        let maybe_index_bitmap = curr_index_meta.fragment_bitmap.clone();
        let (should_remap, bitmap_after_remap) = match maybe_index_bitmap {
            Some(mut index_frag_bitmap) => {
                let mut should_remap = false;
                for group in version.groups.iter() {
                    let mut old_frag_in_index = 0;
                    for old_frag in group.old_frags.iter() {
                        if index_frag_bitmap.remove(old_frag.id as u32) {
                            old_frag_in_index += 1;
                        }
                    }

                    if old_frag_in_index > 0 {
                        if old_frag_in_index != group.old_frags.len() {
                            // this should never happen because we always commit a full rewrite group
                            // and we always reindex either the entire group or nothing.
                            // We use invalid input to be consistent with
                            // dataset::transaction::recalculate_fragment_bitmap
                            return Err(Error::invalid_input(
                                format!("The compaction plan included a rewrite group that was a split of indexed and non-indexed data: {:?}", group.old_frags),
                                location!()));
                        }
                        index_frag_bitmap
                            .extend(group.new_frags.clone().into_iter().map(|f| f.id as u32));
                        should_remap = true;
                    }
                }
                (should_remap, Some(index_frag_bitmap))
            }
            // if there is no fragment bitmap for the index,
            // we attempt remapping but will not update the fragment bitmap.
            None => (true, None),
        };

        if should_remap {
            let remap_result = index::remap_index(dataset, &curr_index_id, row_id_map).await?;

            let new_index_meta = match remap_result {
                RemapResult::Drop => continue,
                RemapResult::Keep(new_id) => IndexMetadata {
                    uuid: new_id,
                    name: curr_index_meta.name.clone(),
                    fields: curr_index_meta.fields.clone(),
                    dataset_version: dataset.manifest.version,
                    fragment_bitmap: bitmap_after_remap,
                    index_details: curr_index_meta.index_details.clone(),
                    index_version: curr_index_meta.index_version,
                    created_at: curr_index_meta.created_at,
                    base_id: None,
                },
                RemapResult::Remapped(remapped_index) => IndexMetadata {
                    uuid: remapped_index.new_id,
                    name: curr_index_meta.name.clone(),
                    fields: curr_index_meta.fields.clone(),
                    dataset_version: dataset.manifest.version,
                    fragment_bitmap: bitmap_after_remap,
                    index_details: Some(Arc::new(remapped_index.index_details)),
                    index_version: remapped_index.index_version as i32,
                    created_at: curr_index_meta.created_at,
                    base_id: None,
                },
            };

            let new_id = new_index_meta.uuid;

            let transaction = Transaction::new(
                dataset.manifest.version,
                Operation::CreateIndex {
                    new_indices: vec![new_index_meta],
                    removed_indices: vec![curr_index_meta.clone()],
                },
                None,
            );

            dataset
                .apply_commit(transaction, &Default::default(), &Default::default())
                .await?;

            curr_index_id = new_id;
        }
    }

    Ok(())
}

pub async fn remap_column_index(
    dataset: &mut Dataset,
    columns: &[&str],
    name: Option<String>,
) -> Result<()> {
    if columns.len() != 1 {
        return Err(Error::Index {
            message: "Only support remapping index on 1 column at the moment".to_string(),
            location: location!(),
        });
    }

    let column = columns[0];
    let Some(field) = dataset.schema().field(column) else {
        return Err(Error::Index {
            message: format!("RemapIndex: column '{column}' does not exist"),
            location: location!(),
        });
    };

    let indices = dataset.load_indices().await?;
    let index_name = name.unwrap_or(format!("{column}_idx"));
    let index = match indices.iter().find(|i| i.name == index_name) {
        None => {
            return Err(Error::Index {
                message: format!("Index with name {} not found", index_name),
                location: location!(),
            });
        }
        Some(index) => {
            if index.fields != [field.id] {
                Err(Error::Index {
                    message: format!(
                        "Index name {} already exists with different fields",
                        index_name
                    ),
                    location: location!(),
                })
            } else {
                Ok(index)
            }
        }
    }?;

    remap_index(dataset, &index.uuid).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_indices() {
        // Sanity test to make sure MissingIds works.  Does not test actual functionality so
        // feel free to remove if it becomes inconvenient
        let frags = vec![
            FragDigest {
                id: 0,
                physical_rows: 5,
                num_deleted_rows: 0,
            },
            FragDigest {
                id: 3,
                physical_rows: 3,
                num_deleted_rows: 0,
            },
        ];
        let rows = [(0, 1), (0, 3), (0, 4), (3, 0), (3, 2)]
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into());

        let missing = MissingAddrs::new(rows, &frags).collect::<Vec<_>>();
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
            FragDigest {
                id: 0,
                physical_rows: 5,
                num_deleted_rows: 0,
            },
            FragDigest {
                id: 3,
                physical_rows: 3,
                num_deleted_rows: 0,
            },
            FragDigest {
                id: 1,
                physical_rows: 3,
                num_deleted_rows: 0,
            },
        ];

        // Written as pairs of (fragment_id, offset)
        let row_addrs = vec![
            (0, 1),
            (0, 3),
            (0, 4),
            (3, 0),
            (3, 2),
            (1, 0),
            (1, 1),
            (1, 2),
        ];
        let row_addrs = row_addrs
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into());
        let result = MissingAddrs::new(row_addrs, &fragments).collect::<Vec<_>>();

        let expected = vec![(0, 0), (0, 2), (3, 1)];
        let expected = expected
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into())
            .collect::<Vec<u64>>();
        assert_eq!(result, expected);
    }
}
