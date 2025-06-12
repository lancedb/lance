// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::{Index, IndexType};
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, ArrayRef, PrimitiveArray, RecordBatch, UInt32Array, UInt64Array};
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use itertools::Itertools;
use lance_core::utils::mask::RowIdTreeMap;
use lance_core::{Error, Result};
use lance_table::format::pb::fragment_reuse_index_details::InlineContent;
use lance_table::format::{pb, ExternalFile, Fragment};
use roaring::{RoaringBitmap, RoaringTreemap};
use serde::{Deserialize, Serialize};
use snafu::location;
use std::{any::Any, collections::HashMap, sync::Arc};

pub const FRAG_REUSE_INDEX_NAME: &str = "__lance_frag_reuse";
pub const FRAG_REUSE_DETAILS_FILE_NAME: &str = "details.binpb";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct FragReuseVersion {
    pub dataset_version: u64,
    pub changed_row_addrs: Vec<u8>,
    pub old_frags: Vec<Fragment>,
    pub new_frags: Vec<u64>,
}

impl From<&FragReuseVersion> for pb::fragment_reuse_index_details::Version {
    fn from(version: &FragReuseVersion) -> Self {
        Self {
            dataset_version: version.dataset_version,
            changed_row_addrs: version.changed_row_addrs.clone(),
            old_fragments: version
                .old_frags
                .iter()
                .map(pb::DataFragment::from)
                .collect::<Vec<_>>(),
            new_fragments: version.new_frags.clone(),
        }
    }
}

impl TryFrom<pb::fragment_reuse_index_details::Version> for FragReuseVersion {
    type Error = Error;

    fn try_from(version: pb::fragment_reuse_index_details::Version) -> Result<Self> {
        Ok(Self {
            dataset_version: version.dataset_version,
            changed_row_addrs: version.changed_row_addrs,
            old_frags: version
                .old_fragments
                .into_iter()
                .map(Fragment::try_from)
                .collect::<Result<_>>()?,
            new_frags: version.new_fragments,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub enum FragReuseIndexDetailsContentType {
    Inline(FragReuseIndexDetails),
    External(ExternalFile),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct FragReuseIndexDetails {
    pub versions: Vec<FragReuseVersion>,
}

impl From<&FragReuseIndexDetails> for InlineContent {
    fn from(details: &FragReuseIndexDetails) -> Self {
        Self {
            versions: details
                .versions
                .iter()
                .map(|m| m.into())
                // sort from oldest to latest version
                .sorted_by_key(|v: &pb::fragment_reuse_index_details::Version| v.dataset_version)
                .collect(),
        }
    }
}

impl TryFrom<InlineContent> for FragReuseIndexDetails {
    type Error = Error;

    fn try_from(content: InlineContent) -> Result<Self> {
        Ok(Self {
            versions: content
                .versions
                .into_iter()
                .map(|m| m.try_into())
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

/// An index that stores row ID maps.
/// A row ID map describes the mapping from old row address to new address after compactions.
/// Each version contains the mapping for one round of compaction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct FragReuseIndex {
    pub row_id_maps: Vec<HashMap<u64, Option<u64>>>,
    pub details: FragReuseIndexDetails,
}

impl FragReuseIndex {
    pub fn new(
        row_id_maps: Vec<HashMap<u64, Option<u64>>>,
        details: FragReuseIndexDetails,
    ) -> Self {
        Self {
            row_id_maps,
            details,
        }
    }

    pub fn remap_row_id(&self, row_id: u64) -> Option<u64> {
        let mut mapped_value = Some(row_id);
        for row_id_map in self.row_id_maps.iter() {
            if mapped_value.is_some() {
                mapped_value = row_id_map
                    .get(&mapped_value.unwrap())
                    .copied()
                    .unwrap_or(mapped_value);
            }
        }

        mapped_value
    }

    pub fn remap_row_ids_tree_map(&self, row_ids: &RowIdTreeMap) -> RowIdTreeMap {
        RowIdTreeMap::from_iter(row_ids.row_ids().unwrap().filter_map(|addr| {
            let addr_as_u64 = u64::from(addr);
            self.remap_row_id(addr_as_u64)
        }))
    }

    pub fn remap_row_ids_roaring_tree_map(&self, row_ids: &RoaringTreemap) -> RoaringTreemap {
        RoaringTreemap::from_iter(row_ids.iter().filter_map(|addr| self.remap_row_id(addr)))
    }

    pub fn remap_row_ids_record_batch(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let row_ids = batch.column(1).as_primitive::<UInt64Type>();
        let val_idx_and_new_id = row_ids
            .values()
            .iter()
            .enumerate()
            .filter_map(|(idx, old_id)| self.remap_row_id(*old_id).map(|new_id| (idx, new_id)))
            .collect::<Vec<_>>();
        let new_ids = Arc::new(UInt64Array::from_iter_values(
            val_idx_and_new_id.iter().copied().map(|(_, new_id)| new_id),
        ));
        let new_val_indices = UInt64Array::from_iter_values(
            val_idx_and_new_id
                .into_iter()
                .map(|(val_idx, _)| val_idx as u64),
        );
        let new_vals = arrow_select::take::take(batch.column(0), &new_val_indices, None)?;
        Ok(RecordBatch::try_new(
            batch.schema(),
            vec![new_vals, new_ids],
        )?)
    }

    pub fn remap_row_ids_array(&self, array: ArrayRef) -> PrimitiveArray<UInt64Type> {
        let primitive_array = array
            .as_any()
            .downcast_ref::<PrimitiveArray<UInt64Type>>()
            .expect("expected row IDs to be uint64 array");
        (0..primitive_array.len())
            .map(|i| {
                if primitive_array.is_null(i) {
                    None
                } else {
                    self.remap_row_id(primitive_array.value(i))
                }
            })
            .collect()
    }

    /// Remap a record batch which has schema (row_id, vector)
    /// The vector column can be of any type
    pub fn remap_row_ids_vector_batch(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let mut indices = Vec::with_capacity(batch.num_rows());
        let mut new_row_ids = Vec::with_capacity(batch.num_rows());

        let row_ids = batch.column(0).as_primitive::<UInt64Type>().values();
        for (i, row_id) in row_ids.iter().enumerate() {
            if let Some(mapped_value) = self.remap_row_id(*row_id) {
                indices.push(i as u32);
                new_row_ids.push(mapped_value);
            }
        }

        let indices = UInt32Array::from(indices);
        let new_row_ids = Arc::new(UInt64Array::from(new_row_ids));
        let new_vectors = arrow::compute::take(batch.column(1), &indices, None)?;

        Ok(RecordBatch::try_new(
            batch.schema(),
            vec![new_row_ids, new_vectors],
        )?)
    }

    pub fn remap_fragment_bitmap(&self, fragment_bitmap: &mut RoaringBitmap) -> Result<()> {
        for version in self.details.versions.iter() {
            if fragment_bitmap.contains(version.old_frags[0].id as u32) {
                let mut removed = 0;
                for old_frag in version.old_frags.iter() {
                    if fragment_bitmap.remove(old_frag.id as u32) {
                        removed += 1;
                    }
                }

                if removed > 0 {
                    if removed != version.old_frags.len() {
                        // This should never happen because we always commit a full rewrite group
                        // and we always reindex either the entire group or nothing.
                        // We use invalid input to be consistent with
                        // dataset::transaction::recalculate_fragment_bitmap
                        return Err(Error::invalid_input(
                            format!("The compaction plan included a rewrite group that was a split of indexed and non-indexed data: {:?}",
                                    version.old_frags.iter().map(|frag| frag.id).collect::<Vec<_>>()),
                            location!()));
                    }

                    for new_frag in version.new_frags.iter() {
                        fragment_bitmap.insert(*new_frag as u32);
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct FragReuseStatistics {
    num_versions: usize,
}

#[async_trait]
impl Index for FragReuseIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(Error::NotSupported {
            source: "FragReuseIndex is not a vector index".into(),
            location: location!(),
        })
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let stats = FragReuseStatistics {
            num_versions: self.details.versions.len(),
        };
        serde_json::to_value(stats).map_err(|e| Error::Internal {
            message: format!("failed to serialize fragment reuse index statistics: {}", e),
            location: location!(),
        })
    }

    async fn prewarm(&self) -> Result<()> {
        Ok(())
    }

    fn index_type(&self) -> IndexType {
        IndexType::FragmentReuse
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!()
    }
}

#[cfg(test)]
pub mod tests {

    use super::*;

    #[tokio::test]
    async fn test_serialize_deserialize_index_details() {
        // Create sample FragReuseVersions with different dataset versions
        let version1 = FragReuseVersion {
            dataset_version: 2,
            changed_row_addrs: vec![1, 2, 3],
            old_frags: vec![Fragment {
                id: 1,
                files: vec![],
                deletion_file: None,
                row_id_meta: None,
                physical_rows: None,
            }],
            new_frags: vec![2, 3],
        };

        let version2 = FragReuseVersion {
            dataset_version: 1,
            changed_row_addrs: vec![4, 5, 6],
            old_frags: vec![Fragment {
                id: 2,
                files: vec![],
                deletion_file: None,
                row_id_meta: None,
                physical_rows: None,
            }],
            new_frags: vec![4, 5],
        };

        // Create FragReuseIndexDetails with versions in reverse order
        let details = FragReuseIndexDetails {
            versions: vec![version1, version2],
        };

        // Convert to protobuf format
        let inline_content: InlineContent = (&details).into();

        // Convert back to FragReuseIndexDetails
        let roundtrip_details = FragReuseIndexDetails::try_from(inline_content).unwrap();

        // Verify the roundtrip
        assert_eq!(roundtrip_details.versions.len(), 2);

        // Verify versions are sorted by dataset_version (oldest to latest)
        assert_eq!(roundtrip_details.versions[0].dataset_version, 1);
        assert_eq!(
            roundtrip_details.versions[0].changed_row_addrs,
            vec![4, 5, 6]
        );
        assert_eq!(roundtrip_details.versions[0].new_frags, vec![4, 5]);

        assert_eq!(roundtrip_details.versions[1].dataset_version, 2);
        assert_eq!(
            roundtrip_details.versions[1].changed_row_addrs,
            vec![1, 2, 3]
        );
        assert_eq!(roundtrip_details.versions[1].new_frags, vec![2, 3]);
    }
}
