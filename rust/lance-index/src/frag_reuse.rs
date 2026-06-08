// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `Index`-trait adapter for the fragment-reuse system index.
//!
//! The data structures and table-format logic live in
//! [`lance_table::system_index::frag_reuse`]; this module re-exports them and
//! implements the local [`Index`] trait for [`FragReuseIndex`].

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, ArrayRef, PrimitiveArray, RecordBatch, UInt64Array};
use std::any::Any;
use std::sync::Arc;
use async_trait::async_trait;
use lance_core::{Error, Result};
use lance_index_core::row_id_remap::RowIdRemapper;
use lance_select::RowAddrTreeMap;
use roaring::{RoaringBitmap, RoaringTreemap};
use serde::Serialize;

pub use lance_table::system_index::frag_reuse::*;

use crate::{Index, IndexType};

impl RowIdRemapper for FragReuseIndex {
    fn remap_row_id(&self, row_id: u64) -> Option<u64> {
        Self::remap_row_id(self, row_id)
    }

    fn remap_row_addrs_tree_map(&self, row_addrs: &RowAddrTreeMap) -> RowAddrTreeMap {
        Self::remap_row_addrs_tree_map(self, row_addrs)
    }

    fn remap_row_ids_roaring_tree_map(&self, row_ids: &RoaringTreemap) -> RoaringTreemap {
        Self::remap_row_ids_roaring_tree_map(self, row_ids)
    }

    fn remap_row_ids_record_batch(
        &self,
        batch: RecordBatch,
        row_id_idx: usize,
    ) -> Result<RecordBatch> {
        Self::remap_row_ids_record_batch(self, batch, row_id_idx)
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

    fn statistics(&self) -> Result<serde_json::Value> {
        let stats = FragReuseStatistics {
            num_versions: self.details.versions.len(),
        };
        serde_json::to_value(stats).map_err(|e| {
            Error::internal(format!(
                "failed to serialize fragment reuse index statistics: {}",
                e
            ))
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
mod tests {
    use lance_table::format::pb::fragment_reuse_index_details::InlineContent;

    use super::*;

    #[tokio::test]
    async fn test_serialize_deserialize_index_details() {
        // Create sample FragReuseVersions with different dataset versions
        let version1 = FragReuseVersion {
            dataset_version: 2,
            groups: vec![FragReuseGroup {
                changed_row_addrs: vec![1, 2, 3],
                old_frags: vec![FragDigest {
                    id: 1,
                    physical_rows: 1,
                    num_deleted_rows: 0,
                }],
                new_frags: vec![
                    FragDigest {
                        id: 2,
                        physical_rows: 1,
                        num_deleted_rows: 0,
                    },
                    FragDigest {
                        id: 3,
                        physical_rows: 1,
                        num_deleted_rows: 0,
                    },
                ],
            }],
        };

        let version2 = FragReuseVersion {
            dataset_version: 1,
            groups: vec![FragReuseGroup {
                changed_row_addrs: vec![4, 5, 6],
                old_frags: vec![FragDigest {
                    id: 2,
                    physical_rows: 1,
                    num_deleted_rows: 0,
                }],
                new_frags: vec![
                    FragDigest {
                        id: 4,
                        physical_rows: 1,
                        num_deleted_rows: 0,
                    },
                    FragDigest {
                        id: 5,
                        physical_rows: 1,
                        num_deleted_rows: 0,
                    },
                ],
            }],
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
            roundtrip_details.versions[0].groups[0].changed_row_addrs,
            vec![4, 5, 6]
        );
        assert_eq!(
            roundtrip_details.versions[0].groups[0].new_frags,
            vec![
                FragDigest {
                    id: 4,
                    physical_rows: 1,
                    num_deleted_rows: 0,
                },
                FragDigest {
                    id: 5,
                    physical_rows: 1,
                    num_deleted_rows: 0,
                }
            ]
        );
        assert_eq!(
            roundtrip_details.versions[0].groups[0].old_frags,
            vec![FragDigest {
                id: 2,
                physical_rows: 1,
                num_deleted_rows: 0,
            }]
        );

        assert_eq!(roundtrip_details.versions[1].dataset_version, 2);
        assert_eq!(
            roundtrip_details.versions[1].groups[0].changed_row_addrs,
            vec![1, 2, 3]
        );
        assert_eq!(
            roundtrip_details.versions[1].groups[0].new_frags,
            vec![
                FragDigest {
                    id: 2,
                    physical_rows: 1,
                    num_deleted_rows: 0,
                },
                FragDigest {
                    id: 3,
                    physical_rows: 1,
                    num_deleted_rows: 0,
                }
            ]
        );
        assert_eq!(
            roundtrip_details.versions[1].groups[0].old_frags,
            vec![FragDigest {
                id: 1,
                physical_rows: 1,
                num_deleted_rows: 0,
            }]
        );
    }
}
