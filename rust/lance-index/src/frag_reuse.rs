// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::{Index, IndexType};
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use itertools::Itertools;
use lance_core::{Error, Result};
use lance_table::format::pb::fragment_reuse_index_details::InlineContent;
use lance_table::format::{pb, ExternalFile, Fragment};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use snafu::location;
use std::{any::Any, collections::HashMap, sync::Arc};

pub const FRAG_REUSE_INDEX_NAME: &str = "fragment-reuse-index";

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
    #[tokio::test]
    async fn test_serialize_deserialize_index_details() {}
}
