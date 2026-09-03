// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The stable partition system index: the ledger of reordered rewrites.
//!
//! A reordered rewrite (e.g. reclustering) scans `n` source fragments in
//! order and distributes their live rows across `m` destination fragments,
//! preserving relative row order within each destination. The mapping cannot
//! be derived from row order the way [`super::frag_reuse`] derives it for
//! compaction, so each rewrite records a [`StablePartitionTransition`]:
//! the ordered sources (whose physical row counts are the addressing base of
//! the row map), the ordered destinations (a row map label indexes this
//! list), and a reference to the row map file written with
//! `lance-index`'s `RowMapWriter`.
//!
//! The transitions accumulate in one manifest index entry named
//! [`STABLE_PARTITION_INDEX_NAME`], separate from the fragment reuse index so
//! readers that predate it skip the entry instead of misreading it. The
//! entry's fragment bitmap holds the union of all transition source ids:
//! provenance, deliberately including fragments no longer in the dataset
//! (see `IndexMetadata::fragment_bitmap`).

use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

use super::frag_reuse::FragDigest;
use crate::format::pb::stable_partition_index_details::InlineContent;
use crate::format::{ExternalFile, pb};

pub const STABLE_PARTITION_INDEX_NAME: &str = "__lance_stable_partition";
pub const STABLE_PARTITION_DETAILS_FILE_NAME: &str = "details.binpb";
/// The row map file of a transition, under `{indices dir}/{row_map_id}/`.
pub const ROW_MAP_FILE_NAME: &str = "row_map.lance";

/// One reordered rewrite.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct StablePartitionTransition {
    /// The dataset version whose commit installed this transition.
    pub dataset_version: u64,
    /// Source fragments in scan order; prefix sums of their physical rows
    /// position each fragment's rows in the row map's label sequence.
    pub sources: Vec<FragDigest>,
    /// Destination fragment ids in label order: a row map label is an index
    /// into this list.
    pub destinations: Vec<u64>,
    /// Directory id of the row map file: `{indices dir}/{row_map_id}/row_map.lance`.
    pub row_map_id: String,
    pub row_map_size_bytes: u64,
}

impl StablePartitionTransition {
    pub fn source_ids(&self) -> RoaringBitmap {
        RoaringBitmap::from_iter(self.sources.iter().map(|frag| frag.id as u32))
    }

    pub fn destination_ids(&self) -> RoaringBitmap {
        RoaringBitmap::from_iter(self.destinations.iter().map(|&id| id as u32))
    }

    /// Total physical source rows: the row count of the row map file.
    pub fn total_source_rows(&self) -> u64 {
        self.sources
            .iter()
            .map(|frag| frag.physical_rows as u64)
            .sum()
    }
}

impl From<&FragDigest> for pb::stable_partition_index_details::FragmentDigest {
    fn from(digest: &FragDigest) -> Self {
        Self {
            id: digest.id,
            physical_rows: digest.physical_rows as u64,
            num_deleted_rows: digest.num_deleted_rows as u64,
        }
    }
}

impl TryFrom<pb::stable_partition_index_details::FragmentDigest> for FragDigest {
    type Error = Error;

    fn try_from(digest: pb::stable_partition_index_details::FragmentDigest) -> Result<Self> {
        Ok(Self {
            id: digest.id,
            physical_rows: digest.physical_rows as usize,
            num_deleted_rows: digest.num_deleted_rows as usize,
        })
    }
}

impl From<&StablePartitionTransition> for pb::stable_partition_index_details::Transition {
    fn from(transition: &StablePartitionTransition) -> Self {
        Self {
            dataset_version: transition.dataset_version,
            sources: transition.sources.iter().map(|f| f.into()).collect(),
            destinations: transition.destinations.clone(),
            row_map_id: transition.row_map_id.clone(),
            row_map_size_bytes: transition.row_map_size_bytes,
        }
    }
}

impl TryFrom<pb::stable_partition_index_details::Transition> for StablePartitionTransition {
    type Error = Error;

    fn try_from(transition: pb::stable_partition_index_details::Transition) -> Result<Self> {
        Ok(Self {
            dataset_version: transition.dataset_version,
            sources: transition
                .sources
                .into_iter()
                .map(FragDigest::try_from)
                .collect::<Result<_>>()?,
            destinations: transition.destinations,
            row_map_id: transition.row_map_id,
            row_map_size_bytes: transition.row_map_size_bytes,
        })
    }
}

/// All transitions of the stable partition index entry, oldest first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct StablePartitionIndexDetails {
    pub transitions: Vec<StablePartitionTransition>,
}

impl StablePartitionIndexDetails {
    /// The union of every transition's source ids: the entry's fragment
    /// bitmap. Sources are retired at commit, so these ids are provenance,
    /// not live coverage.
    pub fn source_bitmap(&self) -> RoaringBitmap {
        self.transitions
            .iter()
            .fold(RoaringBitmap::new(), |mut bitmap, transition| {
                bitmap |= transition.source_ids();
                bitmap
            })
    }
}

impl From<&StablePartitionIndexDetails> for InlineContent {
    fn from(details: &StablePartitionIndexDetails) -> Self {
        let mut transitions: Vec<pb::stable_partition_index_details::Transition> =
            details.transitions.iter().map(|t| t.into()).collect();
        // sort from oldest to latest version
        transitions.sort_by_key(|t| t.dataset_version);
        Self { transitions }
    }
}

impl TryFrom<InlineContent> for StablePartitionIndexDetails {
    type Error = Error;

    fn try_from(content: InlineContent) -> Result<Self> {
        Ok(Self {
            transitions: content
                .transitions
                .into_iter()
                .map(|t| t.try_into())
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

/// The details payload of the manifest entry: inline below the 200KB spill
/// threshold, otherwise a reference to an external `details.binpb`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub enum StablePartitionDetailsContentType {
    Inline(StablePartitionIndexDetails),
    External(ExternalFile),
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    fn sample_details() -> StablePartitionIndexDetails {
        StablePartitionIndexDetails {
            transitions: vec![StablePartitionTransition {
                dataset_version: 7,
                sources: vec![
                    FragDigest {
                        id: 1,
                        physical_rows: 100,
                        num_deleted_rows: 10,
                    },
                    FragDigest {
                        id: 2,
                        physical_rows: 50,
                        num_deleted_rows: 0,
                    },
                ],
                destinations: vec![10, 11, 12],
                row_map_id: "d4c9…".to_string(),
                row_map_size_bytes: 12345,
            }],
        }
    }

    #[test]
    fn test_details_proto_round_trip() {
        let details = sample_details();
        let content = InlineContent::from(&details);
        let decoded = StablePartitionIndexDetails::try_from(
            InlineContent::decode(content.encode_to_vec().as_slice()).unwrap(),
        )
        .unwrap();
        assert_eq!(decoded, details);

        let transition = &details.transitions[0];
        assert_eq!(transition.total_source_rows(), 150);
        assert_eq!(transition.source_ids(), RoaringBitmap::from_iter([1u32, 2]));
        assert_eq!(
            transition.destination_ids(),
            RoaringBitmap::from_iter([10u32, 11, 12])
        );
        assert_eq!(details.source_bitmap(), RoaringBitmap::from_iter([1u32, 2]));
    }
}
