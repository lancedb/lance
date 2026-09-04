// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Manifest references to immutable stable-partition row maps.

use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;
use uuid::Uuid;

use super::pb;
use crate::system_index::frag_reuse::FragDigest;

/// Managed subtree for immutable stable-partition mappings.
pub const ROW_MAPS_DIR: &str = "_row_maps";

/// File containing destination labels and cumulative counts.
pub const ROW_MAP_FILE_NAME: &str = "row_map.lance";

/// A value-preserving rewrite whose destinations retain source order.
///
/// Sources and destinations are ordered: labels address the concatenated
/// physical sources and index the destination list. Row maps are immutable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StablePartitionTransition {
    /// Snapshot scanned by the rewrite job.
    pub source_dataset_version: u64,
    /// Sources in physical scan order, including deleted positions.
    pub sources: Vec<FragDigest>,
    /// Destinations in label order, without deletions at creation.
    pub destinations: Vec<FragDigest>,
    /// Directory under `_row_maps` containing [`ROW_MAP_FILE_NAME`].
    pub row_map_id: Uuid,
    /// Complete immutable row-map file length.
    pub row_map_size_bytes: u64,
    /// Storage base, or this dataset's base when absent.
    pub base_id: Option<u32>,
    /// Installing manifest version; zero while preparing the transaction.
    pub committed_version: u64,
}

impl DeepSizeOf for StablePartitionTransition {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.sources.deep_size_of_children(context)
            + self.destinations.deep_size_of_children(context)
    }
}

impl StablePartitionTransition {
    /// Source fragment membership, independent of source scan order.
    pub fn source_ids(&self) -> RoaringBitmap {
        self.sources.iter().map(|f| f.id as u32).collect()
    }

    /// Destination fragment membership, independent of label order.
    pub fn destination_ids(&self) -> RoaringBitmap {
        self.destinations.iter().map(|f| f.id as u32).collect()
    }

    /// Validate address bounds and uniqueness before writing or reading metadata.
    pub fn validate(&self) -> Result<()> {
        if self.sources.is_empty()
            || self.destinations.is_empty()
            || self.destinations.len() > 65536
            || self.row_map_size_bytes == 0
        {
            return Err(Error::invalid_input(
                "stable partition requires nonempty sources, 1..=65536 destinations, and a nonempty row map",
            ));
        }
        let mut ids = RoaringBitmap::new();
        for fragment in self.sources.iter().chain(&self.destinations) {
            let id = u32::try_from(fragment.id).map_err(|_| {
                Error::invalid_input(format!("fragment id {} exceeds u32", fragment.id))
            })?;
            if !ids.insert(id)
                || fragment.physical_rows > u32::MAX as usize
                || fragment.num_deleted_rows > fragment.physical_rows
            {
                return Err(Error::invalid_input(format!(
                    "invalid or repeated stable-partition fragment {}: physical_rows={}, deleted_rows={}",
                    fragment.id, fragment.physical_rows, fragment.num_deleted_rows
                )));
            }
        }
        if self
            .destinations
            .iter()
            .any(|f| f.id == 0 || f.num_deleted_rows != 0)
        {
            return Err(Error::invalid_input(
                "stable-partition destinations require reserved nonzero IDs and no deleted rows",
            ));
        }
        let live: u64 = self
            .sources
            .iter()
            .map(|f| (f.physical_rows - f.num_deleted_rows) as u64)
            .sum();
        let written: u64 = self
            .destinations
            .iter()
            .map(|f| f.physical_rows as u64)
            .sum();
        if live != written {
            return Err(Error::invalid_input(format!(
                "stable-partition source live rows {live} differ from destination rows {written}"
            )));
        }
        Ok(())
    }
}

impl From<&StablePartitionTransition> for pb::StablePartitionTransition {
    fn from(t: &StablePartitionTransition) -> Self {
        Self {
            source_dataset_version: t.source_dataset_version,
            sources: t.sources.iter().map(Into::into).collect(),
            destinations: t.destinations.iter().map(Into::into).collect(),
            row_map_id: t.row_map_id.to_string(),
            row_map_size_bytes: t.row_map_size_bytes,
            base_id: t.base_id,
            committed_version: t.committed_version,
        }
    }
}

impl TryFrom<pb::StablePartitionTransition> for StablePartitionTransition {
    type Error = Error;
    fn try_from(t: pb::StablePartitionTransition) -> Result<Self> {
        let transition = Self {
            source_dataset_version: t.source_dataset_version,
            sources: t
                .sources
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_>>()?,
            destinations: t
                .destinations
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_>>()?,
            row_map_id: Uuid::parse_str(&t.row_map_id).map_err(|e| {
                Error::invalid_input(format!("invalid row_map_id {}: {e}", t.row_map_id))
            })?,
            row_map_size_bytes: t.row_map_size_bytes,
            base_id: t.base_id,
            committed_version: t.committed_version,
        };
        transition.validate()?;
        Ok(transition)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    fn transition() -> StablePartitionTransition {
        StablePartitionTransition {
            source_dataset_version: 4,
            sources: vec![FragDigest {
                id: 0,
                physical_rows: 3,
                num_deleted_rows: 1,
            }],
            destinations: vec![FragDigest {
                id: 5,
                physical_rows: 2,
                num_deleted_rows: 0,
            }],
            row_map_id: Uuid::new_v4(),
            row_map_size_bytes: 1024,
            base_id: None,
            committed_version: 6,
        }
    }

    #[test]
    fn protobuf_preserves_mapping_identity_and_order() {
        let transition = transition();
        let bytes = pb::StablePartitionTransition::from(&transition).encode_to_vec();
        let decoded = pb::StablePartitionTransition::decode(bytes.as_slice()).unwrap();
        assert_eq!(
            StablePartitionTransition::try_from(decoded).unwrap(),
            transition
        );
    }

    #[test]
    fn transaction_round_trip_preserves_the_partition_descriptor() {
        use crate::format::Fragment;
        use crate::transaction::{Operation, RewriteGroup, Transaction};

        let mut descriptor = transition();
        descriptor.committed_version = 0;
        let mut source = Fragment::new(0);
        source.physical_rows = Some(3);
        let mut destination = Fragment::new(5);
        destination.physical_rows = Some(2);
        let transaction = Transaction::new(
            4,
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments: vec![source],
                    new_fragments: vec![destination],
                }],
                rewritten_indices: Vec::new(),
                frag_reuse_index: None,
                stable_partition: Some(Box::new(descriptor.clone())),
            },
            None,
        );
        let bytes = pb::Transaction::from(&transaction).encode_to_vec();
        let decoded =
            Transaction::try_from(pb::Transaction::decode(bytes.as_slice()).unwrap()).unwrap();
        let Operation::Rewrite {
            stable_partition, ..
        } = decoded.operation
        else {
            panic!("expected rewrite")
        };
        assert_eq!(*stable_partition.unwrap(), descriptor);
    }

    #[rstest::rstest]
    #[case::duplicate_source(0, 2, "repeated")]
    #[case::address_overflow(u64::from(u32::MAX) + 1, 2, "exceeds u32")]
    #[case::row_count_mismatch(5, 1, "source live rows")]
    #[test]
    fn invalid_destinations_are_rejected(
        #[case] id: u64,
        #[case] rows: usize,
        #[case] message: &str,
    ) {
        let mut transition = transition();
        transition.destinations[0].id = id;
        transition.destinations[0].physical_rows = rows;
        let error = transition.validate().unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(message), "{error}");
    }
}
