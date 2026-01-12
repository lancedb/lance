// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use deepsize::DeepSizeOf;
use lance_core::Error;
use lance_table::format::pb;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use snafu::location;
use uuid::Uuid;

use crate::{Index, IndexType};

pub const MEM_WAL_INDEX_NAME: &str = "__lance_mem_wal";

/// Region identifier (UUID v4).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegionId(pub Uuid);

impl PartialOrd for RegionId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RegionId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.as_bytes().cmp(other.0.as_bytes())
    }
}

impl DeepSizeOf for RegionId {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        0 // UUID is 16 bytes fixed size, no heap allocations
    }
}

impl RegionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

impl Default for RegionId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<&RegionId> for pb::Uuid {
    fn from(region_id: &RegionId) -> Self {
        Self {
            uuid: region_id.0.as_bytes().to_vec(),
        }
    }
}

impl TryFrom<&pb::Uuid> for RegionId {
    type Error = Error;

    fn try_from(uuid: &pb::Uuid) -> lance_core::Result<Self> {
        if uuid.uuid.len() != 16 {
            return Err(Error::invalid_input(
                format!("Invalid UUID length: {}", uuid.uuid.len()),
                location!(),
            ));
        }
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&uuid.uuid);
        Ok(Self(Uuid::from_bytes(bytes)))
    }
}

/// A flushed MemTable generation and its storage location.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct FlushedGeneration {
    pub generation: u64,
    pub path: String,
}

impl From<&FlushedGeneration> for pb::FlushedGeneration {
    fn from(fg: &FlushedGeneration) -> Self {
        Self {
            generation: fg.generation,
            path: fg.path.clone(),
        }
    }
}

impl From<pb::FlushedGeneration> for FlushedGeneration {
    fn from(fg: pb::FlushedGeneration) -> Self {
        Self {
            generation: fg.generation,
            path: fg.path,
        }
    }
}

/// A region's merged generation, used in MemWalIndexDetails.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash, Serialize, Deserialize, DeepSizeOf)]
pub struct MergedGeneration {
    pub region_id: RegionId,
    pub generation: u64,
}

impl MergedGeneration {
    pub fn new(region_id: RegionId, generation: u64) -> Self {
        Self {
            region_id,
            generation,
        }
    }
}

impl From<&MergedGeneration> for pb::MergedGeneration {
    fn from(mg: &MergedGeneration) -> Self {
        Self {
            region_id: Some(pb::Uuid::from(&mg.region_id)),
            generation: mg.generation,
        }
    }
}

impl TryFrom<pb::MergedGeneration> for MergedGeneration {
    type Error = Error;

    fn try_from(mg: pb::MergedGeneration) -> lance_core::Result<Self> {
        let region_id = mg.region_id.as_ref().ok_or_else(|| {
            Error::invalid_input("Missing region_id in MergedGeneration", location!())
        })?;
        Ok(Self {
            region_id: RegionId::try_from(region_id)?,
            generation: mg.generation,
        })
    }
}

/// Region manifest containing epoch-based fencing and WAL state.
/// Each region has exactly one active writer at any time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct RegionManifest {
    pub region_id: RegionId,
    pub version: u64,
    pub region_spec_id: u32,
    pub writer_epoch: u64,
    pub replay_after_wal_id: u64,
    pub wal_id_last_seen: u64,
    pub current_generation: u64,
    pub merged_generation: u64,
    pub flushed_generations: Vec<FlushedGeneration>,
}

impl From<&RegionManifest> for pb::RegionManifest {
    fn from(rm: &RegionManifest) -> Self {
        Self {
            region_id: Some(pb::Uuid::from(&rm.region_id)),
            version: rm.version,
            region_spec_id: rm.region_spec_id,
            writer_epoch: rm.writer_epoch,
            replay_after_wal_id: rm.replay_after_wal_id,
            wal_id_last_seen: rm.wal_id_last_seen,
            current_generation: rm.current_generation,
            merged_generation: rm.merged_generation,
            flushed_generations: rm.flushed_generations.iter().map(|fg| fg.into()).collect(),
        }
    }
}

impl TryFrom<pb::RegionManifest> for RegionManifest {
    type Error = Error;

    fn try_from(rm: pb::RegionManifest) -> lance_core::Result<Self> {
        let region_id = rm.region_id.as_ref().ok_or_else(|| {
            Error::invalid_input("Missing region_id in RegionManifest", location!())
        })?;
        Ok(Self {
            region_id: RegionId::try_from(region_id)?,
            version: rm.version,
            region_spec_id: rm.region_spec_id,
            writer_epoch: rm.writer_epoch,
            replay_after_wal_id: rm.replay_after_wal_id,
            wal_id_last_seen: rm.wal_id_last_seen,
            current_generation: rm.current_generation,
            merged_generation: rm.merged_generation,
            flushed_generations: rm
                .flushed_generations
                .into_iter()
                .map(FlushedGeneration::from)
                .collect(),
        })
    }
}

/// Region field definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct RegionField {
    pub field_id: String,
    pub source_ids: Vec<i32>,
    pub transform: Option<String>,
    pub expression: Option<String>,
    pub result_type: String,
    pub parameters: HashMap<String, String>,
}

impl From<&RegionField> for pb::RegionField {
    fn from(rf: &RegionField) -> Self {
        Self {
            field_id: rf.field_id.clone(),
            source_ids: rf.source_ids.clone(),
            transform: rf.transform.clone(),
            expression: rf.expression.clone(),
            result_type: rf.result_type.clone(),
            parameters: rf.parameters.clone(),
        }
    }
}

impl From<pb::RegionField> for RegionField {
    fn from(rf: pb::RegionField) -> Self {
        Self {
            field_id: rf.field_id,
            source_ids: rf.source_ids,
            transform: rf.transform,
            expression: rf.expression,
            result_type: rf.result_type,
            parameters: rf.parameters,
        }
    }
}

/// Region spec definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct RegionSpec {
    pub spec_id: u32,
    pub fields: Vec<RegionField>,
}

impl From<&RegionSpec> for pb::RegionSpec {
    fn from(rs: &RegionSpec) -> Self {
        Self {
            spec_id: rs.spec_id,
            fields: rs.fields.iter().map(|f| f.into()).collect(),
        }
    }
}

impl From<pb::RegionSpec> for RegionSpec {
    fn from(rs: pb::RegionSpec) -> Self {
        Self {
            spec_id: rs.spec_id,
            fields: rs.fields.into_iter().map(RegionField::from).collect(),
        }
    }
}

/// Index details for MemWAL Index, stored in IndexMetadata.index_details.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct MemWalIndexDetails {
    pub snapshot_timestamp: i64,
    pub num_regions: u32,
    pub inline_snapshots: Option<Vec<u8>>,
    pub region_specs: Vec<RegionSpec>,
    pub maintained_indexes: Vec<String>,
    pub merged_generations: Vec<MergedGeneration>,
}

impl From<&MemWalIndexDetails> for pb::MemWalIndexDetails {
    fn from(details: &MemWalIndexDetails) -> Self {
        Self {
            snapshot_timestamp: details.snapshot_timestamp,
            num_regions: details.num_regions,
            inline_snapshots: details.inline_snapshots.clone(),
            region_specs: details.region_specs.iter().map(|rs| rs.into()).collect(),
            maintained_indexes: details.maintained_indexes.clone(),
            merged_generations: details
                .merged_generations
                .iter()
                .map(|mg| mg.into())
                .collect(),
        }
    }
}

impl TryFrom<pb::MemWalIndexDetails> for MemWalIndexDetails {
    type Error = Error;

    fn try_from(details: pb::MemWalIndexDetails) -> lance_core::Result<Self> {
        Ok(Self {
            snapshot_timestamp: details.snapshot_timestamp,
            num_regions: details.num_regions,
            inline_snapshots: details.inline_snapshots,
            region_specs: details
                .region_specs
                .into_iter()
                .map(RegionSpec::from)
                .collect(),
            maintained_indexes: details.maintained_indexes,
            merged_generations: details
                .merged_generations
                .into_iter()
                .map(MergedGeneration::try_from)
                .collect::<lance_core::Result<_>>()?,
        })
    }
}

/// MemWAL Index provides access to MemWAL configuration and state.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct MemWalIndex {
    pub details: MemWalIndexDetails,
}

impl MemWalIndex {
    pub fn new(details: MemWalIndexDetails) -> Self {
        Self { details }
    }

    pub fn merged_generation_for_region(&self, region_id: &RegionId) -> Option<u64> {
        self.details
            .merged_generations
            .iter()
            .find(|mg| &mg.region_id == region_id)
            .map(|mg| mg.generation)
    }
}

#[derive(Serialize)]
struct MemWalStatistics {
    num_regions: u32,
    num_merged_generations: usize,
    num_region_specs: usize,
    num_maintained_indexes: usize,
}

#[async_trait]
impl Index for MemWalIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> lance_core::Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(Error::NotSupported {
            source: "MemWalIndex is not a vector index".into(),
            location: location!(),
        })
    }

    fn statistics(&self) -> lance_core::Result<serde_json::Value> {
        let stats = MemWalStatistics {
            num_regions: self.details.num_regions,
            num_merged_generations: self.details.merged_generations.len(),
            num_region_specs: self.details.region_specs.len(),
            num_maintained_indexes: self.details.maintained_indexes.len(),
        };
        serde_json::to_value(stats).map_err(|e| Error::Internal {
            message: format!("failed to serialize MemWAL index statistics: {}", e),
            location: location!(),
        })
    }

    async fn prewarm(&self) -> lance_core::Result<()> {
        Ok(())
    }

    fn index_type(&self) -> IndexType {
        IndexType::MemWal
    }

    async fn calculate_included_frags(&self) -> lance_core::Result<RoaringBitmap> {
        Ok(RoaringBitmap::new())
    }
}
