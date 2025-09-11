// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::{Index, IndexType};
use async_trait::async_trait;
use lance_core::cache::DeepSizeOf;
use lance_core::Error;
use lance_table::format::pb;
use lance_table::rowids::segment::U64Segment;
use prost::Message;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use snafu::location;
use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

pub const MEM_WAL_INDEX_NAME: &str = "__lance_mem_wal";

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Serialize, Deserialize, DeepSizeOf)]
pub enum State {
    Open,
    Sealed,
    Flushed,
    Merged,
}

impl From<State> for pb::mem_wal_index_details::mem_wal::State {
    fn from(state: State) -> Self {
        match state {
            State::Open => Self::Open,
            State::Sealed => Self::Sealed,
            State::Flushed => Self::Flushed,
            State::Merged => Self::Merged,
        }
    }
}

impl TryFrom<pb::mem_wal_index_details::mem_wal::State> for State {
    type Error = Error;

    fn try_from(state: pb::mem_wal_index_details::mem_wal::State) -> lance_core::Result<Self> {
        match state {
            pb::mem_wal_index_details::mem_wal::State::Open => Ok(Self::Open),
            pb::mem_wal_index_details::mem_wal::State::Sealed => Ok(Self::Sealed),
            pb::mem_wal_index_details::mem_wal::State::Flushed => Ok(Self::Flushed),
            pb::mem_wal_index_details::mem_wal::State::Merged => Ok(Self::Merged),
        }
    }
}

impl TryFrom<i32> for State {
    type Error = Error;

    fn try_from(value: i32) -> lance_core::Result<Self> {
        match value {
            0 => Ok(Self::Open),
            1 => Ok(Self::Sealed),
            2 => Ok(Self::Flushed),
            3 => Ok(Self::Merged),
            _ => Err(Error::invalid_input(
                format!("Unknown MemWAL state value: {}", value),
                location!(),
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Serialize, Deserialize, DeepSizeOf)]
pub struct MemWalId {
    pub region: String,
    pub generation: u64,
}

impl From<&MemWalId> for pb::mem_wal_index_details::MemWalId {
    fn from(mem_wal: &MemWalId) -> Self {
        Self {
            region: mem_wal.region.clone(),
            generation: mem_wal.generation,
        }
    }
}

impl TryFrom<pb::mem_wal_index_details::MemWalId> for MemWalId {
    type Error = Error;

    fn try_from(mem_wal: pb::mem_wal_index_details::MemWalId) -> lance_core::Result<Self> {
        Ok(Self {
            region: mem_wal.region.clone(),
            generation: mem_wal.generation,
        })
    }
}

impl MemWalId {
    pub fn new(region: &str, generation: u64) -> Self {
        Self {
            region: region.to_owned(),
            generation,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Serialize, Deserialize, DeepSizeOf)]
pub struct MemWal {
    pub id: MemWalId,
    pub mem_table_location: String,
    pub wal_location: String,
    pub wal_entries: Vec<u8>,
    pub state: State,
    pub owner_id: String,
    pub last_updated_dataset_version: u64,
}

impl From<&MemWal> for pb::mem_wal_index_details::MemWal {
    fn from(mem_wal: &MemWal) -> Self {
        Self {
            id: Some(pb::mem_wal_index_details::MemWalId::from(&mem_wal.id)),
            mem_table_location: mem_wal.mem_table_location.clone(),
            wal_location: mem_wal.wal_location.clone(),
            wal_entries: mem_wal.wal_entries.clone(),
            state: pb::mem_wal_index_details::mem_wal::State::from(mem_wal.state.clone()) as i32,
            owner_id: mem_wal.owner_id.clone(),
            last_updated_dataset_version: mem_wal.last_updated_dataset_version,
        }
    }
}

impl TryFrom<pb::mem_wal_index_details::MemWal> for MemWal {
    type Error = Error;

    fn try_from(mem_wal: pb::mem_wal_index_details::MemWal) -> lance_core::Result<Self> {
        let state = State::try_from(mem_wal.state)?;

        Ok(Self {
            id: MemWalId::try_from(mem_wal.id.unwrap())?,
            mem_table_location: mem_wal.mem_table_location.clone(),
            wal_location: mem_wal.wal_location.clone(),
            wal_entries: mem_wal.wal_entries,
            state,
            owner_id: mem_wal.owner_id,
            last_updated_dataset_version: mem_wal.last_updated_dataset_version,
        })
    }
}

impl MemWal {
    pub fn new_empty(
        id: MemWalId,
        mem_table_location: &str,
        wal_location: &str,
        owner_id: &str,
    ) -> Self {
        Self {
            id,
            mem_table_location: mem_table_location.to_owned(),
            wal_location: wal_location.to_owned(),
            wal_entries: pb::U64Segment::from(U64Segment::Range(0..0)).encode_to_vec(),
            state: State::Open,
            owner_id: owner_id.to_owned(),
            last_updated_dataset_version: 0, // placeholder, this will be filled during build_manifest
        }
    }

    pub fn wal_entries(&self) -> U64Segment {
        U64Segment::try_from(pb::U64Segment::decode(self.wal_entries.as_slice()).unwrap()).unwrap()
    }

    /// Check if the MemWAL is in the expected state
    pub fn check_state(&self, expected: State) -> lance_core::Result<()> {
        if self.state != expected {
            return Err(Error::invalid_input(
                format!(
                    "MemWAL {:?} is in state {:?}, but expected {:?}",
                    self.id, self.state, expected
                ),
                location!(),
            ));
        }
        Ok(())
    }

    pub fn check_expected_owner_id(&self, expected: &str) -> lance_core::Result<()> {
        if self.owner_id != expected {
            return Err(Error::invalid_input(
                format!(
                    "MemWAL {:?} has owner_id: {}, but expected {}",
                    self.id, self.owner_id, expected
                ),
                location!(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct MemWalIndexDetails {
    pub mem_wal_list: Vec<MemWal>,
}

impl From<&MemWalIndexDetails> for pb::MemWalIndexDetails {
    fn from(details: &MemWalIndexDetails) -> Self {
        Self {
            mem_wal_list: details.mem_wal_list.iter().map(|m| m.into()).collect(),
        }
    }
}

impl TryFrom<pb::MemWalIndexDetails> for MemWalIndexDetails {
    type Error = Error;

    fn try_from(details: pb::MemWalIndexDetails) -> lance_core::Result<Self> {
        Ok(Self {
            mem_wal_list: details
                .mem_wal_list
                .into_iter()
                .map(MemWal::try_from)
                .collect::<lance_core::Result<_>>()?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct MemWalIndex {
    pub mem_wal_map: HashMap<String, BTreeMap<u64, MemWal>>,
}

impl MemWalIndex {
    pub fn new(details: MemWalIndexDetails) -> Self {
        let mut mem_wal_map: HashMap<String, BTreeMap<u64, MemWal>> = HashMap::new();
        for mem_wal in details.mem_wal_list.into_iter() {
            if let Some(generations) = mem_wal_map.get_mut(&mem_wal.id.region) {
                generations.insert(mem_wal.id.generation, mem_wal);
            } else {
                mem_wal_map.insert(
                    mem_wal.id.region.clone(),
                    std::iter::once((mem_wal.id.generation, mem_wal)).collect(),
                );
            }
        }

        Self { mem_wal_map }
    }
}

#[derive(Serialize)]
struct MemWalStatistics {
    num_mem_wal: u64,
    num_regions: u64,
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
            source: "FragReuseIndex is not a vector index".into(),
            location: location!(),
        })
    }

    fn statistics(&self) -> lance_core::Result<serde_json::Value> {
        let stats = MemWalStatistics {
            num_mem_wal: self.mem_wal_map.values().map(|m| m.len()).sum::<usize>() as u64,
            num_regions: self.mem_wal_map.len() as u64,
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
        unimplemented!()
    }
}
