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
    pub sealed: bool,
    pub flushed: bool,
}

impl From<&MemWal> for pb::mem_wal_index_details::MemWal {
    fn from(mem_wal: &MemWal) -> Self {
        Self {
            id: Some(pb::mem_wal_index_details::MemWalId::from(&mem_wal.id)),
            mem_table_location: mem_wal.mem_table_location.clone(),
            wal_location: mem_wal.wal_location.clone(),
            wal_entries: mem_wal.wal_entries.clone(),
            sealed: mem_wal.sealed,
            flushed: mem_wal.flushed,
        }
    }
}

impl TryFrom<pb::mem_wal_index_details::MemWal> for MemWal {
    type Error = Error;

    fn try_from(mem_wal: pb::mem_wal_index_details::MemWal) -> lance_core::Result<Self> {
        Ok(Self {
            id: MemWalId::try_from(mem_wal.id.unwrap())?,
            mem_table_location: mem_wal.mem_table_location.clone(),
            wal_location: mem_wal.wal_location.clone(),
            wal_entries: mem_wal.wal_entries.clone(),
            sealed: mem_wal.sealed,
            flushed: mem_wal.flushed,
        })
    }
}

impl MemWal {
    pub fn new_empty(id: MemWalId, mem_table_location: &str, wal_location: &str) -> Self {
        Self {
            id,
            mem_table_location: mem_table_location.to_owned(),
            wal_location: wal_location.to_owned(),
            wal_entries: pb::U64Segment::from(U64Segment::Range(0..0)).encode_to_vec(),
            sealed: false,
            flushed: false,
        }
    }

    pub fn wal_entries(&self) -> U64Segment {
        U64Segment::try_from(pb::U64Segment::decode(self.wal_entries.as_slice()).unwrap()).unwrap()
    }

    pub fn check_sealed(&self, expected: bool) -> lance_core::Result<()> {
        if self.sealed != expected {
            return Err(Error::invalid_input(
                format!(
                    "MemWAL {:?} is sealed: {}, but expected {}",
                    self.id, self.sealed, expected
                ),
                location!(),
            ));
        }
        Ok(())
    }

    pub fn check_flushed(&self, expected: bool) -> lance_core::Result<()> {
        if self.flushed != expected {
            return Err(Error::invalid_input(
                format!(
                    "MemWAL {:?} is flushed: {}, but expected {}",
                    self.id, self.flushed, expected
                ),
                location!(),
            ));
        }
        Ok(())
    }

    pub fn check_expected_mem_table_location(&self, expected: &str) -> lance_core::Result<()> {
        if self.mem_table_location != expected {
            return Err(Error::invalid_input(
                format!(
                    "MemWAL {:?} has MemTable location: {}, but expected {}",
                    self.id, self.mem_table_location, expected
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
    num_mem_wal: usize,
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
            num_mem_wal: self.mem_wal_map.values().map(|m| m.len()).sum(),
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
