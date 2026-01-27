// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Data source types for LSM scanner.

use std::sync::Arc;

use arrow_schema::SchemaRef;
use uuid::Uuid;

use crate::dataset::mem_wal::write::{BatchStore, IndexStore};
use crate::dataset::Dataset;

/// Generation number in LSM tree.
///
/// The base table has generation 0. MemTables have positive integers
/// starting from 1, where higher numbers represent newer data.
///
/// Ordering: Higher generation = newer data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LsmGeneration(u64);

impl LsmGeneration {
    /// Generation for the base table (merged data).
    pub const BASE_TABLE: Self = Self(0);

    /// Create a generation for a MemTable.
    ///
    /// # Panics
    ///
    /// Panics if `gen` is 0, as generation 0 is reserved for the base table.
    pub fn memtable(gen: u64) -> Self {
        assert!(
            gen > 0,
            "MemTable generation must be >= 1 (0 is reserved for base table)"
        );
        Self(gen)
    }

    /// Get the raw u64 value.
    pub fn as_u64(&self) -> u64 {
        self.0
    }

    /// Check if this is the base table generation.
    pub fn is_base_table(&self) -> bool {
        self.0 == 0
    }
}

impl From<u64> for LsmGeneration {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl std::fmt::Display for LsmGeneration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_base_table() {
            write!(f, "base")
        } else {
            write!(f, "gen{}", self.0)
        }
    }
}

impl Default for LsmGeneration {
    fn default() -> Self {
        Self::BASE_TABLE
    }
}

/// A flushed generation with its storage path.
#[derive(Debug, Clone)]
pub struct FlushedGeneration {
    /// Generation number.
    pub generation: u64,
    /// Path to the flushed MemTable directory (relative to table root).
    pub path: String,
}

/// Snapshot of a region's state at a point in time.
///
/// This is read from the MemWAL index for eventual consistency,
/// or from region manifests directly for strong consistency.
#[derive(Debug, Clone)]
pub struct RegionSnapshot {
    /// Region UUID.
    pub region_id: Uuid,
    /// Region spec ID (0 if manual region).
    pub spec_id: u32,
    /// Current generation being written (next flush will be this generation).
    pub current_generation: u64,
    /// List of flushed generations and their paths.
    pub flushed_generations: Vec<FlushedGeneration>,
}

impl RegionSnapshot {
    /// Create a new region snapshot.
    pub fn new(region_id: Uuid) -> Self {
        Self {
            region_id,
            spec_id: 0,
            current_generation: 1,
            flushed_generations: Vec::new(),
        }
    }

    /// Set the spec ID.
    pub fn with_spec_id(mut self, spec_id: u32) -> Self {
        self.spec_id = spec_id;
        self
    }

    /// Set the current generation.
    pub fn with_current_generation(mut self, gen: u64) -> Self {
        self.current_generation = gen;
        self
    }

    /// Add a flushed generation.
    pub fn with_flushed_generation(mut self, generation: u64, path: String) -> Self {
        self.flushed_generations
            .push(FlushedGeneration { generation, path });
        self
    }
}

/// A data source in the LSM tree that can be scanned.
pub enum LsmDataSource {
    /// Base Lance table (generation = 0).
    BaseTable {
        /// The base dataset.
        dataset: Arc<Dataset>,
    },
    /// Flushed MemTable stored as Lance table on disk.
    FlushedMemTable {
        /// Absolute path to the flushed MemTable directory.
        path: String,
        /// Region this MemTable belongs to.
        region_id: Uuid,
        /// Generation number (1, 2, 3, ...).
        generation: LsmGeneration,
    },
    /// In-memory MemTable (active write buffer).
    ActiveMemTable {
        /// Batch store containing the data.
        batch_store: Arc<BatchStore>,
        /// Index store for the MemTable.
        index_store: Arc<IndexStore>,
        /// Schema of the data.
        schema: SchemaRef,
        /// Region this MemTable belongs to.
        region_id: Uuid,
        /// Generation number.
        generation: LsmGeneration,
    },
}

impl LsmDataSource {
    /// Get the generation of this data source.
    pub fn generation(&self) -> LsmGeneration {
        match self {
            Self::BaseTable { .. } => LsmGeneration::BASE_TABLE,
            Self::FlushedMemTable { generation, .. } => *generation,
            Self::ActiveMemTable { generation, .. } => *generation,
        }
    }

    /// Get the region ID if this is a regional source.
    pub fn region_id(&self) -> Option<Uuid> {
        match self {
            Self::BaseTable { .. } => None,
            Self::FlushedMemTable { region_id, .. } => Some(*region_id),
            Self::ActiveMemTable { region_id, .. } => Some(*region_id),
        }
    }

    /// Check if this is the base table.
    pub fn is_base_table(&self) -> bool {
        matches!(self, Self::BaseTable { .. })
    }

    /// Check if this is an active (in-memory) MemTable.
    pub fn is_active_memtable(&self) -> bool {
        matches!(self, Self::ActiveMemTable { .. })
    }

    /// Get a display name for logging.
    pub fn display_name(&self) -> String {
        match self {
            Self::BaseTable { .. } => "base_table".to_string(),
            Self::FlushedMemTable {
                region_id,
                generation,
                ..
            } => format!("flushed[{}:{}]", &region_id.to_string()[..8], generation),
            Self::ActiveMemTable {
                region_id,
                generation,
                ..
            } => format!("memtable[{}:{}]", &region_id.to_string()[..8], generation),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsm_generation_ordering() {
        let base = LsmGeneration::BASE_TABLE;
        let gen1 = LsmGeneration::memtable(1);
        let gen2 = LsmGeneration::memtable(2);
        let gen10 = LsmGeneration::memtable(10);

        // Base table (gen=0) should be less than all MemTable generations
        assert!(base < gen1);
        assert!(base < gen2);
        assert!(base < gen10);

        // Higher generation = newer data
        assert!(gen1 < gen2);
        assert!(gen2 < gen10);

        // Test display
        assert_eq!(base.to_string(), "base");
        assert_eq!(gen1.to_string(), "gen1");
        assert_eq!(gen10.to_string(), "gen10");

        // Test as_u64
        assert_eq!(base.as_u64(), 0);
        assert_eq!(gen1.as_u64(), 1);
        assert_eq!(gen10.as_u64(), 10);
    }

    #[test]
    fn test_lsm_generation_conversions() {
        let from_u64: LsmGeneration = 5u64.into();
        assert_eq!(from_u64.as_u64(), 5);

        let base: LsmGeneration = 0u64.into();
        assert!(base.is_base_table());
    }

    #[test]
    #[should_panic(expected = "MemTable generation must be >= 1")]
    fn test_memtable_generation_zero_panics() {
        LsmGeneration::memtable(0);
    }

    #[test]
    fn test_region_snapshot_builder() {
        let region_id = Uuid::new_v4();
        let snapshot = RegionSnapshot::new(region_id)
            .with_spec_id(1)
            .with_current_generation(5)
            .with_flushed_generation(1, "abc123_gen_1".to_string())
            .with_flushed_generation(2, "def456_gen_2".to_string());

        assert_eq!(snapshot.region_id, region_id);
        assert_eq!(snapshot.spec_id, 1);
        assert_eq!(snapshot.current_generation, 5);
        assert_eq!(snapshot.flushed_generations.len(), 2);
        assert_eq!(snapshot.flushed_generations[0].generation, 1);
        assert_eq!(snapshot.flushed_generations[1].generation, 2);
    }
}
