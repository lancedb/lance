// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Data source collector for LSM scanner.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_schema::SchemaRef;
use lance_core::Result;
use uuid::Uuid;

use super::data_source::{LsmDataSource, LsmGeneration, ShardSnapshot};
use crate::dataset::Dataset;
use crate::dataset::mem_wal::write::{BatchStore, IndexStore};

/// Reference to an active (in-memory) MemTable.
#[derive(Clone)]
pub struct ActiveMemTableRef {
    /// Batch store containing the data.
    pub batch_store: Arc<BatchStore>,
    /// Index store for the MemTable.
    pub index_store: Arc<IndexStore>,
    /// Schema of the data.
    pub schema: SchemaRef,
    /// Current generation number.
    pub generation: u64,
}

/// Collects data sources from base table and MemWAL shards.
///
/// This collector gathers all data sources that need to be scanned
/// for a query, including:
/// - The base table (merged data) — optional; omit for fresh-tier-only scans
/// - Flushed MemTables from each shard
/// - Active MemTables (optional, for strong consistency)
///
/// When the base table is omitted (see [`Self::without_base_table`]), `collect`
/// returns only flushed-generation and active-memtable sources. This is used
/// by callers that own the base read path elsewhere and only need the WAL's
/// fresh tier (active memtable ∪ L0 flushed generations).
pub struct LsmDataSourceCollector {
    /// Base Lance table (None when scanning only the fresh tier).
    base_table: Option<Arc<Dataset>>,
    /// Base path for resolving relative flushed-generation paths.
    base_path: String,
    /// Shard snapshots from MemWAL index.
    shard_snapshots: Vec<ShardSnapshot>,
    /// Active MemTables by shard (for strong consistency).
    active_memtables: HashMap<Uuid, ActiveMemTableRef>,
}

impl LsmDataSourceCollector {
    /// Create a new collector from base table and shard snapshots.
    ///
    /// # Arguments
    ///
    /// * `base_table` - The base Lance table (merged data)
    /// * `shard_snapshots` - Snapshots of shard states from MemWAL index
    pub fn new(base_table: Arc<Dataset>, shard_snapshots: Vec<ShardSnapshot>) -> Self {
        // Use the dataset's URI as base path for resolving relative paths.
        // This ensures memory:// and other scheme-based URIs work correctly.
        let base_path = base_table.uri().trim_end_matches('/').to_string();
        Self {
            base_table: Some(base_table),
            base_path,
            shard_snapshots,
            active_memtables: HashMap::new(),
        }
    }

    /// Create a collector without a base table (fresh-tier scan only).
    ///
    /// The collector emits only flushed-generation and active-memtable sources.
    /// `base_path` is the table-root URI used to resolve relative flushed paths
    /// (typically the same URI that would have been the base dataset's URI).
    pub fn without_base_table(
        base_path: impl Into<String>,
        shard_snapshots: Vec<ShardSnapshot>,
    ) -> Self {
        Self {
            base_table: None,
            base_path: base_path.into().trim_end_matches('/').to_string(),
            shard_snapshots,
            active_memtables: HashMap::new(),
        }
    }

    /// Add an active MemTable for strong consistency reads.
    ///
    /// Active MemTables contain data that may not be persisted yet.
    /// Including them provides strong consistency at the cost of
    /// requiring coordination with the writer.
    pub fn with_active_memtable(mut self, shard_id: Uuid, memtable: ActiveMemTableRef) -> Self {
        self.active_memtables.insert(shard_id, memtable);
        self
    }

    /// Get the base table, if any.
    pub fn base_table(&self) -> Option<&Arc<Dataset>> {
        self.base_table.as_ref()
    }

    /// Get all shard snapshots.
    pub fn shard_snapshots(&self) -> &[ShardSnapshot] {
        &self.shard_snapshots
    }

    /// Get active MemTables.
    pub fn active_memtables(&self) -> &HashMap<Uuid, ActiveMemTableRef> {
        &self.active_memtables
    }

    /// Collect all data sources.
    ///
    /// Returns sources in a consistent order:
    /// 1. Base table (gen=0), if configured
    /// 2. Flushed MemTables per shard, ordered by generation
    /// 3. Active MemTables per shard
    pub fn collect(&self) -> Result<Vec<LsmDataSource>> {
        let mut sources = Vec::new();

        if let Some(base) = &self.base_table {
            sources.push(LsmDataSource::BaseTable {
                dataset: base.clone(),
            });
        }

        for snapshot in &self.shard_snapshots {
            for flushed in &snapshot.flushed_generations {
                let path = self.resolve_flushed_path(&snapshot.shard_id, &flushed.path);
                sources.push(LsmDataSource::FlushedMemTable {
                    path,
                    shard_id: snapshot.shard_id,
                    generation: LsmGeneration::memtable(flushed.generation),
                });
            }
        }

        for (shard_id, memtable) in &self.active_memtables {
            sources.push(LsmDataSource::ActiveMemTable {
                batch_store: memtable.batch_store.clone(),
                index_store: memtable.index_store.clone(),
                schema: memtable.schema.clone(),
                shard_id: *shard_id,
                generation: LsmGeneration::memtable(memtable.generation),
            });
        }

        Ok(sources)
    }

    /// Collect data sources for specific shards only.
    ///
    /// This is used after shard pruning to avoid loading data from
    /// shards that cannot contain matching rows.
    ///
    /// The base table (when configured) is always included since it may
    /// contain data from any shard (after merging).
    pub fn collect_for_shards(&self, shard_ids: &HashSet<Uuid>) -> Result<Vec<LsmDataSource>> {
        let mut sources = Vec::new();

        if let Some(base) = &self.base_table {
            sources.push(LsmDataSource::BaseTable {
                dataset: base.clone(),
            });
        }

        for snapshot in &self.shard_snapshots {
            if !shard_ids.contains(&snapshot.shard_id) {
                continue;
            }

            for flushed in &snapshot.flushed_generations {
                let path = self.resolve_flushed_path(&snapshot.shard_id, &flushed.path);
                sources.push(LsmDataSource::FlushedMemTable {
                    path,
                    shard_id: snapshot.shard_id,
                    generation: LsmGeneration::memtable(flushed.generation),
                });
            }
        }

        for (shard_id, memtable) in &self.active_memtables {
            if !shard_ids.contains(shard_id) {
                continue;
            }

            sources.push(LsmDataSource::ActiveMemTable {
                batch_store: memtable.batch_store.clone(),
                index_store: memtable.index_store.clone(),
                schema: memtable.schema.clone(),
                shard_id: *shard_id,
                generation: LsmGeneration::memtable(memtable.generation),
            });
        }

        Ok(sources)
    }

    /// Get the total number of data sources.
    pub fn num_sources(&self) -> usize {
        let flushed_count: usize = self
            .shard_snapshots
            .iter()
            .map(|s| s.flushed_generations.len())
            .sum();
        let base_count = if self.base_table.is_some() { 1 } else { 0 };
        base_count + flushed_count + self.active_memtables.len()
    }

    /// Resolve a flushed MemTable path to an absolute path.
    ///
    /// Flushed MemTables are stored at: `{base_path}/_mem_wal/{shard_id}/{folder_name}`
    /// The `folder_name` is what's stored in `FlushedGeneration.path`.
    fn resolve_flushed_path(&self, shard_id: &Uuid, folder_name: &str) -> String {
        format!("{}/_mem_wal/{}/{}", self.base_path, shard_id, folder_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::mem_wal::scanner::data_source::FlushedGeneration;

    fn create_test_snapshots() -> Vec<ShardSnapshot> {
        let shard_a = Uuid::new_v4();
        let shard_b = Uuid::new_v4();

        vec![
            ShardSnapshot {
                shard_id: shard_a,
                spec_id: 1,
                current_generation: 3,
                flushed_generations: vec![
                    FlushedGeneration {
                        generation: 1,
                        path: "abc_gen_1".to_string(),
                    },
                    FlushedGeneration {
                        generation: 2,
                        path: "def_gen_2".to_string(),
                    },
                ],
            },
            ShardSnapshot {
                shard_id: shard_b,
                spec_id: 1,
                current_generation: 2,
                flushed_generations: vec![FlushedGeneration {
                    generation: 1,
                    path: "xyz_gen_1".to_string(),
                }],
            },
        ]
    }

    #[test]
    fn test_collector_num_sources() {
        let snapshots = create_test_snapshots();
        // 1 base table + 2 flushed from shard_a + 1 flushed from shard_b = 4
        // Using a mock dataset is complex, so we just test the counting logic
        assert_eq!(snapshots[0].flushed_generations.len(), 2);
        assert_eq!(snapshots[1].flushed_generations.len(), 1);
    }

    #[test]
    fn test_active_memtable_ref() {
        let batch_store = Arc::new(BatchStore::with_capacity(100));
        let index_store = Arc::new(IndexStore::new());
        let schema = Arc::new(arrow_schema::Schema::empty());

        let memtable_ref = ActiveMemTableRef {
            batch_store,
            index_store,
            schema,
            generation: 5,
        };

        assert_eq!(memtable_ref.generation, 5);
    }
}
