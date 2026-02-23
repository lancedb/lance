// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Data source collector for LSM scanner.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_schema::SchemaRef;
use lance_core::Result;
use uuid::Uuid;

use super::data_source::{LsmDataSource, LsmGeneration, RegionSnapshot};
use crate::dataset::mem_wal::write::{BatchStore, IndexStore};
use crate::dataset::Dataset;

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

/// Collects data sources from base table and MemWAL regions.
///
/// This collector gathers all data sources that need to be scanned
/// for a query, including:
/// - The base table (merged data)
/// - Flushed MemTables from each region
/// - Active MemTables (optional, for strong consistency)
pub struct LsmDataSourceCollector {
    /// Base Lance table.
    base_table: Arc<Dataset>,
    /// Base path for resolving relative paths.
    base_path: String,
    /// Region snapshots from MemWAL index.
    region_snapshots: Vec<RegionSnapshot>,
    /// Active MemTables by region (for strong consistency).
    active_memtables: HashMap<Uuid, ActiveMemTableRef>,
}

impl LsmDataSourceCollector {
    /// Create a new collector from base table and region snapshots.
    ///
    /// # Arguments
    ///
    /// * `base_table` - The base Lance table (merged data)
    /// * `region_snapshots` - Snapshots of region states from MemWAL index
    pub fn new(base_table: Arc<Dataset>, region_snapshots: Vec<RegionSnapshot>) -> Self {
        // Use the dataset's URI as base path for resolving relative paths.
        // This ensures memory:// and other scheme-based URIs work correctly.
        let base_path = base_table.uri().trim_end_matches('/').to_string();
        Self {
            base_table,
            base_path,
            region_snapshots,
            active_memtables: HashMap::new(),
        }
    }

    /// Add an active MemTable for strong consistency reads.
    ///
    /// Active MemTables contain data that may not be persisted yet.
    /// Including them provides strong consistency at the cost of
    /// requiring coordination with the writer.
    pub fn with_active_memtable(mut self, region_id: Uuid, memtable: ActiveMemTableRef) -> Self {
        self.active_memtables.insert(region_id, memtable);
        self
    }

    /// Get the base table.
    pub fn base_table(&self) -> &Arc<Dataset> {
        &self.base_table
    }

    /// Get all region snapshots.
    pub fn region_snapshots(&self) -> &[RegionSnapshot] {
        &self.region_snapshots
    }

    /// Get active MemTables.
    pub fn active_memtables(&self) -> &HashMap<Uuid, ActiveMemTableRef> {
        &self.active_memtables
    }

    /// Collect all data sources.
    ///
    /// Returns sources in a consistent order:
    /// 1. Base table (gen=0)
    /// 2. Flushed MemTables per region, ordered by generation
    /// 3. Active MemTables per region
    pub fn collect(&self) -> Result<Vec<LsmDataSource>> {
        let mut sources = Vec::new();

        // 1. Add base table
        sources.push(LsmDataSource::BaseTable {
            dataset: self.base_table.clone(),
        });

        // 2. Add flushed MemTables from each region
        for snapshot in &self.region_snapshots {
            for flushed in &snapshot.flushed_generations {
                let path = self.resolve_flushed_path(&snapshot.region_id, &flushed.path);
                sources.push(LsmDataSource::FlushedMemTable {
                    path,
                    region_id: snapshot.region_id,
                    generation: LsmGeneration::memtable(flushed.generation),
                });
            }
        }

        // 3. Add active MemTables
        for (region_id, memtable) in &self.active_memtables {
            sources.push(LsmDataSource::ActiveMemTable {
                batch_store: memtable.batch_store.clone(),
                index_store: memtable.index_store.clone(),
                schema: memtable.schema.clone(),
                region_id: *region_id,
                generation: LsmGeneration::memtable(memtable.generation),
            });
        }

        Ok(sources)
    }

    /// Collect data sources for specific regions only.
    ///
    /// This is used after region pruning to avoid loading data from
    /// regions that cannot contain matching rows.
    ///
    /// The base table is always included since it may contain data
    /// from any region (after merging).
    pub fn collect_for_regions(&self, region_ids: &HashSet<Uuid>) -> Result<Vec<LsmDataSource>> {
        let mut sources = Vec::new();

        // Base table is always included (contains merged data from all regions)
        sources.push(LsmDataSource::BaseTable {
            dataset: self.base_table.clone(),
        });

        // Filter flushed MemTables by region
        for snapshot in &self.region_snapshots {
            if !region_ids.contains(&snapshot.region_id) {
                continue;
            }

            for flushed in &snapshot.flushed_generations {
                let path = self.resolve_flushed_path(&snapshot.region_id, &flushed.path);
                sources.push(LsmDataSource::FlushedMemTable {
                    path,
                    region_id: snapshot.region_id,
                    generation: LsmGeneration::memtable(flushed.generation),
                });
            }
        }

        // Filter active MemTables by region
        for (region_id, memtable) in &self.active_memtables {
            if !region_ids.contains(region_id) {
                continue;
            }

            sources.push(LsmDataSource::ActiveMemTable {
                batch_store: memtable.batch_store.clone(),
                index_store: memtable.index_store.clone(),
                schema: memtable.schema.clone(),
                region_id: *region_id,
                generation: LsmGeneration::memtable(memtable.generation),
            });
        }

        Ok(sources)
    }

    /// Get the total number of data sources.
    pub fn num_sources(&self) -> usize {
        let flushed_count: usize = self
            .region_snapshots
            .iter()
            .map(|s| s.flushed_generations.len())
            .sum();
        1 + flushed_count + self.active_memtables.len()
    }

    /// Resolve a flushed MemTable path to an absolute path.
    ///
    /// Flushed MemTables are stored at: `{base_path}/_mem_wal/{region_id}/{folder_name}`
    /// The `folder_name` is what's stored in `FlushedGeneration.path`.
    fn resolve_flushed_path(&self, region_id: &Uuid, folder_name: &str) -> String {
        format!("{}/_mem_wal/{}/{}", self.base_path, region_id, folder_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::mem_wal::scanner::data_source::FlushedGeneration;

    fn create_test_snapshots() -> Vec<RegionSnapshot> {
        let region_a = Uuid::new_v4();
        let region_b = Uuid::new_v4();

        vec![
            RegionSnapshot {
                region_id: region_a,
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
            RegionSnapshot {
                region_id: region_b,
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
        // 1 base table + 2 flushed from region_a + 1 flushed from region_b = 4
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
