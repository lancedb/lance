// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Module for statistics related to the dataset.

use std::{collections::{HashMap, HashSet}, future::Future, sync::Arc};

use super::{fragment::FileFragment, Dataset};
use lance_core::Result;
use lance_index::scalar::AnyQuery;
use lance_index::{metrics::NoOpMetricsCollector, scalar::zonemap::{ZoneMapIndex, ZoneMapStatistics}};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_table::{
    format::RowIdMeta,
    io::deletion::deletion_file_path,
};

/// Statistics about a single field in the dataset
pub struct FieldStatistics {
    /// Id of the field
    pub id: u32,
    /// Amount of data in the field (after compression, if any)
    ///
    /// This will be 0 if the data storage version is less than 2
    pub bytes_on_disk: u64,
}

/// Statistics about the data in the dataset
pub struct DataStatistics {
    /// Statistics about each field in the dataset
    pub fields: Vec<FieldStatistics>,
}

/// A `Split` refers to a subset of a dataset which has been filtered by a query
/// `Split` contains a list of zonemaps which contain fragment that have rows included in the query.
/// It also contains a mapping of fragment IDs to their associated files.
#[derive(Debug, Clone)]
pub struct Split {
    /// Zone statistics for this split
    pub zone_stats: Vec<ZoneMapStatistics>,
    /// Map of fragment ID to all files associated with that fragment
    pub files: HashMap<u64, FragmentFiles>,
}

#[derive(Debug, Default, Clone)]
pub struct FragmentFiles {
    pub data_files: Vec<String>,
    pub deletion_files: Vec<String>,
    pub row_id_files: Vec<String>,
}

pub trait DatasetStatisticsExt {
    /// Get statistics about the data in the dataset
    fn calculate_data_stats(
        self: &Arc<Self>,
    ) -> impl Future<Output = Result<DataStatistics>> + Send;
    
    /// Get splits partitioned by target size with associated file mappings
    fn get_splits(
        self: &Arc<Self>,
        query: &dyn AnyQuery, 
        target_size: usize
    ) -> Result<Vec<Split>>;
    
    /// Build a mapping of fragment ID to FragmentFiles for the given zones
    fn build_fragment_files_mapping(&self, zones: &Vec<ZoneMapStatistics>) -> Result<HashMap<u64, FragmentFiles>>;
}

impl DatasetStatisticsExt for Dataset {
    async fn calculate_data_stats(self: &Arc<Self>) -> Result<DataStatistics> {
        let field_ids = self.schema().field_ids();
        let mut field_stats: HashMap<u32, FieldStatistics> =
            HashMap::from_iter(field_ids.iter().map(|id| {
                (
                    *id as u32,
                    FieldStatistics {
                        id: *id as u32,
                        bytes_on_disk: 0,
                    },
                )
            }));
        if !self.is_legacy_storage() {
            let scan_scheduler = ScanScheduler::new(
                self.object_store.clone(),
                SchedulerConfig::max_bandwidth(self.object_store.as_ref()),
            );
            for fragment in self.fragments().as_ref() {
                let file_fragment = FileFragment::new(self.clone(), fragment.clone());
                file_fragment
                    .update_storage_stats(&mut field_stats, self.schema(), scan_scheduler.clone())
                    .await?;
            }
        }
        let field_stats = field_ids
            .into_iter()
            .map(|id| field_stats.remove(&(id as u32)).unwrap())
            .collect();
        Ok(DataStatistics {
            fields: field_stats,
        })
    }

    fn get_splits(
        self: &Arc<Self>,
        query: &dyn AnyQuery, 
        target_size: usize
    ) -> Result<Vec<Split>> {
        let index: ZoneMapIndex; // TODO: find out how to get access to the zonemap index for all? columns
        let mut filtered_zones = index.fetch_zones_for_query(query, &NoOpMetricsCollector)?;
        // Sort by fragment ID to co-locate overlapping zones
        filtered_zones.sort_by(|z1, z2| {
            if z1.fragment_id != z2.fragment_id {
                return z1.fragment_id.cmp(&z2.fragment_id)
            }
            return z1.zone_start.cmp(&z2.zone_start);
        });

        let mut splits = Vec::new();
        let mut current_split_zones = Vec::new();
        let mut current_size = 0;
        
        for zone in filtered_zones {
            let zone_size = zone.zone_length;

            // TODO: need to confirm if this is right
            if current_size > 0 && current_size + zone_size > target_size {
                let files = self.build_fragment_files_mapping(&current_split_zones)?;
                splits.push(Split {
                    zone_stats: current_split_zones,
                    files,
                });

                current_split_zones = vec![zone];
                current_size = zone_size;
            } else {
                current_split_zones.push(zone);
                current_size += zone_size;
            }
        }

        if !current_split_zones.is_empty() {
            let files = self.build_fragment_files_mapping(&current_split_zones)?;
            splits.push(Split {
                zone_stats: current_split_zones,
                files,
            });
        }
        
        Ok(splits)
    }

    /// Build a mapping of fragment ID to FragmentFiles for the given zones
    /// TODO: maybe move to fragment.rs or similar
    fn build_fragment_files_mapping(&self, zones: &Vec<ZoneMapStatistics>) -> Result<HashMap<u64, FragmentFiles>> {
        let fragment_ids: HashSet<u64> = zones
            .iter()
            .map(|zone| zone.fragment_id)
            .collect();
        
        let mut files_mapping = HashMap::new();
        for fragment_id in fragment_ids {
            if let Some(fragment) = self.get_fragment(fragment_id as usize) {
                let frag_metadata = fragment.metadata();
                let mut fragment_files = FragmentFiles::default();
                
                for data_file in &frag_metadata.files {
                    fragment_files.data_files.push(format!("data/{}", data_file.path));
                }
                
                if let Some(deletion_file) = &frag_metadata.deletion_file {
                    let deletion_path = deletion_file_path(
                        &self.base, 
                        fragment_id, 
                        deletion_file
                    );
                    fragment_files.deletion_files.push(deletion_path.to_string());
                }

                if let Some(row_id_meta) = &frag_metadata.row_id_meta {
                    if let RowIdMeta::External(external_file) = row_id_meta {
                        fragment_files.row_id_files.push(external_file.path.clone());
                    }
                    // Inline row IDs don't have separate files, so we skip them
                }
                
                files_mapping.insert(fragment_id, fragment_files);
            }
        }
        
        Ok(files_mapping)
    }
}
