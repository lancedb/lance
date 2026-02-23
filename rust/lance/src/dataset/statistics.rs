// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Module for statistics related to the dataset.

use std::{collections::HashMap, future::Future, sync::Arc};

use futures::{StreamExt, TryStreamExt};
use lance_core::Result;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};

use super::{fragment::FileFragment, Dataset};

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

pub trait DatasetStatisticsExt {
    /// Get statistics about the data in the dataset
    fn calculate_data_stats(
        self: &Arc<Self>,
    ) -> impl Future<Output = Result<DataStatistics>> + Send;
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
            let schema = self.schema().clone();
            let dataset = self.clone();
            let fragments = self.fragments().as_ref().clone();
            futures::stream::iter(fragments)
                .map(|fragment| {
                    let file_fragment = FileFragment::new(dataset.clone(), fragment);
                    let schema = schema.clone();
                    let scan_scheduler = scan_scheduler.clone();
                    async move { file_fragment.storage_stats(&schema, scan_scheduler).await }
                })
                .buffer_unordered(self.object_store.io_parallelism())
                .try_for_each(|fragment_stats| {
                    for (field_id, bytes) in fragment_stats {
                        if let Some(stats) = field_stats.get_mut(&field_id) {
                            stats.bytes_on_disk += bytes;
                        }
                    }
                    futures::future::ready(Ok(()))
                })
                .await?;
        }
        let field_stats = field_ids
            .into_iter()
            .map(|id| field_stats.remove(&(id as u32)).unwrap())
            .collect();
        Ok(DataStatistics {
            fields: field_stats,
        })
    }
}
