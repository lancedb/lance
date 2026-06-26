// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Module for statistics related to the dataset.

use std::{collections::HashMap, future::Future, sync::Arc};

use futures::{StreamExt, TryStreamExt};
use lance_core::Result;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};

use super::{Dataset, fragment::FileFragment};

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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_file::version::LanceFileVersion;

    use crate::dataset::WriteParams;

    use super::*;

    #[tokio::test]
    async fn test_calculate_data_stats_after_dropping_wide_dataset_columns() {
        let num_columns = 64;
        let num_rows = 128;
        let schema = Arc::new(ArrowSchema::new(
            (0..num_columns)
                .map(|idx| ArrowField::new(format!("col_{idx}"), DataType::Int32, true))
                .collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(
            schema.clone(),
            (0..num_columns)
                .map(|column_idx| {
                    Arc::new(Int32Array::from_iter_values(
                        (0..num_rows).map(|row_idx| row_idx + column_idx),
                    )) as ArrayRef
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let test_dir = TempStrDir::default();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(
            reader,
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let columns_to_drop = (1..num_columns)
            .map(|idx| format!("col_{idx}"))
            .collect::<Vec<_>>();
        let column_refs = columns_to_drop
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        dataset.drop_columns(&column_refs).await.unwrap();

        let stats = Arc::new(dataset).calculate_data_stats().await.unwrap();
        assert_eq!(stats.fields.len(), 1);
        assert_eq!(stats.fields[0].id, 0);
        assert!(
            stats.fields[0].bytes_on_disk > 0,
            "bytes_on_disk should include the remaining column after drop_columns"
        );
    }
}
