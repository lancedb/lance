// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::transaction::Transaction;
use crate::Dataset;
use crate::Result;
use arrow_array::RecordBatch;
use futures::stream::{self, BoxStream, StreamExt, TryStreamExt};
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::Error;
use lance_io::stream::RecordBatchStream;
use lance_table::format::{Fragment, RowLatestUpdateVersionMeta, RowLatestUpdateVersionSequence};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use tracing::instrument;

/// The delta dataset between two versions of a dataset.
pub struct DatasetDelta {
    /// The base version number for comparison.
    pub(crate) begin_version: u64,
    /// The current version number.
    pub(crate) end_version: u64,
    /// Base path of the dataset.
    pub(crate) base_dataset: Dataset,
}

impl DatasetDelta {
    /// Listing the transactions between two versions.
    pub async fn list_transactions(&self) -> Result<Vec<Transaction>> {
        stream::iter((self.begin_version + 1)..=self.end_version)
            .map(|version| {
                let base_dataset = self.base_dataset.clone();
                async move {
                    let current_ds = match base_dataset.checkout_version(version).await {
                        Ok(ds) => ds,
                        Err(err) => {
                            if matches!(err, Error::DatasetNotFound { .. }) {
                                return Err(Error::VersionNotFound {
                                    message: format!(
                                        "Can not find version {}, please check if it has been cleanup.",
                                        version
                                    ),
                                });
                            } else {
                                return Err(err);
                            }
                        }
                    };
                    current_ds.read_transaction().await
                }
            })
            .buffered(get_num_compute_intensive_cpus())
            .try_filter_map(|result| async move { Ok(result) })
            .try_collect()
            .await
    }

    pub async fn to_stream(&self) -> Result<Option<Box<dyn RecordBatchStream>>> {
        Ok(None)
    }

    pub async fn to_insert_stream(&self) -> Result<Option<Box<dyn RecordBatchStream>>> {
        Ok(None)
    }

    pub async fn to_delete_stream(&self) -> Result<Option<Box<dyn RecordBatchStream>>> {
        Ok(None)
    }

    pub async fn to_update_stream(&self) -> Result<Option<Box<dyn RecordBatchStream>>> {
        Ok(None)
    }
}

/// Type of diff operation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffOperation {
    /// Record was inserted in the newer version
    Insert,
    /// Record was updated between versions
    Update,
    /// Record was deleted in the newer version
    Delete,
}

/// A record representing a difference between two dataset versions
#[derive(Debug, Clone)]
pub struct DiffRecord {
    /// The row ID of the record
    pub row_id: u64,
    /// The type of operation (insert, update, delete)
    pub operation: DiffOperation,
    /// The record data in the old version (None for inserts)
    pub old_data: Option<RecordBatch>,
    /// The record data in the new version (None for deletes)
    pub new_data: Option<RecordBatch>,
    /// The version when this change occurred
    pub version: u64,
}

/// Stream of diff records
pub type DiffRecordStream = BoxStream<'static, Result<DiffRecord>>;

/// Configuration for diff operations
#[derive(Debug, Clone)]
pub struct DiffConfig {
    /// Maximum number of records to process in parallel
    pub max_concurrency: usize,
    /// Whether to include the actual record data in diff results
    pub include_data: bool,
    /// Batch size for processing records
    pub batch_size: usize,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 4,
            include_data: true,
            batch_size: 1000,
        }
    }
}

/// Fragment-level diff analyzer
pub struct FragmentDiffAnalyzer {
    current_dataset: Arc<Dataset>,
    old_dataset: Arc<Dataset>,
    compared_version: u64,
    config: DiffConfig,
}

impl FragmentDiffAnalyzer {
    /// Create a new fragment diff analyzer
    pub fn new(
        current_dataset: Arc<Dataset>,
        old_dataset: Arc<Dataset>,
        compared_version: u64,
        config: DiffConfig,
    ) -> Self {
        Self {
            current_dataset,
            old_dataset,
            compared_version,
            config,
        }
    }

    /// Check if a fragment should be skipped during diff analysis
    ///
    /// This uses the min/max_latest_update_version optimization fields
    /// to skip fragments that don't contain any changes.
    pub fn should_skip_fragment(&self, fragment: &Fragment) -> Result<bool> {
        // If fragment doesn't have version metadata, we can't skip it
        if fragment.row_latest_update_version_meta.is_none() {
            return Ok(false);
        }

        // Use min/max version optimization if available
        if let (Some(min_ver), Some(max_ver)) = (
            fragment.min_latest_update_version,
            fragment.max_latest_update_version,
        ) {
            // If all records in this fragment have versions <= compared_version,
            // then there are no changes to report
            if max_ver <= self.compared_version {
                return Ok(true);
            }

            // If all records have versions > compared_version, we need to analyze
            // (they could be inserts or updates)
            if min_ver > self.compared_version {
                return Ok(false);
            }
        }

        // Fragment spans the compared version, need to analyze
        Ok(false)
    }

    /// Analyze a single fragment for differences
    #[instrument(level = "debug", skip_all)]
    pub async fn analyze_fragment(&self, fragment: &Fragment) -> Result<Vec<DiffRecord>> {
        // Check if we can skip this fragment
        if self.should_skip_fragment(fragment)? {
            return Ok(vec![]);
        }

        let mut diff_records = Vec::new();

        // Load the row version sequence for this fragment
        if let Some(version_meta) = &fragment.row_latest_update_version_meta {
            let version_sequence = version_meta
                .load_sequence(&*self.current_dataset.object_store.inner)
                .await?;

            // Find updated and inserted records
            let updated_inserted = self
                .find_updated_and_inserted_records(fragment, &version_sequence)
                .await?;
            diff_records.extend(updated_inserted);
        }

        // Find deleted records by comparing row ID sets
        let deleted_records = self.find_deleted_records(fragment).await?;
        diff_records.extend(deleted_records);

        Ok(diff_records)
    }

    /// Find records that were updated or inserted
    async fn find_updated_and_inserted_records(
        &self,
        _fragment: &Fragment,
        version_sequence: &RowLatestUpdateVersionSequence,
    ) -> Result<Vec<DiffRecord>> {
        let mut records = Vec::new();

        // Get all rows with versions greater than the compared version
        let changed_rows = version_sequence.rows_with_version_greater_than(self.compared_version);

        for row_id in changed_rows {
            let version = version_sequence.get_version(row_id).unwrap();

            // Determine if this is an insert or update
            let old_data = if self.config.include_data {
                self.get_record_data(row_id, &self.old_dataset).await.ok()
            } else {
                None
            };

            let new_data = if self.config.include_data {
                Some(self.get_record_data(row_id, &self.current_dataset).await?)
            } else {
                None
            };

            let operation = if old_data.is_some() {
                DiffOperation::Update
            } else {
                DiffOperation::Insert
            };

            records.push(DiffRecord {
                row_id,
                operation,
                old_data,
                new_data,
                version,
            });
        }

        Ok(records)
    }

    /// Find records that were deleted
    async fn find_deleted_records(&self, fragment: &Fragment) -> Result<Vec<DiffRecord>> {
        let mut records = Vec::new();

        // Get row ID sets from both versions
        let current_row_ids = self
            .get_fragment_row_ids(fragment, &self.current_dataset)
            .await?;
        let old_row_ids = self
            .get_fragment_row_ids(fragment, &self.old_dataset)
            .await?;

        // Find rows that exist in old version but not in current version
        let deleted_row_ids: Vec<u64> = old_row_ids.difference(&current_row_ids).cloned().collect();

        for row_id in deleted_row_ids {
            let old_data = if self.config.include_data {
                Some(self.get_record_data(row_id, &self.old_dataset).await?)
            } else {
                None
            };

            records.push(DiffRecord {
                row_id,
                operation: DiffOperation::Delete,
                old_data,
                new_data: None,
                version: self.current_dataset.manifest.version,
            });
        }

        Ok(records)
    }

    /// Get the set of row IDs in a fragment for a specific dataset version
    async fn get_fragment_row_ids(
        &self,
        fragment: &Fragment,
        dataset: &Dataset,
    ) -> Result<HashSet<u64>> {
        // This is a simplified implementation
        // In practice, this would load the actual row ID sequence from the fragment
        if let Some(row_id_meta) = &fragment.row_id_meta {
            match row_id_meta {
                lance_table::format::RowIdMeta::Inline(data) => {
                    // Deserialize the row ID sequence from inline data
                    let row_ids = self.deserialize_row_ids(data)?;
                    Ok(row_ids.into_iter().collect())
                }
                lance_table::format::RowIdMeta::External(_file) => {
                    // Load from external file
                    // TODO: Implement external file loading
                    todo!("External row ID file loading not yet implemented")
                }
            }
        } else {
            // Legacy fragment without row ID metadata
            panic!("Fragment does not have row ID metadata, please make sure stable row IDs are enabled");
        }
    }

    /// Get record data for a specific row ID from a dataset
    async fn get_record_data(&self, row_id: u64, dataset: &Dataset) -> Result<RecordBatch> {
        // Use the dataset's take_rows functionality to retrieve the specific record by row ID
        let row_ids = vec![row_id];
        let schema = dataset.schema().clone();
        let projection = super::ProjectionRequest::Schema(Arc::new(schema));
        
        let batch = dataset.take_rows(&row_ids, projection).await?;
        
        // Verify we got exactly one row
        if batch.num_rows() == 0 {
            return Err(Error::invalid_input(
                format!("Row ID {} not found in dataset", row_id),
                Default::default(),
            ));
        } else if batch.num_rows() > 1 {
            return Err(Error::invalid_input(
                format!("Expected 1 row for row ID {}, got {}", row_id, batch.num_rows()),
                Default::default(),
            ));
        }
        
        Ok(batch)
    }

    /// Deserialize row IDs from inline data
    fn deserialize_row_ids(&self, _data: &[u8]) -> Result<Vec<u64>> {
        // This is a placeholder implementation
        // In practice, this would deserialize the actual RowIdSequence
        Ok(vec![])
    }

    /// Create a stream of diff records for all fragments
    pub fn create_diff_stream(self) -> DiffRecordStream {
        let fragments = Arc::try_unwrap(self.current_dataset.manifest.fragments.clone())
            .unwrap_or_else(|arc| (*arc).clone());
        let max_concurrency = self.config.max_concurrency;

        let stream = futures::stream::iter(fragments)
            .map(move |fragment| {
                let analyzer = self.clone();
                async move { analyzer.analyze_fragment(&fragment).await }
            })
            .buffer_unordered(max_concurrency)
            .flat_map(|result| {
                futures::stream::iter(match result {
                    Ok(records) => records.into_iter().map(Ok).collect::<Vec<_>>(),
                    Err(e) => vec![Err(e)],
                })
            });

        Box::pin(stream)
    }
}

/// Implement Clone for FragmentDiffAnalyzer to support stream operations
impl Clone for FragmentDiffAnalyzer {
    fn clone(&self) -> Self {
        Self {
            current_dataset: self.current_dataset.clone(),
            old_dataset: self.old_dataset.clone(),
            compared_version: self.compared_version,
            config: self.config.clone(),
        }
    }
}

/// Dataset diff builder for configuring and executing diff operations
pub struct DatasetDiffBuilder {
    dataset: Arc<Dataset>,
    compared_version: u64,
    config: DiffConfig,
}

impl DatasetDiffBuilder {
    /// Create a new dataset diff builder
    pub fn new(dataset: Arc<Dataset>, compared_version: u64) -> Self {
        Self {
            dataset,
            compared_version,
            config: DiffConfig::default(),
        }
    }

    /// Set the maximum concurrency for diff operations
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.config.max_concurrency = max_concurrency;
        self
    }

    /// Set whether to include actual record data in diff results
    pub fn with_include_data(mut self, include_data: bool) -> Self {
        self.config.include_data = include_data;
        self
    }

    /// Set the batch size for processing records
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Execute the diff operation and return a stream of diff records
    pub async fn execute(self) -> Result<DiffRecordStream> {
        // Validate preconditions
        self.validate_preconditions().await?;

        // Load the old version of the dataset
        let old_dataset = Arc::new(self.dataset.checkout_version(self.compared_version).await?);

        // Create the fragment diff analyzer
        let analyzer = FragmentDiffAnalyzer::new(
            self.dataset,
            old_dataset,
            self.compared_version,
            self.config,
        );

        // Return the diff stream
        Ok(analyzer.create_diff_stream())
    }

    /// Validate preconditions for diff operation
    async fn validate_preconditions(&self) -> Result<()> {
        // Check if stable row IDs are enabled
        if !self.dataset.manifest.uses_stable_row_ids() {
            return Err(Error::invalid_input(
                "Diff functionality requires stable row IDs to be enabled",
                Default::default(),
            ));
        }

        // Validate version bounds
        if self.compared_version >= self.dataset.manifest.version {
            return Err(Error::invalid_input(
                format!(
                    "Compared version {} must be less than current version {}",
                    self.compared_version, self.dataset.manifest.version
                ),
                Default::default(),
            ));
        }

        if self.compared_version < 1 {
            return Err(Error::invalid_input(
                format!(
                    "Compared version must be > 0 (got {})",
                    self.compared_version
                ),
                Default::default(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use crate::dataset::transaction::Operation;
    use crate::dataset::{Dataset, WriteParams};
    use arrow_array::types::Int32Type;
    use chrono::Duration;
    use lance_core::utils::testing::MockClock;
    use lance_datagen::{array, BatchCount, RowCount};

    async fn create_test_dataset() -> Dataset {
        let data = lance_datagen::gen_batch()
            .col("key", array::step::<Int32Type>())
            .col("value", array::fill_utf8("value".to_string()))
            .into_reader_rows(RowCount::from(1_000), BatchCount::from(10));

        let write_params = WriteParams {
            ..Default::default()
        };
        Dataset::write(data, "memory://", Some(write_params.clone()))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_diff_meta_no_transaction() {
        let ds = create_test_dataset().await;
        let result = ds.diff_meta(1).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_diff_meta_single_transaction() {
        let mut ds = create_test_dataset().await;
        ds.delete("key = 5").await.unwrap();

        let delta_struct = crate::dataset::delta::DatasetDelta {
            begin_version: 1,
            end_version: ds.version().version,
            base_dataset: ds.clone(),
        };
        let txs = delta_struct.list_transactions().await.unwrap();
        assert_eq!(txs.len(), 1);
        assert!(matches!(txs[0].operation, Operation::Delete { .. }));
    }

    #[tokio::test]
    async fn test_diff_meta_multiple_transactions() {
        let mut ds = create_test_dataset().await;
        ds.delete("key = 5").await.unwrap();
        ds.delete("key = 6").await.unwrap();

        let delta_struct = crate::dataset::delta::DatasetDelta {
            begin_version: 1,
            end_version: ds.version().version,
            base_dataset: ds.clone(),
        };
        let txs = delta_struct.list_transactions().await.unwrap();
        assert_eq!(txs.len(), 2);
    }

    #[tokio::test]
    async fn test_diff_meta_contains_deleted_transaction() {
        let clock = MockClock::new();

        clock.set_system_time(Duration::seconds(1));

        let mut ds = create_test_dataset().await;

        clock.set_system_time(Duration::seconds(2));

        ds.delete("key = 5").await.unwrap();
        ds.delete("key = 6").await.unwrap();
        ds.delete("key = 7").await.unwrap();

        clock.set_system_time(Duration::seconds(3));

        let end_version = ds.version().version;
        let base_dataset = ds.clone();

        clock.set_system_time(Duration::seconds(4));

        ds.cleanup_old_versions(Duration::seconds(1), Some(true), None)
            .await
            .expect("Cleanup old versions failed");

        clock.set_system_time(Duration::seconds(5));

        let delta_struct = crate::dataset::delta::DatasetDelta {
            begin_version: 1,
            end_version,
            base_dataset,
        };

        let result = delta_struct.list_transactions().await;
        match result {
            Err(lance_core::Error::VersionNotFound { message }) => {
                assert!(message.contains("Can not find version"));
            }
            _ => panic!("Expected VersionNotFound error."),
        }
    }
}
