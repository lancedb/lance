// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::transaction::Transaction;
use crate::Dataset;
use crate::Result;
use arrow_array::RecordBatch;
use futures::stream::{self, BoxStream, StreamExt, TryStreamExt};
use futures::FutureExt;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::Error;
use lance_table::format::{Fragment, RowLatestUpdateVersionSequence};
use serde::{Deserialize, Serialize};
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
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 4,
            include_data: true,
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
        if fragment.row_latest_update_version_meta.is_none() {
            return Ok(false);
        }

        // // Use min/max version optimization if available
        // if let (Some(min_ver), Some(max_ver)) = (
        //     fragment.min_latest_update_version,
        //     fragment.max_latest_update_version,
        // ) {
        //     // FIXED: Only skip if max_ver is strictly less than compared_version
        //     // This prevents skipping fragments when max_ver equals compared_version,
        //     // which could happen when fragment metadata is stale after merge operations
        //     if max_ver < self.compared_version {
        //         return Ok(true);
        //     }
        //
        //     // If all records have versions > compared_version, we need to analyze
        //     // (they could be inserts or updates)
        //     if min_ver > self.compared_version {
        //         return Ok(false);
        //     }
        // }

        // Fragment spans the compared version or has equal max version, need to analyze
        Ok(false)
    }

    /// Analyze a single fragment for differences
    #[instrument(level = "debug", skip_all)]
    pub async fn analyze_fragment(&self, fragment: &Fragment) -> Result<Vec<DiffRecord>> {
        // Check if we can skip this fragment using version optimization
        if self.should_skip_fragment(fragment)? {
            return Ok(vec![]);
        }

        let mut diff_records = Vec::new();

        // Load the row version sequence for this fragment
        if let Some(version_meta) = &fragment.row_latest_update_version_meta {
            let version_sequence = version_meta.load_sequence()?;

            // Find updated and inserted records
            let updated_inserted = self
                .find_updated_and_inserted_records(fragment, &version_sequence)
                .await?;
            diff_records.extend(updated_inserted);
        } else {
            // Fragment without version metadata
            // let new_fragment_records = self.handle_new_fragment(fragment).await?;
            // diff_records.extend(new_fragment_records);
            panic!("In stable row id mode, fragment must have the row latest update version meta!")
        }

        // Note: Deleted records are not handled at the fragment level
        // since new fragments created by updates don't contain information about deletions

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

            // Skip rows that were updated after the current dataset version being analyzed
            // This ensures we only include changes that actually occurred between the compared version
            // and the current version we're diffing to
            if version > self.current_dataset.manifest.version {
                continue;
            }

            // Determine operation type by checking if the row existed in the old dataset
            // Try to get the record from the old dataset to see if it exists
            let old_record_result = self.get_record_data(row_id, &self.old_dataset).await;
            let row_exists_in_old = old_record_result.is_ok();

            let operation = if row_exists_in_old {
                // If the row exists in both versions, we need to check if it actually changed
                let old_record = old_record_result.unwrap();
                let new_record = self.get_record_data(row_id, &self.current_dataset).await?;

                // Compare the records to see if they're actually different
                let is_actually_updated = self.records_are_different(&old_record, &new_record)?;

                if is_actually_updated {
                    DiffOperation::Update
                } else {
                    // Record exists in both versions but hasn't changed, skip it
                    continue;
                }
            } else {
                DiffOperation::Insert
            };

            let old_data = if self.config.include_data && matches!(operation, DiffOperation::Update)
            {
                self.get_record_data(row_id, &self.old_dataset).await.ok()
            } else {
                None
            };

            let new_data = if self.config.include_data {
                Some(self.get_record_data(row_id, &self.current_dataset).await?)
            } else {
                None
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
                format!(
                    "Expected 1 row for row ID {}, got {}",
                    row_id,
                    batch.num_rows()
                ),
                Default::default(),
            ));
        }

        Ok(batch)
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

        // Collect all records first, then deduplicate
        let dedup_stream = stream
            .try_collect::<Vec<_>>()
            .then(|result| async move {
                match result {
                    Ok(records) => {
                        let mut seen_row_ids = std::collections::HashSet::new();
                        let mut deduped = Vec::new();

                        for record in records {
                            if seen_row_ids.insert(record.row_id) {
                                deduped.push(Ok(record));
                            }
                        }

                        futures::stream::iter(deduped)
                    }
                    Err(e) => futures::stream::iter(vec![Err(e)]),
                }
            })
            .flatten_stream();

        Box::pin(dedup_stream)
    }

    /// Compare two record batches to see if they're different
    fn records_are_different(
        &self,
        old_record: &RecordBatch,
        new_record: &RecordBatch,
    ) -> Result<bool> {
        // First check if schemas are different
        if old_record.schema() != new_record.schema() {
            return Ok(true);
        }

        // Check if number of rows is different
        if old_record.num_rows() != new_record.num_rows() {
            return Ok(true);
        }

        // Compare each column
        for i in 0..old_record.num_columns() {
            let old_column = old_record.column(i);
            let new_column = new_record.column(i);

            // Use Arrow's equality comparison
            if old_column != new_column {
                return Ok(true);
            }
        }

        Ok(false)
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
    /// Maximum number of records to process in parallel
    max_concurrency: usize,
    /// Whether to include the actual record data in diff results
    include_data: bool,
    /// Batch size for processing records
    batch_size: usize,
}

impl DatasetDiffBuilder {
    /// Create a new dataset diff builder
    pub fn new(dataset: Arc<Dataset>, compared_version: u64) -> Self {
        Self {
            dataset,
            compared_version,
            max_concurrency: 4,
            include_data: true,
            batch_size: 1000,
        }
    }

    /// Set the maximum concurrency for diff operations
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency;
        self
    }

    /// Set whether to include actual record data in diff results
    pub fn with_include_data(mut self, include_data: bool) -> Self {
        self.include_data = include_data;
        self
    }

    /// Set the batch size for processing records
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Execute the diff operation and return a stream of diff records
    pub async fn execute(self) -> Result<DiffRecordStream> {
        // Validate preconditions
        self.validate_preconditions().await?;

        // Load the old version of the dataset
        let old_dataset = Arc::new(self.dataset.checkout_version(self.compared_version).await?);

        let config = DiffConfig {
            max_concurrency: self.max_concurrency,
            include_data: self.include_data,
        };

        // Create the fragment diff analyzer
        let analyzer =
            FragmentDiffAnalyzer::new(self.dataset, old_dataset, self.compared_version, config);

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
    use crate::dataset::delta::{DatasetDiffBuilder, DiffOperation};
    use crate::dataset::{Dataset, WriteMode, WriteParams};
    use arrow_array::types::Int32Type;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use futures::TryStreamExt;
    use lance_datagen::{array, BatchCount, RowCount};
    use std::sync::Arc;

    /// Helper function to create a test dataset with stable row IDs enabled
    async fn create_test_dataset_with_stable_row_ids() -> Dataset {
        let data = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("name", array::fill_utf8("initial".to_string()))
            .col("value", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));

        let write_params = WriteParams {
            mode: WriteMode::Create,
            enable_stable_row_ids: true,
            ..Default::default()
        };

        Dataset::write(data, "memory://test_diff_dataset", Some(write_params))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_diff_merge_insert_sub_schema_update() {
        // Create initial dataset with stable row IDs
        let dataset = create_test_dataset_with_stable_row_ids().await;
        let initial_version = dataset.version().version;

        // Create merge insert data with sub-schema (only updating specific columns)
        // We'll update only the 'name' column, leaving 'value' unchanged
        let merge_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false), // key column for matching
            Field::new("name", DataType::Utf8, false), // column to update
                                                      // Note: 'value' column is intentionally omitted (sub-schema)
        ]));

        // Create merge data that will update records with id 10-14
        let merge_ids: Vec<i32> = (10..15).collect();
        let merge_names: Vec<String> = (10..15).map(|i| format!("merge_updated_{}", i)).collect();

        let merge_batch = arrow_array::RecordBatch::try_new(
            merge_schema.clone(),
            vec![
                Arc::new(Int32Array::from(merge_ids)),
                Arc::new(StringArray::from(merge_names)),
            ],
        )
        .unwrap();

        let merge_batches = arrow_array::RecordBatchIterator::new([Ok(merge_batch)], merge_schema);

        // Perform merge insert using MergeInsertBuilder with sub-schema update
        let mut merge_insert_builder = crate::dataset::MergeInsertBuilder::try_new(
            Arc::new(dataset.clone()),
            vec!["id".to_string()], // match on 'id' column
        )
        .unwrap();

        merge_insert_builder
            .when_matched(crate::dataset::WhenMatched::UpdateAll) // Update matching records
            .when_not_matched_by_source(crate::dataset::WhenNotMatchedBySource::Keep) // Keep non-matching target records
            .when_not_matched(crate::dataset::WhenNotMatched::InsertAll); // Insert new records if any

        let (updated_dataset, merge_stats) = merge_insert_builder
            .try_build()
            .unwrap()
            .execute_reader(Box::new(merge_batches))
            .await
            .unwrap();

        // Verify merge statistics
        assert_eq!(
            merge_stats.num_updated_rows, 5,
            "Should have updated 5 rows (id 10-14)"
        );
        assert_eq!(
            merge_stats.num_inserted_rows, 0,
            "Should have inserted 0 new rows"
        );

        // Execute diff operation to compare before and after merge insert
        let diff_builder = DatasetDiffBuilder::new(updated_dataset.clone(), initial_version)
            .with_include_data(true)
            .with_max_concurrency(1)
            .with_batch_size(10);

        let diff_stream = diff_builder.execute().await.unwrap();
        let diff_records: Vec<_> = diff_stream.try_collect().await.unwrap();

        // Analyze diff results
        let mut update_count = 0;
        let mut updated_ids = Vec::new();

        for record in &diff_records {
            match record.operation {
                DiffOperation::Update => {
                    update_count += 1;

                    // Verify the record has both old and new data
                    assert!(record.old_data.is_some(), "Update should have old data");
                    assert!(record.new_data.is_some(), "Update should have new data");

                    if let (Some(ref old_data), Some(ref new_data)) =
                        (&record.old_data, &record.new_data)
                    {
                        // Get ID from new data
                        let id_array = new_data
                            .column_by_name("id")
                            .unwrap()
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .unwrap();
                        let id_value = id_array.value(0);
                        updated_ids.push(id_value);

                        // Verify this is one of the expected updated records
                        assert!(
                            (10..15).contains(&id_value),
                            "Updated record should have id in range 10-14, got {}",
                            id_value
                        );

                        // Verify that 'name' column was updated
                        let old_name_array = old_data
                            .column_by_name("name")
                            .unwrap()
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap();
                        let old_name = old_name_array.value(0);

                        let new_name_array = new_data
                            .column_by_name("name")
                            .unwrap()
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap();
                        let new_name = new_name_array.value(0);

                        assert_eq!(old_name, "initial", "Old name should be 'initial'");
                        assert_eq!(
                            new_name,
                            format!("merge_updated_{}", id_value),
                            "New name should be 'merge_updated_{}'",
                            id_value
                        );

                        // Verify that 'value' column was NOT changed (sub-schema behavior)
                        let old_value_array = old_data
                            .column_by_name("value")
                            .unwrap()
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .unwrap();
                        let old_value = old_value_array.value(0);

                        let new_value_array = new_data
                            .column_by_name("value")
                            .unwrap()
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .unwrap();
                        let new_value = new_value_array.value(0);

                        assert_eq!(
                            old_value, new_value,
                            "Value column should remain unchanged in sub-schema update, but old={} != new={}",
                            old_value, new_value
                        );
                    }
                }
                DiffOperation::Insert => {
                    panic!("Should not have any insert operations in this test");
                }
                DiffOperation::Delete => {
                    panic!("Should not have any delete operations in this test");
                }
            }
        }

        // Final assertions
        assert_eq!(update_count, 5, "Should have exactly 5 update operations");

        updated_ids.sort();
        let expected_ids: Vec<i32> = (10..15).collect();
        assert_eq!(
            updated_ids, expected_ids,
            "Should have updated exactly IDs 10-14, got {:?}",
            updated_ids
        );
    }

    async fn append_data_to_dataset(mut dataset: Dataset, start_id: i32, count: usize) -> Dataset {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let ids: Vec<i32> = (start_id..start_id + count as i32).collect();
        let names: Vec<String> = (0..count).map(|i| format!("new_{}", i)).collect();
        let values: Vec<i32> = (start_id..start_id + count as i32)
            .map(|x| x * 10)
            .collect();

        let batch = arrow_array::RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
                Arc::new(Int32Array::from(values)),
            ],
        )
        .unwrap();

        let batches = arrow_array::RecordBatchIterator::new([Ok(batch)], schema);

        let write_params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };

        dataset.append(batches, Some(write_params)).await.unwrap();
        dataset
    }

    #[tokio::test]
    async fn test_diff_basic_insert_operations() {
        let initial_dataset = create_test_dataset_with_stable_row_ids().await;
        let initial_version = initial_dataset.version().version;

        // Append new data to create version 2
        let updated_dataset = append_data_to_dataset(initial_dataset, 200, 50).await;

        // Execute diff operation between versions
        let diff_builder = DatasetDiffBuilder::new(Arc::new(updated_dataset), initial_version)
            .with_include_data(true)
            .with_max_concurrency(2)
            .with_batch_size(10);

        let diff_stream = diff_builder.execute().await.unwrap();
        let diff_records: Vec<_> = diff_stream.try_collect().await.unwrap();

        // Verify diff results
        let mut insert_count = 0;
        let mut update_count = 0;
        let mut delete_count = 0;

        for result in diff_records {
            let record = result;
            match record.operation {
                DiffOperation::Insert => {
                    insert_count += 1;
                    assert!(record.old_data.is_none(), "Insert should have no old data");
                    assert!(record.new_data.is_some(), "Insert should have new data");
                }
                DiffOperation::Update => {
                    update_count += 1;
                    assert!(record.old_data.is_some(), "Update should have old data");
                    assert!(record.new_data.is_some(), "Update should have new data");
                }
                DiffOperation::Delete => {
                    delete_count += 1;
                    assert!(record.old_data.is_some(), "Delete should have old data");
                    assert!(record.new_data.is_none(), "Delete should have no new data");
                }
            }
        }

        // For this test, we expect only insert operations
        assert_eq!(insert_count, 50, "Should have 50 insert operations");
        assert_eq!(update_count, 0, "Should have no update operations");
        assert_eq!(delete_count, 0, "Should have no delete operations");
    }

    #[tokio::test]
    async fn test_diff_mixed_operations() {
        let mut dataset = create_test_dataset_with_stable_row_ids().await;
        let initial_version = dataset.version().version;

        // Perform update operation using UpdateBuilder
        // Update records with id 5-9 to have new names and values
        let update_result = crate::dataset::UpdateBuilder::new(Arc::new(dataset.clone()))
            .update_where("id >= 5 AND id <= 9")
            .unwrap()
            .set("name", "'updated_name'")
            .unwrap()
            .set("value", "id * 100")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        dataset = (*update_result.new_dataset).clone();

        // Append new data (insert operation)
        dataset = append_data_to_dataset(dataset, 300, 20).await;

        // Execute diff operation from initial version to final version
        let diff_builder = DatasetDiffBuilder::new(Arc::new(dataset.clone()), initial_version)
            .with_include_data(true) // Enable data inclusion to verify update content
            .with_max_concurrency(1)
            .with_batch_size(5);

        let diff_stream = diff_builder.execute().await.unwrap();
        let diff_records: Vec<_> = diff_stream.try_collect().await.unwrap();

        let mut insert_count = 0;
        let mut update_count = 0;
        let mut updated_ids = Vec::new();

        for result in diff_records {
            let record = result;
            match record.operation {
                DiffOperation::Insert => {
                    insert_count += 1;
                    assert!(record.old_data.is_none(), "Insert should have no old data");
                    assert!(record.new_data.is_some(), "Insert should have new data");
                }
                DiffOperation::Update => {
                    update_count += 1;
                    assert!(record.old_data.is_some(), "Update should have old data");
                    assert!(record.new_data.is_some(), "Update should have new data");

                    // Verify that this is one of the updated records (id 5-9)
                    if let Some(ref new_data) = record.new_data {
                        let id_array = new_data
                            .column_by_name("id")
                            .unwrap()
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .unwrap();
                        let id_value = id_array.value(0);
                        updated_ids.push(id_value);
                        assert!(
                            (5..=9).contains(&id_value),
                            "Updated record should have id in range 5-9, got {}",
                            id_value
                        );

                        // Verify the updated values
                        let name_array = new_data
                            .column_by_name("name")
                            .unwrap()
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap();
                        let name_value = name_array.value(0);
                        assert_eq!(
                            name_value, "updated_name",
                            "Updated name should be 'updated_name', got '{}'",
                            name_value
                        );

                        let value_array = new_data
                            .column_by_name("value")
                            .unwrap()
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .unwrap();
                        let value_value = value_array.value(0);
                        assert_eq!(
                            value_value,
                            id_value * 100,
                            "Updated value should be id * 100 = {}, got {}",
                            id_value * 100,
                            value_value
                        );
                    }
                }
                DiffOperation::Delete => todo!(),
            }
        }

        // Verify we have both insert and update operations, but no delete operations
        assert!(
            insert_count > 0,
            "Should have insert operations from appended data"
        );
        assert!(
            update_count > 0,
            "Should have update operations from UpdateBuilder"
        );

        // Specifically verify we have the expected number of updates (5 records updated: id 5,6,7,8,9)
        assert_eq!(update_count, 5, "Should have exactly 5 update operations");
        assert_eq!(
            update_result.rows_updated, 5,
            "UpdateBuilder should report 5 rows updated"
        );

        // Verify we have the expected number of inserts (20 new records)
        assert_eq!(insert_count, 20, "Should have exactly 20 insert operations");

        // Verify all expected IDs were updated
        updated_ids.sort();
        assert_eq!(
            updated_ids,
            vec![5, 6, 7, 8, 9],
            "Should have updated exactly IDs 5-9, got {:?}",
            updated_ids
        );
    }

    #[tokio::test]
    async fn test_diff_mixed_operations_not_in_latest_version() {
        let mut dataset = create_test_dataset_with_stable_row_ids().await;
        let _v1 = dataset.version().version;

        // First update（id 5-9，version 2）
        let update_result1 = crate::dataset::UpdateBuilder::new(Arc::new(dataset.clone()))
            .update_where("id >= 5 AND id <= 9")
            .unwrap()
            .set("name", "'updated_name1'")
            .unwrap()
            .set("value", "id * 100")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        dataset = (*update_result1.new_dataset).clone();
        let v2 = dataset.version().version;

        // Append data（id 300-319，version 3）
        dataset = append_data_to_dataset(dataset, 300, 20).await;
        let v3 = dataset.version().version;
        let dataset_v3 = dataset.clone();

        // Second update（id 7-8，version 4）
        let update_result2 = crate::dataset::UpdateBuilder::new(Arc::new(dataset.clone()))
            .update_where("id >= 7 AND id <= 8")
            .unwrap()
            .set("name", "'updated_name2'")
            .unwrap()
            .set("value", "id * 200")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        dataset = (*update_result2.new_dataset).clone();
        let _v4 = dataset.version().version;

        // diff v2 and v3（only insert）
        let diff_builder = DatasetDiffBuilder::new(Arc::new(dataset_v3.clone()), v2)
            .with_include_data(true)
            .with_max_concurrency(1)
            .with_batch_size(5);
        let diff_stream = diff_builder.execute().await.unwrap();
        let diff_records: Vec<_> = diff_stream.try_collect().await.unwrap();

        let insert_count = diff_records
            .iter()
            .filter(|r| r.operation == DiffOperation::Insert)
            .count();
        let update_count = diff_records
            .iter()
            .filter(|r| r.operation == DiffOperation::Update)
            .count();

        assert_eq!(
            insert_count, 20,
            "v2-v3 should only contain 20 records insert"
        );
        assert_eq!(update_count, 0, "v2-v3 should not update");

        // diff v3 and v4（only update）
        let diff_builder2 = DatasetDiffBuilder::new(Arc::new(dataset.clone()), v3)
            .with_include_data(true)
            .with_max_concurrency(1)
            .with_batch_size(5);
        let diff_stream2 = diff_builder2.execute().await.unwrap();
        let diff_records2: Vec<_> = diff_stream2.try_collect().await.unwrap();

        let insert_count2 = diff_records2
            .iter()
            .filter(|r| r.operation == DiffOperation::Insert)
            .count();
        let update_count2 = diff_records2
            .iter()
            .filter(|r| r.operation == DiffOperation::Update)
            .count();

        assert_eq!(insert_count2, 0, "v3-v4 should not contain insert");
        assert_eq!(
            update_count2, 2,
            "v3-v4 should contains 2 updated records（id 7,8）"
        );
    }

    #[tokio::test]
    async fn test_diff_error_handling_without_stable_row_ids() {
        let data = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("value", array::fill_utf8("test".to_string()))
            .into_reader_rows(RowCount::from(50), BatchCount::from(1));

        let write_params = WriteParams {
            mode: WriteMode::Create,
            enable_stable_row_ids: false, // Explicitly disable stable row IDs
            ..Default::default()
        };

        let dataset = Dataset::write(data, "memory://test_no_stable_ids", Some(write_params))
            .await
            .unwrap();

        // Try to execute diff operation - should fail
        let diff_builder = DatasetDiffBuilder::new(Arc::new(dataset.clone()), 1);
        let result = diff_builder.execute().await;

        assert!(result.is_err(), "Diff should fail without stable row IDs");
    }

    #[tokio::test]
    async fn test_diff_invalid_version_bounds() {
        let dataset = create_test_dataset_with_stable_row_ids().await;
        let current_version = dataset.version().version;

        // Test with version equal to current version (should fail)
        let diff_builder = DatasetDiffBuilder::new(Arc::new(dataset.clone()), current_version);
        let result = diff_builder.execute().await;
        assert!(
            result.is_err(),
            "Should fail when compared version equals current version"
        );

        // Test with version greater than current version (should fail)
        let diff_builder = DatasetDiffBuilder::new(Arc::new(dataset.clone()), current_version + 1);
        let result = diff_builder.execute().await;
        assert!(
            result.is_err(),
            "Should fail when compared version is greater than current version"
        );

        // Test with version 0 (should fail)
        let diff_builder = DatasetDiffBuilder::new(Arc::new(dataset.clone()), 0);
        let result = diff_builder.execute().await;
        assert!(result.is_err(), "Should fail when compared version is 0");
    }

    #[tokio::test]
    async fn test_diff_with_compaction() {
        let mut dataset = create_test_dataset_with_stable_row_ids().await;
        let v1 = dataset.version().version;

        // first append
        dataset = append_data_to_dataset(dataset, 200, 10).await;

        // second append
        dataset = append_data_to_dataset(dataset, 300, 15).await;

        // first update
        let update_result1 = crate::dataset::UpdateBuilder::new(Arc::new(dataset.clone()))
            .update_where("id >= 5 AND id <= 9")
            .unwrap()
            .set("name", "'updated_before_compaction'")
            .unwrap()
            .set("value", "id * 50")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        dataset = (*update_result1.new_dataset).clone();

        // third append
        dataset = append_data_to_dataset(dataset, 400, 12).await;
        let v5 = dataset.version().version;

        let fragments_before_compaction = dataset.get_fragments().len();
        assert!(
            fragments_before_compaction > 1,
            "Should have multiple fragments before compaction"
        );

        // do compaction
        use crate::dataset::optimize::{compact_files, CompactionOptions};
        let compaction_options = CompactionOptions {
            target_rows_per_fragment: 50,
            max_rows_per_group: 1024,
            materialize_deletions: true,
            materialize_deletions_threshold: 0.0,
            ..Default::default()
        };

        let compaction_metrics = compact_files(&mut dataset, compaction_options, None)
            .await
            .unwrap();
        let v6_after_compaction = dataset.version().version;

        // verify compaction happened
        assert!(
            compaction_metrics.fragments_removed > 0,
            "Compaction should have removed some fragments"
        );
        assert!(
            compaction_metrics.fragments_added > 0,
            "Compaction should have added new fragments"
        );

        let fragments_after_compaction = dataset.get_fragments().len();
        assert!(
            fragments_after_compaction < fragments_before_compaction,
            "Should have fewer fragments after compaction: {} -> {}",
            fragments_before_compaction,
            fragments_after_compaction
        );

        // update after compaction
        let update_result2 = crate::dataset::UpdateBuilder::new(Arc::new(dataset.clone()))
            .update_where("id >= 200 AND id <= 203")
            .unwrap()
            .set("name", "'updated_after_compaction'")
            .unwrap()
            .set("value", "id * 75")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        dataset = (*update_result2.new_dataset).clone();

        // diff v1 ~ v5 (before compaction)
        let diff_builder = DatasetDiffBuilder::new(Arc::new(dataset.clone()), v1)
            .with_include_data(true)
            .with_max_concurrency(1)
            .with_batch_size(10);
        let diff_stream = diff_builder.execute().await.unwrap();
        let diff_records_v1_to_v5: Vec<_> = diff_stream.try_collect().await.unwrap();

        let insert_count_v1_to_v5 = diff_records_v1_to_v5
            .iter()
            .filter(|r| r.operation == DiffOperation::Insert)
            .count();
        let update_count_v1_to_v5 = diff_records_v1_to_v5
            .iter()
            .filter(|r| r.operation == DiffOperation::Update)
            .count();

        assert_eq!(
            insert_count_v1_to_v5, 37,
            "Should have 37(10+15+12) insert operations from v1 to v5"
        );
        assert_eq!(
            update_count_v1_to_v5, 5,
            "Should have 5 update operations from v1 to v5"
        );

        // diff v5 ~ v6 (compaction, should not have changes)
        let dataset_v6 = dataset.checkout_version(v6_after_compaction).await.unwrap();

        let diff_builder_compaction = DatasetDiffBuilder::new(Arc::new(dataset_v6), v5)
            .with_include_data(true)
            .with_max_concurrency(1)
            .with_batch_size(10);
        let diff_stream_compaction = diff_builder_compaction.execute().await.unwrap();
        let diff_records_compaction: Vec<_> = diff_stream_compaction.try_collect().await.unwrap();

        assert_eq!(
            diff_records_compaction.len(),
            0,
            "Compaction should not change data content, diff should be empty"
        );

        // 3. diff v6 ~ v7 (update after compaction)
        let diff_builder_post_compaction =
            DatasetDiffBuilder::new(Arc::new(dataset.clone()), v6_after_compaction)
                .with_include_data(true)
                .with_max_concurrency(1)
                .with_batch_size(10);
        let diff_stream_post_compaction = diff_builder_post_compaction.execute().await.unwrap();
        let diff_records_post_compaction: Vec<_> =
            diff_stream_post_compaction.try_collect().await.unwrap();

        let update_count_post_compaction = diff_records_post_compaction
            .iter()
            .filter(|r| r.operation == DiffOperation::Update)
            .count();
        let insert_count_post_compaction = diff_records_post_compaction
            .iter()
            .filter(|r| r.operation == DiffOperation::Insert)
            .count();

        assert_eq!(
            update_count_post_compaction, 4,
            "Should have 4 update operations after compaction (id 200-203)"
        );
        assert_eq!(
            insert_count_post_compaction, 0,
            "Should have no insert operations after compaction"
        );

        // diff v1 ~ v7
        let diff_builder_full = DatasetDiffBuilder::new(Arc::new(dataset.clone()), v1)
            .with_include_data(true)
            .with_max_concurrency(1)
            .with_batch_size(10);
        let diff_stream_full = diff_builder_full.execute().await.unwrap();
        let diff_records_full: Vec<_> = diff_stream_full.try_collect().await.unwrap();

        let mut updated_ids = Vec::new();
        let mut inserted_count = 0;
        let mut final_update_count = 0;

        for record in &diff_records_full {
            match record.operation {
                DiffOperation::Insert => inserted_count += 1,
                DiffOperation::Update => {
                    final_update_count += 1;
                    if let Some(ref new_data) = record.new_data {
                        let id_array = new_data
                            .column_by_name("id")
                            .unwrap()
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .unwrap();
                        let id_value = id_array.value(0);
                        updated_ids.push(id_value);
                    }
                }
                DiffOperation::Delete => {}
            }
        }

        assert_eq!(inserted_count, 37, "Should have 37 insert operations total");
        assert_eq!(
            final_update_count, 5,
            "Should have 5 update operations total (only id 5-9)"
        );

        updated_ids.sort();
        let expected_updated_ids = vec![5, 6, 7, 8, 9];
        assert_eq!(
            updated_ids, expected_updated_ids,
            "Updated IDs should include both pre and post compaction updates"
        );
    }
}
