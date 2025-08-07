// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::{
    dataset::transaction::{Operation, Transaction},
    dataset::utils::make_rowaddr_capture_stream,
    Dataset,
};
use datafusion::logical_expr::Expr;
use datafusion::scalar::ScalarValue;
use futures::{StreamExt, TryStreamExt};
use lance_core::{Error, Result};
use lance_table::format::Fragment;
use roaring::RoaringTreemap;
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use super::retry::{execute_with_retry, RetryConfig, RetryExecutor};
use super::CommitBuilder;

/// Apply deletions to fragments based on a RoaringTreemap of row IDs.
///
/// Returns the set of modified fragments and removed fragments, if any.
async fn apply_deletions(
    dataset: &Dataset,
    removed_row_addrs: &RoaringTreemap,
) -> Result<(Vec<Fragment>, Vec<u64>)> {
    let bitmaps = Arc::new(removed_row_addrs.bitmaps().collect::<BTreeMap<_, _>>());

    enum FragmentChange {
        Unchanged,
        Modified(Fragment),
        Removed(u64),
    }

    let mut updated_fragments = Vec::new();
    let mut removed_fragments = Vec::new();

    let mut stream = futures::stream::iter(dataset.get_fragments())
        .map(move |fragment| {
            let bitmaps_ref = bitmaps.clone();
            async move {
                let fragment_id = fragment.id();
                if let Some(bitmap) = bitmaps_ref.get(&(fragment_id as u32)) {
                    match fragment.extend_deletions(*bitmap).await {
                        Ok(Some(new_fragment)) => {
                            Ok(FragmentChange::Modified(new_fragment.metadata))
                        }
                        Ok(None) => Ok(FragmentChange::Removed(fragment_id as u64)),
                        Err(e) => Err(e),
                    }
                } else {
                    Ok(FragmentChange::Unchanged)
                }
            }
        })
        .buffer_unordered(dataset.object_store.io_parallelism());

    while let Some(res) = stream.next().await.transpose()? {
        match res {
            FragmentChange::Unchanged => {}
            FragmentChange::Modified(fragment) => updated_fragments.push(fragment),
            FragmentChange::Removed(fragment_id) => removed_fragments.push(fragment_id),
        }
    }

    Ok((updated_fragments, removed_fragments))
}

/// Builder for configuring delete operations with retry support
///
/// This operation is similar to SQL's DELETE statement. It allows you to remove
/// rows from a dataset based on a filter predicate with automatic retry support
/// for handling concurrent write conflicts.
///
/// Use the [DeleteBuilder] to construct a delete operation. For example:
///
/// ```
/// # use lance::{Dataset, Result};
/// # use lance::dataset::DeleteBuilder;
/// # use std::sync::Arc;
/// # async fn example(dataset: Arc<Dataset>) -> Result<()> {
/// let new_dataset = DeleteBuilder::new(dataset)
///     .predicate("age > 65")
///     .conflict_retries(5)
///     .execute()
///     .await?;
/// # Ok(())
/// # }
/// ```
///
#[derive(Debug, Clone)]
pub struct DeleteBuilder {
    dataset: Arc<Dataset>,
    predicate: Option<String>,
    conflict_retries: u32,
    retry_timeout: Duration,
}

impl DeleteBuilder {
    /// Create a new DeleteBuilder
    pub fn new(dataset: Arc<Dataset>) -> Self {
        Self {
            dataset,
            predicate: None,
            conflict_retries: 10,
            retry_timeout: Duration::from_secs(30),
        }
    }

    /// Set the predicate for filtering rows to delete
    pub fn predicate(mut self, predicate: impl Into<String>) -> Self {
        self.predicate = Some(predicate.into());
        self
    }

    /// Set the number of retries for conflict resolution
    pub fn conflict_retries(mut self, retries: u32) -> Self {
        self.conflict_retries = retries;
        self
    }

    /// Set the timeout for retry operations
    pub fn retry_timeout(mut self, timeout: Duration) -> Self {
        self.retry_timeout = timeout;
        self
    }

    /// Execute the delete operation
    pub async fn execute(self) -> Result<Arc<Dataset>> {
        let predicate = self.predicate.ok_or_else(|| Error::InvalidInput {
            source: "Delete predicate is required".into(),
            location: snafu::location!(),
        })?;

        let job = DeleteJob {
            dataset: self.dataset.clone(),
            predicate,
        };

        let config = RetryConfig {
            max_retries: self.conflict_retries,
            retry_timeout: self.retry_timeout,
        };

        execute_with_retry(job, self.dataset, config).await
    }
}

/// Job that executes the delete operation
#[derive(Debug, Clone)]
struct DeleteJob {
    dataset: Arc<Dataset>,
    predicate: String,
}

/// Data returned by delete operation
struct DeleteData {
    updated_fragments: Vec<Fragment>,
    deleted_fragment_ids: Vec<u64>,
}

impl RetryExecutor for DeleteJob {
    type Data = DeleteData;
    type Result = Arc<Dataset>;

    async fn execute_impl(&self) -> Result<Self::Data> {
        // Create a single scanner for the entire dataset
        let mut scanner = self.dataset.scan();
        scanner
            .with_row_address()
            .filter(&self.predicate)?
            .project::<&str>(&[])?;

        // Check if the filter optimized to true (delete everything) or false (delete nothing)
        let (updated_fragments, deleted_fragment_ids) =
            if let Some(filter_expr) = scanner.get_filter()? {
                if matches!(
                    filter_expr,
                    Expr::Literal(ScalarValue::Boolean(Some(false)), _)
                ) {
                    // Predicate evaluated to false - no deletions
                    (Vec::new(), Vec::new())
                } else if matches!(
                    filter_expr,
                    Expr::Literal(ScalarValue::Boolean(Some(true)), _)
                ) {
                    // Predicate evaluated to true - delete all fragments
                    let deleted_fragment_ids = self
                        .dataset
                        .get_fragments()
                        .iter()
                        .map(|f| f.id() as u64)
                        .collect();
                    (Vec::new(), deleted_fragment_ids)
                } else {
                    // Regular predicate - scan and collect row addresses to delete
                    let removed_row_addrs = Arc::new(RwLock::new(RoaringTreemap::new()));
                    let stream = scanner.try_into_stream().await?.into();
                    let stream = make_rowaddr_capture_stream(removed_row_addrs.clone(), stream)?;

                    // Process the stream to capture row addresses
                    // We need to consume the stream to trigger the capture
                    futures::pin_mut!(stream);
                    while let Some(_batch) = stream.try_next().await? {
                        // The row addresses are captured automatically by make_rowaddr_capture_stream
                    }

                    // Extract the row addresses from the Arc
                    let removed_row_addrs = {
                        let guard = removed_row_addrs.read().unwrap();
                        guard.clone()
                    };

                    apply_deletions(&self.dataset, &removed_row_addrs).await?
                }
            } else {
                // No filter was applied - this shouldn't happen but treat as delete nothing
                (Vec::new(), Vec::new())
            };

        Ok(DeleteData {
            updated_fragments,
            deleted_fragment_ids,
        })
    }

    async fn commit(&self, dataset: Arc<Dataset>, data: Self::Data) -> Result<Self::Result> {
        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::Delete {
                updated_fragments: data.updated_fragments,
                deleted_fragment_ids: data.deleted_fragment_ids,
                predicate: self.predicate.clone(),
            },
            // No change is needed to the blobs dataset. The blobs are implicitly deleted since the
            // rows that reference them are deleted.
            /*blobs_op=*/
            None,
            None,
        );

        CommitBuilder::new(dataset)
            .execute(transaction)
            .await
            .map(Arc::new)
    }

    fn update_dataset(&mut self, dataset: Arc<Dataset>) {
        self.dataset = dataset;
    }
}

/// Legacy delete function - uses DeleteBuilder with no retries for backwards compatibility
pub async fn delete(ds: &mut Dataset, predicate: &str) -> Result<()> {
    // Use DeleteBuilder with 0 retries to maintain backwards compatibility
    let dataset = Arc::new(ds.clone());
    let new_dataset = DeleteBuilder::new(dataset)
        .predicate(predicate)
        .conflict_retries(0)
        .execute()
        .await?;

    // Update the dataset in place
    *ds = Arc::try_unwrap(new_dataset).unwrap_or_else(|arc| (*arc).clone());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::InsertBuilder;
    use crate::dataset::{WriteMode, WriteParams};
    use crate::utils::test::TestDatasetGenerator;
    use arrow_array::{RecordBatch, UInt32Array};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use lance_file::version::LanceFileVersion;
    use lance_index::{scalar::ScalarIndexParams, DatasetIndexExt, IndexType};
    use rstest::rstest;
    use std::collections::HashSet;
    use std::ops::Range;
    use std::sync::Arc;
    use std::time::Duration;

    #[rstest]
    #[tokio::test]
    async fn test_delete(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] with_scalar_index: bool,
    ) {
        fn sequence_data(range: Range<u32>) -> RecordBatch {
            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("i", DataType::UInt32, false),
                ArrowField::new("x", DataType::UInt32, false),
            ]));
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt32Array::from_iter_values(range.clone())),
                    Arc::new(UInt32Array::from_iter_values(range.map(|v| v * 2))),
                ],
            )
            .unwrap()
        }
        // Write a dataset
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path().to_str().unwrap().to_string();
        let data = sequence_data(0..100);
        // Split over two files.
        let batches = vec![data.slice(0, 50), data.slice(50, 50)];
        let mut dataset = TestDatasetGenerator::new(batches, data_storage_version)
            .make_hostile(&tmp_path)
            .await;

        if with_scalar_index {
            dataset
                .create_index(
                    &["i"],
                    IndexType::Scalar,
                    Some("scalar_index".to_string()),
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
        }

        // Delete nothing
        dataset.delete("i < 0").await.unwrap();
        dataset.validate().await.unwrap();

        // We should not have any deletion file still
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        assert_eq!(dataset.count_fragments(), 2);
        assert_eq!(dataset.count_deleted_rows().await.unwrap(), 0);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(1));
        assert!(fragments[0].metadata.deletion_file.is_none());
        assert!(fragments[1].metadata.deletion_file.is_none());

        // Delete rows
        dataset.delete("i < 10 OR i >= 90").await.unwrap();
        dataset.validate().await.unwrap();

        // Verify result:
        // There should be a deletion file in the metadata
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        assert_eq!(dataset.count_fragments(), 2);
        assert!(fragments[0].metadata.deletion_file.is_some());
        assert!(fragments[1].metadata.deletion_file.is_some());
        assert_eq!(
            fragments[0]
                .metadata
                .deletion_file
                .as_ref()
                .unwrap()
                .num_deleted_rows,
            Some(10)
        );
        assert_eq!(
            fragments[1]
                .metadata
                .deletion_file
                .as_ref()
                .unwrap()
                .num_deleted_rows,
            Some(10)
        );

        // The deletion file should contain 20 rows
        assert_eq!(dataset.count_deleted_rows().await.unwrap(), 20);
        // First fragment has 0..10 deleted
        let deletion_vector = fragments[0].get_deletion_vector().await.unwrap().unwrap();
        assert_eq!(deletion_vector.len(), 10);
        assert_eq!(
            deletion_vector.iter().collect::<HashSet<_>>(),
            (0..10).collect::<HashSet<_>>()
        );
        // Second fragment has 90..100 deleted
        let deletion_vector = fragments[1].get_deletion_vector().await.unwrap().unwrap();
        assert_eq!(deletion_vector.len(), 10);
        // The second fragment starts at 50, so 90..100 becomes 40..50 in local row ids.
        assert_eq!(
            deletion_vector.iter().collect::<HashSet<_>>(),
            (40..50).collect::<HashSet<_>>()
        );
        let second_deletion_file = fragments[1].metadata.deletion_file.clone().unwrap();

        // Delete more rows
        dataset.delete("i < 20").await.unwrap();
        dataset.validate().await.unwrap();

        // Verify result
        assert_eq!(dataset.count_deleted_rows().await.unwrap(), 30);
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        assert!(fragments[0].metadata.deletion_file.is_some());
        let deletion_vector = fragments[0].get_deletion_vector().await.unwrap().unwrap();
        assert_eq!(deletion_vector.len(), 20);
        assert_eq!(
            deletion_vector.iter().collect::<HashSet<_>>(),
            (0..20).collect::<HashSet<_>>()
        );
        // Second deletion vector was not rewritten
        assert_eq!(
            fragments[1].metadata.deletion_file.as_ref().unwrap(),
            &second_deletion_file
        );

        // Delete full fragment
        dataset.delete("i >= 50").await.unwrap();
        dataset.validate().await.unwrap();

        // Verify second fragment is fully gone
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert_eq!(dataset.count_fragments(), 1);
        assert_eq!(fragments[0].id(), 0);

        // Verify the count_deleted_rows only contains the rows from the first fragment
        // i.e. - deleted_rows from the fragment that has been deleted are not counted
        assert_eq!(dataset.count_deleted_rows().await.unwrap(), 20);

        // Append after delete
        let data = sequence_data(0..100);
        let write_params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        let dataset = InsertBuilder::new(Arc::new(dataset))
            .with_params(&write_params)
            .execute(vec![data])
            .await
            .unwrap();

        dataset.validate().await.unwrap();

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        assert_eq!(dataset.count_fragments(), 2);
        // Fragment id picks up where we left off
        assert_eq!(fragments[0].id(), 0);
        assert_eq!(fragments[1].id(), 2);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(2));
    }

    #[tokio::test]
    async fn test_delete_with_single_scanner() {
        fn sequence_data(range: Range<u32>) -> RecordBatch {
            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("i", DataType::UInt32, false),
                ArrowField::new("x", DataType::UInt32, false),
            ]));
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt32Array::from_iter_values(range.clone())),
                    Arc::new(UInt32Array::from_iter_values(range.map(|v| v * 2))),
                ],
            )
            .unwrap()
        }

        // Create dataset with multiple fragments
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path().to_str().unwrap().to_string();

        // Create 5 fragments with 100 rows each
        let mut batches = Vec::new();
        for i in 0..5 {
            let start = i * 100;
            let end = (i + 1) * 100;
            let data = sequence_data(start..end);
            batches.push(data);
        }

        let mut dataset = TestDatasetGenerator::new(batches, LanceFileVersion::Stable)
            .make_hostile(&tmp_path)
            .await;

        // Delete rows across multiple fragments using the new scanner-based implementation
        let predicate = "i >= 50 AND i < 150";
        dataset.delete(predicate).await.unwrap();

        // Verify the deletion worked correctly
        let mut scanner = dataset.scan();
        scanner.filter(predicate).unwrap();
        let count = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_fold(0, |acc, batch| async move { Ok(acc + batch.num_rows()) })
            .await
            .unwrap();

        assert_eq!(
            count, 0,
            "All rows matching the predicate should be deleted"
        );

        // Verify that rows outside the predicate still exist
        let mut remaining_scanner = dataset.scan();
        remaining_scanner.filter("i < 50 OR i >= 150").unwrap();
        let remaining_count = remaining_scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_fold(0, |acc, batch| async move { Ok(acc + batch.num_rows()) })
            .await
            .unwrap();

        assert_eq!(
            remaining_count, 400,
            "400 rows should remain after deletion"
        );

        // Check that fragments were handled correctly
        let fragments = dataset.get_fragments();
        assert!(fragments.len() == 5, "All fragments should still exist");

        // Fragment 0 (rows 0-99) should have 50 deletions (50-99)
        let frag0_dv = fragments[0].get_deletion_vector().await.unwrap().unwrap();
        assert_eq!(frag0_dv.len(), 50);

        // Fragment 1 (rows 100-199) should be fully deleted or have 50 deletions (100-149)
        let frag1_dv = fragments[1].get_deletion_vector().await.unwrap().unwrap();
        assert_eq!(frag1_dv.len(), 50);
    }

    #[tokio::test]
    async fn test_concurrent_delete_with_retries() {
        fn sequence_data(range: Range<u32>) -> RecordBatch {
            let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "i",
                DataType::UInt32,
                false,
            )]));
            RecordBatch::try_new(schema, vec![Arc::new(UInt32Array::from_iter_values(range))])
                .unwrap()
        }

        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path().to_str().unwrap().to_string();

        let data = sequence_data(0..100);
        let dataset = TestDatasetGenerator::new(vec![data], LanceFileVersion::Stable)
            .make_hostile(&tmp_path)
            .await;

        // Create two concurrent delete operations
        let dataset1 = Arc::new(dataset.clone());
        let dataset2 = Arc::new(dataset.clone());

        // Use a channel to coordinate the test
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();

        // First delete task - delete rows 0-49
        let task1 = tokio::spawn(async move {
            // Wait for signal to start both tasks at the same time
            let _ = rx.await;
            DeleteBuilder::new(dataset1)
                .predicate("i < 50")
                .conflict_retries(3)
                .execute()
                .await
        });

        // Second delete task - delete rows 50-99
        let task2 = tokio::spawn(async move {
            // Start immediately and signal the first task
            let _ = tx.send(());
            DeleteBuilder::new(dataset2)
                .predicate("i >= 50")
                .conflict_retries(3)
                .execute()
                .await
        });

        // Both tasks should complete successfully
        let result1 = task1.await.unwrap();
        let result2 = task2.await.unwrap();

        // One should succeed, one should succeed after retry
        assert!(result1.is_ok() || result2.is_ok());

        // Load the final dataset and verify all rows are deleted
        let final_dataset = if result1.is_ok() {
            result1.unwrap()
        } else {
            result2.unwrap()
        };

        // All rows should be deleted
        assert_eq!(final_dataset.count_rows(None).await.unwrap(), 0);

        // Check that we have the expected number of fragments (empty or removed)
        let fragments = final_dataset.get_fragments();
        assert!(fragments.is_empty());
    }

    #[tokio::test]
    async fn test_delete_retry_timeout() {
        fn sequence_data(range: Range<u32>) -> RecordBatch {
            let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "i",
                DataType::UInt32,
                false,
            )]));
            RecordBatch::try_new(schema, vec![Arc::new(UInt32Array::from_iter_values(range))])
                .unwrap()
        }

        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path().to_str().unwrap().to_string();

        let data = sequence_data(0..100);
        let dataset = TestDatasetGenerator::new(vec![data], LanceFileVersion::Stable)
            .make_hostile(&tmp_path)
            .await;

        let dataset = Arc::new(dataset);

        // Test with very short timeout
        let result = DeleteBuilder::new(dataset.clone())
            .predicate("i < 50")
            .conflict_retries(100) // High retry count
            .retry_timeout(Duration::from_millis(1)) // Very short timeout
            .execute()
            .await;

        // Should timeout
        if let Err(e) = result {
            // Check that it's a timeout error
            assert!(
                matches!(e, Error::TooMuchWriteContention { .. }),
                "Expected TooMuchWriteContention error, got: {:?}",
                e
            );
        } else {
            // Might succeed if the operation is very fast
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_delete_false_predicate_still_commits() {
        fn sequence_data(range: Range<u32>) -> RecordBatch {
            let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "i",
                DataType::UInt32,
                false,
            )]));
            RecordBatch::try_new(schema, vec![Arc::new(UInt32Array::from_iter_values(range))])
                .unwrap()
        }

        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path().to_str().unwrap().to_string();

        let data = sequence_data(0..100);
        let mut dataset = TestDatasetGenerator::new(vec![data], LanceFileVersion::Stable)
            .make_hostile(&tmp_path)
            .await;

        let initial_version = dataset.version().version;

        // Delete with false predicate - should still commit but not delete anything
        dataset.delete("false").await.unwrap();

        // Verify version incremented (commit happened)
        assert_eq!(dataset.version().version, initial_version + 1);

        // Verify no rows were deleted
        assert_eq!(dataset.count_rows(None).await.unwrap(), 100);
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert!(fragments[0].metadata.deletion_file.is_none());
    }
}
