// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::dataset::rowids::get_row_id_index;
use crate::{
    dataset::transaction::{Operation, Transaction},
    dataset::utils::make_rowid_capture_stream,
    Dataset,
};
use datafusion::logical_expr::Expr;
use datafusion::scalar::ScalarValue;
use futures::{StreamExt, TryStreamExt};
use lance_core::{Error, Result};
use lance_table::format::Fragment;
use roaring::RoaringTreemap;
use snafu::location;
use std::collections::BTreeMap;
use std::sync::Arc;
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
/// let new_dataset = DeleteBuilder::new(dataset, "age > 65")
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
    predicate: String,
    conflict_retries: u32,
    retry_timeout: Duration,
}

impl DeleteBuilder {
    /// Create a new DeleteBuilder
    pub fn new(dataset: Arc<Dataset>, predicate: impl Into<String>) -> Self {
        Self {
            dataset,
            predicate: predicate.into(),
            conflict_retries: 10,
            retry_timeout: Duration::from_secs(30),
        }
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
        let job = DeleteJob {
            dataset: self.dataset.clone(),
            predicate: self.predicate,
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
        scanner.with_row_id().filter(&self.predicate)?;

        // Check if the filter optimized to true (delete everything) or false (delete nothing)
        let (updated_fragments, deleted_fragment_ids) = if let Some(filter_expr) =
            scanner.get_filter()?
        {
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
                let deleted_fragment_ids = self.dataset.get_fragments().iter().map(|f| f.id() as u64).collect();
                (Vec::new(), deleted_fragment_ids)
            } else {
                // Regular predicate - scan and collect row addresses to delete
                let stream = scanner.try_into_stream().await?.into();
                let (stream, row_id_rx) =
                    make_rowid_capture_stream(stream, self.dataset.manifest.uses_move_stable_row_ids())?;

                // Process the stream to capture row addresses
                // We need to consume the stream to trigger the capture
                futures::pin_mut!(stream);
                while let Some(_batch) = stream.try_next().await? {
                    // The row addresses are captured automatically by make_rowid_capture_stream
                }

                // Extract the row addresses from the receiver
                let removed_row_ids = row_id_rx.try_recv().map_err(|err| Error::Internal {
                    message: format!("Failed to receive row ids: {}", err),
                    location: location!(),
                })?;
                let row_id_index = get_row_id_index(&self.dataset).await?;
                let removed_row_addrs = removed_row_ids.row_addrs(row_id_index.as_deref());

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
        let operation = Operation::Delete {
            updated_fragments: data.updated_fragments,
            deleted_fragment_ids: data.deleted_fragment_ids,
            predicate: self.predicate.clone(),
        };
        let transaction = Transaction::new(
            dataset.manifest.version,
            operation,
            /*blobs_op=*/ None,
            None,
        );
        let new_dataset = CommitBuilder::new(dataset).execute(transaction).await?;
        Ok(Arc::new(new_dataset))
    }

    fn update_dataset(&mut self, dataset: Arc<Dataset>) {
        self.dataset = dataset;
    }
}

/// Legacy delete function - uses DeleteBuilder with no retries for backwards compatibility
pub async fn delete(ds: &mut Dataset, predicate: &str) -> Result<()> {
    // Use DeleteBuilder with 0 retries to maintain backwards compatibility
    let dataset = Arc::new(ds.clone());
    let new_dataset = DeleteBuilder::new(dataset, predicate).execute().await?;

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
    use arrow::array::AsArray;
    use arrow::datatypes::UInt32Type;
    use arrow_array::{RecordBatch, UInt32Array};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use lance_file::version::LanceFileVersion;
    use lance_index::{scalar::ScalarIndexParams, DatasetIndexExt, IndexType};
    use rstest::rstest;
    use std::collections::HashSet;
    use std::ops::Range;
    use std::sync::Arc;

    #[rstest]
    #[tokio::test]
    async fn test_delete(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] with_scalar_index: bool,
    ) {
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path().to_str().unwrap().to_string();

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::UInt32, false),
            ArrowField::new("x", DataType::UInt32, false),
        ]));

        let batches: Vec<RecordBatch> = (0..20)
            .map(|i| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(UInt32Array::from_iter_values(i * 10..(i + 1) * 10)),
                        Arc::new(UInt32Array::from_iter_values(i * 10..(i + 1) * 10)),
                    ],
                )
                .unwrap()
            })
            .collect();

        let mut write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            ..Default::default()
        };
        write_params.data_storage_version = Some(data_storage_version);

        let batches = RecordBatch::concat(&batches).unwrap();
        let mut dataset = Dataset::write(
            vec![batches].into_iter(),
            &tmp_path,
            Some(write_params.clone()),
        )
        .await
        .unwrap();

        if with_scalar_index {
            dataset
                .create_index(&["i"], IndexType::Scalar, None, &ScalarIndexParams::default(), true)
                .await
                .unwrap();
        }

        let original_len = dataset.count_rows(None).await.unwrap();
        assert_eq!(original_len, 200);

        let original_fragments = dataset.get_fragments().to_vec();

        // Test false predicate
        delete(&mut dataset, "false").await.unwrap();
        dataset.validate().await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 200);

        // Delete some data
        delete(&mut dataset, "i >= 100").await.unwrap();
        dataset.validate().await.unwrap();

        let new_len = dataset.count_rows(None).await.unwrap();
        assert_eq!(new_len, 100);

        // Test that we can delete more data
        delete(&mut dataset, "i >= 50").await.unwrap();
        dataset.validate().await.unwrap();

        let newest_len = dataset.count_rows(None).await.unwrap();
        assert_eq!(newest_len, 50);

        // Double-check with a scan
        let values = dataset
            .scan()
            .try_into_batch()
            .await
            .unwrap()
            .column(0)
            .as_primitive::<UInt32Type>()
            .values()
            .to_vec();
        assert_eq!(values, (0..50).collect::<Vec<_>>());

        // Double check there are no duplicates
        assert_eq!(values.len(), values.into_iter().collect::<HashSet<_>>().len());

        if !with_scalar_index {
            // We should have the same fragments, but two should have deletion vectors
            let current_fragments = dataset.get_fragments();
            assert_eq!(original_fragments.len(), current_fragments.len());
            for (original, current) in original_fragments.iter().zip(current_fragments.iter()) {
                if original.id() == 0 {
                    // Fragment was not touched
                    assert_eq!(
                        original.physical_rows,
                        current.physical_rows,
                        "Fragment {}: expected {} rows but was {}",
                        original.id(),
                        original.physical_rows.unwrap(),
                        current.physical_rows.unwrap(),
                    );
                    assert!(current.deletion_file.is_none());
                } else {
                    // Fragment should have deletion vector
                    assert!(current.deletion_file.is_some());
                }
            }
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_delete_where_scalar_index_covers_predicate(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path().to_str().unwrap().to_string();

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::UInt32, false),
            ArrowField::new("x", DataType::UInt32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from_iter_values(0..200)),
                Arc::new(UInt32Array::from_iter_values(0..200)),
            ],
        )
        .unwrap();

        let mut write_params = WriteParams {
            max_rows_per_file: 50,
            max_rows_per_group: 10,
            ..Default::default()
        };
        write_params.data_storage_version = Some(data_storage_version);

        let mut dataset =
            Dataset::write(vec![batch].into_iter(), &tmp_path, Some(write_params.clone()))
                .await
                .unwrap();

        dataset
            .create_index(&["i"], IndexType::Scalar, None, &ScalarIndexParams::default(), true)
            .await
            .unwrap();

        let original_len = dataset.count_rows(None).await.unwrap();
        assert_eq!(original_len, 200);

        // Delete some data using predicate that's fully covered by scalar index
        delete(&mut dataset, "i >= 100 AND i <= 149").await.unwrap();
        dataset.validate().await.unwrap();

        let new_len = dataset.count_rows(None).await.unwrap();
        assert_eq!(new_len, 150);

        // Check that rows 100-149 are gone
        let remaining_values: Vec<u32> = dataset
            .scan()
            .try_into_batch()
            .await
            .unwrap()
            .column(0)
            .as_primitive::<UInt32Type>()
            .values()
            .to_vec();

        let expected: Vec<u32> = (0..100).chain(150..200).collect();
        assert_eq!(remaining_values, expected);

        // Fragments 2 (rows 100-149) should be fully deleted or have deletion vector
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 4);

        // Fragment 2 (rows 100-149) should be fully deleted or have 50 deletions (100-149)
        let frag2_dv = fragments[2].get_deletion_vector().await.unwrap().unwrap();
        assert_eq!(frag2_dv.len(), 50);
    }

    #[tokio::test]
    async fn test_concurrent_delete_with_retries() {
        use futures::future::try_join_all;
        use tokio::sync::Barrier;

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

        let concurrency = 3;
        let barrier = Arc::new(Barrier::new(concurrency as usize));
        let mut handles = Vec::new();

        // Create multiple concurrent delete operations targeting the same overlapping range
        // All tasks try to delete the same set of rows (0-49), creating maximum conflict
        for _i in 0..concurrency {
            let dataset_ref = Arc::new(dataset.clone());
            let barrier_ref = barrier.clone();

            let handle = tokio::spawn(async move {
                barrier_ref.wait().await;

                DeleteBuilder::new(dataset_ref, "i < 50") // All tasks delete the same rows
                    .conflict_retries(5)
                    .execute()
                    .await
            });
            handles.push(handle);
        }

        // All tasks should complete successfully with retry-based conflict resolution
        let results = try_join_all(handles).await.unwrap();

        // All delete operations should succeed
        for result in &results {
            assert!(
                result.is_ok(),
                "Delete operation should succeed with retries"
            );
        }

        // Get the final dataset from any successful result
        let final_dataset = results.into_iter().find_map(|r| r.ok()).unwrap();

        // Rows 0-49 should be deleted, rows 50-99 should remain
        assert_eq!(final_dataset.count_rows(None).await.unwrap(), 50);

        // Verify the remaining data is rows 50-99
        let data = final_dataset.scan().try_into_batch().await.unwrap();
        let remaining_values: Vec<u32> = data["i"].as_primitive::<UInt32Type>().values().to_vec();
        let expected: Vec<u32> = (50..100).collect();
        assert_eq!(remaining_values, expected);

        // Check that we have the expected fragment structure
        let fragments = final_dataset.get_fragments();
        assert_eq!(
            fragments.len(),
            1,
            "Should have one fragment with deletion vector"
        );

        // The fragment should have a deletion vector with 50 deleted rows
        let deletion_vector = fragments[0].get_deletion_vector().await.unwrap().unwrap();
        assert_eq!(deletion_vector.len(), 50, "Should have 50 deleted rows");

        // Check that the deletion vector contains rows 0-49
        let mut deleted_rows: Vec<u32> = deletion_vector.iter().collect();
        deleted_rows.sort();
        let expected_deleted: Vec<u32> = (0..50).collect();
        assert_eq!(deleted_rows, expected_deleted);
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
        let result = DeleteBuilder::new(dataset.clone(), "i < 50")
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
        let dataset = TestDatasetGenerator::new(vec![data], LanceFileVersion::Stable)
            .make_hostile(&tmp_path)
            .await;

        let dataset = Arc::new(dataset);
        let original_version = dataset.version().version;

        // Delete with false predicate should still create a new version
        let new_dataset = DeleteBuilder::new(dataset.clone(), "false")
            .execute()
            .await
            .unwrap();

        assert_eq!(new_dataset.version().version, original_version + 1);
        assert_eq!(new_dataset.count_rows(None).await.unwrap(), 100);
    }
}