// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::dataset::rowids::get_row_id_index;
use crate::dataset::scanner::ExprFilter;
use crate::{
    Dataset,
    dataset::transaction::{Operation, SchemaMetadataUpdates, Transaction},
    dataset::utils::make_rowid_capture_stream,
};
use datafusion::logical_expr::Expr;
use datafusion::scalar::ScalarValue;
use futures::{StreamExt, TryStreamExt};
use lance_core::{Error, ROW_ID, Result};
use lance_select::RowAddrTreeMap;
use lance_table::format::Fragment;
use roaring::RoaringTreemap;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use super::CommitBuilder;
use super::retry::{RetryConfig, RetryExecutor, execute_with_retry};

/// Result of a delete operation.
#[derive(Debug, Clone)]
pub struct DeleteResult {
    /// The new dataset after the delete operation.
    pub new_dataset: Arc<Dataset>,
    /// The number of rows that were deleted.
    pub num_deleted_rows: u64,
}

/// Result of a staged delete operation.
///
/// The returned transaction can be committed later with [`CommitBuilder`].
/// Pass `affected_rows` to [`CommitBuilder::with_affected_rows`] when present
/// to preserve row-level conflict resolution for concurrent deletes and updates.
#[derive(Debug, Clone)]
pub struct UncommittedDelete {
    /// The transaction to commit.
    pub transaction: Transaction,
    /// The row addresses affected by the delete, if available.
    pub affected_rows: Option<RowAddrTreeMap>,
    /// The number of rows that were deleted.
    pub num_deleted_rows: u64,
}

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
        Modified(Box<Fragment>),
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
                            Ok(FragmentChange::Modified(Box::new(new_fragment.metadata)))
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
            FragmentChange::Modified(fragment) => updated_fragments.push(*fragment),
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
/// let result = DeleteBuilder::new(dataset, "age > 65")
///     .conflict_retries(5)
///     .execute()
///     .await?;
/// println!("Deleted {} rows", result.num_deleted_rows);
/// # Ok(())
/// # }
/// ```
///
#[derive(Debug, Clone)]
pub struct DeleteBuilder {
    dataset: Arc<Dataset>,
    filter: ExprFilter,
    conflict_retries: u32,
    retry_timeout: Duration,
    /// Optional schema/field metadata patch published atomically with the Delete.
    schema_metadata_updates: Option<SchemaMetadataUpdates>,
}

impl DeleteBuilder {
    /// Create a new DeleteBuilder with a SQL predicate string
    pub fn new(dataset: Arc<Dataset>, predicate: impl Into<String>) -> Self {
        Self {
            dataset,
            filter: ExprFilter::Sql(predicate.into()),
            conflict_retries: 10,
            retry_timeout: Duration::from_secs(30),
            schema_metadata_updates: None,
        }
    }

    /// Create a new DeleteBuilder with a DataFusion expression filter
    pub fn from_expr(dataset: Arc<Dataset>, expr: Expr) -> Self {
        Self {
            dataset,
            filter: ExprFilter::Datafusion(expr),
            conflict_retries: 10,
            retry_timeout: Duration::from_secs(30),
            schema_metadata_updates: None,
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

    /// Attach schema/field metadata updates to publish atomically with the Delete.
    ///
    /// The patch is stored on the Delete [`Transaction`] and committed in the same
    /// version as the row deletions, so readers never observe deleted rows without
    /// the metadata (or vice versa). If zero rows match the predicate, the patch
    /// is not attached. Empty / no-op patches are rejected with
    /// [`Error::InvalidInput`].
    pub fn with_schema_metadata_updates(mut self, updates: SchemaMetadataUpdates) -> Result<Self> {
        updates.validate_non_empty()?;
        self.schema_metadata_updates = Some(updates);
        Ok(self)
    }

    /// Execute the delete operation
    pub async fn execute(self) -> Result<DeleteResult> {
        let job = DeleteJob {
            dataset: self.dataset.clone(),
            filter: self.filter,
            schema_metadata_updates: self.schema_metadata_updates,
        };

        let config = RetryConfig {
            max_retries: self.conflict_retries,
            retry_timeout: self.retry_timeout,
        };

        execute_with_retry(job, self.dataset, config).await
    }

    /// Execute the delete operation without committing the transaction.
    ///
    /// Use [`CommitBuilder`] to commit the returned transaction.
    ///
    /// # Example: Delete rows from a dataset
    ///
    /// ```rust
    /// use lance::dataset::{CommitBuilder, DeleteBuilder};
    ///
    /// # use std::sync::Arc;
    /// # use lance::Result;
    /// # use lance::dataset::Dataset;
    /// # async fn example(dataset: Arc<Dataset>) -> Result<()> {
    /// let staged_delete = DeleteBuilder::new(dataset.clone(), "age > 65")
    ///     .execute_uncommitted()
    ///     .await?;
    /// let mut commit_builder = CommitBuilder::new(dataset);
    /// if let Some(affected_rows) = staged_delete.affected_rows {
    ///     commit_builder = commit_builder.with_affected_rows(affected_rows);
    /// }
    /// commit_builder
    ///     .execute(staged_delete.transaction)
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn execute_uncommitted(self) -> Result<UncommittedDelete> {
        let job = DeleteJob {
            dataset: self.dataset,
            filter: self.filter,
            schema_metadata_updates: self.schema_metadata_updates,
        };
        let data = job.execute_impl().await?;
        let DeleteData {
            updated_fragments,
            deleted_fragment_ids,
            affected_rows,
            num_deleted_rows,
        } = data;
        let transaction = job.build_transaction(
            job.dataset.as_ref(),
            updated_fragments,
            deleted_fragment_ids,
            num_deleted_rows,
        )?;
        Ok(UncommittedDelete {
            transaction,
            affected_rows,
            num_deleted_rows,
        })
    }
}

/// Job that executes the delete operation
#[derive(Debug, Clone)]
struct DeleteJob {
    dataset: Arc<Dataset>,
    filter: ExprFilter,
    /// Copied from [`DeleteBuilder`] so retries and clones retain the patch.
    schema_metadata_updates: Option<SchemaMetadataUpdates>,
}

/// Data returned by delete operation
struct DeleteData {
    updated_fragments: Vec<Fragment>,
    deleted_fragment_ids: Vec<u64>,
    affected_rows: Option<RowAddrTreeMap>,
    num_deleted_rows: u64,
}

impl DeleteJob {
    fn build_transaction(
        &self,
        dataset: &Dataset,
        updated_fragments: Vec<Fragment>,
        deleted_fragment_ids: Vec<u64>,
        num_deleted_rows: u64,
    ) -> Result<Transaction> {
        let predicate = match &self.filter {
            ExprFilter::Sql(s) => s.clone(),
            ExprFilter::Datafusion(expr) => expr.to_string(),
            ExprFilter::Substrait(_) => {
                unreachable!("Substrait filters are not supported in DeleteBuilder")
            }
        };
        let operation = Operation::Delete {
            updated_fragments,
            deleted_fragment_ids,
            predicate,
        };
        let transaction = Transaction::new(dataset.manifest.version, operation, None);
        // Attach only when this attempt actually deleted rows; each retry uses
        // its own authoritative count from execute_impl.
        match &self.schema_metadata_updates {
            Some(updates) if num_deleted_rows > 0 => {
                transaction.with_schema_metadata_updates(updates.clone())
            }
            _ => Ok(transaction),
        }
    }
}

impl RetryExecutor for DeleteJob {
    type Data = DeleteData;
    type Result = DeleteResult;

    async fn execute_impl(&self) -> Result<Self::Data> {
        // Create a single scanner for the entire dataset
        let mut scanner = self.dataset.scan();
        scanner.with_row_id().project(&[ROW_ID])?;
        match &self.filter {
            ExprFilter::Sql(s) => {
                scanner.filter(s)?;
            }
            ExprFilter::Datafusion(expr) => {
                scanner.filter_expr(expr.clone());
            }
            ExprFilter::Substrait(_) => {
                unreachable!("Substrait filters are not supported in DeleteBuilder")
            }
        }

        // Check if the filter optimized to true (delete everything) or false (delete nothing)
        let (updated_fragments, deleted_fragment_ids, affected_rows, num_deleted_rows) =
            if let Some(filter_expr) = scanner.get_expr_filter()? {
                if matches!(
                    filter_expr,
                    Expr::Literal(ScalarValue::Boolean(Some(false)), _)
                ) {
                    // Predicate evaluated to false - no deletions
                    (Vec::new(), Vec::new(), Some(RowAddrTreeMap::new()), 0)
                } else if matches!(
                    filter_expr,
                    Expr::Literal(ScalarValue::Boolean(Some(true)), _)
                ) {
                    // Predicate evaluated to true - delete all fragments
                    let fragments = self.dataset.get_fragments();
                    let num_deleted_rows: u64 = fragments
                        .iter()
                        .map(|f| f.metadata.num_rows().unwrap_or(0) as u64)
                        .sum();
                    let deleted_fragment_ids = fragments.iter().map(|f| f.id() as u64).collect();

                    // When deleting everything, we don't have specific row addresses,
                    // so better not to emit affected rows.
                    (Vec::new(), deleted_fragment_ids, None, num_deleted_rows)
                } else {
                    // Regular predicate - scan and collect row addresses to delete
                    let stream = scanner.try_into_stream().await?.into();
                    let (stream, row_id_rx) = make_rowid_capture_stream(
                        stream,
                        self.dataset.manifest.uses_stable_row_ids(),
                    )?;

                    // Process the stream to capture row addresses
                    // We need to consume the stream to trigger the capture
                    futures::pin_mut!(stream);
                    while let Some(_batch) = stream.try_next().await? {
                        // The row addresses are captured automatically by make_rowid_capture_stream
                    }

                    // Extract the row addresses from the receiver
                    let removed_row_ids = row_id_rx.try_recv().map_err(|err| {
                        Error::internal(format!("Failed to receive row ids: {}", err))
                    })?;
                    let row_id_index = get_row_id_index(&self.dataset).await?;
                    let removed_row_addrs = removed_row_ids.row_addrs(row_id_index.as_deref());

                    let (fragments, deleted_ids) =
                        apply_deletions(&self.dataset, &removed_row_addrs).await?;
                    let num_deleted_rows = removed_row_addrs.len();
                    let affected_rows = RowAddrTreeMap::from(removed_row_addrs.as_ref().clone());
                    (
                        fragments,
                        deleted_ids,
                        Some(affected_rows),
                        num_deleted_rows,
                    )
                }
            } else {
                // No filter was applied - this shouldn't happen but treat as delete nothing
                (Vec::new(), Vec::new(), Some(RowAddrTreeMap::new()), 0)
            };

        Ok(DeleteData {
            updated_fragments,
            deleted_fragment_ids,
            affected_rows,
            num_deleted_rows,
        })
    }

    async fn commit(&self, dataset: Arc<Dataset>, data: Self::Data) -> Result<Self::Result> {
        let DeleteData {
            updated_fragments,
            deleted_fragment_ids,
            affected_rows,
            num_deleted_rows,
        } = data;
        let transaction = self.build_transaction(
            dataset.as_ref(),
            updated_fragments,
            deleted_fragment_ids,
            num_deleted_rows,
        )?;

        let mut builder = CommitBuilder::new(dataset);

        if let Some(affected_rows) = affected_rows {
            builder = builder.with_affected_rows(affected_rows);
        }

        let new_dataset = builder.execute(transaction).await.map(Arc::new)?;
        Ok(DeleteResult {
            new_dataset,
            num_deleted_rows,
        })
    }

    fn update_dataset(&mut self, dataset: Arc<Dataset>) {
        // Retry refreshes only the dataset reference; the metadata patch stays.
        self.dataset = dataset;
    }
}

/// Legacy delete function - uses DeleteBuilder with no retries for backwards compatibility
pub async fn delete(ds: &mut Dataset, predicate: &str) -> Result<DeleteResult> {
    // Use DeleteBuilder with 0 retries to maintain backwards compatibility
    let dataset = Arc::new(ds.clone());
    let result = DeleteBuilder::new(dataset, predicate).execute().await?;

    // Update the dataset in place
    *ds = Arc::try_unwrap(result.new_dataset.clone()).unwrap_or_else(|arc| (*arc).clone());
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::transaction::{SchemaMetadataUpdates, UpdateMap, UpdateMapEntry};
    use crate::dataset::{InsertBuilder, UpdateBuilder};
    use crate::dataset::{WriteMode, WriteParams};
    use crate::index::DatasetIndexExt;
    use crate::utils::test::TestDatasetGenerator;
    use arrow::array::AsArray;
    use arrow::datatypes::UInt32Type;
    use arrow_array::{RecordBatch, UInt32Array};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_file::version::LanceFileVersion;
    use lance_index::{IndexType, scalar::ScalarIndexParams};
    use rstest::rstest;
    use std::collections::{HashMap, HashSet};
    use std::ops::Range;
    use std::sync::Arc;

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
        let tmp_dir = TempStrDir::default();
        let tmp_path = tmp_dir.as_str().to_string();
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
        let result = dataset.delete("i < 0").await.unwrap();
        assert_eq!(result.num_deleted_rows, 0);
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
        let result = dataset.delete("i < 10 OR i >= 90").await.unwrap();
        assert_eq!(result.num_deleted_rows, 20);
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

        // Delete more rows (only 10 new rows since 0..10 already deleted)
        let result = dataset.delete("i < 20").await.unwrap();
        assert_eq!(result.num_deleted_rows, 10);
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

        // Delete full fragment (50 rows remaining in fragment 1, 10 already deleted)
        let result = dataset.delete("i >= 50").await.unwrap();
        assert_eq!(result.num_deleted_rows, 40);
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
        let tmp_dir = TempStrDir::default();
        let tmp_path = tmp_dir.as_str().to_string();

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

        let tmp_dir = TempStrDir::default();
        let tmp_path = tmp_dir.as_str().to_string();

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

    #[tokio::test]
    async fn test_delete_execute_uncommitted_preserves_affected_rows_for_rebase() {
        fn sequence_data(range: Range<u32>) -> RecordBatch {
            let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "i",
                DataType::UInt32,
                false,
            )]));
            RecordBatch::try_new(schema, vec![Arc::new(UInt32Array::from_iter_values(range))])
                .unwrap()
        }

        let tmp_dir = TempStrDir::default();
        let tmp_path = tmp_dir.as_str().to_string();

        let dataset = InsertBuilder::new(&tmp_path)
            .execute(vec![sequence_data(0..100)])
            .await
            .unwrap();
        let initial_version = dataset.version().version;

        let staged_delete = DeleteBuilder::new(Arc::new(dataset.clone()), "i < 10")
            .execute_uncommitted()
            .await
            .unwrap();

        let dataset_before_commit = Dataset::open(&tmp_path).await.unwrap();
        assert_eq!(dataset_before_commit.version().version, initial_version);
        assert_eq!(dataset_before_commit.count_rows(None).await.unwrap(), 100);

        assert_eq!(staged_delete.num_deleted_rows, 10);
        assert!(staged_delete.affected_rows.is_some());
        assert_eq!(staged_delete.transaction.read_version, initial_version);
        match &staged_delete.transaction.operation {
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids,
                predicate,
            } => {
                assert_eq!(predicate, "i < 10");
                assert_eq!(updated_fragments.len(), 1);
                assert!(deleted_fragment_ids.is_empty());
            }
            other => panic!("expected delete transaction, got {other:?}"),
        }

        DeleteBuilder::new(Arc::new(dataset.clone()), "i >= 10 AND i < 20")
            .execute()
            .await
            .unwrap();

        let mut commit_builder = CommitBuilder::new(&tmp_path);
        if let Some(affected_rows) = staged_delete.affected_rows {
            commit_builder = commit_builder.with_affected_rows(affected_rows);
        }
        let committed = commit_builder
            .execute(staged_delete.transaction)
            .await
            .unwrap();
        assert_eq!(committed.version().version, initial_version + 2);
        assert_eq!(committed.count_rows(None).await.unwrap(), 80);
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

        let tmp_dir = TempStrDir::default();
        let tmp_path = tmp_dir.as_str().to_string();

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
        let final_result = results.into_iter().find_map(|r| r.ok()).unwrap();
        let final_dataset = final_result.new_dataset;

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
    #[rstest]
    async fn test_delete_concurrency(#[values(false, true)] enable_stable_row_ids: bool) {
        use crate::{
            dataset::{InsertBuilder, ReadParams, WriteParams, builder::DatasetBuilder},
            session::Session,
            utils::test::ThrottledStoreWrapper,
        };
        use futures::future::try_join_all;
        use lance_io::object_store::ObjectStoreParams;
        use object_store::throttle::ThrottleConfig;
        use tokio::sync::Barrier;

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::UInt32,
            false,
        )]));
        let concurrency = 3;
        let initial_data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(UInt32Array::from_iter_values(
                0..(concurrency * 10),
            ))],
        )
        .unwrap();

        // Increase likelihood of contention by throttling the store
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                wait_list_per_call: Duration::from_millis(1),
                wait_get_per_call: Duration::from_millis(1),
                ..Default::default()
            },
        });
        let session = Arc::new(Session::default());

        let mut dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                store_params: Some(ObjectStoreParams {
                    object_store_wrapper: Some(throttled.clone()),
                    ..Default::default()
                }),
                session: Some(session.clone()),
                enable_stable_row_ids,
                ..Default::default()
            })
            .execute(vec![initial_data])
            .await
            .unwrap();

        let barrier = Arc::new(Barrier::new(concurrency as usize));
        let mut handles = Vec::new();
        for i in 0..concurrency {
            let session_ref = session.clone();
            let barrier_ref = barrier.clone();
            let throttled_ref = throttled.clone();
            let handle = tokio::task::spawn(async move {
                let dataset = DatasetBuilder::from_uri("memory://")
                    .with_read_params(ReadParams {
                        store_options: Some(ObjectStoreParams {
                            object_store_wrapper: Some(throttled_ref.clone()),
                            ..Default::default()
                        }),
                        session: Some(session_ref.clone()),
                        ..Default::default()
                    })
                    .load()
                    .await
                    .unwrap();

                barrier_ref.wait().await;

                // Each task deletes a different range of rows to avoid complete overlap
                let start = i * 10;
                let end = (i + 1) * 10;
                DeleteBuilder::new(
                    Arc::new(dataset),
                    format!("id >= {} AND id < {}", start, end),
                )
                .conflict_retries(5)
                .execute()
                .await
                .unwrap()
            });
            handles.push(handle);
        }

        try_join_all(handles).await.unwrap();

        dataset.checkout_latest().await.unwrap();

        // All rows should be deleted since each task deleted a non-overlapping range
        let remaining_count = dataset.count_rows(None).await.unwrap();
        assert_eq!(remaining_count, 0, "All rows should be deleted");

        // Verify no fragments remain or they are all empty
        let fragments = dataset.get_fragments();
        if !fragments.is_empty() {
            // If fragments exist, they should all have deletion vectors covering all rows
            for fragment in &fragments {
                let deletion_vector = fragment.get_deletion_vector().await.unwrap();
                assert!(
                    deletion_vector.is_some(),
                    "Fragment should have deletion vector if any rows remain"
                );
            }
        }
    }

    #[tokio::test]
    #[rstest]
    async fn test_delete_true_update_conflict(#[values(false, true)] enable_stable_row_ids: bool) {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::UInt32, false),
            ArrowField::new("value", DataType::UInt32, false),
        ]));

        // Create two batches to ensure multiple fragments
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from_iter_values(0..100)),
                Arc::new(UInt32Array::from_iter_values(std::iter::repeat_n(100, 100))),
            ],
        )
        .unwrap();

        let dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                enable_stable_row_ids,
                max_rows_per_file: 50,
                ..Default::default()
            })
            .execute(vec![batch])
            .await
            .unwrap();

        // Verify we have 2 fragments initially
        assert_eq!(dataset.get_fragments().len(), 2);
        assert_eq!(dataset.count_rows(None).await.unwrap(), 100);

        let dataset_arc = Arc::new(dataset);
        let delete_job = DeleteJob {
            dataset: dataset_arc.clone(),
            filter: ExprFilter::Sql("true".to_string()),
            schema_metadata_updates: None,
        };
        let delete_data = delete_job.execute_impl().await.unwrap();

        // Verify delete preparation captured all fragments for deletion
        assert_eq!(delete_data.deleted_fragment_ids.len(), 2);
        assert!(delete_data.updated_fragments.is_empty());

        // Run a concurrent update operation that commits
        let update_job = UpdateBuilder::new(dataset_arc.clone())
            .update_where("id < 25")
            .unwrap() // Update first 25 rows
            .set("value", "value + 1000")
            .unwrap()
            .build()
            .unwrap();
        let update_result = update_job.execute().await.unwrap();
        assert_eq!(
            update_result.new_dataset.count_rows(None).await.unwrap(),
            100
        );

        // Now try to commit the delete operation using the stale dataset reference
        // This should fail because the delete was planning to delete fragments that
        // have been modified by the update
        let result = delete_job.commit(dataset_arc.clone(), delete_data).await;

        // When deleting everything with delete("true"), the operation should succeed
        // but it might not delete all rows if concurrent updates moved some rows
        assert!(
            matches!(&result, Err(Error::RetryableCommitConflict { .. })),
            "Expected retryable conflict due to concurrent update, got {:?}",
            result
        );

        // Also verify with the retry mechanism that it works correctly
        let final_result = DeleteBuilder::new(dataset_arc, "true")
            .conflict_retries(5)
            .execute()
            .await
            .unwrap();
        // All rows should be deleted, including the updated ones
        assert_eq!(final_result.new_dataset.count_rows(None).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_delete_with_expr_filter() {
        use datafusion::prelude::{col, lit};

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::UInt32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(UInt32Array::from_iter_values(0..100u32))],
        )
        .unwrap();

        let mut dataset = InsertBuilder::new("memory://")
            .execute(vec![batch])
            .await
            .unwrap();

        // Delete rows where i < 10 using an Expr filter
        let expr = col("i").lt(lit(10u32));
        let result = DeleteBuilder::from_expr(Arc::new(dataset.clone()), expr)
            .execute()
            .await
            .unwrap();

        assert_eq!(result.num_deleted_rows, 10);

        dataset.checkout_latest().await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 90);
    }

    // -------------------------------------------------------------------------
    // A4.4b: DeleteBuilder schema metadata attachment
    // -------------------------------------------------------------------------

    const A44B_SCHEMA_KEY: &str = "a44b.schema";
    const A44B_SCHEMA_VALUE: &str = "schema-v1";
    const A44B_FIELD_KEY: &str = "a44b.field";
    const A44B_FIELD_VALUE: &str = "field-v1";

    fn a44b_metadata_updates() -> SchemaMetadataUpdates {
        SchemaMetadataUpdates {
            schema_metadata_updates: Some(UpdateMap {
                update_entries: vec![UpdateMapEntry::from((A44B_SCHEMA_KEY, A44B_SCHEMA_VALUE))],
                replace: false,
            }),
            field_metadata_updates: HashMap::from([(
                0i32,
                UpdateMap {
                    update_entries: vec![UpdateMapEntry::from((A44B_FIELD_KEY, A44B_FIELD_VALUE))],
                    replace: false,
                },
            )]),
        }
    }

    fn a44b_id_batch(values: Vec<u32>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::UInt32,
            false,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(UInt32Array::from(values))]).unwrap()
    }

    async fn a44b_create_dataset(uri: &str) -> Dataset {
        InsertBuilder::new(uri)
            .execute(vec![a44b_id_batch((0..10).collect())])
            .await
            .expect("A4.4b fixture create must succeed")
    }

    fn a44b_metadata_is_absent(dataset: &Dataset) -> bool {
        let schema_absent = !dataset.schema().metadata.contains_key(A44B_SCHEMA_KEY);
        let field_absent = dataset
            .schema()
            .field_by_id(0)
            .map(|field0| !field0.metadata.contains_key(A44B_FIELD_KEY))
            .unwrap_or(false);
        schema_absent && field_absent
    }

    fn a44b_assert_metadata_absent(dataset: &Dataset) {
        assert!(
            !dataset.schema().metadata.contains_key(A44B_SCHEMA_KEY),
            "{A44B_SCHEMA_KEY} must be absent, got {:?}",
            dataset.schema().metadata
        );
        let field0 = dataset
            .schema()
            .field_by_id(0)
            .expect("field id 0 must exist");
        assert!(
            !field0.metadata.contains_key(A44B_FIELD_KEY),
            "{A44B_FIELD_KEY} must be absent, got {:?}",
            field0.metadata
        );
    }

    fn a44b_assert_metadata_present(dataset: &Dataset) {
        assert_eq!(
            dataset
                .schema()
                .metadata
                .get(A44B_SCHEMA_KEY)
                .map(String::as_str),
            Some(A44B_SCHEMA_VALUE),
            "schema metadata patch must be published"
        );
        let field0 = dataset
            .schema()
            .field_by_id(0)
            .expect("field id 0 must exist");
        assert_eq!(
            field0.metadata.get(A44B_FIELD_KEY).map(String::as_str),
            Some(A44B_FIELD_VALUE),
            "field id 0 metadata patch must be published"
        );
    }

    async fn a44b_assert_remaining_ids(dataset: &Dataset, expected: &[u32]) {
        assert_eq!(dataset.count_rows(None).await.unwrap(), expected.len());
        let batch = dataset.scan().try_into_batch().await.unwrap();
        let values: Vec<u32> = batch["i"].as_primitive::<UInt32Type>().values().to_vec();
        assert_eq!(values, expected);
    }

    /// A4.4b: Delete via DeleteBuilder publishes row deletes + schema/field
    /// metadata in exactly one version; a fresh Dataset::open agrees.
    #[tokio::test]
    async fn test_a44b_delete_publishes_data_and_metadata_in_one_version() {
        let test_dir = TempStrDir::default();
        let uri = test_dir.as_str();
        let dataset = a44b_create_dataset(uri).await;
        let before_version = dataset.version().version;
        a44b_assert_metadata_absent(&dataset);

        let result = DeleteBuilder::new(Arc::new(dataset), "i < 3")
            .with_schema_metadata_updates(a44b_metadata_updates())
            .expect("Delete attachment must construct")
            .execute()
            .await
            .expect("attached Delete must commit");

        assert_eq!(result.num_deleted_rows, 3);
        assert_eq!(
            result.new_dataset.version().version,
            before_version + 1,
            "exactly one new version"
        );
        a44b_assert_remaining_ids(&result.new_dataset, &(3..10).collect::<Vec<_>>()).await;
        a44b_assert_metadata_present(&result.new_dataset);

        let reopened = Dataset::open(uri).await.unwrap();
        assert_eq!(
            reopened.version().version,
            result.new_dataset.version().version
        );
        a44b_assert_remaining_ids(&reopened, &(3..10).collect::<Vec<_>>()).await;
        a44b_assert_metadata_present(&reopened);
    }

    /// A4.4b: execute_uncommitted preserves Delete + exact attachment without
    /// publishing; CommitBuilder with affected_rows then lands both sides.
    #[tokio::test]
    async fn test_a44b_execute_uncommitted_preserves_attachment_until_commit() {
        let test_dir = TempStrDir::default();
        let uri = test_dir.as_str();
        let dataset = Arc::new(a44b_create_dataset(uri).await);
        let before_version = dataset.version().version;
        a44b_assert_metadata_absent(&dataset);

        let updates = a44b_metadata_updates();
        let staged_delete = DeleteBuilder::new(dataset.clone(), "i < 3")
            .with_schema_metadata_updates(updates.clone())
            .expect("Delete attachment must construct")
            .execute_uncommitted()
            .await
            .expect("uncommitted attached Delete must stage");

        assert!(
            matches!(
                &staged_delete.transaction.operation,
                Operation::Delete { .. }
            ),
            "uncommitted transaction must be Delete, got {:?}",
            staged_delete.transaction.operation
        );
        assert_eq!(
            staged_delete.transaction.schema_metadata_updates.as_ref(),
            Some(&updates),
            "attachment must be preserved exactly on the Transaction"
        );
        assert_eq!(staged_delete.num_deleted_rows, 3);

        assert_eq!(dataset.version().version, before_version);
        a44b_assert_remaining_ids(&dataset, &(0..10).collect::<Vec<_>>()).await;
        a44b_assert_metadata_absent(&dataset);

        let before_commit = Dataset::open(uri).await.unwrap();
        assert_eq!(before_commit.version().version, before_version);
        a44b_assert_remaining_ids(&before_commit, &(0..10).collect::<Vec<_>>()).await;
        a44b_assert_metadata_absent(&before_commit);

        let mut commit_builder = CommitBuilder::new(dataset.clone());
        if let Some(affected_rows) = staged_delete.affected_rows {
            commit_builder = commit_builder.with_affected_rows(affected_rows);
        }
        let committed = commit_builder
            .execute(staged_delete.transaction)
            .await
            .expect("committing attached Delete must succeed");

        assert_eq!(committed.version().version, before_version + 1);
        a44b_assert_remaining_ids(&committed, &(3..10).collect::<Vec<_>>()).await;
        a44b_assert_metadata_present(&committed);

        let reopened = Dataset::open(uri).await.unwrap();
        assert_eq!(reopened.version().version, committed.version().version);
        a44b_assert_remaining_ids(&reopened, &(3..10).collect::<Vec<_>>()).await;
        a44b_assert_metadata_present(&reopened);
    }

    /// A4.4b: a substantive patch targeting nonexistent field id 999 must fail
    /// atomically at execute; neither deletes nor metadata are visible on reopen.
    #[tokio::test]
    async fn test_a44b_nonexistent_field_attached_delete_publishes_neither_side() {
        let test_dir = TempStrDir::default();
        let uri = test_dir.as_str();
        let dataset = a44b_create_dataset(uri).await;
        let before_version = dataset.version().version;
        a44b_assert_metadata_absent(&dataset);

        let updates = SchemaMetadataUpdates {
            schema_metadata_updates: Some(UpdateMap {
                update_entries: vec![UpdateMapEntry::from((A44B_SCHEMA_KEY, A44B_SCHEMA_VALUE))],
                replace: false,
            }),
            field_metadata_updates: HashMap::from([(
                999i32,
                UpdateMap {
                    update_entries: vec![UpdateMapEntry::from((A44B_FIELD_KEY, A44B_FIELD_VALUE))],
                    replace: false,
                },
            )]),
        };

        let err = DeleteBuilder::new(Arc::new(dataset), "i < 3")
            .with_schema_metadata_updates(updates)
            .expect("setter may accept; field validity is manifest-relative")
            .execute()
            .await
            .expect_err("nonexistent field id must reject the Delete");

        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "nonexistent field id must be InvalidInput, got {err:?}"
        );
        assert!(
            err.to_string().contains("999"),
            "error must name the missing field id, got {err}"
        );

        let reopened = Dataset::open(uri).await.unwrap();
        assert_eq!(
            reopened.version().version,
            before_version,
            "failed Delete must not advance version"
        );
        a44b_assert_remaining_ids(&reopened, &(0..10).collect::<Vec<_>>()).await;
        a44b_assert_metadata_absent(&reopened);
    }

    // -------------------------------------------------------------------------
    // A4d: zero-row Delete must not publish attached schema/field metadata
    // -------------------------------------------------------------------------

    /// A4d: a no-op Delete (predicate scans but matches zero rows) must not
    /// publish attached schema/field metadata. Version advancement is
    /// unconstrained.
    #[tokio::test]
    async fn test_a4d_noop_delete_does_not_publish_metadata() {
        let test_dir = TempStrDir::default();
        let uri = test_dir.as_str();
        let dataset = a44b_create_dataset(uri).await;
        a44b_assert_metadata_absent(&dataset);
        a44b_assert_remaining_ids(&dataset, &(0..10).collect::<Vec<_>>()).await;

        let result = DeleteBuilder::new(Arc::new(dataset), "i > 100")
            .with_schema_metadata_updates(a44b_metadata_updates())
            .expect("Delete attachment must construct")
            .execute()
            .await
            .expect("no-op attached Delete must complete");

        assert_eq!(result.num_deleted_rows, 0, "predicate must match zero rows");
        a44b_assert_remaining_ids(&result.new_dataset, &(0..10).collect::<Vec<_>>()).await;
        a44b_assert_metadata_absent(&result.new_dataset);

        let reopened = Dataset::open(uri).await.unwrap();
        a44b_assert_remaining_ids(&reopened, &(0..10).collect::<Vec<_>>()).await;
        a44b_assert_metadata_absent(&reopened);
    }

    /// A4d: execute_uncommitted on a zero-row attached Delete must not carry
    /// schema_metadata_updates; CommitBuilder with affected_rows must leave
    /// rows and metadata unchanged. Observations from staged txn, commit, and
    /// reopen are aggregated so the real commit path always runs first.
    #[tokio::test]
    async fn test_a4d_noop_execute_uncommitted_does_not_attach_metadata() {
        let test_dir = TempStrDir::default();
        let uri = test_dir.as_str();
        let dataset = Arc::new(a44b_create_dataset(uri).await);
        a44b_assert_metadata_absent(&dataset);
        a44b_assert_remaining_ids(&dataset, &(0..10).collect::<Vec<_>>()).await;

        let staged_delete = DeleteBuilder::new(dataset.clone(), "i > 100")
            .with_schema_metadata_updates(a44b_metadata_updates())
            .expect("Delete attachment must construct")
            .execute_uncommitted()
            .await
            .expect("uncommitted no-op attached Delete must stage");

        assert_eq!(
            staged_delete.num_deleted_rows, 0,
            "predicate must match zero rows"
        );
        let staged_attachment_absent = staged_delete.transaction.schema_metadata_updates.is_none();
        let staged_attachment = format!("{:?}", staged_delete.transaction.schema_metadata_updates);

        let mut commit_builder = CommitBuilder::new(dataset.clone());
        if let Some(affected_rows) = staged_delete.affected_rows {
            commit_builder = commit_builder.with_affected_rows(affected_rows);
        }
        let committed = commit_builder
            .execute(staged_delete.transaction)
            .await
            .expect("committing no-op Delete must succeed");

        a44b_assert_remaining_ids(&committed, &(0..10).collect::<Vec<_>>()).await;
        let committed_metadata_absent = a44b_metadata_is_absent(&committed);

        let reopened = Dataset::open(uri).await.unwrap();
        a44b_assert_remaining_ids(&reopened, &(0..10).collect::<Vec<_>>()).await;
        let reopened_metadata_absent = a44b_metadata_is_absent(&reopened);

        assert!(
            staged_attachment_absent && committed_metadata_absent && reopened_metadata_absent,
            "zero-row Delete must not attach or publish metadata; \
             staged_attachment_absent={staged_attachment_absent} (updates={staged_attachment}), \
             committed_metadata_absent={committed_metadata_absent}, \
             reopened_metadata_absent={reopened_metadata_absent}"
        );
    }
}
