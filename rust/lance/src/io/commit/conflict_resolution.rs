// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use futures::future::BoxFuture;
use lance_core::utils::mask::RowIdMask;
use lance_io::object_store::ObjectStore;
use lance_table::format::Fragment;

use crate::{io::commit::Transaction, session::Session, Dataset, Result};

async fn resolve_conflicts(
    transaction: Transaction,
    dataset: &Dataset,
) -> Result<(Transaction, Option<BoxFuture<Result<()>>>)> {
    // Assume dataset is already in latest version.
    let start = transaction.read_version + 1;
    let end = dataset.manifest().version;

    let original_dataset = dataset.checkout_version(transaction.read_version).await?;
    let old_fragments = original_dataset.fragments().as_slice();
    let mut possible_conflicts: Vec<(u64, Transaction)> = Vec::new();
    for version in start..=end {
        // Load each transaction

        // For each transaction, there are four possible outcomes:
        // 1. Irreconcilable conflict. Example: overwrite happened.
        // 2. No conflict. Example: append happened. (can ignore)
        // 3. Retry-able conflict. Example: upsert deleted relevant fragment
        // 4. Possible conflict. Example: upsert modified relevant fragment

        // TODO: modify .conflicts_with() to differentiate these four outcomes.
    }

    // If there are no conflicts, we can just return the transaction as-is.

    // Possible conflict:
    // * Either needs a rewrite
    // * Or becomes retry-able conflict

    // Maybe I should grab them in here?
    // TODO: return cleanup task too?
    // TODO: nice errors differentiate retry-able and non-retry-able conflicts
    // TODO: get diff on deletions
    todo!()
}

/// Identify which rows have been deleted or moved by the transaction.
async fn build_diff(
    transaction: &Transaction,
    old_fragments: &[Fragment],
    object_store: &ObjectStore,
    session: &Session,
) -> Result<RowIdMask> {
    todo!()
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use arrow_array::{BooleanArray, Int64Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_core::Error;
    use lance_io::object_store::ObjectStoreParams;
    use lance_table::io::{commit::RenameCommitHandler, deletion::read_deletion_file};
    use url::Url;

    use crate::{
        dataset::{
            transaction::Operation, InsertBuilder, MergeInsertBuilder, MergeStats, WriteParams,
        },
        utils::test::{IoStats, IoTrackingStore},
    };

    use super::*;

    struct IOTrackingDatasetFixture {
        pub dataset: Arc<Dataset>,
        pub io_stats: Arc<Mutex<IoStats>>,
        pub store: Arc<object_store::memory::InMemory>,
    }

    impl IOTrackingDatasetFixture {
        pub async fn new(data: Vec<RecordBatch>) -> Self {
            let store = Arc::new(object_store::memory::InMemory::new());
            let (io_stats_wrapper, io_stats) = IoTrackingStore::new_wrapper();
            let store_params = ObjectStoreParams {
                object_store_wrapper: Some(io_stats_wrapper),
                object_store: Some((store.clone(), Url::parse("memory://test").unwrap())),
                ..Default::default()
            };

            let dataset = InsertBuilder::new("memory://test")
                .with_params(&WriteParams {
                    store_params: Some(store_params.clone()),
                    commit_handler: Some(Arc::new(RenameCommitHandler)),
                    ..Default::default()
                })
                .execute(data)
                .await
                .unwrap();
            let dataset = Arc::new(dataset);

            Self {
                dataset,
                io_stats,
                store,
            }
        }

        pub fn reset_stats(&self) {
            let mut io_stats = self.io_stats.lock().unwrap();
            io_stats.read_bytes = 0;
            io_stats.write_bytes = 0;
            io_stats.read_iops = 0;
            io_stats.write_iops = 0;
        }

        pub fn get_new_stats(&self) -> IoStats {
            let stats = self.io_stats.lock().unwrap().clone();
            self.reset_stats();
            stats
        }
    }

    #[tokio::test]
    async fn test_resolve_conflicts_noop() {
        todo!("Test that append, and other non-conflicting ones just return the same thing")
    }

    struct UpsertFixture {
        pub io_fixture: IOTrackingDatasetFixture,
        pub schema: Arc<ArrowSchema>,
    }

    impl UpsertFixture {
        pub async fn new() -> Self {
            let schema = Arc::new(ArrowSchema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("updated", DataType::Boolean, false),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(vec![1, 2, 3])),
                    Arc::new(BooleanArray::from(vec![false, false, false])),
                ],
            )
            .unwrap();
            let fixture = IOTrackingDatasetFixture::new(vec![batch]).await;

            // do two upsert transactions
            let slf = Self {
                io_fixture: fixture,
                schema,
            };
            slf.do_upsert(vec![2]).await;
            slf
        }

        pub fn upsert_data(&self, ids: Vec<i64>) -> RecordBatch {
            let nrows = ids.len();
            RecordBatch::try_new(
                self.schema.clone(),
                vec![
                    Arc::new(Int64Array::from(ids)),
                    Arc::new(BooleanArray::from(vec![true; nrows])),
                ],
            )
            .unwrap()
        }

        pub async fn do_upsert(&self, ids: Vec<i64>) -> MergeStats {
            let batch = self.upsert_data(ids);
            let reader = RecordBatchIterator::new(vec![Ok(batch)], self.schema.clone());
            MergeInsertBuilder::try_new(self.io_fixture.dataset.clone(), vec!["id".into()])
                .unwrap()
                .try_build()
                .unwrap()
                .execute_reader(reader)
                .await
                .unwrap()
                .1
        }
    }

    #[tokio::test]
    async fn test_resolve_upsert() {
        let fixture = UpsertFixture::new().await;
        let dataset = fixture.io_fixture.dataset.clone();

        // check we get Ok() if we upsert a different row from original read version
        let unique_rows = fixture.upsert_data(vec![3]);
        let reader = RecordBatchIterator::new(vec![Ok(unique_rows)], fixture.schema.clone());
        let old_dataset = Arc::new(dataset.checkout_version(1).await.unwrap());
        let (transaction, stats) = MergeInsertBuilder::try_new(old_dataset, vec!["id".into()])
            .unwrap()
            .try_build()
            .unwrap()
            .execute_uncommitted(reader)
            .await
            .unwrap();
        assert_eq!(stats.num_updated_rows, 1);
        assert_eq!(stats.num_inserted_rows, 0);
        fixture.io_fixture.reset_stats();
        let (new_transaction, cleanup_task) = resolve_conflicts(transaction.clone(), &dataset)
            .await
            .unwrap();
        let io_stats = fixture.io_fixture.get_new_stats();
        // We should have everything in the session cache
        assert_eq!(io_stats.read_bytes, 0);
        assert_eq!(io_stats.read_iops, 0);
        // We needed to write a new deletion file
        assert_eq!(io_stats.write_iops, 1);

        // Transaction should be updated
        // UUID should be re-used
        assert_eq!(transaction.uuid, new_transaction.uuid);
        assert_eq!(transaction.read_version, new_transaction.read_version);

        fn extract_updated_frags(op: &Operation) -> Vec<Fragment> {
            match op {
                Operation::Update {
                    updated_fragments, ..
                } => updated_fragments.clone(),
                _ => panic!("Expected update operation"),
            }
        }
        let updated_frags = extract_updated_frags(&transaction.operation);
        let new_updated_frags = extract_updated_frags(&new_transaction.operation);
        assert_ne!(updated_frags, new_updated_frags);
        assert_eq!(updated_frags.len(), 1);
        assert_eq!(new_updated_frags.len(), 1);
        // Data files should be unmodified
        assert_eq!(updated_frags[0].files, new_updated_frags[0].files);
        // Deletion file should be different
        assert_ne!(
            updated_frags[0].deletion_file,
            new_updated_frags[0].deletion_file
        );

        // Only one should still exist.
        let deletion_vector =
            read_deletion_file(&dataset.base, &updated_frags[0], dataset.object_store())
                .await
                .unwrap()
                .unwrap();
        assert_eq!(
            deletion_vector.to_sorted_iter().collect::<Vec<_>>(),
            vec![2]
        );

        // Should have merged with existing deletion vector
        let deletion_vector =
            read_deletion_file(&dataset.base, &new_updated_frags[0], dataset.object_store())
                .await
                .unwrap()
                .unwrap();
        assert_eq!(
            deletion_vector.to_sorted_iter().collect::<Vec<_>>(),
            vec![1, 2]
        );

        cleanup_task.unwrap().await.unwrap();
        let res =
            read_deletion_file(&dataset.base, &new_updated_frags[0], dataset.object_store()).await;
        assert!(matches!(res, Err(Error::NotFound { .. })), "{:?}", res);
    }

    #[tokio::test]
    async fn test_upsert_no_conflict() {
        let fixture = UpsertFixture::new().await;
        let dataset = fixture.io_fixture.dataset.clone();

        // Check we get Ok() if we upsert a new row
        let new_row = fixture.upsert_data(vec![4]);
        let reader = RecordBatchIterator::new(vec![Ok(new_row)], fixture.schema.clone());
        let old_dataset = Arc::new(dataset.checkout_version(1).await.unwrap());
        let (transaction, stats) = MergeInsertBuilder::try_new(old_dataset, vec!["id".into()])
            .unwrap()
            .try_build()
            .unwrap()
            .execute_uncommitted(reader)
            .await
            .unwrap();
        assert_eq!(stats.num_updated_rows, 0);
        assert_eq!(stats.num_inserted_rows, 1);
        fixture.io_fixture.reset_stats();
        let (new_transaction, cleanup_task) = resolve_conflicts(transaction.clone(), &dataset)
            .await
            .unwrap();
        let io_stats = fixture.io_fixture.get_new_stats();
        // We should have everything in the session cache
        assert_eq!(io_stats.read_bytes, 0);
        assert_eq!(io_stats.read_iops, 0);
        // We didn't need to change any files because there are no conflicts
        assert_eq!(io_stats.write_iops, 0);

        // Transaction should be left the same
        assert_eq!(transaction.uuid, new_transaction.uuid);
        assert_eq!(transaction.read_version, new_transaction.read_version);
        assert_eq!(transaction.operation, new_transaction.operation);

        assert!(cleanup_task.is_none());
    }

    #[tokio::test]
    async fn test_upsert_retry_error() {
        let fixture = UpsertFixture::new().await;
        let dataset = fixture.io_fixture.dataset.clone();

        // We should get a retryable conflict error if we try to upsert the same
        // row.
        let unique_rows = fixture.upsert_data(vec![2]);
        let reader = RecordBatchIterator::new(vec![Ok(unique_rows)], fixture.schema.clone());
        let old_dataset = Arc::new(dataset.checkout_version(1).await.unwrap());
        let (transaction, stats) = MergeInsertBuilder::try_new(old_dataset, vec!["id".into()])
            .unwrap()
            .try_build()
            .unwrap()
            .execute_uncommitted(reader)
            .await
            .unwrap();
        assert_eq!(stats.num_updated_rows, 1);
        assert_eq!(stats.num_inserted_rows, 0);
        fixture.io_fixture.reset_stats();
        let err = resolve_conflicts(transaction.clone(), &dataset).await;

        if let Err(err) = err {
            assert!(matches!(err, Error::CommitConflict { .. }), "{}", err);
        } else {
            panic!("Expected error");
        }
    }
}
