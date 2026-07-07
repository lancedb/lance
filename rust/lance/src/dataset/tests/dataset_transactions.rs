use std::collections::HashMap;
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::vec;

use crate::dataset::builder::DatasetBuilder;
use crate::dataset::transaction::{Operation, Transaction};
use crate::dataset::{ManifestWriteConfig, TRANSACTIONS_DIR, write_manifest_file};
use crate::io::ObjectStoreParams;
use crate::session::Session;
use crate::{Dataset, Result};
use lance_table::io::commit::ManifestNamingScheme;

use crate::dataset::write::{CommitBuilder, InsertBuilder, WriteMode, WriteParams};
use crate::index::DatasetIndexExt;
use arrow_array::Array;
use arrow_array::RecordBatch;
use arrow_array::{Int32Array, RecordBatchIterator, StringArray, types::Int32Type};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_core::utils::tempfile::{TempDir, TempStrDir};
use lance_datagen::{BatchCount, RowCount, array};

use crate::datafusion::LanceTableProvider;
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;
use lance_datafusion::udf::register_functions;
use object_store::ObjectStoreExt;

#[tokio::test]
async fn test_read_transaction_properties() {
    const LANCE_COMMIT_MESSAGE_KEY: &str = "__lance_commit_message";
    // Create a test dataset
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("value", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .unwrap();

    let test_uri = TempStrDir::default();

    // Create WriteParams with properties
    let mut properties1 = HashMap::new();
    properties1.insert(
        LANCE_COMMIT_MESSAGE_KEY.to_string(),
        "First commit".to_string(),
    );
    properties1.insert("custom_prop".to_string(), "custom_value".to_string());

    let write_params = WriteParams {
        transaction_properties: Some(Arc::new(properties1)),
        ..Default::default()
    };

    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch.clone())], schema.clone()),
        &test_uri,
        Some(write_params),
    )
    .await
    .unwrap();

    let transaction = dataset.read_transaction_by_version(1).await.unwrap();
    assert!(transaction.is_some());
    let props = transaction.unwrap().transaction_properties.unwrap();
    assert_eq!(props.len(), 2);
    assert_eq!(
        props.get(LANCE_COMMIT_MESSAGE_KEY),
        Some(&"First commit".to_string())
    );
    assert_eq!(props.get("custom_prop"), Some(&"custom_value".to_string()));

    let mut properties2 = HashMap::new();
    properties2.insert(
        LANCE_COMMIT_MESSAGE_KEY.to_string(),
        "Second commit".to_string(),
    );
    properties2.insert("another_prop".to_string(), "another_value".to_string());

    let write_params = WriteParams {
        transaction_properties: Some(Arc::new(properties2)),
        mode: WriteMode::Append,
        ..Default::default()
    };

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![4, 5])),
            Arc::new(StringArray::from(vec!["d", "e"])),
        ],
    )
    .unwrap();

    let mut dataset = dataset;
    dataset
        .append(
            RecordBatchIterator::new([Ok(batch2)], schema.clone()),
            Some(write_params),
        )
        .await
        .unwrap();

    let transaction = dataset.read_transaction_by_version(2).await.unwrap();
    assert!(transaction.is_some());
    let props = transaction.unwrap().transaction_properties.unwrap();
    assert_eq!(props.len(), 2);
    assert_eq!(
        props.get(LANCE_COMMIT_MESSAGE_KEY),
        Some(&"Second commit".to_string())
    );
    assert_eq!(
        props.get("another_prop"),
        Some(&"another_value".to_string())
    );

    let transaction = dataset.read_transaction_by_version(1).await.unwrap();
    assert!(transaction.is_some());
    let props = transaction.unwrap().transaction_properties.unwrap();
    assert_eq!(props.len(), 2);
    assert_eq!(
        props.get(LANCE_COMMIT_MESSAGE_KEY),
        Some(&"First commit".to_string())
    );
    assert_eq!(props.get("custom_prop"), Some(&"custom_value".to_string()));

    let result = dataset.read_transaction_by_version(999).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_session_store_registry() {
    // Create a session
    let session = Arc::new(Session::default());
    let registry = session.store_registry();
    assert!(registry.active_stores().is_empty());

    // Create a dataset with memory store
    let write_params = WriteParams {
        session: Some(session.clone()),
        ..Default::default()
    };
    let batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::Int32,
            false,
        )])),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let dataset = InsertBuilder::new("memory://test")
        .with_params(&write_params)
        .execute(vec![batch.clone()])
        .await
        .unwrap();

    // Assert there is one active store.
    assert_eq!(registry.active_stores().len(), 1);

    // If we create another dataset also in memory, it should re-use the
    // existing store.
    let dataset2 = InsertBuilder::new("memory://test2")
        .with_params(&write_params)
        .execute(vec![batch.clone()])
        .await
        .unwrap();
    assert_eq!(registry.active_stores().len(), 1);
    assert_eq!(
        Arc::as_ptr(&dataset.object_store.as_ref().inner),
        Arc::as_ptr(&dataset2.object_store.as_ref().inner)
    );

    // If we create another with **different parameters**, it should create a new store.
    let write_params2 = WriteParams {
        session: Some(session.clone()),
        store_params: Some(ObjectStoreParams {
            block_size: Some(10_000),
            ..Default::default()
        }),
        ..Default::default()
    };
    let dataset3 = InsertBuilder::new("memory://test3")
        .with_params(&write_params2)
        .execute(vec![batch.clone()])
        .await
        .unwrap();
    assert_eq!(registry.active_stores().len(), 2);
    assert_ne!(
        Arc::as_ptr(&dataset.object_store.as_ref().inner),
        Arc::as_ptr(&dataset3.object_store.as_ref().inner)
    );

    // Remove both datasets
    drop(dataset3);
    assert_eq!(registry.active_stores().len(), 1);
    drop(dataset2);
    drop(dataset);
    assert_eq!(registry.active_stores().len(), 0);
}

#[tokio::test]
async fn test_migrate_v2_manifest_paths() {
    let test_uri = TempStrDir::default();

    let data = lance_datagen::gen_batch()
        .col("key", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let mut dataset = Dataset::write(
        data,
        &test_uri,
        Some(WriteParams {
            enable_v2_manifest_paths: false,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(
        dataset.manifest_location().naming_scheme,
        ManifestNamingScheme::V1
    );

    dataset.migrate_manifest_paths_v2().await.unwrap();
    assert_eq!(
        dataset.manifest_location().naming_scheme,
        ManifestNamingScheme::V2
    );
}

pub(super) async fn execute_sql(
    sql: &str,
    table: String,
    dataset: Arc<Dataset>,
) -> Result<Vec<RecordBatch>> {
    let ctx = SessionContext::new();
    ctx.register_table(
        table,
        Arc::new(LanceTableProvider::new(dataset, false, false)),
    )?;
    register_functions(&ctx);

    let df = ctx.sql(sql).await?;
    Ok(df
        .execute_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await?)
}

pub(super) fn assert_results<T: Array + PartialEq + 'static>(
    results: Vec<RecordBatch>,
    values: &T,
) {
    assert_eq!(results.len(), 1);
    let results = results.into_iter().next().unwrap();
    assert_eq!(results.num_columns(), 1);

    assert_eq!(
        results.column(0).as_any().downcast_ref::<T>().unwrap(),
        values
    )
}

#[tokio::test]
async fn test_inline_transaction() {
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use std::sync::Arc;

    async fn create_dataset(rows: i32) -> Arc<Dataset> {
        let dir = TempDir::default();
        let uri = dir.path_str();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..rows))],
        )
        .unwrap();
        let ds = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            uri.as_str(),
            None,
        )
        .await
        .unwrap();
        Arc::new(ds)
    }

    fn make_tx(read_version: u64) -> Transaction {
        Transaction::new(read_version, Operation::Append { fragments: vec![] }, None)
    }

    async fn delete_external_tx_file(ds: &Dataset) {
        if let Some(tx_file) = ds.manifest.transaction_file.as_ref() {
            let tx_path = ds
                .base
                .clone()
                .join(TRANSACTIONS_DIR)
                .join(tx_file.as_str());
            let _ = ds.object_store.inner.delete(&tx_path).await; // ignore errors
        }
    }

    let session = Arc::new(Session::default());

    // Case 1: Default write_flag=true, delete external transaction file, read should use inline transaction
    let ds = create_dataset(5).await;
    let read_version = ds.manifest().version;
    let tx = make_tx(read_version);
    let ds2 = CommitBuilder::new(ds.clone())
        .execute(tx.clone())
        .await
        .unwrap();
    delete_external_tx_file(&ds2).await;
    let read_tx = ds2.read_transaction().await.unwrap().unwrap();
    assert_eq!(read_tx, tx.clone());

    // Case 2: reading small manifest caches transaction data, eliminating transaction reading IO.
    let read_ds2 = DatasetBuilder::from_uri(ds2.uri.clone())
        .with_session(session.clone())
        .load()
        .await
        .unwrap();
    let stats = read_ds2.object_store.as_ref().io_stats_incremental(); // Reset
    assert!(stats.read_bytes < 64 * 1024);
    // Because the manifest is so small, we should have opportunistically
    // cached the transaction in memory already.
    let inline_tx = read_ds2.read_transaction().await.unwrap().unwrap();
    let stats = read_ds2.object_store.as_ref().io_stats_incremental();
    assert_eq!(stats.read_iops, 0);
    assert_eq!(stats.read_bytes, 0);
    assert_eq!(inline_tx, tx);

    // Case 3: manifest does not contain inline transaction, read should fall back to external transaction file
    let ds = create_dataset(2).await;
    let tx = make_tx(ds.manifest().version);
    let tx_file =
        crate::io::commit::write_transaction_file(ds.object_store.as_ref(), &ds.base, &tx)
            .await
            .unwrap();
    let (mut manifest, indices) = tx
        .build_manifest(
            Some(ds.manifest.as_ref()),
            ds.load_indices().await.unwrap().as_ref().clone(),
            &tx_file,
            &ManifestWriteConfig::default(),
        )
        .unwrap();
    let location = write_manifest_file(
        ds.object_store.as_ref(),
        ds.commit_handler.as_ref(),
        &ds.base,
        &mut manifest,
        if indices.is_empty() {
            None
        } else {
            Some(indices.clone())
        },
        &ManifestWriteConfig::default(),
        ds.manifest_location.naming_scheme,
        None,
    )
    .await
    .unwrap();
    let ds_new = ds.checkout_version(location.version).await.unwrap();
    assert!(ds_new.manifest.transaction_section.is_none());
    assert!(ds_new.manifest.transaction_file.is_some());
    let read_tx = ds_new.read_transaction().await.unwrap().unwrap();
    assert_eq!(read_tx, tx);
}

#[tokio::test]
async fn test_list_detached_manifests() {
    let test_uri = TempStrDir::default();

    // Create initial dataset
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "id",
        DataType::Int32,
        false,
    )]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();

    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], schema.clone()),
            &test_uri,
            None,
        )
        .await
        .unwrap(),
    );

    // Initially there should be no detached manifests
    let detached = dataset.list_detached_manifests().await.unwrap();
    assert!(detached.is_empty());

    // Create a detached transaction with properties
    let mut properties = HashMap::new();
    properties.insert("detached_key".to_string(), "detached_value".to_string());

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![4, 5, 6]))],
    )
    .unwrap();

    // Use execute_uncommitted + CommitBuilder with_detached(true)
    let transaction = InsertBuilder::new(dataset.clone())
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            transaction_properties: Some(Arc::new(properties.clone())),
            ..Default::default()
        })
        .execute_uncommitted(vec![batch2])
        .await
        .unwrap();

    CommitBuilder::new(dataset.clone())
        .with_detached(true)
        .execute(transaction)
        .await
        .unwrap();

    // Now there should be one detached manifest
    let detached = dataset.list_detached_manifests().await.unwrap();
    assert_eq!(detached.len(), 1);

    // The detached version should have the high bit set
    let detached_version = detached[0].version;
    assert!(lance_table::format::is_detached_version(detached_version));

    // We should be able to checkout the detached version and read transaction properties
    let checked_out = dataset.checkout_version(detached_version).await.unwrap();
    let tx = checked_out.read_transaction().await.unwrap().unwrap();
    let tx_props = tx.transaction_properties.unwrap();
    assert_eq!(
        tx_props.get("detached_key"),
        Some(&"detached_value".to_string())
    );

    // The detached dataset should have more rows
    assert_eq!(checked_out.count_rows(None).await.unwrap(), 6);

    // Create another detached transaction
    let batch3 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![7, 8, 9]))],
    )
    .unwrap();

    let mut properties2 = HashMap::new();
    properties2.insert("second_key".to_string(), "second_value".to_string());

    let transaction2 = InsertBuilder::new(dataset.clone())
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            transaction_properties: Some(Arc::new(properties2)),
            ..Default::default()
        })
        .execute_uncommitted(vec![batch3])
        .await
        .unwrap();

    CommitBuilder::new(dataset.clone())
        .with_detached(true)
        .execute(transaction2)
        .await
        .unwrap();

    // Now there should be two detached manifests
    let detached = dataset.list_detached_manifests().await.unwrap();
    assert_eq!(detached.len(), 2);

    // Both should be detached versions
    for loc in &detached {
        assert!(lance_table::format::is_detached_version(loc.version));
    }

    // Regular versions() should not include detached manifests
    let versions = dataset.versions().await.unwrap();
    assert_eq!(versions.len(), 1);
    assert_eq!(versions[0].version, 1);
}

#[cfg(feature = "unstable-action-transactions")]
mod action_transactions {
    use super::*;
    use crate::dataset::transaction::Operation;
    use lance_table::feature_flags::FLAG_EXPERIMENTAL;
    use lance_table::format::Fragment;
    use lance_table::transaction::{
        Action, AddFragments, AddIndex, NewFragment, UserAction, UserOperation,
    };

    fn schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]))
    }

    fn batch(values: Vec<i32>) -> RecordBatch {
        RecordBatch::try_new(schema(), vec![Arc::new(Int32Array::from(values))]).unwrap()
    }

    /// Write a single uncommitted fragment against `dataset` and return it.
    async fn write_one_fragment(dataset: Arc<Dataset>, values: Vec<i32>) -> Fragment {
        let reader = RecordBatchIterator::new([Ok(batch(values))], schema());
        let params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        let txn = InsertBuilder::new(dataset)
            .with_params(&params)
            .execute_uncommitted_stream(reader)
            .await
            .unwrap();
        match txn.operation {
            Operation::Append { mut fragments } => {
                assert_eq!(fragments.len(), 1, "expected exactly one fragment");
                fragments.pop().unwrap()
            }
            other => panic!("expected Append, got {other}"),
        }
    }

    // A pylance-equivalent flow: append two new fragments and register an index
    // covering only the first, in a single action-based transaction.
    #[tokio::test]
    async fn append_and_add_index_in_one_transaction() {
        let test_uri = TempStrDir::default();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new([Ok(batch(vec![1, 2, 3]))], schema()),
                &test_uri,
                None,
            )
            .await
            .unwrap(),
        );
        let read_version = dataset.manifest().version;
        // Initial dataset has a single fragment with id 0.
        assert_eq!(dataset.get_fragments().len(), 1);

        let frag_a = write_one_fragment(dataset.clone(), vec![4, 5, 6]).await;
        let frag_b = write_one_fragment(dataset.clone(), vec![7, 8]).await;

        let index_uuid = "11111111-1111-1111-1111-111111111111";
        let user_op = UserOperation {
            description: "append + index".to_string(),
            uuid: "test-op".to_string(),
            read_version,
            actions: vec![UserAction {
                description: "append two fragments, index the first".to_string(),
                actions: vec![
                    Action::AddFragments(AddFragments {
                        new_fragments: vec![
                            NewFragment {
                                local_id: 0,
                                fragment: frag_a,
                            },
                            NewFragment {
                                local_id: 1,
                                fragment: frag_b,
                            },
                        ],
                    }),
                    Action::AddIndex(AddIndex {
                        uuid: index_uuid.to_string(),
                        name: "id_idx".to_string(),
                        fields: vec![0],
                        covers_existing: vec![],
                        covers_local: vec![0],
                        index_details: None,
                    }),
                ],
            }],
        };

        let txn = Transaction::new(read_version, Operation::UserOperation(user_op), None);
        let committed = CommitBuilder::new(dataset).execute(txn).await.unwrap();

        // Both fragments landed alongside the original, and all rows are readable.
        assert_eq!(committed.manifest().version, read_version + 1);
        assert_eq!(committed.get_fragments().len(), 3);
        assert_eq!(committed.count_rows(None).await.unwrap(), 8);

        // The dataset now declares the experimental writer feature.
        assert_ne!(
            committed.manifest().writer_feature_flags & FLAG_EXPERIMENTAL,
            0
        );
        assert!(
            committed
                .manifest()
                .experimental_writer_features
                .iter()
                .any(|f| f == "action-transactions")
        );

        // The index was registered and covers exactly the first new fragment
        // (assigned id 1), not the second (id 2) or the original (id 0).
        let indices = committed.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        let index = &indices[0];
        assert_eq!(index.name, "id_idx");
        assert_eq!(index.uuid.to_string(), index_uuid);
        let bitmap = index.fragment_bitmap.as_ref().unwrap();
        assert_eq!(bitmap.len(), 1);
        assert!(bitmap.contains(1));
        assert!(!bitmap.contains(0));
        assert!(!bitmap.contains(2));
    }

    // An action-based append must not conflict with a concurrent plain Append:
    // the stale UserOperation rebases against the newer version and still lands.
    #[tokio::test]
    async fn does_not_conflict_with_concurrent_append() {
        let test_uri = TempStrDir::default();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new([Ok(batch(vec![1, 2, 3]))], schema()),
                &test_uri,
                None,
            )
            .await
            .unwrap(),
        );
        let read_version = dataset.manifest().version;

        // Prepare the action-based append against the original version.
        let frag = write_one_fragment(dataset.clone(), vec![4, 5]).await;
        let user_op = UserOperation {
            description: "action append".to_string(),
            uuid: "action-op".to_string(),
            read_version,
            actions: vec![UserAction {
                description: "append one fragment".to_string(),
                actions: vec![Action::AddFragments(AddFragments {
                    new_fragments: vec![NewFragment {
                        local_id: 0,
                        fragment: frag,
                    }],
                })],
            }],
        };
        let action_txn = Transaction::new(read_version, Operation::UserOperation(user_op), None);

        // A concurrent plain append commits first, advancing the version.
        let after_append = CommitBuilder::new(dataset.clone())
            .execute(Transaction::new(
                read_version,
                Operation::Append {
                    fragments: vec![write_one_fragment(dataset.clone(), vec![6, 7]).await],
                },
                None,
            ))
            .await
            .unwrap();
        assert_eq!(after_append.manifest().version, read_version + 1);

        // Now the stale action-based transaction still commits (rebased).
        let committed = CommitBuilder::new(dataset)
            .execute(action_txn)
            .await
            .unwrap();
        assert_eq!(committed.manifest().version, read_version + 2);
        // Original fragment + concurrent append + action append.
        assert_eq!(committed.get_fragments().len(), 3);
        assert_eq!(committed.count_rows(None).await.unwrap(), 7);
    }
}
