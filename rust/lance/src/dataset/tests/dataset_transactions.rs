use std::collections::HashMap;
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::vec;

use crate::dataset::builder::DatasetBuilder;
use crate::dataset::transaction::{Operation, Transaction};
use crate::dataset::{write_manifest_file, ManifestWriteConfig};
use crate::io::ObjectStoreParams;
use crate::session::Session;
use crate::{Dataset, Result};
use lance_table::io::commit::ManifestNamingScheme;

use crate::dataset::write::{CommitBuilder, InsertBuilder, WriteMode, WriteParams};
use arrow_array::Array;
use arrow_array::RecordBatch;
use arrow_array::{types::Int32Type, Int32Array, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_core::utils::tempfile::{TempDir, TempStrDir};
use lance_datagen::{array, BatchCount, RowCount};
use lance_index::DatasetIndexExt;

use crate::datafusion::LanceTableProvider;
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;
use lance_datafusion::udf::register_functions;

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
        Arc::as_ptr(&dataset.object_store().inner),
        Arc::as_ptr(&dataset2.object_store().inner)
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
        Arc::as_ptr(&dataset.object_store().inner),
        Arc::as_ptr(&dataset3.object_store().inner)
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
    let mut dataset = Dataset::write(data, &test_uri, None).await.unwrap();
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
            let tx_path = ds.base.child("_transactions").child(tx_file.as_str());
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
    let stats = read_ds2.object_store().io_stats_incremental(); // Reset
    assert!(stats.read_bytes < 64 * 1024);
    // Because the manifest is so small, we should have opportunistically
    // cached the transaction in memory already.
    let inline_tx = read_ds2.read_transaction().await.unwrap().unwrap();
    let stats = read_ds2.object_store().io_stats_incremental();
    assert_eq!(stats.read_iops, 0);
    assert_eq!(stats.read_bytes, 0);
    assert_eq!(inline_tx, tx);

    // Case 3: manifest does not contain inline transaction, read should fall back to external transaction file
    let ds = create_dataset(2).await;
    let tx = make_tx(ds.manifest().version);
    let tx_file = crate::io::commit::write_transaction_file(ds.object_store(), &ds.base, &tx)
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
        ds.object_store(),
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
