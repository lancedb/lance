// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::vec;

use crate::dataset::builder::DatasetBuilder;
use crate::dataset::transaction::{Operation, Transaction};
use crate::dataset::UpdateBuilder;
use crate::datatypes::Schema;
use crate::Dataset;
use lance_table::io::commit::ManifestNamingScheme;

use crate::dataset::write::{CommitBuilder, WriteMode, WriteParams};
use arrow_array::RecordBatch;
use arrow_array::RecordBatchReader;
use arrow_array::{types::Int32Type, RecordBatchIterator, UInt32Array};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_core::utils::tempfile::{TempDir, TempStdDir, TempStrDir};
use lance_datagen::{array, gen_batch, BatchCount, RowCount};
use lance_file::version::LanceFileVersion;

use crate::dataset::refs::branch_contents_path;
use futures::TryStreamExt;
use object_store::path::Path;
use rstest::rstest;
use std::cmp::Ordering;

fn assert_all_manifests_use_scheme(test_dir: &TempStdDir, scheme: ManifestNamingScheme) {
    let entries_names = test_dir
        .join("_versions")
        .read_dir()
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect::<Vec<_>>();
    assert!(
        entries_names
            .iter()
            .all(|name| ManifestNamingScheme::detect_scheme(name) == Some(scheme)),
        "Entries: {:?}",
        entries_names
    );
}

#[tokio::test]
async fn test_v2_manifest_path_create() {
    // Can create a dataset, using V2 paths
    let data = lance_datagen::gen_batch()
        .col("key", array::step::<Int32Type>())
        .into_batch_rows(RowCount::from(10))
        .unwrap();
    let test_dir = TempStdDir::default();
    let test_uri = test_dir.to_str().unwrap();
    Dataset::write(
        RecordBatchIterator::new([Ok(data.clone())], data.schema().clone()),
        test_uri,
        Some(WriteParams {
            enable_v2_manifest_paths: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_all_manifests_use_scheme(&test_dir, ManifestNamingScheme::V2);

    // Appending to it will continue to use those paths
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(data.clone())], data.schema().clone()),
        test_uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_all_manifests_use_scheme(&test_dir, ManifestNamingScheme::V2);

    UpdateBuilder::new(Arc::new(dataset))
        .update_where("key = 5")
        .unwrap()
        .set("key", "200")
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap();

    assert_all_manifests_use_scheme(&test_dir, ManifestNamingScheme::V2);
}

#[tokio::test]
async fn test_v2_manifest_path_commit() {
    let schema = Schema::try_from(&ArrowSchema::new(vec![ArrowField::new(
        "x",
        DataType::Int32,
        false,
    )]))
    .unwrap();
    let operation = Operation::Overwrite {
        fragments: vec![],
        schema,
        config_upsert_values: None,
        initial_bases: None,
    };
    let test_dir = TempStdDir::default();
    let test_uri = test_dir.to_str().unwrap();
    let dataset = Dataset::commit(
        test_uri,
        operation,
        None,
        None,
        None,
        Default::default(),
        true, // enable_v2_manifest_paths
    )
    .await
    .unwrap();

    assert!(dataset.manifest_location.naming_scheme == ManifestNamingScheme::V2);

    assert_all_manifests_use_scheme(&test_dir, ManifestNamingScheme::V2);
}

#[tokio::test]
async fn test_strict_overwrite() {
    let schema = Schema::try_from(&ArrowSchema::new(vec![ArrowField::new(
        "x",
        DataType::Int32,
        false,
    )]))
    .unwrap();
    let operation = Operation::Overwrite {
        fragments: vec![],
        schema,
        config_upsert_values: None,
        initial_bases: None,
    };
    let test_uri = TempStrDir::default();
    let read_version_0_transaction = Transaction::new(0, operation, None);
    let strict_builder = CommitBuilder::new(&test_uri).with_max_retries(0);
    let unstrict_builder = CommitBuilder::new(&test_uri).with_max_retries(1);
    strict_builder
        .clone()
        .execute(read_version_0_transaction.clone())
        .await
        .expect("Strict overwrite should succeed when writing a new dataset");
    strict_builder
        .clone()
        .execute(read_version_0_transaction.clone())
        .await
        .expect_err("Strict overwrite should fail when committing to a stale version");
    unstrict_builder
        .clone()
        .execute(read_version_0_transaction.clone())
        .await
        .expect("Unstrict overwrite should succeed when committing to a stale version");
}

#[rstest]
#[tokio::test]
async fn test_restore(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    // Create a table
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::UInt32,
        false,
    )]));

    let test_uri = TempStrDir::default();

    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from_iter_values(0..100))],
    );
    let reader = RecordBatchIterator::new(vec![data.unwrap()].into_iter().map(Ok), schema);
    let mut dataset = Dataset::write(
        reader,
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.manifest.version, 1);
    let original_manifest = dataset.manifest.clone();

    // Delete some rows
    dataset.delete("i > 50").await.unwrap();
    assert_eq!(dataset.manifest.version, 2);

    // Checkout a previous version
    let mut dataset = dataset.checkout_version(1).await.unwrap();
    assert_eq!(dataset.manifest.version, 1);
    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 1);
    assert_eq!(dataset.count_fragments(), 1);
    assert_eq!(fragments[0].metadata.deletion_file, None);
    assert_eq!(dataset.manifest, original_manifest);

    // Checkout latest and then go back.
    dataset.checkout_latest().await.unwrap();
    assert_eq!(dataset.manifest.version, 2);
    let mut dataset = dataset.checkout_version(1).await.unwrap();

    // Restore to a previous version
    dataset.restore().await.unwrap();
    assert_eq!(dataset.manifest.version, 3);
    assert_eq!(dataset.manifest.fragments, original_manifest.fragments);
    assert_eq!(dataset.manifest.schema, original_manifest.schema);

    // Delete some rows again (make sure we can still write as usual)
    dataset.delete("i > 30").await.unwrap();
    assert_eq!(dataset.manifest.version, 4);
    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 1);
    assert_eq!(dataset.count_fragments(), 1);
    assert!(fragments[0].metadata.deletion_file.is_some());
}

#[rstest]
#[tokio::test]
async fn test_tag(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    // Create a table
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::UInt32,
        false,
    )]));

    let test_uri = TempStrDir::default();

    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from_iter_values(0..100))],
    );
    let reader = RecordBatchIterator::new(vec![data.unwrap()].into_iter().map(Ok), schema);
    let mut dataset = Dataset::write(
        reader,
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.manifest.version, 1);

    // delete some rows
    dataset.delete("i > 50").await.unwrap();
    assert_eq!(dataset.manifest.version, 2);

    assert_eq!(dataset.tags().list().await.unwrap().len(), 0);

    let bad_tag_creation = dataset.tags().create("tag1", 3).await;
    assert_eq!(
        bad_tag_creation.err().unwrap().to_string(),
        "Version not found error: version Main::3 does not exist"
    );

    let bad_tag_deletion = dataset.tags().delete("tag1").await;
    assert_eq!(
        bad_tag_deletion.err().unwrap().to_string(),
        "Ref not found error: tag tag1 does not exist"
    );

    dataset.tags().create("tag1", 1).await.unwrap();

    assert_eq!(dataset.tags().list().await.unwrap().len(), 1);

    let another_bad_tag_creation = dataset.tags().create("tag1", 1).await;
    assert_eq!(
        another_bad_tag_creation.err().unwrap().to_string(),
        "Ref conflict error: tag tag1 already exists"
    );

    dataset.tags().delete("tag1").await.unwrap();

    assert_eq!(dataset.tags().list().await.unwrap().len(), 0);

    dataset.tags().create("tag1", 1).await.unwrap();
    dataset.tags().create("tag2", 1).await.unwrap();
    dataset.tags().create("v1.0.0-rc1", 2).await.unwrap();

    let default_order = dataset.tags().list_tags_ordered(None).await.unwrap();
    let default_names: Vec<_> = default_order.iter().map(|t| &t.0).collect();
    assert_eq!(
        default_names,
        ["v1.0.0-rc1", "tag1", "tag2"],
        "Default ordering mismatch"
    );

    let asc_order = dataset
        .tags()
        .list_tags_ordered(Some(Ordering::Less))
        .await
        .unwrap();
    let asc_names: Vec<_> = asc_order.iter().map(|t| &t.0).collect();
    assert_eq!(
        asc_names,
        ["tag1", "tag2", "v1.0.0-rc1"],
        "Ascending ordering mismatch"
    );

    let desc_order = dataset
        .tags()
        .list_tags_ordered(Some(Ordering::Greater))
        .await
        .unwrap();
    let desc_names: Vec<_> = desc_order.iter().map(|t| &t.0).collect();
    assert_eq!(
        desc_names,
        ["v1.0.0-rc1", "tag1", "tag2"],
        "Descending ordering mismatch"
    );

    assert_eq!(dataset.tags().list().await.unwrap().len(), 3);

    let bad_checkout = dataset.checkout_version("tag3").await;
    assert_eq!(
        bad_checkout.err().unwrap().to_string(),
        "Ref not found error: tag tag3 does not exist"
    );

    dataset = dataset.checkout_version("tag1").await.unwrap();
    assert_eq!(dataset.manifest.version, 1);

    let first_ver = DatasetBuilder::from_uri(&test_uri)
        .with_tag("tag1")
        .load()
        .await
        .unwrap();
    assert_eq!(first_ver.version().version, 1);

    // test update tag
    let bad_tag_update = dataset.tags().update("tag3", 1).await;
    assert_eq!(
        bad_tag_update.err().unwrap().to_string(),
        "Ref not found error: tag tag3 does not exist"
    );

    let another_bad_tag_update = dataset.tags().update("tag1", 3).await;
    assert_eq!(
        another_bad_tag_update.err().unwrap().to_string(),
        "Version not found error: version 3 does not exist"
    );

    dataset.tags().update("tag1", 2).await.unwrap();
    dataset = dataset.checkout_version("tag1").await.unwrap();
    assert_eq!(dataset.manifest.version, 2);

    dataset.tags().update("tag1", 1).await.unwrap();
    dataset = dataset.checkout_version("tag1").await.unwrap();
    assert_eq!(dataset.manifest.version, 1);
}

#[rstest]
#[tokio::test]
async fn test_fragment_id_zero_not_reused() {
    // Test case 1: Fragment id zero isn't re-used
    // 1. Create a dataset with 1 fragment
    // 2. Delete all rows
    // 3. Append another fragment
    // 4. Assert new fragment has id 1 not 0

    let test_uri = TempStrDir::default();

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::UInt32,
        false,
    )]));

    // Create dataset with 1 fragment
    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from_iter_values(0..10))],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![data].into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(batches, &test_uri, None).await.unwrap();

    // Verify we have 1 fragment with id 0
    assert_eq!(dataset.get_fragments().len(), 1);
    assert_eq!(dataset.get_fragments()[0].id(), 0);
    assert_eq!(dataset.manifest.max_fragment_id(), Some(0));

    // Delete all rows
    dataset.delete("true").await.unwrap();

    // After deletion, dataset should be empty but max_fragment_id preserved
    assert_eq!(dataset.get_fragments().len(), 0);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 0);
    assert_eq!(dataset.manifest.max_fragment_id(), Some(0));

    // Append another fragment
    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from_iter_values(20..30))],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![data].into_iter().map(Ok), schema.clone());
    let write_params = WriteParams {
        mode: WriteMode::Append,
        ..Default::default()
    };
    let dataset = Dataset::write(batches, &test_uri, Some(write_params))
        .await
        .unwrap();

    // Assert new fragment has id 1, not 0
    assert_eq!(dataset.get_fragments().len(), 1);
    assert_eq!(dataset.get_fragments()[0].id(), 1);
    assert_eq!(dataset.manifest.max_fragment_id(), Some(1));
}

#[rstest]
#[tokio::test]
async fn test_fragment_id_never_reset() {
    // Test case 2: Fragment id is never reset, even if all rows are deleted
    // 1. Create dataset with N fragments
    // 2. Delete all rows
    // 3. Append more fragments
    // 4. Assert new fragments have ids >= N

    let test_uri = TempStrDir::default();

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::UInt32,
        false,
    )]));

    // Create dataset with 3 fragments (N=3)
    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from_iter_values(0..30))],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(data)], schema.clone());
    let write_params = WriteParams {
        max_rows_per_file: 10, // Force multiple fragments
        ..Default::default()
    };
    let mut dataset = Dataset::write(batches, &test_uri, Some(write_params))
        .await
        .unwrap();

    // Verify we have 3 fragments with ids 0, 1, 2
    assert_eq!(dataset.get_fragments().len(), 3);
    assert_eq!(dataset.get_fragments()[0].id(), 0);
    assert_eq!(dataset.get_fragments()[1].id(), 1);
    assert_eq!(dataset.get_fragments()[2].id(), 2);
    assert_eq!(dataset.manifest.max_fragment_id(), Some(2));

    // Delete all rows
    dataset.delete("true").await.unwrap();

    // After deletion, dataset should be empty but max_fragment_id preserved
    assert_eq!(dataset.get_fragments().len(), 0);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 0);
    assert_eq!(dataset.manifest.max_fragment_id(), Some(2));

    // Append more fragments (2 new fragments)
    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from_iter_values(100..120))],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(data)], schema.clone());
    let write_params = WriteParams {
        mode: WriteMode::Append,
        max_rows_per_file: 10, // Force multiple fragments
        ..Default::default()
    };
    let dataset = Dataset::write(batches, &test_uri, Some(write_params))
        .await
        .unwrap();

    // Assert new fragments have ids >= N (3, 4)
    assert_eq!(dataset.get_fragments().len(), 2);
    assert_eq!(dataset.get_fragments()[0].id(), 3);
    assert_eq!(dataset.get_fragments()[1].id(), 4);
    assert_eq!(dataset.manifest.max_fragment_id(), Some(4));
}

#[tokio::test]
async fn test_branch() {
    let tempdir = TempDir::default();
    let test_uri = tempdir.path_str();
    let data_storage_version = LanceFileVersion::Stable;

    // Generate consistent test data batches
    let generate_data = |prefix: &str, start_id: i32, row_count: u64| {
        gen_batch()
            .col("id", array::step_custom::<Int32Type>(start_id, 1))
            .col("value", array::fill_utf8(format!("{prefix}_data")))
            .into_reader_rows(RowCount::from(row_count), BatchCount::from(1))
    };

    // Reusable dataset writer with configurable mode
    async fn write_dataset(
        uri: &str,
        data_reader: impl RecordBatchReader + Send + 'static,
        mode: WriteMode,
        version: LanceFileVersion,
    ) -> Dataset {
        let params = WriteParams {
            max_rows_per_file: 100,
            max_rows_per_group: 20,
            data_storage_version: Some(version),
            mode,
            ..Default::default()
        };
        Dataset::write(data_reader, uri, Some(params))
            .await
            .unwrap()
    }

    // Unified dataset scanning and row counting
    async fn collect_rows(dataset: &Dataset) -> (usize, Vec<RecordBatch>) {
        let batches = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        (batches.iter().map(|b| b.num_rows()).sum(), batches)
    }

    // Phase 1: Create empty dataset, write data batch 1, create branch1 based on version_number, write data batch 2
    let mut dataset = write_dataset(
        &test_uri,
        generate_data("batch1", 0, 50),
        WriteMode::Create,
        data_storage_version,
    )
    .await;

    let original_version = dataset.version().version;
    assert_eq!(original_version, 1);

    // Create branch1 on the latest version and write data batch 2
    let mut branch1_dataset = dataset
        .create_branch("branch1", original_version, None)
        .await
        .unwrap();
    assert_eq!(branch1_dataset.uri, format!("{}/tree/branch1", test_uri));

    branch1_dataset = write_dataset(
        branch1_dataset.uri(),
        generate_data("batch2", 50, 30),
        WriteMode::Append,
        data_storage_version,
    )
    .await;

    // Phase 2: Create branch2 based on branch1's latest version_number, write data batch 3
    let mut branch2_dataset = branch1_dataset
        .create_branch(
            "dev/branch2",
            ("branch1", branch1_dataset.version().version),
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        branch2_dataset.uri,
        format!("{}/tree/dev/branch2", test_uri)
    );

    branch2_dataset = write_dataset(
        branch2_dataset.uri(),
        generate_data("batch3", 80, 20),
        WriteMode::Append,
        data_storage_version,
    )
    .await;

    // Phase 3: Create a tag on branch2, the actual tag content is under root dataset
    // create branch3 based on that tag, write data batch 4
    branch2_dataset
        .tags()
        .create_on_branch(
            "tag1",
            branch2_dataset.version().version,
            Some("dev/branch2"),
        )
        .await
        .unwrap();

    let mut branch3_dataset = branch2_dataset
        .create_branch("feature/nathan/branch3", "tag1", None)
        .await
        .unwrap();
    assert_eq!(
        branch3_dataset.uri,
        format!("{}/tree/feature/nathan/branch3", test_uri)
    );

    branch3_dataset = write_dataset(
        branch3_dataset.uri(),
        generate_data("batch4", 100, 25),
        WriteMode::Append,
        data_storage_version,
    )
    .await;

    // Verify data correctness and independence of each branch
    // Main branch only has data 1 (50 rows)
    let main_dataset = Dataset::open(&test_uri).await.unwrap();
    let (main_rows, _) = collect_rows(&main_dataset).await;
    assert_eq!(main_rows, 50); // only batch1
    assert_eq!(main_dataset.version().version, 1);

    // branch1 has data 1 + 2 (80 rows)
    let updated_branch1 = Dataset::open(branch1_dataset.uri()).await.unwrap();
    let (branch1_rows, _) = collect_rows(&updated_branch1).await;
    assert_eq!(branch1_rows, 80); // batch1+batch2
    assert_eq!(updated_branch1.version().version, 2);

    // branch2 has data 1 + 2 + 3 (100 rows)
    let updated_branch2 = Dataset::open(branch2_dataset.uri()).await.unwrap();
    let (branch2_rows, _) = collect_rows(&updated_branch2).await;
    assert_eq!(branch2_rows, 100); // batch1+batch2+batch3
    assert_eq!(updated_branch2.version().version, 3);

    // branch3 has data 1 + 2 + 3 + 4 (125 rows)
    let updated_branch3 = Dataset::open(branch3_dataset.uri()).await.unwrap();
    let (branch3_rows, _) = collect_rows(&updated_branch3).await;
    assert_eq!(branch3_rows, 125); // batch1+batch2+batch3+batch4
    assert_eq!(updated_branch3.version().version, 4);

    // Use list_branches to get branch list and verify each field of branch_content
    let branches = dataset.list_branches().await.unwrap();
    assert_eq!(branches.len(), 3);
    assert!(branches.contains_key("branch1"));
    assert!(branches.contains_key("dev/branch2"));
    assert!(branches.contains_key("feature/nathan/branch3"));

    // Verify branch1 content
    let branch1_content = branches.get("branch1").unwrap();
    assert_eq!(branch1_content.parent_branch, None); // Created based on main branch
    assert_eq!(branch1_content.parent_version, 1);
    assert!(branch1_content.create_at > 0);
    assert!(branch1_content.manifest_size > 0);

    // Verify branch2 content
    let branch2_content = branches.get("dev/branch2").unwrap();
    assert_eq!(branch2_content.parent_branch.as_deref().unwrap(), "branch1");
    assert_eq!(branch2_content.parent_version, 2);
    assert!(branch2_content.create_at > 0);
    assert!(branch2_content.manifest_size > 0);
    assert!(branch2_content.create_at >= branch1_content.create_at);

    // Verify branch3 content
    let branch3_content = branches.get("feature/nathan/branch3").unwrap();
    // Created based on tag pointed to branch2
    assert_eq!(
        branch3_content.parent_branch.as_deref().unwrap(),
        "dev/branch2"
    );
    assert_eq!(branch3_content.parent_version, 3);
    assert!(branch3_content.create_at > 0);
    assert!(branch3_content.manifest_size > 0);
    assert!(branch3_content.create_at >= branch2_content.create_at);

    // Verify checkout_branch
    let checkout_branch1 = main_dataset.checkout_branch("branch1").await.unwrap();
    let checkout_branch2 = checkout_branch1
        .checkout_branch("dev/branch2")
        .await
        .unwrap();
    let checkout_branch2_tag = checkout_branch1.checkout_version("tag1").await.unwrap();
    let checkout_branch3 = checkout_branch2_tag
        .checkout_branch("feature/nathan/branch3")
        .await
        .unwrap();
    let checkout_branch3_at_version3 = checkout_branch2
        .checkout_version(("feature/nathan/branch3", 3))
        .await
        .unwrap();
    assert_eq!(checkout_branch3.version().version, 4);
    assert_eq!(checkout_branch3_at_version3.version().version, 3);
    assert_eq!(checkout_branch2.version().version, 3);
    assert_eq!(checkout_branch2_tag.version().version, 3);
    assert_eq!(checkout_branch1.version().version, 2);
    assert_eq!(checkout_branch3.count_rows(None).await.unwrap(), 125);
    assert_eq!(
        checkout_branch3_at_version3.count_rows(None).await.unwrap(),
        100
    );
    assert_eq!(checkout_branch2.count_rows(None).await.unwrap(), 100);
    assert_eq!(checkout_branch2_tag.count_rows(None).await.unwrap(), 100);
    assert_eq!(checkout_branch1.count_rows(None).await.unwrap(), 80);
    assert_eq!(
        checkout_branch3.manifest.branch.as_deref().unwrap(),
        "feature/nathan/branch3"
    );
    assert_eq!(
        checkout_branch3_at_version3
            .manifest
            .branch
            .as_deref()
            .unwrap(),
        "feature/nathan/branch3"
    );
    assert_eq!(
        checkout_branch2.manifest.branch.as_deref().unwrap(),
        "dev/branch2"
    );
    assert_eq!(
        checkout_branch2_tag.manifest.branch.as_deref().unwrap(),
        "dev/branch2"
    );
    assert_eq!(
        checkout_branch1.manifest.branch.as_deref().unwrap(),
        "branch1"
    );

    let mut dataset = main_dataset;
    // Finally delete all branches
    dataset.delete_branch("branch1").await.unwrap();
    dataset.delete_branch("dev/branch2").await.unwrap();
    // Test deleting zombie branch
    let root_location = dataset.refs.root().unwrap();
    let branch_file = branch_contents_path(&root_location.path, "feature/nathan/branch3");
    dataset.object_store.delete(&branch_file).await.unwrap();
    // Now "feature/nathan/branch3" is a zombie branch
    // Use delete_branch to verify if the directory is cleaned up
    dataset
        .force_delete_branch("feature/nathan/branch3")
        .await
        .unwrap();
    let cleaned_path = Path::parse(format!("{}/tree/feature", test_uri)).unwrap();
    assert!(!dataset.object_store.exists(&cleaned_path).await.unwrap());

    // Verify list_branches is empty
    let branches_after_delete = dataset.list_branches().await.unwrap();
    assert!(branches_after_delete.is_empty());

    // Verify branch directories are all deleted cleanly
    let test_path = tempdir.obj_path();
    let branches = dataset
        .object_store
        .read_dir(test_path.child("tree"))
        .await
        .unwrap();
    assert!(branches.is_empty());
}
