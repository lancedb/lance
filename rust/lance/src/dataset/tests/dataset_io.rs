// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::vec;

use super::dataset_common::{create_file, require_send};

use crate::dataset::builder::DatasetBuilder;
use crate::dataset::WriteDestination;
use crate::dataset::WriteMode::Overwrite;
use crate::dataset::{write_manifest_file, ManifestWriteConfig};
use crate::session::Session;
use crate::{Dataset, Error, Result};
use lance_table::format::DataStorageFormat;

use crate::dataset::write::{WriteMode, WriteParams};
use arrow::array::as_struct_array;
use arrow::compute::concat_batches;
use arrow_array::RecordBatch;
use arrow_array::RecordBatchReader;
use arrow_array::{
    cast::as_string_array,
    types::{Float32Type, Int32Type},
    ArrayRef, Int32Array, Int64Array, Int8Array, Int8DictionaryArray, RecordBatchIterator,
    StringArray,
};
use arrow_array::{Array, FixedSizeListArray, Int16Array, Int16DictionaryArray, StructArray};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_arrow::bfloat16::{self, BFLOAT16_EXT_NAME};
use lance_arrow::{ARROW_EXT_META_KEY, ARROW_EXT_NAME_KEY};
use lance_core::utils::tempfile::{TempStdDir, TempStrDir};
use lance_datagen::{array, gen_batch, BatchCount, RowCount};
use lance_file::version::LanceFileVersion;
use lance_io::assert_io_eq;
use lance_table::feature_flags;

use futures::TryStreamExt;
use lance_table::io::manifest::read_manifest;
use rstest::rstest;

#[rstest]
#[lance_test_macros::test(tokio::test)]
async fn test_create_dataset(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    // Appending / Overwriting a dataset that does not exist is treated as Create
    for mode in [WriteMode::Create, WriteMode::Append, Overwrite] {
        let test_dir = TempStdDir::default();
        create_file(&test_dir, mode, data_storage_version).await
    }
}

#[rstest]
#[lance_test_macros::test(tokio::test)]
async fn test_create_and_fill_empty_dataset(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_uri = TempStrDir::default();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::Int32,
        false,
    )]));
    let i32_array: ArrayRef = Arc::new(Int32Array::new(vec![].into(), None));
    let batch = RecordBatch::try_from_iter(vec![("i", i32_array)]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
    // check schema of reader and original is same
    assert_eq!(schema.as_ref(), reader.schema().as_ref());
    let result = Dataset::write(
        reader,
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // check dataset empty
    assert_eq!(result.count_rows(None).await.unwrap(), 0);
    // Since the dataset is empty, will return None.
    assert_eq!(result.manifest.max_fragment_id(), None);

    // append rows to dataset
    let mut write_params = WriteParams {
        max_rows_per_file: 40,
        max_rows_per_group: 10,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    // We should be able to append even if the metadata doesn't exactly match.
    let schema_with_meta = Arc::new(
        schema
            .as_ref()
            .clone()
            .with_metadata([("key".to_string(), "value".to_string())].into()),
    );
    let batches = vec![RecordBatch::try_new(
        schema_with_meta,
        vec![Arc::new(Int32Array::from_iter_values(0..10))],
    )
    .unwrap()];
    write_params.mode = WriteMode::Append;
    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    Dataset::write(batches, &test_uri, Some(write_params))
        .await
        .unwrap();

    let expected_batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..10))],
    )
    .unwrap();

    // get actual dataset
    let actual_ds = Dataset::open(&test_uri).await.unwrap();
    // confirm schema is same
    let actual_schema = ArrowSchema::from(actual_ds.schema());
    assert_eq!(&actual_schema, schema.as_ref());
    // check num rows is 10
    assert_eq!(actual_ds.count_rows(None).await.unwrap(), 10);
    // Max fragment id is still 0 since we only have 1 fragment.
    assert_eq!(actual_ds.manifest.max_fragment_id(), Some(0));
    // check expected batch is correct
    let actual_batches = actual_ds
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    // sort
    let actual_batch = concat_batches(&schema, &actual_batches).unwrap();
    let idx_arr = actual_batch.column_by_name("i").unwrap();
    let sorted_indices = sort_to_indices(idx_arr, None, None).unwrap();
    let struct_arr: StructArray = actual_batch.into();
    let sorted_arr = arrow_select::take::take(&struct_arr, &sorted_indices, None).unwrap();
    let expected_struct_arr: StructArray = expected_batch.into();
    assert_eq!(&expected_struct_arr, as_struct_array(sorted_arr.as_ref()));
}

#[rstest]
#[lance_test_macros::test(tokio::test)]
async fn test_create_with_empty_iter(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_uri = TempStrDir::default();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::Int32,
        false,
    )]));
    let reader = RecordBatchIterator::new(vec![].into_iter().map(Ok), schema.clone());
    // check schema of reader and original is same
    assert_eq!(schema.as_ref(), reader.schema().as_ref());
    let write_params = Some(WriteParams {
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    });
    let result = Dataset::write(reader, &test_uri, write_params)
        .await
        .unwrap();

    // check dataset empty
    assert_eq!(result.count_rows(None).await.unwrap(), 0);
    // Since the dataset is empty, will return None.
    assert_eq!(result.manifest.max_fragment_id(), None);
}

#[tokio::test]
async fn test_load_manifest_iops() {
    // Use consistent session so memory store can be reused.
    let session = Arc::new(Session::default());
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let _original_ds = Dataset::write(
        batches,
        "memory://test",
        Some(WriteParams {
            session: Some(session.clone()),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let _ = _original_ds.object_store().io_stats_incremental(); //reset

    let _dataset = DatasetBuilder::from_uri("memory://test")
        .with_session(session)
        .load()
        .await
        .unwrap();

    // There should be only two IOPS:
    // 1. List _versions directory to get the latest manifest location
    // 2. Read the manifest file. (The manifest is small enough to be read in one go.
    //    Larger manifests would result in more IOPS.)
    let io_stats = _dataset.object_store().io_stats_incremental();
    assert_io_eq!(io_stats, read_iops, 2);
}

#[rstest]
#[tokio::test]
async fn test_write_params(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    use crate::dataset::fragment::FragReadConfig;

    let test_uri = TempStrDir::default();

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::Int32,
        false,
    )]));
    let num_rows: usize = 1_000;
    let batches = vec![RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..num_rows as i32))],
    )
    .unwrap()];

    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

    let write_params = WriteParams {
        max_rows_per_file: 100,
        max_rows_per_group: 10,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let dataset = Dataset::write(batches, &test_uri, Some(write_params))
        .await
        .unwrap();

    assert_eq!(dataset.count_rows(None).await.unwrap(), num_rows);

    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 10);
    assert_eq!(dataset.count_fragments(), 10);
    for fragment in &fragments {
        assert_eq!(fragment.count_rows(None).await.unwrap(), 100);
        let reader = fragment
            .open(dataset.schema(), FragReadConfig::default())
            .await
            .unwrap();
        // No group / batch concept in v2
        if data_storage_version == LanceFileVersion::Legacy {
            assert_eq!(reader.legacy_num_batches(), 10);
            for i in 0..reader.legacy_num_batches() as u32 {
                assert_eq!(reader.legacy_num_rows_in_batch(i).unwrap(), 10);
            }
        }
    }
}

#[rstest]
#[tokio::test]
async fn test_write_manifest(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    use lance_table::feature_flags::FLAG_UNKNOWN;

    let test_uri = TempStrDir::default();

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::Int32,
        false,
    )]));
    let batches = vec![RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..20))],
    )
    .unwrap()];

    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let write_fut = Dataset::write(
        batches,
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            auto_cleanup: None,
            ..Default::default()
        }),
    );
    let write_fut = require_send(write_fut);
    let mut dataset = write_fut.await.unwrap();

    // Check it has no flags
    let manifest = read_manifest(
        dataset.object_store(),
        &dataset
            .commit_handler
            .resolve_latest_location(&dataset.base, dataset.object_store())
            .await
            .unwrap()
            .path,
        None,
    )
    .await
    .unwrap();

    assert_eq!(
        manifest.data_storage_format,
        DataStorageFormat::new(data_storage_version)
    );
    assert_eq!(manifest.reader_feature_flags, 0);

    // Create one with deletions
    dataset.delete("i < 10").await.unwrap();
    dataset.validate().await.unwrap();

    // Check it set the flag
    let mut manifest = read_manifest(
        dataset.object_store(),
        &dataset
            .commit_handler
            .resolve_latest_location(&dataset.base, dataset.object_store())
            .await
            .unwrap()
            .path,
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        manifest.writer_feature_flags,
        feature_flags::FLAG_DELETION_FILES
    );
    assert_eq!(
        manifest.reader_feature_flags,
        feature_flags::FLAG_DELETION_FILES
    );

    // Write with custom manifest
    manifest.writer_feature_flags |= FLAG_UNKNOWN; // Set another flag
    manifest.reader_feature_flags |= FLAG_UNKNOWN;
    manifest.version += 1;
    write_manifest_file(
        dataset.object_store(),
        dataset.commit_handler.as_ref(),
        &dataset.base,
        &mut manifest,
        None,
        &ManifestWriteConfig {
            auto_set_feature_flags: false,
            timestamp: None,
            use_stable_row_ids: false,
            use_legacy_format: None,
            storage_format: None,
            disable_transaction_file: false,
        },
        dataset.manifest_location.naming_scheme,
        None,
    )
    .await
    .unwrap();

    // Check it rejects reading it
    let read_result = Dataset::open(&test_uri).await;
    assert!(matches!(read_result, Err(Error::NotSupported { .. })));

    // Check it rejects writing to it.
    let batches = vec![RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..20))],
    )
    .unwrap()];
    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let write_result = Dataset::write(
        batches,
        &test_uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await;

    assert!(matches!(write_result, Err(Error::NotSupported { .. })));
}

#[rstest]
#[tokio::test]
async fn append_dataset(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_uri = TempStrDir::default();

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::Int32,
        false,
    )]));
    let batches = vec![RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..20))],
    )
    .unwrap()];

    let mut write_params = WriteParams {
        max_rows_per_file: 40,
        max_rows_per_group: 10,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    Dataset::write(batches, &test_uri, Some(write_params.clone()))
        .await
        .unwrap();

    let batches = vec![RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(20..40))],
    )
    .unwrap()];
    write_params.mode = WriteMode::Append;
    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    Dataset::write(batches, &test_uri, Some(write_params.clone()))
        .await
        .unwrap();

    let expected_batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..40))],
    )
    .unwrap();

    let actual_ds = Dataset::open(&test_uri).await.unwrap();
    assert_eq!(actual_ds.version().version, 2);
    let actual_schema = ArrowSchema::from(actual_ds.schema());
    assert_eq!(&actual_schema, schema.as_ref());

    let actual_batches = actual_ds
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    // sort
    let actual_batch = concat_batches(&schema, &actual_batches).unwrap();
    let idx_arr = actual_batch.column_by_name("i").unwrap();
    let sorted_indices = sort_to_indices(idx_arr, None, None).unwrap();
    let struct_arr: StructArray = actual_batch.into();
    let sorted_arr = arrow_select::take::take(&struct_arr, &sorted_indices, None).unwrap();

    let expected_struct_arr: StructArray = expected_batch.into();
    assert_eq!(&expected_struct_arr, as_struct_array(sorted_arr.as_ref()));

    // Each fragments has different fragment ID
    assert_eq!(
        actual_ds
            .fragments()
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        (0..2).collect::<Vec<_>>()
    )
}

#[rstest]
#[tokio::test]
async fn test_shallow_clone_with_hybrid_paths(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_dir = TempStdDir::default();
    let base_dir = test_dir.join("base");
    let test_uri = base_dir.to_str().unwrap();
    let clone_dir = test_dir.join("clone");
    let cloned_uri = clone_dir.to_str().unwrap();

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

    // Create initial dataset
    let mut dataset = write_dataset(
        test_uri,
        generate_data("initial", 0, 50),
        WriteMode::Create,
        data_storage_version,
    )
    .await;

    // Store original state for comparison
    let original_version = dataset.version().version;
    let original_fragment_count = dataset.fragments().len();

    // Create tag and shallow clone
    dataset
        .tags()
        .create("test_tag", original_version)
        .await
        .unwrap();
    let cloned_dataset = dataset
        .shallow_clone(cloned_uri, "test_tag", None)
        .await
        .unwrap();

    // Verify cloned dataset state
    let (cloned_rows, _) = collect_rows(&cloned_dataset).await;
    assert_eq!(cloned_rows, 50);
    assert_eq!(cloned_dataset.version().version, original_version);

    // Append data to cloned dataset
    let updated_cloned = write_dataset(
        cloned_uri,
        generate_data("cloned_new", 50, 30),
        WriteMode::Append,
        data_storage_version,
    )
    .await;

    // Verify updated cloned dataset
    let (updated_cloned_rows, updated_batches) = collect_rows(&updated_cloned).await;
    assert_eq!(updated_cloned_rows, 80);
    assert_eq!(updated_cloned.version().version, original_version + 1);

    // Append data to original dataset
    let updated_original = write_dataset(
        test_uri,
        generate_data("original_new", 50, 25),
        WriteMode::Append,
        data_storage_version,
    )
    .await;

    // Verify updated original dataset
    let (original_rows, _) = collect_rows(&updated_original).await;
    assert_eq!(original_rows, 75);
    assert_eq!(updated_original.version().version, original_version + 1);

    // Final validations
    // Verify cloned dataset isolation
    let final_cloned = Dataset::open(cloned_uri).await.unwrap();
    let (final_cloned_rows, _) = collect_rows(&final_cloned).await;

    // Data integrity check
    let combined_batch = concat_batches(&updated_batches[0].schema(), &updated_batches).unwrap();
    assert_eq!(combined_batch.column_by_name("id").unwrap().len(), 80);
    assert_eq!(combined_batch.column_by_name("value").unwrap().len(), 80);

    // Fragment count validation
    assert_eq!(
        updated_original.fragments().len(),
        original_fragment_count + 1
    );
    assert_eq!(final_cloned.fragments().len(), original_fragment_count + 1);

    // Final assertions
    assert_eq!(final_cloned_rows, 80);
    assert_eq!(final_cloned.version().version, original_version + 1);
}

#[rstest]
#[tokio::test]
async fn test_shallow_clone_multiple_times(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_uri = TempStrDir::default();
    let append_row_count = 36;

    // Async dataset writer function
    async fn write_dataset(
        dest: impl Into<WriteDestination<'_>>,
        row_count: u64,
        mode: WriteMode,
        version: LanceFileVersion,
    ) -> Dataset {
        let data = gen_batch()
            .col("index", array::step::<Int32Type>())
            .col("category", array::fill_utf8("base".to_string()))
            .col("score", array::step_custom::<Float32Type>(1.0, 0.5));
        Dataset::write(
            data.into_reader_rows(RowCount::from(row_count), BatchCount::from(1)),
            dest,
            Some(WriteParams {
                max_rows_per_file: 60,
                max_rows_per_group: 12,
                mode,
                data_storage_version: Some(version),
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    let mut current_dataset = write_dataset(
        &test_uri,
        append_row_count,
        WriteMode::Create,
        data_storage_version,
    )
    .await;

    let test_round = 3;
    // Generate clone paths
    let clone_paths = (1..=test_round)
        .map(|i| format!("{}/clone{}", test_uri, i))
        .collect::<Vec<_>>();
    let mut cloned_datasets = Vec::with_capacity(test_round);

    // Unified cloning procedure, write a fragment to each cloned dataset.
    for path in clone_paths.iter() {
        current_dataset
            .tags()
            .create("v1", current_dataset.latest_version_id().await.unwrap())
            .await
            .unwrap();

        current_dataset = current_dataset
            .shallow_clone(path, "v1", None)
            .await
            .unwrap();
        current_dataset = write_dataset(
            Arc::new(current_dataset),
            append_row_count,
            WriteMode::Append,
            data_storage_version,
        )
        .await;
        cloned_datasets.push(current_dataset.clone());
    }

    // Validation function
    async fn validate_dataset(
        dataset: &Dataset,
        expected_rows: usize,
        expected_fragments_count: usize,
        expected_base_paths_count: usize,
    ) {
        let batches = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, expected_rows);
        assert_eq!(dataset.fragments().len(), expected_fragments_count);
        assert_eq!(
            dataset.manifest().base_paths.len(),
            expected_base_paths_count
        );
    }

    // Verify cloned datasets row count, fragment count, base_path count
    for (i, ds) in cloned_datasets.iter().enumerate() {
        validate_dataset(ds, 36 * (i + 2), i + 2, i + 1).await;
    }

    // Verify original dataset row count, fragment count, base_path count
    let original = Dataset::open(&test_uri).await.unwrap();
    validate_dataset(&original, 36, 1, 0).await;
}

#[rstest]
#[tokio::test]
async fn test_self_dataset_append(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_uri = TempStrDir::default();

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::Int32,
        false,
    )]));
    let batches = vec![RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..20))],
    )
    .unwrap()];

    let mut write_params = WriteParams {
        max_rows_per_file: 40,
        max_rows_per_group: 10,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let mut ds = Dataset::write(batches, &test_uri, Some(write_params.clone()))
        .await
        .unwrap();

    let batches = vec![RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(20..40))],
    )
    .unwrap()];
    write_params.mode = WriteMode::Append;
    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

    ds.append(batches, Some(write_params.clone()))
        .await
        .unwrap();

    let expected_batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..40))],
    )
    .unwrap();

    let actual_ds = Dataset::open(&test_uri).await.unwrap();
    assert_eq!(actual_ds.version().version, 2);
    // validate fragment ids
    assert_eq!(actual_ds.fragments().len(), 2);
    assert_eq!(
        actual_ds
            .fragments()
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        (0..2).collect::<Vec<_>>()
    );

    let actual_schema = ArrowSchema::from(actual_ds.schema());
    assert_eq!(&actual_schema, schema.as_ref());

    let actual_batches = actual_ds
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    // sort
    let actual_batch = concat_batches(&schema, &actual_batches).unwrap();
    let idx_arr = actual_batch.column_by_name("i").unwrap();
    let sorted_indices = sort_to_indices(idx_arr, None, None).unwrap();
    let struct_arr: StructArray = actual_batch.into();
    let sorted_arr = arrow_select::take::take(&struct_arr, &sorted_indices, None).unwrap();

    let expected_struct_arr: StructArray = expected_batch.into();
    assert_eq!(&expected_struct_arr, as_struct_array(sorted_arr.as_ref()));

    actual_ds.validate().await.unwrap();
}

#[rstest]
#[tokio::test]
async fn test_self_dataset_append_schema_different(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_uri = TempStrDir::default();

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::Int32,
        false,
    )]));
    let batches = vec![RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..20))],
    )
    .unwrap()];

    let other_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::Int64,
        false,
    )]));
    let other_batches = vec![RecordBatch::try_new(
        other_schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(0..20))],
    )
    .unwrap()];

    let mut write_params = WriteParams {
        max_rows_per_file: 40,
        max_rows_per_group: 10,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let mut ds = Dataset::write(batches, &test_uri, Some(write_params.clone()))
        .await
        .unwrap();

    write_params.mode = WriteMode::Append;
    let other_batches =
        RecordBatchIterator::new(other_batches.into_iter().map(Ok), other_schema.clone());

    let result = ds.append(other_batches, Some(write_params.clone())).await;
    // Error because schema is different
    assert!(matches!(result, Err(Error::SchemaMismatch { .. })))
}

#[rstest]
#[tokio::test]
async fn append_dictionary(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    // We store the dictionary as part of the schema, so we check that the
    // dictionary is consistent between appends.

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "x",
        DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
        false,
    )]));
    let dictionary = Arc::new(StringArray::from(vec!["a", "b"]));
    let indices = Int8Array::from(vec![0, 1, 0]);
    let batches = vec![RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(
            Int8DictionaryArray::try_new(indices, dictionary.clone()).unwrap(),
        )],
    )
    .unwrap()];

    let test_uri = TempStrDir::default();
    let mut write_params = WriteParams {
        max_rows_per_file: 40,
        max_rows_per_group: 10,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    Dataset::write(batches, &test_uri, Some(write_params.clone()))
        .await
        .unwrap();

    // create a new one with same dictionary
    let indices = Int8Array::from(vec![1, 0, 1]);
    let batches = vec![RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(
            Int8DictionaryArray::try_new(indices, dictionary).unwrap(),
        )],
    )
    .unwrap()];

    // Write to dataset (successful)
    write_params.mode = WriteMode::Append;
    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    Dataset::write(batches, &test_uri, Some(write_params.clone()))
        .await
        .unwrap();

    // Create a new one with *different* dictionary
    let dictionary = Arc::new(StringArray::from(vec!["d", "c"]));
    let indices = Int8Array::from(vec![1, 0, 1]);
    let batches = vec![RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(
            Int8DictionaryArray::try_new(indices, dictionary).unwrap(),
        )],
    )
    .unwrap()];

    // Try write to dataset (fails with legacy format)
    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let result = Dataset::write(batches, &test_uri, Some(write_params)).await;
    if data_storage_version == LanceFileVersion::Legacy {
        assert!(result.is_err());
    } else {
        assert!(result.is_ok());
    }
}

#[rstest]
#[tokio::test]
async fn overwrite_dataset(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_uri = TempStrDir::default();

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::Int32,
        false,
    )]));
    let batches = vec![RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..20))],
    )
    .unwrap()];

    let mut write_params = WriteParams {
        max_rows_per_file: 40,
        max_rows_per_group: 10,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let dataset = Dataset::write(batches, &test_uri, Some(write_params.clone()))
        .await
        .unwrap();

    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 1);
    assert_eq!(dataset.manifest.max_fragment_id(), Some(0));

    let new_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        DataType::Utf8,
        false,
    )]));
    let new_batches = vec![RecordBatch::try_new(
        new_schema.clone(),
        vec![Arc::new(StringArray::from_iter_values(
            (20..40).map(|v| v.to_string()),
        ))],
    )
    .unwrap()];
    write_params.mode = Overwrite;
    let new_batch_reader =
        RecordBatchIterator::new(new_batches.into_iter().map(Ok), new_schema.clone());
    let dataset = Dataset::write(new_batch_reader, &test_uri, Some(write_params.clone()))
        .await
        .unwrap();

    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 1);
    // Fragment ids reset after overwrite.
    assert_eq!(fragments[0].id(), 0);
    assert_eq!(dataset.manifest.max_fragment_id(), Some(0));

    let actual_ds = Dataset::open(&test_uri).await.unwrap();
    assert_eq!(actual_ds.version().version, 2);
    let actual_schema = ArrowSchema::from(actual_ds.schema());
    assert_eq!(&actual_schema, new_schema.as_ref());

    let actual_batches = actual_ds
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let actual_batch = concat_batches(&new_schema, &actual_batches).unwrap();

    assert_eq!(new_schema.clone(), actual_batch.schema());
    let arr = actual_batch.column_by_name("s").unwrap();
    assert_eq!(
        &StringArray::from_iter_values((20..40).map(|v| v.to_string())),
        as_string_array(arr)
    );
    assert_eq!(actual_ds.version().version, 2);

    // But we can still check out the first version
    let first_ver = DatasetBuilder::from_uri(&test_uri)
        .with_version(1)
        .load()
        .await
        .unwrap();
    assert_eq!(first_ver.version().version, 1);
    assert_eq!(&ArrowSchema::from(first_ver.schema()), schema.as_ref());
}

#[rstest]
#[tokio::test]
async fn test_fast_count_rows(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_uri = TempStrDir::default();

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::Int32,
        false,
    )]));

    let batches: Vec<RecordBatch> = (0..20)
        .map(|i| {
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20))],
            )
            .unwrap()
        })
        .collect();

    let write_params = WriteParams {
        max_rows_per_file: 40,
        max_rows_per_group: 10,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    Dataset::write(batches, &test_uri, Some(write_params))
        .await
        .unwrap();

    let dataset = Dataset::open(&test_uri).await.unwrap();
    dataset.validate().await.unwrap();
    assert_eq!(10, dataset.fragments().len());
    assert_eq!(400, dataset.count_rows(None).await.unwrap());
    assert_eq!(
        200,
        dataset
            .count_rows(Some("i < 200".to_string()))
            .await
            .unwrap()
    );
}

#[rstest]
#[tokio::test]
async fn test_bfloat16_roundtrip(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) -> Result<()> {
    let inner_field = Arc::new(
        ArrowField::new("item", DataType::FixedSizeBinary(2), true).with_metadata(
            [
                (ARROW_EXT_NAME_KEY.into(), BFLOAT16_EXT_NAME.into()),
                (ARROW_EXT_META_KEY.into(), "".into()),
            ]
            .into(),
        ),
    );
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "fsl",
        DataType::FixedSizeList(inner_field.clone(), 2),
        false,
    )]));

    let values = bfloat16::BFloat16Array::from_iter_values(
        (0..6).map(|i| i as f32).map(half::bf16::from_f32),
    );
    let vectors = FixedSizeListArray::new(inner_field, 2, Arc::new(values.into_inner()), None);

    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();

    let test_uri = TempStrDir::default();

    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone()),
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await?;

    let data = dataset.scan().try_into_batch().await?;
    assert_eq!(batch, data);

    Ok(())
}

#[tokio::test]
async fn test_overwrite_mixed_version() {
    let test_uri = TempStrDir::default();

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "a",
        DataType::Int32,
        false,
    )]));
    let arr = Arc::new(Int32Array::from(vec![1, 2, 3]));

    let data = RecordBatch::try_new(schema.clone(), vec![arr]).unwrap();
    let reader = RecordBatchIterator::new(vec![data.clone()].into_iter().map(Ok), schema.clone());

    let dataset = Dataset::write(
        reader,
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::Legacy),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(
        dataset
            .manifest
            .data_storage_format
            .lance_file_version()
            .unwrap(),
        LanceFileVersion::Legacy
    );

    let reader = RecordBatchIterator::new(vec![data].into_iter().map(Ok), schema);
    let dataset = Dataset::write(
        reader,
        &test_uri,
        Some(WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(
        dataset
            .manifest
            .data_storage_format
            .lance_file_version()
            .unwrap(),
        LanceFileVersion::Legacy
    );
}

#[tokio::test]
async fn test_open_nonexisting_dataset() {
    let temp_dir = TempStdDir::default();
    let dataset_dir = temp_dir.join("non_existing");
    let dataset_uri = dataset_dir.to_str().unwrap();

    let res = Dataset::open(dataset_uri).await;
    assert!(res.is_err());

    assert!(!dataset_dir.exists());
}

#[tokio::test]
async fn test_manifest_partially_fits() {
    // This regresses a bug that occurred when the manifest file was over 4KiB but the manifest
    // itself was less than 4KiB (due to a dictionary).  4KiB is important here because that's the
    // block size we use when reading the "last block"

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "x",
        DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8)),
        false,
    )]));
    let dictionary = Arc::new(StringArray::from_iter_values(
        (0..1000).map(|i| i.to_string()),
    ));
    let indices = Int16Array::from_iter_values(0..1000);
    let batches = vec![RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(
            Int16DictionaryArray::try_new(indices, dictionary.clone()).unwrap(),
        )],
    )
    .unwrap()];

    let test_uri = TempStrDir::default();
    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    Dataset::write(batches, &test_uri, None).await.unwrap();

    let dataset = Dataset::open(&test_uri).await.unwrap();
    assert_eq!(1000, dataset.count_rows(None).await.unwrap());
}

#[tokio::test]
async fn test_dataset_uri_roundtrips() {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "a",
        DataType::Int32,
        false,
    )]));

    let test_uri = TempStrDir::default();
    let vectors = Arc::new(Int32Array::from_iter_values(vec![]));

    let data = RecordBatch::try_new(schema.clone(), vec![vectors]);
    let reader = RecordBatchIterator::new(vec![data.unwrap()].into_iter().map(Ok), schema);
    let dataset = Dataset::write(
        reader,
        &test_uri,
        Some(WriteParams {
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let uri = dataset.uri();
    assert_eq!(uri, test_uri.as_str());

    let ds2 = Dataset::open(uri).await.unwrap();
    assert_eq!(
        ds2.latest_version_id().await.unwrap(),
        dataset.latest_version_id().await.unwrap()
    );
}
