// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::vec;

use crate::dataset::optimize::{compact_files, CompactionOptions};
use crate::dataset::transaction::{DataReplacementGroup, Operation};
use crate::dataset::WriteDestination;
use crate::dataset::ROW_ID;
use crate::dataset::{AutoCleanupParams, ProjectionRequest};
use crate::{Dataset, Error};
use lance_core::ROW_ADDR;
use mock_instant::thread_local::MockClock;

use crate::dataset::write::{InsertBuilder, WriteMode, WriteParams};
use arrow::array::AsArray;
use arrow::compute::concat_batches;
use arrow_array::RecordBatch;
use arrow_array::{
    types::Int32Type, ArrayRef, Float32Array, Int32Array, ListArray, RecordBatchIterator,
    StringArray,
};
use arrow_array::{Array, LargeBinaryArray, StructArray};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_arrow::BLOB_META_KEY;
use lance_core::utils::tempfile::{TempDir, TempStrDir};
use lance_datagen::{array, gen_batch, BatchCount, RowCount};
use lance_file::version::LanceFileVersion;
use lance_file::writer::FileWriter;
use lance_io::utils::CachedFileSize;
use lance_table::format::DataFile;

use futures::TryStreamExt;
use lance_datafusion::datagen::DatafusionDatagenExt;
use object_store::path::Path;
use rand::seq::SliceRandom;
use rstest::rstest;

#[rstest]
#[tokio::test]
async fn test_merge(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
    #[values(false, true)] use_stable_row_id: bool,
) {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("i", DataType::Int32, false),
        ArrowField::new("x", DataType::Float32, false),
    ]));
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Float32Array::from(vec![1.0, 2.0])),
        ],
    )
    .unwrap();
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![3, 2])),
            Arc::new(Float32Array::from(vec![3.0, 4.0])),
        ],
    )
    .unwrap();

    let test_uri = TempStrDir::default();

    let write_params = WriteParams {
        mode: WriteMode::Append,
        data_storage_version: Some(data_storage_version),
        enable_stable_row_ids: use_stable_row_id,
        ..Default::default()
    };

    let batches = RecordBatchIterator::new(vec![batch1].into_iter().map(Ok), schema.clone());
    Dataset::write(batches, &test_uri, Some(write_params.clone()))
        .await
        .unwrap();

    let batches = RecordBatchIterator::new(vec![batch2].into_iter().map(Ok), schema.clone());
    Dataset::write(batches, &test_uri, Some(write_params.clone()))
        .await
        .unwrap();

    let dataset = Dataset::open(&test_uri).await.unwrap();
    assert_eq!(dataset.fragments().len(), 2);
    assert_eq!(dataset.manifest.max_fragment_id(), Some(1));

    let right_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("i2", DataType::Int32, false),
        ArrowField::new("y", DataType::Utf8, true),
    ]));
    let right_batch1 = RecordBatch::try_new(
        right_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["a", "b"])),
        ],
    )
    .unwrap();

    let batches =
        RecordBatchIterator::new(vec![right_batch1].into_iter().map(Ok), right_schema.clone());
    let mut dataset = Dataset::open(&test_uri).await.unwrap();
    dataset.merge(batches, "i", "i2").await.unwrap();
    dataset.validate().await.unwrap();

    assert_eq!(dataset.version().version, 3);
    assert_eq!(dataset.fragments().len(), 2);
    assert_eq!(dataset.fragments()[0].files.len(), 2);
    assert_eq!(dataset.fragments()[1].files.len(), 2);
    assert_eq!(dataset.manifest.max_fragment_id(), Some(1));

    let actual_batches = dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let actual = concat_batches(&actual_batches[0].schema(), &actual_batches).unwrap();
    let expected = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, false),
            ArrowField::new("x", DataType::Float32, false),
            ArrowField::new("y", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 2])),
            Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0, 4.0])),
            Arc::new(StringArray::from(vec![
                Some("a"),
                Some("b"),
                None,
                Some("b"),
            ])),
        ],
    )
    .unwrap();

    assert_eq!(actual, expected);

    // Validate we can still read after re-instantiating dataset, which
    // clears the cache.
    let dataset = Dataset::open(&test_uri).await.unwrap();
    let actual_batches = dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let actual = concat_batches(&actual_batches[0].schema(), &actual_batches).unwrap();
    assert_eq!(actual, expected);
}

#[rstest]
#[tokio::test]
async fn test_large_merge(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
    #[values(false, true)] use_stable_row_id: bool,
) {
    // Tests a merge that spans multiple batches within files

    // This test also tests "null filling" when merging (e.g. when keys do not match
    // we need to insert nulls)

    let data = lance_datagen::gen_batch()
        .col("key", array::step::<Int32Type>())
        .col("value", array::fill_utf8("value".to_string()))
        .into_reader_rows(RowCount::from(1_000), BatchCount::from(10));

    let test_uri = TempStrDir::default();

    let write_params = WriteParams {
        mode: WriteMode::Append,
        data_storage_version: Some(data_storage_version),
        max_rows_per_file: 1024,
        max_rows_per_group: 150,
        enable_stable_row_ids: use_stable_row_id,
        ..Default::default()
    };
    Dataset::write(data, &test_uri, Some(write_params.clone()))
        .await
        .unwrap();

    let mut dataset = Dataset::open(&test_uri).await.unwrap();
    assert_eq!(dataset.fragments().len(), 10);
    assert_eq!(dataset.manifest.max_fragment_id(), Some(9));

    let new_data = lance_datagen::gen_batch()
        .col("key2", array::step_custom::<Int32Type>(500, 1))
        .col("new_value", array::fill_utf8("new_value".to_string()))
        .into_reader_rows(RowCount::from(1_000), BatchCount::from(10));

    dataset.merge(new_data, "key", "key2").await.unwrap();
    dataset.validate().await.unwrap();
}

#[rstest]
#[tokio::test]
async fn test_merge_on_row_id(
    #[values(LanceFileVersion::Stable)] data_storage_version: LanceFileVersion,
    #[values(false, true)] use_stable_row_id: bool,
) {
    // Tests a merge on _rowid

    let data = lance_datagen::gen_batch()
        .col("key", array::step::<Int32Type>())
        .col("value", array::fill_utf8("value".to_string()))
        .into_reader_rows(RowCount::from(1_000), BatchCount::from(10));

    let write_params = WriteParams {
        mode: WriteMode::Append,
        data_storage_version: Some(data_storage_version),
        max_rows_per_file: 1024,
        max_rows_per_group: 150,
        enable_stable_row_ids: use_stable_row_id,
        ..Default::default()
    };
    let mut dataset = Dataset::write(data, "memory://", Some(write_params.clone()))
        .await
        .unwrap();
    assert_eq!(dataset.fragments().len(), 10);
    assert_eq!(dataset.manifest.max_fragment_id(), Some(9));

    let data = dataset.scan().with_row_id().try_into_batch().await.unwrap();
    let row_ids: Arc<dyn Array> = data[ROW_ID].clone();
    let key = data["key"].as_primitive::<Int32Type>();
    let new_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("rowid", DataType::UInt64, false),
        ArrowField::new("new_value", DataType::Int32, false),
    ]));
    let new_value = Arc::new(
        key.into_iter()
            .map(|v| v.unwrap() + 1)
            .collect::<arrow_array::Int32Array>(),
    );
    let len = new_value.len() as u32;
    let new_batch = RecordBatch::try_new(new_schema.clone(), vec![row_ids, new_value]).unwrap();
    // shuffle new_batch
    let mut rng = rand::rng();
    let mut indices: Vec<u32> = (0..len).collect();
    indices.shuffle(&mut rng);
    let indices = arrow_array::UInt32Array::from_iter_values(indices);
    let new_batch = arrow::compute::take_record_batch(&new_batch, &indices).unwrap();
    let new_data = RecordBatchIterator::new(vec![Ok(new_batch)], new_schema.clone());
    dataset.merge(new_data, ROW_ID, "rowid").await.unwrap();
    dataset.validate().await.unwrap();
    assert_eq!(dataset.schema().fields.len(), 3);
    assert!(dataset.schema().field("key").is_some());
    assert!(dataset.schema().field("value").is_some());
    assert!(dataset.schema().field("new_value").is_some());
    let batch = dataset.scan().try_into_batch().await.unwrap();
    let key = batch["key"].as_primitive::<Int32Type>();
    let new_value = batch["new_value"].as_primitive::<Int32Type>();
    for i in 0..key.len() {
        assert_eq!(key.value(i) + 1, new_value.value(i));
    }
}

#[rstest]
#[tokio::test]
async fn test_merge_on_row_addr(
    #[values(LanceFileVersion::Stable)] data_storage_version: LanceFileVersion,
    #[values(false, true)] use_stable_row_id: bool,
) {
    // Tests a merge on _rowaddr

    let data = lance_datagen::gen_batch()
        .col("key", array::step::<Int32Type>())
        .col("value", array::fill_utf8("value".to_string()))
        .into_reader_rows(RowCount::from(1_000), BatchCount::from(10));

    let write_params = WriteParams {
        mode: WriteMode::Append,
        data_storage_version: Some(data_storage_version),
        max_rows_per_file: 1024,
        max_rows_per_group: 150,
        enable_stable_row_ids: use_stable_row_id,
        ..Default::default()
    };
    let mut dataset = Dataset::write(data, "memory://", Some(write_params.clone()))
        .await
        .unwrap();

    assert_eq!(dataset.fragments().len(), 10);
    assert_eq!(dataset.manifest.max_fragment_id(), Some(9));

    let data = dataset
        .scan()
        .with_row_address()
        .try_into_batch()
        .await
        .unwrap();
    let row_addrs = data[ROW_ADDR].clone();
    let key = data["key"].as_primitive::<Int32Type>();
    let new_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("rowaddr", DataType::UInt64, false),
        ArrowField::new("new_value", DataType::Int32, false),
    ]));
    let new_value = Arc::new(
        key.into_iter()
            .map(|v| v.unwrap() + 1)
            .collect::<arrow_array::Int32Array>(),
    );
    let len = new_value.len() as u32;
    let new_batch = RecordBatch::try_new(new_schema.clone(), vec![row_addrs, new_value]).unwrap();
    // shuffle new_batch
    let mut rng = rand::rng();
    let mut indices: Vec<u32> = (0..len).collect();
    indices.shuffle(&mut rng);
    let indices = arrow_array::UInt32Array::from_iter_values(indices);
    let new_batch = arrow::compute::take_record_batch(&new_batch, &indices).unwrap();
    let new_data = RecordBatchIterator::new(vec![Ok(new_batch)], new_schema.clone());
    dataset.merge(new_data, ROW_ADDR, "rowaddr").await.unwrap();
    dataset.validate().await.unwrap();
    assert_eq!(dataset.schema().fields.len(), 3);
    assert!(dataset.schema().field("key").is_some());
    assert!(dataset.schema().field("value").is_some());
    assert!(dataset.schema().field("new_value").is_some());
    let batch = dataset.scan().try_into_batch().await.unwrap();
    let key = batch["key"].as_primitive::<Int32Type>();
    let new_value = batch["new_value"].as_primitive::<Int32Type>();
    for i in 0..key.len() {
        assert_eq!(key.value(i) + 1, new_value.value(i));
    }
}

#[tokio::test]
async fn test_insert_subschema() {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("a", DataType::Int32, false),
        ArrowField::new("b", DataType::Int32, true),
    ]));
    let empty_reader = RecordBatchIterator::new(vec![], schema.clone());
    let mut dataset = Dataset::write(empty_reader, "memory://", None)
        .await
        .unwrap();
    dataset.validate().await.unwrap();

    // If missing columns that aren't nullable, will return an error
    // TODO: provide alternative default than null.
    let just_b = Arc::new(schema.project(&[1]).unwrap());
    let batch =
        RecordBatch::try_new(just_b.clone(), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], just_b.clone());
    let res = dataset.append(reader, None).await;
    assert!(
        matches!(res, Err(Error::SchemaMismatch { .. })),
        "Expected Error::SchemaMismatch, got {:?}",
        res
    );

    // If missing columns that are nullable, the write succeeds.
    let just_a = Arc::new(schema.project(&[0]).unwrap());
    let batch =
        RecordBatch::try_new(just_a.clone(), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], just_a.clone());
    dataset.append(reader, None).await.unwrap();
    dataset.validate().await.unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 1);

    // Looking at the fragments, there is no data file with the missing field
    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 1);
    assert_eq!(fragments[0].metadata.files.len(), 1);
    assert_eq!(&fragments[0].metadata.files[0].fields, &[0]);

    // When reading back, columns that are missing are null
    let data = dataset.scan().try_into_batch().await.unwrap();
    let expected = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![None])),
        ],
    )
    .unwrap();
    assert_eq!(data, expected);

    // Can still insert all columns
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![2])),
            Arc::new(Int32Array::from(vec![3])),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone());
    dataset.append(reader, None).await.unwrap();
    dataset.validate().await.unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 2);

    // When reading back, only missing data is null, otherwise is filled in
    let data = dataset.scan().try_into_batch().await.unwrap();
    let expected = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Int32Array::from(vec![None, Some(3)])),
        ],
    )
    .unwrap();
    assert_eq!(data, expected);

    // Can run compaction. All files should now have all fields.
    compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    dataset.validate().await.unwrap();
    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 1);
    assert_eq!(fragments[0].metadata.files.len(), 1);
    assert_eq!(&fragments[0].metadata.files[0].fields, &[0, 1]);

    // Can scan and get expected data.
    let data = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(data, expected);
}

#[tokio::test]
async fn test_insert_nested_subschemas() {
    // Test subschemas at struct level
    // Test different orders
    // Test the Dataset::write() path
    // Test Take across fragments with different field id sets
    let test_uri = TempStrDir::default();

    let field_a = Arc::new(ArrowField::new("a", DataType::Int32, true));
    let field_b = Arc::new(ArrowField::new("b", DataType::Int32, false));
    let field_c = Arc::new(ArrowField::new("c", DataType::Int32, true));
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        DataType::Struct(vec![field_a.clone(), field_b.clone(), field_c.clone()].into()),
        true,
    )]));
    let empty_reader = RecordBatchIterator::new(vec![], schema.clone());
    let dataset = Dataset::write(empty_reader, &test_uri, None).await.unwrap();
    dataset.validate().await.unwrap();

    let append_options = WriteParams {
        mode: WriteMode::Append,
        ..Default::default()
    };
    // Can insert b, a
    let just_b_a = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        DataType::Struct(vec![field_b.clone(), field_a.clone()].into()),
        true,
    )]));
    let batch = RecordBatch::try_new(
        just_b_a.clone(),
        vec![Arc::new(StructArray::from(vec![
            (
                field_b.clone(),
                Arc::new(Int32Array::from(vec![1])) as ArrayRef,
            ),
            (field_a.clone(), Arc::new(Int32Array::from(vec![2]))),
        ]))],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], just_b_a.clone());
    let dataset = Dataset::write(reader, &test_uri, Some(append_options.clone()))
        .await
        .unwrap();
    dataset.validate().await.unwrap();
    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 1);
    assert_eq!(fragments[0].metadata.files.len(), 1);
    assert_eq!(&fragments[0].metadata.files[0].fields, &[0, 2, 1]);
    assert_eq!(&fragments[0].metadata.files[0].column_indices, &[0, 1, 2]);

    // Can insert c, b
    let just_c_b = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        DataType::Struct(vec![field_c.clone(), field_b.clone()].into()),
        true,
    )]));
    let batch = RecordBatch::try_new(
        just_c_b.clone(),
        vec![Arc::new(StructArray::from(vec![
            (
                field_c.clone(),
                Arc::new(Int32Array::from(vec![4])) as ArrayRef,
            ),
            (field_b.clone(), Arc::new(Int32Array::from(vec![3]))),
        ]))],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], just_c_b.clone());
    let dataset = Dataset::write(reader, &test_uri, Some(append_options.clone()))
        .await
        .unwrap();
    dataset.validate().await.unwrap();
    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 2);
    assert_eq!(fragments[1].metadata.files.len(), 1);
    assert_eq!(&fragments[1].metadata.files[0].fields, &[0, 3, 2]);
    assert_eq!(&fragments[1].metadata.files[0].column_indices, &[0, 1, 2]);

    // Can't insert a, c (b is non-nullable)
    let just_a_c = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        DataType::Struct(vec![field_a.clone(), field_c.clone()].into()),
        true,
    )]));
    let batch = RecordBatch::try_new(
        just_a_c.clone(),
        vec![Arc::new(StructArray::from(vec![
            (
                field_a.clone(),
                Arc::new(Int32Array::from(vec![5])) as ArrayRef,
            ),
            (field_c.clone(), Arc::new(Int32Array::from(vec![6]))),
        ]))],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], just_a_c.clone());
    let res = Dataset::write(reader, &test_uri, Some(append_options)).await;
    assert!(
        matches!(res, Err(Error::SchemaMismatch { .. })),
        "Expected Error::SchemaMismatch, got {:?}",
        res
    );

    // Can scan and get all data
    let data = dataset.scan().try_into_batch().await.unwrap();
    let expected = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(StructArray::from(vec![
            (
                field_a.clone(),
                Arc::new(Int32Array::from(vec![Some(2), None])) as ArrayRef,
            ),
            (field_b.clone(), Arc::new(Int32Array::from(vec![1, 3]))),
            (
                field_c.clone(),
                Arc::new(Int32Array::from(vec![None, Some(4)])),
            ),
        ]))],
    )
    .unwrap();
    assert_eq!(data, expected);

    // Can call take and get rows from all three back in one batch
    let result = dataset
        .take(&[1, 0], Arc::new(dataset.schema().clone()))
        .await
        .unwrap();
    let expected = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(StructArray::from(vec![
            (
                field_a.clone(),
                Arc::new(Int32Array::from(vec![None, Some(2)])) as ArrayRef,
            ),
            (field_b.clone(), Arc::new(Int32Array::from(vec![3, 1]))),
            (
                field_c.clone(),
                Arc::new(Int32Array::from(vec![Some(4), None])),
            ),
        ]))],
    )
    .unwrap();
    assert_eq!(result, expected);
}

#[tokio::test]
async fn test_insert_balanced_subschemas() {
    let test_uri = TempStrDir::default();

    let field_a = ArrowField::new("a", DataType::Int32, true);
    let field_b = ArrowField::new("b", DataType::LargeBinary, true);
    let schema = Arc::new(ArrowSchema::new(vec![
        field_a.clone(),
        field_b
            .clone()
            .with_metadata([(BLOB_META_KEY.to_string(), "true".to_string())].into()),
    ]));
    let empty_reader = RecordBatchIterator::new(vec![], schema.clone());
    let options = WriteParams {
        enable_stable_row_ids: true,
        enable_v2_manifest_paths: true,
        ..Default::default()
    };
    let mut dataset = Dataset::write(empty_reader, &test_uri, Some(options))
        .await
        .unwrap();
    dataset.validate().await.unwrap();

    // Insert left side
    let just_a = Arc::new(ArrowSchema::new(vec![field_a.clone()]));
    let batch =
        RecordBatch::try_new(just_a.clone(), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], just_a.clone());
    dataset.append(reader, None).await.unwrap();
    dataset.validate().await.unwrap();

    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 1);
    assert_eq!(fragments[0].metadata.files.len(), 1);
    assert_eq!(&fragments[0].metadata.files[0].fields, &[0]);

    // Insert right side
    let just_b = Arc::new(ArrowSchema::new(vec![field_b.clone()]));
    let batch = RecordBatch::try_new(
        just_b.clone(),
        vec![Arc::new(LargeBinaryArray::from_iter(vec![Some(vec![2u8])]))],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], just_b.clone());
    dataset.append(reader, None).await.unwrap();
    dataset.validate().await.unwrap();

    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 2);
    assert_eq!(fragments[1].metadata.files.len(), 1);
    assert_eq!(&fragments[1].metadata.files[0].fields, &[1]);

    let data = dataset
        .take(
            &[0, 1],
            ProjectionRequest::from_columns(["a"], dataset.schema()),
        )
        .await
        .unwrap();
    assert_eq!(data.num_rows(), 2);
    let a_column = data.column(0).as_primitive::<Int32Type>();
    assert_eq!(a_column.value(0), 1);
    assert!(a_column.is_null(1));

    let blob_batch = dataset
        .take(
            &[0, 1],
            ProjectionRequest::from_columns(["b"], dataset.schema()),
        )
        .await
        .unwrap();
    let blob_descriptions = blob_batch.column(0).as_struct();
    assert!(blob_descriptions.is_null(0));
    assert!(blob_descriptions.is_valid(1));
}

#[tokio::test]
async fn test_datafile_replacement() {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "a",
        DataType::Int32,
        true,
    )]));
    let empty_reader = RecordBatchIterator::new(vec![], schema.clone());
    let dataset = Arc::new(
        Dataset::write(empty_reader, "memory://", None)
            .await
            .unwrap(),
    );
    dataset.validate().await.unwrap();

    // Test empty replacement should commit a new manifest and do nothing
    let mut dataset = Dataset::commit(
        WriteDestination::Dataset(dataset.clone()),
        Operation::DataReplacement {
            replacements: vec![],
        },
        Some(1),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap();
    dataset.validate().await.unwrap();

    assert_eq!(dataset.version().version, 2);
    assert_eq!(dataset.get_fragments().len(), 0);

    // try the same thing on a non-empty dataset
    let vals: Int32Array = vec![1, 2, 3].into();
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vals)]).unwrap();
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
            None,
        )
        .await
        .unwrap();

    let dataset = Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset)),
        Operation::DataReplacement {
            replacements: vec![],
        },
        Some(3),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap();
    dataset.validate().await.unwrap();

    assert_eq!(dataset.version().version, 4);
    assert_eq!(dataset.get_fragments().len(), 1);

    let batch = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values(),
        &[1, 2, 3]
    );

    // write a new datafile
    let object_writer = dataset
        .object_store
        .create(&Path::from("data/test.lance"))
        .await
        .unwrap();
    let mut writer = FileWriter::try_new(
        object_writer,
        schema.as_ref().try_into().unwrap(),
        Default::default(),
    )
    .unwrap();

    let vals: Int32Array = vec![4, 5, 6].into();
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vals)]).unwrap();
    writer.write_batch(&batch).await.unwrap();
    writer.finish().await.unwrap();

    // find the datafile we want to replace
    let frag = dataset.get_fragment(0).unwrap();
    let data_file = frag.data_file_for_field(0).unwrap();
    let mut new_data_file = data_file.clone();
    new_data_file.path = "test.lance".to_string();

    let dataset = Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset)),
        Operation::DataReplacement {
            replacements: vec![DataReplacementGroup(0, new_data_file)],
        },
        Some(4),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap();

    assert_eq!(dataset.version().version, 5);
    assert_eq!(dataset.get_fragments().len(), 1);
    assert_eq!(dataset.get_fragments()[0].metadata.files.len(), 1);

    let batch = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values(),
        &[4, 5, 6]
    );
}

#[tokio::test]
async fn test_datafile_partial_replacement() {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "a",
        DataType::Int32,
        true,
    )]));
    let empty_reader = RecordBatchIterator::new(vec![], schema.clone());
    let mut dataset = Dataset::write(empty_reader, "memory://", None)
        .await
        .unwrap();
    dataset.validate().await.unwrap();

    let vals: Int32Array = vec![1, 2, 3].into();
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vals)]).unwrap();
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
            None,
        )
        .await
        .unwrap();

    let fragment = dataset.get_fragments().pop().unwrap().metadata;

    let extended_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("a", DataType::Int32, true),
        ArrowField::new("b", DataType::Int32, true),
    ]));

    // add all null column
    let dataset = Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset)),
        Operation::Merge {
            fragments: vec![fragment],
            schema: extended_schema.as_ref().try_into().unwrap(),
        },
        Some(2),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap();

    let partial_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "b",
        DataType::Int32,
        true,
    )]));

    // write a new datafile
    let object_writer = dataset
        .object_store
        .create(&Path::from("data/test.lance"))
        .await
        .unwrap();
    let mut writer = FileWriter::try_new(
        object_writer,
        partial_schema.as_ref().try_into().unwrap(),
        Default::default(),
    )
    .unwrap();

    let vals: Int32Array = vec![4, 5, 6].into();
    let batch = RecordBatch::try_new(partial_schema.clone(), vec![Arc::new(vals)]).unwrap();
    writer.write_batch(&batch).await.unwrap();
    writer.finish().await.unwrap();

    let (major, minor) = lance_file::version::LanceFileVersion::Stable.to_numbers();

    // find the datafile we want to replace
    let new_data_file = DataFile {
        path: "test.lance".to_string(),
        // the second column in the dataset
        fields: vec![1],
        // is located in the first column of this datafile
        column_indices: vec![0],
        file_major_version: major,
        file_minor_version: minor,
        file_size_bytes: CachedFileSize::unknown(),
        base_id: None,
    };

    let dataset = Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset)),
        Operation::DataReplacement {
            replacements: vec![DataReplacementGroup(0, new_data_file)],
        },
        Some(3),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap();

    assert_eq!(dataset.version().version, 4);
    assert_eq!(dataset.get_fragments().len(), 1);
    assert_eq!(dataset.get_fragments()[0].metadata.files.len(), 2);
    assert_eq!(dataset.get_fragments()[0].metadata.files[0].fields, vec![0]);
    assert_eq!(dataset.get_fragments()[0].metadata.files[1].fields, vec![1]);

    let batch = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values(),
        &[1, 2, 3]
    );
    assert_eq!(
        batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values(),
        &[4, 5, 6]
    );

    // do it again but on the first column
    // find the datafile we want to replace
    let new_data_file = DataFile {
        path: "test.lance".to_string(),
        // the first column in the dataset
        fields: vec![0],
        // is located in the first column of this datafile
        column_indices: vec![0],
        file_major_version: major,
        file_minor_version: minor,
        file_size_bytes: CachedFileSize::unknown(),
        base_id: None,
    };

    let dataset = Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset)),
        Operation::DataReplacement {
            replacements: vec![DataReplacementGroup(0, new_data_file)],
        },
        Some(4),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap();

    assert_eq!(dataset.version().version, 5);
    assert_eq!(dataset.get_fragments().len(), 1);
    assert_eq!(dataset.get_fragments()[0].metadata.files.len(), 2);

    let batch = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values(),
        &[4, 5, 6]
    );
    assert_eq!(
        batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values(),
        &[4, 5, 6]
    );
}

#[tokio::test]
async fn test_datafile_replacement_error() {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "a",
        DataType::Int32,
        true,
    )]));
    let empty_reader = RecordBatchIterator::new(vec![], schema.clone());
    let mut dataset = Dataset::write(empty_reader, "memory://", None)
        .await
        .unwrap();
    dataset.validate().await.unwrap();

    let vals: Int32Array = vec![1, 2, 3].into();
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vals)]).unwrap();
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
            None,
        )
        .await
        .unwrap();

    let fragment = dataset.get_fragments().pop().unwrap().metadata;

    let extended_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("a", DataType::Int32, true),
        ArrowField::new("b", DataType::Int32, true),
    ]));

    // add all null column
    let dataset = Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset)),
        Operation::Merge {
            fragments: vec![fragment],
            schema: extended_schema.as_ref().try_into().unwrap(),
        },
        Some(2),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap();

    // find the datafile we want to replace
    let new_data_file = DataFile {
        path: "test.lance".to_string(),
        // the second column in the dataset
        fields: vec![1],
        // is located in the first column of this datafile
        column_indices: vec![0],
        file_major_version: 2,
        file_minor_version: 0,
        file_size_bytes: CachedFileSize::unknown(),
        base_id: None,
    };

    let new_data_file = DataFile {
        fields: vec![0, 1],
        ..new_data_file
    };

    let err = Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset.clone())),
        Operation::DataReplacement {
            replacements: vec![DataReplacementGroup(0, new_data_file)],
        },
        Some(2),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("Expected to modify the fragment but no changes were made"),
        "Expected Error::DataFileReplacementError, got {:?}",
        err
    );
}

#[tokio::test]
async fn test_replace_dataset() {
    let test_dir = TempDir::default();
    let test_uri = test_dir.path_str();
    let test_path = test_dir.obj_path();

    let data = gen_batch()
        .col("int", array::step::<Int32Type>())
        .into_batch_rows(RowCount::from(20))
        .unwrap();
    let data1 = data.slice(0, 10);
    let data2 = data.slice(10, 10);
    let mut ds = InsertBuilder::new(&test_uri)
        .execute(vec![data1])
        .await
        .unwrap();

    ds.object_store().remove_dir_all(test_path).await.unwrap();

    let ds2 = InsertBuilder::new(&test_uri)
        .execute(vec![data2.clone()])
        .await
        .unwrap();

    ds.checkout_latest().await.unwrap();
    let roundtripped = ds.scan().try_into_batch().await.unwrap();
    assert_eq!(roundtripped, data2);

    ds.validate().await.unwrap();
    ds2.validate().await.unwrap();
    assert_eq!(ds.manifest.version, 1);
    assert_eq!(ds2.manifest.version, 1);
}

#[tokio::test]
async fn test_insert_skip_auto_cleanup() {
    let test_uri = TempStrDir::default();

    // Create initial dataset with aggressive auto cleanup (interval=1, older_than=1ms)
    let data = gen_batch()
        .col("id", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(100), BatchCount::from(1));

    let write_params = WriteParams {
        mode: WriteMode::Create,
        auto_cleanup: Some(AutoCleanupParams {
            interval: 1,
            older_than: chrono::TimeDelta::try_milliseconds(0).unwrap(), // Cleanup versions older than 0ms
        }),
        ..Default::default()
    };

    // Start at 1 second after epoch
    MockClock::set_system_time(std::time::Duration::from_secs(1));

    let dataset = Dataset::write(data, &test_uri, Some(write_params))
        .await
        .unwrap();
    assert_eq!(dataset.version().version, 1);

    // Advance time by 1 second
    MockClock::set_system_time(std::time::Duration::from_secs(2));

    // First append WITHOUT skip_auto_cleanup - should trigger cleanup
    let data1 = gen_batch()
        .col("id", array::step::<Int32Type>())
        .into_df_stream(RowCount::from(50), BatchCount::from(1));

    let write_params1 = WriteParams {
        mode: WriteMode::Append,
        skip_auto_cleanup: false,
        ..Default::default()
    };

    let dataset2 = InsertBuilder::new(WriteDestination::Dataset(Arc::new(dataset)))
        .with_params(&write_params1)
        .execute_stream(data1)
        .await
        .unwrap();

    assert_eq!(dataset2.version().version, 2);

    // Advance time
    MockClock::set_system_time(std::time::Duration::from_secs(3));

    // Need to do another commit for cleanup to take effect since cleanup runs on the old dataset
    let data1_extra = gen_batch()
        .col("id", array::step::<Int32Type>())
        .into_df_stream(RowCount::from(10), BatchCount::from(1));

    let dataset2_extra = InsertBuilder::new(WriteDestination::Dataset(Arc::new(dataset2)))
        .with_params(&write_params1)
        .execute_stream(data1_extra)
        .await
        .unwrap();

    assert_eq!(dataset2_extra.version().version, 3);

    // Version 1 should be cleaned up due to auto cleanup (cleanup runs every version)
    assert!(
        dataset2_extra.checkout_version(1).await.is_err(),
        "Version 1 should have been cleaned up"
    );
    // Version 2 should still exist
    assert!(
        dataset2_extra.checkout_version(2).await.is_ok(),
        "Version 2 should still exist"
    );

    // Advance time
    MockClock::set_system_time(std::time::Duration::from_secs(4));

    // Second append WITH skip_auto_cleanup - should NOT trigger cleanup
    let data2 = gen_batch()
        .col("id", array::step::<Int32Type>())
        .into_df_stream(RowCount::from(30), BatchCount::from(1));

    let write_params2 = WriteParams {
        mode: WriteMode::Append,
        skip_auto_cleanup: true, // Skip auto cleanup
        ..Default::default()
    };

    let dataset3 = InsertBuilder::new(WriteDestination::Dataset(Arc::new(dataset2_extra)))
        .with_params(&write_params2)
        .execute_stream(data2)
        .await
        .unwrap();

    assert_eq!(dataset3.version().version, 4);

    // Version 2 should still exist because skip_auto_cleanup was enabled
    assert!(
        dataset3.checkout_version(2).await.is_ok(),
        "Version 2 should still exist because skip_auto_cleanup was enabled"
    );
    // Version 3 should also still exist
    assert!(
        dataset3.checkout_version(3).await.is_ok(),
        "Version 3 should still exist"
    );
}

#[tokio::test]
async fn test_nullable_struct_v2_1_issue_4385() {
    // Test for issue #4385: nullable struct should preserve null values in v2.1 format
    use arrow_array::cast::AsArray;
    use arrow_schema::Fields;

    // Create a struct field with nullable float field
    let struct_fields = Fields::from(vec![ArrowField::new("x", DataType::Float32, true)]);

    // Create outer struct with the nullable struct as a field (not root)
    let outer_fields = Fields::from(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("data", DataType::Struct(struct_fields.clone()), true),
    ]);
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "record",
        DataType::Struct(outer_fields.clone()),
        false,
    )]));

    // Create data with null struct
    let id_values = Int32Array::from(vec![1, 2, 3]);
    let x_values = Float32Array::from(vec![Some(1.0), Some(2.0), Some(3.0)]);
    let inner_struct_array = StructArray::new(
        struct_fields,
        vec![Arc::new(x_values) as ArrayRef],
        Some(vec![true, false, true].into()), // Second struct is null
    );

    let outer_struct_array = StructArray::new(
        outer_fields,
        vec![
            Arc::new(id_values) as ArrayRef,
            Arc::new(inner_struct_array.clone()) as ArrayRef,
        ],
        None, // Outer struct is not nullable
    );

    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(outer_struct_array)]).unwrap();

    // Write dataset with v2.1 format
    let test_uri = TempStrDir::default();

    let write_params = WriteParams {
        mode: WriteMode::Create,
        data_storage_version: Some(LanceFileVersion::V2_1),
        ..Default::default()
    };

    let batches = vec![batch.clone()];
    let batch_reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

    Dataset::write(batch_reader, &test_uri, Some(write_params))
        .await
        .unwrap();

    // Read back the dataset
    let dataset = Dataset::open(&test_uri).await.unwrap();
    let scanner = dataset.scan();
    let result_batches = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    assert_eq!(result_batches.len(), 1);
    let result_batch = &result_batches[0];
    let read_outer_struct = result_batch.column(0).as_struct();
    let read_inner_struct = read_outer_struct.column(1).as_struct(); // "data" field

    // The bug: null struct is not preserved
    assert!(
        read_inner_struct.is_null(1),
        "Second struct should be null but it's not. Read value: {:?}",
        read_inner_struct
    );

    // Verify the null count is preserved
    assert_eq!(
        inner_struct_array.null_count(),
        read_inner_struct.null_count(),
        "Null count should be preserved"
    );
}

#[tokio::test]
async fn test_issue_4902_packed_struct_v2_1_read_error() {
    use std::collections::HashMap;

    use arrow_array::{ArrayRef, Int32Array, RecordBatchIterator, StructArray, UInt32Array};
    use arrow_schema::{Field as ArrowField, Fields, Schema as ArrowSchema};

    let struct_fields = Fields::from(vec![
        ArrowField::new("x", DataType::UInt32, false),
        ArrowField::new("y", DataType::UInt32, false),
    ]);
    let mut packed_metadata = HashMap::new();
    packed_metadata.insert("packed".to_string(), "true".to_string());

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("int_col", DataType::Int32, false),
        ArrowField::new("struct_col", DataType::Struct(struct_fields.clone()), false)
            .with_metadata(packed_metadata),
    ]));

    let int_values = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]));
    let x_values = Arc::new(UInt32Array::from(vec![1, 4, 7, 10, 13, 16, 19, 22]));
    let y_values = Arc::new(UInt32Array::from(vec![2, 5, 8, 11, 14, 17, 20, 23]));
    let struct_array = Arc::new(StructArray::new(
        struct_fields,
        vec![x_values.clone() as ArrayRef, y_values.clone() as ArrayRef],
        None,
    ));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            int_values.clone() as ArrayRef,
            struct_array.clone() as ArrayRef,
        ],
    )
    .unwrap();

    let test_uri = TempStrDir::default();
    let write_params = WriteParams {
        mode: WriteMode::Create,
        data_storage_version: Some(LanceFileVersion::V2_1),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone());
    Dataset::write(reader, &test_uri, Some(write_params))
        .await
        .unwrap();

    let dataset = Dataset::open(&test_uri).await.unwrap();

    let result_batches = dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(result_batches, vec![batch.clone()]);

    let struct_batches = dataset
        .scan()
        .project(&["struct_col"])
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(struct_batches.len(), 1);
    let read_struct = struct_batches[0].column(0).as_struct();
    assert_eq!(read_struct, struct_array.as_ref());
}

#[tokio::test]
async fn test_issue_4429_nested_struct_encoding_v2_1_with_over_65k_structs() {
    // Regression test for miniblock 16KB limit with nested struct patterns
    // Tests encoding behavior when a nested struct<list<struct>> contains
    // large amounts of data that exceeds miniblock encoding limits

    // Create a struct with multiple fields that will trigger miniblock encoding
    // Each field is 4 bytes, making the struct narrow enough for miniblock
    let measurement_fields = vec![
        ArrowField::new("val_a", DataType::Float32, true),
        ArrowField::new("val_b", DataType::Float32, true),
        ArrowField::new("val_c", DataType::Float32, true),
        ArrowField::new("val_d", DataType::Float32, true),
        ArrowField::new("seq_high", DataType::Int32, true),
        ArrowField::new("seq_low", DataType::Int32, true),
    ];
    let measurement_type = DataType::Struct(measurement_fields.clone().into());

    // Create nested schema: struct<measurements: list<struct>>
    // This pattern can trigger encoding issues with large data volumes
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "data",
        DataType::Struct(
            vec![ArrowField::new(
                "measurements",
                DataType::List(Arc::new(ArrowField::new(
                    "item",
                    measurement_type.clone(),
                    true,
                ))),
                true,
            )]
            .into(),
        ),
        true,
    )]));

    // Create large number of measurements that will exceed encoding limits
    // Using 70,520 to match the exact problematic size
    const NUM_MEASUREMENTS: usize = 70_520;

    // Generate data for two full sets (rows 0 and 2 will have data, row 1 empty)
    const TOTAL_MEASUREMENTS: usize = NUM_MEASUREMENTS * 2;

    // Create arrays with realistic values
    let val_a_array =
        Float32Array::from_iter((0..TOTAL_MEASUREMENTS).map(|i| Some(16.66 + (i as f32 * 0.0001))));
    let val_b_array =
        Float32Array::from_iter((0..TOTAL_MEASUREMENTS).map(|i| Some(-3.54 + (i as f32 * 0.0002))));
    let val_c_array =
        Float32Array::from_iter((0..TOTAL_MEASUREMENTS).map(|i| Some(2.94 + (i as f32 * 0.0001))));
    let val_d_array =
        Float32Array::from_iter((0..TOTAL_MEASUREMENTS).map(|i| Some(((i % 50) + 10) as f32)));
    let seq_high_array = Int32Array::from_iter((0..TOTAL_MEASUREMENTS).map(|_| Some(1736962329)));
    let seq_low_array =
        Int32Array::from_iter((0..TOTAL_MEASUREMENTS).map(|i| Some(304403000 + (i * 1000) as i32)));

    // Create the struct array with all measurements
    let struct_array = StructArray::from(vec![
        (
            Arc::new(ArrowField::new("val_a", DataType::Float32, true)),
            Arc::new(val_a_array) as ArrayRef,
        ),
        (
            Arc::new(ArrowField::new("val_b", DataType::Float32, true)),
            Arc::new(val_b_array) as ArrayRef,
        ),
        (
            Arc::new(ArrowField::new("val_c", DataType::Float32, true)),
            Arc::new(val_c_array) as ArrayRef,
        ),
        (
            Arc::new(ArrowField::new("val_d", DataType::Float32, true)),
            Arc::new(val_d_array) as ArrayRef,
        ),
        (
            Arc::new(ArrowField::new("seq_high", DataType::Int32, true)),
            Arc::new(seq_high_array) as ArrayRef,
        ),
        (
            Arc::new(ArrowField::new("seq_low", DataType::Int32, true)),
            Arc::new(seq_low_array) as ArrayRef,
        ),
    ]);

    // Create list array with pattern: [70520 items, 0 items, 70520 items]
    // This pattern triggers the issue with V2.1 encoding
    let offsets = vec![
        0i32,
        NUM_MEASUREMENTS as i32,       // End of row 0
        NUM_MEASUREMENTS as i32,       // End of row 1 (empty)
        (NUM_MEASUREMENTS * 2) as i32, // End of row 2
    ];
    let list_array = ListArray::try_new(
        Arc::new(ArrowField::new("item", measurement_type, true)),
        arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(offsets)),
        Arc::new(struct_array) as ArrayRef,
        None,
    )
    .unwrap();

    // Create the outer struct wrapping the list
    let data_struct = StructArray::from(vec![(
        Arc::new(ArrowField::new(
            "measurements",
            DataType::List(Arc::new(ArrowField::new(
                "item",
                DataType::Struct(measurement_fields.into()),
                true,
            ))),
            true,
        )),
        Arc::new(list_array) as ArrayRef,
    )]);

    // Create the final record batch with 3 rows
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(data_struct) as ArrayRef]).unwrap();

    assert_eq!(batch.num_rows(), 3, "Should have exactly 3 rows");

    let test_uri = TempStrDir::default();

    // Test with V2.1 format which has different encoding behavior
    let batches = vec![batch];
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

    // V2.1 format triggers miniblock encoding for narrow structs
    let write_params = WriteParams {
        data_storage_version: Some(lance_file::version::LanceFileVersion::V2_1),
        ..Default::default()
    };

    // Write dataset - this will panic with miniblock 16KB assertion
    let dataset = Dataset::write(reader, &test_uri, Some(write_params))
        .await
        .unwrap();

    dataset.validate().await.unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 3);
}
