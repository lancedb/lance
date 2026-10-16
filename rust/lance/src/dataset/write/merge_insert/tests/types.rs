// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[tokio::test]
async fn test_merge_insert_full_fragment_rewrite_with_json_columns() {
    // This test verifies the "all rows updated" fast path in handle_fragment
    // correctly converts Arrow JSON (Utf8) to Lance JSON (LargeBinary/JSONB)
    // before writing. Without conversion, the file would have Utf8 data (i32
    // offsets) but schema says LargeBinary (i64 offsets), causing decoder panic
    // on subsequent reads.
    //
    // To trigger the fast path we need:
    // 1. Subschema update (not all columns) → forces v1 update_fragments path
    // 2. ALL rows in a fragment updated → triggers the fast path
    use lance_arrow::ARROW_EXT_NAME_KEY;
    use lance_arrow::json::{ARROW_JSON_EXT_NAME, is_arrow_json_field};

    let test_dir = TempStrDir::default();
    let mut json_metadata = HashMap::new();
    json_metadata.insert(
        ARROW_EXT_NAME_KEY.to_string(),
        ARROW_JSON_EXT_NAME.to_string(),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("score", DataType::Int64, true),
        Field::new("meta", DataType::Utf8, true).with_metadata(json_metadata.clone()),
    ]));
    // Small fragment so ALL rows will be updated
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Int64Array::from(vec![10, 20, 30])),
            Arc::new(StringArray::from(vec![
                r#"{"x":1}"#,
                r#"{"x":2}"#,
                r#"{"x":3}"#,
            ])),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let write_params = WriteParams {
        data_storage_version: Some(LanceFileVersion::V2_2),
        ..Default::default()
    };
    let dataset = Arc::new(
        Dataset::write(reader, test_dir.as_ref(), Some(write_params))
            .await
            .unwrap(),
    );
    assert_eq!(dataset.get_fragments().len(), 1);

    // Subschema update: only provide [id, meta] (missing "name" and "score")
    // This forces the v1 path (update_fragments) instead of v2 (create_plan).
    // Update ALL rows → triggers the "all rows updated" fast path.
    let update_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("meta", DataType::Utf8, true).with_metadata(json_metadata),
    ]));
    let update_batch = RecordBatch::try_new(
        update_schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])), // all rows
            Arc::new(StringArray::from(vec![
                r#"{"updated":true,"id":1}"#,
                r#"{"updated":true,"id":2}"#,
                r#"{"updated":true,"id":3}"#,
            ])),
        ],
    )
    .unwrap();
    let update_reader: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
        vec![Ok(update_batch)],
        update_schema,
    ));
    let stream = reader_to_stream(update_reader);

    // Execute merge_insert with subschema
    let mut builder = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()]).unwrap();
    builder.when_matched(WhenMatched::UpdateAll);
    builder.when_not_matched(WhenNotMatched::DoNothing);
    let job = builder.try_build().unwrap();
    let (updated_dataset, stats) = job.execute(stream).await.unwrap();

    assert_eq!(stats.num_updated_rows, 3);

    // Critical: read the data back. Without the fix, this would PANIC with:
    // "the offset of the new Buffer cannot exceed the existing Length:
    //  slice offset=0 Length=N selfLen=N/2"
    let batches = updated_dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let result = concat_batches(&batches[0].schema(), &batches).unwrap();
    assert_eq!(result.num_rows(), 3);

    // Verify JSON column is in Arrow JSON format (Utf8) on read
    let result_schema = result.schema();
    let meta_field = result_schema.field_with_name("meta").unwrap();
    assert!(
        is_arrow_json_field(meta_field),
        "Expected Arrow JSON (Utf8 + arrow.json), got {:?}",
        meta_field
    );

    // Verify data correctness
    let metas = result
        .column_by_name("meta")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("meta should be StringArray after read conversion");
    for i in 0..3 {
        let val = metas.value(i);
        assert!(
            val.contains("updated"),
            "row {} should have updated meta, got: {}",
            i,
            val
        );
    }

    // Verify non-updated columns are preserved
    let scores = result
        .column_by_name("score")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(scores.values(), &[10, 20, 30]);

    // Also verify via take (exercises the take read conversion path)
    let take_result = updated_dataset
        .take(&[0, 1, 2], updated_dataset.schema().clone())
        .await
        .unwrap();
    let take_schema = take_result.schema();
    let take_meta_field = take_schema.field_with_name("meta").unwrap();
    assert!(
        is_arrow_json_field(take_meta_field),
        "take() should return Arrow JSON, got {:?}",
        take_meta_field
    );
}

#[tokio::test]
async fn test_merge_insert_subschema_with_json_columns() {
    use lance_arrow::ARROW_EXT_NAME_KEY;
    use lance_arrow::json::ARROW_JSON_EXT_NAME;

    // Create a dataset with an Arrow JSON extension column
    let test_dir = TempStrDir::default();
    let mut json_metadata = HashMap::new();
    json_metadata.insert(
        ARROW_EXT_NAME_KEY.to_string(),
        ARROW_JSON_EXT_NAME.to_string(),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("score", DataType::Int64, true),
        Field::new("meta", DataType::Utf8, true).with_metadata(json_metadata.clone()),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"])),
            Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50])),
            Arc::new(StringArray::from(vec![
                r#"{"x":1}"#,
                r#"{"x":2}"#,
                r#"{"x":3}"#,
                r#"{"x":4}"#,
                r#"{"x":5}"#,
            ])),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let dataset = Arc::new(
        Dataset::write(reader, test_dir.as_ref(), None)
            .await
            .unwrap(),
    );

    // Perform a subschema merge_insert: only update "meta" column (JSON type)
    // This exercises the update_fragments path with interleave_batches
    let update_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("meta", DataType::Utf8, true).with_metadata(json_metadata),
    ]));
    let update_batch = RecordBatch::try_new(
        update_schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![2, 4])),
            Arc::new(StringArray::from(vec![
                r#"{"updated":true,"id":2}"#,
                r#"{"updated":true,"id":4}"#,
            ])),
        ],
    )
    .unwrap();
    let update_reader: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
        vec![Ok(update_batch)],
        update_schema,
    ));
    let stream = reader_to_stream(update_reader);

    // Execute merge_insert with subschema (only id + meta columns)
    let mut builder = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()]).unwrap();
    builder.when_matched(WhenMatched::UpdateAll);
    builder.when_not_matched(WhenNotMatched::DoNothing);
    let job = builder.try_build().unwrap();
    let (updated_dataset, stats) = job.execute(stream).await.unwrap();

    // Verify: the merge should not fail with type mismatch
    assert_eq!(stats.num_updated_rows, 2);

    // Read back and verify the JSON column was updated correctly
    let batches = updated_dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let result = concat_batches(&batches[0].schema(), &batches).unwrap();
    assert_eq!(result.num_rows(), 5);

    // Verify the "score" column (not in update) is preserved, and "meta" updated
    let ids = result
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let scores = result
        .column_by_name("score")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let metas = result
        .column_by_name("meta")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for i in 0..5 {
        let id = ids.value(i);
        let score = scores.value(i);
        let meta = metas.value(i);
        // score = id * 10, regardless of row order
        assert_eq!(score, id * 10, "id={} score mismatch", id);
        if id == 2 || id == 4 {
            assert!(
                meta.contains("updated"),
                "id={} should have updated meta, got: {}",
                id,
                meta
            );
        } else {
            assert!(
                meta.contains("\"x\""),
                "id={} should have original meta, got: {}",
                id,
                meta
            );
        }
    }
}

/// A legacy blob source merges into a 2.1 table, and into a 2.2 table whose
/// column is Blob v2. Partial + indexed patches the blob column in place,
/// which fails for Blob v2 sources too, so it is not covered here.
#[rstest::rstest]
#[case::full(false, false)]
#[case::full_indexed(true, false)]
#[case::partial(false, true)]
#[tokio::test]
async fn test_merge_insert_with_blob_v1_source_provides_blob(
    #[case] indexed: bool,
    #[case] partial: bool,
    #[values(LanceFileVersion::V2_1, LanceFileVersion::V2_2)] version: LanceFileVersion,
    #[values(false, true)] delete_unmatched: bool,
) {
    use arrow_array::LargeBinaryArray;
    use arrow_schema::Schema as ArrowSchema;
    use lance_arrow::BLOB_META_KEY;

    let test_dir = TempStrDir::default();
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("blobs", DataType::LargeBinary, true).with_metadata(HashMap::from([(
            BLOB_META_KEY.to_string(),
            "true".to_string(),
        )])),
        Field::new("id", DataType::Int64, true),
        Field::new("other", DataType::Int64, true),
    ]));
    let make_batch = |blob_values: Vec<Option<&[u8]>>, ids, others| {
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(LargeBinaryArray::from(blob_values)),
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int64Array::from(others)),
            ],
        )
        .unwrap()
    };
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(
            vec![Ok(make_batch(
                vec![Some(b"foo"), Some(b"bar")],
                vec![0, 1],
                vec![10, 20],
            ))],
            schema.clone(),
        ),
        &test_dir,
        Some(WriteParams {
            data_storage_version: Some(version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    if indexed {
        dataset
            .create_index(
                &["id"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
    }
    let mut batch = make_batch(vec![Some(b"baz"), Some(b"qux")], vec![1, 2], vec![200, 300]);
    if partial {
        batch = batch.project(&[0, 1]).unwrap();
    }
    let source = Box::new(RecordBatchIterator::new(
        vec![Ok(batch.clone())],
        batch.schema(),
    ));

    let mut builder =
        MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()]).unwrap();
    builder
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll);
    if delete_unmatched {
        builder.when_not_matched_by_source(WhenNotMatchedBySource::Delete);
    }
    let (new_dataset, _) = builder
        .try_build()
        .unwrap()
        .execute_reader(source)
        .await
        .unwrap();

    let ids = new_dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let ids = ids["id"]
        .as_primitive::<arrow_array::types::Int64Type>()
        .values();
    let indices: Vec<u64> = (0..ids.len() as u64).collect();
    let blobs = new_dataset
        .take_blobs_by_indices(&indices, "blobs")
        .await
        .unwrap();
    let mut actual = Vec::new();
    for (id, blob) in ids.iter().zip(blobs) {
        actual.push((*id, blob.unwrap().read().await.unwrap().to_vec()));
    }
    actual.sort();
    let mut expected = vec![(1, b"baz".to_vec()), (2, b"qux".to_vec())];
    if !delete_unmatched {
        expected.insert(0, (0, b"foo".to_vec()));
    }
    assert_eq!(actual, expected);
}

#[tokio::test]
async fn test_merge_insert_with_blob_v2_source_provides_blob() {
    use crate::{BlobArrayBuilder, blob_field};
    use arrow_schema::Schema as ArrowSchema;

    let test_dir = TempStrDir::default();
    let schema = Arc::new(ArrowSchema::new(vec![
        blob_field("blobs", true),
        Field::new("id", DataType::Int64, true),
        Field::new("other", DataType::Int64, true),
    ]));
    let make_batch = |blob_values: &[&[u8]], ids, others| {
        let mut blobs = BlobArrayBuilder::new(blob_values.len());
        for value in blob_values {
            blobs.push_bytes(value).unwrap();
        }
        RecordBatch::try_new(
            schema.clone(),
            vec![
                blobs.finish().unwrap(),
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int64Array::from(others)),
            ],
        )
        .unwrap()
    };
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(
                vec![Ok(make_batch(&[b"foo", b"bar"], vec![0, 1], vec![10, 20]))],
                schema.clone(),
            ),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );
    let source = Box::new(RecordBatchIterator::new(
        vec![Ok(make_batch(
            &[b"baz", b"qux"],
            vec![1, 2],
            vec![200, 300],
        ))],
        schema,
    ));

    let job = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap();
    let (new_dataset, _) = job.execute_reader(source).await.unwrap();
    let blobs = new_dataset
        .take_blobs_by_indices(&[0, 1, 2], "blobs")
        .await
        .unwrap();
    assert_eq!(
        blobs[0].as_ref().unwrap().read().await.unwrap().as_ref(),
        b"foo"
    );
    assert_eq!(
        blobs[1].as_ref().unwrap().read().await.unwrap().as_ref(),
        b"baz"
    );
    assert_eq!(
        blobs[2].as_ref().unwrap().read().await.unwrap().as_ref(),
        b"qux"
    );
}

/// A partial-schema source carrying a Blob v2 column updates and inserts
/// whether or not the join key has a scalar index.
#[rstest::rstest]
#[tokio::test]
async fn test_merge_insert_partial_blob_v2_source(#[values(false, true)] indexed: bool) {
    use crate::{BlobArrayBuilder, blob_field};
    use arrow_schema::Schema as ArrowSchema;

    let test_dir = TempStrDir::default();
    let schema = Arc::new(ArrowSchema::new(vec![
        blob_field("blobs", true),
        Field::new("id", DataType::Int64, true),
        Field::new("other", DataType::Int64, true),
    ]));
    let make_batch = |blob_values: &[&[u8]], ids: Vec<i64>| {
        let mut blobs = BlobArrayBuilder::new(blob_values.len());
        for value in blob_values {
            blobs.push_bytes(value).unwrap();
        }
        let others = ids.iter().map(|id| id * 10).collect::<Vec<_>>();
        RecordBatch::try_new(
            schema.clone(),
            vec![
                blobs.finish().unwrap(),
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int64Array::from(others)),
            ],
        )
        .unwrap()
    };
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(
            vec![Ok(make_batch(&[b"foo", b"bar"], vec![0, 1]))],
            schema.clone(),
        ),
        &test_dir,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    if indexed {
        dataset
            .create_index(
                &["id"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
    }
    let source = make_batch(&[b"baz", b"qux"], vec![1, 2])
        .project(&[0, 1])
        .unwrap();
    let source = Box::new(RecordBatchIterator::new(
        vec![Ok(source.clone())],
        source.schema(),
    ));

    let (new_dataset, _) = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute_reader(source)
        .await
        .unwrap();

    let batch = new_dataset
        .scan()
        .project(&["id", "other"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let blobs = new_dataset
        .take_blobs_by_indices(&[0, 1, 2], "blobs")
        .await
        .unwrap();
    let ids = batch["id"].as_primitive::<arrow_array::types::Int64Type>();
    let others = batch["other"].as_primitive::<arrow_array::types::Int64Type>();
    let mut actual = Vec::new();
    for (row, blob) in blobs.into_iter().enumerate() {
        actual.push((
            ids.value(row),
            others.is_valid(row).then(|| others.value(row)),
            blob.unwrap().read().await.unwrap().to_vec(),
        ));
    }
    actual.sort();
    assert_eq!(
        actual,
        vec![
            (0, Some(0), b"foo".to_vec()),
            (1, Some(10), b"baz".to_vec()),
            (2, None, b"qux".to_vec()),
        ]
    );
}

#[tokio::test]
async fn test_merge_insert_preserves_carried_external_blob() {
    use crate::{BlobArrayBuilder, blob_field};

    let dataset_dir = TempStrDir::default();
    let external_dir = TempStrDir::default();
    let external_path = format!("{external_dir}/external.bin");
    std::fs::write(&external_path, b"external blob").unwrap();
    let external_uri = Url::from_file_path(&external_path).unwrap().to_string();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        blob_field("payload", true),
        Field::new("tag", DataType::Utf8, true),
    ]));
    let mut blobs = BlobArrayBuilder::new(2);
    blobs.push_uri(external_uri.clone()).unwrap();
    blobs.push_bytes(b"internal blob").unwrap();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            blobs.finish().unwrap(),
            Arc::new(StringArray::from(vec!["a", "other"])),
        ],
    )
    .unwrap();
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
            &dataset_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                max_rows_per_file: 1,
                allow_external_blob_outside_bases: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );
    assert_eq!(dataset.get_fragments().len(), 2);

    let source = record_batch!(("id", Int64, [1]), ("tag", Utf8, ["updated"])).unwrap();
    let job = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .try_build()
        .unwrap();
    let (dataset, _) = job
        .execute_reader(RecordBatchIterator::new(
            vec![Ok(source.clone())],
            source.schema(),
        ))
        .await
        .unwrap();

    let result = dataset
        .scan()
        .project(&["id", "tag"])
        .unwrap()
        .scan_in_order(true)
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 2);
    let ids = result["id"].as_primitive::<arrow_array::types::Int64Type>();
    let row = ids.values().iter().position(|id| *id == 1).unwrap();
    assert_eq!(result["tag"].as_string::<i32>().value(row), "updated");
    let blobs = dataset
        .take_blobs_by_indices(&[row as u64], "payload")
        .await
        .unwrap();
    let blob = blobs[0].as_ref().unwrap();
    assert_eq!(blob.kind(), lance_core::datatypes::BlobKind::External);
    assert_eq!(blob.uri(), Some(external_uri.as_str()));
    assert_eq!(blob.read().await.unwrap().as_ref(), b"external blob");

    let mut source_blob = BlobArrayBuilder::new(1);
    source_blob.push_uri(external_uri).unwrap();
    let source = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1])),
            source_blob.finish().unwrap(),
            Arc::new(StringArray::from(vec!["another update"])),
        ],
    )
    .unwrap();
    let job = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .try_build()
        .unwrap();
    let error = job
        .execute_reader(RecordBatchIterator::new(vec![Ok(source)], schema))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
    assert!(
        error
            .to_string()
            .contains("in field 'payload' is outside registered external bases"),
        "{error:?}"
    );
}

#[tokio::test]
async fn test_merge_insert_with_complete_blob_v2_preserves_schema() {
    use arrow_schema::Schema as ArrowSchema;
    use lance_arrow::{ARROW_EXT_NAME_KEY, BLOB_V2_EXT_NAME};
    use lance_core::datatypes::BLOB_V2_LOGICAL_FIELDS;

    let test_dir = TempStrDir::default();
    let blob_field = Field::new(
        "blobs",
        DataType::Struct(BLOB_V2_LOGICAL_FIELDS.clone()),
        true,
    )
    .with_metadata(HashMap::from([(
        ARROW_EXT_NAME_KEY.to_string(),
        BLOB_V2_EXT_NAME.to_string(),
    )]));
    let schema = Arc::new(ArrowSchema::new(vec![
        blob_field,
        Field::new("id", DataType::Int64, true),
        Field::new("other", DataType::Int64, true),
    ]));
    let make_batch = |blob_values: &[&[u8]], ids, others| {
        let blobs: arrow_array::ArrayRef = Arc::new(
            StructArray::try_new(
                BLOB_V2_LOGICAL_FIELDS.clone(),
                vec![
                    Arc::new(arrow_array::LargeBinaryArray::from_iter(
                        blob_values.iter().map(|value| Some(*value)),
                    )),
                    Arc::new(StringArray::from(vec![None::<&str>; blob_values.len()])),
                    Arc::new(UInt64Array::from(vec![None::<u64>; blob_values.len()])),
                    Arc::new(UInt64Array::from(vec![None::<u64>; blob_values.len()])),
                ],
                None,
            )
            .unwrap(),
        );
        RecordBatch::try_new(
            schema.clone(),
            vec![
                blobs,
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int64Array::from(others)),
            ],
        )
        .unwrap()
    };
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(
                vec![Ok(make_batch(&[b"foo", b"bar"], vec![0, 1], vec![10, 20]))],
                schema.clone(),
            ),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );
    let source = Box::new(RecordBatchIterator::new(
        vec![Ok(make_batch(
            &[b"baz", b"qux"],
            vec![1, 2],
            vec![200, 300],
        ))],
        schema,
    ));

    let job = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap();
    let (new_dataset, _) = job.execute_reader(source).await.unwrap();
    let dataset_schema = ArrowSchema::from(new_dataset.schema());
    let DataType::Struct(blob_children) =
        dataset_schema.field_with_name("blobs").unwrap().data_type()
    else {
        panic!("expected complete logical blob struct after merge insert");
    };
    assert_eq!(blob_children.as_ref(), BLOB_V2_LOGICAL_FIELDS.as_ref());
    let blobs = new_dataset
        .take_blobs_by_indices(&[0, 1, 2], "blobs")
        .await
        .unwrap();
    assert_eq!(
        blobs[0].as_ref().unwrap().read().await.unwrap().as_ref(),
        b"foo"
    );
    assert_eq!(
        blobs[1].as_ref().unwrap().read().await.unwrap().as_ref(),
        b"baz"
    );
    assert_eq!(
        blobs[2].as_ref().unwrap().read().await.unwrap().as_ref(),
        b"qux"
    );
}
