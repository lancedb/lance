// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::vec;

use crate::dataset::tests::dataset_migrations::scan_dataset;
use crate::dataset::tests::dataset_transactions::{assert_results, execute_sql};
use crate::dataset::ROW_ID;
use crate::index::vector::VectorIndexParams;
use crate::{Dataset, Error, Result};
use lance_arrow::FixedSizeListArrayExt;

use crate::dataset::write::{WriteMode, WriteParams};
use arrow::array::{AsArray, GenericListBuilder, GenericStringBuilder};
use arrow::datatypes::UInt64Type;
use arrow_array::RecordBatch;
use arrow_array::{
    builder::StringDictionaryBuilder,
    types::{Float32Type, Int32Type},
    ArrayRef, Float32Array, Int32Array, RecordBatchIterator, StringArray,
};
use arrow_array::{Array, GenericStringArray, StructArray, UInt64Array};
use arrow_schema::{
    DataType, Field as ArrowField, Field, Fields as ArrowFields, Schema as ArrowSchema,
};
use lance_arrow::ARROW_EXT_NAME_KEY;
use lance_core::utils::tempfile::TempStrDir;
use lance_datagen::{array, gen_batch, BatchCount, Dimension, RowCount};
use lance_file::version::LanceFileVersion;
use lance_index::scalar::inverted::{
    query::{BooleanQuery, MatchQuery, Occur, Operator, PhraseQuery},
    tokenizer::InvertedIndexParams,
};
use lance_index::scalar::FullTextSearchQuery;
use lance_index::DatasetIndexExt;
use lance_index::{scalar::ScalarIndexParams, vector::DIST_COL, IndexType};
use lance_linalg::distance::MetricType;

use datafusion::common::{assert_contains, assert_not_contains};
use futures::{StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_arrow::json::ARROW_JSON_EXT_NAME;
use lance_index::scalar::inverted::query::{FtsQuery, MultiMatchQuery};
use lance_testing::datagen::generate_random_array;
use rand::Rng;
use rstest::rstest;

#[rstest]
#[tokio::test]
async fn test_create_index(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_uri = TempStrDir::default();

    let dimension = 16;
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "embeddings",
        DataType::FixedSizeList(
            Arc::new(ArrowField::new("item", DataType::Float32, true)),
            dimension,
        ),
        false,
    )]));

    let float_arr = generate_random_array(512 * dimension as usize);
    let vectors = Arc::new(
        <arrow_array::FixedSizeListArray as FixedSizeListArrayExt>::try_new_from_values(
            float_arr, dimension,
        )
        .unwrap(),
    );
    let batches = vec![RecordBatch::try_new(schema.clone(), vec![vectors.clone()]).unwrap()];

    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

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
    dataset.validate().await.unwrap();

    // Make sure valid arguments should create index successfully
    let params = VectorIndexParams::ivf_pq(10, 8, 2, MetricType::L2, 50);
    dataset
        .create_index(&["embeddings"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    dataset.validate().await.unwrap();

    // The version should match the table version it was created from.
    let indices = dataset.load_indices().await.unwrap();
    let actual = indices.first().unwrap().dataset_version;
    let expected = dataset.manifest.version - 1;
    assert_eq!(actual, expected);
    let fragment_bitmap = indices.first().unwrap().fragment_bitmap.as_ref().unwrap();
    assert_eq!(fragment_bitmap.len(), 1);
    assert!(fragment_bitmap.contains(0));

    // Append should inherit index
    let write_params = WriteParams {
        mode: WriteMode::Append,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let batches = vec![RecordBatch::try_new(schema.clone(), vec![vectors.clone()]).unwrap()];
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let dataset = Dataset::write(reader, &test_uri, Some(write_params))
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    let actual = indices.first().unwrap().dataset_version;
    let expected = dataset.manifest.version - 2;
    assert_eq!(actual, expected);
    dataset.validate().await.unwrap();
    // Fragment bitmap should show the original fragments, and not include
    // the newly appended fragment.
    let fragment_bitmap = indices.first().unwrap().fragment_bitmap.as_ref().unwrap();
    assert_eq!(fragment_bitmap.len(), 1);
    assert!(fragment_bitmap.contains(0));

    let actual_statistics: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("embeddings_idx").await.unwrap()).unwrap();
    let actual_statistics = actual_statistics.as_object().unwrap();
    assert_eq!(actual_statistics["index_type"].as_str().unwrap(), "IVF_PQ");

    let deltas = actual_statistics["indices"].as_array().unwrap();
    assert_eq!(deltas.len(), 1);
    assert_eq!(deltas[0]["metric_type"].as_str().unwrap(), "l2");
    assert_eq!(deltas[0]["num_partitions"].as_i64().unwrap(), 10);

    assert!(dataset.index_statistics("non-existent_idx").await.is_err());
    assert!(dataset.index_statistics("").await.is_err());

    // Overwrite should invalidate index
    let write_params = WriteParams {
        mode: WriteMode::Overwrite,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let batches = vec![RecordBatch::try_new(schema.clone(), vec![vectors]).unwrap()];
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let dataset = Dataset::write(reader, &test_uri, Some(write_params))
        .await
        .unwrap();
    assert!(dataset.manifest.index_section.is_none());
    assert!(dataset.load_indices().await.unwrap().is_empty());
    dataset.validate().await.unwrap();

    let fragment_bitmap = indices.first().unwrap().fragment_bitmap.as_ref().unwrap();
    assert_eq!(fragment_bitmap.len(), 1);
    assert!(fragment_bitmap.contains(0));
}

#[rstest]
#[tokio::test]
async fn test_create_scalar_index(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
    #[values(false, true)] use_stable_row_id: bool,
) {
    let test_uri = TempStrDir::default();

    let data = gen_batch().col("int", array::step::<Int32Type>());
    // Write 64Ki rows.  We should get 16 4Ki pages
    let mut dataset = Dataset::write(
        data.into_reader_rows(RowCount::from(16 * 1024), BatchCount::from(4)),
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids: use_stable_row_id,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let index_name = "my_index".to_string();

    dataset
        .create_index(
            &["int"],
            IndexType::Scalar,
            Some(index_name.clone()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    let indices = dataset.load_indices_by_name(&index_name).await.unwrap();

    assert_eq!(indices.len(), 1);
    assert_eq!(indices[0].dataset_version, 1);
    assert_eq!(indices[0].fields, vec![0]);
    assert_eq!(indices[0].name, index_name);

    dataset.index_statistics(&index_name).await.unwrap();
}

async fn create_bad_file(data_storage_version: LanceFileVersion) -> Result<Dataset> {
    let test_uri = TempStrDir::default();

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "a.b.c",
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
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    Dataset::write(
        reader,
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
}

#[tokio::test]
async fn test_create_fts_index_with_empty_table() {
    let test_uri = TempStrDir::default();

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "text",
        DataType::Utf8,
        false,
    )]));

    let batches: Vec<RecordBatch> = vec![];
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(reader, &test_uri, None)
        .await
        .expect("write dataset");

    let params = InvertedIndexParams::default();
    dataset
        .create_index(&["text"], IndexType::Inverted, None, &params, true)
        .await
        .unwrap();

    let batch = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new("lance".to_owned()))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(batch.num_rows(), 0);
}

#[rstest]
#[tokio::test]
async fn test_create_int8_index(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    use lance_testing::datagen::generate_random_int8_array;

    let test_uri = TempStrDir::default();

    let dimension = 16;
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "embeddings",
        DataType::FixedSizeList(
            Arc::new(ArrowField::new("item", DataType::Int8, true)),
            dimension,
        ),
        false,
    )]));

    let int8_arr = generate_random_int8_array(512 * dimension as usize);
    let vectors = Arc::new(
        <arrow_array::FixedSizeListArray as FixedSizeListArrayExt>::try_new_from_values(
            int8_arr, dimension,
        )
        .unwrap(),
    );
    let batches = vec![RecordBatch::try_new(schema.clone(), vec![vectors.clone()]).unwrap()];

    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

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
    dataset.validate().await.unwrap();

    // Make sure valid arguments should create index successfully
    let params = VectorIndexParams::ivf_pq(10, 8, 2, MetricType::L2, 50);
    dataset
        .create_index(&["embeddings"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    dataset.validate().await.unwrap();

    // The version should match the table version it was created from.
    let indices = dataset.load_indices().await.unwrap();
    let actual = indices.first().unwrap().dataset_version;
    let expected = dataset.manifest.version - 1;
    assert_eq!(actual, expected);
    let fragment_bitmap = indices.first().unwrap().fragment_bitmap.as_ref().unwrap();
    assert_eq!(fragment_bitmap.len(), 1);
    assert!(fragment_bitmap.contains(0));

    // Append should inherit index
    let write_params = WriteParams {
        mode: WriteMode::Append,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let batches = vec![RecordBatch::try_new(schema.clone(), vec![vectors.clone()]).unwrap()];
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let dataset = Dataset::write(reader, &test_uri, Some(write_params))
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    let actual = indices.first().unwrap().dataset_version;
    let expected = dataset.manifest.version - 2;
    assert_eq!(actual, expected);
    dataset.validate().await.unwrap();
    // Fragment bitmap should show the original fragments, and not include
    // the newly appended fragment.
    let fragment_bitmap = indices.first().unwrap().fragment_bitmap.as_ref().unwrap();
    assert_eq!(fragment_bitmap.len(), 1);
    assert!(fragment_bitmap.contains(0));

    let actual_statistics: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("embeddings_idx").await.unwrap()).unwrap();
    let actual_statistics = actual_statistics.as_object().unwrap();
    assert_eq!(actual_statistics["index_type"].as_str().unwrap(), "IVF_PQ");

    let deltas = actual_statistics["indices"].as_array().unwrap();
    assert_eq!(deltas.len(), 1);
    assert_eq!(deltas[0]["metric_type"].as_str().unwrap(), "l2");
    assert_eq!(deltas[0]["num_partitions"].as_i64().unwrap(), 10);

    assert!(dataset.index_statistics("non-existent_idx").await.is_err());
    assert!(dataset.index_statistics("").await.is_err());

    // Overwrite should invalidate index
    let write_params = WriteParams {
        mode: WriteMode::Overwrite,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let batches = vec![RecordBatch::try_new(schema.clone(), vec![vectors]).unwrap()];
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let dataset = Dataset::write(reader, &test_uri, Some(write_params))
        .await
        .unwrap();
    assert!(dataset.manifest.index_section.is_none());
    assert!(dataset.load_indices().await.unwrap().is_empty());
    dataset.validate().await.unwrap();

    let fragment_bitmap = indices.first().unwrap().fragment_bitmap.as_ref().unwrap();
    assert_eq!(fragment_bitmap.len(), 1);
    assert!(fragment_bitmap.contains(0));
}

#[tokio::test]
async fn test_create_fts_index_with_empty_strings() {
    let test_uri = TempStrDir::default();

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "text",
        DataType::Utf8,
        false,
    )]));

    let batches: Vec<RecordBatch> = vec![RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(StringArray::from(vec!["", "", ""]))],
    )
    .unwrap()];
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(reader, &test_uri, None)
        .await
        .expect("write dataset");

    let params = InvertedIndexParams::default();
    dataset
        .create_index(&["text"], IndexType::Inverted, None, &params, true)
        .await
        .unwrap();

    let batch = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new("lance".to_owned()))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(batch.num_rows(), 0);
}

#[rstest]
#[tokio::test]
async fn test_bad_field_name(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    // don't allow `.` in the field name
    assert!(create_bad_file(data_storage_version).await.is_err());
}

#[tokio::test]
async fn test_open_dataset_not_found() {
    let result = Dataset::open(".").await;
    assert!(matches!(result.unwrap_err(), Error::DatasetNotFound { .. }));
}

#[rstest]
#[tokio::test]
async fn test_search_empty(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    // Create a table
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "vec",
        DataType::FixedSizeList(
            Arc::new(ArrowField::new("item", DataType::Float32, true)),
            128,
        ),
        false,
    )]));

    let test_uri = TempStrDir::default();

    let vectors = Arc::new(
        <arrow_array::FixedSizeListArray as FixedSizeListArrayExt>::try_new_from_values(
            Float32Array::from_iter_values(vec![]),
            128,
        )
        .unwrap(),
    );

    let data = RecordBatch::try_new(schema.clone(), vec![vectors]);
    let reader = RecordBatchIterator::new(vec![data.unwrap()].into_iter().map(Ok), schema);
    let dataset = Dataset::write(
        reader,
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let mut stream = dataset
        .scan()
        .nearest(
            "vec",
            &Float32Array::from_iter_values((0..128).map(|_| 0.1)),
            1,
        )
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();

    while let Some(batch) = stream.next().await {
        let schema = batch.unwrap().schema();
        assert_eq!(schema.fields.len(), 2);
        assert_eq!(
            schema.field_with_name("vec").unwrap(),
            &ArrowField::new(
                "vec",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    128
                ),
                false,
            )
        );
        assert_eq!(
            schema.field_with_name(DIST_COL).unwrap(),
            &ArrowField::new(DIST_COL, DataType::Float32, true)
        );
    }
}

#[rstest]
#[tokio::test]
async fn test_search_empty_after_delete(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
    #[values(false, true)] use_stable_row_id: bool,
) {
    // Create a table
    let test_uri = TempStrDir::default();

    let data = gen_batch().col("vec", array::rand_vec::<Float32Type>(Dimension::from(32)));
    let reader = data.into_reader_rows(RowCount::from(500), BatchCount::from(1));
    let mut dataset = Dataset::write(
        reader,
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids: use_stable_row_id,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let params = VectorIndexParams::ivf_pq(1, 8, 1, MetricType::L2, 50);
    dataset
        .create_index(&["vec"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    dataset.delete("true").await.unwrap();

    // This behavior will be re-introduced once we work on empty vector index handling.
    // https://github.com/lance-format/lance/issues/4034
    // let indices = dataset.load_indices().await.unwrap();
    // // With the new retention behavior, indices are kept even when all fragments are deleted
    // // This allows the index configuration to persist through data changes
    // assert_eq!(indices.len(), 1);

    // // Verify the index has an empty effective fragment bitmap
    // let index = &indices[0];
    // let effective_bitmap = index
    //     .effective_fragment_bitmap(&dataset.fragment_bitmap)
    //     .unwrap();
    // assert!(effective_bitmap.is_empty());

    let mut stream = dataset
        .scan()
        .nearest(
            "vec",
            &Float32Array::from_iter_values((0..32).map(|_| 0.1)),
            1,
        )
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();

    while let Some(batch) = stream.next().await {
        let schema = batch.unwrap().schema();
        assert_eq!(schema.fields.len(), 2);
        assert_eq!(
            schema.field_with_name("vec").unwrap(),
            &ArrowField::new(
                "vec",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    32
                ),
                false,
            )
        );
        assert_eq!(
            schema.field_with_name(DIST_COL).unwrap(),
            &ArrowField::new(DIST_COL, DataType::Float32, true)
        );
    }

    // predicate with redundant whitespace
    dataset.delete(" True").await.unwrap();

    let mut stream = dataset
        .scan()
        .nearest(
            "vec",
            &Float32Array::from_iter_values((0..32).map(|_| 0.1)),
            1,
        )
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();

    while let Some(batch) = stream.next().await {
        let batch = batch.unwrap();
        let schema = batch.schema();
        assert_eq!(schema.fields.len(), 2);
        assert_eq!(
            schema.field_with_name("vec").unwrap(),
            &ArrowField::new(
                "vec",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    32
                ),
                false,
            )
        );
        assert_eq!(
            schema.field_with_name(DIST_COL).unwrap(),
            &ArrowField::new(DIST_COL, DataType::Float32, true)
        );
        assert_eq!(batch.num_rows(), 0, "Expected no results after delete");
    }
}

#[rstest]
#[tokio::test]
async fn test_num_small_files(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_uri = TempStrDir::default();
    let dimensions = 16;
    let column_name = "vec";
    let field = ArrowField::new(
        column_name,
        DataType::FixedSizeList(
            Arc::new(ArrowField::new("item", DataType::Float32, true)),
            dimensions,
        ),
        false,
    );

    let schema = Arc::new(ArrowSchema::new(vec![field]));

    let float_arr = generate_random_array(512 * dimensions as usize);
    let vectors =
        arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();

    let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();

    let reader = RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());

    let dataset = Dataset::write(
        reader,
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset.validate().await.unwrap();

    assert!(dataset.num_small_files(1024).await > 0);
    assert!(dataset.num_small_files(512).await == 0);
}

#[tokio::test]
async fn test_read_struct_of_dictionary_arrays() {
    let test_uri = TempStrDir::default();

    let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        DataType::Struct(ArrowFields::from(vec![ArrowField::new(
            "d",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            true,
        )])),
        true,
    )]));

    let mut batches: Vec<RecordBatch> = Vec::new();
    for _ in 1..2 {
        let mut dict_builder = StringDictionaryBuilder::<Int32Type>::new();
        dict_builder.append("a").unwrap();
        dict_builder.append("b").unwrap();
        dict_builder.append("c").unwrap();
        dict_builder.append("d").unwrap();

        let struct_array = Arc::new(StructArray::from(vec![(
            Arc::new(ArrowField::new(
                "d",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            )),
            Arc::new(dict_builder.finish()) as ArrayRef,
        )]));

        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![struct_array.clone()]).unwrap();
        batches.push(batch);
    }

    let batch_reader =
        RecordBatchIterator::new(batches.clone().into_iter().map(Ok), arrow_schema.clone());
    Dataset::write(batch_reader, &test_uri, Some(WriteParams::default()))
        .await
        .unwrap();

    let result = scan_dataset(&test_uri).await.unwrap();

    assert_eq!(batches, result);
}

#[tokio::test]
async fn test_fts_fuzzy_query() {
    let params = InvertedIndexParams::default();
    let text_col = GenericStringArray::<i32>::from(vec![
        "fa", "fo", "fob", "focus", "foo", "food", "foul", // # spellchecker:disable-line
    ]);
    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "text",
            text_col.data_type().to_owned(),
            false,
        )])
        .into(),
        vec![Arc::new(text_col) as ArrayRef],
    )
    .unwrap();
    let schema = batch.schema();
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
    let test_uri = TempStrDir::default();
    let mut dataset = Dataset::write(batches, &test_uri, None).await.unwrap();
    dataset
        .create_index(&["text"], IndexType::Inverted, None, &params, true)
        .await
        .unwrap();
    let results = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new_fuzzy("foo".to_owned(), Some(1)))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 4);
    let texts = results["text"]
        .as_string::<i32>()
        .iter()
        .map(|s| s.unwrap().to_owned())
        .collect::<HashSet<_>>();
    assert_eq!(
        texts,
        vec![
            "foo".to_owned(),  // 0 edits
            "fo".to_owned(),   // 1 deletion        # spellchecker:disable-line
            "fob".to_owned(),  // 1 substitution    # spellchecker:disable-line
            "food".to_owned(), // 1 insertion       # spellchecker:disable-line
        ]
        .into_iter()
        .collect()
    );
}

#[tokio::test]
async fn test_fts_on_multiple_columns() {
    let params = InvertedIndexParams::default();
    let title_col =
        GenericStringArray::<i32>::from(vec!["title common", "title hello", "title lance"]);
    let content_col = GenericStringArray::<i32>::from(vec![
        "content world",
        "content database",
        "content common",
    ]);
    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("title", title_col.data_type().to_owned(), false),
            arrow_schema::Field::new("content", title_col.data_type().to_owned(), false),
        ])
        .into(),
        vec![
            Arc::new(title_col) as ArrayRef,
            Arc::new(content_col) as ArrayRef,
        ],
    )
    .unwrap();
    let schema = batch.schema();
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
    let test_uri = TempStrDir::default();
    let mut dataset = Dataset::write(batches, &test_uri, None).await.unwrap();
    dataset
        .create_index(&["title"], IndexType::Inverted, None, &params, true)
        .await
        .unwrap();
    dataset
        .create_index(&["content"], IndexType::Inverted, None, &params, true)
        .await
        .unwrap();

    let results = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new("title".to_owned()))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 3);

    let results = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new("content".to_owned()))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 3);

    let results = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new("common".to_owned()))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 2);

    let results = dataset
        .scan()
        .full_text_search(
            FullTextSearchQuery::new("common".to_owned())
                .with_column("title".to_owned())
                .unwrap(),
        )
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 1);

    let results = dataset
        .scan()
        .full_text_search(
            FullTextSearchQuery::new("common".to_owned())
                .with_column("content".to_owned())
                .unwrap(),
        )
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 1);
}

#[tokio::test]
async fn test_fts_unindexed_data() {
    let params = InvertedIndexParams::default();
    let title_col = StringArray::from(vec!["title hello", "title lance", "title common"]);
    let content_col =
        StringArray::from(vec!["content world", "content database", "content common"]);
    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![
            Field::new("title", title_col.data_type().to_owned(), false),
            Field::new("content", title_col.data_type().to_owned(), false),
        ])
        .into(),
        vec![
            Arc::new(title_col) as ArrayRef,
            Arc::new(content_col) as ArrayRef,
        ],
    )
    .unwrap();
    let schema = batch.schema();
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
    let mut dataset = Dataset::write(batches, "memory://test.lance", None)
        .await
        .unwrap();
    dataset
        .create_index(&["title"], IndexType::Inverted, None, &params, true)
        .await
        .unwrap();

    let results = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new("title".to_owned()))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 3);

    // write new data
    let title_col = StringArray::from(vec!["new title"]);
    let content_col = StringArray::from(vec!["new content"]);
    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![
            Field::new("title", title_col.data_type().to_owned(), false),
            Field::new("content", title_col.data_type().to_owned(), false),
        ])
        .into(),
        vec![
            Arc::new(title_col) as ArrayRef,
            Arc::new(content_col) as ArrayRef,
        ],
    )
    .unwrap();
    let schema = batch.schema();
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
    dataset.append(batches, None).await.unwrap();

    let results = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new("title".to_owned()))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 4);

    let results = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new("new".to_owned()))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 1);
}

#[tokio::test]
async fn test_fts_unindexed_data_on_empty_index() {
    // Empty dataset with fts index
    let params = InvertedIndexParams::default();
    let title_col = StringArray::from(Vec::<&str>::new());
    let content_col = StringArray::from(Vec::<&str>::new());
    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![
            Field::new("title", title_col.data_type().to_owned(), false),
            Field::new("content", title_col.data_type().to_owned(), false),
        ])
        .into(),
        vec![
            Arc::new(title_col) as ArrayRef,
            Arc::new(content_col) as ArrayRef,
        ],
    )
    .unwrap();
    let schema = batch.schema();
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
    let mut dataset = Dataset::write(batches, "memory://test.lance", None)
        .await
        .unwrap();
    dataset
        .create_index(&["title"], IndexType::Inverted, None, &params, true)
        .await
        .unwrap();

    // Test fts search
    let results = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Match(
            MatchQuery::new("title".to_owned()).with_column(Some("title".to_owned())),
        )))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 0);

    // write new data
    let title_col = StringArray::from(vec!["title hello", "title lance", "title common"]);
    let content_col =
        StringArray::from(vec!["content world", "content database", "content common"]);
    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![
            Field::new("title", title_col.data_type().to_owned(), false),
            Field::new("content", title_col.data_type().to_owned(), false),
        ])
        .into(),
        vec![
            Arc::new(title_col) as ArrayRef,
            Arc::new(content_col) as ArrayRef,
        ],
    )
    .unwrap();
    let schema = batch.schema();
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
    dataset.append(batches, None).await.unwrap();

    let results = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Match(
            MatchQuery::new("title".to_owned()).with_column(Some("title".to_owned())),
        )))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 3);
}

#[tokio::test]
async fn test_fts_without_index() {
    // create table without index
    let title_col = StringArray::from(vec!["title hello", "title lance", "title common"]);
    let content_col =
        StringArray::from(vec!["content world", "content database", "content common"]);
    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![
            Field::new("title", title_col.data_type().to_owned(), false),
            Field::new("content", title_col.data_type().to_owned(), false),
        ])
        .into(),
        vec![
            Arc::new(title_col) as ArrayRef,
            Arc::new(content_col) as ArrayRef,
        ],
    )
    .unwrap();
    let schema = batch.schema();
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
    let mut dataset = Dataset::write(batches, "memory://test.lance", None)
        .await
        .unwrap();

    // match query on title and content
    let results = dataset
        .scan()
        .full_text_search(
            FullTextSearchQuery::new("title".to_owned())
                .with_columns(&["title".to_string(), "content".to_string()])
                .unwrap(),
        )
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 3);

    // write new data
    let title_col = StringArray::from(vec!["new title"]);
    let content_col = StringArray::from(vec!["new content"]);
    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![
            Field::new("title", title_col.data_type().to_owned(), false),
            Field::new("content", title_col.data_type().to_owned(), false),
        ])
        .into(),
        vec![
            Arc::new(title_col) as ArrayRef,
            Arc::new(content_col) as ArrayRef,
        ],
    )
    .unwrap();
    let schema = batch.schema();
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
    dataset.append(batches, None).await.unwrap();

    // match query on title and content
    let results = dataset
        .scan()
        .full_text_search(
            FullTextSearchQuery::new("title".to_owned())
                .with_columns(&["title".to_string(), "content".to_string()])
                .unwrap(),
        )
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 4);

    let results = dataset
        .scan()
        .full_text_search(
            FullTextSearchQuery::new("new".to_owned())
                .with_columns(&["title".to_string(), "content".to_string()])
                .unwrap(),
        )
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 1);
}

#[tokio::test]
async fn test_fts_rank() {
    let params = InvertedIndexParams::default();
    let text_col =
        GenericStringArray::<i32>::from(vec!["score", "find score", "try to find score"]);
    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "text",
            text_col.data_type().to_owned(),
            false,
        )])
        .into(),
        vec![Arc::new(text_col) as ArrayRef],
    )
    .unwrap();
    let schema = batch.schema();
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
    let test_uri = TempStrDir::default();
    let mut dataset = Dataset::write(batches, &test_uri, None).await.unwrap();
    dataset
        .create_index(&["text"], IndexType::Inverted, None, &params, true)
        .await
        .unwrap();

    let results = dataset
        .scan()
        .with_row_id()
        .full_text_search(FullTextSearchQuery::new("score".to_owned()))
        .unwrap()
        .limit(Some(3), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 3);
    let row_ids = results[ROW_ID].as_primitive::<UInt64Type>().values();
    assert_eq!(row_ids, &[0, 1, 2]);

    let results = dataset
        .scan()
        .with_row_id()
        .full_text_search(FullTextSearchQuery::new("score".to_owned()))
        .unwrap()
        .limit(Some(2), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 2);
    let row_ids = results[ROW_ID].as_primitive::<UInt64Type>().values();
    assert_eq!(row_ids, &[0, 1]);

    let results = dataset
        .scan()
        .with_row_id()
        .full_text_search(FullTextSearchQuery::new("score".to_owned()))
        .unwrap()
        .limit(Some(1), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(results.num_rows(), 1);
    let row_ids = results[ROW_ID].as_primitive::<UInt64Type>().values();
    assert_eq!(row_ids, &[0]);
}

async fn create_fts_dataset<
    Offset: arrow::array::OffsetSizeTrait,
    ListOffset: arrow::array::OffsetSizeTrait,
>(
    is_list: bool,
    with_position: bool,
    params: InvertedIndexParams,
) -> Dataset {
    let tempdir = TempStrDir::default();
    let uri = tempdir.to_owned();
    drop(tempdir);

    let params = params.with_position(with_position);
    let doc_col: Arc<dyn Array> = if is_list {
        let string_builder = GenericStringBuilder::<Offset>::new();
        let mut list_col = GenericListBuilder::<ListOffset, _>::new(string_builder);
        // Create a list of strings
        list_col.values().append_value("lance database the search"); // for testing phrase query
        list_col.append(true);
        list_col.values().append_value("lance database"); // for testing phrase query
        list_col.append(true);
        list_col.values().append_value("lance search");
        list_col.append(true);
        list_col.values().append_value("database");
        list_col.values().append_value("search");
        list_col.append(true);
        list_col.values().append_value("unrelated doc");
        list_col.append(true);
        list_col.values().append_value("unrelated");
        list_col.append(true);
        list_col.values().append_value("mots");
        list_col.values().append_value("accentués");
        list_col.append(true);
        list_col
            .values()
            .append_value("lance database full text search");
        list_col.append(true);

        // for testing null
        list_col.append(false);

        Arc::new(list_col.finish())
    } else {
        Arc::new(GenericStringArray::<Offset>::from(vec![
            "lance database the search",
            "lance database",
            "lance search",
            "database search",
            "unrelated doc",
            "unrelated",
            "mots accentués",
            "lance database full text search",
        ]))
    };
    let ids = UInt64Array::from_iter_values(0..doc_col.len() as u64);
    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("doc", doc_col.data_type().to_owned(), true),
            arrow_schema::Field::new("id", DataType::UInt64, false),
        ])
        .into(),
        vec![Arc::new(doc_col) as ArrayRef, Arc::new(ids) as ArrayRef],
    )
    .unwrap();
    let schema = batch.schema();
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
    let mut dataset = Dataset::write(batches, &uri, None).await.unwrap();

    dataset
        .create_index(&["doc"], IndexType::Inverted, None, &params, true)
        .await
        .unwrap();

    dataset
}

async fn test_fts_index<
    Offset: arrow::array::OffsetSizeTrait,
    ListOffset: arrow::array::OffsetSizeTrait,
>(
    is_list: bool,
) {
    let ds =
        create_fts_dataset::<Offset, ListOffset>(is_list, false, InvertedIndexParams::default())
            .await;
    let result = ds
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new("lance".to_owned()).limit(Some(3)))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 3, "{:?}", result);
    let ids = result["id"].as_primitive::<UInt64Type>().values();
    assert!(ids.contains(&0), "{:?}", result);
    assert!(ids.contains(&1), "{:?}", result);
    assert!(ids.contains(&2), "{:?}", result);

    let result = ds
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new("database".to_owned()).limit(Some(3)))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 3);
    let ids = result["id"].as_primitive::<UInt64Type>().values();
    assert!(ids.contains(&0), "{:?}", result);
    assert!(ids.contains(&1), "{:?}", result);
    assert!(ids.contains(&3), "{:?}", result);

    let result = ds
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(
            FullTextSearchQuery::new_query(
                MatchQuery::new("lance database".to_owned())
                    .with_operator(Operator::And)
                    .into(),
            )
            .limit(Some(5)),
        )
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 3, "{:?}", result);
    let ids = result["id"].as_primitive::<UInt64Type>().values();
    assert!(ids.contains(&0), "{:?}", result);
    assert!(ids.contains(&1), "{:?}", result);
    assert!(ids.contains(&7), "{:?}", result);

    let result = ds
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new("unknown null".to_owned()).limit(Some(3)))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 0);

    // test phrase query
    // for non-phrasal query, the order of the tokens doesn't matter
    // so there should be 4 documents that contain "database" or "lance"

    // we built the index without position, so the phrase query will not work
    let result = ds
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(
            FullTextSearchQuery::new_query(PhraseQuery::new("lance database".to_owned()).into())
                .limit(Some(10)),
        )
        .unwrap()
        .try_into_batch()
        .await;
    let err = result.unwrap_err().to_string();
    assert!(err.contains("position is not found but required for phrase queries, try recreating the index with position"),"{}",err);

    // recreate the index with position
    let ds =
        create_fts_dataset::<Offset, ListOffset>(is_list, true, InvertedIndexParams::default())
            .await;
    let result = ds
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new("lance database".to_owned()).limit(Some(10)))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 5, "{:?}", result);
    let ids = result["id"].as_primitive::<UInt64Type>().values();
    assert!(ids.contains(&0));
    assert!(ids.contains(&1));
    assert!(ids.contains(&2));
    assert!(ids.contains(&3));
    assert!(ids.contains(&7));

    let result = ds
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(
            FullTextSearchQuery::new_query(PhraseQuery::new("lance database".to_owned()).into())
                .limit(Some(10)),
        )
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let ids = result["id"].as_primitive::<UInt64Type>().values();
    assert_eq!(result.num_rows(), 3, "{:?}", ids);
    assert!(ids.contains(&0));
    assert!(ids.contains(&1));
    assert!(ids.contains(&7));

    let result = ds
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(
            FullTextSearchQuery::new_query(PhraseQuery::new("database lance".to_owned()).into())
                .limit(Some(10)),
        )
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 0);

    let result = ds
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(
            FullTextSearchQuery::new_query(PhraseQuery::new("lance unknown".to_owned()).into())
                .limit(Some(10)),
        )
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 0);

    let result = ds
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(
            FullTextSearchQuery::new_query(PhraseQuery::new("unknown null".to_owned()).into())
                .limit(Some(3)),
        )
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 0);

    let result = ds
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(
            FullTextSearchQuery::new_query(PhraseQuery::new("lance search".to_owned()).into())
                .limit(Some(3)),
        )
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 1);

    let result = ds
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(
            FullTextSearchQuery::new_query(
                PhraseQuery::new("lance search".to_owned())
                    .with_slop(2)
                    .into(),
            )
            .limit(Some(3)),
        )
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 2);

    let result = ds
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(
            FullTextSearchQuery::new_query(
                PhraseQuery::new("search lance".to_owned())
                    .with_slop(2)
                    .into(),
            )
            .limit(Some(3)),
        )
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 0);

    let result = ds
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(
            // must contain "lance" and "database", and may contain "search"
            FullTextSearchQuery::new_query(
                BooleanQuery::new([
                    (
                        Occur::Should,
                        MatchQuery::new("search".to_owned())
                            .with_operator(Operator::And)
                            .into(),
                    ),
                    (
                        Occur::Must,
                        MatchQuery::new("lance database".to_owned())
                            .with_operator(Operator::And)
                            .into(),
                    ),
                ])
                .into(),
            )
            .limit(Some(3)),
        )
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 3, "{:?}", result);
    let ids = result["id"].as_primitive::<UInt64Type>().values();
    assert!(ids.contains(&0), "{:?}", result);
    assert!(ids.contains(&1), "{:?}", result);
    assert!(ids.contains(&7), "{:?}", result);

    let result = ds
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(
            // must contain "lance" and "database", and may contain "search"
            FullTextSearchQuery::new_query(
                BooleanQuery::new([
                    (
                        Occur::Should,
                        MatchQuery::new("search".to_owned())
                            .with_operator(Operator::And)
                            .into(),
                    ),
                    (
                        Occur::Must,
                        MatchQuery::new("lance database".to_owned())
                            .with_operator(Operator::And)
                            .into(),
                    ),
                    (
                        Occur::MustNot,
                        MatchQuery::new("full text".to_owned()).into(),
                    ),
                ])
                .into(),
            )
            .limit(Some(3)),
        )
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 2, "{:?}", result);
    let ids = result["id"].as_primitive::<UInt64Type>().values();
    assert!(ids.contains(&0), "{:?}", result);
    assert!(ids.contains(&1), "{:?}", result);
}

#[tokio::test]
async fn test_fts_index_with_string() {
    test_fts_index::<i32, i32>(false).await;
    test_fts_index::<i32, i32>(true).await;
    test_fts_index::<i32, i64>(true).await;
}

#[tokio::test]
async fn test_fts_index_with_large_string() {
    test_fts_index::<i64, i32>(false).await;
    test_fts_index::<i64, i32>(true).await;
    test_fts_index::<i64, i64>(true).await;
}

#[tokio::test]
async fn test_fts_accented_chars() {
    let ds = create_fts_dataset::<i32, i32>(false, false, InvertedIndexParams::default()).await;
    let result = ds
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new("accentués".to_owned()).limit(Some(3)))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 1);

    let result = ds
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new("accentues".to_owned()).limit(Some(3)))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 0);

    // with ascii folding enabled, the search should be accent-insensitive
    let ds = create_fts_dataset::<i32, i32>(
        false,
        false,
        InvertedIndexParams::default()
            .stem(false)
            .ascii_folding(true),
    )
    .await;
    let result = ds
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new("accentués".to_owned()).limit(Some(3)))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 1);

    let result = ds
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new("accentues".to_owned()).limit(Some(3)))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 1);
}

#[tokio::test]
async fn test_fts_phrase_query() {
    let tmpdir = TempStrDir::default();
    let uri = tmpdir.to_owned();
    drop(tmpdir);

    let words = ["lance", "full", "text", "search"];
    let mut lance_search_count = 0;
    let mut full_text_count = 0;
    let mut doc_array = (0..4096)
        .map(|_| {
            let mut rng = rand::rng();
            let mut text = String::with_capacity(512);
            let len = rng.random_range(127..512);
            for i in 0..len {
                if i > 0 {
                    text.push(' ');
                }
                text.push_str(words[rng.random_range(0..words.len())]);
            }
            if text.contains("lance search") {
                lance_search_count += 1;
            }
            if text.contains("full text") {
                full_text_count += 1;
            }
            text
        })
        .collect_vec();
    // Ensure at least one doc matches each phrase deterministically
    doc_array.push("lance search".to_owned());
    lance_search_count += 1;
    doc_array.push("full text".to_owned());
    full_text_count += 1;
    doc_array.push("position for phrase query".to_owned());

    // 1) Build index without positions and assert phrase query errors
    let params_no_pos = InvertedIndexParams::default().with_position(false);
    let doc_col: Arc<dyn Array> = Arc::new(GenericStringArray::<i32>::from(doc_array.clone()));
    let ids = UInt64Array::from_iter_values(0..doc_col.len() as u64);
    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("doc", doc_col.data_type().to_owned(), true),
            arrow_schema::Field::new("id", DataType::UInt64, false),
        ])
        .into(),
        vec![Arc::new(doc_col) as ArrayRef, Arc::new(ids) as ArrayRef],
    )
    .unwrap();
    let schema = batch.schema();
    let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
    let mut dataset = Dataset::write(batches, &uri, None).await.unwrap();
    dataset
        .create_index(&["doc"], IndexType::Inverted, None, &params_no_pos, true)
        .await
        .unwrap();

    let err = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new_query(
            PhraseQuery::new("lance search".to_owned()).into(),
        ))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("position is not found but required for phrase queries, try recreating the index with position"), "{}", err);
    assert!(err.starts_with("Invalid user input: "), "{}", err);

    // 2) Recreate index with positions and assert phrase query works
    let params_with_pos = InvertedIndexParams::default().with_position(true);
    dataset
        .create_index(&["doc"], IndexType::Inverted, None, &params_with_pos, true)
        .await
        .unwrap();

    let result = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new_query(
            PhraseQuery::new("lance search".to_owned()).into(),
        ))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), lance_search_count);

    let result = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new_query(
            PhraseQuery::new("full text".to_owned()).into(),
        ))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), full_text_count);

    let result = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new_query(
            PhraseQuery::new("phrase query".to_owned()).into(),
        ))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 1);

    let result = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new_query(
            PhraseQuery::new("".to_owned()).into(),
        ))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 0);
}

async fn prepare_json_dataset() -> (Dataset, String) {
    let text_col = Arc::new(StringArray::from(vec![
        r#"{
          "Title": "HarryPotter Chapter One",
          "Content": "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say...",
          "Author": "J.K. Rowling",
          "Price": 128,
          "Language": ["english", "chinese"]
      }"#,
        r#"{
         "Title": "Fairy Talest",
         "Content": "Once upon a time, on a bitterly cold New Year's Eve, a little girl...",
         "Author": "ANDERSEN",
         "Price": 50,
         "Language": ["english", "chinese"]
      }"#,
    ]));
    let json_col = "json_field".to_string();

    // Prepare dataset
    let mut metadata = HashMap::new();
    metadata.insert(
        ARROW_EXT_NAME_KEY.to_string(),
        ARROW_JSON_EXT_NAME.to_string(),
    );
    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![
            Field::new(&json_col, DataType::Utf8, false).with_metadata(metadata)
        ])
        .into(),
        vec![text_col.clone()],
    )
    .unwrap();
    let schema = batch.schema();
    let stream = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
    let dataset = Dataset::write(stream, "memory://test/table", None)
        .await
        .unwrap();

    (dataset, json_col)
}

#[tokio::test]
async fn test_json_inverted_fuzziness_query() {
    let (mut dataset, json_col) = prepare_json_dataset().await;

    // Create inverted index for json col
    dataset
        .create_index(
            &[&json_col],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default().lance_tokenizer("json".to_string()),
            true,
        )
        .await
        .unwrap();

    // Match query with fuzziness
    let query = FullTextSearchQuery {
        query: FtsQuery::Match(
            MatchQuery::new("Content,str,Dursley".to_string()).with_column(Some(json_col.clone())),
        ),
        limit: None,
        wand_factor: None,
    };
    let batch = dataset
        .scan()
        .full_text_search(query)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(1, batch.num_rows());

    let query = FullTextSearchQuery {
        query: FtsQuery::Match(
            MatchQuery::new("Content,str,Bursley".to_string()).with_column(Some(json_col.clone())),
        ),
        limit: None,
        wand_factor: None,
    };
    let batch = dataset
        .scan()
        .full_text_search(query)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(0, batch.num_rows());

    let query = FullTextSearchQuery {
        query: FtsQuery::Match(
            MatchQuery::new("Content,str,Bursley".to_string())
                .with_column(Some(json_col.clone()))
                .with_fuzziness(Some(1)),
        ),
        limit: None,
        wand_factor: None,
    };
    let batch = dataset
        .scan()
        .full_text_search(query)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(1, batch.num_rows());

    let query = FullTextSearchQuery {
        query: FtsQuery::Match(
            MatchQuery::new("Content,str,ABursley".to_string())
                .with_column(Some(json_col.clone()))
                .with_fuzziness(Some(1)),
        ),
        limit: None,
        wand_factor: None,
    };
    let batch = dataset
        .scan()
        .full_text_search(query)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(0, batch.num_rows());

    let query = FullTextSearchQuery {
        query: FtsQuery::Match(
            MatchQuery::new("Content,str,ABursley".to_string())
                .with_column(Some(json_col.clone()))
                .with_fuzziness(Some(2)),
        ),
        limit: None,
        wand_factor: None,
    };
    let batch = dataset
        .scan()
        .full_text_search(query)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(1, batch.num_rows());

    let query = FullTextSearchQuery {
        query: FtsQuery::Match(
            MatchQuery::new("Dontent,str,Bursley".to_string())
                .with_column(Some(json_col.clone()))
                .with_fuzziness(Some(2)),
        ),
        limit: None,
        wand_factor: None,
    };
    let batch = dataset
        .scan()
        .full_text_search(query)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(0, batch.num_rows());
}

#[tokio::test]
async fn test_json_inverted_match_query() {
    let (mut dataset, json_col) = prepare_json_dataset().await;

    // Create inverted index for json col, with max token len 10 and enable stemming,
    // lower case, and remove stop words
    dataset
        .create_index(
            &[&json_col],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default()
                .lance_tokenizer("json".to_string())
                .max_token_length(Some(10))
                .stem(true)
                .lower_case(true)
                .remove_stop_words(true),
            true,
        )
        .await
        .unwrap();

    // Match query with token length exceed max token length
    let query = FullTextSearchQuery {
        query: FtsQuery::Match(
            MatchQuery::new("Title,str,harrypotter".to_string())
                .with_column(Some(json_col.clone())),
        ),
        limit: None,
        wand_factor: None,
    };
    let batch = dataset
        .scan()
        .full_text_search(query)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(0, batch.num_rows());

    // Match query with stemming
    let query = FullTextSearchQuery {
        query: FtsQuery::Match(
            MatchQuery::new("Content,str,onc".to_string()).with_column(Some(json_col.clone())),
        ),
        limit: None,
        wand_factor: None,
    };
    let batch = dataset
        .scan()
        .full_text_search(query)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(1, batch.num_rows());

    // Match query with lower case
    let query = FullTextSearchQuery {
        query: FtsQuery::Match(
            MatchQuery::new("Content,str,DURSLEY".to_string()).with_column(Some(json_col.clone())),
        ),
        limit: None,
        wand_factor: None,
    };
    let batch = dataset
        .scan()
        .full_text_search(query)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(1, batch.num_rows());

    // Match query with stop word
    let query = FullTextSearchQuery {
        query: FtsQuery::Match(
            MatchQuery::new("Content,str,and".to_string()).with_column(Some(json_col.clone())),
        ),
        limit: None,
        wand_factor: None,
    };
    let batch = dataset
        .scan()
        .full_text_search(query)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(0, batch.num_rows());
}

#[tokio::test]
async fn test_json_inverted_flat_match_query() {
    let (mut dataset, json_col) = prepare_json_dataset().await;

    // Create inverted index for json col
    dataset
        .create_index(
            &[&json_col],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default()
                .lance_tokenizer("json".to_string())
                .stem(false),
            true,
        )
        .await
        .unwrap();

    // Append data
    let text_col = Arc::new(StringArray::from(vec![
        r#"{
          "Title": "HarryPotter Chapter Two",
          "Content": "Nearly ten years had passed since the Dursleys had woken up...",
          "Author": "J.K. Rowling",
          "Price": 128,
          "Language": ["english", "chinese"]
        }"#,
    ]));

    let mut metadata = HashMap::new();
    metadata.insert(
        ARROW_EXT_NAME_KEY.to_string(),
        ARROW_JSON_EXT_NAME.to_string(),
    );
    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![
            Field::new(&json_col, DataType::Utf8, false).with_metadata(metadata)
        ])
        .into(),
        vec![text_col.clone()],
    )
    .unwrap();
    let schema = batch.schema();
    let stream = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
    dataset.append(stream, None).await.unwrap();

    // Test match query
    let query = FullTextSearchQuery {
        query: FtsQuery::Match(
            MatchQuery::new("Title,str,harrypotter".to_string())
                .with_column(Some(json_col.clone())),
        ),
        limit: None,
        wand_factor: None,
    };
    let batch = dataset
        .scan()
        .full_text_search(query)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(2, batch.num_rows());
}

#[tokio::test]
async fn test_json_inverted_phrase_query() {
    // Prepare json dataset
    let (mut dataset, json_col) = prepare_json_dataset().await;

    // Create inverted index for json col
    dataset
        .create_index(
            &[&json_col],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default()
                .lance_tokenizer("json".to_string())
                .stem(false)
                .with_position(true),
            true,
        )
        .await
        .unwrap();

    // Test phrase query
    let query = FullTextSearchQuery {
        query: FtsQuery::Phrase(
            PhraseQuery::new("Title,str,harrypotter one chapter".to_string())
                .with_column(Some(json_col.clone())),
        ),
        limit: None,
        wand_factor: None,
    };
    let batch = dataset
        .scan()
        .full_text_search(query)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(0, batch.num_rows());

    let query = FullTextSearchQuery {
        query: FtsQuery::Phrase(
            PhraseQuery::new("Title,str,harrypotter chapter one".to_string())
                .with_column(Some(json_col.clone())),
        ),
        limit: None,
        wand_factor: None,
    };
    let batch = dataset
        .scan()
        .full_text_search(query)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(1, batch.num_rows());
}

#[tokio::test]
async fn test_json_inverted_multimatch_query() {
    // Prepare json dataset
    let (mut dataset, json_col) = prepare_json_dataset().await;

    // Create inverted index for json col
    dataset
        .create_index(
            &[&json_col],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default()
                .lance_tokenizer("json".to_string())
                .stem(false),
            true,
        )
        .await
        .unwrap();

    // Test multi match query
    let query = FullTextSearchQuery {
        query: FtsQuery::MultiMatch(MultiMatchQuery {
            match_queries: vec![
                MatchQuery::new("Title,str,harrypotter".to_string())
                    .with_column(Some(json_col.clone())),
                MatchQuery::new("Language,str,english".to_string())
                    .with_column(Some(json_col.clone())),
            ],
        }),
        limit: None,
        wand_factor: None,
    };
    let batch = dataset
        .scan()
        .full_text_search(query)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(2, batch.num_rows());
}

#[tokio::test]
async fn test_json_inverted_boolean_query() {
    // Prepare json dataset
    let (mut dataset, json_col) = prepare_json_dataset().await;

    // Create inverted index for json col
    dataset
        .create_index(
            &[&json_col],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default()
                .lance_tokenizer("json".to_string())
                .stem(false),
            true,
        )
        .await
        .unwrap();

    // Test boolean query
    let query = FullTextSearchQuery {
        query: FtsQuery::Boolean(BooleanQuery {
            should: vec![],
            must: vec![
                FtsQuery::Match(
                    MatchQuery::new("Language,str,english".to_string())
                        .with_column(Some(json_col.clone())),
                ),
                FtsQuery::Match(
                    MatchQuery::new("Title,str,harrypotter".to_string())
                        .with_column(Some(json_col.clone())),
                ),
            ],
            must_not: vec![],
        }),
        limit: None,
        wand_factor: None,
    };
    let batch = dataset
        .scan()
        .full_text_search(query)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(1, batch.num_rows());
}

#[tokio::test]
async fn test_sql_contains_tokens() {
    let text_col = Arc::new(StringArray::from(vec![
        "a cat catch a fish",
        "a fish catch a cat",
        "a white cat catch a big fish",
        "cat catchup fish",
        "cat fish catch",
    ]));

    // Prepare dataset
    let batch = RecordBatch::try_new(
        arrow_schema::Schema::new(vec![Field::new("text", DataType::Utf8, false)]).into(),
        vec![text_col.clone()],
    )
    .unwrap();
    let schema = batch.schema();
    let stream = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
    let mut dataset = Dataset::write(stream, "memory://test/table", None)
        .await
        .unwrap();

    // Test without fts index
    let results = execute_sql(
        "select * from foo where contains_tokens(text, 'cat catch fish')",
        "foo".to_string(),
        Arc::new(dataset.clone()),
    )
    .await
    .unwrap();

    assert_results(
        results,
        &StringArray::from(vec![
            "a cat catch a fish",
            "a fish catch a cat",
            "a white cat catch a big fish",
            "cat fish catch",
        ]),
    );

    // Verify plan, should not contain ScalarIndexQuery.
    let results = execute_sql(
        "explain select * from foo where contains_tokens(text, 'cat catch fish')",
        "foo".to_string(),
        Arc::new(dataset.clone()),
    )
    .await
    .unwrap();
    let plan = format!("{:?}", results);
    assert_not_contains!(&plan, "ScalarIndexQuery");

    // Test with unsuitable fts index
    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default().base_tokenizer("raw".to_string()),
            true,
        )
        .await
        .unwrap();

    let results = execute_sql(
        "select * from foo where contains_tokens(text, 'cat catch fish')",
        "foo".to_string(),
        Arc::new(dataset.clone()),
    )
    .await
    .unwrap();

    assert_results(
        results,
        &StringArray::from(vec![
            "a cat catch a fish",
            "a fish catch a cat",
            "a white cat catch a big fish",
            "cat fish catch",
        ]),
    );

    // Verify plan, should not contain ScalarIndexQuery because fts index is not unsuitable.
    let results = execute_sql(
        "explain select * from foo where contains_tokens(text, 'cat catch fish')",
        "foo".to_string(),
        Arc::new(dataset.clone()),
    )
    .await
    .unwrap();
    let plan = format!("{:?}", results);
    assert_not_contains!(&plan, "ScalarIndexQuery");

    // Test with suitable fts index
    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default()
                .max_token_length(None)
                .stem(false),
            true,
        )
        .await
        .unwrap();

    let results = execute_sql(
        "select * from foo where contains_tokens(text, 'cat catch fish')",
        "foo".to_string(),
        Arc::new(dataset.clone()),
    )
    .await
    .unwrap();

    assert_results(
        results,
        &StringArray::from(vec![
            "a cat catch a fish",
            "a fish catch a cat",
            "a white cat catch a big fish",
            "cat fish catch",
        ]),
    );

    // Verify plan, should contain ScalarIndexQuery.
    let results = execute_sql(
        "explain select * from foo where contains_tokens(text, 'cat catch fish')",
        "foo".to_string(),
        Arc::new(dataset.clone()),
    )
    .await
    .unwrap();
    let plan = format!("{:?}", results);
    assert_contains!(&plan, "ScalarIndexQuery");
}

#[tokio::test]
async fn test_index_take_batch_size() -> Result<()> {
    use tempfile::tempdir;
    let temp_dir = tempdir()?;

    let dataset_path = temp_dir.path().join("ints_dataset");
    let values: Vec<i32> = (0..1024).collect();
    let array = Int32Array::from(values);
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "ints",
        DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)])?;
    let write_params = WriteParams {
        mode: WriteMode::Create,
        max_rows_per_file: 100,
        ..Default::default()
    };
    let batch_reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    Dataset::write(
        batch_reader,
        dataset_path.to_str().unwrap(),
        Some(write_params),
    )
    .await?;
    let mut dataset = Dataset::open(dataset_path.to_str().unwrap()).await?;
    dataset
        .create_index(
            &["ints"],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await?;

    let mut scanner = dataset.scan();
    scanner.batch_size(50).filter("ints > 0")?.with_row_id();
    let batches: Vec<RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(1023, total_rows);
    assert_eq!(21, batches.len());

    let mut scanner = dataset.scan();
    scanner
        .batch_size(50)
        .filter("ints > 0")?
        .limit(Some(1024), None)?
        .with_row_id();
    let batches: Vec<RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(1023, total_rows);
    assert_eq!(21, batches.len());

    let dataset_path2 = temp_dir.path().join("strings_dataset");
    let strings: Vec<String> = (0..1024).map(|i| format!("string-{}", i)).collect();
    let string_array = StringArray::from(strings);
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "strings",
        DataType::Utf8,
        false,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(string_array)])?;
    let write_params = WriteParams {
        mode: WriteMode::Create,
        max_rows_per_file: 100,
        ..Default::default()
    };
    let batch_reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    Dataset::write(
        batch_reader,
        dataset_path2.to_str().unwrap(),
        Some(write_params),
    )
    .await?;
    let mut dataset2 = Dataset::open(dataset_path2.to_str().unwrap()).await?;
    dataset2
        .create_index(
            &["strings"],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await?;

    let mut scanner = dataset2.scan();
    scanner
        .batch_size(50)
        .filter("contains(strings, 'ing')")?
        .limit(Some(1024), None)?
        .with_row_id();
    let batches: Vec<RecordBatch> = scanner.try_into_stream().await?.try_collect().await?;
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(1024, total_rows);
    assert_eq!(21, batches.len());

    Ok(())
}

#[tokio::test]
async fn test_auto_infer_lance_tokenizer() {
    let (mut dataset, json_col) = prepare_json_dataset().await;

    // Create inverted index for json col. Expect auto-infer 'json' for lance tokenizer.
    dataset
        .create_index(
            &[&json_col],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    // Match query succeed only when lance tokenizer is 'json'
    let query = FullTextSearchQuery {
        query: FtsQuery::Match(
            MatchQuery::new("Content,str,once".to_string()).with_column(Some(json_col.clone())),
        ),
        limit: None,
        wand_factor: None,
    };
    let batch = dataset
        .scan()
        .full_text_search(query)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(1, batch.num_rows());
}
