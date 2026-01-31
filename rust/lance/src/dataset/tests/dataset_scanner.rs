// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;
use std::vec;

use crate::index::vector::VectorIndexParams;
use lance_arrow::json::{is_arrow_json_field, json_field, JsonArray};
use lance_arrow::FixedSizeListArrayExt;

use arrow::compute::concat_batches;
use arrow_array::UInt64Array;
use arrow_array::{Array, FixedSizeListArray};
use arrow_array::{Float32Array, Int32Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema, SchemaRef};
use futures::TryStreamExt;
use lance_arrow::SchemaExt;
use lance_core::cache::LanceCache;
use lance_encoding::decoder::DecoderPlugins;
use lance_file::reader::{describe_encoding, FileReader, FileReaderOptions};
use lance_file::version::LanceFileVersion;
use lance_index::scalar::inverted::{
    query::PhraseQuery, tokenizer::InvertedIndexParams, SCORE_FIELD,
};
use lance_index::scalar::FullTextSearchQuery;
use lance_index::{vector::DIST_COL, DatasetIndexExt, IndexType};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use lance_linalg::distance::MetricType;
use uuid::Uuid;

use crate::dataset::scanner::{DatasetRecordBatchStream, QueryFilter};
use crate::dataset::write::WriteParams;
use crate::Dataset;
use lance_index::scalar::inverted::query::FtsQuery;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::vector::Query;
use pretty_assertions::assert_eq;

#[tokio::test]
async fn test_vector_filter_fts_search() {
    let dataset = prepare_query_filter_dataset().await;
    let schema: ArrowSchema = dataset.schema().into();

    let query_vector = Arc::new(Float32Array::from(vec![300f32, 300f32, 300f32, 300f32]));
    let vector_query = Query {
        column: "vector".to_string(),
        key: query_vector,
        k: 5,
        lower_bound: None,
        upper_bound: None,
        minimum_nprobes: 20,
        maximum_nprobes: None,
        ef: None,
        refine_factor: None,
        metric_type: Some(MetricType::L2),
        use_index: true,
        dist_q_c: 0.0,
    };

    // Case 1: search with prefilter=true, query_filter=vector([300,300,300,300])
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new("text".to_string()))
        .unwrap()
        .prefilter(true)
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300, 299],
    )
    .await;

    // Case 2: search with prefilter=true, query_filter=vector([300,300,300,300]), filter="category='geography'"
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new("text".to_string()))
        .unwrap()
        .prefilter(true)
        .filter("category='geography'")
        .unwrap()
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300],
    )
    .await;

    // Case 3: search with prefilter=true, phrase query, query_filter=vector([300,300,300,300])
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Phrase(
            PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
        )))
        .unwrap()
        .prefilter(true)
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[299, 300],
    )
    .await;

    // Case 4: search with prefilter=true, phrase query, query_filter=vector([300,300,300,300]), filter="category='geography'"
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Phrase(
            PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
        )))
        .unwrap()
        .prefilter(true)
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .filter("category='geography'")
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300],
    )
    .await;

    // Case 5: search with prefilter=false, phrase query, query_filter=vector([300,300,300,300])
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Phrase(
            PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
        )))
        .unwrap()
        .prefilter(false)
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300, 299, 255, 254, 253],
    )
    .await;

    // Case 6: search with prefilter=false, phrase query, query_filter=vector([300,300,300,300]), filter="category='geography'"
    let mut scanner = dataset.scan();
    let stream = scanner
        .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Phrase(
            PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
        )))
        .unwrap()
        .prefilter(false)
        .filter("category='geography'")
        .unwrap()
        .filter_query(QueryFilter::Vector(vector_query.clone()))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema.try_with_column(SCORE_FIELD.clone()).unwrap().into(),
        &[300, 255],
    )
    .await;
}

#[tokio::test]
async fn test_fts_filter_vector_search() {
    let dataset = prepare_query_filter_dataset().await;
    let schema: ArrowSchema = dataset.schema().into();

    // Case 1: search with prefilter=true, query_filter=match("text")
    let query_vector = Float32Array::from(vec![300f32, 300f32, 300f32, 300f32]);
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .prefilter(true)
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new(
            "text".to_string(),
        )))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[300, 299, 255, 254, 253],
    )
    .await;

    // Case 2: search with prefilter=true, query_filter=match("text"), filter="category='geography'"
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .prefilter(true)
        .filter("category='geography'")
        .unwrap()
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new(
            "text".to_string(),
        )))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[300, 255, 252, 249, 246],
    )
    .await;

    // Case 3: search with prefilter=false, query_filter=match("text")
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .prefilter(false)
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new(
            "text".to_string(),
        )))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[300, 299],
    )
    .await;

    // Case 4: search with prefilter=false, query_filter=match("text"), filter="category='geography'"
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .prefilter(false)
        .filter("category='geography'")
        .unwrap()
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new(
            "text".to_string(),
        )))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[300],
    )
    .await;

    // Case 5: search with prefilter=false, query_filter=phrase("text")
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .prefilter(false)
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new_query(
            FtsQuery::Phrase(
                PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
            ),
        )))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[299, 300],
    )
    .await;

    // Case 6: search with prefilter=false, query_filter=phrase("text")
    let mut scanner = dataset.scan();
    let stream = scanner
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .prefilter(false)
        .filter("category='geography'")
        .unwrap()
        .filter_query(QueryFilter::Fts(FullTextSearchQuery::new_query(
            FtsQuery::Phrase(
                PhraseQuery::new("text".to_string()).with_column(Some("text".to_string())),
            ),
        )))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();
    check_results(
        stream,
        schema
            .try_with_column(ArrowField::new(DIST_COL, DataType::Float32, true))
            .unwrap()
            .into(),
        &[300],
    )
    .await;
}

#[tokio::test]
async fn test_scan_limit_offset_preserves_json_extension_metadata() {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        json_field("meta", true),
    ]));

    let json_array = JsonArray::try_from_iter((0..50).map(|i| Some(format!(r#"{{"i":{i}}}"#))))
        .unwrap()
        .into_inner();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..50)),
            Arc::new(json_array),
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let dataset = Dataset::write(reader, "memory://", None).await.unwrap();

    let mut scanner = dataset.scan();
    scanner.limit(Some(10), None).unwrap();
    let batch_no_offset = scanner.try_into_batch().await.unwrap();
    assert!(is_arrow_json_field(
        batch_no_offset.schema().field_with_name("meta").unwrap()
    ));

    let mut scanner = dataset.scan();
    scanner.limit(Some(10), Some(10)).unwrap();
    let batch_with_offset = scanner.try_into_batch().await.unwrap();
    assert!(is_arrow_json_field(
        batch_with_offset.schema().field_with_name("meta").unwrap()
    ));
    assert_eq!(batch_no_offset.schema(), batch_with_offset.schema());
}

#[tokio::test]
async fn test_scan_miniblock_dictionary_out_of_line_bitpacking_does_not_panic() {
    let rows: usize = 10_000;
    let unique_values: usize = 2_000;
    let batch_size: usize = 8_192;

    let mut field_meta = HashMap::new();
    field_meta.insert(
        "lance-encoding:structural-encoding".to_string(),
        "miniblock".to_string(),
    );
    field_meta.insert(
        "lance-encoding:dict-size-ratio".to_string(),
        "0.99".to_string(),
    );

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "d",
        DataType::UInt64,
        false,
    )
    .with_metadata(field_meta)]));

    let values = (0..rows)
        .map(|i| (i % unique_values) as u64)
        .collect::<Vec<_>>();
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(UInt64Array::from(values))]).unwrap();

    let uri = format!("memory://{}", Uuid::new_v4());
    let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema.clone());

    let write_params = WriteParams {
        data_storage_version: Some(LanceFileVersion::V2_2),
        ..WriteParams::default()
    };
    let dataset = Dataset::write(reader, &uri, Some(write_params))
        .await
        .unwrap();

    let field_id = dataset.schema().field("d").unwrap().id as u32;
    let fragment = dataset.get_fragment(0).unwrap();
    let data_file = fragment.data_file_for_field(field_id).unwrap();
    let field_pos = data_file
        .fields
        .iter()
        .position(|id| *id == field_id as i32)
        .unwrap();
    let column_idx = data_file.column_indices[field_pos] as usize;

    let file_path = dataset.data_dir().child(data_file.path.as_str());
    let scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig::max_bandwidth(&dataset.object_store),
    );
    let file_scheduler = scheduler
        .open_file(&file_path, &CachedFileSize::unknown())
        .await
        .unwrap();

    let cache = LanceCache::with_capacity(8 * 1024 * 1024);
    let file_reader = FileReader::try_open(
        file_scheduler,
        None,
        Arc::<DecoderPlugins>::default(),
        &cache,
        FileReaderOptions::default(),
    )
    .await
    .unwrap();

    let col_meta = &file_reader.metadata().column_metadatas[column_idx];
    let encoding = describe_encoding(col_meta.pages.first().unwrap());
    assert!(
        encoding.contains("OutOfLineBitpacking") && encoding.contains("dictionary"),
        "Expected a mini-block dictionary page with out-of-line bitpacking, got: {encoding}"
    );

    let mut scanner = dataset.scan();
    scanner.batch_size(batch_size);
    scanner.project(&["d"]).unwrap();

    let mut stream = scanner.try_into_stream().await.unwrap();
    let batch = stream.try_next().await.unwrap().unwrap();
    assert_eq!(batch.num_columns(), 1);
}

async fn prepare_query_filter_dataset() -> Dataset {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                4,
            ),
            true,
        ),
        ArrowField::new("text", DataType::Utf8, false),
        ArrowField::new("category", DataType::Utf8, false),
    ]));

    // Prepare dataset
    let mut vectors = vec![];
    for i in 1..=300 {
        vectors.extend(vec![i as f32; 4]);
    }

    // id 256..298 has noop, others has text
    let mut text = vec![];
    for i in 1..=255 {
        text.push(format!("text {}", i));
    }
    for i in 256..=298 {
        text.push(format!("noop {}", i));
    }
    text.extend(vec!["text 299".to_string(), "text 300".to_string()]);

    let mut category = vec![];
    for i in 1..=300 {
        if i % 3 == 1 {
            category.push("literature".to_string());
        } else if i % 3 == 2 {
            category.push("science".to_string());
        } else {
            category.push("geography".to_string());
        }
    }

    let vectors = Float32Array::from(vectors);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(1..=300)),
            Arc::new(FixedSizeListArray::try_new_from_values(vectors, 4).unwrap()),
            Arc::new(StringArray::from(text)),
            Arc::new(StringArray::from(category)),
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

    // Create index
    let params = VectorIndexParams::with_ivf_pq_params(
        MetricType::L2,
        IvfBuildParams::new(2),
        PQBuildParams::new(4, 8),
    );
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default().with_position(true),
            true,
        )
        .await
        .unwrap();

    dataset
}

async fn check_results(
    stream: DatasetRecordBatchStream,
    expected_schema: SchemaRef,
    expected_ids: &[i32],
) {
    let results = stream.try_collect::<Vec<_>>().await.unwrap();
    let batch = concat_batches(&results[0].schema(), &results).unwrap();
    assert_eq!(batch.schema(), expected_schema);

    let ids = batch
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(ids.values(), expected_ids);
}
