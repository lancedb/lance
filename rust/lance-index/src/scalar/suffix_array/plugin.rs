// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! SuffixArrayIndexPlugin implementing ScalarIndexPlugin.
//!
//! Handles training (building) and loading of suffix array indices.

use std::sync::Arc;

use arrow_array::Array;
use arrow_schema::{DataType, Field};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use futures::StreamExt;
use lance_core::{Error, Result};
use prost::Message;
use tracing::info;

use crate::frag_reuse::FragReuseIndex;
use crate::pb;
use crate::progress::IndexBuildProgress;
use crate::scalar::expression::ScalarQueryParser;
use crate::scalar::registry::{
    DefaultTrainingRequest, ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
};
use crate::scalar::{CreatedIndex, IndexStore, ScalarIndex};

use super::builder::{build_suffix_array, compact_suffix_array, compute_pointer_width};
use super::index::SuffixArrayIndex;

const SUFFIX_ARRAY_INDEX_VERSION: u32 = 0;

/// Plugin for creating and loading suffix array indices.
#[derive(Debug, Default)]
pub struct SuffixArrayIndexPlugin;

#[async_trait]
impl ScalarIndexPlugin for SuffixArrayIndexPlugin {
    fn name(&self) -> &str {
        "SuffixArray"
    }

    fn new_training_request(
        &self,
        _params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        // Suffix array index supports binary, utf8, and large utf8 columns
        match field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => {}
            other => {
                return Err(Error::invalid_input(format!(
                    "Suffix array index requires Utf8, LargeUtf8, Binary, or LargeBinary column, got {other:?}"
                )));
            }
        }
        Ok(Box::new(DefaultTrainingRequest::new(
            TrainingCriteria::new(TrainingOrdering::None),
        )))
    }

    fn provides_exact_answer(&self) -> bool {
        true
    }

    fn version(&self) -> u32 {
        SUFFIX_ARRAY_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        _index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        // Suffix array queries are dispatched programmatically, not via SQL parsing
        None
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        _request: Box<dyn TrainingRequest>,
        _fragment_ids: Option<Vec<u32>>,
        _progress: Arc<dyn IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        // Phase 1: Read all data from the stream and concatenate into a single byte buffer
        let schema = data.schema();
        let value_col_idx = schema
            .fields()
            .iter()
            .position(|f| f.name() == "value")
            .ok_or_else(|| Error::invalid_input("Training data stream missing 'value' column"))?;

        let mut all_bytes: Vec<u8> = Vec::new();
        let mut total_documents: u64 = 0;

        let mut stream = data;
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            let col = batch.column(value_col_idx);

            // Extract bytes from the column based on its type
            match col.data_type() {
                DataType::Utf8 => {
                    let arr = col
                        .as_any()
                        .downcast_ref::<arrow_array::StringArray>()
                        .ok_or_else(|| Error::invalid_input("Failed to downcast to StringArray"))?;
                    for i in 0..arr.len() {
                        if !arr.is_null(i) {
                            all_bytes.extend_from_slice(arr.value(i).as_bytes());
                            total_documents += 1;
                        }
                    }
                }
                DataType::LargeUtf8 => {
                    let arr = col
                        .as_any()
                        .downcast_ref::<arrow_array::LargeStringArray>()
                        .ok_or_else(|| {
                            Error::invalid_input("Failed to downcast to LargeStringArray")
                        })?;
                    for i in 0..arr.len() {
                        if !arr.is_null(i) {
                            all_bytes.extend_from_slice(arr.value(i).as_bytes());
                            total_documents += 1;
                        }
                    }
                }
                DataType::Binary => {
                    let arr = col
                        .as_any()
                        .downcast_ref::<arrow_array::BinaryArray>()
                        .ok_or_else(|| Error::invalid_input("Failed to downcast to BinaryArray"))?;
                    for i in 0..arr.len() {
                        if !arr.is_null(i) {
                            all_bytes.extend_from_slice(arr.value(i));
                            total_documents += 1;
                        }
                    }
                }
                DataType::LargeBinary => {
                    let arr = col
                        .as_any()
                        .downcast_ref::<arrow_array::LargeBinaryArray>()
                        .ok_or_else(|| {
                            Error::invalid_input("Failed to downcast to LargeBinaryArray")
                        })?;
                    for i in 0..arr.len() {
                        if !arr.is_null(i) {
                            all_bytes.extend_from_slice(arr.value(i));
                            total_documents += 1;
                        }
                    }
                }
                other => {
                    return Err(Error::invalid_input(format!(
                        "Unsupported column type for suffix array index: {other:?}"
                    )));
                }
            }
        }

        let corpus_bytes = all_bytes.len() as u64;
        info!(
            corpus_bytes = corpus_bytes,
            total_documents = total_documents,
            "Building suffix array"
        );

        // Phase 2: Build the suffix array
        let sa = build_suffix_array(&all_bytes);
        let pointer_width = compute_pointer_width(corpus_bytes);

        // Phase 3: Compact the suffix array
        let compact_sa = compact_suffix_array(&sa, pointer_width)?;

        // Phase 4: Write index files to the store as single-row Binary record batches
        let tokenized_schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "data",
            arrow_schema::DataType::LargeBinary,
            false,
        )]));
        let tokenized_batch = arrow_array::RecordBatch::try_new(
            tokenized_schema.clone(),
            vec![Arc::new(arrow_array::LargeBinaryArray::from(vec![
                all_bytes.as_slice(),
            ]))],
        )?;
        let mut writer = index_store
            .new_index_file("tokenized.bin", tokenized_schema.clone())
            .await?;
        writer.write_record_batch(tokenized_batch).await?;
        writer.finish().await?;

        let sa_schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "data",
            arrow_schema::DataType::LargeBinary,
            false,
        )]));
        let sa_batch = arrow_array::RecordBatch::try_new(
            sa_schema.clone(),
            vec![Arc::new(arrow_array::LargeBinaryArray::from(vec![
                compact_sa.as_slice(),
            ]))],
        )?;
        let mut writer = index_store
            .new_index_file("suffix_array.bin", sa_schema)
            .await?;
        writer.write_record_batch(sa_batch).await?;
        writer.finish().await?;

        // Build protobuf details
        let details = pb::SuffixArrayIndexDetails {
            token_width: 1, // byte-level indexing
            pointer_width: pointer_width as u32,
            total_tokens: corpus_bytes, // 1 byte per token for byte-level
            total_documents,
            corpus_bytes,
            tokenizer_name: None,
            vocab_size: None,
            separator_token_id: None,
        };

        let index_details = prost_types::Any::from_msg(&details)?;

        Ok(CreatedIndex {
            index_details,
            index_version: SUFFIX_ARRAY_INDEX_VERSION,
            files: Some(index_store.list_files_with_sizes().await?),
        })
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: &lance_core::cache::LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        let details = pb::SuffixArrayIndexDetails::decode(&*index_details.value)?;

        let pointer_width = details.pointer_width as u8;
        let total_entries = details.total_tokens;

        Ok(SuffixArrayIndex::load(
            index_store,
            frag_reuse_index,
            cache,
            pointer_width,
            total_entries,
        )
        .await? as Arc<dyn ScalarIndex>)
    }
}

#[cfg(test)]
mod tests {
    use crate::scalar::suffix_array::builder::{
        build_suffix_array, compact_suffix_array, compute_pointer_width,
    };
    use crate::scalar::suffix_array::query;

    #[test]
    fn test_sa_construction_small() {
        let data = b"abcabc";
        let sa = build_suffix_array(data);
        assert_eq!(sa.len(), data.len());

        // Verify the suffix array is correctly sorted
        for i in 1..sa.len() {
            let s1 = &data[sa[i - 1] as usize..];
            let s2 = &data[sa[i] as usize..];
            assert!(s1 < s2, "SA not sorted at position {i}: {s1:?} >= {s2:?}");
        }
    }

    #[test]
    fn test_binary_search_correctness() {
        let data = b"mississippi";
        let sa = build_suffix_array(data);
        let ptr_width = compute_pointer_width(data.len() as u64);
        let compact = compact_suffix_array(&sa, ptr_width).unwrap();
        let n = sa.len() as u64;

        // "issi" appears 2 times
        assert_eq!(query::count(data, &compact, ptr_width, n, b"issi"), 2);
        // "ss" appears 2 times
        assert_eq!(query::count(data, &compact, ptr_width, n, b"ss"), 2);
        // "mississippi" appears 1 time
        assert_eq!(
            query::count(data, &compact, ptr_width, n, b"mississippi"),
            1
        );
        // "z" appears 0 times
        assert_eq!(query::count(data, &compact, ptr_width, n, b"z"), 0);
    }

    #[test]
    fn test_count_returns_correct_counts() {
        let data = b"the cat sat on the mat";
        let sa = build_suffix_array(data);
        let ptr_width = compute_pointer_width(data.len() as u64);
        let compact = compact_suffix_array(&sa, ptr_width).unwrap();
        let n = sa.len() as u64;

        // "the" appears 2 times
        assert_eq!(query::count(data, &compact, ptr_width, n, b"the"), 2);
        // "at" appears 3 times (cat, sat, mat)
        assert_eq!(query::count(data, &compact, ptr_width, n, b"at"), 3);
        // " " (space) appears 5 times
        assert_eq!(query::count(data, &compact, ptr_width, n, b" "), 5);
    }

    #[test]
    fn test_empty_data() {
        let data = b"";
        let sa = build_suffix_array(data);
        assert!(sa.is_empty());

        let ptr_width = compute_pointer_width(0);
        let compact = compact_suffix_array(&sa, ptr_width).unwrap();
        assert_eq!(query::count(data, &compact, ptr_width, 0, b"anything"), 0);
    }

    #[test]
    fn test_single_character_data() {
        let data = b"x";
        let sa = build_suffix_array(data);
        let ptr_width = compute_pointer_width(data.len() as u64);
        let compact = compact_suffix_array(&sa, ptr_width).unwrap();
        let n = sa.len() as u64;

        assert_eq!(query::count(data, &compact, ptr_width, n, b"x"), 1);
        assert_eq!(query::count(data, &compact, ptr_width, n, b"y"), 0);
        assert_eq!(query::count(data, &compact, ptr_width, n, b"xx"), 0);
    }

    #[test_log::test(tokio::test)]
    async fn test_train_load_query_roundtrip() {
        use std::sync::Arc;

        use arrow_array::{RecordBatch, StringArray, UInt64Array};
        use arrow_schema::{DataType, Field, Schema};
        use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
        use futures::stream;
        use lance_core::cache::LanceCache;
        use lance_core::utils::tempfile::TempDir;
        use lance_io::object_store::ObjectStore;

        use crate::metrics::NoOpMetricsCollector;
        use crate::progress::NoopIndexBuildProgress;
        use crate::scalar::lance_format::LanceIndexStore;
        use crate::scalar::registry::{ScalarIndexPlugin, VALUE_COLUMN_NAME};

        use super::super::query::SuffixArrayQuery;
        use super::SuffixArrayIndexPlugin;

        // Build test data: a few sentences concatenated
        let texts = vec![
            "the quick brown fox",
            "jumps over the lazy dog",
            "the cat sat on the mat",
        ];
        let text_array = StringArray::from(texts.clone());
        let row_ids = UInt64Array::from_iter_values(0..3u64);
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Utf8, false),
            Field::new("_rowid", DataType::UInt64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(text_array), Arc::new(row_ids)],
        )
        .unwrap();

        let data_stream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::once(std::future::ready(Ok(batch))),
        ));

        // Phase 1: Train the index
        let plugin = SuffixArrayIndexPlugin;
        let field = Field::new(VALUE_COLUMN_NAME, DataType::Utf8, false);
        let request = plugin.new_training_request("", &field).unwrap();

        let tmpdir = Arc::new(TempDir::default());
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ));

        let progress = Arc::new(NoopIndexBuildProgress);
        let created = plugin
            .train_index(data_stream, store.as_ref(), request, None, progress)
            .await
            .unwrap();

        // Verify the created index has the expected files
        assert!(created.files.is_some());
        let files = created.files.as_ref().unwrap();
        assert!(files.iter().any(|f| f.path.contains("tokenized")));
        assert!(files.iter().any(|f| f.path.contains("suffix_array")));

        // Phase 2: Load the index
        let loaded_index = plugin
            .load_index(
                store.clone(),
                &created.index_details,
                None,
                &LanceCache::no_cache(),
            )
            .await
            .unwrap();

        // Phase 3: Query the loaded index
        let metrics = NoOpMetricsCollector;

        // Count "the" — appears in all three texts (3 occurrences total in raw bytes)
        let count_query = SuffixArrayQuery::Count {
            query_bytes: b"the".to_vec(),
        };
        let result = loaded_index.search(&count_query, &metrics).await.unwrap();
        // "the" should appear 4 times: "the quick", "the lazy", "the cat", "the mat"
        match &result {
            crate::scalar::SearchResult::Exact(row_set) => {
                // The count is encoded as a synthetic row address in the result
                let row_addrs: Vec<u64> = row_set
                    .true_rows()
                    .row_addrs()
                    .unwrap()
                    .map(|addr| addr.into())
                    .collect();
                assert!(!row_addrs.is_empty(), "Expected count > 0 for 'the' query");
            }
            other => panic!("Expected Exact result, got {:?}", other),
        }

        // Count "xyz" — should not appear
        let count_query = SuffixArrayQuery::Count {
            query_bytes: b"xyz".to_vec(),
        };
        let result = loaded_index.search(&count_query, &metrics).await.unwrap();
        match &result {
            crate::scalar::SearchResult::Exact(row_set) => {
                let row_addrs: Vec<u64> = row_set
                    .true_rows()
                    .row_addrs()
                    .unwrap()
                    .map(|addr| addr.into())
                    .collect();
                assert!(row_addrs.is_empty(), "Expected count = 0 for 'xyz' query");
            }
            other => panic!("Expected Exact result, got {:?}", other),
        }

        // Search for positions of "fox"
        let search_query = SuffixArrayQuery::Search {
            query_bytes: b"fox".to_vec(),
            max_results: 10,
        };
        let result = loaded_index.search(&search_query, &metrics).await.unwrap();
        match &result {
            crate::scalar::SearchResult::Exact(row_set) => {
                let row_addrs: Vec<u64> = row_set
                    .true_rows()
                    .row_addrs()
                    .unwrap()
                    .map(|addr| addr.into())
                    .collect();
                assert_eq!(
                    row_addrs.len(),
                    1,
                    "Expected exactly 1 match for 'fox', got {:?}",
                    row_addrs
                );
            }
            other => panic!("Expected Exact result, got {:?}", other),
        }
    }

    #[test]
    fn test_plugin_rejects_invalid_field_type() {
        use crate::scalar::registry::ScalarIndexPlugin;
        use arrow_schema::{DataType, Field};

        let plugin = super::SuffixArrayIndexPlugin;
        let field = Field::new("value", DataType::Int32, false);
        let result = plugin.new_training_request("", &field);
        assert!(result.is_err(), "Int32 field should be rejected");
    }

    #[test]
    fn test_plugin_accepts_valid_field_types() {
        use crate::scalar::registry::ScalarIndexPlugin;
        use arrow_schema::{DataType, Field};

        let plugin = super::SuffixArrayIndexPlugin;
        for dt in &[
            DataType::Utf8,
            DataType::LargeUtf8,
            DataType::Binary,
            DataType::LargeBinary,
        ] {
            let field = Field::new("value", dt.clone(), false);
            assert!(
                plugin.new_training_request("", &field).is_ok(),
                "Should accept {:?}",
                dt
            );
        }
    }
}
