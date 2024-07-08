// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{AsArray, LargeStringBuilder, ListBuilder, UInt32Builder, UInt64Builder};
use arrow::datatypes::{self, UInt64Type};
use arrow_array::{ArrayRef, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use futures::{StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_core::{Error, Result, ROW_ID};
use roaring::RoaringBitmap;
use snafu::{location, Location};
use tantivy::tokenizer::TokenFilter;

use crate::vector::graph::OrderedFloat;
use crate::Index;

use super::{IndexReader, IndexStore, ScalarIndex, ScalarQuery};

const TOKENS_FILE: &str = "tokens.lance";
const INVERT_LIST_FILE: &str = "invert.lance";
const DOCS_FILE: &str = "docs.lance";

const TOKEN_COL: &str = "_token";
const TOKEN_ID_COL: &str = "_token_id";
const FREQUENCY_COL: &str = "_frequency";
const NUM_TOKEN_COL: &str = "_num_tokens";

#[derive(Debug, Clone, Default, DeepSizeOf)]
pub struct InvertedIndex {
    tokens: TokenSet,
    invert_list: InvertedList,
    docs: DocSet,
}

impl InvertedIndex {
    // map tokens to token ids
    // ignore tokens that are not in the index cause they won't contribute to the search
    fn map(&self, texts: &[String]) -> Vec<u32> {
        texts
            .iter()
            .filter_map(|text| self.tokens.get(text))
            .collect()
    }

    // search the documents that contain the query
    // return the row ids of the documents sorted by bm25 score
    fn bm25_search(&self, token_ids: Vec<u32>) -> Vec<(u64, f32)> {
        const K1: f32 = 1.2;
        const B: f32 = 0.75;

        let avgdl = self.docs.average_length();
        let mut bm25 = HashMap::new();

        token_ids
            .into_iter()
            .filter_map(|token| self.invert_list.retrieve(token))
            .for_each(|row_freq| {
                // TODO: this can be optimized by parallelizing the calculation
                row_freq.iter().for_each(|(row_id, freq)| {
                    let row_id = *row_id;
                    let freq = *freq as f32;
                    let bm25 = bm25.entry(row_id).or_insert(0.0);
                    *bm25 += self.idf(row_freq.len()) * freq * (K1 + 1.0)
                        / (freq + K1 * (1.0 - B + B * self.docs.num_tokens(row_id) as f32 / avgdl));
                });
            });

        bm25.into_iter()
            .sorted_unstable_by_key(|r| OrderedFloat(-r.1))
            .collect_vec()
    }

    #[inline]
    fn idf(&self, nq: usize) -> f32 {
        let num_docs = self.docs.len() as f32;
        ((num_docs - nq as f32 + 0.5) / (nq as f32 + 0.5) + 1.0).ln()
    }
}

#[async_trait]
impl Index for InvertedIndex {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(Error::invalid_input(
            "inverted index cannot be cast to vector index",
            location!(),
        ))
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "num_tokens": self.tokens.tokens.len(),
            "num_docs": self.docs.token_count.len(),
        }))
    }

    fn index_type(&self) -> crate::IndexType {
        crate::IndexType::Scalar
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!()
    }
}

#[async_trait]
impl ScalarIndex for InvertedIndex {
    // return the row ids of the documents that contain the query
    async fn search(&self, query: &ScalarQuery) -> Result<UInt64Array> {
        let row_ids = match query {
            ScalarQuery::FullTextSearch(tokens) => {
                let token_ids = self.map(tokens);
                self.bm25_search(token_ids)
                    .into_iter()
                    .map(|(row_id, _)| row_id)
            }
            query => {
                return Err(Error::invalid_input(
                    format!("unsupported query {:?} for inverted index", query),
                    location!(),
                ))
            }
        };

        // sort the row ids (documents) by bm25 score

        Ok(UInt64Array::from_iter_values(row_ids))
    }

    async fn load(store: Arc<dyn IndexStore>) -> Result<Arc<Self>>
    where
        Self: Sized,
    {
        let token_reader = store.open_index_file(TOKENS_FILE).await?;
        let invert_list_reader = store.open_index_file(INVERT_LIST_FILE).await?;
        let docs_reader = store.open_index_file(DOCS_FILE).await?;

        let tokens = TokenSet::load(token_reader).await?;
        let invert_list = InvertedList::load(invert_list_reader).await?;
        let docs = DocSet::load(docs_reader).await?;

        Ok(Arc::new(Self {
            tokens,
            invert_list,
            docs,
        }))
    }

    async fn remap(
        &self,
        _mapping: &HashMap<u64, Option<u64>>,
        _dest_store: &dyn IndexStore,
    ) -> Result<()> {
        unimplemented!()
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        let mut token_set = self.tokens.clone();
        let mut invert_list = self.invert_list.clone();
        let mut docs = self.docs.clone();
        let stopword_filter =
            tantivy::tokenizer::StopWordFilter::new(tantivy::tokenizer::Language::English).unwrap();
        let mut tokenizer = tantivy::tokenizer::TextAnalyzer::builder(
            stopword_filter.transform(tantivy::tokenizer::SimpleTokenizer::default()),
        )
        .build();
        let mut stream = new_data.peekable();
        while let Some(batch) = stream.try_next().await? {
            let doc_col = batch.column(0).as_string::<i64>();
            let row_id_col = batch[ROW_ID].as_primitive::<datatypes::UInt64Type>();

            for (doc, row_id) in doc_col.iter().zip(row_id_col.iter()) {
                let doc = doc.unwrap();
                let row_id = row_id.unwrap();
                let mut token_stream = tokenizer.token_stream(doc);
                let mut row_token_cnt = HashMap::new();
                let mut token_cnt = 0;
                while let Some(token) = token_stream.next() {
                    let token_id = token_set.add(token.text.to_owned());
                    row_token_cnt
                        .entry(token_id)
                        .and_modify(|cnt| *cnt += 1)
                        .or_insert(1);
                    token_cnt += 1;
                }
                invert_list.add(row_token_cnt, row_id);
                docs.add(row_id, token_cnt);
            }
        }

        let token_set_batch = token_set.to_batch()?;
        let mut token_set_writer = dest_store
            .new_index_file(TOKENS_FILE, token_set_batch.schema())
            .await?;
        token_set_writer.write_record_batch(token_set_batch).await?;
        token_set_writer.finish().await?;

        let invert_list_batch = invert_list.to_batch()?;
        let mut invert_list_writer = dest_store
            .new_index_file(INVERT_LIST_FILE, invert_list_batch.schema())
            .await?;
        invert_list_writer
            .write_record_batch(invert_list_batch)
            .await?;
        invert_list_writer.finish().await?;

        let docs_batch = docs.to_batch()?;
        let mut docs_writer = dest_store
            .new_index_file(DOCS_FILE, docs_batch.schema())
            .await?;
        docs_writer.write_record_batch(docs_batch).await?;
        docs_writer.finish().await?;

        Ok(())
    }
}

// TokenSet is a mapping from tokens to token ids
// it also records the frequency of each token
#[derive(Debug, Clone, Default, DeepSizeOf)]
struct TokenSet {
    // token -> (token_id, frequency)
    tokens: HashMap<String, (u32, u64)>,
    next_id: u32,
}

impl TokenSet {
    fn to_batch(&self) -> Result<RecordBatch> {
        let mut tokens_builder = LargeStringBuilder::with_capacity(self.tokens.len(), 32);
        let mut token_id_builder = UInt32Builder::with_capacity(self.tokens.len());
        let mut frequency_builder = UInt64Builder::with_capacity(self.tokens.len());
        self.tokens
            .iter()
            .for_each(|(token, (token_id, frequency))| {
                tokens_builder.append_value(token);
                token_id_builder.append_value(*token_id);
                frequency_builder.append_value(*frequency);
            });
        let token_col = tokens_builder.finish();
        let token_id_col = token_id_builder.finish();
        let frequency_col = frequency_builder.finish();

        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new(TOKEN_COL, DataType::LargeUtf8, false),
            arrow_schema::Field::new(TOKEN_ID_COL, DataType::UInt32, false),
            arrow_schema::Field::new(FREQUENCY_COL, DataType::UInt64, false),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(token_col) as ArrayRef,
                Arc::new(token_id_col) as ArrayRef,
                Arc::new(frequency_col) as ArrayRef,
            ],
        )?;
        Ok(batch)
    }

    async fn load(reader: Arc<dyn IndexReader>) -> Result<Self> {
        let mut tokens = HashMap::new();
        let mut next_id = 0;
        for i in 0..reader.num_batches().await {
            let batch = reader.read_record_batch(i).await?;
            let token_col = batch[TOKEN_COL].as_string::<i64>();
            let token_id_col = batch[TOKEN_ID_COL].as_primitive::<datatypes::UInt32Type>();
            let frequency_col = batch[FREQUENCY_COL].as_primitive::<datatypes::UInt64Type>();

            for ((token, token_id), frequency) in token_col
                .iter()
                .zip(token_id_col.iter())
                .zip(frequency_col.iter())
            {
                let token = token.unwrap();
                let token_id = token_id.unwrap();
                let frequency = frequency.unwrap();
                tokens.insert(token.to_owned(), (token_id, frequency));
                next_id = next_id.max(token_id + 1);
            }
        }

        Ok(Self { tokens, next_id })
    }

    fn add(&mut self, token: String) -> u32 {
        let next_id = self.next_id();
        let token_id = self
            .tokens
            .entry(token)
            .and_modify(|(_, freq)| *freq += 1)
            .or_insert((next_id, 1))
            .0;

        // add token if it doesn't exist
        if token_id == next_id {
            self.next_id += 1;
        }

        token_id
    }

    fn get(&self, token: &str) -> Option<u32> {
        self.tokens.get(token).map(|(token_id, _)| *token_id)
    }

    fn next_id(&self) -> u32 {
        self.next_id
    }
}

// InvertedList is a mapping from token ids to row ids
// it's used to retrieve the documents that contain a token
#[derive(Debug, Clone, Default, DeepSizeOf)]
struct InvertedList {
    inverted_list: HashMap<u32, Vec<(u64, u64)>>,
    // tokens: Vec<u32>,
    // row_ids_list: Vec<Vec<u64>>,
    // frequencies_list: Vec<Vec<u64>>,
}

impl InvertedList {
    fn to_batch(&self) -> Result<RecordBatch> {
        let mut token_id_builder = UInt32Builder::with_capacity(self.inverted_list.len());
        let mut row_ids_list_builder =
            ListBuilder::with_capacity(UInt64Builder::new(), self.inverted_list.len());
        let mut frequencies_list_builder =
            ListBuilder::with_capacity(UInt64Builder::new(), self.inverted_list.len());

        for (token_id, list) in &self.inverted_list {
            token_id_builder.append_value(*token_id);
            let row_ids_builder = row_ids_list_builder.values();
            let frequencies_builder = frequencies_list_builder.values();
            for (row_id, frequency) in list {
                row_ids_builder.append_value(*row_id);
                frequencies_builder.append_value(*frequency);
            }
            row_ids_list_builder.append(true);
            frequencies_list_builder.append(true);
        }

        let token_id_col = token_id_builder.finish();
        let row_ids_col = row_ids_list_builder.finish();
        let frequencies_col = frequencies_list_builder.finish();

        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new(TOKEN_ID_COL, DataType::UInt32, false),
            arrow_schema::Field::new(
                ROW_ID,
                DataType::List(Field::new_list_field(DataType::UInt64, true).into()),
                false,
            ),
            arrow_schema::Field::new(
                FREQUENCY_COL,
                DataType::List(Field::new_list_field(DataType::UInt64, true).into()),
                false,
            ),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(token_id_col) as ArrayRef,
                Arc::new(row_ids_col) as ArrayRef,
                Arc::new(frequencies_col) as ArrayRef,
            ],
        )?;
        Ok(batch)
    }

    async fn load(reader: Arc<dyn IndexReader>) -> Result<Self> {
        let mut inverted_list = HashMap::new();
        for i in 0..reader.num_batches().await {
            let batch = reader.read_record_batch(i).await?;
            let token_col = batch[TOKEN_ID_COL].as_primitive::<datatypes::UInt32Type>();
            let row_ids_col = batch[ROW_ID].as_list::<i32>();
            let frequencies_col = batch[FREQUENCY_COL].as_list::<i32>();

            for ((token_id, row_ids), frequencies) in token_col
                .iter()
                .zip(row_ids_col.iter())
                .zip(frequencies_col.iter())
            {
                let token_id = token_id.unwrap();
                let row_ids = row_ids.unwrap();
                let frequencies = frequencies.unwrap();
                let row_ids = row_ids.as_primitive::<UInt64Type>().values();
                let frequencies = frequencies.as_primitive::<UInt64Type>().values();
                let list = row_ids
                    .iter()
                    .cloned()
                    .zip(frequencies.iter().cloned())
                    .collect_vec();
                inverted_list.insert(token_id, list);
            }
        }

        Ok(Self { inverted_list })
    }

    // for efficiency, we don't check if the row_id exists
    // we assume that the row_id is unique and doesn't exist in the list
    fn add(&mut self, token_cnt: HashMap<u32, u64>, row_id: u64) {
        for (token_id, freq) in token_cnt {
            let list = self.inverted_list.entry(token_id).or_default();
            list.push((row_id, freq));
        }
    }

    fn retrieve(&self, token_id: u32) -> Option<&[(u64, u64)]> {
        self.inverted_list
            .get(&token_id)
            .map(|list| list.as_slice())
    }
}

// DocSet is a mapping from row ids to the number of tokens in the document
// It's used to sort the documents by the bm25 score
#[derive(Debug, Clone, Default, DeepSizeOf)]
struct DocSet {
    // row id -> num tokens
    token_count: HashMap<u64, u32>,
    // row_ids: Vec<u64>,
    // num_tokens: Vec<u32>,
    total_tokens: u64,
}

impl DocSet {
    fn len(&self) -> usize {
        self.token_count.len()
    }

    fn average_length(&self) -> f32 {
        self.total_tokens as f32 / self.token_count.len() as f32
    }

    fn to_batch(&self) -> Result<RecordBatch> {
        let row_id_col = UInt64Array::from_iter_values(self.token_count.keys().cloned());
        let num_tokens_col = UInt32Array::from_iter_values(self.token_count.values().cloned());

        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new(ROW_ID, DataType::UInt64, false),
            arrow_schema::Field::new(NUM_TOKEN_COL, DataType::UInt32, false),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(row_id_col) as ArrayRef,
                Arc::new(num_tokens_col) as ArrayRef,
            ],
        )?;
        Ok(batch)
    }

    async fn load(reader: Arc<dyn IndexReader>) -> Result<Self> {
        let mut token_count = HashMap::new();
        let mut total_tokens = 0;
        for i in 0..reader.num_batches().await {
            let batch = reader.read_record_batch(i).await?;
            let row_id_col = batch[ROW_ID].as_primitive::<datatypes::UInt64Type>();
            let num_tokens_col = batch[NUM_TOKEN_COL].as_primitive::<datatypes::UInt32Type>();

            for (row_id, num_tokens) in row_id_col.iter().zip(num_tokens_col.iter()) {
                let row_id = row_id.unwrap();
                let num_tokens = num_tokens.unwrap();
                token_count.insert(row_id, num_tokens);
                total_tokens += num_tokens as u64;
            }
        }

        Ok(Self {
            token_count,
            total_tokens,
        })
    }

    fn num_tokens(&self, row_id: u64) -> u32 {
        self.token_count.get(&row_id).cloned().unwrap_or_default()
    }

    fn add(&mut self, row_id: u64, num_tokens: u32) {
        self.token_count.insert(row_id, num_tokens);
        self.total_tokens += num_tokens as u64;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, LargeStringArray, RecordBatch, UInt64Array};
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream;
    use lance_io::object_store::ObjectStore;
    use object_store::path::Path;

    use crate::scalar::lance_format::LanceIndexStore;
    use crate::scalar::ScalarIndex;

    #[tokio::test]
    async fn test_inverted_index() {
        let tempdir = tempfile::tempdir().unwrap();
        let index_dir = Path::from_filesystem_path(tempdir.path()).unwrap();
        let store = LanceIndexStore::new(ObjectStore::local(), index_dir, None);

        let invert_index = super::InvertedIndex::default();
        let row_id_col = UInt64Array::from(vec![0, 1, 2, 3]);
        let doc_col = LargeStringArray::from(vec![
            "lance database search",
            "lance database",
            "lance search",
            "database search",
        ]);
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![
                arrow_schema::Field::new("doc", arrow_schema::DataType::LargeUtf8, false),
                arrow_schema::Field::new(super::ROW_ID, arrow_schema::DataType::UInt64, false),
            ])
            .into(),
            vec![
                Arc::new(doc_col) as ArrayRef,
                Arc::new(row_id_col) as ArrayRef,
            ],
        )
        .unwrap();
        let stream = RecordBatchStreamAdapter::new(batch.schema(), stream::iter(vec![Ok(batch)]));
        let stream = Box::pin(stream);

        invert_index
            .update(stream, &store)
            .await
            .expect("failed to update invert index");

        let invert_index = super::InvertedIndex::load(Arc::new(store)).await.unwrap();
        let row_ids = invert_index
            .search(&super::ScalarQuery::FullTextSearch(vec![
                "lance".to_string()
            ]))
            .await
            .unwrap();
        assert_eq!(row_ids.len(), 3);
        assert!(row_ids.values().contains(&0));
        assert!(row_ids.values().contains(&1));
        assert!(row_ids.values().contains(&2));

        let row_ids = invert_index
            .search(&super::ScalarQuery::FullTextSearch(vec![
                "database".to_string()
            ]))
            .await
            .unwrap();
        assert_eq!(row_ids.len(), 3);
        assert!(row_ids.values().contains(&0));
        assert!(row_ids.values().contains(&1));
        assert!(row_ids.values().contains(&3));
    }
}
