// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{AsArray, ListBuilder, UInt64Builder};
use arrow::datatypes;
use arrow_array::{ArrayRef, RecordBatch, StringArray, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use futures::{StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_core::{Error, Result, ROW_ID};
use roaring::RoaringBitmap;
use snafu::{location, Location};

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
            "num_docs": self.docs.row_ids.len(),
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
                let mut results = HashSet::<u64>::new();
                token_ids
                    .iter()
                    .filter_map(|token| self.invert_list.retrieve(*token))
                    .for_each(|(row_ids, _)| results.extend(row_ids));
                results.into_iter().collect_vec()
            }
            query => {
                return Err(Error::invalid_input(
                    format!("unsupported query {:?} for inverted index", query),
                    location!(),
                ))
            }
        };

        Ok(UInt64Array::from(row_ids))
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
        let mut tokenizer = tantivy::tokenizer::TextAnalyzer::builder(
            tantivy::tokenizer::SimpleTokenizer::default(),
        )
        .build();
        let mut stream = new_data.peekable();
        while let Some(batch) = stream.try_next().await? {
            let doc_col = batch.column(0).as_string::<i32>();
            let row_id_col = batch[ROW_ID].as_primitive::<datatypes::UInt64Type>();

            for (doc, row_id) in doc_col.iter().zip(row_id_col.iter()) {
                let doc = doc.unwrap();
                let row_id = row_id.unwrap();
                let mut token_stream = tokenizer.token_stream(doc);
                let mut token_cnt = 0;
                while let Some(token) = token_stream.next() {
                    let token_id = token_set.add(token.text.clone());
                    invert_list.add(token_id, row_id);
                    token_cnt += 1;
                }
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
    tokens: Vec<String>,
    ids: Vec<u32>,
    frequencies: Vec<u64>,
}

impl TokenSet {
    fn to_batch(&self) -> Result<RecordBatch> {
        let token_col = StringArray::from(self.tokens.clone());
        let token_id_col = UInt32Array::from(self.ids.clone());
        let frequency_col = UInt64Array::from(self.frequencies.clone());

        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new(TOKEN_COL, DataType::Utf8, false),
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
        let mut tokens = Vec::new();
        let mut ids = Vec::new();
        let mut frequencies = Vec::new();
        for i in 0..reader.num_batches().await {
            let batch = reader.read_record_batch(i).await?;
            let token_col = batch[TOKEN_COL].as_string::<i32>();
            let token_id_col = batch[TOKEN_ID_COL].as_primitive::<datatypes::UInt32Type>();
            let frequency_col = batch[FREQUENCY_COL].as_primitive::<datatypes::UInt64Type>();

            tokens.extend(token_col.iter().map(|v| v.unwrap().to_owned()));
            ids.extend(token_id_col.iter().map(|v| v.unwrap()));
            frequencies.extend(frequency_col.iter().map(|v| v.unwrap()));
        }

        Ok(Self {
            tokens,
            ids,
            frequencies,
        })
    }

    fn add(&mut self, token: String) -> u32 {
        let token_id = match self.get(&token) {
            Some(token_id) => token_id,
            None => self.next_id(),
        };

        // add token if it doesn't exist
        if token_id == self.next_id() {
            self.tokens.push(token);
            self.ids.push(token_id);
            self.frequencies.push(0);
        }

        self.frequencies[token_id as usize] += 1;
        token_id
    }

    fn get(&self, token: &String) -> Option<u32> {
        let pos = self.tokens.binary_search(token).ok()?;
        Some(self.ids[pos])
    }

    fn next_id(&self) -> u32 {
        self.ids.last().map(|id| id + 1).unwrap_or(0)
    }
}

// InvertedList is a mapping from token ids to row ids
// it's used to retrieve the documents that contain a token
#[derive(Debug, Clone, Default, DeepSizeOf)]
struct InvertedList {
    tokens: Vec<u32>,
    row_ids_list: Vec<Vec<u64>>,
    frequencies_list: Vec<Vec<u64>>,
}

impl InvertedList {
    fn to_batch(&self) -> Result<RecordBatch> {
        let token_id_col = UInt32Array::from(self.tokens.clone());
        let mut row_ids_col =
            ListBuilder::with_capacity(UInt64Builder::new(), self.row_ids_list.len());
        let mut frequencies_col =
            ListBuilder::with_capacity(UInt64Builder::new(), self.frequencies_list.len());

        for row_ids in &self.row_ids_list {
            let builder = row_ids_col.values();
            for row_id in row_ids {
                builder.append_value(*row_id);
            }
            row_ids_col.append(true);
        }

        for frequencies in &self.frequencies_list {
            let builder = frequencies_col.values();
            for frequency in frequencies {
                builder.append_value(*frequency);
            }
            frequencies_col.append(true);
        }

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
                Arc::new(row_ids_col.finish()) as ArrayRef,
                Arc::new(frequencies_col.finish()) as ArrayRef,
            ],
        )?;
        Ok(batch)
    }

    async fn load(reader: Arc<dyn IndexReader>) -> Result<Self> {
        let mut tokens = Vec::new();
        let mut row_ids_list = Vec::new();
        let mut frequencies_list = Vec::new();
        for i in 0..reader.num_batches().await {
            let batch = reader.read_record_batch(i).await?;
            let token_col = batch[TOKEN_ID_COL].as_primitive::<datatypes::UInt32Type>();
            let row_ids_col = batch[ROW_ID].as_list::<i32>();
            let frequencies_col = batch[FREQUENCY_COL].as_list::<i32>();

            tokens.extend(token_col.iter().map(|v| v.unwrap()));
            for value in row_ids_col.iter() {
                let value = value.unwrap();
                let row_ids = value
                    .as_primitive::<datatypes::UInt64Type>()
                    .values()
                    .iter()
                    .cloned()
                    .collect_vec();
                row_ids_list.push(row_ids);
            }
            for value in frequencies_col.iter() {
                let value = value.unwrap();
                let frequencies = value
                    .as_primitive::<datatypes::UInt64Type>()
                    .values()
                    .iter()
                    .cloned()
                    .collect_vec();
                frequencies_list.push(frequencies);
            }
        }

        Ok(Self {
            tokens,
            row_ids_list,
            frequencies_list,
        })
    }

    fn add(&mut self, token_id: u32, row_id: u64) {
        let pos = match self.tokens.binary_search(&token_id) {
            Ok(pos) => pos,
            Err(pos) => {
                self.tokens.insert(pos, token_id);
                self.row_ids_list.insert(pos, Vec::new());
                self.frequencies_list.insert(pos, Vec::new());
                pos
            }
        };

        self.row_ids_list[pos].push(row_id);
        self.frequencies_list[pos].push(1);
    }

    fn retrieve(&self, token_id: u32) -> Option<(&[u64], &[u64])> {
        let pos = self.tokens.binary_search(&token_id).ok()?;
        Some((&self.row_ids_list[pos], &self.frequencies_list[pos]))
    }

    // fn rank(&self, row_ids: &[u64]) -> Vec<(u64, u64)> {
    //     let mut ranked = Vec::with_capacity(row_ids.len());
    //     for row_id in row_ids {
    //         if let Some(pos) = self.row_ids.binary_search(row_id).ok() {
    //             ranked.push((self.row_ids[pos], self.occurrences[pos]));
    //         }
    //     }

    //     ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    //     ranked
    // }
}

// DocSet is a mapping from row ids to the number of tokens in the document
// It's used to sort the documents by the bm25 score
#[derive(Debug, Clone, Default, DeepSizeOf)]
struct DocSet {
    row_ids: Vec<u64>,
    num_tokens: Vec<u32>,
}

impl DocSet {
    fn to_batch(&self) -> Result<RecordBatch> {
        let row_id_col = UInt64Array::from(self.row_ids.clone());
        let num_tokens_col = UInt32Array::from(self.num_tokens.clone());

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
        let mut row_ids = Vec::new();
        let mut num_tokens = Vec::new();
        for i in 0..reader.num_batches().await {
            let batch = reader.read_record_batch(i).await?;
            let row_id_col = batch[ROW_ID].as_primitive::<datatypes::UInt64Type>();
            let num_tokens_col = batch[NUM_TOKEN_COL].as_primitive::<datatypes::UInt32Type>();

            row_ids.extend(row_id_col.iter().map(|v| v.unwrap()));
            num_tokens.extend(num_tokens_col.iter().map(|v| v.unwrap()));
        }

        Ok(Self {
            row_ids,
            num_tokens,
        })
    }

    fn add(&mut self, row_id: u64, num_tokens: u32) {
        self.row_ids.push(row_id);
        self.num_tokens.push(num_tokens);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, RecordBatch, StringArray, UInt64Array};
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
        let doc_col = StringArray::from(vec!["a b c", "a b", "a c", "b c"]);
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![
                arrow_schema::Field::new("doc", arrow_schema::DataType::Utf8, false),
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
            .search(&super::ScalarQuery::FullTextSearch(vec!["a".to_string()]))
            .await
            .unwrap();
        assert_eq!(row_ids.len(), 3);
        assert_eq!(row_ids.value(0), 0);
        assert_eq!(row_ids.value(1), 1);
        assert_eq!(row_ids.value(2), 2);

        let row_ids = invert_index
            .search(&super::ScalarQuery::FullTextSearch(vec!["b".to_string()]))
            .await
            .unwrap();
        assert_eq!(row_ids.len(), 3);
        assert_eq!(row_ids.value(0), 0);
        assert_eq!(row_ids.value(1), 1);
        assert_eq!(row_ids.value(2), 3);
    }
}
