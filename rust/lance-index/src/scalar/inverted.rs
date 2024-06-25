// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes;
use arrow_array::UInt64Array;
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use futures::StreamExt;
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
const OCCURENCY_COL: &str = "_occurency";
const NUM_TOKEN_COL: &str = "_num_tokens";

#[derive(Debug, Clone, DeepSizeOf)]
pub struct InvertedIndex {
    tokens: TokenSet,
    invert_list: InvertList,
    docs: Doc,
}

impl InvertedIndex {
    fn tokenize(&self, texts: &[String]) -> Vec<u32> {
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
    async fn search(&self, query: &ScalarQuery) -> Result<UInt64Array> {
        let row_ids = match query {
            ScalarQuery::FullTextSearch(texts) => {
                let tokens = self.tokenize(texts)?;
                let row_ids = tokens
                    .iter()
                    .filter_map(|token| self.invert_list.retrieve(*token))
                    .collect::<Vec<_>>();
                row_ids
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
        let invert_list = InvertList::load(invert_list_reader).await?;
        let docs = Doc::load(docs_reader).await?;

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
        new_data.map(|batch| {
            let batch = batch?;
            let doc_column = batch.column(0).as_string::<i32>();
            let row_ids = batch[ROW_ID].as_primitive::<datatypes::UInt64Type>();

            let tokenizer = tantivy::tokenizer::TextAnalyzer::builder(
                tantivy::tokenizer::SimpleTokenizer::default(),
            )
            .build();
            let tokens = doc_column
                .iter()
                .zip(row_ids.iter())
                .map(|(doc, row_id)| {
                    let (doc, row_id) = (doc.unwrap(), row_id.unwrap());

                    let mut tokens = Vec::new();
                    let mut token_stream = tokenizer.token_stream(doc);
                    let tokens = Vec::new();
                    while let Some(token) = token_stream.next() {
                        tokens.push(token);
                    }

                    (tokens, row_id)
                })
                .collect();

            Result::Ok(tokens)
        });
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
struct TokenSet {
    tokens: Vec<String>,
    ids: Vec<u32>,
    occurrences: Vec<u64>,
}

impl TokenSet {
    async fn load(reader: Arc<dyn IndexReader>) -> Result<Self> {
        let mut tokens = Vec::new();
        let mut ids = Vec::new();
        let mut occurrences = Vec::new();
        for i in 0..reader.num_batches().await {
            let batch = reader.read_record_batch(i).await?;
            let token_col = batch[TOKEN_COL].as_string::<i32>();
            let token_id_col = batch[TOKEN_ID_COL].as_primitive::<datatypes::UInt32Type>();
            let occurency_col = batch[OCCURENCY_COL].as_primitive::<datatypes::UInt64Type>();

            tokens.extend(token_col.iter().map(|v| v.unwrap().to_owned()));
            ids.extend(token_id_col.iter().map(|v| v.unwrap()));
            occurrences.extend(occurency_col.iter().map(|v| v.unwrap()));
        }

        Ok(Self {
            tokens,
            ids,
            occurrences,
        })
    }

    fn get(&self, token: &String) -> Option<u32> {
        let pos = self.tokens.binary_search(token).ok()?;
        Some(self.ids[pos])
    }

    fn max_id(&self) -> u32 {
        self.ids.last().cloned().unwrap_or(0)
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
struct InvertList {
    tokens: Vec<u32>,
    row_ids: Vec<u64>,
    occurrences: Vec<u64>,
}

impl InvertList {
    async fn load(reader: Arc<dyn IndexReader>) -> Result<Self> {
        let mut tokens = Vec::new();
        let mut row_ids = Vec::new();
        let mut occurrences = Vec::new();
        for i in 0..reader.num_batches().await {
            let batch = reader.read_record_batch(i).await?;
            let token_col = batch[TOKEN_ID_COL].as_primitive::<datatypes::UInt32Type>();
            let row_id_col = batch[ROW_ID].as_primitive::<datatypes::UInt64Type>();
            let occurency_col = batch[OCCURENCY_COL].as_primitive::<datatypes::UInt64Type>();

            tokens.extend(token_col.iter().map(|v| v.unwrap()));
            row_ids.extend(row_id_col.iter().map(|v| v.unwrap()));
            occurrences.extend(occurency_col.iter().map(|v| v.unwrap()));
        }

        Ok(Self {
            tokens,
            row_ids,
            occurrences,
        })
    }

    fn retrieve(&self, token_id: u32) -> Option<u64> {
        let pos = self.tokens.binary_search(&token_id).ok()?;
        Some(self.row_ids[pos])
    }

    fn rank(&self, row_ids: &[u64]) -> Vec<(u64, u64)> {
        let mut ranked = Vec::with_capacity(row_ids.len());
        for row_id in row_ids {
            if let Some(pos) = self.row_ids.binary_search(row_id).ok() {
                ranked.push((self.row_ids[pos], self.occurrences[pos]));
            }
        }

        ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        ranked
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
struct Doc {
    row_ids: Vec<u64>,
    num_tokens: Vec<u32>,
}

impl Doc {
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
}
