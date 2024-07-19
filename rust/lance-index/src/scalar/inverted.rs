// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::Arc;

use arrow::array::{AsArray, Float32Builder, ListBuilder, UInt64Builder};
use arrow::datatypes::{self, Float32Type, UInt64Type};
use arrow_array::{ArrayRef, OffsetSizeTrait, RecordBatch, StringArray, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use futures::TryStreamExt;
use itertools::Itertools;
use lance_core::utils::mask::RowIdTreeMap;
use lance_core::{Error, Result, ROW_ID};
use lazy_static::lazy_static;
use roaring::RoaringBitmap;
use snafu::{location, Location};
use tantivy::tokenizer::Language;
use tracing::instrument;

use crate::prefilter::{NoFilter, PreFilter};
use crate::vector::graph::OrderedFloat;
use crate::Index;

use super::{AnyQuery, FullTextSearchQuery, IndexReader, IndexStore, SargableQuery, ScalarIndex};

pub const TOKENS_FILE: &str = "tokens.lance";
pub const INVERT_LIST_FILE: &str = "invert.lance";
pub const DOCS_FILE: &str = "docs.lance";

const TOKEN_COL: &str = "_token";
const TOKEN_ID_COL: &str = "_token_id";
const FREQUENCY_COL: &str = "_frequency";
const NUM_TOKEN_COL: &str = "_num_tokens";
pub const SCORE_COL: &str = "_score";
lazy_static! {
    pub static ref SCORE_FIELD: Field = Field::new(SCORE_COL, DataType::Float32, true);
}

// BM25 parameters
const K1: f32 = 1.2;
const B: f32 = 0.75;

lazy_static! {
    pub static ref TOKENIZER: tantivy::tokenizer::TextAnalyzer = {
        tantivy::tokenizer::TextAnalyzer::builder(tantivy::tokenizer::SimpleTokenizer::default())
            .filter(tantivy::tokenizer::RemoveLongFilter::limit(40))
            .filter(tantivy::tokenizer::StopWordFilter::new(Language::English).unwrap())
            .filter(tantivy::tokenizer::LowerCaser)
            .filter(tantivy::tokenizer::Stemmer::new(Language::English))
            .build()
    };
}

#[derive(Debug, Clone, Default, DeepSizeOf)]
pub struct InvertedIndex {
    tokens: TokenSet,
    invert_list: InvertedList,
    docs: DocSet,
}

impl InvertedIndex {
    pub fn new() -> Self {
        Self::default()
    }

    // map tokens to token ids
    // ignore tokens that are not in the index cause they won't contribute to the search
    #[instrument(level = "debug", skip_all)]
    fn map(&self, texts: &[String]) -> Vec<u32> {
        texts
            .iter()
            .filter_map(|text| self.tokens.get(text))
            .collect()
    }

    #[instrument(level = "debug", skip_all)]
    pub fn full_text_search(
        &self,
        query: &FullTextSearchQuery,
        prefilter: Arc<dyn PreFilter>,
    ) -> impl Iterator<Item = (u64, f32)> {
        let tokens = collect_tokens(&query.query);
        let token_ids = self
            .map(&tokens)
            .into_iter()
            .sorted_unstable()
            .dedup()
            .collect();
        self.bm25_search(token_ids, prefilter)
            .take(query.limit.unwrap_or(i64::MAX) as usize)
    }

    // search the documents that contain the query
    // return the row ids of the documents sorted by bm25 score
    // ref: https://en.wikipedia.org/wiki/Okapi_BM25
    #[instrument(level = "debug", skip_all)]
    fn bm25_search(
        &self,
        token_ids: Vec<u32>,
        prefilter: Arc<dyn PreFilter>,
    ) -> impl Iterator<Item = (u64, f32)> {
        let mut bm25_scores = HashMap::new();

        if prefilter.is_empty() {
            for token in &token_ids {
                let list = self.invert_list.retrieve(*token);
                for i in 0..list.len() {
                    let row_id = list.row_ids[i];
                    let score = list.scores[i];
                    bm25_scores
                        .entry(row_id)
                        .and_modify(|s| *s += score)
                        .or_insert(score);
                }
            }
        } else {
            for token in &token_ids {
                let list = self.invert_list.retrieve(*token);
                let indices = prefilter.filter_row_ids(Box::new(list.row_ids.iter()));
                for index in indices {
                    let index = index as usize;
                    let row_id = list.row_ids[index];
                    let score = list.scores[index];
                    bm25_scores
                        .entry(row_id)
                        .and_modify(|s| *s += score)
                        .or_insert(score);
                }
            }
        }

        bm25_scores
            .into_iter()
            .sorted_unstable_by_key(|k| OrderedFloat(-k.1))
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
    #[instrument(level = "debug", skip(self, query))]
    async fn search(&self, query: &dyn AnyQuery) -> Result<RowIdTreeMap> {
        let query = query.as_any().downcast_ref::<SargableQuery>().unwrap();
        let row_ids = match query {
            SargableQuery::FullTextSearch(query) => self
                .full_text_search(query, Arc::new(NoFilter))
                .map(|(row_id, _)| row_id),
            query => {
                return Err(Error::invalid_input(
                    format!("unsupported query {:?} for inverted index", query),
                    location!(),
                ))
            }
        };

        // sort the row ids (documents) by bm25 score

        Ok(RowIdTreeMap::from_iter(row_ids))
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
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        let token_set = self.tokens.clone();
        let mut invert_list = self.invert_list.clone();
        let mut docs = self.docs.clone();

        let token_set_batch = token_set.to_batch()?;
        invert_list.remap(mapping);
        let invert_list_batch = invert_list.to_batch()?;
        docs.remap(mapping);
        let docs_batch = docs.to_batch()?;

        let mut token_set_writer = dest_store
            .new_index_file(TOKENS_FILE, token_set_batch.schema())
            .await?;
        token_set_writer.write_record_batch(token_set_batch).await?;
        token_set_writer.finish().await?;

        let mut invert_list_writer = dest_store
            .new_index_file(INVERT_LIST_FILE, invert_list_batch.schema())
            .await?;
        invert_list_writer
            .write_record_batch(invert_list_batch)
            .await?;
        invert_list_writer.finish().await?;

        let mut docs_writer = dest_store
            .new_index_file(DOCS_FILE, docs_batch.schema())
            .await?;
        docs_writer.write_record_batch(docs_batch).await?;
        docs_writer.finish().await?;

        Ok(())
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        let mut token_set = self.tokens.clone();
        let mut invert_list = self.invert_list.clone();
        let mut docs = self.docs.clone();

        match new_data.schema().field(0).data_type() {
            DataType::Utf8 => {
                update_index::<i32>(new_data, &mut token_set, &mut invert_list, &mut docs).await?;
            }
            DataType::LargeUtf8 => {
                update_index::<i64>(new_data, &mut token_set, &mut invert_list, &mut docs).await?;
            }
            data_type => {
                return Err(Error::invalid_input(
                    format!("unsupported data type {} for inverted index", data_type),
                    location!(),
                ))
            }
        }

        // calculate bm25 scores
        invert_list.calculate_scores(&docs);

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

async fn update_index<Offset: OffsetSizeTrait>(
    new_data: SendableRecordBatchStream,
    token_set: &mut TokenSet,
    invert_list: &mut InvertedList,
    docs: &mut DocSet,
) -> Result<()> {
    let mut tokenizer = TOKENIZER.clone();
    let mut stream = new_data;
    while let Some(batch) = stream.try_next().await? {
        let doc_col = batch.column(0).as_string::<Offset>();
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

    Ok(())
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
        let mut tokens = Vec::with_capacity(self.tokens.len());
        let mut token_ids = Vec::with_capacity(self.tokens.len());
        let mut frequencies = Vec::with_capacity(self.tokens.len());
        self.tokens
            .iter()
            .for_each(|(token, (token_id, frequency))| {
                tokens.push(token.clone());
                token_ids.push(*token_id);
                frequencies.push(*frequency);
            });
        let token_col = StringArray::from(tokens);
        let token_id_col = UInt32Array::from(token_ids);
        let frequency_col = UInt64Array::from(frequencies);

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
        let mut tokens = HashMap::new();
        let mut next_id = 0;
        for i in 0..reader.num_batches().await {
            let batch = reader.read_record_batch(i).await?;
            let token_col = batch[TOKEN_COL].as_string::<i32>();
            let token_id_col = batch[TOKEN_ID_COL].as_primitive::<datatypes::UInt32Type>();
            let frequency_col = batch[FREQUENCY_COL].as_primitive::<datatypes::UInt64Type>();

            for ((token, &token_id), &frequency) in token_col
                .iter()
                .zip(token_id_col.values().iter())
                .zip(frequency_col.values().iter())
            {
                let token = token.unwrap();
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

#[derive(Debug, Clone, Default, DeepSizeOf)]
struct PostingList {
    row_ids: Vec<u64>,
    frequencies: Vec<f32>,
    scores: Vec<f32>,
}

impl PostingList {
    fn new(row_ids: Vec<u64>, frequencies: Vec<f32>, scores: Vec<f32>) -> Self {
        Self {
            row_ids,
            frequencies,
            scores,
        }
    }

    fn len(&self) -> usize {
        self.row_ids.len()
    }
}

// InvertedList is a mapping from token ids to row ids
// it's used to retrieve the documents that contain a token
#[derive(Debug, Clone, Default, DeepSizeOf)]
struct InvertedList {
    // the index is the token id
    inverted_list: Vec<PostingList>,
}

impl InvertedList {
    fn to_batch(&self) -> Result<RecordBatch> {
        // let mut tokens = Vec::with_capacity(self.inverted_list.len());
        let mut row_ids_list_builder =
            ListBuilder::with_capacity(UInt64Builder::new(), self.inverted_list.len());
        let mut frequencies_list_builder =
            ListBuilder::with_capacity(Float32Builder::new(), self.inverted_list.len());
        let mut bm25_list_builder =
            ListBuilder::with_capacity(Float32Builder::new(), self.inverted_list.len());

        for list in &self.inverted_list {
            // tokens.push(*token_id);
            let row_ids_builder = row_ids_list_builder.values();
            let frequencies_builder = frequencies_list_builder.values();
            let bm25_builder = bm25_list_builder.values();
            row_ids_builder.append_slice(list.row_ids.as_slice());
            frequencies_builder.append_slice(list.frequencies.as_slice());
            bm25_builder.append_slice(list.scores.as_slice());
            row_ids_list_builder.append(true);
            frequencies_list_builder.append(true);
            bm25_list_builder.append(true);
        }

        // let token_id_col = UInt32Array::from(tokens);
        let row_ids_col = row_ids_list_builder.finish();
        let frequencies_col = frequencies_list_builder.finish();
        let bm25_col = bm25_list_builder.finish();

        let schema = arrow_schema::Schema::new(vec![
            // arrow_schema::Field::new(TOKEN_ID_COL, DataType::UInt32, false),
            arrow_schema::Field::new(
                ROW_ID,
                DataType::List(Field::new_list_field(DataType::UInt64, true).into()),
                false,
            ),
            arrow_schema::Field::new(
                FREQUENCY_COL,
                DataType::List(Field::new_list_field(DataType::Float32, true).into()),
                false,
            ),
            arrow_schema::Field::new(
                SCORE_COL,
                DataType::List(Field::new_list_field(DataType::Float32, true).into()),
                false,
            ),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                // Arc::new(token_id_col) as ArrayRef,
                Arc::new(row_ids_col) as ArrayRef,
                Arc::new(frequencies_col) as ArrayRef,
                Arc::new(bm25_col) as ArrayRef,
            ],
        )?;
        Ok(batch)
    }

    async fn load(reader: Arc<dyn IndexReader>) -> Result<Self> {
        let mut inverted_list = Vec::with_capacity(reader.num_rows());
        for i in 0..reader.num_batches().await {
            let batch = reader.read_record_batch(i).await?;
            // let token_col = batch[TOKEN_ID_COL].as_primitive::<datatypes::UInt32Type>();
            let row_ids_col = batch[ROW_ID].as_list::<i32>();
            let frequencies_col = batch[FREQUENCY_COL].as_list::<i32>();
            let scores_col = batch[SCORE_COL].as_list::<i32>();

            for ((row_ids, frequencies), scores) in row_ids_col
                .iter()
                .zip(frequencies_col.iter())
                .zip(scores_col.iter())
            {
                let row_ids = row_ids.unwrap();
                let frequencies = frequencies.unwrap();
                let scores = scores.unwrap();
                let row_ids = row_ids.as_primitive::<UInt64Type>().values().to_vec();
                let frequencies = frequencies.as_primitive::<Float32Type>().values().to_vec();
                let scores = scores.as_primitive::<Float32Type>().values().to_vec();
                inverted_list.push(PostingList::new(row_ids, frequencies, scores));
            }
        }

        Ok(Self { inverted_list })
    }

    fn remap(&mut self, mapping: &HashMap<u64, Option<u64>>) {
        for list in self.inverted_list.iter_mut() {
            let mut new_row_ids = Vec::new();
            let mut new_freqs = Vec::new();
            let mut new_scores = Vec::new();

            for i in 0..list.len() {
                let row_id = list.row_ids[i];
                let freq = list.frequencies[i];
                let score = list.scores[i];

                match mapping.get(&row_id) {
                    Some(Some(new_row_id)) => {
                        new_row_ids.push(*new_row_id);
                        new_freqs.push(freq);
                        new_scores.push(score);
                    }
                    _ => continue,
                }
            }

            *list = PostingList::new(new_row_ids, new_freqs, new_scores);
        }
    }

    // for efficiency, we don't check if the row_id exists
    // we assume that the row_id is unique and doesn't exist in the list
    fn add(&mut self, token_cnt: HashMap<u32, u32>, row_id: u64) {
        for (token_id, freq) in token_cnt {
            let token_id = token_id as usize;
            if token_id >= self.inverted_list.len() {
                self.inverted_list
                    .resize_with(token_id + 1, PostingList::default);
            }
            let list = &mut self.inverted_list[token_id];
            list.row_ids.push(row_id);
            list.frequencies.push(freq as f32);
        }
    }

    fn calculate_scores(&mut self, docs: &DocSet) {
        let avgdl = docs.average_length();
        for list in self.inverted_list.iter_mut() {
            let idf = idf(list.len(), docs.len());
            list.scores.resize(list.len(), 0.0);
            for i in 0..list.len() {
                let row_id = list.row_ids[i];
                let freq = list.frequencies[i];
                let doc_norm = docs.num_tokens(row_id) as f32 / avgdl;
                list.scores[i] = idf * (K1 + 1.0) * freq / (freq + K1 * (1.0 - B + B * doc_norm));
            }
        }
    }

    fn retrieve(&self, token_id: u32) -> &PostingList {
        &self.inverted_list[token_id as usize]
    }
}

// DocSet is a mapping from row ids to the number of tokens in the document
// It's used to sort the documents by the bm25 score
#[derive(Debug, Clone, Default, DeepSizeOf)]
struct DocSet {
    // row id -> (num tokens, norm_len)
    token_count: HashMap<u64, u32>,
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

            for (&row_id, &num_tokens) in row_id_col
                .values()
                .iter()
                .zip(num_tokens_col.values().iter())
            {
                token_count.insert(row_id, num_tokens);
                total_tokens += num_tokens as u64;
            }
        }

        Ok(Self {
            token_count,
            total_tokens,
        })
    }

    fn remap(&mut self, mapping: &HashMap<u64, Option<u64>>) {
        for (old_row_id, new_row_id) in mapping {
            match new_row_id {
                Some(new_row_id) => {
                    if let Some(num_tokens) = self.token_count.remove(old_row_id) {
                        self.token_count.insert(*new_row_id, num_tokens);
                    }
                }
                None => {
                    self.token_count.remove(old_row_id);
                }
            }
        }
    }

    #[inline]
    fn num_tokens(&self, row_id: u64) -> u32 {
        self.token_count.get(&row_id).cloned().unwrap_or_default()
    }

    fn add(&mut self, row_id: u64, num_tokens: u32) {
        self.token_count.insert(row_id, num_tokens);
        self.total_tokens += num_tokens as u64;
    }
}

#[inline]
fn idf(nq: usize, num_docs: usize) -> f32 {
    let num_docs = num_docs as f32;
    ((num_docs - nq as f32 + 0.5) / (nq as f32 + 0.5) + 1.0).ln()
}

#[instrument(level = "debug", skip(batches))]
pub fn flat_full_text_search(
    batches: &[&RecordBatch],
    doc_col: &str,
    query: &str,
) -> Result<Vec<u64>> {
    if batches.is_empty() {
        return Ok(Vec::new());
    }

    match batches[0][doc_col].data_type() {
        DataType::Utf8 => do_flat_full_text_search::<i32>(batches, doc_col, query),
        DataType::LargeUtf8 => do_flat_full_text_search::<i64>(batches, doc_col, query),
        data_type => Err(Error::invalid_input(
            format!("unsupported data type {} for inverted index", data_type),
            location!(),
        )),
    }
}

fn do_flat_full_text_search<Offset: OffsetSizeTrait>(
    batches: &[&RecordBatch],
    doc_col: &str,
    query: &str,
) -> Result<Vec<u64>> {
    let mut results = Vec::new();
    let query_tokens = collect_tokens(query).into_iter().collect::<HashSet<_>>();
    for batch in batches {
        let row_id_array = batch[ROW_ID].as_primitive::<UInt64Type>();
        let doc_array = batch[doc_col].as_string::<Offset>();
        for i in 0..row_id_array.len() {
            let doc = doc_array.value(i);
            let doc_tokens = collect_tokens(doc);
            if doc_tokens.iter().any(|token| query_tokens.contains(token)) {
                results.push(row_id_array.value(i));
                assert!(doc.contains(query));
            }
        }
    }

    Ok(results)
}

fn collect_tokens(text: &str) -> Vec<String> {
    let mut tokenizer = TOKENIZER.clone();
    let mut stream = tokenizer.token_stream(text);
    let mut tokens = Vec::new();
    while let Some(token) = stream.next() {
        tokens.push(token.text.to_owned());
    }
    tokens
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, ArrayRef, GenericStringArray, RecordBatch, UInt64Array};
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream;
    use lance_io::object_store::ObjectStore;
    use object_store::path::Path;

    use crate::scalar::lance_format::LanceIndexStore;
    use crate::scalar::{FullTextSearchQuery, ScalarIndex};

    async fn test_inverted_index<Offset: arrow::array::OffsetSizeTrait>() {
        let tempdir = tempfile::tempdir().unwrap();
        let index_dir = Path::from_filesystem_path(tempdir.path()).unwrap();
        let store = LanceIndexStore::new(ObjectStore::local(), index_dir, None);

        let invert_index = super::InvertedIndex::default();
        let row_id_col = UInt64Array::from(vec![0, 1, 2, 3]);
        let doc_col = GenericStringArray::<Offset>::from(vec![
            "lance database search",
            "lance database",
            "lance search",
            "database search",
        ]);
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![
                arrow_schema::Field::new("doc", doc_col.data_type().to_owned(), false),
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
            .search(&super::SargableQuery::FullTextSearch(
                FullTextSearchQuery::new("lance".to_owned()),
            ))
            .await
            .unwrap();
        assert_eq!(row_ids.len(), Some(3));
        assert!(row_ids.contains(0));
        assert!(row_ids.contains(1));
        assert!(row_ids.contains(2));

        let row_ids = invert_index
            .search(&super::SargableQuery::FullTextSearch(
                FullTextSearchQuery::new("database".to_owned()),
            ))
            .await
            .unwrap();
        assert_eq!(row_ids.len(), Some(3));
        assert!(row_ids.contains(0));
        assert!(row_ids.contains(1));
        assert!(row_ids.contains(3));
    }

    #[tokio::test]
    async fn test_inverted_index_with_string() {
        test_inverted_index::<i32>().await;
    }

    #[tokio::test]
    async fn test_inverted_index_with_large_string() {
        test_inverted_index::<i64>().await;
    }
}
