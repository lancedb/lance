// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{DefaultHasher, Hasher};
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, RwLock};

use crate::scalar::{IndexReader, IndexStore, IndexWriter, InvertedIndexParams};
use crate::vector::graph::OrderedFloat;
use arrow::array::AsArray;
use arrow::datatypes::{self, Float32Type, Int32Type, UInt64Type};
use arrow_schema::SchemaRef;
use datafusion::execution::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use futures::{stream, StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_arrow::iter_str_array;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, CPU_RUNTIME};
use lance_core::{Error, Result, ROW_ID};
use lazy_static::lazy_static;
use rayon::prelude::*;
use snafu::{location, Location};
use tracing::instrument;

use super::index::*;

lazy_static! {
    static ref DOC_CHUNK_SIZE: usize = std::env::var("DOC_CHUNK_SIZE")
        .unwrap_or_else(|_| "2048".to_string())
        .parse()
        .expect("failed to parse DOC_CHUNK_SIZE");
}

#[derive(Debug, Clone, Default, DeepSizeOf)]
pub struct InvertedIndexBuilder {
    params: InvertedIndexParams,

    pub(crate) tokens: TokenSet,
    pub(crate) invert_list: InvertedList,
    pub(crate) docs: DocSet,
}

impl InvertedIndexBuilder {
    pub fn new(params: InvertedIndexParams) -> Self {
        let with_position = params.with_position;
        Self {
            params,
            tokens: TokenSet::default(),
            invert_list: InvertedList::new(with_position),
            docs: DocSet::default(),
        }
    }

    pub fn from_existing_index(tokens: TokenSet, invert_list: InvertedList, docs: DocSet) -> Self {
        let params = InvertedIndexParams::default().with_position(invert_list.with_position);
        Self {
            params,
            tokens,
            invert_list,
            docs,
        }
    }

    pub async fn update(
        &mut self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        self.update_index(new_data).await?;
        self.save(dest_store).await?;
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn update_index(&mut self, mut stream: SendableRecordBatchStream) -> Result<()> {
        const NUM_SHARDS: usize = 32;
        let token_maps = (0..NUM_SHARDS)
            .map(|_| RwLock::new(HashMap::new()))
            .collect_vec();
        // init the token maps
        {
            let mut token_maps = token_maps
                .iter()
                .map(|map| map.write().unwrap())
                .collect_vec();
            for (token, token_id) in std::mem::take(&mut self.tokens.tokens) {
                let mut hasher = DefaultHasher::new();
                hasher.write(token.as_bytes());
                let shard = hasher.finish() as usize % NUM_SHARDS;
                token_maps[shard].insert(token, token_id);
            }
        }
        let next_id = AtomicU32::new(self.tokens.next_id);

        let start = std::time::Instant::now();
        while let Some(batch) = stream.try_next().await? {
            let doc_iter = iter_str_array(batch.column(0));
            let row_id_col = batch[ROW_ID].as_primitive::<datatypes::UInt64Type>();
            let docs = doc_iter
                .zip(row_id_col.values().iter())
                .filter_map(|(doc, row_id)| doc.map(|doc| (doc, *row_id)))
                .collect_vec(); // we have to collect so that we can call `into_par_iter()`

            let docs = docs
                .into_par_iter()
                .map_init(
                    || {
                        (
                            TOKENIZER.clone(),
                            vec![Vec::new(); NUM_SHARDS],
                            Vec::with_capacity(*DOC_CHUNK_SIZE),
                            Vec::with_capacity(*DOC_CHUNK_SIZE),
                            HashMap::new(),
                        )
                    }, // reuse the memory
                    |(tokenizer, token_buffer, tokens, unknown_tokens, token_occurrences),
                     (doc, row_id)| {
                        // tokenize the document
                        let mut token_stream = tokenizer.token_stream(doc);
                        while token_stream.advance() {
                            let token = token_stream.token_mut();
                            let mut hasher = DefaultHasher::new();
                            hasher.write(token.text.as_bytes());
                            let shard = hasher.finish() as usize % NUM_SHARDS;
                            token_buffer[shard]
                                .push((std::mem::take(&mut token.text), token.position as i32));
                        }

                        // map the tokens to token ids
                        tokens.clear();
                        token_buffer
                            .iter_mut()
                            .enumerate()
                            .for_each(|(shard, buffer)| {
                                for chunk in buffer.iter_mut().chunks(*DOC_CHUNK_SIZE).into_iter() {
                                    // when the document is very long, it almost always contains new tokens,
                                    // if we don't chunk the tokens, we will have to always acquire write lock of the token_map for a long time
                                    {
                                        let token_map = token_maps[shard].read().unwrap();
                                        for (text, position) in chunk {
                                            if let Some(token_id) = token_map.get(text) {
                                                tokens.push((*token_id, *position));
                                            } else {
                                                unknown_tokens
                                                    .push((std::mem::take(text), *position));
                                            }
                                        }
                                    }

                                    if unknown_tokens.is_empty() {
                                        continue;
                                    }
                                    {
                                        let mut token_map = token_maps[shard].write().unwrap();
                                        for (text, position) in unknown_tokens.iter_mut() {
                                            let token_id = *token_map
                                                .entry(std::mem::take(text))
                                                .or_insert_with(|| {
                                                    next_id.fetch_add(
                                                        1,
                                                        std::sync::atomic::Ordering::Relaxed,
                                                    )
                                                });
                                            tokens.push((token_id, *position));
                                        }
                                    }
                                    unknown_tokens.clear()
                                }
                                buffer.clear();
                            });

                        // collect the token occurrences
                        let num_tokens = tokens.len() as u32;
                        for (token_id, position) in tokens {
                            let position = *position;
                            token_occurrences
                                .entry(*token_id)
                                .and_modify(|positions: &mut Vec<i32>| positions.push(position))
                                .or_insert_with(|| vec![position]);
                        }

                        // phrase query requires the positions to be sorted
                        if self.invert_list.with_position {
                            token_occurrences.iter_mut().for_each(|(_, positions)| {
                                positions.sort_unstable();
                            });
                        }
                        (num_tokens, std::mem::take(token_occurrences), row_id)
                    },
                )
                .collect::<Vec<_>>();

            self.invert_list
                .resize(next_id.load(std::sync::atomic::Ordering::Relaxed) as usize);
            for (num_tokens, token_occurrences, row_id) in docs {
                self.invert_list.add(token_occurrences, row_id);
                self.docs.add(row_id, num_tokens);
            }
        }

        token_maps.into_iter().for_each(|token_map| {
            self.tokens
                .tokens
                .extend(token_map.write().unwrap().drain());
        });
        self.tokens.next_id = next_id.load(std::sync::atomic::Ordering::Relaxed);
        log::info!("tokenize documents elapsed {:?}", start.elapsed());

        Ok(())
    }

    pub async fn remap(
        &mut self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        // no need to remap the TokenSet,
        // since no row_id is stored in the TokenSet
        self.invert_list.remap(mapping);
        self.docs.remap(mapping);

        self.save(dest_store).await?;
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn save(&mut self, dest_store: &dyn IndexStore) -> Result<()> {
        let start = std::time::Instant::now();
        let token_set_batch = std::mem::take(&mut self.tokens).to_batch()?;
        let mut token_set_writer = dest_store
            .new_index_file(TOKENS_FILE, token_set_batch.schema())
            .await?;
        token_set_writer.write_record_batch(token_set_batch).await?;
        token_set_writer.finish().await?;
        log::info!("write tokens: {:?}", start.elapsed());

        let docs = Arc::new(std::mem::take(&mut self.docs));
        let start = std::time::Instant::now();
        let invert_list_writer = dest_store
            .new_index_file(INVERT_LIST_FILE, self.invert_list.schema().clone())
            .await?;
        std::mem::take(&mut self.invert_list)
            .into_batches(docs.clone(), invert_list_writer)
            .await?;
        log::info!("write inverted lists done: {:?}", start.elapsed());

        let start = std::time::Instant::now();
        let docs_batch = docs.to_batch()?;
        let mut docs_writer = dest_store
            .new_index_file(DOCS_FILE, docs_batch.schema())
            .await?;
        docs_writer.write_record_batch(docs_batch).await?;
        docs_writer.finish().await?;
        log::info!("write docs done: {:?}", start.elapsed());

        Ok(())
    }
}

// InvertedList is a mapping from token ids to row ids
// it's used to retrieve the documents that contain a token
#[derive(Debug, Clone)]
pub struct InvertedList {
    // the index is the token id
    inverted_list: Vec<PostingListBuilder>,
    pub(crate) with_position: bool,
    schema: SchemaRef,
}

impl Default for InvertedList {
    fn default() -> Self {
        Self::new(true)
    }
}

impl DeepSizeOf for InvertedList {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.inverted_list.deep_size_of_children(context)
    }
}

impl InvertedList {
    pub fn new(with_position: bool) -> Self {
        let mut fields = vec![
            arrow_schema::Field::new(ROW_ID, arrow_schema::DataType::UInt64, false),
            arrow_schema::Field::new(FREQUENCY_COL, arrow_schema::DataType::Float32, false),
        ];
        if with_position {
            fields.push(arrow_schema::Field::new(
                POSITION_COL,
                arrow_schema::DataType::List(Arc::new(arrow_schema::Field::new(
                    "item",
                    arrow_schema::DataType::Int32,
                    true,
                ))),
                false,
            ));
        }
        let schema = Arc::new(arrow_schema::Schema::new(fields));
        Self {
            inverted_list: Vec::new(),
            with_position,
            schema,
        }
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    // the schema of the inverted list is | row_id | frequency | positions (optional)
    pub async fn into_batches(
        self,
        docs: Arc<DocSet>,
        mut writer: Box<dyn IndexWriter>,
    ) -> Result<()> {
        let num_lists = self.inverted_list.len();
        let schema = self.schema().clone();
        let mut results = stream::iter(self.inverted_list)
            .map(|list| {
                let schema = schema.clone();
                let docs = docs.clone();
                CPU_RUNTIME.spawn_blocking(move || list.to_batch(schema, docs))
            })
            .buffered(get_num_compute_intensive_cpus());

        let mut offsets = Vec::with_capacity(num_lists);
        let mut max_scores = Vec::with_capacity(num_lists);
        let mut num_rows = 0;
        while let Some(result) = results.try_next().await? {
            let (batch, max_score) = result?;
            offsets.push(num_rows);
            max_scores.push(max_score);
            num_rows += batch.num_rows();
            writer.write_record_batch(batch).await?;
        }
        let metadata = HashMap::from_iter(vec![
            ("offsets".to_owned(), serde_json::to_string(&offsets)?),
            ("max_scores".to_owned(), serde_json::to_string(&max_scores)?),
        ]);

        writer.finish_with_metadata(metadata).await?;
        Ok(())
    }

    pub async fn load(reader: Arc<dyn IndexReader>) -> Result<Self> {
        let mut inverted_lists = Vec::with_capacity(reader.num_rows());
        let offsets = reader
            .schema()
            .metadata
            .get("offsets")
            .ok_or(Error::Index {
                message: "offsets not found".to_string(),
                location: location!(),
            })?;
        let offsets: Vec<usize> = serde_json::from_str(offsets)?;

        for i in 0..offsets.len() {
            let offset = offsets[i];
            let next_offset = if i + 1 < offsets.len() {
                offsets[i + 1]
            } else {
                reader.num_rows()
            };
            let batch = reader.read_range(offset..next_offset, None).await?;
            let row_ids_col = batch[ROW_ID].as_primitive::<UInt64Type>().values();
            let frequencies_col = batch[FREQUENCY_COL].as_primitive::<Float32Type>().values();
            let positions_col = batch.column_by_name(POSITION_COL).map(|col| {
                col.as_list::<i32>()
                    .iter()
                    .map(|positions| {
                        positions
                            .unwrap()
                            .as_primitive::<Int32Type>()
                            .values()
                            .to_vec()
                    })
                    .collect_vec()
            });
            inverted_lists.push(PostingListBuilder::new(
                row_ids_col.to_vec(),
                frequencies_col.to_vec(),
                positions_col,
            ));
        }

        let mut inverted_list = Self::new(reader.schema().field(POSITION_COL).is_some());
        inverted_list.inverted_list = inverted_lists;
        Ok(inverted_list)
    }

    pub fn remap(&mut self, mapping: &HashMap<u64, Option<u64>>) {
        for list in self.inverted_list.iter_mut() {
            let mut new_row_ids = Vec::new();
            let mut new_freqs = Vec::new();
            let mut new_positions = list.positions.as_ref().map(|_| Vec::new());

            for i in 0..list.len() {
                let row_id = list.row_ids[i];
                let freq = list.frequencies[i];
                let positions = list
                    .positions
                    .as_ref()
                    .map(|positions| positions.get(i).to_vec());

                match mapping.get(&row_id) {
                    Some(Some(new_row_id)) => {
                        new_row_ids.push(*new_row_id);
                        new_freqs.push(freq);
                        if let Some(new_positions) = new_positions.as_mut() {
                            new_positions.push(positions.unwrap());
                        }
                    }
                    _ => continue,
                }
            }

            *list = PostingListBuilder::new(new_row_ids, new_freqs, new_positions);
        }
    }

    pub fn resize(&mut self, len: usize) {
        if len > self.inverted_list.len() {
            self.inverted_list
                .resize_with(len, || PostingListBuilder::empty(self.with_position));
        }
    }

    // for efficiency, we don't check if the row_id exists
    // we assume that the row_id is unique and doesn't exist in the list
    pub fn add(&mut self, token_occurrences: HashMap<u32, Vec<i32>>, row_id: u64) {
        token_occurrences
            .into_iter()
            .for_each(|(token_id, term_positions)| {
                let list = &mut self.inverted_list[token_id as usize];
                list.add(row_id, term_positions);
            });
    }
}

#[derive(Debug, Eq, PartialEq, Clone, DeepSizeOf)]
pub struct OrderedDoc {
    pub row_id: u64,
    pub score: OrderedFloat,
}

impl OrderedDoc {
    pub fn new(row_id: u64, score: f32) -> Self {
        Self {
            row_id,
            score: OrderedFloat(score),
        }
    }
}

impl PartialOrd for OrderedDoc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedDoc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.cmp(&other.score)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, ArrayRef, GenericStringArray, RecordBatch, UInt64Array};
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream;
    use lance_core::ROW_ID_FIELD;
    use lance_io::object_store::ObjectStore;
    use object_store::path::Path;

    use crate::scalar::lance_format::LanceIndexStore;
    use crate::scalar::{FullTextSearchQuery, SargableQuery, ScalarIndex};

    async fn test_inverted_index<Offset: arrow::array::OffsetSizeTrait>() {
        let tempdir = tempfile::tempdir().unwrap();
        let index_dir = Path::from_filesystem_path(tempdir.path()).unwrap();
        let store = LanceIndexStore::new(ObjectStore::local(), index_dir, None);

        let mut invert_index = super::InvertedIndexBuilder::default();
        let doc_col = GenericStringArray::<Offset>::from(vec![
            "lance database the search",
            "lance database",
            "lance search",
            "database search",
            "unrelated doc",
            "unrelated",
        ]);
        let row_id_col = UInt64Array::from(Vec::from_iter(0..doc_col.len() as u64));
        let batch = RecordBatch::try_new(
            arrow_schema::Schema::new(vec![
                arrow_schema::Field::new("doc", doc_col.data_type().to_owned(), false),
                ROW_ID_FIELD.clone(),
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
            .search(&SargableQuery::FullTextSearch(
                FullTextSearchQuery::new("lance".to_owned()).limit(Some(3)),
            ))
            .await
            .unwrap();
        assert_eq!(row_ids.len(), Some(3));
        assert!(row_ids.contains(0));
        assert!(row_ids.contains(1));
        assert!(row_ids.contains(2));

        let row_ids = invert_index
            .search(&SargableQuery::FullTextSearch(
                FullTextSearchQuery::new("database".to_owned()).limit(Some(3)),
            ))
            .await
            .unwrap();
        assert_eq!(row_ids.len(), Some(3));
        assert!(row_ids.contains(0));
        assert!(row_ids.contains(1));
        assert!(row_ids.contains(3));

        // test phrase query
        // for non-phrasal query, the order of the tokens doesn't matter
        // so there should be 4 documents that contain "database" or "lance"
        let row_ids = invert_index
            .search(&SargableQuery::FullTextSearch(
                FullTextSearchQuery::new("lance database".to_owned()).limit(Some(10)),
            ))
            .await
            .unwrap();
        assert_eq!(row_ids.len(), Some(4));
        assert!(row_ids.contains(0));
        assert!(row_ids.contains(1));
        assert!(row_ids.contains(2));
        assert!(row_ids.contains(3));

        let row_ids = invert_index
            .search(&SargableQuery::FullTextSearch(
                FullTextSearchQuery::new("\"lance database\"".to_owned()).limit(Some(10)),
            ))
            .await
            .unwrap();
        assert_eq!(row_ids.len(), Some(2));
        assert!(row_ids.contains(0));
        assert!(row_ids.contains(1));

        let row_ids = invert_index
            .search(&SargableQuery::FullTextSearch(
                FullTextSearchQuery::new("\"database lance\"".to_owned()).limit(Some(10)),
            ))
            .await
            .unwrap();
        assert_eq!(row_ids.len(), Some(0));
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
