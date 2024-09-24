// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{DefaultHasher, Hasher};
use std::sync::Arc;

use crate::scalar::lance_format::LanceIndexStore;
use crate::scalar::{IndexReader, IndexStore, IndexWriter, InvertedIndexParams};
use crate::vector::graph::OrderedFloat;
use arrow::array::AsArray;
use arrow::datatypes::{self, Float32Type, Int32Type, UInt64Type};
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::execution::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use futures::stream::repeat_with;
use futures::{stream, StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_arrow::iter_str_array;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, CPU_RUNTIME};
use lance_core::{Error, Result, ROW_ID};
use lance_io::object_store::ObjectStore;
use lazy_static::lazy_static;
use object_store::path::Path;
use rayon::prelude::*;
use snafu::{location, Location};
use tempfile::{tempdir, TempDir};
use tracing::instrument;

use super::index::*;

lazy_static! {
    static ref DOC_CHUNK_SIZE: usize = std::env::var("DOC_CHUNK_SIZE")
        .unwrap_or_else(|_| "2048".to_string())
        .parse()
        .expect("failed to parse DOC_CHUNK_SIZE");
    static ref FLUSH_THRESHOLD: usize = std::env::var("FLUSH_THRESHOLD")
        .unwrap_or_else(|_| "256".to_string())
        .parse()
        .expect("failed to parse FLUSH_THRESHOLD");
}

const NUM_SHARDS: usize = 16;

#[derive(Debug, Default, DeepSizeOf)]
pub struct InvertedIndexBuilder {
    params: InvertedIndexParams,

    pub(crate) tokens: TokenSet,
    inverted_list: Option<Arc<InvertedListReader>>,
    pub(crate) docs: DocSet,

    posting_readers: Vec<PostingReader>,
}

impl InvertedIndexBuilder {
    pub fn new(params: InvertedIndexParams) -> Self {
        Self {
            params,
            tokens: TokenSet::default(),
            inverted_list: None,
            docs: DocSet::default(),
            posting_readers: Vec::new(),
        }
    }

    pub fn from_existing_index(
        tokens: TokenSet,
        inverted_list: Arc<InvertedListReader>,
        docs: DocSet,
    ) -> Self {
        let params = InvertedIndexParams::default().with_position(inverted_list.has_positions());

        Self {
            params,
            tokens,
            inverted_list: Some(inverted_list),
            docs,
            posting_readers: Vec::new(),
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
        // init the token maps
        let mut token_maps = (0..NUM_SHARDS).map(|_| HashMap::new()).collect_vec();
        for (token, token_id) in self.tokens.tokens.iter() {
            let mut hasher = DefaultHasher::new();
            hasher.write(token.as_bytes());
            let shard = hasher.finish() as usize % NUM_SHARDS;
            token_maps[shard].insert(token.clone(), *token_id);
        }

        // spawn workers to build the index
        let mut result_futs = Vec::with_capacity(NUM_SHARDS);
        let mut senders = Vec::with_capacity(NUM_SHARDS);
        for token_map in token_maps.into_iter() {
            let inverted_list = self.inverted_list.clone();
            let worker = IndexWorker::new(token_map, self.params.with_position).await?;
            let (sender, receiver) = tokio::sync::mpsc::channel(2);
            senders.push(sender);
            result_futs.push(tokio::spawn(async move {
                let mut worker = worker;
                let mut receiver = receiver;
                while let Some((tokens, row_id)) = receiver.recv().await {
                    worker.add(tokens, row_id).await?;
                }
                Ok::<_, Error>(worker.into_reader(inverted_list).await?)
            }));
        }

        let start = std::time::Instant::now();
        while let Some(batch) = stream.try_next().await? {
            let doc_iter = iter_str_array(batch.column(0));
            let row_id_col = batch[ROW_ID].as_primitive::<datatypes::UInt64Type>();
            let docs = doc_iter
                .zip(row_id_col.values().iter())
                .filter_map(|(doc, row_id)| doc.map(|doc| (doc, *row_id)))
                .collect_vec(); // we have to collect so that we can call `into_par_iter()`

            let num_tokens = docs
                .into_par_iter()
                .map_init(
                    || (TOKENIZER.clone(), vec![Vec::new(); NUM_SHARDS]),
                    |(tokenizer, token_buffers), (doc, row_id)| {
                        // tokenize the document
                        let mut num_tokens = 0;
                        let mut token_stream = tokenizer.token_stream(doc);
                        while token_stream.advance() {
                            let token = token_stream.token_mut();
                            let mut hasher = DefaultHasher::new();
                            hasher.write(token.text.as_bytes());
                            let shard = hasher.finish() as usize % NUM_SHARDS;
                            token_buffers[shard]
                                .push((std::mem::take(&mut token.text), token.position as i32));
                            num_tokens += 1;
                        }

                        token_buffers.iter_mut().enumerate().map(|(shard, buffer)| {
                            let num_tokens = buffer.len();
                            senders[shard].blocking_send((std::mem::take(buffer), row_id));
                            num_tokens
                        });
                        token_buffers.clear();

                        (row_id, num_tokens)
                    },
                )
                .collect::<Vec<_>>();

            for (row_id, num_tokens) in num_tokens {
                self.docs.add(row_id, num_tokens);
            }
        }

        // wait for the workers to finish
        for result in futures::future::try_join_all(result_futs).await? {
            let result = result?;
            self.posting_readers.push(result);
        }

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
        // self.inverted_list.remap(mapping);
        self.docs.remap(mapping);

        self.save(dest_store).await?;
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn save(&mut self, dest_store: &dyn IndexStore) -> Result<()> {
        let mut tokens = TokenSet::new();
        let docs = Arc::new(std::mem::take(&mut self.docs));
        let start = std::time::Instant::now();
        let schema = inverted_list_schema(self.params.with_position);
        let mut invert_list_writer = dest_store.new_index_file(INVERT_LIST_FILE, schema).await?;
        for reader in self.posting_readers.iter_mut() {
            while let Some((token, batches)) = reader.next().await? {
                tokens.add(token);
                for batch in batches {
                    invert_list_writer.write_record_batch(batch).await?;
                }
            }
        }
        log::info!("write inverted lists done: {:?}", start.elapsed());

        let start = std::time::Instant::now();
        let token_set_batch = tokens.to_batch()?;
        let mut token_set_writer = dest_store
            .new_index_file(TOKENS_FILE, token_set_batch.schema())
            .await?;
        token_set_writer.write_record_batch(token_set_batch).await?;
        token_set_writer.finish().await?;
        log::info!("write tokens: {:?}", start.elapsed());

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

struct IndexWorker {
    // we have to keep track of the existing tokens
    // because we need to know the token id when we receive posting list from existing inverted list
    existing_tokens: HashMap<String, u32>,
    posting_lists: HashMap<String, PostingListBuilder>,
    schema: SchemaRef,
    tmpdir: TempDir,
    store: Arc<dyn IndexStore>,
    writer: Box<dyn IndexWriter>,
    token_offsets: HashMap<String, Vec<(usize, usize)>>,
}

impl IndexWorker {
    async fn new(existing_tokens: HashMap<String, u32>, with_position: bool) -> Result<Self> {
        let tmpdir = tempdir()?;
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local(),
            Path::from_filesystem_path(tmpdir.path())?,
            None,
        ));

        let schema = inverted_list_schema(with_position);
        let writer = store
            .new_index_file(INVERT_LIST_FILE, schema.clone())
            .await?;

        Ok(Self {
            existing_tokens,
            posting_lists: HashMap::new(),
            schema,
            tmpdir,
            store,
            writer,
            token_offsets: HashMap::new(),
        })
    }

    fn has_position(&self) -> bool {
        self.schema.column_with_name(POSITION_COL).is_some()
    }

    async fn add(&mut self, tokens: Vec<(String, i32)>, row_id: u64) -> Result<()> {
        let mut token_occurrences = HashMap::new();
        for (token, position) in tokens {
            token_occurrences
                .entry(token)
                .and_modify(|positions: &mut Vec<i32>| positions.push(position))
                .or_insert_with(|| vec![position]);
        }

        let with_position = self.has_position();
        token_occurrences
            .into_iter()
            .for_each(|(token, term_positions)| {
                self.posting_lists
                    .entry(token)
                    .and_modify(|list| {
                        list.add(row_id, term_positions);
                    })
                    .or_insert_with(|| PostingListBuilder::empty(with_position));
            });

        if self.posting_lists.deep_size_of() > *FLUSH_THRESHOLD * 1024 * 1024 {
            self.flush().await?;
        }

        Ok(())
    }

    async fn flush(&mut self) -> Result<()> {
        if self.posting_lists.is_empty() {
            return Ok(());
        }

        let posting_lists = std::mem::take(&mut self.posting_lists);
        for (token, list) in posting_lists {
            let (batch, _) = list.to_batch(self.schema.clone(), None)?;
            let length = batch.num_rows();
            let offset = self.writer.write_record_batch(batch).await? as usize;
            self.token_offsets
                .entry(token)
                .and_modify(|offsets| offsets.push((offset, length)))
                .or_insert_with(|| vec![(offset, length)]);
        }

        Ok(())
    }

    async fn into_reader(
        mut self,
        inverted_list: Option<Arc<InvertedListReader>>,
    ) -> Result<PostingReader> {
        self.flush().await?;
        self.writer.finish().await?;
        Ok(PostingReader::new(
            Some(self.tmpdir),
            self.existing_tokens,
            inverted_list,
            self.store,
            self.token_offsets,
        )
        .await?)
    }
}

pub(crate) struct PostingReader {
    tmpdir: Option<TempDir>,
    existing_tokens: HashMap<String, u32>,
    tokens: Vec<String>,
    inverted_list_reader: Option<Arc<InvertedListReader>>,
    store: Arc<dyn IndexStore>,
    reader: Arc<dyn IndexReader>,
    token_offsets: HashMap<String, Vec<(usize, usize)>>,
    token_idx: usize,
}

impl Debug for PostingReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostingReader")
            .field("tmpdir", &self.tmpdir)
            .field("tokens", &self.tokens)
            .field("token_offsets", &self.token_offsets)
            .field("token_idx", &self.token_idx)
            .finish()
    }
}

impl DeepSizeOf for PostingReader {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.tokens.deep_size_of_children(context)
            + self.store.deep_size_of_children(context)
            + self.token_offsets.deep_size_of_children(context)
            + self.token_idx.deep_size_of_children(context)
    }
}

impl PostingReader {
    async fn new(
        tmpdir: Option<TempDir>,
        existing_tokens: HashMap<String, u32>,
        inverted_list_reader: Option<Arc<InvertedListReader>>,
        store: Arc<dyn IndexStore>,
        token_offsets: HashMap<String, Vec<(usize, usize)>>,
    ) -> Result<Self> {
        let reader = store.open_index_file(INVERT_LIST_FILE).await?;
        let tokens = token_offsets.keys().cloned().collect_vec();

        Ok(Self {
            tmpdir,
            existing_tokens,
            tokens,
            inverted_list_reader,
            store,
            reader,
            token_offsets,
            token_idx: 0,
        })
    }

    async fn next(&mut self) -> Result<Option<(String, Vec<RecordBatch>)>> {
        if self.token_idx >= self.tokens.len() {
            return Ok(None);
        }

        let token = std::mem::take(&mut self.tokens[self.token_idx]);
        let ranges = std::mem::take(self.token_offsets.get_mut(&token).unwrap());
        let mut batches = stream::iter(ranges)
            .zip(repeat_with(|| self.reader.clone()))
            .map(|((offset, length), reader)| async move {
                reader.read_range(offset..offset + length, None).await
            })
            .buffer_unordered(get_num_compute_intensive_cpus())
            .try_collect::<Vec<_>>()
            .await?;

        if let Some(inverted_list) = self.inverted_list_reader.as_ref() {
            let token_id = self.existing_tokens.get(&token).ok_or(Error::Index {
                message: format!("token {} not found", token),
                location: location!(),
            })?;
            let batch = inverted_list
                .posting_batch(*token_id, inverted_list.has_positions())
                .await?;
            batches.push(batch);
        }

        self.token_idx += 1;
        Ok(Some((token, batches)))
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
        docs: Option<Arc<DocSet>>,
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
        let batch = reader.read_range(0..reader.num_rows(), None).await?;
        for i in 0..offsets.len() {
            let offset = offsets[i];
            let next_offset = if i + 1 < offsets.len() {
                offsets[i + 1]
            } else {
                reader.num_rows()
            };
            let batch = batch.slice(offset, next_offset - offset);
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
        self.inverted_list.par_iter_mut().for_each_init(
            || (Vec::new(), Vec::new(), Vec::new()),
            |(new_row_ids, new_freqs, new_positions), list| {
                let mut new_positions = list.positions.as_ref().map(|_| new_positions);

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
                        Some(None) => {
                            // remove the row_id
                            // do nothing
                        }
                        None => {
                            new_row_ids.push(row_id);
                            new_freqs.push(freq);
                            if let Some(new_positions) = new_positions.as_mut() {
                                new_positions.push(positions.unwrap());
                            }
                        }
                    }
                }

                *list = PostingListBuilder::new(
                    std::mem::take(new_row_ids),
                    std::mem::take(new_freqs),
                    new_positions.map(std::mem::take),
                );
            },
        );
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

fn inverted_list_schema(with_position: bool) -> SchemaRef {
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
    Arc::new(arrow_schema::Schema::new(fields))
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

    use super::InvertedIndex;

    async fn create_index<Offset: arrow::array::OffsetSizeTrait>(
        with_position: bool,
    ) -> Arc<InvertedIndex> {
        let tempdir = tempfile::tempdir().unwrap();
        let index_dir = Path::from_filesystem_path(tempdir.path()).unwrap();
        let store = LanceIndexStore::new(ObjectStore::local(), index_dir, None);

        let params = super::InvertedIndexParams::default().with_position(with_position);
        let mut invert_index = super::InvertedIndexBuilder::new(params);
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

        super::InvertedIndex::load(Arc::new(store)).await.unwrap()
    }

    async fn test_inverted_index<Offset: arrow::array::OffsetSizeTrait>() {
        let invert_index = create_index::<Offset>(false).await;
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

        let row_ids = invert_index
            .search(&SargableQuery::FullTextSearch(
                FullTextSearchQuery::new("unknown null".to_owned()).limit(Some(3)),
            ))
            .await
            .unwrap();
        assert_eq!(row_ids.len(), Some(0));

        // test phrase query
        // for non-phrasal query, the order of the tokens doesn't matter
        // so there should be 4 documents that contain "database" or "lance"

        // we built the index without position, so the phrase query will not work
        let results = invert_index
            .search(&SargableQuery::FullTextSearch(
                FullTextSearchQuery::new("\"unknown null\"".to_owned()).limit(Some(3)),
            ))
            .await;
        assert!(results.unwrap_err().to_string().contains("position is not found but required for phrase queries, try recreating the index with position"));
        let results = invert_index
            .search(&SargableQuery::FullTextSearch(
                FullTextSearchQuery::new("\"lance database\"".to_owned()).limit(Some(10)),
            ))
            .await;
        assert!(results.unwrap_err().to_string().contains("position is not found but required for phrase queries, try recreating the index with position"));

        // recreate the index with position
        let invert_index = create_index::<Offset>(true).await;
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

        let row_ids = invert_index
            .search(&SargableQuery::FullTextSearch(
                FullTextSearchQuery::new("\"lance unknown\"".to_owned()).limit(Some(10)),
            ))
            .await
            .unwrap();
        assert_eq!(row_ids.len(), Some(0));

        let row_ids = invert_index
            .search(&SargableQuery::FullTextSearch(
                FullTextSearchQuery::new("\"unknown null\"".to_owned()).limit(Some(3)),
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
