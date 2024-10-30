// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{DefaultHasher, Hasher};
use std::pin::Pin;
use std::sync::Arc;

use crate::scalar::lance_format::LanceIndexStore;
use crate::scalar::{IndexReader, IndexStore, IndexWriter, InvertedIndexParams};
use crate::vector::graph::OrderedFloat;
use arrow::array::{ArrayBuilder, AsArray, Int32Builder, StringBuilder};
use arrow::datatypes;
use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::SchemaRef;
use crossbeam_queue::ArrayQueue;
use datafusion::execution::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use futures::{stream, Stream, StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_arrow::iter_str_array;
use lance_core::cache::FileMetadataCache;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, CPU_RUNTIME};
use lance_core::{Error, Result, ROW_ID};
use lance_io::object_store::ObjectStore;
use lazy_static::lazy_static;
use object_store::path::Path;
use snafu::{location, Location};
use tempfile::{tempdir, TempDir};
use tracing::instrument;

use super::index::*;

lazy_static! {
    // the size threshold to trigger flush the posting lists while indexing FTS,
    // lower value will result in slower indexing and less memory usage
    // it's in 256MiB by default
    static ref LANCE_FTS_FLUSH_THRESHOLD: usize = std::env::var("LANCE_FTS_FLUSH_THRESHOLD")
        .unwrap_or_else(|_| "256".to_string())
        .parse()
        .expect("failed to parse LANCE_FTS_FLUSH_THRESHOLD");
    // the size of each flush, lower value will result in more frequent flushes, but better IO locality
    // when the `LANCE_FTS_FLUSH_THRESHOLD` is reached, the flush will be triggered,
    // and then flush posting lists until the size of the flushed posting lists reaches `LANCE_FTS_FLUSH_SIZE`
    // it's in 64MiB by default
    static ref LANCE_FTS_FLUSH_SIZE: usize = std::env::var("LANCE_FTS_FLUSH_SIZE")
        .unwrap_or_else(|_| "64".to_string())
        .parse()
        .expect("failed to parse LANCE_FTS_FLUSH_SIZE");
    // the number of shards to split the indexing work,
    // the indexing process would spawn `LANCE_FTS_NUM_SHARDS` workers to build FTS,
    // higher value will result in better parallelism, but more memory usage,
    // it doesn't mean higher value will result in better performance,
    // because the bottleneck can be the IO once the number of shards is large enough,
    // it's 8 by default
    static ref LANCE_FTS_NUM_SHARDS: usize = std::env::var("LANCE_FTS_NUM_SHARDS")
        .unwrap_or_else(|_| "8".to_string())
        .parse()
        .expect("failed to parse LANCE_FTS_NUM_SHARDS");
}

#[derive(Debug, Default, DeepSizeOf)]
pub struct InvertedIndexBuilder {
    params: InvertedIndexParams,

    tokens: TokenSet,
    inverted_list: Option<Arc<InvertedListReader>>,
    docs: DocSet,

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
        self.write(dest_store).await?;
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn update_index(&mut self, stream: SendableRecordBatchStream) -> Result<()> {
        let num_shards = *LANCE_FTS_NUM_SHARDS;

        // init the token maps
        let mut token_maps = vec![HashMap::new(); num_shards];
        for (token, token_id) in self.tokens.tokens.iter() {
            let mut hasher = DefaultHasher::new();
            hasher.write(token.as_bytes());
            let shard = hasher.finish() as usize % num_shards;
            token_maps[shard].insert(token.clone(), *token_id);
        }

        // spawn `num_shards` workers to build the index,
        // this thread will consume the stream and send the tokens to the workers,
        // because the workers can be CPU intensive, we need to limit concurrency the consuming stream,
        // so it's `num_cpus - num_shards`, and higher than `num_shards` may not help much, so we also limit it to `num_shards`
        let buffer_size = get_num_compute_intensive_cpus()
            .saturating_sub(num_shards)
            .max(1)
            .min(num_shards);
        let mut result_futs = Vec::with_capacity(num_shards);
        let mut senders = Vec::with_capacity(num_shards);
        for token_map in token_maps.into_iter() {
            let inverted_list = self.inverted_list.clone();
            let mut worker = IndexWorker::new(token_map, self.params.with_position).await?;

            let (sender, mut receiver) = tokio::sync::mpsc::channel(buffer_size);
            senders.push(sender);
            result_futs.push(tokio::spawn({
                async move {
                    while let Some((row_id, tokens, positions)) = receiver.recv().await {
                        worker.add(row_id, tokens, positions).await?;
                    }
                    let reader = worker.into_reader(inverted_list).await?;
                    Result::Ok(reader)
                }
            }));
        }

        let start = std::time::Instant::now();
        let senders = Arc::new(senders);
        let tokenizer_pool = Arc::new(ArrayQueue::new(num_shards));
        let tokenizer = self.params.tokenizer_config.build()?;
        for _ in 0..num_shards {
            let _ = tokenizer_pool.push(tokenizer.clone());
        }
        let mut stream = stream
            .map(move |batch| {
                let senders = senders.clone();
                let tokenizer_pool = tokenizer_pool.clone();
                CPU_RUNTIME.spawn_blocking(move || {
                    let batch = batch?;
                    let doc_iter = iter_str_array(batch.column(0));
                    let row_id_col = batch[ROW_ID].as_primitive::<datatypes::UInt64Type>();
                    let docs = doc_iter
                        .zip(row_id_col.values().iter())
                        .filter_map(|(doc, row_id)| doc.map(|doc| (doc, *row_id)));

                    let mut tokenizer = tokenizer_pool.pop().unwrap();

                    let num_tokens = docs
                        .map(|(doc, row_id)| {
                            // tokenize the document
                            let predicted_num_tokens = doc.len() / 5 / num_shards;
                            let mut token_buffers = std::iter::repeat_with(|| {
                                (
                                    StringBuilder::with_capacity(
                                        predicted_num_tokens,
                                        doc.len() / num_shards,
                                    ),
                                    Int32Builder::with_capacity(predicted_num_tokens),
                                )
                            })
                            .take(num_shards)
                            .collect_vec();
                            let mut num_tokens = 0;
                            let mut token_stream = tokenizer.token_stream(doc);
                            while token_stream.advance() {
                                let token = token_stream.token_mut();
                                let mut hasher = DefaultHasher::new();
                                hasher.write(token.text.as_bytes());
                                let shard = hasher.finish() as usize % num_shards;
                                let (ref mut token_builder, ref mut position_builder) =
                                    &mut token_buffers[shard];
                                token_builder.append_value(&token.text);
                                position_builder.append_value(token.position as i32);
                                num_tokens += 1;
                            }

                            for (shard, (token_builder, position_builder)) in
                                token_buffers.iter_mut().enumerate()
                            {
                                if token_builder.is_empty() {
                                    continue;
                                }

                                let tokens = token_builder.finish();
                                let positions = position_builder.finish();
                                senders[shard]
                                    .blocking_send((row_id, tokens, positions))
                                    .unwrap();
                            }

                            (row_id, num_tokens)
                        })
                        .collect_vec();

                    let _ = tokenizer_pool.push(tokenizer);
                    Result::Ok(num_tokens)
                })
            })
            .buffer_unordered(buffer_size);
        log::info!(
            "indexing FTS with {} shards and {} buffer concurrency ",
            num_shards,
            buffer_size,
        );

        let mut last_num_rows = 0;
        while let Some(num_tokens) = stream.try_next().await? {
            let num_tokens = num_tokens?;
            for (row_id, num_tokens) in num_tokens {
                self.docs.add(row_id, num_tokens);
            }

            if self.docs.len() >= last_num_rows + 100_000 {
                log::debug!(
                    "indexed {} documents, elapsed: {:?}, speed: {}rows/s",
                    self.docs.len(),
                    start.elapsed(),
                    self.docs.len() as f32 / start.elapsed().as_secs_f32()
                );
                last_num_rows = self.docs.len();
            }
        }
        // drop the stream to stop receivers
        drop(stream);

        // wait for the workers to finish
        for result in result_futs {
            let result = result.await??;
            self.posting_readers.push(result);
        }

        log::info!("FTS indexing documents elapsed {:?}", start.elapsed());

        Ok(())
    }

    pub async fn remap(
        &mut self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        // no need to remap the TokenSet,
        // since no row_id is stored in the TokenSet
        self.docs.remap(mapping);
        if let Some(inverted_list) = self.inverted_list.as_ref() {
            let schema = inverted_list_schema(self.params.with_position);
            let mut writer = dest_store
                .new_index_file(INVERT_LIST_FILE, schema.clone())
                .await?;

            let docs = Arc::new(self.docs.clone());
            let batches = (0..self.tokens.next_id()).map(|token_id| {
                let inverted_list = inverted_list.clone();
                let docs = docs.clone();
                async move {
                    let batch = inverted_list
                        .posting_batch(token_id, inverted_list.has_positions())
                        .await?;
                    let mut posting_builder = PostingListBuilder::from_batches(&[batch]);
                    posting_builder.remap(mapping);
                    let (batch, max_score) = posting_builder.to_batch(Some(docs))?;

                    Result::Ok((batch, max_score))
                }
            });
            let mut stream =
                stream::iter(batches).buffer_unordered(get_num_compute_intensive_cpus());
            let mut offsets = Vec::new();
            let mut max_scores = Vec::new();
            let mut num_rows = 0;
            while let Some((batch, max_score)) = stream.try_next().await? {
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
        }

        self.write(dest_store).await?;
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn write(&mut self, dest_store: &dyn IndexStore) -> Result<()> {
        self.write_posting_lists(dest_store).await?;
        self.write_tokens(dest_store).await?;
        self.write_docs(dest_store).await?;

        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn write_posting_lists(&mut self, store: &dyn IndexStore) -> Result<()> {
        if self.posting_readers.is_empty() {
            // it's a remap operation,
            // the inverted list is already written
            return Ok(());
        }

        let mut writer = store
            .new_index_file(
                INVERT_LIST_FILE,
                inverted_list_schema(self.params.with_position),
            )
            .await?;

        let docs = Arc::new(self.docs.clone());
        let mut offsets = Vec::new();
        let mut max_scores = Vec::new();
        let mut num_rows = 0;
        log::info!("writing {} posting lists", self.posting_readers.len());
        let mut posting_streams = Vec::with_capacity(self.posting_readers.len());
        for reader in std::mem::take(&mut self.posting_readers) {
            posting_streams.push(reader.stream(docs.clone()).await?);
        }
        let mut merged_stream = stream::select_all(posting_streams);
        let mut last_num_rows = 0;
        let start = std::time::Instant::now();
        while let Some(r) = merged_stream.try_next().await? {
            let (token, batch, max_score) = r?;
            self.tokens.add(token);
            offsets.push(num_rows);
            max_scores.push(max_score);
            num_rows += batch.num_rows();
            writer.write_record_batch(batch).await?;
            if num_rows > last_num_rows + 100_000 {
                log::debug!("written {} rows, elapsed: {:?}", num_rows, start.elapsed());
                last_num_rows = num_rows;
            }
        }

        let metadata = HashMap::from_iter(vec![
            ("offsets".to_owned(), serde_json::to_string(&offsets)?),
            ("max_scores".to_owned(), serde_json::to_string(&max_scores)?),
        ]);
        writer.finish_with_metadata(metadata).await?;
        log::info!(
            "finished writing posting lists, elapsed: {:?}",
            start.elapsed()
        );

        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn write_tokens(&mut self, store: &dyn IndexStore) -> Result<()> {
        log::info!("writing tokens");

        let tokens = std::mem::take(&mut self.tokens);
        let batch = tokens.to_batch()?;
        let mut writer = store.new_index_file(TOKENS_FILE, batch.schema()).await?;
        writer.write_record_batch(batch).await?;

        let tokenizer = serde_json::to_string(&self.params.tokenizer_config)?;
        let metadata = HashMap::from_iter(vec![("tokenizer".to_owned(), tokenizer)]);
        writer.finish_with_metadata(metadata).await?;

        log::info!("finished writing tokens");
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn write_docs(&mut self, store: &dyn IndexStore) -> Result<()> {
        log::info!("writing docs");

        let docs = Arc::new(std::mem::take(&mut self.docs));
        let batch = docs.to_batch()?;
        let mut writer = store.new_index_file(DOCS_FILE, batch.schema()).await?;
        writer.write_record_batch(batch).await?;
        writer.finish().await?;

        log::info!("finished writing docs");
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
    offset: usize,
    token_offsets: HashMap<String, Vec<(usize, usize)>>,
    estimated_size: usize,
}

impl IndexWorker {
    async fn new(existing_tokens: HashMap<String, u32>, with_position: bool) -> Result<Self> {
        let tmpdir = tempdir()?;
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local(),
            Path::from_filesystem_path(tmpdir.path())?,
            FileMetadataCache::no_cache(),
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
            offset: 0,
            token_offsets: HashMap::new(),
            estimated_size: 0,
        })
    }

    fn has_position(&self) -> bool {
        self.schema.column_with_name(POSITION_COL).is_some()
    }

    async fn add(&mut self, row_id: u64, tokens: StringArray, positions: Int32Array) -> Result<()> {
        let mut token_occurrences = HashMap::new();
        for (token, position) in tokens.iter().zip(positions.values().into_iter()) {
            let Some(token) = token else {
                continue;
            };
            token_occurrences
                .entry(token)
                .or_insert_with(Vec::new)
                .push(*position);
        }
        let with_position = self.has_position();
        token_occurrences
            .into_iter()
            .for_each(|(token, term_positions)| {
                let posting_list = self
                    .posting_lists
                    .entry(token.to_owned())
                    .or_insert_with(|| PostingListBuilder::empty(with_position));

                let old_size = if posting_list.is_empty() {
                    0
                } else {
                    posting_list.size()
                };
                posting_list.add(row_id, term_positions);
                let new_size = posting_list.size();
                self.estimated_size += new_size - old_size;
            });

        if self.estimated_size > *LANCE_FTS_FLUSH_THRESHOLD * 1024 * 1024 {
            self.flush(false).await?;
        }

        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn flush(&mut self, flush_all: bool) -> Result<()> {
        if self.posting_lists.is_empty() {
            return Ok(());
        }

        let keys = self
            .posting_lists
            .iter()
            .map(|(key, list)| (key, list.size()))
            .sorted_unstable_by_key(|(_, size)| *size)
            .map(|(key, _)| key)
            .rev()
            .cloned()
            .collect_vec();

        let mut flushed_size = 0;
        let mut count = 0;
        for key in keys {
            flushed_size += self.flush_posting_list(key).await?;
            count += 1;
            if !flush_all && flushed_size >= *LANCE_FTS_FLUSH_SIZE * 1024 * 1024 {
                break;
            }
        }
        log::debug!(
            "flushed {} lists of {}MiB, posting_lists num: {}, posting_lists size: {}MiB",
            count,
            flushed_size / 1024 / 1024,
            self.posting_lists.len(),
            self.posting_lists.deep_size_of() / 1024 / 1024,
        );

        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn flush_posting_list(&mut self, token: String) -> Result<usize> {
        if let Some(posting_list) = self.posting_lists.remove(&token) {
            let size = posting_list.size();
            self.estimated_size -= size;
            let (batch, _) = posting_list.to_batch(None)?;
            let length = batch.num_rows();
            assert!(length > 0);
            self.writer.write_record_batch(batch).await?;
            self.token_offsets
                .entry(token)
                .or_default()
                .push((self.offset, length));
            self.offset += length;
            return Ok(size);
        }

        Ok(0)
    }

    async fn into_reader(
        mut self,
        inverted_list: Option<Arc<InvertedListReader>>,
    ) -> Result<PostingReader> {
        self.flush(true).await?;
        self.writer.finish().await?;
        PostingReader::new(
            Some(self.tmpdir),
            self.existing_tokens,
            inverted_list,
            self.store,
            self.token_offsets,
        )
        .await
    }
}

pub struct PostingReader {
    tmpdir: Option<TempDir>,
    existing_tokens: HashMap<String, u32>,
    inverted_list_reader: Option<Arc<InvertedListReader>>,
    store: Arc<dyn IndexStore>,
    reader: Arc<dyn IndexReader>,
    token_offsets: HashMap<String, Vec<(usize, usize)>>,
}

impl Debug for PostingReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostingReader")
            .field("tmpdir", &self.tmpdir)
            .field("token_offsets", &self.token_offsets)
            .finish()
    }
}

impl DeepSizeOf for PostingReader {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.store.deep_size_of_children(context)
            + self.token_offsets.deep_size_of_children(context)
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

        Ok(Self {
            tmpdir,
            existing_tokens,
            inverted_list_reader,
            store,
            reader,
            token_offsets,
        })
    }

    // returns a stream of (token, batch, max_score)
    async fn stream(
        mut self,
        docs: Arc<DocSet>,
    ) -> Result<
        Pin<
            Box<
                dyn Stream<
                        Item = std::result::Result<
                            Result<(String, RecordBatch, f32)>,
                            tokio::task::JoinError,
                        >,
                    > + Send,
            >,
        >,
    > {
        let mut token_offsets = std::mem::take(&mut self.token_offsets);
        for token in self.existing_tokens.keys() {
            if !token_offsets.contains_key(token) {
                token_offsets.insert(token.clone(), Vec::new());
            }
        }

        let schema: Arc<arrow_schema::Schema> = Arc::new(self.reader.schema().into());
        let posting_reader = Arc::new(self);

        let inverted_batches = token_offsets.into_iter().map(move |(token, offsets)| {
            let posting_reader = posting_reader.clone();
            let schema = schema.clone();
            let docs = docs.clone();
            tokio::spawn(async move {
                let batches = offsets.into_iter().map(|(offset, length)| {
                    let reader = posting_reader.reader.clone();
                    let schema = schema.clone();
                    async move {
                        if length == 0 {
                            Ok(RecordBatch::new_empty(schema))
                        } else {
                            reader.read_range(offset..offset + length, None).await
                        }
                    }
                });
                let mut batches = futures::future::try_join_all(batches).await?;

                if let Some(inverted_list) = posting_reader.inverted_list_reader.as_ref() {
                    let token_id =
                        posting_reader
                            .existing_tokens
                            .get(&token)
                            .ok_or(Error::Index {
                                message: format!("token {} not found", token),
                                location: location!(),
                            })?;
                    let batch = inverted_list
                        .posting_batch(*token_id, inverted_list.has_positions())
                        .await?;
                    batches.push(batch);
                }

                let (batch, max_score) =
                    PostingListBuilder::from_batches(&batches).to_batch(Some(docs))?;

                Ok((token, batch, max_score))
            })
        });

        let stream = stream::iter(inverted_batches)
            .buffer_unordered(get_num_compute_intensive_cpus().div_ceil(*LANCE_FTS_NUM_SHARDS));
        Ok(Box::pin(stream))
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

pub fn inverted_list_schema(with_position: bool) -> SchemaRef {
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
    use lance_core::cache::{CapacityMode, FileMetadataCache};
    use lance_core::ROW_ID_FIELD;
    use lance_io::object_store::ObjectStore;
    use object_store::path::Path;

    use crate::scalar::inverted::TokenizerConfig;
    use crate::scalar::lance_format::LanceIndexStore;
    use crate::scalar::{FullTextSearchQuery, SargableQuery, ScalarIndex};

    use super::InvertedIndex;

    async fn create_index<Offset: arrow::array::OffsetSizeTrait>(
        with_position: bool,
        tokenizer: TokenizerConfig,
    ) -> Arc<InvertedIndex> {
        let tempdir = tempfile::tempdir().unwrap();
        let index_dir = Path::from_filesystem_path(tempdir.path()).unwrap();
        let cache = FileMetadataCache::with_capacity(128 * 1024 * 1024, CapacityMode::Bytes);
        let store = LanceIndexStore::new(ObjectStore::local(), index_dir, cache);

        let mut params = super::InvertedIndexParams::default().with_position(with_position);
        params.tokenizer_config = tokenizer;
        let mut invert_index = super::InvertedIndexBuilder::new(params);
        let doc_col = GenericStringArray::<Offset>::from(vec![
            "lance database the search",
            "lance database",
            "lance search",
            "database search",
            "unrelated doc",
            "unrelated",
            "mots accentués",
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
        let invert_index = create_index::<Offset>(false, TokenizerConfig::default()).await;
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
        let invert_index = create_index::<Offset>(true, TokenizerConfig::default()).await;
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

    #[tokio::test]
    async fn test_accented_chars() {
        let invert_index = create_index::<i32>(false, TokenizerConfig::default()).await;
        let row_ids = invert_index
            .search(&SargableQuery::FullTextSearch(
                FullTextSearchQuery::new("accentués".to_owned()).limit(Some(3)),
            ))
            .await
            .unwrap();
        assert_eq!(row_ids.len(), Some(1));

        let row_ids = invert_index
            .search(&SargableQuery::FullTextSearch(
                FullTextSearchQuery::new("accentues".to_owned()).limit(Some(3)),
            ))
            .await
            .unwrap();
        assert_eq!(row_ids.len(), Some(0));

        // with ascii folding enabled, the search should be accent-insensitive
        let invert_index =
            create_index::<i32>(true, TokenizerConfig::default().ascii_folding(true)).await;
        let row_ids = invert_index
            .search(&SargableQuery::FullTextSearch(
                FullTextSearchQuery::new("accentués".to_owned()).limit(Some(3)),
            ))
            .await
            .unwrap();
        assert_eq!(row_ids.len(), Some(1));

        let row_ids = invert_index
            .search(&SargableQuery::FullTextSearch(
                FullTextSearchQuery::new("accentues".to_owned()).limit(Some(3)),
            ))
            .await
            .unwrap();
        assert_eq!(row_ids.len(), Some(1));
    }
}
