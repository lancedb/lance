// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{DefaultHasher, Hasher};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crate::scalar::lance_format::LanceIndexStore;
use crate::scalar::{IndexReader, IndexStore, IndexWriter, InvertedIndexParams};
use crate::vector::graph::OrderedFloat;
use arrow::array::{ArrayBuilder, AsArray, Int32Builder, StringBuilder};
use arrow::datatypes;
use arrow_array::{Array, Int32Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{Field, Schema, SchemaRef};
use crossbeam_queue::ArrayQueue;
use datafusion::execution::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use futures::{stream, Stream, StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_arrow::iter_str_array;
use lance_core::cache::FileMetadataCache;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, CPU_RUNTIME};
use lance_core::{Error, Result, ROW_ID, ROW_ID_FIELD};
use lance_io::object_store::ObjectStore;
use lazy_static::lazy_static;
use object_store::path::Path;
use snafu::location;
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
    pub static ref LANCE_FTS_NUM_SHARDS: usize = std::env::var("LANCE_FTS_NUM_SHARDS")
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
        params: InvertedIndexParams,
        tokens: TokenSet,
        inverted_list: Arc<InvertedListReader>,
        docs: DocSet,
    ) -> Self {
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
        let start = std::time::Instant::now();
        self.update_index(new_data).await?;
        println!("FTS indexing documents elapsed {:?}", start.elapsed());

        let start = std::time::Instant::now();
        self.write(dest_store).await?;
        println!("FTS writing documents elapsed {:?}", start.elapsed());
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn update_index(&mut self, stream: SendableRecordBatchStream) -> Result<()> {
        let flatten_stream = stream.map(|batch| {
            let batch = batch?;
            let doc_col = batch.column(0);
            match doc_col.data_type() {
                datatypes::DataType::Utf8 | datatypes::DataType::LargeUtf8 => Ok(batch),
                datatypes::DataType::List(_)   => {
                    flatten_string_list::<i32>(&batch, doc_col)
                }
                datatypes::DataType::LargeList(_) => {
                    flatten_string_list::<i64>(&batch, doc_col)
                }
                _ => {
                   Err(Error::Index { message: format!("expect data type String, LargeString or List of String/LargeString, but got {}", doc_col.data_type()), location: location!() })
                }
            }
        });

        let num_workers = *LANCE_FTS_NUM_SHARDS;

        // init the token maps
        let token_map = match self.tokens.tokens {
            TokenMap::HashMap(ref tokens) => tokens.clone(),
            _ => unreachable!("tokens must be HashMap at indexing"),
        };

        // spawn `num_shards` workers to build the index,
        // this thread will consume the stream and send the tokens to the workers.
        let tokenizer = self.params.tokenizer_config.build()?;
        let (sender, mut receiver) = tokio::sync::mpsc::channel(num_workers);
        let mut workers = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let worker = IndexWorker::new(
                tokenizer.clone(),
                token_map.clone(),
                self.params.with_position,
            )
            .await?;
            workers.push(worker);
        }

        let indexing_task = tokio::spawn({
            async move {
                let mut wokrer_id = 0;
                while let Some(batch) = receiver.recv().await {
                    workers[wokrer_id].process_batch(batch).await?;
                    wokrer_id = (wokrer_id + 1) % num_workers;
                }
                Result::Ok(workers)
            }
        });
        let mut stream = flatten_stream
            .map(|batch| {
                let sender = sender.clone();
                async move {
                    let batch = batch?;
                    let num_rows = batch.num_rows();
                    sender.send(batch).await.expect("failed to send batch");
                    Result::Ok(num_rows)
                }
            })
            .buffer_unordered(8); // let it be faster than IO then it's enough
        log::info!("indexing FTS with {} workers", num_workers);

        let mut last_num_rows = 0;
        let mut total_num_rows = 0;
        let start = std::time::Instant::now();
        while let Some(num_rows) = stream.try_next().await? {
            total_num_rows += num_rows;
            if total_num_rows >= last_num_rows + 100_000 {
                log::debug!(
                    "indexed {} documents, elapsed: {:?}, speed: {}rows/s",
                    num_rows,
                    start.elapsed(),
                    num_rows as f32 / start.elapsed().as_secs_f32()
                );
                last_num_rows = total_num_rows;
            }
        }
        // drop the sender to stop receivers
        drop(stream);
        debug_assert_eq!(sender.strong_count(), 1);
        drop(sender);
        let duration = std::time::Instant::now() - start;
        println!("tokenize elapsed: {:?}", duration);

        // wait for the workers to finish
        let start = std::time::Instant::now();
        let workers = indexing_task.await??;
        let duration = std::time::Instant::now() - start;
        println!("wait workers indexing elapsed: {:?}", duration);

        // update doc set
        for worker in workers {
            for (row_id, num_tokens) in
                std::iter::zip(worker.row_ids.iter(), worker.doc_token_num.iter())
            {
                self.docs.add(*row_id, *num_tokens as u32);
            }
            self.posting_readers.push(worker.into_reader().await?);
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
            let mut stream = stream::iter(batches).buffered(get_num_compute_intensive_cpus());
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

        let mut offsets = Vec::new();
        let mut max_scores = Vec::new();
        let mut num_rows = 0;
        log::info!("writing {} posting lists", self.posting_readers.len());
        let mut posting_streams = Vec::with_capacity(self.posting_readers.len() + 1);
        for reader in std::mem::take(&mut self.posting_readers) {
            posting_streams.push(reader.into_stream().await?);
        }
        if let Some(inverted_list_reader) = self.inverted_list.as_ref() {}
        let mut merged_stream = stream::select_all(posting_streams);
        let mut last_num_rows = 0;
        self.tokens = TokenSet::default();
        let start = std::time::Instant::now();
        while let Some((token, batches)) = merged_stream.try_next().await? {
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
    tokenizer: tantivy::tokenizer::TextAnalyzer,
    existing_tokens: HashMap<String, u32>,
    posting_lists: HashMap<String, PostingListBuilder>,
    schema: SchemaRef,
    tmpdir: TempDir,
    store: Arc<dyn IndexStore>,
    writer: Box<dyn IndexWriter>,
    offset: usize,
    token_offsets: HashMap<String, Vec<(usize, usize)>>,
    row_ids: Vec<u64>,
    doc_token_num: Vec<usize>,
    estimated_size: usize,

    total_doc_length: usize,
    total_token_num: usize,
}

impl IndexWorker {
    async fn new(
        tokenizer: tantivy::tokenizer::TextAnalyzer,
        existing_tokens: HashMap<String, u32>,
        with_position: bool,
    ) -> Result<Self> {
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
            tokenizer,
            existing_tokens,
            posting_lists: HashMap::new(),
            schema,
            tmpdir,
            store,
            writer,
            offset: 0,
            token_offsets: HashMap::new(),
            row_ids: Vec::new(),
            doc_token_num: Vec::new(),
            estimated_size: 0,
            total_doc_length: 0,
            total_token_num: 0,
        })
    }

    fn has_position(&self) -> bool {
        self.schema.column_with_name(POSITION_COL).is_some()
    }

    async fn process_batch(&mut self, batch: RecordBatch) -> Result<()> {
        let with_position = self.has_position();

        let doc_col = batch.column(0);
        let doc_iter = iter_str_array(doc_col);
        let row_id_col = batch[ROW_ID].as_primitive::<datatypes::UInt64Type>();
        let docs = doc_iter
            .zip(row_id_col.values().iter())
            .filter_map(|(doc, row_id)| doc.map(|doc| (doc, *row_id)));

        for (doc, row_id) in docs {
            let mut token_occurrences = HashMap::with_capacity(self.predict_doc_token_num(doc));
            let mut token_num = 0;
            {
                let mut token_stream = self.tokenizer.token_stream(doc);
                while token_stream.advance() {
                    let token = token_stream.token_mut();
                    let token_text = std::mem::take(&mut token.text);
                    token_occurrences
                        .entry(token_text)
                        .or_insert_with(Vec::new)
                        .push(token.position as i32);
                    token_num += 1;
                }
            }
            self.row_ids.push(row_id);
            self.doc_token_num.push(token_num);
            self.total_token_num += token_num;
            self.total_doc_length += doc.len();

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
        }

        if self.estimated_size > *LANCE_FTS_FLUSH_THRESHOLD * 1024 * 1024 {
            self.flush(false).await?;
        }

        Ok(())
    }

    fn predict_doc_token_num(&self, doc: &str) -> usize {
        match self.total_doc_length {
            0 => doc.len() / 5,
            _ => self.total_token_num * doc.len() / self.total_doc_length,
        }
    }

    #[instrument(level = "debug", skip_all)]
    async fn flush(&mut self, flush_all: bool) -> Result<()> {
        if self.posting_lists.is_empty() {
            return Ok(());
        }

        let tokens = self
            .posting_lists
            .iter()
            .map(|(token, list)| (token, list.size()))
            .sorted_unstable_by_key(|(_, size)| *size)
            .map(|(token, _)| token)
            .rev()
            .cloned()
            .collect_vec();

        let mut flushed_size = 0;
        let mut count = 0;
        for key in tokens {
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

    async fn into_reader(mut self) -> Result<PostingReader> {
        self.flush(true).await?;
        self.writer.finish().await?;
        PostingReader::new(
            Some(self.tmpdir),
            self.existing_tokens,
            self.store,
            self.token_offsets,
        )
        .await
    }
}

pub struct PostingReader {
    tmpdir: Option<TempDir>,
    existing_tokens: HashMap<String, u32>,
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
        store: Arc<dyn IndexStore>,
        token_offsets: HashMap<String, Vec<(usize, usize)>>,
    ) -> Result<Self> {
        let reader = store.open_index_file(INVERT_LIST_FILE).await?;

        Ok(Self {
            tmpdir,
            existing_tokens,
            store,
            reader,
            token_offsets,
        })
    }

    // returns a stream of (token, batch, max_score)
    async fn into_stream(
        mut self,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<(String, Vec<RecordBatch>)>> + Send>>> {
        let io_parallelism = self.store.io_parallelism();
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
            async move {
                // read the posting lists from new data
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
                let batches = futures::future::try_join_all(batches).await?;

                Ok((token, batches))
            }
        });

        let stream = stream::iter(inverted_batches).buffer_unordered(io_parallelism);
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

fn flatten_string_list<Offset: arrow::array::OffsetSizeTrait>(
    batch: &RecordBatch,
    doc_col: &Arc<dyn Array>,
) -> Result<RecordBatch> {
    let docs = doc_col.as_list::<Offset>();
    let row_ids = batch[ROW_ID].as_primitive::<datatypes::UInt64Type>();

    let row_ids = row_ids
        .values()
        .iter()
        .zip(docs.iter())
        .flat_map(|(row_id, doc)| std::iter::repeat_n(*row_id, doc.map(|d| d.len()).unwrap_or(0)));

    let row_ids = Arc::new(UInt64Array::from_iter_values(row_ids));
    let docs = match docs.value_type() {
        datatypes::DataType::Utf8 | datatypes::DataType::LargeUtf8 => docs.values().clone(),
        _ => {
            return Err(Error::Index {
                message: format!(
                    "expect data type String or LargeString but got {}",
                    docs.value_type()
                ),
                location: location!(),
            });
        }
    };

    let schema = Schema::new(vec![
        Field::new(
            batch.schema().field(0).name(),
            docs.data_type().clone(),
            true,
        ),
        ROW_ID_FIELD.clone(),
    ]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![docs, row_ids])?;
    Ok(batch)
}
