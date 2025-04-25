// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;
use std::{fmt::Debug, sync::atomic::AtomicU64};

use crate::scalar::lance_format::LanceIndexStore;
use crate::scalar::{IndexReader, IndexStore};
use crate::vector::graph::OrderedFloat;
use arrow::array::AsArray;
use arrow::datatypes;
use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::{Field, Schema, SchemaRef};
use bitpacking::{BitPacker, BitPacker4x};
use datafusion::execution::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use futures::stream::BoxStream;
use futures::{stream, StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_arrow::iter_str_array;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{cache::FileMetadataCache, utils::tokio::spawn_cpu};
use lance_core::{Error, Result, ROW_ID, ROW_ID_FIELD};
use lance_io::object_store::ObjectStore;
use lazy_static::lazy_static;
use object_store::path::Path;
use snafu::location;
use tempfile::{tempdir, TempDir};
use tracing::instrument;

use super::{index::*, InvertedIndexParams};

// the number of elements in each block
// each block contains 128 row ids and 128 frequencies
// WARNING: changing this value will break the compatibility with existing indexes
pub const BLOCK_SIZE: usize = BitPacker4x::BLOCK_LEN;

lazy_static! {
    // the size threshold to trigger flush the posting lists while indexing FTS,
    // lower value will result in slower indexing and less memory usage.
    // This means the indexing process would eat up to `LANCE_FTS_FLUSH_THRESHOLD * num_cpus * 2` MiB of memory,
    // it's in 512GiB by default
    static ref LANCE_FTS_FLUSH_THRESHOLD: usize = std::env::var("LANCE_FTS_FLUSH_THRESHOLD")
        .unwrap_or_else(|_| "512".to_string())
        .parse()
        .expect("failed to parse LANCE_FTS_FLUSH_THRESHOLD");
    // the size of each flush, lower value will result in more frequent flushes, but better IO locality
    // when the `LANCE_FTS_FLUSH_THRESHOLD` is reached, the flush will be triggered,
    // and then flush posting lists until the size of the flushed posting lists reaches `LANCE_FTS_FLUSH_SIZE`
    // it's in 64MiB by default
    static ref LANCE_FTS_FLUSH_SIZE: usize = std::env::var("LANCE_FTS_FLUSH_SIZE")
        .unwrap_or_else(|_| "16".to_string())
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
    // the partition size limit in GiB (compressed format), larger for better performance,
    // but more memory footprint while indexing, this counts only posting lists size
    pub static ref LANCE_FTS_PARTITION_SIZE: usize = std::env::var("LANCE_FTS_PARTITION_SIZE")
        .unwrap_or_else(|_| "2".to_string())
        .parse()
        .expect("failed to parse LANCE_FTS_PARTITION_SIZE");
}

#[derive(Debug)]
pub struct InvertedIndexBuilder {
    params: InvertedIndexParams,
    partitions: Vec<u64>,

    tmpdir: TempDir,
    local_store: Arc<dyn IndexStore>,
}

impl InvertedIndexBuilder {
    pub fn new(params: InvertedIndexParams) -> Self {
        Self::from_existing_index(params, Vec::new())
    }

    pub fn from_existing_index(params: InvertedIndexParams, partitions: Vec<u64>) -> Self {
        let tmpdir = tempdir().unwrap();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            FileMetadataCache::no_cache(),
        ));
        Self {
            params,
            partitions,
            tmpdir,
            local_store: store,
        }
    }

    pub async fn update(
        &mut self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        self.update_index(new_data).await?;

        let start = std::time::Instant::now();
        self.write(dest_store).await?;
        log::info!("FTS writing documents elapsed {:?}", start.elapsed());
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

        // spawn `num_workers` workers to build the index,
        // this thread will consume the stream and send the tokens to the workers.
        let num_workers = 1;
        let tokenizer = self.params.build()?;
        let with_position = self.params.with_position;
        let next_id = self.partitions.iter().map(|id| id + 1).max().unwrap_or(0);
        let id_alloc = Arc::new(AtomicU64::new(next_id));
        let (sender, receiver) = async_channel::bounded(num_workers);
        let mut index_tasks = Vec::with_capacity(num_workers);
        let workers = futures::future::try_join_all((0..num_workers).map(|_| async {
            let worker = IndexWorker::new(
                self.local_store.clone(),
                tokenizer.clone(),
                with_position,
                id_alloc.clone(),
            )
            .await?;
            Result::Ok(worker)
        }))
        .await?;
        for worker in workers {
            let receiver = receiver.clone();
            let task = spawn_cpu(move || {
                let mut worker = worker;
                while let Ok(batch) = receiver.recv_blocking() {
                    worker.process_batch(batch)?;
                }
                Result::Ok(worker)
            });
            index_tasks.push(task);
        }

        let mut stream = flatten_stream.map(|batch| {
            let batch = batch?;
            let num_rows = batch.num_rows();
            sender.send_blocking(batch).expect("failed to send batch");
            Result::Ok(num_rows)
        });
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
        debug_assert_eq!(sender.sender_count(), 1);
        drop(sender);
        let duration = std::time::Instant::now() - start;
        log::info!("dispatching elapsed: {:?}", duration);

        // wait for the workers to finish
        let start = std::time::Instant::now();
        let workers = futures::future::try_join_all(index_tasks).await?;
        let duration = std::time::Instant::now() - start;
        log::info!("wait workers indexing elapsed: {:?}", duration);

        for worker in workers {
            self.partitions.extend(worker.finish().await?);
        }
        Ok(())
    }

    pub async fn remap(
        &mut self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        // no need to remap the TokenSet,
        // since no row_id is stored in the TokenSet
        // self.docs.remap(mapping);
        // if let Some(inverted_list) = self.inverted_list.as_ref() {
        //     let schema = legacy_inverted_list_schema(self.params.with_position);
        //     let mut writer = dest_store
        //         .new_index_file(INVERT_LIST_FILE, schema.clone())
        //         .await?;

        //     let docs = Arc::new(self.docs.clone());
        //     let batches = (0..self.tokens.next_id()).map(|token_id| {
        //         let inverted_list = inverted_list.clone();
        //         let docs = docs.clone();
        //         async move {
        //             let batch = inverted_list
        //                 .posting_batch(token_id, inverted_list.has_positions())
        //                 .await?;
        //             let mut posting_builder = PostingListBuilder::try_from_batches(&[batch]);
        //             posting_builder.remap(mapping);
        //             let (batch, max_score) = posting_builder.to_sorted_batch(Some(&docs))?;

        //             Result::Ok((batch, max_score))
        //         }
        //     });
        //     let mut stream = stream::iter(batches).buffered(get_num_compute_intensive_cpus());
        //     let mut offsets = Vec::new();
        //     let mut max_scores = Vec::new();
        //     let mut num_rows = 0;
        //     while let Some((batch, max_score)) = stream.try_next().await? {
        //         offsets.push(num_rows);
        //         max_scores.push(max_score);
        //         num_rows += batch.num_rows();
        //         writer.write_record_batch(batch).await?;
        //     }

        //     let metadata = HashMap::from_iter(vec![
        //         ("offsets".to_owned(), serde_json::to_string(&offsets)?),
        //         ("max_scores".to_owned(), serde_json::to_string(&max_scores)?),
        //     ]);
        //     writer.finish_with_metadata(metadata).await?;
        // }

        // self.write(dest_store).await?;
        Ok(())
    }

    async fn write(&self, dest_store: &dyn IndexStore) -> Result<()> {
        for &partition_id in &self.partitions {
            self.local_store
                .copy_index_file(&token_file_path(partition_id), dest_store)
                .await?;
            self.local_store
                .copy_index_file(&posting_file_path(partition_id), dest_store)
                .await?;
            self.local_store
                .copy_index_file(&doc_file_path(partition_id), dest_store)
                .await?;
        }
        let metadata = HashMap::from_iter(vec![
            (
                "partitions".to_owned(),
                serde_json::to_string(&self.partitions)?,
            ),
            ("params".to_owned(), serde_json::to_string(&self.params)?),
        ]);
        let mut writer = dest_store
            .new_index_file(METADATA_FILE, Arc::new(Schema::empty()))
            .await?;
        writer.finish_with_metadata(metadata).await?;
        Ok(())
    }
}

impl Default for InvertedIndexBuilder {
    fn default() -> Self {
        let params = InvertedIndexParams::default();
        Self::new(params)
    }
}

struct Metadata {}

// builder for single partition
#[derive(Debug)]
struct InnerBuilder {
    id: u64,
    tokens: TokenSet,
    posting_lists: Vec<PostingListBuilder>,
    docs: DocSet,
}

impl InnerBuilder {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            tokens: TokenSet::default(),
            posting_lists: Vec::new(),
            docs: DocSet::default(),
        }
    }

    pub async fn write(&mut self, store: &dyn IndexStore) -> Result<()> {
        self.write_posting_lists(store).await?;
        self.write_tokens(store).await?;
        self.write_docs(store).await?;
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn write_posting_lists(&mut self, store: &dyn IndexStore) -> Result<()> {
        let mut writer = store
            .new_index_file(
                &posting_file_path(self.id),
                legacy_inverted_list_schema(self.posting_lists[0].has_positions()),
            )
            .await?;
        let posting_lists = std::mem::take(&mut self.posting_lists);
        for posting_list in posting_lists {
            let batch = posting_list.to_batch(Some(&self.docs))?;
            writer.write_record_batch(batch).await?;
        }
        writer.finish().await?;

        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn write_tokens(&mut self, store: &dyn IndexStore) -> Result<()> {
        let tokens = std::mem::take(&mut self.tokens);
        let batch = tokens.to_batch()?;
        let mut writer = store
            .new_index_file(&token_file_path(self.id), batch.schema())
            .await?;
        writer.write_record_batch(batch).await?;
        writer.finish().await?;
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn write_docs(&mut self, store: &dyn IndexStore) -> Result<()> {
        let docs = Arc::new(std::mem::take(&mut self.docs));
        let batch = docs.to_batch()?;
        let mut writer = store
            .new_index_file(&doc_file_path(self.id), batch.schema())
            .await?;
        writer.write_record_batch(batch).await?;
        writer.finish().await?;
        Ok(())
    }
}

struct IndexWorker {
    tokenizer: tantivy::tokenizer::TextAnalyzer,
    id_alloc: Arc<AtomicU64>,
    builder: InnerBuilder,
    partitions: Vec<u64>,
    flush_channel: tokio::sync::mpsc::Sender<InnerBuilder>,
    flush_task: tokio::task::JoinHandle<Result<()>>,
    schema: SchemaRef,
    estimated_size: usize,
    total_doc_length: usize,
}

impl IndexWorker {
    async fn new(
        store: Arc<dyn IndexStore>,
        tokenizer: tantivy::tokenizer::TextAnalyzer,
        with_position: bool,
        id_alloc: Arc<AtomicU64>,
    ) -> Result<Self> {
        let schema = legacy_inverted_list_schema(with_position);

        let (sender, receiver) = tokio::sync::mpsc::channel::<InnerBuilder>(1);
        let flush_task = tokio::spawn(async move {
            let mut receiver = receiver;
            let store = store;
            while let Some(mut inner) = receiver.recv().await {
                inner.write(store.as_ref()).await?;
            }
            Result::Ok(())
        });

        Ok(Self {
            tokenizer,
            builder: InnerBuilder::new(id_alloc.fetch_add(1, std::sync::atomic::Ordering::Relaxed)),
            partitions: Vec::new(),
            id_alloc,
            flush_channel: sender,
            flush_task,
            schema,
            estimated_size: 0,
            total_doc_length: 0,
        })
    }

    fn has_position(&self) -> bool {
        self.schema.column_with_name(POSITION_COL).is_some()
    }

    fn process_batch(&mut self, batch: RecordBatch) -> Result<()> {
        let doc_col = batch.column(0);
        let doc_iter = iter_str_array(doc_col);
        let row_id_col = batch[ROW_ID].as_primitive::<datatypes::UInt64Type>();
        let docs = doc_iter
            .zip(row_id_col.values().iter())
            .filter_map(|(doc, row_id)| doc.map(|doc| (doc, *row_id)));

        let with_position = self.has_position();
        for (doc, row_id) in docs {
            let mut token_occurrences =
                vec![PositionRecorder::new(with_position); self.builder.tokens.len()];
            let mut token_num = 0;
            {
                let mut token_stream = self.tokenizer.token_stream(doc);
                while token_stream.advance() {
                    let token = token_stream.token_mut();
                    let token_text = std::mem::take(&mut token.text);
                    let token_id = self.builder.tokens.add(token_text) as usize;
                    if token_id >= token_occurrences.len() {
                        // this is a new token
                        token_occurrences.push(PositionRecorder::new(with_position));
                    }

                    token_occurrences[token_id].push(token.position as u32);
                    token_num += 1;
                }
            }
            self.builder
                .posting_lists
                .resize_with(self.builder.tokens.len(), || {
                    PostingListBuilder::empty(with_position)
                });
            let doc_id = self.builder.docs.append(row_id, token_num);
            self.total_doc_length += doc.len();

            token_occurrences
                .into_iter()
                .enumerate()
                .for_each(|(token_id, term_positions)| {
                    let posting_list = &mut self.builder.posting_lists[token_id];

                    let old_size = posting_list.size();
                    posting_list.add(doc_id, term_positions);
                    let new_size = posting_list.size();
                    self.estimated_size += new_size - old_size;
                });
        }

        if self.estimated_size > *LANCE_FTS_FLUSH_THRESHOLD * 1024 * 1024 {
            self.flush()?;
        }

        Ok(())
    }

    fn predict_doc_token_num(&self, doc: &str) -> usize {
        match self.total_doc_length {
            0 => doc.len() / 5,
            _ => self.builder.docs.total_tokens_num() as usize * doc.len() / self.total_doc_length,
        }
    }

    #[instrument(level = "debug", skip_all)]
    fn flush(&mut self) -> Result<()> {
        if self.builder.tokens.len() == 0 {
            return Ok(());
        }

        self.estimated_size = 0;
        let builder = std::mem::replace(&mut self.builder, InnerBuilder::new(0));
        self.partitions.push(builder.id);
        self.flush_channel
            .blocking_send(builder)
            .map_err(|e| Error::Index {
                message: format!("failed to send posting list: {}", e),
                location: location!(),
            })?;

        Ok(())
    }

    async fn finish(mut self) -> Result<Vec<u64>> {
        if self.builder.tokens.len() > 0 {
            self.partitions.push(self.builder.id);
            self.flush_channel
                .send(self.builder)
                .await
                .map_err(|e| Error::Index {
                    message: format!("failed to send posting list: {}", e),
                    location: location!(),
                })?;
        }
        std::mem::drop(self.flush_channel);
        self.flush_task.await??;
        Ok(self.partitions)
    }
}

#[derive(Debug, Clone)]
pub enum PositionRecorder {
    Position(Vec<u32>),
    Count(usize),
}

impl PositionRecorder {
    fn new(with_position: bool) -> Self {
        if with_position {
            Self::Position(Vec::new())
        } else {
            Self::Count(0)
        }
    }

    fn push(&mut self, position: u32) {
        match self {
            Self::Position(positions) => positions.push(position),
            Self::Count(count) => *count += 1,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Position(positions) => positions.len(),
            Self::Count(count) => *count,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn into_vec(self) -> Vec<u32> {
        match self {
            Self::Position(positions) => positions,
            Self::Count(_) => vec![0],
        }
    }
}

pub struct PostingReader {
    tmpdir: Option<TempDir>,
    store: Arc<dyn IndexStore>,
    reader: Arc<dyn IndexReader>,
    token_offsets: Vec<(String, Vec<(usize, usize)>)>,
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
        store: Arc<dyn IndexStore>,
        token_offsets: impl IntoIterator<Item = (String, Vec<(usize, usize)>)>,
    ) -> Result<Self> {
        let reader = store.open_index_file(INVERT_LIST_FILE).await?;

        Ok(Self {
            tmpdir,
            store,
            reader,
            token_offsets: token_offsets
                .into_iter()
                .sorted_unstable_by(|(a, _), (b, _)| a.cmp(b))
                .collect_vec(),
        })
    }

    // returns a stream of (token, batch)
    fn into_stream(mut self) -> BoxStream<'static, Result<(String, Vec<RecordBatch>)>> {
        let io_parallelism = self.store.io_parallelism();
        let token_offsets = std::mem::take(&mut self.token_offsets);
        let schema: Arc<arrow_schema::Schema> = Arc::new(self.reader.schema().into());
        let posting_reader = Arc::new(self);

        let inverted_batches = token_offsets.into_iter().map(move |(token, offsets)| {
            let posting_reader = posting_reader.clone();
            let schema = schema.clone();
            async move {
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
        Box::pin(stream)
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

pub fn legacy_inverted_list_schema(with_position: bool) -> SchemaRef {
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

pub fn inverted_list_schema(with_position: bool) -> SchemaRef {
    let mut fields = vec![
        // we compress the posting lists (including row ids and frequencies),
        // and store the compressed posting lists, so it's a large binary array
        arrow_schema::Field::new(
            POSTING_COL,
            datatypes::DataType::List(Arc::new(Field::new(
                "item",
                datatypes::DataType::LargeBinary,
                true,
            ))),
            false,
        ),
    ];
    if with_position {
        fields.push(arrow_schema::Field::new(
            COMPRESSED_POSITION_COL,
            arrow_schema::DataType::List(Arc::new(arrow_schema::Field::new(
                "item",
                arrow_schema::DataType::Binary,
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

pub(crate) fn token_file_path(partition_id: u64) -> String {
    format!("part_{}_{}", partition_id, TOKENS_FILE)
}

pub(crate) fn posting_file_path(partition_id: u64) -> String {
    format!("part_{}_{}", partition_id, INVERT_LIST_FILE)
}

pub(crate) fn doc_file_path(partition_id: u64) -> String {
    format!("part_{}_{}", partition_id, DOCS_FILE)
}
