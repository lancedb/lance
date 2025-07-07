// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;
use std::{fmt::Debug, sync::atomic::AtomicU64};

use crate::scalar::lance_format::LanceIndexStore;
use crate::scalar::IndexStore;
use crate::vector::graph::OrderedFloat;
use arrow::datatypes;
use arrow::{array::AsArray, compute::concat_batches};
use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::{Field, Schema, SchemaRef};
use bitpacking::{BitPacker, BitPacker4x};
use datafusion::execution::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use futures::{stream, StreamExt, TryStreamExt};
use lance_arrow::iter_str_array;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{cache::LanceCache, utils::tokio::spawn_cpu};
use lance_core::{Error, Result, ROW_ID, ROW_ID_FIELD};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use snafu::location;
use std::sync::LazyLock;
use tempfile::{tempdir, TempDir};
use tracing::instrument;

use super::{
    index::*,
    merger::{Merger, SizeBasedMerger},
    InvertedIndexParams,
};

// the number of elements in each block
// each block contains 128 row ids and 128 frequencies
// WARNING: changing this value will break the compatibility with existing indexes
pub const BLOCK_SIZE: usize = BitPacker4x::BLOCK_LEN;

// the (compressed) size of each flush for posting lists in MiB,
// when the `LANCE_FTS_FLUSH_THRESHOLD` is reached, the flush will be triggered,
// higher for better indexing performance, but more memory usage,
// it's in 16 MiB by default
static LANCE_FTS_FLUSH_SIZE: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("LANCE_FTS_FLUSH_SIZE")
        .unwrap_or_else(|_| "16".to_string())
        .parse()
        .expect("failed to parse LANCE_FTS_FLUSH_SIZE")
});
// the number of shards to split the indexing work,
// the indexing process would spawn `LANCE_FTS_NUM_SHARDS` workers to build FTS,
// higher for faster indexing performance, but more memory usage,
// it's `the number of compute intensive CPUs` by default
pub static LANCE_FTS_NUM_SHARDS: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("LANCE_FTS_NUM_SHARDS")
        .unwrap_or_else(|_| get_num_compute_intensive_cpus().to_string())
        .parse()
        .expect("failed to parse LANCE_FTS_NUM_SHARDS")
});
// the partition size limit in MiB (uncompressed format)
// higher for better indexing & query performance, but more memory usage,
pub static LANCE_FTS_PARTITION_SIZE: LazyLock<u64> = LazyLock::new(|| {
    std::env::var("LANCE_FTS_PARTITION_SIZE")
        .unwrap_or_else(|_| "256".to_string())
        .parse()
        .expect("failed to parse LANCE_FTS_PARTITION_SIZE")
});
// the target size of partition after merging in MiB (uncompressed format)
pub static LANCE_FTS_TARGET_SIZE: LazyLock<u64> = LazyLock::new(|| {
    std::env::var("LANCE_FTS_TARGET_SIZE")
        .unwrap_or_else(|_| "4096".to_string())
        .parse()
        .expect("failed to parse LANCE_FTS_TARGET_SIZE")
});

#[derive(Debug)]
pub struct InvertedIndexBuilder {
    params: InvertedIndexParams,
    partitions: Vec<u64>,
    new_partitions: Vec<u64>,

    _tmpdir: TempDir,
    local_store: Arc<dyn IndexStore>,
    src_store: Arc<dyn IndexStore>,
}

impl InvertedIndexBuilder {
    pub fn new(params: InvertedIndexParams) -> Self {
        Self::from_existing_index(params, None, Vec::new())
    }

    pub fn from_existing_index(
        params: InvertedIndexParams,
        store: Option<Arc<dyn IndexStore>>,
        partitions: Vec<u64>,
    ) -> Self {
        let tmpdir = tempdir().unwrap();
        let local_store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));
        let src_store = store.unwrap_or_else(|| local_store.clone());
        Self {
            params,
            partitions,
            new_partitions: Vec::new(),
            _tmpdir: tmpdir,
            local_store,
            src_store,
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
        let tokenizer = self.params.build()?;
        let with_position = self.params.with_position;
        let next_id = self.partitions.iter().map(|id| id + 1).max().unwrap_or(0);
        let id_alloc = Arc::new(AtomicU64::new(next_id));
        let (sender, receiver) = async_channel::bounded(num_workers);
        let mut index_tasks = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let store = self.local_store.clone();
            let tokenizer = tokenizer.clone();
            let receiver = receiver.clone();
            let id_alloc = id_alloc.clone();
            let task = tokio::task::spawn(async move {
                let mut worker =
                    IndexWorker::new(store, tokenizer, with_position, id_alloc).await?;
                while let Ok(batch) = receiver.recv().await {
                    worker.process_batch(batch).await?;
                }
                let partitions = worker.finish().await?;
                Result::Ok(partitions)
            });
            index_tasks.push(task);
        }

        let sender = Arc::new(sender);

        let mut stream = Box::pin(flatten_stream.then({
            |batch_result| {
                let sender = sender.clone();
                async move {
                    let sender = sender.clone();
                    let batch = batch_result?;
                    let num_rows = batch.num_rows();
                    sender.send(batch).await.expect("failed to send batch");
                    Result::Ok(num_rows)
                }
            }
        }));
        log::info!("indexing FTS with {} workers", num_workers);

        let mut last_num_rows = 0;
        let mut total_num_rows = 0;
        let start = std::time::Instant::now();
        while let Some(num_rows) = stream.try_next().await? {
            total_num_rows += num_rows;
            if total_num_rows >= last_num_rows + 1_000_000 {
                log::debug!(
                    "indexed {} documents, elapsed: {:?}, speed: {}rows/s",
                    total_num_rows,
                    start.elapsed(),
                    total_num_rows as f32 / start.elapsed().as_secs_f32()
                );
                last_num_rows = total_num_rows;
            }
        }
        // drop the sender to stop receivers
        drop(stream);
        debug_assert_eq!(sender.sender_count(), 1);
        drop(sender);
        log::info!("dispatching elapsed: {:?}", start.elapsed());

        // wait for the workers to finish
        let start = std::time::Instant::now();
        for index_task in index_tasks {
            self.new_partitions.extend(index_task.await??);
        }
        log::info!("wait workers indexing elapsed: {:?}", start.elapsed());
        Ok(())
    }

    pub async fn remap(
        &mut self,
        mapping: &HashMap<u64, Option<u64>>,
        src_store: Arc<dyn IndexStore>,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        for part in self.partitions.iter() {
            let part = InvertedPartition::load(src_store.clone(), *part, None).await?;
            let mut builder = part.into_builder().await?;
            builder.remap(mapping).await?;
            builder.write(dest_store).await?;
        }
        self.write_metadata(dest_store, &self.partitions).await?;
        Ok(())
    }

    async fn write_metadata(&self, dest_store: &dyn IndexStore, partitions: &[u64]) -> Result<()> {
        let metadata = HashMap::from_iter(vec![
            ("partitions".to_owned(), serde_json::to_string(&partitions)?),
            ("params".to_owned(), serde_json::to_string(&self.params)?),
        ]);
        let mut writer = dest_store
            .new_index_file(METADATA_FILE, Arc::new(Schema::empty()))
            .await?;
        writer.finish_with_metadata(metadata).await?;
        Ok(())
    }

    async fn write(&self, dest_store: &dyn IndexStore) -> Result<()> {
        let partitions =
            futures::future::try_join_all(
                self.partitions
                    .iter()
                    .map(|part| InvertedPartition::load(self.src_store.clone(), *part, None))
                    .chain(self.new_partitions.iter().map(|part| {
                        InvertedPartition::load(self.local_store.clone(), *part, None)
                    })),
            )
            .await?;
        let mut merger = SizeBasedMerger::new(dest_store, partitions, *LANCE_FTS_TARGET_SIZE << 20);
        let partitions = merger.merge().await?;
        self.write_metadata(dest_store, &partitions).await?;
        Ok(())
    }
}

impl Default for InvertedIndexBuilder {
    fn default() -> Self {
        let params = InvertedIndexParams::default();
        Self::new(params)
    }
}

// builder for single partition
#[derive(Debug)]
pub struct InnerBuilder {
    id: u64,
    pub(crate) tokens: TokenSet,
    pub(crate) posting_lists: Vec<PostingListBuilder>,
    pub(crate) docs: DocSet,
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

    pub fn id(&self) -> u64 {
        self.id
    }

    pub async fn remap(&mut self, mapping: &HashMap<u64, Option<u64>>) -> Result<()> {
        // no need to remap the TokenSet,
        // no row_id is stored in the TokenSet
        let removed = self.docs.remap(mapping);
        for posting_list in self.posting_lists.iter_mut() {
            posting_list.remap(&removed);
        }
        Ok(())
    }

    pub async fn write(&mut self, store: &dyn IndexStore) -> Result<()> {
        let docs = Arc::new(std::mem::take(&mut self.docs));
        self.write_posting_lists(store, docs.clone()).await?;
        self.write_tokens(store).await?;
        self.write_docs(store, docs).await?;
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn write_posting_lists(
        &mut self,
        store: &dyn IndexStore,
        docs: Arc<DocSet>,
    ) -> Result<()> {
        let id = self.id;
        let mut writer = store
            .new_index_file(
                &posting_file_path(self.id),
                inverted_list_schema(self.posting_lists[0].has_positions()),
            )
            .await?;
        let posting_lists = std::mem::take(&mut self.posting_lists);

        log::info!(
            "writing {} posting lists of partition {}, with position {}",
            posting_lists.len(),
            id,
            posting_lists[0].has_positions()
        );
        let schema = inverted_list_schema(posting_lists[0].has_positions());

        let mut batches = stream::iter(posting_lists)
            .map(|posting_list| {
                let block_max_scores = docs.calculate_block_max_scores(
                    posting_list.doc_ids.iter(),
                    posting_list.frequencies.iter(),
                );
                spawn_cpu(move || posting_list.to_batch(block_max_scores))
            })
            .buffered(get_num_compute_intensive_cpus());

        let mut write_duration = std::time::Duration::ZERO;
        let mut num_posting_lists = 0;
        let mut buffer = Vec::new();
        let mut size_sum = 0;
        while let Some(batch) = batches.try_next().await? {
            num_posting_lists += 1;
            size_sum += batch.get_array_memory_size();
            buffer.push(batch);
            if size_sum >= *LANCE_FTS_FLUSH_SIZE << 20 {
                let batch = concat_batches(&schema, buffer.iter())?;
                buffer.clear();
                size_sum = 0;
                let start = std::time::Instant::now();
                writer.write_record_batch(batch).await?;
                write_duration += start.elapsed();
            }

            if num_posting_lists % 500_000 == 0 {
                log::info!(
                    "wrote {} posting lists of partition {}, writing elapsed: {:?}",
                    num_posting_lists,
                    id,
                    write_duration,
                );
            }
        }
        if !buffer.is_empty() {
            let batch = concat_batches(&schema, buffer.iter())?;
            writer.write_record_batch(batch).await?;
        }

        writer.finish().await?;
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn write_tokens(&mut self, store: &dyn IndexStore) -> Result<()> {
        log::info!("writing tokens of partition {}", self.id);
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
    async fn write_docs(&mut self, store: &dyn IndexStore, docs: Arc<DocSet>) -> Result<()> {
        log::info!("writing docs of partition {}", self.id);
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
    store: Arc<dyn IndexStore>,
    tokenizer: tantivy::tokenizer::TextAnalyzer,
    id_alloc: Arc<AtomicU64>,
    builder: InnerBuilder,
    partitions: Vec<u64>,
    schema: SchemaRef,
    estimated_size: u64,
    total_doc_length: usize,
}

impl IndexWorker {
    async fn new(
        store: Arc<dyn IndexStore>,
        tokenizer: tantivy::tokenizer::TextAnalyzer,
        with_position: bool,
        id_alloc: Arc<AtomicU64>,
    ) -> Result<Self> {
        let schema = inverted_list_schema(with_position);

        Ok(Self {
            store,
            tokenizer,
            builder: InnerBuilder::new(id_alloc.fetch_add(1, std::sync::atomic::Ordering::Relaxed)),
            partitions: Vec::new(),
            id_alloc,
            schema,
            estimated_size: 0,
            total_doc_length: 0,
        })
    }

    fn has_position(&self) -> bool {
        self.schema.column_with_name(POSITION_COL).is_some()
    }

    async fn process_batch(&mut self, batch: RecordBatch) -> Result<()> {
        let doc_col = batch.column(0);
        let doc_iter = iter_str_array(doc_col);
        let row_id_col = batch[ROW_ID].as_primitive::<datatypes::UInt64Type>();
        let docs = doc_iter
            .zip(row_id_col.values().iter())
            .filter_map(|(doc, row_id)| doc.map(|doc| (doc, *row_id)));

        let with_position = self.has_position();
        for (doc, row_id) in docs {
            let mut token_occurrences = HashMap::new();
            let mut token_num = 0;
            {
                let mut token_stream = self.tokenizer.token_stream(doc);
                while token_stream.advance() {
                    let token = token_stream.token_mut();
                    let token_text = std::mem::take(&mut token.text);
                    let token_id = self.builder.tokens.add(token_text) as usize;
                    token_occurrences
                        .entry(token_id as u32)
                        .or_insert_with(|| PositionRecorder::new(with_position))
                        .push(token.position as u32);
                    token_num += 1;
                }
            }
            self.builder
                .posting_lists
                .resize_with(self.builder.tokens.len(), || {
                    PostingListBuilder::new(with_position)
                });
            let doc_id = self.builder.docs.append(row_id, token_num);
            self.total_doc_length += doc.len();

            token_occurrences
                .into_iter()
                .for_each(|(token_id, term_positions)| {
                    let posting_list = &mut self.builder.posting_lists[token_id as usize];

                    let old_size = posting_list.size();
                    posting_list.add(doc_id, term_positions);
                    let new_size = posting_list.size();
                    self.estimated_size += new_size - old_size;
                });

            if self.builder.docs.len() as u32 == u32::MAX
                || self.estimated_size >= *LANCE_FTS_PARTITION_SIZE << 20
            {
                self.flush().await?;
            }
        }

        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn flush(&mut self) -> Result<()> {
        if self.builder.tokens.is_empty() {
            return Ok(());
        }

        log::info!(
            "flushing posting lists, estimated size: {} MiB",
            self.estimated_size / (1024 * 1024)
        );
        self.estimated_size = 0;
        let mut builder = std::mem::replace(
            &mut self.builder,
            InnerBuilder::new(
                self.id_alloc
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            ),
        );
        builder.write(self.store.as_ref()).await?;
        self.partitions.push(builder.id);

        Ok(())
    }

    async fn finish(mut self) -> Result<Vec<u64>> {
        if !self.builder.tokens.is_empty() {
            self.flush().await?;
        }
        Ok(self.partitions)
    }
}

#[derive(Debug, Clone)]
pub enum PositionRecorder {
    Position(Vec<u32>),
    Count(u32),
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

    pub fn len(&self) -> u32 {
        match self {
            Self::Position(positions) => positions.len() as u32,
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

#[derive(Debug, Eq, PartialEq, Clone, DeepSizeOf)]
pub struct ScoredDoc {
    pub row_id: u64,
    pub score: OrderedFloat,
}

impl ScoredDoc {
    pub fn new(row_id: u64, score: f32) -> Self {
        Self {
            row_id,
            score: OrderedFloat(score),
        }
    }
}

impl PartialOrd for ScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredDoc {
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
        arrow_schema::Field::new(MAX_SCORE_COL, datatypes::DataType::Float32, false),
        arrow_schema::Field::new(LENGTH_COL, datatypes::DataType::UInt32, false),
    ];
    if with_position {
        fields.push(arrow_schema::Field::new(
            POSITION_COL,
            arrow_schema::DataType::List(Arc::new(arrow_schema::Field::new(
                "item",
                arrow_schema::DataType::List(Arc::new(arrow_schema::Field::new(
                    "item",
                    arrow_schema::DataType::LargeBinary,
                    true,
                ))),
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
