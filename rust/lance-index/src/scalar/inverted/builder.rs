// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{InvertedIndexParams, index::*};
use crate::scalar::IndexStore;
use crate::scalar::inverted::json::JsonTextStream;
use crate::scalar::inverted::lance_tokenizer::DocType;
use crate::scalar::inverted::tokenizer::lance_tokenizer::LanceTokenizer;
#[cfg(test)]
use crate::scalar::lance_format::LanceIndexStore;
use crate::vector::graph::OrderedFloat;
use crate::{progress::IndexBuildProgress, progress::noop_progress};
use arrow::array::AsArray;
use arrow::datatypes;
use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use bitpacking::{BitPacker, BitPacker4x};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream};
use deepsize::DeepSizeOf;
use fst::Streamer;
use futures::{Stream, StreamExt, TryStreamExt};
use lance_arrow::json::JSON_EXT_NAME;
use lance_arrow::{ARROW_EXT_NAME_KEY, iter_str_array};
use lance_core::cache::LanceCache;
use lance_core::error::LanceOptionExt;
use lance_core::utils::tokio::{IO_CORE_RESERVATION, get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::{Error, ROW_ID, ROW_ID_FIELD, Result};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::LazyLock;
use std::task::{Context, Poll};
use std::{fmt::Debug, sync::atomic::AtomicU64};
use tracing::instrument;

// the number of elements in each block
// each block contains 128 row ids and 128 frequencies
// WARNING: changing this value will break the compatibility with existing indexes
pub const BLOCK_SIZE: usize = BitPacker4x::BLOCK_LEN;

// The default number of workers to use for FTS builds.
// By default this is roughly `num_cpus / 2`, but it can be overridden
// with `LANCE_FTS_NUM_SHARDS`.
pub static LANCE_FTS_NUM_SHARDS: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("LANCE_FTS_NUM_SHARDS")
        .unwrap_or_else(|_| default_num_workers().to_string())
        .parse()
        .expect("failed to parse LANCE_FTS_NUM_SHARDS")
});
// The default per-worker memory limit in MiB for FTS builds.
pub static LANCE_FTS_PARTITION_SIZE: LazyLock<u64> = LazyLock::new(|| {
    std::env::var("LANCE_FTS_PARTITION_SIZE")
        .unwrap_or_else(|_| "2048".to_string())
        .parse()
        .expect("failed to parse LANCE_FTS_PARTITION_SIZE")
});
static LANCE_FTS_WRITE_QUEUE_SIZE: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("LANCE_FTS_WRITE_QUEUE_SIZE")
        .unwrap_or_else(|_| "1".to_string())
        .parse()
        .expect("failed to parse LANCE_FTS_WRITE_QUEUE_SIZE")
});
static LANCE_FTS_POSTING_BATCH_ROWS: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("LANCE_FTS_POSTING_BATCH_ROWS")
        .unwrap_or_else(|_| "256".to_string())
        .parse()
        .expect("failed to parse LANCE_FTS_POSTING_BATCH_ROWS")
});

fn default_num_workers() -> usize {
    let total_cpus = get_num_compute_intensive_cpus() + *IO_CORE_RESERVATION;
    std::cmp::max(1, total_cpus / 2)
}

fn resolve_num_workers(params: &InvertedIndexParams) -> usize {
    let max_workers = get_num_compute_intensive_cpus().max(1);
    params
        .num_workers
        .unwrap_or(*LANCE_FTS_NUM_SHARDS)
        .clamp(1, max_workers)
}

fn resolve_worker_memory_limit_bytes(params: &InvertedIndexParams, num_workers: usize) -> u64 {
    let default_worker_memory_limit_bytes = *LANCE_FTS_PARTITION_SIZE << 20;
    params
        .memory_limit_mb
        .map(|memory_limit_mb| (memory_limit_mb << 20) / num_workers as u64)
        .unwrap_or(default_worker_memory_limit_bytes)
}

fn merge_all_tail_partitions(tails: Vec<TailPartition>) -> Result<Option<InnerBuilder>> {
    if tails.is_empty() {
        return Ok(None);
    }
    merge_tail_partition_group(tails).map(Some)
}

fn merge_tail_partition_group(group: Vec<TailPartition>) -> Result<InnerBuilder> {
    let mut group = group.into_iter();
    let mut merged = group
        .next()
        .ok_or_else(|| {
            Error::invalid_input("cannot merge an empty tail partition group".to_owned())
        })?
        .builder;
    for tail in group {
        merged.merge_from(tail.builder)?;
    }
    Ok(merged)
}

#[derive(Debug)]
pub struct InvertedIndexBuilder {
    params: InvertedIndexParams,
    pub(crate) partitions: Vec<u64>,
    new_partitions: Vec<u64>,
    fragment_mask: Option<u64>,
    token_set_format: TokenSetFormat,
    src_store: Option<Arc<dyn IndexStore>>,
    progress: Arc<dyn IndexBuildProgress>,
}

impl InvertedIndexBuilder {
    pub fn new(params: InvertedIndexParams) -> Self {
        Self::new_with_fragment_mask(params, None)
    }

    pub fn new_with_fragment_mask(params: InvertedIndexParams, fragment_mask: Option<u64>) -> Self {
        Self::from_existing_index(
            params,
            None,
            Vec::new(),
            TokenSetFormat::default(),
            fragment_mask,
        )
    }

    /// Creates an InvertedIndexBuilder from existing index with fragment filtering.
    /// This method is used to create a builder from an existing index while applying
    /// fragment-based filtering for distributed indexing scenarios.
    /// fragment_mask Optional mask with fragment_id in high 32 bits for filtering.
    /// Constructed as `(fragment_id as u64) << 32`.
    /// When provided, ensures that generated IDs belong to the specified fragment.
    pub fn from_existing_index(
        params: InvertedIndexParams,
        store: Option<Arc<dyn IndexStore>>,
        partitions: Vec<u64>,
        token_set_format: TokenSetFormat,
        fragment_mask: Option<u64>,
    ) -> Self {
        Self {
            params,
            partitions,
            new_partitions: Vec::new(),
            src_store: store,
            token_set_format,
            fragment_mask,
            progress: noop_progress(),
        }
    }

    pub fn with_progress(mut self, progress: Arc<dyn IndexBuildProgress>) -> Self {
        self.progress = progress;
        self
    }

    pub async fn update(
        &mut self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        let schema = new_data.schema();
        let doc_col = schema.field(0).name();

        // infer lance_tokenizer based on document type
        if self.params.lance_tokenizer.is_none() {
            let schema = new_data.schema();
            let field = schema.column_with_name(doc_col).expect_ok()?.1;
            let doc_type = DocType::try_from(field)?;
            self.params.lance_tokenizer = Some(doc_type.as_ref().to_string());
        }

        let new_data = document_input(new_data, doc_col)?;

        self.progress
            .stage_start("tokenize_docs", None, "rows")
            .await?;
        self.update_index(new_data, dest_store).await?;
        self.progress.stage_complete("tokenize_docs").await?;
        self.write(dest_store).await?;
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn update_index(
        &mut self,
        stream: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        let num_workers = resolve_num_workers(&self.params);
        let tokenizer = self.params.build()?;
        let with_position = self.params.with_position;
        let worker_memory_limit_bytes =
            resolve_worker_memory_limit_bytes(&self.params, num_workers);
        let next_id = self.partitions.iter().map(|id| id + 1).max().unwrap_or(0);
        let id_alloc = Arc::new(AtomicU64::new(next_id));
        let tokenized_count = Arc::new(AtomicU64::new(0));
        let (sender, receiver) = async_channel::bounded(num_workers);
        let (builder_tx, builder_rx) = async_channel::bounded::<InnerBuilder>(1);
        let mut index_tasks = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let tokenizer = tokenizer.clone();
            let receiver: async_channel::Receiver<RecordBatch> = receiver.clone();
            let builder_tx = builder_tx.clone();
            let id_alloc = id_alloc.clone();
            let progress = self.progress.clone();
            let fragment_mask = self.fragment_mask;
            let token_set_format = self.token_set_format;
            let tokenized_count = tokenized_count.clone();
            index_tasks.push(tokio::task::spawn(async move {
                let mut worker = IndexWorker::new(
                    tokenizer,
                    builder_tx,
                    with_position,
                    id_alloc,
                    fragment_mask,
                    token_set_format,
                    worker_memory_limit_bytes,
                )
                .await?;
                while let Ok(batch) = receiver.recv().await {
                    let num_rows = batch.num_rows();
                    worker.process_batch(batch).await?;
                    let tokenized_count = tokenized_count
                        .fetch_add(num_rows as u64, std::sync::atomic::Ordering::Relaxed)
                        + num_rows as u64;
                    progress
                        .stage_progress("tokenize_docs", tokenized_count)
                        .await?;
                }
                worker.finish().await
            }));
        }
        let writer = async {
            while let Ok(mut builder) = builder_rx.recv().await {
                builder.write(dest_store).await?;
            }
            Result::Ok(())
        };

        let index_build = async {
            // Keep the channel lifetime tied to the worker tasks so senders observe
            // worker exits instead of blocking on an orphaned receiver handle.
            drop(receiver);

            let mut stream = Box::pin(stream);
            log::info!("indexing FTS with {} workers", num_workers);

            let mut last_num_rows = 0;
            let mut total_num_rows = 0;
            let start = std::time::Instant::now();
            while let Some(batch) = stream.try_next().await? {
                let num_rows = batch.num_rows();

                if sender.send(batch).await.is_err() {
                    // this only happens if all workers have exited,
                    // so we don't return the send error here,
                    // avoiding hiding the real error from workers.
                    break;
                }

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
            drop(sender);
            log::info!("dispatching elapsed: {:?}", start.elapsed());

            // wait for the workers to finish
            let start = std::time::Instant::now();
            let mut tail_partitions = Vec::new();
            for index_task in index_tasks {
                let output = index_task.await??;
                self.new_partitions.extend(output.partitions);
                if let Some(tail_partition) = output.tail_partition {
                    tail_partitions.push(tail_partition);
                }
            }
            let merged_tail_partitions =
                spawn_cpu(move || merge_all_tail_partitions(tail_partitions)).await?;
            if let Some(builder) = merged_tail_partitions {
                self.new_partitions.push(builder.id());
                builder_tx.send(builder).await.map_err(|err| {
                    Error::execution(format!(
                        "failed to send merged tail partition to writer: {err}"
                    ))
                })?;
            }
            drop(builder_tx);
            log::info!("wait workers indexing elapsed: {:?}", start.elapsed());
            Result::Ok(())
        };

        let (_, _) = tokio::try_join!(writer, index_build)?;
        Ok(())
    }

    pub async fn remap(
        &mut self,
        mapping: &HashMap<u64, Option<u64>>,
        src_store: Arc<dyn IndexStore>,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        for part in self.partitions.iter() {
            let part = InvertedPartition::load(
                src_store.clone(),
                *part,
                None,
                &LanceCache::no_cache(),
                self.token_set_format,
            )
            .await?;
            let mut builder = part.into_builder().await?;
            builder.remap(mapping).await?;
            builder.write(dest_store).await?;
        }
        if self.fragment_mask.is_none() {
            self.write_metadata(dest_store, &self.partitions).await?;
        } else {
            // in distributed mode, the part_temp_metadata is written by the worker
            for &partition_id in &self.partitions {
                self.write_part_metadata(dest_store, partition_id).await?;
            }
        }
        Ok(())
    }

    async fn write_metadata(&self, dest_store: &dyn IndexStore, partitions: &[u64]) -> Result<()> {
        let metadata = HashMap::from_iter(vec![
            ("partitions".to_owned(), serde_json::to_string(&partitions)?),
            ("params".to_owned(), serde_json::to_string(&self.params)?),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                self.token_set_format.to_string(),
            ),
        ]);
        let mut writer = dest_store
            .new_index_file(METADATA_FILE, Arc::new(Schema::empty()))
            .await?;
        writer.finish_with_metadata(metadata).await?;
        Ok(())
    }

    /// Write partition metadata file for a single partition
    ///
    /// In a distributed environment, each worker node can write partition metadata files for the partitions it processes,
    /// which are then merged into a final metadata file using the `merge_metadata_files` function.
    pub(crate) async fn write_part_metadata(
        &self,
        dest_store: &dyn IndexStore,
        partition: u64, // Modify parameter type
    ) -> Result<()> {
        let partitions = vec![partition];
        let metadata = HashMap::from_iter(vec![
            ("partitions".to_owned(), serde_json::to_string(&partitions)?),
            ("params".to_owned(), serde_json::to_string(&self.params)?),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                self.token_set_format.to_string(),
            ),
        ]);
        // Use partition ID to generate a unique temporary filename
        let file_name = part_metadata_file_path(partition);
        let mut writer = dest_store
            .new_index_file(&file_name, Arc::new(Schema::empty()))
            .await?;
        writer.finish_with_metadata(metadata).await?;
        Ok(())
    }

    async fn write_metadata_with_progress(
        &self,
        dest_store: &dyn IndexStore,
        partitions: &[u64],
    ) -> Result<()> {
        let total = if self.fragment_mask.is_none() {
            Some(1)
        } else {
            Some(partitions.len() as u64)
        };
        self.progress
            .stage_start("write_metadata", total, "files")
            .await?;
        if self.fragment_mask.is_none() {
            self.write_metadata(dest_store, partitions).await?;
            self.progress.stage_progress("write_metadata", 1).await?;
        } else {
            let mut completed = 0;
            for &partition_id in partitions {
                self.write_part_metadata(dest_store, partition_id).await?;
                completed += 1;
                self.progress
                    .stage_progress("write_metadata", completed)
                    .await?;
            }
        }
        self.progress.stage_complete("write_metadata").await?;
        Ok(())
    }

    async fn write(&self, dest_store: &dyn IndexStore) -> Result<()> {
        let mut partitions = Vec::with_capacity(self.partitions.len() + self.new_partitions.len());
        partitions.extend_from_slice(&self.partitions);
        partitions.extend_from_slice(&self.new_partitions);
        partitions.sort_unstable();

        self.progress
            .stage_start(
                "copy_partitions",
                Some(partitions.len() as u64),
                "partitions",
            )
            .await?;
        let mut copied = 0;
        for part in self.partitions.iter() {
            self.src_store
                .as_ref()
                .expect("existing partitions require a source store")
                .copy_index_file(&token_file_path(*part), dest_store)
                .await?;
            self.src_store
                .as_ref()
                .expect("existing partitions require a source store")
                .copy_index_file(&posting_file_path(*part), dest_store)
                .await?;
            self.src_store
                .as_ref()
                .expect("existing partitions require a source store")
                .copy_index_file(&doc_file_path(*part), dest_store)
                .await?;
            copied += 1;
            self.progress
                .stage_progress("copy_partitions", copied)
                .await?;
        }
        for _part in self.new_partitions.iter() {
            copied += 1;
            self.progress
                .stage_progress("copy_partitions", copied)
                .await?;
        }
        self.progress.stage_complete("copy_partitions").await?;

        self.write_metadata_with_progress(dest_store, &partitions)
            .await?;
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
    with_position: bool,
    token_set_format: TokenSetFormat,
    pub(crate) tokens: TokenSet,
    pub(crate) posting_lists: Vec<PostingListBuilder>,
    pub(crate) docs: DocSet,
}

impl InnerBuilder {
    pub fn new(id: u64, with_position: bool, token_set_format: TokenSetFormat) -> Self {
        Self {
            id,
            with_position,
            token_set_format,
            tokens: TokenSet::default(),
            posting_lists: Vec::new(),
            docs: DocSet::default(),
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    /// Set the token set for this builder.
    pub fn set_tokens(&mut self, tokens: TokenSet) {
        self.tokens = tokens;
    }

    /// Set the document set for this builder.
    pub fn set_docs(&mut self, docs: DocSet) {
        self.docs = docs;
    }

    /// Set the posting lists for this builder.
    pub fn set_posting_lists(&mut self, posting_lists: Vec<PostingListBuilder>) {
        self.posting_lists = posting_lists;
    }

    pub async fn remap(&mut self, mapping: &HashMap<u64, Option<u64>>) -> Result<()> {
        // for the docs, we need to remove the rows that are removed from the doc set,
        // and update the row ids of the rows that are updated
        let removed = self.docs.remap(mapping);

        // for the posting lists, we need to remap the doc ids:
        // - if the a row is removed, we need to shift the doc ids of the following rows
        // - if a row is updated (assigned a new row id), we don't need to do anything with the posting lists
        let mut token_id = 0;
        let mut removed_token_ids = Vec::new();
        self.posting_lists.retain_mut(|posting_list| {
            posting_list.remap(&removed);
            let keep = !posting_list.is_empty();
            if !keep {
                removed_token_ids.push(token_id as u32);
            }
            token_id += 1;
            keep
        });

        // for the tokens, remap the token ids if any posting list is empty
        self.tokens.remap(&removed_token_ids);

        Ok(())
    }

    pub fn merge_from(&mut self, other: Self) -> Result<()> {
        let Self {
            id: _,
            with_position,
            token_set_format,
            tokens,
            posting_lists,
            docs,
        } = other;

        if self.with_position != with_position {
            return Err(Error::index(format!(
                "cannot merge partitions with mismatched positions settings: {} vs {}",
                self.with_position, with_position
            )));
        }
        if self.token_set_format != token_set_format {
            return Err(Error::index(format!(
                "cannot merge partitions with mismatched token set formats: {:?} vs {:?}",
                self.token_set_format, token_set_format
            )));
        }

        let mut token_id_map = vec![u32::MAX; posting_lists.len()];
        match tokens.tokens {
            TokenMap::HashMap(map) => {
                for (token, token_id) in map {
                    let new_token_id = self.tokens.get_or_add(token.as_str());
                    token_id_map[token_id as usize] = new_token_id;
                }
            }
            TokenMap::Fst(map) => {
                let mut stream = map.stream();
                while let Some((token, token_id)) = stream.next() {
                    let new_token_id = self
                        .tokens
                        .get_or_add(String::from_utf8_lossy(token).as_ref());
                    token_id_map[token_id as usize] = new_token_id;
                }
            }
        }

        let doc_id_offset = self.docs.len() as u32;
        for (row_id, num_tokens) in docs.iter() {
            self.docs.append(*row_id, *num_tokens);
        }
        self.posting_lists
            .resize_with(self.tokens.len(), || PostingListBuilder::new(with_position));

        for (token_id, posting_list) in posting_lists.into_iter().enumerate() {
            if posting_list.is_empty() {
                continue;
            }
            let new_token_id = token_id_map[token_id];
            debug_assert_ne!(new_token_id, u32::MAX);
            let merged_posting = &mut self.posting_lists[new_token_id as usize];
            posting_list.for_each_entry(|doc_id, freq, positions| {
                let positions = match positions {
                    Some(positions) => PositionRecorder::Position(positions.into()),
                    None => PositionRecorder::Count(freq),
                };
                merged_posting.add(doc_id_offset + doc_id, positions);
                Ok::<(), Error>(())
            })?;
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
                inverted_list_schema(self.with_position),
            )
            .await?;
        let posting_lists = std::mem::take(&mut self.posting_lists);

        log::info!(
            "writing {} posting lists of partition {}, with position {}",
            posting_lists.len(),
            id,
            self.with_position
        );
        let with_position = self.with_position;
        let schema = inverted_list_schema(self.with_position);
        let docs_for_batches = docs.clone();
        let schema_for_batches = schema.clone();
        let batch_rows = *LANCE_FTS_POSTING_BATCH_ROWS;
        let (tx, rx) = async_channel::bounded(*LANCE_FTS_WRITE_QUEUE_SIZE);
        let producer = spawn_cpu(move || {
            let mut batch_builder =
                PostingListBatchBuilder::new(schema_for_batches.clone(), with_position, batch_rows);
            for posting_list in posting_lists {
                posting_list.append_to_batch_with_docs(&docs_for_batches, &mut batch_builder)?;
                if batch_builder.len() < batch_rows {
                    continue;
                }

                let batch = batch_builder.finish()?;
                if let Err(err) = tx.send_blocking(batch) {
                    return Err(Error::execution(format!(
                        "failed to send posting list batch to writer: {err}"
                    )));
                }
            }

            if !batch_builder.is_empty() {
                let batch = batch_builder.finish()?;
                if let Err(err) = tx.send_blocking(batch) {
                    return Err(Error::execution(format!(
                        "failed to send posting list batch to writer: {err}"
                    )));
                }
            }

            Result::Ok(())
        });

        while let Ok(batch) = rx.recv().await {
            if let Err(err) = writer.write_record_batch(batch).await {
                drop(rx);
                // Wait for producer to stop; preserve the write error as the primary failure.
                let _ = producer.await;
                return Err(err);
            }
        }
        drop(rx);
        producer.await?;

        writer.finish().await?;
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn write_tokens(&mut self, store: &dyn IndexStore) -> Result<()> {
        log::info!("writing tokens of partition {}", self.id);
        let tokens = std::mem::take(&mut self.tokens);
        let batch = tokens.to_batch(self.token_set_format)?;
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
    tokenizer: Box<dyn LanceTokenizer>,
    builder_tx: async_channel::Sender<InnerBuilder>,
    id_alloc: Arc<AtomicU64>,
    builder: InnerBuilder,
    partitions: Vec<u64>,
    schema: SchemaRef,
    memory_size: u64,
    worker_memory_limit_bytes: u64,
    total_doc_length: usize,
    fragment_mask: Option<u64>,
    token_set_format: TokenSetFormat,
    token_occurrences: HashMap<u32, PositionRecorder>,
    token_ids: Vec<u32>,
    last_token_count: usize,
    last_unique_token_count: usize,
}

struct TailPartition {
    builder: InnerBuilder,
}

struct WorkerOutput {
    partitions: Vec<u64>,
    tail_partition: Option<TailPartition>,
}

impl IndexWorker {
    fn posting_lists_overhead_size(&self) -> u64 {
        (self.builder.posting_lists.capacity() * std::mem::size_of::<PostingListBuilder>()) as u64
    }

    fn adjust_tracked_value(tracked: &mut u64, old: u64, new: u64) {
        if new >= old {
            *tracked += new - old;
        } else {
            *tracked -= old - new;
        }
    }

    fn adjust_tracked_memory_size(&mut self, old_memory_size: u64, new_memory_size: u64) {
        Self::adjust_tracked_value(&mut self.memory_size, old_memory_size, new_memory_size);
    }

    fn apply_delta(total: &mut u64, delta: i64) {
        if delta >= 0 {
            *total += delta as u64;
        } else {
            *total -= (-delta) as u64;
        }
    }

    async fn new(
        tokenizer: Box<dyn LanceTokenizer>,
        builder_tx: async_channel::Sender<InnerBuilder>,
        with_position: bool,
        id_alloc: Arc<AtomicU64>,
        fragment_mask: Option<u64>,
        token_set_format: TokenSetFormat,
        worker_memory_limit_bytes: u64,
    ) -> Result<Self> {
        let schema = inverted_list_schema(with_position);

        Ok(Self {
            tokenizer,
            builder_tx,
            builder: InnerBuilder::new(
                id_alloc.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    | fragment_mask.unwrap_or(0),
                with_position,
                token_set_format,
            ),
            partitions: Vec::new(),
            id_alloc,
            schema,
            memory_size: 0,
            worker_memory_limit_bytes,
            total_doc_length: 0,
            fragment_mask,
            token_set_format,
            token_occurrences: HashMap::new(),
            token_ids: Vec::new(),
            last_token_count: 0,
            last_unique_token_count: 0,
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
            let builder_was_empty = self.builder.docs.is_empty();
            let old_token_memory_size = self.builder.tokens.memory_size() as u64;
            let mut token_num: u32 = 0;
            if with_position {
                if self.token_occurrences.capacity() < self.last_unique_token_count {
                    self.token_occurrences
                        .reserve(self.last_unique_token_count - self.token_occurrences.capacity());
                }
                self.token_occurrences.clear();

                let mut token_stream = self.tokenizer.token_stream_for_doc(doc);
                while token_stream.advance() {
                    let token = token_stream.token_mut();
                    let token_text = std::mem::take(&mut token.text);
                    let token_id = self.builder.tokens.add(token_text);
                    self.token_occurrences
                        .entry(token_id)
                        .or_insert_with(|| PositionRecorder::new(true))
                        .push(token.position as u32);
                    token_num += 1;
                }
            } else {
                if self.token_ids.capacity() < self.last_token_count {
                    self.token_ids
                        .reserve(self.last_token_count - self.token_ids.capacity());
                }
                self.token_ids.clear();

                let mut token_stream = self.tokenizer.token_stream_for_doc(doc);
                while token_stream.advance() {
                    let token = token_stream.token_mut();
                    let token_text = std::mem::take(&mut token.text);
                    let token_id = self.builder.tokens.add(token_text);
                    self.token_ids.push(token_id);
                    token_num += 1;
                }
            }
            self.adjust_tracked_memory_size(
                old_token_memory_size,
                self.builder.tokens.memory_size() as u64,
            );

            let old_posting_lists_overhead_size = self.posting_lists_overhead_size();
            self.builder
                .posting_lists
                .resize_with(self.builder.tokens.len(), || {
                    PostingListBuilder::new(with_position)
                });
            let new_posting_lists_overhead_size = self.posting_lists_overhead_size();
            Self::adjust_tracked_value(
                &mut self.memory_size,
                old_posting_lists_overhead_size,
                new_posting_lists_overhead_size,
            );

            let old_doc_memory_size = self.builder.docs.memory_size() as u64;
            let doc_id = self.builder.docs.append(row_id, token_num);
            self.adjust_tracked_memory_size(
                old_doc_memory_size,
                self.builder.docs.memory_size() as u64,
            );
            self.total_doc_length += doc.len();

            if with_position {
                let mut token_occurrences = std::mem::take(&mut self.token_occurrences);
                let unique_tokens = token_occurrences.len();
                let mut posting_memory_delta = 0i64;
                for (token_id, term_positions) in token_occurrences.drain() {
                    let (old_posting_memory_size, new_posting_memory_size) = {
                        let posting_list = &mut self.builder.posting_lists[token_id as usize];
                        let old_posting_memory_size = posting_list.size();
                        posting_list.add(doc_id, term_positions);
                        let new_posting_memory_size = posting_list.size();
                        (old_posting_memory_size, new_posting_memory_size)
                    };
                    posting_memory_delta +=
                        new_posting_memory_size as i64 - old_posting_memory_size as i64;
                }
                Self::apply_delta(&mut self.memory_size, posting_memory_delta);
                self.token_occurrences = token_occurrences;
                self.last_unique_token_count = unique_tokens;
            } else if token_num > 0 {
                self.token_ids.sort_unstable();
                let mut posting_memory_delta = 0i64;
                let mut iter = self.token_ids.iter();
                let mut current = *iter.next().unwrap();
                let mut count = 1u32;
                for &token_id in iter {
                    if token_id == current {
                        count += 1;
                        continue;
                    }

                    let (old_posting_memory_size, new_posting_memory_size) = {
                        let posting_list = &mut self.builder.posting_lists[current as usize];
                        let old_posting_memory_size = posting_list.size();
                        posting_list.add(doc_id, PositionRecorder::Count(count));
                        let new_posting_memory_size = posting_list.size();
                        (old_posting_memory_size, new_posting_memory_size)
                    };
                    posting_memory_delta +=
                        new_posting_memory_size as i64 - old_posting_memory_size as i64;

                    current = token_id;
                    count = 1;
                }
                let (old_posting_memory_size, new_posting_memory_size) = {
                    let posting_list = &mut self.builder.posting_lists[current as usize];
                    let old_posting_memory_size = posting_list.size();
                    posting_list.add(doc_id, PositionRecorder::Count(count));
                    let new_posting_memory_size = posting_list.size();
                    (old_posting_memory_size, new_posting_memory_size)
                };
                posting_memory_delta +=
                    new_posting_memory_size as i64 - old_posting_memory_size as i64;
                Self::apply_delta(&mut self.memory_size, posting_memory_delta);
            }
            self.last_token_count = token_num as usize;

            if self.builder.docs.len() == 1 && self.memory_size > self.worker_memory_limit_bytes {
                return Err(Error::invalid_input(format!(
                    "single document row_id={} exceeds worker memory limit: {} > {} bytes",
                    row_id, self.memory_size, self.worker_memory_limit_bytes
                )));
            }

            if self.builder.docs.len() as u32 == u32::MAX
                || (!builder_was_empty && self.memory_size >= self.worker_memory_limit_bytes)
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
            "flushing posting lists, memory size: {} MiB",
            self.memory_size / (1024 * 1024)
        );
        self.memory_size = 0;
        let with_position = self.has_position();
        let builder = std::mem::replace(
            &mut self.builder,
            InnerBuilder::new(
                self.id_alloc
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    | self.fragment_mask.unwrap_or(0),
                with_position,
                self.token_set_format,
            ),
        );
        let written_partition_id = builder.id();
        self.builder_tx.send(builder).await.map_err(|err| {
            Error::execution(format!(
                "failed to send finalized partition {} to writer: {err}",
                written_partition_id
            ))
        })?;
        self.partitions.push(written_partition_id);
        Ok(())
    }

    async fn finish(self) -> Result<WorkerOutput> {
        let tail_partition = if self.builder.tokens.is_empty() {
            None
        } else {
            Some(TailPartition {
                builder: self.builder,
            })
        };
        Ok(WorkerOutput {
            partitions: self.partitions,
            tail_partition,
        })
    }
}

#[derive(Debug, Clone)]
pub enum PositionRecorder {
    Position(SmallVec<[u32; 4]>),
    Count(u32),
}

impl PositionRecorder {
    fn new(with_position: bool) -> Self {
        if with_position {
            Self::Position(SmallVec::new())
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
            Self::Position(positions) => positions.into_vec(),
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

/// Flatten the string list stream into a string stream
pub struct FlattenStream {
    /// Inner record batch stream with 2 columns:
    /// 1. doc_col: List(Utf8) or List(LargeUtf8)
    /// 2. row_id_col: UInt64
    inner: SendableRecordBatchStream,
    field_type: DataType,
    data_type: DataType,
}

impl FlattenStream {
    pub fn new(input: SendableRecordBatchStream) -> Self {
        let schema = input.schema();
        let field = schema.field(0);
        let data_type = match field.data_type() {
            DataType::List(f) if matches!(f.data_type(), DataType::Utf8) => DataType::Utf8,
            DataType::List(f) if matches!(f.data_type(), DataType::LargeUtf8) => {
                DataType::LargeUtf8
            }
            DataType::LargeList(f) if matches!(f.data_type(), DataType::Utf8) => DataType::Utf8,
            DataType::LargeList(f) if matches!(f.data_type(), DataType::LargeUtf8) => {
                DataType::LargeUtf8
            }
            _ => panic!(
                "expect data type List(Utf8) or List(LargeUtf8) but got {:?}",
                field.data_type()
            ),
        };
        Self {
            inner: input,
            field_type: field.data_type().clone(),
            data_type,
        }
    }
}

impl Stream for FlattenStream {
    type Item = datafusion_common::Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                let doc_col = batch.column(0);
                let batch = match self.field_type {
                    DataType::List(_) => flatten_string_list::<i32>(&batch, doc_col).map_err(|e| {
                        datafusion_common::error::DataFusionError::Execution(format!(
                            "flatten string list error: {}",
                            e
                        ))
                    }),
                    DataType::LargeList(_) => {
                        flatten_string_list::<i64>(&batch, doc_col).map_err(|e| {
                            datafusion_common::error::DataFusionError::Execution(format!(
                                "flatten string list error: {}",
                                e
                            ))
                        })
                    }
                    _ => unreachable!(
                        "expect data type List or LargeList but got {:?}",
                        self.field_type
                    ),
                };
                Poll::Ready(Some(batch))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl RecordBatchStream for FlattenStream {
    fn schema(&self) -> SchemaRef {
        let schema = Schema::new(vec![
            Field::new(
                self.inner.schema().field(0).name(),
                self.data_type.clone(),
                true,
            ),
            ROW_ID_FIELD.clone(),
        ]);

        Arc::new(schema)
    }
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
            return Err(Error::index(format!(
                "expect data type String or LargeString but got {}",
                docs.value_type()
            )));
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

pub(crate) fn part_metadata_file_path(partition_id: u64) -> String {
    format!("part_{}_{}", partition_id, METADATA_FILE)
}

pub async fn merge_index_files(
    object_store: &ObjectStore,
    index_dir: &Path,
    store: Arc<dyn IndexStore>,
) -> Result<()> {
    // List all partition metadata files in the index directory
    let part_metadata_files = list_metadata_files(object_store, index_dir).await?;

    // Call merge_metadata_files function for inverted index
    merge_metadata_files(store, &part_metadata_files).await
}

/// List and filter metadata files from the index directory
/// Returns partition metadata files
async fn list_metadata_files(object_store: &ObjectStore, index_dir: &Path) -> Result<Vec<String>> {
    // List all partition metadata files in the index directory
    let mut part_metadata_files = Vec::new();
    let mut list_stream = object_store.list(Some(index_dir.clone()));

    while let Some(item) = list_stream.next().await {
        match item {
            Ok(meta) => {
                let file_name = meta.location.filename().unwrap_or_default();
                // Filter files matching the pattern part_*_metadata.lance
                if file_name.starts_with("part_") && file_name.ends_with("_metadata.lance") {
                    part_metadata_files.push(file_name.to_string());
                }
            }
            Err(_) => continue,
        }
    }

    if part_metadata_files.is_empty() {
        return Err(Error::invalid_input_source(
            format!(
                "No partition metadata files found in index directory: {}",
                index_dir
            )
            .into(),
        ));
    }

    Ok(part_metadata_files)
}

/// Merge partition metadata files with partition ID remapping to sequential IDs starting from 0
async fn merge_metadata_files(
    store: Arc<dyn IndexStore>,
    part_metadata_files: &[String],
) -> Result<()> {
    // Collect all partition IDs and params
    let mut all_partitions = Vec::new();
    let mut params = None;
    let mut token_set_format = None;

    for file_name in part_metadata_files {
        let reader = store.open_index_file(file_name).await?;
        let metadata = &reader.schema().metadata;

        let partitions_str = metadata.get("partitions").ok_or(Error::index(format!(
            "partitions not found in {}",
            file_name
        )))?;

        let partition_ids: Vec<u64> = serde_json::from_str(partitions_str)
            .map_err(|e| Error::index(format!("Failed to parse partitions: {}", e)))?;

        all_partitions.extend(partition_ids);

        if params.is_none() {
            let params_str = metadata
                .get("params")
                .ok_or(Error::index(format!("params not found in {}", file_name)))?;
            params = Some(
                serde_json::from_str::<InvertedIndexParams>(params_str)
                    .map_err(|e| Error::index(format!("Failed to parse params: {}", e)))?,
            );
        }

        if token_set_format.is_none()
            && let Some(name) = metadata.get(TOKEN_SET_FORMAT_KEY)
        {
            token_set_format = Some(TokenSetFormat::from_str(name)?);
        }
    }

    // Create ID mapping: sorted original IDs -> 0,1,2...
    let mut sorted_ids = all_partitions.clone();
    sorted_ids.sort();
    sorted_ids.dedup();

    let id_mapping: HashMap<u64, u64> = sorted_ids
        .iter()
        .enumerate()
        .map(|(new_id, &old_id)| (old_id, new_id as u64))
        .collect();

    // Safe rename partition files using temporary files to avoid overwrite
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Phase 1: Move files to temporary locations
    let mut temp_files: Vec<(String, String, String)> = Vec::new(); // (temp_path, old_path, final_path)

    for (&old_id, &new_id) in &id_mapping {
        if old_id != new_id {
            for suffix in [TOKENS_FILE, INVERT_LIST_FILE, DOCS_FILE] {
                let old_path = format!("part_{}_{}", old_id, suffix);
                let new_path = format!("part_{}_{}", new_id, suffix);
                let temp_path = format!("temp_{}_{}", timestamp, old_path);

                // Move to temporary location first to avoid overwrite
                if let Err(e) = store.rename_index_file(&old_path, &temp_path).await {
                    // Rollback phase 1: restore files from temp locations
                    for (temp_name, old_name, _) in temp_files.iter().rev() {
                        let _ = store.rename_index_file(temp_name, old_name).await;
                    }
                    return Err(Error::index(format!(
                        "Failed to move {} to temp {}: {}",
                        old_path, temp_path, e
                    )));
                }
                temp_files.push((temp_path, old_path, new_path));
            }
        }
    }

    // Phase 2: Move from temporary to final locations
    let mut completed_renames: Vec<(String, String)> = Vec::new(); // (final_path, temp_path)

    for (temp_path, _old_path, final_path) in &temp_files {
        if let Err(e) = store.rename_index_file(temp_path, final_path).await {
            // Rollback phase 2: restore completed renames and remaining temps
            for (final_name, temp_name) in completed_renames.iter().rev() {
                let _ = store.rename_index_file(final_name, temp_name).await;
            }
            // Restore remaining temp files to original locations
            for (temp_name, orig_name, _) in temp_files.iter() {
                if !completed_renames.iter().any(|(_, t)| t == temp_name) {
                    let _ = store.rename_index_file(temp_name, orig_name).await;
                }
            }
            return Err(Error::index(format!(
                "Failed to rename {} to {}: {}",
                temp_path, final_path, e
            )));
        }
        completed_renames.push((final_path.clone(), temp_path.clone()));
    }

    // Write merged metadata with remapped IDs
    let remapped_partitions: Vec<u64> = (0..id_mapping.len() as u64).collect();
    let params = params.unwrap_or_default();
    let token_set_format = token_set_format.unwrap_or(TokenSetFormat::Arrow);
    let builder = InvertedIndexBuilder::from_existing_index(
        params,
        None,
        remapped_partitions.clone(),
        token_set_format,
        None,
    );
    builder
        .write_metadata(&*store, &remapped_partitions)
        .await?;

    // Cleanup partition metadata files
    for file_name in part_metadata_files {
        if file_name.starts_with("part_") && file_name.ends_with("_metadata.lance") {
            let _ = store.delete_index_file(file_name).await;
        }
    }

    Ok(())
}

/// Convert input stream into a stream of documents.
///
/// The input stream must be one of:
/// 1. Document in Utf8 or LargeUtf8 format.
/// 2. Document in List(Utf8) or List(LargeUtf8) format.
/// 3. Json document in LargeBinary format.
pub fn document_input(
    input: SendableRecordBatchStream,
    column: &str,
) -> Result<SendableRecordBatchStream> {
    let schema = input.schema();
    let field = schema.column_with_name(column).expect_ok()?.1;
    match field.data_type() {
        DataType::Utf8 | DataType::LargeUtf8 => Ok(input),
        DataType::List(field) | DataType::LargeList(field)
            if matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8) =>
        {
            Ok(Box::pin(FlattenStream::new(input)))
        }
        DataType::LargeBinary => match field.metadata().get(ARROW_EXT_NAME_KEY) {
            Some(name) if name.as_str() == JSON_EXT_NAME => {
                Ok(Box::pin(JsonTextStream::new(input, column.to_string())))
            }
            _ => Err(Error::invalid_input_source(
                format!("column {} is not json", column).into(),
            )),
        },
        _ => Err(Error::invalid_input_source(
            format!(
                "column {} has type {}, is not utf8, large utf8 type/list, or large binary",
                column,
                field.data_type()
            )
            .into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::NoOpMetricsCollector;
    use crate::progress::IndexBuildProgress;
    use crate::scalar::{IndexReader, IndexWriter};
    use arrow_array::{RecordBatch, StringArray, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use async_trait::async_trait;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream;
    use lance_core::ROW_ID;
    use lance_core::cache::LanceCache;
    use lance_core::utils::tempfile::TempDir;
    use std::any::Any;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::sync::Mutex;

    fn make_doc_batch(doc: &str, row_id: u64) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc", DataType::Utf8, true),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        let docs = Arc::new(StringArray::from(vec![Some(doc)]));
        let row_ids = Arc::new(UInt64Array::from(vec![row_id]));
        RecordBatch::try_new(schema, vec![docs, row_ids]).unwrap()
    }

    #[derive(Debug, Default)]
    struct CountingStore {
        write_count: Arc<AtomicUsize>,
    }

    impl CountingStore {
        fn new() -> Self {
            Self {
                write_count: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn write_count(&self) -> usize {
            self.write_count.load(Ordering::SeqCst)
        }
    }

    impl DeepSizeOf for CountingStore {
        fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
            0
        }
    }

    #[derive(Debug)]
    struct CountingWriter {
        write_count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl IndexWriter for CountingWriter {
        async fn write_record_batch(&mut self, _batch: RecordBatch) -> Result<u64> {
            Ok(self.write_count.fetch_add(1, Ordering::SeqCst) as u64)
        }

        async fn finish(&mut self) -> Result<()> {
            Ok(())
        }

        async fn finish_with_metadata(&mut self, _metadata: HashMap<String, String>) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl IndexStore for CountingStore {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn io_parallelism(&self) -> usize {
            1
        }

        async fn new_index_file(
            &self,
            _name: &str,
            _schema: Arc<Schema>,
        ) -> Result<Box<dyn IndexWriter>> {
            Ok(Box::new(CountingWriter {
                write_count: self.write_count.clone(),
            }))
        }

        async fn open_index_file(&self, _name: &str) -> Result<Arc<dyn IndexReader>> {
            Err(Error::not_supported(
                "CountingStore does not support reading",
            ))
        }

        async fn copy_index_file(&self, _name: &str, _dest_store: &dyn IndexStore) -> Result<()> {
            Err(Error::not_supported(
                "CountingStore does not support copying",
            ))
        }

        async fn rename_index_file(&self, _name: &str, _new_name: &str) -> Result<()> {
            Err(Error::not_supported(
                "CountingStore does not support renaming",
            ))
        }

        async fn delete_index_file(&self, _name: &str) -> Result<()> {
            Err(Error::not_supported(
                "CountingStore does not support deleting",
            ))
        }
    }

    #[tokio::test]
    async fn test_write_posting_lists_batches_multiple_rows() -> Result<()> {
        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
        for doc_id in 0..3u64 {
            builder.docs.append(doc_id, 1);
        }

        for doc_id in 0..3u32 {
            let mut posting_list = PostingListBuilder::new(false);
            posting_list.add(doc_id, PositionRecorder::Count(1));
            builder.posting_lists.push(posting_list);
        }

        let store = CountingStore::new();
        let docs = Arc::new(std::mem::take(&mut builder.docs));
        builder.write_posting_lists(&store, docs).await?;

        assert_eq!(store.write_count(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_build_only_path_writes_partitions_as_is() -> Result<()> {
        let src_dir = TempDir::default();
        let dest_dir = TempDir::default();
        let src_store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            src_dir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ));
        let dest_store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            dest_dir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ));

        let params = InvertedIndexParams::default();
        let tokenizer = params.build()?;
        let token_set_format = TokenSetFormat::default();
        let id_alloc = Arc::new(AtomicU64::new(0));
        let (builder_tx, builder_rx) = async_channel::bounded(2);

        let mut worker1 = IndexWorker::new(
            tokenizer.clone(),
            builder_tx.clone(),
            params.with_position,
            id_alloc.clone(),
            None,
            token_set_format,
            u64::MAX,
        )
        .await?;
        worker1
            .process_batch(make_doc_batch("hello world", 0))
            .await?;
        let output1 = worker1.finish().await?;
        let mut partitions = output1.partitions;
        if let Some(mut tail_partition) = output1.tail_partition {
            partitions.push(tail_partition.builder.id());
            tail_partition.builder.write(src_store.as_ref()).await?;
        }
        while let Ok(mut builder) = builder_rx.try_recv() {
            builder.write(src_store.as_ref()).await?;
        }

        let mut worker2 = IndexWorker::new(
            tokenizer.clone(),
            builder_tx,
            params.with_position,
            id_alloc.clone(),
            None,
            token_set_format,
            u64::MAX,
        )
        .await?;
        worker2
            .process_batch(make_doc_batch("goodbye world", 1))
            .await?;
        let output2 = worker2.finish().await?;
        partitions.extend(output2.partitions);
        if let Some(mut tail_partition) = output2.tail_partition {
            partitions.push(tail_partition.builder.id());
            tail_partition.builder.write(src_store.as_ref()).await?;
        }
        partitions.sort_unstable();
        assert_eq!(partitions.len(), 2);
        assert_ne!(partitions[0], partitions[1]);

        // Drain finalized builders and write them to the source store, mirroring the
        // direct-to-destination writer task used by the production indexing path.
        // The bounded channel size keeps these writes backpressured in normal execution.
        while let Ok(mut builder) = builder_rx.try_recv() {
            builder.write(src_store.as_ref()).await?;
        }

        let builder = InvertedIndexBuilder::from_existing_index(
            InvertedIndexParams::default(),
            Some(src_store.clone()),
            partitions.clone(),
            token_set_format,
            None,
        );
        builder.write(dest_store.as_ref()).await?;

        let metadata_reader = dest_store.open_index_file(METADATA_FILE).await?;
        let metadata = &metadata_reader.schema().metadata;
        let partitions_str = metadata
            .get("partitions")
            .expect("partitions missing from metadata");
        let written_partitions: Vec<u64> = serde_json::from_str(partitions_str).unwrap();
        assert_eq!(written_partitions, partitions);

        for id in &partitions {
            dest_store.open_index_file(&token_file_path(*id)).await?;
            dest_store.open_index_file(&posting_file_path(*id)).await?;
            dest_store.open_index_file(&doc_file_path(*id)).await?;
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_inverted_index_without_positions_tracks_frequency() -> Result<()> {
        let index_dir = TempDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            index_dir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ));

        let schema = Arc::new(Schema::new(vec![
            Field::new("doc", DataType::Utf8, true),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        let docs = Arc::new(StringArray::from(vec![Some("hello hello world")]));
        let row_ids = Arc::new(UInt64Array::from(vec![0u64]));
        let batch = RecordBatch::try_new(schema.clone(), vec![docs, row_ids])?;
        let stream = RecordBatchStreamAdapter::new(schema, stream::iter(vec![Ok(batch)]));
        let stream = Box::pin(stream);

        let params = InvertedIndexParams::new(
            "whitespace".to_string(),
            tantivy::tokenizer::Language::English,
        )
        .with_position(false)
        .remove_stop_words(false)
        .stem(false)
        .max_token_length(None);

        let mut builder = InvertedIndexBuilder::new(params);
        builder.update(stream, store.as_ref()).await?;

        let index = InvertedIndex::load(store, None, &LanceCache::no_cache()).await?;
        assert_eq!(index.partitions.len(), 1);
        let partition = &index.partitions[0];
        let token_id = partition.tokens.get("hello").unwrap();
        let posting = partition
            .inverted_list
            .posting_list(token_id, false, &NoOpMetricsCollector)
            .await?;

        let mut iter = posting.iter();
        let (doc_id, freq, positions) = iter.next().unwrap();
        assert_eq!(doc_id, 0);
        assert_eq!(freq, 2);
        assert!(positions.is_none());
        assert!(iter.next().is_none());

        Ok(())
    }

    #[derive(Debug, Default)]
    struct RecordingProgress {
        events: Mutex<Vec<(String, String, u64)>>,
    }

    #[async_trait]
    impl IndexBuildProgress for RecordingProgress {
        async fn stage_start(&self, stage: &str, total: Option<u64>, _unit: &str) -> Result<()> {
            self.events.lock().await.push((
                "start".to_string(),
                stage.to_string(),
                total.unwrap_or(0),
            ));
            Ok(())
        }

        async fn stage_progress(&self, stage: &str, completed: u64) -> Result<()> {
            self.events
                .lock()
                .await
                .push(("progress".to_string(), stage.to_string(), completed));
            Ok(())
        }

        async fn stage_complete(&self, stage: &str) -> Result<()> {
            self.events
                .lock()
                .await
                .push(("complete".to_string(), stage.to_string(), 0));
            Ok(())
        }
    }

    #[derive(Debug, Default)]
    struct FailingProgress;

    #[async_trait]
    impl IndexBuildProgress for FailingProgress {
        async fn stage_start(&self, _stage: &str, _total: Option<u64>, _unit: &str) -> Result<()> {
            Ok(())
        }

        async fn stage_progress(&self, _stage: &str, _completed: u64) -> Result<()> {
            Err(Error::io("injected progress failure"))
        }

        async fn stage_complete(&self, _stage: &str) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_builder_reports_progress_stages() -> Result<()> {
        let index_dir = TempDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            index_dir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ));

        let batch1 = make_doc_batch("hello world", 0);
        let batch2 = make_doc_batch("goodbye world", 1);
        let total_rows = 2u64;
        let stream = RecordBatchStreamAdapter::new(
            batch1.schema(),
            stream::iter(vec![Ok(batch1), Ok(batch2)]),
        );
        let stream = Box::pin(stream);

        let progress = Arc::new(RecordingProgress::default());
        let mut builder = InvertedIndexBuilder::new(InvertedIndexParams::default())
            .with_progress(progress.clone());
        builder.update(stream, store.as_ref()).await?;

        let events = progress.events.lock().await.clone();
        let tags = events
            .iter()
            .map(|(kind, stage, _)| format!("{kind}:{stage}"))
            .collect::<Vec<_>>();
        let tokenize_progress = events
            .iter()
            .filter_map(|(kind, stage, completed)| {
                if kind == "progress" && stage == "tokenize_docs" {
                    Some(*completed)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let tokenize_start = tags
            .iter()
            .position(|e| e == "start:tokenize_docs")
            .expect("missing tokenize_docs start");
        let tokenize_complete = tags
            .iter()
            .position(|e| e == "complete:tokenize_docs")
            .expect("missing tokenize_docs complete");
        let copy_start = tags
            .iter()
            .position(|e| e == "start:copy_partitions")
            .expect("missing copy_partitions start");
        let copy_complete = tags
            .iter()
            .position(|e| e == "complete:copy_partitions")
            .expect("missing copy_partitions complete");
        let metadata_start = tags
            .iter()
            .position(|e| e == "start:write_metadata")
            .expect("missing write_metadata start");
        let metadata_complete = tags
            .iter()
            .position(|e| e == "complete:write_metadata")
            .expect("missing write_metadata complete");

        assert!(tokenize_start < tokenize_complete);
        assert!(tokenize_complete < copy_start);
        assert!(copy_start < copy_complete);
        assert!(copy_complete < metadata_start);
        assert!(metadata_start < metadata_complete);

        assert!(
            tags.iter().any(|e| e == "progress:tokenize_docs"),
            "expected progress callback for tokenize_docs"
        );
        assert!(
            tokenize_progress.len() >= 2,
            "expected at least two progress callbacks for tokenize_docs, got {tokenize_progress:?}"
        );
        assert_eq!(
            tokenize_progress.iter().copied().max().unwrap_or_default(),
            total_rows,
            "expected tokenize_docs progress to reach all rows"
        );
        assert!(
            tags.iter().any(|e| e == "progress:copy_partitions"),
            "expected progress callback for copy_partitions"
        );
        assert!(
            tags.iter().any(|e| e == "progress:write_metadata"),
            "expected progress callback for write_metadata"
        );
        assert!(
            !tags.iter().any(|e| e == "start:merge_partitions"),
            "merge_partitions should not run in the build-only path"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_builder_default_path_skips_merge_stage() -> Result<()> {
        let index_dir = TempDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            index_dir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ));

        let batch = make_doc_batch("hello world", 0);
        let stream = RecordBatchStreamAdapter::new(batch.schema(), stream::iter(vec![Ok(batch)]));
        let stream = Box::pin(stream);

        let progress = Arc::new(RecordingProgress::default());
        let mut builder = InvertedIndexBuilder::new(InvertedIndexParams::default())
            .with_progress(progress.clone());
        builder.update(stream, store.as_ref()).await?;

        let tags = progress
            .events
            .lock()
            .await
            .iter()
            .map(|(kind, stage, _)| format!("{kind}:{stage}"))
            .collect::<Vec<_>>();

        assert!(
            tags.iter().any(|e| e == "start:copy_partitions"),
            "default path should copy finalized partitions"
        );
        assert!(
            !tags.iter().any(|e| e == "start:merge_partitions"),
            "default path should not run merge_partitions"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_worker_memory_limit_rejects_single_large_doc() {
        let index_dir = TempDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            index_dir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ));

        let batch = make_doc_batch("hello world", 42);
        let stream = RecordBatchStreamAdapter::new(batch.schema(), stream::iter(vec![Ok(batch)]));
        let stream = Box::pin(stream);

        let mut builder =
            InvertedIndexBuilder::new(InvertedIndexParams::default().memory_limit_mb(0));
        let err = builder
            .update(stream, store.as_ref())
            .await
            .expect_err("single doc should exceed zero worker memory limit");
        assert!(
            err.to_string().contains("row_id=42"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_resolve_worker_memory_limit_uses_default_when_unset() {
        let params = InvertedIndexParams::default();
        assert_eq!(
            resolve_worker_memory_limit_bytes(&params, 8),
            *LANCE_FTS_PARTITION_SIZE << 20
        );
    }

    #[test]
    fn test_resolve_num_workers_uses_default_when_unset() {
        let expected = default_num_workers().clamp(1, get_num_compute_intensive_cpus().max(1));
        assert_eq!(
            resolve_num_workers(&InvertedIndexParams::default()),
            expected
        );
    }

    #[test]
    fn test_resolve_num_workers_clamps_requested_value() {
        let max_workers = get_num_compute_intensive_cpus().max(1);
        assert_eq!(
            resolve_num_workers(&InvertedIndexParams::default().num_workers(0)),
            1
        );
        assert_eq!(
            resolve_num_workers(&InvertedIndexParams::default().num_workers(max_workers + 10)),
            max_workers
        );
    }

    #[test]
    fn test_resolve_worker_memory_limit_splits_total_memory_limit() {
        let params = InvertedIndexParams::default().memory_limit_mb(4096);
        assert_eq!(resolve_worker_memory_limit_bytes(&params, 16), 256 << 20);
    }

    #[test]
    fn test_merge_all_tail_partitions_combines_everything() -> Result<()> {
        let merged = merge_all_tail_partitions(vec![
            TailPartition {
                builder: InnerBuilder::new(0, false, TokenSetFormat::default()),
            },
            TailPartition {
                builder: InnerBuilder::new(1, false, TokenSetFormat::default()),
            },
            TailPartition {
                builder: InnerBuilder::new(2, false, TokenSetFormat::default()),
            },
        ])?;

        assert_eq!(merged.expect("merged builder should exist").id(), 0);
        Ok(())
    }

    #[test]
    fn test_merge_all_tail_partitions_returns_none_for_empty_input() -> Result<()> {
        assert!(merge_all_tail_partitions(Vec::new())?.is_none());
        Ok(())
    }

    #[test]
    fn test_merge_tail_partition_group_combines_tail_builders() -> Result<()> {
        let mut first = InnerBuilder::new(0, false, TokenSetFormat::default());
        let hello = first.tokens.add("hello".to_owned());
        first
            .posting_lists
            .resize_with(first.tokens.len(), || PostingListBuilder::new(false));
        let first_doc = first.docs.append(10, 1);
        first.posting_lists[hello as usize].add(first_doc, PositionRecorder::Count(1));

        let mut second = InnerBuilder::new(1, false, TokenSetFormat::default());
        let world = second.tokens.add("world".to_owned());
        second
            .posting_lists
            .resize_with(second.tokens.len(), || PostingListBuilder::new(false));
        let second_doc = second.docs.append(20, 2);
        second.posting_lists[world as usize].add(second_doc, PositionRecorder::Count(2));

        let merged = merge_tail_partition_group(vec![
            TailPartition { builder: first },
            TailPartition { builder: second },
        ])?;

        assert_eq!(merged.id(), 0);
        assert_eq!(merged.docs.len(), 2);
        assert_eq!(merged.tokens.len(), 2);
        assert_eq!(merged.posting_lists.len(), 2);
        assert_eq!(
            merged.posting_lists[merged.tokens.get("hello").unwrap() as usize].len(),
            1
        );
        assert_eq!(
            merged.posting_lists[merged.tokens.get("world").unwrap() as usize].len(),
            1
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_update_index_returns_worker_error_when_workers_exit_during_dispatch() {
        let num_batches = (*LANCE_FTS_NUM_SHARDS * 2 + 1) as u64;
        let index_dir = TempDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            index_dir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ));
        let schema = make_doc_batch("hello world", 0).schema();
        let stream = RecordBatchStreamAdapter::new(
            schema,
            stream::iter((0..num_batches).map(|row_id| Ok(make_doc_batch("hello world", row_id)))),
        );
        let stream = Box::pin(stream);

        let mut builder = InvertedIndexBuilder::new(InvertedIndexParams::default())
            .with_progress(Arc::new(FailingProgress));

        let result = tokio::time::timeout(
            Duration::from_secs(5),
            builder.update_index(stream, store.as_ref()),
        )
        .await
        .expect("update_index should not hang")
        .expect_err("worker failure should be returned");

        assert!(
            result.to_string().contains("injected progress failure"),
            "unexpected error: {result}"
        );
    }
}
